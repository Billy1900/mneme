use anyhow::Result;
use mneme_api::{MnemeMemory, MnemeSummary};
use mneme_consolidate::{AnthropicLLM, ConsolidationEngine, ConsolidationLLM, OpenAILLM};
use mneme_core::{MemoryQuery, MnemeConfig};
use mneme_embed::{backends::OpenAIEmbeddingModel, EmbeddingModel};
use mneme_store::{EnvelopeIndex, InMemoryContentStore, InMemoryEnvelopeIndex, MnemeStore};
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

use crate::datasets::{LoCoMoConversation, LongMemEvalItem};
use crate::judge::LLMJudge;
use crate::metrics::{self, QuestionResult};

/// Re-rank candidates by relevance to the question, return top `keep`.
async fn rerank(
    judge: &LLMJudge,
    question: &str,
    candidates: Vec<MnemeSummary>,
    keep: usize,
) -> Vec<MnemeSummary> {
    if candidates.len() <= keep {
        return candidates;
    }
    let texts: Vec<&str> = candidates.iter().map(|s| s.full_text.as_str()).collect();
    match judge.rerank_indices(question, &texts, keep).await {
        Ok(indices) => indices.into_iter().map(|i| candidates[i].clone()).collect(),
        Err(e) => {
            warn!("rerank failed: {e}, using top-{keep} by score");
            candidates.into_iter().take(keep).collect()
        }
    }
}

/// Which LLM backend to use for compaction synthesis.
#[derive(Clone)]
pub enum ConsolidationLlm {
    Anthropic(AnthropicLLM),
    OpenAI(OpenAILLM),
}

// Blanket so we can pass ConsolidationLlm anywhere ConsolidationLLM is needed via dispatch.
// We use dynamic dispatch inside Session to avoid two generic instantiations.
type BenchMemory = MnemeMemory<
    InMemoryEnvelopeIndex,
    InMemoryContentStore,
    OpenAIEmbeddingModel,
    Box<dyn ConsolidationLLM>,
>;

struct Session {
    memory: BenchMemory,
    envelopes: Arc<InMemoryEnvelopeIndex>,
}

fn new_session(embed: OpenAIEmbeddingModel, llm: ConsolidationLlm, config: MnemeConfig) -> Session {
    let envelopes = Arc::new(InMemoryEnvelopeIndex::new());
    let content = Arc::new(InMemoryContentStore::new());

    let engine_store = MnemeStore::new((*envelopes).clone(), (*content).clone());
    let api_store = MnemeStore::new((*envelopes).clone(), (*content).clone());

    let boxed: Box<dyn ConsolidationLLM> = match llm {
        ConsolidationLlm::Anthropic(a) => Box::new(a),
        ConsolidationLlm::OpenAI(o) => Box::new(o),
    };
    let engine = ConsolidationEngine::new(engine_store, embed.clone(), boxed, config.clone());
    let memory = MnemeMemory::new(api_store, engine, embed, config);

    Session { memory, envelopes }
}

/// Recall with fallback: if semantic memory is empty (compaction failed), fall
/// back to all-type search so we still get useful context.
async fn recall_with_fallback(
    session: &Session,
    embed: &OpenAIEmbeddingModel,
    query: &str,
    top_k: usize,
) -> Vec<MnemeSummary> {
    // Primary: semantic recall (post-compaction)
    match session.memory.recall(query, top_k).await {
        Ok(summaries) if !summaries.is_empty() => return summaries,
        Ok(_) => {}
        Err(e) => warn!("recall error: {e}"),
    }

    // Fallback: search working memory directly
    warn!("semantic memory empty — falling back to working memory search");
    let embedding = match embed.embed(query).await {
        Ok(e) => e,
        Err(e) => {
            warn!("embed fallback failed: {e}");
            return vec![];
        }
    };

    let q = MemoryQuery {
        embedding,
        top_k,
        active_only: true,
        memory_type: None, // all types
        min_confidence: Some(0.0),
        recency_weight: 0.1,
        ..Default::default()
    };

    match session.envelopes.search(&q).await {
        Ok(results) => results
            .into_iter()
            .map(|r| MnemeSummary {
                id: r.envelope.id,
                full_text: r.envelope.summary.clone(),
                summary: r.envelope.summary,
                confidence: r.envelope.confidence,
                tags: r.envelope.tags,
                similarity: r.similarity,
                retrieval_score: r.retrieval_score,
                version: 1,
                is_evolved: false,
            })
            .collect(),
        Err(e) => {
            warn!("fallback search error: {e}");
            vec![]
        }
    }
}

pub async fn run_locomo(
    conversations: &[LoCoMoConversation],
    embed: OpenAIEmbeddingModel,
    llm: ConsolidationLlm,
    judge: &LLMJudge,
    top_k: usize,
    use_judge: bool,
    limit: Option<usize>,
) -> Result<Vec<QuestionResult>> {
    let convs = match limit {
        Some(n) => &conversations[..n.min(conversations.len())],
        None => conversations,
    };

    let mut all_results = Vec::new();
    let config = MnemeConfig::default();

    for (ci, conv) in convs.iter().enumerate() {
        info!(
            "LoCoMo conversation {}/{}: {} sessions, {} questions",
            ci + 1,
            convs.len(),
            conv.sessions.len(),
            conv.questions.len()
        );

        let session = new_session(embed.clone(), llm.clone(), config.clone());

        // Ingest sessions
        for s in &conv.sessions {
            for turn in &s.turns {
                if turn.trim().is_empty() {
                    continue;
                }
                if let Err(e) = session.memory.remember(turn, &s.session_id).await {
                    warn!("remember failed: {e}");
                }
            }
            if let Err(e) = session.memory.end_session(&s.session_id).await {
                warn!("end_session failed for {}: {e}", s.session_id);
            }
        }

        // Evaluate questions
        for q in &conv.questions {
            let t0 = Instant::now();
            // Fetch 15 candidates, rerank to top_k
            let candidates = recall_with_fallback(&session, &embed, &q.question, 15).await;
            let summaries = rerank(judge, &q.question, candidates, top_k).await;
            let latency_ms = t0.elapsed().as_millis() as u64;

            let memory_context: String = summaries
                .iter()
                .enumerate()
                .map(|(i, s)| format!("[{}] {}", i + 1, s.full_text))
                .collect::<Vec<_>>()
                .join("\n");

            let tokens_used = metrics::estimate_tokens(&memory_context);

            let predicted = if memory_context.is_empty() {
                "Not found in memory".to_string()
            } else {
                match judge.generate_answer(&q.question, &memory_context).await {
                    Ok(a) => a,
                    Err(e) => {
                        warn!("generate_answer failed: {e}");
                        "Not found in memory".to_string()
                    }
                }
            };

            let em = metrics::exact_match(&predicted, &q.answer);
            let f1 = metrics::token_f1(&predicted, &q.answer);

            let judge_score = if use_judge {
                match judge.score(&q.question, &q.answer, &predicted).await {
                    Ok(s) => Some(s),
                    Err(e) => {
                        warn!("judge failed: {e}");
                        None
                    }
                }
            } else {
                None
            };

            all_results.push(QuestionResult {
                question_id: q.question_id.clone(),
                category: q.category.clone(),
                question: q.question.clone(),
                ground_truth: q.answer.clone(),
                predicted,
                exact_match: em,
                f1,
                judge_score,
                recall_latency_ms: latency_ms,
                tokens_used,
                memories_retrieved: summaries.len(),
            });
        }
    }

    Ok(all_results)
}

pub async fn run_longmemeval(
    items: &[LongMemEvalItem],
    embed: OpenAIEmbeddingModel,
    llm: ConsolidationLlm,
    judge: &LLMJudge,
    top_k: usize,
    use_judge: bool,
    limit: Option<usize>,
) -> Result<Vec<QuestionResult>> {
    let items = match limit {
        Some(n) => &items[..n.min(items.len())],
        None => items,
    };

    let mut all_results = Vec::new();
    let config = MnemeConfig::default();

    for (ii, item) in items.iter().enumerate() {
        info!(
            "LongMemEval item {}/{}: {} turns",
            ii + 1,
            items.len(),
            item.turns.len()
        );

        let session = new_session(embed.clone(), llm.clone(), config.clone());
        let session_id = item.history_id.clone();

        for turn in &item.turns {
            if turn.trim().is_empty() {
                continue;
            }
            if let Err(e) = session.memory.remember(turn, &session_id).await {
                warn!("remember failed: {e}");
            }
        }
        if let Err(e) = session.memory.end_session(&session_id).await {
            warn!("end_session failed: {e}");
        }

        let t0 = Instant::now();
        let candidates = recall_with_fallback(&session, &embed, &item.question, 15).await;
        let summaries = rerank(judge, &item.question, candidates, top_k).await;
        let latency_ms = t0.elapsed().as_millis() as u64;

        let memory_context: String = summaries
            .iter()
            .enumerate()
            .map(|(i, s)| format!("[{}] {}", i + 1, s.full_text))
            .collect::<Vec<_>>()
            .join("\n");

        let tokens_used = metrics::estimate_tokens(&memory_context);

        let predicted = if memory_context.is_empty() {
            "Not found in memory".to_string()
        } else {
            match judge.generate_answer(&item.question, &memory_context).await {
                Ok(a) => a,
                Err(e) => {
                    warn!("generate_answer failed: {e}");
                    "Not found in memory".to_string()
                }
            }
        };

        let em = metrics::exact_match(&predicted, &item.answer);
        let f1 = metrics::token_f1(&predicted, &item.answer);

        let judge_score = if use_judge {
            match judge.score(&item.question, &item.answer, &predicted).await {
                Ok(s) => Some(s),
                Err(e) => {
                    warn!("judge failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        all_results.push(QuestionResult {
            question_id: item.history_id.clone(),
            category: item.category.clone(),
            question: item.question.clone(),
            ground_truth: item.answer.clone(),
            predicted,
            exact_match: em,
            f1,
            judge_score,
            recall_latency_ms: latency_ms,
            tokens_used,
            memories_retrieved: summaries.len(),
        });
    }

    Ok(all_results)
}
