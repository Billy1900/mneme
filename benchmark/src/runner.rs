use anyhow::Result;
use futures::stream::{self, StreamExt};
use mneme_api::{MnemeMemory, MnemeSummary};
use mneme_consolidate::{
    AnthropicLLM, ConsolidationEngine, ConsolidationLLM, DeepSeekLLM, OpenAILLM,
};
use mneme_core::{MemoryQuery, MnemeConfig};
use mneme_embed::EmbeddingModel;
use mneme_store::{
    ContentStore, EnvelopeIndex, InMemoryContentStore, InMemoryEnvelopeIndex, MnemeStore,
};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::{info, warn};
use uuid::Uuid;

/// Max concurrent per-question API calls (recall + rerank + generate + judge).
/// Kept low enough to avoid gpt-4o rate limits (~10k RPM on tier-1).
const QUESTION_CONCURRENCY: usize = 8;

use crate::datasets::{LoCoMoConversation, LongMemEvalItem};
use crate::judge::LLMJudge;
use crate::metrics::{self, QuestionResult};

/// Extract entities from the question and traverse the entity-relation graph
/// (built from triplets extracted at compaction time) out 2 hops, pulling in
/// the source engram of every triplet touched. Catches facts that connect to
/// the question's entities but didn't score high enough on any sub-query's
/// vector similarity to be retrieved directly.
async fn recall_graph(session: &Session, judge: &LLMJudge, question: &str) -> Vec<MnemeSummary> {
    let entities = match judge.extract_entities(question).await {
        Ok(e) => e,
        Err(e) => {
            warn!("entity extraction failed: {e}");
            return vec![];
        }
    };
    if entities.is_empty() {
        return vec![];
    }

    let mut source_ids: HashSet<Uuid> = HashSet::new();
    for entity in &entities {
        match session.memory.engine.graph.neighbors(entity, 2).await {
            Ok(triplets) => source_ids.extend(triplets.into_iter().map(|t| t.source_engram_id)),
            Err(e) => warn!("graph neighbors failed: {e}"),
        }
    }

    let mut out = Vec::with_capacity(source_ids.len());
    for id in source_ids {
        let envelope = match session.memory.store.envelopes.get(id).await {
            Ok(e) if e.is_active() => e,
            _ => continue,
        };
        let full_text = match session.memory.store.content.get(id).await {
            Ok(c) => c.full_text,
            Err(_) => envelope.summary.clone(),
        };
        out.push(MnemeSummary {
            id: envelope.id,
            summary: envelope.summary.clone(),
            full_text,
            confidence: envelope.confidence,
            tags: envelope.tags.clone(),
            // Neutral score — graph hits are a distinct evidence channel from
            // vector similarity, not directly comparable to it. Merged and
            // left to the reranker rather than competing on retrieval_score.
            similarity: 0.5,
            retrieval_score: 0.5,
            version: 1,
            is_evolved: !envelope.supersedes.is_empty(),
        });
    }
    out
}

/// Decompose question into sub-queries, recall for each in parallel, merge and deduplicate.
/// Falls back to single-query recall if decomposition fails or returns one item.
async fn recall_multihop(
    session: &Session,
    embed: &Arc<dyn EmbeddingModel>,
    judge: &LLMJudge,
    question: &str,
    top_k: usize,
) -> Vec<MnemeSummary> {
    let sub_qs = match judge.decompose_question(question).await {
        Ok(qs) => qs,
        Err(e) => {
            warn!("decompose failed: {e}");
            vec![question.to_string()]
        }
    };

    let is_multihop = sub_qs.len() > 1;
    let queries: Vec<String> = if is_multihop {
        sub_qs
    } else {
        vec![question.to_string()]
    };

    // Recall for each sub-query in parallel, then merge deduped by id
    let per_sub = top_k.max(3);
    let mut all: Vec<MnemeSummary> = stream::iter(queries.iter())
        .map(|sq| async move { recall_with_fallback(session, embed, sq, per_sub).await })
        .buffer_unordered(queries.len())
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .flatten()
        .collect();

    // Graph traversal only pays for itself on genuinely multi-hop questions —
    // single-hop ones already get a direct vector hit from recall_with_fallback.
    if is_multihop {
        all.extend(recall_graph(session, judge, question).await);
    }

    // Deduplicate by id, preserving first occurrence (highest score per sub-query)
    let mut seen: HashSet<Uuid> = HashSet::new();
    all.retain(|s| seen.insert(s.id));

    // Sort by retrieval_score descending, keep top_k * 2 for reranker
    all.sort_by(|a, b| {
        b.retrieval_score
            .partial_cmp(&a.retrieval_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Cap the fact share of the pool the reranker sees. A plain truncate here
    // let facts take every slot before the reranker had a choice to make.
    mneme_api::blend_fact_and_source(all, top_k * 3, mneme_api::MAX_FACT_FRACTION)
}

/// Enforce the fact cap on the reranker's *output*, backfilling source text.
///
/// Capping the candidate pool is not enough: the reranker picks freely from
/// what it is given, and short keyword-dense facts read as the most relevant
/// answer to a question, so it returned all-fact sets anyway. This is the last
/// stage before the answer prompt, so it is the only one that decides what the
/// generator actually sees.
///
/// Rerank order is preserved among kept results; dropped fact slots are refilled
/// from the pool, source text first, falling back to facts if no source text is
/// available so the result set is never short.
fn cap_fact_share(
    selected: Vec<MnemeSummary>,
    pool: Vec<MnemeSummary>,
    top_k: usize,
) -> Vec<MnemeSummary> {
    let cap = ((top_k as f32 * mneme_api::MAX_FACT_FRACTION).ceil() as usize).min(top_k);
    if selected
        .iter()
        .filter(|s| mneme_api::is_fact(&s.tags))
        .count()
        <= cap
    {
        return selected;
    }

    let mut kept: Vec<MnemeSummary> = Vec::with_capacity(top_k);
    let mut facts_kept = 0;
    for s in selected {
        if mneme_api::is_fact(&s.tags) {
            if facts_kept < cap {
                facts_kept += 1;
                kept.push(s);
            }
        } else {
            kept.push(s);
        }
    }

    let mut have: HashSet<Uuid> = kept.iter().map(|s| s.id).collect();
    // Source text first; then facts, so a session that produced nothing but
    // facts still returns a full set rather than a truncated one.
    for allow_facts in [false, true] {
        for c in &pool {
            if kept.len() >= top_k {
                break;
            }
            if (allow_facts || !mneme_api::is_fact(&c.tags)) && have.insert(c.id) {
                kept.push(c.clone());
            }
        }
    }

    kept.truncate(top_k);
    kept
}

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
    DeepSeek(DeepSeekLLM),
}

// Blanket so we can pass ConsolidationLlm anywhere ConsolidationLLM is needed via dispatch.
// We use dynamic dispatch inside Session to avoid two generic instantiations.
type BenchMemory = MnemeMemory<
    InMemoryEnvelopeIndex,
    InMemoryContentStore,
    Arc<dyn EmbeddingModel>,
    Box<dyn ConsolidationLLM>,
>;

struct Session {
    memory: BenchMemory,
    envelopes: Arc<InMemoryEnvelopeIndex>,
}

fn new_session(
    embed: Arc<dyn EmbeddingModel>,
    llm: ConsolidationLlm,
    config: MnemeConfig,
) -> Session {
    let envelopes = Arc::new(InMemoryEnvelopeIndex::new());
    let content = Arc::new(InMemoryContentStore::new());

    let engine_store = MnemeStore::new((*envelopes).clone(), (*content).clone());
    let api_store = MnemeStore::new((*envelopes).clone(), (*content).clone());

    let boxed: Box<dyn ConsolidationLLM> = match llm {
        ConsolidationLlm::Anthropic(a) => Box::new(a),
        ConsolidationLlm::OpenAI(o) => Box::new(o),
        ConsolidationLlm::DeepSeek(d) => Box::new(d),
    };
    let engine = ConsolidationEngine::new(engine_store, embed.clone(), boxed, config.clone());
    let memory = MnemeMemory::new(api_store, engine, embed, config);

    Session { memory, envelopes }
}

/// Recall from both semantic (compacted) and working memory, merge by retrieval score.
/// Compaction loses fine-grained facts; raw working-memory turns fill the gap.
async fn recall_with_fallback(
    session: &Session,
    embed: &Arc<dyn EmbeddingModel>,
    query: &str,
    top_k: usize,
) -> Vec<MnemeSummary> {
    // Semantic recall (post-compaction engrams)
    let semantic = match session.memory.recall(query, top_k).await {
        Ok(s) => s,
        Err(e) => {
            warn!("recall error: {e}");
            vec![]
        }
    };

    // Always also search working memory (raw turns) for granular facts
    let embedding = match embed.embed(query).await {
        Ok(e) => e,
        Err(e) => {
            warn!("embed failed: {e}");
            return semantic;
        }
    };

    let wm_q = MemoryQuery {
        embedding,
        top_k,
        active_only: true,
        memory_type: None, // all types including Working
        min_confidence: Some(0.0),
        recency_weight: 0.1,
        // Match the HTTP /search path: answer from what is believed true now,
        // so a fact invalidated by conflict resolution can't come back as
        // evidence alongside the memory that replaced it.
        as_of: Some(chrono::Utc::now()),
        ..Default::default()
    };

    let working: Vec<MnemeSummary> = match session.envelopes.search(&wm_q).await {
        Ok(results) => {
            let mut out = Vec::with_capacity(results.len());
            for r in results {
                // Load the real body rather than reusing the envelope summary
                // as `full_text`. Raw turns are summarized by truncation at
                // ~100 chars, so using the summary here fed the reranker and
                // the answer generator a cut-off turn — on the very channel
                // that exists to supply the verbatim detail compaction loses.
                let full_text = match session.memory.store.content.get(r.envelope.id).await {
                    Ok(body) => body.full_text,
                    Err(_) => r.envelope.summary.clone(),
                };
                out.push(MnemeSummary {
                    id: r.envelope.id,
                    full_text,
                    summary: r.envelope.summary,
                    confidence: r.envelope.confidence,
                    tags: r.envelope.tags,
                    similarity: r.similarity,
                    retrieval_score: r.retrieval_score,
                    version: 1,
                    is_evolved: false,
                });
            }
            out
        }
        Err(e) => {
            warn!("working memory search error: {e}");
            vec![]
        }
    };

    // Merge: deduplicate by id, keep best retrieval_score, sort descending
    let mut seen: HashSet<Uuid> = HashSet::new();
    let mut merged: Vec<MnemeSummary> = semantic
        .into_iter()
        .chain(working)
        .filter(|s| seen.insert(s.id))
        .collect();
    merged.sort_by(|a, b| {
        b.retrieval_score
            .partial_cmp(&a.retrieval_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Cap the fact share of the candidate set. A plain score sort here let
    // short, keyword-dense fact engrams outscore the verbatim turns that exist
    // precisely to supply what compaction drops — the reranker can only pick
    // from what it is handed, so crowding at this step is unrecoverable
    // downstream.
    mneme_api::blend_fact_and_source(merged, top_k * 2, mneme_api::MAX_FACT_FRACTION)
}

pub async fn run_locomo(
    conversations: &[LoCoMoConversation],
    embed: Arc<dyn EmbeddingModel>,
    llm: ConsolidationLlm,
    judge: &LLMJudge,
    top_k: usize,
    use_judge: bool,
    fact_extraction: bool,
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

        // Ingest sessions, prefixing each turn with its session date so relative
        // time references ("last Saturday") can be resolved against an absolute
        // anchor instead of being lost — LoCoMo's ground truth answers are dates.
        for s in &conv.sessions {
            let date_prefix = if s.session_date.trim().is_empty() {
                String::new()
            } else {
                format!("[{}] ", s.session_date)
            };
            let mut window = String::new();
            for turn in &s.turns {
                if turn.trim().is_empty() {
                    continue;
                }
                let dated_turn = format!("{date_prefix}{turn}");
                if let Err(e) = session.memory.remember(&dated_turn, &s.session_id).await {
                    warn!("remember failed: {e}");
                }
                window.push_str(&dated_turn);
                window.push('\n');
            }

            // Distil the session's turns into atomic fact engrams, the same
            // step the HTTP /add path runs. Without this the benchmark would
            // measure raw turns plus compaction only — i.e. not the system
            // that actually gets submitted.
            //
            // One call per session rather than per turn: pronoun resolution
            // needs the surrounding turns, and the date prefix carried into
            // the window is what lets the extractor populate valid time.
            if fact_extraction {
                if let Err(e) = session
                    .memory
                    .remember_facts(&window, &s.session_id, &[])
                    .await
                {
                    warn!("fact extraction failed for {}: {e}", s.session_id);
                }
            }

            if let Err(e) = session.memory.end_session(&s.session_id).await {
                warn!("end_session failed for {}: {e}", s.session_id);
            }
        }

        // Evaluate questions concurrently — recall/rerank/generate/judge are all independent.
        let sem = Arc::new(Semaphore::new(QUESTION_CONCURRENCY));
        let session = Arc::new(session);
        let embed_ref = Arc::new(embed.clone());
        let judge_ref: Arc<&LLMJudge> = Arc::new(judge);

        let mut conv_results: Vec<QuestionResult> = stream::iter(conv.questions.iter())
            .map(|q| {
                let sem = sem.clone();
                let session = session.clone();
                let embed_ref = embed_ref.clone();
                let judge_ref = judge_ref.clone();
                let q = q.clone();
                async move {
                    let _permit = sem.acquire().await.unwrap();
                    let t0 = Instant::now();
                    let candidates =
                        recall_multihop(&session, &embed_ref, &judge_ref, &q.question, 15).await;
                    let pool = candidates.clone();
                    let summaries = rerank(&judge_ref, &q.question, candidates, top_k).await;
                    let summaries = cap_fact_share(summaries, pool, top_k);
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
                        match judge_ref
                            .generate_answer(&q.question, &memory_context)
                            .await
                        {
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
                        if predicted == "Not found in memory" {
                            Some(0.0)
                        } else {
                            match judge_ref.score(&q.question, &q.answer, &predicted).await {
                                Ok(s) => Some(s),
                                Err(e) => {
                                    warn!("judge failed: {e}");
                                    None
                                }
                            }
                        }
                    } else {
                        None
                    };

                    QuestionResult {
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
                    }
                }
            })
            .buffer_unordered(QUESTION_CONCURRENCY)
            .collect()
            .await;

        all_results.append(&mut conv_results);
    }

    Ok(all_results)
}

pub async fn run_longmemeval(
    items: &[LongMemEvalItem],
    embed: Arc<dyn EmbeddingModel>,
    llm: ConsolidationLlm,
    judge: &LLMJudge,
    top_k: usize,
    use_judge: bool,
    fact_extraction: bool,
    limit: Option<usize>,
) -> Result<Vec<QuestionResult>> {
    let items = match limit {
        Some(n) => &items[..n.min(items.len())],
        None => items,
    };

    let config = MnemeConfig::default();
    let sem = Arc::new(Semaphore::new(QUESTION_CONCURRENCY));
    let judge = Arc::new(judge);

    // Each LongMemEval item has its own independent session — fully parallelisable.
    let all_results: Vec<QuestionResult> = stream::iter(items.iter().enumerate())
        .map(|(ii, item)| {
            let sem = sem.clone();
            let embed = embed.clone();
            let llm = llm.clone();
            let judge = judge.clone();
            let config = config.clone();
            let item = item.clone();
            async move {
                let _permit = sem.acquire().await.unwrap();
                info!(
                    "LongMemEval item {}/{}: {} turns",
                    ii + 1,
                    items.len(),
                    item.turns.len()
                );

                let session = new_session(embed.clone(), llm, config);
                let session_id = item.history_id.clone();

                let mut window = String::new();
                for turn in &item.turns {
                    if turn.trim().is_empty() {
                        continue;
                    }
                    if let Err(e) = session.memory.remember(turn, &session_id).await {
                        warn!("remember failed: {e}");
                    }
                    window.push_str(turn);
                    window.push('\n');
                }

                // Same fact-extraction step as the /add path — see the note in
                // `run_locomo`.
                if fact_extraction {
                    if let Err(e) = session
                        .memory
                        .remember_facts(&window, &session_id, &[])
                        .await
                    {
                        warn!("fact extraction failed for {session_id}: {e}");
                    }
                }

                if let Err(e) = session.memory.end_session(&session_id).await {
                    warn!("end_session failed: {e}");
                }

                let t0 = Instant::now();
                let candidates = recall_with_fallback(&session, &embed, &item.question, 15).await;
                let pool = candidates.clone();
                let summaries = rerank(&judge, &item.question, candidates, top_k).await;
                let summaries = cap_fact_share(summaries, pool, top_k);
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
                    if predicted == "Not found in memory" {
                        Some(0.0)
                    } else {
                        match judge.score(&item.question, &item.answer, &predicted).await {
                            Ok(s) => Some(s),
                            Err(e) => {
                                warn!("judge failed: {e}");
                                None
                            }
                        }
                    }
                } else {
                    None
                };

                QuestionResult {
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
                }
            }
        })
        .buffer_unordered(QUESTION_CONCURRENCY)
        .collect()
        .await;

    Ok(all_results)
}

#[cfg(test)]
mod cap_tests {
    use super::*;

    fn s(tag: &str, score: f32) -> MnemeSummary {
        MnemeSummary {
            id: Uuid::new_v4(),
            summary: String::new(),
            full_text: String::new(),
            confidence: 0.5,
            tags: vec![tag.to_string()],
            similarity: score,
            retrieval_score: score,
            version: 1,
            is_evolved: false,
        }
    }

    /// The reranker returning an all-fact set is the case that made 146 of 235
    /// LoCoMo queries answer from under 200 tokens of context.
    #[test]
    fn caps_an_all_fact_rerank_and_backfills_source() {
        let selected: Vec<_> = (0..5).map(|_| s("fact", 0.9)).collect();
        let pool: Vec<_> = selected
            .iter()
            .cloned()
            .chain((0..5).map(|_| s("turn", 0.4)))
            .collect();

        let out = cap_fact_share(selected, pool, 5);

        assert_eq!(out.len(), 5, "result set must stay full");
        assert_eq!(
            out.iter().filter(|x| mneme_api::is_fact(&x.tags)).count(),
            3,
            "facts capped at their share"
        );
        assert_eq!(
            out.iter().filter(|x| !mneme_api::is_fact(&x.tags)).count(),
            2,
            "freed slots refilled with source text"
        );
    }

    /// No source text anywhere: returning 3 of 5 would be a worse bug than the
    /// one being fixed.
    #[test]
    fn refills_with_facts_when_no_source_exists() {
        let selected: Vec<_> = (0..5).map(|_| s("fact", 0.9)).collect();
        let pool: Vec<_> = selected
            .iter()
            .cloned()
            .chain((0..3).map(|_| s("fact", 0.5)))
            .collect();

        let out = cap_fact_share(selected, pool, 5);
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn leaves_an_already_compliant_set_untouched() {
        let selected = vec![s("fact", 0.9), s("turn", 0.8), s("turn", 0.7)];
        let ids: Vec<_> = selected.iter().map(|x| x.id).collect();
        let out = cap_fact_share(selected.clone(), selected, 5);
        assert_eq!(out.iter().map(|x| x.id).collect::<Vec<_>>(), ids);
    }

    #[test]
    fn never_duplicates_a_result() {
        let selected: Vec<_> = (0..5).map(|_| s("fact", 0.9)).collect();
        let pool = selected.iter().cloned().chain([s("turn", 0.4)]).collect();
        let out = cap_fact_share(selected, pool, 5);
        let mut ids: Vec<_> = out.iter().map(|x| x.id).collect();
        ids.sort();
        let before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), before, "backfill must not re-add a kept result");
    }
}
