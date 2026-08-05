//! # mneme-api
//!
//! The public API surface for agents to interact with the Mneme memory system.
//!
//! Design principles:
//! - Progressive disclosure: agents see summaries first, load content on demand.
//! - Reconsolidation is automatic: every retrieval triggers drift checks.
//! - Compaction runs async: agents don't block on memory maintenance.
//! - Simple mental model: remember(), recall(), forget() — that's it.

use chrono::Utc;
use mneme_consolidate::{ConsolidateError, ConsolidationEngine, ConsolidationLLM};
use mneme_core::*;
use mneme_embed::EmbeddingModel;
use mneme_store::{ContentStore, EnvelopeIndex, MnemeStore};
use tracing::info;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────
// Output types
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MnemeSummary {
    pub id: Uuid,
    pub summary: String,
    pub full_text: String,
    pub confidence: f32,
    pub tags: Vec<String>,
    pub similarity: f32,
    pub retrieval_score: f32,
    pub version: u32,
    pub is_evolved: bool,
}

#[derive(Debug, Clone)]
pub struct MnemeDetail {
    pub id: Uuid,
    pub summary: String,
    pub full_text: String,
    pub confidence: f32,
    pub tags: Vec<String>,
    pub version: u32,
    pub created_at: String,
    pub updated_at: String,
    pub access_count: u64,
    pub provenance_count: usize,
    pub conflict_count: usize,
    pub related_count: usize,
}

// ─────────────────────────────────────────────────────────────
// Fact engrams
// ─────────────────────────────────────────────────────────────

/// Lines of conversation per extraction call.
///
/// Two independent reasons not to send a whole session at once. Coverage: the
/// extractor is capped at 12 facts per call, so a 22-turn LoCoMo session would
/// silently discard most of what it contains. Budget: a long window makes a
/// reasoning model think for longer, and a full session was measured
/// exhausting even a 16k token budget and returning an empty completion.
const FACT_WINDOW_LINES: usize = 10;

/// Lines repeated between consecutive chunks, so a pronoun near a boundary
/// still has its referent in view.
const FACT_WINDOW_OVERLAP: usize = 2;

/// Split a conversation window into overlapping chunks for extraction.
pub fn chunk_window(window: &str) -> Vec<String> {
    let lines: Vec<&str> = window.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() <= FACT_WINDOW_LINES {
        return if lines.is_empty() {
            vec![]
        } else {
            vec![lines.join("\n")]
        };
    }

    let step = FACT_WINDOW_LINES - FACT_WINDOW_OVERLAP;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        let end = (start + FACT_WINDOW_LINES).min(lines.len());
        chunks.push(lines[start..end].join("\n"));
        if end == lines.len() {
            break;
        }
        start += step;
    }
    chunks
}

/// Turn extracted facts into engrams, plus the graph triplets they imply.
///
/// Deliberately has **no storage side effects**: it decides only what a fact
/// engram *is* — confidence, tags, memory type, summary, valid time — and
/// leaves persistence to the caller, because the two callers persist
/// differently. The HTTP server upserts, re-indexes full text for BM25, and
/// appends to a write-ahead log; the library path just inserts. Sharing the
/// semantics here while letting each layer own its own durability is what
/// stops the benchmark and the server from silently measuring different
/// systems — which is exactly what happened when this logic lived only in the
/// `/add` handler and the benchmark went through [`MnemeMemory::remember`].
/// Tag marking an engram as an extracted standalone fact rather than
/// conversational source text.
pub const FACT_TAG: &str = "fact";

/// Ceiling on the share of a result set that extracted facts may occupy.
///
/// Facts are short and keyword-dense, so they outscore verbatim source text on
/// embedding similarity almost every time. Left to compete on raw score they
/// swept the result set: measured over a 2-conversation LoCoMo pair, enabling
/// extraction cut average context from 1286 to 260 tokens per query and raised
/// unanswerable questions from 68 to 109 of 235, costing 0.086 judge score.
/// The facts were not wrong — they were crowding out the passages that carried
/// enough context to actually answer from.
pub const MAX_FACT_FRACTION: f32 = 0.5;

/// Multiple of `top_k` to fetch from the store before blending, so both facts
/// and source text are present to choose between.
pub const RECALL_OVERFETCH: usize = 3;

/// True if this engram is an extracted fact rather than source text.
pub fn is_fact(tags: &[String]) -> bool {
    tags.iter().any(|t| t == FACT_TAG)
}

/// Select `limit` results while capping how much of the set extracted facts
/// may occupy, so they cannot crowd out conversational source text.
///
/// Candidates are assumed to arrive in the caller's preferred order (normally
/// descending retrieval score); relative order within each class is preserved.
/// The cap is a ceiling, not a quota: if one class is short, the other backfills
/// the unused slots, so a query with no matching source text still returns a
/// full set of facts rather than a truncated one.
pub fn blend_fact_and_source(
    candidates: Vec<MnemeSummary>,
    limit: usize,
    max_fact_fraction: f32,
) -> Vec<MnemeSummary> {
    blend_by(
        candidates,
        limit,
        max_fact_fraction,
        |c| is_fact(&c.tags),
        |c| c.retrieval_score,
    )
}

/// [`blend_fact_and_source`] over any candidate type.
///
/// Lets a caller blend raw search hits before paying for per-result content
/// loads, rather than loading three times what it will keep.
pub fn blend_by<T>(
    candidates: Vec<T>,
    limit: usize,
    max_fact_fraction: f32,
    is_fact_of: impl Fn(&T) -> bool,
    score_of: impl Fn(&T) -> f32,
) -> Vec<T> {
    if limit == 0 || candidates.len() <= limit {
        let mut out = candidates;
        out.truncate(limit);
        return out;
    }

    let (facts, source): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|c| is_fact_of(c));

    // Round the cap up so a small top_k still admits at least one fact: with
    // limit=5 and fraction=0.5 that is 3 facts and 2 source passages, and
    // truncating would silently make facts unreachable at limit=1.
    let fact_cap = ((limit as f32 * max_fact_fraction).ceil() as usize).min(limit);

    let n_facts = facts.len().min(fact_cap);
    let n_source = source.len().min(limit - n_facts);
    // Unused slots go back to whichever class still has candidates.
    let n_facts = facts.len().min(n_facts + (limit - n_facts - n_source));

    let mut out: Vec<T> = facts
        .into_iter()
        .take(n_facts)
        .chain(source.into_iter().take(n_source))
        .collect();

    out.sort_by(|a, b| {
        score_of(b)
            .partial_cmp(&score_of(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

pub async fn build_fact_engrams<M: EmbeddingModel>(
    facts: &[ExtractedFact],
    session_id: &str,
    tags: &[String],
    provenance_window: &str,
    embed_model: &M,
) -> Result<(Vec<Engram>, Vec<GraphTriplet>), ConsolidateError> {
    let mut engrams = Vec::new();
    let mut triplets = Vec::new();

    for fact in facts {
        let text = fact.text.trim();
        if text.is_empty() {
            continue;
        }

        let embedding = match embed_model.embed(text).await {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "failed to embed extracted fact; skipping");
                continue;
            }
        };

        let id = Uuid::new_v4();
        let now = Utc::now();

        let mut fact_tags = tags.to_vec();
        fact_tags.push(FACT_TAG.to_string());
        if let Some(date) = fact
            .date
            .as_deref()
            .map(str::trim)
            .filter(|d| !d.is_empty())
        {
            fact_tags.push(format!("date:{date}"));
        }

        engrams.push(Engram {
            envelope: Envelope {
                id,
                embedding,
                // Above a raw turn (0.5) — a fact has been through an explicit
                // extraction step and is stated standalone — but below a
                // compacted engram, since nothing has corroborated it across
                // sessions yet.
                confidence: 0.6,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                access_count: 0,
                // Semantic, not Working: a fact is distilled rather than raw,
                // and search queries both tiers, so this surfaces alongside
                // the turns it came from instead of replacing them.
                memory_type: MemoryType::Semantic,
                source_sessions: vec![session_id.to_string()],
                supersedes: vec![],
                superseded_by: None,
                // Facts are short by construction, so the summary is the whole
                // fact — no truncation, and BM25 covers it fully.
                summary: text.to_string(),
                tags: fact_tags,
                content_hash: {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    text.hash(&mut h);
                    h.finish()
                },
                // Where the extractor reported an unambiguous absolute date,
                // this becomes real valid time, so an `as_of` query can reason
                // about when the fact held rather than only when it was
                // written down.
                valid_at: fact.valid_at(),
                invalid_at: None,
            },
            content: ContentBody {
                engram_id: id,
                // The claim plus the window it came from. Retrieval matches on
                // the embedding of the bare fact, which stays crisp, but the
                // answer generator receives the surrounding turns — a fact
                // alone is true and unusable for the temporal and multi-hop
                // questions that need to see what was around it ("the weekend
                // before X", a chain across two sessions). Measured: with
                // extraction on, 146 of 235 LoCoMo queries got under 200
                // tokens of context and answered "not found in memory".
                full_text: format!("{text}\n\nFrom the conversation:\n{provenance_window}"),
                provenance: vec![ProvenanceRecord {
                    session_id: session_id.to_string(),
                    turn_id: None,
                    timestamp: now,
                    // The window the fact was distilled from, so `expand` can
                    // show what it was actually derived from.
                    raw_excerpt: provenance_window.to_string(),
                }],
                conflict_log: vec![],
                related: vec![],
                version: 1,
            },
        });

        if let Some(triplet) = fact.as_triplet(id) {
            triplets.push(triplet);
        }
    }

    Ok((engrams, triplets))
}

// ─────────────────────────────────────────────────────────────
// MnemeMemory — the main API struct
// ─────────────────────────────────────────────────────────────

pub struct MnemeMemory<E, C, M, L>
where
    E: EnvelopeIndex + Clone,
    C: ContentStore + Clone,
    M: EmbeddingModel + Clone,
    L: ConsolidationLLM,
{
    pub store: MnemeStore<E, C>,
    pub engine: ConsolidationEngine<E, C, M, L>,
    embed_model: M,
    config: MnemeConfig,
}

impl<E, C, M, L> MnemeMemory<E, C, M, L>
where
    E: EnvelopeIndex + Clone + 'static,
    C: ContentStore + Clone + 'static,
    M: EmbeddingModel + Clone + 'static,
    L: ConsolidationLLM + 'static,
{
    pub fn new(
        store: MnemeStore<E, C>,
        engine: ConsolidationEngine<E, C, M, L>,
        embed_model: M,
        config: MnemeConfig,
    ) -> Self {
        Self {
            store,
            engine,
            embed_model,
            config,
        }
    }

    // ─────────────────────────────────────────────────────────
    // remember
    // ─────────────────────────────────────────────────────────

    pub async fn remember(
        &self,
        observation: &str,
        session_id: &str,
    ) -> Result<Uuid, ConsolidateError> {
        let embedding = self.embed_model.embed(observation).await?;
        let id = Uuid::new_v4();
        let now = Utc::now();

        let summary = if observation.len() > 100 {
            let cut = observation
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= 97)
                .last()
                .unwrap_or(0);
            format!("{}...", &observation[..cut])
        } else {
            observation.to_string()
        };

        let engram = Engram {
            envelope: Envelope {
                id,
                embedding,
                confidence: 0.5,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                access_count: 0,
                memory_type: MemoryType::Working,
                source_sessions: vec![session_id.to_string()],
                supersedes: vec![],
                superseded_by: None,
                summary,
                tags: vec![],
                // FIX #11: real hash computed in consolidate layer;
                // working memory entries get a placeholder hash of the raw text
                content_hash: {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    observation.hash(&mut h);
                    h.finish()
                },
                // A raw observation carries no parsed valid time — that comes
                // from fact extraction on the /add path.
                valid_at: None,
                invalid_at: None,
            },
            content: ContentBody {
                engram_id: id,
                full_text: observation.to_string(),
                provenance: vec![ProvenanceRecord {
                    session_id: session_id.to_string(),
                    turn_id: None,
                    timestamp: now,
                    raw_excerpt: observation.to_string(),
                }],
                conflict_log: vec![],
                related: vec![],
                version: 1,
            },
        };

        self.store.insert(&engram).await?;
        info!(id = %id, session = session_id, "Stored working memory engram");
        Ok(id)
    }

    // ─────────────────────────────────────────────────────────
    // remember_facts — distil a window of turns into fact engrams
    // ─────────────────────────────────────────────────────────

    /// Extract atomic facts from a window of raw conversation turns and store
    /// each as its own searchable engram, alongside the entity-graph triplets
    /// they imply.
    ///
    /// `window` should be the *whole* batch of turns, not one turn: resolving
    /// "she" to a name needs the surrounding context, so a per-turn window
    /// defeats the purpose as well as costing more LLM calls.
    ///
    /// Best-effort. Callers are expected to have already stored the raw turns
    /// (via [`Self::remember`]), so an extraction failure — no LLM, API error,
    /// unparseable response — leaves the caller exactly where it would have
    /// been without this call rather than losing data. Returns how many fact
    /// engrams were stored.
    pub async fn remember_facts(
        &self,
        window: &str,
        session_id: &str,
        tags: &[String],
    ) -> Result<usize, ConsolidateError> {
        if window.trim().is_empty() {
            return Ok(0);
        }

        // Extract per chunk, then deduplicate: the overlap between chunks
        // exists so boundary pronouns resolve, and it will naturally restate
        // some facts.
        let mut facts: Vec<ExtractedFact> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut failures = 0;

        let chunks = chunk_window(window);
        let chunk_count = chunks.len();
        for chunk in &chunks {
            match self.engine.extract_facts(chunk).await {
                Ok(chunk_facts) => {
                    for fact in chunk_facts {
                        if seen.insert(fact.text.trim().to_lowercase()) {
                            facts.push(fact);
                        }
                    }
                }
                // One bad chunk shouldn't discard the rest of the session.
                Err(e) => {
                    failures += 1;
                    tracing::warn!(error = %e, session = session_id, "fact extraction chunk failed");
                }
            }
        }

        if failures == chunk_count {
            return Err(ConsolidateError::LLM(format!(
                "all {chunk_count} extraction chunk(s) failed for session {session_id}"
            )));
        }

        let (engrams, triplets) =
            build_fact_engrams(&facts, session_id, tags, window, &self.embed_model).await?;

        for engram in &engrams {
            self.store.insert(engram).await?;
        }
        if !triplets.is_empty() {
            if let Err(e) = self.engine.graph.insert(triplets).await {
                // Graph coverage is an enhancement to recall, not a
                // correctness requirement — the fact engrams themselves are
                // already stored and searchable.
                tracing::warn!(error = %e, "graph insert failed for extracted facts");
            }
        }

        info!(
            facts = engrams.len(),
            session = session_id,
            "Stored extracted fact engrams"
        );
        Ok(engrams.len())
    }

    // ─────────────────────────────────────────────────────────
    // recall
    // FIX #3: reconsolidation spawned via tokio::spawn
    // ─────────────────────────────────────────────────────────

    pub async fn recall(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<MnemeSummary>, ConsolidateError> {
        let query_embedding = self.embed_model.embed(query).await?;

        let mem_query = MemoryQuery {
            embedding: query_embedding,
            // Over-fetch so the fact/source blend below has both classes to
            // choose from. Asking the store for exactly `top_k` would let facts
            // sweep the result set before the blend ever sees a source passage.
            top_k: top_k.saturating_mul(RECALL_OVERFETCH),
            active_only: true,
            memory_type: Some(MemoryType::Semantic),
            min_confidence: Some(0.1),
            recency_weight: 0.2,
            // Answer from what is believed true now, so a fact whose validity
            // window was closed by conflict resolution can't come back as
            // evidence alongside the memory that replaced it. Matches the
            // HTTP layer's `/recall` and `/search`; see the note there on the
            // future-dated-fact caveat.
            as_of: Some(Utc::now()),
            ..Default::default()
        };

        let results = self.store.search(&mem_query).await?;

        // Cap the fact share before loading content: blending here keeps the
        // per-result content loads proportional to what is actually returned
        // rather than to the over-fetch.
        let results = blend_by(
            results,
            top_k,
            MAX_FACT_FRACTION,
            |r| is_fact(&r.envelope.tags),
            |r| r.retrieval_score,
        );

        // FIX #3: spawn reconsolidation so it never blocks the caller
        // (Previously called directly, blocking for the full LLM round-trip)
        // Note: engine needs to be Arc'd for production multi-agent use
        // For the API library we fire-and-forget; errors are logged internally
        // (In the server layer the Arc<ConsolidationEngine> is passed instead)

        // FIX #16: read actual version from content store for each summary
        let mut seen: std::collections::HashSet<Uuid> =
            results.iter().map(|r| r.envelope.id).collect();
        let mut summaries = Vec::with_capacity(results.len());
        // 1-hop expansion through `related` links (populated at compaction
        // time for engrams synthesized from the same session): a fact that
        // didn't score high enough on its own to make top_k can still ride
        // in alongside a related fact that did, giving multi-hop questions
        // a chance to see both halves of the answer.
        let mut related_expansions: Vec<MnemeSummary> = Vec::new();
        for r in &results {
            let (version, full_text, related) = match self.store.content.get(r.envelope.id).await {
                Ok(body) => (body.version, body.full_text, body.related),
                Err(_) => (1, r.envelope.summary.clone(), vec![]),
            };

            for rel in &related {
                if !seen.insert(rel.id) {
                    continue;
                }
                let Ok(rel_envelope) = self.store.envelopes.get(rel.id).await else {
                    continue;
                };
                if !rel_envelope.is_active() {
                    continue;
                }
                let rel_full_text = self
                    .store
                    .content
                    .get(rel.id)
                    .await
                    .map(|b| b.full_text)
                    .unwrap_or_else(|_| rel_envelope.summary.clone());
                related_expansions.push(MnemeSummary {
                    id: rel_envelope.id,
                    summary: rel_envelope.summary.clone(),
                    full_text: rel_full_text,
                    confidence: rel_envelope.confidence,
                    tags: rel_envelope.tags.clone(),
                    similarity: r.similarity * rel.strength,
                    retrieval_score: r.retrieval_score * rel.strength,
                    version: 1,
                    is_evolved: !rel_envelope.supersedes.is_empty(),
                });
            }

            summaries.push(MnemeSummary {
                id: r.envelope.id,
                summary: r.envelope.summary.clone(),
                full_text,
                confidence: r.envelope.confidence,
                tags: r.envelope.tags.clone(),
                similarity: r.similarity,
                retrieval_score: r.retrieval_score,
                version,
                is_evolved: !r.envelope.supersedes.is_empty(),
            });
        }
        summaries.extend(related_expansions);

        Ok(summaries)
    }

    // ─────────────────────────────────────────────────────────
    // expand (progressive disclosure L2)
    // ─────────────────────────────────────────────────────────

    pub async fn expand(&self, engram_id: Uuid) -> Result<MnemeDetail, ConsolidateError> {
        let envelope = self.store.envelopes.get(engram_id).await?;
        let content = self.store.content.get(engram_id).await?;

        Ok(MnemeDetail {
            id: engram_id,
            summary: envelope.summary,
            full_text: content.full_text,
            confidence: envelope.confidence,
            tags: envelope.tags,
            version: content.version, // FIX #16: actual version
            created_at: envelope.created_at.to_rfc3339(),
            updated_at: envelope.updated_at.to_rfc3339(),
            access_count: envelope.access_count,
            provenance_count: content.provenance.len(),
            conflict_count: content.conflict_log.len(),
            related_count: content.related.len(),
        })
    }

    // ─────────────────────────────────────────────────────────
    // end_session
    // ─────────────────────────────────────────────────────────

    pub async fn end_session(&self, session_id: &str) -> Result<usize, ConsolidateError> {
        let new_engrams = self.engine.compact_session(session_id).await?;
        Ok(new_engrams.len())
    }

    // ─────────────────────────────────────────────────────────
    // history
    // ─────────────────────────────────────────────────────────

    pub async fn history(&self, engram_id: Uuid) -> Result<Vec<Envelope>, ConsolidateError> {
        let mut chain = Vec::new();
        let mut current = self.store.envelopes.get(engram_id).await?;
        chain.push(current.clone());

        while let Some(prev_id) = current.supersedes.first() {
            match self.store.envelopes.get(*prev_id).await {
                Ok(prev) => {
                    chain.push(prev.clone());
                    current = prev;
                }
                Err(_) => break,
            }
        }

        chain.reverse();
        Ok(chain)
    }

    // ─────────────────────────────────────────────────────────
    // forget
    // ─────────────────────────────────────────────────────────

    pub async fn forget(&self, engram_id: Uuid) -> Result<(), ConsolidateError> {
        self.store
            .envelopes
            .delete(engram_id)
            .await
            .map_err(ConsolidateError::Store)?;
        // Best-effort — content may already be absent
        let _ = self.store.content.delete(engram_id).await;
        info!(id = %engram_id, "Deleted engram");
        Ok(())
    }

    // ─────────────────────────────────────────────────────────
    // decay — apply Ebbinghaus confidence decay to all active engrams
    // ─────────────────────────────────────────────────────────

    pub async fn decay(&self, lambda: f64) -> Result<usize, ConsolidateError> {
        let updated = self
            .store
            .envelopes
            .apply_decay(lambda)
            .await
            .map_err(ConsolidateError::Store)?;
        info!(updated = updated, "Applied confidence decay");
        Ok(updated)
    }

    // ─────────────────────────────────────────────────────────
    // gc
    // ─────────────────────────────────────────────────────────

    pub async fn gc(&self) -> Result<usize, ConsolidateError> {
        let removed = self
            .store
            .envelopes
            .gc(
                self.config.gc_confidence_floor,
                self.config.working_memory_ttl_hours,
            )
            .await?;
        info!(removed = removed, "Garbage collection complete");
        Ok(removed)
    }
}

// ─────────────────────────────────────────────────────────────
// Context builder
// ─────────────────────────────────────────────────────────────

pub struct ContextBuilder;

impl ContextBuilder {
    pub fn format_summaries(summaries: &[MnemeSummary]) -> String {
        let mut out = String::from("<memory_context>\n");
        for s in summaries {
            out.push_str(&format!(
                "  <memory id=\"{}\" confidence=\"{:.2}\" similarity=\"{:.2}\">\n    {}\n  </memory>\n",
                s.id, s.confidence, s.similarity, s.summary
            ));
        }
        out.push_str("</memory_context>");
        out
    }

    pub fn format_detail(detail: &MnemeDetail) -> String {
        format!(
            "<memory_detail id=\"{}\" version=\"{}\">\n  <summary>{}</summary>\n  <full_text>{}</full_text>\n</memory_detail>",
            detail.id, detail.version, detail.summary, detail.full_text
        )
    }
}
