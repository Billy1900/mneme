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
        fact_tags.push("fact".to_string());
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
                full_text: text.to_string(),
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

        let facts = self.engine.extract_facts(window).await?;
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
            top_k,
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
