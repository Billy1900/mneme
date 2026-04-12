//! # mneme-api
//!
//! The public API surface for agents to interact with the Mneme memory system.
//!
//! Design principles:
//! - Progressive disclosure: agents see summaries first, load content on demand.
//! - Reconsolidation is automatic: every retrieval triggers drift checks.
//! - Compaction runs async: agents don't block on memory maintenance.
//! - Simple mental model: remember(), recall(), forget() — that's it.

use mneme_consolidate::*;
use mneme_core::*;
use mneme_embed::EmbeddingModel;
use mneme_store::*;
use uuid::Uuid;
use chrono::Utc;
use tracing::info;

// ─────────────────────────────────────────────────────────────
// The agent-facing memory interface
// ─────────────────────────────────────────────────────────────

/// The primary interface agents use for memory operations.
///
/// ```text
/// let memory = MnemeMemory::new(store, embed, llm, config);
///
/// // During agent work session:
/// memory.remember("User wants sub-2ms latency for trading", session).await;
/// memory.remember("Considering Rust over C++ for safety", session).await;
///
/// // When agent needs context:
/// let context = memory.recall("what language for the trading system?").await;
/// // Returns: [MnemeSummary { summary: "User prefers Rust for systems...", ... }]
///
/// // If agent needs full detail:
/// let details = memory.expand(&context[0].id).await;
///
/// // End of session — compaction runs automatically:
/// memory.end_session(session).await;
/// ```
pub struct MnemeMemory<E, C, M, L>
where
    E: EnvelopeIndex,
    C: ContentStore,
    M: EmbeddingModel,
    L: ConsolidationLLM,
{
    engine: ConsolidationEngine<E, C, M, L>,
    store: MnemeStore<E, C>,
    embed_model: M,
    config: MnemeConfig,
}

/// A lightweight summary returned from recall() — progressive disclosure L1.
/// The agent reads these and decides which to expand().
#[derive(Debug, Clone)]
pub struct MnemeSummary {
    pub id: Uuid,
    pub summary: String,
    pub confidence: f32,
    pub tags: Vec<String>,
    pub similarity: f32,
    pub retrieval_score: f32,
    pub version: u32, // inferred from envelope
    pub is_evolved: bool, // has supersedes entries
}

/// Full memory content returned from expand() — progressive disclosure L2.
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

impl<E, C, M, L> MnemeMemory<E, C, M, L>
where
    E: EnvelopeIndex + Clone,
    C: ContentStore + Clone,
    M: EmbeddingModel + Clone,
    L: ConsolidationLLM,
{
    // ─────────────────────────────────────────────────────────
    // Core API: remember / recall / expand / forget
    // ─────────────────────────────────────────────────────────

    /// Store a new observation in working memory.
    ///
    /// This is a fast, cheap write. The observation gets compacted into
    /// semantic memory later (on session end or buffer threshold).
    pub async fn remember(
        &self,
        observation: &str,
        session_id: &str,
    ) -> Result<Uuid, ConsolidateError> {
        let embedding = self.embed_model.embed(observation).await?;
        let id = Uuid::new_v4();
        let now = Utc::now();

        let summary = if observation.len() > 100 {
            format!("{}...", &observation[..97])
        } else {
            observation.to_string()
        };

        let engram = Engram {
            envelope: Envelope {
                id,
                embedding,
                confidence: 0.5, // working memory starts at neutral confidence
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
                content_hash: 0, // computed during compaction
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
        info!(id = %id, session = session_id, "Remembered observation");

        // Check if buffer threshold reached → trigger compaction
        let wm_count = self
            .store
            .envelopes
            .list_working_memory(session_id)
            .await?
            .len();

        if wm_count >= self.config.compaction_buffer_threshold {
            info!(
                session = session_id,
                count = wm_count,
                "Buffer threshold reached, triggering compaction"
            );
            // Spawn compaction as background task (don't block the agent)
            // In production: tokio::spawn(self.engine.compact_session(session_id));
            let _ = self.engine.compact_session(session_id).await;
        }

        Ok(id)
    }

    /// Retrieve relevant memories for a query.
    ///
    /// Returns lightweight summaries (progressive disclosure L1).
    /// The agent reads summaries and calls expand() for the ones it needs.
    ///
    /// **Reconsolidation runs automatically**: every retrieval checks for
    /// drift and evolves stale engrams in the background.
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
            memory_type: Some(MemoryType::Semantic), // recall from semantic only
            min_confidence: Some(0.1),
            recency_weight: 0.2,
            ..Default::default()
        };

        let results = self.store.search(&mem_query).await?;

        // Trigger reconsolidation in background (non-blocking)
        // In production: tokio::spawn(...)
        let _ = self.engine.reconsolidate(&results, query).await;

        // Convert to summaries
        let summaries = results
            .into_iter()
            .map(|r| MnemeSummary {
                id: r.envelope.id,
                summary: r.envelope.summary.clone(),
                confidence: r.envelope.confidence,
                tags: r.envelope.tags.clone(),
                similarity: r.similarity,
                retrieval_score: r.retrieval_score,
                version: 1, // would need content body for actual version
                is_evolved: !r.envelope.supersedes.is_empty(),
            })
            .collect();

        Ok(summaries)
    }

    /// Load full content for a specific engram (progressive disclosure L2).
    ///
    /// Call this after recall() when the agent needs the full text.
    pub async fn expand(&self, engram_id: Uuid) -> Result<MnemeDetail, ConsolidateError> {
        let envelope = self.store.envelopes.get(engram_id).await?;
        let content = self.store.content.get(engram_id).await?;

        Ok(MnemeDetail {
            id: engram_id,
            summary: envelope.summary,
            full_text: content.full_text,
            confidence: envelope.confidence,
            tags: envelope.tags,
            version: content.version,
            created_at: envelope.created_at.to_rfc3339(),
            updated_at: envelope.updated_at.to_rfc3339(),
            access_count: envelope.access_count,
            provenance_count: content.provenance.len(),
            conflict_count: content.conflict_log.len(),
            related_count: content.related.len(),
        })
    }

    /// Explicitly end a session, triggering compaction of all remaining
    /// working memory entries.
    pub async fn end_session(&self, session_id: &str) -> Result<usize, ConsolidateError> {
        let new_engrams = self.engine.compact_session(session_id).await?;
        Ok(new_engrams.len())
    }

    /// Get the full version history of an engram (the supersession chain).
    /// Useful for debugging and auditing: "how did this memory evolve?"
    pub async fn history(&self, engram_id: Uuid) -> Result<Vec<Envelope>, ConsolidateError> {
        let mut chain = Vec::new();
        let mut current = self.store.envelopes.get(engram_id).await?;
        chain.push(current.clone());

        // Walk backwards through supersedes chain
        while let Some(prev_id) = current.supersedes.first() {
            match self.store.envelopes.get(*prev_id).await {
                Ok(prev) => {
                    chain.push(prev.clone());
                    current = prev;
                }
                Err(_) => break, // GC'd predecessor
            }
        }

        chain.reverse(); // oldest first
        Ok(chain)
    }

    /// Run garbage collection: remove low-confidence, old, superseded engrams.
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
// Context builder: format memories for injection into agent prompt
// ─────────────────────────────────────────────────────────────

/// Formats retrieved memories for injection into an agent's system prompt
/// or context window. Handles the progressive disclosure formatting.
pub struct ContextBuilder;

impl ContextBuilder {
    /// Format summaries for L1 disclosure (agent decides what to expand).
    pub fn format_summaries(summaries: &[MnemeSummary]) -> String {
        if summaries.is_empty() {
            return String::from("<memories>No relevant memories found.</memories>");
        }

        let mut out = String::from("<memories>\n");
        for (i, s) in summaries.iter().enumerate() {
            out.push_str(&format!(
                "  <memory index=\"{}\" id=\"{}\" confidence=\"{:.2}\"",
                i, s.id, s.confidence,
            ));
            if s.is_evolved {
                out.push_str(" evolved=\"true\"");
            }
            out.push_str(&format!(">\n    {}\n  </memory>\n", s.summary));
        }
        out.push_str("</memories>");
        out
    }

    /// Format full details for L2 disclosure (after agent selected specific memories).
    pub fn format_details(details: &[MnemeDetail]) -> String {
        let mut out = String::from("<memory_details>\n");
        for d in details {
            out.push_str(&format!(
                "  <detail id=\"{}\" version=\"{}\" confidence=\"{:.2}\" accessed=\"{}\">\n    {}\n  </detail>\n",
                d.id, d.version, d.confidence, d.access_count, d.full_text
            ));
        }
        out.push_str("</memory_details>");
        out
    }
}
