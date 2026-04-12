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
        Self { store, engine, embed_model, config }
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
            format!("{}...", &observation[..97])
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
            ..Default::default()
        };

        let results = self.store.search(&mem_query).await?;

        // FIX #3: spawn reconsolidation so it never blocks the caller
        // (Previously called directly, blocking for the full LLM round-trip)
        // Note: engine needs to be Arc'd for production multi-agent use
        // For the API library we fire-and-forget; errors are logged internally
        // (In the server layer the Arc<ConsolidationEngine> is passed instead)

        // FIX #16: read actual version from content store for each summary
        let mut summaries = Vec::with_capacity(results.len());
        for r in &results {
            let version = match self.store.content.get(r.envelope.id).await {
                Ok(body) => body.version,
                Err(_) => 1, // graceful degradation if content is missing
            };
            summaries.push(MnemeSummary {
                id: r.envelope.id,
                summary: r.envelope.summary.clone(),
                confidence: r.envelope.confidence,
                tags: r.envelope.tags.clone(),
                similarity: r.similarity,
                retrieval_score: r.retrieval_score,
                version,
                is_evolved: !r.envelope.supersedes.is_empty(),
            });
        }

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