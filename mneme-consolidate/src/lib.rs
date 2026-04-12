//! # mneme-consolidate
//!
//! The consolidation engine + LLM backends.
//!
//! Backends:
//! - `AnthropicLLM`: Claude via the Anthropic API (production)
//! - `MockLLM`: deterministic responses for testing


use async_trait::async_trait;
use chrono::Utc;
use mneme_core::*;
use mneme_embed::{agglomerative_cluster, EmbeddingModel};
use mneme_store::*;
use tracing::{debug, info, warn};
use uuid::Uuid;

pub mod backends;
pub use backends::{AnthropicLLM, MockLLM};

#[async_trait]
pub trait ConsolidationLLM: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<String, ConsolidateError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ConsolidateError {
    #[error("store error: {0}")]
    Store(#[from] StoreError),
    #[error("embedding error: {0}")]
    Embed(#[from] mneme_embed::EmbedError),
    #[error("llm error: {0}")]
    LLM(String),
    #[error("parse error: {0}")]
    Parse(String),
}

pub struct ConsolidationEngine<E, C, M, L>
where
    E: EnvelopeIndex,
    C: ContentStore,
    M: EmbeddingModel,
    L: ConsolidationLLM,
{
    pub store: MnemeStore<E, C>,
    embed_model: M,
    llm: L,
    config: MnemeConfig,
}

pub struct ConflictResolution {
    pub strategy: ConflictStrategy,
    pub winner_id: Uuid,
    pub loser_id: Option<Uuid>,
    pub merged_engram: Option<Engram>,
}

// FIX #11: compute content_hash via DefaultHasher
fn seahash_str(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

impl<E, C, M, L> ConsolidationEngine<E, C, M, L>
where
    E: EnvelopeIndex + Clone + 'static,
    C: ContentStore + Clone + 'static,
    M: EmbeddingModel + Clone + 'static,
    L: ConsolidationLLM + 'static,
{
    pub fn new(store: MnemeStore<E, C>, embed_model: M, llm: L, config: MnemeConfig) -> Self {
        Self { store, embed_model, llm, config }
    }

    // ═══════════════════════════════════════════════════════════
    // OPERATION 1: Compaction
    // ═══════════════════════════════════════════════════════════

    pub async fn compact_session(&self, session_id: &str) -> Result<Vec<Engram>, ConsolidateError> {
        info!(session = session_id, "Starting compaction");

        let wm_envelopes = self.store.envelopes.list_working_memory(session_id).await?;
        if wm_envelopes.is_empty() {
            debug!("No working memory entries to compact");
            return Ok(vec![]);
        }

        let wm_ids: Vec<Uuid> = wm_envelopes.iter().map(|e| e.id).collect();
        let wm_contents = self.store.content.get_batch(&wm_ids).await?;
        let texts: Vec<&str> = wm_contents.iter().map(|c| c.full_text.as_str()).collect();
        let embeddings = self.embed_model.embed_batch(&texts).await?;

        let clusters = agglomerative_cluster(&embeddings, self.config.compaction_cluster_threshold);
        info!(session = session_id, clusters = clusters.len(), entries = wm_envelopes.len(), "Clustered");

        let mut new_engrams = Vec::new();
        for cluster_indices in &clusters {
            let cluster_texts: Vec<&str> = cluster_indices.iter().map(|&i| texts[i]).collect();
            let cluster_ids: Vec<Uuid> = cluster_indices.iter().map(|&i| wm_ids[i]).collect();
            let centroid = self.compute_centroid(
                &cluster_indices.iter().map(|&i| &embeddings[i]).collect::<Vec<_>>(),
            );

            let existing_query = MemoryQuery {
                embedding: centroid.clone(),
                top_k: 3,
                active_only: true,
                memory_type: Some(MemoryType::Semantic),
                min_confidence: Some(0.1),
                ..Default::default()
            };
            let existing = self.store.search(&existing_query).await?;

            if let Some(best) = existing.first() {
                if best.similarity > 0.80 {
                    info!(existing_id = %best.envelope.id, sim = best.similarity, "Merging into existing");
                    let evolved = self
                        .evolve_with_new_evidence(&best.envelope, &cluster_texts, &cluster_ids)
                        .await?;
                    new_engrams.push(evolved);
                    continue;
                }
            }

            let engram = self
                .synthesize_cluster(&cluster_texts, &cluster_ids, &centroid, session_id)
                .await?;
            new_engrams.push(engram);
        }

        for engram in &new_engrams {
            self.store.insert(engram).await?;
            for old_id in &engram.envelope.supersedes {
                self.store
                    .envelopes
                    .mark_superseded(*old_id, engram.envelope.id)
                    .await?;
            }
        }

        info!(session = session_id, new = new_engrams.len(), "Compaction complete");
        Ok(new_engrams)
    }

    async fn synthesize_cluster(
        &self,
        texts: &[&str],
        source_ids: &[Uuid],
        _centroid: &EmbeddingVec,
        session_id: &str,
    ) -> Result<Engram, ConsolidateError> {
        let entries_block = texts
            .iter()
            .enumerate()
            .map(|(i, t)| format!("<entry index=\"{}\">{}</entry>", i, t))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"You are a memory consolidation engine. Distill these working memory entries
into a single semantic knowledge statement.

<working_memory_entries>
{entries_block}
</working_memory_entries>

Rules:
1. Produce a knowledge statement, not a conversation summary.
2. Strip conversational context and session-specific details.
3. Preserve important conditions (e.g., "prefers X for Y context").

Respond in JSON:
{{"full_text": "...", "summary": "1-sentence digest (max 30 words)", "tags": ["..."], "confidence": 0.0-1.0}}"#
        );

        let response = self
            .llm
            .complete(&prompt)
            .await
            .map_err(|e| ConsolidateError::LLM(e.to_string()))?;
        let parsed: serde_json::Value = serde_json::from_str(&response)
            .map_err(|e| ConsolidateError::Parse(e.to_string()))?;

        let id = Uuid::new_v4();
        let now = Utc::now();
        let full_text = parsed["full_text"]
            .as_str()
            .unwrap_or("consolidation failed")
            .to_string();
        let embedding = self.embed_model.embed(&full_text).await?;

        Ok(Engram {
            envelope: Envelope {
                id,
                embedding,
                confidence: parsed["confidence"].as_f64().unwrap_or(0.7) as f32,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                access_count: 0,
                memory_type: MemoryType::Semantic,
                source_sessions: vec![session_id.to_string()],
                supersedes: source_ids.to_vec(),
                superseded_by: None,
                summary: parsed["summary"]
                    .as_str()
                    .unwrap_or(&full_text[..full_text.len().min(100)])
                    .to_string(),
                tags: parsed["tags"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
                // FIX #11: compute actual hash instead of hardcoding 0
                content_hash: seahash_str(&full_text),
            },
            content: ContentBody {
                engram_id: id,
                full_text,
                provenance: texts
                    .iter()
                    .map(|t| ProvenanceRecord {
                        session_id: session_id.to_string(),
                        turn_id: None,
                        timestamp: now,
                        raw_excerpt: t.to_string(),
                    })
                    .collect(),
                conflict_log: vec![],
                related: vec![],
                version: 1,
            },
        })
    }

    // ═══════════════════════════════════════════════════════════
    // OPERATION 2: Evolution / Reconsolidation
    // FIX #3: callers should wrap this in tokio::spawn so it never
    //         blocks request handlers
    // ═══════════════════════════════════════════════════════════

    pub async fn reconsolidate(
        &self,
        retrieved: &[RetrievalResult],
        current_context: &str,
    ) -> Result<Vec<DriftCheck>, ConsolidateError> {
        let context_embedding = self.embed_model.embed(current_context).await?;
        let mut drift_checks = Vec::new();

        for result in retrieved {
            let envelope = &result.envelope;
            let mut drift = DriftCheck::compute(
                &envelope.embedding,
                &context_embedding,
                self.config.evolution_drift_threshold,
            );
            drift.engram_id = envelope.id;

            if drift.needs_evolution {
                info!(engram_id = %envelope.id, drift = drift.drift_score, "Drift detected");
                let content = self.store.content.get(envelope.id).await?;
                let evolved = self
                    .evaluate_evolution(envelope, &content, current_context)
                    .await?;

                if let Some(new_engram) = evolved {
                    self.store.insert(&new_engram).await?;
                    self.store
                        .envelopes
                        .mark_superseded(envelope.id, new_engram.envelope.id)
                        .await?;
                    info!(old = %envelope.id, new = %new_engram.envelope.id, "Evolved");
                }
            }

            let new_confidence = self.compute_reinforced_confidence(
                envelope.confidence,
                result.similarity,
                envelope.access_count,
            );
            self.store.envelopes.touch(envelope.id, new_confidence).await?;
            drift_checks.push(drift);
        }
        Ok(drift_checks)
    }

    async fn evaluate_evolution(
        &self,
        envelope: &Envelope,
        content: &ContentBody,
        current_context: &str,
    ) -> Result<Option<Engram>, ConsolidateError> {
        let prompt = format!(
            r#"You are a memory reconsolidation engine. A stored memory was retrieved and
the current context suggests it may need updating.

<stored_memory>
  <summary>{}</summary>
  <full_text>{}</full_text>
  <confidence>{}</confidence>
  <version>{}</version>
</stored_memory>

<current_context>{}</current_context>

Decision criteria:
- KEEP: memory is still accurate, context adds nothing new.
- UPDATE: context reveals new information that refines or extends the memory.
- CONFLICT: context directly contradicts the memory.

Respond in JSON:
{{"decision": "keep"|"update"|"conflict", "reasoning": "...",
  "updated_text": "... (only if update)", "updated_summary": "... (only if update)",
  "confidence_adjustment": -0.2 to +0.2}}"#,
            envelope.summary,
            content.full_text,
            envelope.confidence,
            content.version,
            current_context,
        );

        let response = self
            .llm
            .complete(&prompt)
            .await
            .map_err(|e| ConsolidateError::LLM(e.to_string()))?;
        let parsed: serde_json::Value = serde_json::from_str(&response)
            .map_err(|e| ConsolidateError::Parse(e.to_string()))?;

        match parsed["decision"].as_str().unwrap_or("keep") {
            "update" => {
                let new_text = parsed["updated_text"]
                    .as_str()
                    .unwrap_or(&content.full_text)
                    .to_string();
                let new_summary = parsed["updated_summary"]
                    .as_str()
                    .unwrap_or(&envelope.summary)
                    .to_string();
                let conf_adj = parsed["confidence_adjustment"]
                    .as_f64()
                    .unwrap_or(0.0) as f32;
                let new_confidence = (envelope.confidence + conf_adj).clamp(0.0, 1.0);
                let new_embedding = self.embed_model.embed(&new_text).await?;
                let id = Uuid::new_v4();
                let now = Utc::now();

                // FIX #16: read actual version from content store instead of hardcoding 1
                let new_version = content.version + 1;

                Ok(Some(Engram {
                    envelope: Envelope {
                        id,
                        embedding: new_embedding,
                        confidence: new_confidence,
                        created_at: envelope.created_at,
                        updated_at: now,
                        last_accessed_at: now,
                        access_count: envelope.access_count,
                        memory_type: MemoryType::Semantic,
                        source_sessions: envelope.source_sessions.clone(),
                        supersedes: vec![envelope.id],
                        superseded_by: None,
                        summary: new_summary,
                        tags: envelope.tags.clone(),
                        content_hash: seahash_str(&new_text),
                    },
                    content: ContentBody {
                        engram_id: id,
                        full_text: new_text,
                        provenance: {
                            let mut p = content.provenance.clone();
                            p.push(ProvenanceRecord {
                                session_id: "reconsolidation".to_string(),
                                turn_id: None,
                                timestamp: now,
                                raw_excerpt: current_context.to_string(),
                            });
                            p
                        },
                        conflict_log: content.conflict_log.clone(),
                        related: content.related.clone(),
                        version: new_version,
                    },
                }))
            }
            "conflict" => {
                warn!(engram_id = %envelope.id, "Conflict detected during reconsolidation");
                // For now: keep memory, reduce confidence
                let new_confidence =
                    (envelope.confidence - 0.1).clamp(0.0, 1.0);
                self.store
                    .envelopes
                    .touch(envelope.id, new_confidence)
                    .await?;
                Ok(None)
            }
            _ => Ok(None), // "keep"
        }
    }

    async fn evolve_with_new_evidence(
        &self,
        existing: &Envelope,
        new_texts: &[&str],
        new_ids: &[Uuid],
    ) -> Result<Engram, ConsolidateError> {
        let existing_content = self.store.content.get(existing.id).await?;

        let entries_block = new_texts
            .iter()
            .enumerate()
            .map(|(i, t)| format!("<entry index=\"{}\">{}</entry>", i, t))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"You are a memory evolution engine. An existing semantic memory is being
updated with new evidence from working memory.

<existing_memory>
  <summary>{}</summary>
  <full_text>{}</full_text>
</existing_memory>

<new_evidence>
{entries_block}
</new_evidence>

Synthesize a single updated memory that integrates both.

Respond in JSON:
{{"full_text": "...", "summary": "max 30 words", "confidence": 0.0-1.0}}"#,
            existing.summary, existing_content.full_text,
        );

        let response = self
            .llm
            .complete(&prompt)
            .await
            .map_err(|e| ConsolidateError::LLM(e.to_string()))?;
        let parsed: serde_json::Value = serde_json::from_str(&response)
            .map_err(|e| ConsolidateError::Parse(e.to_string()))?;

        let full_text = parsed["full_text"]
            .as_str()
            .unwrap_or("evolution failed")
            .to_string();
        let embedding = self.embed_model.embed(&full_text).await?;
        let id = Uuid::new_v4();
        let now = Utc::now();

        let mut supersedes = vec![existing.id];
        supersedes.extend_from_slice(new_ids);

        Ok(Engram {
            envelope: Envelope {
                id,
                embedding,
                confidence: parsed["confidence"].as_f64().unwrap_or(0.8) as f32,
                created_at: existing.created_at,
                updated_at: now,
                last_accessed_at: now,
                access_count: existing.access_count,
                memory_type: MemoryType::Semantic,
                source_sessions: existing.source_sessions.clone(),
                supersedes,
                superseded_by: None,
                summary: parsed["summary"]
                    .as_str()
                    .unwrap_or(&full_text[..full_text.len().min(100)])
                    .to_string(),
                tags: existing.tags.clone(),
                content_hash: seahash_str(&full_text),
            },
            content: ContentBody {
                engram_id: id,
                full_text,
                provenance: {
                    let mut p = existing_content.provenance.clone();
                    p.extend(new_texts.iter().map(|t| ProvenanceRecord {
                        session_id: "compaction".into(),
                        turn_id: None,
                        timestamp: now,
                        raw_excerpt: t.to_string(),
                    }));
                    p
                },
                conflict_log: existing_content.conflict_log.clone(),
                related: existing_content.related.clone(),
                version: existing_content.version + 1,
            },
        })
    }

    // ═══════════════════════════════════════════════════════════
    // Utilities
    // ═══════════════════════════════════════════════════════════

    fn compute_centroid(&self, vecs: &[&EmbeddingVec]) -> EmbeddingVec {
        let dim = vecs[0].dim();
        let mut centroid = vec![0.0f32; dim];
        for v in vecs {
            for (i, val) in v.0.iter().enumerate() {
                centroid[i] += val;
            }
        }
        let n = vecs.len() as f32;
        for val in &mut centroid {
            *val /= n;
        }
        EmbeddingVec(centroid)
    }

    fn compute_reinforced_confidence(
        &self,
        current: f32,
        similarity: f32,
        access_count: u64,
    ) -> f32 {
        let reinforcement = (similarity - 0.5) * 0.1;
        let diminishing = 1.0 / (1.0 + access_count as f32 * 0.01);
        (current + reinforcement * diminishing).clamp(0.0, 1.0)
    }
}