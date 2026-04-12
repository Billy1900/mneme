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
    store: MnemeStore<E, C>,
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

fn seahash_str(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

impl<E, C, M, L> ConsolidationEngine<E, C, M, L>
where
    E: EnvelopeIndex,
    C: ContentStore,
    M: EmbeddingModel,
    L: ConsolidationLLM,
{
    pub fn new(store: MnemeStore<E, C>, embed_model: M, llm: L, config: MnemeConfig) -> Self {
        Self { store, embed_model, llm, config }
    }

    // ═══════════════════════════════════════════════════════════
    // OPERATION 1: Compaction (记忆压缩)
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

            // Check if similar semantic engram already exists
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
                    let evolved = self.evolve_with_new_evidence(&best.envelope, &cluster_texts, &cluster_ids).await?;
                    new_engrams.push(evolved);
                    continue;
                }
            }

            let engram = self.synthesize_cluster(&cluster_texts, &cluster_ids, &centroid, session_id).await?;
            new_engrams.push(engram);
        }

        for engram in &new_engrams {
            self.store.insert(engram).await?;
            for old_id in &engram.envelope.supersedes {
                self.store.envelopes.mark_superseded(*old_id, engram.envelope.id).await?;
            }
        }

        info!(session = session_id, new = new_engrams.len(), "Compaction complete");
        Ok(new_engrams)
    }

    async fn synthesize_cluster(
        &self, texts: &[&str], source_ids: &[Uuid], _centroid: &EmbeddingVec, session_id: &str,
    ) -> Result<Engram, ConsolidateError> {
        let entries_block = texts.iter().enumerate()
            .map(|(i, t)| format!("<entry index=\"{}\">{}</entry>", i, t))
            .collect::<Vec<_>>().join("\n");

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

        let response = self.llm.complete(&prompt).await.map_err(|e| ConsolidateError::LLM(e.to_string()))?;
        let parsed: serde_json::Value = serde_json::from_str(&response).map_err(|e| ConsolidateError::Parse(e.to_string()))?;

        let id = Uuid::new_v4();
        let now = Utc::now();
        let full_text = parsed["full_text"].as_str().unwrap_or("consolidation failed").to_string();
        let embedding = self.embed_model.embed(&full_text).await?;

        Ok(Engram {
            envelope: Envelope {
                id, embedding,
                confidence: parsed["confidence"].as_f64().unwrap_or(0.7) as f32,
                created_at: now, updated_at: now, last_accessed_at: now,
                access_count: 0, memory_type: MemoryType::Semantic,
                source_sessions: vec![session_id.to_string()],
                supersedes: source_ids.to_vec(), superseded_by: None,
                summary: parsed["summary"].as_str().unwrap_or(&full_text[..full_text.len().min(100)]).to_string(),
                tags: parsed["tags"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default(),
                content_hash: seahash_str(&full_text),
            },
            content: ContentBody {
                engram_id: id, full_text,
                provenance: texts.iter().map(|t| ProvenanceRecord {
                    session_id: session_id.to_string(), turn_id: None, timestamp: now, raw_excerpt: t.to_string(),
                }).collect(),
                conflict_log: vec![], related: vec![], version: 1,
            },
        })
    }

    // ═══════════════════════════════════════════════════════════
    // OPERATION 2: Evolution (记忆演化 / Reconsolidation)
    // ═══════════════════════════════════════════════════════════

    pub async fn reconsolidate(
        &self, retrieved: &[RetrievalResult], current_context: &str,
    ) -> Result<Vec<DriftCheck>, ConsolidateError> {
        let context_embedding = self.embed_model.embed(current_context).await?;
        let mut drift_checks = Vec::new();

        for result in retrieved {
            let envelope = &result.envelope;
            let mut drift = DriftCheck::compute(
                &envelope.embedding, &context_embedding, self.config.evolution_drift_threshold,
            );
            drift.engram_id = envelope.id;

            if drift.needs_evolution {
                info!(engram_id = %envelope.id, drift = drift.drift_score, "Drift detected");
                let content = self.store.content.get(envelope.id).await?;
                let evolved = self.evaluate_evolution(envelope, &content, current_context).await?;

                if let Some(new_engram) = evolved {
                    self.store.insert(&new_engram).await?;
                    self.store.envelopes.mark_superseded(envelope.id, new_engram.envelope.id).await?;
                    info!(old = %envelope.id, new = %new_engram.envelope.id, "Evolved");
                }
            }

            let new_confidence = self.compute_reinforced_confidence(
                envelope.confidence, result.similarity, envelope.access_count,
            );
            self.store.envelopes.touch(envelope.id, new_confidence).await?;
            drift_checks.push(drift);
        }
        Ok(drift_checks)
    }

    async fn evaluate_evolution(
        &self, envelope: &Envelope, content: &ContentBody, current_context: &str,
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
            envelope.summary, content.full_text, envelope.confidence,
            content.version, current_context,
        );

        let response = self.llm.complete(&prompt).await.map_err(|e| ConsolidateError::LLM(e.to_string()))?;
        let parsed: serde_json::Value = serde_json::from_str(&response).map_err(|e| ConsolidateError::Parse(e.to_string()))?;

        match parsed["decision"].as_str().unwrap_or("keep") {
            "update" => {
                let new_text = parsed["updated_text"].as_str()
                    .ok_or_else(|| ConsolidateError::Parse("missing updated_text".into()))?;
                let new_summary = parsed["updated_summary"].as_str().unwrap_or(&new_text[..new_text.len().min(100)]);
                let new_embedding = self.embed_model.embed(new_text).await?;
                let conf_adj = parsed["confidence_adjustment"].as_f64().unwrap_or(0.0) as f32;
                let new_id = Uuid::new_v4();
                let now = Utc::now();

                Ok(Some(Engram {
                    envelope: Envelope {
                        id: new_id, embedding: new_embedding,
                        confidence: (envelope.confidence + conf_adj).clamp(0.0, 1.0),
                        created_at: envelope.created_at, updated_at: now, last_accessed_at: now,
                        access_count: envelope.access_count + 1,
                        memory_type: MemoryType::Semantic,
                        source_sessions: envelope.source_sessions.clone(),
                        supersedes: vec![envelope.id], superseded_by: None,
                        summary: new_summary.to_string(),
                        tags: envelope.tags.clone(),
                        content_hash: seahash_str(new_text),
                    },
                    content: ContentBody {
                        engram_id: new_id, full_text: new_text.to_string(),
                        provenance: content.provenance.clone(),
                        conflict_log: content.conflict_log.clone(),
                        related: content.related.clone(),
                        version: content.version + 1,
                    },
                }))
            }
            "conflict" => {
                warn!(engram_id = %envelope.id, "Conflict detected during reconsolidation");
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    // ═══════════════════════════════════════════════════════════
    // OPERATION 3: Conflict Resolution (记忆冲突解决)
    // ═══════════════════════════════════════════════════════════

    pub async fn resolve_conflict(
        &self, a: &Engram, b: &Engram,
    ) -> Result<ConflictResolution, ConsolidateError> {
        let score_a = self.evidence_score(&a.envelope);
        let score_b = self.evidence_score(&b.envelope);
        let score_gap = (score_a - score_b).abs();

        let strategy = if score_gap > self.config.conflict_score_gap_threshold {
            ConflictStrategy::TemporalSupersede
        } else if score_a.min(score_b) > 0.7 {
            ConflictStrategy::ConfidenceMerge
        } else {
            self.evaluate_coexistence(a, b).await?
        };

        info!(a = %a.envelope.id, b = %b.envelope.id, ?strategy, "Resolving conflict");

        match strategy {
            ConflictStrategy::TemporalSupersede => {
                let (winner, loser) = if a.envelope.updated_at > b.envelope.updated_at { (a, b) } else { (b, a) };
                Ok(ConflictResolution { strategy, winner_id: winner.envelope.id, loser_id: Some(loser.envelope.id), merged_engram: None })
            }
            ConflictStrategy::ConfidenceMerge => {
                let merged = self.merge_engrams(a, b).await?;
                Ok(ConflictResolution { strategy, winner_id: merged.envelope.id, loser_id: None, merged_engram: Some(merged) })
            }
            ConflictStrategy::ConditionalCoexist => {
                Ok(ConflictResolution { strategy, winner_id: a.envelope.id, loser_id: None, merged_engram: None })
            }
            ConflictStrategy::Escalated => {
                Ok(ConflictResolution { strategy, winner_id: a.envelope.id, loser_id: None, merged_engram: None })
            }
        }
    }

    async fn evaluate_coexistence(&self, a: &Engram, b: &Engram) -> Result<ConflictStrategy, ConsolidateError> {
        let prompt = format!(
r#"Two memories contradict each other. Determine the relationship.

<memory_a>{} (confidence: {}, updated: {})</memory_a>
<memory_b>{} (confidence: {}, updated: {})</memory_b>

1. "factual_update" — B replaces A (e.g., diet changed)
2. "context_dependent" — both true in different contexts
3. "unresolvable" — genuinely contradictory, needs human review

Respond in JSON:
{{"relationship": "factual_update"|"context_dependent"|"unresolvable", "reasoning": "..."}}"#,
            a.content.full_text, a.envelope.confidence, a.envelope.updated_at,
            b.content.full_text, b.envelope.confidence, b.envelope.updated_at,
        );

        let response = self.llm.complete(&prompt).await.map_err(|e| ConsolidateError::LLM(e.to_string()))?;
        let parsed: serde_json::Value = serde_json::from_str(&response).map_err(|e| ConsolidateError::Parse(e.to_string()))?;

        match parsed["relationship"].as_str().unwrap_or("unresolvable") {
            "factual_update" => Ok(ConflictStrategy::TemporalSupersede),
            "context_dependent" => Ok(ConflictStrategy::ConditionalCoexist),
            _ => Ok(ConflictStrategy::Escalated),
        }
    }

    async fn merge_engrams(&self, a: &Engram, b: &Engram) -> Result<Engram, ConsolidateError> {
        let prompt = format!(
r#"Merge two related memories into one comprehensive memory.

<memory_a>{}</memory_a>
<memory_b>{}</memory_b>

Respond in JSON:
{{"full_text": "merged memory", "summary": "1-sentence digest", "confidence": 0.0-1.0}}"#,
            a.content.full_text, b.content.full_text,
        );

        let response = self.llm.complete(&prompt).await.map_err(|e| ConsolidateError::LLM(e.to_string()))?;
        let parsed: serde_json::Value = serde_json::from_str(&response).map_err(|e| ConsolidateError::Parse(e.to_string()))?;
        let full_text = parsed["full_text"].as_str().unwrap_or("merge failed").to_string();
        let embedding = self.embed_model.embed(&full_text).await?;
        let id = Uuid::new_v4();
        let now = Utc::now();

        let mut sessions = a.envelope.source_sessions.clone();
        sessions.extend(b.envelope.source_sessions.clone());
        sessions.sort(); sessions.dedup();
        let mut tags = a.envelope.tags.clone();
        tags.extend(b.envelope.tags.clone());
        tags.sort(); tags.dedup();

        Ok(Engram {
            envelope: Envelope {
                id, embedding,
                confidence: parsed["confidence"].as_f64().unwrap_or(0.8) as f32,
                created_at: a.envelope.created_at.min(b.envelope.created_at),
                updated_at: now, last_accessed_at: now, access_count: 0,
                memory_type: MemoryType::Semantic, source_sessions: sessions,
                supersedes: vec![a.envelope.id, b.envelope.id], superseded_by: None,
                summary: parsed["summary"].as_str().unwrap_or(&full_text[..full_text.len().min(100)]).to_string(),
                tags, content_hash: seahash_str(&full_text),
            },
            content: ContentBody {
                engram_id: id, full_text,
                provenance: { let mut p = a.content.provenance.clone(); p.extend(b.content.provenance.clone()); p },
                conflict_log: vec![ConflictRecord {
                    timestamp: now, loser_id: b.envelope.id,
                    old_value: format!("A: {} | B: {}", a.content.full_text, b.content.full_text),
                    new_value: "merged".to_string(), strategy: ConflictStrategy::ConfidenceMerge, confidence_delta: 0.0,
                }],
                related: vec![], version: 1,
            },
        })
    }

    async fn evolve_with_new_evidence(
        &self, existing: &Envelope, new_texts: &[&str], new_ids: &[Uuid],
    ) -> Result<Engram, ConsolidateError> {
        let existing_content = self.store.content.get(existing.id).await?;
        let evidence = new_texts.join("\n---\n");
        let prompt = format!(
r#"A semantic memory exists and new evidence has arrived.

<existing_memory>{}</existing_memory>
<new_evidence>{}</new_evidence>

Update the memory to incorporate the new evidence. Never lose existing info unless contradicted.

Respond in JSON:
{{"full_text": "...", "summary": "...", "confidence": 0.0-1.0, "change_type": "reinforced"|"extended"|"corrected"}}"#,
            existing_content.full_text, evidence,
        );

        let response = self.llm.complete(&prompt).await.map_err(|e| ConsolidateError::LLM(e.to_string()))?;
        let parsed: serde_json::Value = serde_json::from_str(&response).map_err(|e| ConsolidateError::Parse(e.to_string()))?;
        let full_text = parsed["full_text"].as_str().unwrap_or("evolution failed").to_string();
        let embedding = self.embed_model.embed(&full_text).await?;
        let id = Uuid::new_v4();
        let now = Utc::now();

        let mut supersedes = vec![existing.id];
        supersedes.extend(new_ids);

        Ok(Engram {
            envelope: Envelope {
                id, embedding,
                confidence: parsed["confidence"].as_f64().unwrap_or(0.8) as f32,
                created_at: existing.created_at, updated_at: now, last_accessed_at: now,
                access_count: existing.access_count, memory_type: MemoryType::Semantic,
                source_sessions: existing.source_sessions.clone(),
                supersedes, superseded_by: None,
                summary: parsed["summary"].as_str().unwrap_or(&full_text[..full_text.len().min(100)]).to_string(),
                tags: existing.tags.clone(), content_hash: seahash_str(&full_text),
            },
            content: ContentBody {
                engram_id: id, full_text,
                provenance: { let mut p = existing_content.provenance; p.extend(new_texts.iter().map(|t| ProvenanceRecord {
                    session_id: "compaction".into(), turn_id: None, timestamp: now, raw_excerpt: t.to_string(),
                })); p },
                conflict_log: existing_content.conflict_log,
                related: existing_content.related,
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
        for v in vecs { for (i, val) in v.0.iter().enumerate() { centroid[i] += val; } }
        let n = vecs.len() as f32;
        for val in &mut centroid { *val /= n; }
        EmbeddingVec(centroid)
    }

    fn evidence_score(&self, env: &Envelope) -> f32 {
        let recency = env.time_decay(self.config.decay_lambda) as f32;
        let frequency = (env.access_count as f32).ln_1p() / 10.0;
        let provenance = (env.source_sessions.len() as f32).ln_1p() / 5.0;
        env.confidence * 0.4 + recency * 0.3 + frequency * 0.15 + provenance * 0.15
    }

    fn compute_reinforced_confidence(&self, current: f32, similarity: f32, access_count: u64) -> f32 {
        let reinforcement = (similarity - 0.5) * 0.1;
        let diminishing = 1.0 / (1.0 + access_count as f32 * 0.01);
        (current + reinforcement * diminishing).clamp(0.0, 1.0)
    }
}
