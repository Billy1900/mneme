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
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

pub mod backends;
pub use backends::{AnthropicLLM, DeepSeekLLM, MockLLM, OllamaLLM, OpenAILLM};

#[async_trait]
pub trait ConsolidationLLM: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<String, ConsolidateError>;
}

#[async_trait]
impl ConsolidationLLM for Box<dyn ConsolidationLLM> {
    async fn complete(&self, prompt: &str) -> Result<String, ConsolidateError> {
        (**self).complete(prompt).await
    }
}

// Lets `Arc<dyn ConsolidationLLM>` itself satisfy the trait, so callers
// (e.g. mneme-server) can pick a concrete LLM backend at runtime behind one
// trait object — same pattern as `EmbeddingModel`'s `Arc<dyn EmbeddingModel>`
// impl in mneme-embed.
#[async_trait]
impl ConsolidationLLM for std::sync::Arc<dyn ConsolidationLLM> {
    async fn complete(&self, prompt: &str) -> Result<String, ConsolidateError> {
        (**self).complete(prompt).await
    }
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
    /// Entity-relation graph built from compacted engram text. Behind a
    /// trait object (same pattern as `Arc<dyn EmbeddingModel>`) so it isn't
    /// a 5th generic parameter threaded through every caller.
    pub graph: Arc<dyn GraphIndex>,
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
        Self {
            store,
            graph: Arc::new(InMemoryGraphIndex::new()),
            embed_model,
            llm,
            config,
        }
    }

    pub fn with_graph(
        store: MnemeStore<E, C>,
        embed_model: M,
        llm: L,
        config: MnemeConfig,
        graph: Arc<dyn GraphIndex>,
    ) -> Self {
        Self {
            store,
            graph,
            embed_model,
            llm,
            config,
        }
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
        info!(
            session = session_id,
            clusters = clusters.len(),
            entries = wm_envelopes.len(),
            "Clustered"
        );

        let mut new_engrams = Vec::new();
        for cluster_indices in &clusters {
            let cluster_texts: Vec<&str> = cluster_indices.iter().map(|&i| texts[i]).collect();
            let cluster_ids: Vec<Uuid> = cluster_indices.iter().map(|&i| wm_ids[i]).collect();
            let centroid = self.compute_centroid(
                &cluster_indices
                    .iter()
                    .map(|&i| &embeddings[i])
                    .collect::<Vec<_>>(),
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
                    match self
                        .evolve_with_new_evidence(&best.envelope, &cluster_texts, &cluster_ids)
                        .await
                    {
                        Ok(evolved) => {
                            new_engrams.push(evolved);
                            continue;
                        }
                        Err(e) => {
                            warn!("evolve_with_new_evidence failed, synthesizing fresh: {e}");
                        }
                    }
                }
            }

            // Carry forward every source envelope's tags (deduplicated) —
            // this is what `uid:{user_id}` multi-tenant isolation and any
            // other tag-based filtering (search, tests) rides on. Without
            // this, a synthesized engram silently drops out of every
            // tag-filtered search the moment compaction touches it.
            let mut cluster_tags: Vec<String> = cluster_indices
                .iter()
                .flat_map(|&i| wm_envelopes[i].tags.iter().cloned())
                .collect();
            cluster_tags.sort();
            cluster_tags.dedup();

            let engram = match self
                .synthesize_cluster(
                    &cluster_texts,
                    &cluster_ids,
                    &centroid,
                    session_id,
                    &cluster_tags,
                )
                .await
            {
                Ok(e) => e,
                Err(e) => {
                    warn!("synthesize_cluster failed, skipping cluster: {e}");
                    continue;
                }
            };
            new_engrams.push(engram);
        }

        // Link every engram synthesized from this compaction batch to its
        // batch-mates. Facts distilled from the same session tend to be the
        // ones a multi-hop question needs combined (e.g. "who" + "when"
        // clustered separately but co-occurring in one conversation) — this
        // gives recall a graph hop to follow instead of relying solely on
        // each sub-query's vector similarity happening to surface both.
        if new_engrams.len() > 1 {
            let batch_ids: Vec<Uuid> = new_engrams.iter().map(|e| e.envelope.id).collect();
            for engram in &mut new_engrams {
                engram.content.related = batch_ids
                    .iter()
                    .filter(|&&id| id != engram.envelope.id)
                    .map(|&id| RelatedEngram {
                        id,
                        relationship: RelationType::Related,
                        strength: 0.5,
                    })
                    .collect();
            }
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

        // Extract entity-relation triplets from each newly synthesized
        // engram and add them to the graph, so multi-hop recall can
        // traverse from an entity named in the question to a connected fact
        // even when that fact didn't score high enough on its own to be a
        // vector-similarity hit. Best-effort: extraction failures don't
        // fail compaction, they just mean that engram stays graph-invisible.
        for engram in &new_engrams {
            match self
                .extract_triplets(&engram.content.full_text, engram.envelope.id)
                .await
            {
                Ok(triplets) if !triplets.is_empty() => {
                    if let Err(e) = self.graph.insert(triplets).await {
                        warn!("graph insert failed: {e}");
                    }
                }
                Ok(_) => {}
                Err(e) => warn!("triplet extraction failed: {e}"),
            }
        }

        info!(
            session = session_id,
            new = new_engrams.len(),
            "Compaction complete"
        );
        Ok(new_engrams)
    }

    async fn synthesize_cluster(
        &self,
        texts: &[&str],
        source_ids: &[Uuid],
        centroid: &EmbeddingVec,
        session_id: &str,
        carried_tags: &[String],
    ) -> Result<Engram, ConsolidateError> {
        // Fast path: single-element clusters don't need LLM synthesis.
        // Just store the raw text directly as a semantic engram.
        if texts.len() == 1 {
            let id = Uuid::new_v4();
            let now = Utc::now();
            let full_text = texts[0].to_string();
            let summary = full_text.clone();
            return Ok(Engram {
                envelope: Envelope {
                    id,
                    embedding: centroid.clone(),
                    confidence: 0.6,
                    created_at: now,
                    updated_at: now,
                    last_accessed_at: now,
                    access_count: 0,
                    memory_type: MemoryType::Semantic,
                    source_sessions: vec![session_id.to_string()],
                    supersedes: source_ids.to_vec(),
                    superseded_by: None,
                    summary,
                    tags: carried_tags.to_vec(),
                    content_hash: seahash_str(&full_text),
                },
                content: ContentBody {
                    engram_id: id,
                    full_text,
                    provenance: vec![ProvenanceRecord {
                        session_id: session_id.to_string(),
                        turn_id: None,
                        timestamp: now,
                        raw_excerpt: texts[0].to_string(),
                    }],
                    conflict_log: vec![],
                    related: vec![],
                    version: 1,
                },
            });
        }

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
2. Strip conversational filler but KEEP all factual content.
3. Preserve important conditions (e.g., "prefers X for Y context").

Respond in JSON:
{{"full_text": "...", "summary": "1-sentence digest (max 30 words)", "tags": ["..."], "confidence": 0.0-1.0}}"#
        );

        let response = self
            .llm
            .complete(&prompt)
            .await
            .map_err(|e| ConsolidateError::LLM(e.to_string()))?;

        // Parse with fallback: if LLM returns unparseable content, use the raw texts
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap_or_else(|e| {
            warn!("synthesize_cluster: LLM response not valid JSON ({e}), using raw text fallback");
            serde_json::Value::Null
        });

        let id = Uuid::new_v4();
        let now = Utc::now();
        let full_text = parsed["full_text"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| texts.join(" | "));
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
                    .filter(|s| !s.is_empty())
                    .unwrap_or(&full_text)
                    .to_string(),
                tags: {
                    // Union of carried-forward tags (isolation, etc. — must
                    // survive compaction regardless of what the LLM does)
                    // and whatever descriptive tags the LLM suggested.
                    let mut tags = carried_tags.to_vec();
                    if let Some(llm_tags) = parsed["tags"].as_array() {
                        tags.extend(llm_tags.iter().filter_map(|v| v.as_str().map(String::from)));
                    }
                    tags.sort();
                    tags.dedup();
                    tags
                },
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

    /// Extract (subject, relation, object, date) triplets from a synthesized
    /// engram's text via the LLM. Cheap, targeted extraction — not a full
    /// NER/RE pipeline — deliberately kept to explicit factual relations
    /// rather than every noun pair.
    async fn extract_triplets(
        &self,
        text: &str,
        source_engram_id: Uuid,
    ) -> Result<Vec<GraphTriplet>, ConsolidateError> {
        let prompt = format!(
            r#"Extract factual (subject, relation, object) triplets from this text.

Text:
{text}

Rules:
1. Only extract explicit, factual relations (e.g. "works at", "lives in", "is married to", "prefers", "happened on").
2. Subject and object should be short entity names or noun phrases, not full sentences.
3. If the text carries a date/time this fact pertains to (e.g. a "[8 May, 2023]" prefix), put it in "date"; otherwise null.
4. Return at most 5 triplets. If there are no clear factual relations, return an empty array.

Respond ONLY with a JSON array, e.g.:
[{{"subject": "Alice", "relation": "works at", "object": "Acme Corp", "date": null}}]"#
        );

        let response = self
            .llm
            .complete(&prompt)
            .await
            .map_err(|e| ConsolidateError::LLM(e.to_string()))?;

        let cleaned = response
            .trim()
            .strip_prefix("```json")
            .unwrap_or(response.trim())
            .strip_prefix("```")
            .unwrap_or(response.trim())
            .strip_suffix("```")
            .unwrap_or(response.trim())
            .trim();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(cleaned).unwrap_or_default();

        Ok(parsed
            .into_iter()
            .filter_map(|v| {
                let subject = v["subject"].as_str()?.trim().to_string();
                let relation = v["relation"].as_str()?.trim().to_string();
                let object = v["object"].as_str()?.trim().to_string();
                if subject.is_empty() || relation.is_empty() || object.is_empty() {
                    return None;
                }
                Some(GraphTriplet {
                    subject,
                    relation,
                    object,
                    date: v["date"].as_str().map(String::from),
                    source_engram_id,
                })
            })
            .collect())
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
            self.store
                .envelopes
                .touch(envelope.id, new_confidence)
                .await?;
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
        let parsed: serde_json::Value =
            serde_json::from_str(&response).map_err(|e| ConsolidateError::Parse(e.to_string()))?;

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
                let conf_adj = parsed["confidence_adjustment"].as_f64().unwrap_or(0.0) as f32;
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
                let reasoning = parsed["reasoning"]
                    .as_str()
                    .unwrap_or("no reasoning given")
                    .to_string();

                // Previously this just decayed the stale memory's confidence
                // and left it active, so a contradicted fact kept resurfacing
                // in recall alongside (or instead of) the truth. Now the new
                // context wins: supersede the old engram (excluded from
                // active recall via is_active()) and record what it
                // contradicted, mirroring the "update" evolve/supersede path
                // above instead of a dead-end confidence nudge.
                let decayed = (envelope.confidence * (1.0 - self.config.conflict_loser_decay))
                    .clamp(0.0, 1.0);
                self.store.envelopes.touch(envelope.id, decayed).await?;

                let new_text = current_context.to_string();
                let new_embedding = self.embed_model.embed(&new_text).await?;
                let id = Uuid::new_v4();
                let now = Utc::now();
                let new_summary = if new_text.len() > 100 {
                    let cut = new_text
                        .char_indices()
                        .map(|(i, _)| i)
                        .take_while(|&i| i <= 97)
                        .last()
                        .unwrap_or(0);
                    format!("{}...", &new_text[..cut])
                } else {
                    new_text.clone()
                };

                Ok(Some(Engram {
                    envelope: Envelope {
                        id,
                        embedding: new_embedding,
                        confidence: 0.6,
                        created_at: now,
                        updated_at: now,
                        last_accessed_at: now,
                        access_count: 0,
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
                                session_id: "reconsolidation-conflict".to_string(),
                                turn_id: None,
                                timestamp: now,
                                raw_excerpt: current_context.to_string(),
                            });
                            p
                        },
                        conflict_log: {
                            let mut log = content.conflict_log.clone();
                            log.push(ConflictRecord {
                                conflicting_id: envelope.id,
                                resolution: ConflictStrategy::TemporalSupersede,
                                resolved_at: now,
                                resolver_notes: reasoning,
                            });
                            log
                        },
                        related: content.related.clone(),
                        version: content.version + 1,
                    },
                }))
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
            r#"You are a memory consolidation engine. Merge the existing memory with new evidence into one updated memory.

Existing memory:
{existing_full}

New evidence to integrate:
{entries_block}

Output a JSON object with exactly these keys:
- "full_text": the complete merged memory (preserve all specific facts, names, dates)
- "summary": one sentence, max 30 words
- "confidence": float 0.0-1.0

JSON:"#,
            existing_full = existing_content.full_text,
        );

        let response = self
            .llm
            .complete(&prompt)
            .await
            .map_err(|e| ConsolidateError::LLM(e.to_string()))?;

        let parsed: serde_json::Value =
            serde_json::from_str(&response).unwrap_or_else(|_| serde_json::Value::Null);

        let full_text = parsed["full_text"]
            .as_str()
            .filter(|s| !s.is_empty())
            .map(String::from)
            .unwrap_or_else(|| {
                // fallback: concatenate existing + new evidence
                let mut parts = vec![existing_content.full_text.as_str()];
                parts.extend_from_slice(new_texts);
                parts.join(" | ")
            });
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
                    .filter(|s| !s.is_empty())
                    .unwrap_or(&full_text)
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
