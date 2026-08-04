//! In-memory envelope index using brute-force cosine similarity.
//!
//! No external dependencies. Good for development, testing,
//! and small-scale deployments (<10K engrams).
//!
//! For production at scale, use the Qdrant backend instead.

use async_trait::async_trait;
use chrono::Utc;
use mneme_core::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

use crate::{EnvelopeIndex, StoreError, StoreStats};

#[derive(Clone)]
pub struct InMemoryEnvelopeIndex {
    envelopes: Arc<RwLock<HashMap<Uuid, Envelope>>>,
    /// tag -> envelope ids with that tag. Lets `search()` narrow to a
    /// candidate set via AND-intersection before scoring, instead of
    /// scanning every envelope in the store on every query. Without this,
    /// a tag filter like `uid:{user_id}` (used for multi-tenant isolation)
    /// only affects *correctness*, not cost — every user's search would
    /// still pay for every other user's data.
    tag_index: Arc<RwLock<HashMap<String, HashSet<Uuid>>>>,
}

impl InMemoryEnvelopeIndex {
    pub fn new() -> Self {
        Self {
            envelopes: Arc::new(RwLock::new(HashMap::new())),
            tag_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn len(&self) -> usize {
        self.envelopes.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for InMemoryEnvelopeIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EnvelopeIndex for InMemoryEnvelopeIndex {
    async fn upsert(&self, envelope: &Envelope) -> Result<(), StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let mut tag_index = self.tag_index.write().unwrap();

        // Re-upserting an existing id can change its tags — drop the old
        // tag associations before adding the new ones, or the index would
        // accumulate stale entries pointing at a now-different tag set.
        if let Some(old) = store.get(&envelope.id) {
            for t in &old.tags {
                if let Some(set) = tag_index.get_mut(t) {
                    set.remove(&envelope.id);
                }
            }
        }
        for t in &envelope.tags {
            tag_index.entry(t.clone()).or_default().insert(envelope.id);
        }

        store.insert(envelope.id, envelope.clone());
        Ok(())
    }

    async fn search(&self, query: &MemoryQuery) -> Result<Vec<RetrievalResult>, StoreError> {
        let store = self.envelopes.read().unwrap();

        // When tags are given, narrow to the AND-intersection of their
        // candidate sets first — this is what makes a tag filter (e.g.
        // `uid:{user_id}`) actually cheap instead of just correct.
        let candidate_ids: Option<Vec<Uuid>> = if query.tags.is_empty() {
            None
        } else {
            let tag_index = self.tag_index.read().unwrap();
            let mut tags = query.tags.iter();
            let mut candidates: HashSet<Uuid> = match tags.next() {
                Some(first) => tag_index.get(first).cloned().unwrap_or_default(),
                None => HashSet::new(),
            };
            for t in tags {
                if candidates.is_empty() {
                    break;
                }
                let set = tag_index.get(t).cloned().unwrap_or_default();
                candidates = candidates.intersection(&set).copied().collect();
            }
            Some(candidates.into_iter().collect())
        };

        let envelopes_iter: Box<dyn Iterator<Item = &Envelope>> = match &candidate_ids {
            Some(ids) => Box::new(ids.iter().filter_map(|id| store.get(id))),
            None => Box::new(store.values()),
        };

        let mut results: Vec<RetrievalResult> = envelopes_iter
            .filter(|env| {
                if query.active_only && !env.is_active() {
                    return false;
                }
                if let Some(ref mt) = query.memory_type {
                    if env.memory_type != *mt {
                        return false;
                    }
                }
                if let Some(min_conf) = query.min_confidence {
                    if env.confidence < min_conf {
                        return false;
                    }
                }
                true
            })
            .map(|env| {
                let similarity = env.embedding.cosine_similarity(&query.embedding);
                let recency = env.time_decay(0.05) as f32;
                let retrieval_score =
                    (1.0 - query.recency_weight) * similarity + query.recency_weight * recency;
                RetrievalResult {
                    envelope: env.clone(),
                    similarity,
                    retrieval_score,
                }
            })
            .collect();

        results.sort_by(|a, b| b.retrieval_score.partial_cmp(&a.retrieval_score).unwrap());
        results.truncate(query.top_k);
        Ok(results)
    }

    async fn get(&self, id: Uuid) -> Result<Envelope, StoreError> {
        let store = self.envelopes.read().unwrap();
        store.get(&id).cloned().ok_or(StoreError::NotFound(id))
    }

    async fn get_batch(&self, ids: &[Uuid]) -> Result<Vec<Envelope>, StoreError> {
        let store = self.envelopes.read().unwrap();
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            match store.get(id) {
                Some(env) => results.push(env.clone()),
                None => return Err(StoreError::NotFound(*id)),
            }
        }
        Ok(results)
    }

    async fn list_working_memory(&self, session_id: &str) -> Result<Vec<Envelope>, StoreError> {
        let store = self.envelopes.read().unwrap();
        let results = store
            .values()
            .filter(|env| {
                env.memory_type == MemoryType::Working
                    && env.source_sessions.iter().any(|s| s == session_id)
            })
            .cloned()
            .collect();
        Ok(results)
    }

    async fn mark_superseded(&self, id: Uuid, successor: Uuid) -> Result<(), StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let env = store.get_mut(&id).ok_or(StoreError::NotFound(id))?;
        env.superseded_by = Some(successor);
        env.updated_at = Utc::now();
        Ok(())
    }

    async fn gc(&self, confidence_floor: f32, older_than_hours: u64) -> Result<usize, StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let cutoff = Utc::now() - chrono::Duration::hours(older_than_hours as i64);
        let to_remove: Vec<Uuid> = store
            .iter()
            .filter(|(_, env)| {
                let too_old = env.memory_type == MemoryType::Working && env.created_at < cutoff;
                let too_low = env.confidence < confidence_floor && !env.is_active();
                too_old || too_low
            })
            .map(|(id, _)| *id)
            .collect();

        let mut tag_index = self.tag_index.write().unwrap();
        for id in &to_remove {
            if let Some(env) = store.remove(id) {
                for t in &env.tags {
                    if let Some(set) = tag_index.get_mut(t) {
                        set.remove(id);
                    }
                }
            }
        }
        Ok(to_remove.len())
    }

    async fn delete(&self, id: Uuid) -> Result<(), StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let env = store.remove(&id).ok_or(StoreError::NotFound(id))?;
        let mut tag_index = self.tag_index.write().unwrap();
        for t in &env.tags {
            if let Some(set) = tag_index.get_mut(t) {
                set.remove(&id);
            }
        }
        Ok(())
    }

    async fn touch(&self, id: Uuid, new_confidence: f32) -> Result<(), StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let env = store.get_mut(&id).ok_or(StoreError::NotFound(id))?;
        env.access_count += 1;
        env.last_accessed_at = Utc::now();
        env.confidence = new_confidence;
        Ok(())
    }

    async fn apply_decay(&self, lambda: f64) -> Result<usize, StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let mut updated = 0;
        for env in store.values_mut() {
            if env.is_active() {
                let new_conf = (env.confidence as f64 * env.time_decay(lambda)) as f32;
                env.confidence = new_conf.max(0.0);
                updated += 1;
            }
        }
        Ok(updated)
    }

    async fn stats(&self) -> Result<StoreStats, StoreError> {
        let store = self.envelopes.read().unwrap();
        let total = store.len();
        let working = store
            .values()
            .filter(|e| e.memory_type == MemoryType::Working)
            .count();
        let semantic = store
            .values()
            .filter(|e| e.memory_type == MemoryType::Semantic)
            .count();
        let superseded = store.values().filter(|e| !e.is_active()).count();
        let avg_confidence = if total == 0 {
            0.0
        } else {
            store.values().map(|e| e.confidence).sum::<f32>() / total as f32
        };
        Ok(StoreStats {
            total_engrams: total,
            working_memory_count: working,
            semantic_memory_count: semantic,
            superseded_count: superseded,
            avg_confidence,
        })
    }
}