//! In-memory envelope index using brute-force cosine similarity.
//!
//! No external dependencies. Good for development, testing,
//! and small-scale deployments (<10K engrams).
//!
//! For production at scale, use the Qdrant backend instead.

use async_trait::async_trait;
use chrono::Utc;
use mneme_core::*;
use std::collections::HashMap;
use std::sync::RwLock;
use uuid::Uuid;

use crate::{EnvelopeIndex, StoreError};

pub struct InMemoryEnvelopeIndex {
    envelopes: RwLock<HashMap<Uuid, Envelope>>,
}

impl InMemoryEnvelopeIndex {
    pub fn new() -> Self {
        Self {
            envelopes: RwLock::new(HashMap::new()),
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
        store.insert(envelope.id, envelope.clone());
        Ok(())
    }

    async fn search(&self, query: &MemoryQuery) -> Result<Vec<RetrievalResult>, StoreError> {
        let store = self.envelopes.read().unwrap();

        let mut results: Vec<RetrievalResult> = store
            .values()
            .filter(|env| {
                // Filter: active only
                if query.active_only && !env.is_active() {
                    return false;
                }
                // Filter: memory type
                if let Some(ref mt) = query.memory_type {
                    if env.memory_type != *mt {
                        return false;
                    }
                }
                // Filter: minimum confidence
                if let Some(min_conf) = query.min_confidence {
                    if env.confidence < min_conf {
                        return false;
                    }
                }
                // Filter: tags (AND semantics)
                if !query.tags.is_empty()
                    && !query.tags.iter().all(|t| env.tags.contains(t))
                {
                    return false;
                }
                true
            })
            .map(|env| {
                let similarity = env.embedding.cosine_similarity(&query.embedding);
                let retrieval_score =
                    env.retrieval_score(similarity, query.recency_weight);
                RetrievalResult {
                    envelope: env.clone(),
                    similarity,
                    retrieval_score,
                }
            })
            .collect();

        // Sort by retrieval score descending
        results.sort_by(|a, b| {
            b.retrieval_score
                .partial_cmp(&a.retrieval_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top_k
        results.truncate(query.top_k);
        Ok(results)
    }

    async fn get(&self, id: Uuid) -> Result<Envelope, StoreError> {
        let store = self.envelopes.read().unwrap();
        store
            .get(&id)
            .cloned()
            .ok_or(StoreError::NotFound(id))
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

    async fn list_working_memory(
        &self,
        session_id: &str,
    ) -> Result<Vec<Envelope>, StoreError> {
        let store = self.envelopes.read().unwrap();
        let results: Vec<Envelope> = store
            .values()
            .filter(|env| {
                env.memory_type == MemoryType::Working
                    && env.is_active()
                    && env.source_sessions.iter().any(|s| s == session_id)
            })
            .cloned()
            .collect();
        Ok(results)
    }

    async fn mark_superseded(&self, id: Uuid, successor: Uuid) -> Result<(), StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let env = store
            .get_mut(&id)
            .ok_or(StoreError::NotFound(id))?;
        env.superseded_by = Some(successor);
        env.updated_at = Utc::now();
        Ok(())
    }

    async fn gc(
        &self,
        confidence_floor: f32,
        older_than_hours: u64,
    ) -> Result<usize, StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let now = Utc::now();
        let cutoff = now - chrono::Duration::hours(older_than_hours as i64);

        let to_remove: Vec<Uuid> = store
            .values()
            .filter(|env| {
                env.confidence < confidence_floor
                    && env.last_accessed_at < cutoff
                    && env.superseded_by.is_some() // only GC superseded entries
            })
            .map(|env| env.id)
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            store.remove(&id);
        }
        Ok(count)
    }

    async fn touch(&self, id: Uuid, new_confidence: f32) -> Result<(), StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let env = store
            .get_mut(&id)
            .ok_or(StoreError::NotFound(id))?;
        env.last_accessed_at = Utc::now();
        env.access_count += 1;
        env.confidence = new_confidence;
        Ok(())
    }
}
