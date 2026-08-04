//! In-memory entity-relation graph over engram-derived triplets.
//!
//! Facts extracted from compacted engrams (subject, relation, object) let
//! multi-hop recall traverse from an entity named in the question to a
//! connected fact, instead of relying solely on each sub-query's vector
//! similarity happening to surface every piece independently.

use async_trait::async_trait;
use mneme_core::GraphTriplet;
use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, RwLock};

use crate::StoreError;

#[async_trait]
pub trait GraphIndex: Send + Sync {
    async fn insert(&self, triplets: Vec<GraphTriplet>) -> Result<(), StoreError>;

    /// Breadth-first traversal from `entity` (case-insensitive match against
    /// subject or object) out to `hops` edges. Returns every triplet touched,
    /// deduplicated.
    async fn neighbors(&self, entity: &str, hops: usize) -> Result<Vec<GraphTriplet>, StoreError>;

    /// Every triplet currently held, for snapshotting. The graph is derived
    /// data, but re-deriving it means re-running LLM extraction over the whole
    /// corpus, so it's cheaper to persist than to rebuild after a restart.
    async fn all(&self) -> Result<Vec<GraphTriplet>, StoreError>;
}

#[derive(Clone, Default)]
pub struct InMemoryGraphIndex {
    triplets: Arc<RwLock<Vec<GraphTriplet>>>,
}

impl InMemoryGraphIndex {
    pub fn new() -> Self {
        Self::default()
    }

    fn matches(t: &GraphTriplet, needle: &str) -> Option<String> {
        if t.subject.to_lowercase().contains(needle) {
            return Some(t.object.clone());
        }
        if t.object.to_lowercase().contains(needle) {
            return Some(t.subject.clone());
        }
        None
    }
}

#[async_trait]
impl GraphIndex for InMemoryGraphIndex {
    async fn insert(&self, triplets: Vec<GraphTriplet>) -> Result<(), StoreError> {
        let mut store = self
            .triplets
            .write()
            .map_err(|_| StoreError::DocumentStore("graph lock poisoned".into()))?;
        store.extend(triplets);
        Ok(())
    }

    async fn all(&self) -> Result<Vec<GraphTriplet>, StoreError> {
        let store = self
            .triplets
            .read()
            .map_err(|_| StoreError::DocumentStore("graph lock poisoned".into()))?;
        Ok(store.clone())
    }

    async fn neighbors(&self, entity: &str, hops: usize) -> Result<Vec<GraphTriplet>, StoreError> {
        let store = self
            .triplets
            .read()
            .map_err(|_| StoreError::DocumentStore("graph lock poisoned".into()))?;

        let needle = entity.to_lowercase();
        let mut frontier: VecDeque<String> = VecDeque::from([needle.clone()]);
        let mut visited_entities: HashSet<String> = HashSet::from([needle]);
        let mut visited_triplet_ids: HashSet<(String, String, String)> = HashSet::new();
        let mut out = Vec::new();

        for _ in 0..hops.max(1) {
            let mut next_frontier = VecDeque::new();
            while let Some(current) = frontier.pop_front() {
                for t in store.iter() {
                    if let Some(other) = Self::matches(t, &current) {
                        let key = (t.subject.clone(), t.relation.clone(), t.object.clone());
                        if visited_triplet_ids.insert(key) {
                            out.push(t.clone());
                        }
                        let other_lower = other.to_lowercase();
                        if visited_entities.insert(other_lower.clone()) {
                            next_frontier.push_back(other_lower);
                        }
                    }
                }
            }
            frontier = next_frontier;
            if frontier.is_empty() {
                break;
            }
        }

        Ok(out)
    }
}
