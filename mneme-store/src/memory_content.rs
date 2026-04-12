//! In-memory content store using HashMap.
//!
//! Pairs with InMemoryEnvelopeIndex for a fully in-memory
//! development setup. No persistence — data is lost on restart.

use async_trait::async_trait;
use mneme_core::*;
use std::collections::HashMap;
use std::sync::RwLock;
use uuid::Uuid;

use crate::{ContentStore, StoreError};

pub struct InMemoryContentStore {
    bodies: RwLock<HashMap<Uuid, ContentBody>>,
}

impl InMemoryContentStore {
    pub fn new() -> Self {
        Self {
            bodies: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryContentStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContentStore for InMemoryContentStore {
    async fn put(&self, content: &ContentBody) -> Result<(), StoreError> {
        let mut store = self.bodies.write().unwrap();
        store.insert(content.engram_id, content.clone());
        Ok(())
    }

    async fn get(&self, engram_id: Uuid) -> Result<ContentBody, StoreError> {
        let store = self.bodies.read().unwrap();
        store
            .get(&engram_id)
            .cloned()
            .ok_or(StoreError::NotFound(engram_id))
    }

    async fn get_batch(&self, ids: &[Uuid]) -> Result<Vec<ContentBody>, StoreError> {
        let store = self.bodies.read().unwrap();
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            match store.get(id) {
                Some(body) => results.push(body.clone()),
                None => return Err(StoreError::NotFound(*id)),
            }
        }
        Ok(results)
    }

    async fn delete(&self, engram_id: Uuid) -> Result<(), StoreError> {
        let mut store = self.bodies.write().unwrap();
        store.remove(&engram_id);
        Ok(())
    }

    async fn append_conflict(
        &self,
        engram_id: Uuid,
        record: ConflictRecord,
    ) -> Result<(), StoreError> {
        let mut store = self.bodies.write().unwrap();
        let body = store
            .get_mut(&engram_id)
            .ok_or(StoreError::NotFound(engram_id))?;
        body.conflict_log.push(record);
        Ok(())
    }
}
