//! # mneme-store
//!
//! Storage backends for the Mneme memory system.
//!
//! ## Backend matrix
//!
//! | Backend | EnvelopeIndex | ContentStore | Use case |
//! |---------|---------------|--------------|----------|
//! | `memory` | InMemoryEnvelopeIndex | InMemoryContentStore | Dev/testing |
//! | `sqlite` | SqliteEnvelopeIndex | SqliteContentStore | Single-node prod |
//! | `qdrant` | QdrantEnvelopeIndex | (use sqlite content) | Scaled prod (ANN) |

use async_trait::async_trait;
use mneme_core::*;
use std::sync::Arc;
use uuid::Uuid;

pub mod memory;
pub mod memory_content;

#[cfg(feature = "sqlite")]
pub mod sqlite_envelope;
#[cfg(feature = "sqlite")]
pub mod sqlite_content;

pub use memory::InMemoryEnvelopeIndex;
pub use memory_content::InMemoryContentStore;

#[cfg(feature = "sqlite")]
pub use sqlite_envelope::SqliteEnvelopeIndex;
#[cfg(feature = "sqlite")]
pub use sqlite_content::SqliteContentStore;

// ─────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("engram not found: {0}")]
    NotFound(Uuid),
    #[error("vector index error: {0}")]
    VectorIndex(String),
    #[error("document store error: {0}")]
    DocumentStore(String),
    #[error("serialization error: {0}")]
    Serialization(String),
}

// ─────────────────────────────────────────────────────────────
// Traits
// ─────────────────────────────────────────────────────────────

#[async_trait]
pub trait EnvelopeIndex: Send + Sync {
    async fn upsert(&self, envelope: &Envelope) -> Result<(), StoreError>;
    async fn search(&self, query: &MemoryQuery) -> Result<Vec<RetrievalResult>, StoreError>;
    async fn get(&self, id: Uuid) -> Result<Envelope, StoreError>;
    async fn get_batch(&self, ids: &[Uuid]) -> Result<Vec<Envelope>, StoreError>;
    async fn list_working_memory(&self, session_id: &str) -> Result<Vec<Envelope>, StoreError>;
    async fn mark_superseded(&self, id: Uuid, successor: Uuid) -> Result<(), StoreError>;
    async fn gc(&self, confidence_floor: f32, older_than_hours: u64) -> Result<usize, StoreError>;
    async fn touch(&self, id: Uuid, new_confidence: f32) -> Result<(), StoreError>;
    async fn stats(&self) -> Result<StoreStats, StoreError>;
}

#[async_trait]
pub trait ContentStore: Send + Sync {
    async fn put(&self, content: &ContentBody) -> Result<(), StoreError>;
    async fn get(&self, engram_id: Uuid) -> Result<ContentBody, StoreError>;
    async fn get_batch(&self, ids: &[Uuid]) -> Result<Vec<ContentBody>, StoreError>;
    async fn delete(&self, engram_id: Uuid) -> Result<(), StoreError>;
    async fn append_conflict(&self, engram_id: Uuid, record: ConflictRecord) -> Result<(), StoreError>;
}

// ─────────────────────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct StoreStats {
    pub total_engrams: usize,
    pub working_memory_count: usize,
    pub semantic_memory_count: usize,
    pub superseded_count: usize,
    pub avg_confidence: f32,
}

// ─────────────────────────────────────────────────────────────
// Combined store facade
// ─────────────────────────────────────────────────────────────

pub struct MnemeStore<E: EnvelopeIndex, C: ContentStore> {
    pub envelopes: E,
    pub content: C,
}

impl<E: EnvelopeIndex, C: ContentStore> MnemeStore<E, C> {
    pub fn new(envelopes: E, content: C) -> Self {
        Self { envelopes, content }
    }

    pub async fn insert(&self, engram: &Engram) -> Result<(), StoreError> {
        self.envelopes.upsert(&engram.envelope).await?;
        self.content.put(&engram.content).await?;
        Ok(())
    }

    pub async fn search(&self, query: &MemoryQuery) -> Result<Vec<RetrievalResult>, StoreError> {
        self.envelopes.search(query).await
    }

    pub async fn load_content(&self, ids: &[Uuid]) -> Result<Vec<ContentBody>, StoreError> {
        self.content.get_batch(ids).await
    }

    pub async fn materialize(&self, ids: &[Uuid]) -> Result<Vec<Engram>, StoreError> {
        let envelopes = self.envelopes.get_batch(ids).await?;
        let contents = self.content.get_batch(ids).await?;
        Ok(envelopes
            .into_iter()
            .zip(contents)
            .map(|(envelope, content)| Engram { envelope, content })
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────
// Arc-based shared store — FIX #1: split store bug
// ─────────────────────────────────────────────────────────────

/// A store that holds Arc'd backends so it can be cloned and shared
/// between the HTTP server and the consolidation engine, ensuring they
/// operate on the *same* underlying data.
pub struct SharedMnemeStore<E: EnvelopeIndex, C: ContentStore> {
    pub envelopes: Arc<E>,
    pub content: Arc<C>,
}

impl<E: EnvelopeIndex, C: ContentStore> SharedMnemeStore<E, C> {
    pub fn new(envelopes: Arc<E>, content: Arc<C>) -> Self {
        Self { envelopes, content }
    }

    pub async fn insert(&self, engram: &Engram) -> Result<(), StoreError> {
        self.envelopes.upsert(&engram.envelope).await?;
        self.content.put(&engram.content).await?;
        Ok(())
    }

    pub async fn search(&self, query: &MemoryQuery) -> Result<Vec<RetrievalResult>, StoreError> {
        self.envelopes.search(query).await
    }
}

/// Constructor for a shared in-memory store. Returns Arc'd backends that
/// can be handed both to the server state and to the ConsolidationEngine.
pub fn new_shared_memory_store() -> (
    Arc<InMemoryEnvelopeIndex>,
    Arc<InMemoryContentStore>,
) {
    (
        Arc::new(InMemoryEnvelopeIndex::new()),
        Arc::new(InMemoryContentStore::new()),
    )
}

/// Constructor for a shared SQLite store.
#[cfg(feature = "sqlite")]
pub fn new_shared_sqlite_store(
    path: &str,
) -> Result<(Arc<SqliteEnvelopeIndex>, Arc<SqliteContentStore>), StoreError> {
    Ok((
        Arc::new(SqliteEnvelopeIndex::new(path)?),
        Arc::new(SqliteContentStore::new(path)?),
    ))
}