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
use uuid::Uuid;

// Backends
pub mod memory;
pub mod memory_content;

#[cfg(feature = "sqlite")]
pub mod sqlite_envelope;
#[cfg(feature = "sqlite")]
pub mod sqlite_content;

#[cfg(feature = "qdrant")]
pub mod qdrant_envelope;

// Re-exports
pub use memory::InMemoryEnvelopeIndex;
pub use memory_content::InMemoryContentStore;

#[cfg(feature = "sqlite")]
pub use sqlite_envelope::SqliteEnvelopeIndex;
#[cfg(feature = "sqlite")]
pub use sqlite_content::SqliteContentStore;

#[cfg(feature = "qdrant")]
pub use qdrant_envelope::QdrantEnvelopeIndex;

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
        Ok(envelopes.into_iter().zip(contents)
            .map(|(envelope, content)| Engram { envelope, content })
            .collect())
    }
}
