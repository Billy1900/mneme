//! # mneme-embed
//!
//! Embedding model abstraction + backends.
//!
//! Backends:
//! - `MockEmbeddingModel`: deterministic hash-based vectors (testing)
//! - `OpenAIEmbeddingModel`: text-embedding-3-small via API

use async_trait::async_trait;
use mneme_core::EmbeddingVec;

pub mod backends;

pub use backends::{MockEmbeddingModel, OpenAIEmbeddingModel};

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("embedding model error: {0}")]
    Model(String),
    #[error("batch too large: {0} > max {1}")]
    BatchTooLarge(usize, usize),
}

#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, text: &str) -> Result<EmbeddingVec, EmbedError>;
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingVec>, EmbedError>;
    fn dim(&self) -> usize;
}

// ─────────────────────────────────────────────────────────────
// Clustering (used by compaction)
// ─────────────────────────────────────────────────────────────

pub fn agglomerative_cluster(vectors: &[EmbeddingVec], threshold: f32) -> Vec<Vec<usize>> {
    let n = vectors.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            if vectors[i].cosine_similarity(&vectors[j]) > threshold {
                union(&mut parent, i, j);
            }
        }
    }

    let mut clusters: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        clusters.entry(find(&mut parent, i)).or_default().push(i);
    }
    clusters.into_values().collect()
}
