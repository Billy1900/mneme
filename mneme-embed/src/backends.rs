//! Concrete embedding model implementations.
//!
//! - `MockEmbeddingModel`: deterministic hash-based vectors for testing
//! - `OpenAIEmbeddingModel`: OpenAI text-embedding-3-small via HTTP API

use async_trait::async_trait;
use mneme_core::EmbeddingVec;

use crate::{EmbedError, EmbeddingModel};

// ─────────────────────────────────────────────────────────────
// Mock embedding model (for testing)
// ─────────────────────────────────────────────────────────────

/// Deterministic embedding model for tests.
///
/// Produces consistent vectors for the same input text using a hash function.
/// NOT for production — the vectors have no semantic meaning.
/// But they DO produce consistent cosine similarities, so clustering
/// and drift detection logic can be tested.
pub struct MockEmbeddingModel {
    dim: usize,
}

impl MockEmbeddingModel {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Default for MockEmbeddingModel {
    fn default() -> Self {
        Self::new(128)
    }
}

#[async_trait]
impl EmbeddingModel for MockEmbeddingModel {
    async fn embed(&self, text: &str) -> Result<EmbeddingVec, EmbedError> {
        // Deterministic pseudo-random vector from text hash
        let mut vec = vec![0.0f32; self.dim];
        let bytes = text.as_bytes();

        for (i, val) in vec.iter_mut().enumerate() {
            let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3); // FNV prime
            }
            h ^= i as u64;
            h = h.wrapping_mul(0x100000001b3);
            // Map to [-1, 1] range
            *val = ((h % 10000) as f32 / 5000.0) - 1.0;
        }

        // Normalize to unit vector
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }

        Ok(EmbeddingVec(vec))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingVec>, EmbedError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ─────────────────────────────────────────────────────────────
// OpenAI embedding model
// ─────────────────────────────────────────────────────────────

/// OpenAI text-embedding-3-small (1536 dimensions).
///
/// Requires OPENAI_API_KEY environment variable.
/// Good for prototyping; for production consolidation sweeps
/// prefer a local model (BGE, Nomic) to avoid API costs.
pub struct OpenAIEmbeddingModel {
    api_key: String,
    model: String,
    dim: usize,
    client: reqwest::Client,
}

impl OpenAIEmbeddingModel {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "text-embedding-3-small".to_string(),
            dim: 1536,
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(api_key: String, model: &str, dim: usize) -> Self {
        Self {
            api_key,
            model: model.to_string(),
            dim,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EmbeddingModel for OpenAIEmbeddingModel {
    async fn embed(&self, text: &str) -> Result<EmbeddingVec, EmbedError> {
        let batch = self.embed_batch(&[text]).await?;
        batch
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Model("empty response".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingVec>, EmbedError> {
        if texts.len() > 2048 {
            return Err(EmbedError::BatchTooLarge(texts.len(), 2048));
        }

        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbedError::Model(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(EmbedError::Model(format!(
                "OpenAI API error {}: {}",
                status, body
            )));
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| EmbedError::Model(e.to_string()))?;

        let embeddings = data["data"]
            .as_array()
            .ok_or_else(|| EmbedError::Model("missing data array".into()))?;

        let mut results = Vec::with_capacity(texts.len());
        for item in embeddings {
            let vec: Vec<f32> = item["embedding"]
                .as_array()
                .ok_or_else(|| EmbedError::Model("missing embedding".into()))?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            results.push(EmbeddingVec(vec));
        }

        Ok(results)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
