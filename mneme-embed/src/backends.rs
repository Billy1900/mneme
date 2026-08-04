//! Concrete embedding model implementations.

use async_trait::async_trait;
use mneme_core::EmbeddingVec;

use crate::{EmbedError, EmbeddingModel};

// ─────────────────────────────────────────────────────────────
// Mock embedding model (for testing)
// ─────────────────────────────────────────────────────────────

#[derive(Clone)]
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
        let mut vec = vec![0.0f32; self.dim];
        let bytes = text.as_bytes();

        for (i, val) in vec.iter_mut().enumerate() {
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h ^= i as u64;
            h = h.wrapping_mul(0x100000001b3);
            *val = ((h % 10000) as f32 / 5000.0) - 1.0;
        }

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
// Local embedding model (fastembed / ONNX, no API key required)
// ─────────────────────────────────────────────────────────────

/// Real embeddings without depending on a paid external API. Runs
/// BGE-small-en-v1.5 (384-dim) locally via ONNX Runtime; the model weights
/// are downloaded from Hugging Face on first use and cached under
/// `~/.cache/fastembed` afterwards, so there's no per-request network call
/// or per-token cost. This is the default backend when `OPENAI_API_KEY`
/// isn't set — a missing/exhausted API key shouldn't be the reason
/// retrieval quality can't be evaluated.
#[derive(Clone)]
pub struct LocalEmbeddingModel {
    model: std::sync::Arc<std::sync::Mutex<fastembed::TextEmbedding>>,
    dim: usize,
}

impl LocalEmbeddingModel {
    pub fn new() -> Result<Self, EmbedError> {
        let model = fastembed::TextEmbedding::try_new(
            fastembed::InitOptions::new(fastembed::EmbeddingModel::BGESmallENV15)
                .with_show_download_progress(true),
        )
        .map_err(|e| EmbedError::Model(format!("failed to load local embedding model: {e}")))?;
        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(model)),
            dim: 384,
        })
    }
}

#[async_trait]
impl EmbeddingModel for LocalEmbeddingModel {
    async fn embed(&self, text: &str) -> Result<EmbeddingVec, EmbedError> {
        let batch = self.embed_batch(&[text]).await?;
        batch
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Model("empty response".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingVec>, EmbedError> {
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let model = self.model.clone();
        // ONNX inference is blocking CPU work — run it off the async
        // executor so it doesn't stall other requests on the same runtime.
        tokio::task::spawn_blocking(move || {
            let mut model = model.lock().unwrap();
            model.embed(owned, None)
        })
        .await
        .map_err(|e| EmbedError::Model(format!("embedding task panicked: {e}")))?
        .map_err(|e| EmbedError::Model(format!("local embedding failed: {e}")))
        .map(|vecs| vecs.into_iter().map(EmbeddingVec).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ─────────────────────────────────────────────────────────────
// OpenAI embedding model
// ─────────────────────────────────────────────────────────────

#[derive(Clone)]
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
