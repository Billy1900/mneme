//! # mneme-core
//!
//! Core data types for the Mneme memory system.
//!
//! The atomic unit is the [`Engram`], which has two layers:
//! - [`Envelope`]: lightweight metadata, always loaded, used for search/filter
//! - [`ContentBody`]: full memory content, loaded on demand (progressive disclosure)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────
// Embedding vector
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingVec(pub Vec<f32>);

impl EmbeddingVec {
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }
}

// ─────────────────────────────────────────────────────────────
// Memory type
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Working,
    Semantic,
}

// ─────────────────────────────────────────────────────────────
// Envelope — the lightweight metadata layer
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub id: Uuid,
    pub embedding: EmbeddingVec,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
    pub access_count: u64,
    pub memory_type: MemoryType,
    pub source_sessions: Vec<String>,
    pub supersedes: Vec<Uuid>,
    pub superseded_by: Option<Uuid>,
    pub summary: String,
    pub tags: Vec<String>,
    pub content_hash: u64,
}

impl Envelope {
    pub fn is_active(&self) -> bool {
        self.superseded_by.is_none()
    }

    /// Ebbinghaus forgetting curve decay.
    pub fn time_decay(&self, lambda: f64) -> f64 {
        let hours_since_access = Utc::now()
            .signed_duration_since(self.last_accessed_at)
            .num_seconds() as f64
            / 3600.0;
        (-lambda * hours_since_access).exp()
    }
}

// ─────────────────────────────────────────────────────────────
// Content body
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub session_id: String,
    pub turn_id: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub raw_excerpt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRecord {
    pub conflicting_id: Uuid,
    pub resolution: ConflictStrategy,
    pub resolved_at: DateTime<Utc>,
    pub resolver_notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictStrategy {
    TemporalSupersede,
    SemanticMerge,
    ContextualCoexist,
    Escalated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedEngram {
    pub id: Uuid,
    pub relationship: RelationType,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    Related,
    Elaborates,
    Contradicts,
    Prerequisite,
    Consequence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBody {
    pub engram_id: Uuid,
    pub full_text: String,
    pub provenance: Vec<ProvenanceRecord>,
    pub conflict_log: Vec<ConflictRecord>,
    pub related: Vec<RelatedEngram>,
    pub version: u32,
}

// ─────────────────────────────────────────────────────────────
// Engram — the composite unit
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Engram {
    pub envelope: Envelope,
    pub content: ContentBody,
}

// ─────────────────────────────────────────────────────────────
// Query types
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub embedding: EmbeddingVec,
    pub top_k: usize,
    pub active_only: bool,
    pub memory_type: Option<MemoryType>,
    pub tags: Vec<String>,
    pub min_confidence: Option<f32>,
    pub recency_weight: f32,
}

impl Default for MemoryQuery {
    fn default() -> Self {
        Self {
            embedding: EmbeddingVec(vec![]),
            top_k: 5,
            active_only: true,
            memory_type: None,
            tags: vec![],
            min_confidence: None,
            recency_weight: 0.2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub envelope: Envelope,
    pub similarity: f32,
    pub retrieval_score: f32,
}

// ─────────────────────────────────────────────────────────────
// Drift check result
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DriftCheck {
    pub engram_id: Uuid,
    pub drift_score: f32,
    pub needs_evolution: bool,
}

impl DriftCheck {
    pub fn compute(stored: &EmbeddingVec, current: &EmbeddingVec, threshold: f32) -> Self {
        let similarity = stored.cosine_similarity(current);
        let drift_score = 1.0 - similarity;
        Self {
            engram_id: Uuid::nil(),
            drift_score,
            needs_evolution: drift_score > threshold,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MnemeConfig {
    pub compaction_cluster_threshold: f32,
    pub compaction_buffer_threshold: usize,
    pub evolution_drift_threshold: f32,
    pub conflict_score_gap_threshold: f32,
    pub gc_confidence_floor: f32,
    pub decay_lambda: f64,
    pub working_memory_ttl_hours: u64,
    pub conflict_loser_decay: f32,
}

impl Default for MnemeConfig {
    fn default() -> Self {
        Self {
            compaction_cluster_threshold: 0.85,
            compaction_buffer_threshold: 20,
            evolution_drift_threshold: 0.3,
            conflict_score_gap_threshold: 0.4,
            gc_confidence_floor: 0.05,
            decay_lambda: 0.05,
            working_memory_ttl_hours: 168,
            conflict_loser_decay: 0.3,
        }
    }
}