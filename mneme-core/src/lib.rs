//! # mneme-core
//!
//! Core data types for the Mneme memory system.
//!
//! The atomic unit is the [`Engram`], which has two layers:
//! - [`Envelope`]: lightweight metadata, always loaded, used for search/filter
//! - [`ContentBody`]: full memory content, loaded on demand (progressive disclosure)
//!
//! This split is the key to token efficiency: search across thousands of envelopes,
//! but only load 2-3 content bodies into the agent's context window.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────
// Embedding vector
// ─────────────────────────────────────────────────────────────

/// Dense embedding vector. Dimensionality depends on the model
/// (e.g., 768 for bge-large, 1536 for text-embedding-3-small).
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

/// The two memory stores in our dual-system architecture.
///
/// Maps to CLS theory:
/// - Working = hippocampal fast-learning system (episode-specific, session-bound)
/// - Semantic = neocortical slow-learning system (decontextualized, persistent)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// Raw observations from agent sessions. High fidelity, short-lived.
    /// Gets compacted into Semantic via the consolidation engine.
    Working,
    /// Distilled knowledge, detached from specific sessions.
    /// The persistent knowledge base. Evolves via reconsolidation.
    Semantic,
}

// ─────────────────────────────────────────────────────────────
// Envelope — the lightweight metadata layer
// ─────────────────────────────────────────────────────────────

/// The envelope is always loaded during search. It carries enough information
/// for the retrieval system to rank, filter, and present a summary to the agent
/// without loading the full content body.
///
/// Progressive disclosure L1: the agent sees summaries and decides which
/// content bodies to load (L2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub id: Uuid,

    /// Dense vector from the embedding model. Used for ANN search
    /// and drift detection during reconsolidation.
    pub embedding: EmbeddingVec,

    /// Confidence in this memory's accuracy. Range [0.0, 1.0].
    /// - Reinforced on retrieval if context confirms
    /// - Decayed on retrieval if context contradicts
    /// - Decayed over time (Ebbinghaus curve)
    /// - Boosted by multiple provenance sources
    pub confidence: f32,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,

    /// How many times this engram has been retrieved.
    /// High access_count + high confidence = core knowledge.
    pub access_count: u64,

    pub memory_type: MemoryType,

    /// Which session(s) contributed to this engram.
    pub source_sessions: Vec<String>,

    /// IDs of engrams that this one replaced (reconsolidation chain).
    /// For compacted engrams: the working memory entries it was distilled from.
    /// For evolved engrams: the previous version it superseded.
    pub supersedes: Vec<Uuid>,

    /// If this engram has been superseded by a newer version, points to it.
    /// None = this is the current/active version.
    pub superseded_by: Option<Uuid>,

    /// 1-2 sentence digest for progressive disclosure.
    /// The agent reads this before deciding whether to load full content.
    pub summary: String,

    /// Freeform tags for metadata-based filtering.
    pub tags: Vec<String>,

    /// Hash of content body for deduplication during compaction.
    pub content_hash: u64,
}

impl Envelope {
    /// Check if this engram is still the active (non-superseded) version.
    pub fn is_active(&self) -> bool {
        self.superseded_by.is_none()
    }

    /// Compute the Ebbinghaus decay factor based on time since last access.
    /// λ controls decay rate; typical value ~0.1 for daily granularity.
    pub fn time_decay(&self, lambda: f64) -> f64 {
        let elapsed = Utc::now()
            .signed_duration_since(self.last_accessed_at)
            .num_hours() as f64;
        (-lambda * elapsed).exp()
    }

    /// Effective retrieval score combining similarity, confidence, and recency.
    /// This is our triple-weighted retrieval à la Generative Agents,
    /// but with confidence as the third axis instead of importance.
    pub fn retrieval_score(&self, similarity: f32, recency_weight: f32) -> f32 {
        let recency = self.time_decay(0.05) as f32; // ~14-hour half-life
        let conf = self.confidence;
        // Weights: similarity dominates, confidence gates, recency boosts
        similarity * 0.5 + conf * 0.3 + recency * recency_weight
    }
}

// ─────────────────────────────────────────────────────────────
// Content body — loaded on demand
// ─────────────────────────────────────────────────────────────

/// A single provenance record linking back to the raw source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub session_id: String,
    pub turn_id: Option<u64>,
    pub timestamp: DateTime<Utc>,
    /// The raw excerpt from the conversation that produced this memory.
    pub raw_excerpt: String,
}

/// A record of a conflict that was resolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRecord {
    pub timestamp: DateTime<Utc>,
    pub loser_id: Uuid,
    pub old_value: String,
    pub new_value: String,
    pub strategy: ConflictStrategy,
    pub confidence_delta: f32,
}

/// How a conflict between two engrams was resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictStrategy {
    /// Newer engram supersedes older. Simple temporal precedence.
    /// Use when: clear factual update (diet changed, job changed).
    TemporalSupersede,

    /// Both engrams have high confidence; synthesize a merged view.
    /// Use when: both sources are credible and partially overlapping.
    ConfidenceMerge,

    /// Both are true in different contexts; both survive with tags.
    /// Use when: "uses Python for ML, Rust for systems" — neither is wrong.
    ConditionalCoexist,

    /// Conflict is unresolvable; flagged for agent/user attention.
    Escalated,
}

/// A typed relationship to another engram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedEngram {
    pub id: Uuid,
    pub relationship: RelationType,
    /// Strength of the relationship [0.0, 1.0].
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    /// Same topic, different aspect.
    Related,
    /// This engram elaborates on the target.
    Elaborates,
    /// This engram contradicts the target (pre-resolution).
    Contradicts,
    /// This engram is a prerequisite for understanding the target.
    Prerequisite,
    /// This engram is a consequence/follow-up of the target.
    Consequence,
}

/// The full content body, loaded on demand (progressive disclosure L2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBody {
    pub engram_id: Uuid,

    /// The actual memory content. For working memory: raw observation text.
    /// For semantic memory: synthesized knowledge statement.
    pub full_text: String,

    /// Chain of provenance back to the original conversations.
    pub provenance: Vec<ProvenanceRecord>,

    /// History of conflicts resolved involving this engram.
    pub conflict_log: Vec<ConflictRecord>,

    /// Typed relationships to other engrams.
    pub related: Vec<RelatedEngram>,

    /// Version number. Incremented on each evolution (reconsolidation).
    pub version: u32,
}

// ─────────────────────────────────────────────────────────────
// Engram — the composite unit
// ─────────────────────────────────────────────────────────────

/// The atomic memory unit. Combining envelope + content body.
///
/// In practice, you often work with just the Envelope (for search/filter)
/// and only materialize the full Engram when the agent needs deep context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Engram {
    pub envelope: Envelope,
    pub content: ContentBody,
}

// ─────────────────────────────────────────────────────────────
// Query types
// ─────────────────────────────────────────────────────────────

/// A query against the semantic memory store.
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    /// The query embedding (computed by the caller).
    pub embedding: EmbeddingVec,
    /// Maximum number of envelopes to return from ANN search (phase 1).
    pub top_k: usize,
    /// Only return active (non-superseded) engrams.
    pub active_only: bool,
    /// Filter by memory type.
    pub memory_type: Option<MemoryType>,
    /// Filter by tags (AND semantics).
    pub tags: Vec<String>,
    /// Minimum confidence threshold.
    pub min_confidence: Option<f32>,
    /// Recency weight in the retrieval score formula.
    pub recency_weight: f32,
}

impl Default for MemoryQuery {
    fn default() -> Self {
        Self {
            embedding: EmbeddingVec(vec![]),
            top_k: 10,
            active_only: true,
            memory_type: None,
            tags: vec![],
            min_confidence: Some(0.1),
            recency_weight: 0.2,
        }
    }
}

/// Result of a retrieval query — envelopes only (progressive disclosure L1).
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub envelope: Envelope,
    pub similarity: f32,
    pub retrieval_score: f32,
}

// ─────────────────────────────────────────────────────────────
// Drift detection (for reconsolidation)
// ─────────────────────────────────────────────────────────────

/// Result of checking whether a retrieved engram has drifted from current context.
#[derive(Debug, Clone)]
pub struct DriftCheck {
    pub engram_id: Uuid,
    pub drift_score: f32, // 1.0 - cosine_similarity(engram, current_context)
    pub needs_evolution: bool,
}

impl DriftCheck {
    pub fn compute(
        mneme_embedding: &EmbeddingVec,
        context_embedding: &EmbeddingVec,
        threshold: f32,
    ) -> Self {
        let sim = mneme_embedding.cosine_similarity(context_embedding);
        let drift = 1.0 - sim;
        Self {
            engram_id: Uuid::nil(), // caller sets this
            drift_score: drift,
            needs_evolution: drift > threshold,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

/// Tunable parameters for the memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnemeConfig {
    /// Cosine similarity threshold for clustering during compaction.
    pub compaction_cluster_threshold: f32,
    /// Max working memory entries before triggering compaction.
    pub compaction_buffer_threshold: usize,
    /// Drift score threshold for triggering evolution on retrieval.
    pub evolution_drift_threshold: f32,
    /// Score gap threshold for choosing temporal supersede vs merge.
    pub conflict_score_gap_threshold: f32,
    /// Minimum confidence for an engram to survive GC.
    pub gc_confidence_floor: f32,
    /// Time-decay lambda for Ebbinghaus forgetting curve.
    pub decay_lambda: f64,
    /// TTL for working memory entries after compaction (hours).
    pub working_memory_ttl_hours: u64,
    /// Confidence decay factor applied to losing engram in conflict.
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
            decay_lambda: 0.05, // ~14-hour half-life
            working_memory_ttl_hours: 168, // 7 days
            conflict_loser_decay: 0.3,
        }
    }
}
