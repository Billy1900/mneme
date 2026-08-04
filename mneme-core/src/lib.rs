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
    /// **Valid time**: when the asserted fact became true *in the world*.
    ///
    /// This is a different axis from `created_at`/`updated_at`, which are
    /// *transaction time* — when the system learned or recorded something.
    /// "Melanie adopted Cooper in May 2023", recorded today, has a
    /// `valid_at` of May 2023 and a `created_at` of today. Keeping both is
    /// what makes "what was true as of date X" answerable, as opposed to
    /// "what did we write down before date X".
    ///
    /// `None` means unknown — treat the fact as valid for all time up to
    /// `invalid_at`. Optional and `#[serde(default)]` so envelopes written
    /// before this field existed still deserialize from snapshots, WAL lines
    /// and Qdrant payloads.
    #[serde(default)]
    pub valid_at: Option<DateTime<Utc>>,
    /// **Valid time**: when the asserted fact stopped being true.
    ///
    /// `None` means still believed true. Set by conflict resolution when a
    /// newer fact contradicts this one, which is the distinction between
    /// invalidation and supersession: `superseded_by` says "a newer *version*
    /// of this memory exists", `invalid_at` says "this was true, and then it
    /// wasn't". A superseded engram was never true-then-false; an invalidated
    /// one still correctly answers a question asked about an earlier time.
    #[serde(default)]
    pub invalid_at: Option<DateTime<Utc>>,
}

impl Envelope {
    pub fn is_active(&self) -> bool {
        self.superseded_by.is_none()
    }

    /// Whether this engram's fact was true at `at`, on the valid-time axis.
    ///
    /// An unknown `valid_at` is treated as "true from the beginning" rather
    /// than "never true" — most engrams carry no explicit valid time, and
    /// excluding them would filter away nearly the whole store.
    pub fn is_valid_at(&self, at: DateTime<Utc>) -> bool {
        let started = self.valid_at.is_none_or(|start| at >= start);
        let not_yet_ended = self.invalid_at.is_none_or(|end| at < end);
        started && not_yet_ended
    }

    /// Whether the fact is believed true right now.
    pub fn is_valid_now(&self) -> bool {
        self.is_valid_at(Utc::now())
    }

    /// Mark the fact as having stopped being true at `at`.
    ///
    /// Idempotent in the sense that the earliest invalidation wins: once a
    /// fact is known to have ended, a later contradiction shouldn't extend
    /// its validity window.
    pub fn invalidate(&mut self, at: DateTime<Utc>) {
        self.invalid_at = Some(match self.invalid_at {
            Some(existing) => existing.min(at),
            None => at,
        });
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
    /// Raw query text for lexical (BM25) matching against envelope summaries,
    /// fused with vector similarity via Reciprocal Rank Fusion. Empty string
    /// disables the lexical channel and falls back to pure vector search.
    pub query_text: String,
    /// Valid-time filter: keep only engrams whose fact was true at this
    /// instant (see [`Envelope::is_valid_at`]). `Utc::now()` answers "what is
    /// true today"; a past instant answers "what was believed true then".
    ///
    /// `None` disables validity filtering entirely and returns invalidated
    /// engrams alongside current ones. That is the default deliberately —
    /// it keeps every existing caller's behaviour unchanged, and makes
    /// dropping contradicted facts an explicit decision at each call site
    /// rather than a silent global one.
    pub as_of: Option<DateTime<Utc>>,
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
            query_text: String::new(),
            as_of: None,
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
// Graph triplet — entity-relation facts extracted from engram text,
// enabling multi-hop traversal that pure vector similarity on decomposed
// sub-queries can miss when the connecting fact wasn't retrieved directly.
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTriplet {
    pub subject: String,
    pub relation: String,
    pub object: String,
    /// Free-form date/time this fact pertains to, if any (e.g. from a
    /// session's date-anchored turn). Not parsed/normalized.
    pub date: Option<String>,
    /// The engram this triplet was extracted from — traversal resolves
    /// back to this id to pull the full text into a recall candidate.
    pub source_engram_id: Uuid,
}

// ─────────────────────────────────────────────────────────────
// Extracted fact — an atomic, self-contained statement distilled from
// raw conversation turns on the write path.
//
// Raw turns are context-dependent ("yeah, I switched last month") and
// compacted engrams are cluster-level summaries that lose specifics. A
// fact is neither: one standalone assertion with its pronouns resolved,
// which is what a retrieval query can actually match against.
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    /// The fact as a single self-contained sentence, e.g.
    /// "Melanie adopted a golden retriever named Cooper."
    pub text: String,
    /// Free-form date/time the fact pertains to, if the source turn carried
    /// one. Not parsed/normalized — same convention as [`GraphTriplet::date`].
    pub date: Option<String>,
    /// Optional (subject, relation, object) decomposition of the same fact,
    /// fed straight into the entity graph. Populated in the same LLM call
    /// that produces `text`, so graph coverage costs no extra round-trip.
    pub subject: Option<String>,
    pub relation: Option<String>,
    pub object: Option<String>,
}

impl ExtractedFact {
    /// Best-effort conversion of the free-form `date` string into a valid-time
    /// instant for [`Envelope::valid_at`].
    ///
    /// The extractor returns whatever the conversation said — "8 May 2023",
    /// "May 2023", "2023-05-08". Only unambiguous absolute forms are accepted;
    /// relative expressions ("last month", "yesterday") are deliberately not
    /// resolved here, because doing so correctly needs the conversation's own
    /// reference date and guessing would write a confidently wrong timestamp
    /// into the validity window — worse than leaving it unknown.
    ///
    /// Month-only forms resolve to the first of the month: a fact known to
    /// hold "in May 2023" is valid from the start of May, which is the
    /// conservative reading for an `as_of` query.
    pub fn valid_at(&self) -> Option<DateTime<Utc>> {
        let raw = self.date.as_deref()?.trim();
        if raw.is_empty() {
            return None;
        }

        let midnight_utc = |d: chrono::NaiveDate| d.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc());

        // Full dates first, then month-only, most-specific to least.
        for fmt in ["%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y"] {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(raw, fmt) {
                return midnight_utc(d);
            }
        }
        for fmt in ["%B %Y", "%b %Y", "%Y-%m"] {
            if let Ok(d) =
                chrono::NaiveDate::parse_from_str(&format!("1 {raw}"), &format!("%d {fmt}"))
            {
                return midnight_utc(d);
            }
        }
        None
    }

    /// The triplet form of this fact, if the extractor decomposed it.
    pub fn as_triplet(&self, source_engram_id: Uuid) -> Option<GraphTriplet> {
        let subject = self.subject.as_ref()?.trim();
        let relation = self.relation.as_ref()?.trim();
        let object = self.object.as_ref()?.trim();
        if subject.is_empty() || relation.is_empty() || object.is_empty() {
            return None;
        }
        Some(GraphTriplet {
            subject: subject.to_string(),
            relation: relation.to_string(),
            object: object.to_string(),
            date: self.date.clone(),
            source_engram_id,
        })
    }
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
