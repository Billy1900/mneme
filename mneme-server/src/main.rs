//! # mneme-server
//!
//! HTTP server exposing the Mneme memory system as a REST API.
//!
//! Endpoints:
//!   POST /remember     — store an observation in working memory
//!   POST /recall       — search semantic memory (returns summaries)
//!   GET  /expand/:id   — load full content body (progressive disclosure L2)
//!   POST /end_session  — trigger compaction for a session
//!   GET  /history/:id  — get version history of an engram
//!   POST /gc           — run garbage collection
//!   GET  /health       — health check
//!   GET  /stats        — memory store statistics

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use mneme_api::{ContextBuilder, MnemeDetail, MnemeSummary};
use mneme_consolidate::{ConsolidateError, ConsolidationEngine, MockLLM};
use mneme_core::*;
use mneme_embed::MockEmbeddingModel;
use mneme_store::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────
// Application state
// ─────────────────────────────────────────────────────────────

/// Shared application state holding the memory system.
///
/// In production, swap InMemory* for SqliteEnvelopeIndex + SqliteContentStore
/// or QdrantEnvelopeIndex + SqliteContentStore.
struct AppState {
    engine: ConsolidationEngine<
        InMemoryEnvelopeIndex,
        InMemoryContentStore,
        MockEmbeddingModel,
        MockLLM,
    >,
    store: Arc<MnemeStore<InMemoryEnvelopeIndex, InMemoryContentStore>>,
    embed_model: MockEmbeddingModel,
    config: MnemeConfig,
}

type SharedState = Arc<AppState>;

// ─────────────────────────────────────────────────────────────
// Request / response types
// ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RememberRequest {
    observation: String,
    session_id: String,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Serialize)]
struct RememberResponse {
    id: String,
    session_id: String,
}

#[derive(Deserialize)]
struct RecallRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    tags: Vec<String>,
    /// If true, return XML-formatted context for prompt injection.
    #[serde(default)]
    as_context: bool,
}

fn default_top_k() -> usize {
    5
}

#[derive(Serialize)]
struct RecallResponse {
    memories: Vec<MemorySummaryJson>,
    /// XML-formatted context for direct prompt injection (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    context_xml: Option<String>,
}

#[derive(Serialize)]
struct MemorySummaryJson {
    id: String,
    summary: String,
    confidence: f32,
    tags: Vec<String>,
    similarity: f32,
    retrieval_score: f32,
    is_evolved: bool,
}

#[derive(Serialize)]
struct ExpandResponse {
    id: String,
    summary: String,
    full_text: String,
    confidence: f32,
    tags: Vec<String>,
    version: u32,
    created_at: String,
    updated_at: String,
    access_count: u64,
    provenance_count: usize,
    conflict_count: usize,
    related_count: usize,
}

#[derive(Deserialize)]
struct EndSessionRequest {
    session_id: String,
}

#[derive(Serialize)]
struct EndSessionResponse {
    session_id: String,
    compacted_engrams: usize,
}

#[derive(Serialize)]
struct HistoryEntry {
    id: String,
    summary: String,
    confidence: f32,
    created_at: String,
    updated_at: String,
    superseded_by: Option<String>,
    is_active: bool,
}

#[derive(Serialize)]
struct HistoryResponse {
    engram_id: String,
    versions: Vec<HistoryEntry>,
}

#[derive(Serialize)]
struct GcResponse {
    removed: usize,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ─────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn remember(
    State(state): State<SharedState>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<RememberResponse>, AppError> {
    let embedding = state
        .embed_model
        .embed(&req.observation)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    let id = Uuid::new_v4();
    let now = chrono::Utc::now();

    let summary = if req.observation.len() > 100 {
        format!("{}...", &req.observation[..97])
    } else {
        req.observation.clone()
    };

    let engram = Engram {
        envelope: Envelope {
            id,
            embedding,
            confidence: 0.5,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            access_count: 0,
            memory_type: MemoryType::Working,
            source_sessions: vec![req.session_id.clone()],
            supersedes: vec![],
            superseded_by: None,
            summary,
            tags: req.tags,
            content_hash: 0,
        },
        content: ContentBody {
            engram_id: id,
            full_text: req.observation.clone(),
            provenance: vec![ProvenanceRecord {
                session_id: req.session_id.clone(),
                turn_id: None,
                timestamp: now,
                raw_excerpt: req.observation,
            }],
            conflict_log: vec![],
            related: vec![],
            version: 1,
        },
    };

    state
        .store
        .insert(&engram)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    // Check buffer threshold
    let wm_count = state
        .store
        .envelopes
        .list_working_memory(&req.session_id)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
        .len();

    if wm_count >= state.config.compaction_buffer_threshold {
        tracing::info!(session = %req.session_id, count = wm_count, "Auto-compacting");
        let _ = state.engine.compact_session(&req.session_id).await;
    }

    Ok(Json(RememberResponse {
        id: id.to_string(),
        session_id: req.session_id,
    }))
}

async fn recall(
    State(state): State<SharedState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<RecallResponse>, AppError> {
    let query_embedding = state
        .embed_model
        .embed(&req.query)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    let mem_query = MemoryQuery {
        embedding: query_embedding,
        top_k: req.top_k,
        active_only: true,
        memory_type: Some(MemoryType::Semantic),
        tags: req.tags,
        min_confidence: Some(0.1),
        recency_weight: 0.2,
    };

    let results = state
        .store
        .search(&mem_query)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    // Background reconsolidation
    let _ = state.engine.reconsolidate(&results, &req.query).await;

    let summaries: Vec<MnemeSummary> = results
        .iter()
        .map(|r| MnemeSummary {
            id: r.envelope.id,
            summary: r.envelope.summary.clone(),
            confidence: r.envelope.confidence,
            tags: r.envelope.tags.clone(),
            similarity: r.similarity,
            retrieval_score: r.retrieval_score,
            version: 1,
            is_evolved: !r.envelope.supersedes.is_empty(),
        })
        .collect();

    let context_xml = if req.as_context {
        Some(ContextBuilder::format_summaries(&summaries))
    } else {
        None
    };

    let memories = summaries
        .into_iter()
        .map(|s| MemorySummaryJson {
            id: s.id.to_string(),
            summary: s.summary,
            confidence: s.confidence,
            tags: s.tags,
            similarity: s.similarity,
            retrieval_score: s.retrieval_score,
            is_evolved: s.is_evolved,
        })
        .collect();

    Ok(Json(RecallResponse {
        memories,
        context_xml,
    }))
}

async fn expand(
    State(state): State<SharedState>,
    Path(id_str): Path<String>,
) -> Result<Json<ExpandResponse>, AppError> {
    let id = Uuid::parse_str(&id_str).map_err(|e| AppError::BadRequest(e.to_string()))?;

    let envelope = state
        .store
        .envelopes
        .get(id)
        .await
        .map_err(|e| AppError::NotFound(e.to_string()))?;
    let content = state
        .store
        .content
        .get(id)
        .await
        .map_err(|e| AppError::NotFound(e.to_string()))?;

    Ok(Json(ExpandResponse {
        id: id.to_string(),
        summary: envelope.summary,
        full_text: content.full_text,
        confidence: envelope.confidence,
        tags: envelope.tags,
        version: content.version,
        created_at: envelope.created_at.to_rfc3339(),
        updated_at: envelope.updated_at.to_rfc3339(),
        access_count: envelope.access_count,
        provenance_count: content.provenance.len(),
        conflict_count: content.conflict_log.len(),
        related_count: content.related.len(),
    }))
}

async fn end_session(
    State(state): State<SharedState>,
    Json(req): Json<EndSessionRequest>,
) -> Result<Json<EndSessionResponse>, AppError> {
    let new_engrams = state
        .engine
        .compact_session(&req.session_id)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    Ok(Json(EndSessionResponse {
        session_id: req.session_id,
        compacted_engrams: new_engrams.len(),
    }))
}

async fn history(
    State(state): State<SharedState>,
    Path(id_str): Path<String>,
) -> Result<Json<HistoryResponse>, AppError> {
    let id = Uuid::parse_str(&id_str).map_err(|e| AppError::BadRequest(e.to_string()))?;

    let mut chain = Vec::new();
    let mut current = state
        .store
        .envelopes
        .get(id)
        .await
        .map_err(|e| AppError::NotFound(e.to_string()))?;
    chain.push(current.clone());

    while let Some(prev_id) = current.supersedes.first() {
        match state.store.envelopes.get(*prev_id).await {
            Ok(prev) => {
                chain.push(prev.clone());
                current = prev;
            }
            Err(_) => break,
        }
    }
    chain.reverse();

    let versions = chain
        .iter()
        .map(|env| HistoryEntry {
            id: env.id.to_string(),
            summary: env.summary.clone(),
            confidence: env.confidence,
            created_at: env.created_at.to_rfc3339(),
            updated_at: env.updated_at.to_rfc3339(),
            superseded_by: env.superseded_by.map(|id| id.to_string()),
            is_active: env.is_active(),
        })
        .collect();

    Ok(Json(HistoryResponse {
        engram_id: id.to_string(),
        versions,
    }))
}

async fn run_gc(State(state): State<SharedState>) -> Result<Json<GcResponse>, AppError> {
    let removed = state
        .store
        .envelopes
        .gc(
            state.config.gc_confidence_floor,
            state.config.working_memory_ttl_hours,
        )
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    Ok(Json(GcResponse { removed }))
}

// ─────────────────────────────────────────────────────────────
// Error handling
// ─────────────────────────────────────────────────────────────

enum AppError {
    BadRequest(String),
    NotFound(String),
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };
        (status, Json(ErrorResponse { error: message })).into_response()
    }
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mneme_server=info,tower_http=debug".into()),
        )
        .init();

    let config = MnemeConfig::default();

    // Build stores
    let envelope_index = InMemoryEnvelopeIndex::new();
    let content_store = InMemoryContentStore::new();
    let store = Arc::new(MnemeStore::new(envelope_index, content_store));

    // Build embedding model and LLM
    let embed_model = MockEmbeddingModel::new(128);
    let llm = MockLLM::new();

    // Build consolidation engine
    // Note: engine needs its own store reference — in production use Arc'd backends
    let engine_envelope_index = InMemoryEnvelopeIndex::new();
    let engine_content_store = InMemoryContentStore::new();
    let engine_store = MnemeStore::new(engine_envelope_index, engine_content_store);
    let engine_embed = MockEmbeddingModel::new(128);
    let engine = ConsolidationEngine::new(engine_store, engine_embed, llm, config.clone());

    let state = Arc::new(AppState {
        engine,
        store,
        embed_model,
        config,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/remember", post(remember))
        .route("/recall", post(recall))
        .route("/expand/{id}", get(expand))
        .route("/end_session", post(end_session))
        .route("/history/{id}", get(history))
        .route("/gc", post(run_gc))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = "0.0.0.0:3377";
    tracing::info!("Mneme server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
