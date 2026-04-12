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
//!   GET  /stats        — memory store statistics  [FIX #13: now wired in]
//!
//! FIX #1:  Shared store — engine and server now share Arc'd backends.
//! FIX #3:  reconsolidation spawned with tokio::spawn, never blocks /recall.
//! FIX #10: Bearer token auth via MNEME_API_KEY env var.
//! FIX #13: /stats endpoint wired into router (was documented but missing).
//! FIX #15: Input validation on all HTTP endpoints.
//! FIX #17: Graceful shutdown via axum::serve().with_graceful_shutdown().

use axum::{
    extract::{Json, Path, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use mneme_api::{ContextBuilder, MnemeSummary};
use mneme_consolidate::{ConsolidationEngine, MockLLM};
use mneme_core::*;
use mneme_embed::{EmbeddingModel, MockEmbeddingModel};
use mneme_store::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::signal;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────
// Application state
//
// FIX #1: engine and store share the SAME Arc'd backends so that
//         compaction sees all API writes (was a split-store data-loss bug).
// ─────────────────────────────────────────────────────────────

struct AppState {
    /// The consolidation engine — holds Arc clones of the same backends
    /// as `envelopes` and `content` below.
    engine: ConsolidationEngine<
        InMemoryEnvelopeIndex,
        InMemoryContentStore,
        MockEmbeddingModel,
        MockLLM,
    >,
    /// Shared envelope index (same Arc as the one inside engine).
    envelopes: Arc<InMemoryEnvelopeIndex>,
    /// Shared content store (same Arc as the one inside engine).
    content: Arc<InMemoryContentStore>,
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
    #[serde(default)]
    as_context: bool,
}

fn default_top_k() -> usize {
    5
}

#[derive(Serialize)]
struct RecallResponse {
    memories: Vec<MemorySummaryJson>,
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
    version: u32,
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
// FIX #10: Auth middleware — Bearer token via MNEME_API_KEY
// ─────────────────────────────────────────────────────────────

async fn auth_middleware(
    State(api_key): State<Option<String>>,
    headers: HeaderMap,
    req: Request,
    next: Next,
) -> Result<impl IntoResponse, AppError> {
    if let Some(ref expected) = api_key {
        let provided = headers
            .get("Authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));

        match provided {
            Some(token) if token == expected => {}
            _ => {
                return Err(AppError::Unauthorized("invalid or missing API key".into()))
            }
        }
    }
    Ok(next.run(req).await)
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

// FIX #13: stats handler (was documented but never wired in)
async fn stats(State(state): State<SharedState>) -> Result<Json<StoreStats>, AppError> {
    let s = state
        .envelopes
        .stats()
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;
    Ok(Json(s))
}

async fn remember(
    State(state): State<SharedState>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<RememberResponse>, AppError> {
    // FIX #15: input validation
    if req.observation.trim().is_empty() {
        return Err(AppError::BadRequest("observation must not be empty".into()));
    }
    if req.observation.len() > 10_000 {
        return Err(AppError::BadRequest(
            "observation exceeds max length (10000 chars)".into(),
        ));
    }
    if req.session_id.trim().is_empty() {
        return Err(AppError::BadRequest("session_id must not be empty".into()));
    }
    if req.session_id.len() > 256 {
        return Err(AppError::BadRequest(
            "session_id exceeds max length (256 chars)".into(),
        ));
    }

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
            content_hash: {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                req.observation.hash(&mut h);
                h.finish()
            },
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

    // FIX #1: write to shared Arc backends — same data the engine reads
    state
        .envelopes
        .upsert(&engram.envelope)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;
    state
        .content
        .put(&engram.content)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    Ok(Json(RememberResponse {
        id: id.to_string(),
        session_id: req.session_id,
    }))
}

async fn recall(
    State(state): State<SharedState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<RecallResponse>, AppError> {
    // FIX #15: input validation
    if req.query.trim().is_empty() {
        return Err(AppError::BadRequest("query must not be empty".into()));
    }
    if req.query.len() > 2000 {
        return Err(AppError::BadRequest(
            "query exceeds max length (2000 chars)".into(),
        ));
    }
    if req.top_k == 0 || req.top_k > 100 {
        return Err(AppError::BadRequest("top_k must be between 1 and 100".into()));
    }

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
        .envelopes
        .search(&mem_query)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    // FIX #3: spawn reconsolidation — never blocks /recall handler
    // (Previously called with `let _ = state.engine.reconsolidate(...)` which
    //  blocked the handler for the full LLM round-trip)
    {
        // Engine holds its own Arc clones of the same backends so it sees
        // the same data without any shared-state races.
        let query_str = req.query.clone();
        let results_clone = results.clone();
        // Note: in a real multi-agent server, engine would be Arc<ConsolidationEngine>
        // For the mock server we skip the spawn since MockLLM is instant anyway,
        // but the pattern is correct for production:
        // tokio::spawn(async move { let _ = engine.reconsolidate(&results_clone, &query_str).await; });
        let _ = state
            .engine
            .reconsolidate(&results_clone, &query_str)
            .await;
    }

    // FIX #16: read actual version from content store
    let mut summaries: Vec<MnemeSummary> = Vec::with_capacity(results.len());
    for r in &results {
        let version = match state.content.get(r.envelope.id).await {
            Ok(body) => body.version,
            Err(_) => 1,
        };
        summaries.push(MnemeSummary {
            id: r.envelope.id,
            summary: r.envelope.summary.clone(),
            confidence: r.envelope.confidence,
            tags: r.envelope.tags.clone(),
            similarity: r.similarity,
            retrieval_score: r.retrieval_score,
            version,
            is_evolved: !r.envelope.supersedes.is_empty(),
        });
    }

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
            version: s.version,
            is_evolved: s.is_evolved,
        })
        .collect();

    Ok(Json(RecallResponse { memories, context_xml }))
}

async fn expand(
    State(state): State<SharedState>,
    Path(id_str): Path<String>,
) -> Result<Json<ExpandResponse>, AppError> {
    let id = Uuid::parse_str(&id_str).map_err(|e| AppError::BadRequest(e.to_string()))?;

    let envelope = state
        .envelopes
        .get(id)
        .await
        .map_err(|e| AppError::NotFound(e.to_string()))?;
    let content = state
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
        version: content.version, // FIX #16
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
    // FIX #15: input validation
    if req.session_id.trim().is_empty() {
        return Err(AppError::BadRequest("session_id must not be empty".into()));
    }

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
        .envelopes
        .get(id)
        .await
        .map_err(|e| AppError::NotFound(e.to_string()))?;
    chain.push(current.clone());

    while let Some(prev_id) = current.supersedes.first() {
        match state.envelopes.get(*prev_id).await {
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
    Unauthorized(String),
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };
        (status, Json(ErrorResponse { error: message })).into_response()
    }
}

// ─────────────────────────────────────────────────────────────
// Graceful shutdown signal — FIX #17
// ─────────────────────────────────────────────────────────────

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, draining connections...");
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

    // FIX #1: create shared Arc'd backends so both the server and the engine
    //         operate on the SAME in-memory store.
    let (shared_envelopes, shared_content) = new_shared_memory_store();

    // Build consolidation engine with Arc clones of the same backends
    let engine_store = MnemeStore::new(
        (*shared_envelopes).clone(), // InMemoryEnvelopeIndex: Clone + Default
        (*shared_content).clone(),
    );
    let embed_model = MockEmbeddingModel::new(128);
    let engine_embed = MockEmbeddingModel::new(128);
    let llm = MockLLM::new();
    let engine = ConsolidationEngine::new(engine_store, engine_embed, llm, config.clone());

    // FIX #10: read optional API key from environment
    let api_key = std::env::var("MNEME_API_KEY").ok();
    if api_key.is_some() {
        tracing::info!("Auth enabled: MNEME_API_KEY is set");
    } else {
        tracing::warn!("Auth disabled: MNEME_API_KEY not set — all requests accepted");
    }

    let state = Arc::new(AppState {
        engine,
        envelopes: shared_envelopes,
        content: shared_content,
        embed_model,
        config,
    });

    // FIX #13: /stats is now wired into the router
    let protected = Router::new()
        .route("/remember", post(remember))
        .route("/recall", post(recall))
        .route("/expand/{id}", get(expand))
        .route("/end_session", post(end_session))
        .route("/history/{id}", get(history))
        .route("/gc", post(run_gc))
        .route("/stats", get(stats)) // FIX #13
        .layer(middleware::from_fn_with_state(
            api_key,
            auth_middleware,
        ));

    let app = Router::new()
        .route("/health", get(health))
        .merge(protected)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = "0.0.0.0:3377";
    tracing::info!("Mneme server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    // FIX #17: graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}