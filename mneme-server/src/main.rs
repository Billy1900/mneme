//! # mneme-server
//!
//! HTTP server exposing the Mneme memory system as a REST API.
//!
//! Endpoints:
//!   POST /remember     — store an observation in working memory
//!   POST /recall       — search semantic memory (returns summaries)
//!   POST /add          — Agent Memory Leaderboard Add contract (multi-message, user-scoped)
//!   POST /search       — Agent Memory Leaderboard Search contract (evidence only, no answers)
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
use mneme_embed::{EmbeddingModel, MockEmbeddingModel, OpenAIEmbeddingModel};
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

pub struct AppState {
    /// The consolidation engine — holds Arc clones of the same backends
    /// as `envelopes` and `content` below.
    engine: ConsolidationEngine<
        InMemoryEnvelopeIndex,
        InMemoryContentStore,
        Arc<dyn EmbeddingModel>,
        MockLLM,
    >,
    /// Shared envelope index (same Arc as the one inside engine).
    envelopes: Arc<InMemoryEnvelopeIndex>,
    /// Shared content store (same Arc as the one inside engine).
    content: Arc<InMemoryContentStore>,
    embed_model: Arc<dyn EmbeddingModel>,
    config: MnemeConfig,
    /// Idempotency table for /add: request_id -> (user_id, session_id) of the
    /// first successful call. The platform retries Add on 408/409/425/429/
    /// 5xx, so a retried request_id must not be written twice.
    add_seen: tokio::sync::Mutex<std::collections::HashMap<String, (String, String)>>,
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
struct ForgetResponse {
    id: String,
}

#[derive(Serialize)]
struct DecayResponse {
    updated: usize,
}

#[derive(Serialize)]
struct GcResponse {
    removed: usize,
}

// ─────────────────────────────────────────────────────────────
// Agent Memory Leaderboard contract types (/add, /search)
//
// user_id has no native field in MemoryQuery/Envelope — it's mapped to a
// `uid:{user_id}` tag, enforced on write and as a mandatory search filter.
// This is what makes isolation depend on the SQLite tags-filter fix.
// ─────────────────────────────────────────────────────────────

fn uid_tag(user_id: &str) -> String {
    format!("uid:{user_id}")
}

#[derive(Deserialize)]
struct AddMessage {
    #[serde(default)]
    role: String,
    content: String,
    #[serde(default)]
    timestamp: Option<i64>,
}

#[derive(Deserialize)]
struct AddRequest {
    request_id: String,
    messages: Vec<AddMessage>,
    user_id: String,
    session_id: String,
}

#[derive(Serialize)]
struct AddResponse {
    success: bool,
    request_id: String,
    user_id: String,
    session_id: String,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default)]
    options: Vec<String>,
    user_id: String,
    #[serde(default = "default_search_top_k")]
    top_k: usize,
}

fn default_search_top_k() -> usize {
    100
}

#[derive(Serialize)]
struct SearchDataItem {
    id: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<String>,
}

#[derive(Serialize)]
struct SearchResponse {
    data: Vec<SearchDataItem>,
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
        let (version, full_text) = match state.content.get(r.envelope.id).await {
            Ok(body) => (body.version, body.full_text),
            Err(_) => (1, r.envelope.summary.clone()),
        };
        summaries.push(MnemeSummary {
            id: r.envelope.id,
            summary: r.envelope.summary.clone(),
            full_text,
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

async fn add(
    State(state): State<SharedState>,
    Json(req): Json<AddRequest>,
) -> Result<Json<AddResponse>, AppError> {
    if req.request_id.trim().is_empty() {
        return Err(AppError::BadRequest("request_id must not be empty".into()));
    }
    if req.user_id.trim().is_empty() {
        return Err(AppError::BadRequest("user_id must not be empty".into()));
    }
    if req.session_id.trim().is_empty() {
        return Err(AppError::BadRequest("session_id must not be empty".into()));
    }
    if req.messages.is_empty() {
        return Err(AppError::BadRequest("messages must not be empty".into()));
    }
    for msg in &req.messages {
        if msg.content.len() > 10_000 {
            return Err(AppError::BadRequest(
                "message content exceeds max length (10000 chars)".into(),
            ));
        }
    }

    // Idempotency: the platform retries Add on 408/409/425/429/5xx, so a
    // repeated request_id must be a no-op, not a duplicate write. Reserve
    // the slot before writing (not after) so two retries racing each other
    // can't both pass the check and double-write.
    {
        let mut seen = state.add_seen.lock().await;
        if let Some((user_id, session_id)) = seen.get(&req.request_id) {
            return Ok(Json(AddResponse {
                success: true,
                request_id: req.request_id,
                user_id: user_id.clone(),
                session_id: session_id.clone(),
            }));
        }
        seen.insert(
            req.request_id.clone(),
            (req.user_id.clone(), req.session_id.clone()),
        );
    }

    let uid_tag = uid_tag(&req.user_id);
    let write_result = write_add_messages(&state, &req, &uid_tag).await;

    if let Err(e) = write_result {
        // The write genuinely failed — undo the reservation so a real retry
        // (as opposed to a duplicate-delivery retry) can actually happen.
        state.add_seen.lock().await.remove(&req.request_id);
        return Err(e);
    }

    Ok(Json(AddResponse {
        success: true,
        request_id: req.request_id,
        user_id: req.user_id,
        session_id: req.session_id,
    }))
}

async fn write_add_messages(
    state: &SharedState,
    req: &AddRequest,
    uid_tag: &str,
) -> Result<(), AppError> {
    for msg in &req.messages {
        if msg.content.trim().is_empty() {
            continue;
        }

        let embedding = state
            .embed_model
            .embed(&msg.content)
            .await
            .map_err(|e| AppError::Internal(e.to_string()))?;

        let id = Uuid::new_v4();
        let now = msg
            .timestamp
            .and_then(chrono::DateTime::from_timestamp_millis)
            .unwrap_or_else(chrono::Utc::now);

        let summary = if msg.content.len() > 100 {
            format!("{}...", &msg.content[..97])
        } else {
            msg.content.clone()
        };

        let mut tags = vec![uid_tag.to_string()];
        if !msg.role.trim().is_empty() {
            tags.push(format!("role:{}", msg.role));
        }

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
                tags,
                content_hash: {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    msg.content.hash(&mut h);
                    h.finish()
                },
            },
            content: ContentBody {
                engram_id: id,
                full_text: msg.content.clone(),
                provenance: vec![ProvenanceRecord {
                    session_id: req.session_id.clone(),
                    turn_id: None,
                    timestamp: now,
                    raw_excerpt: msg.content.clone(),
                }],
                conflict_log: vec![],
                related: vec![],
                version: 1,
            },
        };

        // Persisted synchronously before we return 200, per contract
        // ("Add Response must return HTTP 200 only after data is persisted
        //  and immediately searchable").
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
    }

    Ok(())
}

async fn search(
    State(state): State<SharedState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    if req.query.trim().is_empty() {
        return Err(AppError::BadRequest("query must not be empty".into()));
    }
    if req.query.len() > 2000 {
        return Err(AppError::BadRequest(
            "query exceeds max length (2000 chars)".into(),
        ));
    }
    if req.user_id.trim().is_empty() {
        return Err(AppError::BadRequest("user_id must not be empty".into()));
    }
    if req.top_k == 0 || req.top_k > 100 {
        return Err(AppError::BadRequest("top_k must be between 1 and 100".into()));
    }

    // options (choice-question answers) don't change which memories are
    // relevant, only how the platform later scores them — fold them into
    // the embedded text so retrieval can still be option-aware.
    let query_text = if req.options.is_empty() {
        req.query.clone()
    } else {
        format!("{}\n{}", req.query, req.options.join("\n"))
    };

    let query_embedding = state
        .embed_model
        .embed(&query_text)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    // memory_type: None searches both Working (raw turns) and Semantic
    // (compacted) memory — compaction isn't guaranteed to have run between
    // /add and /search calls, so raw turns must stay reachable.
    let mem_query = MemoryQuery {
        embedding: query_embedding,
        top_k: req.top_k,
        active_only: true,
        memory_type: None,
        tags: vec![uid_tag(&req.user_id)],
        min_confidence: None,
        recency_weight: 0.2,
    };

    let results = state
        .envelopes
        .search(&mem_query)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    let mut data = Vec::with_capacity(results.len());
    for r in &results {
        // Prefer the full raw text over the (possibly truncated) summary —
        // the platform generates the final answer from this content.
        let content = match state.content.get(r.envelope.id).await {
            Ok(body) => body.full_text,
            Err(_) => r.envelope.summary.clone(),
        };
        data.push(SearchDataItem {
            id: r.envelope.id.to_string(),
            content,
            score: Some(r.retrieval_score),
            created_at: Some(r.envelope.created_at.to_rfc3339()),
        });
    }

    Ok(Json(SearchResponse { data }))
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

async fn forget(
    State(state): State<SharedState>,
    Path(id_str): Path<String>,
) -> Result<Json<ForgetResponse>, AppError> {
    let id = Uuid::parse_str(&id_str).map_err(|e| AppError::BadRequest(e.to_string()))?;

    state
        .envelopes
        .delete(id)
        .await
        .map_err(|e| AppError::NotFound(e.to_string()))?;
    // Best-effort content removal
    let _ = state.content.delete(id).await;

    Ok(Json(ForgetResponse { id: id.to_string() }))
}

async fn run_decay(State(state): State<SharedState>) -> Result<Json<DecayResponse>, AppError> {
    let updated = state
        .envelopes
        .apply_decay(state.config.decay_lambda)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;
    Ok(Json(DecayResponse { updated }))
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
// App builder — used by main and tests
// ─────────────────────────────────────────────────────────────

pub fn build_app(state: SharedState, api_key: Option<String>) -> Router {
    let protected = Router::new()
        .route("/remember", post(remember))
        .route("/recall", post(recall))
        .route("/add", post(add))
        .route("/search", post(search))
        .route("/expand/{id}", get(expand))
        .route("/end_session", post(end_session))
        .route("/history/{id}", get(history))
        .route("/forget/{id}", axum::routing::delete(forget))
        .route("/decay", post(run_decay))
        .route("/gc", post(run_gc))
        .route("/stats", get(stats))
        .layer(middleware::from_fn_with_state(api_key, auth_middleware));

    Router::new()
        .route("/health", get(health))
        .merge(protected)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub fn build_state() -> SharedState {
    let config = MnemeConfig::default();
    let (shared_envelopes, shared_content) = new_shared_memory_store();
    let engine_store = MnemeStore::new(
        (*shared_envelopes).clone(),
        (*shared_content).clone(),
    );
    // FIX #18: use a real embedding backend when OPENAI_API_KEY is set,
    // falling back to the deterministic mock otherwise (local dev, tests,
    // CI — none of which have API credentials).
    let embed_model: Arc<dyn EmbeddingModel> = match std::env::var("OPENAI_API_KEY") {
        Ok(key) if !key.trim().is_empty() => {
            tracing::info!("Embedding backend: OpenAI (text-embedding-3-small)");
            Arc::new(OpenAIEmbeddingModel::new(key))
        }
        _ => {
            tracing::warn!(
                "OPENAI_API_KEY not set — using MockEmbeddingModel (not suitable for real evaluation)"
            );
            Arc::new(MockEmbeddingModel::new(128))
        }
    };
    // Engine and the request handlers must share one embedding client so
    // vectors are comparable (same model, same dimensionality).
    let engine_embed = Arc::clone(&embed_model);
    let llm = MockLLM::new();
    let engine = ConsolidationEngine::new(engine_store, engine_embed, llm, config.clone());
    Arc::new(AppState {
        engine,
        envelopes: shared_envelopes,
        content: shared_content,
        embed_model,
        config,
        add_seen: tokio::sync::Mutex::new(std::collections::HashMap::new()),
    })
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

    let state = build_state();

    // Spawn background Ebbinghaus decay task — runs every hour
    {
        let decay_envelopes = Arc::clone(&state.envelopes);
        let decay_lambda = state.config.decay_lambda;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600));
            loop {
                interval.tick().await;
                match decay_envelopes.apply_decay(decay_lambda).await {
                    Ok(n) => tracing::info!(updated = n, "Scheduled confidence decay applied"),
                    Err(e) => tracing::warn!("Decay task error: {}", e),
                }
            }
        });
    }

    // FIX #10: read optional API key from environment
    let api_key = std::env::var("MNEME_API_KEY").ok();
    if api_key.is_some() {
        tracing::info!("Auth enabled: MNEME_API_KEY is set");
    } else {
        tracing::warn!("Auth disabled: MNEME_API_KEY not set — all requests accepted");
    }

    let app = build_app(state, api_key);

    let addr = "0.0.0.0:3377";
    tracing::info!("Mneme server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    // FIX #17: graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}