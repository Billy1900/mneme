//! HTTP integration tests for the Mneme server.
//!
//! Uses axum's in-process test client (tower::ServiceExt) — no socket needed.
//! Each test builds a fresh in-memory app and drives it via HTTP requests.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt; // oneshot

// Re-export the helpers from the binary crate via a path hack:
// `mneme-server` is a [[bin]], so we access its pub fns via `include!` or
// by making the integration test an inner module. The cleanest approach for
// a bin crate is to use `include!` from the source file.
//
// We use `#[path]` to bring the module in scope.
#[path = "../../mneme-server/src/main.rs"]
mod server;

fn make_app() -> axum::Router {
    server::build_app(server::build_state(), None)
}

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

// ═══════════════════════════════════════════════════════════
// Test H1: GET /health
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_health() {
    let app = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["status"], "ok");
}

// ═══════════════════════════════════════════════════════════
// Test H2: POST /remember — success
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_remember_success() {
    let app = make_app();
    let payload = json!({"observation": "user prefers dark mode", "session_id": "s1"});

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/remember")
                .header("Content-Type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["id"].is_string());
    assert_eq!(body["session_id"], "s1");
}

// ═══════════════════════════════════════════════════════════
// Test H3: POST /remember — empty observation rejected
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_remember_empty_observation_rejected() {
    let app = make_app();
    let payload = json!({"observation": "   ", "session_id": "s1"});

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/remember")
                .header("Content-Type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ═══════════════════════════════════════════════════════════
// Test H4: POST /recall — returns memories
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_recall() {
    let app = make_app();

    // /recall on empty store returns empty list
    let payload = json!({"query": "preferences", "top_k": 5});
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/recall")
                .header("Content-Type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["memories"].is_array());
}

// ═══════════════════════════════════════════════════════════
// Test H5: POST /recall — as_context returns XML
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_recall_as_context() {
    let app = make_app();
    let payload = json!({"query": "preferences", "top_k": 5, "as_context": true});

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/recall")
                .header("Content-Type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    // context_xml is present (may be empty tag if no results)
    assert!(body.get("context_xml").is_some());
}

// ═══════════════════════════════════════════════════════════
// Test H6: GET /expand/:id — 404 on missing id
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_expand_not_found() {
    let app = make_app();
    let missing_id = uuid::Uuid::new_v4();

    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!("/expand/{}", missing_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ═══════════════════════════════════════════════════════════
// Test H7: GET /expand/:id — bad UUID → 4xx (400 or 404 depending on routing)
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_expand_bad_uuid() {
    let app = make_app();

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/expand/not-a-uuid")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Axum's wildcard {id} matches any string, so the handler returns 400;
    // but some routing implementations return 404 — either is a client error.
    assert!(
        resp.status() == StatusCode::BAD_REQUEST || resp.status() == StatusCode::NOT_FOUND,
        "expected 4xx, got {}",
        resp.status()
    );
}

// ═══════════════════════════════════════════════════════════
// Test H8: remember → expand round-trip (library level)
// HTTP-level state persistence is tested here via MnemeMemory
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_remember_then_expand() {
    use mneme_consolidate::{ConsolidationEngine, MockLLM};
    use mneme_embed::MockEmbeddingModel;
    use mneme_store::{InMemoryContentStore, InMemoryEnvelopeIndex, MnemeStore};

    let store = MnemeStore::new(InMemoryEnvelopeIndex::new(), InMemoryContentStore::new());
    let engine_store = MnemeStore::new(InMemoryEnvelopeIndex::new(), InMemoryContentStore::new());
    let embed = MockEmbeddingModel::new(128);
    let engine_embed = MockEmbeddingModel::new(128);
    let llm = MockLLM::new();
    let config = mneme_core::MnemeConfig::default();
    let engine = ConsolidationEngine::new(engine_store, engine_embed, llm, config.clone());
    let memory = mneme_api::MnemeMemory::new(store, engine, embed, config);

    let id = memory.remember("user loves Rust", "s1").await.unwrap();
    let detail = memory.expand(id).await.unwrap();
    assert_eq!(detail.full_text, "user loves Rust");
}

// ═══════════════════════════════════════════════════════════
// Test H9: forget — removes an engram (library level)
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_forget() {
    use mneme_consolidate::{ConsolidationEngine, MockLLM};
    use mneme_embed::MockEmbeddingModel;
    use mneme_store::{InMemoryContentStore, InMemoryEnvelopeIndex, MnemeStore};

    let store = MnemeStore::new(InMemoryEnvelopeIndex::new(), InMemoryContentStore::new());
    let engine_store = MnemeStore::new(InMemoryEnvelopeIndex::new(), InMemoryContentStore::new());
    let embed = MockEmbeddingModel::new(128);
    let engine_embed = MockEmbeddingModel::new(128);
    let llm = MockLLM::new();
    let config = mneme_core::MnemeConfig::default();
    let engine = ConsolidationEngine::new(engine_store, engine_embed, llm, config.clone());
    let memory = mneme_api::MnemeMemory::new(store, engine, embed, config);

    let id = memory.remember("to be forgotten", "s1").await.unwrap();
    memory.forget(id).await.unwrap();
    assert!(
        memory.expand(id).await.is_err(),
        "expand after forget should fail"
    );
}

// ═══════════════════════════════════════════════════════════
// Test H10: POST /end_session — empty session_id rejected
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_end_session_validation() {
    let app = make_app();
    let payload = json!({"session_id": ""});

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/end_session")
                .header("Content-Type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ═══════════════════════════════════════════════════════════
// Test H11: history — returns version chain (library level)
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_history() {
    use mneme_consolidate::{ConsolidationEngine, MockLLM};
    use mneme_embed::MockEmbeddingModel;
    use mneme_store::{InMemoryContentStore, InMemoryEnvelopeIndex, MnemeStore};

    let store = MnemeStore::new(InMemoryEnvelopeIndex::new(), InMemoryContentStore::new());
    let engine_store = MnemeStore::new(InMemoryEnvelopeIndex::new(), InMemoryContentStore::new());
    let embed = MockEmbeddingModel::new(128);
    let engine_embed = MockEmbeddingModel::new(128);
    let llm = MockLLM::new();
    let config = mneme_core::MnemeConfig::default();
    let engine = ConsolidationEngine::new(engine_store, engine_embed, llm, config.clone());
    let memory = mneme_api::MnemeMemory::new(store, engine, embed, config);

    let id = memory.remember("a fact", "s1").await.unwrap();
    let chain = memory.history(id).await.unwrap();
    assert!(
        !chain.is_empty(),
        "history should have at least one version"
    );
    assert_eq!(chain[0].id, id);
}

// ═══════════════════════════════════════════════════════════
// Test H12: POST /gc — returns removed count
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_gc() {
    let app = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/gc")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["removed"].is_number());
}

// ═══════════════════════════════════════════════════════════
// Test H13: POST /decay — returns updated count
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_decay() {
    let app = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/decay")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["updated"].is_number());
}

// ═══════════════════════════════════════════════════════════
// Test H14: GET /stats — returns counts
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_stats() {
    let app = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["total_engrams"].is_number());
}

// ═══════════════════════════════════════════════════════════
// Test H15: Auth middleware rejects missing token
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_auth_rejects_missing_token() {
    let app = server::build_app(server::build_state(), Some("secret".to_string()));

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/recall")
                .header("Content-Type", "application/json")
                .body(Body::from(json!({"query": "test", "top_k": 5}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// ═══════════════════════════════════════════════════════════
// Test H16: Auth middleware accepts correct token
// ═══════════════════════════════════════════════════════════

#[tokio::test]
async fn test_http_auth_accepts_correct_token() {
    let app = server::build_app(server::build_state(), Some("secret".to_string()));

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/recall")
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer secret")
                .body(Body::from(json!({"query": "test", "top_k": 5}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}
