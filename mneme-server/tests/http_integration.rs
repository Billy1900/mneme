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

/// POST `payload` to `uri` against a running `app`, cloning the router (cheap
/// — shares the same `Arc<AppState>`) so callers can issue multiple requests
/// against one app/store within a single test.
async fn post_json(app: &axum::Router, uri: &str, payload: Value) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("Content-Type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let body = body_json(resp.into_body()).await;
    (status, body)
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

// ═══════════════════════════════════════════════════════════
// Agent Memory Leaderboard contract: POST /add, POST /search
// ═══════════════════════════════════════════════════════════

// Test H17: POST /add — success
#[tokio::test]
async fn test_http_add_success() {
    let app = make_app();
    let (status, body) = post_json(
        &app,
        "/add",
        json!({
            "request_id": "r1",
            "user_id": "alice",
            "session_id": "s1",
            "messages": [{"role": "user", "content": "my favorite language is Rust"}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["success"], true);
    assert_eq!(body["request_id"], "r1");
    assert_eq!(body["user_id"], "alice");
    assert_eq!(body["session_id"], "s1");
}

// Test H18: POST /add — validation rejects empty required fields
#[tokio::test]
async fn test_http_add_validation() {
    let app = make_app();

    let cases = [
        json!({"request_id": "", "user_id": "a", "session_id": "s", "messages": [{"content": "x"}]}),
        json!({"request_id": "r", "user_id": "", "session_id": "s", "messages": [{"content": "x"}]}),
        json!({"request_id": "r", "user_id": "a", "session_id": "", "messages": [{"content": "x"}]}),
        json!({"request_id": "r", "user_id": "a", "session_id": "s", "messages": []}),
    ];
    for payload in cases {
        let (status, _) = post_json(&app, "/add", payload.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "payload: {payload}");
    }
}

// Test H19: POST /add — repeated request_id is deduplicated, not double-written
#[tokio::test]
async fn test_http_add_dedup() {
    let app = make_app();
    let payload = json!({
        "request_id": "dup1",
        "user_id": "bob",
        "session_id": "s1",
        "messages": [{"role": "user", "content": "duplicate-sensitive fact"}],
    });

    let (status1, body1) = post_json(&app, "/add", payload.clone()).await;
    let (status2, body2) = post_json(&app, "/add", payload).await;

    assert_eq!(status1, StatusCode::OK);
    assert_eq!(status2, StatusCode::OK);
    assert_eq!(
        body1, body2,
        "retried request_id should return the same response"
    );

    let (_, search_body) = post_json(
        &app,
        "/search",
        json!({"query": "duplicate-sensitive fact", "user_id": "bob", "top_k": 10}),
    )
    .await;
    assert_eq!(
        search_body["data"].as_array().unwrap().len(),
        1,
        "a retried request_id must not create a second engram"
    );
}

// Test H20: POST /search — user isolation via uid tag
#[tokio::test]
async fn test_http_search_isolation() {
    let app = make_app();
    post_json(
        &app,
        "/add",
        json!({
            "request_id": "iso-a",
            "user_id": "alice",
            "session_id": "s1",
            "messages": [{"role": "user", "content": "alice's secret project codename is FALCON"}],
        }),
    )
    .await;
    post_json(
        &app,
        "/add",
        json!({
            "request_id": "iso-b",
            "user_id": "bob",
            "session_id": "s1",
            "messages": [{"role": "user", "content": "bob's secret project codename is OSPREY"}],
        }),
    )
    .await;

    let (_, alice_results) = post_json(
        &app,
        "/search",
        json!({"query": "secret project codename", "user_id": "alice", "top_k": 10}),
    )
    .await;
    let alice_data = alice_results["data"].as_array().unwrap();
    assert_eq!(alice_data.len(), 1);
    assert!(alice_data[0]["content"]
        .as_str()
        .unwrap()
        .contains("FALCON"));

    let (_, bob_results) = post_json(
        &app,
        "/search",
        json!({"query": "secret project codename", "user_id": "bob", "top_k": 10}),
    )
    .await;
    let bob_data = bob_results["data"].as_array().unwrap();
    assert_eq!(bob_data.len(), 1);
    assert!(bob_data[0]["content"].as_str().unwrap().contains("OSPREY"));
}

// Test H21: POST /search — BM25 lexical channel finds a keyword past the
// summary's ~100-char truncation point, where the vector channel alone
// (over a hash-based MockEmbeddingModel, with no real semantic signal)
// has no reliable way to rank the right memory first.
#[tokio::test]
async fn test_http_search_bm25_full_text_match() {
    let app = make_app();
    let long_message = format!(
        "{}ZORBAXIL{}",
        "padding text to push the keyword past the hundred character mark ".repeat(2),
        " more padding text after the keyword as well for good measure"
    );
    assert!(
        long_message.find("ZORBAXIL").unwrap() > 100,
        "test setup: keyword must be past the summary truncation point"
    );

    post_json(
        &app,
        "/add",
        json!({
            "request_id": "bm25-1",
            "user_id": "carol",
            "session_id": "s1",
            "messages": [{"role": "user", "content": long_message}],
        }),
    )
    .await;

    let (_, results) = post_json(
        &app,
        "/search",
        json!({"query": "ZORBAXIL", "user_id": "carol", "top_k": 5}),
    )
    .await;
    let data = results["data"].as_array().unwrap();
    assert_eq!(data.len(), 1, "BM25 should find the keyword past char 100");
    assert!(data[0]["content"].as_str().unwrap().contains("ZORBAXIL"));
}

// Test H22: POST /search — validation rejects bad input
#[tokio::test]
async fn test_http_search_validation() {
    let app = make_app();

    let cases = [
        json!({"query": "", "user_id": "a", "top_k": 5}),
        json!({"query": "x", "user_id": "", "top_k": 5}),
        json!({"query": "x", "user_id": "a", "top_k": 0}),
        json!({"query": "x", "user_id": "a", "top_k": 101}),
    ];
    for payload in cases {
        let (status, _) = post_json(&app, "/search", payload.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "payload: {payload}");
    }
}

// Test H23: POST /search — response is capped so a flood of long memories
// with a high top_k can't return an unbounded body.
#[tokio::test]
async fn test_http_search_response_budget_cap() {
    let app = make_app();
    let big = "The user described their project in great detail. ".repeat(150); // ~7800 chars
    for i in 0..10 {
        post_json(
            &app,
            "/add",
            json!({
                "request_id": format!("cap-{i}"),
                "user_id": "dave",
                "session_id": "s1",
                "messages": [{"role": "user", "content": format!("{big} (memory {i})")}],
            }),
        )
        .await;
    }

    let (_, results) = post_json(
        &app,
        "/search",
        json!({"query": "project detail", "user_id": "dave", "top_k": 100}),
    )
    .await;
    let data = results["data"].as_array().unwrap();
    assert!(
        data.len() < 10,
        "budget cap should drop lower-ranked results, not return all 10"
    );
    let total_chars: usize = data
        .iter()
        .map(|d| d["content"].as_str().unwrap().len())
        .sum();
    assert!(total_chars <= 24_000, "total content chars: {total_chars}");
    for item in data {
        assert!(item["content"].as_str().unwrap().len() <= 4_003);
    }
}
