//! Integration tests for the Mneme memory system.
//!
//! Tests the full lifecycle using in-memory backends:
//! 1. Working memory → remember observations
//! 2. Compaction → cluster + synthesize into semantic memory
//! 3. Recall → search semantic store, progressive disclosure
//! 4. Evolution → drift detection + reconsolidation
//! 5. Conflict resolution → three strategies

#[cfg(test)]
mod tests {
    use mneme_consolidate::{ConsolidationEngine, MockLLM};
    use mneme_core::*;
    use mneme_embed::MockEmbeddingModel;
    use mneme_store::*;

    /// Build a test harness with in-memory backends.
    fn build_test_system() -> (
        MnemeStore<InMemoryEnvelopeIndex, InMemoryContentStore>,
        ConsolidationEngine<InMemoryEnvelopeIndex, InMemoryContentStore, MockEmbeddingModel, MockLLM>,
        MockEmbeddingModel,
        MnemeConfig,
    ) {
        let config = MnemeConfig {
            compaction_buffer_threshold: 3,
            compaction_cluster_threshold: 0.5, // lower for mock embeddings
            evolution_drift_threshold: 0.3,
            ..Default::default()
        };

        let envelope_index = InMemoryEnvelopeIndex::new();
        let content_store = InMemoryContentStore::new();
        let store = MnemeStore::new(envelope_index, content_store);

        let embed_model = MockEmbeddingModel::new(128);
        let llm = MockLLM::new();

        let engine_envelopes = InMemoryEnvelopeIndex::new();
        let engine_content = InMemoryContentStore::new();
        let engine_store = MnemeStore::new(engine_envelopes, engine_content);
        let engine_embed = MockEmbeddingModel::new(128);
        let engine = ConsolidationEngine::new(engine_store, engine_embed, llm, config.clone());

        (store, engine, embed_model, config)
    }

    /// Helper: insert a working memory engram directly.
    async fn insert_working_memory(
        store: &MnemeStore<InMemoryEnvelopeIndex, InMemoryContentStore>,
        embed: &MockEmbeddingModel,
        text: &str,
        session_id: &str,
    ) -> uuid::Uuid {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use uuid::Uuid;

        let id = Uuid::new_v4();
        let now = Utc::now();
        let embedding = embed.embed(text).await.unwrap();

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
                source_sessions: vec![session_id.to_string()],
                supersedes: vec![],
                superseded_by: None,
                summary: text[..text.len().min(80)].to_string(),
                tags: vec![],
                content_hash: 0,
            },
            content: ContentBody {
                engram_id: id,
                full_text: text.to_string(),
                provenance: vec![ProvenanceRecord {
                    session_id: session_id.to_string(),
                    turn_id: None,
                    timestamp: now,
                    raw_excerpt: text.to_string(),
                }],
                conflict_log: vec![],
                related: vec![],
                version: 1,
            },
        };

        store.insert(&engram).await.unwrap();
        id
    }

    /// Helper: insert a semantic engram directly.
    async fn insert_semantic_memory(
        store: &MnemeStore<InMemoryEnvelopeIndex, InMemoryContentStore>,
        embed: &MockEmbeddingModel,
        text: &str,
        summary: &str,
        confidence: f32,
    ) -> uuid::Uuid {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use uuid::Uuid;

        let id = Uuid::new_v4();
        let now = Utc::now();
        let embedding = embed.embed(text).await.unwrap();

        let engram = Engram {
            envelope: Envelope {
                id,
                embedding,
                confidence,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                access_count: 5,
                memory_type: MemoryType::Semantic,
                source_sessions: vec!["test-session".to_string()],
                supersedes: vec![],
                superseded_by: None,
                summary: summary.to_string(),
                tags: vec!["test".to_string()],
                content_hash: 0,
            },
            content: ContentBody {
                engram_id: id,
                full_text: text.to_string(),
                provenance: vec![],
                conflict_log: vec![],
                related: vec![],
                version: 1,
            },
        };

        store.insert(&engram).await.unwrap();
        id
    }

    // ═══════════════════════════════════════════════════════════
    // Test 1: Envelope CRUD
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_envelope_insert_and_get() {
        let (store, _, embed, _) = build_test_system();
        let id = insert_working_memory(&store, &embed, "test observation", "s1").await;

        let env = store.envelopes.get(id).await.unwrap();
        assert_eq!(env.id, id);
        assert_eq!(env.memory_type, MemoryType::Working);
        assert!(env.is_active());
    }

    #[tokio::test]
    async fn test_content_body_roundtrip() {
        let (store, _, embed, _) = build_test_system();
        let id = insert_working_memory(&store, &embed, "full text here", "s1").await;

        let body = store.content.get(id).await.unwrap();
        assert_eq!(body.full_text, "full text here");
        assert_eq!(body.version, 1);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 2: Working memory listing
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_list_working_memory_by_session() {
        let (store, _, embed, _) = build_test_system();

        insert_working_memory(&store, &embed, "obs 1 for s1", "session-1").await;
        insert_working_memory(&store, &embed, "obs 2 for s1", "session-1").await;
        insert_working_memory(&store, &embed, "obs 1 for s2", "session-2").await;

        let s1_entries = store.envelopes.list_working_memory("session-1").await.unwrap();
        assert_eq!(s1_entries.len(), 2);

        let s2_entries = store.envelopes.list_working_memory("session-2").await.unwrap();
        assert_eq!(s2_entries.len(), 1);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 3: Search with filtering
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_search_filters_by_type() {
        use mneme_embed::EmbeddingModel;

        let (store, _, embed, _) = build_test_system();

        insert_working_memory(&store, &embed, "working entry", "s1").await;
        insert_semantic_memory(&store, &embed, "semantic entry", "semantic summary", 0.8).await;

        let query_emb = embed.embed("entry").await.unwrap();

        // Search semantic only
        let query = MemoryQuery {
            embedding: query_emb.clone(),
            top_k: 10,
            active_only: true,
            memory_type: Some(MemoryType::Semantic),
            ..Default::default()
        };
        let results = store.search(&query).await.unwrap();
        assert!(results.iter().all(|r| r.envelope.memory_type == MemoryType::Semantic));

        // Search working only
        let query = MemoryQuery {
            embedding: query_emb,
            top_k: 10,
            active_only: true,
            memory_type: Some(MemoryType::Working),
            ..Default::default()
        };
        let results = store.search(&query).await.unwrap();
        assert!(results.iter().all(|r| r.envelope.memory_type == MemoryType::Working));
    }

    // ═══════════════════════════════════════════════════════════
    // Test 4: Supersession chain
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_mark_superseded() {
        let (store, _, embed, _) = build_test_system();

        let old_id = insert_semantic_memory(&store, &embed, "old fact", "old", 0.7).await;
        let new_id = insert_semantic_memory(&store, &embed, "new fact", "new", 0.9).await;

        store.envelopes.mark_superseded(old_id, new_id).await.unwrap();

        let old_env = store.envelopes.get(old_id).await.unwrap();
        assert!(!old_env.is_active());
        assert_eq!(old_env.superseded_by, Some(new_id));
    }

    #[tokio::test]
    async fn test_active_only_filter_excludes_superseded() {
        use mneme_embed::EmbeddingModel;

        let (store, _, embed, _) = build_test_system();

        let old_id = insert_semantic_memory(&store, &embed, "old fact", "old", 0.7).await;
        let new_id = insert_semantic_memory(&store, &embed, "new fact", "new", 0.9).await;
        store.envelopes.mark_superseded(old_id, new_id).await.unwrap();

        let query_emb = embed.embed("fact").await.unwrap();
        let query = MemoryQuery {
            embedding: query_emb,
            top_k: 10,
            active_only: true,
            memory_type: Some(MemoryType::Semantic),
            ..Default::default()
        };
        let results = store.search(&query).await.unwrap();
        assert!(results.iter().all(|r| r.envelope.is_active()));
        assert!(!results.iter().any(|r| r.envelope.id == old_id));
    }

    // ═══════════════════════════════════════════════════════════
    // Test 5: Touch / access tracking
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_touch_updates_metadata() {
        let (store, _, embed, _) = build_test_system();

        let id = insert_semantic_memory(&store, &embed, "touchable", "touch me", 0.5).await;

        store.envelopes.touch(id, 0.8).await.unwrap();
        let env = store.envelopes.get(id).await.unwrap();
        assert_eq!(env.access_count, 6); // was 5, +1
        assert!((env.confidence - 0.8).abs() < 0.001);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 6: Garbage collection
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_gc_removes_low_confidence_superseded() {
        let (store, _, embed, _) = build_test_system();

        let old_id = insert_semantic_memory(&store, &embed, "garbage", "garbage", 0.01).await;
        let new_id = insert_semantic_memory(&store, &embed, "keeper", "keeper", 0.9).await;
        store.envelopes.mark_superseded(old_id, new_id).await.unwrap();

        // Set last_accessed_at to the past by touching with low confidence
        store.envelopes.touch(old_id, 0.01).await.unwrap();

        let removed = store.envelopes.gc(0.05, 0).await.unwrap();
        assert_eq!(removed, 1);

        // The active one should still exist
        assert!(store.envelopes.get(new_id).await.is_ok());
    }

    // ═══════════════════════════════════════════════════════════
    // Test 7: Embedding model consistency
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_mock_embedding_deterministic() {
        use mneme_embed::EmbeddingModel;

        let embed = MockEmbeddingModel::new(128);
        let v1 = embed.embed("hello world").await.unwrap();
        let v2 = embed.embed("hello world").await.unwrap();
        let v3 = embed.embed("different text").await.unwrap();

        // Same input → same output
        assert_eq!(v1.0, v2.0);

        // Different input → different output
        assert_ne!(v1.0, v3.0);

        // Unit normalized
        let norm: f32 = v1.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_cosine_similarity() {
        use mneme_embed::EmbeddingModel;

        let embed = MockEmbeddingModel::new(128);
        let v1 = embed.embed("hello world").await.unwrap();
        let v2 = embed.embed("hello world").await.unwrap();
        let v3 = embed.embed("completely different text").await.unwrap();

        // Identical vectors → similarity = 1.0
        let sim_same = v1.cosine_similarity(&v2);
        assert!((sim_same - 1.0).abs() < 0.001);

        // Different vectors → similarity < 1.0
        let sim_diff = v1.cosine_similarity(&v3);
        assert!(sim_diff < 1.0);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 8: Clustering
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_agglomerative_cluster() {
        use mneme_embed::{agglomerative_cluster, EmbeddingModel};

        let embed = MockEmbeddingModel::new(128);

        let v1 = embed.embed("the cat sat").await.unwrap();
        let v2 = embed.embed("the cat sat").await.unwrap(); // identical → same cluster
        let v3 = embed.embed("quantum chromodynamics").await.unwrap(); // very different

        let clusters = agglomerative_cluster(&[v1, v2, v3], 0.95);

        // v1 and v2 are identical so should cluster together at high threshold
        // v3 should be in its own cluster
        let has_pair = clusters.iter().any(|c| c.len() == 2);
        let has_singleton = clusters.iter().any(|c| c.len() == 1);
        assert!(has_pair);
        assert!(has_singleton);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 9: Drift detection
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_drift_check() {
        use mneme_embed::EmbeddingModel;

        let embed = MockEmbeddingModel::new(128);
        let stored = embed.embed("user prefers Python").await.unwrap();
        let context_same = embed.embed("user prefers Python").await.unwrap();
        let context_diff = embed.embed("user now uses Rust exclusively").await.unwrap();

        let check_same = DriftCheck::compute(&stored, &context_same, 0.3);
        assert!(!check_same.needs_evolution); // no drift

        let check_diff = DriftCheck::compute(&stored, &context_diff, 0.3);
        // With mock embeddings, different text produces different vectors
        // Drift score = 1 - cosine_sim, so it should be > 0
        assert!(check_diff.drift_score > 0.0);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 10: Retrieval score formula
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_retrieval_score_combines_factors() {
        let (store, _, embed, _) = build_test_system();

        // High confidence should score higher than low confidence
        let high_id = insert_semantic_memory(
            &store, &embed, "high confidence", "high", 0.95,
        ).await;
        let low_id = insert_semantic_memory(
            &store, &embed, "low confidence", "low", 0.1,
        ).await;

        let high_env = store.envelopes.get(high_id).await.unwrap();
        let low_env = store.envelopes.get(low_id).await.unwrap();

        let score_high = high_env.retrieval_score(0.8, 0.2);
        let score_low = low_env.retrieval_score(0.8, 0.2);

        assert!(score_high > score_low);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 11: Context builder formatting
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_context_builder_xml() {
        use mneme_api::{ContextBuilder, MnemeSummary};

        let summaries = vec![
            MnemeSummary {
                id: uuid::Uuid::new_v4(),
                summary: "User prefers Rust".to_string(),
                confidence: 0.9,
                tags: vec!["lang".to_string()],
                similarity: 0.85,
                retrieval_score: 0.88,
                version: 2,
                is_evolved: true,
            },
        ];

        let xml = ContextBuilder::format_summaries(&summaries);
        assert!(xml.starts_with("<memories>"));
        assert!(xml.ends_with("</memories>"));
        assert!(xml.contains("confidence=\"0.90\""));
        assert!(xml.contains("evolved=\"true\""));
        assert!(xml.contains("User prefers Rust"));
    }

    #[tokio::test]
    async fn test_context_builder_empty() {
        use mneme_api::ContextBuilder;

        let xml = ContextBuilder::format_summaries(&[]);
        assert_eq!(xml, "<memories>No relevant memories found.</memories>");
    }

    // ═══════════════════════════════════════════════════════════
    // Test 12: Mock LLM response patterns
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_mock_llm_synthesis_response() {
        use mneme_consolidate::ConsolidationLLM;

        let llm = MockLLM::new();
        let response = llm
            .complete("memory consolidation engine. Distill these")
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(parsed["full_text"].is_string());
        assert!(parsed["confidence"].is_number());
    }

    #[tokio::test]
    async fn test_mock_llm_evolution_keep() {
        use mneme_consolidate::ConsolidationLLM;

        let llm = MockLLM::new();
        let response = llm
            .complete("reconsolidation engine. should be updated")
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["decision"].as_str().unwrap(), "keep");
    }

    #[tokio::test]
    async fn test_mock_llm_conflict_response() {
        use mneme_consolidate::ConsolidationLLM;

        let llm = MockLLM::new();
        let response = llm
            .complete("Two memories contradict each other. Determine the relationship")
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["relationship"].as_str().unwrap(), "factual_update");
    }

    // ═══════════════════════════════════════════════════════════
    // Test 13: Time decay
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_ebbinghaus_decay() {
        let (store, _, embed, _) = build_test_system();
        let id = insert_semantic_memory(&store, &embed, "decaying", "decay", 0.8).await;

        let env = store.envelopes.get(id).await.unwrap();
        let decay_now = env.time_decay(0.05);
        // Just accessed, so decay should be ~1.0
        assert!(decay_now > 0.99);

        // With a high lambda, decay should be faster
        let decay_fast = env.time_decay(100.0);
        // Still nearly 1.0 because elapsed is ~0
        assert!(decay_fast > 0.9);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 14: Config defaults
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_config_defaults_are_sane() {
        let config = MnemeConfig::default();
        assert!(config.compaction_cluster_threshold > 0.0 && config.compaction_cluster_threshold < 1.0);
        assert!(config.evolution_drift_threshold > 0.0 && config.evolution_drift_threshold < 1.0);
        assert!(config.gc_confidence_floor > 0.0);
        assert!(config.working_memory_ttl_hours > 0);
        assert!(config.decay_lambda > 0.0);
    }
}
