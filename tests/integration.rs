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

    fn build_test_system() -> (
        MnemeStore<InMemoryEnvelopeIndex, InMemoryContentStore>,
        ConsolidationEngine<
            InMemoryEnvelopeIndex,
            InMemoryContentStore,
            MockEmbeddingModel,
            MockLLM,
        >,
        MockEmbeddingModel,
        MnemeConfig,
    ) {
        let config = MnemeConfig {
            compaction_buffer_threshold: 3,
            compaction_cluster_threshold: 0.5,
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
        let engine =
            ConsolidationEngine::new(engine_store, engine_embed, llm, config.clone());

        (store, engine, embed_model, config)
    }

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
    // Test 1: Basic working memory insert + retrieval
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_remember_and_retrieve_working_memory() {
        use mneme_embed::EmbeddingModel;

        let (store, _, embed, _) = build_test_system();

        let id = insert_working_memory(&store, &embed, "user prefers dark mode", "s1").await;

        let env = store.envelopes.get(id).await.unwrap();
        assert_eq!(env.memory_type, MemoryType::Working);
        assert_eq!(env.source_sessions, vec!["s1"]);

        let wm = store.envelopes.list_working_memory("s1").await.unwrap();
        assert_eq!(wm.len(), 1);
        assert_eq!(wm[0].id, id);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 2: Content body round-trip
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_content_body_roundtrip() {
        let (store, _, embed, _) = build_test_system();

        let id =
            insert_working_memory(&store, &embed, "detailed observation text", "s2").await;

        let body = store.content.get(id).await.unwrap();
        assert_eq!(body.full_text, "detailed observation text");
        assert_eq!(body.version, 1);
        assert_eq!(body.provenance.len(), 1);
        assert_eq!(body.provenance[0].session_id, "s2");
    }

    // ═══════════════════════════════════════════════════════════
    // Test 3: Memory type filter in search
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_memory_type_filter() {
        use mneme_embed::EmbeddingModel;

        let (store, _, embed, _) = build_test_system();

        insert_working_memory(&store, &embed, "working memory item", "s3").await;
        insert_semantic_memory(&store, &embed, "semantic memory item", "semantic item", 0.8)
            .await;

        let query_emb = embed.embed("memory").await.unwrap();

        // Search semantic only
        let query = MemoryQuery {
            embedding: query_emb.clone(),
            top_k: 10,
            active_only: true,
            memory_type: Some(MemoryType::Semantic),
            ..Default::default()
        };
        let results = store.search(&query).await.unwrap();
        assert!(results
            .iter()
            .all(|r| r.envelope.memory_type == MemoryType::Semantic));

        // Search working only
        let query = MemoryQuery {
            embedding: query_emb,
            top_k: 10,
            active_only: true,
            memory_type: Some(MemoryType::Working),
            ..Default::default()
        };
        let results = store.search(&query).await.unwrap();
        assert!(results
            .iter()
            .all(|r| r.envelope.memory_type == MemoryType::Working));
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

    // ═══════════════════════════════════════════════════════════
    // Test 5: active_only filter excludes superseded
    // ═══════════════════════════════════════════════════════════

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
    // Test 6: Touch / access tracking
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_touch_updates_metadata() {
        let (store, _, embed, _) = build_test_system();

        let id = insert_semantic_memory(&store, &embed, "touchable", "touch me", 0.5).await;

        store.envelopes.touch(id, 0.8).await.unwrap();
        let env = store.envelopes.get(id).await.unwrap();
        assert_eq!(env.access_count, 6); // was 5 (from insert_semantic_memory), +1
        assert_eq!(env.confidence, 0.8);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 7: GC removes low-confidence superseded engrams
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_gc_removes_low_confidence_superseded() {
        let (store, _, embed, _) = build_test_system();

        let old_id =
            insert_semantic_memory(&store, &embed, "stale memory", "stale", 0.04).await;
        let new_id = insert_semantic_memory(&store, &embed, "fresh memory", "fresh", 0.9).await;
        store.envelopes.mark_superseded(old_id, new_id).await.unwrap();

        let removed = store
            .envelopes
            .gc(0.05, 24 * 365 * 10) // high TTL so only confidence filter triggers
            .await
            .unwrap();

        assert_eq!(removed, 1);
        assert!(store.envelopes.get(old_id).await.is_err());
        assert!(store.envelopes.get(new_id).await.is_ok());
    }

    // ═══════════════════════════════════════════════════════════
    // Test 8: Clustering
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_agglomerative_cluster() {
        use mneme_embed::{agglomerative_cluster, EmbeddingModel};

        let embed = MockEmbeddingModel::new(128);

        let v1 = embed.embed("the cat sat").await.unwrap();
        let v2 = embed.embed("the cat sat").await.unwrap(); // identical
        let v3 = embed.embed("quantum chromodynamics").await.unwrap();

        let clusters = agglomerative_cluster(&[v1, v2, v3], 0.95);

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
        let context_diff = embed
            .embed("quantum mechanics wave function collapse")
            .await
            .unwrap();

        let check_same = DriftCheck::compute(&stored, &context_same, 0.3);
        assert!(!check_same.needs_evolution);

        let check_diff = DriftCheck::compute(&stored, &context_diff, 0.3);
        assert!(check_diff.drift_score > 0.0);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 10: Cosine similarity properties
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_cosine_similarity() {
        use mneme_embed::EmbeddingModel;

        let embed = MockEmbeddingModel::new(128);
        let v1 = embed.embed("identical text").await.unwrap();
        let v2 = embed.embed("identical text").await.unwrap();
        let v3 = embed.embed("completely different xyzzy").await.unwrap();

        let sim_same = v1.cosine_similarity(&v2);
        assert!((sim_same - 1.0).abs() < 0.001);

        let sim_diff = v1.cosine_similarity(&v3);
        assert!(sim_diff < 1.0);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 11: Conflict record append
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_append_conflict_record() {
        use chrono::Utc;

        let (store, _, embed, _) = build_test_system();
        let id =
            insert_semantic_memory(&store, &embed, "contested fact", "contested", 0.7).await;
        let other_id =
            insert_semantic_memory(&store, &embed, "conflicting fact", "conflicting", 0.7)
                .await;

        let record = ConflictRecord {
            conflicting_id: other_id,
            resolution: ConflictStrategy::TemporalSupersede,
            resolved_at: Utc::now(),
            resolver_notes: "newer evidence wins".to_string(),
        };

        store.content.append_conflict(id, record).await.unwrap();

        let body = store.content.get(id).await.unwrap();
        assert_eq!(body.conflict_log.len(), 1);
    }

    // ═══════════════════════════════════════════════════════════
    // Test 12: Content body delete
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_content_delete() {
        let (store, _, embed, _) = build_test_system();
        let id = insert_working_memory(&store, &embed, "temp memory", "s1").await;

        store.content.delete(id).await.unwrap();
        assert!(store.content.get(id).await.is_err());
    }

    // ═══════════════════════════════════════════════════════════
    // Test 13: Tags filter in search
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_tags_filter() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use uuid::Uuid;

        let (store, _, embed, _) = build_test_system();
        let embedding = embed.embed("tagged memory").await.unwrap();
        let id = Uuid::new_v4();
        let now = Utc::now();

        let engram = Engram {
            envelope: Envelope {
                id,
                embedding: embedding.clone(),
                confidence: 0.8,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                access_count: 0,
                memory_type: MemoryType::Semantic,
                source_sessions: vec!["s1".to_string()],
                supersedes: vec![],
                superseded_by: None,
                summary: "tagged memory".to_string(),
                tags: vec!["rust".to_string(), "systems".to_string()],
                content_hash: 0,
            },
            content: ContentBody {
                engram_id: id,
                full_text: "tagged memory".to_string(),
                provenance: vec![],
                conflict_log: vec![],
                related: vec![],
                version: 1,
            },
        };
        store.insert(&engram).await.unwrap();

        // Matching tag
        let q = MemoryQuery {
            embedding: embedding.clone(),
            top_k: 10,
            active_only: true,
            tags: vec!["rust".to_string()],
            ..Default::default()
        };
        let results = store.search(&q).await.unwrap();
        assert!(results.iter().any(|r| r.envelope.id == id));

        // Non-matching tag
        let q2 = MemoryQuery {
            embedding,
            top_k: 10,
            active_only: true,
            tags: vec!["python".to_string()],
            ..Default::default()
        };
        let results2 = store.search(&q2).await.unwrap();
        assert!(!results2.iter().any(|r| r.envelope.id == id));
    }

    // ═══════════════════════════════════════════════════════════
    // Test 14: Stats
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_stats() {
        let (store, _, embed, _) = build_test_system();

        insert_working_memory(&store, &embed, "wm entry", "s1").await;
        insert_semantic_memory(&store, &embed, "sm entry", "semantic", 0.8).await;

        let stats = store.envelopes.stats().await.unwrap();
        assert_eq!(stats.total_engrams, 2);
        assert_eq!(stats.working_memory_count, 1);
        assert_eq!(stats.semantic_memory_count, 1);
        assert_eq!(stats.superseded_count, 0);
    }

    // ═══════════════════════════════════════════════════════════
    // REGRESSION TEST for FIX #1: shared store — engine sees API writes
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_shared_store_engine_sees_api_writes() {
        use mneme_store::new_shared_memory_store;

        // Create shared backends
        let (shared_envelopes, shared_content) = new_shared_memory_store();

        // Server-side store (what the HTTP handler writes to)
        let server_store = MnemeStore::new(
            (*shared_envelopes).clone(),
            (*shared_content).clone(),
        );

        // Engine-side store (uses the SAME Arc clones — FIX #1)
        let engine_store = MnemeStore::new(
            (*shared_envelopes).clone(),
            (*shared_content).clone(),
        );

        let embed = MockEmbeddingModel::new(128);
        let engine_embed = MockEmbeddingModel::new(128);
        let llm = MockLLM::new();
        let config = MnemeConfig {
            compaction_cluster_threshold: 0.5,
            ..Default::default()
        };
        let engine = ConsolidationEngine::new(engine_store, engine_embed, llm, config);

        // Write via "server" path
        let id1 =
            insert_working_memory(&server_store, &embed, "observation A", "shared-session")
                .await;
        let id2 =
            insert_working_memory(&server_store, &embed, "observation B", "shared-session")
                .await;

        // Engine should see both entries — the old bug would have found 0
        let wm = engine
            .store
            .envelopes
            .list_working_memory("shared-session")
            .await
            .unwrap();
        assert_eq!(
            wm.len(),
            2,
            "Engine must see entries written by server (shared Arc backends)"
        );
        assert!(wm.iter().any(|e| e.id == id1));
        assert!(wm.iter().any(|e| e.id == id2));

        // Compaction should now be able to find and process those entries
        let compacted = engine.compact_session("shared-session").await.unwrap();
        assert!(
            !compacted.is_empty(),
            "Compaction should produce at least one semantic engram"
        );
    }

    // ═══════════════════════════════════════════════════════════
    // REGRESSION TEST for FIX #8: exact session_id match, no false positives
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_session_id_exact_match_no_false_positives() {
        let (store, _, embed, _) = build_test_system();

        // Insert entries for "session-1" and "session-12"
        // The old LIKE-based query would return session-1 results when querying session-12
        insert_working_memory(&store, &embed, "entry for session-1", "session-1").await;
        insert_working_memory(&store, &embed, "entry for session-12", "session-12").await;

        // Query for "session-1" should return exactly 1 result
        let s1_results = store
            .envelopes
            .list_working_memory("session-1")
            .await
            .unwrap();
        assert_eq!(
            s1_results.len(),
            1,
            "session-1 query must return exactly 1 result, not accidentally match session-12"
        );
        assert_eq!(s1_results[0].source_sessions, vec!["session-1"]);

        // Query for "session-12" should return exactly 1 result
        let s12_results = store
            .envelopes
            .list_working_memory("session-12")
            .await
            .unwrap();
        assert_eq!(
            s12_results.len(),
            1,
            "session-12 query must not bleed into session-1"
        );
        assert_eq!(s12_results[0].source_sessions, vec!["session-12"]);
    }
}