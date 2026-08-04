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
    use mneme_consolidate::{ConsolidateError, ConsolidationEngine, MockLLM};
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
        let engine = ConsolidationEngine::new(engine_store, engine_embed, llm, config.clone());

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

        let id = insert_working_memory(&store, &embed, "detailed observation text", "s2").await;

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
        insert_semantic_memory(&store, &embed, "semantic memory item", "semantic item", 0.8).await;

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

        store
            .envelopes
            .mark_superseded(old_id, new_id)
            .await
            .unwrap();

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
        store
            .envelopes
            .mark_superseded(old_id, new_id)
            .await
            .unwrap();

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

        let old_id = insert_semantic_memory(&store, &embed, "stale memory", "stale", 0.04).await;
        let new_id = insert_semantic_memory(&store, &embed, "fresh memory", "fresh", 0.9).await;
        store
            .envelopes
            .mark_superseded(old_id, new_id)
            .await
            .unwrap();

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
        let id = insert_semantic_memory(&store, &embed, "contested fact", "contested", 0.7).await;
        let other_id =
            insert_semantic_memory(&store, &embed, "conflicting fact", "conflicting", 0.7).await;

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
        let server_store = MnemeStore::new((*shared_envelopes).clone(), (*shared_content).clone());

        // Engine-side store (uses the SAME Arc clones — FIX #1)
        let engine_store = MnemeStore::new((*shared_envelopes).clone(), (*shared_content).clone());

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
            insert_working_memory(&server_store, &embed, "observation A", "shared-session").await;
        let id2 =
            insert_working_memory(&server_store, &embed, "observation B", "shared-session").await;

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
    // REGRESSION TEST: compaction must carry tags forward.
    //
    // Found via a real end-to-end LoCoMo run: synthesize_cluster() built the
    // new Semantic engram's tags purely from the LLM's own JSON output
    // (multi-item clusters) or an empty vec (single-item clusters) — the
    // uid:{user_id} isolation tag on the source Working entries was
    // silently dropped. Every engram compaction touched became invisible
    // to tag-filtered /search, with zero errors anywhere — a live but
    // silent recall bug in exactly the isolation mechanism /add and
    // /search depend on.
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_compaction_preserves_tags() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use uuid::Uuid;

        let (envelopes, content) = (InMemoryEnvelopeIndex::new(), InMemoryContentStore::new());
        let store = MnemeStore::new(envelopes.clone(), content.clone());
        let embed = MockEmbeddingModel::new(128);
        let engine_embed = MockEmbeddingModel::new(128);
        // MockLLM's canned response includes tags: ["mock"] — deliberately
        // NOT "uid:alice", so this reproduces the exact failure mode: the
        // LLM's own tag suggestions must be unioned with, not replace, the
        // carried-forward isolation tag.
        let llm = MockLLM::new();
        let config = MnemeConfig {
            // Identical text below already gives cosine similarity 1.0, but
            // keep the threshold generous so this isn't sensitive to
            // MockEmbeddingModel implementation details.
            compaction_cluster_threshold: 0.5,
            ..Default::default()
        };
        let engine = ConsolidationEngine::new(store, engine_embed, llm, config);

        // Two entries with identical text (guarantees they cluster into one
        // group under MockEmbeddingModel's hash-based embeddings) and the
        // uid:alice isolation tag.
        let text = "user prefers dark mode";
        for _ in 0..2 {
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
                    source_sessions: vec!["s1".to_string()],
                    supersedes: vec![],
                    superseded_by: None,
                    summary: text.to_string(),
                    tags: vec!["uid:alice".to_string()],
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
            envelopes.upsert(&engram.envelope).await.unwrap();
            content.put(&engram.content).await.unwrap();
        }

        let compacted = engine.compact_session("s1").await.unwrap();
        assert_eq!(
            compacted.len(),
            1,
            "identical text should cluster into one engram"
        );
        let synthesized = &compacted[0];
        assert!(
            synthesized.envelope.tags.contains(&"uid:alice".to_string()),
            "compaction must carry the isolation tag forward, got tags: {:?}",
            synthesized.envelope.tags
        );

        // The real-world failure mode: a tag-filtered search for this
        // user's data must still find the synthesized engram.
        let query_embedding = embed.embed(text).await.unwrap();
        let results = envelopes
            .search(&MemoryQuery {
                embedding: query_embedding,
                top_k: 10,
                active_only: true,
                tags: vec!["uid:alice".to_string()],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(
            results
                .iter()
                .any(|r| r.envelope.id == synthesized.envelope.id),
            "tag-filtered search must find the post-compaction engram"
        );
    }

    // ═══════════════════════════════════════════════════════════
    // forget() — deletes envelope and content
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_forget_removes_engram() {
        use mneme_api::MnemeMemory;

        let (envelopes, content) = (InMemoryEnvelopeIndex::new(), InMemoryContentStore::new());
        let store = MnemeStore::new(envelopes, content);
        let engine_store =
            MnemeStore::new(InMemoryEnvelopeIndex::new(), InMemoryContentStore::new());
        let embed = MockEmbeddingModel::new(128);
        let engine_embed = MockEmbeddingModel::new(128);
        let llm = MockLLM::new();
        let config = MnemeConfig::default();
        let engine = ConsolidationEngine::new(engine_store, engine_embed, llm, config.clone());
        let memory = MnemeMemory::new(store, engine, embed, config);

        let id = memory.remember("to be forgotten", "s1").await.unwrap();
        memory.forget(id).await.unwrap();

        assert!(
            memory.expand(id).await.is_err(),
            "expand after forget should fail"
        );
    }

    // ═══════════════════════════════════════════════════════════
    // apply_decay — reduces confidence for old memories
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_decay_reduces_confidence() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use uuid::Uuid;

        let store = InMemoryEnvelopeIndex::new();
        let embed = MockEmbeddingModel::new(128);
        let id = Uuid::new_v4();
        let now = Utc::now();
        // last_accessed 200 hours ago so decay is significant
        let old_access = now - chrono::Duration::hours(200);
        let embedding = embed.embed("decayable memory").await.unwrap();

        let envelope = Envelope {
            id,
            embedding,
            confidence: 1.0,
            created_at: now,
            updated_at: now,
            last_accessed_at: old_access,
            access_count: 0,
            memory_type: MemoryType::Semantic,
            source_sessions: vec!["s1".to_string()],
            supersedes: vec![],
            superseded_by: None,
            summary: "decayable memory".to_string(),
            tags: vec![],
            content_hash: 0,
        };
        store.upsert(&envelope).await.unwrap();

        let updated = store.apply_decay(0.05).await.unwrap();
        assert!(updated >= 1, "at least one engram should be decayed");

        let after = store.get(id).await.unwrap();
        assert!(
            after.confidence < 1.0,
            "confidence should decrease after decay with old last_accessed, got {}",
            after.confidence
        );
    }

    // ═══════════════════════════════════════════════════════════
    // SQLite backend: basic CRUD
    // ═══════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_sqlite_envelope_upsert_and_get() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use mneme_store::{SqliteContentStore, SqliteEnvelopeIndex};
        use uuid::Uuid;

        let idx = SqliteEnvelopeIndex::in_memory().unwrap();
        let embed = MockEmbeddingModel::new(64);
        let id = Uuid::new_v4();
        let now = Utc::now();
        let embedding = embed.embed("sqlite test").await.unwrap();

        let envelope = Envelope {
            id,
            embedding,
            confidence: 0.7,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            access_count: 0,
            memory_type: MemoryType::Semantic,
            source_sessions: vec!["sqlite-session".to_string()],
            supersedes: vec![],
            superseded_by: None,
            summary: "sqlite test memory".to_string(),
            tags: vec!["test".to_string()],
            content_hash: 42,
        };

        idx.upsert(&envelope).await.unwrap();
        let retrieved = idx.get(id).await.unwrap();

        assert_eq!(retrieved.id, id);
        assert!((retrieved.confidence - 0.7).abs() < 0.001);
        assert_eq!(retrieved.memory_type, MemoryType::Semantic);
        assert_eq!(retrieved.summary, "sqlite test memory");
        assert_eq!(retrieved.tags, vec!["test"]);
    }

    #[tokio::test]
    async fn test_sqlite_envelope_search() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use mneme_store::SqliteEnvelopeIndex;
        use uuid::Uuid;

        let idx = SqliteEnvelopeIndex::in_memory().unwrap();
        let embed = MockEmbeddingModel::new(64);
        let now = Utc::now();

        let id = Uuid::new_v4();
        let embedding = embed.embed("vector search test").await.unwrap();
        let envelope = Envelope {
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
            summary: "searchable memory".to_string(),
            tags: vec![],
            content_hash: 0,
        };
        idx.upsert(&envelope).await.unwrap();

        let results = idx
            .search(&MemoryQuery {
                embedding,
                top_k: 5,
                active_only: true,
                memory_type: Some(MemoryType::Semantic),
                min_confidence: Some(0.1),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].envelope.id, id);
    }

    #[tokio::test]
    async fn test_sqlite_envelope_search_tags_filter() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use mneme_store::SqliteEnvelopeIndex;
        use uuid::Uuid;

        let idx = SqliteEnvelopeIndex::in_memory().unwrap();
        let embed = MockEmbeddingModel::new(64);
        let now = Utc::now();

        let id = Uuid::new_v4();
        let embedding = embed.embed("tagged sqlite memory").await.unwrap();
        let envelope = Envelope {
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
            summary: "tagged sqlite memory".to_string(),
            tags: vec!["uid:alice".to_string()],
            content_hash: 0,
        };
        idx.upsert(&envelope).await.unwrap();

        // Matching tag
        let results = idx
            .search(&MemoryQuery {
                embedding: embedding.clone(),
                top_k: 10,
                active_only: true,
                tags: vec!["uid:alice".to_string()],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(results.iter().any(|r| r.envelope.id == id));

        // Non-matching tag (e.g. a different user_id) must not leak this envelope
        let results2 = idx
            .search(&MemoryQuery {
                embedding,
                top_k: 10,
                active_only: true,
                tags: vec!["uid:bob".to_string()],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(!results2.iter().any(|r| r.envelope.id == id));
    }

    // Regression test for the SQLite backend's BM25 lexical channel (FTS5)
    // fused with vector similarity via RRF, mirroring InMemoryEnvelopeIndex.
    // Uses two envelopes with near-identical mock embeddings so a pure
    // vector search can't reliably distinguish them; only the BM25 channel
    // (via `index_full_text`, matching a keyword the summary alone doesn't
    // contain) can pick the right one.
    #[tokio::test]
    async fn test_sqlite_envelope_search_bm25_full_text() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use mneme_store::SqliteEnvelopeIndex;
        use uuid::Uuid;

        let idx = SqliteEnvelopeIndex::in_memory().unwrap();
        let embed = MockEmbeddingModel::new(64);
        let now = Utc::now();

        let target_id = Uuid::new_v4();
        let other_id = Uuid::new_v4();
        let query_embedding = embed.embed("some unrelated query text").await.unwrap();

        for (id, summary) in [
            (target_id, "short summary"),
            (other_id, "a different short summary"),
        ] {
            let envelope = Envelope {
                id,
                embedding: query_embedding.clone(),
                confidence: 0.8,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                access_count: 0,
                memory_type: MemoryType::Working,
                source_sessions: vec!["s1".to_string()],
                supersedes: vec![],
                superseded_by: None,
                summary: summary.to_string(),
                tags: vec![],
                content_hash: 0,
            };
            idx.upsert(&envelope).await.unwrap();
        }

        // Full text far exceeds what's in `summary` — only reachable via
        // `index_full_text`, same pattern as the in-memory backend.
        idx.index_full_text(target_id, "a long message about many topics that eventually mentions the unique keyword QUARTZFALCON near the end")
            .await
            .unwrap();

        let results = idx
            .search(&MemoryQuery {
                embedding: query_embedding,
                top_k: 10,
                active_only: true,
                query_text: "QUARTZFALCON".to_string(),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(
            !results.is_empty(),
            "BM25 channel should find the envelope matching the keyword"
        );
        assert_eq!(
            results[0].envelope.id, target_id,
            "the envelope matching the BM25 keyword should rank first despite identical vectors"
        );
    }

    #[tokio::test]
    async fn test_sqlite_envelope_mark_superseded() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use mneme_store::SqliteEnvelopeIndex;
        use uuid::Uuid;

        let idx = SqliteEnvelopeIndex::in_memory().unwrap();
        let embed = MockEmbeddingModel::new(64);
        let now = Utc::now();

        async fn make_env(
            id: Uuid,
            summary: &str,
            embed: &MockEmbeddingModel,
            now: chrono::DateTime<Utc>,
        ) -> Envelope {
            let emb = embed.embed(summary).await.unwrap();
            Envelope {
                id,
                embedding: emb,
                confidence: 0.7,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                access_count: 0,
                memory_type: MemoryType::Semantic,
                source_sessions: vec!["s1".to_string()],
                supersedes: vec![],
                superseded_by: None,
                summary: summary.to_string(),
                tags: vec![],
                content_hash: 0,
            }
        }

        let old_id = Uuid::new_v4();
        let new_id = Uuid::new_v4();
        idx.upsert(&make_env(old_id, "old fact", &embed, now).await)
            .await
            .unwrap();
        idx.upsert(&make_env(new_id, "new fact", &embed, now).await)
            .await
            .unwrap();
        idx.mark_superseded(old_id, new_id).await.unwrap();

        let old = idx.get(old_id).await.unwrap();
        assert!(!old.is_active());
        assert_eq!(old.superseded_by, Some(new_id));
    }

    #[tokio::test]
    async fn test_sqlite_envelope_gc() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use mneme_store::SqliteEnvelopeIndex;
        use uuid::Uuid;

        let idx = SqliteEnvelopeIndex::in_memory().unwrap();
        let embed = MockEmbeddingModel::new(64);
        let now = Utc::now();

        async fn make_env_with_conf(
            id: Uuid,
            confidence: f32,
            summary: &str,
            embed: &MockEmbeddingModel,
            now: chrono::DateTime<Utc>,
        ) -> Envelope {
            let emb = embed.embed(summary).await.unwrap();
            Envelope {
                id,
                embedding: emb,
                confidence,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                access_count: 0,
                memory_type: MemoryType::Semantic,
                source_sessions: vec!["s1".to_string()],
                supersedes: vec![],
                superseded_by: None,
                summary: summary.to_string(),
                tags: vec![],
                content_hash: 0,
            }
        }

        let stale_id = Uuid::new_v4();
        let fresh_id = Uuid::new_v4();
        idx.upsert(&make_env_with_conf(stale_id, 0.02, "stale", &embed, now).await)
            .await
            .unwrap();
        idx.upsert(&make_env_with_conf(fresh_id, 0.9, "fresh", &embed, now).await)
            .await
            .unwrap();
        idx.mark_superseded(stale_id, fresh_id).await.unwrap();

        let removed = idx.gc(0.05, 24 * 365 * 10).await.unwrap();
        assert_eq!(removed, 1);
        assert!(idx.get(stale_id).await.is_err());
        assert!(idx.get(fresh_id).await.is_ok());
    }

    #[tokio::test]
    async fn test_sqlite_envelope_delete() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use mneme_store::SqliteEnvelopeIndex;
        use uuid::Uuid;

        let idx = SqliteEnvelopeIndex::in_memory().unwrap();
        let embed = MockEmbeddingModel::new(64);
        let id = Uuid::new_v4();
        let now = Utc::now();
        let embedding = embed.embed("deletable").await.unwrap();

        let envelope = Envelope {
            id,
            embedding,
            confidence: 0.5,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            access_count: 0,
            memory_type: MemoryType::Semantic,
            source_sessions: vec!["s1".to_string()],
            supersedes: vec![],
            superseded_by: None,
            summary: "deletable".to_string(),
            tags: vec![],
            content_hash: 0,
        };
        idx.upsert(&envelope).await.unwrap();
        idx.delete(id).await.unwrap();
        assert!(idx.get(id).await.is_err());
    }

    #[tokio::test]
    async fn test_sqlite_envelope_apply_decay() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use mneme_store::SqliteEnvelopeIndex;
        use uuid::Uuid;

        let idx = SqliteEnvelopeIndex::in_memory().unwrap();
        let embed = MockEmbeddingModel::new(64);
        let id = Uuid::new_v4();
        let now = Utc::now();
        let embedding = embed.embed("decayable").await.unwrap();

        let envelope = Envelope {
            id,
            embedding,
            confidence: 1.0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now - chrono::Duration::hours(100),
            access_count: 0,
            memory_type: MemoryType::Semantic,
            source_sessions: vec!["s1".to_string()],
            supersedes: vec![],
            superseded_by: None,
            summary: "decayable".to_string(),
            tags: vec![],
            content_hash: 0,
        };
        idx.upsert(&envelope).await.unwrap();

        let updated = idx.apply_decay(0.05).await.unwrap();
        assert!(updated >= 1);

        let after = idx.get(id).await.unwrap();
        assert!(
            after.confidence < 1.0,
            "confidence should decay, got {}",
            after.confidence
        );
    }

    #[tokio::test]
    async fn test_sqlite_envelope_stats() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use mneme_store::SqliteEnvelopeIndex;
        use uuid::Uuid;

        let idx = SqliteEnvelopeIndex::in_memory().unwrap();
        let embed = MockEmbeddingModel::new(64);
        let now = Utc::now();

        async fn make_typed(
            id: Uuid,
            mt: MemoryType,
            summary: &str,
            embed: &MockEmbeddingModel,
            now: chrono::DateTime<Utc>,
        ) -> Envelope {
            let emb = embed.embed(summary).await.unwrap();
            Envelope {
                id,
                embedding: emb,
                confidence: 0.8,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                access_count: 0,
                memory_type: mt,
                source_sessions: vec!["s1".to_string()],
                supersedes: vec![],
                superseded_by: None,
                summary: summary.to_string(),
                tags: vec![],
                content_hash: 0,
            }
        }

        idx.upsert(&make_typed(Uuid::new_v4(), MemoryType::Working, "wm", &embed, now).await)
            .await
            .unwrap();
        idx.upsert(&make_typed(Uuid::new_v4(), MemoryType::Semantic, "sm", &embed, now).await)
            .await
            .unwrap();

        let stats = idx.stats().await.unwrap();
        assert_eq!(stats.total_engrams, 2);
        assert_eq!(stats.working_memory_count, 1);
        assert_eq!(stats.semantic_memory_count, 1);
    }

    #[tokio::test]
    async fn test_sqlite_content_roundtrip() {
        use chrono::Utc;
        use mneme_store::SqliteContentStore;
        use uuid::Uuid;

        let store = SqliteContentStore::in_memory().unwrap();
        let id = Uuid::new_v4();
        let now = Utc::now();

        let body = ContentBody {
            engram_id: id,
            full_text: "SQLite content test".to_string(),
            provenance: vec![ProvenanceRecord {
                session_id: "s1".to_string(),
                turn_id: None,
                timestamp: now,
                raw_excerpt: "SQLite content test".to_string(),
            }],
            conflict_log: vec![],
            related: vec![],
            version: 1,
        };

        store.put(&body).await.unwrap();
        let retrieved = store.get(id).await.unwrap();
        assert_eq!(retrieved.full_text, "SQLite content test");
        assert_eq!(retrieved.provenance.len(), 1);
        assert_eq!(retrieved.version, 1);
    }

    #[tokio::test]
    async fn test_sqlite_content_delete() {
        use chrono::Utc;
        use mneme_store::SqliteContentStore;
        use uuid::Uuid;

        let store = SqliteContentStore::in_memory().unwrap();
        let id = Uuid::new_v4();
        let body = ContentBody {
            engram_id: id,
            full_text: "to delete".to_string(),
            provenance: vec![],
            conflict_log: vec![],
            related: vec![],
            version: 1,
        };
        store.put(&body).await.unwrap();
        store.delete(id).await.unwrap();
        assert!(store.get(id).await.is_err());
    }

    #[tokio::test]
    async fn test_sqlite_session_id_exact_match() {
        use chrono::Utc;
        use mneme_embed::EmbeddingModel;
        use mneme_store::SqliteEnvelopeIndex;
        use uuid::Uuid;

        let idx = SqliteEnvelopeIndex::in_memory().unwrap();
        let embed = MockEmbeddingModel::new(64);
        let now = Utc::now();

        async fn make_working(
            session: &str,
            summary: &str,
            embed: &MockEmbeddingModel,
            now: chrono::DateTime<Utc>,
        ) -> Envelope {
            let emb = embed.embed(summary).await.unwrap();
            Envelope {
                id: Uuid::new_v4(),
                embedding: emb,
                confidence: 0.5,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                access_count: 0,
                memory_type: MemoryType::Working,
                source_sessions: vec![session.to_string()],
                supersedes: vec![],
                superseded_by: None,
                summary: summary.to_string(),
                tags: vec![],
                content_hash: 0,
            }
        }

        idx.upsert(&make_working("session-A", "entry A", &embed, now).await)
            .await
            .unwrap();
        idx.upsert(&make_working("session-AB", "entry AB", &embed, now).await)
            .await
            .unwrap();

        let results_a = idx.list_working_memory("session-A").await.unwrap();
        assert_eq!(results_a.len(), 1, "session-A must not match session-AB");

        let results_ab = idx.list_working_memory("session-AB").await.unwrap();
        assert_eq!(results_ab.len(), 1, "session-AB must not match session-A");
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

    // ═══════════════════════════════════════════════════════════
    // Fact extraction (write path)
    // ═══════════════════════════════════════════════════════════

    /// An LLM that always returns one canned response, so extraction parsing
    /// can be tested against the exact shapes real models emit.
    struct ScriptedLLM(String);

    #[async_trait::async_trait]
    impl mneme_consolidate::ConsolidationLLM for ScriptedLLM {
        async fn complete(&self, _prompt: &str) -> Result<String, ConsolidateError> {
            Ok(self.0.clone())
        }
    }

    fn engine_with_llm(
        response: &str,
    ) -> ConsolidationEngine<
        InMemoryEnvelopeIndex,
        InMemoryContentStore,
        MockEmbeddingModel,
        ScriptedLLM,
    > {
        let store = MnemeStore::new(InMemoryEnvelopeIndex::new(), InMemoryContentStore::new());
        ConsolidationEngine::new(
            store,
            MockEmbeddingModel::new(128),
            ScriptedLLM(response.to_string()),
            MnemeConfig::default(),
        )
    }

    #[tokio::test]
    async fn test_extract_facts_parses_fields() {
        let engine = engine_with_llm(
            r#"[{"text": "Melanie adopted a golden retriever named Cooper.", "date": "May 2023", "subject": "Melanie", "relation": "adopted", "object": "Cooper"},
                {"text": "Melanie lives in Denver.", "date": null, "subject": null, "relation": null, "object": null}]"#,
        );

        let facts = engine.extract_facts("...").await.unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(
            facts[0].text,
            "Melanie adopted a golden retriever named Cooper."
        );
        assert_eq!(facts[0].date.as_deref(), Some("May 2023"));
        assert_eq!(facts[1].date, None);
    }

    #[tokio::test]
    async fn test_extract_facts_tolerates_code_fence() {
        // Models wrap JSON in a fence regardless of prompt wording, and an
        // unhandled fence would parse to zero facts — silently degrading to
        // "this batch asserted nothing" rather than erroring.
        let engine = engine_with_llm(
            "```json\n[{\"text\": \"Cooper is a golden retriever.\", \"date\": null, \"subject\": \"Cooper\", \"relation\": \"is a\", \"object\": \"golden retriever\"}]\n```",
        );

        let facts = engine.extract_facts("...").await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].text, "Cooper is a golden retriever.");
    }

    #[tokio::test]
    async fn test_extract_facts_unparseable_response_is_empty_not_error() {
        // /add treats extraction as best-effort on top of already-durable raw
        // turns, so a garbage response must not surface as a failed write.
        let engine = engine_with_llm("I'm sorry, I can't help with that.");
        assert!(engine.extract_facts("...").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_extracted_fact_to_triplet() {
        let engine = engine_with_llm(
            r#"[{"text": "Melanie works at Acme.", "date": null, "subject": "Melanie", "relation": "works at", "object": "Acme"},
                {"text": "It was a nice day.", "date": null, "subject": null, "relation": null, "object": null},
                {"text": "Blank subject.", "date": null, "subject": "  ", "relation": "x", "object": "y"}]"#,
        );

        let facts = engine.extract_facts("...").await.unwrap();
        let source = uuid::Uuid::new_v4();
        let triplets: Vec<_> = facts.iter().filter_map(|f| f.as_triplet(source)).collect();

        // Only the fact with a complete, non-blank decomposition becomes an
        // edge — a partial triplet would poison graph traversal.
        assert_eq!(triplets.len(), 1);
        assert_eq!(triplets[0].subject, "Melanie");
        assert_eq!(triplets[0].object, "Acme");
        assert_eq!(triplets[0].source_engram_id, source);
    }

    #[tokio::test]
    async fn test_graph_all_round_trips_for_snapshot() {
        let graph = InMemoryGraphIndex::new();
        let source = uuid::Uuid::new_v4();
        graph
            .insert(vec![GraphTriplet {
                subject: "Melanie".into(),
                relation: "works at".into(),
                object: "Acme".into(),
                date: None,
                source_engram_id: source,
            }])
            .await
            .unwrap();

        // What the snapshot writes must be re-insertable into a fresh graph
        // and still traversable — otherwise a restart silently loses every
        // edge and multi-hop recall degrades to vector search alone.
        let restored = InMemoryGraphIndex::new();
        restored.insert(graph.all().await.unwrap()).await.unwrap();
        assert_eq!(restored.neighbors("Melanie", 1).await.unwrap().len(), 1);
    }
}
