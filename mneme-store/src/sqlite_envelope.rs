//! SQLite-backed envelope index.
//!
//! Stores envelopes in a single SQLite database with embedding vectors
//! as BLOB columns. Search is brute-force cosine similarity over all
//! active envelopes, with pre-filtering via SQL WHERE clauses on metadata.
//!
//! Good for single-node deployments up to ~100K engrams.
//! Beyond that, use the Qdrant backend for proper ANN indexing.


use async_trait::async_trait;
use chrono::Utc;
use mneme_core::*;
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::{EnvelopeIndex, StoreError, StoreStats};

#[derive(Clone)]
pub struct SqliteEnvelopeIndex {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteEnvelopeIndex {
    pub fn new(path: &str) -> Result<Self, StoreError> {
        let conn = Connection::open(path)
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;

             CREATE TABLE IF NOT EXISTS envelopes (
                 id TEXT PRIMARY KEY,
                 embedding BLOB NOT NULL,
                 confidence REAL NOT NULL DEFAULT 0.5,
                 created_at TEXT NOT NULL,
                 updated_at TEXT NOT NULL,
                 last_accessed_at TEXT NOT NULL,
                 access_count INTEGER NOT NULL DEFAULT 0,
                 memory_type TEXT NOT NULL,
                 source_sessions TEXT NOT NULL DEFAULT '[]',
                 supersedes TEXT NOT NULL DEFAULT '[]',
                 superseded_by TEXT,
                 summary TEXT NOT NULL,
                 tags TEXT NOT NULL DEFAULT '[]',
                 content_hash INTEGER NOT NULL DEFAULT 0
             );

             CREATE INDEX IF NOT EXISTS idx_envelopes_memory_type
                 ON envelopes(memory_type);
             CREATE INDEX IF NOT EXISTS idx_envelopes_superseded_by
                 ON envelopes(superseded_by);
             CREATE INDEX IF NOT EXISTS idx_envelopes_confidence
                 ON envelopes(confidence);",
        )
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn in_memory() -> Result<Self, StoreError> {
        Self::new(":memory:")
    }
}

fn embedding_to_bytes(emb: &EmbeddingVec) -> Vec<u8> {
    emb.0
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

fn bytes_to_embedding(bytes: &[u8]) -> EmbeddingVec {
    let floats = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    EmbeddingVec(floats)
}

#[async_trait]
impl EnvelopeIndex for SqliteEnvelopeIndex {
    async fn upsert(&self, envelope: &Envelope) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let env = envelope.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let emb_bytes = embedding_to_bytes(&env.embedding);
            let sessions_json = serde_json::to_string(&env.source_sessions).unwrap();
            let supersedes_json = serde_json::to_string(&env.supersedes)
                .unwrap_or_else(|_| "[]".to_string());
            let tags_json = serde_json::to_string(&env.tags).unwrap();
            let superseded_by = env.superseded_by.map(|id| id.to_string());
            conn.execute(
                "INSERT OR REPLACE INTO envelopes
                 (id, embedding, confidence, created_at, updated_at, last_accessed_at,
                  access_count, memory_type, source_sessions, supersedes, superseded_by,
                  summary, tags, content_hash)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
                params![
                    env.id.to_string(),
                    emb_bytes,
                    env.confidence,
                    env.created_at.to_rfc3339(),
                    env.updated_at.to_rfc3339(),
                    env.last_accessed_at.to_rfc3339(),
                    env.access_count as i64,
                    format!("{:?}", env.memory_type),
                    sessions_json,
                    supersedes_json,
                    superseded_by,
                    env.summary,
                    tags_json,
                    env.content_hash as i64,
                ],
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            Ok::<_, StoreError>(())
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))??;
        Ok(())
    }

    async fn search(&self, query: &MemoryQuery) -> Result<Vec<RetrievalResult>, StoreError> {
        let conn = Arc::clone(&self.conn);
        let query = query.clone();
        let rows = tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT id, embedding, confidence, created_at, updated_at,
                            last_accessed_at, access_count, memory_type, source_sessions,
                            supersedes, superseded_by, summary, tags, content_hash
                     FROM envelopes
                     WHERE (?1 IS NULL OR superseded_by IS NULL)
                       AND (?2 IS NULL OR memory_type = ?2)
                       AND (?3 IS NULL OR confidence >= ?3)",
                )
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

            let active_filter: Option<&str> = if query.active_only { Some("1") } else { None };
            let type_filter: Option<String> = query.memory_type.map(|t| format!("{:?}", t));
            let conf_filter: Option<f32> = query.min_confidence;

            let rows: Vec<(String, Vec<u8>, f32, String, String, String, i64, String, String, String, Option<String>, String, String, i64)> = stmt
                .query_map(
                    params![active_filter, type_filter, conf_filter],
                    |row| Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, f32>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, i64>(6)?,
                        row.get::<_, String>(7)?,
                        row.get::<_, String>(8)?,
                        row.get::<_, String>(9)?,
                        row.get::<_, Option<String>>(10)?,
                        row.get::<_, String>(11)?,
                        row.get::<_, String>(12)?,
                        row.get::<_, i64>(13)?,
                    )),
                )
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            Ok::<_, StoreError>(rows)
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))??;

        let mut results: Vec<RetrievalResult> = rows
            .into_iter()
            .map(|(id, emb_bytes, confidence, created_at, updated_at, last_accessed_at,
                   access_count, memory_type_str, sessions_json, supersedes_json,
                   superseded_by_str, summary, tags_json, content_hash)| {
                let embedding = bytes_to_embedding(&emb_bytes);
                let similarity = embedding.cosine_similarity(&query.embedding);
                let env = Envelope {
                    id: id.parse().unwrap(),
                    embedding,
                    confidence,
                    created_at: created_at.parse().unwrap_or_else(|_| Utc::now()),
                    updated_at: updated_at.parse().unwrap_or_else(|_| Utc::now()),
                    last_accessed_at: last_accessed_at.parse().unwrap_or_else(|_| Utc::now()),
                    access_count: access_count as u64,
                    memory_type: if memory_type_str.contains("Working") {
                        MemoryType::Working
                    } else {
                        MemoryType::Semantic
                    },
                    source_sessions: serde_json::from_str(&sessions_json).unwrap_or_default(),
                    supersedes: serde_json::from_str(&supersedes_json).unwrap_or_default(),
                    superseded_by: superseded_by_str.and_then(|s| s.parse().ok()),
                    summary,
                    tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                    content_hash: content_hash as u64,
                };
                let recency = env.time_decay(0.05) as f32;
                let retrieval_score =
                    (1.0 - query.recency_weight) * similarity + query.recency_weight * recency;
                RetrievalResult { envelope: env, similarity, retrieval_score }
            })
            .collect();

        results.sort_by(|a, b| b.retrieval_score.partial_cmp(&a.retrieval_score).unwrap());
        results.truncate(query.top_k);
        Ok(results)
    }

    async fn get(&self, id: Uuid) -> Result<Envelope, StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.query_row(
                "SELECT id, embedding, confidence, created_at, updated_at, last_accessed_at,
                         access_count, memory_type, source_sessions, supersedes, superseded_by,
                         summary, tags, content_hash
                  FROM envelopes WHERE id = ?1",
                params![id.to_string()],
                |row| {
                    let emb_bytes: Vec<u8> = row.get(1)?;
                    Ok(Envelope {
                        id: row.get::<_, String>(0)?.parse().unwrap(),
                        embedding: bytes_to_embedding(&emb_bytes),
                        confidence: row.get(2)?,
                        created_at: row.get::<_, String>(3)?.parse().unwrap_or_else(|_| Utc::now()),
                        updated_at: row.get::<_, String>(4)?.parse().unwrap_or_else(|_| Utc::now()),
                        last_accessed_at: row.get::<_, String>(5)?.parse().unwrap_or_else(|_| Utc::now()),
                        access_count: row.get::<_, i64>(6)? as u64,
                        memory_type: if row.get::<_, String>(7)?.contains("Working") {
                            MemoryType::Working
                        } else {
                            MemoryType::Semantic
                        },
                        source_sessions: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
                        supersedes: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
                        superseded_by: row.get::<_, Option<String>>(10)?.and_then(|s| s.parse().ok()),
                        summary: row.get(11)?,
                        tags: serde_json::from_str(&row.get::<_, String>(12)?).unwrap_or_default(),
                        content_hash: row.get::<_, i64>(13)? as u64,
                    })
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => StoreError::NotFound(id),
                _ => StoreError::VectorIndex(e.to_string()),
            })
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?
    }

    async fn get_batch(&self, ids: &[Uuid]) -> Result<Vec<Envelope>, StoreError> {
        // FIX #14: single spawn_blocking with internal loop
        let conn = Arc::clone(&self.conn);
        let ids = ids.to_vec();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut results = Vec::with_capacity(ids.len());
            for id in ids {
                let env = conn.query_row(
                    "SELECT id, embedding, confidence, created_at, updated_at, last_accessed_at,
                             access_count, memory_type, source_sessions, supersedes, superseded_by,
                             summary, tags, content_hash
                      FROM envelopes WHERE id = ?1",
                    params![id.to_string()],
                    |row| {
                        let emb_bytes: Vec<u8> = row.get(1)?;
                        Ok(Envelope {
                            id: row.get::<_, String>(0)?.parse().unwrap(),
                            embedding: bytes_to_embedding(&emb_bytes),
                            confidence: row.get(2)?,
                            created_at: row.get::<_, String>(3)?.parse().unwrap_or_else(|_| Utc::now()),
                            updated_at: row.get::<_, String>(4)?.parse().unwrap_or_else(|_| Utc::now()),
                            last_accessed_at: row.get::<_, String>(5)?.parse().unwrap_or_else(|_| Utc::now()),
                            access_count: row.get::<_, i64>(6)? as u64,
                            memory_type: if row.get::<_, String>(7)?.contains("Working") {
                                MemoryType::Working
                            } else {
                                MemoryType::Semantic
                            },
                            source_sessions: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
                            supersedes: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
                            superseded_by: row.get::<_, Option<String>>(10)?.and_then(|s| s.parse().ok()),
                            summary: row.get(11)?,
                            tags: serde_json::from_str(&row.get::<_, String>(12)?).unwrap_or_default(),
                            content_hash: row.get::<_, i64>(13)? as u64,
                        })
                    },
                )
                .map_err(|e| match e {
                    rusqlite::Error::QueryReturnedNoRows => StoreError::NotFound(id),
                    _ => StoreError::VectorIndex(e.to_string()),
                })?;
                results.push(env);
            }
            Ok::<_, StoreError>(results)
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?
    }

    // FIX #8: json_each for exact session_id match
    async fn list_working_memory(&self, session_id: &str) -> Result<Vec<Envelope>, StoreError> {
        let conn = Arc::clone(&self.conn);
        let session_id = session_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT e.id, e.embedding, e.confidence, e.created_at, e.updated_at,
                        e.last_accessed_at, e.access_count, e.memory_type, e.source_sessions,
                        e.supersedes, e.superseded_by, e.summary, e.tags, e.content_hash
                 FROM envelopes e, json_each(e.source_sessions) s
                 WHERE e.memory_type = 'Working'
                   AND s.value = ?1",
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

            let rows: Vec<Envelope> = stmt
                .query_map(params![session_id], |row| {
                    let emb_bytes: Vec<u8> = row.get(1)?;
                    Ok(Envelope {
                        id: row.get::<_, String>(0)?.parse().unwrap(),
                        embedding: bytes_to_embedding(&emb_bytes),
                        confidence: row.get(2)?,
                        created_at: row.get::<_, String>(3)?.parse().unwrap_or_else(|_| Utc::now()),
                        updated_at: row.get::<_, String>(4)?.parse().unwrap_or_else(|_| Utc::now()),
                        last_accessed_at: row.get::<_, String>(5)?.parse().unwrap_or_else(|_| Utc::now()),
                        access_count: row.get::<_, i64>(6)? as u64,
                        memory_type: MemoryType::Working,
                        source_sessions: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
                        supersedes: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
                        superseded_by: row.get::<_, Option<String>>(10)?.and_then(|s| s.parse().ok()),
                        summary: row.get(11)?,
                        tags: serde_json::from_str(&row.get::<_, String>(12)?).unwrap_or_default(),
                        content_hash: row.get::<_, i64>(13)? as u64,
                    })
                })
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            Ok::<_, StoreError>(rows)
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?
    }

    async fn mark_superseded(&self, id: Uuid, successor: Uuid) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "UPDATE envelopes SET superseded_by = ?1, updated_at = ?2 WHERE id = ?3",
                params![successor.to_string(), Utc::now().to_rfc3339(), id.to_string()],
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            Ok::<_, StoreError>(())
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?
    }

    async fn gc(&self, confidence_floor: f32, older_than_hours: u64) -> Result<usize, StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let cutoff = Utc::now() - chrono::Duration::hours(older_than_hours as i64);
            let removed = conn.execute(
                "DELETE FROM envelopes
                 WHERE (memory_type = 'Working' AND created_at < ?1)
                    OR (confidence < ?2 AND superseded_by IS NOT NULL)",
                params![cutoff.to_rfc3339(), confidence_floor],
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            Ok::<_, StoreError>(removed)
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?
    }

    async fn touch(&self, id: Uuid, new_confidence: f32) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "UPDATE envelopes SET access_count = access_count + 1,
                  last_accessed_at = ?1, confidence = ?2 WHERE id = ?3",
                params![Utc::now().to_rfc3339(), new_confidence, id.to_string()],
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            Ok::<_, StoreError>(())
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?
    }

    async fn stats(&self) -> Result<StoreStats, StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let total: usize = conn
                .query_row("SELECT COUNT(*) FROM envelopes", [], |r| r.get(0))
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            let working: usize = conn
                .query_row("SELECT COUNT(*) FROM envelopes WHERE memory_type = 'Working'", [], |r| r.get(0))
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            let semantic: usize = conn
                .query_row("SELECT COUNT(*) FROM envelopes WHERE memory_type = 'Semantic'", [], |r| r.get(0))
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            let superseded: usize = conn
                .query_row("SELECT COUNT(*) FROM envelopes WHERE superseded_by IS NOT NULL", [], |r| r.get(0))
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            let avg_confidence: f32 = if total == 0 {
                0.0
            } else {
                conn.query_row("SELECT AVG(confidence) FROM envelopes", [], |r| r.get(0))
                    .map_err(|e| StoreError::VectorIndex(e.to_string()))?
            };
            Ok::<_, StoreError>(StoreStats {
                total_engrams: total,
                working_memory_count: working,
                semantic_memory_count: semantic,
                superseded_count: superseded,
                avg_confidence,
            })
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?
    }
}