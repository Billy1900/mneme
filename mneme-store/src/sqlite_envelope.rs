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
use std::sync::Mutex;
use uuid::Uuid;

use crate::{EnvelopeIndex, StoreError};

pub struct SqliteEnvelopeIndex {
    conn: Mutex<Connection>,
}

impl SqliteEnvelopeIndex {
    pub fn new(path: &str) -> Result<Self, StoreError> {
        let conn = Connection::open(path)
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS envelopes (
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
            conn: Mutex::new(conn),
        })
    }

    /// Open an in-memory SQLite database (for testing).
    pub fn in_memory() -> Result<Self, StoreError> {
        Self::new(":memory:")
    }

    fn envelope_to_row(env: &Envelope) -> EnvelopeRow {
        EnvelopeRow {
            id: env.id.to_string(),
            embedding: embedding_to_bytes(&env.embedding),
            confidence: env.confidence,
            created_at: env.created_at.to_rfc3339(),
            updated_at: env.updated_at.to_rfc3339(),
            last_accessed_at: env.last_accessed_at.to_rfc3339(),
            access_count: env.access_count as i64,
            memory_type: format!("{:?}", env.memory_type).to_lowercase(),
            source_sessions: serde_json::to_string(&env.source_sessions)
                .unwrap_or_default(),
            supersedes: serde_json::to_string(&env.supersedes).unwrap_or_default(),
            superseded_by: env.superseded_by.map(|id| id.to_string()),
            summary: env.summary.clone(),
            tags: serde_json::to_string(&env.tags).unwrap_or_default(),
            content_hash: env.content_hash as i64,
        }
    }

    fn row_to_envelope(row: &EnvelopeRow) -> Result<Envelope, StoreError> {
        let id = Uuid::parse_str(&row.id)
            .map_err(|e| StoreError::Serialization(e.to_string()))?;

        let memory_type = match row.memory_type.as_str() {
            "working" => MemoryType::Working,
            "semantic" => MemoryType::Semantic,
            _ => {
                return Err(StoreError::Serialization(format!(
                    "unknown memory type: {}",
                    row.memory_type
                )))
            }
        };

        Ok(Envelope {
            id,
            embedding: bytes_to_embedding(&row.embedding),
            confidence: row.confidence,
            created_at: chrono::DateTime::parse_from_rfc3339(&row.created_at)
                .map_err(|e| StoreError::Serialization(e.to_string()))?
                .with_timezone(&Utc),
            updated_at: chrono::DateTime::parse_from_rfc3339(&row.updated_at)
                .map_err(|e| StoreError::Serialization(e.to_string()))?
                .with_timezone(&Utc),
            last_accessed_at: chrono::DateTime::parse_from_rfc3339(&row.last_accessed_at)
                .map_err(|e| StoreError::Serialization(e.to_string()))?
                .with_timezone(&Utc),
            access_count: row.access_count as u64,
            memory_type,
            source_sessions: serde_json::from_str(&row.source_sessions)
                .unwrap_or_default(),
            supersedes: serde_json::from_str(&row.supersedes).unwrap_or_default(),
            superseded_by: row
                .superseded_by
                .as_ref()
                .and_then(|s| Uuid::parse_str(s).ok()),
            summary: row.summary.clone(),
            tags: serde_json::from_str(&row.tags).unwrap_or_default(),
            content_hash: row.content_hash as u64,
        })
    }
}

struct EnvelopeRow {
    id: String,
    embedding: Vec<u8>,
    confidence: f32,
    created_at: String,
    updated_at: String,
    last_accessed_at: String,
    access_count: i64,
    memory_type: String,
    source_sessions: String,
    supersedes: String,
    superseded_by: Option<String>,
    summary: String,
    tags: String,
    content_hash: i64,
}

/// Serialize f32 vector to bytes (little-endian, no overhead).
fn embedding_to_bytes(vec: &EmbeddingVec) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vec.0.len() * 4);
    for &v in &vec.0 {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Deserialize bytes back to f32 vector.
fn bytes_to_embedding(bytes: &[u8]) -> EmbeddingVec {
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    EmbeddingVec(floats)
}

#[async_trait]
impl EnvelopeIndex for SqliteEnvelopeIndex {
    async fn upsert(&self, envelope: &Envelope) -> Result<(), StoreError> {
        let row = Self::envelope_to_row(envelope);
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO envelopes
                (id, embedding, confidence, created_at, updated_at,
                 last_accessed_at, access_count, memory_type,
                 source_sessions, supersedes, superseded_by,
                 summary, tags, content_hash)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                row.id,
                row.embedding,
                row.confidence,
                row.created_at,
                row.updated_at,
                row.last_accessed_at,
                row.access_count,
                row.memory_type,
                row.source_sessions,
                row.supersedes,
                row.superseded_by,
                row.summary,
                row.tags,
                row.content_hash,
            ],
        )
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
        Ok(())
    }

    async fn search(&self, query: &MemoryQuery) -> Result<Vec<RetrievalResult>, StoreError> {
        let conn = self.conn.lock().unwrap();

        // Build WHERE clause for metadata pre-filtering
        let mut conditions = Vec::new();
        let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if query.active_only {
            conditions.push("superseded_by IS NULL".to_string());
        }
        if let Some(ref mt) = query.memory_type {
            let mt_str = match mt {
                MemoryType::Working => "working",
                MemoryType::Semantic => "semantic",
            };
            sql_params.push(Box::new(mt_str.to_string()));
            conditions.push(format!("memory_type = ?{}", sql_params.len()));
        }
        if let Some(min_conf) = query.min_confidence {
            sql_params.push(Box::new(min_conf as f64));
            conditions.push(format!("confidence >= ?{}", sql_params.len()));
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let sql = format!(
            "SELECT id, embedding, confidence, created_at, updated_at,
                    last_accessed_at, access_count, memory_type,
                    source_sessions, supersedes, superseded_by,
                    summary, tags, content_hash
             FROM envelopes {}",
            where_clause
        );

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            sql_params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

        let rows = stmt
            .query_map(params_refs.as_slice(), |row| {
                Ok(EnvelopeRow {
                    id: row.get(0)?,
                    embedding: row.get(1)?,
                    confidence: row.get(2)?,
                    created_at: row.get(3)?,
                    updated_at: row.get(4)?,
                    last_accessed_at: row.get(5)?,
                    access_count: row.get(6)?,
                    memory_type: row.get(7)?,
                    source_sessions: row.get(8)?,
                    supersedes: row.get(9)?,
                    superseded_by: row.get(10)?,
                    summary: row.get(11)?,
                    tags: row.get(12)?,
                    content_hash: row.get(13)?,
                })
            })
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

        let mut results: Vec<RetrievalResult> = Vec::new();
        for row_result in rows {
            let row = row_result.map_err(|e| StoreError::VectorIndex(e.to_string()))?;

            // Tag filtering (done in Rust since SQLite JSON querying is awkward)
            if !query.tags.is_empty() {
                let tags: Vec<String> =
                    serde_json::from_str(&row.tags).unwrap_or_default();
                if !query.tags.iter().all(|t| tags.contains(t)) {
                    continue;
                }
            }

            let envelope = Self::row_to_envelope(&row)?;
            let similarity = envelope.embedding.cosine_similarity(&query.embedding);
            let retrieval_score =
                envelope.retrieval_score(similarity, query.recency_weight);

            results.push(RetrievalResult {
                envelope,
                similarity,
                retrieval_score,
            });
        }

        // Sort by retrieval score descending, take top_k
        results.sort_by(|a, b| {
            b.retrieval_score
                .partial_cmp(&a.retrieval_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(query.top_k);

        Ok(results)
    }

    async fn get(&self, id: Uuid) -> Result<Envelope, StoreError> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT id, embedding, confidence, created_at, updated_at,
                        last_accessed_at, access_count, memory_type,
                        source_sessions, supersedes, superseded_by,
                        summary, tags, content_hash
                 FROM envelopes WHERE id = ?1",
                params![id.to_string()],
                |row| {
                    Ok(EnvelopeRow {
                        id: row.get(0)?,
                        embedding: row.get(1)?,
                        confidence: row.get(2)?,
                        created_at: row.get(3)?,
                        updated_at: row.get(4)?,
                        last_accessed_at: row.get(5)?,
                        access_count: row.get(6)?,
                        memory_type: row.get(7)?,
                        source_sessions: row.get(8)?,
                        supersedes: row.get(9)?,
                        superseded_by: row.get(10)?,
                        summary: row.get(11)?,
                        tags: row.get(12)?,
                        content_hash: row.get(13)?,
                    })
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => StoreError::NotFound(id),
                _ => StoreError::VectorIndex(e.to_string()),
            })?;

        Self::row_to_envelope(&row)
    }

    async fn get_batch(&self, ids: &[Uuid]) -> Result<Vec<Envelope>, StoreError> {
        // SQLite doesn't have great batch support, so we loop
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            results.push(self.get(*id).await?);
        }
        Ok(results)
    }

    async fn list_working_memory(
        &self,
        session_id: &str,
    ) -> Result<Vec<Envelope>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, embedding, confidence, created_at, updated_at,
                        last_accessed_at, access_count, memory_type,
                        source_sessions, supersedes, superseded_by,
                        summary, tags, content_hash
                 FROM envelopes
                 WHERE memory_type = 'working'
                   AND superseded_by IS NULL
                   AND source_sessions LIKE ?1",
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

        let pattern = format!("%\"{}%", session_id);
        let rows = stmt
            .query_map(params![pattern], |row| {
                Ok(EnvelopeRow {
                    id: row.get(0)?,
                    embedding: row.get(1)?,
                    confidence: row.get(2)?,
                    created_at: row.get(3)?,
                    updated_at: row.get(4)?,
                    last_accessed_at: row.get(5)?,
                    access_count: row.get(6)?,
                    memory_type: row.get(7)?,
                    source_sessions: row.get(8)?,
                    supersedes: row.get(9)?,
                    superseded_by: row.get(10)?,
                    summary: row.get(11)?,
                    tags: row.get(12)?,
                    content_hash: row.get(13)?,
                })
            })
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

        let mut results = Vec::new();
        for row_result in rows {
            let row = row_result.map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            results.push(Self::row_to_envelope(&row)?);
        }
        Ok(results)
    }

    async fn mark_superseded(&self, id: Uuid, successor: Uuid) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE envelopes SET superseded_by = ?1, updated_at = ?2 WHERE id = ?3",
            params![successor.to_string(), now, id.to_string()],
        )
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
        Ok(())
    }

    async fn gc(
        &self,
        confidence_floor: f32,
        older_than_hours: u64,
    ) -> Result<usize, StoreError> {
        let conn = self.conn.lock().unwrap();
        let cutoff =
            (Utc::now() - chrono::Duration::hours(older_than_hours as i64)).to_rfc3339();

        let count = conn
            .execute(
                "DELETE FROM envelopes
                 WHERE confidence < ?1
                   AND last_accessed_at < ?2
                   AND superseded_by IS NOT NULL",
                params![confidence_floor as f64, cutoff],
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

        Ok(count)
    }

    async fn touch(&self, id: Uuid, new_confidence: f32) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE envelopes
             SET last_accessed_at = ?1,
                 access_count = access_count + 1,
                 confidence = ?2
             WHERE id = ?3",
            params![now, new_confidence as f64, id.to_string()],
        )
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
        Ok(())
    }
}
