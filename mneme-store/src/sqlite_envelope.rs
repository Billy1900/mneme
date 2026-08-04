//! SQLite-backed envelope index.
//!
//! Stores envelopes in a single SQLite database with embedding vectors
//! as BLOB columns. Search is brute-force cosine similarity over all
//! active envelopes, with pre-filtering via SQL WHERE clauses on metadata.
//!
//! Good for single-node deployments up to ~100K engrams.
//! Beyond that, use the Qdrant backend for proper ANN indexing.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use mneme_core::*;
use rusqlite::{params, params_from_iter, Connection};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::{EnvelopeIndex, StoreError, StoreStats};

#[derive(Clone)]
pub struct SqliteEnvelopeIndex {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteEnvelopeIndex {
    pub fn new(path: &str) -> Result<Self, StoreError> {
        let conn = Connection::open(path).map_err(|e| StoreError::VectorIndex(e.to_string()))?;

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
                 content_hash INTEGER NOT NULL DEFAULT 0,
                 -- Valid time (when the fact was true in the world), as
                 -- opposed to the transaction time already in created_at/
                 -- updated_at. NULL valid_at means true from the beginning;
                 -- NULL invalid_at means still true. See Envelope::is_valid_at.
                 valid_at TEXT,
                 invalid_at TEXT
             );

             CREATE INDEX IF NOT EXISTS idx_envelopes_invalid_at
                 ON envelopes(invalid_at);
             CREATE INDEX IF NOT EXISTS idx_envelopes_memory_type
                 ON envelopes(memory_type);
             CREATE INDEX IF NOT EXISTS idx_envelopes_superseded_by
                 ON envelopes(superseded_by);
             CREATE INDEX IF NOT EXISTS idx_envelopes_confidence
                 ON envelopes(confidence);

             -- Relational tag index: lets tag-filtered search() narrow via
             -- an indexed lookup instead of pulling every row and filtering
             -- in the app layer, matching InMemoryEnvelopeIndex's tag_index.
             CREATE TABLE IF NOT EXISTS envelope_tags (
                 envelope_id TEXT NOT NULL,
                 tag TEXT NOT NULL,
                 PRIMARY KEY (envelope_id, tag)
             );
             CREATE INDEX IF NOT EXISTS idx_envelope_tags_tag
                 ON envelope_tags(tag);

             -- BM25 lexical channel, backed by SQLite's native FTS5 module
             -- (works with the plain `bundled` rusqlite feature — this
             -- SQLite build already has FTS5 compiled in) instead of a
             -- hand-rolled index — matches InMemoryEnvelopeIndex's BM25
             -- channel's role (fused with vector similarity via RRF in
             -- search()), giving the SQLite backend feature parity.
             CREATE VIRTUAL TABLE IF NOT EXISTS envelope_fts USING fts5(
                 id UNINDEXED,
                 text,
                 tokenize = 'porter unicode61'
             );",
        )
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

        // `CREATE TABLE IF NOT EXISTS` is a no-op on a database that predates
        // the valid-time columns, so opening an existing store would leave
        // them missing and every query referencing them would fail. Add them
        // if absent — both are nullable with no default, so existing rows
        // read back as "valid for all time", which is the correct reading of
        // "we never recorded a validity window for this fact".
        let existing: Vec<String> = {
            let mut stmt = conn
                .prepare("PRAGMA table_info(envelopes)")
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            let cols = stmt
                .query_map([], |row| row.get::<_, String>(1))
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            cols
        };
        for column in ["valid_at", "invalid_at"] {
            if !existing.iter().any(|c| c == column) {
                conn.execute_batch(&format!("ALTER TABLE envelopes ADD COLUMN {column} TEXT;"))
                    .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            }
        }

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn in_memory() -> Result<Self, StoreError> {
        Self::new(":memory:")
    }

    /// Re-index `id` for BM25 using `text` in place of whatever was indexed
    /// for it before (typically the truncated `summary` indexed by
    /// `upsert`). Does not touch the envelope itself — call after `upsert`.
    /// Mirrors `InMemoryEnvelopeIndex::index_full_text`.
    pub async fn index_full_text(&self, id: Uuid, text: &str) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let id_str = id.to_string();
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute("DELETE FROM envelope_fts WHERE id = ?1", params![id_str])
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            conn.execute(
                "INSERT INTO envelope_fts (id, text) VALUES (?1, ?2)",
                params![id_str, text],
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            Ok::<_, StoreError>(())
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?
    }
}

/// Build an FTS5 MATCH query that matches any of `text`'s words (OR
/// semantics, mirroring the in-memory BM25 channel's per-term scoring
/// rather than requiring every word to be present). Each word is quoted as
/// a literal phrase so FTS5 query-syntax characters in the input (`"`,
/// `*`, `-`, `:`, ...) can't be interpreted as operators.
fn fts_match_query(text: &str) -> Option<String> {
    let terms: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| format!("\"{}\"", s.replace('"', "\"\"")))
        .collect();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

fn embedding_to_bytes(emb: &EmbeddingVec) -> Vec<u8> {
    emb.0.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Parse a stored RFC3339 valid-time column. An unparseable value is treated
/// as absent rather than as "now" — a bad timestamp must not silently make a
/// fact look currently-invalid (or currently-valid) when we don't know.
fn parse_opt_time(raw: Option<String>) -> Option<DateTime<Utc>> {
    raw.and_then(|s| s.parse::<DateTime<Utc>>().ok())
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
            let id_str = env.id.to_string();
            let emb_bytes = embedding_to_bytes(&env.embedding);
            let sessions_json = serde_json::to_string(&env.source_sessions).unwrap();
            let supersedes_json =
                serde_json::to_string(&env.supersedes).unwrap_or_else(|_| "[]".to_string());
            let tags_json = serde_json::to_string(&env.tags).unwrap();
            let superseded_by = env.superseded_by.map(|id| id.to_string());
            conn.execute(
                "INSERT OR REPLACE INTO envelopes
                 (id, embedding, confidence, created_at, updated_at, last_accessed_at,
                  access_count, memory_type, source_sessions, supersedes, superseded_by,
                  summary, tags, content_hash, valid_at, invalid_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
                params![
                    id_str,
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
                    env.valid_at.map(|t| t.to_rfc3339()),
                    env.invalid_at.map(|t| t.to_rfc3339()),
                ],
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

            // Re-upserting an existing id can change its tags — drop the old
            // associations before adding the new ones, same reasoning as
            // InMemoryEnvelopeIndex's tag_index maintenance.
            conn.execute(
                "DELETE FROM envelope_tags WHERE envelope_id = ?1",
                params![id_str],
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            for tag in &env.tags {
                conn.execute(
                    "INSERT OR IGNORE INTO envelope_tags (envelope_id, tag) VALUES (?1, ?2)",
                    params![id_str, tag],
                )
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            }

            // Baseline lexical coverage from `summary` (always present).
            // Callers with the full untruncated text should follow up with
            // `index_full_text` — same pattern as the in-memory backend.
            conn.execute("DELETE FROM envelope_fts WHERE id = ?1", params![id_str])
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            conn.execute(
                "INSERT INTO envelope_fts (id, text) VALUES (?1, ?2)",
                params![id_str, env.summary],
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

            // Tag filter pushed into SQL via the envelope_tags index — an
            // indexed lookup (AND-intersection across all requested tags,
            // via GROUP BY/HAVING) instead of pulling every row and
            // filtering in the app layer. Matches InMemoryEnvelopeIndex's
            // tag_index in effect: a tag filter like `uid:{user_id}` scales
            // with that tag's own row count, not the whole table.
            let tag_clause = if query.tags.is_empty() {
                String::new()
            } else {
                let placeholders = query.tags.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                format!(
                    " AND id IN (SELECT envelope_id FROM envelope_tags \
                       WHERE tag IN ({placeholders}) \
                       GROUP BY envelope_id HAVING COUNT(DISTINCT tag) = {})",
                    query.tags.len()
                )
            };

            let sql = format!(
                "SELECT id, embedding, confidence, created_at, updated_at,
                        last_accessed_at, access_count, memory_type, source_sessions,
                        supersedes, superseded_by, summary, tags, content_hash,
                        valid_at, invalid_at
                 FROM envelopes
                 WHERE (?1 IS NULL OR superseded_by IS NULL)
                   AND (?2 IS NULL OR memory_type = ?2)
                   AND (?3 IS NULL OR confidence >= ?3)
                   {tag_clause}"
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

            let active_filter: Option<&str> = if query.active_only { Some("1") } else { None };
            let type_filter: Option<String> = query.memory_type.map(|t| format!("{:?}", t));
            let conf_filter: Option<f32> = query.min_confidence;

            let mut param_values: Vec<Box<dyn rusqlite::ToSql>> = vec![
                Box::new(active_filter.map(|s| s.to_string())),
                Box::new(type_filter),
                Box::new(conf_filter),
            ];
            for tag in &query.tags {
                param_values.push(Box::new(tag.clone()));
            }

            let rows: Vec<(
                String,
                Vec<u8>,
                f32,
                String,
                String,
                String,
                i64,
                String,
                String,
                String,
                Option<String>,
                String,
                String,
                i64,
                Option<String>,
                Option<String>,
            )> = stmt
                .query_map(
                    params_from_iter(param_values.iter().map(|b| b.as_ref())),
                    |row| {
                        Ok((
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
                            row.get::<_, Option<String>>(14)?,
                            row.get::<_, Option<String>>(15)?,
                        ))
                    },
                )
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

            // BM25 lexical channel via FTS5, restricted to the ids just
            // fetched above (same reasoning as the tag filter: bound the
            // scan to this query's own candidate set, not the whole table).
            let bm25_ranks: std::collections::HashMap<String, usize> =
                if !query.query_text.trim().is_empty() && !rows.is_empty() {
                    match fts_match_query(&query.query_text) {
                        Some(match_expr) => {
                            let candidate_ids: Vec<String> =
                                rows.iter().map(|r| r.0.clone()).collect();
                            let id_placeholders = candidate_ids
                                .iter()
                                .map(|_| "?")
                                .collect::<Vec<_>>()
                                .join(",");
                            let fts_sql = format!(
                                "SELECT id, bm25(envelope_fts) as score FROM envelope_fts \
                             WHERE envelope_fts MATCH ? AND id IN ({id_placeholders}) \
                             ORDER BY score ASC"
                            );
                            let mut fts_stmt = conn
                                .prepare(&fts_sql)
                                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
                            let mut fts_params: Vec<Box<dyn rusqlite::ToSql>> =
                                vec![Box::new(match_expr)];
                            for id in &candidate_ids {
                                fts_params.push(Box::new(id.clone()));
                            }
                            let ranked_ids: Vec<String> = fts_stmt
                                .query_map(
                                    params_from_iter(fts_params.iter().map(|b| b.as_ref())),
                                    |row| row.get::<_, String>(0),
                                )
                                .map_err(|e| StoreError::VectorIndex(e.to_string()))?
                                .collect::<Result<Vec<_>, _>>()
                                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
                            ranked_ids
                                .into_iter()
                                .enumerate()
                                .map(|(rank, id)| (id, rank))
                                .collect()
                        }
                        None => std::collections::HashMap::new(),
                    }
                } else {
                    std::collections::HashMap::new()
                };

            Ok::<_, StoreError>((rows, bm25_ranks))
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))??;

        let (rows, bm25_ranks) = rows;

        let mut results: Vec<RetrievalResult> = rows
            .into_iter()
            .map(
                |(
                    id,
                    emb_bytes,
                    confidence,
                    created_at,
                    updated_at,
                    last_accessed_at,
                    access_count,
                    memory_type_str,
                    sessions_json,
                    supersedes_json,
                    superseded_by_str,
                    summary,
                    tags_json,
                    content_hash,
                    valid_at,
                    invalid_at,
                )| {
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
                        valid_at: parse_opt_time(valid_at),
                        invalid_at: parse_opt_time(invalid_at),
                    };
                    let recency = env.time_decay(0.05) as f32;
                    let retrieval_score =
                        (1.0 - query.recency_weight) * similarity + query.recency_weight * recency;
                    RetrievalResult {
                        envelope: env,
                        similarity,
                        retrieval_score,
                    }
                },
            )
            // Valid-time filter, applied here rather than in the SQL WHERE
            // clause: this backend already scores every candidate row in the
            // app layer (brute-force cosine), so filtering here costs nothing
            // extra and keeps the semantics byte-identical to
            // InMemoryEnvelopeIndex's `is_valid_at` rather than depending on
            // lexicographic comparison of RFC3339 text columns.
            .filter(|r| match query.as_of {
                Some(as_of) => r.envelope.is_valid_at(as_of),
                None => true,
            })
            .collect();

        results.sort_by(|a, b| b.retrieval_score.partial_cmp(&a.retrieval_score).unwrap());

        // Fuse the vector ranking above with the BM25 lexical ranking via
        // Reciprocal Rank Fusion — same RRF_K and formula as
        // InMemoryEnvelopeIndex, so both backends rank identically given
        // the same candidate set and scores.
        const RRF_K: f32 = 60.0;
        if !bm25_ranks.is_empty() {
            let mut fused: Vec<(RetrievalResult, f32)> = results
                .into_iter()
                .enumerate()
                .map(|(vec_rank, r)| {
                    let vec_rrf = 1.0 / (RRF_K + vec_rank as f32 + 1.0);
                    let lex_rrf = bm25_ranks
                        .get(&r.envelope.id.to_string())
                        .map(|&rank| 1.0 / (RRF_K + rank as f32 + 1.0))
                        .unwrap_or(0.0);
                    (r, vec_rrf + lex_rrf)
                })
                .collect();
            fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            results = fused.into_iter().map(|(r, _)| r).collect();
        }

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
                         summary, tags, content_hash, valid_at, invalid_at
                  FROM envelopes WHERE id = ?1",
                params![id.to_string()],
                |row| {
                    let emb_bytes: Vec<u8> = row.get(1)?;
                    Ok(Envelope {
                        id: row.get::<_, String>(0)?.parse().unwrap(),
                        embedding: bytes_to_embedding(&emb_bytes),
                        confidence: row.get(2)?,
                        created_at: row
                            .get::<_, String>(3)?
                            .parse()
                            .unwrap_or_else(|_| Utc::now()),
                        updated_at: row
                            .get::<_, String>(4)?
                            .parse()
                            .unwrap_or_else(|_| Utc::now()),
                        last_accessed_at: row
                            .get::<_, String>(5)?
                            .parse()
                            .unwrap_or_else(|_| Utc::now()),
                        access_count: row.get::<_, i64>(6)? as u64,
                        memory_type: if row.get::<_, String>(7)?.contains("Working") {
                            MemoryType::Working
                        } else {
                            MemoryType::Semantic
                        },
                        source_sessions: serde_json::from_str(&row.get::<_, String>(8)?)
                            .unwrap_or_default(),
                        supersedes: serde_json::from_str(&row.get::<_, String>(9)?)
                            .unwrap_or_default(),
                        superseded_by: row
                            .get::<_, Option<String>>(10)?
                            .and_then(|s| s.parse().ok()),
                        summary: row.get(11)?,
                        tags: serde_json::from_str(&row.get::<_, String>(12)?).unwrap_or_default(),
                        content_hash: row.get::<_, i64>(13)? as u64,
                        valid_at: parse_opt_time(row.get::<_, Option<String>>(14)?),
                        invalid_at: parse_opt_time(row.get::<_, Option<String>>(15)?),
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
                             summary, tags, content_hash, valid_at, invalid_at, valid_at, invalid_at
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
                        valid_at: parse_opt_time(row.get::<_, Option<String>>(14)?),
                        invalid_at: parse_opt_time(row.get::<_, Option<String>>(15)?),
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
            let mut stmt = conn
                .prepare(
                    "SELECT e.id, e.embedding, e.confidence, e.created_at, e.updated_at,
                        e.last_accessed_at, e.access_count, e.memory_type, e.source_sessions,
                        e.supersedes, e.superseded_by, e.summary, e.tags, e.content_hash,
                        e.valid_at, e.invalid_at
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
                        created_at: row
                            .get::<_, String>(3)?
                            .parse()
                            .unwrap_or_else(|_| Utc::now()),
                        updated_at: row
                            .get::<_, String>(4)?
                            .parse()
                            .unwrap_or_else(|_| Utc::now()),
                        last_accessed_at: row
                            .get::<_, String>(5)?
                            .parse()
                            .unwrap_or_else(|_| Utc::now()),
                        access_count: row.get::<_, i64>(6)? as u64,
                        memory_type: MemoryType::Working,
                        source_sessions: serde_json::from_str(&row.get::<_, String>(8)?)
                            .unwrap_or_default(),
                        supersedes: serde_json::from_str(&row.get::<_, String>(9)?)
                            .unwrap_or_default(),
                        superseded_by: row
                            .get::<_, Option<String>>(10)?
                            .and_then(|s| s.parse().ok()),
                        summary: row.get(11)?,
                        tags: serde_json::from_str(&row.get::<_, String>(12)?).unwrap_or_default(),
                        content_hash: row.get::<_, i64>(13)? as u64,
                        valid_at: parse_opt_time(row.get::<_, Option<String>>(14)?),
                        invalid_at: parse_opt_time(row.get::<_, Option<String>>(15)?),
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
                params![
                    successor.to_string(),
                    Utc::now().to_rfc3339(),
                    id.to_string()
                ],
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            Ok::<_, StoreError>(())
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?
    }

    async fn invalidate(&self, id: Uuid, at: DateTime<Utc>) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            // MIN() so the earliest invalidation wins, matching
            // Envelope::invalidate — a later contradiction must not extend a
            // fact's validity window. MIN ignores NULL, so a first
            // invalidation on a still-valid row simply takes the new value.
            conn.execute(
                "UPDATE envelopes
                 SET invalid_at = MIN(COALESCE(invalid_at, ?1), ?1), updated_at = ?2
                 WHERE id = ?3",
                params![at.to_rfc3339(), Utc::now().to_rfc3339(), id.to_string()],
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

            // Collect matching ids first — envelope_tags/envelope_fts have
            // no foreign key to `envelopes` (FTS5 virtual tables can't be
            // FK targets), so a bulk DELETE FROM envelopes alone would
            // leave orphaned rows in both.
            let mut stmt = conn
                .prepare(
                    "SELECT id FROM envelopes
                     WHERE (memory_type = 'Working' AND created_at < ?1)
                        OR (confidence < ?2 AND superseded_by IS NOT NULL)",
                )
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            let ids: Vec<String> = stmt
                .query_map(params![cutoff.to_rfc3339(), confidence_floor], |row| {
                    row.get::<_, String>(0)
                })
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            drop(stmt);

            for id in &ids {
                conn.execute("DELETE FROM envelopes WHERE id = ?1", params![id])
                    .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
                conn.execute(
                    "DELETE FROM envelope_tags WHERE envelope_id = ?1",
                    params![id],
                )
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
                conn.execute("DELETE FROM envelope_fts WHERE id = ?1", params![id])
                    .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            }
            Ok::<_, StoreError>(ids.len())
        })
        .await
        .map_err(|e| StoreError::VectorIndex(e.to_string()))?
    }

    async fn delete(&self, id: Uuid) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let id_str = id.to_string();
            let n = conn
                .execute("DELETE FROM envelopes WHERE id = ?1", params![id_str])
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            if n == 0 {
                return Err(StoreError::NotFound(id));
            }
            conn.execute(
                "DELETE FROM envelope_tags WHERE envelope_id = ?1",
                params![id_str],
            )
            .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            conn.execute("DELETE FROM envelope_fts WHERE id = ?1", params![id_str])
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            Ok::<_, StoreError>(())
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

    async fn apply_decay(&self, lambda: f64) -> Result<usize, StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            // Read all active envelopes, compute new confidence, batch-update
            let mut stmt = conn
                .prepare(
                    "SELECT id, confidence, last_accessed_at FROM envelopes
                     WHERE superseded_by IS NULL",
                )
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

            let rows: Vec<(String, f32, String)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;

            let now = Utc::now();
            let mut updated = 0;
            for (id, confidence, last_accessed_str) in rows {
                let last_accessed = last_accessed_str
                    .parse::<chrono::DateTime<Utc>>()
                    .unwrap_or(now);
                let hours = now.signed_duration_since(last_accessed).num_seconds() as f64 / 3600.0;
                let decay = (-lambda * hours).exp();
                let new_conf = ((confidence as f64 * decay) as f32).max(0.0);
                conn.execute(
                    "UPDATE envelopes SET confidence = ?1 WHERE id = ?2",
                    params![new_conf, id],
                )
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
                updated += 1;
            }
            Ok::<_, StoreError>(updated)
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
                .query_row(
                    "SELECT COUNT(*) FROM envelopes WHERE memory_type = 'Working'",
                    [],
                    |r| r.get(0),
                )
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            let semantic: usize = conn
                .query_row(
                    "SELECT COUNT(*) FROM envelopes WHERE memory_type = 'Semantic'",
                    [],
                    |r| r.get(0),
                )
                .map_err(|e| StoreError::VectorIndex(e.to_string()))?;
            let superseded: usize = conn
                .query_row(
                    "SELECT COUNT(*) FROM envelopes WHERE superseded_by IS NOT NULL",
                    [],
                    |r| r.get(0),
                )
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
