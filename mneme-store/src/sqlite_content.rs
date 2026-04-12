//! SQLite-backed content store.
//!
//! Stores ContentBody as JSON blobs keyed by engram_id.
//! Can share the same SQLite file as SqliteEnvelopeIndex
//! or use a separate database.

use async_trait::async_trait;
use mneme_core::*;
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::{ContentStore, StoreError};

#[derive(Clone)]
pub struct SqliteContentStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteContentStore {
    pub fn new(path: &str) -> Result<Self, StoreError> {
        let conn = Connection::open(path)
            .map_err(|e| StoreError::DocumentStore(e.to_string()))?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;

             CREATE TABLE IF NOT EXISTS content_bodies (
                 engram_id TEXT PRIMARY KEY,
                 body_json TEXT NOT NULL
             );",
        )
        .map_err(|e| StoreError::DocumentStore(e.to_string()))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn in_memory() -> Result<Self, StoreError> {
        Self::new(":memory:")
    }
}

#[async_trait]
impl ContentStore for SqliteContentStore {
    async fn put(&self, content: &ContentBody) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let content = content.clone();
        tokio::task::spawn_blocking(move || {
            let json = serde_json::to_string(&content)
                .map_err(|e| StoreError::Serialization(e.to_string()))?;
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT OR REPLACE INTO content_bodies (engram_id, body_json) VALUES (?1, ?2)",
                params![content.engram_id.to_string(), json],
            )
            .map_err(|e| StoreError::DocumentStore(e.to_string()))?;
            Ok::<_, StoreError>(())
        })
        .await
        .map_err(|e| StoreError::DocumentStore(e.to_string()))?
    }

    async fn get(&self, engram_id: Uuid) -> Result<ContentBody, StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let json: String = conn
                .query_row(
                    "SELECT body_json FROM content_bodies WHERE engram_id = ?1",
                    params![engram_id.to_string()],
                    |row| row.get(0),
                )
                .map_err(|e| match e {
                    rusqlite::Error::QueryReturnedNoRows => StoreError::NotFound(engram_id),
                    _ => StoreError::DocumentStore(e.to_string()),
                })?;
            serde_json::from_str(&json).map_err(|e| StoreError::Serialization(e.to_string()))
        })
        .await
        .map_err(|e| StoreError::DocumentStore(e.to_string()))?
    }

    // FIX #14: single spawn_blocking with internal loop instead of N awaits
    async fn get_batch(&self, ids: &[Uuid]) -> Result<Vec<ContentBody>, StoreError> {
        let conn = Arc::clone(&self.conn);
        let ids = ids.to_vec();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let mut results = Vec::with_capacity(ids.len());
            for id in ids {
                let json: String = conn
                    .query_row(
                        "SELECT body_json FROM content_bodies WHERE engram_id = ?1",
                        params![id.to_string()],
                        |row| row.get(0),
                    )
                    .map_err(|e| match e {
                        rusqlite::Error::QueryReturnedNoRows => StoreError::NotFound(id),
                        _ => StoreError::DocumentStore(e.to_string()),
                    })?;
                let body: ContentBody = serde_json::from_str(&json)
                    .map_err(|e| StoreError::Serialization(e.to_string()))?;
                results.push(body);
            }
            Ok::<_, StoreError>(results)
        })
        .await
        .map_err(|e| StoreError::DocumentStore(e.to_string()))?
    }

    async fn delete(&self, engram_id: Uuid) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "DELETE FROM content_bodies WHERE engram_id = ?1",
                params![engram_id.to_string()],
            )
            .map_err(|e| StoreError::DocumentStore(e.to_string()))?;
            Ok::<_, StoreError>(())
        })
        .await
        .map_err(|e| StoreError::DocumentStore(e.to_string()))?
    }

    async fn append_conflict(&self, engram_id: Uuid, record: ConflictRecord) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let json: String = conn
                .query_row(
                    "SELECT body_json FROM content_bodies WHERE engram_id = ?1",
                    params![engram_id.to_string()],
                    |row| row.get(0),
                )
                .map_err(|e| match e {
                    rusqlite::Error::QueryReturnedNoRows => StoreError::NotFound(engram_id),
                    _ => StoreError::DocumentStore(e.to_string()),
                })?;
            let mut body: ContentBody = serde_json::from_str(&json)
                .map_err(|e| StoreError::Serialization(e.to_string()))?;
            body.conflict_log.push(record);
            let new_json = serde_json::to_string(&body)
                .map_err(|e| StoreError::Serialization(e.to_string()))?;
            conn.execute(
                "UPDATE content_bodies SET body_json = ?1 WHERE engram_id = ?2",
                params![new_json, engram_id.to_string()],
            )
            .map_err(|e| StoreError::DocumentStore(e.to_string()))?;
            Ok::<_, StoreError>(())
        })
        .await
        .map_err(|e| StoreError::DocumentStore(e.to_string()))?
    }
}