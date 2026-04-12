//! SQLite-backed content store.
//!
//! Stores ContentBody as JSON blobs keyed by engram_id.
//! Can share the same SQLite file as SqliteEnvelopeIndex
//! or use a separate database.

use async_trait::async_trait;
use mneme_core::*;
use rusqlite::{params, Connection};
use std::sync::Mutex;
use uuid::Uuid;

use crate::{ContentStore, StoreError};

pub struct SqliteContentStore {
    conn: Mutex<Connection>,
}

impl SqliteContentStore {
    pub fn new(path: &str) -> Result<Self, StoreError> {
        let conn = Connection::open(path)
            .map_err(|e| StoreError::DocumentStore(e.to_string()))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS content_bodies (
                engram_id TEXT PRIMARY KEY,
                body_json TEXT NOT NULL
            );",
        )
        .map_err(|e| StoreError::DocumentStore(e.to_string()))?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn in_memory() -> Result<Self, StoreError> {
        Self::new(":memory:")
    }
}

#[async_trait]
impl ContentStore for SqliteContentStore {
    async fn put(&self, content: &ContentBody) -> Result<(), StoreError> {
        let json = serde_json::to_string(content)
            .map_err(|e| StoreError::Serialization(e.to_string()))?;

        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO content_bodies (engram_id, body_json)
             VALUES (?1, ?2)",
            params![content.engram_id.to_string(), json],
        )
        .map_err(|e| StoreError::DocumentStore(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, engram_id: Uuid) -> Result<ContentBody, StoreError> {
        let conn = self.conn.lock().unwrap();
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

        serde_json::from_str(&json)
            .map_err(|e| StoreError::Serialization(e.to_string()))
    }

    async fn get_batch(&self, ids: &[Uuid]) -> Result<Vec<ContentBody>, StoreError> {
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            results.push(self.get(*id).await?);
        }
        Ok(results)
    }

    async fn delete(&self, engram_id: Uuid) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM content_bodies WHERE engram_id = ?1",
            params![engram_id.to_string()],
        )
        .map_err(|e| StoreError::DocumentStore(e.to_string()))?;
        Ok(())
    }

    async fn append_conflict(
        &self,
        engram_id: Uuid,
        record: ConflictRecord,
    ) -> Result<(), StoreError> {
        let mut body = self.get(engram_id).await?;
        body.conflict_log.push(record);
        self.put(&body).await
    }
}
