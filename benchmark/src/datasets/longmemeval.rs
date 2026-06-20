use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LongMemEvalItem {
    pub history_id: String,
    pub turns: Vec<String>,
    pub question: String,
    pub answer: String,
    pub category: String,
}

pub fn load(path: &Path) -> Result<Vec<LongMemEvalItem>> {
    let text = std::fs::read_to_string(path)?;
    let data: Vec<LongMemEvalItem> = serde_json::from_str(&text)?;
    Ok(data)
}
