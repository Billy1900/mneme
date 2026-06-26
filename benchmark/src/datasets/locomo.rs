use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoCoMoConversation {
    pub conversation_id: String,
    pub sessions: Vec<LoCoMoSession>,
    pub questions: Vec<LoCoMoQuestion>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoCoMoSession {
    pub session_id: String,
    #[serde(default)]
    pub session_date: String,
    pub turns: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoCoMoQuestion {
    pub question_id: String,
    pub question: String,
    pub answer: String,
    pub category: String,
}

pub fn load(path: &Path) -> Result<Vec<LoCoMoConversation>> {
    let text = std::fs::read_to_string(path)?;
    let data: Vec<LoCoMoConversation> = serde_json::from_str(&text)?;
    Ok(data)
}
