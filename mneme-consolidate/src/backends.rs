//! LLM backends for consolidation.

use async_trait::async_trait;

use crate::{ConsolidateError, ConsolidationLLM};

// ─────────────────────────────────────────────────────────────
// Anthropic LLM (production)
// ─────────────────────────────────────────────────────────────

pub struct AnthropicLLM {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl AnthropicLLM {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "claude-sonnet-4-20250514".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(api_key: String, model: &str) -> Self {
        Self {
            api_key,
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl ConsolidationLLM for AnthropicLLM {
    async fn complete(&self, prompt: &str) -> Result<String, ConsolidateError> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        });

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ConsolidateError::LLM(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(ConsolidateError::LLM(format!(
                "Anthropic API error {}: {}",
                status, body
            )));
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ConsolidateError::LLM(e.to_string()))?;

        let text = data["content"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|block| block["text"].as_str())
            .ok_or_else(|| ConsolidateError::LLM("no text in response".into()))?;

        // Strip any markdown code fences
        let cleaned = text
            .trim()
            .strip_prefix("```json")
            .unwrap_or(text.trim())
            .strip_prefix("```")
            .unwrap_or(text.trim())
            .strip_suffix("```")
            .unwrap_or(text.trim())
            .trim()
            .to_string();

        Ok(cleaned)
    }
}

// ─────────────────────────────────────────────────────────────
// Mock LLM (for testing)
// ─────────────────────────────────────────────────────────────

pub struct MockLLM {
    default_response: String,
}

impl MockLLM {
    pub fn new() -> Self {
        Self {
            default_response: serde_json::json!({
                "full_text": "Mock consolidated memory",
                "summary": "Mock summary",
                "tags": ["mock"],
                "confidence": 0.8
            })
            .to_string(),
        }
    }

    pub fn with_default_response(response: &str) -> Self {
        Self {
            default_response: response.to_string(),
        }
    }
}

impl Default for MockLLM {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConsolidationLLM for MockLLM {
    async fn complete(&self, prompt: &str) -> Result<String, ConsolidateError> {
        if prompt.contains("reconsolidation engine") {
            if prompt.contains("CONFLICT_TRIGGER") {
                return Ok(serde_json::json!({
                    "decision": "conflict",
                    "reasoning": "Mock: conflict detected",
                    "confidence_adjustment": -0.1
                })
                .to_string());
            }
            if prompt.contains("UPDATE_TRIGGER") {
                return Ok(serde_json::json!({
                    "decision": "update",
                    "reasoning": "Mock: update needed",
                    "updated_text": "Mock updated memory text",
                    "updated_summary": "Mock updated summary",
                    "confidence_adjustment": 0.05
                })
                .to_string());
            }
            return Ok(serde_json::json!({
                "decision": "keep",
                "reasoning": "Mock: memory is still accurate",
                "confidence_adjustment": 0.0
            })
            .to_string());
        }

        if prompt.contains("evolution engine") {
            return Ok(serde_json::json!({
                "full_text": "Mock evolved memory combining old and new evidence",
                "summary": "Mock evolved summary",
                "confidence": 0.85
            })
            .to_string());
        }

        if prompt.contains("conflict") || prompt.contains("contradiction") {
            return Ok(serde_json::json!({
                "strategy": "temporal_supersede",
                "reasoning": "Mock: newer evidence supersedes older",
                "winner_index": 1
            })
            .to_string());
        }

        Ok(self.default_response.clone())
    }
}