//! Concrete LLM implementations for the consolidation engine.
//!
//! - `AnthropicLLM`: Claude via the Anthropic API (production)
//! - `MockLLM`: deterministic responses for testing

use async_trait::async_trait;

use crate::{ConsolidateError, ConsolidationLLM};

// ─────────────────────────────────────────────────────────────
// Anthropic Claude LLM
// ─────────────────────────────────────────────────────────────

/// Claude via the Anthropic Messages API.
///
/// Used for consolidation judgments: synthesis, evolution evaluation,
/// coexistence analysis, and merge operations.
///
/// Uses claude-sonnet-4-20250514 by default for cost efficiency
/// during frequent consolidation operations.
pub struct AnthropicLLM {
    api_key: String,
    model: String,
    max_tokens: u32,
    client: reqwest::Client,
}

impl AnthropicLLM {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(api_key: String, model: &str) -> Self {
        Self {
            api_key,
            model: model.to_string(),
            max_tokens: 1024,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl ConsolidationLLM for AnthropicLLM {
    async fn complete(&self, prompt: &str) -> Result<String, ConsolidateError> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "system": "You are a memory consolidation engine. Always respond with valid JSON only. No markdown, no preamble, no explanation outside the JSON object."
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

        // Extract text from the first content block
        let text = data["content"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|block| block["text"].as_str())
            .ok_or_else(|| ConsolidateError::LLM("no text in response".into()))?;

        // Strip any markdown code fences the model might add
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

/// Deterministic LLM for testing consolidation logic.
///
/// Returns pre-configured responses based on the prompt content.
/// Allows testing compaction, evolution, and conflict resolution
/// without making API calls.
pub struct MockLLM {
    /// Default response when no specific pattern matches.
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
        // Pattern match on prompt content to return appropriate mock responses
        if prompt.contains("reconsolidation engine") || prompt.contains("should be updated") {
            // Evolution evaluation prompt
            if prompt.contains("CONFLICT_TRIGGER") {
                return Ok(serde_json::json!({
                    "decision": "conflict",
                    "reasoning": "Mock: conflict detected",
                    "confidence_adjustment": -0.1
                })
                .to_string());
            }
            return Ok(serde_json::json!({
                "decision": "keep",
                "reasoning": "Mock: memory is still accurate",
                "confidence_adjustment": 0.05
            })
            .to_string());
        }

        if prompt.contains("contradict each other") || prompt.contains("Determine the relationship") {
            // Conflict resolution evaluation
            return Ok(serde_json::json!({
                "relationship": "factual_update",
                "reasoning": "Mock: newer information supersedes older"
            })
            .to_string());
        }

        if prompt.contains("Merge two related memories") {
            // Merge prompt
            return Ok(serde_json::json!({
                "full_text": "Mock merged memory combining both sources",
                "summary": "Mock merged summary",
                "confidence": 0.85
            })
            .to_string());
        }

        if prompt.contains("new evidence has arrived") {
            // Evolution with new evidence
            return Ok(serde_json::json!({
                "full_text": "Mock evolved memory with new evidence",
                "summary": "Mock evolved summary",
                "confidence": 0.8,
                "change_type": "extended"
            })
            .to_string());
        }

        // Default: synthesis/compaction prompt
        Ok(self.default_response.clone())
    }
}
