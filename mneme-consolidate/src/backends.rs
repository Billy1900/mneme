//! LLM backends for consolidation.

use async_trait::async_trait;

use crate::{ConsolidateError, ConsolidationLLM};

/// Extract the first valid JSON object from a response string.
/// More robust than stripping markdown fences character-by-character.
fn extract_json(text: &str) -> &str {
    let s = text.trim();
    if let (Some(start), Some(end)) = (s.find('{'), s.rfind('}')) {
        if start <= end {
            return &s[start..=end];
        }
    }
    s
}

// ─────────────────────────────────────────────────────────────
// Anthropic LLM (production)
// ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AnthropicLLM {
    api_key: String,
    /// True if api_key is an OAuth Bearer token (sk-ant-oat01-…),
    /// false if it's a plain API key (sk-ant-api03-…).
    use_bearer: bool,
    model: String,
    client: reqwest::Client,
}

impl AnthropicLLM {
    pub fn new(api_key: String) -> Self {
        let use_bearer = api_key.starts_with("sk-ant-oat");
        Self {
            api_key,
            use_bearer,
            model: "claude-haiku-4-5-20251001".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(api_key: String, model: &str) -> Self {
        let use_bearer = api_key.starts_with("sk-ant-oat");
        Self {
            api_key,
            use_bearer,
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Load credentials from ~/.claude/.credentials.json (Claude Code OAuth).
    pub fn from_claude_credentials() -> Result<Self, String> {
        let home = std::env::var("HOME").map_err(|e| e.to_string())?;
        let path = format!("{}/.claude/.credentials.json", home);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
        let val: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("parse credentials: {e}"))?;
        let token = val["claudeAiOauth"]["accessToken"]
            .as_str()
            .ok_or("no accessToken")?
            .to_string();
        Ok(Self::new(token))
    }
}

#[async_trait]
impl ConsolidationLLM for AnthropicLLM {
    async fn complete(&self, prompt: &str) -> Result<String, ConsolidateError> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        });

        let mut req = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        req = if self.use_bearer {
            req.header("Authorization", format!("Bearer {}", self.api_key))
        } else {
            req.header("x-api-key", &self.api_key)
        };

        let response = req
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

        let cleaned = extract_json(text).to_string();

        Ok(cleaned)
    }
}

// ─────────────────────────────────────────────────────────────
// OpenAI LLM (gpt-4o-mini, for benchmarks without Anthropic key)
// ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct OpenAILLM {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAILLM {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "gpt-4o-mini".to_string(),
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
impl ConsolidationLLM for OpenAILLM {
    async fn complete(&self, prompt: &str) -> Result<String, ConsolidateError> {
        let body = serde_json::json!({
            "model": self.model,
            // Fact extraction returns up to 12 facts, each a full sentence
            // plus a subject/relation/object decomposition. 512 truncated
            // that mid-object, which under `json_object` mode comes back as
            // an empty completion rather than a partial one.
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
            // This mode can only produce a top-level JSON *object*, so every
            // prompt sent through this backend must ask for one — a request
            // for a bare array returns empty. See `parse_extraction_list`.
            "response_format": {"type": "json_object"},
        });

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
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
                "OpenAI API error {}: {}",
                status, body
            )));
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ConsolidateError::LLM(e.to_string()))?;

        let text = data["choices"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|c| c["message"]["content"].as_str())
            .ok_or_else(|| ConsolidateError::LLM("no content in response".into()))?;

        // An empty completion with `finish_reason: length` means the token
        // budget ran out before any visible content — for a reasoning model,
        // usually spent entirely on reasoning tokens. Surfacing it as an error
        // matters because the callers here parse JSON out of this string, and
        // an empty string parses to "nothing extracted", silently turning a
        // misconfiguration into a plausible-looking result.
        if text.trim().is_empty() {
            let finish_reason = data["choices"][0]["finish_reason"]
                .as_str()
                .unwrap_or("unknown");
            return Err(ConsolidateError::LLM(format!(
                "empty completion (finish_reason: {finish_reason}) — if 'length', raise max_tokens; reasoning models spend it before emitting content"
            )));
        }

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
// DeepSeek LLM (OpenAI-compatible chat/completions API)
// ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct DeepSeekLLM {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl DeepSeekLLM {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "deepseek-v4-flash".to_string(),
            client: build_client(),
        }
    }

    pub fn with_model(api_key: String, model: &str) -> Self {
        Self {
            api_key,
            model: model.to_string(),
            client: build_client(),
        }
    }
}

/// DeepSeek's endpoint occasionally accepts a connection and then never
/// responds (observed hang with zero bytes received, indefinitely). A bare
/// `reqwest::Client::new()` has no timeout, so a request can wedge forever;
/// bound it so callers can retry instead of hanging the whole pipeline.
fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .expect("failed to build reqwest client")
}

#[async_trait]
impl ConsolidationLLM for DeepSeekLLM {
    async fn complete(&self, prompt: &str) -> Result<String, ConsolidateError> {
        let body = serde_json::json!({
            "model": self.model,
            // DeepSeek's V4 models are reasoners, and reasoning tokens are
            // charged against `max_tokens` before any visible content is
            // emitted. Measured on the fact-extraction prompt: at 2048 the
            // model burned all 2048 on reasoning and returned
            // `finish_reason: length` with an EMPTY content string — which
            // looks exactly like "this excerpt had no facts". At 8192 the same
            // three-turn prompt answered normally, but a full LoCoMo session
            // (~22 turns) exhausted 8192 too. Hence the large ceiling; it is a
            // floor on reasoning headroom, not an expectation about answer
            // length. 16384 is accepted by the API; 32768 also is, if needed.
            "max_tokens": 16384,
            "messages": [{"role": "user", "content": prompt}],
            // This mode can only produce a top-level JSON *object*, so every
            // prompt sent through this backend must ask for one — a request
            // for a bare array returns empty. See `parse_extraction_list`.
            "response_format": {"type": "json_object"},
        });

        let response = self
            .client
            .post("https://api.deepseek.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
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
                "DeepSeek API error {}: {}",
                status, body
            )));
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ConsolidateError::LLM(e.to_string()))?;

        let text = data["choices"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|c| c["message"]["content"].as_str())
            .ok_or_else(|| ConsolidateError::LLM("no content in response".into()))?;

        // An empty completion with `finish_reason: length` means the token
        // budget ran out before any visible content — for a reasoning model,
        // usually spent entirely on reasoning tokens. Surfacing it as an error
        // matters because the callers here parse JSON out of this string, and
        // an empty string parses to "nothing extracted", silently turning a
        // misconfiguration into a plausible-looking result.
        if text.trim().is_empty() {
            let finish_reason = data["choices"][0]["finish_reason"]
                .as_str()
                .unwrap_or("unknown");
            return Err(ConsolidateError::LLM(format!(
                "empty completion (finish_reason: {finish_reason}) — if 'length', raise max_tokens; reasoning models spend it before emitting content"
            )));
        }

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
// Ollama LLM (local, no API key)
// ─────────────────────────────────────────────────────────────

/// Talks to a local Ollama daemon (default `http://localhost:11434`) instead
/// of a paid cloud API. Lets compaction produce real (if lower-quality)
/// synthesis without an OpenAI/Anthropic key — useful when those are
/// unavailable (exhausted quota, no credentials) but a local model is.
#[derive(Clone)]
pub struct OllamaLLM {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OllamaLLM {
    pub fn new(model: &str) -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl ConsolidationLLM for OllamaLLM {
    async fn complete(&self, prompt: &str) -> Result<String, ConsolidateError> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": false,
            // Ollama's structured-output mode: constrains the model to emit
            // valid JSON, so we don't need markdown-fence stripping like the
            // cloud backends above.
            "format": "json",
            // Reasoning models (e.g. Qwen3) emit a long chain-of-thought
            // before the final JSON by default — 5-10x slower for no
            // quality benefit on a short extraction/synthesis prompt like
            // ours. Ignored by non-reasoning models.
            "think": false,
        });

        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| ConsolidateError::LLM(format!("Ollama request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(ConsolidateError::LLM(format!(
                "Ollama API error {}: {}",
                status, body
            )));
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ConsolidateError::LLM(e.to_string()))?;

        let text = data["message"]["content"]
            .as_str()
            .ok_or_else(|| ConsolidateError::LLM("no content in Ollama response".into()))?;

        Ok(extract_json(text).to_string())
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
