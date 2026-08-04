use anyhow::{anyhow, Result};
use reqwest::Client;
use std::time::Duration;

fn strip_json_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s.strip_prefix("```json").unwrap_or(s);
    let s = s.strip_prefix("```").unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    s.trim()
}

/// DeepSeek's endpoint occasionally accepts a connection and never responds
/// (observed hang, zero bytes, indefinitely). A bare `Client::new()` has no
/// timeout, so bound it here so a hang surfaces as a retryable error instead
/// of wedging the whole benchmark run.
fn build_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .expect("failed to build reqwest client")
}

/// POST to the chat-completions endpoint with up to 4 retries on 429, doubling delay each time.
async fn chat_post_with_retry(
    client: &Client,
    base_url: &str,
    api_key: &str,
    body: &serde_json::Value,
) -> Result<serde_json::Value> {
    let mut delay = Duration::from_secs(5);
    for attempt in 0..=3 {
        let resp = client
            .post(base_url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await?;
        if resp.status().as_u16() == 429 && attempt < 3 {
            tokio::time::sleep(delay).await;
            delay *= 2;
            continue;
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("chat API error {status}: {text}"));
        }
        return Ok(resp.json().await?);
    }
    Err(anyhow!("chat API request failed after retries"))
}

pub struct LLMJudge {
    client: Client,
    api_key: String,
    /// Chat-completions endpoint (OpenAI-compatible wire format).
    base_url: String,
    /// Strong model for answer generation and judge scoring.
    model_strong: String,
    /// Cheap model for high-volume reranking and decomposition.
    model_fast: String,
}

impl LLMJudge {
    pub fn new(api_key: String) -> Self {
        Self {
            client: build_client(),
            api_key,
            base_url: "https://api.openai.com/v1/chat/completions".to_string(),
            model_strong: "gpt-4o".to_string(),
            model_fast: "gpt-4o-mini".to_string(),
        }
    }

    /// DeepSeek's chat/completions API is OpenAI wire-compatible. DeepSeek only
    /// ships one general-purpose chat model, so strong/fast both map to it.
    pub fn new_deepseek(api_key: String) -> Self {
        Self {
            client: build_client(),
            api_key,
            base_url: "https://api.deepseek.com/v1/chat/completions".to_string(),
            model_strong: "deepseek-v4-flash".to_string(),
            model_fast: "deepseek-v4-flash".to_string(),
        }
    }

    /// Score predicted answer vs ground truth on [0, 1].
    /// 1.0 = fully correct, 0.5 = partially correct, 0.0 = wrong.
    pub async fn score(&self, question: &str, ground_truth: &str, predicted: &str) -> Result<f64> {
        let prompt = format!(
            r#"You are an evaluation judge. Score whether the predicted answer correctly answers the question given the ground truth.

Question: {question}
Ground truth answer: {ground_truth}
Predicted answer: {predicted}

Scoring guide:
- 1.0: Predicted answer is fully correct (same meaning as ground truth, even if worded differently)
- 0.5: Partially correct (contains the key fact but is incomplete or has minor errors)
- 0.0: Wrong, irrelevant, or "I don't know" when the answer is knowable

Respond ONLY with JSON: {{"score": 0.0|0.5|1.0, "reason": "one sentence"}}"#
        );

        let body = serde_json::json!({
            "model": self.model_strong,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": prompt}],
        });

        let data =
            chat_post_with_retry(&self.client, &self.base_url, &self.api_key, &body).await?;
        let content = data["choices"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|c| c["message"]["content"].as_str())
            .ok_or_else(|| anyhow!("no content in judge response"))?
            .to_string();

        let cleaned = strip_json_fences(&content);
        let parsed: serde_json::Value = serde_json::from_str(cleaned)
            .map_err(|e| anyhow!("judge parse error: {e} on: {cleaned}"))?;
        let score = parsed["score"]
            .as_f64()
            .ok_or_else(|| anyhow!("no score in judge response"))?;

        Ok(score.clamp(0.0, 1.0))
    }

    /// Return indices (0-based) of the `keep` most relevant candidates for the question.
    pub async fn rerank_indices(
        &self,
        question: &str,
        candidates: &[&str],
        keep: usize,
    ) -> Result<Vec<usize>> {
        let numbered: String = candidates
            .iter()
            .enumerate()
            .map(|(i, t)| format!("[{}] {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Question: {question}\n\nMemory candidates:\n{numbered}\n\n\
            Return a JSON array of the {keep} most relevant candidate numbers (1-indexed), \
            most relevant first. Output ONLY the JSON array, e.g. [3,1,5].",
            keep = keep,
        );

        let body = serde_json::json!({
            "model": self.model_fast,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        });

        let resp = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let data: serde_json::Value = resp.json().await?;
        let content = data["choices"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|c| c["message"]["content"].as_str())
            .unwrap_or("{}");

        // response_format json_object wraps array as {"result":[...]} or similar
        let parsed: serde_json::Value =
            serde_json::from_str(content).unwrap_or(serde_json::Value::Null);
        // try common wrapper keys, or parse content directly as array
        let arr = parsed
            .as_array()
            .or_else(|| parsed["result"].as_array())
            .or_else(|| parsed["indices"].as_array())
            .or_else(|| parsed["rankings"].as_array());

        if let Some(arr) = arr {
            let indices: Vec<usize> = arr
                .iter()
                .filter_map(|v| v.as_u64())
                .filter(|&i| i >= 1 && i <= candidates.len() as u64)
                .take(keep)
                .map(|i| (i - 1) as usize)
                .collect();
            if !indices.is_empty() {
                return Ok(indices);
            }
        }

        // fallback: identity order
        Ok((0..keep.min(candidates.len())).collect())
    }

    /// Generate an answer from retrieved memory context.
    pub async fn generate_answer(&self, question: &str, memory_context: &str) -> Result<String> {
        let prompt = format!(
            r#"You have access to memory summaries about a person or situation.
Each memory may be prefixed with the date it was recorded, like "[8 May, 2023] ...".

Memory context:
{memory_context}

Based ONLY on the above memories, answer this question concisely:
{question}

If the question asks for a date or time and a memory expresses it relative to
its own recorded date (e.g. "last Saturday", "in 3 years"), resolve it to an
absolute date using that memory's date prefix. Never output the "[date]"
prefix bracket itself — only the resolved answer.
If the memories do not contain the answer, say "Not found in memory".
Give only the answer, no explanation."#
        );

        let body = serde_json::json!({
            "model": self.model_strong,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": prompt}],
        });

        let data =
            chat_post_with_retry(&self.client, &self.base_url, &self.api_key, &body).await?;
        let content = data["choices"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|c| c["message"]["content"].as_str())
            .unwrap_or("Not found in memory")
            .trim()
            .to_string();

        Ok(content)
    }

    /// Decompose a question into sub-queries for parallel retrieval.
    /// Returns the original question as the sole element if it is simple/single-hop.
    pub async fn decompose_question(&self, question: &str) -> Result<Vec<String>> {
        let prompt = format!(
            r#"You are a query planner for a memory retrieval system.

Question: {question}

If this question requires combining information from multiple distinct facts or events (multi-hop), break it into 2-3 simpler sub-questions, each retrievable independently. If it is a simple single-fact question, return it unchanged.

Respond ONLY with a JSON array of strings (1-3 items), e.g.:
["sub-question 1", "sub-question 2"]"#
        );

        let body = serde_json::json!({
            "model": self.model_fast,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        });

        let resp = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let data: serde_json::Value = resp.json().await?;
        let content = data["choices"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|c| c["message"]["content"].as_str())
            .unwrap_or("[]")
            .trim();

        // Strip markdown fences if present
        let cleaned = content
            .strip_prefix("```json")
            .unwrap_or(content)
            .strip_prefix("```")
            .unwrap_or(content)
            .strip_suffix("```")
            .unwrap_or(content)
            .trim();

        let sub_qs: Vec<String> = serde_json::from_str(cleaned).unwrap_or_default();

        if sub_qs.is_empty() {
            Ok(vec![question.to_string()])
        } else {
            Ok(sub_qs)
        }
    }

    /// Extract the entity names (people, places, organizations) mentioned or
    /// implied in a question, to seed graph traversal.
    pub async fn extract_entities(&self, question: &str) -> Result<Vec<String>> {
        let prompt = format!(
            r#"Question: {question}

List the named entities (people, places, organizations — not dates or generic
nouns) mentioned or implied in this question. Respond ONLY with a JSON array
of strings, e.g. ["Alice", "Acme Corp"]. If there are none, return []."#
        );

        let body = serde_json::json!({
            "model": self.model_fast,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
        });

        let resp = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let data: serde_json::Value = resp.json().await?;
        let content = data["choices"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|c| c["message"]["content"].as_str())
            .unwrap_or("[]");

        let cleaned = strip_json_fences(content);
        Ok(serde_json::from_str(cleaned).unwrap_or_default())
    }
}
