use anyhow::{anyhow, Result};
use reqwest::Client;

pub struct LLMJudge {
    client: Client,
    api_key: String,
    model: String,
}

impl LLMJudge {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: "gpt-4o-mini".to_string(),
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
            "model": self.model,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": prompt}],
        });

        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("OpenAI judge error {status}: {text}"));
        }

        let data: serde_json::Value = resp.json().await?;
        let content = data["choices"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|c| c["message"]["content"].as_str())
            .ok_or_else(|| anyhow!("no content in judge response"))?;

        // Parse JSON from response, strip markdown fences if present
        let cleaned = content
            .trim()
            .strip_prefix("```json")
            .unwrap_or(content.trim())
            .strip_prefix("```")
            .unwrap_or(content.trim())
            .strip_suffix("```")
            .unwrap_or(content.trim())
            .trim();

        let parsed: serde_json::Value = serde_json::from_str(cleaned)
            .map_err(|e| anyhow!("judge parse error: {e} on: {cleaned}"))?;

        let score = parsed["score"]
            .as_f64()
            .ok_or_else(|| anyhow!("no score in judge response"))?;

        Ok(score.clamp(0.0, 1.0))
    }

    /// Generate an answer from retrieved memory context.
    pub async fn generate_answer(&self, question: &str, memory_context: &str) -> Result<String> {
        let prompt = format!(
            r#"You have access to memory summaries about a person or situation.

Memory context:
{memory_context}

Based ONLY on the above memories, answer this question concisely:
{question}

If the memories do not contain the answer, say "Not found in memory".
Give only the answer, no explanation."#
        );

        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": prompt}],
        });

        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("OpenAI generate error {status}: {text}"));
        }

        let data: serde_json::Value = resp.json().await?;
        let content = data["choices"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|c| c["message"]["content"].as_str())
            .unwrap_or("Not found in memory")
            .trim()
            .to_string();

        Ok(content)
    }
}
