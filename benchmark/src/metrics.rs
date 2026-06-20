use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResult {
    pub question_id: String,
    pub category: String,
    pub question: String,
    pub ground_truth: String,
    pub predicted: String,
    pub exact_match: bool,
    pub f1: f64,
    pub judge_score: Option<f64>,
    pub recall_latency_ms: u64,
    pub tokens_used: usize,
    pub memories_retrieved: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryMetrics {
    pub count: usize,
    pub exact_match: f64,
    pub f1: f64,
    pub judge_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub benchmark: String,
    pub model_embed: String,
    pub model_llm: String,
    pub total_questions: usize,
    pub exact_match: f64,
    pub f1: f64,
    pub judge_score: Option<f64>,
    pub avg_tokens_per_query: f64,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub per_category: HashMap<String, CategoryMetrics>,
    pub results: Vec<QuestionResult>,
}

/// Normalize text for scoring: lowercase, strip punctuation, collapse whitespace.
pub fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn exact_match(pred: &str, gold: &str) -> bool {
    normalize(pred) == normalize(gold)
}

pub fn token_f1(pred: &str, gold: &str) -> f64 {
    let pred_norm = normalize(pred);
    let pred_set: std::collections::HashSet<&str> = pred_norm.split_whitespace().collect();
    let gold_norm = normalize(gold);
    let gold_set: std::collections::HashSet<&str> = gold_norm.split_whitespace().collect();

    if pred_set.is_empty() || gold_set.is_empty() {
        return if pred_set == gold_set { 1.0 } else { 0.0 };
    }

    let common = pred_set.intersection(&gold_set).count() as f64;
    let precision = common / pred_set.len() as f64;
    let recall = common / gold_set.len() as f64;

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Estimate token count (~4 chars per token).
pub fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

pub fn aggregate(
    benchmark: &str,
    model_embed: &str,
    model_llm: &str,
    results: Vec<QuestionResult>,
) -> BenchmarkSummary {
    let total = results.len();
    if total == 0 {
        return BenchmarkSummary {
            benchmark: benchmark.to_string(),
            model_embed: model_embed.to_string(),
            model_llm: model_llm.to_string(),
            total_questions: 0,
            exact_match: 0.0,
            f1: 0.0,
            judge_score: None,
            avg_tokens_per_query: 0.0,
            p50_latency_ms: 0,
            p95_latency_ms: 0,
            per_category: HashMap::new(),
            results,
        };
    }

    let exact_match = results.iter().filter(|r| r.exact_match).count() as f64 / total as f64;
    let f1 = results.iter().map(|r| r.f1).sum::<f64>() / total as f64;

    let judge_scores: Vec<f64> = results.iter().filter_map(|r| r.judge_score).collect();
    let judge_score = if judge_scores.is_empty() {
        None
    } else {
        Some(judge_scores.iter().sum::<f64>() / judge_scores.len() as f64)
    };

    let avg_tokens = results.iter().map(|r| r.tokens_used as f64).sum::<f64>() / total as f64;

    let mut latencies: Vec<u64> = results.iter().map(|r| r.recall_latency_ms).collect();
    latencies.sort_unstable();
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() * 95) / 100];

    // Per-category aggregation
    let mut by_cat: HashMap<String, Vec<&QuestionResult>> = HashMap::new();
    for r in &results {
        by_cat.entry(r.category.clone()).or_default().push(r);
    }
    let per_category = by_cat
        .into_iter()
        .map(|(cat, rs)| {
            let n = rs.len();
            let em = rs.iter().filter(|r| r.exact_match).count() as f64 / n as f64;
            let f = rs.iter().map(|r| r.f1).sum::<f64>() / n as f64;
            let js: Vec<f64> = rs.iter().filter_map(|r| r.judge_score).collect();
            let js_avg = if js.is_empty() {
                None
            } else {
                Some(js.iter().sum::<f64>() / js.len() as f64)
            };
            (cat, CategoryMetrics { count: n, exact_match: em, f1: f, judge_score: js_avg })
        })
        .collect();

    BenchmarkSummary {
        benchmark: benchmark.to_string(),
        model_embed: model_embed.to_string(),
        model_llm: model_llm.to_string(),
        total_questions: total,
        exact_match,
        f1,
        judge_score,
        avg_tokens_per_query: avg_tokens,
        p50_latency_ms: p50,
        p95_latency_ms: p95,
        per_category,
        results,
    }
}
