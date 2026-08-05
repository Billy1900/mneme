# Mneme Benchmark Results

Generated 2026-08-04 21:26 by `benchmark/scripts/write_results.py`.

## Read this before comparing to anything

These runs are **not** leaderboard-comparable, for three reasons:

1. **Subsets, not the full benchmark.** LoCoMo runs cover 2 of 10 conversations; the full set is 1,542 questions. A full DeepSeek run was measured at 7+ hours.
2. **Wrong model for a submission.** The Agent Memory Leaderboard mandates gpt-4o-mini for the memory system's Add and Search. These runs use DeepSeek throughout because no funded OpenAI key was available. Expect the compliant number to differ — DeepSeek V4 is a reasoning model and extracts facts better than gpt-4o-mini will.
3. **Self-judged.** The judge is also DeepSeek, so the same model family generates and scores the answers.

Use these to compare configurations **against each other** — that is what the on/off pair is for — not to compare mneme against published numbers from other systems.

## Results

### LoCoMo — fact extraction ON

Current system. 2 conversations.

| Metric | Value |
|--------|-------|
| Questions | 235 |
| Exact match | 13.2% |
| Token F1 | 0.271 |
| **Judge score** | **0.389** |
| Judge-scored questions | 234 |
| **Unscored (NOT in judge mean)** | **1** |
| Avg tokens/query | 260 |
| Latency p50 / p95 | 22112ms / 34980ms |
| Models | `memory=deepseek-v4-flash judge=deepseek-v4-flash facts=on`, embed `bge-small-en-v1.5` |

| Category | n | Exact match | Token F1 | Judge score |
|----------|---|-------------|----------|-------------|
| single-hop | 43 | 7.0% | 0.208 | 0.244 |
| multi-hop | 63 | 17.5% | 0.263 | 0.298 |
| temporal | 114 | 14.9% | 0.316 | 0.500 |
| open-domain | 13 | 0.0% | 0.159 | 0.385 |
| cat_5 | 2 | 0.0% | 0.000 | 0.000 |

### LoCoMo — fact extraction OFF (ablation)

Identical except extraction disabled. 2 conversations.

| Metric | Value |
|--------|-------|
| Questions | 235 |
| Exact match | 14.9% |
| Token F1 | 0.314 |
| **Judge score** | **0.474** |
| Judge-scored questions | 235 |
| Avg tokens/query | 1286 |
| Latency p50 / p95 | 23507ms / 40396ms |
| Models | `memory=deepseek-v4-flash judge=deepseek-v4-flash facts=off`, embed `bge-small-en-v1.5` |

| Category | n | Exact match | Token F1 | Judge score |
|----------|---|-------------|----------|-------------|
| single-hop | 43 | 4.7% | 0.245 | 0.453 |
| multi-hop | 63 | 19.0% | 0.316 | 0.365 |
| temporal | 114 | 18.4% | 0.352 | 0.544 |
| open-domain | 13 | 0.0% | 0.208 | 0.385 |
| cat_5 | 2 | 0.0% | 0.325 | 1.000 |

### LongMemEval — fact extraction ON

20 items.

> **VOID — not a measurement.** 20 of 20 answers were never generated (the run recorded the placeholder that a failed answer call produces, which scores 0.0 just like a real miss). The scores below reflect an outage, not retrieval quality. Check the run log for the cause and re-run before drawing any conclusion from this table.

| Metric | Value |
|--------|-------|
| Questions | 20 |
| Exact match | 0.0% |
| Token F1 | 0.011 |
| **Judge score** | **0.000** |
| Judge-scored questions | 20 |
| Avg tokens/query | 953 |
| Latency p50 / p95 | 773ms / 1340ms |
| Models | `memory=deepseek-v4-flash judge=deepseek-v4-flash facts=on`, embed `bge-small-en-v1.5` |

| Category | n | Exact match | Token F1 | Judge score |
|----------|---|-------------|----------|-------------|
| temporal-reasoning | 20 | 0.0% | 0.011 | 0.000 |

## Fact extraction: on vs off

| Config | Judge score | Token F1 | Exact match |
|--------|-------------|----------|-------------|
| Extraction ON | 0.389 | 0.271 | 13.2% |
| Extraction OFF | 0.474 | 0.314 | 14.9% |
| **Delta** | **-0.086** | -0.044 | -1.7pp |

Both halves ran on the same conversations with the same models, so the delta isolates the feature. It is a 2-conversation sample though — treat the sign as more reliable than the magnitude.

## Superseded history

Earlier result files in this directory (`locomo_hybrid_v2_full.json` and everything before it, including the frequently-quoted **0.388** overall judge score) were produced by a harness with defects found in a later audit. In those runs:

- Fact extraction never ran — it existed only in the HTTP `/add` handler, while the benchmark wrote through `MnemeMemory::remember`.
- Reranking and entity extraction were starved of output tokens on a DeepSeek judge and silently returned degraded defaults.
- Working-memory candidates were passed downstream truncated at ~100 characters.
- Transport errors and 5xx were not retried, and a failed answer generation was recorded as a 0.0 indistinguishable from a genuine retrieval miss.

Those numbers are a floor on the old system, not a fair measurement of it, and are not comparable to the runs above. The JSONs are kept for provenance.
