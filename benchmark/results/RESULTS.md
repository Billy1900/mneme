# Mneme Benchmark Results

Generated 2026-08-05 02:25 by `benchmark/scripts/write_results.py`.

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
| Exact match | 13.6% |
| Token F1 | 0.263 |
| **Judge score** | **0.355** |
| Judge-scored questions | 234 |
| **Unscored (NOT in judge mean)** | **1** |
| Avg tokens/query | 376 |
| Latency p50 / p95 | 20419ms / 33323ms |
| Models | `memory=deepseek-v4-flash judge=deepseek-v4-flash facts=on`, embed `bge-small-en-v1.5` |

| Category | n | Exact match | Token F1 | Judge score |
|----------|---|-------------|----------|-------------|
| single-hop | 43 | 9.3% | 0.242 | 0.302 |
| multi-hop | 63 | 9.5% | 0.173 | 0.202 |
| temporal | 114 | 17.5% | 0.322 | 0.447 |
| open-domain | 13 | 0.0% | 0.147 | 0.346 |
| cat_5 | 2 | 100.0% | 1.000 | 1.000 |

### LoCoMo — fact extraction OFF (ablation)

Identical except extraction disabled. 2 conversations.

| Metric | Value |
|--------|-------|
| Questions | 235 |
| Exact match | 16.2% |
| Token F1 | 0.344 |
| **Judge score** | **0.511** |
| Judge-scored questions | 234 |
| **Unscored (NOT in judge mean)** | **1** |
| Avg tokens/query | 1120 |
| Latency p50 / p95 | 21874ms / 37746ms |
| Models | `memory=deepseek-v4-flash judge=deepseek-v4-flash facts=off`, embed `bge-small-en-v1.5` |

| Category | n | Exact match | Token F1 | Judge score |
|----------|---|-------------|----------|-------------|
| single-hop | 43 | 7.0% | 0.332 | 0.477 |
| multi-hop | 63 | 19.0% | 0.309 | 0.379 |
| temporal | 114 | 17.5% | 0.362 | 0.592 |
| open-domain | 13 | 7.7% | 0.286 | 0.462 |
| cat_5 | 2 | 100.0% | 1.000 | 1.000 |

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
| Avg tokens/query | 911 |
| Latency p50 / p95 | 813ms / 4009ms |
| Models | `memory=deepseek-v4-flash judge=deepseek-v4-flash facts=on`, embed `bge-small-en-v1.5` |

| Category | n | Exact match | Token F1 | Judge score |
|----------|---|-------------|----------|-------------|
| temporal-reasoning | 20 | 0.0% | 0.011 | 0.000 |

## Fact extraction: on vs off

| Config | Judge score | Token F1 | Exact match |
|--------|-------------|----------|-------------|
| Extraction ON | 0.355 | 0.263 | 13.6% |
| Extraction OFF | 0.511 | 0.344 | 16.2% |
| **Delta** | **-0.156** | -0.080 | -2.6pp |

Both halves ran on the same conversations with the same models, so the delta isolates the feature. It is a 2-conversation sample though — treat the sign as more reliable than the magnitude.

## Superseded history

Earlier result files in this directory (`locomo_hybrid_v2_full.json` and everything before it, including the frequently-quoted **0.388** overall judge score) were produced by a harness with defects found in a later audit. In those runs:

- Fact extraction never ran — it existed only in the HTTP `/add` handler, while the benchmark wrote through `MnemeMemory::remember`.
- Reranking and entity extraction were starved of output tokens on a DeepSeek judge and silently returned degraded defaults.
- Working-memory candidates were passed downstream truncated at ~100 characters.
- Transport errors and 5xx were not retried, and a failed answer generation was recorded as a 0.0 indistinguishable from a genuine retrieval miss.

Those numbers are a floor on the old system, not a fair measurement of it, and are not comparable to the runs above. The JSONs are kept for provenance.
