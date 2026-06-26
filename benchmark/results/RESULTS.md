# Mneme Benchmark Results

## LoCoMo — Best Run (`locomo_hybrid_v2_full.json`)
**Date:** 2026-06-26  
**Model:** gpt-4o (answer generation + judge), gpt-4o-mini (reranking)  
**Embedding:** text-embedding-3-small  
**top_k:** 5  
**Retrieval:** Hybrid (semantic engrams + raw working memory turns, merged + reranked)

| Category    | n    | Exact Match | Token F1 | Judge Score |
|-------------|------|-------------|----------|-------------|
| temporal    | 841  | 17.1%       | 0.375    | **0.532**   |
| single-hop  | 282  | 4.3%        | 0.225    | **0.314**   |
| multi-hop   | 321  | 1.9%        | 0.071    | 0.159       |
| open-domain | 96   | 6.2%        | 0.100    | 0.115       |
| **Overall** | 1542 | **10.9%**   | **0.267**| **0.388**   |

Not-found: 0 (0.0%) — None scores: 0

---

## Run History (LoCoMo, all 1542 questions)

| Run | Date | Overall Judge | Not-found | None scores | Notes |
|-----|------|--------------|-----------|-------------|-------|
| baseline gpt-4o-mini | 2026-06-22 | 0.466 | ~0% | ~0% | Inflated — hallucinated answers scored ~0.5 by judge |
| step1_2_full | 2026-06-22 | ~0.45 | ~5% | ~5% | Date injection (anchor entry) |
| step3_full | 2026-06-23 | ~0.44 | ~10% | ~10% | Multi-hop decomposition |
| step4_v2_full | 2026-06-23 | 0.029 | 70% | 21% | gpt-4o + Not-found→0.0; exposed real retrieval failure |
| **hybrid_v1_full** | 2026-06-26 | 0.276 | 17.5% | 102 | Hybrid recall; 429 retries not yet added |
| **hybrid_v2_full** | 2026-06-26 | **0.388** | **0%** | **0** | Hybrid recall + retry logic + judge parse fix |

---

## Key Findings

### Root cause of poor retrieval (step4_v2 collapse)
`recall()` only searched compacted semantic engrams (`MemoryType::Semantic`). Compaction of 30+ conversation turns into 5–10 engrams loses fine-grained facts. ~70–85% of questions returned "Not found in memory".

### Fix: Hybrid recall
`recall_with_fallback` now always searches **both**:
1. Semantic engrams (compacted, higher-level knowledge)
2. Raw working memory turns (original conversation text, verbatim facts)

Results are merged by retrieval score and reranked. Not-found rate dropped from 70% → 0%.

### Why baseline 0.466 was inflated
gpt-4o-mini hallucinated partial answers when memory was absent. The judge scored those ~0.5. gpt-4o correctly refuses, scoring 0.0. The 0.388 (gpt-4o, hybrid recall) is a reliable measurement against 0.466 (gpt-4o-mini, hallucinating).

### Remaining gaps
- **multi-hop (0.159)**: Answers require combining facts across multiple engrams; single-pass retrieval misses the combination.
- **open-domain (0.115)**: Questions require broad knowledge not always present in the conversation turns.

---

## LongMemEval
Only a 50-item smoke test was run (`longmemeval_50.json`), no full evaluation completed.
