# Mneme — Reconsolidation-native memory for AI agents

> *mneme* `/ˈniː.miː/` — Greek: memory. The cognitive faculty itself, not just a trace.

A Rust memory engine for autonomous AI agents built on three operations from cognitive neuroscience that no existing system implements together: **memory compaction**, **memory evolution**, and **memory conflict resolution**.

## Why another memory system?

Every existing agent memory system treats memory as a **database problem**: store, index, retrieve. Mneme treats memory as a **living process** — memories compress, evolve on retrieval, and resolve contradictions using three distinct strategies. This directly implements mechanisms from neuroscience that the field has been talking about but nobody has built.

| Mechanism | Neuroscience | Existing systems | Mneme |
|-----------|-------------|-----------------|-------|
| Reconsolidation | Memories become labile on retrieval and can update | Append-only or latest-wins | Every retrieval runs drift detection → evolution |
| Dual-system consolidation | Hippocampus (fast) → Neocortex (slow) | Single flat store | Working memory → Semantic memory via compaction |
| Context-dependent memory | Same person, different facts in different contexts | Overwrite or duplicate | Three conflict strategies: supersede, merge, coexist |
| Progressive disclosure | Attention gates what enters working memory | Load everything into context | Metadata envelopes → summaries → full content |
| Forgetting curve | Ebbinghaus exponential decay | No decay | Confidence decay with reinforcement on access |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Agent (any framework: LangChain, custom, etc.)     │
│  remember() / recall() / expand() / end_session()   │
└────────────────────────┬────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │     mneme-api       │  ← 4 verbs, progressive disclosure
              └──────────┬──────────┘
                         │
        ┌────────────────▼────────────────┐
        │     mneme-consolidate           │  ← The three operations
        │  compact / evolve / resolve     │
        │  + LLM prompt templates         │
        └───┬─────────────────────┬───────┘
            │                     │
   ┌────────▼────────┐   ┌───────▼───────┐
   │   mneme-store   │   │  mneme-embed  │
   │  EnvelopeIndex  │   │ EmbeddingModel│
   │  ContentStore   │   └───────────────┘
   └─────────────────┘
```

### The Engram data model

Every memory is an **Engram** (neuroscience: the physical trace of a memory in the brain) with two layers:

**Envelope** (always loaded, ~200 bytes) — embedding vector, confidence score, timestamps, access count, supersession chain, 1-2 sentence summary, tags. This is what the retrieval system touches 99% of the time.

**Content Body** (loaded on demand) — full text, provenance chain back to raw conversations, conflict resolution log, typed relationships to other engrams, version number. Only loaded when the agent explicitly needs deep context.

This split enables **progressive disclosure**: search across thousands of envelopes, read 5-10 summaries, load 2-3 full bodies into context. Token savings are 10-50x vs flat approaches.

## The three operations

### 1. Compaction

Working memory entries from agent sessions get clustered by embedding similarity and synthesized into semantic engrams via LLM.

```
Working memory (session-bound, raw)
  "User mentioned wanting sub-2ms latency"
  "Discussed using Rust for the trading engine"
  "Considered C++ but worried about memory safety"
        │
        ▼  embed → cluster → synthesize
Semantic memory (persistent, distilled)
  "User is building a trading system requiring sub-2ms latency,
   preferring Rust over C++ for memory safety guarantees"
```

If a similar semantic engram already exists, compaction routes to **evolution** instead of creating a new one.

### 2. Evolution

Every retrieval triggers a **drift check**: cosine distance between the stored embedding and the current context. If drift exceeds a threshold (default 0.3), the LLM evaluates whether the memory should update.

This is **reconsolidation** — the mechanism Nader discovered in 2000. No existing agent memory system does this.

```
Stored: "Henry uses Python for all projects"
Current context: agent session discussing Rust systems work
  → drift = 0.45 (exceeds 0.3 threshold)
  → LLM verdict: UPDATE
  → New version: "Henry uses Python for ML and Rust for systems programming"
  → Old version: superseded (preserved in chain)
```

### 3. Conflict resolution

When two engrams contradict each other, the system selects from three strategies based on evidence strength:

| Strategy | When | Example |
|----------|------|---------|
| **Temporal supersede** | Clear factual update, large score gap | "was vegetarian" → "is now omnivore" |
| **Confidence merge** | Both sources credible, partially overlapping | Two reports on market conditions → synthesized view |
| **Conditional coexist** | Both true in different contexts | "uses Python for ML, Rust for systems" |

Every other system uses only "latest wins" or "keep both." Mneme is the first to distinguish between factual updates, synthesizable knowledge, and genuinely context-dependent truths.

## Storage backends

| Backend | Type | Use case | Scale |
|---------|------|----------|-------|
| `InMemoryEnvelopeIndex` + `InMemoryContentStore` | In-memory | Dev/testing | <1K engrams |
| `SqliteEnvelopeIndex` + `SqliteContentStore` | SQLite | Single-node production | <100K engrams |
| `QdrantEnvelopeIndex` + `SqliteContentStore` | Qdrant + SQLite | Scaled production | Millions |

Mix and match: Qdrant for ANN envelope search + SQLite for content body storage is the recommended production setup.

## Quick start

```rust
use mneme_api::MnemeMemory;
use mneme_consolidate::MockLLM;
use mneme_embed::MockEmbeddingModel;
use mneme_store::{InMemoryEnvelopeIndex, InMemoryContentStore, MnemeStore};
use mneme_core::MnemeConfig;

#[tokio::main]
async fn main() {
    let config = MnemeConfig::default();
    let store = MnemeStore::new(
        InMemoryEnvelopeIndex::new(),
        InMemoryContentStore::new(),
    );
    let embed = MockEmbeddingModel::new(128);
    let llm = MockLLM::new();

    // During agent session:
    memory.remember("User prefers Rust for systems work", "session-1").await;
    memory.remember("Building a trading engine with sub-2ms latency", "session-1").await;

    // When the agent needs context:
    let summaries = memory.recall("what language for the trading system?", 5).await;
    // → returns lightweight summaries (progressive disclosure L1)

    // If the agent needs full detail:
    let detail = memory.expand(summaries[0].id).await;
    // → loads full content body (progressive disclosure L2)

    // End of session — compaction runs:
    memory.end_session("session-1").await;
    // → working memory clustered + synthesized into semantic memory
}
```

## HTTP server

```bash
cargo run --bin mneme-server
# Listening on 0.0.0.0:3377
```

```bash
# Remember an observation
curl -X POST http://localhost:3377/remember \
  -H "Content-Type: application/json" \
  -d '{"observation": "User prefers dark mode", "session_id": "s1"}'

# Recall with context XML for prompt injection
curl -X POST http://localhost:3377/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "user preferences", "top_k": 5, "as_context": true}'

# Expand a specific memory
curl http://localhost:3377/expand/{engram-id}

# View evolution history
curl http://localhost:3377/history/{engram-id}

# End session (triggers compaction)
curl -X POST http://localhost:3377/end_session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1"}'
```

## Crate structure

```
mneme/
├── mneme-core/            361 lines — Engram, Envelope, ContentBody, config
├── mneme-store/           925 lines — 5 storage backends
│   ├── memory.rs          — InMemoryEnvelopeIndex
│   ├── memory_content.rs  — InMemoryContentStore
│   ├── sqlite_envelope.rs — SqliteEnvelopeIndex
│   ├── sqlite_content.rs  — SqliteContentStore
│   └── qdrant_envelope.rs — QdrantEnvelopeIndex
├── mneme-embed/           252 lines — embedding abstraction + 2 backends
│   └── backends.rs        — MockEmbeddingModel, OpenAIEmbeddingModel
├── mneme-consolidate/     704 lines — the three operations + 2 LLM backends
│   └── backends.rs        — AnthropicLLM, MockLLM
├── mneme-api/             330 lines — agent-facing API (4 verbs)
├── mneme-server/          511 lines — Axum HTTP server, 7 endpoints
└── tests/                 536 lines — 14 integration tests
```

## How Mneme compares

| | Mem0 | Zep | Letta (MemGPT) | Mneme |
|--|------|-----|----------------|-------|
| **Storage** | Vector + KV + Graph | Temporal knowledge graph | Tiered (core/recall/archival) | Dual-system (working + semantic) |
| **Reconsolidation** | No | No | No | Yes — drift-triggered on every retrieval |
| **Conflict resolution** | Latest wins | No | No | 3 strategies: supersede, merge, coexist |
| **Progressive disclosure** | No — loads all matches | No | Agent pages manually | Yes — envelope → summary → full content |
| **Consolidation** | Passive extraction | LLM entity extraction | Agent self-edits | LLM-powered embed → cluster → synthesize |
| **Forgetting curve** | No | No | No | Yes — Ebbinghaus decay with reinforcement |
| **Language** | Python | Python | Python | Rust |
| **Framework lock-in** | None (API) | None (API) | Full runtime | None (library or HTTP) |

## Neuroscience foundations

The design draws directly from five concepts in cognitive neuroscience:

1. **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly, 1995) — working memory as a fast-learning hippocampal system, semantic memory as a slow-learning neocortical system.

2. **Reconsolidation** (Nader, Schafe & LeDoux, 2000) — retrieved memories become labile and can be modified. The drift-score mechanism is our computational analog.

3. **Ebbinghaus forgetting curve** (1885) — confidence decays exponentially without access, reinforced on retrieval. The `time_decay(λ)` function implements this.

4. **Baddeley's working memory model** (2000) — the central executive decides what enters and leaves the limited-capacity scratchpad. Progressive disclosure implements this gating.

5. **Conway's Self-Memory System** — episodic memories organize hierarchically. The supersession chain provides version-level hierarchy; session-level grouping provides temporal hierarchy.

## License

MIT
