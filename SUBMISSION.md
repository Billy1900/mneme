# Agent Memory Leaderboard — Submission Notes

This document is for reviewers evaluating Mneme for the **Agent Memory Leaderboard**
(Textual Memory track, Academic Methods category). It covers how to build and run
the server, how the leaderboard's Add/Search contract maps onto Mneme's own API,
and known limitations.

Mneme itself is a general-purpose memory engine for AI agents — see
[README.md](README.md) for the architecture, the neuroscience motivation, and the
non-leaderboard-specific API. This file only covers what's relevant to reproducing
and evaluating this submission.

## Quick start

```bash
git clone https://github.com/Billy1900/mneme.git
cd mneme
cargo build --release -p mneme-server

# Real embeddings, no API key needed (recommended — see "Embedding backends" below):
export MNEME_EMBED_BACKEND=local

# Optional: require an API key on all routes except /health
export MNEME_API_KEY=some-shared-secret

# Recommended for a long-running evaluation — see "Persistence" below:
export MNEME_SNAPSHOT_PATH=./mneme-snapshot.json

./target/release/mneme-server
# Listening on 0.0.0.0:3377
```

## Persistence

The default store (`InMemoryEnvelopeIndex` / `InMemoryContentStore`) lives
entirely in process memory. Set `MNEME_SNAPSHOT_PATH` to a file path and the
server combines two mechanisms so a crash between periodic saves doesn't
lose everything written since the last one:

- **Full snapshot** — loaded on startup if it exists, saved every 5 minutes
  in the background, and saved once more on graceful shutdown
  (`SIGTERM`/`SIGINT`). Write-then-rename, so a crash mid-write can't
  corrupt the file — the previous snapshot stays valid until the new one is
  fully on disk.
- **Write-ahead log** (`{path}.wal`) — every successful `/add` message and
  `/remember` call, and every `/add` idempotency-table change (a new
  `request_id` seen, or one rolled back after a failed write), is appended
  to this file as one JSON line (`WalEntry`), synchronously, before the
  request returns. On startup, after loading the base snapshot, the WAL is
  replayed on top of it, so writes that happened after the last snapshot
  (including ones that never made it into any snapshot at all, if the
  process was killed before the first periodic save) are recovered too. A
  snapshot save truncates the WAL right after the new snapshot is durably
  on disk, under the same lock used for appends — see the doc comment on
  `wal_append`/`save_snapshot` in `mneme-server/src/main.rs` for why a write
  can never land in the gap between "snapshot read the store" and "WAL
  truncated" and be silently dropped.

The `/add` idempotency table (`AppState.add_seen`) is part of this same
snapshot + WAL, not a separate mechanism — so a `request_id` seen before a
restart is still recognized as a duplicate afterward, not silently retried
as new. Entries are capped at `ADD_SEEN_TTL_HOURS` (24h) by a background
eviction task, so the table doesn't grow unbounded over a long-running
evaluation.

If `MNEME_SNAPSHOT_PATH` is unset, both mechanisms are disabled and the
server is purely in-memory — a restart loses everything, same as before.
Verified with a hard `kill -9` (no graceful shutdown, before any periodic
save had fired): a memory written via `/add`, with a BM25-relevant keyword
past the summary truncation point, was recovered via WAL replay alone after
restart and was still findable by that keyword.

This was chosen over switching the default backend to `SqliteEnvelopeIndex`
because the tag index (multi-tenant isolation) and BM25 lexical channel
(see below) are only implemented for `InMemoryEnvelopeIndex` — snapshotting
keeps those while adding durability, rather than trading them away for it.
Restoring a snapshot replays each engram through the same `upsert` +
`index_full_text` path a live write would use, so the tag index and BM25
coverage come back exactly as they were, not just the raw data. Verified by
seeding a memory with a keyword past the BM25 truncation point, killing and
restarting the server with the same `MNEME_SNAPSHOT_PATH`, and confirming
the same keyword search still finds it.

## Embedding backends

Selected at startup, in priority order:

1. **`OPENAI_API_KEY`** (if set and non-empty) — OpenAI `text-embedding-3-small`.
2. **`MNEME_EMBED_BACKEND=local`** — BGE-small-en-v1.5 running locally via
   ONNX Runtime (`fastembed` crate). Real, meaningful embeddings with no API
   key, no per-request network call, and no per-token cost. Weights (~130MB)
   download once from Hugging Face on first startup and are cached under
   `~/.cache/fastembed`; every embed call after that is local CPU inference.
   This is the recommended way to run the server for evaluation, since it
   doesn't depend on a reviewer having (or this submission still having) a
   funded OpenAI key.
3. **Neither set** — falls back to a deterministic mock embedding model.
   Useful for smoke-testing the contract shape (does `/add`/`/search` return
   the right JSON), but the vectors carry no real semantic signal, so this is
   not representative of retrieval quality. This is also what
   `cargo test --workspace` uses, since tests must run without any external
   dependency or slow model download.

Retrieval itself is hybrid regardless of backend: cosine similarity over the
embedding is fused with a BM25 lexical channel via Reciprocal Rank Fusion
(`mneme-store/src/memory.rs`), so exact keyword/entity matches (names,
numbers) that a pure embedding can miss still surface — this matters for
whichever backend is generating the vectors. BM25 indexes each envelope's
`summary` by default (~100 chars, always loaded), then `/add` and `/remember`
immediately re-index with the full untruncated message
(`InMemoryEnvelopeIndex::index_full_text`) — otherwise a keyword past the
100-char mark would be invisible to lexical search even though the vector
channel can still find it via the full embedded text.

Health check:

```bash
curl http://localhost:3377/health
# {"status":"ok","version":"0.1.0"}
```

## Add / Search contract mapping

The leaderboard's fixed Add/Search schema is implemented at `POST /add` and
`POST /search` (`mneme-server/src/main.rs`), sitting on top of Mneme's own
envelope/content storage rather than replacing it. Mneme's own `/remember` and
`/recall` endpoints remain available for non-leaderboard use but aren't part of
this contract.

**`POST /add`**
- Each `messages[]` entry becomes one Working-memory engram (embedded, persisted
  synchronously before the response is returned).
- `user_id` is stamped onto every engram as a `uid:{user_id}` tag — this is what
  provides retrieval isolation between samples, since Mneme's core `MemoryQuery`
  has no native tenant field.
- `request_id` is deduplicated in an in-memory table: a repeated `request_id`
  returns the original response without writing again (see [Limitations](#known-limitations)).
- Response: `{success, request_id, user_id, session_id}`, matching the required
  shape.
- After the raw turns are written, the batch is passed through **LLM fact
  extraction** (see below) and each extracted fact is stored as its own
  additional engram.

**`POST /search`**
- `query` (and `options`, if present, concatenated in) is embedded and searched
  against both Working memory (raw turns) and Semantic memory (compacted),
  scoped to `uid:{user_id}` — search intentionally does **not** wait for or
  require compaction to have run, since the platform may call Search shortly
  after Add.
- Returns `data[]` with the full raw text (`content`) rather than a truncated
  summary, since the platform generates the final answer from this field.
- Only evidence is returned — no answer generation happens on this path, per
  contract.
- **Zero decay influence on ranking** (`recency_weight: 0.0`) — ranking here
  is pure relevance (vector + BM25 via RRF), with no penalty for memories
  that haven't been accessed recently. Mneme's background hourly decay task
  (Ebbinghaus forgetting curve) still runs and lowers confidence over time,
  but that's a separate signal from ranking; over a long-running evaluation,
  letting decay affect `/search` ranking would push correct-but-old memories
  out of `top_k`, which these datasets have no reason to penalize.

## Fact extraction on the write path

`POST /add` stores the raw turns, then distils the batch into **atomic facts**
and stores each as its own searchable engram
(`write_extracted_facts` in `mneme-server/src/main.rs`, backed by
`ConsolidationEngine::extract_facts`).

The motivation is that neither existing layer is a good retrieval target:

- A **raw turn** is context-dependent. "yeah, I switched last month" embeds
  nothing a later question can match, because the referent lives in a
  neighbouring turn.
- A **compacted engram** is a cluster-level summary. Compaction was supposed to
  fix the above, but it drops exactly the fine-grained specifics — names,
  numbers, dates — that the questions ask about, which is why
  `recall_with_fallback` had to start searching raw turns again.

**This feature does not currently pay for itself on LoCoMo.** See
[Measured performance](#measured-performance) before enabling it for a scored
run.

A fact is the missing middle layer: one self-contained assertion with its
pronouns resolved against the surrounding turns, and its date preserved. That
is a unit a query can actually match.

Details:

- **Windowed, not per message** — resolving "she" to a name requires the
  surrounding turns, so a per-message call would defeat the purpose as well as
  costing more. A batch is chunked into 10-line windows with 2 lines of overlap
  (`chunk_window`), one LLM call each, results deduplicated. Overlap exists so a
  pronoun near a window boundary still has its referent in view. Chunking was
  added after a full 22-turn LoCoMo session both exhausted the token budget and
  hit the extractor's 12-fact-per-call ceiling, silently discarding most of the
  session.
- **One code path.** The engram's shape — confidence, tags, memory type,
  summary, valid time — comes from `mneme_api::build_fact_engrams`, called by
  both this HTTP handler and `MnemeMemory::remember_facts` (the library path
  the benchmark harness uses). Persistence stays layer-specific, since the
  server has durability concerns the library doesn't (BM25 full-text
  re-indexing, write-ahead log). This split exists because the logic
  originally lived only in the `/add` handler while the benchmark went through
  `MnemeMemory::remember`, so **the benchmark was measuring a system with no
  fact extraction at all** — a parity test now pins the contract.
- Facts are stored as **Semantic** engrams tagged `uid:{user_id}` and `fact`
  (plus `date:{...}` when the fact carries one). `/search` already queries both
  tiers, so facts surface alongside the turns they came from rather than
  replacing them. Confidence is 0.6 — above a raw turn (0.5), below a compacted
  engram, since nothing has corroborated the fact across sessions yet.
- Facts are short by construction, so the envelope summary is the whole fact:
  no truncation, and BM25 covers it fully.
- **The stored body carries the fact *and* the window it came from.** The
  summary and the embedding stay the bare claim, so retrieval matches on
  something crisp, but `full_text` — what the answer generator receives —
  appends the source turns. A bare fact is true and frequently unusable: "the
  weekend before 13 September" cannot be answered from a sentence that has been
  stripped of everything around it. See
  [Measured performance](#measured-performance) for the numbers that forced
  this.
- **Facts are capped as a share of retrieval results** (`MAX_FACT_FRACTION`,
  0.5). Facts are short and keyword-dense, so they outscore verbatim passages on
  embedding similarity almost every time and, left uncapped, took every slot.
  The cap is a ceiling rather than a quota: unused slots return to whichever
  class still has candidates, so a query matching only facts still gets a full
  result set.
- The same call also returns an optional `(subject, relation, object)`
  decomposition, which goes straight into the **entity-relation graph**. Graph
  coverage therefore now comes from the write path instead of waiting on
  compaction to run — and is built from extracted facts rather than from the
  lossy compaction layer. Triplets are written to the WAL
  (`WalEntry::Triplets`) and included in snapshots, so the graph survives a
  restart rather than needing full re-extraction.
- **Best-effort by design.** The raw turns are already durably written and
  searchable before extraction runs, so every failure mode — no LLM configured,
  API error, timeout, unparseable response — degrades to exactly the previous
  behaviour rather than failing the Add. This matters under the harness
  specifically: it retries Add on 408/5xx, so turning a slow extraction into a
  failed request would cost far more than the facts are worth. The call is
  capped at `MNEME_FACT_EXTRACTION_TIMEOUT_SECS` (default 30s — enough for
  gpt-4o-mini; reasoning models need far more, see below).
- Enabled by default whenever a real LLM backend is configured. Set
  `MNEME_FACT_EXTRACTION=0` to store raw turns only. Forced **off** under
  `MockLLM`, which would otherwise fill the store with placeholder "facts" that
  dilute retrieval — this is why `cargo test` exercises the raw-turn path.

### LLM backend for a compliant run

The leaderboard mandates **gpt-4o-mini** for both Add and Search, and fact
extraction runs on the Add path, so a submission run must use:

```bash
export OPENAI_API_KEY=sk-...
export MNEME_LLM_BACKEND=openai   # gpt-4o-mini; override with MNEME_OPENAI_MODEL
```

Backend priority is otherwise `ANTHROPIC_API_KEY` > `MNEME_LLM_BACKEND`
(`openai` | `deepseek` | `ollama`) > `MockLLM`. Note that `ANTHROPIC_API_KEY`
takes precedence, so it must be unset for a compliant run.

**If you use a reasoning model** (e.g. `MNEME_LLM_BACKEND=deepseek`), note two
things measured during development, both of which make a misconfigured run look
like "this conversation contained no facts" rather than like an error:

- Reasoning tokens are charged against `max_tokens` *before* any visible
  content. On the extraction prompt, DeepSeek V4 burned all 2048 tokens
  reasoning and returned `finish_reason: length` with empty content; at 8192 it
  used ~6000 reasoning tokens and then answered normally, and a full session
  later exhausted 8192 as well. The backend now requests 16384 and raises an
  explicit error on an empty completion rather than storing zero facts quietly.
- The round-trip takes well over the default 30s timeout. Set
  `MNEME_FACT_EXTRACTION_TIMEOUT_SECS=180`.

Verified end to end with local BGE embeddings + DeepSeek: a three-turn `/add`
produced 3–4 fact engrams, and a query asking "What breed is Cooper and who
picked her up?" returned the self-contained facts *"Cooper is a golden
retriever."* and *"The user and Melanie picked up Cooper in Denver."*
alongside the raw turns — where the raw turns alone only offer the
pronoun-bound *"She is a golden retriever. Melanie and I picked her up in
Denver."* Facts and graph triplets were confirmed to survive a `kill -9`
restart via WAL replay.

## Measured performance

**No leaderboard-comparable number exists for this submission yet.** Everything
below was measured with DeepSeek `deepseek-v4-flash` on both the memory system
and the judge, over 2 of LoCoMo's 10 conversations (235 questions). The
leaderboard mandates gpt-4o-mini for Add and Search; that run has not been
performed, because no funded OpenAI key was available during development. Treat
these as configuration comparisons, not scores.

### Fact extraction currently costs about 0.15 judge score

| LoCoMo, 2 conversations | Judge | Token F1 | Exact match |
|---|---|---|---|
| Fact extraction **off** | **0.511** | 0.344 | 16.2% |
| Fact extraction **on** | 0.355 | 0.263 | 13.6% |

The diagnosed cause is context starvation, not bad facts. Both configurations
retrieve 5 memories per question, but with extraction on those 5 carried a
median of **121 tokens** against 1,077 with it off, and **146 of 235 questions
were answered from under 200 tokens** of context (2 of 235 with it off). 44
questions regressed against 15 improved, and 36 of the 44 regressions were the
model replying "Not found in memory" — retrieval failures rather than reasoning
failures, concentrated in the temporal (26) and multi-hop (14) categories that
need surrounding narrative.

Two fixes have been implemented and unit-tested but **not yet measured**: the
retrieval cap described above, and storing each fact together with its source
window. An earlier attempt at the cap was placed at the wrong pipeline stage and
silently did nothing, which is why the fix is described here as unverified
rather than as a solution.

Until a run confirms otherwise, `MNEME_FACT_EXTRACTION=0` is the better
configuration for a scored LoCoMo run, despite extraction being enabled by
default whenever a real LLM backend is present.

### The harness cannot currently resolve differences this small

The extraction-off configuration was run twice across a code change that
provably could not affect it, and moved **0.474 → 0.511** on its own. That
±0.04 of run-to-run nondeterminism — a sampling model both generating and
judging — is the same size as several deltas previously reported as findings.
The 0.15 extraction gap is well outside it; smaller movements from this harness
should not be trusted without more conversations and a temperature-0 judge.

### Earlier numbers in this repository are superseded

Results predating the harness audit — including the frequently-quoted **0.388**
LoCoMo score and the multi-hop 0.067 → 0.151 improvement cited below — were
produced by a benchmark harness with several silent-failure paths: fact
extraction never ran on the benchmark path at all, reranking and entity
extraction were starved of output tokens and returned degraded defaults,
working-memory text was truncated to ~100 characters before reaching the answer
generator, and unretried transport errors were recorded as scores of 0.0. Those
numbers are a floor on the old system rather than a measurement of it.
`benchmark/results/RESULTS.md` regenerates from the run JSONs and flags any run
whose answers were mostly never generated as VOID instead of reporting its
zeros.

## Bitemporal validity (valid time vs transaction time)

`Envelope` carries two time axes rather than one:

- **Transaction time** — `created_at` / `updated_at`: when the system learned
  or recorded something. Already present.
- **Valid time** — `valid_at` / `invalid_at`: when the asserted fact was true
  *in the world*. New.

"Melanie adopted Cooper in May 2023", recorded today, has `valid_at` = May 2023
and `created_at` = today. Keeping both is what makes *"what was true as of date
X"* answerable, as distinct from *"what had we written down by date X"* — and
temporal reasoning is one of the leaderboard's scored dimensions.

**Invalidation is not supersession.** These were previously the same thing;
they aren't:

- `superseded_by` means "a newer *version* of this memory exists" — the old
  one was a worse rendering of the same fact, and is hidden from active recall.
- `invalid_at` means "this was true, and then it stopped being true". The old
  engram is not a mistake; it is still the *correct* answer to a question asked
  about an earlier date.

Conflict resolution now does both: the contradicted engram is superseded (so it
can't answer questions about the present) *and* invalidated at the moment of
contradiction (so it can still answer questions about the past). "Melanie was
vegetarian" doesn't become wrong when she starts eating meat — it acquires an
end date.

**Querying.** `MemoryQuery::as_of` filters on the valid-time axis:

- `None` (the default) disables validity filtering entirely — every existing
  caller behaves exactly as before, and dropping contradicted facts is an
  explicit decision per call site rather than a silent global one.
- `Some(Utc::now())` — what is believed true now. This is what `/search` and
  `/recall` pass.
- `Some(past_instant)` — what was believed true then.

**Where valid time comes from.** Fact extraction already asks the model for the
date a fact pertains to; `ExtractedFact::valid_at()` parses the unambiguous
absolute forms ("2023-05-08", "8 May 2023", "May 8, 2023", "May 2023" → 1 May)
into a real timestamp. Relative expressions ("last month", "yesterday") are
deliberately **not** resolved — doing that correctly needs the conversation's
own reference date, and guessing would write a confidently wrong timestamp into
the validity window, which is worse than leaving it unknown. Raw turns and
compacted engrams carry no valid time.

**One deliberate caveat.** Because `/search` queries as of *now*, a fact whose
`valid_at` is in the future ("I start the new job in September") is excluded —
it isn't true yet. Those should be rare, since valid time is only set from
unambiguous absolute dates, which are overwhelmingly past. If a measured run
shows recall lost on forward-looking questions, that filter is the line to
revisit; it's a single call site.

**Storage.** Both columns are nullable and `#[serde(default)]`, so envelopes
written before these fields existed still load from snapshots, WAL lines and
Qdrant payloads, reading back as "valid for all time" — which is the correct
interpretation of "we never recorded a validity window". The SQLite backend
adds the columns via `ALTER TABLE` on open if they're missing, since
`CREATE TABLE IF NOT EXISTS` is a no-op against a pre-existing database.

## Isolation and performance

Multi-tenant isolation (`uid:{user_id}` tags) is backed by a tag → engram-id
index in `mneme-store/src/memory.rs` (`InMemoryEnvelopeIndex`), so a `/search`
call only scans that user's own data, not the whole store. This was load-tested
by seeding 50,000 background engrams under other user ids and confirming a
500-engram user's search latency stayed flat (~1.3–1.4ms p50) rather than
scaling with total store size.

A background GC task runs hourly alongside the existing decay task
(`gc_confidence_floor` / `working_memory_ttl_hours` from `MnemeConfig`),
so superseded/low-confidence engrams and stale Working memory don't
accumulate unbounded over a long-running evaluation without an external
`/gc` caller.

`/search` also caps its own response size: each item's `content` is
truncated at 4,000 chars, and once the running total across `data[]`
exceeds 24,000 chars, remaining (lower-ranked) results are dropped rather
than included — `top_k=100` with long raw turns could otherwise return an
unbounded response and blow the platform's token budget for the downstream
answer-generation call. Verified by seeding 10 ~6,300-char memories and
confirming the response stops at 5 truncated items (~20K chars) instead of
returning all 10 at full length.

## Running tests

```bash
# Full workspace test suite (uses mock embeddings/LLM — no API key needed)
unset OPENAI_API_KEY
cargo test --workspace
```

`mneme-tests` (root `tests/`) covers the storage/consolidation engine,
including both backends' BM25 lexical channel and tag-index isolation.
`mneme-server`'s `tests/http_integration.rs` covers the HTTP layer end to
end, including the leaderboard contract: `/add` success + validation +
`request_id` dedup, `/search` user isolation, the BM25 full-text lexical
match (a keyword past the summary's truncation point), the response budget
cap, and validation — alongside the pre-existing `/remember`, `/recall`,
auth, and `/gc`/`/decay` coverage. 23 tests in that file, 79 across the
workspace.

## Multi-hop retrieval and consolidation improvements

Three changes to the core consolidation/recall pipeline (`mneme-consolidate`,
`mneme-api`, `mneme-store`), aimed at multi-hop question answering, where a
single vector-similarity pass tends to surface only one of several facts a
question needs combined:

- **Entity-relation graph.** `ConsolidationEngine::compact_session` now
  extracts `(subject, relation, object, date)` triplets from each newly
  synthesized engram via the LLM and adds them to an in-memory graph index
  (`mneme-store/src/graph.rs`, `GraphIndex`/`InMemoryGraphIndex`), keyed off
  the engram it came from. A question's named entities can then be traversed
  2 hops out to pull in connected facts that didn't score high enough on
  their own to be a direct vector hit.
- **`related[]` engram linking.** Engrams synthesized from the same
  compaction batch are now linked to each other (`RelatedEngram`,
  previously an unused field on `ContentBody`). `MnemeMemory::recall`
  (`mneme-api`) does a 1-hop expansion through these links before returning
  results.
- **Reconsolidation conflict resolution.** Previously a detected `CONFLICT`
  during reconsolidation just decayed the stale memory's confidence and left
  it active, so a contradicted fact could keep resurfacing in recall. It now
  supersedes the old engram (excluded from active recall via
  `Envelope::is_active`) with a new one reflecting the current context, and
  logs a `ConflictRecord` with the LLM's reasoning — history stays
  traceable via the existing `supersedes` chain.

These were originally validated against the LoCoMo multi-hop category via the
`benchmark` crate (DeepSeek `deepseek-v4-flash` + local embeddings): a small
sample (n=63, 2 conversations) went from judge score 0.067 to 0.151 after
adding graph traversal. **That measurement is superseded** — it came from the
pre-audit harness, and at n=63 it sits close to the noise floor described in
[Measured performance](#measured-performance). The mechanisms are sound and
tested; the quoted improvement should be treated as unconfirmed.

**Caveat:** the graph traversal and `related[]` expansion are consumed by
`benchmark`'s `recall_multihop`/`MnemeMemory::recall` path, not yet by
`mneme-server`'s `/recall` or `/search` handlers, which query
`state.envelopes.search()` directly rather than going through
`MnemeMemory`. The reconsolidation conflict fix, by contrast, is in the
consolidation engine itself and does apply to the server. Wiring graph
traversal into `/search` is the natural next step if multi-hop performance
matters for the leaderboard evaluation.

## Known limitations

These are open items, not blockers for a smoke test, but relevant to reviewers
assessing production-readiness for a long-running (72h) full evaluation:

- **`SqliteEnvelopeIndex` isn't wired into `build_state()`.** It now has the
  same tag-index-backed isolation (via a relational `envelope_tags` table)
  and BM25 lexical channel (via SQLite's native FTS5, fused with vector
  similarity through the same RRF logic) as `InMemoryEnvelopeIndex` — see
  `mneme-store/src/sqlite_envelope.rs` and its regression test
  (`test_sqlite_envelope_search_bm25_full_text`) — but it still isn't the
  backend the server actually runs on. Durability is instead handled via
  snapshot + WAL on top of the in-memory backend (see
  [Persistence](#persistence)), which was simpler than swapping backends
  and re-verifying every existing behavior against a new one under
  evaluation-time pressure.

- **`QdrantEnvelopeIndex` does not exist as working code.**
  `mneme-store/src/qdrant_envelope.rs` is present in the tree but is not
  declared as a module in `mneme-store/src/lib.rs`, so it has never been
  compiled and cannot be selected at runtime. README's backend table lists
  Qdrant + SQLite as "the recommended production setup" for millions of
  engrams; that claim is not currently backed by reachable code. The largest
  configuration actually exercised is the in-memory backend with 50,000 seeded
  engrams (see [Isolation and performance](#isolation-and-performance)).
- **No gpt-4o-mini run has been performed**, so the mandated submission
  configuration is unvalidated end to end. Fact extraction in particular was
  tuned against a reasoning model whose token behaviour differs substantially
  (see the backend notes above), and the 30s default extraction timeout was
  chosen for gpt-4o-mini but only ever exercised at 180s against DeepSeek.
- **Benchmark coverage is thin.** Reported runs cover 2 of 10 LoCoMo
  conversations. LongMemEval has no valid recent run — the two most recent
  attempts both exhausted their API balance partway through and are marked VOID
  in `benchmark/results/RESULTS.md`.

`/add` idempotency now survives a restart via the same snapshot + WAL path as
engrams (see [Persistence](#persistence)), and is bounded by
`ADD_SEEN_TTL_HOURS` so it can't grow unbounded over a long-running evaluation.

## License

MIT — see [LICENSE](LICENSE).
