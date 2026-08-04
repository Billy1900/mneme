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
auth, and `/gc`/`/decay` coverage. 23 tests in that file, 53 across the
workspace.

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

Nothing else from earlier review rounds remains open: `/add` idempotency
now survives a restart via the same snapshot + WAL path as engrams (see
[Persistence](#persistence)), and is bounded by `ADD_SEEN_TTL_HOURS` so it
can't grow unbounded over a long-running evaluation.

## License

MIT — see [LICENSE](LICENSE).
