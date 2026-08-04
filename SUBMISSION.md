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

# Real embeddings (recommended for evaluation):
export OPENAI_API_KEY=sk-...
# Optional: require an API key on all routes except /health
export MNEME_API_KEY=some-shared-secret

./target/release/mneme-server
# Listening on 0.0.0.0:3377
```

Without `OPENAI_API_KEY` set, the server falls back to a deterministic mock
embedding model — useful for smoke-testing the contract shape locally, but not
representative of real retrieval quality. Set the key for anything used to
actually score recall.

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

## Isolation and performance

Multi-tenant isolation (`uid:{user_id}` tags) is backed by a tag → engram-id
index in `mneme-store/src/memory.rs` (`InMemoryEnvelopeIndex`), so a `/search`
call only scans that user's own data, not the whole store. This was load-tested
by seeding 50,000 background engrams under other user ids and confirming a
500-engram user's search latency stayed flat (~1.3–1.4ms p50) rather than
scaling with total store size.

## Running tests

```bash
# Full workspace test suite (uses mock embeddings/LLM — no API key needed)
unset OPENAI_API_KEY
cargo test --workspace
```

`mneme-tests` (root `tests/`) covers the storage/consolidation engine; the
`mneme-server` crate's `tests/http_integration.rs` covers the HTTP layer,
including `/remember`, `/recall`, auth, and validation. `/add` and `/search`
have been exercised via manual `curl` smoke tests (isolation, dedup, and
retrieval correctness) but do not yet have automated integration tests in
that suite — see below.

## Known limitations

These are open items, not blockers for a smoke test, but relevant to reviewers
assessing production-readiness for a long-running (72h) full evaluation:

- **In-memory persistence only in the default configuration.** `mneme-server`
  runs on `InMemoryEnvelopeIndex` / `InMemoryContentStore`. A process restart
  or crash loses all data written up to that point. `SqliteEnvelopeIndex` /
  `SqliteContentStore` exist in `mneme-store` as a durable alternative but
  aren't wired into `build_state()` yet, and the SQLite backend's tag filter
  is currently a post-scan (correct, but without the same per-tag index
  optimization as the in-memory backend).
- **`request_id` dedup table is in-memory** (`AppState.add_seen`) and is lost
  on restart, same caveat as above — a retried Add that lands after a restart
  will be treated as new rather than deduplicated.
- **No automatic garbage collection.** A background task decays confidence
  scores hourly, but nothing calls `/gc` automatically — long-running
  evaluation could grow memory usage unboundedly without an external GC
  trigger.
- **No automated tests for `/add` / `/search`** yet in `mneme-server/tests/` —
  correctness so far has been verified via manual smoke tests, not CI-covered
  regression tests.

## License

MIT — see [LICENSE](LICENSE).
