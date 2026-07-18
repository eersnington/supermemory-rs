# Verification status

This page records executed checks. It is not a claim of full v0.0.5 parity.

## SDK compatibility

The release binary passed these local smoke tests on July 19, 2026:

- `supermemory` JavaScript SDK 4.0.0: add, status polling, V3 search, V4 document search, and profile.
- `supermemory` Python SDK 3.51.0: the same request sequence.
- `@supermemory/tools` 2.1.1: all seven AI SDK and OpenAI tool definitions constructed against the local base URL.

The scripts are in `compat/`.

## Legacy migration

The migration test decrypted a real v0.0.5 `SMD1` snapshot, loaded it through the matching `PGlite` WASM and filesystem bundle, exported stable JSONL, and imported it into a fresh SQLite database. The source snapshot was not modified. The test also covers `SME1` credential re-encryption.

Run this check with:

```sh
SUPERMEMORY_TEST_LEGACY=1 cargo test -p supermemory --test legacy --locked
```

## Local performance smoke test

The release build processed 50 short documents and 100 requests per endpoint on the development machine. This was a smoke test, not a controlled benchmark.

```text
accepted ingestion: 1889.99 documents/s
processing pipeline: 59.60 documents/s
search p95: 2.99 ms
profile p95: 0.21 ms
steady RSS after workload: 185152 KiB
```

The raw performance harness is `bench/smoke.mjs`. The controlled cross-server workload below
supersedes the earlier unmatched v0.0.5 RSS observation.

## MemoryBench

The official MemoryBench runner completed a controlled five-question LoCoMo comparison on July
19, 2026. Both servers used fresh isolated databases, the same 127 episodes, the same five
question IDs, the same local BGE assets, Gemini memory extraction, and Gemini 2.5 Flash for
answering and judging. Each process was sampled once per second from readiness through the full
run. v0.0.5 memory includes both `supermemory-server` and its detached Rivet engine.

| Measurement | Rust | v0.0.5 | Relative result |
| --- | ---: | ---: | --- |
| Accuracy | 80% (4/5) | 100% (5/5) | v0.0.5 1.25x higher |
| Hit@10 | 80% | 80% | Equal |
| MRR | 0.700 | 0.640 | Rust 1.09x higher |
| NDCG | 0.726 | 0.632 | Rust 1.15x higher |
| Accepted-ingestion mean | 47 ms | 1,900 ms | Rust 40.4x faster |
| Cold-indexing mean | 419,121 ms | 349,442 ms | Rust 1.20x slower |
| Search mean | 32 ms | 111 ms | Rust 3.47x faster |
| Search p95 | 47 ms | 158 ms | Rust 3.36x faster |
| Answer context mean | 7,297 tokens | 11,122 tokens | Rust used 1.52x fewer tokens |
| Ready RSS | 218,640 KiB | 1,581,312 KiB | Rust used 7.23x less memory |
| Workload mean RSS | 337,907 KiB | 1,353,826 KiB | Rust used 4.01x less memory |
| Workload peak RSS | 412,352 KiB | 1,824,096 KiB | Rust used 4.42x less memory |
| Populated restart RSS | 188,384 KiB | 1,041,314 KiB | Rust used 5.53x less memory |

On this matched run, Rust search was 3.5 times faster, but Rust cold indexing was 20% slower.
Rust used 4.4 times less memory at workload peak and 5.5 times less after reopening the populated
database. Answer and judge latency are not server performance measurements because they include
external Gemini calls. Five questions are enough for a compatibility and resource comparison,
not a statistically stable quality ranking.

Repeated v0.0.5 attempts initially remained queued because the server left an orphaned global
Rivet engine listening on `127.0.0.1:6420` after shutdown. Fresh server processes reused that
stale engine and accepted documents without dispatching their workflows. Stopping the orphaned
engine restored processing; a one-document probe then transitioned from queued to indexing to
done in six seconds before the completed benchmark run.

## Remaining parity gaps

- Temporal query parsing and provider-backed query rewriting are incomplete.
- V3 adjacent-chunk context and Workers AI reranking are incomplete.
- Batch `forget-matching` and direct memory update/create routes are incomplete.
- Dynamic profile diversification and summary caching are simpler than v0.0.5.
- URL, PDF, image, audio, and multipart file extraction are not implemented.
- The full SDK route inventory has not passed.
