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

The v0.0.5 process reached 1,296,992 KiB RSS in a disposable data directory while reusing the existing model. A full request comparison could not run because another local service already held v0.0.5's fixed port 6767. The raw performance harness is `bench/smoke.mjs`.

## MemoryBench

MemoryBench has not run. The official runner requires an OpenAI, Anthropic, or Google judge key in its process environment. No judge key was present. A score cannot be reported until that credential is available.

## Remaining parity gaps

- Temporal query parsing and provider-backed query rewriting are incomplete.
- V3 adjacent-chunk context and Workers AI reranking are incomplete.
- Batch `forget-matching` and direct memory update/create routes are incomplete.
- Dynamic profile diversification and summary caching are simpler than v0.0.5.
- URL, PDF, image, audio, and multipart file extraction are not implemented.
- The full SDK route inventory and controlled Bun performance comparison have not passed.
