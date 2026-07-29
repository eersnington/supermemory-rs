# Verify compatibility and performance

This page records checks that have run against the release binary. The strongest result is a matched five-question LoCoMo comparison with `supermemory-server` v0.0.5. Passing these checks does not establish full parity.

## Review verified coverage

The release binary passes the tested client flows, legacy migration, local embedding fixture, and matched MemoryBench workload. Known application programming interface and retrieval gaps remain.

| Area | Status | Evidence |
| --- | --- | --- |
| JavaScript client | Passed | `supermemory` 4.0.0 smoke flow |
| Python client | Passed | `supermemory` 3.51.0 smoke flow |
| Tool definitions | Passed | Seven `@supermemory/tools` 2.1.1 definitions |
| Legacy migration | Passed | Real encrypted v0.0.5 snapshot |
| Matched MemoryBench run | Passed | Same questions, episodes, models, and fresh data |
| Full parity | Not established | Gaps listed below |

## Compare matched MemoryBench results

The official MemoryBench runner tested three storage and search configurations on the same five-question LoCoMo panel. Each run used fresh data, the same 95 episode references, and question IDs `conv-26-q0` through `conv-26-q4`. The runs used the local `bge-base-en-v1.5` embedding model and Gemini 2.5 Flash for memory extraction, answers, and judging.

The `tursopg` run used Turso through its PostgreSQL protocol and ranked vectors with pgvector SQL. The earlier Rust run stored data in SQLite and searched vectors from filesystem-backed indexes. The third run used `supermemory-server` v0.0.5 with its detached Rivet engine.

| Measurement | Turso via `tursopg` + pgvector SQL search | SQLite + filesystem vector search | `supermemory-server` v0.0.5 |
| --- | ---: | ---: | ---: |
| Answer accuracy | 100% (5/5) | 60% (3/5) | 100% (5/5) |
| Retrieval Hit@10 | 100% | 100% | 100% |
| Mean reciprocal rank (MRR) | 0.900 | 0.469 | 1.000 |
| Normalized discounted cumulative gain (NDCG) | 0.856 | 0.563 | 0.937 |
| Ingestion acceptance, mean | 171 ms | 145 ms | 1,392 ms |
| Cold indexing, mean | 3,428,997 ms | 3,322,927 ms | 236,437 ms |
| Search, mean | 367 ms | 398 ms | 83 ms |
| Search, p95 | 397 ms | 638 ms | 109 ms |
| Answer context, mean | 822 tokens | 7,353 tokens | 9,679 tokens |

The `tursopg` configuration answered all five questions correctly and cut mean answer context from 7,353 to 822 tokens compared with the SQLite and filesystem search configuration. Search improved slightly, but cold indexing remained about 14.5 times slower than `supermemory-server`.

Do not use answer and judge latency to compare server performance because those phases include external Gemini requests. Five questions can expose compatibility and performance differences, but they cannot establish a stable quality ranking.

### Reset the v0.0.5 Rivet engine

v0.0.5 can leave its Rivet engine listening on `127.0.0.1:6420` after the server exits. A new server may reuse that stale process and leave accepted documents queued. Stop the orphaned Rivet process before running v0.0.5 with a fresh data directory.

This failure occurred during verification. After the stale engine stopped, a one-document probe moved from `queued` to `indexing` to `done` in 6s, and the matched benchmark completed.

## Check client compatibility

The release binary passed these local client checks on July 19, 2026:

- `supermemory` JavaScript software development kit (SDK) 4.0.0: add, status polling, V3 search, V4 document search, and profile
- `supermemory` Python SDK 3.51.0: the same request sequence
- `@supermemory/tools` 2.1.1: all seven artificial intelligence (AI) SDK and OpenAI tool definitions constructed against the local URL

Run the scripts from `compat/`. Read the [compatibility check instructions](../compat/README.md) for their scope and dependencies.

## Check legacy migration

The migration test decrypts a real v0.0.5 `SMD1` snapshot and opens it with the matching PGlite WebAssembly runtime and filesystem bundle. It exports deterministic JSON Lines, imports them into a fresh SQLite database, and verifies `SME1` credential re-encryption. The test does not modify the source snapshot.

Run the migration check with:

```sh
SUPERMEMORY_TEST_LEGACY=1 cargo test -p supermemory --test legacy --locked
```

Read the [legacy migration instructions](../migration/README.md) for the exporter dependency and startup path.

## Run the application programming interface smoke test

`compat/api-smoke.mjs` is a manual development check for application programming interface (API) throughput and latency. It is not a controlled comparison and does not run in continuous integration.

One release run processed 50 short documents and sent 100 requests to each measured endpoint:

| Measurement | Result |
| --- | ---: |
| Ingestion acceptance | 1,889.99 documents/s |
| Processing throughput | 59.60 documents/s |
| Search p95 | 2.99 ms |
| Profile p95 | 0.21 ms |
| Steady RSS after the workload | 185,152 KiB |

Use the matched MemoryBench results for cross-server conclusions.

## Track remaining parity gaps

The following behavior still differs from or lacks full verification against v0.0.5:

- Temporal query parsing and provider-backed query rewriting
- V3 adjacent-chunk context and Workers AI reranking
- Batch `forget-matching` and direct memory creation and update routes
- Dynamic profile diversification and summary caching
- URL, PDF, image, audio, and multipart file extraction
- Full JavaScript and Python SDK route inventories
