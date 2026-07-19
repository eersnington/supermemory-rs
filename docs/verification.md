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

The official MemoryBench runner tested both servers on July 19, 2026. Each run used fresh data, the same 127 LoCoMo episodes, and the same five question IDs. Both runs used the existing local `bge-base-en-v1.5` embedding model, Gemini memory extraction, and Gemini 2.5 Flash for answers and judging.

Memory sampling ran once per second from server readiness through evaluation. The v0.0.5 measurements include `supermemory-server` and its detached Rivet engine.

| Measurement | Rust | v0.0.5 | Relative result |
| --- | ---: | ---: | --- |
| Answer accuracy | 80% (4/5) | 100% (5/5) | v0.0.5: +20 percentage points |
| Retrieval Hit@10 | 80% | 80% | Equal |
| Mean reciprocal rank (MRR) | 0.700 | 0.640 | Rust: 1.09x higher |
| Normalized discounted cumulative gain (NDCG) | 0.726 | 0.632 | Rust: 1.15x higher |
| Ingestion acceptance, mean | 47 ms | 1,900 ms | Rust: 40.4x faster |
| Cold indexing, mean | 419,121 ms | 349,442 ms | Rust: 1.20x slower |
| Search, mean | 32 ms | 111 ms | Rust: 3.47x faster |
| Search, p95 | 47 ms | 158 ms | Rust: 3.36x faster |
| Answer context, mean | 7,297 tokens | 11,122 tokens | v0.0.5: 1.52x as many |
| Ready resident set size (RSS) | 218,640 KiB | 1,581,312 KiB | Rust: 7.23x lower |
| Workload RSS, mean | 337,907 KiB | 1,353,826 KiB | Rust: 4.01x lower |
| Workload RSS, peak | 412,352 KiB | 1,824,096 KiB | Rust: 4.42x lower |
| Populated restart RSS | 188,384 KiB | 1,041,314 KiB | Rust: 5.53x lower |

Rust used less memory and returned search results faster, but its cold indexing took 20% longer. v0.0.5 answered one additional question correctly, while both implementations achieved the same Hit@10.

Do not use answer and judge latency to compare server performance because those phases include external Gemini requests. Five questions can expose compatibility and resource differences, but they cannot establish a stable quality ranking.

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
