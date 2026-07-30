# Verification

The release binary passes the tested SDK flows, legacy migration, local embedding fixture, and LoCoMo workload. Full compatibility with `supermemory-server` v0.0.5 is not established.

## Coverage

| Area | Evidence |
| --- | --- |
| JavaScript SDK | `supermemory` 4.0.0 smoke flow |
| Python SDK | `supermemory` 3.51.0 smoke flow |
| Tools | Seven `@supermemory/tools` 2.1.1 definitions |
| Legacy migration | Encrypted v0.0.5 snapshot |
| Embeddings | Local `bge-base-en-v1.5` fixture |
| MemoryBench | Five-question LoCoMo run |

## LoCoMo results

The July 30, 2026 run used MemoryBench commit `118209a746d97d0d85e5a7234267f0b6962857e9`, 95 episodes, local BGE embeddings, and Gemini 2.5 Flash. The `supermemory-server` result used an earlier 127-episode run, so that comparison is directional.

| Measurement | Turso/Postgres | Current SQLite | `supermemory-server` v0.0.5 |
| --- | ---: | ---: | ---: |
| Accuracy | 100% | 80% | 100% |
| Hit@10 | 100% | 100% | 80% |
| MRR | 0.900 | 0.700 | 0.640 |
| NDCG | 0.856 | 0.737 | 0.632 |
| Indexing, mean | 57m 9s | 5m 21s | 5m 49s |
| Search, mean | 367 ms | 54 ms | 111 ms |
| Search, p95 | 397 ms | 73 ms | 158 ms |
| Context, mean | 822 tokens | 316 tokens | 11,122 tokens |
| Ready RSS | Not measured | 214 MiB | 1,544 MiB |
| Workload RSS, mean | Not measured | 286 MiB | 1,322 MiB |
| Workload RSS, peak | Not measured | 445 MiB | 1,781 MiB |
| Populated restart RSS | Not measured | 213 MiB | 1,017 MiB |

Ten extraction workers brought indexing below the `supermemory-server` result. Current SQLite also searched faster and used 4 to 7 times less memory. Accuracy ranged from 80% to 100% across two runs with the same code, so this sample does not establish a stable quality difference.

Answer and judge latency are excluded because they measure external Gemini requests.

## Client compatibility

The JavaScript and Python SDK checks cover add, status polling, V3 search, V4 document search, and profile. All seven AI SDK and OpenAI tool definitions also construct against the local URL.

Run the checks from `compat/`. See [compatibility check instructions](../compat/README.md).

## Legacy migration

The migration test decrypts a v0.0.5 `SMD1` snapshot, imports it into SQLite, and verifies `SME1` credential re-encryption without modifying the source.

```sh
SUPERMEMORY_TEST_LEGACY=1 cargo test -p supermemory --test legacy --locked
```

See [legacy migration instructions](../migration/README.md).

## SQLite vector search

SQLite ranks normalized embedding blobs with a registered `cosine_similarity` function. Filtering runs before ranking, and only selected rows are hydrated.

`sqlite-vec` 0.1.9 was not used because registration requires unsafe Rust and its `vec0` tables would duplicate the existing embedding tables.

## Remaining gaps

- Temporal query parsing and provider-backed query rewriting
- V3 adjacent-chunk context and Workers AI reranking
- Batch forgetting and direct memory mutation routes
- Dynamic profile diversification and summary caching
- URL, PDF, image, audio, and multipart extraction
- Full JavaScript and Python SDK route coverage

v0.0.5 may leave Rivet on `127.0.0.1:6420` after shutdown. Stop that process before a clean comparison run.
