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

| Measurement | Meaning | `supermemory-rs` | `supermemory-server` v0.0.5 | Better by |
| --- | --- | ---: | ---: | --- |
| Accuracy | Questions judged correct | 80% | 100% | `supermemory-server` by 20 percentage points |
| Hit@10 | Questions with a relevant result in the top 10 | 100% | 80% | `supermemory-rs` by 20 percentage points |
| MRR | How early the first relevant result appears | 0.700 | 0.640 | `supermemory-rs` by 9.4% |
| NDCG | Overall ordering of relevant results | 0.737 | 0.632 | `supermemory-rs` by 16.6% |
| Ingestion, mean | Time to accept a document | 47 ms | 1,900 ms | `supermemory-rs` 40.4x faster |
| Indexing, mean | Time until submitted episodes finish indexing | 5m 21s | 5m 49s | `supermemory-rs` 8.1% faster |
| Search, mean | Average search request time | 54 ms | 111 ms | `supermemory-rs` 2.06x faster |
| Search, p95 | Time covering 95% of search requests | 73 ms | 158 ms | `supermemory-rs` 2.16x faster |
| Context, mean | Retrieved tokens sent to the answer model | 316 tokens | 11,122 tokens | `supermemory-rs` uses 97.2% fewer |
| Ready RSS | Memory after startup | 214 MiB | 1,544 MiB | `supermemory-rs` uses 7.2x less |
| Workload RSS, mean | Average memory during the benchmark | 286 MiB | 1,322 MiB | `supermemory-rs` uses 4.6x less |
| Workload RSS, peak | Highest memory during the benchmark | 445 MiB | 1,781 MiB | `supermemory-rs` uses 4.0x less |
| Populated restart RSS | Memory after restarting with indexed data | 213 MiB | 1,017 MiB | `supermemory-rs` uses 4.8x less |

`supermemory-rs` was faster, retrieved relevant results more consistently, used less context, and used less memory. `supermemory-server` answered one more question correctly in the latest run. `supermemory-rs` scored 100% in another run with the same code, so five questions are not enough to establish a stable accuracy difference.

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
