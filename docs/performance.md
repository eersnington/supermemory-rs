# Performance

This project optimizes for bounded memory first, then indexing time. Do not compare runs that use different code, models, episode/question sets, machine load, or cache state.

## Current result

These one-off runs used the pinned five-question LoCoMo sample, 95 episodes, Gemini 2.5 Flash extraction, local BGE embeddings, and provider concurrency 8 on August 2, 2026.

| Embedding batch | Indexing mean | Mean RSS | Peak RSS |
| --- | ---: | ---: | ---: |
| Old default: `32 / 8192` | 434 s | 286 MiB | 849 MiB |
| `8 / 2048` | 433 s | 206 MiB | 420 MiB |
| Current default: `4 / 1024` | 441 s | 249 MiB | 264 MiB |

Large batches did not improve end-to-end indexing, because provider extraction dominated the run. Smaller batches cut the memory peak sharply. The current default is `4 / 1024` because it had the lowest observed peak.

These are single runs, not a final benchmark result. Provider output varies, so the retrieval and answer scores from these runs are not evidence that batch size affects quality.

## Configure

New installations write these settings, with comments, to `~/.supermemory-rs/config.toml` or beside a custom database:

```toml
[performance]
provider_concurrency = 8
embedding_queue_capacity = 64
embedding_max_items = 4
embedding_max_padded_tokens = 1024
```

Environment variables override the file for an experiment:

| Variable | Range |
| --- | ---: |
| `SUPERMEMORY_PROVIDER_CONCURRENCY` | 1-16 |
| `SUPERMEMORY_EMBEDDING_QUEUE_CAPACITY` | 1-1024 |
| `SUPERMEMORY_EMBEDDING_MAX_ITEMS` | 1-128 |
| `SUPERMEMORY_EMBEDDING_MAX_PADDED_TOKENS` | 512-32768 |

## Reproduce

Set `GOOGLE_API_KEY` and ensure the local BGE model and ONNX Runtime paths exist. Each command builds a release binary, creates a fresh database, and writes artifacts under `.performance/runs/<run-id>/`.

```sh
SUPERMEMORY_RUN_ID=default-r1 \
  bash scripts/run-locomo-experiment.sh

SUPERMEMORY_RUN_ID=batch-8x2048-r1 \
SUPERMEMORY_EMBEDDING_MAX_ITEMS=8 \
SUPERMEMORY_EMBEDDING_MAX_PADDED_TOKENS=2048 \
  bash scripts/run-locomo-experiment.sh
```

The run directory contains the effective configuration, binary hash, server log, RSS samples, restart RSS, and MemoryBench report. It is ignored by version control.

For all LoCoMo questions, set `SUPERMEMORY_BENCH_LIMIT=1986`. This has significant provider cost.

## Before changing defaults

Use at least three randomized repetitions per configuration. Reject a configuration if it exceeds any of these initial limits:

| Metric | Limit |
| --- | ---: |
| Ready RSS | 225 MiB |
| Mean workload RSS | 320 MiB |
| Peak workload RSS | 500 MiB |
| Search p95 | 75 ms |

Among qualifying configurations, choose the fastest median indexing time. Then test provider concurrency `1, 2, 4, 6, 8, 10` with the selected embedding geometry. Retrieval policy changes need a frozen corpus of at least 100 questions; do not use a five-question Gemini answer score to select them.
