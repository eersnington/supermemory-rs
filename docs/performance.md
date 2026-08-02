# Performance Experiment Protocol

Do not alter more than one runtime policy in a run. A result is comparable only
when it uses the pinned MemoryBench commit, the same episode set, provider
model, answer model, machine, and release binary.

## Runtime Controls

The server logs the selected limits at startup. These environment variables are
intended for experiments; invalid values fall back to `config.toml`.

New installations write these defaults to `~/.supermemory-rs/config.toml` (or
next to an explicitly configured database). They are explained inline in the
file. The `4 / 1024` embedding default is the memory-safe choice from the
initial measurements; it remains subject to the repeat protocol below.

| Variable | Default | Allowed range | Purpose |
| --- | ---: | ---: | --- |
| `SUPERMEMORY_PROVIDER_CONCURRENCY` | 8 | 1-16 | In-flight provider extraction requests |
| `SUPERMEMORY_EMBEDDING_QUEUE_CAPACITY` | 64 | 1-1024 | Bounded embedding requests waiting for inference |
| `SUPERMEMORY_EMBEDDING_MAX_ITEMS` | 32 | 1-128 | Maximum inputs in one ONNX batch |
| `SUPERMEMORY_EMBEDDING_MAX_PADDED_TOKENS` | 8192 | 512-32768 | Maximum batch item-count times longest token length |

The server emits structured events for `memory extraction completed`,
`embedding_batch`, and `sqlite_write`. Run with `RUST_LOG=server=debug,memory_engine=info`
to retain writer queue wait observations.

## Objective

Reject a configuration when any hard constraint fails:

| Constraint | Initial limit |
| --- | ---: |
| Ready RSS | <= 225 MiB |
| Mean workload RSS | <= 320 MiB |
| Peak workload RSS | <= 500 MiB |
| Search p95 | <= 75 ms |
| Retrieval quality | Non-inferior on frozen questions |

Among qualifying configurations, minimize indexing wall time. Then minimize
search p95 and provider token use. Do not use Gemini answer correctness from a
five-question sample as an optimization signal.

## Run Record

Store one directory per run with:

```text
configuration.env
server.log
rss.csv                 # timestamp, rss_kib
memorybench-report.json
rss-summary.json
```

The run record must include the git commit, binary hash, benchmark commit,
machine model, provider model, answer model, question IDs, episode IDs, start
and end timestamps, and all runtime controls.

Run one experiment with:

```sh
bash scripts/run-locomo-experiment.sh
```

Set `SUPERMEMORY_RUN_ID` to an explicit identifier. The runner creates a fresh
database at `.performance/runs/<run-id>/`, persists server logs and RSS samples,
and never deletes another run's artifacts.

## Initial Batch-Geometry Results

These single runs used the pinned five-question LoCoMo sample, 95 episodes,
Gemini 2.5 Flash extraction, local BGE embeddings, and provider concurrency 8
on August 2, 2026. They establish that the ONNX padded-token limit controls the
workload memory peak. They do not select a production default: provider output
and latency vary between runs, and each configuration needs randomized repeats.

| Batch geometry | Indexing mean | Mean RSS | Peak RSS | Largest padded batch |
| --- | ---: | ---: | ---: | ---: |
| Default: 32 items / 8192 tokens | 434 s | 286 MiB | 849 MiB | 8140 tokens |
| 8 items / 2048 tokens | 433 s | 206 MiB | 420 MiB | 2028 tokens |
| 4 items / 1024 tokens | 441 s | 249 MiB | 264 MiB | 1014 tokens |

The large default batch showed no indexing-time benefit over either bounded
configuration. Provider extraction, rather than local embedding, dominated the
end-to-end wall time. Writer queue waits were negligible in these runs.

Artifacts are retained locally under `.performance/runs/` and intentionally
ignored by version control.

### Reproduce

Set `GOOGLE_API_KEY`, ensure the local model and ONNX Runtime paths from the
runner exist, then run each command from the repository root. Use new run IDs
for every repetition.

```sh
SUPERMEMORY_RUN_ID=default-r1 \
  bash scripts/run-locomo-experiment.sh

SUPERMEMORY_RUN_ID=geometry-8x2048-r1 \
SUPERMEMORY_EMBEDDING_MAX_ITEMS=8 \
SUPERMEMORY_EMBEDDING_MAX_PADDED_TOKENS=2048 \
  bash scripts/run-locomo-experiment.sh

SUPERMEMORY_RUN_ID=geometry-4x1024-r1 \
SUPERMEMORY_EMBEDDING_MAX_ITEMS=4 \
SUPERMEMORY_EMBEDDING_MAX_PADDED_TOKENS=1024 \
  bash scripts/run-locomo-experiment.sh
```

Each run writes `configuration.env`, `binary.sha256`, `server.log`, `rss.csv`,
`rss-summary.json`, and `memorybench-report.json`. Compare medians only after
at least three randomized repetitions per geometry.

## Experiment Sequence

### 1. Establish a Reference

Run the current defaults three times on the pinned 95-episode corpus. Randomize
run order in every later experiment. Record median and range; never compare a
single run to a different single run.

### 2. Attribute Embedding Memory

Keep provider concurrency fixed at 8. Run each configuration at least three
times:

```text
items/padded tokens
4/1024
8/2048
16/4096
32/8192
32/2048
8/8192
```

Inspect `embedding_batch` events alongside `rss.csv`.

- RSS tracks padded tokens: ONNX sequence workspace dominates.
- RSS tracks item count: per-item intermediates dominate.
- RSS stays high after one batch: ONNX arena retention dominates.
- RSS grows before inference: queued prepared inputs dominate.

Choose the smallest memory-safe geometry before changing provider concurrency.

### 3. Find the Provider-Concurrency Knee

Fix the selected embedding geometry. Test provider concurrency `1, 2, 4, 6, 8,
10`, three randomized repetitions each. Record provider p50/p95 duration,
rate-limit retries, input/output tokens, indexing wall time, and RSS.

Choose the smallest value within 5% of the fastest median indexing time that
does not violate a hard constraint.

### 4. Freeze Search Evaluation

Build one database snapshot and execute the same fixed question set for every
search policy. Use at least 100 questions while selecting policies and the full
available set for final acceptance. Measure Hit@K, precision, recall, MRR,
NDCG, duplicate@K, complete-fact coverage, context tokens, and p50/p95/p99.

Compare semantic only, lexical only, always hybrid, and conditional hybrid.
Retain a policy only when it is non-inferior on retrieval quality and improves a
measured resource or latency metric.

## Non-Comparable Runs

Do not compare runs that differ in any of these conditions:

- cold versus warm model or SQLite page cache
- source build time included in workload time
- changed question or episode IDs
- changed provider, model, prompt, or rate-limit state
- concurrent unrelated machine load
- answer-model correctness without fixed retrieval artifacts
