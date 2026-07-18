# Semantic porting guidelines

This project is a Rust port of `supermemory-server` v0.0.5. It is not a new memory engine inspired by Supermemory.

The implementation may replace Bun, Hono, PGlite, Drizzle, Rivet, and other generic runtime machinery. It must preserve Supermemory's externally visible behavior and application-specific decisions.

## Source of truth

Use these sources in order:

1. Behavior observed from the v0.0.5 binary.
2. Implementation recovered from the binary's embedded Bun bundle.
3. The v0.0.5 self-hosting documentation.

When they disagree, the binary wins. Record the disagreement in the test that covers it.

Do not invent behavior because it seems reasonable. If a rule cannot be traced to the bundle or demonstrated against the binary, treat it as unknown.

## What to port

Port Supermemory-specific behavior, including:

- routes, validation, response bodies, errors, and status transitions;
- prompts, model selection, structured outputs, and retry rules;
- content normalization, chunking, and embedding preprocessing;
- document identity and update behavior;
- memory extraction, reconciliation, versioning, and forgetting;
- graph relationships and profile projection;
- search filtering, score fusion, ranking constants, and tie-breaking;
- configuration, authentication, startup, and persistence behavior.

These rules belong in Rust code or versioned assets with tests that identify their evidence.

## What may change

Generic infrastructure may be replaced when the replacement preserves behavior:

| Existing implementation | Rust implementation |
| --- | --- |
| Bun and Hono | Tokio and Axum |
| PGlite and Drizzle | SQLite and rusqlite |
| Transformers.js | ONNX Runtime and tokenizers |
| pgvector candidate search | Embedded vector index plus exact reranking |
| Rivet workflow execution | Persisted Rust job state machine |
| Better Auth local setup | Equivalent local identity and API-key handling |

An infrastructure replacement must not silently change ordering, retries, transaction boundaries, durability, or error behavior.

## Implementation workflow

Implement one observable behavior at a time:

```text
locate behavior in the bundle
  -> confirm it against the binary
  -> capture a failing Rust integration test
  -> port the behavior
  -> compare both implementations
  -> optimize without changing the test result
```

Keep recovered evidence out of production interfaces. Tests should exercise crate interfaces rather than private implementation details.

## Compatibility requirements

A feature is compatible only when all applicable checks pass:

- Existing Supermemory SDK calls require no application changes other than the server URL.
- Requests, responses, status codes, and errors match v0.0.5.
- Persisted jobs survive interruption and resume with equivalent behavior.
- The same model configuration produces equivalent chunks, memories, profiles, and search results within defined numeric tolerances.
- Existing local data can be migrated without losing documents, memories, relationships, files, or credentials.
- MemoryBench quality does not regress.
- Memory usage and latency meet the project's replacement targets.

Matching the API while changing memory or search semantics is not sufficient.

## Rules for dependencies

Add a dependency when it replaces generic machinery or implements a standard format. Do not use a dependency to substitute an unverified memory algorithm.

Before adding one, document:

- which existing implementation it replaces;
- which behavior must remain unchanged;
- how parity will be tested;
- its expected effect on memory, latency, and binary size.

Prefer direct libraries over servers and frameworks. The local product should remain one process unless profiling proves that process isolation is needed for the embedding model.

## Optimization rules

Establish parity before optimizing. Keep a simple reference path when an optimization can change results.

For example, an approximate vector index may select candidates, but final ranking must use exact stored vectors and the recovered Supermemory scoring rules. A faster implementation is not acceptable if it changes benchmark quality or observable ordering beyond the agreed tolerance.

Every performance change needs measurements against the same corpus and model configuration used for the Bun baseline. Track steady RSS, peak RSS, throughput, and p95 latency.

## Definition of done

Do not describe the project or a subsystem as a drop-in replacement while known v0.0.5 behavior is missing or unverified. State the implemented compatibility surface precisely.

The full replacement is done when the process, API, behavior, and migration contracts are covered by executable tests and the quality and performance gates pass.
