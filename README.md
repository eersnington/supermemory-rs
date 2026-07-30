# supermemory-rs

<img width="1468" height="807" alt="image" src="https://github.com/user-attachments/assets/7c86ad07-9a88-4262-acd9-fd5e47986037" />

---

A low-memory Rust reimplementation of [Supermemory Local](https://github.com/supermemoryai/supermemory), based on `supermemory-server` [v0.0.5 release](https://github.com/supermemoryai/supermemory/releases/tag/server-v0.0.5).

The goal is to run the same local Supermemory service with a much smaller memory footprint. Instead of Bun, Hono, PGlite, Drizzle, Rivet, and Transformers.js, this port uses Tokio, Axum, SQLite, rusqlite, and ONNX Runtime in a single process. Supermemory's API and memory behavior should remain the same; the runtime underneath it is what changes.

> ⚠️ Note: This is still a work in progress. The implemented routes work with the Supermemory SDKs, but some v0.0.5 behavior is not available yet.

## Why this exists

`supermemory-server` carries a large JavaScript runtime, an embedded Postgres database, and a separate workflow engine. That is expensive for a service meant to run on a small local machine.

`supermemory-rs` replaces that infrastructure while preserving Supermemory-specific behavior:

- V3 document ingestion and lookup
- V3 and V4 semantic search
- Memory extraction, reconciliation, versioning, and forgetting
- Memory relationships and profile projection
- Container tags, metadata filters, and organization-scoped data
- Local BGE embeddings with the same 768-dimensional model
- Migration from an existing `~/.supermemory` installation

The worker uses a durable, revision-guarded queue. Chunks and embeddings are committed together, interrupted jobs resume after restart, and stale workers cannot overwrite newer document revisions.

See the [semantic porting guidelines](docs/semantic-porting.md) for the compatibility rules that guide the implementation.

## Requirements

- Rust 1.88 or newer
- The BGE tokenizer and ONNX model under `~/.supermemory/models/Xenova/bge-base-en-v1.5`
- The ONNX Runtime library used by Supermemory under `~/.supermemory/runtime/ort-native/`

The model and runtime paths can be changed with `--model` and `--ort-library`, or with `SUPERMEMORY_MODEL` and `SUPERMEMORY_ORT_LIBRARY`.

## Install and run

```sh
cargo install --path crates/supermemory --locked
supermemory-rs
```

This installs the executable at `~/.cargo/bin/supermemory-rs`. Run the install command again after updating the source.

To run directly from the checkout during development:

```sh
cargo run -p supermemory
```

The server listens on `127.0.0.1:6767` and stores its database and configuration in `~/.supermemory-rs`.

Other startup options:

```text
--bind <address>          SUPERMEMORY_BIND
--database <path>        SUPERMEMORY_DATABASE
--model <path>           SUPERMEMORY_MODEL
--ort-library <path>     SUPERMEMORY_ORT_LIBRARY
```

Open [http://localhost:6767](http://localhost:6767) for the local landing page. The API reference is available at `/v4/reference`, and the OpenAPI document is at `/v4/openapi`.

## Add and search documents

Local requests do not need an authorization header when the TCP connection comes directly from an IPv4 or IPv6 loopback address.

```sh
curl -X POST http://127.0.0.1:6767/v3/documents \
  -H 'Content-Type: application/json' \
  -d '{"content":"A distinctive kingfisher observation"}'

curl -X POST http://127.0.0.1:6767/v4/search \
  -H 'Content-Type: application/json' \
  -d '{"q":"kingfisher","limit":10,"searchMode":"documents"}'
```

Document ingestion is asynchronous. `POST /v3/documents` returns the document ID and its current status; use `GET /v3/documents/:id` to follow processing.

V4 search supports three modes:

- `memories` searches extracted memories
- `documents` searches document chunks
- `hybrid` searches both and gives memory results a small ranking boost

Without a loaded embedding model, document search falls back to SQLite FTS5. Memory search requires embeddings.

## Memory extraction

An LLM provider is optional. Without one, the server still chunks, embeds, stores, and searches documents, but it does not extract memories.

On an interactive first run, choose OpenAI, Anthropic, Gemini, or skip provider setup. The selected key is encrypted in `~/.supermemory-rs/env.enc` with owner-only file permissions.

For unattended startup, set one of these variables:

| Provider | Environment variable |
| --- | --- |
| OpenAI or OpenAI-compatible | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Groq | `GROQ_API_KEY` |

When several keys are present, the server chooses OpenAI, Anthropic, Gemini, then Groq. Set `OPENAI_BASE_URL` for an OpenAI-compatible endpoint.

Provider models live in `~/.supermemory-rs/config.toml`:

```toml
[providers.openai]
model = "gpt-5.6-luna"
reasoning_effort = "medium"

[providers.anthropic]
model = "claude-haiku-4-5"

[providers.gemini]
model = "gemini-3.5-flash"

[providers.groq]
model = "openai/gpt-oss-120b"
```

Model names are read from this file rather than environment variables.

## Authentication

The server creates a local API key in `~/.supermemory-rs/api-key`. Set `SUPERMEMORY_API_KEY` to supply your own key, especially when binding to a non-loopback address.

Send the key as a bearer token:

```sh
curl http://127.0.0.1:6767/v3/documents/doc_id \
  -H "Authorization: Bearer $SUPERMEMORY_API_KEY"
```

Loopback authentication is based on the TCP peer address. A reverse proxy on the same machine appears to be a local client, so do not expose an unauthenticated local proxy.

## Move from Supermemory Local

On first startup, `supermemory-rs` imports compatible organizations, API keys, documents, chunks, vectors, memories, relations, sources, spaces, and provider credentials from `~/.supermemory`. It writes the imported data to `~/.supermemory-rs` and leaves the original Supermemory Local data unchanged.

## Document identity

Custom IDs are scoped by the exact ordered `containerTags` array used during creation. `GET /v3/documents/:id` cannot accept tags and falls back from an internal ID to a custom ID. If the same custom ID exists under more than one ordered tag array, which matching document is returned is unspecified.

## Development

```sh
cargo fmt --all --check
cargo check --workspace --locked
cargo lint
cargo test --workspace --locked
cargo nextest run --workspace --locked
```

`cargo lint` runs Clippy across the workspace with warnings denied. `cargo nextest` requires a separate installation.

## Compatibility limits

This is not a complete replacement for `supermemory-server` yet. Missing work includes file and URL extraction, temporal query parsing, provider-backed query rewriting, adjacent-chunk context, Workers AI reranking, batch forgetting, direct memory mutation routes, and parts of profile generation. Sentence and Markdown splitting also differ in some edge cases.

The current compatibility and resource measurements are documented in [docs/verification.md](docs/verification.md).
