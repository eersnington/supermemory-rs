# supermemory-rs

<img width="1468" height="807" alt="supermemory-rs landing page" src="https://github.com/user-attachments/assets/7c86ad07-9a88-4262-acd9-fd5e47986037" />

`supermemory-rs` is a low-memory Rust implementation of [Supermemory Local](https://github.com/supermemoryai/supermemory), based on `supermemory-server` v0.0.6. It keeps the local API and memory model while replacing the JavaScript runtime, PGlite, and workflow engine with Tokio, Axum, SQLite, and ONNX Runtime.

It works with the JavaScript and Python SDK smoke flows. It is not yet a complete replacement. See [verification](docs/verification.md) and [compatibility limits](#compatibility-limits).

## Run it

Requirements:

- Rust 1.88 or newer
- BGE model assets in `~/.supermemory/models/Xenova/bge-base-en-v1.5`
- ONNX Runtime from `~/.supermemory/runtime/ort-native/`

```sh
just install
supermemory-rs
```

Or, from a checkout:

```sh
cargo run -p supermemory
```

The service listens on `127.0.0.1:6767`. By default, data lives in `~/.supermemory-rs`.

| Option | Environment variable | Purpose |
| --- | --- | --- |
| `--bind` | `SUPERMEMORY_BIND` | Listen address |
| `--database` | `SUPERMEMORY_DATABASE` | SQLite database path |
| `--model` | `SUPERMEMORY_MODEL` | BGE model directory |
| `--ort-library` | `SUPERMEMORY_ORT_LIBRARY` | ONNX Runtime library |
| `--monitor` | `SUPERMEMORY_MONITOR` | Show live RSS in an interactive terminal |

Open [localhost:6767](http://localhost:6767) for the local page. API reference: `/v4/reference`; OpenAPI: `/v4/openapi`.

## Use it

Loopback requests do not need a bearer token.

```sh
curl -X POST http://127.0.0.1:6767/v3/documents \
  -H 'Content-Type: application/json' \
  -d '{"content":"A distinctive kingfisher observation"}'

curl -X POST http://127.0.0.1:6767/v4/search \
  -H 'Content-Type: application/json' \
  -d '{"q":"kingfisher","limit":10,"searchMode":"documents"}'
```

Ingestion returns before background indexing finishes. Poll `GET /v3/documents/:id` for status.

V4 search modes:

- `memories`: extracted facts
- `documents`: document chunks
- `hybrid`: both

## Memory extraction and configuration

An LLM provider is optional. Without one, the service still stores, embeds, and searches documents but does not extract memories.

On an interactive first run, choose a provider. For unattended startup, set one key:

| Provider | Environment variable |
| --- | --- |
| OpenAI or compatible endpoint | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Groq | `GROQ_API_KEY` |

When more than one key is present, the order is OpenAI, Anthropic, Gemini, then Groq. Use `OPENAI_BASE_URL` for a compatible OpenAI endpoint.

`~/.supermemory-rs/config.toml` is created on first run. It contains provider model names and bounded performance settings with inline comments. The default embedding batch is deliberately small (`4` items, `1024` padded tokens) to keep indexing memory bounded. See [performance experiments](docs/performance.md) before changing it.

## Authentication and migration

The server creates an API key at `~/.supermemory-rs/api-key`. Set `SUPERMEMORY_API_KEY` when binding beyond loopback, then send it as a bearer token:

```sh
curl http://127.0.0.1:6767/v3/documents/doc_id \
  -H "Authorization: Bearer $SUPERMEMORY_API_KEY"
```

Loopback access is based on the TCP peer address. Do not expose a local reverse proxy without authentication.

On first startup, the service imports compatible data and credentials from `~/.supermemory`, writes the result under `~/.supermemory-rs`, and leaves the source data unchanged.

## Development

```sh
cargo fmt --all --check
cargo check --workspace --locked
cargo lint
cargo test --workspace --locked
```

See [semantic porting guidelines](docs/semantic-porting.md) for compatibility rules.

## Compatibility limits

Missing work includes file and URL extraction, temporal query parsing, provider-backed query rewriting, adjacent-chunk context, Workers AI reranking, batch forgetting, direct memory mutation routes, and parts of profile generation. Some sentence and Markdown chunking edge cases also differ from `supermemory-server`.
