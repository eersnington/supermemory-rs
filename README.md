# supermemory-rs

Early Rust implementation of a self-hosted Supermemory-compatible service. It does not yet claim full API, retrieval-quality, or performance parity.

Development follows the [semantic porting guidelines](docs/semantic-porting.md): preserve behavior recovered from `supermemory-server` v0.0.5 while replacing its generic runtime infrastructure.

Executed compatibility checks and remaining gaps are tracked in [verification status](docs/verification.md).

## Current status

The executable serves document creation and lookup, V3/V4 semantic search, memory extraction and reconciliation, graph context, profiles, direct forgetting, and request-scoped organizations. A revision-guarded worker chunks, embeds, and atomically indexes queued documents; interrupted jobs recover on restart. First startup imports compatible legacy organizations, API keys, documents, chunks, vectors, memories, relations, sources, spaces, and encrypted provider credentials without modifying `~/.supermemory`.

Start it directly for loopback-only access:

```sh
supermemory-rs
```

Set `SUPERMEMORY_API_KEY` only when non-loopback clients need bearer authentication.

The default address is `127.0.0.1:6767`. Processing uses the recovered chunk limits, UTF-16 accounting, overlap, and local BGE model. Exact Compromise sentence boundaries, specialized Markdown table/code splitting, extraction formats, several secondary routes, and retrieval-quality parity remain unfinished.

Submit and search a document:

```sh
curl -X POST http://127.0.0.1:6767/v3/documents \
  -H 'Content-Type: application/json' \
  -d '{"content":"A distinctive kingfisher observation"}'

curl -X POST http://127.0.0.1:6767/v4/search \
  -H 'Content-Type: application/json' \
  -d '{"q":"kingfisher","limit":10}'
```

Requests normally require the configured bearer key. For local development compatibility, requests with no `Authorization` or session cookie automatically use the local identity only when the TCP peer address is IPv4 or IPv6 loopback. Supplying an invalid bearer key still fails. When using a reverse proxy, requests are authenticated as connections from the proxy; do not run an unauthenticated proxy on the same host.

## Development

```sh
cargo fmt --all --check
cargo check --workspace --locked
cargo lint
cargo test --workspace --locked
cargo nextest run --workspace --locked
```

`cargo lint` runs Clippy for the entire workspace, all targets, and all features with warnings denied. The nextest command requires `cargo-nextest` to be installed.

## Document identity

As in v0.0.5, custom IDs are scoped by their exact ordered container-tag array when documents are created. `GET /v3/documents/:id` has no tag parameter and falls back from internal ID to custom ID, so if several documents share a custom ID across different ordered tag arrays, the returned matching document is intentionally unspecified.
