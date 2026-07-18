# supermemory-rs

Early Rust implementation of a self-hosted Supermemory-compatible service. It does not yet claim full API, retrieval-quality, or performance parity.

Development follows the [semantic porting guidelines](docs/semantic-porting.md): preserve behavior recovered from `supermemory-server` v0.0.5 while replacing its generic runtime infrastructure.

## Current status

The executable currently serves public `GET /health` plus `POST /v3/documents` and `GET /v3/documents/:id`. Documents are scoped to a generated, persisted local organization. Creation implements v0.0.5-compatible content sanitization, SHA-1 duplicate identity, organization/container-scoped custom IDs, and atomic upsert/job behavior. Ordered schema migrations are tracked in `schema_migrations`.

Start it with a private bearer key in the environment:

```sh
SUPERMEMORY_API_KEY=replace-with-a-private-key cargo run -p supermemory
```

The default address is `127.0.0.1:6767`. Processing workers, search, retrieval-quality parity, and the generated-key setup wizard are not implemented.

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
