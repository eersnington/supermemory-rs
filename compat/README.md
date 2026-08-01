# Compatibility checks

This directory contains manual checks for public clients and application programming interface (API) performance. The server does not load these files at runtime.

- `sdk-js.mjs` checks the JavaScript software development kit (SDK) against a running server
- `sdk-python.py` checks the Python SDK against a running server
- `tools.mjs` checks the `@supermemory/tools` definitions
- `api-smoke.mjs` measures ingestion throughput and API latency
- `parity.mjs` executes the v0.0.6 black-box operation matrix against the
  upstream binary and a Rust candidate, normalizes nondeterministic values, and
  compares the resulting fixtures
- `v0.0.6-cases.mjs` is the complete in-scope endpoint and validation matrix;
  OAuth connection-provider routes are deliberately absent

Install the JavaScript dependencies with `bun install --cwd compat --frozen-lockfile`. Each script reads its server URL and API key from the environment.

Run a differential probe after starting the Rust server on a separate port:

```sh
bun compat/parity.mjs \
  --oracle http://127.0.0.1:6767 \
  --candidate http://127.0.0.1:6768
```

The first run writes reviewed upstream fixtures under `compat/fixtures/v0.0.6`.
Later runs fail on status, normalized response-shape, or body differences. Use
`--accept-oracle` only when intentionally refreshing evidence from the exact
upstream binary. The runner records OpenAPI disagreements as observations but
does not use OpenAPI as an assertion source.
