# Compatibility checks

This directory contains manual checks for public clients and application programming interface (API) performance. The server does not load these files at runtime.

- `sdk-js.mjs` checks the JavaScript software development kit (SDK) against a running server
- `sdk-python.py` checks the Python SDK against a running server
- `tools.mjs` checks the `@supermemory/tools` definitions
- `api-smoke.mjs` measures ingestion throughput and API latency

Install the JavaScript dependencies with `bun install --cwd compat --frozen-lockfile`. Each script reads its server URL and API key from the environment.
