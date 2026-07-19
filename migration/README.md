# Legacy migration

This directory contains the JavaScript sidecar for importing v0.0.5 data. Normal requests do not use it.

On startup, `crates/supermemory/src/legacy.rs` checks for `~/.supermemory/data`. It decrypts that snapshot, runs `export.mjs` with Node.js, and imports the resulting JSON Lines into SQLite. The exporter opens the snapshot with the pinned PGlite version and the legacy PGlite runtime files.

Install the sidecar dependency with `npm ci --prefix migration` before importing legacy data. Set `SUPERMEMORY_MIGRATION_EXPORTER` only when the exporter lives outside this source tree.
