#!/usr/bin/env bash
# Runs five live extraction jobs against fresh storage and fails on retries or panics.
set -Eeuo pipefail

: "${GOOGLE_API_KEY:?set GOOGLE_API_KEY before running this smoke test}"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA=$(mktemp -d /tmp/supermemory-five-document.XXXXXX)
LOG=$(mktemp /tmp/supermemory-five-document.XXXXXX.log)
PORT=${SUPERMEMORY_SMOKE_PORT:-6771}
PID=""

cleanup() {
  local status=$?
  if ((status != 0)); then
    printf '\n=== server log ===\n' >&2
    cat "$LOG" >&2 || true
  fi
  if [[ -n "$PID" ]]; then
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
  rm -rf "$DATA" "$LOG"
  return "$status"
}
trap cleanup EXIT

(cd "$ROOT" && CARGO_INCREMENTAL=0 cargo build --release -p supermemory --bin supermemory-rs)

RUST_LOG="${RUST_LOG:-server=info}" \
SUPERMEMORY_BIND="127.0.0.1:$PORT" \
SUPERMEMORY_DATA="$DATA" \
GEMINI_API_KEY="$GOOGLE_API_KEY" \
"$ROOT/target/release/supermemory-rs" >"$LOG" 2>&1 &
PID=$!

for _ in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

payloads=(
  '{"content":"Caroline attended an LGBTQ support group on 7 May 2023.","customId":"smoke-1","metadata":{"date":"2023-05-07"}}'
  '{"content":"Caroline started volunteering at the community garden on 12 June 2023.","customId":"smoke-2","metadata":{"date":"2023-06-12"}}'
  '{"content":"Caroline prefers concise status updates sent on Friday afternoons.","customId":"smoke-3","metadata":{"date":"2023-07-01"}}'
  '{"content":"Caroline booked a train to Edinburgh for 18 August 2023.","customId":"smoke-4","metadata":{"date":"2023-08-10"}}'
  '{"content":"Caroline decided to use SQLite WAL mode for the local project.","customId":"smoke-5","metadata":{"date":"2023-09-02"}}'
)
ids=()
for payload in "${payloads[@]}"; do
  created=$(curl -fsS -X POST "http://127.0.0.1:$PORT/v3/documents" \
    -H 'content-type: application/json' -d "$payload")
  ids+=("$(printf '%s' "$created" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')")
done

for id in "${ids[@]}"; do
  status=""
  for _ in $(seq 1 300); do
    document=$(curl -fsS "http://127.0.0.1:$PORT/v3/documents/$id")
    status=$(printf '%s' "$document" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    case "$status" in
      done) break ;;
      failed) printf '%s\n' "$document" >&2; exit 1 ;;
    esac
    sleep 1
  done
  [[ "$status" == done ]]
done

result=$(curl -fsS -X POST "http://127.0.0.1:$PORT/v4/search" \
  -H 'content-type: application/json' \
  -d '{"q":"What database mode did Caroline choose?","limit":10,"searchMode":"memories"}')
printf '%s' "$result" | python3 -c 'import json,sys; data=json.load(sys.stdin); assert any("WAL" in item["memory"] for item in data["results"]); print(json.dumps(data))'
if rg -n 'panic|service degraded|worker stopped|memory extraction attempt failed' "$LOG"; then
  exit 1
fi
printf '\n=== stage timings ===\n'
rg -n 'document job completed|memory job completed|search completed' "$LOG"
