#!/usr/bin/env bash
set -euo pipefail

# Replays captured LoCoMo questions against a populated database. Record mode
# freezes ranked responses; check mode rejects any response change or slow p95.
MODE=${1:-}
DATABASE=${2:-}
RESULTS_DIR=${3:-}
BASELINE=${4:-}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ITERATIONS=${SUPERMEMORY_FROZEN_ITERATIONS:-10}
MAX_P95_MS=${SUPERMEMORY_FROZEN_MAX_P95_MS:-75}
PORT=${SUPERMEMORY_FROZEN_PORT:-6769}
MODEL=${SUPERMEMORY_MODEL_DIR:-"$HOME/.supermemory/models/Xenova/bge-base-en-v1.5"}
ORT=${SUPERMEMORY_ORT_LIBRARY:-"$HOME/.supermemory/runtime/ort-native/onnxruntime-node/bin/napi-v6/darwin/arm64/libonnxruntime.1.23.2.dylib"}

usage() {
  echo "usage: $0 record|check DATABASE RESULTS_DIR BASELINE.json" >&2
  exit 2
}

[[ "$MODE" == "record" || "$MODE" == "check" ]] || usage
[[ -n "$DATABASE" && -n "$RESULTS_DIR" && -n "$BASELINE" ]] || usage
for command in cargo curl jq sqlite3; do
  command -v "$command" >/dev/null || { echo "required command not found: $command" >&2; exit 1; }
done
[[ -f "$DATABASE" ]] || { echo "database not found: $DATABASE" >&2; exit 1; }
compgen -G "$RESULTS_DIR/*.json" >/dev/null || { echo "no question results found under $RESULTS_DIR" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "embedding model not found: $MODEL" >&2; exit 1; }
[[ -f "$ORT" ]] || { echo "ONNX Runtime library not found: $ORT" >&2; exit 1; }
if [[ "$MODE" == "check" && ! -f "$BASELINE" ]]; then
  echo "baseline not found: $BASELINE; create it with record mode first" >&2
  exit 1
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/supermemory-frozen.XXXXXX")
SERVER_PID=
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

# SQLite backup produces a consistent snapshot even when the source has a WAL.
sqlite3 "$DATABASE" ".backup '$WORK/supermemory.db'"
cargo build --release -p supermemory --locked --manifest-path "$REPO/Cargo.toml"
"$REPO/target/release/supermemory-rs" \
  --bind "127.0.0.1:$PORT" \
  --database "$WORK/supermemory.db" \
  --model "$MODEL" \
  --ort-library "$ORT" >"$WORK/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 120); do
  if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -100 "$WORK/server.log" >&2
    exit 1
  fi
  sleep 1
done
curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null || {
  echo "server did not become healthy" >&2
  exit 1
}

request() {
  local question_file=$1
  local question container_tag body
  question=$(jq -r .question "$question_file")
  container_tag=$(jq -r .containerTag "$question_file")
  body=$(jq -nc --arg q "$question" --arg tag "$container_tag" \
    '{q:$q,containerTag:$tag,limit:30,threshold:0.3,searchMode:"hybrid",include:{summaries:true,chunks:true}}')
  curl --silent --fail -X POST "http://127.0.0.1:$PORT/v4/search" \
    -H 'content-type: application/json' -d "$body"
}

CURRENT="$WORK/current.json"
printf '{}\n' >"$CURRENT"
for question_file in "$RESULTS_DIR"/*.json; do
  question_id=$(jq -r .questionId "$question_file")
  response=$(request "$question_file")
  jq --arg id "$question_id" --argjson results "$(jq -c .results <<<"$response")" \
    '. + {($id): $results}' "$CURRENT" >"$CURRENT.next"
  mv "$CURRENT.next" "$CURRENT"
done
jq --sort-keys . "$CURRENT" >"$CURRENT.sorted"

if [[ "$MODE" == "record" ]]; then
  mkdir -p "$(dirname "$BASELINE")"
  cp "$CURRENT.sorted" "$BASELINE"
  echo "recorded frozen search baseline: $BASELINE"
else
  jq --sort-keys . "$BASELINE" >"$WORK/baseline.sorted"
  if ! diff -u "$WORK/baseline.sorted" "$CURRENT.sorted"; then
    echo "frozen search quality gate failed: ranked responses changed" >&2
    exit 1
  fi
fi

TIMINGS="$WORK/timings.txt"
for question_file in "$RESULTS_DIR"/*.json; do
  for _ in $(seq 1 "$ITERATIONS"); do
    request "$question_file" | jq -r .timing >>"$TIMINGS"
  done
done
sort -n "$TIMINGS" >"$TIMINGS.sorted"
COUNT=$(wc -l <"$TIMINGS.sorted" | tr -d ' ')
P95_INDEX=$(( (COUNT * 95 + 99) / 100 ))
P95=$(sed -n "${P95_INDEX}p" "$TIMINGS.sorted")
MEAN=$(awk '{ total += $1 } END { printf "%.1f", total / NR }' "$TIMINGS")
printf 'frozen search: requests=%s mean_ms=%s p95_ms=%s\n' "$COUNT" "$MEAN" "$P95"
awk -v measured="$P95" -v maximum="$MAX_P95_MS" 'BEGIN { exit !(measured <= maximum) }' || {
  echo "frozen search performance gate failed: p95 ${P95}ms exceeds ${MAX_P95_MS}ms" >&2
  exit 1
}
