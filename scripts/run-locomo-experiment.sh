#!/usr/bin/env bash
set -euo pipefail

# Runs one controlled LoCoMo experiment. Each invocation owns a fresh database
# and artifact directory; repeat configurations through separate invocations.
REPO=$(cd "$(dirname "$0")/.." && pwd)
PERF_ROOT=${SUPERMEMORY_PERF_ROOT:-"$REPO/.performance"}
BENCH=${SUPERMEMORY_MEMORYBENCH_DIR:-"$PERF_ROOT/memorybench"}
RUN_ID=${SUPERMEMORY_RUN_ID:-"locomo-$(date +%Y%m%d-%H%M%S)"}
RUN_ROOT="$PERF_ROOT/runs/$RUN_ID"
PORT=${SUPERMEMORY_BENCH_PORT:-6768}
API_KEY=${SUPERMEMORY_BENCH_API_KEY:-local-memorybench-key}
LIMIT=${SUPERMEMORY_BENCH_LIMIT:-5}
MEMORYBENCH_COMMIT=118209a746d97d0d85e5a7234267f0b6962857e9
MODEL=${SUPERMEMORY_MODEL_DIR:-"$HOME/.supermemory/models/Xenova/bge-base-en-v1.5"}
ORT=${SUPERMEMORY_ORT_LIBRARY:-"$HOME/.supermemory/runtime/ort-native/onnxruntime-node/bin/napi-v6/darwin/arm64/libonnxruntime.1.23.2.dylib"}
SERVER_PID=
SAMPLER_PID=

if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "GOOGLE_API_KEY must be set for provider extraction and MemoryBench." >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "experiment directory already exists: $RUN_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/home" "$RUN_ROOT/data"

process_tree_rss() {
  ps -axo pid=,ppid=,rss= | awk -v root="$1" '
    { parent[$1]=$2; rss[$1]=$3 }
    END {
      included[root]=1
      do {
        changed=0
        for (pid in parent)
          if ((parent[pid] in included) && included[parent[pid]] && !(pid in included)) {
            included[pid]=1; changed=1
          }
      } while (changed)
      for (pid in included) if (included[pid]) total+=rss[pid]
      print total+0
    }'
}

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}

cleanup() {
  if [[ -n "$SAMPLER_PID" ]]; then kill "$SAMPLER_PID" 2>/dev/null || true; fi
  stop_server
}
trap cleanup EXIT

start_server() {
  local log=$1
  HOME="$RUN_ROOT/home" \
  GEMINI_API_KEY="$GOOGLE_API_KEY" \
  SUPERMEMORY_API_KEY="$API_KEY" \
  RUST_LOG="${RUST_LOG:-server=debug,memory_engine=info}" \
  "$REPO/target/release/supermemory-rs" \
    --bind "127.0.0.1:$PORT" \
    --database "$RUN_ROOT/data/supermemory.db" \
    --model "$MODEL" \
    --ort-library "$ORT" >"$log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null && return
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -100 "$log" >&2
      exit 1
    fi
    sleep 1
  done
  echo "server did not become healthy" >&2
  exit 1
}

cat >"$RUN_ROOT/data/config.toml" <<'TOML'
[providers.gemini]
model = "gemini-2.5-flash"
TOML

{
  printf 'run_id=%q\n' "$RUN_ID"
  printf 'memorybench_commit=%q\n' "$MEMORYBENCH_COMMIT"
  printf 'limit=%q\n' "$LIMIT"
  printf 'machine=%q\n' "$(uname -a)"
  if (cd "$REPO" && jj root >/dev/null 2>&1); then
    printf 'revision=%q\n' "$(cd "$REPO" && jj log -r @ --no-graph -T 'commit_id')"
  else
    printf 'revision=%q\n' "$(cd "$REPO" && git rev-parse HEAD)"
  fi
  env | grep '^SUPERMEMORY_\(PROVIDER_CONCURRENCY\|EMBEDDING_\|BENCH_\)' | sort || true
} >"$RUN_ROOT/configuration.env"

(cd "$REPO" && cargo build --release -p supermemory --locked)
shasum -a 256 "$REPO/target/release/supermemory-rs" >"$RUN_ROOT/binary.sha256"
if [[ ! -d "$BENCH/.git" ]]; then
  git clone https://github.com/supermemoryai/memorybench.git "$BENCH"
fi
git -C "$BENCH" fetch --quiet origin "$MEMORYBENCH_COMMIT"
git -C "$BENCH" checkout --quiet "$MEMORYBENCH_COMMIT"
(cd "$BENCH" && bun install --frozen-lockfile)

start_server "$RUN_ROOT/server.log"
READY_RSS_KIB=$(process_tree_rss "$SERVER_PID")
printf 'epoch,rss_kib\n' >"$RUN_ROOT/rss.csv"
(
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    printf '%s,%s\n' "$(date +%s)" "$(process_tree_rss "$SERVER_PID")" >>"$RUN_ROOT/rss.csv"
    sleep 1
  done
) &
SAMPLER_PID=$!

(
  cd "$BENCH"
  SUPERMEMORY_API_KEY="$API_KEY" SUPERMEMORY_BASE_URL="http://127.0.0.1:$PORT" \
  GOOGLE_API_KEY="$GOOGLE_API_KEY" bun run src/index.ts run \
    --provider supermemory --benchmark locomo --judge gemini-2.5-flash \
    --answering-model gemini-2.5-flash --run-id "$RUN_ID" --limit "$LIMIT" --force
)

kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
SAMPLER_PID=
stop_server
read -r MEAN_RSS_KIB PEAK_RSS_KIB < <(
  awk -F, 'NR>1 { total+=$2; n++; if ($2>peak) peak=$2 } END { printf "%.0f %.0f\n", total/n, peak }' "$RUN_ROOT/rss.csv"
)
start_server "$RUN_ROOT/restart.log"
sleep 1
RESTART_RSS_KIB=$(process_tree_rss "$SERVER_PID")
stop_server
cp "$BENCH/data/runs/$RUN_ID/report.json" "$RUN_ROOT/memorybench-report.json"
printf '{"readyRssKiB":%s,"workloadMeanRssKiB":%s,"workloadPeakRssKiB":%s,"populatedRestartRssKiB":%s}\n' \
  "$READY_RSS_KIB" "$MEAN_RSS_KIB" "$PEAK_RSS_KIB" "$RESTART_RSS_KIB" >"$RUN_ROOT/rss-summary.json"
printf 'experiment complete: %s\n' "$RUN_ROOT"
