#!/usr/bin/env bash
# End-to-end runtime load spike: compares PrefetchPolicy variants
# against the real Runtime::load_model_with_config path (GgufFile::open
# + optional prefetch + MetalModel::load_with_options).
#
# Uses per-file cache invalidation via msync(MS_INVALIDATE) so each
# arm starts cold without disturbing global cache.
#
# Usage: ./scripts/bench-runtime-load.sh <path-to-model.gguf> [rounds]
#
# Arms:
#   D   policy=off, invalidate=true          (baseline: mmap demand paging)
#   A   policy=always, invalidate=true       (parallel-pread warmup)
#   AC  policy=cold-only, invalidate=true    (should behave like A when cold)
#   W   policy=always, invalidate=false      (warm-load regression check)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model.gguf> [rounds]" >&2
  exit 2
fi

MODEL="$1"
ROUNDS="${2:-1}"

if [[ ! -f "$MODEL" ]]; then
  echo "error: $MODEL not found" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT/target/release/examples/runtime_load_spike"

if [[ ! -x "$BIN" ]]; then
  echo "error: build first: cargo build --release -p qwen-llm --example runtime_load_spike" >&2
  exit 1
fi

OUT_DIR="$ROOT/target/runtime-load-results"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$OUT_DIR/${STAMP}.log"

FILE_SIZE_GIB=$(python3 -c "import os; print(f'{os.path.getsize(\"$MODEL\") / (1<<30):.2f}')")
{
  echo "model:  $MODEL ($FILE_SIZE_GIB GiB)"
  echo "rounds: $ROUNDS"
  echo "start:  $(date -u +%FT%TZ)"
  echo
} | tee "$LOG"

# Rotate arm order across rounds to catch order-dependent artefacts.
ARMS_R1=(D A AC W)
ARMS_R2=(AC W D A)
ARMS_R3=(W AC A D)

run_arm() {
  local arm="$1"
  local label="round${round}/${arm}"

  local args=()
  case "$arm" in
    D)  args=(--policy off       --invalidate) ;;
    A)  args=(--policy always    --invalidate) ;;
    AC) args=(--policy cold-only --invalidate) ;;
    # W deliberately does not pass --invalidate: it measures the cost
    # of a warm-cache load with prefetch on, which is the regression
    # scenario to watch.
    W)  args=(--policy always) ;;
    *)  echo "unknown arm: $arm" >&2; exit 1 ;;
  esac

  echo "===== $label =====" | tee -a "$LOG"
  "$BIN" "$MODEL" "${args[@]}" 2>&1 | tee -a "$LOG"
  echo | tee -a "$LOG"
}

for round in $(seq 1 "$ROUNDS"); do
  case "$round" in
    1) arms=("${ARMS_R1[@]}") ;;
    2) arms=("${ARMS_R2[@]}") ;;
    *) arms=("${ARMS_R3[@]}") ;;
  esac
  for arm in "${arms[@]}"; do
    run_arm "$arm"
  done
done

echo ">>> done. results: $LOG"
