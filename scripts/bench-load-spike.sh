#!/usr/bin/env bash
# Compare cold-cache load-spike arms across the four candidate strategies.
#
# Uses per-file targeted cache invalidation (`msync(MS_INVALIDATE)`)
# rather than global `sudo purge`. Only touches the target GGUF's
# cache pages, not any other in-flight work's cache.
#
# Usage: ./scripts/bench-load-spike.sh <path-to-model.gguf> [rounds]
#   rounds defaults to 1
#
# Arms:
#   D  baseline               (just GgufFile::open + touch)
#   C  madvise(WILLNEED)      (open + madvise + touch)
#   A  parallel-pread warmup  (prefetch + open + touch)
#   AC prefetch + madvise      (both)
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
BIN="$ROOT/target/release/examples/load_spike"

if [[ ! -x "$BIN" ]]; then
  echo "error: build the example first:" >&2
  echo "  cargo build --release -p qwen-llm --example load_spike" >&2
  exit 1
fi

OUT_DIR="$ROOT/target/load-spike-results"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$OUT_DIR/${STAMP}.log"

FILE_SIZE_GIB=$(python3 -c "import os; print(f'{os.path.getsize(\"$MODEL\") / (1<<30):.2f}')")
{
  echo "model: $MODEL ($FILE_SIZE_GIB GiB)"
  echo "rounds: $ROUNDS"
  echo "start: $(date -u +%FT%TZ)"
  echo "invalidation: per-file msync(MS_INVALIDATE) — no global purge"
  echo
} | tee "$LOG"

# Round 1 fixed order; rounds >= 2 rotate to expose any within-round drift.
ARMS_R1=(D C A AC)
ARMS_R2=(AC A C D)
ARMS_R3=(C AC D A)

run_arm() {
  local arm="$1"
  local label="round${round}/${arm}"

  local flags=(--invalidate --touch-all-pages)
  case "$arm" in
    D)  ;;
    C)  flags+=(--madvise-willneed) ;;
    A)  flags+=(--prefetch) ;;
    AC) flags+=(--prefetch --madvise-willneed) ;;
    *)  echo "unknown arm: $arm" >&2; exit 1 ;;
  esac

  echo "===== $label =====" | tee -a "$LOG"
  "$BIN" "$MODEL" "${flags[@]}" 2>&1 | tee -a "$LOG"
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
