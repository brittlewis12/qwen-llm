#!/usr/bin/env bash
# End-to-end first-byte spike: measures how the cold-load prefetch win
# propagates to time-to-first-token.
#
# Usage: ./scripts/bench-first-byte.sh <path-to-model.gguf> [rounds] [prompt]
#
# Arms:
#   D    policy=off,       invalidate=true  (baseline cold)
#   A    policy=always,    invalidate=true  (parallel-pread warmup, cold)
#   AC   policy=cold-only, invalidate=true  (auto-gated, cold => prefetch)
#   Wo   policy=off,       invalidate=false (baseline warm)
#   Wa   policy=always,    invalidate=false (regression check)
#   Wc   policy=cold-only, invalidate=false (regression check — should skip)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model.gguf> [rounds] [prompt]" >&2
  exit 2
fi

MODEL="$1"
ROUNDS="${2:-1}"
PROMPT="${3:-The capital of France is}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT/target/release/examples/first_byte_spike"

if [[ ! -x "$BIN" ]]; then
  echo "error: build first: cargo build --release -p qwen-llm --example first_byte_spike" >&2
  exit 1
fi

OUT_DIR="$ROOT/target/first-byte-results"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$OUT_DIR/${STAMP}.log"

FILE_SIZE_GIB=$(python3 -c "import os; print(f'{os.path.getsize(\"$MODEL\") / (1<<30):.2f}')")
{
  echo "model:  $MODEL ($FILE_SIZE_GIB GiB)"
  echo "prompt: $PROMPT"
  echo "rounds: $ROUNDS"
  echo "start:  $(date -u +%FT%TZ)"
  echo
} | tee "$LOG"

# Rotate arm order to expose position-dependent artefacts.
ARMS_R1=(D A AC Wo Wa Wc)
ARMS_R2=(Wc Wa Wo AC A D)
ARMS_R3=(A D Wa Wc AC Wo)

run_arm() {
  local arm="$1"
  local label="round${round}/${arm}"

  local args=(--prompt "$PROMPT")
  case "$arm" in
    D)  args+=(--policy off       --invalidate) ;;
    A)  args+=(--policy always    --invalidate) ;;
    AC) args+=(--policy cold-only --invalidate) ;;
    Wo) args+=(--policy off) ;;
    Wa) args+=(--policy always) ;;
    Wc) args+=(--policy cold-only) ;;
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
