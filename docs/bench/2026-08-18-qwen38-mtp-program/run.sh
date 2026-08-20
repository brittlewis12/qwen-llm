#!/bin/bash
# P0 MTP-program baseline runner — Qwen 3.8.
# Preregistration: README.md in this same dir.
#
# Derivation: sed-copy from
#   docs/bench/2026-08-18-qwen36-mtp-program/run.sh
# with only the model path changed to Qwen3.8-27B-Q4_K_M.gguf. See README.md
# ERRATUM section for why 3.8 supersedes 3.6-MTP as the P0 model.
#
# Invocation (from this dir):
#   ./run.sh 2>&1 | tee run.log
#
# Estimated wall: ~30-40 min (p01 ~3 min × 9 reps + p02 ~1 min × 9 reps).
# NOTE: The "resume-audit tolerated" branch is retained from the 3.6 runner
# but is expected to fire less often on 3.8 due to Q8_0 eh_proj (higher
# numerical precision than 3.6-MTP's Q4_K on the same tensor).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
BENCH="$REPO/target/release/qwen-bench"
MODEL="/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf"

# Queue politely if another qwen process holds the Metal process lease.
# Machine is shared with other OpenCode sessions; do not preempt.
export QWEN_METAL_LEASE_WAIT=1

mkdir -p "$HERE/baseline"

if [ ! -x "$BENCH" ]; then
  echo "missing $BENCH; run: cargo build --release -p qwen-cli --bin qwen-bench" >&2
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "missing model $MODEL" >&2
  exit 1
fi

# (label, spec_tokens, physical_n) triples.
CONFIGS=(
  "D1 1 2"
  "D3 3 4"
  "D7 7 8"
)

gen_for() {
  # Per-anchor gen budget matches the perflog anchor exactly.
  case "$1" in
    p01) echo 256 ;;   # v0.587 anchor
    p02) echo 128 ;;   # v0.556 anchor
    *) echo "unknown prompt $1" >&2; exit 1 ;;
  esac
}

run_one() {
  local prompt_name="$1"   # p01 or p02
  local cfg_label="$2"     # D1 D3 D7
  local spec="$3"
  local pn="$4"
  local rep="$5"

  local pf
  pf=$(ls "$HERE"/prompts/${prompt_name}-*.txt)
  local gen
  gen=$(gen_for "$prompt_name")
  local out="$HERE/baseline/${prompt_name}-${cfg_label}-r${rep}.json"
  local err="$HERE/baseline/${prompt_name}-${cfg_label}-r${rep}.out"

  # Resumable: skip if this cell already has a JSON.
  if [ -s "$out" ]; then
    echo "[skip] ${prompt_name} ${cfg_label} rep=${rep} (already present)"
    return 0
  fi

  echo "[run] ${prompt_name} ${cfg_label} rep=${rep} gen=${gen} spec=${spec} n=${pn}"
  # Load prompt into a variable so argv delivers the full text.
  local prompt_text
  prompt_text=$(cat "$pf")

  # Tolerate nonzero exit if it's the terminal resume audit (greedy stream
  # PASS but numerical KV state drift beyond internal tolerance). JSON is
  # still written before that error. Fail hard on any other nonzero.
  local rc=0
  "$BENCH" mtp \
    --model "$MODEL" \
    --prompt "$prompt_text" \
    --spec-tokens "$spec" \
    --mtp-physical-n "$pn" \
    --tokens "$gen" \
    --output "$out" \
    > "$err" 2>&1 || rc=$?
  if [ $rc -ne 0 ]; then
    if [ ! -f "$out" ] || ! grep -q "native MTP terminal resume audit failed" "$err"; then
      echo "  FAIL rc=$rc (see $err)" >&2
      exit $rc
    fi
    echo "  wrote $out (resume-audit tolerated, greedy PASS in .out)"
  else
    echo "  wrote $out"
  fi
}

# Sequential; do not parallelize (thermal/wall-clock contamination).
for prompt_name in p01 p02; do
  for cfg in "${CONFIGS[@]}"; do
    read -r label spec pn <<< "$cfg"
    for rep in 1 2 3; do
      run_one "$prompt_name" "$label" "$spec" "$pn" "$rep"
    done
  done
done

echo "[done] all 18 invocations complete"
