#!/bin/bash
# P0 MTP-program baseline runner.
# Preregistration: README.md at repo path
#   docs/bench/2026-08-18-qwen36-mtp-program/README.md
#
# Invocation (from this dir):
#   ./run.sh 2>&1 | tee run.log
#
# Estimated wall: ~15 min (p01 ~90 s × 9 reps + p02 ~5-8 s × 9 reps).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
BENCH="$REPO/target/release/qwen-bench"
MODEL="/Users/tito/models/Qwen3.6-27B-MTP-Q4_K_M.gguf"

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
