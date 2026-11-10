#!/bin/bash
# spec-econ GLM campaign: cost curve at 8K/48K and DFlash2 acceptance at 48K.
# Runs inside scripts/bench/with-metal-lease (QWEN_METAL_LEASE_FD inherited).
set -uo pipefail
WT=$(git rev-parse --show-toplevel)
BIN="$WT/target/release/examples"
MODEL=${MODEL:?set MODEL to the GLM UD-IQ3_XXS GGUF}
DRAFTER=${DRAFTER:?set DRAFTER to the GLM DFlash2 Q8_0 GGUF}
OUT=${OUT:-$HOME/qwen-runs/2026-11-09-spec-econ}
mkdir -p "$OUT"
echo "campaign start $(date -u +%FT%TZ) thermal=$(notifyutil -g com.apple.system.thermalpressurelevel | awk '{print $NF}')"

run() {
  local name=$1; shift
  echo "== $name =="
  "$@" 2>&1 | tee "$OUT/$name.jsonl"
  echo "exit=$? thermal=$(notifyutil -g com.apple.system.thermalpressurelevel | awk '{print $NF}')"
}

run cost-48k  "$BIN/spec_verify_cost" "$MODEL" 49152 4 3
run dflash-t1 "$BIN/spec_dflash_accept" "$MODEL" "$DRAFTER" 49152 96 1 20261109 8
run dflash-t0 "$BIN/spec_dflash_accept" "$MODEL" "$DRAFTER" 49152 96 0 20261109 8
run dflash-t1b "$BIN/spec_dflash_accept" "$MODEL" "$DRAFTER" 49152 96 1 20261110 9
echo "campaign end $(date -u +%FT%TZ)"
