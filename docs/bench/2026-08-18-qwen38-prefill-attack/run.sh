#!/bin/bash
# Prefill-attack baseline: qwen-llm vs llama.cpp on Qwen 3.8-27B-Q4_K_M at fine
# ctx granularity. First step in closing the ~3.7× M4-adjusted prefill gap to
# MLX.fast leaderboard record (~917 t/s at 27B on M4-equivalent).
#
# Wide-sweep did pp{512,4096,16384}. This adds pp{1024,2048,8192,32768} to
# resolve the curve shape (linear vs quadratic scaling), which discriminates
# attention-body vs FFN/GDN as the dominant cost.
#
# Cost estimate: pp32768 ≈ 32768/220 ≈ 149s per rep × 5 reps × 2 engines ≈ 25 min
# just for pp32768. Total ~30-40 min.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
QWEN_BENCH="$REPO/target/release/qwen-bench"
LCPP_BENCH="/Users/tito/code/llama.cpp/build/bin/llama-bench"
MODEL="/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf"

export QWEN_METAL_LEASE_WAIT=1

if [ ! -x "$QWEN_BENCH" ] || [ ! -x "$LCPP_BENCH" ] || [ ! -f "$MODEL" ]; then
  echo "missing binaries or model" >&2; exit 1
fi

# Fine-grained ctx points. pp32768 is expensive; include it to catch quadratic
# growth if any.
PPS="512 1024 2048 4096 8192 16384 32768"

echo "[prefill-attack] qwen-bench pp sweep"
for n in $PPS; do
  out="$HERE/qwen-pp${n}.json"
  if [ -s "$out" ]; then
    echo "  [skip] pp$n"
    continue
  fi
  echo "  pp$n x 5 reps"
  "$QWEN_BENCH" pp --allow-dirty --model "$MODEL" --n-prompt "$n" --runs 5 --output json \
    > "$out" 2> "$HERE/qwen-pp${n}.out"
done

echo "[prefill-attack] llama-bench pp sweep"
# lcpp accepts -p as comma-list. Do -r 5 for reps. Emit json.
LCPP_OUT="$HERE/lcpp-pp.json"
if [ ! -s "$LCPP_OUT" ]; then
  # Format string joining ",": 512,1024,...
  PP_LIST=$(echo $PPS | tr ' ' ',')
  echo "  -p $PP_LIST -r 5"
  "$LCPP_BENCH" -m "$MODEL" -p "$PP_LIST" -n 0 -r 5 -o json > "$LCPP_OUT" 2> "$HERE/lcpp-pp.out"
fi

echo "[done]"
