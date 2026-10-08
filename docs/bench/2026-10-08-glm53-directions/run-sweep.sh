#!/bin/zsh
# Native dose-response sweep as run (binary built at 80d9a299, rebased as
# 17ebacd4). Inputs: r.f32le is the archived direction's .npy payload
# (F32 x 4096, bytes unchanged); prompts.tsv holds the archived sweep's eight
# prompts (id<TAB>text, from its scripts/sweep.py); sweep-plan-a<dose>.json is
# plan-template.json with the dose filled in at both operations (at dose 0
# the coefficient-zero operations are filtered and none is applied). Outputs
# <id>-a<dose>.json per point.
cd /Users/tito/code/qwen-llm-directions
H=80d9a299
OUT=$PWD/target/profiles/glm53/directions/sweep
mkdir -p $OUT
GGUF=/Volumes/wdblack/weights-archive/glm-5.3-flash-ud-iq3_xxs/glm5-next/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf
D=$PWD/target/profiles/glm53/directions
export QWEN_METAL_LEASE_WAIT=1 MTL_DEBUG_LAYER=1
{
  while read -r a; do
    while IFS=$'\t' read -r id q; do
      /tmp/qwen_lens_$H run --model $GGUF --plan $D/sweep-plan-a$a.json --user "$q" \
        --message-mode low --assistant-prefill '{"channel":"final","text":""}' --max-new-tokens 8 --temperature 0 \
        --logprobs-top-k 10 --logprobs-token-ids 40,19116 --output $OUT/$id-a$a.json --format summary > /dev/null 2>> $OUT/errors.log
      echo "$id alpha=$a exit=$? $(date +%T)"
    done < $D/runs/prompts.tsv
  done < $D/sweep-alphas.txt
  echo "sweep done $(date)"
} > $OUT/sweep.log 2>&1
