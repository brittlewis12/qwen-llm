#!/bin/sh
set -eu

: "${QWEN_BINARY:?pinned native binary required}"
: "${QWEN_SOURCE_ROOT:?pinned source checkout required}"
: "${MAYA_MODEL:?first Maya shard required}"
: "${Q4_DOWNLOAD:?completed Q4 download directory required}"
: "${IQ3_DOWNLOAD:?IQ3 download directory required}"

root=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
expected=d7b2a05b5ad1b9706486c69718a1e11dbbc58937
revision=621d456e93e926e4b52f85cff5f634358c1828f9

check_source() {
    test "$(git -C "$QWEN_SOURCE_ROOT" rev-parse HEAD)" = "$expected"
    test -z "$(git -C "$QWEN_SOURCE_ROOT" status --porcelain)"
}

run_arm() {
    check_source
    env -u QWEN_BINARY -u QWEN_SOURCE_ROOT \
        uv run --offline --no-project python -B "$root/run.py" \
        --binary "$QWEN_BINARY" --model "$2" --manifest "$root/$3.json" \
        --cases "$root/cases.json" --arm "$1" --out "$root/$1"
    printf 'QUALITY_ARM_DONE=%s\n' "$1"
}

q4=$(uv run --offline --no-project python -B "$root/stage.py" \
    --download "$Q4_DOWNLOAD" --manifest "$root/stock-q4.json")
run_arm arm-01 "$MAYA_MODEL" maya
run_arm arm-02 "$q4" stock-q4

env -u HF_HUB_DISABLE_XET HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HUB_DISABLE_UPDATE_CHECK=1 \
    uvx --from huggingface_hub==2.2.0 --with hf_xet==1.7.0 hf download \
    unsloth/GLM-5.3-Flash-GGUF --revision "$revision" \
    --include 'UD-IQ3_XXS/*' \
    --include 'Shard_Rewrite/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf_file' \
    --local-dir "$IQ3_DOWNLOAD" --max-workers 4 --format human
iq3=$(uv run --offline --no-project python -B "$root/stage.py" \
    --download "$IQ3_DOWNLOAD" --manifest "$root/stock-iq3.json")
run_arm arm-03 "$iq3" stock-iq3
printf 'QUALITY_QUEUE_DONE\n'
