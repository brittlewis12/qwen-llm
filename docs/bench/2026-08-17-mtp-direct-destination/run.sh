#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
MODEL="/Users/tito/models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
BIN="$ROOT/target/release/qwen-bench"
RAW="/tmp/qwen-mtp-direct-destination-20260817/measure"

[[ -x "$BIN" ]] || { printf 'missing benchmark binary: %s\n' "$BIN" >&2; exit 1; }
[[ -f "$MODEL" ]] || { printf 'missing fixture: %s\n' "$MODEL" >&2; exit 1; }
mkdir -p "$RAW"

run_arm() {
    local pair="$1"
    local order="$2"
    local arm="$3"
    local flag="$4"
    local stem="$RAW/pair${pair}-${order}-${arm}"

    if ps -p 8770 >/dev/null 2>&1; then
        printf 'protected PID 8770 is active; refusing benchmark\n' >&2
        exit 1
    fi
    for path in "$stem.json" "$stem.stderr" "$stem.stdout" "$stem.time"; do
        [[ ! -e "$path" ]] || { printf 'refusing to overwrite %s\n' "$path" >&2; exit 1; }
    done

    printf 'pair=%s order=%s arm=%s direct=%s\n' "$pair" "$order" "$arm" "$flag"
    /usr/bin/time -l -o "$stem.time" \
        env QWEN_MTP_DIRECT_F32_DEST="$flag" QWEN_MTP_MOE_NATIVE_BANKS=0 \
        "$BIN" mtp \
        --allow-dirty \
        --model "$MODEL" \
        --spec-tokens 1 \
        --tokens 16 --no-warmup \
        --include-token-ids --output "$stem.json" \
        >"$stem.stdout" 2>"$stem.stderr"
}

for pair in 01 02 03 04 05 06; do
    if ((10#$pair % 2 == 1)); then
        run_arm "$pair" ab control 0
        run_arm "$pair" ab direct 1
    else
        run_arm "$pair" ba direct 1
        run_arm "$pair" ba control 0
    fi
done
