#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
MODEL="/Users/tito/models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
PROMPT="The quick brown fox jumps over the lazy dog"
BIN="$ROOT/target/release/qwen-bench"
RAW="/tmp/qwen-mtp-direct-destination-20260817/campaign-v4"
MANIFEST="$RAW/chronology.tsv"

[[ -x "$BIN" ]] || { printf 'missing benchmark binary: %s\n' "$BIN" >&2; exit 1; }
[[ -f "$MODEL" ]] || { printf 'missing fixture: %s\n' "$MODEL" >&2; exit 1; }
[[ ! -e "$RAW" ]] || { printf 'refusing to reuse campaign root: %s\n' "$RAW" >&2; exit 1; }
mkdir -p "$RAW"
printf 'event_sequence\tevent\tpair\torder\tarm\tdirect\tstarted_utc\tended_utc\texit_status\tartifact\n' >"$MANIFEST"

event_sequence=0

protected_process_absent() {
    if ps -p 8770 >/dev/null 2>&1; then
        printf 'protected PID 8770 is active; refusing benchmark\n' >&2
        exit 1
    fi
}

append_manifest() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >>"$MANIFEST"
}

run_build_info() {
    local event="$1"
    local artifact="$2"
    local started_utc
    local ended_utc
    local status=0

    event_sequence=$((event_sequence + 1))
    started_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    "$BIN" build-info --allow-dirty -o json >"$artifact" || status=$?
    ended_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    append_manifest "$event_sequence" "$event" "" "" "" "" \
        "$started_utc" "$ended_utc" "$status" "$(basename "$artifact")"
    ((status == 0)) || return "$status"
}

run_packed_control() {
    local stem="$RAW/packed-control"
    local started_utc
    local ended_utc
    local status=0

    protected_process_absent
    event_sequence=$((event_sequence + 1))
    printf 'event=%s packed staged control\n' "$event_sequence"
    started_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    /usr/bin/time -l -o "$stem.time" \
        env QWEN_MTP_DIRECT_F32_DEST=0 QWEN_MTP_MOE_NATIVE_BANKS=0 \
        "$BIN" mtp \
        --allow-dirty \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --spec-tokens 1 --mtp-physical-n 2 \
        --tokens 16 --no-warmup \
        --include-token-ids --output "$stem.json" \
        >"$stem.stdout" 2>"$stem.stderr" || status=$?
    ended_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    append_manifest "$event_sequence" "packed-control" "" "" "control" "0" \
        "$started_utc" "$ended_utc" "$status" "packed-control.json"
    ((status == 1)) || {
        printf 'packed control returned %s instead of expected audit failure 1\n' "$status" >&2
        return 1
    }
}

run_byte_oracle() {
    local stem="$RAW/byte-oracle"
    local started_utc
    local ended_utc
    local status=0

    protected_process_absent
    event_sequence=$((event_sequence + 1))
    printf 'event=%s exact byte oracle\n' "$event_sequence"
    started_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    /usr/bin/time -l -o "$stem.time" \
        env \
        QWEN_MTP_DIRECT_F32_ORACLE_OUT="$stem.json" \
        QWEN_MTP_DIRECT_F32_ORACLE_BUILD_IDENTITY="$RAW/build-info-before.json" \
        cargo test -p qwen-llm \
        'metal_mtp::tests::mtp_direct_f32_banks_match_staged_bytes' \
        --lib -- --ignored --exact --nocapture --test-threads=1 \
        >"$stem.stdout" 2>"$stem.stderr" || status=$?
    ended_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    append_manifest "$event_sequence" "byte-oracle" "" "" "" "" \
        "$started_utc" "$ended_utc" "$status" "byte-oracle.json"
    ((status == 0)) || return "$status"
}

run_arm() {
    local pair="$1"
    local order="$2"
    local arm="$3"
    local flag="$4"
    local stem="$RAW/pair${pair}-${order}-${arm}"
    local started_utc
    local ended_utc
    local status=0

    protected_process_absent
    event_sequence=$((event_sequence + 1))
    printf 'event=%s pair=%s order=%s arm=%s direct=%s\n' \
        "$event_sequence" "$pair" "$order" "$arm" "$flag"
    started_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    /usr/bin/time -l -o "$stem.time" \
        env QWEN_MTP_DIRECT_F32_DEST="$flag" QWEN_MTP_MOE_NATIVE_BANKS=0 \
        "$BIN" mtp \
        --allow-dirty \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --spec-tokens 1 \
        --tokens 16 --no-warmup \
        --include-token-ids --output "$stem.json" \
        >"$stem.stdout" 2>"$stem.stderr" || status=$?
    ended_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    append_manifest "$event_sequence" "arm" "$pair" "$order" "$arm" "$flag" \
        "$started_utc" "$ended_utc" "$status" "$(basename "$stem").json"
    ((status == 0)) || return "$status"
}

run_build_info "build-info-before" "$RAW/build-info-before.json"
run_packed_control
run_byte_oracle
run_build_info "build-info-after-oracle" "$RAW/build-info-after-oracle.json"

for pair in 01 02 03 04 05 06; do
    if ((10#$pair % 2 == 1)); then
        run_arm "$pair" ab control 0
        run_arm "$pair" ab direct 1
    else
        run_arm "$pair" ba direct 1
        run_arm "$pair" ba control 0
    fi
done

run_build_info "build-info-after-campaign" "$RAW/build-info-after-campaign.json"
