#!/usr/bin/env bash
# Paired A/B runner for the hybrid GDN equivalence changes.
#
# Each arm is a fresh process because the engine latches QWEN_* flags on first
# read. GPU benchmarks are intentionally serial. Use SURFACE=gdn for the
# isolated GDN replay, or SURFACE=decode for model-level decode bulk.
#
# Examples:
#   SURFACE=gdn ROUNDS=3 ./scripts/bench/hybrid-gdn-ab.sh "$MODEL"
#   SURFACE=decode TOKENS=128 RUNS=3 ./scripts/bench/hybrid-gdn-ab.sh "$MODEL"
#   SURFACE=decode ARMS="control moe" ./scripts/bench/hybrid-gdn-ab.sh "$MODEL"
#   SURFACE=decode ARMS="control moe" BASE_BETA=1 BASE_ROPE=1 ./scripts/bench/hybrid-gdn-ab.sh "$MODEL"
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s <model.gguf>\n' "$0" >&2
    exit 2
fi

MODEL="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${QWEN_BENCH:-$ROOT/target/release/qwen-bench}"
SURFACE="${SURFACE:-gdn}"
ROUNDS="${ROUNDS:-2}"
RUNS="${RUNS:-10}"
WARMUP="${WARMUP:-3}"
TOKENS="${TOKENS:-128}"
OUT_DIR="${OUT_DIR:-$ROOT/target/profiles/hybrid-gdn-ab}"
ARM_LIST="${ARMS:-control beta rope moe all}"
read -r -a requested_arms <<<"$ARM_LIST"
BASE_BETA="${BASE_BETA:-0}"
BASE_ROPE="${BASE_ROPE:-0}"
BASE_GROUPED_MOE="${BASE_GROUPED_MOE:-0}"
SHUFFLE_SEED="${SHUFFLE_SEED:-$(date +%s)}"
if [[ ! "$SHUFFLE_SEED" =~ ^[0-9]+$ ]]; then
    printf 'error: SHUFFLE_SEED must be an integer, got %s\n' "$SHUFFLE_SEED" >&2
    exit 2
fi
SHUFFLE_STATE="$SHUFFLE_SEED"
for base_flag in "$BASE_BETA" "$BASE_ROPE" "$BASE_GROUPED_MOE"; do
    if [[ "$base_flag" != 0 && "$base_flag" != 1 ]]; then
        printf 'error: BASE_* flags must be 0 or 1, got %s\n' "$base_flag" >&2
        exit 2
    fi
done
if [[ "${#requested_arms[@]}" -eq 0 ]]; then
    printf 'error: ARMS must contain at least one arm\n' >&2
    exit 2
fi
for requested in "${requested_arms[@]}"; do
    case "$requested" in
        control|all) ;;
        beta)
            [[ "$BASE_BETA" == 0 ]] || { printf 'error: beta arm requires BASE_BETA=0\n' >&2; exit 2; }
            ;;
        rope)
            [[ "$BASE_ROPE" == 0 ]] || { printf 'error: rope arm requires BASE_ROPE=0\n' >&2; exit 2; }
            ;;
        moe)
            [[ "$BASE_GROUPED_MOE" == 0 ]] || { printf 'error: moe arm requires BASE_GROUPED_MOE=0\n' >&2; exit 2; }
            ;;
        *)
            printf 'error: unknown arm %s\n' "$requested" >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "$MODEL" ]]; then
    printf 'error: model not found: %s\n' "$MODEL" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    printf 'error: benchmark binary not found or not executable: %s\n' "$BIN" >&2
    printf 'build with: cargo build --release -p qwen-cli --bin qwen-bench\n' >&2
    exit 1
fi
case "$SURFACE" in
    gdn|decode) ;;
    *)
        printf 'error: SURFACE must be gdn or decode, got %s\n' "$SURFACE" >&2
        exit 2
        ;;
esac

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$OUT_DIR/${STAMP}-${SURFACE}.log"

printf 'model=%s\n' "$MODEL" | tee "$LOG"
printf 'binary=%s\n' "$BIN" | tee -a "$LOG"
if [[ "$SURFACE" == gdn ]]; then
    EFFECTIVE_WARMUP="$WARMUP"
else
    EFFECTIVE_WARMUP="tg-internal-one"
fi
printf 'surface=%s rounds=%s runs=%s warmup=%s tokens=%s\n' \
    "$SURFACE" "$ROUNDS" "$RUNS" "$EFFECTIVE_WARMUP" "$TOKENS" | tee -a "$LOG"
printf 'arms=%s\n' "$ARM_LIST" | tee -a "$LOG"
printf 'base_flags: beta=%s rope_pair=%s grouped_moe_finalizer=%s\n' \
    "$BASE_BETA" "$BASE_ROPE" "$BASE_GROUPED_MOE" | tee -a "$LOG"
printf 'shuffle_seed=%s order=balanced-random-rotations\n' "$SHUFFLE_SEED" | tee -a "$LOG"
printf 'inherited_qwen_env:\n' | tee -a "$LOG"
while IFS='=' read -r name value; do
    case "$name" in
        QWEN_*)
            case "$name" in
                *_KEY|*_TOKEN|*_SECRET|*_PASSWORD) value='<redacted>' ;;
            esac
            printf '%s=%s\n' "$name" "$value" | tee -a "$LOG"
            ;;
    esac
done < <(env | sort)
printf '\n' | tee -a "$LOG"
printf 'start=%s\n\n' "$(date -u +%FT%TZ)" | tee -a "$LOG"

run_arm() {
    local arm="$1"
    local beta="$BASE_BETA"
    local rope="$BASE_ROPE"
    local grouped_moe="$BASE_GROUPED_MOE"
    case "$arm" in
        control) ;;
        beta) beta=1 ;;
        rope) rope=1 ;;
        moe) grouped_moe=1 ;;
        all) beta=1; rope=1; grouped_moe=1 ;;
        *)
            printf 'error: unknown arm %s\n' "$arm" >&2
            exit 2
            ;;
    esac

    printf '===== round=%s arm=%s =====\n' "$round" "$arm" | tee -a "$LOG"
    printf 'flags: beta=%s rope_pair=%s grouped_moe_finalizer=%s shared_moe_fusion=1\n' \
        "$beta" "$rope" "$grouped_moe" | tee -a "$LOG"

    local -a args
    if [[ "$SURFACE" == gdn ]]; then
        args=(
            --allow-dirty
            decode-gdn-chain-replay
            -m "$MODEL"
            --tokens "${GDN_TOKENS:-4,16}"
            --layers "${GDN_LAYERS:-4}"
            --iters "$RUNS"
            --warmup "$WARMUP"
        )
    else
        args=(
            --allow-dirty
            tg
            -m "$MODEL"
            --n-gen "$TOKENS"
            --runs "$RUNS"
        )
    fi

    env \
        QWEN_DECODE_GDN_FUSED_BETA_PROJ="$beta" \
        QWEN_DECODE_ROPE_PAIR="$rope" \
        QWEN_DECODE_MOE_FUSED_FINALIZER=1 \
        QWEN_DECODE_MOE_GROUPED_FINALIZER="$grouped_moe" \
        "$BIN" "${args[@]}" 2>&1 | tee -a "$LOG"
    printf '\n' | tee -a "$LOG"
}

shuffle_requested_arms() {
    shuffle_result=("${requested_arms[@]}")
    local i j tmp
    for ((i=${#shuffle_result[@]} - 1; i > 0; i--)); do
        SHUFFLE_STATE=$(( (SHUFFLE_STATE * 1103515245 + 12345) & 2147483647 ))
        j=$((SHUFFLE_STATE % (i + 1)))
        tmp="${shuffle_result[i]}"
        shuffle_result[i]="${shuffle_result[j]}"
        shuffle_result[j]="$tmp"
    done
}

arm_count="${#requested_arms[@]}"
for round in $(seq 1 "$ROUNDS"); do
    offset=$(( (round - 1) % arm_count ))
    if [[ "$offset" -eq 0 ]]; then
        shuffle_requested_arms
        cycle_arms=("${shuffle_result[@]}")
    fi
    ordered_arms=()
    for ((i=0; i<arm_count; i++)); do
        ordered_arms+=("${cycle_arms[(offset + i) % arm_count]}")
    done
    printf 'round=%s order=%s\n' "$round" "$(IFS=,; printf '%s' "${ordered_arms[*]}")" | tee -a "$LOG"
    for arm in "${ordered_arms[@]}"; do
        run_arm "$arm"
    done
done

printf 'end=%s\n' "$(date -u +%FT%TZ)" | tee -a "$LOG"
printf 'results=%s\n' "$LOG"
