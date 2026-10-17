#!/bin/sh
set -eu

family=${1:?usage: run-model.sh qwen35b|glm MODEL}
model=${2:?model path required}
root="$HOME/qwen-runs/2026-10-16-first-light"
out="$root/$family"
expected=d7b2a05b5ad1b9706486c69718a1e11dbbc58937

case "$family" in
    qwen35b) expected_model=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf ;;
    glm) expected_model=GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf ;;
    *) echo "Unsupported first-light family: $family" >&2; exit 2 ;;
esac
if [ "$(basename "$model")" != "$expected_model" ]; then
    echo "Unexpected model entrypoint for $family: $model" >&2
    exit 2
fi

cd "$HOME/code/qwen-llm"
test "$(git rev-parse HEAD)" = "$expected"
git diff --exit-code
git diff --cached --exit-code
test -f "$model"
mkdir "$out"
finish() {
    status=$?
    trap - EXIT
    {
        date -u
        sysctl vm.swapusage
        vm_stat
    } > "$out/after.txt" 2>&1 || :
    printf "%s\n" "$status" > "$out/exit-code.txt"
    exit "$status"
}
trap finish EXIT

(cd "$root" && shasum -a 256 -c binaries.sha256) > "$out/binary-integrity.txt" 2>&1
(cd "$(dirname "$model")" && shasum -a 256 -c "$root/$family.sha256") \
    > "$out/model-integrity.txt" 2>&1

bounded() {
    env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=0 \
        uv run --offline --no-project "$root/deadline.py" --seconds 900 "$@"
}

{
    date -u
    sw_vers
    sysctl machdep.cpu.brand_string hw.memsize iogpu.wired_limit_mb vm.swapusage
    vm_stat
    git rev-parse HEAD
    shasum -a 256 "$root/bin/qwen" "$root/bin/qwen-bench"
} > "$out/before.txt"

bounded "$root/bin/qwen" info -m "$model" --json > "$out/info.json"
prompt='Write a Rust function sum_even(xs: &[i64]) -> i64 that sums the even elements. Include one assertion using [1, 2, 3, 4]. Keep the answer brief.'
if [ "$family" = glm ]; then
    set -- --reasoning-effort low -n 1024
else
    set -- --no-thinking -n 256
fi
bounded "$root/bin/qwen" run \
    -m "$model" --temp 0 "$@" --user "$prompt" \
    > "$out/smoke.txt" 2> "$out/smoke.stderr"
cat "$out/smoke.txt" "$out/smoke.stderr"
if ! rg -q 'stop_reason=eos( |$)' "$out/smoke.stderr"; then
    echo "Smoke did not finish naturally; inspect output before benchmarking." >&2
    exit 1
fi

bounded "$root/bin/qwen-bench" suite \
    -m "$model" --pp 128,512,2048 --tg 128,256 -d 0 --runs 3 \
    > "$out/suite.json" 2> "$out/suite.stderr"
bounded "$root/bin/qwen-bench" suite \
    -m "$model" --tg 128 -d 8192 --runs 3 \
    > "$out/depth8192.json" 2> "$out/depth8192.stderr"

jq '[.[] | {test,n_depth,n_repetitions,avg_ts,stddev_ts,build_status:.build_identity.status}]' \
    "$out/suite.json" "$out/depth8192.json"
printf '\nFIRST_LIGHT_DONE=%s\n' "$family"
