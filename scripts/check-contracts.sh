#!/usr/bin/env bash
# Structural contracts that need no GPU or model, checked on the committed
# tree: no machine-specific paths in code, scripts or maintained docs,
# formatting, the generated knob reference, the model-witness and path
# normalizer self-tests, and the numeric comparison tripwire. `--full` also
# compiles every workspace target in release mode and runs the README family
# table test.
set -euo pipefail

full=0
case "${1:-}" in
  "") ;;
  --full) full=1 ;;
  *) echo "usage: $0 [--full]" >&2; exit 2 ;;
esac

cd "$(git rev-parse --show-toplevel)"

worktree_clean() {
  git diff --quiet HEAD -- && [ -z "$(git ls-files --others --exclude-standard)" ]
}

if ! worktree_clean; then
  echo "check-contracts: worktree has uncommitted changes" >&2
  exit 1
fi

commit=$(git rev-parse HEAD)
tree=$(git rev-parse 'HEAD^{tree}')

check_tree_has_no_match() {
  local description="$1"
  local pattern="$2"
  local output
  local result
  # The normalizer names the paths it replaces; docs are checked by it.
  if output=$(git grep -n -E -e "$pattern" "$tree" -- . ':!docs/**' \
    ':!scripts/reference/normalize_paths.py' 2>&1); then
    printf '%s\n' "$output" >&2
    echo "check-contracts: $description" >&2
    exit 1
  else
    result=$?
    if [ "$result" -ne 1 ]; then
      printf '%s\n' "$output" >&2
      echo "check-contracts: git grep failed while checking $description" >&2
      exit "$result"
    fi
  fi
}

check_sample_matches() {
  local pattern="$1"
  local sample="$2"
  if ! grep -E -q -- "$pattern" <<<"$sample"; then
    echo "check-contracts: guard fixture did not match: $sample" >&2
    exit 1
  fi
}

relative_model_pattern="['\"](\\./)?models/"
home_models_pattern="Path[.]home\\(\\)[[:space:]]*/[[:space:]]*[\"']models[\"']"
home_code_pattern="Path[.]home\\(\\)[[:space:]]*/[[:space:]]*[\"']code[\"']"

# Exercise spellings that previously escaped the tracked-tree guard.
home_expr='Path.home() / '
model_name='models'
code_name='code'
tilde='~'
single_quote="'"
double_quote='"'
dot_slash='./'
check_sample_matches "$relative_model_pattern" "${single_quote}${model_name}/sample.gguf${single_quote}"
check_sample_matches "$relative_model_pattern" "${double_quote}${dot_slash}${model_name}/sample.gguf${double_quote}"
check_sample_matches "$home_models_pattern" "${home_expr}${double_quote}${model_name}${double_quote} / weights.gguf"
check_sample_matches "$home_models_pattern" "${home_expr}${single_quote}${model_name}${single_quote} / weights.gguf"
check_sample_matches "$home_code_pattern" "${home_expr}${double_quote}${code_name}${double_quote} / llama.cpp"
check_sample_matches "${tilde}/${code_name}/" "${tilde}/${code_name}/llama.cpp"

slash='/'
for segment in Users Volumes home; do
  fragment="${slash}${segment}${slash}"
  check_tree_has_no_match "machine-specific path found outside docs: $fragment" "$fragment"
done
check_tree_has_no_match "home model path found outside docs" "${tilde}/${model_name}"
home_variable='$HOME'
home_model_sample="${home_variable}/${model_name}/sample.gguf"
dollar='\$'
home_name='HOME'
home_models_path_pattern="${dollar}${home_name}/${model_name}"
check_sample_matches "$home_models_path_pattern" "$home_model_sample"
check_tree_has_no_match "HOME model path found outside docs" "$home_models_path_pattern"
check_tree_has_no_match "home checkout path found outside docs" "${tilde}/${code_name}/"
check_tree_has_no_match "home-directory model path expression found outside docs" "$home_models_pattern"
check_tree_has_no_match "home-directory checkout path expression found outside docs" "$home_code_pattern"
check_tree_has_no_match "relative model-directory literal found in Rust or scripts" "$relative_model_pattern"

cargo fmt --all --check
uv run scripts/reference/env_knobs.py --check
uv run scripts/reference/identity_gates.py --self-test
uv run scripts/reference/identity_gates.py --check
uv run scripts/model-witnesses.py --self-test
uv run scripts/qualification-witnesses.py --self-test
uv run scripts/reference/normalize_paths.py --self-test
uv run scripts/reference/normalize_paths.py --check
uv run --no-project python scripts/reference/deepseek_v41/check.py
if [ "$full" -eq 1 ]; then
  cargo test --release --workspace --no-run
  release_qwen=$(
    cargo build --release -p qwen-cli --bin qwen --message-format=json |
      python3 -c '
import json
import sys

executables = []
for line in sys.stdin:
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        continue
    target = message.get("target", {})
    executable = message.get("executable")
    if (
        message.get("reason") == "compiler-artifact"
        and target.get("name") == "qwen"
        and "bin" in target.get("kind", [])
        and executable
    ):
        executables.append(executable)

if not executables:
    sys.exit("check-contracts: cargo build reported no qwen executable")
if len(set(executables)) != 1:
    sys.exit("check-contracts: cargo build reported multiple qwen executables")
print(executables[0])
'
  )
  if [ ! -x "$release_qwen" ]; then
    echo "check-contracts: cargo-reported qwen executable is missing or not executable: $release_qwen" >&2
    exit 1
  fi
  metal_undefined=$(nm -m -u "$release_qwen" | awk '/\(from Metal\)$/')
  if [ -z "$metal_undefined" ]; then
    echo "check-contracts: release qwen has no undefined Metal symbols to inspect" >&2
    exit 1
  fi
  strong_metal_undefined=$(awk '/\(from Metal\)$/ && $0 !~ /\(undefined\) weak external /' <<<"$metal_undefined")
  if [ -n "$strong_metal_undefined" ]; then
    printf '%s\n' "$strong_metal_undefined" >&2
    echo "check-contracts: release qwen has strong undefined Metal symbols" >&2
    exit 1
  fi
  if ! grep -Fq '_OBJC_CLASS_$_MTLResidencySetDescriptor' <<<"$metal_undefined"; then
    echo "check-contracts: release qwen does not contain the residency-set class reference" >&2
    exit 1
  fi
  if ! grep -F '_OBJC_CLASS_$_MTLResidencySetDescriptor' <<<"$metal_undefined" | grep -Fq 'weak external'; then
    echo "check-contracts: residency-set class reference is not weak" >&2
    exit 1
  fi
fi

cargo test --release -p qwen-llm --test comparison_tripwire
status=0
output=$(cargo test --release -p qwen-llm --lib deepseek_v41:: 2>&1) || status=$?
printf '%s\n' "$output"
[ "$status" -eq 0 ] || exit "$status"
if ! grep -Eq '^test result: ok\. [1-9][0-9]* passed; 0 failed; 0 ignored' <<<"$output"; then
  echo "check-contracts: V4.1 model-free tests did not execute without skips" >&2
  exit 1
fi
if [ "$full" -eq 1 ]; then
  readme_test=family_profile::tests::readme_family_table_matches_profiles
  status=0
  output=$(cargo test --release -p qwen-cli --bin qwen -- --exact "$readme_test" 2>&1) ||
    status=$?
  printf '%s\n' "$output"
  [ "$status" -eq 0 ] || exit "$status"
  pass_count=$(grep -Ec '^test result: ok\. 1 passed; 0 failed([;[:space:]]|$)' <<<"$output" || true)
  if [ "$pass_count" -ne 1 ]; then
    echo "check-contracts: $readme_test did not run exactly once" >&2
    exit 1
  fi
fi

if [ "$(git rev-parse HEAD)" != "$commit" ] || [ "$(git rev-parse 'HEAD^{tree}')" != "$tree" ] || ! worktree_clean; then
  echo "check-contracts: checked tree changed during verification" >&2
  exit 1
fi
echo "check-contracts: ok for tree $tree (commit ${commit:0:8})"
