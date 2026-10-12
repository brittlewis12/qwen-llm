#!/usr/bin/env bash
# Structural contracts that need no GPU or model. `--full` also compiles every
# workspace test, example, benchmark and integration target in release mode.
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
  if output=$(git grep -n -E -e "$pattern" "$tree" -- . ':!docs/**' 2>&1); then
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
check_tree_has_no_match "HOME model path found outside docs" "${home_variable}/${model_name}"
check_tree_has_no_match "home checkout path found outside docs" "${tilde}/${code_name}/"
check_tree_has_no_match "home-directory model path expression found outside docs" "$home_models_pattern"
check_tree_has_no_match "home-directory checkout path expression found outside docs" "$home_code_pattern"
check_tree_has_no_match "relative model-directory literal found in Rust or scripts" "$relative_model_pattern"

cargo fmt --all --check
uv run scripts/reference/env_knobs.py --check
if [ "$full" -eq 1 ]; then
  cargo test --release --workspace --no-run
fi

if [ "$(git rev-parse HEAD)" != "$commit" ] || [ "$(git rev-parse 'HEAD^{tree}')" != "$tree" ] || ! worktree_clean; then
  echo "check-contracts: checked tree changed during verification" >&2
  exit 1
fi
echo "check-contracts: ok for tree $tree (commit ${commit:0:8})"
