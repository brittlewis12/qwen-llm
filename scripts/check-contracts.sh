#!/usr/bin/env bash
# Structural contracts that need no GPU or model: formatting, the generated
# environment-knob catalog and the numeric comparison tripwire. `--full` also
# checks the README family table against FamilyProfile, which builds the qwen
# binary's tests.
#
# Checks the committed tree: the worktree must be clean and must still be
# clean, at the same commit, when the checks finish. On success the checked
# tree id is printed. Edits made and reverted while the checks run are not
# detected.
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
  echo "check-contracts: worktree has uncommitted changes; commit or stash them first" >&2
  exit 1
fi

machine_path_patterns=(' /''Users/' ' /''Volumes/' ' /''home/' '~''/models' '$HOME''/models')
for pattern in "${machine_path_patterns[@]}"; do
  pattern=${pattern# }
  if git grep -n -F "$pattern" HEAD -- . ':!docs/**'; then
    echo "check-contracts: machine-specific path found outside docs" >&2
    exit 1
  fi
done

commit=$(git rev-parse HEAD)
tree=$(git rev-parse 'HEAD^{tree}')

cargo fmt --all --check
uv run scripts/reference/env_knobs.py --check
cargo test -q --release -p qwen-llm --test comparison_tripwire
if [ "$full" -eq 1 ]; then
  readme_test=family_profile::tests::readme_family_table_matches_profiles
  status=0
  output=$(cargo test -q --release -p qwen-cli --bin qwen -- --exact "$readme_test" 2>&1) ||
    status=$?
  printf '%s\n' "$output"
  [ "$status" -eq 0 ] || exit "$status"
  if ! grep -q 'test result: ok\. 1 passed' <<<"$output"; then
    echo "check-contracts: $readme_test did not run exactly once" >&2
    exit 1
  fi
fi

if [ "$(git rev-parse HEAD)" != "$commit" ] || ! worktree_clean; then
  echo "check-contracts: the worktree changed while the checks ran; nothing was certified" >&2
  exit 1
fi
echo "check-contracts: ok for tree $tree (commit ${commit:0:8})"
