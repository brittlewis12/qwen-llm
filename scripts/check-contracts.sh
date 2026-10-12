#!/usr/bin/env bash
# Structural contracts that need no GPU or model: formatting, the generated
# environment-knob catalog and the numeric comparison tripwire. `--full` also
# checks the README family table against FamilyProfile, which builds the qwen
# binary's tests.
#
# Checks the committed tree, so a clean worktree is required; on success the
# checked tree id is printed.
set -euo pipefail

full=0
case "${1:-}" in
  "") ;;
  --full) full=1 ;;
  *) echo "usage: $0 [--full]" >&2; exit 2 ;;
esac

cd "$(git rev-parse --show-toplevel)"
if ! git diff --quiet HEAD -- || [ -n "$(git ls-files --others --exclude-standard)" ]; then
  echo "check-contracts: worktree has uncommitted changes; commit or stash them first" >&2
  exit 1
fi

cargo fmt --all --check
uv run scripts/reference/env_knobs.py --check
cargo test -q --release -p qwen-llm --test comparison_tripwire
if [ "$full" -eq 1 ]; then
  cargo test -q --release -p qwen-cli --bin qwen -- --exact \
    family_profile::tests::readme_family_table_matches_profiles
fi

echo "check-contracts: ok for tree $(git rev-parse 'HEAD^{tree}') (commit $(git rev-parse --short HEAD))"
