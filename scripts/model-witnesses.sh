#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
filter=${1:-}

# Each selected function is run by name so unrelated ignored profile and
# benchmark tests remain opt-in under their own workflows.
python3 - "$filter" <<'PY'
import pathlib
import re
import subprocess
import sys

filter_text = sys.argv[1]
roots = [pathlib.Path("crates/qwen-llm/src"), pathlib.Path("crates/qwen-llm/tests"), pathlib.Path("crates/qwen-cli/src"), pathlib.Path("crates/qwen-cli/tests")]
by_package = {"qwen-llm": set(), "qwen-cli": set()}
for root in roots:
    package = "qwen-cli" if "qwen-cli" in root.parts else "qwen-llm"
    for path in root.rglob("*.rs"):
        if "qwen4exp_" in path.name:
            continue
        source = path.read_text()
        for match in re.finditer(r'#\[ignore\s*=\s*"model witness: [^"]+"\]\s*fn\s+(\w+)', source):
            name = match.group(1)
            if not filter_text or filter_text in name:
                by_package[package].add(name)

for package, names in by_package.items():
    for name in sorted(names):
        subprocess.run(["cargo", "test", "--release", "-p", package, name, "--", "--ignored", "--test-threads=1"], check=True)
PY
