#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run the ignored tests that need local model files or reference outputs.

These tests are ignored by default so `cargo test` means the same thing on
every machine. Fixtures resolve through QWEN_MODEL_SEARCH_PATH (and the
per-fixture variables); a missing fixture fails the selected test. Each
witness runs under its exact test path, so other ignored tests (profiles,
benchmarks) never run as a side effect, and the run fails unless every
selected witness executed and passed.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

WITNESS_REASON = re.compile(
    r'#\[ignore\s*=\s*"(?:model witness: [^"]+|needs reference logits from the external llm tool)"\]'
    r"\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?fn\s+(\w+)"
)
PACKAGES = {
    "qwen-llm": ("crates/qwen-llm/src", "crates/qwen-llm/tests"),
    "qwen-cli": ("crates/qwen-cli/src", "crates/qwen-cli/tests"),
}


def witness_names(roots: tuple[str, ...], name_filter: str) -> set[str]:
    names = set()
    for root in roots:
        for path in pathlib.Path(root).rglob("*.rs"):
            for match in WITNESS_REASON.finditer(path.read_text()):
                if name_filter in match.group(1):
                    names.add(match.group(1))
    return names


def ignored_test_paths(package: str) -> list[str]:
    listing = subprocess.run(
        [
            "cargo",
            "test",
            "--release",
            "-p",
            package,
            "--",
            "--ignored",
            "--list",
            "--format",
            "terse",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [
        line.removesuffix(": test")
        for line in listing.splitlines()
        if line.endswith(": test")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "filter",
        nargs="?",
        default="",
        help="run witnesses whose function name contains this text",
    )
    args = parser.parse_args()
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    repo = pathlib.Path(root.stdout.strip())

    selected_total = 0
    for package, roots in PACKAGES.items():
        names = witness_names(tuple(str(repo / r) for r in roots), args.filter)
        if not names:
            continue
        paths = sorted(
            p for p in ignored_test_paths(package) if p.rsplit("::", 1)[-1] in names
        )
        found = {p.rsplit("::", 1)[-1] for p in paths}
        if missing := sorted(names - found):
            print(
                f"{package}: witnesses not found in the ignored test list: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 1
        selected_total += len(paths)
        print(f"{package}: running {len(paths)} witnesses", flush=True)
        result = subprocess.run(
            [
                "cargo",
                "test",
                "--release",
                "-p",
                package,
                "--",
                "--ignored",
                "--exact",
                "--test-threads=1",
                *paths,
            ],
            capture_output=True,
            text=True,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            return result.returncode
        passed = sum(
            int(n) for n in re.findall(r"test result: ok\. (\d+) passed", result.stdout)
        )
        if passed != len(paths):
            print(
                f"{package}: {passed} of {len(paths)} selected witnesses ran",
                file=sys.stderr,
            )
            return 1
    if selected_total == 0:
        print("no witnesses matched", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
