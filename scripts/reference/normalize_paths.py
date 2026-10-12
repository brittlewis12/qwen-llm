#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Replace machine-specific absolute paths in maintained docs with placeholders.

Models become $MODELS/, this repository (any worktree) becomes $REPO/, other
paths under a home directory become $HOME/, and other volumes $VOLUME/.
Benchmark evidence under docs/bench/ is left unchanged: packets there are
sealed by recorded digests, which a rewrite would invalidate.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
)
RULES = [
    (re.compile(r"/Users/[A-Za-z0-9._-]+/models/"), "$MODELS/"),
    (re.compile(r"/Volumes/[A-Za-z0-9._-]+/weights-archive/"), "$MODELS/"),
    (
        re.compile(r"/Users/[A-Za-z0-9._-]+/code/qwen-llm(?:-[A-Za-z0-9._-]+)?/"),
        "$REPO/",
    ),
    (re.compile(r"/Users/[A-Za-z0-9._-]+/"), "$HOME/"),
    (re.compile(r"/Volumes/[A-Za-z0-9._-]+/"), "$VOLUME/"),
]


def maintained_docs() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "docs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        ROOT / name
        for name in listed.stdout.splitlines()
        if not name.startswith("docs/bench/")
    ]


def normalize(text: str) -> str:
    for pattern, replacement in RULES:
        text = pattern.sub(replacement, text)
    return text


def self_test() -> None:
    cases = {
        "/Users/someone/models/Qwen3.5-2B-Q8_0.gguf": "$MODELS/Qwen3.5-2B-Q8_0.gguf",
        "/Volumes/disk/weights-archive/glm/x.gguf": "$MODELS/glm/x.gguf",
        "/Users/someone/code/qwen-llm-work/target/release/qwen": "$REPO/target/release/qwen",
        "/Users/someone/code/qwen-llm/docs/ENV.md": "$REPO/docs/ENV.md",
        "/Users/someone/code/llama.cpp/build": "$HOME/code/llama.cpp/build",
        "/Users/someone/.cache/qwen-llm/x": "$HOME/.cache/qwen-llm/x",
        "/Volumes/other/data": "$VOLUME/data",
        "relative/path and $MODELS/x": "relative/path and $MODELS/x",
    }
    for given, expected in cases.items():
        actual = normalize(given)
        assert actual == expected, f"{given!r}: {actual!r} != {expected!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if a maintained doc needs normalizing",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="run the replacement fixtures"
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    stale = []
    for path in maintained_docs():
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        normalized = normalize(text)
        if normalized != text:
            stale.append(path.relative_to(ROOT))
            if not args.check:
                path.write_text(normalized)
    if args.check and stale:
        print(
            "machine-specific paths in maintained docs; run scripts/reference/normalize_paths.py:",
            file=sys.stderr,
        )
        for name in stale:
            print(f"  {name}", file=sys.stderr)
        return 1
    if not args.check:
        print(f"normalized {len(stale)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
