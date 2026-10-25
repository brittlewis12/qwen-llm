#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Exact CPU acceptance inventory; GPU/model witnesses are listed, never run.

Run under scripts/bench/with-metal-lease with QWEN_METAL_LEASE_WAIT=1.
After composing source lanes, review and extend acceptance_inventory.json by
full test name, class, ignore state and reason. Discovery is not classification.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = Path(__file__).with_suffix(".json")
TARGETS = {
    "lib": ["-p", "qwen-llm", "--lib"],
    "oracle": ["-p", "qwen-cli", "--test", "deepseek_v41_oracle"],
}
CLASSES = {"cpu", "gpu", "model", "fixture", "timing"}


def validate_entries(entries: list[dict]) -> None:
    seen = set()
    for entry in entries:
        if set(entry) != {"target", "name", "class", "ignored", "reason"}:
            raise ValueError(f"invalid inventory fields: {entry}")
        key = (entry["target"], entry["name"])
        if key in seen:
            raise ValueError(f"duplicate inventory entry: {key}")
        seen.add(key)
        if entry["target"] not in TARGETS or entry["class"] not in CLASSES:
            raise ValueError(f"invalid classification: {entry}")
        if not re.fullmatch(r"\w+(?:::\w+)*", entry["name"]):
            raise ValueError(f"inventory requires full names, not patterns: {entry}")
        if type(entry["ignored"]) is not bool or not entry["reason"].strip():
            raise ValueError(f"missing ignore state/reason: {entry}")
        if entry["class"] == "cpu" and entry["ignored"]:
            raise ValueError(
                f"ignored correctness test cannot enter CPU acceptance: {entry}"
            )
        if entry["target"] == "lib" and not entry["name"].startswith("deepseek_v41::"):
            raise ValueError(f"outside the maintained V41 lib scope: {entry}")
    for target in TARGETS:
        if not any(e["target"] == target and e["class"] == "cpu" for e in entries):
            raise ValueError(f"empty CPU inventory: {target}")


def listed(output: str) -> set[str]:
    names = re.findall(r"^(.+): test$", output, re.M)
    summary = re.findall(r"^(\d+) tests?, 0 benchmarks?$", output, re.M)
    if len(names) != len(set(names)) or summary != [str(len(names))]:
        raise ValueError(f"malformed/duplicate test discovery: {output}")
    return set(names)


def check_discovery(
    entries: list[dict], target: str, all_tests: set[str], ignored: set[str]
) -> None:
    expected = {e["name"] for e in entries if e["target"] == target}
    expected_ignored = {
        e["name"] for e in entries if e["target"] == target and e["ignored"]
    }
    if all_tests != expected or ignored != expected_ignored:
        raise ValueError(
            f"{target}: inventory mismatch; unclassified={sorted(all_tests - expected)}, "
            f"stale={sorted(expected - all_tests)}, "
            f"unexpected_ignored={sorted(ignored - expected_ignored)}, "
            f"no_longer_ignored={sorted(expected_ignored - ignored)}"
        )


def check_execution(output: str, names: set[str], filtered: int) -> None:
    results = re.findall(r"^test (\S+) \.\.\. (\S+).*$", output, re.M)
    expected = sorted((name, "ok") for name in names)
    if sorted(results) != expected:
        raise ValueError(f"unexpected tests, failures or skips: {results}")
    summary = re.findall(
        r"^test result: ok\. (\d+) passed; 0 failed; 0 ignored; 0 measured; (\d+) filtered out;.*$",
        output,
        re.M,
    )
    if summary != [(str(len(names)), str(filtered))] or not names:
        raise ValueError("execution count mismatch or empty correctness run")


def self_test() -> None:
    entries = [
        {
            "target": target,
            "name": name,
            "class": "cpu",
            "ignored": False,
            "reason": "test",
        }
        for target, name in [("lib", "deepseek_v41::positive"), ("oracle", "positive")]
    ]
    validate_entries(entries)
    check_discovery(entries, "lib", {"deepseek_v41::positive"}, set())
    for all_tests, ignored in [
        (set(), set()),
        ({"deepseek_v41::positive", "new"}, set()),
        ({"deepseek_v41::positive"}, {"deepseek_v41::positive"}),
    ]:
        try:
            check_discovery(entries, "lib", all_tests, ignored)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted changed inventory/ignore state")
    valid = "test positive ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out; finished in 0.0s\n"
    check_execution(valid, {"positive"}, 7)
    for bad in [
        valid.replace("... ok", "... ignored"),
        valid.replace("1 passed", "0 passed"),
        valid.replace("0 ignored", "1 ignored"),
        valid.replace("7 filtered", "8 filtered"),
        valid + "test new ... ok\n",
        "",
    ]:
        try:
            check_execution(bad, {"positive"}, 7)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted skipped/changed execution")
    for bad in [
        entries + [entries[0]],
        [dict(e, name="*") for e in entries],
        [dict(e, ignored=True) for e in entries],
    ]:
        try:
            validate_entries(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid registry")
    print("acceptance_inventory self-tests: 12 corrupt controls rejected")
    assert listed("positive: test\n\n1 test, 0 benchmarks\n") == {"positive"}
    for bad in [
        "",
        "positive: test\n\n0 tests, 0 benchmarks\n",
        "positive: test\npositive: test\n\n2 tests, 0 benchmarks\n",
    ]:
        try:
            listed(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted malformed discovery")
    print("acceptance_inventory discovery self-tests: 3 corrupt controls rejected")


def binary(target: str) -> Path:
    result = subprocess.run(
        [
            "cargo",
            "test",
            "--release",
            *TARGETS[target],
            "--no-run",
            "--message-format=json",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
    )
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    for message in messages:
        if message.get("reason") == "compiler-message":
            rendered = message.get("message", {}).get("rendered")
            if rendered:
                print(rendered, end="", file=sys.stderr)
    result.check_returncode()
    executables = {
        message["executable"]
        for message in messages
        if message.get("reason") == "compiler-artifact"
        and message.get("executable")
        and message.get("profile", {}).get("test")
    }
    if len(executables) != 1:
        raise ValueError(f"ambiguous {target} artifact: {executables}")
    return Path(executables.pop())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="compile and check discovery, without execution",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if os.environ.get("QWEN_METAL_LEASE_WAIT") != "1":
        raise ValueError(
            "CPU compile/test requires QWEN_METAL_LEASE_WAIT=1 and the production lease wrapper"
        )
    entries = json.loads(INVENTORY.read_text())
    validate_entries(entries)
    artifacts = {}
    for target in TARGETS:
        path = binary(target)
        total = listed(
            subprocess.check_output([str(path), "--list"], text=True, cwd=ROOT)
        )
        artifacts[target] = (path, len(total))
        scope = ["deepseek_v41::"] if target == "lib" else []
        discover = lambda extra: listed(
            subprocess.check_output(
                [str(path), *scope, "--list", *extra],
                text=True,
                cwd=ROOT,
            )
        )
        check_discovery(entries, target, discover([]), discover(["--ignored"]))
    for target, (path, total) in artifacts.items():
        subset = [e for e in entries if e["target"] == target]
        names = {e["name"] for e in subset if e["class"] == "cpu"}
        print(
            f"{target}: inventory={len(subset)} cpu={len(names)} excluded={len(subset) - len(names)} "
            f"ignored={sum(e['ignored'] for e in subset)} classes={dict(Counter(e['class'] for e in subset))}",
            flush=True,
        )
        if not args.check_only:
            result = subprocess.run(
                [str(path), "--exact", "--test-threads=1", *sorted(names)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )
            print(result.stdout, end="", flush=True)
            result.check_returncode()
            check_execution(result.stdout, names, total - len(names))
            print(
                f"{target}: acceptance passed exactly {len(names)} CPU tests; 0 ignored; 0 skipped",
                flush=True,
            )


if __name__ == "__main__":
    main()
