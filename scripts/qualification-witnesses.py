#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run device qualification witnesses and write an evidence report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import time

KINDS = ("replay", "m4_characterization_skip", "scope_only", "none")
RESULT = re.compile(
    r"^test result: (?P<outcome>ok|FAILED)\. "
    r"(?P<passed>\d+) passed; (?P<failed>\d+) failed(?:;|$)",
    re.MULTILINE,
)
RUNNING = re.compile(r"^running (?P<count>\d+) tests?$", re.MULTILINE)


def parse_command(value: str, kind: str) -> tuple[dict[str, str], list[str]]:
    """Parse a witness cargo command and any leading environment assignments."""
    command = value
    if kind == "scope_only":
        prefix = "scope-only: "
        if not command.startswith(prefix):
            raise ValueError("scope_only command is missing 'scope-only: ' prefix")
        command = command[len(prefix) :]
    if kind == "none":
        if not command.startswith("none: ") or not command[len("none: ") :].strip():
            raise ValueError("none witness must state a reason after 'none: '")
        return {}, []
    tokens = shlex.split(command)
    environment: dict[str, str] = {}
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        key, value = tokens.pop(0).split("=", 1)
        environment[key] = value
    if not tokens or tokens[0] != "cargo" or len(tokens) < 2 or tokens[1] != "test":
        raise ValueError("witness command must start with cargo test after environment assignments")
    if any(token in {";", "&&", "||", "|", ">", "<"} for token in tokens):
        raise ValueError("witness command may not contain shell operators")
    return environment, tokens


def libtest_passes(output: str) -> bool:
    """Accept a libtest result only when at least one selected test passed."""
    results = list(RESULT.finditer(output))
    running = list(RUNNING.finditer(output))
    if len(results) != 1 or len(running) != 1:
        return False
    result = results[0]
    passed = int(result.group("passed"))
    failed = int(result.group("failed"))
    return (
        result.group("outcome") == "ok"
        and passed >= 1
        and failed == 0
        and int(running[0].group("count")) == passed + failed
    )


def self_test() -> None:
    env, command = parse_command(
        "QWEN_ONE=two QWEN_MANY='three four' cargo test --release -p qwen-llm --lib case -- --exact",
        "replay",
    )
    assert env == {"QWEN_ONE": "two", "QWEN_MANY": "three four"}
    assert command[:4] == ["cargo", "test", "--release", "-p"]
    _, scope_command = parse_command("scope-only: cargo test --release -p qwen-llm case -- --exact", "scope_only")
    assert scope_command[:2] == ["cargo", "test"]
    assert parse_command("none: no executable witness was recorded", "none") == ({}, [])
    assert libtest_passes("running 1 test\ntest sample ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n")
    assert libtest_passes("running 3 tests\ntest result: ok. 3 passed; 0 failed; 0 ignored; 14 measured; 20 filtered out\n")
    assert not libtest_passes("running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out\n")
    assert not libtest_passes("running 2 tests\ntest result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n")
    assert not libtest_passes("running 1 test\ntest result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n")
    assert not libtest_passes("running 2 tests\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out\n")

    root = pathlib.Path("/repo/root")
    record = {
        "path": "sample.path",
        "holds_for_device": True,
        "correctness": "cargo test --release -p qwen-llm --lib sample -- --exact",
        "correctness_kind": "replay",
    }
    args = argparse.Namespace(dry_run=False)
    successful = subprocess.CompletedProcess(
        args=record["correctness"], returncode=0,
        stdout="running 1 test\ntest sample ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n",
        stderr="measurement gpu_ms=1.25\n",
    )
    calls = []

    def mocked_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, "revision-before-run\n", "")
        return successful

    from unittest.mock import patch

    with patch("subprocess.run", side_effect=mocked_run):
        revision = git_revision(root)
        witness, failed = witness_entry(record, "correctness", args, {}, root)
    assert revision == "revision-before-run"
    assert [call[0][0] for call in calls] == ["git", record["correctness"].split()[0]]
    assert calls[1][1]["cwd"] == root
    assert witness["status"] == "passed" and not failed
    assert witness["measurement_lines"] == ["measurement gpu_ms=1.25"]

    for code, expected_status in ((0, "passed"), (1, "failed")):
        characterization = {
            **record,
            "correctness": "cargo test --release -p qwen-llm --lib characterization -- --exact",
            "correctness_kind": "m4_characterization_skip",
        }
        completed = subprocess.CompletedProcess(
            args=characterization["correctness"],
            returncode=code,
            stdout="running 1 test\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n" if code == 0 else "running 1 test\ntest result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\nstdout failure tail\n",
            stderr="stderr failure tail\n" if code else "",
        )
        with patch("subprocess.run", return_value=completed):
            witness, failed = witness_entry(characterization, "correctness", args, {}, root)
        assert witness["status"] == expected_status
        assert failed is (code != 0)
        if code:
            assert witness["stdout_tail"].endswith("stdout failure tail")
            assert witness["stderr_tail"].endswith("stderr failure tail")

    scope_record = {
        **record,
        "correctness": "scope-only: cargo test --release -p qwen-llm --lib scope -- --exact",
        "correctness_kind": "scope_only",
    }
    with patch("subprocess.run", return_value=successful):
        witness, failed = witness_entry(scope_record, "correctness", args, {}, root)
    assert witness["status"] == "scope-only" and witness["execution_status"] == "passed" and not failed
    failed_scope = subprocess.CompletedProcess(
        args=scope_record["correctness"], returncode=1,
        stdout="running 1 test\ntest result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n",
        stderr="scope failure\n",
    )
    with patch("subprocess.run", return_value=failed_scope):
        witness, failed = witness_entry(scope_record, "correctness", args, {}, root)
    assert witness["status"] == "failed" and witness["execution_status"] == "failed" and failed

    with tempfile.TemporaryDirectory() as directory:
        report_path = pathlib.Path(directory) / "report.json"
        report = {"schema_version": 1, "git_commit": "revision-before-run", "records": []}
        report_record = {"path": "sample.path", "witnesses": []}
        report["records"].append(report_record)
        with patch("subprocess.run", side_effect=[successful, OSError("launch failed")]):
            first_failed = record_witness(report, report_record, record, "correctness", args, {}, root, report_path)
            saved_after_first = json.loads(report_path.read_text())
            failed_record = {
                **record,
                "performance": "cargo test --release -p qwen-llm --lib launch -- --exact",
                "performance_kind": "replay",
            }
            second_failed = record_witness(report, report_record, failed_record, "performance", args, {}, root, report_path)
        saved_after_second = json.loads(report_path.read_text())
        assert not first_failed and second_failed
        assert len(saved_after_first["records"][0]["witnesses"]) == 1
        assert len(saved_after_second["records"][0]["witnesses"]) == 2
        launch_failure = saved_after_second["records"][0]["witnesses"][1]
        assert launch_failure["status"] == "failed"
        assert launch_failure["launch_error"] == "launch failed"
        assert launch_failure["stdout_tail"] == "" and launch_failure["stderr_tail"] == ""


def run_info(binary: pathlib.Path, root: pathlib.Path) -> dict:
    if not binary.is_file():
        raise RuntimeError(f"qwen binary not found: {binary} (build it explicitly; this script does not build)")
    result = subprocess.run([str(binary), "info", "--json"], cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{binary} info --json failed with exit code {result.returncode}:\n{tail(result.stdout + result.stderr)}")
    try:
        facts = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{binary} info --json did not return valid JSON: {error}") from error
    if not isinstance(facts, dict) or not isinstance(facts.get("qualifications"), list):
        raise RuntimeError("qwen info --json is missing its qualifications array")
    return facts


def tail(output: str, lines: int = 40) -> str:
    return "\n".join(output.rstrip().splitlines()[-lines:])


def witness_entry(
    record: dict,
    facet: str,
    args: argparse.Namespace,
    facts: dict,
    root: pathlib.Path,
) -> tuple[dict, bool]:
    kind = record[f"{facet}_kind"]
    value = record[facet]
    entry = {
        "facet": facet,
        "kind": kind,
        "command": value,
        "status": None,
        "seconds": None,
        "measurement_lines": [],
        "output_tail": None,
    }
    if kind not in KINDS:
        raise RuntimeError(f"unknown witness kind {kind!r} for {record.get('path')} {facet}")
    if kind == "none":
        reason = value.removeprefix("none: ")
        entry["status"] = "none"
        entry["reason"] = reason
        return entry, False
    if kind == "m4_characterization_skip" and not record.get("holds_for_device", False):
        entry["status"] = "not applicable on this device"
        entry["decline_reason"] = record.get("decline_reason")
        entry["reason"] = "qualification does not hold for this device; skipped to avoid a vacuous pass"
        return entry, False
    environment, command = parse_command(value, kind)
    entry["argv"] = command
    entry["environment"] = environment
    if kind == "scope_only":
        entry["reason"] = "scope-only check; its result is not execution evidence"
    elif kind == "m4_characterization_skip":
        entry["reason"] = "qualification holds for this device; run the M4 characterization"
    else:
        entry["reason"] = "replay witness"
    if args.dry_run:
        entry["status"] = "would run" if kind != "scope_only" else "would run (scope-only)"
        return entry, False

    child_env = os.environ.copy()
    child_env.update(environment)
    child_env["QWEN_METAL_LEASE_WAIT"] = "1"
    child_env["QWEN_REQUIRE_METAL_TESTS"] = "1"
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=root, env=child_env, capture_output=True, text=True)
    except OSError as error:
        entry["seconds"] = round(time.monotonic() - started, 3)
        entry["status"] = "failed"
        entry["exit_code"] = None
        entry["launch_error"] = str(error)
        entry["stdout_tail"] = ""
        entry["stderr_tail"] = ""
        return entry, True
    entry["seconds"] = round(time.monotonic() - started, 3)
    output = result.stdout + result.stderr
    entry["measurement_lines"] = [
        line for line in result.stderr.splitlines() if "_ms=" in line or "tok/s" in line or "gpu_ms" in line
    ]
    passed = result.returncode == 0 and libtest_passes(output)
    if passed:
        entry["status"] = "scope-only" if kind == "scope_only" else "passed"
        if kind == "scope_only":
            entry["execution_status"] = "passed"
        return entry, False
    entry["status"] = "failed"
    if kind == "scope_only":
        entry["execution_status"] = "failed"
    entry["exit_code"] = result.returncode
    entry["stdout_tail"] = tail(result.stdout)
    entry["stderr_tail"] = tail(result.stderr)
    return entry, True


def git_revision(root: pathlib.Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def write_report(report: dict, output_path: pathlib.Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary_path = pathlib.Path(temporary.name)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def record_witness(
    report: dict,
    report_record: dict,
    record: dict,
    facet: str,
    args: argparse.Namespace,
    facts: dict,
    root: pathlib.Path,
    output_path: pathlib.Path,
) -> bool:
    witness, failed = witness_entry(record, facet, args, facts, root)
    report_record["witnesses"].append(witness)
    write_report(report, output_path)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=pathlib.Path, help="qwen binary (default: target/release/qwen)")
    parser.add_argument("--filter", help="only include qualification paths containing this substring")
    parser.add_argument("--out", type=pathlib.Path, help="required JSON report path")
    parser.add_argument("--dry-run", action="store_true", help="list decisions without executing witnesses")
    parser.add_argument("--self-test", action="store_true", help="run parser and libtest pass-count checks")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("qualification-witnesses self-test passed")
        return 0
    if args.out is None:
        parser.error("--out PATH is required")
    root = pathlib.Path(__file__).resolve().parents[1]
    binary = (args.qwen or root / "target/release/qwen").expanduser().resolve()
    try:
        facts = run_info(binary, root)
        records = facts["qualifications"]
        if args.filter:
            records = [record for record in records if args.filter in record.get("path", "")]
        if not records:
            raise RuntimeError("no qualification records matched the requested filter")
        commit = git_revision(root)
        report_records = []
        any_failed = False
        report = {
            "schema_version": 1,
            "git_commit": commit,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "device_facts": facts,
            "dry_run": args.dry_run,
            "records": report_records,
        }
        write_report(report, args.out)
        for record in records:
            report_record = {
                "path": record["path"],
                "holds_for_device": record["holds_for_device"],
                "decline_reason": record.get("decline_reason"),
                "witnesses": [],
            }
            report_records.append(report_record)
            for facet in ("correctness", "performance"):
                any_failed |= record_witness(
                    report, report_record, record, facet, args, facts, root, args.out
                )
        print(f"{'PATH':52} {'WITNESS':13} {'STATUS':30} WHY")
        for record in report_records:
            for witness in record["witnesses"]:
                status = witness["status"]
                if witness.get("execution_status") == "failed":
                    status += " (execution failed)"
                print(f"{record['path']:52} {witness['facet']:13} {status:30} {witness.get('reason', '')}")
        print(f"Report: {args.out}")
        return 1 if any_failed else 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"qualification-witnesses: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
