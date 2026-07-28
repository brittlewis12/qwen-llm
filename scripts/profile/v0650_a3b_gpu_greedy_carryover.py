#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PREREG = ROOT / "docs/bench/v0650-a3b-gpu-greedy-carryover.md"
PERF_LOG = ROOT / "docs/PERF-LOG.md"
PERF_ROADMAP = ROOT / "docs/PERF-ROADMAP.md"
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
QWEN = ROOT / "target/release/qwen"
QWEN_BENCH = ROOT / "target/release/qwen-bench"
IMPLEMENTATION_SOURCE = (
    ROOT / "crates/qwen-cli/src/main.rs",
    ROOT / "crates/qwen-llm/src/runtime.rs",
)
PACKET = ROOT / "target/profiles/v0650-a3b-gpu-greedy-carryover-p1"

PREREG_COMMIT = "371d1e959ebd5f9942e674485c356fcbc000b0f1"
IMPLEMENTATION_COMMIT = "15a7092779e81e190442fe89e6d59d6ee7e6d3f7"
RESULT_DOC_COMMIT = "543f71c2610099b888719f56d59885bedcc3fc30"
MODEL = {
    "name": "a3b",
    "path": Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    "sha256": "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
    "runtime_identity_kind": "metadata_compatibility_v1",
    "runtime_model_id": "e6024ce53109fdf7",
    "runtime_tokenizer_id": "a4b0b26f8a8c9917",
}
PROMPT_SHA256 = "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
PROMPT_BYTES = 1_891
PROMPT_TOKENS = 419
PAIR_ORDERS = ("AB", "BA") * 8
COOLDOWN_SECONDS = 5.0
SCHEMA_VERSION = 7
T95_ONE_SIDED_N16 = 1.75305
PINNED_QWEN_ENV = {
    "QWEN_DECODE_GDN_FUSED_BETA_PROJ": "1",
    "QWEN_DECODE_ROPE_PAIR": "1",
    "QWEN_DECODE_MOE_GROUPED_FINALIZER": "1",
    "QWEN_DECODE_MOE_FUSED_FINALIZER": "1",
}
TOKEN_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
KNOWN_INFERENCE_EXECUTABLES = {
    "qwen",
    "qwen-bench",
    "llama-cli",
    "llama-server",
    "llama-bench",
    "ollama",
    "mlx_lm",
}
RUSAGE_FIELDS = (
    "ru_utime",
    "ru_stime",
    "ru_maxrss",
    "ru_minflt",
    "ru_majflt",
    "ru_inblock",
    "ru_oublock",
    "ru_nvcsw",
    "ru_nivcsw",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int = 900,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def command_text(argv: list[str]) -> str:
    return command(argv).stdout.strip()


def rusage_snapshot() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {field: getattr(usage, field) for field in RUSAGE_FIELDS}


def rusage_record(
    before: dict[str, float | int], after: dict[str, float | int]
) -> dict[str, Any]:
    return {
        "scope": "descriptive_rusage_children_around_qwen_command_only",
        "validity_or_decision_use": "forbidden",
        "before": before,
        "after": after,
        "delta": {
            field: after[field] - before[field]
            for field in RUSAGE_FIELDS
            if field != "ru_maxrss"
        },
        "maxrss_before": before["ru_maxrss"],
        "maxrss_after": after["ru_maxrss"],
    }


def archive_command(
    stem: str,
    argv: list[str],
    *,
    env: dict[str, str],
    timeout: int = 900,
    record_child_rusage: bool = False,
) -> subprocess.CompletedProcess[str]:
    started_ns = time.time_ns()
    before_usage = rusage_snapshot() if record_child_rusage else None
    try:
        result = command(argv, env=env, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        after_usage = rusage_snapshot() if record_child_rusage else None
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        (PACKET / f"{stem}.stdout.txt").write_text(stdout)
        (PACKET / f"{stem}.stderr.txt").write_text(stderr)
        metadata: dict[str, Any] = {
            "argv": argv,
            "elapsed_ms": (time.time_ns() - started_ns) / 1e6,
            "returncode": None,
            "timed_out": True,
            "timeout_seconds": timeout,
        }
        if before_usage is not None and after_usage is not None:
            metadata["child_rusage"] = rusage_record(before_usage, after_usage)
        write_json(PACKET / f"{stem}.command.json", metadata)
        raise RuntimeError(f"{stem} timed out after {timeout} seconds") from error
    after_usage = rusage_snapshot() if record_child_rusage else None
    (PACKET / f"{stem}.stdout.txt").write_text(result.stdout)
    (PACKET / f"{stem}.stderr.txt").write_text(result.stderr)
    metadata = {
        "argv": argv,
        "elapsed_ms": (time.time_ns() - started_ns) / 1e6,
        "returncode": result.returncode,
        "timed_out": False,
        "timeout_seconds": timeout,
    }
    if before_usage is not None and after_usage is not None:
        metadata["child_rusage"] = rusage_record(before_usage, after_usage)
    write_json(PACKET / f"{stem}.command.json", metadata)
    return result


def normalized_environment(arm: str = "A") -> dict[str, str]:
    require(arm in {"A", "B"}, f"unknown arm: {arm}")
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("QWEN_")
    }
    env["NO_COLOR"] = "1"
    env.update(PINNED_QWEN_ENV)
    env["QWEN_GREEDY_GPU_ARGMAX"] = "0" if arm == "A" else "1"
    return env


def file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def assert_frozen_blob(path: Path, commit: str) -> None:
    expected = command_text(["git", "rev-parse", f"{commit}:{relative(path)}"])
    actual = command_text(["git", "hash-object", str(path)])
    require(actual == expected, f"frozen blob changed: {relative(path)}")


def assert_implementation_unchanged(*, runner_must_be_committed: bool = False) -> None:
    for commit in (IMPLEMENTATION_COMMIT, RESULT_DOC_COMMIT, PREREG_COMMIT):
        command(["git", "merge-base", "--is-ancestor", commit, "HEAD"])
    script_relative = relative(SCRIPT)
    changed = sorted(
        command_text(
            ["git", "diff", "--name-only", f"{IMPLEMENTATION_COMMIT}..HEAD"]
        ).splitlines()
    )
    expected_docs = sorted(
        [relative(PERF_LOG), relative(PERF_ROADMAP), relative(PREREG)]
    )
    expected_all = sorted([*expected_docs, script_relative])
    if runner_must_be_committed:
        require(
            changed == expected_all,
            f"post-implementation delta is {changed}, not {expected_all}",
        )
    else:
        require(
            changed in (expected_docs, expected_all),
            f"unexpected post-implementation delta: {changed}",
        )
    assert_frozen_blob(PREREG, PREREG_COMMIT)
    assert_frozen_blob(PERF_LOG, RESULT_DOC_COMMIT)
    assert_frozen_blob(PERF_ROADMAP, RESULT_DOC_COMMIT)
    assert_frozen_blob(PROMPT, IMPLEMENTATION_COMMIT)
    for path in IMPLEMENTATION_SOURCE:
        assert_frozen_blob(path, IMPLEMENTATION_COMMIT)


def static_check() -> None:
    required = (
        SCRIPT,
        PREREG,
        PERF_LOG,
        PERF_ROADMAP,
        PROMPT,
        MODEL["path"],
        *IMPLEMENTATION_SOURCE,
    )
    for path in required:
        require(path.exists(), f"missing required path: {path}")
    require(PROMPT.stat().st_size == PROMPT_BYTES, "prompt byte count changed")
    require(sha256_file(PROMPT) == PROMPT_SHA256, "prompt hash changed")
    assert_implementation_unchanged()
    require(PAIR_ORDERS == ("AB", "BA") * 8, "pair order drifted")
    require(len(PAIR_ORDERS) == 16, "sample count drifted")
    env_a = normalized_environment("A")
    env_b = normalized_environment("B")
    differences = {
        key for key in env_a.keys() | env_b.keys() if env_a.get(key) != env_b.get(key)
    }
    require(differences == {"QWEN_GREEDY_GPU_ARGMAX"}, "arm environments drifted")
    require(env_a["QWEN_GREEDY_GPU_ARGMAX"] == "0", "A is not rollback")
    require(env_b["QWEN_GREEDY_GPU_ARGMAX"] == "1", "B is not force-enabled")
    require(all(env_a[key] == "1" for key in PINNED_QWEN_ENV), "hybrid pins drifted")
    synthetic = [
        {
            "A": {"metrics": {"carry": 1.0, "treatment_prefill": 1.0}},
            "B": {"metrics": {"carry": 1.01, "treatment_prefill": 1.0}},
        }
        for _ in PAIR_ORDERS
    ]
    summary = carryover_summary(synthetic)
    require(
        summary["n"] == 16 and summary["carry"]["estimate"] > 1.0,
        "statistic self-test failed",
    )

    def synthetic_result(baseline: float, lower: float, upper: float) -> dict[str, Any]:
        return {
            "pairs_completed": 16,
            "summary": {
                "baseline": {"geometric_ratio": baseline},
                "carry": {
                    "lower_95_one_sided": lower,
                    "upper_95_one_sided": upper,
                },
            },
        }

    require(
        decide(synthetic_result(0.96, 0.99, 1.01))["verdict"]
        == "INCONCLUSIVE_BASELINE_IMBALANCE",
        "baseline decision self-test failed",
    )
    require(
        decide(synthetic_result(1.0, 1.031, 1.05))["verdict"]
        == "CARRYOVER_HARM_SIGNAL",
        "harm decision self-test failed",
    )
    require(
        decide(synthetic_result(1.0, 0.98, 1.029))["verdict"]
        == "NO_MATERIAL_CARRYOVER_AT_3_PERCENT",
        "no-material decision self-test failed",
    )
    require(
        decide(synthetic_result(1.0, 0.99, 1.03))["verdict"] == "INCONCLUSIVE",
        "boundary decision self-test failed",
    )
    require(
        classify_competing_processes(
            "1 opencode opencode recall search qwen --from 2026-05-01\n"
        )
        == [],
        "ordinary query arguments trigger process predicate",
    )
    classified = classify_competing_processes(
        "2 qwen /tmp/qwen --model x\n"
        "3 Python /usr/bin/python3 -m mlx_lm.generate --model x\n"
        "4 ollama /usr/local/bin/ollama serve\n"
    )
    require(len(classified) == 3, "known inference process self-test failed")
    require(
        classify_competing_processes(
            "5 Python python3 unrelated.py --query -m mlx_lm.generate\n"
            "6 Python python3 -c 'print(1)' -m mlx_lm.generate\n"
            "7 Python python3 -- -m mlx_lm.generate\n"
        )
        == [],
        "post-interpreter data triggers module predicate",
    )


def is_mlx_entry_point(value: str) -> bool:
    return value == "mlx_lm" or value.startswith("mlx_lm.")


def is_known_inference_executable(value: str) -> bool:
    name = Path(value).name
    return name in KNOWN_INFERENCE_EXECUTABLES or is_mlx_entry_point(name)


def is_python_interpreter(value: str) -> bool:
    return re.fullmatch(r"python(?:3(?:\.\d+)*)?", Path(value).name.lower()) is not None


def python_module_operand(argv: list[str]) -> str | None:
    if not argv or not is_python_interpreter(argv[0]):
        return None
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "-m":
            return argv[index + 1] if index + 1 < len(argv) else None
        if token in {"-c", "--", "-"} or not token.startswith("-"):
            return None
        index += 2 if token in {"-W", "-X"} and index + 1 < len(argv) else 1
    return None


def classify_competing_processes(raw_processes: str) -> list[str]:
    competitors = []
    for line in raw_processes.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) < 2:
            continue
        comm = fields[1]
        arguments = fields[2] if len(fields) == 3 else ""
        try:
            argv = shlex.split(arguments)
        except ValueError:
            argv = arguments.split()
        argv0 = argv[0] if argv else ""
        module = python_module_operand(argv)
        if (
            is_known_inference_executable(comm)
            or (argv0 and is_known_inference_executable(argv0))
            or (module is not None and is_mlx_entry_point(module))
        ):
            competitors.append(line.strip())
    return competitors


def capture_host_command(argv: list[str], timeout: int = 30) -> dict[str, Any]:
    started_ns = time.time_ns()
    try:
        result = subprocess.run(
            argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False
        )
        return {
            "argv": argv,
            "elapsed_ms": (time.time_ns() - started_ns) / 1e6,
            "returncode": result.returncode,
            "timed_out": False,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return {
            "argv": argv,
            "elapsed_ms": (time.time_ns() - started_ns) / 1e6,
            "returncode": None,
            "timed_out": True,
            "stdout": stdout,
            "stderr": stderr,
        }


def collect_host_snapshot() -> dict[str, Any]:
    return {
        "commands": {
            "memory_pressure": capture_host_command(["memory_pressure", "-Q"]),
            "thermal": capture_host_command(["pmset", "-g", "therm"]),
            "processes": capture_host_command(["ps", "-axo", "pid=,comm=,args="]),
        }
    }


def evaluate_host_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(raw)
    commands = snapshot["commands"]
    memory_stdout = commands["memory_pressure"]["stdout"]
    thermal_stdout = commands["thermal"]["stdout"]
    free_line = next(
        (
            line
            for line in memory_stdout.splitlines()
            if line.startswith("System-wide memory free percentage:")
        ),
        None,
    )
    free_percent = None
    free_percent_error = None
    if free_line is not None:
        try:
            free_percent = int(free_line.rsplit(" ", 1)[-1].rstrip("%"))
        except ValueError as error:
            free_percent_error = str(error)
    snapshot.update(
        {
            "free_percent": free_percent,
            "free_percent_error": free_percent_error,
            "competitors": classify_competing_processes(
                commands["processes"]["stdout"]
            ),
            "thermal_warning_clear": "No thermal warning level has been recorded"
            in thermal_stdout,
            "performance_warning_clear": "No performance warning level has been recorded"
            in thermal_stdout,
        }
    )
    return snapshot


def validate_host_snapshot(snapshot: dict[str, Any]) -> None:
    for name, record in snapshot["commands"].items():
        require(not record["timed_out"], f"host command timed out: {name}")
        require(record["returncode"] == 0, f"host command failed: {name}")
    require(snapshot["free_percent_error"] is None, "invalid memory free percentage")
    require(
        snapshot["free_percent"] is not None, "memory pressure omitted free percentage"
    )
    require(
        snapshot["free_percent"] >= 85,
        f"memory free percentage too low: {snapshot['free_percent']}",
    )
    require(
        not snapshot["competitors"],
        f"competing inference processes: {snapshot['competitors']}",
    )
    require(snapshot["thermal_warning_clear"], "thermal warning present")
    require(snapshot["performance_warning_clear"], "performance warning present")


def host_snapshot(stem: str) -> dict[str, Any]:
    path = PACKET / f"{stem}.host.json"
    raw = collect_host_snapshot()
    write_json(path, raw)
    snapshot = evaluate_host_snapshot(raw)
    write_json(path, snapshot)
    validate_host_snapshot(snapshot)
    return snapshot


def execution_source_preflight() -> tuple[dict[str, Any], dict[str, str]]:
    require(not PACKET.exists(), f"packet already exists: {PACKET}")
    static_check()
    dirty = command_text(["git", "status", "--porcelain=v1"])
    require(not dirty, f"source is dirty: {dirty!r}")
    for path in (SCRIPT, PREREG, PERF_LOG, PERF_ROADMAP, PROMPT):
        command(["git", "ls-files", "--error-unmatch", relative(path)])
    assert_implementation_unchanged(runner_must_be_committed=True)
    require(PACKET.parent.is_dir(), f"packet parent does not exist: {PACKET.parent}")
    inherited = {
        key: value for key, value in os.environ.items() if key.startswith("QWEN_")
    }
    snapshot = evaluate_host_snapshot(collect_host_snapshot())
    validate_host_snapshot(snapshot)
    return snapshot, inherited


def source_and_build_identity() -> tuple[str, dict[str, Any]]:
    dirty = command_text(["git", "status", "--porcelain=v1"])
    require(not dirty, f"source is dirty: {dirty!r}")
    assert_implementation_unchanged(runner_must_be_committed=True)
    commit = command_text(["git", "rev-parse", "HEAD"])
    build = json.loads(
        command_text([str(QWEN_BENCH), "build-info", "--output", "json"])
    )
    require(build["build_commit"] == commit, "build commit does not match source")
    require(build["runtime_commit"] == commit, "runtime commit does not match source")
    require(build["status"] == "match", "build/runtime identity mismatch")
    require(
        build["build_dirty"] is False and build["runtime_dirty"] is False,
        "dirty build/runtime",
    )
    require(
        build["build_source_state"] == build["runtime_source_state"],
        "source-state mismatch",
    )
    return commit, build


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), f"non-object row in {path}")
            rows.append(value)
    return rows


def expected_policy(arm: str) -> tuple[str, str]:
    if arm == "A":
        return "greedy_argmax", "disabled_by_explicit_rollback"
    return "greedy_gpu_argmax", "force_enabled"


def validate_digest(row: dict[str, Any], where: str) -> str:
    digest = row.get("generated_token_sha256")
    require(
        isinstance(digest, str) and TOKEN_SHA256_RE.fullmatch(digest) is not None,
        f"invalid token digest in {where}",
    )
    return digest


def finite(
    row: dict[str, Any], key: str, where: str, *, positive: bool = True
) -> float:
    value = float(row[key])
    require(math.isfinite(value), f"non-finite {where} {key}")
    require(
        value > 0.0 if positive else value >= 0.0, f"invalid {where} {key}: {value}"
    )
    return value


def validate_policy(row: dict[str, Any], arm: str, where: str) -> None:
    policy, reason = expected_policy(arm)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "decode_policy": policy,
        "greedy_gpu_selection_reason": reason,
        "stop_reason": "token_limit",
        "terminal_token_target_transition_consumed": False,
    }
    for key, value in expected.items():
        require(row.get(key) == value, f"{where} {key}: {row.get(key)!r} != {value!r}")


def run_charged_command(
    stem: str, argv: list[str], *, env: dict[str, str], qwen_process: bool
) -> subprocess.CompletedProcess[str]:
    host_snapshot(f"{stem}-pre")
    try:
        result = archive_command(
            stem,
            argv,
            env=env,
            timeout=1_200,
            record_child_rusage=qwen_process,
        )
    finally:
        host_snapshot(f"{stem}-post")
    require(result.returncode == 0, f"{stem} exited {result.returncode}")
    return result


def validate_conformance_row(
    row: dict[str, Any], arm: str, commit: str, build: dict[str, Any]
) -> None:
    validate_policy(row, arm, "conformance")
    expected = {
        "request_epoch": "first_post_model_load",
        "request_index": 0,
        "prefix_cache_used": False,
        "build_commit": commit,
        "build_dirty": "0",
        "build_source_state": build["build_source_state"],
        "model": str(MODEL["path"]),
        "runtime_identity_kind": MODEL["runtime_identity_kind"],
        "runtime_model_id": MODEL["runtime_model_id"],
        "runtime_tokenizer_id": MODEL["runtime_tokenizer_id"],
        "prompt_source": "file",
        "prompt_bytes": PROMPT_BYTES,
        "prompt_tokens": PROMPT_TOKENS,
        "requested_tokens": 16,
        "generated_tokens": 16,
        "prefill_chunk_requested": 1024,
        "prefill_chunk_effective": PROMPT_TOKENS,
        "max_context_tokens": 1024,
        "transition_count": 15,
    }
    for key, value in expected.items():
        require(
            row.get(key) == value, f"conformance {key}: {row.get(key)!r} != {value!r}"
        )
    require(
        "sampling" not in row and "prompt_lookup" not in row,
        "conformance used sampling/prompt lookup",
    )
    validate_digest(row, "conformance")
    for key in ("prefill_ms", "generation_ms", "ttft_ms", "transition_ms"):
        finite(row, key, "conformance")


def run_conformance(commit: str, build: dict[str, Any]) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for arm in ("A", "B"):
        stem = f"a3b-conformance-{arm.lower()}"
        timing_path = PACKET / f"{stem}.timing.jsonl"
        result = run_charged_command(
            stem,
            [
                str(QWEN),
                "--model",
                str(MODEL["path"]),
                "--prompt-file",
                str(PROMPT),
                "--tokens",
                "16",
                "--temp",
                "0",
                "--prefill-chunk",
                "1024",
                "--max-context-tokens",
                "1024",
                "--prefix-cache-max-mib",
                "0",
                "--cache-prefix-auto-min-tokens",
                "0",
                "--request-timings",
                str(timing_path),
            ],
            env=normalized_environment(arm),
            qwen_process=True,
        )
        rows = parse_jsonl(timing_path)
        require(len(rows) == 1, f"{stem} emitted {len(rows)} timing rows")
        validate_conformance_row(rows[0], arm, commit, build)
        observations[arm] = {"stdout": result.stdout, "timing": rows[0]}
        time.sleep(COOLDOWN_SECONDS)
    require(
        observations["A"]["stdout"] == observations["B"]["stdout"],
        "conformance stdout differs",
    )
    require(
        observations["A"]["timing"]["generated_token_sha256"]
        == observations["B"]["timing"]["generated_token_sha256"],
        "conformance token digest differs",
    )
    write_json(PACKET / "a3b-conformance.json", observations)
    return observations


def make_requests() -> Path:
    path = PACKET / "requests.jsonl"
    rows = [
        {
            "id": "treatment",
            "prompt_file": str(PROMPT),
            "tokens": 128,
            "cache_prefix_tokens": 0,
        },
        {
            "id": "sentinel",
            "prompt_file": str(PROMPT),
            "tokens": 1,
            "cache_prefix_tokens": 0,
        },
    ]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    return path


def validate_jsonl_rows(
    stats: list[dict[str, Any]], outputs: list[dict[str, Any]], arm: str
) -> dict[str, tuple[str, str]]:
    require(len(stats) == 2 and len(outputs) == 2, "wrong JSONL row count")
    ids = ["treatment", "sentinel"]
    require([row.get("id") for row in stats] == ids, "stats IDs differ")
    require([row.get("id") for row in outputs] == ids, "output IDs differ")
    identities: dict[str, tuple[str, str]] = {}
    prompt_hash: str | None = None
    for index, (stats_row, output_row) in enumerate(zip(stats, outputs, strict=True)):
        request_id = ids[index]
        requested = 128 if request_id == "treatment" else 1
        transitions = requested - 1
        where = f"{arm} {request_id}"
        validate_policy(stats_row, arm, where)
        for forbidden in (
            "runtime_identity_kind",
            "runtime_model_id",
            "runtime_tokenizer_id",
        ):
            require(
                forbidden not in stats_row,
                f"{where} contains removed runtime identity field {forbidden}",
            )
        expected = {
            "line": index + 1,
            "model": str(MODEL["path"]),
            "prompt_tokens": PROMPT_TOKENS,
            "requested_tokens": requested,
            "generated_tokens": requested,
            "decode_transitions": transitions,
            "prefill_chunk": 1024,
            "max_context_tokens": 1024,
            "no_special_tokens": False,
            "cache_prefix_tokens": None,
            "cache_prefix_source": "request_disabled",
            "cache_prefix_hash": None,
            "auto_cache_prefix_tokens": None,
            "auto_cache_future_hits": 0,
            "cache_hit": False,
            "matched_prefix_tokens": 0,
            "matched_prefix_hash": None,
            "exact_cache_hit": False,
            "cache_entries": 0,
            "cache_bytes": 0,
            "cache_max_bytes": 0,
            "prefix_inserted_bytes": 0,
            "prefix_insert_ms": 0.0,
        }
        for key, value in expected.items():
            require(
                stats_row.get(key) == value,
                f"{where} {key}: {stats_row.get(key)!r} != {value!r}",
            )
        require(
            "sampling" not in stats_row and "prompt_lookup" not in stats_row,
            f"{where} used sampling/prompt lookup",
        )
        require(
            finite(stats_row, "restore_ms", where, positive=False) >= 0.0,
            "invalid restore timing",
        )
        require(
            finite(stats_row, "prefix_insert_ms", where, positive=False) >= 0.0,
            "invalid insert timing",
        )
        for key in ("prefill_ms", "decode_ms", "model_ttft_ms", "total_ms"):
            finite(stats_row, key, where)
        transition_ms = finite(
            stats_row, "transition_ms", where, positive=request_id == "treatment"
        )
        transition_tps = finite(
            stats_row, "transition_tps", where, positive=request_id == "treatment"
        )
        if request_id == "sentinel":
            require(transition_ms == 0.0, "sentinel transition wall is not zero")
            require(transition_tps == 0.0, "sentinel transition tps is not zero")
        stats_digest = validate_digest(stats_row, f"{where} stats")
        output_expected = {
            "id": request_id,
            "prompt_tokens": PROMPT_TOKENS,
            "generated_tokens": requested,
            "stop_reason": "token_limit",
            "terminal_token_target_transition_consumed": False,
        }
        for key, value in output_expected.items():
            require(output_row.get(key) == value, f"{where} output {key} differs")
        output_digest = validate_digest(output_row, f"{where} output")
        require(stats_digest == output_digest, f"{where} stats/output digest differs")
        text = output_row.get("generated_text")
        require(isinstance(text, str), f"{where} generated text is not a string")
        identities[request_id] = (output_digest, text)
        current_prompt_hash = stats_row.get("prompt_hash")
        require(
            isinstance(current_prompt_hash, str) and current_prompt_hash,
            f"{where} invalid prompt hash",
        )
        if prompt_hash is None:
            prompt_hash = current_prompt_hash
        else:
            require(
                current_prompt_hash == prompt_hash,
                "identical prompts have different token hashes",
            )
    return identities


def session_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    treatment, sentinel = rows
    treatment_prefill = float(treatment["prefill_ms"])
    sentinel_prefill = float(sentinel["prefill_ms"])
    metrics = {
        "carry": sentinel_prefill / treatment_prefill,
        "treatment_prefill": treatment_prefill,
        "sentinel_prefill": sentinel_prefill,
        "treatment_decode_per_token": float(treatment["decode_ms"]) / 128.0,
    }
    require(
        all(math.isfinite(value) and value > 0.0 for value in metrics.values()),
        "invalid session metric",
    )
    return metrics


def run_session(
    arm: str, pair_index: int, order_index: int, requests: Path
) -> dict[str, Any]:
    stem = f"a3b-pair-{pair_index:02d}-{order_index}-{arm.lower()}"
    stats_path = PACKET / f"{stem}.stats.jsonl"
    result = run_charged_command(
        stem,
        [
            str(QWEN),
            "--model",
            str(MODEL["path"]),
            "--requests-jsonl",
            str(requests),
            "--request-stats",
            str(stats_path),
            "--tokens",
            "128",
            "--temp",
            "0",
            "--prefill-chunk",
            "1024",
            "--max-context-tokens",
            "1024",
            "--prefix-cache-max-mib",
            "0",
            "--cache-prefix-auto-min-tokens",
            "0",
        ],
        env=normalized_environment(arm),
        qwen_process=True,
    )
    stats = parse_jsonl(stats_path)
    outputs = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    identities = validate_jsonl_rows(stats, outputs, arm)
    session = {
        "arm": arm,
        "metrics": session_metrics(stats),
        "output_identities": {
            key: {"generated_token_sha256": value[0], "generated_text": value[1]}
            for key, value in identities.items()
        },
        "outputs": outputs,
        "stats": stats,
    }
    write_json(PACKET / f"{stem}.summary.json", session)
    time.sleep(COOLDOWN_SECONDS)
    return session


def carryover_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(pairs) == 16, "summary requires exactly 16 pairs")
    d_logs = [
        math.log(pair["B"]["metrics"]["carry"] / pair["A"]["metrics"]["carry"])
        for pair in pairs
    ]
    q_logs = [
        math.log(
            pair["A"]["metrics"]["treatment_prefill"]
            / pair["B"]["metrics"]["treatment_prefill"]
        )
        for pair in pairs
    ]
    mean_d = statistics.mean(d_logs)
    sd_d = statistics.stdev(d_logs)
    half_width = T95_ONE_SIDED_N16 * sd_d / math.sqrt(16)
    estimate = math.exp(mean_d)
    lower = math.exp(mean_d - half_width)
    upper = math.exp(mean_d + half_width)
    baseline = math.exp(statistics.mean(q_logs))
    require(
        all(
            math.isfinite(value) and value > 0.0
            for value in (estimate, lower, upper, baseline)
        )
        and math.isfinite(sd_d)
        and sd_d >= 0.0,
        "non-finite carryover statistic",
    )
    return {
        "n": 16,
        "carry": {
            "estimate": estimate,
            "lower_95_one_sided": lower,
            "upper_95_one_sided": upper,
            "log_stddev": sd_d,
            "log_half_width": half_width,
            "critical": T95_ONE_SIDED_N16,
            "pair_ratios": [math.exp(value) for value in d_logs],
            "pair_logs": d_logs,
        },
        "baseline": {
            "geometric_ratio": baseline,
            "pair_ratios": [math.exp(value) for value in q_logs],
            "pair_logs": q_logs,
        },
    }


def acquire(requests: Path) -> dict[str, Any]:
    pairs = []
    canonical: dict[str, tuple[str, str]] = {}
    canonical_prompt_hash: str | None = None
    for pair_index, order in enumerate(PAIR_ORDERS):
        sessions = {}
        for order_index, arm in enumerate(order):
            session = run_session(arm, pair_index, order_index, requests)
            for request_id, identity in session["output_identities"].items():
                value = (identity["generated_token_sha256"], identity["generated_text"])
                if request_id not in canonical:
                    canonical[request_id] = value
                else:
                    require(
                        value == canonical[request_id],
                        f"cross-session {request_id} output changed",
                    )
            for row in session["stats"]:
                if canonical_prompt_hash is None:
                    canonical_prompt_hash = row["prompt_hash"]
                else:
                    require(
                        row["prompt_hash"] == canonical_prompt_hash,
                        "cross-session prompt token hash changed",
                    )
            sessions[arm] = session
        pairs.append({"index": pair_index, "order": order, **sessions})
    summary = carryover_summary(pairs)
    result = {
        "model": "a3b",
        "pairs_completed": len(pairs),
        "arm_processes_completed": len(pairs) * 2,
        "pair_orders": PAIR_ORDERS,
        "request_output_identities": {
            key: {"generated_token_sha256": value[0], "generated_text": value[1]}
            for key, value in canonical.items()
        },
        "prompt_token_hash": canonical_prompt_hash,
        "summary": summary,
        "pairs": pairs,
    }
    write_json(PACKET / "a3b-result.json", result)
    return result


def decide(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    carry = summary["carry"]
    baseline = summary["baseline"]["geometric_ratio"]
    if not 0.97 <= baseline <= 1.03:
        verdict = "INCONCLUSIVE_BASELINE_IMBALANCE"
    elif carry["lower_95_one_sided"] > 1.03:
        verdict = "CARRYOVER_HARM_SIGNAL"
    elif carry["upper_95_one_sided"] < 1.03:
        verdict = "NO_MATERIAL_CARRYOVER_AT_3_PERCENT"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "schema_version": 1,
        "verdict": verdict,
        "authority": "diagnostic-only",
        "product_admission": "forbidden",
        "result": {"pairs_completed": result["pairs_completed"], "summary": summary},
        "one_shot_consumed": True,
    }


def write_inventory() -> str:
    entries = []
    for path in sorted(PACKET.rglob("*")):
        if path.is_file() and path.name not in {
            "artifact-inventory.sha256",
            "packet-complete.json",
        }:
            entries.append(f"{sha256_file(path)}  {path.relative_to(PACKET)}")
    inventory = PACKET / "artifact-inventory.sha256"
    inventory.write_text("\n".join(entries) + "\n")
    return sha256_file(inventory)


def final_identity_check(manifest: dict[str, Any]) -> dict[str, Any]:
    commit, build = source_and_build_identity()
    require(commit == manifest["source_commit"], "source commit changed during packet")
    require(build == manifest["build_identity"], "build identity changed during packet")
    hashes = {
        "qwen": sha256_file(QWEN),
        "qwen_bench": sha256_file(QWEN_BENCH),
        "script": sha256_file(SCRIPT),
        "prereg": sha256_file(PREREG),
        "perf_log": sha256_file(PERF_LOG),
        "perf_roadmap": sha256_file(PERF_ROADMAP),
        "prompt": sha256_file(PROMPT),
        **{
            f"implementation_source:{relative(path)}": sha256_file(path)
            for path in IMPLEMENTATION_SOURCE
        },
    }
    require(
        hashes == manifest["small_file_sha256"],
        "binary/source/script/docs/prompt identity changed",
    )
    identity = file_identity(MODEL["path"])
    require(identity == manifest["model_file_identity"], "model file identity changed")
    result = {
        "source_commit": commit,
        "build_identity": build,
        "small_file_sha256": hashes,
        "model_sha256": manifest["model_sha256"],
        "model_file_identity": identity,
    }
    write_json(PACKET / "final-identity.json", result)
    return result


def seal_invalid(error: BaseException) -> None:
    if not PACKET.exists():
        return
    failure = {
        "schema_version": 1,
        "error": f"{type(error).__name__}: {error}",
        "one_shot_consumed": True,
    }
    write_json(PACKET / "failure.json", failure)
    decision = {
        "schema_version": 1,
        "verdict": "INVALID",
        "authority": "none",
        "product_admission": "forbidden",
        "error": failure["error"],
        "one_shot_consumed": True,
    }
    write_json(PACKET / "decision.json", decision)
    inventory_sha256 = write_inventory()
    write_json(
        PACKET / "packet-complete.json",
        {
            "schema_version": 1,
            "verdict": "INVALID",
            "one_shot_consumed": True,
            "decision_sha256": sha256_file(PACKET / "decision.json"),
            "inventory_sha256": inventory_sha256,
        },
    )


def execute() -> None:
    preflight_host, inherited_qwen = execution_source_preflight()
    PACKET.mkdir(parents=True)
    try:
        write_json(PACKET / "execution-preflight.host.json", preflight_host)
        write_json(PACKET / "inherited-qwen-environment.json", inherited_qwen)
        before = file_identity(MODEL["path"])
        model_hash = sha256_file(MODEL["path"])
        after = file_identity(MODEL["path"])
        require(before == after, "model changed while hashing")
        require(model_hash == MODEL["sha256"], "model hash changed")

        build_result = archive_command(
            "build",
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "qwen-cli",
                "--bin",
                "qwen",
                "--bin",
                "qwen-bench",
            ],
            env=normalized_environment(),
            timeout=1_200,
        )
        require(build_result.returncode == 0, f"build exited {build_result.returncode}")
        commit, build = source_and_build_identity()
        hashes = {
            "qwen": sha256_file(QWEN),
            "qwen_bench": sha256_file(QWEN_BENCH),
            "script": sha256_file(SCRIPT),
            "prereg": sha256_file(PREREG),
            "perf_log": sha256_file(PERF_LOG),
            "perf_roadmap": sha256_file(PERF_ROADMAP),
            "prompt": sha256_file(PROMPT),
            **{
                f"implementation_source:{relative(path)}": sha256_file(path)
                for path in IMPLEMENTATION_SOURCE
            },
        }
        manifest = {
            "schema_version": 1,
            "source_commit": commit,
            "prereg_commit": PREREG_COMMIT,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "result_doc_commit": RESULT_DOC_COMMIT,
            "build_identity": build,
            "small_file_sha256": hashes,
            "model_sha256": model_hash,
            "model_file_identity": after,
            "runtime_identity": {
                "kind": MODEL["runtime_identity_kind"],
                "model_id": MODEL["runtime_model_id"],
                "tokenizer_id": MODEL["runtime_tokenizer_id"],
            },
            "authority": "diagnostic-only",
            "product_admission": "forbidden",
            "pair_orders": PAIR_ORDERS,
            "pairs": 16,
            "arm_processes": 32,
            "requests_per_process": 2,
            "cooldown_seconds": COOLDOWN_SECONDS,
            "inherited_qwen_environment_artifact": "inherited-qwen-environment.json",
            "normalized_pins": PINNED_QWEN_ENV,
            "arm_policy_env": {
                "A": "QWEN_GREEDY_GPU_ARGMAX=0",
                "B": "QWEN_GREEDY_GPU_ARGMAX=1",
            },
        }
        write_json(PACKET / "manifest.json", manifest)
        host_snapshot("packet-start")

        exact = run_charged_command(
            "exact-state-gate",
            [
                "cargo",
                "test",
                "--release",
                "-p",
                "qwen-llm",
                "metal_exact_greedy_chain_matches_full_logits_moe",
                "--",
                "--nocapture",
                "--test-threads=1",
            ],
            env=normalized_environment(),
            qwen_process=False,
        )
        exact_output = exact.stdout + exact.stderr
        marker = "[greedy-chain-moe-a3b] exact-state PASS"
        require(marker in exact_output, f"missing exact-state marker: {marker}")
        require("skipped" not in exact_output.lower(), "exact-state test skipped")

        conformance = run_conformance(commit, build)
        requests = make_requests()
        result = acquire(requests)
        host_snapshot("packet-end")
        final_identity_check(manifest)
        decision = decide(result)
        decision["exact_state_gate"] = {"passed": True, "marker": marker}
        decision["conformance"] = {"passed": True, "arms": sorted(conformance)}
        write_json(PACKET / "decision.json", decision)
        inventory_sha256 = write_inventory()
        write_json(
            PACKET / "packet-complete.json",
            {
                "schema_version": 1,
                "verdict": decision["verdict"],
                "authority": "diagnostic-only",
                "product_admission": "forbidden",
                "one_shot_consumed": True,
                "decision_sha256": sha256_file(PACKET / "decision.json"),
                "inventory_sha256": inventory_sha256,
            },
        )
        print(json.dumps(decision, indent=2, sort_keys=True))
    except BaseException as error:
        seal_invalid(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        static_check()
        print("v0.650 static check: PASS")
        return
    execute()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"v0.650 failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
