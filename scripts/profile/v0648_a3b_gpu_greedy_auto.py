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
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PREREG = ROOT / "docs/bench/v0648-a3b-gpu-greedy-auto.md"
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
QWEN = ROOT / "target/release/qwen"
QWEN_BENCH = ROOT / "target/release/qwen-bench"
PACKET = ROOT / "target/profiles/v0648-a3b-gpu-greedy-auto-p1"
INITIAL_PREREG_COMMIT = "d007d043779609784685b34a769747a74682f6c9"
FINAL_PREREG_COMMIT = "18ac0eb6560a07d768c6b7ec77c0c52ae2d6c022"
IMPLEMENTATION_COMMIT = "43af9d0f2effd0f86062391a6fddac97bbf46d4a"

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
OUTPUT_TOKENS = 128
TRANSITIONS = 127
STEADY_REQUESTS = 4
PAIR_ORDERS = ("AB", "BA") * 4
COOLDOWN_SECONDS = 5.0
T95_ONE_SIDED_N8 = 1.894579
SCHEMA_VERSION = 7
PINNED_QWEN_ENV = {
    "QWEN_DECODE_GDN_FUSED_BETA_PROJ": "1",
    "QWEN_DECODE_ROPE_PAIR": "1",
    "QWEN_DECODE_MOE_GROUPED_FINALIZER": "1",
    "QWEN_DECODE_MOE_FUSED_FINALIZER": "1",
}
AUTHORITY = {
    "identity_kind": "metadata_compatibility_v1",
    "model_id": MODEL["runtime_model_id"],
    "tokenizer_id": MODEL["runtime_tokenizer_id"],
    "request_scope": "temperature-zero serial generation without prompt lookup",
    "implementation_commit": IMPLEMENTATION_COMMIT,
    "rollback": "QWEN_GREEDY_GPU_ARGMAX=0",
}
TOKEN_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def archive_command(
    stem: str,
    argv: list[str],
    *,
    env: dict[str, str],
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    started_ns = time.time_ns()
    try:
        result = command(argv, env=env, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        (PACKET / f"{stem}.stdout.txt").write_text(stdout)
        (PACKET / f"{stem}.stderr.txt").write_text(stderr)
        write_json(
            PACKET / f"{stem}.command.json",
            {
                "argv": argv,
                "elapsed_ms": (time.time_ns() - started_ns) / 1e6,
                "returncode": None,
                "timed_out": True,
                "timeout_seconds": timeout,
            },
        )
        raise RuntimeError(f"{stem} timed out after {timeout} seconds") from error
    (PACKET / f"{stem}.stdout.txt").write_text(result.stdout)
    (PACKET / f"{stem}.stderr.txt").write_text(result.stderr)
    write_json(
        PACKET / f"{stem}.command.json",
        {
            "argv": argv,
            "elapsed_ms": (time.time_ns() - started_ns) / 1e6,
            "returncode": result.returncode,
            "timed_out": False,
            "timeout_seconds": timeout,
        },
    )
    require(result.returncode == 0, f"{stem} exited {result.returncode}")
    return result


def normalized_environment(arm: str = "A") -> dict[str, str]:
    require(arm in {"A", "B"}, f"unknown arm: {arm}")
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("QWEN_")
    }
    env["NO_COLOR"] = "1"
    env.update(PINNED_QWEN_ENV)
    if arm == "A":
        env["QWEN_GREEDY_GPU_ARGMAX"] = "0"
    return env


def file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def assert_implementation_unchanged(*, runner_must_be_committed: bool = False) -> None:
    command(["git", "merge-base", "--is-ancestor", IMPLEMENTATION_COMMIT, "HEAD"])
    script_relative = str(SCRIPT.relative_to(ROOT))
    changed = command_text(
        ["git", "diff", "--name-only", f"{IMPLEMENTATION_COMMIT}..HEAD"]
    ).splitlines()
    allowed = [script_relative]
    if runner_must_be_committed:
        require(
            changed == allowed, f"post-implementation delta is {changed}, not {allowed}"
        )
    else:
        require(
            all(path == script_relative for path in changed),
            f"implementation changed after {IMPLEMENTATION_COMMIT}: {changed}",
        )
    prereg_relative = str(PREREG.relative_to(ROOT))
    expected_prereg_blob = command_text(
        ["git", "rev-parse", f"{FINAL_PREREG_COMMIT}:{prereg_relative}"]
    )
    actual_prereg_blob = command_text(["git", "hash-object", str(PREREG)])
    require(
        actual_prereg_blob == expected_prereg_blob,
        "preregistration changed after its frozen correction commit",
    )


def static_check() -> None:
    for path in (SCRIPT, PREREG, PROMPT, MODEL["path"]):
        require(path.exists(), f"missing required path: {path}")
    require(PROMPT.stat().st_size == PROMPT_BYTES, "prompt byte count changed")
    require(sha256_file(PROMPT) == PROMPT_SHA256, "prompt hash changed")
    assert_implementation_unchanged()
    require(PAIR_ORDERS == ("AB", "BA") * 4, "pair order drifted")
    require(len(PAIR_ORDERS) == 8, "sample count drifted")
    require(STEADY_REQUESTS == 4 and OUTPUT_TOKENS == 128, "request contract drifted")
    env_a = normalized_environment("A")
    env_b = normalized_environment("B")
    differences = {
        key for key in env_a.keys() | env_b.keys() if env_a.get(key) != env_b.get(key)
    }
    require(differences == {"QWEN_GREEDY_GPU_ARGMAX"}, "arm environments drifted")
    require(env_a["QWEN_GREEDY_GPU_ARGMAX"] == "0", "A is not rollback")
    require("QWEN_GREEDY_GPU_ARGMAX" not in env_b, "B is not automatic mode")
    require(
        all(env_a[key] == "1" for key in PINNED_QWEN_ENV), "pinned environment drifted"
    )
    synthetic = [
        {"A": {"metrics": {"decode": 1.01}}, "B": {"metrics": {"decode": 1.0}}}
        for _ in PAIR_ORDERS
    ]
    require(
        paired_summary(synthetic, "decode")["n"] == 8,
        "paired statistic self-test failed",
    )


def collect_host_snapshot() -> dict[str, Any]:
    memory = command(["memory_pressure", "-Q"])
    thermal = command(["pmset", "-g", "therm"])
    processes = command(["ps", "-axo", "pid=,comm=,args="])
    free_line = next(
        (
            line
            for line in memory.stdout.splitlines()
            if line.startswith("System-wide memory free percentage:")
        ),
        "",
    )
    require(free_line, "memory pressure output omitted free percentage")
    free_percent = int(free_line.rsplit(" ", 1)[-1].rstrip("%"))
    names = {
        "qwen",
        "qwen-bench",
        "llama-cli",
        "llama-server",
        "llama-bench",
        "mlx_lm",
        "ollama",
    }
    competitors = []
    for line in processes.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) < 2:
            continue
        executable = Path(fields[1]).name
        arguments = fields[2] if len(fields) == 3 else ""
        try:
            argument_tokens = shlex.split(arguments)
        except ValueError:
            argument_tokens = arguments.split()
        argument_names = {Path(token).name for token in argument_tokens}
        module_inference = any(
            token == "mlx_lm" or token.startswith("mlx_lm.")
            for token in argument_tokens
        )
        if (
            executable in names
            or argument_names.intersection(names)
            or module_inference
        ):
            competitors.append(line.strip())
    require(free_percent >= 85, f"memory free percentage too low: {free_percent}")
    require(not competitors, f"competing inference processes: {competitors}")
    require(
        "No thermal warning level has been recorded" in thermal.stdout,
        "thermal warning present",
    )
    require(
        "No performance warning level has been recorded" in thermal.stdout,
        "performance warning present",
    )
    return {
        "free_percent": free_percent,
        "competitors": competitors,
        "processes": processes.stdout,
        "memory_pressure": memory.stdout,
        "thermal": thermal.stdout,
    }


def host_snapshot(stem: str) -> dict[str, Any]:
    snapshot = collect_host_snapshot()
    write_json(PACKET / f"{stem}.host.json", snapshot)
    return snapshot


def execution_source_preflight() -> tuple[dict[str, Any], dict[str, str]]:
    require(not PACKET.exists(), f"packet already exists: {PACKET}")
    static_check()
    dirty = command_text(["git", "status", "--porcelain=v1"])
    require(not dirty, f"source is dirty: {dirty!r}")
    for path in (SCRIPT, PREREG, PROMPT):
        command(["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))])
    assert_implementation_unchanged(runner_must_be_committed=True)
    require(PACKET.parent.is_dir(), f"packet parent does not exist: {PACKET.parent}")
    inherited = {
        key: value for key, value in os.environ.items() if key.startswith("QWEN_")
    }
    return collect_host_snapshot(), inherited


def source_and_build_identity() -> tuple[str, dict[str, Any]]:
    dirty = command_text(["git", "status", "--porcelain=v1"])
    require(not dirty, f"source is dirty: {dirty!r}")
    assert_implementation_unchanged(runner_must_be_committed=True)
    for path in (SCRIPT, PREREG, PROMPT):
        command(["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))])
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
    return "greedy_gpu_argmax", "auto_metadata_a3b_v1"


def validate_digest(row: dict[str, Any], where: str) -> str:
    digest = row.get("generated_token_sha256")
    require(
        isinstance(digest, str) and TOKEN_SHA256_RE.fullmatch(digest) is not None,
        f"invalid token digest in {where}",
    )
    return digest


def validate_identity_policy(row: dict[str, Any], arm: str, where: str) -> None:
    policy, reason = expected_policy(arm)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "runtime_identity_kind": MODEL["runtime_identity_kind"],
        "runtime_model_id": MODEL["runtime_model_id"],
        "runtime_tokenizer_id": MODEL["runtime_tokenizer_id"],
        "decode_policy": policy,
        "greedy_gpu_selection_reason": reason,
        "stop_reason": "token_limit",
        "terminal_token_target_transition_consumed": False,
    }
    for key, value in expected.items():
        require(row.get(key) == value, f"{where} {key}: {row.get(key)!r} != {value!r}")


def validate_timing_row(
    row: dict[str, Any], arm: str, commit: str, build: dict[str, Any]
) -> None:
    validate_identity_policy(row, arm, "conformance timing")
    expected = {
        "request_epoch": "first_post_model_load",
        "request_index": 0,
        "prefix_cache_used": False,
        "build_commit": commit,
        "build_dirty": "0",
        "build_source_state": build["build_source_state"],
        "model": str(MODEL["path"]),
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
    validate_digest(row, "conformance timing")
    require(
        "sampling" not in row and "prompt_lookup" not in row,
        "conformance used sampling/prompt lookup",
    )


def run_conformance(commit: str, build: dict[str, Any]) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for arm in ("A", "B"):
        stem = f"a3b-conformance-{arm.lower()}"
        host_snapshot(f"{stem}-pre")
        timing_path = PACKET / f"{stem}.timing.jsonl"
        argv = [
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
        ]
        result = archive_command(stem, argv, env=normalized_environment(arm))
        rows = parse_jsonl(timing_path)
        require(len(rows) == 1, f"{stem} emitted {len(rows)} timing rows")
        validate_timing_row(rows[0], arm, commit, build)
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
    rows = []
    for index in range(STEADY_REQUESTS + 1):
        rows.append(
            json.dumps(
                {
                    "id": "first-use" if index == 0 else f"steady-{index}",
                    "prompt_file": str(PROMPT),
                    "tokens": OUTPUT_TOKENS,
                    "cache_prefix_tokens": 0,
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(rows) + "\n")
    return path


def validate_jsonl_rows(
    stats: list[dict[str, Any]], outputs: list[dict[str, Any]], arm: str
) -> None:
    require(len(stats) == 5 and len(outputs) == 5, "wrong JSONL row count")
    ids = ["first-use", *(f"steady-{index}" for index in range(1, 5))]
    require([row.get("id") for row in stats] == ids, "stats IDs differ")
    require([row.get("id") for row in outputs] == ids, "output IDs differ")
    canonical: tuple[str, str] | None = None
    prompt_hashes = set()
    for index, (stats_row, output_row) in enumerate(zip(stats, outputs, strict=True)):
        where = f"request {index}"
        validate_identity_policy(stats_row, arm, where)
        expected = {
            "model": str(MODEL["path"]),
            "prompt_tokens": PROMPT_TOKENS,
            "requested_tokens": OUTPUT_TOKENS,
            "generated_tokens": OUTPUT_TOKENS,
            "decode_transitions": TRANSITIONS,
            "prefill_chunk": 1024,
            "max_context_tokens": 1024,
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
        for key in ("transition_ms", "decode_ms", "model_ttft_ms", "prefill_ms"):
            value = float(stats_row[key])
            require(
                math.isfinite(value) and value > 0.0, f"invalid {where} {key}: {value}"
            )
        stats_digest = validate_digest(stats_row, f"{where} stats")
        require(
            output_row.get("prompt_tokens") == PROMPT_TOKENS,
            f"wrong {where} output prompt count",
        )
        require(
            output_row.get("generated_tokens") == OUTPUT_TOKENS,
            f"wrong {where} output token count",
        )
        require(
            output_row.get("stop_reason") == "token_limit", f"wrong {where} output stop"
        )
        require(
            output_row.get("terminal_token_target_transition_consumed") is False,
            f"wrong {where} output terminal transition",
        )
        output_digest = validate_digest(output_row, f"{where} output")
        require(stats_digest == output_digest, f"{where} stats/output digest differs")
        text = output_row.get("generated_text")
        require(isinstance(text, str), f"{where} generated text is not a string")
        identity = (output_digest, text)
        if canonical is None:
            canonical = identity
        else:
            require(
                identity == canonical, "identical requests generated different output"
            )
        prompt_hashes.add(stats_row.get("prompt_hash"))
    require(len(prompt_hashes) == 1, "prompt token hash changed within session")


def session_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    steady = rows[1:]
    require(len(steady) == STEADY_REQUESTS, "wrong steady request count")
    return {
        "decode": statistics.median(
            float(row["decode_ms"]) / int(row["generated_tokens"]) for row in steady
        ),
        "ttft": statistics.median(float(row["model_ttft_ms"]) for row in steady),
        "prefill": statistics.median(float(row["prefill_ms"]) for row in steady),
        "transition": statistics.median(
            float(row["transition_ms"]) / int(row["decode_transitions"])
            for row in steady
        ),
        "first_use_decode": float(rows[0]["decode_ms"])
        / int(rows[0]["generated_tokens"]),
    }


def run_session(
    arm: str, pair_index: int, order_index: int, requests: Path
) -> dict[str, Any]:
    stem = f"a3b-pair-{pair_index:02d}-{order_index}-{arm.lower()}"
    host_snapshot(f"{stem}-pre")
    stats_path = PACKET / f"{stem}.stats.jsonl"
    argv = [
        str(QWEN),
        "--model",
        str(MODEL["path"]),
        "--requests-jsonl",
        str(requests),
        "--request-stats",
        str(stats_path),
        "--tokens",
        str(OUTPUT_TOKENS),
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
    ]
    result = archive_command(stem, argv, env=normalized_environment(arm))
    stats = parse_jsonl(stats_path)
    outputs = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    validate_jsonl_rows(stats, outputs, arm)
    session = {
        "arm": arm,
        "metrics": session_metrics(stats),
        "outputs": outputs,
        "stats": stats,
    }
    write_json(PACKET / f"{stem}.summary.json", session)
    time.sleep(COOLDOWN_SECONDS)
    return session


def paired_summary(pairs: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    require(len(pairs) == 8, "paired summary requires exactly eight pairs")
    logs = [
        math.log(pair["A"]["metrics"][metric] / pair["B"]["metrics"][metric])
        for pair in pairs
    ]
    mean = statistics.mean(logs)
    stddev = statistics.stdev(logs)
    half_width = T95_ONE_SIDED_N8 * stddev / math.sqrt(8)
    return {
        "n": 8,
        "estimate": math.exp(mean),
        "lower_95_one_sided": math.exp(mean - half_width),
        "upper_95_one_sided": math.exp(mean + half_width),
        "log_stddev": stddev,
        "log_half_width": half_width,
        "critical": T95_ONE_SIDED_N8,
        "pair_ratios": [math.exp(value) for value in logs],
    }


def acquire(requests: Path) -> dict[str, Any]:
    pairs = []
    canonical: tuple[str, str] | None = None
    for pair_index, order in enumerate(PAIR_ORDERS):
        sessions = {}
        for order_index, arm in enumerate(order):
            session = run_session(arm, pair_index, order_index, requests)
            identities = [
                (row["generated_token_sha256"], row["generated_text"])
                for row in session["outputs"]
            ]
            require(len(set(identities)) == 1, "session output changed")
            if canonical is None:
                canonical = identities[0]
            else:
                require(identities[0] == canonical, "cross-session output changed")
            sessions[arm] = session
        pairs.append({"index": pair_index, "order": order, **sessions})
    summaries = {
        metric: paired_summary(pairs, metric)
        for metric in ("decode", "first_use_decode", "ttft", "prefill", "transition")
    }
    result = {
        "model": "a3b",
        "pairs_completed": len(pairs),
        "pair_orders": PAIR_ORDERS,
        "summaries": summaries,
        "pairs": pairs,
    }
    write_json(PACKET / "a3b-result.json", result)
    return result


def decide(result: dict[str, Any]) -> dict[str, Any]:
    summaries = result["summaries"]
    if not 0.97 <= summaries["prefill"]["estimate"] <= 1.03:
        verdict = "INCONCLUSIVE_CONTAMINATION"
    elif (
        summaries["first_use_decode"]["lower_95_one_sided"] < 0.97
        or summaries["ttft"]["lower_95_one_sided"] < 0.97
    ):
        verdict = "KEEP_DEFAULT_OFF_REGRESSION"
    elif (
        summaries["decode"]["estimate"] < 1.01
        or summaries["decode"]["lower_95_one_sided"] <= 1.0
    ):
        verdict = "KEEP_DEFAULT_OFF_EFFECT_MISS"
    else:
        verdict = "ADMIT_A3B_AUTO_METADATA_V1"
    return {
        "schema_version": 1,
        "verdict": verdict,
        "authority": AUTHORITY if verdict == "ADMIT_A3B_AUTO_METADATA_V1" else "none",
        "results": {
            "a3b": {
                "pairs_completed": result["pairs_completed"],
                "summaries": summaries,
            }
        },
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
        "prompt": sha256_file(PROMPT),
    }
    require(
        hashes == manifest["small_file_sha256"],
        "binary/script/prereg/prompt identity changed",
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

        archive_command(
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
        commit, build = source_and_build_identity()
        hashes = {
            "qwen": sha256_file(QWEN),
            "qwen_bench": sha256_file(QWEN_BENCH),
            "script": sha256_file(SCRIPT),
            "prereg": sha256_file(PREREG),
            "prompt": sha256_file(PROMPT),
        }
        manifest = {
            "schema_version": 1,
            "source_commit": commit,
            "initial_prereg_commit": INITIAL_PREREG_COMMIT,
            "final_prereg_commit": FINAL_PREREG_COMMIT,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "build_identity": build,
            "small_file_sha256": hashes,
            "model_sha256": model_hash,
            "model_file_identity": after,
            "runtime_identity": {
                "kind": MODEL["runtime_identity_kind"],
                "model_id": MODEL["runtime_model_id"],
                "tokenizer_id": MODEL["runtime_tokenizer_id"],
            },
            "authority_on_admission": AUTHORITY,
            "pair_orders": PAIR_ORDERS,
            "pairs": 8,
            "first_use_requests_per_session": 1,
            "steady_requests_per_session": 4,
            "inherited_qwen_environment_artifact": "inherited-qwen-environment.json",
            "normalized_pins": PINNED_QWEN_ENV,
        }
        write_json(PACKET / "manifest.json", manifest)
        host_snapshot("packet-start")

        host_snapshot("exact-state-gate-pre")
        exact = archive_command(
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
            timeout=1_200,
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
        decision["conformance"] = {"passed": True, "arms": sorted(conformance)}
        write_json(PACKET / "decision.json", decision)
        inventory_sha256 = write_inventory()
        write_json(
            PACKET / "packet-complete.json",
            {
                "schema_version": 1,
                "verdict": decision["verdict"],
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
        print("v0.648 static check: PASS")
        return
    execute()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"v0.648 failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
