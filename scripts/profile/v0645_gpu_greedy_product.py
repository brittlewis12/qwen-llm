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
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PREREG = ROOT / "docs/bench/v0645-gpu-greedy-product.md"
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
QWEN = ROOT / "target/release/qwen"
QWEN_BENCH = ROOT / "target/release/qwen-bench"
PACKET = ROOT / "target/profiles/v0645-gpu-greedy-product-p1"
IMPLEMENTATION_COMMIT = "7add76d"

PROMPT_SHA256 = "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
PROMPT_BYTES = 1_891
PROMPT_TOKENS = 419
OUTPUT_TOKENS = 128
TRANSITIONS = 127
SCORED_REQUESTS = 4
INITIAL_PAIRS = 8
MAX_PAIRS = 16
COOLDOWN_SECONDS = 5.0
PAIR_ORDERS = ("AB", "BA") * 8

MODELS = (
    {
        "name": "a3b",
        "path": Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
        "sha256": "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
        "runtime_model_id": "e6024ce53109fdf7",
        "runtime_tokenizer_id": "a4b0b26f8a8c9917",
    },
    {
        "name": "dense27b",
        "path": Path("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf"),
        "sha256": "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0",
        "runtime_model_id": "6247cb71b536c975",
        "runtime_tokenizer_id": "a4b0b26f8a8c9917",
    },
)

T95_ONE_SIDED = {
    7: 1.894579,
    8: 1.859548,
    9: 1.833113,
    10: 1.812461,
    11: 1.795885,
    12: 1.782288,
    13: 1.770933,
    14: 1.761310,
    15: 1.753050,
}


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
    if arm == "B":
        env["QWEN_GREEDY_GPU_ARGMAX"] = "1"
    return env


def file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def assert_implementation_unchanged() -> None:
    command(["git", "merge-base", "--is-ancestor", IMPLEMENTATION_COMMIT, "HEAD"])
    result = command(
        [
            "git",
            "diff",
            "--quiet",
            f"{IMPLEMENTATION_COMMIT}..HEAD",
            "--",
            "Cargo.toml",
            "Cargo.lock",
            "crates",
            "kernels",
        ],
        check=False,
    )
    require(result.returncode == 0, "implementation changed after exact-state gate")


def static_check() -> None:
    for path in (SCRIPT, PREREG, PROMPT, *(model["path"] for model in MODELS)):
        require(path.exists(), f"missing required path: {path}")
    require(PROMPT.stat().st_size == PROMPT_BYTES, "prompt byte count changed")
    require(sha256_file(PROMPT) == PROMPT_SHA256, "prompt hash changed")
    assert_implementation_unchanged()
    synthetic = [
        {
            "A": {"metrics": {"transition": 10.0}},
            "B": {"metrics": {"transition": 9.9}},
        }
        for _ in range(INITIAL_PAIRS)
    ]
    summary = paired_summary(synthetic, "transition")
    require(summary["estimate"] > 1.01, "paired summary self-test failed")
    require(required_pairs(synthetic) == INITIAL_PAIRS, "sample-count self-test failed")
    env_a = normalized_environment("A")
    env_b = normalized_environment("B")
    differences = {
        key for key in env_a.keys() | env_b.keys() if env_a.get(key) != env_b.get(key)
    }
    require(differences == {"QWEN_GREEDY_GPU_ARGMAX"}, "arm environments drifted")


def execution_source_preflight() -> dict[str, Any]:
    require(not PACKET.exists(), f"packet already exists: {PACKET}")
    static_check()
    dirty = command_text(["git", "status", "--porcelain=v1"])
    require(not dirty, f"source is dirty: {dirty!r}")
    for path in (SCRIPT, PREREG, PROMPT):
        command(["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))])
    require(PACKET.parent.is_dir(), f"packet parent does not exist: {PACKET.parent}")
    return collect_host_snapshot()


def source_and_build_identity() -> tuple[str, dict[str, Any]]:
    dirty = command_text(["git", "status", "--porcelain=v1"])
    require(not dirty, f"source is dirty: {dirty!r}")
    assert_implementation_unchanged()
    for path in (SCRIPT, PREREG, PROMPT):
        command(["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))])
    commit = command_text(["git", "rev-parse", "HEAD"])
    build = json.loads(
        command_text([str(QWEN_BENCH), "build-info", "--output", "json"])
    )
    require(build["build_commit"] == commit, "build commit does not match source")
    require(build["runtime_commit"] == commit, "runtime commit does not match source")
    require(build["status"] == "match", "build/runtime identity mismatch")
    require(build["build_dirty"] is False, "build is dirty")
    require(build["runtime_dirty"] is False, "runtime is dirty")
    require(
        build["build_source_state"] == build["runtime_source_state"],
        "build/runtime source-state mismatch",
    )
    return commit, build


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
    competitors = []
    names = {
        "qwen",
        "qwen-bench",
        "llama-cli",
        "llama-server",
        "llama-bench",
        "mlx_lm",
        "ollama",
    }
    for line in processes.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) >= 2 and Path(fields[1]).name in names:
            competitors.append(line.strip())
    require(free_percent >= 85, f"memory free percentage too low: {free_percent}")
    require(not competitors, f"competing inference processes: {competitors}")
    require(
        "No thermal warning level has been recorded" in thermal.stdout,
        f"thermal warning present: {thermal.stdout.strip()}",
    )
    require(
        "No performance warning level has been recorded" in thermal.stdout,
        f"performance warning present: {thermal.stdout.strip()}",
    )
    return {
        "free_percent": free_percent,
        "competitors": competitors,
        "memory_pressure": memory.stdout,
        "thermal": thermal.stdout,
    }


def host_snapshot(stem: str) -> dict[str, Any]:
    snapshot = collect_host_snapshot()
    write_json(PACKET / f"{stem}.host.json", snapshot)
    return snapshot


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), f"non-object JSONL row in {path}")
            rows.append(value)
    return rows


def validate_timing_row(
    row: dict[str, Any],
    model: dict[str, Any],
    arm: str,
    commit: str,
    build: dict[str, Any],
) -> None:
    expected = {
        "request_epoch": "first_post_model_load",
        "request_index": 0,
        "prefix_cache_used": False,
        "build_commit": commit,
        "build_dirty": "0",
        "build_source_state": build["build_source_state"],
        "model": str(model["path"]),
        "runtime_model_id": model["runtime_model_id"],
        "runtime_tokenizer_id": model["runtime_tokenizer_id"],
        "prompt_source": "file",
        "prompt_bytes": PROMPT_BYTES,
        "prompt_tokens": PROMPT_TOKENS,
        "requested_tokens": 16,
        "generated_tokens": 16,
        "stop_reason": "token_limit",
        "decode_policy": "greedy_gpu_argmax" if arm == "B" else "greedy_argmax",
        "terminal_token_target_transition_consumed": False,
        "prefill_chunk_requested": 1024,
        "prefill_chunk_effective": PROMPT_TOKENS,
        "max_context_tokens": 1024,
        "transition_count": 15,
    }
    for key, value in expected.items():
        require(row.get(key) == value, f"timing {key}: {row.get(key)!r} != {value!r}")
    require("sampling" not in row, "greedy timing row contains sampling telemetry")
    require("prompt_lookup" not in row, "timing row contains prompt-lookup telemetry")


def run_conformance(
    model: dict[str, Any], commit: str, build: dict[str, Any]
) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for arm in ("A", "B"):
        stem = f"{model['name']}-conformance-{arm.lower()}"
        host_snapshot(f"{stem}-pre")
        timing_path = PACKET / f"{stem}.timing.jsonl"
        argv = [
            str(QWEN),
            "--model",
            str(model["path"]),
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
        validate_timing_row(rows[0], model, arm, commit, build)
        observations[arm] = {"stdout": result.stdout, "timing": rows[0]}
        time.sleep(COOLDOWN_SECONDS)
    require(
        observations["A"]["stdout"] == observations["B"]["stdout"],
        f"{model['name']} single-turn stdout differs",
    )
    write_json(PACKET / f"{model['name']}-conformance.json", observations)
    return observations


def make_requests() -> Path:
    path = PACKET / "requests.jsonl"
    lines = []
    for index in range(SCORED_REQUESTS + 1):
        lines.append(
            json.dumps(
                {
                    "id": "warmup" if index == 0 else f"score-{index}",
                    "prompt_file": str(PROMPT),
                    "tokens": OUTPUT_TOKENS,
                    "cache_prefix_tokens": 0,
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def validate_jsonl_rows(
    stats: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    arm: str,
    model: dict[str, Any],
) -> None:
    require(len(stats) == SCORED_REQUESTS + 1, "wrong JSONL stats row count")
    require(len(outputs) == SCORED_REQUESTS + 1, "wrong JSONL output row count")
    expected_ids = ["warmup", *(f"score-{i}" for i in range(1, SCORED_REQUESTS + 1))]
    require([row.get("id") for row in stats] == expected_ids, "stats IDs differ")
    require([row.get("id") for row in outputs] == expected_ids, "output IDs differ")
    prompt_hashes = set()
    for row in stats:
        require(row.get("model") == str(model["path"]), "wrong model path")
        require(row.get("prompt_tokens") == PROMPT_TOKENS, "wrong prompt count")
        require(row.get("requested_tokens") == OUTPUT_TOKENS, "wrong requested count")
        require(row.get("generated_tokens") == OUTPUT_TOKENS, "wrong generated count")
        require(row.get("decode_transitions") == TRANSITIONS, "wrong transition count")
        require(row.get("prefill_chunk") == 1024, "wrong prefill chunk")
        require(row.get("max_context_tokens") == 1024, "wrong context capacity")
        require(row.get("cache_hit") is False, "unexpected cache hit")
        require(row.get("cache_entries") == 0, "unexpected cache entry")
        require(row.get("cache_bytes") == 0, "unexpected cache bytes")
        require(row.get("cache_max_bytes") == 0, "unexpected cache budget")
        require(row.get("exact_cache_hit") is False, "unexpected exact cache hit")
        require(row.get("matched_prefix_tokens") == 0, "unexpected matched prefix")
        require(row.get("prefix_inserted_bytes") == 0, "unexpected prefix insertion")
        require(
            row.get("cache_prefix_source") == "request_disabled",
            "wrong cache-prefix source",
        )
        require(row.get("auto_cache_prefix_tokens") is None, "unexpected auto prefix")
        require("sampling" not in row, "greedy JSONL row contains sampling telemetry")
        require("prompt_lookup" not in row, "JSONL row contains prompt lookup")
        prompt_hashes.add(row.get("prompt_hash"))
        for key in ("transition_ms", "decode_ms", "model_ttft_ms", "prefill_ms"):
            value = float(row[key])
            require(math.isfinite(value) and value > 0.0, f"invalid {key}: {value}")
        if arm == "A":
            require(
                "decode_policy" not in row, "baseline JSONL policy must remain omitted"
            )
        else:
            require(row.get("decode_policy") == "greedy_gpu_argmax", "wrong B policy")
    for row in outputs:
        require(row.get("prompt_tokens") == PROMPT_TOKENS, "wrong output prompt count")
        require(
            row.get("generated_tokens") == OUTPUT_TOKENS, "wrong output token count"
        )
        require(
            isinstance(row.get("generated_text"), str), "generated text is not a string"
        )
    require(len(prompt_hashes) == 1, "prompt hash changed within session")
    texts = [row.get("generated_text") for row in outputs]
    require(len(set(texts)) == 1, "identical requests produced different text")


def session_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    scored = rows[1:]
    require(len(scored) == SCORED_REQUESTS, "wrong scored request count")
    return {
        "transition": statistics.median(
            float(row["transition_ms"]) / int(row["decode_transitions"])
            for row in scored
        ),
        "decode": statistics.median(
            float(row["decode_ms"]) / int(row["generated_tokens"]) for row in scored
        ),
        "ttft": statistics.median(float(row["model_ttft_ms"]) for row in scored),
        "prefill": statistics.median(float(row["prefill_ms"]) for row in scored),
    }


def run_session(
    model: dict[str, Any], arm: str, pair_index: int, order_index: int, requests: Path
) -> dict[str, Any]:
    stem = f"{model['name']}-pair-{pair_index:02d}-{order_index}-{arm.lower()}"
    host_snapshot(f"{stem}-pre")
    stats_path = PACKET / f"{stem}.stats.jsonl"
    argv = [
        str(QWEN),
        "--model",
        str(model["path"]),
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
    validate_jsonl_rows(stats, outputs, arm, model)
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
    logs = [
        math.log(pair["A"]["metrics"][metric] / pair["B"]["metrics"][metric])
        for pair in pairs
    ]
    require(len(logs) >= 2, "paired summary needs at least two pairs")
    mean = statistics.mean(logs)
    stddev = statistics.stdev(logs)
    critical = T95_ONE_SIDED[len(logs) - 1]
    half_width = critical * stddev / math.sqrt(len(logs))
    return {
        "n": len(logs),
        "estimate": math.exp(mean),
        "lower_95_one_sided": math.exp(mean - half_width),
        "upper_95_one_sided": math.exp(mean + half_width),
        "log_stddev": stddev,
        "log_half_width": half_width,
        "pair_ratios": [math.exp(value) for value in logs],
    }


def required_pairs(pairs: list[dict[str, Any]]) -> int:
    logs = [
        math.log(
            pair["A"]["metrics"]["transition"] / pair["B"]["metrics"]["transition"]
        )
        for pair in pairs
    ]
    stddev = statistics.stdev(logs)
    target = math.log(1.01)
    required = math.ceil(((1.645 + 0.84) * stddev / target) ** 2)
    return max(INITIAL_PAIRS, required)


def outputs_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return [row["generated_text"] for row in left["outputs"]] == [
        row["generated_text"] for row in right["outputs"]
    ]


def acquire_model(model: dict[str, Any], requests: Path) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    canonical_outputs: list[str] | None = None

    def acquire_pair(index: int) -> None:
        nonlocal canonical_outputs
        order = PAIR_ORDERS[index]
        sessions = {}
        for order_index, arm in enumerate(order):
            sessions[arm] = run_session(model, arm, index, order_index, requests)
            outputs = [row["generated_text"] for row in sessions[arm]["outputs"]]
            if canonical_outputs is None:
                canonical_outputs = outputs
            else:
                require(outputs == canonical_outputs, "cross-session outputs differ")
        require(outputs_equal(sessions["A"], sessions["B"]), "paired outputs differ")
        pairs.append({"index": index, "order": order, **sessions})

    for index in range(INITIAL_PAIRS):
        acquire_pair(index)
    target_pairs = required_pairs(pairs)
    if target_pairs <= MAX_PAIRS:
        for index in range(INITIAL_PAIRS, target_pairs):
            acquire_pair(index)

    summaries = {
        metric: paired_summary(pairs, metric)
        for metric in ("transition", "decode", "ttft", "prefill")
    }
    require(
        0.97 <= summaries["prefill"]["estimate"] <= 1.03,
        f"{model['name']} aggregate prefill ratio is contaminated",
    )
    result = {
        "model": model["name"],
        "initial_pairs": INITIAL_PAIRS,
        "required_pairs": target_pairs,
        "resolution_cap_exceeded": target_pairs > MAX_PAIRS,
        "pairs_completed": len(pairs),
        "summaries": summaries,
        "pairs": pairs,
    }
    write_json(PACKET / f"{model['name']}-result.json", result)
    return result


def decide(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if any(result["resolution_cap_exceeded"] for result in results.values()):
        verdict = "INCONCLUSIVE_RESOLUTION"
    else:
        a3b = results["a3b"]["summaries"]
        dense = results["dense27b"]["summaries"]
        noninferior = (
            a3b["transition"]["lower_95_one_sided"] >= 0.990
            and dense["transition"]["lower_95_one_sided"] >= 0.990
            and a3b["decode"]["lower_95_one_sided"] >= 0.980
            and dense["decode"]["lower_95_one_sided"] >= 0.980
            and a3b["ttft"]["lower_95_one_sided"] >= 0.970
            and dense["ttft"]["lower_95_one_sided"] >= 0.970
        )
        efficacy = (
            a3b["transition"]["estimate"] >= 1.010
            and a3b["transition"]["lower_95_one_sided"] > 1.000
        )
        if not noninferior:
            verdict = "KEEP_DEFAULT_OFF_REGRESSION"
        elif efficacy:
            verdict = "ADMIT_EXACT_ASSETS"
        elif a3b["transition"]["upper_95_one_sided"] < 1.010:
            verdict = "KEEP_DEFAULT_OFF_EFFECT_MISS"
        else:
            verdict = "INCONCLUSIVE_RESOLUTION"
    decision_results = {
        name: {
            "initial_pairs": result["initial_pairs"],
            "required_pairs": result["required_pairs"],
            "resolution_cap_exceeded": result["resolution_cap_exceeded"],
            "pairs_completed": result["pairs_completed"],
            "summaries": result["summaries"],
        }
        for name, result in results.items()
    }
    return {
        "schema_version": 1,
        "verdict": verdict,
        "authority": "exact-dense27b-a3b-default-admission"
        if verdict == "ADMIT_EXACT_ASSETS"
        else "none",
        "results": decision_results,
    }


def write_inventory() -> str:
    entries = []
    for path in sorted(PACKET.rglob("*")):
        if path.is_file() and path.name not in {
            "artifact-inventory.sha256",
            "packet-complete.json",
        }:
            entries.append(f"{sha256_file(path)}  {path.relative_to(PACKET)}")
    text = "\n".join(entries) + "\n"
    inventory = PACKET / "artifact-inventory.sha256"
    inventory.write_text(text)
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
    require(hashes == manifest["small_file_sha256"], "small-file hash changed")
    model_identities = {model["name"]: file_identity(model["path"]) for model in MODELS}
    require(
        model_identities == manifest["model_file_identity"],
        "model file identity changed during packet",
    )
    result = {
        "source_commit": commit,
        "build_identity": build,
        "small_file_sha256": hashes,
        "model_file_identity": model_identities,
    }
    write_json(PACKET / "final-identity.json", result)
    return result


def seal_invalid(error: BaseException) -> None:
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
    preflight_host = execution_source_preflight()
    PACKET.mkdir(parents=True)
    try:
        write_json(PACKET / "execution-preflight.host.json", preflight_host)
        model_hashes = {}
        model_identities = {}
        for model in MODELS:
            before = file_identity(model["path"])
            digest = sha256_file(model["path"])
            after = file_identity(model["path"])
            require(before == after, f"{model['name']} changed while hashing")
            require(digest == model["sha256"], f"{model['name']} model hash changed")
            model_hashes[model["name"]] = digest
            model_identities[model["name"]] = after

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
        require(build_result.returncode == 0, "release build failed")
        commit, build = source_and_build_identity()
        small_file_hashes = {
            "qwen": sha256_file(QWEN),
            "qwen_bench": sha256_file(QWEN_BENCH),
            "script": sha256_file(SCRIPT),
            "prereg": sha256_file(PREREG),
            "prompt": sha256_file(PROMPT),
        }
        manifest = {
            "schema_version": 1,
            "source_commit": commit,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "build_identity": build,
            "small_file_sha256": small_file_hashes,
            "model_sha256": model_hashes,
            "model_file_identity": model_identities,
            "admission_scope": "exact-dense27b-a3b-identity-allowlist",
            "rollback": "QWEN_GREEDY_GPU_ARGMAX=0",
            "pair_orders": PAIR_ORDERS,
            "initial_pairs": INITIAL_PAIRS,
            "max_pairs": MAX_PAIRS,
            "scored_requests_per_session": SCORED_REQUESTS,
        }
        write_json(PACKET / "manifest.json", manifest)
        host_snapshot("packet-start")

        exact_state = archive_command(
            "exact-state-gates",
            [
                "cargo",
                "test",
                "--release",
                "-p",
                "qwen-llm",
                "metal_exact_greedy_chain_matches_full_logits",
                "--",
                "--nocapture",
                "--test-threads=1",
            ],
            env=normalized_environment(),
            timeout=1_200,
        )
        exact_state_output = exact_state.stdout + exact_state.stderr
        for marker in (
            "[greedy-chain-dense-27b] exact-state PASS",
            "[greedy-chain-moe-a3b] exact-state PASS",
        ):
            require(
                marker in exact_state_output, f"missing exact-state marker: {marker}"
            )
        require(
            "skipped - fixture missing" not in exact_state_output,
            "exact-state test skipped",
        )

        requests = make_requests()
        conformance = {
            model["name"]: run_conformance(model, commit, build) for model in MODELS
        }
        results = {}
        for model in MODELS:
            result = acquire_model(model, requests)
            results[model["name"]] = result
            if result["resolution_cap_exceeded"]:
                break
        host_snapshot("packet-end")
        final_identity_check(manifest)
        decision = decide(results)
        decision["conformance_models"] = sorted(conformance)
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
        print("v0.645 static check: PASS")
        return
    execute()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"v0.645 failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
