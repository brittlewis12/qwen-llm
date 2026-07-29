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
import selectors
import shlex
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
EXPECTED_SCRIPT = ROOT / "scripts/profile/v0651_a3b_one_shot_gpu_greedy_auto.py"
PREREG = ROOT / "docs/bench/v0651-a3b-one-shot-gpu-greedy-auto.md"
PERF_LOG = ROOT / "docs/PERF-LOG.md"
PERF_ROADMAP = ROOT / "docs/PERF-ROADMAP.md"
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
QWEN = ROOT / "target/release/qwen"
QWEN_BENCH = ROOT / "target/release/qwen-bench"
PACKET = ROOT / "target/profiles/v0651-a3b-one-shot-gpu-greedy-auto-p1"

PREREG_COMMIT = "0ee75e29073396eae4c36d6e08c6466ab3c461c0"
IMPLEMENTATION_COMMIT = "791c328954a95f65440f3e1097857a9826b17546"
RESULT_DOC_COMMIT = "f73fdc3a967259d976b4dfa3567d2cc8e880018d"
MODEL = Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
MODEL_BYTES = 22_134_528_992
MODEL_SHA256 = "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
PROMPT_BYTES = 1_891
PROMPT_TOKENS = 419
PROMPT_SHA256 = "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
RUNTIME_IDENTITY = {
    "kind": "metadata_compatibility_v1",
    "model_id": "e6024ce53109fdf7",
    "tokenizer_id": "a4b0b26f8a8c9917",
}
SOURCE_BLOBS = {
    "crates/qwen-cli/src/main.rs": "a441e1174d3fca90db084bfcd12dde7a1d3542ee",
    "crates/qwen-llm/src/metal_forward.rs": "e601287141c5d44e125db747c290297e571a9a9c",
    "crates/qwen-llm/src/runtime.rs": "3a9da1245e109a266b6532cfbd2beac0a8dac81c",
}
FROZEN_BLOBS = {
    "docs/bench/v0651-a3b-one-shot-gpu-greedy-auto.md": (
        PREREG_COMMIT,
        "4de9e7276946c0e784d0916bc794f6bf298d2f02",
    ),
    "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt": (
        IMPLEMENTATION_COMMIT,
        "e98f7f929822626af1a7908a8ae9dc68170e5351",
    ),
    "docs/PERF-LOG.md": (RESULT_DOC_COMMIT, None),
    "docs/PERF-ROADMAP.md": (RESULT_DOC_COMMIT, None),
}
FROZEN_SHA256 = {
    "docs/bench/v0651-a3b-one-shot-gpu-greedy-auto.md": "7db3d0700b797383dee0eef322f72a3d9afc571ddcaad8f7c8fb5f36dd7659dc",
    "docs/PERF-LOG.md": "ed134bf27f2c03a1d4a0b6cd4fbdde4a4173ec4afb9eb1345225f89f851cd792",
    "docs/PERF-ROADMAP.md": "f53af3c5ccc568e1f0cbab934ad0144333decf911371eff45be401ac559e64f6",
}

SCHEMA_VERSION = 7
COOLDOWN_SECONDS = 5.0
READ_BYTES = 64 * 1024
T_N128 = 1.894579
T_N512 = 2.353363
EXPECTED_DEVICE = (
    "device: Apple M4 Max | unified_memory=true | max_threadgroup_memory=32768 bytes"
)
EXPECTED_HW_MEMSIZE = 137_438_953_472
PINNED_QWEN_ENV = {
    "QWEN_DECODE_GDN_FUSED_BETA_PROJ": "1",
    "QWEN_DECODE_ROPE_PAIR": "1",
    "QWEN_DECODE_MOE_GROUPED_FINALIZER": "1",
    "QWEN_DECODE_MOE_FUSED_FINALIZER": "1",
}
CELLS = (
    (1, ("AB", "BA")),
    (16, ("BA", "AB")),
    (128, ("AB", "BA") * 4),
    (512, ("AB", "BA") * 2),
)
TOKEN_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
KNOWN_INFERENCE_EXECUTABLES = {
    "qwen",
    "qwen-bench",
    "llama-cli",
    "llama-server",
    "llama-bench",
    "ollama",
    "mlx_lm",
}
POLICY_LINE = (
    "[metal-load] native quantized token embedding policy: "
    "auto-promoted (Q8_0 [2048, 248320])"
)
AUTO_POLICY_LINE = "[metal-gguf-parallel-policy] mode=auto profile=a3b-q4km-v1"
PREAD_PREFIX = (
    "[metal-gguf-parallel-pread] schema=1 resources=733 bytes=22123538944 "
    "workers=4 cuts=155,359,539 tasks=155,204,180,194 "
    "worker_bytes=5532746240,5462315776,5595522304,5532954624 "
    "first_offsets=10990048,5543736288,11006052064,16601574368 "
    "last_offsets=5392741344,11004937952,16450579424,22134520800 "
    "create=shared,default_cache,default observed=shared,default_cache,tracked "
    "page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 "
    "layout=0x5ae645df5cf7d568 "
    "inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 "
    "plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af "
)
LEDGER_LINE = (
    "[metal-load-ledger] source=733/22123538944 "
    "direct_copy=733/22123538944 direct_view=0/0 direct_alias=0/0 "
    "tail_fallback=0/0 converted=0/0/0 derived=0/0"
)
PREFETCH_LINE = (
    "[runtime-prefetch] schema=1 configured=cold-only action=suppressed "
    "reason=authenticated-disposable-auto-a3b-direct-pread "
    "profile=a3b-q4km-v1 population=pread"
)
TIMING_FIELDS = ("allocation_us", "source_us", "copy_us", "binding_us", "ready_us")
ENDPOINTS = (
    "external_F_ms",
    "external_L_ms",
    "external_E_ms",
    "external_X_ms",
    "runtime_and_model_load_ms",
    "prefill_ms",
    "ttft_ms",
    "generation_per_token_ms",
    "total_request_ms",
)
NARROW_AUTHORITY = {
    "profile": "DisposableA3bQ4kmV1PreadLogicalExactRetainedPlanV1",
    "scope": "fresh ordinary single-turn request index zero",
    "tokens": "128..=512",
    "request": "temperature-zero exact greedy without prompt lookup, warm follow-up, or durable store",
    "host": EXPECTED_DEVICE,
    "rollback": "QWEN_GREEDY_GPU_ARGMAX=0",
    "implementation_commit": IMPLEMENTATION_COMMIT,
}
ROLLBACK_NOTICE = (
    "mandatory committed rollback: restore absent QWEN_GREEDY_GPU_ARGMAX to "
    "default-off before further GPU/product experiment or candidate use"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def json_value(text: str) -> Any:
    return json.loads(text, parse_constant=reject_constant)


def json_text(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value, indent=2 if pretty else None, sort_keys=True, allow_nan=False
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json_text(value, pretty=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json_text(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def command(
    argv: list[str], *, env: dict[str, str] | None = None, timeout: int = 120
) -> str:
    result = subprocess.run(
        argv,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    require(
        result.returncode == 0,
        f"command failed ({result.returncode}): {shlex.join(argv)}\n{result.stdout}\n{result.stderr}",
    )
    return result.stdout.strip()


def git(*args: str) -> str:
    return command(["git", *args])


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def normalized_environment(
    arm: str = "A", extra: dict[str, str] | None = None
) -> dict[str, str]:
    require(arm in {"A", "B"}, f"unknown arm {arm}")
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("QWEN_")
    }
    env["NO_COLOR"] = "1"
    env.update(PINNED_QWEN_ENV)
    if arm == "A":
        env["QWEN_GREEDY_GPU_ARGMAX"] = "0"
    if extra:
        env.update(extra)
    return env


def assert_source_contract(*, acquisition: bool) -> None:
    require(SCRIPT == EXPECTED_SCRIPT, f"runner path drifted: {SCRIPT}")
    for commit in (RESULT_DOC_COMMIT, PREREG_COMMIT, IMPLEMENTATION_COMMIT):
        git("merge-base", "--is-ancestor", commit, "HEAD")
    require(
        git("rev-parse", f"{PREREG_COMMIT}^") == RESULT_DOC_COMMIT,
        "prereg ancestry drifted",
    )
    require(
        git("rev-parse", f"{IMPLEMENTATION_COMMIT}^") == PREREG_COMMIT,
        "implementation ancestry drifted",
    )
    for path, expected_blob in SOURCE_BLOBS.items():
        require(
            git("rev-parse", f"{IMPLEMENTATION_COMMIT}:{path}") == expected_blob,
            f"frozen source tree blob drifted: {path}",
        )
        require(
            git("hash-object", str(ROOT / path)) == expected_blob,
            f"candidate source changed: {path}",
        )
    for path, (commit, literal_blob) in FROZEN_BLOBS.items():
        expected = git("rev-parse", f"{commit}:{path}")
        if literal_blob is not None:
            require(expected == literal_blob, f"frozen literal blob drifted: {path}")
        require(
            git("hash-object", str(ROOT / path)) == expected,
            f"frozen file changed: {path}",
        )
    runner = relative(SCRIPT)
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    status_lines = status.splitlines()
    if head == IMPLEMENTATION_COMMIT:
        require(not acquisition, "acquisition requires the committed runner state")
        require(
            status_lines in ([f"?? {runner}"], [f"A  {runner}"]),
            "pre-commit state must contain only the untracked or staged runner addition",
        )
        require(
            not git("ls-tree", "--name-only", IMPLEMENTATION_COMMIT, "--", runner),
            "runner unexpectedly exists in the implementation commit",
        )
        if status_lines == [f"A  {runner}"]:
            require(
                git("diff", "--cached", "--name-status").splitlines()
                == [f"A\t{runner}"],
                "staged pre-commit runner addition drifted",
            )
        return
    require(
        git("rev-parse", "HEAD^") == IMPLEMENTATION_COMMIT,
        "committed runner parent is not the implementation commit",
    )
    require(
        git("rev-list", "--count", f"{IMPLEMENTATION_COMMIT}..HEAD") == "1",
        "committed runner state must be exactly one commit after implementation",
    )
    require(not status_lines, f"post-commit runner tree is dirty: {status_lines}")
    require(
        git("diff", "--name-status", f"{IMPLEMENTATION_COMMIT}..HEAD").splitlines()
        == [f"A\t{runner}"],
        "committed delta is not the exact runner addition",
    )
    head_blob = git("rev-parse", f"HEAD:{runner}")
    require(
        git("hash-object", str(SCRIPT)) == head_blob,
        "runner worktree blob differs from committed HEAD blob",
    )
    index_fields = git("ls-files", "--stage", runner).split()
    require(
        len(index_fields) >= 2 and index_fields[1] == head_blob,
        "runner index blob differs from committed HEAD blob",
    )
    if acquisition:
        require(
            head != IMPLEMENTATION_COMMIT,
            "acquisition did not reach committed runner state",
        )


def finite(
    value: Any, label: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} is not numeric",
    )
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite")
    if positive:
        require(number > 0.0, f"{label} is not positive")
    if nonnegative:
        require(number >= 0.0, f"{label} is negative")
    return number


def summarize(
    values_a: list[float], values_b: list[float], critical: float
) -> dict[str, Any]:
    require(
        len(values_a) == len(values_b) and len(values_a) in {4, 8},
        "paired sample count drifted",
    )
    logs = [math.log(a / b) for a, b in zip(values_a, values_b, strict=True)]
    mean = statistics.mean(logs)
    sd = statistics.stdev(logs)
    half = critical * sd / math.sqrt(len(logs))
    return {
        "n": len(logs),
        "estimate": math.exp(mean),
        "lower_95_one_sided": math.exp(mean - half),
        "upper_95_one_sided": math.exp(mean + half),
        "log_stddev": sd,
        "log_half_width": half,
        "critical": critical,
        "pair_ratios": [math.exp(value) for value in logs],
        "pair_logs": logs,
        "raw_A": values_a,
        "raw_B": values_b,
    }


def gate_n128(summary: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {
        "L_effect": summary["external_L_ms"]["estimate"] >= 1.02
        and summary["external_L_ms"]["lower_95_one_sided"] > 1.0,
        "generation_effect": summary["generation_per_token_ms"]["estimate"] >= 1.05
        and summary["generation_per_token_ms"]["lower_95_one_sided"] > 1.03,
        "total_effect": summary["total_request_ms"]["estimate"] >= 1.02
        and summary["total_request_ms"]["lower_95_one_sided"] > 1.0,
        "F_noninferiority": summary["external_F_ms"]["lower_95_one_sided"] >= 0.97,
        "runtime_band": 0.97
        <= summary["runtime_and_model_load_ms"]["estimate"]
        <= 1.03,
        "prefill_band": 0.97 <= summary["prefill_ms"]["estimate"] <= 1.03,
        "ttft_band": 0.97 <= summary["ttft_ms"]["estimate"] <= 1.03,
    }


def gate_n512(summary: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {
        "L_effect": summary["external_L_ms"]["estimate"] >= 1.02
        and summary["external_L_ms"]["lower_95_one_sided"] > 1.0,
        "generation_effect": summary["generation_per_token_ms"]["estimate"] >= 1.02
        and summary["generation_per_token_ms"]["lower_95_one_sided"] > 1.0,
        "total_effect": summary["total_request_ms"]["estimate"] >= 1.02
        and summary["total_request_ms"]["lower_95_one_sided"] > 1.0,
        "F_noninferiority": summary["external_F_ms"]["lower_95_one_sided"] >= 0.97,
    }


def decision_from_summaries(
    n128: dict[str, dict[str, Any]],
    n512: dict[str, dict[str, Any]],
    *,
    valid: bool = True,
) -> str:
    if not valid:
        return "INVALID"
    g128 = gate_n128(n128)
    g512 = gate_n512(n512)
    if not all(g128[key] for key in ("runtime_band", "prefill_band", "ttft_band")):
        return "INCONCLUSIVE_CONTAMINATION"
    if not g128["F_noninferiority"]:
        return "KEEP_DEFAULT_OFF_COLD_REGRESSION"
    if not all(g512.values()):
        return "KEEP_DEFAULT_OFF_ENVELOPE_GUARD_MISS"
    if not all(g128[key] for key in ("L_effect", "generation_effect", "total_effect")):
        return "KEEP_DEFAULT_OFF_EFFECT_MISS"
    return "ADMIT_A3B_ONE_SHOT_AUTO_V1"


def expected_policy(
    arm: str, tokens: int, *, jsonl: bool = False, no_profile: bool = False
) -> tuple[str, str]:
    if arm == "A":
        return "greedy_argmax", "disabled_by_explicit_rollback"
    if jsonl:
        return "greedy_argmax", "default_off_reusable_path"
    if no_profile:
        return "greedy_argmax", "default_off_no_disposable_profile"
    if tokens < 128 or tokens > 512:
        return "greedy_argmax", "default_off_requested_tokens_outside_128_512"
    return "greedy_gpu_argmax", "auto_disposable_a3b_q4km_v1"


def static_check() -> None:
    require(SCRIPT == EXPECTED_SCRIPT, "script path is not frozen path")
    for path in (
        SCRIPT,
        PREREG,
        PERF_LOG,
        PERF_ROADMAP,
        PROMPT,
        MODEL,
        *[ROOT / value for value in SOURCE_BLOBS],
    ):
        require(path.is_file(), f"required fixture unavailable: {path}")
    require(MODEL.stat().st_size == MODEL_BYTES, "model size drifted")
    require(PROMPT.stat().st_size == PROMPT_BYTES, "prompt size drifted")
    require(sha256_file(PROMPT) == PROMPT_SHA256, "prompt SHA-256 drifted")
    for path, digest in FROZEN_SHA256.items():
        require(
            sha256_file(ROOT / path) == digest, f"frozen file SHA-256 drifted: {path}"
        )
    assert_source_contract(acquisition=False)
    require(
        sum(len(orders) * 2 for _, orders in CELLS) == 32,
        "scored process count drifted",
    )
    require(
        CELLS
        == (
            (1, ("AB", "BA")),
            (16, ("BA", "AB")),
            (128, ("AB", "BA") * 4),
            (512, ("AB", "BA") * 2),
        ),
        "cell order drifted",
    )
    env_a, env_b = normalized_environment("A"), normalized_environment("B")
    differences = {
        key for key in env_a.keys() | env_b.keys() if env_a.get(key) != env_b.get(key)
    }
    require(
        differences == {"QWEN_GREEDY_GPU_ARGMAX"},
        "arm environments differ beyond greedy variable",
    )
    require(
        env_a.get("QWEN_GREEDY_GPU_ARGMAX") == "0"
        and "QWEN_GREEDY_GPU_ARGMAX" not in env_b,
        "arm policy drifted",
    )
    require(all(env_a.get(key) == "1" for key in PINNED_QWEN_ENV), "hybrid pin drifted")
    boundaries = {n: expected_policy("B", n) for n in (1, 16, 127, 128, 512, 513)}
    require(
        boundaries[127][1].startswith("default_off_requested")
        and boundaries[128][0] == "greedy_gpu_argmax",
        "lower boundary self-test failed",
    )
    require(
        boundaries[512][0] == "greedy_gpu_argmax"
        and boundaries[513][1].startswith("default_off_requested"),
        "upper boundary self-test failed",
    )
    require(
        expected_policy("B", 128, jsonl=True)[1] == "default_off_reusable_path",
        "JSONL boundary self-test failed",
    )
    require(
        expected_policy("B", 128, no_profile=True)[1]
        == "default_off_no_disposable_profile",
        "profile boundary self-test failed",
    )
    synthetic = {name: summarize([1.1] * 8, [1.0] * 8, T_N128) for name in ENDPOINTS}
    require(
        synthetic["external_L_ms"]["n"] == 8
        and synthetic["external_L_ms"]["critical"] == T_N128,
        "statistic self-test failed",
    )
    clean128 = {name: summarize([1.06] * 8, [1.0] * 8, T_N128) for name in ENDPOINTS}
    for name in ("runtime_and_model_load_ms", "prefill_ms", "ttft_ms"):
        clean128[name] = summarize([1.0] * 8, [1.0] * 8, T_N128)
    clean512 = {name: summarize([1.03] * 4, [1.0] * 4, T_N512) for name in ENDPOINTS}
    require(
        decision_from_summaries(clean128, clean512) == "ADMIT_A3B_ONE_SHOT_AUTO_V1",
        "GO decision self-test failed",
    )
    require(
        decision_from_summaries(clean128, clean512, valid=False) == "INVALID",
        "INVALID decision precedence self-test failed",
    )
    contaminated = dict(clean128)
    contaminated["prefill_ms"] = summarize([1.031] * 8, [1.0] * 8, T_N128)
    require(
        decision_from_summaries(contaminated, clean512) == "INCONCLUSIVE_CONTAMINATION",
        "decision precedence self-test failed",
    )
    cold = dict(clean128)
    cold["external_F_ms"] = summarize([0.96] * 8, [1.0] * 8, T_N128)
    require(
        decision_from_summaries(cold, clean512) == "KEEP_DEFAULT_OFF_COLD_REGRESSION",
        "cold decision self-test failed",
    )
    envelope = dict(clean512)
    envelope["external_L_ms"] = summarize([1.01] * 4, [1.0] * 4, T_N512)
    require(
        decision_from_summaries(clean128, envelope)
        == "KEEP_DEFAULT_OFF_ENVELOPE_GUARD_MISS",
        "envelope decision self-test failed",
    )
    effect = dict(clean128)
    effect["external_L_ms"] = summarize([1.01] * 8, [1.0] * 8, T_N128)
    require(
        decision_from_summaries(effect, clean512) == "KEEP_DEFAULT_OFF_EFFECT_MISS",
        "effect decision self-test failed",
    )
    require(
        classify_competing_processes(
            "1 opencode opencode recall search qwen --from now\n"
        )
        == [],
        "query process classifier self-test failed",
    )
    classified = classify_competing_processes(
        "2 qwen /tmp/qwen --model x\n3 Python /usr/bin/python3 -m mlx_lm.generate --model x\n4 ollama /usr/local/bin/ollama serve\n"
    )
    require(len(classified) == 3, "inference process classifier self-test failed")
    require(
        classify_competing_processes(
            "5 Python python3 x.py --query -m mlx_lm.generate\n6 Python python3 -c 'print(1)' -m mlx_lm.generate\n7 Python python3 -- -m mlx_lm.generate\n"
        )
        == [],
        "positional process classifier self-test failed",
    )


def is_mlx(value: str) -> bool:
    return value == "mlx_lm" or value.startswith("mlx_lm.")


def is_inference_executable(value: str) -> bool:
    name = Path(value).name
    return name in KNOWN_INFERENCE_EXECUTABLES or is_mlx(name)


def is_python(value: str) -> bool:
    return re.fullmatch(r"python(?:3(?:\.\d+)*)?", Path(value).name.lower()) is not None


def python_module_operand(argv: list[str]) -> str | None:
    if not argv or not is_python(argv[0]):
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


def classify_competing_processes(raw: str) -> list[str]:
    competitors = []
    for line in raw.splitlines():
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
            is_inference_executable(comm)
            or (argv0 and is_inference_executable(argv0))
            or (module is not None and is_mlx(module))
        ):
            competitors.append(line.strip())
    return competitors


def capture_host_command(argv: list[str]) -> dict[str, Any]:
    started = time.time_ns()
    try:
        result = subprocess.run(
            argv, cwd=ROOT, text=True, capture_output=True, timeout=30, check=False
        )
        return {
            "argv": argv,
            "returncode": result.returncode,
            "timed_out": False,
            "elapsed_ms": (time.time_ns() - started) / 1e6,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else error.stdout or ""
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else error.stderr or ""
        )
        return {
            "argv": argv,
            "returncode": None,
            "timed_out": True,
            "elapsed_ms": (time.time_ns() - started) / 1e6,
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
    result = dict(raw)
    commands = raw["commands"]
    line = next(
        (
            value
            for value in commands["memory_pressure"]["stdout"].splitlines()
            if value.startswith("System-wide memory free percentage:")
        ),
        None,
    )
    free = None
    error = None
    if line is not None:
        try:
            free = int(line.rsplit(" ", 1)[-1].rstrip("%"))
        except ValueError as caught:
            error = str(caught)
    thermal = commands["thermal"]["stdout"]
    result.update(
        {
            "free_percent": free,
            "free_percent_error": error,
            "competitors": classify_competing_processes(
                commands["processes"]["stdout"]
            ),
            "thermal_warning_clear": "No thermal warning level has been recorded"
            in thermal,
            "performance_warning_clear": "No performance warning level has been recorded"
            in thermal,
        }
    )
    return result


def validate_host(snapshot: dict[str, Any], label: str) -> None:
    for name, record in snapshot["commands"].items():
        require(
            not record["timed_out"] and record["returncode"] == 0,
            f"{label} host command failed: {name}",
        )
    require(
        snapshot["free_percent_error"] is None and snapshot["free_percent"] is not None,
        f"{label} memory evidence malformed",
    )
    require(
        snapshot["free_percent"] >= 85,
        f"{label} free memory below 85%: {snapshot['free_percent']}",
    )
    require(
        not snapshot["competitors"],
        f"{label} inference competitor: {snapshot['competitors']}",
    )
    require(snapshot["thermal_warning_clear"], f"{label} thermal warning")
    require(snapshot["performance_warning_clear"], f"{label} performance warning")


def raw_host_snapshot(stem: str) -> dict[str, Any]:
    raw = collect_host_snapshot()
    raw_path = PACKET / f"{stem}.host.raw.json"
    write_json(raw_path, raw)
    evaluated = evaluate_host_snapshot(raw)
    write_json(PACKET / f"{stem}.host.json", evaluated)
    validate_host(evaluated, stem)
    return evaluated


def rusage_dict(usage: Any) -> dict[str, Any]:
    return {
        name: getattr(usage, name)
        for name in (
            "ru_utime",
            "ru_stime",
            "ru_maxrss",
            "ru_ixrss",
            "ru_idrss",
            "ru_isrss",
            "ru_minflt",
            "ru_majflt",
            "ru_nswap",
            "ru_inblock",
            "ru_oublock",
            "ru_msgsnd",
            "ru_msgrcv",
            "ru_nsignals",
            "ru_nvcsw",
            "ru_nivcsw",
        )
    }


def wait4_nohang(pid: int) -> tuple[int, int, Any] | None:
    while True:
        try:
            observed, status, usage = os.wait4(pid, os.WNOHANG)
            if observed == 0:
                return None
            require(observed == pid, f"wait4 reaped {observed}, expected {pid}")
            return observed, status, usage
        except InterruptedError:
            continue


def terminate_exact(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if wait4_nohang(pid) is not None:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    while wait4_nohang(pid) is None:
        time.sleep(0.05)


def direct_child(
    stem: str,
    argv: list[str],
    env: dict[str, str],
    *,
    timeout: float,
    scored: bool,
    require_stdout: bool = True,
    expected_model_identity: dict[str, int] | None = None,
) -> dict[str, Any]:
    stdout_path = PACKET / f"{stem}.stdout.bin"
    stderr_path = PACKET / f"{stem}.stderr.bin"
    command_path = PACKET / f"{stem}.command.json"
    for path in (stdout_path, stderr_path, command_path):
        require(not path.exists(), f"refusing to overwrite {path}")
    pre = raw_host_snapshot(f"{stem}.pre")
    identity_before = file_identity(MODEL)
    append_jsonl(
        PACKET / "events.jsonl",
        {
            "event": "launch-intent",
            "stem": stem,
            "argv": argv,
            "unix_ns": time.time_ns(),
            "scored": scored,
        },
    )
    process: subprocess.Popen[bytes] | None = None
    launch_ns: int | None = None
    stdout = bytearray()
    read_events: list[dict[str, int]] = []
    reaped: tuple[int, int, Any] | None = None
    reaped_ns: int | None = None
    error: BaseException | None = None
    selector: selectors.BaseSelector | None = None
    with stderr_path.open("xb", buffering=0) as stderr_handle:
        try:
            launch_ns = time.monotonic_ns()
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                bufsize=0,
            )
            append_jsonl(
                PACKET / "events.jsonl",
                {
                    "event": "launch",
                    "stem": stem,
                    "argv": argv,
                    "monotonic_ns": launch_ns,
                    "unix_ns": time.time_ns(),
                    "pid": process.pid,
                    "scored": scored,
                },
            )
            require(process.stdout is not None, "stdout pipe unavailable")
            selector = selectors.DefaultSelector()
            selector.register(process.stdout.fileno(), selectors.EVENT_READ)
            deadline = time.monotonic() + timeout
            eof = False
            while not eof:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{stem} exceeded generous timeout {timeout}s")
                for key, _ in selector.select(timeout=0.25):
                    chunk = os.read(key.fd, READ_BYTES)
                    observed_ns = time.monotonic_ns()
                    if chunk:
                        stdout.extend(chunk)
                        read_events.append(
                            {"monotonic_ns": observed_ns, "bytes": len(chunk)}
                        )
                    else:
                        selector.unregister(key.fd)
                        eof = True
            while reaped is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{stem} exceeded generous timeout {timeout}s")
                reaped = wait4_nohang(process.pid)
                if reaped is None:
                    time.sleep(0.05)
            reaped_ns = time.monotonic_ns()
            process.returncode = os.waitstatus_to_exitcode(reaped[1])
        except BaseException as caught:
            error = caught
            if process is not None and reaped is None:
                terminate_exact(process.pid)
        finally:
            if selector is not None:
                selector.close()
            if process is not None and process.stdout is not None:
                process.stdout.close()
    require(launch_ns is not None, f"{stem} launch timing origin absent")
    end_ns = time.monotonic_ns()
    stdout_path.write_bytes(stdout)
    post_raw = collect_host_snapshot()
    write_json(PACKET / f"{stem}.post.host.raw.json", post_raw)
    post = evaluate_host_snapshot(post_raw)
    write_json(PACKET / f"{stem}.post.host.json", post)
    identity_after = file_identity(MODEL)
    returncode = process.returncode if process is not None else None
    usage = rusage_dict(reaped[2]) if reaped is not None else None
    completion = {
        "event": "completion",
        "stem": stem,
        "monotonic_ns": end_ns,
        "unix_ns": time.time_ns(),
        "pid": process.pid if process else None,
        "returncode": returncode,
        "error": f"{type(error).__name__}: {error}" if error else None,
    }
    append_jsonl(PACKET / "events.jsonl", completion)
    record = {
        "argv": argv,
        "environment_sha256": environment_sha256(env),
        "qwen_environment": {
            key: value for key, value in sorted(env.items()) if key.startswith("QWEN_")
        },
        "launch_monotonic_ns": launch_ns,
        "read_events": read_events,
        "wait_monotonic_ns": reaped_ns,
        "end_monotonic_ns": end_ns,
        "pid": process.pid if process else None,
        "returncode": returncode,
        "timeout_seconds": timeout,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_bytes": stderr_path.stat().st_size,
        "stderr_sha256": sha256_file(stderr_path),
        "rusage": usage,
        "model_identity_before": identity_before,
        "model_identity_after": identity_after,
        "pre_host_free_percent": pre["free_percent"],
        "post_host_free_percent": post["free_percent"],
    }
    write_json(command_path, record)
    if error is not None:
        raise error
    validate_host(post, f"{stem}.post")
    require(identity_before == identity_after, f"{stem} model identity changed")
    if expected_model_identity is not None:
        require(
            identity_before == expected_model_identity,
            f"{stem} model identity differs from initial manifest",
        )
    require(returncode == 0, f"{stem} exited {returncode}")
    require(read_events or not require_stdout, f"{stem} produced no stdout")
    if scored:
        require(usage is not None, f"{stem} exact-child rusage absent")
        require(usage["ru_inblock"] == 0, f"{stem} child block input operations")
        require(usage["ru_majflt"] == 0, f"{stem} child major faults")
    require(reaped_ns is not None, f"{stem} exact wait timestamp absent")
    first_ns = read_events[0]["monotonic_ns"] if read_events else reaped_ns
    last_ns = read_events[-1]["monotonic_ns"] if read_events else reaped_ns
    return {
        "stdout": bytes(stdout),
        "stderr": stderr_path.read_bytes(),
        "rusage": usage,
        "external": {
            "external_F_ms": (first_ns - launch_ns) / 1e6,
            "external_L_ms": (last_ns - launch_ns) / 1e6,
            "external_E_ms": (reaped_ns - launch_ns) / 1e6,
            "external_X_ms": (reaped_ns - last_ns) / 1e6,
            "F_monotonic_ns": first_ns,
            "L_monotonic_ns": last_ns,
            "E_monotonic_ns": reaped_ns,
        },
        "model_identity": identity_after,
    }


def environment_sha256(env: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(env.items()):
        for item in (key.encode(), value.encode()):
            digest.update(len(item).to_bytes(8, "little"))
            digest.update(item)
    return digest.hexdigest()


def parse_timing(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    require(data.endswith(b"\n"), f"timing file lacks final newline: {path}")
    lines = [line for line in data.decode("utf-8").splitlines() if line]
    require(len(lines) == 1, f"expected one timing row in {path}, got {len(lines)}")
    row = json_value(lines[0])
    require(isinstance(row, dict), f"timing row is not object: {path}")
    return row


def parse_pread_marker(line: str) -> dict[str, int]:
    suffix = line.removeprefix(PREAD_PREFIX)
    require(suffix != line, "pread marker authenticated prefix drifted")
    fields = suffix.split(" ")
    require(len(fields) == len(TIMING_FIELDS), "pread timing field count drifted")
    values = {}
    for field, name in zip(fields, TIMING_FIELDS, strict=True):
        raw = field.removeprefix(f"{name}=")
        require(
            raw != field and raw.isascii() and raw.isdecimal() and str(int(raw)) == raw,
            f"pread {name} malformed",
        )
        values[name] = int(raw)
    require(
        values["ready_us"] > 0
        and abs(values["ready_us"] - sum(values[name] for name in TIMING_FIELDS[:-1]))
        <= 4,
        "pread timings do not reconcile",
    )
    return values


def parse_exact_load(stderr: bytes, *, expect_profile: bool = True) -> dict[str, Any]:
    text = stderr.decode("utf-8")
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-policy]",
        "[metal-gguf-parallel-pread]",
        "[metal-load-ledger]",
        "[runtime-prefetch]",
    )
    recognized = [line for line in text.splitlines() if line.startswith(prefixes)]
    if not expect_profile:
        disposable_lines = [
            line
            for line in text.splitlines()
            if line.startswith(
                (
                    "[metal-gguf-parallel-policy]",
                    "[metal-gguf-parallel-pread]",
                    "[runtime-prefetch]",
                )
            )
        ]
        require(
            not disposable_lines,
            "no-profile fallback realized Auto profile",
        )
        return {
            "realized_profile": False,
            "auto_policy_present": False,
            "pread_present": False,
            "disposable_prefetch_present": False,
            "recognized_lines": recognized,
        }
    require(
        text.splitlines().count(POLICY_LINE) == 1, "native embedding marker drifted"
    )
    require(
        text.splitlines().count(AUTO_POLICY_LINE) == 1, "A3B Auto policy marker drifted"
    )
    require(text.splitlines().count(LEDGER_LINE) == 1, "copied ledger marker drifted")
    require(text.splitlines().count(PREFETCH_LINE) == 1, "pread runtime marker drifted")
    pread = [
        line
        for line in text.splitlines()
        if line.startswith("[metal-gguf-parallel-pread]")
    ]
    require(len(pread) == 1, "pread marker count drifted")
    phase = parse_pread_marker(pread[0])
    require(
        sorted(recognized)
        == sorted(
            [POLICY_LINE, AUTO_POLICY_LINE, pread[0], LEDGER_LINE, PREFETCH_LINE]
        ),
        "recognized loader marker set drifted",
    )
    ordered = [
        text.index(POLICY_LINE),
        text.index(AUTO_POLICY_LINE),
        text.index(pread[0]),
        text.index(LEDGER_LINE),
    ]
    require(ordered == sorted(ordered), "loader marker order drifted")
    forbidden = (
        "[metal-gguf-parallel-copied]",
        "[metal-gguf-parallel-page-rounded]",
        "[metal-gguf-owned]",
        "[metal-gguf-no-copy]",
    )
    require(not any(value in text for value in forbidden), "unexpected loader surface")
    return {
        "realized_profile": True,
        "policy": AUTO_POLICY_LINE,
        "population": "Pread",
        "destination": "LogicalExact",
        "proof": "A3bRetainedPlan",
        "marker": pread[0],
        "phase_us": phase,
    }


def validate_allocations(value: Any, where: str) -> None:
    sample_keys = (
        "process_model_ready",
        "request_start",
        "after_scratch",
        "after_sequence",
        "after_prefill",
        "after_first_stdout_flush",
        "request_end_before_state_drop",
        "after_request_state_drop",
    )
    fields = {
        "current_bytes",
        "delta_from_model_ready_bytes",
        "delta_from_request_start_bytes",
    }
    expected_keys = {*sample_keys, "current_allocated_sampled_max_bytes"}
    require(
        isinstance(value, dict) and set(value) == expected_keys,
        f"{where} Metal allocation sample keys drifted",
    )
    maximum = value["current_allocated_sampled_max_bytes"]
    finite(
        maximum,
        f"{where}.allocated_max",
        nonnegative=True,
    )
    require(
        isinstance(maximum, int) and not isinstance(maximum, bool),
        f"{where} allocation maximum is not an integer",
    )
    currents: dict[str, int] = {}
    for key in sample_keys:
        sample = value[key]
        require(
            isinstance(sample, dict) and set(sample) == fields,
            f"{where}.{key} allocation fields drifted",
        )
        current = sample["current_bytes"]
        require(
            isinstance(current, int) and not isinstance(current, bool) and current >= 0,
            f"{where}.{key}.current_bytes is not a nonnegative integer",
        )
        currents[key] = current
        for field in (
            "delta_from_model_ready_bytes",
            "delta_from_request_start_bytes",
        ):
            require(
                isinstance(sample[field], int) and not isinstance(sample[field], bool),
                f"{where}.{key}.{field} is not a signed integer",
            )
    model_ready = currents["process_model_ready"]
    request_start = currents["request_start"]
    for key in sample_keys:
        sample = value[key]
        require(
            sample["delta_from_model_ready_bytes"] == currents[key] - model_ready,
            f"{where}.{key} model-ready allocation delta does not reconcile",
        )
        require(
            sample["delta_from_request_start_bytes"] == currents[key] - request_start,
            f"{where}.{key} request-start allocation delta does not reconcile",
        )
    require(
        maximum == max(currents.values()),
        f"{where} sampled allocation maximum does not reconcile",
    )


def validate_pso(value: Any, where: str) -> None:
    require(
        isinstance(value, dict) and set(value) == {"prefill", "generation", "total"},
        f"{where} PSO metrics malformed",
    )
    for phase, metrics in value.items():
        require(isinstance(metrics, dict), f"{where}.{phase} PSO phase malformed")
        for key in ("misses", "miss_wall_ns", "compiler_wall_ns"):
            finite(metrics.get(key), f"{where}.{phase}.{key}", nonnegative=True)


def validate_single_turn_row(
    row: dict[str, Any],
    arm: str,
    tokens: int,
    source: dict[str, Any],
    where: str,
    *,
    no_profile: bool = False,
) -> dict[str, float]:
    policy, reason = expected_policy(arm, tokens, no_profile=no_profile)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "request_epoch": "first_post_model_load",
        "request_index": 0,
        "tokenizer_reused": False,
        "pair_requested": False,
        "pair_id": None,
        "pair_request_equal": None,
        "pair_generated_tokens_equal": None,
        "prefix_cache_used": False,
        "build_commit": source["commit"],
        "build_dirty": "0",
        "build_source_state": source["build"]["build_source_state"],
        "model": str(MODEL),
        "runtime_identity_kind": RUNTIME_IDENTITY["kind"],
        "runtime_model_id": RUNTIME_IDENTITY["model_id"],
        "runtime_tokenizer_id": RUNTIME_IDENTITY["tokenizer_id"],
        "greedy_gpu_selection_reason": reason,
        "stdout_sink": "redirected",
        "ttft_endpoint": "stdout_flush_complete",
        "prompt_source": "file",
        "prompt_bytes": PROMPT_BYTES,
        "prompt_tokens": PROMPT_TOKENS,
        "requested_tokens": tokens,
        "generated_tokens": tokens,
        "stop_reason": "token_limit",
        "decode_policy": policy,
        "terminal_token_target_transition_consumed": False,
        "no_special_tokens": False,
        "prefill_chunk_requested": 1024,
        "prefill_chunk_effective": PROMPT_TOKENS,
        "max_context_tokens": 1024,
        "transition_count": tokens - 1,
    }
    for key, value in expected.items():
        require(row.get(key) == value, f"{where} {key}: {row.get(key)!r} != {value!r}")
    require(
        "sampling" not in row and "prompt_lookup" not in row,
        f"{where} sampling/lookup surface present",
    )
    require(
        TOKEN_SHA_RE.fullmatch(str(row.get("generated_token_sha256"))) is not None,
        f"{where} token digest malformed",
    )
    positive = (
        "runtime_and_model_load_ms",
        "prompt_acquisition_ms",
        "tokenizer_init_ms",
        "tokenization_ms",
        "capacity_validation_ms",
        "scratch_allocation_ms",
        "sequence_allocation_ms",
        "prefill_ms",
        "first_token_selection_ms",
        "first_token_callback_duration_ms",
        "first_token_ready_ms",
        "ttft_ms",
        "generation_ms",
        "inference_complete_ms",
        "total_request_ms",
    )
    for key in positive:
        finite(row.get(key), f"{where}.{key}", positive=True)
    finite(
        row.get("transition_ms"),
        f"{where}.transition_ms",
        positive=tokens > 1,
        nonnegative=tokens == 1,
    )
    finite(
        row.get("transition_tps"),
        f"{where}.transition_tps",
        positive=tokens > 1,
        nonnegative=tokens == 1,
    )
    require(
        row["first_token_ready_ms"]
        <= row["ttft_ms"]
        <= row["inference_complete_ms"]
        <= row["total_request_ms"],
        f"{where} internal timing order invalid",
    )
    require(
        row["prefill_ms"] <= row["first_token_ready_ms"],
        f"{where} prefill/first-token order invalid",
    )
    validate_pso(row.get("pso_cache"), where)
    validate_allocations(row.get("metal_allocated"), where)
    return {
        "runtime_and_model_load_ms": float(row["runtime_and_model_load_ms"]),
        "prefill_ms": float(row["prefill_ms"]),
        "ttft_ms": float(row["ttft_ms"]),
        "generation_per_token_ms": float(row["generation_ms"]) / tokens,
        "total_request_ms": float(row["total_request_ms"]),
    }


def qwen_command(tokens: int, timing: Path) -> list[str]:
    return [
        str(QWEN),
        "--model",
        str(MODEL),
        "--prompt-file",
        str(PROMPT),
        "--tokens",
        str(tokens),
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
        str(timing),
    ]


def run_single(
    stem: str,
    arm: str,
    tokens: int,
    source: dict[str, Any],
    model_identity: dict[str, int],
    *,
    scored: bool,
    extra_env: dict[str, str] | None = None,
    no_profile: bool = False,
) -> dict[str, Any]:
    timing = PACKET / f"{stem}.timing.jsonl"
    require(not timing.exists(), f"timing path already exists: {timing}")
    timeout = 7_200.0 if tokens == 512 else 3_600.0
    child = direct_child(
        stem,
        qwen_command(tokens, timing),
        normalized_environment(arm, extra_env),
        timeout=timeout,
        scored=scored,
        expected_model_identity=model_identity,
    )
    row = parse_timing(timing)
    metrics = validate_single_turn_row(
        row, arm, tokens, source, stem, no_profile=no_profile
    )
    load = parse_exact_load(child["stderr"], expect_profile=not no_profile)
    stdout = child["stdout"]
    require(
        stdout.endswith(b"\n") and not stdout.endswith(b"\n\n"),
        f"{stem} stdout is not response plus exactly one newline",
    )
    response = stdout[:-1]
    external = child["external"]
    require(
        0
        < external["external_F_ms"]
        <= external["external_L_ms"]
        <= external["external_E_ms"],
        f"{stem} external timing order invalid",
    )
    require(external["external_X_ms"] >= 0, f"{stem} teardown timing invalid")
    return {
        "stem": stem,
        "arm": arm,
        "tokens": tokens,
        "timing": row,
        "load": load,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "response_sha256": hashlib.sha256(response).hexdigest(),
        "response_hex": response.hex(),
        "generated_token_sha256": row["generated_token_sha256"],
        "metrics": {
            **metrics,
            **{key: external[key] for key in external if key.startswith("external_")},
        },
        "external": external,
        "rusage": child["rusage"],
        "model_identity": child["model_identity"],
    }


def assert_same_output(
    reference: dict[str, Any], observed: dict[str, Any], where: str
) -> None:
    for key in ("response_hex", "stdout_sha256", "generated_token_sha256"):
        require(
            observed[key] == reference[key], f"{where} {key} differs from A reference"
        )
    immutable = (
        "prompt_tokens",
        "requested_tokens",
        "generated_tokens",
        "transition_count",
        "stop_reason",
        "terminal_token_target_transition_consumed",
        "prompt_bytes",
        "prefill_chunk_effective",
        "max_context_tokens",
        "runtime_model_id",
        "runtime_tokenizer_id",
        "build_commit",
        "build_source_state",
    )
    for key in immutable:
        require(
            observed["timing"].get(key) == reference["timing"].get(key),
            f"{where} timing identity {key} differs",
        )


def run_jsonl_gate(
    stem: str,
    force: bool,
    source: dict[str, Any],
    model_identity: dict[str, int],
) -> dict[str, Any]:
    requests = PACKET / f"{stem}.requests.jsonl"
    stats = PACKET / f"{stem}.stats.jsonl"
    requests.write_text(
        json_text(
            {
                "id": stem,
                "prompt_file": str(PROMPT),
                "tokens": 128,
                "cache_prefix_tokens": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    argv = [
        str(QWEN),
        "--model",
        str(MODEL),
        "--requests-jsonl",
        str(requests),
        "--request-stats",
        str(stats),
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
    ]
    env = normalized_environment(
        "B", {"QWEN_GREEDY_GPU_ARGMAX": "1"} if force else None
    )
    child = direct_child(
        stem,
        argv,
        env,
        timeout=3_600,
        scored=False,
        expected_model_identity=model_identity,
    )
    stats_bytes = stats.read_bytes()
    stdout_bytes = child["stdout"]
    require(
        stats_bytes.endswith(b"\n")
        and stats_bytes.count(b"\n") == 1
        and stats_bytes.strip(),
        f"{stem} stats is not exactly one newline-terminated JSONL row",
    )
    require(
        stdout_bytes.endswith(b"\n")
        and stdout_bytes.count(b"\n") == 1
        and stdout_bytes.strip(),
        f"{stem} stdout is not exactly one newline-terminated JSONL row",
    )
    rows = [json_value(stats_bytes[:-1].decode("utf-8"))]
    outputs = [json_value(stdout_bytes[:-1].decode("utf-8"))]
    require(len(rows) == len(outputs) == 1, f"{stem} JSONL row count drifted")
    row, output = rows[0], outputs[0]
    require(
        isinstance(row, dict) and isinstance(output, dict),
        f"{stem} JSONL rows are not objects",
    )
    reason = "force_enabled" if force else "default_off_reusable_path"
    policy = "greedy_gpu_argmax" if force else "greedy_argmax"
    expected = {
        "schema_version": 7,
        "greedy_gpu_selection_reason": reason,
        "decode_policy": policy,
        "prompt_tokens": PROMPT_TOKENS,
        "requested_tokens": 128,
        "generated_tokens": 128,
        "decode_transitions": 127,
        "stop_reason": "token_limit",
        "terminal_token_target_transition_consumed": False,
        "cache_hit": False,
        "cache_entries": 0,
        "cache_bytes": 0,
        "cache_max_bytes": 0,
    }
    for key, value in expected.items():
        actual = (
            output.get(key)
            if key in {"stop_reason", "terminal_token_target_transition_consumed"}
            else row.get(key)
        )
        require(actual == value, f"{stem} {key} drifted: {actual!r}")
    require(
        "sampling" not in row and "prompt_lookup" not in row,
        f"{stem} JSONL sampling/lookup surface",
    )
    require(
        row.get("generated_token_sha256") == output.get("generated_token_sha256"),
        f"{stem} JSONL token digest differs",
    )
    require(
        TOKEN_SHA_RE.fullmatch(str(row.get("generated_token_sha256"))) is not None,
        f"{stem} JSONL token digest is malformed",
    )
    for key, value in {
        "id": stem,
        "prompt_tokens": PROMPT_TOKENS,
        "generated_tokens": 128,
        "stop_reason": "token_limit",
        "terminal_token_target_transition_consumed": False,
    }.items():
        require(output.get(key) == value, f"{stem} output {key} drifted")
    require(
        isinstance(output.get("generated_text"), str),
        f"{stem} JSONL generated text missing",
    )
    load = parse_exact_load(child["stderr"], expect_profile=False)
    require(
        load.get("realized_profile") is False
        and load.get("auto_policy_present") is False
        and load.get("pread_present") is False
        and load.get("disposable_prefetch_present") is False,
        f"{stem} JSONL unexpectedly realized disposable Auto/pread/prefetch",
    )
    return {
        "timing": row,
        "output": output,
        "load": load,
        "stdout_sha256": hashlib.sha256(child["stdout"]).hexdigest(),
    }


def archive_cpu_command(
    stem: str,
    argv: list[str],
    env: dict[str, str],
    timeout: int,
    *,
    expected_model_identity: dict[str, int] | None = None,
) -> dict[str, Any]:
    child = direct_child(
        stem,
        argv,
        env,
        timeout=timeout,
        scored=False,
        require_stdout=False,
        expected_model_identity=expected_model_identity,
    )
    return {
        "stdout": child["stdout"].decode(errors="replace"),
        "stderr": child["stderr"].decode(errors="replace"),
    }


def source_build_identity() -> dict[str, Any]:
    assert_source_contract(acquisition=True)
    commit = git("rev-parse", "HEAD")
    build = json_value(
        command(
            [str(QWEN_BENCH), "build-info", "--output", "json"],
            env=normalized_environment(),
        )
    )
    require(isinstance(build, dict), "build identity is not object")
    require(
        build.get("build_commit") == commit
        and build.get("runtime_commit") == commit
        and build.get("status") == "match",
        "build/source identity mismatch",
    )
    require(
        build.get("build_dirty") is False and build.get("runtime_dirty") is False,
        "dirty build identity",
    )
    require(
        build.get("build_source_state") == build.get("runtime_source_state"),
        "build/runtime source state mismatch",
    )
    return {"commit": commit, "build": build}


def validate_host_device() -> dict[str, Any]:
    device = command([str(QWEN), "--info"], env=normalized_environment())
    memory = int(command(["sysctl", "-n", "hw.memsize"]))
    require(device == EXPECTED_DEVICE, f"device contract drifted: {device!r}")
    require(memory == EXPECTED_HW_MEMSIZE, f"host memory contract drifted: {memory}")
    return {
        "device": device,
        "hw_memsize": memory,
        "macos_product": command(["sw_vers", "-productVersion"]),
        "macos_build": command(["sw_vers", "-buildVersion"]),
    }


def initial_manifest(
    source: dict[str, Any],
    host: dict[str, Any],
    model_hash: str,
    model_identity: dict[str, int],
) -> dict[str, Any]:
    hashes = {
        "qwen": sha256_file(QWEN),
        "qwen_bench": sha256_file(QWEN_BENCH),
        "runner": sha256_file(SCRIPT),
        "prereg": sha256_file(PREREG),
        "perf_log": sha256_file(PERF_LOG),
        "perf_roadmap": sha256_file(PERF_ROADMAP),
        "prompt": sha256_file(PROMPT),
        **{f"source:{path}": sha256_file(ROOT / path) for path in SOURCE_BLOBS},
    }
    return {
        "schema_version": 1,
        "source": source,
        "host_contract": host,
        "prereg_commit": PREREG_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "result_doc_commit": RESULT_DOC_COMMIT,
        "expected_script": str(EXPECTED_SCRIPT),
        "model": {
            "path": str(MODEL),
            "bytes": MODEL_BYTES,
            "sha256": model_hash,
            "file_identity": model_identity,
        },
        "prompt": {
            "path": str(PROMPT),
            "bytes": PROMPT_BYTES,
            "tokens": PROMPT_TOKENS,
            "sha256": PROMPT_SHA256,
        },
        "runtime_identity": RUNTIME_IDENTITY,
        "small_file_sha256": hashes,
        "inherited_qwen_environment_artifact": "inherited-qwen-environment.json",
        "normalized_pins": PINNED_QWEN_ENV,
        "arm_environments": {
            "A": "QWEN_GREEDY_GPU_ARGMAX=0",
            "B": "QWEN_GREEDY_GPU_ARGMAX absent",
        },
        "cells": [
            {"tokens": tokens, "orders": list(orders)} for tokens, orders in CELLS
        ],
        "scored_processes": 32,
        "retries": 0,
        "continuations": 0,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "authority_on_go": NARROW_AUTHORITY,
    }


def run_gates(
    source: dict[str, Any],
    model_identity: dict[str, int],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    policy_tests = archive_cpu_command(
        "gate-pure-policy-tests",
        [
            "cargo",
            "test",
            "--release",
            "-p",
            "qwen-cli",
            "--bin",
            "qwen",
            "--",
            "--test-threads=1",
        ],
        normalized_environment(),
        3_600,
    )
    policy_output = policy_tests["stdout"] + policy_tests["stderr"]
    for test_name in (
        "greedy_gpu_env_parsing_preserves_absent_force_and_rollback_states",
        "absent_gpu_greedy_requires_exact_realized_marker",
        "absent_gpu_greedy_is_narrowly_scoped",
        "absent_gpu_greedy_enforces_requested_token_envelope",
        "gpu_greedy_precedence_is_rollback_then_request_eligibility_then_force",
    ):
        require(
            policy_output.splitlines().count(f"test tests::{test_name} ... ok") == 1,
            f"pure policy test did not emit one exact passing record: {test_name}",
        )
    require(
        len(
            re.findall(
                r"^test result: ok\. 56 passed; 0 failed; 0 ignored; "
                r"0 measured; 0 filtered out; finished in [0-9.]+s$",
                policy_output,
                re.MULTILINE,
            )
        )
        == 1,
        "qwen binary unit suite did not report the exact 56-test pass count",
    )
    profile_test_names = (
        (
            "runtime::tests::model_load_intent_scopes_parallel_copy_auto_admission",
            "model_load_intent_scopes_parallel_copy_auto_admission",
        ),
        (
            "runtime::tests::prefetch_action_selector_table_is_fail_closed",
            "prefetch_action_selector_table_is_fail_closed",
        ),
        (
            "metal_forward::tests::"
            "realized_auto_load_marker_candidate_is_exact_and_versioned",
            "realized_auto_load_marker_candidate_is_exact_and_versioned",
        ),
        (
            "metal_forward::tests::"
            "prepared_auto_prefetch_advice_selector_table_is_fail_closed",
            "prepared_auto_prefetch_advice_selector_table_is_fail_closed",
        ),
    )
    profile_test_records = []
    for index, (qualified_name, filter_name) in enumerate(profile_test_names, 1):
        profile_test = archive_cpu_command(
            f"gate-pure-profile-test-{index}",
            [
                "cargo",
                "test",
                "--release",
                "-p",
                "qwen-llm",
                filter_name,
                "--",
                "--test-threads=1",
            ],
            normalized_environment(),
            3_600,
            expected_model_identity=model_identity,
        )
        profile_output = profile_test["stdout"] + profile_test["stderr"]
        require(
            profile_output.splitlines().count(f"test {qualified_name} ... ok") == 1,
            f"pure realized-profile test did not pass exactly once: {qualified_name}",
        )
        require(
            len(
                re.findall(
                    r"^test result: ok\. 1 passed; 0 failed; 0 ignored; "
                    r"0 measured; [0-9]+ filtered out; finished in [0-9.]+s$",
                    profile_output,
                    re.MULTILINE,
                )
            )
            == 1,
            f"pure realized-profile filter count drifted: {qualified_name}",
        )
        profile_test_records.append(qualified_name)
    exact = archive_cpu_command(
        "gate-exact-state",
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
        normalized_environment(),
        7_200,
        expected_model_identity=model_identity,
    )
    combined = exact["stdout"] + exact["stderr"]
    marker = "[greedy-chain-moe-a3b] exact-state PASS"
    require(
        combined.count(marker) == 1 and "skipped" not in combined.lower(),
        "exact state gate did not prove exact tokens/logits/KV/GDN state",
    )
    conformance_a = run_single(
        "gate-n128-a", "A", 128, source, model_identity, scored=False
    )
    time.sleep(COOLDOWN_SECONDS)
    conformance_b = run_single(
        "gate-n128-b", "B", 128, source, model_identity, scored=False
    )
    assert_same_output(conformance_a, conformance_b, "N128 conformance")
    time.sleep(COOLDOWN_SECONDS)
    jsonl_absent = run_jsonl_gate("gate-jsonl-absent", False, source, model_identity)
    time.sleep(COOLDOWN_SECONDS)
    jsonl_force = run_jsonl_gate("gate-jsonl-force", True, source, model_identity)
    time.sleep(COOLDOWN_SECONDS)
    no_profile = run_single(
        "gate-no-profile",
        "B",
        128,
        source,
        model_identity,
        scored=False,
        extra_env={"QWEN_GGUF_PARALLEL_COPY": "0"},
        no_profile=True,
    )
    time.sleep(COOLDOWN_SECONDS)
    reference512 = run_single(
        "gate-n512-reference-a", "A", 512, source, model_identity, scored=False
    )
    time.sleep(COOLDOWN_SECONDS)
    gates = {
        "pure_policy_matrix": {
            "passed": True,
            "suite_tests_passed": 56,
            "exact_policy_records": 5,
        },
        "pure_realized_profile": {
            "passed": True,
            "tests": len(profile_test_records),
            "records": profile_test_records,
        },
        "exact_state": {"passed": True, "marker": marker},
        "n128_conformance": {"passed": True, "A": conformance_a, "B": conformance_b},
        "jsonl_absent": jsonl_absent,
        "jsonl_force": jsonl_force,
        "no_profile": no_profile,
        "n512_reference": reference512,
    }
    write_json(PACKET / "gates.json", gates)
    return gates, {128: conformance_a, 512: reference512}


def acquire_scored(
    source: dict[str, Any],
    references: dict[int, dict[str, Any]],
    model_identity: dict[str, int],
) -> dict[str, Any]:
    pairs_by_cell: dict[int, list[dict[str, Any]]] = {}
    pending: dict[int, list[dict[str, Any]]] = {}
    process_count = 0
    for tokens, orders in CELLS:
        pairs = []
        pending[tokens] = []
        for pair_index, order in enumerate(orders, 1):
            arms = {}
            for position, arm in enumerate(order, 1):
                stem = f"scored-n{tokens}-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
                row = run_single(stem, arm, tokens, source, model_identity, scored=True)
                process_count += 1
                if arm == "A" and tokens not in references:
                    references[tokens] = row
                    for waiting in pending[tokens]:
                        assert_same_output(
                            row, waiting, f"{waiting['stem']} delayed A reference"
                        )
                    pending[tokens].clear()
                if tokens in references:
                    assert_same_output(references[tokens], row, stem)
                else:
                    pending[tokens].append(row)
                arms[arm] = row
                append_jsonl(
                    PACKET / "attempts.jsonl",
                    {
                        "stage": "scored",
                        "cell_tokens": tokens,
                        "pair_index": pair_index,
                        "order": order,
                        "position": position,
                        "arm": arm,
                        "row": row,
                    },
                )
                time.sleep(COOLDOWN_SECONDS)
            pairs.append(
                {"tokens": tokens, "pair_index": pair_index, "order": order, **arms}
            )
        require(not pending[tokens], f"N{tokens} observations lack A reference")
        pairs_by_cell[tokens] = pairs
    require(process_count == 32, f"scored process count is {process_count}, not 32")
    summaries = {}
    for tokens in (128, 512):
        critical = T_N128 if tokens == 128 else T_N512
        pairs = pairs_by_cell[tokens]
        summaries[str(tokens)] = {
            "n": len(pairs),
            "critical": critical,
            "orders": [pair["order"] for pair in pairs],
            "endpoints": {
                name: summarize(
                    [float(pair["A"]["metrics"][name]) for pair in pairs],
                    [float(pair["B"]["metrics"][name]) for pair in pairs],
                    critical,
                )
                for name in ENDPOINTS
            },
            "order_strata": {
                order: {
                    name: [
                        float(pair["A"]["metrics"][name])
                        / float(pair["B"]["metrics"][name])
                        for pair in pairs
                        if pair["order"] == order
                    ]
                    for name in ENDPOINTS
                }
                for order in ("AB", "BA")
            },
        }
    result = {
        "scored_processes": process_count,
        "cells": {str(key): value for key, value in pairs_by_cell.items()},
        "summaries": summaries,
        "references": references,
    }
    write_json(PACKET / "scored-result.json", result)
    write_json(PACKET / "summaries.json", summaries)
    return result


def final_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    source = source_build_identity()
    require(source == manifest["source"], "source/build changed during packet")
    hashes = {
        "qwen": sha256_file(QWEN),
        "qwen_bench": sha256_file(QWEN_BENCH),
        "runner": sha256_file(SCRIPT),
        "prereg": sha256_file(PREREG),
        "perf_log": sha256_file(PERF_LOG),
        "perf_roadmap": sha256_file(PERF_ROADMAP),
        "prompt": sha256_file(PROMPT),
        **{f"source:{path}": sha256_file(ROOT / path) for path in SOURCE_BLOBS},
    }
    require(
        hashes == manifest["small_file_sha256"],
        "small-file identity changed during packet",
    )
    before = file_identity(MODEL)
    digest = sha256_file(MODEL)
    after = file_identity(MODEL)
    require(
        before == after == manifest["model"]["file_identity"],
        "model file identity changed",
    )
    require(
        digest == MODEL_SHA256 == manifest["model"]["sha256"],
        "final model SHA-256 drifted",
    )
    result = {
        "source": source,
        "small_file_sha256": hashes,
        "model": {"path": str(MODEL), "sha256": digest, "file_identity": after},
    }
    write_json(PACKET / "final-identity.json", result)
    return result


def decision(result: dict[str, Any]) -> dict[str, Any]:
    n128 = result["summaries"]["128"]["endpoints"]
    n512 = result["summaries"]["512"]["endpoints"]
    verdict = decision_from_summaries(n128, n512)
    go = verdict == "ADMIT_A3B_ONE_SHOT_AUTO_V1"
    return {
        "schema_version": 1,
        "verdict": verdict,
        "authority": NARROW_AUTHORITY if go else "none",
        "mandatory_committed_rollback": None if go else ROLLBACK_NOTICE,
        "n128_gates": gate_n128(n128),
        "n512_gates": gate_n512(n512),
        "external_E_and_X": "diagnostic-only; excluded from decision",
        "one_shot_consumed": True,
    }


def inventory() -> str:
    rows = []
    for path in sorted(PACKET.rglob("*")):
        if path.is_file() and path.name not in {
            "artifact-inventory.sha256",
            "packet-complete.json",
        }:
            rows.append(f"{sha256_file(path)}  {path.relative_to(PACKET)}")
    inventory_path = PACKET / "artifact-inventory.sha256"
    inventory_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return sha256_file(inventory_path)


def seal(verdict: str) -> None:
    inventory_sha = inventory()
    write_json(
        PACKET / "packet-complete.json",
        {
            "schema_version": 1,
            "verdict": verdict,
            "one_shot_consumed": True,
            "decision_sha256": sha256_file(PACKET / "decision.json"),
            "inventory_sha256": inventory_sha,
            "seal": "packet-complete",
        },
    )


def seal_invalid(error: BaseException) -> None:
    if not PACKET.exists() or (PACKET / "packet-complete.json").exists():
        return
    failure = {
        "schema_version": 1,
        "error": f"{type(error).__name__}: {error}",
        "one_shot_consumed": True,
    }
    write_json(PACKET / "failure.json", failure)
    write_json(
        PACKET / "decision.json",
        {
            "schema_version": 1,
            "verdict": "INVALID",
            "authority": "none",
            "mandatory_committed_rollback": ROLLBACK_NOTICE,
            "error": failure["error"],
            "one_shot_consumed": True,
        },
    )
    seal("INVALID")


def execute() -> None:
    require(
        not PACKET.exists(), f"sealed one-shot packet path already exists: {PACKET}"
    )
    reserved = False
    try:
        PACKET.mkdir()
        reserved = True
        precreation_raw = collect_host_snapshot()
        write_json(PACKET / "precreation.host.raw.json", precreation_raw)
        precreation = evaluate_host_snapshot(precreation_raw)
        write_json(PACKET / "precreation.host.json", precreation)
        validate_host(precreation, "precreation")
        static_check()
        assert_source_contract(acquisition=True)
        write_json(
            PACKET / "inherited-qwen-environment.json",
            {
                key: value
                for key, value in os.environ.items()
                if key.startswith("QWEN_")
            },
        )
        raw_host_snapshot("packet-start")
        build = archive_cpu_command(
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
            normalized_environment(),
            3_600,
        )
        write_json(PACKET / "build-result.json", build)
        source = source_build_identity()
        host = validate_host_device()
        before = file_identity(MODEL)
        model_hash = sha256_file(MODEL)
        after = file_identity(MODEL)
        require(
            before == after and model_hash == MODEL_SHA256,
            "initial full model identity failed",
        )
        manifest = initial_manifest(source, host, model_hash, after)
        write_json(PACKET / "manifest.json", manifest)
        gates, references = run_gates(source, after)
        result = acquire_scored(source, references, after)
        raw_host_snapshot("packet-end")
        final_identity(manifest)
        outcome = decision(result)
        outcome["mandatory_gates"] = {
            "passed": True,
            "artifacts": "gates.json",
            "names": sorted(gates),
        }
        write_json(PACKET / "decision.json", outcome)
        seal(outcome["verdict"])
        print(json_text(outcome, pretty=True))
    except BaseException as error:
        if reserved:
            seal_invalid(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="sealed v0.651 A3B one-shot GPU-greedy Auto runner"
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        static_check()
        print("v0.651 check-only: PASS")
        return
    execute()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"v0.651 failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
