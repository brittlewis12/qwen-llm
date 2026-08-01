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
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/profile/v0658_positive_temperature_sampling_attribution.py"
PREREG = ROOT / "docs/bench/v0658-positive-temperature-sampling-attribution.md"
PREDECESSOR_SCRIPT = (
    ROOT / "scripts/profile/v0657_positive_temperature_sampling_attribution.py"
)
PREDECESSOR_PREREG = (
    ROOT / "docs/bench/v0657-positive-temperature-sampling-attribution.md"
)
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
MODEL = Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
QWEN = ROOT / "target/release/qwen"
QWEN_BENCH = ROOT / "target/release/qwen-bench"
PACKET = ROOT / "target/profiles/v0658-positive-temperature-sampling-attribution-p1"

PREREG_SHA256 = "341dda990f6856c7b2e9bb619eaa87ec9d701de1c0b9291abcaf684cbbc57cad"
PREDECESSOR_PREREG_SHA256 = (
    "51aaaf5c4ddad604d539b90609d6c8b2d98201e18f22ea65cc465961108b9a1d"
)
PREDECESSOR_SCRIPT_SHA256 = (
    "f9e6f1db659f2431c380901dba3ebbdfdd44b831d166bd7502a1e5f041ca6ae7"
)
MODEL_BYTES = 22_134_528_992
MODEL_SHA256 = "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
PROMPT_BYTES = 1_891
PROMPT_SHA256 = "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
PROMPT_TOKENS = 419
PROMPT_TOKEN_SHA256 = "fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f"
PROFILE_RUNS = 6
COOLDOWN_SECONDS = 5.0
CHILD_TIMEOUT_SECONDS = 30 * 60
KNOWN_COMPETITORS = {
    "qwen",
    "qwen-bench",
    "llama-cli",
    "llama-server",
    "llama-bench",
    "ollama",
    "mlx_lm",
}
METAL_BENCHMARK_MARKERS = (
    "metal-bench",
    "metal_bench",
    "gpu-bench",
    "gpu_bench",
    "metal-capture",
    "metal_capture",
)
CHILD_ENV_ALLOWLIST = {
    "CARGO_HOME",
    "COMMAND_MODE",
    "DEVELOPER_DIR",
    "HOME",
    "LANG",
    "LOGNAME",
    "MACOSX_DEPLOYMENT_TARGET",
    "PATH",
    "RUSTUP_HOME",
    "SDKROOT",
    "SHELL",
    "TERM",
    "TMPDIR",
    "USER",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
RUNTIME_ID_RE = re.compile(r"[0-9a-f]{16}\Z")
CPU_IDLE_RE = re.compile(r"CPU usage:.*?([0-9]+(?:\.[0-9]+)?)% idle")
MEMORY_RE = re.compile(r"System-wide memory free percentage:\s*([0-9]+)%")
SWAP_USED_RE = re.compile(r"used\s*=\s*([0-9.]+)([KMG])")
REFERENCE_KEYS = {
    "schema_version",
    "request_epoch",
    "request_index",
    "tokenizer_reused",
    "pair_requested",
    "pair_id",
    "pair_request_equal",
    "pair_generated_tokens_equal",
    "prefix_cache_used",
    "build_commit",
    "build_dirty",
    "build_source_state",
    "model",
    "runtime_identity_kind",
    "runtime_model_id",
    "runtime_tokenizer_id",
    "greedy_gpu_selection_reason",
    "request_start_unix_ms",
    "runtime_and_model_load_ms",
    "stdout_sink",
    "ttft_endpoint",
    "prompt_source",
    "prompt_bytes",
    "prompt_tokens",
    "requested_tokens",
    "generated_tokens",
    "generated_token_sha256",
    "stop_reason",
    "decode_policy",
    "sampling",
    "terminal_token_target_transition_consumed",
    "no_special_tokens",
    "prefill_chunk_requested",
    "prefill_chunk_effective",
    "prefill_attention_query",
    "max_context_tokens",
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
    "transition_count",
    "transition_ms",
    "transition_tps",
    "inference_complete_ms",
    "total_request_ms",
    "pso_cache",
    "metal_allocated",
}
FROZEN_REQUEST_FIELDS = (
    "request_epoch",
    "request_index",
    "tokenizer_reused",
    "pair_requested",
    "pair_id",
    "pair_request_equal",
    "pair_generated_tokens_equal",
    "prefix_cache_used",
    "build_commit",
    "build_dirty",
    "build_source_state",
    "model",
    "runtime_identity_kind",
    "runtime_model_id",
    "runtime_tokenizer_id",
    "greedy_gpu_selection_reason",
    "stdout_sink",
    "ttft_endpoint",
    "prompt_source",
    "prompt_bytes",
    "prompt_tokens",
    "requested_tokens",
    "generated_tokens",
    "generated_token_sha256",
    "stop_reason",
    "decode_policy",
    "sampling",
    "terminal_token_target_transition_consumed",
    "no_special_tokens",
    "prefill_chunk_requested",
    "prefill_chunk_effective",
    "prefill_attention_query",
    "max_context_tokens",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def environment_commitments(environment: dict[str, str]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "value_bytes": len(value.encode()),
            "value_sha256": sha256_bytes(value.encode()),
        }
        for key, value in sorted(environment.items())
    }


def write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    require(not temporary.exists(), f"stale temporary JSON file {temporary}")
    try:
        with temporary.open("xb", buffering=0) as handle:
            handle.write(payload)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def fsync_file(path: Path) -> None:
    with path.open("rb", buffering=0) as handle:
        os.fsync(handle.fileno())


def file_observation(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False, "bytes": None, "sha256": None, "error": None}
    try:
        return {
            "present": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "error": None,
        }
    except OSError as error:
        return {"present": True, "bytes": None, "sha256": None, "error": str(error)}


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def parse_json(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def read_jsonl_one(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 1, f"expected exactly one physical JSONL row in {path}")
    require(lines[0].strip() == lines[0] and lines[0], f"invalid JSONL row in {path}")
    row = parse_json(lines[0])
    require(isinstance(row, dict), f"JSONL row in {path} is not an object")
    return row


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def runner_contract_checks() -> dict[str, bool]:
    duplicate_rejected = False
    nonfinite_rejected = False
    try:
        parse_json('{"x": 1, "x": 2}')
    except RuntimeError:
        duplicate_rejected = True
    try:
        parse_json('{"x": NaN}')
    except ValueError:
        nonfinite_rejected = True
    require(duplicate_rejected, "strict JSON parser accepted a duplicate key")
    require(nonfinite_rejected, "strict JSON parser accepted a non-finite value")
    committed_environment = environment_commitments({"API_KEY": "secret-value"})
    require(
        "secret-value" not in json.dumps(committed_environment),
        "environment commitments retained a plaintext value",
    )
    require(
        "secret-value" not in concat_command_failure(["helper"], 1, "secret-value"),
        "command failure retained plaintext output",
    )
    require(
        all(
            (key in CHILD_ENV_ALLOWLIST or key.startswith("LC_"))
            and not key.startswith("QWEN_")
            for key in child_env()
        ),
        "child environment escaped the frozen allowlist",
    )
    require(
        failure_disposition("candidate", True) == "KILL"
        and failure_disposition("environment", True) == "INVALID"
        and failure_disposition("candidate", False) == "CONSUMED_NO_AUTHORITY",
        "failure-disposition contract changed",
    )
    sample_process = {
        "pid": 1,
        "comm": "sample",
        "argv0": "sample",
        "args_bytes": 0,
        "args_sha256": sha256_bytes(b""),
    }
    ready = readiness_record("test-readiness", [sample_process], [])
    require_readiness(ready, "test-readiness")
    competitor_rejected = False
    blocked = readiness_record("test-readiness", [sample_process], [sample_process])
    try:
        require_readiness(blocked, "test-readiness")
    except RuntimeError:
        competitor_rejected = True
    require(competitor_rejected, "readiness accepted a competitor")
    authority_cases = (
        (
            {
                "workspace": False,
                "borrowed": False,
                "combined": False,
                "structural": False,
            },
            ("KILL", []),
        ),
        (
            {
                "workspace": True,
                "borrowed": False,
                "combined": True,
                "structural": True,
            },
            ("GO_BOUNDED_IMPLEMENTATION", ["workspace"]),
        ),
        (
            {
                "workspace": False,
                "borrowed": True,
                "combined": True,
                "structural": True,
            },
            ("GO_BOUNDED_IMPLEMENTATION", ["borrowed"]),
        ),
        (
            {"workspace": True, "borrowed": True, "combined": True, "structural": True},
            ("GO_BOUNDED_IMPLEMENTATION", ["workspace", "borrowed"]),
        ),
        (
            {
                "workspace": False,
                "borrowed": False,
                "combined": True,
                "structural": True,
            },
            ("GO_BOUNDED_IMPLEMENTATION", ["combined"]),
        ),
        (
            {
                "workspace": False,
                "borrowed": False,
                "combined": False,
                "structural": True,
            },
            ("GO_BOUNDED_IMPLEMENTATION", ["structural"]),
        ),
    )
    for passes, expected in authority_cases:
        require(select_authority(passes) == expected, "authority partition changed")
    return {
        "duplicate_json_rejected": duplicate_rejected,
        "nonfinite_json_rejected": nonfinite_rejected,
        "failure_dispositions_checked": True,
        "authority_partitions_checked": True,
        "environment_redaction_checked": True,
        "readiness_preflight_checked": True,
    }


def run_text(argv: list[str], *, timeout: int = 120) -> str:
    result = subprocess.run(
        argv,
        cwd=ROOT,
        env=child_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    require(
        result.returncode == 0,
        concat_command_failure(argv, result.returncode, result.stdout),
    )
    return result.stdout


def concat_command_failure(argv: list[str], returncode: int, output: str) -> str:
    encoded = output.encode()
    return (
        f"command failed {argv!r}: returncode={returncode} "
        f"output_bytes={len(encoded)} output_sha256={sha256_bytes(encoded)}"
    )


def static_checks() -> dict[str, Any]:
    require(SCRIPT.is_file(), f"missing runner {SCRIPT}")
    require(PREREG.is_file(), f"missing preregistration {PREREG}")
    require(PREDECESSOR_SCRIPT.is_file(), f"missing predecessor {PREDECESSOR_SCRIPT}")
    require(PREDECESSOR_PREREG.is_file(), f"missing predecessor {PREDECESSOR_PREREG}")
    require(PROMPT.is_file(), f"missing prompt {PROMPT}")
    require(MODEL.is_file(), f"missing model {MODEL}")
    require(QWEN.is_file(), f"missing release binary {QWEN}")
    require(QWEN_BENCH.is_file(), f"missing release binary {QWEN_BENCH}")
    require(MODEL.stat().st_size == MODEL_BYTES, "model byte size mismatch")
    require(PROMPT.stat().st_size == PROMPT_BYTES, "prompt byte size mismatch")
    require(sha256_file(PREREG) == PREREG_SHA256, "preregistration SHA-256 mismatch")
    require(
        sha256_file(PREDECESSOR_PREREG) == PREDECESSOR_PREREG_SHA256,
        "predecessor preregistration SHA-256 mismatch",
    )
    require(
        sha256_file(PREDECESSOR_SCRIPT) == PREDECESSOR_SCRIPT_SHA256,
        "predecessor runner SHA-256 mismatch",
    )
    require(sha256_file(PROMPT) == PROMPT_SHA256, "prompt SHA-256 mismatch")
    require(sha256_file(MODEL) == MODEL_SHA256, "model SHA-256 mismatch")
    qwen_env = {
        key: value for key, value in os.environ.items() if key.startswith("QWEN_")
    }
    require(
        not qwen_env,
        f"inherited QWEN_* environment must be empty: keys={sorted(qwen_env)}",
    )
    build = parse_json(run_text([str(QWEN_BENCH), "build-info", "--output", "json"]))
    require(build.get("status") == "match", f"build identity mismatch: {build}")
    require(build.get("build_dirty") is False, "build is dirty")
    require(build.get("runtime_dirty") is False, "runtime source is dirty")
    require(build.get("overrides") == [], "build identity has overrides")
    require(build.get("problems") == [], "build identity has problems")
    model_stat = MODEL.stat()
    return {
        "script_sha256": sha256_file(SCRIPT),
        "prereg_sha256": PREREG_SHA256,
        "predecessor_prereg_sha256": PREDECESSOR_PREREG_SHA256,
        "predecessor_script_sha256": PREDECESSOR_SCRIPT_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "model_sha256": MODEL_SHA256,
        "model_file_identity": {
            "device": model_stat.st_dev,
            "inode": model_stat.st_ino,
            "bytes": model_stat.st_size,
            "mtime_ns": model_stat.st_mtime_ns,
        },
        "qwen_binary_sha256": sha256_file(QWEN),
        "qwen_bench_binary_sha256": sha256_file(QWEN_BENCH),
        "build_identity": build,
        "runner_contract_checks": runner_contract_checks(),
        "parent_environment_commitments": environment_commitments(dict(os.environ)),
        "child_environment_commitments": environment_commitments(child_env()),
    }


def parse_vm_stat(text: str) -> dict[str, int]:
    first = text.splitlines()[0]
    page_match = re.search(r"page size of ([0-9]+) bytes", first)
    require(page_match is not None, "cannot parse vm_stat page size")
    result: dict[str, int] = {"page_size": int(page_match.group(1))}
    for line in text.splitlines()[1:]:
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        value = raw.strip().rstrip(".").replace(".", "")
        if value.isdigit():
            result[key.strip()] = int(value)
    return result


def swap_used_bytes(text: str) -> int:
    match = SWAP_USED_RE.search(text)
    require(match is not None, f"cannot parse vm.swapusage: {text!r}")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def vm_snapshot() -> dict[str, Any]:
    vm_text = run_text(["vm_stat"])
    swap_text = run_text(["sysctl", "-n", "vm.swapusage"])
    vm = parse_vm_stat(vm_text)
    required = {
        "Pageouts": "pageouts",
        "Compressions": "compressions",
        "Swapouts": "swapouts",
        "Pages stored in compressor": "compressor_pages_stored",
        "Pages occupied by compressor": "compressor_pages_occupied",
    }
    counters: dict[str, int] = {}
    for source, destination in required.items():
        require(source in vm, f"vm_stat is missing required counter {source!r}")
        counters[destination] = vm[source]
    return {
        "vm_stat": vm,
        "swapusage": swap_text.strip(),
        "swap_used_bytes": swap_used_bytes(swap_text),
        **counters,
    }


def process_census() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = run_text(["ps", "-axo", "pid=,comm=,args="])
    census: list[dict[str, Any]] = []
    competitors: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 2:
            continue
        pid = int(fields[0])
        comm = Path(fields[1]).name
        args = fields[2] if len(fields) == 3 else ""
        try:
            argv = shlex.split(args)
        except ValueError:
            argv = args.split()
        names = {comm}
        names.update(Path(token).name for token in argv)
        argv0 = Path(argv[0]).name if argv else comm
        observation = {
            "pid": pid,
            "comm": comm,
            "argv0": argv0,
            "args_bytes": len(args.encode()),
            "args_sha256": sha256_bytes(args.encode()),
        }
        census.append(observation)
        python_module = any(
            token == "-m"
            and index + 1 < len(argv)
            and (argv[index + 1] == "mlx_lm" or argv[index + 1].startswith("mlx_lm."))
            for index, token in enumerate(argv)
        )
        metal_benchmark = any(
            marker in name.lower()
            for marker in METAL_BENCHMARK_MARKERS
            for name in names
        )
        if pid != os.getpid() and (
            names.intersection(KNOWN_COMPETITORS)
            or python_module
            or metal_benchmark
            or any(name.startswith(("qwen-", "llama-")) for name in names)
        ):
            competitors.append(observation)
    return census, competitors


def readiness_record(
    label: str,
    census: list[dict[str, Any]],
    competitors: list[dict[str, Any]],
) -> dict[str, Any]:
    census_bytes = json.dumps(census, sort_keys=True, separators=(",", ":")).encode()
    return {
        "label": label,
        "process_census": census,
        "process_census_sha256": sha256_bytes(census_bytes),
        "competitors": competitors,
    }


def capture_readiness(label: str) -> dict[str, Any]:
    census, competitors = process_census()
    return readiness_record(label, census, competitors)


def require_readiness(result: dict[str, Any], label: str) -> None:
    require(result.get("label") == label, f"{label}: readiness label mismatch")
    census = result.get("process_census")
    competitors = result.get("competitors")
    require(isinstance(census, list), f"{label}: invalid process census")
    require(isinstance(competitors, list), f"{label}: invalid competitor list")
    census_bytes = json.dumps(census, sort_keys=True, separators=(",", ":")).encode()
    require(
        result.get("process_census_sha256") == sha256_bytes(census_bytes),
        f"{label}: process census digest mismatch",
    )
    require(
        all(competitor in census for competitor in competitors),
        f"{label}: competitor is absent from process census",
    )
    require(not competitors, f"{label}: competing inference process: {competitors}")


def cpu_idle_sample() -> float:
    text = run_text(["top", "-l", "2", "-s", "1", "-n", "0"], timeout=30)
    matches = CPU_IDLE_RE.findall(text)
    require(len(matches) >= 2, "cannot parse one-second top CPU idle sample")
    return float(matches[-1])


def host_capture(label: str) -> dict[str, Any]:
    batt = run_text(["pmset", "-g", "batt"])
    therm = run_text(["pmset", "-g", "therm"])
    memory = run_text(["memory_pressure", "-Q"])
    memory_match = MEMORY_RE.search(memory)
    require(memory_match is not None, "cannot parse memory_pressure")
    idle = [cpu_idle_sample() for _ in range(3)]
    census, competitors = process_census()
    census_bytes = json.dumps(census, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "label": label,
        "pmset_batt": batt,
        "pmset_therm": therm,
        "memory_pressure": memory,
        "memory_available_percent": int(memory_match.group(1)),
        "cpu_idle_samples": idle,
        "cpu_idle_median_percent": statistics.median(idle),
        "process_census": census,
        "process_census_sha256": sha256_bytes(census_bytes),
        "competitors": competitors,
        "ac_power": "AC Power" in batt,
        "thermal_warning": "No thermal warning level has been recorded" not in therm,
        "performance_warning": (
            "No performance warning level has been recorded" not in therm
        ),
    }
    require(result["ac_power"], f"{label}: AC power required")
    require(not result["thermal_warning"], f"{label}: thermal warning")
    require(not result["performance_warning"], f"{label}: performance warning")
    require(result["memory_available_percent"] >= 50, f"{label}: memory below 50%")
    require(result["cpu_idle_median_percent"] >= 75.0, f"{label}: CPU idle below 75%")
    require(not competitors, f"{label}: competing inference process: {competitors}")
    return result


def rusage_children() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "utime_s": usage.ru_utime,
        "stime_s": usage.ru_stime,
        "major_faults": usage.ru_majflt,
        "input_blocks": usage.ru_inblock,
        "output_blocks": usage.ru_oublock,
    }


def rusage_delta(
    before: dict[str, float | int], after: dict[str, float | int]
) -> dict[str, Any]:
    return {key: after[key] - before[key] for key in before}


def child_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if (key in CHILD_ENV_ALLOWLIST or key.startswith("LC_"))
        and not key.startswith("QWEN_")
    }


def product_argv(timing: Path, profiled: bool) -> list[str]:
    argv = [
        str(QWEN),
        "--model",
        str(MODEL),
        "--prompt-file",
        str(PROMPT.relative_to(ROOT)),
        "--tokens",
        "128",
        "--temp",
        "0.7",
        "--top-k",
        "200",
        "--top-p",
        "1.0",
        "--min-p",
        "0.05",
        "--seed",
        "42",
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
    if profiled:
        argv.append("--sampling-attribution")
    return argv


def process_group_exists(group: int) -> bool:
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for_process_group_exit(group: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while process_group_exists(group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def stop_child_group(process: subprocess.Popen[bytes]) -> list[str]:
    actions: list[str] = []
    group = process.pid
    if process.poll() is None:
        require(
            os.getpgid(process.pid) == group,
            f"refusing to signal unexpected process group for PID {process.pid}",
        )
    if process_group_exists(group):
        os.killpg(group, signal.SIGTERM)
        actions.append("SIGTERM")
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    if not wait_for_process_group_exit(group, 2.0):
        os.killpg(group, signal.SIGKILL)
        actions.append("SIGKILL")
        process.wait(timeout=10)
        require(
            wait_for_process_group_exit(group, 2.0),
            f"process group {group} survived SIGKILL",
        )
    else:
        process.wait()
    return actions


def run_child(
    label: str,
    argv: list[str],
    stdout: Path,
    stderr: Path,
    attempt: Path,
    on_spawn: Any | None = None,
    on_result: Any | None = None,
    extra_artifacts: tuple[Path, ...] = (),
) -> dict[str, Any]:
    require(
        not stdout.exists() and not stderr.exists() and not attempt.exists(),
        f"{label}: output or attempt record already exists",
    )
    started = time.monotonic_ns()
    initial = {
        "label": label,
        "argv": argv,
        "state": "spawning",
        "start_monotonic_ns": started,
    }
    write_json(attempt, initial)
    before = rusage_children()
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    leaked_process_group = False
    interrupted: BaseException | None = None
    termination_actions: list[str] = []
    try:
        with stdout.open("xb") as out_handle, stderr.open("xb") as err_handle:
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=child_env(),
                stdout=out_handle,
                stderr=err_handle,
                start_new_session=True,
            )
            group = os.getpgid(process.pid)
            require(
                group == process.pid, f"{label}: child process group is not isolated"
            )
            write_json(
                attempt,
                {
                    **initial,
                    "state": "running",
                    "pid": process.pid,
                    "process_group": group,
                },
            )
            if on_spawn is not None:
                on_spawn()
            try:
                process.wait(timeout=CHILD_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                termination_actions = stop_child_group(process)
            except BaseException as error:
                interrupted = error
                termination_actions = stop_child_group(process)
            if process_group_exists(group):
                leaked_process_group = True
                termination_actions.extend(stop_child_group(process))
            if on_result is not None:
                on_result()
            out_handle.flush()
            err_handle.flush()
            os.fsync(out_handle.fileno())
            os.fsync(err_handle.fileno())
    except BaseException as error:
        if process is not None and process_group_exists(process.pid):
            termination_actions.extend(stop_child_group(process))
        if interrupted is None:
            interrupted = error
    ended = time.monotonic_ns()
    after = rusage_children()
    artifact_status = []
    for artifact in extra_artifacts:
        status: dict[str, Any] = {
            "path": str(artifact),
            "present": artifact.is_file(),
            "fsync_error": None,
        }
        if artifact.is_file():
            try:
                fsync_file(artifact)
            except OSError as error:
                status["fsync_error"] = str(error)
        artifact_status.append(status)
    stdout_observation = file_observation(stdout)
    stderr_observation = file_observation(stderr)
    record = {
        "label": label,
        "argv": argv,
        "state": "complete",
        "pid": None if process is None else process.pid,
        "process_group": None if process is None else process.pid,
        "returncode": None if process is None else process.returncode,
        "timed_out": timed_out,
        "leaked_process_group": leaked_process_group,
        "termination_actions": termination_actions,
        "interrupted": None if interrupted is None else type(interrupted).__name__,
        "start_monotonic_ns": started,
        "end_monotonic_ns": ended,
        "wall_ms": (ended - started) / 1e6,
        "stdout": stdout_observation,
        "stderr": stderr_observation,
        "extra_artifacts": artifact_status,
        "rusage": rusage_delta(before, after),
    }
    write_json(attempt, record)
    if interrupted is not None:
        raise interrupted
    require(not timed_out, f"{label}: child timed out")
    require(not leaked_process_group, f"{label}: child leaked process-group members")
    require(process is not None, f"{label}: child did not spawn")
    require(process.returncode == 0, f"{label}: child exited {process.returncode}")
    require(
        stdout_observation["present"]
        and stdout_observation["error"] is None
        and stderr_observation["present"]
        and stderr_observation["error"] is None,
        f"{label}: child output observation failed",
    )
    failed_artifacts = [
        status
        for status in artifact_status
        if not status["present"] or status["fsync_error"] is not None
    ]
    require(
        not failed_artifacts, f"{label}: child artifact failure: {failed_artifacts}"
    )
    return record


def run_cpu_conformance() -> list[dict[str, Any]]:
    commands = (
        (
            "cpu-sampling",
            ["cargo", "test", "--locked", "-p", "qwen-llm", "sampling::", "--lib"],
        ),
        (
            "cpu-cli",
            ["cargo", "test", "--locked", "-p", "qwen-cli", "--bin", "qwen"],
        ),
    )
    records = []
    for label, argv in commands:
        record = run_child(
            label,
            argv,
            PACKET / f"{label}.stdout",
            PACKET / f"{label}.stderr",
            PACKET / f"{label}.json",
        )
        records.append(record)
    return records


def run_model_conformance() -> dict[str, Any]:
    stdout = PACKET / "model-conformance.stdout"
    stderr = PACKET / "model-conformance.stderr"
    argv = [
        "cargo",
        "test",
        "--locked",
        "--release",
        "-p",
        "qwen-llm",
        "metal_sampled_attribution_matches_production_a3b",
        "--",
        "--ignored",
        "--nocapture",
    ]
    record = run_child(
        "model-conformance",
        argv,
        stdout,
        stderr,
        PACKET / "model-conformance.attempt.json",
    )
    combined = stdout.read_text(errors="replace") + stderr.read_text(errors="replace")
    require(
        combined.count("[sampling-attribution-a3b] exact-state PASS") == 1,
        "model conformance PASS marker missing or duplicated",
    )
    return record


def require_number(value: Any, label: str, *, signed: bool = False) -> float:
    require(type(value) in (int, float), f"{label} is not numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} is non-finite")
    if not signed:
        require(number >= 0.0, f"{label} is negative")
    return number


def require_int(value: Any, label: str, expected: int | None = None) -> int:
    require(type(value) is int and value >= 0, f"{label} is not a nonnegative integer")
    if expected is not None:
        require(value == expected, f"{label} changed: {value} != {expected}")
    return value


def require_bool(value: Any, label: str, expected: bool) -> None:
    require(type(value) is bool and value is expected, f"{label} changed")


def validate_product_build(row: dict[str, Any], static: dict[str, Any]) -> None:
    build = static["build_identity"]
    require(row.get("build_commit") == build["build_commit"], "product build commit")
    require(row.get("build_dirty") == "0", "product build dirty stamp")
    require(
        row.get("build_source_state") == build["build_source_state"],
        "product build source-state stamp",
    )
    require(sha256_file(QWEN) == static["qwen_binary_sha256"], "qwen binary changed")
    require(
        sha256_file(QWEN_BENCH) == static["qwen_bench_binary_sha256"],
        "qwen-bench binary changed",
    )
    model_stat = MODEL.stat()
    require(
        {
            "device": model_stat.st_dev,
            "inode": model_stat.st_ino,
            "bytes": model_stat.st_size,
            "mtime_ns": model_stat.st_mtime_ns,
        }
        == static["model_file_identity"],
        "model file identity changed after authentication",
    )


def validate_reference(row: dict[str, Any], static: dict[str, Any]) -> None:
    require(set(row) == REFERENCE_KEYS, "reference schema-10 top keys changed")
    require_int(row.get("schema_version"), "reference schema", 10)
    require("sampling_attribution" not in row, "reference unexpectedly has attribution")
    validate_product_build(row, static)
    require(row.get("request_epoch") == "first_post_model_load", "request epoch")
    require_int(row.get("request_index"), "request index", 0)
    require_bool(row.get("tokenizer_reused"), "tokenizer reuse", False)
    require_bool(row.get("pair_requested"), "pair request", False)
    require(row.get("pair_id") is None, "pair id")
    require(row.get("pair_request_equal") is None, "pair request equality")
    require(row.get("pair_generated_tokens_equal") is None, "pair token equality")
    require_bool(row.get("prefix_cache_used"), "prefix cache use", False)
    require(row.get("model") == str(MODEL), "model path")
    require(
        row.get("runtime_identity_kind") == "metadata_compatibility_v1", "identity kind"
    )
    require(row.get("stdout_sink") == "redirected", "stdout sink")
    require(row.get("ttft_endpoint") == "stdout_flush_complete", "TTFT endpoint")
    require(row.get("prompt_source") == "file", "prompt source")
    require_int(row.get("prompt_bytes"), "reference prompt byte count", PROMPT_BYTES)
    require_int(row.get("prompt_tokens"), "reference prompt-token count", PROMPT_TOKENS)
    require_int(row.get("requested_tokens"), "reference requested-token count", 128)
    require_int(row.get("generated_tokens"), "reference generated-token count", 128)
    require_int(row.get("transition_count"), "reference transition count", 127)
    require(row.get("stop_reason") == "token_limit", "reference stop reason")
    require(row.get("decode_policy") == "sampled_cpu", "reference decode policy")
    require(
        row.get("greedy_gpu_selection_reason") == "ineligible_request",
        "greedy GPU selection reason",
    )
    require_bool(row.get("no_special_tokens"), "special-token policy", False)
    require_int(row.get("prefill_chunk_requested"), "requested prefill chunk", 1024)
    require_int(row.get("prefill_chunk_effective"), "effective chunk", PROMPT_TOKENS)
    require_int(row.get("max_context_tokens"), "context capacity", 1024)
    query = row.get("prefill_attention_query")
    require(isinstance(query, dict), "prefill query topology missing")
    require(
        set(query)
        == {"outer_chunk_rows", "query_rows", "tiled_layer_calls", "query_tile_calls"},
        "prefill query topology keys changed",
    )
    require_int(query.get("outer_chunk_rows"), "query outer chunk", PROMPT_TOKENS)
    require_int(query.get("query_rows"), "prefill query rows", PROMPT_TOKENS)
    require_int(query.get("tiled_layer_calls"), "tiled layer calls")
    require_int(query.get("query_tile_calls"), "query tile calls")
    require(
        row.get("terminal_token_target_transition_consumed") is False, "terminal state"
    )
    require(
        SHA256_RE.fullmatch(row.get("generated_token_sha256", "")) is not None,
        "token digest",
    )
    require(
        RUNTIME_ID_RE.fullmatch(row.get("runtime_model_id", "")) is not None, "model id"
    )
    require(
        RUNTIME_ID_RE.fullmatch(row.get("runtime_tokenizer_id", "")) is not None,
        "tokenizer id",
    )
    sampling = row.get("sampling")
    require(isinstance(sampling, dict), "reference sampling telemetry missing")
    require(
        set(sampling)
        == {
            "algorithm_version",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "effective_seed",
            "draws",
        },
        "sampling telemetry keys changed",
    )
    require_int(sampling.get("algorithm_version"), "sampler version", 1)
    require(
        require_number(sampling.get("temperature"), "temperature") == 0.7, "temperature"
    )
    require_int(sampling.get("top_k"), "top-k", 200)
    require(require_number(sampling.get("top_p"), "top-p") == 1.0, "top-p")
    require(require_number(sampling.get("min_p"), "min-p") == 0.05, "min-p")
    require_int(sampling.get("effective_seed"), "seed", 42)
    require_int(sampling.get("draws"), "draw count", 128)


def validate_count_summary(value: Any, label: str, calls: int) -> None:
    require(isinstance(value, dict), f"{label} is not an object")
    require(set(value) == {"total", "min", "max"}, f"{label} keys changed")
    total, minimum, maximum = value["total"], value["min"], value["max"]
    require(
        all(type(item) is int and item >= 0 for item in (total, minimum, maximum)),
        label,
    )
    require(
        minimum <= maximum and calls * minimum <= total <= calls * maximum,
        f"{label} values invalid",
    )


def validate_profile(
    row: dict[str, Any], reference: dict[str, Any], static: dict[str, Any]
) -> None:
    require_int(row.get("schema_version"), "profile schema", 11)
    require(
        set(row) == set(reference) | {"sampling_attribution"},
        "schema-11 top keys changed",
    )
    validate_product_build(row, static)
    for key in FROZEN_REQUEST_FIELDS:
        require(row.get(key) == reference.get(key), f"profile/reference {key} mismatch")
    value = row.get("sampling_attribution")
    require(isinstance(value, dict), "sampling_attribution missing")
    require(
        set(value)
        == {
            "version",
            "prompt_token_ids_sha256",
            "clock_probe",
            "sampler",
            "transitions",
            "bounds",
        },
        "sampling_attribution keys changed",
    )
    require_int(value["version"], "attribution version", 1)
    require(
        value["prompt_token_ids_sha256"] == PROMPT_TOKEN_SHA256, "prompt token digest"
    )

    clock = value["clock_probe"]
    require(
        set(clock)
        == {
            "batches",
            "iterations_per_batch",
            "pair_ns",
            "upper_pair_ns",
            "new_timer_spans",
            "observer_upper_ms",
        },
        "clock-probe keys changed",
    )
    require_int(clock["batches"], "clock batches", 7)
    require_int(clock["iterations_per_batch"], "clock iterations", 100_000)
    require_int(clock["new_timer_spans"], "clock timer spans", 1_662)
    require(
        isinstance(clock["pair_ns"], list) and len(clock["pair_ns"]) == 7, "clock pairs"
    )
    pairs = [require_number(item, "clock pair") for item in clock["pair_ns"]]
    upper = require_number(clock["upper_pair_ns"], "clock upper")
    observer = require_number(clock["observer_upper_ms"], "observer upper")
    require(upper == math.ceil(max(pairs)), "clock upper is not outward rounded")
    require(abs(observer - upper * 1_662 / 1e6) <= 1e-9, "observer upper mismatch")

    sampler = value["sampler"]
    sampler_keys = {
        "calls",
        "timer_spans",
        "input_logits_total",
        "input_logits_min",
        "input_logits_max",
        "wall_ms",
        "shape_validation_ms",
        "candidate_alloc_ms",
        "candidate_fill_ms",
        "top_k_order_ms",
        "min_p_ms",
        "positive_infinity_ms",
        "temperature_scale_ms",
        "probability_weights_ms",
        "top_p_ms",
        "categorical_ms",
        "residual_ms",
        "candidate_capacity_bytes_total",
        "candidate_capacity_bytes_peak",
        "probability_capacity_bytes_total",
        "probability_capacity_bytes_peak",
        "after_top_k",
        "after_min_p",
        "after_positive_infinity",
        "after_top_p",
        "candidate_index",
    }
    require(set(sampler) == sampler_keys, "sampler attribution keys changed")
    require_int(sampler["calls"], "sampler calls", 128)
    require_int(sampler["timer_spans"], "sampler timer spans", 1_408)
    require_int(sampler["input_logits_total"], "input-logit total", 128 * 248_320)
    require_int(sampler["input_logits_min"], "input-logit min", 248_320)
    require_int(sampler["input_logits_max"], "input-logit max", 248_320)
    for key in (
        "candidate_capacity_bytes_total",
        "candidate_capacity_bytes_peak",
        "probability_capacity_bytes_total",
        "probability_capacity_bytes_peak",
    ):
        require(type(sampler[key]) is int and sampler[key] > 0, f"sampler {key}")
    require(
        sampler["candidate_capacity_bytes_peak"]
        <= sampler["candidate_capacity_bytes_total"]
        <= sampler["calls"] * sampler["candidate_capacity_bytes_peak"],
        "candidate capacity-byte accounting",
    )
    require(
        sampler["probability_capacity_bytes_peak"]
        <= sampler["probability_capacity_bytes_total"]
        <= sampler["calls"] * sampler["probability_capacity_bytes_peak"],
        "probability capacity-byte accounting",
    )
    for key in sampler_keys - {
        "residual_ms",
        "after_top_k",
        "after_min_p",
        "after_positive_infinity",
        "after_top_p",
        "candidate_index",
    }:
        if key.endswith("_ms"):
            require_number(sampler[key], f"sampler {key}")
    require_number(sampler["residual_ms"], "sampler residual", signed=True)
    for key in (
        "after_top_k",
        "after_min_p",
        "after_positive_infinity",
        "after_top_p",
        "candidate_index",
    ):
        validate_count_summary(sampler[key], key, sampler["calls"])
    require(
        sampler["after_top_k"] == {"total": 128 * 200, "min": 200, "max": 200},
        "top-k retained-count accounting changed",
    )
    require(
        0 < sampler["after_min_p"]["total"] <= sampler["after_top_k"]["total"]
        and 0
        < sampler["after_positive_infinity"]["total"]
        <= sampler["after_min_p"]["total"]
        and sampler["after_top_p"] == sampler["after_positive_infinity"],
        "sampler filter totals are not monotonic",
    )
    require(
        sampler["candidate_index"]["max"] < sampler["after_top_p"]["max"],
        "candidate-index summary exceeds final support",
    )
    require(
        sampler["candidate_capacity_bytes_peak"] >= 248_320 * 16
        and sampler["candidate_capacity_bytes_total"]
        == sampler["calls"] * sampler["candidate_capacity_bytes_peak"],
        "candidate capacity cannot hold the frozen full-vocabulary rows",
    )
    require(
        sampler["probability_capacity_bytes_peak"]
        >= sampler["after_positive_infinity"]["max"] * 8,
        "probability capacity cannot hold the retained candidates",
    )
    phase_sum = sum(
        require_number(sampler[key], key, signed=(key == "residual_ms"))
        for key in (
            "shape_validation_ms",
            "candidate_alloc_ms",
            "candidate_fill_ms",
            "top_k_order_ms",
            "min_p_ms",
            "positive_infinity_ms",
            "temperature_scale_ms",
            "probability_weights_ms",
            "top_p_ms",
            "categorical_ms",
            "residual_ms",
        )
    )
    require(
        abs(phase_sum - sampler["wall_ms"]) <= 0.001, "sampler phases do not reconcile"
    )

    transitions = value["transitions"]
    transition_keys = {
        "calls",
        "new_timer_spans",
        "logits_bytes_per_call",
        "logits_bytes_total",
        "outer_wall_ms",
        "inner_wall_ms",
        "cpu_encode_ms",
        "completion_wait_ms",
        "gpu_ms_nested",
        "logits_alloc_zero_ms",
        "logits_copy_ms",
        "inner_residual_ms",
        "outer_wrapper_advance_ms",
    }
    require(set(transitions) == transition_keys, "transition attribution keys changed")
    require_int(transitions["calls"], "transition calls", 127)
    require_int(transitions["new_timer_spans"], "transition timer spans", 254)
    require_int(transitions["logits_bytes_per_call"], "logits bytes/call", 993_280)
    require_int(transitions["logits_bytes_total"], "logits bytes total", 126_146_560)
    for key in transition_keys - {"inner_residual_ms", "outer_wrapper_advance_ms"}:
        if key.endswith("_ms"):
            require_number(transitions[key], f"transition {key}")
    require_number(transitions["inner_residual_ms"], "inner residual", signed=True)
    require_number(
        transitions["outer_wrapper_advance_ms"], "outer residual", signed=True
    )
    inner = (
        transitions["cpu_encode_ms"]
        + transitions["completion_wait_ms"]
        + transitions["logits_alloc_zero_ms"]
        + transitions["logits_copy_ms"]
        + transitions["inner_residual_ms"]
    )
    require(abs(inner - transitions["inner_wall_ms"]) <= 0.001, "inner wall mismatch")
    outer = transitions["inner_wall_ms"] + transitions["outer_wrapper_advance_ms"]
    require(abs(outer - transitions["outer_wall_ms"]) <= 0.001, "outer wall mismatch")
    transition_ms = require_number(row.get("transition_ms"), "transition_ms")
    require(abs(outer - transition_ms) <= 0.001, "row transition mismatch")
    require(
        transitions["gpu_ms_nested"] <= transitions["completion_wait_ms"] + 0.001,
        "nested GPU wall exceeds completion wait",
    )

    bounds = value["bounds"]
    bound_keys = {
        "observer_upper_ms",
        "workspace_raw_ms",
        "workspace_adjusted_ms",
        "workspace_adjusted_fraction",
        "borrowed_raw_ms",
        "borrowed_adjusted_ms",
        "borrowed_adjusted_fraction",
        "combined_raw_ms",
        "combined_adjusted_ms",
        "combined_adjusted_fraction",
        "structural_raw_ms",
        "structural_adjusted_ms",
        "structural_adjusted_fraction",
    }
    require(set(bounds) == bound_keys, "bound keys changed")
    for key in bound_keys:
        require_number(bounds[key], f"bound {key}")
    require(
        abs(bounds["observer_upper_ms"] - observer) <= 1e-9,
        "bound observer does not match clock probe",
    )
    expected = {
        "workspace": transitions["logits_alloc_zero_ms"]
        + sampler["candidate_alloc_ms"],
        "borrowed": transitions["logits_alloc_zero_ms"] + transitions["logits_copy_ms"],
    }
    expected["combined"] = expected["borrowed"] + sampler["candidate_alloc_ms"]
    expected["structural"] = (
        expected["combined"] + sampler["candidate_fill_ms"] + sampler["top_k_order_ms"]
    )
    generation_ms = require_number(row["generation_ms"], "generation_ms")
    require(generation_ms > 0.0, "generation_ms must be positive")
    for name, raw in expected.items():
        require(abs(bounds[f"{name}_raw_ms"] - raw) <= 0.001, f"{name} raw mismatch")
        adjusted = max(0.0, raw - observer)
        require(
            abs(bounds[f"{name}_adjusted_ms"] - adjusted) <= 0.001,
            f"{name} adjusted mismatch",
        )
        require(
            abs(bounds[f"{name}_adjusted_fraction"] - adjusted / generation_ms) <= 1e-9,
            f"{name} fraction mismatch",
        )


def vm_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "swap_used_bytes",
        "pageouts",
        "compressions",
        "swapouts",
        "compressor_pages_stored",
        "compressor_pages_occupied",
    )
    delta = {key: after[key] - before[key] for key in keys}
    fatal = [
        key
        for key in (
            "swap_used_bytes",
            "compressor_pages_stored",
            "compressor_pages_occupied",
        )
        if delta[key] > 0
    ]
    delta["fatal_growth"] = fatal
    return delta


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("workspace", "borrowed", "combined", "structural")
    result: dict[str, Any] = {}
    for name in names:
        raw = [row["sampling_attribution"]["bounds"][f"{name}_raw_ms"] for row in rows]
        observer = [
            row["sampling_attribution"]["bounds"]["observer_upper_ms"] for row in rows
        ]
        adjusted = [
            row["sampling_attribution"]["bounds"][f"{name}_adjusted_ms"] for row in rows
        ]
        generation = [row["generation_ms"] for row in rows]
        fractions = [
            row["sampling_attribution"]["bounds"][f"{name}_adjusted_fraction"]
            for row in rows
        ]
        clears = [
            value >= 5.0 and fraction >= 0.05
            for value, fraction in zip(adjusted, fractions)
        ]
        result[name] = {
            "raw_ms": raw,
            "raw_ms_range": [min(raw), max(raw)],
            "observer_upper_ms": observer,
            "observer_upper_ms_range": [min(observer), max(observer)],
            "adjusted_ms": adjusted,
            "adjusted_ms_range": [min(adjusted), max(adjusted)],
            "generation_ms": generation,
            "generation_ms_range": [min(generation), max(generation)],
            "adjusted_fraction": fractions,
            "adjusted_fraction_range": [min(fractions), max(fractions)],
            "median_adjusted_ms": statistics.median(adjusted),
            "median_adjusted_fraction": statistics.median(fractions),
            "clear_count": sum(clears),
            "passes": (
                statistics.median(adjusted) >= 5.0
                and statistics.median(fractions) >= 0.05
                and sum(clears) >= 5
            ),
        }
    disposition, authority = select_authority(
        {name: result[name]["passes"] for name in names}
    )
    return {
        "bounds": result,
        "disposition": disposition,
        "authority": authority,
        "max_authorized_packets": 0 if disposition == "KILL" else 1,
    }


def select_authority(passes: dict[str, bool]) -> tuple[str, list[str]]:
    if not passes["structural"]:
        return "KILL", []
    independent = [name for name in ("workspace", "borrowed") if passes[name]]
    if independent:
        return "GO_BOUNDED_IMPLEMENTATION", independent
    if passes["combined"]:
        return "GO_BOUNDED_IMPLEMENTATION", ["combined"]
    return "GO_BOUNDED_IMPLEMENTATION", ["structural"]


def packet_inventory() -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(PACKET.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            inventory.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        except OSError as error:
            inventory.append({"name": path.name, "inventory_error": str(error)})
    return inventory


def failure_disposition(category: str, profile_result_observed: bool) -> str:
    if not profile_result_observed:
        return "CONSUMED_NO_AUTHORITY"
    return "KILL" if category == "candidate" else "INVALID"


def acquire(attested: bool) -> None:
    require(attested, "acquisition requires --attest-no-other-user-gpu-workload")
    require(not PACKET.exists(), f"packet root already exists: {PACKET}")
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    PACKET.mkdir()
    fsync_directory(PACKET.parent)
    stage = "host-readiness-capture"
    category = "environment"
    profile_state = {"spawned": False, "result_observed": False}
    decision: dict[str, Any] | None = None
    try:
        readiness_label = "before-static-and-cpu-conformance"
        readiness = capture_readiness(readiness_label)
        stage = "host-readiness-publication"
        category = "infrastructure"
        write_json(PACKET / "host-readiness.json", readiness)
        stage = "host-readiness-validation"
        category = "environment"
        require_readiness(readiness, readiness_label)

        stage = "static"
        category = "infrastructure"
        static = static_checks()
        write_json(PACKET / "static.json", static)
        stage = "vm-before"
        category = "environment"
        vm_before = vm_snapshot()
        category = "infrastructure"
        write_json(PACKET / "vm-before.json", vm_before)

        stage = "cpu-conformance"
        cpu_conformance = run_cpu_conformance()
        write_json(PACKET / "cpu-conformance.json", cpu_conformance)

        stage = "host-model-conformance"
        category = "environment"
        host = host_capture("before-model-conformance")
        category = "infrastructure"
        write_json(PACKET / "host-model-conformance.json", host)
        stage = "model-conformance"
        model_conformance = run_model_conformance()
        write_json(PACKET / "model-conformance.json", model_conformance)
        stage = "static-post-conformance"
        static_after = static_checks()
        require(static_after == static, "identity changed during conformance")
        write_json(PACKET / "static-post-conformance.json", static_after)
        stage = "cooldown-before-reference"
        category = "infrastructure"
        time.sleep(COOLDOWN_SECONDS)

        stage = "host-reference"
        category = "environment"
        reference_host = host_capture("before-reference")
        category = "infrastructure"
        write_json(PACKET / "host-reference.json", reference_host)
        reference_timing = PACKET / "reference.timing.jsonl"
        stage = "reference-child"
        reference = run_child(
            "reference",
            product_argv(reference_timing, False),
            PACKET / "reference.stdout",
            PACKET / "reference.stderr",
            PACKET / "reference.attempt.json",
            extra_artifacts=(reference_timing,),
        )
        stage = "reference-validation"
        reference_row = read_jsonl_one(reference_timing)
        validate_reference(reference_row, static)
        reference["timing_sha256"] = sha256_file(reference_timing)
        reference["timing"] = reference_row
        write_json(PACKET / "reference.json", reference)

        rows: list[dict[str, Any]] = []
        children: list[dict[str, Any]] = []
        reference_stdout = (PACKET / "reference.stdout").read_bytes()
        for index in range(PROFILE_RUNS):
            stage = f"cooldown-profile-{index}"
            category = "infrastructure"
            time.sleep(COOLDOWN_SECONDS)
            stage = f"host-profile-{index}"
            category = "environment"
            capture = host_capture(f"before-profile-{index}")
            category = "infrastructure"
            write_json(PACKET / f"host-profile-{index}.json", capture)
            timing = PACKET / f"profile-{index}.timing.jsonl"
            stdout = PACKET / f"profile-{index}.stdout"
            stage = f"profile-{index}-child"
            category = "candidate"
            child = run_child(
                f"profile-{index}",
                product_argv(timing, True),
                stdout,
                PACKET / f"profile-{index}.stderr",
                PACKET / f"profile-{index}.attempt.json",
                on_spawn=lambda: profile_state.__setitem__("spawned", True),
                on_result=lambda: profile_state.__setitem__("result_observed", True),
                extra_artifacts=(timing,),
            )
            stage = f"profile-{index}-validation"
            row = read_jsonl_one(timing)
            validate_profile(row, reference_row, static)
            require(
                stdout.read_bytes() == reference_stdout,
                f"profile-{index}: stdout mismatch",
            )
            child["timing_sha256"] = sha256_file(timing)
            child["timing"] = row
            write_json(PACKET / f"profile-{index}.json", child)
            rows.append(row)
            children.append(child)

        stage = "host-after-profiles"
        category = "environment"
        final_host = host_capture("after-profiles")
        category = "infrastructure"
        write_json(PACKET / "host-after.json", final_host)
        stage = "vm-after"
        category = "environment"
        vm_after = vm_snapshot()
        category = "infrastructure"
        write_json(PACKET / "vm-after.json", vm_after)
        delta = vm_delta(vm_before, vm_after)
        category = "environment"
        require(not delta["fatal_growth"], f"fatal VM growth: {delta}")
        category = "infrastructure"
        stage = "reduction"
        decision = {
            "schema": "qwen-v0658-sampling-attribution/v1",
            "host_readiness": readiness,
            "static": static,
            "cpu_conformance": cpu_conformance,
            "model_conformance": model_conformance,
            "reference": reference,
            "profiles": children,
            "vm_before": vm_before,
            "vm_after": vm_after,
            "vm_delta": delta,
            "reduction": summarize(rows),
            "user_gpu_attestation": True,
        }
        stage = "decision-publication"
        write_json(PACKET / "decision.json", decision)
    except BaseException as error:
        disposition = failure_disposition(category, profile_state["result_observed"])
        failure = {
            "schema": "qwen-v0658-sampling-attribution-failure/v1",
            "stage": stage,
            "disposition": disposition,
            "authority": [],
            "failure_category": category,
            "profile_spawned": profile_state["spawned"],
            "profile_result_observed": profile_state["result_observed"],
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "packet_inventory_before_terminal_record": packet_inventory(),
            "user_gpu_attestation": True,
        }
        write_json(PACKET / "failure.json", failure)
        write_json(PACKET / "decision.json", failure)
        raise
    assert decision is not None
    try:
        print(json.dumps(decision["reduction"], indent=2, sort_keys=True))
    except BrokenPipeError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("check-only", "acquire"), required=True)
    parser.add_argument("--attest-no-other-user-gpu-workload", action="store_true")
    args = parser.parse_args()
    if args.phase == "check-only":
        print(json.dumps(static_checks(), indent=2, sort_keys=True))
        return 0
    acquire(args.attest_no_other_user_gpu_workload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"v0.658 runner failed: {error}", file=sys.stderr)
        raise
