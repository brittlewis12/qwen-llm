#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
MODEL = Path("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf")
BINARY = ROOT / "target/release/qwen-bench"
PREREG = ROOT / "docs/bench/v0644-q4-matmat-attribution.md"
SCRIPT = Path(__file__).resolve()
KERNEL = ROOT / "kernels/mat_mat_q4_k.metal"
PACKET = ROOT / "target/profiles/v0644-q4-matmat-attribution-p1"

MODEL_BYTES = 16_817_244_384
MODEL_SHA256 = "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0"
TENSOR_SHA256 = "966f5cd8316a4f8f4baf141be13465747440c0f97846816ff4c5351980cc59d9"
HISTORICAL_A_TFLOPS = 13.22195
CHARGED_GATE = 1.132304
MATERIAL_GATE = 1.01
SIMULTANEOUS_T = 2.70
POWER_Z = 0.842
MAX_RELATIVE_MDE = 0.005
N_IN = 5120
N_OUT = 17408
N_QUERY = 1024
NOMINAL_FLOPS = 2 * N_IN * N_OUT * N_QUERY
TENSOR_BYTES = 50_135_040
GUARD_ELEMENTS = 4096
GUARD_BITS = struct.unpack("<I", struct.pack("<f", -1234.5))[0]
COMPETITOR_PATTERN = re.compile(
    r"(?:^|[/\s])(?:qwen(?:-bench|-cli)?|llama(?:-cli|-server|-bench)?|"
    r"mlx(?:[_-]lm)?(?:\.[\w.-]+)?|ollama(?:[_-][\w.-]+)?)(?:[\s/]|$)",
    re.IGNORECASE,
)

ARMS = (
    "a_production",
    "b_source_segments_live_no_dequant",
    "c_no_source_no_dequant",
    "e0_mma_only",
    "e8_mma_only_tgm8_cap_matched",
)
SEQUENCES = (
    (ARMS[0], ARMS[1], ARMS[4], ARMS[2], ARMS[3]),
    (ARMS[1], ARMS[2], ARMS[0], ARMS[3], ARMS[4]),
    (ARMS[2], ARMS[3], ARMS[1], ARMS[4], ARMS[0]),
    (ARMS[3], ARMS[4], ARMS[2], ARMS[0], ARMS[1]),
    (ARMS[4], ARMS[0], ARMS[3], ARMS[1], ARMS[2]),
    (ARMS[3], ARMS[2], ARMS[4], ARMS[1], ARMS[0]),
    (ARMS[4], ARMS[3], ARMS[0], ARMS[2], ARMS[1]),
    (ARMS[0], ARMS[4], ARMS[1], ARMS[3], ARMS[2]),
    (ARMS[1], ARMS[0], ARMS[2], ARMS[4], ARMS[3]),
    (ARMS[2], ARMS[1], ARMS[3], ARMS[0], ARMS[4]),
)

CONTRASTS = {
    "a_over_b": (ARMS[0], ARMS[1]),
    "b_over_c": (ARMS[1], ARMS[2]),
    "c_over_e0": (ARMS[2], ARMS[3]),
    "c_over_e8": (ARMS[2], ARMS[4]),
    "a_over_c": (ARMS[0], ARMS[2]),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(
    argv: list[str],
    *,
    timeout: int = 600,
    check: bool = True,
    env: dict[str, str] | None = None,
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


def archive_command(
    stem: str,
    argv: list[str],
    *,
    env: dict[str, str],
    timeout: int = 600,
    check: bool = True,
    after_return: Callable[[subprocess.CompletedProcess[str]], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    started_ns = time.time_ns()
    try:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
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
        raise RuntimeError(
            f"command timed out after {timeout}s: {' '.join(argv)}"
        ) from error
    except OSError as error:
        (PACKET / f"{stem}.stdout.txt").write_text("")
        (PACKET / f"{stem}.stderr.txt").write_text(str(error) + "\n")
        write_json(
            PACKET / f"{stem}.command.json",
            {
                "argv": argv,
                "elapsed_ms": (time.time_ns() - started_ns) / 1e6,
                "returncode": None,
                "spawn_error": f"{type(error).__name__}: {error}",
                "timed_out": False,
                "timeout_seconds": timeout,
            },
        )
        raise RuntimeError(f"command could not start: {' '.join(argv)}") from error

    callback_error = None
    if after_return is not None:
        try:
            after_return(result)
        except BaseException as error:
            callback_error = error

    (PACKET / f"{stem}.stdout.txt").write_text(result.stdout)
    (PACKET / f"{stem}.stderr.txt").write_text(result.stderr)
    write_json(
        PACKET / f"{stem}.command.json",
        {
            "argv": argv,
            "elapsed_ms": (time.time_ns() - started_ns) / 1e6,
            "returncode": result.returncode,
            "spawn_error": None,
            "timed_out": False,
            "timeout_seconds": timeout,
        },
    )
    if callback_error is not None:
        raise callback_error
    if check:
        require(
            result.returncode == 0,
            f"command failed ({result.returncode}): {' '.join(argv)}",
        )
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def hash_stable_file(path: Path) -> tuple[dict[str, int], str]:
    before = file_identity(path)
    digest = sha256_file(path)
    after = file_identity(path)
    require(before == after, f"file changed while hashing: {path}")
    return after, digest


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def canonical_environment() -> tuple[dict[str, str], list[str]]:
    env = dict(os.environ)
    removed = []
    build_keys = {
        "AR",
        "CARGO_BUILD_TARGET",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_INCREMENTAL",
        "CARGO_TARGET_DIR",
        "CC",
        "CFLAGS",
        "CXX",
        "CXXFLAGS",
        "DEVELOPER_DIR",
        "LDFLAGS",
        "MACOSX_DEPLOYMENT_TARGET",
        "RUSTC",
        "RUSTC_BOOTSTRAP",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTDOCFLAGS",
        "RUSTFLAGS",
        "SDKROOT",
    }
    for key in sorted(list(env)):
        if (
            key.startswith(("QWEN_", "MTL_", "METAL_", "DYLD_"))
            or key.startswith(("CARGO_BUILD_", "CARGO_PROFILE_", "CARGO_TARGET_"))
            or key == "RUST_LOG"
            or key in build_keys
        ):
            removed.append(key)
            env.pop(key)
    env["RUST_LOG"] = "warn"
    return env, removed


def process_rows() -> list[dict[str, Any]]:
    result = run(["ps", "-axo", "pid=,ppid=,command="])
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        rows.append({"pid": pid, "ppid": ppid, "command": fields[2]})
    return rows


def quiet_snapshot() -> dict[str, Any]:
    thermal = run(["pmset", "-g", "therm"]).stdout
    pressure = run(["memory_pressure", "-Q"]).stdout
    match = re.search(r"System-wide memory free percentage: (\d+)%", pressure)
    require(match is not None, "memory_pressure output did not contain free percentage")
    rows = process_rows()
    parent_by_pid = {row["pid"]: row["ppid"] for row in rows}
    excluded = {os.getpid()}
    cursor = os.getpid()
    while cursor in parent_by_pid and parent_by_pid[cursor] > 0:
        cursor = parent_by_pid[cursor]
        if cursor in excluded:
            break
        excluded.add(cursor)
    competitors = [
        row
        for row in rows
        if row["pid"] not in excluded and COMPETITOR_PATTERN.search(row["command"])
    ]
    free_percent = int(match.group(1))
    return {
        "thermal": thermal,
        "memory_pressure": pressure,
        "memory_free_percent": free_percent,
        "excluded_pids": sorted(excluded),
        "competitors": competitors,
        "valid": (
            "No thermal warning level has been recorded" in thermal
            and "No performance warning level has been recorded" in thermal
            and free_percent >= 90
            and not competitors
        ),
    }


def function_body(llvm: str, symbol: str) -> str:
    marker = f"define void @{symbol}("
    start = llvm.index(marker)
    next_definition = llvm.find("\ndefine ", start + len(marker))
    return llvm[start:] if next_definition < 0 else llvm[start:next_definition]


def inspect_llvm(llvm: str) -> dict[str, Any]:
    b_symbol = "kernel_mat_mat_q4_K_f32_n64_source_segments_live_no_dequant"
    c_symbol = "kernel_mat_mat_q4_K_f32_n64_no_source_no_dequant"
    b = function_body(llvm, b_symbol)
    c = function_body(llvm, c_symbol)
    volatile_lines = [
        line for line in b.splitlines() if "load volatile <4 x i32>" in line
    ]
    volatile_results = [line.strip().split(" =", 1)[0] for line in volatile_lines]
    load_a = b.index("icmp ult i16 %7, 128")
    first_volatile = b.index("load volatile <4 x i32>")
    first_threadgroup_barrier = b.index("@air.wg.barrier")
    source_derivation = re.search(
        r"getelementptr inbounds i8, i8 addrspace\(1\)\* %1,", b
    )
    c_header = c.splitlines()[0]
    checks = {
        "b_two_volatile_128_bit_loads": len(volatile_lines) == 2,
        "b_volatile_loads_align_16": all("align 16" in line for line in volatile_lines),
        "b_volatile_results_dead": all(
            len(re.findall(rf"(?<![A-Za-z0-9_.]){re.escape(result)}(?!\d)", b)) == 1
            for result in volatile_results
        ),
        "b_source_a_derivation_present": source_derivation is not None,
        "b_source_a_has_one_body_derivation": (
            len(re.findall(r"(?<!\d)%1(?!\d)", b)) == 2
        ),
        "b_loads_after_load_a_before_first_barrier": (
            source_derivation is not None
            and load_a
            < source_derivation.start()
            < first_volatile
            < first_threadgroup_barrier
        ),
        "b_no_dequant_call": "dequantize_q4_K_half" not in b,
        "b_two_threadgroup_barrier_callsites": b.count("@air.wg.barrier") == 2,
        "b_three_simdgroup_barrier_callsites": b.count("@air.simdgroup.barrier") == 3,
        "b_two_matrix_load_callsites": b.count("@air.simdgroup_matrix_8x8_load") == 2,
        "b_one_mma_callsite": b.count("@air.simdgroup_matrix_8x8_multiply_accumulate")
        == 1,
        "b_mma_is_half_half_to_float": b.count(
            "@air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32"
        )
        == 1,
        "b_one_store_callsite": b.count("@air.simdgroup_matrix_8x8_store") == 1,
        "c_source_argument_readnone": (
            'i8 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %1'
            in c_header
        ),
        "c_source_argument_unused_in_body": len(re.findall(r"(?<!\d)%1(?!\d)", c)) == 1,
        "c_no_volatile_load": "load volatile" not in c,
        "c_no_dequant_call": "dequantize_q4_K_half" not in c,
        "c_two_threadgroup_barrier_callsites": c.count("@air.wg.barrier") == 2,
        "c_three_simdgroup_barrier_callsites": c.count("@air.simdgroup.barrier") == 3,
        "c_two_matrix_load_callsites": c.count("@air.simdgroup_matrix_8x8_load") == 2,
        "c_one_mma_callsite": c.count("@air.simdgroup_matrix_8x8_multiply_accumulate")
        == 1,
        "c_mma_is_half_half_to_float": c.count(
            "@air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32"
        )
        == 1,
        "c_one_store_callsite": c.count("@air.simdgroup_matrix_8x8_store") == 1,
    }
    require(all(checks.values()), f"optimized LLVM contract failed: {checks}")
    return checks


def write_inventory(status: str) -> None:
    inventory = {}
    for path in sorted(PACKET.iterdir()):
        if path.is_file() and path.name != "inventory.json":
            inventory[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    write_json(
        PACKET / "inventory.json",
        {"schema_version": 1, "status": status, "files": inventory},
    )


def sample_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def require_close(
    label: str,
    actual: float,
    expected: float,
    *,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> None:
    require(
        math.isfinite(actual)
        and math.isclose(
            actual,
            expected,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ),
        f"{label} drifted: {actual} != {expected}",
    )


def validate_build_identity(identity: dict[str, Any], expected: dict[str, Any]) -> None:
    require(
        identity == expected, "result build identity differs from pre-run build-info"
    )
    require(identity.get("schema_version") == 2, "wrong build-identity schema")
    require(identity.get("status") == "match", "build identity is not clean/matching")
    require(identity.get("problems") == [], "build identity reports problems")
    require(identity.get("overrides") == [], "canonical row used a build override")
    require(identity.get("build_dirty") is False, "binary was built dirty")
    require(identity.get("runtime_dirty") is False, "runtime worktree is dirty")
    require(
        identity.get("build_commit") == identity.get("runtime_commit"),
        "build/runtime commit mismatch",
    )
    require(
        identity.get("build_source_state") == identity.get("runtime_source_state"),
        "build/runtime source-state mismatch",
    )


def validate_row_contract(row: dict[str, Any], expected_build: dict[str, Any]) -> None:
    require(row.get("schema_version") == 2, "wrong result schema")
    require(row.get("test") == "q4_matmat_attribution", "wrong test name")
    require(
        row.get("claim_scope")
        == "production-grid synthetic attribution bounds; no production authority",
        "claim scope drifted",
    )
    validate_build_identity(row["build_identity"], expected_build)

    expected_model = {
        "path": str(MODEL),
        "primary_file_bytes": MODEL_BYTES,
        "total_mapped_bytes": MODEL_BYTES,
        "shards": 1,
        "shard_mapped_bytes": [MODEL_BYTES],
        "architecture_matches_qwen3_27b": True,
    }
    require(row.get("model") == expected_model, "model metadata drifted")
    require(
        row.get("tensor")
        == {
            "name": "blk.0.ffn_gate.weight",
            "dtype": "Q4_K",
            "shape": [N_IN, N_OUT],
            "bytes": TENSOR_BYTES,
            "sha256": TENSOR_SHA256,
        },
        "tensor metadata drifted",
    )
    require(
        row.get("geometry")
        == {
            "n_in": N_IN,
            "n_out": N_OUT,
            "n_query": N_QUERY,
            "grid": [16, 272, 1],
            "threads_per_threadgroup": 256,
            "simdgroups_per_threadgroup": 8,
            "k_step": 32,
            "k_loops": 160,
            "inner_substeps": 4,
            "mma_calls_per_substep_per_simdgroup": 8,
            "nominal_flops": NOMINAL_FLOPS,
        },
        "geometry drifted",
    )

    expected_arms = [
        {
            "label": ARMS[0],
            "kernel": "kernel_mat_mat_q4_K_f32_n64",
            "dynamic_tgm_bytes": 8192,
        },
        {
            "label": ARMS[1],
            "kernel": ("kernel_mat_mat_q4_K_f32_n64_source_segments_live_no_dequant"),
            "dynamic_tgm_bytes": 8192,
            "qualification": (
                "two aligned volatile source-segment proxies; "
                "not exact production load timing"
            ),
        },
        {
            "label": ARMS[2],
            "kernel": "kernel_mat_mat_q4_K_f32_n64_no_source_no_dequant",
            "dynamic_tgm_bytes": 8192,
        },
        {
            "label": ARMS[3],
            "kernel": "kernel_mat_mat_q4_K_f32_n64_mma_ceiling",
            "dynamic_tgm_bytes": 0,
        },
        {
            "label": ARMS[4],
            "kernel": "kernel_mat_mat_q4_K_f32_n64_mma_ceiling_tgm8",
            "dynamic_tgm_bytes": 8192,
            "qualification": "8-KiB-cap-matched, not occupancy-equivalent",
        },
    ]
    require(row.get("arms") == expected_arms, "arm metadata drifted")

    pipeline_rows = row.get("pipeline_reflection")
    require(isinstance(pipeline_rows, list), "pipeline reflection is not a list")
    require(len(pipeline_rows) == len(expected_arms), "pipeline-row count drifted")
    expected_pipeline = {
        arm["kernel"]: arm["dynamic_tgm_bytes"] for arm in expected_arms
    }
    seen_pipeline = set()
    for pipeline in pipeline_rows:
        require(
            set(pipeline)
            == {
                "kernel",
                "dynamic_tgm_bytes",
                "thread_execution_width",
                "max_total_threads_per_threadgroup",
                "static_threadgroup_memory_length",
                "supports_indirect_command_buffers",
            },
            "pipeline-row fields drifted",
        )
        kernel = pipeline["kernel"]
        require(kernel in expected_pipeline, f"unexpected pipeline row {kernel}")
        require(kernel not in seen_pipeline, f"duplicate pipeline row {kernel}")
        seen_pipeline.add(kernel)
        require(
            pipeline["dynamic_tgm_bytes"] == expected_pipeline[kernel],
            f"dynamic TGM drifted for {kernel}",
        )
        require(pipeline["thread_execution_width"] == 32, "SIMD width drifted")
        require(
            pipeline["max_total_threads_per_threadgroup"] >= 256,
            "pipeline cannot admit the frozen 256-thread grid",
        )
        require(
            isinstance(pipeline["static_threadgroup_memory_length"], int)
            and pipeline["static_threadgroup_memory_length"] >= 0,
            "invalid static TGM reflection",
        )
        require(
            isinstance(pipeline["supports_indirect_command_buffers"], bool),
            "invalid ICB reflection",
        )
    require(seen_pipeline == set(expected_pipeline), "pipeline membership drifted")

    measurement = row.get("measurement")
    require(
        set(measurement)
        == {
            "warmups_per_arm",
            "sequence_repeats",
            "samples_per_arm",
            "sequences",
            "sequence_order",
            "unscored_wash_in",
            "nonce",
            "qwen_matmat_q4_k_n64_env",
            "one_dispatch_per_command_buffer",
            "primary_clock",
            "wall_clock",
            "command_status_required",
        },
        "measurement fields drifted",
    )
    require(measurement["warmups_per_arm"] == 12, "warmup drift")
    require(measurement["sequence_repeats"] == 6, "sequence-repeat drift")
    require(measurement["samples_per_arm"] == 60, "sample-count drift")
    require(
        measurement["sequences"] == [list(sequence) for sequence in SEQUENCES],
        "sequence drift",
    )
    require(measurement["nonce"] == 1, "nonce drifted")
    require(
        measurement["sequence_order"]
        == "ten-sequence Williams design, rotated by repeat",
        "sequence-order description drifted",
    )
    require(
        measurement["unscored_wash_in"]
        == "one first-arm dispatch before every sequence",
        "wash-in description drifted",
    )
    require(measurement["qwen_matmat_q4_k_n64_env"] is None, "N64 env leaked")
    require(measurement["one_dispatch_per_command_buffer"] is True, "dispatch drift")
    require(
        measurement["primary_clock"] == "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "primary clock drifted",
    )
    require(
        measurement["wall_clock"]
        == "commit through waitUntilCompleted; encoding excluded",
        "wall-clock scope drifted",
    )
    require(
        measurement["command_status_required"] == "Completed with no error",
        "command-status contract drifted",
    )
    require(
        row.get("guards")
        == {
            "elements_each_side": GUARD_ELEMENTS,
            "value_bits": GUARD_BITS,
            "intact_after_validation_and_measurement": True,
        },
        "guard contract drifted",
    )


def contrast_row(
    blocks: list[dict[str, float]], control: str, candidate: str
) -> dict[str, Any]:
    logs = [math.log(block[control] / block[candidate]) for block in blocks]
    n = len(logs)
    avg = statistics.mean(logs)
    sd = sample_stdev(logs)
    se = sd / math.sqrt(n)
    half_width = SIMULTANEOUS_T * se
    mde_log = (SIMULTANEOUS_T + POWER_Z) * se
    result = {
        "control": control,
        "candidate": candidate,
        "blocks": n,
        "geometric_speedup": math.exp(avg),
        "simultaneous_lcb": math.exp(avg - half_width),
        "simultaneous_ucb": math.exp(avg + half_width),
        "log_sample_stdev": sd,
        "relative_mde": math.exp(mde_log) - 1.0,
        "log_ratios": logs,
    }
    require(
        all(
            math.isfinite(value)
            for key, value in result.items()
            if key not in {"control", "candidate", "blocks", "log_ratios"}
        )
        and all(math.isfinite(value) for value in logs),
        f"non-finite contrast for {control}/{candidate}",
    )
    return result


def analyze(row: dict[str, Any], expected_build: dict[str, Any]) -> dict[str, Any]:
    validate_row_contract(row, expected_build)

    validation = row["validation"]
    validation_fields = {
        "production_all_finite",
        "production_nonzero",
        "production_repeat_bit_exact",
        "production_sha256",
        "b_exact_analytic",
        "c_exact_analytic",
        "b_c_bit_exact",
        "b_c_repeat_bit_exact",
        "b_c_alternate_nonce",
        "b_c_nonce_changes_output",
        "b_c_half_activation_exact",
        "timed_activation_restored_to_one",
        "e0_exact_analytic",
        "e8_exact_analytic",
        "e0_e8_bit_exact",
        "e_repeat_bit_exact",
        "nonce",
        "meaningful_nonce_bits",
        "alternate_a_nonce",
        "alternate_b_nonce",
        "both_nonce_bits_change_output",
        "elements_checked_per_arm",
    }
    require(set(validation) == validation_fields, "validation fields drifted")
    required_validation = (
        "production_all_finite",
        "production_nonzero",
        "production_repeat_bit_exact",
        "b_exact_analytic",
        "c_exact_analytic",
        "b_c_bit_exact",
        "b_c_repeat_bit_exact",
        "b_c_nonce_changes_output",
        "b_c_half_activation_exact",
        "timed_activation_restored_to_one",
        "e0_exact_analytic",
        "e8_exact_analytic",
        "e0_e8_bit_exact",
        "e_repeat_bit_exact",
        "both_nonce_bits_change_output",
    )
    require(
        all(validation.get(key) is True for key in required_validation),
        "validation gate failed",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", validation["production_sha256"]) is not None,
        "invalid production-output SHA-256",
    )
    require(validation["nonce"] == 1, "validation nonce drifted")
    require(validation["b_c_alternate_nonce"] == 0, "B/C alternate nonce drifted")
    require(validation["alternate_a_nonce"] == 0, "E A nonce drifted")
    require(validation["alternate_b_nonce"] == 3, "E B nonce drifted")
    require(validation["meaningful_nonce_bits"] == [0, 1], "nonce bits drifted")
    require(
        validation["elements_checked_per_arm"] == N_QUERY * N_OUT,
        "validated element count drifted",
    )

    samples = row["samples"]
    require(len(samples) == 300, "expected 300 scored samples")
    sample_fields = {
        "ordinal",
        "repeat",
        "sequence",
        "position",
        "dispatch_predecessor",
        "sequence_wash_in",
        "arm",
        "gpu_ms",
        "wall_ms",
        "nominal_tflops",
    }
    for block_index in range(60):
        repeat = block_index // 10
        block = block_index % 10
        sequence_index = (block + repeat) % 10
        expected_order = SEQUENCES[sequence_index]
        block_samples = samples[block_index * 5 : (block_index + 1) * 5]
        for position, (sample, expected_arm) in enumerate(
            zip(block_samples, expected_order)
        ):
            require(set(sample) == sample_fields, "sample fields drifted")
            require(
                sample["ordinal"] == block_index * 5 + position + 1,
                "sample ordinal drifted",
            )
            require(sample["repeat"] == repeat + 1, "sample repeat drifted")
            require(sample["sequence"] == sequence_index + 1, "sample sequence drifted")
            require(sample["position"] == position + 1, "sample position drifted")
            require(sample["arm"] == expected_arm, "sample arm order drifted")
            expected_predecessor = (
                expected_order[0] if position == 0 else expected_order[position - 1]
            )
            require(
                sample["dispatch_predecessor"] == expected_predecessor,
                "dispatch predecessor drifted",
            )
            require(
                sample["sequence_wash_in"] is (position == 0), "wash-in marker drifted"
            )
            require(
                math.isfinite(sample["wall_ms"]) and sample["wall_ms"] > 0,
                "invalid wall time",
            )
            require(
                math.isfinite(sample["nominal_tflops"])
                and sample["nominal_tflops"] > 0,
                "invalid nominal TFLOP/s",
            )
            require_close(
                "sample nominal TFLOP/s",
                sample["nominal_tflops"],
                NOMINAL_FLOPS / (sample["gpu_ms"] * 1e9),
                relative_tolerance=1e-12,
            )
    counts = {arm: 0 for arm in ARMS}
    blocks_by_key: dict[tuple[int, int], dict[str, float]] = {}
    for sample in samples:
        arm = sample["arm"]
        require(arm in counts, f"unknown arm {arm}")
        counts[arm] += 1
        require(
            sample["gpu_ms"] > 0 and math.isfinite(sample["gpu_ms"]), "invalid GPU time"
        )
        key = (sample["repeat"], sample["sequence"])
        block = blocks_by_key.setdefault(key, {})
        require(arm not in block, f"duplicate arm in block {key}")
        block[arm] = sample["gpu_ms"]
    require(
        all(count == 60 for count in counts.values()), f"arm counts drifted: {counts}"
    )
    require(len(blocks_by_key) == 60, "expected 60 complete sequence blocks")
    require(
        all(set(block) == set(ARMS) for block in blocks_by_key.values()),
        "incomplete block",
    )
    blocks = [blocks_by_key[key] for key in sorted(blocks_by_key)]

    contrasts = {
        name: contrast_row(blocks, control, candidate)
        for name, (control, candidate) in CONTRASTS.items()
    }
    mde_valid = all(
        row["relative_mde"] <= MAX_RELATIVE_MDE for row in contrasts.values()
    )
    summary_rows = row["summaries"]
    require(len(summary_rows) == len(ARMS), "summary count drifted")
    summaries = {summary["arm"]: summary for summary in summary_rows}
    require(len(summaries) == len(ARMS), "duplicate summary arm")
    require(set(summaries) == set(ARMS), "summary membership drifted")
    summary_fields = {
        "arm",
        "samples",
        "mean_gpu_ms",
        "median_gpu_ms",
        "sample_stdev_gpu_ms",
        "mean_nominal_tflops",
        "median_nominal_tflops",
        "sample_stdev_nominal_tflops",
    }
    samples_by_arm = {
        arm: [sample for sample in samples if sample["arm"] == arm] for arm in ARMS
    }
    for arm, summary in summaries.items():
        require(set(summary) == summary_fields, f"summary fields drifted for {arm}")
        arm_samples = samples_by_arm[arm]
        gpu_values = [sample["gpu_ms"] for sample in arm_samples]
        tflops_values = [sample["nominal_tflops"] for sample in arm_samples]
        require(summary["samples"] == 60, f"summary sample count drifted for {arm}")
        for field, expected in (
            ("mean_gpu_ms", statistics.mean(gpu_values)),
            ("median_gpu_ms", statistics.median(gpu_values)),
            ("sample_stdev_gpu_ms", statistics.stdev(gpu_values)),
            ("mean_nominal_tflops", statistics.mean(tflops_values)),
            ("median_nominal_tflops", statistics.median(tflops_values)),
            ("sample_stdev_nominal_tflops", statistics.stdev(tflops_values)),
        ):
            require_close(f"{arm} {field}", summary[field], expected)

    a_mean_gpu_ms = statistics.mean(
        sample["gpu_ms"] for sample in samples_by_arm[ARMS[0]]
    )
    a_tflops = NOMINAL_FLOPS / (a_mean_gpu_ms * 1e9)
    a_health_ratio = a_tflops / HISTORICAL_A_TFLOPS
    a_health_valid = abs(a_health_ratio - 1.0) <= 0.01

    ab = contrasts["a_over_b"]
    bc = contrasts["b_over_c"]
    ac = contrasts["a_over_c"]
    ce_material = (
        contrasts["c_over_e0"]["simultaneous_lcb"] >= MATERIAL_GATE
        or contrasts["c_over_e8"]["simultaneous_lcb"] >= MATERIAL_GATE
    )
    valid = mde_valid and a_health_valid
    if not valid:
        verdict = "INCONCLUSIVE_RESOLUTION_OR_A_HEALTH"
        authority = "none"
    elif ab["simultaneous_lcb"] >= CHARGED_GATE:
        verdict = "OPEN_SEMANTIC_DEQUANT_DESIGN"
        authority = "design-only"
    elif (
        ac["simultaneous_lcb"] >= CHARGED_GATE
        and bc["simultaneous_lcb"] >= MATERIAL_GATE
    ):
        verdict = "PRIORITIZE_SOURCE_PROXY_DESIGN"
        authority = "design-only"
    elif (
        ab["simultaneous_ucb"] < CHARGED_GATE and ac["simultaneous_ucb"] < CHARGED_GATE
    ):
        verdict = "CLOSE_TESTED_BC_ABLATION_LANE"
        authority = "closure-only"
    else:
        verdict = "ATTRIBUTION_ONLY_NO_IMPLEMENTATION_AUTHORITY"
        authority = "none"

    return {
        "schema_version": 2,
        "verdict": verdict,
        "authority": authority,
        "fresh_a_from_mean_gpu_tflops": a_tflops,
        "fresh_a_over_historical": a_health_ratio,
        "fresh_a_health_valid": a_health_valid,
        "mde_valid": mde_valid,
        "contrasts": contrasts,
        "material_flags": {
            "a_to_b": ab["simultaneous_lcb"] >= MATERIAL_GATE,
            "b_to_c": bc["simultaneous_lcb"] >= MATERIAL_GATE,
            "c_to_e": ce_material,
            "a_to_b_clears_charged_gate": ab["simultaneous_lcb"] >= CHARGED_GATE,
            "a_to_c_clears_charged_gate": ac["simultaneous_lcb"] >= CHARGED_GATE,
        },
        "qualifications": [
            "A/E performance was observed in a disclosed dirty-tree pilot and is replication only.",
            "Contrasts are non-additive synthetic bounds.",
            "B/C source segments are transaction proxies, not exact production timing.",
            "No verdict authorizes a production kernel or changes default dispatch.",
        ],
    }


def main() -> int:
    require(
        not PACKET.exists(), f"packet already exists: {PACKET}; no retry is authorized"
    )
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    PACKET.mkdir()
    phase = "sealed"
    try:
        phase = "preflight"
        require(PREREG.is_file(), f"missing preregistration {PREREG}")
        require(SCRIPT.is_file(), f"missing runner {SCRIPT}")
        require(KERNEL.is_file(), f"missing kernel {KERNEL}")
        require(MODEL.is_file(), f"missing model {MODEL}")
        require(MODEL.stat().st_size == MODEL_BYTES, "model size drifted")
        require(
            run(["git", "status", "--porcelain", "--untracked-files=all"]).stdout == "",
            "worktree must be clean",
        )
        head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
        script_sha256 = sha256_file(SCRIPT)
        prereg_sha256 = sha256_file(PREREG)
        kernel_sha256 = sha256_file(KERNEL)
        model_identity, model_sha256 = hash_stable_file(MODEL)
        require(model_sha256 == MODEL_SHA256, "model SHA-256 drifted")
        write_json(
            PACKET / "started.json",
            {
                "schema_version": 2,
                "head": head,
                "model": str(MODEL),
                "model_bytes": MODEL_BYTES,
                "model_sha256": MODEL_SHA256,
                "script_sha256": script_sha256,
                "prereg_sha256": prereg_sha256,
                "kernel_sha256": kernel_sha256,
                "started_unix_ns": time.time_ns(),
            },
        )
        write_json(
            PACKET / "model-before.json",
            {
                "schema_version": 1,
                "identity": model_identity,
                "sha256": model_sha256,
            },
        )

        env, removed_environment = canonical_environment()
        write_json(
            PACKET / "environment.json",
            {
                "schema_version": 1,
                "removed_names": removed_environment,
                "path": env.get("PATH"),
                "rust_log": "warn",
            },
        )

        phase = "build"
        archive_command(
            "build",
            ["cargo", "build", "--release", "-p", "qwen-cli", "--bin", "qwen-bench"],
            timeout=1_200,
            env=env,
        )
        build_info_result = archive_command(
            "build-info",
            ["target/release/qwen-bench", "build-info", "-o", "json"],
            env=env,
        )
        build_info = json.loads(build_info_result.stdout)
        validate_build_identity(build_info, build_info)
        require(build_info["build_commit"] == head, "binary build commit drifted")
        require(build_info["runtime_commit"] == head, "runtime commit drifted")
        write_json(PACKET / "build-info.json", build_info)
        binary_identity, binary_sha256 = hash_stable_file(BINARY)
        write_json(
            PACKET / "binary-before.json",
            {
                "schema_version": 1,
                "identity": binary_identity,
                "sha256": binary_sha256,
            },
        )

        phase = "toolchain-identity"
        archive_command("rustc-version", ["rustc", "--version", "--verbose"], env=env)
        archive_command("cargo-version", ["cargo", "--version"], env=env)
        archive_command("xcode-select", ["xcode-select", "-p"], env=env)
        archive_command("xcodebuild-version", ["xcodebuild", "-version"], env=env)
        archive_command("sdk-path", ["xcrun", "--show-sdk-path"], env=env)
        archive_command(
            "metal-find", ["xcrun", "-sdk", "macosx", "-f", "metal"], env=env
        )
        archive_command("metallib-find", ["xcrun", "-f", "metallib"], env=env)
        archive_command("metal-objdump-find", ["xcrun", "-f", "metal-objdump"], env=env)

        phase = "static-compilation"
        llvm_path = PACKET / "mat_mat_q4_k.optimized.ll"
        air_path = PACKET / "mat_mat_q4_k.air"
        metallib_path = PACKET / "mat_mat_q4_k.metallib"
        archive_command(
            "metal-llvm",
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-S",
                "-emit-llvm",
                "-O3",
                "-ffast-math",
                str(KERNEL),
                "-o",
                str(llvm_path),
            ],
            env=env,
        )
        archive_command(
            "metal-air",
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                "-O3",
                "-ffast-math",
                str(KERNEL),
                "-o",
                str(air_path),
            ],
            env=env,
        )
        archive_command(
            "metallib",
            ["xcrun", "metallib", str(air_path), "-o", str(metallib_path)],
            env=env,
        )
        archive_command(
            "reflection",
            [
                "xcrun",
                "metal-objdump",
                "--metallib",
                "--reflection",
                str(metallib_path),
            ],
            env=env,
        )
        llvm_checks = inspect_llvm(llvm_path.read_text())
        write_json(PACKET / "llvm-checks.json", llvm_checks)
        archive_command(
            "metal-version",
            ["xcrun", "metal", "--version"],
            env=env,
        )

        phase = "quiet-preflight"
        pre_quiet = quiet_snapshot()
        write_json(PACKET / "quiet-before-cooldown.json", pre_quiet)
        require(pre_quiet["valid"], "quiet-box preflight failed")
        time.sleep(10)
        launch_quiet = quiet_snapshot()
        write_json(PACKET / "quiet-at-launch.json", launch_quiet)
        require(launch_quiet["valid"], "quiet-box launch gate failed")

        phase = "launch-seal"
        launch_head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
        launch_status = run(
            ["git", "status", "--porcelain", "--untracked-files=all"]
        ).stdout
        launch_seal = {
            "schema_version": 1,
            "head": launch_head,
            "worktree_porcelain": launch_status,
            "script_sha256": sha256_file(SCRIPT),
            "prereg_sha256": sha256_file(PREREG),
            "kernel_sha256": sha256_file(KERNEL),
            "model_identity": file_identity(MODEL),
            "binary_identity": file_identity(BINARY),
            "binary_sha256": sha256_file(BINARY),
        }
        write_json(PACKET / "launch-seal.json", launch_seal)
        require(launch_head == head, "HEAD changed before launch")
        require(launch_status == "", "worktree changed before launch")
        require(launch_seal["script_sha256"] == script_sha256, "runner changed")
        require(launch_seal["prereg_sha256"] == prereg_sha256, "prereg changed")
        require(launch_seal["kernel_sha256"] == kernel_sha256, "kernel changed")
        require(launch_seal["model_identity"] == model_identity, "model stat changed")
        require(
            launch_seal["binary_identity"] == binary_identity, "binary stat changed"
        )
        require(launch_seal["binary_sha256"] == binary_sha256, "binary changed")

        command = [
            "target/release/qwen-bench",
            "q4-mma-ceiling",
            "-m",
            str(MODEL),
            "--warmups",
            "12",
            "--sequence-repeats",
            "6",
            "--nonce",
            "1",
        ]
        phase = "benchmark"
        post_quiet_holder: dict[str, Any] = {}

        def capture_post_quiet(_: subprocess.CompletedProcess[str]) -> None:
            post_quiet_holder["snapshot"] = quiet_snapshot()

        result = archive_command(
            "benchmark",
            command,
            timeout=600,
            env=env,
            check=False,
            after_return=capture_post_quiet,
        )
        phase = "post-quiet"
        post_quiet = post_quiet_holder["snapshot"]
        write_json(PACKET / "quiet-after.json", post_quiet)
        require(
            result.returncode == 0,
            f"benchmark failed ({result.returncode}); see benchmark.stderr.txt",
        )

        phase = "post-identity"
        post_binary_identity, post_binary_sha256 = hash_stable_file(BINARY)
        post_model_identity, post_model_sha256 = hash_stable_file(MODEL)
        post_head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
        post_status = run(
            ["git", "status", "--porcelain", "--untracked-files=all"]
        ).stdout
        post_seal = {
            "schema_version": 1,
            "head": post_head,
            "worktree_porcelain": post_status,
            "script_sha256": sha256_file(SCRIPT),
            "prereg_sha256": sha256_file(PREREG),
            "kernel_sha256": sha256_file(KERNEL),
        }
        write_json(PACKET / "post-seal.json", post_seal)
        write_json(
            PACKET / "model-after.json",
            {
                "schema_version": 1,
                "identity": post_model_identity,
                "sha256": post_model_sha256,
            },
        )
        write_json(
            PACKET / "binary-after.json",
            {
                "schema_version": 1,
                "identity": post_binary_identity,
                "sha256": post_binary_sha256,
            },
        )
        require(post_head == head, "HEAD changed during benchmark")
        require(post_status == "", "worktree changed during benchmark")
        require(post_seal["script_sha256"] == script_sha256, "runner changed")
        require(post_seal["prereg_sha256"] == prereg_sha256, "prereg changed")
        require(post_seal["kernel_sha256"] == kernel_sha256, "kernel changed")
        require(post_model_identity == model_identity, "model stat changed")
        require(post_model_sha256 == MODEL_SHA256, "model SHA-256 changed")
        require(post_binary_identity == binary_identity, "binary stat changed")
        require(post_binary_sha256 == binary_sha256, "binary SHA-256 changed")
        phase = "post-quiet-gate"
        require(post_quiet["valid"], "quiet-box post-child gate failed")

        phase = "analysis"
        row = json.loads(result.stdout)
        write_json(PACKET / "raw.json", row)
        analysis = analyze(row, build_info)
        write_json_atomic(PACKET / "decision.json", analysis)

        phase = "success-artifact-seal"
        command_stems = {
            "build",
            "build-info",
            "metal-llvm",
            "metal-air",
            "metallib",
            "reflection",
            "metal-version",
            "rustc-version",
            "cargo-version",
            "xcode-select",
            "xcodebuild-version",
            "sdk-path",
            "metal-find",
            "metallib-find",
            "metal-objdump-find",
            "benchmark",
        }
        expected_artifacts = {
            "started.json",
            "environment.json",
            "model-before.json",
            "binary-before.json",
            "build-info.json",
            "mat_mat_q4_k.optimized.ll",
            "mat_mat_q4_k.air",
            "mat_mat_q4_k.metallib",
            "llvm-checks.json",
            "quiet-before-cooldown.json",
            "quiet-at-launch.json",
            "launch-seal.json",
            "quiet-after.json",
            "post-seal.json",
            "model-after.json",
            "binary-after.json",
            "raw.json",
            "decision.json",
        }
        for stem in command_stems:
            expected_artifacts.update(
                {
                    f"{stem}.command.json",
                    f"{stem}.stdout.txt",
                    f"{stem}.stderr.txt",
                }
            )
        actual_artifacts = {path.name for path in PACKET.iterdir() if path.is_file()}
        require(
            actual_artifacts == expected_artifacts,
            "success artifact set drifted: "
            f"missing={sorted(expected_artifacts - actual_artifacts)} "
            f"extra={sorted(actual_artifacts - expected_artifacts)}",
        )
        write_inventory("success")
        print(json.dumps(analysis, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        previous_decision = None
        decision_path = PACKET / "decision.json"
        if decision_path.is_file():
            try:
                previous_decision = json.loads(decision_path.read_text())
            except Exception:
                previous_decision = {"unreadable": True}
        write_json_atomic(
            decision_path,
            {
                "schema_version": 2,
                "verdict": "INCONCLUSIVE_TERMINAL_PACKET_FAILURE",
                "authority": "none",
                "failed_phase": phase,
                "retry_authorized": False,
                "terminal": True,
            },
        )
        write_json(
            PACKET / "failure.json",
            {
                "schema_version": 2,
                "error": str(error),
                "error_type": type(error).__name__,
                "failed_phase": phase,
                "authority": "none",
                "retry_authorized": False,
                "terminal": True,
                "failed_unix_ns": time.time_ns(),
                "pre_failure_decision": previous_decision,
            },
        )
        write_inventory("failure")
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"v0.644 failed: {error}", file=sys.stderr)
        raise
