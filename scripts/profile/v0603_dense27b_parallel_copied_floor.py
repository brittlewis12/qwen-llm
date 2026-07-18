#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

import v0593_demand_paged_no_copy as host_common
import v0602_a3b_parallel_copied_loader as protocol


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0603-dense27b-parallel-copied-floor-p1"
PREREG = ROOT / "docs/bench/v0603-dense27b-parallel-copied-floor.md"
DENSE_EVIDENCE = ROOT / "docs/bench/v0603-dense27b-floor-describe.json"
A3B_EVIDENCE = ROOT / "docs/bench/v0603-a3b-floor-describe.json"
MODEL = Path("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf")
BINARY = ROOT / "target/release/qwen-bench"
HOST_HELPER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
PROTOCOL_HELPER = ROOT / "scripts/profile/v0602_a3b_parallel_copied_loader.py"

EXPECTED_MODEL_SHA256 = (
    "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0"
)
EXPECTED_PROFILE = "dense27b-q4km-v1"
EXPECTED_ARCHITECTURE = "qwen35"
EXPECTED_DESCRIPTOR_DIGEST = "0xd116405fd99f54d9"
EXPECTED_INVENTORY_DIGEST = (
    "50e9af4e4f590fc85687a71f5602ce035e7fdf0e2a31e928b2c7a2be10458a07"
)
EXPECTED_REQUEST_COUNT = 851
EXPECTED_COPY_BYTES = 16_806_250_496
EXPECTED_SHARD_LENGTHS = [16_817_244_384]
EXPECTED_PAGE_SIZE = 16_384
EXPECTED_ALIGNMENT = 32
EXPECTED_MAX_BUFFER_LENGTH = 77_309_411_328
EXPECTED_DEVICE = "Apple M4 Max"
EXPECTED_ARCH_TUPLE = {
    "kind": "dense",
    "n_layer": 64,
    "hidden_size": 5120,
    "intermediate_size": 17_408,
    "vocab_size": 248_320,
    "full_attention_interval": 4,
    "n_q_heads": 24,
    "n_kv_heads": 4,
    "attn_head_dim": 256,
    "rope_theta": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "gdn_n_v_heads": 48,
    "gdn_n_k_heads": 16,
    "gdn_head_dim": 128,
    "gdn_conv_kernel": 4,
    "expert_count": 0,
    "expert_used_count": 0,
    "expert_feed_forward_length": 0,
    "expert_shared_feed_forward_length": 0,
    "mtp_n_hidden_layers": 0,
}
EXPECTED_SCHEDULE = {
    "workers": 4,
    "cuts": [136, 377, 618],
    "task_counts": [136, 241, 241, 233],
    "worker_bytes": [4_194_110_464, 4_214_375_808, 4_204_933_376, 4_192_830_848],
    "max_to_ideal": 1.0030496234726596,
    "max_to_min": 1.0051385235372128,
    "partitions": [
        {
            "start": 0,
            "end": 136,
            "task_count": 136,
            "bytes": 4_194_110_464,
            "first": {
                "request_index": 2,
                "name": "output.weight",
                "shard_idx": 0,
                "source_offset": 10_993_888,
                "n_bytes": 1_042_944_000,
            },
            "last": {
                "request_index": 135,
                "name": "blk.9.ssm_norm.weight",
                "shard_idx": 0,
                "source_offset": 4_205_103_840,
                "n_bytes": 512,
            },
        },
        {
            "start": 136,
            "end": 377,
            "task_count": 241,
            "bytes": 4_214_375_808,
            "first": {
                "request_index": 136,
                "name": "blk.9.ssm_out.weight",
                "shard_idx": 0,
                "source_offset": 4_205_104_352,
                "n_bytes": 21_626_880,
            },
            "last": {
                "request_index": 379,
                "name": "blk.28.attn_qkv.weight",
                "shard_idx": 0,
                "source_offset": 8_376_472_160,
                "n_bytes": 43_008_000,
            },
        },
        {
            "start": 377,
            "end": 618,
            "task_count": 241,
            "bytes": 4_204_933_376,
            "first": {
                "request_index": 378,
                "name": "blk.28.ffn_down.weight",
                "shard_idx": 0,
                "source_offset": 8_419_480_160,
                "n_bytes": 73_113_600,
            },
            "last": {
                "request_index": 618,
                "name": "blk.46.ffn_down.weight",
                "shard_idx": 0,
                "source_offset": 12_551_299_936,
                "n_bytes": 73_113_600,
            },
        },
        {
            "start": 618,
            "end": 851,
            "task_count": 233,
            "bytes": 4_192_830_848,
            "first": {
                "request_index": 616,
                "name": "blk.46.ffn_gate.weight",
                "shard_idx": 0,
                "source_offset": 12_624_413_536,
                "n_bytes": 50_135_040,
            },
            "last": {
                "request_index": 844,
                "name": "blk.63.post_attention_norm.weight",
                "shard_idx": 0,
                "source_offset": 16_817_223_904,
                "n_bytes": 20_480,
            },
        },
    ],
}
EXPECTED_RESOURCE_MODES = {
    "creation_storage": "shared",
    "creation_cpu_cache": "default_cache",
    "creation_hazard_tracking": "default",
    "observed_storage": "shared",
    "observed_cpu_cache": "default_cache",
    "observed_hazard_tracking": "tracked",
}

BLOCK_ORDERS = ("AB", "BA", "BA", "AB", "AB", "BA")
ARM_NAMES = {"A": "copied", "B": "parallel-copied"}
COOLDOWN_S = 30.0
HOST_SAMPLE_LIMIT = 6
HOST_SAMPLE_INTERVAL_S = 30.0
U64_MAX = (1 << 64) - 1


class InconclusivePacket(RuntimeError):
    def __init__(
        self, child: str, reasons: list[str], *, category: str = "validity"
    ) -> None:
        super().__init__(f"invalid sole child {child}: {reasons}")
        self.child = child
        self.reasons = reasons
        self.category = category


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def parse_json(text: str) -> object:
    return json.loads(text, parse_constant=reject_json_constant)


def json_text(value: object, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        indent=2 if pretty else None,
        sort_keys=True,
        allow_nan=False,
    )


def finite_number(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite")
    if positive and number <= 0:
        raise RuntimeError(f"{label} is not positive")
    if nonnegative and number < 0:
        raise RuntimeError(f"{label} is negative")
    return number


def unsigned_integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} is not an integer")
    if value < 0 or value > U64_MAX or (positive and value == 0):
        raise RuntimeError(f"{label} is outside unsigned 64-bit range")
    return value


def exact_unsigned(value: object, expected: int, label: str) -> None:
    if unsigned_integer(value, label) != expected:
        raise RuntimeError(f"{label} drifted")


def validate_architecture_tuple(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(EXPECTED_ARCH_TUPLE):
        raise RuntimeError("architecture tuple keys drifted")
    for key, expected in EXPECTED_ARCH_TUPLE.items():
        actual = value[key]
        if isinstance(expected, int):
            exact_unsigned(actual, expected, f"architecture tuple {key}")
        elif isinstance(expected, float):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, float)
                or actual != expected
            ):
                raise RuntimeError(f"architecture tuple {key} drifted")
        elif actual != expected or not isinstance(actual, type(expected)):
            raise RuntimeError(f"architecture tuple {key} drifted")


def validate_typed_json(actual: object, expected: object, label: str) -> None:
    if isinstance(expected, bool):
        if type(actual) is not bool or actual is not expected:
            raise RuntimeError(f"{label} drifted")
    elif isinstance(expected, int):
        exact_unsigned(actual, expected, label)
    elif isinstance(expected, float):
        if type(actual) is not float or actual != expected:
            raise RuntimeError(f"{label} drifted")
    elif isinstance(expected, str):
        if type(actual) is not str or actual != expected:
            raise RuntimeError(f"{label} drifted")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"{label} list shape drifted")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            validate_typed_json(left, right, f"{label}[{index}]")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"{label} object keys drifted")
        for key, value in expected.items():
            validate_typed_json(actual[key], value, f"{label}.{key}")
    elif actual is not None or expected is not None:
        raise RuntimeError(f"{label} has unsupported expected type")


def command_text(command: list[str], env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def required_manifest_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PREREG,
        DENSE_EVIDENCE,
        A3B_EVIDENCE,
        HOST_HELPER,
        PROTOCOL_HELPER,
        MODEL,
        BINARY,
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    for path in required_manifest_paths()[:-2]:
        command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    commit = command_text(["git", "rev-parse", "HEAD"]).strip()
    dirty = command_text(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"]
    ).strip()
    if dirty:
        raise RuntimeError(f"tracked source is dirty: {dirty!r}")
    build = parse_json(command_text([str(BINARY), "build-info", "--output", "json"]))
    if not isinstance(build, dict):
        raise RuntimeError("build identity is not an object")
    if (
        build.get("build_commit") != commit
        or build.get("runtime_commit") != commit
        or build.get("status") != "match"
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
    ):
        raise RuntimeError(f"source/build identity mismatch: {build}")
    return commit, build


def child_environment_record(env: dict[str, str]) -> dict[str, object]:
    digest = hashlib.sha256()
    for key, value in sorted(env.items()):
        key_bytes = key.encode()
        value_bytes = value.encode()
        digest.update(len(key_bytes).to_bytes(8, "little"))
        digest.update(key_bytes)
        digest.update(len(value_bytes).to_bytes(8, "little"))
        digest.update(value_bytes)
    controls = {
        key: value
        for key, value in sorted(env.items())
        if key.startswith(("QWEN_", "METAL_", "MTL_")) or key == "RUST_LOG"
    }
    if controls:
        raise RuntimeError(f"normalized environment retains controls: {controls}")
    return {
        "schema": 1,
        "complete_sha256": digest.hexdigest(),
        "keys": sorted(env),
        "performance_controls": controls,
    }


def schedule_projection(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("schedule is not an object")
    partitions = value.get("partitions")
    if not isinstance(partitions, list) or len(partitions) != 4:
        raise RuntimeError("schedule partitions drifted")
    projected_partitions = []
    for index, partition in enumerate(partitions):
        if not isinstance(partition, dict):
            raise RuntimeError(f"schedule partition {index} is not an object")
        projected_partitions.append(
            {
                key: partition.get(key)
                for key in ("start", "end", "task_count", "bytes", "first", "last")
            }
        )
    return {
        "workers": value.get("workers"),
        "cuts": value.get("cuts"),
        "task_counts": value.get("task_counts"),
        "worker_bytes": value.get("worker_bytes"),
        "max_to_ideal": value.get("max_to_ideal"),
        "max_to_min": value.get("max_to_min"),
        "partitions": projected_partitions,
    }


def validate_schedule(value: object) -> None:
    projection = schedule_projection(value)
    validate_typed_json(projection, EXPECTED_SCHEDULE, "dense frozen schedule")


def validate_profile_projection(
    row: dict[str, object], build: dict[str, object]
) -> None:
    exact_unsigned(row.get("schema_version"), 2, "floor schema")
    if type(row.get("model")) is not str or row.get("model") != str(MODEL):
        raise RuntimeError("model path drifted")
    if (
        type(row.get("architecture")) is not str
        or row.get("architecture") != EXPECTED_ARCHITECTURE
    ):
        raise RuntimeError("architecture drifted")
    validate_architecture_tuple(row.get("architecture_tuple"))
    if row.get("tied_embeddings") is not False or row.get("mtp_present") is not False:
        raise RuntimeError("attachment state drifted")
    validate_typed_json(
        row.get("shard_mapped_lengths"), EXPECTED_SHARD_LENGTHS, "shards"
    )
    if row.get("descriptor_layout_digest") != EXPECTED_DESCRIPTOR_DIGEST:
        raise RuntimeError("descriptor digest drifted")
    if row.get("inventory_digest") != EXPECTED_INVENTORY_DIGEST:
        raise RuntimeError("inventory digest drifted")
    exact_unsigned(row.get("request_count"), EXPECTED_REQUEST_COUNT, "request count")
    exact_unsigned(row.get("logical_copy_bytes"), EXPECTED_COPY_BYTES, "logical bytes")
    exact_unsigned(row.get("page_size"), EXPECTED_PAGE_SIZE, "page size")
    exact_unsigned(row.get("required_alignment"), EXPECTED_ALIGNMENT, "alignment")
    exact_unsigned(
        row.get("max_buffer_length"), EXPECTED_MAX_BUFFER_LENGTH, "max buffer length"
    )
    if (
        row.get("device_name") != EXPECTED_DEVICE
        or row.get("unified_memory") is not True
    ):
        raise RuntimeError("Metal device geometry drifted")
    if row.get("native_quant_embedding") is not True:
        raise RuntimeError("native embedding policy drifted")
    if row.get("native_quant_embedding_supported") is not True:
        raise RuntimeError("native embedding support drifted")
    if row.get("native_quant_embedding_selection") != "production-auto-promoted":
        raise RuntimeError("native embedding selection drifted")
    validate_typed_json(row.get("build_identity"), build, "child build identity")


def describe_control(
    base_env: dict[str, str], build: dict[str, object]
) -> dict[str, object]:
    command = [
        str(BINARY),
        "gguf-arena-floor",
        "--model",
        str(MODEL),
        "--describe",
        "--output",
        "json",
    ]
    value = parse_json(command_text(command, env=base_env))
    if not isinstance(value, dict):
        raise RuntimeError("describe control is not an object")
    validate_profile_projection(value, build)
    if type(value.get("mode")) is not str or value.get("mode") != "describe":
        raise RuntimeError("describe mode drifted")
    if value.get("materialization_supported") is not True:
        raise RuntimeError("describe does not support materialization")
    if value.get("materialization_environment_admissible") is not True:
        raise RuntimeError("describe environment is not admissible")
    if (
        type(value.get("matched_profile")) is not str
        or value.get("matched_profile") != EXPECTED_PROFILE
    ):
        raise RuntimeError("describe profile match drifted")
    validate_schedule(value.get("parallel_copy_schedule"))
    validate_schedule(value.get("computed_schedule"))
    validate_schedule(value.get("frozen_schedule"))
    if value.get("parallel_copy_schedule") != value.get("computed_schedule"):
        raise RuntimeError("describe schedule alias drifted")
    capability = value.get("usage_capability")
    if not isinstance(capability, dict):
        raise RuntimeError("usage capability is missing")
    if (
        capability.get("getrusage") is not True
        or capability.get("proc_pid_rusage_v4") is not True
    ):
        raise RuntimeError("usage capability is unavailable")
    for key in (
        "sample_minor_faults",
        "sample_major_faults",
        "sample_instructions_raw",
        "sample_cycles_raw",
        "sample_billed_energy_raw",
        "sample_serviced_energy_raw",
    ):
        unsigned_integer(capability.get(key), f"usage capability {key}")
    return {"command": command, "output": value}


def preflight_time_resources(base_env: dict[str, str]) -> dict[str, object]:
    result = subprocess.run(
        ["/usr/bin/time", "-l", "/usr/bin/true"],
        cwd=ROOT,
        env=base_env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("/usr/bin/time -l preflight failed")
    resources = protocol.process_resources(result.stderr)
    return {"stderr": result.stderr, "resources": resources}


def build_manifest(
    removed_environment: list[str], base_env: dict[str, str]
) -> dict[str, object]:
    commit, build = source_and_build_identity()
    hashes = {
        str(path): host_common.sha256_file(path) for path in required_manifest_paths()
    }
    if hashes[str(MODEL)] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA-256 drifted")
    describe = describe_control(base_env, build)
    vm_state = protocol.capture_vm_state()
    if vm_state["capture_errors"]:
        raise RuntimeError(f"VM preflight failed: {vm_state['capture_errors']}")
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "build_identity": build,
        "removed_environment": removed_environment,
        "child_environment": child_environment_record(base_env),
        "sha256": hashes,
        "model_size_bytes": MODEL.stat().st_size,
        "model_sha256": hashes[str(MODEL)],
        "profile": EXPECTED_PROFILE,
        "block_orders": list(BLOCK_ORDERS),
        "cooldown_s": COOLDOWN_S,
        "host_sample_limit": HOST_SAMPLE_LIMIT,
        "host_sample_interval_s": HOST_SAMPLE_INTERVAL_S,
        "os_product_version": command_text(["sw_vers", "-productVersion"]).strip(),
        "os_build_version": command_text(["sw_vers", "-buildVersion"]).strip(),
        "describe_control": describe,
        "time_preflight": preflight_time_resources(base_env),
        "vm_preflight": vm_state,
    }


def verify_non_model_identity(manifest: dict[str, object]) -> None:
    commit, build = source_and_build_identity()
    if commit != manifest["source_commit"] or build != manifest["build_identity"]:
        raise RuntimeError("packet source/build identity drifted")
    expected = {
        path: digest
        for path, digest in manifest["sha256"].items()
        if path != str(MODEL)
    }
    actual = {
        str(path): host_common.sha256_file(path)
        for path in required_manifest_paths()
        if path != MODEL
    }
    if actual != expected:
        raise RuntimeError("packet non-model hashes drifted")


def verify_packet_identity(manifest: dict[str, object]) -> None:
    verify_non_model_identity(manifest)
    if host_common.sha256_file(MODEL) != manifest["model_sha256"]:
        raise RuntimeError("packet completion model hash drifted")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_fsync(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json_text(row) + "\n")
        output.flush()
        os.fsync(output.fileno())


def reserve_artifact(manifest: dict[str, object]) -> None:
    preparing = ARTIFACT.with_name(f"{ARTIFACT.name}.preparing")
    if preparing.exists() or ARTIFACT.exists():
        raise RuntimeError("packet directory or preparation directory already exists")
    preparing.mkdir(parents=True, exist_ok=False)
    with (preparing / "manifest.json").open("x", encoding="utf-8") as output:
        output.write(json_text(manifest, pretty=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    fsync_directory(preparing)
    os.replace(preparing, ARTIFACT)
    fsync_directory(ARTIFACT.parent)


def capture_valid_host_samples(
    label: str, manifest: dict[str, object]
) -> list[dict[str, object]]:
    samples = []
    for index in range(HOST_SAMPLE_LIMIT):
        verify_non_model_identity(manifest)
        sample = protocol.capture_host_state()
        samples.append(sample)
        if sample["valid"]:
            return samples
        if index + 1 < HOST_SAMPLE_LIMIT:
            time.sleep(HOST_SAMPLE_INTERVAL_S)
    raise InconclusivePacket(
        label, [f"{label}_host_sampler_exhausted"], category="host"
    )


def warm_model_file() -> tuple[float, int, str]:
    started = time.perf_counter()
    total = 0
    digest = hashlib.sha256()
    buffer = bytearray(8 * 1024 * 1024)
    with MODEL.open("rb", buffering=0) as handle:
        while True:
            count = handle.readinto(buffer)
            if count == 0:
                break
            total += count
            digest.update(memoryview(buffer)[:count])
    return (time.perf_counter() - started) * 1e3, total, digest.hexdigest()


def pressure_interval(
    label: str, before: dict[str, object], after: dict[str, object]
) -> dict[str, object]:
    value = protocol.vm_interval(label, before, after)
    reasons = list(value["failure_reasons"])
    deltas = value["deltas"]
    for gauge in ("compressor_stored_pages", "compressor_occupied_pages"):
        delta = deltas.get(gauge)
        if isinstance(delta, int) and delta > 0:
            reasons.append(f"{label}_{gauge}_growth")
    return {**value, "failure_reasons": reasons}


def condition_for_child(stem: str, manifest: dict[str, object]) -> dict[str, object]:
    pre_cache_samples = capture_valid_host_samples(f"{stem}_pre_cache", manifest)
    vm_before_cache = protocol.capture_vm_state()
    cache_ms, cache_bytes, cache_sha256 = warm_model_file()
    if cache_bytes != manifest["model_size_bytes"]:
        raise RuntimeError("cache read byte count drifted")
    if cache_sha256 != manifest["model_sha256"]:
        raise RuntimeError("cache read model hash drifted")
    time.sleep(COOLDOWN_S)
    prelaunch_samples = capture_valid_host_samples(f"{stem}_prelaunch", manifest)
    vm_before_spawn = protocol.capture_vm_state()
    cache_interval = pressure_interval("cache", vm_before_cache, vm_before_spawn)
    if cache_interval["failure_reasons"]:
        raise InconclusivePacket(
            stem, list(cache_interval["failure_reasons"]), category="pressure"
        )
    return {
        "pre_cache_host_samples": pre_cache_samples,
        "vm_before_cache": vm_before_cache,
        "cache_ms": cache_ms,
        "cache_bytes": cache_bytes,
        "cache_sha256": cache_sha256,
        "prelaunch_host_samples": prelaunch_samples,
        "host_before_spawn": prelaunch_samples[-1],
        "vm_before_spawn": vm_before_spawn,
        "cache_interval": cache_interval,
    }


def record_launch(
    stem: str,
    command: list[str],
    arm: str,
    block_index: int,
    position: int,
    order: str,
) -> None:
    append_fsync(
        ARTIFACT / "launch-seal.jsonl",
        {
            "event": "launch",
            "unix_ms": time.time_ns() // 1_000_000,
            "artifact_stem": stem,
            "command": command,
            "arm": arm,
            "block_index": block_index,
            "position": position,
            "block_order": order,
        },
    )


def record_completion(stem: str, returncode: int | None, error: str | None) -> None:
    append_fsync(
        ARTIFACT / "launch-seal.jsonl",
        {
            "event": "completion",
            "unix_ms": time.time_ns() // 1_000_000,
            "artifact_stem": stem,
            "returncode": returncode,
            "error": error,
        },
    )


def validate_ms_us(
    timing: dict[str, object], ms_key: str, us_key: str, *, nullable: bool
) -> int | None:
    microseconds = timing.get(us_key)
    milliseconds = timing.get(ms_key)
    if microseconds is None or milliseconds is None:
        if nullable and microseconds is None and milliseconds is None:
            return None
        raise RuntimeError(f"{ms_key}/{us_key} nullability drifted")
    parsed_us = unsigned_integer(microseconds, us_key)
    parsed_ms = finite_number(milliseconds, ms_key, nonnegative=True)
    if abs(parsed_ms - parsed_us / 1000.0) > 0.002:
        raise RuntimeError(f"{ms_key}/{us_key} units disagree")
    return parsed_us


def validate_result(row: dict[str, object], arm: str, build: dict[str, object]) -> None:
    validate_profile_projection(row, build)
    if (
        type(row.get("profile")) is not str
        or type(row.get("arm")) is not str
        or row.get("profile") != EXPECTED_PROFILE
        or row.get("arm") != ARM_NAMES[arm]
    ):
        raise RuntimeError(f"result arm or profile drifted for {arm}")
    exact_unsigned(row.get("resource_count"), EXPECTED_REQUEST_COUNT, "resource count")
    exact_unsigned(row.get("binding_count"), EXPECTED_REQUEST_COUNT, "binding count")
    exact_unsigned(
        row.get("physical_copy_bytes"), EXPECTED_COPY_BYTES, "physical bytes"
    )
    validate_typed_json(
        row.get("resource_modes"), EXPECTED_RESOURCE_MODES, "resource modes"
    )
    validate_schedule(row.get("parallel_copy_schedule"))
    expected_workers = 4 if arm == "B" else 0
    exact_unsigned(row.get("worker_count"), expected_workers, "worker count")
    validate_typed_json(
        row.get("correctness"),
        {
            "passed": True,
            "payload_bytes_checked": EXPECTED_COPY_BYTES,
            "entries_checked": EXPECTED_REQUEST_COUNT,
        },
        "correctness",
    )

    timing = row.get("timing")
    if not isinstance(timing, dict):
        raise RuntimeError("timing row is missing")
    ready_us = validate_ms_us(timing, "ready_wall_ms", "ready_us", nullable=False)
    if ready_us is None or ready_us == 0:
        raise RuntimeError("ready timing is not positive")
    binding_us = validate_ms_us(timing, "binding_wall_ms", "binding_us", nullable=False)
    if binding_us is None:
        raise RuntimeError("binding timing is unavailable")
    validate_ms_us(timing, "teardown_wall_ms", "teardown_us", nullable=False)
    if arm == "A":
        validate_ms_us(timing, "allocation_wall_ms", "allocation_us", nullable=True)
        validate_ms_us(timing, "source_resolution_wall_ms", "source_us", nullable=True)
        if timing.get("source_resolution_us") is not None:
            raise RuntimeError("copied arm reports source-resolution timing")
        validate_ms_us(timing, "copy_wall_ms", "copy_us", nullable=True)
        validate_ms_us(timing, "unattributed_wall_ms", "unattributed_us", nullable=True)
    else:
        allocation_us = validate_ms_us(
            timing, "allocation_wall_ms", "allocation_us", nullable=False
        )
        source_us = validate_ms_us(
            timing, "source_resolution_wall_ms", "source_us", nullable=False
        )
        source_alias_us = unsigned_integer(
            timing.get("source_resolution_us"), "source_resolution_us"
        )
        if source_us is None or source_alias_us != source_us:
            raise RuntimeError("source timing alias drifted")
        copy_us = validate_ms_us(timing, "copy_wall_ms", "copy_us", nullable=False)
        unattributed_us = validate_ms_us(
            timing, "unattributed_wall_ms", "unattributed_us", nullable=False
        )
        if allocation_us is None or copy_us is None or unattributed_us is None:
            raise RuntimeError("parallel timing is unavailable")
        if unattributed_us > 4:
            raise RuntimeError("parallel unattributed wall exceeds truncation bound")
        phase_sum = allocation_us + source_us + copy_us + binding_us
        if abs(ready_us - phase_sum) > 4:
            raise RuntimeError("parallel phase timing does not reconcile")

    throughput = row.get("throughput")
    if not isinstance(throughput, dict):
        raise RuntimeError("throughput row is missing")
    finite_number(
        throughput.get("ready_gbps_decimal"), "ready throughput", positive=True
    )
    if arm == "A":
        if throughput.get("copy_gbps_decimal") is not None:
            raise RuntimeError("copied arm reports copy throughput")
    else:
        finite_number(
            throughput.get("copy_gbps_decimal"), "copy throughput", positive=True
        )

    rusage = row.get("rusage")
    if not isinstance(rusage, dict):
        raise RuntimeError("rusage row is missing")
    for key in ("timer_minor_faults", "timer_major_faults"):
        unsigned_integer(rusage.get(key), f"rusage {key}")
    user_cpu_us = unsigned_integer(rusage.get("user_cpu_us"), "rusage user_cpu_us")
    system_cpu_us = unsigned_integer(
        rusage.get("system_cpu_us"), "rusage system_cpu_us"
    )
    total_cpu_us = unsigned_integer(rusage.get("total_cpu_us"), "rusage total_cpu_us")
    if total_cpu_us != user_cpu_us + system_cpu_us:
        raise RuntimeError("total CPU time does not equal user plus system")
    cpu_per_wall = finite_number(
        rusage.get("cpu_per_wall"), "cpu_per_wall", nonnegative=True
    )
    expected_cpu_per_wall = total_cpu_us / ready_us
    if abs(cpu_per_wall - expected_cpu_per_wall) > max(
        1e-12, expected_cpu_per_wall * 1e-12
    ):
        raise RuntimeError("CPU-per-wall ratio is inconsistent")
    proc = row.get("proc_rusage_v4")
    if not isinstance(proc, dict):
        raise RuntimeError("proc rusage row is missing")
    for key in (
        "instructions_delta_raw",
        "cycles_delta_raw",
        "billed_energy_delta_raw",
        "serviced_energy_delta_raw",
    ):
        unsigned_integer(proc.get(key), f"proc rusage {key}")


def child_post_exit(conditioning: dict[str, object]) -> dict[str, object]:
    host_after_exit = protocol.capture_host_state()
    vm_after_exit = protocol.capture_vm_state()
    return {
        "host_after_exit": host_after_exit,
        "vm_after_exit": vm_after_exit,
        "child_interval": pressure_interval(
            "child", conditioning["vm_before_spawn"], vm_after_exit
        ),
    }


def run_one(
    arm: str,
    block_index: int,
    position: int,
    order: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
    attempts_path: Path,
) -> dict[str, object]:
    stem = f"b{block_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    if stdout_path.exists() or stderr_path.exists():
        raise RuntimeError(f"refusing to reuse child artifact {stem}")
    if child_environment_record(base_env) != manifest["child_environment"]:
        raise RuntimeError("child environment drifted")
    conditioning = condition_for_child(stem, manifest)
    command = [
        "/usr/bin/time",
        "-l",
        str(BINARY),
        "gguf-arena-floor",
        "--model",
        str(MODEL),
        "--profile",
        EXPECTED_PROFILE,
        "--arm",
        ARM_NAMES[arm],
        "--output",
        "json",
    ]
    verify_non_model_identity(manifest)
    record_launch(stem, command, arm, block_index, position, order)
    process = None
    returncode = None
    wait_interrupts: list[str] = []
    completion_error = None
    deferred_sigint: list[int] = []

    def defer_sigint(signum: int, _frame: object) -> None:
        deferred_sigint.append(signum)

    prior_sigint = signal.signal(signal.SIGINT, defer_sigint)
    started = time.perf_counter()
    try:
        try:
            with (
                stdout_path.open("xb") as stdout_file,
                stderr_path.open("xb") as stderr_file,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=base_env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                returncode, wait_interrupts = protocol.wait_for_child(process)
                stdout_file.flush()
                os.fsync(stdout_file.fileno())
                stderr_file.flush()
                os.fsync(stderr_file.fileno())
        except OSError as error:
            completion_error = f"io:{type(error).__name__}:{error}"
    finally:
        if process is not None and returncode is None:
            returncode, deferred_wait = protocol.wait_for_child(process)
            wait_interrupts.extend(deferred_wait)
        signal.signal(signal.SIGINT, prior_sigint)
        record_completion(stem, returncode, completion_error)
    process_wall_ms = (time.perf_counter() - started) * 1e3
    post_exit = child_post_exit(conditioning)
    raw_sha256 = {
        "stdout": host_common.sha256_file(stdout_path),
        "stderr": host_common.sha256_file(stderr_path),
    }
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    reasons = list(post_exit["child_interval"]["failure_reasons"])
    if not post_exit["host_after_exit"]["valid"]:
        reasons.append("post_exit_host_invalid")
    try:
        process_resources = protocol.process_resources(stderr)
    except Exception as error:
        process_resources = None
        reasons.append(f"process_resource_parse={type(error).__name__}:{error}")
    if (
        process_resources is not None
        and process_resources["block_input_operations"] != 0
    ):
        reasons.append("child_block_input")
    if completion_error is not None:
        reasons.append(completion_error)
    if deferred_sigint:
        reasons.append(f"operator_sigint_deferred={len(deferred_sigint)}")
    if returncode != 0:
        reasons.append(f"child_returncode={returncode}")

    result = None
    parse_error = None
    if returncode == 0:
        try:
            parsed = parse_json(stdout_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise RuntimeError("child result is not an object")
            validate_result(parsed, arm, manifest["build_identity"])
            result = parsed
            if parsed["rusage"]["timer_major_faults"] != 0:
                reasons.append("timer_local_major_faults")
        except Exception as error:
            parse_error = f"{type(error).__name__}:{error}"

    attempt = {
        "schema": 1,
        "artifact_stem": stem,
        "block_index": block_index,
        "block_order": order,
        "position": position,
        "arm": arm,
        "command": command,
        "child_environment": manifest["child_environment"],
        "conditioning": conditioning,
        "post_exit": post_exit,
        "process_wall_ms": process_wall_ms,
        "returncode": returncode,
        "wait_interrupts": wait_interrupts,
        "process_resources": process_resources,
        "process_resource_units": {
            "maximum_resident_set_size": "bytes",
            "peak_memory_footprint": "bytes",
        },
        "raw_sha256": raw_sha256,
        "validity_reasons": reasons,
        "parse_error": parse_error,
        "result": result,
    }
    append_fsync(attempts_path, attempt)
    if parse_error is not None:
        raise RuntimeError(f"{stem} result contract failed: {parse_error}")
    if reasons:
        category = "operator_interrupt" if deferred_sigint else "execution"
        if post_exit["child_interval"]["failure_reasons"]:
            category = "pressure"
        raise InconclusivePacket(stem, reasons, category=category)
    if result is None:
        raise RuntimeError(f"{stem} completed without a result")
    return attempt


def analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 12:
        raise RuntimeError("complete floor packet requires 12 rows")
    blocks = []
    cursor = 0
    for block_index, order in enumerate(BLOCK_ORDERS, 1):
        selected = rows[cursor : cursor + 2]
        cursor += 2
        if [row["arm"] for row in selected] != list(order):
            raise RuntimeError(f"block {block_index} order drifted")
        if any(row["block_index"] != block_index for row in selected):
            raise RuntimeError(f"block {block_index} identity drifted")
        blocks.append({row["arm"]: row for row in selected})

    metrics = []
    for order, block in zip(BLOCK_ORDERS, blocks, strict=True):
        a_us = unsigned_integer(block["A"]["result"]["timing"]["ready_us"], "A ready")
        b_us = unsigned_integer(block["B"]["result"]["timing"]["ready_us"], "B ready")
        a_rss = finite_number(
            block["A"]["process_resources"]["maximum_resident_set_size"],
            "A RSS",
            positive=True,
        )
        b_rss = finite_number(
            block["B"]["process_resources"]["maximum_resident_set_size"],
            "B RSS",
            positive=True,
        )
        a_foot = finite_number(
            block["A"]["process_resources"]["peak_memory_footprint"],
            "A footprint",
            positive=True,
        )
        b_foot = finite_number(
            block["B"]["process_resources"]["peak_memory_footprint"],
            "B footprint",
            positive=True,
        )
        metrics.append(
            {
                "order": order,
                "a_ready_us": a_us,
                "b_ready_us": b_us,
                "saving_ms": (a_us - b_us) / 1000.0,
                "b_over_a": b_us / a_us,
                "b_wins": b_us < a_us,
                "rss_b_over_a": b_rss / a_rss,
                "footprint_b_over_a": b_foot / a_foot,
            }
        )
    savings = [row["saving_ms"] for row in metrics]
    ratios = [row["b_over_a"] for row in metrics]
    ab_savings = [row["saving_ms"] for row in metrics if row["order"] == "AB"]
    ba_savings = [row["saving_ms"] for row in metrics if row["order"] == "BA"]
    median_saving = statistics.median(savings)
    median_ratio = statistics.median(ratios)
    wins = sum(row["b_wins"] for row in metrics)
    ab_wins = sum(row["b_wins"] for row in metrics if row["order"] == "AB")
    ba_wins = sum(row["b_wins"] for row in metrics if row["order"] == "BA")
    max_rss = max(row["rss_b_over_a"] for row in metrics)
    max_footprint = max(row["footprint_b_over_a"] for row in metrics)
    gates = {
        "median_saving_at_least_750_ms": median_saving >= 750.0,
        "median_b_over_a_at_most_0_70": median_ratio <= 0.70,
        "wins_at_least_5_of_6": wins >= 5,
        "ab_median_saving_at_least_600_ms": statistics.median(ab_savings) >= 600.0,
        "ba_median_saving_at_least_600_ms": statistics.median(ba_savings) >= 600.0,
        "ab_wins_at_least_2_of_3": ab_wins >= 2,
        "ba_wins_at_least_2_of_3": ba_wins >= 2,
        "max_rss_b_over_a_at_most_1_05": max_rss <= 1.05,
        "max_footprint_b_over_a_at_most_1_05": max_footprint <= 1.05,
    }
    return {
        "metrics": metrics,
        "median_saving_ms": median_saving,
        "median_b_over_a": median_ratio,
        "wins": wins,
        "ab_savings_ms": ab_savings,
        "ba_savings_ms": ba_savings,
        "ab_median_saving_ms": statistics.median(ab_savings),
        "ba_median_saving_ms": statistics.median(ba_savings),
        "ab_wins": ab_wins,
        "ba_wins": ba_wins,
        "max_rss_b_over_a": max_rss,
        "max_footprint_b_over_a": max_footprint,
        "arm_ready_us": {
            arm: [
                unsigned_integer(
                    block[arm]["result"]["timing"]["ready_us"], f"{arm} ready"
                )
                for block in blocks
            ]
            for arm in ("A", "B")
        },
        "arm_rusage": {
            arm: [block[arm]["result"]["rusage"] for block in blocks]
            for arm in ("A", "B")
        },
        "arm_proc_rusage_v4": {
            arm: [block[arm]["result"]["proc_rusage_v4"] for block in blocks]
            for arm in ("A", "B")
        },
        "gates": gates,
        "passes": all(gates.values()),
    }


def publication_started() -> bool:
    names = (
        "decision.json",
        "artifact-inventory.sha256",
        "packet-complete.json",
        ".decision.json.tmp",
        ".artifact-inventory.sha256.tmp",
        ".packet-complete.json.tmp",
    )
    return any((ARTIFACT / name).exists() for name in names)


def expected_child_stems() -> list[str]:
    return [
        f"b{block_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
        for block_index, order in enumerate(BLOCK_ORDERS, 1)
        for position, arm in enumerate(order, 1)
    ]


def attempt_rows_sha256(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json_text(row) + "\n").encode())
    return digest.hexdigest()


def verify_attempt_artifacts(decision: dict[str, object]) -> dict[Path, str]:
    attempts_path = ARTIFACT / "attempts.jsonl"
    expected: dict[Path, str] = {}
    attempt_rows = []
    if attempts_path.exists():
        attempts_sha256 = host_common.sha256_file(attempts_path)
        expected[attempts_path] = attempts_sha256
        attempt_lines = attempts_path.read_text(encoding="utf-8").splitlines()
    else:
        attempts_sha256 = None
        attempt_lines = []
    for line_number, line in enumerate(attempt_lines, 1):
        value = parse_json(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"attempt row {line_number} is not an object")
        attempt_rows.append(value)
        stem = value.get("artifact_stem")
        raw = value.get("raw_sha256")
        if type(stem) is not str or not isinstance(raw, dict):
            raise RuntimeError(f"attempt row {line_number} lacks raw hashes")
        for suffix, key in (("out", "stdout"), ("err", "stderr")):
            digest = raw.get(key)
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError(f"attempt row {line_number} {key} hash is invalid")
            path = ARTIFACT / f"{stem}.{suffix}"
            if path in expected:
                raise RuntimeError(f"duplicate raw artifact identity {path.name}")
            if host_common.sha256_file(path) != digest:
                raise RuntimeError(f"raw artifact {path.name} changed after parsing")
            expected[path] = digest

    launch_path = ARTIFACT / "launch-seal.jsonl"
    launch_rows = []
    if launch_path.exists():
        expected[launch_path] = host_common.sha256_file(launch_path)
        for line_number, line in enumerate(
            launch_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            value = parse_json(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"launch row {line_number} is not an object")
            launch_rows.append(value)
    if len(launch_rows) % 2 != 0:
        raise RuntimeError("launch/completion ledger is incomplete")
    launched_stems = []
    for index in range(0, len(launch_rows), 2):
        launch = launch_rows[index]
        completion = launch_rows[index + 1]
        if launch.get("event") != "launch" or completion.get("event") != "completion":
            raise RuntimeError("launch/completion event order drifted")
        stem = launch.get("artifact_stem")
        if type(stem) is not str or completion.get("artifact_stem") != stem:
            raise RuntimeError("launch/completion stem drifted")
        launched_stems.append(stem)

    expected_stems = expected_child_stems()
    attempt_stems = [row.get("artifact_stem") for row in attempt_rows]
    if len(set(launched_stems)) != len(launched_stems):
        raise RuntimeError("duplicate launched child identity")
    if len(set(attempt_stems)) != len(attempt_stems):
        raise RuntimeError("duplicate attempt child identity")
    if launched_stems != expected_stems[: len(launched_stems)]:
        raise RuntimeError("launched children are not the frozen packet prefix")
    if attempt_stems != launched_stems[: len(attempt_stems)]:
        raise RuntimeError("attempt rows do not match the launch prefix")
    if len(launched_stems) - len(attempt_stems) > 1:
        raise RuntimeError("more than one launched child lacks an attempt row")
    if decision.get("status") in ("go", "kill") and (
        launched_stems != expected_stems or attempt_stems != expected_stems
    ):
        raise RuntimeError("performance decision lacks the complete frozen packet")
    if decision.get("status") in ("go", "kill") and (
        type(decision.get("attempts_sha256")) is not str
        or decision.get("attempts_sha256") != attempts_sha256
    ):
        raise RuntimeError("attempt ledger differs from the scored in-memory rows")

    expected_metadata = [
        {
            "block_index": block_index,
            "block_order": order,
            "position": position,
            "arm": arm,
        }
        for block_index, order in enumerate(BLOCK_ORDERS, 1)
        for position, arm in enumerate(order, 1)
    ]
    for index, row in enumerate(attempt_rows):
        metadata = expected_metadata[index]
        launch = launch_rows[2 * index]
        completion = launch_rows[2 * index + 1]
        exact_unsigned(row.get("schema"), 1, f"attempt {index} schema")
        for key, value in metadata.items():
            validate_typed_json(row.get(key), value, f"attempt {index} {key}")
            validate_typed_json(launch.get(key), value, f"launch {index} {key}")
        validate_typed_json(
            row.get("command"), launch.get("command"), f"attempt {index} command"
        )
        validate_typed_json(
            row.get("returncode"),
            completion.get("returncode"),
            f"attempt {index} returncode",
        )

    actual_raw_paths = {
        path
        for pattern in ("*.out", "*.err")
        for path in ARTIFACT.glob(pattern)
        if path.is_file()
    }
    expected_raw_paths = {
        ARTIFACT / f"{stem}.{suffix}"
        for stem in attempt_stems
        for suffix in ("out", "err")
    }
    if actual_raw_paths != expected_raw_paths:
        raise RuntimeError("raw artifact set does not match sealed attempt rows")
    return expected


def publish_decision(
    decision: dict[str, object], sealed_raw_hashes: dict[Path, str]
) -> None:
    decision_path = ARTIFACT / "decision.json"
    inventory_path = ARTIFACT / "artifact-inventory.sha256"
    complete_path = ARTIFACT / "packet-complete.json"
    decision_tmp = ARTIFACT / ".decision.json.tmp"
    inventory_tmp = ARTIFACT / ".artifact-inventory.sha256.tmp"
    complete_tmp = ARTIFACT / ".packet-complete.json.tmp"
    publication_paths = {
        decision_path,
        inventory_path,
        complete_path,
        decision_tmp,
        inventory_tmp,
        complete_tmp,
    }
    if any(path.exists() for path in publication_paths):
        raise RuntimeError("decision publication already started")
    decision_bytes = (json_text(decision, pretty=True) + "\n").encode()
    with decision_tmp.open("xb") as output:
        output.write(decision_bytes)
        output.flush()
        os.fsync(output.fileno())
    entries = []
    encountered_sealed_paths = set()
    for path in sorted(ARTIFACT.iterdir()):
        if path.is_file() and path not in publication_paths:
            digest = host_common.sha256_file(path)
            if path in sealed_raw_hashes and digest != sealed_raw_hashes[path]:
                raise RuntimeError(
                    f"raw artifact {path.name} changed during publication"
                )
            if path in sealed_raw_hashes:
                encountered_sealed_paths.add(path)
            entries.append(f"{digest}  {path.relative_to(ROOT)}")
    if encountered_sealed_paths != set(sealed_raw_hashes):
        raise RuntimeError("sealed evidence path disappeared during publication")
    decision_sha256 = hashlib.sha256(decision_bytes).hexdigest()
    entries.append(f"{decision_sha256}  {decision_path.relative_to(ROOT)}")
    inventory_bytes = ("\n".join(entries) + "\n").encode()
    with inventory_tmp.open("xb") as output:
        output.write(inventory_bytes)
        output.flush()
        os.fsync(output.fileno())
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    complete = {
        "schema": 1,
        "decision_sha256": decision_sha256,
        "inventory_sha256": inventory_sha256,
    }
    with complete_tmp.open("xb") as output:
        output.write((json_text(complete, pretty=True) + "\n").encode())
        output.flush()
        os.fsync(output.fileno())
    os.replace(decision_tmp, decision_path)
    fsync_directory(ARTIFACT)
    os.replace(inventory_tmp, inventory_path)
    fsync_directory(ARTIFACT)
    os.replace(complete_tmp, complete_path)
    fsync_directory(ARTIFACT)
    try:
        print(json_text(decision, pretty=True))
    except BrokenPipeError:
        pass


def publish_identity_checked(
    decision: dict[str, object], manifest: dict[str, object]
) -> None:
    prior = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            verify_packet_identity(manifest)
            identity_error = None
            identity_io_error = None
        except OSError as error:
            identity_error = None
            identity_io_error = f"{type(error).__name__}:{error}"
        except Exception as error:
            identity_error = f"{type(error).__name__}:{error}"
            identity_io_error = None
        if identity_io_error is not None:
            decision = {
                **decision,
                "reported_status_before_identity_check": decision.get("status"),
                "status": "inconclusive",
                "category": "io",
                "authority": "none",
                "identity_io_error": identity_io_error,
            }
        if identity_error is not None:
            decision = {
                **decision,
                "reported_status_before_identity_check": decision.get("status"),
                "status": "implementation_or_contract_defect",
                "authority": "none",
                "identity_error": identity_error,
            }
        sealed_raw_hashes = verify_attempt_artifacts(decision)
        publish_decision(decision, sealed_raw_hashes)
    finally:
        signal.signal(signal.SIGINT, prior)


def write_final_model_hash(manifest: dict[str, object]) -> None:
    value = {
        "model": str(MODEL),
        "size_bytes": MODEL.stat().st_size,
        "sha256": host_common.sha256_file(MODEL),
    }
    if value["size_bytes"] != manifest["model_size_bytes"]:
        raise RuntimeError("final model size drifted")
    if value["sha256"] != manifest["model_sha256"]:
        raise RuntimeError("final model hash drifted")
    with (ARTIFACT / "final-model-sha256.json").open("x", encoding="utf-8") as output:
        output.write(json_text(value, pretty=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def launch_occurred() -> bool:
    path = ARTIFACT / "launch-seal.jsonl"
    return path.exists() and path.stat().st_size > 0


def main(*, preflight_only: bool) -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse packet directory {ARTIFACT}")
    base_env, removed_environment = host_common.normalized_environment()
    manifest = build_manifest(removed_environment, base_env)
    if preflight_only:
        print(
            json_text(
                {
                    "status": "preflight-passed",
                    "source_commit": manifest["source_commit"],
                    "manifest_sha256": hashlib.sha256(
                        (json_text(manifest, pretty=True) + "\n").encode()
                    ).hexdigest(),
                },
                pretty=True,
            )
        )
        return

    reserve_artifact(manifest)
    attempts_path = ARTIFACT / "attempts.jsonl"
    rows = []
    try:
        for block_index, order in enumerate(BLOCK_ORDERS, 1):
            for position, arm in enumerate(order, 1):
                rows.append(
                    run_one(
                        arm,
                        block_index,
                        position,
                        order,
                        base_env,
                        manifest,
                        attempts_path,
                    )
                )
        write_final_model_hash(manifest)
        verify_packet_identity(manifest)
        result = analyze(rows)
        status = "go" if result["passes"] else "kill"
        authority = (
            "implement-force-only-dense27b-loader-pilot" if status == "go" else "none"
        )
        publish_identity_checked(
            {
                "schema": 1,
                "status": status,
                "authority": authority,
                "source_commit": manifest["source_commit"],
                "completed_blocks": len(BLOCK_ORDERS),
                "attempts_sha256": attempt_rows_sha256(rows),
                "result": result,
            },
            manifest,
        )
    except InconclusivePacket as error:
        publish_identity_checked(
            {
                "schema": 1,
                "status": "inconclusive",
                "category": error.category,
                "authority": "none",
                "source_commit": manifest["source_commit"],
                "failed_child": error.child,
                "reasons": error.reasons,
                "completed_children": len(rows),
            },
            manifest,
        )
    except KeyboardInterrupt:
        publish_identity_checked(
            {
                "schema": 1,
                "status": "inconclusive",
                "category": "operator_interrupt",
                "authority": "none",
                "source_commit": manifest["source_commit"],
                "reasons": ["operator_interrupt_after_child_cleanup"],
                "completed_children": len(rows),
            },
            manifest,
        )
    except OSError as error:
        if publication_started():
            raise
        publish_identity_checked(
            {
                "schema": 1,
                "status": "inconclusive",
                "category": "io",
                "authority": "none",
                "source_commit": manifest["source_commit"],
                "error_type": type(error).__name__,
                "error": str(error),
                "launched": launch_occurred(),
                "completed_children": len(rows),
            },
            manifest,
        )
    except Exception as error:
        if publication_started():
            raise
        publish_identity_checked(
            {
                "schema": 1,
                "status": "implementation_or_contract_defect",
                "authority": "none",
                "source_commit": manifest["source_commit"],
                "error_type": type(error).__name__,
                "error": str(error),
                "launched": launch_occurred(),
                "completed_children": len(rows),
            },
            manifest,
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    main(preflight_only=arguments.preflight_only)
