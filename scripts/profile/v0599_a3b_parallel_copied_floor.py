#!/usr/bin/env python3

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import v0593_demand_paged_no_copy as common
import v0595_a3b_generic_retained_cold as prior


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0599-a3b-parallel-copied-floor-p1"
PREREG = ROOT / "docs/bench/v0599-a3b-parallel-copied-floor.md"
MODEL = Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
BINARY = ROOT / "target/release/qwen-bench"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
PRIOR_RUNNER = ROOT / "scripts/profile/v0595_a3b_generic_retained_cold.py"
EXPECTED_MODEL_SHA256 = (
    "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
)
EXPECTED_DESCRIPTOR_DIGEST = "0x5ae645df5cf7d568"
EXPECTED_INVENTORY_DIGEST = (
    "f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5"
)
EXPECTED_PLANNER_DIGEST = (
    "fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af"
)
EXPECTED_BUILD_GEOMETRY = {
    "architecture": "qwen35moe",
    "native_quant_embedding": True,
    "page_size": 16_384,
    "required_alignment": 32,
    "max_buffer_length": 77_309_411_328,
    "request_count": 733,
    "view_count": 732,
    "window_count": 1,
    "fallback_count": 1,
    "alias_count": 0,
    "logical_copy_bytes": 22_123_538_944,
    "unique_view_bytes": 22_123_530_752,
    "logical_view_bytes": 22_123_530_752,
    "fallback_bytes": 8_192,
    "window_bytes": 22_123_544_576,
    "planner_gap_bytes": 13_824,
    "arena_copy_bytes": 22_123_552_768,
    "fallback_reasons": ["FinalPartialPage"],
}
EXPECTED_SCHEDULE = {
    "workers": 4,
    "cuts": [155, 359, 539],
    "task_counts": [155, 204, 180, 194],
    "worker_bytes": [
        5_532_746_240,
        5_462_315_776,
        5_595_522_304,
        5_532_954_624,
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
ARM_NAMES = {"A": "copied", "B": "parallel-copied"}
BLOCK_ORDERS = ("AB", "BA", "BA", "AB", "AB", "BA")
COOLDOWN_S = 30.0
MAX_BLOCK_ATTEMPTS = 3
FROZEN_FIRST_BYTE_MS = 2463.65
TRANSFER_HAIRCUT = 0.96


class InconclusivePacket(RuntimeError):
    def __init__(self, block_index: int, order: str) -> None:
        super().__init__(f"failed to obtain valid block {block_index} order={order}")
        self.block_index = block_index
        self.order = order


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def parse_json(text: str) -> object:
    return json.loads(text, parse_constant=reject_json_constant)


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


def child_environment_record(env: dict[str, str]) -> dict[str, object]:
    digest = hashlib.sha256()
    for key, value in sorted(env.items()):
        key_bytes = key.encode("utf-8")
        value_bytes = value.encode("utf-8")
        digest.update(len(key_bytes).to_bytes(8, "little"))
        digest.update(key_bytes)
        digest.update(len(value_bytes).to_bytes(8, "little"))
        digest.update(value_bytes)
    performance_controls = {
        key: value
        for key, value in sorted(env.items())
        if key.startswith(("QWEN_", "METAL_", "MTL_")) or key == "RUST_LOG"
    }
    if performance_controls:
        raise RuntimeError(
            f"normalized child environment retains controls: {performance_controls}"
        )
    return {
        "schema": 1,
        "complete_sha256": digest.hexdigest(),
        "keys": sorted(env),
        "performance_controls": performance_controls,
    }


def json_text(value: object, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        indent=2 if pretty else None,
        sort_keys=True,
        allow_nan=False,
    )


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
        COMMON_RUNNER,
        PRIOR_RUNNER,
        MODEL,
        BINARY,
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    for path in required_manifest_paths()[:4]:
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


def warm_file() -> tuple[float, int]:
    started = time.perf_counter()
    total = 0
    buffer = bytearray(8 * 1024 * 1024)
    with MODEL.open("rb", buffering=0) as handle:
        while True:
            count = handle.readinto(buffer)
            if count == 0:
                break
            total += count
    return (time.perf_counter() - started) * 1e3, total


def validate_schedule(value: object) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("parallel-copy schedule is missing")
    for key, expected in EXPECTED_SCHEDULE.items():
        if value.get(key) != expected:
            raise RuntimeError(f"schedule {key} drifted")
    max_to_min = finite_number(
        value.get("max_to_min"), "schedule max/min", positive=True
    )
    max_to_ideal = finite_number(
        value.get("max_to_ideal"), "schedule max/ideal", positive=True
    )
    if abs(max_to_min - 1.024386) > 0.000001:
        raise RuntimeError("schedule max/min drifted")
    if abs(max_to_ideal - 1.011687) > 0.000001:
        raise RuntimeError("schedule max/ideal drifted")
    partitions = value.get("partitions")
    if not isinstance(partitions, list) or len(partitions) != 4:
        raise RuntimeError("schedule partitions drifted")
    boundaries = [
        0,
        *EXPECTED_SCHEDULE["cuts"],
        EXPECTED_BUILD_GEOMETRY["request_count"],
    ]
    cursor = 0
    byte_total = 0
    for index, partition in enumerate(partitions):
        if not isinstance(partition, dict):
            raise RuntimeError("schedule partition is malformed")
        if partition.get("start") != boundaries[index] or cursor != boundaries[index]:
            raise RuntimeError("schedule partition start drifted")
        cursor = int(partition.get("end", -1))
        if cursor != boundaries[index + 1]:
            raise RuntimeError("schedule partition end drifted")
        if partition.get("task_count") != EXPECTED_SCHEDULE["task_counts"][index]:
            raise RuntimeError("schedule partition task count drifted")
        if partition.get("task_count") != cursor - boundaries[index]:
            raise RuntimeError("schedule partition extent drifted")
        if partition.get("bytes") != EXPECTED_SCHEDULE["worker_bytes"][index]:
            raise RuntimeError("schedule partition bytes drifted")
        byte_total += int(partition["bytes"])
        for field in (
            "first_shard",
            "first_source_offset",
            "last_shard",
            "last_source_offset",
        ):
            if not isinstance(partition.get(field), int):
                raise RuntimeError(f"schedule partition {field} is missing")
    if cursor != EXPECTED_BUILD_GEOMETRY["request_count"]:
        raise RuntimeError("schedule partition union drifted")
    if byte_total != EXPECTED_BUILD_GEOMETRY["logical_copy_bytes"]:
        raise RuntimeError("schedule byte union drifted")


def validate_geometry(row: dict[str, object], build: dict[str, object]) -> None:
    if row.get("schema_version") != 1:
        raise RuntimeError("geometry schema drifted")
    if row.get("descriptor_layout_digest") != EXPECTED_DESCRIPTOR_DIGEST:
        raise RuntimeError("descriptor-layout digest drifted")
    if row.get("inventory_digest") != EXPECTED_INVENTORY_DIGEST:
        raise RuntimeError("inventory digest drifted")
    if row.get("planner_digest") != EXPECTED_PLANNER_DIGEST:
        raise RuntimeError("planner digest drifted")
    for key, expected in EXPECTED_BUILD_GEOMETRY.items():
        if row.get(key) != expected:
            raise RuntimeError(f"geometry {key} drifted")
    validate_schedule(row.get("parallel_copy_schedule"))
    if row.get("build_identity") != build:
        raise RuntimeError("geometry build identity drifted")


def build_manifest(
    removed_environment: list[str],
    base_env: dict[str, str],
) -> dict[str, object]:
    commit, build = source_and_build_identity()
    hashes = {str(path): common.sha256_file(path) for path in required_manifest_paths()}
    if hashes[str(MODEL)] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA-256 drifted")
    describe_command = [
        str(BINARY),
        "gguf-arena-floor",
        "--model",
        str(MODEL),
        "--describe",
        "--output",
        "json",
    ]
    describe = parse_json(command_text(describe_command, env=base_env))
    if not isinstance(describe, dict):
        raise RuntimeError("geometry control is not an object")
    if describe.get("mode") != "describe":
        raise RuntimeError("geometry control mode drifted")
    validate_geometry(describe, build)
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "build_identity": build,
        "removed_environment": removed_environment,
        "child_environment": child_environment_record(base_env),
        "sha256": hashes,
        "model_size_bytes": MODEL.stat().st_size,
        "block_orders": list(BLOCK_ORDERS),
        "cooldown_s": COOLDOWN_S,
        "max_block_attempts": MAX_BLOCK_ATTEMPTS,
        "frozen_first_byte_ms": FROZEN_FIRST_BYTE_MS,
        "transfer_haircut": TRANSFER_HAIRCUT,
        "geometry_control": describe,
        "describe_command": describe_command,
    }


def verify_packet_identity(manifest: dict[str, object]) -> None:
    commit, build = source_and_build_identity()
    if commit != manifest["source_commit"] or build != manifest["build_identity"]:
        raise RuntimeError("packet completion source/build identity drifted")
    hashes = {str(path): common.sha256_file(path) for path in required_manifest_paths()}
    if hashes != manifest["sha256"]:
        raise RuntimeError("packet completion file hashes drifted")


def validate_row(
    row: dict[str, object],
    arm: str,
    build: dict[str, object],
) -> None:
    validate_geometry(row, build)
    if row.get("arm") != ARM_NAMES[arm]:
        raise RuntimeError(f"arm label drifted for {arm}")
    if row.get("resource_count") != 733 or row.get("binding_count") != 733:
        raise RuntimeError("copied topology count drifted")
    if row.get("physical_copy_bytes") != 22_123_538_944:
        raise RuntimeError("physical copied bytes drifted")
    if row.get("worker_count") != (4 if arm == "B" else 0):
        raise RuntimeError("worker count drifted")
    if row.get("resource_modes") != EXPECTED_RESOURCE_MODES:
        raise RuntimeError("resource modes drifted")
    correctness = row.get("correctness")
    expected_correctness = {
        "passed": True,
        "full_windows_checked": 0,
        "fallback_bytes_checked": 8_192,
        "entries_checked": 733,
        "aliases_checked": 0,
    }
    if correctness != expected_correctness:
        raise RuntimeError(f"correctness drifted for {arm}: {correctness}")
    timing = row.get("timing")
    if not isinstance(timing, dict):
        raise RuntimeError("ready timing is missing")
    finite_number(timing.get("ready_wall_ms"), "ready wall", positive=True)
    finite_number(timing.get("binding_wall_ms"), "binding wall", nonnegative=True)
    finite_number(timing.get("teardown_wall_ms"), "teardown wall", nonnegative=True)
    if arm == "A":
        for key in (
            "allocation_wall_ms",
            "source_resolution_wall_ms",
            "copy_wall_ms",
            "unattributed_wall_ms",
        ):
            if timing.get(key) is not None:
                raise RuntimeError(f"copied arm unexpectedly reports {key}")
    else:
        components = [
            finite_number(timing.get(key), f"parallel {key}", nonnegative=True)
            for key in (
                "allocation_wall_ms",
                "source_resolution_wall_ms",
                "copy_wall_ms",
                "binding_wall_ms",
                "unattributed_wall_ms",
            )
        ]
        if min(components) < 0:
            raise RuntimeError("negative parallel timing component")
        ready_wall = finite_number(timing["ready_wall_ms"], "ready wall", positive=True)
        if abs(sum(components) - ready_wall) > 0.01:
            raise RuntimeError("parallel timing components do not reconcile")
    throughput = row.get("throughput")
    if not isinstance(throughput, dict):
        raise RuntimeError("throughput row is missing")
    finite_number(
        throughput.get("ready_gbps_decimal"), "ready throughput", positive=True
    )
    copy_throughput = throughput.get("copy_gbps_decimal")
    if arm == "A":
        if copy_throughput is not None:
            raise RuntimeError("copied arm unexpectedly reports copy throughput")
    else:
        finite_number(copy_throughput, "parallel copy throughput", positive=True)
    rusage = row.get("rusage")
    if not isinstance(rusage, dict):
        raise RuntimeError("rusage row is missing")
    finite_number(
        rusage.get("timer_major_faults"), "timer major faults", nonnegative=True
    )
    finite_number(
        rusage.get("timer_minor_faults"), "timer minor faults", nonnegative=True
    )
    allocated = row.get("metal_allocated_bytes")
    if not isinstance(allocated, dict):
        raise RuntimeError("Metal allocation row is missing")
    for key in ("before", "ready", "after_drop"):
        finite_number(allocated.get(key), f"Metal allocation {key}", nonnegative=True)


def run_one(
    arm: str,
    block_index: int,
    attempt: int,
    run_index: int,
    order: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
    environment_record = child_environment_record(base_env)
    if environment_record != manifest["child_environment"]:
        raise RuntimeError("child environment drifted")
    stem = f"b{block_index:02d}-a{attempt:02d}-r{run_index}-{arm.lower()}"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    for path in (stdout_path, stderr_path):
        if path.exists():
            raise RuntimeError(f"refusing to reuse attempt artifact {path}")

    host_before_cache = common.wait_for_valid_host(f"{stem} cache read")
    vm_before_cache = prior.capture_vm_state()
    cache_ms, cache_bytes = warm_file()
    if cache_bytes != MODEL.stat().st_size:
        raise RuntimeError("cache precondition did not read the complete model")
    time.sleep(COOLDOWN_S)
    host_before_spawn = common.wait_for_valid_host(f"{stem} process spawn")
    vm_before_spawn = prior.capture_vm_state()
    cache_pageout_delta = vm_before_spawn["pageouts"] - vm_before_cache["pageouts"]
    cache_swap_delta = (
        vm_before_spawn["swap_used_bytes"] - vm_before_cache["swap_used_bytes"]
    )
    if cache_pageout_delta != 0 or cache_swap_delta > 0:
        raise RuntimeError(
            f"{stem} cache conditioning changed VM pressure: "
            f"pageouts={cache_pageout_delta} swap={cache_swap_delta}"
        )

    command = [
        "/usr/bin/time",
        "-l",
        str(BINARY),
        "gguf-arena-floor",
        "--model",
        str(MODEL),
        "--arm",
        ARM_NAMES[arm],
        "--output",
        "json",
    ]
    started = time.perf_counter()
    with stdout_path.open("xb") as stdout_file, stderr_path.open("xb") as stderr_file:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=base_env,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    process_wall_ms = (time.perf_counter() - started) * 1e3
    if process.returncode != 0:
        raise RuntimeError(f"{stem} exited {process.returncode}; see {stderr_path}")
    row = parse_json(stdout_path.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise RuntimeError("benchmark row is not an object")
    validate_row(row, arm, manifest["build_identity"])
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    host_after_exit = common.capture_host_state()
    vm_after_exit = prior.capture_vm_state()
    block_inputs = common.parse_resource(stderr, "block input operations")
    page_faults = common.parse_resource(stderr, "page faults")
    page_reclaims = common.parse_resource(stderr, "page reclaims")
    maximum_rss = common.parse_resource(stderr, "maximum resident set size")
    peak_footprint = common.parse_resource(stderr, "peak memory footprint")
    finite_number(process_wall_ms, "process wall", positive=True)
    finite_number(cache_ms, "cache precondition wall", positive=True)
    finite_number(maximum_rss, "maximum resident set size", positive=True)
    finite_number(peak_footprint, "peak memory footprint", positive=True)
    finite_number(page_reclaims, "page reclaims", nonnegative=True)
    finite_number(page_faults, "page faults", nonnegative=True)
    finite_number(block_inputs, "block input operations", nonnegative=True)
    process_pageout_delta = vm_after_exit["pageouts"] - vm_before_spawn["pageouts"]
    process_swap_delta = (
        vm_after_exit["swap_used_bytes"] - vm_before_spawn["swap_used_bytes"]
    )
    timer_major_faults = row["rusage"]["timer_major_faults"]
    validity_reasons = []
    if timer_major_faults != 0:
        validity_reasons.append(f"timer_major_faults={timer_major_faults}")
    if block_inputs != 0:
        validity_reasons.append(f"block_input_operations={block_inputs}")
    if process_pageout_delta != 0:
        validity_reasons.append(f"process_pageout_delta={process_pageout_delta}")
    if process_swap_delta > 0:
        validity_reasons.append(f"process_swap_growth_bytes={process_swap_delta}")
    if not host_after_exit["valid"]:
        validity_reasons.append("post_exit_host_invalid")
    return {
        "artifact_stem": stem,
        "block_index": block_index,
        "block_order": order,
        "attempt": attempt,
        "run_index": run_index,
        "arm": arm,
        "command": command,
        "child_environment": environment_record,
        "cache_precondition_ms": cache_ms,
        "host_before_cache": host_before_cache,
        "host_before_spawn": host_before_spawn,
        "host_after_exit": host_after_exit,
        "vm_before_cache": vm_before_cache,
        "vm_before_spawn": vm_before_spawn,
        "vm_after_exit": vm_after_exit,
        "cache_pageout_delta": cache_pageout_delta,
        "cache_swap_delta_bytes": cache_swap_delta,
        "process_pageout_delta": process_pageout_delta,
        "process_swap_delta_bytes": process_swap_delta,
        "process_wall_ms": process_wall_ms,
        "maximum_resident_set_size": maximum_rss,
        "peak_memory_footprint": peak_footprint,
        "page_reclaims": page_reclaims,
        "page_faults": page_faults,
        "block_input_operations": block_inputs,
        "valid": not validity_reasons,
        "validity_reasons": validity_reasons,
        "result": row,
    }


def append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json_text(row) + "\n")


def run_valid_block(
    block_index: int,
    order: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
    attempts_path: Path,
    blocks_path: Path,
) -> list[dict[str, object]]:
    for attempt in range(1, MAX_BLOCK_ATTEMPTS + 1):
        rows = []
        for run_index, arm in enumerate(order, 1):
            row = run_one(
                arm,
                block_index,
                attempt,
                run_index,
                order,
                base_env,
                manifest,
            )
            append_row(attempts_path, row)
            rows.append(row)
        accepted = all(row["valid"] for row in rows)
        append_row(
            blocks_path,
            {
                "block_index": block_index,
                "block_order": order,
                "attempt": attempt,
                "accepted": accepted,
                "artifact_stems": [row["artifact_stem"] for row in rows],
                "invalid_rows": [
                    {"arm": row["arm"], "reasons": row["validity_reasons"]}
                    for row in rows
                    if not row["valid"]
                ],
            },
        )
        if accepted:
            return rows
    raise InconclusivePacket(block_index, order)


def analyze(
    rows: list[dict[str, object]], manifest: dict[str, object]
) -> dict[str, object]:
    accepted_blocks = []
    for block_index, order in enumerate(BLOCK_ORDERS, 1):
        selected = [
            row
            for row in rows
            if row["block_index"] == block_index
            and row["block_order"] == order
            and row["valid"]
        ]
        if len(selected) != 2:
            raise RuntimeError(f"accepted block {block_index} is incomplete")
        accepted_blocks.append({row["arm"]: row for row in selected})

    savings = []
    ratios = []
    wins = 0
    stratum_savings = {"AB": [], "BA": []}
    stratum_wins = {"AB": 0, "BA": 0}
    rss_ratios = []
    footprint_ratios = []
    for block, order in zip(accepted_blocks, BLOCK_ORDERS, strict=True):
        baseline_ms = finite_number(
            block["A"]["result"]["timing"]["ready_wall_ms"],
            "scored baseline ready wall",
            positive=True,
        )
        candidate_ms = finite_number(
            block["B"]["result"]["timing"]["ready_wall_ms"],
            "scored candidate ready wall",
            positive=True,
        )
        saving = baseline_ms - candidate_ms
        ratio = candidate_ms / baseline_ms
        finite_number(saving, "scored saving")
        finite_number(ratio, "scored ready ratio", positive=True)
        savings.append(saving)
        ratios.append(ratio)
        wins += candidate_ms < baseline_ms
        stratum_savings[order].append(saving)
        stratum_wins[order] += candidate_ms < baseline_ms
        baseline_rss = finite_number(
            block["A"]["maximum_resident_set_size"],
            "scored baseline RSS",
            positive=True,
        )
        candidate_rss = finite_number(
            block["B"]["maximum_resident_set_size"],
            "scored candidate RSS",
            positive=True,
        )
        baseline_footprint = finite_number(
            block["A"]["peak_memory_footprint"],
            "scored baseline footprint",
            positive=True,
        )
        candidate_footprint = finite_number(
            block["B"]["peak_memory_footprint"],
            "scored candidate footprint",
            positive=True,
        )
        rss_ratio = candidate_rss / baseline_rss
        footprint_ratio = candidate_footprint / baseline_footprint
        finite_number(rss_ratio, "scored RSS ratio", positive=True)
        finite_number(footprint_ratio, "scored footprint ratio", positive=True)
        rss_ratios.append(rss_ratio)
        footprint_ratios.append(footprint_ratio)
    median_saving = statistics.median(savings)
    median_ratio = statistics.median(ratios)
    stratum_medians = {
        order: statistics.median(values) for order, values in stratum_savings.items()
    }
    denominator = FROZEN_FIRST_BYTE_MS - TRANSFER_HAIRCUT * median_saving
    projection = FROZEN_FIRST_BYTE_MS / denominator if denominator > 0 else None
    for label, value in (
        ("median saving", median_saving),
        ("median ratio", median_ratio),
        ("AB median saving", stratum_medians["AB"]),
        ("BA median saving", stratum_medians["BA"]),
        ("projection denominator", denominator),
        ("max paired RSS ratio", max(rss_ratios)),
        ("max paired footprint ratio", max(footprint_ratios)),
    ):
        finite_number(value, label)
    if projection is not None:
        finite_number(projection, "first-byte projection", positive=True)
    ready_medians = {
        arm: statistics.median(
            finite_number(
                block[arm]["result"]["timing"]["ready_wall_ms"],
                f"{arm} summary ready wall",
                positive=True,
            )
            for block in accepted_blocks
        )
        for arm in ("A", "B")
    }
    copy_median = statistics.median(
        finite_number(
            block["B"]["result"]["throughput"]["copy_gbps_decimal"],
            "B summary copy throughput",
            positive=True,
        )
        for block in accepted_blocks
    )
    ready_throughput_medians = {
        arm: statistics.median(
            finite_number(
                block[arm]["result"]["throughput"]["ready_gbps_decimal"],
                f"{arm} summary ready throughput",
                positive=True,
            )
            for block in accepted_blocks
        )
        for arm in ("A", "B")
    }
    gates = {
        "median_saving_at_least_500_ms": median_saving >= 500.0,
        "median_ratio_at_most_0_85": median_ratio <= 0.85,
        "wins_at_least_5_of_6": wins >= 5,
        "ab_saving_at_least_500_ms": stratum_medians["AB"] >= 500.0,
        "ba_saving_at_least_500_ms": stratum_medians["BA"] >= 500.0,
        "ab_wins_at_least_2_of_3": stratum_wins["AB"] >= 2,
        "ba_wins_at_least_2_of_3": stratum_wins["BA"] >= 2,
        "positive_projection_denominator": denominator > 0,
        "first_byte_projection_at_least_1_20x": (
            projection is not None and projection >= 1.20
        ),
        "paired_rss_ratio_at_most_1_05": max(rss_ratios) <= 1.05,
        "paired_footprint_ratio_at_most_1_05": max(footprint_ratios) <= 1.05,
    }
    qualifies = all(gates.values())
    return {
        "schema": 1,
        "status": "go" if qualifies else "kill",
        "authority": "force-only-a3b-parallel-copied-loader-pilot"
        if qualifies
        else "none",
        "required_sequence_before_pilot": "query-capped-auto-prefill",
        "source_commit": manifest["source_commit"],
        "accepted_blocks": len(accepted_blocks),
        "savings_ms_by_block": savings,
        "ratios_by_block": ratios,
        "median_saving_ms": median_saving,
        "median_ratio": median_ratio,
        "wins": wins,
        "stratum_savings_ms": stratum_savings,
        "stratum_median_saving_ms": stratum_medians,
        "stratum_wins": stratum_wins,
        "projected_first_byte_speedup": projection,
        "paired_rss_ratios": rss_ratios,
        "paired_footprint_ratios": footprint_ratios,
        "max_paired_rss_ratio": max(rss_ratios),
        "max_paired_footprint_ratio": max(footprint_ratios),
        "median_ready_wall_ms": ready_medians,
        "median_ready_gbps_decimal": ready_throughput_medians,
        "median_parallel_copy_gbps_decimal": copy_median,
        "gates": gates,
        "qualifies": qualifies,
    }


def main() -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse packet directory {ARTIFACT}")
    ARTIFACT.mkdir(parents=True)
    base_env, removed_environment = common.normalized_environment()
    manifest = build_manifest(removed_environment, base_env)
    (ARTIFACT / "manifest.json").write_text(
        json_text(manifest, pretty=True) + "\n",
        encoding="utf-8",
    )
    (ARTIFACT / "geometry-control.json").write_text(
        json_text(manifest["geometry_control"], pretty=True) + "\n",
        encoding="utf-8",
    )
    attempts_path = ARTIFACT / "attempts.jsonl"
    blocks_path = ARTIFACT / "block-attempts.jsonl"
    decision_path = ARTIFACT / "decision.json"
    accepted_rows = []
    try:
        for block_index, order in enumerate(BLOCK_ORDERS, 1):
            accepted_rows.extend(
                run_valid_block(
                    block_index,
                    order,
                    base_env,
                    manifest,
                    attempts_path,
                    blocks_path,
                )
            )
    except InconclusivePacket as error:
        verify_packet_identity(manifest)
        decision = {
            "schema": 1,
            "status": "inconclusive",
            "authority": "none",
            "source_commit": manifest["source_commit"],
            "accepted_blocks": error.block_index - 1,
            "failed_block_index": error.block_index,
            "failed_block_order": error.order,
            "reason": str(error),
        }
        decision_path.write_text(
            json_text(decision, pretty=True) + "\n", encoding="utf-8"
        )
        print(json_text(decision, pretty=True))
        return
    verify_packet_identity(manifest)
    decision = analyze(accepted_rows, manifest)
    decision_path.write_text(json_text(decision, pretty=True) + "\n", encoding="utf-8")
    print(json_text(decision, pretty=True))


if __name__ == "__main__":
    main()
