#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import v0593_demand_paged_no_copy as common
import v0595_a3b_generic_retained_cold as prior


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0597-a3b-owned-arena-floor-p1"
PREREG = ROOT / "docs/bench/v0597-a3b-owned-arena-floor.md"
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
ARM_NAMES = {
    "A": "copied",
    "B": "arena-serial",
    "C": "arena-four",
}
BLOCK_ORDERS = ("ABC", "BCA", "CAB", "CBA", "ACB", "BAC")
COOLDOWN_S = 30.0
MAX_BLOCK_ATTEMPTS = 3
FROZEN_FIRST_BYTE_MS = 2463.13


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
    build = json.loads(command_text([str(BINARY), "build-info", "--output", "json"]))
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
            raise RuntimeError(
                f"geometry {key} drifted: {row.get(key)!r} != {expected!r}"
            )
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
    describe = json.loads(command_text(describe_command, env=base_env))
    if describe.get("mode") != "describe":
        raise RuntimeError("geometry control mode drifted")
    validate_geometry(describe, build)
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "build_identity": build,
        "removed_environment": removed_environment,
        "sha256": hashes,
        "model_size_bytes": MODEL.stat().st_size,
        "block_orders": list(BLOCK_ORDERS),
        "cooldown_s": COOLDOWN_S,
        "max_block_attempts": MAX_BLOCK_ATTEMPTS,
        "frozen_first_byte_ms": FROZEN_FIRST_BYTE_MS,
        "geometry_control": describe,
        "describe_command": describe_command,
    }


def validate_row(
    row: dict[str, object],
    arm: str,
    build: dict[str, object],
) -> None:
    validate_geometry(row, build)
    if row.get("arm") != ARM_NAMES[arm]:
        raise RuntimeError(f"arm label drifted for {arm}")
    expected_resources = 733 if arm == "A" else 2
    expected_windows = 0 if arm == "A" else 1
    expected_workers = 4 if arm == "C" else 0
    expected_physical = 22_123_538_944 if arm == "A" else 22_123_552_768
    if row.get("resource_count") != expected_resources:
        raise RuntimeError(f"resource count drifted for {arm}")
    if row.get("binding_count") != 733:
        raise RuntimeError(f"binding count drifted for {arm}")
    if row.get("physical_copy_bytes") != expected_physical:
        raise RuntimeError(f"physical-copy bytes drifted for {arm}")
    if row.get("worker_count") != expected_workers:
        raise RuntimeError(f"worker count drifted for {arm}")
    correctness = row.get("correctness")
    if not isinstance(correctness, dict):
        raise RuntimeError("correctness row is missing")
    expected_correctness = {
        "passed": True,
        "full_windows_checked": expected_windows,
        "fallback_bytes_checked": 8_192,
        "entries_checked": 733,
        "aliases_checked": 0,
    }
    if correctness != expected_correctness:
        raise RuntimeError(f"correctness drifted for {arm}: {correctness}")
    timing = row.get("timing")
    if not isinstance(timing, dict) or float(timing.get("ready_wall_ms", 0)) <= 0:
        raise RuntimeError("ready timing is missing")
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
            float(timing[key])
            for key in (
                "allocation_wall_ms",
                "source_resolution_wall_ms",
                "copy_wall_ms",
                "binding_wall_ms",
                "unattributed_wall_ms",
            )
        ]
        if min(components) < 0:
            raise RuntimeError("negative arena timing component")
        if abs(sum(components) - float(timing["ready_wall_ms"])) > 0.01:
            raise RuntimeError("arena timing components do not reconcile")


def run_one(
    arm: str,
    block_index: int,
    attempt: int,
    run_index: int,
    order: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
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
    row = json.loads(stdout_path.read_text(encoding="utf-8"))
    validate_row(row, arm, manifest["build_identity"])
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    host_after_exit = common.capture_host_state()
    vm_after_exit = prior.capture_vm_state()
    block_inputs = common.parse_resource(stderr, "block input operations")
    page_faults = common.parse_resource(stderr, "page faults")
    page_reclaims = common.parse_resource(stderr, "page reclaims")
    maximum_rss = common.parse_resource(stderr, "maximum resident set size")
    peak_footprint = common.parse_resource(stderr, "peak memory footprint")
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
        output.write(json.dumps(row, sort_keys=True) + "\n")


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
        block = {
            "block_index": block_index,
            "block_order": order,
            "attempt": attempt,
            "accepted": accepted,
            "artifact_stems": [row["artifact_stem"] for row in rows],
            "invalid_rows": [
                {
                    "arm": row["arm"],
                    "reasons": row["validity_reasons"],
                }
                for row in rows
                if not row["valid"]
            ],
        }
        append_row(blocks_path, block)
        if accepted:
            return rows
    raise RuntimeError(f"failed to obtain valid block {block_index} order={order}")


def candidate_decision(
    blocks: list[dict[str, dict[str, object]]],
    candidate: str,
) -> dict[str, object]:
    savings = []
    ratios = []
    wins = 0
    for block in blocks:
        baseline_ms = block["A"]["result"]["timing"]["ready_wall_ms"]
        candidate_ms = block[candidate]["result"]["timing"]["ready_wall_ms"]
        savings.append(baseline_ms - candidate_ms)
        ratios.append(candidate_ms / baseline_ms)
        wins += candidate_ms < baseline_ms
    median_saving = statistics.median(savings)
    median_ratio = statistics.median(ratios)
    denominator = FROZEN_FIRST_BYTE_MS - median_saving
    projection = FROZEN_FIRST_BYTE_MS / denominator if denominator > 0 else None
    max_a_rss = max(block["A"]["maximum_resident_set_size"] for block in blocks)
    max_x_rss = max(block[candidate]["maximum_resident_set_size"] for block in blocks)
    max_a_foot = max(block["A"]["peak_memory_footprint"] for block in blocks)
    max_x_foot = max(block[candidate]["peak_memory_footprint"] for block in blocks)
    rss_ratio = max_x_rss / max_a_rss
    footprint_ratio = max_x_foot / max_a_foot
    gates = {
        "median_saving_at_least_500_ms": median_saving >= 500.0,
        "median_ratio_at_most_0_85": median_ratio <= 0.85,
        "wins_at_least_5_of_6": wins >= 5,
        "first_byte_projection_at_least_1_20x": (
            projection is not None and projection >= 1.20
        ),
        "max_rss_ratio_at_most_1_05": rss_ratio <= 1.05,
        "max_footprint_ratio_at_most_1_05": footprint_ratio <= 1.05,
    }
    return {
        "candidate": candidate,
        "arm_name": ARM_NAMES[candidate],
        "savings_ms_by_block": savings,
        "ratios_by_block": ratios,
        "median_saving_ms": median_saving,
        "median_ratio": median_ratio,
        "wins": wins,
        "projected_first_byte_speedup": projection,
        "max_rss_ratio": rss_ratio,
        "max_footprint_ratio": footprint_ratio,
        "gates": gates,
        "qualifies": all(gates.values()),
    }


def analyze(
    rows: list[dict[str, object]], manifest: dict[str, object]
) -> dict[str, object]:
    accepted_blocks = []
    for block_index, order in enumerate(BLOCK_ORDERS, 1):
        matching = [
            row
            for row in rows
            if row["block_index"] == block_index
            and row["block_order"] == order
            and row["valid"]
        ]
        attempts = sorted({row["attempt"] for row in matching})
        if not attempts:
            raise RuntimeError(f"accepted block {block_index} is missing")
        selected_attempt = attempts[-1]
        selected = [row for row in matching if row["attempt"] == selected_attempt]
        if len(selected) != 3:
            raise RuntimeError(f"accepted block {block_index} is incomplete")
        accepted_blocks.append({row["arm"]: row for row in selected})
    decisions = {
        candidate: candidate_decision(accepted_blocks, candidate)
        for candidate in ("B", "C")
    }
    qualifying = [
        candidate for candidate, decision in decisions.items() if decision["qualifies"]
    ]
    selected = None
    if qualifying:
        selected = min(
            qualifying,
            key=lambda candidate: statistics.median(
                block[candidate]["result"]["timing"]["ready_wall_ms"]
                for block in accepted_blocks
            ),
        )
    return {
        "schema": 1,
        "status": "go" if selected else "kill",
        "authority": "force-only-a3b-loader-pilot" if selected else "none",
        "selected_candidate": selected,
        "selected_arm_name": ARM_NAMES[selected] if selected else None,
        "source_commit": manifest["source_commit"],
        "accepted_blocks": len(accepted_blocks),
        "candidate_decisions": decisions,
    }


def main() -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse packet directory {ARTIFACT}")
    ARTIFACT.mkdir(parents=True)
    base_env, removed_environment = common.normalized_environment()
    manifest = build_manifest(removed_environment, base_env)
    (ARTIFACT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ARTIFACT / "geometry-control.json").write_text(
        json.dumps(manifest["geometry_control"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempts_path = ARTIFACT / "attempts.jsonl"
    blocks_path = ARTIFACT / "block-attempts.jsonl"
    accepted_rows = []
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
    decision = analyze(accepted_rows, manifest)
    (ARTIFACT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
