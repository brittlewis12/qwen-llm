#!/usr/bin/env python3

"""v0.624 storage-cold A3B direct-pread prefetch suppression packet."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

import v0602_a3b_parallel_copied_loader as protocol
import v0621_a3b_parallel_pread_auto as predecessor


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0624-a3b-pread-prefetch-suppression-p1"
PREREG = ROOT / "docs/bench/v0624-a3b-pread-prefetch-suppression.md"
BASE_PROTOCOL = ROOT / "scripts/profile/v0602_a3b_parallel_copied_loader.py"
PREDECESSOR_RUNNER = ROOT / "scripts/profile/v0621_a3b_parallel_pread_auto.py"
PRODUCT_PROTOCOL = ROOT / "scripts/profile/v0620_a3b_parallel_pread_product.py"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
EXAMPLE_BINARY = ROOT / "target/release/examples/first_byte_spike"
PREDECESSOR = ROOT / "target/profiles/v0621-a3b-parallel-pread-auto-p1"
PARENT_COMMIT = "bd0c0495bff030b34b022cf22f5e3ef671d5c2be"
PREDECESSOR_COMPLETE_SHA256 = (
    "8321ed29ee33c5abdd5fb42a6e6fb13aa6afc58d9bb6c8fe1858a58d4a395004"
)
PREDECESSOR_DECISION_SHA256 = (
    "e46e0d3966d0d01c950b2bb50042dbe7a63141a7488b9dbcffad4393e784696e"
)
PREDECESSOR_INVENTORY_SHA256 = (
    "cdd0281bc3f1bdbe77fc6fe9f701d40f5f22fe07b823e79beb4d9062a9551827"
)
PAIR_ORDERS = ("AB", "BA", "BA", "AB", "AB", "BA")
MODEL_PAGES = predecessor.MODEL_PAGES
PHYSICAL_READ_MIN_GIB = 20.50
PHYSICAL_READ_MAX_GIB = 20.75
EXPECTED_FIRST_TOKEN = '11751 piece=" Paris"'
EXPECTED_EXAMPLE_SHA256 = (
    "23f05c9f83cbda4c281c59aad6960d2e8d529f1b2b8780901046943f5f09eb8f"
)


class EvidenceDurabilityError(RuntimeError):
    pass


def parse_json_file(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=protocol.reject_json_constant,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def verify_predecessor() -> dict[str, object]:
    manifest, decision = predecessor.verify_inventory(
        PREDECESSOR,
        PREDECESSOR_COMPLETE_SHA256,
        PREDECESSOR_DECISION_SHA256,
        PREDECESSOR_INVENTORY_SHA256,
    )
    cold = decision.get("stages", {}).get("cold")
    if (
        decision.get("status") != "go"
        or decision.get("authority") != "auto-exact-a3b-pread-disposable"
        or decision.get("source_commit") != manifest.get("source_commit")
        or not isinstance(cold, dict)
        or cold.get("admission_passes") is not True
    ):
        raise RuntimeError("v0.621 predecessor authority drifted")
    return {
        "packet_complete_sha256": PREDECESSOR_COMPLETE_SHA256,
        "decision_sha256": PREDECESSOR_DECISION_SHA256,
        "inventory_sha256": PREDECESSOR_INVENTORY_SHA256,
        "source_commit": decision["source_commit"],
        "authority": decision["authority"],
    }


def required_manifest_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PREREG,
        BASE_PROTOCOL,
        PREDECESSOR_RUNNER,
        PRODUCT_PROTOCOL,
        COMMON_RUNNER,
        protocol.MODEL,
        protocol.CLI_BINARY,
        protocol.BENCH_BINARY,
        EXAMPLE_BINARY,
        PREDECESSOR / "packet-complete.json",
        PREDECESSOR / "decision.json",
        PREDECESSOR / "artifact-inventory.sha256",
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    tracked = (Path(__file__).resolve(), PREREG)
    for path in tracked:
        protocol.command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    commit = protocol.command_text(["git", "rev-parse", "HEAD"]).strip()
    dirty = protocol.command_text(["git", "status", "--porcelain=v1"]).strip()
    if dirty:
        raise RuntimeError(f"source is dirty: {dirty!r}")
    parent = protocol.command_text(["git", "rev-parse", "HEAD^"]).strip()
    changed = set(
        protocol.command_text(
            ["git", "diff", "--name-only", f"{PARENT_COMMIT}..{commit}"]
        ).splitlines()
    )
    expected = {str(path.relative_to(ROOT)) for path in tracked}
    if parent != PARENT_COMMIT or changed != expected:
        raise RuntimeError(
            f"packet source boundary drifted: parent={parent} changed={sorted(changed)}"
        )
    build = protocol.parse_json(
        protocol.command_text(
            [str(protocol.BENCH_BINARY), "build-info", "--output", "json"]
        )
    )
    if not isinstance(build, dict):
        raise RuntimeError("build identity is not an object")
    if (
        build.get("build_commit") != commit
        or build.get("runtime_commit") != commit
        or build.get("status") != "match"
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
        or build.get("build_source_state") != build.get("runtime_source_state")
    ):
        raise RuntimeError(f"source/build identity mismatch: {build}")
    return commit, build


def build_manifest(
    removed_environment: list[str], base_env: dict[str, str]
) -> dict[str, object]:
    imported = verify_predecessor()
    commit, build = source_and_build_identity()
    paths = required_manifest_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing packet inputs: {missing}")
    hashes = {str(path): protocol.common.sha256_file(path) for path in paths}
    if hashes[str(protocol.MODEL)] != protocol.EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA-256 drifted")
    if hashes[str(EXAMPLE_BINARY)] != EXPECTED_EXAMPLE_SHA256:
        raise RuntimeError("sealed v0.621 first_byte_spike binary drifted")
    device = protocol.command_text(
        [str(protocol.CLI_BINARY), "--info"], env=base_env
    ).strip()
    macos = protocol.command_text(["sw_vers", "-productVersion"], env=base_env).strip()
    hw_memsize = int(
        protocol.command_text(["sysctl", "-n", "hw.memsize"], env=base_env)
    )
    if (
        device != protocol.EXPECTED_DEVICE
        or not macos.startswith("15.")
        or hw_memsize != protocol.EXPECTED_HW_MEMSIZE
    ):
        raise RuntimeError("host boundary drifted")
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "build_identity": build,
        "device": device,
        "macos": macos,
        "hw_memsize": hw_memsize,
        "removed_environment": removed_environment,
        "child_environment": protocol.child_environment_record(base_env),
        "time_resource_probe": protocol.preflight_time_resources(base_env),
        "sha256": hashes,
        "model_size_bytes": protocol.MODEL.stat().st_size,
        "pair_orders": list(PAIR_ORDERS),
        "cooldown_s": protocol.COOLDOWN_S,
        "host_sample_limit": protocol.HOST_SAMPLE_LIMIT,
        "host_sample_interval_s": protocol.HOST_SAMPLE_INTERVAL_S,
        "child_retry_count": 0,
        "imported_predecessor": imported,
        "physical_read_window_gib": [PHYSICAL_READ_MIN_GIB, PHYSICAL_READ_MAX_GIB],
        "gates": {
            "overall_saving_ms": 400.0,
            "stratum_saving_ms": 250.0,
            "physical_read_ratio": [0.98, 1.02],
            "total_cpu_ratio_max": 1.10,
            "footprint_ratio_max": 1.05,
            "passthrough_median_ms": 100.0,
            "passthrough_pair_max_ms": 250.0,
        },
    }


def parse_process_times(stderr: str) -> dict[str, float]:
    return predecessor.parse_process_times(stderr)


def unique_match(pattern: str, text: str, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) != 1:
        raise RuntimeError(f"{label} occurrence count drifted: {len(matches)}")
    return matches[0]


def parse_cold_stdout(stdout: str, arm: str) -> dict[str, object]:
    model = unique_match(r"^model:\s+(.+)$", stdout, "model")
    size = unique_match(r"^size:\s+([0-9.]+) GiB$", stdout, "model size")
    intent = unique_match(r"^intent:\s+(.+)$", stdout, "intent")
    invalidate_flag = unique_match(
        r"^invalidate:\s+(true|false)$",
        stdout,
        "invalidate flag",
    )
    prompt = unique_match(r"^prompt:\s+(.+)$", stdout, "prompt")
    encoded = unique_match(
        r"^\s+prompt encoded to (\d+) tokens: (.+)$",
        stdout,
        "encoded prompt",
    )
    pre_arm = unique_match(
        r"^pre-arm residency: (\d+)/(\d+) pages \(([0-9]+\.[0-9])%\)$",
        stdout,
        "pre-arm residency",
    )
    invalidation = unique_match(
        r"^invalidate: (\d+)/(\d+) -> (\d+)/(\d+)$",
        stdout,
        "invalidation",
    )
    load = unique_match(
        r"^load:\s+([0-9.]+) s\s+pageins=\s*(\d+)\s+diskR=\s*([0-9.]+) GiB$",
        stdout,
        "load",
    )
    first = unique_match(
        r"^FIRST BYTE:\s+([0-9.]+) s$",
        stdout,
        "first byte",
    )
    total = unique_match(
        r"^rusage total: pageins=\s*(\d+)\s+diskR=\s*([0-9.]+) GiB\s+"
        r"diskW=\s*([0-9.]+) MiB\s+\u0394RSS=\s*([+-][0-9.]+) GiB$",
        stdout,
        "total resources",
    )
    token = unique_match(
        r"^first token: id=(\d+) piece=(.*)$",
        stdout,
        "first token",
    )
    post = unique_match(
        r"^post-arm residency: (\d+)/(\d+) pages \(([0-9]+\.[0-9])%\)$",
        stdout,
        "post-arm residency",
    )
    policy = unique_match(r"^policy:\s+(.+)$", stdout, "policy")
    if (
        model.group(1) != str(protocol.MODEL)
        or float(size.group(1)) != 20.61
        or intent.group(1) != "DisposableSingleTurn"
        or invalidate_flag.group(1) != "true"
        or prompt.group(1) != '"The capital of France is"'
        or encoded.group(1) != "5"
        or encoded.group(2) != "[760, 6511, 314, 9338, 369]"
    ):
        raise RuntimeError("cold product-shape output drifted")
    pre_resident, pre_total, pre_percent = pre_arm.groups()
    before, total_before, after, total_after = map(int, invalidation.groups())
    post_resident, post_total, post_percent = post.groups()
    expected_pre_percent = round(int(pre_resident) / int(pre_total) * 100.0, 1)
    expected_post_percent = round(int(post_resident) / int(post_total) * 100.0, 1)
    if (
        int(pre_resident) != MODEL_PAGES
        or int(pre_total) != MODEL_PAGES
        or float(pre_percent) != expected_pre_percent
        or before != MODEL_PAGES
        or total_before != MODEL_PAGES
        or after != 0
        or total_after != MODEL_PAGES
        or int(post_total) != MODEL_PAGES
        or int(post_resident) / int(post_total) < 0.99
        or float(post_percent) != expected_post_percent
    ):
        raise RuntimeError("cold residency contract drifted")
    expected_policy = (
        "ColdOnly { threshold: ResidencyThreshold(0.9) }" if arm == "A" else "Off"
    )
    if policy.group(1) != expected_policy:
        raise RuntimeError("prefetch policy output drifted")
    prefetch_lines = [
        line for line in stdout.splitlines() if line.strip().startswith("prefetch:")
    ]
    prefetch_matches = re.findall(
        r"^\s+prefetch:\s+([0-9.]+) s\s+(\d+) shards prefetched, "
        r"(\d+) skipped, ([0-9.]+) GiB returned$",
        stdout,
        re.MULTILINE,
    )
    if arm == "A":
        if len(prefetch_lines) != 1 or len(prefetch_matches) != 1:
            raise RuntimeError("ColdOnly prefetch phase drifted")
        wall_s, prefetched, skipped, returned_gib = prefetch_matches[0]
        if int(prefetched) != 1 or int(skipped) != 0 or float(returned_gib) != 20.61:
            raise RuntimeError("ColdOnly did not prefetch the complete shard")
        prefetch = {
            "wall_s": float(wall_s),
            "shards_prefetched": int(prefetched),
            "shards_skipped": int(skipped),
            "returned_gib": float(returned_gib),
        }
    else:
        if prefetch_lines or prefetch_matches:
            raise RuntimeError("Off arm unexpectedly emitted a prefetch phase")
        prefetch = None
    disk_gib = float(total.group(2))
    if not PHYSICAL_READ_MIN_GIB <= disk_gib <= PHYSICAL_READ_MAX_GIB:
        raise RuntimeError("cold child physical-read window missed")
    first_token = f"{token.group(1)} piece={token.group(2)}"
    if first_token != EXPECTED_FIRST_TOKEN:
        raise RuntimeError(f"cold first token drifted: {first_token!r}")
    return {
        "pre_arm_resident_pages": int(pre_resident),
        "invalidated_before_pages": before,
        "invalidated_after_pages": after,
        "load_s": float(load.group(1)),
        "load_pageins": int(load.group(2)),
        "load_disk_gib": float(load.group(3)),
        "first_byte_s": float(first.group(1)),
        "total_pageins": int(total.group(1)),
        "total_disk_gib": disk_gib,
        "total_disk_write_mib": float(total.group(3)),
        "total_rss_delta_gib": float(total.group(4)),
        "first_token": first_token,
        "post_resident_pages": int(post_resident),
        "post_resident_fraction": int(post_resident) / int(post_total),
        "prefetch": prefetch,
    }


def fsync_raw_artifacts(paths: tuple[Path, ...]) -> None:
    try:
        for path in paths:
            if not path.exists():
                continue
            with path.open("rb+") as artifact:
                artifact.flush()
                os.fsync(artifact.fileno())
        protocol.fsync_directory(ARTIFACT)
    except OSError as error:
        raise EvidenceDurabilityError(
            f"raw child artifact fsync failed: {type(error).__name__}: {error}"
        ) from error


def run_child(
    arm: str,
    pair_index: int,
    order: str,
    position: int,
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
    stem = f"cold-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    for path in (stdout_path, stderr_path):
        if path.exists():
            raise RuntimeError(f"refusing to reuse child artifact {path}")
    conditioning = protocol.condition_for_child(stem, manifest)
    env = base_env.copy()
    policy = "cold-only" if arm == "A" else "off"
    command = [
        "/usr/bin/time",
        "-l",
        str(EXAMPLE_BINARY),
        str(protocol.MODEL),
        "--policy",
        policy,
        "--invalidate",
        "--prompt",
        "The capital of France is",
        "--tokens",
        "0",
        "--intent",
        "disposable",
    ]
    protocol.record_launch("cold", stem, command, arm, pair_index, position)
    process = None
    wait_errors: list[str] = []
    started = time.perf_counter()
    try:
        with (
            stdout_path.open("xb") as stdout_file,
            stderr_path.open("xb") as stderr_file,
        ):
            prior_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                returncode, wait_errors = protocol.wait_for_child(process)
                process_wall_ms = (time.perf_counter() - started) * 1e3
            finally:
                signal.signal(signal.SIGINT, prior_sigint)
        fsync_raw_artifacts((stdout_path, stderr_path))
    except (OSError, KeyboardInterrupt) as error:
        error_text = f"{type(error).__name__}:{error}"
        if process is None:
            fsync_raw_artifacts((stdout_path, stderr_path))
            protocol.record_completion("cold", stem, None, error_text)
            if isinstance(error, OSError):
                protocol.record_spawn_failure_evidence(
                    "cold", stem, conditioning, error
                )
            raise protocol.InconclusivePacket(
                "cold", stem, [f"child_spawn_or_pipe_failed={error_text}"]
            ) from error
        returncode, deferred = protocol.wait_for_child(process)
        process_wall_ms = (time.perf_counter() - started) * 1e3
        wait_errors.extend(deferred)
        wait_errors.append(error_text)
        fsync_raw_artifacts((stdout_path, stderr_path))
    protocol.record_completion(
        "cold",
        stem,
        returncode,
        ";".join(wait_errors) if wait_errors else None,
    )
    post_exit = protocol.capture_post_exit_state(conditioning)
    protocol.record_post_exit_state("cold", stem, returncode, post_exit)
    stdout_bytes = stdout_path.read_bytes()
    stderr_bytes = stderr_path.read_bytes()
    stdout = stdout_bytes.decode("utf-8")
    stderr = stderr_bytes.decode("utf-8")
    reasons = list(post_exit["child_interval"]["failure_reasons"])
    if not post_exit["host_after_exit"]["valid"]:
        reasons.append("post_exit_host_invalid")
    if wait_errors:
        reasons.append(f"child_wait_interrupted={';'.join(wait_errors)}")
    if returncode != 0:
        reasons.append(f"child_nonzero_exit={returncode}")
    try:
        resources = protocol.process_resources(stderr)
    except Exception as error:
        resources = None
        reasons.append(
            f"child_process_resource_parse_invalid={type(error).__name__}:{error}"
        )
    if resources is not None:
        if resources["swaps"] != 0:
            reasons.append("child_swaps")
        if resources["block_input_operations"] != 0:
            reasons.append("child_block_input")
    try:
        times = parse_process_times(stderr)
    except Exception as error:
        times = {
            "real_s": None,
            "user_s": None,
            "system_s": None,
            "total_cpu_s": None,
        }
        reasons.append(
            f"child_process_time_parse_invalid={type(error).__name__}:{error}"
        )
    if returncode == 0:
        parsed = parse_cold_stdout(stdout, arm)
        load_contract = predecessor.parse_load_contract(stderr, "B")
    else:
        parsed = None
        load_contract = predecessor.parse_observed_load_contract(stderr, "B")
    validity = {**post_exit, "process_resources": resources}
    protocol.record_post_exit_evidence("cold", stem, returncode, validity, reasons)
    return {
        "stage": "cold",
        "artifact_stem": stem,
        "arm": arm,
        "pair_index": pair_index,
        "pair_order": order,
        "position": position,
        "command": command,
        "arm_environment": {
            "QWEN_GGUF_PARALLEL_COPY": None,
            "prefetch_policy": policy,
        },
        **conditioning,
        **validity,
        **times,
        "process_wall_ms": process_wall_ms,
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "load_contract": load_contract,
        "cold": parsed,
        "valid": not reasons,
        "validity_reasons": reasons,
    }


def run_stage(
    base_env: dict[str, str],
    manifest: dict[str, object],
    attempts_path: Path,
) -> list[dict[str, object]]:
    rows = []
    for pair_index, order in enumerate(PAIR_ORDERS, 1):
        for position, arm in enumerate(order, 1):
            row = run_child(
                arm,
                pair_index,
                order,
                position,
                base_env,
                manifest,
            )
            protocol.append_row(attempts_path, row)
            rows.append(row)
            if not row["valid"]:
                raise protocol.InconclusivePacket(
                    "cold", row["artifact_stem"], list(row["validity_reasons"])
                )
    if len(rows) != 12:
        raise RuntimeError("cold child count drifted")
    return rows


def ratio(numerator: float, denominator: float, label: str) -> float:
    if not all(
        math.isfinite(value) and value > 0 for value in (numerator, denominator)
    ):
        raise RuntimeError(f"{label} ratio inputs are invalid")
    return numerator / denominator


def stratum_median(metrics: list[dict[str, object]], order: str, key: str) -> float:
    values = [float(row[key]) for row in metrics if row["pair_order"] == order]
    if len(values) != 3:
        raise RuntimeError(f"{order} stratum size drifted for {key}")
    return statistics.median(values)


def analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 12:
        raise RuntimeError("cold analysis requires 12 rows")
    metrics = []
    for pair_index, order in enumerate(PAIR_ORDERS, 1):
        pair = [row for row in rows if row["pair_index"] == pair_index]
        if len(pair) != 2 or [row["arm"] for row in pair] != list(order):
            raise RuntimeError(f"cold pair {pair_index} membership drifted")
        by_arm = {row["arm"]: row for row in pair}
        if set(by_arm) != {"A", "B"}:
            raise RuntimeError(f"cold pair {pair_index} arms drifted")
        a = by_arm["A"]
        b = by_arm["B"]
        load_saving_ms = (a["cold"]["load_s"] - b["cold"]["load_s"]) * 1e3
        first_saving_ms = (a["cold"]["first_byte_s"] - b["cold"]["first_byte_s"]) * 1e3
        physical_read_ratio = ratio(
            b["cold"]["total_disk_gib"],
            a["cold"]["total_disk_gib"],
            "physical_read",
        )
        cpu_ratio = ratio(b["total_cpu_s"], a["total_cpu_s"], "total_cpu")
        footprint_ratio = ratio(
            b["process_resources"]["peak_memory_footprint"],
            a["process_resources"]["peak_memory_footprint"],
            "footprint",
        )
        passthrough_ms = abs(load_saving_ms - first_saving_ms)
        row = {
            "pair_index": pair_index,
            "pair_order": order,
            "load_saving_ms": load_saving_ms,
            "first_byte_saving_ms": first_saving_ms,
            "process_wall_saving_ms": a["process_wall_ms"] - b["process_wall_ms"],
            "physical_read_b_over_a": physical_read_ratio,
            "total_cpu_b_over_a": cpu_ratio,
            "footprint_b_over_a": footprint_ratio,
            "passthrough_abs_delta_ms": passthrough_ms,
            "a": a,
            "b": b,
        }
        row["pair_gates"] = {
            "load_b_wins": load_saving_ms > 0,
            "first_byte_b_wins": first_saving_ms > 0,
            "physical_read_ratio": 0.98 <= physical_read_ratio <= 1.02,
            "total_cpu_ratio": cpu_ratio <= 1.10,
            "footprint_ratio": footprint_ratio <= 1.05,
            "passthrough": passthrough_ms <= 250.0,
        }
        metrics.append(row)
    load_values = [float(row["load_saving_ms"]) for row in metrics]
    first_values = [float(row["first_byte_saving_ms"]) for row in metrics]
    passthrough_values = [float(row["passthrough_abs_delta_ms"]) for row in metrics]
    load_ab = stratum_median(metrics, "AB", "load_saving_ms")
    load_ba = stratum_median(metrics, "BA", "load_saving_ms")
    first_ab = stratum_median(metrics, "AB", "first_byte_saving_ms")
    first_ba = stratum_median(metrics, "BA", "first_byte_saving_ms")
    gates = {
        "all_pair_gates": all(all(row["pair_gates"].values()) for row in metrics),
        "load_wins_6_of_6": all(value > 0 for value in load_values),
        "first_byte_wins_6_of_6": all(value > 0 for value in first_values),
        "load_overall_median": statistics.median(load_values) >= 400.0,
        "first_byte_overall_median": statistics.median(first_values) >= 400.0,
        "load_ab_median": load_ab >= 250.0,
        "load_ba_median": load_ba >= 250.0,
        "first_byte_ab_median": first_ab >= 250.0,
        "first_byte_ba_median": first_ba >= 250.0,
        "passthrough_median": statistics.median(passthrough_values) <= 100.0,
    }
    return {
        "stage": "cold",
        "pairs": metrics,
        "load_saving_ms": {
            "values": load_values,
            "median": statistics.median(load_values),
            "ab_median": load_ab,
            "ba_median": load_ba,
        },
        "first_byte_saving_ms": {
            "values": first_values,
            "median": statistics.median(first_values),
            "ab_median": first_ab,
            "ba_median": first_ba,
        },
        "passthrough_abs_delta_ms": {
            "values": passthrough_values,
            "median": statistics.median(passthrough_values),
            "maximum": max(passthrough_values),
        },
        "gates": gates,
        "passes": all(gates.values()),
    }


def configure_protocol() -> None:
    protocol.ARTIFACT = ARTIFACT
    protocol.PREREG = PREREG
    protocol.required_manifest_paths = required_manifest_paths
    protocol.source_and_build_identity = source_and_build_identity
    protocol.build_manifest = build_manifest


def write_decision(decision: dict[str, object], manifest: dict[str, object]) -> None:
    if decision.get("status") == "go":
        imported = verify_predecessor()
        stage = decision.get("stages", {}).get("cold")
        if (
            decision.get("authority")
            != "one-a3b-coldonly-suppression-selector-implementation"
            or decision.get("source_commit") != manifest.get("source_commit")
            or decision.get("imported_predecessor")
            != manifest.get("imported_predecessor")
            or imported != manifest.get("imported_predecessor")
            or not isinstance(stage, dict)
            or stage.get("passes") is not True
        ):
            raise RuntimeError("v0.624 GO authority conjunction is incomplete")
    elif decision.get("authority") != "none":
        raise RuntimeError("non-GO v0.624 decision carries authority")
    protocol.write_identity_checked_decision(decision, manifest)


def run(*, preflight_only: bool) -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse packet directory {ARTIFACT}")
    base_env, removed_environment = protocol.common.normalized_environment()
    manifest = build_manifest(removed_environment, base_env)
    if preflight_only:
        vm_state = protocol.capture_vm_state()
        if vm_state["capture_errors"]:
            raise RuntimeError(
                f"VM preflight capture failed: {vm_state['capture_errors']}"
            )
        print(
            protocol.json_text(
                {
                    "status": "preflight-passed",
                    "source_commit": manifest["source_commit"],
                    "manifest_sha256": hashlib.sha256(
                        (protocol.json_text(manifest, pretty=True) + "\n").encode()
                    ).hexdigest(),
                },
                pretty=True,
            )
        )
        return
    protocol.reserve_artifact(manifest)
    attempts_path = ARTIFACT / "attempts.jsonl"
    stages = {}
    try:
        rows = run_stage(base_env, manifest, attempts_path)
        stages["cold"] = analyze(rows)
        status = "go" if stages["cold"]["passes"] else "kill"
        authority = (
            "one-a3b-coldonly-suppression-selector-implementation"
            if status == "go"
            else "none"
        )
        write_decision(
            {
                "schema": 1,
                "status": status,
                "authority": authority,
                "stopped_after": "cold",
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
    except protocol.InconclusivePacket as error:
        write_decision(
            {
                "schema": 1,
                "status": "inconclusive",
                "authority": "none",
                "stopped_after": error.stage,
                "failed_child": error.child,
                "reasons": error.reasons,
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
    except KeyboardInterrupt:
        write_decision(
            {
                "schema": 1,
                "status": "inconclusive",
                "authority": "none",
                "stopped_after": "operator-interrupt",
                "reasons": ["operator_interrupt_after_child_cleanup"],
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
    except EvidenceDurabilityError:
        raise
    except Exception as error:
        if protocol.decision_publication_started():
            raise
        write_decision(
            {
                "schema": 1,
                "status": "implementation_or_contract_defect",
                "authority": "none",
                "stopped_after": "cold",
                "error_type": type(error).__name__,
                "error": str(error),
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    configure_protocol()
    run(preflight_only=arguments.preflight_only)
