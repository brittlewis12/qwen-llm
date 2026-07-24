#!/usr/bin/env python3

"""v0.621 actual-Auto admission and storage-cold composition guard."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

import v0602_a3b_parallel_copied_loader as protocol
import v0620_a3b_parallel_pread_product as product


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0621-a3b-parallel-pread-auto-p1"
PREREG = ROOT / "docs/bench/v0621-a3b-parallel-pread-auto.md"
BASE_PROTOCOL = ROOT / "scripts/profile/v0602_a3b_parallel_copied_loader.py"
PRODUCT_PROTOCOL = ROOT / "scripts/profile/v0620_a3b_parallel_pread_product.py"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
PREDECESSOR = ROOT / "target/profiles/v0620-a3b-parallel-pread-product-p1"
PREDECESSOR_COMPLETE_SHA256 = (
    "b11eaa87eb5b5c44767f427fb58eaab11dce005636b5b62cb5a389627046a1b2"
)
PREDECESSOR_DECISION_SHA256 = (
    "b60b2ea1755b7f4563b2c31a630271722cd057d76dd7a4c110c61615a211e4cd"
)
PREDECESSOR_INVENTORY_SHA256 = (
    "18916efb2daf43a48fa4fd5ad0f70fa13903996aeded158360867c54540e57ac"
)
SELECTOR_COMMIT = "532f7a5db8aaa73fa1ad8f8cd1fde17ae8191413"
SELECTOR_PARENT = "545c264b84c6189ec2175ca95da67f3c65b88dfe"
EXAMPLE_BINARY = ROOT / "target/release/examples/first_byte_spike"
EXPECTED_STDOUT_SHA256 = product.EXPECTED_STDOUT_SHA256
AUTO_POLICY_LINE = "[metal-gguf-parallel-policy] mode=auto profile=a3b-q4km-v1"
COPIED_MARKER = product.COPIED_MARKER
PREAD_MARKER = product.PREAD_MARKER
MODEL_PAGES = 1_350_985
MIN_COLD_DISK_GIB = 16.50
BASE_RUN_FRESH_CHILD = protocol.run_fresh_child
COLD_PAIRS = (
    ("MP", "M", "P"),
    ("PM", "P", "M"),
)


def parse_json_file(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=protocol.reject_json_constant,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def verify_inventory(
    root: Path,
    complete_sha256: str,
    decision_sha256: str,
    inventory_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    complete_path = root / "packet-complete.json"
    decision_path = root / "decision.json"
    inventory_path = root / "artifact-inventory.sha256"
    expected_hashes = {
        complete_path: complete_sha256,
        decision_path: decision_sha256,
        inventory_path: inventory_sha256,
    }
    for path, expected in expected_hashes.items():
        if protocol.common.sha256_file(path) != expected:
            raise RuntimeError(f"predecessor {path.name} digest drifted")
    complete = parse_json_file(complete_path)
    if (
        complete.get("schema") != 1
        or complete.get("decision_sha256") != decision_sha256
        or complete.get("inventory_sha256") != inventory_sha256
    ):
        raise RuntimeError("predecessor completion seal drifted")
    listed = []
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise RuntimeError("predecessor inventory row is malformed")
        path = ROOT / relative
        if path.parent != root or path.name in {
            "artifact-inventory.sha256",
            "packet-complete.json",
        }:
            raise RuntimeError("predecessor inventory path escapes its packet")
        if protocol.common.sha256_file(path) != digest:
            raise RuntimeError(f"predecessor inventory drifted at {path.name}")
        listed.append(path)
    actual = {path for path in root.iterdir() if path.is_file()}
    expected = set(listed) | {inventory_path, complete_path}
    if len(listed) != len(set(listed)) or actual != expected:
        raise RuntimeError("predecessor packet inventory drifted")
    return parse_json_file(root / "manifest.json"), parse_json_file(decision_path)


def verify_predecessor() -> dict[str, object]:
    manifest, decision = verify_inventory(
        PREDECESSOR,
        PREDECESSOR_COMPLETE_SHA256,
        PREDECESSOR_DECISION_SHA256,
        PREDECESSOR_INVENTORY_SHA256,
    )
    fresh = decision.get("stages", {}).get("fresh_128")
    imported = decision.get("imported_predecessor")
    if (
        decision.get("status") != "go"
        or decision.get("authority") != "force-only-exact-a3b-pread-over-forced-copy"
        or decision.get("source_commit") != manifest.get("source_commit")
        or not isinstance(fresh, dict)
        or fresh.get("passes") is not True
        or fresh.get("global_output_sha256") != EXPECTED_STDOUT_SHA256
        or not isinstance(imported, dict)
        or imported.get("loaded_pairs") != 6
    ):
        raise RuntimeError("predecessor decision contract drifted")
    if product.verify_predecessor() != imported:
        raise RuntimeError("predecessor v0.619 bridge drifted")
    return {
        "packet_complete_sha256": PREDECESSOR_COMPLETE_SHA256,
        "decision_sha256": PREDECESSOR_DECISION_SHA256,
        "inventory_sha256": PREDECESSOR_INVENTORY_SHA256,
        "source_commit": decision["source_commit"],
        "global_output_sha256": fresh["global_output_sha256"],
        "load_saving_median_ms": fresh["load_saving_ms"]["median"],
        "first_byte_saving_median_ms": fresh["first_byte_saving_ms"]["median"],
        "exit_saving_median_ms": fresh["exit_saving_ms"]["median"],
    }


def required_manifest_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PREREG,
        BASE_PROTOCOL,
        PRODUCT_PROTOCOL,
        COMMON_RUNNER,
        protocol.MODEL,
        protocol.PROMPT,
        protocol.CLI_BINARY,
        protocol.BENCH_BINARY,
        EXAMPLE_BINARY,
        PREDECESSOR / "packet-complete.json",
        PREDECESSOR / "decision.json",
        PREDECESSOR / "artifact-inventory.sha256",
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    tracked = (
        Path(__file__).resolve(),
        PREREG,
        BASE_PROTOCOL,
        PRODUCT_PROTOCOL,
        COMMON_RUNNER,
        protocol.PROMPT,
    )
    for path in tracked:
        protocol.command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    commit = protocol.command_text(["git", "rev-parse", "HEAD"]).strip()
    dirty = protocol.command_text(["git", "status", "--porcelain=v1"]).strip()
    if dirty:
        raise RuntimeError(f"source is dirty: {dirty!r}")
    selector_parent = protocol.command_text(
        ["git", "rev-parse", f"{SELECTOR_COMMIT}^"]
    ).strip()
    packet_parent = protocol.command_text(["git", "rev-parse", "HEAD^"]).strip()
    selector_paths = set(
        protocol.command_text(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", SELECTOR_COMMIT]
        ).splitlines()
    )
    if (
        selector_parent != SELECTOR_PARENT
        or packet_parent != SELECTOR_COMMIT
        or selector_paths != {"crates/qwen-llm/src/metal_forward.rs"}
    ):
        raise RuntimeError("Auto selector evidence bridge drifted")
    changed = set(
        protocol.command_text(
            ["git", "diff", "--name-only", f"{SELECTOR_COMMIT}..{commit}"]
        ).splitlines()
    )
    expected_changed = {
        str(PREREG.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
    }
    if changed != expected_changed:
        raise RuntimeError(f"packet source delta drifted: {sorted(changed)}")
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
    embedded = (commit.encode(), str(build["build_source_state"]).encode())
    qwen_binary = protocol.CLI_BINARY.read_bytes()
    if any(value not in qwen_binary for value in embedded):
        raise RuntimeError("qwen identity is not embedded")
    return commit, build


def build_manifest(
    removed_environment: list[str], base_env: dict[str, str]
) -> dict[str, object]:
    predecessor = verify_predecessor()
    commit, build = source_and_build_identity()
    paths = required_manifest_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing packet inputs: {missing}")
    hashes = {str(path): protocol.common.sha256_file(path) for path in paths}
    if hashes[str(protocol.MODEL)] != protocol.EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA-256 drifted")
    if hashes[str(protocol.PROMPT)] != protocol.EXPECTED_PROMPT_SHA256:
        raise RuntimeError("prompt SHA-256 drifted")
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
        "prompt_bytes": protocol.PROMPT.stat().st_size,
        "prompt_tokens": protocol.EXPECTED_PROMPT_TOKENS,
        "output_tokens": protocol.EXPECTED_OUTPUT_TOKENS,
        "transition_count": protocol.EXPECTED_TRANSITIONS,
        "pair_orders": list(protocol.PAIR_ORDERS),
        "cold_pairs": [list(pair) for pair in COLD_PAIRS],
        "cooldown_s": protocol.COOLDOWN_S,
        "host_sample_limit": protocol.HOST_SAMPLE_LIMIT,
        "host_sample_interval_s": protocol.HOST_SAMPLE_INTERVAL_S,
        "child_retry_count": 0,
        "imported_predecessor": predecessor,
        "expected_stdout_sha256": EXPECTED_STDOUT_SHA256,
    }


def arm_environment(arm: str) -> dict[str, str | None]:
    if arm not in ("A", "B"):
        raise ValueError(f"unknown arm {arm!r}")
    return {
        "QWEN_GGUF_PARALLEL_COPY": "1" if arm == "A" else None,
        "QWEN_GGUF_OWNED_ARENA": None,
        "QWEN_GGUF_NO_COPY": None,
        "QWEN_GGUF_NO_COPY_PREFAULT": None,
        "QWEN_NATIVE_QUANT_EMBED": None,
        "QWEN_MOE_ROUTER_F16": None,
        "QWEN_MOE_IQ3_EXPERT_NATIVE": None,
        "QWEN_PREFILL_MOE_GROUPED_IQ3_GATEUP": None,
        "QWEN_PREFILL_ATTN_FUSED_QKV_G8": None,
    }


def expected_load_lines(arm: str) -> tuple[str, ...]:
    if arm == "A":
        return (protocol.POLICY_LINE, COPIED_MARKER, protocol.LEDGER_LINE)
    return (
        protocol.POLICY_LINE,
        AUTO_POLICY_LINE,
        PREAD_MARKER,
        protocol.LEDGER_LINE,
    )


def parse_load_contract(stderr: str, arm: str) -> dict[str, object]:
    expected = expected_load_lines(arm)
    marker = COPIED_MARKER if arm == "A" else PREAD_MARKER
    marker_count = 1 if arm == "A" else 2
    if stderr.count("[metal-gguf-") != marker_count:
        raise RuntimeError(f"{arm} Metal policy/marker count drifted")
    if stderr.count(marker) != 1:
        raise RuntimeError(f"{arm} population marker count drifted")
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-policy]",
        COPIED_MARKER,
        PREAD_MARKER,
        "[metal-load-ledger]",
    )
    recognized = [line for line in stderr.splitlines() if line.startswith(prefixes)]
    if len(recognized) != len(expected):
        raise RuntimeError(f"{arm} load-line count drifted: {recognized!r}")
    for line, required in zip(recognized, expected, strict=True):
        if required in (COPIED_MARKER, PREAD_MARKER):
            product.parse_marker(line, required)
        elif line != required:
            raise RuntimeError(f"{arm} load-line ordering drifted: {recognized!r}")
    return {
        "storage": "parallel-copied" if arm == "A" else "auto-parallel-pread",
        "policy": None if arm == "A" else AUTO_POLICY_LINE,
        "marker": next(line for line in recognized if line.startswith(marker)),
        "phase_us": product.parse_marker(
            next(line for line in recognized if line.startswith(marker)), marker
        ),
    }


def parse_observed_load_contract(stderr: str, arm: str) -> dict[str, object] | None:
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-policy]",
        COPIED_MARKER,
        PREAD_MARKER,
        "[metal-load-ledger]",
    )
    recognized = [line for line in stderr.splitlines() if line.startswith(prefixes)]
    gguf_lines = [
        line for line in stderr.splitlines() if line.startswith("[metal-gguf-")
    ]
    if sum(line.startswith("[metal-gguf-") for line in recognized) != len(gguf_lines):
        raise RuntimeError("observed unrequested Metal storage marker")
    if not recognized:
        return None
    expected = expected_load_lines(arm)
    if len(recognized) > len(expected):
        raise RuntimeError("observed load contract has extra lines")
    for line, required in zip(recognized, expected, strict=False):
        if required in (COPIED_MARKER, PREAD_MARKER):
            product.parse_marker(line, required)
        elif line != required:
            raise RuntimeError("observed load contract contradicts expected prefix")
    if len(recognized) < len(expected):
        return {"status": "incomplete-valid-prefix", "recognized_lines": recognized}
    return {"status": "complete", "contract": parse_load_contract(stderr, arm)}


def run_fresh_child(*args: object, **kwargs: object) -> dict[str, object]:
    row = BASE_RUN_FRESH_CHILD(*args, **kwargs)
    if row.get("valid") is True and row.get("stdout_sha256") != EXPECTED_STDOUT_SHA256:
        raise RuntimeError("fresh child generated stdout digest drifted")
    return row


def analyze_fresh(rows: list[dict[str, object]]) -> dict[str, object]:
    analysis = product.pread_protocol.analyze_fresh(rows)
    if analysis.get("global_output_sha256") != EXPECTED_STDOUT_SHA256:
        raise RuntimeError("fresh generated stdout digest drifted")
    return analysis


def cold_environment(base_env: dict[str, str], arm: str) -> dict[str, str]:
    env = base_env.copy()
    if arm == "M":
        env["QWEN_GGUF_PARALLEL_COPY"] = "1"
    return env


def parse_process_times(stderr: str) -> dict[str, float]:
    matches = re.findall(
        r"^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys$",
        stderr,
        re.MULTILINE,
    )
    if len(matches) != 1:
        raise RuntimeError("cold process time row drifted")
    real_s, user_s, system_s = (float(value) for value in matches[0])
    if (
        not all(
            math.isfinite(value) and value >= 0 for value in (real_s, user_s, system_s)
        )
        or real_s <= 0
        or user_s + system_s <= 0
    ):
        raise RuntimeError("cold process times are invalid")
    return {
        "real_s": real_s,
        "user_s": user_s,
        "system_s": system_s,
        "total_cpu_s": user_s + system_s,
    }


def parse_cold_stdout(stdout: str, arm: str) -> dict[str, object]:
    invalidation = re.search(
        r"^invalidate: (\d+)/(\d+) -> (\d+)/(\d+)$",
        stdout,
        re.MULTILINE,
    )
    load = re.search(
        r"^load:\s+([0-9.]+) s\s+pageins=\s*(\d+)\s+diskR=\s*([0-9.]+) GiB$",
        stdout,
        re.MULTILINE,
    )
    first = re.search(r"^FIRST BYTE:\s+([0-9.]+) s$", stdout, re.MULTILINE)
    total = re.search(
        r"^rusage total: pageins=\s*(\d+)\s+diskR=\s*([0-9.]+) GiB",
        stdout,
        re.MULTILINE,
    )
    token = re.search(r"^first token: id=(\d+) piece=(.*)$", stdout, re.MULTILINE)
    post = re.search(
        r"^post-arm residency: (\d+)/(\d+) pages \(([0-9.]+)%\)$",
        stdout,
        re.MULTILINE,
    )
    if any(value is None for value in (invalidation, load, first, total, token, post)):
        raise RuntimeError("cold stdout shape drifted")
    assert invalidation is not None
    assert load is not None
    assert first is not None
    assert total is not None
    assert token is not None
    assert post is not None
    before, total_before, after, total_after = map(int, invalidation.groups())
    post_resident, post_total, _post_percent = post.groups()
    post_resident = int(post_resident)
    post_total = int(post_total)
    post_fraction = post_resident / post_total
    if (
        total_before != MODEL_PAGES
        or total_after != MODEL_PAGES
        or after != 0
        or post_total != MODEL_PAGES
        or post_fraction < 0.99
    ):
        raise RuntimeError("cold residency contract drifted")
    prefetch_matches = re.findall(
        r"^\s+prefetch:\s+([0-9.]+) s\s+(\d+) shards prefetched, "
        r"(\d+) skipped, ([0-9.]+) GiB returned$",
        stdout,
        re.MULTILINE,
    )
    if len(prefetch_matches) != 1:
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
    disk_gib = float(total.group(2))
    if disk_gib < MIN_COLD_DISK_GIB:
        raise RuntimeError("cold child physical read floor missed")
    return {
        "invalidated_before_pages": before,
        "invalidated_after_pages": after,
        "load_s": float(load.group(1)),
        "load_disk_gib": float(load.group(3)),
        "first_byte_s": float(first.group(1)),
        "total_pageins": int(total.group(1)),
        "total_disk_gib": disk_gib,
        "first_token": f"{token.group(1)} {token.group(2)}",
        "post_resident_pages": post_resident,
        "post_resident_fraction": post_fraction,
        "prefetch": prefetch,
    }


def run_cold_child(
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
    env = cold_environment(base_env, arm)
    policy = "cold-only"
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
    wait_errors = []
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
    except (OSError, KeyboardInterrupt) as error:
        error_text = f"{type(error).__name__}:{error}"
        if process is None:
            protocol.record_completion("cold", stem, None, error_text)
            if isinstance(error, OSError):
                protocol.record_spawn_failure_evidence(
                    "cold", stem, conditioning, error
                )
            raise protocol.InconclusivePacket(
                "cold",
                stem,
                [f"child_spawn_or_pipe_failed={error_text}"],
            ) from error
        returncode, deferred = protocol.wait_for_child(process)
        process_wall_ms = (time.perf_counter() - started) * 1e3
        wait_errors.extend(deferred)
        wait_errors.append(error_text)
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
    resources = protocol.process_resources(stderr)
    times = parse_process_times(stderr)
    if returncode == 0:
        parsed = parse_cold_stdout(stdout, arm)
        load_contract = parse_load_contract(stderr, "A" if arm == "M" else "B")
    else:
        parsed = None
        load_contract = parse_observed_load_contract(stderr, "A" if arm == "M" else "B")
    validity = {
        **post_exit,
        "process_resources": resources,
    }
    protocol.record_post_exit_evidence(
        "cold",
        stem,
        returncode,
        validity,
        reasons,
    )
    return {
        "stage": "cold",
        "artifact_stem": stem,
        "arm": arm,
        "pair_index": pair_index,
        "pair_order": order,
        "position": position,
        "command": command,
        "arm_environment": {
            "QWEN_GGUF_PARALLEL_COPY": "1" if arm == "M" else None,
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


def run_cold_stage(
    base_env: dict[str, str],
    manifest: dict[str, object],
    attempts_path: Path,
) -> list[dict[str, object]]:
    rows = []
    for pair_index, (order, first, second) in enumerate(COLD_PAIRS, 1):
        for position, arm in enumerate((first, second), 1):
            row = run_cold_child(
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
    return rows


def analyze_cold(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 4:
        raise RuntimeError("cold child count drifted")
    tokens = {row["cold"]["first_token"] for row in rows}
    if len(tokens) != 1:
        raise RuntimeError("cold first-token identity differs")
    pair_results = []
    for pair_index, (order, first, second) in enumerate(COLD_PAIRS, 1):
        pair = [row for row in rows if row["pair_index"] == pair_index]
        if len(pair) != 2 or [row["arm"] for row in pair] != [first, second]:
            raise RuntimeError("cold pair membership drifted")
        by_arm = {row["arm"]: row for row in pair}
        if set(by_arm) != {"M", "P"}:
            raise RuntimeError("cold admission arm membership drifted")
        baseline = by_arm["M"]
        candidate = by_arm["P"]
        load_delta_ms = (candidate["cold"]["load_s"] - baseline["cold"]["load_s"]) * 1e3
        first_delta_ms = (
            candidate["cold"]["first_byte_s"] - baseline["cold"]["first_byte_s"]
        ) * 1e3
        disk_ratio = (
            candidate["cold"]["total_disk_gib"] / baseline["cold"]["total_disk_gib"]
        )
        cpu_ratio = candidate["total_cpu_s"] / baseline["total_cpu_s"]
        footprint_ratio = (
            candidate["process_resources"]["peak_memory_footprint"]
            / baseline["process_resources"]["peak_memory_footprint"]
        )
        passes = (
            load_delta_ms <= 112.0
            and first_delta_ms <= 112.0
            and disk_ratio <= 1.10
            and cpu_ratio <= 1.10
            and footprint_ratio <= 1.05
        )
        pair_results.append(
            {
                "pair_index": pair_index,
                "pair_order": order,
                "load_delta_ms": load_delta_ms,
                "first_byte_delta_ms": first_delta_ms,
                "physical_read_ratio": disk_ratio,
                "total_cpu_ratio": cpu_ratio,
                "footprint_ratio": footprint_ratio,
                "passes": passes,
            }
        )
    return {
        "stage": "cold",
        "global_first_token": next(iter(tokens)),
        "pairs": pair_results,
        "admission_passes": all(row["passes"] for row in pair_results),
        "admission_load_delta_median_ms": statistics.median(
            row["load_delta_ms"] for row in pair_results
        ),
        "admission_first_byte_delta_median_ms": statistics.median(
            row["first_byte_delta_ms"] for row in pair_results
        ),
    }


def configure_protocol() -> None:
    protocol.ARTIFACT = ARTIFACT
    protocol.PREREG = PREREG
    protocol.required_manifest_paths = required_manifest_paths
    protocol.source_and_build_identity = source_and_build_identity
    protocol.build_manifest = build_manifest
    protocol.arm_environment = arm_environment
    protocol.parse_load_contract = parse_load_contract
    protocol.parse_observed_load_contract = parse_observed_load_contract
    protocol.run_fresh_child = run_fresh_child
    protocol.analyze_fresh = analyze_fresh


def write_decision(decision: dict[str, object], manifest: dict[str, object]) -> None:
    if decision.get("status") == "go":
        predecessor = verify_predecessor()
        fresh = decision.get("stages", {}).get("fresh_128")
        cold = decision.get("stages", {}).get("cold")
        if (
            decision.get("authority") != "auto-exact-a3b-pread-disposable"
            or decision.get("stopped_after") != "cold"
            or decision.get("source_commit") != manifest.get("source_commit")
            or not isinstance(fresh, dict)
            or fresh.get("passes") is not True
            or fresh.get("global_output_sha256") != EXPECTED_STDOUT_SHA256
            or not isinstance(cold, dict)
            or cold.get("admission_passes") is not True
            or [row.get("pair_order") for row in cold.get("pairs", [])] != ["MP", "PM"]
            or decision.get("imported_predecessor") != predecessor
            or manifest.get("imported_predecessor") != predecessor
        ):
            raise RuntimeError("v0.621 GO authority conjunction is incomplete")
    elif decision.get("authority") != "none":
        raise RuntimeError("non-GO v0.621 decision carries authority")
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
    stopped_after = "fresh-128"
    try:
        fresh_rows = protocol.run_stage("fresh-128", base_env, manifest, attempts_path)
        stages["fresh_128"] = analyze_fresh(fresh_rows)
        if not stages["fresh_128"]["passes"]:
            write_decision(
                {
                    "schema": 1,
                    "status": "kill",
                    "authority": "none",
                    "stopped_after": stopped_after,
                    "source_commit": manifest["source_commit"],
                    "imported_predecessor": manifest["imported_predecessor"],
                    "stages": stages,
                },
                manifest,
            )
            return
        stopped_after = "cold"
        cold_rows = run_cold_stage(base_env, manifest, attempts_path)
        stages["cold"] = analyze_cold(cold_rows)
        status = "go" if stages["cold"]["admission_passes"] else "kill"
        authority = "auto-exact-a3b-pread-disposable" if status == "go" else "none"
        write_decision(
            {
                "schema": 1,
                "status": status,
                "authority": authority,
                "stopped_after": stopped_after,
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
                "stopped_after": stopped_after,
                "reasons": ["operator_interrupt_after_child_cleanup"],
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
    except Exception as error:
        if protocol.decision_publication_started():
            raise
        write_decision(
            {
                "schema": 1,
                "status": "implementation_or_contract_defect",
                "authority": "none",
                "stopped_after": stopped_after,
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
