#!/usr/bin/env python3

import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import v0595_a3b_generic_retained_cold as cold
import v0596_a3b_retained_route_warm as warm


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0598-a3b-owned-arena-pilot-p2"
P1_ROOT = ROOT / "target/profiles/v0598-a3b-owned-arena-pilot-p1"
P1_MANIFEST = P1_ROOT / "manifest.json"
P1_CORRECTNESS = P1_ROOT / "correctness.out"
PREREG = ROOT / "docs/bench/v0598-a3b-owned-arena-pilot.md"
MODEL = Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
CLI_BINARY = ROOT / "target/release/qwen"
BENCH_BINARY = ROOT / "target/release/qwen-bench"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
COLD_RUNNER = ROOT / "scripts/profile/v0595_a3b_generic_retained_cold.py"
WARM_RUNNER = ROOT / "scripts/profile/v0596_a3b_retained_route_warm.py"
EXPECTED_MODEL_SHA256 = (
    "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
)
EXPECTED_PROMPT_SHA256 = (
    "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
)
EXPECTED_P1_COMMIT = "701de5087f21da6376b444eb3f4f6369f0e8e04d"
EXPECTED_P1_MANIFEST_SHA256 = (
    "050e11395b7f6fdc14c233d2653810aafb546c11469b6a794e859033b3a2f4c1"
)
EXPECTED_P1_CORRECTNESS_SHA256 = (
    "68ac2a83f2e2d3e6f0261475445035f43c73d8641fc4f7a9d03e3a47b8712cb0"
)
BLOCK_ORDERS = ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA")
LENGTHS = (1, 128)
RUNS = 5
TOKENS = 127
COOLDOWN_S = 30.0
MAX_BLOCK_ATTEMPTS = 3


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
        COLD_RUNNER,
        WARM_RUNNER,
        MODEL,
        PROMPT,
        CLI_BINARY,
        BENCH_BINARY,
        P1_MANIFEST,
        P1_CORRECTNESS,
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    for path in required_manifest_paths()[:5]:
        command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    commit = command_text(["git", "rev-parse", "HEAD"]).strip()
    dirty = command_text(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"]
    ).strip()
    if dirty:
        raise RuntimeError(f"tracked source is dirty: {dirty!r}")
    build = json.loads(
        command_text([str(BENCH_BINARY), "build-info", "--output", "json"])
    )
    if (
        build.get("build_commit") != commit
        or build.get("runtime_commit") != commit
        or build.get("status") != "match"
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
    ):
        raise RuntimeError(f"source/build identity mismatch: {build}")
    return commit, build


def build_manifest(removed_environment: list[str]) -> dict[str, object]:
    commit, build = source_and_build_identity()
    hashes = {
        str(path): cold.common.sha256_file(path) for path in required_manifest_paths()
    }
    if hashes[str(MODEL)] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA-256 drifted")
    if hashes[str(PROMPT)] != EXPECTED_PROMPT_SHA256:
        raise RuntimeError("prompt SHA-256 drifted")
    if hashes[str(P1_MANIFEST)] != EXPECTED_P1_MANIFEST_SHA256:
        raise RuntimeError("p1 manifest SHA-256 drifted")
    if hashes[str(P1_CORRECTNESS)] != EXPECTED_P1_CORRECTNESS_SHA256:
        raise RuntimeError("p1 correctness SHA-256 drifted")
    p1_manifest = json.loads(P1_MANIFEST.read_text(encoding="utf-8"))
    if p1_manifest.get("source_commit") != EXPECTED_P1_COMMIT:
        raise RuntimeError("p1 source identity drifted")
    p1_entries = sorted(path.name for path in P1_ROOT.iterdir() if path.is_file())
    if p1_entries != ["correctness.out", "manifest.json"]:
        raise RuntimeError(f"p1 stop boundary drifted: {p1_entries}")
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "build_identity": build,
        "removed_environment": removed_environment,
        "sha256": hashes,
        "model_size_bytes": MODEL.stat().st_size,
        "prompt_bytes": PROMPT.stat().st_size,
        "block_orders": list(BLOCK_ORDERS),
        "lengths": list(LENGTHS),
        "loaded_runs": RUNS,
        "loaded_decode_calls": TOKENS,
        "cooldown_s": COOLDOWN_S,
        "max_block_attempts": MAX_BLOCK_ATTEMPTS,
        "p1_source_commit": EXPECTED_P1_COMMIT,
        "p1_manifest_sha256": EXPECTED_P1_MANIFEST_SHA256,
        "p1_correctness_sha256": EXPECTED_P1_CORRECTNESS_SHA256,
    }


def arm_environment(arm: str) -> dict[str, str | None]:
    if arm == "A":
        return {
            "QWEN_GGUF_OWNED_ARENA": "0",
            "QWEN_GGUF_NO_COPY": "0",
            "QWEN_GGUF_NO_COPY_PREFAULT": None,
        }
    if arm == "B":
        return {
            "QWEN_GGUF_OWNED_ARENA": "1",
            "QWEN_GGUF_NO_COPY": "0",
            "QWEN_GGUF_NO_COPY_PREFAULT": None,
        }
    if arm == "C":
        return {
            "QWEN_GGUF_OWNED_ARENA": "0",
            "QWEN_GGUF_NO_COPY": "1",
            "QWEN_GGUF_NO_COPY_PREFAULT": "0",
        }
    raise ValueError(f"unknown arm {arm!r}")


def environment_for_arm(base_env: dict[str, str], arm: str) -> dict[str, str]:
    env = base_env.copy()
    for key, value in arm_environment(arm).items():
        if value is not None:
            env[key] = value
    return env


def parse_load_contract(stderr: str, arm: str) -> dict[str, object]:
    policy = (
        "[metal-load] native quantized token embedding policy: "
        "auto-promoted (Q8_0 [2048, 248320])"
    )
    policy_lines = re.findall(
        r"^\[metal-load\] native quantized token embedding policy:.*$",
        stderr,
        re.MULTILINE,
    )
    ledger_lines = re.findall(r"^\[metal-load-ledger\].*$", stderr, re.MULTILINE)
    owned_lines = re.findall(r"^\[metal-gguf-owned\].*$", stderr, re.MULTILINE)
    retained_lines = re.findall(r"^\[metal-gguf-retained\].*$", stderr, re.MULTILINE)
    legacy_lines = re.findall(r"^\[metal-gguf-no-copy\].*$", stderr, re.MULTILINE)
    if policy_lines != [policy] or legacy_lines:
        raise RuntimeError("native embedding or legacy load contract drifted")

    copied_ledger = (
        "[metal-load-ledger] source=733/22123538944 "
        "direct_copy=733/22123538944 direct_view=0/0 direct_alias=0/0 "
        "tail_fallback=0/0 converted=0/0/0 derived=0/0"
    )
    owned_ledger = (
        "[metal-load-ledger] source=733/22123538944 "
        "direct_copy=732/22123530752 direct_view=0/0 direct_alias=0/0 "
        "tail_fallback=1/8192 converted=0/0/0 derived=0/0"
    )
    retained_ledger = (
        "[metal-load-ledger] source=733/22123538944 direct_copy=0/0 "
        "direct_view=732/22123530752 direct_alias=0/0 "
        "tail_fallback=1/8192 converted=0/0/0 derived=0/0"
    )
    retained_line = (
        "[metal-gguf-retained] windows=1 window_bytes=22123544576 "
        "direct=733 view=732/22123530752 alias=0/0 fallback=1/8192 "
        "page=16384 max_buffer=77309411328 alignment=32 "
        "prefault=disabled prefault_pages=0 prefault_bytes=0 "
        "prefault_ms=0.000 checksum=0x0000000000000000"
    )
    if arm == "A":
        if ledger_lines != [copied_ledger] or owned_lines or retained_lines:
            raise RuntimeError("copied load contract drifted")
        return {"storage": "copied"}
    if arm == "C":
        if (
            ledger_lines != [retained_ledger]
            or retained_lines != [retained_line]
            or owned_lines
        ):
            raise RuntimeError("retained load contract drifted")
        return {"storage": "retained", "prefault": False}

    if ledger_lines != [owned_ledger] or retained_lines or len(owned_lines) != 1:
        raise RuntimeError("owned load contract drifted")
    match = re.fullmatch(
        r"\[metal-gguf-owned\] windows=1 window_bytes=22123544576 "
        r"gaps=13824 fallback=1/8192 resources=2/22123552768 workers=4 "
        r"page=16384 alignment=32 allocation_ms=([0-9.]+) "
        r"copy_ms=([0-9.]+) ready_ms=([0-9.]+)",
        owned_lines[0],
    )
    if match is None:
        raise RuntimeError("owned physical ledger drifted")
    allocation_ms, copy_ms, ready_ms = (float(value) for value in match.groups())
    if min(allocation_ms, copy_ms, ready_ms) < 0 or ready_ms < copy_ms:
        raise RuntimeError("invalid owned materialization timing")
    return {
        "storage": "owned-four",
        "allocation_ms": allocation_ms,
        "copy_ms": copy_ms,
        "ready_ms": ready_ms,
    }


def run_correctness(base_env: dict[str, str]) -> dict[str, object]:
    stdout_path = ARTIFACT / "correctness.out"
    command = [
        "cargo",
        "test",
        "--release",
        "-p",
        "qwen-llm",
        "gguf_owned_arena_a3b_q4_is_bit_exact",
        "--",
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ]
    started = time.perf_counter()
    with stdout_path.open("xb") as output:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=base_env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_ms = (time.perf_counter() - started) * 1e3
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    if result.returncode != 0 or "test result: ok. 1 passed;" not in text:
        raise RuntimeError(f"owned full-state correctness failed; see {stdout_path}")
    owned_lines = re.findall(r"\[metal-gguf-owned\].*$", text, re.MULTILINE)
    if len(owned_lines) != 2:
        raise RuntimeError("correctness expected two owned materializations")
    copied_ledger = (
        "[metal-load-ledger] source=733/22123538944 "
        "direct_copy=733/22123538944 direct_view=0/0 direct_alias=0/0 "
        "tail_fallback=0/0 converted=0/0/0 derived=0/0"
    )
    owned_ledger = (
        "[metal-load-ledger] source=733/22123538944 "
        "direct_copy=732/22123530752 direct_view=0/0 direct_alias=0/0 "
        "tail_fallback=1/8192 converted=0/0/0 derived=0/0"
    )
    ledgers = re.findall(r"^\[metal-load-ledger\].*$", text, re.MULTILINE)
    if ledgers != [copied_ledger, owned_ledger]:
        raise RuntimeError("correctness loader ledgers drifted")
    return {
        "command": command,
        "wall_ms": wall_ms,
        "stdout_path": str(stdout_path),
        "stdout_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "owned_materializations": owned_lines,
        "ledgers": ledgers,
        "passed": True,
    }


def append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")


def condition_cache(stem: str) -> dict[str, object]:
    host_before_cache = cold.common.wait_for_valid_host(f"{stem} cache read")
    vm_before_cache = cold.capture_vm_state()
    cache_ms, cache_bytes = cold.warm_file()
    if cache_bytes != MODEL.stat().st_size:
        raise RuntimeError("cache precondition did not read the complete model")
    time.sleep(COOLDOWN_S)
    host_before_spawn = cold.common.wait_for_valid_host(f"{stem} process spawn")
    vm_before_spawn = cold.capture_vm_state()
    cache_pageout_delta = vm_before_spawn["pageouts"] - vm_before_cache["pageouts"]
    cache_swap_delta = (
        vm_before_spawn["swap_used_bytes"] - vm_before_cache["swap_used_bytes"]
    )
    if cache_pageout_delta != 0 or cache_swap_delta > 0:
        raise RuntimeError(
            f"{stem} cache conditioning changed VM pressure: "
            f"pageouts={cache_pageout_delta} swap={cache_swap_delta}"
        )
    return {
        "cache_ms": cache_ms,
        "host_before_cache": host_before_cache,
        "host_before_spawn": host_before_spawn,
        "vm_before_cache": vm_before_cache,
        "vm_before_spawn": vm_before_spawn,
        "cache_pageout_delta": cache_pageout_delta,
        "cache_swap_delta_bytes": cache_swap_delta,
    }


def process_resources(
    stderr: str,
    vm_before: dict[str, object],
    host_after: dict[str, object],
    vm_after: dict[str, object],
) -> dict[str, object]:
    return {
        "host_after_exit": host_after,
        "vm_after_exit": vm_after,
        "maximum_resident_set_size": cold.common.parse_resource(
            stderr, "maximum resident set size"
        ),
        "peak_memory_footprint": cold.common.parse_resource(
            stderr, "peak memory footprint"
        ),
        "page_reclaims": cold.common.parse_resource(stderr, "page reclaims"),
        "page_faults": cold.common.parse_resource(stderr, "page faults"),
        "block_input_operations": cold.common.parse_resource(
            stderr, "block input operations"
        ),
        "process_pageout_delta": vm_after["pageouts"] - vm_before["pageouts"],
        "process_swap_delta_bytes": (
            vm_after["swap_used_bytes"] - vm_before["swap_used_bytes"]
        ),
    }


def validity_reasons(resources: dict[str, object], gate_faults: bool) -> list[str]:
    reasons = []
    if gate_faults and resources["page_faults"] != 0:
        reasons.append(f"major_page_faults={resources['page_faults']}")
    if resources["block_input_operations"] != 0:
        reasons.append(f"block_input_operations={resources['block_input_operations']}")
    if resources["process_pageout_delta"] != 0:
        reasons.append(f"process_pageout_delta={resources['process_pageout_delta']}")
    if resources["process_swap_delta_bytes"] > 0:
        reasons.append(
            f"process_swap_growth_bytes={resources['process_swap_delta_bytes']}"
        )
    if not resources["host_after_exit"]["valid"]:
        reasons.append("post_exit_host_invalid")
    return reasons


def run_fresh_one(
    length: int,
    arm: str,
    block_index: int,
    order: str,
    position: int,
    attempt: int,
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
    stem = (
        f"fresh-n{length:03d}-b{block_index:02d}-{order.lower()}-"
        f"r{position}-{arm.lower()}-a{attempt:02d}"
    )
    timing_path = ARTIFACT / f"{stem}.timing.jsonl"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    for path in (timing_path, stdout_path, stderr_path):
        if path.exists():
            raise RuntimeError(f"refusing to reuse artifact {path}")
    conditioning = condition_cache(stem)
    env = environment_for_arm(base_env, arm)
    command = [
        "/usr/bin/time",
        "-l",
        str(CLI_BINARY),
        "--model",
        str(MODEL),
        "--prompt-file",
        str(PROMPT),
        "--tokens",
        str(length),
        "--prefill-chunk",
        "1024",
        "--max-context-tokens",
        "1024",
        "--prefix-cache-max-mib",
        "0",
        "--request-timings",
        str(timing_path),
    ]
    started = time.perf_counter()
    with stderr_path.open("xb") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
        assert process.stdout is not None
        first = process.stdout.read(1)
        first_byte_ms = (time.perf_counter() - started) * 1e3 if first else None
        rest = process.stdout.read()
        returncode = process.wait()
    exit_ms = (time.perf_counter() - started) * 1e3
    host_after_exit = cold.common.capture_host_state()
    vm_after_exit = cold.capture_vm_state()
    stdout = first + rest
    stdout_path.write_bytes(stdout)
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    if returncode != 0 or not first or first_byte_ms is None:
        raise RuntimeError(f"{stem} failed; see {stderr_path}")
    timings = [
        json.loads(line) for line in timing_path.read_text().splitlines() if line
    ]
    if len(timings) != 1:
        raise RuntimeError(f"{stem} expected one timing row")
    timing = timings[0]
    cold.common.validate_timing(timing, length, manifest)
    load_contract = parse_load_contract(stderr, arm)
    resources = process_resources(
        stderr,
        conditioning["vm_before_spawn"],
        host_after_exit,
        vm_after_exit,
    )
    reasons = validity_reasons(resources, gate_faults=True)
    return {
        "packet": f"fresh-{length}",
        "artifact_stem": stem,
        "length": length,
        "arm": arm,
        "block_index": block_index,
        "block_order": order,
        "position": position,
        "attempt": attempt,
        "command": command,
        "arm_environment": arm_environment(arm),
        **conditioning,
        **resources,
        "spawn_to_first_byte_ms": first_byte_ms,
        "spawn_to_exit_ms": exit_ms,
        "outer_residual_ms": (
            first_byte_ms - timing["runtime_and_model_load_ms"] - timing["ttft_ms"]
        ),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "load_contract": load_contract,
        "timing": timing,
        "valid": not reasons,
        "validity_reasons": reasons,
    }


def parse_loaded_bench(stderr: str) -> dict[str, object]:
    parsed = warm.parse_bench(stderr)
    request_matches = re.findall(
        r"^\[bench\] rep\s+(\d+) request\s+([0-9.]+) ms$",
        stderr,
        re.MULTILINE,
    )
    if len(request_matches) != RUNS or [int(row[0]) for row in request_matches] != list(
        range(1, RUNS + 1)
    ):
        raise RuntimeError("loaded request-wall rows drifted")
    request_walls = [float(row[1]) for row in request_matches]
    if any(not math.isfinite(value) or value <= 0 for value in request_walls):
        raise RuntimeError("invalid loaded request wall")
    for repetition, request_wall in zip(
        parsed["repetitions"], request_walls, strict=True
    ):
        repetition["request_wall_ms"] = request_wall
        if request_wall + 0.2 < repetition["prefill_ms"] + repetition["decode_ms"]:
            raise RuntimeError("loaded request wall excludes measured phase work")
    average_matches = re.findall(
        r"^\[bench\] request wall: ([0-9.]+) ms avg$", stderr, re.MULTILINE
    )
    if len(average_matches) != 1:
        raise RuntimeError("loaded request-wall average drifted")
    parsed["request_wall_average_reported_ms"] = float(average_matches[0])
    if abs(statistics.mean(request_walls) - float(average_matches[0])) > 0.11:
        raise RuntimeError("loaded request-wall average is inconsistent with rows")
    return parsed


def run_loaded_one(
    arm: str,
    block_index: int,
    order: str,
    position: int,
    attempt: int,
    prompt_text: str,
    base_env: dict[str, str],
) -> dict[str, object]:
    stem = (
        f"loaded-b{block_index:02d}-{order.lower()}-"
        f"r{position}-{arm.lower()}-a{attempt:02d}"
    )
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    for path in (stdout_path, stderr_path):
        if path.exists():
            raise RuntimeError(f"refusing to reuse artifact {path}")
    conditioning = condition_cache(stem)
    env = environment_for_arm(base_env, arm)
    command = [
        "/usr/bin/time",
        "-l",
        str(BENCH_BINARY),
        "decode",
        "--model",
        str(MODEL),
        "--prompt",
        prompt_text,
        "--tokens",
        str(TOKENS),
        "--runs",
        str(RUNS),
        "--prefill-chunk",
        "1024",
        "--kv-capacity",
        "1024",
        "--full-logits-decode",
    ]
    started = time.perf_counter()
    with stdout_path.open("xb") as stdout_file, stderr_path.open("xb") as stderr_file:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    process_wall_ms = (time.perf_counter() - started) * 1e3
    host_after_exit = cold.common.capture_host_state()
    vm_after_exit = cold.capture_vm_state()
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    if result.returncode != 0 or stdout_path.stat().st_size != 0:
        raise RuntimeError(f"{stem} failed; see {stderr_path}")
    load_contract = parse_load_contract(stderr, arm)
    bench = parse_loaded_bench(stderr)
    resources = process_resources(
        stderr,
        conditioning["vm_before_spawn"],
        host_after_exit,
        vm_after_exit,
    )
    reasons = validity_reasons(resources, gate_faults=False)
    return {
        "packet": "loaded",
        "artifact_stem": stem,
        "arm": arm,
        "block_index": block_index,
        "block_order": order,
        "position": position,
        "attempt": attempt,
        "command": command,
        "arm_environment": arm_environment(arm),
        **conditioning,
        **resources,
        "process_wall_ms": process_wall_ms,
        "load_contract": load_contract,
        "bench": bench,
        "valid": not reasons,
        "validity_reasons": reasons,
    }


def run_valid_block(
    packet: str,
    block_index: int,
    order: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
    prompt_text: str,
    attempts_path: Path,
    blocks_path: Path,
) -> list[dict[str, object]]:
    for attempt in range(1, MAX_BLOCK_ATTEMPTS + 1):
        rows = []
        for position, arm in enumerate(order, 1):
            if packet.startswith("fresh-"):
                length = int(packet.split("-", 1)[1])
                row = run_fresh_one(
                    length,
                    arm,
                    block_index,
                    order,
                    position,
                    attempt,
                    base_env,
                    manifest,
                )
            else:
                row = run_loaded_one(
                    arm,
                    block_index,
                    order,
                    position,
                    attempt,
                    prompt_text,
                    base_env,
                )
            append_row(attempts_path, row)
            rows.append(row)
        accepted = all(row["valid"] for row in rows)
        block = {
            "packet": packet,
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
        }
        append_row(blocks_path, block)
        if accepted:
            return rows
    raise RuntimeError(f"failed to obtain valid {packet} block {block_index}")


def packet_blocks(
    rows: list[dict[str, object]], packet: str
) -> list[dict[str, object]]:
    selected = []
    for block_index, order in enumerate(BLOCK_ORDERS, 1):
        matching = [
            row
            for row in rows
            if row["packet"] == packet
            and row["block_index"] == block_index
            and row["block_order"] == order
            and row["valid"]
        ]
        attempts = sorted({row["attempt"] for row in matching})
        if not attempts:
            raise RuntimeError(f"accepted {packet} block {block_index} is missing")
        chosen = [row for row in matching if row["attempt"] == attempts[-1]]
        if len(chosen) != 3:
            raise RuntimeError(f"accepted {packet} block {block_index} is incomplete")
        selected.append({row["arm"]: row for row in chosen})
    return selected


def analyze_fresh(rows: list[dict[str, object]], length: int) -> dict[str, object]:
    packet = f"fresh-{length}"
    blocks = packet_blocks(rows, packet)
    stdout_hashes = {row[arm]["stdout_sha256"] for row in blocks for arm in "ABC"}
    if len(stdout_hashes) != 1:
        raise RuntimeError(f"{packet} output mismatch")
    speedups = []
    before = []
    after = []
    wins = 0
    rss_ratios = []
    footprint_ratios = []
    for order, block in zip(BLOCK_ORDERS, blocks, strict=True):
        a = block["A"]
        b = block["B"]
        speedup = a["spawn_to_first_byte_ms"] / b["spawn_to_first_byte_ms"]
        speedups.append(speedup)
        (before if order.index("B") < order.index("A") else after).append(speedup)
        wins += b["spawn_to_first_byte_ms"] < a["spawn_to_first_byte_ms"]
        rss_ratios.append(
            b["maximum_resident_set_size"] / a["maximum_resident_set_size"]
        )
        footprint_ratios.append(b["peak_memory_footprint"] / a["peak_memory_footprint"])
    gates = {
        "median_speedup_at_least_1_20x": statistics.median(speedups) >= 1.20,
        "b_before_a_speedup_at_least_1_20x": statistics.median(before) >= 1.20,
        "a_before_b_speedup_at_least_1_20x": statistics.median(after) >= 1.20,
        "wins_at_least_5_of_6": wins >= 5,
        "paired_rss_ratio_at_most_1_05": max(rss_ratios) <= 1.05,
        "paired_footprint_ratio_at_most_1_05": max(footprint_ratios) <= 1.05,
    }
    return {
        "packet": packet,
        "speedups": speedups,
        "median_speedup": statistics.median(speedups),
        "b_before_a_speedups": before,
        "a_before_b_speedups": after,
        "wins": wins,
        "paired_rss_ratios": rss_ratios,
        "paired_footprint_ratios": footprint_ratios,
        "gates": gates,
        "passes": all(gates.values()),
    }


def late_metrics(row: dict[str, object]) -> dict[str, object]:
    repetitions = row["bench"]["repetitions"]
    late = repetitions[2:5]
    prefill_ms = statistics.median(rep["prefill_ms"] for rep in late)
    decode_ms = statistics.median(rep["decode_ms"] for rep in late)
    request_ms = statistics.median(rep["request_wall_ms"] for rep in late)
    decode_tps = [TOKENS * 1000.0 / rep["decode_ms"] for rep in repetitions]
    late_tps = decode_tps[2:5]
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "request_ms": request_ms,
        "decode_tps": decode_tps,
        "rep5_over_rep3_tps": decode_tps[4] / decode_tps[2],
        "late_tps_relative_range": (
            (max(late_tps) - min(late_tps)) / statistics.median(late_tps)
        ),
    }


def analyze_loaded(rows: list[dict[str, object]]) -> dict[str, object]:
    blocks = packet_blocks(rows, "loaded")
    output_hashes = {
        block[arm]["bench"]["generated_sha256"] for block in blocks for arm in "ABC"
    }
    if len(output_hashes) != 1:
        raise RuntimeError("loaded generated output mismatch")
    metrics = []
    stability_results = []
    performance_results = []
    for block in blocks:
        a = late_metrics(block["A"])
        b = late_metrics(block["B"])
        row = {
            "prefill_a_over_b": a["prefill_ms"] / b["prefill_ms"],
            "decode_a_over_b": a["decode_ms"] / b["decode_ms"],
            "request_b_over_a": b["request_ms"] / a["request_ms"],
            "a_rep5_over_rep3_tps": a["rep5_over_rep3_tps"],
            "b_rep5_over_rep3_tps": b["rep5_over_rep3_tps"],
            "a_late_tps_relative_range": a["late_tps_relative_range"],
            "b_late_tps_relative_range": b["late_tps_relative_range"],
            "rss_b_over_a": (
                block["B"]["maximum_resident_set_size"]
                / block["A"]["maximum_resident_set_size"]
            ),
            "footprint_b_over_a": (
                block["B"]["peak_memory_footprint"]
                / block["A"]["peak_memory_footprint"]
            ),
        }
        stability_gates = {
            "a_rep5_over_rep3_stable": 0.98 <= row["a_rep5_over_rep3_tps"] <= 1.02,
            "b_rep5_over_rep3_stable": 0.98 <= row["b_rep5_over_rep3_tps"] <= 1.02,
            "a_late_range_stable": row["a_late_tps_relative_range"] <= 0.03,
            "b_late_range_stable": row["b_late_tps_relative_range"] <= 0.03,
        }
        performance_gates = {
            "prefill_parity": row["prefill_a_over_b"] >= 0.99,
            "decode_parity": row["decode_a_over_b"] >= 0.99,
            "request_nonregression": row["request_b_over_a"] <= 1.00,
            "rss_ratio": row["rss_b_over_a"] <= 1.05,
            "footprint_ratio": row["footprint_b_over_a"] <= 1.05,
        }
        row["stability_gates"] = stability_gates
        row["performance_gates"] = performance_gates
        row["stable"] = all(stability_gates.values())
        row["performance_passes"] = all(performance_gates.values())
        metrics.append(row)
        stability_results.append(row["stable"])
        performance_results.append(row["performance_passes"])
    stable = all(stability_results)
    performance_passes = all(performance_results)
    return {
        "packet": "loaded",
        "blocks": metrics,
        "stable": stable,
        "performance_passes": performance_passes,
        "passes": stable and performance_passes,
    }


def run_packet(
    packet: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
    prompt_text: str,
    attempts_path: Path,
    blocks_path: Path,
) -> list[dict[str, object]]:
    rows = []
    for block_index, order in enumerate(BLOCK_ORDERS, 1):
        rows.extend(
            run_valid_block(
                packet,
                block_index,
                order,
                base_env,
                manifest,
                prompt_text,
                attempts_path,
                blocks_path,
            )
        )
    return rows


def write_decision(decision: dict[str, object]) -> None:
    (ARTIFACT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse packet directory {ARTIFACT}")
    ARTIFACT.mkdir(parents=True)
    base_env, removed_environment = cold.common.normalized_environment()
    manifest = build_manifest(removed_environment)
    (ARTIFACT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    correctness = run_correctness(base_env)
    (ARTIFACT / "correctness.json").write_text(
        json.dumps(correctness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempts_path = ARTIFACT / "attempts.jsonl"
    blocks_path = ARTIFACT / "block-attempts.jsonl"
    prompt_text = PROMPT.read_text(encoding="utf-8")
    accepted_rows = []
    stage_results = {}

    rows = run_packet(
        "fresh-1",
        base_env,
        manifest,
        prompt_text,
        attempts_path,
        blocks_path,
    )
    accepted_rows.extend(rows)
    stage_results["fresh_1"] = analyze_fresh(accepted_rows, 1)
    if not stage_results["fresh_1"]["passes"]:
        write_decision(
            {
                "schema": 1,
                "status": "kill",
                "authority": "none",
                "stopped_after": "fresh-1",
                "source_commit": manifest["source_commit"],
                "correctness": correctness,
                "stages": stage_results,
            }
        )
        return

    rows = run_packet(
        "fresh-128",
        base_env,
        manifest,
        prompt_text,
        attempts_path,
        blocks_path,
    )
    accepted_rows.extend(rows)
    stage_results["fresh_128"] = analyze_fresh(accepted_rows, 128)
    if not stage_results["fresh_128"]["passes"]:
        write_decision(
            {
                "schema": 1,
                "status": "kill",
                "authority": "none",
                "stopped_after": "fresh-128",
                "source_commit": manifest["source_commit"],
                "correctness": correctness,
                "stages": stage_results,
            }
        )
        return

    rows = run_packet(
        "loaded",
        base_env,
        manifest,
        prompt_text,
        attempts_path,
        blocks_path,
    )
    accepted_rows.extend(rows)
    stage_results["loaded"] = analyze_loaded(accepted_rows)
    loaded = stage_results["loaded"]
    if not loaded["stable"]:
        status = "inconclusive"
        authority = "none"
    elif loaded["performance_passes"]:
        status = "go"
        authority = "force-only-exact-a3b"
    else:
        status = "kill"
        authority = "none"
    write_decision(
        {
            "schema": 1,
            "status": status,
            "authority": authority,
            "stopped_after": "loaded",
            "source_commit": manifest["source_commit"],
            "correctness": correctness,
            "stages": stage_results,
        }
    )


if __name__ == "__main__":
    main()
