#!/usr/bin/env python3

import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

import v0595_a3b_generic_retained_cold as prior


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0596-a3b-retained-route-warm-p1"
PREREG = ROOT / "docs/bench/v0596-a3b-retained-route-warm.md"
MODEL = Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
BENCH_BINARY = ROOT / "target/release/qwen-bench"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
PRIOR_RUNNER = ROOT / "scripts/profile/v0595_a3b_generic_retained_cold.py"
PRIOR_MANIFEST = (
    ROOT / "target/profiles/v0595-a3b-generic-retained-cold-p1/manifest.json"
)
PRIOR_DECISION = (
    ROOT / "target/profiles/v0595-a3b-generic-retained-cold-p1/decision.json"
)
EXPECTED_PRIOR_COMMIT = "cc91ab2b1a0e1f04e13c678506c0c47507c2d65a"
EXPECTED_PRIOR_MANIFEST_SHA256 = (
    "eeca0499013aead772a0ff5c3333704f903c4100b74707a04ae2d663700a0fdb"
)
EXPECTED_PRIOR_DECISION_SHA256 = (
    "05ec19c433af94d170f296359e8006b58d7291060039c6045ff9fba2d7db6a73"
)
EXPECTED_MODEL_SHA256 = (
    "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
)
EXPECTED_PROMPT_SHA256 = (
    "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
)
BLOCK_ORDERS = ("ABC", "BCA", "CAB")
RUNS = 5
TOKENS = 127
PROMPT_TOKENS = 419
COOLDOWN_S = 30.0
MAX_BLOCK_ATTEMPTS = 3
EXPECTED_PREFAULT_PAGES = 1_350_314
EXPECTED_PREFAULT_BYTES = 22_123_544_576


def command_text(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    tracked = (Path(__file__).resolve(), PREREG, COMMON_RUNNER, PRIOR_RUNNER)
    for path in tracked:
        relative = path.relative_to(ROOT)
        command_text(["git", "ls-files", "--error-unmatch", str(relative)])
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


def required_manifest_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PREREG,
        COMMON_RUNNER,
        PRIOR_RUNNER,
        MODEL,
        PROMPT,
        BENCH_BINARY,
        PRIOR_MANIFEST,
        PRIOR_DECISION,
    )


def build_manifest(removed_environment: list[str]) -> dict[str, object]:
    commit, build = source_and_build_identity()
    hashes = {
        str(path): prior.common.sha256_file(path) for path in required_manifest_paths()
    }
    if hashes[str(MODEL)] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA-256 drifted")
    if hashes[str(PROMPT)] != EXPECTED_PROMPT_SHA256:
        raise RuntimeError("prompt SHA-256 drifted")
    if hashes[str(PRIOR_MANIFEST)] != EXPECTED_PRIOR_MANIFEST_SHA256:
        raise RuntimeError("v0.595 manifest SHA-256 drifted")
    if hashes[str(PRIOR_DECISION)] != EXPECTED_PRIOR_DECISION_SHA256:
        raise RuntimeError("v0.595 decision SHA-256 drifted")
    prior_manifest = json.loads(PRIOR_MANIFEST.read_text(encoding="utf-8"))
    prior_decision = json.loads(PRIOR_DECISION.read_text(encoding="utf-8"))
    if prior_manifest.get("source_commit") != EXPECTED_PRIOR_COMMIT:
        raise RuntimeError("v0.595 source identity drifted")
    prior_transition = prior_decision["metrics_by_length"]["128"]["full"]["transition"][
        "median_paired_retained_over_baseline"
    ]
    if (
        prior_decision.get("status") != "needs_review"
        or prior_transition != 0.867726599824634
    ):
        raise RuntimeError("v0.595 decision identity drifted")
    for helper in (COMMON_RUNNER, PRIOR_RUNNER):
        if hashes[str(helper)] != prior_manifest["sha256"].get(str(helper)):
            raise RuntimeError(f"v0.595 imported helper drifted: {helper}")
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "build_identity": build,
        "removed_environment": removed_environment,
        "sha256": hashes,
        "model_size_bytes": MODEL.stat().st_size,
        "block_orders": list(BLOCK_ORDERS),
        "runs_per_process": RUNS,
        "decode_calls_per_run": TOKENS,
        "prompt_tokens": PROMPT_TOKENS,
        "kv_capacity": 1024,
        "cooldown_s": COOLDOWN_S,
        "prior_source_commit": EXPECTED_PRIOR_COMMIT,
        "prior_transition_ratio": prior_transition,
    }


def expected_arm_environment(arm: str) -> dict[str, str | None]:
    if arm not in "ABC":
        raise ValueError(f"unknown arm {arm!r}")
    return {
        "QWEN_GGUF_NO_COPY": "0" if arm == "A" else "1",
        "QWEN_GGUF_NO_COPY_PREFAULT": (
            None if arm == "A" else "1" if arm == "C" else "0"
        ),
        "QWEN_NATIVE_QUANT_EMBED": None,
    }


def parse_load_contract(stderr: str, arm: str) -> dict[str, object]:
    expected_policy = (
        "[metal-load] native quantized token embedding policy: "
        "auto-promoted (Q8_0 [2048, 248320])"
    )
    policy_lines = re.findall(
        r"^\[metal-load\] native quantized token embedding policy:.*$",
        stderr,
        re.MULTILINE,
    )
    ledger_lines = re.findall(r"^\[metal-load-ledger\].*$", stderr, re.MULTILINE)
    backing_lines = re.findall(r"^\[metal-gguf-retained\].*$", stderr, re.MULTILINE)
    legacy_lines = re.findall(r"^\[metal-gguf-no-copy\].*$", stderr, re.MULTILINE)
    if policy_lines != [expected_policy]:
        raise RuntimeError("native embedding production policy drifted")

    copied_ledger = (
        "[metal-load-ledger] source=733/22123538944 "
        "direct_copy=733/22123538944 direct_view=0/0 direct_alias=0/0 "
        "tail_fallback=0/0 converted=0/0/0 derived=0/0"
    )
    retained_ledger = (
        "[metal-load-ledger] source=733/22123538944 direct_copy=0/0 "
        "direct_view=732/22123530752 direct_alias=0/0 "
        "tail_fallback=1/8192 converted=0/0/0 derived=0/0"
    )
    if arm == "A":
        if ledger_lines != [copied_ledger] or backing_lines or legacy_lines:
            raise RuntimeError("A copied load contract drifted")
        return {"storage": "copied", "native_embedding": "auto-promoted"}

    if ledger_lines != [retained_ledger] or len(backing_lines) != 1 or legacy_lines:
        raise RuntimeError(f"{arm} retained load contract drifted")
    prefix = (
        r"^\[metal-gguf-retained\] windows=1 window_bytes=22123544576 "
        r"direct=733 view=732/22123530752 alias=0/0 fallback=1/8192 "
        r"page=16384 max_buffer=77309411328 alignment=32 "
    )
    if arm == "B":
        pattern = (
            prefix
            + r"prefault=disabled prefault_pages=0 prefault_bytes=0 "
            + r"prefault_ms=0\.000 checksum=0x0000000000000000$"
        )
        if re.fullmatch(pattern, backing_lines[0]) is None:
            raise RuntimeError("B retained backing contract drifted")
        return {
            "storage": "generic_retained_demand_paged",
            "native_embedding": "auto-promoted",
            "prefault": False,
            "prefault_pages": 0,
            "prefault_bytes": 0,
            "prefault_ms": 0.0,
            "prefault_checksum": "0x0000000000000000",
        }

    pattern = (
        prefix
        + rf"prefault=enabled prefault_pages={EXPECTED_PREFAULT_PAGES} "
        + rf"prefault_bytes={EXPECTED_PREFAULT_BYTES} "
        + r"prefault_ms=([0-9]+(?:\.[0-9]+)?) checksum=(0x[0-9a-f]{16})$"
    )
    match = re.fullmatch(pattern, backing_lines[0])
    if match is None:
        raise RuntimeError("C retained backing contract drifted")
    prefault_ms = float(match.group(1))
    if not math.isfinite(prefault_ms) or prefault_ms <= 0:
        raise RuntimeError(f"C prefault wall is invalid: {prefault_ms!r}")
    return {
        "storage": "generic_retained_cpu_prefault",
        "native_embedding": "auto-promoted",
        "prefault": True,
        "prefault_pages": EXPECTED_PREFAULT_PAGES,
        "prefault_bytes": EXPECTED_PREFAULT_BYTES,
        "prefault_ms": prefault_ms,
        "prefault_checksum": match.group(2),
    }


def require_unique_line(stderr: str, line: str, label: str) -> None:
    if stderr.splitlines().count(line) != 1:
        raise RuntimeError(f"expected one exact {label} line")


def parse_bench(stderr: str) -> dict[str, object]:
    header_pattern = (
        r"^\[bench\] model=.* \(419 tokens\), gen=127 tokens, "
        r"kv_capacity=1024$"
    )
    if len(re.findall(header_pattern, stderr, re.MULTILINE)) != 1:
        raise RuntimeError("benchmark prompt/token contract drifted")
    require_unique_line(stderr, "[bench] decode mode: full-logits", "decode mode")
    require_unique_line(
        stderr, "[bench] prefill mode: packed layer-major", "prefill mode"
    )
    require_unique_line(stderr, "[bench] prefill chunk: 1024", "prefill chunk")

    rep_pattern = re.compile(
        r"^\[bench\] rep\s+(\d+): prefill\s+([0-9.]+) ms "
        r"\(([0-9.]+) t/s\)\s+decode\s+([0-9.]+) ms "
        r"\(([0-9.]+) t/s\)$",
        re.MULTILINE,
    )
    matches = rep_pattern.findall(stderr)
    if len(matches) != RUNS or [int(row[0]) for row in matches] != list(
        range(1, RUNS + 1)
    ):
        raise RuntimeError(f"expected repetitions 1-{RUNS}, got {matches!r}")
    repetitions = [
        {
            "rep": int(rep),
            "prefill_ms": float(prefill_ms),
            "prefill_tps_reported": float(prefill_tps),
            "decode_ms": float(decode_ms),
            "decode_tps_reported": float(decode_tps),
        }
        for rep, prefill_ms, prefill_tps, decode_ms, decode_tps in matches
    ]
    for row in repetitions:
        for key, value in row.items():
            if key != "rep" and (not math.isfinite(value) or value <= 0):
                raise RuntimeError(f"invalid repetition metric {key}={value!r}")
        calculated_prefill_tps = PROMPT_TOKENS * 1000.0 / row["prefill_ms"]
        calculated_decode_tps = TOKENS * 1000.0 / row["decode_ms"]
        if abs(calculated_prefill_tps - row["prefill_tps_reported"]) > 1.0:
            raise RuntimeError("reported prefill throughput is inconsistent with wall")
        if abs(calculated_decode_tps - row["decode_tps_reported"]) > 0.2:
            raise RuntimeError("reported decode throughput is inconsistent with wall")

    profile_pattern = re.compile(
        r"^\[bench\] avg/token: total ([0-9.]+) ms \| "
        r"cpu_encode ([0-9.]+) ms \| gpu_kernel ([0-9.]+) ms \| "
        r"commit\+wait ([0-9.]+) ms \| cpu_route ([0-9.]+) ms \| "
        r"cmd_bufs ([0-9.]+)$",
        re.MULTILINE,
    )
    profile_matches = profile_pattern.findall(stderr)
    if len(profile_matches) != 1:
        raise RuntimeError("expected one last-repetition MoE profile")
    values = [float(value) for value in profile_matches[0]]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError("invalid MoE profile metric")
    profile = dict(
        zip(
            (
                "total_ms",
                "cpu_encode_ms",
                "gpu_kernel_ms",
                "commit_wait_ms",
                "cpu_route_ms",
                "command_buffers",
            ),
            values,
            strict=True,
        )
    )

    generated_lines = re.findall(r"^\[bench\] generated: (.*)$", stderr, re.MULTILINE)
    if len(generated_lines) != 1:
        raise RuntimeError("expected one generated-output line")
    generated = generated_lines[0]
    return {
        "repetitions": repetitions,
        "moe_profile_last_rep": profile,
        "generated_debug": generated,
        "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
    }


def append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")


def run_one(
    arm: str,
    block_index: int,
    block_order: str,
    order_position: int,
    attempt: int,
    prompt_text: str,
    base_env: dict[str, str],
) -> dict[str, object]:
    stem = (
        f"b{block_index:02d}-{block_order.lower()}-"
        f"r{order_position}-{arm.lower()}-a{attempt:02d}"
    )
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    for path in (stdout_path, stderr_path):
        if path.exists():
            raise RuntimeError(f"refusing to reuse attempt artifact {path}")

    before_cache = prior.common.wait_for_valid_host(f"{stem} cache read")
    vm_before_cache = prior.capture_vm_state()
    cache_ms, cache_bytes = prior.warm_file()
    if cache_bytes != MODEL.stat().st_size:
        raise RuntimeError("cache precondition did not read the complete model")
    time.sleep(COOLDOWN_S)
    before_spawn = prior.common.wait_for_valid_host(f"{stem} process spawn")
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

    arm_environment = expected_arm_environment(arm)
    env = base_env.copy()
    for key, value in arm_environment.items():
        if value is not None:
            env[key] = value
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
    after_exit = prior.common.capture_host_state()
    vm_after = prior.capture_vm_state()
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"{stem} exited {result.returncode}; see {stderr_path}")
    stdout = stdout_path.read_bytes()
    if stdout:
        raise RuntimeError(f"{stem} unexpectedly emitted stdout")

    load_contract = parse_load_contract(stderr, arm)
    bench = parse_bench(stderr)
    block_inputs = prior.common.parse_resource(stderr, "block input operations")
    page_faults = prior.common.parse_resource(stderr, "page faults")
    maximum_rss = prior.common.parse_resource(stderr, "maximum resident set size")
    peak_footprint = prior.common.parse_resource(stderr, "peak memory footprint")
    page_reclaims = prior.common.parse_resource(stderr, "page reclaims")
    process_pageout_delta = vm_after["pageouts"] - vm_before_spawn["pageouts"]
    process_swap_delta = (
        vm_after["swap_used_bytes"] - vm_before_spawn["swap_used_bytes"]
    )
    validity_reasons = []
    if block_inputs != 0:
        validity_reasons.append(f"block_input_operations={block_inputs}")
    if page_faults != 0:
        validity_reasons.append(f"major_page_faults={page_faults}")
    if process_pageout_delta != 0:
        validity_reasons.append(f"process_pageout_delta={process_pageout_delta}")
    if process_swap_delta > 0:
        validity_reasons.append(f"process_swap_growth_bytes={process_swap_delta}")
    if not after_exit["valid"]:
        validity_reasons.append("post_exit_host_invalid")
    return {
        "artifact_stem": stem,
        "block_index": block_index,
        "block_order": block_order,
        "order_position": order_position,
        "attempt": attempt,
        "arm": arm,
        "command": command,
        "arm_environment": arm_environment,
        "cache_precondition_ms": cache_ms,
        "process_wall_ms": process_wall_ms,
        "host_before_cache": before_cache,
        "host_before_spawn": before_spawn,
        "host_after_exit": after_exit,
        "vm_before_cache": vm_before_cache,
        "vm_before_spawn": vm_before_spawn,
        "vm_after_exit": vm_after,
        "cache_pageout_delta": cache_pageout_delta,
        "cache_swap_delta_bytes": cache_swap_delta,
        "process_pageout_delta": process_pageout_delta,
        "process_swap_delta_bytes": process_swap_delta,
        "maximum_resident_set_size": maximum_rss,
        "peak_memory_footprint": peak_footprint,
        "page_reclaims": page_reclaims,
        "page_faults": page_faults,
        "block_input_operations": block_inputs,
        "valid": not validity_reasons,
        "validity_reasons": validity_reasons,
        "load_contract": load_contract,
        "bench": bench,
    }


def run_valid_block(
    block_index: int,
    block_order: str,
    prompt_text: str,
    base_env: dict[str, str],
    attempts_path: Path,
    block_attempts_path: Path,
) -> list[dict[str, object]]:
    for attempt in range(1, MAX_BLOCK_ATTEMPTS + 1):
        rows = []
        for order_position, arm in enumerate(block_order, 1):
            row = run_one(
                arm,
                block_index,
                block_order,
                order_position,
                attempt,
                prompt_text,
                base_env,
            )
            rows.append(row)
            append_row(attempts_path, row)
            repetitions = row["bench"]["repetitions"]
            late_ms = statistics.median(rep["decode_ms"] for rep in repetitions[2:])
            print(
                f"block={block_index} order={block_order} "
                f"attempt={attempt} pos={order_position} arm={arm} "
                f"rep1={repetitions[0]['decode_ms']:.1f} ms "
                f"late={late_ms:.1f} ms valid={row['valid']}"
            )
        generated_hashes = {row["bench"]["generated_sha256"] for row in rows}
        if len(generated_hashes) != 1:
            raise RuntimeError(f"block {block_index} generated outputs differ")
        reasons = [
            f"{row['arm']}: {reason}"
            for row in rows
            for reason in row["validity_reasons"]
        ]
        append_row(
            block_attempts_path,
            {
                "schema": 1,
                "block_index": block_index,
                "block_order": block_order,
                "attempt": attempt,
                "accepted": not reasons,
                "reasons": reasons,
                "artifact_stems": [row["artifact_stem"] for row in rows],
            },
        )
        if not reasons:
            return rows
        print(
            f"block {block_index} attempt {attempt} invalid "
            f"({'; '.join(reasons)}); retrying full block"
        )
    raise RuntimeError(f"block {block_index} exhausted block attempts")


def process_metrics(row: dict[str, object]) -> dict[str, object]:
    repetitions = row["bench"]["repetitions"]
    decode_ms = [rep["decode_ms"] for rep in repetitions]
    prefill_ms = [rep["prefill_ms"] for rep in repetitions]
    decode_tps = [TOKENS * 1000.0 / wall for wall in decode_ms]
    prefill_tps = [PROMPT_TOKENS * 1000.0 / wall for wall in prefill_ms]
    late_decode_ms = statistics.median(decode_ms[2:])
    late_prefill_ms = statistics.median(prefill_ms[2:])
    late_decode_tps = TOKENS * 1000.0 / late_decode_ms
    late_prefill_tps = PROMPT_TOKENS * 1000.0 / late_prefill_ms
    late_decode_tps_samples = decode_tps[2:]
    late_spread = (
        max(late_decode_tps_samples) - min(late_decode_tps_samples)
    ) / statistics.median(late_decode_tps_samples)
    return {
        "block_index": row["block_index"],
        "block_order": row["block_order"],
        "arm": row["arm"],
        "rep1_decode_ms": decode_ms[0],
        "rep1_decode_tps": decode_tps[0],
        "late_decode_ms": late_decode_ms,
        "late_decode_tps": late_decode_tps,
        "rep1_to_late_recovery": late_decode_tps / decode_tps[0],
        "rep3_to_rep5_tps_ratio": decode_tps[4] / decode_tps[2],
        "late_decode_tps_relative_range": late_spread,
        "rep1_prefill_ms": prefill_ms[0],
        "rep1_prefill_tps": prefill_tps[0],
        "late_prefill_ms": late_prefill_ms,
        "late_prefill_tps": late_prefill_tps,
        "rep1_to_late_prefill_recovery": late_prefill_tps / prefill_tps[0],
        "decode_ms": decode_ms,
        "decode_tps": decode_tps,
        "prefill_ms": prefill_ms,
        "prefill_tps": prefill_tps,
    }


def block_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    effects = []
    for block_index, order in enumerate(BLOCK_ORDERS, 1):
        block = [row for row in rows if row["block_index"] == block_index]
        if len(block) != 3 or {row["arm"] for row in block} != set("ABC"):
            raise RuntimeError(f"block {block_index} is incomplete")
        arm_metrics = {
            arm: process_metrics(next(row for row in block if row["arm"] == arm))
            for arm in "ABC"
        }
        a = arm_metrics["A"]
        b = arm_metrics["B"]
        c = arm_metrics["C"]
        effects.append(
            {
                "block_index": block_index,
                "block_order": order,
                "arms": arm_metrics,
                "rep1_b_over_a": b["rep1_decode_tps"] / a["rep1_decode_tps"],
                "rep1_c_over_a": c["rep1_decode_tps"] / a["rep1_decode_tps"],
                "late_b_over_a": b["late_decode_tps"] / a["late_decode_tps"],
                "late_c_over_a": c["late_decode_tps"] / a["late_decode_tps"],
                "b_recovery_normalized_by_a": (
                    b["rep1_to_late_recovery"] / a["rep1_to_late_recovery"]
                ),
                "c_recovery_normalized_by_a": (
                    c["rep1_to_late_recovery"] / a["rep1_to_late_recovery"]
                ),
                "late_prefill_b_over_a": (
                    b["late_prefill_tps"] / a["late_prefill_tps"]
                ),
                "late_prefill_c_over_a": (
                    c["late_prefill_tps"] / a["late_prefill_tps"]
                ),
                "b_prefill_recovery_normalized_by_a": (
                    b["rep1_to_late_prefill_recovery"]
                    / a["rep1_to_late_prefill_recovery"]
                ),
                "c_prefill_recovery_normalized_by_a": (
                    c["rep1_to_late_prefill_recovery"]
                    / a["rep1_to_late_prefill_recovery"]
                ),
            }
        )
    return effects


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    effects = block_metrics(rows)
    rep1_b_over_a = [effect["rep1_b_over_a"] for effect in effects]
    late_b_over_a = [effect["late_b_over_a"] for effect in effects]
    a_metrics = [effect["arms"]["A"] for effect in effects]
    b_metrics = [effect["arms"]["B"] for effect in effects]
    reproduced = statistics.median(rep1_b_over_a) <= 0.93
    a_bounded = all(
        0.97 <= metric["rep1_to_late_recovery"] <= 1.03 for metric in a_metrics
    )
    late_stable = all(
        0.98 <= metric["rep3_to_rep5_tps_ratio"] <= 1.02
        and metric["late_decode_tps_relative_range"] <= 0.03
        for metric in a_metrics + b_metrics
    )
    b_material_recovery = (
        all(metric["rep1_to_late_recovery"] >= 1.03 for metric in b_metrics)
        and statistics.median(metric["rep1_to_late_recovery"] for metric in b_metrics)
        >= 1.05
    )
    loaded_recovery = (
        reproduced
        and all(ratio >= 0.95 for ratio in late_b_over_a)
        and statistics.median(late_b_over_a) >= 0.97
        and a_bounded
        and late_stable
        and b_material_recovery
    )
    persistent = (
        reproduced
        and all(ratio < 0.95 for ratio in late_b_over_a)
        and statistics.median(late_b_over_a) <= 0.93
        and a_bounded
        and late_stable
    )
    if loaded_recovery:
        classification = "same_request_loaded_recovery"
    elif persistent:
        classification = "persistent_retained_tax"
    else:
        classification = "mechanism_inconclusive"

    c_prefault_ms = [
        row["load_contract"]["prefault_ms"] for row in rows if row["arm"] == "C"
    ]
    return {
        "schema": 1,
        "status": classification,
        "classification": classification,
        "reproduced_material_rep1_deficit": reproduced,
        "copied_rep1_to_late_movement_bounded": a_bounded,
        "copied_and_retained_late_rows_stable": late_stable,
        "retained_absolute_recovery_material": b_material_recovery,
        "thresholds": {
            "rep1_median_b_over_a_max": 0.93,
            "loaded_recovery_every_late_b_over_a_min": 0.95,
            "loaded_recovery_median_late_b_over_a_min": 0.97,
            "loaded_recovery_every_b_absolute_recovery_min": 1.03,
            "loaded_recovery_median_b_absolute_recovery_min": 1.05,
            "every_a_absolute_recovery_min": 0.97,
            "every_a_absolute_recovery_max": 1.03,
            "every_a_b_rep5_over_rep3_min": 0.98,
            "every_a_b_rep5_over_rep3_max": 1.02,
            "every_a_b_late_relative_range_max": 0.03,
            "persistent_every_late_b_over_a_strict_max": 0.95,
            "persistent_median_late_b_over_a_max": 0.93,
        },
        "block_effects": effects,
        "median_rep1_b_over_a": statistics.median(rep1_b_over_a),
        "median_late_b_over_a": statistics.median(late_b_over_a),
        "median_rep1_c_over_a": statistics.median(
            effect["rep1_c_over_a"] for effect in effects
        ),
        "median_late_c_over_a": statistics.median(
            effect["late_c_over_a"] for effect in effects
        ),
        "median_b_recovery_normalized_by_a": statistics.median(
            effect["b_recovery_normalized_by_a"] for effect in effects
        ),
        "median_c_recovery_normalized_by_a": statistics.median(
            effect["c_recovery_normalized_by_a"] for effect in effects
        ),
        "median_late_prefill_b_over_a": statistics.median(
            effect["late_prefill_b_over_a"] for effect in effects
        ),
        "median_late_prefill_c_over_a": statistics.median(
            effect["late_prefill_c_over_a"] for effect in effects
        ),
        "median_b_prefill_recovery_normalized_by_a": statistics.median(
            effect["b_prefill_recovery_normalized_by_a"] for effect in effects
        ),
        "median_c_prefill_recovery_normalized_by_a": statistics.median(
            effect["c_prefill_recovery_normalized_by_a"] for effect in effects
        ),
        "material_b_prefill_recovery_relative_to_a": statistics.median(
            effect["b_prefill_recovery_normalized_by_a"] for effect in effects
        )
        >= 1.05,
        "c_prefault_wall_ms": c_prefault_ms,
        "median_c_prefault_wall_ms": statistics.median(c_prefault_ms),
        "authority": "same_request_loaded_process_convergence_only",
    }


def main() -> None:
    for path in required_manifest_paths():
        if not path.is_file():
            raise SystemExit(f"missing required path {path}")
    if prior.MODEL != MODEL:
        raise SystemExit("imported cache-conditioning model drifted")
    prompt_text = PROMPT.read_text(encoding="utf-8")
    base_env, removed = prior.common.normalized_environment()
    manifest = build_manifest(removed)
    if ARTIFACT.exists():
        raise SystemExit(
            f"packet directory {ARTIFACT} already exists; interruptions are terminal"
        )
    if not ARTIFACT.parent.is_dir():
        raise SystemExit(f"missing packet parent directory {ARTIFACT.parent}")
    ARTIFACT.mkdir()
    (ARTIFACT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows_path = ARTIFACT / "rows.jsonl"
    attempts_path = ARTIFACT / "attempts.jsonl"
    block_attempts_path = ARTIFACT / "block-attempts.jsonl"
    rows = []
    expected_generated_sha256 = None
    for block_index, order in enumerate(BLOCK_ORDERS, 1):
        block_rows = run_valid_block(
            block_index,
            order,
            prompt_text,
            base_env,
            attempts_path,
            block_attempts_path,
        )
        block_generated_sha256 = block_rows[0]["bench"]["generated_sha256"]
        if expected_generated_sha256 is None:
            expected_generated_sha256 = block_generated_sha256
        elif block_generated_sha256 != expected_generated_sha256:
            raise RuntimeError(f"block {block_index} generated output drifted")
        rows.extend(block_rows)
        for row in block_rows:
            append_row(rows_path, row)

    hashes = {row["bench"]["generated_sha256"] for row in rows}
    if len(hashes) != 1 or expected_generated_sha256 not in hashes:
        raise RuntimeError(f"accepted generated outputs differ: {sorted(hashes)}")
    c_checksums = {
        row["load_contract"]["prefault_checksum"] for row in rows if row["arm"] == "C"
    }
    if len(c_checksums) != 1:
        raise RuntimeError(f"C prefault checksums differ: {sorted(c_checksums)}")
    decision = summarize(rows)
    decision["generated_sha256"] = next(iter(hashes))
    decision["c_prefault_checksum"] = next(iter(c_checksums))
    (ARTIFACT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
