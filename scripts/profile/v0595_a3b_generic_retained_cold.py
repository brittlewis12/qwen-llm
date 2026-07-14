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

import v0593_demand_paged_no_copy as common


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0595-a3b-generic-retained-cold-p1"
PREREG = ROOT / "docs/bench/v0595-a3b-generic-retained-cold.md"
MODEL = Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
BINARY = ROOT / "target/release/qwen"
BENCH_BINARY = ROOT / "target/release/qwen-bench"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
EXPECTED_MODEL_SHA256 = (
    "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
)
EXPECTED_PROMPT_SHA256 = (
    "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
)
EXPECTED_RUNTIME_MODEL_ID = "e6024ce53109fdf7"
EXPECTED_RUNTIME_TOKENIZER_ID = "a4b0b26f8a8c9917"
PAIR_ORDERS = ("BA", "AB", "AB", "BA", "BA", "AB")
LENGTHS = (1, 128)
SCREEN_PAIRS = 2
COOLDOWN_S = 30.0
MAX_PAIR_ATTEMPTS = 3
MIN_FOOTPRINT_SAVING = 19_911_177_677

common.MODEL = MODEL
common.EXPECTED_RUNTIME_MODEL_ID = EXPECTED_RUNTIME_MODEL_ID
common.EXPECTED_RUNTIME_TOKENIZER_ID = EXPECTED_RUNTIME_TOKENIZER_ID


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
    for path in (Path(__file__).resolve(), PREREG):
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
        COMMON_RUNNER,
        PREREG,
        MODEL,
        PROMPT,
        BINARY,
        BENCH_BINARY,
    )


def build_manifest(removed_environment: list[str]) -> dict[str, object]:
    commit, build = source_and_build_identity()
    hashes = {str(path): common.sha256_file(path) for path in required_manifest_paths()}
    if hashes[str(MODEL)] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA-256 drifted")
    if hashes[str(PROMPT)] != EXPECTED_PROMPT_SHA256:
        raise RuntimeError("prompt SHA-256 drifted")
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "build_identity": build,
        "removed_environment": removed_environment,
        "sha256": hashes,
        "model_size_bytes": MODEL.stat().st_size,
        "lengths": list(LENGTHS),
        "pair_orders": list(PAIR_ORDERS),
        "screen_pairs": SCREEN_PAIRS,
        "cooldown_s": COOLDOWN_S,
    }


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


def parse_swap_bytes(text: str) -> int:
    match = re.search(r"\bused\s*=\s*([0-9.]+)([KMG])", text)
    if match is None:
        raise RuntimeError(f"could not parse swap usage: {text!r}")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return round(float(match.group(1)) * scale)


def capture_vm_state() -> dict[str, object]:
    vm_stat = command_text(["vm_stat"])
    swap = command_text(["sysctl", "-n", "vm.swapusage"])
    pageouts = re.search(r"^Pageouts:\s+(\d+)\.$", vm_stat, re.MULTILINE)
    if pageouts is None:
        raise RuntimeError("could not parse vm_stat pageouts")
    return {
        "pageouts": int(pageouts.group(1)),
        "swap_used_bytes": parse_swap_bytes(swap),
        "vm_stat": vm_stat,
        "swapusage": swap,
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
    generic_lines = re.findall(r"^\[metal-gguf-retained\].*$", stderr, re.MULTILINE)
    legacy_lines = re.findall(r"^\[metal-gguf-no-copy\].*$", stderr, re.MULTILINE)
    if policy_lines != [expected_policy]:
        raise RuntimeError("native embedding production policy drifted")
    if arm == "A":
        expected_ledger = (
            "[metal-load-ledger] source=733/22123538944 "
            "direct_copy=733/22123538944 direct_view=0/0 direct_alias=0/0 "
            "tail_fallback=0/0 converted=0/0/0 derived=0/0"
        )
        if ledger_lines != [expected_ledger] or generic_lines or legacy_lines:
            raise RuntimeError("A copied load contract drifted")
        return {"storage": "copied", "native_embedding": "auto-promoted"}

    expected_backing = (
        "[metal-gguf-retained] windows=1 window_bytes=22123544576 "
        "direct=733 view=732/22123530752 alias=0/0 fallback=1/8192 "
        "page=16384 max_buffer=77309411328 alignment=32 "
        "prefault=disabled prefault_pages=0 prefault_bytes=0 "
        "prefault_ms=0.000 checksum=0x0000000000000000"
    )
    expected_ledger = (
        "[metal-load-ledger] source=733/22123538944 direct_copy=0/0 "
        "direct_view=732/22123530752 direct_alias=0/0 "
        "tail_fallback=1/8192 converted=0/0/0 derived=0/0"
    )
    if (
        generic_lines != [expected_backing]
        or ledger_lines != [expected_ledger]
        or legacy_lines
    ):
        raise RuntimeError("B generic retained load contract drifted")
    return {
        "storage": "generic_retained_demand_paged",
        "native_embedding": "auto-promoted",
        "prefault": False,
    }


def expected_arm_environment(arm: str) -> dict[str, str | None]:
    return {
        "QWEN_GGUF_NO_COPY": "1" if arm == "B" else "0",
        "QWEN_GGUF_NO_COPY_PREFAULT": "0" if arm == "B" else None,
        "QWEN_NATIVE_QUANT_EMBED": None,
    }


def run_one(
    arm: str,
    tokens: int,
    pair_index: int,
    attempt: int,
    run_index: int,
    order: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
    stem = f"n{tokens:03d}-p{pair_index:02d}-a{attempt:02d}-r{run_index}-{arm.lower()}"
    timing_path = ARTIFACT / f"{stem}.timing.jsonl"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    for path in (timing_path, stdout_path, stderr_path):
        if path.exists():
            raise RuntimeError(f"refusing to reuse attempt artifact {path}")

    before_cache = common.wait_for_valid_host(f"{stem} cache read")
    vm_before_cache = capture_vm_state()
    cache_ms, cache_bytes = warm_file()
    if cache_bytes != MODEL.stat().st_size:
        raise RuntimeError("cache precondition did not read the complete model")
    time.sleep(COOLDOWN_S)
    before_spawn = common.wait_for_valid_host(f"{stem} process spawn")
    vm_before_spawn = capture_vm_state()
    cache_pageout_delta = vm_before_spawn["pageouts"] - vm_before_cache["pageouts"]
    cache_swap_delta = (
        vm_before_spawn["swap_used_bytes"] - vm_before_cache["swap_used_bytes"]
    )
    if cache_pageout_delta != 0 or cache_swap_delta > 0:
        raise RuntimeError(
            f"{stem} cache conditioning changed VM pressure: "
            f"pageouts={cache_pageout_delta} swap={cache_swap_delta}"
        )

    env = base_env.copy()
    arm_environment = expected_arm_environment(arm)
    env["QWEN_GGUF_NO_COPY"] = arm_environment["QWEN_GGUF_NO_COPY"] or "0"
    if arm_environment["QWEN_GGUF_NO_COPY_PREFAULT"] is not None:
        env["QWEN_GGUF_NO_COPY_PREFAULT"] = "0"
    command = [
        "/usr/bin/time",
        "-l",
        str(BINARY),
        "--model",
        str(MODEL),
        "--prompt-file",
        str(PROMPT),
        "--tokens",
        str(tokens),
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
    after_exit = common.capture_host_state()
    vm_after = capture_vm_state()
    stdout = first + rest
    stdout_path.write_bytes(stdout)
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    if returncode != 0:
        raise RuntimeError(f"{stem} exited {returncode}; see {stderr_path}")
    if not first or first_byte_ms is None:
        raise RuntimeError(f"{stem} emitted no stdout")

    timings = [
        json.loads(line) for line in timing_path.read_text().splitlines() if line
    ]
    if len(timings) != 1:
        raise RuntimeError(f"{stem} expected one timing row, got {len(timings)}")
    timing = timings[0]
    common.validate_timing(timing, tokens, manifest)
    block_inputs = common.parse_resource(stderr, "block input operations")
    page_faults = common.parse_resource(stderr, "page faults")
    maximum_rss = common.parse_resource(stderr, "maximum resident set size")
    peak_footprint = common.parse_resource(stderr, "peak memory footprint")
    page_reclaims = common.parse_resource(stderr, "page reclaims")
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
    outer_residual_ms = (
        first_byte_ms - timing["runtime_and_model_load_ms"] - timing["ttft_ms"]
    )
    return {
        "artifact_stem": stem,
        "tokens": tokens,
        "pair_index": pair_index,
        "pair_order": order,
        "attempt": attempt,
        "run_index": run_index,
        "arm": arm,
        "command": command,
        "arm_environment": arm_environment,
        "cache_precondition_ms": cache_ms,
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
        "spawn_to_first_byte_ms": first_byte_ms,
        "spawn_to_exit_ms": exit_ms,
        "outer_residual_ms": outer_residual_ms,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "maximum_resident_set_size": maximum_rss,
        "peak_memory_footprint": peak_footprint,
        "page_reclaims": page_reclaims,
        "page_faults": page_faults,
        "block_input_operations": block_inputs,
        "valid": not validity_reasons,
        "validity_reasons": validity_reasons,
        "load_contract": parse_load_contract(stderr, arm),
        "timing": timing,
    }


def append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")


def run_valid_pair(
    tokens: int,
    pair_index: int,
    order: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
    attempts_path: Path,
    pair_attempts_path: Path,
) -> list[dict[str, object]]:
    for attempt in range(1, MAX_PAIR_ATTEMPTS + 1):
        rows = []
        for run_index, arm in enumerate(order, 1):
            row = run_one(
                arm,
                tokens,
                pair_index,
                attempt,
                run_index,
                order,
                base_env,
                manifest,
            )
            rows.append(row)
            append_row(attempts_path, row)
            print(
                f"n{tokens} p{pair_index} a{attempt} r{run_index} {arm} "
                f"first={row['spawn_to_first_byte_ms']:.1f} "
                f"exit={row['spawn_to_exit_ms']:.1f} "
                f"load={row['timing']['runtime_and_model_load_ms']:.1f} "
                f"prefill={row['timing']['prefill_ms']:.1f} "
                f"tps={row['timing']['transition_tps']:.3f} valid={row['valid']}"
            )
        reasons = []
        for row in rows:
            reasons.extend(
                f"{row['arm']}: {reason}" for reason in row["validity_reasons"]
            )
        output_mismatch = rows[0]["stdout_sha256"] != rows[1]["stdout_sha256"]
        if output_mismatch:
            reasons.append("A/B output mismatch")
        append_row(
            pair_attempts_path,
            {
                "schema": 1,
                "tokens": tokens,
                "pair_index": pair_index,
                "pair_order": order,
                "attempt": attempt,
                "accepted": not reasons,
                "reasons": reasons,
                "artifact_stems": [row["artifact_stem"] for row in rows],
            },
        )
        if output_mismatch:
            raise RuntimeError(f"n{tokens} p{pair_index} A/B output mismatch")
        if not reasons:
            return rows
        print(
            f"n{tokens} p{pair_index} attempt {attempt} invalid "
            f"({'; '.join(reasons)}); retrying full pair"
        )
    raise RuntimeError(f"n{tokens} p{pair_index} exhausted pair attempts")


def median_summary(effects: list[dict[str, object]]) -> dict[str, float]:
    keys = (
        "first_speedup",
        "first_saving_ms",
        "exit_speedup",
        "exit_saving_ms",
        "load_saving_ms",
        "footprint_saving_bytes",
        "outer_residual_shift_ms",
    )
    return {f"median_{key}": statistics.median(e[key] for e in effects) for key in keys}


def paired_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    effects = []
    tokens = rows[0]["tokens"]
    for pair_index in sorted({row["pair_index"] for row in rows}):
        pair = [row for row in rows if row["pair_index"] == pair_index]
        a = next(row for row in pair if row["arm"] == "A")
        b = next(row for row in pair if row["arm"] == "B")
        a_misses = tuple(
            a["timing"]["pso_cache"][phase]["misses"]
            for phase in ("prefill", "generation", "total")
        )
        b_misses = tuple(
            b["timing"]["pso_cache"][phase]["misses"]
            for phase in ("prefill", "generation", "total")
        )
        effect = {
            "pair_index": pair_index,
            "pair_order": a["pair_order"],
            "first_speedup": a["spawn_to_first_byte_ms"] / b["spawn_to_first_byte_ms"],
            "first_saving_ms": a["spawn_to_first_byte_ms"]
            - b["spawn_to_first_byte_ms"],
            "exit_speedup": a["spawn_to_exit_ms"] / b["spawn_to_exit_ms"],
            "exit_saving_ms": a["spawn_to_exit_ms"] - b["spawn_to_exit_ms"],
            "load_saving_ms": a["timing"]["runtime_and_model_load_ms"]
            - b["timing"]["runtime_and_model_load_ms"],
            "footprint_saving_bytes": a["peak_memory_footprint"]
            - b["peak_memory_footprint"],
            "outer_residual_shift_ms": a["outer_residual_ms"] - b["outer_residual_ms"],
            "pso_miss_tuple_a": a_misses,
            "pso_miss_tuple_b": b_misses,
            "pso_misses_equal": a_misses == b_misses,
        }
        if tokens > 1:
            transition_count = a["timing"]["transition_count"]
            if transition_count != b["timing"]["transition_count"]:
                raise RuntimeError("A/B transition counts differ")
            a_ms = a["timing"]["transition_ms"] / transition_count
            b_ms = b["timing"]["transition_ms"] / transition_count
            effect.update(
                {
                    "transition_tps_b_over_a": b["timing"]["transition_tps"]
                    / a["timing"]["transition_tps"],
                    "transition_a_ms_per_token": a_ms,
                    "transition_b_ms_per_token": b_ms,
                    "transition_b_minus_a_ms_per_token": b_ms - a_ms,
                }
            )
        effects.append(effect)
    order_summaries = {}
    for order in ("BA", "AB"):
        order_effects = [effect for effect in effects if effect["pair_order"] == order]
        order_summaries[order] = median_summary(order_effects)
    result = {
        "pair_count": len(effects),
        "effects": effects,
        "overall": median_summary(effects),
        "by_order": order_summaries,
        "outer_residual_order_interaction_ms": (
            order_summaries["BA"]["median_outer_residual_shift_ms"]
            - order_summaries["AB"]["median_outer_residual_shift_ms"]
        ),
    }
    if tokens > 1:
        a_tps = statistics.median(
            row["timing"]["transition_tps"] for row in rows if row["arm"] == "A"
        )
        b_tps = statistics.median(
            row["timing"]["transition_tps"] for row in rows if row["arm"] == "B"
        )
        result["transition"] = {
            "marginal_baseline_tps": a_tps,
            "marginal_retained_tps": b_tps,
            "marginal_retained_over_baseline": b_tps / a_tps,
            "median_paired_retained_over_baseline": statistics.median(
                effect["transition_tps_b_over_a"] for effect in effects
            ),
            "median_paired_baseline_ms_per_transition": statistics.median(
                effect["transition_a_ms_per_token"] for effect in effects
            ),
            "median_paired_retained_ms_per_transition": statistics.median(
                effect["transition_b_ms_per_token"] for effect in effects
            ),
            "median_paired_delta_ms_per_transition": statistics.median(
                effect["transition_b_minus_a_ms_per_token"] for effect in effects
            ),
        }
    else:
        result["transition"] = None
    return result


def thresholds(tokens: int) -> dict[str, float]:
    return {
        "first_speedup": 1.20,
        "first_saving_ms": 750.0,
        "exit_speedup": 1.20 if tokens == 1 else 1.10,
        "exit_saving_ms": 750.0 if tokens == 1 else 500.0,
    }


def metrics_pass(tokens: int, metrics: dict[str, object]) -> bool:
    gate = thresholds(tokens)
    effects = metrics["effects"]
    if not all(effect["first_saving_ms"] > 0 for effect in effects):
        return False
    if not all(effect["exit_saving_ms"] > 0 for effect in effects):
        return False
    for summary in [metrics["overall"], *metrics["by_order"].values()]:
        if summary["median_first_speedup"] < gate["first_speedup"]:
            return False
        if summary["median_first_saving_ms"] < gate["first_saving_ms"]:
            return False
        if summary["median_exit_speedup"] < gate["exit_speedup"]:
            return False
        if summary["median_exit_saving_ms"] < gate["exit_saving_ms"]:
            return False
    return (
        metrics["overall"]["median_load_saving_ms"] > 0
        and metrics["overall"]["median_footprint_saving_bytes"] >= MIN_FOOTPRINT_SAVING
    )


def local_review_warnings(tokens: int, metrics: dict[str, object]) -> list[str]:
    warnings = []
    if any(not effect["pso_misses_equal"] for effect in metrics["effects"]):
        warnings.append(f"n{tokens} A/B PSO miss-count tuples differ")
    if any(
        abs(effect["outer_residual_shift_ms"]) > 200.0 for effect in metrics["effects"]
    ):
        warnings.append(f"n{tokens} pairwise outer residual shift exceeds 200 ms")
    for order in ("BA", "AB"):
        shift = metrics["by_order"][order]["median_outer_residual_shift_ms"]
        if abs(shift) > 150.0:
            warnings.append(
                f"n{tokens} {order} median outer residual shift exceeds 150 ms"
            )
    transition = metrics["transition"]
    if (
        transition is not None
        and transition["median_paired_retained_over_baseline"] < 0.95
    ):
        warnings.append(
            "retained transition throughput regresses by more than five percent"
        )
    return warnings


def cross_length_warnings(metrics_by_length: dict[str, object]) -> list[str]:
    n1 = metrics_by_length["1"]["full"]
    n128 = metrics_by_length["128"]["full"]
    first_delta = (
        n128["overall"]["median_first_saving_ms"]
        - n1["overall"]["median_first_saving_ms"]
    )
    warnings = []
    if abs(first_delta) > 200.0:
        warnings.append(
            f"n128 first-byte saving differs from n1 by {first_delta:.1f} ms"
        )
    return warnings


def write_decision(path: Path, decision: dict[str, object]) -> None:
    path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")


def require_single_output(rows: list[dict[str, object]], tokens: int) -> None:
    if len({row["stdout_sha256"] for row in rows}) != 1:
        raise RuntimeError(f"n{tokens} canonical outputs differ across pairs")


def main() -> None:
    for path in required_manifest_paths():
        if not path.is_file():
            raise SystemExit(f"missing required path {path}")
    base_env, removed = common.normalized_environment()
    manifest = build_manifest(removed)
    if ARTIFACT.exists():
        raise SystemExit(
            f"packet directory {ARTIFACT} already exists; interruptions are terminal"
        )
    if not ARTIFACT.parent.is_dir():
        raise SystemExit(f"missing packet parent directory {ARTIFACT.parent}")
    ARTIFACT.mkdir()
    (ARTIFACT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    canonical_path = ARTIFACT / "rows.jsonl"
    attempts_path = ARTIFACT / "attempts.jsonl"
    pair_attempts_path = ARTIFACT / "pair-attempts.jsonl"
    decision_path = ARTIFACT / "decision.json"
    metrics_by_length = {}
    accumulated_warnings = []
    for tokens in LENGTHS:
        length_rows = []
        for pair_index, order in enumerate(PAIR_ORDERS[:SCREEN_PAIRS], 1):
            pair_rows = run_valid_pair(
                tokens,
                pair_index,
                order,
                base_env,
                manifest,
                attempts_path,
                pair_attempts_path,
            )
            length_rows.extend(pair_rows)
            for row in pair_rows:
                append_row(canonical_path, row)
        screen = paired_metrics(length_rows)
        metrics_by_length[str(tokens)] = {"screen": screen, "full": None}
        require_single_output(length_rows, tokens)
        if not metrics_pass(tokens, screen):
            screen_warnings = local_review_warnings(tokens, screen)
            decision = {
                "schema": 1,
                "status": "stopped_screen",
                "failed_length": tokens,
                "metrics_by_length": metrics_by_length,
                "warnings": accumulated_warnings + screen_warnings,
            }
            write_decision(decision_path, decision)
            print(json.dumps(decision, indent=2, sort_keys=True))
            return

        for pair_index, order in enumerate(
            PAIR_ORDERS[SCREEN_PAIRS:], SCREEN_PAIRS + 1
        ):
            pair_rows = run_valid_pair(
                tokens,
                pair_index,
                order,
                base_env,
                manifest,
                attempts_path,
                pair_attempts_path,
            )
            length_rows.extend(pair_rows)
            for row in pair_rows:
                append_row(canonical_path, row)
        require_single_output(length_rows, tokens)
        full = paired_metrics(length_rows)
        metrics_by_length[str(tokens)]["full"] = full
        local_warnings = local_review_warnings(tokens, full)
        if not metrics_pass(tokens, full):
            decision = {
                "schema": 1,
                "status": "stopped_full",
                "failed_length": tokens,
                "metrics_by_length": metrics_by_length,
                "warnings": accumulated_warnings + local_warnings,
            }
            write_decision(decision_path, decision)
            print(json.dumps(decision, indent=2, sort_keys=True))
            return
        if local_warnings:
            decision = {
                "schema": 1,
                "status": "needs_review",
                "failed_length": None,
                "review_length": tokens,
                "metrics_by_length": metrics_by_length,
                "warnings": accumulated_warnings + local_warnings,
                "endpoint": "instrumented_cli_spawn_to_first_byte_and_exit",
                "fit": None,
            }
            write_decision(decision_path, decision)
            print(json.dumps(decision, indent=2, sort_keys=True))
            return
        accumulated_warnings.extend(local_warnings)

    warnings = accumulated_warnings + cross_length_warnings(metrics_by_length)
    decision = {
        "schema": 1,
        "status": "needs_review" if warnings else "passed_packet",
        "failed_length": None,
        "metrics_by_length": metrics_by_length,
        "warnings": warnings,
        "endpoint": "instrumented_cli_spawn_to_first_byte_and_exit",
        "fit": None,
    }
    write_decision(decision_path, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
