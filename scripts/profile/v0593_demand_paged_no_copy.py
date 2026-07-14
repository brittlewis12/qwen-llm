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


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0593-demand-paged-no-copy-p1"
PREREG = ROOT / "docs/bench/v0593-demand-paged-no-copy.md"
MODEL = Path("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf")
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
BINARY = ROOT / "target/release/qwen"
BENCH_BINARY = ROOT / "target/release/qwen-bench"
EXPECTED_MODEL_SHA256 = (
    "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0"
)
EXPECTED_PROMPT_SHA256 = (
    "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
)
EXPECTED_RUNTIME_MODEL_ID = "6247cb71b536c975"
EXPECTED_RUNTIME_TOKENIZER_ID = "a4b0b26f8a8c9917"
NORMALIZED_PREFIXES = ("QWEN_", "MTL_", "METAL_")
LENGTHS = (1, 256, 32, 128)
PAIR_ORDERS = ("BA", "AB")
COOLDOWN_S = 30.0
MAX_PAIR_ATTEMPTS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def normalized_environment() -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    removed = sorted(
        key for key in env if key.startswith(NORMALIZED_PREFIXES) or key == "RUST_LOG"
    )
    for key in removed:
        env.pop(key)
    return env, removed


def required_manifest_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), PREREG, MODEL, PROMPT, BINARY, BENCH_BINARY)


def build_manifest(removed_environment: list[str]) -> dict[str, object]:
    commit, build = source_and_build_identity()
    files = required_manifest_paths()
    hashes = {str(file): sha256_file(file) for file in files}
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
    }


def capture_host_state() -> dict[str, object]:
    thermal = command_text(["pmset", "-g", "therm"])
    battery = command_text(["pmset", "-g", "batt"])
    memory = command_text(["memory_pressure", "-Q"])
    match = re.search(r"System-wide memory free percentage: (\d+)%", memory)
    available = int(match.group(1)) if match else None
    valid = (
        "No thermal warning level has been recorded" in thermal
        and "No performance warning level has been recorded" in thermal
        and "AC Power" in battery
        and available is not None
        and available >= 50
    )
    return {
        "thermal": thermal,
        "battery": battery,
        "memory_pressure": memory,
        "memory_available_percent": available,
        "valid": valid,
    }


def wait_for_valid_host(label: str) -> dict[str, object]:
    for attempt in range(1, 7):
        state = capture_host_state()
        if state["valid"]:
            return state
        if attempt == 6:
            raise RuntimeError(f"host remained invalid before {label}: {state}")
        time.sleep(30.0)
    raise AssertionError("unreachable")


def warm_file() -> tuple[float, int]:
    started = time.perf_counter()
    total = 0
    buf = bytearray(8 * 1024 * 1024)
    with MODEL.open("rb", buffering=0) as handle:
        while True:
            count = handle.readinto(buf)
            if count == 0:
                break
            total += count
    return (time.perf_counter() - started) * 1e3, total


def parse_resource(stderr: str, label: str) -> int:
    matches = re.findall(rf"^\s*(\d+)\s+{re.escape(label)}$", stderr, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label!r} counter, got {len(matches)}")
    return int(matches[0])


def parse_load_contract(stderr: str, arm: str) -> dict[str, object]:
    if arm == "A":
        ledger = re.search(
            r"^\[metal-load-ledger\] source=851/16806250496 "
            r"direct_copy=851/16806250496 direct_view=0/0 "
            r"tail_fallback=0/0 converted=0/0/0 derived=0/0$",
            stderr,
            re.MULTILINE,
        )
        if ledger is None or "[metal-gguf-no-copy]" in stderr:
            raise RuntimeError("A copied load contract drifted")
        return {"storage": "copied", "prefault": None}

    backing = re.search(
        r"^\[metal-gguf-no-copy\] "
        r"mapped=16817244384 exposed=16817242112 suffix=2272 "
        r"page=16384 pages=1026443 alignment=32 "
        r"prefault=disabled prefault_pages=0 prefault_ms=0\.000 "
        r"checksum=0x0000000000000000$",
        stderr,
        re.MULTILINE,
    )
    ledger = re.search(
        r"^\[metal-load-ledger\] source=851/16806250496 direct_copy=0/0 "
        r"direct_view=850/16806230016 tail_fallback=1/20480 "
        r"converted=0/0/0 derived=0/0$",
        stderr,
        re.MULTILINE,
    )
    if backing is None or ledger is None:
        raise RuntimeError("B demand-paged load contract drifted")
    return {"storage": "retained_demand_paged", "prefault": False}


def require_nonnegative_number(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric: {value!r}")
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"{label} is not finite and nonnegative: {value!r}")


def require_integer(value: object, label: str, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} is not an integer: {value!r}")
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{label} is below {minimum}: {value!r}")


def validate_timing(
    timing: dict[str, object],
    tokens: int,
    manifest: dict[str, object],
) -> None:
    expected = {
        "schema_version": 3,
        "request_epoch": "first_post_model_load",
        "request_index": 0,
        "tokenizer_reused": False,
        "pair_requested": False,
        "pair_id": None,
        "pair_request_equal": None,
        "pair_generated_tokens_equal": None,
        "prefix_cache_used": False,
        "build_commit": manifest["source_commit"],
        "build_dirty": "0",
        "build_source_state": manifest["build_identity"]["build_source_state"],
        "model": str(MODEL),
        "runtime_identity_kind": "metadata_compatibility_v1",
        "runtime_model_id": EXPECTED_RUNTIME_MODEL_ID,
        "runtime_tokenizer_id": EXPECTED_RUNTIME_TOKENIZER_ID,
        "stdout_sink": "redirected",
        "ttft_endpoint": "stdout_flush_complete",
        "prompt_source": "file",
        "prompt_bytes": 1891,
        "prompt_tokens": 419,
        "requested_tokens": tokens,
        "generated_tokens": tokens,
        "stop_reason": "token_limit",
        "decode_policy": "greedy_argmax",
        "terminal_token_target_transition_consumed": False,
        "no_special_tokens": False,
        "prefill_chunk_requested": 1024,
        "prefill_chunk_effective": 419,
        "max_context_tokens": 1024,
        "transition_count": tokens - 1,
    }
    required_scalars = {
        "runtime_and_model_load_ms",
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
        "transition_ms",
        "transition_tps",
        "inference_complete_ms",
        "total_request_ms",
    }
    expected_keys = (
        set(expected)
        | required_scalars
        | {
            "request_start_unix_ms",
            "pso_cache",
            "metal_allocated",
        }
    )
    if set(timing) != expected_keys:
        missing = sorted(expected_keys - set(timing))
        extra = sorted(set(timing) - expected_keys)
        raise RuntimeError(f"timing schema drifted: missing={missing}, extra={extra}")
    for key, value in expected.items():
        if timing.get(key) != value:
            raise RuntimeError(f"timing field {key}={timing.get(key)!r} != {value!r}")
    require_integer(timing["request_start_unix_ms"], "request_start_unix_ms", 1)
    for key in required_scalars:
        require_nonnegative_number(timing[key], f"timing.{key}")

    pso_cache = timing["pso_cache"]
    if not isinstance(pso_cache, dict):
        raise RuntimeError("pso_cache is not an object")
    if set(pso_cache) != {"prefill", "generation", "total"}:
        raise RuntimeError("PSO cache checkpoints drifted")
    pso_metrics = {"misses", "miss_wall_ns", "compiler_wall_ns"}
    for phase, metrics in pso_cache.items():
        if not isinstance(metrics, dict) or set(metrics) != pso_metrics:
            raise RuntimeError(f"PSO cache schema drifted for {phase}")
        for metric, value in metrics.items():
            require_integer(value, f"pso_cache.{phase}.{metric}", 0)
    for metric in pso_metrics:
        expected_total = pso_cache["prefill"][metric] + pso_cache["generation"][metric]
        if pso_cache["total"][metric] != expected_total:
            raise RuntimeError(f"PSO cache total drifted for {metric}")

    expected_allocations = {
        "process_model_ready",
        "request_start",
        "after_scratch",
        "after_sequence",
        "after_prefill",
        "after_first_stdout_flush",
        "request_end_before_state_drop",
        "after_request_state_drop",
        "current_allocated_sampled_max_bytes",
    }
    metal_allocated = timing["metal_allocated"]
    if not isinstance(metal_allocated, dict):
        raise RuntimeError("metal_allocated is not an object")
    if set(metal_allocated) != expected_allocations:
        raise RuntimeError("Metal allocation checkpoints drifted")
    sample_keys = {
        "current_bytes",
        "delta_from_model_ready_bytes",
        "delta_from_request_start_bytes",
    }
    for checkpoint in expected_allocations - {"current_allocated_sampled_max_bytes"}:
        sample = metal_allocated[checkpoint]
        if not isinstance(sample, dict) or set(sample) != sample_keys:
            raise RuntimeError(f"Metal allocation schema drifted for {checkpoint}")
        require_integer(
            sample["current_bytes"], f"metal_allocated.{checkpoint}.current", 0
        )
        require_integer(
            sample["delta_from_model_ready_bytes"],
            f"metal_allocated.{checkpoint}.delta_model",
        )
        require_integer(
            sample["delta_from_request_start_bytes"],
            f"metal_allocated.{checkpoint}.delta_request",
        )
    require_integer(
        metal_allocated["current_allocated_sampled_max_bytes"],
        "metal_allocated.current_allocated_sampled_max_bytes",
        0,
    )


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

    before_cache = wait_for_valid_host(f"{stem} cache read")
    cache_ms, cache_bytes = warm_file()
    if cache_bytes != MODEL.stat().st_size:
        raise RuntimeError("cache precondition did not read the complete model")
    time.sleep(COOLDOWN_S)
    before_spawn = wait_for_valid_host(f"{stem} process spawn")

    env = base_env.copy()
    env["QWEN_NATIVE_QUANT_EMBED"] = "1"
    env["QWEN_GGUF_NO_COPY"] = "1" if arm == "B" else "0"
    if arm == "B":
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
    after_exit = capture_host_state()
    stdout = first + rest
    stdout_path.write_bytes(stdout)
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    if returncode != 0:
        raise RuntimeError(f"{stem} exited {returncode}; see {stderr_path}")
    if not first:
        raise RuntimeError(f"{stem} emitted no stdout")

    timings = [
        json.loads(line) for line in timing_path.read_text().splitlines() if line
    ]
    if len(timings) != 1:
        raise RuntimeError(f"{stem} expected one timing row, got {len(timings)}")
    timing = timings[0]
    validate_timing(timing, tokens, manifest)
    block_inputs = parse_resource(stderr, "block input operations")
    page_faults = parse_resource(stderr, "page faults")
    maximum_rss = parse_resource(stderr, "maximum resident set size")
    peak_footprint = parse_resource(stderr, "peak memory footprint")
    page_reclaims = parse_resource(stderr, "page reclaims")
    validity_reasons = []
    if block_inputs != 0:
        validity_reasons.append(f"block_input_operations={block_inputs}")
    if page_faults != 0:
        validity_reasons.append(f"major_page_faults={page_faults}")
    if not after_exit["valid"]:
        validity_reasons.append("post_exit_host_invalid")
    return {
        "artifact_stem": stem,
        "tokens": tokens,
        "pair_index": pair_index,
        "pair_order": order,
        "attempt": attempt,
        "run_index": run_index,
        "arm": arm,
        "cache_precondition_ms": cache_ms,
        "host_before_cache": before_cache,
        "host_before_spawn": before_spawn,
        "host_after_exit": after_exit,
        "spawn_to_first_byte_ms": first_byte_ms,
        "spawn_to_exit_ms": exit_ms,
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
        if rows[0]["stdout_sha256"] != rows[1]["stdout_sha256"]:
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
        if not reasons:
            return rows
        print(
            f"n{tokens} p{pair_index} attempt {attempt} invalid "
            f"({'; '.join(reasons)}); retrying full pair"
        )
    raise RuntimeError(f"n{tokens} p{pair_index} exhausted pair attempts")


def paired_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    effects = []
    for pair_index in (1, 2):
        pair = [row for row in rows if row["pair_index"] == pair_index]
        a = next(row for row in pair if row["arm"] == "A")
        b = next(row for row in pair if row["arm"] == "B")
        effects.append(
            {
                "pair_index": pair_index,
                "first_speedup": a["spawn_to_first_byte_ms"]
                / b["spawn_to_first_byte_ms"],
                "first_saving_ms": a["spawn_to_first_byte_ms"]
                - b["spawn_to_first_byte_ms"],
                "exit_speedup": a["spawn_to_exit_ms"] / b["spawn_to_exit_ms"],
                "exit_saving_ms": a["spawn_to_exit_ms"] - b["spawn_to_exit_ms"],
            }
        )
    return {
        "effects": effects,
        "median_first_speedup": statistics.median(
            effect["first_speedup"] for effect in effects
        ),
        "median_first_saving_ms": statistics.median(
            effect["first_saving_ms"] for effect in effects
        ),
        "median_exit_speedup": statistics.median(
            effect["exit_speedup"] for effect in effects
        ),
        "median_exit_saving_ms": statistics.median(
            effect["exit_saving_ms"] for effect in effects
        ),
    }


def length_passes(tokens: int, metrics: dict[str, object]) -> bool:
    effects = metrics["effects"]
    if not all(effect["first_saving_ms"] > 0 for effect in effects):
        return False
    if not all(effect["exit_saving_ms"] > 0 for effect in effects):
        return False
    if metrics["median_first_speedup"] < 1.20:
        return False
    if metrics["median_first_saving_ms"] < 750.0:
        return False
    if tokens == 1:
        return True
    return (
        metrics["median_exit_speedup"] >= 1.10
        and metrics["median_exit_saving_ms"] >= 500.0
    )


def linear_fit(points: list[tuple[float, float]]) -> dict[str, float]:
    x_mean = statistics.mean(point[0] for point in points)
    y_mean = statistics.mean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    beta = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    return {"alpha_ms": y_mean - beta * x_mean, "beta_ms_per_transition": beta}


def fit_arms(rows: list[dict[str, object]]) -> dict[str, object]:
    fits = {}
    for arm in ("A", "B"):
        points = []
        for tokens in sorted(set(row["tokens"] for row in rows)):
            walls = [
                row["spawn_to_exit_ms"]
                for row in rows
                if row["tokens"] == tokens and row["arm"] == arm
            ]
            points.append((float(tokens - 1), statistics.median(walls)))
        fits[arm] = linear_fit(points)
    delta_alpha = fits["A"]["alpha_ms"] - fits["B"]["alpha_ms"]
    delta_beta = (
        fits["B"]["beta_ms_per_transition"] - fits["A"]["beta_ms_per_transition"]
    )
    crossover = delta_alpha / delta_beta if delta_beta > 0 else None
    return {
        "arms": fits,
        "estimated_crossover_transitions": crossover,
    }


def main() -> None:
    for path in required_manifest_paths():
        if not path.is_file():
            raise SystemExit(f"missing required path {path}")
    base_env, removed = normalized_environment()
    manifest = build_manifest(removed)
    if ARTIFACT.exists():
        raise SystemExit(
            f"packet directory {ARTIFACT} already exists; interruptions are terminal"
        )
    if not ARTIFACT.parent.is_dir():
        raise SystemExit(f"missing packet parent directory {ARTIFACT.parent}")
    ARTIFACT.mkdir()
    manifest_path = ARTIFACT / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")

    canonical_path = ARTIFACT / "rows.jsonl"
    attempts_path = ARTIFACT / "attempts.jsonl"
    pair_attempts_path = ARTIFACT / "pair-attempts.jsonl"
    decision_path = ARTIFACT / "decision.json"
    all_rows = []
    metrics_by_length = {}
    for tokens in LENGTHS:
        length_rows = []
        for pair_index, order in enumerate(PAIR_ORDERS, 1):
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
        output_hashes = {row["stdout_sha256"] for row in length_rows}
        if len(output_hashes) != 1:
            decision = {
                "schema": 1,
                "status": "protocol_failure",
                "failed_length": tokens,
                "reason": "canonical outputs differ across pairs",
                "metrics_by_length": metrics_by_length,
                "fit": None,
            }
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n"
            )
            raise RuntimeError(f"n{tokens} canonical outputs differ across pairs")
        for row in length_rows:
            append_row(canonical_path, row)
        all_rows.extend(length_rows)
        metrics = paired_metrics(length_rows)
        metrics_by_length[str(tokens)] = metrics
        if not length_passes(tokens, metrics):
            decision = {
                "schema": 1,
                "status": "stopped",
                "failed_length": tokens,
                "metrics_by_length": metrics_by_length,
                "fit": None,
            }
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n"
            )
            print(f"stopped after n{tokens}: {metrics}")
            return

    first_saving = metrics_by_length["1"]["median_first_saving_ms"]
    warnings = []
    for tokens in LENGTHS[1:]:
        saving = metrics_by_length[str(tokens)]["median_first_saving_ms"]
        if abs(saving - first_saving) > 200.0:
            warnings.append(
                f"n{tokens} first-byte saving differs from n1 by "
                f"{saving - first_saving:.1f} ms"
            )
    decision = {
        "schema": 1,
        "status": "needs_review" if warnings else "passed_screen",
        "failed_length": None,
        "metrics_by_length": metrics_by_length,
        "fit": fit_arms(all_rows),
        "warnings": warnings,
        "endpoint": "instrumented_cli_spawn_to_first_byte_and_exit",
    }
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
