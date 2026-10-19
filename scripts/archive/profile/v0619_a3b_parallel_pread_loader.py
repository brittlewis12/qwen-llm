#!/usr/bin/env python3

"""Sealed v0.619 adaptation of the v0.602 A3B loader protocol."""

import argparse
from pathlib import Path
import statistics
import subprocess
import time

import v0602_a3b_parallel_copied_loader as protocol


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0619-a3b-parallel-pread-loader-p1"
PREREG = ROOT / "docs/bench/v0619-a3b-parallel-pread-loader.md"
BASE_PROTOCOL = ROOT / "scripts/profile/v0602_a3b_parallel_copied_loader.py"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
MARKER = "[metal-gguf-parallel-pread]"
MARKER_PREFIX = protocol.MARKER_PREFIX.replace(
    "[metal-gguf-parallel-copied]", MARKER, 1
)
CORRECTNESS_TEST = "gguf_parallel_pread_a3b_q4_is_bit_exact"
CORRECTNESS_HARNESS_PREFIX = f"test metal_forward::tests::{CORRECTNESS_TEST} ... "


def required_manifest_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PREREG,
        BASE_PROTOCOL,
        COMMON_RUNNER,
        protocol.MODEL,
        protocol.PROMPT,
        protocol.CLI_BINARY,
        protocol.BENCH_BINARY,
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    tracked = (
        Path(__file__).resolve(),
        PREREG,
        BASE_PROTOCOL,
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
    if device != protocol.EXPECTED_DEVICE:
        raise RuntimeError(f"device boundary drifted: {device!r}")
    if not macos.startswith("15."):
        raise RuntimeError(f"macOS boundary drifted: {macos!r}")
    if hw_memsize != protocol.EXPECTED_HW_MEMSIZE:
        raise RuntimeError(f"memory boundary drifted: {hw_memsize}")
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
        "loaded_runs": protocol.RUNS,
        "cooldown_s": protocol.COOLDOWN_S,
        "host_sample_limit": protocol.HOST_SAMPLE_LIMIT,
        "host_sample_interval_s": protocol.HOST_SAMPLE_INTERVAL_S,
        "child_retry_count": 0,
        "protocol_base": str(BASE_PROTOCOL),
        "candidate": "forced-retained-descriptor-pread",
        "predecessor_v0618_authority": "none-evidence-only",
    }


def arm_environment(arm: str) -> dict[str, str | None]:
    if arm not in ("A", "B"):
        raise ValueError(f"unknown arm {arm!r}")
    return {
        "QWEN_GGUF_PARALLEL_COPY": "pread" if arm == "B" else "0",
        "QWEN_GGUF_OWNED_ARENA": "0",
        "QWEN_GGUF_NO_COPY": "0",
        "QWEN_GGUF_NO_COPY_PREFAULT": None,
        "QWEN_NATIVE_QUANT_EMBED": None,
        "QWEN_MOE_ROUTER_F16": None,
    }


def parse_load_contract(stderr: str, arm: str) -> dict[str, object]:
    expected_markers = 1 if arm == "B" else 0
    if (
        stderr.count(MARKER) != expected_markers
        or stderr.count("[metal-gguf-") != expected_markers
    ):
        raise RuntimeError(f"{arm} pread marker occurrence count drifted")
    if stderr.count("[metal-load] native quantized token embedding policy:") != 1:
        raise RuntimeError(f"{arm} native-policy occurrence count drifted")
    if stderr.count("[metal-load-ledger]") != 1:
        raise RuntimeError(f"{arm} load-ledger occurrence count drifted")
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        MARKER,
        "[metal-load-ledger]",
    )
    recognized = [line for line in stderr.splitlines() if line.startswith(prefixes)]
    if arm == "A":
        if recognized != [protocol.POLICY_LINE, protocol.LEDGER_LINE]:
            raise RuntimeError(f"A load contract drifted: {recognized!r}")
        return {"storage": "copied", "marker": None}
    if len(recognized) != 3 or recognized[0] != protocol.POLICY_LINE:
        raise RuntimeError(f"B load-line count or policy drifted: {recognized!r}")
    if recognized[2] != protocol.LEDGER_LINE:
        raise RuntimeError("B copied ledger drifted")
    timings = protocol.parse_marker(recognized[1])
    return {
        "storage": "parallel-pread",
        "marker": recognized[1],
        "phase_us": timings,
    }


def parse_observed_load_contract(stderr: str, arm: str) -> dict[str, object] | None:
    if stderr.count("[metal-gguf-") != stderr.count(MARKER):
        raise RuntimeError("observed unrequested storage marker")
    tokens = (
        "[metal-load] native quantized token embedding policy:",
        MARKER,
        "[metal-load-ledger]",
    )
    occurrence_count = sum(stderr.count(token) for token in tokens)
    if occurrence_count == 0:
        return None
    recognized = [line for line in stderr.splitlines() if line.startswith(tokens)]
    if occurrence_count != len(recognized):
        raise RuntimeError("observed load token is not an exact line prefix")
    expected = ("policy", "ledger") if arm == "A" else ("policy", "marker", "ledger")
    marker_timings = None
    for line, kind in zip(recognized, expected, strict=False):
        if kind == "policy" and line != protocol.POLICY_LINE:
            raise RuntimeError("observed native policy contradicts contract")
        if kind == "ledger" and line != protocol.LEDGER_LINE:
            raise RuntimeError("observed copied ledger contradicts contract")
        if kind == "marker":
            marker_timings = protocol.parse_marker(line)
    if len(recognized) > len(expected):
        raise RuntimeError("observed load contract has extra lines")
    if len(recognized) < len(expected):
        return {
            "status": "incomplete-valid-prefix",
            "recognized_lines": recognized,
            "marker_phase_us": marker_timings,
        }
    return {"status": "complete", "contract": parse_load_contract(stderr, arm)}


def extract_correctness_load_lines(text: str) -> list[str]:
    tokens = (
        "[metal-load] native quantized token embedding policy:",
        MARKER,
        "[metal-load-ledger]",
    )
    recognized = []
    for line in text.splitlines():
        if not recognized and line == CORRECTNESS_HARNESS_PREFIX + protocol.POLICY_LINE:
            recognized.append(protocol.POLICY_LINE)
            continue
        matching = [token for token in tokens if line.startswith(token)]
        if len(matching) == 1:
            recognized.append(line)
            continue
        if "[metal-load" in line or MARKER in line:
            raise RuntimeError(f"malformed correctness load line: {line!r}")
    return recognized


def validate_correctness_load_text(text: str) -> tuple[list[str], dict[str, int]]:
    if text.count(MARKER) != 1 or text.count("[metal-gguf-") != 1:
        raise RuntimeError("correctness candidate marker occurrence count drifted")
    if text.count("[metal-load] native quantized token embedding policy:") != 2:
        raise RuntimeError("correctness native-policy occurrence count drifted")
    if text.count("[metal-load-ledger]") != 2:
        raise RuntimeError("correctness ledger occurrence count drifted")
    recognized = extract_correctness_load_lines(text)
    if len(recognized) != 5:
        raise RuntimeError(f"correctness load-line count drifted: {recognized!r}")
    if recognized[:3] != [
        protocol.POLICY_LINE,
        protocol.LEDGER_LINE,
        protocol.POLICY_LINE,
    ]:
        raise RuntimeError("correctness A/B policy or A ledger ordering drifted")
    if recognized[4] != protocol.LEDGER_LINE:
        raise RuntimeError("correctness B ledger drifted")
    return recognized, protocol.parse_marker(recognized[3])


def run_correctness(base_env: dict[str, str]) -> dict[str, object]:
    output_path = ARTIFACT / "correctness.out"
    command = [
        "cargo",
        "test",
        "--release",
        "-p",
        "qwen-llm",
        "--lib",
        f"metal_forward::tests::{CORRECTNESS_TEST}",
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]
    started = time.perf_counter()
    with output_path.open("xb") as output:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=base_env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_ms = (time.perf_counter() - started) * 1e3
    text = output_path.read_text(encoding="utf-8")
    if result.returncode != 0 or "test result: ok. 1 passed;" not in text:
        raise RuntimeError("release full-state correctness gate failed")
    recognized, marker = validate_correctness_load_text(text)
    return {
        "command": command,
        "wall_ms": wall_ms,
        "output_path": str(output_path),
        "output_sha256": protocol.common.sha256_file(output_path),
        "recognized_load_lines": recognized,
        "candidate_phase_us": marker,
        "passed": True,
    }


def analyze_fresh(rows: list[dict[str, object]]) -> dict[str, object]:
    pairs = protocol.paired_rows(rows, "fresh-128")
    output_hashes = {row["stdout_sha256"] for pair in pairs for row in pair}
    if len(output_hashes) != 1:
        raise RuntimeError("fresh generated output identity differs")
    metrics = []
    for pair_index, (a, b) in enumerate(pairs, 1):
        timing_a = a["timing"]
        timing_b = b["timing"]
        load_a = timing_a["runtime_and_model_load_ms"]
        load_b = timing_b["runtime_and_model_load_ms"]
        metrics.append(
            {
                "pair_index": pair_index,
                "pair_order": protocol.PAIR_ORDERS[pair_index - 1],
                "load_saving_ms": protocol.finite_difference(load_a, load_b, "fresh.L"),
                "load_b_over_a": protocol.positive_ratio(load_b, load_a, "fresh.Q"),
                "first_byte_saving_ms": protocol.finite_difference(
                    a["spawn_to_first_byte_ms"], b["spawn_to_first_byte_ms"], "fresh.F"
                ),
                "exit_saving_ms": protocol.finite_difference(
                    a["spawn_to_exit_ms"], b["spawn_to_exit_ms"], "fresh.E"
                ),
                "prefill_b_over_a": protocol.positive_ratio(
                    timing_b["prefill_ms"], timing_a["prefill_ms"], "fresh.prefill"
                ),
                "ttft_b_over_a": protocol.positive_ratio(
                    timing_b["ttft_ms"], timing_a["ttft_ms"], "fresh.ttft"
                ),
                "rss_b_over_a": protocol.positive_ratio(
                    b["process_resources"]["maximum_resident_set_size"],
                    a["process_resources"]["maximum_resident_set_size"],
                    "fresh.rss",
                ),
                "footprint_b_over_a": protocol.positive_ratio(
                    b["process_resources"]["peak_memory_footprint"],
                    a["process_resources"]["peak_memory_footprint"],
                    "fresh.footprint",
                ),
            }
        )
    for row in metrics:
        row["load_b_wins"] = row["load_saving_ms"] > 0
        row["first_byte_b_wins"] = row["first_byte_saving_ms"] > 0
        row["exit_b_wins"] = row["exit_saving_ms"] > 0

    def values(name: str, order: str | None = None) -> list[float]:
        selected = [
            row[name] for row in metrics if order is None or row["pair_order"] == order
        ]
        expected = 6 if order is None else 3
        if len(selected) != expected:
            raise RuntimeError(f"fresh {name} {order} membership drifted")
        return selected

    def wins(name: str, order: str | None = None) -> int:
        return sum(
            bool(row[name])
            for row in metrics
            if order is None or row["pair_order"] == order
        )

    load = values("load_saving_ms")
    load_ab = values("load_saving_ms", "AB")
    load_ba = values("load_saving_ms", "BA")
    load_ratio = values("load_b_over_a")
    load_ratio_ab = values("load_b_over_a", "AB")
    load_ratio_ba = values("load_b_over_a", "BA")
    first = values("first_byte_saving_ms")
    first_ab = values("first_byte_saving_ms", "AB")
    first_ba = values("first_byte_saving_ms", "BA")
    exit_values = values("exit_saving_ms")
    exit_ab = values("exit_saving_ms", "AB")
    exit_ba = values("exit_saving_ms", "BA")
    prefill = values("prefill_b_over_a")
    ttft = values("ttft_b_over_a")
    rss = values("rss_b_over_a")
    footprint = values("footprint_b_over_a")
    load_wins = wins("load_b_wins")
    load_ab_wins = wins("load_b_wins", "AB")
    load_ba_wins = wins("load_b_wins", "BA")
    first_wins = wins("first_byte_b_wins")
    first_ab_wins = wins("first_byte_b_wins", "AB")
    first_ba_wins = wins("first_byte_b_wins", "BA")
    exit_wins = wins("exit_b_wins")
    exit_ab_wins = wins("exit_b_wins", "AB")
    exit_ba_wins = wins("exit_b_wins", "BA")
    gates = {
        "load_saving_median": statistics.median(load) >= 112.0,
        "load_saving_ab": statistics.median(load_ab) >= 112.0,
        "load_saving_ba": statistics.median(load_ba) >= 112.0,
        "load_ratio_median": statistics.median(load_ratio) <= 0.85,
        "load_wins": load_wins >= 5,
        "load_ab_wins": load_ab_wins >= 2,
        "load_ba_wins": load_ba_wins >= 2,
        "first_byte_saving_median": statistics.median(first) >= 112.0,
        "first_byte_saving_ab": statistics.median(first_ab) >= 112.0,
        "first_byte_saving_ba": statistics.median(first_ba) >= 112.0,
        "first_byte_wins": first_wins >= 5,
        "first_byte_ab_wins": first_ab_wins >= 2,
        "first_byte_ba_wins": first_ba_wins >= 2,
        "exit_saving_median": statistics.median(exit_values) >= 112.0,
        "exit_saving_ab": statistics.median(exit_ab) >= 112.0,
        "exit_saving_ba": statistics.median(exit_ba) >= 112.0,
        "exit_wins": exit_wins >= 5,
        "exit_ab_wins": exit_ab_wins >= 2,
        "exit_ba_wins": exit_ba_wins >= 2,
        "rss_ratio": max(rss) <= 1.05,
        "footprint_ratio": max(footprint) <= 1.05,
    }
    return {
        "stage": "fresh-128",
        "global_output_sha256": next(iter(output_hashes)),
        "pairs": metrics,
        "load_saving_ms": {
            "values": load,
            "ab": load_ab,
            "ba": load_ba,
            "median": statistics.median(load),
            "ab_median": statistics.median(load_ab),
            "ba_median": statistics.median(load_ba),
            "wins": load_wins,
            "ab_wins": load_ab_wins,
            "ba_wins": load_ba_wins,
        },
        "load_b_over_a": {
            "values": load_ratio,
            "ab": load_ratio_ab,
            "ba": load_ratio_ba,
            "median": statistics.median(load_ratio),
            "ab_median": statistics.median(load_ratio_ab),
            "ba_median": statistics.median(load_ratio_ba),
        },
        "first_byte_saving_ms": {
            "values": first,
            "ab": first_ab,
            "ba": first_ba,
            "median": statistics.median(first),
            "ab_median": statistics.median(first_ab),
            "ba_median": statistics.median(first_ba),
            "wins": first_wins,
            "ab_wins": first_ab_wins,
            "ba_wins": first_ba_wins,
        },
        "exit_saving_ms": {
            "values": exit_values,
            "ab": exit_ab,
            "ba": exit_ba,
            "median": statistics.median(exit_values),
            "ab_median": statistics.median(exit_ab),
            "ba_median": statistics.median(exit_ba),
            "wins": exit_wins,
            "ab_wins": exit_ab_wins,
            "ba_wins": exit_ba_wins,
        },
        "fresh_diagnostics": {
            "prefill_b_over_a": {
                "values": prefill,
                "median": statistics.median(prefill),
            },
            "ttft_b_over_a": {
                "values": ttft,
                "median": statistics.median(ttft),
            },
        },
        "rss_b_over_a": {
            "values": rss,
            "maximum": max(rss),
        },
        "footprint_b_over_a": {
            "values": footprint,
            "maximum": max(footprint),
        },
        "gates": gates,
        "passes": all(gates.values()),
    }


def configure_protocol() -> None:
    protocol.ARTIFACT = ARTIFACT
    protocol.PREREG = PREREG
    protocol.MARKER_PREFIX = MARKER_PREFIX
    protocol.CORRECTNESS_HARNESS_PREFIX = CORRECTNESS_HARNESS_PREFIX
    protocol.required_manifest_paths = required_manifest_paths
    protocol.source_and_build_identity = source_and_build_identity
    protocol.build_manifest = build_manifest
    protocol.arm_environment = arm_environment
    protocol.parse_load_contract = parse_load_contract
    protocol.parse_observed_load_contract = parse_observed_load_contract
    protocol.extract_correctness_load_lines = extract_correctness_load_lines
    protocol.validate_correctness_load_text = validate_correctness_load_text
    protocol.run_correctness = run_correctness
    protocol.analyze_fresh = analyze_fresh
    original_write = protocol.write_identity_checked_decision

    def write_decision(
        decision: dict[str, object], manifest: dict[str, object]
    ) -> None:
        inherited_authority = "force-only-exact-a3b"
        if decision.get("status") == "go":
            stages = decision.get("stages")
            correctness = decision.get("correctness")
            valid_go = (
                decision.get("stopped_after") == "fresh-128"
                and decision.get("authority") == inherited_authority
                and isinstance(correctness, dict)
                and correctness.get("passed") is True
                and isinstance(stages, dict)
                and isinstance(stages.get("loaded"), dict)
                and stages["loaded"].get("passes") is True
                and isinstance(stages.get("fresh_128"), dict)
                and stages["fresh_128"].get("passes") is True
            )
            if not valid_go:
                raise RuntimeError("GO authority conjunction is incomplete")
            decision = {**decision, "authority": "force-only-exact-a3b-pread"}
        elif decision.get("authority") == inherited_authority:
            raise RuntimeError("non-GO decision carries inherited authority")
        original_write(decision, manifest)

    protocol.write_identity_checked_decision = write_decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    configure_protocol()
    protocol.main(preflight_only=arguments.preflight_only)
