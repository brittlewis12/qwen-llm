#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


PAIRED_SCHEDULE = [
    [("A", "X"), ("B", "Y")],
    [("B", "X"), ("A", "Y")],
    [("B", "Y"), ("A", "X")],
    [("A", "Y"), ("B", "X")],
    [("A", "X"), ("B", "Y")],
    [("B", "X"), ("A", "Y")],
]
SINGLE_BANKS = ["X", "Y", "Y", "X", "X", "Y"]
THREADS = 256
LAYERS = 16
N_KV = 4
HEAD_DIM = 256
KV_DIM = N_KV * HEAD_DIM
SCREEN_TEST = "metal::tests::attn_matrix_vt_rebuild_screen"
PROBE_TEST = "metal::tests::attn_matrix_vt_environment_probe"
CORRECTNESS_TESTS = [
    "metal::tests::attn_matrix_vt_dispatch_groups_cover_exact_thread_range",
    "metal::tests::attn_matrix_vt_compact_dispatch_matches_legacy_nonzero_span",
    "metal::tests::attn_matrix_vt_prefix_rebuild_preserves_scattered_suffix",
]
SYSTEM_COMMANDS = {
    "sw_vers": ["sw_vers"],
    "uname": ["uname", "-a"],
    "physical_memory": ["sysctl", "-n", "hw.memsize"],
    "power": ["pmset", "-g", "batt"],
    "memory_pressure": ["memory_pressure"],
    "thermal": ["pmset", "-g", "therm"],
}
BUILD_COMMAND = [
    "cargo",
    "test",
    "-p",
    "qwen-llm",
    "--release",
    "--lib",
    "--no-run",
    "--message-format=json",
]
FAILURE_PATTERN = re.compile(r"V_T command failed arm=(D0|D1|D2|PREP_D2)")
META_KEYS = {
    "kind",
    "schema_version",
    "mode",
    "prefix",
    "chunk",
    "n_pos",
    "vt_stride",
    "layers",
    "n_kv",
    "head_dim",
    "kv_dim",
    "device_registry_id",
    "device",
    "max_buffer_length",
    "recommended_max_working_set_size",
    "bytes_per_layer_buffer",
    "total_requested_bytes",
    "allocated_before",
    "allocated_after",
}
ARM_COMMON_KEYS = {
    "kind",
    "schema_version",
    "mode",
    "prefix",
    "chunk",
    "role",
    "arm",
    "bank",
    "base_pos",
    "rows",
    "threadgroups_per_layer",
    "thread_slots_per_layer",
    "logical_bytes",
    "wall_ms",
    "gpu_ms",
    "gb_s",
}


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median of empty values")
    return float(statistics.median(values))


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def close(a: float, b: float, *, rel: float = 1e-9, abs_: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def stream_bytes(packet: dict[str, Any]) -> bytes:
    require(set(packet) == {"base64", "sha256", "length"}, "stream schema drift")
    data = base64.b64decode(packet["base64"], validate=True)
    require(len(data) == packet["length"], "stream length drift")
    require(hashlib.sha256(data).hexdigest() == packet["sha256"], "stream hash drift")
    return data


def execution_text(packet: dict[str, Any]) -> str:
    return (
        stream_bytes(packet["stdout"]) + b"\n" + stream_bytes(packet["stderr"])
    ).decode("utf-8", errors="strict")


def validate_execution(
    packet: dict[str, Any],
    expected_command: list[str],
    *,
    expected_overrides: dict[str, str] | None = None,
    expected_removed: list[str] | None = None,
) -> None:
    expected_keys = {
        "command",
        "environment_overrides",
        "environment_removed",
        "pid",
        "started_at",
        "finished_at",
        "elapsed_s",
        "returncode",
        "termination_signal",
        "spawn_callback_error",
        "stdout",
        "stderr",
    }
    require(set(packet) == expected_keys, "execution schema drift")
    require(packet["command"] == expected_command, "execution command drift")
    require(
        packet["environment_overrides"] == (expected_overrides or {}),
        "execution environment override drift",
    )
    require(
        packet["environment_removed"] == (expected_removed or []),
        "execution environment removal drift",
    )
    require(packet["spawn_callback_error"] is None, "spawn persistence failed")
    require(isinstance(packet["pid"], int) and packet["pid"] > 0, "invalid child PID")
    require(
        math.isfinite(packet["elapsed_s"]) and packet["elapsed_s"] >= 0,
        "invalid elapsed time",
    )
    require(isinstance(packet["returncode"], int), "invalid return code")
    expected_signal = -packet["returncode"] if packet["returncode"] < 0 else None
    require(packet["termination_signal"] == expected_signal, "termination signal drift")
    stream_bytes(packet["stdout"])
    stream_bytes(packet["stderr"])


def validate_exact_test(
    packet: dict[str, Any],
    binary: str,
    name: str,
    ignored: bool,
    ambient_qwen: list[str],
    require_metal: bool = False,
) -> None:
    command = [binary, name]
    if ignored:
        command.append("--ignored")
    command.extend(["--exact", "--nocapture", "--test-threads=1"])
    overrides = {"QWEN_REQUIRE_METAL_TESTS": "1"} if require_metal else {}
    validate_execution(
        packet,
        command,
        expected_overrides=overrides,
        expected_removed=ambient_qwen,
    )
    require(packet["returncode"] == 0, f"exact test failed: {name}")
    text = execution_text(packet)
    require(f"test {name} ..." in text, f"named test missing: {name}")
    require("running 1 test" in text, f"test cardinality missing: {name}")
    require("1 passed; 0 failed" in text, f"test result drift: {name}")


def expected_arm(
    mode: str, role: str, prefix: int, chunk: int
) -> tuple[str, bool, int]:
    n_pos = prefix + chunk
    mapping = {
        ("dispatch", "A"): ("D0", False, n_pos),
        ("dispatch", "B"): ("D1", True, n_pos),
        ("overlap", "A"): ("D1", True, n_pos),
        ("overlap", "B"): ("D2", True, prefix),
        ("compact", "S"): ("D1", True, n_pos),
    }
    try:
        return mapping[(mode, role)]
    except KeyError as error:
        raise ValueError(f"invalid mode/role {mode}/{role}") from error


def validate_record(
    record: dict[str, Any], mode: str, prefix: int, chunk: int, role: str, bank: str
) -> None:
    expected_keys = ARM_COMMON_KEYS | (
        {"sample"} if mode == "compact" else {"pair", "order", "sequence"}
    )
    require(set(record) == expected_keys, "arm record schema drift")
    require(record["kind"] == "arm", "record kind is not arm")
    require(record["schema_version"] == 2, "arm schema drift")
    require(record["mode"] == mode, "arm mode drift")
    require(record["prefix"] == prefix, "arm prefix drift")
    require(record["chunk"] == chunk, "arm chunk drift")
    require(record["role"] == role, "arm role drift")
    require(record["bank"] == bank, "arm bank drift")
    arm, compact, rows = expected_arm(mode, role, prefix, chunk)
    require(record["arm"] == arm, "arm variant drift")
    require(record["base_pos"] == 0, "measured base_pos drift")
    require(record["rows"] == rows, "arm row count drift")
    total = KV_DIM * rows
    threadgroups = math.ceil(total / THREADS) if compact else total
    require(record["threadgroups_per_layer"] == threadgroups, "threadgroup count drift")
    require(
        record["thread_slots_per_layer"] == threadgroups * THREADS,
        "thread-slot count drift",
    )
    logical_bytes = 4 * LAYERS * KV_DIM * rows
    require(record["logical_bytes"] == logical_bytes, "logical-byte drift")
    wall_ms = float(record["wall_ms"])
    gpu_ms = float(record["gpu_ms"])
    gb_s = float(record["gb_s"])
    require(math.isfinite(wall_ms) and wall_ms > 0, "invalid wall time")
    require(math.isfinite(gpu_ms) and gpu_ms > 0, "invalid GPU time")
    require(math.isfinite(gb_s) and gb_s > 0, "invalid GB/s")
    require(close(gb_s, logical_bytes / (gpu_ms * 1e6)), "GB/s arithmetic drift")


def validate_probe(
    packet: dict[str, Any], binary: str, ambient_qwen: list[str]
) -> dict[str, Any]:
    validate_exact_test(packet, binary, PROBE_TEST, True, ambient_qwen)
    records = []
    stdout = stream_bytes(packet["stdout"]).decode("utf-8", errors="strict")
    stderr = stream_bytes(packet["stderr"]).decode("utf-8", errors="strict")
    require("VT_ENV_JSON " not in stderr, "probe marker appeared on stderr")
    for line in stdout.splitlines():
        if "VT_ENV_JSON " in line:
            records.append(json.loads(line.split("VT_ENV_JSON ", 1)[1]))
    require(len(records) == 1, "probe record count drift")
    record = records[0]
    require(
        set(record)
        == {
            "schema_version",
            "test",
            "device_registry_id",
            "device",
            "max_buffer_length",
            "recommended_max_working_set_size",
        },
        "probe schema drift",
    )
    require(
        record["schema_version"] == 1 and record["test"] == PROBE_TEST,
        "probe identity drift",
    )
    require(record["max_buffer_length"] > 0, "invalid maxBufferLength")
    require(record["recommended_max_working_set_size"] > 0, "invalid working-set size")
    return record


def validate_environment(
    snapshot: dict[str, Any],
    expected_identity: dict[str, Any],
    binary: str,
    ambient_qwen: list[str],
) -> dict[str, Any]:
    require(
        set(snapshot)
        == {"captured_at", "identity", "system", "protected_pid", "metal_probe"},
        "environment schema drift",
    )
    require(snapshot["identity"] == expected_identity, "environment identity drift")
    require(set(snapshot["system"]) == set(SYSTEM_COMMANDS), "system command set drift")
    for name, command in SYSTEM_COMMANDS.items():
        packet = snapshot["system"][name]
        validate_execution(packet, command)
        require(packet["returncode"] == 0, f"environment command failed: {name}")
    protected = snapshot["protected_pid"]
    require("active" in protected, "protected PID state missing")
    protected_execution = {
        key: value for key, value in protected.items() if key != "active"
    }
    validate_execution(
        protected_execution, ["ps", "-p", "8770", "-o", "pid=,state=,etime=,command="]
    )
    require(protected.get("active") is False, "protected PID was active")
    require(protected["returncode"] == 1, "protected PID absence status drift")
    return validate_probe(snapshot["metal_probe"], binary, ambient_qwen)


def validate_child(
    child: dict[str, Any], expected_identity: dict[str, Any], ambient_qwen: list[str]
) -> dict[str, Any]:
    expected_child_keys = {
        "mode",
        "prefix",
        "chunk",
        "state",
        "command",
        "identity_before",
        "pid",
        "spawned_at",
        "execution",
        "identity_after",
        "failure_arm",
        "records",
    }
    require(set(child) == expected_child_keys, "child schema drift")
    require(child["state"] == "parsed", "child was not parsed")
    require(child["identity_before"] == expected_identity, "child pre-identity drift")
    require(child["identity_after"] == expected_identity, "child post-identity drift")
    require(child["failure_arm"] is None, "successful child has failure arm")
    binary = expected_identity["binary_path"]
    expected_command = [
        binary,
        SCREEN_TEST,
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]
    require(child["command"] == expected_command, "child command drift")
    validate_execution(
        child["execution"],
        expected_command,
        expected_overrides={
            "QWEN_VT_REBUILD_MODE": child["mode"],
            "QWEN_VT_REBUILD_PREFIX": str(child["prefix"]),
            "QWEN_VT_REBUILD_CHUNK": str(child["chunk"]),
        },
        expected_removed=ambient_qwen,
    )
    require(child["execution"]["pid"] == child["pid"], "child PID drift")
    require(child["execution"]["returncode"] == 0, "child process failed")
    text = execution_text(child["execution"])
    require(f"test {SCREEN_TEST} ..." in text, "screen test identity missing")
    require(
        "running 1 test" in text and "1 passed; 0 failed" in text, "screen count drift"
    )
    stdout = stream_bytes(child["execution"]["stdout"]).decode("utf-8", errors="strict")
    stderr = stream_bytes(child["execution"]["stderr"]).decode("utf-8", errors="strict")
    require("VT_REBUILD_JSON " not in stderr, "benchmark marker appeared on stderr")
    raw_records = [
        json.loads(line.split("VT_REBUILD_JSON ", 1)[1])
        for line in stdout.splitlines()
        if "VT_REBUILD_JSON " in line
    ]
    require(
        raw_records == child["records"],
        "parsed records drift from authenticated output",
    )
    mode = child["mode"]
    prefix = child["prefix"]
    chunk = child["chunk"]
    require(mode in {"dispatch", "compact", "overlap"}, "invalid child mode")
    require(
        isinstance(prefix, int) and isinstance(chunk, int), "invalid child geometry"
    )
    records = child["records"]
    expected_arm_count = 6 if mode == "compact" else 12
    require(len(records) == expected_arm_count + 1, "raw record cardinality drift")
    require(records[0].get("kind") == "meta", "metadata must be first")
    require(
        all(record.get("kind") == "arm" for record in records[1:]),
        "unknown/interleaved raw record",
    )
    meta = records[0]
    arms = records[1:]
    require(set(meta) == META_KEYS, "metadata record schema drift")
    require(meta["schema_version"] == 2, "metadata schema drift")
    require(meta["mode"] == mode, "metadata mode drift")
    require(
        meta["prefix"] == prefix and meta["chunk"] == chunk, "metadata geometry drift"
    )
    require(meta["n_pos"] == prefix + chunk, "metadata n_pos drift")
    require(meta["vt_stride"] == prefix + chunk, "metadata vt_stride drift")
    require(
        meta["layers"] == LAYERS and meta["n_kv"] == N_KV, "metadata topology drift"
    )
    require(
        meta["head_dim"] == HEAD_DIM and meta["kv_dim"] == KV_DIM,
        "metadata dimension drift",
    )
    bytes_per_buffer = 2 * (prefix + chunk) * KV_DIM
    require(meta["bytes_per_layer_buffer"] == bytes_per_buffer, "buffer-byte drift")
    require(
        meta["total_requested_bytes"] == bytes_per_buffer * LAYERS * 4,
        "allocation-byte drift",
    )
    require(
        meta["max_buffer_length"] >= bytes_per_buffer, "buffer exceeds device limit"
    )
    require(meta["recommended_max_working_set_size"] > 0, "missing working-set limit")
    require(
        meta["allocated_after"] >= meta["allocated_before"],
        "allocation accounting reversed",
    )

    if mode == "compact":
        for index, (record, bank) in enumerate(
            zip(arms, SINGLE_BANKS, strict=True), start=1
        ):
            require(record["sample"] == index, "compact sample order drift")
            validate_record(record, mode, prefix, chunk, "S", bank)
    else:
        cursor = 0
        for pair_index, pair in enumerate(PAIRED_SCHEDULE, start=1):
            expected_order = "".join(role for role, _ in pair)
            for sequence, (role, bank) in enumerate(pair, start=1):
                record = arms[cursor]
                cursor += 1
                require(record["pair"] == pair_index, "pair index drift")
                require(record["order"] == expected_order, "pair order drift")
                require(record["sequence"] == sequence, "pair sequence drift")
                validate_record(record, mode, prefix, chunk, role, bank)
    return {"meta": meta, "arms": arms}


def paired_summary(child: dict[str, Any], validated: dict[str, Any]) -> dict[str, Any]:
    arms = validated["arms"]
    pairs: list[dict[str, Any]] = []
    for pair_index in range(1, 7):
        pair_arms = [record for record in arms if record["pair"] == pair_index]
        require(len(pair_arms) == 2, "pair cardinality drift")
        by_role = {record["role"]: record for record in pair_arms}
        require(set(by_role) == {"A", "B"}, "pair role drift")
        a = by_role["A"]
        b = by_role["B"]
        pairs.append(
            {
                "pair": pair_index,
                "order": a["order"],
                "gpu_saving_ms": a["gpu_ms"] - b["gpu_ms"],
                "wall_saving_ms": a["wall_ms"] - b["wall_ms"],
                "gpu_win": a["gpu_ms"] > b["gpu_ms"],
                "wall_win": a["wall_ms"] > b["wall_ms"],
            }
        )
    a_gpu = [record["gpu_ms"] for record in arms if record["role"] == "A"]
    b_gpu = [record["gpu_ms"] for record in arms if record["role"] == "B"]
    a_wall = [record["wall_ms"] for record in arms if record["role"] == "A"]
    b_wall = [record["wall_ms"] for record in arms if record["role"] == "B"]
    strata: dict[str, Any] = {}
    for order in ["AB", "BA"]:
        order_pairs = [pair for pair in pairs if pair["order"] == order]
        order_arms = [record for record in arms if record["order"] == order]
        strata[order] = {
            "gpu_saving_median_ms": median(
                [pair["gpu_saving_ms"] for pair in order_pairs]
            ),
            "wall_saving_median_ms": median(
                [pair["wall_saving_ms"] for pair in order_pairs]
            ),
            "gpu_speedup": median(
                [record["gpu_ms"] for record in order_arms if record["role"] == "A"]
            )
            / median(
                [record["gpu_ms"] for record in order_arms if record["role"] == "B"]
            ),
        }
    return {
        "mode": child["mode"],
        "prefix": child["prefix"],
        "chunk": child["chunk"],
        "arm_a": expected_arm(child["mode"], "A", child["prefix"], child["chunk"])[0],
        "arm_b": expected_arm(child["mode"], "B", child["prefix"], child["chunk"])[0],
        "a_gpu_median_ms": median(a_gpu),
        "b_gpu_median_ms": median(b_gpu),
        "a_wall_median_ms": median(a_wall),
        "b_wall_median_ms": median(b_wall),
        "gpu_speedup": median(a_gpu) / median(b_gpu),
        "wall_speedup": median(a_wall) / median(b_wall),
        "gpu_saving_median_ms": median([pair["gpu_saving_ms"] for pair in pairs]),
        "wall_saving_median_ms": median([pair["wall_saving_ms"] for pair in pairs]),
        "gpu_wins": sum(pair["gpu_win"] for pair in pairs),
        "wall_wins": sum(pair["wall_win"] for pair in pairs),
        "strata": strata,
        "pairs": pairs,
    }


def compact_summary(child: dict[str, Any], validated: dict[str, Any]) -> dict[str, Any]:
    arms = validated["arms"]
    return {
        "mode": "compact",
        "prefix": child["prefix"],
        "chunk": child["chunk"],
        "gpu_median_ms": median([record["gpu_ms"] for record in arms]),
        "wall_median_ms": median([record["wall_ms"] for record in arms]),
        "gb_s_median": median([record["gb_s"] for record in arms]),
        "samples": arms,
    }


def validate_packet_identity(
    packet: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ambient_qwen = packet["ambient_qwen_environment"]
    require(
        isinstance(ambient_qwen, list)
        and ambient_qwen == sorted(set(ambient_qwen))
        and all(
            isinstance(name, str) and name.startswith("QWEN_") for name in ambient_qwen
        ),
        "ambient QWEN environment drift",
    )
    require(
        packet["tracked_tree_clean"] is True, "campaign did not start tracked-clean"
    )
    expected = packet["expected_identity"]
    require(expected["tracked_clean"] is True, "expected identity was dirty")
    require(
        packet["build_identity"]
        == {"head": expected["head"], "tree": expected["tree"]},
        "build identity drift",
    )
    require(packet["binary_path"] == expected["binary_path"], "binary path drift")
    require(packet["binary_sha256"] == expected["binary_sha256"], "binary hash drift")
    validate_execution(packet["build"], BUILD_COMMAND)
    require(packet["build"]["returncode"] == 0, "build command failed")
    binary = expected["binary_path"]
    correctness = packet["correctness"]
    require(len(correctness) == len(CORRECTNESS_TESTS), "correctness test count drift")
    for test_packet, name in zip(correctness, CORRECTNESS_TESTS, strict=True):
        validate_exact_test(test_packet, binary, name, False, ambient_qwen, True)
    before_probe = validate_environment(
        packet["environment_before"], expected, binary, ambient_qwen
    )
    after_probe = validate_environment(
        packet["environment_after"], expected, binary, ambient_qwen
    )
    require(before_probe == after_probe, "Metal probe drift")
    return expected, before_probe


def validate_safety(safety: dict[str, Any], p2048_validated: dict[str, Any]) -> bool:
    p2048_d0_wall = [
        record["wall_ms"] for record in p2048_validated["arms"] if record["arm"] == "D0"
    ]
    computed = max(p2048_d0_wall) <= 1000.0 and 4.0 * median(p2048_d0_wall) <= 2000.0
    require(
        set(safety)
        == {
            "p2048_d0_wall_ms",
            "p2048_d0_wall_median_ms",
            "p2048_d0_wall_max_ms",
            "run_dispatch_p8192",
        },
        "safety schema drift",
    )
    require(safety["p2048_d0_wall_ms"] == p2048_d0_wall, "safety raw values drift")
    require(
        close(safety["p2048_d0_wall_median_ms"], median(p2048_d0_wall)),
        "safety median drift",
    )
    require(
        close(safety["p2048_d0_wall_max_ms"], max(p2048_d0_wall)), "safety max drift"
    )
    require(safety["run_dispatch_p8192"] == computed, "P8192 safety decision drift")
    return computed


def analyze_d1_cells(
    cell_summaries: list[dict[str, Any]], validated_children: list[dict[str, Any]]
) -> dict[str, Any]:
    require(
        len(cell_summaries) >= 5 and len(validated_children) >= 5,
        "D1 evidence incomplete",
    )
    paired_dispatch = [
        summary for summary in cell_summaries[:5] if summary["mode"] == "dispatch"
    ]
    largest_dispatch = max(paired_dispatch, key=lambda summary: summary["prefix"])
    p8192_d1 = [
        record for record in validated_children[2]["arms"] if record["arm"] == "D1"
    ]
    require(len(p8192_d1) == 6, "P8192 D1 sample count drift")
    compact_16 = cell_summaries[3]
    compact_32 = cell_summaries[4]
    d1_gpu_medians = [
        median([record["gpu_ms"] for record in p8192_d1]),
        compact_16["gpu_median_ms"],
        compact_32["gpu_median_ms"],
    ]
    d1_rate_16 = compact_16["gb_s_median"]
    d1_rate_32 = compact_32["gb_s_median"]
    gates = {
        "all_six_gpu_wins": largest_dispatch["gpu_wins"] == 6,
        "ab_gpu_saving_positive": largest_dispatch["strata"]["AB"][
            "gpu_saving_median_ms"
        ]
        > 0,
        "ba_gpu_saving_positive": largest_dispatch["strata"]["BA"][
            "gpu_saving_median_ms"
        ]
        > 0,
        "overall_gpu_speedup_at_least_8x": largest_dispatch["gpu_speedup"] >= 8.0,
        "ab_gpu_speedup_at_least_8x": largest_dispatch["strata"]["AB"]["gpu_speedup"]
        >= 8.0,
        "ba_gpu_speedup_at_least_8x": largest_dispatch["strata"]["BA"]["gpu_speedup"]
        >= 8.0,
        "d1_gpu_time_nondecreasing_8k_32k": d1_gpu_medians[0]
        <= d1_gpu_medians[1]
        <= d1_gpu_medians[2],
        "d1_16k_rate_credible": 50.0 <= d1_rate_16 <= 800.0,
        "d1_32k_rate_credible": 50.0 <= d1_rate_32 <= 800.0,
    }
    return {
        "largest_dispatch": largest_dispatch,
        "scaling": {
            "prefixes": [8192, 16384, 32768],
            "gpu_medians_ms": d1_gpu_medians,
            "gb_s_medians_16k_32k": [d1_rate_16, d1_rate_32],
        },
        "gates": gates,
        "disposition": "ADVANCE_PRODUCT_AB" if all(gates.values()) else "KILL",
    }


def analyze_packet(packet: dict[str, Any]) -> dict[str, Any]:
    require(
        set(packet)
        == {
            "schema_version",
            "started_at",
            "finished_at",
            "build_identity",
            "binary_path",
            "binary_sha256",
            "tracked_tree_clean",
            "ambient_qwen_environment",
            "expected_identity",
            "build",
            "correctness",
            "environment_before",
            "environment_after",
            "children",
            "safety",
        },
        "chronology top-level schema drift",
    )
    require(packet.get("schema_version") == 2, "chronology schema drift")
    require(
        "failure" not in packet and "postflight_failure" not in packet,
        "failed chronology cannot pass",
    )
    expected_identity, probe = validate_packet_identity(packet)
    ambient_qwen = packet["ambient_qwen_environment"]
    children = packet["children"]
    require(len(children) == 6, "campaign must contain six fresh-process cells")
    expected_prefixes = [("dispatch", 512, 128), ("dispatch", 2048, 128)]
    require(
        [(child["mode"], child["prefix"], child["chunk"]) for child in children[:2]]
        == expected_prefixes,
        "opening chronology drift",
    )
    validated_children = [
        validate_child(child, expected_identity, ambient_qwen) for child in children
    ]

    safety = packet["safety"]
    computed_run_p8192 = validate_safety(safety, validated_children[1])
    third_mode = "dispatch" if computed_run_p8192 else "compact"
    expected_tail = [
        (third_mode, 8192, 128),
        ("compact", 16384, 128),
        ("compact", 32768, 128),
        ("overlap", 32768, 1024),
    ]
    require(
        [(child["mode"], child["prefix"], child["chunk"]) for child in children[2:]]
        == expected_tail,
        "closing chronology drift",
    )
    require(
        all(
            validated["meta"]["device"] == probe["device"]
            for validated in validated_children
        ),
        "child device-name drift",
    )
    require(
        all(
            validated["meta"]["device_registry_id"] == probe["device_registry_id"]
            for validated in validated_children
        ),
        "child device-registry drift",
    )
    require(
        all(
            validated["meta"]["max_buffer_length"] == probe["max_buffer_length"]
            for validated in validated_children
        ),
        "child maxBufferLength drift",
    )

    cell_summaries: list[dict[str, Any]] = []
    for child, validated in zip(children, validated_children, strict=True):
        cell_summaries.append(
            compact_summary(child, validated)
            if child["mode"] == "compact"
            else paired_summary(child, validated)
        )

    d1 = analyze_d1_cells(cell_summaries, validated_children)
    overlap = cell_summaries[5]
    d2_gates = {
        "at_least_five_gpu_wins": overlap["gpu_wins"] >= 5,
        "ab_gpu_saving_positive": overlap["strata"]["AB"]["gpu_saving_median_ms"] > 0,
        "ba_gpu_saving_positive": overlap["strata"]["BA"]["gpu_saving_median_ms"] > 0,
        "median_gpu_saving_at_least_0_25_ms": overlap["gpu_saving_median_ms"] >= 0.25,
    }
    return {
        "schema_version": 2,
        "implementation_commit": expected_identity["head"],
        "implementation_tree": expected_identity["tree"],
        "binary_sha256": expected_identity["binary_sha256"],
        "device_registry_id": probe["device_registry_id"],
        "device": probe["device"],
        "safety": safety,
        "cells": cell_summaries,
        "d1_scaling": d1["scaling"],
        "gates": {"D1": d1["gates"], "D2": d2_gates},
        "disposition": {
            "D1": d1["disposition"],
            "D2": "RETAIN_PREFIX_ONLY" if all(d2_gates.values()) else "KILL",
        },
    }


def validate_failure_child(
    child: dict[str, Any], expected_identity: dict[str, Any], ambient_qwen: list[str]
) -> str:
    required = {
        "mode",
        "prefix",
        "chunk",
        "state",
        "command",
        "identity_before",
        "pid",
        "spawned_at",
        "execution",
        "identity_after",
        "failure_arm",
    }
    allowed = required | {"records", "parse_error"}
    require(required <= set(child) <= allowed, "failure child schema drift")
    require(child["identity_before"] == expected_identity, "failure pre-identity drift")
    require(child["identity_after"] == expected_identity, "failure post-identity drift")
    expected_command = [
        expected_identity["binary_path"],
        SCREEN_TEST,
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]
    require(child["command"] == expected_command, "failure child command drift")
    validate_execution(
        child["execution"],
        expected_command,
        expected_overrides={
            "QWEN_VT_REBUILD_MODE": child["mode"],
            "QWEN_VT_REBUILD_PREFIX": str(child["prefix"]),
            "QWEN_VT_REBUILD_CHUNK": str(child["chunk"]),
        },
        expected_removed=ambient_qwen,
    )
    require(child["execution"]["pid"] == child["pid"], "failure child PID drift")
    require(child["execution"]["returncode"] != 0, "failure child returned success")
    matches = FAILURE_PATTERN.findall(execution_text(child["execution"]))
    require(len(matches) == 1, "failure label missing or ambiguous")
    derived = matches[0]
    require(child["failure_arm"] == derived, "stored failure label drift")
    return derived


def analyze_completed_d1_prefix(packet: dict[str, Any]) -> dict[str, Any]:
    expected_identity, probe = validate_packet_identity(packet)
    ambient_qwen = packet["ambient_qwen_environment"]
    children = packet["children"]
    require(len(children) >= 5, "completed D1 evidence missing")
    first_five = children[:5]
    validated = [
        validate_child(child, expected_identity, ambient_qwen) for child in first_five
    ]
    require(
        [(child["mode"], child["prefix"], child["chunk"]) for child in first_five[:2]]
        == [("dispatch", 512, 128), ("dispatch", 2048, 128)],
        "D1 opening chronology drift",
    )
    run_p8192 = validate_safety(packet["safety"], validated[1])
    expected = [
        ("dispatch" if run_p8192 else "compact", 8192, 128),
        ("compact", 16384, 128),
        ("compact", 32768, 128),
    ]
    require(
        [(child["mode"], child["prefix"], child["chunk"]) for child in first_five[2:]]
        == expected,
        "D1 closing chronology drift",
    )
    for item in validated:
        require(
            item["meta"]["device_registry_id"] == probe["device_registry_id"],
            "D1 device drift",
        )
        require(
            item["meta"]["max_buffer_length"] == probe["max_buffer_length"],
            "D1 buffer limit drift",
        )
    summaries = [
        compact_summary(child, item)
        if child["mode"] == "compact"
        else paired_summary(child, item)
        for child, item in zip(first_five, validated, strict=True)
    ]
    return analyze_d1_cells(summaries, validated)


def _analyze_failure_packet_strict(packet: dict[str, Any]) -> dict[str, Any]:
    success_keys = {
        "schema_version",
        "started_at",
        "finished_at",
        "build_identity",
        "binary_path",
        "binary_sha256",
        "tracked_tree_clean",
        "ambient_qwen_environment",
        "expected_identity",
        "build",
        "correctness",
        "environment_before",
        "environment_after",
        "children",
        "safety",
        "failure",
    }
    postflight_failure = packet.get("postflight_failure")
    if postflight_failure:
        without_environment = (success_keys - {"environment_after"}) | {
            "postflight_failure"
        }
        with_environment = success_keys | {"postflight_failure"}
        require(
            set(packet) == without_environment or set(packet) == with_environment,
            "postflight failure packet schema drift",
        )
        return {
            "schema_version": 2,
            "implementation_commit": packet.get("expected_identity", {}).get("head"),
            "binary_sha256": packet.get("expected_identity", {}).get("binary_sha256"),
            "failure": packet.get("failure"),
            "postflight_failure": postflight_failure,
            "status": "POSTFLIGHT_INVALID",
            "disposition": {"D1": "INVALID", "D2": "INVALID"},
        }

    require(set(packet) == success_keys, "failure packet top-level schema drift")
    expected_identity, _ = validate_packet_identity(packet)
    ambient_qwen = packet["ambient_qwen_environment"]
    children = packet["children"]
    require(children, "failure packet has no attempted child")
    failure_arm = validate_failure_child(children[-1], expected_identity, ambient_qwen)
    if failure_arm == "D0":
        require(children[-1]["mode"] == "dispatch", "D0 failure outside dispatch cell")
        d1, d2, status = "INCONCLUSIVE_D0_FAILURE", "NOT_RUN", "D0_FAILURE"
    elif failure_arm == "D1":
        d1, d2, status = "KILL", "NOT_RUN", "D1_FAILURE"
    else:
        require(children[-1]["mode"] == "overlap", "D2 failure outside overlap cell")
        d1_result = analyze_completed_d1_prefix(packet)
        d1, d2, status = d1_result["disposition"], "KILL", f"{failure_arm}_FAILURE"
    return {
        "schema_version": 2,
        "implementation_commit": expected_identity["head"],
        "binary_sha256": expected_identity["binary_sha256"],
        "failure": packet["failure"],
        "postflight_failure": None,
        "status": status,
        "disposition": {"D1": d1, "D2": d2},
    }


def analyze_failure_packet(packet: dict[str, Any]) -> dict[str, Any]:
    try:
        return _analyze_failure_packet_strict(packet)
    except Exception as error:
        return {
            "schema_version": 2,
            "implementation_commit": packet.get("expected_identity", {}).get("head"),
            "binary_sha256": packet.get("expected_identity", {}).get("binary_sha256"),
            "failure": packet.get("failure"),
            "postflight_failure": packet.get("postflight_failure"),
            "status": "UNATTRIBUTED_FAILURE",
            "validation_error": f"{type(error).__name__}: {error}",
            "disposition": {"D1": "INVALID", "D2": "INVALID"},
        }


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit("usage: analyze.py CHRONOLOGY.json [RESULTS.json]")
    source = Path(sys.argv[1])
    packet = json.loads(source.read_text())
    result = (
        analyze_failure_packet(packet)
        if "failure" in packet
        else analyze_packet(packet)
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if len(sys.argv) == 3:
        Path(sys.argv[2]).write_text(rendered)
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
