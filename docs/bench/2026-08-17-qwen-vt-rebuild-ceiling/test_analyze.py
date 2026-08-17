#!/usr/bin/env python3
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import unittest

from analyze import (
    ARM_COMMON_KEYS,
    BUILD_COMMAND,
    CORRECTNESS_TESTS,
    HEAD_DIM,
    KV_DIM,
    LAYERS,
    N_KV,
    PAIRED_SCHEDULE,
    PROBE_TEST,
    SCREEN_TEST,
    SINGLE_BANKS,
    SYSTEM_COMMANDS,
    THREADS,
    analyze_failure_packet,
    analyze_packet,
    expected_arm,
)


BINARY = "/tmp/vt-campaign/qwen_llm_test"
IDENTITY = {
    "head": "abc123",
    "tree": "def456",
    "tracked_clean": True,
    "binary_path": BINARY,
    "binary_sha256": "f" * 64,
}


def stream(data: str) -> dict:
    raw = data.encode()
    return {
        "base64": base64.b64encode(raw).decode(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "length": len(raw),
    }


def execution(
    command: list[str],
    output: str,
    *,
    pid: int = 100,
    returncode: int = 0,
    overrides: dict[str, str] | None = None,
    removed: list[str] | None = None,
) -> dict:
    return {
        "command": command,
        "environment_overrides": overrides or {},
        "environment_removed": removed or [],
        "pid": pid,
        "started_at": "2026-08-17T00:00:00+00:00",
        "finished_at": "2026-08-17T00:00:01+00:00",
        "elapsed_s": 1.0,
        "returncode": returncode,
        "termination_signal": -returncode if returncode < 0 else None,
        "spawn_callback_error": None,
        "stdout": stream(output),
        "stderr": stream(""),
    }


def exact_test_packet(
    name: str,
    *,
    ignored: bool = False,
    marker: str = "",
    require_metal: bool = False,
) -> dict:
    command = [BINARY, name]
    if ignored:
        command.append("--ignored")
    command.extend(["--exact", "--nocapture", "--test-threads=1"])
    output = f"running 1 test\ntest {name} ... {marker}\nok\ntest result: ok. 1 passed; 0 failed\n"
    overrides = {"QWEN_REQUIRE_METAL_TESTS": "1"} if require_metal else {}
    return execution(command, output, overrides=overrides)


def arm_record(
    mode: str,
    prefix: int,
    chunk: int,
    role: str,
    bank: str,
    gpu_ms: float,
    *,
    pair: int | None = None,
    order: str | None = None,
    sequence: int | None = None,
    sample: int | None = None,
) -> dict:
    arm, compact, rows = expected_arm(mode, role, prefix, chunk)
    total = KV_DIM * rows
    groups = math.ceil(total / THREADS) if compact else total
    logical_bytes = 4 * LAYERS * KV_DIM * rows
    result = {
        "kind": "arm",
        "schema_version": 2,
        "mode": mode,
        "prefix": prefix,
        "chunk": chunk,
        "role": role,
        "arm": arm,
        "bank": bank,
        "base_pos": 0,
        "rows": rows,
        "threadgroups_per_layer": groups,
        "thread_slots_per_layer": groups * THREADS,
        "logical_bytes": logical_bytes,
        "wall_ms": gpu_ms + 0.1,
        "gpu_ms": gpu_ms,
        "gb_s": logical_bytes / (gpu_ms * 1e6),
    }
    if pair is not None:
        result.update({"pair": pair, "order": order, "sequence": sequence})
    if sample is not None:
        result["sample"] = sample
    expected_keys = ARM_COMMON_KEYS | (
        {"sample"} if mode == "compact" else {"pair", "order", "sequence"}
    )
    assert set(result) == expected_keys
    return result


def meta(mode: str, prefix: int, chunk: int) -> dict:
    bytes_per_buffer = 2 * (prefix + chunk) * KV_DIM
    return {
        "kind": "meta",
        "schema_version": 2,
        "mode": mode,
        "prefix": prefix,
        "chunk": chunk,
        "n_pos": prefix + chunk,
        "vt_stride": prefix + chunk,
        "layers": LAYERS,
        "n_kv": N_KV,
        "head_dim": HEAD_DIM,
        "kv_dim": KV_DIM,
        "device_registry_id": 42,
        "device": "Apple Test GPU",
        "max_buffer_length": 1 << 34,
        "recommended_max_working_set_size": 1 << 37,
        "bytes_per_layer_buffer": bytes_per_buffer,
        "total_requested_bytes": bytes_per_buffer * LAYERS * 4,
        "allocated_before": 0,
        "allocated_after": bytes_per_buffer * LAYERS * 4,
    }


def child(mode: str, prefix: int, chunk: int, records: list[dict], pid: int) -> dict:
    command = [
        BINARY,
        SCREEN_TEST,
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]
    markers = "\n".join(
        f"VT_REBUILD_JSON {json.dumps(record, separators=(',', ':'))}"
        for record in records
    )
    output = (
        f"running 1 test\ntest {SCREEN_TEST} ... {markers}\n"
        "ok\ntest result: ok. 1 passed; 0 failed\n"
    )
    return {
        "mode": mode,
        "prefix": prefix,
        "chunk": chunk,
        "state": "parsed",
        "command": command,
        "identity_before": copy.deepcopy(IDENTITY),
        "pid": pid,
        "spawned_at": "2026-08-17T00:00:00+00:00",
        "execution": execution(
            command,
            output,
            pid=pid,
            overrides={
                "QWEN_VT_REBUILD_MODE": mode,
                "QWEN_VT_REBUILD_PREFIX": str(prefix),
                "QWEN_VT_REBUILD_CHUNK": str(chunk),
            },
        ),
        "identity_after": copy.deepcopy(IDENTITY),
        "failure_arm": None,
        "records": records,
    }


def paired_child(
    mode: str, prefix: int, chunk: int, a_gpu: float, b_gpu: float, pid: int
) -> dict:
    records = [meta(mode, prefix, chunk)]
    factors = [1.00, 1.01, 0.99, 1.02, 0.98, 1.00]
    for pair_index, pair in enumerate(PAIRED_SCHEDULE, start=1):
        order = "".join(role for role, _ in pair)
        for sequence, (role, bank) in enumerate(pair, start=1):
            base = a_gpu if role == "A" else b_gpu
            records.append(
                arm_record(
                    mode,
                    prefix,
                    chunk,
                    role,
                    bank,
                    base * factors[pair_index - 1],
                    pair=pair_index,
                    order=order,
                    sequence=sequence,
                )
            )
    return child(mode, prefix, chunk, records, pid)


def compact_child(prefix: int, gpu_ms: float, pid: int) -> dict:
    chunk = 128
    records = [meta("compact", prefix, chunk)]
    factors = [1.00, 1.01, 0.99, 1.02, 0.98, 1.00]
    for sample, (bank, factor) in enumerate(
        zip(SINGLE_BANKS, factors, strict=True), start=1
    ):
        records.append(
            arm_record(
                "compact",
                prefix,
                chunk,
                "S",
                bank,
                gpu_ms * factor,
                sample=sample,
            )
        )
    return child("compact", prefix, chunk, records, pid)


def system_packet(command: list[str], pid: int) -> dict:
    return execution(command, "ok\n", pid=pid)


def environment(pid_base: int) -> dict:
    probe_record = {
        "schema_version": 1,
        "test": PROBE_TEST,
        "device_registry_id": 42,
        "device": "Apple Test GPU",
        "max_buffer_length": 1 << 34,
        "recommended_max_working_set_size": 1 << 37,
    }
    probe = exact_test_packet(
        PROBE_TEST,
        ignored=True,
        marker=f"VT_ENV_JSON {json.dumps(probe_record, separators=(',', ':'))}",
    )
    protected = execution(
        ["ps", "-p", "8770", "-o", "pid=,state=,etime=,command="],
        "",
        pid=pid_base + 20,
        returncode=1,
    )
    protected["active"] = False
    return {
        "captured_at": "2026-08-17T00:00:00+00:00",
        "identity": copy.deepcopy(IDENTITY),
        "system": {
            name: system_packet(command, pid_base + index)
            for index, (name, command) in enumerate(SYSTEM_COMMANDS.items())
        },
        "protected_pid": protected,
        "metal_probe": probe,
    }


def passing_packet() -> dict:
    children = [
        paired_child("dispatch", 512, 128, 10.0, 0.2, 201),
        paired_child("dispatch", 2048, 128, 40.0, 0.7, 202),
        paired_child("dispatch", 8192, 128, 120.0, 2.5, 203),
        compact_child(16384, 5.0, 204),
        compact_child(32768, 10.0, 205),
        paired_child("overlap", 32768, 1024, 10.4, 10.0, 206),
    ]
    p2048_d0_walls = [
        record["wall_ms"]
        for record in children[1]["records"]
        if record.get("arm") == "D0"
    ]
    return {
        "schema_version": 2,
        "started_at": "2026-08-17T00:00:00+00:00",
        "finished_at": "2026-08-17T00:01:00+00:00",
        "build_identity": {"head": IDENTITY["head"], "tree": IDENTITY["tree"]},
        "binary_path": BINARY,
        "binary_sha256": IDENTITY["binary_sha256"],
        "tracked_tree_clean": True,
        "expected_identity": copy.deepcopy(IDENTITY),
        "ambient_qwen_environment": [],
        "build": execution(BUILD_COMMAND, "build\n", pid=50),
        "correctness": [
            exact_test_packet(name, require_metal=True) for name in CORRECTNESS_TESTS
        ],
        "environment_before": environment(300),
        "environment_after": environment(400),
        "children": children,
        "safety": {
            "p2048_d0_wall_ms": p2048_d0_walls,
            "p2048_d0_wall_median_ms": 40.1,
            "p2048_d0_wall_max_ms": max(p2048_d0_walls),
            "run_dispatch_p8192": True,
        },
    }


def reset_gpu(record: dict, gpu_ms: float) -> None:
    record["gpu_ms"] = gpu_ms
    record["wall_ms"] = gpu_ms + 0.1
    record["gb_s"] = record["logical_bytes"] / (gpu_ms * 1e6)


def refresh_child_output(child_packet: dict) -> None:
    records = child_packet["records"]
    markers = "\n".join(
        f"VT_REBUILD_JSON {json.dumps(record, separators=(',', ':'))}"
        for record in records
    )
    output = (
        f"running 1 test\ntest {SCREEN_TEST} ... {markers}\n"
        "ok\ntest result: ok. 1 passed; 0 failed\n"
    )
    child_packet["execution"]["stdout"] = stream(output)


class AnalyzeTests(unittest.TestCase):
    def test_passing_packet_advances_d1_and_retains_d2(self) -> None:
        result = analyze_packet(passing_packet())
        self.assertEqual(result["disposition"]["D1"], "ADVANCE_PRODUCT_AB")
        self.assertEqual(result["disposition"]["D2"], "RETAIN_PREFIX_ONLY")

    def test_command_mutation_fails_closed(self) -> None:
        packet = passing_packet()
        packet["children"][0]["command"].remove("--exact")
        with self.assertRaisesRegex(ValueError, "command drift"):
            analyze_packet(packet)

    def test_binary_identity_mutation_fails_closed(self) -> None:
        packet = passing_packet()
        packet["children"][2]["identity_after"]["binary_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "post-identity drift"):
            analyze_packet(packet)

    def test_schedule_mutation_fails_closed(self) -> None:
        packet = passing_packet()
        packet["children"][0]["records"][1]["bank"] = "Y"
        refresh_child_output(packet["children"][0])
        with self.assertRaisesRegex(ValueError, "bank drift"):
            analyze_packet(packet)

    def test_missing_arm_fails_closed(self) -> None:
        packet = passing_packet()
        packet["children"][1]["records"].pop()
        refresh_child_output(packet["children"][1])
        with self.assertRaisesRegex(ValueError, "cardinality drift"):
            analyze_packet(packet)

    def test_unknown_record_fails_closed(self) -> None:
        packet = passing_packet()
        packet["children"][1]["records"].insert(1, {"kind": "unknown"})
        refresh_child_output(packet["children"][1])
        with self.assertRaisesRegex(
            ValueError, "cardinality drift|unknown/interleaved"
        ):
            analyze_packet(packet)

    def test_threadgroup_mutation_fails_closed(self) -> None:
        packet = passing_packet()
        packet["children"][2]["records"][1]["threadgroups_per_layer"] += 1
        refresh_child_output(packet["children"][2])
        with self.assertRaisesRegex(ValueError, "threadgroup count drift"):
            analyze_packet(packet)

    def test_stream_mutation_fails_closed(self) -> None:
        packet = passing_packet()
        packet["children"][0]["execution"]["stdout"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stream hash drift"):
            analyze_packet(packet)

    def test_safety_mutation_fails_closed(self) -> None:
        packet = passing_packet()
        packet["safety"]["run_dispatch_p8192"] = False
        with self.assertRaisesRegex(ValueError, "safety decision drift"):
            analyze_packet(packet)

    def test_protected_pid_mutation_fails_closed(self) -> None:
        packet = passing_packet()
        packet["environment_after"]["protected_pid"]["active"] = True
        packet["environment_after"]["protected_pid"]["returncode"] = 0
        with self.assertRaisesRegex(ValueError, "protected PID was active"):
            analyze_packet(packet)

    def test_subthreshold_dispatch_is_killed(self) -> None:
        packet = passing_packet()
        for record in packet["children"][2]["records"]:
            if record.get("arm") == "D0":
                reset_gpu(record, 10.0)
            elif record.get("arm") == "D1":
                reset_gpu(record, 2.0)
        refresh_child_output(packet["children"][2])
        result = analyze_packet(packet)
        self.assertEqual(result["disposition"]["D1"], "KILL")

    def test_nonmonotonic_d1_is_killed(self) -> None:
        packet = passing_packet()
        for record in packet["children"][4]["records"]:
            if record.get("arm") == "D1":
                reset_gpu(record, 4.0)
        refresh_child_output(packet["children"][4])
        result = analyze_packet(packet)
        self.assertFalse(result["gates"]["D1"]["d1_gpu_time_nondecreasing_8k_32k"])

    def test_subthreshold_overlap_is_killed(self) -> None:
        packet = passing_packet()
        for record in packet["children"][5]["records"]:
            if record.get("arm") == "D1":
                reset_gpu(record, 10.05)
            elif record.get("arm") == "D2":
                reset_gpu(record, 10.0)
        refresh_child_output(packet["children"][5])
        result = analyze_packet(packet)
        self.assertEqual(result["disposition"]["D2"], "KILL")

    def test_failure_attribution(self) -> None:
        def mark_failure(packet: dict, label: str) -> None:
            child_packet = packet["children"][-1]
            child_packet["failure_arm"] = label
            child_packet["execution"]["returncode"] = 101
            child_packet["execution"]["termination_signal"] = None
            child_packet["execution"]["stdout"] = stream(
                f"running 1 test\ntest {SCREEN_TEST} ... V_T command failed arm={label}\nFAILED\n"
            )
            packet["failure"] = {"detail": "failed"}

        packet = passing_packet()
        packet["children"] = packet["children"][:1]
        mark_failure(packet, "D0")
        self.assertEqual(
            analyze_failure_packet(packet)["disposition"]["D1"],
            "INCONCLUSIVE_D0_FAILURE",
        )

        packet = passing_packet()
        packet["children"] = packet["children"][:1]
        mark_failure(packet, "D1")
        self.assertEqual(analyze_failure_packet(packet)["disposition"]["D1"], "KILL")

        packet = passing_packet()
        mark_failure(packet, "PREP_D2")
        self.assertEqual(analyze_failure_packet(packet)["disposition"]["D2"], "KILL")
        self.assertEqual(
            analyze_failure_packet(packet)["disposition"]["D1"],
            "ADVANCE_PRODUCT_AB",
        )


if __name__ == "__main__":
    unittest.main()
