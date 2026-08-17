#!/usr/bin/env python3
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import unittest

from analyze import (
    CELLS,
    MODEL,
    MODEL_BYTES,
    MODEL_SHA256,
    PREREGISTRATION_SHA256,
    SCHEDULE,
    SECTION_NAMES,
    WARMUP,
    analyze_packet,
    expected_command,
)


HEX = "a" * 64
BINARY = "/campaign/qwen-bench"
IDENTITY = {
    "head": "1" * 64,
    "tree": "2" * 64,
    "tracked_clean": True,
    "binary_path": BINARY,
    "binary_sha256": "3" * 64,
}
PHYSICAL = 128 << 30


def stream(raw: bytes) -> dict:
    return {
        "base64": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "length": len(raw),
    }


def plan(block: int, matrix: int, rows: int) -> dict:
    allocation = {"name": "matrix_vt", "logical_bytes": 4096, "dtype": "F16"}
    return {
        "block_size": block,
        "matrix_max_pos": matrix,
        "matrix_query_rows": rows,
        "logical_bytes": 4096,
        "maximum_logical_bytes": 8192,
        "allocation_count": 1,
        "deferred_allocation_count": 0,
        "allocations": [allocation],
        "deferred_allocations": [],
    }


def snapshot(prefix: int, seed: str = "a") -> dict:
    return {
        "fingerprint": seed * 64,
        "snapshot_bytes": 1_000_000 + prefix,
        "prefix_len": prefix,
        "pending_token": None,
        "kv_n_pos": [prefix] * 16,
        "sections": [
            {"name": name, "bytes": index * 16, "sha256": seed * 64}
            for index, name in enumerate(SECTION_NAMES, start=1)
        ],
    }


def snapshot_identity() -> dict:
    return {
        "model_id": 1,
        "tokenizer_id": 2,
        "layout_version": 1,
        "n_attn_layers": 16,
        "n_gdn_layers": 48,
        "kv_dim_elements": 1024,
        "kv_bytes_per_token": 2048,
        "kv_storage_kind": "F16",
        "gdn_state_elements_per_layer": 4096,
        "gdn_conv_elements_per_layer": 512,
    }


def vt(prefix: int, chunk: int, role: str) -> dict:
    elements = 16 * 4 * 256 * prefix
    return {
        "calls": 16,
        "row_sum": 16 * prefix,
        "element_sum": elements,
        "threadgroup_sum": elements if role == "A" else 16 * 4 * prefix,
        "compact_calls": 16 if role == "B" else 0,
        "legacy_calls": 16 if role == "A" else 0,
        "base_pos_sum": 0,
        "n_pos_sum": 16 * (prefix + chunk),
    }


def arm(
    prefix: int,
    chunk: int,
    kind: str,
    role: str,
    bank: str,
    pair: int,
    order: str,
    sequence: int,
    suffix_wall: float,
) -> dict:
    snap = snapshot(prefix)
    is_a = role == "A"
    suffix_gpu = suffix_wall - 100.0 if is_a else suffix_wall - 50.0
    ttft = suffix_wall + 200.0
    request = suffix_wall + 300.0
    return {
        "schema_version": 1,
        "kind": kind,
        "phase": kind,
        "cell": f"P{prefix}/C{chunk}",
        "prefix": prefix,
        "chunk": chunk,
        "role": role,
        "compact": role == "B",
        "bank": bank,
        "pair": pair,
        "order": order,
        "sequence": sequence,
        "capacity": prefix + chunk + 8,
        "sequence_create_ms": 2.0,
        "restore_ms": 3.0,
        "suffix_wall_ms": suffix_wall,
        "suffix_gpu_ms": suffix_gpu,
        "post_restore_ttft_ms": ttft,
        "request_wall_ms": request,
        "first_token": 42,
        "logits_sha256": "b" * 64,
        "snapshot_fingerprint": snap["fingerprint"],
        "snapshot_bytes": snap["snapshot_bytes"],
        "snapshot_identity": snapshot_identity(),
        "snapshot": snap,
        "scratch_plan": plan(chunk, prefix + chunk, chunk),
        "capture_owner_thread": "ThreadId(1)",
        "vt_dispatch": vt(prefix, chunk, role),
    }


def setup(prefix: int, chunk: int) -> dict:
    file_identity = {
        "device": 1,
        "inode": 2,
        "bytes": MODEL_BYTES,
        "modified_seconds": 3,
        "modified_nanoseconds": 4,
    }
    return {
        "schema_version": 1,
        "kind": "setup",
        "build_identity": {"head": IDENTITY["head"], "tree": IDENTITY["tree"]},
        "device": "Apple Test GPU",
        "device_registry_id": 99,
        "max_buffer_length": 1 << 34,
        "model": MODEL,
        "model_bytes": MODEL_BYTES,
        "model_sha256": MODEL_SHA256,
        "model_sha256_after_load": MODEL_SHA256,
        "model_file_identity": file_identity,
        "model_path_identity_after_load": copy.deepcopy(file_identity),
        "prefix": prefix,
        "chunk": chunk,
        "capacity": prefix + chunk + 8,
        "prefix_token_sha256": "c" * 64,
        "suffix_token_sha256": "d" * 64,
        "prefix_gpu_ms": 100.0,
        "prefix_plan": plan(1024, prefix, 1024),
        "suffix_plan": plan(chunk, prefix + chunk, chunk),
        "snapshot": snapshot(prefix),
        "correctness_snapshot_bytes": 2_000_000,
        "session_allocation_delta": 1000,
        "allocated_before_prefix": 100,
        "allocated_after_builder_drop": 200,
        "peak_bound": 64 << 30,
        "physical_memory_bytes": PHYSICAL,
        "preflight_only": False,
        "allocated_before_x": 200,
        "allocated_after_x": 300,
        "allocated_after_y": 400,
        "scratch_x_buffer_ids": [10, 11],
        "scratch_y_buffer_ids": [20, 21],
    }


def correctness(prefix: int, chunk: int) -> dict:
    return {
        "schema_version": 1,
        "kind": "correctness",
        "prefix": prefix,
        "chunk": chunk,
        "match": True,
        "suffix_logits_sha256": "b" * 64,
        "state": snapshot(prefix + chunk, "e"),
        "continuation": {
            "token_ids": [42, 43, 44, 45, 46, 47, 48, 49],
            "logits_sha256": ["f" * 64] * 8,
        },
    }


def records(prefix: int, chunk: int, a_wall: float, b_wall: float) -> list[dict]:
    result = [setup(prefix, chunk)]
    result.append(arm(prefix, chunk, "correctness_arm", "A", "X", 0, "AB", 1, a_wall))
    result.append(arm(prefix, chunk, "correctness_arm", "B", "Y", 0, "AB", 2, b_wall))
    result.append(correctness(prefix, chunk))
    for sequence, (role, bank) in enumerate(WARMUP, start=3):
        result.append(
            arm(
                prefix,
                chunk,
                "warmup",
                role,
                bank,
                0,
                "W",
                sequence,
                a_wall if role == "A" else b_wall,
            )
        )
    sequence = 7
    for pair, (first, second, order) in enumerate(SCHEDULE, start=1):
        for role, bank in [first, second]:
            result.append(
                arm(
                    prefix,
                    chunk,
                    "arm",
                    role,
                    bank,
                    pair,
                    order,
                    sequence,
                    a_wall if role == "A" else b_wall,
                )
            )
            sequence += 1
    result.append(
        {
            "schema_version": 1,
            "kind": "complete",
            "prefix": prefix,
            "chunk": chunk,
            "expected_logits_sha256": "b" * 64,
        }
    )
    return result


def refresh(child: dict) -> None:
    output = b"".join(
        b"VT_PRODUCT_JSON "
        + json.dumps(record, separators=(",", ":"), allow_nan=True).encode("ascii")
        + b"\n"
        for record in child["records"]
    )
    child["execution"]["stdout"] = stream(output)


def child(prefix: int, chunk: int, a_wall: float, b_wall: float, pid: int) -> dict:
    command = expected_command(BINARY, prefix, chunk, PHYSICAL)
    result = {
        "prefix": prefix,
        "chunk": chunk,
        "state": "parsed",
        "command": command,
        "identity_before": copy.deepcopy(IDENTITY),
        "pid": pid,
        "spawned_at": "2026-08-17T00:00:00Z",
        "execution": {
            "command": command,
            "environment_overrides": {},
            "environment_removed": [],
            "pid": pid,
            "started_at": "2026-08-17T00:00:00Z",
            "finished_at": "2026-08-17T00:01:00Z",
            "elapsed_s": 60.0,
            "returncode": 0,
            "termination_signal": None,
            "spawn_callback_error": None,
            "stdout": stream(b""),
            "stderr": stream(b""),
        },
        "identity_after": copy.deepcopy(IDENTITY),
        "records": records(prefix, chunk, a_wall, b_wall),
    }
    refresh(result)
    return result


def environment() -> dict:
    return {
        "captured_at": "2026-08-17T00:00:00Z",
        "identity": copy.deepcopy(IDENTITY),
        "physical_memory_bytes": PHYSICAL,
        "memory_pressure": "normal",
        "swap_bytes": 0,
        "ac_power": True,
        "thermal_state": "nominal",
        "protected_pid_8770_active": False,
    }


def measured(child_packet: dict) -> list[dict]:
    return child_packet["records"][8:20]


def a_walls(child_packet: dict) -> list[float]:
    return [
        record["suffix_wall_ms"]
        for record in measured(child_packet)
        if record["role"] == "A"
    ]


def safety(children: list[dict]) -> dict:
    first, second = a_walls(children[0]), a_walls(children[1])
    return {
        "p8192_c128_a_suffix_wall_ms": first,
        "p8192_c128_a_suffix_wall_median_ms": float(sorted(first)[2] + sorted(first)[3])
        / 2,
        "p8192_c128_a_suffix_wall_max_ms": max(first),
        "run_p16384_c128": True,
        "p16384_c128_a_suffix_wall_ms": second,
        "p16384_c128_a_suffix_wall_median_ms": float(
            sorted(second)[2] + sorted(second)[3]
        )
        / 2,
        "p16384_c128_a_suffix_wall_max_ms": max(second),
        "memory_pressure_normal": True,
        "swap_not_increased": True,
        "run_p16384_c1024": True,
    }


def valid_packet() -> dict:
    children = [
        child(8192, 128, 2000.0, 1000.0, 101),
        child(16384, 128, 3000.0, 1000.0, 102),
        child(16384, 1024, 3500.0, 1000.0, 103),
    ]
    return {
        "schema_version": 1,
        "started_at": "2026-08-17T00:00:00Z",
        "finished_at": "2026-08-17T01:00:00Z",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "tracked_tree_clean": True,
        "build_identity": {"head": IDENTITY["head"], "tree": IDENTITY["tree"]},
        "expected_identity": copy.deepcopy(IDENTITY),
        "binary_path": BINARY,
        "binary_sha256": IDENTITY["binary_sha256"],
        "environment_removed": [],
        "environment_before": environment(),
        "environment_after": environment(),
        "children": children,
        "safety": safety(children),
    }


def command_packet(argv: list[str], pid: int = 700, output: bytes = b"") -> dict:
    return {
        "argv": argv,
        "pid": pid,
        "started_at": "2026-08-17T00:00:00Z",
        "completed_at": "2026-08-17T00:00:01Z",
        "elapsed_seconds": 1.0,
        "returncode": 0,
        "termination_signal": None,
        "stdout": collector_stream(stream(output)),
        "stderr": collector_stream(stream(b"")),
    }


def collector_stream(value: dict) -> dict:
    return {
        "base64": value["base64"],
        "bytes": value["length"],
        "sha256": value["sha256"],
    }


def collector_system() -> dict:
    commands = {
        "os": ["sw_vers"],
        "uname": ["uname", "-a"],
        "physical_memory": ["sysctl", "-n", "hw.memsize"],
        "ac_power": ["pmset", "-g", "batt"],
        "memory_pressure": ["memory_pressure", "-Q"],
        "swap": ["sysctl", "-n", "vm.swapusage"],
        "thermal": ["pmset", "-g", "therm"],
    }
    return {
        "captured_at": "2026-08-17T00:00:00Z",
        "commands": {
            name: command_packet(argv, 800 + index)
            for index, (name, argv) in enumerate(commands.items())
        },
        "physical_memory_bytes": PHYSICAL,
        "ac_power": True,
        "memory_pressure_normal": True,
        "swap_used_bytes": 0,
    }


def collector_provenance() -> dict:
    query = command_packet(["ps", "-axo", "pid=,ppid=,state=,command=", "-ww"], 850)
    return {
        "captured_at": "2026-08-17T00:00:00Z",
        "process_guard": {
            "query": query,
            "protected_pid": None,
            "competing_qwen_model_processes": [],
        },
        "system": collector_system(),
    }


def collector_identity(binary: dict) -> dict:
    return {
        "head": IDENTITY["head"],
        "tree": IDENTITY["tree"],
        "untracked": [],
        "binary": binary,
    }


def as_collector_child(
    old: dict, binary: dict, source_identity: dict, *, preflight: bool = False
) -> dict:
    child_records = copy.deepcopy(old["records"])
    argv = list(old["command"])
    if preflight:
        argv.append("--preflight-only")
        child_records = child_records[:4] + [
            {
                "schema_version": 1,
                "kind": "preflight_complete",
                "prefix": old["prefix"],
                "chunk": old["chunk"],
                "expected_logits_sha256": "b" * 64,
            }
        ]
        child_records[0]["preflight_only"] = True
    output = b"".join(
        b"VT_PRODUCT_JSON "
        + json.dumps(record, separators=(",", ":")).encode("ascii")
        + b"\n"
        for record in child_records
    )
    process_query = command_packet(
        [
            "ps",
            "-p",
            str(old["pid"]),
            "-o",
            "pid=,ppid=,state=,lstart=,command=",
            "-ww",
        ],
        900 + old["pid"],
    )
    return {
        "cell": f"P{old['prefix']}/C{old['chunk']}",
        "preflight_only": preflight,
        "state": "validated",
        "argv": argv,
        "environment": {
            "explicit": {
                "HOME": "/tmp",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "TMPDIR": "/tmp",
            },
            "removed_inherited_names": [],
            "forbidden_present_after_sanitization": [],
        },
        "identity_before": copy.deepcopy(source_identity),
        "prepared_at": "2026-08-17T00:00:00Z",
        "pid": old["pid"],
        "started_at": "2026-08-17T00:00:00Z",
        "process_identity": {"query": process_query, "pid": old["pid"]},
        "completed_at": "2026-08-17T00:01:00Z",
        "elapsed_seconds": 60.0,
        "returncode": 0,
        "termination_signal": None,
        "stdout": collector_stream(stream(output)),
        "stderr": collector_stream(stream(b"")),
        "identity_after": copy.deepcopy(source_identity),
        "records": child_records,
    }


def collector_packet() -> dict:
    old = valid_packet()
    binary = {
        "canonical_path": BINARY,
        "bytes": 1000,
        "device": 1,
        "inode": 2,
        "mode": 0o100500,
        "sha256": IDENTITY["binary_sha256"],
    }
    source = {
        "head": IDENTITY["head"],
        "tree": IDENTITY["tree"],
        "tracked_clean": True,
        "allowed_untracked": [],
    }
    identity = collector_identity(binary)
    environment_record = {
        "explicit": {
            "HOME": "/tmp",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "TMPDIR": "/tmp",
        },
        "removed_inherited_names": [],
        "forbidden_present_after_sanitization": [],
    }
    timed = [as_collector_child(item, binary, identity) for item in old["children"]]
    admission = as_collector_child(old["children"][2], binary, identity, preflight=True)
    repairable = as_collector_child(
        old["children"][0], binary, identity, preflight=True
    )
    first = a_walls(old["children"][0])
    second = a_walls(old["children"][1])
    system_before, system_after = collector_system(), collector_system()
    safety_value = {
        "before_p16384_c128": {
            "predicate": "frozen",
            "source_cell_valid": True,
            "samples_ms": first,
            "median_ms": 2000.0,
            "maximum_ms": 2000.0,
            "allowed": True,
            "recorded_at": "2026-08-17T00:00:00Z",
        },
        "before_p16384_c1024": {
            "predicate": "frozen",
            "source_cell_valid": True,
            "samples_ms": second,
            "median_ms": 3000.0,
            "maximum_ms": 3000.0,
            "timing_allowed": True,
            "environment_before_admission": system_before,
            "environment_after_admission": system_after,
            "fresh_admission_ok": True,
            "allowed": True,
            "state": "applied",
            "recorded_at": "2026-08-17T00:00:00Z",
            "applied_at": "2026-08-17T00:00:01Z",
        },
    }
    build = command_packet(
        ["cargo", "build", "--release", "-p", "qwen-cli", "--bin", "qwen-bench"], 600
    )
    build["environment_removed"] = []
    return {
        "schema_version": 1,
        "campaign": "qwen-vt-product-ab",
        "started_at": "2026-08-17T00:00:00Z",
        "preregistration": {
            "path": "docs/bench/2026-08-17-qwen-vt-product-ab/README.md",
            "sha256": PREREGISTRATION_SHA256,
        },
        "source": source,
        "model": {
            "canonical_path": MODEL,
            "bytes": MODEL_BYTES,
            "sha256": MODEL_SHA256,
            "device": 1,
            "inode": 2,
            "modified_ns": 3_000_000_004,
        },
        "binary": binary,
        "expected_identity": identity,
        "build": build,
        "sanitized_child_environment": environment_record,
        "repairable_preflight": {
            "environment_before": collector_system(),
            "execution": {
                key: repairable[key]
                for key in [
                    "argv",
                    "pid",
                    "process_identity",
                    "started_at",
                    "completed_at",
                    "elapsed_seconds",
                    "returncode",
                    "termination_signal",
                    "stdout",
                    "stderr",
                    "records",
                ]
            },
            "environment_after": collector_system(),
        },
        "environment_before": collector_provenance(),
        "children": [timed[0], timed[1], admission, timed[2]],
        "safety": safety_value,
        "environment_after": collector_provenance(),
        "completed_at": "2026-08-17T01:00:00Z",
        "state": "completed",
    }


def set_role_endpoint(
    packet: dict, cell_index: int, endpoint: str, role: str, values: list[float]
) -> None:
    selected = [
        record
        for record in measured(packet["children"][cell_index])
        if record["role"] == role
    ]
    for record, value in zip(selected, values, strict=True):
        record[endpoint] = value
    refresh(packet["children"][cell_index])


class AnalyzeTests(unittest.TestCase):
    def test_collector_chronology_promotes(self) -> None:
        self.assertEqual(
            analyze_packet(collector_packet())["disposition"], "PROMOTE_DEFAULT"
        )

    def test_collector_provenance_mutation_fails_closed(self) -> None:
        packet = collector_packet()
        packet["expected_identity"]["binary"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "binary identity"):
            analyze_packet(packet)

    def test_valid_packet_promotes_and_reports_raw_pairs(self) -> None:
        result = analyze_packet(valid_packet())
        self.assertEqual(result["disposition"], "PROMOTE_DEFAULT")
        endpoint = result["cells"][0]["endpoints"]["suffix_wall_ms"]
        self.assertEqual(endpoint["paired_savings_ms"], [1000.0] * 6)
        self.assertEqual(endpoint["wins"], 6)
        self.assertEqual(endpoint["strata"]["AB"]["median_saving_ms"], 1000.0)

    def test_results_are_deterministic(self) -> None:
        packet = valid_packet()
        self.assertEqual(analyze_packet(packet), analyze_packet(copy.deepcopy(packet)))

    def test_promotion_threshold_equality_promotes(self) -> None:
        packet = valid_packet()
        set_role_endpoint(packet, 0, "suffix_wall_ms", "B", [1500.0] * 6)
        set_role_endpoint(packet, 0, "suffix_gpu_ms", "B", [1400.0] * 6)
        self.assertEqual(analyze_packet(packet)["disposition"], "PROMOTE_DEFAULT")

    def test_below_promotion_threshold_keeps_default_off(self) -> None:
        packet = valid_packet()
        set_role_endpoint(packet, 0, "suffix_wall_ms", "B", [1500.01] * 6)
        self.assertEqual(analyze_packet(packet)["disposition"], "KEEP_DEFAULT_OFF")

    def test_tie_is_not_a_win_and_keeps_default_off(self) -> None:
        packet = valid_packet()
        values = [1000.0] * 5 + [2000.0]
        set_role_endpoint(packet, 0, "suffix_wall_ms", "B", values)
        self.assertEqual(analyze_packet(packet)["disposition"], "KEEP_DEFAULT_OFF")

    def test_ttft_five_win_boundary_promotes(self) -> None:
        packet = valid_packet()
        set_role_endpoint(
            packet, 1, "post_restore_ttft_ms", "B", [2200.0] * 5 + [3200.0]
        )
        self.assertEqual(analyze_packet(packet)["disposition"], "PROMOTE_DEFAULT")

    def test_ttft_four_wins_keeps_default_off(self) -> None:
        packet = valid_packet()
        set_role_endpoint(
            packet, 1, "post_restore_ttft_ms", "B", [2200.0] * 4 + [3200.0] * 2
        )
        self.assertEqual(analyze_packet(packet)["disposition"], "KEEP_DEFAULT_OFF")

    def test_request_stratum_zero_keeps_default_off(self) -> None:
        packet = valid_packet()
        set_role_endpoint(
            packet,
            1,
            "request_wall_ms",
            "B",
            [3300.0, 1300.0, 3300.0, 1300.0, 3300.0, 1300.0],
        )
        self.assertEqual(analyze_packet(packet)["disposition"], "KEEP_DEFAULT_OFF")

    def test_immediate_regression_at_boundary_does_not_kill(self) -> None:
        packet = valid_packet()
        set_role_endpoint(packet, 0, "suffix_gpu_ms", "B", [1938.0] * 6)
        self.assertEqual(analyze_packet(packet)["disposition"], "KEEP_DEFAULT_OFF")

    def test_immediate_regression_above_boundary_kills(self) -> None:
        packet = valid_packet()
        set_role_endpoint(packet, 0, "suffix_gpu_ms", "B", [1938.01] * 6)
        self.assertEqual(analyze_packet(packet)["disposition"], "KILL")

    def test_missing_record_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][0]["records"].pop(9)
        refresh(packet["children"][0])
        with self.assertRaisesRegex(ValueError, "cardinality"):
            analyze_packet(packet)

    def test_duplicate_record_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][0]["records"].insert(
            9, copy.deepcopy(packet["children"][0]["records"][9])
        )
        refresh(packet["children"][0])
        with self.assertRaisesRegex(ValueError, "cardinality"):
            analyze_packet(packet)

    def test_reordered_records_fail_closed(self) -> None:
        packet = valid_packet()
        records_value = packet["children"][0]["records"]
        records_value[8], records_value[9] = records_value[9], records_value[8]
        refresh(packet["children"][0])
        with self.assertRaisesRegex(ValueError, "schedule|role|bank"):
            analyze_packet(packet)

    def test_global_sequence_mutation_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][1]["records"][14]["sequence"] = 99
        refresh(packet["children"][1])
        with self.assertRaisesRegex(ValueError, "schedule"):
            analyze_packet(packet)

    def test_binary_identity_drift_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][2]["identity_after"]["binary_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "binary identity"):
            analyze_packet(packet)

    def test_snapshot_identity_drift_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][0]["records"][10]["snapshot_identity"]["model_id"] = 999
        refresh(packet["children"][0])
        with self.assertRaisesRegex(ValueError, "common identity"):
            analyze_packet(packet)

    def test_logits_drift_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][0]["records"][12]["logits_sha256"] = "0" * 64
        refresh(packet["children"][0])
        with self.assertRaisesRegex(ValueError, "common identity"):
            analyze_packet(packet)

    def test_topology_drift_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][2]["records"][8]["vt_dispatch"]["threadgroup_sum"] += 1
        refresh(packet["children"][2])
        with self.assertRaisesRegex(ValueError, "topology"):
            analyze_packet(packet)

    def test_correctness_mismatch_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][0]["records"][3]["match"] = False
        refresh(packet["children"][0])
        with self.assertRaisesRegex(ValueError, "correctness mismatch"):
            analyze_packet(packet)

    def test_safety_decision_mismatch_fails_closed(self) -> None:
        packet = valid_packet()
        packet["safety"]["run_p16384_c1024"] = False
        with self.assertRaisesRegex(ValueError, "safety decision"):
            analyze_packet(packet)

    def test_nonfinite_timing_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][0]["records"][8]["suffix_wall_ms"] = math.nan
        refresh(packet["children"][0])
        with self.assertRaisesRegex(ValueError, "invalid suffix_wall_ms"):
            analyze_packet(packet)

    def test_negative_timing_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][0]["records"][8]["restore_ms"] = -0.1
        refresh(packet["children"][0])
        with self.assertRaisesRegex(ValueError, "invalid restore_ms"):
            analyze_packet(packet)

    def test_authenticated_stdout_mutation_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][0]["execution"]["stdout"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stream hash"):
            analyze_packet(packet)

    def test_unknown_arm_field_fails_closed(self) -> None:
        packet = valid_packet()
        packet["children"][0]["records"][8]["surprise"] = True
        refresh(packet["children"][0])
        with self.assertRaisesRegex(ValueError, "schema drift"):
            analyze_packet(packet)


if __name__ == "__main__":
    unittest.main()
