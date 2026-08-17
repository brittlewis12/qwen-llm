#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PREREGISTRATION_SHA256 = (
    "2ced683c74ebd5a287b16c804550e73ba86ee2a176efb8265e198567b01d21f7"
)
MODEL = "/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf"
MODEL_BYTES = 17_106_773_984
MODEL_SHA256 = "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
CELLS = [(8192, 128), (16384, 128), (16384, 1024)]
SCHEDULE = [
    (("A", "X"), ("B", "Y"), "AB"),
    (("B", "X"), ("A", "Y"), "BA"),
    (("B", "Y"), ("A", "X"), "BA"),
    (("A", "Y"), ("B", "X"), "AB"),
    (("A", "X"), ("B", "Y"), "AB"),
    (("B", "X"), ("A", "Y"), "BA"),
]
WARMUP = [("A", "X"), ("B", "Y"), ("A", "Y"), ("B", "X")]
ENDPOINTS = [
    "suffix_wall_ms",
    "suffix_gpu_ms",
    "post_restore_ttft_ms",
    "request_wall_ms",
]
LAYERS = 16
N_KV = 4
HEAD_DIM = 256

TOP_KEYS = {
    "schema_version",
    "started_at",
    "finished_at",
    "preregistration_sha256",
    "tracked_tree_clean",
    "build_identity",
    "expected_identity",
    "binary_path",
    "binary_sha256",
    "environment_removed",
    "environment_before",
    "environment_after",
    "children",
    "safety",
}
IDENTITY_KEYS = {"head", "tree", "tracked_clean", "binary_path", "binary_sha256"}
ENV_KEYS = {
    "captured_at",
    "identity",
    "physical_memory_bytes",
    "memory_pressure",
    "swap_bytes",
    "ac_power",
    "thermal_state",
    "protected_pid_8770_active",
}
EXECUTION_KEYS = {
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
CHILD_KEYS = {
    "prefix",
    "chunk",
    "state",
    "command",
    "identity_before",
    "pid",
    "spawned_at",
    "execution",
    "identity_after",
    "records",
}
PLAN_KEYS = {
    "block_size",
    "matrix_max_pos",
    "matrix_query_rows",
    "logical_bytes",
    "maximum_logical_bytes",
    "allocation_count",
    "deferred_allocation_count",
    "allocations",
    "deferred_allocations",
}
ALLOCATION_KEYS = {"name", "logical_bytes", "dtype"}
SNAPSHOT_KEYS = {
    "fingerprint",
    "snapshot_bytes",
    "prefix_len",
    "pending_token",
    "kv_n_pos",
    "sections",
}
SECTION_KEYS = {"name", "bytes", "sha256"}
SECTION_NAMES = [
    "identity",
    "prefix_tokens",
    "pending_token",
    "kv_n_pos",
    "kv_k_arena",
    "kv_v_arena",
    "gdn_conv_arena",
    "gdn_state_arena",
    "final_logits",
]
SNAPSHOT_IDENTITY_KEYS = {
    "model_id",
    "tokenizer_id",
    "layout_version",
    "n_attn_layers",
    "n_gdn_layers",
    "kv_dim_elements",
    "kv_bytes_per_token",
    "kv_storage_kind",
    "gdn_state_elements_per_layer",
    "gdn_conv_elements_per_layer",
}
ARM_KEYS = {
    "schema_version",
    "kind",
    "phase",
    "cell",
    "prefix",
    "chunk",
    "role",
    "compact",
    "bank",
    "pair",
    "order",
    "sequence",
    "capacity",
    "sequence_create_ms",
    "restore_ms",
    "suffix_wall_ms",
    "suffix_gpu_ms",
    "post_restore_ttft_ms",
    "request_wall_ms",
    "first_token",
    "logits_sha256",
    "snapshot_fingerprint",
    "snapshot_bytes",
    "snapshot_identity",
    "snapshot",
    "scratch_plan",
    "capture_owner_thread",
    "vt_dispatch",
}
VT_KEYS = {
    "calls",
    "row_sum",
    "element_sum",
    "threadgroup_sum",
    "compact_calls",
    "legacy_calls",
    "base_pos_sum",
    "n_pos_sum",
}
SETUP_KEYS = {
    "schema_version",
    "kind",
    "build_identity",
    "device",
    "device_registry_id",
    "max_buffer_length",
    "model",
    "model_bytes",
    "model_sha256",
    "model_sha256_after_load",
    "model_file_identity",
    "model_path_identity_after_load",
    "prefix",
    "chunk",
    "capacity",
    "prefix_token_sha256",
    "suffix_token_sha256",
    "prefix_gpu_ms",
    "prefix_plan",
    "suffix_plan",
    "snapshot",
    "correctness_snapshot_bytes",
    "session_allocation_delta",
    "allocated_before_prefix",
    "allocated_after_builder_drop",
    "peak_bound",
    "physical_memory_bytes",
    "preflight_only",
    "allocated_before_x",
    "allocated_after_x",
    "allocated_after_y",
    "scratch_x_buffer_ids",
    "scratch_y_buffer_ids",
}
FILE_IDENTITY_KEYS = {
    "device",
    "inode",
    "bytes",
    "modified_seconds",
    "modified_nanoseconds",
}


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def exact_keys(value: Any, keys: set[str], detail: str) -> None:
    require(isinstance(value, dict) and set(value) == keys, detail)


def finite_positive(value: Any, detail: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), detail)
    result = float(value)
    require(math.isfinite(result) and result > 0.0, detail)
    return result


def nonnegative_int(value: Any, detail: str) -> None:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0, detail
    )


def digest(value: Any, detail: str) -> None:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        detail,
    )


def object_id(value: Any, detail: str) -> None:
    require(
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value),
        detail,
    )


def median(values: list[float]) -> float:
    require(bool(values), "median of empty values")
    return float(statistics.median(values))


def stream_bytes(value: Any) -> bytes:
    exact_keys(value, {"base64", "sha256", "length"}, "stream schema drift")
    require(
        isinstance(value["length"], int) and value["length"] >= 0,
        "stream length invalid",
    )
    digest(value["sha256"], "stream digest invalid")
    try:
        raw = base64.b64decode(value["base64"], validate=True)
    except Exception as error:
        raise ValueError("stream base64 invalid") from error
    require(len(raw) == value["length"], "stream length drift")
    require(hashlib.sha256(raw).hexdigest() == value["sha256"], "stream hash drift")
    return raw


def validate_identity(value: Any) -> None:
    exact_keys(value, IDENTITY_KEYS, "identity schema drift")
    require(value["tracked_clean"] is True, "identity is not tracked-clean")
    object_id(value["head"], "implementation HEAD invalid")
    object_id(value["tree"], "implementation tree invalid")
    digest(value["binary_sha256"], "binary digest invalid")
    require(
        isinstance(value["binary_path"], str) and value["binary_path"],
        "binary path invalid",
    )


def validate_environment(value: Any, identity: dict[str, Any]) -> None:
    exact_keys(value, ENV_KEYS, "environment schema drift")
    require(value["identity"] == identity, "environment identity drift")
    require(
        isinstance(value["captured_at"], str) and value["captured_at"],
        "capture timestamp invalid",
    )
    require(
        isinstance(value["physical_memory_bytes"], int)
        and value["physical_memory_bytes"] > 0,
        "physical memory invalid",
    )
    require(value["memory_pressure"] == "normal", "memory pressure was not normal")
    nonnegative_int(value["swap_bytes"], "swap value invalid")
    require(value["ac_power"] is True, "AC power absent")
    require(value["thermal_state"] == "nominal", "thermal state not nominal")
    require(value["protected_pid_8770_active"] is False, "protected PID 8770 active")


def expected_command(binary: str, prefix: int, chunk: int, physical: int) -> list[str]:
    return [
        binary,
        "prefix-cache-vt-ab",
        "--model",
        MODEL,
        "--prefix-len",
        str(prefix),
        "--chunk-len",
        str(chunk),
        "--physical-memory-bytes",
        str(physical),
    ]


def validate_execution(value: Any, command: list[str], removed: list[str]) -> None:
    exact_keys(value, EXECUTION_KEYS, "execution schema drift")
    require(value["command"] == command, "execution command drift")
    require(
        value["environment_overrides"] == {}, "treatment environment override present"
    )
    require(value["environment_removed"] == removed, "removed environment drift")
    require(isinstance(value["pid"], int) and value["pid"] > 0, "child PID invalid")
    require(value["spawn_callback_error"] is None, "spawn persistence failed")
    finite_positive(value["elapsed_s"], "child elapsed time invalid")
    require(value["returncode"] == 0, "child process failed")
    require(value["termination_signal"] is None, "child was signalled")
    stdout = stream_bytes(value["stdout"])
    stderr = stream_bytes(value["stderr"])
    require(b"VT_PRODUCT_JSON " not in stderr, "Rust record appeared on stderr")
    stdout.decode("utf-8", errors="strict")
    stderr.decode("utf-8", errors="strict")


def validate_plan(plan: Any, block: int, matrix: int, rows: int) -> None:
    exact_keys(plan, PLAN_KEYS, "scratch plan schema drift")
    require(plan["block_size"] == block, "scratch block size drift")
    require(plan["matrix_max_pos"] == matrix, "scratch matrix capacity drift")
    require(plan["matrix_query_rows"] == rows, "scratch query rows drift")
    for key in [
        "logical_bytes",
        "maximum_logical_bytes",
        "allocation_count",
        "deferred_allocation_count",
    ]:
        nonnegative_int(plan[key], f"scratch {key} invalid")
    require(
        plan["maximum_logical_bytes"] >= plan["logical_bytes"],
        "scratch maximum below logical bytes",
    )
    for key, count_key in [
        ("allocations", "allocation_count"),
        ("deferred_allocations", "deferred_allocation_count"),
    ]:
        require(
            isinstance(plan[key], list) and len(plan[key]) == plan[count_key],
            "scratch allocation count drift",
        )
        for allocation in plan[key]:
            exact_keys(allocation, ALLOCATION_KEYS, "scratch allocation schema drift")
            require(
                isinstance(allocation["name"], str) and allocation["name"],
                "scratch allocation name invalid",
            )
            require(
                isinstance(allocation["dtype"], str) and allocation["dtype"],
                "scratch dtype invalid",
            )
            nonnegative_int(
                allocation["logical_bytes"], "scratch allocation bytes invalid"
            )


def validate_snapshot(snapshot: Any, prefix: int) -> None:
    exact_keys(snapshot, SNAPSHOT_KEYS, "snapshot schema drift")
    digest(snapshot["fingerprint"], "snapshot fingerprint invalid")
    require(
        isinstance(snapshot["snapshot_bytes"], int) and snapshot["snapshot_bytes"] > 0,
        "snapshot bytes invalid",
    )
    require(snapshot["prefix_len"] == prefix, "snapshot prefix length drift")
    require(
        snapshot["pending_token"] is None or isinstance(snapshot["pending_token"], int),
        "snapshot pending token invalid",
    )
    require(
        isinstance(snapshot["kv_n_pos"], list) and len(snapshot["kv_n_pos"]) == LAYERS,
        "snapshot kv_n_pos topology drift",
    )
    require(
        all(value == prefix for value in snapshot["kv_n_pos"]),
        "snapshot kv_n_pos drift",
    )
    require(
        isinstance(snapshot["sections"], list)
        and len(snapshot["sections"]) == len(SECTION_NAMES),
        "snapshot section count drift",
    )
    for section, name in zip(snapshot["sections"], SECTION_NAMES, strict=True):
        exact_keys(section, SECTION_KEYS, "snapshot section schema drift")
        require(section["name"] == name, "snapshot section order drift")
        nonnegative_int(section["bytes"], "snapshot section bytes invalid")
        digest(section["sha256"], "snapshot section digest invalid")


def validate_snapshot_identity(identity: Any) -> None:
    exact_keys(identity, SNAPSHOT_IDENTITY_KEYS, "snapshot identity schema drift")
    require(identity["n_attn_layers"] == LAYERS, "snapshot attention topology drift")
    require(identity["n_gdn_layers"] > 0, "snapshot GDN topology invalid")
    require(
        identity["kv_dim_elements"] == N_KV * HEAD_DIM, "snapshot KV dimension drift"
    )
    require(
        identity["kv_bytes_per_token"] == 2 * N_KV * HEAD_DIM,
        "snapshot KV storage drift",
    )
    require(identity["kv_storage_kind"] == "F16", "snapshot KV kind drift")
    for key in SNAPSHOT_IDENTITY_KEYS - {"kv_storage_kind"}:
        require(
            isinstance(identity[key], int) and identity[key] >= 0,
            f"snapshot identity {key} invalid",
        )


def validate_vt(record: dict[str, Any], prefix: int, chunk: int, role: str) -> None:
    vt = record["vt_dispatch"]
    exact_keys(vt, VT_KEYS, "V_T dispatch schema drift")
    elements = LAYERS * N_KV * HEAD_DIM * prefix
    expected = {
        "calls": LAYERS,
        "row_sum": LAYERS * prefix,
        "element_sum": elements,
        "threadgroup_sum": elements if role == "A" else LAYERS * N_KV * prefix,
        "compact_calls": LAYERS if role == "B" else 0,
        "legacy_calls": LAYERS if role == "A" else 0,
        "base_pos_sum": 0,
        "n_pos_sum": LAYERS * (prefix + chunk),
    }
    require(vt == expected, "V_T topology drift")


def validate_arm(
    record: Any,
    prefix: int,
    chunk: int,
    kind: str,
    role: str,
    bank: str,
    pair: int,
    order: str,
    sequence: int,
    common: dict[str, Any],
) -> None:
    exact_keys(record, ARM_KEYS, "arm record schema drift")
    require(record["schema_version"] == SCHEMA_VERSION, "arm schema version drift")
    require(record["kind"] == kind and record["phase"] == kind, "arm phase drift")
    require(record["cell"] == f"P{prefix}/C{chunk}", "arm cell drift")
    require(
        record["prefix"] == prefix and record["chunk"] == chunk, "arm geometry drift"
    )
    require(
        record["role"] == role and record["compact"] is (role == "B"), "arm role drift"
    )
    require(record["bank"] == bank, "arm bank drift")
    require(
        record["pair"] == pair
        and record["order"] == order
        and record["sequence"] == sequence,
        "arm schedule drift",
    )
    require(record["capacity"] == prefix + chunk + 8, "arm capacity drift")
    for endpoint in ["sequence_create_ms", "restore_ms", *ENDPOINTS]:
        finite_positive(record[endpoint], f"invalid {endpoint}")
    require(
        isinstance(record["first_token"], int) and record["first_token"] >= 0,
        "first token invalid",
    )
    digest(record["logits_sha256"], "logits digest invalid")
    require(
        isinstance(record["capture_owner_thread"], str)
        and record["capture_owner_thread"],
        "capture owner missing",
    )
    validate_snapshot_identity(record["snapshot_identity"])
    validate_snapshot(record["snapshot"], prefix)
    validate_plan(record["scratch_plan"], chunk, prefix + chunk, chunk)
    require(
        record["snapshot_fingerprint"] == record["snapshot"]["fingerprint"],
        "snapshot fingerprint field drift",
    )
    require(
        record["snapshot_bytes"] == record["snapshot"]["snapshot_bytes"],
        "snapshot byte field drift",
    )
    validate_vt(record, prefix, chunk, role)
    fields = {
        "first_token": record["first_token"],
        "logits_sha256": record["logits_sha256"],
        "snapshot_fingerprint": record["snapshot_fingerprint"],
        "snapshot_bytes": record["snapshot_bytes"],
        "snapshot_identity": record["snapshot_identity"],
        "snapshot": record["snapshot"],
        "scratch_plan": record["scratch_plan"],
    }
    if common:
        require(fields == common, "common identity/logits/snapshot drift")
    else:
        common.update(fields)


def validate_setup(
    record: Any,
    prefix: int,
    chunk: int,
    build: dict[str, str],
    physical: int,
    *,
    preflight: bool = False,
) -> dict[str, Any]:
    exact_keys(record, SETUP_KEYS, "setup record schema drift")
    require(
        record["schema_version"] == SCHEMA_VERSION and record["kind"] == "setup",
        "setup identity drift",
    )
    rust_build = record["build_identity"]
    if rust_build != build:
        exact_keys(
            rust_build,
            {
                "schema_version",
                "build_commit",
                "build_commit_short",
                "build_dirty",
                "build_source_state",
                "stamp_source",
                "stamp_error",
                "runtime_commit",
                "runtime_dirty",
                "runtime_source_state",
                "status",
                "problems",
                "overrides",
            },
            "Rust build identity schema drift",
        )
        require(
            rust_build["schema_version"] == 2
            and rust_build["build_commit"] == build["head"]
            and rust_build["runtime_commit"] == build["head"]
            and rust_build["build_dirty"] is False
            and rust_build["runtime_dirty"] is False
            and rust_build["status"] == "match"
            and rust_build["problems"] == []
            and rust_build["overrides"] == [],
            "Rust build identity drift",
        )
    require(
        record["prefix"] == prefix and record["chunk"] == chunk, "setup geometry drift"
    )
    require(record["capacity"] == prefix + chunk + 8, "setup capacity drift")
    require(
        record["model"] == MODEL and record["model_bytes"] == MODEL_BYTES,
        "model identity drift",
    )
    require(
        record["model_sha256"] == MODEL_SHA256
        and record["model_sha256_after_load"] == MODEL_SHA256,
        "model digest drift",
    )
    for value in [
        record["model_file_identity"],
        record["model_path_identity_after_load"],
    ]:
        exact_keys(value, FILE_IDENTITY_KEYS, "model file identity schema drift")
    require(
        record["model_file_identity"] == record["model_path_identity_after_load"],
        "model file identity changed",
    )
    require(
        record["model_file_identity"]["bytes"] == MODEL_BYTES, "model stat bytes drift"
    )
    for key in ["prefix_token_sha256", "suffix_token_sha256"]:
        digest(record[key], f"{key} invalid")
    finite_positive(record["prefix_gpu_ms"], "prefix GPU time invalid")
    validate_plan(record["prefix_plan"], 1024, prefix, 1024)
    validate_plan(record["suffix_plan"], chunk, prefix + chunk, chunk)
    validate_snapshot(record["snapshot"], prefix)
    require(record["physical_memory_bytes"] == physical, "physical memory drift")
    require(record["peak_bound"] <= physical, "memory admission failed")
    require(record["preflight_only"] is preflight, "setup preflight flag drift")
    for key in [
        "max_buffer_length",
        "correctness_snapshot_bytes",
        "session_allocation_delta",
        "allocated_before_prefix",
        "allocated_after_builder_drop",
        "peak_bound",
        "allocated_before_x",
        "allocated_after_x",
        "allocated_after_y",
    ]:
        nonnegative_int(record[key], f"setup {key} invalid")
    require(
        record["max_buffer_length"] > 0 and record["session_allocation_delta"] > 0,
        "setup limits invalid",
    )
    largest = max(
        (
            item["logical_bytes"]
            for name in ["allocations", "deferred_allocations"]
            for item in record["suffix_plan"][name]
        ),
        default=0,
    )
    require(
        largest <= record["max_buffer_length"], "planned buffer exceeds maxBufferLength"
    )
    require(
        record["allocated_before_x"]
        <= record["allocated_after_x"]
        <= record["allocated_after_y"],
        "scratch allocation accounting drift",
    )
    x_ids, y_ids = record["scratch_x_buffer_ids"], record["scratch_y_buffer_ids"]
    require(
        isinstance(x_ids, list) and isinstance(y_ids, list) and x_ids and y_ids,
        "scratch buffer IDs missing",
    )
    require(
        len(x_ids) == len(set(x_ids))
        and len(y_ids) == len(set(y_ids))
        and set(x_ids).isdisjoint(y_ids),
        "scratch banks alias",
    )
    return record["snapshot"]


def validate_correctness(
    record: Any, prefix: int, chunk: int, common: dict[str, Any]
) -> None:
    keys = {
        "schema_version",
        "kind",
        "prefix",
        "chunk",
        "match",
        "suffix_logits_sha256",
        "state",
        "continuation",
    }
    exact_keys(record, keys, "correctness record schema drift")
    require(
        record["schema_version"] == SCHEMA_VERSION and record["kind"] == "correctness",
        "correctness identity drift",
    )
    require(
        record["prefix"] == prefix
        and record["chunk"] == chunk
        and record["match"] is True,
        "correctness mismatch",
    )
    require(
        record["suffix_logits_sha256"] == common["logits_sha256"],
        "correctness logits drift",
    )
    validate_snapshot(record["state"], prefix + chunk)
    continuation = record["continuation"]
    exact_keys(
        continuation, {"token_ids", "logits_sha256"}, "continuation schema drift"
    )
    require(
        isinstance(continuation["token_ids"], list)
        and len(continuation["token_ids"]) == 8
        and all(
            isinstance(value, int) and value >= 0 for value in continuation["token_ids"]
        ),
        "continuation token drift",
    )
    require(
        isinstance(continuation["logits_sha256"], list)
        and len(continuation["logits_sha256"]) == 8,
        "continuation logits count drift",
    )
    for value in continuation["logits_sha256"]:
        digest(value, "continuation logits digest invalid")
    require(
        continuation["token_ids"][0] == common["first_token"],
        "continuation first token drift",
    )


def records_from_stdout(execution: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for line in (
        stream_bytes(execution["stdout"]).decode("utf-8", errors="strict").splitlines()
    ):
        if "VT_PRODUCT_JSON " in line:
            records.append(json.loads(line.split("VT_PRODUCT_JSON ", 1)[1]))
    return records


def validate_child(
    child: Any, prefix: int, chunk: int, packet: dict[str, Any]
) -> list[dict[str, Any]]:
    exact_keys(child, CHILD_KEYS, "child schema drift")
    require(
        child["prefix"] == prefix and child["chunk"] == chunk, "cell chronology drift"
    )
    require(child["state"] == "parsed", "child was not parsed")
    identity = packet["expected_identity"]
    require(
        child["identity_before"] == identity and child["identity_after"] == identity,
        "child binary identity drift",
    )
    physical = packet["environment_before"]["physical_memory_bytes"]
    command = expected_command(identity["binary_path"], prefix, chunk, physical)
    require(child["command"] == command, "child command drift")
    validate_execution(child["execution"], command, packet["environment_removed"])
    require(child["pid"] == child["execution"]["pid"], "child PID drift")
    require(
        isinstance(child["spawned_at"], str) and child["spawned_at"],
        "spawn timestamp invalid",
    )
    authenticated_records = records_from_stdout(child["execution"])
    records = child["records"]
    require(
        isinstance(records, list) and len(records) == 21, "record cardinality drift"
    )
    setup_snapshot = validate_setup(
        records[0], prefix, chunk, packet["build_identity"], physical
    )
    common: dict[str, Any] = {}
    validate_arm(
        records[1], prefix, chunk, "correctness_arm", "A", "X", 0, "AB", 1, common
    )
    validate_arm(
        records[2], prefix, chunk, "correctness_arm", "B", "Y", 0, "AB", 2, common
    )
    require(common["snapshot"] == setup_snapshot, "setup/arm snapshot drift")
    validate_correctness(records[3], prefix, chunk, common)
    for offset, (role, bank) in enumerate(WARMUP, start=3):
        validate_arm(
            records[offset + 1],
            prefix,
            chunk,
            "warmup",
            role,
            bank,
            0,
            "W",
            offset,
            common,
        )
    cursor = 8
    for pair, (first, second, order) in enumerate(SCHEDULE, start=1):
        for role, bank in [first, second]:
            validate_arm(
                records[cursor],
                prefix,
                chunk,
                "arm",
                role,
                bank,
                pair,
                order,
                cursor - 1,
                common,
            )
            cursor += 1
    complete = records[20]
    exact_keys(
        complete,
        {"schema_version", "kind", "prefix", "chunk", "expected_logits_sha256"},
        "complete record schema drift",
    )
    require(
        complete
        == {
            "schema_version": SCHEMA_VERSION,
            "kind": "complete",
            "prefix": prefix,
            "chunk": chunk,
            "expected_logits_sha256": common["logits_sha256"],
        },
        "complete record drift",
    )
    require(
        authenticated_records == records,
        "parsed records drift from authenticated stdout",
    )
    return records[8:20]


def endpoint_summary(arms: list[dict[str, Any]], endpoint: str) -> dict[str, Any]:
    pairs = []
    for pair in range(1, 7):
        selected = [record for record in arms if record["pair"] == pair]
        require(len(selected) == 2, "pair cardinality drift")
        by_role = {record["role"]: record for record in selected}
        require(set(by_role) == {"A", "B"}, "pair role drift")
        a, b = float(by_role["A"][endpoint]), float(by_role["B"][endpoint])
        pairs.append(
            {
                "pair": pair,
                "order": by_role["A"]["order"],
                "a_ms": a,
                "b_ms": b,
                "saving_ms": a - b,
                "ratio": a / b,
                "win": a > b,
            }
        )
    a_values = [pair["a_ms"] for pair in pairs]
    b_values = [pair["b_ms"] for pair in pairs]
    savings = [pair["saving_ms"] for pair in pairs]
    strata = {}
    for order in ["AB", "BA"]:
        selected = [pair for pair in pairs if pair["order"] == order]
        strata[order] = {
            "savings_ms": [pair["saving_ms"] for pair in selected],
            "median_saving_ms": median([pair["saving_ms"] for pair in selected]),
            "ratios": [pair["ratio"] for pair in selected],
            "wins": sum(pair["win"] for pair in selected),
        }
    return {
        "a_ms": a_values,
        "b_ms": b_values,
        "paired_savings_ms": savings,
        "ratios": [pair["ratio"] for pair in pairs],
        "wins": sum(pair["win"] for pair in pairs),
        "median_a_ms": median(a_values),
        "median_b_ms": median(b_values),
        "median_saving_ms": median(savings),
        "median_ratio": median([pair["ratio"] for pair in pairs]),
        "strata": strata,
        "pairs": pairs,
    }


def validate_safety(
    safety: Any,
    cells: list[dict[str, Any]],
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    keys = {
        "p8192_c128_a_suffix_wall_ms",
        "p8192_c128_a_suffix_wall_median_ms",
        "p8192_c128_a_suffix_wall_max_ms",
        "run_p16384_c128",
        "p16384_c128_a_suffix_wall_ms",
        "p16384_c128_a_suffix_wall_median_ms",
        "p16384_c128_a_suffix_wall_max_ms",
        "memory_pressure_normal",
        "swap_not_increased",
        "run_p16384_c1024",
    }
    exact_keys(safety, keys, "safety schema drift")
    first = cells[0]["endpoints"]["suffix_wall_ms"]["a_ms"]
    second = cells[1]["endpoints"]["suffix_wall_ms"]["a_ms"]
    require(safety["p8192_c128_a_suffix_wall_ms"] == first, "P8K safety samples drift")
    require(
        safety["p8192_c128_a_suffix_wall_median_ms"] == median(first),
        "P8K safety median drift",
    )
    require(
        safety["p8192_c128_a_suffix_wall_max_ms"] == max(first),
        "P8K safety maximum drift",
    )
    run_second = max(first) <= 10_000.0 and 2.0 * median(first) <= 8_000.0
    require(
        safety["run_p16384_c128"] is run_second and run_second,
        "P16K/C128 safety decision drift",
    )
    require(
        safety["p16384_c128_a_suffix_wall_ms"] == second, "P16K safety samples drift"
    )
    require(
        safety["p16384_c128_a_suffix_wall_median_ms"] == median(second),
        "P16K safety median drift",
    )
    require(
        safety["p16384_c128_a_suffix_wall_max_ms"] == max(second),
        "P16K safety maximum drift",
    )
    pressure = before["memory_pressure"] == after["memory_pressure"] == "normal"
    swap = after["swap_bytes"] <= before["swap_bytes"]
    require(
        safety["memory_pressure_normal"] is pressure, "memory-pressure safety drift"
    )
    require(safety["swap_not_increased"] is swap, "swap safety drift")
    run_third = (
        max(second) <= 10_000.0
        and 8.0 * median(second) <= 40_000.0
        and pressure
        and swap
    )
    require(
        safety["run_p16384_c1024"] is run_third and run_third,
        "P16K/C1024 safety decision drift",
    )


def decide(cells: list[dict[str, Any]]) -> tuple[str, dict[str, bool]]:
    regressions = []
    for cell in cells:
        for endpoint in ENDPOINTS[:3]:
            summary = cell["endpoints"][endpoint]
            regression = -summary["median_saving_ms"]
            regressions.append(regression <= max(1.0, 0.02 * summary["median_a_ms"]))
    if not all(regressions):
        return "KILL", {"no_immediate_regression": False}
    gates: dict[str, bool] = {"no_immediate_regression": True}
    for index, cell in enumerate(cells):
        wall = cell["endpoints"]["suffix_wall_ms"]
        gpu = cell["endpoints"]["suffix_gpu_ms"]
        gates[f"cell_{index + 1}_all_suffix_wins"] = wall["wins"] == gpu["wins"] == 6
        gates[f"cell_{index + 1}_positive_suffix_strata"] = all(
            wall["strata"][order]["median_saving_ms"] > 0
            and gpu["strata"][order]["median_saving_ms"] > 0
            for order in ["AB", "BA"]
        )
        threshold = 500.0 if index == 0 else 1000.0
        gates[f"cell_{index + 1}_suffix_median_threshold"] = (
            wall["median_saving_ms"] >= threshold
            and gpu["median_saving_ms"] >= threshold
        )
    for index in [1, 2]:
        ttft = cells[index]["endpoints"]["post_restore_ttft_ms"]
        request = cells[index]["endpoints"]["request_wall_ms"]
        gates[f"cell_{index + 1}_ttft"] = (
            ttft["wins"] >= 5
            and ttft["median_saving_ms"] >= 750.0
            and all(
                ttft["strata"][order]["median_saving_ms"] > 0 for order in ["AB", "BA"]
            )
        )
        gates[f"cell_{index + 1}_request"] = request["wins"] >= 5 and all(
            request["strata"][order]["median_saving_ms"] > 0 for order in ["AB", "BA"]
        )
    p8_request = cells[0]["endpoints"]["request_wall_ms"]
    gates["p8_request_nonregression"] = -p8_request["median_saving_ms"] <= max(
        100.0, 0.02 * p8_request["median_a_ms"]
    )
    return ("PROMOTE_DEFAULT" if all(gates.values()) else "KEEP_DEFAULT_OFF"), gates


def _analyze_legacy_packet(packet: Any) -> dict[str, Any]:
    exact_keys(packet, TOP_KEYS, "chronology top-level schema drift")
    require(
        packet["schema_version"] == SCHEMA_VERSION, "chronology schema version drift"
    )
    require(
        packet["preregistration_sha256"] == PREREGISTRATION_SHA256,
        "preregistration digest drift",
    )
    require(
        packet["tracked_tree_clean"] is True,
        "implementation tree was not tracked-clean",
    )
    validate_identity(packet["expected_identity"])
    identity = packet["expected_identity"]
    require(
        packet["build_identity"]
        == {"head": identity["head"], "tree": identity["tree"]},
        "build identity drift",
    )
    require(
        packet["binary_path"] == identity["binary_path"]
        and packet["binary_sha256"] == identity["binary_sha256"],
        "binary provenance drift",
    )
    removed = packet["environment_removed"]
    require(
        isinstance(removed, list)
        and removed == sorted(set(removed))
        and all(
            isinstance(name, str)
            and (name == "RUST_LOG" or name.startswith(("QWEN", "MTL", "METAL")))
            for name in removed
        ),
        "removed environment provenance drift",
    )
    validate_environment(packet["environment_before"], identity)
    validate_environment(packet["environment_after"], identity)
    require(
        packet["environment_before"]["physical_memory_bytes"]
        == packet["environment_after"]["physical_memory_bytes"],
        "physical memory drift",
    )
    require(
        packet["environment_after"]["swap_bytes"]
        <= packet["environment_before"]["swap_bytes"],
        "swap increased",
    )
    require(
        isinstance(packet["started_at"], str)
        and packet["started_at"]
        and isinstance(packet["finished_at"], str)
        and packet["finished_at"],
        "campaign timestamps invalid",
    )
    require(
        isinstance(packet["children"], list) and len(packet["children"]) == 3,
        "campaign cell cardinality drift",
    )
    cells = []
    common_route = None
    for child, (prefix, chunk) in zip(packet["children"], CELLS, strict=True):
        arms = validate_child(child, prefix, chunk, packet)
        setup_record = child["records"][0]
        route = {
            "device": setup_record["device"],
            "device_registry_id": setup_record["device_registry_id"],
            "max_buffer_length": setup_record["max_buffer_length"],
            "model_file_identity": setup_record["model_file_identity"],
            "snapshot_identity": child["records"][1]["snapshot_identity"],
        }
        if common_route is None:
            common_route = route
        else:
            require(route == common_route, "cross-cell route/identity drift")
        cells.append(
            {
                "cell": f"P{prefix}/C{chunk}",
                "prefix": prefix,
                "chunk": chunk,
                "endpoints": {
                    endpoint: endpoint_summary(arms, endpoint) for endpoint in ENDPOINTS
                },
            }
        )
    validate_safety(
        packet["safety"],
        cells,
        packet["environment_before"],
        packet["environment_after"],
    )
    disposition, gates = decide(cells)
    return {
        "schema_version": SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "implementation_commit": identity["head"],
        "implementation_tree": identity["tree"],
        "binary_sha256": identity["binary_sha256"],
        "cells": cells,
        "gates": gates,
        "disposition": disposition,
    }


COLLECTOR_TOP_KEYS = {
    "schema_version",
    "campaign",
    "started_at",
    "preregistration",
    "source",
    "model",
    "binary",
    "expected_identity",
    "build",
    "sanitized_child_environment",
    "repairable_preflight",
    "environment_before",
    "children",
    "safety",
    "environment_after",
    "completed_at",
    "state",
}
COMMAND_PACKET_KEYS = {
    "argv",
    "pid",
    "started_at",
    "completed_at",
    "elapsed_seconds",
    "returncode",
    "termination_signal",
    "stdout",
    "stderr",
}
COLLECTOR_CHILD_KEYS = {
    "cell",
    "preflight_only",
    "state",
    "argv",
    "environment",
    "identity_before",
    "prepared_at",
    "pid",
    "started_at",
    "process_identity",
    "completed_at",
    "elapsed_seconds",
    "returncode",
    "termination_signal",
    "stdout",
    "stderr",
    "identity_after",
    "records",
}


def collector_stream_bytes(value: Any) -> bytes:
    exact_keys(value, {"base64", "bytes", "sha256"}, "collector stream schema drift")
    require(
        isinstance(value["bytes"], int) and value["bytes"] >= 0,
        "collector stream bytes invalid",
    )
    digest(value["sha256"], "collector stream digest invalid")
    try:
        raw = base64.b64decode(value["base64"], validate=True)
    except Exception as error:
        raise ValueError("collector stream base64 invalid") from error
    require(len(raw) == value["bytes"], "collector stream length drift")
    require(
        hashlib.sha256(raw).hexdigest() == value["sha256"],
        "collector stream hash drift",
    )
    return raw


def validate_command_packet(value: Any, argv: list[str] | None = None) -> None:
    exact_keys(value, COMMAND_PACKET_KEYS, "command packet schema drift")
    require(
        isinstance(value["argv"], list)
        and all(isinstance(item, str) for item in value["argv"]),
        "command argv invalid",
    )
    if argv is not None:
        require(value["argv"] == argv, "command argv drift")
    require(isinstance(value["pid"], int) and value["pid"] > 0, "command PID invalid")
    finite_positive(value["elapsed_seconds"], "command elapsed time invalid")
    require(isinstance(value["returncode"], int), "command return status invalid")
    expected_signal = -value["returncode"] if value["returncode"] < 0 else None
    require(
        value["termination_signal"] == expected_signal,
        "command termination signal drift",
    )
    collector_stream_bytes(value["stdout"])
    collector_stream_bytes(value["stderr"])


def validate_binary(value: Any) -> None:
    exact_keys(
        value,
        {"canonical_path", "bytes", "device", "inode", "mode", "sha256"},
        "binary schema drift",
    )
    require(
        isinstance(value["canonical_path"], str) and value["canonical_path"],
        "binary path invalid",
    )
    for key in ["bytes", "device", "inode", "mode"]:
        require(isinstance(value[key], int) and value[key] > 0, f"binary {key} invalid")
    digest(value["sha256"], "binary digest invalid")


def validate_source_identity(
    value: Any, source: dict[str, Any], binary: dict[str, Any]
) -> None:
    exact_keys(
        value, {"head", "tree", "untracked", "binary"}, "source identity schema drift"
    )
    require(
        value["head"] == source["head"] and value["tree"] == source["tree"],
        "source object identity drift",
    )
    require(
        value["untracked"] == source["allowed_untracked"], "untracked inventory drift"
    )
    require(value["binary"] == binary, "source binary identity drift")


def validate_system_snapshot(value: Any) -> None:
    exact_keys(
        value,
        {
            "captured_at",
            "commands",
            "physical_memory_bytes",
            "ac_power",
            "memory_pressure_normal",
            "swap_used_bytes",
        },
        "system snapshot schema drift",
    )
    expected_commands = {
        "os": ["sw_vers"],
        "uname": ["uname", "-a"],
        "physical_memory": ["sysctl", "-n", "hw.memsize"],
        "ac_power": ["pmset", "-g", "batt"],
        "memory_pressure": ["memory_pressure", "-Q"],
        "swap": ["sysctl", "-n", "vm.swapusage"],
        "thermal": ["pmset", "-g", "therm"],
    }
    require(
        set(value["commands"]) == set(expected_commands), "system command set drift"
    )
    for name, argv in expected_commands.items():
        validate_command_packet(value["commands"][name], argv)
        require(
            value["commands"][name]["returncode"] == 0, f"system command failed: {name}"
        )
    require(
        isinstance(value["physical_memory_bytes"], int)
        and value["physical_memory_bytes"] > 0,
        "physical memory invalid",
    )
    require(value["ac_power"] is True, "AC power absent")
    require(value["memory_pressure_normal"] is True, "memory pressure was not normal")
    nonnegative_int(value["swap_used_bytes"], "swap usage invalid")


def validate_process_guard(value: Any) -> None:
    exact_keys(
        value,
        {"query", "protected_pid", "competing_qwen_model_processes"},
        "process guard schema drift",
    )
    validate_command_packet(
        value["query"], ["ps", "-axo", "pid=,ppid=,state=,command=", "-ww"]
    )
    require(value["query"]["returncode"] == 0, "process guard query failed")
    require(value["protected_pid"] is None, "protected PID 8770 active")
    require(
        value["competing_qwen_model_processes"] == [], "competing model process present"
    )


def validate_provenance(value: Any) -> None:
    exact_keys(
        value, {"captured_at", "process_guard", "system"}, "provenance schema drift"
    )
    validate_process_guard(value["process_guard"])
    validate_system_snapshot(value["system"])


def collector_records(execution: dict[str, Any]) -> list[dict[str, Any]]:
    stderr = collector_stream_bytes(execution["stderr"])
    require(b"VT_PRODUCT_JSON " not in stderr, "Rust record appeared on stderr")
    records = []
    for line in collector_stream_bytes(execution["stdout"]).splitlines():
        require(
            not (
                b"VT_PRODUCT_JSON " in line and not line.startswith(b"VT_PRODUCT_JSON ")
            ),
            "Rust marker position drift",
        )
        if line.startswith(b"VT_PRODUCT_JSON "):
            records.append(json.loads(line[len(b"VT_PRODUCT_JSON ") :]))
    return records


def normalized_identity(
    source: dict[str, Any], binary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "head": source["head"],
        "tree": source["tree"],
        "tracked_clean": True,
        "binary_path": binary["canonical_path"],
        "binary_sha256": binary["sha256"],
    }


def validate_collector_child(
    child: Any, prefix: int, chunk: int, packet: dict[str, Any], *, preflight: bool
) -> list[dict[str, Any]]:
    exact_keys(child, COLLECTOR_CHILD_KEYS, "child schema drift")
    require(
        child["cell"] == f"P{prefix}/C{chunk}" and child["preflight_only"] is preflight,
        "cell chronology drift",
    )
    require(child["state"] == "validated", "child was not validated")
    validate_source_identity(
        child["identity_before"], packet["source"], packet["binary"]
    )
    validate_source_identity(
        child["identity_after"], packet["source"], packet["binary"]
    )
    physical = packet["environment_before"]["system"]["physical_memory_bytes"]
    argv = expected_command(
        packet["binary"]["canonical_path"], prefix, chunk, physical
    ) + (["--preflight-only"] if preflight else [])
    require(child["argv"] == argv, "child argv drift")
    exact_keys(
        child["environment"],
        {"explicit", "removed_inherited_names", "forbidden_present_after_sanitization"},
        "child environment schema drift",
    )
    require(
        child["environment"] == packet["sanitized_child_environment"],
        "child environment drift",
    )
    require(
        child["environment"]["forbidden_present_after_sanitization"] == [],
        "forbidden child environment present",
    )
    require(
        child["environment"]["explicit"].get("LANG") == "C"
        and child["environment"]["explicit"].get("LC_ALL") == "C",
        "child locale drift",
    )
    require(
        child["pid"] == child["process_identity"]["pid"],
        "child process identity PID drift",
    )
    exact_keys(
        child["process_identity"], {"query", "pid"}, "process identity schema drift"
    )
    validate_command_packet(
        child["process_identity"]["query"],
        [
            "ps",
            "-p",
            str(child["pid"]),
            "-o",
            "pid=,ppid=,state=,lstart=,command=",
            "-ww",
        ],
    )
    require(
        child["returncode"] == 0 and child["termination_signal"] is None,
        "child process failed",
    )
    finite_positive(child["elapsed_seconds"], "child elapsed time invalid")
    authenticated = collector_records(child)
    require(
        authenticated == child["records"],
        "parsed records drift from authenticated stdout",
    )
    if preflight:
        require(
            [record.get("kind") for record in child["records"]]
            == [
                "setup",
                "correctness_arm",
                "correctness_arm",
                "correctness",
                "preflight_complete",
            ],
            "preflight record chronology drift",
        )
        setup_snapshot = validate_setup(
            child["records"][0],
            prefix,
            chunk,
            {"head": packet["source"]["head"], "tree": packet["source"]["tree"]},
            physical,
            preflight=True,
        )
        common: dict[str, Any] = {}
        validate_arm(
            child["records"][1],
            prefix,
            chunk,
            "correctness_arm",
            "A",
            "X",
            0,
            "AB",
            1,
            common,
        )
        validate_arm(
            child["records"][2],
            prefix,
            chunk,
            "correctness_arm",
            "B",
            "Y",
            0,
            "AB",
            2,
            common,
        )
        require(common["snapshot"] == setup_snapshot, "setup/arm snapshot drift")
        validate_correctness(child["records"][3], prefix, chunk, common)
        complete = child["records"][4]
        exact_keys(
            complete,
            {"schema_version", "kind", "prefix", "chunk", "expected_logits_sha256"},
            "preflight complete schema drift",
        )
        require(
            complete["kind"] == "preflight_complete"
            and complete["expected_logits_sha256"] == common["logits_sha256"],
            "preflight complete drift",
        )
        return []
    identity = normalized_identity(packet["source"], packet["binary"])
    converted_stream = lambda value: {
        "base64": value["base64"],
        "sha256": value["sha256"],
        "length": value["bytes"],
    }
    normalized = {
        "prefix": prefix,
        "chunk": chunk,
        "state": "parsed",
        "command": argv,
        "identity_before": identity,
        "pid": child["pid"],
        "spawned_at": child["started_at"],
        "execution": {
            "command": argv,
            "environment_overrides": {},
            "environment_removed": child["environment"]["removed_inherited_names"],
            "pid": child["pid"],
            "started_at": child["started_at"],
            "finished_at": child["completed_at"],
            "elapsed_s": child["elapsed_seconds"],
            "returncode": child["returncode"],
            "termination_signal": child["termination_signal"],
            "spawn_callback_error": None,
            "stdout": converted_stream(child["stdout"]),
            "stderr": converted_stream(child["stderr"]),
        },
        "identity_after": identity,
        "records": child["records"],
    }
    normalized_packet = {
        "expected_identity": identity,
        "environment_before": {"physical_memory_bytes": physical},
        "environment_removed": child["environment"]["removed_inherited_names"],
        "build_identity": {
            "head": packet["source"]["head"],
            "tree": packet["source"]["tree"],
        },
    }
    return validate_child(normalized, prefix, chunk, normalized_packet)


def validate_collector_safety(value: Any, timed_children: list[dict[str, Any]]) -> None:
    exact_keys(
        value, {"before_p16384_c128", "before_p16384_c1024"}, "safety schema drift"
    )
    first = value["before_p16384_c128"]
    exact_keys(
        first,
        {
            "predicate",
            "source_cell_valid",
            "samples_ms",
            "median_ms",
            "maximum_ms",
            "allowed",
            "recorded_at",
        },
        "P16K/C128 safety schema drift",
    )
    first_samples = [
        record["suffix_wall_ms"]
        for record in timed_children[0]["records"]
        if record.get("kind") == "arm" and record.get("role") == "A"
    ]
    require(
        first["source_cell_valid"] is True and first["samples_ms"] == first_samples,
        "P8K safety samples drift",
    )
    require(
        first["median_ms"] == median(first_samples)
        and first["maximum_ms"] == max(first_samples),
        "P8K safety statistics drift",
    )
    require(
        first["allowed"]
        is (max(first_samples) <= 10_000 and 2 * median(first_samples) <= 8_000)
        and first["allowed"],
        "P16K/C128 safety decision drift",
    )
    second = value["before_p16384_c1024"]
    exact_keys(
        second,
        {
            "predicate",
            "source_cell_valid",
            "samples_ms",
            "median_ms",
            "maximum_ms",
            "timing_allowed",
            "environment_before_admission",
            "environment_after_admission",
            "fresh_admission_ok",
            "allowed",
            "state",
            "recorded_at",
            "applied_at",
        },
        "P16K/C1024 safety schema drift",
    )
    second_samples = [
        record["suffix_wall_ms"]
        for record in timed_children[1]["records"]
        if record.get("kind") == "arm" and record.get("role") == "A"
    ]
    timing = max(second_samples) <= 10_000 and 8 * median(second_samples) <= 40_000
    require(
        second["source_cell_valid"] is True and second["samples_ms"] == second_samples,
        "P16K safety samples drift",
    )
    require(
        second["median_ms"] == median(second_samples)
        and second["maximum_ms"] == max(second_samples)
        and second["timing_allowed"] is timing,
        "P16K safety statistics drift",
    )
    validate_system_snapshot(second["environment_before_admission"])
    validate_system_snapshot(second["environment_after_admission"])
    admission = (
        second["environment_after_admission"]["swap_used_bytes"]
        <= second["environment_before_admission"]["swap_used_bytes"]
    )
    require(
        second["fresh_admission_ok"] is admission
        and second["allowed"] is (timing and admission)
        and second["allowed"]
        and second["state"] == "applied",
        "P16K/C1024 safety decision drift",
    )


def analyze_collector_packet(packet: Any) -> dict[str, Any]:
    exact_keys(packet, COLLECTOR_TOP_KEYS, "chronology top-level schema drift")
    require(
        packet["schema_version"] == 1
        and packet["campaign"] == "qwen-vt-product-ab"
        and packet["state"] == "completed",
        "chronology identity/state drift",
    )
    require(
        isinstance(packet["started_at"], str)
        and packet["started_at"]
        and isinstance(packet["completed_at"], str)
        and packet["completed_at"],
        "campaign timestamps invalid",
    )
    require(
        packet["preregistration"]
        == {
            "path": "docs/bench/2026-08-17-qwen-vt-product-ab/README.md",
            "sha256": PREREGISTRATION_SHA256,
        },
        "preregistration provenance drift",
    )
    source = packet["source"]
    exact_keys(
        source,
        {"head", "tree", "tracked_clean", "allowed_untracked"},
        "source schema drift",
    )
    object_id(source["head"], "implementation HEAD invalid")
    object_id(source["tree"], "implementation tree invalid")
    require(
        source["tracked_clean"] is True
        and isinstance(source["allowed_untracked"], list),
        "source cleanliness drift",
    )
    inventory = source["allowed_untracked"]
    require(
        inventory == sorted(inventory, key=lambda item: item.get("path", "")),
        "untracked inventory order drift",
    )
    for item in inventory:
        exact_keys(
            item, {"path", "bytes", "sha256"}, "untracked inventory schema drift"
        )
        require(
            isinstance(item["path"], str) and item["path"], "untracked path invalid"
        )
        nonnegative_int(item["bytes"], "untracked file bytes invalid")
        digest(item["sha256"], "untracked file digest invalid")
    validate_binary(packet["binary"])
    validate_source_identity(packet["expected_identity"], source, packet["binary"])
    exact_keys(
        packet["model"],
        {"canonical_path", "bytes", "sha256", "device", "inode", "modified_ns"},
        "model provenance schema drift",
    )
    require(
        packet["model"]["canonical_path"] == MODEL
        and packet["model"]["bytes"] == MODEL_BYTES
        and packet["model"]["sha256"] == MODEL_SHA256,
        "model provenance drift",
    )
    build = dict(packet["build"])
    removed = build.pop("environment_removed", None)
    validate_command_packet(
        build, ["cargo", "build", "--release", "-p", "qwen-cli", "--bin", "qwen-bench"]
    )
    require(
        build["returncode"] == 0
        and removed == packet["sanitized_child_environment"]["removed_inherited_names"],
        "build provenance drift",
    )
    validate_provenance(packet["environment_before"])
    validate_provenance(packet["environment_after"])
    require(
        packet["environment_after"]["system"]["swap_used_bytes"]
        <= packet["environment_before"]["system"]["swap_used_bytes"],
        "campaign swap increased",
    )
    preflight = packet["repairable_preflight"]
    exact_keys(
        preflight,
        {"environment_before", "execution", "environment_after"},
        "repairable preflight schema drift",
    )
    validate_system_snapshot(preflight["environment_before"])
    validate_system_snapshot(preflight["environment_after"])
    require(
        preflight["environment_after"]["swap_used_bytes"]
        <= preflight["environment_before"]["swap_used_bytes"],
        "repairable preflight swap increased",
    )
    execution = preflight["execution"]
    exact_keys(
        execution,
        COMMAND_PACKET_KEYS | {"process_identity", "records"},
        "repairable preflight execution schema drift",
    )
    # The pre-chronology execution has the same authenticated record contract.
    faux = {
        "cell": "P8192/C128",
        "preflight_only": True,
        "state": "validated",
        "argv": execution["argv"],
        "environment": packet["sanitized_child_environment"],
        "identity_before": packet["expected_identity"],
        "prepared_at": execution["started_at"],
        "pid": execution["pid"],
        "started_at": execution["started_at"],
        "process_identity": execution["process_identity"],
        "completed_at": execution["completed_at"],
        "elapsed_seconds": execution["elapsed_seconds"],
        "returncode": execution["returncode"],
        "termination_signal": execution["termination_signal"],
        "stdout": execution["stdout"],
        "stderr": execution["stderr"],
        "identity_after": packet["expected_identity"],
        "records": execution["records"],
    }
    validate_collector_child(faux, 8192, 128, packet, preflight=True)
    require(len(packet["children"]) == 4, "campaign child chronology drift")
    expected = [
        (8192, 128, False),
        (16384, 128, False),
        (16384, 1024, True),
        (16384, 1024, False),
    ]
    timed_children = []
    arms_by_cell = []
    common_route = None
    for child, (prefix, chunk, is_preflight) in zip(
        packet["children"], expected, strict=True
    ):
        arms = validate_collector_child(
            child, prefix, chunk, packet, preflight=is_preflight
        )
        if not is_preflight:
            setup_record = child["records"][0]
            route = {
                "device": setup_record["device"],
                "device_registry_id": setup_record["device_registry_id"],
                "max_buffer_length": setup_record["max_buffer_length"],
                "model_file_identity": setup_record["model_file_identity"],
                "snapshot_identity": child["records"][1]["snapshot_identity"],
            }
            if common_route is None:
                common_route = route
            else:
                require(route == common_route, "cross-cell route/identity drift")
            model_file = setup_record["model_file_identity"]
            require(
                model_file["device"] == packet["model"]["device"]
                and model_file["inode"] == packet["model"]["inode"]
                and model_file["bytes"] == packet["model"]["bytes"]
                and model_file["modified_seconds"] * 1_000_000_000
                + model_file["modified_nanoseconds"]
                == packet["model"]["modified_ns"],
                "Rust/collector model stat drift",
            )
            timed_children.append(child)
            arms_by_cell.append(arms)
    validate_collector_safety(packet["safety"], timed_children)
    cells = [
        {
            "cell": f"P{prefix}/C{chunk}",
            "prefix": prefix,
            "chunk": chunk,
            "endpoints": {
                endpoint: endpoint_summary(arms, endpoint) for endpoint in ENDPOINTS
            },
        }
        for (prefix, chunk), arms in zip(CELLS, arms_by_cell, strict=True)
    ]
    disposition, gates = decide(cells)
    return {
        "schema_version": 1,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "implementation_commit": source["head"],
        "implementation_tree": source["tree"],
        "binary_sha256": packet["binary"]["sha256"],
        "cells": cells,
        "gates": gates,
        "disposition": disposition,
    }


def analyze_packet(packet: Any) -> dict[str, Any]:
    if isinstance(packet, dict) and "campaign" in packet:
        return analyze_collector_packet(packet)
    return _analyze_legacy_packet(packet)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: analyze.py INPUT OUTPUT")
    source, destination = map(Path, sys.argv[1:])
    packet = json.loads(source.read_text(encoding="utf-8"))
    rendered = (
        json.dumps(analyze_packet(packet), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    destination.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
