#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Strict reducer for qwen.dflash_e0_lockstep schema-v1 JSONL."""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "qwen.dflash_e0_lockstep"
VERSION = 1
REDUCTION_SCHEMA = "qwen.dflash_e0_evidence_reduction"
REDUCTION_VERSION = 1
SAMPLER_VERSION = 1
SEMANTICS = "token_major_serial_vs_multi_hidden_lockstep_development_only"
HIDDEN_SEMANTICS = "poison_then_capture_exact_source_to_active_dflash_context_row"
AUTHORITY = "development_only_no_product_authority"
BINDING_MANIFEST_SCHEMA = "qwen.dflash_e0_binding_manifest"
BINDING_MANIFEST_VERSION = 1
SCRIPT = Path(__file__).resolve()
MASK64 = (1 << 64) - 1

BOOT_COMMON = {
    "schema",
    "schema_version",
    "run_id",
    "build_identity",
    "lease_env",
    "command",
    "paths",
    "event",
    "payload",
}
RUN_COMMON = BOOT_COMMON | {
    "classification",
    "config",
    "assets",
    "binding",
    "binding_manifest",
    "host",
    "prompt",
    "sessions",
}
PAYLOAD_KEYS = {
    "bootstrap_start": {"started_utc"},
    "bootstrap_end_ok": {"status", "observed"},
    "bootstrap_end_error": {"status", "observed", "replaceable", "error"},
    "run_start": {"started_utc", "protocol", "packed_verifier"},
    "prompt_step": {
        "phase",
        "step_index",
        "token",
        "position",
        "consumed_prefix_len",
        "consumed_prefix_sha256_i32le",
        "arm_order",
        "serial",
        "capture",
        "hidden_transfer",
        "comparisons",
        "first_mismatch",
    },
    "target_transition": {
        "phase",
        "step_index",
        "token",
        "position",
        "consumed_prefix_len",
        "consumed_prefix_sha256_i32le",
        "arm_order",
        "serial",
        "capture",
        "hidden_transfer",
        "comparisons",
        "first_mismatch",
    },
    "sample_frontier": {
        "sample_index",
        "target_position",
        "consumed_prefix_len",
        "consumed_prefix_sha256_i32le",
        "serial_logits_sha256_f32le",
        "capture_logits_sha256_f32le",
        "serial",
        "capture",
        "committed_token",
        "terminal",
        "stop_reason",
        "eos_hit",
        "token_limit_hit",
        "comparisons",
        "first_distribution_mismatch",
    },
    "terminal_boundary": {
        "generated_ids",
        "generated_ids_sha256_i32le",
        "stop_reason",
        "terminal_token",
        "eos_hit",
        "token_limit_hit",
        "serial_sampler_draws",
        "capture_sampler_draws",
        "consumed_prefix_len",
        "pending_token",
        "serial_state",
        "capture_state",
        "dflash_target_ctx_n",
        "comparisons",
        "first_state_mismatch",
    },
    "continuation": {"transition", "next_frontier", "comparisons"},
    "observed_failure": {
        "phase",
        "step_index",
        "token",
        "position",
        "error",
        "observed_e0_failure",
        "replaceable_infrastructure_failure",
    },
    "run_end_ok": {
        "status",
        "e0_status",
        "authority",
        "packed_verifier_status",
        "generated_ids",
        "generated_ids_sha256_i32le",
        "stop_reason",
        "terminal_token",
        "eos_hit",
        "token_limit_hit",
        "prompt_steps",
        "target_transitions",
        "sample_frontiers",
        "serial_sampler_draws",
        "capture_sampler_draws",
        "continuation_compared",
        "continuation_equal",
        "final_consumed_prefix_len",
        "final_dflash_target_ctx_n",
        "all_comparisons",
        "elapsed_seconds_f64_bits",
        "timing_semantics",
        "state_sidecar",
    },
    "run_end_mismatch": {
        "status",
        "e0_status",
        "authority",
        "failed_phase",
        "prompt_steps",
        "target_transitions",
        "sample_frontiers",
        "generated_ids",
        "generated_ids_sha256_i32le",
        "serial_sampler_draws",
        "capture_sampler_draws",
        "continuation_compared",
        "elapsed_seconds_f64_bits",
        "timing_semantics",
        "state_sidecar",
    },
    "run_end_error": {
        "status",
        "e0_status",
        "authority",
        "error",
        "state_sidecar",
    },
}
STATE_KEYS = {
    "identity",
    "identity_sha256_canonical_le",
    "prefix_len",
    "prefix_token_ids_sha256_i32le",
    "pending_token",
    "pending_token_sha256_tagged_i32le",
    "kv_positions",
    "kv_positions_sha256_u64le",
    "sections",
    "final_logits_present",
    "capture_tail_present",
    "section_sidecars",
}
IDENTITY_KEYS = {
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
STATE_COMPARE_KEYS = {
    "identity",
    "prefix",
    "pending_token",
    "kv_positions",
    "kv_k_bytes",
    "kv_v_bytes",
    "gdn_conv_bytes",
    "gdn_state_bytes",
    "all",
}


class EvidenceError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise EvidenceError(message)


def keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{name} must be an object")
    require(
        set(value) == expected,
        f"{name} keys mismatch: missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}",
    )
    return value


def integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    require(
        value >= minimum and (maximum is None or value <= maximum),
        f"{name} is out of range",
    )
    return value


def boolean(value: Any, name: str) -> bool:
    require(isinstance(value, bool), f"{name} must be boolean")
    return value


def text(value: Any, name: str) -> str:
    require(isinstance(value, str) and value, f"{name} must be a nonempty string")
    return value


def sha(value: Any, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        f"{name} must be lowercase SHA-256",
    )
    return value


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items:
        require(key not in out, f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def bad_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON constant {value}")


def finite_json(value: Any, name: str = "JSON") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{name} contains non-finite number")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_json(child, f"{name}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_json(child, f"{name}[{index}]")


def bits(value: Any, width: int, name: str, *, finite: bool = True) -> float:
    count = width // 4
    require(
        isinstance(value, str)
        and value.startswith("0x")
        and len(value) == count + 2
        and all(c in "0123456789abcdef" for c in value[2:]),
        f"{name} has malformed {width}-bit hex",
    )
    raw = int(value[2:], 16)
    result = struct.unpack(
        ">f" if width == 32 else ">d", raw.to_bytes(width // 8, "big")
    )[0]
    if finite:
        require(math.isfinite(result), f"{name} is non-finite")
    return result


def enc64(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def enc32(value: float) -> str:
    return f"0x{struct.unpack('>I', struct.pack('>f', value))[0]:08x}"


def raw_hex(value: Any, byte_count: int, name: str) -> bytes:
    require(
        isinstance(value, str)
        and len(value) == byte_count * 2
        and all(c in "0123456789abcdef" for c in value),
        f"{name} has malformed full hex bytes",
    )
    return bytes.fromhex(value)


def tokens_digest(values: list[int]) -> str:
    return sha_bytes(b"".join(struct.pack("<i", value) for value in values))


def positions_digest(values: list[int]) -> str:
    return sha_bytes(b"".join(struct.pack("<Q", value) for value in values))


def pending_digest(value: int | None) -> str:
    return sha_bytes(b"\0" if value is None else b"\1" + struct.pack("<i", value))


def rotate(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (64 - shift))) & MASK64


def splitmix(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (value ^ (value >> 31)) & MASK64


def rng_initial(seed: int) -> list[int]:
    state, out = seed, []
    for _ in range(4):
        state, word = splitmix(state)
        out.append(word)
    return out


def rng_next(state: list[int]) -> tuple[int, list[int]]:
    s0, s1, s2, s3 = state
    raw = (rotate((s0 + s3) & MASK64, 23) + s0) & MASK64
    temporary = (s1 << 17) & MASK64
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= temporary
    s3 = rotate(s3, 45)
    return raw, [word & MASK64 for word in (s0, s1, s2, s3)]


def state_hex(state: list[int]) -> list[str]:
    return [f"0x{word:016x}" for word in state]


def categorical(weights: list[float], unit: float) -> int:
    total = 0.0
    for weight in weights:
        total += weight
    target = unit * total
    cumulative = 0.0
    selected = len(weights) - 1
    for index, weight in enumerate(weights):
        cumulative += weight
        if target < cumulative:
            selected = index
            break
    return selected


def sampler_v1_distribution(
    logits_f32le: bytes, config: dict[str, Any]
) -> dict[str, Any]:
    require(len(logits_f32le) % 4 == 0 and logits_f32le, "sampler logits malformed")
    logits = list(struct.unpack(f"<{len(logits_f32le) // 4}f", logits_f32le))
    require(
        all(not math.isnan(value) for value in logits), "sampler logits contain NaN"
    )

    def compare(left: tuple[int, float], right: tuple[int, float]) -> int:
        left_token, left_logit = left
        right_token, right_logit = right
        if left_logit > right_logit:
            return -1
        if left_logit < right_logit:
            return 1
        if left_logit == 0.0 and right_logit == 0.0:
            left_negative = math.copysign(1.0, left_logit) < 0.0
            right_negative = math.copysign(1.0, right_logit) < 0.0
            if left_negative != right_negative:
                return 1 if left_negative else -1
        return (left_token > right_token) - (left_token < right_token)

    candidates = sorted(enumerate(logits), key=functools.cmp_to_key(compare))[
        : config["top_k"]
    ]
    minimum = bits(config["min_p_f32_bits"], 32, "sampler min_p")
    if minimum > 0.0:
        maximum = candidates[0][1]
        if math.isfinite(maximum):
            threshold = maximum + math.log(minimum)
            candidates = [item for item in candidates if item[1] >= threshold]
    require(candidates, "sampler-v1 produced no candidates after min-p")
    if candidates[0][1] == math.inf:
        candidates = [item for item in candidates if item[1] == math.inf]
    temperature = bits(config["temperature_f32_bits"], 32, "sampler temperature")
    scaled = [(token, logit / temperature) for token, logit in candidates]
    if any(logit == math.inf for _, logit in scaled):
        weights = [1.0 if logit == math.inf else 0.0 for _, logit in scaled]
    else:
        maximum = scaled[0][1]
        require(maximum != -math.inf, "sampler-v1 has only negative infinity")
        weights = [math.exp(logit - maximum) for _, logit in scaled]
        require(
            all(math.isfinite(weight) for weight in weights), "sampler weights invalid"
        )
    top_p = bits(config["top_p_f32_bits"], 32, "sampler top_p")
    if top_p < 1.0:
        total = 0.0
        for weight in weights:
            total += weight
        target = total * top_p
        cumulative = 0.0
        keep = 0
        for weight in weights:
            cumulative += weight
            keep += 1
            if cumulative >= target:
                break
        keep = max(keep, 1)
        scaled = scaled[:keep]
        weights = weights[:keep]
    total = 0.0
    for weight in weights:
        total += weight
    require(math.isfinite(total) and total > 0.0, "sampler total weight invalid")
    return {
        "tokens": [token for token, _ in scaled],
        "weight_bits": [enc64(weight) for weight in weights],
        "total_weight_bits": enc64(total),
    }


def validate_label(value: Any, name: str) -> None:
    value = text(value, name)
    require(
        len(value) <= 128
        and all(c.isascii() and (c.isalnum() or c in "-_./") for c in value),
        f"{name} is not a development label",
    )


def validate_build(value: Any, name: str, *, require_clean: bool = False) -> None:
    build = keys(
        value,
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
        name,
    )
    integer(build["schema_version"], f"{name}.schema_version", 1)
    commit = text(build["build_commit"], f"{name}.build_commit")
    require(
        len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
        f"{name} commit invalid",
    )
    require(
        build["runtime_commit"] == commit
        and isinstance(build["build_commit_short"], str)
        and commit.startswith(build["build_commit_short"]),
        f"{name} commit disagreement",
    )
    require(
        isinstance(build["build_dirty"], bool)
        and build["runtime_dirty"] is build["build_dirty"],
        f"{name} dirty disagreement",
    )
    source = text(build["build_source_state"], f"{name}.build_source_state")
    require(
        source.startswith("git-source-sha256-v2:")
        and build["runtime_source_state"] == source,
        f"{name} source disagreement",
    )
    sha(source.removeprefix("git-source-sha256-v2:"), f"{name}.source")
    text(build["stamp_source"], f"{name}.stamp_source")
    require(
        build["stamp_error"] is None or isinstance(build["stamp_error"], str),
        f"{name}.stamp_error invalid",
    )
    text(build["status"], f"{name}.status")
    for key in ("problems", "overrides"):
        require(
            isinstance(build[key], list)
            and all(isinstance(x, str) for x in build[key]),
            f"{name}.{key} invalid",
        )
    if require_clean:
        require(
            build["status"] == "match"
            and build["build_dirty"] is False
            and build["runtime_dirty"] is False
            and build["stamp_error"] is None
            and build["problems"] == []
            and build["overrides"] == [],
            f"{name} is not a clean matching acquisition build",
        )


def validate_asset(value: Any, name: str, gguf: bool) -> str:
    if not gguf:
        asset = keys(value, {"path", "bytes", "sha256"}, name)
        text(asset["path"], f"{name}.path")
        integer(asset["bytes"], f"{name}.bytes", 1)
        return sha(asset["sha256"], f"{name}.sha256")
    asset = keys(value, {"aggregate_sha256_index_size_digest_le", "shards"}, name)
    shards = asset["shards"]
    require(isinstance(shards, list) and shards, f"{name}.shards invalid")
    digest = hashlib.sha256()
    seen: set[str] = set()
    for index, raw in enumerate(shards):
        shard = keys(raw, {"index", "path", "bytes", "sha256"}, f"{name}.shard")
        require(shard["index"] == index, f"{name} shard indexes not contiguous")
        path = text(shard["path"], f"{name}.shard.path")
        require(path not in seen, f"{name} duplicate shard path")
        seen.add(path)
        size = integer(shard["bytes"], f"{name}.shard.bytes", 1)
        item_sha = sha(shard["sha256"], f"{name}.shard.sha256")
        digest.update(
            struct.pack("<Q", index) + struct.pack("<Q", size) + bytes.fromhex(item_sha)
        )
    expected = digest.hexdigest()
    require(
        asset["aggregate_sha256_index_size_digest_le"] == expected,
        f"{name} aggregate digest mismatch",
    )
    return expected


def validate_bootstrap_common(row: dict[str, Any], name: str) -> None:
    keys(row, BOOT_COMMON, name)
    validate_build(row["build_identity"], f"{name}.build_identity")
    lease = row["lease_env"]
    require(
        isinstance(lease, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in lease.items()),
        f"{name}.lease_env invalid",
    )
    command = row["command"]
    require(
        isinstance(command, list)
        and command
        and all(isinstance(x, str) for x in command),
        f"{name}.command invalid",
    )
    paths = keys(
        row["paths"],
        {
            "model",
            "drafter",
            "binding_manifest",
            "output",
            "state_sidecar",
            "executable",
        },
        f"{name}.paths",
    )
    for key, value in paths.items():
        text(value, f"{name}.paths.{key}")
    require(
        len(set(paths.values())) == len(paths),
        f"{name}.paths contains aliased artifacts",
    )


def validate_common(row: dict[str, Any], run_id: str) -> dict[str, Any]:
    keys(row, RUN_COMMON, f"run {run_id} event")
    validate_bootstrap_common({key: row[key] for key in BOOT_COMMON}, f"run {run_id}")
    validate_build(
        row["build_identity"], f"run {run_id}.build_identity", require_clean=True
    )
    lease = row["lease_env"]
    require(
        lease.get("QWEN_METAL_LEASE_WAIT") == "1",
        f"run {run_id} lacks literal lease wait",
    )
    classification = keys(
        row["classification"],
        {
            "evidence_role",
            "fixture_id",
            "fixture_role",
            "target_arm",
            "drafter_arm",
            "authority",
        },
        f"run {run_id}.classification",
    )
    require(
        classification["evidence_role"] == "development"
        and classification["authority"] == AUTHORITY,
        f"run {run_id} claims invalid authority",
    )
    for key in ("fixture_id", "fixture_role", "target_arm", "drafter_arm"):
        validate_label(classification[key], f"run {run_id}.{key}")
    config = keys(
        row["config"],
        {
            "tokens",
            "context_capacity",
            "stop_tokens",
            "temperature_f32_bits",
            "top_k",
            "top_p_f32_bits",
            "min_p_f32_bits",
            "seed",
            "sampler_algorithm_version",
            "no_warmup",
            "arm_order",
            "semantics",
            "hidden_transfer_semantics",
        },
        f"run {run_id}.config",
    )
    integer(config["tokens"], "tokens", 1)
    integer(config["context_capacity"], "context_capacity", 1, (1 << 31) - 1)
    integer(config["top_k"], "top_k", 1, 200)
    temperature = bits(config["temperature_f32_bits"], 32, "temperature")
    top_p = bits(config["top_p_f32_bits"], 32, "top_p")
    min_p = bits(config["min_p_f32_bits"], 32, "min_p")
    require(
        temperature > 0 and 0 < top_p <= 1 and 0 <= min_p <= 1,
        f"run {run_id} sampling config invalid",
    )
    integer(config["seed"], "seed", 0, MASK64)
    require(
        config["sampler_algorithm_version"] == SAMPLER_VERSION
        and isinstance(config["no_warmup"], bool),
        f"run {run_id} sampler contract invalid",
    )
    require(
        config["arm_order"] in {"serial_then_capture", "capture_then_serial"},
        f"run {run_id} arm order invalid",
    )
    require(
        config["semantics"] == SEMANTICS
        and config["hidden_transfer_semantics"] == HIDDEN_SEMANTICS,
        f"run {run_id} semantics invalid",
    )
    stops = config["stop_tokens"]
    require(isinstance(stops, list), f"run {run_id} stop tokens invalid")
    for token in stops:
        integer(token, "stop token", 0, (1 << 31) - 1)
    require(len(stops) == len(set(stops)), f"run {run_id} duplicate stop tokens")
    assets = keys(
        row["assets"], {"target", "drafter", "executable"}, f"run {run_id}.assets"
    )
    validate_asset(assets["target"], "target asset", True)
    validate_asset(assets["drafter"], "drafter asset", True)
    validate_asset(assets["executable"], "executable asset", False)
    validate_asset(row["binding_manifest"], "binding manifest", False)
    require(
        row["binding_manifest"]["path"] == row["paths"]["binding_manifest"],
        f"run {run_id} binding-manifest path mismatch",
    )
    require(
        assets["executable"]["path"] == row["paths"]["executable"],
        f"run {run_id} executable path mismatch",
    )
    binding = keys(
        row["binding"],
        {"target_architecture", "target", "drafter", "resolved_stop_tokens"},
        f"run {run_id}.binding",
    )
    text(binding["target_architecture"], "target architecture")
    target = keys(
        binding["target"], {"n_layer", "hidden_size", "vocab_size"}, "target binding"
    )
    for key in target:
        integer(target[key], f"target.{key}", 1, (1 << 32) - 1)
    require(target["vocab_size"] <= (1 << 31), "target vocab exceeds i32")
    drafter = keys(
        binding["drafter"],
        {
            "n_layer",
            "hidden_size",
            "block_size",
            "swa_window",
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
            "target_layer_ids",
        },
        "drafter binding",
    )
    for key in (
        "n_layer",
        "hidden_size",
        "block_size",
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ):
        integer(drafter[key], f"drafter.{key}", 1, (1 << 32) - 1)
    integer(drafter["swa_window"], "drafter.swa_window", 0, (1 << 32) - 1)
    require(
        drafter["hidden_size"] == target["hidden_size"]
        and drafter["selector_top_k"] == 16,
        "target/drafter hidden size or DFlash2 selector binding mismatch",
    )
    layers = drafter["target_layer_ids"]
    require(isinstance(layers, list) and layers, "capture layers missing")
    for layer in layers:
        integer(layer, "capture layer", 0, target["n_layer"] - 1)
    require(
        len(layers) == len(set(layers)) and binding["resolved_stop_tokens"] == stops,
        "binding layers/stops invalid",
    )
    for token in stops:
        require(token < target["vocab_size"], "stop token outside target vocab")
    host = keys(row["host"], {"os", "arch", "metal_device"}, "host")
    for value in host.values():
        text(value, "host value")
    prompt = keys(
        row["prompt"],
        {"utf8_len", "utf8_sha256", "token_count", "token_ids_sha256_i32le"},
        "prompt",
    )
    integer(prompt["utf8_len"], "prompt bytes", 1)
    integer(prompt["token_count"], "prompt tokens", 1)
    require(
        config["context_capacity"] == prompt["token_count"] + config["tokens"] + 33,
        f"run {run_id} context-capacity contract mismatch",
    )
    sha(prompt["utf8_sha256"], "prompt text digest")
    sha(prompt["token_ids_sha256_i32le"], "prompt token digest")
    sessions = keys(
        row["sessions"],
        {"serial_target", "capture_target", "capture_drafter"},
        "sessions",
    )
    values = list(sessions.values())
    require(
        all(isinstance(x, str) and x.startswith(run_id + "/") for x in values)
        and len(set(values)) == 3,
        "session identity invalid or shared",
    )
    return config


def load_binding_manifest(common: dict[str, Any], run_id: str) -> dict[str, Any]:
    identity = common["binding_manifest"]
    path = Path(identity["path"])
    try:
        data = path.read_bytes()
    except OSError as error:
        raise EvidenceError(
            f"run {run_id} cannot read binding manifest: {error}"
        ) from error
    require(
        identity["bytes"] == len(data) and identity["sha256"] == sha_bytes(data),
        f"run {run_id} binding-manifest file identity mismatch",
    )
    try:
        manifest = json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, EvidenceError) as error:
        raise EvidenceError(
            f"run {run_id} binding manifest is invalid JSON: {error}"
        ) from error
    finite_json(manifest, f"run {run_id} binding manifest")
    manifest = keys(
        manifest,
        {
            "schema",
            "schema_version",
            "evidence_role",
            "fixture_id",
            "fixture_role",
            "target_asset_sha256",
            "drafter_asset_sha256",
            "target_arm",
            "drafter_arm",
            "target",
            "drafter",
            "snapshot_abi",
            "allowed_arm_orders",
        },
        f"run {run_id} binding manifest",
    )
    require(
        manifest["schema"] == BINDING_MANIFEST_SCHEMA
        and manifest["schema_version"] == BINDING_MANIFEST_VERSION
        and manifest["evidence_role"] == "development",
        f"run {run_id} binding-manifest schema/authority mismatch",
    )
    require(
        manifest["target_asset_sha256"]
        == common["assets"]["target"]["aggregate_sha256_index_size_digest_le"]
        and manifest["drafter_asset_sha256"]
        == common["assets"]["drafter"]["aggregate_sha256_index_size_digest_le"],
        f"run {run_id} binding-manifest asset mismatch",
    )
    require(
        manifest["fixture_id"] == common["classification"]["fixture_id"]
        and manifest["fixture_role"] == common["classification"]["fixture_role"]
        and manifest["target_arm"] == common["classification"]["target_arm"]
        and manifest["drafter_arm"] == common["classification"]["drafter_arm"],
        f"run {run_id} binding-manifest fixture/arm mismatch",
    )
    expected_target = {
        "architecture": common["binding"]["target_architecture"],
        **common["binding"]["target"],
    }
    require(
        manifest["target"] == expected_target
        and manifest["drafter"] == common["binding"]["drafter"],
        f"run {run_id} binding-manifest geometry mismatch",
    )
    abi = keys(
        manifest["snapshot_abi"],
        IDENTITY_KEYS - {"model_id", "tokenizer_id"},
        "snapshot ABI",
    )
    for key, value in abi.items():
        if key == "kv_storage_kind":
            require(value in {"None", "F16", "Q8_0"}, "snapshot ABI storage invalid")
        else:
            integer(value, f"snapshot ABI {key}", 0, (1 << 32) - 1)
    require(
        abi["n_attn_layers"] + abi["n_gdn_layers"]
        == common["binding"]["target"]["n_layer"],
        f"run {run_id} snapshot ABI layer geometry mismatch",
    )
    require(
        manifest["allowed_arm_orders"] == ["serial_then_capture", "capture_then_serial"]
        and common["config"]["arm_order"] in manifest["allowed_arm_orders"],
        f"run {run_id} binding-manifest arm-order mismatch",
    )
    return manifest


def identity_digest(identity: dict[str, Any]) -> str:
    storage = {"None": 0, "F16": 1, "Q8_0": 2}
    require(identity["kv_storage_kind"] in storage, "unsupported KV storage")
    data = struct.pack("<QQ", identity["model_id"], identity["tokenizer_id"])
    for key in (
        "layout_version",
        "n_attn_layers",
        "n_gdn_layers",
        "kv_dim_elements",
        "kv_bytes_per_token",
    ):
        data += struct.pack("<I", identity[key])
    data += struct.pack("<I", storage[identity["kv_storage_kind"]])
    data += struct.pack(
        "<II",
        identity["gdn_state_elements_per_layer"],
        identity["gdn_conv_elements_per_layer"],
    )
    return sha_bytes(data)


def validate_state(
    value: Any,
    prefix: list[int],
    pending: int | None,
    common: dict[str, Any],
    sidecar: bytes,
    sidecar_cursor: list[int],
    name: str,
) -> dict[str, Any]:
    state = keys(value, STATE_KEYS, name)
    identity = keys(state["identity"], IDENTITY_KEYS, f"{name}.identity")
    for key, value in identity.items():
        if key == "kv_storage_kind":
            require(isinstance(value, str), f"{name}.storage invalid")
        else:
            integer(
                value,
                f"{name}.identity.{key}",
                0,
                MASK64 if key in {"model_id", "tokenizer_id"} else (1 << 32) - 1,
            )
    require(
        state["identity_sha256_canonical_le"] == identity_digest(identity),
        f"{name} identity digest mismatch",
    )
    asset = bytes.fromhex(
        common["assets"]["target"]["aggregate_sha256_index_size_digest_le"]
    )
    require(
        identity["model_id"] == struct.unpack("<Q", asset[:8])[0]
        and identity["tokenizer_id"] == struct.unpack("<Q", asset[8:16])[0],
        f"{name} identity is not target-asset bound",
    )
    require(
        identity["n_attn_layers"] + identity["n_gdn_layers"]
        == common["binding"]["target"]["n_layer"],
        f"{name} layer geometry mismatch",
    )
    observed_abi = {
        key: identity[key] for key in IDENTITY_KEYS - {"model_id", "tokenizer_id"}
    }
    require(
        observed_abi == common["__binding_manifest_content"]["snapshot_abi"],
        f"{name} snapshot ABI is not prospectively manifest-bound",
    )
    prefix_len = integer(state["prefix_len"], f"{name}.prefix_len", 1)
    require(
        prefix_len == len(prefix)
        and state["prefix_token_ids_sha256_i32le"] == tokens_digest(prefix),
        f"{name} prefix identity mismatch",
    )
    if pending is None:
        require(state["pending_token"] is None, f"{name} pending token must be null")
    else:
        integer(
            state["pending_token"],
            f"{name}.pending_token",
            0,
            common["binding"]["target"]["vocab_size"] - 1,
        )
    require(
        state["pending_token"] == pending
        and state["pending_token_sha256_tagged_i32le"] == pending_digest(pending),
        f"{name} pending identity mismatch",
    )
    positions = state["kv_positions"]
    require(
        isinstance(positions, list) and len(positions) == identity["n_attn_layers"],
        f"{name} KV positions geometry mismatch",
    )
    for position in positions:
        integer(position, f"{name} KV position", 0, MASK64)
    require(
        all(position == len(prefix) for position in positions)
        and state["kv_positions_sha256_u64le"] == positions_digest(positions),
        f"{name} KV positions invalid",
    )
    sections = keys(
        state["sections"], {"kv_k", "kv_v", "gdn_conv", "gdn_state"}, f"{name}.sections"
    )
    sidecars = keys(
        state["section_sidecars"],
        {"kv_k", "kv_v", "gdn_conv", "gdn_state"},
        f"{name}.section_sidecars",
    )
    expected = {
        "kv_k": identity["n_attn_layers"]
        * len(prefix)
        * identity["kv_bytes_per_token"],
        "kv_v": identity["n_attn_layers"]
        * len(prefix)
        * identity["kv_bytes_per_token"],
        "gdn_conv": identity["n_gdn_layers"]
        * identity["gdn_conv_elements_per_layer"]
        * 4,
        "gdn_state": identity["n_gdn_layers"]
        * identity["gdn_state_elements_per_layer"]
        * 4,
    }
    raw_sections: dict[str, bytes] = {}
    for key, size in expected.items():
        section = keys(sections[key], {"bytes", "sha256"}, f"{name}.{key}")
        require(
            integer(section["bytes"], f"{name}.{key}.bytes", 0) == size,
            f"{name}.{key} byte geometry mismatch",
        )
        section_sha = sha(section["sha256"], f"{name}.{key}.sha256")
        reference = keys(
            sidecars[key], {"offset", "bytes", "sha256"}, f"{name}.{key}.sidecar"
        )
        offset = integer(reference["offset"], f"{name}.{key}.offset", 0)
        count = integer(reference["bytes"], f"{name}.{key}.sidecar_bytes", 0)
        require(
            offset == sidecar_cursor[0] and count == size,
            f"{name}.{key} sidecar range is noncontiguous or has wrong geometry",
        )
        end = offset + count
        require(end <= len(sidecar), f"{name}.{key} sidecar range exceeds file")
        raw = sidecar[offset:end]
        raw_sha = sha_bytes(raw)
        require(
            reference["sha256"] == raw_sha == section_sha,
            f"{name}.{key} sidecar/hash mismatch",
        )
        raw_sections[key] = raw
        sidecar_cursor[0] = end
    require(
        state["final_logits_present"] is False
        and state["capture_tail_present"] is False,
        f"{name} retains transient state",
    )
    validated = dict(state)
    validated["__raw_sections"] = raw_sections
    return validated


def state_comparisons(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    result = {
        "identity": left["identity"] == right["identity"]
        and left["identity_sha256_canonical_le"]
        == right["identity_sha256_canonical_le"],
        "prefix": left["prefix_len"] == right["prefix_len"]
        and left["prefix_token_ids_sha256_i32le"]
        == right["prefix_token_ids_sha256_i32le"],
        "pending_token": left["pending_token"] == right["pending_token"]
        and left["pending_token_sha256_tagged_i32le"]
        == right["pending_token_sha256_tagged_i32le"],
        "kv_positions": left["kv_positions"] == right["kv_positions"]
        and left["kv_positions_sha256_u64le"] == right["kv_positions_sha256_u64le"],
        "kv_k_bytes": left["__raw_sections"]["kv_k"] == right["__raw_sections"]["kv_k"],
        "kv_v_bytes": left["__raw_sections"]["kv_v"] == right["__raw_sections"]["kv_v"],
        "gdn_conv_bytes": left["__raw_sections"]["gdn_conv"]
        == right["__raw_sections"]["gdn_conv"],
        "gdn_state_bytes": left["__raw_sections"]["gdn_state"]
        == right["__raw_sections"]["gdn_state"],
    }
    result["all"] = all(result.values())
    return result


def validate_claim(actual: Any, derived: dict[str, bool], name: str) -> None:
    keys(actual, set(derived), name)
    require(actual == derived, f"{name} producer booleans disagree with derivation")


def validate_mismatch(value: Any, name: str) -> None:
    if value is None:
        return
    require(isinstance(value, dict), f"{name} must be null or an object")
    allowed = (
        {"index", "serial_f32_bits", "capture_f32_bits"},
        {"index", "serial_len", "capture_len"},
        {"offset", "serial_byte", "capture_byte"},
        {"offset", "serial_len", "capture_len"},
        {"section"},
        {"section", "index", "serial", "capture"},
        {"section", "offset", "serial_byte", "capture_byte"},
        {"section", "offset", "serial_len", "capture_len"},
        {"field", "serial", "capture"},
        {"field", "index", "serial", "capture"},
    )
    require(set(value) in allowed, f"{name} has unknown diagnostic shape")
    if "serial_f32_bits" in value:
        bits(value["serial_f32_bits"], 32, f"{name}.serial_f32_bits", finite=False)
        bits(value["capture_f32_bits"], 32, f"{name}.capture_f32_bits", finite=False)
    for key in (
        "index",
        "offset",
        "serial_len",
        "capture_len",
        "serial_byte",
        "capture_byte",
    ):
        if key in value:
            integer(value[key], f"{name}.{key}", 0)
    if "section" in value:
        require(
            value["section"]
            in {
                "identity",
                "prefix_tokens",
                "pending_token",
                "kv_positions",
                "kv_k",
                "kv_v",
                "gdn_conv",
                "gdn_state",
            },
            f"{name}.section invalid",
        )
    if "field" in value:
        require(
            value["field"]
            in {"sampled", "total_weight_f64_bits", "candidate", "candidate_count"},
            f"{name}.field invalid",
        )
        if value["field"] == "sampled":
            keys(value["serial"], {"token", "candidate_index"}, f"{name}.serial")
            keys(value["capture"], {"token", "candidate_index"}, f"{name}.capture")
        elif value["field"] == "candidate":
            keys(value["serial"], {"token", "weight_f64_bits"}, f"{name}.serial")
            keys(value["capture"], {"token", "weight_f64_bits"}, f"{name}.capture")
            bits(value["serial"]["weight_f64_bits"], 64, f"{name}.serial.weight")
            bits(value["capture"]["weight_f64_bits"], 64, f"{name}.capture.weight")
        elif value["field"] == "total_weight_f64_bits":
            bits(value["serial"], 64, f"{name}.serial")
            bits(value["capture"], 64, f"{name}.capture")


def validate_distribution(
    value: Any,
    vocab: int,
    top_k: int,
    expected: dict[str, Any],
    rng_state: list[int],
    draw: int,
    name: str,
) -> tuple[dict[str, Any], list[int]]:
    distribution = keys(
        value["distribution"],
        {
            "selected_token",
            "candidate_index",
            "ordered_support",
            "total_weight_f64_bits",
        },
        f"{name}.distribution",
    )
    support = distribution["ordered_support"]
    require(
        isinstance(support, list) and 1 <= len(support) <= top_k,
        f"{name} support length invalid",
    )
    tokens, weights = [], []
    for index, raw in enumerate(support):
        item = keys(raw, {"token", "weight_f64_bits"}, f"{name}.support[{index}]")
        token = integer(item["token"], f"{name}.support token", 0, vocab - 1)
        weight = bits(item["weight_f64_bits"], 64, f"{name}.support weight")
        require(weight >= 0, f"{name} negative weight")
        tokens.append(token)
        weights.append(weight)
    require(
        len(tokens) == len(set(tokens))
        and weights[0] == 1.0
        and all(a >= b for a, b in zip(weights, weights[1:])),
        f"{name} support ordering invalid",
    )
    require(
        tokens == expected["tokens"]
        and [item["weight_f64_bits"] for item in support] == expected["weight_bits"],
        f"{name} support/weights do not reconstruct from full logits",
    )
    total = 0.0
    for weight in weights:
        total += weight
    require(
        math.isfinite(total)
        and total > 0
        and distribution["total_weight_f64_bits"] == enc64(total),
        f"{name} total weight mismatch",
    )
    require(
        distribution["total_weight_f64_bits"] == expected["total_weight_bits"],
        f"{name} total weight does not reconstruct from full logits",
    )
    selected_index = integer(
        distribution["candidate_index"], f"{name}.candidate_index", 0, len(tokens) - 1
    )
    selected = integer(distribution["selected_token"], f"{name}.selected", 0, vocab - 1)
    require(tokens[selected_index] == selected, f"{name} selected support mismatch")
    rng = keys(
        value["rng"],
        {
            "draws_before",
            "draws_after",
            "state_before",
            "state_after",
            "raw_u64",
            "raw_uniform_f64_bits",
        },
        f"{name}.rng",
    )
    require(
        integer(rng["draws_before"], f"{name}.rng.draws_before", 0) == draw
        and integer(rng["draws_after"], f"{name}.rng.draws_after", 0) == draw + 1
        and rng["state_before"] == state_hex(rng_state),
        f"{name} RNG prestate/draw mismatch",
    )
    raw, after = rng_next(rng_state)
    require(
        rng["state_after"] == state_hex(after) and rng["raw_u64"] == f"0x{raw:016x}",
        f"{name} RNG transition mismatch",
    )
    unit = (raw >> 11) / float(1 << 53)
    require(rng["raw_uniform_f64_bits"] == enc64(unit), f"{name} uniform mismatch")
    replay = categorical(weights, unit)
    require(
        replay == selected_index and tokens[replay] == selected,
        f"{name} categorical replay mismatch",
    )
    return distribution, after


def decode_logits(
    arm: Any,
    vocab: int,
    common: dict[str, Any],
    prefix: list[int],
    sidecar: bytes,
    sidecar_cursor: list[int],
    name: str,
) -> tuple[bytes, dict[str, Any]]:
    arm = keys(
        arm,
        {"api", "logits_count", "logits_sha256_f32le", "logits_f32le_hex", "state"},
        name,
    )
    count = integer(arm["logits_count"], f"{name}.logits_count", 0, vocab * 2)
    data = raw_hex(arm["logits_f32le_hex"], count * 4, f"{name}.logits")
    require(
        len(struct.unpack(f"<{count}f", data)) == count,
        f"{name} full f32le logits decode failed",
    )
    require(
        arm["logits_sha256_f32le"] == sha_bytes(data), f"{name} logits hash mismatch"
    )
    state = validate_state(
        arm["state"],
        prefix,
        None,
        common,
        sidecar,
        sidecar_cursor,
        f"{name}.state",
    )
    return data, state


def validate_transition(
    payload: Any,
    prefix_before: list[int],
    expected_phase: str,
    expected_step: int,
    expected_token: int | None,
    context_bytes: bytes,
    positions: list[int],
    common: dict[str, Any],
    sidecar: bytes,
    sidecar_cursor: list[int],
    name: str,
) -> tuple[list[int], bytes, bytes, bool]:
    keys(payload, PAYLOAD_KEYS["prompt_step"], name)
    vocab = common["binding"]["target"]["vocab_size"]
    token = integer(payload["token"], f"{name}.token", 0, vocab - 1)
    if expected_token is not None:
        require(token == expected_token, f"{name} does not consume the sampled token")
    position = len(prefix_before)
    prefix = prefix_before + [token]
    require(
        payload["phase"] == expected_phase
        and integer(payload["step_index"], f"{name}.step_index", 0) == expected_step
        and integer(payload["position"], f"{name}.position", 0, (1 << 31) - 1)
        == position,
        f"{name} phase/step/position mismatch",
    )
    require(
        integer(payload["consumed_prefix_len"], f"{name}.consumed_prefix_len", 1)
        == len(prefix)
        and payload["consumed_prefix_sha256_i32le"] == tokens_digest(prefix),
        f"{name} prefix evidence mismatch",
    )
    require(
        payload["arm_order"] == common["config"]["arm_order"],
        f"{name} arm order mismatch",
    )
    serial_bytes, serial_state = decode_logits(
        payload["serial"],
        vocab,
        common,
        prefix,
        sidecar,
        sidecar_cursor,
        f"{name}.serial",
    )
    capture_bytes, capture_state = decode_logits(
        payload["capture"],
        vocab,
        common,
        prefix,
        sidecar,
        sidecar_cursor,
        f"{name}.capture",
    )
    require(
        payload["serial"]["api"] == "single_token"
        and payload["capture"]["api"] == "single_token_with_multi_hidden",
        f"{name} API identity mismatch",
    )
    hidden = keys(
        payload["hidden_transfer"],
        {
            "semantics",
            "target_layer_ids",
            "shape",
            "source_bytes",
            "source_sha256_f32le",
            "source_f32le_hex",
            "poison_f32_bits",
            "first_retained_poison_index",
            "first_nonfinite_index",
            "destination_bytes",
            "destination_sha256_f32le",
            "destination_f32le_hex",
            "active_context_bytes",
            "active_context_sha256_f32le",
            "active_context_f32le_hex",
            "expected_active_context_sha256_f32le",
            "active_positions",
            "expected_active_positions",
            "target_ctx_index",
            "target_ctx_n_after",
            "captured_position",
            "ctx_h_ready_n",
            "kv_ctx_ready_n",
        },
        f"{name}.hidden_transfer",
    )
    layers = common["binding"]["drafter"]["target_layer_ids"]
    width = common["binding"]["target"]["hidden_size"]
    row_size = len(layers) * width * 4
    require(
        hidden["semantics"] == HIDDEN_SEMANTICS
        and hidden["target_layer_ids"] == layers
        and hidden["shape"] == [len(layers), width],
        f"{name} hidden binding/shape mismatch",
    )
    require(
        hidden["poison_f32_bits"] == "0x7fa5a5a5", f"{name} poison identity mismatch"
    )
    source = raw_hex(hidden["source_f32le_hex"], row_size, f"{name}.hidden source")
    destination_size = integer(
        hidden["destination_bytes"], f"{name}.destination_bytes", 0, row_size
    )
    require(
        destination_size in {0, row_size},
        f"{name} destination must be absent or one complete row",
    )
    destination = raw_hex(
        hidden["destination_f32le_hex"],
        destination_size,
        f"{name}.hidden destination",
    )
    require(
        integer(hidden["source_bytes"], f"{name}.source_bytes", 0) == row_size
        and hidden["source_sha256_f32le"] == sha_bytes(source)
        and hidden["destination_sha256_f32le"] == sha_bytes(destination),
        f"{name} hidden byte/hash mismatch",
    )
    words = [
        struct.unpack_from("<I", source, offset)[0]
        for offset in range(0, len(source), 4)
    ]
    retained = next((i for i, word in enumerate(words) if word == 0x7FA5A5A5), None)
    nonfinite = next(
        (
            i
            for i, word in enumerate(words)
            if not math.isfinite(struct.unpack("<f", struct.pack("<I", word))[0])
        ),
        None,
    )
    require(
        hidden["first_retained_poison_index"] == retained
        and hidden["first_nonfinite_index"] == nonfinite,
        f"{name} hidden poison/finite diagnosis mismatch",
    )
    active = context_bytes + source
    active_positions = positions + [position]
    active_size = integer(
        hidden["active_context_bytes"],
        f"{name}.active_context_bytes",
        0,
        common["config"]["context_capacity"] * row_size,
    )
    require(active_size % row_size == 0, f"{name} active context has a partial row")
    observed_active = raw_hex(
        hidden["active_context_f32le_hex"],
        active_size,
        f"{name}.active_context",
    )
    require(
        hidden["active_context_sha256_f32le"] == sha_bytes(observed_active),
        f"{name} observed active-context hash mismatch",
    )
    require(
        hidden["expected_active_context_sha256_f32le"] == sha_bytes(active),
        f"{name} expected context hash is not independently reconstructed",
    )
    require(
        isinstance(hidden["active_positions"], list)
        and isinstance(hidden["expected_active_positions"], list),
        f"{name} active positions invalid",
    )
    for value in hidden["active_positions"] + hidden["expected_active_positions"]:
        integer(value, f"{name} active position", 0, (1 << 31) - 1)
    require(
        hidden["expected_active_positions"] == active_positions,
        f"{name} expected positions are not independently reconstructed",
    )
    for key in (
        "target_ctx_index",
        "target_ctx_n_after",
        "ctx_h_ready_n",
        "kv_ctx_ready_n",
    ):
        integer(hidden[key], f"{name}.{key}", 0, (1 << 31) - 1)
    require(
        hidden["captured_position"] is None
        or isinstance(hidden["captured_position"], int)
        and not isinstance(hidden["captured_position"], bool),
        f"{name}.captured_position invalid",
    )
    if hidden["captured_position"] is not None:
        integer(
            hidden["captured_position"],
            f"{name}.captured_position",
            0,
            (1 << 31) - 1,
        )
    state_derived = state_comparisons(serial_state, capture_state)
    derived = {
        "logits_bits": len(serial_bytes) == len(capture_bytes) == vocab * 4
        and serial_bytes == capture_bytes,
        "state": state_derived,
        "hidden_overwrite_complete": retained is None,
        "hidden_values_finite": nonfinite is None,
        "hidden_transfer_bytes": source == destination,
        "active_context_history": observed_active == active,
        "active_position_history": hidden["active_positions"] == active_positions,
        "target_ctx_length": hidden["target_ctx_index"] == len(positions)
        and hidden["target_ctx_n_after"] == len(prefix),
        "target_ctx_position": hidden["captured_position"] == position,
        "target_ctx_watermarks": hidden["ctx_h_ready_n"]
        == hidden["kv_ctx_ready_n"]
        == 0,
    }
    derived["all"] = (
        derived["logits_bits"]
        and state_derived["all"]
        and all(
            value
            for key, value in derived.items()
            if key not in {"state", "logits_bits", "all"}
        )
    )
    comparisons = keys(payload["comparisons"], set(derived), f"{name}.comparisons")
    validate_claim(comparisons["state"], state_derived, f"{name}.comparisons.state")
    require(
        {key: value for key, value in comparisons.items() if key != "state"}
        == {key: value for key, value in derived.items() if key != "state"},
        f"{name} producer comparison disagreement",
    )
    mismatch = keys(
        payload["first_mismatch"],
        {"logits", "state", "hidden_transfer", "active_context"},
        f"{name}.first_mismatch",
    )
    for key, value in mismatch.items():
        validate_mismatch(value, f"{name}.first_mismatch.{key}")
    if derived["all"]:
        require(
            all(value is None for value in mismatch.values()),
            f"{name} reports mismatch despite exact evidence",
        )
    return prefix, active, serial_bytes, derived["all"]


def validate_sample(
    payload: Any,
    prefix: list[int],
    logits: bytes,
    draw: int,
    rng_state: list[int],
    common: dict[str, Any],
    name: str,
    *,
    continuation: bool = False,
) -> tuple[int, list[int], bool]:
    if not continuation:
        keys(payload, PAYLOAD_KEYS["sample_frontier"], name)
        require(
            integer(payload["sample_index"], f"{name}.sample_index", 0) == draw
            and integer(
                payload["target_position"],
                f"{name}.target_position",
                0,
                (1 << 31) - 1,
            )
            == len(prefix) - 1
            and integer(
                payload["consumed_prefix_len"], f"{name}.consumed_prefix_len", 1
            )
            == len(prefix)
            and payload["consumed_prefix_sha256_i32le"] == tokens_digest(prefix),
            f"{name} frontier identity mismatch",
        )
        digest = sha_bytes(logits)
        require(
            payload["serial_logits_sha256_f32le"] == digest
            and payload["capture_logits_sha256_f32le"] == digest,
            f"{name} logits linkage mismatch",
        )
    vocab = common["binding"]["target"]["vocab_size"]
    require(
        len(logits) == vocab * 4,
        f"{name} cannot sample from a malformed target-logit vector",
    )
    expected_distribution = sampler_v1_distribution(logits, common["config"])
    arms = []
    after_state: list[int] | None = None
    for arm_name in ("serial", "capture"):
        arm = keys(
            payload[arm_name],
            {"distribution", "rng", "live_sample"}
            if not continuation
            else {"distribution", "rng", "live_draws_unchanged"},
            f"{name}.{arm_name}",
        )
        distribution, after = validate_distribution(
            arm,
            vocab,
            common["config"]["top_k"],
            expected_distribution,
            rng_state,
            draw,
            f"{name}.{arm_name}",
        )
        if not continuation:
            live = keys(
                arm["live_sample"],
                {"token", "candidate_index", "draws_after"},
                f"{name}.{arm_name}.live",
            )
            require(
                integer(live["token"], f"{name}.{arm_name}.live.token", 0, vocab - 1)
                == distribution["selected_token"]
                and integer(
                    live["candidate_index"],
                    f"{name}.{arm_name}.live.candidate_index",
                    0,
                )
                == distribution["candidate_index"]
                and integer(
                    live["draws_after"], f"{name}.{arm_name}.live.draws_after", 0
                )
                == draw + 1,
                f"{name}.{arm_name} live sample mismatch",
            )
        else:
            require(
                integer(
                    arm["live_draws_unchanged"],
                    f"{name}.{arm_name}.live_draws_unchanged",
                    0,
                )
                == draw,
                f"{name}.{arm_name} diagnostic advanced RNG",
            )
        arms.append(distribution)
        after_state = after
    derived = {
        "distribution_bits": arms[0] == arms[1],
        "rng_transition": payload["serial"]["rng"] == payload["capture"]["rng"],
    }
    if continuation:
        derived["diagnostics_nonadvancing"] = (
            payload["serial"]["live_draws_unchanged"]
            == payload["capture"]["live_draws_unchanged"]
            == draw
        )
    else:
        derived["diagnosed_vs_live"] = all(
            payload[arm]["live_sample"]["token"] == arms[index]["selected_token"]
            and payload[arm]["live_sample"]["candidate_index"]
            == arms[index]["candidate_index"]
            for index, arm in enumerate(("serial", "capture"))
        )
        derived["live_sample"] = (
            payload["serial"]["live_sample"] == payload["capture"]["live_sample"]
        )
        derived["draw_counts"] = (
            payload["serial"]["live_sample"]["draws_after"]
            == payload["capture"]["live_sample"]["draws_after"]
            == draw + 1
        )
    derived["all"] = all(derived.values())
    validate_claim(payload["comparisons"], derived, f"{name}.comparisons")
    validate_mismatch(
        payload["first_distribution_mismatch"],
        f"{name}.first_distribution_mismatch",
    )
    if derived["all"]:
        require(
            payload["first_distribution_mismatch"] is None,
            f"{name} false mismatch diagnostic",
        )
    token = arms[0]["selected_token"]
    if not continuation:
        emitted = draw + 1
        terminal = (
            emitted >= common["config"]["tokens"]
            or token in common["config"]["stop_tokens"]
        )
        eos = token in common["config"]["stop_tokens"]
        if derived["all"]:
            require(
                integer(
                    payload["committed_token"], f"{name}.committed_token", 0, vocab - 1
                )
                == token
                and payload["terminal"] is terminal
                and (
                    payload["stop_reason"] == ("eos" if eos else "token_limit")
                    if terminal
                    else payload["stop_reason"] is None
                ),
                f"{name} terminal evidence mismatch",
            )
            require(
                payload["eos_hit"] is (eos if terminal else False)
                and payload["token_limit_hit"]
                is (emitted >= common["config"]["tokens"] if terminal else False),
                f"{name} stop booleans mismatch",
            )
        else:
            require(
                payload["committed_token"] is None
                and payload["terminal"] is False
                and payload["stop_reason"] is None
                and payload["eos_hit"] is False
                and payload["token_limit_hit"] is False,
                f"{name} mismatching sample was committed",
            )
    return token, after_state or rng_state, derived["all"]


def payload_kind(row: dict[str, Any]) -> str:
    event, payload = row["event"], row["payload"]
    if event == "bootstrap_end":
        return (
            "bootstrap_end_ok"
            if payload.get("status") == "ok"
            else "bootstrap_end_error"
        )
    if event == "run_end":
        return (
            "run_end_ok"
            if payload.get("status") == "ok"
            else (
                "run_end_mismatch"
                if payload.get("status") == "mismatch"
                else "run_end_error"
            )
        )
    return event


def read_inputs(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, inputs, seen = [], [], set()
    for argument in paths:
        path = argument.resolve()
        require(path not in seen, f"duplicate input {path}")
        seen.add(path)
        try:
            data = path.read_bytes()
        except OSError as error:
            raise EvidenceError(f"cannot read {path}: {error}") from error
        require(
            data and data.endswith(b"\n"), f"empty or incomplete JSONL input {path}"
        )
        inputs.append(
            {"path": str(path), "bytes": len(data), "sha256": sha_bytes(data)}
        )
        for line_number, raw in enumerate(data.splitlines(), 1):
            require(raw.strip(), f"blank JSONL line at {path}:{line_number}")
            try:
                row = json.loads(
                    raw.decode("utf-8"),
                    object_pairs_hook=pairs,
                    parse_constant=bad_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, EvidenceError) as error:
                raise EvidenceError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            require(
                isinstance(row, dict), f"row at {path}:{line_number} is not an object"
            )
            finite_json(row, f"{path}:{line_number}")
            require(
                row.get("schema") == SCHEMA and row.get("schema_version") == VERSION,
                f"schema mismatch at {path}:{line_number}",
            )
            text(row.get("run_id"), "run_id")
            require(
                row.get("event")
                in {
                    "bootstrap_start",
                    "bootstrap_end",
                    "run_start",
                    "prompt_step",
                    "sample_frontier",
                    "target_transition",
                    "terminal_boundary",
                    "continuation",
                    "observed_failure",
                    "run_end",
                },
                f"unknown event at {path}:{line_number}",
            )
            kind = payload_kind(row)
            keys(
                row.get("payload"),
                PAYLOAD_KEYS[kind],
                f"{kind} payload at {path}:{line_number}",
            )
            row["__input"] = str(path)
            row["__source"] = f"{path}:{line_number}"
            row["__line"] = line_number
            rows.append(row)
    return rows, sorted(inputs, key=lambda item: item["path"])


def strip_internal(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("__")}


def reduce_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_id = rows[0]["run_id"]
    require(len({row["__input"] for row in rows}) == 1, f"run {run_id} spans inputs")
    line_numbers = [row["__line"] for row in rows]
    require(
        line_numbers == list(range(line_numbers[0], line_numbers[0] + len(rows))),
        f"run {run_id} is interleaved with another run",
    )
    require(
        [row["event"] for row in rows[:2]] == ["bootstrap_start", "bootstrap_end"],
        f"run {run_id} bootstrap order invalid",
    )
    bootstrap = strip_internal(rows[0])
    for row in rows[:2]:
        validate_bootstrap_common(strip_internal(row), f"run {run_id} bootstrap")
        for key in BOOT_COMMON - {"event", "payload"}:
            require(
                row[key] == rows[0][key],
                f"run {run_id} bootstrap common metadata changed",
            )
    require(
        bootstrap["paths"]["output"] == rows[0]["__input"],
        f"run {run_id} recorded output path does not identify its input trace",
    )
    text(rows[0]["payload"]["started_utc"], "bootstrap timestamp")
    boot_end = rows[1]["payload"]
    if boot_end["status"] == "infrastructure_error":
        require(
            len(rows) == 2
            and boot_end["observed"] is False
            and boot_end["replaceable"] is False
            and isinstance(boot_end["error"], str),
            f"run {run_id} malformed bootstrap failure",
        )
        return {
            "run_id": run_id,
            "status": "invalid_pre_observation",
            "development_lockstep_passed": False,
            "observed_failure": False,
            "emitted": 0,
            "transitions": 0,
        }
    require(
        boot_end == {"status": "ok", "observed": False},
        f"run {run_id} malformed bootstrap success",
    )
    require(
        len(rows) >= 4
        and rows[2]["event"] == "run_start"
        and rows[-1]["event"] == "run_end",
        f"run {run_id} run boundaries invalid",
    )
    common = strip_internal(rows[2])
    config = validate_common(common, run_id)
    common["__binding_manifest_content"] = load_binding_manifest(common, run_id)
    for key in BOOT_COMMON - {"event", "payload"}:
        require(
            common[key] == bootstrap[key],
            f"run {run_id} bootstrap/run identity mismatch for {key}",
        )
    sidecar_path = Path(common["paths"]["state_sidecar"])
    try:
        sidecar = sidecar_path.read_bytes()
    except OSError as error:
        raise EvidenceError(
            f"run {run_id} cannot read state sidecar: {error}"
        ) from error
    sidecar_cursor = [0]
    for row in rows[2:]:
        clean = strip_internal(row)
        keys(clean, RUN_COMMON, f"run {run_id} event")
        for key in RUN_COMMON - {"event", "payload"}:
            require(
                row[key] == rows[2][key],
                f"run {run_id} common metadata changed at {row['__source']}",
            )
    start = rows[2]["payload"]
    require(
        start["protocol"] == "E0-CLARIFICATION.md"
        and start["packed_verifier"] == "excluded_different_distribution",
        f"run {run_id} protocol authority invalid",
    )
    text(start["started_utc"], "run timestamp")
    prefix: list[int] = []
    context = b""
    positions: list[int] = []
    logits: bytes | None = None
    generated: list[int] = []
    rng_state = rng_initial(config["seed"])
    prompt_steps = transitions = samples = 0
    terminal_seen = continuation_seen = observed_failure = mismatch_detected = False
    observed_failure_phase: str | None = None
    phase = "prompt"
    end = rows[-1]["payload"]
    for row in rows[3:-1]:
        event, payload = row["event"], row["payload"]
        require(
            not observed_failure and not mismatch_detected,
            f"run {run_id} has events after a terminal failure",
        )
        if event == "prompt_step":
            require(
                phase == "prompt" and prompt_steps < common["prompt"]["token_count"],
                f"run {run_id} prompt event out of order",
            )
            prefix, context, logits, parity = validate_transition(
                payload,
                prefix,
                "prompt",
                prompt_steps,
                None,
                context,
                positions,
                common,
                sidecar,
                sidecar_cursor,
                f"run {run_id} prompt {prompt_steps}",
            )
            positions.append(len(positions))
            prompt_steps += 1
            if not parity:
                mismatch_detected = True
                phase = "failed"
        elif event == "sample_frontier":
            require(
                prompt_steps == common["prompt"]["token_count"]
                and phase in {"prompt", "generated"}
                and logits is not None,
                f"run {run_id} sample frontier out of order",
            )
            token, rng_state, parity = validate_sample(
                payload,
                prefix,
                logits,
                samples,
                rng_state,
                common,
                f"run {run_id} sample {samples}",
            )
            samples += 1
            if parity:
                generated.append(token)
                terminal = samples >= config["tokens"] or token in config["stop_tokens"]
                phase = "boundary" if terminal else "need_transition"
            else:
                mismatch_detected = True
                phase = "failed"
        elif event == "target_transition":
            require(
                phase == "need_transition" and generated,
                f"run {run_id} transition has no causal sample",
            )
            prefix, context, logits, parity = validate_transition(
                payload,
                prefix,
                "generated",
                transitions,
                generated[-1],
                context,
                positions,
                common,
                sidecar,
                sidecar_cursor,
                f"run {run_id} transition {transitions}",
            )
            positions.append(len(positions))
            transitions += 1
            phase = "generated" if parity else "failed"
            mismatch_detected = not parity
        elif event == "terminal_boundary":
            require(
                phase == "boundary" and not terminal_seen and generated,
                f"run {run_id} boundary out of order",
            )
            parity = validate_boundary(
                payload,
                prefix,
                generated,
                context,
                common,
                sidecar,
                sidecar_cursor,
                run_id,
                samples,
            )
            terminal_seen = True
            phase = "continuation" if parity else "failed"
            mismatch_detected = not parity
        elif event == "continuation":
            require(
                phase == "continuation" and terminal_seen and not continuation_seen,
                f"run {run_id} continuation out of order",
            )
            transition = payload["transition"]
            prefix, context, continuation_logits, transition_parity = (
                validate_transition(
                    transition,
                    prefix,
                    "continuation",
                    0,
                    generated[-1],
                    context,
                    positions,
                    common,
                    sidecar,
                    sidecar_cursor,
                    f"run {run_id} continuation",
                )
            )
            positions.append(len(positions))
            frontier = keys(
                payload["next_frontier"],
                {"serial", "capture", "comparisons", "first_distribution_mismatch"},
                "continuation next frontier",
            )
            _, _, frontier_parity = validate_sample(
                frontier,
                prefix,
                continuation_logits,
                samples,
                rng_state,
                common,
                f"run {run_id} continuation frontier",
                continuation=True,
            )
            derived = {
                "transition": transition["comparisons"]["all"],
                "next_frontier": frontier["comparisons"]["all"],
            }
            derived["all"] = all(derived.values())
            validate_claim(payload["comparisons"], derived, "continuation comparisons")
            require(
                derived["transition"] is transition_parity
                and derived["next_frontier"] is frontier_parity,
                f"run {run_id} continuation derivation disagreement",
            )
            continuation_seen = derived["all"]
            mismatch_detected = not derived["all"]
            phase = "done" if derived["all"] else "failed"
        elif event == "observed_failure":
            require(
                event == "observed_failure" and phase != "done",
                f"run {run_id} observed failure misplaced",
            )
            require(
                payload["observed_e0_failure"] is True
                and payload["replaceable_infrastructure_failure"] is False
                and isinstance(payload["error"], str),
                f"run {run_id} observed failure claims invalid",
            )
            integer(payload["step_index"], "failure step", 0)
            require(
                payload["phase"]
                in {
                    "prompt_step",
                    "sample_frontier",
                    "target_transition",
                    "sampler_accounting",
                    "terminal_boundary",
                    "continuation",
                    "continuation_frontier",
                },
                f"run {run_id} observed failure phase invalid",
            )
            expected_failure = {
                "prompt_step": (phase == "prompt", prompt_steps, None, len(prefix)),
                "sample_frontier": (
                    phase in {"prompt", "generated"},
                    samples,
                    None,
                    len(prefix) - 1,
                ),
                "target_transition": (
                    phase == "need_transition" and bool(generated),
                    transitions,
                    generated[-1] if generated else None,
                    len(prefix),
                ),
                "sampler_accounting": (
                    phase == "boundary",
                    samples,
                    None,
                    len(prefix) - 1,
                ),
                "terminal_boundary": (
                    phase == "boundary" and bool(generated),
                    0,
                    generated[-1] if generated else None,
                    len(prefix),
                ),
                "continuation": (
                    phase == "continuation" and bool(generated),
                    0,
                    generated[-1] if generated else None,
                    len(prefix),
                ),
                "continuation_frontier": (
                    phase == "continuation",
                    0,
                    None,
                    len(prefix),
                ),
            }[payload["phase"]]
            phase_ok, expected_step, expected_token, expected_position = (
                expected_failure
            )
            require(
                phase_ok
                and payload["step_index"] == expected_step
                and (
                    payload["token"] == expected_token
                    if payload["phase"] != "prompt_step"
                    else payload["token"] is not None
                )
                and payload["position"] == expected_position,
                f"run {run_id} observed failure is not causally bound",
            )
            for key in ("token", "position"):
                require(
                    payload[key] is None
                    or isinstance(payload[key], int)
                    and not isinstance(payload[key], bool),
                    f"run {run_id} observed failure {key} invalid",
                )
            if payload["token"] is not None:
                integer(
                    payload["token"],
                    f"run {run_id} observed failure token",
                    0,
                    common["binding"]["target"]["vocab_size"] - 1,
                )
            if payload["position"] is not None:
                integer(payload["position"], f"run {run_id} failure position", 0)
            observed_failure = True
            observed_failure_phase = payload["phase"]
        else:
            raise EvidenceError(f"run {run_id} unexpected {event} before run_end")
    if prompt_steps == common["prompt"]["token_count"]:
        require(
            tokens_digest(prefix[: common["prompt"]["token_count"]])
            == common["prompt"]["token_ids_sha256_i32le"],
            f"run {run_id} reconstructed prompt identity mismatch",
        )
    kind = payload_kind(rows[-1])
    validate_sidecar_summary(
        end["state_sidecar"],
        common["paths"]["state_sidecar"],
        sidecar,
        sidecar_cursor[0],
        not observed_failure,
        f"run {run_id} state sidecar",
    )
    if kind == "run_end_error":
        require(
            end["status"] == "infrastructure_error"
            and end["e0_status"] == "invalid_not_observed"
            and end["authority"] == AUTHORITY
            and isinstance(end["error"], str),
            f"run {run_id} malformed retained infrastructure failure",
        )
        require(
            len(rows) == 4
            and not continuation_seen
            and prompt_steps == transitions == samples == 0,
            f"run {run_id} observed work was laundered as infrastructure failure",
        )
        status = "invalid_pre_observation"
    elif kind == "run_end_mismatch":
        require(
            end["failed_phase"]
            in {
                "prompt_step",
                "sample_frontier",
                "target_transition",
                "sampler_accounting",
                "terminal_boundary",
                "continuation",
                "continuation_frontier",
            },
            f"run {run_id} failed phase invalid",
        )
        require(
            end["status"] == "mismatch"
            and end["e0_status"] == "development_failed"
            and end["authority"] == AUTHORITY
            and end["generated_ids"] == generated
            and end["generated_ids_sha256_i32le"] == tokens_digest(generated),
            f"run {run_id} mismatch terminal invalid",
        )
        require(
            end["prompt_steps"] == prompt_steps
            and end["target_transitions"] == transitions
            and end["sample_frontiers"] == samples
            and end["continuation_compared"] is False,
            f"run {run_id} mismatch accounting invalid",
        )
        for key in ("serial_sampler_draws", "capture_sampler_draws"):
            integer(end[key], f"run {run_id} {key}", 0)
        if not observed_failure:
            require(
                end["serial_sampler_draws"] == end["capture_sampler_draws"] == samples,
                f"run {run_id} mismatch draw accounting invalid",
            )
        else:
            require(
                end["failed_phase"] == observed_failure_phase,
                f"run {run_id} run_end failure phase disagrees with observed failure",
            )
        require(
            mismatch_detected or observed_failure,
            f"run {run_id} producer claims mismatch without independently observed failure",
        )
        status = "development_failed"
    else:
        require(
            phase == "done"
            and terminal_seen
            and continuation_seen
            and not observed_failure,
            f"run {run_id} successful run is incomplete or observed failure",
        )
        require(not mismatch_detected, f"run {run_id} successful run contains mismatch")
        require(
            end["status"] == "ok"
            and end["e0_status"] == "development_lockstep_passed"
            and end["authority"] == AUTHORITY
            and end["packed_verifier_status"] == "not_measured",
            f"run {run_id} success authority/status invalid",
        )
        require(
            end["generated_ids"] == generated
            and end["generated_ids_sha256_i32le"] == tokens_digest(generated),
            f"run {run_id} final stream mismatch",
        )
        require(
            end["prompt_steps"] == prompt_steps
            and end["target_transitions"] == transitions
            and end["sample_frontiers"] == samples
            and end["serial_sampler_draws"] == end["capture_sampler_draws"] == samples,
            f"run {run_id} success accounting mismatch",
        )
        require(
            end["continuation_compared"] is True
            and end["continuation_equal"] is True
            and end["all_comparisons"] is True
            and end["final_consumed_prefix_len"] == len(prefix)
            and end["final_dflash_target_ctx_n"] == len(prefix),
            f"run {run_id} producer success claims disagree",
        )
        validate_stop(end, generated, config, "run_end")
        status = "development_lockstep_passed"
    if "elapsed_seconds_f64_bits" in end:
        bits(end["elapsed_seconds_f64_bits"], 64, "elapsed")
        require(
            end["timing_semantics"] == "diagnostic_only_non_performance",
            f"run {run_id} timing authority invalid",
        )
    return {
        "run_id": run_id,
        "status": status,
        "development_lockstep_passed": status == "development_lockstep_passed",
        "observed_failure": observed_failure,
        "emitted": len(generated),
        "transitions": prompt_steps + transitions + int(continuation_seen),
    }


def validate_stop(
    payload: dict[str, Any], generated: list[int], config: dict[str, Any], name: str
) -> None:
    require(generated, f"{name} empty stream")
    terminal = generated[-1]
    eos = terminal in config["stop_tokens"]
    limit = len(generated) >= config["tokens"]
    require(
        payload["terminal_token"] == terminal
        and payload["eos_hit"] is eos
        and payload["token_limit_hit"] is limit
        and payload["stop_reason"] == ("eos" if eos else "token_limit")
        and (eos or limit),
        f"{name} stop evidence invalid",
    )


def validate_sidecar_summary(
    value: Any,
    expected_path: str,
    sidecar: bytes,
    cursor: int,
    require_full_coverage: bool,
    name: str,
) -> None:
    summary = keys(value, {"path", "bytes", "sha256"}, name)
    require(
        summary["path"] == expected_path
        and integer(summary["bytes"], f"{name}.bytes", 0) == len(sidecar)
        and summary["sha256"] == sha_bytes(sidecar),
        f"{name} identity mismatch",
    )
    require(cursor <= len(sidecar), f"{name} consumed beyond sidecar")
    if require_full_coverage:
        require(cursor == len(sidecar), f"{name} contains unreferenced bytes")


def validate_boundary(
    payload: dict[str, Any],
    prefix: list[int],
    generated: list[int],
    context: bytes,
    common: dict[str, Any],
    sidecar: bytes,
    sidecar_cursor: list[int],
    run_id: str,
    draws: int,
) -> bool:
    require(
        payload["generated_ids"] == generated
        and payload["generated_ids_sha256_i32le"] == tokens_digest(generated),
        f"run {run_id} boundary stream mismatch",
    )
    validate_stop(payload, generated, common["config"], "terminal boundary")
    require(
        payload["pending_token"] == generated[-1]
        and payload["consumed_prefix_len"] == len(prefix)
        and payload["dflash_target_ctx_n"] == len(prefix)
        and payload["serial_sampler_draws"]
        == payload["capture_sampler_draws"]
        == draws,
        f"run {run_id} boundary accounting mismatch",
    )
    serial = validate_state(
        payload["serial_state"],
        prefix,
        generated[-1],
        common,
        sidecar,
        sidecar_cursor,
        "boundary serial",
    )
    capture = validate_state(
        payload["capture_state"],
        prefix,
        generated[-1],
        common,
        sidecar,
        sidecar_cursor,
        "boundary capture",
    )
    derived_state = state_comparisons(serial, capture)
    derived = {
        "state": derived_state,
        "target_ctx_length": len(context)
        == len(prefix)
        * len(common["binding"]["drafter"]["target_layer_ids"])
        * common["binding"]["target"]["hidden_size"]
        * 4,
    }
    derived["all"] = derived_state["all"] and derived["target_ctx_length"]
    comparisons = keys(payload["comparisons"], set(derived), "boundary comparisons")
    validate_claim(comparisons["state"], derived_state, "boundary state comparisons")
    require(
        comparisons["target_ctx_length"] is derived["target_ctx_length"]
        and comparisons["all"] is derived["all"],
        "boundary comparison disagreement",
    )
    if derived["all"]:
        require(
            payload["first_state_mismatch"] is None,
            "boundary false mismatch diagnostic",
        )
    validate_mismatch(payload["first_state_mismatch"], "boundary first_state_mismatch")
    return derived["all"]


def reduce(paths: list[Path]) -> dict[str, Any]:
    require(len(paths) == 1, "E0 reduction requires exactly one input trace")
    rows, inputs = read_inputs(paths)
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["run_id"]].append(row)
    require(grouped, "no runs found")
    require(len(grouped) == 1, "E0 reduction requires exactly one run")
    runs = [reduce_run(grouped[run_id]) for run_id in sorted(grouped)]
    passed = all(run["development_lockstep_passed"] for run in runs)
    if passed:
        gate = "development_lockstep_passed"
    elif any(run["status"] == "development_failed" for run in runs):
        gate = "development_failed"
    else:
        gate = "invalid_pre_observation"
    result = {
        "schema": REDUCTION_SCHEMA,
        "schema_version": REDUCTION_VERSION,
        "authority": AUTHORITY,
        "development_gate": gate,
        "development_lockstep_passed": passed,
        "aggregate": {
            "runs": len(runs),
            "passed": sum(run["development_lockstep_passed"] for run in runs),
            "invalid_pre_observation": sum(
                run["status"] == "invalid_pre_observation" for run in runs
            ),
            "development_failed": sum(
                run["status"] == "development_failed" for run in runs
            ),
        },
        "runs": runs,
        "inputs": inputs,
        "reducer": {"path": str(SCRIPT), "sha256": sha_bytes(SCRIPT.read_bytes())},
    }
    finite_json(result, "reduction")
    return result


def synthetic_rows(run_id: str = "pass", tokens: int = 2) -> list[dict[str, Any]]:
    def gguf(name: str) -> dict[str, Any]:
        item = sha_bytes(name.encode())
        aggregate = sha_bytes(struct.pack("<QQ", 0, len(name)) + bytes.fromhex(item))
        return {
            "aggregate_sha256_index_size_digest_le": aggregate,
            "shards": [
                {"index": 0, "path": f"/{name}", "bytes": len(name), "sha256": item}
            ],
        }

    target_asset, drafter_asset = gguf("target"), gguf("drafter")
    asset = bytes.fromhex(target_asset["aggregate_sha256_index_size_digest_le"])
    identity = {
        "model_id": struct.unpack("<Q", asset[:8])[0],
        "tokenizer_id": struct.unpack("<Q", asset[8:16])[0],
        "layout_version": 1,
        "n_attn_layers": 1,
        "n_gdn_layers": 0,
        "kv_dim_elements": 2,
        "kv_bytes_per_token": 4,
        "kv_storage_kind": "F16",
        "gdn_state_elements_per_layer": 0,
        "gdn_conv_elements_per_layer": 0,
    }
    source_state = sha_bytes(b"source")
    boot = {
        "schema": SCHEMA,
        "schema_version": VERSION,
        "run_id": run_id,
        "build_identity": {
            "schema_version": 2,
            "build_commit": "1" * 40,
            "build_commit_short": "1" * 9,
            "build_dirty": False,
            "build_source_state": f"git-source-sha256-v2:{source_state}",
            "stamp_source": "git",
            "stamp_error": None,
            "runtime_commit": "1" * 40,
            "runtime_dirty": False,
            "runtime_source_state": f"git-source-sha256-v2:{source_state}",
            "status": "match",
            "problems": [],
            "overrides": [],
        },
        "lease_env": {"QWEN_METAL_LEASE_WAIT": "1"},
        "command": ["qwen-bench"],
        "paths": {
            "model": "/target",
            "drafter": "/drafter",
            "binding_manifest": "/manifest",
            "output": "/trace",
            "state_sidecar": "/state",
            "executable": "/bin",
        },
    }
    common = {
        **boot,
        "classification": {
            "evidence_role": "development",
            "fixture_id": "self-test",
            "fixture_role": "sentinel",
            "target_arm": "serial",
            "drafter_arm": "capture",
            "authority": AUTHORITY,
        },
        "config": {
            "tokens": tokens,
            "context_capacity": tokens + 34,
            "stop_tokens": [],
            "temperature_f32_bits": enc32(1.0),
            "top_k": 2,
            "top_p_f32_bits": enc32(1.0),
            "min_p_f32_bits": enc32(0.0),
            "seed": 0,
            "sampler_algorithm_version": 1,
            "no_warmup": True,
            "arm_order": "serial_then_capture",
            "semantics": SEMANTICS,
            "hidden_transfer_semantics": HIDDEN_SEMANTICS,
        },
        "assets": {
            "target": target_asset,
            "drafter": drafter_asset,
            "executable": {"path": "/bin", "bytes": 1, "sha256": sha_bytes(b"bin")},
        },
        "binding_manifest": {
            "path": "/manifest",
            "bytes": 1,
            "sha256": sha_bytes(b"x"),
        },
        "binding": {
            "target_architecture": "test",
            "target": {"n_layer": 1, "hidden_size": 2, "vocab_size": 4},
            "drafter": {
                "n_layer": 1,
                "hidden_size": 2,
                "block_size": 2,
                "swa_window": 0,
                "conv_kernel_size": 2,
                "conv_group_size": 1,
                "selector_rank": 1,
                "selector_top_k": 16,
                "target_layer_ids": [0],
            },
            "resolved_stop_tokens": [],
        },
        "host": {"os": "test", "arch": "test", "metal_device": "test"},
        "prompt": {
            "utf8_len": 1,
            "utf8_sha256": sha_bytes(b"p"),
            "token_count": 1,
            "token_ids_sha256_i32le": tokens_digest([1]),
        },
        "sessions": {
            "serial_target": f"{run_id}/serial",
            "capture_target": f"{run_id}/capture",
            "capture_drafter": f"{run_id}/drafter",
        },
    }

    def event(
        base: dict[str, Any], kind: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {**base, "event": kind, "payload": payload}

    empty = sha_bytes(b"")
    sidecar_offset = 0

    def snapshot(prefix: list[int], pending: int | None = None) -> dict[str, Any]:
        nonlocal sidecar_offset
        size = len(prefix) * 4
        raw_sections = {
            "kv_k": bytes(size),
            "kv_v": bytes(size),
            "gdn_conv": b"",
            "gdn_state": b"",
        }
        references = {}
        for key in ("kv_k", "kv_v", "gdn_conv", "gdn_state"):
            raw = raw_sections[key]
            references[key] = {
                "offset": sidecar_offset,
                "bytes": len(raw),
                "sha256": sha_bytes(raw),
            }
            sidecar_offset += len(raw)
        return {
            "identity": identity,
            "identity_sha256_canonical_le": identity_digest(identity),
            "prefix_len": len(prefix),
            "prefix_token_ids_sha256_i32le": tokens_digest(prefix),
            "pending_token": pending,
            "pending_token_sha256_tagged_i32le": pending_digest(pending),
            "kv_positions": [len(prefix)],
            "kv_positions_sha256_u64le": positions_digest([len(prefix)]),
            "sections": {
                "kv_k": {"bytes": size, "sha256": sha_bytes(bytes(size))},
                "kv_v": {"bytes": size, "sha256": sha_bytes(bytes(size))},
                "gdn_conv": {"bytes": 0, "sha256": empty},
                "gdn_state": {"bytes": 0, "sha256": empty},
            },
            "section_sidecars": references,
            "final_logits_present": False,
            "capture_tail_present": False,
        }

    state_claim = {key: True for key in STATE_COMPARE_KEYS}
    logits_raw = struct.pack("<4f", 1, 0, -1, -2)
    hidden_rows: list[bytes] = []

    def transition(
        prefix: list[int], token: int, phase: str, step: int
    ) -> dict[str, Any]:
        consumed = prefix + [token]
        source = struct.pack("<2f", float(len(consumed)), 2.0)
        hidden_rows.append(source)
        arm = lambda api: {
            "api": api,
            "logits_count": 4,
            "logits_sha256_f32le": sha_bytes(logits_raw),
            "logits_f32le_hex": logits_raw.hex(),
            "state": snapshot(consumed),
        }
        comparisons = {
            "logits_bits": True,
            "state": state_claim,
            "hidden_overwrite_complete": True,
            "hidden_values_finite": True,
            "hidden_transfer_bytes": True,
            "active_context_history": True,
            "active_position_history": True,
            "target_ctx_length": True,
            "target_ctx_position": True,
            "target_ctx_watermarks": True,
            "all": True,
        }
        active = b"".join(hidden_rows)
        positions = list(range(len(consumed)))
        return {
            "phase": phase,
            "step_index": step,
            "token": token,
            "position": len(prefix),
            "consumed_prefix_len": len(consumed),
            "consumed_prefix_sha256_i32le": tokens_digest(consumed),
            "arm_order": "serial_then_capture",
            "serial": arm("single_token"),
            "capture": arm("single_token_with_multi_hidden"),
            "hidden_transfer": {
                "semantics": HIDDEN_SEMANTICS,
                "target_layer_ids": [0],
                "shape": [1, 2],
                "source_bytes": 8,
                "source_sha256_f32le": sha_bytes(source),
                "source_f32le_hex": source.hex(),
                "poison_f32_bits": "0x7fa5a5a5",
                "first_retained_poison_index": None,
                "first_nonfinite_index": None,
                "destination_bytes": 8,
                "destination_sha256_f32le": sha_bytes(source),
                "destination_f32le_hex": source.hex(),
                "active_context_bytes": len(active),
                "active_context_sha256_f32le": sha_bytes(active),
                "active_context_f32le_hex": active.hex(),
                "expected_active_context_sha256_f32le": sha_bytes(active),
                "active_positions": positions,
                "expected_active_positions": positions,
                "target_ctx_index": len(prefix),
                "target_ctx_n_after": len(consumed),
                "captured_position": len(prefix),
                "ctx_h_ready_n": 0,
                "kv_ctx_ready_n": 0,
            },
            "comparisons": comparisons,
            "first_mismatch": {
                "logits": None,
                "state": None,
                "hidden_transfer": None,
                "active_context": None,
            },
        }

    rng = rng_initial(0)

    def sample(
        index: int, prefix: list[int], continuation: bool = False
    ) -> dict[str, Any]:
        nonlocal rng
        raw, after = rng_next(rng)
        unit = (raw >> 11) / float(1 << 53)
        rng_payload = {
            "draws_before": index,
            "draws_after": index + 1,
            "state_before": state_hex(rng),
            "state_after": state_hex(after),
            "raw_u64": f"0x{raw:016x}",
            "raw_uniform_f64_bits": enc64(unit),
        }
        prepared = sampler_v1_distribution(logits_raw, common["config"])
        prepared_weights = [
            bits(value, 64, "synthetic prepared weight")
            for value in prepared["weight_bits"]
        ]
        candidate_index = categorical(prepared_weights, unit)
        token = prepared["tokens"][candidate_index]
        distribution = {
            "selected_token": token,
            "candidate_index": candidate_index,
            "ordered_support": [
                {"token": prepared_token, "weight_f64_bits": weight_bits}
                for prepared_token, weight_bits in zip(
                    prepared["tokens"], prepared["weight_bits"], strict=True
                )
            ],
            "total_weight_f64_bits": prepared["total_weight_bits"],
        }
        if continuation:
            arm = {
                "distribution": distribution,
                "rng": rng_payload,
                "live_draws_unchanged": index,
            }
            return {
                "serial": copy.deepcopy(arm),
                "capture": copy.deepcopy(arm),
                "comparisons": {
                    "distribution_bits": True,
                    "rng_transition": True,
                    "diagnostics_nonadvancing": True,
                    "all": True,
                },
                "first_distribution_mismatch": None,
            }
        rng = after
        arm = {
            "distribution": distribution,
            "rng": rng_payload,
            "live_sample": {
                "token": token,
                "candidate_index": candidate_index,
                "draws_after": index + 1,
            },
        }
        terminal = index + 1 >= tokens
        return {
            "sample_index": index,
            "target_position": len(prefix) - 1,
            "consumed_prefix_len": len(prefix),
            "consumed_prefix_sha256_i32le": tokens_digest(prefix),
            "serial_logits_sha256_f32le": sha_bytes(logits_raw),
            "capture_logits_sha256_f32le": sha_bytes(logits_raw),
            "serial": copy.deepcopy(arm),
            "capture": copy.deepcopy(arm),
            "committed_token": token,
            "terminal": terminal,
            "stop_reason": "token_limit" if terminal else None,
            "eos_hit": False,
            "token_limit_hit": terminal,
            "comparisons": {
                "distribution_bits": True,
                "rng_transition": True,
                "diagnosed_vs_live": True,
                "live_sample": True,
                "draw_counts": True,
                "all": True,
            },
            "first_distribution_mismatch": None,
        }

    rows = [
        event(boot, "bootstrap_start", {"started_utc": "1970-01-01T00:00:00Z"}),
        event(boot, "bootstrap_end", {"status": "ok", "observed": False}),
        event(
            common,
            "run_start",
            {
                "started_utc": "1970-01-01T00:00:00Z",
                "protocol": "E0-CLARIFICATION.md",
                "packed_verifier": "excluded_different_distribution",
            },
        ),
    ]
    prefix = []
    rows.append(event(common, "prompt_step", transition(prefix, 1, "prompt", 0)))
    prefix = [1]
    generated = []
    for index in range(tokens):
        sample_payload = sample(index, prefix)
        token = sample_payload["committed_token"]
        rows.append(event(common, "sample_frontier", sample_payload))
        generated.append(token)
        if index + 1 < tokens:
            rows.append(
                event(
                    common,
                    "target_transition",
                    transition(prefix, token, "generated", index),
                )
            )
            prefix.append(token)
    serial_boundary_state = snapshot(prefix, generated[-1])
    capture_boundary_state = snapshot(prefix, generated[-1])
    rows.append(
        event(
            common,
            "terminal_boundary",
            {
                "generated_ids": generated,
                "generated_ids_sha256_i32le": tokens_digest(generated),
                "stop_reason": "token_limit",
                "terminal_token": generated[-1],
                "eos_hit": False,
                "token_limit_hit": True,
                "serial_sampler_draws": tokens,
                "capture_sampler_draws": tokens,
                "consumed_prefix_len": len(prefix),
                "pending_token": generated[-1],
                "serial_state": serial_boundary_state,
                "capture_state": capture_boundary_state,
                "dflash_target_ctx_n": len(prefix),
                "comparisons": {
                    "state": state_claim,
                    "target_ctx_length": True,
                    "all": True,
                },
                "first_state_mismatch": None,
            },
        )
    )
    continuation_transition = transition(prefix, generated[-1], "continuation", 0)
    prefix.append(generated[-1])
    frontier = sample(tokens, prefix, continuation=True)
    rows.append(
        event(
            common,
            "continuation",
            {
                "transition": continuation_transition,
                "next_frontier": frontier,
                "comparisons": {"transition": True, "next_frontier": True, "all": True},
            },
        )
    )
    rows.append(
        event(
            common,
            "run_end",
            {
                "status": "ok",
                "e0_status": "development_lockstep_passed",
                "authority": AUTHORITY,
                "packed_verifier_status": "not_measured",
                "generated_ids": generated,
                "generated_ids_sha256_i32le": tokens_digest(generated),
                "stop_reason": "token_limit",
                "terminal_token": generated[-1],
                "eos_hit": False,
                "token_limit_hit": True,
                "prompt_steps": 1,
                "target_transitions": tokens - 1,
                "sample_frontiers": tokens,
                "serial_sampler_draws": tokens,
                "capture_sampler_draws": tokens,
                "continuation_compared": True,
                "continuation_equal": True,
                "final_consumed_prefix_len": len(prefix),
                "final_dflash_target_ctx_n": len(prefix),
                "all_comparisons": True,
                "elapsed_seconds_f64_bits": enc64(1.0),
                "timing_semantics": "diagnostic_only_non_performance",
                "state_sidecar": {
                    "path": "/state",
                    "bytes": sidecar_offset,
                    "sha256": sha_bytes(bytes(sidecar_offset)),
                },
            },
        )
    )
    return rows


def run_self_test() -> None:
    require(
        rng_initial(0)
        == [
            0xE220A8397B1DCDAF,
            0x6E789E6AA1B965F4,
            0x06C45D188009454F,
            0xF88BB8A8724C81EC,
        ],
        "SplitMix golden vector failed",
    )
    with tempfile.TemporaryDirectory(prefix="dflash-e0-self-test-") as directory:
        root = Path(directory)

        def write(name: str, rows: list[dict[str, Any]]) -> Path:
            path = root / f"{name}.jsonl"
            output_identity = str(path.resolve())
            for row in rows:
                row["paths"]["output"] = output_identity
            run_start = next((row for row in rows if row["event"] == "run_start"), None)
            manifest_path = root / f"{name}.binding.json"
            if run_start is not None:
                first_state = next(
                    row["payload"]["serial"]["state"]
                    for row in rows
                    if row["event"] == "prompt_step"
                )
                identity = first_state["identity"]
                manifest = {
                    "schema": BINDING_MANIFEST_SCHEMA,
                    "schema_version": BINDING_MANIFEST_VERSION,
                    "evidence_role": "development",
                    "fixture_id": run_start["classification"]["fixture_id"],
                    "fixture_role": run_start["classification"]["fixture_role"],
                    "target_asset_sha256": run_start["assets"]["target"][
                        "aggregate_sha256_index_size_digest_le"
                    ],
                    "drafter_asset_sha256": run_start["assets"]["drafter"][
                        "aggregate_sha256_index_size_digest_le"
                    ],
                    "target_arm": run_start["classification"]["target_arm"],
                    "drafter_arm": run_start["classification"]["drafter_arm"],
                    "target": {
                        "architecture": run_start["binding"]["target_architecture"],
                        **run_start["binding"]["target"],
                    },
                    "drafter": run_start["binding"]["drafter"],
                    "snapshot_abi": {
                        key: identity[key]
                        for key in IDENTITY_KEYS - {"model_id", "tokenizer_id"}
                    },
                    "allowed_arm_orders": [
                        "serial_then_capture",
                        "capture_then_serial",
                    ],
                }
                manifest_data = (
                    json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n"
                ).encode()
                manifest_path.write_bytes(manifest_data)
                manifest_identity = {
                    "path": str(manifest_path.resolve()),
                    "bytes": len(manifest_data),
                    "sha256": sha_bytes(manifest_data),
                }
                for row in rows:
                    row["paths"]["binding_manifest"] = manifest_identity["path"]
                    if "binding_manifest" in row:
                        row["binding_manifest"] = manifest_identity
            sidecar_bytes = 0

            def inspect(value: Any) -> None:
                nonlocal sidecar_bytes
                if isinstance(value, dict):
                    if set(value) == {"offset", "bytes", "sha256"}:
                        sidecar_bytes = max(
                            sidecar_bytes,
                            int(value["offset"]) + int(value["bytes"]),
                        )
                    for child in value.values():
                        inspect(child)
                elif isinstance(value, list):
                    for child in value:
                        inspect(child)

            inspect(rows)
            sidecar = root / f"{name}.state.bin"
            sidecar.write_bytes(bytes(sidecar_bytes))
            sidecar_identity = {
                "path": str(sidecar.resolve()),
                "bytes": sidecar_bytes,
                "sha256": sha_bytes(bytes(sidecar_bytes)),
            }
            for row in rows:
                row["paths"]["state_sidecar"] = sidecar_identity["path"]
                if row["event"] == "run_end":
                    row["payload"]["state_sidecar"] = sidecar_identity
            path.write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            return path

        require(
            reduce([write("pass", synthetic_rows())])["development_lockstep_passed"],
            "synthetic pass failed",
        )
        require(
            reduce([write("one", synthetic_rows("one", 1))])["runs"][0]["emitted"] == 1,
            "one-token boundary failed",
        )
        first = write("single-input-a", synthetic_rows("single-input-a"))
        second = write("single-input-b", synthetic_rows("single-input-b"))
        try:
            reduce([first, second])
        except EvidenceError:
            pass
        else:
            raise AssertionError("multiple input traces were accepted")
        multi_run = synthetic_rows("multi-a") + synthetic_rows("multi-b")
        try:
            reduce([write("multi-run", multi_run)])
        except EvidenceError:
            pass
        else:
            raise AssertionError("multiple runs in one trace were accepted")
        bootstrap = synthetic_rows("bootstrap")[:2]
        bootstrap[1]["payload"] = {
            "status": "infrastructure_error",
            "observed": False,
            "replaceable": False,
            "error": "synthetic",
        }
        result = reduce([write("bootstrap", bootstrap)])
        require(
            result["development_gate"] == "invalid_pre_observation",
            "bootstrap failure not retained",
        )
        retained = synthetic_rows("retained-observed")[:4]
        retained.append(
            event_from(
                retained[2],
                "observed_failure",
                {
                    "phase": "sample_frontier",
                    "step_index": 0,
                    "token": None,
                    "position": 0,
                    "error": "synthetic observed failure",
                    "observed_e0_failure": True,
                    "replaceable_infrastructure_failure": False,
                },
            )
        )
        retained.append(
            event_from(
                retained[2],
                "run_end",
                {
                    "status": "mismatch",
                    "e0_status": "development_failed",
                    "authority": AUTHORITY,
                    "failed_phase": "sample_frontier",
                    "prompt_steps": 1,
                    "target_transitions": 0,
                    "sample_frontiers": 0,
                    "generated_ids": [],
                    "generated_ids_sha256_i32le": tokens_digest([]),
                    "serial_sampler_draws": 0,
                    "capture_sampler_draws": 0,
                    "continuation_compared": False,
                    "elapsed_seconds_f64_bits": enc64(1.0),
                    "timing_semantics": "diagnostic_only_non_performance",
                },
            )
        )
        require(
            reduce([write("retained-observed", retained)])["runs"][0]["status"]
            == "development_failed",
            "observed failure was not retained",
        )

        over_advanced = synthetic_rows("over-advanced")[:4]
        transition_payload = over_advanced[-1]["payload"]
        hidden = transition_payload["hidden_transfer"]
        expected_active = bytes.fromhex(hidden["active_context_f32le_hex"])
        observed_active = expected_active + bytes.fromhex(hidden["source_f32le_hex"])
        hidden["active_context_bytes"] = len(observed_active)
        hidden["active_context_sha256_f32le"] = sha_bytes(observed_active)
        hidden["active_context_f32le_hex"] = observed_active.hex()
        hidden["active_positions"] = [0, 1]
        hidden["target_ctx_n_after"] = 2
        transition_payload["comparisons"]["active_context_history"] = False
        transition_payload["comparisons"]["active_position_history"] = False
        transition_payload["comparisons"]["target_ctx_length"] = False
        transition_payload["comparisons"]["all"] = False
        transition_payload["first_mismatch"]["active_context"] = {
            "offset": len(expected_active),
            "serial_len": len(expected_active),
            "capture_len": len(observed_active),
        }
        over_advanced.append(
            event_from(
                over_advanced[2],
                "run_end",
                {
                    "status": "mismatch",
                    "e0_status": "development_failed",
                    "authority": AUTHORITY,
                    "failed_phase": "prompt_step",
                    "prompt_steps": 1,
                    "target_transitions": 0,
                    "sample_frontiers": 0,
                    "generated_ids": [],
                    "generated_ids_sha256_i32le": tokens_digest([]),
                    "serial_sampler_draws": 0,
                    "capture_sampler_draws": 0,
                    "continuation_compared": False,
                    "elapsed_seconds_f64_bits": enc64(1.0),
                    "timing_semantics": "diagnostic_only_non_performance",
                },
            )
        )
        require(
            reduce([write("over-advanced", over_advanced)])["runs"][0]["status"]
            == "development_failed",
            "over-advanced context was not retained as an E0 failure",
        )

        length_mismatch = synthetic_rows("length-mismatch")[:4]
        transition_payload = length_mismatch[-1]["payload"]
        capture_logits = bytes.fromhex(
            transition_payload["capture"]["logits_f32le_hex"]
        )[:-4]
        transition_payload["capture"]["logits_count"] = 3
        transition_payload["capture"]["logits_f32le_hex"] = capture_logits.hex()
        transition_payload["capture"]["logits_sha256_f32le"] = sha_bytes(capture_logits)
        transition_payload["comparisons"]["logits_bits"] = False
        transition_payload["comparisons"]["all"] = False
        transition_payload["first_mismatch"]["logits"] = {
            "index": 3,
            "serial_len": 4,
            "capture_len": 3,
        }
        length_mismatch.append(
            event_from(
                length_mismatch[2],
                "run_end",
                {
                    "status": "mismatch",
                    "e0_status": "development_failed",
                    "authority": AUTHORITY,
                    "failed_phase": "prompt_step",
                    "prompt_steps": 1,
                    "target_transitions": 0,
                    "sample_frontiers": 0,
                    "generated_ids": [],
                    "generated_ids_sha256_i32le": tokens_digest([]),
                    "serial_sampler_draws": 0,
                    "capture_sampler_draws": 0,
                    "continuation_compared": False,
                    "elapsed_seconds_f64_bits": enc64(1.0),
                    "timing_semantics": "diagnostic_only_non_performance",
                },
            )
        )
        require(
            reduce([write("length-mismatch", length_mismatch)])["runs"][0]["status"]
            == "development_failed",
            "logit-length mismatch was not retained as an E0 failure",
        )

        def reject(name: str, mutate: Any) -> None:
            rows = synthetic_rows(name)
            mutate(rows)
            try:
                reduce([write(name, rows)])
            except EvidenceError:
                return
            raise AssertionError(f"adversarial mutation accepted: {name}")

        first_transition = lambda rows: next(
            row for row in rows if row["event"] == "prompt_step"
        )["payload"]
        first_sample = lambda rows: next(
            row for row in rows if row["event"] == "sample_frontier"
        )["payload"]
        reject(
            "malformed-hex",
            lambda r: first_transition(r)["serial"].__setitem__(
                "logits_f32le_hex", "xx"
            ),
        )

        def omit_run_executable(rows: list[dict[str, Any]]) -> None:
            for row in rows[2:]:
                row["paths"] = {
                    key: value
                    for key, value in row["paths"].items()
                    if key != "executable"
                }

        reject("run-path-drift", omit_run_executable)

        def alias_artifact_paths(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                row["paths"]["drafter"] = row["paths"]["model"]

        reject("aliased-artifact-paths", alias_artifact_paths)
        reject(
            "logit-mismatch",
            lambda r: first_transition(r)["capture"].__setitem__(
                "logits_f32le_hex", "00" * 16
            ),
        )
        reject(
            "oov-zero",
            lambda r: first_sample(r)["serial"]["distribution"][
                "ordered_support"
            ].append({"token": 4, "weight_f64_bits": enc64(0.0)}),
        )

        def forge_in_vocab_support(rows: list[dict[str, Any]]) -> None:
            payload = first_sample(rows)
            token = payload["committed_token"]
            for arm_name in ("serial", "capture"):
                arm = payload[arm_name]
                arm["distribution"]["candidate_index"] = 0
                arm["distribution"]["ordered_support"] = [
                    {"token": token, "weight_f64_bits": enc64(1.0)}
                ]
                arm["distribution"]["total_weight_f64_bits"] = enc64(1.0)
                arm["live_sample"]["candidate_index"] = 0

        reject("forged-in-vocab-support", forge_in_vocab_support)
        reject(
            "rng-tamper",
            lambda r: first_sample(r)["serial"]["rng"].__setitem__(
                "raw_u64", "0x0000000000000000"
            ),
        )
        reject(
            "retained-poison",
            lambda r: first_transition(r)["hidden_transfer"].__setitem__(
                "source_f32le_hex", struct.pack("<II", 0x7FA5A5A5, 0x40000000).hex()
            ),
        )
        reject(
            "nonfinite-hidden",
            lambda r: first_transition(r)["hidden_transfer"].__setitem__(
                "source_f32le_hex", struct.pack("<II", 0x7F800000, 0x40000000).hex()
            ),
        )
        reject(
            "prior-context",
            lambda r: next(row for row in r if row["event"] == "target_transition")[
                "payload"
            ]["hidden_transfer"].__setitem__("active_context_sha256_f32le", "0" * 64),
        )
        reject(
            "snapshot-geometry",
            lambda r: first_transition(r)["serial"]["state"]["sections"][
                "kv_k"
            ].__setitem__("bytes", 0),
        )
        reject(
            "snapshot-state",
            lambda r: first_transition(r)["serial"]["state"]["sections"][
                "kv_k"
            ].__setitem__("sha256", "f" * 64),
        )

        def forge_equal_state_hashes(rows: list[dict[str, Any]]) -> None:
            transition_payload = first_transition(rows)
            for arm_name in ("serial", "capture"):
                state = transition_payload[arm_name]["state"]
                state["sections"]["kv_k"]["sha256"] = "f" * 64
                state["section_sidecars"]["kv_k"]["sha256"] = "f" * 64

        reject("forged-equal-state-hashes", forge_equal_state_hashes)
        reject("event-reorder", lambda r: r.__setitem__(slice(3, 5), [r[4], r[3]]))
        reject(
            "event-omission",
            lambda r: r.pop(
                next(
                    i for i, row in enumerate(r) if row["event"] == "target_transition"
                )
            ),
        )
        reject(
            "boolean-tamper",
            lambda r: first_transition(r)["comparisons"].__setitem__("all", False),
        )
        reject(
            "continuation-advances",
            lambda r: next(row for row in r if row["event"] == "continuation")[
                "payload"
            ]["next_frontier"]["serial"].__setitem__("live_draws_unchanged", 3),
        )

        def dirty_build(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                build = row["build_identity"]
                build["build_dirty"] = True
                build["runtime_dirty"] = True
                build["status"] = "dirty"
                build["problems"] = ["dirty"]

        reject("dirty-build", dirty_build)
        reject(
            "observed-failure",
            lambda r: r.insert(
                -1,
                event_from(
                    r[2],
                    "observed_failure",
                    {
                        "phase": "continuation",
                        "step_index": 0,
                        "token": None,
                        "position": None,
                        "error": "synthetic",
                        "observed_e0_failure": True,
                        "replaceable_infrastructure_failure": False,
                    },
                ),
            ),
        )
        occupied = root / "occupied.json"
        occupied.write_text("occupied", encoding="ascii")
        try:
            write_output({}, occupied)
        except EvidenceError:
            pass
        else:
            raise AssertionError("exclusive output test failed")
    print("self-test: PASS")


def event_from(
    row: dict[str, Any], kind: str, payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        key: value for key, value in row.items() if key not in {"event", "payload"}
    } | {"event": kind, "payload": payload}


def write_output(value: dict[str, Any], output: Path | None) -> None:
    serialized = (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    )
    if output is None:
        sys.stdout.write(serialized)
        return
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise EvidenceError(f"refusing to overwrite output: {output}") from error
    except OSError as error:
        raise EvidenceError(f"cannot write output {output}: {error}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly reduce schema-v1 DFlash E0 lockstep evidence."
    )
    parser.add_argument("--input", action="append", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Exclusive-create compact reduction path"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test and (args.input or args.output):
        parser.error("--self-test cannot be combined with input/output")
    if not args.self_test and not args.input:
        parser.error("at least one --input is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            run_self_test()
        else:
            write_output(
                reduce([path for group in args.input for path in group]), args.output
            )
        return 0
    except (EvidenceError, OSError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
