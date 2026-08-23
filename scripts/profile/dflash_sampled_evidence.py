#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
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


SCHEMA = "qwen.dflash_sampled_oracle"
SCHEMA_VERSION = 4
SAMPLER_ALGORITHM_VERSION = 1
DEVELOPMENT_SEMANTICS = (
    "exact_serial_e1a_one_hot_non_performance_e0_unmeasured_"
    "product_aligned_packed_prompt_prefill"
)
PROMPT_PREFILL_SEMANTICS = (
    "normative_product_aligned_packed_prefill_tokens_with_multi_hidden"
)
STATUS = "development_diagnostic_only_e0_unmeasured_no_product_authority"
UPSTREAM_CONTRACTS = {
    "vllm": "b389ac29465b33f9e9c534df221ea3c129e9793f",
    "llama.cpp": "1deefcca395743049c3820ab8f9b15043f3e9446",
}
SCRIPT = Path(__file__).resolve()
MASK64 = (1 << 64) - 1
COMMON_KEYS = {
    "schema",
    "schema_version",
    "run_id",
    "build_identity",
    "lease_env",
    "classification",
    "config",
    "assets",
    "binding",
    "command",
    "host",
    "paths",
    "prompt",
    "sessions",
    "event",
    "payload",
}
PAYLOAD_KEYS = {
    "run_start": {"started_utc"},
    "sample_decision": {
        "sample_index",
        "frontier",
        "target_position",
        "logits_sha256_f32le",
        "distribution",
        "rng",
        "live_draws_after",
    },
    "reference_sample_decision": {
        "sample_index",
        "frontier",
        "target_position",
        "logits_sha256_f32le",
        "distribution",
        "rng",
        "live_draws_after",
    },
    "one_hot_decision": {
        "block_index",
        "depth",
        "carry_in",
        "proposal",
        "sampled_target",
        "accepted",
        "terminal",
        "stop_reason",
        "terminal_token",
        "eos_hit",
        "token_limit_hit",
        "proposal_present_in_support",
        "proposal_weight_f64_bits",
        "total_weight_f64_bits",
        "proposal_probability_f64_bits",
        "draw_index",
    },
    "proposal_block": {
        "block_index",
        "noise_start_pos",
        "carry_in",
        "proposals",
        "accepted",
        "mismatch_depth",
        "selector_depths",
    },
    "serial_reference_end": {
        "generated_ids",
        "generated_ids_sha256_i32le",
        "stop_reason",
        "terminal_token",
        "eos_hit",
        "token_limit_hit",
        "sampler_draws",
        "transition_logits_sha256_f32le",
        "target_kv_positions",
        "target_state",
        "target_state_comparisons",
        "continuation_position",
        "continuation_logits_sha256_f32le",
        "comparisons",
    },
    "run_end": {
        "status",
        "generated_ids",
        "generated_ids_sha256_i32le",
        "stop_reason",
        "terminal_token",
        "eos_hit",
        "token_limit_hit",
        "sampler_draws",
        "transition_logits_sha256_f32le",
        "target_kv_positions",
        "target_state",
        "target_state_comparisons",
        "dflash_target_ctx_n",
        "processed_target_position",
        "consumed_prefix_len",
        "continuation_position",
        "continuation_logits_sha256_f32le",
        "continuation_boundary_equal",
        "one_token_continuation_compared",
        "one_token_continuation_equal",
        "blocks",
        "attempts_by_depth",
        "accepts_by_depth",
        "elapsed_seconds_f64_bits",
        "timing_semantics",
    },
}
OPTIONAL_DIAGNOSTIC_KEYS = {"diagnostics"}


class EvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def integer(value: Any, name: str, minimum: int | None = None) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    if minimum is not None:
        require(value >= minimum, f"{name} must be >= {minimum}")
    return value


def token_id(value: Any, name: str, vocab_size: int) -> int:
    token = integer(value, name, 0)
    require(token < vocab_size, f"{name} is outside the recorded target vocabulary")
    return token


def finite_number(value: Any, name: str, positive: bool = False) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{name} must be finite")
    if positive:
        require(result > 0.0, f"{name} must be positive")
    return result


def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON constant {value}")


def validate_json_numbers(value: Any, where: str = "JSON") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{where} contains a non-finite number")
    elif isinstance(value, dict):
        for key, child in value.items():
            validate_json_numbers(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_json_numbers(child, f"{where}[{index}]")


def decode_bits(value: Any, width: int, name: str) -> float:
    digits = width // 4
    require(
        isinstance(value, str)
        and len(value) == digits + 2
        and value.startswith("0x")
        and all(char in "0123456789abcdef" for char in value[2:]),
        f"{name} must be canonical 0x-prefixed lowercase {width}-bit hex",
    )
    raw = int(value[2:], 16)
    decoded = struct.unpack(
        ">f" if width == 32 else ">d", raw.to_bytes(width // 8, "big")
    )[0]
    require(math.isfinite(decoded), f"{name} decodes to a non-finite value")
    return decoded


def validate_hex_bits(value: Any, width: int, name: str) -> None:
    digits = width // 4
    require(
        isinstance(value, str)
        and len(value) == digits + 2
        and value.startswith("0x")
        and all(char in "0123456789abcdef" for char in value[2:]),
        f"{name} must be canonical 0x-prefixed lowercase {width}-bit hex",
    )


def validate_sha256(value: Any, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value),
        f"{name} must be canonical lowercase SHA-256 hex",
    )
    return value


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )


def require_keys(value: dict[str, Any], required: set[str], name: str) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - OPTIONAL_DIAGNOSTIC_KEYS
    require(not missing, f"{name} missing required keys: {sorted(missing)}")
    require(not extra, f"{name} has unexpected keys: {sorted(extra)}")


def rotate_left_u64(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (64 - shift))) & MASK64


def splitmix64_next(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (value ^ (value >> 31)) & MASK64


def xoshiro256pp_initial_state(seed: int) -> list[int]:
    require(0 <= seed <= MASK64, "sampler seed must fit u64")
    splitmix_state = seed
    result = []
    for _ in range(4):
        splitmix_state, word = splitmix64_next(splitmix_state)
        result.append(word)
    return result


def xoshiro256pp_next(state: list[int]) -> tuple[int, list[int]]:
    require(len(state) == 4, "xoshiro256++ state must contain four words")
    s0, s1, s2, s3 = state
    raw = (rotate_left_u64((s0 + s3) & MASK64, 23) + s0) & MASK64
    temporary = (s1 << 17) & MASK64
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= temporary
    s3 = rotate_left_u64(s3, 45)
    return raw, [s0 & MASK64, s1 & MASK64, s2 & MASK64, s3 & MASK64]


def state_hex(state: list[int]) -> list[str]:
    return [f"0x{word:016x}" for word in state]


def categorical_replay_index(weights: list[float], unit: float) -> int:
    require(weights, "categorical replay requires candidates")
    total = 0.0
    for weight in weights:
        total += weight
    target = unit * total
    selected = len(weights) - 1
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if target < cumulative:
            selected = index
            break
    return selected


def encode_f32(value: float) -> str:
    return f"0x{struct.unpack('>I', struct.pack('>f', value))[0]:08x}"


def encode_f64(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def validate_bit_fields(value: Any, where: str = "row") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("_f32_bits") and child is not None:
                if isinstance(child, list):
                    for index, item in enumerate(child):
                        if item is not None:
                            decode_bits(item, 32, f"{where}.{key}[{index}]")
                else:
                    decode_bits(child, 32, f"{where}.{key}")
            elif key.endswith("_f64_bits") and child is not None:
                if isinstance(child, list):
                    for index, item in enumerate(child):
                        if item is not None:
                            decode_bits(item, 64, f"{where}.{key}[{index}]")
                else:
                    decode_bits(child, 64, f"{where}.{key}")
            elif key == "raw_u64" and child is not None:
                validate_hex_bits(child, 64, f"{where}.{key}")
            elif key in {"state_before", "state_after"} and child is not None:
                require(
                    isinstance(child, list) and len(child) == 4,
                    f"{where}.{key} must contain four u64 words",
                )
                for index, word in enumerate(child):
                    validate_hex_bits(word, 64, f"{where}.{key}[{index}]")
            validate_bit_fields(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_bit_fields(child, f"{where}[{index}]")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_inputs(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path_arg in paths:
        path = path_arg.resolve()
        require(path not in seen, f"duplicate --input path: {path}")
        seen.add(path)
        try:
            data = path.read_bytes()
        except OSError as error:
            raise EvidenceError(f"cannot read {path}: {error}") from error
        require(data, f"input is empty: {path}")
        require(data.endswith(b"\n"), f"incomplete final JSONL line in {path}")
        inputs.append(
            {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data)}
        )
        for line_number, raw in enumerate(data.splitlines(), 1):
            require(raw.strip(), f"blank JSONL line at {path}:{line_number}")
            try:
                text = raw.decode("utf-8")
                row = json.loads(
                    text,
                    object_pairs_hook=object_pairs,
                    parse_constant=reject_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, EvidenceError) as error:
                raise EvidenceError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            require(
                isinstance(row, dict),
                f"JSONL row at {path}:{line_number} must be an object",
            )
            validate_json_numbers(row, f"{path}:{line_number}")
            require(
                row.get("schema") == SCHEMA, f"schema mismatch at {path}:{line_number}"
            )
            require(
                row.get("schema_version") == SCHEMA_VERSION,
                f"schema version mismatch at {path}:{line_number}",
            )
            run_id = row.get("run_id")
            require(
                isinstance(run_id, str) and run_id,
                f"invalid run_id at {path}:{line_number}",
            )
            require(
                isinstance(row.get("event"), str),
                f"invalid event at {path}:{line_number}",
            )
            require(
                isinstance(row.get("payload"), dict),
                f"invalid payload at {path}:{line_number}",
            )
            event = row["event"]
            require(
                event in PAYLOAD_KEYS, f"unknown event at {path}:{line_number}: {event}"
            )
            require_keys(row, COMMON_KEYS, f"event at {path}:{line_number}")
            require_keys(
                row["payload"],
                PAYLOAD_KEYS[event],
                f"{event} payload at {path}:{line_number}",
            )
            if event == "proposal_block":
                bit_row = dict(row)
                bit_row["payload"] = dict(row["payload"])
                bit_row["payload"].pop("selector_depths")
                validate_bit_fields(bit_row, f"{path}:{line_number}")
            else:
                validate_bit_fields(row, f"{path}:{line_number}")
            row["__source"] = f"{path}:{line_number}"
            row["__input_path"] = str(path)
            rows.append(row)
    return rows, sorted(inputs, key=lambda item: item["path"])


def strict_softmax(scores: list[float], temperature: float) -> list[float]:
    require(
        scores and all(math.isfinite(score) for score in scores),
        "softmax scores must be finite and nonempty",
    )
    require(
        math.isfinite(temperature) and temperature > 0.0,
        "softmax temperature must be finite and positive",
    )
    maximum = max(scores)
    weights = [math.exp((score - maximum) / temperature) for score in scores]
    total = math.fsum(weights)
    require(math.isfinite(total) and total > 0.0, "softmax normalization failed")
    probabilities = [weight / total for weight in weights]
    require(
        all(math.isfinite(q) and q >= 0.0 for q in probabilities),
        "softmax produced an invalid q",
    )
    return probabilities


def first_max_index(values: list[float]) -> int:
    require(values, "first-max requires values")
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def validate_label(value: Any, name: str) -> str:
    require(isinstance(value, str) and value, f"{name} must be nonempty")
    require(
        len(value) <= 128
        and all(
            char.isascii() and (char.isalnum() or char in "-_./") for char in value
        ),
        f"{name} contains invalid label characters",
    )
    return value


def validate_config(config: Any, run_id: str) -> dict[str, Any]:
    require(isinstance(config, dict), f"run {run_id} config missing")
    require_keys(
        config,
        {
            "tokens",
            "stop_tokens",
            "temperature_f32_bits",
            "top_k",
            "top_p_f32_bits",
            "min_p_f32_bits",
            "seed",
            "sampler_algorithm_version",
            "no_warmup",
            "semantics",
            "prompt_prefill",
        },
        f"run {run_id} config",
    )
    integer(config["tokens"], f"run {run_id} token limit", 1)
    top_k = integer(config["top_k"], f"run {run_id} top_k", 1)
    require(top_k <= 200, f"run {run_id} top_k exceeds bounded trace scope")
    temperature = decode_bits(
        config["temperature_f32_bits"], 32, f"run {run_id} target temperature"
    )
    top_p = decode_bits(config["top_p_f32_bits"], 32, f"run {run_id} top_p")
    min_p = decode_bits(config["min_p_f32_bits"], 32, f"run {run_id} min_p")
    require(temperature > 0.0, f"run {run_id} target temperature must be positive")
    require(0.0 < top_p <= 1.0, f"run {run_id} top_p must be in (0, 1]")
    require(0.0 <= min_p <= 1.0, f"run {run_id} min_p must be in [0, 1]")
    seed = integer(config["seed"], f"run {run_id} sampler seed", 0)
    require(seed <= MASK64, f"run {run_id} sampler seed exceeds u64")
    require(
        config["sampler_algorithm_version"] == SAMPLER_ALGORITHM_VERSION,
        f"run {run_id} is not sampler-v{SAMPLER_ALGORITHM_VERSION}",
    )
    require(isinstance(config["no_warmup"], bool), f"run {run_id} no_warmup invalid")
    require(
        config["semantics"] == DEVELOPMENT_SEMANTICS,
        f"run {run_id} semantics mismatch",
    )
    require(
        config["prompt_prefill"] == PROMPT_PREFILL_SEMANTICS,
        f"run {run_id} prompt-prefill semantics mismatch",
    )
    stops = config["stop_tokens"]
    require(isinstance(stops, list), f"run {run_id} stop_tokens invalid")
    for index, token in enumerate(stops):
        token = integer(token, f"run {run_id} stop_tokens[{index}]", 0)
        require(token < 1 << 31, f"run {run_id} stop token exceeds i32")
    require(len(stops) == len(set(stops)), f"run {run_id} stop tokens are duplicated")
    return config


def validate_asset(asset: Any, label: str, *, gguf: bool) -> None:
    require(isinstance(asset, dict), f"{label} asset must be an object")
    if not gguf:
        require_keys(asset, {"path", "bytes", "sha256"}, label)
        require(
            isinstance(asset["path"], str) and asset["path"], f"{label} path invalid"
        )
        integer(asset["bytes"], f"{label} bytes", 1)
        validate_sha256(asset["sha256"], f"{label} sha256")
        return
    require_keys(
        asset,
        {"aggregate_sha256_index_size_digest_le", "shards"},
        label,
    )
    shards = asset["shards"]
    require(isinstance(shards, list) and shards, f"{label} shards invalid")
    aggregate = hashlib.sha256()
    paths: set[str] = set()
    for expected_index, shard in enumerate(shards):
        require(isinstance(shard, dict), f"{label} shard {expected_index} invalid")
        require_keys(shard, {"index", "path", "bytes", "sha256"}, f"{label} shard")
        require(
            integer(shard["index"], f"{label} shard index", 0) == expected_index,
            f"{label} shard indexes are not contiguous",
        )
        path = shard["path"]
        require(
            isinstance(path, str) and path and path not in paths,
            f"{label} shard path invalid",
        )
        paths.add(path)
        size = integer(shard["bytes"], f"{label} shard bytes", 1)
        digest = validate_sha256(shard["sha256"], f"{label} shard sha256")
        aggregate.update(struct.pack("<Q", expected_index))
        aggregate.update(struct.pack("<Q", size))
        aggregate.update(bytes.fromhex(digest))
    require(
        asset["aggregate_sha256_index_size_digest_le"] == aggregate.hexdigest(),
        f"{label} aggregate digest mismatch",
    )


def validate_build_identity(identity: Any, run_id: str) -> None:
    require(isinstance(identity, dict), f"run {run_id} build identity missing")
    require_keys(
        identity,
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
        f"run {run_id} build identity",
    )
    integer(identity["schema_version"], f"run {run_id} build schema", 1)
    commit = identity["build_commit"]
    require(
        isinstance(commit, str)
        and len(commit) == 40
        and all(char in "0123456789abcdef" for char in commit),
        f"run {run_id} build commit invalid",
    )
    require(
        identity["runtime_commit"] == commit
        and isinstance(identity["build_commit_short"], str)
        and commit.startswith(identity["build_commit_short"]),
        f"run {run_id} build/runtime commit identity mismatch",
    )
    require(
        isinstance(identity["build_dirty"], bool)
        and identity["runtime_dirty"] is identity["build_dirty"],
        f"run {run_id} build/runtime dirty identity mismatch",
    )
    source = identity["build_source_state"]
    require(
        isinstance(source, str)
        and source.startswith("git-source-sha256-v2:")
        and identity["runtime_source_state"] == source,
        f"run {run_id} build/runtime source identity mismatch",
    )
    validate_sha256(
        source.removeprefix("git-source-sha256-v2:"), f"run {run_id} source state"
    )
    require(
        isinstance(identity["stamp_source"], str)
        and identity["stamp_source"]
        and (
            identity["stamp_error"] is None or isinstance(identity["stamp_error"], str)
        ),
        f"run {run_id} build stamp invalid",
    )
    require(
        isinstance(identity["status"], str)
        and isinstance(identity["problems"], list)
        and isinstance(identity["overrides"], list)
        and all(isinstance(value, str) for value in identity["problems"])
        and all(isinstance(value, str) for value in identity["overrides"]),
        f"run {run_id} build status invalid",
    )


def validate_binding(binding: Any, config: dict[str, Any], run_id: str) -> int:
    require(isinstance(binding, dict), f"run {run_id} binding missing")
    require_keys(
        binding,
        {"target_architecture", "target", "drafter", "resolved_stop_tokens"},
        f"run {run_id} binding",
    )
    require(
        isinstance(binding["target_architecture"], str)
        and binding["target_architecture"],
        f"run {run_id} target architecture invalid",
    )
    target = binding["target"]
    require(isinstance(target, dict), f"run {run_id} target binding invalid")
    require_keys(
        target, {"n_layer", "hidden_size", "vocab_size"}, f"run {run_id} target binding"
    )
    target_layers = integer(target["n_layer"], f"run {run_id} target layers", 1)
    target_hidden = integer(target["hidden_size"], f"run {run_id} target hidden", 1)
    target_vocab = integer(target["vocab_size"], f"run {run_id} target vocab", 1)
    require(target_vocab <= 1 << 31, f"run {run_id} target vocab exceeds i32 scope")
    drafter = binding["drafter"]
    require(isinstance(drafter, dict), f"run {run_id} drafter binding invalid")
    require_keys(
        drafter,
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
        f"run {run_id} drafter binding",
    )
    integer(drafter["n_layer"], f"run {run_id} drafter layers", 1)
    require(
        integer(drafter["hidden_size"], f"run {run_id} drafter hidden", 1)
        == target_hidden,
        f"run {run_id} target/drafter hidden mismatch",
    )
    block_size = integer(drafter["block_size"], f"run {run_id} block size", 2)
    for key in (
        "swa_window",
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ):
        integer(drafter[key], f"run {run_id} drafter {key}", 0)
    require(
        drafter["selector_rank"] > 0 and drafter["selector_top_k"] == 16,
        f"run {run_id} binding is not the scoped DFlash2 selector",
    )
    layer_ids = drafter["target_layer_ids"]
    require(
        isinstance(layer_ids, list) and layer_ids, f"run {run_id} target layers invalid"
    )
    parsed_layers = [
        integer(layer, f"run {run_id} target layer id", 0) for layer in layer_ids
    ]
    require(
        len(parsed_layers) == len(set(parsed_layers))
        and all(layer < target_layers for layer in parsed_layers),
        f"run {run_id} target layer binding invalid",
    )
    require(
        binding["resolved_stop_tokens"] == config["stop_tokens"],
        f"run {run_id} resolved stop-token binding mismatch",
    )
    for index, token in enumerate(config["stop_tokens"]):
        token_id(token, f"run {run_id} stop_tokens[{index}]", target_vocab)
    return block_size - 1


def validate_common_metadata(start: dict[str, Any], run_id: str) -> None:
    config = validate_config(start.get("config"), run_id)
    classification = start.get("classification")
    require(isinstance(classification, dict), f"run {run_id} classification missing")
    require_keys(
        classification,
        {
            "evidence_role",
            "fixture_id",
            "fixture_role",
            "target_arm",
            "drafter_arm",
            "e0_status",
        },
        f"run {run_id} classification",
    )
    require(
        classification["evidence_role"] == "development"
        and classification["e0_status"] == "not_measured",
        f"run {run_id} attempts to claim non-development or measured-E0 authority",
    )
    for key in ("fixture_id", "fixture_role", "target_arm", "drafter_arm"):
        validate_label(classification[key], f"run {run_id} {key}")
    lease = start.get("lease_env")
    require(
        isinstance(lease, dict) and lease.get("QWEN_METAL_LEASE_WAIT") == "1",
        f"run {run_id} lacks the required literal Metal lease setting",
    )
    assets = start.get("assets")
    require(isinstance(assets, dict), f"run {run_id} assets missing")
    require_keys(assets, {"target", "drafter", "executable"}, f"run {run_id} assets")
    validate_asset(assets["target"], f"run {run_id} target", gguf=True)
    validate_asset(assets["drafter"], f"run {run_id} drafter", gguf=True)
    validate_asset(assets["executable"], f"run {run_id} executable", gguf=False)
    prompt = start.get("prompt")
    require(isinstance(prompt, dict), f"run {run_id} prompt identity missing")
    require_keys(
        prompt,
        {"utf8_len", "utf8_sha256", "token_count", "token_ids_sha256_i32le"},
        f"run {run_id} prompt identity",
    )
    integer(prompt["utf8_len"], f"run {run_id} prompt bytes", 1)
    integer(prompt["token_count"], f"run {run_id} prompt tokens", 1)
    validate_sha256(prompt["utf8_sha256"], f"run {run_id} prompt bytes sha256")
    validate_sha256(
        prompt["token_ids_sha256_i32le"], f"run {run_id} prompt token sha256"
    )
    paths = start.get("paths")
    require(isinstance(paths, dict), f"run {run_id} paths missing")
    require_keys(paths, {"model", "drafter", "output"}, f"run {run_id} paths")
    require(
        all(isinstance(paths[key], str) and paths[key] for key in paths),
        f"run {run_id} path identity invalid",
    )
    command = start.get("command")
    require(
        isinstance(command, list)
        and command
        and all(isinstance(arg, str) for arg in command),
        f"run {run_id} command identity invalid",
    )
    host = start.get("host")
    require(isinstance(host, dict), f"run {run_id} host identity missing")
    require_keys(host, {"os", "arch", "metal_device"}, f"run {run_id} host")
    require(
        all(isinstance(value, str) and value for value in host.values()),
        f"run {run_id} host identity invalid",
    )
    sessions = start.get("sessions")
    require(isinstance(sessions, dict), f"run {run_id} sessions missing")
    require_keys(
        sessions,
        {"oracle_target", "oracle_drafter", "serial_reference_target"},
        f"run {run_id} sessions",
    )
    session_ids = list(sessions.values())
    require(
        all(isinstance(value, str) and value for value in session_ids)
        and len(set(session_ids)) == len(session_ids),
        f"run {run_id} session identities are invalid or shared",
    )
    validate_binding(start.get("binding"), config, run_id)
    validate_build_identity(start.get("build_identity"), run_id)


def check_common(rows: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    common_keys = (
        "schema",
        "schema_version",
        "run_id",
        "build_identity",
        "lease_env",
        "classification",
        "config",
        "assets",
        "binding",
        "command",
        "host",
        "paths",
        "prompt",
        "sessions",
    )
    start_rows = [row for row in rows if row["event"] == "run_start"]
    serial_rows = [row for row in rows if row["event"] == "serial_reference_end"]
    end_rows = [row for row in rows if row["event"] == "run_end"]
    require(len(start_rows) == 1, f"run {run_id} requires exactly one run_start")
    require(
        len(serial_rows) == 1, f"run {run_id} requires exactly one serial_reference_end"
    )
    require(len(end_rows) == 1, f"run {run_id} requires exactly one run_end")
    require(
        rows[0]["event"] == "run_start", f"run {run_id} does not start with run_start"
    )
    require(rows[-1]["event"] == "run_end", f"run {run_id} does not end with run_end")
    require(
        rows.index(serial_rows[0]) < rows.index(end_rows[0]),
        f"run {run_id} terminal events are out of order",
    )
    start = start_rows[0]
    for row in rows:
        require(
            all(row.get(key) == start.get(key) for key in common_keys),
            f"run {run_id} common event metadata mismatch at {row['__source']}",
        )
    validate_common_metadata(start, run_id)
    return start


def check_event_state_machine(rows: list[dict[str, Any]], run_id: str) -> None:
    phase = "oracle"
    open_block: int | None = None
    pending_draw: int | None = None
    oracle_sample_count = 0
    saw_reference_sample = False
    for index, row in enumerate(rows):
        event = row["event"]
        if index == 0:
            require(event == "run_start", f"run {run_id} must begin with run_start")
            continue
        if event == "sample_decision":
            require(
                phase == "oracle",
                f"run {run_id} oracle sample appears after reference phase",
            )
            sample_index = integer(
                row["payload"].get("sample_index"), f"run {run_id} sample index", 0
            )
            require(
                sample_index == oracle_sample_count,
                f"run {run_id} oracle sample indexes are not event-order contiguous",
            )
            oracle_sample_count += 1
            if sample_index == 0:
                require(
                    open_block is None
                    and pending_draw is None
                    and row["payload"].get("frontier") == "prompt_prefill",
                    f"run {run_id} initial oracle sample is misplaced",
                )
            else:
                require(
                    pending_draw is None
                    and row["payload"].get("frontier") == "target_transition",
                    f"run {run_id} sample {sample_index} is not awaiting exactly one decision",
                )
                pending_draw = sample_index
        elif event == "one_hot_decision":
            require(
                phase == "oracle" and pending_draw is not None,
                f"run {run_id} one-hot decision is out of order",
            )
            require(
                row["payload"].get("draw_index") == pending_draw,
                f"run {run_id} one-hot decision does not immediately consume its sample",
            )
            pending_draw = None
            block_index = integer(
                row["payload"]["block_index"], f"run {run_id} decision block", 0
            )
            if open_block is None:
                open_block = block_index
            require(
                open_block == block_index, f"run {run_id} interleaves proposal blocks"
            )
        elif event == "proposal_block":
            require(
                phase == "oracle" and open_block is not None and pending_draw is None,
                f"run {run_id} proposal block precedes decisions",
            )
            require(
                row["payload"]["block_index"] == open_block,
                f"run {run_id} closes the wrong proposal block",
            )
            open_block = None
        elif event == "reference_sample_decision":
            require(
                open_block is None
                and pending_draw is None
                and oracle_sample_count > 0
                and phase in {"oracle", "reference"},
                f"run {run_id} reference sample is out of order",
            )
            phase = "reference"
            saw_reference_sample = True
        elif event == "serial_reference_end":
            require(
                phase == "reference" and saw_reference_sample,
                f"run {run_id} serial terminal precedes reference samples",
            )
            phase = "terminal"
        elif event == "run_end":
            require(
                phase == "terminal" and index == len(rows) - 1,
                f"run {run_id} run_end is out of order",
            )
            phase = "done"
        else:
            require(event == "run_start", f"run {run_id} unexpected event ordering")
    require(
        phase == "done" and open_block is None and pending_draw is None,
        f"run {run_id} event stream is incomplete",
    )


def check_samples(
    rows: list[dict[str, Any]], run_id: str, event_name: str
) -> list[dict[str, Any]]:
    samples = [row["payload"] for row in rows if row["event"] == event_name]
    require(samples, f"run {run_id} requires {event_name} events")
    config = validate_config(rows[0].get("config"), run_id)
    temperature = decode_bits(
        config.get("temperature_f32_bits"), 32, f"run {run_id} target temperature"
    )
    seed = integer(config.get("seed"), f"run {run_id} sampler seed", 0)
    require(seed <= MASK64, f"run {run_id} sampler seed exceeds u64")
    prompt_tokens = integer(
        rows[0]["prompt"].get("token_count"), f"run {run_id} prompt token count", 1
    )
    target_vocab = integer(
        rows[0]["binding"]["target"]["vocab_size"],
        f"run {run_id} target vocab",
        1,
    )
    trusted_state = xoshiro256pp_initial_state(seed)
    for expected, sample in enumerate(samples):
        label = f"run {run_id} {event_name} {expected}"
        require(
            integer(sample.get("sample_index"), f"{label} sample_index", 0) == expected,
            f"{label} sample indexes are not contiguous",
        )
        require(
            sample.get("frontier")
            == ("prompt_prefill" if expected == 0 else "target_transition"),
            f"{label} frontier mismatch",
        )
        require(
            integer(sample.get("target_position"), f"{label} target_position", 0)
            == prompt_tokens - 1 + expected,
            f"{label} target position mismatch",
        )
        validate_sha256(sample.get("logits_sha256_f32le"), f"{label} logits digest")
        distribution = sample.get("distribution")
        require(isinstance(distribution, dict), f"{label} distribution missing")
        require_keys(
            distribution,
            {
                "selected_token",
                "candidate_index",
                "ordered_support",
                "total_weight_f64_bits",
            },
            f"{label} distribution",
        )
        support = distribution["ordered_support"]
        require(
            isinstance(support, list) and 1 <= len(support) <= config["top_k"],
            f"{label} support length is outside sampler-v1 top-k bounds",
        )
        tokens: list[int] = []
        weights: list[float] = []
        for candidate_index, candidate in enumerate(support):
            require(isinstance(candidate, dict), f"{label} candidate invalid")
            require_keys(candidate, {"token", "weight_f64_bits"}, f"{label} candidate")
            token = token_id(candidate["token"], f"{label} support token", target_vocab)
            tokens.append(token)
            weights.append(
                decode_bits(candidate["weight_f64_bits"], 64, f"{label} support weight")
            )
        require(
            len(tokens) == len(set(tokens)), f"{label} support has duplicate tokens"
        )
        require(
            all(weight >= 0.0 for weight in weights), f"{label} has negative weight"
        )
        require(weights[0] == 1.0, f"{label} leading sampler-v1 weight is not 1.0")
        require(
            all(later <= earlier for earlier, later in zip(weights, weights[1:])),
            f"{label} sampler-v1 weights are not non-increasing",
        )
        left_total = 0.0
        for weight in weights:
            left_total += weight
        require(
            distribution["total_weight_f64_bits"] == encode_f64(left_total)
            and math.isfinite(left_total)
            and left_total > 0.0,
            f"{label} support total bits mismatch",
        )
        selected = token_id(
            distribution["selected_token"], f"{label} selected token", target_vocab
        )
        selected_index = integer(
            distribution["candidate_index"], f"{label} candidate index", 0
        )
        require(
            selected_index < len(tokens) and tokens[selected_index] == selected,
            f"{label} selected candidate mismatch",
        )
        rng = sample.get("rng")
        if temperature > 0.0:
            require(isinstance(rng, dict), f"{label} requires RNG evidence")
            require_keys(
                rng,
                {
                    "draws_before",
                    "draws_after",
                    "state_before",
                    "state_after",
                    "raw_u64",
                    "raw_uniform_f64_bits",
                },
                f"{label} RNG",
            )
            require(
                integer(rng["draws_before"], f"{label} draws_before", 0) == expected,
                f"{label} draws_before mismatch",
            )
            require(
                integer(rng["draws_after"], f"{label} draws_after", 0) == expected + 1,
                f"{label} draws_after mismatch",
            )
            expected_before = state_hex(trusted_state)
            require(
                rng["state_before"] == expected_before,
                f"{label} state_before disagrees with seeded xoshiro256++",
            )
            raw_u64, trusted_state = xoshiro256pp_next(trusted_state)
            require(
                rng["raw_u64"] == f"0x{raw_u64:016x}",
                f"{label} raw_u64 disagrees with xoshiro256++",
            )
            require(
                rng["state_after"] == state_hex(trusted_state),
                f"{label} state_after disagrees with xoshiro256++",
            )
            unit = (raw_u64 >> 11) / float(1 << 53)
            require(
                rng["raw_uniform_f64_bits"] == encode_f64(unit),
                f"{label} uniform conversion mismatch",
            )
            replay_index = categorical_replay_index(weights, unit)
            require(
                replay_index == selected_index and tokens[replay_index] == selected,
                f"{label} categorical replay selection mismatch",
            )
        else:
            require(rng is None, f"{label} zero-temperature sample has RNG evidence")
        require(
            integer(sample["live_draws_after"], f"{label} live draws", 0)
            == (expected + 1 if temperature > 0.0 else 0),
            f"{label} live draw count mismatch",
        )
    return samples


def token_ids_sha256_i32le(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        require(-(1 << 31) <= token < (1 << 31), "generated token does not fit i32")
        digest.update(struct.pack("<i", token))
    return digest.hexdigest()


def pending_token_sha256(token: int | None) -> str:
    digest = hashlib.sha256()
    if token is None:
        digest.update(b"\x00")
    else:
        integer(token, "pending token")
        require(-(1 << 31) <= token < (1 << 31), "pending token does not fit i32")
        digest.update(b"\x01")
        digest.update(struct.pack("<i", token))
    return digest.hexdigest()


def validate_stop_evidence(
    payload: dict[str, Any],
    config: dict[str, Any],
    emitted_count: int,
    vocab_size: int,
    label: str,
) -> None:
    terminal_token = token_id(
        payload.get("terminal_token"), f"{label} terminal_token", vocab_size
    )
    stops = config.get("stop_tokens")
    require(isinstance(stops, list), f"{label} stop_tokens invalid")
    limit = integer(config.get("tokens"), f"{label} token limit", 1)
    eos_hit = terminal_token in stops
    token_limit_hit = emitted_count >= limit
    require(payload.get("eos_hit") is eos_hit, f"{label} eos_hit mismatch")
    require(
        payload.get("token_limit_hit") is token_limit_hit,
        f"{label} token_limit_hit mismatch",
    )
    require(
        payload.get("stop_reason") == ("eos" if eos_hit else "token_limit"),
        f"{label} stop_reason mismatch",
    )
    require(eos_hit or token_limit_hit, f"{label} has no valid stop condition")


def validate_pending_state(
    payload: dict[str, Any],
    terminal_token: int,
    target_vocab: int,
    run_id: str,
    arm: str,
) -> None:
    state = payload.get("target_state")
    require(isinstance(state, dict), f"run {run_id} {arm} target_state invalid")
    require(
        state.get("pending_token") == terminal_token,
        f"run {run_id} {arm} pending token mismatch",
    )
    token_id(
        state.get("pending_token"),
        f"run {run_id} {arm} pending token",
        target_vocab,
    )
    require(
        state.get("pending_token_sha256_tagged_i32le")
        == pending_token_sha256(terminal_token),
        f"run {run_id} {arm} pending token digest mismatch",
    )
    require(
        state.get("final_logits_present") is False,
        f"run {run_id} {arm} snapshot retains final logits",
    )
    require(
        state.get("capture_tail_present") is False,
        f"run {run_id} {arm} snapshot retains capture tail",
    )


def compare_sample_arms(
    oracle: list[dict[str, Any]], reference: list[dict[str, Any]], run_id: str
) -> None:
    require(
        len(oracle) == len(reference),
        f"run {run_id} oracle/reference sample counts differ",
    )
    fields = ("logits_sha256_f32le", "distribution")
    for index, (oracle_sample, reference_sample) in enumerate(
        zip(oracle, reference, strict=True)
    ):
        require(
            all(oracle_sample[field] == reference_sample[field] for field in fields),
            f"run {run_id} oracle/reference sample {index} distribution or logits mismatch",
        )
        require(
            oracle_sample["rng"] == reference_sample["rng"],
            f"run {run_id} oracle/reference sample {index} RNG evidence mismatch",
        )


def validate_reconstructed_stream(
    samples: list[dict[str, Any]],
    terminal: dict[str, Any],
    target_vocab: int,
    run_id: str,
    arm: str,
) -> list[int]:
    stream = [
        token_id(
            sample["distribution"]["selected_token"],
            f"run {run_id} {arm} token",
            target_vocab,
        )
        for sample in samples
    ]
    terminal_stream = terminal.get("generated_ids")
    require(
        isinstance(terminal_stream, list), f"run {run_id} {arm} terminal stream invalid"
    )
    for index, token in enumerate(terminal_stream):
        token_id(
            token,
            f"run {run_id} {arm} terminal token {index}",
            target_vocab,
        )
    require(
        stream == terminal_stream, f"run {run_id} {arm} reconstructed stream mismatch"
    )
    digest = token_ids_sha256_i32le(stream)
    require(
        terminal.get("generated_ids_sha256_i32le") == digest,
        f"run {run_id} {arm} generated stream digest mismatch",
    )
    return stream


def validate_transition_evidence(
    samples: list[dict[str, Any]], terminal: dict[str, Any], run_id: str, arm: str
) -> None:
    reconstructed = [sample["logits_sha256_f32le"] for sample in samples]
    terminal_hashes = terminal.get("transition_logits_sha256_f32le")
    require(
        isinstance(terminal_hashes, list),
        f"run {run_id} {arm} terminal transition hashes invalid",
    )
    for index, digest in enumerate(terminal_hashes):
        validate_sha256(digest, f"run {run_id} {arm} terminal transition {index}")
    require(
        terminal_hashes == reconstructed,
        f"run {run_id} {arm} terminal transition hashes do not match sample events",
    )


def selector_result(
    selector: dict[str, Any],
    expected_depth: int,
    proposal: int,
    predecessor: int,
    predecessor_choice: int | None,
    temperature: float,
    target_vocab: int,
    label: str,
) -> dict[str, Any]:
    require_keys(
        selector,
        {
            "depth",
            "predecessor_token",
            "predecessor_choice_index",
            "top_k_ids",
            "unary_logits_f32_bits",
            "final_scores_f32_bits",
            "greedy_score_f32_bits",
            "greedy_index",
            "greedy_token",
            "issues",
        },
        label,
    )
    require(
        integer(selector.get("depth"), f"{label} selector depth", 0) == expected_depth,
        f"{label} selector depths are not contiguous",
    )
    require(
        selector.get("predecessor_token") == predecessor,
        f"{label} predecessor token mismatch",
    )
    require(
        selector.get("predecessor_choice_index") == predecessor_choice,
        f"{label} predecessor choice mismatch",
    )
    issues = selector.get("issues")
    require(
        isinstance(issues, list) and not issues,
        f"{label} selector reports structural issues",
    )
    ids = selector.get("top_k_ids")
    score_bits = selector.get("final_scores_f32_bits")
    unary = selector.get("unary_logits_f32_bits")
    require(
        isinstance(ids, list) and len(ids) == 16, f"{label} selector must have 16 IDs"
    )
    require(
        isinstance(score_bits, list) and len(score_bits) == 16,
        f"{label} selector must have 16 final scores",
    )
    require(
        isinstance(unary, list) and len(unary) == 16,
        f"{label} selector must have 16 unary logits",
    )
    valid_ids = [token_id(token, f"{label} selector ID", target_vocab) for token in ids]
    require(len(set(valid_ids)) == 16, f"{label} selector IDs must be unique")
    scores = [decode_bits(bits, 32, f"{label} final score") for bits in score_bits]
    for bits in unary:
        decode_bits(bits, 32, f"{label} unary logit")
    expected_index = first_max_index(scores)
    greedy_index = integer(selector.get("greedy_index"), f"{label} greedy index", 0)
    greedy_token = token_id(
        selector.get("greedy_token"), f"{label} greedy token", target_vocab
    )
    require(greedy_index == expected_index, f"{label} greedy index is not first-max")
    require(
        greedy_token == valid_ids[greedy_index] == proposal,
        f"{label} greedy token mismatch",
    )
    greedy_score = decode_bits(
        selector.get("greedy_score_f32_bits"), 32, f"{label} greedy score"
    )
    require(
        struct.pack(">f", greedy_score) == struct.pack(">f", scores[greedy_index]),
        f"{label} greedy score bits mismatch",
    )
    q = strict_softmax(scores, temperature)
    q_sum = math.fsum(q)
    return {
        "depth": expected_depth,
        "greedy_index": greedy_index,
        "greedy_token": greedy_token,
        "chosen_token_q": q[greedy_index],
        "entropy_nats": -math.fsum(
            probability * math.log(probability)
            for probability in q
            if probability > 0.0
        ),
        "normalized_sparse_q": [
            {"q": probability, "token": token}
            for token, probability in zip(valid_ids, q, strict=True)
        ],
        "q_sum": q_sum,
        "q_sum_error": abs(q_sum - 1.0),
    }


def validate_selector_token_bounds(
    selector: dict[str, Any], target_vocab: int, label: str
) -> None:
    for key in ("predecessor_token", "greedy_token"):
        if key in selector:
            token_id(selector[key], f"{label} {key}", target_vocab)
    ids = selector.get("top_k_ids")
    if isinstance(ids, list):
        for index, token in enumerate(ids):
            token_id(token, f"{label} top_k_ids[{index}]", target_vocab)


def positions_sha256(positions: list[int]) -> str:
    digest = hashlib.sha256()
    for position in positions:
        require(0 <= position <= MASK64, "KV position does not fit u64")
        digest.update(struct.pack("<Q", position))
    return digest.hexdigest()


def snapshot_identity_sha256(identity: dict[str, Any]) -> str:
    storage_kinds = {"None": 0, "F16": 1, "Q8_0": 2}
    storage = identity["kv_storage_kind"]
    require(
        storage in storage_kinds, f"unsupported snapshot KV storage kind {storage!r}"
    )
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", identity["model_id"]))
    digest.update(struct.pack("<Q", identity["tokenizer_id"]))
    for key in (
        "layout_version",
        "n_attn_layers",
        "n_gdn_layers",
        "kv_dim_elements",
        "kv_bytes_per_token",
    ):
        digest.update(struct.pack("<I", identity[key]))
    digest.update(struct.pack("<I", storage_kinds[storage]))
    for key in ("gdn_state_elements_per_layer", "gdn_conv_elements_per_layer"):
        digest.update(struct.pack("<I", identity[key]))
    return digest.hexdigest()


def validate_snapshot_state(
    state: Any, target_vocab: int, run_id: str, arm: str
) -> dict[str, Any]:
    require(isinstance(state, dict), f"run {run_id} {arm} target_state invalid")
    require_keys(
        state,
        {
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
        },
        f"run {run_id} {arm} target_state",
    )
    identity = state["identity"]
    require(isinstance(identity, dict), f"run {run_id} {arm} snapshot identity invalid")
    require_keys(
        identity,
        {
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
        },
        f"run {run_id} {arm} snapshot identity",
    )
    for key, value in identity.items():
        if key == "kv_storage_kind":
            require(
                isinstance(value, str) and value, f"run {run_id} {arm} {key} invalid"
            )
        else:
            parsed = integer(value, f"run {run_id} {arm} identity {key}", 0)
            maximum = MASK64 if key in {"model_id", "tokenizer_id"} else (1 << 32) - 1
            require(parsed <= maximum, f"run {run_id} {arm} identity {key} overflows")
    require(
        state["identity_sha256_canonical_le"] == snapshot_identity_sha256(identity),
        f"run {run_id} {arm} snapshot identity digest mismatch",
    )
    prefix_len = integer(state["prefix_len"], f"run {run_id} {arm} prefix_len", 1)
    validate_sha256(
        state["prefix_token_ids_sha256_i32le"],
        f"run {run_id} {arm} prefix digest",
    )
    pending = token_id(
        state["pending_token"], f"run {run_id} {arm} pending token", target_vocab
    )
    require(
        state["pending_token_sha256_tagged_i32le"] == pending_token_sha256(pending),
        f"run {run_id} {arm} pending token digest mismatch",
    )
    positions = state["kv_positions"]
    require(isinstance(positions, list), f"run {run_id} {arm} KV positions invalid")
    parsed_positions = [
        integer(position, f"run {run_id} {arm} KV position", 0)
        for position in positions
    ]
    require(
        len(parsed_positions) == identity["n_attn_layers"]
        and all(position == prefix_len for position in parsed_positions),
        f"run {run_id} {arm} KV-position geometry mismatch",
    )
    require(
        state["kv_positions_sha256_u64le"] == positions_sha256(parsed_positions),
        f"run {run_id} {arm} KV-position digest mismatch",
    )
    sections = state["sections"]
    require(isinstance(sections, dict), f"run {run_id} {arm} state sections invalid")
    require_keys(
        sections,
        {"kv_k", "kv_v", "gdn_conv", "gdn_state"},
        f"run {run_id} {arm} state sections",
    )
    for key, section in sections.items():
        require(isinstance(section, dict), f"run {run_id} {arm} {key} section invalid")
        require_keys(section, {"bytes", "sha256"}, f"run {run_id} {arm} {key}")
        integer(section["bytes"], f"run {run_id} {arm} {key} bytes", 0)
        validate_sha256(section["sha256"], f"run {run_id} {arm} {key} sha256")
    expected_section_bytes = {
        "kv_k": identity["n_attn_layers"] * prefix_len * identity["kv_bytes_per_token"],
        "kv_v": identity["n_attn_layers"] * prefix_len * identity["kv_bytes_per_token"],
        "gdn_conv": identity["n_gdn_layers"]
        * identity["gdn_conv_elements_per_layer"]
        * 4,
        "gdn_state": identity["n_gdn_layers"]
        * identity["gdn_state_elements_per_layer"]
        * 4,
    }
    require(
        all(
            sections[key]["bytes"] == expected
            for key, expected in expected_section_bytes.items()
        ),
        f"run {run_id} {arm} snapshot section geometry mismatch",
    )
    require(
        state["final_logits_present"] is False
        and state["capture_tail_present"] is False,
        f"run {run_id} {arm} snapshot retains excluded transient data",
    )
    return state


def derived_state_comparisons(
    oracle: dict[str, Any], reference: dict[str, Any]
) -> dict[str, bool]:
    result = {
        "identity": oracle["identity"] == reference["identity"]
        and oracle["identity_sha256_canonical_le"]
        == reference["identity_sha256_canonical_le"],
        "prefix": oracle["prefix_len"] == reference["prefix_len"]
        and oracle["prefix_token_ids_sha256_i32le"]
        == reference["prefix_token_ids_sha256_i32le"],
        "pending_token": oracle["pending_token"] == reference["pending_token"]
        and oracle["pending_token_sha256_tagged_i32le"]
        == reference["pending_token_sha256_tagged_i32le"],
        "kv_positions": oracle["kv_positions"] == reference["kv_positions"]
        and oracle["kv_positions_sha256_u64le"]
        == reference["kv_positions_sha256_u64le"],
        "kv_k_bytes": oracle["sections"]["kv_k"] == reference["sections"]["kv_k"],
        "kv_v_bytes": oracle["sections"]["kv_v"] == reference["sections"]["kv_v"],
        "gdn_conv_bytes": oracle["sections"]["gdn_conv"]
        == reference["sections"]["gdn_conv"],
        "gdn_state_bytes": oracle["sections"]["gdn_state"]
        == reference["sections"]["gdn_state"],
    }
    result["all"] = all(result.values())
    return result


def exact_parity(
    serial: dict[str, Any], end: dict[str, Any], run_id: str, start: dict[str, Any]
) -> dict[str, bool]:
    target_vocab = integer(
        start["binding"]["target"]["vocab_size"], f"run {run_id} target vocab", 1
    )
    target_layers = integer(
        start["binding"]["target"]["n_layer"], f"run {run_id} target layers", 1
    )
    oracle_state = validate_snapshot_state(
        end.get("target_state"), target_vocab, run_id, "oracle"
    )
    reference_state = validate_snapshot_state(
        serial.get("target_state"), target_vocab, run_id, "reference"
    )
    target_digest = bytes.fromhex(
        start["assets"]["target"]["aggregate_sha256_index_size_digest_le"]
    )
    expected_model_id = struct.unpack("<Q", target_digest[:8])[0]
    expected_tokenizer_id = struct.unpack("<Q", target_digest[8:16])[0]
    for state_name, state_payload in (
        ("oracle", oracle_state),
        ("reference", reference_state),
    ):
        require(
            state_payload["identity"]["n_attn_layers"]
            + state_payload["identity"]["n_gdn_layers"]
            == target_layers,
            f"run {run_id} {state_name} snapshot layer geometry disagrees with target",
        )
        require(
            state_payload["identity"]["model_id"] == expected_model_id
            and state_payload["identity"]["tokenizer_id"] == expected_tokenizer_id,
            f"run {run_id} {state_name} snapshot identity is not asset-bound",
        )
    state = derived_state_comparisons(oracle_state, reference_state)
    generated_ids = serial.get("generated_ids") == end.get("generated_ids")
    sampler_draw_count = serial.get("sampler_draws") == end.get("sampler_draws")
    aligned_logits = serial.get("transition_logits_sha256_f32le") == end.get(
        "transition_logits_sha256_f32le"
    )
    kv_positions = serial.get("target_kv_positions") == end.get("target_kv_positions")
    require(
        end.get("target_kv_positions") == oracle_state["kv_positions"]
        and serial.get("target_kv_positions") == reference_state["kv_positions"],
        f"run {run_id} top-level and snapshot KV positions disagree",
    )
    boundary = (
        generated_ids
        and sampler_draw_count
        and aligned_logits
        and kv_positions
        and state["all"]
        and serial.get("continuation_position") == end.get("continuation_position")
    )
    serial_continuation = validate_sha256(
        serial.get("continuation_logits_sha256_f32le"),
        f"run {run_id} reference continuation digest",
    )
    oracle_continuation = validate_sha256(
        end.get("continuation_logits_sha256_f32le"),
        f"run {run_id} oracle continuation digest",
    )
    derived = {
        "generated_ids": generated_ids,
        "sampler_draw_count": sampler_draw_count,
        "aligned_transition_logits": aligned_logits,
        "target_kv_positions": kv_positions,
        "target_state": state["all"],
        "continuation_boundary_equal": boundary,
        "one_token_continuation_compared": True,
        "one_token_continuation_logits": serial_continuation == oracle_continuation,
    }
    comparisons = serial.get("comparisons")
    require(
        isinstance(comparisons, dict) and set(comparisons) == set(derived),
        f"run {run_id} producer serial comparison fields mismatch",
    )
    require(
        comparisons == derived,
        f"run {run_id} producer serial comparisons disagree with reducer derivation",
    )
    for payload, name in ((serial, "serial"), (end, "run_end")):
        producer_state = payload.get("target_state_comparisons")
        require(
            producer_state == state,
            f"run {run_id} {name} state comparisons disagree with reducer derivation",
        )
    require(
        end.get("continuation_boundary_equal") is boundary
        and end.get("one_token_continuation_compared") is True
        and end.get("one_token_continuation_equal")
        is derived["one_token_continuation_logits"],
        f"run {run_id} producer continuation claims disagree with reducer derivation",
    )
    result = dict(derived)
    result.update(
        {f"target_state_{key}": value for key, value in state.items() if key != "all"}
    )
    require(all(result.values()), f"run {run_id} failed independently derived parity")
    return result


def cluster_metadata(start: dict[str, Any]) -> dict[str, Any]:
    classification = start["classification"]
    config_without_seed = {
        key: value for key, value in start["config"].items() if key != "seed"
    }
    config_id = canonical_json_sha256(config_without_seed)
    request_material = {
        "fixture_id": classification["fixture_id"],
        "fixture_role": classification["fixture_role"],
        "target_arm": classification["target_arm"],
        "drafter_arm": classification["drafter_arm"],
        "target_asset": start["assets"]["target"][
            "aggregate_sha256_index_size_digest_le"
        ],
        "drafter_asset": start["assets"]["drafter"][
            "aggregate_sha256_index_size_digest_le"
        ],
        "prompt_utf8_sha256": start["prompt"]["utf8_sha256"],
        "prompt_tokens_sha256": start["prompt"]["token_ids_sha256_i32le"],
        "config_id": config_id,
    }
    request_id = canonical_json_sha256(request_material)
    seed = start["config"]["seed"]
    return {
        **{key: classification[key] for key in classification},
        "config_id": config_id,
        "request_id": request_id,
        "seed": seed,
        "target_asset_sha256": request_material["target_asset"],
        "drafter_asset_sha256": request_material["drafter_asset"],
        "cluster_id": canonical_json_sha256({"request_id": request_id, "seed": seed}),
    }


def reduce_run(
    rows: list[dict[str, Any]], proposal_temperature: float
) -> dict[str, Any]:
    run_id = rows[0]["run_id"]
    start = check_common(rows, run_id)
    config = validate_config(start.get("config"), run_id)
    target_vocab = integer(
        start["binding"]["target"]["vocab_size"], f"run {run_id} target vocab", 1
    )
    check_event_state_machine(rows, run_id)
    samples = check_samples(rows, run_id, "sample_decision")
    reference_samples = check_samples(rows, run_id, "reference_sample_decision")
    compare_sample_arms(samples, reference_samples, run_id)
    serial = next(
        row["payload"] for row in rows if row["event"] == "serial_reference_end"
    )
    end = next(row["payload"] for row in rows if row["event"] == "run_end")
    require(end.get("status") == "ok", f"run {run_id} did not finish ok")
    oracle_stream = validate_reconstructed_stream(
        samples, end, target_vocab, run_id, "oracle"
    )
    reference_stream = validate_reconstructed_stream(
        reference_samples, serial, target_vocab, run_id, "reference"
    )
    require(
        oracle_stream == reference_stream, f"run {run_id} reconstructed streams differ"
    )
    validate_transition_evidence(samples, end, run_id, "oracle")
    validate_transition_evidence(reference_samples, serial, run_id, "reference")
    validate_stop_evidence(
        end,
        config,
        len(oracle_stream),
        target_vocab,
        f"run {run_id} oracle terminal",
    )
    validate_stop_evidence(
        serial,
        config,
        len(reference_stream),
        target_vocab,
        f"run {run_id} reference terminal",
    )
    require(
        end["terminal_token"] == oracle_stream[-1]
        and serial["terminal_token"] == reference_stream[-1],
        f"run {run_id} terminal token disagrees with reconstructed stream",
    )
    validate_pending_state(end, end["terminal_token"], target_vocab, run_id, "oracle")
    validate_pending_state(
        serial, serial["terminal_token"], target_vocab, run_id, "reference"
    )
    prompt_tokens = integer(
        start["prompt"].get("token_count"), f"run {run_id} prompt token count", 1
    )
    expected_consumed = prompt_tokens + len(oracle_stream) - 1
    require(
        end["target_state"].get("prefix_len") == expected_consumed
        and serial["target_state"].get("prefix_len") == expected_consumed,
        f"run {run_id} snapshot consumed-prefix length mismatch",
    )
    processed_position = integer(
        end.get("processed_target_position"), f"run {run_id} processed position", 0
    )
    require(
        end.get("consumed_prefix_len") == expected_consumed
        and end.get("dflash_target_ctx_n") == expected_consumed
        and processed_position + 1 == expected_consumed,
        f"run {run_id} oracle consumed state boundary mismatch",
    )
    require(
        end.get("continuation_position") == expected_consumed
        and serial.get("continuation_position") == expected_consumed,
        f"run {run_id} continuation position mismatch",
    )
    parity = exact_parity(serial, end, run_id, start)
    target_temperature = decode_bits(
        config.get("temperature_f32_bits"), 32, f"run {run_id} target temperature"
    )
    require(
        target_temperature > 0.0, f"run {run_id} target temperature must be positive"
    )
    temperature = float(proposal_temperature)
    require(
        math.isfinite(temperature) and temperature > 0.0,
        "proposal temperature must be explicitly positive and finite",
    )
    samples_by_draw = {
        integer(sample["rng"]["draws_before"], f"run {run_id} sample draw", 0): sample
        for sample in samples
    }
    require(
        len(samples_by_draw) == len(samples),
        f"run {run_id} sample draw indexes are not unique",
    )

    decisions_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    decision_order: list[dict[str, Any]] = []
    for row in rows:
        if row["event"] != "one_hot_decision":
            continue
        decision = row["payload"]
        block = integer(decision.get("block_index"), f"run {run_id} decision block", 0)
        decisions_by_block[block].append(decision)
        decision_order.append(decision)
    blocks = [row["payload"] for row in rows if row["event"] == "proposal_block"]
    require(
        integer(end.get("blocks"), f"run {run_id} terminal blocks", 0) == len(blocks),
        f"run {run_id} block count mismatch",
    )
    require(
        [
            integer(block.get("block_index"), f"run {run_id} block index", 0)
            for block in blocks
        ]
        == list(range(len(blocks))),
        f"run {run_id} block indexes are not contiguous",
    )

    depth_attempts: defaultdict[int, int] = defaultdict(int)
    depth_accepts: defaultdict[int, int] = defaultdict(int)
    depth_probability_sum: defaultdict[int, float] = defaultdict(float)
    depth_support_misses: defaultdict[int, int] = defaultdict(int)
    k0_rows: list[dict[str, Any]] = []
    k0_findings: list[dict[str, Any]] = []
    draw_indexes: list[int] = []
    required_depth_count = validate_binding(start["binding"], config, run_id)
    for block_index, block in enumerate(blocks):
        proposals_raw = block.get("proposals")
        selectors = block.get("selector_depths")
        require(
            isinstance(proposals_raw, list) and proposals_raw,
            f"run {run_id} block {block_index} proposals invalid",
        )
        proposals = [
            token_id(value, f"run {run_id} proposal", target_vocab)
            for value in proposals_raw
        ]
        require(
            len(proposals) == required_depth_count,
            f"run {run_id} block {block_index} proposal count disagrees with binding",
        )
        selector_count = len(selectors) if isinstance(selectors, list) else 0
        if not isinstance(selectors, list):
            k0_findings.append(
                {"block_index": block_index, "error": "selector_depths is not an array"}
            )
        block_decisions = decisions_by_block.pop(block_index, [])
        require(block_decisions, f"run {run_id} block {block_index} has no decisions")
        require(
            [
                integer(item.get("depth"), f"run {run_id} decision depth", 0)
                for item in block_decisions
            ]
            == list(range(len(block_decisions))),
            f"run {run_id} block {block_index} decision depths are not contiguous",
        )
        require(
            len(block_decisions) <= required_depth_count,
            f"run {run_id} block {block_index} has too many decisions",
        )
        accepted_count = 0
        mismatch_depth: int | None = None
        first_draw_index: int | None = None
        for depth, decision in enumerate(block_decisions):
            require(
                depth < len(proposals) and decision.get("proposal") == proposals[depth],
                f"run {run_id} block {block_index} decision proposal mismatch",
            )
            proposal = proposals[depth]
            sampled = token_id(
                decision.get("sampled_target"),
                f"run {run_id} sampled target",
                target_vocab,
            )
            accepted = decision.get("accepted")
            require(
                isinstance(accepted, bool) and accepted == (proposal == sampled),
                f"run {run_id} block {block_index} one-hot decision inconsistent",
            )
            require(
                isinstance(decision.get("terminal"), bool),
                f"run {run_id} block {block_index} terminal flag invalid",
            )
            if accepted:
                require(mismatch_depth is None, f"run {run_id} accepts after mismatch")
                accepted_count += 1
            else:
                require(
                    depth == len(block_decisions) - 1,
                    f"run {run_id} continues after mismatch",
                )
                mismatch_depth = depth
            probability = decode_bits(
                decision.get("proposal_probability_f64_bits"),
                64,
                f"run {run_id} proposal probability",
            )
            proposal_weight = decode_bits(
                decision.get("proposal_weight_f64_bits"),
                64,
                f"run {run_id} proposal weight",
            )
            total_weight = decode_bits(
                decision.get("total_weight_f64_bits"),
                64,
                f"run {run_id} proposal total weight",
            )
            present = decision.get("proposal_present_in_support")
            require(
                isinstance(present, bool), f"run {run_id} support-presence flag invalid"
            )
            require(
                total_weight > 0.0 and proposal_weight >= 0.0,
                f"run {run_id} proposal weights invalid",
            )
            require(
                0.0 <= probability <= 1.0
                and (present or proposal_weight == probability == 0.0),
                f"run {run_id} proposal probability inconsistent",
            )
            require(
                probability == proposal_weight / total_weight,
                f"run {run_id} proposal probability normalization mismatch",
            )
            draw_index = integer(
                decision.get("draw_index"), f"run {run_id} draw index", 0
            )
            if first_draw_index is None:
                first_draw_index = draw_index
                require(
                    draw_index > 0
                    and samples[draw_index - 1]["distribution"]["selected_token"]
                    == block.get("carry_in"),
                    f"run {run_id} block {block_index} carry does not precede its first draw",
                )
            require(
                draw_index == first_draw_index + depth,
                f"run {run_id} block {block_index} decision draws are not contiguous",
            )
            require(
                decision.get("carry_in") == block.get("carry_in"),
                f"run {run_id} block {block_index} decision carry mismatch",
            )
            require(
                draw_index in samples_by_draw,
                f"run {run_id} one-hot decision references an unknown sample draw",
            )
            draw_sample = samples_by_draw[draw_index]
            require(
                draw_sample.get("frontier") == "target_transition",
                f"run {run_id} one-hot decision does not reference a target transition",
            )
            distribution = draw_sample["distribution"]
            require(
                distribution.get("selected_token") == sampled,
                f"run {run_id} one-hot sampled target disagrees with sample draw",
            )
            support_candidate = next(
                (
                    candidate
                    for candidate in distribution["ordered_support"]
                    if candidate["token"] == proposal
                ),
                None,
            )
            require(
                present == (support_candidate is not None),
                f"run {run_id} one-hot support-presence evidence disagrees with sample draw",
            )
            expected_weight_bits = (
                support_candidate["weight_f64_bits"]
                if support_candidate is not None
                else encode_f64(0.0)
            )
            require(
                decision.get("proposal_weight_f64_bits") == expected_weight_bits
                and decision.get("total_weight_f64_bits")
                == distribution.get("total_weight_f64_bits"),
                f"run {run_id} one-hot weight evidence disagrees with sample draw",
            )
            draw_indexes.append(draw_index)
            stop_tokens = config.get("stop_tokens")
            require(isinstance(stop_tokens, list), f"run {run_id} stop_tokens invalid")
            token_limit = integer(config.get("tokens"), f"run {run_id} token limit", 1)
            expected_terminal = draw_index + 1 >= token_limit or sampled in stop_tokens
            require(
                decision["terminal"] == expected_terminal,
                f"run {run_id} block {block_index} terminal decision inconsistent",
            )
            if decision["terminal"]:
                validate_stop_evidence(
                    decision,
                    config,
                    draw_index + 1,
                    target_vocab,
                    f"run {run_id} terminal decision",
                )
                require(
                    decision["terminal_token"] == sampled,
                    f"run {run_id} terminal decision token mismatch",
                )
                require(
                    draw_index == len(samples) - 1,
                    f"run {run_id} has samples after a terminal decision",
                )
            else:
                require(
                    decision["stop_reason"] is None
                    and decision["terminal_token"] is None
                    and decision["eos_hit"] is False
                    and decision["token_limit_hit"] is False,
                    f"run {run_id} nonterminal decision carries stop evidence",
                )
            depth_attempts[depth] += 1
            depth_accepts[depth] += int(accepted)
            depth_probability_sum[depth] += probability
            depth_support_misses[depth] += int(not present)
        require(
            integer(block.get("accepted"), f"run {run_id} block accepted", 0)
            == accepted_count,
            f"run {run_id} block accepted count mismatch",
        )
        require(
            block.get("mismatch_depth") == mismatch_depth,
            f"run {run_id} block mismatch depth inconsistent",
        )
        carry = token_id(
            block.get("carry_in"), f"run {run_id} block carry", target_vocab
        )
        predecessor = carry
        predecessor_choice: int | None = None
        selector_list = selectors if isinstance(selectors, list) else []
        if len(selector_list) != len(proposals):
            k0_findings.append(
                {
                    "block_index": block_index,
                    "error": f"selector/proposal length mismatch: {len(selector_list)} != {len(proposals)}",
                }
            )
        for depth, proposal in enumerate(proposals):
            if depth >= len(selector_list):
                k0_findings.append(
                    {
                        "block_index": block_index,
                        "depth": depth + 1,
                        "error": "missing selector depth",
                    }
                )
                predecessor = proposal
                predecessor_choice = None
                continue
            selector = selector_list[depth]
            selector_label = f"run {run_id} block {block_index} depth {depth + 1}"
            if isinstance(selector, dict):
                validate_selector_token_bounds(selector, target_vocab, selector_label)
            try:
                require(isinstance(selector, dict), "selector is not an object")
                result = selector_result(
                    selector,
                    depth + 1,
                    proposal,
                    predecessor,
                    predecessor_choice,
                    temperature,
                    target_vocab,
                    selector_label,
                )
            except EvidenceError as error:
                finding: dict[str, Any] = {
                    "block_index": block_index,
                    "depth": depth + 1,
                    "error": str(error),
                }
                if isinstance(selector, dict) and "issues" in selector:
                    finding["observed_issues"] = selector["issues"]
                k0_findings.append(finding)
                predecessor = proposal
                predecessor_choice = None
                continue
            result["block_index"] = block_index
            result["proposal_depth"] = depth
            k0_rows.append(result)
            predecessor = proposal
            predecessor_choice = result["greedy_index"]
        for depth in range(len(proposals), len(selector_list)):
            k0_findings.append(
                {
                    "block_index": block_index,
                    "depth": depth + 1,
                    "error": "extra selector depth",
                }
            )
    require(
        not decisions_by_block, f"run {run_id} has decisions without proposal blocks"
    )
    require(
        draw_indexes == list(range(1, len(samples))),
        f"run {run_id} decisions must consume every post-prefill sample exactly once",
    )

    terminal_attempts = end.get("attempts_by_depth")
    terminal_accepts = end.get("accepts_by_depth")
    require(
        isinstance(terminal_attempts, list) and isinstance(terminal_accepts, list),
        f"run {run_id} terminal depth counts invalid",
    )
    for index, value in enumerate(terminal_attempts):
        integer(value, f"run {run_id} attempts_by_depth[{index}]", 0)
    for index, value in enumerate(terminal_accepts):
        integer(value, f"run {run_id} accepts_by_depth[{index}]", 0)
    depth_count = required_depth_count
    require(
        len(terminal_attempts) == depth_count and len(terminal_accepts) == depth_count,
        f"run {run_id} terminal depth vectors are truncated or extended",
    )
    attempts = [depth_attempts[index] for index in range(depth_count)]
    accepts = [depth_accepts[index] for index in range(depth_count)]
    require(terminal_attempts == attempts, f"run {run_id} attempts_by_depth mismatch")
    require(terminal_accepts == accepts, f"run {run_id} accepts_by_depth mismatch")
    generated_ids = end.get("generated_ids")
    require(isinstance(generated_ids, list), f"run {run_id} emitted stream invalid")
    for index, value in enumerate(generated_ids):
        token_id(value, f"run {run_id} generated_ids[{index}]", target_vocab)
    emitted = len(generated_ids)
    require(emitted > 0, f"run {run_id} emitted stream invalid")
    configured_limit = integer(config.get("tokens"), f"run {run_id} token limit", 1)
    configured_stops = config.get("stop_tokens")
    require(isinstance(configured_stops, list), f"run {run_id} stop_tokens invalid")
    require(emitted <= configured_limit, f"run {run_id} emitted beyond token limit")
    require(
        emitted == configured_limit or generated_ids[-1] in configured_stops,
        f"run {run_id} stream ended before limit without a stop token",
    )
    sampler_draws = integer(end.get("sampler_draws"), f"run {run_id} sampler draws", 0)
    require(
        sampler_draws == emitted == len(samples),
        f"run {run_id} sampler draw/emitted/sample count mismatch",
    )

    depths = []
    for depth in range(depth_count):
        count = attempts[depth]
        depths.append(
            {
                "acceptance_empirical": accepts[depth] / count if count else None,
                "accepts": accepts[depth],
                "attempts": count,
                "depth": depth,
                "mean_proposal_probability": depth_probability_sum[depth] / count
                if count
                else None,
                "support_misses": depth_support_misses[depth],
            }
        )
    k0_by_depth = []
    for depth in sorted({row["depth"] for row in k0_rows}):
        selected = [row for row in k0_rows if row["depth"] == depth]
        k0_by_depth.append(
            {
                "depth": depth,
                "mean_chosen_token_q": math.fsum(
                    row["chosen_token_q"] for row in selected
                )
                / len(selected),
                "mean_entropy_nats": math.fsum(row["entropy_nats"] for row in selected)
                / len(selected),
                "max_q_sum_error": max(row["q_sum_error"] for row in selected),
                "selector_rows": len(selected),
            }
        )
    return {
        "blocks": len(blocks),
        "depths": depths,
        "cluster": cluster_metadata(start),
        "draw_index_exact_coverage": True,
        "e0": {
            "status": "not_measured",
            "required_before_any_product_authority": True,
        },
        "e1a": {
            "oracle_reference_step_parity": True,
            "reconstructed_stream_parity": True,
            "rng_replay_from_seed": True,
            "status": "development_parity_passed_e0_unmeasured",
            "gate_status": "blocked_by_e0",
        },
        "emitted": emitted,
        "exact_parity": parity,
        "k0": {
            "by_depth": k0_by_depth,
            "findings": k0_findings,
            "status": "failed"
            if k0_findings
            else ("local_reference_only" if k0_rows else "not_observed_no_proposals"),
            "authority": "local_reference_only",
            "reference_only_no_cross_implementation_trace_parity_claim": True,
            "selector_rows": k0_rows,
            "temperature": temperature,
            "temperature_mode": "explicit_local_reference_only",
        },
        "mean_emitted_per_block": emitted / len(blocks) if blocks else None,
        "run_id": run_id,
        "sampler_draws": sampler_draws,
    }


def projection(
    anchors: dict[str, float | None], mean_emitted: float, target_speedup: float
) -> dict[str, Any] | None:
    supplied = [value is not None for value in anchors.values()]
    require(
        all(supplied) or not any(supplied),
        "economic anchors must be supplied all together",
    )
    if not any(supplied):
        return None
    serial = finite_number(anchors["serial_ms"], "--serial-ms", positive=True)
    draft = finite_number(anchors["draft_ms"], "--draft-ms")
    verify = finite_number(anchors["verify_ms"], "--verify-ms")
    other = finite_number(anchors["other_ms"], "--other-ms")
    require(
        draft >= 0.0 and verify >= 0.0 and other >= 0.0,
        "--draft-ms, --verify-ms, and --other-ms must be nonnegative",
    )
    packet = draft + verify + other
    require(packet > 0.0, "economic packet cost must be positive")
    return {
        "authoritative": False,
        "label": "non_authoritative_optimistic_projection",
        "inputs_ms": {
            "draft_ms": draft,
            "other_ms": other,
            "serial_ms": serial,
            "verify_ms": verify,
        },
        "sampler_cost_inference": {
            "inferred_sampler_cost_ms": other,
            "named_input_source": "other_ms",
        },
        "optimistic_charged_packet_cost_ms": packet,
        "projected_speedup": mean_emitted * serial / packet,
        "break_even_emitted": packet / serial,
        "exact_verifier_budget_ms_at_target_speedup": mean_emitted
        * serial
        / target_speedup
        - draft
        - other,
        "target_speedup": target_speedup,
        "equations": {
            "packet_cost": "draft_ms + verify_ms + other_ms",
            "projected_speedup": "empirical_mean_emitted_per_block * serial_ms / packet_cost",
            "exact_verifier_budget": "empirical_mean_emitted_per_block * serial_ms / target_speedup - draft_ms - other_ms",
        },
    }


def reduce(
    paths: list[Path],
    proposal_temperature: float,
    target_speedup: float,
    anchors: dict[str, float | None],
) -> dict[str, Any]:
    rows, inputs = read_inputs(paths)
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["run_id"]].append(row)
    for run_id, run_rows in grouped.items():
        require(
            len({row["__input_path"] for row in run_rows}) == 1,
            f"run {run_id} spans multiple input files",
        )
    runs = [
        reduce_run(grouped[run_id], proposal_temperature) for run_id in sorted(grouped)
    ]
    require(runs, "no runs found")
    total_blocks = sum(run["blocks"] for run in runs)
    total_emitted = sum(run["emitted"] for run in runs)
    mean_emitted = total_emitted / total_blocks if total_blocks else None
    all_depths = sorted({item["depth"] for run in runs for item in run["depths"]})
    aggregate_depths = []
    for depth in all_depths:
        entries = [
            item for run in runs for item in run["depths"] if item["depth"] == depth
        ]
        attempts = sum(item["attempts"] for item in entries)
        accepts = sum(item["accepts"] for item in entries)
        probability_total = math.fsum(
            item["mean_proposal_probability"] * item["attempts"]
            for item in entries
            if item["attempts"]
        )
        aggregate_depths.append(
            {
                "acceptance_empirical": accepts / attempts if attempts else None,
                "accepts": accepts,
                "attempts": attempts,
                "depth": depth,
                "mean_proposal_probability": probability_total / attempts
                if attempts
                else None,
                "support_misses": sum(item["support_misses"] for item in entries),
            }
        )
    k0_entries = [item for run in runs for item in run["k0"]["by_depth"]]
    k0_aggregate = []
    for depth in sorted({item["depth"] for item in k0_entries}):
        selected = [item for item in k0_entries if item["depth"] == depth]
        count = sum(item["selector_rows"] for item in selected)
        k0_aggregate.append(
            {
                "depth": depth,
                "mean_chosen_token_q": math.fsum(
                    item["mean_chosen_token_q"] * item["selector_rows"]
                    for item in selected
                )
                / count,
                "mean_entropy_nats": math.fsum(
                    item["mean_entropy_nats"] * item["selector_rows"]
                    for item in selected
                )
                / count,
                "max_q_sum_error": max(item["max_q_sum_error"] for item in selected),
                "selector_rows": count,
            }
        )
    clustered: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        clustered[run["cluster"]["cluster_id"]].append(run)
    if any(value is not None for value in anchors.values()):
        require(
            len(clustered) == 1,
            "global timing anchors are allowed only for a single request/seed cluster",
        )
    cluster_summaries = []
    for cluster_id in sorted(clustered):
        selected = clustered[cluster_id]
        blocks = sum(run["blocks"] for run in selected)
        emitted = sum(run["emitted"] for run in selected)
        metadata = selected[0]["cluster"]
        require(
            all(run["cluster"] == metadata for run in selected),
            f"cluster {cluster_id} metadata mismatch",
        )
        cluster_summaries.append(
            {
                "cluster": metadata,
                "runs": [run["run_id"] for run in selected],
                "run_count": len(selected),
                "blocks": blocks,
                "emitted": emitted,
                "mean_emitted_per_block": emitted / blocks if blocks else None,
                "projection": projection(anchors, emitted / blocks, target_speedup)
                if blocks
                else None,
            }
        )
    arm_groups: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for cluster in cluster_summaries:
        key = (
            cluster["cluster"]["target_arm"],
            cluster["cluster"]["drafter_arm"],
            cluster["cluster"]["target_asset_sha256"],
            cluster["cluster"]["drafter_asset_sha256"],
        )
        arm_groups[key].append(cluster)
    arm_summaries = []
    for (
        target_arm,
        drafter_arm,
        target_asset_sha256,
        drafter_asset_sha256,
    ), selected in sorted(arm_groups.items()):
        blocks = sum(cluster["blocks"] for cluster in selected)
        emitted = sum(cluster["emitted"] for cluster in selected)
        arm_summaries.append(
            {
                "target_arm": target_arm,
                "drafter_arm": drafter_arm,
                "target_asset_sha256": target_asset_sha256,
                "drafter_asset_sha256": drafter_asset_sha256,
                "clusters": [cluster["cluster"]["cluster_id"] for cluster in selected],
                "cluster_count": len(selected),
                "blocks": blocks,
                "emitted": emitted,
                "descriptive_mean_emitted_per_block": emitted / blocks
                if blocks
                else None,
                "inferential_status": "not_computed_development_only",
            }
        )
    result = {
        "aggregate": {
            "aggregation_semantics": "descriptive_pooled_counts_only_not_request_cluster_inference",
            "blocks": total_blocks,
            "depths": aggregate_depths,
            "e0_status": "not_measured",
            "e1a_development_parity_all_runs": True,
            "gate_status": "blocked_by_e0",
            "emitted": total_emitted,
            "exact_parity_all_runs": all(
                all(run["exact_parity"].values()) for run in runs
            ),
            "mean_emitted_per_block": mean_emitted,
            "runs": len(runs),
            "sampler_draws": sum(run["sampler_draws"] for run in runs),
        },
        "arms": arm_summaries,
        "clusters": cluster_summaries,
        "inputs": inputs,
        "k0": {
            "aggregate_by_depth": k0_aggregate,
            "authority": "local_reference_only",
            "findings": [
                {"run_id": run["run_id"], **finding}
                for run in runs
                for finding in run["k0"]["findings"]
            ],
            "label": "local_reference_q_diagnostic_only",
            "no_cross_implementation_trace_parity_claim": True,
            "pinned_upstream_contracts": UPSTREAM_CONTRACTS,
            "status": "failed"
            if any(run["k0"]["status"] == "failed" for run in runs)
            else (
                "local_reference_only"
                if any(run["k0"]["status"] == "local_reference_only" for run in runs)
                else "not_observed_no_proposals"
            ),
        },
        "projection": {
            "authority": False,
            "aggregation": "per_request_seed_cluster_only",
            "available": any(
                cluster["projection"] is not None for cluster in cluster_summaries
            ),
            "no_pooled_projection": True,
        },
        "runs": runs,
        "schema": "qwen.dflash_sampled_evidence_reduction",
        "schema_version": 2,
        "script": {"path": str(SCRIPT), "sha256": sha256_bytes(SCRIPT.read_bytes())},
        "status": STATUS,
    }
    validate_json_numbers(result, "reduction")
    return result


def synthetic_rows(run_id: str = "self-test") -> list[dict[str, Any]]:
    ids = list(range(10, 26))
    scores = [1.0, 1.0] + [0.5 - index / 100.0 for index in range(14)]
    state_comparisons = {
        "all": True,
        "identity": True,
        "prefix": True,
        "pending_token": True,
        "kv_positions": True,
        "kv_k_bytes": True,
        "kv_v_bytes": True,
        "gdn_conv_bytes": True,
        "gdn_state_bytes": True,
    }
    empty_sha = sha256_bytes(b"")
    kv_prefix_bytes = b"\x00" * 8

    def synthetic_gguf_asset(name: str) -> dict[str, Any]:
        content_digest = sha256_bytes(name.encode("ascii"))
        aggregate = hashlib.sha256()
        aggregate.update(struct.pack("<Q", 0))
        aggregate.update(struct.pack("<Q", len(name)))
        aggregate.update(bytes.fromhex(content_digest))
        return {
            "aggregate_sha256_index_size_digest_le": aggregate.hexdigest(),
            "shards": [
                {
                    "index": 0,
                    "path": f"/{name}.gguf",
                    "bytes": len(name),
                    "sha256": content_digest,
                }
            ],
        }

    target_asset = synthetic_gguf_asset("target")
    drafter_asset = synthetic_gguf_asset("drafter")
    target_digest = bytes.fromhex(target_asset["aggregate_sha256_index_size_digest_le"])
    snapshot_identity = {
        "model_id": struct.unpack("<Q", target_digest[:8])[0],
        "tokenizer_id": struct.unpack("<Q", target_digest[8:16])[0],
        "layout_version": 1,
        "n_attn_layers": 1,
        "n_gdn_layers": 0,
        "kv_dim_elements": 1,
        "kv_bytes_per_token": 4,
        "kv_storage_kind": "F16",
        "gdn_state_elements_per_layer": 0,
        "gdn_conv_elements_per_layer": 0,
    }
    source_state = sha256_bytes(b"synthetic-source")
    common = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "build_identity": {
            "schema_version": 2,
            "build_commit": "1" * 40,
            "build_commit_short": "1" * 9,
            "build_dirty": True,
            "build_source_state": f"git-source-sha256-v2:{source_state}",
            "stamp_source": "git",
            "stamp_error": None,
            "runtime_commit": "1" * 40,
            "runtime_dirty": True,
            "runtime_source_state": f"git-source-sha256-v2:{source_state}",
            "status": "dirty",
            "problems": ["dirty"],
            "overrides": ["allow_dirty"],
        },
        "lease_env": {"QWEN_METAL_LEASE_WAIT": "1"},
        "classification": {
            "evidence_role": "development",
            "fixture_id": "self-test",
            "fixture_role": "development-sentinel",
            "target_arm": "synthetic-target",
            "drafter_arm": "synthetic-drafter",
            "e0_status": "not_measured",
        },
        "config": {
            "temperature_f32_bits": encode_f32(1.0),
            "top_k": 20,
            "top_p_f32_bits": encode_f32(1.0),
            "min_p_f32_bits": encode_f32(0.0),
            "seed": 0,
            "tokens": 2,
            "stop_tokens": [],
            "sampler_algorithm_version": SAMPLER_ALGORITHM_VERSION,
            "no_warmup": True,
            "semantics": DEVELOPMENT_SEMANTICS,
            "prompt_prefill": PROMPT_PREFILL_SEMANTICS,
        },
        "assets": {
            "target": target_asset,
            "drafter": drafter_asset,
            "executable": {
                "path": "/qwen-bench",
                "bytes": 1,
                "sha256": sha256_bytes(b"binary"),
            },
        },
        "binding": {
            "target_architecture": "qwen35",
            "target": {"n_layer": 1, "hidden_size": 8, "vocab_size": 32},
            "drafter": {
                "n_layer": 1,
                "hidden_size": 8,
                "block_size": 2,
                "swa_window": 0,
                "conv_kernel_size": 2,
                "conv_group_size": 1,
                "selector_rank": 2,
                "selector_top_k": 16,
                "target_layer_ids": [0],
            },
            "resolved_stop_tokens": [],
        },
        "command": ["qwen-bench", "dflash-sampled-oracle"],
        "host": {"os": "test", "arch": "test", "metal_device": "test"},
        "paths": {
            "model": "/target.gguf",
            "drafter": "/drafter.gguf",
            "output": "/trace.jsonl",
        },
        "prompt": {
            "utf8_len": 1,
            "utf8_sha256": sha256_bytes(b"p"),
            "token_count": 1,
            "token_ids_sha256_i32le": token_ids_sha256_i32le([1]),
        },
        "sessions": {
            "oracle_target": f"{run_id}/oracle",
            "oracle_drafter": f"{run_id}/drafter",
            "serial_reference_target": f"{run_id}/reference",
        },
    }

    def event(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {**common, "event": kind, "payload": payload}

    selector = {
        "depth": 1,
        "predecessor_token": 7,
        "predecessor_choice_index": None,
        "top_k_ids": ids,
        "unary_logits_f32_bits": [encode_f32(value) for value in scores],
        "final_scores_f32_bits": [encode_f32(value) for value in scores],
        "greedy_score_f32_bits": encode_f32(1.0),
        "greedy_index": 0,
        "greedy_token": 10,
        "issues": [],
    }
    rng_state0 = xoshiro256pp_initial_state(common["config"]["seed"])
    raw0, rng_state1 = xoshiro256pp_next(rng_state0)
    raw1, rng_state2 = xoshiro256pp_next(rng_state1)
    sample0 = {
        "sample_index": 0,
        "frontier": "prompt_prefill",
        "target_position": 0,
        "logits_sha256_f32le": "0" * 64,
        "distribution": {
            "selected_token": 7,
            "candidate_index": 0,
            "ordered_support": [{"token": 7, "weight_f64_bits": encode_f64(1.0)}],
            "total_weight_f64_bits": encode_f64(1.0),
        },
        "rng": {
            "draws_before": 0,
            "draws_after": 1,
            "state_before": state_hex(rng_state0),
            "state_after": state_hex(rng_state1),
            "raw_u64": f"0x{raw0:016x}",
            "raw_uniform_f64_bits": encode_f64((raw0 >> 11) / float(1 << 53)),
        },
        "live_draws_after": 1,
    }
    sample1 = {
        "sample_index": 1,
        "frontier": "target_transition",
        "target_position": 1,
        "logits_sha256_f32le": "4" * 64,
        "distribution": {
            "selected_token": 10,
            "candidate_index": 0,
            "ordered_support": [{"token": 10, "weight_f64_bits": encode_f64(1.0)}],
            "total_weight_f64_bits": encode_f64(1.0),
        },
        "rng": {
            "draws_before": 1,
            "draws_after": 2,
            "state_before": state_hex(rng_state1),
            "state_after": state_hex(rng_state2),
            "raw_u64": f"0x{raw1:016x}",
            "raw_uniform_f64_bits": encode_f64((raw1 >> 11) / float(1 << 53)),
        },
        "live_draws_after": 2,
    }
    terminal_common = {
        "generated_ids": [7, 10],
        "generated_ids_sha256_i32le": token_ids_sha256_i32le([7, 10]),
        "stop_reason": "token_limit",
        "terminal_token": 10,
        "eos_hit": False,
        "token_limit_hit": True,
        "sampler_draws": 2,
        "transition_logits_sha256_f32le": ["0" * 64, "4" * 64],
        "target_kv_positions": [2],
        "target_state": {
            "identity": snapshot_identity,
            "identity_sha256_canonical_le": snapshot_identity_sha256(snapshot_identity),
            "prefix_len": 2,
            "prefix_token_ids_sha256_i32le": token_ids_sha256_i32le([1, 7]),
            "pending_token": 10,
            "pending_token_sha256_tagged_i32le": pending_token_sha256(10),
            "kv_positions": [2],
            "kv_positions_sha256_u64le": positions_sha256([2]),
            "sections": {
                "kv_k": {
                    "bytes": len(kv_prefix_bytes),
                    "sha256": sha256_bytes(kv_prefix_bytes),
                },
                "kv_v": {
                    "bytes": len(kv_prefix_bytes),
                    "sha256": sha256_bytes(kv_prefix_bytes),
                },
                "gdn_conv": {"bytes": 0, "sha256": empty_sha},
                "gdn_state": {"bytes": 0, "sha256": empty_sha},
            },
            "final_logits_present": False,
            "capture_tail_present": False,
        },
        "target_state_comparisons": state_comparisons,
        "continuation_position": 2,
        "continuation_logits_sha256_f32le": "3" * 64,
    }
    serial = {
        **terminal_common,
        "comparisons": {
            "generated_ids": True,
            "sampler_draw_count": True,
            "aligned_transition_logits": True,
            "target_kv_positions": True,
            "target_state": True,
            "continuation_boundary_equal": True,
            "one_token_continuation_compared": True,
            "one_token_continuation_logits": True,
        },
    }
    end = {
        **terminal_common,
        "status": "ok",
        "blocks": 1,
        "attempts_by_depth": [1],
        "accepts_by_depth": [1],
        "dflash_target_ctx_n": 2,
        "processed_target_position": 1,
        "consumed_prefix_len": 2,
        "continuation_boundary_equal": True,
        "one_token_continuation_compared": True,
        "one_token_continuation_equal": True,
        "elapsed_seconds_f64_bits": encode_f64(1.0),
        "timing_semantics": "diagnostic_only_exact_serial_oracle_non_performance",
    }
    return [
        event("run_start", {"started_utc": "1970-01-01T00:00:00Z"}),
        event("sample_decision", sample0),
        event("sample_decision", sample1),
        event(
            "one_hot_decision",
            {
                "block_index": 0,
                "depth": 0,
                "carry_in": 7,
                "proposal": 10,
                "sampled_target": 10,
                "accepted": True,
                "terminal": True,
                "stop_reason": "token_limit",
                "terminal_token": 10,
                "eos_hit": False,
                "token_limit_hit": True,
                "proposal_present_in_support": True,
                "proposal_weight_f64_bits": encode_f64(1.0),
                "total_weight_f64_bits": encode_f64(1.0),
                "proposal_probability_f64_bits": encode_f64(1.0),
                "draw_index": 1,
            },
        ),
        event(
            "proposal_block",
            {
                "block_index": 0,
                "noise_start_pos": 1,
                "carry_in": 7,
                "proposals": [10],
                "accepted": 1,
                "mismatch_depth": None,
                "selector_depths": [selector],
            },
        ),
        event(
            "reference_sample_decision",
            json.loads(json.dumps(sample0)),
        ),
        event(
            "reference_sample_decision",
            json.loads(json.dumps(sample1)),
        ),
        event("serial_reference_end", serial),
        event("run_end", end),
    ]


def synthetic_zero_block_rows(run_id: str = "zero-block") -> list[dict[str, Any]]:
    rows = json.loads(json.dumps(synthetic_rows(run_id)))
    rows = [
        row
        for row in rows
        if row["event"] in {"run_start", "serial_reference_end", "run_end"}
        or (
            row["event"] in {"sample_decision", "reference_sample_decision"}
            and row["payload"]["sample_index"] == 0
        )
    ]
    for row in rows:
        row["config"]["tokens"] = 1
        if row["event"] not in {"serial_reference_end", "run_end"}:
            continue
        payload = row["payload"]
        payload["generated_ids"] = [7]
        payload["generated_ids_sha256_i32le"] = token_ids_sha256_i32le([7])
        payload["terminal_token"] = 7
        payload["sampler_draws"] = 1
        payload["transition_logits_sha256_f32le"] = ["0" * 64]
        payload["continuation_position"] = 1
        payload["target_kv_positions"] = [1]
        state = payload["target_state"]
        state["prefix_len"] = 1
        state["prefix_token_ids_sha256_i32le"] = token_ids_sha256_i32le([1])
        state["pending_token"] = 7
        state["pending_token_sha256_tagged_i32le"] = pending_token_sha256(7)
        state["kv_positions"] = [1]
        state["kv_positions_sha256_u64le"] = positions_sha256([1])
        kv_prefix_bytes = b"\x00" * 4
        for section_name in ("kv_k", "kv_v"):
            state["sections"][section_name] = {
                "bytes": len(kv_prefix_bytes),
                "sha256": sha256_bytes(kv_prefix_bytes),
            }
        if row["event"] == "run_end":
            payload["blocks"] = 0
            payload["attempts_by_depth"] = [0]
            payload["accepts_by_depth"] = [0]
            payload["dflash_target_ctx_n"] = 1
            payload["processed_target_position"] = 0
            payload["consumed_prefix_len"] = 1
    return rows


def run_self_test() -> None:
    require(decode_bits("0x3f800000", 32, "test") == 1.0, "f32 decoding test failed")
    require(
        xoshiro256pp_initial_state(0)
        == [
            0xE220A8397B1DCDAF,
            0x6E789E6AA1B965F4,
            0x06C45D188009454F,
            0xF88BB8A8724C81EC,
        ],
        "SplitMix64 initialization golden vector failed",
    )
    golden_state = xoshiro256pp_initial_state(0x123456789ABCDEF0)
    golden_raw = []
    for _ in range(4):
        raw, golden_state = xoshiro256pp_next(golden_state)
        golden_raw.append(raw)
    require(
        golden_raw
        == [
            0x4D4F7607A97A1BD6,
            0x9BA027C76910D021,
            0x87ADB062153AE0BC,
            0xB750F7B1FF944783,
        ],
        "xoshiro256++ transition golden vector failed",
    )
    require(first_max_index([2.0, 2.0, 1.0]) == 0, "first-max tie test failed")
    q = strict_softmax([0.0, 0.0], 1.0)
    require(q == [0.5, 0.5] and math.fsum(q) == 1.0, "softmax test failed")
    require(
        categorical_replay_index(
            [0.0, float.fromhex("0x0.0000000000001p-1022")], math.nextafter(1.0, 0.0)
        )
        == 1,
        "categorical final-candidate fallback test failed",
    )
    with tempfile.TemporaryDirectory(prefix="dflash-evidence-self-test-") as directory:
        root = Path(directory)
        trace = root / "trace.jsonl"
        trace.write_text(
            "".join(
                json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                for row in synthetic_rows()
            ),
            encoding="utf-8",
        )
        result = reduce(
            [trace],
            1.0,
            1.10,
            {"serial_ms": None, "draft_ms": None, "verify_ms": None, "other_ms": None},
        )
        require(
            result["aggregate"]["emitted"] == 2
            and result["aggregate"]["depths"][0]["accepts"] == 1
            and result["aggregate"]["e0_status"] == "not_measured"
            and result["aggregate"]["gate_status"] == "blocked_by_e0"
            and len(result["clusters"]) == 1,
            "one-hot reduction test failed",
        )
        require(
            result["k0"]["status"] == "local_reference_only",
            "valid local K0 diagnostic status test failed",
        )
        zero_trace = root / "zero-block.jsonl"
        zero_trace.write_text(
            "".join(
                json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                for row in synthetic_zero_block_rows()
            ),
            encoding="utf-8",
        )
        zero_result = reduce(
            [zero_trace],
            1.0,
            1.10,
            {"serial_ms": None, "draft_ms": None, "verify_ms": None, "other_ms": None},
        )
        require(
            zero_result["aggregate"]["blocks"] == 0
            and zero_result["aggregate"]["mean_emitted_per_block"] is None
            and zero_result["clusters"][0]["projection"] is None,
            "valid zero-proposal terminal trace was not reduced",
        )
        malformed = root / "malformed.jsonl"
        malformed.write_text('{"schema":"wrong"}\n', encoding="utf-8")
        try:
            read_inputs([malformed])
            raise AssertionError(
                "strict event validation did not reject malformed input"
            )
        except EvidenceError:
            pass
        empty_anchors = {
            "serial_ms": None,
            "draft_ms": None,
            "verify_ms": None,
            "other_ms": None,
        }

        def expect_rows_rejected(name: str, rows: list[dict[str, Any]]) -> None:
            path = root / f"{name}.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            try:
                reduce([path], 1.0, 1.10, empty_anchors)
                raise AssertionError(f"{name} was not rejected")
            except EvidenceError:
                pass

        issue_rows = synthetic_rows("issues")
        next(row for row in issue_rows if row["event"] == "proposal_block")["payload"][
            "selector_depths"
        ][0]["issues"] = [{"kind": "sentinel"}]
        issue_path = root / "selector_issue.jsonl"
        issue_path.write_text(
            "".join(json.dumps(row) + "\n" for row in issue_rows), encoding="utf-8"
        )
        issue_result = reduce([issue_path], 1.0, 1.10, empty_anchors)
        require(
            issue_result["aggregate"]["exact_parity_all_runs"]
            and issue_result["k0"]["status"] == "failed"
            and issue_result["k0"]["findings"],
            "E1a development parity with K0 structural failure test failed",
        )

        sampler_version_rows = synthetic_rows("sampler-version")
        for row in sampler_version_rows:
            row["config"]["sampler_algorithm_version"] = 2
        expect_rows_rejected("sampler_version", sampler_version_rows)

        asset_rows = synthetic_rows("asset-digest")
        for row in asset_rows:
            row["assets"]["target"]["aggregate_sha256_index_size_digest_le"] = "0" * 64
        expect_rows_rejected("asset_digest", asset_rows)

        tampered_seed_rows = synthetic_rows("tampered-seed")
        for row in tampered_seed_rows:
            row["config"]["seed"] = 1
        expect_rows_rejected("tampered_seed", tampered_seed_rows)

        malformed_state_rows = synthetic_rows("malformed-state")
        malformed_state_rows[1]["payload"]["rng"]["state_before"] = "0x0000000000000001"
        expect_rows_rejected("malformed_state_array", malformed_state_rows)

        tampered_state_rows = synthetic_rows("tampered-state")
        tampered_state_rows[1]["payload"]["rng"]["state_before"][0] = (
            "0x0000000000000000"
        )
        expect_rows_rejected("tampered_state", tampered_state_rows)

        tampered_raw_rows = synthetic_rows("tampered-raw")
        tampered_raw_rows[1]["payload"]["rng"]["raw_u64"] = "0x0000000000000000"
        expect_rows_rejected("tampered_raw_transition", tampered_raw_rows)

        malformed_uniform_rows = synthetic_rows("malformed-uniform")
        malformed_uniform_rows[1]["payload"]["rng"]["raw_uniform_f64_bits"] = (
            encode_f64(0.5)
        )
        expect_rows_rejected("malformed_uniform", malformed_uniform_rows)

        malformed_selection_rows = synthetic_rows("malformed-selection")
        malformed_selection_rows[2]["payload"]["distribution"]["selected_token"] = 9
        malformed_selection_rows[2]["payload"]["distribution"]["candidate_index"] = 0
        expect_rows_rejected("malformed_selection", malformed_selection_rows)

        arm_mismatch_rows = synthetic_rows("arm-mismatch")
        reference = next(
            row
            for row in arm_mismatch_rows
            if row["event"] == "reference_sample_decision"
            and row["payload"]["sample_index"] == 1
        )
        reference["payload"]["distribution"]["ordered_support"][0][
            "weight_f64_bits"
        ] = encode_f64(2.0)
        reference["payload"]["distribution"]["total_weight_f64_bits"] = encode_f64(2.0)
        expect_rows_rejected(
            "oracle_reference_distribution_mismatch", arm_mismatch_rows
        )

        terminal_hash_rows = synthetic_rows("terminal-hash")
        for row in terminal_hash_rows:
            if row["event"] in {"serial_reference_end", "run_end"}:
                row["payload"]["transition_logits_sha256_f32le"] = [
                    "f" * 64,
                    "e" * 64,
                ]
        expect_rows_rejected("terminal_transition_not_sample_bound", terminal_hash_rows)

        oversized_support_rows = synthetic_rows("oversized-support")
        for row in oversized_support_rows:
            if row["event"] not in {"sample_decision", "reference_sample_decision"}:
                continue
            if row["payload"]["sample_index"] != 0:
                continue
            support = row["payload"]["distribution"]["ordered_support"]
            support.extend(
                {
                    "token": token,
                    "weight_f64_bits": encode_f64(0.0),
                }
                for token in range(100, 120)
            )
        expect_rows_rejected("support_exceeds_top_k", oversized_support_rows)

        out_of_vocab_support_rows = synthetic_rows("out-of-vocab-support")
        for row in out_of_vocab_support_rows:
            if row["event"] not in {"sample_decision", "reference_sample_decision"}:
                continue
            if row["payload"]["sample_index"] != 0:
                continue
            row["payload"]["distribution"]["ordered_support"].append(
                {"token": 32, "weight_f64_bits": encode_f64(0.0)}
            )
        expect_rows_rejected(
            "zero_weight_support_outside_target_vocab", out_of_vocab_support_rows
        )

        truncated_snapshot_rows = synthetic_rows("truncated-snapshot")
        for row in truncated_snapshot_rows:
            if row["event"] in {"serial_reference_end", "run_end"}:
                row["payload"]["target_state"]["sections"]["kv_k"]["bytes"] = 4
        expect_rows_rejected("truncated_snapshot_section", truncated_snapshot_rows)

        snapshot_asset_rows = synthetic_rows("snapshot-asset")
        snapshot_end = next(
            row for row in snapshot_asset_rows if row["event"] == "run_end"
        )
        identity = snapshot_end["payload"]["target_state"]["identity"]
        identity["model_id"] ^= 1
        snapshot_end["payload"]["target_state"]["identity_sha256_canonical_le"] = (
            snapshot_identity_sha256(identity)
        )
        expect_rows_rejected("snapshot_not_asset_bound", snapshot_asset_rows)

        reordered_rows = synthetic_rows("event-reorder")
        decision_index = next(
            index
            for index, row in enumerate(reordered_rows)
            if row["event"] == "one_hot_decision"
        )
        block_index = next(
            index
            for index, row in enumerate(reordered_rows)
            if row["event"] == "proposal_block"
        )
        reordered_rows[decision_index], reordered_rows[block_index] = (
            reordered_rows[block_index],
            reordered_rows[decision_index],
        )
        expect_rows_rejected("event_reorder", reordered_rows)

        sample_after_decision_rows = synthetic_rows("sample-after-decision")
        sample_index = next(
            index
            for index, row in enumerate(sample_after_decision_rows)
            if row["event"] == "sample_decision" and row["payload"]["sample_index"] == 1
        )
        decision_index = next(
            index
            for index, row in enumerate(sample_after_decision_rows)
            if row["event"] == "one_hot_decision"
        )
        (
            sample_after_decision_rows[sample_index],
            sample_after_decision_rows[decision_index],
        ) = (
            sample_after_decision_rows[decision_index],
            sample_after_decision_rows[sample_index],
        )
        expect_rows_rejected("sample_after_decision", sample_after_decision_rows)

        trusted_boolean_rows = synthetic_rows("trusted-boolean")
        serial_end = next(
            row
            for row in trusted_boolean_rows
            if row["event"] == "serial_reference_end"
        )
        serial_end["payload"]["comparisons"]["generated_ids"] = False
        expect_rows_rejected("producer_boolean_disagreement", trusted_boolean_rows)

        truncated_rows = synthetic_rows("depth-truncation")
        run_end = next(row for row in truncated_rows if row["event"] == "run_end")
        run_end["payload"]["attempts_by_depth"] = []
        run_end["payload"]["accepts_by_depth"] = []
        expect_rows_rejected("depth_truncation", truncated_rows)

        bad_stop_rows = synthetic_rows("bad-stop")
        bad_stop_end = next(row for row in bad_stop_rows if row["event"] == "run_end")
        bad_stop_end["payload"]["stop_reason"] = "eos"
        expect_rows_rejected("terminal_stop_mismatch", bad_stop_rows)

        bad_pending_rows = synthetic_rows("bad-pending")
        bad_pending_end = next(
            row for row in bad_pending_rows if row["event"] == "run_end"
        )
        bad_pending_end["payload"]["target_state"][
            "pending_token_sha256_tagged_i32le"
        ] = "0" * 64
        expect_rows_rejected("pending_token_digest_mismatch", bad_pending_rows)

        output = root / "output.json"
        output.write_text("occupied", encoding="ascii")
        try:
            write_output({"ok": True}, output)
            raise AssertionError("output overwrite was not rejected")
        except EvidenceError:
            pass
    print("self-test: PASS")


def proposal_temperature(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def write_output(result: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(output.parent or Path("."), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise EvidenceError(f"refusing to overwrite output: {output}") from error
    except OSError as error:
        raise EvidenceError(f"cannot write output {output}: {error}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly reduce schema-v4 DFlash sampled-oracle JSONL."
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs="+",
        type=Path,
        help="Input JSONL files (option may be repeated).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New reduction JSON path; existing paths are refused.",
    )
    parser.add_argument("--serial-ms", type=float)
    parser.add_argument("--draft-ms", type=float)
    parser.add_argument("--verify-ms", type=float)
    parser.add_argument("--other-ms", type=float)
    parser.add_argument("--target-speedup", type=float, default=1.10)
    parser.add_argument(
        "--proposal-temperature",
        type=proposal_temperature,
        help="Explicit local-reference sparse-q temperature (required outside self-test).",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test and not args.input:
        parser.error("at least one --input is required unless --self-test is used")
    if args.input and args.self_test:
        parser.error("--self-test cannot be combined with --input")
    if not args.self_test and args.proposal_temperature is None:
        parser.error(
            "--proposal-temperature is required for explicit local q diagnostics"
        )
    try:
        finite_number(args.target_speedup, "--target-speedup", positive=True)
    except EvidenceError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            run_self_test()
            return 0
        if args.output is not None:
            require(
                not args.output.exists(), f"refusing to overwrite output: {args.output}"
            )
        result = reduce(
            [path for group in args.input for path in group],
            args.proposal_temperature,
            args.target_speedup,
            {
                "serial_ms": args.serial_ms,
                "draft_ms": args.draft_ms,
                "verify_ms": args.verify_ms,
                "other_ms": args.other_ms,
            },
        )
        write_output(result, args.output)
        return 0
    except (EvidenceError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
