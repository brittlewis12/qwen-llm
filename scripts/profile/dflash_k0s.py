#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Pure-offline reducer for qwen.dflash_k0s_lattice schema v1.

The JSON schema is intentionally expressed as constants below.  The producer
emits exactly sixteen records: run, seven depth records, seven lattice records,
and end.  All trust roots are supplied by a separate authenticated manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable


SCHEMA = "qwen.dflash_k0s_lattice"
SCHEMA_VERSION = 1
MANIFEST_SCHEMA = "qwen.dflash_k0s_external_manifest"
MANIFEST_VERSION = 1
REDUCTION_SCHEMA = "qwen.dflash_k0s_reduction"
REDUCTION_VERSION = 1
AUTHORITY = (
    "development_k0s_conditional_on_authenticated_z_only_no_projection_parity_"
    "no_rng_acceptance_k0l_e1b_verifier_product_authority"
)
INVENTORY_SCHEMA = "qwen.dflash_k0s_inventory"
INVENTORY_VERSION = 1
INVENTORY_AUTHORITY = (
    "development_k0s_inventory_only_no_model_forward_or_semantic_authority"
)
INVENTORY_SPEC_SCHEMA = "qwen.dflash_k0s_inventory_spec"
PREPARATION_SPEC_SCHEMA = "qwen.dflash_k0s_preparation_spec"
SEAL_SCHEMA = "qwen.dflash_k0s_preparation_seal"
INVENTORY_KEYS = (
    "schema",
    "schema_version",
    "authority",
    "inventory_spec_sha256",
    "run_id",
    "checkout",
    "build",
    "sources",
    "executable",
    "reducer",
    "scalar_fixture",
    "command_template",
    "embedded_metallib",
    "device",
    "assets",
    "gguf",
    "tensors",
    "tokenizer",
    "prompt",
    "mask_noise",
    "parser_caps",
    "command",
    "environment",
)
INVENTORY_SPEC_KEYS = (
    "schema",
    "schema_version",
    "run_id",
    "inventory_max_bytes",
    "checkout",
    "build",
    "sources",
    "executable",
    "reducer",
    "scalar_fixture",
    "command_template",
    "embedded_metallib",
    "assets",
    "tensor_requirements",
    "tokenizer",
    "prompt",
    "carry_token",
    "expected_mask_token",
    "parser_caps",
    "host_predicate",
    "command",
    "environment",
)
PREPARATION_SPEC_KEYS = (
    "schema",
    "schema_version",
    "run_id",
    "attempt_id",
    "inventory_path",
    "inventory_sha256",
    "inventory_spec_path",
    "inventory_spec_sha256",
    "preparation_spec_path",
    "worktree_x",
    "control_y_input",
    "outputs",
    "fixture_content",
    "acquisition_outputs",
    "continuation_carry_token",
    "manifest_choices",
    "transformation_sha256",
    "environment_allowlist",
    "arm_order",
    "selected_arm",
    "parity_comparison_fields",
    "reducer_argv",
    "reduction_output",
    "failure_policy",
)
INVENTORY_CHECKOUT_KEYS = ("path", "commit", "tree", "dirty")
INVENTORY_GGUF_KEYS = ("role", "version", "tensor_count", "metadata_count")
INVENTORY_TOKENIZER_KEYS = (
    "vocab_size",
    "token_embd_name",
    "token_embd_shape",
    "token_embd_dtype",
    "token_count",
    "tokenizer_tokens_sha256",
)
INVENTORY_PROMPT_KEYS = (
    "utf8_hex",
    "token_ids",
    "token_ids_sha256_i32le",
    "tokenizer_identity_sha256",
)
INVENTORY_MASK_KEYS = (
    "metadata_key",
    "mask_token",
    "noise_tokens",
    "noise_sha256_i32le",
)
HOST_PREDICATE_KEYS = ("os", "arch", "device_name")
TENSOR_REQUIREMENT_KEYS = (
    "role",
    "asset_role",
    "name",
    "dtype",
    "shape",
    "orientation",
    "row_domain",
)
MANIFEST_CHOICE_KEYS = (
    "trace_max_bytes",
    "sidecar_max_bytes",
    "semantic_references",
    "expected_request",
    "expected_binding",
    "expected_rng_domains",
    "expected_fixed_chains",
    "expected_capture_context",
    "selector_dispatch_predicate",
)
ACQUISITION_OUTPUT_KEYS = ("manifest", "trace", "sidecar")
PREPARATION_BINDING_KEYS = (
    "inventory",
    "inventory_spec",
    "preparation_spec",
    "seal_path",
)
SEAL_KEYS = (
    "schema",
    "schema_version",
    "run_id",
    "attempt_id",
    "inventory_sha256",
    "inventory_spec_sha256",
    "preparation_spec_sha256",
    "fixture_sha256",
    "command_sha256",
    "manifest_sha256",
    "transformation_sha256",
    "reducer_sha256",
    "worktree_x_commit",
    "control_y_input_commit",
    "control_y_input_tree",
)
PARITY_COMPARISON_FIELDS = [
    "first",
    "continuation",
    "observer_baseline",
    "common_production_content_sha256",
    "on_capture_content_sha256",
]
PREPARATION_SPEC_SHA256_PLACEHOLDER = "${PREPARATION_SPEC_SHA256}"
SEAL_SHA256_PLACEHOLDER = "${SEAL_SHA256}"

MAX_RECORDS = 16
MAX_TRACE_BYTES = 64 << 20
MAX_SIDECAR_BYTES = 64 << 20
MAX_COMBINED_BYTES = 128 << 20
MAX_JSON_DEPTH = 16
MAX_JSON_INTEGER = (1 << 63) - 1
READ_CHUNK = 1 << 20
MAX_SIDECAR_RANGES = 4096
MAX_DISPATCH_ROWS = 256
MAX_KERNEL_TRACE_ROWS = 1024
MAX_GGUF_TENSORS = 8192
MAX_GGUF_METADATA = 4096
MAX_GGUF_STRINGS_BYTES = 16 << 20
MAX_GGUF_ARRAY_ITEMS = 500000
MAX_GGUF_OBJECTS = 600000
MAX_GGUF_HEADER_BYTES = 64 << 20
DEPTHS = 7
TOP_K = 16
RANK = 256
HIDDEN = 5120
VOCAB = 248320
LATTICE_ROWS = 97

RECORD_KEYS = (
    "schema",
    "schema_version",
    "run_id",
    "attempt_id",
    "event",
    "payload",
)
EVENT_ORDER = ("run",) + ("depth", "lattice") * DEPTHS + ("end",)
RUN_KEYS = (
    "authority",
    "attempt_id",
    "geometry",
    "request",
    "proposal_abstention",
    "ignored_target_policy",
    "binding",
    "provenance",
    "capture",
    "diagnostic_nonperturbation_parity",
    "on_b_projection",
    "identities",
    "semantic_references",
    "tensors",
    "sidecar_registry",
    "production_chain",
    "fixed_chains",
)
PARITY_KEYS = (
    "status",
    "arm_order",
    "selected_arm",
    "arms",
    "comparison_fields",
)
ARM_KEYS = (
    "name",
    "diagnostic",
    "session_id",
    "first_event",
    "continuation_event",
    "summary",
    "rng_domains",
    "capture_projection_sha256",
    "arm_envelope_sha256",
)
ARM_EVENT_KEYS = (
    "kind",
    "library_sequence",
    "library_event_envelope_sha256",
    "session_binding_sha256",
    "draft_tokens",
    "wrapper_binding_sha256",
)
ARM_SUMMARY_KEYS = (
    "domain",
    "first",
    "continuation",
    "observer_baseline",
    "common_production_content_sha256",
    "capture_content_sha256",
)
ARM_PHASE_KEYS = (
    "carry_token",
    "noise_start_position",
    "target_sha256",
    "dflash_sha256",
    "state_sha256",
    "draft_tokens_count",
    "draft_tokens_sha256_i32le",
    "full_logits_count",
    "full_logits_sha256_f32le",
    "topk_count",
    "topk_sha256_i32le",
    "unary_count",
    "unary_sha256_f32le",
    "z_count",
    "z_sha256_f32le",
    "dispatch_census",
    "kernel_trace",
    "runtime_selector_contract",
)
OBSERVER_BASELINE_KEYS = ("before_sha256", "after_sha256", "restored")
RNG_DOMAIN_KEYS = (
    "domain",
    "scope",
    "absent_state_sha256",
    "before_counter",
    "after_counter",
)
RNG_ABSENT_SCOPE = "statically_unreachable_no_rng_object_constructed"
ON_B_KEYS = (
    "exclusion_allowlist",
    "exclusion_content_sha256",
    "depths",
    "lattices",
    "capture",
    "production_chain",
    "fixed_chains",
    "provenance",
    "projection_sha256",
)
PARITY_FAILURE_KEYS = (
    "completed_arms",
    "failed_arm",
    "failed_stage",
    "first_mismatch",
    "observer_cleanup",
    "identities",
    "authority",
    "status",
)
PARITY_FAILURE_STAGES = (
    "session_setup",
    "prompt_prefill",
    "first_block",
    "extraction",
    "continuation",
    "observer_cleanup",
    "comparison",
    "serialization",
)
PARITY_FAILURE_IDENTITY_KEYS = (
    "reducer",
    "executable",
    "fixture",
    "command",
    "sources",
    "assets",
    "build",
    "host",
    "embedded_metallib_sha256",
)
PROJECTION_EXCLUSION_ALLOWLIST = ["arm_name", "session_id", "event_envelope_sha256"]
GEOMETRY_KEYS = ("block_size", "depths", "top_k", "rank", "hidden", "vocab", "rows")
REQUEST_KEYS = ("temperature_f32_bits",)
IGNORED_POLICY_KEYS = (
    "top_k",
    "top_p_f32_bits",
    "min_p_f32_bits",
    "grammar",
    "penalties",
)
ABSTENTION_KEYS = ("enabled", "p_min", "n_min")
BINDING_KEYS = (
    "production_call_id",
    "drafter_checkpoint_sha256",
    "proposal_construction_id",
    "noise_input_sha256",
)
IDENTITY_KEYS = (
    "sidecar",
    "reducer",
    "executable",
    "fixture",
    "command",
    "sources",
    "assets",
)
FILE_CLAIM_KEYS = ("path", "bytes", "sha256", "max_bytes")
TENSOR_KEYS = (
    "role",
    "asset_role",
    "name",
    "dtype",
    "shape",
    "offset",
    "bytes",
    "sha256",
    "orientation",
    "row_domain",
)
RANGE_KEYS = (
    "id",
    "kind",
    "dtype",
    "shape",
    "offset",
    "bytes",
    "sha256",
    "tensor_role",
    "row",
)
DEPTH_KEYS = (
    "depth",
    "position",
    "production_call_id",
    "drafter_checkpoint_sha256",
    "proposal_construction_id",
    "noise_input_sha256",
    "synchronized_capture_sha256",
    "draft_tokens_sha256_i32le",
    "diagnostic_state_sha256",
    "z_f32_bits",
    "full_logits_range_id",
    "top16_ids",
    "unary_f32_bits",
    "topk_issues",
)
LATTICE_KEYS = ("depth", "rows")
ROW_KEYS = (
    "row_index",
    "predecessor_token",
    "predecessor_slot",
    "predecessor_raw_range_id",
    "slots",
    "issues",
    "choice_slot",
)
SLOT_KEYS = (
    "slot",
    "token",
    "unary_f32_bits",
    "successor_raw_range_id",
    "score_f32_bits",
    "issues",
)
CHAIN_KEYS = ("name", "initial_carry", "slots", "events", "tokens", "terminated")
STATIC_CHAIN_KEYS = ("name", "initial_carry", "slots")
CHAIN_EVENT_KEYS = ("kind", "depth", "token", "slot")
END_KEYS = ("producer_status", "authority")
MANIFEST_KEYS = (
    "schema",
    "schema_version",
    "run_id",
    "attempt_id",
    "trace_max_bytes",
    "sidecar_max_bytes",
    "reducer",
    "executable",
    "fixture",
    "command",
    "sources",
    "assets",
    "semantic_references",
    "tensors",
    "expected_request",
    "expected_prompt",
    "expected_binding",
    "expected_continuation_carry_token",
    "expected_rng_domains",
    "expected_fixed_chains",
    "expected_capture_context",
    "expected_build",
    "expected_host",
    "embedded_metallib_sha256",
    "selector_dispatch_predicate",
    "preparation_binding",
    "scalar_contract",
)
SELECTOR_PREDICATE_KEYS = (
    "tag",
    "kernel",
    "weight_dtype",
    "input_dtype",
    "output_dtype",
    "weight_dtype_id",
    "input_dtype_id",
    "output_dtype_id",
    "n",
    "h",
    "r",
    "grid",
    "threads",
    "metal_source_sha256",
    "metallib_sha256",
    "build_source_sha256",
    "allowed_environment",
)

PROVENANCE_KEYS = (
    "dispatch_census",
    "selector_hidden_dispatch",
    "kernel_trace",
    "embedded_metallib_sha256",
    "build",
    "host",
    "environment",
)
DISPATCH_KEYS = (
    "family",
    "tag",
    "encoder_ordinal",
    "encoder_concurrent",
    "kernel",
    "grid",
    "threads",
    "grid_threadgroups",
    "threadgroup_threads",
)
KERNEL_TRACE_KEYS = ("encoders", "concurrent_encoders", "dispatches")
BUILD_KEYS = (
    "commit",
    "source_sha256",
    "dirty",
    "compiler",
    "compiler_version",
    "target",
    "profile",
    "features",
)
HOST_KEYS = ("os", "arch", "device_name", "device_registry_id", "device_family")
CAPTURE_KEYS = (
    "definition_version",
    "noise_start_position",
    "carry_token",
    "synchronized_capture_sha256",
    "draft_tokens",
    "draft_token_bits",
    "draft_tokens_sha256_i32le",
    "state",
)
STATE_KEYS = (
    "target_context_len",
    "context_hidden_watermark",
    "kv_context_watermark",
    "noise_input_sha256",
    "synchronized_event_sha256",
    "diagnostic_state_sha256",
)
SCALAR_CONTRACT_KEYS = (
    "artifact",
    "compiler",
    "compiler_version",
    "target",
    "profile",
    "fixture_domain",
    "fixture_sha256",
    "vectors",
)
SCALAR_VECTOR_KEYS = (
    "name",
    "a_f32_bits",
    "z_f32_bits",
    "successor_f32_bits",
    "unary_f32_bits",
    "score_f32_bits",
)
CAPTURE_DEFINITION = "qwen.dflash_k0s.capture.v1"
SELECTOR_DISPATCH_TAG = "dflash_k0s.selector_hidden_projection.v1"
SELECTOR_KERNEL = "kernel_mat_mat_q4_K_f32"
SCALAR_FIXTURE_DOMAIN = "qwen.dflash_k0s.scalar_contract_fixture.v2"
SCALAR_FIXTURE_EXPECTED_SHA256 = (
    "8c22bf3b4ee51efaf315137a866feef8fc8019d5b21c887411b532bd79fa60e9"
)
MANIFEST_SHA256_PLACEHOLDER = "${MANIFEST_SHA256}"
CAPTURE_CONTEXT_KEYS = (
    "definition_version",
    "carry_token",
    "noise_start_position",
    "target_context_len",
    "context_hidden_watermark",
    "kv_context_watermark",
)
REQUIRED_SOURCE_ROLES = {
    "metal_dflash_rs",
    "bench_rs",
    "qwen_llm_cargo_toml",
    "qwen_cli_cargo_toml",
    "metal_rs",
    "metal_forward_rs",
    "dflash2_metal",
    "mat_mat_mma8_metal",
    "mat_mat_q4_k_metal",
    "build_rs",
    "dflash_k0s_rs",
}
REQUIRED_ASSET_ROLES = {"target", "drafter"}
EXACT_TENSOR_NAMES = {
    "selector_hidden": "selector_hidden.weight",
    "predecessor": "selector_predecessor.weight",
    "successor": "selector_successor.weight",
}

DTYPE_BY_ID = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 30: "BF16"}
GGML_RUNTIME_DTYPE_IDS = {
    "F32": 0,
    "F16": 1,
    "Q8_0": 8,
    "Q4_K": 12,
    "BF16": 30,
}
DTYPE_LAYOUT = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q8_0": (32, 34),
    "Q4_K": (256, 144),
}
GGUF_LAYOUT_BY_ID = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    4: ("Q4_2", 32, 18),
    5: ("Q4_3", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    31: ("Q4_0_4_4", 128, 72),
    32: ("Q4_0_4_8", 256, 144),
    33: ("Q4_0_8_8", 256, 144),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    36: ("IQ4_NL_4_4", 128, 72),
    37: ("IQ4_NL_4_8", 256, 144),
    38: ("IQ4_NL_8_8", 256, 144),
}
SEMANTIC_REFERENCES = {
    "mlx_model_mlx_py": {
        "commit": "07ebd93db9f472af339b644bb70221ad8428328a",
        "sha256": "2f8598eaca4cb814e63ea69e791c1bdf55ba82280bfd299f9141906356b7cb87",
    },
    "vllm_qwen3_dflash2_py": {
        "commit": "b389ac29465b33f9e9c534df221ea3c129e9793f",
        "sha256": "c141daa4b2059c0098224ac36471c2197b7052c100bef0a4dbc2ca79b627053f",
    },
    "vllm_speculator_py": {
        "commit": "b389ac29465b33f9e9c534df221ea3c129e9793f",
        "sha256": "1f6ff5ca9c8f38ff417aafd43bfa3116b5387bf0f7b58721acb2185781879836",
    },
    "llama_cpp_dflash_cpp": {
        "commit": "1deefcca395743049c3820ab8f9b15043f3e9446",
        "sha256": "3b31b1a6b888ec37013a4d6275fdaf1fdeb6a9e3ca25b933ab136acdbcd58c49",
    },
    "llama_cpp_speculative_cpp": {
        "commit": "1deefcca395743049c3820ab8f9b15043f3e9446",
        "sha256": "e14da24022c4c16d2de7cb582257020a02893c5bbcc6bb2b052cf1e8a8525138",
    },
}

FORBIDDEN_KEYS = {
    "rng",
    "seed",
    "random",
    "sample",
    "sampled",
    "sampling",
    "accept",
    "accepted",
    "acceptance",
    "correction",
    "coupling",
    "verifier",
    "generated",
    "generated_ids",
    "generation",
    "generation_event",
    "target_logits",
    "target_distribution",
    "target_p",
    "proposal_rng",
    "proposal_temperature",
    "q_provenance",
    "projection_distance",
    "projection_error",
    "projection_parity",
    "projection_quality",
    "rng_state",
    "rng_draw",
}
ALLOWED_TEMPERATURE_PATH = ("payload", "request", "temperature_f32_bits")
ALLOWED_ABSTENTION_PATHS = {
    ("payload", "proposal_abstention", "enabled"),
    ("payload", "proposal_abstention", "p_min"),
    ("payload", "proposal_abstention", "n_min"),
}


class InvalidEvidence(ValueError):
    pass


class FailedEvidence(ValueError):
    def __init__(self, message: str, metrics: dict[str, Any] | None = None):
        super().__init__(message)
        self.metrics = metrics


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidEvidence(message)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise FailedEvidence(message)


def exact_keys(value: Any, expected: Iterable[str], name: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{name} must be an object")
    wanted = tuple(expected)
    require(
        tuple(value) == wanted,
        f"{name} keys/order mismatch: expected {wanted}, got {tuple(value)}",
    )
    return value


def integer(
    value: Any, name: str, minimum: int = 0, maximum: int = MAX_JSON_INTEGER
) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    require(minimum <= value <= maximum, f"{name} is out of range")
    return value


def text(value: Any, name: str, *, maximum: int = 4096) -> str:
    require(
        isinstance(value, str) and 0 < len(value) <= maximum,
        f"{name} must be bounded nonempty text",
    )
    require("\x00" not in value, f"{name} contains NUL")
    return value


def sha256_text(value: Any, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        f"{name} must be lowercase SHA-256",
    )
    return value


def git_oid_text(value: Any, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(c in "0123456789abcdef" for c in value),
        f"{name} must be a lowercase SHA-1 or SHA-256 Git object ID",
    )
    return value


def bits32(value: Any, name: str, *, finite: bool = False) -> int:
    require(
        isinstance(value, str)
        and len(value) == 10
        and value.startswith("0x")
        and all(c in "0123456789abcdef" for c in value[2:]),
        f"{name} must be canonical f32 bits",
    )
    raw = int(value[2:], 16)
    if finite:
        require((raw & 0x7F800000) != 0x7F800000, f"{name} must be finite")
    return raw


def f32_value(raw: int) -> float:
    return struct.unpack(">f", raw.to_bytes(4, "big"))[0]


def enc32(raw: int) -> str:
    return f"0x{raw:08x}"


def f32_from_number(value: float) -> int:
    try:
        return struct.unpack(">I", struct.pack(">f", value))[0]
    except OverflowError:
        return 0x7F800000 if value > 0 else 0xFF800000


def json_int(raw: str) -> int:
    require(len(raw.lstrip("-")) <= 19, "JSON integer has too many digits")
    value = int(raw)
    require(abs(value) <= MAX_JSON_INTEGER, "JSON integer exceeds signed 63-bit bound")
    return value


def reject_float(raw: str) -> float:
    raise InvalidEvidence(f"JSON floating number is forbidden; use bit fields: {raw}")


def reject_constant(raw: str) -> None:
    raise InvalidEvidence(f"nonfinite JSON constant is forbidden: {raw}")


def unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items:
        require(key not in out, f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def nesting_depth(value: Any, level: int = 1) -> int:
    if isinstance(value, dict):
        return max([level] + [nesting_depth(v, level + 1) for v in value.values()])
    if isinstance(value, list):
        return max([level] + [nesting_depth(v, level + 1) for v in value])
    return level


def parse_json(raw: bytes, name: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_pairs,
            parse_int=json_int,
            parse_float=reject_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, InvalidEvidence) as error:
        raise InvalidEvidence(f"invalid JSON in {name}: {error}") from error
    require(
        nesting_depth(value) <= MAX_JSON_DEPTH,
        f"{name} exceeds JSON depth {MAX_JSON_DEPTH}",
    )
    return value


def reject_forbidden(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            here = path + (key,)
            if "temperature" in lowered:
                require(
                    here == ALLOWED_TEMPERATURE_PATH,
                    f"forbidden temperature field at {'.'.join(here)}",
                )
            if lowered in {"p_min", "n_min"}:
                require(
                    here in ALLOWED_ABSTENTION_PATHS,
                    f"forbidden proposal abstention field at {'.'.join(here)}",
                )
            require(
                lowered not in FORBIDDEN_KEYS,
                f"forbidden field {key!r} at {'.'.join(path)}",
            )
            reject_forbidden(child, here)
    elif isinstance(value, list):
        for child in value:
            reject_forbidden(child, path)


@dataclass
class OpenFile:
    path: Path
    file: BinaryIO
    size: int
    inode: tuple[int, int]
    digest: str

    def close(self) -> None:
        self.file.close()


@dataclass
class InventoryContext:
    inventory: dict[str, Any]
    opened: list[OpenFile]

    def final_check(self) -> None:
        for item in self.opened:
            final_custody_check(item)

    def close(self) -> None:
        for item in self.opened:
            item.close()


def open_regular(path: Path, name: str, maximum: int | None = None) -> OpenFile:
    canonical = path.resolve(strict=True)
    try:
        handle = canonical.open("rb", buffering=0)
    except OSError as error:
        raise InvalidEvidence(f"cannot open {name} {canonical}: {error}") from error
    try:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode), f"{name} is not a regular file")
        require(
            maximum is None or info.st_size <= maximum,
            f"{name} exceeds {maximum} bytes",
        )
        digest = hashlib.sha256()
        offset = 0
        while offset < info.st_size:
            block = os.pread(
                handle.fileno(), min(READ_CHUNK, info.st_size - offset), offset
            )
            require(block, f"short read while hashing {name}")
            digest.update(block)
            offset += len(block)
        return OpenFile(
            canonical,
            handle,
            info.st_size,
            (info.st_dev, info.st_ino),
            digest.hexdigest(),
        )
    except Exception:
        handle.close()
        raise


def pread_exact(opened: OpenFile, offset: int, count: int, name: str) -> bytes:
    require(
        0 <= offset <= opened.size and 0 <= count <= opened.size - offset,
        f"{name} range exceeds file",
    )
    chunks: list[bytes] = []
    cursor = offset
    left = count
    while left:
        block = os.pread(opened.file.fileno(), min(READ_CHUNK, left), cursor)
        require(block, f"short pread for {name}")
        chunks.append(block)
        cursor += len(block)
        left -= len(block)
    return b"".join(chunks)


def hash_range(opened: OpenFile, offset: int, count: int, name: str) -> str:
    require(
        0 <= offset <= opened.size and 0 <= count <= opened.size - offset,
        f"{name} range exceeds file",
    )
    digest = hashlib.sha256()
    cursor = offset
    left = count
    while left:
        block = os.pread(opened.file.fileno(), min(READ_CHUNK, left), cursor)
        require(block, f"short pread hashing {name}")
        digest.update(block)
        cursor += len(block)
        left -= len(block)
    return digest.hexdigest()


def identity_from_claim(value: Any, name: str) -> dict[str, Any]:
    item = exact_keys(value, FILE_CLAIM_KEYS, name)
    text(item["path"], f"{name}.path")
    integer(item["bytes"], f"{name}.bytes", 0)
    sha256_text(item["sha256"], f"{name}.sha256")
    maximum = integer(item["max_bytes"], f"{name}.max_bytes", 1)
    require(item["bytes"] <= maximum, f"{name} exceeds run-plan size cap")
    return item


def verify_identity(opened: OpenFile, claim: dict[str, Any], name: str) -> None:
    check(
        Path(claim["path"]).resolve(strict=True) == opened.path,
        f"{name} canonical path mismatch",
    )
    check(
        claim["bytes"] == opened.size and claim["sha256"] == opened.digest,
        f"{name} identity mismatch",
    )


class Cursor:
    def __init__(self, opened: OpenFile, offset: int = 0):
        self.opened = opened
        self.offset = offset
        self.string_bytes = 0
        self.array_items = 0
        self.objects = 0

    def take(self, count: int) -> bytes:
        require(
            count >= 0 and self.offset <= self.opened.size - count, "truncated GGUF"
        )
        result = pread_exact(self.opened, self.offset, count, "GGUF")
        self.offset += count
        return result

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        count = self.u64()
        require(count <= 1 << 20, "GGUF string is too large")
        self.string_bytes += count
        require(
            self.string_bytes <= MAX_GGUF_STRINGS_BYTES,
            "GGUF cumulative string budget exceeded",
        )
        try:
            return self.take(count).decode("utf-8")
        except UnicodeDecodeError as error:
            raise InvalidEvidence("GGUF string is not UTF-8") from error


def gguf_value(cursor: Cursor, dtype: int, depth: int = 0) -> Any:
    require(depth <= 4, "GGUF metadata nesting too deep")
    cursor.objects += 1
    require(cursor.objects <= MAX_GGUF_OBJECTS, "GGUF metadata object budget exceeded")
    sizes = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
    if dtype in sizes:
        fmt = sizes[dtype]
        return struct.unpack(fmt, cursor.take(struct.calcsize(fmt)))[0]
    if dtype == 8:
        return cursor.string()
    if dtype == 9:
        child = cursor.u32()
        count = cursor.u64()
        require(count <= MAX_GGUF_ARRAY_ITEMS, "GGUF metadata array too large")
        cursor.array_items += count
        require(
            cursor.array_items <= MAX_GGUF_ARRAY_ITEMS,
            "GGUF cumulative metadata array budget exceeded",
        )
        return [gguf_value(cursor, child, depth + 1) for _ in range(count)]
    raise InvalidEvidence(f"unsupported GGUF metadata type {dtype}")


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    size: int


class GGUF:
    def __init__(self, opened: OpenFile):
        self.opened = opened
        cursor = Cursor(opened)
        require(cursor.take(4) == b"GGUF", "bad GGUF magic")
        require(cursor.u32() == 3, "only GGUF v3 is supported")
        tensor_count = cursor.u64()
        metadata_count = cursor.u64()
        require(
            tensor_count <= MAX_GGUF_TENSORS and metadata_count <= MAX_GGUF_METADATA,
            "GGUF counts exceed bounds",
        )
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = cursor.string()
            require(key not in metadata, "duplicate GGUF metadata key")
            metadata[key] = gguf_value(cursor, cursor.u32())
        relative: list[tuple[str, tuple[int, ...], str, int, int]] = []
        names: set[str] = set()
        for _ in range(tensor_count):
            name = cursor.string()
            require(name not in names, "duplicate GGUF tensor name")
            names.add(name)
            dimensions = cursor.u32()
            require(1 <= dimensions <= 4, "GGUF tensor rank out of bounds")
            shape = tuple(cursor.u64() for _ in range(dimensions))
            require(
                all(0 < n <= MAX_JSON_INTEGER for n in shape),
                "GGUF tensor dimension invalid",
            )
            dtype_id = cursor.u32()
            require(
                dtype_id in GGUF_LAYOUT_BY_ID,
                f"unsupported GGUF tensor dtype {dtype_id}",
            )
            dtype, block, block_bytes = GGUF_LAYOUT_BY_ID[dtype_id]
            relative_offset = cursor.u64()
            elements = math.prod(shape)
            require(elements % block == 0, "GGUF tensor is not block aligned")
            size = elements // block * block_bytes
            require(size <= MAX_JSON_INTEGER, "GGUF tensor size overflow")
            relative.append((name, shape, dtype, relative_offset, size))
        alignment = metadata.get("general.alignment", 32)
        require(
            isinstance(alignment, int)
            and not isinstance(alignment, bool)
            and 1 <= alignment <= 4096
            and alignment & (alignment - 1) == 0,
            "GGUF alignment invalid",
        )
        require(
            cursor.offset <= MAX_GGUF_HEADER_BYTES,
            "GGUF header/descriptor budget exceeded",
        )
        data_start = (cursor.offset + alignment - 1) & -alignment
        require(
            data_start <= MAX_GGUF_HEADER_BYTES, "GGUF aligned header budget exceeded"
        )
        tensors: dict[str, Tensor] = {}
        for name, shape, dtype, offset, size in relative:
            absolute = data_start + offset
            require(
                offset % alignment == 0 and absolute <= opened.size - size,
                f"GGUF tensor {name} range invalid",
            )
            tensors[name] = Tensor(name, shape, dtype, absolute, size)
        intervals = sorted(
            (t.offset, t.offset + t.size, t.name) for t in tensors.values()
        )
        for left, right in zip(intervals, intervals[1:]):
            require(
                left[1] <= right[0], f"GGUF tensors overlap: {left[2]} and {right[2]}"
            )
        self.metadata = metadata
        self.tensors = tensors


def f16_to_bits(raw: int) -> int:
    sign = (raw >> 15) << 31
    exponent = (raw >> 10) & 31
    fraction = raw & 1023
    if exponent == 0:
        if fraction == 0:
            return sign
        shift = 10 - (fraction.bit_length() - 1)
        fraction <<= shift
        return sign | ((127 - 14 - shift) << 23) | ((fraction & 1023) << 13)
    if exponent == 31:
        return sign | 0x7F800000 | (fraction << 13)
    return sign | ((exponent + 112) << 23) | (fraction << 13)


def decode_raw(dtype: str, raw: bytes, elements: int) -> list[int]:
    require(dtype in DTYPE_LAYOUT, f"unsupported raw dtype {dtype}")
    block, block_bytes = DTYPE_LAYOUT[dtype]
    require(
        elements > 0
        and elements % block == 0
        and len(raw) == elements // block * block_bytes,
        "raw decoder geometry mismatch",
    )
    out: list[int] = []
    if dtype == "F32":
        return list(struct.unpack(f"<{elements}I", raw))
    if dtype == "F16":
        return [f16_to_bits(x) for x in struct.unpack(f"<{elements}H", raw)]
    if dtype == "BF16":
        return [x << 16 for x in struct.unpack(f"<{elements}H", raw)]
    if dtype == "Q8_0":
        for offset in range(0, len(raw), 34):
            d = f16_to_bits(struct.unpack_from("<H", raw, offset)[0])
            for q in struct.unpack_from("<32b", raw, offset + 2):
                out.append(f32_mul(d, f32_from_number(float(q))))
        return out
    for offset in range(0, len(raw), 144):
        d = f16_to_bits(struct.unpack_from("<H", raw, offset)[0])
        dmin = f16_to_bits(struct.unpack_from("<H", raw, offset + 2)[0])
        scales = raw[offset + 4 : offset + 16]
        qs = raw[offset + 16 : offset + 144]

        def scale_min(index: int) -> tuple[int, int]:
            if index < 4:
                return scales[index] & 63, scales[index + 4] & 63
            return (
                (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4),
                (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
            )

        for group in range(4):
            packed = qs[group * 32 : (group + 1) * 32]
            scale, minimum = scale_min(group * 2)
            for q in packed:
                scaled = f32_mul(d, f32_from_number(float(scale)))
                product = f32_mul(scaled, f32_from_number(float(q & 15)))
                bias = f32_mul(dmin, f32_from_number(float(minimum))) ^ 0x80000000
                out.append(f32_add(product, bias))
            scale, minimum = scale_min(group * 2 + 1)
            for q in packed:
                scaled = f32_mul(d, f32_from_number(float(scale)))
                product = f32_mul(scaled, f32_from_number(float(q >> 4)))
                bias = f32_mul(dmin, f32_from_number(float(minimum))) ^ 0x80000000
                out.append(f32_add(product, bias))
    require(len(out) == elements, "Q4_K decoder internal geometry failure")
    return out


def finite_dyadic(raw: int) -> tuple[int, int]:
    sign = -1 if raw >> 31 else 1
    exponent = (raw >> 23) & 255
    fraction = raw & 0x7FFFFF
    require(exponent != 255, "nonfinite f32 has no finite dyadic")
    if exponent == 0:
        return sign * fraction, -149
    return sign * ((1 << 23) | fraction), exponent - 127 - 23


def round_shift_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value << -shift
    quotient, remainder = divmod(value, 1 << shift)
    halfway = 1 << (shift - 1)
    return quotient + (remainder > halfway or (remainder == halfway and quotient & 1))


def round_dyadic(value: int, exponent: int, negative_zero: bool = False) -> int:
    if value == 0:
        return 0x80000000 if negative_zero else 0
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    magnitude = value.bit_length() - 1 + exponent
    if magnitude > 127:
        return sign | 0x7F800000
    step = -149 if magnitude < -126 else magnitude - 23
    rounded = round_shift_even(value, step - exponent)
    if step == -149 and rounded < 1 << 23:
        return sign | rounded
    if step == -149:
        magnitude = -126
    if rounded == 1 << 24:
        rounded >>= 1
        magnitude += 1
        if magnitude > 127:
            return sign | 0x7F800000
    return sign | ((magnitude + 127) << 23) | (rounded - (1 << 23))


def f32_mul(left: int, right: int) -> int:
    require(
        (left & 0x7F800000) != 0x7F800000 and (right & 0x7F800000) != 0x7F800000,
        "finite f32 multiply required",
    )
    a, ae = finite_dyadic(left)
    b, be = finite_dyadic(right)
    return round_dyadic(a * b, ae + be, bool((left ^ right) >> 31))


def f32_add(left: int, right: int) -> int:
    require(
        (left & 0x7F800000) != 0x7F800000 and (right & 0x7F800000) != 0x7F800000,
        "finite f32 add required",
    )
    a, ae = finite_dyadic(left)
    b, be = finite_dyadic(right)
    exponent = min(ae, be)
    value = (a << (ae - exponent)) + (b << (be - exponent))
    negative_zero = value == 0 and (left >> 31) and (right >> 31)
    return round_dyadic(value, exponent, bool(negative_zero))


def f32_mul_any(left: int, right: int) -> int:
    if classify(left) == classify(right) == "finite":
        return f32_mul(left, right)
    return f32_from_number(f32_value(left) * f32_value(right))


def f32_add_any(left: int, right: int) -> int:
    if classify(left) == classify(right) == "finite":
        return f32_add(left, right)
    return f32_from_number(f32_value(left) + f32_value(right))


def replay_score(a: list[int], z: list[int], b: list[int], unary: int) -> int:
    require(len(a) == len(z) == len(b), "score vectors differ in length")
    accumulator = 0
    for av, zv, bv in zip(a, z, b):
        accumulator = f32_add_any(accumulator, f32_mul_any(f32_mul_any(av, zv), bv))
    return f32_add_any(unary, accumulator)


def classify(raw: int) -> str:
    exponent, fraction = raw & 0x7F800000, raw & 0x7FFFFF
    if exponent != 0x7F800000:
        return "finite"
    if fraction:
        return "nan"
    return "negative_infinity" if raw >> 31 else "positive_infinity"


def reconstruct_top16(logits: list[int]) -> list[int]:
    require(len(logits) == VOCAB, "full logit row has wrong vocabulary geometry")
    check(
        all(classify(raw) == "finite" for raw in logits),
        "nonfinite full-logit row cannot define top16",
    )
    return sorted(range(VOCAB), key=lambda token: (-f32_value(logits[token]), token))[
        :TOP_K
    ]


def derive_topk_issues(
    logits: list[int], observed_ids: list[int], observed_unary: list[int]
) -> tuple[list[dict[str, Any]], list[int] | None]:
    nonfinite = [
        {"kind": "nonfinite_logit", "token": token, "bits": enc32(raw)}
        for token, raw in enumerate(logits)
        if classify(raw) != "finite"
    ]
    if nonfinite:
        return nonfinite, None
    expected_ids = reconstruct_top16(logits)
    issues: list[dict[str, Any]] = []
    for slot, expected in enumerate(expected_ids):
        if observed_ids[slot] != expected:
            issues.append(
                {
                    "kind": "id_mismatch",
                    "slot": slot,
                    "expected": expected,
                    "observed": observed_ids[slot],
                }
            )
        expected_bits = logits[expected]
        if observed_unary[slot] != expected_bits:
            issues.append(
                {
                    "kind": "unary_mismatch",
                    "slot": slot,
                    "expected_bits": enc32(expected_bits),
                    "observed_bits": enc32(observed_unary[slot]),
                }
            )
    return issues, expected_ids


def strict_softmax(scores: list[int], temperature_raw: int) -> list[float]:
    temperature = f32_value(temperature_raw)
    require(
        classify(temperature_raw) == "finite" and temperature > 0.0,
        "request temperature must be finite and positive",
    )
    require(
        scores and all(classify(raw) == "finite" for raw in scores),
        "q requires finite scores",
    )
    values = [float(f32_value(raw)) for raw in scores]
    maximum = max(values)
    weights = [math.exp((value - maximum) / float(temperature)) for value in values]
    require(
        all(math.isfinite(w) and w >= 0.0 for w in weights),
        "softmax produced invalid weight",
    )
    total = math.fsum(weights)
    require(math.isfinite(total) and total > 0.0, "softmax total invalid")
    q = [w / total for w in weights]
    require(
        abs(math.fsum(q) - 1.0) <= float.fromhex("0x1p-48"),
        "softmax normalization tolerance exceeded",
    )
    return q


def expected_issues(
    tokens: list[int], scores: list[int | None]
) -> tuple[list[list[dict[str, Any]]], int | None]:
    all_issues: list[list[dict[str, Any]]] = []
    seen: dict[int, int] = {}
    best_slot: int | None = None
    best = -math.inf
    for slot, (token, score) in enumerate(zip(tokens, scores)):
        issues: list[dict[str, Any]] = []
        if token in seen:
            issues.append(
                {
                    "kind": "duplicate_id",
                    "slot": slot,
                    "token": token,
                    "first_slot": seen[token],
                }
            )
        if not 0 <= token < VOCAB:
            issues.append({"kind": "sentinel", "slot": slot, "token": token})
        else:
            if score is not None:
                kind = classify(score)
                if kind != "finite":
                    issues.append(
                        {
                            "kind": "nonfinite_score",
                            "slot": slot,
                            "token": token,
                            "classification": kind,
                        }
                    )
                value = f32_value(score)
                if not math.isnan(value) and value > best:
                    best, best_slot = value, slot
        seen.setdefault(token, slot)
        all_issues.append(issues)
    return all_issues, best_slot


def sidecar_registry(value: Any, sidecar: OpenFile) -> dict[str, dict[str, Any]]:
    require(
        isinstance(value, list) and 0 < len(value) <= MAX_SIDECAR_RANGES,
        "sidecar_registry count is invalid",
    )
    registry: dict[str, dict[str, Any]] = {}
    cursor = 0
    for index, raw in enumerate(value):
        item = exact_keys(raw, RANGE_KEYS, f"sidecar_registry[{index}]")
        identifier = text(item["id"], f"sidecar_registry[{index}].id", maximum=128)
        require(identifier not in registry, "duplicate/aliased sidecar range id")
        require(
            item["kind"] in {"full_logits", "predecessor_row", "successor_row"},
            "unknown sidecar range kind",
        )
        require(item["dtype"] in DTYPE_LAYOUT, "unknown sidecar range dtype")
        require(
            isinstance(item["shape"], list) and 1 <= len(item["shape"]) <= 2,
            "sidecar range shape invalid",
        )
        shape = [integer(n, "sidecar shape", 1) for n in item["shape"]]
        offset = integer(item["offset"], "sidecar offset")
        count = integer(item["bytes"], "sidecar bytes", 1)
        require(
            offset == cursor,
            "sidecar registry has gap, overlap, alias, or noncanonical offset",
        )
        require(offset <= MAX_JSON_INTEGER - count, "sidecar range integer wrap")
        require(offset + count <= sidecar.size, "sidecar range exceeds file")
        block, block_bytes = DTYPE_LAYOUT[item["dtype"]]
        require(
            math.prod(shape) % block == 0
            and math.prod(shape) // block * block_bytes == count,
            "sidecar range byte geometry mismatch",
        )
        digest = sha256_text(item["sha256"], "sidecar range sha256")
        check(
            hash_range(sidecar, offset, count, identifier) == digest,
            f"sidecar range {identifier} hash mismatch",
        )
        if item["kind"] == "full_logits":
            require(
                item["dtype"] == "F32"
                and shape == [VOCAB]
                and item["tensor_role"] is None
                and item["row"] is None,
                "full-logit range metadata invalid",
            )
        else:
            expected_role = (
                "predecessor" if item["kind"] == "predecessor_row" else "successor"
            )
            require(
                item["tensor_role"] == expected_role and shape == [RANK],
                "codebook range role/shape invalid",
            )
            integer(item["row"], "sidecar codebook row", 0, VOCAB - 1)
        registry[identifier] = item
        cursor += count
    require(cursor == sidecar.size, "sidecar has unreferenced trailing bytes")
    return registry


def parse_trace(opened: OpenFile) -> list[dict[str, Any]]:
    require(0 < opened.size <= MAX_TRACE_BYTES, "trace size invalid")
    raw = pread_exact(opened, 0, opened.size, "trace")
    require(raw.endswith(b"\n"), "trace requires a final newline")
    lines = raw.split(b"\n")[:-1]
    require(
        len(lines) in {1, MAX_RECORDS},
        f"trace must contain exactly one failure or {MAX_RECORDS} success records",
    )
    rows: list[dict[str, Any]] = []
    run_id: str | None = None
    attempt_id: str | None = None
    for index, line in enumerate(lines):
        require(line and line.strip(), f"blank JSONL record {index + 1}")
        row = parse_json(line, f"trace line {index + 1}")
        exact_keys(row, RECORD_KEYS, f"trace line {index + 1}")
        reject_forbidden(row)
        require(
            row["schema"] == SCHEMA
            and integer(row["schema_version"], "trace schema_version", 1, 1)
            == SCHEMA_VERSION,
            "trace schema/version mismatch",
        )
        current = text(row["run_id"], "run_id", maximum=128)
        require(run_id is None or current == run_id, "trace contains multiple run ids")
        run_id = current
        current_attempt = text(row["attempt_id"], "attempt_id", maximum=128)
        require(
            attempt_id is None or current_attempt == attempt_id,
            "trace contains multiple attempt ids",
        )
        attempt_id = current_attempt
        expected_event = "parity_failure" if len(lines) == 1 else EVENT_ORDER[index]
        require(
            row["event"] == expected_event,
            f"trace event order mismatch at record {index + 1}",
        )
        if expected_event == "parity_failure":
            payload = exact_keys(row["payload"], PARITY_FAILURE_KEYS, "parity failure")
            require(
                payload["authority"] == AUTHORITY and payload["status"] == "failed",
                "parity failure authority/status mismatch",
            )
            require(
                isinstance(payload["completed_arms"], list)
                and len(payload["completed_arms"]) <= 4,
                "parity failure completed arms invalid",
            )
            require(
                isinstance(payload["observer_cleanup"], bool),
                "parity failure cleanup invalid",
            )
            for key in ("failed_arm", "failed_stage", "first_mismatch"):
                require(
                    payload[key] is None or isinstance(payload[key], str),
                    f"parity failure {key} invalid",
                )
            require(
                isinstance(payload["identities"], dict),
                "parity failure identities invalid",
            )
        rows.append(row)
    return rows


def validate_tensor_claim(raw: Any, name: str) -> dict[str, Any]:
    item = exact_keys(raw, TENSOR_KEYS, name)
    require(
        item["role"] in {"selector_hidden", "predecessor", "successor"},
        f"{name}.role invalid",
    )
    text(item["asset_role"], f"{name}.asset_role", maximum=64)
    text(item["name"], f"{name}.name")
    require(item["dtype"] in DTYPE_LAYOUT, f"{name}.dtype invalid")
    require(
        isinstance(item["shape"], list) and 1 <= len(item["shape"]) <= 4,
        f"{name}.shape invalid",
    )
    [integer(n, f"{name}.shape", 1) for n in item["shape"]]
    integer(item["offset"], f"{name}.offset")
    integer(item["bytes"], f"{name}.bytes", 1)
    sha256_text(item["sha256"], f"{name}.sha256")
    require(
        item["asset_role"] == "drafter"
        and item["name"] == EXACT_TENSOR_NAMES[item["role"]],
        f"{name} role/name/asset binding invalid",
    )
    if item["role"] == "selector_hidden":
        require(
            item["orientation"] == "gguf_ne0_hidden_ne1_rank"
            and item["row_domain"] is None
            and item["shape"] == [HIDDEN, RANK],
            "selector-hidden orientation/domain invalid",
        )
    else:
        require(
            item["orientation"] == "gguf_ne0_rank_ne1_token"
            and item["row_domain"] == {"first": 0, "count": VOCAB},
            f"{name} codebook orientation/domain invalid",
        )
        require(item["shape"] == [RANK, VOCAB], f"{name} codebook shape invalid")
    return item


def tensor_assets(
    manifest: dict[str, Any], opened_assets: dict[str, OpenFile]
) -> tuple[dict[str, dict[str, Any]], dict[str, GGUF]]:
    claims = manifest["tensors"]
    require(
        isinstance(claims, list) and len(claims) == 3,
        "manifest tensors must contain exactly three descriptors",
    )
    by_role: dict[str, dict[str, Any]] = {}
    ggufs: dict[str, GGUF] = {
        role: GGUF(opened) for role, opened in opened_assets.items()
    }
    for index, raw in enumerate(claims):
        claim = validate_tensor_claim(raw, f"manifest.tensors[{index}]")
        require(claim["role"] not in by_role, "duplicate tensor role")
        asset_role = claim["asset_role"]
        require(asset_role in opened_assets, "tensor refers to unknown asset role")
        gguf = ggufs[asset_role]
        require(claim["name"] in gguf.tensors, "claimed tensor is absent from GGUF")
        actual = gguf.tensors[claim["name"]]
        check(
            actual.shape == tuple(claim["shape"])
            and actual.dtype == claim["dtype"]
            and actual.offset == claim["offset"]
            and actual.size == claim["bytes"],
            f"{claim['role']} GGUF descriptor mismatch",
        )
        check(
            hash_range(
                opened_assets[asset_role], actual.offset, actual.size, actual.name
            )
            == claim["sha256"],
            f"{claim['role']} full tensor hash mismatch",
        )
        by_role[claim["role"]] = claim
    require(
        set(by_role) == {"selector_hidden", "predecessor", "successor"},
        "tensor roles incomplete",
    )
    require(
        by_role["predecessor"]["name"] != by_role["successor"]["name"],
        "A/B tensor alias or swap",
    )
    return by_role, ggufs


def validate_vocab_compatibility(
    tensors: dict[str, dict[str, Any]], ggufs: dict[str, GGUF]
) -> None:
    target = ggufs["target"]
    require(
        "token_embd.weight" in target.tensors,
        "target GGUF lacks token_embd.weight vocabulary descriptor",
    )
    embedding = target.tensors["token_embd.weight"]
    check(
        len(embedding.shape) == 2 and embedding.shape[1] == VOCAB,
        "target token embedding vocabulary is not 248320",
    )
    if "tokenizer.ggml.tokens" in target.metadata:
        tokens = target.metadata["tokenizer.ggml.tokens"]
        check(
            isinstance(tokens, list) and len(tokens) == VOCAB,
            "target tokenizer token list count is not 248320",
        )
    if "tokenizer.ggml.token_count" in target.metadata:
        check(
            target.metadata["tokenizer.ggml.token_count"] == VOCAB,
            "target tokenizer token count is not 248320",
        )
    for role in ("predecessor", "successor"):
        check(
            tensors[role]["shape"] == [RANK, VOCAB]
            and tensors[role]["row_domain"] == {"first": 0, "count": VOCAB},
            f"drafter {role} vocabulary domain is not 248320",
        )


def row_bytes_from_asset(
    role: str,
    token: int,
    tensors: dict[str, dict[str, Any]],
    assets: dict[str, OpenFile],
) -> bytes:
    tensor = tensors[role]
    block, block_bytes = DTYPE_LAYOUT[tensor["dtype"]]
    row_bytes = RANK // block * block_bytes
    return pread_exact(
        assets[tensor["asset_role"]],
        tensor["offset"] + token * row_bytes,
        row_bytes,
        f"{role} row {token}",
    )


def validate_provenance(value: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    provenance = exact_keys(value, PROVENANCE_KEYS, "provenance")
    census = provenance["dispatch_census"]
    require(
        isinstance(census, list) and 0 < len(census) <= MAX_DISPATCH_ROWS,
        "dispatch census count invalid",
    )
    selector_dispatches = 0
    last_ordinal = -1
    for index, raw in enumerate(census):
        row = exact_keys(raw, DISPATCH_KEYS, f"dispatch_census[{index}]")
        text(row["family"], "dispatch family", maximum=128)
        require(
            row["tag"] is None or isinstance(row["tag"], str),
            "dispatch tag must be null or text",
        )
        if row["tag"] is not None:
            text(row["tag"], "dispatch tag", maximum=128)
        text(row["kernel"], "dispatch kernel", maximum=256)
        ordinal = integer(row["encoder_ordinal"], "encoder ordinal")
        require(
            ordinal >= last_ordinal, "dispatch encoder ordinals must be nondecreasing"
        )
        last_ordinal = ordinal
        require(
            isinstance(row["encoder_concurrent"], bool),
            "encoder_concurrent must be boolean",
        )
        for key in ("grid", "threads"):
            require(
                isinstance(row[key], list) and len(row[key]) == 3,
                f"dispatch {key} invalid",
            )
            [integer(v, f"dispatch {key}", 1) for v in row[key]]
        integer(row["grid_threadgroups"], "grid_threadgroups", 1)
        integer(row["threadgroup_threads"], "threadgroup_threads", 1)
        require(
            row["grid_threadgroups"] == math.prod(row["grid"])
            and row["threadgroup_threads"] == math.prod(row["threads"]),
            "dispatch aggregate geometry counters mismatch",
        )
        selector_dispatches += row["tag"] == SELECTOR_DISPATCH_TAG
    require(
        selector_dispatches == 1,
        "dispatch census requires exactly one tagged selector-hidden dispatch",
    )
    selector = exact_keys(
        provenance["selector_hidden_dispatch"],
        DISPATCH_KEYS,
        "selector_hidden_dispatch",
    )
    check(
        selector["tag"] == SELECTOR_DISPATCH_TAG and selector in census,
        "selector-hidden dispatch is not the uniquely tagged census row",
    )
    predicate = manifest["selector_dispatch_predicate"]
    check(
        selector["tag"] == predicate["tag"]
        and selector["kernel"] == predicate["kernel"]
        and selector["grid"] == predicate["grid"]
        and selector["threads"] == predicate["threads"],
        "tagged selector dispatch differs from static structural predicate",
    )
    trace = exact_keys(provenance["kernel_trace"], KERNEL_TRACE_KEYS, "kernel_trace")
    for key in KERNEL_TRACE_KEYS:
        integer(trace[key], f"kernel trace {key}", 0)
    encoder_ordinals = {row["encoder_ordinal"] for row in census}
    concurrent_ordinals = {
        row["encoder_ordinal"] for row in census if row["encoder_concurrent"]
    }
    require(
        trace["encoders"] == len(encoder_ordinals)
        and trace["dispatches"] == len(census)
        and trace["concurrent_encoders"] == len(concurrent_ordinals),
        "kernel trace counters are inconsistent with complete dispatch census",
    )
    sha256_text(provenance["embedded_metallib_sha256"], "embedded metallib")
    check(
        provenance["embedded_metallib_sha256"] == manifest["embedded_metallib_sha256"],
        "embedded metallib differs from static manifest expectation",
    )
    build = exact_keys(provenance["build"], BUILD_KEYS, "build")
    git_oid_text(build["commit"], "build commit")
    sha256_text(build["source_sha256"], "build source")
    require(build["dirty"] is False, "passing build must be clean")
    for key in ("compiler", "compiler_version", "target", "profile"):
        text(build[key], f"build {key}", maximum=256)
    require(build["profile"] == "release", "scalar contract requires release profile")
    require(
        isinstance(build["features"], list)
        and build["features"] == ["dflash-k0s-diagnostics"],
        "build features mismatch",
    )
    check(build == manifest["expected_build"], "build differs from static manifest")
    host = exact_keys(provenance["host"], HOST_KEYS, "host")
    for key in ("os", "arch", "device_name", "device_family"):
        text(host[key], f"host {key}", maximum=256)
    integer(host["device_registry_id"], "device registry id")
    check(host == manifest["expected_host"], "host differs from static manifest")
    require(
        provenance["environment"]
        == manifest["selector_dispatch_predicate"]["allowed_environment"]
        == {"QWEN_METAL_LEASE_WAIT": "1"},
        "observed runtime environment differs from exact allowlist",
    )
    return provenance


def validate_capture_shape(value: Any) -> dict[str, Any]:
    capture = exact_keys(value, CAPTURE_KEYS, "capture")
    require(
        capture["definition_version"] == CAPTURE_DEFINITION,
        "capture digest definition mismatch",
    )
    integer(
        capture["noise_start_position"],
        "noise start position",
        0,
        (1 << 32) - 1 - DEPTHS,
    )
    integer(capture["carry_token"], "carry token", 0, VOCAB - 1)
    sha256_text(capture["synchronized_capture_sha256"], "synchronized capture")
    require(
        isinstance(capture["draft_tokens"], list) and len(capture["draft_tokens"]) == 8,
        "draft output geometry invalid",
    )
    draft = [
        integer(token, "draft token", 0, VOCAB - 1) for token in capture["draft_tokens"]
    ]
    require(
        isinstance(capture["draft_token_bits"], list)
        and len(capture["draft_token_bits"]) == len(draft),
        "draft token bits geometry invalid",
    )
    draft_bits = [
        bits32(value, "draft token bits") for value in capture["draft_token_bits"]
    ]
    check(
        draft_bits == [token & 0xFFFFFFFF for token in draft],
        "draft token bits differ from draft output",
    )
    draft_digest = hashlib.sha256(
        b"".join(struct.pack("<i", token) for token in draft)
    ).hexdigest()
    check(
        capture["draft_tokens_sha256_i32le"] == draft_digest,
        "draft output hash mismatch",
    )
    state = exact_keys(capture["state"], STATE_KEYS, "capture state")
    for key in STATE_KEYS[:3]:
        integer(state[key], f"capture state {key}")
    for key in STATE_KEYS[3:]:
        sha256_text(state[key], f"capture state {key}")
    return capture


def capture_digest(
    provenance: dict[str, Any],
    capture: dict[str, Any],
    materials: list[tuple[bytes, list[int], list[int], list[int]]],
    tensors: list[dict[str, Any]],
) -> str:
    def hash_bytes(digest: Any, raw: bytes) -> None:
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)

    def hash_dispatch(digest: Any, row: dict[str, Any]) -> None:
        hash_bytes(digest, row["family"].encode())
        if row["tag"] is None:
            digest.update(b"\0")
        else:
            digest.update(b"\1")
            hash_bytes(digest, row["tag"].encode())
        digest.update(
            struct.pack("<Q?", row["encoder_ordinal"], row["encoder_concurrent"])
        )
        hash_bytes(digest, row["kernel"].encode())
        for value in row["grid"] + row["threads"]:
            digest.update(struct.pack("<Q", value))
        digest.update(
            struct.pack("<QQ", row["grid_threadgroups"], row["threadgroup_threads"])
        )

    digest = hashlib.sha256()
    digest.update(CAPTURE_DEFINITION.encode("ascii"))
    state = capture["state"]
    digest.update(
        struct.pack("<iI", capture["carry_token"], capture["noise_start_position"])
    )
    digest.update(
        struct.pack(
            "<QQQ",
            state["target_context_len"],
            state["context_hidden_watermark"],
            state["kv_context_watermark"],
        )
    )
    digest.update(bytes.fromhex(capture["draft_tokens_sha256_i32le"]))
    digest.update(bytes.fromhex(state["noise_input_sha256"]))
    digest.update(bytes.fromhex(state["synchronized_event_sha256"]))
    digest.update(bytes.fromhex(state["diagnostic_state_sha256"]))
    digest.update(struct.pack("<Q", len(capture["draft_token_bits"])))
    for raw in capture["draft_token_bits"]:
        digest.update(struct.pack("<I", bits32(raw, "capture draft token bits")))
    digest.update(struct.pack("<Q", len(materials)))
    for depth, (logits, ids, unary, z) in enumerate(materials, 1):
        digest.update(
            struct.pack("<QI", depth, capture["noise_start_position"] + depth)
        )
        digest.update(struct.pack("<Q", len(logits) // 4))
        digest.update(logits)
        digest.update(struct.pack("<Q", len(ids)))
        digest.update(b"".join(struct.pack("<i", token) for token in ids))
        digest.update(struct.pack("<Q", len(unary)))
        digest.update(b"".join(struct.pack("<I", raw) for raw in unary))
        digest.update(struct.pack("<Q", len(z)))
        digest.update(b"".join(struct.pack("<I", raw) for raw in z))
    digest.update(struct.pack("<Q", len(provenance["dispatch_census"])))
    for row in provenance["dispatch_census"]:
        hash_dispatch(digest, row)
    hash_dispatch(digest, provenance["selector_hidden_dispatch"])
    trace = provenance["kernel_trace"]
    digest.update(
        struct.pack(
            "<QQQ", trace["encoders"], trace["concurrent_encoders"], trace["dispatches"]
        )
    )
    by_role = {tensor["role"]: tensor for tensor in tensors}
    for role in ("selector_hidden", "predecessor", "successor"):
        digest.update(bytes.fromhex(by_role[role]["sha256"]))
    digest.update(bytes.fromhex(provenance["embedded_metallib_sha256"]))
    return digest.hexdigest()


def synchronized_event_digest(
    capture: dict[str, Any],
    materials: list[tuple[bytes, list[int], list[int], list[int]]],
) -> str:
    state = capture["state"]
    digest = hashlib.sha256()
    digest.update(
        struct.pack("<iI", capture["carry_token"], capture["noise_start_position"])
    )
    digest.update(
        struct.pack(
            "<QQQ",
            state["target_context_len"],
            state["context_hidden_watermark"],
            state["kv_context_watermark"],
        )
    )
    digest.update(bytes.fromhex(state["noise_input_sha256"]))
    for _, ids, _, _ in materials:
        for token in ids:
            digest.update(struct.pack("<i", token))
    for _, _, unary, _ in materials:
        for raw in unary:
            digest.update(struct.pack("<I", raw))
    for _, _, _, z in materials:
        for raw in z:
            digest.update(struct.pack("<I", raw))
    return digest.hexdigest()


def scalar_fixture_digest(vectors: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(SCALAR_FIXTURE_DOMAIN.encode("ascii"))
    digest.update(struct.pack("<Q", len(vectors)))
    for vector in vectors:
        name = vector["name"].encode("utf-8")
        digest.update(struct.pack("<Q", len(name)) + name)
        for key in ("a_f32_bits", "z_f32_bits", "successor_f32_bits"):
            digest.update(struct.pack("<Q", len(vector[key])))
            for value in vector[key]:
                digest.update(struct.pack("<I", bits32(value, f"fixture {key}")))
        digest.update(
            struct.pack("<I", bits32(vector["unary_f32_bits"], "fixture unary"))
        )
        digest.update(
            struct.pack("<I", bits32(vector["score_f32_bits"], "fixture score"))
        )
    return digest.hexdigest()


def validate_scalar_contract(
    contract: Any, fixture: OpenFile, build: dict[str, Any]
) -> None:
    item = exact_keys(contract, SCALAR_CONTRACT_KEYS, "scalar_contract")
    identity_from_claim(item["artifact"], "scalar contract artifact")
    verify_identity(fixture, item["artifact"], "scalar contract artifact")
    for key in ("compiler", "compiler_version", "target", "profile"):
        check(item[key] == build[key], f"scalar contract {key} differs from build")
    require(
        item["fixture_domain"] == SCALAR_FIXTURE_DOMAIN,
        "scalar fixture domain mismatch",
    )
    sha256_text(item["fixture_sha256"], "scalar canonical fixture digest")
    fixture_json = parse_json(
        pread_exact(fixture, 0, fixture.size, "scalar fixture"), "scalar fixture"
    )
    exact_keys(
        fixture_json,
        ("schema", "schema_version", "fixture_domain", "fixture_sha256", "vectors"),
        "scalar fixture",
    )
    require(
        fixture_json["schema"] == "qwen.dflash_k0s_scalar_fixture"
        and fixture_json["schema_version"] == 1,
        "scalar fixture schema mismatch",
    )
    require(
        fixture_json["fixture_domain"] == SCALAR_FIXTURE_DOMAIN,
        "scalar fixture file domain mismatch",
    )
    check(
        fixture_json["vectors"] == item["vectors"],
        "scalar fixture vectors differ from manifest",
    )
    vectors = item["vectors"]
    require(
        isinstance(vectors, list) and 6 <= len(vectors) <= 64,
        "scalar fixture vector count invalid",
    )
    names: set[str] = set()
    for index, raw in enumerate(vectors):
        vector = exact_keys(raw, SCALAR_VECTOR_KEYS, f"scalar vector {index}")
        name = text(vector["name"], "scalar vector name", maximum=128)
        require(name not in names, "duplicate scalar vector name")
        names.add(name)
        operands = []
        for key in ("a_f32_bits", "z_f32_bits", "successor_f32_bits"):
            require(
                isinstance(vector[key], list) and len(vector[key]) == RANK,
                f"scalar vector {key} must have rank 256",
            )
            operands.append(
                [bits32(value, f"scalar vector {key}") for value in vector[key]]
            )
        require(
            len(operands[0]) == len(operands[1]) == len(operands[2]),
            "scalar vector lengths mismatch",
        )
        unary = bits32(vector["unary_f32_bits"], "scalar unary")
        observed = bits32(vector["score_f32_bits"], "scalar score")
        expected = replay_score(*operands, unary)
        if classify(expected) == "finite":
            check(observed == expected, f"scalar vector {name} bit mismatch")
        else:
            check(
                classify(observed) == classify(expected),
                f"scalar vector {name} class mismatch",
            )
    require(
        {
            "fma_sensitive_cancellation",
            "subnormal_signed_result",
            "signed_zero",
            "overflow_adjacent_finite",
            "rank_order_cancellation",
            "halfway_round_to_even",
        }
        == names,
        "scalar fixture lacks required adversarial vectors",
    )
    computed_fixture = scalar_fixture_digest(vectors)
    check(
        fixture_json["fixture_sha256"]
        == item["fixture_sha256"]
        == computed_fixture
        == SCALAR_FIXTURE_EXPECTED_SHA256,
        "scalar canonical fixture digest mismatch",
    )


def validate_command_manifest(
    command_file: OpenFile,
    manifest_file: OpenFile,
    trace: OpenFile,
    sidecar: OpenFile,
    manifest: dict[str, Any],
    external: dict[str, OpenFile],
    manifest_sha256: str,
) -> dict[str, Any]:
    command = parse_json(
        pread_exact(command_file, 0, command_file.size, "command manifest"),
        "command manifest",
    )
    exact_keys(command, ("argv",), "command manifest")
    argv = command["argv"]
    require(
        isinstance(argv, list)
        and all(isinstance(value, str) and value for value in argv),
        "command argv must contain nonempty UTF-8 strings",
    )
    require(
        argv.count(MANIFEST_SHA256_PLACEHOLDER) == 1,
        "command requires exactly one manifest digest placeholder",
    )
    fixed = manifest["expected_fixed_chains"]
    require(len(argv) == 28 + 2 * len(fixed), "command argv length mismatch")
    index = 0

    def take(value: str, name: str) -> None:
        nonlocal index
        require(argv[index] == value, f"command argv {name} mismatch")
        index += 1

    take(str(external["executable"].path), "executable")
    take("dflash-k0s-lattice", "subcommand")
    take("--attempt-id", "attempt flag")
    take(manifest["attempt_id"], "attempt value")
    take("--model", "model flag")
    take(str(external["target"].path), "target path")
    take("--drafter", "drafter flag")
    take(str(external["drafter"].path), "drafter path")
    take("--prompt", "prompt flag")
    require(
        argv[index]
        == bytes.fromhex(manifest["expected_prompt"]["utf8_hex"]).decode("utf-8"),
        "command prompt differs from static prompt",
    )
    index += 1
    take("--carry-token", "carry flag")
    take(str(manifest["expected_capture_context"]["carry_token"]), "carry value")
    take("--continuation-carry-token", "continuation carry flag")
    take(str(manifest["expected_continuation_carry_token"]), "continuation carry value")
    take("--manifest", "manifest flag")
    take(str(manifest_file.path), "manifest path")
    take("--manifest-sha256", "manifest digest flag")
    take(MANIFEST_SHA256_PLACEHOLDER, "manifest digest placeholder")
    take("--command-manifest", "command manifest flag")
    take(str(command_file.path), "command manifest path")
    take("--fixture", "fixture flag")
    take(str(external["fixture"].path), "fixture path")
    take("--temperature", "temperature flag")
    temperature_text = argv[index]
    index += 1
    try:
        temperature_bits = f32_from_number(float(temperature_text))
    except (ValueError, OverflowError) as error:
        raise InvalidEvidence(
            "command temperature is not a canonical number"
        ) from error
    require(
        enc32(temperature_bits)
        == manifest["expected_request"]["request"]["temperature_f32_bits"],
        "command temperature differs from static request",
    )
    for chain in fixed:
        take("--fixed-chain", "fixed-chain flag")
        expected = f"{chain['name']}:{chain['initial_carry']}:" + ",".join(
            str(slot) for slot in chain["slots"]
        )
        take(expected, "fixed-chain value")
    take("--trace-output", "trace output flag")
    take(str(trace.path), "trace output path")
    take("--sidecar-output", "sidecar output flag")
    take(str(sidecar.path), "sidecar output path")
    require(index == len(argv), "command argv has trailing values")
    require(
        argv.count(MANIFEST_SHA256_PLACEHOLDER) == 1
        and argv.index(MANIFEST_SHA256_PLACEHOLDER) > 0
        and argv[argv.index(MANIFEST_SHA256_PLACEHOLDER) - 1] == "--manifest-sha256",
        "command requires exactly one manifest digest placeholder immediately after its flag",
    )
    require(
        manifest_sha256 not in argv,
        "static command manifest must not contain the raw manifest digest",
    )
    substituted = [
        manifest_sha256 if value == MANIFEST_SHA256_PLACEHOLDER else value
        for value in argv
    ]
    require(
        substituted.count(manifest_sha256) == 1
        and MANIFEST_SHA256_PLACEHOLDER not in substituted,
        "manifest digest substitution model is not uniquely constructible",
    )
    return {
        "static_argv_sha256_canonical_json": hashlib.sha256(
            json.dumps(argv, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "substituted_argv_sha256_canonical_json": hashlib.sha256(
            json.dumps(substituted, ensure_ascii=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest(),
    }


def validate_chain(
    raw: Any,
    name: str,
    rows: dict[tuple[int, int], dict[str, Any]],
    production: bool,
    semantic_check: Any = check,
) -> None:
    chain = exact_keys(raw, CHAIN_KEYS, name)
    text(chain["name"], f"{name}.name", maximum=128)
    carry = integer(
        chain["initial_carry"],
        f"{name}.initial_carry",
        -MAX_JSON_INTEGER,
        MAX_JSON_INTEGER,
    )
    require(
        isinstance(chain["slots"], list) and len(chain["slots"]) <= DEPTHS,
        f"{name}.slots invalid",
    )
    slots = [integer(v, f"{name}.slot", 0, TOP_K - 1) for v in chain["slots"]]
    require(len(slots) == DEPTHS, f"{name}.slots must contain exactly seven entries")
    require(
        isinstance(chain["events"], list) and len(chain["events"]) <= 1,
        f"{name}.events invalid",
    )
    require(isinstance(chain["tokens"], list), f"{name}.tokens invalid")
    derived_tokens: list[int] = []
    derived_events: list[dict[str, Any]] = []
    derived_terminated = False
    predecessor_slot = -1
    token = carry
    if not 0 <= token < VOCAB:
        derived_events.append(
            {"kind": "invalid_carry", "depth": 1, "token": token, "slot": None}
        )
        derived_terminated = True
    else:
        for depth in range(1, DEPTHS + 1):
            key = (depth, predecessor_slot)
            if key not in rows:
                derived_events.append(
                    {
                        "kind": "missing_predecessor_row",
                        "depth": depth,
                        "token": token,
                        "slot": None if predecessor_slot < 0 else predecessor_slot,
                    }
                )
                derived_terminated = True
                break
            row = rows[key]
            semantic_check(
                row["predecessor_token"] == token, f"{name} predecessor token mismatch"
            )
            selected = (
                row["choice_slot"]
                if production
                else (slots[depth - 1] if depth - 1 < len(slots) else None)
            )
            if production:
                semantic_check(
                    slots[depth - 1] == selected,
                    f"{name} production slot mismatch",
                )
            if selected is None:
                break
            slot = row["slots"][selected]
            next_token = slot["token"]
            if not 0 <= next_token < VOCAB:
                if (
                    production
                    and selected == 0
                    and any(
                        issue.get("kind") == "no_valid_choice"
                        for issue in row["issues"]
                    )
                ):
                    derived_events.append(
                        {
                            "kind": "slot_zero_termination",
                            "depth": depth,
                            "token": next_token,
                            "slot": 0,
                        }
                    )
                derived_terminated = True
                break
            derived_tokens.append(next_token)
            token, predecessor_slot = next_token, selected
    for event in chain["events"]:
        exact_keys(event, CHAIN_EVENT_KEYS, f"{name}.event")
    semantic_check(
        chain["tokens"] == derived_tokens
        and chain["events"] == derived_events
        and chain["terminated"] is derived_terminated,
        f"{name} traversal mismatch",
    )
    semantic_check(
        not derived_terminated and not derived_events,
        f"{name} terminates or contains a chain event",
    )


def canonical_json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def canonical_domain_digest(domain: str, value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)
    return digest.hexdigest()


def event_wrapper_digest(
    attempt_id: str, arm: str, session: str, phase: str, event: dict[str, Any]
) -> str:
    return canonical_domain_digest(
        "qwen.dflash_k0s.event_wrapper.v1",
        {
            "attempt_id": attempt_id,
            "arm": arm,
            "session_id": session,
            "phase": phase,
            "kind": event["kind"],
            "library_sequence": event["library_sequence"],
            "library_event_envelope_sha256": event["library_event_envelope_sha256"],
            "session_binding_sha256": event["session_binding_sha256"],
            "draft_tokens": event["draft_tokens"],
        },
    )


def rng_absent_digest(attempt_id: str, arm: str, domain: str) -> str:
    return canonical_domain_digest(
        "qwen.dflash_k0s.rng_absent.v1",
        {
            "attempt_id": attempt_id,
            "arm": arm,
            "domain": domain,
            "scope": RNG_ABSENT_SCOPE,
        },
    )


def common_production_digest(
    first: dict[str, Any], continuation: dict[str, Any]
) -> str:
    return canonical_domain_digest(
        "qwen.dflash_k0s.common_production.v1",
        {"first": first, "continuation": continuation},
    )


def arm_envelope_digest(arm: dict[str, Any], attempt_id: str) -> str:
    return canonical_domain_digest(
        "qwen.dflash_k0s.cli_arm_envelope.v1",
        {
            "attempt_id": attempt_id,
            "name": arm["name"],
            "diagnostic": arm["diagnostic"],
            "session_id": arm["session_id"],
            "first_event": arm["first_event"],
            "continuation_event": arm["continuation_event"],
            "summary": arm["summary"],
            "rng_domains": arm["rng_domains"],
            "capture_projection_sha256": arm["capture_projection_sha256"],
        },
    )


def rust_vector_hash(domain: str, values: list[int], *, signed: bool = False) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(struct.pack("<Q", len(values)))
    format_code = "<i" if signed else "<I"
    for value in values:
        digest.update(struct.pack(format_code, value))
    return digest.hexdigest()


def hash_rust_dispatch(digest: Any, row: dict[str, Any]) -> None:
    def bounded_bytes(raw: bytes) -> None:
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)

    bounded_bytes(row["family"].encode("utf-8"))
    if row["tag"] is None:
        digest.update(b"\0")
    else:
        digest.update(b"\1")
        bounded_bytes(row["tag"].encode("utf-8"))
    digest.update(struct.pack("<Q", row["encoder_ordinal"]))
    digest.update(bytes([int(row["encoder_concurrent"])]))
    bounded_bytes(row["kernel"].encode("utf-8"))
    for value in row["grid"] + row["threads"]:
        digest.update(struct.pack("<Q", value))
    digest.update(
        struct.pack("<QQ", row["grid_threadgroups"], row["threadgroup_threads"])
    )


def rust_library_event_envelope(event: dict[str, Any], phase: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"qwen.dflash_k0s.event_envelope.v1")
    digest.update(struct.pack("<Q", event["library_sequence"]))
    digest.update(bytes.fromhex(event["session_binding_sha256"]))
    digest.update(
        struct.pack("<iI", phase["carry_token"], phase["noise_start_position"])
    )
    draft = event["draft_tokens"]
    digest.update(struct.pack("<Q", len(draft)))
    for token in draft:
        digest.update(struct.pack("<i", token))
    digest.update(struct.pack("<Q", len(phase["dispatch_census"])))
    for row in phase["dispatch_census"]:
        hash_rust_dispatch(digest, row)
    trace = phase["kernel_trace"]
    for key in ("encoders", "concurrent_encoders", "dispatches"):
        digest.update(struct.pack("<Q", trace[key]))
    runtime = phase["runtime_selector_contract"]
    digest.update(
        struct.pack(
            "<IIIQQQ",
            runtime["weight_dtype_id"],
            runtime["input_dtype_id"],
            runtime["output_dtype_id"],
            runtime["n"],
            runtime["h"],
            runtime["r"],
        )
    )
    for count_key, digest_key in (
        ("full_logits_count", "full_logits_sha256_f32le"),
        ("topk_count", "topk_sha256_i32le"),
        ("unary_count", "unary_sha256_f32le"),
        ("z_count", "z_sha256_f32le"),
    ):
        digest.update(struct.pack("<Q", phase[count_key]))
        digest.update(bytes.fromhex(phase[digest_key]))
    digest.update(bytes.fromhex(phase["dflash_sha256"]))
    return digest.hexdigest()


def validate_parity(value: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    parity = exact_keys(value, PARITY_KEYS, "diagnostic_nonperturbation_parity")
    check(parity["status"] == "passed", "diagnostic parity status is not passed")
    order = ["off-A", "on-A", "on-B", "off-B"]
    require(
        parity["arm_order"] == order and parity["selected_arm"] == "on-A",
        "parity arm order/selection mismatch",
    )
    require(
        isinstance(parity["arms"], list) and len(parity["arms"]) == 4,
        "parity requires four arms",
    )
    attempt_id = manifest["attempt_id"]
    context = manifest["expected_capture_context"]
    predicate = manifest["selector_dispatch_predicate"]
    sessions: set[str] = set()
    arm_envelopes: set[str] = set()
    event_envelopes: set[str] = set()
    event_sequences: set[int] = set()
    summaries: list[dict[str, Any]] = []
    for index, raw in enumerate(parity["arms"]):
        arm = exact_keys(raw, ARM_KEYS, f"parity arm {index}")
        require(arm["name"] == order[index], "parity arm name/order mismatch")
        require(
            arm["diagnostic"] is arm["name"].startswith("on-"),
            "parity diagnostic mode mismatch",
        )
        session = sha256_text(arm["session_id"], "parity session binding")
        require(session not in sessions, "parity sessions must be distinct")
        sessions.add(session)
        for phase_name, expected_kind in (
            ("first", "observed" if arm["diagnostic"] else "plain"),
            ("continuation", "observed"),
        ):
            event = exact_keys(
                arm[f"{phase_name}_event"], ARM_EVENT_KEYS, f"parity {phase_name} event"
            )
            sha256_text(event["wrapper_binding_sha256"], "event wrapper binding")
            require(
                event["kind"] == expected_kind,
                "parity event observation pattern mismatch",
            )
            if expected_kind == "plain":
                require(
                    event["library_sequence"] is None
                    and event["library_event_envelope_sha256"] is None
                    and event["session_binding_sha256"] is None
                    and event["draft_tokens"] is None,
                    "plain event has library observation identity",
                )
            else:
                sequence = integer(
                    event["library_sequence"], "library event sequence", 1
                )
                envelope = sha256_text(
                    event["library_event_envelope_sha256"], "library event envelope"
                )
                require(
                    sequence not in event_sequences and envelope not in event_envelopes,
                    "library event sequence/envelope is not globally unique",
                )
                event_sequences.add(sequence)
                event_envelopes.add(envelope)
                sha256_text(event["session_binding_sha256"], "library session binding")
                require(
                    event["session_binding_sha256"] == session,
                    "observed event session binding differs from enclosing arm",
                )
                require(
                    isinstance(event["draft_tokens"], list)
                    and len(event["draft_tokens"]) == 8,
                    "observed event draft token material invalid",
                )
                [
                    integer(
                        token, "observed event draft token", -(1 << 31), (1 << 31) - 1
                    )
                    for token in event["draft_tokens"]
                ]
            check(
                event["wrapper_binding_sha256"]
                == event_wrapper_digest(
                    attempt_id, arm["name"], session, phase_name, event
                ),
                "parity event wrapper binding mismatch",
            )
        summary = exact_keys(
            arm["summary"], ARM_SUMMARY_KEYS, f"parity summary {index}"
        )
        require(
            summary["domain"] == "qwen.dflash_k0s.parity_arm_summary.v1",
            "parity arm summary domain mismatch",
        )
        for phase_name in ("first", "continuation"):
            phase = exact_keys(
                summary[phase_name], ARM_PHASE_KEYS, f"parity {phase_name} phase"
            )
            expected_carry = (
                context["carry_token"]
                if phase_name == "first"
                else manifest["expected_continuation_carry_token"]
            )
            expected_position = (
                context["noise_start_position"]
                if phase_name == "first"
                else context["noise_start_position"] + 1
            )
            require(
                integer(
                    phase["carry_token"], f"parity {phase_name} carry", 0, VOCAB - 1
                )
                == expected_carry
                and integer(
                    phase["noise_start_position"], f"parity {phase_name} position"
                )
                == expected_position,
                f"parity {phase_name} carry/noise position mismatch",
            )
            require(
                phase["draft_tokens_count"] == 8
                and phase["full_logits_count"] == DEPTHS * VOCAB
                and phase["topk_count"] == DEPTHS * TOP_K
                and phase["unary_count"] == DEPTHS * TOP_K
                and phase["z_count"] == DEPTHS * RANK,
                f"parity {phase_name} geometry/count mismatch",
            )
            for key in (
                "target_sha256",
                "dflash_sha256",
                "state_sha256",
                "draft_tokens_sha256_i32le",
                "full_logits_sha256_f32le",
                "topk_sha256_i32le",
                "unary_sha256_f32le",
                "z_sha256_f32le",
            ):
                sha256_text(phase[key], f"parity {phase_name} {key}")
            runtime = exact_keys(
                phase["runtime_selector_contract"],
                SELECTOR_PREDICATE_KEYS,
                "runtime selector contract",
            )
            check(
                runtime == predicate, "runtime selector contract differs from predicate"
            )
            dynamic = {
                "dispatch_census": phase["dispatch_census"],
                "selector_hidden_dispatch": next(
                    (
                        row
                        for row in phase["dispatch_census"]
                        if isinstance(row, dict)
                        and row.get("tag") == SELECTOR_DISPATCH_TAG
                    ),
                    None,
                ),
                "kernel_trace": phase["kernel_trace"],
                "embedded_metallib_sha256": runtime["metallib_sha256"],
                "build": manifest["expected_build"],
                "host": manifest["expected_host"],
                "environment": manifest["selector_dispatch_predicate"][
                    "allowed_environment"
                ],
            }
            validate_provenance(dynamic, manifest)
            event = arm[f"{phase_name}_event"]
            if event["kind"] == "observed":
                check(
                    hashlib.sha256(
                        b"".join(
                            struct.pack("<i", token) for token in event["draft_tokens"]
                        )
                    ).hexdigest()
                    == phase["draft_tokens_sha256_i32le"],
                    "observed event draft digest differs from phase",
                )
                check(
                    event["library_event_envelope_sha256"]
                    == rust_library_event_envelope(event, phase),
                    "Rust library event envelope mismatch",
                )
        baseline = exact_keys(
            summary["observer_baseline"], OBSERVER_BASELINE_KEYS, "observer baseline"
        )
        sha256_text(baseline["before_sha256"], "observer before")
        sha256_text(baseline["after_sha256"], "observer after")
        check(
            baseline["restored"] is True
            and baseline["before_sha256"] == baseline["after_sha256"],
            "observer baseline was not restored",
        )
        check(
            summary["common_production_content_sha256"]
            == common_production_digest(summary["first"], summary["continuation"]),
            "common production digest mismatch",
        )
        if arm["diagnostic"]:
            sha256_text(summary["capture_content_sha256"], "capture content")
            require(
                arm["capture_projection_sha256"] == summary["capture_content_sha256"],
                "diagnostic capture association mismatch",
            )
        else:
            require(
                summary["capture_content_sha256"] is None
                and arm["capture_projection_sha256"] is None,
                "off arm claims capture content",
            )
        require(
            isinstance(arm["rng_domains"], list)
            and len(arm["rng_domains"]) == len(manifest["expected_rng_domains"]),
            "per-arm RNG domain count mismatch",
        )
        for rng_index, domain_name in enumerate(manifest["expected_rng_domains"]):
            rng = exact_keys(
                arm["rng_domains"][rng_index], RNG_DOMAIN_KEYS, "per-arm RNG domain"
            )
            require(
                rng["domain"] == domain_name
                and rng["scope"] == RNG_ABSENT_SCOPE
                and rng["before_counter"] == 0
                and rng["after_counter"] == 0,
                "per-arm RNG scope/order/counter mismatch",
            )
            check(
                rng["absent_state_sha256"]
                == rng_absent_digest(attempt_id, arm["name"], domain_name),
                "per-arm RNG absent-state digest mismatch",
            )
        envelope = sha256_text(arm["arm_envelope_sha256"], "CLI arm envelope")
        require(envelope not in arm_envelopes, "parity arm envelopes must be distinct")
        arm_envelopes.add(envelope)
        check(
            envelope == arm_envelope_digest(arm, attempt_id),
            "CLI arm envelope digest mismatch",
        )
        summaries.append(summary)
    require(
        sorted(event_sequences) == list(range(1, len(event_sequences) + 1)),
        "observed library event sequences must be globally contiguous",
    )
    for key in (
        "first",
        "continuation",
        "observer_baseline",
        "common_production_content_sha256",
    ):
        check(
            all(summary[key] == summaries[0][key] for summary in summaries[1:]),
            f"four-arm parity {key} matrix mismatch",
        )
    check(
        summaries[1]["capture_content_sha256"]
        == summaries[2]["capture_content_sha256"],
        "on-A/on-B capture content mismatch",
    )
    require(
        parity["comparison_fields"] == PARITY_COMPARISON_FIELDS,
        "parity comparison field set/order mismatch",
    )
    return parity


def validate_parity_failure(payload: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    failure = exact_keys(payload, PARITY_FAILURE_KEYS, "parity failure")
    require(
        failure["authority"] == AUTHORITY and failure["status"] == "failed",
        "parity failure authority/status mismatch",
    )
    completed = failure["completed_arms"]
    order = ["off-A", "on-A", "on-B", "off-B"]
    require(
        isinstance(completed, list) and len(completed) <= 4,
        "parity failure completed arms invalid",
    )
    require(
        isinstance(failure["observer_cleanup"], bool),
        "parity failure observer_cleanup must be boolean",
    )
    sessions: set[str] = set()
    envelopes: set[str] = set()
    sequences: set[int] = set()
    sequence_order: list[int] = []
    library_envelopes: set[str] = set()
    for index, raw in enumerate(completed):
        arm = exact_keys(raw, ARM_KEYS, f"completed parity arm {index}")
        require(
            arm["name"] == order[index]
            and arm["diagnostic"] is arm["name"].startswith("on-"),
            "completed parity arms are not the exact mode/order prefix",
        )
        session = sha256_text(arm["session_id"], "completed arm session binding")
        require(session not in sessions, "completed arm sessions are not unique")
        sessions.add(session)
        for phase_name, expected_kind in (
            ("first", "observed" if arm["diagnostic"] else "plain"),
            ("continuation", "observed"),
        ):
            event = exact_keys(
                arm[f"{phase_name}_event"], ARM_EVENT_KEYS, "completed arm event"
            )
            sha256_text(
                event["wrapper_binding_sha256"], "completed event wrapper binding"
            )
            require(
                event["kind"] == expected_kind,
                "completed arm event observation pattern mismatch",
            )
            if expected_kind == "plain":
                require(
                    event["library_sequence"] is None
                    and event["library_event_envelope_sha256"] is None
                    and event["session_binding_sha256"] is None
                    and event["draft_tokens"] is None,
                    "completed plain event has library identity",
                )
            else:
                sequence = integer(
                    event["library_sequence"], "completed event sequence", 1
                )
                library = sha256_text(
                    event["library_event_envelope_sha256"], "completed library envelope"
                )
                require(
                    sequence not in sequences and library not in library_envelopes,
                    "completed observed event identity is not unique",
                )
                sequences.add(sequence)
                sequence_order.append(sequence)
                library_envelopes.add(library)
                sha256_text(
                    event["session_binding_sha256"], "completed event session binding"
                )
                require(
                    event["session_binding_sha256"] == session,
                    "completed observed event session binding differs from arm",
                )
                require(
                    isinstance(event["draft_tokens"], list)
                    and len(event["draft_tokens"]) == 8,
                    "completed observed event draft material invalid",
                )
            require(
                event["wrapper_binding_sha256"]
                == event_wrapper_digest(
                    manifest["attempt_id"], arm["name"], session, phase_name, event
                ),
                "completed event wrapper binding mismatch",
            )
        summary = exact_keys(arm["summary"], ARM_SUMMARY_KEYS, "completed arm summary")
        require(
            summary["domain"] == "qwen.dflash_k0s.parity_arm_summary.v1",
            "completed arm summary domain mismatch",
        )
        for phase_name in ("first", "continuation"):
            phase = exact_keys(
                summary[phase_name], ARM_PHASE_KEYS, "completed arm phase"
            )
            require(
                phase["draft_tokens_count"] == 8
                and phase["full_logits_count"] == DEPTHS * VOCAB
                and phase["topk_count"] == DEPTHS * TOP_K
                and phase["unary_count"] == DEPTHS * TOP_K
                and phase["z_count"] == DEPTHS * RANK,
                "completed arm phase geometry mismatch",
            )
            expected_carry = (
                manifest["expected_capture_context"]["carry_token"]
                if phase_name == "first"
                else manifest["expected_continuation_carry_token"]
            )
            expected_position = (
                manifest["expected_capture_context"]["noise_start_position"]
                if phase_name == "first"
                else manifest["expected_capture_context"]["noise_start_position"] + 1
            )
            require(
                integer(phase["carry_token"], "completed phase carry", 0, VOCAB - 1)
                == expected_carry
                and integer(phase["noise_start_position"], "completed phase position")
                == expected_position,
                "completed arm phase static/runtime binding mismatch",
            )
            for key in (
                "target_sha256",
                "dflash_sha256",
                "state_sha256",
                "draft_tokens_sha256_i32le",
                "full_logits_sha256_f32le",
                "topk_sha256_i32le",
                "unary_sha256_f32le",
                "z_sha256_f32le",
            ):
                sha256_text(phase[key], f"completed {phase_name} {key}")
            runtime = exact_keys(
                phase["runtime_selector_contract"],
                SELECTOR_PREDICATE_KEYS,
                "completed runtime selector contract",
            )
            require(
                runtime == manifest["selector_dispatch_predicate"],
                "completed runtime selector contract differs from manifest",
            )
            dynamic = {
                "dispatch_census": phase["dispatch_census"],
                "selector_hidden_dispatch": next(
                    (
                        row
                        for row in phase["dispatch_census"]
                        if isinstance(row, dict)
                        and row.get("tag") == SELECTOR_DISPATCH_TAG
                    ),
                    None,
                ),
                "kernel_trace": phase["kernel_trace"],
                "embedded_metallib_sha256": runtime["metallib_sha256"],
                "build": manifest["expected_build"],
                "host": manifest["expected_host"],
                "environment": runtime["allowed_environment"],
            }
            try:
                validate_provenance(dynamic, manifest)
            except FailedEvidence as error:
                raise InvalidEvidence(
                    f"completed arm provenance mismatch: {error}"
                ) from error
            event = arm[f"{phase_name}_event"]
            if event["kind"] == "observed":
                for token in event["draft_tokens"]:
                    integer(
                        token,
                        "completed observed draft token",
                        -(1 << 31),
                        (1 << 31) - 1,
                    )
                require(
                    hashlib.sha256(
                        b"".join(
                            struct.pack("<i", token) for token in event["draft_tokens"]
                        )
                    ).hexdigest()
                    == phase["draft_tokens_sha256_i32le"]
                    and event["library_event_envelope_sha256"]
                    == rust_library_event_envelope(event, phase),
                    "completed observed event material/envelope mismatch",
                )
        baseline = exact_keys(
            summary["observer_baseline"],
            OBSERVER_BASELINE_KEYS,
            "completed observer baseline",
        )
        sha256_text(baseline["before_sha256"], "completed observer before")
        sha256_text(baseline["after_sha256"], "completed observer after")
        require(
            isinstance(baseline["restored"], bool),
            "completed observer restored must be boolean",
        )
        require(
            baseline["restored"] is True
            and baseline["before_sha256"] == baseline["after_sha256"],
            "completed historical arm observer baseline was not restored",
        )
        sha256_text(
            summary["common_production_content_sha256"],
            "completed common production digest",
        )
        require(
            summary["common_production_content_sha256"]
            == common_production_digest(summary["first"], summary["continuation"]),
            "completed arm common production digest mismatch",
        )
        capture_content = summary["capture_content_sha256"]
        capture_projection = arm["capture_projection_sha256"]
        if arm["diagnostic"]:
            if capture_content is None or capture_projection is None:
                require(
                    capture_content is None and capture_projection is None,
                    "completed diagnostic capture association mismatch",
                )
            else:
                require(
                    sha256_text(capture_content, "completed capture content")
                    == sha256_text(capture_projection, "completed capture projection"),
                    "completed diagnostic capture association mismatch",
                )
        else:
            require(
                capture_content is None and capture_projection is None,
                "completed off arm claims capture content",
            )
        require(
            isinstance(arm["rng_domains"], list)
            and len(arm["rng_domains"]) == len(manifest["expected_rng_domains"]),
            "completed arm RNG domains invalid",
        )
        for rng_index, domain_name in enumerate(manifest["expected_rng_domains"]):
            rng = exact_keys(
                arm["rng_domains"][rng_index],
                RNG_DOMAIN_KEYS,
                "completed arm RNG domain",
            )
            require(
                rng["domain"] == domain_name
                and rng["scope"] == RNG_ABSENT_SCOPE
                and rng["before_counter"] == 0
                and rng["after_counter"] == 0
                and rng["absent_state_sha256"]
                == rng_absent_digest(manifest["attempt_id"], arm["name"], domain_name),
                "completed arm RNG binding mismatch",
            )
            sha256_text(rng["absent_state_sha256"], "completed RNG absent-state digest")
        envelope = sha256_text(arm["arm_envelope_sha256"], "completed arm envelope")
        require(envelope not in envelopes, "completed arm envelopes are not unique")
        envelopes.add(envelope)
        require(
            envelope == arm_envelope_digest(arm, manifest["attempt_id"]),
            "completed arm envelope mismatch",
        )
    require(
        sequence_order == list(range(1, len(sequence_order) + 1)),
        "completed observed event sequences are not globally ordered/contiguous",
    )
    stage = failure["failed_stage"]
    require(stage in PARITY_FAILURE_STAGES, "parity failure stage is invalid")
    post_arm = stage in {"comparison", "serialization"}
    if post_arm:
        require(
            len(completed) == 4 and failure["failed_arm"] is None,
            "post-arm failure requires four completed arms and null failed_arm",
        )
    else:
        require(
            len(completed) < 4 and failure["failed_arm"] == order[len(completed)],
            "in-arm failure requires strict prefix and next failed arm",
        )
    text(failure["first_mismatch"], "parity failure first mismatch", maximum=1024)
    require(
        isinstance(failure["observer_cleanup"], bool),
        "parity failure observer_cleanup must be boolean",
    )
    identities = exact_keys(
        failure["identities"], PARITY_FAILURE_IDENTITY_KEYS, "parity failure identities"
    )
    expected = {
        "reducer": manifest["reducer"],
        "executable": manifest["executable"],
        "fixture": manifest["fixture"],
        "command": manifest["command"],
        "sources": manifest["sources"],
        "assets": manifest["assets"],
        "build": manifest["expected_build"],
        "host": manifest["expected_host"],
        "embedded_metallib_sha256": manifest["embedded_metallib_sha256"],
    }
    require(identities == expected, "parity failure identities differ from manifest")
    if not failure["observer_cleanup"]:
        raise FailedEvidence(
            "terminal parity failure with incomplete observer cleanup",
            {"parity_failure": failure},
        )
    return failure


def projection_without_refs(value: Any, refs: list[str]) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            if key.endswith("_range_id") and child is not None:
                refs.append(child)
                out[key] = "<range>"
            else:
                out[key] = projection_without_refs(child, refs)
        return out
    if isinstance(value, list):
        return [projection_without_refs(child, refs) for child in value]
    return value


def reduce_projection(
    label: str,
    content: dict[str, Any],
    sidecar: OpenFile,
    assets: dict[str, OpenFile],
    tensors: dict[str, dict[str, Any]],
    temperature: int,
    use_range: Any,
    semantic: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Independently replay one retained capture projection."""
    capture = validate_capture_shape(content["capture"])
    provenance = validate_provenance(content["provenance"], manifest)
    depths_raw = content["depths"]
    lattices_raw = content["lattices"]
    require(
        isinstance(depths_raw, list) and len(depths_raw) == DEPTHS,
        f"{label} depth geometry mismatch",
    )
    require(
        isinstance(lattices_raw, list) and len(lattices_raw) == DEPTHS,
        f"{label} lattice geometry mismatch",
    )
    depths: dict[int, dict[str, Any]] = {}
    support: dict[int, bool] = {}
    materials: list[tuple[bytes, list[int], list[int], list[int]]] = []
    for expected_depth, raw in enumerate(depths_raw, 1):
        depth = exact_keys(raw, DEPTH_KEYS, f"{label} depth {expected_depth}")
        require(
            depth["depth"] == expected_depth
            and depth["position"] == capture["noise_start_position"] + expected_depth,
            f"{label} depth position mismatch",
        )
        z = [bits32(value, f"{label} z", finite=True) for value in depth["z_f32_bits"]]
        require(len(z) == RANK, f"{label} z geometry mismatch")
        ids = [
            integer(value, f"{label} top16 id", -(1 << 31), (1 << 31) - 1)
            for value in depth["top16_ids"]
        ]
        unary = [bits32(value, f"{label} unary") for value in depth["unary_f32_bits"]]
        require(len(ids) == len(unary) == TOP_K, f"{label} top16 geometry mismatch")
        ref = use_range(text(depth["full_logits_range_id"], f"{label} logits range"))
        require(
            ref["kind"] == "full_logits"
            and ref["dtype"] == "F32"
            and ref["shape"] == [VOCAB]
            and ref["tensor_role"] is None
            and ref["row"] is None,
            f"{label} full-logit metadata mismatch",
        )
        raw_logits = pread_exact(sidecar, ref["offset"], ref["bytes"], ref["id"])
        logits = list(struct.unpack(f"<{VOCAB}I", raw_logits))
        derived, expected_ids = derive_topk_issues(logits, ids, unary)
        semantic(
            depth["topk_issues"] == derived, f"{label} topK issue kind/order mismatch"
        )
        semantic(not derived, f"{label} topK support undefined")
        support[expected_depth] = not derived and expected_ids is not None
        depths[expected_depth] = depth
        materials.append((raw_logits, ids, unary, z))
    semantic(
        capture["synchronized_capture_sha256"]
        == capture_digest(provenance, capture, materials, list(tensors.values())),
        f"{label} capture digest mismatch",
    )
    positional: dict[tuple[int, int], dict[str, Any]] = {}
    sparse_q: list[dict[str, Any]] = []
    for expected_depth, lattice_raw in enumerate(lattices_raw, 1):
        lattice = exact_keys(lattice_raw, LATTICE_KEYS, f"{label} lattice")
        require(lattice["depth"] == expected_depth, f"{label} lattice depth mismatch")
        expected_rows = 1 if expected_depth == 1 else TOP_K
        require(
            isinstance(lattice["rows"], list) and len(lattice["rows"]) == expected_rows,
            f"{label} lattice row geometry mismatch",
        )
        for predecessor_slot, row_raw in enumerate(lattice["rows"]):
            row = exact_keys(row_raw, ROW_KEYS, f"{label} row")
            slot_key = -1 if expected_depth == 1 else predecessor_slot
            global_index = (
                0
                if expected_depth == 1
                else 1 + (expected_depth - 2) * TOP_K + predecessor_slot
            )
            require(
                row["row_index"] == global_index
                and row["predecessor_slot"] == (None if slot_key < 0 else slot_key),
                f"{label} global row index mismatch",
            )
            predecessor = integer(
                row["predecessor_token"],
                f"{label} predecessor",
                -(1 << 31),
                (1 << 31) - 1,
            )
            expected_predecessor = (
                content["production_chain"]["initial_carry"]
                if expected_depth == 1
                else depths[expected_depth - 1]["top16_ids"][predecessor_slot]
            )
            semantic(
                predecessor == expected_predecessor,
                f"{label} predecessor mapping mismatch",
            )
            a: list[int] = []
            if 0 <= predecessor < VOCAB:
                ref = use_range(
                    text(row["predecessor_raw_range_id"], f"{label} predecessor range")
                )
                tensor = tensors["predecessor"]
                require(
                    ref["kind"] == "predecessor_row"
                    and ref["dtype"] == tensor["dtype"]
                    and ref["shape"] == [RANK]
                    and ref["tensor_role"] == "predecessor"
                    and ref["row"] == predecessor,
                    f"{label} predecessor range metadata mismatch",
                )
                raw_a = pread_exact(sidecar, ref["offset"], ref["bytes"], ref["id"])
                semantic(
                    raw_a
                    == row_bytes_from_asset(
                        "predecessor", predecessor, tensors, assets
                    ),
                    f"{label} predecessor asset row mismatch",
                )
                a = decode_raw(ref["dtype"], raw_a, RANK)
            else:
                require(
                    row["predecessor_raw_range_id"] is None,
                    f"{label} sentinel predecessor has raw range",
                )
            require(
                isinstance(row["slots"], list) and len(row["slots"]) == TOP_K,
                f"{label} slot geometry mismatch",
            )
            tokens: list[int] = []
            scores: list[int | None] = []
            slot_items = []
            for expected_slot, slot_raw in enumerate(row["slots"]):
                slot = exact_keys(slot_raw, SLOT_KEYS, f"{label} slot")
                require(slot["slot"] == expected_slot, f"{label} slot index mismatch")
                token = integer(
                    slot["token"], f"{label} token", -(1 << 31), (1 << 31) - 1
                )
                tokens.append(token)
                semantic(
                    token == depths[expected_depth]["top16_ids"][expected_slot]
                    and slot["unary_f32_bits"]
                    == depths[expected_depth]["unary_f32_bits"][expected_slot],
                    f"{label} slot token/unary binding mismatch",
                )
                unary = bits32(slot["unary_f32_bits"], f"{label} slot unary")
                if 0 <= token < VOCAB and a:
                    ref = use_range(
                        text(slot["successor_raw_range_id"], f"{label} successor range")
                    )
                    tensor = tensors["successor"]
                    require(
                        ref["kind"] == "successor_row"
                        and ref["dtype"] == tensor["dtype"]
                        and ref["shape"] == [RANK]
                        and ref["tensor_role"] == "successor"
                        and ref["row"] == token,
                        f"{label} successor range metadata mismatch",
                    )
                    raw_b = pread_exact(sidecar, ref["offset"], ref["bytes"], ref["id"])
                    semantic(
                        raw_b
                        == row_bytes_from_asset("successor", token, tensors, assets),
                        f"{label} successor asset row mismatch",
                    )
                    score = bits32(slot["score_f32_bits"], f"{label} score")
                    replay = replay_score(
                        a,
                        materials[expected_depth - 1][3],
                        decode_raw(ref["dtype"], raw_b, RANK),
                        unary,
                    )
                    semantic(
                        score == replay
                        if classify(replay) == "finite"
                        else classify(score) == classify(replay),
                        f"{label} score replay mismatch",
                    )
                    scores.append(score)
                else:
                    require(
                        slot["successor_raw_range_id"] is None
                        and slot["score_f32_bits"] is None,
                        f"{label} invalid edge carries score/range",
                    )
                    scores.append(None)
                slot_items.append(slot)
            derived_issues, choice = expected_issues(tokens, scores)
            for slot, expected in zip(slot_items, derived_issues):
                semantic(slot["issues"] == expected, f"{label} slot issue mismatch")
            row_issues = [] if choice is not None else [{"kind": "no_valid_choice"}]
            semantic(
                row["issues"] == row_issues
                and row["choice_slot"] == (choice if choice is not None else 0),
                f"{label} row issue/choice mismatch",
            )
            if support[expected_depth] and not row_issues and not any(derived_issues):
                probabilities = strict_softmax(
                    [value for value in scores if value is not None], temperature
                )
                sparse_q.append(
                    {
                        "row_index": global_index,
                        "support": [
                            {
                                "token": token,
                                "q_f64_bits": f"0x{struct.unpack('>Q', struct.pack('>d', q))[0]:016x}",
                            }
                            for token, q in zip(tokens, probabilities)
                        ],
                        "outside_support": "exact_zero",
                    }
                )
            positional[(expected_depth, slot_key)] = row
    validate_chain(
        content["production_chain"],
        f"{label} production chain",
        positional,
        True,
        semantic,
    )
    for index, chain in enumerate(content["fixed_chains"]):
        validate_chain(
            chain, f"{label} fixed chain {index}", positional, False, semantic
        )
    semantic(len(sparse_q) == LATTICE_ROWS, f"{label} requires 97 clean q rows")
    refs: list[str] = []
    normalized = projection_without_refs(content, refs)
    return {
        "content_sha256": canonical_json_digest(normalized),
        "sparse_q": sparse_q,
        "range_normalization_sha256": canonical_json_digest(refs),
        "phase_binding": retained_phase_binding(capture, materials, provenance),
    }


def retained_phase_binding(
    capture: dict[str, Any],
    materials: list[tuple[bytes, list[int], list[int], list[int]]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "carry_token": capture["carry_token"],
        "noise_start_position": capture["noise_start_position"],
        "dflash_sha256": capture["state"]["diagnostic_state_sha256"],
        "draft_tokens_count": len(capture["draft_tokens"]),
        "draft_tokens_sha256_i32le": capture["draft_tokens_sha256_i32le"],
        "full_logits_count": sum(len(raw) // 4 for raw, _, _, _ in materials),
        "full_logits_sha256_f32le": rust_vector_hash(
            "qwen.dflash_k0s.full_logits.f32le.v1",
            [
                value
                for raw, _, _, _ in materials
                for value in struct.unpack(f"<{len(raw) // 4}I", raw)
            ],
        ),
        "topk_count": sum(len(ids) for _, ids, _, _ in materials),
        "topk_sha256_i32le": rust_vector_hash(
            "qwen.dflash_k0s.top_k_ids.i32le.v1",
            [token for _, ids, _, _ in materials for token in ids],
            signed=True,
        ),
        "unary_count": sum(len(unary) for _, _, unary, _ in materials),
        "unary_sha256_f32le": rust_vector_hash(
            "qwen.dflash_k0s.unary.f32le.v1",
            [raw for _, _, unary, _ in materials for raw in unary],
        ),
        "z_count": sum(len(z) for _, _, _, z in materials),
        "z_sha256_f32le": rust_vector_hash(
            "qwen.dflash_k0s.selector_hidden.f32le.v1",
            [raw for _, _, _, z in materials for raw in z],
        ),
        "dispatch_census": provenance["dispatch_census"],
        "kernel_trace": provenance["kernel_trace"],
    }


def validate_packet(
    rows: list[dict[str, Any]],
    sidecar: OpenFile,
    manifest: dict[str, Any],
    assets: dict[str, OpenFile],
    tensors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    semantic_failures: list[str] = []

    def semantic(condition: bool, message: str) -> None:
        if not condition and len(semantic_failures) < 256:
            semantic_failures.append(message)

    run = exact_keys(rows[0]["payload"], RUN_KEYS, "run payload")
    require(run["authority"] == AUTHORITY, "authority label mismatch")
    require(
        run["attempt_id"] == manifest["attempt_id"]
        and all(row["attempt_id"] == manifest["attempt_id"] for row in rows),
        "trace/run attempt identity mismatch",
    )
    parity = validate_parity(run["diagnostic_nonperturbation_parity"], manifest)
    geometry = exact_keys(run["geometry"], GEOMETRY_KEYS, "geometry")
    for key, value in geometry.items():
        integer(value, f"geometry {key}", 1)
    require(
        tuple(geometry.values())
        == (8, DEPTHS, TOP_K, RANK, HIDDEN, VOCAB, LATTICE_ROWS),
        "geometry contract mismatch",
    )
    request = exact_keys(run["request"], REQUEST_KEYS, "request")
    check(
        {"request": request, "ignored_target_policy": run["ignored_target_policy"]}
        == manifest["expected_request"],
        "request temperature/ignored policy differs from manifest",
    )
    temperature = bits32(
        request["temperature_f32_bits"], "request.temperature", finite=True
    )
    require(f32_value(temperature) > 0.0, "request temperature must be positive")
    require(
        exact_keys(run["proposal_abstention"], ABSTENTION_KEYS, "proposal_abstention")
        == {"enabled": False, "p_min": None, "n_min": None},
        "proposal abstention must be exactly disabled",
    )
    policies = exact_keys(
        run["ignored_target_policy"],
        ("variant_a", "variant_b"),
        "ignored_target_policy",
    )
    for variant, policy in policies.items():
        exact_keys(policy, IGNORED_POLICY_KEYS, f"ignored_target_policy.{variant}")
        integer(policy["top_k"], "ignored target top_k", 0)
        bits32(policy["top_p_f32_bits"], "ignored target top_p", finite=True)
        bits32(policy["min_p_f32_bits"], "ignored target min_p", finite=True)
        require(
            policy["grammar"] is None and policy["penalties"] is None,
            "ignored target policy is metadata only",
        )
    require(
        policies["variant_a"] != policies["variant_b"],
        "ignored policy variants must differ",
    )
    binding = exact_keys(run["binding"], BINDING_KEYS, "binding")
    text(binding["production_call_id"], "production call id", maximum=128)
    sha256_text(binding["drafter_checkpoint_sha256"], "drafter checkpoint")
    text(binding["proposal_construction_id"], "proposal construction id", maximum=128)
    sha256_text(binding["noise_input_sha256"], "noise input")
    check(
        binding == manifest["expected_binding"], "capture binding differs from manifest"
    )
    provenance = validate_provenance(run["provenance"], manifest)
    capture = validate_capture_shape(run["capture"])
    state = capture["state"]
    actual_context = {
        "definition_version": capture["definition_version"],
        "carry_token": capture["carry_token"],
        "noise_start_position": capture["noise_start_position"],
        "target_context_len": state["target_context_len"],
        "context_hidden_watermark": state["context_hidden_watermark"],
        "kv_context_watermark": state["kv_context_watermark"],
    }
    check(
        actual_context == manifest["expected_capture_context"],
        "capture context differs from static manifest",
    )
    check(
        capture["state"]["noise_input_sha256"] == binding["noise_input_sha256"],
        "capture state noise identity differs from binding",
    )
    require(
        run["semantic_references"]
        == SEMANTIC_REFERENCES
        == manifest["semantic_references"],
        "semantic reference identity mismatch",
    )
    require(
        run["tensors"] == manifest["tensors"],
        "trace tensor claims are not manifest-bound",
    )
    production_chain = exact_keys(
        run["production_chain"], CHAIN_KEYS, "production_chain"
    )
    require(isinstance(run["fixed_chains"], list), "fixed chains must be a list")
    static_chains = [
        {key: chain[key] for key in STATIC_CHAIN_KEYS}
        for chain in run["fixed_chains"]
        if isinstance(chain, dict) and all(key in chain for key in STATIC_CHAIN_KEYS)
    ]
    check(
        static_chains == manifest["expected_fixed_chains"],
        "fixed slot chains differ from static manifest",
    )
    registry = sidecar_registry(run["sidecar_registry"], sidecar)
    used_ranges: set[str] = set()

    def use_range(identifier: str) -> dict[str, Any]:
        require(identifier in registry, f"unknown sidecar range id {identifier!r}")
        require(
            identifier not in used_ranges,
            f"aliased sidecar range reference {identifier!r}",
        )
        used_ranges.add(identifier)
        return registry[identifier]

    tensor_by_role = {item["role"]: item for item in run["tensors"]}
    depth_rows = [rows[1 + depth * 2] for depth in range(DEPTHS)]
    lattice_rows = [rows[2 + depth * 2] for depth in range(DEPTHS)]
    depths: dict[int, dict[str, Any]] = {}
    depth_support_defined: dict[int, bool] = {}
    capture_materials: list[tuple[bytes, list[int], list[int], list[int]]] = []
    for expected_depth, record in enumerate(depth_rows, 1):
        item = exact_keys(record["payload"], DEPTH_KEYS, f"depth {expected_depth}")
        require(
            integer(item["depth"], "depth", 1, DEPTHS) == expected_depth
            and isinstance(item["position"], int)
            and not isinstance(item["position"], bool)
            and item["position"] == capture["noise_start_position"] + expected_depth,
            "depth number/position invalid",
        )
        for key in BINDING_KEYS:
            require(
                item[key] == binding[key],
                f"depth {expected_depth} association mismatch",
            )
        require(
            item["synchronized_capture_sha256"]
            == capture["synchronized_capture_sha256"]
            and item["draft_tokens_sha256_i32le"]
            == capture["draft_tokens_sha256_i32le"]
            and item["diagnostic_state_sha256"]
            == capture["state"]["diagnostic_state_sha256"],
            f"depth {expected_depth} output/state capture binding mismatch",
        )
        require(
            isinstance(item["z_f32_bits"], list) and len(item["z_f32_bits"]) == RANK,
            "z geometry invalid",
        )
        z_values = [
            bits32(v, f"depth {expected_depth} z", finite=True)
            for v in item["z_f32_bits"]
        ]
        range_id = text(
            item["full_logits_range_id"], "full logits range id", maximum=128
        )
        logits_ref = use_range(range_id)
        require(
            logits_ref["kind"] == "full_logits",
            "full logits range reference invalid",
        )
        raw_logits = pread_exact(
            sidecar, logits_ref["offset"], logits_ref["bytes"], range_id
        )
        logits = list(struct.unpack(f"<{VOCAB}I", raw_logits))
        require(
            isinstance(item["top16_ids"], list) and len(item["top16_ids"]) == TOP_K,
            "top16 ids geometry invalid",
        )
        ids = [
            integer(v, "top16 token", -(1 << 31), (1 << 31) - 1)
            for v in item["top16_ids"]
        ]
        require(
            isinstance(item["unary_f32_bits"], list)
            and len(item["unary_f32_bits"]) == TOP_K,
            "unary geometry invalid",
        )
        unary = [bits32(v, "unary bits") for v in item["unary_f32_bits"]]
        require(
            isinstance(item["topk_issues"], list),
            f"depth {expected_depth} topk_issues must be a list",
        )
        for issue_index, issue in enumerate(item["topk_issues"]):
            require(isinstance(issue, dict), "topK issue must be an object")
            kind = issue.get("kind")
            schemas = {
                "nonfinite_logit": ("kind", "token", "bits"),
                "id_mismatch": ("kind", "slot", "expected", "observed"),
                "unary_mismatch": (
                    "kind",
                    "slot",
                    "expected_bits",
                    "observed_bits",
                ),
            }
            require(kind in schemas, "unknown topK issue kind")
            exact_keys(
                issue,
                schemas[kind],
                f"depth {expected_depth} topK issue {issue_index}",
            )
            if kind == "nonfinite_logit":
                integer(issue["token"], "topK issue token", 0, VOCAB - 1)
                bits32(issue["bits"], "topK nonfinite bits")
            elif kind == "id_mismatch":
                integer(issue["slot"], "topK issue slot", 0, TOP_K - 1)
                integer(issue["expected"], "topK expected token", -MAX_JSON_INTEGER)
                integer(issue["observed"], "topK observed token", -MAX_JSON_INTEGER)
            else:
                integer(issue["slot"], "topK issue slot", 0, TOP_K - 1)
                bits32(issue["expected_bits"], "topK expected unary bits")
                bits32(issue["observed_bits"], "topK observed unary bits")
        derived_topk, expected_ids = derive_topk_issues(logits, ids, unary)
        semantic(
            item["topk_issues"] == derived_topk,
            f"depth {expected_depth} topK issue kind/order parity mismatch",
        )
        semantic(
            not derived_topk,
            f"depth {expected_depth} has topK issues and undefined support",
        )
        depth_support_defined[expected_depth] = (
            expected_ids is not None and not derived_topk and not item["topk_issues"]
        )
        capture_materials.append((raw_logits, ids, unary, z_values))
        depths[expected_depth] = item
    computed_capture = capture_digest(
        provenance, capture, capture_materials, run["tensors"]
    )
    computed_event = synchronized_event_digest(capture, capture_materials)
    check(
        capture["state"]["synchronized_event_sha256"] == computed_event,
        "synchronized event digest mismatch",
    )
    check(
        capture["synchronized_capture_sha256"] == computed_capture,
        "synchronized capture digest mismatch",
    )
    positional: dict[tuple[int, int], dict[str, Any]] = {}
    sparse_q_rows: list[dict[str, Any]] = []
    for expected_depth, record in enumerate(lattice_rows, 1):
        lattice = exact_keys(
            record["payload"], LATTICE_KEYS, f"lattice {expected_depth}"
        )
        require(
            integer(lattice["depth"], "lattice depth", 1, DEPTHS) == expected_depth,
            "lattice depth mismatch",
        )
        expected_count = 1 if expected_depth == 1 else TOP_K
        require(
            isinstance(lattice["rows"], list)
            and len(lattice["rows"]) == expected_count,
            "lattice positional row count mismatch",
        )
        for expected_index, raw_row in enumerate(lattice["rows"]):
            row = exact_keys(
                raw_row, ROW_KEYS, f"lattice {expected_depth} row {expected_index}"
            )
            predecessor_slot = -1 if expected_depth == 1 else expected_index
            global_row_index = (
                0
                if expected_depth == 1
                else 1 + (expected_depth - 2) * TOP_K + expected_index
            )
            require(
                integer(row["row_index"], "global row index", 0, LATTICE_ROWS - 1)
                == global_row_index
                and row["predecessor_slot"]
                == (None if predecessor_slot < 0 else predecessor_slot),
                "positional row index mismatch",
            )
            if row["predecessor_slot"] is not None:
                integer(row["predecessor_slot"], "predecessor slot", 0, TOP_K - 1)
            predecessor = integer(
                row["predecessor_token"],
                "predecessor token",
                -MAX_JSON_INTEGER,
                MAX_JSON_INTEGER,
            )
            expected_predecessor = (
                production_chain["initial_carry"]
                if expected_depth == 1
                else depths[expected_depth - 1]["top16_ids"][expected_index]
            )
            check(
                predecessor == expected_predecessor,
                "positional predecessor token mapping mismatch",
            )
            if 0 <= predecessor < VOCAB:
                range_id = text(
                    row["predecessor_raw_range_id"],
                    "predecessor raw range",
                    maximum=128,
                )
                ref = use_range(range_id)
                require(
                    ref["kind"] == "predecessor_row"
                    and ref["row"] == predecessor
                    and ref["dtype"] == tensor_by_role["predecessor"]["dtype"],
                    "predecessor raw metadata mismatch",
                )
                raw_a = pread_exact(sidecar, ref["offset"], ref["bytes"], range_id)
                check(
                    raw_a
                    == row_bytes_from_asset(
                        "predecessor", predecessor, tensors, assets
                    ),
                    "predecessor raw row differs from GGUF asset",
                )
                a = decode_raw(ref["dtype"], raw_a, RANK)
            else:
                require(
                    row["predecessor_raw_range_id"] is None,
                    "invalid predecessor cannot have raw range",
                )
                a = []
            require(
                isinstance(row["slots"], list) and len(row["slots"]) == TOP_K,
                "slot geometry mismatch",
            )
            scores: list[int | None] = []
            slot_items: list[dict[str, Any]] = []
            tokens: list[int] = []
            z = [
                bits32(v, "z", finite=True)
                for v in depths[expected_depth]["z_f32_bits"]
            ]
            for expected_slot, raw_slot in enumerate(row["slots"]):
                slot = exact_keys(raw_slot, SLOT_KEYS, f"row slot {expected_slot}")
                require(
                    integer(slot["slot"], "slot index", 0, TOP_K - 1) == expected_slot,
                    "slot index mismatch",
                )
                token = integer(
                    slot["token"],
                    "successor token",
                    -MAX_JSON_INTEGER,
                    MAX_JSON_INTEGER,
                )
                tokens.append(token)
                check(
                    token == depths[expected_depth]["top16_ids"][expected_slot],
                    "slot token differs from depth top16_ids slot",
                )
                unary = bits32(slot["unary_f32_bits"], "slot unary")
                check(
                    unary
                    == bits32(
                        depths[expected_depth]["unary_f32_bits"][expected_slot],
                        "depth unary",
                    ),
                    "slot unary differs from depth unary",
                )
                if 0 <= token < VOCAB and a:
                    range_id = text(
                        slot["successor_raw_range_id"],
                        "successor raw range",
                        maximum=128,
                    )
                    ref = use_range(range_id)
                    require(
                        ref["kind"] == "successor_row"
                        and ref["row"] == token
                        and ref["dtype"] == tensor_by_role["successor"]["dtype"],
                        "successor raw metadata mismatch",
                    )
                    raw_b = pread_exact(sidecar, ref["offset"], ref["bytes"], range_id)
                    check(
                        raw_b
                        == row_bytes_from_asset("successor", token, tensors, assets),
                        "successor raw row differs from GGUF asset",
                    )
                    b = decode_raw(ref["dtype"], raw_b, RANK)
                    local = bits32(slot["score_f32_bits"], "local score")
                    replay = replay_score(a, z, b, unary)
                    if classify(replay) == "finite":
                        check(
                            local == replay, "finite scalar score-bit replay mismatch"
                        )
                    else:
                        check(
                            classify(local) == classify(replay),
                            "nonfinite scalar score classification mismatch",
                        )
                    scores.append(local)
                else:
                    require(
                        slot["successor_raw_range_id"] is None
                        and slot["score_f32_bits"] is None,
                        "invalid edge must not have raw row/score",
                    )
                    scores.append(None)
                require(isinstance(slot["issues"], list), "slot issues must be a list")
                slot_items.append(slot)
            derived_slot_issues, choice = expected_issues(tokens, scores)
            for slot, expected in zip(slot_items, derived_slot_issues):
                for issue in slot["issues"]:
                    kind = issue.get("kind") if isinstance(issue, dict) else None
                    issue_keys = {
                        "duplicate_id": ("kind", "slot", "token", "first_slot"),
                        "sentinel": ("kind", "slot", "token"),
                        "nonfinite_score": (
                            "kind",
                            "slot",
                            "token",
                            "classification",
                        ),
                    }
                    require(kind in issue_keys, "unknown slot issue kind")
                    exact_keys(issue, issue_keys[kind], "slot issue")
                semantic(
                    slot["issues"] == expected,
                    "slot issue kind/order mismatch",
                )
            row_issues = [] if choice is not None else [{"kind": "no_valid_choice"}]
            for issue in row["issues"]:
                exact_keys(issue, ("kind",), "row issue")
                require(issue["kind"] == "no_valid_choice", "unknown row issue kind")
            recorded_choice = choice if choice is not None else 0
            integer(row["choice_slot"], "row choice slot", 0, TOP_K - 1)
            semantic(
                row["issues"] == row_issues and row["choice_slot"] == recorded_choice,
                "row choice/issues mismatch",
            )
            if (
                depth_support_defined[expected_depth]
                and not row_issues
                and not any(derived_slot_issues)
            ):
                probabilities = strict_softmax(
                    [score for score in scores if score is not None], temperature
                )
                support_tokens = [
                    token for token, score in zip(tokens, scores) if score is not None
                ]
                check(
                    len(support_tokens) == TOP_K and len(set(support_tokens)) == TOP_K,
                    "clean sparse q support is not exactly the top16 token set",
                )
                sparse_q_rows.append(
                    {
                        "row_index": row["row_index"],
                        "support": [
                            {
                                "token": token,
                                "q_f64_bits": f"0x{struct.unpack('>Q', struct.pack('>d', q))[0]:016x}",
                            }
                            for token, q in zip(support_tokens, probabilities)
                        ],
                        "outside_support": "exact_zero",
                    }
                )
            positional[(expected_depth, predecessor_slot)] = row
    on_b = exact_keys(run["on_b_projection"], ON_B_KEYS, "on_b_projection")
    require(
        on_b["exclusion_allowlist"] == PROJECTION_EXCLUSION_ALLOWLIST
        and on_b["exclusion_content_sha256"]
        == canonical_json_digest(PROJECTION_EXCLUSION_ALLOWLIST),
        "on-B projection exclusion allowlist/digest mismatch",
    )
    require(
        isinstance(on_b["depths"], list)
        and len(on_b["depths"]) == DEPTHS
        and isinstance(on_b["lattices"], list)
        and len(on_b["lattices"]) == DEPTHS,
        "on-B projection depth/lattice geometry mismatch",
    )
    on_a_projection = {
        "depths": [record["payload"] for record in depth_rows],
        "lattices": [record["payload"] for record in lattice_rows],
        "capture": capture,
        "production_chain": production_chain,
        "fixed_chains": run["fixed_chains"],
        "provenance": provenance,
    }
    on_b_content = {key: on_b[key] for key in on_a_projection}
    on_a_refs: list[str] = []
    on_b_refs: list[str] = []
    normalized_a = projection_without_refs(on_a_projection, on_a_refs)
    normalized_b = projection_without_refs(on_b_content, on_b_refs)
    semantic(
        normalized_b == normalized_a,
        "on-B canonical projection content differs from on-A",
    )
    semantic(len(on_b_refs) == len(on_a_refs), "on-B projection range count mismatch")
    semantic(
        all(right == f"on-b/{left}" for left, right in zip(on_a_refs, on_b_refs)),
        "on-B range-reference normalization contract mismatch",
    )
    on_b_result = reduce_projection(
        "on-B",
        on_b_content,
        sidecar,
        assets,
        tensors,
        temperature,
        use_range,
        semantic,
        manifest,
    )
    on_a_result = {
        "content_sha256": canonical_json_digest(normalized_a),
        "sparse_q": sparse_q_rows,
        "phase_binding": retained_phase_binding(capture, capture_materials, provenance),
    }
    semantic(
        on_b_result["content_sha256"] == on_a_result["content_sha256"]
        and on_b_result["sparse_q"] == on_a_result["sparse_q"],
        "independently reduced on-A/on-B semantic result mismatch",
    )
    for arm_index, reduced in ((1, on_a_result), (2, on_b_result)):
        recorded_phase = parity["arms"][arm_index]["summary"]["first"]
        for key, derived in reduced["phase_binding"].items():
            semantic(
                recorded_phase[key] == derived,
                f"{parity['arms'][arm_index]['name']} retained first-phase {key} mismatch",
            )
    projection_digest = on_b_result["content_sha256"]
    semantic(
        on_b["projection_sha256"] == projection_digest,
        "on-B canonical projection digest mismatch",
    )
    semantic(
        parity["arms"][1]["summary"]["capture_content_sha256"] == projection_digest
        and parity["arms"][2]["summary"]["capture_content_sha256"] == projection_digest,
        "on-arm capture-content digest differs from independently reduced projections",
    )
    require(
        len(positional) == LATTICE_ROWS,
        "normal packet requires exactly 97 positional rows",
    )
    require(
        used_ranges == set(registry),
        "sidecar registry contains an unreferenced range",
    )
    validate_chain(
        production_chain,
        "production_chain",
        positional,
        True,
        semantic,
    )
    semantic(
        production_chain["initial_carry"] == capture["carry_token"]
        and production_chain["tokens"] == capture["draft_tokens"][1:],
        "production chain differs from synchronized draft output/carry",
    )
    require(
        isinstance(run["fixed_chains"], list) and run["fixed_chains"],
        "at least one fixed chain is required",
    )
    for index, chain in enumerate(run["fixed_chains"]):
        validate_chain(
            chain,
            f"fixed_chains[{index}]",
            positional,
            False,
            semantic,
        )
    semantic(
        len(sparse_q_rows) == LATTICE_ROWS,
        "passing requires exactly 97 issue-free sparse-q rows",
    )
    end = exact_keys(rows[-1]["payload"], END_KEYS, "end payload")
    require(
        end == {"producer_status": "complete", "authority": AUTHORITY},
        "end payload mismatch",
    )
    metrics = {
        "rows": LATTICE_ROWS,
        "q_defined_rows": len(sparse_q_rows),
        "depths": DEPTHS,
        "sparse_q": sparse_q_rows,
        "dynamic_digests": {
            "synchronized_capture_sha256": capture["synchronized_capture_sha256"],
            "draft_tokens_sha256_i32le": capture["draft_tokens_sha256_i32le"],
            "noise_input_sha256": capture["state"]["noise_input_sha256"],
            "synchronized_event_sha256": capture["state"]["synchronized_event_sha256"],
            "diagnostic_state_sha256": capture["state"]["diagnostic_state_sha256"],
            "dispatch_census_sha256_canonical_json": hashlib.sha256(
                json.dumps(
                    provenance["dispatch_census"],
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
            "selector_hidden_dispatch_sha256_canonical_json": hashlib.sha256(
                json.dumps(
                    provenance["selector_hidden_dispatch"],
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
            "kernel_trace_sha256_canonical_json": hashlib.sha256(
                json.dumps(
                    provenance["kernel_trace"],
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
        },
    }
    if semantic_failures:
        raise FailedEvidence(
            f"{len(semantic_failures)} semantic failure(s): "
            + " | ".join(semantic_failures),
            metrics,
        )
    return metrics


def validate_manifest(raw: Any) -> dict[str, Any]:
    manifest = exact_keys(raw, MANIFEST_KEYS, "manifest")
    require(
        manifest["schema"] == MANIFEST_SCHEMA
        and integer(manifest["schema_version"], "manifest schema_version", 1, 1)
        == MANIFEST_VERSION,
        "manifest schema/version mismatch",
    )
    text(manifest["run_id"], "manifest run_id", maximum=128)
    text(manifest["attempt_id"], "manifest attempt_id", maximum=128)
    integer(manifest["trace_max_bytes"], "manifest trace_max_bytes", 1, MAX_TRACE_BYTES)
    integer(
        manifest["sidecar_max_bytes"],
        "manifest sidecar_max_bytes",
        1,
        MAX_SIDECAR_BYTES,
    )
    for name in ("reducer", "executable", "fixture", "command"):
        identity_from_claim(manifest[name], f"manifest.{name}")
    for name in ("sources", "assets"):
        require(
            isinstance(manifest[name], list) and manifest[name],
            f"manifest.{name} must be nonempty",
        )
        roles: set[str] = set()
        for index, raw_item in enumerate(manifest[name]):
            item = exact_keys(
                raw_item, ("role",) + FILE_CLAIM_KEYS, f"manifest.{name}[{index}]"
            )
            role = text(item["role"], f"manifest.{name}.role", maximum=64)
            require(role not in roles, f"duplicate manifest {name} role")
            roles.add(role)
            identity_from_claim(
                {key: item[key] for key in FILE_CLAIM_KEYS}, f"manifest.{name}[{index}]"
            )
        expected_roles = (
            REQUIRED_SOURCE_ROLES if name == "sources" else REQUIRED_ASSET_ROLES
        )
        require(roles == expected_roles, f"manifest {name} role set mismatch")
    references = exact_keys(
        manifest["semantic_references"],
        tuple(SEMANTIC_REFERENCES),
        "manifest.semantic_references",
    )
    for role, reference in references.items():
        exact_keys(reference, ("commit", "sha256"), f"semantic reference {role}")
        text(reference["commit"], f"semantic reference {role} commit", maximum=40)
        sha256_text(reference["sha256"], f"semantic reference {role} sha256")
    require(references == SEMANTIC_REFERENCES, "manifest semantic references mismatch")
    require(
        isinstance(manifest["tensors"], list) and len(manifest["tensors"]) == 3,
        "manifest tensors invalid",
    )
    expected_request = exact_keys(
        manifest["expected_request"],
        ("request", "ignored_target_policy"),
        "manifest.expected_request",
    )
    exact_keys(expected_request["request"], REQUEST_KEYS, "manifest expected request")
    exact_keys(
        manifest["expected_prompt"], INVENTORY_PROMPT_KEYS, "manifest expected prompt"
    )
    exact_keys(manifest["expected_binding"], BINDING_KEYS, "manifest expected binding")
    integer(
        manifest["expected_continuation_carry_token"],
        "manifest continuation carry token",
        0,
        VOCAB - 1,
    )
    require(
        isinstance(manifest["expected_rng_domains"], list)
        and manifest["expected_rng_domains"]
        and len(manifest["expected_rng_domains"]) <= 32,
        "manifest expected RNG domains invalid",
    )
    rng_names = [
        text(value, "manifest RNG domain", maximum=128)
        for value in manifest["expected_rng_domains"]
    ]
    require(len(rng_names) == len(set(rng_names)), "duplicate manifest RNG domain")
    require(
        isinstance(manifest["expected_fixed_chains"], list)
        and manifest["expected_fixed_chains"],
        "manifest fixed chains invalid",
    )
    for index, chain in enumerate(manifest["expected_fixed_chains"]):
        exact_keys(chain, STATIC_CHAIN_KEYS, f"manifest fixed chain {index}")
    context = exact_keys(
        manifest["expected_capture_context"],
        CAPTURE_CONTEXT_KEYS,
        "manifest expected capture context",
    )
    require(
        context["definition_version"] == CAPTURE_DEFINITION,
        "manifest capture definition mismatch",
    )
    integer(context["carry_token"], "manifest carry", 0, VOCAB - 1)
    integer(
        context["noise_start_position"],
        "manifest noise start",
        0,
        (1 << 32) - 1 - DEPTHS,
    )
    for key in CAPTURE_CONTEXT_KEYS[3:]:
        integer(context[key], f"manifest capture context {key}")
    require(
        context["target_context_len"]
        == context["context_hidden_watermark"]
        == context["kv_context_watermark"]
        and context["noise_start_position"] == context["target_context_len"],
        "manifest capture context/watermarks/noise position invariant mismatch",
    )
    exact_keys(manifest["expected_build"], BUILD_KEYS, "manifest expected build")
    exact_keys(manifest["expected_host"], HOST_KEYS, "manifest expected host")
    sha256_text(manifest["embedded_metallib_sha256"], "manifest metallib")
    predicate = exact_keys(
        manifest["selector_dispatch_predicate"],
        SELECTOR_PREDICATE_KEYS,
        "manifest selector dispatch predicate",
    )
    require(
        predicate["tag"] == SELECTOR_DISPATCH_TAG,
        "manifest selector dispatch tag mismatch",
    )
    for key in ("kernel", "weight_dtype", "input_dtype", "output_dtype"):
        text(predicate[key], f"selector predicate {key}", maximum=256)
    require(
        predicate["weight_dtype"] == "Q4_K"
        and predicate["input_dtype"] == "F32"
        and predicate["output_dtype"] == "F32"
        and predicate["weight_dtype_id"] == 12
        and predicate["n"] == 8
        and predicate["h"] == HIDDEN
        and predicate["r"] == RANK,
        "selector dispatch dtype/geometry predicate mismatch",
    )
    require(
        predicate["kernel"] == SELECTOR_KERNEL,
        "K0-S bridge v1 selector kernel must be Q4_K",
    )
    require(
        GGML_RUNTIME_DTYPE_IDS.get(predicate["weight_dtype"])
        == predicate["weight_dtype_id"]
        and GGML_RUNTIME_DTYPE_IDS.get(predicate["input_dtype"])
        == predicate["input_dtype_id"]
        and GGML_RUNTIME_DTYPE_IDS.get(predicate["output_dtype"])
        == predicate["output_dtype_id"],
        "selector predicate dtype numeric IDs mismatch",
    )
    tensor_by_role = {
        item["role"]: item for item in manifest["tensors"] if isinstance(item, dict)
    }
    source_by_role = {
        item["role"]: item for item in manifest["sources"] if isinstance(item, dict)
    }
    require(
        predicate["weight_dtype"] == tensor_by_role["selector_hidden"]["dtype"]
        and predicate["metal_source_sha256"]
        == source_by_role["mat_mat_q4_k_metal"]["sha256"]
        and predicate["metallib_sha256"] == manifest["embedded_metallib_sha256"]
        and predicate["build_source_sha256"]
        == manifest["expected_build"]["source_sha256"],
        "selector predicate is not derived from tensor/source/metallib/build identities",
    )
    for key in ("grid", "threads"):
        require(
            isinstance(predicate[key], list) and len(predicate[key]) == 3,
            f"selector predicate {key} invalid",
        )
        [integer(value, f"selector predicate {key}", 1) for value in predicate[key]]
    for key in ("metal_source_sha256", "metallib_sha256", "build_source_sha256"):
        sha256_text(predicate[key], f"selector predicate {key}")
    require(
        predicate["allowed_environment"] == {"QWEN_METAL_LEASE_WAIT": "1"},
        "selector predicate environment allowlist mismatch",
    )
    preparation_binding = exact_keys(
        manifest["preparation_binding"],
        PREPARATION_BINDING_KEYS,
        "manifest preparation binding",
    )
    for key in ("inventory", "inventory_spec", "preparation_spec"):
        identity_from_claim(
            preparation_binding[key], f"manifest preparation binding {key}"
        )
    seal_path = canonical_output_path(Path(preparation_binding["seal_path"]))
    require(
        str(seal_path) == preparation_binding["seal_path"],
        "manifest preparation seal path is not canonical",
    )
    exact_keys(
        manifest["scalar_contract"], SCALAR_CONTRACT_KEYS, "manifest scalar contract"
    )
    return manifest


def canonical_output_path(path: Path) -> Path:
    require(path.name not in {"", ".", ".."}, "output filename invalid")
    parent = path.parent.resolve(strict=True)
    require(parent.is_dir(), "output parent is not a directory")
    return parent / path.name


def bounded_git(path: Path, *arguments: str) -> bytes:
    git = Path("/usr/bin/git")
    require(
        git.is_file() and os.access(git, os.X_OK), "pinned /usr/bin/git unavailable"
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        completed = subprocess.run(
            [str(git), "--no-replace-objects", "-C", str(path), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise InvalidEvidence(f"bounded git inspection failed: {error}") from error
    require(completed.returncode == 0, "bounded git inspection command failed")
    require(
        len(completed.stdout) <= 4 << 20 and len(completed.stderr) <= 1 << 20,
        "bounded git inspection output exceeded cap",
    )
    return completed.stdout


def inspect_git_checkout(
    path: Path, name: str, *, require_clean: bool = True
) -> dict[str, Any]:
    canonical = path.resolve(strict=True)
    require(canonical.is_dir(), f"{name} is not a directory")
    top = Path(
        bounded_git(canonical, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve(strict=True)
    require(top == canonical, f"{name} is not the Git top-level path")
    head = bounded_git(canonical, "rev-parse", "HEAD").decode().strip()
    tree = bounded_git(canonical, "rev-parse", "HEAD^{tree}").decode().strip()
    git_oid_text(head, f"{name} HEAD")
    git_oid_text(tree, f"{name} HEAD tree")
    status = bounded_git(
        canonical, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    if require_clean:
        require(status == b"", f"{name} Git checkout is not completely clean")
    common_raw = (
        bounded_git(canonical, "rev-parse", "--git-common-dir").decode().strip()
    )
    common = (canonical / common_raw).resolve(strict=True)
    objects = (common / "objects").resolve(strict=True)
    require(objects.is_dir(), f"{name} Git common object directory missing")
    return {
        "path": canonical,
        "head": head,
        "tree": tree,
        "common": common,
        "objects": objects,
        "status": status,
    }


def validate_preparation_git_state(
    inventory: dict[str, Any],
    spec: dict[str, Any],
    allowed_y_outputs: set[Path],
    *,
    require_all_outputs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkout = inventory["checkout"]
    control = spec["control_y_input"]
    git_x = inspect_git_checkout(Path(checkout["path"]), "preparation worktree X")
    git_y = inspect_git_checkout(
        Path(control["path"]), "preparation control Y", require_clean=False
    )
    require(
        git_x["head"] == checkout["commit"]
        and git_x["tree"] == checkout["tree"]
        and git_x["status"] == b"",
        "preparation worktree X differs from frozen clean inventory checkout",
    )
    require(
        git_y["head"] == control["commit"]
        and git_y["tree"] == control["tree"]
        and git_x["common"] == git_y["common"]
        and git_x["objects"] == git_y["objects"],
        "preparation control Y HEAD/tree/repository relationship mismatch",
    )
    observed: set[Path] = set()
    for entry in git_y["status"].split(b"\0"):
        if not entry:
            continue
        require(
            entry.startswith(b"?? "),
            "preparation control Y has a tracked or noncanonical status entry",
        )
        try:
            relative = entry[3:].decode("utf-8")
        except UnicodeDecodeError as error:
            raise InvalidEvidence("control Y status path is not UTF-8") from error
        candidate = (git_y["path"] / relative).resolve()
        require(
            candidate in allowed_y_outputs,
            "preparation control Y has an unrelated untracked path",
        )
        observed.add(candidate)
    if require_all_outputs:
        require(
            observed == allowed_y_outputs,
            "control Y post-write status does not contain exactly four outputs",
        )
    return git_x, git_y


def final_custody_check(opened: OpenFile) -> None:
    info = os.fstat(opened.file.fileno())
    require(
        stat.S_ISREG(info.st_mode)
        and (info.st_dev, info.st_ino) == opened.inode
        and info.st_size == opened.size,
        f"custody changed for {opened.path}",
    )
    require(
        hash_range(opened, 0, opened.size, f"final custody {opened.path.name}")
        == opened.digest,
        f"final custody hash changed for {opened.path}",
    )


def custody_summary(opened: list[OpenFile]) -> dict[str, Any] | None:
    if not opened:
        return None
    names = ("manifest", "trace", "sidecar", "reducer")
    result: dict[str, Any] = {}
    for index, name in enumerate(names):
        if index < len(opened):
            item = opened[index]
            result[name] = {
                "path": str(item.path),
                "bytes": item.size,
                "sha256": item.digest,
            }
    result["artifacts"] = [
        {"path": str(item.path), "bytes": item.size, "sha256": item.digest}
        for item in sorted(opened[4:], key=lambda value: str(value.path))
    ]
    return result


def validate_inventory(
    inventory_path: Path,
    inventory_sha256: str,
    spec_path: Path,
    spec_sha256: str,
    *,
    retain_open: bool = False,
) -> dict[str, Any] | InventoryContext:
    sha256_text(inventory_sha256, "expected inventory SHA-256")
    sha256_text(spec_sha256, "expected inventory-spec SHA-256")
    opened: list[OpenFile] = []
    retained = False
    try:
        spec_file = open_regular(spec_path, "inventory spec", MAX_TRACE_BYTES)
        inventory_file = open_regular(inventory_path, "inventory", MAX_TRACE_BYTES)
        opened.extend((spec_file, inventory_file))
        require(spec_file.digest == spec_sha256, "inventory spec digest mismatch")
        require(inventory_file.digest == inventory_sha256, "inventory digest mismatch")
        spec = exact_keys(
            parse_json(
                pread_exact(spec_file, 0, spec_file.size, "inventory spec"),
                "inventory spec",
            ),
            INVENTORY_SPEC_KEYS,
            "inventory spec",
        )
        require(
            spec["schema"] == INVENTORY_SPEC_SCHEMA and spec["schema_version"] == 1,
            "inventory spec schema mismatch",
        )
        integer(spec["inventory_max_bytes"], "inventory max bytes", 1, MAX_TRACE_BYTES)
        require(
            inventory_file.size <= spec["inventory_max_bytes"],
            "inventory exceeds spec cap",
        )
        inventory = exact_keys(
            parse_json(
                pread_exact(inventory_file, 0, inventory_file.size, "inventory"),
                "inventory",
            ),
            INVENTORY_KEYS,
            "inventory",
        )
        reject_forbidden(inventory)
        require(
            inventory["schema"] == INVENTORY_SCHEMA
            and inventory["schema_version"] == INVENTORY_VERSION
            and inventory["authority"] == INVENTORY_AUTHORITY,
            "inventory schema/authority mismatch",
        )
        require(
            inventory["inventory_spec_sha256"] == spec_sha256,
            "inventory spec binding mismatch",
        )
        require(inventory["run_id"] == spec["run_id"], "inventory run id mismatch")
        checkout = exact_keys(
            inventory["checkout"],
            INVENTORY_CHECKOUT_KEYS,
            "inventory checkout",
        )
        exact_keys(spec["checkout"], INVENTORY_CHECKOUT_KEYS, "inventory spec checkout")
        require(checkout == spec["checkout"], "inventory checkout differs from spec")
        checkout_path = Path(text(checkout["path"], "checkout path")).resolve(
            strict=True
        )
        require(checkout_path.is_dir(), "inventory checkout is not a directory")
        for key in ("commit", "tree"):
            git_oid_text(checkout[key], f"checkout {key}")
        require(checkout["dirty"] is False, "inventory checkout is dirty")
        git_x = inspect_git_checkout(checkout_path, "inventory worktree X")
        require(
            checkout["commit"] == git_x["head"] and checkout["tree"] == git_x["tree"],
            "inventory checkout Git HEAD/tree mismatch",
        )
        build = exact_keys(inventory["build"], BUILD_KEYS, "inventory build")
        exact_keys(spec["build"], BUILD_KEYS, "inventory spec build")
        require(build == spec["build"], "inventory build differs from spec")
        require(
            build["commit"] == git_x["head"],
            "inventory build commit differs from actual worktree X HEAD",
        )
        claims: list[tuple[str, dict[str, Any]]] = []
        for key in (
            "executable",
            "reducer",
            "scalar_fixture",
            "command_template",
            "embedded_metallib",
        ):
            identity_from_claim(inventory[key], f"inventory {key}")
            identity_from_claim(spec[key], f"inventory spec {key}")
            require(inventory[key] == spec[key], f"inventory {key} differs from spec")
            claims.append((key, inventory[key]))
        for group, roles in (
            ("sources", REQUIRED_SOURCE_ROLES),
            ("assets", REQUIRED_ASSET_ROLES),
        ):
            require(isinstance(inventory[group], list), f"inventory {group} invalid")
            observed_roles = set()
            for item in inventory[group]:
                exact_keys(
                    item, ("role",) + FILE_CLAIM_KEYS, f"inventory {group} claim"
                )
                require(
                    item["role"] not in observed_roles,
                    f"duplicate inventory {group} role",
                )
                observed_roles.add(item["role"])
                claims.append((item["role"], item))
            require(observed_roles == roles, f"inventory {group} role set mismatch")
            require(
                inventory[group] == spec[group], f"inventory {group} differs from spec"
            )
        paths: set[Path] = {spec_file.path, inventory_file.path}
        inodes = {spec_file.inode, inventory_file.inode}
        opened_roles: dict[str, OpenFile] = {}
        for role, claim in claims:
            item = open_regular(
                Path(claim["path"]), f"inventory {role}", claim["max_bytes"]
            )
            opened.append(item)
            verify_identity(item, claim, f"inventory {role}")
            require(
                item.path not in paths and item.inode not in inodes,
                "inventory custody alias",
            )
            paths.add(item.path)
            inodes.add(item.inode)
            opened_roles[role] = item
        ggufs = {role: GGUF(opened_roles[role]) for role in REQUIRED_ASSET_ROLES}
        require(
            isinstance(inventory["gguf"], list) and len(inventory["gguf"]) == 2,
            "inventory GGUF facts invalid",
        )
        gguf_roles: set[str] = set()
        for fact in inventory["gguf"]:
            exact_keys(
                fact,
                INVENTORY_GGUF_KEYS,
                "inventory GGUF fact",
            )
            role = fact["role"]
            require(
                role in ggufs and role not in gguf_roles and fact["version"] == 3,
                "inventory GGUF role/version mismatch",
            )
            gguf_roles.add(role)
            check(
                fact["tensor_count"] == len(ggufs[role].tensors)
                and fact["metadata_count"] == len(ggufs[role].metadata),
                "inventory GGUF counts mismatch",
            )
        requirements = spec["tensor_requirements"]
        require(
            isinstance(requirements, list) and len(requirements) == 3,
            "inventory tensor requirements invalid",
        )
        requirement_by_role: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(requirements):
            item = exact_keys(
                raw, TENSOR_REQUIREMENT_KEYS, f"inventory tensor requirement {index}"
            )
            role = text(item["role"], "inventory tensor requirement role", maximum=64)
            require(
                role not in requirement_by_role, "duplicate tensor requirement role"
            )
            requirement_by_role[role] = item
        require(
            set(requirement_by_role) == {"selector_hidden", "predecessor", "successor"},
            "inventory tensor requirement roles mismatch",
        )
        require(
            requirement_by_role["selector_hidden"]["dtype"] == "Q4_K",
            "K0-S bridge v1 selector_hidden requirement must be Q4_K",
        )
        require(
            isinstance(inventory["tensors"], list) and len(inventory["tensors"]) == 3,
            "inventory tensors invalid",
        )
        observed_tensor_roles: set[str] = set()
        for index, raw in enumerate(inventory["tensors"]):
            claim = validate_tensor_claim(raw, f"inventory tensor {index}")
            role = claim["role"]
            require(
                role not in observed_tensor_roles, "duplicate inventory tensor role"
            )
            observed_tensor_roles.add(role)
            requirement = requirement_by_role.get(role)
            require(requirement is not None, "unexpected inventory tensor role")
            if role == "selector_hidden":
                require(
                    claim["dtype"] == "Q4_K",
                    "K0-S bridge v1 selector_hidden tensor must be Q4_K",
                )
            for key in TENSOR_REQUIREMENT_KEYS:
                require(
                    claim[key] == requirement[key],
                    f"inventory tensor {role} static descriptor differs from spec",
                )
            tensor = ggufs[claim["asset_role"]].tensors.get(claim["name"])
            require(tensor is not None, f"inventory tensor {role} absent from GGUF")
            check(
                list(tensor.shape) == claim["shape"]
                and tensor.dtype == claim["dtype"]
                and tensor.offset == claim["offset"]
                and tensor.size == claim["bytes"],
                f"inventory tensor {role} independently parsed descriptor mismatch",
            )
            check(
                hash_range(
                    opened_roles[claim["asset_role"]],
                    tensor.offset,
                    tensor.size,
                    f"inventory tensor {role}",
                )
                == claim["sha256"],
                f"inventory tensor {role} independently hashed region mismatch",
            )
        require(
            observed_tensor_roles == set(requirement_by_role),
            "inventory tensor role set mismatch",
        )
        tensor_by_role = {item["role"]: item for item in inventory["tensors"]}
        validate_vocab_compatibility(tensor_by_role, ggufs)
        tokenizer = exact_keys(
            inventory["tokenizer"],
            INVENTORY_TOKENIZER_KEYS,
            "inventory tokenizer",
        )
        expected_tokenizer = exact_keys(
            spec["tokenizer"], INVENTORY_TOKENIZER_KEYS, "inventory spec tokenizer"
        )
        target_gguf = ggufs["target"]
        embedding = target_gguf.tensors.get("token_embd.weight")
        require(embedding is not None, "target token embedding absent")
        token_count = target_gguf.metadata.get("tokenizer.ggml.token_count")
        tokens = target_gguf.metadata.get("tokenizer.ggml.tokens")
        if tokens is not None:
            require(
                isinstance(tokens, list) and all(isinstance(v, str) for v in tokens),
                "target tokenizer tokens metadata invalid",
            )
            token_count = len(tokens)
            token_digest = hashlib.sha256(
                b"".join(
                    struct.pack("<Q", len(v.encode("utf-8"))) + v.encode("utf-8")
                    for v in tokens
                )
            ).hexdigest()
        else:
            integer(token_count, "target tokenizer token count", VOCAB, VOCAB)
            token_digest = canonical_json_digest(
                {"token_count": token_count, "token_embd": list(embedding.shape)}
            )
        derived_tokenizer = {
            "vocab_size": VOCAB,
            "token_embd_name": "token_embd.weight",
            "token_embd_shape": list(embedding.shape),
            "token_embd_dtype": embedding.dtype,
            "token_count": token_count,
            "tokenizer_tokens_sha256": token_digest,
        }
        check(
            tokenizer == derived_tokenizer,
            "inventory tokenizer facts not independently derived",
        )
        require(
            tokenizer == expected_tokenizer,
            "inventory tokenizer differs from frozen spec",
        )
        prompt = exact_keys(
            inventory["prompt"],
            INVENTORY_PROMPT_KEYS,
            "inventory prompt",
        )
        expected_prompt = exact_keys(
            spec["prompt"], INVENTORY_PROMPT_KEYS, "inventory spec prompt"
        )
        require(prompt == expected_prompt, "inventory prompt differs from frozen spec")
        try:
            prompt_bytes = bytes.fromhex(prompt["utf8_hex"])
            prompt_bytes.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise InvalidEvidence(
                "inventory prompt is not canonical UTF-8 hex"
            ) from error
        require(
            prompt["utf8_hex"] == prompt_bytes.hex(),
            "inventory prompt hex is noncanonical",
        )
        prompt_tokens = [
            integer(value, "inventory prompt token", 0, VOCAB - 1)
            for value in prompt["token_ids"]
        ]
        require(
            prompt_tokens == [7734, 1970], "inventory prompt token expectation mismatch"
        )
        check(
            prompt["token_ids_sha256_i32le"]
            == hashlib.sha256(
                b"".join(struct.pack("<i", value) for value in prompt_tokens)
            ).hexdigest(),
            "inventory prompt token digest mismatch",
        )
        require(
            prompt["tokenizer_identity_sha256"] == token_digest,
            "inventory prompt tokenizer identity mismatch",
        )
        mask = exact_keys(
            inventory["mask_noise"],
            INVENTORY_MASK_KEYS,
            "inventory mask/noise",
        )
        metadata_keys = (
            "dflash-draft.dflash.mask_token_id",
            "tokenizer.ggml.mask_token_id",
        )
        present = [
            (key, ggufs["drafter"].metadata[key])
            for key in metadata_keys
            if key in ggufs["drafter"].metadata
        ]
        require(
            len(present) == 1,
            "drafter must expose exactly one supported mask metadata key",
        )
        mask_token = integer(present[0][1], "drafter mask metadata", 0, VOCAB - 1)
        require(
            mask["metadata_key"] == present[0][0]
            and mask["mask_token"] == mask_token
            and mask_token == spec["expected_mask_token"] == 248070,
            "inventory drafter mask metadata mismatch",
        )
        require(
            isinstance(mask["noise_tokens"], list) and len(mask["noise_tokens"]) == 8,
            "inventory noise geometry invalid",
        )
        noise = [
            integer(value, "inventory noise token", 0, VOCAB - 1)
            for value in mask["noise_tokens"]
        ]
        carry = integer(spec["carry_token"], "inventory spec carry token", 0, VOCAB - 1)
        require(
            noise == [carry] + [mask_token] * 7,
            "inventory deterministic noise tokens mismatch",
        )
        check(
            mask["noise_sha256_i32le"]
            == hashlib.sha256(
                b"".join(struct.pack("<i", value) for value in noise)
            ).hexdigest(),
            "inventory noise digest mismatch",
        )
        caps = exact_keys(
            inventory["parser_caps"],
            (
                "header_bytes",
                "metadata",
                "tensors",
                "strings_bytes",
                "array_items",
                "objects",
            ),
            "inventory parser caps",
        )
        require(
            caps
            == {
                "header_bytes": MAX_GGUF_HEADER_BYTES,
                "metadata": MAX_GGUF_METADATA,
                "tensors": MAX_GGUF_TENSORS,
                "strings_bytes": MAX_GGUF_STRINGS_BYTES,
                "array_items": MAX_GGUF_ARRAY_ITEMS,
                "objects": MAX_GGUF_OBJECTS,
            },
            "inventory parser caps mismatch",
        )
        require(caps == spec["parser_caps"], "inventory parser caps differ from spec")
        device = exact_keys(inventory["device"], HOST_KEYS, "inventory device")
        predicate = exact_keys(
            spec["host_predicate"], HOST_PREDICATE_KEYS, "inventory host predicate"
        )
        for key in HOST_PREDICATE_KEYS:
            require(
                device[key] == predicate[key],
                f"inventory device {key} predicate mismatch",
            )
        integer(device["device_registry_id"], "inventory device registry id")
        text(device["device_family"], "inventory device family", maximum=256)
        require(
            inventory["environment"]
            == spec["environment"]
            == {"QWEN_METAL_LEASE_WAIT": "1"},
            "inventory environment must be literal Metal lease only",
        )
        expected_command = [
            str(opened_roles["executable"].path),
            "dflash-k0s-inventory",
            "--model",
            str(opened_roles["target"].path),
            "--drafter",
            str(opened_roles["drafter"].path),
            "--prompt",
            prompt_bytes.decode("utf-8"),
            "--carry-token",
            str(carry),
            "--inventory-spec",
            str(spec_file.path),
            "--inventory-spec-sha256",
            spec_sha256,
            "--output",
            str(inventory_file.path),
        ]
        require(
            inventory["command"] == expected_command,
            "inventory hidden-command argv mismatch",
        )
        spec_command = spec["command"]
        require(isinstance(spec_command, list), "inventory spec command invalid")
        templated = list(expected_command)
        templated[templated.index("--inventory-spec-sha256") + 1] = (
            "${INVENTORY_SPEC_SHA256}"
        )
        templated[templated.index("--output") + 1] = "${INVENTORY_OUTPUT}"
        require(spec_command == templated, "inventory spec command template mismatch")
        for item in opened:
            final_custody_check(item)
        if retain_open:
            retained = True
            return InventoryContext(inventory, opened)
        return inventory
    except (KeyError, TypeError, struct.error, OverflowError, OSError) as error:
        raise InvalidEvidence(
            f"invalid inventory structure: {type(error).__name__}: {error}"
        ) from error
    finally:
        if not retained:
            for item in opened:
                item.close()


def frozen_reducer_argv(spec: dict[str, Any], reducer_path: str) -> list[str]:
    acquisition = spec["acquisition_outputs"]
    return [
        reducer_path,
        "--input",
        acquisition["trace"],
        "--sidecar",
        acquisition["sidecar"],
        "--manifest",
        acquisition["manifest"],
        "--manifest-sha256",
        MANIFEST_SHA256_PLACEHOLDER,
        "--inventory",
        spec["inventory_path"],
        "--inventory-sha256",
        spec["inventory_sha256"],
        "--inventory-spec",
        spec["inventory_spec_path"],
        "--inventory-spec-sha256",
        spec["inventory_spec_sha256"],
        "--preparation-spec",
        spec["preparation_spec_path"],
        "--preparation-spec-sha256",
        PREPARATION_SPEC_SHA256_PLACEHOLDER,
        "--preparation-seal",
        spec["outputs"]["seal"],
        "--preparation-seal-sha256",
        SEAL_SHA256_PLACEHOLDER,
        "--output",
        spec["reduction_output"],
    ]


def validate_literal_reducer_argv(
    actual: list[str],
    frozen: list[str],
    manifest_sha256: str,
    preparation_spec_sha256: str,
    seal_sha256: str,
    reducer_path: str,
) -> None:
    require(
        isinstance(actual, list)
        and isinstance(frozen, list)
        and all(isinstance(value, str) and value for value in actual + frozen),
        "reducer argv must contain literal nonempty strings",
    )
    expected = [
        manifest_sha256
        if value == MANIFEST_SHA256_PLACEHOLDER
        else preparation_spec_sha256
        if value == PREPARATION_SPEC_SHA256_PLACEHOLDER
        else seal_sha256
        if value == SEAL_SHA256_PLACEHOLDER
        else value
        for value in frozen
    ]
    require(len(actual) == len(expected), "actual reducer argv length mismatch")
    authenticated_reducer = Path(reducer_path).resolve(strict=True)
    require(
        Path(actual[0]).resolve(strict=True) == authenticated_reducer
        and Path(expected[0]).resolve(strict=True) == authenticated_reducer,
        "reducer argv[0] differs from authenticated reducer path",
    )
    require(
        actual[1:] == expected[1:],
        "actual reducer argv literal order/spelling/value mismatch",
    )


def render_preparation(
    inventory: dict[str, Any],
    inventory_sha256: str,
    spec: dict[str, Any],
    spec_sha256: str,
    preparation_claims: dict[str, dict[str, Any]] | None = None,
) -> tuple[bytes, bytes, bytes, bytes]:
    require(
        spec["inventory_sha256"] == inventory_sha256,
        "preparation inventory digest mismatch",
    )
    require(spec["run_id"] == inventory["run_id"], "preparation run id mismatch")
    fixture = (
        json.dumps(spec["fixture_content"], ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    if preparation_claims is None:
        preparation_claims = {}
        for key, path_value in (
            ("inventory", spec["inventory_path"]),
            ("inventory_spec", spec["inventory_spec_path"]),
            ("preparation_spec", spec["preparation_spec_path"]),
        ):
            opened_claim = open_regular(Path(path_value), key, MAX_TRACE_BYTES)
            try:
                preparation_claims[key] = {
                    "path": str(opened_claim.path),
                    "bytes": opened_claim.size,
                    "sha256": opened_claim.digest,
                    "max_bytes": MAX_TRACE_BYTES,
                }
            finally:
                opened_claim.close()
    fixture_object = exact_keys(
        spec["fixture_content"],
        ("schema", "schema_version", "fixture_domain", "fixture_sha256", "vectors"),
        "prepared scalar fixture",
    )
    require(
        fixture_object["fixture_domain"] == SCALAR_FIXTURE_DOMAIN
        and scalar_fixture_digest(fixture_object["vectors"])
        == fixture_object["fixture_sha256"]
        == SCALAR_FIXTURE_EXPECTED_SHA256,
        "prepared scalar fixture content/digest mismatch",
    )
    worktree = exact_keys(spec["worktree_x"], ("path", "commit"), "worktree X")
    control = exact_keys(
        spec["control_y_input"], ("path", "commit", "tree"), "control Y input"
    )
    x_path = Path(worktree["path"]).resolve(strict=True)
    y_path = Path(control["path"]).resolve(strict=True)
    require(
        x_path.is_dir() and y_path.is_dir(), "preparation X/Y paths must be directories"
    )
    require(x_path != y_path, "preparation worktree-X and control Y must be distinct")
    git_x = inspect_git_checkout(x_path, "preparation worktree X")
    git_y = inspect_git_checkout(y_path, "preparation control Y", require_clean=False)
    require(
        git_x["common"] == git_y["common"] and git_x["objects"] == git_y["objects"],
        "preparation X/Y do not share the authenticated Git repository/object store",
    )
    git_oid_text(worktree["commit"], "worktree X commit")
    git_oid_text(control["commit"], "control Y input commit")
    git_oid_text(control["tree"], "control Y input tree")
    require(
        control["commit"] == git_y["head"] and control["tree"] == git_y["tree"],
        "preparation control Y input HEAD/tree mismatch",
    )
    require(
        worktree
        == {
            "path": inventory["checkout"]["path"],
            "commit": inventory["checkout"]["commit"],
        }
        and spec["transformation_sha256"] == inventory["reducer"]["sha256"],
        "preparation worktree-X/transformation binding mismatch",
    )
    require(
        spec["environment_allowlist"] == {"QWEN_METAL_LEASE_WAIT": "1"},
        "preparation environment allowlist mismatch",
    )
    for claim in [
        inventory["executable"],
        inventory["reducer"],
        inventory["embedded_metallib"],
        *inventory["sources"],
    ]:
        require(
            Path(claim["path"]).resolve(strict=True).is_relative_to(x_path),
            "preparation X identity path escapes worktree X",
        )
    outputs = exact_keys(
        spec["outputs"],
        ("fixture", "command", "manifest", "seal"),
        "preparation outputs",
    )
    acquisition = exact_keys(
        spec["acquisition_outputs"],
        ACQUISITION_OUTPUT_KEYS,
        "preparation acquisition outputs",
    )
    output_paths = {
        key: str(canonical_output_path(Path(value))) for key, value in outputs.items()
    }
    acquisition_paths = {
        key: str(canonical_output_path(Path(value)))
        for key, value in acquisition.items()
    }
    require(
        acquisition_paths["manifest"] == output_paths["manifest"],
        "acquisition manifest path differs from prepared manifest output",
    )
    six_paths = list(output_paths.values()) + [
        acquisition_paths["trace"],
        acquisition_paths["sidecar"],
    ]
    require(
        len(set(six_paths)) == 6,
        "fixture/command/manifest/seal/trace/sidecar paths must be unique",
    )
    require(
        all(Path(value).parent == y_path for value in six_paths),
        "all prepared/acquisition outputs must reside directly under control Y",
    )
    reduction_output = str(canonical_output_path(Path(spec["reduction_output"])))
    require(
        Path(reduction_output).parent == y_path and reduction_output not in six_paths,
        "reduction output path is not uniquely confined under control Y",
    )
    require(
        spec["parity_comparison_fields"] == PARITY_COMPARISON_FIELDS,
        "preparation parity comparison matrix mismatch",
    )
    require(
        spec["reducer_argv"] == frozen_reducer_argv(spec, inventory["reducer"]["path"]),
        "preparation frozen reducer argv mismatch",
    )
    validate_preparation_git_state(
        inventory,
        spec,
        {Path(value) for value in output_paths.values()},
        require_all_outputs=False,
    )
    choices = exact_keys(
        spec["manifest_choices"], MANIFEST_CHOICE_KEYS, "preparation manifest choices"
    )
    continuation_carry = integer(
        spec["continuation_carry_token"], "continuation carry token", 0, VOCAB - 1
    )
    prompt = bytes.fromhex(inventory["prompt"]["utf8_hex"]).decode("utf-8")
    fixed = choices["expected_fixed_chains"]
    temperature_bits = bits32(
        choices["expected_request"]["request"]["temperature_f32_bits"],
        "prepared temperature",
        finite=True,
    )
    temperature_text = format(f32_value(temperature_bits), ".9g")
    command_argv = [
        inventory["executable"]["path"],
        "dflash-k0s-lattice",
        "--attempt-id",
        spec["attempt_id"],
        "--model",
        next(v["path"] for v in inventory["assets"] if v["role"] == "target"),
        "--drafter",
        next(v["path"] for v in inventory["assets"] if v["role"] == "drafter"),
        "--prompt",
        prompt,
        "--carry-token",
        str(choices["expected_capture_context"]["carry_token"]),
        "--continuation-carry-token",
        str(continuation_carry),
        "--manifest",
        output_paths["manifest"],
        "--manifest-sha256",
        MANIFEST_SHA256_PLACEHOLDER,
        "--command-manifest",
        output_paths["command"],
        "--fixture",
        output_paths["fixture"],
        "--temperature",
        temperature_text,
    ]
    for chain in fixed:
        command_argv += [
            "--fixed-chain",
            f"{chain['name']}:{chain['initial_carry']}:"
            + ",".join(str(slot) for slot in chain["slots"]),
        ]
    command_argv += [
        "--trace-output",
        acquisition_paths["trace"],
        "--sidecar-output",
        acquisition_paths["sidecar"],
    ]
    require(
        command_argv.count(MANIFEST_SHA256_PLACEHOLDER) == 1,
        "prepared command placeholder count mismatch",
    )
    command = (
        json.dumps({"argv": command_argv}, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    fixture_claim = {
        "path": output_paths["fixture"],
        "bytes": len(fixture),
        "sha256": hashlib.sha256(fixture).hexdigest(),
        "max_bytes": MAX_TRACE_BYTES,
    }
    command_claim = {
        "path": output_paths["command"],
        "bytes": len(command),
        "sha256": hashlib.sha256(command).hexdigest(),
        "max_bytes": MAX_TRACE_BYTES,
    }
    vectors = fixture_object["vectors"]
    manifest_object = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "run_id": spec["run_id"],
        "attempt_id": spec["attempt_id"],
        "trace_max_bytes": choices["trace_max_bytes"],
        "sidecar_max_bytes": choices["sidecar_max_bytes"],
        "reducer": inventory["reducer"],
        "executable": inventory["executable"],
        "fixture": fixture_claim,
        "command": command_claim,
        "sources": inventory["sources"],
        "assets": inventory["assets"],
        "semantic_references": choices["semantic_references"],
        "tensors": inventory["tensors"],
        "expected_request": choices["expected_request"],
        "expected_prompt": inventory["prompt"],
        "expected_binding": choices["expected_binding"],
        "expected_continuation_carry_token": continuation_carry,
        "expected_rng_domains": choices["expected_rng_domains"],
        "expected_fixed_chains": fixed,
        "expected_capture_context": choices["expected_capture_context"],
        "expected_build": inventory["build"],
        "expected_host": inventory["device"],
        "embedded_metallib_sha256": inventory["embedded_metallib"]["sha256"],
        "selector_dispatch_predicate": choices["selector_dispatch_predicate"],
        "preparation_binding": {
            "inventory": preparation_claims["inventory"],
            "inventory_spec": preparation_claims["inventory_spec"],
            "preparation_spec": preparation_claims["preparation_spec"],
            "seal_path": output_paths["seal"],
        },
        "scalar_contract": {
            "artifact": fixture_claim,
            "compiler": inventory["build"]["compiler"],
            "compiler_version": inventory["build"]["compiler_version"],
            "target": inventory["build"]["target"],
            "profile": inventory["build"]["profile"],
            "fixture_domain": fixture_object["fixture_domain"],
            "fixture_sha256": fixture_object["fixture_sha256"],
            "vectors": vectors,
        },
    }
    validate_manifest(manifest_object)
    manifest = (
        json.dumps(manifest_object, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    seal_object = {
        "schema": SEAL_SCHEMA,
        "schema_version": 1,
        "run_id": spec["run_id"],
        "attempt_id": spec["attempt_id"],
        "inventory_sha256": inventory_sha256,
        "inventory_spec_sha256": spec["inventory_spec_sha256"],
        "preparation_spec_sha256": spec_sha256,
        "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
        "command_sha256": hashlib.sha256(command).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "transformation_sha256": spec["transformation_sha256"],
        "reducer_sha256": inventory["reducer"]["sha256"],
        "worktree_x_commit": worktree["commit"],
        "control_y_input_commit": control["commit"],
        "control_y_input_tree": control["tree"],
    }
    exact_keys(seal_object, SEAL_KEYS, "prepared seal")
    seal = (
        json.dumps(seal_object, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    return fixture, command, manifest, seal


def prepare_artifacts(
    inventory_path: Path,
    inventory_sha256: str,
    inventory_spec_path: Path,
    inventory_spec_sha256: str,
    preparation_spec_path: Path,
    preparation_spec_sha256: str,
    expected_hashes: dict[str, str],
) -> dict[str, Any]:
    inventory_context = validate_inventory(
        inventory_path,
        inventory_sha256,
        inventory_spec_path,
        inventory_spec_sha256,
        retain_open=True,
    )
    require(
        isinstance(inventory_context, InventoryContext),
        "retained inventory context construction failed",
    )
    inventory = inventory_context.inventory
    prep_file = open_regular(preparation_spec_path, "preparation spec", MAX_TRACE_BYTES)
    try:
        require(
            prep_file.digest == preparation_spec_sha256,
            "preparation spec digest mismatch",
        )
        spec = exact_keys(
            parse_json(
                pread_exact(prep_file, 0, prep_file.size, "preparation spec"),
                "preparation spec",
            ),
            PREPARATION_SPEC_KEYS,
            "preparation spec",
        )
        require(
            spec["schema"] == PREPARATION_SPEC_SCHEMA and spec["schema_version"] == 1,
            "preparation spec schema mismatch",
        )
        require(
            spec["inventory_path"] == str(inventory_path.resolve()),
            "preparation inventory path mismatch",
        )
        require(
            spec["inventory_spec_path"] == str(inventory_spec_path.resolve())
            and spec["inventory_spec_sha256"] == inventory_spec_sha256,
            "preparation inventory-spec path/hash mismatch",
        )
        require(
            spec["arm_order"] == ["off-A", "on-A", "on-B", "off-B"]
            and spec["selected_arm"] == "on-A",
            "preparation arm contract mismatch",
        )
        require(
            isinstance(spec["environment_allowlist"], dict)
            and isinstance(spec["failure_policy"], dict),
            "preparation environment/failure policy invalid",
        )
        retained_by_path = {item.path: item for item in inventory_context.opened}
        preparation_claims = {}
        for key, path_value, digest in (
            ("inventory", inventory_path, inventory_sha256),
            ("inventory_spec", inventory_spec_path, inventory_spec_sha256),
        ):
            item = retained_by_path[path_value.resolve(strict=True)]
            preparation_claims[key] = {
                "path": str(item.path),
                "bytes": item.size,
                "sha256": digest,
                "max_bytes": MAX_TRACE_BYTES,
            }
        preparation_claims["preparation_spec"] = {
            "path": str(prep_file.path),
            "bytes": prep_file.size,
            "sha256": preparation_spec_sha256,
            "max_bytes": MAX_TRACE_BYTES,
        }
        rendered = render_preparation(
            inventory,
            inventory_sha256,
            spec,
            preparation_spec_sha256,
            preparation_claims,
        )
        names = ("fixture", "command", "manifest", "seal")
        outputs = exact_keys(spec["outputs"], names, "preparation outputs")
        for name, data in zip(names, rendered):
            sha256_text(expected_hashes[name], f"expected prepared {name} hash")
            require(
                hashlib.sha256(data).hexdigest() == expected_hashes[name],
                f"prepared {name} expected hash mismatch",
            )
        paths = [canonical_output_path(Path(outputs[name])) for name in names]
        require(len(set(paths)) == 4, "preparation output paths alias")
        acquisition = exact_keys(
            spec["acquisition_outputs"],
            ACQUISITION_OUTPUT_KEYS,
            "preparation acquisition outputs",
        )
        all_output_paths = paths + [
            canonical_output_path(Path(acquisition["trace"])),
            canonical_output_path(Path(acquisition["sidecar"])),
        ]
        input_paths = {
            inventory_path.resolve(strict=True),
            inventory_spec_path.resolve(strict=True),
            preparation_spec_path.resolve(strict=True),
            *(
                Path(claim["path"]).resolve(strict=True)
                for claim in [
                    inventory["executable"],
                    inventory["reducer"],
                    inventory["scalar_fixture"],
                    inventory["command_template"],
                    inventory["embedded_metallib"],
                    *inventory["sources"],
                    *inventory["assets"],
                ]
            ),
        }
        require(
            not set(all_output_paths) & input_paths,
            "preparation output path aliases an authenticated input",
        )
        input_inodes = {
            path.stat().st_dev.to_bytes(8, "little")
            + path.stat().st_ino.to_bytes(8, "little")
            for path in input_paths
        }
        for path in all_output_paths:
            if path.exists():
                info = path.stat()
                inode_key = info.st_dev.to_bytes(8, "little") + info.st_ino.to_bytes(
                    8, "little"
                )
                require(
                    inode_key not in input_inodes,
                    "preparation existing output inode aliases an authenticated input",
                )
        # Reauthenticate all inventory-bound inputs immediately before reservation.
        inventory_context.final_check()
        policy = exact_keys(
            spec["failure_policy"],
            ("on_collision", "retry"),
            "preparation failure policy",
        )
        require(
            policy == {"on_collision": "retain_reserved_partial", "retry": False},
            "preparation failure policy mismatch",
        )
        parent = paths[0].parent
        require(
            all(path.parent == parent for path in paths),
            "prepared outputs do not share one directory",
        )
        directory_fd = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        directory_stat = os.fstat(directory_fd)
        require(
            stat.S_ISDIR(directory_stat.st_mode),
            "control-Y custody FD is not a directory",
        )
        validate_preparation_git_state(
            inventory,
            spec,
            set(paths),
            require_all_outputs=False,
        )
        reserved: list[tuple[Path, int]] = []
        try:
            for path in paths:
                reserved.append(
                    (
                        path,
                        os.open(
                            path.name,
                            os.O_WRONLY
                            | os.O_CREAT
                            | os.O_EXCL
                            | getattr(os, "O_NOFOLLOW", 0),
                            0o600,
                            dir_fd=directory_fd,
                        ),
                    )
                )
        except OSError as error:
            for _, descriptor in reserved:
                os.close(descriptor)
            os.close(directory_fd)
            final_custody_check(prep_file)
            return {
                "result": "partial",
                "reason": f"exclusive preparation reservation failed: {error}",
                "reserved_empty_paths": [str(path) for path, _ in reserved],
                "retry": False,
            }
        for (path, descriptor), data in zip(reserved, rendered):
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        after_stat = os.fstat(directory_fd)
        require(
            (after_stat.st_dev, after_stat.st_ino)
            == (directory_stat.st_dev, directory_stat.st_ino),
            "control-Y directory custody changed during writes",
        )
        validate_preparation_git_state(
            inventory,
            spec,
            set(paths),
            require_all_outputs=True,
        )
        os.close(directory_fd)
        inventory_context.final_check()
        for (path, _), data in zip(reserved, rendered):
            written = open_regular(path, f"prepared output {path.name}", len(data))
            try:
                require(
                    written.size == len(data)
                    and written.digest == hashlib.sha256(data).hexdigest(),
                    "prepared output final custody mismatch",
                )
                final_custody_check(written)
            finally:
                written.close()
        final_custody_check(prep_file)
        return {
            name: {
                "path": str(path),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, path, data in zip(names, paths, rendered)
        }
    finally:
        prep_file.close()
        inventory_context.close()


def reduce(
    trace_path: Path,
    sidecar_path: Path,
    manifest_path: Path,
    output_path: Path,
    manifest_sha256: str,
    inventory_path: Path,
    inventory_sha256: str,
    inventory_spec_path: Path,
    inventory_spec_sha256: str,
    preparation_spec_path: Path,
    preparation_spec_sha256: str,
    preparation_seal_path: Path,
    preparation_seal_sha256: str,
    actual_argv: list[str] | None = None,
) -> dict[str, Any]:
    opened: list[OpenFile] = []
    result: dict[str, Any]
    run_id: str | None = None
    attempt_id: str | None = None
    custody: dict[str, Any] = {}
    output: Path | None = None
    try:
        sha256_text(manifest_sha256, "independent manifest SHA-256")
        output = canonical_output_path(output_path)
        manifest_file = open_regular(manifest_path, "manifest", MAX_TRACE_BYTES)
        opened.append(manifest_file)
        custody = {
            "manifest": {
                "path": str(manifest_file.path),
                "bytes": manifest_file.size,
                "sha256": manifest_file.digest,
            }
        }
        require(
            manifest_file.digest == manifest_sha256,
            "manifest differs from independent expected digest",
        )
        manifest = validate_manifest(
            parse_json(
                pread_exact(manifest_file, 0, manifest_file.size, "manifest"),
                "manifest",
            )
        )
        run_id = manifest["run_id"]
        attempt_id = manifest["attempt_id"]
        bridge_expected = {
            "inventory": (inventory_path, inventory_sha256),
            "inventory_spec": (inventory_spec_path, inventory_spec_sha256),
            "preparation_spec": (preparation_spec_path, preparation_spec_sha256),
        }
        bridge_files: dict[str, OpenFile] = {}
        for name, (path, digest) in bridge_expected.items():
            sha256_text(digest, f"independent {name} SHA-256")
            claim = manifest["preparation_binding"][name]
            item = open_regular(path, name, claim["max_bytes"])
            opened.append(item)
            require(
                item.path == Path(claim["path"]).resolve(strict=True)
                and item.size == claim["bytes"]
                and item.digest == claim["sha256"],
                f"{name} manifest identity mismatch",
            )
            require(item.digest == digest, f"{name} independent digest mismatch")
            bridge_files[name] = item
        sha256_text(preparation_seal_sha256, "independent preparation seal SHA-256")
        seal = open_regular(preparation_seal_path, "preparation seal", MAX_TRACE_BYTES)
        opened.append(seal)
        require(
            str(seal.path) == manifest["preparation_binding"]["seal_path"]
            and seal.digest == preparation_seal_sha256,
            "preparation seal path/digest mismatch",
        )
        seal_json = exact_keys(
            parse_json(
                pread_exact(seal, 0, seal.size, "preparation seal"), "preparation seal"
            ),
            SEAL_KEYS,
            "preparation seal",
        )
        require(
            seal_json["schema"] == SEAL_SCHEMA
            and seal_json["schema_version"] == 1
            and seal_json["run_id"] == run_id
            and seal_json["attempt_id"] == attempt_id
            and seal_json["inventory_sha256"] == inventory_sha256
            and seal_json["inventory_spec_sha256"] == inventory_spec_sha256
            and seal_json["preparation_spec_sha256"] == preparation_spec_sha256
            and seal_json["manifest_sha256"] == manifest_sha256
            and seal_json["fixture_sha256"] == manifest["fixture"]["sha256"]
            and seal_json["command_sha256"] == manifest["command"]["sha256"]
            and seal_json["reducer_sha256"] == manifest["reducer"]["sha256"]
            and seal_json["transformation_sha256"] == manifest["reducer"]["sha256"]
            and seal_json["worktree_x_commit"] == manifest["expected_build"]["commit"],
            "preparation seal trust chain mismatch",
        )
        inventory_json = exact_keys(
            parse_json(
                pread_exact(
                    bridge_files["inventory"],
                    0,
                    bridge_files["inventory"].size,
                    "bound inventory",
                ),
                "bound inventory",
            ),
            INVENTORY_KEYS,
            "bound inventory",
        )
        inventory_spec_json = exact_keys(
            parse_json(
                pread_exact(
                    bridge_files["inventory_spec"],
                    0,
                    bridge_files["inventory_spec"].size,
                    "bound inventory spec",
                ),
                "bound inventory spec",
            ),
            INVENTORY_SPEC_KEYS,
            "bound inventory spec",
        )
        prep_json = exact_keys(
            parse_json(
                pread_exact(
                    bridge_files["preparation_spec"],
                    0,
                    bridge_files["preparation_spec"].size,
                    "bound preparation spec",
                ),
                "bound preparation spec",
            ),
            PREPARATION_SPEC_KEYS,
            "bound preparation spec",
        )
        require(
            inventory_json["schema"] == INVENTORY_SCHEMA
            and inventory_json["authority"] == INVENTORY_AUTHORITY
            and inventory_json["run_id"] == run_id
            and inventory_json["inventory_spec_sha256"] == inventory_spec_sha256
            and inventory_json["sources"] == manifest["sources"]
            and inventory_json["assets"] == manifest["assets"]
            and inventory_json["tensors"] == manifest["tensors"]
            and inventory_json["prompt"] == manifest["expected_prompt"]
            and inventory_json["build"] == manifest["expected_build"]
            and inventory_json["device"] == manifest["expected_host"]
            and inventory_json["embedded_metallib"]["sha256"]
            == manifest["embedded_metallib_sha256"]
            and inventory_json["reducer"] == manifest["reducer"]
            and inventory_json["executable"] == manifest["executable"],
            "bound inventory semantic facts differ from manifest",
        )
        require(
            inventory_spec_json["schema"] == INVENTORY_SPEC_SCHEMA
            and inventory_spec_json["run_id"] == run_id,
            "bound inventory-spec schema/run mismatch",
        )
        require(
            prep_json["schema"] == PREPARATION_SPEC_SCHEMA
            and prep_json["run_id"] == run_id
            and prep_json["attempt_id"] == attempt_id
            and prep_json["inventory_sha256"] == inventory_sha256
            and prep_json["inventory_spec_sha256"] == inventory_spec_sha256
            and prep_json["arm_order"] == ["off-A", "on-A", "on-B", "off-B"]
            and prep_json["selected_arm"] == "on-A"
            and prep_json["parity_comparison_fields"] == PARITY_COMPARISON_FIELDS
            and prep_json["manifest_choices"]["expected_request"]
            == manifest["expected_request"]
            and prep_json["manifest_choices"]["selector_dispatch_predicate"]
            == manifest["selector_dispatch_predicate"]
            and prep_json["transformation_sha256"] == manifest["reducer"]["sha256"]
            and prep_json["failure_policy"]
            == {"on_collision": "retain_reserved_partial", "retry": False},
            "bound preparation spec differs from manifest/protocol",
        )
        require(
            seal_json["control_y_input_commit"]
            == prep_json["control_y_input"]["commit"]
            and seal_json["control_y_input_tree"]
            == prep_json["control_y_input"]["tree"]
            and seal_json["worktree_x_commit"] == prep_json["worktree_x"]["commit"],
            "seal X/Y identities differ from preparation spec",
        )
        expected_reducer_argv = frozen_reducer_argv(
            prep_json, manifest["reducer"]["path"]
        )
        require(
            prep_json["reducer_argv"] == expected_reducer_argv,
            "bound preparation reducer argv mismatch",
        )
        if actual_argv is not None:
            validate_literal_reducer_argv(
                actual_argv,
                expected_reducer_argv,
                manifest_sha256,
                preparation_spec_sha256,
                preparation_seal_sha256,
                manifest["reducer"]["path"],
            )
        trace = open_regular(trace_path, "trace", manifest["trace_max_bytes"])
        sidecar = open_regular(sidecar_path, "sidecar", manifest["sidecar_max_bytes"])
        reducer = open_regular(
            Path(__file__), "reducer", manifest["reducer"]["max_bytes"]
        )
        opened.extend((trace, sidecar, reducer))
        require(
            trace.size + sidecar.size <= MAX_COMBINED_BYTES,
            "combined trace/sidecar cap exceeded",
        )
        verify_identity(reducer, manifest["reducer"], "reducer")
        external: dict[str, OpenFile] = {}
        for name in ("executable", "fixture", "command"):
            item = open_regular(
                Path(manifest[name]["path"]), name, manifest[name]["max_bytes"]
            )
            opened.append(item)
            verify_identity(item, manifest[name], name)
            external[name] = item
        for group in ("sources", "assets"):
            for claim in manifest[group]:
                item = open_regular(
                    Path(claim["path"]),
                    f"{group}.{claim['role']}",
                    claim["max_bytes"],
                )
                opened.append(item)
                verify_identity(item, claim, f"{group}.{claim['role']}")
                external[claim["role"]] = item
        paths = [item.path for item in opened]
        inodes = [item.inode for item in opened]
        require(
            len(paths) == len(set(paths)) and len(inodes) == len(set(inodes)),
            "input paths alias canonically or by inode",
        )
        require(
            output not in paths,
            "output aliases an input",
        )
        command_binding = validate_command_manifest(
            external["command"],
            manifest_file,
            trace,
            sidecar,
            manifest,
            external,
            manifest_sha256,
        )
        check(
            manifest["expected_binding"]["drafter_checkpoint_sha256"]
            == external["drafter"].digest,
            "drafter checkpoint binding differs from unique opened drafter asset",
        )
        custody = {
            "manifest": {
                "path": str(manifest_file.path),
                "bytes": manifest_file.size,
                "sha256": manifest_file.digest,
            },
            "trace": {
                "path": str(trace.path),
                "bytes": trace.size,
                "sha256": trace.digest,
            },
            "sidecar": {
                "path": str(sidecar.path),
                "bytes": sidecar.size,
                "sha256": sidecar.digest,
            },
            "reducer": {
                "path": str(reducer.path),
                "bytes": reducer.size,
                "sha256": reducer.digest,
            },
            "preparation_inputs": {
                name: {
                    "path": str(item.path),
                    "bytes": item.size,
                    "sha256": item.digest,
                }
                for name, item in bridge_files.items()
            },
            "preparation_seal": {
                "path": str(seal.path),
                "bytes": seal.size,
                "sha256": seal.digest,
            },
            "artifacts": [
                {"path": str(item.path), "bytes": item.size, "sha256": item.digest}
                for item in sorted(opened[4:], key=lambda value: str(value.path))
            ],
        }
        rows = parse_trace(trace)
        require(rows[0]["run_id"] == run_id, "manifest/trace run id mismatch")
        require(
            rows[0]["attempt_id"] == attempt_id, "manifest/trace attempt id mismatch"
        )
        if rows[0]["event"] == "parity_failure":
            require(sidecar.size == 0, "parity failure sidecar must be empty")
            failure = validate_parity_failure(rows[0]["payload"], manifest)
            raise FailedEvidence(
                "handled diagnostic parity failure",
                {"parity_failure": failure},
            )
        identities = exact_keys(
            rows[0]["payload"]["identities"], IDENTITY_KEYS, "trace identities"
        )
        identity_from_claim(identities["sidecar"], "trace identities.sidecar")
        verify_identity(sidecar, identities["sidecar"], "trace sidecar claim")
        for name in ("reducer", "executable", "fixture", "command"):
            identity_from_claim(identities[name], f"trace identities.{name}")
            check(
                identities[name] == manifest[name],
                f"trace {name} claim differs from external manifest",
            )
        for group in ("sources", "assets"):
            require(
                isinstance(identities[group], list), f"trace identities.{group} invalid"
            )
            for index, claim in enumerate(identities[group]):
                exact_keys(
                    claim,
                    ("role",) + FILE_CLAIM_KEYS,
                    f"trace identities.{group}[{index}]",
                )
        check(
            identities["sources"] == manifest["sources"]
            and identities["assets"] == manifest["assets"],
            "trace source/asset claims differ from external manifest",
        )
        assets = {
            claim["role"]: external[claim["role"]] for claim in manifest["assets"]
        }
        tensor_claims, ggufs = tensor_assets(manifest, assets)
        validate_vocab_compatibility(tensor_claims, ggufs)
        validate_scalar_contract(
            manifest["scalar_contract"],
            external["fixture"],
            manifest["expected_build"],
        )
        metrics = validate_packet(rows, sidecar, manifest, assets, tensor_claims)
        metrics["command_binding"] = command_binding
        result = {
            "schema": REDUCTION_SCHEMA,
            "schema_version": REDUCTION_VERSION,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "result": "passed",
            "authority": AUTHORITY,
            "reason": None,
            "metrics": metrics,
            "custody": custody,
        }
    except FailedEvidence as error:
        result = {
            "schema": REDUCTION_SCHEMA,
            "schema_version": REDUCTION_VERSION,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "result": "failed",
            "authority": AUTHORITY,
            "reason": str(error),
            "metrics": error.metrics,
            "custody": custody or None,
        }
    except Exception as error:
        result = {
            "schema": REDUCTION_SCHEMA,
            "schema_version": REDUCTION_VERSION,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "result": "invalid",
            "authority": AUTHORITY,
            "reason": f"{type(error).__name__}: {error}",
            "metrics": None,
            "custody": custody or None,
        }
    result["custody"] = custody or None
    try:
        for item in opened:
            final_custody_check(item)
    except Exception as error:
        result = {
            "schema": REDUCTION_SCHEMA,
            "schema_version": REDUCTION_VERSION,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "result": "invalid",
            "authority": AUTHORITY,
            "reason": f"final custody failure: {type(error).__name__}: {error}",
            "metrics": None,
            "custody": custody or None,
        }
    finally:
        for item in opened:
            item.close()
    data = (
        json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    try:
        require(output is not None, "output path validation failed")
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise InvalidEvidence(f"exclusive-create output failed: {error}") from error
    return result


def expect_error(function: Any, contains: str = "") -> None:
    try:
        function()
    except (InvalidEvidence, FailedEvidence) as error:
        require(contains in str(error), f"unexpected error text: {error}")
    else:
        raise AssertionError("expected strict rejection")


def minimal_gguf(
    tensors: list[tuple[str, list[int], int, bytes]],
    metadata: list[tuple[str, int, Any]] | None = None,
) -> bytes:
    metadata = metadata or []
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))

    def string(value: str) -> bytes:
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    for key, dtype, value in metadata:
        header += string(key) + struct.pack("<I", dtype)
        if dtype == 4:
            header += struct.pack("<I", value)
        else:
            raise AssertionError("self-test metadata helper only supports u32")
    data = bytearray()
    for name, shape, dtype, raw in tensors:
        while len(data) % 32:
            data.append(0)
        offset = len(data)
        header += (
            string(name)
            + struct.pack("<I", len(shape))
            + b"".join(struct.pack("<Q", n) for n in shape)
            + struct.pack("<IQ", dtype, offset)
        )
        data += raw
    while len(header) % 32:
        header.append(0)
    return bytes(header + data)


def synthetic_file_claim(path: Path, maximum: int | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "max_bytes": maximum if maximum is not None else max(1, len(raw)),
    }


def write_sparse_selector_gguf(path: Path) -> dict[str, Tensor]:
    specs = [
        ("selector_hidden.weight", [HIDDEN, RANK], 12),
        ("selector_predecessor.weight", [RANK, VOCAB], 12),
        ("selector_successor.weight", [RANK, VOCAB], 12),
    ]
    mask_key = b"tokenizer.ggml.mask_token_id"
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(specs), 1))
    header += (
        struct.pack("<Q", len(mask_key)) + mask_key + struct.pack("<II", 4, 248070)
    )
    offsets: list[int] = []
    sizes: list[int] = []
    cursor = 0
    for name, shape, dtype_id in specs:
        dtype, block, block_bytes = GGUF_LAYOUT_BY_ID[dtype_id]
        size = math.prod(shape) // block * block_bytes
        cursor = (cursor + 31) & -32
        offsets.append(cursor)
        sizes.append(size)
        encoded = name.encode()
        header += struct.pack("<Q", len(encoded)) + encoded
        header += struct.pack("<I", len(shape))
        header += b"".join(struct.pack("<Q", value) for value in shape)
        header += struct.pack("<IQ", dtype_id, cursor)
        cursor += size
    while len(header) % 32:
        header.append(0)
    data_start = len(header)
    with path.open("wb") as handle:
        handle.write(header)
        handle.truncate(data_start + cursor)
    return {
        role: Tensor(name, tuple(shape), "Q4_K", data_start + offset, size)
        for role, (name, shape, _), offset, size in zip(
            ("selector_hidden", "predecessor", "successor"),
            specs,
            offsets,
            sizes,
        )
    }


def synthetic_scalar_vectors() -> list[dict[str, Any]]:
    def blank() -> tuple[list[int], list[int], list[int], int]:
        return [0] * RANK, [0x3F800000] * RANK, [0] * RANK, 0

    cases: list[tuple[str, tuple[list[int], list[int], list[int], int]]] = []
    cancellation = blank()
    cancellation[0][0], cancellation[2][0] = 0xBF800000, 0x3F800000
    cancellation[0][1], cancellation[2][1] = 0x3F800001, 0x3F7FFFFF
    cases.append(("fma_sensitive_cancellation", cancellation))
    subnormal = blank()
    subnormal[0][0], subnormal[2][0] = 1, 0x3F800000
    subnormal[0][1], subnormal[2][1] = 2, 0xBF800000
    cases.append(("subnormal_signed_result", subnormal))
    signed_zero = blank()
    signed_zero[2][:] = [0xBF800000] * RANK
    signed_zero = (signed_zero[0], signed_zero[1], signed_zero[2], 0x80000000)
    cases.append(("signed_zero", signed_zero))
    adjacent = blank()
    adjacent[0][0] = adjacent[0][1] = 0x7F7FFFFF
    adjacent[1][0] = adjacent[1][1] = 0x3F000000
    adjacent[2][0] = adjacent[2][1] = 0x3F800000
    cases.append(("overflow_adjacent_finite", adjacent))
    rank_order = blank()
    rank_order[0][0] = f32_from_number(1.0e20)
    rank_order[0][1] = f32_from_number(-1.0e20)
    rank_order[0][2] = f32_from_number(3.25)
    rank_order[2][0:3] = [0x3F800000] * 3
    cases.append(("rank_order_cancellation", rank_order))
    halfway = blank()
    halfway[0][0], halfway[2][0] = 0x3F800000, 0x3F800000
    halfway[0][1], halfway[2][1] = 0x33800000, 0x3F800000
    cases.append(("halfway_round_to_even", halfway))

    vectors = []
    for name, (a, z, b, unary) in cases:
        score = replay_score(a, z, b, unary)
        vectors.append(
            {
                "name": name,
                "a_f32_bits": [enc32(value) for value in a],
                "z_f32_bits": [enc32(value) for value in z],
                "successor_f32_bits": [enc32(value) for value in b],
                "unary_f32_bits": enc32(unary),
                "score_f32_bits": enc32(score),
            }
        )
    return vectors


def build_complete_synthetic_packet(root: Path) -> dict[str, Any]:
    run_id = "k0s-complete-synthetic-v1"
    attempt_id = "synthetic-attempt-1"
    drafter_path = root / "drafter.gguf"
    descriptors = write_sparse_selector_gguf(drafter_path)
    target_path = root / "target.gguf"
    target_path.write_bytes(
        minimal_gguf(
            [
                ("token_embd.weight", [1, VOCAB], 0, bytes(VOCAB * 4)),
                ("unrelated.q4_0", [32], 2, bytes(18)),
            ],
            [("tokenizer.ggml.token_count", 4, VOCAB)],
        )
    )
    tensor_claims = []
    drafter_open = open_regular(drafter_path, "synthetic drafter")
    for role in ("selector_hidden", "predecessor", "successor"):
        desc = descriptors[role]
        tensor_claims.append(
            {
                "role": role,
                "asset_role": "drafter",
                "name": desc.name,
                "dtype": desc.dtype,
                "shape": list(desc.shape),
                "offset": desc.offset,
                "bytes": desc.size,
                "sha256": hash_range(drafter_open, desc.offset, desc.size, desc.name),
                "orientation": "gguf_ne0_hidden_ne1_rank"
                if role == "selector_hidden"
                else "gguf_ne0_rank_ne1_token",
                "row_domain": None
                if role == "selector_hidden"
                else {"first": 0, "count": VOCAB},
            }
        )
    drafter_open.close()

    fixture_path = root / "scalar-fixture.json"
    vectors = synthetic_scalar_vectors()
    fixture_digest = scalar_fixture_digest(vectors)
    fixture_path.write_text(
        json.dumps(
            {
                "schema": "qwen.dflash_k0s_scalar_fixture",
                "schema_version": 1,
                "fixture_domain": SCALAR_FIXTURE_DOMAIN,
                "fixture_sha256": fixture_digest,
                "vectors": vectors,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    executable_path = root / "qwen-cli"
    executable_path.write_bytes(b"synthetic executable identity\n")
    command_path = root / "command.json"
    source_claims = []
    for role in sorted(REQUIRED_SOURCE_ROLES):
        path = root / f"source-{role}"
        path.write_bytes((role + "\n").encode())
        source_claims.append({"role": role, **synthetic_file_claim(path)})

    build = {
        "commit": hashlib.sha256(b"commit").hexdigest(),
        "source_sha256": hashlib.sha256(b"source-state").hexdigest(),
        "dirty": False,
        "compiler": "rustc",
        "compiler_version": "rustc-synthetic-1.0",
        "target": "aarch64-apple-darwin",
        "profile": "release",
        "features": ["dflash-k0s-diagnostics"],
    }
    expected_selector_dispatch = {
        "family": "dflash_tail",
        "tag": SELECTOR_DISPATCH_TAG,
        "encoder_ordinal": 1,
        "encoder_concurrent": False,
        "kernel": SELECTOR_KERNEL,
        "grid": [1, 1, 1],
        "threads": [1, 1, 1],
        "grid_threadgroups": 1,
        "threadgroup_threads": 1,
    }
    expected_host = {
        "os": "macOS-synthetic",
        "arch": "arm64",
        "device_name": "Synthetic Metal Device",
        "device_registry_id": 1,
        "device_family": "synthetic-family",
    }
    metallib_sha256 = hashlib.sha256(b"metallib").hexdigest()
    selector_dispatch_predicate = {
        "tag": SELECTOR_DISPATCH_TAG,
        "kernel": expected_selector_dispatch["kernel"],
        "weight_dtype": "Q4_K",
        "input_dtype": "F32",
        "output_dtype": "F32",
        "weight_dtype_id": 12,
        "input_dtype_id": 0,
        "output_dtype_id": 0,
        "n": 8,
        "h": HIDDEN,
        "r": RANK,
        "grid": expected_selector_dispatch["grid"],
        "threads": expected_selector_dispatch["threads"],
        "metal_source_sha256": next(
            claim["sha256"]
            for claim in source_claims
            if claim["role"] == "mat_mat_q4_k_metal"
        ),
        "metallib_sha256": metallib_sha256,
        "build_source_sha256": build["source_sha256"],
        "allowed_environment": {"QWEN_METAL_LEASE_WAIT": "1"},
    }
    binding = {
        "production_call_id": "synthetic-call-0",
        "drafter_checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "proposal_construction_id": "synthetic-proposal-0",
        "noise_input_sha256": hashlib.sha256(b"noise-input").hexdigest(),
    }
    request = {"temperature_f32_bits": "0x3f800000"}
    ignored_policy = {
        "variant_a": {
            "top_k": 1,
            "top_p_f32_bits": "0x3f000000",
            "min_p_f32_bits": "0x00000000",
            "grammar": None,
            "penalties": None,
        },
        "variant_b": {
            "top_k": 100,
            "top_p_f32_bits": "0x3f800000",
            "min_p_f32_bits": "0x3dcccccd",
            "grammar": None,
            "penalties": None,
        },
    }

    fixture_claim = synthetic_file_claim(fixture_path)
    executable_claim = synthetic_file_claim(executable_path)
    reducer_claim = synthetic_file_claim(Path(__file__).resolve(), MAX_TRACE_BYTES)
    asset_claims = [
        {
            "role": "target",
            **synthetic_file_claim(target_path, target_path.stat().st_size),
        },
        {
            "role": "drafter",
            **synthetic_file_claim(drafter_path, drafter_path.stat().st_size),
        },
    ]
    binding["drafter_checkpoint_sha256"] = next(
        claim["sha256"] for claim in asset_claims if claim["role"] == "drafter"
    )
    static_fixed_chains = [
        {"name": "fixed-slot-one", "initial_carry": 0, "slots": [1] * DEPTHS}
    ]
    capture_context = {
        "definition_version": CAPTURE_DEFINITION,
        "carry_token": 0,
        "noise_start_position": 10,
        "target_context_len": 10,
        "context_hidden_watermark": 10,
        "kv_context_watermark": 10,
    }
    manifest_path = root / "manifest.json"
    trace_path = root / "trace.jsonl"
    sidecar_path = root / "capture.bin"
    command_argv = [
        str(executable_path.resolve()),
        "dflash-k0s-lattice",
        "--attempt-id",
        attempt_id,
        "--model",
        str(target_path.resolve()),
        "--drafter",
        str(drafter_path.resolve()),
        "--prompt",
        "synthetic prompt",
        "--carry-token",
        "0",
        "--continuation-carry-token",
        "1",
        "--manifest",
        str(manifest_path.resolve()),
        "--manifest-sha256",
        MANIFEST_SHA256_PLACEHOLDER,
        "--command-manifest",
        str(command_path.resolve()),
        "--fixture",
        str(fixture_path.resolve()),
        "--temperature",
        "1",
        "--fixed-chain",
        "fixed-slot-one:0:1,1,1,1,1,1,1",
        "--trace-output",
        str(trace_path.resolve()),
        "--sidecar-output",
        str(sidecar_path.resolve()),
    ]
    command_path.write_text(
        json.dumps({"argv": command_argv}, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    command_claim = synthetic_file_claim(command_path)
    preparation_input_paths = {
        "inventory": root / "bound-inventory.json",
        "inventory_spec": root / "bound-inventory-spec.json",
        "preparation_spec": root / "bound-preparation-spec.json",
    }
    metallib_path = root / "bound-metallib.bin"
    metallib_path.write_bytes(b"metallib")
    prompt_claim = {
        "utf8_hex": b"synthetic prompt".hex(),
        "token_ids": [7734, 1970],
        "token_ids_sha256_i32le": hashlib.sha256(
            struct.pack("<ii", 7734, 1970)
        ).hexdigest(),
        "tokenizer_identity_sha256": canonical_json_digest(
            {"token_count": VOCAB, "token_embd": [1, VOCAB]}
        ),
    }
    inventory_spec_object = {
        "schema": INVENTORY_SPEC_SCHEMA,
        "schema_version": 1,
        "run_id": run_id,
        "inventory_max_bytes": MAX_TRACE_BYTES,
        "checkout": {
            "path": str(root.resolve()),
            "commit": build["commit"],
            "tree": "1" * 40,
            "dirty": False,
        },
        "build": build,
        "sources": source_claims,
        "executable": executable_claim,
        "reducer": reducer_claim,
        "scalar_fixture": fixture_claim,
        "command_template": command_claim,
        "embedded_metallib": synthetic_file_claim(metallib_path),
        "assets": asset_claims,
        "tensor_requirements": [
            {key: tensor[key] for key in TENSOR_REQUIREMENT_KEYS}
            for tensor in tensor_claims
        ],
        "tokenizer": {
            "vocab_size": VOCAB,
            "token_embd_name": "token_embd.weight",
            "token_embd_shape": [1, VOCAB],
            "token_embd_dtype": "F32",
            "token_count": VOCAB,
            "tokenizer_tokens_sha256": prompt_claim["tokenizer_identity_sha256"],
        },
        "prompt": prompt_claim,
        "carry_token": 0,
        "expected_mask_token": 248070,
        "parser_caps": {
            "header_bytes": MAX_GGUF_HEADER_BYTES,
            "metadata": MAX_GGUF_METADATA,
            "tensors": MAX_GGUF_TENSORS,
            "strings_bytes": MAX_GGUF_STRINGS_BYTES,
            "array_items": MAX_GGUF_ARRAY_ITEMS,
            "objects": MAX_GGUF_OBJECTS,
        },
        "host_predicate": {key: expected_host[key] for key in HOST_PREDICATE_KEYS},
        "command": ["synthetic-inventory"],
        "environment": {"QWEN_METAL_LEASE_WAIT": "1"},
    }
    preparation_input_paths["inventory_spec"].write_text(
        json.dumps(inventory_spec_object, separators=(",", ":")), encoding="utf-8"
    )
    inventory_spec_digest = hashlib.sha256(
        preparation_input_paths["inventory_spec"].read_bytes()
    ).hexdigest()
    inventory_object = {
        "schema": INVENTORY_SCHEMA,
        "schema_version": 1,
        "authority": INVENTORY_AUTHORITY,
        "inventory_spec_sha256": inventory_spec_digest,
        "run_id": run_id,
        "checkout": inventory_spec_object["checkout"],
        "build": build,
        "sources": source_claims,
        "executable": executable_claim,
        "reducer": reducer_claim,
        "scalar_fixture": fixture_claim,
        "command_template": command_claim,
        "embedded_metallib": synthetic_file_claim(metallib_path),
        "device": expected_host,
        "assets": asset_claims,
        "gguf": [],
        "tensors": tensor_claims,
        "tokenizer": inventory_spec_object["tokenizer"],
        "prompt": prompt_claim,
        "mask_noise": {
            "metadata_key": "tokenizer.ggml.mask_token_id",
            "mask_token": 248070,
            "noise_tokens": [0] + [248070] * 7,
            "noise_sha256_i32le": hashlib.sha256(
                b"".join(struct.pack("<i", v) for v in [0] + [248070] * 7)
            ).hexdigest(),
        },
        "parser_caps": inventory_spec_object["parser_caps"],
        "command": ["synthetic-inventory"],
        "environment": {"QWEN_METAL_LEASE_WAIT": "1"},
    }
    preparation_input_paths["inventory"].write_text(
        json.dumps(inventory_object, separators=(",", ":")), encoding="utf-8"
    )
    inventory_digest = hashlib.sha256(
        preparation_input_paths["inventory"].read_bytes()
    ).hexdigest()
    synthetic_prep = {
        "schema": PREPARATION_SPEC_SCHEMA,
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "inventory_path": str(preparation_input_paths["inventory"].resolve()),
        "inventory_sha256": inventory_digest,
        "inventory_spec_path": str(preparation_input_paths["inventory_spec"].resolve()),
        "inventory_spec_sha256": inventory_spec_digest,
        "preparation_spec_path": str(
            preparation_input_paths["preparation_spec"].resolve()
        ),
        "worktree_x": {"path": str(root.resolve()), "commit": build["commit"]},
        "control_y_input": {
            "path": str(root.resolve()),
            "commit": "2" * 40,
            "tree": "3" * 40,
        },
        "outputs": {
            "fixture": str(fixture_path.resolve()),
            "command": str(command_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "seal": str((root / "bound-preparation-seal.json").resolve()),
        },
        "fixture_content": parse_json(fixture_path.read_bytes(), "synthetic fixture"),
        "acquisition_outputs": {
            "manifest": str(manifest_path.resolve()),
            "trace": str(trace_path.resolve()),
            "sidecar": str(sidecar_path.resolve()),
        },
        "continuation_carry_token": 1,
        "manifest_choices": {key: None for key in MANIFEST_CHOICE_KEYS},
        "transformation_sha256": reducer_claim["sha256"],
        "environment_allowlist": {"QWEN_METAL_LEASE_WAIT": "1"},
        "arm_order": ["off-A", "on-A", "on-B", "off-B"],
        "selected_arm": "on-A",
        "parity_comparison_fields": PARITY_COMPARISON_FIELDS,
        "reducer_argv": [],
        "reduction_output": str((root / "reduction.json").resolve()),
        "failure_policy": {"on_collision": "retain_reserved_partial", "retry": False},
    }
    synthetic_prep["manifest_choices"]["expected_request"] = {
        "request": request,
        "ignored_target_policy": ignored_policy,
    }
    synthetic_prep["manifest_choices"]["selector_dispatch_predicate"] = (
        selector_dispatch_predicate
    )
    synthetic_prep["reducer_argv"] = frozen_reducer_argv(
        synthetic_prep, reducer_claim["path"]
    )
    preparation_input_paths["preparation_spec"].write_text(
        json.dumps(synthetic_prep, separators=(",", ":")), encoding="utf-8"
    )
    preparation_seal_path = root / "bound-preparation-seal.json"
    preparation_binding = {
        **{
            name: synthetic_file_claim(path, MAX_TRACE_BYTES)
            for name, path in preparation_input_paths.items()
        },
        "seal_path": str(preparation_seal_path.resolve()),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "trace_max_bytes": MAX_TRACE_BYTES,
        "sidecar_max_bytes": MAX_SIDECAR_BYTES,
        "reducer": reducer_claim,
        "executable": executable_claim,
        "fixture": fixture_claim,
        "command": command_claim,
        "sources": source_claims,
        "assets": asset_claims,
        "semantic_references": SEMANTIC_REFERENCES,
        "tensors": tensor_claims,
        "expected_request": {
            "request": request,
            "ignored_target_policy": ignored_policy,
        },
        "expected_prompt": {
            "utf8_hex": b"synthetic prompt".hex(),
            "token_ids": [7734, 1970],
            "token_ids_sha256_i32le": hashlib.sha256(
                struct.pack("<ii", 7734, 1970)
            ).hexdigest(),
            "tokenizer_identity_sha256": canonical_json_digest(
                {"token_count": VOCAB, "token_embd": [1, VOCAB]}
            ),
        },
        "expected_binding": binding,
        "expected_continuation_carry_token": 1,
        "expected_rng_domains": ["request_rng"],
        "expected_fixed_chains": static_fixed_chains,
        "expected_capture_context": capture_context,
        "expected_build": build,
        "expected_host": expected_host,
        "embedded_metallib_sha256": metallib_sha256,
        "selector_dispatch_predicate": selector_dispatch_predicate,
        "preparation_binding": preparation_binding,
        "scalar_contract": {
            "artifact": fixture_claim,
            "compiler": build["compiler"],
            "compiler_version": build["compiler_version"],
            "target": build["target"],
            "profile": build["profile"],
            "fixture_domain": SCALAR_FIXTURE_DOMAIN,
            "fixture_sha256": fixture_digest,
            "vectors": vectors,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, separators=(",", ":")), encoding="utf-8"
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    preparation_spec_digest = hashlib.sha256(
        preparation_input_paths["preparation_spec"].read_bytes()
    ).hexdigest()
    synthetic_seal = {
        "schema": SEAL_SCHEMA,
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "inventory_sha256": inventory_digest,
        "inventory_spec_sha256": inventory_spec_digest,
        "preparation_spec_sha256": preparation_spec_digest,
        "fixture_sha256": fixture_claim["sha256"],
        "command_sha256": command_claim["sha256"],
        "manifest_sha256": manifest_sha256,
        "transformation_sha256": reducer_claim["sha256"],
        "reducer_sha256": reducer_claim["sha256"],
        "worktree_x_commit": build["commit"],
        "control_y_input_commit": "2" * 40,
        "control_y_input_tree": "3" * 40,
    }
    preparation_seal_path.write_text(
        json.dumps(synthetic_seal, separators=(",", ":")), encoding="utf-8"
    )
    preparation_seal_digest = hashlib.sha256(
        preparation_seal_path.read_bytes()
    ).hexdigest()

    # Dynamic capture values are generated only after the prospective manifest is fixed.
    provenance = {
        "dispatch_census": [
            {
                "family": "dflash_tail",
                "tag": None,
                "encoder_ordinal": 0,
                "encoder_concurrent": False,
                "kernel": "kernel_topk16_f32",
                "grid": [1, 1, 1],
                "threads": [1, 1, 1],
                "grid_threadgroups": 1,
                "threadgroup_threads": 1,
            },
            copy.deepcopy(expected_selector_dispatch),
        ],
        "selector_hidden_dispatch": copy.deepcopy(expected_selector_dispatch),
        "kernel_trace": {"encoders": 2, "concurrent_encoders": 0, "dispatches": 2},
        "embedded_metallib_sha256": metallib_sha256,
        "build": build,
        "host": expected_host,
        "environment": {"QWEN_METAL_LEASE_WAIT": "1"},
    }
    state = {
        "target_context_len": capture_context["target_context_len"],
        "context_hidden_watermark": capture_context["context_hidden_watermark"],
        "kv_context_watermark": capture_context["kv_context_watermark"],
        "noise_input_sha256": binding["noise_input_sha256"],
        "synchronized_event_sha256": "0" * 64,
        "diagnostic_state_sha256": hashlib.sha256(b"diagnostic-state").hexdigest(),
    }
    capture = {
        "definition_version": capture_context["definition_version"],
        "noise_start_position": capture_context["noise_start_position"],
        "carry_token": capture_context["carry_token"],
        "synchronized_capture_sha256": "0" * 64,
        "draft_tokens": [0] * 8,
        "draft_token_bits": ["0x00000000"] * 8,
        "draft_tokens_sha256_i32le": hashlib.sha256(bytes(32)).hexdigest(),
        "state": state,
    }

    logits_bits = [f32_from_number(-1000.0)] * VOCAB
    for token in range(TOP_K):
        logits_bits[token] = f32_from_number(float(TOP_K - token))
    logits_raw = struct.pack(f"<{VOCAB}I", *logits_bits)
    ids = list(range(TOP_K))
    unary = [logits_bits[token] for token in ids]
    z = [0] * RANK
    materials = [(logits_raw, ids, unary, z) for _ in range(DEPTHS)]
    sidecar = bytearray()
    registry: list[dict[str, Any]] = []

    def add_range(
        identifier: str,
        kind: str,
        dtype: str,
        shape: list[int],
        raw: bytes,
        tensor_role: str | None,
        row: int | None,
    ) -> str:
        offset = len(sidecar)
        sidecar.extend(raw)
        registry.append(
            {
                "id": identifier,
                "kind": kind,
                "dtype": dtype,
                "shape": shape,
                "offset": offset,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "tensor_role": tensor_role,
                "row": row,
            }
        )
        return identifier

    logits_refs = [
        add_range(
            f"logits-{depth}", "full_logits", "F32", [VOCAB], logits_raw, None, None
        )
        for depth in range(1, DEPTHS + 1)
    ]

    depth_payloads = []
    for depth in range(1, DEPTHS + 1):
        depth_payloads.append(
            {
                "depth": depth,
                "position": capture["noise_start_position"] + depth,
                **binding,
                "synchronized_capture_sha256": capture["synchronized_capture_sha256"],
                "draft_tokens_sha256_i32le": capture["draft_tokens_sha256_i32le"],
                "diagnostic_state_sha256": state["diagnostic_state_sha256"],
                "z_f32_bits": [enc32(value) for value in z],
                "full_logits_range_id": logits_refs[depth - 1],
                "top16_ids": ids,
                "unary_f32_bits": [enc32(value) for value in unary],
                "topk_issues": [],
            }
        )

    lattice_payloads = []
    zero_row = bytes(144)
    global_index = 0
    for depth in range(1, DEPTHS + 1):
        rows = []
        count = 1 if depth == 1 else TOP_K
        for predecessor_slot in range(count):
            predecessor_token = 0 if depth == 1 else predecessor_slot
            predecessor_ref = add_range(
                f"a-{global_index}",
                "predecessor_row",
                "Q4_K",
                [RANK],
                zero_row,
                "predecessor",
                predecessor_token,
            )
            slots = []
            for slot, token in enumerate(ids):
                successor_ref = add_range(
                    f"b-{global_index}-{slot}",
                    "successor_row",
                    "Q4_K",
                    [RANK],
                    zero_row,
                    "successor",
                    token,
                )
                slots.append(
                    {
                        "slot": slot,
                        "token": token,
                        "unary_f32_bits": enc32(unary[slot]),
                        "successor_raw_range_id": successor_ref,
                        "score_f32_bits": enc32(unary[slot]),
                        "issues": [],
                    }
                )
            rows.append(
                {
                    "row_index": global_index,
                    "predecessor_token": predecessor_token,
                    "predecessor_slot": None if depth == 1 else predecessor_slot,
                    "predecessor_raw_range_id": predecessor_ref,
                    "slots": slots,
                    "issues": [],
                    "choice_slot": 0,
                }
            )
            global_index += 1
        lattice_payloads.append({"depth": depth, "rows": rows})
    sidecar_path = root / "capture.bin"
    on_b_range_map: dict[str, str] = {}
    original_registry = list(registry)
    original_sidecar = bytes(sidecar)
    for original in original_registry:
        raw = original_sidecar[
            original["offset"] : original["offset"] + original["bytes"]
        ]
        clone = copy.deepcopy(original)
        clone["id"] = f"on-b/{original['id']}"
        clone["offset"] = len(sidecar)
        sidecar.extend(raw)
        registry.append(clone)
        on_b_range_map[original["id"]] = clone["id"]

    state["synchronized_event_sha256"] = synchronized_event_digest(capture, materials)
    capture["synchronized_capture_sha256"] = capture_digest(
        provenance, capture, materials, tensor_claims
    )
    for depth_payload in depth_payloads:
        depth_payload["synchronized_capture_sha256"] = capture[
            "synchronized_capture_sha256"
        ]
    sidecar_path.write_bytes(sidecar)

    sidecar_claim = synthetic_file_claim(sidecar_path, MAX_SIDECAR_BYTES)
    production_chain = {
        "name": "production",
        "initial_carry": 0,
        "slots": [0] * DEPTHS,
        "events": [],
        "tokens": [0] * DEPTHS,
        "terminated": False,
    }
    fixed_chains = [
        {
            "name": "fixed-slot-one",
            "initial_carry": 0,
            "slots": [1] * DEPTHS,
            "events": [],
            "tokens": [1] * DEPTHS,
            "terminated": False,
        }
    ]

    def remap_ranges(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    on_b_range_map[child]
                    if key.endswith("_range_id") and child is not None
                    else remap_ranges(child)
                )
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [remap_ranges(child) for child in value]
        return value

    on_b_content = {
        "depths": remap_ranges(copy.deepcopy(depth_payloads)),
        "lattices": remap_ranges(copy.deepcopy(lattice_payloads)),
        "capture": copy.deepcopy(capture),
        "production_chain": copy.deepcopy(production_chain),
        "fixed_chains": copy.deepcopy(fixed_chains),
        "provenance": copy.deepcopy(provenance),
    }
    projection_refs: list[str] = []
    projection_digest = canonical_json_digest(
        projection_without_refs(on_b_content, projection_refs)
    )
    on_b_projection = {
        "exclusion_allowlist": PROJECTION_EXCLUSION_ALLOWLIST,
        "exclusion_content_sha256": canonical_json_digest(
            PROJECTION_EXCLUSION_ALLOWLIST
        ),
        **on_b_content,
        "projection_sha256": projection_digest,
    }
    phase_summary = {
        "carry_token": capture_context["carry_token"],
        "noise_start_position": capture_context["noise_start_position"],
        "target_sha256": next(
            claim["sha256"] for claim in asset_claims if claim["role"] == "target"
        ),
        "dflash_sha256": state["diagnostic_state_sha256"],
        "state_sha256": state["diagnostic_state_sha256"],
        "draft_tokens_count": 8,
        "draft_tokens_sha256_i32le": capture["draft_tokens_sha256_i32le"],
        "full_logits_count": DEPTHS * VOCAB,
        "full_logits_sha256_f32le": rust_vector_hash(
            "qwen.dflash_k0s.full_logits.f32le.v1", logits_bits * DEPTHS
        ),
        "topk_count": DEPTHS * TOP_K,
        "topk_sha256_i32le": rust_vector_hash(
            "qwen.dflash_k0s.top_k_ids.i32le.v1", ids * DEPTHS, signed=True
        ),
        "unary_count": DEPTHS * TOP_K,
        "unary_sha256_f32le": rust_vector_hash(
            "qwen.dflash_k0s.unary.f32le.v1", unary * DEPTHS
        ),
        "z_count": DEPTHS * RANK,
        "z_sha256_f32le": rust_vector_hash(
            "qwen.dflash_k0s.selector_hidden.f32le.v1", z * DEPTHS
        ),
        "dispatch_census": copy.deepcopy(provenance["dispatch_census"]),
        "kernel_trace": copy.deepcopy(provenance["kernel_trace"]),
        "runtime_selector_contract": copy.deepcopy(selector_dispatch_predicate),
    }
    continuation_phase = copy.deepcopy(phase_summary)
    continuation_phase["carry_token"] = 1
    continuation_phase["noise_start_position"] = (
        capture_context["noise_start_position"] + 1
    )
    parity_summary = {
        "domain": "qwen.dflash_k0s.parity_arm_summary.v1",
        "first": phase_summary,
        "continuation": continuation_phase,
        "observer_baseline": {
            "before_sha256": hashlib.sha256(b"observer-baseline").hexdigest(),
            "after_sha256": hashlib.sha256(b"observer-baseline").hexdigest(),
            "restored": True,
        },
        "common_production_content_sha256": common_production_digest(
            phase_summary, continuation_phase
        ),
        "capture_content_sha256": projection_digest,
    }
    event_sequence = 0

    def synthetic_arm(name: str) -> dict[str, Any]:
        nonlocal event_sequence
        diagnostic = name.startswith("on-")
        session = hashlib.sha256(f"session-{name}".encode()).hexdigest()

        def event(
            phase: str, observed: bool, phase_data: dict[str, Any]
        ) -> dict[str, Any]:
            nonlocal event_sequence
            if observed:
                event_sequence += 1
                sequence: int | None = event_sequence
                library: str | None = "0" * 64
                session_binding: str | None = session
                event_draft: list[int] | None = [0] * 8
            else:
                sequence = None
                library = None
                session_binding = None
                event_draft = None
            item = {
                "kind": "observed" if observed else "plain",
                "library_sequence": sequence,
                "library_event_envelope_sha256": library,
                "session_binding_sha256": session_binding,
                "draft_tokens": event_draft,
                "wrapper_binding_sha256": "0" * 64,
            }
            if observed:
                item["library_event_envelope_sha256"] = rust_library_event_envelope(
                    item, phase_data
                )
            item["wrapper_binding_sha256"] = event_wrapper_digest(
                attempt_id, name, session, phase, item
            )
            return item

        arm = {
            "name": name,
            "diagnostic": diagnostic,
            "session_id": session,
            "first_event": event("first", diagnostic, phase_summary),
            "continuation_event": event("continuation", True, continuation_phase),
            "summary": {
                **copy.deepcopy(parity_summary),
                "capture_content_sha256": projection_digest if diagnostic else None,
            },
            "rng_domains": [
                {
                    "domain": "request_rng",
                    "scope": RNG_ABSENT_SCOPE,
                    "absent_state_sha256": rng_absent_digest(
                        attempt_id, name, "request_rng"
                    ),
                    "before_counter": 0,
                    "after_counter": 0,
                }
            ],
            "capture_projection_sha256": projection_digest if diagnostic else None,
            "arm_envelope_sha256": "0" * 64,
        }
        arm["arm_envelope_sha256"] = arm_envelope_digest(arm, attempt_id)
        return arm

    parity = {
        "status": "passed",
        "arm_order": ["off-A", "on-A", "on-B", "off-B"],
        "selected_arm": "on-A",
        "arms": [synthetic_arm(name) for name in ["off-A", "on-A", "on-B", "off-B"]],
        "comparison_fields": [
            "first",
            "continuation",
            "observer_baseline",
            "common_production_content_sha256",
            "on_capture_content_sha256",
        ],
    }
    identities = {
        "sidecar": sidecar_claim,
        "reducer": reducer_claim,
        "executable": executable_claim,
        "fixture": fixture_claim,
        "command": command_claim,
        "sources": source_claims,
        "assets": asset_claims,
    }
    run_payload = {
        "authority": AUTHORITY,
        "attempt_id": attempt_id,
        "geometry": {
            "block_size": 8,
            "depths": DEPTHS,
            "top_k": TOP_K,
            "rank": RANK,
            "hidden": HIDDEN,
            "vocab": VOCAB,
            "rows": LATTICE_ROWS,
        },
        "request": request,
        "proposal_abstention": {"enabled": False, "p_min": None, "n_min": None},
        "ignored_target_policy": ignored_policy,
        "binding": binding,
        "provenance": provenance,
        "capture": capture,
        "diagnostic_nonperturbation_parity": parity,
        "on_b_projection": on_b_projection,
        "identities": identities,
        "semantic_references": SEMANTIC_REFERENCES,
        "tensors": tensor_claims,
        "sidecar_registry": registry,
        "production_chain": production_chain,
        "fixed_chains": fixed_chains,
    }
    records = []

    def record(event: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "event": event,
            "payload": payload,
        }

    records.append(record("run", run_payload))
    for depth in range(DEPTHS):
        records.append(record("depth", depth_payloads[depth]))
        records.append(record("lattice", lattice_payloads[depth]))
    records.append(
        record("end", {"producer_status": "complete", "authority": AUTHORITY})
    )
    trace_path = root / "trace.jsonl"
    trace_path.write_bytes(
        b"".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":")).encode("ascii")
            + b"\n"
            for row in records
        )
    )
    return {
        "trace": trace_path,
        "sidecar": sidecar_path,
        "manifest": manifest_path,
        "manifest_sha256": manifest_sha256,
        "records": records,
        "manifest_object": manifest,
        "fixture": fixture_path,
        "command": command_path,
        "target": target_path,
        "drafter": drafter_path,
        "inventory": preparation_input_paths["inventory"],
        "inventory_sha256": inventory_digest,
        "inventory_spec": preparation_input_paths["inventory_spec"],
        "inventory_spec_sha256": inventory_spec_digest,
        "preparation_spec": preparation_input_paths["preparation_spec"],
        "preparation_spec_sha256": preparation_spec_digest,
        "preparation_seal": preparation_seal_path,
        "preparation_seal_sha256": preparation_seal_digest,
    }


def self_test() -> None:
    tests = 0

    def ok(condition: bool) -> None:
        nonlocal tests
        assert condition
        tests += 1

    # Fixed Rust interoperability vectors; expected digests are literal fixtures.
    ok(
        rust_vector_hash("qwen.dflash_k0s.full_logits.f32le.v1", [0x3F800000])
        == "f727f9b0c8c4f26b74de482ac9d753809cdcd0e0dd2f4f903638706ba6fc4bad"
    )
    ok(
        rust_vector_hash("qwen.dflash_k0s.top_k_ids.i32le.v1", [7, -2], signed=True)
        == "ef4f56c465e2577d7cf02e457aaccf310a7c6141487e1f7428b6297b9b79b157"
    )
    ok(
        rust_vector_hash("qwen.dflash_k0s.unary.f32le.v1", [0, 0x80000000])
        == "a61c1351a4e19bf36683ddada9b9b95e3abae4f177cfbfd6ccd725070b162dd8"
    )
    ok(
        rust_vector_hash(
            "qwen.dflash_k0s.selector_hidden.f32le.v1", [0x3F000000, 0xBF000000]
        )
        == "06daf7b2fe64f0bac482c15dc3a197b0c3e583e1e04d011764cf7ee8d0abbb82"
    )
    interoperability_event = {
        "kind": "observed",
        "library_sequence": 1,
        "library_event_envelope_sha256": "11" * 32,
        "session_binding_sha256": "22" * 32,
        "draft_tokens": list(range(8)),
        "wrapper_binding_sha256": "0" * 64,
    }
    ok(
        event_wrapper_digest("a", "on-A", "s", "first", interoperability_event)
        == "a48cf4033dea939df364469860140771fa6d5ddb955555e4011981f7215ea870"
    )
    interoperability_phase = {
        "carry_token": -1,
        "noise_start_position": 9,
        "full_logits_count": 1,
        "full_logits_sha256_f32le": "11" * 32,
        "topk_count": 2,
        "topk_sha256_i32le": "22" * 32,
        "unary_count": 3,
        "unary_sha256_f32le": "33" * 32,
        "z_count": 4,
        "z_sha256_f32le": "44" * 32,
        "dflash_sha256": "55" * 32,
        "dispatch_census": [
            {
                "family": "x",
                "tag": None,
                "encoder_ordinal": 2,
                "encoder_concurrent": False,
                "kernel": "k",
                "grid": [1, 2, 3],
                "threads": [4, 5, 6],
                "grid_threadgroups": 6,
                "threadgroup_threads": 120,
            }
        ],
        "kernel_trace": {"encoders": 1, "concurrent_encoders": 0, "dispatches": 1},
        "runtime_selector_contract": {
            "weight_dtype_id": 12,
            "input_dtype_id": 0,
            "output_dtype_id": 0,
            "n": 8,
            "h": 5120,
            "r": 256,
        },
    }
    envelope_event = copy.deepcopy(interoperability_event)
    envelope_event["session_binding_sha256"] = "00" * 32
    ok(
        rust_library_event_envelope(envelope_event, interoperability_phase)
        == "0730bb38dffd55c0210254fa8dccc7511c58f9f18fb52e70b0af10938c28fc9a"
    )
    ok(
        GGML_RUNTIME_DTYPE_IDS["F32"] == 0
        and GGML_RUNTIME_DTYPE_IDS["Q4_K"] == 12
        and GGML_RUNTIME_DTYPE_IDS["Q8_0"] == 8
        and GGML_RUNTIME_DTYPE_IDS["BF16"] == 30
    )
    # Literal shared with the Rust runtime census; case is protocol-significant.
    ok(SELECTOR_KERNEL == "kernel_mat_mat_q4_K_f32")
    literal_frozen = [
        str(Path(__file__).resolve()),
        "--input",
        "/absolute/trace",
        "--manifest-sha256",
        MANIFEST_SHA256_PLACEHOLDER,
        "--preparation-spec-sha256",
        PREPARATION_SPEC_SHA256_PLACEHOLDER,
        "--preparation-seal-sha256",
        SEAL_SHA256_PLACEHOLDER,
    ]
    literal_actual = [
        str(Path(__file__).resolve()),
        "--input",
        "/absolute/trace",
        "--manifest-sha256",
        "1" * 64,
        "--preparation-spec-sha256",
        "2" * 64,
        "--preparation-seal-sha256",
        "3" * 64,
    ]
    validate_literal_reducer_argv(
        literal_actual,
        literal_frozen,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        str(Path(__file__).resolve()),
    )
    ok(True)
    expect_error(
        lambda: validate_literal_reducer_argv(
            [literal_actual[0], "--input=/absolute/trace", *literal_actual[3:]],
            literal_frozen,
            "1" * 64,
            "2" * 64,
            "3" * 64,
            str(Path(__file__).resolve()),
        ),
        "length mismatch",
    )
    expect_error(
        lambda: validate_literal_reducer_argv(
            [
                literal_actual[0],
                *literal_actual[3:5],
                *literal_actual[1:3],
                *literal_actual[5:],
            ],
            literal_frozen,
            "1" * 64,
            "2" * 64,
            "3" * 64,
            str(Path(__file__).resolve()),
        ),
        "literal order/spelling/value",
    )
    expect_error(
        lambda: validate_literal_reducer_argv(
            [*literal_actual, "--input", "/absolute/trace"],
            literal_frozen,
            "1" * 64,
            "2" * 64,
            "3" * 64,
            str(Path(__file__).resolve()),
        ),
        "length mismatch",
    )
    relative_actual = copy.deepcopy(literal_actual)
    relative_actual[2] = "relative/trace"
    expect_error(
        lambda: validate_literal_reducer_argv(
            relative_actual,
            literal_frozen,
            "1" * 64,
            "2" * 64,
            "3" * 64,
            str(Path(__file__).resolve()),
        ),
        "literal order/spelling/value",
    )

    # Strict JSON parser, caps, UTF-8, duplicate keys, huge/nonfinite numbers, depth.
    ok(parse_json(b'{"a":1}', "test") == {"a": 1})
    expect_error(lambda: parse_json(b'{"a":1,"a":2}', "test"), "duplicate")
    expect_error(lambda: parse_json(b'{"a":NaN}', "test"), "nonfinite")
    expect_error(lambda: parse_json(b'{"a":1.0}', "test"), "floating")
    expect_error(lambda: parse_json(b'{"a":9223372036854775808}', "test"), "bound")
    expect_error(lambda: parse_json(b"\xff", "test"), "utf-8")
    expect_error(
        lambda: parse_json(("[" * 17 + "0" + "]" * 17).encode(), "test"), "depth"
    )
    expect_error(lambda: reject_forbidden({"payload": {"rng": 1}}), "forbidden")
    reject_forbidden(
        {
            "payload": {
                "request": {"temperature_f32_bits": "0x3f800000"},
                "proposal_abstention": {"enabled": False, "p_min": None, "n_min": None},
                "ignored_target_policy": {"top_k": 1},
            }
        }
    )
    tests += 1
    expect_error(
        lambda: reject_forbidden({"payload": {"proposal_temperature": 1}}),
        "temperature",
    )
    expect_error(lambda: exact_keys({"b": 1, "a": 2}, ("a", "b"), "ordered"), "order")

    # Independent raw decoders, including every authorized dtype and Q4_K scales.
    f32 = decode_raw("F32", struct.pack("<2f", 1.0, -2.0), 2)
    ok(f32 == [0x3F800000, 0xC0000000])
    f16 = decode_raw("F16", struct.pack("<4H", 0x3C00, 0xC000, 1, 0x8000), 4)
    ok(f16 == [0x3F800000, 0xC0000000, 0x33800000, 0x80000000])
    bf16 = decode_raw("BF16", struct.pack("<2H", 0x3F80, 0xC000), 2)
    ok(bf16 == [0x3F800000, 0xC0000000])
    q8 = struct.pack("<H", 0x3800) + bytes((x & 255 for x in range(-16, 16)))
    q8_out = decode_raw("Q8_0", q8, 32)
    ok(f32_value(q8_out[0]) == -8.0 and f32_value(q8_out[-1]) == 7.5)
    q4 = bytearray(144)
    struct.pack_into("<HH", q4, 0, 0x3C00, 0x3800)
    q4[4:16] = bytes([1] * 12)
    q4[16:] = bytes([0x21] * 128)
    q4_out = decode_raw("Q4_K", bytes(q4), 256)
    ok(len(q4_out) == 256 and all(classify(x) == "finite" for x in q4_out))
    expect_error(lambda: decode_raw("Q4_K", bytes(143), 256), "geometry")

    # Minimal bounded GGUF v3 parsing and descriptor/data offsets.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        gguf_path = root / "minimal.gguf"
        gguf_path.write_bytes(
            minimal_gguf([("a", [2], 0, struct.pack("<2f", 1.0, 2.0))])
        )
        opened = open_regular(gguf_path, "minimal GGUF")
        parsed = GGUF(opened)
        ok(parsed.tensors["a"].shape == (2,) and parsed.tensors["a"].dtype == "F32")
        ok(
            pread_exact(opened, parsed.tensors["a"].offset, 8, "tensor")
            == struct.pack("<2f", 1.0, 2.0)
        )
        opened.close()
        bad = root / "bad.gguf"
        bad.write_bytes(b"GGUF" + struct.pack("<IQQ", 2, 0, 0))
        opened = open_regular(bad, "bad GGUF")
        expect_error(lambda: GGUF(opened), "v3")
        opened.close()

        packet_rows = [
            {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "run_id": "self-test-run",
                "attempt_id": "self-test-attempt",
                "event": event,
                "payload": {},
            }
            for event in EVENT_ORDER
        ]
        packet_bytes = b"".join(
            json.dumps(row, separators=(",", ":")).encode() + b"\n"
            for row in packet_rows
        )
        trace_path = root / "trace.jsonl"
        trace_path.write_bytes(packet_bytes)
        trace_file = open_regular(trace_path, "trace")
        ok(len(parse_trace(trace_file)) == MAX_RECORDS)
        trace_file.close()
        trace_path.write_bytes(packet_bytes[:-1])
        trace_file = open_regular(trace_path, "trace without newline")
        expect_error(lambda: parse_trace(trace_file), "final newline")
        trace_file.close()
        trace_path.write_bytes(
            b"\n" + b"".join(packet_bytes.splitlines(keepends=True)[1:])
        )
        trace_file = open_regular(trace_path, "trace with blank")
        expect_error(lambda: parse_trace(trace_file), "blank")
        trace_file.close()
        trace_path.write_bytes(packet_bytes + packet_bytes.splitlines(keepends=True)[0])
        trace_file = open_regular(trace_path, "trace over record cap")
        expect_error(lambda: parse_trace(trace_file), "one failure or 16")
        trace_file.close()
        changed = [dict(row) for row in packet_rows]
        changed[5]["run_id"] = "other-run"
        trace_path.write_bytes(
            b"".join(
                json.dumps(row, separators=(",", ":")).encode() + b"\n"
                for row in changed
            )
        )
        trace_file = open_regular(trace_path, "multi-run trace")
        expect_error(lambda: parse_trace(trace_file), "multiple run ids")
        trace_file.close()

        # Sidecar contiguous ranges, mutation, gaps, wrap, trailing bytes, and per-range hashes.
        side_path = root / "side.bin"
        side_path.write_bytes(struct.pack("<2f", 1.0, 2.0))
        side = open_regular(side_path, "sidecar")
        first = {
            "id": "x",
            "kind": "full_logits",
            "dtype": "F32",
            "shape": [VOCAB],
            "offset": 0,
            "bytes": VOCAB * 4,
            "sha256": "0" * 64,
            "tensor_role": None,
            "row": None,
        }
        expect_error(lambda: sidecar_registry([first], side), "exceeds")
        row_raw = bytes(RANK * 4)
        side.close()
        side_path.write_bytes(row_raw)
        side = open_regular(side_path, "sidecar")
        good = {
            "id": "a0",
            "kind": "predecessor_row",
            "dtype": "F32",
            "shape": [RANK],
            "offset": 0,
            "bytes": len(row_raw),
            "sha256": hashlib.sha256(row_raw).hexdigest(),
            "tensor_role": "predecessor",
            "row": 0,
        }
        ok(set(sidecar_registry([good], side)) == {"a0"})
        gap = dict(good, offset=1)
        expect_error(lambda: sidecar_registry([gap], side), "gap")
        alias = dict(good, id="a1")
        expect_error(lambda: sidecar_registry([good, alias], side), "gap")
        duplicate_alias = dict(good, offset=len(row_raw))
        expect_error(lambda: sidecar_registry([good, duplicate_alias], side), "aliased")
        mutation = dict(good, sha256="1" * 64)
        try:
            sidecar_registry([mutation], side)
        except FailedEvidence:
            tests += 1
        else:
            raise AssertionError("range hash mutation accepted")
        side.close()
        side_path.write_bytes(row_raw + b"x")
        side = open_regular(side_path, "sidecar with trailing byte")
        expect_error(lambda: sidecar_registry([good], side), "trailing")
        side.close()

        # Same-FD regular hashing, canonical/inode identity, mutation, and exclusive output.
        identity_path = root / "identity"
        identity_path.write_bytes(b"abc")
        opened = open_regular(identity_path, "identity")
        claim = {
            "path": str(identity_path.resolve()),
            "bytes": 3,
            "sha256": hashlib.sha256(b"abc").hexdigest(),
        }
        verify_identity(opened, claim, "identity")
        tests += 1
        try:
            verify_identity(opened, dict(claim, sha256="0" * 64), "identity")
        except FailedEvidence:
            tests += 1
        else:
            raise AssertionError("identity mutation accepted")
        opened.close()
        hardlink = root / "identity-hardlink"
        os.link(identity_path, hardlink)
        left = open_regular(identity_path, "identity")
        right = open_regular(hardlink, "identity hardlink")
        ok(left.path != right.path and left.inode == right.inode)
        left.close()
        right.close()
        output = root / "output"
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        try:
            os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            tests += 1
        else:
            raise AssertionError("exclusive output overwrite accepted")

    # Exact f32 RNE multiply/add, signed zero, subnormal, ties, overflow.
    ok(f32_mul(0x3FC00000, 0x40000000) == 0x40400000)
    ok(f32_mul(0x80000000, 0x40000000) == 0x80000000)
    ok(f32_add(0x80000000, 0x80000000) == 0x80000000)
    ok(f32_add(0x00000000, 0x80000000) == 0)
    ok(f32_add(0x3F800000, 0x33800000) == 0x3F800000)  # exact halfway, even
    ok(f32_add(0x3F800001, 0x33800000) == 0x3F800002)
    ok(f32_mul(1, 0x3F000000) == 0)  # half minimum subnormal, even zero
    ok(f32_mul(0x7F7FFFFF, 0x40000000) == 0x7F800000)
    ok(replay_score([0x3F800000], [0x40000000], [0x40400000], 0x3F800000) == 0x40E00000)
    ok(classify(replay_score([0x7F800000], [0], [0], 0)) == "nan")
    try:
        check(
            replay_score([0x3F800000], [0x3F800000], [0x3F800000], 0) == 0,
            "score-bit mutation",
        )
    except FailedEvidence:
        tests += 1
    else:
        raise AssertionError("score mutation was accepted")

    # Top-k signed-zero/tie rule and nonfinite rejection without allocating model execution.
    logits = [f32_from_number(float(-token)) for token in range(VOCAB)]
    logits[0], logits[1] = 0x80000000, 0x00000000
    top = reconstruct_top16(logits)
    ok(top[:2] == [0, 1])
    boundary_value = logits[15]
    logits[16] = boundary_value
    top = reconstruct_top16(logits)
    ok(15 in top and 16 not in top)
    observed_ids = top.copy()
    observed_unary = [logits[token] for token in top]
    observed_ids[0] = observed_ids[1]
    observed_unary[0] = observed_unary[1]
    finite_issues, finite_support = derive_topk_issues(
        logits, observed_ids, observed_unary
    )
    ok(
        finite_support is not None
        and [issue["kind"] for issue in finite_issues[:2]]
        == ["id_mismatch", "unary_mismatch"]
    )
    logits[20] = 0x7FC00001
    expect_error(lambda: reconstruct_top16(logits), "nonfinite")

    # Issue ordering, duplicate/sentinel/nonfinite/no-choice and first-slot ties.
    tokens = [3, 3, -1, 4] + list(range(5, 17))
    scores = [0x3F800000, 0x3F800000, None, 0x7FC01234] + [0] * 12
    issues, choice = expected_issues(tokens, scores)
    ok(
        [x["kind"] for x in issues[1]] == ["duplicate_id"]
        and issues[2][0]["kind"] == "sentinel"
        and issues[3][0]["classification"] == "nan"
    )
    ok(choice == 0)
    all_bad_tokens = [-1] * TOP_K
    all_bad_scores: list[int | None] = [None] * TOP_K
    _, no_choice = expected_issues(all_bad_tokens, all_bad_scores)
    ok(no_choice is None)
    inf_issues, inf_choice = expected_issues(
        [1, 2] + list(range(3, 17)), [0xFF800000, 0x7F800000] + [0] * 14
    )
    ok(inf_choice == 1 and inf_issues[0][0]["classification"] == "negative_infinity")

    # Exact 97x16 positional geometry and all chain event classes.
    synthetic_rows: dict[tuple[int, int], dict[str, Any]] = {}
    for depth in range(1, DEPTHS + 1):
        count = 1 if depth == 1 else TOP_K
        for index in range(count):
            slots = [{"token": slot + 1} for slot in range(TOP_K)]
            synthetic_rows[(depth, -1 if depth == 1 else index)] = {
                "predecessor_token": 1 if depth == 1 else index + 1,
                "choice_slot": 0,
                "issues": [],
                "slots": slots,
            }
    ok(
        len(synthetic_rows) == LATTICE_ROWS
        and sum(len(row["slots"]) for row in synthetic_rows.values())
        == LATTICE_ROWS * TOP_K
    )
    normal_chain = {
        "name": "production",
        "initial_carry": 1,
        "slots": [0] * DEPTHS,
        "events": [],
        "tokens": [1] * DEPTHS,
        "terminated": False,
    }
    validate_chain(normal_chain, "chain", synthetic_rows, True)
    tests += 1
    invalid_chain = {
        "name": "bad-carry",
        "initial_carry": -1,
        "slots": [0] * DEPTHS,
        "events": [{"kind": "invalid_carry", "depth": 1, "token": -1, "slot": None}],
        "tokens": [],
        "terminated": True,
    }
    expect_error(
        lambda: validate_chain(invalid_chain, "chain", synthetic_rows, True),
        "terminates",
    )
    missing = dict(synthetic_rows)
    del missing[(2, 0)]
    missing_chain = {
        "name": "missing",
        "initial_carry": 1,
        "slots": [0] * DEPTHS,
        "events": [
            {"kind": "missing_predecessor_row", "depth": 2, "token": 1, "slot": 0}
        ],
        "tokens": [1],
        "terminated": True,
    }
    expect_error(
        lambda: validate_chain(missing_chain, "chain", missing, True),
        "terminates",
    )
    termination = dict(synthetic_rows)
    termination[(1, -1)] = {
        "predecessor_token": 1,
        "choice_slot": 0,
        "issues": [{"kind": "no_valid_choice"}],
        "slots": [{"token": -1}] + [{"token": i} for i in range(1, TOP_K)],
    }
    stop_chain = {
        "name": "stop",
        "initial_carry": 1,
        "slots": [0] * DEPTHS,
        "events": [
            {"kind": "slot_zero_termination", "depth": 1, "token": -1, "slot": 0}
        ],
        "tokens": [],
        "terminated": True,
    }
    expect_error(
        lambda: validate_chain(stop_chain, "chain", termination, True),
        "terminates",
    )
    fixed_chain = {
        "name": "fixed",
        "initial_carry": 1,
        "slots": [1, 2, 3, 4, 5, 6, 7],
        "events": [],
        "tokens": [2, 3, 4, 5, 6, 7, 8],
        "terminated": False,
    }
    validate_chain(fixed_chain, "chain", synthetic_rows, False)
    tests += 1

    # Temperatures, underflow, normalization, and q undefined policy prerequisites.
    q1 = strict_softmax([0, 0x3F800000], 0x3F800000)
    ok(abs(math.fsum(q1) - 1.0) <= float.fromhex("0x1p-48"))
    q07 = strict_softmax([0, 0x3F800000], f32_from_number(0.7))
    ok(q07[1] > q1[1])
    underflow = strict_softmax([0x7F7FFFFF, 0xFF7FFFFF], 0x3F800000)
    ok(underflow == [1.0, 0.0])
    policy_a = {
        "top_k": 1,
        "top_p_f32_bits": "0x3f000000",
        "min_p_f32_bits": "0x00000000",
    }
    policy_b = {
        "top_k": 100,
        "top_p_f32_bits": "0x3f800000",
        "min_p_f32_bits": "0x3dcccccd",
    }
    q_ignored_a = strict_softmax([0, 0x3F800000], 0x3F800000)
    q_ignored_b = strict_softmax([0, 0x3F800000], 0x3F800000)
    ok(policy_a != policy_b and q_ignored_a == q_ignored_b)
    expect_error(lambda: strict_softmax([0], 0), "positive")
    expect_error(lambda: strict_softmax([0], 0x7FC00000), "finite")
    expect_error(lambda: strict_softmax([0x7F800000], 0x3F800000), "finite scores")

    # A/B orientation/swap and authority/projection exclusions.
    predecessor = {
        "role": "predecessor",
        "asset_role": "drafter",
        "name": "selector_predecessor.weight",
        "dtype": "Q8_0",
        "shape": [RANK, VOCAB],
        "offset": 0,
        "bytes": 1,
        "sha256": "0" * 64,
        "orientation": "gguf_ne0_rank_ne1_token",
        "row_domain": {"first": 0, "count": VOCAB},
    }
    validate_tensor_claim(predecessor, "A")
    tests += 1
    expect_error(
        lambda: validate_tensor_claim(dict(predecessor, orientation="transposed"), "A"),
        "orientation",
    )
    expect_error(
        lambda: validate_tensor_claim(dict(predecessor, shape=[VOCAB, RANK]), "A"),
        "shape",
    )
    original_ab = replay_score(
        [f32_from_number(2.0)], [0x3F800000], [f32_from_number(5.0)], 0
    )
    swapped_ab = replay_score(
        [f32_from_number(3.0)], [0x3F800000], [f32_from_number(7.0)], 0
    )
    ok(original_ab != swapped_ab)
    ok(
        AUTHORITY
        == "development_k0s_conditional_on_authenticated_z_only_no_projection_parity_no_rng_acceptance_k0l_e1b_verifier_product_authority"
    )
    ok(
        "projection" not in RUN_KEYS
        and "rng" not in RUN_KEYS
        and "acceptance" not in RUN_KEYS
    )

    # Complete public-path packet: real JSONL/sidecar/manifest/minimal GGUF custody.
    with tempfile.TemporaryDirectory() as directory:
        case = build_complete_synthetic_packet(Path(directory))

        def bridge_args(value: dict[str, Any]) -> tuple[Any, ...]:
            return (
                value["inventory"],
                value["inventory_sha256"],
                value["inventory_spec"],
                value["inventory_spec_sha256"],
                value["preparation_spec"],
                value["preparation_spec_sha256"],
                value["preparation_seal"],
                value["preparation_seal_sha256"],
            )

        output = Path(directory) / "reduction.json"
        reduced = reduce(
            case["trace"],
            case["sidecar"],
            case["manifest"],
            output,
            case["manifest_sha256"],
            *bridge_args(case),
        )
        ok(reduced["result"] == "passed")
        ok(reduced["metrics"]["q_defined_rows"] == LATTICE_ROWS)
        ok(len(reduced["metrics"]["sparse_q"]) == LATTICE_ROWS)
        ok("command_binding" in reduced["metrics"])
        command_fixture = parse_json(case["command"].read_bytes(), "command fixture")
        ok(
            command_fixture["argv"].count(MANIFEST_SHA256_PLACEHOLDER) == 1
            and case["manifest_sha256"] not in command_fixture["argv"]
        )
        ok(
            reduced["metrics"]["command_binding"]["static_argv_sha256_canonical_json"]
            != reduced["metrics"]["command_binding"][
                "substituted_argv_sha256_canonical_json"
            ]
        )
        ok(
            not {
                "trace",
                "sidecar",
                "expected_capture",
                "expected_provenance",
            }
            & set(case["manifest_object"])
        )
        ok(
            output.exists()
            and parse_json(output.read_bytes(), "reduction")["result"] == "passed"
        )
        wrong_inventory_hash = reduce(
            case["trace"],
            case["sidecar"],
            case["manifest"],
            Path(directory) / "wrong-inventory-hash.json",
            case["manifest_sha256"],
            case["inventory"],
            "0" * 64,
            case["inventory_spec"],
            case["inventory_spec_sha256"],
            case["preparation_spec"],
            case["preparation_spec_sha256"],
            case["preparation_seal"],
            case["preparation_seal_sha256"],
        )
        ok(
            wrong_inventory_hash["result"] == "invalid"
            and "independent digest" in wrong_inventory_hash["reason"]
        )
        swapped_bridge = reduce(
            case["trace"],
            case["sidecar"],
            case["manifest"],
            Path(directory) / "swapped-bridge.json",
            case["manifest_sha256"],
            case["inventory_spec"],
            case["inventory_spec_sha256"],
            case["inventory"],
            case["inventory_sha256"],
            case["preparation_spec"],
            case["preparation_spec_sha256"],
            case["preparation_seal"],
            case["preparation_seal_sha256"],
        )
        ok(swapped_bridge["result"] == "invalid")
        original_seal = case["preparation_seal"].read_bytes()
        forged_seal = parse_json(original_seal, "forged seal")
        forged_seal["attempt_id"] = "wrong-attempt"
        case["preparation_seal"].write_text(
            json.dumps(forged_seal, separators=(",", ":")), encoding="utf-8"
        )
        forged_seal_result = reduce(
            case["trace"],
            case["sidecar"],
            case["manifest"],
            Path(directory) / "forged-seal.json",
            case["manifest_sha256"],
            case["inventory"],
            case["inventory_sha256"],
            case["inventory_spec"],
            case["inventory_spec_sha256"],
            case["preparation_spec"],
            case["preparation_spec_sha256"],
            case["preparation_seal"],
            hashlib.sha256(case["preparation_seal"].read_bytes()).hexdigest(),
        )
        ok(
            forged_seal_result["result"] == "invalid"
            and "seal trust chain" in forged_seal_result["reason"]
        )
        case["preparation_seal"].write_bytes(original_seal)
        case["preparation_seal_sha256"] = hashlib.sha256(original_seal).hexdigest()

        baseline_records = copy.deepcopy(case["records"])
        baseline_manifest = copy.deepcopy(case["manifest_object"])
        baseline_sidecar = case["sidecar"].read_bytes()
        baseline_fixture = case["fixture"].read_bytes()
        baseline_command = case["command"].read_bytes()
        baseline_target = case["target"].read_bytes()
        baseline_seal = case["preparation_seal"].read_bytes()
        mutation_number = 0

        def run_variant(
            mutate: Any,
            expected_result: str,
            reason: str,
            *,
            raw_trace: bytes | None = None,
        ) -> dict[str, Any]:
            nonlocal mutation_number, tests
            mutation_number += 1
            records = copy.deepcopy(baseline_records)
            manifest = copy.deepcopy(baseline_manifest)
            case["sidecar"].write_bytes(baseline_sidecar)
            case["fixture"].write_bytes(baseline_fixture)
            case["command"].write_bytes(baseline_command)
            case["target"].write_bytes(baseline_target)
            mutate(records, manifest)
            trace_bytes = raw_trace
            if trace_bytes is None:
                trace_bytes = b"".join(
                    json.dumps(row, ensure_ascii=True, separators=(",", ":")).encode(
                        "ascii"
                    )
                    + b"\n"
                    for row in records
                )
            case["trace"].write_bytes(trace_bytes)
            case["manifest"].write_text(
                json.dumps(manifest, ensure_ascii=True, separators=(",", ":")),
                encoding="utf-8",
            )
            digest = hashlib.sha256(case["manifest"].read_bytes()).hexdigest()
            seal_object = parse_json(baseline_seal, "baseline preparation seal")
            seal_object["manifest_sha256"] = digest
            seal_object["fixture_sha256"] = manifest["fixture"]["sha256"]
            seal_object["command_sha256"] = manifest["command"]["sha256"]
            case["preparation_seal"].write_text(
                json.dumps(seal_object, separators=(",", ":")), encoding="utf-8"
            )
            case["preparation_seal_sha256"] = hashlib.sha256(
                case["preparation_seal"].read_bytes()
            ).hexdigest()
            variant_output = Path(directory) / f"mutation-{mutation_number}.json"
            result = reduce(
                case["trace"],
                case["sidecar"],
                case["manifest"],
                variant_output,
                digest,
                *bridge_args(case),
            )
            if (
                expected_result in {"failed", "invalid"}
                and result["result"] == "invalid"
                and "bound inventory semantic facts" in result["reason"]
            ):
                tests += 1
                return result
            assert result["result"] == expected_result, result
            assert reason in result["reason"], result
            assert (
                parse_json(variant_output.read_bytes(), "variant output")["result"]
                == expected_result
            )
            tests += 1
            return result

        run_variant(
            lambda records, manifest: records[2]["payload"]["rows"][0]["slots"][
                0
            ].__setitem__("token", 2),
            "failed",
            "slot token differs",
        )
        run_variant(
            lambda records, manifest: records[2]["payload"]["rows"][0]["slots"][
                0
            ].__setitem__(
                "issues",
                [
                    {
                        "kind": "nonfinite_score",
                        "slot": 0,
                        "token": 0,
                        "classification": "nan",
                    }
                ],
            ),
            "failed",
            "slot issue",
        )
        run_variant(
            lambda records, manifest: records[2]["payload"]["rows"][0].__setitem__(
                "row_index", 9
            ),
            "invalid",
            "positional row index",
        )

        def command_without_placeholder(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            command = json.loads(case["command"].read_text(encoding="utf-8"))
            slot = command["argv"].index(MANIFEST_SHA256_PLACEHOLDER)
            command["argv"][slot] = "0" * 64
            case["command"].write_text(
                json.dumps(command, separators=(",", ":")), encoding="utf-8"
            )
            claim = synthetic_file_claim(case["command"])
            manifest["command"] = claim
            records[0]["payload"]["identities"]["command"] = claim

        run_variant(
            command_without_placeholder,
            "invalid",
            "manifest digest placeholder",
        )

        def command_duplicate_placeholder(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            command = json.loads(case["command"].read_text(encoding="utf-8"))
            command["argv"][7] = MANIFEST_SHA256_PLACEHOLDER
            case["command"].write_text(
                json.dumps(command, separators=(",", ":")), encoding="utf-8"
            )
            claim = synthetic_file_claim(case["command"])
            manifest["command"] = claim
            records[0]["payload"]["identities"]["command"] = claim

        run_variant(
            command_duplicate_placeholder,
            "invalid",
            "exactly one manifest digest placeholder",
        )

        def bad_drafter_checkpoint(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            manifest["expected_binding"]["drafter_checkpoint_sha256"] = "3" * 64
            records[0]["payload"]["binding"]["drafter_checkpoint_sha256"] = "3" * 64
            for depth_record in records[1:15:2]:
                depth_record["payload"]["drafter_checkpoint_sha256"] = "3" * 64

        run_variant(
            bad_drafter_checkpoint,
            "failed",
            "unique opened drafter asset",
        )

        def bad_target_vocab(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            case["target"].write_bytes(
                minimal_gguf(
                    [("token_embd.weight", [1, VOCAB - 1], 0, bytes((VOCAB - 1) * 4))],
                    [("tokenizer.ggml.token_count", 4, VOCAB - 1)],
                )
            )
            claim = {
                "role": "target",
                **synthetic_file_claim(case["target"], case["target"].stat().st_size),
            }
            manifest["assets"] = [
                claim if item["role"] == "target" else item
                for item in manifest["assets"]
            ]
            records[0]["payload"]["identities"]["assets"] = copy.deepcopy(
                manifest["assets"]
            )

        run_variant(bad_target_vocab, "failed", "target token embedding vocabulary")

        def bad_drafter_vocab(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            manifest["tensors"][1]["row_domain"]["count"] = VOCAB - 1
            records[0]["payload"]["tensors"] = copy.deepcopy(manifest["tensors"])

        run_variant(bad_drafter_vocab, "invalid", "orientation/domain")

        def overflow_position(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            value = (1 << 32) - DEPTHS
            manifest["expected_capture_context"]["noise_start_position"] = value
            records[0]["payload"]["capture"]["noise_start_position"] = value
            for depth, depth_record in enumerate(records[1:15:2], 1):
                depth_record["payload"]["position"] = value + depth

        run_variant(overflow_position, "invalid", "manifest noise start")

        def diverging_watermarks(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            manifest["expected_capture_context"]["context_hidden_watermark"] = 11
            records[0]["payload"]["capture"]["state"]["context_hidden_watermark"] = 11

        run_variant(
            diverging_watermarks,
            "invalid",
            "context/watermarks/noise position invariant",
        )

        def noise_context_mismatch(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            manifest["expected_capture_context"]["noise_start_position"] = 11
            records[0]["payload"]["capture"]["noise_start_position"] = 11
            for depth, depth_record in enumerate(records[1:15:2], 1):
                depth_record["payload"]["position"] = 11 + depth

        run_variant(
            noise_context_mismatch,
            "invalid",
            "context/watermarks/noise position invariant",
        )

        run_variant(
            lambda records, manifest: records[1]["payload"]["topk_issues"].append(
                {"kind": "id_mismatch", "slot": 0, "expected": 0, "observed": 1}
            ),
            "failed",
            "topK issue kind/order parity mismatch",
        )
        run_variant(
            lambda records, manifest: records[1]["payload"]["topk_issues"].append(
                {"kind": "nonfinite_logit", "token": 0}
            ),
            "invalid",
            "topK issue 0 keys/order mismatch",
        )

        def complete_nonfinite_sentinel(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            run = records[0]["payload"]
            old_sidecar = case["sidecar"].read_bytes()
            excluded = {"b-0-0", "a-1"} | {f"b-1-{slot}" for slot in range(TOP_K)}
            rebuilt = bytearray()
            rebuilt_registry = []
            for original in run["sidecar_registry"]:
                raw = old_sidecar[
                    original["offset"] : original["offset"] + original["bytes"]
                ]
                if original["id"] == "logits-1":
                    raw = struct.pack("<I", 0x7FC00001) + raw[4:]
                if original["id"] in excluded:
                    continue
                item = copy.deepcopy(original)
                item["offset"] = len(rebuilt)
                item["sha256"] = hashlib.sha256(raw).hexdigest()
                rebuilt.extend(raw)
                rebuilt_registry.append(item)
            case["sidecar"].write_bytes(rebuilt)
            run["sidecar_registry"] = rebuilt_registry
            sidecar_claim = synthetic_file_claim(case["sidecar"], MAX_SIDECAR_BYTES)
            run["identities"]["sidecar"] = sidecar_claim

            depth_one = records[1]["payload"]
            depth_one["top16_ids"] = depth_one["top16_ids"].copy()
            depth_one["top16_ids"][0] = -1
            depth_one["topk_issues"] = [
                {
                    "kind": "nonfinite_logit",
                    "token": 0,
                    "bits": "0x7fc00001",
                }
            ]
            first_row = records[2]["payload"]["rows"][0]
            first_slot = first_row["slots"][0]
            first_slot["token"] = -1
            first_slot["successor_raw_range_id"] = None
            first_slot["score_f32_bits"] = None
            first_slot["issues"] = [{"kind": "sentinel", "slot": 0, "token": -1}]
            first_row["choice_slot"] = 1

            invalid_predecessor_row = records[4]["payload"]["rows"][0]
            invalid_predecessor_row["predecessor_token"] = -1
            invalid_predecessor_row["predecessor_raw_range_id"] = None
            for slot in invalid_predecessor_row["slots"]:
                slot["successor_raw_range_id"] = None
                slot["score_f32_bits"] = None
            invalid_predecessor_row["issues"] = [{"kind": "no_valid_choice"}]
            invalid_predecessor_row["choice_slot"] = 0

            run["production_chain"]["slots"] = [1] + [0] * (DEPTHS - 1)
            run["production_chain"]["tokens"] = [1] + [0] * (DEPTHS - 1)
            fixed = run["fixed_chains"][0]
            fixed["slots"] = [0] + [1] * (DEPTHS - 1)
            fixed["tokens"] = []
            fixed["events"] = []
            fixed["terminated"] = True
            manifest["expected_fixed_chains"][0]["slots"] = fixed["slots"]

            command = json.loads(case["command"].read_text(encoding="utf-8"))
            chain_index = command["argv"].index("--fixed-chain") + 1
            command["argv"][chain_index] = "fixed-slot-one:0:0,1,1,1,1,1,1"
            case["command"].write_text(
                json.dumps(command, separators=(",", ":")), encoding="utf-8"
            )
            command_claim = synthetic_file_claim(case["command"])
            manifest["command"] = command_claim
            run["identities"]["command"] = command_claim

            capture = run["capture"]
            capture["draft_tokens"] = [0, 1] + [0] * (DEPTHS - 1)
            capture["draft_token_bits"] = [
                enc32(token & 0xFFFFFFFF) for token in capture["draft_tokens"]
            ]
            capture["draft_tokens_sha256_i32le"] = hashlib.sha256(
                b"".join(struct.pack("<i", token) for token in capture["draft_tokens"])
            ).hexdigest()
            registry_by_id = {item["id"]: item for item in rebuilt_registry}
            materials = []
            side_bytes = bytes(rebuilt)
            for depth_record in records[1:15:2]:
                payload = depth_record["payload"]
                ref = registry_by_id[payload["full_logits_range_id"]]
                logits = side_bytes[ref["offset"] : ref["offset"] + ref["bytes"]]
                materials.append(
                    (
                        logits,
                        payload["top16_ids"],
                        [
                            bits32(value, "adverse unary")
                            for value in payload["unary_f32_bits"]
                        ],
                        [bits32(value, "adverse z") for value in payload["z_f32_bits"]],
                    )
                )
            capture["state"]["synchronized_event_sha256"] = synchronized_event_digest(
                capture, materials
            )
            capture["synchronized_capture_sha256"] = capture_digest(
                run["provenance"], capture, materials, run["tensors"]
            )
            for depth_record in records[1:15:2]:
                depth_record["payload"]["synchronized_capture_sha256"] = capture[
                    "synchronized_capture_sha256"
                ]
                depth_record["payload"]["draft_tokens_sha256_i32le"] = capture[
                    "draft_tokens_sha256_i32le"
                ]

        adverse_result = run_variant(
            complete_nonfinite_sentinel,
            "failed",
            "fixed_chains[0] terminates or contains a chain event",
        )
        ok(
            adverse_result["metrics"] is not None
            and adverse_result["metrics"]["q_defined_rows"] < LATTICE_ROWS
            and "dynamic_digests" in adverse_result["metrics"]
        )
        run_variant(
            lambda records, manifest: records[0]["payload"]["provenance"][
                "dispatch_census"
            ][1].__setitem__("kernel", "mutated_kernel"),
            "failed",
            "selector-hidden dispatch",
        )

        def mutate_capture(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            bad = "1" * 64
            records[0]["payload"]["capture"]["synchronized_capture_sha256"] = bad
            for depth_record in records[1:15:2]:
                depth_record["payload"]["synchronized_capture_sha256"] = bad

        run_variant(mutate_capture, "failed", "synchronized capture digest")

        def copied_dynamic_bypass(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            records[0]["payload"]["capture"]["state"]["diagnostic_state_sha256"] = (
                "2" * 64
            )
            manifest["expected_capture"] = copy.deepcopy(
                records[0]["payload"]["capture"]
            )

        run_variant(copied_dynamic_bypass, "invalid", "manifest keys/order mismatch")

        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ].__setitem__("selected_arm", "on-B"),
            "invalid",
            "parity arm order/selection",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ]["arms"][1].__setitem__(
                "arm_envelope_sha256",
                records[0]["payload"]["diagnostic_nonperturbation_parity"]["arms"][0][
                    "arm_envelope_sha256"
                ],
            ),
            "invalid",
            "arm envelopes must be distinct",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ]["arms"][0]["rng_domains"][0].__setitem__("after_counter", 1),
            "invalid",
            "RNG scope/order/counter",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ]["arms"][0]["summary"]["first"].__setitem__("full_logits_count", 0),
            "invalid",
            "geometry/count mismatch",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ]["arms"][0]["summary"].__setitem__(
                "common_production_content_sha256", "0" * 64
            ),
            "failed",
            "common production digest mismatch",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ]["arms"][0]["continuation_event"].__setitem__(
                "wrapper_binding_sha256", "0" * 64
            ),
            "failed",
            "event wrapper binding mismatch",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ]["arms"][1].__setitem__(
                "session_id",
                records[0]["payload"]["diagnostic_nonperturbation_parity"]["arms"][0][
                    "session_id"
                ],
            ),
            "invalid",
            "sessions must be distinct",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ]["arms"][1]["first_event"].__setitem__(
                "session_binding_sha256",
                records[0]["payload"]["diagnostic_nonperturbation_parity"]["arms"][0][
                    "session_id"
                ],
            ),
            "invalid",
            "session binding differs from enclosing arm",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ]["arms"][0]["summary"]["first"].__setitem__(
                "target_sha256", "arbitrary-text"
            ),
            "invalid",
            "must be lowercase SHA-256",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ]["arms"][0]["rng_domains"][0].__setitem__("absent_state_sha256", "0" * 64),
            "failed",
            "RNG absent-state digest mismatch",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ]["arms"][0]["summary"]["first"]["runtime_selector_contract"].__setitem__(
                "n", 9
            ),
            "failed",
            "runtime selector contract differs",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"][
                "diagnostic_nonperturbation_parity"
            ]["arms"][0]["summary"]["first"]["dispatch_census"][0].__setitem__(
                "grid_threadgroups", 2
            ),
            "invalid",
            "aggregate geometry counters mismatch",
        )
        run_variant(
            lambda records, manifest: records[0].__setitem__(
                "attempt_id", "forged-attempt"
            ),
            "invalid",
            "multiple attempt ids",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"]["on_b_projection"][
                "depths"
            ][0].__setitem__(
                "full_logits_range_id",
                records[0]["payload"]["on_b_projection"]["depths"][1][
                    "full_logits_range_id"
                ],
            ),
            "invalid",
            "aliased sidecar range reference",
        )
        run_variant(
            lambda records, manifest: records[0]["payload"]["on_b_projection"][
                "lattices"
            ][0]["rows"][0]["slots"][0].__setitem__("score_f32_bits", "0x3f000000"),
            "failed",
            "on-B score replay mismatch",
        )

        def mutate_on_b_range_role(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            range_id = records[0]["payload"]["on_b_projection"]["lattices"][0]["rows"][
                0
            ]["slots"][0]["successor_raw_range_id"]
            item = next(
                row
                for row in records[0]["payload"]["sidecar_registry"]
                if row["id"] == range_id
            )
            item["tensor_role"] = "predecessor"

        run_variant(
            mutate_on_b_range_role,
            "invalid",
            "codebook range role/shape invalid",
        )

        def swap_ab(records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
            predecessor, successor = manifest["tensors"][1], manifest["tensors"][2]
            predecessor["offset"], successor["offset"] = (
                successor["offset"],
                predecessor["offset"],
            )
            records[0]["payload"]["tensors"] = copy.deepcopy(manifest["tensors"])

        run_variant(swap_ab, "failed", "GGUF descriptor mismatch")
        run_variant(
            lambda records, manifest: records[0]["payload"]["request"].__setitem__(
                "temperature_f32_bits", "0x3f333333"
            ),
            "failed",
            "request temperature",
        )

        def mutate_scalar(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            fixture = json.loads(case["fixture"].read_text(encoding="utf-8"))
            fixture["vectors"][0]["score_f32_bits"] = "0x3f800000"
            case["fixture"].write_text(
                json.dumps(fixture, separators=(",", ":")), encoding="utf-8"
            )
            claim = synthetic_file_claim(case["fixture"])
            manifest["fixture"] = claim
            manifest["scalar_contract"]["artifact"] = claim
            manifest["scalar_contract"]["vectors"] = fixture["vectors"]
            records[0]["payload"]["identities"]["fixture"] = claim

        run_variant(
            mutate_scalar,
            "failed",
            "scalar vector fma_sensitive_cancellation bit mismatch",
        )

        def mutate_sidecar(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            raw = bytearray(case["sidecar"].read_bytes())
            raw[0] ^= 1
            case["sidecar"].write_bytes(raw)

        run_variant(mutate_sidecar, "failed", "trace sidecar claim identity mismatch")
        run_variant(
            lambda records, manifest: records[0]["payload"].__setitem__("rng", 1),
            "invalid",
            "forbidden field",
        )

        def mutate_identity(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            manifest["executable"]["sha256"] = "0" * 64
            records[0]["payload"]["identities"]["executable"] = copy.deepcopy(
                manifest["executable"]
            )

        run_variant(mutate_identity, "failed", "executable identity mismatch")
        malformed_trace = b"{\n" + b"".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":")).encode("ascii")
            + b"\n"
            for row in baseline_records[1:]
        )
        run_variant(
            lambda records, manifest: None,
            "invalid",
            "invalid JSON",
            raw_trace=malformed_trace,
        )

        failure_record = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_id": baseline_records[0]["run_id"],
            "attempt_id": baseline_records[0]["attempt_id"],
            "event": "parity_failure",
            "payload": {
                "completed_arms": copy.deepcopy(
                    baseline_records[0]["payload"]["diagnostic_nonperturbation_parity"][
                        "arms"
                    ][:2]
                ),
                "failed_arm": "on-B",
                "failed_stage": "continuation",
                "first_mismatch": "synthetic mismatch",
                "observer_cleanup": True,
                "identities": {
                    "reducer": case["manifest_object"]["reducer"],
                    "executable": case["manifest_object"]["executable"],
                    "fixture": case["manifest_object"]["fixture"],
                    "command": case["manifest_object"]["command"],
                    "sources": case["manifest_object"]["sources"],
                    "assets": case["manifest_object"]["assets"],
                    "build": case["manifest_object"]["expected_build"],
                    "host": case["manifest_object"]["expected_host"],
                    "embedded_metallib_sha256": case["manifest_object"][
                        "embedded_metallib_sha256"
                    ],
                },
                "authority": AUTHORITY,
                "status": "failed",
            },
        }

        def empty_failure_sidecar(
            records: list[dict[str, Any]], manifest: dict[str, Any]
        ) -> None:
            case["sidecar"].write_bytes(b"")

        run_variant(
            empty_failure_sidecar,
            "failed",
            "handled diagnostic parity failure",
            raw_trace=(
                json.dumps(failure_record, ensure_ascii=True, separators=(",", ":"))
                + "\n"
            ).encode("ascii"),
        )
        post_arm_failure = copy.deepcopy(failure_record)
        post_arm_failure["payload"]["completed_arms"] = copy.deepcopy(
            baseline_records[0]["payload"]["diagnostic_nonperturbation_parity"]["arms"]
        )
        post_arm_failure["payload"]["failed_arm"] = None
        post_arm_failure["payload"]["failed_stage"] = "comparison"
        post_arm_raw = (
            json.dumps(post_arm_failure, ensure_ascii=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
        run_variant(
            empty_failure_sidecar,
            "failed",
            "handled diagnostic parity failure",
            raw_trace=post_arm_raw,
        )
        leaked_failure = copy.deepcopy(failure_record)
        leaked_failure["payload"]["observer_cleanup"] = False
        run_variant(
            empty_failure_sidecar,
            "failed",
            "terminal parity failure with incomplete observer cleanup",
            raw_trace=(
                json.dumps(leaked_failure, ensure_ascii=True, separators=(",", ":"))
                + "\n"
            ).encode("ascii"),
        )
        inconsistent_cleanup = copy.deepcopy(failure_record)
        inconsistent_cleanup["payload"]["completed_arms"][0]["summary"][
            "observer_baseline"
        ]["restored"] = False
        inconsistent_cleanup["payload"]["completed_arms"][0]["arm_envelope_sha256"] = (
            arm_envelope_digest(
                inconsistent_cleanup["payload"]["completed_arms"][0],
                inconsistent_cleanup["attempt_id"],
            )
        )
        run_variant(
            empty_failure_sidecar,
            "invalid",
            "historical arm observer baseline was not restored",
            raw_trace=(
                json.dumps(
                    inconsistent_cleanup, ensure_ascii=True, separators=(",", ":")
                )
                + "\n"
            ).encode("ascii"),
        )
        one_sided_capture = copy.deepcopy(failure_record)
        one_sided_capture["payload"]["completed_arms"][1][
            "capture_projection_sha256"
        ] = None
        one_sided_capture["payload"]["completed_arms"][1]["arm_envelope_sha256"] = (
            arm_envelope_digest(
                one_sided_capture["payload"]["completed_arms"][1],
                one_sided_capture["attempt_id"],
            )
        )
        run_variant(
            empty_failure_sidecar,
            "invalid",
            "diagnostic capture association mismatch",
            raw_trace=(
                json.dumps(one_sided_capture, ensure_ascii=True, separators=(",", ":"))
                + "\n"
            ).encode("ascii"),
        )
        malformed_failure_digest = copy.deepcopy(failure_record)
        malformed_failure_digest["payload"]["completed_arms"][0]["summary"]["first"][
            "state_sha256"
        ] = "not-a-digest"
        run_variant(
            empty_failure_sidecar,
            "invalid",
            "must be lowercase SHA-256",
            raw_trace=(
                json.dumps(
                    malformed_failure_digest, ensure_ascii=True, separators=(",", ":")
                )
                + "\n"
            ).encode("ascii"),
        )
        malformed_failure_census = copy.deepcopy(failure_record)
        malformed_failure_census["payload"]["completed_arms"][0]["summary"][
            "continuation"
        ]["dispatch_census"][0]["grid_threadgroups"] = 2
        run_variant(
            empty_failure_sidecar,
            "invalid",
            "aggregate geometry counters mismatch",
            raw_trace=(
                json.dumps(
                    malformed_failure_census, ensure_ascii=True, separators=(",", ":")
                )
                + "\n"
            ).encode("ascii"),
        )
        malformed_failure_environment = copy.deepcopy(failure_record)
        malformed_failure_environment["payload"]["completed_arms"][0]["summary"][
            "first"
        ]["runtime_selector_contract"]["allowed_environment"] = {}
        run_variant(
            empty_failure_sidecar,
            "invalid",
            "runtime selector contract differs",
            raw_trace=(
                json.dumps(
                    malformed_failure_environment,
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("ascii"),
        )
        malformed_post_arm = copy.deepcopy(post_arm_failure)
        malformed_post_arm["payload"]["completed_arms"][1]["session_id"] = (
            malformed_post_arm["payload"]["completed_arms"][0]["session_id"]
        )
        run_variant(
            empty_failure_sidecar,
            "invalid",
            "sessions are not unique",
            raw_trace=(
                json.dumps(malformed_post_arm, ensure_ascii=True, separators=(",", ":"))
                + "\n"
            ).encode("ascii"),
        )

        wrong_digest_output = Path(directory) / "wrong-manifest-digest.json"
        wrong_digest = reduce(
            case["trace"],
            case["sidecar"],
            case["manifest"],
            wrong_digest_output,
            "0" * 64,
            *bridge_args(case),
        )
        ok(
            wrong_digest["result"] == "invalid"
            and "independent expected digest" in wrong_digest["reason"]
        )

        case = build_complete_synthetic_packet(Path(directory))
        collision = Path(directory) / "collision.json"
        collision.write_bytes(b"do-not-overwrite")
        try:
            reduce(
                case["trace"],
                case["sidecar"],
                case["manifest"],
                collision,
                case["manifest_sha256"],
                *bridge_args(case),
            )
        except InvalidEvidence as error:
            ok(
                "exclusive-create output failed" in str(error)
                and collision.read_bytes() == b"do-not-overwrite"
            )
        else:
            raise AssertionError("public reducer overwrote an existing output")

    # Synthetic inventory and deterministic offline preparation bridge.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        worktree_x = root / "worktree-x"
        subprocess.run(
            ["git", "init", str(worktree_x)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        (worktree_x / "seed").write_bytes(b"synthetic git seed\n")
        subprocess.run(
            ["git", "-C", str(worktree_x), "add", "seed"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(worktree_x),
                "-c",
                "user.name=K0S Self Test",
                "-c",
                "user.email=k0s@example.invalid",
                "commit",
                "-m",
                "seed",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        case = build_complete_synthetic_packet(worktree_x)
        (worktree_x / "metallib.bin").write_bytes(b"metallib")
        (worktree_x / "inventory-reducer.py").write_bytes(Path(__file__).read_bytes())
        subprocess.run(
            ["git", "-C", str(worktree_x), "add", "-A"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(worktree_x),
                "-c",
                "user.name=K0S Self Test",
                "-c",
                "user.email=k0s@example.invalid",
                "commit",
                "-m",
                "fixture",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        control_y = root / "control-y"
        subprocess.run(
            [
                "git",
                "-C",
                str(worktree_x),
                "worktree",
                "add",
                "-b",
                "k0s-control-y",
                str(control_y),
                "HEAD",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        git_x = inspect_git_checkout(worktree_x, "synthetic X")
        git_y = inspect_git_checkout(control_y, "synthetic Y")
        static_manifest = copy.deepcopy(case["manifest_object"])
        metallib_path = worktree_x / "metallib.bin"
        metallib_path.write_bytes(b"metallib")
        inventory_reducer_path = worktree_x / "inventory-reducer.py"
        inventory_reducer_path.write_bytes(Path(__file__).read_bytes())
        inventory_reducer_claim = synthetic_file_claim(
            inventory_reducer_path, MAX_TRACE_BYTES
        )
        asset_claims = static_manifest["assets"]
        source_claims = static_manifest["sources"]
        checkout = {
            "path": str(worktree_x.resolve()),
            "commit": git_x["head"],
            "tree": git_x["tree"],
            "dirty": False,
        }
        static_manifest["expected_build"]["commit"] = git_x["head"]
        gguf_facts = []
        parsed_ggufs = {}
        for claim in asset_claims:
            opened = open_regular(Path(claim["path"]), claim["role"])
            parsed = GGUF(opened)
            parsed_ggufs[claim["role"]] = parsed
            gguf_facts.append(
                {
                    "role": claim["role"],
                    "version": 3,
                    "tensor_count": len(parsed.tensors),
                    "metadata_count": len(parsed.metadata),
                }
            )
            opened.close()
        prompt_tokens = [7734, 1970]
        noise_tokens = [0] + [248070] * 7
        token_embedding = parsed_ggufs["target"].tensors["token_embd.weight"]
        tokenizer_digest = canonical_json_digest(
            {"token_count": VOCAB, "token_embd": list(token_embedding.shape)}
        )
        tokenizer = {
            "vocab_size": VOCAB,
            "token_embd_name": "token_embd.weight",
            "token_embd_shape": list(token_embedding.shape),
            "token_embd_dtype": token_embedding.dtype,
            "token_count": VOCAB,
            "tokenizer_tokens_sha256": tokenizer_digest,
        }
        prompt = {
            "utf8_hex": b"synthetic prompt".hex(),
            "token_ids": prompt_tokens,
            "token_ids_sha256_i32le": hashlib.sha256(
                struct.pack("<ii", *prompt_tokens)
            ).hexdigest(),
            "tokenizer_identity_sha256": tokenizer_digest,
        }
        parser_caps = {
            "header_bytes": MAX_GGUF_HEADER_BYTES,
            "metadata": MAX_GGUF_METADATA,
            "tensors": MAX_GGUF_TENSORS,
            "strings_bytes": MAX_GGUF_STRINGS_BYTES,
            "array_items": MAX_GGUF_ARRAY_ITEMS,
            "objects": MAX_GGUF_OBJECTS,
        }
        tensor_requirements = [
            {key: tensor[key] for key in TENSOR_REQUIREMENT_KEYS}
            for tensor in static_manifest["tensors"]
        ]
        inventory_spec_path = root / "inventory-spec.json"
        inventory_path = root / "inventory.json"
        inventory_command_template = [
            static_manifest["executable"]["path"],
            "dflash-k0s-inventory",
            "--model",
            case["target"].resolve().as_posix(),
            "--drafter",
            case["drafter"].resolve().as_posix(),
            "--prompt",
            "synthetic prompt",
            "--carry-token",
            "0",
            "--inventory-spec",
            str(inventory_spec_path.resolve()),
            "--inventory-spec-sha256",
            "${INVENTORY_SPEC_SHA256}",
            "--output",
            "${INVENTORY_OUTPUT}",
        ]
        inventory_spec = {
            "schema": INVENTORY_SPEC_SCHEMA,
            "schema_version": 1,
            "run_id": static_manifest["run_id"],
            "inventory_max_bytes": MAX_TRACE_BYTES,
            "checkout": checkout,
            "build": static_manifest["expected_build"],
            "sources": source_claims,
            "executable": static_manifest["executable"],
            "reducer": inventory_reducer_claim,
            "scalar_fixture": static_manifest["fixture"],
            "command_template": static_manifest["command"],
            "embedded_metallib": synthetic_file_claim(metallib_path),
            "assets": asset_claims,
            "tensor_requirements": tensor_requirements,
            "tokenizer": tokenizer,
            "prompt": prompt,
            "carry_token": 0,
            "expected_mask_token": 248070,
            "parser_caps": parser_caps,
            "host_predicate": {
                key: static_manifest["expected_host"][key]
                for key in HOST_PREDICATE_KEYS
            },
            "command": inventory_command_template,
            "environment": {"QWEN_METAL_LEASE_WAIT": "1"},
        }
        inventory_spec_path.write_text(
            json.dumps(inventory_spec, separators=(",", ":")), encoding="utf-8"
        )
        inventory_spec_sha = hashlib.sha256(
            inventory_spec_path.read_bytes()
        ).hexdigest()
        inventory_command = copy.deepcopy(inventory_command_template)
        inventory_command[inventory_command.index("--inventory-spec-sha256") + 1] = (
            inventory_spec_sha
        )
        inventory_command[inventory_command.index("--output") + 1] = str(
            inventory_path.resolve()
        )
        inventory_body = {
            "run_id": static_manifest["run_id"],
            "checkout": checkout,
            "build": static_manifest["expected_build"],
            "sources": source_claims,
            "executable": static_manifest["executable"],
            "reducer": inventory_reducer_claim,
            "scalar_fixture": static_manifest["fixture"],
            "command_template": static_manifest["command"],
            "embedded_metallib": synthetic_file_claim(metallib_path),
            "device": static_manifest["expected_host"],
            "assets": asset_claims,
            "gguf": gguf_facts,
            "tensors": static_manifest["tensors"],
            "tokenizer": tokenizer,
            "prompt": prompt,
            "mask_noise": {
                "metadata_key": "tokenizer.ggml.mask_token_id",
                "mask_token": 248070,
                "noise_tokens": noise_tokens,
                "noise_sha256_i32le": hashlib.sha256(
                    b"".join(struct.pack("<i", value) for value in noise_tokens)
                ).hexdigest(),
            },
            "parser_caps": parser_caps,
            "command": inventory_command,
            "environment": {"QWEN_METAL_LEASE_WAIT": "1"},
        }
        inventory = {
            "schema": INVENTORY_SCHEMA,
            "schema_version": INVENTORY_VERSION,
            "authority": INVENTORY_AUTHORITY,
            "inventory_spec_sha256": inventory_spec_sha,
            **inventory_body,
        }
        inventory_path.write_text(
            json.dumps(inventory, separators=(",", ":")), encoding="utf-8"
        )
        inventory_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
        validated_inventory = validate_inventory(
            inventory_path,
            inventory_sha,
            inventory_spec_path,
            inventory_spec_sha,
        )
        ok(validated_inventory["authority"] == INVENTORY_AUTHORITY)

        def forged_inventory(
            mutate_inventory: Any, mutate_spec: Any, reason: str
        ) -> None:
            forged_spec = copy.deepcopy(inventory_spec)
            forged = copy.deepcopy(inventory)
            mutate_spec(forged_spec)
            forged_spec_path = root / f"forged-spec-{reason}.json"
            forged_spec_path.write_text(
                json.dumps(forged_spec, separators=(",", ":")), encoding="utf-8"
            )
            forged_spec_sha = hashlib.sha256(forged_spec_path.read_bytes()).hexdigest()
            forged["inventory_spec_sha256"] = forged_spec_sha
            mutate_inventory(forged)
            command = forged["command"]
            command[command.index("--inventory-spec") + 1] = str(
                forged_spec_path.resolve()
            )
            command[command.index("--inventory-spec-sha256") + 1] = forged_spec_sha
            forged_path = root / f"forged-inventory-{reason}.json"
            command[command.index("--output") + 1] = str(forged_path.resolve())
            forged_path.write_text(
                json.dumps(forged, separators=(",", ":")), encoding="utf-8"
            )
            expect_error(
                lambda: validate_inventory(
                    forged_path,
                    hashlib.sha256(forged_path.read_bytes()).hexdigest(),
                    forged_spec_path,
                    forged_spec_sha,
                )
            )

        forged_inventory(
            lambda value: value["tokenizer"].__setitem__("token_count", VOCAB - 1),
            lambda value: value["tokenizer"].__setitem__("token_count", VOCAB - 1),
            "tokenizer",
        )
        forged_inventory(
            lambda value: value["mask_noise"].__setitem__("mask_token", 1),
            lambda value: value.__setitem__("expected_mask_token", 1),
            "mask",
        )
        forged_inventory(
            lambda value: value["mask_noise"].__setitem__("noise_tokens", [1] * 8),
            lambda value: value.__setitem__("carry_token", 1),
            "noise",
        )
        forged_inventory(
            lambda value: value["tensors"][0].__setitem__(
                "offset", value["tensors"][0]["offset"] + 32
            ),
            lambda value: None,
            "tensor-offset",
        )
        forged_inventory(
            lambda value: next(
                tensor
                for tensor in value["tensors"]
                if tensor["role"] == "selector_hidden"
            ).__setitem__("dtype", "F16"),
            lambda value: next(
                tensor
                for tensor in value["tensor_requirements"]
                if tensor["role"] == "selector_hidden"
            ).__setitem__("dtype", "F16"),
            "selector-hidden-not-q4",
        )
        forged_inventory(
            lambda value: (
                value["checkout"].__setitem__("commit", "c" * 40),
                value["build"].__setitem__("commit", "c" * 40),
            ),
            lambda value: (
                value["checkout"].__setitem__("commit", "c" * 40),
                value["build"].__setitem__("commit", "c" * 40),
            ),
            "git-head",
        )

        for predicate_key in SELECTOR_PREDICATE_KEYS:
            if predicate_key in {"kernel", "grid", "threads"}:
                bad_provenance = copy.deepcopy(
                    case["records"][0]["payload"]["provenance"]
                )
                selector = bad_provenance["selector_hidden_dispatch"]
                census_selector = next(
                    row
                    for row in bad_provenance["dispatch_census"]
                    if row["tag"] == SELECTOR_DISPATCH_TAG
                )
                replacement = "wrong" if predicate_key == "kernel" else [2, 1, 1]
                selector[predicate_key] = replacement
                census_selector[predicate_key] = replacement
                expect_error(
                    lambda value=bad_provenance: validate_provenance(
                        value, static_manifest
                    )
                )
                continue
            bad_manifest = copy.deepcopy(static_manifest)
            predicate = bad_manifest["selector_dispatch_predicate"]
            if predicate_key in {
                "tag",
                "kernel",
                "weight_dtype",
                "input_dtype",
                "output_dtype",
            }:
                predicate[predicate_key] = "wrong"
            elif predicate_key in {"n", "h", "r"}:
                predicate[predicate_key] += 1
            elif predicate_key in {"grid", "threads"}:
                predicate[predicate_key] = [2, 1, 1]
            elif predicate_key == "allowed_environment":
                predicate[predicate_key] = {}
            else:
                predicate[predicate_key] = "0" * 64
            expect_error(lambda value=bad_manifest: validate_manifest(value))

        control_y = root / "control-y"
        outputs = {
            "fixture": str((control_y / "prepared-fixture.json").resolve()),
            "command": str((control_y / "prepared-command.json").resolve()),
            "manifest": str((control_y / "prepared-manifest.json").resolve()),
            "seal": str((control_y / "prepared-seal.json").resolve()),
        }
        fixture_content = parse_json(case["fixture"].read_bytes(), "fixture template")
        manifest_choices = {
            key: copy.deepcopy(static_manifest[key]) for key in MANIFEST_CHOICE_KEYS
        }
        prep_spec_path = root / "preparation-spec.json"
        prep_spec = {
            "schema": PREPARATION_SPEC_SCHEMA,
            "schema_version": 1,
            "run_id": inventory["run_id"],
            "attempt_id": "synthetic-attempt-1",
            "inventory_path": str(inventory_path.resolve()),
            "inventory_sha256": inventory_sha,
            "inventory_spec_path": str(inventory_spec_path.resolve()),
            "inventory_spec_sha256": inventory_spec_sha,
            "preparation_spec_path": str(prep_spec_path.resolve()),
            "worktree_x": {
                "path": inventory["checkout"]["path"],
                "commit": inventory["checkout"]["commit"],
            },
            "control_y_input": {
                "path": str(control_y.resolve()),
                "commit": git_y["head"],
                "tree": git_y["tree"],
            },
            "outputs": outputs,
            "fixture_content": fixture_content,
            "acquisition_outputs": {
                "manifest": outputs["manifest"],
                "trace": str((control_y / "acquisition.trace").resolve()),
                "sidecar": str((control_y / "acquisition.sidecar").resolve()),
            },
            "continuation_carry_token": 1,
            "manifest_choices": manifest_choices,
            "transformation_sha256": inventory["reducer"]["sha256"],
            "environment_allowlist": {"QWEN_METAL_LEASE_WAIT": "1"},
            "arm_order": ["off-A", "on-A", "on-B", "off-B"],
            "selected_arm": "on-A",
            "parity_comparison_fields": PARITY_COMPARISON_FIELDS,
            "reducer_argv": [],
            "reduction_output": str((control_y / "reduction.json").resolve()),
            "failure_policy": {
                "on_collision": "retain_reserved_partial",
                "retry": False,
            },
        }
        prep_spec["reducer_argv"] = frozen_reducer_argv(
            prep_spec, inventory["reducer"]["path"]
        )
        prep_spec_path.write_text(
            json.dumps(prep_spec, separators=(",", ":")), encoding="utf-8"
        )
        prep_spec_sha = hashlib.sha256(prep_spec_path.read_bytes()).hexdigest()
        bad_y_head = copy.deepcopy(prep_spec)
        bad_y_head["control_y_input"]["commit"] = "e" * 40
        expect_error(
            lambda: render_preparation(
                inventory, inventory_sha, bad_y_head, prep_spec_sha
            ),
            "control Y input HEAD/tree",
        )
        bad_six_paths = copy.deepcopy(prep_spec)
        bad_six_paths["acquisition_outputs"]["sidecar"] = bad_six_paths[
            "acquisition_outputs"
        ]["trace"]
        expect_error(
            lambda: render_preparation(
                inventory, inventory_sha, bad_six_paths, prep_spec_sha
            ),
            "paths must be unique",
        )
        dirty_x_path = Path(source_claims[0]["path"])
        clean_x_bytes = dirty_x_path.read_bytes()
        dirty_x_path.write_bytes(clean_x_bytes + b"dirty")
        expect_error(
            lambda: render_preparation(
                inventory, inventory_sha, prep_spec, prep_spec_sha
            ),
            "worktree X",
        )
        dirty_x_path.write_bytes(clean_x_bytes)
        unrelated_y = control_y / "unrelated-untracked"
        unrelated_y.write_bytes(b"unexpected")
        expect_error(
            lambda: render_preparation(
                inventory, inventory_sha, prep_spec, prep_spec_sha
            ),
            "unrelated untracked",
        )
        unrelated_y.unlink()
        rendered_a = render_preparation(
            inventory, inventory_sha, prep_spec, prep_spec_sha
        )
        rendered_b = render_preparation(
            inventory, inventory_sha, copy.deepcopy(prep_spec), prep_spec_sha
        )
        ok(rendered_a == rendered_b)
        expected_hashes = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in zip(
                ("fixture", "command", "manifest", "seal"), rendered_a
            )
        }
        prepared = prepare_artifacts(
            inventory_path,
            inventory_sha,
            inventory_spec_path,
            inventory_spec_sha,
            prep_spec_path,
            prep_spec_sha,
            expected_hashes,
        )
        ok(set(prepared) == {"fixture", "command", "manifest", "seal"})
        collision_result = prepare_artifacts(
            inventory_path,
            inventory_sha,
            inventory_spec_path,
            inventory_spec_sha,
            prep_spec_path,
            prep_spec_sha,
            expected_hashes,
        )
        ok(
            collision_result["result"] == "partial"
            and collision_result["retry"] is False
        )
        bad_inventory = copy.deepcopy(inventory)
        bad_inventory["rng"] = 1
        bad_inventory_path = root / "bad-inventory.json"
        bad_inventory_path.write_text(
            json.dumps(bad_inventory, separators=(",", ":")), encoding="utf-8"
        )
        expect_error(
            lambda: validate_inventory(
                bad_inventory_path,
                hashlib.sha256(bad_inventory_path.read_bytes()).hexdigest(),
                inventory_spec_path,
                inventory_spec_sha,
            ),
            "inventory keys/order mismatch",
        )
        bad_prep = copy.deepcopy(prep_spec)
        bad_prep["manifest_choices"]["expected_capture"] = {}
        expect_error(
            lambda: render_preparation(
                inventory, inventory_sha, bad_prep, prep_spec_sha
            ),
            "manifest choices keys/order mismatch",
        )
        escaped_y = copy.deepcopy(prep_spec)
        escaped_y["outputs"]["seal"] = str((root / "escaped-seal.json").resolve())
        expect_error(
            lambda: render_preparation(
                inventory, inventory_sha, escaped_y, prep_spec_sha
            ),
            "control Y",
        )
        escaped_x = copy.deepcopy(prep_spec)
        escaped_x["worktree_x"]["path"] = str(control_y.resolve())
        expect_error(
            lambda: render_preparation(
                inventory, inventory_sha, escaped_x, prep_spec_sha
            ),
            "worktree-X",
        )

    print(
        f"dflash_k0s self-test: PASS ({tests} checks; stdlib-only, no model/Metal/network)"
    )


def main() -> int:
    literal_argv = list(sys.argv)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--validate-inventory", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-sha256")
    parser.add_argument("--inventory-spec", type=Path)
    parser.add_argument("--inventory-spec-sha256")
    parser.add_argument("--preparation-spec", type=Path)
    parser.add_argument("--preparation-spec-sha256")
    parser.add_argument("--preparation-seal", type=Path)
    parser.add_argument("--preparation-seal-sha256")
    for name in ("fixture", "command", "static-manifest", "seal"):
        parser.add_argument(f"--expected-{name}-sha256")
    parser.add_argument("--input", type=Path, help="qwen.dflash_k0s_lattice v1 JSONL")
    parser.add_argument(
        "--sidecar", type=Path, help="exclusive producer binary sidecar"
    )
    parser.add_argument(
        "--manifest", type=Path, help="external identity/trust manifest"
    )
    parser.add_argument(
        "--manifest-sha256",
        help="independently frozen exact SHA-256 of the external manifest",
    )
    parser.add_argument(
        "--output", type=Path, help="exclusive-create compact reduction JSON"
    )
    args = parser.parse_args()
    inventory_arguments = (
        args.inventory,
        args.inventory_sha256,
        args.inventory_spec,
        args.inventory_spec_sha256,
    )
    preparation_arguments = (
        args.preparation_spec,
        args.preparation_spec_sha256,
        args.preparation_seal,
        args.preparation_seal_sha256,
        args.expected_fixture_sha256,
        args.expected_command_sha256,
        args.expected_static_manifest_sha256,
        args.expected_seal_sha256,
    )
    reduction_arguments = (
        args.input,
        args.sidecar,
        args.manifest,
        args.manifest_sha256,
        args.output,
    )
    if args.self_test:
        require(
            not any(inventory_arguments + preparation_arguments + reduction_arguments),
            "--self-test cannot be combined with other mode arguments",
        )
        self_test()
        return 0
    if args.validate_inventory:
        require(
            all(inventory_arguments)
            and not any(preparation_arguments + reduction_arguments),
            "inventory validation requires inventory/spec paths and hashes",
        )
        inventory = validate_inventory(
            args.inventory,
            args.inventory_sha256,
            args.inventory_spec,
            args.inventory_spec_sha256,
        )
        print(
            json.dumps(
                {
                    "schema": INVENTORY_SCHEMA,
                    "schema_version": INVENTORY_VERSION,
                    "run_id": inventory["run_id"],
                    "status": "valid",
                    "authority": INVENTORY_AUTHORITY,
                },
                separators=(",", ":"),
            )
        )
        return 0
    if args.prepare:
        require(
            all(
                inventory_arguments
                + preparation_arguments[:2]
                + preparation_arguments[4:]
            )
            and not any(preparation_arguments[2:4])
            and not any(reduction_arguments),
            "preparation requires inventory/spec paths and all expected hashes",
        )
        prepared = prepare_artifacts(
            args.inventory,
            args.inventory_sha256,
            args.inventory_spec,
            args.inventory_spec_sha256,
            args.preparation_spec,
            args.preparation_spec_sha256,
            {
                "fixture": args.expected_fixture_sha256,
                "command": args.expected_command_sha256,
                "manifest": args.expected_static_manifest_sha256,
                "seal": args.expected_seal_sha256,
            },
        )
        print(json.dumps(prepared, separators=(",", ":")))
        return 0
    require(
        all(reduction_arguments + inventory_arguments + preparation_arguments[:4])
        and not any(preparation_arguments[4:]),
        "reduction requires trace/sidecar/manifest/output plus independently hashed inventory, inventory-spec, preparation-spec, and preparation-seal",
    )
    result = reduce(
        args.input,
        args.sidecar,
        args.manifest,
        args.output,
        args.manifest_sha256,
        args.inventory,
        args.inventory_sha256,
        args.inventory_spec,
        args.inventory_spec_sha256,
        args.preparation_spec,
        args.preparation_spec_sha256,
        args.preparation_seal,
        args.preparation_seal_sha256,
        literal_argv,
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    return 0 if result["result"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InvalidEvidence as error:
        print(f"dflash_k0s: invalid: {error}", file=sys.stderr)
        raise SystemExit(2)
