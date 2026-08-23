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
import selectors
import stat
import struct
import subprocess
import sys
import tempfile
import time
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
INVENTORY_VERSION = 2
INVENTORY_AUTHORITY = (
    "development_k0s_inventory_only_no_model_forward_or_semantic_authority"
)
BOOTSTRAP_TARGET_PATH = "/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf"
BOOTSTRAP_TARGET_BYTES = 17106773984
BOOTSTRAP_TARGET_SHA256 = (
    "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
)
BOOTSTRAP_DRAFTER_PATH = (
    "/Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
)
BOOTSTRAP_DRAFTER_MAX_BYTES = 2147483648
BOOTSTRAP_DRAFTER_SHA256 = (
    "18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594"
)
INVENTORY_SPEC_SCHEMA = "qwen.dflash_k0s_inventory_spec"
PREPARATION_SPEC_SCHEMA = "qwen.dflash_k0s_preparation_spec"
PREPARATION_SPEC_VERSION = 2
PREPARATION_CHOICES_SCHEMA = "qwen.dflash_k0s_preparation_choices"
PREPARATION_CHOICES_VERSION = 1
SEAL_SCHEMA = "qwen.dflash_k0s_preparation_seal"
INVENTORY_KEYS = (
    "schema",
    "schema_version",
    "authority",
    "inventory_spec_sha256",
    "run_id",
    "expected",
    "observed",
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
    "build_report",
    "sources",
    "executable",
    "reducer",
    "embedded_metallib",
    "assets",
    "tensor_requirements",
    "tokenizer_predicate",
    "prompt_predicate",
    "mask_predicate",
    "parser_caps",
    "host_predicate",
    "command",
    "environment",
)
INVENTORY_EXPECTED_KEYS = INVENTORY_SPEC_KEYS[4:]
INVENTORY_OBSERVED_KEYS = (
    "checkout",
    "build",
    "build_report",
    "sources",
    "executable",
    "reducer",
    "embedded_metallib",
    "device",
    "assets",
    "gguf",
    "tensors",
    "tokenizer",
    "prompt",
    "mask_noise",
    "parser_caps",
)
ASSET_EXPECTATION_KEYS = ("role", "path", "expected_bytes", "max_bytes", "sha256")
TOKENIZER_PREDICATE_KEYS = (
    "vocab_size",
    "token_embd_name",
    "token_embd_rank",
    "token_embd_hidden",
    "token_embd_vocab_axis",
    "allowed_token_embd_dtypes",
    "require_token_metadata",
    "metadata_identity_domain",
)
PROMPT_PREDICATE_KEYS = (
    "utf8_hex",
    "utf8_sha256",
    "add_special",
    "expected_token_ids",
    "expected_token_ids_sha256_i32le",
)
MASK_PREDICATE_KEYS = ("allowed_metadata_keys", "expected_mask_token")
PARSER_CAP_KEYS = (
    "header_bytes",
    "metadata",
    "tensors",
    "strings_bytes",
    "array_items_per_array",
    "array_items",
    "objects",
)
TOKENIZER_OBSERVATION_KEYS = (
    "vocab_size",
    "token_embd_name",
    "token_embd_shape",
    "token_embd_dtype",
    "token_count",
    "model",
    "pre",
    "bos_token_id",
    "eos_token_id",
    "add_bos_token",
    "add_eos_token",
    "token_list_sha256",
    "token_type_sha256",
    "merges_sha256",
    "metadata_identity_sha256",
)
PROMPT_OBSERVATION_KEYS = (
    "utf8_hex",
    "add_special",
    "token_ids",
    "token_ids_sha256_i32le",
    "tokenizer_metadata_identity_sha256",
)
HOST_PREDICATE_V2_KEYS = (
    "os",
    "arch",
    "device_name",
    "required_families",
    "family_match",
)
BUILD_REPORT_SCHEMA = "qwen.dflash_k0s_build_identity_report"
BUILD_REPORT_AUTHORITY = "development_k0s_build_identity_only_no_asset_model_forward_or_acquisition_authority"
BUILD_REPORT_KEYS = (
    "schema",
    "schema_version",
    "authority",
    "run_id",
    "attempt_id",
    "checkout",
    "build_command",
    "build_root",
    "executable",
    "embedded_metallib",
    "reducer",
    "sources",
    "compiler",
    "target",
    "profile",
    "features",
    "build_info",
    "environment",
)
BUILD_REPORT_COMPILER_KEYS = (
    "path",
    "bytes",
    "sha256",
    "version_verbose",
    "version_verbose_sha256",
)
BUILD_ROOT_KEYS = ("path", "bytes", "max_bytes")
RUST_BUILD_COMMAND_SUFFIX = [
    "build",
    "--locked",
    "--offline",
    "--release",
    "-p",
    "qwen-cli",
    "--bin",
    "qwen-bench",
    "--features",
    "dflash-k0s-diagnostics",
]
BUILD_INFO_REPORT_KEYS = (
    "artifact",
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
)
PREPARATION_SPEC_KEYS = (
    "schema",
    "schema_version",
    "run_id",
    "attempt_id",
    "preparation_choices",
    "inventory_path",
    "inventory_sha256",
    "inventory_spec_path",
    "inventory_spec_sha256",
    "preparation_spec_path",
    "worktree_x",
    "planner_p",
    "control_y_input",
    "outputs",
    "fixture_content",
    "acquisition_outputs",
    "reduction_output",
    "continuation_carry_token",
    "manifest_choices",
    "transformation_sha256",
    "environment_allowlist",
    "arm_order",
    "selected_arm",
    "parity_comparison_fields",
    "reducer_argv",
    "failure_policy",
)
PREPARATION_CHOICES_KEYS = (
    "schema",
    "schema_version",
    "run_id",
    "attempt_id",
    "worktree_x",
    "planner_p",
    "control_y",
    "outputs",
    "preparation_spec_path",
    "fixture_content",
    "acquisition_outputs",
    "reduction_output",
    "continuation_carry_token",
    "manifest_choices",
    "transformation_sha256",
    "environment_allowlist",
    "arm_order",
    "selected_arm",
    "parity_comparison_fields",
    "reducer_argv_template",
    "failure_policy",
)
PREPARATION_JOIN_FIELDS = (
    "inventory_path",
    "inventory_sha256",
    "inventory_spec_path",
    "inventory_spec_sha256",
)
MAX_BOOTSTRAP_SPEC_BYTES = 1 << 20
MAX_PREPARATION_HASH_REPORT_BYTES = 65536
MAX_GIT_STDOUT_BYTES = 64 << 20
MAX_GIT_STDERR_BYTES = 1 << 20
MAX_BUILD_ROOT_BYTES = 64 << 30
MAX_COMPILER_BYTES = 1 << 30
PREPARATION_HASH_REPORT_SCHEMA = "qwen.dflash_k0s_preparation_hash_observation"
PREPARATION_HASH_REPORT_AUTHORITY = "development_k0s_preparation_hashes_only_no_asset_discovery_model_forward_or_acquisition_authority"
PREPARATION_HASH_REPORT_KEYS = (
    "schema",
    "schema_version",
    "authority",
    "run_id",
    "attempt_id",
    "inventory",
    "inventory_spec",
    "preparation_choices",
    "preparation_spec",
    "reducer",
    "rendered",
    "worktree_x",
    "control_y",
    "report",
    "environment",
)
GIT_REPORT_IDENTITY_KEYS = (
    "path",
    "head",
    "tree",
    "status_bytes",
    "status_sha256",
    "ignored_bytes",
    "ignored_sha256",
    "common_git_dir",
    "object_store",
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
    "preparation_choices",
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
    "preparation_choices_sha256",
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
PREPARATION_CHOICES_SHA256_PLACEHOLDER = "${PREPARATION_CHOICES_SHA256}"
INVENTORY_PATH_PLACEHOLDER = "${INVENTORY_PATH}"
INVENTORY_SHA256_PLACEHOLDER = "${INVENTORY_SHA256}"
INVENTORY_SPEC_PATH_PLACEHOLDER = "${INVENTORY_SPEC_PATH}"
INVENTORY_SPEC_SHA256_PLACEHOLDER = "${INVENTORY_SPEC_SHA256}"

MAX_RECORDS = 16
MAX_TRACE_BYTES = 64 << 20
MAX_SIDECAR_BYTES = 64 << 20
MAX_COMBINED_BYTES = 128 << 20
MAX_JSON_DEPTH = 16
MAX_JSON_INTEGER = (1 << 63) - 1
MIN_JSON_INTEGER = -(1 << 63)
MAX_JSON_U64 = (1 << 64) - 1
READ_CHUNK = 1 << 20
MAX_SIDECAR_RANGES = 4096
MAX_DISPATCH_ROWS = 256
MAX_KERNEL_TRACE_ROWS = 1024
MAX_GGUF_TENSORS = 8192
MAX_GGUF_METADATA = 4096
MAX_GGUF_STRINGS_BYTES = 16 << 20
MAX_GGUF_ARRAY_ITEMS_PER_ARRAY = 500000
MAX_GGUF_ARRAY_ITEMS = 2000000
MAX_GGUF_OBJECTS = 2500000
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
SOURCE_ROLE_PATHS = (
    ("metal_dflash_rs", "crates/qwen-llm/src/metal_dflash.rs"),
    ("bench_rs", "crates/qwen-cli/src/bench.rs"),
    ("dflash_k0s_rs", "crates/qwen-cli/src/dflash_k0s.rs"),
    ("qwen_llm_cargo_toml", "crates/qwen-llm/Cargo.toml"),
    ("qwen_cli_cargo_toml", "crates/qwen-cli/Cargo.toml"),
    ("metal_rs", "crates/qwen-llm/src/metal.rs"),
    ("metal_forward_rs", "crates/qwen-llm/src/metal_forward.rs"),
    ("dflash2_metal", "kernels/dflash2.metal"),
    ("mat_mat_mma8_metal", "kernels/mat_mat_mma8.metal"),
    ("mat_mat_q4_k_metal", "kernels/mat_mat_q4_k.metal"),
    ("build_rs", "crates/qwen-llm/build.rs"),
    ("tokenizer_rs", "crates/qwen-llm/src/tokenizer.rs"),
    ("gguf_rs", "crates/qwen-llm/src/gguf.rs"),
    ("source_identity_rs", "crates/qwen-cli/source_identity.rs"),
    ("workspace_cargo_toml", "Cargo.toml"),
    ("cargo_lock", "Cargo.lock"),
    ("qwen_cli_build_rs", "crates/qwen-cli/build.rs"),
    ("qwen_llm_lib_rs", "crates/qwen-llm/src/lib.rs"),
)
REQUIRED_SOURCE_ROLES = {role for role, _ in SOURCE_ROLE_PATHS}
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


def bounded_ascii(value: Any, name: str, maximum: int) -> str:
    require(isinstance(value, str), f"{name} must be ASCII text")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise InvalidEvidence(f"{name} must be ASCII text") from error
    require(
        0 < len(encoded) <= maximum and b"\0" not in encoded,
        f"{name} must be bounded nonempty ASCII text",
    )
    return value


def bounded_utf8_bytes(value: Any, name: str, maximum: int) -> bytes:
    require(isinstance(value, str), f"{name} must be UTF-8 text")
    encoded = value.encode("utf-8")
    require(
        0 < len(encoded) <= maximum and b"\0" not in encoded,
        f"{name} must be bounded nonempty UTF-8 text",
    )
    return encoded


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
    require(len(raw.lstrip("-")) <= 20, "JSON integer has too many digits")
    value = int(raw)
    require(
        MIN_JSON_INTEGER <= value <= MAX_JSON_U64,
        "JSON integer exceeds i64/u64 grammar bound",
    )
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


def canonical_json_bytes(value: Any) -> bytes:
    require(isinstance(value, dict), "canonical JSON root must be an object")
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def parse_canonical_json_object(
    opened: OpenFile, keys: tuple[str, ...], name: str
) -> dict[str, Any]:
    raw = pread_exact(opened, 0, opened.size, name)
    require(
        raw.endswith(b"\n") and not raw.endswith(b"\n\n"),
        f"{name} newline invalid",
    )
    value = exact_keys(parse_json(raw, name), keys, name)
    require(raw == canonical_json_bytes(value), f"{name} bytes are not canonical JSON")
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
    parent_path: Path
    parent_fd: int
    parent_snapshot: tuple[int, int, int, int]
    file: BinaryIO
    size: int
    inode: tuple[int, int]
    mtime_ns: int
    ctime_ns: int
    digest: str

    def close(self) -> None:
        try:
            self.file.close()
        finally:
            os.close(self.parent_fd)


@dataclass
class InventoryContext:
    inventory: dict[str, Any]
    opened: list[OpenFile]
    build_root_custody: dict[str, Any] | None = None

    def final_check(self) -> None:
        if self.build_root_custody is not None:
            require(
                measure_directory(
                    self.build_root_custody["path"],
                    self.build_root_custody["maximum"],
                )
                == self.build_root_custody["measurement"],
                "build root changed after inventory validation",
            )
        for item in self.opened:
            final_custody_check(item)

    def close(self) -> None:
        for item in self.opened:
            item.close()


def open_regular(path: Path, name: str, maximum: int | None = None) -> OpenFile:
    require(path.is_absolute(), f"{name} path must be absolute")
    canonical = path.resolve(strict=True)
    require(path == canonical, f"{name} path must be lexically canonical")
    parent = canonical.parent
    parent_fd = -1
    handle: BinaryIO | None = None
    try:
        parent_fd = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        parent_info = os.fstat(parent_fd)
        require(stat.S_ISDIR(parent_info.st_mode), f"{name} parent is not a directory")
        descriptor = os.open(
            canonical.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        handle = os.fdopen(descriptor, "rb", buffering=0)
    except OSError as error:
        if parent_fd >= 0:
            os.close(parent_fd)
        raise InvalidEvidence(f"cannot open {name} {canonical}: {error}") from error
    try:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode), f"{name} is not a regular file")
        require(info.st_nlink == 1, f"{name} must have exactly one hard link")
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
            parent,
            parent_fd,
            (
                parent_info.st_dev,
                parent_info.st_ino,
                parent_info.st_mtime_ns,
                parent_info.st_ctime_ns,
            ),
            handle,
            info.st_size,
            (info.st_dev, info.st_ino),
            info.st_mtime_ns,
            info.st_ctime_ns,
            digest.hexdigest(),
        )
    except Exception:
        handle.close()
        os.close(parent_fd)
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
    claimed_path = Path(text(item["path"], f"{name}.path"))
    require(claimed_path.is_absolute(), f"{name}.path must be absolute")
    require(
        claimed_path.parent.resolve(strict=True) / claimed_path.name == claimed_path,
        f"{name}.path must be lexically canonical",
    )
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
    if dtype == 7:
        raw = cursor.take(1)[0]
        require(raw in (0, 1), "GGUF boolean is not canonically encoded")
        return bool(raw)
    sizes = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
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
        require(
            count <= MAX_GGUF_ARRAY_ITEMS_PER_ARRAY,
            "GGUF metadata array exceeds per-array cap",
        )
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
    integer(host["device_registry_id"], "device registry id", 0, MAX_JSON_U64)
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


def tokenizer_string_array_digest(domain: str, values: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(struct.pack("<Q", len(values)))
    for value in values:
        require(isinstance(value, str), "tokenizer string array contains non-string")
        raw = value.encode("utf-8")
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def tokenizer_i64_array_digest(domain: str, values: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(struct.pack("<Q", len(values)))
    for value in values:
        digest.update(
            struct.pack("<q", integer(value, "token type", -(1 << 63), (1 << 63) - 1))
        )
    return digest.hexdigest()


def tokenizer_metadata_identity(
    architecture: str,
    model: str,
    pre: str,
    token_count: int,
    token_digest: str,
    token_type_count: int,
    token_type_digest: str,
    merges_count: int,
    merges_digest: str,
    bos_token_id: int | None,
    eos_token_id: int | None,
    add_bos_token: bool | None,
    add_eos_token: bool | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"qwen.dflash_k0s.tokenizer_metadata.v1")

    def framed(value: bytes) -> None:
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)

    for key, value in (
        ("general.architecture", architecture),
        ("tokenizer.ggml.model", model),
        ("tokenizer.ggml.pre", pre),
    ):
        framed(key.encode("ascii"))
        framed(bounded_utf8_bytes(value, key, 1 << 20))
    for key, count, component in (
        ("tokenizer.ggml.tokens", token_count, token_digest),
        ("tokenizer.ggml.token_type", token_type_count, token_type_digest),
        ("tokenizer.ggml.merges", merges_count, merges_digest),
    ):
        framed(key.encode("ascii"))
        digest.update(struct.pack("<Q", integer(count, f"{key} count")))
        digest.update(bytes.fromhex(sha256_text(component, f"{key} digest")))
    for key, value in (
        ("tokenizer.ggml.bos_token_id", bos_token_id),
        ("tokenizer.ggml.eos_token_id", eos_token_id),
    ):
        framed(key.encode("ascii"))
        if value is None:
            digest.update(b"\0")
        else:
            digest.update(
                b"\1"
                + struct.pack("<q", integer(value, key, -(1 << 63), (1 << 63) - 1))
            )
    for key, value in (
        ("tokenizer.ggml.add_bos_token", add_bos_token),
        ("tokenizer.ggml.add_eos_token", add_eos_token),
    ):
        framed(key.encode("ascii"))
        if value is None:
            digest.update(b"\0")
        else:
            require(isinstance(value, bool), f"{key} must be optional boolean")
            digest.update(bytes((1, int(value))))
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
    for key in (
        "inventory",
        "inventory_spec",
        "preparation_choices",
        "preparation_spec",
    ):
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


def leaf_present(path: Path) -> bool:
    try:
        os.lstat(path)
        return True
    except FileNotFoundError:
        return False


def open_directory_custody(path: Path, name: str) -> tuple[int, os.stat_result]:
    require(
        path.is_absolute() and path.resolve(strict=True) == path,
        f"{name} path is not canonical absolute",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    snapshot = os.fstat(descriptor)
    path_snapshot = os.stat(path, follow_symlinks=False)
    require(
        stat.S_ISDIR(snapshot.st_mode)
        and (path_snapshot.st_dev, path_snapshot.st_ino)
        == (snapshot.st_dev, snapshot.st_ino),
        f"{name} directory path/FD identity mismatch",
    )
    return descriptor, snapshot


def verify_directory_custody(
    descriptor: int,
    path: Path,
    snapshot: os.stat_result,
    name: str,
    *,
    metadata_stable: bool,
) -> None:
    current = os.fstat(descriptor)
    path_current = os.stat(path, follow_symlinks=False)
    require(
        stat.S_ISDIR(current.st_mode)
        and (current.st_dev, current.st_ino) == (snapshot.st_dev, snapshot.st_ino)
        and (path_current.st_dev, path_current.st_ino)
        == (snapshot.st_dev, snapshot.st_ino)
        and path.resolve(strict=True) == path,
        f"{name} directory path/FD custody changed",
    )
    if metadata_stable:
        require(
            (current.st_mtime_ns, current.st_ctime_ns)
            == (snapshot.st_mtime_ns, snapshot.st_ctime_ns)
            and (path_current.st_mtime_ns, path_current.st_ctime_ns)
            == (snapshot.st_mtime_ns, snapshot.st_ctime_ns),
            f"{name} directory metadata custody changed",
        )


def verify_open_output_fd(
    descriptor: int,
    parent_fd: int,
    leaf_name: str,
    expected: bytes,
    snapshot: os.stat_result,
    absolute_path: Path | None = None,
) -> str:
    require(absolute_path is not None, "exclusive output absolute path is required")
    parent_before = os.fstat(parent_fd)
    relative_fd = -1
    absolute_fd = -1
    try:
        relative_fd = os.open(
            leaf_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        absolute_fd = os.open(absolute_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        identities = [
            os.fstat(value) for value in (descriptor, relative_fd, absolute_fd)
        ]
        require(
            all(
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and info.st_size == len(expected)
                and (info.st_dev, info.st_ino) == (snapshot.st_dev, snapshot.st_ino)
                for info in identities
            )
            and (
                identities[0].st_mtime_ns,
                identities[0].st_ctime_ns,
            )
            == (snapshot.st_mtime_ns, snapshot.st_ctime_ns),
            "exclusive output FD/path custody changed",
        )

        def descriptor_digest(value: int) -> bytes:
            digest = hashlib.sha256()
            offset = 0
            while offset < len(expected):
                block = os.pread(value, min(READ_CHUNK, len(expected) - offset), offset)
                require(block, "short read rehashing exclusive output")
                digest.update(block)
                offset += len(block)
            return digest.digest()

        expected_digest = hashlib.sha256(expected).digest()
        require(
            all(
                descriptor_digest(value) == expected_digest
                for value in (descriptor, relative_fd, absolute_fd)
            ),
            "exclusive output final hash mismatch",
        )
        parent_after = os.fstat(parent_fd)
        require(
            (
                parent_after.st_dev,
                parent_after.st_ino,
                parent_after.st_mtime_ns,
                parent_after.st_ctime_ns,
            )
            == (
                parent_before.st_dev,
                parent_before.st_ino,
                parent_before.st_mtime_ns,
                parent_before.st_ctime_ns,
            ),
            "exclusive output parent changed during dual-path verification",
        )
        return expected_digest.hex()
    finally:
        if relative_fd >= 0:
            os.close(relative_fd)
        if absolute_fd >= 0:
            os.close(absolute_fd)


def write_all_and_fsync(descriptor: int, data: bytes, name: str) -> None:
    written = 0
    while written < len(data):
        count = os.write(descriptor, data[written:])
        require(count > 0, f"short write for {name}")
        written += count
    os.fsync(descriptor)


def fsync_partial_evidence(descriptor: int, parent_fd: int, name: str) -> None:
    try:
        os.fsync(descriptor)
        os.fsync(parent_fd)
    except OSError as error:
        raise InvalidEvidence(
            f"cannot durably bind retained partial {name}: {error}"
        ) from error


def finalize_reduction_output(
    descriptor: int,
    parent_fd: int,
    path: Path,
    expected: bytes,
    snapshot: os.stat_result,
    terminal_parent_snapshot: os.stat_result,
) -> str:
    digest = verify_open_output_fd(
        descriptor, parent_fd, path.name, expected, snapshot, path
    )
    retained = os.fstat(descriptor)
    relative = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    absolute = os.stat(path, follow_symlinks=False)
    require(
        all(
            stat.S_ISREG(info.st_mode)
            and info.st_nlink == 1
            and (info.st_dev, info.st_ino) == (snapshot.st_dev, snapshot.st_ino)
            and info.st_size == snapshot.st_size
            and info.st_mtime_ns == snapshot.st_mtime_ns
            and info.st_ctime_ns == snapshot.st_ctime_ns
            for info in (retained, relative, absolute)
        ),
        "reduction output changed during final joint leaf sweep",
    )
    # This stable parent FD/path check is intentionally the terminal custody operation.
    verify_directory_custody(
        parent_fd,
        path.parent,
        terminal_parent_snapshot,
        "reduction output parent",
        metadata_stable=True,
    )
    return digest


def bounded_subprocess(
    argv: list[str],
    environment: dict[str, str],
    *,
    stdout_cap: int,
    stderr_cap: int,
    timeout: float,
    name: str,
) -> tuple[int, bytes, bytes]:
    require(
        argv
        and all(isinstance(value, str) and value for value in argv)
        and 0 <= stdout_cap <= 64 << 20
        and 0 <= stderr_cap <= 64 << 20
        and 0 < timeout <= 30,
        f"{name} bounded subprocess contract invalid",
    )
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()

    def terminate_and_reap() -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)

    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            bufsize=0,
        )
        require(
            process.stdout is not None and process.stderr is not None,
            f"{name} pipes unavailable",
        )
        for pipe, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, label)
        deadline = time.monotonic() + timeout
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate_and_reap()
                raise InvalidEvidence(f"{name} timed out")
            events = selector.select(min(remaining, 0.1))
            if not events and process.poll() is not None:
                events = [
                    (key, selectors.EVENT_READ) for key in selector.get_map().values()
                ]
            for key, _ in events:
                destination = stdout if key.data == "stdout" else stderr
                cap = stdout_cap if key.data == "stdout" else stderr_cap
                try:
                    block = os.read(
                        key.fileobj.fileno(), min(65536, cap - len(destination) + 1)
                    )
                except BlockingIOError:
                    continue
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                destination.extend(block)
                if len(destination) > cap:
                    terminate_and_reap()
                    raise InvalidEvidence(f"{name} {key.data} exceeded cap")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            terminate_and_reap()
            raise InvalidEvidence(f"{name} timed out")
        returncode = process.wait(timeout=remaining)
        return returncode, bytes(stdout), bytes(stderr)
    except InvalidEvidence:
        terminate_and_reap()
        raise
    except (OSError, subprocess.SubprocessError) as error:
        terminate_and_reap()
        raise InvalidEvidence(f"{name} failed: {error}") from error
    finally:
        selector.close()
        if process is not None:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()


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
    returncode, stdout, _stderr = bounded_subprocess(
        [
            str(git),
            "--no-optional-locks",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(path),
            *arguments,
        ],
        environment,
        stdout_cap=MAX_GIT_STDOUT_BYTES,
        stderr_cap=MAX_GIT_STDERR_BYTES,
        timeout=5,
        name="bounded git inspection",
    )
    require(returncode == 0, "bounded git inspection command failed")
    return stdout


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
    ignored = bounded_git(
        canonical,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
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
        "ignored": ignored,
    }


def git_relative_entries(raw: bytes, name: str) -> tuple[Path, ...]:
    result: list[Path] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        try:
            text_entry = entry.decode("utf-8")
        except UnicodeDecodeError as error:
            raise InvalidEvidence(f"{name} path is not UTF-8") from error
        relative = Path(text_entry)
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.parts,
            f"{name} path is noncanonical",
        )
        result.append(relative)
    require(len(result) == len(set(result)), f"duplicate {name} path")
    return tuple(result)


def validate_ignored_entries(
    checkout: dict[str, Any],
    name: str,
    *,
    allowed_root: Path | None = None,
    allowed_paths: set[Path] | None = None,
) -> set[Path]:
    entries = git_relative_entries(checkout["ignored"], f"{name} ignored entry")
    root = checkout["path"]
    absolute = {root / entry for entry in entries}
    allowed_paths = allowed_paths or set()
    allowed_relative = (
        allowed_root.relative_to(root) if allowed_root is not None else None
    )
    require(
        all(
            root / entry in allowed_paths
            or (
                allowed_relative is not None
                and (entry == allowed_relative or allowed_relative in entry.parents)
            )
            for entry in entries
        ),
        f"{name} contains an unauthorized ignored entry",
    )
    return absolute


def inventory_build_root(facts: dict[str, Any], x_path: Path) -> Path:
    claim = identity_from_claim(facts["build_report"], "inventory build report")
    opened = open_regular(
        Path(claim["path"]), "inventory build report", claim["max_bytes"]
    )
    try:
        verify_identity(opened, claim, "inventory build report")
        report = exact_keys(
            parse_json(
                pread_exact(opened, 0, opened.size, "inventory build report"),
                "inventory build report",
            ),
            BUILD_REPORT_KEYS,
            "inventory build report",
        )
        root_claim = exact_keys(report["build_root"], BUILD_ROOT_KEYS, "build root")
        root = Path(root_claim["path"])
        require(
            root.is_absolute()
            and root.resolve(strict=True) == root
            and root.is_dir()
            and root.is_relative_to(x_path),
            "inventory build root is not canonical under X",
        )
        return root
    finally:
        opened.close()


def validate_preparation_git_state(
    inventory: dict[str, Any],
    spec: dict[str, Any],
    allowed_y_outputs: set[Path],
    *,
    require_all_outputs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory_facts = inventory.get("observed", inventory)
    checkout = inventory_facts["checkout"]
    control = spec["control_y_input"]
    git_x = inspect_git_checkout(Path(checkout["path"]), "preparation worktree X")
    git_y = inspect_git_checkout(
        Path(control["path"]), "preparation control Y", require_clean=False
    )
    if git_x["ignored"]:
        validate_ignored_entries(
            git_x,
            "preparation worktree X",
            allowed_root=inventory_build_root(inventory_facts, git_x["path"]),
        )
    ignored_outputs = validate_ignored_entries(
        git_y,
        "preparation control Y",
        allowed_paths=allowed_y_outputs,
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
    observed: set[Path] = set(ignored_outputs)
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


def final_custody_check(
    opened: OpenFile,
    parent_snapshot_override: tuple[int, int, int, int] | None = None,
) -> None:
    expected_parent = parent_snapshot_override or opened.parent_snapshot
    relative_fd = -1
    absolute_fd = -1
    try:
        relative_fd = os.open(
            opened.path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=opened.parent_fd,
        )
        absolute_fd = os.open(opened.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))

        def expected_identity(info: os.stat_result) -> bool:
            return (
                stat.S_ISREG(info.st_mode)
                and (info.st_dev, info.st_ino) == opened.inode
                and info.st_nlink == 1
                and info.st_size == opened.size
                and info.st_mtime_ns == opened.mtime_ns
                and info.st_ctime_ns == opened.ctime_ns
            )

        descriptors = (opened.file.fileno(), relative_fd, absolute_fd)
        require(
            all(expected_identity(os.fstat(descriptor)) for descriptor in descriptors),
            f"custody changed for {opened.path}",
        )

        def digest_fd(descriptor: int) -> str:
            digest = hashlib.sha256()
            offset = 0
            while offset < opened.size:
                block = os.pread(
                    descriptor, min(READ_CHUNK, opened.size - offset), offset
                )
                require(block, f"short read during final custody for {opened.path}")
                digest.update(block)
                offset += len(block)
            return digest.hexdigest()

        require(
            all(digest_fd(descriptor) == opened.digest for descriptor in descriptors),
            f"final custody hash changed for {opened.path}",
        )
        require(
            all(expected_identity(os.fstat(descriptor)) for descriptor in descriptors),
            f"custody changed after hashing for {opened.path}",
        )
        relative_info = os.stat(
            opened.path.name, dir_fd=opened.parent_fd, follow_symlinks=False
        )
        absolute_info = os.stat(opened.path, follow_symlinks=False)
        parent_info = os.fstat(opened.parent_fd)
        parent_path_info = os.stat(opened.parent_path, follow_symlinks=False)
        require(
            expected_identity(relative_info)
            and expected_identity(absolute_info)
            and opened.path.resolve(strict=True) == opened.path,
            f"leaf path custody changed for {opened.path}",
        )
        require(
            (
                parent_info.st_dev,
                parent_info.st_ino,
                parent_info.st_mtime_ns,
                parent_info.st_ctime_ns,
            )
            == expected_parent
            and stat.S_ISDIR(parent_path_info.st_mode)
            and (
                parent_path_info.st_dev,
                parent_path_info.st_ino,
                parent_path_info.st_mtime_ns,
                parent_path_info.st_ctime_ns,
            )
            == expected_parent,
            f"parent directory custody changed for {opened.path}",
        )
    except OSError as error:
        raise InvalidEvidence(
            f"final custody open failed for {opened.path}: {error}"
        ) from error
    finally:
        if relative_fd >= 0:
            os.close(relative_fd)
        if absolute_fd >= 0:
            os.close(absolute_fd)


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


def measure_directory(root: Path, maximum: int) -> dict[str, Any]:
    before = os.stat(root, follow_symlinks=False)
    require(stat.S_ISDIR(before.st_mode), "measured build root is not a directory")
    total = 0
    entries = 0
    stack = [root]
    seen_directories = {(before.st_dev, before.st_ino)}
    seen_files: set[tuple[int, int]] = set()
    manifest: list[tuple[Any, ...]] = []
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as iterator:
            rows = sorted(iterator, key=lambda row: os.fsencode(row.name))
        for row in rows:
            entries += 1
            require(entries <= 200000, "build root entry cap exceeded")
            info = row.stat(follow_symlinks=False)
            require(not stat.S_ISLNK(info.st_mode), "build root contains a symlink")
            identity = (info.st_dev, info.st_ino)
            relative = str(Path(row.path).relative_to(root))
            if stat.S_ISDIR(info.st_mode):
                require(
                    identity not in seen_directories, "build root directory aliases"
                )
                seen_directories.add(identity)
                stack.append(Path(row.path))
                kind = "directory"
            elif stat.S_ISREG(info.st_mode):
                require(
                    info.st_nlink == 1 and identity not in seen_files,
                    "build root file aliases or is hard-linked",
                )
                seen_files.add(identity)
                total += info.st_size
                require(total <= maximum, "build root measured bytes exceed cap")
                kind = "file"
            else:
                raise InvalidEvidence("build root contains a special file")
            manifest.append(
                (
                    relative,
                    kind,
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_nlink,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
            )
    after = os.stat(root, follow_symlinks=False)
    require(
        (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        "build root custody changed while measuring",
    )
    return {
        "root": (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ),
        "entries": tuple(sorted(manifest)),
        "entry_count": entries,
        "bytes": total,
    }


def measure_directory_bytes(root: Path, maximum: int) -> int:
    return measure_directory(root, maximum)["bytes"]


def validate_build_info_report(
    value: Any, checkout: dict[str, Any], build: dict[str, Any]
) -> dict[str, Any]:
    info = exact_keys(value, BUILD_INFO_REPORT_KEYS, "build-info report")
    expected_source = f"git-source-sha256-v2:{build['source_sha256']}"
    require(
        info["schema_version"] == 2
        and info["build_commit"] == checkout["commit"]
        and info["build_commit_short"] == checkout["commit"][:9]
        and info["build_dirty"] is False
        and info["build_source_state"] == expected_source
        and info["stamp_source"] == "git"
        and info["stamp_error"] is None
        and info["runtime_commit"] == checkout["commit"]
        and info["runtime_dirty"] is False
        and info["runtime_source_state"] == expected_source
        and info["status"] == "match"
        and info["problems"] == []
        and info["overrides"] == [],
        "build-info report mismatch",
    )
    return info


def validate_build_identity_report(
    value: Any,
    claim: dict[str, Any],
    run_id: str,
    checkout: dict[str, Any],
    build: dict[str, Any],
    sources: list[dict[str, Any]],
    executable: dict[str, Any],
    metallib: dict[str, Any],
    reducer: dict[str, Any],
    measurement_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = exact_keys(value, BUILD_REPORT_KEYS, "build identity report")
    require(
        report["schema"] == BUILD_REPORT_SCHEMA
        and report["schema_version"] == 1
        and report["authority"] == BUILD_REPORT_AUTHORITY
        and report["run_id"] == run_id,
        "build identity report schema/authority/run mismatch",
    )
    bounded_ascii(report["attempt_id"], "build report attempt", 128)
    require(report["checkout"] == checkout, "build report checkout mismatch")
    checkout_path = Path(checkout["path"])
    require(
        isinstance(report["build_command"], list)
        and report["build_command"]
        and all(isinstance(v, str) and v for v in report["build_command"]),
        "build report command invalid",
    )
    root = exact_keys(report["build_root"], BUILD_ROOT_KEYS, "build root")
    build_root_path = Path(text(root["path"], "build root path"))
    require(
        build_root_path.is_absolute()
        and build_root_path.resolve(strict=True) == build_root_path,
        "build root path is not canonical absolute",
    )
    require(build_root_path.is_dir(), "build root is not an existing directory")
    require(
        build_root_path.is_relative_to(checkout_path),
        "build root is not contained under worktree X",
    )
    root_maximum = integer(root["max_bytes"], "build root max", 1, MAX_BUILD_ROOT_BYTES)
    measurement_before = measure_directory(build_root_path, root_maximum)
    measured_root_bytes = measurement_before["bytes"]
    integer(root["bytes"], "build root bytes")
    require(
        root["bytes"] == measured_root_bytes,
        "build root measured-byte claim mismatch",
    )
    build_command = report["build_command"]
    cargo_path = Path(build_command[0])
    require(
        cargo_path.is_absolute()
        and cargo_path.resolve(strict=True) == cargo_path
        and cargo_path.is_file()
        and os.access(cargo_path, os.X_OK)
        and build_command[1:] == RUST_BUILD_COMMAND_SUFFIX,
        "build report cargo argv suffix/order mismatch",
    )
    for key, expected in (
        ("executable", executable),
        ("embedded_metallib", metallib),
        ("reducer", reducer),
    ):
        identity_from_claim(report[key], f"build report {key}")
        require(report[key] == expected, f"build report {key} mismatch")
    require(report["sources"] == sources, "build report source claims mismatch")
    require(
        Path(executable["path"]) == build_root_path / "release" / "qwen-bench",
        "build executable is not exact root/release/qwen-bench",
    )
    for label, item in (
        ("embedded metallib", metallib),
        ("reducer", reducer),
        *((f"source {item['role']}", item) for item in sources),
    ):
        require(
            Path(item["path"]).is_relative_to(checkout_path),
            f"build report {label} path escapes worktree X",
        )
    compiler = exact_keys(
        report["compiler"], BUILD_REPORT_COMPILER_KEYS, "build report compiler"
    )
    compiler_path = Path(text(compiler["path"], "compiler path"))
    require(
        compiler_path.is_absolute()
        and compiler_path.resolve(strict=True) == compiler_path,
        "compiler path is not canonical",
    )
    integer(compiler["bytes"], "compiler bytes", 1, MAX_COMPILER_BYTES)
    sha256_text(compiler["sha256"], "compiler sha256")
    text(compiler["version_verbose"], "compiler verbose version", maximum=1 << 20)
    require(
        hashlib.sha256(compiler["version_verbose"].encode("utf-8")).hexdigest()
        == sha256_text(compiler["version_verbose_sha256"], "compiler version digest"),
        "compiler verbose-version digest mismatch",
    )
    require(
        report["target"] == "aarch64-apple-darwin"
        and report["profile"] == "release"
        and report["features"] == ["dflash-k0s-diagnostics"]
        and report["target"] == build["target"]
        and report["profile"] == build["profile"]
        and report["features"] == build["features"]
        and compiler["path"] == build["compiler"]
        and compiler["version_verbose"] == build["compiler_version"],
        "build report target/profile/features mismatch",
    )
    info = validate_build_info_report(report["build_info"], checkout, build)
    identity_from_claim(info["artifact"], "build-info artifact")
    require(
        report["environment"]
        == {
            "CARGO_TARGET_DIR": str(build_root_path),
            "QWEN_METAL_LEASE_WAIT": "1",
        }
        and list(report["environment"])
        == [
            "CARGO_TARGET_DIR",
            "QWEN_METAL_LEASE_WAIT",
        ],
        "build report environment map is not exact",
    )
    require(
        Path(info["artifact"]["path"]).is_relative_to(checkout_path),
        "build-info artifact escapes worktree X",
    )
    measurement_after = measure_directory(build_root_path, root_maximum)
    require(
        measurement_after == measurement_before
        and measurement_after["bytes"] == root["bytes"],
        "build root changed across report validation",
    )
    if measurement_out is not None:
        measurement_out.update(
            {
                "path": build_root_path,
                "maximum": root_maximum,
                "measurement": measurement_before,
            }
        )
    return report


def validate_bootstrap_asset_expectations(value: Any) -> list[dict[str, Any]]:
    require(
        isinstance(value, list) and len(value) == 2,
        "inventory requires target and drafter asset expectations",
    )
    assets = [
        exact_keys(item, ASSET_EXPECTATION_KEYS, f"asset expectation {index}")
        for index, item in enumerate(value)
    ]
    require(
        assets
        == [
            {
                "role": "target",
                "path": BOOTSTRAP_TARGET_PATH,
                "expected_bytes": BOOTSTRAP_TARGET_BYTES,
                "max_bytes": BOOTSTRAP_TARGET_BYTES,
                "sha256": BOOTSTRAP_TARGET_SHA256,
            },
            {
                "role": "drafter",
                "path": BOOTSTRAP_DRAFTER_PATH,
                "expected_bytes": None,
                "max_bytes": BOOTSTRAP_DRAFTER_MAX_BYTES,
                "sha256": BOOTSTRAP_DRAFTER_SHA256,
            },
        ],
        "inventory asset predicates differ from frozen Q4 bootstrap identities",
    )
    return assets


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
        spec_file = open_regular(spec_path, "inventory spec", MAX_BOOTSTRAP_SPEC_BYTES)
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
            "inventory spec v2",
        )
        require(
            spec["schema"] == INVENTORY_SPEC_SCHEMA and spec["schema_version"] == 2,
            "inventory spec must be incompatible v2",
        )
        integer(spec["inventory_max_bytes"], "inventory max bytes", 1, MAX_TRACE_BYTES)
        require(
            inventory_file.size <= spec["inventory_max_bytes"],
            "inventory exceeds authenticated cap",
        )
        inventory = exact_keys(
            parse_json(
                pread_exact(inventory_file, 0, inventory_file.size, "inventory"),
                "inventory",
            ),
            INVENTORY_KEYS,
            "inventory artifact v2",
        )
        reject_forbidden(inventory)
        require(
            inventory["schema"] == INVENTORY_SCHEMA
            and inventory["schema_version"] == INVENTORY_VERSION == 2
            and inventory["authority"] == INVENTORY_AUTHORITY
            and inventory["inventory_spec_sha256"] == spec_sha256
            and inventory["run_id"] == spec["run_id"],
            "inventory v2 schema/authority/spec/run mismatch",
        )
        expected = exact_keys(
            inventory["expected"], INVENTORY_EXPECTED_KEYS, "inventory expected"
        )
        require(
            expected == {key: spec[key] for key in INVENTORY_EXPECTED_KEYS},
            "inventory expected fields do not exactly repeat authenticated spec",
        )
        observed = exact_keys(
            inventory["observed"], INVENTORY_OBSERVED_KEYS, "inventory observed"
        )
        checkout = exact_keys(
            spec["checkout"], INVENTORY_CHECKOUT_KEYS, "inventory checkout"
        )
        require(
            observed["checkout"] == checkout and checkout["dirty"] is False,
            "inventory checkout ownership mismatch",
        )
        checkout_path = Path(text(checkout["path"], "inventory checkout path"))
        require(
            checkout_path.is_absolute()
            and checkout_path.resolve(strict=True) == checkout_path,
            "inventory checkout path is not canonical absolute",
        )
        git_oid_text(checkout["commit"], "inventory checkout commit")
        git_oid_text(checkout["tree"], "inventory checkout tree")
        git_x = inspect_git_checkout(checkout_path, "inventory worktree X")
        require(
            git_x["head"] == checkout["commit"] and git_x["tree"] == checkout["tree"],
            "inventory checkout differs from actual clean X",
        )
        build = exact_keys(spec["build"], BUILD_KEYS, "inventory build")
        require(
            observed["build"] == build
            and build["commit"] == checkout["commit"]
            and build["dirty"] is False
            and build["target"] == "aarch64-apple-darwin"
            and build["profile"] == "release"
            and build["features"] == ["dflash-k0s-diagnostics"],
            "inventory build mismatch",
        )
        git_oid_text(build["commit"], "inventory build commit")
        sha256_text(build["source_sha256"], "inventory build source digest")

        def open_claim(claim: dict[str, Any], name: str) -> OpenFile:
            identity_from_claim(claim, name)
            item = open_regular(Path(claim["path"]), name, claim["max_bytes"])
            verify_identity(item, claim, name)
            require(
                item.path not in {value.path for value in opened}
                and item.inode not in {value.inode for value in opened},
                "inventory claimed files alias",
            )
            opened.append(item)
            return item

        build_report_claim = identity_from_claim(
            spec["build_report"], "inventory build report"
        )
        require(
            build_report_claim["max_bytes"] <= MAX_BOOTSTRAP_SPEC_BYTES,
            "build report cap exceeds bootstrap authorization",
        )
        require(
            observed["build_report"] == build_report_claim,
            "observed build report claim mismatch",
        )
        report_file = open_claim(build_report_claim, "build identity report")
        source_claims = spec["sources"]
        require(
            isinstance(source_claims, list) and len(source_claims) == 18,
            "inventory requires exactly 18 source roles",
        )
        require(
            observed["sources"] == source_claims,
            "observed source claims differ from expected",
        )
        x_path = Path(checkout["path"])
        for index, ((required_role, relative), claim) in enumerate(
            zip(SOURCE_ROLE_PATHS, source_claims)
        ):
            item = exact_keys(
                claim, ("role",) + FILE_CLAIM_KEYS, f"source claim {index}"
            )
            require(item["role"] == required_role, "source role order/map mismatch")
            required_path = (x_path / relative).resolve(strict=True)
            require(
                Path(item["path"]) == required_path
                and required_path.is_relative_to(x_path),
                "source role path differs from exact X-relative map",
            )
            open_claim(
                {key: item[key] for key in FILE_CLAIM_KEYS}, f"source {required_role}"
            )
        for key in ("executable", "reducer", "embedded_metallib"):
            identity_from_claim(spec[key], f"inventory {key}")
            require(observed[key] == spec[key], f"observed {key} differs from expected")
            opened_item = open_claim(spec[key], f"inventory {key}")
            if key == "reducer":
                require(
                    opened_item.path == Path(__file__).resolve(),
                    "running reducer path differs from authenticated reducer claim",
                )
        report_json = parse_json(
            pread_exact(report_file, 0, report_file.size, "build report"),
            "build report",
        )
        build_root_custody: dict[str, Any] = {}
        validate_build_identity_report(
            report_json,
            build_report_claim,
            spec["run_id"],
            checkout,
            build,
            source_claims,
            spec["executable"],
            spec["embedded_metallib"],
            spec["reducer"],
            build_root_custody,
        )
        validate_ignored_entries(
            git_x,
            "inventory worktree X",
            allowed_root=Path(report_json["build_root"]["path"]),
        )
        compiler_claim = report_json["compiler"]
        compiler_file = open_regular(
            Path(compiler_claim["path"]),
            "build report compiler",
            MAX_COMPILER_BYTES,
        )
        require(
            compiler_file.size == compiler_claim["bytes"]
            and compiler_file.digest == compiler_claim["sha256"],
            "build report compiler file identity mismatch",
        )
        require(
            compiler_file.path not in {value.path for value in opened}
            and compiler_file.inode not in {value.inode for value in opened},
            "build report compiler aliases another input",
        )
        opened.append(compiler_file)
        build_info_claim = report_json["build_info"]["artifact"]
        build_info_file = open_claim(build_info_claim, "build-info artifact")
        opened_build_info = parse_json(
            pread_exact(
                build_info_file, 0, build_info_file.size, "opened build-info artifact"
            ),
            "opened build-info artifact",
        )
        reported_build_info = {
            key: report_json["build_info"][key]
            for key in BUILD_INFO_REPORT_KEYS
            if key != "artifact"
        }
        require(
            opened_build_info == reported_build_info,
            "opened build-info artifact differs from parsed build report",
        )

        assets = validate_bootstrap_asset_expectations(spec["assets"])
        observed_assets = observed["assets"]
        require(
            isinstance(observed_assets, list) and len(observed_assets) == 2,
            "inventory observed assets invalid",
        )
        asset_files: dict[str, OpenFile] = {}
        for index, (raw_expectation, raw_claim) in enumerate(
            zip(assets, observed_assets)
        ):
            expectation = exact_keys(
                raw_expectation, ASSET_EXPECTATION_KEYS, f"asset expectation {index}"
            )
            claim = exact_keys(
                raw_claim, ("role",) + FILE_CLAIM_KEYS, f"observed asset {index}"
            )
            role = ("target", "drafter")[index]
            require(
                expectation["role"] == claim["role"] == role,
                "asset role/order substitution",
            )
            expected_bytes = expectation["expected_bytes"]
            require(
                expected_bytes is None
                or (
                    isinstance(expected_bytes, int)
                    and not isinstance(expected_bytes, bool)
                    and expected_bytes >= 0
                ),
                "asset expected_bytes must be integer or null",
            )
            maximum = integer(expectation["max_bytes"], "asset max bytes", 1)
            sha256_text(expectation["sha256"], "asset expected sha256")
            require(
                Path(expectation["path"]) == Path(claim["path"])
                and claim["sha256"] == expectation["sha256"]
                and claim["max_bytes"] == maximum
                and claim["bytes"] <= maximum
                and (expected_bytes is None or claim["bytes"] == expected_bytes),
                "asset expectation/observation mismatch",
            )
            asset_files[role] = open_claim(
                {key: claim[key] for key in FILE_CLAIM_KEYS}, f"asset {role}"
            )
        ggufs = {role: GGUF(item) for role, item in asset_files.items()}
        gguf_facts = observed["gguf"]
        require(
            isinstance(gguf_facts, list) and len(gguf_facts) == 2,
            "observed GGUF facts invalid",
        )
        for index, role in enumerate(("target", "drafter")):
            fact = exact_keys(
                gguf_facts[index], INVENTORY_GGUF_KEYS, f"GGUF fact {role}"
            )
            require(
                fact
                == {
                    "role": role,
                    "version": 3,
                    "tensor_count": len(ggufs[role].tensors),
                    "metadata_count": len(ggufs[role].metadata),
                },
                "independently parsed GGUF counts mismatch",
            )
        requirements = spec["tensor_requirements"]
        require(
            isinstance(requirements, list) and len(requirements) == 3,
            "tensor requirements invalid",
        )
        expected_requirements = (
            (
                "selector_hidden",
                "selector_hidden.weight",
                [HIDDEN, RANK],
                "gguf_ne0_hidden_ne1_rank",
                None,
            ),
            (
                "predecessor",
                "selector_predecessor.weight",
                [RANK, VOCAB],
                "gguf_ne0_rank_ne1_token",
                {"first": 0, "count": VOCAB},
            ),
            (
                "successor",
                "selector_successor.weight",
                [RANK, VOCAB],
                "gguf_ne0_rank_ne1_token",
                {"first": 0, "count": VOCAB},
            ),
        )
        for raw, expected_requirement in zip(requirements, expected_requirements):
            requirement = exact_keys(
                raw, TENSOR_REQUIREMENT_KEYS, "tensor requirement predicate"
            )
            role, tensor_name, shape, orientation, row_domain = expected_requirement
            require(
                requirement
                == {
                    "role": role,
                    "asset_role": "drafter",
                    "name": tensor_name,
                    "dtype": "Q4_K",
                    "shape": shape,
                    "orientation": orientation,
                    "row_domain": row_domain,
                },
                "Q4 tensor requirement differs from frozen descriptor predicate",
            )
        observed_tensors = observed["tensors"]
        require(
            isinstance(observed_tensors, list) and len(observed_tensors) == 3,
            "observed tensors invalid",
        )
        tensor_by_role: dict[str, dict[str, Any]] = {}
        for requirement_raw, claim_raw in zip(requirements, observed_tensors):
            requirement = exact_keys(
                requirement_raw, TENSOR_REQUIREMENT_KEYS, "tensor requirement"
            )
            claim = validate_tensor_claim(claim_raw, "observed tensor")
            for key in TENSOR_REQUIREMENT_KEYS:
                require(
                    claim[key] == requirement[key],
                    "tensor observed descriptor differs from predicate",
                )
            actual = ggufs[claim["asset_role"]].tensors.get(claim["name"])
            require(
                actual is not None
                and list(actual.shape) == claim["shape"]
                and actual.dtype == claim["dtype"]
                and actual.offset == claim["offset"]
                and actual.size == claim["bytes"],
                "independently parsed tensor descriptor mismatch",
            )
            require(
                hash_range(
                    asset_files[claim["asset_role"]],
                    actual.offset,
                    actual.size,
                    actual.name,
                )
                == claim["sha256"],
                "independently hashed tensor region mismatch",
            )
            tensor_by_role[claim["role"]] = claim
        require(
            set(tensor_by_role) == {"selector_hidden", "predecessor", "successor"}
            and tensor_by_role["selector_hidden"]["dtype"] == "Q4_K",
            "Q4 selector tensor roles incomplete",
        )

        tokenizer_predicate = exact_keys(
            spec["tokenizer_predicate"], TOKENIZER_PREDICATE_KEYS, "tokenizer predicate"
        )
        require(
            tokenizer_predicate
            == {
                "vocab_size": VOCAB,
                "token_embd_name": "token_embd.weight",
                "token_embd_rank": 2,
                "token_embd_hidden": HIDDEN,
                "token_embd_vocab_axis": 1,
                "allowed_token_embd_dtypes": ["Q4_K"],
                "require_token_metadata": True,
                "metadata_identity_domain": "qwen.dflash_k0s.tokenizer_metadata.v1",
            },
            "tokenizer predicate differs from frozen v2 contract",
        )
        target = ggufs["target"]
        embedding = target.tensors.get("token_embd.weight")
        require(
            embedding is not None
            and len(embedding.shape) == 2
            and embedding.shape[0] == HIDDEN
            and embedding.shape[1] == VOCAB
            and embedding.dtype == "Q4_K",
            "target token embedding differs from v2 predicate",
        )
        metadata = target.metadata
        required_metadata = (
            "general.architecture",
            "tokenizer.ggml.model",
            "tokenizer.ggml.pre",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges",
        )
        require(
            all(key in metadata for key in required_metadata),
            "required tokenizer metadata absent; fallback forbidden",
        )
        architecture, model, pre = (metadata[key] for key in required_metadata[:3])
        tokens, token_types, merges = (metadata[key] for key in required_metadata[3:])
        require(
            isinstance(tokens, list)
            and len(tokens) == VOCAB
            and isinstance(token_types, list)
            and len(token_types) == VOCAB
            and isinstance(merges, list),
            "tokenizer metadata arrays invalid",
        )
        token_digest = tokenizer_string_array_digest(
            "qwen.dflash_k0s.tokenizer.tokens.v1", tokens
        )
        type_digest = tokenizer_i64_array_digest(
            "qwen.dflash_k0s.tokenizer.token_type.v1", token_types
        )
        merges_digest = tokenizer_string_array_digest(
            "qwen.dflash_k0s.tokenizer.merges.v1", merges
        )
        bos = metadata.get("tokenizer.ggml.bos_token_id")
        eos = metadata.get("tokenizer.ggml.eos_token_id")
        add_bos = metadata.get("tokenizer.ggml.add_bos_token")
        add_eos = metadata.get("tokenizer.ggml.add_eos_token")
        require(
            (add_bos is None or isinstance(add_bos, bool))
            and (add_eos is None or isinstance(add_eos, bool)),
            "tokenizer add-BOS/add-EOS metadata flags are invalid",
        )
        identity = tokenizer_metadata_identity(
            architecture,
            model,
            pre,
            len(tokens),
            token_digest,
            len(token_types),
            type_digest,
            len(merges),
            merges_digest,
            bos,
            eos,
            add_bos,
            add_eos,
        )
        tokenizer = exact_keys(
            observed["tokenizer"], TOKENIZER_OBSERVATION_KEYS, "tokenizer observation"
        )
        require(
            tokenizer
            == {
                "vocab_size": VOCAB,
                "token_embd_name": "token_embd.weight",
                "token_embd_shape": list(embedding.shape),
                "token_embd_dtype": "Q4_K",
                "token_count": VOCAB,
                "model": model,
                "pre": pre,
                "bos_token_id": bos,
                "eos_token_id": eos,
                "add_bos_token": add_bos,
                "add_eos_token": add_eos,
                "token_list_sha256": token_digest,
                "token_type_sha256": type_digest,
                "merges_sha256": merges_digest,
                "metadata_identity_sha256": identity,
            },
            "tokenizer observation not independently reproduced",
        )
        prompt_predicate = exact_keys(
            spec["prompt_predicate"], PROMPT_PREDICATE_KEYS, "prompt predicate"
        )
        prompt_bytes = bytes.fromhex(prompt_predicate["utf8_hex"])
        require(
            prompt_bytes == b"Write code"
            and hashlib.sha256(prompt_bytes).hexdigest()
            == prompt_predicate["utf8_sha256"]
            and prompt_predicate["add_special"] is False
            and prompt_predicate["expected_token_ids"] == [7734, 1970]
            and prompt_predicate["expected_token_ids_sha256_i32le"]
            == hashlib.sha256(struct.pack("<ii", 7734, 1970)).hexdigest(),
            "prompt predicate mismatch",
        )
        prompt = exact_keys(
            observed["prompt"], PROMPT_OBSERVATION_KEYS, "prompt observation"
        )
        require(
            prompt
            == {
                "utf8_hex": prompt_predicate["utf8_hex"],
                "add_special": False,
                "token_ids": [7734, 1970],
                "token_ids_sha256_i32le": prompt_predicate[
                    "expected_token_ids_sha256_i32le"
                ],
                "tokenizer_metadata_identity_sha256": identity,
            },
            "prompt observation differs from fixed IDs/authenticated tokenizer identity",
        )
        mask_predicate = exact_keys(
            spec["mask_predicate"], MASK_PREDICATE_KEYS, "mask predicate"
        )
        require(
            mask_predicate
            == {
                "allowed_metadata_keys": [
                    "dflash-draft.dflash.mask_token_id",
                    "tokenizer.ggml.mask_token_id",
                ],
                "expected_mask_token": 248070,
            },
            "mask predicate mismatch",
        )
        present_masks = [
            (key, ggufs["drafter"].metadata[key])
            for key in mask_predicate["allowed_metadata_keys"]
            if key in ggufs["drafter"].metadata
        ]
        require(
            len(present_masks) == 1 and present_masks[0][1] == 248070,
            "drafter mask metadata mismatch",
        )
        drafter_metadata = ggufs["drafter"].metadata

        def metadata_u64(*keys: str) -> int | None:
            for key in keys:
                value = drafter_metadata.get(key)
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                ):
                    return value
            return None

        require(
            metadata_u64("dflash.block_size", "dflash-draft.dflash.block_size")
            == DEPTHS + 1
            and metadata_u64("dflash.embedding_length", "dflash-draft.embedding_length")
            == HIDDEN
            and metadata_u64("dflash.selector_rank") == RANK
            and metadata_u64("dflash.selector_top_k") == TOP_K,
            "drafter K0-S geometry metadata mismatch",
        )
        command = spec["command"]
        require(
            isinstance(command, list), "inventory command template is not an argv array"
        )
        carry = 364
        noise = [carry] + [248070] * 7
        mask_noise = exact_keys(
            observed["mask_noise"], INVENTORY_MASK_KEYS, "mask/noise observation"
        )
        require(
            mask_noise
            == {
                "metadata_key": present_masks[0][0],
                "mask_token": 248070,
                "noise_tokens": noise,
                "noise_sha256_i32le": hashlib.sha256(
                    b"".join(struct.pack("<i", value) for value in noise)
                ).hexdigest(),
            },
            "mask/noise observation mismatch",
        )
        host_predicate = exact_keys(
            spec["host_predicate"], HOST_PREDICATE_V2_KEYS, "host predicate"
        )
        require(
            host_predicate
            == {
                "os": "macos",
                "arch": "aarch64",
                "device_name": "Apple M4 Max",
                "required_families": ["apple9", "mac2", "common3", "metal3"],
                "family_match": "all",
            },
            "host predicate mismatch",
        )
        device = exact_keys(observed["device"], HOST_KEYS, "device observation")
        integer(
            device["device_registry_id"],
            "device registry id",
            0,
            MAX_JSON_U64,
        )
        family = text(device["device_family"], "device family", maximum=4096)
        require(
            family.startswith("mtl-gpu-family-v1:"),
            "device family observation domain mismatch",
        )
        families = family.removeprefix("mtl-gpu-family-v1:").split(",")
        require(
            len(families) == len(set(families))
            and all(
                families.count(value) == 1
                for value in host_predicate["required_families"]
            )
            and device["os"] == host_predicate["os"]
            and device["arch"] == host_predicate["arch"]
            and device["device_name"] == host_predicate["device_name"],
            "device does not satisfy all-of host family predicate",
        )
        require(
            exact_keys(spec["parser_caps"], PARSER_CAP_KEYS, "inventory parser caps")
            == {
                "header_bytes": MAX_GGUF_HEADER_BYTES,
                "metadata": MAX_GGUF_METADATA,
                "tensors": MAX_GGUF_TENSORS,
                "strings_bytes": MAX_GGUF_STRINGS_BYTES,
                "array_items_per_array": MAX_GGUF_ARRAY_ITEMS_PER_ARRAY,
                "array_items": MAX_GGUF_ARRAY_ITEMS,
                "objects": MAX_GGUF_OBJECTS,
            }
            and observed["parser_caps"] == spec["parser_caps"],
            "observed parser caps differ from expected",
        )
        expected_command = [
            spec["executable"]["path"],
            "dflash-k0s-inventory",
            "--model",
            str(asset_files["target"].path),
            "--drafter",
            str(asset_files["drafter"].path),
            "--prompt",
            "Write code",
            "--carry-token",
            "364",
            "--inventory-spec",
            str(spec_file.path),
            "--inventory-spec-sha256",
            spec_sha256,
            "--output",
            str(inventory_file.path),
        ]
        expected_template = list(expected_command)
        expected_template[13] = "${INVENTORY_SPEC_SHA256}"
        expected_template[15] = "${INVENTORY_OUTPUT}"
        require(
            inventory["command"] == expected_command
            and command == expected_template
            and inventory["environment"]
            == spec["environment"]
            == {"QWEN_METAL_LEASE_WAIT": "1"},
            "inventory command/environment mismatch",
        )
        require(
            measure_directory(build_root_custody["path"], build_root_custody["maximum"])
            == build_root_custody["measurement"],
            "build root changed across the complete inventory operation",
        )
        for item in opened:
            final_custody_check(item)
        if retain_open:
            retained = True
            return InventoryContext(inventory, opened, build_root_custody)
        return inventory
    except (
        KeyError,
        TypeError,
        ValueError,
        struct.error,
        OverflowError,
        OSError,
    ) as error:
        if isinstance(error, InvalidEvidence):
            raise
        raise InvalidEvidence(
            f"invalid inventory v2: {type(error).__name__}: {error}"
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
        "--preparation-choices",
        spec["preparation_choices"]["path"],
        "--preparation-choices-sha256",
        PREPARATION_CHOICES_SHA256_PLACEHOLDER,
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
    preparation_choices_sha256: str,
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
        else preparation_choices_sha256
        if value == PREPARATION_CHOICES_SHA256_PLACEHOLDER
        else seal_sha256
        if value == SEAL_SHA256_PLACEHOLDER
        else value
        for value in frozen
    ]
    require(len(actual) == len(expected), "actual reducer argv length mismatch")
    require(
        actual[0] == expected[0] == reducer_path,
        "reducer argv[0] literal spelling differs from authenticated reducer path",
    )
    require(
        actual[1:] == expected[1:],
        "actual reducer argv literal order/spelling/value mismatch",
    )


def validate_literal_mode_argv(
    actual: list[str], expected: list[str], reducer_path: str
) -> None:
    require(
        isinstance(actual, list)
        and isinstance(expected, list)
        and all(isinstance(value, str) and value for value in actual + expected),
        "mode argv must contain literal nonempty strings",
    )
    require(len(actual) == len(expected), "mode argv literal length mismatch")
    require(
        actual[0] == expected[0] == reducer_path,
        "mode argv[0] literal spelling differs from authenticated reducer path",
    )
    require(
        actual[1:] == expected[1:],
        "mode argv literal order/spelling/value mismatch",
    )


def validate_preparation_environment(environment: dict[str, str]) -> dict[str, str]:
    require(
        isinstance(environment, dict)
        and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ),
        "process environment must be a string map",
    )
    affecting = {
        key: value
        for key, value in environment.items()
        if key.startswith(("QWEN_", "MTL_", "METAL_", "GGML_METAL_", "DYLD_"))
    }
    require(
        affecting == {"QWEN_METAL_LEASE_WAIT": "1"},
        "preparation process environment differs from exact behavior allowlist",
    )
    require(
        "PYTHONPATH" not in environment and "PYTHONHOME" not in environment,
        "preparation process environment contains Python path injection",
    )
    uv_environment = {
        key: value for key, value in environment.items() if key.startswith("UV_")
    }
    require(
        uv_environment in ({}, {"UV_RUN_RECURSION_DEPTH": "1"}),
        "preparation process environment contains unexpected uv configuration",
    )
    return {"QWEN_METAL_LEASE_WAIT": "1"}


def validate_preparation_nested_order(
    value: dict[str, Any], *, is_spec: bool, name: str
) -> None:
    exact_keys(
        value,
        PREPARATION_SPEC_KEYS if is_spec else PREPARATION_CHOICES_KEYS,
        name,
    )
    if is_spec:
        exact_keys(value["preparation_choices"], FILE_CLAIM_KEYS, f"{name}.choices")
    exact_keys(value["worktree_x"], ("path", "commit"), f"{name}.worktree_x")
    exact_keys(value["planner_p"], ("path",), f"{name}.planner_p")
    exact_keys(
        value["control_y_input"] if is_spec else value["control_y"],
        ("path", "commit", "tree"),
        f"{name}.control_y",
    )
    exact_keys(
        value["outputs"],
        ("fixture", "command", "manifest", "seal"),
        f"{name}.outputs",
    )
    fixture = exact_keys(
        value["fixture_content"],
        ("schema", "schema_version", "fixture_domain", "fixture_sha256", "vectors"),
        f"{name}.fixture_content",
    )
    require(isinstance(fixture["vectors"], list), f"{name}.fixture vectors invalid")
    for index, vector in enumerate(fixture["vectors"]):
        exact_keys(vector, SCALAR_VECTOR_KEYS, f"{name}.fixture vector {index}")
    exact_keys(
        value["acquisition_outputs"],
        ACQUISITION_OUTPUT_KEYS,
        f"{name}.acquisition_outputs",
    )
    choices = exact_keys(
        value["manifest_choices"], MANIFEST_CHOICE_KEYS, f"{name}.manifest_choices"
    )
    references = exact_keys(
        choices["semantic_references"],
        tuple(SEMANTIC_REFERENCES),
        f"{name}.semantic_references",
    )
    for role, reference in references.items():
        exact_keys(reference, ("commit", "sha256"), f"{name}.reference.{role}")
    expected_request = exact_keys(
        choices["expected_request"],
        ("request", "ignored_target_policy"),
        f"{name}.expected_request",
    )
    exact_keys(expected_request["request"], REQUEST_KEYS, f"{name}.request")
    ignored = exact_keys(
        expected_request["ignored_target_policy"],
        ("variant_a", "variant_b"),
        f"{name}.ignored_target_policy",
    )
    for variant, policy in ignored.items():
        exact_keys(policy, IGNORED_POLICY_KEYS, f"{name}.{variant}")
    exact_keys(choices["expected_binding"], BINDING_KEYS, f"{name}.expected_binding")
    require(
        isinstance(choices["expected_fixed_chains"], list),
        f"{name}.fixed chains invalid",
    )
    for index, chain in enumerate(choices["expected_fixed_chains"]):
        exact_keys(chain, STATIC_CHAIN_KEYS, f"{name}.fixed chain {index}")
    exact_keys(
        choices["expected_capture_context"],
        CAPTURE_CONTEXT_KEYS,
        f"{name}.capture_context",
    )
    predicate = exact_keys(
        choices["selector_dispatch_predicate"],
        SELECTOR_PREDICATE_KEYS,
        f"{name}.selector predicate",
    )
    exact_keys(
        predicate["allowed_environment"],
        ("QWEN_METAL_LEASE_WAIT",),
        f"{name}.selector environment",
    )
    exact_keys(
        value["environment_allowlist"],
        ("QWEN_METAL_LEASE_WAIT",),
        f"{name}.environment_allowlist",
    )
    exact_keys(
        value["failure_policy"],
        ("on_collision", "retry"),
        f"{name}.failure_policy",
    )


def validate_preparation_join(
    spec: dict[str, Any],
    choices: dict[str, Any],
    choices_claim: dict[str, Any],
) -> None:
    validate_preparation_nested_order(spec, is_spec=True, name="preparation spec v2")
    validate_preparation_nested_order(
        choices, is_spec=False, name="preparation choices v1"
    )
    require(
        spec["schema"] == PREPARATION_SPEC_SCHEMA
        and spec["schema_version"] == PREPARATION_SPEC_VERSION,
        "preparation spec must be incompatible v2",
    )
    require(
        choices["schema"] == PREPARATION_CHOICES_SCHEMA
        and choices["schema_version"] == PREPARATION_CHOICES_VERSION,
        "preparation choices schema/version mismatch",
    )
    identity_from_claim(choices_claim, "preparation choices claim")
    require(
        spec["preparation_choices"] == choices_claim,
        "preparation spec choices claim mismatch",
    )
    direct = {
        "run_id": "run_id",
        "attempt_id": "attempt_id",
        "worktree_x": "worktree_x",
        "planner_p": "planner_p",
        "outputs": "outputs",
        "preparation_spec_path": "preparation_spec_path",
        "fixture_content": "fixture_content",
        "acquisition_outputs": "acquisition_outputs",
        "reduction_output": "reduction_output",
        "continuation_carry_token": "continuation_carry_token",
        "manifest_choices": "manifest_choices",
        "transformation_sha256": "transformation_sha256",
        "environment_allowlist": "environment_allowlist",
        "arm_order": "arm_order",
        "selected_arm": "selected_arm",
        "parity_comparison_fields": "parity_comparison_fields",
        "failure_policy": "failure_policy",
    }
    for spec_key, choices_key in direct.items():
        require(
            spec[spec_key] == choices[choices_key],
            f"preparation template join mutation at {spec_key}",
        )
    require(
        spec["control_y_input"] == choices["control_y"],
        "preparation template join mutation at control_y",
    )
    require(
        spec["preparation_spec_path"] == choices["preparation_spec_path"],
        "preparation spec self path differs from template",
    )
    replacements = {
        INVENTORY_PATH_PLACEHOLDER: spec["inventory_path"],
        INVENTORY_SHA256_PLACEHOLDER: spec["inventory_sha256"],
        INVENTORY_SPEC_PATH_PLACEHOLDER: spec["inventory_spec_path"],
        INVENTORY_SPEC_SHA256_PLACEHOLDER: spec["inventory_spec_sha256"],
    }
    template = choices["reducer_argv_template"]
    require(
        isinstance(template, list)
        and all(isinstance(value, str) and value for value in template),
        "preparation reducer argv template is invalid",
    )
    for placeholder, final_value in replacements.items():
        require(
            final_value not in template,
            f"preparation template contains literal final inventory value for {placeholder}",
        )
        require(
            template.count(placeholder) == 1,
            f"preparation inventory placeholder {placeholder} must occur exactly once",
        )
    require(
        not any(
            value.startswith("${INVENTORY_") and value not in replacements
            for value in template
        ),
        "preparation template contains an unknown inventory placeholder",
    )
    joined_argv = [replacements.get(value, value) for value in template]
    require(
        not any(value in replacements for value in joined_argv),
        "preparation join left an unresolved inventory placeholder",
    )
    require(
        spec["reducer_argv"] == joined_argv,
        "preparation reducer argv is not the exact four-field template join",
    )
    require(
        spec["reducer_argv"].count(PREPARATION_CHOICES_SHA256_PLACEHOLDER) == 1
        and choices["reducer_argv_template"].count(
            PREPARATION_CHOICES_SHA256_PLACEHOLDER
        )
        == 1,
        "preparation choices digest placeholder must occur exactly once",
    )
    serialized = json.dumps(spec, ensure_ascii=True, separators=(",", ":"))
    for forbidden in (
        "expected_fixture_sha256",
        "expected_command_sha256",
        "expected_static_manifest_sha256",
        "expected_seal_sha256",
    ):
        require(
            forbidden not in serialized,
            "expected rendered hash creates a template/output cycle",
        )


def validate_preparation_prerender(
    inventory: dict[str, Any],
    inventory_sha256: str,
    spec: dict[str, Any],
    choices: dict[str, Any],
    choices_claim: dict[str, Any],
) -> tuple[Path, tuple[Path, ...]]:
    validate_preparation_join(spec, choices, choices_claim)
    require(
        spec["inventory_sha256"] == inventory_sha256
        and spec["run_id"] == inventory["run_id"],
        "preparation inventory digest/run mismatch",
    )
    require(
        spec["arm_order"] == ["off-A", "on-A", "on-B", "off-B"]
        and spec["selected_arm"] == "on-A"
        and spec["parity_comparison_fields"] == PARITY_COMPARISON_FIELDS,
        "preparation arm/parity contract mismatch",
    )
    require(
        spec["environment_allowlist"] == {"QWEN_METAL_LEASE_WAIT": "1"},
        "preparation environment allowlist mismatch",
    )
    require(
        exact_keys(
            spec["failure_policy"],
            ("on_collision", "retry"),
            "preparation failure policy",
        )
        == {"on_collision": "retain_reserved_partial", "retry": False},
        "preparation failure policy mismatch",
    )
    exact_keys(
        spec["manifest_choices"], MANIFEST_CHOICE_KEYS, "preparation manifest choices"
    )
    facts = inventory["observed"]
    checkout = exact_keys(
        facts["checkout"], INVENTORY_CHECKOUT_KEYS, "inventory checkout"
    )
    worktree = exact_keys(spec["worktree_x"], ("path", "commit"), "worktree X")
    planner = exact_keys(spec["planner_p"], ("path",), "planner P")
    control = exact_keys(
        spec["control_y_input"], ("path", "commit", "tree"), "control Y input"
    )
    x_path = Path(worktree["path"])
    p_path = Path(planner["path"])
    y_path = Path(control["path"])
    for path, label in ((x_path, "X"), (p_path, "P"), (y_path, "Y")):
        require(
            path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
            f"preparation {label} path is not canonical existing directory",
        )
    require(len({x_path, p_path, y_path}) == 3, "preparation P/X/Y paths alias")
    git_x = inspect_git_checkout(x_path, "preparation worktree X")
    git_p = inspect_git_checkout(p_path, "preparation planner P")
    git_y = inspect_git_checkout(y_path, "preparation control Y")
    validate_ignored_entries(git_p, "preparation planner P")
    validate_ignored_entries(git_y, "preparation control Y")
    if git_x["ignored"]:
        validate_ignored_entries(
            git_x,
            "preparation worktree X",
            allowed_root=inventory_build_root(facts, x_path),
        )
    require(
        checkout
        == {
            "path": str(x_path),
            "commit": git_x["head"],
            "tree": git_x["tree"],
            "dirty": False,
        }
        and worktree == {"path": str(x_path), "commit": git_x["head"]},
        "preparation X differs from authenticated inventory checkout",
    )
    require(
        control["commit"] == git_y["head"]
        and control["tree"] == git_y["tree"]
        and git_x["common"] == git_p["common"] == git_y["common"]
        and git_x["objects"] == git_p["objects"] == git_y["objects"],
        "preparation P/X/Y Git relationship mismatch",
    )
    require(
        spec["transformation_sha256"] == facts["reducer"]["sha256"],
        "preparation transformation/reducer mismatch",
    )
    for claim in [
        facts["executable"],
        facts["reducer"],
        facts["embedded_metallib"],
        *facts["sources"],
    ]:
        claimed = Path(claim["path"])
        require(
            claimed.is_absolute()
            and claimed.resolve(strict=True) == claimed
            and claimed.is_relative_to(x_path),
            "preparation X identity path escapes worktree X",
        )
    outputs = exact_keys(
        spec["outputs"], ("fixture", "command", "manifest", "seal"), "outputs"
    )
    acquisition = exact_keys(
        spec["acquisition_outputs"], ACQUISITION_OUTPUT_KEYS, "acquisition outputs"
    )
    require(
        outputs["manifest"] == acquisition["manifest"],
        "prepared/acquisition manifest paths differ",
    )
    future_raw = [
        outputs["fixture"],
        outputs["command"],
        outputs["manifest"],
        outputs["seal"],
        acquisition["trace"],
        acquisition["sidecar"],
        spec["reduction_output"],
    ]
    future_paths: list[Path] = []
    for raw in future_raw:
        path = Path(raw)
        canonical = canonical_output_path(path)
        require(
            path.is_absolute() and path == canonical and path.parent == y_path,
            "preparation future output is not canonical/directly confined under Y",
        )
        future_paths.append(path)
    require(len(set(future_paths)) == 7, "preparation seven future leaves alias")
    require(
        not any(leaf_present(path) for path in future_paths),
        "preparation future leaf already exists",
    )
    require(
        spec["reducer_argv"] == frozen_reducer_argv(spec, facts["reducer"]["path"]),
        "preparation frozen reducer argv mismatch",
    )
    return y_path, tuple(future_paths)


def render_preparation(
    inventory: dict[str, Any],
    inventory_sha256: str,
    spec: dict[str, Any],
    spec_sha256: str,
    preparation_claims: dict[str, dict[str, Any]] | None = None,
    choices: dict[str, Any] | None = None,
    choices_claim: dict[str, Any] | None = None,
) -> tuple[bytes, bytes, bytes, bytes]:
    require(
        choices is not None and choices_claim is not None,
        "v2 preparation requires authenticated preparation choices",
    )
    validate_preparation_prerender(
        inventory, inventory_sha256, spec, choices, choices_claim
    )
    require(
        spec["inventory_sha256"] == inventory_sha256,
        "preparation inventory digest mismatch",
    )
    require(spec["run_id"] == inventory["run_id"], "preparation run id mismatch")
    inventory = {
        **inventory["expected"],
        **inventory["observed"],
        "run_id": inventory["run_id"],
    }
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
    planner = exact_keys(spec["planner_p"], ("path",), "planner P")
    x_raw = Path(worktree["path"])
    y_raw = Path(control["path"])
    p_raw = Path(planner["path"])
    x_path = x_raw.resolve(strict=True)
    y_path = y_raw.resolve(strict=True)
    p_path = p_raw.resolve(strict=True)
    require(
        x_raw.is_absolute()
        and y_raw.is_absolute()
        and p_raw.is_absolute()
        and x_raw == x_path
        and y_raw == y_path
        and p_raw == p_path
        and x_path.is_dir()
        and y_path.is_dir()
        and p_path.is_dir(),
        "preparation P/X/Y paths must be directories",
    )
    require(x_path != y_path, "preparation worktree-X and control Y must be distinct")
    require(
        len({x_path, y_path, p_path}) == 3,
        "preparation planner P/worktree X/control Y must be distinct",
    )
    git_x = inspect_git_checkout(x_path, "preparation worktree X")
    git_y = inspect_git_checkout(y_path, "preparation control Y", require_clean=False)
    git_p = inspect_git_checkout(p_path, "preparation planner P")
    require(
        git_x["common"] == git_y["common"] == git_p["common"]
        and git_x["objects"] == git_y["objects"] == git_p["objects"],
        "preparation P/X/Y do not share the authenticated Git repository/object store",
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
    output_paths: dict[str, str] = {}
    for key, value in outputs.items():
        raw_path = Path(value)
        canonical_path = canonical_output_path(raw_path)
        require(
            raw_path.is_absolute() and raw_path == canonical_path,
            f"prepared {key} output path is not canonical absolute",
        )
        output_paths[key] = str(canonical_path)
    acquisition_paths: dict[str, str] = {}
    for key, value in acquisition.items():
        raw_path = Path(value)
        canonical_path = canonical_output_path(raw_path)
        require(
            raw_path.is_absolute() and raw_path == canonical_path,
            f"acquisition {key} output path is not canonical absolute",
        )
        acquisition_paths[key] = str(canonical_path)
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
    reduction_raw = Path(spec["reduction_output"])
    reduction_output = str(canonical_output_path(reduction_raw))
    require(
        reduction_raw.is_absolute()
        and reduction_raw == Path(reduction_output)
        and Path(reduction_output).parent == y_path
        and reduction_output not in six_paths,
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
        "expected_prompt": {
            "utf8_hex": inventory["prompt"]["utf8_hex"],
            "token_ids": inventory["prompt"]["token_ids"],
            "token_ids_sha256_i32le": inventory["prompt"]["token_ids_sha256_i32le"],
            "tokenizer_identity_sha256": inventory["prompt"][
                "tokenizer_metadata_identity_sha256"
            ],
        },
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
            "preparation_choices": choices_claim,
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
        "preparation_choices_sha256": choices_claim["sha256"],
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


def preparation_file_claim(opened: OpenFile, maximum: int) -> dict[str, Any]:
    return {
        "path": str(opened.path),
        "bytes": opened.size,
        "sha256": opened.digest,
        "max_bytes": maximum,
    }


def validate_preparation_input_binding(
    spec: dict[str, Any],
    inventory_file: OpenFile,
    inventory_sha256: str,
    inventory_spec_file: OpenFile,
    inventory_spec_sha256: str,
    choices_file: OpenFile,
    choices_sha256: str,
    preparation_spec_file: OpenFile,
    preparation_spec_sha256: str,
) -> None:
    require(
        inventory_file.digest == inventory_sha256
        and inventory_spec_file.digest == inventory_spec_sha256
        and choices_file.digest == choices_sha256
        and preparation_spec_file.digest == preparation_spec_sha256,
        "preparation authenticated input digest mismatch",
    )
    require(
        spec["inventory_path"] == str(inventory_file.path)
        and spec["inventory_sha256"] == inventory_sha256
        and spec["inventory_spec_path"] == str(inventory_spec_file.path)
        and spec["inventory_spec_sha256"] == inventory_spec_sha256
        and spec["preparation_choices"]["path"] == str(choices_file.path)
        and spec["preparation_choices"]["sha256"] == choices_sha256
        and spec["preparation_spec_path"] == str(preparation_spec_file.path),
        "preparation paths/hashes differ from authenticated inputs",
    )


def git_report_identity(value: dict[str, Any]) -> dict[str, Any]:
    checkout = exact_keys(
        value,
        ("path", "head", "tree", "common", "objects", "status", "ignored"),
        "Git report source identity",
    )
    require(
        isinstance(checkout["status"], bytes)
        and isinstance(checkout["ignored"], bytes),
        "Git report status/ignored bindings must be bytes",
    )
    git_oid_text(checkout["head"], "Git report HEAD")
    git_oid_text(checkout["tree"], "Git report tree")
    identity = {
        "path": str(value["path"]),
        "head": value["head"],
        "tree": value["tree"],
        "status_bytes": len(value["status"]),
        "status_sha256": hashlib.sha256(value["status"]).hexdigest(),
        "ignored_bytes": len(value["ignored"]),
        "ignored_sha256": hashlib.sha256(value["ignored"]).hexdigest(),
        "common_git_dir": str(value["common"]),
        "object_store": str(value["objects"]),
    }
    exact_keys(identity, GIT_REPORT_IDENTITY_KEYS, "Git report identity")
    return identity


def observe_preparation_hashes(
    inventory_path: Path,
    inventory_sha256: str,
    inventory_spec_path: Path,
    inventory_spec_sha256: str,
    preparation_choices_path: Path,
    preparation_choices_sha256: str,
    preparation_spec_path: Path,
    preparation_spec_sha256: str,
    report_output: Path,
    report_max_bytes: int,
    observed_environment: dict[str, str],
) -> dict[str, Any]:
    require(
        observed_environment == validate_preparation_environment(dict(os.environ)),
        "observation environment was not authenticated from the current process",
    )
    require(
        report_max_bytes == MAX_PREPARATION_HASH_REPORT_BYTES,
        "preparation hash report cap must be exactly 65536",
    )
    context = validate_inventory(
        inventory_path,
        inventory_sha256,
        inventory_spec_path,
        inventory_spec_sha256,
        retain_open=True,
    )
    require(isinstance(context, InventoryContext), "inventory retention failed")
    prep_file = open_regular(
        preparation_spec_path, "preparation spec", MAX_BOOTSTRAP_SPEC_BYTES
    )
    choices_file = open_regular(
        preparation_choices_path, "preparation choices", MAX_BOOTSTRAP_SPEC_BYTES
    )
    report_fd: int | None = None
    planner_fd: int | None = None
    planner_stat: os.stat_result | None = None
    x_fd: int | None = None
    y_fd: int | None = None
    x_stat: os.stat_result | None = None
    y_stat: os.stat_result | None = None
    x_path: Path | None = None
    y_path: Path | None = None
    try:
        require(
            prep_file.digest == preparation_spec_sha256
            and choices_file.digest == preparation_choices_sha256,
            "preparation observation input digest mismatch",
        )
        spec = parse_canonical_json_object(
            prep_file, PREPARATION_SPEC_KEYS, "preparation spec"
        )
        choices = parse_canonical_json_object(
            choices_file, PREPARATION_CHOICES_KEYS, "preparation choices"
        )
        choices_claim = preparation_file_claim(choices_file, MAX_BOOTSTRAP_SPEC_BYTES)
        validate_preparation_join(spec, choices, choices_claim)
        retained = {item.path: item for item in context.opened}
        validate_preparation_input_binding(
            spec,
            retained[inventory_path.resolve(strict=True)],
            inventory_sha256,
            retained[inventory_spec_path.resolve(strict=True)],
            inventory_spec_sha256,
            choices_file,
            preparation_choices_sha256,
            prep_file,
            preparation_spec_sha256,
        )
        x_path = Path(spec["worktree_x"]["path"])
        y_path = Path(spec["control_y_input"]["path"])
        x_fd, x_stat = open_directory_custody(x_path, "observation worktree X")
        y_fd, y_stat = open_directory_custody(y_path, "observation control Y")
        verify_directory_custody(
            x_fd, x_path, x_stat, "observation worktree X", metadata_stable=True
        )
        verify_directory_custody(
            y_fd, y_path, y_stat, "observation control Y", metadata_stable=True
        )
        planner_early = exact_keys(spec["planner_p"], ("path",), "planner P")
        planner_early_path = Path(planner_early["path"])
        require(
            planner_early_path.is_absolute()
            and planner_early_path.resolve(strict=True) == planner_early_path,
            "planner P path is not canonical absolute",
        )
        planner_fd = os.open(
            planner_early_path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        planner_stat = os.fstat(planner_fd)
        require(
            stat.S_ISDIR(planner_stat.st_mode),
            "planner P custody FD is not a directory",
        )
        planner_path_stat = os.stat(planner_early_path, follow_symlinks=False)
        require(
            (planner_path_stat.st_dev, planner_path_stat.st_ino)
            == (planner_stat.st_dev, planner_stat.st_ino),
            "planner P path/FD identity mismatch",
        )
        preparation_claims = {
            "inventory": preparation_file_claim(
                retained[inventory_path.resolve(strict=True)], MAX_TRACE_BYTES
            ),
            "inventory_spec": preparation_file_claim(
                retained[inventory_spec_path.resolve(strict=True)],
                MAX_BOOTSTRAP_SPEC_BYTES,
            ),
            "preparation_spec": preparation_file_claim(
                prep_file, MAX_BOOTSTRAP_SPEC_BYTES
            ),
        }
        rendered = render_preparation(
            context.inventory,
            inventory_sha256,
            spec,
            preparation_spec_sha256,
            preparation_claims,
            choices,
            choices_claim,
        )
        outputs = exact_keys(
            spec["outputs"],
            ("fixture", "command", "manifest", "seal"),
            "preparation outputs",
        )
        acquisition = exact_keys(
            spec["acquisition_outputs"], ACQUISITION_OUTPUT_KEYS, "acquisition outputs"
        )
        future_paths = [
            canonical_output_path(Path(outputs[name]))
            for name in ("fixture", "command", "manifest", "seal")
        ] + [
            canonical_output_path(Path(acquisition["trace"])),
            canonical_output_path(Path(acquisition["sidecar"])),
            canonical_output_path(Path(spec["reduction_output"])),
        ]
        require(
            len(set(future_paths)) == 7
            and not any(leaf_present(path) for path in future_paths)
            and not set(future_paths)
            & {item.path for item in [*context.opened, prep_file, choices_file]},
            "all seven future preparation/acquisition/reduction leaves must be absent",
        )
        planner = exact_keys(spec["planner_p"], ("path",), "planner P")
        planner_path = Path(planner["path"]).resolve(strict=True)
        require(
            planner_path == planner_early_path, "planner P path changed during render"
        )
        report_path = canonical_output_path(report_output)
        require(
            report_output.is_absolute()
            and report_output == report_path
            and report_path.parent == planner_path
            and report_path not in future_paths
            and report_path
            not in {item.path for item in [*context.opened, prep_file, choices_file]},
            "preparation report path is not uniquely confined under planner P",
        )
        report_preexisting = leaf_present(report_path)
        git_p = inspect_git_checkout(
            planner_path, "preparation planner P", require_clean=False
        )
        git_x = inspect_git_checkout(
            Path(spec["worktree_x"]["path"]), "preparation worktree X"
        )
        git_y = inspect_git_checkout(
            Path(spec["control_y_input"]["path"]), "preparation control Y"
        )
        require(
            git_x["common"] == git_p["common"] == git_y["common"]
            and git_x["objects"] == git_p["objects"] == git_y["objects"],
            "preparation observation P/X/Y Git relationship mismatch",
        )
        require(
            len({git_p["path"], git_x["path"], git_y["path"]}) == 3,
            "preparation observation P/X/Y paths must be distinct",
        )

        def planner_output_set(checkout: dict[str, Any]) -> set[Path]:
            observed = validate_ignored_entries(
                checkout,
                "preparation planner P",
                allowed_paths={report_path},
            )
            for entry in checkout["status"].split(b"\0"):
                if not entry:
                    continue
                require(
                    entry.startswith(b"?? "), "planner P has a tracked status change"
                )
                try:
                    relative = entry[3:].decode("utf-8")
                except UnicodeDecodeError as error:
                    raise InvalidEvidence(
                        "planner P status path is not UTF-8"
                    ) from error
                observed.add(planner_path / relative)
            return observed

        initial_planner_outputs = planner_output_set(git_p)
        require(
            initial_planner_outputs == ({report_path} if report_preexisting else set()),
            "planner P has an unauthorized initial output",
        )

        def final_observation_custody(
            terminal_parent_stat: os.stat_result,
            expected_report: bytes | None = None,
            report_snapshot: os.stat_result | None = None,
            report_present: bool = True,
        ) -> str | None:
            context.final_check()
            final_custody_check(prep_file)
            final_custody_check(choices_file)
            require(
                planner_fd is not None
                and planner_stat is not None
                and x_fd is not None
                and y_fd is not None
                and x_stat is not None
                and y_stat is not None
                and x_path is not None
                and y_path is not None,
                "observation terminal custody state absent",
            )
            verify_directory_custody(
                x_fd, x_path, x_stat, "observation worktree X", metadata_stable=True
            )
            verify_directory_custody(
                y_fd, y_path, y_stat, "observation control Y", metadata_stable=True
            )
            require(
                inspect_git_checkout(x_path, "preparation worktree X") == git_x
                and inspect_git_checkout(y_path, "preparation control Y") == git_y,
                "observation X/Y Git identity drifted at terminal custody",
            )
            final_git_p = inspect_git_checkout(
                planner_path, "preparation planner P", require_clean=False
            )
            require(
                final_git_p["head"] == git_p["head"]
                and final_git_p["tree"] == git_p["tree"]
                and final_git_p["common"] == git_p["common"]
                and final_git_p["objects"] == git_p["objects"]
                and planner_output_set(final_git_p)
                == ({report_path} if report_present else set()),
                "planner P changed beyond the observation report",
            )
            require(
                not any(leaf_present(path) for path in future_paths),
                "observation terminal custody found a future output",
            )
            digest = None
            if expected_report is not None:
                require(
                    report_fd is not None and report_snapshot is not None,
                    "observation report FD custody state absent",
                )
                digest = verify_open_output_fd(
                    report_fd,
                    planner_fd,
                    report_path.name,
                    expected_report,
                    report_snapshot,
                    report_path,
                )
            if report_present:
                require(
                    report_snapshot is not None,
                    "observation report leaf snapshot is absent",
                )
                relative_info = os.stat(
                    report_path.name, dir_fd=planner_fd, follow_symlinks=False
                )
                absolute_info = os.stat(report_path, follow_symlinks=False)
                retained_info = (
                    os.fstat(report_fd) if report_fd is not None else relative_info
                )
                require(
                    all(
                        stat.S_ISREG(info.st_mode)
                        and info.st_nlink == 1
                        and (info.st_dev, info.st_ino)
                        == (report_snapshot.st_dev, report_snapshot.st_ino)
                        and info.st_size == report_snapshot.st_size
                        and info.st_mtime_ns == report_snapshot.st_mtime_ns
                        and info.st_ctime_ns == report_snapshot.st_ctime_ns
                        for info in (retained_info, relative_info, absolute_info)
                    ),
                    "observation report changed during final lightweight sweep",
                )
            # This stable P FD/path check is intentionally the final custody operation.
            verify_directory_custody(
                planner_fd,
                planner_path,
                terminal_parent_stat,
                "observation planner P",
                metadata_stable=True,
            )
            return digest

        if report_preexisting:
            require(planner_stat is not None, "planner P custody snapshot absent")
            final_observation_custody(
                planner_stat,
                report_snapshot=os.stat(report_path, follow_symlinks=False),
            )
            return {
                "result": "partial",
                "reason": "exclusive observation report path already exists",
                "reserved_empty_path": None,
                "retry": False,
            }
        report_object = {
            "schema": PREPARATION_HASH_REPORT_SCHEMA,
            "schema_version": 1,
            "authority": PREPARATION_HASH_REPORT_AUTHORITY,
            "run_id": spec["run_id"],
            "attempt_id": spec["attempt_id"],
            "inventory": preparation_claims["inventory"],
            "inventory_spec": preparation_claims["inventory_spec"],
            "preparation_choices": choices_claim,
            "preparation_spec": preparation_claims["preparation_spec"],
            "reducer": context.inventory["observed"]["reducer"],
            "rendered": {
                name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in zip(
                    ("fixture", "command", "manifest", "seal"), rendered
                )
            },
            "worktree_x": git_report_identity(git_x),
            "control_y": git_report_identity(git_y),
            "report": {"path": str(report_path), "max_bytes": report_max_bytes},
            "environment": observed_environment,
        }
        exact_keys(
            report_object, PREPARATION_HASH_REPORT_KEYS, "preparation hash report"
        )
        exact_keys(
            report_object["worktree_x"],
            GIT_REPORT_IDENTITY_KEYS,
            "preparation hash report worktree_x",
        )
        exact_keys(
            report_object["control_y"],
            GIT_REPORT_IDENTITY_KEYS,
            "preparation hash report control_y",
        )
        report_bytes = (
            json.dumps(report_object, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        require(
            len(report_bytes) <= report_max_bytes, "preparation hash report exceeds cap"
        )
        context.final_check()
        final_custody_check(prep_file)
        final_custody_check(choices_file)
        require(planner_stat is not None, "planner P custody snapshot absent")
        planner_before_write = os.fstat(planner_fd)
        require(
            (
                planner_before_write.st_dev,
                planner_before_write.st_ino,
                planner_before_write.st_mtime_ns,
                planner_before_write.st_ctime_ns,
            )
            == (
                planner_stat.st_dev,
                planner_stat.st_ino,
                planner_stat.st_mtime_ns,
                planner_stat.st_ctime_ns,
            ),
            "planner parent custody drifted before report reservation",
        )
        try:
            report_fd = os.open(
                report_path.name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=planner_fd,
            )
        except OSError as error:
            collision_parent = os.fstat(planner_fd)
            collision_present = leaf_present(report_path)
            final_observation_custody(
                collision_parent,
                report_snapshot=(
                    os.stat(report_path, follow_symlinks=False)
                    if collision_present
                    else None
                ),
                report_present=collision_present,
            )
            return {
                "result": "partial",
                "reason": f"exclusive observation report reservation failed: {error}",
                "reserved_empty_path": None,
                "retry": False,
            }
        try:
            os.fsync(planner_fd)
        except OSError as error:
            fsync_partial_evidence(report_fd, planner_fd, "observation report")
            planner_after_report = os.fstat(planner_fd)
            report_stat = os.fstat(report_fd)
            report_digest = final_observation_custody(
                planner_after_report, b"", report_stat
            )
            return {
                "result": "partial",
                "reason": f"observation report directory fsync failed: {error}",
                "reserved_path": str(report_path),
                "bytes": 0,
                "sha256": report_digest,
                "retry": False,
            }
        planner_after_report = os.fstat(planner_fd)

        def retained_report_bytes() -> tuple[bytes, os.stat_result]:
            require(report_fd is not None, "observation report FD is absent")
            info = os.fstat(report_fd)
            require(
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and 0 <= info.st_size <= report_max_bytes,
                "partial observation report exceeds authenticated bound",
            )
            data = bytearray()
            offset = 0
            while offset < info.st_size:
                block = os.pread(
                    report_fd, min(READ_CHUNK, info.st_size - offset), offset
                )
                require(block, "short read binding partial observation report")
                data.extend(block)
                offset += len(block)
            return bytes(data), os.fstat(report_fd)

        try:
            write_all_and_fsync(
                report_fd, report_bytes, "preparation observation report"
            )
            report_stat = os.fstat(report_fd)
            require(
                stat.S_ISREG(report_stat.st_mode)
                and report_stat.st_nlink == 1
                and report_stat.st_size == len(report_bytes),
                "preparation report FD shape mismatch",
            )
        except (OSError, InvalidEvidence) as error:
            fsync_partial_evidence(report_fd, planner_fd, "observation report")
            actual_report, report_stat = retained_report_bytes()
            report_digest = final_observation_custody(
                planner_after_report, actual_report, report_stat
            )
            return {
                "result": "partial",
                "reason": f"observation report write/fsync failed: {error}",
                "reserved_path": str(report_path),
                "bytes": len(actual_report),
                "sha256": report_digest,
                "retry": False,
            }
        require(
            (os.fstat(planner_fd).st_dev, os.fstat(planner_fd).st_ino)
            == (planner_stat.st_dev, planner_stat.st_ino),
            "planner directory custody changed during report write",
        )
        planner_path_after = os.stat(planner_path, follow_symlinks=False)
        require(
            (planner_path_after.st_dev, planner_path_after.st_ino)
            == (planner_stat.st_dev, planner_stat.st_ino),
            "planner path was replaced during report write",
        )
        try:
            report_digest = final_observation_custody(
                planner_after_report, report_bytes, report_stat
            )
        except (OSError, InvalidEvidence) as error:
            fsync_partial_evidence(report_fd, planner_fd, "observation report")
            actual_report, report_stat = retained_report_bytes()
            report_digest = final_observation_custody(
                planner_after_report, actual_report, report_stat
            )
            return {
                "result": "partial",
                "reason": f"observation report verification failed: {error}",
                "reserved_path": str(report_path),
                "bytes": len(actual_report),
                "sha256": report_digest,
                "retry": False,
            }
        require(report_digest is not None, "observation report digest is absent")
        return {
            "result": "observed",
            "path": str(report_path),
            "bytes": len(report_bytes),
            "sha256": report_digest,
            "rendered": report_object["rendered"],
        }
    except OSError as error:
        raise InvalidEvidence(
            f"exclusive preparation hash report creation failed: {error}"
        ) from error
    finally:
        if report_fd is not None:
            os.close(report_fd)
        if planner_fd is not None:
            os.close(planner_fd)
        if x_fd is not None:
            os.close(x_fd)
        if y_fd is not None:
            os.close(y_fd)
        prep_file.close()
        choices_file.close()
        context.close()


def prepare_artifacts(
    inventory_path: Path,
    inventory_sha256: str,
    inventory_spec_path: Path,
    inventory_spec_sha256: str,
    preparation_choices_path: Path,
    preparation_choices_sha256: str,
    preparation_spec_path: Path,
    preparation_spec_sha256: str,
    expected_hashes: dict[str, str],
    observed_environment: dict[str, str],
) -> dict[str, Any]:
    require(
        observed_environment == validate_preparation_environment(dict(os.environ)),
        "writing preparation environment was not authenticated from the current process",
    )
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
    exact_keys(
        expected_hashes,
        ("fixture", "command", "manifest", "seal"),
        "expected preparation hashes",
    )
    inventory = inventory_context.inventory
    prep_file = open_regular(
        preparation_spec_path, "preparation spec", MAX_BOOTSTRAP_SPEC_BYTES
    )
    choices_file = open_regular(
        preparation_choices_path, "preparation choices", MAX_BOOTSTRAP_SPEC_BYTES
    )
    reserved: list[tuple[Path, int]] = []
    output_snapshots: dict[Path, os.stat_result] = {}
    directory_fd: int | None = None
    try:
        require(
            prep_file.digest == preparation_spec_sha256,
            "preparation spec digest mismatch",
        )
        spec = parse_canonical_json_object(
            prep_file, PREPARATION_SPEC_KEYS, "preparation spec"
        )
        require(
            spec["schema"] == PREPARATION_SPEC_SCHEMA
            and spec["schema_version"] == PREPARATION_SPEC_VERSION,
            "preparation spec schema mismatch",
        )
        require(
            choices_file.digest == preparation_choices_sha256,
            "preparation choices digest mismatch",
        )
        choices = parse_canonical_json_object(
            choices_file, PREPARATION_CHOICES_KEYS, "preparation choices"
        )
        choices_claim = preparation_file_claim(choices_file, MAX_BOOTSTRAP_SPEC_BYTES)
        validate_preparation_join(spec, choices, choices_claim)
        retained_by_path = {item.path: item for item in inventory_context.opened}
        validate_preparation_input_binding(
            spec,
            retained_by_path[inventory_path.resolve(strict=True)],
            inventory_sha256,
            retained_by_path[inventory_spec_path.resolve(strict=True)],
            inventory_spec_sha256,
            choices_file,
            preparation_choices_sha256,
            prep_file,
            preparation_spec_sha256,
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
        preparation_claims = {}
        for key, path_value, digest, maximum in (
            ("inventory", inventory_path, inventory_sha256, MAX_TRACE_BYTES),
            (
                "inventory_spec",
                inventory_spec_path,
                inventory_spec_sha256,
                MAX_BOOTSTRAP_SPEC_BYTES,
            ),
        ):
            item = retained_by_path[path_value.resolve(strict=True)]
            preparation_claims[key] = {
                "path": str(item.path),
                "bytes": item.size,
                "sha256": digest,
                "max_bytes": maximum,
            }
        preparation_claims["preparation_spec"] = {
            "path": str(prep_file.path),
            "bytes": prep_file.size,
            "sha256": preparation_spec_sha256,
            "max_bytes": MAX_BOOTSTRAP_SPEC_BYTES,
        }
        rendered = render_preparation(
            inventory,
            inventory_sha256,
            spec,
            preparation_spec_sha256,
            preparation_claims,
            choices,
            choices_claim,
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
        reduction_path = canonical_output_path(Path(spec["reduction_output"]))
        require(
            not any(
                leaf_present(path) for path in all_output_paths[4:] + [reduction_path]
            ),
            "future trace/sidecar/reduction leaves must be absent before preparation",
        )
        input_paths = {
            *(item.path for item in inventory_context.opened),
            prep_file.path,
            choices_file.path,
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
            if leaf_present(path):
                info = os.lstat(path)
                inode_key = info.st_dev.to_bytes(8, "little") + info.st_ino.to_bytes(
                    8, "little"
                )
                require(
                    inode_key not in input_inodes,
                    "preparation existing output inode aliases an authenticated input",
                )
        # Reauthenticate all inventory-bound inputs immediately before reservation.
        inventory_context.final_check()
        final_custody_check(prep_file)
        final_custody_check(choices_file)
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
        require(
            parent == Path(spec["control_y_input"]["path"]),
            "prepared output parent differs from exact spec control Y path",
        )
        directory_fd, directory_stat = open_directory_custody(
            parent, "writing control Y"
        )
        require(
            stat.S_ISDIR(directory_stat.st_mode),
            "control-Y custody FD is not a directory",
        )
        verify_directory_custody(
            directory_fd,
            parent,
            directory_stat,
            "writing control Y",
            metadata_stable=True,
        )
        git_x_before, git_y_before = validate_preparation_git_state(
            inventory,
            spec,
            set(paths),
            require_all_outputs=False,
        )
        planner = exact_keys(spec["planner_p"], ("path",), "planner P")
        git_p_before = inspect_git_checkout(
            Path(planner["path"]), "preparation planner P"
        )

        def final_prepare_custody(
            expected_outputs: dict[Path, bytes],
            terminal_directory_stat: os.stat_result,
            collision_path: Path | None = None,
        ) -> dict[Path, str]:
            inventory_context.final_check()
            final_custody_check(prep_file)
            final_custody_check(choices_file)
            require(directory_fd is not None, "writing control-Y FD is absent")
            descriptors = {path: descriptor for path, descriptor in reserved}
            require(
                set(descriptors) == set(expected_outputs),
                "prepared output FD set differs from terminal output subset",
            )
            digests = {
                path: verify_open_output_fd(
                    descriptors[path],
                    directory_fd,
                    path.name,
                    data,
                    output_snapshots[path],
                    path,
                )
                for path, data in expected_outputs.items()
            }
            allowed_terminal = set(expected_outputs)
            if collision_path is not None:
                require(
                    leaf_present(collision_path),
                    "preparation collision leaf disappeared before terminal custody",
                )
                allowed_terminal.add(collision_path)
            git_x_final, git_y_final = validate_preparation_git_state(
                inventory,
                spec,
                allowed_terminal,
                require_all_outputs=True,
            )
            require(
                git_x_final == git_x_before
                and git_y_final["head"] == git_y_before["head"]
                and git_y_final["tree"] == git_y_before["tree"]
                and git_y_final["common"] == git_y_before["common"]
                and git_y_final["objects"] == git_y_before["objects"],
                "writing preparation X/Y Git identity drifted",
            )
            require(
                inspect_git_checkout(Path(planner["path"]), "preparation planner P")
                == git_p_before,
                "preparation planner P changed during terminal custody",
            )
            require(
                not any(
                    leaf_present(path)
                    for path in all_output_paths[4:] + [reduction_path]
                ),
                "preparation terminal custody found a future output",
            )
            inventory_context.final_check()
            final_custody_check(prep_file)
            final_custody_check(choices_file)
            for path, descriptor in reserved:
                current_fd = os.fstat(descriptor)
                relative = os.stat(
                    path.name, dir_fd=directory_fd, follow_symlinks=False
                )
                absolute = os.stat(path, follow_symlinks=False)
                snapshot = output_snapshots[path]
                require(
                    all(
                        stat.S_ISREG(info.st_mode)
                        and info.st_nlink == 1
                        and (info.st_dev, info.st_ino)
                        == (snapshot.st_dev, snapshot.st_ino)
                        and info.st_size == snapshot.st_size
                        and info.st_mtime_ns == snapshot.st_mtime_ns
                        and info.st_ctime_ns == snapshot.st_ctime_ns
                        for info in (current_fd, relative, absolute)
                    ),
                    "prepared output changed during final joint sweep",
                )
            # This stable Y FD/path check is intentionally the final custody operation.
            verify_directory_custody(
                directory_fd,
                parent,
                terminal_directory_stat,
                "writing control Y",
                metadata_stable=True,
            )
            return digests

        try:
            for path in paths:
                descriptor = os.open(
                    path.name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
                reserved.append((path, descriptor))
                output_snapshots[path] = os.fstat(descriptor)
                os.fsync(directory_fd)
        except OSError as error:
            for _, reserved_descriptor in reserved:
                fsync_partial_evidence(
                    reserved_descriptor, directory_fd, "prepared output"
                )
            terminal_directory_stat = os.fstat(directory_fd)
            final_prepare_custody(
                {reserved_path: b"" for reserved_path, _ in reserved},
                terminal_directory_stat,
                path if leaf_present(path) else None,
            )
            return {
                "result": "partial",
                "reason": f"exclusive preparation reservation failed: {error}",
                "reserved_empty_paths": [str(path) for path, _ in reserved],
                "retry": False,
            }
        terminal_directory_stat = os.fstat(directory_fd)

        def retained_output_bytes() -> dict[Path, bytes]:
            actual: dict[Path, bytes] = {}
            limits = {path: len(data) for path, data in zip(paths, rendered)}
            for path, descriptor in reserved:
                info = os.fstat(descriptor)
                require(
                    stat.S_ISREG(info.st_mode)
                    and info.st_nlink == 1
                    and 0 <= info.st_size <= limits[path],
                    "partial prepared output exceeds authenticated bound",
                )
                data = bytearray()
                offset = 0
                while offset < info.st_size:
                    block = os.pread(
                        descriptor, min(READ_CHUNK, info.st_size - offset), offset
                    )
                    require(block, "short read binding partial prepared output")
                    data.extend(block)
                    offset += len(block)
                output_snapshots[path] = os.fstat(descriptor)
                actual[path] = bytes(data)
            return actual

        try:
            for (path, descriptor), data in zip(reserved, rendered):
                write_all_and_fsync(descriptor, data, f"prepared output {path.name}")
                snapshot = os.fstat(descriptor)
                require(
                    stat.S_ISREG(snapshot.st_mode)
                    and snapshot.st_nlink == 1
                    and snapshot.st_size == len(data),
                    f"prepared output FD shape mismatch for {path.name}",
                )
                output_snapshots[path] = snapshot
        except (OSError, InvalidEvidence) as error:
            for _, reserved_descriptor in reserved:
                fsync_partial_evidence(
                    reserved_descriptor, directory_fd, "prepared output"
                )
            actual_outputs = retained_output_bytes()
            final_prepare_custody(actual_outputs, terminal_directory_stat)
            return {
                "result": "partial",
                "reason": f"prepared output write/fsync failed: {error}",
                "reserved_paths": [str(path) for path, _ in reserved],
                "bytes": {
                    str(path): len(data) for path, data in actual_outputs.items()
                },
                "retry": False,
            }
        after_stat = os.fstat(directory_fd)
        require(
            (after_stat.st_dev, after_stat.st_ino)
            == (directory_stat.st_dev, directory_stat.st_ino),
            "control-Y directory custody changed during writes",
        )
        try:
            output_digests = final_prepare_custody(
                {path: data for path, data in zip(paths, rendered)},
                terminal_directory_stat,
            )
        except (OSError, InvalidEvidence) as error:
            for _, reserved_descriptor in reserved:
                fsync_partial_evidence(
                    reserved_descriptor, directory_fd, "prepared output"
                )
            actual_outputs = retained_output_bytes()
            output_digests = final_prepare_custody(
                actual_outputs, terminal_directory_stat
            )
            return {
                "result": "partial",
                "reason": f"prepared output verification failed: {error}",
                "reserved_paths": [str(path) for path, _ in reserved],
                "bytes": {
                    str(path): len(data) for path, data in actual_outputs.items()
                },
                "sha256": {str(path): output_digests[path] for path in actual_outputs},
                "retry": False,
            }
        return {
            name: {
                "path": str(path),
                "bytes": len(data),
                "sha256": output_digests[path],
            }
            for name, path, data in zip(names, paths, rendered)
        }
    finally:
        for _, descriptor in reserved:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)
        prep_file.close()
        choices_file.close()
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
    preparation_choices_path: Path,
    preparation_choices_sha256: str,
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
    output_parent_fd: int | None = None
    output_parent_initial: os.stat_result | None = None
    output_fd: int | None = None
    try:
        validate_preparation_environment(dict(os.environ))
        sha256_text(manifest_sha256, "independent manifest SHA-256")
        output = canonical_output_path(output_path)
        output_parent_fd, output_parent_initial = open_directory_custody(
            output.parent, "reduction output parent"
        )
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
            "preparation_choices": (
                preparation_choices_path,
                preparation_choices_sha256,
            ),
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
            and seal_json["preparation_choices_sha256"] == preparation_choices_sha256
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
        choices_json = parse_canonical_json_object(
            bridge_files["preparation_choices"],
            PREPARATION_CHOICES_KEYS,
            "bound preparation choices",
        )
        prep_json = parse_canonical_json_object(
            bridge_files["preparation_spec"],
            PREPARATION_SPEC_KEYS,
            "bound preparation spec",
        )
        validate_preparation_join(
            prep_json,
            choices_json,
            manifest["preparation_binding"]["preparation_choices"],
        )
        inventory_expected = exact_keys(
            inventory_json["expected"],
            INVENTORY_EXPECTED_KEYS,
            "bound inventory expected",
        )
        inventory_observed = exact_keys(
            inventory_json["observed"],
            INVENTORY_OBSERVED_KEYS,
            "bound inventory observed",
        )
        exact_keys(
            inventory_observed["tokenizer"],
            TOKENIZER_OBSERVATION_KEYS,
            "bound inventory tokenizer",
        )
        exact_keys(
            inventory_observed["prompt"],
            PROMPT_OBSERVATION_KEYS,
            "bound inventory prompt",
        )
        observed_prompt = inventory_observed["prompt"]
        manifest_prompt = {
            "utf8_hex": observed_prompt["utf8_hex"],
            "token_ids": observed_prompt["token_ids"],
            "token_ids_sha256_i32le": observed_prompt["token_ids_sha256_i32le"],
            "tokenizer_identity_sha256": observed_prompt[
                "tokenizer_metadata_identity_sha256"
            ],
        }
        require(
            inventory_json["schema"] == INVENTORY_SCHEMA
            and inventory_json["schema_version"] == INVENTORY_VERSION == 2
            and inventory_json["authority"] == INVENTORY_AUTHORITY
            and inventory_json["run_id"] == run_id
            and inventory_json["inventory_spec_sha256"] == inventory_spec_sha256
            and inventory_observed["sources"] == manifest["sources"]
            and inventory_observed["assets"] == manifest["assets"]
            and inventory_observed["tensors"] == manifest["tensors"]
            and manifest_prompt == manifest["expected_prompt"]
            and inventory_observed["build"] == manifest["expected_build"]
            and inventory_observed["device"] == manifest["expected_host"]
            and inventory_observed["embedded_metallib"]["sha256"]
            == manifest["embedded_metallib_sha256"]
            and inventory_observed["reducer"] == manifest["reducer"]
            and inventory_observed["executable"] == manifest["executable"],
            "bound inventory semantic facts differ from manifest",
        )
        require(
            inventory_spec_json["schema"] == INVENTORY_SPEC_SCHEMA
            and inventory_spec_json["schema_version"] == INVENTORY_VERSION
            and inventory_spec_json["run_id"] == run_id
            and inventory_expected
            == {key: inventory_spec_json[key] for key in INVENTORY_EXPECTED_KEYS},
            "bound inventory-spec schema/run mismatch",
        )
        require(
            choices_json["schema"] == PREPARATION_CHOICES_SCHEMA
            and choices_json["schema_version"] == PREPARATION_CHOICES_VERSION
            and choices_json["run_id"] == run_id
            and choices_json["attempt_id"] == attempt_id
            and prep_json["schema_version"] == PREPARATION_SPEC_VERSION
            and prep_json["schema"] == PREPARATION_SPEC_SCHEMA
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
                preparation_choices_sha256,
                preparation_seal_sha256,
                manifest["reducer"]["path"],
            )
        trace = open_regular(trace_path, "trace", manifest["trace_max_bytes"])
        sidecar = open_regular(sidecar_path, "sidecar", manifest["sidecar_max_bytes"])
        reducer = open_regular(
            Path(__file__).resolve(), "reducer", manifest["reducer"]["max_bytes"]
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
    data = (
        json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    try:
        require(
            output is not None
            and output_parent_fd is not None
            and output_parent_initial is not None,
            "output path custody validation failed",
        )
        verify_directory_custody(
            output_parent_fd,
            output.parent,
            output_parent_initial,
            "reduction output parent",
            metadata_stable=True,
        )
        try:
            output_fd = os.open(
                output.name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=output_parent_fd,
            )
        except OSError as error:
            raise InvalidEvidence(f"exclusive-create output failed: {error}") from error
        try:
            os.fsync(output_parent_fd)
        except OSError as error:
            fsync_partial_evidence(output_fd, output_parent_fd, "reduction output")
            terminal_parent = os.fstat(output_parent_fd)
            terminal_parent_tuple = (
                terminal_parent.st_dev,
                terminal_parent.st_ino,
                terminal_parent.st_mtime_ns,
                terminal_parent.st_ctime_ns,
            )
            for item in opened:
                item_parent = os.fstat(item.parent_fd)
                shares_output_parent = item.parent_path == output.parent and (
                    item_parent.st_dev,
                    item_parent.st_ino,
                ) == (terminal_parent.st_dev, terminal_parent.st_ino)
                final_custody_check(
                    item, terminal_parent_tuple if shares_output_parent else None
                )
            output_snapshot = os.fstat(output_fd)
            output_digest = finalize_reduction_output(
                output_fd,
                output_parent_fd,
                output,
                b"",
                output_snapshot,
                terminal_parent,
            )
            return {
                "schema": REDUCTION_SCHEMA,
                "schema_version": REDUCTION_VERSION,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "result": "partial",
                "authority": AUTHORITY,
                "reason": f"reduction output directory fsync failed: {error}",
                "metrics": None,
                "custody": custody or None,
                "output_bytes": 0,
                "output_sha256": output_digest,
                "retry": False,
            }
        terminal_parent = os.fstat(output_parent_fd)

        def retained_reduction_bytes() -> tuple[bytes, os.stat_result]:
            require(output_fd is not None, "reduction output FD is absent")
            info = os.fstat(output_fd)
            require(
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and 0 <= info.st_size <= len(data),
                "partial reduction output exceeds authenticated bound",
            )
            actual = bytearray()
            offset = 0
            while offset < info.st_size:
                block = os.pread(
                    output_fd, min(READ_CHUNK, info.st_size - offset), offset
                )
                require(block, "short read binding partial reduction output")
                actual.extend(block)
                offset += len(block)
            return bytes(actual), os.fstat(output_fd)

        def finalize_inputs_then_output(
            expected: bytes, output_snapshot: os.stat_result
        ) -> str:
            terminal_parent_tuple = (
                terminal_parent.st_dev,
                terminal_parent.st_ino,
                terminal_parent.st_mtime_ns,
                terminal_parent.st_ctime_ns,
            )
            for item in opened:
                item_parent = os.fstat(item.parent_fd)
                shares_output_parent = item.parent_path == output.parent and (
                    item_parent.st_dev,
                    item_parent.st_ino,
                ) == (terminal_parent.st_dev, terminal_parent.st_ino)
                final_custody_check(
                    item,
                    terminal_parent_tuple if shares_output_parent else None,
                )
            # Output verification is deliberately last after every input check.
            return finalize_reduction_output(
                output_fd,
                output_parent_fd,
                output,
                expected,
                output_snapshot,
                terminal_parent,
            )

        try:
            write_all_and_fsync(output_fd, data, "reduction output")
            output_snapshot = os.fstat(output_fd)
            require(
                stat.S_ISREG(output_snapshot.st_mode)
                and output_snapshot.st_nlink == 1
                and output_snapshot.st_size == len(data),
                "reduction output FD shape mismatch",
            )
            finalize_inputs_then_output(data, output_snapshot)
        except (OSError, InvalidEvidence) as error:
            fsync_partial_evidence(output_fd, output_parent_fd, "reduction output")
            actual, output_snapshot = retained_reduction_bytes()
            finalize_inputs_then_output(actual, output_snapshot)
            return {
                "schema": REDUCTION_SCHEMA,
                "schema_version": REDUCTION_VERSION,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "result": "partial",
                "authority": AUTHORITY,
                "reason": f"reduction output write/fsync/verification failed: {error}",
                "metrics": None,
                "custody": custody or None,
                "output_bytes": len(actual),
                "output_sha256": hashlib.sha256(actual).hexdigest(),
                "retry": False,
            }
        return result
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if output_parent_fd is not None:
            os.close(output_parent_fd)
        for item in opened:
            item.close()


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
        elif dtype == 7:
            header += bytes((value,))
        else:
            raise AssertionError("self-test metadata helper only supports u32/bool")
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
    root = root.resolve(strict=True)
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
        "preparation_choices": root / "bound-preparation-choices.json",
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
    build_report_path = root / "bound-build-report.json"
    build_report_path.write_text("{}", encoding="utf-8")
    build_report_claim = synthetic_file_claim(build_report_path)
    tokenizer_predicate = {
        "vocab_size": VOCAB,
        "token_embd_name": "token_embd.weight",
        "token_embd_rank": 2,
        "token_embd_hidden": HIDDEN,
        "token_embd_vocab_axis": 1,
        "allowed_token_embd_dtypes": ["Q4_K"],
        "require_token_metadata": True,
        "metadata_identity_domain": "qwen.dflash_k0s.tokenizer_metadata.v1",
    }
    prompt_predicate = {
        "utf8_hex": b"Write code".hex(),
        "utf8_sha256": hashlib.sha256(b"Write code").hexdigest(),
        "add_special": False,
        "expected_token_ids": [7734, 1970],
        "expected_token_ids_sha256_i32le": hashlib.sha256(
            struct.pack("<ii", 7734, 1970)
        ).hexdigest(),
    }
    mask_predicate = {
        "allowed_metadata_keys": [
            "dflash-draft.dflash.mask_token_id",
            "tokenizer.ggml.mask_token_id",
        ],
        "expected_mask_token": 248070,
    }
    asset_expectations = [
        {
            "role": claim["role"],
            "path": claim["path"],
            "expected_bytes": claim["bytes"] if claim["role"] == "target" else None,
            "max_bytes": claim["max_bytes"],
            "sha256": claim["sha256"],
        }
        for claim in asset_claims
    ]
    inventory_spec_object = {
        "schema": INVENTORY_SPEC_SCHEMA,
        "schema_version": INVENTORY_VERSION,
        "run_id": run_id,
        "inventory_max_bytes": MAX_TRACE_BYTES,
        "checkout": {
            "path": str(root.resolve()),
            "commit": build["commit"],
            "tree": "1" * 40,
            "dirty": False,
        },
        "build": build,
        "build_report": build_report_claim,
        "sources": source_claims,
        "executable": executable_claim,
        "reducer": reducer_claim,
        "embedded_metallib": synthetic_file_claim(metallib_path),
        "assets": asset_expectations,
        "tensor_requirements": [
            {key: tensor[key] for key in TENSOR_REQUIREMENT_KEYS}
            for tensor in tensor_claims
        ],
        "tokenizer_predicate": tokenizer_predicate,
        "prompt_predicate": prompt_predicate,
        "mask_predicate": mask_predicate,
        "parser_caps": {
            "header_bytes": MAX_GGUF_HEADER_BYTES,
            "metadata": MAX_GGUF_METADATA,
            "tensors": MAX_GGUF_TENSORS,
            "strings_bytes": MAX_GGUF_STRINGS_BYTES,
            "array_items_per_array": MAX_GGUF_ARRAY_ITEMS_PER_ARRAY,
            "array_items": MAX_GGUF_ARRAY_ITEMS,
            "objects": MAX_GGUF_OBJECTS,
        },
        "host_predicate": {
            "os": "macos",
            "arch": "aarch64",
            "device_name": "Apple M4 Max",
            "required_families": ["apple9", "mac2", "common3", "metal3"],
            "family_match": "all",
        },
        "command": ["synthetic-inventory"],
        "environment": {"QWEN_METAL_LEASE_WAIT": "1"},
    }
    preparation_input_paths["inventory_spec"].write_text(
        json.dumps(inventory_spec_object, separators=(",", ":")), encoding="utf-8"
    )
    inventory_spec_digest = hashlib.sha256(
        preparation_input_paths["inventory_spec"].read_bytes()
    ).hexdigest()
    inventory_expected = {
        key: copy.deepcopy(inventory_spec_object[key])
        for key in INVENTORY_EXPECTED_KEYS
    }
    inventory_observed = {
        "checkout": inventory_spec_object["checkout"],
        "build": build,
        "build_report": build_report_claim,
        "sources": source_claims,
        "executable": executable_claim,
        "reducer": reducer_claim,
        "embedded_metallib": synthetic_file_claim(metallib_path),
        "device": expected_host,
        "assets": asset_claims,
        "gguf": [],
        "tensors": tensor_claims,
        "tokenizer": {
            "vocab_size": VOCAB,
            "token_embd_name": "token_embd.weight",
            "token_embd_shape": [1, VOCAB],
            "token_embd_dtype": "F32",
            "token_count": VOCAB,
            "model": "synthetic",
            "pre": "synthetic",
            "bos_token_id": None,
            "eos_token_id": None,
            "add_bos_token": None,
            "add_eos_token": None,
            "token_list_sha256": prompt_claim["tokenizer_identity_sha256"],
            "token_type_sha256": prompt_claim["tokenizer_identity_sha256"],
            "merges_sha256": prompt_claim["tokenizer_identity_sha256"],
            "metadata_identity_sha256": prompt_claim["tokenizer_identity_sha256"],
        },
        "prompt": {
            "utf8_hex": prompt_claim["utf8_hex"],
            "add_special": False,
            "token_ids": prompt_claim["token_ids"],
            "token_ids_sha256_i32le": prompt_claim["token_ids_sha256_i32le"],
            "tokenizer_metadata_identity_sha256": prompt_claim[
                "tokenizer_identity_sha256"
            ],
        },
        "mask_noise": {
            "metadata_key": "tokenizer.ggml.mask_token_id",
            "mask_token": 248070,
            "noise_tokens": [0] + [248070] * 7,
            "noise_sha256_i32le": hashlib.sha256(
                b"".join(struct.pack("<i", v) for v in [0] + [248070] * 7)
            ).hexdigest(),
        },
        "parser_caps": inventory_spec_object["parser_caps"],
    }
    inventory_object = {
        "schema": INVENTORY_SCHEMA,
        "schema_version": INVENTORY_VERSION,
        "authority": INVENTORY_AUTHORITY,
        "inventory_spec_sha256": inventory_spec_digest,
        "run_id": run_id,
        "expected": inventory_expected,
        "observed": inventory_observed,
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
        "schema_version": PREPARATION_SPEC_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "preparation_choices": None,
        "inventory_path": str(preparation_input_paths["inventory"].resolve()),
        "inventory_sha256": inventory_digest,
        "inventory_spec_path": str(preparation_input_paths["inventory_spec"].resolve()),
        "inventory_spec_sha256": inventory_spec_digest,
        "preparation_spec_path": str(
            preparation_input_paths["preparation_spec"].resolve()
        ),
        "worktree_x": {"path": str(root.resolve()), "commit": build["commit"]},
        "planner_p": {"path": str(root.resolve())},
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
        "reduction_output": str((root / "reduction.json").resolve()),
        "continuation_carry_token": 1,
        "manifest_choices": {
            "trace_max_bytes": MAX_TRACE_BYTES,
            "sidecar_max_bytes": MAX_SIDECAR_BYTES,
            "semantic_references": copy.deepcopy(SEMANTIC_REFERENCES),
            "expected_request": {
                "request": request,
                "ignored_target_policy": ignored_policy,
            },
            "expected_binding": binding,
            "expected_rng_domains": ["request_rng"],
            "expected_fixed_chains": static_fixed_chains,
            "expected_capture_context": capture_context,
            "selector_dispatch_predicate": selector_dispatch_predicate,
        },
        "transformation_sha256": reducer_claim["sha256"],
        "environment_allowlist": {"QWEN_METAL_LEASE_WAIT": "1"},
        "arm_order": ["off-A", "on-A", "on-B", "off-B"],
        "selected_arm": "on-A",
        "parity_comparison_fields": PARITY_COMPARISON_FIELDS,
        "reducer_argv": [],
        "failure_policy": {"on_collision": "retain_reserved_partial", "retry": False},
    }
    choices_object = {
        "schema": PREPARATION_CHOICES_SCHEMA,
        "schema_version": PREPARATION_CHOICES_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "worktree_x": synthetic_prep["worktree_x"],
        "planner_p": synthetic_prep["planner_p"],
        "control_y": synthetic_prep["control_y_input"],
        "outputs": synthetic_prep["outputs"],
        "preparation_spec_path": synthetic_prep["preparation_spec_path"],
        "fixture_content": synthetic_prep["fixture_content"],
        "acquisition_outputs": synthetic_prep["acquisition_outputs"],
        "reduction_output": synthetic_prep["reduction_output"],
        "continuation_carry_token": synthetic_prep["continuation_carry_token"],
        "manifest_choices": synthetic_prep["manifest_choices"],
        "transformation_sha256": synthetic_prep["transformation_sha256"],
        "environment_allowlist": synthetic_prep["environment_allowlist"],
        "arm_order": synthetic_prep["arm_order"],
        "selected_arm": synthetic_prep["selected_arm"],
        "parity_comparison_fields": synthetic_prep["parity_comparison_fields"],
        "reducer_argv_template": [],
        "failure_policy": synthetic_prep["failure_policy"],
    }
    placeholder_spec = copy.deepcopy(synthetic_prep)
    placeholder_spec["preparation_choices"] = {
        "path": str(preparation_input_paths["preparation_choices"].resolve()),
        "bytes": 1,
        "sha256": "0" * 64,
        "max_bytes": MAX_BOOTSTRAP_SPEC_BYTES,
    }
    choices_object["reducer_argv_template"] = [
        INVENTORY_PATH_PLACEHOLDER
        if value == synthetic_prep["inventory_path"]
        else INVENTORY_SHA256_PLACEHOLDER
        if value == synthetic_prep["inventory_sha256"]
        else INVENTORY_SPEC_PATH_PLACEHOLDER
        if value == synthetic_prep["inventory_spec_path"]
        else INVENTORY_SPEC_SHA256_PLACEHOLDER
        if value == synthetic_prep["inventory_spec_sha256"]
        else value
        for value in frozen_reducer_argv(placeholder_spec, reducer_claim["path"])
    ]
    preparation_input_paths["preparation_choices"].write_bytes(
        canonical_json_bytes(choices_object)
    )
    choices_claim = synthetic_file_claim(
        preparation_input_paths["preparation_choices"], MAX_BOOTSTRAP_SPEC_BYTES
    )
    synthetic_prep["preparation_choices"] = choices_claim
    synthetic_prep["reducer_argv"] = frozen_reducer_argv(
        synthetic_prep, reducer_claim["path"]
    )
    preparation_input_paths["preparation_spec"].write_bytes(
        canonical_json_bytes(synthetic_prep)
    )
    preparation_seal_path = root / "bound-preparation-seal.json"
    preparation_binding = {
        "inventory": synthetic_file_claim(
            preparation_input_paths["inventory"], MAX_TRACE_BYTES
        ),
        "inventory_spec": synthetic_file_claim(
            preparation_input_paths["inventory_spec"], MAX_TRACE_BYTES
        ),
        "preparation_choices": choices_claim,
        "preparation_spec": synthetic_file_claim(
            preparation_input_paths["preparation_spec"], MAX_TRACE_BYTES
        ),
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
        "preparation_choices_sha256": choices_claim["sha256"],
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
        "preparation_choices": preparation_input_paths["preparation_choices"],
        "preparation_choices_sha256": choices_claim["sha256"],
        "preparation_spec": preparation_input_paths["preparation_spec"],
        "preparation_spec_sha256": preparation_spec_digest,
        "preparation_seal": preparation_seal_path,
        "preparation_seal_sha256": preparation_seal_digest,
    }


def self_test() -> None:
    tests = 0
    strict_expect_error = globals()["expect_error"]

    def ok(condition: bool) -> None:
        nonlocal tests
        assert condition
        tests += 1

    def expect_error(function: Any, contains: str = "") -> None:
        nonlocal tests
        strict_expect_error(function, contains)
        tests += 1

    reducer_script = Path(__file__).resolve()
    ok(
        os.access(reducer_script, os.X_OK)
        and reducer_script.read_bytes().startswith(b"#!/usr/bin/env -S uv run\n")
    )

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
    ok(
        RUST_BUILD_COMMAND_SUFFIX
        == [
            "build",
            "--locked",
            "--offline",
            "--release",
            "-p",
            "qwen-cli",
            "--bin",
            "qwen-bench",
            "--features",
            "dflash-k0s-diagnostics",
        ]
    )
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
        "4" * 64,
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
            "4" * 64,
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
            "4" * 64,
            "3" * 64,
            str(Path(__file__).resolve()),
        ),
        "literal order/spelling/value",
    )
    reducer_alias_argv = copy.deepcopy(literal_actual)
    reducer_alias_argv[0] = str(
        Path(__file__).resolve().parent / ".." / "profile" / Path(__file__).name
    )
    expect_error(
        lambda: validate_literal_reducer_argv(
            reducer_alias_argv,
            literal_frozen,
            "1" * 64,
            "2" * 64,
            "4" * 64,
            "3" * 64,
            str(Path(__file__).resolve()),
        ),
        "literal spelling",
    )
    reducer_symlink_argv = copy.deepcopy(literal_actual)
    reducer_symlink_argv[0] = str(Path(__file__).resolve().parent / "reducer-symlink")
    expect_error(
        lambda: validate_literal_reducer_argv(
            reducer_symlink_argv,
            literal_frozen,
            "1" * 64,
            "2" * 64,
            "4" * 64,
            "3" * 64,
            str(Path(__file__).resolve()),
        ),
        "literal spelling",
    )
    mode_expected = [
        str(Path(__file__).resolve()),
        "--prepare",
        "--inventory",
        "/absolute/inventory.json",
        "--inventory-sha256",
        "1" * 64,
    ]
    validate_literal_mode_argv(
        copy.deepcopy(mode_expected), mode_expected, str(Path(__file__).resolve())
    )
    ok(True)
    expect_error(
        lambda: validate_literal_mode_argv(
            [
                mode_expected[0],
                "--prepare",
                "--inventory=/absolute/inventory.json",
                *mode_expected[4:],
            ],
            mode_expected,
            str(Path(__file__).resolve()),
        ),
        "length mismatch",
    )
    expect_error(
        lambda: validate_literal_mode_argv(
            [*mode_expected, "--inventory-sha256", "1" * 64],
            mode_expected,
            str(Path(__file__).resolve()),
        ),
        "length mismatch",
    )
    reordered_mode = [
        mode_expected[0],
        "--inventory",
        mode_expected[3],
        "--prepare",
        *mode_expected[4:],
    ]
    expect_error(
        lambda: validate_literal_mode_argv(
            reordered_mode, mode_expected, str(Path(__file__).resolve())
        ),
        "order/spelling/value",
    )
    symlink_spelling = copy.deepcopy(mode_expected)
    symlink_spelling[0] = str(Path(__file__).resolve().parent / "reducer-symlink")
    expect_error(
        lambda: validate_literal_mode_argv(
            symlink_spelling, mode_expected, str(Path(__file__).resolve())
        ),
        "literal spelling",
    )
    ok(
        validate_preparation_environment(
            {
                "PATH": "/usr/bin:/bin",
                "HOME": "/tmp",
                "QWEN_METAL_LEASE_WAIT": "1",
            }
        )
        == {"QWEN_METAL_LEASE_WAIT": "1"}
    )
    ok(
        validate_preparation_environment(
            {
                "QWEN_METAL_LEASE_WAIT": "1",
                "UV_RUN_RECURSION_DEPTH": "1",
            }
        )
        == {"QWEN_METAL_LEASE_WAIT": "1"}
    )
    for forbidden_environment_name in (
        "QWEN_OTHER",
        "MTL_DEBUG_LAYER",
        "METAL_DEVICE_WRAPPER_TYPE",
        "GGML_METAL_LOG_LEVEL",
        "DYLD_INSERT_LIBRARIES",
    ):
        expect_error(
            lambda name=forbidden_environment_name: validate_preparation_environment(
                {"QWEN_METAL_LEASE_WAIT": "1", name: "x"}
            ),
            "behavior allowlist",
        )
    expect_error(lambda: validate_preparation_environment({}), "behavior allowlist")
    expect_error(
        lambda: validate_preparation_environment({"QWEN_METAL_LEASE_WAIT": "0"}),
        "behavior allowlist",
    )
    for injected_environment in (
        {"QWEN_METAL_LEASE_WAIT": "1", "PYTHONPATH": "/tmp/inject"},
        {"QWEN_METAL_LEASE_WAIT": "1", "PYTHONHOME": "/tmp/inject"},
        {"QWEN_METAL_LEASE_WAIT": "1", "UV_INDEX_URL": "https://invalid"},
        {"QWEN_METAL_LEASE_WAIT": "1", "UV_RUN_RECURSION_DEPTH": "2"},
    ):
        expect_error(
            lambda value=injected_environment: validate_preparation_environment(value),
            "environment",
        )
    bounded_test_environment = {"LC_ALL": "C"}
    ok(
        bounded_subprocess(
            [sys.executable, "-c", "import os;os.write(1,b'ok')"],
            bounded_test_environment,
            stdout_cap=16,
            stderr_cap=16,
            timeout=2,
            name="synthetic bounded process",
        )
        == (0, b"ok", b"")
    )
    expect_error(
        lambda: bounded_subprocess(
            [sys.executable, "-c", "import os;os.write(1,b'x'*65)"],
            bounded_test_environment,
            stdout_cap=64,
            stderr_cap=64,
            timeout=2,
            name="synthetic stdout cap",
        ),
        "stdout exceeded cap",
    )
    expect_error(
        lambda: bounded_subprocess(
            [sys.executable, "-c", "import os;os.write(2,b'x'*1024)"],
            bounded_test_environment,
            stdout_cap=64,
            stderr_cap=64,
            timeout=2,
            name="synthetic stderr cap",
        ),
        "stderr exceeded cap",
    )
    original_popen = subprocess.Popen
    spawned_processes: list[subprocess.Popen[bytes]] = []

    def tracking_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = original_popen(*args, **kwargs)
        spawned_processes.append(process)
        return process

    subprocess.Popen = tracking_popen
    try:
        expect_error(
            lambda: bounded_subprocess(
                [sys.executable, "-c", "import time;time.sleep(5)"],
                bounded_test_environment,
                stdout_cap=64,
                stderr_cap=64,
                timeout=0.05,
                name="synthetic timeout",
            ),
            "timed out",
        )
    finally:
        subprocess.Popen = original_popen
    ok(len(spawned_processes) == 1 and spawned_processes[0].poll() is not None)
    dotdot_spelling = copy.deepcopy(mode_expected)
    dotdot_spelling[0] = str(
        Path(__file__).resolve().parent / ".." / "profile" / Path(__file__).name
    )
    expect_error(
        lambda: validate_literal_mode_argv(
            dotdot_spelling, mode_expected, str(Path(__file__).resolve())
        ),
        "literal spelling",
    )
    expect_error(
        lambda: validate_literal_reducer_argv(
            [*literal_actual, "--input", "/absolute/trace"],
            literal_frozen,
            "1" * 64,
            "2" * 64,
            "4" * 64,
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
            "4" * 64,
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
    large_registry_id = parse_json(
        b'{"device_registry_id":9223372036854775808}', "test"
    )
    ok(
        integer(
            large_registry_id["device_registry_id"],
            "device registry id",
            0,
            MAX_JSON_U64,
        )
        == 9223372036854775808
    )
    expect_error(
        lambda: integer(large_registry_id["device_registry_id"], "ordinary integer"),
        "out of range",
    )
    expect_error(
        lambda: parse_json(b'{"device_registry_id":18446744073709551616}', "test"),
        "grammar bound",
    )
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
        root = Path(directory).resolve(strict=True)
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
        canonical_bool = root / "canonical-bool.gguf"
        canonical_bool.write_bytes(minimal_gguf([], [("flag", 7, 1)]))
        opened = open_regular(canonical_bool, "canonical bool GGUF")
        ok(GGUF(opened).metadata["flag"] is True)
        opened.close()
        noncanonical_bool = root / "noncanonical-bool.gguf"
        noncanonical_bool.write_bytes(minimal_gguf([], [("flag", 7, 2)]))
        opened = open_regular(noncanonical_bool, "noncanonical bool GGUF")
        expect_error(lambda: GGUF(opened), "canonically encoded")
        opened.close()
        unknown_array = root / "unknown-array.bin"
        unknown_array.write_bytes(struct.pack("<IQII", 4, 2, 1, 2))
        opened = open_regular(unknown_array, "unknown metadata array")
        cursor = Cursor(opened)
        ok(gguf_value(cursor, 9) == [1, 2] and cursor.objects == 3)
        opened.close()
        oversized_array = root / "oversized-array.bin"
        oversized_array.write_bytes(
            struct.pack("<IQ", 4, MAX_GGUF_ARRAY_ITEMS_PER_ARRAY + 1)
        )
        opened = open_regular(oversized_array, "oversized metadata array")
        expect_error(lambda: gguf_value(Cursor(opened), 9), "per-array cap")
        opened.close()
        cumulative_array = root / "cumulative-array.bin"
        cumulative_array.write_bytes(struct.pack("<IQI", 4, 1, 0))
        opened = open_regular(cumulative_array, "cumulative metadata array")
        cumulative_cursor = Cursor(opened)
        cumulative_cursor.array_items = MAX_GGUF_ARRAY_ITEMS
        expect_error(
            lambda: gguf_value(cumulative_cursor, 9), "cumulative metadata array"
        )
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
        expect_error(lambda: open_regular(identity_path, "identity"), "hard link")
        expect_error(lambda: open_regular(hardlink, "identity hardlink"), "hard link")
        hardlink.unlink()
        symlink = root / "identity-symlink"
        symlink.symlink_to(identity_path)
        expect_error(
            lambda: open_regular(symlink, "identity symlink"),
            "lexically canonical",
        )
        symlink.unlink()
        broken_leaf = root / "broken-output-leaf"
        broken_leaf.symlink_to(root / "absent-target")
        ok(leaf_present(broken_leaf))
        broken_leaf.unlink()
        ok(not leaf_present(broken_leaf))
        timestamp_path = root / "timestamp-custody"
        timestamp_path.write_bytes(b"fixed")
        timestamp_opened = open_regular(timestamp_path, "timestamp custody")
        os.utime(
            timestamp_path,
            ns=(timestamp_opened.mtime_ns, timestamp_opened.mtime_ns + 1),
        )
        expect_error(lambda: final_custody_check(timestamp_opened), "custody changed")
        timestamp_opened.close()
        parent_path = root / "parent-custody"
        parent_path.mkdir()
        parent_leaf = parent_path / "leaf"
        parent_leaf.write_bytes(b"fixed")
        parent_opened = open_regular(parent_leaf, "parent custody")
        moved_parent = root / "parent-custody-moved"
        parent_path.rename(moved_parent)
        parent_path.mkdir()
        expect_error(lambda: final_custody_check(parent_opened), "parent")
        parent_opened.close()
        transient_parent = root / "transient-parent-custody"
        transient_parent.mkdir()
        transient_leaf = transient_parent / "leaf"
        transient_leaf.write_bytes(b"fixed")
        transient_opened = open_regular(transient_leaf, "transient parent custody")
        ignored_sibling = transient_parent / "ignored-sibling"
        ignored_sibling.write_bytes(b"ignored")
        ignored_sibling.unlink()
        expect_error(
            lambda: final_custody_check(transient_opened), "parent directory custody"
        )
        transient_opened.close()
        measured_root = root / "measured-build-root"
        measured_root.mkdir()
        (measured_root / "a").write_bytes(b"abc")
        (measured_root / "nested").mkdir()
        (measured_root / "nested" / "b").write_bytes(b"de")
        ok(measure_directory_bytes(measured_root, 5) == 5)
        ok(measure_directory(measured_root, 5)["entry_count"] == 3)
        ok(
            measure_directory(measured_root, MAX_BUILD_ROOT_BYTES)
            == measure_directory(measured_root, MAX_BUILD_ROOT_BYTES)
        )
        expect_error(lambda: measure_directory_bytes(measured_root, 4), "exceed cap")
        (measured_root / "bad-link").symlink_to(measured_root / "a")
        expect_error(lambda: measure_directory_bytes(measured_root, 100), "symlink")
        ignored_checkout = {
            "path": root,
            "ignored": b"measured-build-root/a\0",
        }
        ok(
            validate_ignored_entries(
                ignored_checkout,
                "synthetic X",
                allowed_root=measured_root,
            )
            == {measured_root / "a"}
        )
        ignored_checkout["ignored"] = b".DS_Store\0"
        expect_error(
            lambda: validate_ignored_entries(
                ignored_checkout,
                "synthetic X",
                allowed_root=measured_root,
            ),
            "unauthorized ignored entry",
        )
        report_x = root / "build-report-x"
        report_x.mkdir()
        report_root = report_x / "target"
        (report_root / "release").mkdir(parents=True)
        report_executable_path = report_root / "release" / "qwen-bench"
        report_executable_path.write_bytes(b"qwen-bench")
        report_metallib_path = report_x / "embedded.metallib"
        report_metallib_path.write_bytes(b"metallib")
        report_reducer_path = report_x / "reducer.py"
        report_reducer_path.write_bytes(b"reducer")
        report_source_path = report_x / "source.rs"
        report_source_path.write_bytes(b"source")
        report_compiler_path = report_x / "cargo"
        report_compiler_path.write_bytes(b"cargo")
        report_compiler_path.chmod(0o700)
        report_info_path = report_x / "build-info.json"
        report_info_path.write_bytes(b"{}")
        report_commit = "c" * 40
        report_source_digest = "d" * 64
        report_checkout = {
            "path": str(report_x),
            "commit": report_commit,
            "tree": "e" * 40,
            "dirty": False,
        }
        report_build = {
            "commit": report_commit,
            "source_sha256": report_source_digest,
            "dirty": False,
            "compiler": str(report_compiler_path),
            "compiler_version": "cargo synthetic",
            "target": "aarch64-apple-darwin",
            "profile": "release",
            "features": ["dflash-k0s-diagnostics"],
        }
        report_executable = synthetic_file_claim(report_executable_path)
        report_metallib = synthetic_file_claim(report_metallib_path)
        report_reducer = synthetic_file_claim(report_reducer_path)
        report_sources = [
            {"role": "synthetic", **synthetic_file_claim(report_source_path)}
        ]
        report_compiler = synthetic_file_claim(report_compiler_path)
        report_info_claim = synthetic_file_claim(report_info_path)
        report_build_info = {
            "artifact": report_info_claim,
            "schema_version": 2,
            "build_commit": report_commit,
            "build_commit_short": report_commit[:9],
            "build_dirty": False,
            "build_source_state": f"git-source-sha256-v2:{report_source_digest}",
            "stamp_source": "git",
            "stamp_error": None,
            "runtime_commit": report_commit,
            "runtime_dirty": False,
            "runtime_source_state": f"git-source-sha256-v2:{report_source_digest}",
            "status": "match",
            "problems": [],
            "overrides": [],
        }
        measured_report_root = measure_directory(report_root, MAX_BUILD_ROOT_BYTES)[
            "bytes"
        ]
        build_report = {
            "schema": BUILD_REPORT_SCHEMA,
            "schema_version": 1,
            "authority": BUILD_REPORT_AUTHORITY,
            "run_id": "synthetic-build-report",
            "attempt_id": "synthetic-build-attempt",
            "checkout": report_checkout,
            "build_command": [
                str(report_compiler_path),
                *RUST_BUILD_COMMAND_SUFFIX,
            ],
            "build_root": {
                "path": str(report_root),
                "bytes": measured_report_root,
                "max_bytes": MAX_BUILD_ROOT_BYTES,
            },
            "executable": report_executable,
            "embedded_metallib": report_metallib,
            "reducer": report_reducer,
            "sources": report_sources,
            "compiler": {
                "path": str(report_compiler_path),
                "bytes": report_compiler["bytes"],
                "sha256": report_compiler["sha256"],
                "version_verbose": "cargo synthetic",
                "version_verbose_sha256": hashlib.sha256(
                    b"cargo synthetic"
                ).hexdigest(),
            },
            "target": "aarch64-apple-darwin",
            "profile": "release",
            "features": ["dflash-k0s-diagnostics"],
            "build_info": report_build_info,
            "environment": {
                "CARGO_TARGET_DIR": str(report_root),
                "QWEN_METAL_LEASE_WAIT": "1",
            },
        }
        synthetic_root_custody: dict[str, Any] = {}
        validate_build_identity_report(
            build_report,
            report_info_claim,
            "synthetic-build-report",
            report_checkout,
            report_build,
            report_sources,
            report_executable,
            report_metallib,
            report_reducer,
            synthetic_root_custody,
        )
        ok(
            synthetic_root_custody["measurement"]
            == measure_directory(
                synthetic_root_custody["path"], synthetic_root_custody["maximum"]
            )
        )
        tight_cap_report = copy.deepcopy(build_report)
        tight_cap_report["build_root"]["max_bytes"] = measured_report_root
        validate_build_identity_report(
            tight_cap_report,
            report_info_claim,
            "synthetic-build-report",
            report_checkout,
            report_build,
            report_sources,
            report_executable,
            report_metallib,
            report_reducer,
        )
        ok(True)
        for field, replacement, reason in (
            ("build_command", [str(report_compiler_path), "build"], "cargo argv"),
            ("environment", {"QWEN_METAL_LEASE_WAIT": "1"}, "environment map"),
            (
                "executable",
                {**report_executable, "path": str(report_x / "wrong-qwen-bench")},
                "executable",
            ),
            (
                "build_root",
                {**build_report["build_root"], "max_bytes": measured_report_root - 1},
                "exceed cap",
            ),
            (
                "compiler",
                {**build_report["compiler"], "bytes": MAX_COMPILER_BYTES + 1},
                "compiler bytes",
            ),
        ):
            forged_report = copy.deepcopy(build_report)
            forged_report[field] = replacement
            expect_error(
                lambda value=forged_report: validate_build_identity_report(
                    value,
                    report_info_claim,
                    "synthetic-build-report",
                    report_checkout,
                    report_build,
                    report_sources,
                    report_executable,
                    report_metallib,
                    report_reducer,
                ),
                reason,
            )
        for bad_attempt in ("", "a" * 129, "nul\0attempt", "unicode-é"):
            forged_report = copy.deepcopy(build_report)
            forged_report["attempt_id"] = bad_attempt
            expect_error(
                lambda value=forged_report: validate_build_identity_report(
                    value,
                    report_info_claim,
                    "synthetic-build-report",
                    report_checkout,
                    report_build,
                    report_sources,
                    report_executable,
                    report_metallib,
                    report_reducer,
                ),
                "build report attempt",
            )
        ok(
            len(bounded_utf8_bytes("a" * (1 << 20), "tokenizer architecture", 1 << 20))
            == 1 << 20
        )
        ok(
            len(bounded_utf8_bytes("é" * (1 << 19), "tokenizer model", 1 << 20))
            == 1 << 20
        )
        expect_error(
            lambda: bounded_utf8_bytes("é" * ((1 << 19) + 1), "tokenizer pre", 1 << 20),
            "bounded nonempty UTF-8",
        )
        expect_error(
            lambda: bounded_utf8_bytes("", "tokenizer architecture", 1 << 20),
            "bounded nonempty UTF-8",
        )
        (report_root / "late-mutation").write_bytes(b"late")
        ok(
            measure_directory(
                synthetic_root_custody["path"], synthetic_root_custody["maximum"]
            )
            != synthetic_root_custody["measurement"]
        )
        fd_root = root / "open-fd-custody"
        fd_root.mkdir()
        oversized_compiler = fd_root / "oversized-compiler"
        oversized_descriptor = os.open(
            oversized_compiler,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o700,
        )
        os.ftruncate(oversized_descriptor, MAX_COMPILER_BYTES + 1)
        os.close(oversized_descriptor)
        expect_error(
            lambda: open_regular(
                oversized_compiler, "oversized compiler", MAX_COMPILER_BYTES
            ),
            "exceeds 1073741824 bytes",
        )
        fd_root_descriptor = os.open(
            fd_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        output_descriptor = os.open(
            "report",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=fd_root_descriptor,
        )
        output_bytes = b"authenticated report\n"
        os.write(output_descriptor, output_bytes)
        os.fsync(output_descriptor)
        output_snapshot = os.fstat(output_descriptor)
        ok(
            verify_open_output_fd(
                output_descriptor,
                fd_root_descriptor,
                "report",
                output_bytes,
                output_snapshot,
                fd_root / "report",
            )
            == hashlib.sha256(output_bytes).hexdigest()
        )
        os.rename(
            "report",
            "moved",
            src_dir_fd=fd_root_descriptor,
            dst_dir_fd=fd_root_descriptor,
        )
        replacement = os.open(
            "report",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=fd_root_descriptor,
        )
        os.write(replacement, output_bytes)
        os.close(replacement)
        expect_error(
            lambda: verify_open_output_fd(
                output_descriptor,
                fd_root_descriptor,
                "report",
                output_bytes,
                output_snapshot,
                fd_root / "report",
            ),
            "FD/path custody changed",
        )
        os.close(output_descriptor)
        hardlink_descriptor = os.open(
            "hardlink-report",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=fd_root_descriptor,
        )
        os.write(hardlink_descriptor, output_bytes)
        os.fsync(hardlink_descriptor)
        hardlink_snapshot = os.fstat(hardlink_descriptor)
        os.link(
            "hardlink-report",
            "hardlink-alias",
            src_dir_fd=fd_root_descriptor,
            dst_dir_fd=fd_root_descriptor,
        )
        expect_error(
            lambda: verify_open_output_fd(
                hardlink_descriptor,
                fd_root_descriptor,
                "hardlink-report",
                output_bytes,
                hardlink_snapshot,
                fd_root / "hardlink-report",
            ),
            "FD/path custody changed",
        )
        os.close(hardlink_descriptor)
        swap_a = os.open(
            "swap-a",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=fd_root_descriptor,
        )
        swap_b = os.open(
            "swap-b",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=fd_root_descriptor,
        )
        os.write(swap_a, b"a")
        os.write(swap_b, b"b")
        os.fsync(swap_a)
        os.fsync(swap_b)
        swap_a_snapshot = os.fstat(swap_a)
        os.rename(
            "swap-a",
            "swap-tmp",
            src_dir_fd=fd_root_descriptor,
            dst_dir_fd=fd_root_descriptor,
        )
        os.rename(
            "swap-b",
            "swap-a",
            src_dir_fd=fd_root_descriptor,
            dst_dir_fd=fd_root_descriptor,
        )
        os.rename(
            "swap-tmp",
            "swap-b",
            src_dir_fd=fd_root_descriptor,
            dst_dir_fd=fd_root_descriptor,
        )
        expect_error(
            lambda: verify_open_output_fd(
                swap_a,
                fd_root_descriptor,
                "swap-a",
                b"a",
                swap_a_snapshot,
                fd_root / "swap-a",
            ),
            "FD/path custody changed",
        )
        os.close(swap_a)
        os.close(swap_b)
        injected = os.open(
            "injected-partial",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=fd_root_descriptor,
        )
        original_write = os.write
        injection_calls = 0

        def injected_short_write(descriptor: int, data: bytes) -> int:
            nonlocal injection_calls
            injection_calls += 1
            if injection_calls == 1:
                return original_write(descriptor, data[:3])
            return 0

        os.write = injected_short_write
        try:
            expect_error(
                lambda: write_all_and_fsync(
                    injected, b"partial-data", "injected output"
                ),
                "short write",
            )
        finally:
            os.write = original_write
        ok(os.pread(injected, 3, 0) == b"par" and os.fstat(injected).st_size == 3)
        original_fsync = os.fsync

        def injected_fsync(_descriptor: int) -> None:
            raise OSError("injected fsync failure")

        os.fsync = injected_fsync
        try:
            try:
                write_all_and_fsync(injected, b"x", "injected fsync output")
            except OSError as error:
                ok("injected fsync failure" in str(error))
            else:
                raise AssertionError("injected fsync failure was accepted")
        finally:
            os.fsync = original_fsync
        os.close(injected)
        reduction_output_fd = os.open(
            "reduction-output",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=fd_root_descriptor,
        )
        write_all_and_fsync(reduction_output_fd, output_bytes, "synthetic reduction")
        reduction_snapshot = os.fstat(reduction_output_fd)
        reduction_parent_snapshot = os.fstat(fd_root_descriptor)
        ok(
            finalize_reduction_output(
                reduction_output_fd,
                fd_root_descriptor,
                fd_root / "reduction-output",
                output_bytes,
                reduction_snapshot,
                reduction_parent_snapshot,
            )
            == hashlib.sha256(output_bytes).hexdigest()
        )
        try:
            os.open(
                "reduction-output",
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=fd_root_descriptor,
            )
        except FileExistsError:
            tests += 1
        else:
            raise AssertionError("reduction output overwrite was accepted")
        os.link(
            "reduction-output",
            "reduction-output-hardlink",
            src_dir_fd=fd_root_descriptor,
            dst_dir_fd=fd_root_descriptor,
        )
        expect_error(
            lambda: finalize_reduction_output(
                reduction_output_fd,
                fd_root_descriptor,
                fd_root / "reduction-output",
                output_bytes,
                reduction_snapshot,
                os.fstat(fd_root_descriptor),
            ),
            "FD/path custody changed",
        )
        os.close(reduction_output_fd)
        reduction_sub_fd = os.open(
            "reduction-substitution",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=fd_root_descriptor,
        )
        write_all_and_fsync(reduction_sub_fd, output_bytes, "substitution reduction")
        reduction_sub_snapshot = os.fstat(reduction_sub_fd)
        os.rename(
            "reduction-substitution",
            "reduction-substitution-moved",
            src_dir_fd=fd_root_descriptor,
            dst_dir_fd=fd_root_descriptor,
        )
        reduction_replacement = os.open(
            "reduction-substitution",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=fd_root_descriptor,
        )
        os.write(reduction_replacement, output_bytes)
        os.close(reduction_replacement)
        expect_error(
            lambda: finalize_reduction_output(
                reduction_sub_fd,
                fd_root_descriptor,
                fd_root / "reduction-substitution",
                output_bytes,
                reduction_sub_snapshot,
                os.fstat(fd_root_descriptor),
            ),
            "FD/path custody changed",
        )
        os.close(reduction_sub_fd)
        os.close(fd_root_descriptor)
        stable_directory = root / "stable-directory-custody"
        stable_directory.mkdir()
        stable_fd, stable_snapshot = open_directory_custody(
            stable_directory, "stable synthetic Y"
        )
        verify_directory_custody(
            stable_fd,
            stable_directory,
            stable_snapshot,
            "stable synthetic Y",
            metadata_stable=True,
        )
        transient = stable_directory / "transient"
        transient.write_bytes(b"x")
        transient.unlink()
        expect_error(
            lambda: verify_directory_custody(
                stable_fd,
                stable_directory,
                stable_snapshot,
                "stable synthetic Y",
                metadata_stable=True,
            ),
            "metadata custody changed",
        )
        os.close(stable_fd)
        renamed_directory = root / "renamed-directory-custody"
        renamed_directory.mkdir()
        renamed_fd, renamed_snapshot = open_directory_custody(
            renamed_directory, "renamed synthetic Y"
        )
        moved_directory = root / "renamed-directory-custody-moved"
        renamed_directory.rename(moved_directory)
        renamed_directory.mkdir()
        expect_error(
            lambda: verify_directory_custody(
                renamed_fd,
                renamed_directory,
                renamed_snapshot,
                "renamed synthetic Y",
                metadata_stable=False,
            ),
            "path/FD custody changed",
        )
        os.close(renamed_fd)
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
        case = build_complete_synthetic_packet(Path(directory).resolve(strict=True))

        def bridge_args(value: dict[str, Any]) -> tuple[Any, ...]:
            return (
                value["inventory"],
                value["inventory_sha256"],
                value["inventory_spec"],
                value["inventory_spec_sha256"],
                value["preparation_choices"],
                value["preparation_choices_sha256"],
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
            case["preparation_choices"],
            case["preparation_choices_sha256"],
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
            case["preparation_choices"],
            case["preparation_choices_sha256"],
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
            case["preparation_choices"],
            case["preparation_choices_sha256"],
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
        original_output_writer = globals()["write_all_and_fsync"]

        def mutate_input_during_output(descriptor: int, data: bytes, name: str) -> None:
            original_output_writer(descriptor, data, name)
            if name == "reduction output":
                mutated = bytearray(baseline_target)
                mutated[-1] ^= 1
                case["target"].write_bytes(mutated)

        globals()["write_all_and_fsync"] = mutate_input_during_output
        try:
            expect_error(
                lambda: reduce(
                    case["trace"],
                    case["sidecar"],
                    case["manifest"],
                    Path(directory) / "input-mutated-during-output.json",
                    case["manifest_sha256"],
                    *bridge_args(case),
                ),
                "custody",
            )
        finally:
            globals()["write_all_and_fsync"] = original_output_writer
            case["target"].write_bytes(baseline_target)
        original_evidence_fsync = os.fsync
        directory_fsync_failed = False

        def fail_reduction_directory_fsync_once(descriptor: int) -> None:
            nonlocal directory_fsync_failed
            if (
                stat.S_ISDIR(os.fstat(descriptor).st_mode)
                and not directory_fsync_failed
            ):
                directory_fsync_failed = True
                raise OSError("injected reduction directory fsync failure")
            original_evidence_fsync(descriptor)

        os.fsync = fail_reduction_directory_fsync_once
        try:
            directory_partial = reduce(
                case["trace"],
                case["sidecar"],
                case["manifest"],
                Path(directory) / "directory-fsync-partial.json",
                case["manifest_sha256"],
                *bridge_args(case),
            )
        finally:
            os.fsync = original_evidence_fsync
        ok(
            directory_fsync_failed
            and directory_partial["result"] == "partial"
            and directory_partial["retry"] is False
            and directory_partial["output_bytes"] == 0
        )
        file_fsync_failed = False

        def fail_reduction_file_fsync_once(descriptor: int) -> None:
            nonlocal file_fsync_failed
            if stat.S_ISREG(os.fstat(descriptor).st_mode) and not file_fsync_failed:
                file_fsync_failed = True
                raise OSError("injected reduction file fsync failure")
            original_evidence_fsync(descriptor)

        os.fsync = fail_reduction_file_fsync_once
        try:
            file_partial = reduce(
                case["trace"],
                case["sidecar"],
                case["manifest"],
                Path(directory) / "file-fsync-partial.json",
                case["manifest_sha256"],
                *bridge_args(case),
            )
        finally:
            os.fsync = original_evidence_fsync
        ok(
            file_fsync_failed
            and file_partial["result"] == "partial"
            and file_partial["retry"] is False
            and file_partial["output_bytes"] > 0
        )
        for failure_kind in ("directory", "file"):

            def fail_evidence_fsync_persistently(
                descriptor: int, kind: str = failure_kind
            ) -> None:
                info = os.fstat(descriptor)
                if (kind == "directory" and stat.S_ISDIR(info.st_mode)) or (
                    kind == "file" and stat.S_ISREG(info.st_mode)
                ):
                    raise OSError(f"persistent {kind} fsync failure")
                original_evidence_fsync(descriptor)

            os.fsync = fail_evidence_fsync_persistently
            try:
                expect_error(
                    lambda kind=failure_kind: reduce(
                        case["trace"],
                        case["sidecar"],
                        case["manifest"],
                        Path(directory) / f"persistent-{kind}-fsync.json",
                        case["manifest_sha256"],
                        *bridge_args(case),
                    ),
                    "cannot durably bind retained partial",
                )
            finally:
                os.fsync = original_evidence_fsync
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

        case = build_complete_synthetic_packet(Path(directory).resolve(strict=True))
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

    # Frozen Q4 inventory-bootstrap interoperability and non-circular joins.
    ok(INVENTORY_VERSION == 2 and PREPARATION_SPEC_VERSION == 2)
    ok(len(SOURCE_ROLE_PATHS) == 18 and len(REQUIRED_SOURCE_ROLES) == 18)
    ok(
        SOURCE_ROLE_PATHS[-7:]
        == (
            ("tokenizer_rs", "crates/qwen-llm/src/tokenizer.rs"),
            ("gguf_rs", "crates/qwen-llm/src/gguf.rs"),
            ("source_identity_rs", "crates/qwen-cli/source_identity.rs"),
            ("workspace_cargo_toml", "Cargo.toml"),
            ("cargo_lock", "Cargo.lock"),
            ("qwen_cli_build_rs", "crates/qwen-cli/build.rs"),
            ("qwen_llm_lib_rs", "crates/qwen-llm/src/lib.rs"),
        )
    )
    token_vector = tokenizer_string_array_digest(
        "qwen.dflash_k0s.tokenizer.tokens.v1", ["a", "é", ""]
    )
    type_vector = tokenizer_i64_array_digest(
        "qwen.dflash_k0s.tokenizer.token_type.v1", [1, -2, 3]
    )
    merges_vector = tokenizer_string_array_digest(
        "qwen.dflash_k0s.tokenizer.merges.v1", ["a b", "c d"]
    )
    ok(
        token_vector
        == "09998d7bfbc29d046d3977772881975a5bba7cc5f53ab59f549ba1423bf2acb7"
    )
    ok(
        type_vector
        == "5250f1aace45739ec2bc66a954af11588f289061135f91af92ce92f4149705c5"
    )
    ok(
        merges_vector
        == "ce5370b088dd78101a85a79ce384ca468733cf9d08449b0231782eb01a8d9166"
    )
    ok(
        tokenizer_metadata_identity(
            "qwen",
            "gpt2",
            "qwen2",
            3,
            token_vector,
            3,
            type_vector,
            2,
            merges_vector,
            None,
            151645,
            False,
            True,
        )
        == "62fb1655306b266dd3971988e0bfc3c2b2491c8b107afdffb355f2d54083ebf2"
    )
    ok(
        BOOTSTRAP_TARGET_BYTES == 17106773984
        and BOOTSTRAP_DRAFTER_MAX_BYTES == 2147483648
        and SELECTOR_KERNEL == "kernel_mat_mat_q4_K_f32"
    )
    ok(
        dict(
            zip(
                PARSER_CAP_KEYS,
                (
                    MAX_GGUF_HEADER_BYTES,
                    MAX_GGUF_METADATA,
                    MAX_GGUF_TENSORS,
                    MAX_GGUF_STRINGS_BYTES,
                    MAX_GGUF_ARRAY_ITEMS_PER_ARRAY,
                    MAX_GGUF_ARRAY_ITEMS,
                    MAX_GGUF_OBJECTS,
                ),
            )
        )
        == {
            "header_bytes": 67108864,
            "metadata": 4096,
            "tensors": 8192,
            "strings_bytes": 16777216,
            "array_items_per_array": 500000,
            "array_items": 2000000,
            "objects": 2500000,
        }
    )
    frozen_assets = [
        {
            "role": "target",
            "path": BOOTSTRAP_TARGET_PATH,
            "expected_bytes": BOOTSTRAP_TARGET_BYTES,
            "max_bytes": BOOTSTRAP_TARGET_BYTES,
            "sha256": BOOTSTRAP_TARGET_SHA256,
        },
        {
            "role": "drafter",
            "path": BOOTSTRAP_DRAFTER_PATH,
            "expected_bytes": None,
            "max_bytes": BOOTSTRAP_DRAFTER_MAX_BYTES,
            "sha256": BOOTSTRAP_DRAFTER_SHA256,
        },
    ]
    ok(validate_bootstrap_asset_expectations(frozen_assets) == frozen_assets)
    swapped_assets = copy.deepcopy(frozen_assets)
    swapped_assets.reverse()
    expect_error(
        lambda: validate_bootstrap_asset_expectations(swapped_assets),
        "asset predicates",
    )
    wrong_target_size = copy.deepcopy(frozen_assets)
    wrong_target_size[0]["expected_bytes"] = None
    expect_error(
        lambda: validate_bootstrap_asset_expectations(wrong_target_size),
        "asset predicates",
    )
    wrong_drafter_size = copy.deepcopy(frozen_assets)
    wrong_drafter_size[1]["expected_bytes"] = 1
    expect_error(
        lambda: validate_bootstrap_asset_expectations(wrong_drafter_size),
        "asset predicates",
    )
    wrong_asset_cap = copy.deepcopy(frozen_assets)
    wrong_asset_cap[1]["max_bytes"] += 1
    expect_error(
        lambda: validate_bootstrap_asset_expectations(wrong_asset_cap),
        "asset predicates",
    )
    ok(
        PREPARATION_HASH_REPORT_KEYS
        == (
            "schema",
            "schema_version",
            "authority",
            "run_id",
            "attempt_id",
            "inventory",
            "inventory_spec",
            "preparation_choices",
            "preparation_spec",
            "reducer",
            "rendered",
            "worktree_x",
            "control_y",
            "report",
            "environment",
        )
        and "sha256" not in {"path", "max_bytes"}
    )
    ok(
        GIT_REPORT_IDENTITY_KEYS
        == (
            "path",
            "head",
            "tree",
            "status_bytes",
            "status_sha256",
            "ignored_bytes",
            "ignored_sha256",
            "common_git_dir",
            "object_store",
        )
        and MAX_GIT_STDOUT_BYTES == 64 << 20
        and MAX_GIT_STDERR_BYTES == 1 << 20
    )
    synthetic_git_report_source = {
        "path": Path("/tmp/k0s-x"),
        "head": "a" * 40,
        "tree": "b" * 40,
        "common": Path("/tmp/k0s-common"),
        "objects": Path("/tmp/k0s-common/objects"),
        "status": b"?? x\0",
        "ignored": b"ignored/path\0",
    }
    compact_git_identity = git_report_identity(synthetic_git_report_source)
    ok(
        tuple(compact_git_identity) == GIT_REPORT_IDENTITY_KEYS
        and compact_git_identity["status_bytes"] == 5
        and compact_git_identity["status_sha256"]
        == "48a28470edd91aafa57cf13c56fae2cff6b5a96f4f0eb4d802241005a3332ef0"
        and compact_git_identity["ignored_bytes"] == 13
        and compact_git_identity["ignored_sha256"]
        == "dc0aa908327d00962f4332b3bcb488bda5e576e51585a0efcf7d9c2895df8ec8"
    )
    empty_git_identity = git_report_identity(
        {**synthetic_git_report_source, "status": b"", "ignored": b""}
    )
    ok(
        empty_git_identity["status_sha256"]
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        and empty_git_identity["status_sha256"] != compact_git_identity["status_sha256"]
        and empty_git_identity["status_bytes"] == 0
    )
    observed_a4_ignored_bytes = 11_701_963
    ignored_entry = b"target/dependency\0"
    large_ignored = ignored_entry * (
        observed_a4_ignored_bytes // len(ignored_entry) + 1
    )
    large_git_identity = git_report_identity(
        {**synthetic_git_report_source, "status": b"", "ignored": large_ignored}
    )
    large_identity_bytes = json.dumps(
        {"worktree_x": large_git_identity, "control_y": large_git_identity},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    ok(
        large_git_identity["ignored_bytes"] > observed_a4_ignored_bytes
        and large_ignored.endswith(b"\0")
        and len(large_identity_bytes) < 2048
        and b"target/dependency" not in large_identity_bytes
        and large_ignored not in large_identity_bytes
    )
    build_checkout = {"commit": "a" * 40}
    build_claim = {"source_sha256": "b" * 64}
    build_info_v2 = {
        "artifact": {
            "path": "/tmp/build-info.json",
            "bytes": 1,
            "sha256": "0" * 64,
            "max_bytes": 1,
        },
        "schema_version": 2,
        "build_commit": "a" * 40,
        "build_commit_short": "a" * 9,
        "build_dirty": False,
        "build_source_state": f"git-source-sha256-v2:{'b' * 64}",
        "stamp_source": "git",
        "stamp_error": None,
        "runtime_commit": "a" * 40,
        "runtime_dirty": False,
        "runtime_source_state": f"git-source-sha256-v2:{'b' * 64}",
        "status": "match",
        "problems": [],
        "overrides": [],
    }
    ok(
        validate_build_info_report(build_info_v2, build_checkout, build_claim)
        == build_info_v2
    )
    build_info_v1 = copy.deepcopy(build_info_v2)
    build_info_v1["schema_version"] = 1
    expect_error(
        lambda: validate_build_info_report(build_info_v1, build_checkout, build_claim),
        "build-info report mismatch",
    )
    build_info_extra = copy.deepcopy(build_info_v2)
    build_info_extra["extra"] = None
    expect_error(
        lambda: validate_build_info_report(
            build_info_extra, build_checkout, build_claim
        ),
        "keys/order mismatch",
    )
    with tempfile.TemporaryDirectory() as bootstrap_directory:
        bootstrap_root = Path(bootstrap_directory).resolve(strict=True)
        v1_spec_path = bootstrap_root / "inventory-spec-v1.json"
        v1_inventory_path = bootstrap_root / "inventory-v1.json"
        v1_spec = {
            key: (
                INVENTORY_SPEC_SCHEMA
                if key == "schema"
                else 1
                if key == "schema_version"
                else "synthetic-v1"
                if key == "run_id"
                else MAX_TRACE_BYTES
                if key == "inventory_max_bytes"
                else None
            )
            for key in INVENTORY_SPEC_KEYS
        }
        v1_spec_path.write_text(
            json.dumps(v1_spec, separators=(",", ":")), encoding="utf-8"
        )
        v1_inventory_path.write_text("{}", encoding="utf-8")
        expect_error(
            lambda: validate_inventory(
                v1_inventory_path,
                hashlib.sha256(v1_inventory_path.read_bytes()).hexdigest(),
                v1_spec_path,
                hashlib.sha256(v1_spec_path.read_bytes()).hexdigest(),
            ),
            "incompatible v2",
        )
        bootstrap_case = build_complete_synthetic_packet(bootstrap_root)
        prep_value = parse_json(
            bootstrap_case["preparation_spec"].read_bytes(), "bootstrap prep"
        )
        choices_value = parse_json(
            bootstrap_case["preparation_choices"].read_bytes(), "bootstrap choices"
        )
        for name, forged_bytes in (
            (
                "whitespace",
                json.dumps(choices_value, ensure_ascii=True, indent=1).encode("ascii")
                + b"\n",
            ),
            (
                "order",
                canonical_json_bytes(
                    {
                        "schema_version": choices_value["schema_version"],
                        "schema": choices_value["schema"],
                        **{
                            key: value
                            for key, value in choices_value.items()
                            if key not in {"schema", "schema_version"}
                        },
                    }
                ),
            ),
            (
                "escape",
                bootstrap_case["preparation_choices"]
                .read_bytes()
                .replace(b'"schema"', b'"\\u0073chema"', 1),
            ),
        ):
            forged_path = bootstrap_root / f"choices-{name}.json"
            forged_path.write_bytes(forged_bytes)
            opened_forgery = open_regular(
                forged_path, f"choices {name} forgery", MAX_BOOTSTRAP_SPEC_BYTES
            )
            try:
                expect_error(
                    lambda item=opened_forgery: parse_canonical_json_object(
                        item, PREPARATION_CHOICES_KEYS, "forged preparation choices"
                    ),
                    "canonical JSON" if name != "order" else "keys/order mismatch",
                )
            finally:
                opened_forgery.close()
        forged_spec_path = bootstrap_root / "preparation-whitespace.json"
        forged_spec_path.write_bytes(
            json.dumps(prep_value, ensure_ascii=True, separators=(", ", ": ")).encode(
                "ascii"
            )
            + b"\n"
        )
        opened_forgery = open_regular(
            forged_spec_path,
            "preparation spec whitespace forgery",
            MAX_BOOTSTRAP_SPEC_BYTES,
        )
        try:
            expect_error(
                lambda: parse_canonical_json_object(
                    opened_forgery, PREPARATION_SPEC_KEYS, "forged preparation spec"
                ),
                "canonical JSON",
            )
        finally:
            opened_forgery.close()
        validate_preparation_join(
            prep_value,
            choices_value,
            prep_value["preparation_choices"],
        )
        ok(True)
        bad_version = copy.deepcopy(prep_value)
        bad_version["schema_version"] = 1
        expect_error(
            lambda: validate_preparation_join(
                bad_version, choices_value, prep_value["preparation_choices"]
            ),
            "incompatible v2",
        )
        bad_join = copy.deepcopy(prep_value)
        bad_join["continuation_carry_token"] += 1
        expect_error(
            lambda: validate_preparation_join(
                bad_join, choices_value, prep_value["preparation_choices"]
            ),
            "template join mutation",
        )
        reordered_spec_failure = copy.deepcopy(prep_value)
        reordered_spec_failure["failure_policy"] = {
            "retry": False,
            "on_collision": "retain_reserved_partial",
        }
        expect_error(
            lambda: validate_preparation_join(
                reordered_spec_failure,
                choices_value,
                prep_value["preparation_choices"],
            ),
            "failure_policy keys/order mismatch",
        )
        reordered_choices_failure = copy.deepcopy(choices_value)
        reordered_choices_failure["failure_policy"] = {
            "retry": False,
            "on_collision": "retain_reserved_partial",
        }
        expect_error(
            lambda: validate_preparation_join(
                prep_value,
                reordered_choices_failure,
                prep_value["preparation_choices"],
            ),
            "failure_policy keys/order mismatch",
        )
        reordered_manifest_request = copy.deepcopy(choices_value)
        request_value = reordered_manifest_request["manifest_choices"][
            "expected_request"
        ]
        reordered_manifest_request["manifest_choices"]["expected_request"] = {
            "ignored_target_policy": request_value["ignored_target_policy"],
            "request": request_value["request"],
        }
        expect_error(
            lambda: validate_preparation_join(
                prep_value,
                reordered_manifest_request,
                prep_value["preparation_choices"],
            ),
            "expected_request keys/order mismatch",
        )
        reordered_selector_environment = copy.deepcopy(choices_value)
        selector = reordered_selector_environment["manifest_choices"][
            "selector_dispatch_predicate"
        ]
        selector["allowed_environment"] = {
            "QWEN_METAL_LEASE_WAIT": selector["allowed_environment"][
                "QWEN_METAL_LEASE_WAIT"
            ]
        }
        # A one-key object has no alternate order; use the predicate itself.
        reordered_selector_environment["manifest_choices"][
            "selector_dispatch_predicate"
        ] = {
            "kernel": selector["kernel"],
            **{key: value for key, value in selector.items() if key != "kernel"},
        }
        expect_error(
            lambda: validate_preparation_join(
                prep_value,
                reordered_selector_environment,
                prep_value["preparation_choices"],
            ),
            "selector predicate keys/order mismatch",
        )
        for placeholder, final_key in (
            (INVENTORY_PATH_PLACEHOLDER, "inventory_path"),
            (INVENTORY_SHA256_PLACEHOLDER, "inventory_sha256"),
            (INVENTORY_SPEC_PATH_PLACEHOLDER, "inventory_spec_path"),
            (INVENTORY_SPEC_SHA256_PLACEHOLDER, "inventory_spec_sha256"),
        ):
            missing = copy.deepcopy(choices_value)
            index = missing["reducer_argv_template"].index(placeholder)
            missing["reducer_argv_template"][index] = "${MISSING}"
            expect_error(
                lambda value=missing: validate_preparation_join(
                    prep_value, value, prep_value["preparation_choices"]
                ),
                "exactly once",
            )
            duplicate = copy.deepcopy(choices_value)
            duplicate["reducer_argv_template"].append(placeholder)
            expect_error(
                lambda value=duplicate: validate_preparation_join(
                    prep_value, value, prep_value["preparation_choices"]
                ),
                "exactly once",
            )
            literal = copy.deepcopy(choices_value)
            literal_index = literal["reducer_argv_template"].index(placeholder)
            literal["reducer_argv_template"][literal_index] = prep_value[final_key]
            expect_error(
                lambda value=literal: validate_preparation_join(
                    prep_value, value, prep_value["preparation_choices"]
                ),
                "literal final inventory value",
            )
        unknown_placeholder = copy.deepcopy(choices_value)
        unknown_placeholder["reducer_argv_template"].append("${INVENTORY_UNKNOWN}")
        expect_error(
            lambda: validate_preparation_join(
                prep_value, unknown_placeholder, prep_value["preparation_choices"]
            ),
            "unknown inventory placeholder",
        )
        bad_cycle = copy.deepcopy(prep_value)
        cycle_choices = copy.deepcopy(choices_value)
        bad_cycle["failure_policy"]["on_collision"] = "expected_fixture_sha256"
        cycle_choices["failure_policy"]["on_collision"] = "expected_fixture_sha256"
        expect_error(
            lambda: validate_preparation_join(
                bad_cycle, cycle_choices, prep_value["preparation_choices"]
            ),
            "output cycle",
        )
        ok(
            prep_value["reducer_argv"].count(PREPARATION_CHOICES_SHA256_PLACEHOLDER)
            == 1
            and choices_value["reducer_argv_template"].count(
                PREPARATION_CHOICES_SHA256_PLACEHOLDER
            )
            == 1
        )
    with tempfile.TemporaryDirectory() as render_directory:
        render_root = Path(render_directory).resolve(strict=True)
        render_x = render_root / "x"
        render_x.mkdir()
        render_case = build_complete_synthetic_packet(render_x)
        render_reducer_path = render_x / "reducer-copy.py"
        render_reducer_path.write_bytes(Path(__file__).resolve().read_bytes())
        (render_x / ".gitignore").write_text(".DS_Store\ntarget/\n", encoding="ascii")
        git_bin = "/usr/bin/git"
        for command in (
            [git_bin, "init", str(render_x)],
            [git_bin, "-C", str(render_x), "add", "-A"],
            [
                git_bin,
                "-C",
                str(render_x),
                "-c",
                "user.name=K0S Self Test",
                "-c",
                "user.email=k0s@example.invalid",
                "commit",
                "-m",
                "synthetic X",
            ],
        ):
            subprocess.run(
                command,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        for key, value in (
            ("core.fsmonitor", "/tmp/k0s-forbidden-fsmonitor"),
            ("core.untrackedCache", "true"),
            ("core.hooksPath", "/tmp/k0s-forbidden-hooks"),
        ):
            subprocess.run(
                [git_bin, "-C", str(render_x), "config", key, value],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        ok(
            bounded_git(render_x, "config", "--get", "core.fsmonitor") == b"false\n"
            and bounded_git(render_x, "config", "--get", "core.untrackedCache")
            == b"false\n"
            and bounded_git(render_x, "config", "--get", "core.hooksPath")
            == b"/dev/null\n"
        )
        render_p = render_root / "p"
        render_y = render_root / "y"
        for branch, path in (("k0s-p", render_p), ("k0s-y", render_y)):
            subprocess.run(
                [
                    git_bin,
                    "-C",
                    str(render_x),
                    "worktree",
                    "add",
                    "-b",
                    branch,
                    str(path),
                    "HEAD",
                ],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        git_x = inspect_git_checkout(render_x, "render X")
        git_y = inspect_git_checkout(render_y, "render Y")
        render_inventory = parse_json(
            render_case["inventory"].read_bytes(), "render inventory"
        )
        render_reducer_claim = synthetic_file_claim(
            render_reducer_path, MAX_TRACE_BYTES
        )
        for section in (render_inventory["expected"], render_inventory["observed"]):
            section["checkout"] = {
                "path": str(render_x),
                "commit": git_x["head"],
                "tree": git_x["tree"],
                "dirty": False,
            }
            section["build"]["commit"] = git_x["head"]
            section["reducer"] = render_reducer_claim
        render_choices = parse_json(
            render_case["preparation_choices"].read_bytes(), "render choices"
        )
        render_choices["manifest_choices"] = {
            key: copy.deepcopy(render_case["manifest_object"][key])
            for key in MANIFEST_CHOICE_KEYS
        }
        render_choices["worktree_x"] = {
            "path": str(render_x),
            "commit": git_x["head"],
        }
        render_choices["planner_p"] = {"path": str(render_p)}
        render_choices["control_y"] = {
            "path": str(render_y),
            "commit": git_y["head"],
            "tree": git_y["tree"],
        }
        output_names = {
            "fixture": str(render_y / "prepared-fixture.json"),
            "command": str(render_y / "prepared-command.json"),
            "manifest": str(render_y / "prepared-manifest.json"),
            "seal": str(render_y / "prepared-seal.json"),
        }
        render_choices["outputs"] = output_names
        render_choices["acquisition_outputs"] = {
            "manifest": output_names["manifest"],
            "trace": str(render_y / "acquisition.trace"),
            "sidecar": str(render_y / "acquisition.sidecar"),
        }
        render_choices["reduction_output"] = str(render_y / "reduction.json")
        render_choices["transformation_sha256"] = render_reducer_claim["sha256"]
        choices_path = render_p / "choices.json"
        prep_path = render_p / "preparation.json"
        render_choices["preparation_spec_path"] = str(prep_path)
        temporary_spec = copy.deepcopy(prep_value)
        temporary_spec.update(
            {
                "inventory_path": str(render_case["inventory"]),
                "inventory_sha256": render_case["inventory_sha256"],
                "inventory_spec_path": str(render_case["inventory_spec"]),
                "inventory_spec_sha256": render_case["inventory_spec_sha256"],
                "preparation_spec_path": str(prep_path),
                "worktree_x": render_choices["worktree_x"],
                "planner_p": render_choices["planner_p"],
                "control_y_input": render_choices["control_y"],
                "outputs": output_names,
                "acquisition_outputs": render_choices["acquisition_outputs"],
                "reduction_output": render_choices["reduction_output"],
                "transformation_sha256": render_reducer_claim["sha256"],
                "manifest_choices": render_choices["manifest_choices"],
            }
        )
        placeholder_claim = {
            "path": str(choices_path),
            "bytes": 1,
            "sha256": "0" * 64,
            "max_bytes": MAX_BOOTSTRAP_SPEC_BYTES,
        }
        temporary_spec["preparation_choices"] = placeholder_claim
        actual_template = frozen_reducer_argv(
            temporary_spec, render_inventory["observed"]["reducer"]["path"]
        )
        substitutions = {
            temporary_spec["inventory_path"]: INVENTORY_PATH_PLACEHOLDER,
            temporary_spec["inventory_sha256"]: INVENTORY_SHA256_PLACEHOLDER,
            temporary_spec["inventory_spec_path"]: INVENTORY_SPEC_PATH_PLACEHOLDER,
            temporary_spec["inventory_spec_sha256"]: INVENTORY_SPEC_SHA256_PLACEHOLDER,
        }
        render_choices["reducer_argv_template"] = [
            substitutions.get(value, value) for value in actual_template
        ]
        choices_path.write_bytes(canonical_json_bytes(render_choices))
        choices_claim = synthetic_file_claim(choices_path, MAX_BOOTSTRAP_SPEC_BYTES)
        temporary_spec["preparation_choices"] = choices_claim
        temporary_spec["reducer_argv"] = frozen_reducer_argv(
            temporary_spec, render_inventory["observed"]["reducer"]["path"]
        )
        prep_path.write_bytes(canonical_json_bytes(temporary_spec))
        subprocess.run(
            [git_bin, "-C", str(render_p), "add", "choices.json", "preparation.json"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                git_bin,
                "-C",
                str(render_p),
                "-c",
                "user.name=K0S Self Test",
                "-c",
                "user.email=k0s@example.invalid",
                "commit",
                "-m",
                "synthetic P",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        prep_claim = synthetic_file_claim(prep_path, MAX_BOOTSTRAP_SPEC_BYTES)
        render_claims = {
            "inventory": synthetic_file_claim(
                render_case["inventory"], MAX_TRACE_BYTES
            ),
            "inventory_spec": synthetic_file_claim(
                render_case["inventory_spec"], MAX_BOOTSTRAP_SPEC_BYTES
            ),
            "preparation_spec": prep_claim,
        }
        for key, replacement, reason in (
            ("arm_order", ["on-A", "off-A", "on-B", "off-B"], "arm/parity"),
            ("selected_arm", "on-B", "arm/parity"),
            ("parity_comparison_fields", [], "arm/parity"),
            ("environment_allowlist", {}, "environment_allowlist"),
            (
                "failure_policy",
                {"on_collision": "retain_reserved_partial", "retry": True},
                "failure policy",
            ),
        ):
            forged_spec = copy.deepcopy(temporary_spec)
            forged_choices = copy.deepcopy(render_choices)
            forged_spec[key] = copy.deepcopy(replacement)
            choices_key = "control_y" if key == "control_y_input" else key
            forged_choices[choices_key] = copy.deepcopy(replacement)
            expect_error(
                lambda s=forged_spec, c=forged_choices: validate_preparation_prerender(
                    render_inventory,
                    render_case["inventory_sha256"],
                    s,
                    c,
                    choices_claim,
                ),
                reason,
            )
        aliased_output_spec = copy.deepcopy(temporary_spec)
        aliased_output_choices = copy.deepcopy(render_choices)
        aliased_output_spec["acquisition_outputs"]["sidecar"] = aliased_output_spec[
            "acquisition_outputs"
        ]["trace"]
        aliased_output_choices["acquisition_outputs"] = copy.deepcopy(
            aliased_output_spec["acquisition_outputs"]
        )
        expect_error(
            lambda: validate_preparation_prerender(
                render_inventory,
                render_case["inventory_sha256"],
                aliased_output_spec,
                aliased_output_choices,
                choices_claim,
            ),
            "seven future leaves alias",
        )
        broken_future = Path(temporary_spec["acquisition_outputs"]["trace"])
        broken_future.symlink_to(render_y / "missing-target")
        expect_error(
            lambda: validate_preparation_prerender(
                render_inventory,
                render_case["inventory_sha256"],
                temporary_spec,
                render_choices,
                choices_claim,
            )
        )
        broken_future.unlink()
        ignored_y = render_y / ".DS_Store"
        ignored_y.write_bytes(b"ignored")
        expect_error(
            lambda: validate_preparation_prerender(
                render_inventory,
                render_case["inventory_sha256"],
                temporary_spec,
                render_choices,
                choices_claim,
            ),
            "ignored",
        )
        ignored_y.unlink()
        ignored_p = render_p / ".DS_Store"
        ignored_p.write_bytes(b"ignored")
        expect_error(
            lambda: validate_preparation_prerender(
                render_inventory,
                render_case["inventory_sha256"],
                temporary_spec,
                render_choices,
                choices_claim,
            ),
            "ignored",
        )
        ignored_p.unlink()
        x_drift_path = render_reducer_path
        x_drift_bytes = x_drift_path.read_bytes()
        x_drift_path.write_bytes(x_drift_bytes + b"drift")
        expect_error(
            lambda: validate_preparation_prerender(
                render_inventory,
                render_case["inventory_sha256"],
                temporary_spec,
                render_choices,
                choices_claim,
            ),
            "not completely clean",
        )
        x_drift_path.write_bytes(x_drift_bytes)
        y_drift_path = render_y / "manifest.json"
        y_drift_bytes = y_drift_path.read_bytes()
        y_drift_path.write_bytes(y_drift_bytes + b"drift")
        expect_error(
            lambda: validate_preparation_prerender(
                render_inventory,
                render_case["inventory_sha256"],
                temporary_spec,
                render_choices,
                choices_claim,
            ),
            "not completely clean",
        )
        y_drift_path.write_bytes(y_drift_bytes)
        rendered_once = render_preparation(
            render_inventory,
            render_case["inventory_sha256"],
            temporary_spec,
            prep_claim["sha256"],
            render_claims,
            render_choices,
            choices_claim,
        )
        rendered_twice = render_preparation(
            copy.deepcopy(render_inventory),
            render_case["inventory_sha256"],
            copy.deepcopy(temporary_spec),
            prep_claim["sha256"],
            copy.deepcopy(render_claims),
            copy.deepcopy(render_choices),
            copy.deepcopy(choices_claim),
        )
        ok(rendered_once == rendered_twice and len(rendered_once) == 4)
    print(
        f"dflash_k0s self-test: PASS ({tests} checks; stdlib-only, no model/Metal/network)"
    )
    return


def main() -> int:
    literal_argv = list(sys.argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--validate-inventory", action="store_true")
    modes.add_argument("--observe-preparation-hashes", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-sha256")
    parser.add_argument("--inventory-spec", type=Path)
    parser.add_argument("--inventory-spec-sha256")
    parser.add_argument("--preparation-choices", type=Path)
    parser.add_argument("--preparation-choices-sha256")
    parser.add_argument("--preparation-spec", type=Path)
    parser.add_argument("--preparation-spec-sha256")
    parser.add_argument("--preparation-seal", type=Path)
    parser.add_argument("--preparation-seal-sha256")
    for name in ("fixture", "command", "static-manifest", "seal"):
        parser.add_argument(f"--expected-{name}-sha256")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--report-max-bytes", type=int)
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
        args.preparation_choices,
        args.preparation_choices_sha256,
        args.preparation_spec,
        args.preparation_spec_sha256,
        args.preparation_seal,
        args.preparation_seal_sha256,
        args.expected_fixture_sha256,
        args.expected_command_sha256,
        args.expected_static_manifest_sha256,
        args.expected_seal_sha256,
        args.report_output,
        args.report_max_bytes,
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
        validate_preparation_environment(dict(os.environ))
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
                + preparation_arguments[:4]
                + preparation_arguments[6:10]
            )
            and not any(preparation_arguments[4:6] + preparation_arguments[10:])
            and not any(reduction_arguments),
            "preparation requires inventory/spec paths and all expected hashes",
        )
        observed_environment = validate_preparation_environment(dict(os.environ))
        validate_literal_mode_argv(
            literal_argv,
            [
                str(Path(__file__).resolve()),
                "--prepare",
                "--inventory",
                str(args.inventory),
                "--inventory-sha256",
                args.inventory_sha256,
                "--inventory-spec",
                str(args.inventory_spec),
                "--inventory-spec-sha256",
                args.inventory_spec_sha256,
                "--preparation-choices",
                str(args.preparation_choices),
                "--preparation-choices-sha256",
                args.preparation_choices_sha256,
                "--preparation-spec",
                str(args.preparation_spec),
                "--preparation-spec-sha256",
                args.preparation_spec_sha256,
                "--expected-fixture-sha256",
                args.expected_fixture_sha256,
                "--expected-command-sha256",
                args.expected_command_sha256,
                "--expected-static-manifest-sha256",
                args.expected_static_manifest_sha256,
                "--expected-seal-sha256",
                args.expected_seal_sha256,
            ],
            str(Path(__file__).resolve()),
        )
        prepared = prepare_artifacts(
            args.inventory,
            args.inventory_sha256,
            args.inventory_spec,
            args.inventory_spec_sha256,
            args.preparation_choices,
            args.preparation_choices_sha256,
            args.preparation_spec,
            args.preparation_spec_sha256,
            {
                "fixture": args.expected_fixture_sha256,
                "command": args.expected_command_sha256,
                "manifest": args.expected_static_manifest_sha256,
                "seal": args.expected_seal_sha256,
            },
            observed_environment,
        )
        print(json.dumps(prepared, separators=(",", ":")))
        return 0
    if args.observe_preparation_hashes:
        require(
            all(
                inventory_arguments
                + preparation_arguments[:4]
                + preparation_arguments[10:]
            )
            and not any(preparation_arguments[4:10])
            and not any(reduction_arguments),
            "preparation hash observation requires four authenticated inputs and one report output/cap",
        )
        observed_environment = validate_preparation_environment(dict(os.environ))
        validate_literal_mode_argv(
            literal_argv,
            [
                str(Path(__file__).resolve()),
                "--observe-preparation-hashes",
                "--inventory",
                str(args.inventory),
                "--inventory-sha256",
                args.inventory_sha256,
                "--inventory-spec",
                str(args.inventory_spec),
                "--inventory-spec-sha256",
                args.inventory_spec_sha256,
                "--preparation-choices",
                str(args.preparation_choices),
                "--preparation-choices-sha256",
                args.preparation_choices_sha256,
                "--preparation-spec",
                str(args.preparation_spec),
                "--preparation-spec-sha256",
                args.preparation_spec_sha256,
                "--report-output",
                str(args.report_output),
                "--report-max-bytes",
                str(args.report_max_bytes),
            ],
            str(Path(__file__).resolve()),
        )
        observed = observe_preparation_hashes(
            args.inventory,
            args.inventory_sha256,
            args.inventory_spec,
            args.inventory_spec_sha256,
            args.preparation_choices,
            args.preparation_choices_sha256,
            args.preparation_spec,
            args.preparation_spec_sha256,
            args.report_output,
            args.report_max_bytes,
            observed_environment,
        )
        print(json.dumps(observed, separators=(",", ":")))
        return 0 if observed["result"] == "observed" else 1
    require(
        all(reduction_arguments + inventory_arguments + preparation_arguments[:6])
        and not any(preparation_arguments[6:]),
        "reduction requires trace/sidecar/manifest/output plus independently hashed inventory, inventory-spec, preparation-choices, preparation-spec, and preparation-seal",
    )
    validate_preparation_environment(dict(os.environ))
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
        args.preparation_choices,
        args.preparation_choices_sha256,
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
