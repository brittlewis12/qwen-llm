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

RECORD_KEYS = ("schema", "schema_version", "run_id", "event", "payload")
EVENT_ORDER = ("run",) + ("depth", "lattice") * DEPTHS + ("end",)
RUN_KEYS = (
    "authority",
    "geometry",
    "request",
    "proposal_abstention",
    "ignored_target_policy",
    "binding",
    "provenance",
    "capture",
    "identities",
    "semantic_references",
    "tensors",
    "sidecar_registry",
    "production_chain",
    "fixed_chains",
)
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
    "expected_binding",
    "expected_fixed_chains",
    "expected_capture_context",
    "expected_build",
    "expected_host",
    "embedded_metallib_sha256",
    "expected_selector_dispatch",
    "scalar_contract",
)

PROVENANCE_KEYS = (
    "dispatch_census",
    "selector_hidden_dispatch",
    "kernel_trace",
    "embedded_metallib_sha256",
    "build",
    "host",
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
        len(lines) == MAX_RECORDS, f"trace must contain exactly {MAX_RECORDS} records"
    )
    rows: list[dict[str, Any]] = []
    run_id: str | None = None
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
        require(
            row["event"] == EVENT_ORDER[index],
            f"trace event order mismatch at record {index + 1}",
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
    check(
        selector == manifest["expected_selector_dispatch"],
        "tagged selector dispatch differs from static manifest expectation",
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
    fixed = manifest["expected_fixed_chains"]
    require(len(argv) == 24 + 2 * len(fixed), "command argv length mismatch")
    index = 0

    def take(value: str, name: str) -> None:
        nonlocal index
        require(argv[index] == value, f"command argv {name} mismatch")
        index += 1

    take(str(external["executable"].path), "executable")
    take("dflash-k0s-lattice", "subcommand")
    take("--model", "model flag")
    take(str(external["target"].path), "target path")
    take("--drafter", "drafter flag")
    take(str(external["drafter"].path), "drafter path")
    take("--prompt", "prompt flag")
    text(argv[index], "command prompt", maximum=1 << 20)
    index += 1
    take("--carry-token", "carry flag")
    take(str(manifest["expected_capture_context"]["carry_token"]), "carry value")
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
    exact_keys(manifest["expected_binding"], BINDING_KEYS, "manifest expected binding")
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
    selector = exact_keys(
        manifest["expected_selector_dispatch"],
        DISPATCH_KEYS,
        "manifest expected selector dispatch",
    )
    require(
        selector["tag"] == SELECTOR_DISPATCH_TAG,
        "manifest selector dispatch tag mismatch",
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


def reduce(
    trace_path: Path,
    sidecar_path: Path,
    manifest_path: Path,
    output_path: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    opened: list[OpenFile] = []
    result: dict[str, Any]
    run_id: str | None = None
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
            "artifacts": [
                {"path": str(item.path), "bytes": item.size, "sha256": item.digest}
                for item in sorted(opened[4:], key=lambda value: str(value.path))
            ],
        }
        rows = parse_trace(trace)
        require(rows[0]["run_id"] == run_id, "manifest/trace run id mismatch")
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
            "result": "invalid",
            "authority": AUTHORITY,
            "reason": f"{type(error).__name__}: {error}",
            "metrics": None,
            "custody": custody or None,
        }
    custody = custody_summary(opened) or custody
    result["custody"] = custody or None
    try:
        for item in opened:
            final_custody_check(item)
    except Exception as error:
        result = {
            "schema": REDUCTION_SCHEMA,
            "schema_version": REDUCTION_VERSION,
            "run_id": run_id,
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
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(specs), 0))
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
        "kernel": "kernel_mat_mat_q4_k_f32",
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
        "--model",
        str(target_path.resolve()),
        "--drafter",
        str(drafter_path.resolve()),
        "--prompt",
        "synthetic prompt",
        "--carry-token",
        "0",
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
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "run_id": run_id,
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
        "expected_binding": binding,
        "expected_fixed_chains": static_fixed_chains,
        "expected_capture_context": capture_context,
        "expected_build": build,
        "expected_host": expected_host,
        "embedded_metallib_sha256": metallib_sha256,
        "expected_selector_dispatch": expected_selector_dispatch,
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
    sidecar_path.write_bytes(sidecar)

    state["synchronized_event_sha256"] = synchronized_event_digest(capture, materials)
    capture["synchronized_capture_sha256"] = capture_digest(
        provenance, capture, materials, tensor_claims
    )
    for depth_payload in depth_payloads:
        depth_payload["synchronized_capture_sha256"] = capture[
            "synchronized_capture_sha256"
        ]

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
    }


def self_test() -> None:
    tests = 0

    def ok(condition: bool) -> None:
        nonlocal tests
        assert condition
        tests += 1

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
        expect_error(lambda: parse_trace(trace_file), "exactly 16")
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
        output = Path(directory) / "reduction.json"
        reduced = reduce(
            case["trace"],
            case["sidecar"],
            case["manifest"],
            output,
            case["manifest_sha256"],
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

        baseline_records = copy.deepcopy(case["records"])
        baseline_manifest = copy.deepcopy(case["manifest_object"])
        baseline_sidecar = case["sidecar"].read_bytes()
        baseline_fixture = case["fixture"].read_bytes()
        baseline_command = case["command"].read_bytes()
        baseline_target = case["target"].read_bytes()
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
            variant_output = Path(directory) / f"mutation-{mutation_number}.json"
            result = reduce(
                case["trace"],
                case["sidecar"],
                case["manifest"],
                variant_output,
                digest,
            )
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

        wrong_digest_output = Path(directory) / "wrong-manifest-digest.json"
        wrong_digest = reduce(
            case["trace"],
            case["sidecar"],
            case["manifest"],
            wrong_digest_output,
            "0" * 64,
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
            )
        except InvalidEvidence as error:
            ok(
                "exclusive-create output failed" in str(error)
                and collision.read_bytes() == b"do-not-overwrite"
            )
        else:
            raise AssertionError("public reducer overwrote an existing output")

    print(
        f"dflash_k0s self-test: PASS ({tests} checks; stdlib-only, no model/Metal/network)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
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
    if args.self_test:
        require(
            not any(
                (
                    args.input,
                    args.sidecar,
                    args.manifest,
                    args.manifest_sha256,
                    args.output,
                )
            ),
            "--self-test cannot be combined with reduction arguments",
        )
        self_test()
        return 0
    require(
        all(
            (
                args.input,
                args.sidecar,
                args.manifest,
                args.manifest_sha256,
                args.output,
            )
        ),
        "reduction requires --input, --sidecar, --manifest, --manifest-sha256, and --output",
    )
    result = reduce(
        args.input,
        args.sidecar,
        args.manifest,
        args.output,
        args.manifest_sha256,
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    return 0 if result["result"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InvalidEvidence as error:
        print(f"dflash_k0s: invalid: {error}", file=sys.stderr)
        raise SystemExit(2)
