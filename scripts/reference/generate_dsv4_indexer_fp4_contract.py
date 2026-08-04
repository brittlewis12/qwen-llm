# /// script
# requires-python = "==3.14.*"
# dependencies = []
# ///

"""Generate the packed DeepSeek V4 Lightning Indexer FP4 contract fixture.

Reference sources are read with ``git show REV:PATH``. The pinned revisions
must exist in the local object stores, but checkout HEAD and working-tree state
do not participate in generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT / "crates/qwen-llm/tests/fixtures/deepseek_v4_indexer_fp4_contract_v1.json"
)

OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
OFFICIAL_HASHES = {
    "inference/kernel.py": "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2",
    "inference/model.py": "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f",
}
VLLM_REVISION = "b40d859c7b07ae244bcd8c6eecdcdbd9a3afaa07"
VLLM_HASHES = {
    "vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py": (
        "8671d02dc1c2c495c3b5f16608faca59fd8d6507664fdcd93eda49ab0be1c6cb"
    ),
    "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py": (
        "2c0f33f5a6b06af371011fb5683af84fcf7ee502043041a57f74bb4b116d0495"
    ),
    "tests/kernels/test_fused_indexer_q_rope_quant.py": (
        "4536a4eb2315102f5a2d9684a613f1599a10992e813d355e5b1b9dfb7f31442a"
    ),
}
DWARFSTAR_REVISION = "54b36ed9ba42da31b24f2d1a5feb075c2475dbb1"
DWARFSTAR_HASHES = {
    "ds4.c": "af5df58420632c453657ffdfc2c7cb84e75135bbcc20deaca3fedf970c13930c",
    "metal/dsv4_kv.metal": "b8a77a47f145ec13ee1528a038a86baff19f66f6843c3f639a51128aa0551835",
}

VALUES_PER_ROW = 128
BLOCK_VALUES = 32
BLOCK_COUNT = 4
VALUE_BYTES = 64
ROW_BYTES = 68
AMAX_FLOOR_BITS = 0x01C00000
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def from_f32_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bf16_round(value: float) -> float:
    bits = f32_bits(value)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return from_f32_bits(rounded)


def encode_scale(maximum: float) -> int:
    floor = from_f32_bits(AMAX_FLOOR_BITS)
    ratio = f32(f32(max(maximum, floor)) * f32(1.0 / 6.0))
    bits = f32_bits(ratio)
    exponent_field = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent_field in (0, 0xFF):
        raise ValueError("scale ratio must be finite and normal")
    exponent = exponent_field - 127 + int(mantissa != 0)
    code = exponent + 127
    if not 1 <= code <= 253:
        raise ValueError(f"noncanonical scale code {code}")
    return code


def decode_scale(code: int) -> float:
    if not 1 <= code <= 253:
        raise ValueError(f"noncanonical scale code {code}")
    return from_f32_bits(code << 23)


def encode_e2m1(value: float) -> int:
    absolute = min(abs(value), 6.0)
    if absolute > 5.0:
        magnitude = 7
    elif absolute >= 3.5:
        magnitude = 6
    elif absolute > 2.5:
        magnitude = 5
    elif absolute >= 1.75:
        magnitude = 4
    elif absolute > 1.25:
        magnitude = 3
    elif absolute >= 0.75:
        magnitude = 2
    elif absolute > 0.25:
        magnitude = 1
    else:
        magnitude = 0
    return magnitude | (((f32_bits(value) >> 31) & 1) << 3)


def decode_e2m1(code: int, scale: float) -> float:
    value = f32(f32(E2M1_VALUES[code & 7]) * scale)
    return f32(-value if code & 8 else value)


def pack_row(values: list[float]) -> tuple[list[int], list[float]]:
    if len(values) != VALUES_PER_ROW:
        raise ValueError("row must contain 128 values")
    rounded = [bf16_round(value) for value in values]
    if not all(math.isfinite(value) for value in rounded):
        raise ValueError("BF16 conversion overflow")
    output = [0] * ROW_BYTES
    for block_index in range(BLOCK_COUNT):
        block = rounded[block_index * BLOCK_VALUES : (block_index + 1) * BLOCK_VALUES]
        code = encode_scale(max(abs(value) for value in block))
        scale = decode_scale(code)
        output[VALUE_BYTES + block_index] = code
        byte_start = block_index * (BLOCK_VALUES // 2)
        for pair in range(BLOCK_VALUES // 2):
            low = encode_e2m1(f32(block[2 * pair] / scale))
            high = encode_e2m1(f32(block[2 * pair + 1] / scale))
            output[byte_start + pair] = low | (high << 4)
    return output, unpack_row(output)


def unpack_row(row: list[int]) -> list[float]:
    if len(row) != ROW_BYTES:
        raise ValueError("packed row must contain 68 bytes")
    values = [0.0] * VALUES_PER_ROW
    for block_index in range(BLOCK_COUNT):
        scale = decode_scale(row[VALUE_BYTES + block_index])
        value_start = block_index * BLOCK_VALUES
        byte_start = block_index * (BLOCK_VALUES // 2)
        for pair in range(BLOCK_VALUES // 2):
            packed = row[byte_start + pair]
            values[value_start + 2 * pair] = decode_e2m1(packed & 0xF, scale)
            values[value_start + 2 * pair + 1] = decode_e2m1(packed >> 4, scale)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("decoded row is nonfinite")
    return values


def dot(left: list[float], right: list[float]) -> float:
    total = f32(0.0)
    for lhs, rhs in zip(left, right, strict=True):
        total = f32(total + f32(lhs * rhs))
    return total


def packed_scores(
    queries: list[list[int]], scaled_weights: list[float], keys: list[list[int]]
) -> list[float]:
    decoded_queries = [unpack_row(row) for row in queries]
    decoded_keys = [unpack_row(row) for row in keys]
    scores = []
    for key in decoded_keys:
        score = f32(0.0)
        for query, weight in zip(decoded_queries, scaled_weights, strict=True):
            contribution = f32(max(dot(query, key), 0.0) * weight)
            score = f32(score + contribution)
        scores.append(score)
    return scores


def checkout(env_name: str, default: str) -> Path:
    repo = Path(os.environ.get(env_name, os.path.expanduser(default))).resolve()
    if not repo.is_dir():
        raise SystemExit(f"{env_name} checkout missing: {repo}")
    return repo


def verify_revision(repo: Path, revision: str, hashes: dict[str, str]) -> None:
    for relative, expected in hashes.items():
        try:
            blob = subprocess.run(
                ["git", "show", f"{revision}:{relative}"],
                cwd=repo,
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode(errors="replace").strip()
            raise SystemExit(
                f"cannot read {repo} {revision}:{relative}: {detail}"
            ) from error
        observed_hash = hashlib.sha256(blob).hexdigest()
        if observed_hash != expected:
            raise SystemExit(
                f"{repo} {revision}:{relative} sha256 {observed_hash} "
                f"does not match pinned {expected}"
            )


def row_fixture(name: str, values: list[float]) -> dict[str, object]:
    packed, decoded = pack_row(values)
    return {
        "name": name,
        "input_bits": [f32_bits(value) for value in values],
        "packed_bytes": packed,
        "decoded_bits": [f32_bits(value) for value in decoded],
    }


def score_fixture(
    name: str,
    query_values: list[list[float]],
    raw_weights: list[float],
    key_values: list[list[float]],
) -> dict[str, object]:
    head_count = len(query_values)
    if head_count == 0 or len(raw_weights) != head_count:
        raise ValueError("score fixture requires one weight per nonempty query head")
    normalization = f32(1.0 / f32(math.sqrt(f32(head_count * VALUES_PER_ROW))))
    scaled_weights = [f32(f32(weight) * normalization) for weight in raw_weights]
    query_rows = [pack_row(values)[0] for values in query_values]
    key_rows = [pack_row(values)[0] for values in key_values]
    scores = packed_scores(query_rows, scaled_weights, key_rows)
    return {
        "name": name,
        "evidence_kind": "deterministic scalar transcription over decoded packed operands",
        "head_weight_semantics": "projection weights pre-scaled once by the recorded normalization",
        "normalization_bits": f32_bits(normalization),
        "raw_head_weight_bits": [f32_bits(value) for value in raw_weights],
        "scaled_head_weight_bits": [f32_bits(value) for value in scaled_weights],
        "query_rows": query_rows,
        "key_rows": key_rows,
        "score_bits": [f32_bits(value) for value in scores],
        "top2": sorted(range(len(scores)), key=lambda index: (-scores[index], index))[
            :2
        ],
    }


def invalid_row_fixture(
    name: str, mutations: list[tuple[int, int]], error_category: str
) -> dict[str, object]:
    row = [0] * ROW_BYTES
    row[VALUE_BYTES:] = [127] * BLOCK_COUNT
    for byte_index, byte_value in mutations:
        row[byte_index] = byte_value
    try:
        unpack_row(row)
    except (OverflowError, ValueError):
        pass
    else:
        raise ValueError(f"invalid row fixture {name} unexpectedly decoded")
    return {
        "name": name,
        "mutations": [
            {"byte_index": byte_index, "byte_value": byte_value}
            for byte_index, byte_value in mutations
        ],
        "error_category": error_category,
    }


def build_fixture() -> dict[str, object]:
    official = checkout("DSV4_OFFICIAL_DIR", "~/code/DeepSeek-V4-Flash-0731")
    vllm = checkout("DSV4_VLLM_DIR", "~/code/vllm")
    dwarfstar = checkout("DSV4_DWARFSTAR_DIR", "~/code/ds4")
    verify_revision(official, OFFICIAL_REVISION, OFFICIAL_HASHES)
    verify_revision(vllm, VLLM_REVISION, VLLM_HASHES)
    verify_revision(dwarfstar, DWARFSTAR_REVISION, DWARFSTAR_HASHES)

    known = [0.0] * VALUES_PER_ROW
    known[:16] = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -0.0,
        -0.125,
    ]
    large = [0.0] * VALUES_PER_ROW
    large[0] = 70_000.0
    bf16_boundary = [0.0] * VALUES_PER_ROW
    bf16_boundary[31] = from_f32_bits(f32_bits(6.0) + 1)
    bf16_boundary[32] = from_f32_bits(0x40C10000)
    bf16_boundary[63] = -from_f32_bits(0x40C10000)
    bf16_boundary[64] = 24.0
    bf16_boundary[95] = -24.0
    bf16_boundary[96] = 48.0
    query_values = [
        [f32((index - 63.5) / 32.0) for index in range(128)],
        [f32(((index * 17) % 43 - 21.0) / 16.0) for index in range(128)],
    ]
    key_values = [
        [f32(((index * 7) % 31 - 15.0) / 8.0) for index in range(128)],
        [f32(((index * 11) % 37 - 18.0) / 8.0) for index in range(128)],
        [f32(((index * 13) % 41 - 20.0) / 8.0) for index in range(128)],
    ]
    relu_query_values = [[0.0] * 128 for _ in range(2)]
    relu_query_values[0][0] = 1.0
    relu_query_values[1][0] = -1.0
    relu_key_values = [[0.0] * 128 for _ in range(3)]
    relu_key_values[0][0] = -1.0
    relu_key_values[1][0] = 1.0

    cancellation_query_values = [[0.0] * 128 for _ in range(2)]
    cancellation_query_values[0][0] = 1.0
    cancellation_query_values[1][0] = 1.0
    cancellation_key_values = [[0.0] * 128 for _ in range(3)]
    cancellation_key_values[0][0] = 1.0
    cancellation_key_values[1][0] = 2.0

    rounding_cases = []
    for midpoint in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        bits = f32_bits(midpoint)
        for case_bits in (
            bits - 1,
            bits,
            bits + 1,
            (bits - 1) | 0x80000000,
            bits | 0x80000000,
            (bits + 1) | 0x80000000,
        ):
            value = from_f32_bits(case_bits)
            rounding_cases.append({"input_bits": case_bits, "code": encode_e2m1(value)})

    scale_inputs = [("zero", 0.0), ("floor", from_f32_bits(AMAX_FLOOR_BITS))]
    for exponent in (-80, -20, 0, 42, 100):
        maximum = f32(6.0 * math.ldexp(1.0, exponent))
        scale_inputs.extend(
            [
                (f"e{exponent}_exact", maximum),
                (f"e{exponent}_above", from_f32_bits(f32_bits(maximum) + 1)),
            ]
        )
    scale_inputs.append(("bf16_max", from_f32_bits(0x7F7F0000)))

    return {
        "schema_version": 1,
        "generator_version": 1,
        "evidence_kind": (
            "official BF16 QAT source, vLLM packed reference, and DwarfStar "
            "QAT cross-check"
        ),
        "sources": {
            "official": {
                "repository": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731",
                "revision": OFFICIAL_REVISION,
                "files": OFFICIAL_HASHES,
                "role": "model placement and BF16-input E2M1/UE8M0 semantics",
            },
            "vllm": {
                "repository": "https://github.com/vllm-project/vllm",
                "revision": VLLM_REVISION,
                "files": VLLM_HASHES,
                "role": "independent packed nibble, scale, Q, and paged K reference",
            },
            "dwarfstar": {
                "repository": "https://github.com/antirez/ds4",
                "revision": DWARFSTAR_REVISION,
                "files": DWARFSTAR_HASHES,
                "role": "independent Hadamard plus FP4 simulation cross-check",
            },
        },
        "layout": {
            "values_per_row": VALUES_PER_ROW,
            "block_values": BLOCK_VALUES,
            "block_count": BLOCK_COUNT,
            "value_bytes": VALUE_BYTES,
            "scale_bytes": BLOCK_COUNT,
            "scale_offset": VALUE_BYTES,
            "row_bytes": ROW_BYTES,
            "upstream_storage": (
                "Q uses separate value/scale tensors; paged K stores all block "
                "values before block scales"
            ),
            "fixture_row_envelope": "64 packed value bytes followed by 4 scale bytes",
            "nibble_order": "dimension 2*i low; dimension 2*i+1 high",
            "scale_order": "dimensions 0..31,32..63,64..95,96..127",
            "bf16_input_semantics": True,
            "rounding": "E2M1 round-to-nearest-even",
            "rounding_case_input": "finite scaled E2M1 conversion operand",
            "scale_case_input": "already-BF16-rounded nonnegative block amax",
            "scalar_row_domain": (
                "construction requires every decoded value to remain finite F32; "
                "physically encodable overflow rows are rejected"
            ),
            "amax_floor_bits": AMAX_FLOOR_BITS,
            "canonical_scale_codes": [1, 253],
        },
        "codebook_bits": [
            f32_bits(-value if code & 8 else value)
            for code in range(16)
            for value in [E2M1_VALUES[code & 7]]
        ],
        "rounding_cases": rounding_cases,
        "scale_cases": [
            {
                "name": name,
                "maximum_bits": f32_bits(maximum),
                "code": encode_scale(maximum),
                "decoded_scale_bits": f32_bits(decode_scale(encode_scale(maximum))),
            }
            for name, maximum in scale_inputs
        ],
        "rows": [
            row_fixture("known_codebook_and_signed_zero", known),
            row_fixture("bf16_finite_beyond_f16_range", large),
            row_fixture("bf16_amax_and_four_block_order", bf16_boundary),
            row_fixture("deterministic_query_0", query_values[0]),
            row_fixture("deterministic_key_0", key_values[0]),
        ],
        "invalid_rows": [
            *[
                invalid_row_fixture(
                    f"scale_code_{code}_block_{block}",
                    [(VALUE_BYTES + block, code)],
                    "noncanonical_scale_code",
                )
                for block in range(BLOCK_COUNT)
                for code in (0, 254, 255)
            ],
            *[
                invalid_row_fixture(
                    f"decoded_f32_overflow_block_{block}_{nibble}",
                    [
                        (VALUE_BYTES + block, 253),
                        (
                            block * (BLOCK_VALUES // 2),
                            0x06 if nibble == "low" else 0x60,
                        ),
                    ],
                    "decoded_f32_overflow",
                )
                for block in range(BLOCK_COUNT)
                for nibble in ("low", "high")
            ],
        ],
        "pack_rejections": [
            {
                "name": "bf16_max_decodes_beyond_f32",
                "dimension": 0,
                "input_bits": 0x7F7F0000,
                "error_category": "decoded_f32_overflow",
            }
        ],
        "score_cases": [
            score_fixture(
                "deterministic_mixed",
                query_values,
                [f32(0.75), f32(-0.25)],
                key_values,
            ),
            score_fixture(
                "relu_before_negative_weight",
                relu_query_values,
                [f32(1.0), f32(-0.5)],
                relu_key_values,
            ),
            score_fixture(
                "exact_head_cancellation_and_row_tie",
                cancellation_query_values,
                [f32(1.0), f32(-1.0)],
                cancellation_key_values,
            ),
        ],
    }


def main() -> None:
    serialized = json.dumps(build_fixture(), indent=2, sort_keys=True) + "\n"
    arguments = sys.argv[1:]
    if arguments == ["--check"]:
        if not OUTPUT.is_file() or OUTPUT.read_text() != serialized:
            raise SystemExit(f"fixture drift: regenerate {OUTPUT.relative_to(ROOT)}")
        print(f"verified {OUTPUT.relative_to(ROOT)}")
    elif arguments:
        raise SystemExit("usage: generate_dsv4_indexer_fp4_contract.py [--check]")
    else:
        OUTPUT.write_text(serialized)
        print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
