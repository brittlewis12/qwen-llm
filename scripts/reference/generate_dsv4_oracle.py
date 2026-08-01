# /// script
# requires-python = "==3.14.*"
# dependencies = ["numpy==2.5.1"]
# ///

"""Generate deterministic DeepSeek V4 operation fixtures.

The NumPy cases are manual equation transcriptions from the pinned references.
Additional harnesses execute pinned DwarfStar and llama.cpp CPU implementations,
so those cases are external differential vectors rather than sibling formulas.
The fixture stays small enough for normal CPU CI and never loads model weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "crates/qwen-llm/tests/fixtures/deepseek_v4_oracle_v1.json"
F = np.float32
DWARFSTAR_REVISION = "54b36ed9ba42da31b24f2d1a5feb075c2475dbb1"
DWARFSTAR_SOURCE_SHA256 = (
    "af5df58420632c453657ffdfc2c7cb84e75135bbcc20deaca3fedf970c13930c"
)
LLAMA_CPP_REVISION = "876a4321163249c43ca4e986818fab5ab081f282"
LLAMA_CPP_TREE = "b127fd3e9b45bef820e6e6914f53f71270a9a6f9"
LLAMA_CPP_SOURCE_SHA256 = {
    "ggml/include/ggml.h": (
        "c65c30fdb4dce95eac71c26bb38ae8423fbc80d79db91d2b2ffaea8c4e46276a"
    ),
    "ggml/src/ggml.c": (
        "9e40ad07323c7925f06a105119dfb07c1d4a21d3263a9e9bd0bd21792c42e1e4"
    ),
    "ggml/src/ggml-cpu/ops.cpp": (
        "dd7265a7402515002d4679f18d5d1d423456e87f256f7cc11af3bbb4148bf773"
    ),
}


def f(value: float | np.floating) -> np.float32:
    return F(value)


def as_json(values: np.ndarray | list[float]) -> list[float | None]:
    return [
        float(F(value)) if math.isfinite(float(value)) else None for value in values
    ]


def dot(left: np.ndarray, right: np.ndarray) -> np.float32:
    total = f(0.0)
    for lhs, rhs in zip(left, right, strict=True):
        total = f(total + f(lhs * rhs))
    return total


def matvec(weights: np.ndarray, input_values: np.ndarray) -> np.ndarray:
    return np.asarray([dot(row, input_values) for row in weights], dtype=np.float32)


def rms_norm(values: np.ndarray, weight: np.ndarray | None, eps: float) -> np.ndarray:
    sum_squares = sum(float(value) * float(value) for value in values)
    mean_squares = f(sum_squares / len(values))
    scale = f(1.0 / math.sqrt(float(f(mean_squares + f(eps)))))
    output = np.empty_like(values)
    for index, value in enumerate(values):
        output[index] = f(value * scale)
        if weight is not None:
            output[index] = f(output[index] * weight[index])
    return output


def head_rms_norm(values: np.ndarray, head_count: int, head_dim: int) -> np.ndarray:
    output = values.copy()
    for head in range(head_count):
        start = head * head_dim
        output[start : start + head_dim] = rms_norm(
            output[start : start + head_dim], None, 1.0e-6
        )
    return output


def sigmoid(value: float | np.floating) -> np.float32:
    return f(1.0 / (1.0 + math.exp(-float(value))))


def softplus(value: float | np.floating) -> np.float32:
    value = float(value)
    if value > 20.0:
        return f(value)
    if value < -20.0:
        return f(math.exp(value))
    return f(math.log1p(math.exp(value)))


def split_sinkhorn(
    mixes: np.ndarray,
    scale: np.ndarray,
    base: np.ndarray,
    count: int,
    iterations: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = np.empty(count, dtype=np.float32)
    post = np.empty(count, dtype=np.float32)
    for index in range(count):
        affine = f(f(mixes[index] * scale[0]) + base[index])
        pre[index] = f(sigmoid(affine) + f(eps))
        offset = count + index
        affine = f(f(mixes[offset] * scale[1]) + base[offset])
        post[index] = f(f(2.0) * sigmoid(affine))

    matrix = np.empty((count, count), dtype=np.float32)
    offset = 2 * count
    for row in range(count):
        for column in range(count):
            index = row * count + column
            matrix[row, column] = f(
                f(mixes[offset + index] * scale[2]) + base[offset + index]
            )
        maximum = max(float(value) for value in matrix[row])
        row_sum = f(0.0)
        for column in range(count):
            matrix[row, column] = f(math.exp(float(f(matrix[row, column] - maximum))))
            row_sum = f(row_sum + matrix[row, column])
        inverse = f(1.0 / float(row_sum))
        for column in range(count):
            matrix[row, column] = f(f(matrix[row, column] * inverse) + f(eps))

    normalize_columns(matrix, eps)
    for _ in range(1, iterations):
        normalize_rows(matrix, eps)
        normalize_columns(matrix, eps)
    return pre, post, matrix


def normalize_rows(matrix: np.ndarray, eps: float) -> None:
    for row in range(matrix.shape[0]):
        total = f(0.0)
        for value in matrix[row]:
            total = f(total + value)
        inverse = f(1.0 / float(f(total + f(eps))))
        for column in range(matrix.shape[1]):
            matrix[row, column] = f(matrix[row, column] * inverse)


def normalize_columns(matrix: np.ndarray, eps: float) -> None:
    for column in range(matrix.shape[1]):
        total = f(0.0)
        for row in range(matrix.shape[0]):
            total = f(total + matrix[row, column])
        inverse = f(1.0 / float(f(total + f(eps))))
        for row in range(matrix.shape[0]):
            matrix[row, column] = f(matrix[row, column] * inverse)


def mhc_fixture() -> dict[str, object]:
    hidden_size = 3
    count = 4
    residual = np.asarray(
        [(index - 5.0) * 0.2 for index in range(12)], dtype=np.float32
    )
    function = np.asarray(
        [((index * 17 % 29) - 14.0) * 0.006 for index in range(12 * 24)],
        dtype=np.float32,
    ).reshape(24, 12)
    scale = np.asarray([0.5, -0.25, 0.8], dtype=np.float32)
    base = np.asarray(
        [((index * 5 % 11) - 5.0) * 0.03 for index in range(24)],
        dtype=np.float32,
    )
    normalized = rms_norm(residual, None, 1.0e-6)
    mixes = matvec(function, normalized)
    pre, post, combination = split_sinkhorn(mixes, scale, base, count, 20, 1.0e-6)
    collapsed = np.zeros(hidden_size, dtype=np.float32)
    for stream in range(count):
        for dimension in range(hidden_size):
            collapsed[dimension] = f(
                collapsed[dimension]
                + f(residual[stream * hidden_size + dimension] * pre[stream])
            )

    block_output = np.asarray([0.4, -0.7, 1.1], dtype=np.float32)
    post_output = np.empty_like(residual)
    for destination in range(count):
        for dimension in range(hidden_size):
            value = f(block_output[dimension] * post[destination])
            for source in range(count):
                value = f(
                    value
                    + f(
                        combination[source, destination]
                        * residual[source * hidden_size + dimension]
                    )
                )
            post_output[destination * hidden_size + dimension] = value

    head_function = function[:count]
    head_scale = f(0.75)
    head_base = base[:count]
    head_mixes = matvec(head_function, normalized)
    head_pre = np.asarray(
        [
            f(sigmoid(f(f(mix * head_scale) + bias)) + f(1.0e-6))
            for mix, bias in zip(head_mixes, head_base, strict=True)
        ],
        dtype=np.float32,
    )
    head_output = np.zeros(hidden_size, dtype=np.float32)
    for stream in range(count):
        for dimension in range(hidden_size):
            head_output[dimension] = f(
                head_output[dimension]
                + f(residual[stream * hidden_size + dimension] * head_pre[stream])
            )

    return {
        "hidden_size": hidden_size,
        "connection_count": count,
        "rms_eps": 1.0e-6,
        "hc_eps": 1.0e-6,
        "sinkhorn_iterations": 20,
        "residual": as_json(residual),
        "function": as_json(function.reshape(-1)),
        "scale": as_json(scale),
        "base": as_json(base),
        "expected_mixes": as_json(mixes),
        "expected_pre": as_json(pre),
        "expected_post": as_json(post),
        "expected_combination": as_json(combination.reshape(-1)),
        "expected_input": as_json(collapsed),
        "block_output": as_json(block_output),
        "expected_post_output": as_json(post_output),
        "head_function": as_json(head_function.reshape(-1)),
        "head_scale": float(head_scale),
        "head_base": as_json(head_base),
        "expected_head_output": as_json(head_output),
    }


def yarn_correction_range(
    rotary_dim: int,
    theta: float,
    original_context: int,
    beta_fast: float,
    beta_slow: float,
) -> tuple[np.float32, np.float32]:
    def correction(rotations: float) -> float:
        return (
            rotary_dim
            * math.log(original_context / (rotations * 2.0 * math.pi))
            / (2.0 * math.log(theta))
        )

    return f(max(0.0, math.floor(correction(beta_fast)))), f(
        min(rotary_dim - 1, math.ceil(correction(beta_slow)))
    )


def rope_tail(
    values: np.ndarray,
    head_count: int,
    head_dim: int,
    position: int,
    rotary_dim: int,
    theta: float,
    scaling_factor: float,
    original_context: int,
    beta_fast: float,
    beta_slow: float,
    inverse: bool = False,
) -> np.ndarray:
    output = values.copy()
    theta_scale = f(theta ** (-2.0 / rotary_dim))
    frequency_scale = f(1.0 / scaling_factor)
    yarn = scaling_factor > 1.0
    low, high = (
        yarn_correction_range(rotary_dim, theta, original_context, beta_fast, beta_slow)
        if yarn
        else (f(0.0), f(0.0))
    )
    sine_sign = -1.0 if inverse else 1.0
    tail_offset = head_dim - rotary_dim
    for head in range(head_count):
        extrapolated = f(position)
        start = head * head_dim + tail_offset
        for pair_offset in range(0, rotary_dim, 2):
            interpolated = f(frequency_scale * extrapolated)
            if yarn:
                denominator = max(0.001, float(f(high - low)))
                ratio = min(
                    1.0,
                    max(0.0, (pair_offset // 2 - float(low)) / denominator),
                )
                ramp = f(1.0 - ratio)
                angle = f(f(interpolated * f(1.0 - ramp)) + f(extrapolated * ramp))
            else:
                angle = extrapolated
            sine = f(math.sin(float(angle)) * sine_sign)
            cosine = f(math.cos(float(angle)))
            first = output[start + pair_offset]
            second = output[start + pair_offset + 1]
            output[start + pair_offset] = f(f(first * cosine) - f(second * sine))
            output[start + pair_offset + 1] = f(f(first * sine) + f(second * cosine))
            extrapolated = f(extrapolated * theta_scale)
    return output


def rope_fixture() -> dict[str, object]:
    values = np.asarray([9.0, 8.0, 7.0, 6.0, 1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    local = rope_tail(values, 1, 8, 17, 4, 10_000.0, 1.0, 0, 32.0, 1.0)
    yarn = rope_tail(values, 1, 8, 65_536, 4, 160_000.0, 16.0, 65_536, 32.0, 1.0)
    return {
        "input": as_json(values),
        "head_count": 1,
        "head_dim": 8,
        "rotary_dim": 4,
        "local_position": 17,
        "expected_local": as_json(local),
        "yarn_position": 65_536,
        "expected_yarn": as_json(yarn),
    }


def shared_kv_projection_fixture() -> dict[str, object]:
    hidden_size = 4
    q_lora_rank = 3
    head_count = 2
    head_dim = 4
    input_values = np.asarray([0.5, -1.0, 1.5, 0.25], dtype=np.float32)
    q_a = np.asarray(
        [
            ((row * 7 + column * 3) % 11 - 5.0) * 0.08
            for row in range(q_lora_rank)
            for column in range(hidden_size)
        ],
        dtype=np.float32,
    ).reshape(q_lora_rank, hidden_size)
    q_a_norm = np.asarray([1.0, 0.75, 1.25], dtype=np.float32)
    q_b = np.asarray(
        [
            ((row * 5 + column * 2) % 13 - 6.0) * 0.06
            for row in range(head_count * head_dim)
            for column in range(q_lora_rank)
        ],
        dtype=np.float32,
    ).reshape(head_count * head_dim, q_lora_rank)
    kv_weight = np.asarray(
        [
            ((row * 3 + column * 5) % 9 - 4.0) * 0.09
            for row in range(head_dim)
            for column in range(hidden_size)
        ],
        dtype=np.float32,
    ).reshape(head_dim, hidden_size)
    kv_norm = np.asarray([1.0, 0.5, 1.5, 0.8], dtype=np.float32)
    q_lora_raw = matvec(q_a, input_values)
    q_lora = rms_norm(q_lora_raw, q_a_norm, 1.0e-6)
    queries = head_rms_norm(matvec(q_b, q_lora), head_count, head_dim)
    kv_raw = matvec(kv_weight, input_values)
    kv = rms_norm(kv_raw, kv_norm, 1.0e-6)
    return {
        "hidden_size": hidden_size,
        "q_lora_rank": q_lora_rank,
        "head_count": head_count,
        "head_dim": head_dim,
        "input": as_json(input_values),
        "q_a": as_json(q_a.reshape(-1)),
        "q_a_norm": as_json(q_a_norm),
        "q_b": as_json(q_b.reshape(-1)),
        "kv_weight": as_json(kv_weight.reshape(-1)),
        "kv_norm": as_json(kv_norm),
        "expected_q_lora_raw": as_json(q_lora_raw),
        "expected_q_lora": as_json(q_lora),
        "expected_queries": as_json(queries),
        "expected_kv_raw": as_json(kv_raw),
        "expected_kv": as_json(kv),
    }


def mixed_attention(
    queries: np.ndarray,
    head_count: int,
    head_dim: int,
    raw_kv: np.ndarray,
    compressed_kv: np.ndarray,
    allowed: list[bool],
    sinks: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(queries)
    scale = f(1.0 / math.sqrt(head_dim))
    for head in range(head_count):
        query = queries[head * head_dim : (head + 1) * head_dim]
        rows = list(raw_kv) + [
            row for row, include in zip(compressed_kv, allowed, strict=True) if include
        ]
        logits = [f(dot(query, row) * scale) for row in rows]
        maximum = max([float(sinks[head]), *[float(value) for value in logits]])
        denominator = f(math.exp(float(f(sinks[head] - maximum))))
        head_output = np.zeros(head_dim, dtype=np.float32)
        for logit, row in zip(logits, rows, strict=True):
            weight = f(math.exp(float(f(logit - maximum))))
            denominator = f(denominator + weight)
            for index in range(head_dim):
                head_output[index] = f(head_output[index] + f(weight * row[index]))
        for index in range(head_dim):
            head_output[index] = f(head_output[index] / denominator)
        output[head * head_dim : (head + 1) * head_dim] = head_output
    return output


def attention_output_fixture() -> dict[str, object]:
    head_count = 4
    head_dim = 4
    group_count = 2
    rank = 3
    hidden_size = 5
    queries = np.asarray(
        [((index * 7) % 17 - 8.0) * 0.1 for index in range(head_count * head_dim)],
        dtype=np.float32,
    )
    raw_kv = np.asarray(
        [
            ((row * 5 + index * 3) % 13 - 6.0) * 0.12
            for row in range(3)
            for index in range(head_dim)
        ],
        dtype=np.float32,
    ).reshape(3, head_dim)
    compressed_kv = np.asarray(
        [
            ((row * 11 + index * 2) % 15 - 7.0) * 0.08
            for row in range(3)
            for index in range(head_dim)
        ],
        dtype=np.float32,
    ).reshape(3, head_dim)
    allowed = [True, False, True]
    sinks = np.asarray([-0.25, 0.0, 0.5, -1.0], dtype=np.float32)
    attention = mixed_attention(
        queries, head_count, head_dim, raw_kv, compressed_kv, allowed, sinks
    )
    inverse = rope_tail(
        attention,
        head_count,
        head_dim,
        131_071,
        4,
        160_000.0,
        16.0,
        65_536,
        32.0,
        1.0,
        inverse=True,
    )
    group_width = head_count * head_dim // group_count
    low_rank_width = group_count * rank
    output_a = np.asarray(
        [
            ((row * 13 + column * 7) % 19 - 9.0) * 0.04
            for row in range(low_rank_width)
            for column in range(group_width)
        ],
        dtype=np.float32,
    ).reshape(low_rank_width, group_width)
    output_b = np.asarray(
        [
            ((row * 3 + column * 5) % 17 - 8.0) * 0.05
            for row in range(hidden_size)
            for column in range(low_rank_width)
        ],
        dtype=np.float32,
    ).reshape(hidden_size, low_rank_width)
    low_rank = np.empty(low_rank_width, dtype=np.float32)
    for group in range(group_count):
        group_input = inverse[group * group_width : (group + 1) * group_width]
        for rank_index in range(rank):
            output_index = group * rank + rank_index
            low_rank[output_index] = dot(output_a[output_index], group_input)
    output = matvec(output_b, low_rank)
    return {
        "head_count": head_count,
        "head_dim": head_dim,
        "queries": as_json(queries),
        "raw_kv": as_json(raw_kv.reshape(-1)),
        "compressed_kv": as_json(compressed_kv.reshape(-1)),
        "compressed_allowed": allowed,
        "sinks": as_json(sinks),
        "expected_attention": as_json(attention),
        "inverse_position": 131_071,
        "expected_inverse": as_json(inverse),
        "group_count": group_count,
        "rank": rank,
        "hidden_size": hidden_size,
        "output_a": as_json(output_a.reshape(-1)),
        "output_b": as_json(output_b.reshape(-1)),
        "expected_low_rank": as_json(low_rank),
        "expected_output": as_json(output),
    }


class Compressor:
    def __init__(self, ratio: int, head_dim: int):
        self.ratio = ratio
        self.head_dim = head_dim
        self.coff = 2 if ratio == 4 else 1
        self.width = self.coff * head_dim
        self.kv = np.zeros((self.coff * ratio, self.width), dtype=np.float32)
        self.scores = np.full_like(self.kv, -np.inf)

    def push(
        self,
        position: int,
        projected_kv: np.ndarray,
        projected_scores: np.ndarray,
        ape: np.ndarray,
        norm_weight: np.ndarray,
        rope: dict[str, float | int],
    ) -> np.ndarray | None:
        position_in_group = position % self.ratio
        row = self.ratio + position_in_group if self.ratio == 4 else position_in_group
        self.kv[row] = projected_kv
        self.scores[row] = projected_scores + ape[position_in_group]
        if (position + 1) % self.ratio:
            return None

        pooled = np.zeros(self.head_dim, dtype=np.float32)
        for dimension in range(self.head_dim):
            entries: list[tuple[np.float32, np.float32]] = []
            if self.ratio == 4:
                for state_row in range(self.ratio):
                    entries.append(
                        (
                            self.scores[state_row, dimension],
                            self.kv[state_row, dimension],
                        )
                    )
                    entries.append(
                        (
                            self.scores[
                                self.ratio + state_row, self.head_dim + dimension
                            ],
                            self.kv[self.ratio + state_row, self.head_dim + dimension],
                        )
                    )
            else:
                entries = [
                    (self.scores[state_row, dimension], self.kv[state_row, dimension])
                    for state_row in range(self.ratio)
                ]
            maximum = max(float(score) for score, _ in entries)
            denominator = f(0.0)
            numerator = f(0.0)
            for score, value in entries:
                weight = f(math.exp(float(f(score - maximum))))
                denominator = f(denominator + weight)
                numerator = f(numerator + f(weight * value))
            pooled[dimension] = f(numerator / denominator)

        normalized = rms_norm(pooled, norm_weight, 1.0e-6)
        output = rope_tail(
            normalized,
            1,
            self.head_dim,
            position + 1 - self.ratio,
            int(rope["rotary_dim"]),
            float(rope["theta"]),
            float(rope["scaling_factor"]),
            int(rope["original_context_length"]),
            float(rope["beta_fast"]),
            float(rope["beta_slow"]),
        )
        if self.ratio == 4:
            current_kv = self.kv[self.ratio :].copy()
            current_scores = self.scores[self.ratio :].copy()
            self.kv[: self.ratio] = current_kv
            self.kv[self.ratio :] = current_kv
            self.scores[: self.ratio] = current_scores
            self.scores[self.ratio :] = current_scores
        return output


def ratio4_fixture() -> dict[str, object]:
    compressor = Compressor(4, 4)
    ape = np.asarray(
        [
            ((position * 8 + index) % 7 - 3.0) * 0.04
            for position in range(4)
            for index in range(8)
        ],
        dtype=np.float32,
    ).reshape(4, 8)
    norm = np.asarray([1.0, 0.75, 1.25, 0.5], dtype=np.float32)
    rope = {
        "rotary_dim": 4,
        "theta": 160_000.0,
        "scaling_factor": 16.0,
        "original_context_length": 65_536,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }
    projected_kv = []
    projected_scores = []
    emitted = []
    snapshots = []
    for position in range(9):
        kv = np.asarray(
            [((position + 1) * (index + 2) % 17 - 8.0) * 0.2 for index in range(8)],
            dtype=np.float32,
        )
        scores = np.asarray(
            [((position * 5 + index * 3) % 11 - 5.0) * 0.13 for index in range(8)],
            dtype=np.float32,
        )
        projected_kv.append(as_json(kv))
        projected_scores.append(as_json(scores))
        row = compressor.push(position, kv, scores, ape, norm, rope)
        if row is not None:
            emitted.append({"position": position, "value": as_json(row)})
        snapshots.append(
            {
                "position": position,
                "kv": as_json(compressor.kv.reshape(-1)),
                "scores": as_json(compressor.scores.reshape(-1)),
            }
        )
    return {
        "head_dim": 4,
        "ape": as_json(ape.reshape(-1)),
        "norm_weight": as_json(norm),
        "rope": rope,
        "projected_kv": projected_kv,
        "projected_scores": projected_scores,
        "emitted": emitted,
        "snapshots": snapshots,
    }


def ratio128_fixture() -> dict[str, object]:
    compressor = Compressor(128, 2)
    ape = np.asarray(
        [
            ((position * 2 + index) % 9 - 4.0) * 0.015
            for position in range(128)
            for index in range(2)
        ],
        dtype=np.float32,
    ).reshape(128, 2)
    norm = np.asarray([1.0, 0.625], dtype=np.float32)
    rope = {
        "rotary_dim": 2,
        "theta": 160_000.0,
        "scaling_factor": 16.0,
        "original_context_length": 65_536,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }
    projected_kv = []
    projected_scores = []
    emitted = []
    snapshots = []
    snapshot_positions = {126, 127, 128, 254, 255, 256}
    for position in range(257):
        kv = np.asarray(
            [math.sin(position * 0.07), math.cos(position * 0.11)],
            dtype=np.float32,
        )
        scores = np.asarray(
            [((position * 3) % 17 - 8.0) * 0.08, ((position * 7) % 19 - 9.0) * 0.06],
            dtype=np.float32,
        )
        projected_kv.append(as_json(kv))
        projected_scores.append(as_json(scores))
        row = compressor.push(position, kv, scores, ape, norm, rope)
        if row is not None:
            emitted.append({"position": position, "value": as_json(row)})
        if position in snapshot_positions:
            snapshots.append(
                {
                    "position": position,
                    "kv": as_json(compressor.kv.reshape(-1)),
                    "scores": as_json(compressor.scores.reshape(-1)),
                }
            )
    return {
        "head_dim": 2,
        "ape": as_json(ape.reshape(-1)),
        "norm_weight": as_json(norm),
        "rope": rope,
        "projected_kv": projected_kv,
        "projected_scores": projected_scores,
        "emitted": emitted,
        "snapshots": snapshots,
        "expected_kv_state": as_json(compressor.kv.reshape(-1)),
        "expected_score_state": as_json(compressor.scores.reshape(-1)),
    }


E2M1 = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def nearest_even_code(value: float, codebook: np.ndarray) -> np.float32:
    absolute = min(abs(value), float(codebook[-1]))
    best = 0
    best_difference = abs(absolute - float(codebook[0]))
    for index in range(1, len(codebook)):
        difference = abs(absolute - float(codebook[index]))
        if difference < best_difference or (
            difference == best_difference and index % 2 == 0 and best % 2 != 0
        ):
            best = index
            best_difference = difference
    sign = -1.0 if value < 0.0 else 1.0
    return f(sign * float(codebook[best]))


def hadamard_128(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    stride = 1
    while stride < 128:
        for base in range(0, 128, 2 * stride):
            for index in range(stride):
                first = output[base + index]
                second = output[base + stride + index]
                output[base + index] = f(first + second)
                output[base + stride + index] = f(first - second)
        stride *= 2
    scale = f(1.0 / math.sqrt(128.0))
    for index in range(128):
        output[index] = f(output[index] * scale)
    return output


def fp4_roundtrip(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    for offset in range(0, len(output), 32):
        block = output[offset : offset + 32]
        maximum = max(max(abs(float(value)) for value in block), 7.052966104933725e-38)
        scale = f(2.0 ** math.ceil(math.log2(maximum / 6.0)))
        for index in range(32):
            normalized = min(6.0, max(-6.0, float(block[index] / scale)))
            block[index] = f(nearest_even_code(normalized, E2M1) * scale)
    return output


def bf16_roundtrip_scalar(value: float | np.floating) -> np.float32:
    bits = struct.unpack("<I", struct.pack("<f", float(F(value))))[0]
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = (bits + rounding_bias) & 0xFFFF0000
    return f(struct.unpack("<f", struct.pack("<I", rounded))[0])


def e4m3fn_value(index: int) -> np.float32:
    exponent_scale = [
        0.0,
        0.015625,
        0.03125,
        0.0625,
        0.125,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        256.0,
    ]
    exponent = (index >> 3) & 0x0F
    mantissa = index & 0x07
    if exponent == 0:
        return f(mantissa * 0.001953125)
    return f((1.0 + mantissa * 0.125) * exponent_scale[exponent])


def e4m3fn_roundtrip(value: float | np.floating) -> np.float32:
    value = float(value)
    sign = -1.0 if value < 0.0 else 1.0
    absolute = min(abs(value), 448.0)
    low = 0
    high = 126
    while low < high:
        middle = (low + high + 1) // 2
        if float(e4m3fn_value(middle)) <= absolute:
            low = middle
        else:
            high = middle - 1
    best = low
    if best < 126:
        current_difference = abs(absolute - float(e4m3fn_value(best)))
        next_difference = abs(absolute - float(e4m3fn_value(best + 1)))
        if next_difference < current_difference or (
            next_difference == current_difference
            and (best + 1) % 2 == 0
            and best % 2 != 0
        ):
            best += 1
    return f(sign * float(e4m3fn_value(best)))


def cache_roundtrip_fixture() -> dict[str, object]:
    rotary_dim = 64
    values = np.asarray(
        [
            math.sin(index * 0.071) * 3.25 + math.cos(index * 0.019) * 0.2
            for index in range(128)
        ],
        dtype=np.float32,
    )
    expected = values.copy()
    for offset in range(0, len(expected) - rotary_dim, 64):
        block = expected[offset : offset + 64]
        for index in range(64):
            block[index] = bf16_roundtrip_scalar(block[index])
        maximum = max(max(abs(float(value)) for value in block), 1.0e-4)
        scale = f(2.0 ** math.ceil(math.log2(maximum / 448.0)))
        for index in range(64):
            normalized = min(448.0, max(-448.0, float(block[index] / scale)))
            block[index] = f(e4m3fn_roundtrip(normalized) * scale)
    for index in range(len(expected) - rotary_dim, len(expected)):
        expected[index] = bf16_roundtrip_scalar(expected[index])
    bf16_values = np.asarray(
        [0.0, 1.00390625, -2.0078125, 65_504.0, 1.0e-7], dtype=np.float32
    )
    expected_bf16 = np.asarray(
        [bf16_roundtrip_scalar(value) for value in bf16_values], dtype=np.float32
    )
    return {
        "rotary_dim": rotary_dim,
        "input": as_json(values),
        "expected_attention_cache": as_json(expected),
        "bf16_input": as_json(bf16_values),
        "expected_bf16": as_json(expected_bf16),
    }


def indexer_fixture() -> dict[str, object]:
    qat_input = np.asarray(
        [
            math.sin(index * 0.17) + math.cos(index * 0.031) * 0.25
            for index in range(128)
        ],
        dtype=np.float32,
    )
    qat_output = fp4_roundtrip(hadamard_128(qat_input))
    head_count = 3
    head_dim = 4
    queries = np.asarray(
        [((index * 7) % 13 - 6.0) * 0.12 for index in range(head_count * head_dim)],
        dtype=np.float32,
    )
    weights = np.asarray([0.8, -0.35, 1.1], dtype=np.float32)
    keys = np.asarray(
        [
            ((row * 11 + index * 5) % 17 - 8.0) * 0.09
            for row in range(6)
            for index in range(head_dim)
        ],
        dtype=np.float32,
    ).reshape(6, head_dim)
    scale = f(1.0 / math.sqrt(head_count * head_dim))
    scores = np.zeros(6, dtype=np.float32)
    for row in range(6):
        for head in range(head_count):
            head_dot = max(
                0.0,
                float(dot(queries[head * head_dim : (head + 1) * head_dim], keys[row])),
            )
            scores[row] = f(scores[row] + f(f(head_dot * weights[head]) * scale))
    topk = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[
        :3
    ]
    return {
        "qat_input": as_json(qat_input),
        "expected_qat": as_json(qat_output),
        "head_count": head_count,
        "head_dim": head_dim,
        "queries": as_json(queries),
        "head_weights": as_json(weights),
        "compressed_keys": as_json(keys.reshape(-1)),
        "expected_scores": as_json(scores),
        "top_k": 3,
        "expected_top_k": topk,
    }


def routing_fixture() -> dict[str, object]:
    logits = np.asarray([-4.0, -0.5, 0.0, 1.25, 3.0, 0.0, -2.5, 2.0], dtype=np.float32)
    scores = np.asarray(
        [f(math.sqrt(float(softplus(value)))) for value in logits], dtype=np.float32
    )
    bias = np.asarray([0.0, 0.25, 0.0, -0.1, -3.0, 0.0, 2.5, 0.2], dtype=np.float32)
    selected = sorted(
        range(len(scores)),
        key=lambda index: (-float(f(scores[index] + bias[index])), index),
    )[:3]
    weights = np.asarray([scores[index] for index in selected], dtype=np.float32)
    denominator = max(float(np.sum(weights, dtype=np.float32)), 6.103515625e-5)
    weights = np.asarray(
        [f(f(weight / denominator) * f(1.5)) for weight in weights], dtype=np.float32
    )
    hash_ids = [5, 2, 7]
    hash_weights = np.asarray([scores[index] for index in hash_ids], dtype=np.float32)
    denominator = max(float(np.sum(hash_weights, dtype=np.float32)), 6.103515625e-5)
    hash_weights = np.asarray(
        [f(f(weight / denominator) * f(1.5)) for weight in hash_weights],
        dtype=np.float32,
    )
    return {
        "logits": as_json(logits),
        "expected_scores": as_json(scores),
        "bias": as_json(bias),
        "top_k": 3,
        "routed_scale": 1.5,
        "expected_experts": selected,
        "expected_weights": as_json(weights),
        "hash_experts": hash_ids,
        "expected_hash_weights": as_json(hash_weights),
    }


def require_json_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return value


def reject_non_finite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def require_exact_keys(value: dict[str, object], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RuntimeError(
            f"{name} keys mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )


def require_shape(
    value: dict[str, object], key: str, expected: list[int], name: str
) -> None:
    actual = value.get(key)
    if (
        not isinstance(actual, list)
        or any(type(dimension) is not int for dimension in actual)
        or actual != expected
    ):
        raise RuntimeError(f"{name}.{key} mismatch: expected {expected}, got {actual}")


def require_integer(
    value: dict[str, object], key: str, expected: int, name: str
) -> None:
    actual = value.get(key)
    if type(actual) is not int or actual != expected:
        raise RuntimeError(f"{name}.{key} mismatch: expected {expected}, got {actual}")


def require_string(
    value: dict[str, object], key: str, expected: str, name: str
) -> None:
    actual = value.get(key)
    if type(actual) is not str or actual != expected:
        raise RuntimeError(
            f"{name}.{key} mismatch: expected {expected!r}, got {actual!r}"
        )


def require_numeric_array(
    value: dict[str, object], key: str, expected_length: int, name: str
) -> list[int | float]:
    array = value.get(key)
    if not isinstance(array, list) or len(array) != expected_length:
        actual = len(array) if isinstance(array, list) else type(array).__name__
        raise RuntimeError(
            f"{name}.{key} length mismatch: expected {expected_length}, got {actual}"
        )
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in array
    ):
        raise RuntimeError(f"{name}.{key} must contain only finite numbers")
    return array


def validate_llama_cpp_vectors(value: object) -> dict[str, object]:
    vectors = require_json_object(value, "llama.cpp output")
    require_exact_keys(
        vectors,
        {
            "status",
            "backend",
            "thread_count",
            "tensor_layout",
            "hc_comb",
            "hc_pre",
            "hc_post",
        },
        "llama.cpp output",
    )
    require_string(vectors, "status", "success", "llama.cpp output")
    require_string(vectors, "backend", "ggml-cpu-reference", "llama.cpp output")
    require_integer(vectors, "thread_count", 1, "llama.cpp output")
    require_string(
        vectors,
        "tensor_layout",
        "flat GGML ne[0]-fastest",
        "llama.cpp output",
    )

    combination = require_json_object(vectors["hc_comb"], "llama.cpp hc_comb")
    require_exact_keys(
        combination,
        {
            "connection_count",
            "token_count",
            "mixes_shape",
            "mixes",
            "scale_shape",
            "scale",
            "base_shape",
            "base",
            "epsilon",
            "iterations",
            "output_shape",
            "output",
        },
        "llama.cpp hc_comb",
    )
    require_integer(combination, "connection_count", 4, "llama.cpp hc_comb")
    require_integer(combination, "token_count", 3, "llama.cpp hc_comb")
    require_integer(combination, "iterations", 20, "llama.cpp hc_comb")
    epsilon = combination["epsilon"]
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or not math.isclose(epsilon, 1.0e-6, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise RuntimeError("llama.cpp hc_comb controls changed")
    require_shape(combination, "mixes_shape", [24, 3], "llama.cpp hc_comb")
    require_numeric_array(combination, "mixes", 72, "llama.cpp hc_comb")
    require_shape(combination, "scale_shape", [3], "llama.cpp hc_comb")
    require_numeric_array(combination, "scale", 3, "llama.cpp hc_comb")
    require_shape(combination, "base_shape", [24], "llama.cpp hc_comb")
    require_numeric_array(combination, "base", 24, "llama.cpp hc_comb")
    require_shape(combination, "output_shape", [4, 4, 3], "llama.cpp hc_comb")
    combination_output = require_numeric_array(
        combination, "output", 48, "llama.cpp hc_comb"
    )

    pre = require_json_object(vectors["hc_pre"], "llama.cpp hc_pre")
    require_exact_keys(
        pre,
        {
            "hidden_size",
            "connection_count",
            "token_count",
            "residual_shape",
            "residual",
            "weights_shape",
            "weights",
            "output_shape",
            "output",
        },
        "llama.cpp hc_pre",
    )
    require_integer(pre, "hidden_size", 7, "llama.cpp hc_pre")
    require_integer(pre, "connection_count", 4, "llama.cpp hc_pre")
    require_integer(pre, "token_count", 3, "llama.cpp hc_pre")
    require_shape(pre, "residual_shape", [7, 4, 3], "llama.cpp hc_pre")
    pre_residual = require_numeric_array(pre, "residual", 84, "llama.cpp hc_pre")
    require_shape(pre, "weights_shape", [4, 3], "llama.cpp hc_pre")
    require_numeric_array(pre, "weights", 12, "llama.cpp hc_pre")
    require_shape(pre, "output_shape", [7, 3], "llama.cpp hc_pre")
    require_numeric_array(pre, "output", 21, "llama.cpp hc_pre")

    post = require_json_object(vectors["hc_post"], "llama.cpp hc_post")
    require_exact_keys(
        post,
        {
            "hidden_size",
            "connection_count",
            "token_count",
            "block_output_shape",
            "block_output",
            "residual_shape",
            "residual",
            "weights_shape",
            "weights",
            "combination_shape",
            "combination",
            "output_shape",
            "output",
        },
        "llama.cpp hc_post",
    )
    require_integer(post, "hidden_size", 7, "llama.cpp hc_post")
    require_integer(post, "connection_count", 4, "llama.cpp hc_post")
    require_integer(post, "token_count", 3, "llama.cpp hc_post")
    require_shape(post, "block_output_shape", [7, 3], "llama.cpp hc_post")
    require_numeric_array(post, "block_output", 21, "llama.cpp hc_post")
    require_shape(post, "residual_shape", [7, 4, 3], "llama.cpp hc_post")
    post_residual = require_numeric_array(post, "residual", 84, "llama.cpp hc_post")
    require_shape(post, "weights_shape", [4, 3], "llama.cpp hc_post")
    require_numeric_array(post, "weights", 12, "llama.cpp hc_post")
    require_shape(post, "combination_shape", [4, 4, 3], "llama.cpp hc_post")
    post_combination = require_numeric_array(
        post, "combination", 48, "llama.cpp hc_post"
    )
    require_shape(post, "output_shape", [7, 4, 3], "llama.cpp hc_post")
    require_numeric_array(post, "output", 84, "llama.cpp hc_post")
    if post_residual != pre_residual or post_combination != combination_output:
        raise RuntimeError(
            "llama.cpp fused primitive inputs are not internally coherent"
        )
    return vectors


def llama_cpp_cpu_fixture() -> dict[str, object]:
    llama_cpp = Path(
        os.environ.get("DSV4_LLAMA_CPP_DIR", Path.home() / "code" / "llama.cpp")
    ).resolve()
    if not (llama_cpp / "ggml" / "CMakeLists.txt").is_file():
        raise RuntimeError(
            f"llama.cpp checkout not found at {llama_cpp}; set DSV4_LLAMA_CPP_DIR"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=llama_cpp,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != LLAMA_CPP_REVISION:
        raise RuntimeError(
            f"llama.cpp revision mismatch: expected {LLAMA_CPP_REVISION}, got {revision}"
        )
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=llama_cpp,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tree != LLAMA_CPP_TREE:
        raise RuntimeError(
            f"llama.cpp tree mismatch: expected {LLAMA_CPP_TREE}, got {tree}"
        )
    tracked_changes = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=llama_cpp,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_changes:
        raise RuntimeError(
            "llama.cpp has tracked worktree changes; direct vectors require a clean "
            f"pinned checkout:\n{tracked_changes}"
        )

    source_files = []
    for relative, expected_sha256 in LLAMA_CPP_SOURCE_SHA256.items():
        source = llama_cpp / relative
        actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"llama.cpp {relative} content mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        source_files.append({"path": relative, "sha256": actual_sha256})

    reference_dir = Path(__file__).parent
    harness = reference_dir / "dsv4_llama_cpp_cpu_oracle.cpp"
    build_project = reference_dir / "CMakeLists.txt"
    cmake = os.environ.get("CMAKE", "cmake")
    with tempfile.TemporaryDirectory(prefix="dsv4-llama-cpp-oracle-") as temporary:
        build = Path(temporary) / "build"
        commands = [
            [
                cmake,
                "-S",
                str(reference_dir),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DLLAMA_CPP_DIR={llama_cpp}",
            ],
            [
                cmake,
                "--build",
                str(build),
                "--target",
                "dsv4_llama_cpp_cpu_oracle",
                "--parallel",
                "--config",
                "Release",
            ],
        ]
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"llama.cpp oracle command failed: {command}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
        toolchain_file = build / "dsv4-reference-toolchain.txt"
        toolchain = {}
        for line in toolchain_file.read_text().splitlines():
            key, separator, value = line.partition("=")
            if not separator or not key or key in toolchain:
                raise RuntimeError(
                    f"invalid llama.cpp toolchain metadata line: {line!r}"
                )
            toolchain[key] = value
        expected_toolchain_keys = {
            "cmake_version",
            "generator",
            "c_compiler",
            "c_compiler_id",
            "c_compiler_target",
            "c_compiler_version",
            "c_flags",
            "c_flags_release",
            "cxx_compiler",
            "cxx_compiler_id",
            "cxx_compiler_target",
            "cxx_compiler_version",
            "cxx_flags",
            "cxx_flags_release",
            "exe_linker_flags",
            "exe_linker_flags_release",
            "osx_architectures",
            "osx_deployment_target",
            "osx_sysroot",
            "system_name",
            "system_processor",
            "system_version",
        }
        if set(toolchain) != expected_toolchain_keys or any(
            not value for value in toolchain.values()
        ):
            raise RuntimeError(f"invalid llama.cpp toolchain metadata: {toolchain}")
        executable = (
            build
            / "bin"
            / (
                "dsv4_llama_cpp_cpu_oracle.exe"
                if sys.platform == "win32"
                else "dsv4_llama_cpp_cpu_oracle"
            )
        )
        result = subprocess.run([str(executable)], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"llama.cpp oracle execution failed\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
    try:
        vectors = json.loads(result.stdout, parse_constant=reject_non_finite_json)
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid llama.cpp oracle JSON: {error}") from error

    return {
        "revision": revision,
        "tree": tree,
        "repository": "https://github.com/ggml-org/llama.cpp",
        "license": "MIT",
        "tracked_worktree_clean": True,
        "source_files": source_files,
        "harness_path": str(harness.relative_to(ROOT)),
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "build_project_path": str(build_project.relative_to(ROOT)),
        "build_project_sha256": hashlib.sha256(build_project.read_bytes()).hexdigest(),
        "build_toolchain": toolchain,
        "configuration": "static CPU-only GGML, one thread, reference compute plan",
        "symbols": [
            "ggml_dsv4_hc_comb",
            "ggml_dsv4_hc_pre",
            "ggml_dsv4_hc_post",
        ],
        "vectors": validate_llama_cpp_vectors(vectors),
    }


def dwarfstar_direct_fixture() -> dict[str, object]:
    dwarfstar = Path(
        os.environ.get("DSV4_DWARFSTAR_DIR", Path.home() / "code" / "ds4")
    ).resolve()
    source = dwarfstar / "ds4.c"
    if not source.is_file():
        raise RuntimeError(
            f"DwarfStar source not found at {source}; set DSV4_DWARFSTAR_DIR"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=dwarfstar,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != DWARFSTAR_REVISION:
        raise RuntimeError(
            f"DwarfStar revision mismatch: expected {DWARFSTAR_REVISION}, got {revision}"
        )
    tracked_changes = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=dwarfstar,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_changes:
        raise RuntimeError(
            "DwarfStar has tracked worktree changes; direct vectors require a clean "
            f"pinned checkout:\n{tracked_changes}"
        )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha256 != DWARFSTAR_SOURCE_SHA256:
        raise RuntimeError(
            "DwarfStar ds4.c content mismatch: "
            f"expected {DWARFSTAR_SOURCE_SHA256}, got {source_sha256}"
        )

    harness = Path(__file__).with_name("dsv4_dwarfstar_oracle.c")
    with tempfile.TemporaryDirectory(prefix="dsv4-dwarfstar-oracle-") as temporary:
        executable = Path(temporary) / "dsv4-dwarfstar-oracle"
        dead_strip = (
            "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections"
        )
        subprocess.run(
            [
                os.environ.get("CC", "clang"),
                "-std=c11",
                "-O0",
                "-ffunction-sections",
                "-fdata-sections",
                f'-DDSV4_DWARFSTAR_SOURCE="{source}"',
                f"-I{dwarfstar}",
                str(harness),
                "-o",
                str(executable),
                dead_strip,
                "-lm",
                "-lpthread",
            ],
            check=True,
        )
        output = subprocess.run(
            [str(executable)], check=True, capture_output=True, text=True
        ).stdout
    vectors = json.loads(output)
    return {
        "revision": revision,
        "repository": "https://github.com/antirez/ds4",
        "license": "MIT",
        "source_path": "ds4.c",
        "source_sha256": source_sha256,
        "tracked_worktree_clean": True,
        "symbols": [
            "hc_split_sinkhorn_one",
            "hc_post_one",
            "rope_tail_ext_inplace",
            "compressor_pool_decode_state",
            "dsv4_indexer_qat_row_inplace_cpu",
            "softplus_stable",
            "topk_desc",
            "swiglu",
        ],
        "vectors": vectors,
    }


def main() -> None:
    fixture = {
        "schema_version": 1,
        "generator_version": 4,
        "sources": {
            "vllm": "b40d859c7b07ae244bcd8c6eecdcdbd9a3afaa07",
            "sglang": "58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8",
            "llama_cpp": "876a4321163249c43ca4e986818fab5ab081f282",
            "dwarfstar": "54b36ed9ba42da31b24f2d1a5feb075c2475dbb1",
        },
        "transcription_provenance": [
            {
                "cases": [
                    "mhc",
                    "rope",
                    "shared_kv_projection",
                    "attention_output",
                ],
                "sources": [
                    "sglang/python/sglang/srt/models/deepseek_v4.py:1470",
                    "sglang/python/sglang/srt/models/deepseek_v4.py:726",
                    "llama.cpp/src/models/deepseek4.cpp:200",
                    "llama.cpp/src/models/deepseek4.cpp:831",
                    "ds4/ds4.c:9656",
                    "ds4/ds4.c:10070",
                ],
            },
            {
                "cases": ["compressor_ratio4", "compressor_ratio128"],
                "sources": [
                    "vllm/vllm/models/deepseek_v4/compressor.py:329",
                    "sglang/python/sglang/srt/layers/attention/dsv4/compressor.py:67",
                    "ds4/ds4.c:12390",
                ],
            },
            {
                "cases": ["indexer", "routing", "cache_roundtrip"],
                "sources": [
                    "vllm/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:49",
                    "vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:254",
                    "ds4/ds4.c:3211",
                ],
            },
        ],
        "known_reference_differences": {
            "fp4_tiny_scale_and_midpoints": (
                "The oracle follows vLLM MXFP4 and DwarfStar: UE8M0 power-of-two "
                "scale floor near 2^-126 and round-to-nearest-even E2M1. SGLang "
                "fp4_indexer.py floors the pre-rounded scale at 1e-4 and resolves "
                "exact E2M1 midpoints toward the lower code."
            )
        },
        "mhc": mhc_fixture(),
        "rope": rope_fixture(),
        "shared_kv_projection": shared_kv_projection_fixture(),
        "attention_output": attention_output_fixture(),
        "compressor_ratio4": ratio4_fixture(),
        "compressor_ratio128": ratio128_fixture(),
        "indexer": indexer_fixture(),
        "routing": routing_fixture(),
        "cache_roundtrip": cache_roundtrip_fixture(),
        "llama_cpp_cpu": llama_cpp_cpu_fixture(),
        "dwarfstar_direct": dwarfstar_direct_fixture(),
    }
    serialized = json.dumps(fixture, indent=2, sort_keys=True) + "\n"
    arguments = sys.argv[1:]
    if arguments == ["--check"]:
        if not OUTPUT.is_file() or OUTPUT.read_text() != serialized:
            raise SystemExit(f"fixture drift: regenerate {OUTPUT.relative_to(ROOT)}")
        print(f"verified {OUTPUT.relative_to(ROOT)}")
    elif arguments:
        raise SystemExit("usage: generate_dsv4_oracle.py [--check]")
    else:
        OUTPUT.write_text(serialized)
        print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
