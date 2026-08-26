# /// script
# requires-python = "==3.11.*"
# dependencies = ["numpy==1.26.4", "torch==2.4.1"]
# ///

"""Generate the independent Qwen3.8 Flash-Next GDN Metal oracle.

The reference starts from grouped Hugging Face-style tensors, evaluates three
decode steps with nonzero causal state, then applies llama.cpp's converter
permutation to the expected state. The resulting fixture therefore tests the
GGUF tiled-head contract without sharing the Rust implementation's equations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
JSON_OUTPUT = ROOT / "crates/qwen-llm/tests/fixtures/qwen4exp_gdn_grouped_tiled_v1.json"
BINARY_OUTPUT = (
    ROOT / "crates/qwen-llm/tests/fixtures/qwen4exp_gdn_grouped_tiled_v1.f32"
)

SGLANG_REVISION = "73a255206f916366c8d26d4022f82ddfb0ab558d"
SGLANG_TREE = "dc134c86f21a7396d89bdb01a8019e8db81d763a"
SGLANG_SOURCE = (
    "python/sglang/test/kits/attention_unittest/attention_methods/gdn_attention.py"
)
SGLANG_SHA256 = "f7df7659b8e904d704ce04021e00a6fc20fdaf99f88e5828cbed09980bff786d"

LLAMA_CPP_REVISION = "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3"
LLAMA_CPP_TREE = "c946a1b3c9fbc8c0b1313eb77b48d34d8ce1884f"
LLAMA_CPP_SOURCES = {
    "conversion/qwen.py": (
        "56c65e7bc7817e624be3c5d9f6ca4a28e9e0ef89d4c6833d877bd86ecd798ecc"
    ),
    "src/models/qwen4exp.cpp": (
        "1afe4a3c677452680bf8ccbbb98573ee6a645d0729df74a15b0faad38bdf0bc7"
    ),
}

HIDDEN = 4
KEY_HEADS = 2
VALUE_HEADS = 4
HEAD_DIM = 128
CONV_KERNEL = 4
EPS = 1.0e-6
TOKENS = 3
KEY_WIDTH = KEY_HEADS * HEAD_DIM
VALUE_WIDTH = VALUE_HEADS * HEAD_DIM
CONV_WIDTH = 2 * KEY_WIDTH + VALUE_WIDTH
HEAD_PERMUTATION = [0, 2, 1, 3]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_source_tree(
    root: Path,
    revision: str,
    tree: str,
    files: dict[str, str],
) -> None:
    actual_revision = git(root, "rev-parse", "HEAD")
    actual_tree = git(root, "rev-parse", "HEAD^{tree}")
    if actual_revision != revision or actual_tree != tree:
        raise RuntimeError(
            f"source identity mismatch for {root}: "
            f"revision={actual_revision}, tree={actual_tree}"
        )
    for relative, expected in files.items():
        actual = sha256_path(root / relative)
        if actual != expected:
            raise RuntimeError(
                f"source digest mismatch for {root / relative}: {actual}"
            )


def formula2(
    rows: int,
    columns: int,
    multipliers: tuple[int, int],
    add: int,
    modulus: int,
    center: int,
    scale: float,
) -> torch.Tensor:
    row = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
    column = torch.arange(columns, dtype=torch.int64).unsqueeze(0)
    values = (row * multipliers[0] + column * multipliers[1] + add) % modulus
    return (values - center).to(torch.float32) * scale


def formula3(
    first: int,
    second: int,
    third: int,
    multipliers: tuple[int, int, int],
    add: int,
    modulus: int,
    center: int,
    scale: float,
) -> torch.Tensor:
    i = torch.arange(first, dtype=torch.int64).view(first, 1, 1)
    j = torch.arange(second, dtype=torch.int64).view(1, second, 1)
    k = torch.arange(third, dtype=torch.int64).view(1, 1, third)
    values = (
        i * multipliers[0] + j * multipliers[1] + k * multipliers[2] + add
    ) % modulus
    return (values - center).to(torch.float32) * scale


def grouped_inputs() -> dict[str, torch.Tensor]:
    qkv = formula2(CONV_WIDTH, HIDDEN, (17, 11), 3, 31, 15, 0.025)
    gate = formula2(VALUE_WIDTH, HIDDEN, (19, 7), 5, 37, 18, 0.030)
    beta = formula2(VALUE_HEADS, HIDDEN, (5, 9), 2, 23, 11, 0.080)
    alpha = formula2(VALUE_HEADS, HIDDEN, (7, 4), 1, 19, 9, 0.070)
    conv = formula2(CONV_WIDTH, CONV_KERNEL, (13, 7), 4, 29, 14, 0.018)
    norm = 0.8 + (torch.arange(HEAD_DIM) % 9).to(torch.float32) * 0.03
    output = formula2(HIDDEN, VALUE_WIDTH, (11, 23), 6, 41, 20, 0.012)
    conv_state = formula2(3, CONV_WIDTH, (29, 5), 8, 43, 21, 0.009)
    delta_state = formula3(
        VALUE_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        (31, 7, 13),
        10,
        47,
        23,
        0.0007,
    )
    return {
        "qkv": qkv,
        "gate": gate,
        "beta": beta,
        "alpha": alpha,
        "a": torch.tensor([-0.4, -0.7, -1.1, -1.6], dtype=torch.float32),
        "dt_bias": torch.tensor([-0.3, 0.1, 0.4, -0.2], dtype=torch.float32),
        "conv": conv,
        "norm": norm,
        "output": output,
        "conv_state": conv_state,
        "delta_state": delta_state,
        "inputs": torch.tensor(
            [
                [1.0, -0.5, 0.25, 2.0],
                [-0.75, 1.5, -1.0, 0.5],
                [0.2, -0.4, 1.2, -1.6],
            ],
            dtype=torch.float32,
        ),
    }


def l2_norm(values: torch.Tensor, mode: str) -> torch.Tensor:
    sum_squares = torch.sum(values * values, dim=-1, keepdim=True)
    if mode == "ggml_max":
        denominator = torch.maximum(
            torch.sqrt(sum_squares),
            torch.tensor(EPS, dtype=torch.float32),
        )
    elif mode == "sglang_add":
        denominator = torch.sqrt(sum_squares + EPS)
    else:
        raise ValueError(f"unknown L2 mode {mode}")
    return values / denominator


def l2_sentinel() -> dict[str, torch.Tensor | float]:
    query = torch.zeros((KEY_HEADS, HEAD_DIM), dtype=torch.float32)
    key = torch.zeros((KEY_HEADS, HEAD_DIM), dtype=torch.float32)
    query[0, 0] = 7.5e-7
    query[0, 17] = -2.5e-7
    query[1, 3] = 5.0e-4
    query[1, 91] = -1.0e-4
    key[0, 11] = -6.0e-7
    key[0, 73] = 3.0e-7
    key[1, 5] = -7.5e-4
    key[1, 127] = 2.5e-4
    query_ggml = l2_norm(query, "ggml_max")
    key_ggml = l2_norm(key, "ggml_max")
    query_additive = l2_norm(query, "sglang_add")
    key_additive = l2_norm(key, "sglang_add")
    difference = max(
        maximum_difference(query_ggml, query_additive),
        maximum_difference(key_ggml, key_additive),
    )
    return {
        "query": query,
        "key": key,
        "query_ggml": query_ggml,
        "key_ggml": key_ggml,
        "max_abs_additive_delta": difference,
    }


def run_grouped_reference(
    fixture: dict[str, torch.Tensor],
    *,
    l2_mode: str,
    key_mapping: str = "grouped",
    output_gate: str = "sigmoid",
    zero_conv_state: bool = False,
    zero_delta_state: bool = False,
) -> dict[str, torch.Tensor]:
    conv_state = fixture["conv_state"].clone()
    delta_state = fixture["delta_state"].clone()
    if zero_conv_state:
        conv_state.zero_()
    if zero_delta_state:
        delta_state.zero_()

    outputs = []
    recurrent_scaled = []
    for hidden in fixture["inputs"]:
        qkv = fixture["qkv"] @ hidden
        gate = fixture["gate"] @ hidden
        beta = torch.sigmoid(fixture["beta"] @ hidden)
        alpha = fixture["alpha"] @ hidden
        decay = torch.exp(
            torch.nn.functional.softplus(alpha + fixture["dt_bias"]) * fixture["a"]
        )

        history = torch.cat((conv_state, qkv.unsqueeze(0)), dim=0)
        convolved = torch.nn.functional.silu(
            torch.sum(fixture["conv"] * history.transpose(0, 1), dim=1)
        )
        conv_state = history[1:].clone()

        query = l2_norm(convolved[:KEY_WIDTH].reshape(KEY_HEADS, HEAD_DIM), l2_mode)
        key = l2_norm(
            convolved[KEY_WIDTH : 2 * KEY_WIDTH].reshape(KEY_HEADS, HEAD_DIM),
            l2_mode,
        )
        value = convolved[2 * KEY_WIDTH :].reshape(VALUE_HEADS, HEAD_DIM)

        current = torch.empty((VALUE_HEADS, HEAD_DIM), dtype=torch.float32)
        for value_head in range(VALUE_HEADS):
            if key_mapping == "grouped":
                key_head = value_head // (VALUE_HEADS // KEY_HEADS)
            elif key_mapping == "modulo":
                key_head = value_head % KEY_HEADS
            else:
                raise ValueError(f"unknown key mapping {key_mapping}")
            head_state = delta_state[value_head] * decay[value_head]
            prediction = torch.sum(head_state * key[key_head].unsqueeze(0), dim=1)
            residual = (value[value_head] - prediction) * beta[value_head]
            head_state = head_state + residual.unsqueeze(1) * key[key_head].unsqueeze(0)
            delta_state[value_head] = head_state
            current[value_head] = torch.sum(
                head_state * query[key_head].unsqueeze(0), dim=1
            ) / math.sqrt(HEAD_DIM)

        recurrent_scaled.append(current)
        mean_square = torch.mean(current * current, dim=1, keepdim=True)
        normalized = current * torch.rsqrt(mean_square + EPS)
        gate = gate.reshape(VALUE_HEADS, HEAD_DIM)
        if output_gate == "sigmoid":
            activated_gate = torch.sigmoid(gate)
        elif output_gate == "silu":
            activated_gate = torch.nn.functional.silu(gate)
        else:
            raise ValueError(f"unknown output gate {output_gate}")
        gated = normalized * fixture["norm"].unsqueeze(0) * activated_gate
        outputs.append(fixture["output"] @ gated.reshape(-1))

    return {
        "output": torch.stack(outputs),
        "recurrent_scaled": torch.stack(recurrent_scaled),
        "conv_state": conv_state,
        "delta_state": delta_state,
    }


def permute_value_heads(values: torch.Tensor, dimension: int) -> torch.Tensor:
    indices = torch.tensor(HEAD_PERMUTATION, dtype=torch.int64)
    return values.index_select(dimension, indices)


def tiled_expected(reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    recurrent = permute_value_heads(reference["recurrent_scaled"], 1)
    recurrent = recurrent * math.sqrt(HEAD_DIM)
    delta_state = permute_value_heads(reference["delta_state"], 0)

    conv_state = reference["conv_state"]
    qk = conv_state[:, : 2 * KEY_WIDTH]
    value = conv_state[:, 2 * KEY_WIDTH :].reshape(3, VALUE_HEADS, HEAD_DIM)
    value = permute_value_heads(value, 1).reshape(3, VALUE_WIDTH)
    return {
        "output": reference["output"],
        "recurrent_unscaled_tiled": recurrent,
        "delta_state_tiled": delta_state,
        "conv_state_tiled": torch.cat((qk, value), dim=1),
    }


def flattened_f32(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().to(torch.float32).reshape(-1)]


def maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item())


def write_binary(
    sections: list[tuple[str, torch.Tensor]],
) -> tuple[bytes, dict[str, Any]]:
    output = bytearray()
    manifest: dict[str, Any] = {}
    offset = 0
    for name, tensor in sections:
        values = flattened_f32(tensor)
        output.extend(struct.pack(f"<{len(values)}f", *values))
        manifest[name] = {
            "offset_f32": offset,
            "count_f32": len(values),
            "shape": list(tensor.shape),
        }
        offset += len(values)
    return bytes(output), manifest


def recipe_manifest() -> dict[str, Any]:
    return {
        "qkv": {
            "multipliers": [17, 11],
            "add": 3,
            "modulus": 31,
            "center": 15,
            "scale": 0.025,
        },
        "gate": {
            "multipliers": [19, 7],
            "add": 5,
            "modulus": 37,
            "center": 18,
            "scale": 0.030,
        },
        "beta": {
            "multipliers": [5, 9],
            "add": 2,
            "modulus": 23,
            "center": 11,
            "scale": 0.080,
        },
        "alpha": {
            "multipliers": [7, 4],
            "add": 1,
            "modulus": 19,
            "center": 9,
            "scale": 0.070,
        },
        "conv": {
            "multipliers": [13, 7],
            "add": 4,
            "modulus": 29,
            "center": 14,
            "scale": 0.018,
        },
        "output": {
            "multipliers": [11, 23],
            "add": 6,
            "modulus": 41,
            "center": 20,
            "scale": 0.012,
        },
        "conv_state": {
            "multipliers": [29, 5],
            "add": 8,
            "modulus": 43,
            "center": 21,
            "scale": 0.009,
        },
        "delta_state": {
            "multipliers": [31, 7, 13],
            "add": 10,
            "modulus": 47,
            "center": 23,
            "scale": 0.0007,
        },
        "norm": {"base": 0.8, "step": 0.03, "modulus": 9},
        "transformed_a": [-0.4, -0.7, -1.1, -1.6],
        "dt_bias": [-0.3, 0.1, 0.4, -0.2],
        "inputs": [
            [1.0, -0.5, 0.25, 2.0],
            [-0.75, 1.5, -1.0, 0.5],
            [0.2, -0.4, 1.2, -1.6],
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-root", type=Path)
    parser.add_argument("--llama-cpp-root", type=Path)
    parser.add_argument("--require-sources", action="store_true")
    args = parser.parse_args()

    if args.require_sources and (
        args.sglang_root is None or args.llama_cpp_root is None
    ):
        parser.error("--require-sources needs both source roots")
    if args.sglang_root is not None:
        verify_source_tree(
            args.sglang_root,
            SGLANG_REVISION,
            SGLANG_TREE,
            {SGLANG_SOURCE: SGLANG_SHA256},
        )
    if args.llama_cpp_root is not None:
        verify_source_tree(
            args.llama_cpp_root,
            LLAMA_CPP_REVISION,
            LLAMA_CPP_TREE,
            LLAMA_CPP_SOURCES,
        )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    inputs = grouped_inputs()
    ggml_reference = run_grouped_reference(inputs, l2_mode="ggml_max")
    sglang_reference = run_grouped_reference(inputs, l2_mode="sglang_add")
    expected = tiled_expected(ggml_reference)
    l2 = l2_sentinel()

    wrong_mapping = run_grouped_reference(
        inputs,
        l2_mode="ggml_max",
        key_mapping="modulo",
    )
    silu_gate = run_grouped_reference(
        inputs,
        l2_mode="ggml_max",
        output_gate="silu",
    )
    zero_conv = run_grouped_reference(
        inputs,
        l2_mode="ggml_max",
        zero_conv_state=True,
    )
    zero_delta = run_grouped_reference(
        inputs,
        l2_mode="ggml_max",
        zero_delta_state=True,
    )

    binary, section_manifest = write_binary(
        [
            ("output", expected["output"]),
            ("recurrent_unscaled_tiled", expected["recurrent_unscaled_tiled"]),
            ("delta_state_tiled", expected["delta_state_tiled"]),
            ("conv_state_tiled", expected["conv_state_tiled"]),
            ("l2_query", l2["query"]),
            ("l2_key", l2["key"]),
            ("l2_query_ggml", l2["query_ggml"]),
            ("l2_key_ggml", l2["key_ggml"]),
        ]
    )
    BINARY_OUTPUT.write_bytes(binary)

    manifest = {
        "schema_version": 1,
        "generator_version": 1,
        "description": (
            "Three-token grouped-checkpoint GDN decode with nonzero convolution "
            "and recurrent state, converted to GGUF tiled value-head order."
        ),
        "sources": {
            "sglang": {
                "repository": "https://github.com/sgl-project/sglang",
                "revision": SGLANG_REVISION,
                "tree": SGLANG_TREE,
                "path": SGLANG_SOURCE,
                "sha256": SGLANG_SHA256,
                "contract": "grouped-head GDN recurrence and query scaling",
            },
            "llama_cpp": {
                "repository": "https://github.com/ggml-org/llama.cpp",
                "revision": LLAMA_CPP_REVISION,
                "tree": LLAMA_CPP_TREE,
                "files": [
                    {"path": path, "sha256": digest}
                    for path, digest in LLAMA_CPP_SOURCES.items()
                ],
                "contract": (
                    "grouped-to-tiled value-head conversion, ggml L2 norm, "
                    "and Qwen3.8 sigmoid output gate"
                ),
            },
        },
        "toolchain": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": 1,
        },
        "geometry": {
            "hidden_size": HIDDEN,
            "key_heads": KEY_HEADS,
            "value_heads": VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "conv_kernel": CONV_KERNEL,
            "eps": EPS,
            "tokens": TOKENS,
        },
        "layout": {
            "checkpoint_value_heads": ["k0v0", "k0v1", "k1v0", "k1v1"],
            "gguf_value_heads": ["k0v0", "k1v0", "k0v1", "k1v1"],
            "gguf_from_checkpoint": HEAD_PERMUTATION,
            "gguf_key_head_rule": "value_head % key_heads",
            "checkpoint_key_head_rule": "value_head // values_per_key",
        },
        "semantics": {
            "transformed_a": "-exp(A_log), supplied after checkpoint transform",
            "convolution_state_order": "oldest_to_newest_raw_projection",
            "l2_norm": "x / max(sqrt(sum(x*x)), eps)",
            "query_scale": "1/sqrt(head_dim), folded into Metal gated-RMS epsilon",
            "output_gate": "sigmoid",
        },
        "l2_sentinel": {
            "description": (
                "Near-epsilon head vectors distinguish ggml's denominator clamp "
                "from additive-epsilon normalization."
            ),
            "max_abs_additive_delta": l2["max_abs_additive_delta"],
        },
        "recipe": recipe_manifest(),
        "binary": {
            "file": BINARY_OUTPUT.name,
            "dtype": "f32",
            "byte_order": "little",
            "sha256": sha256_bytes(binary),
            "sections": section_manifest,
        },
        "cross_reference": {
            "sglang_additive_l2_output": flattened_f32(sglang_reference["output"]),
            "max_abs_output_delta_from_ggml_l2": maximum_difference(
                sglang_reference["output"], ggml_reference["output"]
            ),
        },
        "fault_sensitivity_max_abs_output": {
            "grouped_modulo_mapping": maximum_difference(
                wrong_mapping["output"], ggml_reference["output"]
            ),
            "silu_output_gate": maximum_difference(
                silu_gate["output"], ggml_reference["output"]
            ),
            "zero_initial_convolution_state": maximum_difference(
                zero_conv["output"], ggml_reference["output"]
            ),
            "zero_initial_recurrent_state": maximum_difference(
                zero_delta["output"], ggml_reference["output"]
            ),
        },
    }
    JSON_OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {JSON_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {BINARY_OUTPUT.relative_to(ROOT)} ({len(binary)} bytes)")
    print(f"binary sha256 {sha256_bytes(binary)}")


if __name__ == "__main__":
    main()
