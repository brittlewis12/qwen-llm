# /// script
# requires-python = "==3.11.*"
# dependencies = ["numpy==1.26.4", "torch==2.4.1"]
# ///

"""Generate an independent PyTorch oracle for text-only Qwen4Exp QSA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
JSON_OUTPUT = ROOT / "crates/qwen-llm/tests/fixtures/qwen4exp_qsa_text_f16_v1.json"
BINARY_OUTPUT = ROOT / "crates/qwen-llm/tests/fixtures/qwen4exp_qsa_text_f16_v1.f32"

VLLM_REVISION = "02f2b4c15dd987d9436e125aab29604447c77405"
VLLM_TREE = "72eaeeeb9bfff19494a9d19ee85a6a83745d0602"
VLLM_FILES = {
    "vllm/models/qwen4_exp/nvidia/indexer_qsa.py": "668706c3a59c51e2c1ed51d19bd9a0e1564a0aad68c5e2bc20d5fb9e65cc2f98",
    "vllm/models/qwen4_exp/nvidia/ops/qsa.py": "faa8d358c79745f304edd363e4da21992e4cf015a22316b14980500bd199a0ad",
    "tests/models/qwen4_exp/test_qsa_reference.py": "7396d4482c2e7a0529bd927b2922925a709916910b651b2ed5da790ec2d385f1",
}
SGLANG_REVISION = "73a255206f916366c8d26d4022f82ddfb0ab558d"
SGLANG_TREE = "dc134c86f21a7396d89bdb01a8019e8db81d763a"
SGLANG_FILES = {
    "python/sglang/srt/layers/attention/qsa/qsa_indexer.py": "bb57ce1e9abc4fbfcba2c9aaaf125b9e625983966497df57165b6d4c6461afe2",
    "python/sglang/srt/layers/attention/qsa/kernel.py": "5482e38d30bfaf1624ec0625b4896cbb395a1637f75c183c8ca723c9f6055ff8",
    "python/sglang/srt/layers/attention/qsa/mqa.py": "af36d5c8f4fbda5b0e82b7f31046a95c9a709fcc57b3600c6473c49e87b7629f",
    "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py": "c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2",
}

HIDDEN = 16
INDEX_HEADS = 4
INDEX_DIM = 128
QUERY_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64
RATIO = 4
TOKEN_BUDGET = 8
CAPACITY = 16
TOKENS = 13
THETA = 10_000_000.0
EPS = 1.0e-6
OUTPUT_WIDTH = TOKEN_BUDGET + RATIO - 1

RECIPES: dict[str, dict[str, Any]] = {
    "index_query": {
        "multipliers": [37, 11],
        "add": 7,
        "modulus": 67,
        "center": 33,
        "scale": 0.006,
    },
    "index_key": {
        "multipliers": [29, 17],
        "add": 3,
        "modulus": 71,
        "center": 35,
        "scale": 0.007,
    },
    "query_gate": {
        "multipliers": [43, 13],
        "add": 5,
        "modulus": 79,
        "center": 39,
        "scale": 0.005,
    },
    "key": {
        "multipliers": [31, 19],
        "add": 9,
        "modulus": 73,
        "center": 36,
        "scale": 0.006,
    },
    "value": {
        "multipliers": [47, 7],
        "add": 11,
        "modulus": 83,
        "center": 41,
        "scale": 0.012,
    },
    "output": {
        "multipliers": [23, 41],
        "add": 13,
        "modulus": 89,
        "center": 44,
        "scale": 0.004,
    },
    "inputs": {
        "multipliers": [17, 31],
        "add": 43,
        "modulus": 59,
        "center": 29,
        "scale": 0.035,
    },
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def verify_source(root: Path, revision: str, tree: str, files: dict[str, str]) -> None:
    actual_revision = git(root, "rev-parse", "HEAD")
    actual_tree = git(root, "rev-parse", "HEAD^{tree}")
    if (actual_revision, actual_tree) != (revision, tree):
        raise RuntimeError(
            f"source identity mismatch for {root}: revision={actual_revision}, tree={actual_tree}"
        )
    for relative, expected in files.items():
        actual = sha256((root / relative).read_bytes())
        if actual != expected:
            raise RuntimeError(f"source digest mismatch for {relative}: {actual}")


def formula(rows: int, columns: int, recipe: dict[str, Any]) -> torch.Tensor:
    row = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
    column = torch.arange(columns, dtype=torch.int64).unsqueeze(0)
    raw = (
        row * recipe["multipliers"][0]
        + column * recipe["multipliers"][1]
        + recipe["add"]
    ) % recipe["modulus"]
    return (raw - recipe["center"]).to(torch.float32) * recipe["scale"]


def norm_recipe(width: int, base: float, step: float, modulus: int) -> torch.Tensor:
    return base + (torch.arange(width) % modulus).to(torch.float32) * step


def rms_rope(rows: torch.Tensor, weight: torch.Tensor, position: int) -> torch.Tensor:
    normalized = rows * torch.rsqrt(torch.mean(rows * rows, dim=-1, keepdim=True) + EPS)
    weighted = normalized * weight
    left = weighted[..., : ROTARY_DIM // 2]
    right = weighted[..., ROTARY_DIM // 2 : ROTARY_DIM]
    pairs = torch.arange(ROTARY_DIM // 2, dtype=torch.float32)
    angles = position * torch.pow(torch.tensor(THETA), -2.0 * pairs / ROTARY_DIM)
    rotated = torch.cat(
        (
            left * torch.cos(angles) - right * torch.sin(angles),
            right * torch.cos(angles) + left * torch.sin(angles),
        ),
        dim=-1,
    )
    return torch.cat((rotated, weighted[..., ROTARY_DIM:]), dim=-1)


def round_f16(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float16).to(torch.float32)


def make_fixture() -> tuple[
    dict[str, torch.Tensor], list[dict[str, Any]], dict[str, float], dict[str, Any]
]:
    weights = {
        "index_query": formula(INDEX_HEADS * INDEX_DIM, HIDDEN, RECIPES["index_query"]),
        "index_key": formula(INDEX_DIM, HIDDEN, RECIPES["index_key"]),
        "query_gate": formula(
            2 * QUERY_HEADS * HEAD_DIM, HIDDEN, RECIPES["query_gate"]
        ),
        "key": formula(KV_HEADS * HEAD_DIM, HIDDEN, RECIPES["key"]),
        "value": formula(KV_HEADS * HEAD_DIM, HIDDEN, RECIPES["value"]),
        "output": formula(HIDDEN, QUERY_HEADS * HEAD_DIM, RECIPES["output"]),
    }
    inputs = formula(TOKENS, HIDDEN, RECIPES["inputs"])
    norms = {
        "index_query": norm_recipe(INDEX_DIM, 0.72, 0.031, 9),
        "index_key": norm_recipe(INDEX_DIM, 0.81, 0.027, 7),
        "query": norm_recipe(HEAD_DIM, 0.76, 0.021, 13),
        "key": norm_recipe(HEAD_DIM, 0.83, 0.019, 11),
    }
    pending = torch.zeros(RATIO, INDEX_DIM)
    compressed = torch.zeros(CAPACITY // RATIO, INDEX_DIM)
    key_cache = torch.zeros(CAPACITY, KV_HEADS, HEAD_DIM)
    value_cache = torch.zeros_like(key_cache)
    records: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "output",
            "attention",
            "index_query",
            "query",
            "raw_gate",
            "key",
            "value",
            "scores",
        )
    }
    decisions: list[dict[str, Any]] = []
    alternative_outputs = {"modulo": [], "silu": []}

    def attention_for(
        q: torch.Tensor,
        gate: torch.Tensor,
        ids: list[int],
        mapping: str,
        gate_kind: str,
    ) -> torch.Tensor:
        result = torch.empty_like(q)
        for qh in range(QUERY_HEADS):
            kvh = (
                qh // (QUERY_HEADS // KV_HEADS)
                if mapping == "grouped"
                else qh % KV_HEADS
            )
            keys = key_cache[ids, kvh]
            values = value_cache[ids, kvh]
            logits = torch.mv(keys, q[qh]) / math.sqrt(HEAD_DIM)
            attended = torch.sum(
                torch.softmax(logits, dim=0).unsqueeze(1) * values, dim=0
            )
            activated = torch.sigmoid(gate[qh])
            if gate_kind == "silu":
                activated = gate[qh] * activated
            result[qh] = attended * activated
        return result

    final_partial_fault_output = None
    for position, hidden in enumerate(inputs):
        length = position + 1
        index_q_raw = torch.mv(weights["index_query"], hidden).reshape(
            INDEX_HEADS, INDEX_DIM
        )
        index_q = rms_rope(index_q_raw, norms["index_query"], position)
        index_k_raw = torch.mv(weights["index_key"], hidden)
        pending[position % RATIO] = index_k_raw
        if length % RATIO == 0:
            block = length // RATIO - 1
            pooled = round_f16(torch.mean(pending, dim=0))
            compressed[block] = round_f16(
                rms_rope(pooled, norms["index_key"], block * RATIO)
            )

        visible = length // RATIO
        visible_scores = torch.empty(0)
        if visible:
            dots = torch.einsum("hd,bd->bh", index_q, compressed[:visible])
            visible_scores = torch.relu(dots).sum(dim=1) / math.sqrt(INDEX_DIM)
        selected_count = min(visible, TOKEN_BUDGET // RATIO)
        if selected_count:
            ranked = sorted(
                range(visible), key=lambda block: (-float(visible_scores[block]), block)
            )
            selected = sorted(ranked[:selected_count])
        else:
            selected = []
        ids = [
            token
            for block in selected
            for token in range(block * RATIO, (block + 1) * RATIO)
        ]
        ids.extend(range(visible * RATIO, length))

        qg = torch.mv(weights["query_gate"], hidden).reshape(QUERY_HEADS, 2, HEAD_DIM)
        q = rms_rope(qg[:, 0], norms["query"], position)
        raw_gate = qg[:, 1].clone()
        raw_k = torch.mv(weights["key"], hidden).reshape(KV_HEADS, HEAD_DIM)
        k = rms_rope(raw_k, norms["key"], position)
        value = torch.mv(weights["value"], hidden).reshape(KV_HEADS, HEAD_DIM)
        key_cache[position] = round_f16(k)
        value_cache[position] = round_f16(value)

        attention = attention_for(q, raw_gate, ids, "grouped", "sigmoid")
        output = torch.mv(weights["output"], attention.reshape(-1))
        alternative_outputs["modulo"].append(
            torch.mv(
                weights["output"],
                attention_for(q, raw_gate, ids, "modulo", "sigmoid").reshape(-1),
            )
        )
        alternative_outputs["silu"].append(
            torch.mv(
                weights["output"],
                attention_for(q, raw_gate, ids, "grouped", "silu").reshape(-1),
            )
        )
        padded_scores = torch.zeros(CAPACITY // RATIO)
        padded_scores[:visible] = visible_scores
        for name, tensor in (
            ("output", output),
            ("attention", attention),
            ("index_query", index_q),
            ("query", q),
            ("raw_gate", raw_gate),
            ("key", k),
            ("value", value),
            ("scores", padded_scores),
        ):
            records[name].append(tensor.clone())

        margin = None
        if visible > selected_count and selected_count:
            selected_scores = [float(visible_scores[b]) for b in selected]
            excluded_scores = [
                float(visible_scores[b]) for b in range(visible) if b not in selected
            ]
            margin = min(selected_scores) - max(excluded_scores)
        decisions.append(
            {
                "position": position,
                "length": length,
                "visible_blocks": visible,
                "scores": [float(x) for x in visible_scores],
                "selected_blocks": selected,
                "token_ids": ids + [-1] * (OUTPUT_WIDTH - len(ids)),
                "selection_margin": margin,
                "newly_completed_third_block_selected": bool(
                    length >= 12 and 2 in selected
                ),
            }
        )

        if length == 13:
            # llama.cpp ranks duplicated per-token block scores at a fixed
            # token_budget + ratio - 1 width. With a one-token tail, that
            # admits two members of the highest excluded complete block.
            ranked = sorted(
                range(visible),
                key=lambda block: (-float(visible_scores[block]), block),
            )
            selected_ranked = ranked[: TOKEN_BUDGET // RATIO]
            excluded_block = ranked[TOKEN_BUDGET // RATIO]
            tail = list(range(visible * RATIO, length))
            extra_count = RATIO - 1 - len(tail)
            wrong_ids = [
                token
                for block in sorted(selected_ranked)
                for token in range(block * RATIO, (block + 1) * RATIO)
            ]
            extra_ids = list(
                range(
                    excluded_block * RATIO,
                    excluded_block * RATIO + extra_count,
                )
            )
            wrong_ids.extend(extra_ids)
            wrong_ids.extend(tail)
            wrong_attention = attention_for(
                q, raw_gate, wrong_ids, "grouped", "sigmoid"
            )
            final_partial_fault_output = torch.mv(
                weights["output"], wrong_attention.reshape(-1)
            )
            decisions[-1]["partial_extra_block"] = {
                "excluded_block": excluded_block,
                "extra_token_ids": extra_ids,
                "token_ids": wrong_ids,
            }

    stacked = {name: torch.stack(values) for name, values in records.items()}
    stacked["compressed_cache"] = compressed
    stacked["key_cache"] = key_cache
    stacked["value_cache"] = value_cache
    reference_output = stacked["output"]
    sensitivities = {
        "modulo_gqa_mapping": float(
            torch.max(
                torch.abs(torch.stack(alternative_outputs["modulo"]) - reference_output)
            )
        ),
        "silu_gate": float(
            torch.max(
                torch.abs(torch.stack(alternative_outputs["silu"]) - reference_output)
            )
        ),
        "partial_extra_block_tail_residue_1": float(
            torch.max(torch.abs(final_partial_fault_output - reference_output[-1]))
        ),
    }
    recipe = {
        "formula": "((row*m0 + column*m1 + add) % modulus - center) * scale",
        "weights_and_inputs": RECIPES,
        "norms": {
            "index_query": {"base": 0.72, "step": 0.031, "modulus": 9},
            "index_key": {"base": 0.81, "step": 0.027, "modulus": 7},
            "query": {"base": 0.76, "step": 0.021, "modulus": 13},
            "key": {"base": 0.83, "step": 0.019, "modulus": 11},
        },
    }
    return stacked, decisions, sensitivities, recipe


def write_binary(sections: dict[str, torch.Tensor]) -> tuple[bytes, dict[str, Any]]:
    data = bytearray()
    manifest = {}
    offset = 0
    for name, tensor in sections.items():
        values = tensor.detach().to(torch.float32).contiguous().reshape(-1).tolist()
        data.extend(struct.pack(f"<{len(values)}f", *values))
        manifest[name] = {
            "offset_f32": offset,
            "count_f32": len(values),
            "shape": list(tensor.shape),
        }
        offset += len(values)
    return bytes(data), manifest


def source_manifest(
    repository: str, revision: str, tree: str, files: dict[str, str]
) -> dict[str, Any]:
    return {
        "repository": repository,
        "revision": revision,
        "tree": tree,
        "files": [{"path": path, "sha256": digest} for path, digest in files.items()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--sglang-root", type=Path)
    parser.add_argument("--require-sources", action="store_true")
    args = parser.parse_args()
    if args.require_sources and (args.vllm_root is None or args.sglang_root is None):
        parser.error("--require-sources needs --vllm-root and --sglang-root")
    if args.vllm_root:
        verify_source(args.vllm_root, VLLM_REVISION, VLLM_TREE, VLLM_FILES)
    if args.sglang_root:
        verify_source(args.sglang_root, SGLANG_REVISION, SGLANG_TREE, SGLANG_FILES)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sections, decisions, sensitivities, recipe = make_fixture()
    if min(sensitivities.values()) <= 1.0e-2:
        raise RuntimeError(f"insufficient fault sensitivity: {sensitivities}")
    sparse_margins = [
        d["selection_margin"] for d in decisions if d["selection_margin"] is not None
    ]
    if not sparse_margins or min(sparse_margins) <= 1.0e-3:
        raise RuntimeError(f"insufficient sparse score margin: {sparse_margins}")
    binary, binary_sections = write_binary(sections)
    BINARY_OUTPUT.write_bytes(binary)
    manifest = {
        "schema": "qwen4exp-qsa-text-f16-oracle",
        "schema_version": 1,
        "generator_version": 1,
        "description": "Independent 13-step PyTorch QSA decode oracle for the Metal F16 cache checkpoint.",
        "sources": {
            "vllm": source_manifest(
                "https://github.com/vllm-project/vllm",
                VLLM_REVISION,
                VLLM_TREE,
                VLLM_FILES,
            ),
            "sglang": source_manifest(
                "https://github.com/sgl-project/sglang",
                SGLANG_REVISION,
                SGLANG_TREE,
                SGLANG_FILES,
            ),
        },
        "toolchain": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "threads": 1,
        },
        "geometry": {
            "hidden_size": HIDDEN,
            "index_query_heads": INDEX_HEADS,
            "index_head_dim": INDEX_DIM,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "rotary_dim": ROTARY_DIM,
            "compress_ratio": RATIO,
            "token_budget": TOKEN_BUDGET,
            "capacity": CAPACITY,
            "tokens": TOKENS,
            "theta": THETA,
            "eps": EPS,
        },
        "semantics": {
            "positions": "plain text NEOX, zero based; compressed block uses its first token position",
            "index_cache": "F32 raw pending; FP32 mean -> F16 -> F32 RMSNorm/RoPE -> F16 cache",
            "main_cache": "F32 normalized/rotated K and raw V -> F16 caches; Q and raw gates remain F32",
            "index_score": "sum_heads(relu(dot(q_head, compressed_key))) / sqrt(128)",
            "attention": "grouped GQA, softmax scale 1/sqrt(256), sigmoid gate after value reduction",
        },
        "recipe": recipe,
        "steps": decisions,
        "minimum_sparse_score_margin": min(sparse_margins),
        "fault_sensitivity_max_abs_output": sensitivities,
        "binary": {
            "file": BINARY_OUTPUT.name,
            "dtype": "f32",
            "byte_order": "little",
            "sha256": sha256(binary),
            "sections": binary_sections,
        },
    }
    JSON_OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {JSON_OUTPUT.relative_to(ROOT)} ({JSON_OUTPUT.stat().st_size} bytes)")
    print(f"wrote {BINARY_OUTPUT.relative_to(ROOT)} ({len(binary)} bytes)")
    print(f"binary sha256 {sha256(binary)}")
    print(f"sensitivities {sensitivities}")
    print(f"sparse margins {sparse_margins}")


if __name__ == "__main__":
    main()
