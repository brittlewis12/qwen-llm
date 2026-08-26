# /// script
# requires-python = "==3.11.*"
# dependencies = ["numpy==1.26.4", "torch==2.4.1"]
# ///

"""Generate an independent PyTorch routing oracle for Qwen4Exp MoE."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
JSON_OUTPUT = ROOT / "crates/qwen-llm/tests/fixtures/qwen4exp_moe_routing_v1.json"
BINARY_OUTPUT = ROOT / "crates/qwen-llm/tests/fixtures/qwen4exp_moe_routing_v1.f32"

HIDDEN = 256
EXPERTS = 16
TOP_K = 10
ROUTED_INTERMEDIATE = 32
SHARED_INTERMEDIATE = 32


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_fixture() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    index = torch.arange(HIDDEN, dtype=torch.int64)
    raw_input = ((index * 37 + index // 7 * 11 + 5) % 101) - 50
    hidden = raw_input.to(torch.float32) * 0.0125
    norm_sq = torch.dot(hidden, hidden)

    targets = torch.tensor(
        [
            0.25,
            0.80,
            -0.40,
            1.70,
            -0.90,
            0.30,
            -0.25,
            0.55,
            -0.60,
            1.10,
            0.45,
            -0.10,
            1.35,
            -0.75,
            1.70,
            0.95,
        ],
        dtype=torch.float32,
    )
    router = torch.stack(tuple(target * hidden / norm_sq for target in targets))
    router[14] = router[3]

    shared_raw = ((index * 19 + index // 5 * 7 + 3) % 79) - 39
    shared_router = shared_raw.to(torch.float32) * 0.003

    logits = torch.mv(router, hidden)
    ids = sorted(range(EXPERTS), key=lambda expert: (-float(logits[expert]), expert))[
        :TOP_K
    ]
    selected = logits[ids]
    weights = torch.softmax(selected, dim=0)
    shared_gate = torch.sigmoid(torch.dot(shared_router, hidden)).reshape(1)

    expected_ids = [3, 14, 12, 9, 15, 1, 7, 10, 5, 0]
    if ids != expected_ids:
        raise RuntimeError(
            f"routing recipe changed: expected {expected_ids}, got {ids}"
        )
    if not torch.equal(router[3], router[14]) or logits[3] != logits[14]:
        raise RuntimeError("routing tie sentinel is not exact")

    tensors = {
        "input": hidden,
        "router": router,
        "shared_router": shared_router,
        "router_logits": logits,
        "topk_weights": weights,
        "shared_gate": shared_gate,
    }
    full_softmax_selected_sum = float(torch.softmax(logits, dim=0)[ids].sum())
    metadata = {
        "schema_version": 1,
        "generator_version": 1,
        "geometry": {
            "hidden_size": HIDDEN,
            "expert_count": EXPERTS,
            "experts_per_token": TOP_K,
            "routed_intermediate_size": ROUTED_INTERMEDIATE,
            "shared_intermediate_size": SHARED_INTERMEDIATE,
        },
        "semantics": {
            "router": "F32 matrix-vector product",
            "tie_break": "lower expert ID first",
            "normalization": "softmax over selected logits",
            "shared_gate": "sigmoid(dot(shared_router, input))",
        },
        "topk_ids": ids,
        "sentinels": {
            "exact_tie_ids": [3, 14],
            "selected_weight_sum": float(weights.sum()),
            "full_softmax_selected_sum_without_renormalization": full_softmax_selected_sum,
        },
    }
    return tensors, metadata


def encode_fixture() -> tuple[bytes, bytes]:
    tensors, metadata = make_fixture()
    payload = bytearray()
    sections: dict[str, Any] = {}
    for name, tensor in tensors.items():
        values = tensor.contiguous().view(-1).tolist()
        offset = len(payload) // 4
        payload.extend(struct.pack(f"<{len(values)}f", *values))
        sections[name] = {
            "offset_f32": offset,
            "count_f32": len(values),
            "shape": list(tensor.shape),
        }

    binary = bytes(payload)
    metadata["binary"] = {
        "file": BINARY_OUTPUT.name,
        "dtype": "f32",
        "byte_order": "little",
        "sha256": sha256(binary),
        "sections": sections,
    }
    encoded_json = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    return encoded_json, binary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded_json, binary = encode_fixture()
    if args.check:
        if (
            JSON_OUTPUT.read_bytes() != encoded_json
            or BINARY_OUTPUT.read_bytes() != binary
        ):
            raise SystemExit("Qwen4Exp MoE oracle fixture is stale")
        return
    JSON_OUTPUT.write_bytes(encoded_json)
    BINARY_OUTPUT.write_bytes(binary)


if __name__ == "__main__":
    main()
