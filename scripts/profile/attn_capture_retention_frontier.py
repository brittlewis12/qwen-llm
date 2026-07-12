#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]
# ///

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from attn_capture_analyze import check_file, mmap_tensor, validate_manifest


VALID_FRACTIONS = (0.5, 0.25, 0.125, 0.0625)


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = k.astype(np.float64) @ q.astype(np.float64) / math.sqrt(q.size)
    scores -= np.max(scores)
    weights = np.exp(scores)
    weights /= np.sum(weights)
    return weights.astype(np.float32), weights @ v.astype(np.float64)


def output_for(
    weights: np.ndarray, v: np.ndarray, indices: np.ndarray
) -> tuple[float, np.ndarray]:
    selected = weights[indices].astype(np.float64)
    mass = float(np.sum(selected))
    return mass, selected @ v[indices].astype(np.float64) / mass


def error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = candidate - reference
    ref_norm = float(np.linalg.norm(reference))
    cand_norm = float(np.linalg.norm(candidate))
    rms = float(np.sqrt(np.mean(np.square(reference))))
    return {
        "cosine": float(np.dot(reference, candidate) / (ref_norm * cand_norm)),
        "relative_l2": float(np.linalg.norm(delta) / ref_norm),
        "rms_normalized_max": float(np.max(np.abs(delta)) / rms),
    }


def top_indices(values: np.ndarray, keep: int) -> np.ndarray:
    return np.argpartition(values, -keep)[-keep:]


def chunk_indices(values: np.ndarray, keep: int, chunk: int) -> np.ndarray:
    chunks = math.ceil(len(values) / chunk)
    padded = np.pad(values, (0, chunks * chunk - len(values)))
    masses = padded.reshape(chunks, chunk).sum(axis=1)
    keep_chunks = max(1, math.ceil(keep / chunk))
    chosen = top_indices(masses, keep_chunks)
    indices = (chosen[:, None] * chunk + np.arange(chunk)).reshape(-1)
    return indices[indices < len(values)]


def summarize(rows: list[dict]) -> dict:
    result = {"count": len(rows)}
    for metric in ("mass", "cosine", "relative_l2", "rms_normalized_max"):
        values = np.array([row[metric] for row in rows])
        for percentile in (1, 5, 50, 95, 99):
            result[f"{metric}_p{percentile}"] = float(np.percentile(values, percentile))
        result[f"{metric}_worst"] = float(
            np.min(values) if metric in {"mass", "cosine"} else np.max(values)
        )
    physical = np.array([row["physical_fraction"] for row in rows])
    result["physical_fraction_p50"] = float(np.percentile(physical, 50))
    result["physical_fraction_p99"] = float(np.percentile(physical, 99))
    return result


def evaluate_view(
    positions: list[int],
    rows: list[int],
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    sample_queries: int,
    fractions: tuple[float, ...],
) -> dict[str, list[dict]]:
    sampled_ordinals = np.linspace(0, len(rows) - 1, sample_queries, dtype=int)
    sampled = set(int(index) for index in sampled_ordinals)
    records: dict[str, list[dict]] = {}
    for kv_head in range(2):
        all_weights: list[list[np.ndarray]] = []
        references: list[list[np.ndarray]] = []
        for row in rows:
            length = positions[row] + 1
            head_weights = []
            head_references = []
            for local_head in range(8):
                head = kv_head * 8 + local_head
                weights, reference = attention(
                    q[row, head], k[:length, kv_head], v[:length, kv_head]
                )
                head_weights.append(weights)
                head_references.append(reference)
            all_weights.append(head_weights)
            references.append(head_references)
        first_length = len(all_weights[0][0])
        aggregate = np.zeros(first_length, dtype=np.float64)
        for query_weights in all_weights:
            for weights in query_weights:
                aggregate += weights[:first_length]
        storage_sets = {}
        for fraction in fractions:
            keep = max(1, int(first_length * fraction))
            storage_sets[("storage_token", fraction)] = top_indices(aggregate, keep)
            for chunk in (64, 256):
                storage_sets[(f"storage_chunk{chunk}", fraction)] = chunk_indices(
                    aggregate, keep, chunk
                )
        for ordinal, row in enumerate(rows):
            if ordinal not in sampled:
                continue
            length = positions[row] + 1
            weights_by_head = all_weights[ordinal]
            aggregate_query = np.sum(weights_by_head, axis=0)
            for fraction in fractions:
                keep = max(1, int(length * fraction))
                shared_sets = {
                    "recent": np.arange(length - keep, length),
                    "token_group": top_indices(aggregate_query, keep),
                    "chunk64_group": chunk_indices(aggregate_query, keep, 64),
                    "chunk256_group": chunk_indices(aggregate_query, keep, 256),
                }
                for policy, indices in storage_sets.items():
                    policy_name, policy_fraction = policy
                    if policy_fraction == fraction:
                        shared_sets[policy_name] = indices[indices < length]
                ideal_sets = [top_indices(weights, keep) for weights in weights_by_head]
                union = np.unique(np.concatenate(ideal_sets))
                for local_head, (weights, reference) in enumerate(
                    zip(weights_by_head, references[ordinal], strict=True)
                ):
                    mass, candidate = output_for(
                        weights, v[:length, kv_head], ideal_sets[local_head]
                    )
                    key = f"token_per_head:{fraction}"
                    records.setdefault(key, []).append(
                        {
                            "mass": mass,
                            "physical_fraction": len(union) / length,
                            **error(reference, candidate),
                        }
                    )
                    for policy, indices in shared_sets.items():
                        mass, candidate = output_for(
                            weights, v[:length, kv_head], indices
                        )
                        key = f"{policy}:{fraction}"
                        records.setdefault(key, []).append(
                            {
                                "mass": mass,
                                "physical_fraction": len(indices) / length,
                                **error(reference, candidate),
                            }
                        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Price long-attention retention frontiers"
    )
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-queries", type=int, default=8)
    parser.add_argument("--fractions", default="0.5")
    parser.add_argument("--blocks", default="3,19,39")
    args = parser.parse_args()
    if not 1 <= args.sample_queries <= 32:
        raise ValueError("sample queries must be in 1..32")
    fractions = tuple(float(value) for value in args.fractions.split(","))
    if not fractions or any(value not in VALID_FRACTIONS for value in fractions):
        raise ValueError(f"fractions must be drawn from {VALID_FRACTIONS}")
    selected_blocks = {int(value) for value in args.blocks.split(",")}
    if not selected_blocks or not selected_blocks <= {3, 19, 39}:
        raise ValueError("blocks must be drawn from 3,19,39")
    manifest = json.loads((args.capture / "manifest.json").read_text())
    validate_manifest(manifest)
    if manifest["capture_tier"] != "canonical":
        raise ValueError("retention frontier requires the canonical artifact")
    entries = [manifest["tokens"], *manifest["tensors"]]
    for entry in entries:
        check_file(args.capture, entry)
    tensors = {entry["name"]: entry for entry in manifest["tensors"]}
    positions = manifest["unique_positions"]
    packet = {
        "schema_version": 1,
        "capture": str(args.capture),
        "sample_queries_per_tail": args.sample_queries,
        "fractions": fractions,
        "semantics": {
            "token_per_head": "exact-score ideal; physical fraction is eight-head union",
            "token_group": "exact-score aggregate shared by eight Q heads",
            "chunk_group": "exact-score chunk-mass aggregate shared by eight Q heads",
            "storage": "clairvoyant tail-window aggregate restricted to first-query prefix",
        },
        "results": {},
    }
    for block in manifest["blocks"]:
        index = block["block"]
        if index not in selected_blocks:
            continue
        print(f"block {index}", file=sys.stderr, flush=True)
        k = mmap_tensor(args.capture, tensors[f"block-{index}-k.f16le"])
        v = mmap_tensor(args.capture, tensors[f"block-{index}-v.f16le"])
        q = mmap_tensor(args.capture, tensors[f"block-{index}-q.f32le"])
        for view in manifest["views"]:
            tail_rows = view["deduplicated_rows"][32:]
            records = evaluate_view(
                positions, tail_rows, q, k, v, args.sample_queries, fractions
            )
            context_label = "32k" if view["context"] == 32768 else "131k"
            label = f"block{index}:{context_label}_tail"
            packet["results"][label] = {
                policy: summarize(rows) for policy, rows in sorted(records.items())
            }
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"strata": sorted(packet["results"])}, indent=2))


if __name__ == "__main__":
    main()
