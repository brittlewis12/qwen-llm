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

from attn_capture_analyze import check_file, mmap_tensor, replay_head, validate_manifest


BITS = (8, 6, 4)
QMAX = {bits: (1 << (bits - 1)) - 1 for bits in BITS}


def candidate_parts(name: str) -> tuple[str, int, int]:
    if name.startswith("g16"):
        group_size = 16
    elif name.startswith("g32"):
        group_size = 32
    else:
        group_size = 256
    core = name[3:] if group_size != 256 else name
    return core[:-1], int(core[-1]), group_size


def quantized_row_bytes(bits: int, group_size: int) -> int:
    payload = (256 * bits + 7) // 8
    scales = 256 // group_size * 2
    return (scales + payload + 15) // 16 * 16


def quantize_symmetric(source: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    output = np.empty(source.shape, dtype=np.float16)
    qmax = QMAX[bits]
    for start in range(0, source.shape[0], 4096):
        chunk = source[start : start + 4096].astype(np.float32)
        grouped = chunk.reshape(*chunk.shape[:-1], 256 // group_size, group_size)
        max_abs = np.max(np.abs(grouped), axis=-1, keepdims=True)
        scale = (max_abs / qmax).astype(np.float16).astype(np.float32)
        scale[scale == 0] = 1.0
        codes = np.clip(np.rint(grouped / scale), -qmax, qmax)
        output[start : start + len(chunk)] = (
            (codes * scale).reshape(chunk.shape).astype(np.float16)
        )
    return output


def classify(context: int, ordinal: int, position: int) -> str:
    if ordinal >= 32:
        return "32k_tail" if context == 32768 else "131k_tail"
    if context == 32768:
        return "32k_distributed"
    band = min(position // 32768, 3)
    return f"131k_distributed_band{band}"


def query_rows(manifest: dict, per_class: int | None) -> list[tuple[int, str]]:
    positions = manifest["unique_positions"]
    selected: list[tuple[int, str]] = []
    for view in manifest["views"]:
        rows = view["deduplicated_rows"]
        for class_rows in (rows[:32], rows[32:]):
            if per_class is not None and len(class_rows) > per_class:
                indices = np.linspace(0, len(class_rows) - 1, per_class, dtype=int)
                class_rows = [
                    class_rows[index] for index in sorted(set(indices.tolist()))
                ]
            for row in class_rows:
                ordinal = rows.index(row)
                selected.append(
                    (row, classify(view["context"], ordinal, positions[row]))
                )
    return selected


def metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    delta = actual - expected
    expected_norm = float(np.linalg.norm(expected))
    actual_norm = float(np.linalg.norm(actual))
    rms = float(np.sqrt(np.mean(np.square(expected))))
    return {
        "cosine": float(np.dot(expected, actual) / (expected_norm * actual_norm)),
        "relative_l2": float(np.linalg.norm(delta) / expected_norm),
        "rms_normalized_max": float(np.max(np.abs(delta)) / rms),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def summarize(values: list[dict[str, float]]) -> dict:
    result: dict[str, float | int] = {"count": len(values)}
    for name in ("cosine", "relative_l2", "rms_normalized_max", "rmse", "max_abs"):
        array = np.array([value[name] for value in values])
        for percentile in (1, 5, 50, 95, 99):
            result[f"{name}_p{percentile}"] = float(np.percentile(array, percentile))
        result[f"{name}_worst"] = float(
            np.min(array) if name == "cosine" else np.max(array)
        )
    return result


def disposition(strata: dict[str, dict]) -> str:
    green = all(
        row["relative_l2_p99"] <= 0.01
        and row["cosine_p1"] >= 0.9999
        and row["rms_normalized_max_p99"] <= 0.10
        and row["relative_l2_worst"] <= 0.05
        for row in strata.values()
    )
    red = any(
        key.endswith(":131k_tail")
        and (row["relative_l2_p95"] > 0.05 or row["cosine_p5"] < 0.999)
        for key, row in strata.items()
    )
    return "green" if green else "red" if red else "yellow"


def candidate_bytes(name: str) -> dict:
    target, bits, group_size = candidate_parts(name)
    packed = quantized_row_bytes(bits, group_size)
    k_bytes = packed if target in {"k", "kv"} else 512
    v_bytes = packed if target in {"v", "kv"} else 512
    total = k_bytes + v_bytes
    return {
        "k_row_bytes": k_bytes,
        "v_row_bytes": v_bytes,
        "total_row_bytes": total,
        "fraction_of_f16": total / 1024,
        "scale_dtype": "f16",
        "group_size": group_size,
        "row_alignment": 16,
    }


def run_block(
    root: Path,
    manifest: dict,
    tensors: dict[str, dict],
    block: int,
    selected: list[tuple[int, str]],
    candidates: list[str],
) -> dict[str, list[dict]]:
    positions = manifest["unique_positions"]
    k = mmap_tensor(root, tensors[f"block-{block}-k.f16le"])
    v = mmap_tensor(root, tensors[f"block-{block}-v.f16le"])
    q = mmap_tensor(root, tensors[f"block-{block}-q.f32le"])
    references: dict[tuple[int, int], np.ndarray] = {}
    print(f"block {block}: replay full-F16 reference", file=sys.stderr, flush=True)
    for row, _ in selected:
        length = positions[row] + 1
        for head in range(16):
            references[(row, head)] = replay_head(
                q[row, head], k[:length, head // 8], v[:length, head // 8]
            )
    result: dict[str, list[dict]] = {}
    for candidate in candidates:
        target, bits, group_size = candidate_parts(candidate)
        print(f"block {block}: {candidate}", file=sys.stderr, flush=True)
        candidate_k = (
            quantize_symmetric(k, bits, group_size) if target in {"k", "kv"} else k
        )
        candidate_v = (
            quantize_symmetric(v, bits, group_size) if target in {"v", "kv"} else v
        )
        rows = []
        for row, query_class in selected:
            length = positions[row] + 1
            for head in range(16):
                actual = replay_head(
                    q[row, head],
                    candidate_k[:length, head // 8],
                    candidate_v[:length, head // 8],
                )
                rows.append(
                    {
                        "block": block,
                        "row": row,
                        "position": positions[row],
                        "head": head,
                        "kv_head": head // 8,
                        "query_class": query_class,
                        **metrics(references[(row, head)], actual),
                    }
                )
        result[candidate] = rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Price grouped compressed-KV fidelity")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries-per-class", type=int)
    parser.add_argument(
        "--candidates",
        default="k8,v8,kv8,k6,v6,kv6,k4,v4,kv4",
        help="comma-separated row/group candidates",
    )
    args = parser.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    validate_manifest(manifest)
    if manifest["capture_tier"] != "canonical":
        raise ValueError("quant frontier requires the canonical 131K artifact")
    entries = [manifest["tokens"], *manifest["tensors"]]
    for entry in entries:
        check_file(args.capture, entry)
    tensors = {entry["name"]: entry for entry in manifest["tensors"]}
    candidates = args.candidates.split(",")
    valid = {
        f"{group}{prefix}{bits}"
        for bits in BITS
        for prefix in ("k", "v", "kv")
        for group in ("", "g32", "g16")
    }
    if not candidates or any(candidate not in valid for candidate in candidates):
        raise ValueError(f"candidates must be drawn from {sorted(valid)}")
    selected = query_rows(manifest, args.queries_per_class)
    records = {candidate: [] for candidate in candidates}
    for block in manifest["blocks"]:
        block_records = run_block(
            args.capture,
            manifest,
            tensors,
            block["block"],
            selected,
            candidates,
        )
        for candidate in candidates:
            records[candidate].extend(block_records[candidate])
    results = {}
    for candidate in candidates:
        strata = {}
        keys = sorted(
            {(row["block"], row["query_class"]) for row in records[candidate]}
        )
        for block, query_class in keys:
            values = [
                row
                for row in records[candidate]
                if row["block"] == block and row["query_class"] == query_class
            ]
            strata[f"block{block}:{query_class}"] = summarize(values)
        results[candidate] = {
            "format": candidate_bytes(candidate),
            "disposition": disposition(strata),
            "strata": strata,
        }
    packet = {
        "schema_version": 1,
        "capture": str(args.capture),
        "capture_model_sha256": manifest["identity"]["model_sha256"],
        "quantizer": {
            "kind": "groupwise_symmetric_max_abs",
            "elements_per_row": 256,
            "rounding": "nearest",
            "signed_qmax": QMAX,
            "scale_dtype": "f16",
        },
        "selected_queries": len(selected),
        "thresholds": {
            "green": (
                "all strata p99 rel_l2<=.01, p1 cosine>=.9999, "
                "p99 max/RMS<=.10, worst rel_l2<=.05"
            ),
            "red": "any 131k_tail stratum p95 rel_l2>.05 or p5 cosine<.999",
        },
        "results": results,
    }
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                candidate: {
                    "disposition": result["disposition"],
                    "fraction_of_f16": result["format"]["fraction_of_f16"],
                }
                for candidate, result in results.items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
