#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


RANKS = (8, 16, 24, 32, 48, 64, 96)
RESIDUAL_TARGETS = (1e-1, 1e-2, 1e-3, 1e-4)
ALPHA_THRESHOLDS = (0.0, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0)


def file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def validate_tensor(root: Path, tensor: dict[str, Any]) -> Path:
    path = root / tensor["name"]
    size, digest = file_sha256(path)
    if size != tensor["byte_length"] or digest != tensor["sha256"]:
        raise ValueError(f"tensor identity mismatch: {path}")
    expected = math.prod(tensor["shape"]) * np.dtype("<f4").itemsize
    if size != expected or tensor["dtype"] != "f32le":
        raise ValueError(f"tensor shape/dtype mismatch: {path}")
    return path


def percentile(values: np.ndarray, q: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    index = max(0, math.ceil(ordered.size * q) - 1)
    return float(ordered[index])


def summary(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("metric summary received empty or non-finite values")
    return {
        "samples": int(flat.size),
        "min": float(np.min(flat)),
        "mean": float(np.mean(flat)),
        "median": percentile(flat, 0.5),
        "p95": percentile(flat, 0.95),
        "max": float(np.max(flat)),
    }


def key(value: float) -> str:
    return f"{value:.6g}"


def analyze_alpha(values: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("alpha capture is outside [0,1]")
    positive_interior = values[(values > 0) & (values < 1)].astype(np.float64)
    half_life = np.log(0.5) / np.log(positive_interior)
    return {
        "values": summary(values),
        "exact_zero": int(np.count_nonzero(values == 0)),
        "exact_one": int(np.count_nonzero(values == 1)),
        "cdf_le": {
            key(threshold): float(np.count_nonzero(values <= threshold) / values.size)
            for threshold in ALPHA_THRESHOLDS
        },
        "decay_half_life_tokens": summary(half_life),
    }


def rank_for_residual(squared: np.ndarray, target: float) -> np.ndarray:
    total = np.sum(squared, axis=-1, keepdims=True)
    retained = np.cumsum(squared, axis=-1)
    residual = np.sqrt(np.maximum(0.0, 1.0 - retained / total))
    reached = residual <= target
    if not np.all(np.any(reached, axis=-1)):
        raise ValueError("SVD residual target was not reached")
    return np.argmax(reached, axis=-1) + 1


def analyze_states(entries: list[tuple[dict[str, Any], np.ndarray]]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    by_position: dict[int, list[np.ndarray]] = defaultdict(list)
    all_singular: list[np.ndarray] = []
    for metadata, states in entries:
        singular = np.linalg.svd(states.astype(np.float64), compute_uv=False)
        squared = np.square(singular)
        total = np.sum(squared, axis=-1)
        if np.any(total == 0):
            raise ValueError("captured an all-zero GDN state head")
        residual_by_rank = {
            str(rank): summary(np.sqrt(np.sum(squared[:, rank:], axis=-1) / total))
            for rank in RANKS
        }
        rank_by_target = {
            key(target): summary(rank_for_residual(squared, target))
            for target in RESIDUAL_TARGETS
        }
        stable_rank = total / squared[:, 0]
        group = {
            "position": metadata["position"],
            "gdn_index": metadata["gdn_index"],
            "absolute_layer": metadata["absolute_layer"],
            "stable_rank": summary(stable_rank),
            "relative_frobenius_residual_by_rank": residual_by_rank,
            "rank_for_relative_frobenius_residual": rank_by_target,
        }
        groups.append(group)
        by_position[metadata["position"]].append(singular)
        all_singular.append(singular)

    def aggregate_singular(collection: list[np.ndarray]) -> dict[str, Any]:
        singular = np.concatenate(collection, axis=0)
        squared = np.square(singular)
        total = np.sum(squared, axis=-1)
        return {
            "heads": int(singular.shape[0]),
            "stable_rank": summary(total / squared[:, 0]),
            "relative_frobenius_residual_by_rank": {
                str(rank): summary(np.sqrt(np.sum(squared[:, rank:], axis=-1) / total))
                for rank in RANKS
            },
            "rank_for_relative_frobenius_residual": {
                key(target): summary(rank_for_residual(squared, target))
                for target in RESIDUAL_TARGETS
            },
        }

    return {
        "aggregate": aggregate_singular(all_singular),
        "by_position": {
            str(position): aggregate_singular(collection)
            for position, collection in sorted(by_position.items())
        },
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure GDN decay endpoints and live-state numerical rank."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 1:
        raise SystemExit("unsupported GDN census manifest schema")
    root = args.manifest.parent
    alpha_path = validate_tensor(root, manifest["alpha"])
    alpha_shape = tuple(manifest["alpha"]["shape"])
    alpha = np.memmap(alpha_path, dtype="<f4", mode="r", shape=alpha_shape)

    state_entries: list[tuple[dict[str, Any], np.ndarray]] = []
    for state in manifest["states"]:
        path = validate_tensor(root, state["tensor"])
        shape = tuple(state["tensor"]["shape"])
        values = np.memmap(path, dtype="<f4", mode="r", shape=shape)
        if not np.isfinite(values).all():
            raise SystemExit(f"state contains non-finite values: {path}")
        state_entries.append((state, values))

    report = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "alpha": analyze_alpha(alpha),
        "states": analyze_states(state_entries),
    }
    alpha_report = report["alpha"]
    mature_position = max(report["states"]["by_position"], key=int)
    state_report = report["states"]["by_position"][mature_position]
    print("metric\tvalue")
    print(f"alpha_exact_zero\t{alpha_report['exact_zero']}")
    print(f"alpha_exact_one\t{alpha_report['exact_one']}")
    print(f"alpha_median\t{alpha_report['values']['median']:.9f}")
    print(
        "alpha_half_life_median\t"
        f"{alpha_report['decay_half_life_tokens']['median']:.3f}"
    )
    print(f"mature_position\t{mature_position}")
    print(f"stable_rank_median\t{state_report['stable_rank']['median']:.3f}")
    for rank in (24, 32, 64):
        residual = state_report["relative_frobenius_residual_by_rank"][str(rank)]
        print(f"rank_{rank}_residual_median\t{residual['median']:.6f}")
        print(f"rank_{rank}_residual_p95\t{residual['p95']:.6f}")
    for target in (1e-2, 1e-3):
        ranks = state_report["rank_for_relative_frobenius_residual"][key(target)]
        print(f"rank_for_residual_{key(target)}_median\t{ranks['median']:.1f}")
        print(f"rank_for_residual_{key(target)}_p95\t{ranks['p95']:.1f}")

    output = args.json_out or root / "analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
