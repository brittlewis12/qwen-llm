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
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_THRESHOLDS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
ENERGY_BUDGETS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
ENERGY_COVERAGE = (0.5, 0.9, 0.95, 0.99)


def parse_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in raw.split(",") if item.strip())
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError(
            "expected comma-separated finite nonnegative values"
        )
    return values


def file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def percentile(values: np.ndarray, q: float) -> float:
    flat = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    index = max(0, math.ceil(flat.size * q) - 1)
    return float(flat[index])


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
    return f"{value:.0e}" if value != 0 else "0"


def analyze_layer(
    values: np.ndarray,
    thresholds: tuple[float, ...],
    block_size: int,
) -> dict[str, Any]:
    absolute = np.abs(values)
    blocks = values.reshape(values.shape[0], -1, block_size)
    block_max = np.max(np.abs(blocks), axis=-1)
    block_energy = np.sum(np.square(blocks.astype(np.float64)), axis=-1)
    total_energy = np.sum(block_energy, axis=-1, keepdims=True)
    if np.any(total_energy == 0):
        raise ValueError("captured an all-zero FFN inner vector")
    shares = block_energy / total_energy
    ascending = np.sort(shares, axis=-1)
    ascending_cumulative = np.cumsum(ascending, axis=-1)
    descending_cumulative = np.cumsum(ascending[..., ::-1], axis=-1)

    scalar_cdf = {
        key(threshold): float(np.count_nonzero(absolute <= threshold) / absolute.size)
        for threshold in thresholds
    }
    block_cdf = {
        key(threshold): float(np.count_nonzero(block_max <= threshold) / block_max.size)
        for threshold in thresholds
    }
    removable = {
        key(budget): summary(
            np.count_nonzero(ascending_cumulative <= budget, axis=-1) / shares.shape[-1]
        )
        for budget in ENERGY_BUDGETS
    }
    coverage = {
        f"{target:.2f}": summary(
            np.argmax(descending_cumulative >= target, axis=-1) + 1
        )
        for target in ENERGY_COVERAGE
    }
    effective_blocks = 1.0 / np.sum(np.square(shares), axis=-1)
    bits = values.view(np.uint32)
    return {
        "scalar_cdf_abs_le": scalar_cdf,
        "block_cdf_max_abs_le": block_cdf,
        "positive_zero_values": int(np.count_nonzero(bits == 0)),
        "negative_zero_values": int(np.count_nonzero(bits == 0x80000000)),
        "effective_block_count": summary(effective_blocks),
        "removable_block_fraction_by_energy_budget": removable,
        "largest_blocks_for_energy_coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure scalar and block-256 occupancy of captured FFN inner vectors."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--thresholds",
        type=parse_floats,
        default=DEFAULT_THRESHOLDS,
        help="comma-separated absolute-value thresholds",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 1:
        raise SystemExit("unsupported FFN census manifest schema")
    tensor = manifest["tensor"]
    if tensor["dtype"] != "f32le":
        raise SystemExit("FFN census tensor must be f32le")
    path = args.manifest.parent / tensor["name"]
    size, digest = file_sha256(path)
    if size != tensor["byte_length"] or digest != tensor["sha256"]:
        raise SystemExit("FFN census tensor identity mismatch")
    shape = tuple(int(value) for value in tensor["shape"])
    if len(shape) != 3 or shape[2] % args.block_size != 0:
        raise SystemExit("FFN census shape is incompatible with block size")
    values = np.memmap(path, dtype="<f4", mode="r", shape=shape)
    if not np.isfinite(values).all():
        raise SystemExit("FFN census contains non-finite values")

    layers = manifest["layers"]
    if len(layers) != shape[1]:
        raise SystemExit("FFN census layer metadata disagrees with tensor shape")
    aggregate = analyze_layer(
        values.reshape(shape[0] * shape[1], shape[2]),
        args.thresholds,
        args.block_size,
    )
    per_layer = {
        str(layer): analyze_layer(values[:, index, :], args.thresholds, args.block_size)
        for index, layer in enumerate(layers)
    }
    report = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "tensor_sha256": digest,
        "samples": shape[0],
        "layers": layers,
        "intermediate_size": shape[2],
        "block_size": args.block_size,
        "blocks_per_layer": shape[2] // args.block_size,
        "thresholds": list(args.thresholds),
        "aggregate": aggregate,
        "per_layer": per_layer,
    }

    print("metric\tvalue")
    print(f"exact_zero_fraction\t{aggregate['scalar_cdf_abs_le']['0']:.9f}")
    for threshold in (1e-3, 1e-2, 1e-1):
        threshold_key = key(threshold)
        if threshold_key in aggregate["scalar_cdf_abs_le"]:
            print(
                f"scalar_abs_le_{threshold_key}\t"
                f"{aggregate['scalar_cdf_abs_le'][threshold_key]:.6f}"
            )
            print(
                f"blocks_max_abs_le_{threshold_key}\t"
                f"{aggregate['block_cdf_max_abs_le'][threshold_key]:.6f}"
            )
    print(
        f"effective_blocks_median\t{aggregate['effective_block_count']['median']:.3f}"
    )
    print(
        "removable_blocks_at_1e-3_energy_median\t"
        f"{aggregate['removable_block_fraction_by_energy_budget']['1e-03']['median']:.6f}"
    )
    print(
        "blocks_for_90pct_energy_median\t"
        f"{aggregate['largest_blocks_for_energy_coverage']['0.90']['median']:.1f}"
    )

    output = args.json_out or args.manifest.parent / "analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
