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
    allow_zero_energy: bool = False,
) -> dict[str, Any]:
    absolute = np.abs(values)
    blocks = values.reshape(values.shape[0], -1, block_size)
    block_max = np.max(np.abs(blocks), axis=-1)
    block_energy = np.sum(np.square(blocks.astype(np.float64)), axis=-1)
    total_energy = np.sum(block_energy, axis=-1, keepdims=True)
    if np.any(total_energy == 0) and not allow_zero_energy:
        raise ValueError("captured an all-zero FFN inner vector")
    shares = np.divide(
        block_energy,
        total_energy,
        out=np.zeros_like(block_energy),
        where=total_energy != 0,
    )
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
            np.where(
                total_energy[:, 0] == 0,
                0,
                np.argmax(descending_cumulative >= target, axis=-1) + 1,
            )
        )
        for target in ENERGY_COVERAGE
    }
    concentration = np.sum(np.square(shares), axis=-1)
    effective_blocks = np.divide(
        1.0, concentration, out=np.zeros_like(concentration), where=concentration != 0
    )
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


def q8_ffn_ledger(hidden: int, intermediate: int, layers: int) -> dict[str, int]:
    if any(
        type(value) is not int or value <= 0 for value in (hidden, intermediate, layers)
    ):
        raise ValueError("Q8 FFN dimensions must be positive integers")
    if hidden % 32 or intermediate % 32:
        raise ValueError("Q8 FFN dimensions must be multiples of 32")
    matrix = hidden * intermediate // 32 * 34
    return {
        "weight_bytes_per_matrix": matrix,
        "weight_bytes_per_layer": 3 * matrix,
        "weight_bytes_all_layers": layers * 3 * matrix,
        "up_bytes_per_removed_block": 32 * (hidden // 32) * 34,
        "down_bytes_per_removed_block": hidden * 34,
        "fusion_weight_bytes_removed": 0,
        "fusion_dispatches_removed_all_layers": 2 * layers,
        "fusion_intermediate_bytes_removed_per_layer": 4 * intermediate * 4,
        # The lcpp kernels reread x per two-output-row threadgroup. Cache traffic is unknown.
        "fusion_shader_x_load_bytes_removed_per_layer": (intermediate // 2)
        * hidden
        * 4,
        "fusion_x_unique_bytes": hidden * 4,
        "fusion_threadgroup_bytes_before": 32 * 2 * 4,
        "fusion_threadgroup_bytes_after": 32 * 2 * 2 * 4,
    }


def analyze_swiglu(
    gate: np.ndarray, up: np.ndarray, inner: np.ndarray, thresholds: tuple[float, ...]
) -> dict[str, Any]:
    if gate.shape != up.shape or gate.shape != inner.shape or inner.ndim != 2:
        raise ValueError("gate/up/inner must have identical [vectors, channels] shapes")
    if not inner.size or inner.shape[1] % 32:
        raise ValueError("SwiGLU census requires complete Q8 blocks of 32 channels")
    if any(not np.isfinite(values).all() for values in (gate, up, inner)):
        raise ValueError("SwiGLU census contains non-finite values")
    if not thresholds or any(t < 0 or not math.isfinite(t) for t in thresholds):
        raise ValueError("thresholds must be finite and nonnegative")
    if len({key(t) for t in thresholds}) != len(thresholds):
        raise ValueError("thresholds collide in report keys")

    raw = gate.astype(np.float64)
    exponential = np.exp(-np.abs(raw))
    sigmoid = np.where(raw >= 0, 1 / (1 + exponential), exponential / (1 + exponential))
    proxy = raw * sigmoid
    shape = (inner.shape[0], -1, 32)
    gate_max = np.max(np.abs(proxy.reshape(shape)), axis=-1)
    raw_gate_max = np.max(np.abs(raw.reshape(shape)), axis=-1)
    product_max = np.max(np.abs(inner.reshape(shape)), axis=-1)
    energy = np.sum(np.square(inner.astype(np.float64).reshape(shape)), axis=-1)
    total = np.sum(energy, axis=-1, keepdims=True)
    shares = np.divide(energy, total, out=np.zeros_like(energy), where=total != 0)
    cumulative = np.cumsum(np.sort(shares, axis=-1), axis=-1)
    oracle = {}
    for budget in ENERGY_BUDGETS:
        removable = np.count_nonzero(cumulative <= budget, axis=-1) / energy.shape[1]
        oracle[key(budget)] = {
            "block_fraction": summary(removable),
            "all_ffn_weight_fraction_down_only": summary(removable / 3),
        }
    candidates = {}
    for threshold in thresholds:
        mask = gate_max <= threshold
        product_mask = product_max <= threshold
        count = int(np.count_nonzero(mask))
        violations = int(np.count_nonzero(mask & ~product_mask))
        candidates[key(threshold)] = {
            "selected_blocks": count,
            "raw_gate_blocks_at_same_threshold": int(
                np.count_nonzero(raw_gate_max <= threshold)
            ),
            "product_threshold_violations": violations,
            "conditional_violation_fraction": violations / count if count else None,
            "missed_product_threshold_blocks": int(
                np.count_nonzero(~mask & product_mask)
            ),
            "block_fraction": summary(np.mean(mask, axis=-1)),
            "discarded_observed_inner_energy_fraction": summary(
                np.sum(shares * mask, axis=-1)
            ),
            "all_ffn_weight_fraction_up_down_hypothesis": summary(
                np.mean(mask, axis=-1) * 2 / 3
            ),
        }
    observed_up_max = np.max(np.abs(up.astype(np.float64)), axis=-1)
    return {
        "interpretation": "offline activation diagnostic; not an output-error certificate or runtime selector",
        "gate_signal": "F64 SiLU(raw gate) proxy, not deployed Metal arithmetic",
        "threshold_semantics": "paired same-numeric-threshold diagnostic curves, not a gate-to-product implication",
        "product_signal": "captured deployed inner, not CPU-reconstructed gate times up",
        "denominator": {
            "vectors": inner.shape[0],
            "blocks_per_vector": energy.shape[1],
            "includes_valid_zero_energy_vectors": True,
        },
        "zero_energy_vectors": int(np.count_nonzero(total == 0)),
        "raw_gate_max_abs": summary(np.max(raw_gate_max, axis=-1)),
        "observed_up_max_abs": summary(observed_up_max),
        "post_product_oracle": oracle,
        "gate_only_candidates": candidates,
        "cost_scope": "gate remains dense; product oracle can remove only down; selection/gather/repacking and quality unpriced",
    }


def load_tensor(manifest_path: Path, tensor: dict[str, Any]) -> tuple[np.ndarray, str]:
    if tensor["dtype"] != "f32le":
        raise ValueError("FFN census tensor must be f32le")
    shape = tensor["shape"]
    if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("FFN census shape must have three positive integer dimensions")
    path = manifest_path.parent / tensor["name"]
    size, digest = file_sha256(path)
    if (
        size != tensor["byte_length"]
        or size != math.prod(shape) * 4
        or digest != tensor["sha256"]
    ):
        raise ValueError("FFN census tensor identity or shape/byte length mismatch")
    values = np.memmap(path, dtype="<f4", mode="r", shape=tuple(shape))
    if not np.isfinite(values).all():
        raise ValueError("FFN census contains non-finite values")
    return values, digest


def validate_swiglu_provenance(manifest: dict[str, Any], samples: int) -> None:
    layer_count = manifest.get("layer_count")
    if (
        type(layer_count) is not int
        or layer_count <= 0
        or manifest.get("layers") != list(range(layer_count))
    ):
        raise ValueError(
            "v2 requires complete ordered layer coverage, not extrapolation from a subset"
        )
    if (
        manifest.get("weight_dtype") != "Q8_0"
        or manifest.get("matvec_variant") != "lcpp_nr0_2_nsg_4"
    ):
        raise ValueError("v2 ledger requires declared Q8_0 singleton lcpp geometry")
    for name in ("source_commit", "model_content_identity", "prefix_identity"):
        if not isinstance(manifest.get(name), str) or not manifest[name].strip():
            raise ValueError(
                f"v2 requires {name}; tensor hashes do not authenticate model/state"
            )
    for name in ("captured_token_ids", "captured_positions"):
        values = manifest.get(name)
        if (
            not isinstance(values, list)
            or len(values) != samples
            or any(type(value) is not int or value < 0 for value in values)
        ):
            raise ValueError(f"v2 {name} must identify every captured sample")
    positions = manifest["captured_positions"]
    if positions != sorted(set(positions)):
        raise ValueError("v2 captured positions must be strictly increasing")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure physical-block occupancy of captured FFN inner vectors or Q8 SwiGLU streams."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--thresholds",
        type=parse_floats,
        default=DEFAULT_THRESHOLDS,
        help="comma-separated absolute-value thresholds",
    )
    parser.add_argument(
        "--block-size", type=int, help="default: 256 for v1; Q8 block32 for v2"
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    version = manifest.get("schema_version")
    if version not in (1, 2):
        raise SystemExit("unsupported FFN census manifest schema")
    block_size = (
        args.block_size
        if args.block_size is not None
        else (32 if version == 2 else 256)
    )
    if block_size <= 0 or (version == 2 and block_size != 32):
        raise SystemExit("block size must be positive; v2 requires Q8 block32")
    tensors = manifest["tensors"] if version == 2 else {"inner": manifest["tensor"]}
    if version == 2 and set(tensors) != {"gate", "up", "inner"}:
        raise SystemExit("v2 requires gate, up and inner streams")
    loaded = {
        name: load_tensor(args.manifest, tensor) for name, tensor in tensors.items()
    }
    values, digest = loaded["inner"]
    shape = values.shape
    if shape[2] % block_size or any(
        array.shape != shape for array, _ in loaded.values()
    ):
        raise SystemExit("FFN census shape is incompatible with block size")

    if version == 2:
        validate_swiglu_provenance(manifest, shape[0])
        if (
            type(manifest.get("intermediate_size")) is not int
            or manifest["intermediate_size"] != shape[2]
        ):
            raise ValueError("v2 intermediate_size disagrees with captured shape")
    layers = manifest["layers"]
    if (
        len(layers) != shape[1]
        or any(type(layer) is not int or layer < 0 for layer in layers)
        or layers != sorted(set(layers))
    ):
        raise SystemExit("FFN census layer metadata disagrees with tensor shape")
    aggregate = analyze_layer(
        values.reshape(shape[0] * shape[1], shape[2]),
        args.thresholds,
        block_size,
        allow_zero_energy=version == 2,
    )
    per_layer = {
        str(layer): analyze_layer(
            values[:, index, :],
            args.thresholds,
            block_size,
            allow_zero_energy=version == 2,
        )
        for index, layer in enumerate(layers)
    }
    report = {
        "schema_version": version,
        "manifest": str(args.manifest),
        "tensor_sha256": digest,
        "samples": shape[0],
        "layers": layers,
        "intermediate_size": shape[2],
        "block_size": block_size,
        "blocks_per_layer": shape[2] // block_size,
        "thresholds": list(args.thresholds),
        "aggregate": aggregate,
        "per_layer": per_layer,
    }
    if version == 2:
        gate, up = loaded["gate"][0], loaded["up"][0]
        report["tensor_sha256_by_stream"] = {
            name: checksum for name, (_, checksum) in loaded.items()
        }
        report["q8_source_ledger"] = q8_ffn_ledger(
            manifest["hidden_size"], shape[2], len(layers)
        )
        report["q8_source_ledger_scope"] = (
            "all declared layers, Q8_0 singleton lcpp NR0=2/NSG=4; source loads are not DRAM traffic"
        )
        report["provenance_scope"] = (
            "stream bytes/hashes verified; model, prefix and source identities are producer assertions, not authenticated here"
        )
        report["capture_provenance"] = {
            name: manifest[name]
            for name in (
                "source_commit",
                "model_content_identity",
                "prefix_identity",
                "captured_token_ids",
                "captured_positions",
            )
        }
        report["swiglu"] = analyze_swiglu(
            gate.reshape(-1, shape[2]),
            up.reshape(-1, shape[2]),
            values.reshape(-1, shape[2]),
            args.thresholds,
        )
        report["swiglu_per_layer"] = {
            str(layer): analyze_swiglu(
                gate[:, index], up[:, index], values[:, index], args.thresholds
            )
            for index, layer in enumerate(layers)
        }

    print("metric\tvalue")
    if "0" in aggregate["scalar_cdf_abs_le"]:
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
