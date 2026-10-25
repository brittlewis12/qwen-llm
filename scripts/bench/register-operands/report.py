#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Summarize retained raw evidence; spread is descriptive, never a run veto."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    samples = defaultdict(list)
    parity = []
    operands = []
    prices = []
    for record in records:
        if record["event"] == "sample" and record["repeats"] > 1:
            samples[record["name"]].append(record["gpu_ns"])
        if record["event"] == "output_parity":
            parity.append(record)
        if record["event"] == "operand_parity":
            operands.append(record)
        if record["event"] == "price":
            prices.append(record)
    timing = {
        name: {
            "samples_ns": values,
            "median_ns": statistics.median(values),
            "min_ns": min(values),
            "max_ns": max(values),
            "max_over_min": max(values) / min(values),
        }
        for name, values in samples.items()
    }
    baseline = next(
        (v for k, v in timing.items() if k.startswith("kernel_moe_down")), None
    )
    for name, stats in timing.items():
        reference = (
            timing.get("staged_resident_" + name.rsplit("_resident_", 1)[-1])
            if "resident" in name
            else baseline
        )
        if reference:
            stats["matched_median_speedup"] = (
                reference["median_ns"] / stats["median_ns"]
            )
    return {
        "path": str(path),
        "timing": timing,
        "max_priced_bytes": max((r["priced_bytes"] for r in prices), default=0),
        "output_parity_cases": len(parity),
        "output_bit_mismatches": sum(r["bit_mismatches_vs_baseline"] for r in parity),
        "max_output_delta_vs_baseline": max(
            (r["max_abs_vs_baseline"] for r in parity), default=0
        ),
        "min_cosine": min((r["cosine"] for r in parity), default=None),
        "operand_half_values_checked": sum(r["half_values_checked"] for r in operands),
        "operand_bit_mismatches": sum(r["bit_mismatches"] for r in operands),
        "stability_policy": "descriptive only; all samples retained, no spread veto",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.paths], indent=2))
