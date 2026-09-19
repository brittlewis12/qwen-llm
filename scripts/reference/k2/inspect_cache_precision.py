# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Summarize a complete K2 cache/backend precision diagnostic without GPU use."""

import argparse
import json
import math
from pathlib import Path

from inspect_oracle_metrics import read_json

PAIRS = ("native_f16_vs_ifm_f16", "native_f16_vs_ifm_f32", "ifm_f16_vs_ifm_f32")


def summarize(directory):
    manifest = read_json(directory / "manifest.json")
    evidence = read_json(directory / "metrics.json")
    if manifest["qualification"] is not False or evidence["qualification"] is not False:
        raise ValueError("expected diagnostic-only evidence")
    groups = {}
    for row in evidence["cases"]:
        groups.setdefault((row["corpus"], row["base"]), []).append(row)
    expected_groups = {(name, base) for name, base, _ in manifest["inputs"]}
    if groups.keys() != expected_groups:
        raise ValueError("corpus/base groups do not match manifest")
    results = []
    for name, base, tokens in manifest["inputs"]:
        rows = groups[(name, base)]
        rows.sort(key=lambda row: row["visible_length"])
        if len(rows) != len(tokens):
            raise ValueError("incomplete corpus")
        for length, (row, token) in enumerate(zip(rows, tokens), 1):
            if row["visible_length"] != length or row["token"] != token:
                raise ValueError("duplicate, missing, or mismatched row")
        comparisons = {}
        for pair in PAIRS:
            logits = [row[pair]["logits"] for row in rows]
            distributions = [row[pair]["distribution"] for row in rows]
            for metrics in logits + distributions:
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise ValueError("nonfinite metric")
            comparisons[pair] = {
                "max_abs": max(m["max_abs"] for m in logits),
                "pooled_rmse": math.sqrt(
                    sum(m["rmse"] ** 2 for m in logits) / len(rows)
                ),
                "max_centered_rmse": max(m["centered_rmse"] for m in distributions),
                "mean_kl_reference_to_actual": sum(
                    m["kl_reference_to_actual"] for m in distributions
                )
                / len(rows),
                "max_kl_reference_to_actual": max(
                    m["kl_reference_to_actual"] for m in distributions
                ),
                "mean_total_variation": sum(m["total_variation"] for m in distributions)
                / len(rows),
                "max_total_variation": max(m["total_variation"] for m in distributions),
                "top1_mismatch_lengths": [
                    row["visible_length"]
                    for row in rows
                    if row[pair]["logits"]["actual_top1"]
                    != row[pair]["logits"]["reference_top1"]
                ],
            }
        results.append(
            {
                "corpus": name,
                "base": base,
                "rows": len(rows),
                "comparisons": comparisons,
            }
        )
    return {
        "qualification": False,
        "pair_order": manifest["pair_order"],
        "corpora": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
