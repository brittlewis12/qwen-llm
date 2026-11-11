#!/usr/bin/env python3
"""Summarize a small GLM packed-append non-inferiority cohort JSONL."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

BOOTSTRAP_SAMPLES = 20_000
SEED = 0x474C4D53
ARMS = (
    "sequential_fast_half",
    "packed_fast_half",
    "packed_dense_f32_suffix",
    "packed_f32_suffix",
)
WIDTHS = (8, 16, 32, 64)


def quantile(sorted_values: list[float], p: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    at = (len(sorted_values) - 1) * p
    lo = int(at)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = at - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bootstrap(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("empty prompt cohort")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {
            "mean": mean,
            "ci90": [mean, mean],
            "upper95": mean,
            "prompts": 1,
            "ci_note": "one prompt; interval is descriptive only",
        }
    rng = random.Random(SEED + len(values))
    n = len(values)
    means = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    means.sort()
    return {
        "mean": mean,
        "ci90": [quantile(means, 0.05), quantile(means, 0.95)],
        "upper95": quantile(means, 0.95),
        "prompts": n,
        "ci_note": "cluster bootstrap over prompts, 20,000 resamples",
    }


def load(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                if row.get("kind") == "quality_result":
                    rows.append(row)
    if not rows:
        raise ValueError("no quality_result rows found")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = sorted({row["prompt"] for row in rows})
    by_key = {(row["prompt"], row["width"], row["arm"]): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("duplicate prompt/width/arm result")
    strata = sorted({row["stratum"] for row in rows})
    out = []
    for width in WIDTHS:
        for stratum in strata + ["all"]:
            members = [
                prompt
                for prompt in prompts
                if stratum == "all"
                or next(row["stratum"] for row in rows if row["prompt"] == prompt) == stratum
            ]
            if not members:
                continue
            arm_summaries = {}
            for arm in ARMS:
                selected = [by_key[(prompt, width, arm)] for prompt in members]
                dnll = bootstrap([row["mean_dnll_vs_reference"] for row in selected])
                arm_summaries[arm] = {
                    "dnll_vs_reference": dnll,
                    "mean_kl_reference_to_arm": statistics.fmean(
                        row["mean_kl_reference_to_arm"] for row in selected
                    ),
                    "mean_kl_arm_to_reference": statistics.fmean(
                        row["mean_kl_arm_to_reference"] for row in selected
                    ),
                    "top1_rate": statistics.fmean(row["top1_rate"] for row in selected),
                    "per_prompt": {
                        row["prompt"]: {
                            "mean_dnll_vs_reference": row["mean_dnll_vs_reference"],
                            "reference_kind": row["reference_kind"],
                            "mean_dnll_vs_sequential_fast": row[
                                "mean_dnll_vs_sequential_fast"
                            ],
                            "mean_kl_reference_to_arm": row[
                                "mean_kl_reference_to_arm"
                            ],
                            "top1_rate": row["top1_rate"],
                            "route_set_changes": sum(
                                layer["changed_rows"]
                                for layer in row["router_set_changes_vs_reference"]
                            ),
                        }
                        for row in selected
                    },
                }

            deltas = [
                by_key[(prompt, width, "packed_fast_half")][
                    "mean_dnll_vs_sequential_fast"
                ]
                for prompt in members
            ]
            paired = bootstrap(deltas)
            threshold = 0.005 if stratum == "all" else 0.010
            reference_kinds = sorted({row["reference_kind"] for row in rows if row["prompt"] in members})
            exact_reference_available = reference_kinds == ["Exact"]
            out.append(
                {
                    "stratum": stratum,
                    "reference_kinds": reference_kinds,
                    "exact_reference_available": exact_reference_available,
                    "width": width,
                    "prompts": members,
                    "noninferiority_limit_nats_per_token": threshold,
                    "arms": arm_summaries,
                    "packed_half_minus_sequential_fast": paired,
                    "packed_half_minus_sequential_ci90_includes_zero": (
                        paired["ci90"][0] <= 0.0 <= paired["ci90"][1]
                    ),
                    "packed_half_noninferior_to_sequential_at_policy_margin": (
                        paired["upper95"] <= threshold
                    ),
                    "packed_half_exact_95pct_upper_passes_policy_limit": (
                        arm_summaries["packed_fast_half"]["dnll_vs_reference"]["upper95"]
                        <= threshold if exact_reference_available else None
                    ),
                }
            )
    return {
        "schema": "glm53.append_noninferiority_summary.v1",
        "estimand": "equal-prompt mean of per-prompt mean signed dNLL; positive is worse than the declared reference",
        "ci": "central 90% cluster bootstrap over prompts; one-sided 95% upper uses percentile 95",
        "quality_v1_limits_nats_per_token": {"overall": 0.005, "stratum": 0.010},
        "prompts": prompts,
        "results": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = summarize(load(args.jsonl))
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(data)
    print(data, end="")


if __name__ == "__main__":
    main()
