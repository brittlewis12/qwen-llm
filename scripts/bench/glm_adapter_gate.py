#!/usr/bin/env python3
"""Summarize the GLM explicit-adapter route gate.

Reads ``kind=route`` JSON lines from
``glm_small_append_price <corpus> routes <depth>``. For each depth and width it
pairs the reference (rollback) arm with the candidate production arm and
reports the ordinary-wall and routed-expert-stage difference. Positive means
the candidate is faster/smaller. The interval is a 20,000-resample bootstrap
over the three paired interleaved blocks, so it describes this packet rather
than a population.

The route sweep reports ``state_bitwise`` and ``max_abs_logit_diff`` against
its own incumbent, which on the fixed binary is the production selection.

Usage:
  glm_adapter_gate.py --reference expanded_tensor --candidate incumbent routes-*.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

BOOTSTRAP_SAMPLES = 20_000
SEED = 0x474C4D41


def quantile(sorted_values: list[float], p: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    at = (len(sorted_values) - 1) * p
    lo = int(at)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = at - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bootstrap(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"mean": mean, "ci90": [mean, mean], "samples": 1}
    rng = random.Random(SEED + len(values))
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return {
        "mean": mean,
        "ci90": [quantile(means, 0.05), quantile(means, 0.95)],
        "samples": n,
    }


def load(paths: list[Path]) -> dict[tuple[int, int, str], dict]:
    rows: dict[tuple[int, int, str], dict] = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: {error}") from error
            if row.get("kind") != "route":
                continue
            key = (int(row["depth"]), int(row["width"]), str(row["arm"]))
            if key in rows:
                raise SystemExit(f"{path}:{line_number}: duplicate route {key}")
            rows[key] = row
    if not rows:
        raise SystemExit("no kind=route rows")
    return rows


SCALAR_FALLBACK = {
    "routed_experts_ms_samples": "routed_experts_ms",
    "staged_walls_ms": "staged_wall_ms",
}


def samples(row: dict, field: str) -> list[float]:
    """Older probe builds report one sample under a scalar field."""
    if field in row:
        return row[field]
    scalar = SCALAR_FALLBACK.get(field)
    if scalar and scalar in row:
        return [row[scalar]]
    raise SystemExit(f"missing {field}")


def paired(reference: dict, candidate: dict, field: str) -> dict:
    a = samples(reference, field)
    b = samples(candidate, field)
    if len(a) != len(b):
        raise SystemExit(f"{field}: unpaired sample count")
    return bootstrap([x - y for x, y in zip(a, b)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="expanded_tensor")
    parser.add_argument("--candidate", default="incumbent")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    rows = load(args.paths)
    keys = sorted({(depth, width) for depth, width, _ in rows})
    out = []
    for depth, width in keys:
        reference = rows.get((depth, width, args.reference))
        candidate = rows.get((depth, width, args.candidate))
        if reference is None or candidate is None:
            continue
        wall = paired(reference, candidate, "walls_ms")
        stage = paired(reference, candidate, "routed_experts_ms_samples")
        staged = paired(reference, candidate, "staged_walls_ms")
        ref_wall = statistics.median(reference["walls_ms"])
        cand_wall = statistics.median(candidate["walls_ms"])
        ref_stage = statistics.median(samples(reference, "routed_experts_ms_samples"))
        cand_stage = statistics.median(samples(candidate, "routed_experts_ms_samples"))
        out.append(
            {
                "depth": depth,
                "width": width,
                "reference": args.reference,
                "candidate": args.candidate,
                "reference_wall_ms": ref_wall,
                "candidate_wall_ms": cand_wall,
                "wall_speedup": ref_wall / cand_wall,
                "wall_saved_ms": wall["mean"],
                "wall_saved_ci90": wall["ci90"],
                "wall_ci_excludes_zero": wall["ci90"][0] > 0.0,
                "stage_speedup": ref_stage / cand_stage,
                "stage_saved_ms": stage["mean"],
                "stage_saved_ci90": stage["ci90"],
                "staged_wall_saved_ms": staged["mean"],
                "staged_wall_saved_ci90": staged["ci90"],
                "candidate_state_bitwise": candidate.get("state_bitwise"),
                "candidate_max_abs_logit_diff": candidate["max_abs_logit_diff"],
                "reference_max_abs_logit_diff": reference["max_abs_logit_diff"],
                "candidate_staged_wall_ms": statistics.median(
                    samples(candidate, "staged_walls_ms")
                ),
                "reference_staged_wall_ms": statistics.median(
                    samples(reference, "staged_walls_ms")
                ),
                "pressure": max(reference["pressure"], candidate["pressure"]),
            }
        )
    if not out:
        raise SystemExit("no paired widths")
    for row in out:
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
