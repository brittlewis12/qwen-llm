#!/usr/bin/env python3
"""Summarize the GLM append route sweep and traffic-weight the policy.

Reads the JSON lines written by
``glm_small_append_price <corpus> routes <depth>``. For each depth and width it
picks the cheapest legal routed-expert route by ordinary warm wall, then maps a
real append-width distribution onto the measured grid and reports the
traffic-weighted append wall of the incumbent against the implied policy.

The grid mapping is deliberately bucketed, not interpolated: the cost is a
kernel-switch staircase, so an off-grid width uses the nearest measured width.
Widths above the grid keep the incumbent policy (no route change is claimed).

Usage:
  glm_append_routes.py routes-48k.jsonl routes-80k.jsonl
  glm_append_routes.py --calls-csv steady-calls.csv --cell glm-default routes-80k.jsonl
  glm_append_routes.py --weights 32,45,67,184 routes-80k.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ARM_ORDER = (
    "incumbent",
    "per_slot_native",
    "grouped_generic",
    "expanded_tensor",
    "tensor_iq",
)


def parse_routes(paths: list[Path]) -> dict[tuple[int, int, str], dict]:
    rows: dict[tuple[int, int, str], dict] = {}
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{path}:{line_number}: {error}") from error
                if row.get("kind") != "route":
                    continue
                key = (int(row["depth"]), int(row["width"]), row["arm"])
                if key in rows:
                    raise SystemExit(f"{path}:{line_number}: duplicate route {key}")
                rows[key] = row
    if not rows:
        raise SystemExit("no kind=route rows")
    return rows


def widths_at(rows: dict[tuple[int, int, str], dict], depth: int) -> list[int]:
    return sorted({width for (d, width, _) in rows if d == depth})


def arms_at(
    rows: dict[tuple[int, int, str], dict], depth: int, width: int
) -> list[str]:
    present = {arm for (d, w, arm) in rows if d == depth and w == width}
    return [arm for arm in ARM_ORDER if arm in present]


def cheapest(
    rows: dict[tuple[int, int, str], dict], depth: int, width: int
) -> tuple[str, dict]:
    arms = arms_at(rows, depth, width)
    return min(
        ((arm, rows[(depth, width, arm)]) for arm in arms),
        key=lambda x: x[1]["wall_ms"],
    )


def policy_ranges(
    rows: dict[tuple[int, int, str], dict], depth: int
) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for width in widths_at(rows, depth):
        arm, _ = cheapest(rows, depth, width)
        if out and out[-1][2] == arm:
            out[-1] = (out[-1][0], width, arm)
        else:
            out.append((width, width, arm))
    return out


def grid_for(rows: dict[tuple[int, int, str], dict], depth: int) -> list[int]:
    return widths_at(rows, depth)


def nearest(grid: list[int], width: int) -> int | None:
    if width > grid[-1]:
        return None
    return min(grid, key=lambda candidate: (abs(candidate - width), candidate))


def load_weights(args) -> list[int]:
    if args.weights:
        return [int(item) for item in args.weights.split(",") if item.strip()]
    rows = []
    with Path(args.calls_csv).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if args.cell and row.get("cell") != args.cell:
                continue
            value = row.get("new") or row.get("append") or row.get("append_tokens")
            try:
                width = int(float(value))
            except (TypeError, ValueError):
                continue
            if width > 0:
                rows.append(width)
    if not rows:
        raise SystemExit("no positive append widths in the calls CSV")
    return rows


def weighted(rows, depth: int, weights: list[int]) -> dict:
    grid = grid_for(rows, depth)
    incumbent_total = candidate_total = 0.0
    used = 0
    capped = 0
    by_arm: dict[str, int] = defaultdict(int)
    for width in weights:
        measured = nearest(grid, width)
        if measured is None or width < grid[0]:
            # Above the grid, and below the smallest measured width, the
            # policy is unchanged; charge both arms the incumbent cost.
            arm = "incumbent"
            measured = grid[-1] if measured is None else measured
            chosen_ms = rows[(depth, measured, "incumbent")]["wall_ms"]
            capped += 1
        else:
            arm, chosen = cheapest(rows, depth, measured)
            chosen_ms = chosen["wall_ms"]
        incumbent = rows[(depth, measured, "incumbent")]["wall_ms"]
        incumbent_total += incumbent
        candidate_total += chosen_ms
        by_arm[arm] += 1
        used += 1
    return {
        "n": used,
        "above_grid": capped,
        "incumbent_ms": incumbent_total / used,
        "candidate_ms": candidate_total / used,
        "reduction_pct": 100.0 * (incumbent_total - candidate_total) / incumbent_total,
        "by_arm": dict(by_arm),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("routes", nargs="+", type=Path)
    parser.add_argument("--weights", help="explicit comma-separated append widths")
    parser.add_argument(
        "--calls-csv", type=Path, help="CSV with a per-call append width column"
    )
    parser.add_argument("--cell", help="restrict --calls-csv rows to this cell value")
    args = parser.parse_args()

    rows = parse_routes(args.routes)
    depths = sorted({depth for (depth, _, _) in rows})

    for depth in depths:
        print(f"## Route sweep at depth {depth}\n")
        print(
            "| width | incumbent ms | cheapest | cheapest ms | saving | incumbent routed ms | cheapest routed ms |"
        )
        print("| ---: | ---: | --- | ---: | ---: | ---: | ---: |")
        for width in widths_at(rows, depth):
            incumbent = rows[(depth, width, "incumbent")]
            arm, chosen = cheapest(rows, depth, width)
            saving = (
                100.0
                * (incumbent["wall_ms"] - chosen["wall_ms"])
                / incumbent["wall_ms"]
            )
            print(
                f"| {width} | {incumbent['wall_ms']:.1f} | {arm} | "
                f"{chosen['wall_ms']:.1f} | {saving:.1f}% | "
                f"{incumbent.get('routed_experts_ms') or float('nan'):.1f} | "
                f"{chosen.get('routed_experts_ms') or float('nan'):.1f} |"
            )
        print()
        print(
            "Implied policy:",
            ", ".join(
                f"{lo}..{hi} {arm}" for lo, hi, arm in policy_ranges(rows, depth)
            ),
        )
        print()

    if args.weights or args.calls_csv:
        weights = load_weights(args)
        weights.sort()
        print("## Traffic-weighted append wall\n")
        print(
            f"Distribution: n={len(weights)} median={statistics.median(weights):.0f} "
            f"mean={statistics.mean(weights):.1f} p90={weights[9 * len(weights) // 10]} max={weights[-1]}"
        )
        print()
        print("| depth | incumbent ms | candidate ms | reduction | routes used |")
        print("| ---: | ---: | ---: | ---: | --- |")
        for depth in depths:
            result = weighted(rows, depth, weights)
            used = ", ".join(
                f"{arm}:{count}" for arm, count in sorted(result["by_arm"].items())
            )
            print(
                f"| {depth} | {result['incumbent_ms']:.1f} | {result['candidate_ms']:.1f} | "
                f"{result['reduction_pct']:.2f}% | {used} |"
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
