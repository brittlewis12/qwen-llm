#!/usr/bin/env python3
"""Summarize the GLM route-change non-inferiority cohort.

Reads the ``kind=route_quality`` JSON lines written by
``glm_small_append_price <prompt> route-quality <depth>``. Each input file is
one prompt cluster. For every width and forced arm it reports the equal-prompt
mean signed dNLL of the candidate against the incumbent, a prompt-cluster
bootstrap interval and the one-sided 95% upper bound. Positive dNLL means the
candidate gives the actual next token a lower log-probability.

Usage:
  glm_route_quality.py --candidate tensor_iq --margin 0.01 quality-*.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

BOOTSTRAP_SAMPLES = 20_000
SEED = 0x474C4D53


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
        return {"mean": mean, "ci90": [mean, mean], "upper95": mean, "prompts": 1}
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
    }


def load(paths: list[Path]) -> dict[tuple[int, int, str], dict]:
    rows = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: {error}") from error
            if row.get("kind") != "route_quality":
                continue
            key = (path.stem, int(row["width"]), row["arm"])
            if key in rows:
                raise SystemExit(f"{path}:{line_number}: duplicate {key}")
            rows[key] = row
    if not rows:
        raise SystemExit("no kind=route_quality rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--candidate", default="tensor_iq")
    parser.add_argument("--margin", type=float, default=0.01)
    args = parser.parse_args()

    rows = load(args.paths)
    prompts = sorted({prompt for (prompt, _, _) in rows})
    widths = sorted({width for (_, width, _) in rows})
    arms = sorted({arm for (_, _, arm) in rows})
    if args.candidate not in arms:
        raise SystemExit(f"candidate {args.candidate} absent; have {arms}")

    print(
        f"Cohort: {len(prompts)} prompts, widths {widths}, candidate {args.candidate}"
    )
    print(
        f"Non-inferiority margin: candidate minus incumbent mean dNLL <= {args.margin:+.4f}\n"
    )
    print(
        "| width | arm | mean dNLL | 90% CI | 95% upper | mean KL | top-1 | verdict |"
    )
    print("| ---: | --- | ---: | --- | ---: | ---: | ---: | --- |")
    verdicts = []
    for width in widths:
        for arm in arms:
            selected = [
                rows[(prompt, width, arm)]
                for prompt in prompts
                if (prompt, width, arm) in rows
            ]
            if len(selected) != len(prompts):
                continue
            dnll = [row["mean_dnll_vs_incumbent"] for row in selected]
            stats = bootstrap(dnll)
            kl = statistics.fmean(row["mean_kl_incumbent_arm"] for row in selected)
            top1 = statistics.fmean(
                row["top1_agree"] / row["positions"] for row in selected
            )
            verdict = "n/a"
            if arm == args.candidate:
                verdict = "PASS" if stats["upper95"] <= args.margin else "FAIL"
                verdicts.append((width, verdict, stats["upper95"]))
            print(
                f"| {width} | {arm} | {stats['mean']:+.5f} | "
                f"[{stats['ci90'][0]:+.5f}, {stats['ci90'][1]:+.5f}] | "
                f"{stats['upper95']:+.5f} | {kl:.2e} | {100 * top1:.1f}% | {verdict} |"
            )
    if verdicts:
        print()
        failed = [width for width, verdict, _ in verdicts if verdict == "FAIL"]
        print(
            "Candidate verdict: "
            + ("PASS at every width" if not failed else f"FAIL at widths {failed}")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
