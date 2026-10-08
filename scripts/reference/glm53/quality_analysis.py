# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Map #12 quality comparison: verdicts for the Fast packed-prefill lineage
against Exact, from `glm5_next_metal::tests::quality::quality_cohort_evaluate`.

PREREGISTERED (2026-10-08, committed with `quality-v1.json` before any GPU
run; do not edit the limits after seeing results):

Primary measure, per item (one document at one prefix length): the mean
negative log-likelihood (nats/token) of the 64 real tokens that follow the
prefix, with only the prefix read differently (Exact, Fast at 512 rows, Fast
at 97 rows) and the 64 tokens fed one at a time identically in every arm.
delta = mean NLL(arm) - mean NLL(Exact); positive means the arm is worse.

Uncertainty: bootstrap over documents (the two prefix lengths of a document
move together), 20,000 resamples, seed 20261008, stratified by stratum for
the overall estimate. Each item weighs equally.

Limits (non-inferiority margins):
- overall: the one-sided 95% upper bound of mean delta <= 0.005 nats/token
  (about a 0.5% perplexity increase);
- each stratum (prose, code, long past the sparse frontier): upper bound
  <= 0.010 nats/token;
- top-1 accuracy on the real next token, overall: the one-sided 95% lower
  bound of (arm - Exact) >= -0.01 (one point);
- tool tasks (12 tasks, low power, descriptive): greedy correct calls of the
  arm >= Exact's minus 1; sampled correct rate of the arm >= Exact's - 0.10.
Verdict per limit: PASS if the bound meets it; FAIL if even the opposite
95% bound is beyond it (confidently worse than the margin); otherwise
INCONCLUSIVE. Fast qualifies on quality only if every limit for Fast at 512
rows (the serve default) is PASS. Fast at 97 rows is reported, not gated.

Why these margins: the cost of the faster path should be small next to the
perplexity differences usually reported between adjacent quantization levels
(on the order of 1-3%), so 0.5% overall; strata get twice that because they
are smaller. Reference noise: Fast at 97 vs Fast at 512 rows (batching alone)
is reported with the same statistics for scale.

  uv run scripts/reference/glm53/quality_analysis.py REPORT.json [--out verdict.json]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OVERALL_NLL_MARGIN = 0.005
STRATUM_NLL_MARGIN = 0.010
TOP1_MARGIN = -0.01
TOOL_GREEDY_SLACK = 1
TOOL_SAMPLED_SLACK = 0.10
RESAMPLES = 20_000
SEED = 20261008
CONTINUATION = 64


def bootstrap(by_doc, strata_of, stratified, statistic):
    """Bootstrap distribution of `statistic(items)` resampling documents."""
    rng = np.random.default_rng(SEED)
    docs = list(by_doc)
    groups = defaultdict(list)
    for d in docs:
        groups[strata_of[d] if stratified else "all"].append(d)
    out = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        values = []
        for members in groups.values():
            picks = rng.integers(0, len(members), len(members))
            for p in picks:
                values.extend(by_doc[members[p]])
        out[r] = statistic(np.array(values))
    return out


def verdict_upper(samples, margin):
    upper, lower = np.percentile(samples, 95), np.percentile(samples, 5)
    status = (
        "PASS" if upper <= margin else ("FAIL" if lower > margin else "INCONCLUSIVE")
    )
    return {
        "mean": float(samples.mean()),
        "lower95": float(lower),
        "upper95": float(upper),
        "margin": margin,
        "verdict": status,
    }


def verdict_lower(samples, margin):
    upper, lower = np.percentile(samples, 95), np.percentile(samples, 5)
    status = (
        "PASS" if lower >= margin else ("FAIL" if upper < margin else "INCONCLUSIVE")
    )
    return {
        "mean": float(samples.mean()),
        "lower95": float(lower),
        "upper95": float(upper),
        "margin": margin,
        "verdict": status,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    strata_of = {}
    per_arm = {}
    for arm, base in (
        ("fast_512", "exact_512"),
        ("fast_97", "exact_512"),
        ("fast_97", "fast_512"),
    ):
        nll = defaultdict(list)
        top1 = defaultdict(list)
        for item in report["items"]:
            doc = item["path"]
            strata_of[doc] = item["stratum"]
            a, b = item["arms"][arm], item["arms"][base]
            nll[doc].append(a["mean_nll"] - b["mean_nll"])
            top1[doc].append((a["top1"] - b["top1"]) / CONTINUATION)
        key = f"{arm}_vs_{base}"
        result = {}
        overall = bootstrap(nll, strata_of, True, np.mean)
        result["overall_nll"] = verdict_upper(overall, OVERALL_NLL_MARGIN)
        result["overall_top1"] = verdict_lower(
            bootstrap(top1, strata_of, True, np.mean), TOP1_MARGIN
        )
        for stratum in sorted(set(strata_of.values())):
            sub = {d: v for d, v in nll.items() if strata_of[d] == stratum}
            result[f"{stratum}_nll"] = verdict_upper(
                bootstrap(sub, strata_of, False, np.mean), STRATUM_NLL_MARGIN
            )
        per_arm[key] = result

    tools = {}
    for arm in ("exact_512", "fast_512"):
        greedy = sampled = sampled_total = 0
        for task in report["tool_tasks"]:
            for row in task["arms"][arm]:
                if row["generation"]["mode"] == "greedy":
                    greedy += int(row["correct"])
                else:
                    sampled += int(row["correct"])
                    sampled_total += 1
        tools[arm] = {
            "greedy_correct": greedy,
            "greedy_total": len(report["tool_tasks"]),
            "sampled_correct": sampled,
            "sampled_total": sampled_total,
            "sampled_rate": sampled / sampled_total if sampled_total else None,
        }
    tool_verdict = {
        "greedy": "PASS"
        if tools["fast_512"]["greedy_correct"]
        >= tools["exact_512"]["greedy_correct"] - TOOL_GREEDY_SLACK
        else "FAIL",
        "sampled": "PASS"
        if tools["fast_512"]["sampled_rate"]
        >= tools["exact_512"]["sampled_rate"] - TOOL_SAMPLED_SLACK
        else "FAIL",
        "note": "12 tasks: low power, descriptive",
    }
    gated = per_arm["fast_512_vs_exact_512"]
    statuses = [v["verdict"] for v in gated.values()] + [
        tool_verdict["greedy"],
        tool_verdict["sampled"],
    ]
    overall = (
        "QUALIFIES"
        if all(s == "PASS" for s in statuses)
        else ("FAILS" if "FAIL" in statuses else "INCONCLUSIVE")
    )
    verdict = {
        "schema": "glm53.quality_verdict.v1",
        "fast_512_quality": overall,
        "comparisons": per_arm,
        "tools": tools,
        "tool_verdict": tool_verdict,
        "limits": {
            "overall_nll": OVERALL_NLL_MARGIN,
            "stratum_nll": STRATUM_NLL_MARGIN,
            "top1": TOP1_MARGIN,
            "tool_greedy_slack": TOOL_GREEDY_SLACK,
            "tool_sampled_slack": TOOL_SAMPLED_SLACK,
        },
        "evaluator_commit": report.get("evaluator_commit"),
        "fixture_sha256": report.get("fixture_sha256"),
    }
    for key, result in per_arm.items():
        print(key)
        for name, v in result.items():
            print(
                f"  {name:12} mean {v['mean']:+.5f}  95% [{v['lower95']:+.5f}, {v['upper95']:+.5f}]  margin {v['margin']:+.3f}  {v['verdict']}"
            )
    print("tools", json.dumps(tools), json.dumps(tool_verdict))
    print("Fast at 512 rows on quality:", overall)
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=2) + "\n")


if __name__ == "__main__":
    main()
