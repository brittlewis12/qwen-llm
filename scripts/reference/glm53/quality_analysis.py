# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Map #12 quality comparison: verdicts for the Fast packed-prefill lineage
against Exact, from `glm5_next_metal::tests::quality::quality_cohort_evaluate`.

PREREGISTERED (2026-10-08, committed with `quality-v1.json` before any GPU
run). AMENDED 2026-10-08 after the cx 01a10cc preregistration review and
before any result was viewed: separate text and tool verdicts; stricter tool
scoring done here from the recorded calls; per-task weighting of sampled
tool results; report-completeness checks; honest interval labels and
rationale. The numeric limits are unchanged.

Text cohort (the statistical qualification). Per item (one repository
document at one prefix length): the mean negative log-likelihood
(nats/token) of the 64 real tokens after the prefix, with only the prefix
read differently (Exact, Fast at 512 rows, Fast at 97 rows) and the 64
tokens fed one at a time identically in every arm. delta = mean NLL(arm) -
mean NLL(Exact); positive means the arm is worse. Uncertainty: bootstrap
over documents (a document's prefix lengths move together), 20,000
resamples, seed 20261008, stratified by stratum overall. Items weigh
equally (overall weights: prose 16/38, code 16/38, long 6/38).

Limits (conservative policy choices, not established quality tolerances):
- overall mean delta: one-sided 95% upper bound <= 0.005 nats/token
  (a 0.50% perplexity increase);
- each stratum (prose, code, long past the sparse frontier): upper bound
  <= 0.010; the wider stratum margin allows for fewer documents, not more
  acceptable degradation;
- top-1 accuracy on the real next token, overall: one-sided 95% lower bound
  of (arm - Exact) >= -0.01 (one point).
Verdict per limit: PASS if the bound meets it; FAIL if even the opposite
one-sided 95% bound is beyond it; otherwise INCONCLUSIVE. The printed
interval is the central 90% (two one-sided 95% bounds); the printed mean is
the observed mean. Fast at 512 rows (serve's default) is gated; Fast at 97
rows and Fast 97 vs 512 (batching alone) are reported for scale.

Tool screen (finite, descriptive, low power; never a non-inferiority claim):
12 tasks with known calls, scored here from the recorded calls: exactly one
call; the expected tool; weather: the city equals the expected city after
normalization (case, accents, a trailing ", country"), the T03 request has
days == 3, and any supplied days is an integer in 1..=7; currency: a JSON
number equal to the amount, and from/to as ISO codes or unambiguous names
(bare "dollars", "francs" or "pounds" are wrong). A generation that did not
end at <|observation|> or did not parse is wrong. Greedy: correct tasks per
arm. Sampled: per task, the mean over its seeds; then the mean over tasks
(short and long-context groups also reported). Screen verdict: UNINFORMATIVE
if Exact's greedy correct count is below 9 of 12; otherwise PASS if Fast's
greedy count >= Exact's - 1 and Fast's sampled mean >= Exact's - 0.10, else
FAIL.

A combined pass reads "passes this text cohort and the tool screen". It
measures continuation prediction on this repository's documents, not human
chat or general agent quality, and does not settle reuse consistency.

  uv run scripts/reference/glm53/quality_analysis.py REPORT.json [--fixture F] [--out verdict.json]
"""

import argparse
import hashlib
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OVERALL_NLL_MARGIN = 0.005
STRATUM_NLL_MARGIN = 0.010
TOP1_MARGIN = -0.01
TOOL_GREEDY_SLACK = 1
TOOL_SAMPLED_SLACK = 0.10
TOOL_EXACT_FLOOR = 9
RESAMPLES = 20_000
SEED = 20261008
CONTINUATION = 64
ARMS = ("exact_512", "fast_512", "fast_97")
CURRENCY = {
    "usd": "USD",
    "us dollar": "USD",
    "us dollars": "USD",
    "united states dollar": "USD",
    "united states dollars": "USD",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "gbp": "GBP",
    "british pound": "GBP",
    "british pounds": "GBP",
    "pound sterling": "GBP",
    "jpy": "JPY",
    "yen": "JPY",
    "japanese yen": "JPY",
    "chf": "CHF",
    "swiss franc": "CHF",
    "swiss francs": "CHF",
    "cad": "CAD",
    "canadian dollar": "CAD",
    "canadian dollars": "CAD",
    "aud": "AUD",
    "australian dollar": "AUD",
    "australian dollars": "AUD",
}


def norm_city(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return text.split(",")[0].strip().lower()


def tool_correct(row, expected, task_id):
    if row.get("parse_error") or row.get("stop") != 154829:
        return False
    calls = row.get("calls") or []
    if len(calls) != 1:
        return False
    call, args = calls[0], calls[0]["arguments"]
    if call["name"] != expected["name"]:
        return False
    if expected["name"] == "get_weather":
        if norm_city(args.get("city", "")) != expected["city"].lower():
            return False
        days = args.get("days")
        if days is not None and not (
            isinstance(days, int) and not isinstance(days, bool) and 1 <= days <= 7
        ):
            return False
        if task_id == "T03" and days != 3:
            return False
        return True
    amount = args.get("amount")
    if not (
        isinstance(amount, (int, float))
        and not isinstance(amount, bool)
        and math.isclose(amount, expected["amount"], abs_tol=1e-9)
    ):
        return False
    code = lambda v: CURRENCY.get(str(v).strip().lower())
    return (
        code(args.get("from")) == expected["from"]
        and code(args.get("to")) == expected["to"]
    )


def validate(report, fixture, fixture_bytes):
    assert report["fixture_sha256"] == hashlib.sha256(fixture_bytes).hexdigest(), (
        "fixture hash"
    )
    expected = {
        (d["path"], L): d["stratum"]
        for d in fixture["documents"]
        for L in d["prefix_lengths"]
    }
    assert len(expected) == 38, len(expected)
    seen = {}
    for item in report["items"]:
        key = (item["path"], item["prefix"])
        assert key in expected and key not in seen, key
        assert item["stratum"] == expected[key], key
        seen[key] = True
        for arm in ARMS:
            a = item["arms"][arm]
            assert len(a["nll"]) == CONTINUATION and all(
                math.isfinite(x) for x in a["nll"]
            ), (key, arm)
            assert 0 <= a["top1"] <= CONTINUATION, (key, arm)
            assert math.isclose(
                a["mean_nll"], sum(a["nll"]) / CONTINUATION, rel_tol=1e-9
            ), (key, arm)
    assert len(seen) == len(expected), "missing items"
    tasks = {t["id"]: t for t in fixture["tool_tasks"]}
    assert sorted(t["id"] for t in report["tool_tasks"]) == sorted(tasks), "tool tasks"
    for task in report["tool_tasks"]:
        long = tasks[task["id"]]["context_level"] > 0
        want = ["greedy", "sampled"] + ([] if long else ["sampled", "sampled"])
        seeds = fixture["tool_seeds"][: 1 if long else 3]
        for arm in ("exact_512", "fast_512"):
            rows = task["arms"][arm]
            assert [r["generation"]["mode"] for r in rows] == want, (task["id"], arm)
            assert [
                r["generation"]["seed"]
                for r in rows
                if r["generation"]["mode"] == "sampled"
            ] == seeds


def bootstrap(by_doc, strata_of, stratified, statistic):
    rng = np.random.default_rng(SEED)
    groups = defaultdict(list)
    for d in by_doc:
        groups[strata_of[d] if stratified else "all"].append(d)
    out = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        values = []
        for members in groups.values():
            for p in rng.integers(0, len(members), len(members)):
                values.extend(by_doc[members[p]])
        out[r] = statistic(np.array(values))
    return out


def judge(observed, samples, margin, upper_is_bad):
    lower, upper = float(np.percentile(samples, 5)), float(np.percentile(samples, 95))
    if upper_is_bad:
        status = (
            "PASS"
            if upper <= margin
            else ("FAIL" if lower > margin else "INCONCLUSIVE")
        )
    else:
        status = (
            "PASS"
            if lower >= margin
            else ("FAIL" if upper < margin else "INCONCLUSIVE")
        )
    return {
        "observed_mean": observed,
        "central90": [lower, upper],
        "margin": margin,
        "verdict": status,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--fixture", default=str(HERE / "quality-v1.json"))
    parser.add_argument("--out")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    fixture_bytes = Path(args.fixture).read_bytes()
    fixture = json.loads(fixture_bytes)
    validate(report, fixture, fixture_bytes)

    strata_of = {}
    comparisons = {}
    for arm, base in (
        ("fast_512", "exact_512"),
        ("fast_97", "exact_512"),
        ("fast_97", "fast_512"),
    ):
        nll, top1 = defaultdict(list), defaultdict(list)
        for item in report["items"]:
            doc = item["path"]
            strata_of[doc] = item["stratum"]
            a, b = item["arms"][arm], item["arms"][base]
            nll[doc].append(a["mean_nll"] - b["mean_nll"])
            top1[doc].append((a["top1"] - b["top1"]) / CONTINUATION)
        flat = lambda d: float(np.mean([v for vs in d.values() for v in vs]))
        result = {
            "overall_nll": judge(
                flat(nll),
                bootstrap(nll, strata_of, True, np.mean),
                OVERALL_NLL_MARGIN,
                True,
            ),
            "overall_top1": judge(
                flat(top1),
                bootstrap(top1, strata_of, True, np.mean),
                TOP1_MARGIN,
                False,
            ),
        }
        for stratum in sorted(set(strata_of.values())):
            sub = {d: v for d, v in nll.items() if strata_of[d] == stratum}
            result[f"{stratum}_nll"] = judge(
                flat(sub),
                bootstrap(sub, strata_of, False, np.mean),
                STRATUM_NLL_MARGIN,
                True,
            )
        comparisons[f"{arm}_vs_{base}"] = result

    tasks = {t["id"]: t for t in fixture["tool_tasks"]}
    tools = {}
    for arm in ("exact_512", "fast_512"):
        greedy, per_task, groups = 0, {}, defaultdict(list)
        for task in report["tool_tasks"]:
            expected = tasks[task["id"]]["expected"]
            rows = task["arms"][arm]
            scored = [tool_correct(r, expected, task["id"]) for r in rows]
            greedy += int(scored[0])
            sampled = [
                s for r, s in zip(rows, scored) if r["generation"]["mode"] == "sampled"
            ]
            per_task[task["id"]] = float(np.mean(sampled))
            groups[
                "long" if tasks[task["id"]]["context_level"] > 0 else "short"
            ].append(per_task[task["id"]])
        tools[arm] = {
            "greedy_correct": greedy,
            "greedy_total": len(per_task),
            "sampled_mean": float(np.mean(list(per_task.values()))),
            "sampled_by_group": {g: float(np.mean(v)) for g, v in groups.items()},
            "sampled_by_task": per_task,
        }
    e, f = tools["exact_512"], tools["fast_512"]
    if e["greedy_correct"] < TOOL_EXACT_FLOOR:
        screen = "UNINFORMATIVE"
    elif (
        f["greedy_correct"] >= e["greedy_correct"] - TOOL_GREEDY_SLACK
        and f["sampled_mean"] >= e["sampled_mean"] - TOOL_SAMPLED_SLACK
    ):
        screen = "PASS"
    else:
        screen = "FAIL"

    gated = comparisons["fast_512_vs_exact_512"]
    statuses = [v["verdict"] for v in gated.values()]
    text = (
        "QUALIFIES"
        if all(s == "PASS" for s in statuses)
        else ("FAILS" if "FAIL" in statuses else "INCONCLUSIVE")
    )
    verdict = {
        "schema": "glm53.quality_verdict.v2",
        "text_cohort_fast_512": text,
        "tool_screen": screen,
        "comparisons": comparisons,
        "tools": tools,
        "limits": {
            "overall_nll": OVERALL_NLL_MARGIN,
            "stratum_nll": STRATUM_NLL_MARGIN,
            "top1": TOP1_MARGIN,
            "tool_greedy_slack": TOOL_GREEDY_SLACK,
            "tool_sampled_slack": TOOL_SAMPLED_SLACK,
            "tool_exact_floor": TOOL_EXACT_FLOOR,
        },
        "evaluator_commit": report.get("evaluator_commit"),
        "fixture_sha256": report.get("fixture_sha256"),
    }
    for key, result in comparisons.items():
        print(key)
        for name, v in result.items():
            lo, hi = v["central90"]
            print(
                f"  {name:12} observed {v['observed_mean']:+.5f}  central 90% [{lo:+.5f}, {hi:+.5f}]  margin {v['margin']:+.3f}  {v['verdict']}"
            )
    print(
        "tools",
        json.dumps(
            {
                a: {
                    k: t[k]
                    for k in ("greedy_correct", "sampled_mean", "sampled_by_group")
                }
                for a, t in tools.items()
            }
        ),
    )
    print("text cohort (Fast 512):", text, "| tool screen:", screen)
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=2) + "\n")


if __name__ == "__main__":
    main()
