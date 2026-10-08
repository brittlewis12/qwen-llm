# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Map #12/#15: verdicts for serve's Fast shared-prefix split
(fast_shared_split_v1) against Exact, from
`glm5_next_metal::tests::quality_split::quality_split_evaluate`.

PREREGISTERED (2026-10-08, committed with `quality-split-v1.json` before
any GPU run of the split arms). It reuses quality-v1's limits, bootstrap and
strict tool scoring (`quality_analysis.py`, imported unchanged).

The schedule: prefill [0, s) then [s, N), each in 512-row chunks from its
own start. Serve's hit (restore at s) is bit-identical to its miss, so
qualifying the miss qualifies both.

Text (gated). Per item, delta = mean NLL(arm) - mean NLL(exact_512) over the
64 real tokens after the prefix; quality-v1's 38 items, document bootstrap
(20,000 resamples, seed 20261008), equal item weights.
- fast_split_grid (cut at the deployed agent prefix's in-chunk offset) and
  fast_split_tail (a 17-20 token final segment): each must meet every
  quality-v1 limit against exact_512: overall mean NLL one-sided 95% upper
  bound <= 0.005; each stratum's upper bound <= 0.010; overall top-1 lower
  bound >= -0.01.
- fast_split_frontier (long items only; cuts 2050-2053 across the 2,052
  sparse frontier): long-stratum NLL upper bound <= 0.010 and long top-1
  lower bound >= -0.01 (bootstrap within the six long documents).
Text verdict: QUALIFIES if every gated limit of every split arm passes;
FAILS if any fails; otherwise INCONCLUSIVE. No arm is chosen after the fact.
Reported, not gated: each split arm against fast_512 (the schedule change
alone), fast_512 against exact_512 (the control), and whether this run's
exact_512 and fast_512 reproduce the committed quality-v1 report item by
item (exact equality of the recorded mean NLL, top-1 and per-token NLL;
not a logit-identity claim).

Tools (gated, quality-v1's screen). The 12 tasks split at their
renderer-recorded shared-prefix end (255 tokens; T09-T12 then read 1,200 or
4,750 more tokens in the second segment). Strict scoring as quality-v1:
UNINFORMATIVE if Exact's greedy count is below 9 of 12; otherwise PASS if
fast_split's greedy count >= Exact's - 1 and its sampled mean >= Exact's -
0.10, else FAIL.

Agent cases (descriptive, n = 4 x 2 generations per arm). opencode's
instructions and 12 tools (11,104-token shared prefix, cut there), effort
low. A generation is right if it ends at <|observation|> with a first call
of the expected tool whose argument contains the expected substring.
Reported per arm, with each generation's prompt-end KL from Exact. Flagged
for investigation (holds the decision until reviewed, changes no score):
parse errors in any arm. Invalid evidence (refused before any verdict): an
incomplete report, any mismatch with the frozen manifest (items, strata,
cuts, chunk starts, arms, modes, seeds, caps), non-finite metrics, or an
Exact-restore check that was not bitwise.

Decision rule. fast_shared_split_v1 may become serve's Fast default only if
the text verdict is QUALIFIES and the tool screen PASSES (and, separately,
after a release-build time-to-first-token screen). INCONCLUSIVE or
UNINFORMATIVE keeps it opt-in; neither is a failure diagnosis nor licence to
widen limits. The cohort is visible (quality-v1 was viewed before); it is
reused because the schedule is mechanically specified and untuned. Any
choice made from these results (cuts, precision, dispatch) needs a fresh
confirmatory cohort.

  uv run scripts/reference/glm53/quality_split_analysis.py REPORT.json \\
      [--manifest M] [--committed C] [--out verdict.json]
"""

import argparse
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("qa", HERE / "quality_analysis.py")
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)

COMMITTED = REPO / "docs/bench/2026-10-08-glm53-fast-quality/report.json"
GATED_FULL = ("fast_split_grid", "fast_split_tail")
FRONTIER = "fast_split_frontier"
TEXT_ARMS = ("exact_512", "fast_512") + GATED_FULL
# KL values are recorded raw; a computed KL may be a tiny negative number
# from floating-point roundoff when two distributions (nearly) coincide.
# Amended before any result was viewed (cx 01a10cc confirmation review).
KL_ROUNDOFF = -1e-9
TOOL_ARMS = ("exact_512", "fast_512", "fast_split")
AGENT_ARMS = ("exact", "fast_512", "fast_split")


def chunk_starts(length, cut, rows):
    if cut is None:
        return list(range(0, length, rows))
    return list(range(0, cut, rows)) + list(range(cut, length, rows))


def generation_modes(rows, seeds):
    return [r["generation"]["mode"] for r in rows], [
        r["generation"]["seed"] for r in rows if r["generation"]["mode"] == "sampled"
    ]


def validate(report, manifest, manifest_bytes, quality):
    """Refuses a report that does not match the frozen manifest exactly, or
    whose bitwise invariants or metrics are invalid (invalid evidence, before
    any quality verdict)."""
    assert report.get("complete") is True, "incomplete report"
    assert report["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest(), (
        "manifest hash"
    )
    rows = manifest["rows"]
    expected = {(i["path"], i["prefix"]): i for i in manifest["text_items"]}
    assert len(expected) == 38, len(expected)
    seen = set()
    for item in report["items"]:
        key = (item["path"], item["prefix"])
        assert key in expected and key not in seen, key
        seen.add(key)
        want = expected[key]
        assert item["stratum"] == want["stratum"], key
        assert item["doc_index"] == want["doc_index"], key
        arms = list(TEXT_ARMS) + ([FRONTIER] if want["cuts"][FRONTIER] else [])
        assert sorted(item["arms"]) == sorted(arms), (key, sorted(item["arms"]))
        for arm in arms:
            a = item["arms"][arm]
            cut = want["cuts"].get(arm)
            assert a["cut"] == cut, (key, arm)
            assert a["chunk_starts"] == chunk_starts(item["prefix"], cut, rows), (key, arm)
            assert len(a["nll"]) == qa.CONTINUATION and all(
                isinstance(x, float) and math.isfinite(x) for x in a["nll"]
            ), (key, arm)
            assert len(a["hits"]) == qa.CONTINUATION and all(
                isinstance(h, bool) for h in a["hits"]
            ), (key, arm)
            assert a["top1"] == sum(a["hits"]), (key, arm)
            assert math.isclose(
                a["mean_nll"], sum(a["nll"]) / qa.CONTINUATION, rel_tol=1e-9
            ), (key, arm)
            assert len(a["logits_sha256"]) == qa.CONTINUATION, (key, arm)
            if arm == "exact_512":
                assert a["kl_from_exact"] == [], key
            else:
                assert len(a["kl_from_exact"]) == qa.CONTINUATION and all(
                    math.isfinite(x) and x >= KL_ROUNDOFF for x in a["kl_from_exact"]
                ), (key, arm)
    assert len(seen) == 38, "missing items"

    quality_tasks = {t["id"]: t for t in quality["tool_tasks"]}
    cuts = {t["id"]: t["shared_prefix_tokens"] for t in manifest["tool_tasks"]}
    report_ids = [t["id"] for t in report["tool_tasks"]]
    assert sorted(report_ids) == sorted(cuts) and len(set(report_ids)) == len(report_ids)
    for task in report["tool_tasks"]:
        long = quality_tasks[task["id"]]["context_level"] > 0
        assert task["context_level"] == quality_tasks[task["id"]]["context_level"]
        assert task["cut"] == cuts[task["id"]], task["id"]
        want_modes = ["greedy", "sampled"] + ([] if long else ["sampled", "sampled"])
        want_seeds = manifest["tool_seeds"][: 1 if long else 3]
        assert sorted(task["arms"]) == sorted(TOOL_ARMS), task["id"]
        for arm in TOOL_ARMS:
            modes, seeds = generation_modes(task["arms"][arm], want_seeds)
            assert modes == want_modes and seeds == want_seeds, (task["id"], arm)
            assert all(
                r["generation"]["cap"] == manifest["tool_cap"] for r in task["arms"][arm]
            ), (task["id"], arm)

    agent = report["agent"]
    assert agent["exact_restore_equals_unsplit"] is True, (
        "invalid evidence: Exact restored at the shared cut is not bitwise equal to an unsplit Exact prefill"
    )
    frozen = {t["id"]: t for t in manifest["agent_tasks"]}
    agent_ids = [t["id"] for t in agent["tasks"]]
    assert sorted(agent_ids) == sorted(frozen) and len(set(agent_ids)) == len(agent_ids)
    for task in agent["tasks"]:
        assert task["shared_prefix_tokens"] == frozen[task["id"]]["shared_prefix_tokens"]
        assert task["prompt_tokens"] == frozen[task["id"]]["prompt_tokens"]
        assert sorted(task["arms"]) == sorted(AGENT_ARMS), task["id"]
        for arm in AGENT_ARMS:
            modes, seeds = generation_modes(task["arms"][arm], None)
            assert modes == ["greedy", "sampled"] and seeds == [manifest["agent_seed"]], (
                task["id"],
                arm,
            )
            for row in task["arms"][arm]:
                assert row["generation"]["cap"] == manifest["agent_cap"]
                kl = row["prompt_end_kl_from_exact"]
                assert isinstance(kl, float) and math.isfinite(kl) and kl >= KL_ROUNDOFF, (
                    "invalid evidence: non-finite prompt-end KL",
                    task["id"],
                    arm,
                )


def comparison(items, arm, base, strata_filter=None):
    nll, top1, strata_of = defaultdict(list), defaultdict(list), {}
    for item in items:
        if arm not in item["arms"] or base not in item["arms"]:
            continue
        if strata_filter and item["stratum"] not in strata_filter:
            continue
        doc = item["path"]
        strata_of[doc] = item["stratum"]
        a, b = item["arms"][arm], item["arms"][base]
        nll[doc].append(a["mean_nll"] - b["mean_nll"])
        top1[doc].append((a["top1"] - b["top1"]) / qa.CONTINUATION)
    flat = lambda d: float(np.mean([v for vs in d.values() for v in vs]))
    result = {}
    if not strata_filter:
        result["overall_nll"] = qa.judge(
            flat(nll),
            qa.bootstrap(nll, strata_of, True, np.mean),
            qa.OVERALL_NLL_MARGIN,
            True,
        )
        result["overall_top1"] = qa.judge(
            flat(top1),
            qa.bootstrap(top1, strata_of, True, np.mean),
            qa.TOP1_MARGIN,
            False,
        )
    for stratum in sorted(set(strata_of.values())):
        sub = {d: v for d, v in nll.items() if strata_of[d] == stratum}
        result[f"{stratum}_nll"] = qa.judge(
            flat(sub),
            qa.bootstrap(sub, strata_of, False, np.mean),
            qa.STRATUM_NLL_MARGIN,
            True,
        )
        if strata_filter:
            sub = {d: v for d, v in top1.items() if strata_of[d] == stratum}
            result[f"{stratum}_top1"] = qa.judge(
                flat(sub),
                qa.bootstrap(sub, strata_of, False, np.mean),
                qa.TOP1_MARGIN,
                False,
            )
    return result


def reproduces(items, committed):
    """Per arm: items whose recorded summary metrics (mean NLL, top-1) and
    per-token NLL exactly equal the committed quality-v1 report's. Equal
    recorded metrics are not a logit-identity claim; the logit hashes
    recorded here serve later identity claims."""
    old = {(i["path"], i["prefix"]): i["arms"] for i in committed["items"]}
    out = {}
    for arm in ("exact_512", "fast_512"):
        summary = per_token = 0
        for item in items:
            new, prev = item["arms"][arm], old[(item["path"], item["prefix"])][arm]
            summary += int(new["mean_nll"] == prev["mean_nll"] and new["top1"] == prev["top1"])
            per_token += int(new["nll"] == prev["nll"])
        out[arm] = {
            "equal_summary_metrics": summary,
            "equal_per_token_nll": per_token,
            "items": len(items),
        }
    return out


def tool_scores(report, quality, arm):
    tasks = {t["id"]: t for t in quality["tool_tasks"]}
    greedy, per_task, groups = 0, {}, defaultdict(list)
    for task in report["tool_tasks"]:
        expected = tasks[task["id"]]["expected"]
        rows = task["arms"][arm]
        scored = [qa.tool_correct(r, expected, task["id"]) for r in rows]
        greedy += int(scored[0])
        sampled = [
            s for r, s in zip(rows, scored) if r["generation"]["mode"] == "sampled"
        ]
        per_task[task["id"]] = float(np.mean(sampled))
        groups["long" if tasks[task["id"]]["context_level"] > 0 else "short"].append(
            per_task[task["id"]]
        )
    return {
        "greedy_correct": greedy,
        "greedy_total": len(per_task),
        "sampled_mean": float(np.mean(list(per_task.values()))),
        "sampled_by_group": {g: float(np.mean(v)) for g, v in groups.items()},
        "sampled_by_task": per_task,
    }


def agent_correct(row, expected):
    if row.get("parse_error") or row.get("stop") != 154829 or not row.get("calls"):
        return False
    call = row["calls"][0]
    value = call["arguments"].get(expected["argument"])
    return (
        call["name"] == expected["name"]
        and isinstance(value, str)
        and expected["contains"] in value
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--manifest", default=str(HERE / "quality-split-v1.json"))
    parser.add_argument("--quality", default=str(HERE / "quality-v1.json"))
    parser.add_argument("--committed", default=str(COMMITTED))
    parser.add_argument("--out")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
    quality_bytes = Path(args.quality).read_bytes()
    assert (
        hashlib.sha256(quality_bytes).hexdigest()
        == manifest["base"]["quality_v1"]["sha256"]
    )
    quality = json.loads(quality_bytes)
    committed_bytes = Path(args.committed).read_bytes()
    committed = json.loads(committed_bytes)
    validate(report, manifest, manifest_bytes, quality)
    items = report["items"]

    comparisons = {}
    for arm in GATED_FULL:
        comparisons[f"{arm}_vs_exact_512"] = comparison(items, arm, "exact_512")
    comparisons[f"{FRONTIER}_vs_exact_512"] = comparison(
        items, FRONTIER, "exact_512", {"long"}
    )
    reported = {
        "fast_512_vs_exact_512": comparison(items, "fast_512", "exact_512"),
        **{
            f"{arm}_vs_fast_512": comparison(items, arm, "fast_512")
            for arm in GATED_FULL
        },
        f"{FRONTIER}_vs_fast_512": comparison(items, FRONTIER, "fast_512", {"long"}),
    }
    statuses = [v["verdict"] for c in comparisons.values() for v in c.values()]
    text = (
        "QUALIFIES"
        if all(s == "PASS" for s in statuses)
        else ("FAILS" if "FAIL" in statuses else "INCONCLUSIVE")
    )

    tools = {arm: tool_scores(report, quality, arm) for arm in TOOL_ARMS}
    e, f = tools["exact_512"], tools["fast_split"]
    if e["greedy_correct"] < qa.TOOL_EXACT_FLOOR:
        screen = "UNINFORMATIVE"
    elif (
        f["greedy_correct"] >= e["greedy_correct"] - qa.TOOL_GREEDY_SLACK
        and f["sampled_mean"] >= e["sampled_mean"] - qa.TOOL_SAMPLED_SLACK
    ):
        screen = "PASS"
    else:
        screen = "FAIL"

    agent, flags = {}, []
    frozen = {t["id"]: t["expected"] for t in manifest["agent_tasks"]}
    for arm in AGENT_ARMS:
        rows = []
        for task in report["agent"]["tasks"]:
            for row in task["arms"][arm]:
                rows.append(
                    {
                        "task": task["id"],
                        "mode": row["generation"]["mode"],
                        "correct": agent_correct(row, frozen[task["id"]]),
                        "first_call": row["calls"][0] if row["calls"] else None,
                        "prompt_end_kl_from_exact": row["prompt_end_kl_from_exact"],
                    }
                )
                if row.get("parse_error"):
                    flags.append(f"agent {task['id']} {arm}: parse error")
        agent[arm] = {
            "correct": sum(r["correct"] for r in rows),
            "total": len(rows),
            "worst_prompt_end_kl": max(r["prompt_end_kl_from_exact"] for r in rows),
            "rows": rows,
        }
    for task in report["tool_tasks"]:
        for arm in TOOL_ARMS:
            for row in task["arms"][arm]:
                if row.get("parse_error"):
                    flags.append(f"tool {task['id']} {arm}: parse error (scored wrong)")

    # Investigation flags (parse errors) do not change scores; they hold the
    # decision until reviewed.
    default_ok = text == "QUALIFIES" and screen == "PASS" and not flags
    verdict = {
        "schema": "glm53.quality_split_verdict.v1",
        "text_split": text,
        "tool_screen_split": screen,
        "fast_shared_split_may_become_default": default_ok,
        "pending_review": bool(flags),
        "decision_note": "subject also to a release-build time-to-first-token screen; investigation flags hold the decision until reviewed",
        "comparisons_gated": comparisons,
        "comparisons_reported": reported,
        "reproduces_committed": reproduces(items, committed),
        "committed_sha256": hashlib.sha256(committed_bytes).hexdigest(),
        "tools": tools,
        "agent": agent,
        "investigate": flags,
        "manifest_sha256": report["manifest_sha256"],
        "evaluator_commit": report.get("evaluator_commit"),
    }
    for key, result in {**comparisons, **reported}.items():
        print(key, "(gated)" if key in comparisons else "(reported)")
        for name, v in result.items():
            lo, hi = v["central90"]
            print(
                f"  {name:12} observed {v['observed_mean']:+.5f}  central 90% [{lo:+.5f}, {hi:+.5f}]  margin {v['margin']:+.3f}  {v['verdict']}"
            )
    print("reproduces committed:", json.dumps(verdict["reproduces_committed"]))
    print(
        "tools",
        json.dumps(
            {
                a: {k: t[k] for k in ("greedy_correct", "sampled_mean")}
                for a, t in tools.items()
            }
        ),
    )
    print(
        "agent",
        json.dumps(
            {
                a: {k: v[k] for k in ("correct", "total", "worst_prompt_end_kl")}
                for a, v in agent.items()
            }
        ),
    )
    print("investigate:", flags or "none")
    print(
        f"text (split arms): {text} | tool screen (split): {screen} | may become default: {default_ok}"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=2) + "\n")


if __name__ == "__main__":
    main()
