#!/usr/bin/env python3
"""Width and recoverable-time ledger for the 2026-11-09 steady serve runs.

Read-only.  The primary 180 s evidence lives in
``$HOME/qwen-runs/2026-11-08-realistic-steady`` on the laptop (and mirrored on
the M5).  The later 90 s/180 s pairs live under ``$HOME/qwen-runs`` on the M5
(``metagross``).  Nothing is written; only summary.json/calls.jsonl are read.

Native round costs are constants below because the PERF-LOG entries publish
whole-round ABBA results, not per-call timings.  Each constant carries its
source label.  The script re-derives per-width transition counts from the
harness' own ``phases.batch`` counters using the same formula as
``native_widths`` in scripts/bench/serve_realistic.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field

# Whole-round native costs, ms/round.  See docs/PERF-LOG.md entries cited in the
# report.  "causal" means the default-on dependency-wave executor.
V41_COST_32 = {"b1": 13.45, "b2": 20.68, "b3": 24.69, "b4": 29.06}
V41_COST_64 = {"b1": 13.59, "b2": 20.68, "b3": 26.51, "b4": 29.06}
# GLM 8K/32K causal-wave on costs.  B1 is the 8K single-stream token time; no
# 32K B1 whole-round packet is published.
GLM_COST_32 = {"b1": 17.18, "b2": 22.78, "b3": 26.96, "b4": 31.53}
GLM_COST_64 = {"b1": 17.18, "b2": 22.78, "b3": 29.79, "b4": 31.53}
# Flash-Next: B1 is the 32K serial-step cost from the final B4 standing.
# B2/B3 post-stack costs are not published; B2 is the earlier 32K native value
# and B3 is inferred (see report).  B4 is the final production standing.
FN_COST_32 = {"b1": 13.49, "b2": 24.08, "b3": 27.00, "b4": 30.12}
FN_COST_64 = dict(FN_COST_32)

RUNS = {
    "v41-180": dict(
        host=None,
        path="~/qwen-runs/2026-11-08-realistic-steady/v41-default",
        family="v41",
        label="V4.1 default 180 s",
        costs=(V41_COST_32, V41_COST_64),
    ),
    "v41-serial-180": dict(
        host=None,
        path="~/qwen-runs/2026-11-08-realistic-steady/v41-serial",
        family="v41",
        label="V4.1 serial 180 s",
        costs=(V41_COST_32, V41_COST_64),
    ),
    "glm-180": dict(
        host=None,
        path="~/qwen-runs/2026-11-08-realistic-steady/glm-default",
        family="glm",
        label="GLM default 180 s",
        costs=(GLM_COST_32, GLM_COST_64),
    ),
    "glm-serial-180": dict(
        host=None,
        path="~/qwen-runs/2026-11-08-realistic-steady/glm-serial",
        family="glm",
        label="GLM serial 180 s",
        costs=(GLM_COST_32, GLM_COST_64),
    ),
    "fn-serial-180": dict(
        host=None,
        path="~/qwen-runs/2026-11-08-realistic-steady/fn-serial",
        family="fn",
        label="Flash-Next serial 180 s",
        costs=(FN_COST_32, FN_COST_64),
    ),
    "fn-direct-180": dict(
        host="metagross",
        path="~/qwen-runs/2026-11-09-fn-direct/serve-default/result",
        family="fn",
        label="Flash-Next default 180 s",
        costs=(FN_COST_32, FN_COST_64),
    ),
    "fn-direct-serial-180": dict(
        host="metagross",
        path="~/qwen-runs/2026-11-09-fn-direct/serve-serial/result",
        family="fn",
        label="Flash-Next serial 180 s (direct)",
        costs=(FN_COST_32, FN_COST_64),
    ),
    "v41-wave-off-90": dict(
        host="metagross",
        path="~/qwen-runs/2026-11-09-v41-batch-wave/steady-off-redo",
        family="v41",
        label="V4.1 causal off 90 s",
        costs=(V41_COST_32, V41_COST_64),
    ),
    "v41-wave-on-90": dict(
        host="metagross",
        path="~/qwen-runs/2026-11-09-v41-batch-wave/steady-on",
        family="v41",
        label="V4.1 causal on 90 s",
        costs=(V41_COST_32, V41_COST_64),
    ),
    "glm-final-off-90": dict(
        host="metagross",
        path="~/qwen-runs/2026-11-09-glm-batch-q6/serve-final/off1",
        family="glm",
        label="GLM batched off 90 s",
        costs=(GLM_COST_32, GLM_COST_64),
    ),
    "glm-final-on-90": dict(
        host="metagross",
        path="~/qwen-runs/2026-11-09-glm-batch-q6/serve-final/on1",
        family="glm",
        label="GLM batched on 90 s",
        costs=(GLM_COST_32, GLM_COST_64),
    ),
    "fn-shapes-rb-180": dict(
        host="metagross",
        path="~/qwen-runs/2026-11-09-fn-append-shapes/serve-final-rollback",
        family="fn",
        label="FN append rollback 180 s",
        costs=(FN_COST_32, FN_COST_64),
    ),
    "fn-shapes-def-180": dict(
        host="metagross",
        path="~/qwen-runs/2026-11-09-fn-append-shapes/serve-final-default-cooled",
        family="fn",
        label="FN append default 180 s",
        costs=(FN_COST_32, FN_COST_64),
    ),
}


def read(host: str | None, path: str) -> str:
    p = os.path.expanduser(path)
    if host is None:
        with open(p, "r", encoding="utf-8") as fh:
            return fh.read()
    cmd = ["ssh", "-o", "BatchMode=yes", host, f"cat {p}"]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def native_widths(batch: dict) -> list[int]:
    steps = int(batch.get("stepped_transitions", 0))
    if not steps:
        return [0, 0, 0]
    shared = int(batch["batched_transitions"])
    one = steps - shared
    three = round(steps * float(batch["width_mean"])) - one - 2 * shared
    two = shared - three
    if min(one, two, three) < 0 or int(batch["width_max"]) > 3:
        raise ValueError(f"width census requires at most three members: {batch}")
    return [one, two, three]


@dataclass
class Run:
    key: str
    label: str
    family: str
    window: float
    tps: float
    tokens: int
    trace: dict[str, float]
    calls: int
    carry: int
    covered_tokens: int
    uncovered_tokens: int
    observed_trans: float
    shares: list[float]
    trans_total: float
    rounds: list[float]
    append_s: float
    append_rows: int
    gap_gt_half_s: float
    cost: tuple[dict, dict]
    service: list[float] = field(default_factory=list)
    overhead: list[float] = field(default_factory=list)


def analyze(key: str, spec: dict) -> Run:
    summary = json.loads(read(spec["host"], spec["path"] + "/summary.json"))
    try:
        lines = read(spec["host"], spec["path"] + "/calls.jsonl").splitlines()
        rows = [json.loads(x) for x in lines if x.strip()]
    except (OSError, subprocess.CalledProcessError):
        rows = []
    w = num(summary.get("window_seconds", 180.0))
    begin = num(summary.get("window_begin"))
    tr = summary.get("trace_20s", [])
    trace = {
        "b1": sum(num(b.get("decode_stream_width_1_s")) for b in tr),
        "b2": sum(num(b.get("decode_stream_width_2_s")) for b in tr),
        "b3": sum(num(b.get("decode_stream_width_3_s")) for b in tr),
        "predecode": sum(num(b.get("predecode_only_s")) for b in tr),
        "idle": sum(num(b.get("idle_s")) for b in tr),
    }
    member = [0.0, 0.0, 0.0]
    covered = uncovered = 0
    append_ms = append_rows = 0
    carry = 0
    gap_gt_half = 0.0
    for r in rows:
        ph = r.get("phases") or {}
        batch = ph.get("batch")
        sent = r.get("sent")
        first = r.get("first")
        end = r.get("end")
        sent_r = (sent - begin) if sent is not None else None
        first_r = (first - begin) if first is not None else None
        end_r = (end - begin) if end is not None else None
        if batch and first_r is not None and end_r is not None and end_r > first_r:
            overlap = max(0.0, min(end_r, w) - max(first_r, 0.0)) / (end_r - first_r)
            width = native_widths(batch)
            for i in range(3):
                member[i] += width[i] * overlap
        if r.get("carry_in"):
            carry += 1
        win_tokens = int(num(r.get("window_tokens")))
        if batch:
            covered += win_tokens
        else:
            uncovered += win_tokens
        if sent_r is not None and 0 <= sent_r < w:
            append_ms += num(ph.get("prefill_ms"))
            append_rows += int(num(r.get("new")))
    for b in tr:
        g = num(b.get("gap_max_s"))
        if g > 0.5:
            gap_gt_half += g
    observed = sum(member)
    shares = [x / observed for x in member] if observed else [0.0, 0.0, 0.0]
    total_tokens = int(num(summary.get("window_tokens")))
    ratio = observed / covered if covered else 1.0
    trans_total = total_tokens * ratio
    rounds = [trans_total * shares[i] / (i + 1) for i in range(3)]
    service = [
        sum(rounds[i] * cost[f"b{i + 1}"] for i in range(3)) / 1000.0
        for cost in spec["costs"]
    ]
    return Run(
        key=key,
        label=spec["label"],
        family=spec["family"],
        window=w,
        tps=num(summary.get("aggregate_tps")),
        tokens=total_tokens,
        trace=trace,
        calls=len(rows),
        carry=carry,
        covered_tokens=covered,
        uncovered_tokens=uncovered,
        observed_trans=observed,
        shares=shares,
        trans_total=trans_total,
        rounds=rounds,
        append_s=append_ms / 1000.0,
        append_rows=append_rows,
        gap_gt_half_s=gap_gt_half,
        cost=spec["costs"],
        service=service,
        overhead=[w - s for s in service],
    )


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def observed_table(runs: list[Run]) -> None:
    print("## Observed width distribution (trace wall; member-transition share)\n")
    print(
        "| run | window | tok/s | B=1 wall | B=2 wall | B=3 wall | predecode | idle | B=1 trans | B=2 trans | B=3 trans |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for r in runs:
        t = r.trace
        print(
            f"| {r.label} | {r.window:.0f} s | {r.tps:.2f} | "
            f"{t['b1']:.2f} s ({pct(t['b1'] / r.window)}) | "
            f"{t['b2']:.2f} s ({pct(t['b2'] / r.window)}) | "
            f"{t['b3']:.2f} s ({pct(t['b3'] / r.window)}) | "
            f"{t['predecode']:.2f} s | {t['idle']:.2f} s | "
            + (
                f"{pct(r.shares[0])} | {pct(r.shares[1])} | {pct(r.shares[2])} |"
                if r.observed_trans
                else "- | - | - |"
            )
        )
    print()


def ledger_table(runs: list[Run]) -> None:
    print("## Native-round attribution and counterfactuals\n")
    print(
        "`S32`/`S64` are best-measured native decode service at 32K/64K; "
        "`E = decode wall - S` is the residual decode overhead.  "
        "Append is the sum of per-request prefill_ms for turns sent inside the window "
        "(overlapping request-seconds, not additive wall).\n"
    )
    print("| run | delivered | S32 | S64 | E32 | E64 | append req-s | >0.5 s gaps |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in runs:
        print(
            f"| {r.label} | {r.tokens} | {r.service[0]:.1f} s | {r.service[1]:.1f} s | "
            f"{r.overhead[0]:.1f} s | {r.overhead[1]:.1f} s | {r.append_s:.1f} s | "
            f"{r.gap_gt_half_s:.1f} s |"
        )
    print()
    print(
        "| run | current | native floor 32K | native floor 64K | all-at-B3 | B1->B3 | B1+B2->B3 | append 40% (alpha=.5) |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in runs:
        if not r.observed_trans:
            continue
        t = r.tokens
        floor32 = t / (r.service[0] + r.trace["predecode"] + r.trace["idle"])
        floor64 = t / (r.service[1] + r.trace["predecode"] + r.trace["idle"])
        s3 = (r.trans_total / 3.0) * r.cost[0]["b3"] / 1000.0
        rate_s3 = t / (r.window - (r.service[0] - s3))
        b1 = r.trans_total * r.shares[0]
        b2 = r.trans_total * r.shares[1]
        s_shift = (
            r.service[0]
            - b1 * r.cost[0]["b1"] / 1000.0
            + (b1 / 3.0) * r.cost[0]["b3"] / 1000.0
        )
        s_shift12 = (
            r.service[0]
            - b1 * r.cost[0]["b1"] / 1000.0
            - b2 * r.cost[0]["b2"] / 1000.0
            + ((b1 + b2) / 3.0) * r.cost[0]["b3"] / 1000.0
        )
        rate_shift = t / (r.window - (r.service[0] - s_shift))
        rate_shift12 = t / (r.window - (r.service[0] - s_shift12))
        alpha = 0.5
        append_crit = min(alpha * r.append_s, r.overhead[0])
        rate_app = t / (r.window - 0.4 * append_crit)
        print(
            f"| {r.label} | {r.tps:.2f} | {floor32:.2f} | {floor64:.2f} | "
            f"{rate_s3:.2f} | {rate_shift:.2f} | {rate_shift12:.2f} | {rate_app:.2f} |"
        )
    print()


def pair_table(pairs: list[tuple[Run, Run, str]]) -> None:
    print("## Later steady pairs referenced by PERF-LOG\n")
    print("| pair | control tok/s | candidate tok/s | ratio | window |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for a, b, label in pairs:
        print(
            f"| {label} | {a.tps:.3f} | {b.tps:.3f} | {b.tps / a.tps:.4f}x | {a.window:.0f} s |"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="metagross", help="SSH host for later runs")
    args = ap.parse_args()
    runs = {}
    for key, spec in RUNS.items():
        spec = dict(spec)
        if args.host:
            spec["host"] = {None: None, "metagross": args.host}.get(
                spec["host"], spec["host"]
            )
        runs[key] = analyze(key, spec)

    primary = [runs[k] for k in ("v41-180", "glm-180", "fn-direct-180")]
    baselines = [runs[k] for k in ("v41-serial-180", "glm-serial-180", "fn-serial-180")]
    observed_table(primary + baselines)
    ledger_table(primary)
    pair_table(
        [
            (
                runs["v41-wave-off-90"],
                runs["v41-wave-on-90"],
                "V4.1 causal waves (90 s)",
            ),
            (
                runs["glm-final-off-90"],
                runs["glm-final-on-90"],
                "GLM B=4 kernels (90 s)",
            ),
            (
                runs["fn-direct-serial-180"],
                runs["fn-direct-180"],
                "FN batching + direct outputs (180 s)",
            ),
            (
                runs["fn-shapes-rb-180"],
                runs["fn-shapes-def-180"],
                "FN append shapes (180 s)",
            ),
        ]
    )
    print("## Native costs used\n")
    for name, cost in (
        ("V4.1 32K", V41_COST_32),
        ("V4.1 64K", V41_COST_64),
        ("GLM 32K", GLM_COST_32),
        ("GLM 64K", GLM_COST_64),
        ("FN", FN_COST_32),
    ):
        print(f"- {name}: " + ", ".join(f"B{k[1:]}={v:.2f} ms" for k, v in cost.items()))


if __name__ == "__main__":
    main()
