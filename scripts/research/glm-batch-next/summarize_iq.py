#!/usr/bin/env python3
"""Whole-round B=N IQ load-schedule screen: each variant vs shipped, both orders.

Arm order is forward, reverse, reverse, forward, so every variant's mean arm
position is equal and linear drift cancels; halves give the two orders.
"""

import argparse, collections, json, statistics as st
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("input", type=Path)
p.add_argument("--out", type=Path)
a = p.parse_args()
rows = []
for line in a.input.read_text().splitlines():
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if isinstance(r, dict) and "kind" in r:
        rows.append(r)
assert rows[-1]["kind"] == "complete", "incomplete probe"
header = rows[0]
timing = [r for r in rows if r["kind"] == "timing"]
n = len(timing)
half = n // 2
arm = lambda r: st.mean(r["wall_ms"])
by = collections.defaultdict(list)
for r in timing:
    by[r["variant"]].append(r)
base = [arm(r) for r in by["shipped"]]
base_halves = [
    st.mean(arm(r) for r in by["shipped"] if (r["arm"] < half) == h)
    for h in (True, False)
]
out = dict(header=header, arms=n, variants={})
for v, rs in by.items():
    means = [arm(r) for r in rs]
    halves = [
        st.mean(arm(r) for r in rs if (r["arm"] < half) == h) for h in (True, False)
    ]
    q = [r for r in rows if r["kind"] == "quality" and r["variant"] == v]
    s = [r for r in rows if r["kind"] == "state" and r["variant"] == v]
    out["variants"][v] = dict(
        round_ms=st.mean(means),
        arm_ms=[round(x, 3) for x in means],
        median_round_ms=st.median(x for r in rs for x in r["wall_ms"]),
        saved_ms=st.mean(base) - st.mean(means),
        saved_by_order_ms=[b - h for b, h in zip(base_halves, halves)],
        thermal_max=max(max(r["thermal"]) for r in rs),
        quality_vectors=len(q),
        quality_bits=sum(r["bits"] for r in q),
        max_abs_dnll=max((abs(r["dnll"]) for r in q), default=0),
        mean_abs_dnll=(st.mean([abs(r["dnll"]) for r in q]) if q else 0),
        max_kl=max((r["kl"] for r in q), default=0),
        mean_kl=(st.mean([r["kl"] for r in q]) if q else 0),
        state_bits=all(all(r["state_bits"]) for r in s) if s else None,
    )
noise = [r for r in rows if r["kind"] == "noise"]
out["noise"] = dict(vectors=len(noise), bits=sum(r["bits"] for r in noise))
print(
    "depth",
    header["depth"],
    "width",
    header.get("width", 4),
    "arms",
    n,
    "noise bits",
    out["noise"],
)
print(
    "variant\tround ms\tsaved ms\tsaved fwd/rev ms\tthermal max\tquality bits\tstate bits\tdNLL max/mean\tKL max/mean"
)
for v, c in sorted(out["variants"].items(), key=lambda kv: -kv[1]["saved_ms"]):
    print(
        v,
        round(c["round_ms"], 3),
        round(c["saved_ms"], 3),
        "/".join(str(round(x, 3)) for x in c["saved_by_order_ms"]),
        c["thermal_max"],
        f"{c['quality_bits']}/{c['quality_vectors']}",
        c["state_bits"],
        f"{c['max_abs_dnll']:.2e}/{c['mean_abs_dnll']:.2e}",
        f"{c['max_kl']:.2e}/{c['mean_kl']:.2e}",
        sep="\t",
    )
if a.out:
    a.out.write_text(json.dumps(out, indent=2) + "\n")
