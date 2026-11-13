#!/usr/bin/env python3
"""Aggregate the dispatch census emitted by fn_prefill_census.

Reads `kind=dispatch` records and prints per-kernel GPU core time for one cell.
Use the same cell in both arms and normalize by command rows when comparing a
2048-row command with a 4096-row command.
"""

import argparse
import collections
import json
from pathlib import Path

CATS = [
    (
        "routed",
        (
            "moe_swiglu",
            "moe_up_",
            "moe_gate",
            "moe_down",
            "moe_weighted_sum",
            "moe_wide_probe",
            "grouped_slots",
        ),
    ),
    ("router", ("router", "topk", "bucket", "route")),
    ("gdn", ("gdn_", "ssm_")),
    ("qsa", ("qsa", "selected_", "attn_matrix", "attn_", "index_")),
    ("hc", ("hc_", "hyper")),
    ("dense", ("mat_mat", "mat_vec", "gemm", "gemv")),
    ("shared", ("shexp", "shared")),
]


def category(kernel):
    k = kernel.lower()
    for name, parts in CATS:
        if any(p in k for p in parts):
            return name
    return "other"


def agg(rows):
    out = collections.defaultdict(
        lambda: {"calls": 0, "core": 0.0, "gap": 0.0, "iso": 0.0}
    )
    prev = min(r["gpu_start"] for r in rows if r.get("gpu_start"))
    for r in rows:
        s, e = r.get("gpu_start"), r.get("gpu_end")
        if s is None or e is None:
            continue
        a = out[r["kernel"]]
        a["calls"] += 1
        a["iso"] += (e - s) / 1e6
        a["gap"] += max(0, s - prev) / 1e6
        a["core"] += max(0, e - max(s, prev)) / 1e6
        prev = max(prev, e)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    recs = [json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
    cands = [r for r in recs if r["kind"] == "dispatch" and r["cell"] == a.cell]
    if a.rows is not None:
        cands = [r for r in cands if r["rows"] == a.rows]
    if a.depth is not None:
        cands = [r for r in cands if r["depth"] == a.depth]
    print(f"{len(cands)} dispatch record(s)")
    for r in cands:
        rows = r["dispatches"]
        a_ = agg(rows)
        total = sum(v["core"] for v in a_.values())
        cats = collections.defaultdict(float)
        for k, v in a_.items():
            cats[category(k)] += v["core"]
        print(
            f"\ncell={r['cell']} depth={r['depth']} rows={r['rows']} "
            f"wall={r['wall_ms']:.1f}ms dispatches={len(rows)} overflow={r['overflow']} "
            f"hash={r.get('logits_hash', '')[:16]}"
        )
        print("  category      ms     share   per-2048-row ms")
        for name, ms in sorted(cats.items(), key=lambda kv: -kv[1]):
            print(
                f"  {name:12} {ms:8.2f} {ms / total * 100:6.1f}% {ms * 2048 / r['rows']:10.2f}"
            )
        print("  top kernels:")
        for k, v in sorted(a_.items(), key=lambda kv: -kv[1]["core"])[: a.top]:
            print(f"    {v['core']:8.2f}ms {v['calls']:5} calls  {k}")


if __name__ == "__main__":
    main()
