#!/usr/bin/env python3
"""Reduce spec-econ probe JSONL into the economic comparison.

Usage: analyze.py RUN_DIR [NEXTN_DRAFT_MS]

Reads cost-8k.jsonl, cost-48k.jsonl, dflash-t1.jsonl, dflash-t0.jsonl from
RUN_DIR. Prints a compact JSON summary and a human table. Pure stdlib.
"""

import json
import sys
from pathlib import Path


def events(path):
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def one(evs, kind):
    return [e for e in evs if e.get("event") == kind]


def cost_summary(path):
    evs = events(path)
    if not evs:
        return None
    hdr = one(evs, "header")[0]
    serial = one(evs, "serial")
    verify = one(evs, "verify")
    batch = one(evs, "batch")
    return {
        "depth": hdr["depth"],
        "vocab": hdr.get("vocab"),
        "serial_ms": serial[0]["median"] if serial else None,
        "verify": {v["rows"]: v["total_median"] for v in verify},
        "verify_begin": {v["rows"]: v["begin_median"] for v in verify},
        "verify_forward": {v["rows"]: v["verify_median"] for v in verify},
        "batch": batch[0] if batch else None,
        "thermal": (
            hdr.get("thermal_before"),
            one(evs, "complete")[-1].get("thermal_after")
            if one(evs, "complete")
            else None,
        ),
    }


def dflash_summary(path):
    evs = events(path)
    if not evs:
        return None
    hdr = one(evs, "header")[0]
    summ = one(evs, "summary")
    if not summ:
        return None
    s = summ[-1]
    steps = [e["detail"] for e in one(evs, "step")]
    return {**hdr, **s, "steps": steps}


def main():
    run = Path(sys.argv[1])
    nextn_draft_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    c8 = cost_summary(run / "cost-8k.jsonl")
    c48 = cost_summary(run / "cost-48k.jsonl")
    t1 = dflash_summary(run / "dflash-t1.jsonl")
    t0 = dflash_summary(run / "dflash-t0.jsonl")
    t1b = dflash_summary(run / "dflash-t1b.jsonl")

    out = {
        "cost8k": c8,
        "cost48k": c48,
        "dflash_t1": t1,
        "dflash_t0": t0,
        "nextn_one_step_alpha": 0.715891,
    }

    # NextN one-step break-even: alpha* = (V2 + draft)/S - 1.
    if c48 and c48["serial_ms"] and 2 in c48["verify"]:
        s = c48["serial_ms"]
        v2 = c48["verify"][2]
        out["nextn"] = {
            "S_ms": s,
            "V2_ms": v2,
            "V2_over_S": v2 / s,
            "breakeven_one_step_alpha": (v2 + nextn_draft_ms) / s - 1.0,
            "measured_one_step_alpha": 0.715891,
            "speedup_b1_at_measured_alpha": (1.0 + 0.715891)
            * s
            / (v2 + nextn_draft_ms),
            "draft_ms_assumed": nextn_draft_ms,
        }

    # DFlash speedups from the measured runner components. The initial packet's
    # `total_ms` is the prefill, so it is excluded; its one emitted token stays.
    for tag, d in (("t1", t1), ("t0", t0), ("t1b", t1b)):
        if not d or not c48 or not c48["serial_ms"]:
            continue
        steps = d["steps"]
        spec = [x for x in steps if not x["anchor_only"]]
        anchor = [x for x in steps if x["anchor_only"]]
        emitted = 1 + sum(x["emitted"] for x in steps)
        decode = sum(x["total_ms"] for x in steps)
        s = c48["serial_ms"]
        c3 = c48["batch"]["median"] if c48.get("batch") else None
        out[f"dflash_{tag}_econ"] = {
            "block": d["block"],
            "alpha_per_draft": d["alpha_per_draft"],
            "emitted_per_packet": d["emitted_per_packet"],
            "spec_packets": len(spec),
            "anchor_only_packets": len(anchor),
            "emitted_total": emitted,
            "decode_ms": decode,
            "decode_tokens_per_s": emitted / decode * 1e3,
            "serial_tokens_per_s": 1e3 / s,
            "b1_speedup": emitted * s / decode,
            "b3_aggregate_ratio_serialized": (
                (emitted / decode) / (3.0 / c3) if c3 else None
            ),
            "c3_ms": c3,
            "components_per_spec_packet_ms": {
                k: sum(x[k] for x in spec) / max(len(spec), 1)
                for k in (
                    "draft_ms",
                    "verify_ms",
                    "checkpoint_ms",
                    "commit_ms",
                    "append_ms",
                    "total_ms",
                )
            },
        }

    print(json.dumps(out, indent=2))

    print("\n## Verify cost curve (48K)")
    if c48:
        s = c48["serial_ms"]
        print(f"S1 = {s:.2f} ms")
        print("rows  V(ms)  V/S  begin(ms)  forward(ms)")
        for m in sorted(c48["verify"]):
            v = c48["verify"][m]
            print(
                f"{m:>4}  {v:7.2f}  {v / s:4.2f}  {c48['verify_begin'][m]:9.2f}  {c48['verify_forward'][m]:11.2f}"
            )
    if c48 and c48.get("batch"):
        b = c48["batch"]
        print(
            f"\nB={b['width']} round = {b['median']:.2f} ms ({b['per_member_median']:.2f} ms/member)"
        )
    for tag, d in (("T=1", t1), ("T=0", t0), ("T=1b", t1b)):
        if d:
            spec = [x for x in d["steps"] if not x["anchor_only"]]
            emitted = 1 + sum(x["emitted"] for x in d["steps"])
            decode = sum(x["total_ms"] for x in d["steps"])
            print(
                f"\nDFlash {tag}: block={d['block']} alpha/draft={d['alpha_per_draft']:.3f} "
                f"emitted/packet={d['emitted_per_packet']:.2f} "
                f"decode={emitted / decode * 1e3:.1f} tok/s "
                f"vs-serial={emitted * c48['serial_ms'] / decode:.2f}x "
                f"verify={sum(x['verify_ms'] for x in spec) / max(len(spec), 1):.1f} ms "
                f"commit={sum(x['commit_ms'] for x in spec) / max(len(spec), 1):.1f} ms"
            )


if __name__ == "__main__":
    main()
