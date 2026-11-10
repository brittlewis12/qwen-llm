#!/usr/bin/env python3
"""Price measured GLM NextN speculation against the measured batch path.

Usage: nextn_project.py [--json]

Inputs are the 2026-11-09 measurements: S=18.60 ms serial native forward,
V(2)/V(3)=23.69/27.81 ms on the batch-path proxy, head 1.722 ms, sampler draw
0.953 ms and per-row state ring 1.201 ms. Acceptance is the measured 48K code
T=1/top_p=0.95 value. The speculative round charges one target draw and one
draft draw plus head per draft step; the non-speculative baseline charges one
target draw per token. The fixed-cost scenarios remove F ms of batched fixed
overhead (2-3 ms on other families; GLM's own fit implies ~0.8 ms) and add the
same 0.75 ms causal penalty. Break-even counts the 3.881 s head-history pass
against the per-token saving.
"""

import json
import sys

S = 18.60
SAMPLE = 0.95325
HEAD = 1.722041
RING = 1.200604
V_BATCH = {2: 23.692917, 3: 27.813604}
V_PACKED = {2: 32.67, 3: 77.74}
A1 = 0.7991684704778343
A2 = 0.8141741366405731
C3 = 27.813604
HEAD_HISTORY = 3881.44525


def emitted(k, a2=A2):
    return 1.0 + sum(A1 * a2 ** (j - 1) for j in range(1, k + 1))


def b1(verifier, k, fixed=0.0, serial=0.0, charge_sample=True):
    v = verifier[k + 1] - fixed + serial
    round_ms = v + k * HEAD + k * RING
    if charge_sample:
        round_ms += k * SAMPLE + SAMPLE
    tokens = emitted(k)
    base = S + (SAMPLE if charge_sample else 0.0)
    return {
        "k": k,
        "fixed_ms": fixed,
        "serial_ms": serial,
        "verifier_ms": v,
        "round_ms": round_ms,
        "emitted": tokens,
        "speedup": tokens * base / round_ms,
    }


def breakeven(k, fixed=0.0, serial=0.0):
    row = b1(V_BATCH, k, fixed, serial)
    per = row["round_ms"] / row["emitted"]
    return HEAD_HISTORY / ((S + SAMPLE) - per)


def b3(k, fixed=0.0, serial=0.0):
    row = b1(V_BATCH, k, fixed, serial)
    base_round = C3 - fixed + 3 * SAMPLE
    return {
        "k": k,
        "fixed_ms": fixed,
        "spec_rate_per_ms": row["emitted"] / row["round_ms"],
        "batch_rate_per_ms": 3.0 / base_round,
        "ratio": (row["emitted"] / row["round_ms"]) / (3.0 / base_round),
    }


def main():
    out = {
        "measured": {
            "serial_ms": S,
            "sampler_ms": SAMPLE,
            "head_ms": HEAD,
            "ring_ms": RING,
            "a1": A1,
            "a2": A2,
            "v_batch": V_BATCH,
            "v_packed": V_PACKED,
        },
        "b1_batch_path": [b1(V_BATCH, k) for k in (1, 2)],
        "b1_batch_path_serial075": [b1(V_BATCH, k, serial=0.75) for k in (1, 2)],
        "b1_batch_path_no_sample": [
            b1(V_BATCH, k, charge_sample=False) for k in (1, 2)
        ],
        "b1_batch_path_fixed": {
            str(f): [b1(V_BATCH, k, fixed=f, serial=0.75) for k in (1, 2)]
            for f in (0.8, 2.0, 2.5, 3.0)
        },
        "b1_packed": [b1(V_PACKED, k) for k in (1, 2)],
        "b3_adaptive": {
            str(f): [b3(k, fixed=f) for k in (1, 2)] for f in (0.0, 0.8, 2.5)
        },
    }
    if "--json" in sys.argv:
        print(json.dumps(out, indent=2))
        return
    print("B=1, batch-path verifier, measured head + sampler + ring")
    for row in out["b1_batch_path"]:
        print(
            f"  K={row['k']}  round={row['round_ms']:.2f} ms  "
            f"emitted={row['emitted']:.3f}  speedup={row['speedup']:.3f}x"
        )
    print("B=1, +0.75 ms causal serialization")
    for row in out["b1_batch_path_serial075"]:
        print(
            f"  K={row['k']}  round={row['round_ms']:.2f} ms  "
            f"speedup={row['speedup']:.3f}x"
        )
    print("B=1, fixed batched cost removed (+0.75 ms causal, sampling charged)")
    for f, rows in out["b1_batch_path_fixed"].items():
        print(
            f"  F={f} ms: "
            + "  ".join(
                f"K={r['k']} {r['speedup']:.3f}x "
                f"(be {breakeven(r['k'], fixed=float(f), serial=0.75):.0f} tok)"
                for r in rows
            )
        )
    print("B=1, packed verifier today")
    for row in out["b1_packed"]:
        print(
            f"  K={row['k']}  round={row['round_ms']:.2f} ms  "
            f"speedup={row['speedup']:.3f}x"
        )
    print("B=3 adaptive ratio (spec verify blocks the batch; <1 loses)")
    for f, rows in out["b3_adaptive"].items():
        print(
            f"  F={f} ms: " + "  ".join(f"K={r['k']} {r['ratio']:.3f}x" for r in rows)
        )


if __name__ == "__main__":
    main()
