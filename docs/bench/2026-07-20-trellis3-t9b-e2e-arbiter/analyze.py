#!/usr/bin/env python3
"""T9b paired-bootstrap analyzer (frozen alongside the prereg; run when
the pipeline's five nll dumps exist).

Usage: python3 analyze.py <ppl_dir>  (expects nll-{f32,q3k,q4k,a0,a3}.f64)

Implements the preregistered analysis exactly:
- damage D(X) = mean_nll(X) - mean_nll(f32)
- paired per-SEGMENT bootstrap (10,000 resamples over 64 segments,
  seeded) for CIs on damage differences and on the D(a3)/D(q3k) ratio
- gate evaluation G1/G2/G3 per the frozen text
"""

import struct
import sys
import random

SEG_LEN = 511  # predictions per 512-token segment
N_BOOT = 10_000
SEED = 0x79B5


def read_nll(path):
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 8
    return list(struct.unpack(f"<{n}d", raw))


def seg_means(x):
    assert len(x) % SEG_LEN == 0, f"len {len(x)} not a multiple of {SEG_LEN}"
    return [
        sum(x[i * SEG_LEN : (i + 1) * SEG_LEN]) / SEG_LEN
        for i in range(len(x) // SEG_LEN)
    ]


def main(d):
    arms = ["f32", "q3k", "q4k", "a0", "a3"]
    nll = {a: read_nll(f"{d}/nll-{a}.f64") for a in arms}
    n = len(nll["f32"])
    for a in arms:
        assert len(nll[a]) == n, f"length mismatch {a}"
    segs = {a: seg_means(nll[a]) for a in arms}
    ns = len(segs["f32"])
    mean = {a: sum(segs[a]) / ns for a in arms}
    dmg = {a: mean[a] - mean["f32"] for a in arms}
    print(f"predicted tokens: {n}  segments: {ns}")
    for a in arms:
        import math

        print(
            f"  {a:>4}: mean_nll={mean[a]:.6f} ppl={math.exp(mean[a]):.4f} "
            f"damage={dmg[a]:+.6f}"
        )

    rng = random.Random(SEED)
    idx = list(range(ns))

    def boot(stat):
        vals = []
        for _ in range(N_BOOT):
            sample = [idx[rng.randrange(ns)] for _ in range(ns)]
            vals.append(stat(sample))
        vals.sort()
        return (
            vals[int(0.025 * N_BOOT)],
            vals[int(0.5 * N_BOOT)],
            vals[int(0.975 * N_BOOT)],
        )

    def dmg_of(a, sample):
        return sum(segs[a][i] - segs["f32"][i] for i in sample) / len(sample)

    # G1: D(a3) - D(a0)
    lo, med, hi = boot(lambda s: dmg_of("a3", s) - dmg_of("a0", s))
    print(f"\nG1 D(a3)-D(a0): median {med:+.6f}  95% CI [{lo:+.6f}, {hi:+.6f}]")
    if hi < 0:
        g1 = "r_H VALIDATED (a3 strictly better end-to-end)"
    elif lo > 0:
        g1 = "r_H FALSIFIED (a3 strictly worse; plain-P was right)"
    else:
        g1 = "TIED (CI straddles zero)"
    print(f"G1 verdict: {g1}")

    # G2/G3: ratio D(a3)/D(q3k)
    lo_r, med_r, hi_r = boot(lambda s: dmg_of("a3", s) / max(dmg_of("q3k", s), 1e-12))
    print(f"G2 D(a3)/D(q3k): median {med_r:.4f}  95% CI [{lo_r:.4f}, {hi_r:.4f}]")
    reopen = med_r <= 0.75 and hi_r <= 0.85
    g3 = dmg["a3"] >= dmg["q3k"] and dmg["a0"] >= dmg["q3k"]
    if reopen:
        print("G2 verdict: REOPEN-TIER (calibrated-domain evidence)")
    elif g3:
        print("G3 verdict: TIER STAYS CLOSED (no quality story at 3.06 bpw)")
    else:
        print(
            "G2/G3 verdict: INCONCLUSIVE-BAND (see prereg for the one permitted extension)"
        )

    # Context: a0 vs q3k ratio, q4k sanity.
    lo0, med0, hi0 = boot(lambda s: dmg_of("a0", s) / max(dmg_of("q3k", s), 1e-12))
    print(f"context D(a0)/D(q3k): median {med0:.4f}  95% CI [{lo0:.4f}, {hi0:.4f}]")
    print(
        f"sanity: D(q4k) < D(q3k): {dmg['q4k'] < dmg['q3k']}   all D>0: "
        f"{all(dmg[a] > 0 for a in ['q3k', 'q4k', 'a0', 'a3'])}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
