#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["lxml>=5"]
# ///
"""A0 discriminator: intra-kick occupancy time-series (H1/H2 flat-low vs
H3 drain-sawtooth vs H4 tail-imbalance).

Reads a matched (gpu-counter-value, metal-gpu-execution-points) export pair
from one metal-counters capture and reports, for Kernel Occupancy (id 3) and
Compute SIMD Groups Inflight (id 54):

  - per-kick-index mean profile over normalized kick time (interior only);
  - interior valley recurrence: fraction of 25 us bins below 50% of the
    kick's p90, with a 50 us guard band at kick edges (cx condition: only
    valleys that recur INSIDE long kicks count toward H3/H4);
  - pooled interior bin histogram (bimodal => sawtooth; unimodal-low =>
    flat-low);
  - a tail-slope metric (mean of last-quartile bins vs middle-half bins)
    to separate H4 (end-of-kick decay) from H3 (recurring valleys).

Pre-registered interpretation (docs/bench/2026-07-03-xcode-decode-capture/):
  flat-low  -> H1/H2 (per-kernel residency caps) -> Program A next.
  interior recurring valleys -> H3 (drain)      -> Program B0/B1a next.
  monotone tail decay        -> H4 (imbalance)  -> B0 + work-stealing shape.

Usage:
  scripts/profile/gpu_occupancy_timeseries.py \
      --values /tmp/nwg-default-gpu-counter-value.xml \
      --exec-points /tmp/nwg-default-metal-gpu-execution-points.xml
"""

import argparse
import bisect
from collections import defaultdict

from lxml import etree as ET

BIN_S = 25e-6
GUARD_S = 50e-6
COUNTERS = {3: "Kernel Occupancy", 54: "SIMD Inflight"}


def parse_ts(s):
    mm, rest = s.split(":")
    p = rest.split(".")
    return (
        int(mm) * 60
        + int(p[0])
        + int(p[1]) / 1e3
        + (int(p[2]) if len(p) > 2 else 0) / 1e6
    )


def kick_windows(path):
    tree = ET.parse(path)
    root = tree.getroot()
    interned = {}
    for el in root.iter():
        if "id" in el.attrib:
            interned[(el.tag, el.attrib["id"])] = el.attrib.get("fmt", el.text or "")

    def val(el):
        if "ref" in el.attrib:
            return interned.get((el.tag, el.attrib["ref"]), "")
        return el.attrib.get("fmt", el.text or "")

    by_id = defaultdict(list)
    for r in root.findall(".//row"):
        ts = None
        cb = None
        for c in r:
            if c.tag == "start-time" and ts is None:
                ts = val(c)
            elif c.tag == "metal-command-buffer-id":
                cb = val(c)  # rows carry TWO: first is queue-scoped, LAST is
                # the real per-command-buffer id (see A0 root-cause note).
        if cb and ts:
            by_id[cb].append(parse_ts(ts))
    iv = sorted((min(t), max(t)) for t in by_id.values() if len(t) >= 2)
    big = [x for x in iv if x[1] - x[0] > 0.002]
    tokens, cur = [], []
    for a, b in big:
        if cur and a - cur[-1][1] > 0.0008:
            if len(cur) == 3:
                tokens.append(cur)
            cur = []
        cur.append((a, b))
    if len(cur) == 3:
        tokens.append(cur)
    return tokens


def stream_series(values_xml, want):
    """Yield (t, cid, value) for counter ids in `want`, streaming."""
    ts_i, u_i, d_i, cur = {}, {}, {}, {}
    for _, el in ET.iterparse(values_xml, events=("end",)):
        tag = el.tag
        if tag == "event-time":
            if "id" in el.attrib:
                v = int(el.text) / 1e9
                ts_i[el.attrib["id"]] = v
                cur["t"] = v
            else:
                cur["t"] = ts_i.get(el.attrib.get("ref"))
        elif tag == "uint32":
            if "id" in el.attrib:
                v = int(el.attrib.get("fmt", "0").replace(",", ""))
                u_i[el.attrib["id"]] = v
                if "cid" not in cur:
                    cur["cid"] = v
            else:
                v = u_i.get(el.attrib.get("ref"), 0)
                if "cid" not in cur:
                    cur["cid"] = v
        elif tag == "fixed-decimal":
            if "id" in el.attrib:
                v = float(el.text)
                d_i[el.attrib["id"]] = v
                cur["val"] = v
            else:
                cur["val"] = d_i.get(el.attrib.get("ref"), 0.0)
        elif tag == "row":
            t, cid, v = cur.get("t"), cur.get("cid"), cur.get("val")
            if t is not None and cid in want and v is not None:
                yield t, cid, v
            cur.clear()
            el.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True)
    ap.add_argument("--exec-points", required=True)
    args = ap.parse_args()

    tokens = kick_windows(args.exec_points)
    if not tokens:
        raise SystemExit("no 3-kick tokens")
    segs = []
    for t in tokens:
        for j, (a, b) in enumerate(t):
            segs.append((a, b, j))
    segs.sort()
    starts = [s[0] for s in segs]
    print(
        f"tokens: {len(tokens)}; kick medians "
        + "/".join(
            f"{sorted((t[j][1] - t[j][0]) for t in tokens)[len(tokens) // 2] * 1e3:.2f}"
            for j in range(3)
        )
        + " ms"
    )

    # Per (kick-idx): normalized 40-bin mean profile; interior valley stats;
    # pooled interior histogram; tail metric. Time-weighting: 25 us bins with
    # bin means (one sample per bin at ~21 us cadence ~= time-weighted).
    NPROF = 40
    prof = [[[0.0, 0] for _ in range(NPROF)] for _ in range(3)]
    valley_fracs = [[] for _ in range(3)]
    hist = [defaultdict(int) for _ in range(3)]
    tails = [[] for _ in range(3)]
    per_kick_bins = defaultdict(lambda: defaultdict(list))  # (seg_idx)->bin->vals

    for t, cid, v in stream_series(args.values, {3}):
        i = bisect.bisect_right(starts, t) - 1
        if i < 0:
            continue
        a, b, j = segs[i]
        if not (a + GUARD_S <= t <= b - GUARD_S):
            continue
        rel = (t - a) / (b - a)
        p = prof[j][min(int(rel * NPROF), NPROF - 1)]
        p[0] += v
        p[1] += 1
        per_kick_bins[i][int((t - a) / BIN_S)].append(v)

    for i, bins in per_kick_bins.items():
        a, b, j = segs[i]
        vals = [sum(vs) / len(vs) for _, vs in sorted(bins.items())]
        if len(vals) < 8:
            continue
        p90 = sorted(vals)[int(len(vals) * 0.9)]
        if p90 <= 0:
            continue
        valley_fracs[j].append(sum(1 for x in vals if x < 0.5 * p90) / len(vals))
        for x in vals:
            hist[j][min(int(x // 10), 9)] += 1
        q = len(vals) // 4
        mid = vals[q : 3 * q]
        tail = vals[3 * q :]
        if mid and tail:
            tails[j].append((sum(tail) / len(tail)) / (sum(mid) / len(mid)))

    import statistics as st

    for j in range(3):
        pf = [p[0] / p[1] if p[1] else 0 for p in prof[j]]
        spark = "".join(" .:-=+*#%@"[min(int(x / 10), 9)] for x in pf)
        print(f"\nkick[{j}] profile (norm time, 40 bins, occ%): |{spark}|")
        print(f"  bins: " + " ".join(f"{x:.0f}" for x in pf[::4]))
        vf = valley_fracs[j]
        tl = tails[j]
        print(
            f"  interior valley fraction (<50% of kick p90): med={st.median(vf):.3f} "
            f"p90={sorted(vf)[int(len(vf) * 0.9)]:.3f} (n={len(vf)} kicks)"
        )
        print(f"  tail/mid ratio: med={st.median(tl):.3f}")
        total = sum(hist[j].values())
        print(
            "  occ histogram (decades 0-100): "
            + " ".join(
                f"{hist[j].get(d, 0) * 100 // max(total, 1)}%" for d in range(10)
            )
        )


if __name__ == "__main__":
    main()
