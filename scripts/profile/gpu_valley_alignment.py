#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["lxml>=5"]
# ///
"""A0 follow-up: align interior occupancy valleys to cross-channel
micro-submission bursts (the only sub-kick structure the exec-points table
actually carries).

Structure of `metal-gpu-execution-points` (M4 Max, metal-counters template):
  - decode compute kicks are single begin/end (function 1/2) intervals on one
    channel (e.g. 0x123459 = Compute); there are NO intra-kick compute-channel
    markers - a kick is one opaque interval;
  - interior rows belong to OTHER channels (0x123457 = Vertex, 0x123458 =
    Fragment): paired micro-submissions, mostly 4-20 us long, in bursts at
    interior offsets of the compute kicks. IDENTIFIED via the
    `metal-gpu-intervals` table as WindowServer display compositing (plus a
    trickle of Browser Helper) - NOT qwen aux traffic, NOT paging.

This tool tests whether A0's interior occupancy valleys (25 us bins < 50% of
the kick's own p90, 50 us edge guards - definitions identical to
gpu_occupancy_timeseries.py) are time-aligned with those bursts.

Pre-registered interpretation and RESULT (see PERF-LOG v0.492 entry):
  P1 burst-aligned: valley bins concentrate near bursts vs circular-shift
     control (enrichment >= 2x within +-100 us) and the burst-onset PSTH
     shows an occupancy dip -> valleys correlate with cross-channel work.
  P2 unaligned: distance-to-burst distribution matches control -> micro-subs
     are epiphenomenal; H3 inter-dispatch drain stands on the sawtooth
     evidence; B0 proceeds unchanged.
  VERDICT (nwg-default capture): P2. Enrichment 1.15x (< 2x gate); valley
  bins inside burst envelopes 15.4% vs 12.1% baseline cover. The PSTH shows
  a real but minor time-locked dip (flat ~29% pre-onset -> ~22-24% for
  ~400 us post-onset, ~1 burst / 8.3 ms display refresh) accounting for
  roughly 3% of the occupancy shortfall. Compositor interference excluded
  as the valley mechanism; H3 remains dominant.

Note the qwen side of `metal-gpu-intervals` is ONE interval per kick (each
kick is a single compute encoder), so trace-based valley-to-dispatch
alignment is impossible on this hardware; the causal drain measurement
moves into the B0 probe (synthetic dispatch-chain vs persistent-chain arm).

Controls: circular shift of the per-kick valley mask (preserves valley run
structure/autocorrelation); 200 shifts per kick, pooled.

Usage:
  scripts/profile/gpu_valley_alignment.py \
      --values /tmp/nwg-default-gpu-counter-value.xml \
      --exec-points /tmp/nwg-default-metal-gpu-execution-points.xml
"""

import argparse
import bisect
import random
import statistics as st
from collections import defaultdict

from lxml import etree as ET

BIN_S = 25e-6
GUARD_S = 50e-6
BURST_GAP_S = 50e-6
NEAR_S = 100e-6
OCC_ID = 3  # Kernel Occupancy counter id


def parse_exec_rows(path):
    """Return [(t_s, channel, function, submission_id)] resolving interned refs.

    Rows carry TWO metal-command-buffer-id elements: FIRST is the queue-scoped
    channel id, LAST is the real submission id (v0.491 root-cause note).
    """
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

    def ts_seconds(el):
        if "ref" in el.attrib:
            s = interned.get((el.tag, el.attrib["ref"]), "")
            mm, rest = s.split(":")
            p = rest.split(".")
            return (
                int(mm) * 60
                + int(p[0])
                + int(p[1]) / 1e3
                + (int(p[2]) if len(p) > 2 else 0) / 1e6
            )
        return int(el.text) / 1e9

    rows = []
    for r in root.findall(".//row"):
        t = chan = func = sub = None
        for c in r:
            if c.tag == "start-time" and t is None:
                t = ts_seconds(c)
            elif c.tag == "metal-command-buffer-id":
                if chan is None:
                    chan = val(c)
                sub = val(c)  # LAST id wins
            elif c.tag == "uint32" and func is None:
                func = val(c)
        if t is not None and sub:
            rows.append((t, chan, func, sub))
    rows.sort()
    return rows


def token_kicks(rows):
    """3-kick token windows, identical grouping to gpu_occupancy_timeseries."""
    by_id = defaultdict(list)
    for t, _, _, sub in rows:
        by_id[sub].append(t)
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


def micro_intervals(rows, kick_chan_ids):
    """Begin/end-paired intervals for non-kick submissions, per channel."""
    open_at = {}
    out = []
    for t, chan, func, sub in rows:
        if sub in kick_chan_ids:
            continue
        if func == "1":
            open_at[sub] = (t, chan)
        elif func == "2" and sub in open_at:
            t0, chan0 = open_at.pop(sub)
            out.append((t0, t, chan0, sub))
    out.sort()
    return out


def cluster_bursts(events):
    """Merge micro-sub intervals with gap <= BURST_GAP_S into burst envelopes."""
    bursts = []
    for a, b in events:
        if bursts and a - bursts[-1][1] <= BURST_GAP_S:
            bursts[-1][1] = max(bursts[-1][1], b)
            bursts[-1][2] += 1
        else:
            bursts.append([a, b, 1])
    return bursts


def stream_occupancy(values_xml):
    """Yield (t, occ_value) for the Kernel Occupancy counter, streaming."""
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
            if t is not None and cid == OCC_ID and v is not None:
                yield t, v
            cur.clear()
            el.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True)
    ap.add_argument("--exec-points", required=True)
    ap.add_argument("--shifts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = parse_exec_rows(args.exec_points)
    tokens = token_kicks(rows)
    if not tokens:
        raise SystemExit("no 3-kick tokens")

    # kick submission ids = ids of the big intervals
    by_id = defaultdict(list)
    for t, _, _, sub in rows:
        by_id[sub].append(t)
    kick_ids = {
        sub for sub, ts in by_id.items() if len(ts) >= 2 and max(ts) - min(ts) > 0.002
    }
    micros = micro_intervals(rows, kick_ids)
    print(
        f"tokens: {len(tokens)}; micro-submissions: {len(micros)} "
        f"(dur med={st.median((b - a) for a, b, _, _ in micros) * 1e6:.1f} us, "
        f"p99={sorted((b - a) for a, b, _, _ in micros)[int(len(micros) * 0.99)] * 1e6:.0f} us)"
    )

    # segments: (a, b, kick_idx) sorted; same as A0
    segs = []
    for t in tokens:
        for j, (a, b) in enumerate(t):
            segs.append((a, b, j))
    segs.sort()
    starts = [s[0] for s in segs]

    # per-kick interior 25us occupancy bins (identical to A0 accounting)
    per_kick_bins = defaultdict(lambda: defaultdict(list))
    for t, v in stream_occupancy(args.values):
        i = bisect.bisect_right(starts, t) - 1
        if i < 0:
            continue
        a, b, j = segs[i]
        if not (a + GUARD_S <= t <= b - GUARD_S):
            continue
        per_kick_bins[i][int((t - a) / BIN_S)].append(v)

    # micro-sub intervals + burst envelopes per kick interior
    micro_by_seg = defaultdict(list)
    for a, b, chan, sub in micros:
        i = bisect.bisect_right(starts, a) - 1
        if i < 0:
            continue
        sa, sb, _ = segs[i]
        if a >= sa and b <= sb:
            micro_by_seg[i].append((a, b))

    # ---- alignment stats ----
    valley_dt, nonvalley_dt = [], []
    ctrl_valley_dt = []
    n_valley = n_bins = 0
    burst_cover_s = interior_s = 0.0
    valley_in_burst = 0
    psth_num = defaultdict(float)
    psth_den = defaultdict(int)
    kicks_with_bursts = 0

    for i, bins in sorted(per_kick_bins.items()):
        a, b, j = segs[i]
        items = sorted(bins.items())
        if len(items) < 8:
            continue
        vals = [sum(vs) / len(vs) for _, vs in items]
        keys = [k for k, _ in items]
        p90 = sorted(vals)[int(len(vals) * 0.9)]
        if p90 <= 0:
            continue
        vmask = [x < 0.5 * p90 for x in vals]
        centers = [a + (k + 0.5) * BIN_S for k in keys]

        bursts = cluster_bursts(micro_by_seg.get(i, []))
        interior_s += b - a - 2 * GUARD_S
        burst_cover_s += sum(min(e, b) - max(s, a) for s, e, _ in bursts)
        if not bursts:
            # kicks without any aux activity still count toward valley totals
            n_bins += len(vals)
            n_valley += sum(vmask)
            continue
        kicks_with_bursts += 1

        edges = []
        for s, e, _ in bursts:
            edges.append(s)
            edges.append(e)

        def dist_to_burst(t):
            k = bisect.bisect_left(edges, t)
            if k % 2 == 1:
                return 0.0  # inside an envelope
            cands = []
            if k > 0:
                cands.append(t - edges[k - 1])
            if k < len(edges):
                cands.append(edges[k] - t)
            return min(cands)

        n_bins += len(vals)
        for c, m, x in zip(centers, vmask, vals):
            d = dist_to_burst(c)
            if m:
                n_valley += 1
                valley_dt.append(d)
                if d == 0.0:
                    valley_in_burst += 1
            else:
                nonvalley_dt.append(d)

        # circular-shift control: shift valley mask, keep bursts fixed
        L = len(vmask)
        for _ in range(args.shifts):
            off = rng.randrange(1, L) if L > 1 else 0
            for idx in range(L):
                if vmask[(idx + off) % L]:
                    ctrl_valley_dt.append(dist_to_burst(centers[idx]))

        # PSTH: occupancy vs offset from burst onsets (kick-interior samples)
        for s, e, _ in bursts:
            for c, x in zip(centers, vals):
                off = c - s
                if -250e-6 <= off <= 400e-6:
                    kbin = int(off // BIN_S)
                    psth_num[kbin] += x
                    psth_den[kbin] += 1

    print(f"kicks analyzed: {len(per_kick_bins)}; with aux bursts: {kicks_with_bursts}")
    print(
        f"interior time: {interior_s * 1e3:.1f} ms; burst envelope cover: "
        f"{burst_cover_s * 1e3:.1f} ms ({burst_cover_s / interior_s * 100:.1f}%)"
    )
    print(
        f"valley bins: {n_valley}/{n_bins} ({n_valley / n_bins * 100:.1f}%); "
        f"inside burst envelope: {valley_in_burst} "
        f"({valley_in_burst / max(n_valley, 1) * 100:.1f}%)"
    )

    def frac_near(ds):
        return sum(1 for d in ds if d <= NEAR_S) / max(len(ds), 1)

    fv, fn, fc = (
        frac_near(valley_dt),
        frac_near(nonvalley_dt),
        frac_near(ctrl_valley_dt),
    )
    print(
        f"fraction within +-{NEAR_S * 1e6:.0f} us of a burst: "
        f"valley={fv:.3f} nonvalley={fn:.3f} shifted-control={fc:.3f} "
        f"enrichment(valley/control)={fv / max(fc, 1e-9):.2f}x"
    )
    if valley_dt and ctrl_valley_dt:
        print(
            f"median distance to burst: valley={st.median(valley_dt) * 1e6:.0f} us "
            f"control={st.median(ctrl_valley_dt) * 1e6:.0f} us"
        )

    print("\nPSTH: mean occupancy vs offset from burst onset (25 us bins)")
    for kbin in sorted(psth_num):
        if psth_den[kbin] >= 50:
            off_lo = kbin * BIN_S * 1e6
            m = psth_num[kbin] / psth_den[kbin]
            bar = "#" * int(m / 2)
            print(f"  {off_lo:+7.0f} us: {m:5.1f} |{bar}")


if __name__ == "__main__":
    main()
