#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["lxml>=5"]
# ///
"""Headless Apple GPU performance-limiter capture + per-kick analysis.

One command per experiment: ramp a decode-window workload, record with the
user-saved `metal-counters` Instruments template, export counter and
execution-point tables, join counter samples into exact per-kick windows,
and print/persist the standard table. Designed so "change kernel -> rerun
one command -> diff CSVs" is the whole loop.

Requires (one-time, already done on zekrom):
  ~/Library/Application Support/Instruments/Templates/metal-counters.tracetemplate
  (Metal GPU Counters, Counter Set = Performance Limiters,
   Performance State = Maximum, shader profiler on)

Fast path when iterating on the SAME kernel change many times:

  # 1. warm one workload once and hold at ready:
  scripts/profile/gpu_limiter_capture.py hold --model a3b --ctx 16384

  # 2. run experiments against the printed PID (each ~30s, no ramp):
  scripts/profile/gpu_limiter_capture.py capture --reuse-pid PID --label baseline
  scripts/profile/gpu_limiter_capture.py capture --reuse-pid PID --label nwg192 \\
      --env QWEN_ATTN_V4_NWG=192   # note: env applies only to fresh ramps

  # one-shot ramp + capture per experiment:
  scripts/profile/gpu_limiter_capture.py capture --model a3b --ctx 16384 --label foo

  # re-analyze an existing .trace:
  scripts/profile/gpu_limiter_capture.py analyze --trace /tmp/foo.trace --label foo

Pitfalls encoded here (see docs/bench/2026-07-03-xcode-decode-capture/):
  - recording must END before the target exits (truncated-bundle bug);
    the window is sized so decode outlives --seconds.
  - traced busy fractions are inflated by instrument overhead; use this
    tool for counter attribution, qwen-bench for throughput claims.
  - shader-profiler per-kernel tables are kick-sampling-biased: names are
    metadata; quantitative attribution is the per-kick counter join below.
  - quiet box: any concurrent GPU-heavy process invalidates the run.
  - env vars in --env apply only when this tool RAMPS the workload; when
    reusing a PID via `hold`, set env vars before starting hold.
"""

import argparse
import bisect
import json
import os
import pickle
import subprocess
import sys
import time
from collections import Counter, defaultdict

from lxml import etree as ET  # ~4x faster than stdlib on iterparse

MODELS = {
    "a3b": "~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "27b": "~/models/Qwen3.5-27B-Q4_K_M.gguf",
    "a10b": (
        "~/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/"
        "Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf"
    ),
}
TEMPLATE = "metal-counters"
TEMPLATE_PATH = os.path.expanduser(
    "~/Library/Application Support/Instruments/Templates/metal-counters.tracetemplate"
)
READY = "/tmp/qwen-capture.ready"
GO = "/tmp/qwen-capture.go"
# Human-table subset; the CSV carries all counters.
KEY_COUNTERS = [3, 54, 4, 61, 62, 63, 7, 8, 12, 13, 15, 17, 21, 23, 24, 47, 11, 6]


def sh(cmd, check=False, **kw):
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw
    )
    if check and r.returncode:
        raise SystemExit(f"cmd failed: {cmd}\n{r.stderr}")
    return r


def parse_ts(s):
    mm, rest = s.split(":")
    p = rest.split(".")
    return (
        int(mm) * 60
        + int(p[0])
        + int(p[1]) / 1e3
        + (int(p[2]) if len(p) > 2 else 0) / 1e6
    )


def export(trace, schema, out, extra_pred=""):
    """Export one xctrace table. `extra_pred` is an XPath predicate
    appended inside the [] of the schema selector."""
    xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"{extra_pred}]'
    r = sh(
        [
            "xcrun",
            "xctrace",
            "export",
            "--input",
            trace,
            "--xpath",
            xpath,
            "--output",
            out,
        ]
    )
    if "Finished export" not in (r.stdout + r.stderr):
        raise SystemExit(f"export failed for {schema}: {r.stderr.strip()[:200]}")


def _schema_col_names(path):
    """Return the ordered list of column mnemonics for the table's schema.
    Row elements are tag-typed (uint32, string, ...), so multiple columns
    of the same type collide when keyed by tag; positional mapping via
    the schema is the correct decode."""
    root = ET.parse(path).getroot()
    return [c.findtext("mnemonic") for c in root.findall(".//schema/col")]


def stream_rows(path):
    """Stream <row> elements as schema-column-keyed dicts, resolving
    id/ref interning document-wide. Positional decode is required
    because multiple columns can share a type tag (e.g., counter-info
    has four `uint32` columns)."""
    cols = _schema_col_names(path)
    intern = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "row":
            if "id" in el.attrib:
                intern[(el.tag, el.attrib["id"])] = el.attrib.get("fmt", el.text or "")
            continue
        row = {}
        for i, ch in enumerate(el):
            if i >= len(cols):
                break
            if "ref" in ch.attrib:
                row[cols[i]] = intern.get((ch.tag, ch.attrib["ref"]), "")
            else:
                row[cols[i]] = ch.attrib.get("fmt", ch.text or "")
        yield row
        el.clear()


def kick_windows(exec_points_xml):
    """Bucket GPU-execution-point timestamps by cmdbuf-id, keep intervals
    > 2 ms, and group into per-token kick windows using > 0.8 ms
    inter-kick gaps. Normal decode currently has three large kicks/token;
    noop isolation runs can legitimately remove a kick, so use the dominant
    group size instead of hard-coding 3. The schema calls the
    metal-command-buffer-id column `slot-id`."""
    by_id = defaultdict(list)
    for row in stream_rows(exec_points_xml):
        cb = row.get("slot-id")
        st = row.get("timestamp")
        if cb and st:
            by_id[cb].append(parse_ts(st))
    iv = sorted((min(t), max(t)) for t in by_id.values() if len(t) >= 2)
    big = [x for x in iv if x[1] - x[0] > 0.002]
    groups, cur = [], []
    for a, b in big:
        if cur and a - cur[-1][1] > 0.0008:
            groups.append(cur)
            cur = []
        cur.append((a, b))
    if cur:
        groups.append(cur)
    hist = Counter(len(g) for g in groups)
    if not hist:
        return [], len(iv), len(big), {}, 0
    target = max(hist, key=lambda n: (hist[n], n))
    tokens = [g for g in groups if len(g) == target]
    return tokens, len(iv), len(big), dict(sorted(hist.items())), target


def counter_names(info_xml):
    """Extract {counter_id: human_name}. Schema-column-keyed via
    stream_rows() (the row has four uint32 columns; positional decode
    is required)."""
    out = {}
    for r in stream_rows(info_xml):
        cid = r.get("counter-id")
        name = r.get("name")
        if cid and name:
            out[int(cid.replace(",", ""))] = name
    return out


def sibling_counter_names(info_xml):
    """Reuse counter names from another capture in the same outdir when a
    trace bundle is gone and this capture's filtered info table is empty.
    Counter IDs are template/device metadata, so sibling captures from the
    same limiter suite are valid for offline re-analysis."""
    outdir = os.path.dirname(info_xml) or "."
    for name in sorted(os.listdir(outdir)):
        if not name.endswith(".xml") or "-counter-info" not in name:
            continue
        path = os.path.join(outdir, name)
        if path == info_xml:
            continue
        names = counter_names(path)
        if names:
            return names, path
    return {}, None


def load_counter_names(trace, info_xml):
    """Load counter-id names, falling back to the unfiltered table when
    xctrace's shader-profiler=0 selector produces an empty metadata table.
    Some captures use a different profiler table index even though the
    counter-value export is valid."""
    if not os.path.exists(info_xml):
        export(
            trace, "gpu-counter-info", info_xml, extra_pred=' and @shader-profiler="0"'
        )
    names = counter_names(info_xml)
    if names:
        return names, info_xml
    if not os.path.exists(trace):
        names, sibling = sibling_counter_names(info_xml)
        if names:
            return names, sibling
        raise SystemExit(
            f"missing trace bundle {trace}; cannot re-export empty "
            f"counter-info table {info_xml}"
        )
    base, ext = os.path.splitext(info_xml)
    fallback = f"{base}-all{ext}"
    export(trace, "gpu-counter-info", fallback)
    names = counter_names(fallback)
    if not names:
        raise SystemExit(
            "gpu-counter-info export contained no counter names, even without "
            f"the shader-profiler filter: {info_xml}"
        )
    return names, fallback


def join_counters(values_xml, tokens, n_kicks):
    """Manual streamer for gpu-counter-value (100M+ elements). Skips the
    ElementTree row-dict construction to run faster than stream_rows()
    on this table specifically."""
    segs = []
    for t in tokens:
        for j, (a, b) in enumerate(t):
            segs.append((a, b, j))
    segs.sort()
    starts = [s[0] for s in segs]
    acc = [defaultdict(lambda: [0.0, 0]) for _ in range(n_kicks)]
    dev = defaultdict(lambda: [0.0, 0])
    ts_i, u_i, d_i = {}, {}, {}
    t_cur = cid_cur = v_cur = None
    got_cid = False
    for _, el in ET.iterparse(values_xml, events=("end",)):
        tag = el.tag
        if tag == "event-time":
            if "id" in el.attrib:
                t_cur = int(el.text) / 1e9
                ts_i[el.attrib["id"]] = t_cur
            else:
                t_cur = ts_i.get(el.attrib.get("ref"))
        elif tag == "uint32":
            if not got_cid:
                if "id" in el.attrib:
                    cid_cur = int(el.attrib.get("fmt", "0").replace(",", ""))
                    u_i[el.attrib["id"]] = cid_cur
                else:
                    cid_cur = u_i.get(el.attrib.get("ref"), 0)
                got_cid = True
        elif tag == "fixed-decimal":
            if "id" in el.attrib:
                v_cur = float(el.text)
                d_i[el.attrib["id"]] = v_cur
            else:
                v_cur = d_i.get(el.attrib.get("ref"), 0.0)
        elif tag == "row":
            if t_cur is not None and cid_cur is not None and v_cur is not None:
                d = dev[cid_cur]
                d[0] += v_cur
                d[1] += 1
                i = bisect.bisect_right(starts, t_cur) - 1
                if i >= 0:
                    a, b, j = segs[i]
                    if t_cur <= b:
                        x = acc[j][cid_cur]
                        x[0] += v_cur
                        x[1] += 1
            got_cid = False
            el.clear()
    return acc, dev


def analyze(trace, label, outdir):
    """Analyze a .trace. Exports each XML table only if not already on
    disk (labels are unique per experiment; re-running analyze on the
    same label skips the ~6 min counter-value export)."""
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    pts = f"{outdir}/{label}-exec-points.xml"
    vals = f"{outdir}/{label}-counter-values.xml"
    info = f"{outdir}/{label}-counter-info.xml"
    if not os.path.exists(pts):
        export(trace, "metal-gpu-execution-points", pts)
    if not os.path.exists(vals):
        export(trace, "gpu-counter-value", vals)
    t1 = time.time()
    tokens, n_iv, n_big, kick_hist, n_kicks = kick_windows(pts)
    if len(tokens) < 10:
        raise SystemExit(
            f"only {len(tokens)} {n_kicks}-kick tokens found from "
            f"{n_iv} intervals ({n_big} > 2 ms, hist={kick_hist}). "
            "Wrong workload shape or truncated bundle? Check /tmp/dw-*.log"
        )
    n_kicks = len(tokens[0])
    group_count = sum(kick_hist.values())
    dominant_count = kick_hist.get(n_kicks, 0)
    dominant_fraction = dominant_count / group_count if group_count else 0.0
    if dominant_fraction < 0.5:
        print(
            "WARNING: unstable kick histogram; per-kick rows are topology-biased. "
            "Use device means and external throughput logs for this run.",
            file=sys.stderr,
        )
    names, info_used = load_counter_names(trace, info)
    # Persist the (acc, dev) tuple so subsequent analyze calls on the
    # same label skip the ~1-3 min join. defaultdict-with-lambda is
    # unpicklable; convert to plain dicts.
    join_cache = f"{outdir}/{label}-join.pkl"
    if os.path.exists(join_cache) and os.path.getsize(join_cache) > 0:
        with open(join_cache, "rb") as f:
            acc, dev = pickle.load(f)
        if len(acc) != n_kicks:
            acc, dev = None, None
    else:
        acc, dev = None, None
    if acc is None:
        acc_dd, dev_dd = join_counters(vals, tokens, n_kicks)
        acc = [dict(a) for a in acc_dd]
        dev = dict(dev_dd)
        with open(join_cache, "wb") as f:
            pickle.dump((acc, dev), f, protocol=pickle.HIGHEST_PROTOCOL)
    t2 = time.time()

    import statistics as st

    kd = [st.median([t[j][1] - t[j][0] for t in tokens]) * 1e3 for j in range(n_kicks)]
    print(
        f"\n== {label}: {len(tokens)} tokens, kick medians "
        f"{'/'.join(f'{x:.2f}' for x in kd)} ms  "
        f"(export {t1 - t0:.1f}s, join {t2 - t1:.1f}s)"
    )
    kick_header = "".join(f" {'kick' + str(j):>8}" for j in range(n_kicks))
    print(f"{'id':>3} {'counter':<40}{kick_header} {'device':>8}")
    csv_path = f"{outdir}/{label}-per-kick.csv"
    with open(csv_path, "w") as f:
        kick_cols = ",".join(f"kick{j}" for j in range(n_kicks))
        sample_cols = ",".join(f"samples{j}" for j in range(n_kicks))
        f.write(f"counter_id,counter,{kick_cols},device,{sample_cols}\n")
        for cid in sorted(names):
            row, ns = [], []
            for j in range(n_kicks):
                s, n = acc[j].get(cid, [0, 0])
                row.append(s / n if n else 0.0)
                ns.append(n)
            ds, dn = dev.get(cid, [0, 0])
            dmean = ds / dn if dn else 0.0
            row_csv = ",".join(f"{x:.3f}" for x in row)
            ns_csv = ",".join(str(n) for n in ns)
            f.write(f'{cid},"{names[cid]}",{row_csv},{dmean:.3f},{ns_csv}\n')
            if cid in KEY_COUNTERS:
                row_txt = "".join(f" {x:8.1f}" for x in row)
                print(f"{cid:>3} {names[cid]:<40}{row_txt} {dmean:8.1f}")
    meta = dict(
        label=label,
        trace=trace,
        tokens=len(tokens),
        kick_ms=kd,
        kick_count=n_kicks,
        kick_histogram=kick_hist,
        kick_groups=group_count,
        kick_dominant_fraction=dominant_fraction,
        intervals=n_iv,
        big_intervals=n_big,
        counter_info=info_used,
        xctrace=sh(["xcrun", "xctrace", "version"]).stdout.strip(),
        commit=sh(["git", "rev-parse", "--short", "HEAD"]).stdout.strip(),
        time=time.strftime("%F %T"),
        export_s=round(t1 - t0, 1),
        join_s=round(t2 - t1, 1),
    )
    with open(f"{outdir}/{label}-meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"csv: {csv_path}")
    # Keep exec-points + counter-info for later re-joins. The multi-GB
    # counter-values XML stays too (~3.9 GB per experiment) so re-running
    # analyze on the same label is instant; delete manually if disk
    # pressure matters (`rm outdir/LABEL-counter-values.xml`).


def ramp(model_key, ctx, window, streams=1, env=None):
    if not os.path.exists(TEMPLATE_PATH):
        raise SystemExit(
            f"missing template {TEMPLATE_PATH} - see "
            "docs/bench/2026-07-03-xcode-decode-capture/README.md"
        )
    other = sh("pgrep -fl qwen-bench").stdout.strip()
    if other:
        print(
            f"WARNING: concurrent qwen-bench (quiet-box rule):\n{other}",
            file=sys.stderr,
        )
    model = os.path.expanduser(MODELS[model_key])
    for p in (READY, GO):
        if os.path.exists(p):
            os.unlink(p)
    bench = os.path.join(os.path.dirname(__file__), "../../target/release/qwen-bench")
    ev = dict(os.environ)
    for kv in env or []:
        k, v = kv.split("=", 1)
        ev[k] = v
    log = f"/tmp/dw-{model_key}-hold.log"
    dwlog = open(log, "w")
    proc = subprocess.Popen(
        [
            bench,
            "decode-window",
            "-m",
            model,
            "--target-ctx",
            str(ctx),
            "--window",
            str(window),
            "--streams",
            str(streams),
            "--ready-file",
            READY,
            "--go-file",
            GO,
        ],
        stdout=dwlog,
        stderr=subprocess.STDOUT,
        env=ev,
    )
    print(f"ramping {model_key} to ctx={ctx} streams={streams} (log {log}) ...")
    while not os.path.exists(READY):
        if proc.poll() is not None:
            raise SystemExit(f"decode-window exited during ramp; tail {log}")
        time.sleep(2)
    return proc


def record_and_analyze(pid, label, seconds, outdir):
    trace = f"/tmp/qwen-{label}.trace"
    sh(f"rm -rf '{trace}'")
    rec = subprocess.Popen(
        [
            "xcrun",
            "xctrace",
            "record",
            "--no-prompt",
            "--template",
            TEMPLATE,
            "--attach",
            str(pid),
            "--time-limit",
            f"{seconds}s",
            "--output",
            trace,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(2)
    if not os.path.exists(GO):
        open(GO, "w").close()
    try:
        rec.wait(timeout=seconds + 90)
    except subprocess.TimeoutExpired:
        rec.terminate()
        try:
            out, _ = rec.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            rec.kill()
            out, _ = rec.communicate()
        if os.path.exists(trace):
            print(
                "WARNING: xctrace record timed out after saving the trace; "
                "continuing to analysis",
                file=sys.stderr,
            )
        else:
            raise SystemExit(f"xctrace record timed out; output:\n{out[-4000:]}")
    time.sleep(3)  # xctrace write is async
    analyze(trace, label, outdir)


def cmd_capture(args):
    if args.reuse_pid:
        record_and_analyze(args.reuse_pid, args.label, args.seconds, args.outdir)
        return
    proc = ramp(args.model, args.ctx, args.window, args.streams, args.env)
    try:
        record_and_analyze(proc.pid, args.label, args.seconds, args.outdir)
    finally:
        proc.wait(timeout=600)


def cmd_hold(args):
    proc = ramp(args.model, args.ctx, args.window, args.streams, args.env)
    print(
        f"READY. PID={proc.pid}. Release the window when done:\n"
        f"  touch {GO}\n"
        "Run one or more captures against this PID:\n"
        f"  scripts/profile/gpu_limiter_capture.py capture "
        f"--reuse-pid {proc.pid} --label LABEL"
    )
    proc.wait()


def cmd_analyze(args):
    analyze(args.trace, args.label, args.outdir)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_ramp_args(p):
        p.add_argument("--model", choices=MODELS, default="a3b")
        p.add_argument("--ctx", type=int, default=16384)
        p.add_argument(
            "--window",
            type=int,
            default=1200,
            help="decode tokens; MUST outlive --seconds",
        )
        p.add_argument(
            "--streams",
            type=int,
            default=1,
            help="decode-window streams for concurrency discrimination",
        )
        p.add_argument("--env", action="append", metavar="K=V")

    c = sub.add_parser("capture", help="record + analyze one experiment")
    add_ramp_args(c)
    c.add_argument("--seconds", type=int, default=8)
    c.add_argument("--label", required=True)
    c.add_argument("--outdir", default="target/profiles/gpu-limiters")
    c.add_argument(
        "--reuse-pid",
        type=int,
        default=None,
        help="skip ramp; attach to an existing decode-window PID (from `hold`)",
    )

    h = sub.add_parser("hold", help="ramp workload and wait for captures")
    add_ramp_args(h)

    a = sub.add_parser("analyze", help="analyze an existing .trace bundle")
    a.add_argument("--trace", required=True)
    a.add_argument("--label", required=True)
    a.add_argument("--outdir", default="target/profiles/gpu-limiters")

    args = ap.parse_args()
    dispatch = {"capture": cmd_capture, "hold": cmd_hold, "analyze": cmd_analyze}
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
