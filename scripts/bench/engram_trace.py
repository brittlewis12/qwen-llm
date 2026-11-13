#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Summarize QWEN_V41_ENGRAM_TRACE row/access files.

The trace is two files written by the opt-in Engram read hook:

  <prefix>.rows   nine-byte records: u8 layer, u64 little-endian row id
  <prefix>.times  sixteen-byte records: u64 epoch_ns, u64 rows written so far

Rows and times are appended under one lock, so row-file order is call order.
Each read call emits one time record after its rows. This script reports the
distinct working set, reuse, hot-set coverage, page footprint, and growth with
tokens served. It is measurement-only; no product code depends on it.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

# This artifact's companion tables: blk.1 at file offset 416 and blk.14 at
# 55,296,888,608, both IQ4_NL with a 144-byte row. Override for other files.
DEFAULT_BASES = {1: 416, 14: 55_296_888_608}
COVERAGE = (0.90, 0.95, 0.99)


def read_rows(path):
    dtype = np.dtype(
        {
            "names": ["layer", "id"],
            "formats": ["u1", "<u8"],
            "offsets": [0, 1],
            "itemsize": 9,
        }
    )
    raw = np.fromfile(path, dtype=dtype)
    return raw["layer"].astype(np.int64), raw["id"].astype(np.int64)


def read_times(path):
    dtype = np.dtype([("stamp", "<u8"), ("rows", "<u8")])
    raw = np.fromfile(path, dtype=dtype)
    return raw["stamp"].astype(np.float64) / 1e9, raw["rows"]


def key_of(layer, row):
    return (layer << 40) | row


def page_of(layer, row, bases, row_bytes, page_bytes):
    base = np.vectorize(bases.__getitem__, otypes=[np.int64])(layer)
    return (base + row * row_bytes) // page_bytes


def hot_sets(keys, counts, bases, row_bytes, page_bytes):
    order = np.argsort(counts)[::-1]
    ranked = counts[order]
    cumulative = np.cumsum(ranked)
    total = int(ranked.sum())
    result = {}
    for fraction in COVERAGE:
        n = int(np.searchsorted(cumulative, fraction * total, side="left")) + 1
        hot = keys[order[:n]]
        layer = hot >> 40
        row = hot & ((1 << 40) - 1)
        pages = np.unique(page_of(layer, row, bases, row_bytes, page_bytes))
        result[f"{int(fraction * 100)}"] = {
            "distinct_rows": n,
            "logical_bytes": n * row_bytes,
            "pages": int(pages.size),
            "resident_bytes": int(pages.size) * page_bytes,
            "access_share": float(ranked[:n].sum() / total),
        }
    return result


def reuse(counts):
    total = int(counts.sum())
    distinct = int(counts.size)
    once = int(np.count_nonzero(counts == 1))
    sorted_counts = np.sort(counts)
    result = {
        "distinct_rows": distinct,
        "accesses": total,
        "mean_reuse": float(total / distinct),
        "singletons": once,
        "singleton_share": float(once / distinct),
        "access_share_top_1pct": float(
            np.sort(counts)[::-1][: max(1, math.ceil(distinct / 100))].sum() / total
        ),
        "histogram": {},
    }
    edges = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 64), (65, None)]
    for low, high in edges:
        if high is None:
            mask = sorted_counts >= low
            label = f"{low}+"
        elif low == high:
            mask = sorted_counts == low
            label = str(low)
        else:
            mask = (sorted_counts >= low) & (sorted_counts <= high)
            label = f"{low}-{high}"
        rows = int(np.count_nonzero(mask))
        result["histogram"][label] = {
            "rows": rows,
            "row_share": rows / distinct,
            "access_share": float(sorted_counts[mask].sum() / total),
        }
    return result


def load_calls(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            first, end = row.get("first"), row.get("end")
            if first is None or end is None or end <= first:
                continue
            tokens = row.get("window_tokens") or row.get("generated") or 0
            rows.append((float(first), float(end), float(tokens)))
    return rows


def tokens_between(calls, begin, end, at):
    """Rough cumulative generated tokens at `at`, clipped to [begin, end]."""
    total = 0.0
    for first, last, tokens in calls:
        lo, hi = max(first, begin), min(last, end)
        if hi <= lo or at <= lo:
            continue
        total += tokens * min(1.0, (at - lo) / (hi - lo))
    return total


def growth(layer, row, times, begin, end, calls):
    stamps, written = times
    in_window = (stamps >= begin) & (stamps < end)
    if not np.any(in_window):
        return {}
    indices = np.flatnonzero(in_window)
    first, last = int(indices[0]), int(indices[-1])
    start = int(written[first - 1]) if first > 0 else 0
    stop = int(written[last])
    window_stamps = stamps[first : last + 1]
    window_written = written[first : last + 1]
    points = []
    for step in range(1, 11):
        at = begin + (end - begin) * step / 10
        upto = int(np.searchsorted(window_stamps, at, side="right"))
        upto_rows = int(window_written[upto - 1]) if upto > 0 else start
        upto_rows = max(0, min(upto_rows, stop))
        count = upto_rows - start
        keys = key_of(layer[start:upto_rows], row[start:upto_rows])
        distinct = int(np.unique(keys).size) if count else 0
        tokens = tokens_between(calls, begin, end, at)
        points.append(
            {
                "elapsed_s": at - begin,
                "tokens": tokens,
                "distinct_rows": distinct,
                "distinct_per_1k_tokens": distinct / (tokens / 1000)
                if tokens
                else None,
            }
        )
    x = np.array([point["tokens"] for point in points], dtype=float)
    y = np.array([point["distinct_rows"] for point in points], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 and x[-1] > x[0] else None
    return {"points": points, "distinct_rows_per_token": slope}


def disk_read_bytes(calls_path, begin=None, end=None):
    total = 0
    with Path(calls_path).open() as handle:
        for line in handle:
            row = json.loads(line)
            first = row.get("first")
            if begin is not None and (first is None or not (begin <= first < end)):
                continue
            phases = row.get("phases", {})
            for name in ("process_token_io", "process_prefill_io"):
                total += int(
                    phases.get(name, {})
                    .get("process_disk_read_bytes", {})
                    .get("total", 0)
                )
    return total


def prefill_stats(path, begin, end):
    rows = wall = disk = calls = 0
    means = []
    for entry in json.loads(Path(path).read_text()):
        stamp = entry.get("time")
        if stamp is not None and not (begin <= stamp < end):
            continue
        line = entry.get("line", "")
        if not line.startswith("serve native prefill timing: "):
            continue
        profile = json.loads(line.split(": ", 1)[1])["profile"]
        count = profile["engram_read_rows"]
        if not count:
            continue
        calls += 1
        rows += count
        wall += profile["engram_read_wall_ms"]
        disk += profile["process_disk_read_bytes"] or 0
        means.append(profile["engram_read_wall_ms"] * 1000 / count)
    if not rows:
        return {}
    mean_us = wall * 1000 / rows
    hit = max(0.0, min(1.0, (65.0 - mean_us) / (65.0 - 0.5)))
    buckets = {}
    for low, high in ((0, 1), (1, 2), (2, 5), (5, 20), (20, 100), (100, 1e9)):
        buckets[f"{low}-{high if high < 1e9 else 'inf'}"] = sum(
            1 for mean in means if low <= mean < high
        )
    return {
        "calls": calls,
        "rows": rows,
        "wall_ms": wall,
        "us_per_row": mean_us,
        "disk_read_bytes": disk,
        "estimated_cached_share": hit,
        "call_buckets": buckets,
    }


def residency_stats(path, begin, end):
    samples = [json.loads(line) for line in Path(path).open()]
    if not samples:
        return {}
    before = [s for s in samples if s["time"] <= begin]
    during = [s for s in samples if s["time"] <= end]
    post = [s for s in samples if s["time"] > end]
    start = before[-1] if before else samples[0]
    stop = during[-1] if during else samples[-1]
    window = [s for s in samples if begin <= s["time"] <= end]
    return {
        "start": {k: start[k] for k in ("time", "resident_pages", "resident_bytes")},
        "end": {k: stop[k] for k in ("time", "resident_pages", "resident_bytes")},
        "post": {k: post[0][k] for k in ("time", "resident_pages", "resident_bytes")}
        if post
        else None,
        "new_pages": stop["resident_pages"] - start["resident_pages"],
        "window_min_pages": min((s["resident_pages"] for s in window), default=None),
        "window_max_pages": max((s["resident_pages"] for s in window), default=None),
        "vm_start": start["vm"],
        "vm_end": stop["vm"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--calls", type=Path)
    parser.add_argument("--native-lines", type=Path)
    parser.add_argument("--residency", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--row-bytes", type=int, default=144)
    parser.add_argument("--page-bytes", type=int, default=16384)
    parser.add_argument(
        "--layer-base",
        action="append",
        default=[],
        metavar="LAYER:OFFSET",
        help="file offset of a layer's first row (repeatable)",
    )
    parser.add_argument("--window-begin", type=float)
    parser.add_argument("--window-end", type=float)
    parser.add_argument(
        "--whole-session",
        action="store_true",
        help="analyze every traced row instead of the measurement window",
    )
    args = parser.parse_args()

    bases = dict(DEFAULT_BASES)
    for item in args.layer_base:
        layer, _, offset = item.partition(":")
        bases[int(layer)] = int(offset)
    if args.summary:
        summary = json.loads(args.summary.read_text())
        args.window_begin = summary["window_begin"]
        args.window_end = summary["window_end"]
    if not args.whole_session and (
        args.window_begin is None or args.window_end is None
    ):
        parser.error(
            "give --summary or both --window-begin/--window-end, or --whole-session"
        )

    layer, row = read_rows(Path(f"{args.prefix}.rows"))
    times, written = read_times(Path(f"{args.prefix}.times"))
    all_layer, all_row = layer, row
    all_times, all_written = times, written
    if len(times) and int(written[-1]) != len(layer):
        raise SystemExit(
            f"trace row count {len(layer)} != final time count {int(written[-1])}"
        )

    if not args.whole_session:
        stamps = times
        in_window = (stamps >= args.window_begin) & (stamps < args.window_end)
        indices = np.flatnonzero(in_window)
        if not indices.size:
            raise SystemExit("no trace calls inside the requested window")
        first, last = int(indices[0]), int(indices[-1])
        start = int(written[first - 1]) if first > 0 else 0
        stop = int(written[last])
        layer, row = layer[start:stop], row[start:stop]
        times, written = stamps[first : last + 1], written[first : last + 1]

    keys = key_of(layer, row)
    unique, counts = np.unique(keys, return_counts=True)
    total = int(keys.size)
    pages = np.unique(
        page_of(
            unique >> 40,
            unique & ((1 << 40) - 1),
            bases,
            args.row_bytes,
            args.page_bytes,
        )
    )

    result = {
        "prefix": str(args.prefix),
        "window": None
        if args.whole_session
        else {"begin": args.window_begin, "end": args.window_end},
        "accesses": total,
        "distinct_rows": int(unique.size),
        "logical_bytes": total * args.row_bytes,
        "distinct_logical_bytes": int(unique.size) * args.row_bytes,
        "distinct_pages": int(pages.size),
        "distinct_resident_bytes": int(pages.size) * args.page_bytes,
        "by_layer": {},
        "reuse": reuse(counts),
        "hot_sets": hot_sets(unique, counts, bases, args.row_bytes, args.page_bytes),
    }
    for layer_id in np.unique(layer):
        mask = layer == layer_id
        result["by_layer"][str(int(layer_id))] = {
            "accesses": int(np.count_nonzero(mask)),
            "distinct_rows": int(np.unique(row[mask]).size),
        }
    if args.calls:
        calls = load_calls(args.calls)
        session_disk = disk_read_bytes(args.calls)
        window_disk = disk_read_bytes(args.calls, args.window_begin, args.window_end)
        result["disk"] = {
            "session_read_bytes": session_disk,
            "window_read_bytes": window_disk,
            "window_bytes_per_access": window_disk / total if total else None,
        }
        if args.native_lines:
            prefill = prefill_stats(
                args.native_lines, args.window_begin, args.window_end
            )
            result["prefill"] = prefill
            result["decode"] = {
                "rows": total - prefill.get("rows", 0),
                "disk_read_bytes": window_disk - prefill.get("disk_read_bytes", 0),
            }
        if args.residency:
            result["residency"] = residency_stats(
                args.residency, args.window_begin, args.window_end
            )
            new_pages = result["residency"].get("new_pages", 0)
            result["residency"]["retained_share_of_distinct_pages"] = (
                new_pages / int(pages.size) if pages.size else None
            )
        result["growth_with_tokens"] = growth(
            all_layer,
            all_row,
            (all_times, all_written),
            args.window_begin if not args.whole_session else all_times[0],
            args.window_end if not args.whole_session else all_times[-1],
            calls,
        )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
