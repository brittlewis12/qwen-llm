#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


LAYER_RE = re.compile(
    r"\[prefill-layer-phase\] chunk=(?P<chunk>\d+) start=(?P<start>\d+) "
    r"layer=(?P<layer>\d+) kind=(?P<kind>\S+) phase=(?P<phase>\S+) "
    r"gpu_ms=(?P<gpu_ms>[0-9.]+)"
)
ATTN_RE = re.compile(
    r"\[prefill-attn-phase\] chunk=(?P<chunk>\d+) start=(?P<start>\d+) "
    r"layer=(?P<layer>\d+) phase=(?P<phase>\S+) gpu_ms=(?P<gpu_ms>[0-9.]+)"
)


def iter_lines(paths: list[Path]):
    if not paths:
        yield from sys.stdin
        return
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            yield from handle


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q + 0.5)))
    return ordered[idx]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize qwen prefill phase trace stderr logs."
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    parser.add_argument(
        "--pass-index",
        type=int,
        help="Only summarize one inferred pass index. Passes increment when chunk ids reset.",
    )
    parser.add_argument(
        "--last-pass",
        action="store_true",
        help="Only summarize the final inferred pass, useful for warmup+timed logs.",
    )
    parser.add_argument(
        "--by-layer",
        action="store_true",
        help="Include layer id in the grouping key.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Include min/p50/p95/max timing columns.",
    )
    args = parser.parse_args()

    if args.pass_index is not None and args.last_pass:
        parser.error("--pass-index and --last-pass are mutually exclusive")

    raw_records = []
    pass_index = 0
    last_chunk = None
    for line in iter_lines(args.paths):
        match = LAYER_RE.search(line)
        if match:
            source = "layer"
            kind = match.group("kind")
        else:
            match = ATTN_RE.search(line)
            if not match:
                continue
            source = "attn-detail"
            kind = "attn"
        chunk = int(match.group("chunk"))
        if last_chunk is not None and chunk < last_chunk:
            pass_index += 1
        last_chunk = chunk
        raw_records.append(
            {
                "pass_index": pass_index,
                "source": source,
                "kind": kind,
                "layer": int(match.group("layer")),
                "phase": match.group("phase"),
                "chunk": chunk,
                "start": int(match.group("start")),
                "gpu_ms": float(match.group("gpu_ms")),
            }
        )

    pass_counts: dict[int, int] = defaultdict(int)
    for record in raw_records:
        pass_counts[record["pass_index"]] += 1

    selected_pass = args.pass_index
    if args.last_pass and raw_records:
        selected_pass = max(record["pass_index"] for record in raw_records)

    if selected_pass is not None:
        raw_records = [
            record for record in raw_records if record["pass_index"] == selected_pass
        ]

    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for record in raw_records:
        if args.by_layer:
            key = (
                record["source"],
                record["kind"],
                str(record["layer"]),
                record["phase"],
            )
        else:
            key = (record["source"], record["kind"], record["phase"])
        groups[key].append(record["gpu_ms"])

    total_ms = sum(sum(values) for values in groups.values())
    rows = []
    for key, values in groups.items():
        if args.by_layer:
            source, kind, layer, phase = key
        else:
            source, kind, phase = key
            layer = None
        count = len(values)
        gpu_ms = sum(values)
        rows.append(
            {
                "source": source,
                "kind": kind,
                "layer": layer,
                "phase": phase,
                "count": count,
                "gpu_ms": gpu_ms,
                "pct": (100.0 * gpu_ms / total_ms) if total_ms else 0.0,
                "avg_ms": (gpu_ms / count) if count else 0.0,
                "min_ms": min(values) if values else 0.0,
                "p50_ms": percentile(values, 0.50),
                "p95_ms": percentile(values, 0.95),
                "max_ms": max(values) if values else 0.0,
            }
        )
    rows.sort(key=lambda item: item["gpu_ms"], reverse=True)

    if args.json:
        print(
            json.dumps(
                {
                    "records": len(raw_records),
                    "total_gpu_ms": total_ms,
                    "selected_pass": selected_pass,
                    "pass_counts": dict(sorted(pass_counts.items())),
                    "rows": rows,
                },
                separators=(",", ":"),
            )
        )
        return 0

    columns = ["source", "kind"]
    if args.by_layer:
        columns.append("layer")
    columns += ["phase", "count", "gpu_ms", "pct", "avg_ms"]
    if args.stats:
        columns += ["min_ms", "p50_ms", "p95_ms", "max_ms"]
    print("\t".join(columns))
    for row in rows:
        values = [row["source"], row["kind"]]
        if args.by_layer:
            values.append(row["layer"])
        values += [
            row["phase"],
            str(row["count"]),
            f"{row['gpu_ms']:.2f}",
            f"{row['pct']:.1f}",
            f"{row['avg_ms']:.3f}",
        ]
        if args.stats:
            values += [
                f"{row['min_ms']:.3f}",
                f"{row['p50_ms']:.3f}",
                f"{row['p95_ms']:.3f}",
                f"{row['max_ms']:.3f}",
            ]
        print("\t".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
