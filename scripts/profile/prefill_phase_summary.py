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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize qwen prefill phase trace stderr logs."
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    args = parser.parse_args()

    groups: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "gpu_ms": 0.0}
    )
    records = 0
    for line in iter_lines(args.paths):
        match = LAYER_RE.search(line)
        if match:
            key = ("layer", match.group("kind"), match.group("phase"))
        else:
            match = ATTN_RE.search(line)
            if not match:
                continue
            key = ("attn-detail", "attn", match.group("phase"))
        groups[key]["count"] += 1.0
        groups[key]["gpu_ms"] += float(match.group("gpu_ms"))
        records += 1

    total_ms = sum(item["gpu_ms"] for item in groups.values())
    rows = []
    for (source, kind, phase), item in groups.items():
        count = int(item["count"])
        gpu_ms = item["gpu_ms"]
        rows.append(
            {
                "source": source,
                "kind": kind,
                "phase": phase,
                "count": count,
                "gpu_ms": gpu_ms,
                "pct": (100.0 * gpu_ms / total_ms) if total_ms else 0.0,
                "avg_ms": (gpu_ms / count) if count else 0.0,
            }
        )
    rows.sort(key=lambda item: item["gpu_ms"], reverse=True)

    if args.json:
        print(json.dumps({"records": records, "total_gpu_ms": total_ms, "rows": rows}))
        return 0

    print("source\tkind\tphase\tcount\tgpu_ms\tpct\tavg_ms")
    for row in rows:
        print(
            f"{row['source']}\t{row['kind']}\t{row['phase']}\t{row['count']}\t"
            f"{row['gpu_ms']:.2f}\t{row['pct']:.1f}\t{row['avg_ms']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
