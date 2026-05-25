#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


PROFILE_RE = re.compile(
    r"ggml_metal_profile_op: node=(?P<node>\d+) op=(?P<op>\S+) "
    r"name=(?P<name>.*?) type=(?P<type>\S+) ne=\[(?P<ne>[^\]]+)\] "
    r"gpu_ms=(?P<gpu_ms>[0-9.]+)"
)
LAYER_SUFFIX_RE = re.compile(r"-\d+(?:\s|$)")


def iter_lines(paths: list[Path]):
    if not paths:
        yield from sys.stdin
        return
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            yield from handle


def canonical_name(name: str) -> str:
    name = name.strip()
    name = LAYER_SUFFIX_RE.sub("-* ", name, count=1).strip()
    for prefix in (
        "ffn_gate",
        "ffn_up",
        "ffn_swiglu",
        "ffn_out",
        "l_out",
        "attn_norm",
        "attn_post_norm",
        "attn_residual",
        "linear_attn_out",
        "conv_output_raw",
        "conv_output_silu",
        "conv_input",
        "conv_state_update",
        "beta_sigmoid",
        "a_softplus",
        "q_conv_predelta",
        "k_conv_predelta",
        "cache_s_l",
        "cache_r_l",
        "__fgdn_ch__",
    ):
        if name.startswith(prefix):
            return prefix
    if name.startswith("z-"):
        return "z"
    if name.startswith("norm-"):
        return "norm"
    if name.startswith("node_"):
        return "node"
    return name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize llama.cpp GGML_METAL_PROFILE_OPS logs."
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    args = parser.parse_args()

    groups: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "gpu_ms": 0.0}
    )
    records = 0
    for line in iter_lines(args.paths):
        match = PROFILE_RE.search(line)
        if not match:
            continue
        key = (match.group("op"), canonical_name(match.group("name")))
        groups[key]["count"] += 1.0
        groups[key]["gpu_ms"] += float(match.group("gpu_ms"))
        records += 1

    total_ms = sum(item["gpu_ms"] for item in groups.values())
    rows = []
    for (op, name), item in groups.items():
        count = int(item["count"])
        gpu_ms = item["gpu_ms"]
        rows.append(
            {
                "op": op,
                "name": name,
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

    print("op\tname\tcount\tgpu_ms\tpct\tavg_ms")
    for row in rows:
        print(
            f"{row['op']}\t{row['name']}\t{row['count']}\t"
            f"{row['gpu_ms']:.2f}\t{row['pct']:.1f}\t{row['avg_ms']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
