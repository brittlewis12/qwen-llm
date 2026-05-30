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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q + 0.5)))
    return ordered[idx]


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
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        help="Infer profile passes from node-0 chunk widths and this prompt length.",
    )
    parser.add_argument(
        "--pass-index",
        type=int,
        help="Only summarize one inferred pass index.",
    )
    parser.add_argument(
        "--last-pass",
        action="store_true",
        help="Only summarize the final inferred pass.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Include min/p50/p95/max timing columns.",
    )
    args = parser.parse_args()

    if args.last_pass and args.pass_index is not None:
        parser.error("--last-pass and --pass-index are mutually exclusive")
    if (args.last_pass or args.pass_index is not None) and not args.prompt_tokens:
        parser.error("--last-pass/--pass-index require --prompt-tokens")

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    records = 0
    pass_counts: dict[int, int] = defaultdict(int)
    pass_index = 0
    pass_tokens = 0
    for line in iter_lines(args.paths):
        match = PROFILE_RE.search(line)
        if not match:
            continue
        if args.prompt_tokens and int(match.group("node")) == 0:
            ne = [int(part.strip()) for part in match.group("ne").split(",")]
            chunk_tokens = ne[1] if len(ne) > 1 else 0
            if pass_tokens >= args.prompt_tokens:
                pass_index += 1
                pass_tokens = 0
            pass_tokens += chunk_tokens
        pass_counts[pass_index] += 1
        key = (match.group("op"), canonical_name(match.group("name")))
        groups[key].append(float(match.group("gpu_ms")))
        records += 1

    selected_pass = args.pass_index
    if args.last_pass:
        selected_pass = max(pass_counts.keys(), default=0)

    if selected_pass is not None:
        groups = defaultdict(list)
        records = 0
        pass_counts = defaultdict(int)
        pass_index = 0
        pass_tokens = 0
        for line in iter_lines(args.paths):
            match = PROFILE_RE.search(line)
            if not match:
                continue
            if args.prompt_tokens and int(match.group("node")) == 0:
                ne = [int(part.strip()) for part in match.group("ne").split(",")]
                chunk_tokens = ne[1] if len(ne) > 1 else 0
                if pass_tokens >= args.prompt_tokens:
                    pass_index += 1
                    pass_tokens = 0
                pass_tokens += chunk_tokens
            pass_counts[pass_index] += 1
            if pass_index != selected_pass:
                continue
            key = (match.group("op"), canonical_name(match.group("name")))
            groups[key].append(float(match.group("gpu_ms")))
            records += 1

    total_ms = sum(sum(values) for values in groups.values())
    rows = []
    for (op, name), values in groups.items():
        count = len(values)
        gpu_ms = sum(values)
        row = {
            "op": op,
            "name": name,
            "count": count,
            "gpu_ms": gpu_ms,
            "pct": (100.0 * gpu_ms / total_ms) if total_ms else 0.0,
            "avg_ms": (gpu_ms / count) if count else 0.0,
        }
        if args.stats:
            row.update(
                {
                    "min_ms": min(values) if values else 0.0,
                    "p50_ms": percentile(values, 0.50),
                    "p95_ms": percentile(values, 0.95),
                    "max_ms": max(values) if values else 0.0,
                }
            )
        rows.append(row)
    rows.sort(key=lambda item: item["gpu_ms"], reverse=True)

    if args.json:
        print(
            json.dumps(
                {
                    "records": records,
                    "total_gpu_ms": total_ms,
                    "selected_pass": selected_pass,
                    "pass_counts": dict(sorted(pass_counts.items())),
                    "rows": rows,
                },
                separators=(",", ":"),
            )
        )
        return 0

    columns = ["op", "name", "count", "gpu_ms", "pct", "avg_ms"]
    if args.stats:
        columns.extend(["min_ms", "p50_ms", "p95_ms", "max_ms"])
    print("\t".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if column == "pct":
                    values.append(f"{value:.1f}")
                elif column == "avg_ms":
                    values.append(f"{value:.3f}")
                else:
                    values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        print("\t".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
