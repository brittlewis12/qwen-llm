#!/usr/bin/env python3
"""Simulate simple MTP top-k rescue policies from rank JSONL traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_steps(rows: list[dict]) -> list[list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["step"])].append(row)
    return [sorted(grouped[k], key=lambda r: int(r["depth"])) for k in sorted(grouped)]


def tok_at(row: dict, idx: int) -> int | None:
    toks = row.get("top_tokens") or []
    if idx >= len(toks):
        return None
    return int(toks[idx])


def margin(row: dict, idx: int = 1) -> float:
    vals = row.get("top_logits") or []
    if len(vals) <= idx:
        return float("inf")
    return float(vals[0]) - float(vals[idx])


def base_emitted(steps: list[list[dict]]) -> int:
    # Each step emits the carry plus all accepted draft rows.
    return sum(1 + sum(1 for row in step if row["accepted"]) for step in steps)


def terminal_mismatches(steps: list[list[dict]]) -> list[dict]:
    out: list[dict] = []
    for step in steps:
        for row in step:
            if not row["accepted"]:
                out.append(row)
                break
    return out


def oracle_one_rescue(steps: list[list[dict]], k: int) -> int:
    emitted = base_emitted(steps)
    for row in terminal_mismatches(steps):
        target = int(row["target_tok"])
        if target in [int(t) for t in row.get("top_tokens", [])[:k]]:
            emitted += 1
    return emitted


def simulate_margin_swap(steps: list[list[dict]], tau: float, alt_idx: int) -> int:
    emitted = 0
    for step in steps:
        emitted += 1  # carry
        for row in step:
            choose_idx = alt_idx if margin(row, alt_idx) <= tau else 0
            chosen = tok_at(row, choose_idx)
            if chosen == int(row["target_tok"]):
                emitted += 1
                # If we rescued a terminal mismatch with a non-top1 token, the
                # trace has no alternate continuation. Credit one token and end.
                if choose_idx != 0 and not row["accepted"]:
                    break
                continue
            break
    return emitted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--alts", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    rows = load_rows(args.trace)
    steps = group_steps(rows)
    n_steps = len(steps)
    base = base_emitted(steps)
    mismatches = terminal_mismatches(steps)

    print(f"steps\t{n_steps}")
    print(f"rows\t{len(rows)}")
    print(f"terminal_mismatches\t{len(mismatches)}")
    print(f"base_emitted\t{base}\t{base / n_steps:.3f}")

    for k in [2, 4, 8, 16]:
        emitted = oracle_one_rescue(steps, k)
        print(f"oracle_one_rescue_top{k}\t{emitted}\t{emitted / n_steps:.3f}")

    margins = sorted({margin(row, alt) for row in rows for alt in args.alts})
    for alt in args.alts:
        best = (base, float("-inf"))
        for tau in margins:
            emitted = simulate_margin_swap(steps, tau, alt)
            if emitted > best[0]:
                best = (emitted, tau)
        print(
            f"best_margin_swap_top{alt + 1}\t{best[0]}\t"
            f"{best[0] / n_steps:.3f}\ttau={best[1]:.6f}"
        )


if __name__ == "__main__":
    main()
