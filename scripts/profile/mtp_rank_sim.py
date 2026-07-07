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


def maybe_clamp(value: int, max_emitted: int | None) -> int:
    if max_emitted is None:
        return value
    return min(value, max_emitted)


def base_result(
    steps: list[list[dict]], max_emitted: int | None = None
) -> tuple[int, int]:
    # Each step emits the carry plus all accepted draft rows.
    emitted = 0
    steps_used = 0
    for step in steps:
        if max_emitted is not None and emitted >= max_emitted:
            break
        steps_used += 1
        emitted += 1
        for row in step:
            if not row["accepted"]:
                break
            emitted += 1
            if max_emitted is not None and emitted >= max_emitted:
                return max_emitted, steps_used
    return maybe_clamp(emitted, max_emitted), steps_used


def base_emitted(steps: list[list[dict]], max_emitted: int | None = None) -> int:
    emitted, _ = base_result(steps, max_emitted)
    return emitted


def terminal_mismatches(
    steps: list[list[dict]], max_emitted: int | None = None
) -> list[dict]:
    out: list[dict] = []
    emitted = 0
    for step in steps:
        if max_emitted is not None and emitted >= max_emitted:
            break
        emitted += 1
        if max_emitted is not None and emitted >= max_emitted:
            break
        for row in step:
            if row["accepted"]:
                emitted += 1
                if max_emitted is not None and emitted >= max_emitted:
                    break
                continue
            else:
                out.append(row)
                break
    return out


def oracle_one_rescue(
    steps: list[list[dict]], k: int, max_emitted: int | None = None
) -> int:
    emitted, _ = oracle_one_rescue_result(steps, k, max_emitted)
    return emitted


def oracle_one_rescue_result(
    steps: list[list[dict]], k: int, max_emitted: int | None = None
) -> tuple[int, int]:
    emitted = 0
    steps_used = 0
    for step in steps:
        if max_emitted is not None and emitted >= max_emitted:
            break
        steps_used += 1
        emitted += 1
        for row in step:
            if row["accepted"]:
                emitted += 1
                if max_emitted is not None and emitted >= max_emitted:
                    return max_emitted, steps_used
                continue
            target = int(row["target_tok"])
            if target in [int(t) for t in row.get("top_tokens", [])[:k]]:
                emitted += 1
                if max_emitted is not None and emitted >= max_emitted:
                    return max_emitted, steps_used
            break
    return maybe_clamp(emitted, max_emitted), steps_used


def simulate_margin_swap(
    steps: list[list[dict]], tau: float, alt_idx: int, max_emitted: int | None = None
) -> int:
    emitted, _ = simulate_margin_swap_result(steps, tau, alt_idx, max_emitted)
    return emitted


def simulate_margin_swap_result(
    steps: list[list[dict]], tau: float, alt_idx: int, max_emitted: int | None = None
) -> tuple[int, int]:
    emitted = 0
    steps_used = 0
    for step in steps:
        if max_emitted is not None and emitted >= max_emitted:
            break
        steps_used += 1
        emitted += 1  # carry
        for row in step:
            choose_idx = alt_idx if margin(row, alt_idx) <= tau else 0
            chosen = tok_at(row, choose_idx)
            if chosen == int(row["target_tok"]):
                emitted += 1
                if max_emitted is not None and emitted >= max_emitted:
                    return max_emitted, steps_used
                # If we rescued a terminal mismatch with a non-top1 token, the
                # trace has no alternate continuation. Credit one token and end.
                if choose_idx != 0 and not row["accepted"]:
                    break
                continue
            break
    return maybe_clamp(emitted, max_emitted), steps_used


def summary_emitted(path: Path) -> int:
    row = json.loads(path.read_text())
    return int(row["speculative"]["emitted"])


def print_result(label: str, emitted: int, steps_used: int) -> None:
    print(f"{label}\t{emitted}\t{emitted / steps_used:.3f}\tsteps_used={steps_used}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--max-emitted", type=int)
    ap.add_argument("--alts", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    max_emitted = args.max_emitted
    if args.summary is not None:
        summary_max = summary_emitted(args.summary)
        if max_emitted is not None and max_emitted != summary_max:
            raise SystemExit(
                f"--max-emitted {max_emitted} disagrees with --summary {summary_max}"
            )
        max_emitted = summary_max

    rows = load_rows(args.trace)
    steps = group_steps(rows)
    n_steps = len(steps)
    base, base_steps = base_result(steps, max_emitted)
    mismatches = terminal_mismatches(steps, max_emitted)

    print(f"steps\t{n_steps}")
    print(f"rows\t{len(rows)}")
    if max_emitted is not None:
        print(f"max_emitted\t{max_emitted}")
    print(f"terminal_mismatches\t{len(mismatches)}")
    print_result("base_emitted", base, base_steps)

    for k in [2, 4, 8, 16]:
        emitted, steps_used = oracle_one_rescue_result(steps, k, max_emitted)
        print_result(f"oracle_one_rescue_top{k}", emitted, steps_used)

    margins = sorted({margin(row, alt) for row in rows for alt in args.alts})
    for alt in args.alts:
        best = (base, base_steps, float("-inf"))
        for tau in margins:
            emitted, steps_used = simulate_margin_swap_result(
                steps, tau, alt, max_emitted
            )
            if emitted / steps_used > best[0] / best[1]:
                best = (emitted, steps_used, tau)
        print(
            f"best_margin_swap_top{alt + 1}\t{best[0]}\t"
            f"{best[0] / best[1]:.3f}\tsteps_used={best[1]}\ttau={best[2]:.6f}"
        )


if __name__ == "__main__":
    main()
