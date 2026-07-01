#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhaseSummary:
    phase_sum_ms: float
    gateup_ms: float
    down_ms: float


@dataclass(frozen=True)
class SweepRow:
    slot_order: str
    tokens: int
    combined_ms_per_token: float


@dataclass(frozen=True)
class SweepSummary:
    q4_layers: int
    q5_layers: int
    rows: list[SweepRow]


def parse_phase(path: Path) -> PhaseSummary:
    phase_sum_ms: float | None = None
    gateup_ms: float | None = None
    down_ms: float | None = None

    for line in path.read_text().splitlines():
        if match := re.search(r"phase_sum=([0-9.]+) ms", line):
            phase_sum_ms = float(match.group(1))
        if "moe ffn gate/up wave" in line:
            gateup_ms = parse_phase_row_ms(line)
        if "moe ffn down wave" in line:
            down_ms = parse_phase_row_ms(line)

    missing = [
        name
        for name, value in (
            ("phase_sum", phase_sum_ms),
            ("moe ffn gate/up wave", gateup_ms),
            ("moe ffn down wave", down_ms),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)} in {path}")

    return PhaseSummary(
        phase_sum_ms=phase_sum_ms,
        gateup_ms=gateup_ms,
        down_ms=down_ms,
    )


def parse_phase_row_ms(line: str) -> float:
    match = re.search(r"\s([0-9.]+) ms\s+\(", line)
    if not match:
        raise SystemExit(f"could not parse phase row ms: {line}")
    return float(match.group(1))


def parse_sweep(path: Path) -> SweepSummary:
    q4_layers: int | None = None
    q5_layers: int | None = None
    rows: list[SweepRow] = []

    header_fields: list[str] | None = None
    for line in path.read_text().splitlines():
        if line.startswith("[moe-batch-sweep]"):
            q4_layers = parse_header_int(line, "q4_layers")
            q5_layers = parse_header_int(line, "q5_layers")
            continue
        if line.startswith("slot_order\t") or line.startswith("tokens\t"):
            header_fields = line.split("\t")
            continue
        if not line or line.startswith("[") or header_fields is None:
            continue
        parts = line.split("\t")
        if len(parts) != len(header_fields):
            continue
        row = dict(zip(header_fields, parts, strict=True))
        slot_order = row.get("slot_order", "exact")
        rows.append(
            SweepRow(
                slot_order=slot_order,
                tokens=int(row["tokens"]),
                combined_ms_per_token=float(row["combined_ms_per_token"]),
            )
        )

    if q4_layers is None or q5_layers is None:
        raise SystemExit(f"missing q4_layers/q5_layers in {path}")
    if not rows:
        raise SystemExit(f"no sweep rows found in {path}")
    return SweepSummary(q4_layers=q4_layers, q5_layers=q5_layers, rows=rows)


def parse_header_int(line: str, name: str) -> int | None:
    match = re.search(rf"\b{name}=(\d+)", line)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the ideal end-to-end ceiling for MoE token batching."
    )
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument(
        "--slot-order",
        default="exact",
        help="Only summarize rows with this slot order (default: exact).",
    )
    args = parser.parse_args()

    phase = parse_phase(args.phase)
    sweep = parse_sweep(args.sweep)
    routed_baseline_ms = phase.gateup_ms + phase.down_ms
    fallback_down_layers = max(0, sweep.q4_layers - sweep.q5_layers)
    fallback_down_ms = phase.down_ms * fallback_down_layers / max(1, sweep.q4_layers)

    print(
        "tokens\tmicro_ms_per_token\tfallback_down_ms\tprojected_routed_ms"
        "\tsaved_ms\tsaved_pct_phase\tideal_speedup"
    )
    for row in sweep.rows:
        if row.slot_order != args.slot_order:
            continue
        projected_ms = row.combined_ms_per_token + fallback_down_ms
        saved_ms = max(0.0, routed_baseline_ms - projected_ms)
        saved_pct = saved_ms / phase.phase_sum_ms * 100.0
        ideal_speedup = phase.phase_sum_ms / (phase.phase_sum_ms - saved_ms)
        print(
            f"{row.tokens}\t{row.combined_ms_per_token:.4f}\t"
            f"{fallback_down_ms:.4f}\t{projected_ms:.4f}\t"
            f"{saved_ms:.4f}\t{saved_pct:.2f}\t{ideal_speedup:.4f}"
        )


if __name__ == "__main__":
    main()
