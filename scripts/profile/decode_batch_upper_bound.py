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
    rows: dict[str, float]


@dataclass(frozen=True)
class ProjRow:
    component: str
    mode: str
    tokens: int
    avg_gpu_ms_per_tok: float
    saving_ms_per_tok: float


@dataclass(frozen=True)
class MoeSweepRow:
    slot_order: str
    tokens: int
    combined_ms_per_token: float


@dataclass(frozen=True)
class MoeSweep:
    q4_layers: int
    q5_layers: int
    rows: list[MoeSweepRow]


def parse_phase(path: Path) -> PhaseSummary:
    phase_sum: float | None = None
    rows: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if match := re.search(r"phase_sum=([0-9.]+) ms", line):
            phase_sum = float(match.group(1))
        if match := re.search(r"\]   (.+?)\s+([0-9.]+) ms", line):
            raw_name = match.group(1).strip()
            base_name = re.sub(r"\s+\(x\d+\)$", "", raw_name)
            rows[base_name] = float(match.group(2))
    if phase_sum is None:
        raise SystemExit(f"missing phase_sum in {path}")
    return PhaseSummary(phase_sum_ms=phase_sum, rows=rows)


def parse_tsv_rows(path: Path) -> list[dict[str, str]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("["):
            continue
        if line.startswith("component\t") or line.startswith("slot_order\t"):
            header = line.split("\t")
            continue
        if header is None:
            continue
        parts = line.split("\t")
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts, strict=True)))
    return rows


def parse_proj(path: Path) -> dict[int, dict[tuple[str, str], ProjRow]]:
    by_tokens: dict[int, dict[tuple[str, str], ProjRow]] = {}
    for row in parse_tsv_rows(path):
        if "component" not in row:
            continue
        tokens = int(row["tokens"])
        parsed = ProjRow(
            component=row["component"],
            mode=row["mode"],
            tokens=tokens,
            avg_gpu_ms_per_tok=float(row["avg_gpu_ms_per_tok"]),
            saving_ms_per_tok=float(row["saving_ms_per_tok"]),
        )
        by_tokens.setdefault(tokens, {})[(parsed.component, parsed.mode)] = parsed
    if not by_tokens:
        raise SystemExit(f"no decode-proj-batch rows found in {path}")
    return by_tokens


def parse_header_int(line: str, name: str) -> int | None:
    match = re.search(rf"\b{name}=(\d+)", line)
    return int(match.group(1)) if match else None


def parse_moe_sweep(path: Path) -> MoeSweep:
    q4_layers: int | None = None
    q5_layers: int | None = None
    rows: list[MoeSweepRow] = []
    for line in path.read_text().splitlines():
        if line.startswith("[moe-batch-sweep]"):
            q4_layers = parse_header_int(line, "q4_layers")
            q5_layers = parse_header_int(line, "q5_layers")
    for row in parse_tsv_rows(path):
        if "combined_ms_per_token" not in row:
            continue
        rows.append(
            MoeSweepRow(
                slot_order=row.get("slot_order", "exact"),
                tokens=int(row["tokens"]),
                combined_ms_per_token=float(row["combined_ms_per_token"]),
            )
        )
    if q4_layers is None or q5_layers is None:
        raise SystemExit(f"missing q4_layers/q5_layers in {path}")
    if not rows:
        raise SystemExit(f"no moe-batch-sweep rows found in {path}")
    return MoeSweep(q4_layers=q4_layers, q5_layers=q5_layers, rows=rows)


def routed_baseline_ms(phase: PhaseSummary) -> tuple[float, float]:
    gate_up = phase.rows.get("moe ffn routed gate/up")
    down = phase.rows.get("moe ffn routed down")
    if gate_up is None or down is None:
        raise SystemExit(
            "--moe-sweep requires a deep MoE phase profile "
            "(QWEN_PHASE_MOE_FFN_SPLIT=deep)"
        )
    return gate_up, down


def moe_rows_by_tokens(sweep: MoeSweep, slot_order: str) -> dict[int, MoeSweepRow]:
    return {row.tokens: row for row in sweep.rows if row.slot_order == slot_order}


def aggregate_projection_save(rows: dict[tuple[str, str], ProjRow], mode: str) -> float:
    row = rows.get(("aggregate_one_encoder", mode))
    if row is None:
        raise SystemExit(
            f"decode-proj-batch output missing aggregate_one_encoder {mode} row"
        )
    return row.saving_ms_per_tok


def component_save(
    rows: dict[tuple[str, str], ProjRow], component: str, mode: str
) -> float:
    row = rows.get((component, mode))
    return row.saving_ms_per_tok if row is not None else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate charged decode batching upper bound from measured phase and micro rows."
    )
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--proj", type=Path, required=True)
    parser.add_argument("--moe-sweep", type=Path)
    parser.add_argument(
        "--slot-order",
        default="exact",
        help="moe-batch-sweep slot_order to use when --moe-sweep is present",
    )
    parser.add_argument(
        "--projection-mode",
        default="matmat_batch",
        choices=("matmat_batch", "matmat_with_layout"),
        help="decode-proj-batch projection row to use (default: matmat_batch)",
    )
    args = parser.parse_args()

    phase = parse_phase(args.phase)
    proj = parse_proj(args.proj)
    moe_sweep = parse_moe_sweep(args.moe_sweep) if args.moe_sweep else None
    moe_by_tokens = moe_rows_by_tokens(moe_sweep, args.slot_order) if moe_sweep else {}
    routed_gate_up = routed_down = 0.0
    fallback_down_ms = 0.0
    if moe_sweep is not None:
        routed_gate_up, routed_down = routed_baseline_ms(phase)
        fallback_layers = max(0, moe_sweep.q4_layers - moe_sweep.q5_layers)
        fallback_down_ms = routed_down * fallback_layers / max(1, moe_sweep.q4_layers)

    print(
        "tokens\tphase_ms\tprojection_save_ms\trouted_save_ms\tcharged_ms"
        "\tsaved_ms\tsaved_pct_phase\tideal_speedup\tprojection_gdn_save_ms"
        "\tprojection_attn_save_ms\tprojection_shared_ffn_save_ms"
        "\tprojection_lm_head_save_ms\trouted_projected_ms\tfallback_down_ms"
    )
    for tokens in sorted(proj):
        rows = proj[tokens]
        projection_save = aggregate_projection_save(rows, args.projection_mode)
        routed_projected = 0.0
        routed_save = 0.0
        if moe_sweep is not None:
            moe_row = moe_by_tokens.get(tokens)
            if moe_row is None:
                continue
            routed_projected = moe_row.combined_ms_per_token + fallback_down_ms
            routed_save = routed_gate_up + routed_down - routed_projected

        saved_ms = projection_save + routed_save
        charged_ms = phase.phase_sum_ms - saved_ms
        saved_pct = saved_ms / phase.phase_sum_ms * 100.0
        ideal_speedup = phase.phase_sum_ms / charged_ms if charged_ms > 0.0 else 0.0
        gdn_save = component_save(
            rows, "gdn_qkv_z", args.projection_mode
        ) + component_save(rows, "gdn_out", args.projection_mode)
        attn_save = component_save(
            rows, "attn_qkv", args.projection_mode
        ) + component_save(rows, "attn_o", args.projection_mode)
        shared_ffn_save = component_save(
            rows, "ffn_gate_up_dense_or_shared", args.projection_mode
        ) + component_save(rows, "ffn_down_dense_or_shared", args.projection_mode)
        lm_head_save = component_save(rows, "lm_head", args.projection_mode)
        print(
            f"{tokens}\t{phase.phase_sum_ms:.4f}\t{projection_save:.4f}\t"
            f"{routed_save:.4f}\t{charged_ms:.4f}\t{saved_ms:.4f}\t"
            f"{saved_pct:.2f}\t{ideal_speedup:.4f}\t{gdn_save:.4f}\t"
            f"{attn_save:.4f}\t{shared_ffn_save:.4f}\t{lm_head_save:.4f}\t"
            f"{routed_projected:.4f}\t{fallback_down_ms:.4f}"
        )


if __name__ == "__main__":
    main()
