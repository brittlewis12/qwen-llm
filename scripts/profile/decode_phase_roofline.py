#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


DTYPE_BYTES_PER_BLOCK: dict[str, tuple[int, int]] = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q8_0": (32, 34),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
}


@dataclass(frozen=True)
class TensorRow:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class PhaseRow:
    raw_name: str
    base_name: str
    ms: float
    pct: float


def ggml_nbytes(dtype: str, shape: tuple[int, ...]) -> int | None:
    layout = DTYPE_BYTES_PER_BLOCK.get(dtype)
    if layout is None:
        return None
    block, block_bytes = layout
    elems = math.prod(shape)
    if elems % block != 0:
        return None
    return elems // block * block_bytes


def parse_tensors(path: Path) -> list[TensorRow]:
    rows: list[TensorRow] = []
    for line in path.read_text().splitlines():
        match = re.match(
            r"\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|",
            line,
        )
        if not match:
            continue
        name, dtype, shape = (part.strip() for part in match.groups())
        if name == "Name":
            continue
        dims = tuple(int(part) for part in shape.split(","))
        nbytes = ggml_nbytes(dtype, dims)
        if nbytes is None:
            continue
        rows.append(TensorRow(name=name, dtype=dtype, shape=dims, nbytes=nbytes))
    return rows


def parse_metadata_int(path: Path | None, key: str) -> int | None:
    if path is None:
        return None
    pattern = re.compile(rf"\|\s*\d+\s*\|\s*{re.escape(key)}\s*\|\s*(\d+)\s*")
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def parse_phase(path: Path) -> tuple[float, list[PhaseRow]]:
    phase_sum: float | None = None
    raw_rows: list[tuple[str, str, float]] = []
    for line in path.read_text().splitlines():
        sum_match = re.search(r"phase_sum=([0-9.]+) ms", line)
        if sum_match:
            phase_sum = float(sum_match.group(1))
        row_match = re.search(r"\]   (.+?)\s+([0-9.]+) ms", line)
        if row_match:
            raw_name = row_match.group(1).strip()
            base_name = re.sub(r"\s+\(x\d+\)$", "", raw_name)
            raw_rows.append((raw_name, base_name, float(row_match.group(2))))
    if phase_sum is None:
        raise SystemExit(f"no phase_sum found in {path}")
    return phase_sum, [
        PhaseRow(raw_name=raw, base_name=base, ms=ms, pct=ms / phase_sum * 100.0)
        for raw, base, ms in raw_rows
    ]


def tensor_sum(rows: list[TensorRow], pred, scale: float = 1.0) -> int:
    return int(sum(row.nbytes * scale for row in rows if pred(row)))


def category_bytes(
    rows: list[TensorRow],
    expert_used: int,
    expert_count: int,
) -> dict[str, tuple[int, str]]:
    expert_scale = expert_used / expert_count
    moe_routed = tensor_sum(
        rows,
        lambda r: any(
            part in r.name
            for part in (
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            )
        ),
        expert_scale,
    )
    moe_shared = tensor_sum(
        rows,
        lambda r: any(
            part in r.name
            for part in (
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
            )
        ),
    )
    return {
        "gdn front proj": (
            tensor_sum(
                rows,
                lambda r: any(
                    part in r.name
                    for part in (
                        "attn_qkv.weight",
                        "attn_gate.weight",
                        "ssm_alpha.weight",
                        "ssm_beta.weight",
                    )
                ),
            ),
            "weights only; excludes activations",
        ),
        "gdn out_proj": (
            tensor_sum(rows, lambda r: "ssm_out.weight" in r.name),
            "weights only; excludes activations",
        ),
        "attn mixer": (
            tensor_sum(
                rows,
                lambda r: any(
                    part in r.name
                    for part in (
                        "attn_q.weight",
                        "attn_k.weight",
                        "attn_v.weight",
                        "attn_output.weight",
                    )
                ),
            ),
            "projection weights only; excludes KV traffic",
        ),
        "moe routed ffn": (
            moe_routed,
            f"expert weights scaled by {expert_used}/{expert_count}",
        ),
        "moe shared ffn": (
            moe_shared,
            "weights only; excludes activations",
        ),
        "moe ffn apply": (
            moe_routed + moe_shared,
            f"routed expert weights scaled by {expert_used}/{expert_count} plus shared FFN",
        ),
        "moe route": (
            tensor_sum(
                rows,
                lambda r: "ffn_gate_inp.weight" in r.name
                or "ffn_gate_inp_shexp.weight" in r.name,
            ),
            "router weights only; excludes topk/reduce",
        ),
        "lm head": (
            tensor_sum(rows, lambda r: r.name == "output.weight"),
            "output weight only; excludes argmax/logit traffic",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate decode phase weight bandwidth from qwen-bench phase output."
    )
    parser.add_argument("--phase", required=True, type=Path)
    parser.add_argument("--tensors", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--expert-count", type=int)
    parser.add_argument("--expert-used", type=int)
    parser.add_argument("--peak-gb-s", type=float, default=474.0)
    args = parser.parse_args()

    expert_count = (
        args.expert_count
        or parse_metadata_int(args.metadata, "qwen35moe.expert_count")
        or 256
    )
    expert_used = (
        args.expert_used
        or parse_metadata_int(args.metadata, "qwen35moe.expert_used_count")
        or 8
    )

    tensors = parse_tensors(args.tensors)
    phase_sum, phases = parse_phase(args.phase)
    estimates = category_bytes(tensors, expert_used, expert_count)

    print(f"phase_sum_ms\t{phase_sum:.4f}")
    print(f"expert_used\t{expert_used}")
    print(f"expert_count\t{expert_count}")
    print("phase\tms\tpct\test_weight_gb\test_gb_s\tpct_stream\tnote")
    for phase in phases:
        nbytes, note = estimates.get(phase.base_name, (0, "unestimated"))
        if nbytes > 0 and phase.ms > 0:
            gb = nbytes / 1e9
            gb_s = gb / (phase.ms / 1000.0)
            pct_stream = gb_s / args.peak_gb_s * 100.0
            print(
                f"{phase.raw_name}\t{phase.ms:.4f}\t{phase.pct:.2f}\t"
                f"{gb:.4f}\t{gb_s:.1f}\t{pct_stream:.1f}\t{note}"
            )
        else:
            print(f"{phase.raw_name}\t{phase.ms:.4f}\t{phase.pct:.2f}\t\t\t\t{note}")


if __name__ == "__main__":
    main()
