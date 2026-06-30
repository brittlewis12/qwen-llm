#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    "IQ2_S": (256, 82),
    "IQ3_XXS": (256, 98),
    "IQ3_S": (256, 110),
    "IQ4_NL": (32, 18),
    "IQ4_XS": (256, 136),
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


def parse_metadata_int_any(path: Path | None, suffix: str) -> int | None:
    for prefix in ("qwen35moe", "qwen3", "qwen2"):
        value = parse_metadata_int(path, f"{prefix}.{suffix}")
        if value is not None:
            return value
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


def count_from_phase_name(raw_name: str) -> int | None:
    match = re.search(r"\(x(\d+)\)$", raw_name)
    if not match:
        return None
    return int(match.group(1))


def attn_v4_nwg(ctx_len: int, group: int) -> int:
    if group == 8 and ctx_len >= 16384:
        return 256
    if group in {8, 16} and ctx_len >= 256:
        return 64
    if group in {4, 6} and ctx_len >= 4096:
        return 64
    if ctx_len < 256:
        return 16
    return 32


def attn_v4_tile_c(ctx_len: int, group: int) -> int:
    if group == 16 and ctx_len >= 32768:
        return 128
    if group in {8, 16} and ctx_len >= 256:
        return 64
    return 32


def attn_v4_group_tile(ctx_len: int, group: int) -> int:
    if ctx_len < 256:
        return group
    if group == 8:
        if ctx_len >= 16384:
            return 4
        return 2
    if group == 16:
        return 4
    return group


def print_attention_kv_estimate(
    metadata: Path | None,
    phases: list[PhaseRow],
    ctx_len: int | None,
    kv_bytes_per_elem: int,
    group_tile_override: int | None,
    nwg_override: int | None,
    tile_c_override: int | None,
) -> None:
    if metadata is None or ctx_len is None:
        return

    n_q_heads = parse_metadata_int_any(metadata, "attention.head_count")
    n_kv_heads = parse_metadata_int_any(metadata, "attention.head_count_kv")
    key_len = parse_metadata_int_any(metadata, "attention.key_length")
    value_len = parse_metadata_int_any(metadata, "attention.value_length") or key_len
    block_count = parse_metadata_int_any(metadata, "block_count")
    interval = parse_metadata_int_any(metadata, "full_attention_interval")
    if not all((n_q_heads, n_kv_heads, key_len, value_len)):
        return

    attn_phase = next(
        (p for p in phases if p.base_name in {"attn mixer", "attn layers"}),
        None,
    )
    if attn_phase is None or attn_phase.ms <= 0.0:
        return

    attn_layers = count_from_phase_name(attn_phase.raw_name)
    if attn_layers is None and block_count and interval:
        attn_layers = max(1, block_count // interval)
    if attn_layers is None:
        return

    group = n_q_heads // max(1, n_kv_heads)
    group_tile = group_tile_override or attn_v4_group_tile(ctx_len, group)
    subgroups = max(1, group // max(1, group_tile))
    nwg = nwg_override or attn_v4_nwg(ctx_len, group)
    tile_c = tile_c_override or attn_v4_tile_c(ctx_len, group)

    logical_bytes = (
        attn_layers * n_kv_heads * ctx_len * (key_len + value_len) * kv_bytes_per_elem
    )
    subgroup_bytes = logical_bytes * subgroups
    # Partial traffic is small versus long-context KV reads, but keeping it
    # visible prevents the estimate from pretending reduce is free.
    partial_bytes_per_layer = (
        n_kv_heads * nwg * group * value_len * 4 + n_kv_heads * nwg * group * 2 * 4
    )
    partial_bytes = partial_bytes_per_layer * attn_layers
    attn_s = attn_phase.ms / 1000.0

    print(f"attention_kv_ctx\t{ctx_len}")
    print(f"attention_kv_layers\t{attn_layers}")
    print(f"attention_kv_group\t{group}")
    print(f"attention_kv_group_tile\t{group_tile}")
    print(f"attention_kv_subgroups\t{subgroups}")
    print(f"attention_kv_nwg\t{nwg}")
    print(f"attention_kv_tile_c\t{tile_c}")
    print(f"attention_kv_logical_gb\t{logical_bytes / 1e9:.4f}")
    print(f"attention_kv_subgroup_gb\t{subgroup_bytes / 1e9:.4f}")
    print(f"attention_kv_partial_gb\t{partial_bytes / 1e9:.4f}")
    print(f"attention_kv_subgroup_gb_s\t{subgroup_bytes / 1e9 / attn_s:.1f}")
    print(
        f"attention_kv_subgroup_plus_partial_gb_s\t"
        f"{(subgroup_bytes + partial_bytes) / 1e9 / attn_s:.1f}"
    )


def tensor_sum(rows: list[TensorRow], pred, scale: float = 1.0) -> int:
    return int(sum(row.nbytes * scale for row in rows if pred(row)))


def category_bytes(
    rows: list[TensorRow],
    expert_used: int,
    expert_count: int,
) -> dict[str, tuple[int, str]]:
    expert_scale = expert_used / expert_count
    moe_routed_gate_up = tensor_sum(
        rows,
        lambda r: any(
            part in r.name for part in ("ffn_gate_exps.weight", "ffn_up_exps.weight")
        ),
        expert_scale,
    )
    moe_routed_down = tensor_sum(
        rows,
        lambda r: "ffn_down_exps.weight" in r.name,
        expert_scale,
    )
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
    moe_shared_gate_up = tensor_sum(
        rows,
        lambda r: any(
            part in r.name for part in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight")
        ),
    )
    moe_shared_down = tensor_sum(
        rows,
        lambda r: "ffn_down_shexp.weight" in r.name,
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
    gdn_qkv = tensor_sum(rows, lambda r: "attn_qkv.weight" in r.name)
    gdn_z = tensor_sum(rows, lambda r: "attn_gate.weight" in r.name)
    gdn_beta = tensor_sum(rows, lambda r: "ssm_beta.weight" in r.name)
    gdn_alpha = tensor_sum(rows, lambda r: "ssm_alpha.weight" in r.name)
    gdn_front = gdn_qkv + gdn_z + gdn_beta + gdn_alpha

    return {
        "gdn front proj": (
            gdn_front,
            "weights only; excludes activations",
        ),
        "gdn qkv proj": (gdn_qkv, "QKV projection weights only"),
        "gdn z proj": (gdn_z, "Z projection weights only"),
        "gdn beta proj": (gdn_beta, "beta projection weights only"),
        "gdn alpha proj": (gdn_alpha, "alpha projection weights only"),
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
        "moe ffn routed gate/up": (
            moe_routed_gate_up,
            f"routed gate+up expert weights scaled by {expert_used}/{expert_count}",
        ),
        "moe ffn routed down": (
            moe_routed_down,
            f"routed down expert weights scaled by {expert_used}/{expert_count}",
        ),
        "moe ffn shared gate/up": (
            moe_shared_gate_up,
            "shared gate+up weights only; excludes activations",
        ),
        "moe ffn shared down": (
            moe_shared_down,
            "shared down weights only; excludes activations",
        ),
        "moe ffn gate/up wave": (
            moe_routed_gate_up + moe_shared_gate_up,
            f"routed gate+up scaled by {expert_used}/{expert_count} plus shared gate+up",
        ),
        "moe ffn down wave": (
            moe_routed_down + moe_shared_down,
            f"routed down scaled by {expert_used}/{expert_count} plus shared down",
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


def active_decode_weight_bytes(estimates: dict[str, tuple[int, str]]) -> int:
    return sum(
        estimates[name][0]
        for name in (
            "gdn front proj",
            "gdn out_proj",
            "attn mixer",
            "moe ffn apply",
            "moe route",
            "lm head",
        )
        if name in estimates
    )


def _first_bench_record(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        if not data:
            raise SystemExit("empty bench JSON list")
        return _first_bench_record(data[0])
    if not isinstance(data, dict):
        raise SystemExit("bench JSON root must be an object or list")
    if isinstance(data.get("bench"), dict):
        return data["bench"]
    if isinstance(data.get("variants"), list) and data["variants"]:
        return _first_bench_record(data["variants"][0])
    return data


def parse_bench_json(path: Path) -> dict[str, Any]:
    return _first_bench_record(json.loads(path.read_text()))


def print_decode_roofline_summary(
    bench_json: Path | None,
    decode_tps: float | None,
    avg_wall_ms_token: float | None,
    avg_gpu_ms_token: float | None,
    estimates: dict[str, tuple[int, str]],
    peak_gb_s: float,
) -> None:
    bench: dict[str, Any] = {}
    if bench_json is not None:
        bench = parse_bench_json(bench_json)
    avg_ts = decode_tps if decode_tps is not None else bench.get("avg_ts")
    if avg_ts is None:
        return
    active_bytes = active_decode_weight_bytes(estimates)
    active_gb = active_bytes / 1e9
    active_gb_s = active_gb * float(avg_ts)
    print(f"decode_tps\t{float(avg_ts):.4f}")
    print(f"decode_active_weight_gb_per_token\t{active_gb:.4f}")
    print(f"decode_active_weight_gb_s\t{active_gb_s:.1f}")
    print(f"decode_active_weight_pct_stream\t{active_gb_s / peak_gb_s * 100.0:.1f}")

    n_tokens = bench.get("n_tokens") or bench.get("n_gen")
    avg_ns = bench.get("avg_ns")
    avg_gpu_ns = bench.get("avg_gpu_ns")
    if avg_wall_ms_token is not None:
        print(f"decode_avg_wall_ms_token\t{avg_wall_ms_token:.4f}")
    if avg_gpu_ms_token is not None:
        print(f"decode_avg_gpu_ms_token\t{avg_gpu_ms_token:.4f}")
    if avg_wall_ms_token is None and n_tokens and avg_ns is not None:
        print(f"decode_avg_wall_ms_token\t{float(avg_ns) / float(n_tokens) / 1e6:.4f}")
    if avg_gpu_ms_token is None and n_tokens and avg_gpu_ns is not None:
        print(
            f"decode_avg_gpu_ms_token\t{float(avg_gpu_ns) / float(n_tokens) / 1e6:.4f}"
        )
    for key in (
        "kernel_trace_command_buffers_per_token",
        "kernel_trace_encoders_per_token",
        "kernel_trace_concurrent_encoders_per_token",
        "kernel_trace_dispatches_per_token",
    ):
        value = bench.get(key)
        if value is not None:
            print(f"decode_{key}\t{value}")


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
    parser.add_argument(
        "--ctx", type=int, help="decode context length for KV estimates"
    )
    parser.add_argument("--kv-bytes-per-elem", type=int, default=2)
    parser.add_argument("--group-tile", type=int, help="override attn_v4 group tile")
    parser.add_argument("--nwg", type=int, help="override attn_v4 split-K count")
    parser.add_argument("--tile-c", type=int, help="override attn_v4 KV tile length")
    parser.add_argument(
        "--bench-json",
        type=Path,
        help="optional qwen-bench JSON row for decode t/s and active-weight roofline",
    )
    parser.add_argument(
        "--decode-tps",
        type=float,
        help="manual decode tokens/sec, useful for ctx-sweep text rows",
    )
    parser.add_argument("--avg-wall-ms-token", type=float)
    parser.add_argument("--avg-gpu-ms-token", type=float)
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
    print_decode_roofline_summary(
        args.bench_json,
        args.decode_tps,
        args.avg_wall_ms_token,
        args.avg_gpu_ms_token,
        estimates,
        args.peak_gb_s,
    )
    print_attention_kv_estimate(
        args.metadata,
        phases,
        args.ctx,
        args.kv_bytes_per_elem,
        args.group_tile,
        args.nwg,
        args.tile_c,
    )
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
