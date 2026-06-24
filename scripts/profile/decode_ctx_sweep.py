#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    label: str
    env: dict[str, str]


def parse_variant(raw: str) -> Variant:
    label, sep, env_blob = raw.partition(":")
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"invalid empty variant label: {raw!r}")
    env: dict[str, str] = {}
    if sep:
        for pair in env_blob.split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, eq, value = pair.partition("=")
            if not eq or not key:
                raise argparse.ArgumentTypeError(
                    f"invalid variant env pair {pair!r} in {raw!r}; expected KEY=VALUE"
                )
            env[key] = value
    return Variant(label=label, env=env)


def capture_text(command: list[str]) -> str:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if proc.returncode != 0:
        msg = err or out or f"exit {proc.returncode}"
        raise RuntimeError(f"{' '.join(command)} failed: {msg}")
    return out if out else err


def sample_thermal() -> str:
    return capture_text(["pmset", "-g", "therm"])


def sample_memory_pressure() -> str:
    return capture_text(["memory_pressure", "-Q"])


def build_bench_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.bench_bin),
        "ctx-sweep",
        "-m",
        args.model,
        "--checkpoints",
        args.checkpoints,
        "--window",
        str(args.window),
    ]
    if args.fresh_per_checkpoint:
        cmd.append("--fresh-per-checkpoint")
    for extra in args.extra_arg:
        cmd.append(extra)
    return cmd


def parse_ctx_rows(stdout: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    pattern = re.compile(
        r"^\[ctx-sweep\]\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
    )
    for line in stdout.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        context, total_ms, gpu_ms, cpu_enc_ms, tps = match.groups()
        rows.append(
            {
                "context": int(context),
                "total_ms": float(total_ms),
                "gpu_ms": float(gpu_ms),
                "cpu_enc_ms": float(cpu_enc_ms),
                "tps": float(tps),
            }
        )
    if not rows:
        raise RuntimeError(f"ctx-sweep output had no parseable rows:\n{stdout}")
    return rows


def run_variant(base_cmd: list[str], variant: Variant, cooldown_seconds: float) -> dict:
    if cooldown_seconds > 0:
        time.sleep(cooldown_seconds)
    env = os.environ.copy()
    env.update(variant.env)
    therm_before = sample_thermal()
    mem_before = sample_memory_pressure()
    wall_start = time.perf_counter()
    proc = subprocess.run(
        base_cmd, capture_output=True, text=True, env=env, check=False
    )
    wall_s = time.perf_counter() - wall_start
    therm_after = sample_thermal()
    mem_after = sample_memory_pressure()
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"variant {variant.label!r} failed: {msg}")
    return {
        "label": variant.label,
        "env": variant.env,
        "rows": parse_ctx_rows(proc.stdout),
        "wall_s_outer": wall_s,
        "thermal_before": therm_before,
        "thermal_after": therm_after,
        "memory_before": mem_before,
        "memory_after": mem_after,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def build_run_plan(
    variants: list[Variant], repeat_blocks: int, shuffle_seed: int | None
) -> list[tuple[int, int, Variant]]:
    plan: list[tuple[int, int, Variant]] = []
    for block_idx in range(repeat_blocks):
        block_variants = list(variants)
        if shuffle_seed is not None:
            random.Random(shuffle_seed + block_idx).shuffle(block_variants)
        for order_idx, variant in enumerate(block_variants):
            plan.append((block_idx, order_idx, variant))
    return plan


def print_summary(results: list[dict]) -> None:
    print(
        "block\torder\tlabel\tcontext\ttotal_ms\tgpu_ms\tcpu_enc_ms\ttps\touter_wall_s"
    )
    for item in results:
        for row in item["rows"]:
            print(
                f"{item['block']}\t{item['order']}\t{item['label']}\t"
                f"{row['context']}\t{row['total_ms']:.2f}\t{row['gpu_ms']:.2f}\t"
                f"{row['cpu_enc_ms']:.2f}\t{row['tps']:.1f}\t{item['wall_s_outer']:.1f}"
            )


def build_summary(
    base_cmd: list[str],
    args: argparse.Namespace,
    run_plan: list[tuple[int, int, Variant]],
    results: list[dict],
) -> dict:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_cmd": base_cmd,
        "repeat_blocks": args.repeat_blocks,
        "shuffle_seed": args.shuffle_seed,
        "run_plan": [
            {
                "block": block,
                "order": order,
                "label": variant.label,
                "env": variant.env,
            }
            for block, order, variant in run_plan
        ],
        "variants": [
            {
                "block": r["block"],
                "order": r["order"],
                "label": r["label"],
                "env": r["env"],
                "rows": r["rows"],
                "wall_s_outer": r["wall_s_outer"],
                "thermal_before": r["thermal_before"],
                "thermal_after": r["thermal_after"],
                "memory_before": r["memory_before"],
                "memory_after": r["memory_after"],
            }
            for r in results
        ],
    }


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run sequential qwen-bench ctx-sweep packets with env variants."
    )
    parser.add_argument(
        "--bench-bin", type=Path, default=Path("target/release/qwen-bench")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--fresh-per-checkpoint", action="store_true")
    parser.add_argument("--cooldown-seconds", type=float, default=30.0)
    parser.add_argument("--repeat-blocks", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument(
        "--variant",
        type=parse_variant,
        action="append",
        default=[],
        help="label[:KEY=VALUE,KEY2=VALUE2]",
    )
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--output")
    args = parser.parse_args()

    if not args.bench_bin.exists():
        raise SystemExit(f"bench binary missing: {args.bench_bin}")
    if args.repeat_blocks < 1:
        raise SystemExit("--repeat-blocks must be >= 1")
    if args.window < 1:
        raise SystemExit("--window must be >= 1")
    if not args.variant:
        args.variant = [Variant(label="baseline", env={})]

    output_path = Path(args.output) if args.output else None
    base_cmd = build_bench_cmd(args)
    print(
        f"[decode-ctx-sweep] bench command: {' '.join(shlex.quote(x) for x in base_cmd)}",
        flush=True,
    )
    print(
        f"[decode-ctx-sweep] variants: {', '.join(v.label for v in args.variant)}",
        flush=True,
    )
    print(f"[decode-ctx-sweep] cooldown_seconds: {args.cooldown_seconds}", flush=True)
    print(f"[decode-ctx-sweep] repeat_blocks: {args.repeat_blocks}", flush=True)
    if args.shuffle_seed is not None:
        print(f"[decode-ctx-sweep] shuffle_seed: {args.shuffle_seed}", flush=True)

    run_plan = build_run_plan(args.variant, args.repeat_blocks, args.shuffle_seed)
    results: list[dict] = []
    first = True
    for block_idx, order_idx, variant in run_plan:
        cooldown = 0.0 if first else args.cooldown_seconds
        first = False
        print(
            f"[decode-ctx-sweep] running block={block_idx} order={order_idx} "
            f"{variant.label} env={variant.env}",
            flush=True,
        )
        result = run_variant(base_cmd, variant, cooldown)
        result["block"] = block_idx
        result["order"] = order_idx
        results.append(result)
        if output_path is not None:
            write_summary(output_path, build_summary(base_cmd, args, run_plan, results))
            print(f"[decode-ctx-sweep] checkpointed {args.output}", flush=True)

    summary = build_summary(base_cmd, args, run_plan, results)
    print_summary(results)
    if output_path is not None:
        write_summary(output_path, summary)
        print(f"[decode-ctx-sweep] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
