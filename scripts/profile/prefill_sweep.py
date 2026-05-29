#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
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


def run_fastpath_audit(model: str) -> dict:
    script = Path(__file__).with_name("gguf_fastpath_audit.py")
    if not script.exists():
        raise RuntimeError(f"fast-path audit script missing: {script}")
    proc = subprocess.run(
        [sys.executable, "-B", str(script), "--json", model],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"fast-path audit failed: {msg}")
    payload = json.loads(proc.stdout)
    rows = payload.get("models")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("fast-path audit produced unexpected JSON payload")
    return rows[0]


def print_fastpath_audit(row: dict) -> None:
    print(
        "[prefill-sweep] fastpath audit: "
        f"dense_ffn={row.get('dense_ffn_fast')} "
        f"gdn={row.get('gdn_matrix_fast')} "
        f"attn={row.get('attn_matrix_fast')} "
        f"moe={row.get('moe_grouped_fast')} "
        f"lm={row.get('lm_fast')}",
        flush=True,
    )
    gaps = row.get("gaps")
    if gaps and gaps != "-":
        print(f"[prefill-sweep] fastpath gaps: {gaps}", flush=True)


def build_bench_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.bench_bin),
        "pp",
        "-m",
        args.model,
        "--runs",
        str(args.runs),
        "-o",
        "json",
    ]
    if args.n_prompt is not None:
        cmd.extend(["-p", str(args.n_prompt)])
    elif args.file is not None:
        cmd.extend(["--file", args.file])
    elif args.messages is not None:
        cmd.extend(["--messages", args.messages])
        if args.messages_max is not None:
            cmd.extend(["--messages-max", str(args.messages_max)])
        if args.messages_preserve_thinking:
            cmd.append("--messages-preserve-thinking")
        if args.messages_strip_thinking:
            cmd.append("--messages-strip-thinking")
        if args.messages_no_generation_prompt:
            cmd.append("--messages-no-generation-prompt")
    else:
        raise AssertionError("one prompt source is required")
    if args.prefill_chunk is not None:
        cmd.extend(["--prefill-chunk", str(args.prefill_chunk)])
    if args.with_tail:
        cmd.append("--with-tail")
    if args.no_warmup:
        cmd.append("--no-warmup")
    for extra in args.extra_arg:
        cmd.append(extra)
    return cmd


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
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"variant {variant.label!r} produced non-JSON stdout: {proc.stdout[:400]!r}"
        ) from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
    ):
        raise RuntimeError(
            f"variant {variant.label!r} produced unexpected JSON payload"
        )
    row = payload[0]
    return {
        "label": variant.label,
        "env": variant.env,
        "wall_s_outer": wall_s,
        "thermal_before": therm_before,
        "thermal_after": therm_after,
        "memory_before": mem_before,
        "memory_after": mem_after,
        "bench": row,
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
    print("block\torder\tlabel\tavg_ts\tavg_ms/token\tavg_gpu_ms/token\touter_wall_s")
    for item in results:
        row = item["bench"]
        avg_ts = float(row["avg_ts"])
        avg_ns = float(row["avg_ns"])
        avg_gpu_ns = float(row.get("avg_gpu_ns") or 0)
        n_tokens = int(row["n_tokens"])
        avg_ms_per_tok = (avg_ns / 1e6) / max(1, n_tokens)
        avg_gpu_ms_per_tok = (
            (avg_gpu_ns / 1e6) / max(1, n_tokens) if avg_gpu_ns else 0.0
        )
        print(
            f"{item['block']}\t{item['order']}\t{item['label']}\t"
            f"{avg_ts:.2f}\t{avg_ms_per_tok:.4f}\t"
            f"{avg_gpu_ms_per_tok:.4f}\t{item['wall_s_outer']:.1f}"
        )


def build_summary(
    base_cmd: list[str],
    args: argparse.Namespace,
    run_plan: list[tuple[int, int, Variant]],
    results: list[dict],
    fastpath_audit: dict | None,
) -> dict:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_cmd": base_cmd,
        "fastpath_audit": fastpath_audit,
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
                "bench": r["bench"],
                "wall_s_outer": r["wall_s_outer"],
                "thermal_before": r["thermal_before"],
                "thermal_after": r["thermal_after"],
                "memory_before": r["memory_before"],
                "memory_after": r["memory_after"],
            }
            for r in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cooled sequential qwen-bench prefill sweeps with env variants."
    )
    parser.add_argument(
        "--bench-bin", type=Path, default=Path("target/release/qwen-bench")
    )
    parser.add_argument("--model", required=True)

    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--n-prompt", type=int)
    prompt.add_argument("--file")
    prompt.add_argument("--messages")

    parser.add_argument("--messages-max", type=int)
    parser.add_argument("--messages-preserve-thinking", action="store_true")
    parser.add_argument("--messages-strip-thinking", action="store_true")
    parser.add_argument("--messages-no-generation-prompt", action="store_true")
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--cooldown-seconds", type=float, default=30.0)
    parser.add_argument(
        "--repeat-blocks",
        type=int,
        default=1,
        help="Repeat the full variant list N times to expose run-order drift.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help="Shuffle variant order independently in each block using seed+block.",
    )
    parser.add_argument("--with-tail", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--variant",
        type=parse_variant,
        action="append",
        default=[],
        help="label[:KEY=VALUE,KEY2=VALUE2]",
    )
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument(
        "--no-fastpath-audit",
        action="store_true",
        help="Skip the default static GGUF fast-path coverage audit.",
    )
    parser.add_argument(
        "--require-fastpath-clean",
        action="store_true",
        help="Fail before benchmarking if the static audit reports gaps.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.bench_bin.exists():
        raise SystemExit(f"bench binary missing: {args.bench_bin}")
    if args.repeat_blocks < 1:
        raise SystemExit("--repeat-blocks must be >= 1")
    if not args.variant:
        args.variant = [Variant(label="baseline", env={})]

    fastpath_audit = None
    if not args.no_fastpath_audit:
        try:
            fastpath_audit = run_fastpath_audit(args.model)
        except Exception as exc:
            if args.require_fastpath_clean:
                raise SystemExit(
                    f"fast-path audit failed; use --no-fastpath-audit to skip: {exc}"
                )
            print(
                f"[prefill-sweep] fastpath audit failed; continuing without audit: {exc}",
                flush=True,
            )
        else:
            print_fastpath_audit(fastpath_audit)
            if args.require_fastpath_clean and fastpath_audit.get("gaps") not in (
                None,
                "-",
            ):
                raise SystemExit("fast-path audit has gaps; refusing to benchmark")

    base_cmd = build_bench_cmd(args)
    print(
        f"[prefill-sweep] bench command: {' '.join(shlex.quote(x) for x in base_cmd)}",
        flush=True,
    )
    print(
        f"[prefill-sweep] variants: {', '.join(v.label for v in args.variant)}",
        flush=True,
    )
    print(f"[prefill-sweep] cooldown_seconds: {args.cooldown_seconds}", flush=True)
    print(f"[prefill-sweep] repeat_blocks: {args.repeat_blocks}", flush=True)
    if args.shuffle_seed is not None:
        print(f"[prefill-sweep] shuffle_seed: {args.shuffle_seed}", flush=True)

    run_plan = build_run_plan(args.variant, args.repeat_blocks, args.shuffle_seed)
    results = []
    first = True
    for block_idx, order_idx, variant in run_plan:
        cooldown = 0.0 if first else args.cooldown_seconds
        first = False
        print(
            f"[prefill-sweep] running block={block_idx} order={order_idx} "
            f"{variant.label} env={variant.env}",
            flush=True,
        )
        result = run_variant(base_cmd, variant, cooldown)
        result["block"] = block_idx
        result["order"] = order_idx
        results.append(result)
        if output_path is not None:
            output_path.write_text(
                json.dumps(
                    build_summary(base_cmd, args, run_plan, results, fastpath_audit),
                    indent=2,
                )
                + "\n"
            )
            print(f"[prefill-sweep] checkpointed {args.output}", flush=True)

    summary = build_summary(base_cmd, args, run_plan, results, fastpath_audit)
    print_summary(results)
    if output_path is not None:
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[prefill-sweep] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
