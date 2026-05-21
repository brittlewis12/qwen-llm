#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import json
import os
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


def print_summary(results: list[dict]) -> None:
    print("label\tavg_ts\tavg_ms/token\tavg_gpu_ms/token\touter_wall_s")
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
            f"{item['label']}\t{avg_ts:.2f}\t{avg_ms_per_tok:.4f}\t{avg_gpu_ms_per_tok:.4f}\t{item['wall_s_outer']:.1f}"
        )


def build_summary(base_cmd: list[str], results: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_cmd": base_cmd,
        "variants": [
            {
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
    parser.add_argument("--output")
    args = parser.parse_args()

    if not args.bench_bin.exists():
        raise SystemExit(f"bench binary missing: {args.bench_bin}")
    if not args.variant:
        args.variant = [Variant(label="baseline", env={})]

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

    results = []
    first = True
    for variant in args.variant:
        cooldown = 0.0 if first else args.cooldown_seconds
        first = False
        print(f"[prefill-sweep] running {variant.label} env={variant.env}", flush=True)
        results.append(run_variant(base_cmd, variant, cooldown))
        if args.output:
            Path(args.output).write_text(
                json.dumps(build_summary(base_cmd, results), indent=2) + "\n"
            )
            print(f"[prefill-sweep] checkpointed {args.output}", flush=True)

    summary = build_summary(base_cmd, results)
    print_summary(results)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[prefill-sweep] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
