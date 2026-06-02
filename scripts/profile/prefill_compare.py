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
class EngineRun:
    engine: str
    env: dict[str, str]


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


def parse_env_pair(raw: str) -> tuple[str, str]:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(
            f"invalid env pair {raw!r}; expected KEY=VALUE"
        )
    return key, value


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


def build_qwen_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.bench_bin),
        "pp",
        "-m",
        args.model,
        "--runs",
        str(args.runs),
        "-o",
        "json",
        "-p",
        str(args.n_prompt),
    ]
    if args.prefill_chunk is not None:
        cmd.extend(["--prefill-chunk", str(args.prefill_chunk)])
    if args.no_warmup:
        cmd.append("--no-warmup")
    for extra in args.qwen_extra_arg:
        cmd.append(extra)
    return cmd


def build_lcpp_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.lcpp_bin),
        "-m",
        args.model,
        "-p",
        str(args.n_prompt),
        "-n",
        "0",
        "-r",
        str(args.runs),
        "-o",
        "json",
        "-fa",
        str(args.flash_attn),
    ]
    if args.no_warmup:
        cmd.append("--no-warmup")
    for extra in args.lcpp_extra_arg:
        cmd.append(extra)
    return cmd


def run_command(
    engine_run: EngineRun,
    command: list[str],
    cooldown_seconds: float,
) -> dict:
    if cooldown_seconds > 0:
        time.sleep(cooldown_seconds)
    env = os.environ.copy()
    env.update(engine_run.env)
    thermal_before = sample_thermal()
    memory_before = sample_memory_pressure()
    wall_start = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    outer_wall_s = time.perf_counter() - wall_start
    thermal_after = sample_thermal()
    memory_after = sample_memory_pressure()
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"{engine_run.engine} failed: {msg}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{engine_run.engine} produced non-JSON stdout: {proc.stdout[:400]!r}"
        ) from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
    ):
        raise RuntimeError(f"{engine_run.engine} produced unexpected JSON payload")
    return {
        "engine": engine_run.engine,
        "env": engine_run.env,
        "bench": payload[0],
        "outer_wall_s": outer_wall_s,
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "stderr": proc.stderr,
    }


def build_run_plan(args: argparse.Namespace, qwen_env: dict[str, str]) -> list[dict]:
    engines = [EngineRun("qwen", qwen_env), EngineRun("llama.cpp", {})]
    plan = []
    for block in range(args.repeat_blocks):
        block_engines = list(engines)
        if args.shuffle_seed is not None:
            random.Random(args.shuffle_seed + block).shuffle(block_engines)
        elif block % 2 == 1:
            block_engines.reverse()
        for order, engine_run in enumerate(block_engines):
            plan.append({"block": block, "order": order, "engine_run": engine_run})
    return plan


def build_pairs(results: list[dict], discard_first_block: bool) -> list[dict]:
    by_block: dict[int, dict[str, dict]] = {}
    for item in results:
        by_block.setdefault(item["block"], {})[item["engine"]] = item
    pairs = []
    for block in sorted(by_block):
        if discard_first_block and block == 0:
            continue
        row = by_block[block]
        qwen = row.get("qwen")
        lcpp = row.get("llama.cpp")
        if not qwen or not lcpp:
            continue
        qwen_ts = float(qwen["bench"]["avg_ts"])
        lcpp_ts = float(lcpp["bench"]["avg_ts"])
        pairs.append(
            {
                "block": block,
                "qwen_order": qwen["order"],
                "lcpp_order": lcpp["order"],
                "qwen_ts": qwen_ts,
                "lcpp_ts": lcpp_ts,
                "qwen_over_lcpp": qwen_ts / lcpp_ts if lcpp_ts else None,
                "delta_ts": qwen_ts - lcpp_ts,
            }
        )
    return pairs


def print_summary(results: list[dict], pairs: list[dict]) -> None:
    print("block\torder\tdiscard\tengine\tavg_ts\tavg_ms/token\touter_wall_s")
    for item in results:
        bench = item["bench"]
        avg_ts = float(bench["avg_ts"])
        avg_ns = float(bench["avg_ns"])
        n_tokens = int(bench.get("n_tokens") or bench.get("n_prompt") or 1)
        avg_ms_per_token = (avg_ns / 1e6) / max(1, n_tokens)
        print(
            f"{item['block']}\t{item['order']}\t{int(item.get('discard', False))}\t"
            f"{item['engine']}\t"
            f"{avg_ts:.2f}\t{avg_ms_per_token:.4f}\t{item['outer_wall_s']:.1f}"
        )
    if pairs:
        print("\nblock\tqwen_order\tlcpp_order\tqwen\tlcpp\tqwen/lcpp\tdelta")
        for pair in pairs:
            print(
                f"{pair['block']}\t{pair['qwen_order']}\t{pair['lcpp_order']}\t"
                f"{pair['qwen_ts']:.2f}\t{pair['lcpp_ts']:.2f}\t"
                f"{pair['qwen_over_lcpp']:.3f}\t{pair['delta_ts']:.2f}"
            )


def build_output(
    args: argparse.Namespace,
    qwen_cmd: list[str],
    lcpp_cmd: list[str],
    plan: list[dict],
    results: list[dict],
    pairs: list[dict],
    fastpath_audit: dict | None,
) -> dict:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "n_prompt": args.n_prompt,
        "runs": args.runs,
        "cooldown_seconds": args.cooldown_seconds,
        "repeat_blocks": args.repeat_blocks,
        "discard_first_block": args.discard_first_block,
        "shuffle_seed": args.shuffle_seed,
        "qwen_cmd": qwen_cmd,
        "lcpp_cmd": lcpp_cmd,
        "fastpath_audit": fastpath_audit,
        "run_plan": [
            {
                "block": item["block"],
                "order": item["order"],
                "engine": item["engine_run"].engine,
                "env": item["engine_run"].env,
            }
            for item in plan
        ],
        "results": [
            {
                "block": item["block"],
                "order": item["order"],
                "discard": item.get("discard", False),
                "engine": item["engine"],
                "env": item["env"],
                "bench": item["bench"],
                "outer_wall_s": item["outer_wall_s"],
                "thermal_before": item["thermal_before"],
                "thermal_after": item["thermal_after"],
                "memory_before": item["memory_before"],
                "memory_after": item["memory_after"],
                "stderr": item["stderr"],
            }
            for item in results
        ],
        "pairs": pairs,
    }


def write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run paired qwen-bench and llama-bench prefill comparisons."
    )
    parser.add_argument(
        "--bench-bin", type=Path, default=Path("target/release/qwen-bench")
    )
    parser.add_argument(
        "--lcpp-bin",
        type=Path,
        default=Path("/Users/tito/code/llama.cpp/build/bin/llama-bench"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-prompt", type=int, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cooldown-seconds", type=float, default=15.0)
    parser.add_argument("--repeat-blocks", type=int, default=2)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--discard-first-block", action="store_true")
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--flash-attn", type=int, choices=(0, 1), default=0)
    parser.add_argument("--qwen-env", type=parse_env_pair, action="append", default=[])
    parser.add_argument("--qwen-extra-arg", action="append", default=[])
    parser.add_argument("--lcpp-extra-arg", action="append", default=[])
    parser.add_argument("--no-fastpath-audit", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.n_prompt < 1:
        raise SystemExit("--n-prompt must be >= 1")
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.repeat_blocks < 1:
        raise SystemExit("--repeat-blocks must be >= 1")
    if not args.bench_bin.exists():
        raise SystemExit(f"qwen bench binary missing: {args.bench_bin}")
    if not args.lcpp_bin.exists():
        raise SystemExit(f"llama.cpp bench binary missing: {args.lcpp_bin}")

    qwen_env = dict(args.qwen_env)
    qwen_cmd = build_qwen_cmd(args)
    lcpp_cmd = build_lcpp_cmd(args)
    output_path = Path(args.output) if args.output else None

    fastpath_audit = None
    if not args.no_fastpath_audit:
        try:
            fastpath_audit = run_fastpath_audit(args.model)
        except Exception as exc:
            print(f"[prefill-compare] fastpath audit failed: {exc}", flush=True)
        else:
            print(
                "[prefill-compare] fastpath audit: "
                f"dense_ffn={fastpath_audit.get('dense_ffn_fast')} "
                f"gdn={fastpath_audit.get('gdn_matrix_fast')} "
                f"attn={fastpath_audit.get('attn_matrix_fast')} "
                f"moe={fastpath_audit.get('moe_grouped_fast')} "
                f"lm={fastpath_audit.get('lm_fast')}",
                flush=True,
            )

    print(
        f"[prefill-compare] qwen: {' '.join(shlex.quote(x) for x in qwen_cmd)}",
        flush=True,
    )
    print(
        f"[prefill-compare] lcpp: {' '.join(shlex.quote(x) for x in lcpp_cmd)}",
        flush=True,
    )
    print(f"[prefill-compare] cooldown_seconds: {args.cooldown_seconds}", flush=True)
    print(f"[prefill-compare] repeat_blocks: {args.repeat_blocks}", flush=True)
    if args.shuffle_seed is not None:
        print(f"[prefill-compare] shuffle_seed: {args.shuffle_seed}", flush=True)
    if args.discard_first_block:
        print("[prefill-compare] first paired block will be marked discard", flush=True)

    plan = build_run_plan(args, qwen_env)
    results = []
    first = True
    for item in plan:
        engine_run = item["engine_run"]
        command = qwen_cmd if engine_run.engine == "qwen" else lcpp_cmd
        cooldown = 0.0 if first else args.cooldown_seconds
        first = False
        print(
            f"[prefill-compare] running block={item['block']} order={item['order']} "
            f"engine={engine_run.engine} env={engine_run.env}",
            flush=True,
        )
        result = run_command(engine_run, command, cooldown)
        result["block"] = item["block"]
        result["order"] = item["order"]
        result["discard"] = args.discard_first_block and item["block"] == 0
        results.append(result)
        pairs = build_pairs(results, args.discard_first_block)
        if output_path is not None:
            write_checkpoint(
                output_path,
                build_output(
                    args, qwen_cmd, lcpp_cmd, plan, results, pairs, fastpath_audit
                ),
            )
            print(f"[prefill-compare] checkpointed {output_path}", flush=True)

    pairs = build_pairs(results, args.discard_first_block)
    print_summary(results, pairs)
    if output_path is not None:
        write_checkpoint(
            output_path,
            build_output(
                args, qwen_cmd, lcpp_cmd, plan, results, pairs, fastpath_audit
            ),
        )
        print(f"[prefill-compare] wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
