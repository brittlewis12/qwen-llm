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


BENCH_DIR = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH_DIR))

from llama_cpp import (  # noqa: E402
    LOCK_PATH as DEFAULT_LLAMA_CPP_LOCK,
    load_lock,
    lock_summary,
    missing_tool_message,
    resolve_tool,
    validate_probe_row,
)


@dataclass(frozen=True)
class EngineRun:
    engine: str
    env: dict[str, str]


@dataclass(frozen=True)
class PromptInfo:
    source: str
    lcpp_n_prompt: int
    lcpp_prompt_mode: str


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


def strip_think(text: str) -> str:
    trimmed = text.lstrip()
    if not trimmed.startswith("<think>"):
        return text
    rest = trimmed[len("<think>") :]
    _head, sep, tail = rest.partition("</think>")
    if not sep:
        return text
    return tail.strip()


def load_messages_prompt(args: argparse.Namespace) -> str:
    raw = Path(args.messages).read_text()
    value = json.loads(raw)
    if isinstance(value, list):
        messages = value
        meta: dict = {}
    elif isinstance(value, dict):
        if "messages" not in value:
            raise RuntimeError("wrapped messages input must contain `messages`")
        messages = value["messages"]
        meta = {}
        raw_meta = value.get("meta")
        if isinstance(raw_meta, dict):
            meta.update(raw_meta)
        for key, item in value.items():
            if key not in ("messages", "meta"):
                meta[key] = item
    else:
        raise RuntimeError("messages input must be a list or wrapped object")
    if args.messages_max is not None:
        messages = messages[: args.messages_max]
    if not messages:
        raise RuntimeError("messages input contains no messages")
    if args.messages_preserve_thinking:
        preserve_thinking = True
    elif args.messages_strip_thinking:
        preserve_thinking = False
    else:
        preserve_thinking = (
            bool(meta.get("preserve_thinking"))
            or "qwen3.6" in str(meta.get("model", "")).lower()
        )

    out: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise RuntimeError("each message must contain string role/content")
        if role == "assistant" and not preserve_thinking:
            content = strip_think(content)
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if not args.messages_no_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


def prompt_text_for_count(args: argparse.Namespace) -> tuple[str, str]:
    if args.prompt is not None:
        return args.prompt, f"text prompt ({len(args.prompt)} chars)"
    if args.file is not None:
        text = Path(args.file).read_text()
        return text, f"file prompt:{args.file} ({len(text)} chars)"
    if args.messages is not None:
        text = load_messages_prompt(args)
        return text, f"messages:{args.messages} ({len(text)} chars)"
    raise AssertionError("no real prompt source")


def count_lcpp_prompt_tokens(args: argparse.Namespace, text: str) -> int:
    proc = subprocess.run(
        [
            str(args.lcpp_tokenize_bin),
            "-m",
            args.model,
            "--stdin",
            "--ids",
            "--show-count",
            "--no-bos",
            "--log-disable",
        ],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"llama-tokenize failed: {msg}")
    marker = "Total number of tokens:"
    for line in reversed(proc.stdout.splitlines()):
        if marker in line:
            return int(line.rsplit(marker, 1)[1].strip())
    raise RuntimeError("llama-tokenize output did not include token count")


def resolve_prompt_info(args: argparse.Namespace) -> PromptInfo:
    if args.n_prompt is not None:
        return PromptInfo(
            source=f"synthetic:{args.n_prompt}",
            lcpp_n_prompt=args.n_prompt,
            lcpp_prompt_mode="synthetic",
        )
    text, source = prompt_text_for_count(args)
    n_prompt = count_lcpp_prompt_tokens(args, text)
    if n_prompt < 1:
        raise RuntimeError("real prompt tokenized to an empty sequence")
    return PromptInfo(
        source=source,
        lcpp_n_prompt=n_prompt,
        lcpp_prompt_mode="synthetic_length_anchor",
    )


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
    ]
    if args.n_prompt is not None:
        cmd.extend(["-p", str(args.n_prompt)])
    elif args.prompt is not None:
        cmd.extend(["--prompt", args.prompt])
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
    if args.no_warmup:
        cmd.append("--no-warmup")
    for extra in args.qwen_extra_arg:
        cmd.append(extra)
    return cmd


def build_lcpp_cmd(args: argparse.Namespace, n_prompt: int) -> list[str]:
    cmd = [
        str(args.lcpp_bin),
        "-m",
        args.model,
        "-p",
        str(n_prompt),
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
    prompt_info: PromptInfo,
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
        "prompt_source": prompt_info.source,
        "n_prompt": prompt_info.lcpp_n_prompt,
        "lcpp_prompt_mode": prompt_info.lcpp_prompt_mode,
        "runs": args.runs,
        "cooldown_seconds": args.cooldown_seconds,
        "repeat_blocks": args.repeat_blocks,
        "discard_first_block": args.discard_first_block,
        "shuffle_seed": args.shuffle_seed,
        "qwen_cmd": qwen_cmd,
        "lcpp_cmd": lcpp_cmd,
        "llama_cpp_lock": lock_summary(args.llama_cpp_lock_data),
        "llama_cpp_locked": args.llama_cpp_locked,
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
        help="explicit llama-bench path; default resolves the pinned llama.cpp lock",
    )
    parser.add_argument(
        "--lcpp-tokenize-bin",
        type=Path,
        help="explicit llama-tokenize path; default resolves the pinned llama.cpp lock",
    )
    parser.add_argument("--llama-cpp-lock", type=Path, default=DEFAULT_LLAMA_CPP_LOCK)
    parser.add_argument(
        "--allow-unpinned-lcpp",
        action="store_true",
        help="permit a llama.cpp binary whose build_commit/backends do not match the lock",
    )
    parser.add_argument("--model", required=True)

    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--n-prompt", type=int)
    prompt.add_argument("--prompt")
    prompt.add_argument("--file")
    prompt.add_argument("--messages")

    parser.add_argument("--messages-max", type=int)
    parser.add_argument("--messages-preserve-thinking", action="store_true")
    parser.add_argument("--messages-strip-thinking", action="store_true")
    parser.add_argument("--messages-no-generation-prompt", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cooldown-seconds", type=float, default=15.0)
    parser.add_argument("--repeat-blocks", type=int, default=2)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--discard-first-block", action="store_true")
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--flash-attn", type=int, choices=(-1, 0, 1), default=-1)
    parser.add_argument("--qwen-env", type=parse_env_pair, action="append", default=[])
    parser.add_argument("--qwen-extra-arg", action="append", default=[])
    parser.add_argument("--lcpp-extra-arg", action="append", default=[])
    parser.add_argument("--no-fastpath-audit", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.n_prompt is not None and args.n_prompt < 1:
        raise SystemExit("--n-prompt must be >= 1")
    if args.messages_preserve_thinking and args.messages_strip_thinking:
        raise SystemExit("preserve and strip thinking modes are mutually exclusive")
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.repeat_blocks < 1:
        raise SystemExit("--repeat-blocks must be >= 1")
    args.llama_cpp_lock_data = load_lock(args.llama_cpp_lock)
    args.lcpp_bin, args.llama_cpp_locked = resolve_tool(
        "llama-bench",
        explicit=args.lcpp_bin,
        env_var="LLAMA_BENCH",
        lock=args.llama_cpp_lock_data,
    )
    args.lcpp_tokenize_bin, _ = resolve_tool(
        "llama-tokenize",
        explicit=args.lcpp_tokenize_bin,
        env_var="LLAMA_TOKENIZE",
        lock=args.llama_cpp_lock_data,
    )
    if not args.bench_bin.exists():
        raise SystemExit(f"qwen bench binary missing: {args.bench_bin}")
    if not args.lcpp_bin.exists():
        raise SystemExit(missing_tool_message(args.lcpp_bin))
    if args.n_prompt is None and not args.lcpp_tokenize_bin.exists():
        raise SystemExit(missing_tool_message(args.lcpp_tokenize_bin))

    qwen_env = dict(args.qwen_env)
    prompt_info = resolve_prompt_info(args)
    qwen_cmd = build_qwen_cmd(args)
    lcpp_cmd = build_lcpp_cmd(args, prompt_info.lcpp_n_prompt)
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
    print(
        f"[prefill-compare] prompt_source: {prompt_info.source} "
        f"lcpp_n_prompt={prompt_info.lcpp_n_prompt} "
        f"lcpp_prompt_mode={prompt_info.lcpp_prompt_mode}",
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
        if engine_run.engine == "llama.cpp":
            err = validate_probe_row(
                result["bench"],
                args.llama_cpp_lock_data,
                allow_unpinned=args.allow_unpinned_lcpp,
            )
            if err:
                raise RuntimeError(err)
        result["block"] = item["block"]
        result["order"] = item["order"]
        result["discard"] = args.discard_first_block and item["block"] == 0
        results.append(result)
        pairs = build_pairs(results, args.discard_first_block)
        if output_path is not None:
            write_checkpoint(
                output_path,
                build_output(
                    args,
                    prompt_info,
                    qwen_cmd,
                    lcpp_cmd,
                    plan,
                    results,
                    pairs,
                    fastpath_audit,
                ),
            )
            print(f"[prefill-compare] checkpointed {output_path}", flush=True)

    pairs = build_pairs(results, args.discard_first_block)
    print_summary(results, pairs)
    if output_path is not None:
        write_checkpoint(
            output_path,
            build_output(
                args,
                prompt_info,
                qwen_cmd,
                lcpp_cmd,
                plan,
                results,
                pairs,
                fastpath_audit,
            ),
        )
        print(f"[prefill-compare] wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
