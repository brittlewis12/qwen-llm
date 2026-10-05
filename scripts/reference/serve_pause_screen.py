# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Idle re-wire screen for `qwen serve`: prefill time and wired memory of
fresh short requests back to back and after pauses, with idle residency off
(`--idle-residency-secs 0`) and at its default. Starts and stops only the
server process it launched (SIGINT, then SIGTERM after a bound).

Usage:
  uv run scripts/reference/serve_pause_screen.py --qwen target/release/qwen \
    --model <gguf> --out <dir> [--max-context-tokens 4096] [--pauses 10,30] \
    [--extra-arg=--flag ...]

Writes <out>/report.json and one server log per arm. Run detached under tool
supervision (a timeout must not kill a model-scale server mid-command).
"""

import argparse
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")
WORDS = (
    "harbour lantern copper meadow orchard falcon granite willow ember canyon "
    "saffron glacier thimble quarry beacon marble tundra pewter juniper atlas"
).split()


def wired_gib() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"Pages wired down:\s+(\d+)", out)
    page_size = re.search(r"page size of (\d+) bytes", out)
    size = int(page_size.group(1)) if page_size else PAGE
    return int(match.group(1)) * size / 2**30


def prompt(index: int) -> str:
    # Fresh text per request: no prefix or snapshot reuse between requests.
    words = [WORDS[(index * 7 + i * 3) % len(WORDS)] for i in range(18)]
    return f"Request {index}: describe these words in one line: " + " ".join(words)


def post(addr: str, model_id: str, text: str, max_tokens: int) -> float:
    body = json.dumps(
        {
            "model": model_id,
            "input": text,
            "max_output_tokens": max_tokens,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://{addr}/v1/responses",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=600) as response:
        response.read()
    return (time.monotonic() - started) * 1e3


def wait_listening(log: Path, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"server exited early: {process.returncode}; see {log}")
        if "serve: listening on" in log.read_text(errors="replace"):
            return
        time.sleep(0.5)
    raise SystemExit(f"server not listening after {timeout} s; see {log}")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=30)


def run_arm(args, name: str, idle_secs: int | None, port: int) -> dict:
    out = Path(args.out)
    log = out / f"serve-{name}.log"
    addr = f"127.0.0.1:{port}"
    model_id = Path(args.model).stem
    command = [
        args.qwen,
        "serve",
        "-m",
        args.model,
        "--addr",
        addr,
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--max-tokens",
        str(args.max_tokens),
        *args.extra_arg,
    ]
    if idle_secs is not None:
        command += ["--idle-residency-secs", str(idle_secs)]
    env = dict(os.environ, QWEN_METAL_LEASE_WAIT="1", RUST_LOG="info")
    env.pop("MTL_DEBUG_LAYER", None)
    with log.open("w") as sink:
        process = subprocess.Popen(
            command, stdout=sink, stderr=subprocess.STDOUT, env=env
        )
    rows = []
    try:
        wait_listening(log, process, args.ready_timeout)
        schedule = [("warm", 0.0), ("back_to_back", 0.0)] + [
            (f"after_{p}s", float(p)) for p in args.pauses
        ]
        for index, (label, pause) in enumerate(schedule):
            time.sleep(pause)
            before = wired_gib()
            wall = post(
                addr,
                model_id,
                prompt(index + (100 if name == "keep" else 0)),
                args.max_tokens,
            )
            after = wired_gib()
            rows.append(
                {
                    "request": label,
                    "pause_s": pause,
                    "wall_ms": round(wall, 1),
                    "wired_before_gib": round(before, 2),
                    "wired_after_gib": round(after, 2),
                }
            )
        time.sleep(1.0)
    finally:
        stop(process)
    phases = [
        float(m.group(1))
        for m in re.finditer(
            r"serve phases:.*?prefill_ms=([0-9.]+)", log.read_text(errors="replace")
        )
    ]
    for row, prefill in zip(rows, phases):
        row["prefill_ms"] = prefill
    return {
        "arm": name,
        "idle_residency_secs": idle_secs,
        "exit_code": process.returncode,
        "rows": rows,
        "pid": process.pid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--pauses", default="10,30")
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args()
    args.pauses = [int(p) for p in args.pauses.split(",") if p]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    report = {
        "model": Path(args.model).name,
        "qwen": args.qwen,
        "wired_idle_gib": round(wired_gib(), 2),
        "arms": [run_arm(args, "off", 0, 8791), run_arm(args, "keep", None, 8792)],
        "wired_end_gib": round(wired_gib(), 2),
    }
    (Path(args.out) / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
