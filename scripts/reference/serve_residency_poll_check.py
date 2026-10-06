# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Live idle-residency lifecycle check for `qwen serve`: non-inference traffic
(`GET /v1/models` polling) must neither open nor renew the keep-alive
window; a request that runs the model must. Uses a short window and samples
wired memory (`vm_stat`) through phases:

  1. after startup (GLM's warm-up opens the window; families without a
     warm-up start closed), then past the window while polling models;
  2. one real request, then past the window while polling models.

Starts and stops only the server it launched. Run detached.

  uv run scripts/reference/serve_residency_poll_check.py --qwen target/release/qwen \\
    --model <gguf> --out <dir> [--window 10] [--max-context-tokens 4096]
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


def wired_gib():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    pages = int(re.search(r"Pages wired down:\s+(\d+)", out).group(1))
    size = int(re.search(r"page size of (\d+) bytes", out).group(1))
    return round(pages * size / 2**30, 2)


def get_models(addr):
    with urllib.request.urlopen(f"http://{addr}/v1/models", timeout=30) as response:
        response.read()


def post(addr, model_id):
    body = json.dumps(
        {
            "model": model_id,
            "input": "Say hi in three words.",
            "max_output_tokens": 48,
            "reasoning": {"effort": "low"},
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://{addr}/v1/responses",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())["status"]


def poll_phase(addr, seconds, label, samples):
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        get_models(addr)
        samples.append(
            {
                "phase": label,
                "t": round(time.monotonic() - start, 1),
                "wired_gib": wired_gib(),
            }
        )
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "serve.log"
    addr = "127.0.0.1:8794"
    env = dict(os.environ, QWEN_METAL_LEASE_WAIT="1", RUST_LOG="info")
    env.pop("MTL_DEBUG_LAYER", None)
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
        "256",
        "--idle-residency-secs",
        str(args.window),
    ]
    samples = [{"phase": "before_start", "t": 0, "wired_gib": wired_gib()}]
    with log.open("w") as sink:
        process = subprocess.Popen(
            command, stdout=sink, stderr=subprocess.STDOUT, env=env
        )
    status = None
    try:
        deadline = time.monotonic() + 1800
        while "serve: listening on" not in log.read_text(errors="replace"):
            if process.poll() is not None or time.monotonic() > deadline:
                raise SystemExit(f"server not listening; see {log}")
            time.sleep(1)
        span = args.window + 12
        poll_phase(addr, span, "startup_then_polling", samples)
        status = post(addr, Path(args.model).stem)
        samples.append({"phase": "after_request", "t": 0, "wired_gib": wired_gib()})
        poll_phase(addr, span, "request_then_polling", samples)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=60)
    samples.append({"phase": "after_exit", "t": 0, "wired_gib": wired_gib()})
    report = {
        "model": Path(args.model).name,
        "window_s": args.window,
        "request_status": status,
        "samples": samples,
        "exit_code": process.returncode,
        "residency_lines": re.findall(
            r"serve idle residency: .*", log.read_text(errors="replace")
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
