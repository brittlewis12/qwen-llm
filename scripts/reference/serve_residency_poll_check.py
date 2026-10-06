# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Live idle-residency lifecycle check for `qwen serve`: only a request that
submitted GPU compute work opens or renews the keep-alive window.
Non-inference traffic (`GET /v1/models` polling) and a request refused
before submission must not; a completed request and one aborted by the
client after submission must. The verdicts come from the server's own
per-finish lines (`RUST_LOG=info,qwen_diag=debug`), with wired memory
(`vm_stat`, host-wide and so noisy) sampled alongside. Phases, each
followed by polling past a short window:

  1. startup (GLM's warm-up opens the window; families without a warm-up
     start closed);
  2. a prompt longer than the context (refused before any GPU work);
  3. one real request;
  4. a streamed request the client closes after its first output event.

Starts and stops only the server it launched. Run detached.

  uv run scripts/reference/serve_residency_poll_check.py --qwen target/release/qwen \\
    --model <gguf> --out <dir> [--window 10] [--max-context-tokens 4096]
"""

import argparse
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
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


def post_refused(addr, model_id, max_context_tokens):
    """A prompt longer than the context: refused before any GPU work."""
    body = json.dumps(
        {
            "model": model_id,
            "input": "word " * (max_context_tokens + 512),
            "max_output_tokens": 16,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://{addr}/v1/responses",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def post_and_abort(addr, model_id):
    """Stream a request and close the connection after the first output
    event: aborted after submission."""
    host, port = addr.split(":")
    body = json.dumps(
        {
            "model": model_id,
            "input": "Count from one to two hundred in words.",
            "max_output_tokens": 256,
            "reasoning": {"effort": "low"},
            "stream": True,
        }
    ).encode()
    connection = http.client.HTTPConnection(host, int(port), timeout=600)
    connection.request(
        "POST",
        "/v1/responses",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    seen = None
    while True:
        line = response.fp.readline()
        if not line:
            break
        if line.startswith(b"event: ") and b"delta" in line:
            seen = line.decode().strip()
            break
    connection.sock.shutdown(socket.SHUT_RDWR)
    connection.close()
    return seen


FINISHED = re.compile(
    r"serve idle residency: family=\S+ finished compute_encoders=(\S+) window=(\S+)"
)


def finishes(text):
    return [
        {"compute_encoders": encoders, "window": window}
        for encoders, window in FINISHED.findall(text)
    ]


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
    env = dict(os.environ, QWEN_METAL_LEASE_WAIT="1", RUST_LOG="info,qwen_diag=debug")
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
    model_id = Path(args.model).stem
    phases = {}
    outcomes = {}

    def phase(label, action):
        mark = len(log.read_text(errors="replace"))
        if action is not None:
            outcomes[label] = action()
            samples.append({"phase": label, "t": 0, "wired_gib": wired_gib()})
        poll_phase(addr, span, f"{label}_then_polling", samples)
        phases[label] = finishes(log.read_text(errors="replace")[mark:])

    try:
        deadline = time.monotonic() + 1800
        while "serve: listening on" not in log.read_text(errors="replace"):
            if process.poll() is not None or time.monotonic() > deadline:
                raise SystemExit(f"server not listening; see {log}")
            time.sleep(1)
        span = args.window + 12
        phase("startup", None)
        phase(
            "refused",
            lambda: post_refused(addr, model_id, args.max_context_tokens),
        )
        phase("request", lambda: post(addr, model_id))
        phase("aborted_after_submission", lambda: post_and_abort(addr, model_id))
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=60)
    samples.append({"phase": "after_exit", "t": 0, "wired_gib": wired_gib()})
    def renewed(label):
        return sum(1 for f in phases.get(label, []) if f["window"] == "renewed")

    verdicts = {
        # Model lists finish without a request start.
        "polling_never_renews": all(
            f["window"] == "unchanged"
            for label in phases
            for f in phases[label]
            if f["compute_encoders"] == "none"
        ),
        "refused_is_4xx": isinstance(outcomes.get("refused"), int)
        and 400 <= outcomes["refused"] < 500,
        "refused_never_renews": renewed("refused") == 0,
        "request_renews_once": renewed("request") == 1,
        "abort_after_submission_renews_once": outcomes.get("aborted_after_submission")
        is not None
        and renewed("aborted_after_submission") == 1,
    }
    report = {
        "model": Path(args.model).name,
        "window_s": args.window,
        "outcomes": outcomes,
        "verdicts": verdicts,
        "passed": all(verdicts.values()),
        "finishes": {
            label: {
                "total": len(found),
                "with_request": [f for f in found if f["compute_encoders"] != "none"],
                "renewed": renewed(label),
            }
            for label, found in phases.items()
        },
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
