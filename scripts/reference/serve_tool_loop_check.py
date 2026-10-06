# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Live tool-loop check for `qwen serve` (map #11): one request with a
declared function, then the replayed call plus its output, non-streamed and
streamed. Records the returned items, the parsed call arguments, and each
request's `serve phases:` line (reused vs prefilled tokens). Starts and stops
only the server process it launched.

Usage:
  uv run scripts/reference/serve_tool_loop_check.py --qwen target/release/qwen \
    --model <gguf> --out <dir> [--max-context-tokens 4096] [--effort low]

Run detached under tool supervision (it drives a model-scale server).
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

TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "description": "Forecast days (1-7)"},
            },
            "required": ["city"],
        },
    }
]
QUESTION = "What's the weather in Paris for the next 2 days? Use the tool."


def post(addr, body, stream):
    data = json.dumps({**body, "stream": stream}).encode()
    request = urllib.request.Request(
        f"http://{addr}/v1/responses",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        raw = response.read().decode()
    if not stream:
        return json.loads(raw), []
    events = []
    completed = None
    for block in raw.split("\n\n"):
        payload = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if not payload or payload[0] == "[DONE]":
            continue
        event = json.loads(payload[0])
        events.append(event.get("type"))
        if event.get("type") == "response.completed":
            completed = event["response"]
    return completed, events


def wait_listening(log, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"server exited early: {process.returncode}; see {log}")
        if "serve: listening on" in log.read_text(errors="replace"):
            return
        time.sleep(1)
    raise SystemExit(f"server not listening after {timeout} s; see {log}")


def loop(addr, model_id, effort, stream):
    first, first_events = post(
        addr,
        {
            "model": model_id,
            "input": QUESTION,
            "tools": TOOLS,
            "reasoning": {"effort": effort},
        },
        stream,
    )
    output = first["output"]
    calls = [item for item in output if item["type"] == "function_call"]
    record = {
        "stream": stream,
        "first_status": first["status"],
        "first_items": [item["type"] for item in output],
        "calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls],
        "first_events": sorted(set(first_events)),
    }
    if not calls:
        record["first_output"] = output
        return record
    replay = [{"role": "user", "content": QUESTION}]
    for item in output:
        if item["type"] == "reasoning":
            replay.append({"type": "reasoning", "content": item["content"]})
        elif item["type"] == "message":
            replay.append(
                {"type": "message", "role": "assistant", "content": item["content"]}
            )
        elif item["type"] == "function_call":
            replay.append(
                {
                    "type": "function_call",
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": item["arguments"],
                }
            )
    for call in calls:
        replay.append(
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": "Paris: 18C clear, then 21C sunny.",
            }
        )
    second, _ = post(
        addr,
        {
            "model": model_id,
            "input": replay,
            "tools": TOOLS,
            "reasoning": {"effort": effort},
        },
        stream,
    )
    record["second_status"] = second["status"]
    record["second_items"] = [item["type"] for item in second["output"]]
    record["second_text"] = "".join(
        part.get("text", "")
        for item in second["output"]
        if item["type"] == "message"
        for part in item["content"]
    )[:400]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--ready-timeout", type=float, default=1200.0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "serve.log"
    addr = "127.0.0.1:8793"
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
        str(args.max_tokens),
    ]
    with log.open("w") as sink:
        process = subprocess.Popen(
            command, stdout=sink, stderr=subprocess.STDOUT, env=env
        )
    records = []
    try:
        wait_listening(log, process, args.ready_timeout)
        model_id = Path(args.model).stem
        for stream in (False, True):
            records.append(loop(addr, model_id, args.effort, stream))
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=60)
    phases = re.findall(r"serve phases: .*", log.read_text(errors="replace"))
    report = {
        "model": Path(args.model).name,
        "records": records,
        "serve_phases": phases,
        "exit_code": process.returncode,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
