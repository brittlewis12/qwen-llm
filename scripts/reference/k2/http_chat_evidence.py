# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""HTTP evidence for `check_chat_cli.py --http-evidence`: K2 serve answers the
same greedy low-effort request the CLI check runs ("What is 2+2? Answer
briefly.", 128 tokens), non-streaming. Starts and stops only its own server.

  uv run scripts/reference/k2/http_chat_evidence.py --binary target/release/qwen \\
    --model K2-Horizon-7B-Q8_0.gguf --output evidence.json
"""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ADDR = "127.0.0.1:8795"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    log = args.output.with_suffix(".serve.log")
    env = dict(os.environ, QWEN_METAL_LEASE_WAIT="1")
    with log.open("w") as sink:
        server = subprocess.Popen(
            [
                str(args.binary),
                "serve",
                "-m",
                str(args.model),
                "--addr",
                ADDR,
                "--max-context-tokens",
                "1024",
                "--max-tokens",
                "128",
            ],
            stdout=sink,
            stderr=subprocess.STDOUT,
            env=env,
        )
    try:
        deadline = time.monotonic() + 600
        while "serve: listening on" not in log.read_text(errors="replace"):
            assert server.poll() is None and time.monotonic() < deadline, log
            time.sleep(0.5)
        body = json.dumps(
            {
                "model": args.model.stem,
                "input": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
                "reasoning": {"effort": "low"},
                "max_output_tokens": 128,
                "temperature": 0,
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            f"http://{ADDR}/v1/responses",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as reply:
            response = json.loads(reply.read())
    finally:
        server.send_signal(signal.SIGINT)
        server.wait(timeout=120)
    evidence = {
        "status": "passed" if response["status"] == "completed" else "failed",
        "cases": [{"effort": "low", "budget": 128, "response": response}],
    }
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": evidence["status"], "usage": response["usage"]}))


if __name__ == "__main__":
    main()
