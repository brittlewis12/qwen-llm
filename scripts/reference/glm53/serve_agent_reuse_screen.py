# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Map #13 scripted serve screen: how much of an agent-shaped GLM-5.3-Flash
conversation `qwen serve` reuses, request by request.

A scripted client sends the patched opencode client's request shape
(`opencode-shape-v1.json`: its instructions and 12 tool schemas; replayed
reasoning as `reasoning_text` content, assistant text as `output_text`,
calls as `function_call`, results as `function_call_output`) to a serve
process this script launches. No opencode process runs, and no request goes
to any server this script did not start. Tool results are canned.

Requests, in order:
  A1 cold        first request of session A
  A2.. continued each tool step of session A (results appended)
  B  branch      session A's first request with an edited user turn
  C  new session same instructions and tools, a different task
  A' return      session A again, continued after C

Per request: input items and reasoning items sent, the `serve phases:`
line (reused and prefilled tokens, prefill and decode ms), client time to
first streamed delta and wall time.

  uv run scripts/reference/glm53/serve_agent_reuse_screen.py --qwen target/release/qwen \\
    --model <gguf> --out <dir> [--max-context-tokens 32768] [--effort low]

Run detached under tool supervision (it drives a model-scale server). A
cost screen, not a quality or default-policy decision.
"""

import argparse
import http.client
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_A = "List the source files in this project, then read the README and summarize what the project does in two sentences."
TASK_A_EDITED = "List the source files in this project, then read Cargo.toml and say which crates it depends on."
TASK_C = "Find where errors are handled in this project and explain the approach in two sentences."
CANNED = {
    "bash": "Cargo.toml\nREADME.md\nsrc/\nsrc/main.rs\nsrc/lib.rs\nsrc/error.rs\n",
    "glob": "/work/project/src/main.rs\n/work/project/src/lib.rs\n/work/project/src/error.rs\n",
    "grep": "/work/project/src/error.rs:\n  Line 3: pub enum AppError {\n  Line 18: impl From<std::io::Error> for AppError {\n",
    "read": "<file>\n00001| # tally\n00002| \n00003| A small command-line tool that counts words, lines and bytes in text files,\n00004| like `wc`, with JSON output and glob support.\n00005| \n00006| ## Usage\n00007| \n00008| tally [--json] <paths...>\n</file>\n",
}


def client_shape(item):
    """The patched opencode client's replay of one output item."""
    kind = item["type"]
    if kind == "reasoning":
        text = "".join(part.get("text", "") for part in item.get("content", []))
        return {
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": text}],
        }
    if kind == "message":
        text = "".join(part.get("text", "") for part in item["content"])
        return {"role": "assistant", "content": [{"type": "output_text", "text": text}]}
    if kind == "function_call":
        return {
            "type": "function_call",
            "call_id": item["call_id"],
            "name": item["name"],
            "arguments": item["arguments"],
        }
    raise SystemExit(f"unexpected output item {kind}")


def post_stream(addr, body):
    """POST a streamed request; returns (response, first delta seconds, wall seconds)."""
    host, port = addr.split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=1800)
    started = time.monotonic()
    connection.request(
        "POST",
        "/v1/responses",
        body=json.dumps({**body, "stream": True}),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raise SystemExit(f"HTTP {response.status}: {response.read()[:400]!r}")
    first_delta, completed, buffer = None, None, b""
    while True:
        chunk = response.read1(65536)
        if not chunk:
            break
        buffer += chunk
        while b"\n\n" in buffer:
            block, buffer = buffer.split(b"\n\n", 1)
            data = [
                line[6:]
                for line in block.decode().splitlines()
                if line.startswith("data: ")
            ]
            if not data or data[0] == "[DONE]":
                continue
            event = json.loads(data[0])
            kind = event.get("type", "")
            if first_delta is None and kind.endswith(".delta"):
                first_delta = time.monotonic() - started
            if kind in ("response.completed", "response.incomplete"):
                completed = event["response"]
            if kind == "response.failed":
                raise SystemExit(f"response failed: {json.dumps(event)[:400]}")
    wall = time.monotonic() - started
    connection.close()
    if completed is None:
        raise SystemExit("stream ended without a terminal response")
    return completed, first_delta, wall


def wait_listening(log, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"server exited early: {process.returncode}; see {log}")
        if "serve: listening on" in log.read_text(errors="replace"):
            return
        time.sleep(1)
    raise SystemExit(f"server not listening after {timeout} s; see {log}")


class Screen:
    def __init__(self, addr, model_id, shape, effort, max_tokens, log):
        self.addr, self.model_id, self.shape = addr, model_id, shape
        self.effort, self.max_tokens, self.log = effort, max_tokens, log
        self.records = []

    def phases(self):
        return re.findall(r"serve phases: .*", self.log.read_text(errors="replace"))

    def send(self, label, kind, input_items):
        before = len(self.phases())
        body = {
            "model": self.model_id,
            "instructions": self.shape["instructions"],
            "tools": self.shape["tools"],
            "tool_choice": self.shape.get("tool_choice") or "auto",
            "input": input_items,
            "max_output_tokens": self.max_tokens,
            "reasoning": {"effort": self.effort},
        }
        response, first_delta, wall = post_stream(self.addr, body)
        time.sleep(0.5)
        new = self.phases()[before:]
        line = new[-1] if new else None
        fields = dict(re.findall(r"(\w+)=([\w.]+)", line)) if line else {}
        record = {
            "label": label,
            "kind": kind,
            "status": response["status"],
            "input_items": len(input_items),
            "reasoning_items_sent": sum(
                1 for i in input_items if i.get("type") == "reasoning"
            ),
            "output_items": [item["type"] for item in response["output"]],
            "usage": response.get("usage"),
            "first_delta_s": first_delta,
            "wall_s": wall,
            "serve_phases": line,
            "reused_tokens": int(fields["reused_tokens"])
            if "reused_tokens" in fields
            else None,
            "prefill_tokens": int(fields["prefill_tokens"])
            if "prefill_tokens" in fields
            else None,
            "prefill_ms": float(fields["prefill_ms"])
            if "prefill_ms" in fields
            else None,
            "decode_ms": float(fields["decode_ms"]) if "decode_ms" in fields else None,
        }
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in (
                        "label",
                        "kind",
                        "status",
                        "reused_tokens",
                        "prefill_tokens",
                        "prefill_ms",
                        "first_delta_s",
                        "wall_s",
                    )
                }
            ),
            flush=True,
        )
        self.records.append(record)
        return response

    def steps(self, session, history, first_label, steps):
        """Send `history`, then append results and continue while calls come."""
        response = self.send(
            first_label,
            "cold" if not self.records else "new_session",
            history,
        )
        for step in range(2, steps + 2):
            calls = [
                item for item in response["output"] if item["type"] == "function_call"
            ]
            if not calls:
                break
            history = history + [client_shape(item) for item in response["output"]]
            for call in calls:
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": CANNED.get(call["name"], "ok"),
                    }
                )
            response = self.send(f"{session}{step}", "continued", history)
        return history, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shape", default=str(HERE / "opencode-shape-v1.json"))
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--addr", default="127.0.0.1:8797")
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shape = json.loads(Path(args.shape).read_text())
    log = out / "serve.log"
    env = dict(os.environ, QWEN_METAL_LEASE_WAIT="1", RUST_LOG="info")
    env.pop("MTL_DEBUG_LAYER", None)
    command = [
        args.qwen,
        "serve",
        "-m",
        args.model,
        "--addr",
        args.addr,
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--max-tokens",
        str(args.max_tokens),
    ]
    with log.open("w") as sink:
        process = subprocess.Popen(
            command, stdout=sink, stderr=subprocess.STDOUT, env=env
        )
    # Serve answers only to the loaded model's id (the GGUF file stem).
    screen = Screen(
        args.addr, Path(args.model).stem, shape, args.effort, args.max_tokens, log
    )
    error = None
    try:
        wait_listening(log, process, args.ready_timeout)
        first = [{"role": "user", "content": TASK_A}]
        history_a, last_a = screen.steps("A", first, "A1", args.steps)
        screen.send("B", "branch", [{"role": "user", "content": TASK_A_EDITED}])
        screen.steps("C", [{"role": "user", "content": TASK_C}], "C1", 0)
        # Session A again: its latest output, then results or a follow-up.
        history = history_a + [client_shape(item) for item in last_a["output"]]
        calls = [item for item in last_a["output"] if item["type"] == "function_call"]
        for call in calls:
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": CANNED.get(call["name"], "ok"),
                }
            )
        if not calls:
            history.append(
                {
                    "role": "user",
                    "content": "Thanks. Now say which file you would read next, in one sentence.",
                }
            )
        screen.send("A'", "return", history)
    except SystemExit as exit_error:
        error = str(exit_error)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=60)
    report = {
        "schema": "glm53.serve_agent_reuse_screen.v1",
        "model": Path(args.model).name,
        "shape": Path(args.shape).name,
        "effort": args.effort,
        "max_context_tokens": args.max_context_tokens,
        "requests": screen.records,
        "error": error,
        "exit_code": process.returncode,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if error:
        raise SystemExit(error)


if __name__ == "__main__":
    main()
