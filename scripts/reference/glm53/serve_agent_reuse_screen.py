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
to any server this script did not start. Tool results are explicitly synthetic;
no tool command executes and no requested host path is read.

Requests, in order:
  A1 cold        first request of session A
  A2.. continued each tool step of session A (results appended)
  B  branch      session A's first request with an edited user turn
  C  new session same instructions and tools, a different task
  A' return      session A again, continued after C

Per request: input items and reasoning items sent, the `serve phases:`
line (reused and prefilled tokens, prefill and decode ms), client time to
first streamed delta (including reasoning), first answer delta, completed tool
availability and wall time. events.jsonl is flushed incrementally; report.json
summarizes outcomes even after failures. The inherited environment is unchanged;
validation-enabled runs are labelled. The 0.5-second settle is not readiness,
and HTTP 503 is reported without retry. B/C/return use the same bounded tool loop.

  uv run scripts/reference/glm53/serve_agent_reuse_screen.py --qwen target/release/qwen \\
    --model <gguf> --out <dir> [--max-context-tokens 32768] [--effort low]

Run detached under tool supervision (it drives a model-scale server). A
cost screen, not a quality or default-policy decision.
"""

import argparse
import hashlib
import http.client
import ipaddress
import json
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_A = "List the source files in this project, then read the README and summarize what the project does in two sentences."
TASK_A_EDITED = "List the source files in this project, then read Cargo.toml and say which crates it depends on."
TASK_C = "Find where errors are handled in this project and explain the approach in two sentences."
SETTLE_SECONDS = 0.5
FILES = {
    "README.md": "# tally\nA small command-line tool that counts words, lines and bytes in text files, like wc, with JSON output and glob support.\nUsage: tally [--json] <paths...>\n",
    "Cargo.toml": '[package]\nname = "tally"\nversion = "0.1.0"\nedition = "2021"\n[dependencies]\nclap = { version = "4", features = ["derive"] }\nserde_json = "1"\nglob = "0.3"\n',
    "src/error.rs": "use std::{fmt, io};\n#[derive(Debug)]\npub enum AppError { Io(io::Error), InvalidInput(String) }\nimpl From<io::Error> for AppError { fn from(e: io::Error) -> Self { Self::Io(e) } }\nimpl fmt::Display for AppError { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { match self { Self::Io(e) => write!(f, \"{e}\"), Self::InvalidInput(s) => write!(f, \"{s}\") } } }\nimpl std::error::Error for AppError {}\n",
}


def parsed_call(call):
    result = {k: call.get(k) for k in ("call_id", "name", "arguments")}
    try:
        def reject_constant(value):
            raise ValueError(f"non-JSON constant {value}")
        arguments = json.loads(call["arguments"], parse_constant=reject_constant)
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
        result.update(arguments_parsed=arguments, parse_error=None)
    except (ValueError, TypeError, KeyError) as error:
        result.update(arguments_parsed=None, parse_error=str(error))
    return result


def canned_result(call):
    """Virtual fixture only: never execute commands or read requested host paths."""
    parsed = parsed_call(call)
    args, name = parsed["arguments_parsed"], parsed["name"]
    result = {"synthetic": True, "fixture": "tally-canned-v2", "error": parsed["parse_error"]}
    if args is not None:
        if name == "read":
            path = str(args.get("filePath", "")).removeprefix("/work/project/").removeprefix("./")
            if path in FILES:
                result.update(path=path, content=FILES[path])
            else:
                result["error"] = f"unsupported synthetic read path: {path}"
        elif name in ("glob", "grep"):
            result["content"] = (
                "/work/project/src/main.rs\n/work/project/src/lib.rs\n/work/project/src/error.rs\n"
                if name == "glob" else "/work/project/src/error.rs:3: pub enum AppError\n/work/project/src/error.rs:4: impl From<io::Error> for AppError\n"
            )
            result["note"] = "canned discovery result, not a filesystem search"
        elif name == "bash":
            try:
                command = shlex.split(args.get("command", ""))
            except (ValueError, TypeError):
                command = []
            if command and (command[0] in ("ls", "find") or command[:2] == ["rg", "--files"]):
                result["content"] = "Cargo.toml\nREADME.md\nsrc/main.rs\nsrc/lib.rs\nsrc/error.rs\n"
                result["note"] = "canned listing, command was NOT executed"
            else:
                result["error"] = "unsupported synthetic bash command; nothing executed"
        else:
            result["error"] = f"unsupported synthetic tool: {name}; nothing executed"
    return {"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(result)}


def response_outcomes(response):
    output = response.get("output", [])
    text = "".join(part.get("text", "") for item in output if item["type"] == "message" for part in item.get("content", []))
    return {"answer_text": text, "has_answer_text": bool(text.strip()),
            "tool_calls": [parsed_call(i) for i in output if i["type"] == "function_call"],
            "incomplete_details": response.get("incomplete_details"), "response_error": response.get("error")}


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


def post_stream(addr, body, evidence):
    """Times include client SSE handling; first delta is not necessarily useful text."""
    host, port = addr.rsplit(":", 1)
    host = host.removeprefix("[").removesuffix("]")
    connection = http.client.HTTPConnection(host, int(port), timeout=1800)
    started = time.monotonic()
    timings = {"first_delta_s": None, "first_answer_delta_s": None, "first_tool_available_s": None}
    observations, completed, buffer = [], None, b""
    try:
        connection.request("POST", "/v1/responses", body=json.dumps({**body, "stream": True}),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        http_result = {"http_status": response.status, "retry_after": response.getheader("Retry-After")}
        if response.status != 200:
            http_result["body"] = response.read().decode("utf-8", errors="replace")
        evidence({"event": "http", **http_result})
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {http_result['body']}")
        while True:
            chunk = response.read1(65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n\n" in buffer:
                block, buffer = buffer.split(b"\n\n", 1)
                data = [line[6:] for line in block.decode().splitlines() if line.startswith("data: ")]
                if not data or data[0] == "[DONE]":
                    continue
                event = json.loads("\n".join(data))
                kind = event.get("type", "")
                elapsed = time.monotonic() - started
                # Keep partial output in memory; healthy streams incur no
                # per-delta evidence writes or flushes.
                if kind.endswith(".delta") or kind.startswith(("response.output_item.", "response.content_part.")):
                    observations.append({"elapsed_s": elapsed, "sse": event})
                if timings["first_delta_s"] is None and kind.endswith(".delta"):
                    timings["first_delta_s"] = elapsed
                if timings["first_answer_delta_s"] is None and kind == "response.output_text.delta":
                    timings["first_answer_delta_s"] = elapsed
                if timings["first_tool_available_s"] is None and kind == "response.output_item.done" and event.get("item", {}).get("type") == "function_call":
                    timings["first_tool_available_s"] = elapsed
                if kind in ("response.completed", "response.incomplete", "response.failed"):
                    completed = event["response"]
                    if timings["first_tool_available_s"] is None and any(i["type"] == "function_call" for i in completed.get("output", [])):
                        timings["first_tool_available_s"] = elapsed
                    evidence({"event": "terminal_response", "response": completed, "timings": dict(timings), "terminal_s": elapsed})
        timings["wall_s"] = time.monotonic() - started
        if completed is None:
            raise RuntimeError("stream ended without a terminal response")
        return completed, timings
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        if observations or buffer:
            evidence({"event": "stream_failure", "error": f"{type(error).__name__}: {error}",
                      "timings": {**timings, "wall_s": time.monotonic() - started},
                      "terminal_received": completed is not None, "observations": observations,
                      "unparsed_sse_tail": buffer.decode("utf-8", errors="replace")})
        raise
    finally:
        connection.close()


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
    def __init__(self, addr, model_id, shape, effort, max_tokens, log, evidence):
        self.addr, self.model_id, self.shape = addr, model_id, shape
        self.effort, self.max_tokens, self.log = effort, max_tokens, log
        self.evidence = evidence
        self.records = []
        self.workspace_rows = None

    def lines(self):
        return self.log.read_text(errors="replace").splitlines()

    def send(self, label, kind, input_items):
        before = len(self.lines())
        body = {"model": self.model_id, "instructions": self.shape["instructions"],
                "tools": self.shape["tools"], "tool_choice": self.shape.get("tool_choice") or "auto",
                "input": input_items, "max_output_tokens": self.max_tokens,
                "reasoning": {"effort": self.effort}}
        record = {"label": label, "kind": kind, "status": "client_error", "input_items": len(input_items),
                  "reasoning_items_sent": sum(i.get("type") == "reasoning" for i in input_items),
                  "settle_seconds": SETTLE_SECONDS, "error": None}
        def emit(event):
            if event["event"] == "http":
                record.update({k: v for k, v in event.items() if k != "event"})
            if event["event"] == "terminal_response":
                record["response"] = event["response"]
            if event["event"] == "stream_failure":
                record["stream_failure"] = event
            self.evidence({"label": label, **event})
        emit({"event": "request", "kind": kind, "body": {**body, "stream": True}})
        try:
            response, timings = post_stream(self.addr, body, emit)
            record.update(timings, status=response["status"], response=response,
                          output_items=[i["type"] for i in response.get("output", [])],
                          usage=response.get("usage"), **response_outcomes(response))
            if response["status"] == "failed":
                raise RuntimeError(f"response failed: {response.get('error')}")
            return response
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            # Retained settle, NOT a readiness handshake; never retry a 503.
            time.sleep(SETTLE_SECONDS)
            lines = self.lines()[before:]
            phase = next((line[line.index("serve phases:"):] for line in reversed(lines) if "serve phases:" in line), None)
            fields = dict(re.findall(r"(\w+)=([\w.]+)", phase or ""))
            for line in lines:
                match = re.search(r"glm5_next prefill plan rows=(\d+)", line)
                if match:
                    self.workspace_rows = int(match[1])
            record.update(raw_serve_lines=lines, serve_phases=phase, workspace_rows=self.workspace_rows,
                          cached_tokens=(record.get("usage") or {}).get("input_tokens_details", {}).get("cached_tokens"))
            for key in ("reused_tokens", "prefill_tokens", "snapshot_cache_bytes", "snapshot_cache_entries"):
                record[key] = int(fields[key]) if key in fields else None
            for key in ("prefill_ms", "decode_ms", "restore_ms", "boundary_ms", "snapshot_capture_ms"):
                record[key] = float(fields[key]) if key in fields else None
            for key in ("reuse_source", "lineage", "snapshot_schedule"):
                record[key] = fields.get(key)
            for key in ("cuts", "captures"):
                match = re.search(rf"{key}=\[([^]]*)\]", phase or "")
                record[key] = match[1] if match else None
            self.records.append(record)
            emit({"event": "request_result", **record})
            print(json.dumps({k: record.get(k) for k in ("label", "status", "reuse_source", "cached_tokens", "prefill_tokens", "workspace_rows", "first_delta_s", "wall_s", "error")}), flush=True)

    def replay(self, history, response):
        history = history + [client_shape(item) for item in response["output"]]
        return history + [canned_result(i) for i in response["output"] if i["type"] == "function_call"]

    def steps(self, session, history, first_label, steps, kind="new_session"):
        response = self.send(first_label, kind, history)
        used = 0
        while response["status"] == "completed" and any(i["type"] == "function_call" for i in response["output"]) and used < steps:
            history = self.replay(history, response)
            used += 1
            response = self.send(f"{session}{used + 1}", "continued", history)
        pending = any(i["type"] == "function_call" for i in response["output"])
        self.evidence({"event": "conversation_end", "session": session, "status": response["status"],
                       "continuations_used": used, "continuation_budget": steps,
                       "step_budget_exhausted": response["status"] == "completed" and pending and used == steps,
                       "pending_tools": pending, **response_outcomes(response)})
        return history, response


def file_sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def stop_owned(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                # Both earlier signals are cooperative in the CLI. Escalate
                # only this exact owned Popen, never a PID lookup/process group.
                process.kill()
                process.wait(timeout=60)


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
    try:
        host, port = args.addr.rsplit(":", 1)
        host = host.removeprefix("[").removesuffix("]")
        if not ipaddress.ip_address(host).is_loopback or not 0 < int(port) < 65536:
            raise ValueError("not loopback")
    except ValueError:
        parser.error("--addr must be a literal loopback address and positive port")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    log = out / "serve.log"
    report = {"schema": "glm53.serve_agent_reuse_screen.v2", "model": str(Path(args.model).resolve()),
              "shape": str(Path(args.shape).resolve()), "effort": args.effort,
              "max_context_tokens": args.max_context_tokens, "max_output_tokens": args.max_tokens,
              "continuation_budget": args.steps, "synthetic_tools": True, "canned_fixture": "tally-canned-v2", "requests": [],
              "error": None, "teardown_error": None, "exit_code": None}
    process = screen = None
    with (out / "events.jsonl").open("w") as events:
        def evidence(event):
            events.write(json.dumps(event) + "\n")
            events.flush()  # Process-failure evidence; no per-event fsync.
        try:
            shape = json.loads(Path(args.shape).read_text())
            env = os.environ.copy()  # No silent precision, logging, lease or MTL changes.
            overrides = {k: v for k, v in env.items() if k.startswith(("QWEN_GLM_", "QWEN_METAL_", "QWEN_SERVE_", "MTL_")) or k == "RUST_LOG"}
            command = [str(Path(args.qwen).resolve()), "serve", "-m", args.model, "--addr", args.addr,
                       "--max-context-tokens", str(args.max_context_tokens), "--max-tokens", str(args.max_tokens)]
            report.update(command=command, binary_sha256=file_sha(args.qwen), fixture_sha256=file_sha(args.shape),
                          script_sha256=file_sha(__file__), inherited_overrides=overrides,
                          source_revision_scope="checkout at launch; binary SHA is recorded separately, not a source attestation",
                          source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
                          source_dirty=subprocess.check_output(["git", "status", "--porcelain"], cwd=HERE, text=True),
                          timing_mode="validation-on" if any(env.get(k, "0") not in ("", "0") for k in ("MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION")) else "validation-not-requested",
                          timing_definition="first_delta includes reasoning/arguments; first_answer_delta is output_text.delta; tool availability is completed function_call; wall through EOF includes client terminal-evidence write, excludes 0.5s settle")
            evidence({"event": "run", **report})
            with log.open("w") as sink:
                process = subprocess.Popen(command, stdout=sink, stderr=subprocess.STDOUT, env=env)
            report["owned_pid"] = process.pid
            evidence({"event": "server_started", "owned_pid": process.pid})
            screen = Screen(args.addr, Path(args.model).stem, shape, args.effort, args.max_tokens, log, evidence)
            wait_listening(log, process, args.ready_timeout)
            evidence({"event": "startup", "raw_serve_lines": screen.lines()})
            history_a, last_a = screen.steps("A", [{"role": "user", "content": TASK_A}], "A1", args.steps, "cold")
            screen.steps("B", [{"role": "user", "content": TASK_A_EDITED}], "B", args.steps, "branch")
            screen.steps("C", [{"role": "user", "content": TASK_C}], "C1", args.steps)
            if last_a["status"] == "completed":
                history = screen.replay(history_a, last_a)
                if not any(i["type"] == "function_call" for i in last_a["output"]):
                    history.append({"role": "user", "content": "Thanks. Now say which file you would read next, in one sentence."})
                screen.steps("A_return", history, "A'", args.steps, "return")
            else:
                evidence({"event": "return_skipped", "reason": "A ended incomplete/failed; no fabricated completed turn"})
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            report["error"] = f"{type(error).__name__}: {error}"
            evidence({"event": "error", "error": report["error"]})
        finally:
            if process is not None:
                try:
                    stop_owned(process)
                except (Exception, KeyboardInterrupt) as error:
                    report["teardown_error"] = f"{type(error).__name__}: {error}"
                report["exit_code"] = process.poll()
            report["requests"] = screen.records if screen else []
            evidence({"event": "end", "error": report["error"], "teardown_error": report["teardown_error"], "exit_code": report["exit_code"]})
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["error"] or report["teardown_error"]:
        raise SystemExit(report["error"] or report["teardown_error"])


if __name__ == "__main__":
    main()
