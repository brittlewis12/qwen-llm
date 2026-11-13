#!/usr/bin/env python3
"""Raw-socket SSE cadence probe for `qwen serve` (stdlib only).

Launches one resident server, warms one or three agents from a source corpus,
then runs a fixed measurement window of streamed Responses requests. The
socket client parses HTTP framing (including chunked responses) itself and
records every recv chunk and every complete SSE event. It writes a JSON
summary plus per-response event traces.

Run under scripts/bench/with-metal-lease; the server inherits the lease fd.

    with-metal-lease --task sse --for 15m -- python3 sse_cadence.py run \
      --binary .../qwen --tokenizer .../qwen-tok --model ...gguf \
      --corpus .../code.txt --family v41 --agents 3 --window-seconds 45 \
      --out .../sse-v41-three
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import select
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_realistic as bench  # shared corpus/tokenizer/thermal helpers

CANONICAL_DELTAS = {
    "response.output_text.delta",
    "response.reasoning.delta",
    "response.function_call_arguments.delta",
}
TERMINAL = {"response.completed", "response.incomplete", "response.failed"}


def now():
    return time.perf_counter()


def pct(values, fraction):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, math.ceil(fraction * len(values)) - 1)]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class Dechunker:
    """Incremental HTTP/1.1 chunked body decoder."""

    def __init__(self):
        self.buf = b""
        self.remaining = 0
        self.need_size = True
        self.done = False

    def feed(self, data):
        self.buf += data
        out = bytearray()
        while not self.done:
            if self.need_size:
                index = self.buf.find(b"\r\n")
                if index < 0:
                    break
                line = self.buf[:index].split(b";", 1)[0].strip()
                self.buf = self.buf[index + 2:]
                try:
                    size = int(line, 16)
                except ValueError:
                    raise RuntimeError(f"bad chunk size {line!r}")
                if size == 0:
                    self.done = True
                    break
                self.remaining = size
                self.need_size = False
            else:
                take = min(self.remaining, len(self.buf))
                out += self.buf[:take]
                self.buf = self.buf[take:]
                self.remaining -= take
                if self.remaining:
                    break
                if self.buf[:2] != b"\r\n":
                    if len(self.buf) < 2:
                        break
                    raise RuntimeError("missing chunk terminator")
                self.buf = self.buf[2:]
                self.need_size = True
        return bytes(out)


def parse_headers(raw):
    lines = raw.split(b"\r\n")
    status = lines[0].decode("latin-1")
    headers = {}
    for line in lines[1:]:
        if not line:
            continue
        name, _, value = line.partition(b":")
        headers[name.decode("latin-1").strip().lower()] = value.decode("latin-1").strip()
    return status, headers


class Probe:
    def __init__(self, host, port, path, body, timeout=900):
        self.host, self.port, self.path = host, port, path
        self.body = body
        self.timeout = timeout
        self.sent = None
        self.header_at = None
        self.chunks = []          # (relative seconds, bytes)
        self.events = []          # decoded SSE events
        self.error = None
        self.status = None
        self.headers = {}
        self.body_bytes = bytearray()

    def run(self):
        payload = json.dumps(self.body, ensure_ascii=False).encode()
        request = (
            f"POST {self.path} HTTP/1.1\r\n"
            f"host: {self.host}:{self.port}\r\n"
            "content-type: application/json\r\n"
            "accept: text/event-stream\r\n"
            "connection: close\r\n"
            f"content-length: {len(payload)}\r\n\r\n"
        ).encode() + payload
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        try:
            sock.setblocking(False)
            self.sent = now()
            view = memoryview(request)
            while view:
                _, ready, _ = select.select([], [sock], [], self.timeout)
                if not ready:
                    raise TimeoutError("request send stalled")
                sent = sock.send(view)
                view = view[sent:]
            buf = bytearray()
            header_end = None
            while True:
                ready, _, _ = select.select([sock], [], [], self.timeout)
                if not ready:
                    raise TimeoutError("stream read stalled")
                data = sock.recv(65536)
                t = now()
                if not data:
                    break
                self.chunks.append((t - self.sent, len(data)))
                if header_end is None:
                    buf += data
                    marker = buf.find(b"\r\n\r\n")
                    if marker < 0:
                        continue
                    header_end = marker + 4
                    raw_headers = bytes(buf[:marker])
                    body = bytes(buf[header_end:])
                    self.status, self.headers = parse_headers(raw_headers)
                    self.header_at = t - self.sent
                    self.body_bytes += body
                else:
                    body = data
                    self.body_bytes += body
                self._feed_body(body, t - self.sent)
            # Flush any final complete events.
            self._drain_events(now() - self.sent)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            try:
                sock.close()
            except OSError:
                pass
        return self

    def _feed_body(self, data, stamp):
        chunked = self.headers.get("transfer-encoding", "").lower() == "chunked"
        if chunked:
            if not hasattr(self, "_dechunker"):
                self._dechunker = Dechunker()
            decoded = self._dechunker.feed(data)
        else:
            decoded = data
        if decoded:
            self._buffer = getattr(self, "_buffer", b"") + decoded
            self._drain_events(stamp)

    def _drain_events(self, stamp):
        buf = getattr(self, "_buffer", b"")
        while True:
            index = buf.find(b"\n\n")
            if index < 0:
                break
            frame, buf = buf[:index], buf[index + 2:]
            self._record_event(frame, stamp)
        self._buffer = buf

    def _record_event(self, frame, stamp):
        event_type = None
        data_lines = []
        for line in frame.split(b"\n"):
            if line.startswith(b"event:"):
                event_type = line[6:].strip().decode("utf-8", "replace")
            elif line.startswith(b"data:"):
                data_lines.append(line[5:].strip())
        if event_type is None:
            return
        raw = b"\n".join(data_lines)
        if raw == b"[DONE]":
            self.events.append({"t": stamp, "type": "[DONE]"})
            return
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {"_raw": raw.decode("utf-8", "replace")}
        self.events.append({
            "t": stamp,
            "type": event_type,
            "sequence_number": payload.get("sequence_number"),
            "output_index": payload.get("output_index"),
            "delta": payload.get("delta") if isinstance(payload.get("delta"), str) else None,
            "response": payload.get("response") if event_type in TERMINAL else None,
        })

    def summary(self):
        deltas = [e for e in self.events if e["type"] in CANONICAL_DELTAS]
        deltas = [e for e in deltas if e.get("delta")]
        intervals = [(b["t"] - a["t"]) * 1e3 for a, b in zip(deltas, deltas[1:])]
        raw_times = [t for t, _ in self.chunks]
        # Largest no-byte gap after the first content delta.
        first_delta = deltas[0]["t"] if deltas else None
        gaps = []
        if first_delta is not None:
            marks = [first_delta] + [t for t in raw_times if t >= first_delta]
            gaps = [(b - a) * 1e3 for a, b in zip(marks, marks[1:])]
        terminal = next((e for e in self.events if e["type"] in TERMINAL), None)
        response = terminal.get("response") if terminal and isinstance(terminal.get("response"), dict) else {}
        usage = response.get("usage") or {}
        phases = ((response.get("x_qwen") or {}).get("phases") or {})
        transitions = phases.get("transitions")
        if transitions is None:
            transitions = (phases.get("batch") or {}).get("stepped_transitions")
        if transitions is None and usage.get("output_tokens"):
            transitions = max(1, usage["output_tokens"] - 1)
        transition_ms = phases.get("transition_ms")
        if transition_ms is None:
            transition_ms = phases.get("decode_ms")
        server_itl = (transition_ms / transitions) if transition_ms is not None and transitions else None
        text = {
            "reasoning": "".join(e["delta"] for e in self.events
                                if e["type"] == "response.reasoning.delta" and e.get("delta")),
            "visible": "".join(e["delta"] for e in self.events
                               if e["type"] == "response.output_text.delta" and e.get("delta")),
            "calls": "".join(e["delta"] for e in self.events
                             if e["type"] == "response.function_call_arguments.delta" and e.get("delta")),
        }
        return {
            "status": self.status,
            "headers": self.headers,
            "error": self.error,
            "sent": self.sent,
            "header_s": self.header_at,
            "first_event_s": (self.events[0]["t"] if self.events else None),
            "first_delta_s": first_delta,
            "terminal_s": (terminal["t"] if terminal else None),
            "event_count": len(self.events),
            "delta_count": len(deltas),
            "inter_delta_ms": {
                "p50": pct(intervals, 0.50),
                "p95": pct(intervals, 0.95),
                "p99": pct(intervals, 0.99),
                "max": max(intervals, default=None),
            },
            "largest_no_byte_gap_ms_after_first_delta": max(gaps, default=None),
            "recv_chunks": len(self.chunks),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
            "phases": phases,
            "transitions": transitions,
            "server_itl_ms": server_itl,
            "server_prefill_ms": phases.get("prefill_ms"),
            "text_sha256": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in text.items()},
            "text_chars": {key: len(value) for key, value in text.items()},
            "events": self.events,
        }


class ServerLog:
    def __init__(self, process):
        self.process = process
        self.lines = []
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            self.lines.append({"time": time.time(), "line": line.rstrip()})

    def join(self):
        self.thread.join(timeout=5)

    def phases(self):
        return [bench.kv(entry["line"]) for entry in self.lines
                if "serve phases:" in entry["line"]]


class Runner:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.lease_fd = int(os.environ["QWEN_METAL_LEASE_FD"])
        os.fstat(self.lease_fd)
        self.server = None
        self.log = None
        self.model_id = None
        self.stop = threading.Event()
        self.rows = []
        self.lock = threading.Lock()

    def pressure(self):
        return bench.pressure()

    def settle(self, seconds, limit=1, label="preflight"):
        if seconds <= 0:
            return
        since = None
        started = now()
        while True:
            level = self.pressure()
            if level < 0 or level > limit:
                since = None
            else:
                since = since if since is not None else now()
                if now() - since >= seconds:
                    return
            if now() - started >= self.args.settle_timeout:
                raise RuntimeError(
                    f"thermal did not settle at <={limit} for {seconds}s "
                    f"within {self.args.settle_timeout}s during {label}")
            time.sleep(1)

    def prepare(self):
        args = self.args
        text = args.corpus.read_text().replace('<|', '‹|').replace('<｜', '‹｜')
        ids = bench.token_text(args.tokenizer, args.model, text=text)
        n = args.warm_tokens
        if len(ids) < n * 3:
            raise ValueError("corpus needs three disjoint warm roots")
        roots = [bench.token_text(args.tokenizer, args.model, ids=ids[a*n:(a+1)*n])
                 for a in range(3)]
        # Per-turn tool-style suffix excerpts, drawn outside the warm roots.
        sizes = sorted(set(args.reply_tokens))
        suffix = {}
        for a in range(3):
            for size in sizes:
                start = (a * 977) % (len(ids) - size)
                suffix[(a, size)] = bench.token_text(args.tokenizer, args.model,
                                                     ids=ids[start:start + size])
        return roots, suffix

    def start_server(self):
        args = self.args
        env = dict(os.environ, RUST_LOG="warn,qwen_diag=info", NO_COLOR="1")
        env.pop("QWEN_SERVE_V41_MAX_ACTIVE", None)
        for item in args.env:
            key, sep, value = item.partition("=")
            if not sep or not key:
                raise ValueError(f"--env wants KEY=VALUE, got {item!r}")
            env[key] = value
        maximum = max(args.reply_tokens)
        command = [str(args.binary.resolve()), "serve", "-m", str(args.model.resolve()),
                   "--addr", f"127.0.0.1:{args.port}",
                   "--max-context-tokens", str(args.context),
                   "--max-tokens", str(maximum)]
        dump(self.out / "launch.json", {
            "command": command, "env": {k: v for k, v in env.items() if k.startswith("QWEN_")},
            "family": args.family, "agents": args.agents,
            "window_seconds": args.window_seconds, "reply_tokens": args.reply_tokens,
            "warm_tokens": args.warm_tokens, "context": args.context,
            "harness_commit": args.harness_commit,
        })
        self.server = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       pass_fds=(self.lease_fd,))
        self.log = ServerLog(self.server)
        started = time.time()
        while time.time() - started < 300:
            if self.server.poll() is not None:
                raise RuntimeError(f"server exited {self.server.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", args.port), timeout=1) as s:
                    s.sendall(b"GET /v1/models HTTP/1.1\r\nhost: x\r\n\r\n")
                    data = b""
                    while b"\r\n\r\n" not in data:
                        more = s.recv(65536)
                        if not more:
                            break
                        data += more
                    if b"HTTP/1.1 200" in data:
                        head, _, body = data.partition(b"\r\n\r\n")
                        # Close-delimited JSON body; read the rest if needed.
                        while not body.rstrip().endswith(b"}"):
                            more = s.recv(65536)
                            if not more:
                                break
                            body += more
                        self.model_id = json.loads(body)["data"][0]["id"]
                        return
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.25)
        raise TimeoutError("server startup")

    def body(self, history, count, seed):
        args = self.args
        body = {
            "model": self.model_id,
            "input": history,
            "instructions": bench.INSTRUCTIONS,
            "stream": True,
            "max_output_tokens": count,
            "x_qwen": {"stats": True, "seed": seed},
            "tools": [{"type": "function", "name": "read_file", "description": "Read source",
                       "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                      "required": ["path"], "additionalProperties": False}}],
        }
        if args.family in ("glm", "fn"):
            body["reasoning"] = {"effort": "low"}
        return body

    def request(self, history, count, label, warm, seed):
        probe = Probe("127.0.0.1", self.args.port, "/v1/responses",
                      self.body(history, count, seed), timeout=self.args.timeout).run()
        summary = probe.summary()
        summary.update(label=label, warm=warm, target=count, seed=seed)
        with self.lock:
            self.rows.append(summary)
        # Trim the bulky event trace from memory in the aggregate; it is
        # written separately.
        return summary, probe

    def run(self):
        args = self.args
        roots, suffix = self.prepare()
        self.settle(args.settle_seconds, args.settle_limit, "before_load")
        self.start_server()
        histories = []
        traces = []
        # Warm each agent, then build a continuing tool-loop history.
        for agent in range(args.agents):
            history = [{"role": "user",
                        "content": f"Independent agent {agent}. Source corpus follows.\n" + roots[agent]}]
            summary, probe = self.request(history, 8, f"agent{agent}-warm", True,
                                          args.seed + agent)
            final = next((e["response"] for e in probe.events
                          if e["type"] in TERMINAL and e.get("response")), None)
            if final is None:
                raise RuntimeError(f"warm-up failed: {summary.get('error')}")
            history = history + bench.history_items(final.get("output", []))
            histories.append(history)
            traces.append(("warm", agent, summary))
            if args.warm_cool and agent + 1 < args.agents:
                time.sleep(args.warm_cool)
        self.settle(args.settle_seconds, args.settle_limit, "after_warm")
        barrier = threading.Barrier(args.agents + 1)
        window_begin = now() + args.ramp
        window_end = window_begin + args.window_seconds
        results = []

        def agent_loop(agent):
            history = histories[agent]
            rng = random.Random(f"{args.seed}:{agent}")
            turn = 0
            barrier.wait()
            while not self.stop.is_set() and now() < window_end:
                level = self.pressure()
                if level < 0 or level > args.abort_thermal:
                    self.stop.set()
                    break
                count = args.reply_tokens[turn % len(args.reply_tokens)]
                size = count
                call = f"agent{agent}_read{turn}"
                history = history + [
                    {"type": "function_call", "call_id": call, "name": "read_file",
                     "arguments": json.dumps({"path": f"src/agent{agent}/excerpt{turn}.rs"})},
                    {"type": "function_call_output", "call_id": call,
                     "output": suffix[(agent, size)]},
                ]
                summary, probe = self.request(history, count, f"agent{agent}-t{turn}", False,
                                              args.seed + 1000 + agent * 100 + turn)
                final = next((e["response"] for e in probe.events
                              if e["type"] in TERMINAL and e.get("response")), None)
                if final is None:
                    results.append(summary)
                    break
                history = history + bench.history_items(final.get("output", []))
                with self.lock:
                    traces.append(("measured", agent, summary))
                turn += 1
                if args.turns and turn >= args.turns:
                    break
                if args.tool_gap:
                    time.sleep(args.tool_gap)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.agents) as pool:
            futures = [pool.submit(agent_loop, agent) for agent in range(args.agents)]
            barrier.wait()
            while now() < window_end and not self.stop.is_set():
                if all(future.done() for future in futures):
                    if args.turns:
                        break
                    for future in futures:
                        future.result()
                    raise RuntimeError("agent exited before the common cutoff")
                time.sleep(0.1)
            self.stop.set()
            for future in concurrent.futures.as_completed(futures, timeout=120):
                future.result()
        elapsed = now() - window_begin
        measured = [row for kind, _, row in traces if kind == "measured"]
        if args.turns:
            starts = [row["sent"] for row in measured]
            ends = [row["sent"] + row["terminal_s"] for row in measured
                    if row["terminal_s"] is not None]
            if starts and ends:
                elapsed = max(ends) - min(starts)
        # Throughput: completed measured output inside the window.
        completed = [r for r in measured if r["output_tokens"] is not None
                     and r["terminal_s"] is not None and r["terminal_s"] <= window_end]
        tokens = sum(r["output_tokens"] for r in completed)
        throughput = tokens / elapsed if elapsed > 0 else None
        # Cadence: all measured responses with at least two canonical deltas.
        cadence = [r for r in measured if r["delta_count"] >= 2]
        all_intervals = [x for r in cadence for x in
                         [b["t"] - a["t"] for a, b in zip(
                             [e for e in r["events"] if e["type"] in CANONICAL_DELTAS and e.get("delta")],
                             [e for e in r["events"] if e["type"] in CANONICAL_DELTAS and e.get("delta")][1:])]]
        all_intervals = [x * 1e3 for x in all_intervals]
        per_response = [{k: r[k] for k in (
            "label", "sent", "header_s", "first_event_s", "first_delta_s", "terminal_s",
            "event_count", "delta_count", "inter_delta_ms",
            "largest_no_byte_gap_ms_after_first_delta", "input_tokens", "output_tokens",
            "cached_tokens", "transitions", "server_itl_ms", "server_prefill_ms",
            "text_sha256", "text_chars", "status", "error")} for r in measured]
        summary = {
            "family": args.family,
            "agents": args.agents,
            "window_seconds": args.window_seconds,
            "window_begin": window_begin,
            "window_end": window_end,
            "elapsed_s": elapsed,
            "reply_tokens": args.reply_tokens,
            "warm_tokens": args.warm_tokens,
            "requests_measured": len(measured),
            "requests_completed": len(completed),
            "output_tokens": tokens,
            "aggregate_tps": throughput,
            "cadence_responses": len(cadence),
            "inter_delta_ms": {
                "p50": pct(all_intervals, 0.50),
                "p95": pct(all_intervals, 0.95),
                "p99": pct(all_intervals, 0.99),
                "max": max(all_intervals, default=None),
            },
            "first_delta_s": {
                "p50": pct([r["first_delta_s"] for r in cadence], 0.50),
                "p95": pct([r["first_delta_s"] for r in cadence], 0.95),
                "p99": pct([r["first_delta_s"] for r in cadence], 0.99),
            },
            "server_prefill_ms": {
                "p50": pct([r["server_prefill_ms"] for r in cadence], 0.50),
                "p95": pct([r["server_prefill_ms"] for r in cadence], 0.95),
            },
            "server_itl_ms": {
                "p50": pct([r["server_itl_ms"] for r in cadence], 0.50),
                "p95": pct([r["server_itl_ms"] for r in cadence], 0.95),
                "p99": pct([r["server_itl_ms"] for r in cadence], 0.99),
            },
            "largest_no_byte_gap_ms_after_first_delta": max(
                (r["largest_no_byte_gap_ms_after_first_delta"] for r in cadence), default=None),
            "responses": per_response,
            "transport": "chunked" if any(
                r.get("headers", {}).get("transfer-encoding", "").lower() == "chunked"
                for _, _, r in traces) else "close-delimited",
        }
        dump(self.out / "summary.json", summary)
        dump(self.out / "events.json", [
            {"kind": kind, "agent": agent, **{k: row[k] for k in (
                "label", "sent", "header_s", "first_event_s", "first_delta_s", "terminal_s",
                "event_count", "delta_count", "inter_delta_ms", "server_itl_ms",
                "server_prefill_ms", "events")}}
            for kind, agent, row in traces])
        # Compact server lines for transition/cadence evidence.
        dump(self.out / "server-lines.json", self.log.lines)
        print(json.dumps({k: v for k, v in summary.items() if k != "responses"},
                         indent=2), flush=True)
        return summary

    def stop_server(self):
        if self.server is not None and self.server.poll() is None:
            self.server.send_signal(signal.SIGINT)
            try:
                self.server.wait(timeout=40)
            except subprocess.TimeoutExpired:
                self.server.terminate()
                self.server.wait(timeout=10)
        if self.log is not None:
            self.log.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("binary", "tokenizer", "model", "corpus", "out"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--family", choices=("v41", "glm", "fn"), required=True)
    run.add_argument("--agents", type=int, choices=(1, 3), default=1)
    run.add_argument("--window-seconds", type=float, default=45)
    run.add_argument("--turns", type=int, default=0,
                     help="stop each agent after this many measured requests (0: until the window ends)")
    run.add_argument("--reply-tokens", type=int, nargs="+", default=[512])
    run.add_argument("--warm-tokens", type=int, default=12000)
    run.add_argument("--context", type=int, default=32768)
    run.add_argument("--ramp", type=float, default=5)
    run.add_argument("--tool-gap", type=float, default=0.0)
    run.add_argument("--warm-cool", type=float, default=0.0)
    run.add_argument("--port", type=int, default=18738)
    run.add_argument("--timeout", type=float, default=900)
    run.add_argument("--seed", type=int, default=20261112)
    run.add_argument("--settle-seconds", type=float, default=0)
    run.add_argument("--settle-timeout", type=float, default=600)
    run.add_argument("--settle-limit", type=int, default=1)
    run.add_argument("--allow-thermal", type=int, default=2)
    run.add_argument("--abort-thermal", type=int, default=3)
    run.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--harness-commit", default="working-tree")
    args = parser.parse_args()
    if args.command != "run":
        parser.error("unknown command")
    if any(size < 1 for size in args.reply_tokens):
        parser.error("--reply-tokens must be positive")
    runner = Runner(args)

    def stop(sig, frame):
        runner.stop.set()
        runner.stop_server()
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        runner.run()
    finally:
        runner.stop_server()


if __name__ == "__main__":
    main()
