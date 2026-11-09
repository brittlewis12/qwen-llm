#!/usr/bin/env python3
"""Seeded warm-agent concurrency workload (stdlib only).

plan samples paired append/reply/tool-duration observations from the recorded
OpenCode calls. run records a serial reference, then replays its new inputs
with each candidate's own replies. Frozen-history replay is a diagnostic option.
Warm-up is separate. compare refuses unequal offered work/output counts.
Run each arm under the build lock and with-metal-lease; see the packet.
"""

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import statistics
import subprocess
import threading
import time


DEFAULT_CALLS = Path(__file__).resolve().parents[2] / "docs/bench/2026-11-08-real-opencode-agent-calls.csv"
DELTA_TYPES = {"response.output_text.delta", "response.reasoning_text.delta",
               "response.function_call_arguments.delta"}
INSTRUCTIONS = (
    "You are reviewing a Rust/Metal inference engine. The source corpus is authentic. "
    "Produce a detailed line-by-line engineering review, including invariants, failure "
    "paths and proposed tests for every function in the supplied source. This is a "
    "continuing report: keep writing substantive observations until interrupted by "
    "the output limit. Do not conclude, summarize, or call tools. Later excerpts are "
    "tool-style source results; continue the same report after each one."
)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(fraction * len(values)) - 1)]


def make_plan(rows, family, seed, seconds):
    # Snapshot fallbacks include replayed history rather than just new tool input.
    # Cold roots, title calls and effort changes are not warm-agent appends.
    rows = [r for r in rows if r["model"] == family and r["purpose"] == "primary"
            and int(r["reused"]) > 0 and r["reuse_source"] in ("live_session", "live")]
    if not rows:
        raise ValueError("no primary live-prefix observations for family")
    rng = random.Random(f"{seed}:{family}")
    agents = [[] for _ in range(3)]
    loads = [0.0] * 3
    while sum(loads) < seconds or any(not a for a in agents):
        agent = min(range(3), key=lambda a: loads[a])
        row = rng.choice(rows)
        turn = {"source_call": row["call"], "append_tokens": int(row["new"]),
                "reply_tokens": int(row["generated"]),
                "tool_gap_s": float(row["tool_union_ms"]) / 1000,
                "source_active_s": (float(row["server_pre_ms"]) + float(row["decode_ms"])) / 1000}
        agents[agent].append(turn)
        loads[agent] += turn["source_active_s"] + turn["tool_gap_s"]
    return {"version": 1, "family": family, "seed": seed, "target_serial_s": seconds,
            "population": len(rows), "agents": agents, "estimated_agent_work_s": loads}


def token_text(binary, model, text=None, ids=None):
    if ids is None:
        cmd = [str(binary), "-m", str(model), "--file", "-", "--ids"]
        result = subprocess.check_output(cmd, input=text.encode()).decode().splitlines()
        return list(map(int, result[1:]))
    cmd = [str(binary), "-m", str(model), "--decode-ids", "-"]
    return subprocess.check_output(cmd, input=" ".join(map(str, ids)).encode()).decode()


def history_items(items):
    out = []
    has_assistant = False
    for item in items:
        kind = item.get("type")
        if kind == "reasoning":
            out.append({"type": "reasoning", "content": "".join(p.get("text", "") for p in item.get("content", []))})
        elif kind == "message":
            out.append({"role": "assistant", "content": "".join(p.get("text", "") for p in item.get("content", []))})
            has_assistant = True
        elif kind == "function_call":
            out.append({k: item[k] for k in ("type", "call_id", "name", "arguments")})
            out.append({"type": "function_call_output", "call_id": item["call_id"],
                        "output": "Source is supplied in the next excerpt."})
            has_assistant = True
    if not has_assistant:
        out.append({"role": "assistant", "content": ""})
    return out


def pressure():
    return int(subprocess.check_output(
        ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], text=True).split()[-1])


def kv(line):
    return {a: b.strip('"') for a, b in re.findall(r'(\w+)=("[^"\n]*"|\[[^\]]*\]|[^ \n]+)', line)}


def interval_buckets(rows, begin, end):
    """Exclusive *observed stream* overlap, not GPU kernel/batch service time.

    Pre-first-token spans include queueing. Decode spans include interleaved
    appends. Native batch counters are reported separately to avoid equating
    the number of open streams with the width of a native forward.
    """
    edges = [(end, 0, 0)]
    for r in rows:
        edges.extend((t, d, w) for t, d, w in
                     [(r["sent"], 0, 1), (r["first"], 1, -1), (r["end"], -1, 0)]
                     if begin < t < end)
    decode = sum(r["first"] <= begin < r["end"] for r in rows)
    waiting = sum(r["sent"] <= begin < r["first"] for r in rows)
    result = {"decode_stream_width_1_s": 0.0, "decode_stream_width_2_s": 0.0,
              "decode_stream_width_3_s": 0.0, "predecode_only_s": 0.0, "idle_s": 0.0}
    last = begin
    for t, d, w in sorted(edges):
        if t > last:
            key = f"decode_stream_width_{decode}_s" if decode else ("predecode_only_s" if waiting else "idle_s")
            result[key] += t - last
        decode += d
        waiting += w
        last = t
    return result


def native_widths(batch):
    steps = batch.get("stepped_transitions", 0)
    if not steps:
        return [0, 0, 0]
    shared = batch["batched_transitions"]
    one = steps - shared
    three = round(steps * batch["width_mean"]) - one - 2 * shared
    two = shared - three
    if min(one, two, three) < 0 or batch["width_max"] > 3:
        raise ValueError("width census requires at most three agents")
    return [one, two, three]


class Arm:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.plan = json.loads(args.plan.read_text())
        self.server = None
        self.stop = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.rows = []
        self.samples = []
        self.lines = []
        self.deadline = time.monotonic() + args.timeout

    def emit(self, **record):
        with self.lock:
            print(json.dumps(record), flush=True)
            with (self.out / "client.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")

    def check(self):
        if self.stop.is_set() or time.monotonic() > self.deadline:
            raise RuntimeError("arm stopped (thermal, timeout, signal or request failure)")
        if self.server is not None and self.server.poll() is not None:
            raise RuntimeError(f"server exited: {self.server.returncode}")

    def memory(self):
        raw = subprocess.check_output(["vm_stat"], text=True)
        page = int(re.search(r"page size of (\d+)", raw)[1])
        values = {a: int(b) for a, b in re.findall(r"^([^:\n]+):\s+(\d+)", raw, re.M)}
        available = (values.get("Pages free", 0) + max(0, values.get("Pages inactive", 0) - values.get("Anonymous pages", 0))) * page
        rss = subprocess.check_output(["ps", "-o", "rss=", "-p", str(self.server.pid)], text=True).strip()
        return {"host_conservative_available": available, "rss": int(rss or 0) * 1024}

    def watch(self):
        try:
            while not self.done.is_set():
                row = {"time": time.time(), "thermal": pressure(), **self.memory()}
                self.samples.append(row)
                if row["thermal"] or time.monotonic() > self.deadline:
                    self.emit(event="stop", sample=row)
                    self.stop.set()
                    if self.server.poll() is None:
                        self.server.send_signal(signal.SIGINT)
                    return
                self.done.wait(1)
        except Exception as error:
            if not self.done.is_set() and self.server.poll() is None:
                self.emit(event="monitor_error", error=str(error))
                self.stop.set()
                self.server.send_signal(signal.SIGINT)

    def readlog(self):
        with (self.out / "server.log").open("w") as f, (self.out / "server-lines.jsonl").open("w") as j:
            for line in self.server.stdout:
                record = {"time": time.time(), "line": line.rstrip()}
                self.lines.append(record)
                f.write(line)
                f.flush()
                j.write(json.dumps(record) + "\n")
                j.flush()

    def request(self, body, label, warm=False):
        self.check()
        path = self.out / f"{label}.request.json"
        dump(path, body)
        sent = time.time()
        first = None
        final = None
        events = []
        connection = http.client.HTTPConnection("127.0.0.1", self.args.port, timeout=200)
        try:
            connection.request("POST", "/v1/responses", body=path.read_bytes(), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            headers = time.time()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {response.read().decode()[:1500]}")
            while True:
                self.check()
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    break
                event = json.loads(raw)
                now = time.time()
                events.append({"time": now, "data": event})
                if event.get("type") in DELTA_TYPES and event.get("delta") and first is None:
                    first = now
                if event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                    final = event["response"]
        finally:
            connection.close()
        end = time.time()
        dump(self.out / f"{label}.events.json", events)
        dump(self.out / f"{label}.response.json", final)
        if not final or final.get("error"):
            self.stop.set()
            raise RuntimeError(f"invalid terminal response {label}: {str(final)[:1000]}")
        usage = final["usage"]
        prompt = usage["input_tokens"]
        cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        deltas = [e["time"] for e in events if e["data"].get("type") in DELTA_TYPES and e["data"].get("delta")]
        # A stop token may have no visible delta; mark the missing observation.
        first = first if first is not None else end
        row = {"label": label, "warm": warm, "sent": sent, "headers": headers,
               "first": first, "end": end, "prompt": prompt, "new": prompt - cached,
               "reused": cached, "generated": usage["output_tokens"],
               "requested_output": body["max_output_tokens"], "ttft_s": first - sent,
               "wall_s": end - sent, "stream_decode_tps": max(0, usage["output_tokens"] - 1) / max(.000001, end - first),
               "delta_gaps_s": [b - a for a, b in zip(deltas, deltas[1:])],
               "delta_times": deltas, "request_sha256": digest(path), "x_qwen": final.get("x_qwen", {})}
        self.rows.append(row)
        self.emit(event="turn", **{k: v for k, v in row.items() if k not in ("x_qwen", "delta_times", "delta_gaps_s")})
        return final

    def prepare(self):
        args = self.args
        if args.reference:
            for agent in range(3):
                for name in [f"a{agent}-warm"] + [f"a{agent}-t{i}" for i in range(len(self.plan["agents"][agent]))]:
                    path = args.reference / f"{name}.request.json"
                    if not path.exists():
                        raise ValueError(f"incomplete reference: {path}")
            return None, None
        corpus = args.corpus.read_text().replace("<|", "‹|").replace("<｜", "‹｜")
        ids = token_text(args.tokenizer, args.model, text=corpus)
        if len(ids) < 3 * args.warm_tokens:
            raise ValueError("corpus needs three distinct warm contexts")
        warm = [token_text(args.tokenizer, args.model, ids=ids[a * args.warm_tokens:(a + 1) * args.warm_tokens]) for a in range(3)]
        suffixes = {}
        offset = 0
        for agent, turns in enumerate(self.plan["agents"]):
            for turn, spec in enumerate(turns):
                # Content count is empirical. Chat wrappers are measured separately.
                n = spec["append_tokens"]
                if offset + n > len(ids):
                    offset = 0
                suffixes[(agent, turn)] = token_text(args.tokenizer, args.model, ids=ids[offset:offset + n])
                offset += n
        return warm, suffixes

    def body(self, history, tokens):
        body = {"model": self.model_id, "input": copy.deepcopy(history), "instructions": INSTRUCTIONS,
                "stream": True, "max_output_tokens": tokens, "temperature": 0,
                "top_p": 1, "x_qwen": {"stats": True, "seed": self.plan["seed"]},
                "tools": [{"type": "function", "name": "read_file", "description": "Read source",
                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                          "required": ["path"], "additionalProperties": False}}]}
        if self.plan["family"] in ("glm", "fn"):
            body["reasoning"] = {"effort": "low"}
        return body

    def run(self):
        args = self.args
        fd = int(os.environ["QWEN_METAL_LEASE_FD"])
        os.fstat(fd)
        warm, suffixes = self.prepare()  # CPU tokenization is outside server timing.
        dump(self.out / "plan.json", self.plan)
        time.sleep(25)  # Cooling gap, including when an unrelated owner just exited.
        for _ in range(3):
            if pressure():
                raise RuntimeError("nonzero thermal pressure before arm")
            time.sleep(5)
        check = subprocess.run(["lsof", "-nP", f"-iTCP:{args.port}", "-sTCP:LISTEN"], stdout=subprocess.DEVNULL)
        if check.returncode != 1:
            raise RuntimeError("loopback port occupied or lsof failed")
        env = dict(os.environ)
        for key in ("QWEN_SERVE_V41_MAX_ACTIVE", "QWEN_SERVE_GLM_MAX_ACTIVE"):
            env.pop(key, None)
        if args.arm == "serial" and self.plan["family"] in ("v41", "glm"):
            env["QWEN_SERVE_V41_MAX_ACTIVE" if self.plan["family"] == "v41" else "QWEN_SERVE_GLM_MAX_ACTIVE"] = "1"
        env.update(RUST_LOG="warn,qwen_diag=info", NO_COLOR="1")
        command = [str(args.binary.resolve()), "serve", "-m", str(args.model.resolve()),
                   "--addr", f"127.0.0.1:{args.port}", "--max-context-tokens", str(args.context),
                   "--max-tokens", str(max(t["reply_tokens"] for a in self.plan["agents"] for t in a))]
        identity = {"source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "binary_sha256": digest(args.binary), "harness_sha256": digest(Path(__file__)),
                    "command": command, "env": {k: v for k, v in env.items() if k.startswith("QWEN_")},
                    "reference": str(args.reference) if args.reference else None,
                    "history": args.history}
        dump(self.out / "launch.json", identity)
        started = time.time()
        self.server = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, pass_fds=(fd,))
        reader = threading.Thread(target=self.readlog)
        watcher = threading.Thread(target=self.watch)
        reader.start()
        watcher.start()
        try:
            while time.time() - started < 180:
                self.check()
                try:
                    c = http.client.HTTPConnection("127.0.0.1", args.port, timeout=1)
                    c.request("GET", "/v1/models")
                    response = c.getresponse()
                    data = json.loads(response.read())
                    c.close()
                    if response.status == 200:
                        self.model_id = data["data"][0]["id"]
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError("server startup")
            ready = time.time()
            histories = []
            for a in range(3):
                if args.reference:
                    body = json.loads((args.reference / f"a{a}-warm.request.json").read_text())
                    if body["model"] != self.model_id:
                        raise ValueError("reference model ID mismatch")
                else:
                    body = self.body([{"role": "user", "content": f"Independent agent {a}. Source corpus follows.\n" + warm[a]}], 1)
                final = self.request(body, f"a{a}-warm", warm=True)
                histories.append(body["input"] + history_items(final.get("output", [])))
            warmed = time.time()
            self.emit(event="warmed", startup_s=ready - started, warmup_s=warmed - ready,
                      thermal=pressure())
            if pressure():
                raise RuntimeError("nonzero thermal after warm-up")
            barrier = threading.Barrier(3)

            def agent(a):
                history = histories[a]
                barrier.wait(timeout=10)
                for turn, spec in enumerate(self.plan["agents"][a]):
                    self.check()
                    label = f"a{a}-t{turn}"
                    if args.reference:
                        body = json.loads((args.reference / f"{label}.request.json").read_text())
                        if args.history == "own":
                            history.extend(copy.deepcopy(body["input"][-2:]))
                            body["input"] = copy.deepcopy(history)
                    else:
                        call = f"agent{a}_read{turn}"
                        history.extend([
                            {"type": "function_call", "call_id": call, "name": "read_file",
                             "arguments": json.dumps({"path": f"src/agent{a}/excerpt{turn}.rs"})},
                            {"type": "function_call_output", "call_id": call, "output": suffixes[(a, turn)]},
                        ])
                        body = self.body(history, spec["reply_tokens"])
                    final = self.request(body, label)
                    history += history_items(final.get("output", []))
                    if turn + 1 < len(self.plan["agents"][a]):
                        if self.stop.wait(spec["tool_gap_s"]):
                            self.check()

            begin = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(agent, a) for a in range(3)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except BaseException:
                        self.stop.set()
                        if self.server.poll() is None:
                            self.server.send_signal(signal.SIGINT)
                        raise
            end = time.time()
            self.check()
            summary = {"family": self.plan["family"], "arm": args.arm, "begin": begin, "end": end,
                       "wall_s": end - begin, "startup_s": ready - started, "warmup_s": warmed - ready,
                       "thermal_after": pressure(), "rows": sorted(self.rows, key=lambda r: r["label"])}
            if summary["thermal_after"]:
                raise RuntimeError("thermal pressure at arm end")
            dump(self.out / "summary.json", summary)
            self.emit(event="arm_done", wall_s=end - begin, warmup_s=warmed - ready)
        finally:
            self.stop.set()
            if self.server.poll() is None:
                self.server.send_signal(signal.SIGINT)
            try:
                self.server.wait(timeout=40)
            except subprocess.TimeoutExpired:
                self.server.terminate()
                self.server.wait(timeout=10)
            self.done.set()
            watcher.join(timeout=5)
            reader.join(timeout=5)
            dump(self.out / "memory-thermal.json", self.samples)
            self.emit(event="server_stopped", code=self.server.returncode)


def analyze(directory):
    summary = json.loads((directory / "summary.json").read_text())
    launch = json.loads((directory / "launch.json").read_text())
    rows = [r for r in summary["rows"] if not r["warm"]]
    native = []
    phases = {}
    for line in (directory / "server-lines.jsonl").read_text().splitlines():
        record = json.loads(line)
        if "serve phases:" in record["line"]:
            phases = kv(record["line"])
        if "serve stats:" in record["line"]:
            native.append({"time": record["time"], "phases": phases, "stats": kv(record["line"])})
            phases = {}
    used = set()
    for row in summary["rows"]:
        matches = [(abs(n["time"] - row["end"]), i, n) for i, n in enumerate(native)
                   if i not in used and int(n["stats"]["prompt_tokens"]) == row["prompt"]
                   and int(n["stats"]["generated_tokens"]) == row["generated"]]
        if not matches:
            raise ValueError(f"unmatched native timings: {row['label']}")
        _, i, n = min(matches)
        used.add(i)
        p = row["x_qwen"].get("phases") or n["phases"]
        row["phases"] = p
        row["native_decode_tps"] = float(n["stats"]["decode_tps"])
        batch = p.get("batch", {})
        row["native_width_transitions"] = native_widths(batch) if batch else [max(0, row["generated"] - 1), 0, 0]
        # Header delay is the serial proxy; native queue_ms is owner-only.
        native_ms = float(p.get("prefill_ms", 0)) + float(p.get("decode_ms", 0)) + float(p.get("restore_ms", 0))
        row["queue_proxy_s"] = batch.get("queue_ms", max(0, row["wall_s"] * 1000 - native_ms)) / 1000
        row["queue_kind"] = "owner_submission_to_admission" if batch else "serial_wall_minus_logged_phases_proxy"
        row["reuse_source"] = p.get("reuse_source", "")
        row["restore_origin"] = p.get("restore_origin", "")
        row["prefill_wall_s"] = float(p.get("prefill_ms", 0)) / 1000
        row["decode_wall_s"] = float(p.get("decode_ms", 0)) / 1000
        row["header_delay_s"] = row["headers"] - row["sent"]
    samples = json.loads((directory / "memory-thermal.json").read_text())
    if any(s["thermal"] for s in samples):
        raise ValueError("nonzero thermal sample")
    result = {k: v for k, v in summary.items() if k != "rows"}
    result["plan_sha256"] = digest(directory / "plan.json")
    result["history"] = launch.get("history", "legacy_frozen" if launch.get("reference") else "legacy_record")
    result.update(generated=sum(r["generated"] for r in rows),
                  new=sum(r["new"] for r in rows), reused=sum(r["reused"] for r in rows),
                  aggregate_tps=sum(r["generated"] for r in rows) / summary["wall_s"],
                  intervals=interval_buckets(rows, summary["begin"], summary["end"]),
                  native_width_transitions=[sum(r["native_width_transitions"][i] for r in rows) for i in range(3)],
                  queue_request_seconds=sum(r["queue_proxy_s"] for r in rows),
                  min_host_available=min(s["host_conservative_available"] for s in samples),
                  max_rss=max(s["rss"] for s in samples), thermal_max=max(s["thermal"] for s in samples),
                  output_shortfalls=[r["label"] for r in rows if r["generated"] != r["requested_output"]],
                  rows=rows)
    bins = []
    for start in range(0, math.ceil(summary["wall_s"]), 20):
        values = [b - a for r in rows for a, b in zip(r["delta_times"], r["delta_times"][1:])
                  if start <= b - summary["begin"] < start + 20]
        lo, hi = summary["begin"] + start, min(summary["end"], summary["begin"] + start + 20)
        telemetry = [s for s in samples if lo <= s["time"] < hi]
        bins.append({"start_s": start, "duration_s": hi - lo, "chunks": len(values),
                     "gap_p50_s": percentile(values, .5), "gap_p95_s": percentile(values, .95),
                     "gap_max_s": max(values, default=None), **interval_buckets(rows, lo, hi),
                     "thermal_max": max((s["thermal"] for s in telemetry), default=None),
                     "min_host_available": min((s["host_conservative_available"] for s in telemetry), default=None)})
    result["trace_20s"] = bins
    result["agents"] = []
    for agent in range(3):
        selected = [r for r in rows if r["label"].startswith(f"a{agent}-")]
        gaps = [g for r in selected for g in r["delta_gaps_s"]]
        result["agents"].append({
            "agent": agent, "calls": len(selected),
            "generated": sum(r["generated"] for r in selected),
            "ttft_p50_s": percentile([r["ttft_s"] for r in selected], .5),
            "ttft_p95_s": percentile([r["ttft_s"] for r in selected], .95),
            "stream_decode_tps": sum(max(0, r["generated"] - 1) for r in selected) /
                                 max(.000001, sum(r["end"] - r["first"] for r in selected)),
            "native_decode_tps": sum(r["generated"] for r in selected) /
                                 max(.000001, sum(r["decode_wall_s"] for r in selected)),
            "gap_p95_s": percentile(gaps, .95), "gap_max_s": max(gaps, default=None),
            "queue_request_seconds": sum(r["queue_proxy_s"] for r in selected),
            "header_delay_seconds": sum(r["header_delay_s"] for r in selected),
        })
    with (directory / "requests.csv").open("w", newline="") as f:
        fields = ["label", "prompt", "new", "reused", "generated", "requested_output",
                  "ttft_s", "wall_s", "stream_decode_tps", "native_decode_tps", "queue_proxy_s",
                  "queue_kind", "reuse_source", "restore_origin", "prefill_wall_s", "decode_wall_s", "header_delay_s"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "trace-20s.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bins[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(bins)
    dump(directory / "analysis.json", result)
    return result


def compare(reference, candidate, own_history=False):
    a, b = analyze(reference), analyze(candidate)
    x = {r["label"]: r for r in a["rows"]}
    y = {r["label"]: r for r in b["rows"]}
    if x.keys() != y.keys():
        raise ValueError("different request sets")
    errors = []
    prompt_deltas = []
    if a.get("plan_sha256") != b.get("plan_sha256"):
        errors.append("different seeded plans or tool gaps")
    if own_history and b.get("history") != "own":
        errors.append("candidate did not run with its own history")
    if own_history:
        for agent in range(3):
            name = f"a{agent}-warm.request.json"
            if digest(reference / name) != digest(candidate / name):
                errors.append(f"different warm root {agent}")
    for label in x:
        if own_history:
            lhs = json.loads((reference / f"{label}.request.json").read_text())
            rhs = json.loads((candidate / f"{label}.request.json").read_text())
            # Batched rounding can change generated history. Fresh tool input
            # bytes and every generation setting must remain identical.
            lhs["input"], rhs["input"] = lhs["input"][-2:], rhs["input"][-2:]
            if lhs != rhs:
                errors.append(f"{label}: unequal new input or generation settings")
            if x[label]["prompt"] != y[label]["prompt"]:
                prompt_deltas.append({"label": label, "tokens": y[label]["prompt"] - x[label]["prompt"]})
        for field in (("generated",) if own_history else ("request_sha256", "prompt", "generated")):
            if x[label][field] != y[label][field]:
                errors.append(f"{label}: unequal {field}: {x[label][field]} != {y[label][field]}")
    # Differences in *processed* rows are intentionally surfaced: cache effects
    # are a result, not something to hide with a denominator adjustment.
    if errors:
        raise ValueError("unmatched work; no speed ratio:\n" + "\n".join(errors))
    return {"reference_wall_s": a["wall_s"], "candidate_wall_s": b["wall_s"],
            "speed_ratio": a["wall_s"] / b["wall_s"], "generated": a["generated"],
            "reference_new": a["new"], "candidate_new": b["new"],
            "identical_new_inputs_and_generated_counts": True,
            "history": "own" if own_history else "frozen", "full_prompt_count_deltas": prompt_deltas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--calls-csv", type=Path, default=DEFAULT_CALLS)
    plan.add_argument("--family", choices=("v41", "glm", "fn"), required=True)
    plan.add_argument("--seed", type=int, default=20261108)
    plan.add_argument("--seconds", type=float, default=240)
    plan.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run")
    for name in ("binary", "model", "plan", "out"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--tokenizer", type=Path)
    run.add_argument("--corpus", type=Path)
    run.add_argument("--reference", type=Path)
    run.add_argument("--history", choices=("own", "frozen"), default="own",
                     help="own is the agent protocol; frozen is a canonical-history diagnostic")
    run.add_argument("--arm", choices=("serial", "default"), required=True)
    run.add_argument("--warm-tokens", type=int, default=48000)
    run.add_argument("--context", type=int, default=98304)
    run.add_argument("--port", type=int, default=18737)
    run.add_argument("--timeout", type=float, default=780)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("directory", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--history", choices=("own", "frozen"), default="own",
                            help="own compares fresh inputs/output counts; frozen also requires identical full requests")
    args = parser.parse_args()
    if args.command == "plan":
        if args.seconds <= 0:
            parser.error("--seconds must be positive")
        with args.calls_csv.open() as f:
            value = make_plan(list(csv.DictReader(f)), args.family, args.seed, args.seconds)
        value["calls_sha256"] = digest(args.calls_csv)
        dump(args.out, value)
        print(json.dumps(value))
    elif args.command == "run":
        if not args.reference and (not args.corpus or not args.tokenizer):
            parser.error("recording requires --corpus and --tokenizer")
        arm = Arm(args)

        def stop(signum, frame):
            arm.stop.set()
            if arm.server is not None and arm.server.poll() is None:
                arm.server.send_signal(signal.SIGINT)
            raise KeyboardInterrupt

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop)
        arm.run()
    elif args.command == "analyze":
        print(json.dumps({k: v for k, v in analyze(args.directory).items() if k not in ("rows", "trace_20s")}))
    else:
        print(json.dumps(compare(args.reference, args.candidate, args.history == "own")))


if __name__ == "__main__":
    main()
