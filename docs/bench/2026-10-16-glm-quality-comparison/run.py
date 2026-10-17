#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Resident native evaluator. Raw server logs are PRIVATE, not publication artifacts."""

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
from contextlib import contextmanager

from cases import check_answer

OVERRIDES = {
    "QWEN_GLM_PREFIX_REUSE": "0",
    "QWEN_GLM_FAST_SNAPSHOTS": "0",
    "QWEN_GLM_FAST_PRECISION": "half",
    "QWEN_METAL_LEASE_WAIT": "0",
}
MODEL_OUTPUT_ERRORS = frozenset(
    {
        "invalid GLM-5.3-Flash model output: stop before reasoning close",
        "invalid GLM-5.3-Flash model output: unsupported stop token",
    }
)


class InvalidRun(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidRun(message)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
    tmp.replace(path)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stamp(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def integrity(binary, model, manifest):
    entries = manifest.get("shards")
    require(isinstance(entries, list) and entries, "missing shards")
    files = [(binary, manifest.get("binary_sha256"), None)]
    names = set()
    for entry in entries:
        name = entry.get("name")
        require(
            isinstance(name, str) and Path(name).name == name and name not in names,
            "invalid or duplicate shard name",
        )
        names.add(name)
        path = Path(entry.get("path", name))
        if not path.is_absolute():
            path = model.parent / path
        require(
            path.resolve() == (model.parent / name).resolve(), "shard path mismatch"
        )
        require(
            type(entry.get("size")) is int and entry["size"] > 0, "invalid shard size"
        )
        files.append((model.parent / name, entry.get("sha256"), entry["size"]))
    require(model.name in names, "model absent from manifest")
    # GGUF split loaders discover siblings themselves: reject omitted/extra shards.
    split = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", model.name)
    if split:
        prefix, first, count = split.groups()
        require(int(first) == 1, "model must be first shard")
        expected = {
            f"{prefix}-{i:05d}-of-{count}.gguf" for i in range(1, int(count) + 1)
        }
        require(names == expected, "incomplete split manifest")
    else:
        require(names == {model.name}, "unexpected shards for unsplit model")
    stamps, public = {}, []
    for path, sha, size in files:
        require(
            isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha),
            "invalid sha256",
        )
        target = path.resolve()
        before = stamp(target)
        require(size is None or before[2] == size, "size mismatch")
        require(
            digest(target) == sha and stamp(target) == before and stamp(path) == before,
            "hash mismatch or changing file",
        )
        stamps[path] = before
        stamps[target] = before
        public.append({"name": path.name, "size": before[2], "sha256": sha})
    return stamps, public


def guard(stamps):
    require(
        all(stamp(p) == s for p, s in stamps.items()), "integrity changed during run"
    )


def environment():
    require(
        not any(k.startswith("QWEN_") and k not in OVERRIDES for k in os.environ),
        "unknown inherited QWEN_ setting",
    )
    env = dict(os.environ)
    env.update(OVERRIDES)
    env.pop("MTL_DEBUG_LAYER", None)
    return env


@contextmanager
def deadline(seconds):
    # Unix main-thread wall deadline also bounds slow trickle HTTP streams.
    def expired(*_):
        raise TimeoutError("operation deadline exceeded")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def cleanup(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def parse_sse(stream, raw, events, start):
    event, data = "", []
    terminal = None
    done = False
    ttft = visible = None
    while True:
        line = stream.readline()
        if not line:
            require(not data, "unterminated SSE frame")
            break
        raw.write(line)
        raw.flush()
        text = line.decode("utf-8").rstrip("\r\n")
        if text:
            key, _, value = text.partition(":")
            value = value.removeprefix(" ")
            if key == "event":
                event = value
            elif key == "data":
                data.append(value)
            continue
        if not data:
            event = ""
            continue
        elapsed = time.monotonic() - start
        encoded = "\n".join(data)
        payload = "[DONE]" if encoded == "[DONE]" else json.loads(encoded)
        require(isinstance(payload, dict) or payload == "[DONE]", "invalid SSE payload")
        kind = "[DONE]" if encoded == "[DONE]" else event or payload.get("type", "")
        events.write(
            json.dumps({"seconds": elapsed, "event": kind, "data": payload}) + "\n"
        )
        events.flush()
        require(not done, "event after DONE")
        if encoded == "[DONE]":
            require(not event, "named DONE event")
            require(terminal is not None, "premature DONE")
            done = True
            event, data = "", []
            continue
        require(terminal is None, "event after terminal")
        require(payload.get("type", kind) == kind, "SSE event type mismatch")
        require(
            kind not in {"error", "response.error"},
            "server SSE error",
        )
        delta = payload.get("delta")
        if (
            kind in {"response.reasoning.delta", "response.output_text.delta"}
            and isinstance(delta, str)
            and delta
        ):
            if ttft is None:
                ttft = elapsed
            if kind == "response.output_text.delta" and visible is None:
                visible = elapsed
        if kind in {"response.completed", "response.incomplete", "response.failed"}:
            terminal = payload["response"]
            require(
                terminal.get("status") == kind.split(".")[1], "terminal status mismatch"
            )
        event, data = "", []
    require(terminal is not None, "missing terminal response")
    require(done, "missing DONE sentinel")
    return terminal, {
        "client_ttft_s": ttft,
        "first_visible_answer_s": visible,
        "request_elapsed_s": time.monotonic() - start,
    }


def validate_response(response):
    status = response.get("status")
    if status == "failed":
        error = response.get("error", {})
        require(
            isinstance(error, dict)
            and error.get("type") == "server_error"
            and error.get("code") == ""
            and error.get("param") == ""
            and error.get("message") in MODEL_OUTPUT_ERRORS,
            "unclassified response.failed",
        )
        require(
            response.get("usage") is None and "x_qwen" not in response,
            "unexpected failed response accounting",
        )
        answer, reasoning = output_text(response)
        return {
            "status": "model_output_error",
            "wire_status": "failed",
            "error": {key: error[key] for key in ("type", "code", "param", "message")},
            "answer": answer,
            "reasoning": reasoning,
            "text_scope": "wire output only; native failure may omit pending delimiter bytes",
            "usage": None,
            "x_qwen": None,
            "native_stop_reason": None,
            "cache_validation": "HTTP counters unavailable on response.failed",
        }
    require(status in {"completed", "incomplete"}, "invalid terminal status")
    if status == "incomplete":
        require(
            response.get("incomplete_details", {}).get("reason") == "max_output_tokens",
            "unexpected incomplete reason",
        )
    usage = response.get("usage", {})
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        require(type(usage.get(field)) is int and usage[field] >= 0, "invalid usage")
    require(
        usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"],
        "usage sum mismatch",
    )
    cached = usage.get("input_tokens_details", {}).get("cached_tokens")
    matched = response.get("x_qwen", {}).get("matched_tokens")
    require(
        type(cached) is int and cached == 0 and type(matched) is int and matched == 0,
        "missing or nonzero cache counters",
    )
    answer, reasoning = output_text(response)
    return {
        "status": status,
        "native_stop_reason": "eos" if status == "completed" else "token_limit",
        "answer": answer,
        "reasoning": reasoning,
        "usage": usage,
        "x_qwen": response["x_qwen"],
    }


def output_text(response):
    answer, reasoning = [], []
    require(isinstance(response.get("output"), list), "missing output")
    for item in response["output"]:
        for part in item.get("content", []):
            if item.get("type") == "message" and part.get("type") == "output_text":
                answer.append(part["text"])
            if item.get("type") == "reasoning" and part.get("type") == "reasoning_text":
                reasoning.append(part["text"])
    return "".join(answer), "".join(reasoning)


def parse_stats(text, ordinal, result):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    phases = [
        s.split("serve phases:", 1)[1]
        for s in text.splitlines()
        if "serve phases:" in s
    ]
    stats = [
        s.split("serve stats:", 1)[1] for s in text.splitlines() if "serve stats:" in s
    ]
    failed = result["status"] == "model_output_error"
    if failed and len(phases) == len(stats) == ordinal - 1:
        return {
            "prefill_ms": None,
            "decode_ms": None,
            "transitions": None,
            "phases": None,
            "stats": None,
            "unavailable_reason": "no native final stats",
        }
    require(
        len(phases) == ordinal and len(stats) == ordinal, "native log ordinal mismatch"
    )

    def fields(line):
        return dict(re.findall(r"(\w+)=([^\s]+)", line))

    p, s = fields(phases[-1]), fields(stats[-1])
    require(
        p.get("family") == "glm5_next" and p.get("reused_tokens") == "0",
        "native reuse or family mismatch",
    )
    require(
        s.get("version") == "serve_stats_v1" and s.get("matched_tokens") == "0",
        "invalid native stats",
    )
    require(s.get("stop_reason") in {"eos", "token_limit"}, "unknown native stop")
    if not failed:
        require(
            s.get("stop_reason") == result["native_stop_reason"], "native stop mismatch"
        )
        require(
            int(s["prompt_tokens"]) == result["usage"]["input_tokens"]
            and int(s["generated_tokens"]) == result["usage"]["output_tokens"],
            "native usage mismatch",
        )
    measured = {key: float(p[key]) for key in ("prefill_ms", "decode_ms")}
    require(
        all(math.isfinite(v) and v >= 0 for v in measured.values()),
        "invalid native timing",
    )
    measured["transitions"] = int(p["transitions"])
    require(measured["transitions"] >= 0, "invalid transitions")
    return {**measured, "phases": p, "stats": s}


def prefill_plan(text):
    marker = "serve: glm5_next prefill plan "
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [
        (number, line.split(marker, 1)[1])
        for number, line in enumerate(text.splitlines(), 1)
        if marker in line
    ]
    if not lines:
        return None
    # d7b2a05b freezes rows after successful allocation; warmup does not log this.
    require(len(lines) == 1, "multiple frozen prefill plans in one server")
    number, line = lines[0]
    fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
    result = {"stderr_line": number}
    for key in ("rows", "session_bytes", "prefix_budget"):
        value = fields.get(key, "")
        require(re.fullmatch(r"[0-9]+", value), "invalid native prefill plan")
        result[key] = int(value)
    require(
        fields.get("prefix_preference_held") in {"true", "false"},
        "invalid native prefix preference",
    )
    result["prefix_preference_held"] = fields["prefix_preference_held"] == "true"
    return result


def startup_descriptor(text):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [
        line for line in text.splitlines() if "serve limits: family=glm5_next " in line
    ]
    result = {
        "capacity": None,
        "prefill_rows_max": None,
        "fast_precision": None,
        "selected_prefill_rows": None,
        "selected_prefill_rows_observability": "not a startup measurement; see per-outcome prefill_plan",
    }
    if lines:
        require(len(lines) == 1, "ambiguous startup descriptor")
        fields = dict(re.findall(r"(\w+)=([^\s]+)", lines[0]))
        for key in ("capacity", "prefill_rows_max"):
            result[key] = int(fields[key])
            require(result[key] > 0, "invalid startup descriptor")
        require(
            fields.get("fast_precision") in {"half", "float"}, "unknown fast precision"
        )
        result["fast_precision"] = fields["fast_precision"]
    else:
        result["unavailable_reason"] = "startup descriptor missing"
    return result


def request_body(model, case):
    return {
        "model": model,
        "instructions": case["system"],
        "input": [{"role": "user", "content": case["user"]}],
        "stream": True,
        "store": False,
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": case["max_tokens"],
        "reasoning": {"effort": "low"},
        "x_qwen": {
            "seed": 0,
            "top_k": 0,
            "min_p": 0,
            "stats": True,
            "prefill_lineage": "fast",
        },
    }


def memory():
    result = {}
    for name, command in (
        ("vm_stat", ["vm_stat"]),
        ("swapusage", ["sysctl", "vm.swapusage"]),
    ):
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[name] = {"returncode": p.returncode, "stdout": p.stdout}
        except (OSError, subprocess.TimeoutExpired):
            result[name] = {"unavailable": True}
    return result


def free_port(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port or 0))
        return sock.getsockname()[1]


def start_server(binary, model, port, out, env):
    end = time.monotonic() + 600
    for attempt in range(1, 4):
        selected = free_port(port)
        log = out / f"server-{attempt}.stderr.log"
        command = [
            str(binary),
            "serve",
            "--model",
            str(model),
            "--addr",
            f"127.0.0.1:{selected}",
            "--max-context-tokens",
            "32768",
            "--max-tokens",
            "6144",
            "--snapshot-cache-mib",
            "0",
            "--durable-snapshot-dir",
            "off",
            "--idle-residency-secs",
            "60",
        ]
        child = None
        try:
            with (
                log.open("wb") as err,
                (out / f"server-{attempt}.stdout.log").open("wb") as stdout,
            ):
                child = subprocess.Popen(command, stdout=stdout, stderr=err, env=env)
            while time.monotonic() < end:
                text = log.read_text(errors="replace")
                if child.poll() is not None:
                    if not port and (
                        "Address already in use" in text
                        or "address already in use" in text
                    ):
                        break
                    raise InvalidRun("owned server exited during startup")
                # Never probe an unrelated listener that won the ephemeral-port race.
                if f"serve: listening on http://127.0.0.1:{selected} " in text:
                    conn = http.client.HTTPConnection("127.0.0.1", selected, timeout=2)
                    try:
                        conn.request("GET", "/v1/models")
                        response = conn.getresponse()
                        if response.status == 200:
                            models = json.loads(response.read())["data"]
                            require(
                                len(models) == 1 and isinstance(models[0]["id"], str),
                                "invalid models response",
                            )
                            require(child.poll() is None, "server exited at readiness")
                            return child, selected, models[0]["id"], log
                    except (OSError, http.client.HTTPException):
                        pass
                    finally:
                        conn.close()
                time.sleep(0.1)
            else:
                raise TimeoutError("startup deadline exceeded")
        except BaseException:
            cleanup(child)
            raise
        cleanup(child)
    raise InvalidRun("port retry limit exceeded")


def evaluate(port, body, directory):
    atomic_json(directory / "request.json", body)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    start = time.monotonic()
    try:
        with (
            deadline(900),
            (directory / "response.sse").open("wb") as raw,
            (directory / "events.jsonl").open("w") as events,
        ):
            conn.request(
                "POST",
                "/v1/responses",
                json.dumps(body),
                {"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            atomic_json(
                directory / "http.json",
                {"status": response.status, "headers": response.getheaders()},
            )
            if response.status != 200:
                raw.write(response.read())
                raise InvalidRun("HTTP request failed")
            require(
                response.getheader("Content-Type", "").split(";")[0]
                == "text/event-stream",
                "not SSE",
            )
            terminal, timing = parse_sse(response, raw, events, start)
            atomic_json(directory / "response.json", terminal)
            return {**validate_response(terminal), **timing}
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("binary", "model", "out", "cases", "manifest"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    require(args.port is None or 1 <= args.port <= 65535, "invalid port")
    env = environment()
    # Split discovery uses the requested basename, even when weights are symlinks.
    binary, model = args.binary.resolve(), args.model.absolute()
    frozen = args.cases.read_bytes()
    cases = json.loads(frozen)
    require(isinstance(cases, list) and len(cases) == 8, "expected eight frozen cases")
    for c in cases:
        require(
            isinstance(c, dict)
            and isinstance(c.get("id"), str)
            and bool(c["id"].strip())
            and isinstance(c.get("system"), str)
            and isinstance(c.get("user"), str)
            and bool(c["user"].strip())
            and type(c.get("max_tokens")) is int
            and 0 < c["max_tokens"] <= 6144,
            "invalid case",
        )
    require(len({c["id"] for c in cases}) == 8, "duplicate cases")
    args.out.mkdir(mode=0o700, parents=False, exist_ok=False)
    summary = {
        "arm": args.arm,
        "source_pin": "d7b2a05b",
        "source_pin_verified_by": "parent",
        "runner_sha256": digest(Path(__file__)),
        "checker_sha256": digest(Path(__file__).with_name("cases.py")),
        "cases_sha256": hashlib.sha256(frozen).hexdigest(),
        "environment": OVERRIDES,
        "status": "running",
        "results": [],
        "servers": [],
        "comparison_scope": "whole native configuration; not a pure quantization causal comparison",
        "raw_artifacts_private": True,
    }
    child = None
    stamps = None
    try:
        atomic_json(args.out / "summary.json", summary)
        (args.out / "cases.json").write_bytes(frozen)
        stamps, summary["integrity"] = integrity(
            binary, model, json.loads(args.manifest.read_bytes())
        )
        summary["memory_before"] = memory()
        guard(stamps)
        for ordinal, case in enumerate(cases, 1):
            if child is None:
                guard(stamps)
                server_number = len(summary["servers"]) + 1
                server_dir = args.out / f"server-{server_number:02d}"
                server_dir.mkdir()
                server_record = {
                    "server_number": server_number,
                    "before_case": ordinal,
                    "reason": "initial" if server_number == 1 else "model_output_error",
                    "status": "starting",
                    "startup_elapsed_s": None,
                }
                summary["servers"].append(server_record)
                atomic_json(args.out / "summary.json", summary)
                started = time.monotonic()
                try:
                    child, port, model_id, log = start_server(
                        binary, model, args.port, server_dir, env
                    )
                finally:
                    server_record["startup_elapsed_s"] = time.monotonic() - started
                    server_record["status"] = (
                        "ready" if child is not None else "startup_failed"
                    )
                server_record["descriptor"] = startup_descriptor(
                    log.read_text(errors="replace")
                )
                server_ordinal = 0
            require(child.poll() is None, "server exited")
            guard(stamps)
            directory = args.out / f"case-{ordinal:02d}"
            directory.mkdir()
            server_ordinal += 1
            result = {
                "ordinal": ordinal,
                "id": case["id"],
                "status": "running",
                "server_number": server_number,
                "server_request_ordinal": server_ordinal,
            }
            summary["results"].append(result)
            atomic_json(args.out / "summary.json", summary)
            result.update(evaluate(port, request_body(model_id, case), directory))
            native_log = log.read_text(errors="replace")
            plan = prefill_plan(native_log)
            if plan is not None:
                if "prefill_plan" not in server_record:
                    server_record["prefill_plan"] = {
                        **plan,
                        "server_number": server_number,
                        "observed_at_server_request_ordinal": server_ordinal,
                        "stderr_artifact": str(log.relative_to(args.out)),
                    }
                result["prefill_plan"] = server_record["prefill_plan"]
                result["selected_prefill_rows"] = plan["rows"]
                result["selected_prefill_rows_observability"] = (
                    "native post-allocation plan; frozen for this server"
                )
            else:
                result["prefill_plan"] = None
                result["selected_prefill_rows"] = None
                result["selected_prefill_rows_observability"] = (
                    "no plan logged by this server at outcome"
                )
            result["checks"] = check_answer(case, result["answer"])
            # Synchronous native stderr writes precede response.finish's terminal SSE.
            result["native"] = parse_stats(native_log, server_ordinal, result)
            guard(stamps)
            atomic_json(directory / "result.json", result)
            atomic_json(args.out / "summary.json", summary)
            if result["status"] == "model_output_error":
                stopped = time.monotonic()
                cleanup(child)
                child = None
                server_record["cleanup_elapsed_s"] = time.monotonic() - stopped
                server_record["status"] = "stopped_after_model_output_error"
                atomic_json(args.out / "summary.json", summary)
        require(child is None or child.poll() is None, "server exited")
        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "aborted"
        if summary["results"]:
            summary["results"][-1]["validation_error_type"] = type(exc).__name__
            if summary["results"][-1]["status"] == "running":
                summary["results"][-1]["status"] = "error"
        # Exception strings can contain local paths, credentials, or host identifiers.
        summary["error_type"] = type(exc).__name__
        if isinstance(exc, InvalidRun):
            summary["error"] = str(exc)
    except BaseException:
        summary["status"] = "interrupted"
        raise
    finally:
        try:
            cleanup(child)
            if stamps is not None:
                guard(stamps)
                summary["integrity_after"] = "unchanged"
        except Exception as exc:
            summary["status"] = "aborted"
            summary["cleanup_or_integrity_error"] = type(exc).__name__
        summary["memory_after"] = memory()
        atomic_json(args.out / "summary.json", summary)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            f"Evaluator refused input ({type(exc).__name__}); no private paths printed."
        )
        raise SystemExit(1)
