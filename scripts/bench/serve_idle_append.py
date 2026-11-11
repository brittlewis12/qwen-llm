#!/usr/bin/env python3
"""Measure first-append TTFT after an idle gap in a real `qwen serve`.

One 48K warm root, then for each idle length: refresh the root (reopening the
idle-residency window), sleep exactly that long, and send a small append that
reuses the 48K prefix. Reports TTFT and the server's own prefill/reuse stats.

Run under scripts/bench/with-metal-lease; the server inherits the lease fd.

Usage:
  serve_idle_append.py --binary B --tokenizer T --model M --corpus C \
      --out DIR --idle 15,45,90 --keepalive 60 --port 18751
"""

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DELTA_TYPES = {
    "response.output_text.delta",
    "response.reasoning_text.delta",
    "response.function_call_arguments.delta",
}
INSTRUCTIONS = "You are reviewing a Rust/Metal inference engine. Be concise."
WARM_TOKENS = 48 * 1024


def emit(value):
    print(json.dumps(value), flush=True)


def pressure():
    try:
        return int(
            subprocess.check_output(
                ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], text=True
            ).split()[-1]
        )
    except (OSError, ValueError):
        return -1


def token_ids(binary, model, text):
    cmd = [str(binary), "-m", str(model), "--file", "-", "--ids"]
    out = subprocess.check_output(cmd, input=text.encode()).decode().splitlines()
    return [int(x) for x in out[1:]]


def decode_ids(binary, model, ids):
    cmd = [str(binary), "-m", str(model), "--decode-ids", "-"]
    return subprocess.check_output(cmd, input=" ".join(map(str, ids)).encode()).decode()


def post(port, body, timeout=600):
    payload = json.dumps(body, ensure_ascii=False).encode()
    sent = time.time()
    first = None
    final = None
    events = 0
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(
            "POST",
            "/v1/responses",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read()[:1000]!r}")
        while True:
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
            if event.get("type") in DELTA_TYPES and event.get("delta"):
                events += 1
                if first is None:
                    first = now
            if event.get("type") in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                final = event["response"]
    finally:
        conn.close()
    ended = time.time()
    usage = (final or {}).get("usage", {})
    stats = (final or {}).get("x_qwen", {})
    return dict(
        sent=sent,
        first=first,
        end=ended,
        ttft_s=(first - sent) if first else None,
        completed=bool(final and not (final or {}).get("error")),
        events=events,
        generated=usage.get("output_tokens"),
        prompt=usage.get("input_tokens"),
        reused=(usage.get("input_tokens_details") or {}).get("cached_tokens"),
        x_qwen=stats,
    )


def body(model_id, history, count, seed):
    return dict(
        model=model_id,
        input=history,
        instructions=INSTRUCTIONS,
        stream=True,
        max_output_tokens=count,
        reasoning=dict(effort="low"),
        x_qwen=dict(stats=True, seed=seed),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--idle", default="15,45,90")
    parser.add_argument("--keepalive", type=int, default=60)
    parser.add_argument("--port", type=int, default=18751)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fd = int(os.environ["QWEN_METAL_LEASE_FD"])
    os.fstat(fd)

    text = args.corpus.read_text()
    ids = token_ids(args.tokenizer, args.model, text)
    if len(ids) < WARM_TOKENS + 4096:
        raise RuntimeError(f"corpus has {len(ids)} tokens, need {WARM_TOKENS + 4096}")
    root = decode_ids(args.tokenizer, args.model, ids[:WARM_TOKENS])
    # A distinct small suffix the append adds after the 48K root.
    suffix_ids = ids[WARM_TOKENS + 1024 : WARM_TOKENS + 1056]
    suffix = decode_ids(args.tokenizer, args.model, suffix_ids)
    emit(
        dict(
            kind="prepared",
            corpus_tokens=len(ids),
            warm_tokens=WARM_TOKENS,
            suffix_tokens=len(suffix_ids),
            root_chars=len(root),
            keepalive_s=args.keepalive,
            idle=[float(x) for x in args.idle.split(",")],
        )
    )

    env = dict(os.environ, RUST_LOG="warn,qwen_diag=info", NO_COLOR="1")
    env["QWEN_SERVE_IDLE_RESIDENCY_SECS"] = str(args.keepalive)
    command = [
        str(args.binary.resolve()),
        "serve",
        "-m",
        str(args.model.resolve()),
        "--addr",
        f"127.0.0.1:{args.port}",
        "--max-context-tokens",
        str(args.context),
        "--max-tokens",
        "64",
    ]
    log = (args.out / "server.log").open("w")
    server = subprocess.Popen(
        command,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        pass_fds=(fd,),
    )
    started = time.time()
    try:
        model_id = None
        while time.time() - started < 240:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}")
            try:
                conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=2)
                conn.request("GET", "/v1/models")
                response = conn.getresponse()
                data = json.loads(response.read())
                conn.close()
                if response.status == 200:
                    model_id = data["data"][0]["id"]
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.25)
        if model_id is None:
            raise TimeoutError("server startup")
        emit(dict(kind="ready", startup_s=time.time() - started, model=model_id))

        warm_history = [dict(role="user", content=root)]
        warm = post(args.port, body(model_id, warm_history, 4, args.seed))
        emit(
            dict(
                kind="warm",
                **{k: warm[k] for k in ("ttft_s", "prompt", "reused", "completed")},
                thermal=pressure(),
            )
        )
        if not warm["completed"]:
            raise RuntimeError("warm request failed")
        # Pin the exact prefix the append reuses; a small assistant turn plus a
        # tool-style user suffix is what a real agent append looks like.
        append_history = [
            dict(role="user", content=root),
            dict(role="assistant", content="Understood."),
            dict(role="user", content="Continue with this excerpt:\n" + suffix),
        ]

        for idle_s in [float(x) for x in args.idle.split(",")]:
            while pressure() > 1:
                time.sleep(2)
            # Refresh: re-send the root so the window opens on a completed
            # compute request, then idle. With residency 0 this only re-warms.
            refresh = post(args.port, body(model_id, warm_history, 1, args.seed))
            if not refresh["completed"]:
                raise RuntimeError("refresh request failed")
            finished = time.time()
            emit(
                dict(
                    kind="refresh",
                    idle_s=idle_s,
                    ttft_s=refresh["ttft_s"],
                    reused=refresh["reused"],
                    thermal=pressure(),
                )
            )
            time.sleep(idle_s)
            append = post(args.port, body(model_id, append_history, 8, args.seed))
            emit(
                dict(
                    kind="append",
                    idle_s=idle_s,
                    keepalive_s=args.keepalive,
                    observed_idle_s=append["sent"] - finished,
                    ttft_s=append["ttft_s"],
                    prompt=append["prompt"],
                    reused=append["reused"],
                    completed=append["completed"],
                    x_qwen=append["x_qwen"],
                    thermal=pressure(),
                    pressure=pressure(),
                )
            )
            time.sleep(1.0)
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.terminate()
            server.wait(timeout=10)
        log.close()
        emit(dict(kind="server_stopped", code=server.returncode))


if __name__ == "__main__":
    sys.exit(main())
