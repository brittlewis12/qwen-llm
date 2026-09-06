# /// script
# requires-python = ">=3.12"
# ///
"""Probe serial Qwen SSE endpoints; retain every event and cooperative teardown.

--requests is a JSON array of {name, request} objects. Optional continue=true
prepends the preceding input and response output to the new user input.
First-delta timing excludes headers/heartbeats; wall ends at [DONE], not EOF.
The text/hash fields concatenate string delta events, not authoritative token IDs.
Output includes prompts and generated text; the new output directory is private.
"""

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--binary", required=True)
p.add_argument("--model", required=True)
p.add_argument("--drafter")
p.add_argument("--off-ctx")
p.add_argument("--requests", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
requests = json.loads(a.requests.read_text())
a.out.mkdir(mode=0o700, parents=True, exist_ok=False)
env = os.environ.copy()
if a.off_ctx is not None:
    env["QWEN_DFLASH_OFF_CTX"] = a.off_ctx
with socket.socket() as reservation:
    reservation.bind(("127.0.0.1", 0))
    port = reservation.getsockname()[1]
command = [
    a.binary,
    "serve",
    "-m",
    a.model,
    "--addr",
    f"127.0.0.1:{port}",
    "--max-tokens",
    "128",
    "--snapshot-cache-mib",
    "4096",
]
if a.drafter:
    command += ["--drafter", a.drafter]


def host():
    return {
        name: subprocess.check_output(cmd, text=True)
        for name, cmd in [
            ("vm_stat", ["vm_stat"]),
            ("thermal", ["pmset", "-g", "therm"]),
            ("power", ["pmset", "-g", "batt"]),
        ]
    }


manifest = {
    "command": command,
    "qwen_env": {k: v for k, v in env.items() if k.startswith("QWEN_")},
    "requests": requests,
    "host_before": host(),
}
(a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
with (a.out / "server.log").open("x") as log:
    child = subprocess.Popen(
        command, env=env, stdout=log, stderr=log, start_new_session=True
    )
    print(f"owned server pid={child.pid}", flush=True)
    try:
        deadline = time.monotonic() + 120
        while True:
            if child.poll() is not None:
                raise RuntimeError(f"server exited {child.returncode}; see server.log")
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            try:
                conn.request("GET", "/v1/models")
                response = conn.getresponse()
                if response.status == 200:
                    model_id = json.loads(response.read())["data"][0]["id"]
                    break
            except (OSError, http.client.HTTPException):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
            finally:
                conn.close()
        previous_input = None
        previous_response = None
        with (a.out / "rows.jsonl").open("x") as rows:
            for index, case in enumerate(requests):
                body = dict(case["request"])
                if case.get("continue"):
                    history = previous_input
                    if isinstance(history, str):
                        history = [
                            {"type": "message", "role": "user", "content": history}
                        ]
                    body["input"] = (
                        history
                        + previous_response["output"]
                        + [
                            {
                                "type": "message",
                                "role": "user",
                                "content": body["input"],
                            }
                        ]
                    )
                body.update(model=model_id, stream=True)
                encoded = json.dumps(body).encode()
                (a.out / f"request-{index}.json").write_bytes(encoded)
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
                started = time.perf_counter()
                conn.request(
                    "POST",
                    "/v1/responses",
                    encoded,
                    {"Content-Type": "application/json"},
                )
                response = conn.getresponse()
                assert response.status == 200, (response.status, response.read())
                first_ms = None
                completed = None
                completed_ms = None
                pieces = []
                with (a.out / f"events-{index}.jsonl").open("x") as events:
                    while line := response.readline():
                        if not line.startswith(b"data: "):
                            continue
                        data = line[6:].strip()
                        if data == b"[DONE]":
                            break
                        event = json.loads(data)
                        elapsed = (time.perf_counter() - started) * 1000
                        events.write(json.dumps({"ms": elapsed, "event": event}) + "\n")
                        if (
                            event.get("type", "").endswith(".delta")
                            and isinstance(event.get("delta"), str)
                            and event["delta"]
                        ):
                            if first_ms is None:
                                first_ms = elapsed
                            pieces.append(event["delta"])
                        if event.get("type") in (
                            "response.completed",
                            "response.incomplete",
                        ):
                            completed = event["response"]
                            completed_ms = elapsed
                        if event.get("type") in ("error", "response.failed"):
                            raise RuntimeError(event)
                wall_ms = (time.perf_counter() - started) * 1000
                conn.close()
                assert completed is not None, "missing terminal response"
                text = "".join(pieces)
                row = {
                    "case": case["name"],
                    "first_delta_ms": first_ms,
                    "completed_ms": completed_ms,
                    "wall_ms": wall_ms,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "text": text,
                    "response": completed,
                }
                rows.write(json.dumps(row) + "\n")
                rows.flush()
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in ("response", "text")}
                    ),
                    flush=True,
                )
                previous_input, previous_response = body["input"], completed
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"owned PID {child.pid} still alive; no forced kill attempted"
                )
        print(f"owned server pid={child.pid} exited={child.returncode}", flush=True)
        (a.out / "host-after.json").write_text(json.dumps(host(), indent=2))
