# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Map #13: which past reasoning does opencode replay, within one multi-step
tool run and across runs, in-process (`run --local`) and server-attached
(`opencode serve` + `run --attach`)? Runs against
`opencode_tool_mock.py` (no model, no GPU) with isolated opencode
config/data/state directories; the provider package comes from opencode's
own cache, so nothing is installed.

Modes: `local` and `attached` use the provider id `@ai-sdk/open-responses`,
which opencode resolves to its bundled copy; `attached-file-provider` points
the same provider entry at the cached 2.0.29 build through a `file://` npm
path (opencode loads `file://` providers directly; 2.0.29 implements the AI
SDK 6 provider specification, which this opencode may refuse);
`attached-openai` uses opencode's bundled `@ai-sdk/openai` Responses
provider against the same server.

For each mode: a first run whose model calls one tool and then answers,
then a second run continuing the session. Every request body is logged and
summarized: the input item sequence, which reasoning texts it carries, and
whether reasoning items keep their ids and content.

Starts and stops only the processes it launched.

  uv run scripts/reference/opencode_reasoning_replay.py --out <dir>
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port, deadline=30):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise SystemExit(f"port {port} never opened")


CACHED_PROVIDER = (
    Path.home() / ".cache/opencode/node_modules/@ai-sdk/open-responses/dist/index.js"
)


def config(mock_port, npm):
    options = {"url": f"http://127.0.0.1:{mock_port}/v1/responses"}
    if npm == "@ai-sdk/openai":
        options = {"baseURL": f"http://127.0.0.1:{mock_port}/v1", "apiKey": "unused"}
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": "mockserve/mock-model",
        "small_model": "mockserve/mock-model",
        "autoupdate": False,
        "share": "disabled",
        "provider": {
            "mockserve": {
                "npm": npm,
                "name": "mock serve",
                "options": options,
                "models": {
                    "mock-model": {
                        "name": "mock",
                        "tool_call": True,
                        "reasoning": True,
                        "limit": {"context": 32768, "output": 1024},
                    }
                },
            }
        },
    }


def summarize(body):
    items = body.get("input") or []
    sequence, reasoning = [], []
    for item in items:
        kind = item.get("type") or ("message" if "role" in item else "?")
        if kind == "message":
            kind = f"{item.get('role')}"
        sequence.append(kind)
        if item.get("type") == "reasoning":
            texts = [c.get("text") for c in item.get("content") or []]
            reasoning.append(
                {"id": item.get("id"), "content": texts, "summary": item.get("summary")}
            )
    text = json.dumps(items)
    if "Generate a title" in text:
        role = "title"
    elif items and items[-1].get("type") == "function_call_output":
        role = "after_tool"
    else:
        role = "user_turn"
    return {
        "role": role,
        "sequence": sequence,
        "reasoning": reasoning,
        "has_tools": bool(body.get("tools")),
        "store": body.get("store"),
        "include": body.get("include"),
    }


def run_mode(mode, out, opencode):
    root = out / mode
    if root.exists():
        shutil.rmtree(root)
    work = root / "work"
    work.mkdir(parents=True)
    for name in ["a.txt", "b.txt", "c.txt"]:
        (work / name).write_text(f"{name}\n")
    # Nothing from a parent opencode session (its server URL, database or
    # caller ids) reaches the isolated client.
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("OPENCODE")}
    for key, sub in [
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_STATE_HOME", "state"),
    ]:
        env[key] = str(root / sub)
        (root / sub).mkdir()
    env.pop("OPENCODE_CONFIG", None)
    env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
    mock_port = free_port()
    (root / "config" / "opencode").mkdir()
    npm = {
        "attached-file-provider": f"file://{CACHED_PROVIDER}",
        "attached-openai": "@ai-sdk/openai",
    }.get(mode, "@ai-sdk/open-responses")
    (root / "config" / "opencode" / "opencode.json").write_text(
        json.dumps(config(mock_port, npm), indent=2)
    )
    log = root / "requests.jsonl"
    processes = []
    try:
        mock = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "opencode_tool_mock.py"),
                "--port",
                str(mock_port),
                "--log",
                str(log),
                "--dir",
                str(work),
            ],
            stdout=subprocess.DEVNULL,
            stderr=open(root / "mock.err", "w"),
        )
        processes.append(mock)
        wait_port(mock_port)
        base = [opencode, "run", "-m", "mockserve/mock-model", "--format", "json"]
        if mode.startswith("attached"):
            server_port = free_port()
            server = subprocess.Popen(
                [
                    opencode,
                    "serve",
                    "--port",
                    str(server_port),
                    "--hostname",
                    "127.0.0.1",
                ],
                cwd=work,
                env=env,
                stdout=open(root / "serve.out", "w"),
                stderr=subprocess.STDOUT,
            )
            processes.append(server)
            wait_port(server_port)
            base += ["--attach", f"http://127.0.0.1:{server_port}", "--dir", str(work)]
        else:
            base += ["--local", "--dir", str(work)]
        runs = []
        for label, extra in [
            (
                "first",
                ["List the files in this directory, then say how many there are."],
            ),
            ("continued", ["-c", "And now say it again."]),
        ]:
            result = subprocess.run(
                base + extra,
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            (root / f"{label}.stdout").write_text(result.stdout)
            (root / f"{label}.stderr").write_text(result.stderr)
            runs.append({"run": label, "exit": result.returncode})
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)
    requests = (
        [json.loads(line)["body"] for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return {
        "mode": mode,
        "npm": npm,
        "runs": runs,
        "requests": [summarize(r) for r in requests],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--opencode", default=shutil.which("opencode"))
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(
        [args.opencode, "--version"], capture_output=True, text=True
    ).stdout.strip()
    report = {
        "opencode": version,
        "modes": [
            run_mode(m, out, args.opencode)
            for m in ["local", "attached", "attached-file-provider", "attached-openai"]
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
