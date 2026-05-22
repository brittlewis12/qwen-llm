#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch a process, wait for a ready pattern, then attach xctrace."
    )
    p.add_argument("--trace", required=True, help="Output .trace path")
    p.add_argument("--stdout", required=True, help="Captured target stdout path")
    p.add_argument("--stderr", required=True, help="Captured target stderr path")
    p.add_argument(
        "--template", default="Metal System Trace", help="xctrace template name"
    )
    p.add_argument(
        "--time-limit",
        default="45s",
        help="xctrace recording time limit (e.g. 45s, 2m)",
    )
    p.add_argument(
        "--ready-pattern",
        default=r"\[pp\] lowering:",
        help="Regex that must appear in stdout or stderr before attach",
    )
    p.add_argument(
        "--ready-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for the ready pattern before failing",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between ready-pattern polls",
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra environment variable in KEY=VALUE form",
    )
    p.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = p.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        p.error("missing command after --")
    return args


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def build_env(pairs: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"invalid --env {pair!r}; expected KEY=VALUE")
        env[key] = value
    return env


def main() -> int:
    args = parse_args()
    trace = Path(args.trace)
    stdout_path = Path(args.stdout)
    stderr_path = Path(args.stderr)
    trace.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(args.env)
    pattern = re.compile(args.ready_pattern)

    print(
        f"[trace-attach] launching: {' '.join(shlex.quote(x) for x in args.command)}",
        flush=True,
    )
    with stdout_path.open("w") as out_f, stderr_path.open("w") as err_f:
        proc = subprocess.Popen(args.command, stdout=out_f, stderr=err_f, env=env)
        try:
            deadline = time.monotonic() + args.ready_timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                merged = read_text(stdout_path) + "\n" + read_text(stderr_path)
                if pattern.search(merged):
                    print(f"[trace-attach] ready: pid={proc.pid}", flush=True)
                    break
                time.sleep(args.poll_interval)
            else:
                proc.terminate()
                raise SystemExit(
                    f"ready pattern {args.ready_pattern!r} not found within {args.ready_timeout}s"
                )

            if proc.poll() is not None:
                raise SystemExit(
                    f"target exited before attach (code {proc.returncode}); inspect {stdout_path} and {stderr_path}"
                )

            xctrace_cmd = [
                "xcrun",
                "xctrace",
                "record",
                "--no-prompt",
                "--template",
                args.template,
                "--time-limit",
                args.time_limit,
                "--output",
                str(trace),
                "--attach",
                str(proc.pid),
            ]
            print(
                f"[trace-attach] attaching: {' '.join(shlex.quote(x) for x in xctrace_cmd)}",
                flush=True,
            )
            subprocess.run(xctrace_cmd, check=True)
            proc.wait()
            print(
                f"[trace-attach] done: trace={trace} stdout={stdout_path} stderr={stderr_path}",
                flush=True,
            )
            return proc.returncode or 0
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
