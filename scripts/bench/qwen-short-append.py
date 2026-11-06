#!/usr/bin/env python3
"""Short-append ABBA / pinned comparator, one bounded campaign per invocation.

Build first. Run under lockf build.lock and with-metal-lease --for 10m.
Example: qwen-short-append.py qwen --model MODEL --pp 512 --depth 32768
The qwen comparison changes only QWEN_SHORT_APPEND_TENSOR=0 versus unset.
Prints every native benchmark row, plus source/binary identity and thermals.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from llama_cpp import load_lock, resolve_tool, validate_probe_row


def emit(**record):
    print(json.dumps(record), flush=True)


def thermal(arm, where):
    raw = subprocess.check_output(
        ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], text=True
    ).strip()
    emit(record="thermal", arm=arm, where=where, value=raw)
    if raw.split()[-1] != "0":
        raise RuntimeError("nonzero thermal pressure; stop without retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["qwen", "llama"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pp", default="512")
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    fd = int(os.environ["QWEN_METAL_LEASE_FD"])
    os.fstat(fd)
    child = None
    expired = False

    def stop(sig=None, frame=None):
        nonlocal expired
        expired = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
        if sig is not None:
            raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    timer = threading.Timer(570, stop)
    timer.start()
    try:
        lock = load_lock()
        binary = (
            Path("target/release/qwen-bench")
            if args.engine == "qwen"
            else resolve_tool("llama-bench", lock=lock)[0]
        )
        emit(
            record="source",
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            engine=args.engine,
            model=args.model.name,
            pp=args.pp,
            depth=args.depth,
            runs=args.runs,
        )
        arms = [False, True, True, False] if args.engine == "qwen" else [True]
        for index, on in enumerate(arms):
            time.sleep(30)
            if expired:
                raise TimeoutError("campaign exceeded 570 seconds")
            thermal(index, "before")
            env = dict(os.environ)
            env.pop("QWEN_SHORT_APPEND_TENSOR", None)
            if not on:
                env["QWEN_SHORT_APPEND_TENSOR"] = "0"
            if args.engine == "qwen":
                cmd = [str(binary), "suite", "-m", str(args.model), "--pp", args.pp,
                       "--depth", str(args.depth), "--runs", str(args.runs),
                       "--seed", "1", "-o", "json"]
            else:
                cmd = [str(binary), "-m", str(args.model), "-p", args.pp, "-n", "0",
                       "-d", str(args.depth), "-b", "2048", "-ub", "2048",
                       "-ctk", "f16", "-ctv", "f16", "-r", str(args.runs), "-o", "json"]
            emit(record="arm", arm=index, candidate=on, command=cmd)
            child = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True, pass_fds=(fd,))
            stdout, _ = child.communicate()
            thermal(index, "after")
            if child.returncode or expired:
                emit(record="failed_stdout", stdout=stdout)
                raise RuntimeError(f"arm failed: code={child.returncode}, expired={expired}")
            rows = json.loads(stdout)
            for row in rows:
                if args.engine == "llama":
                    error = validate_probe_row(row, lock)
                    if error:
                        raise RuntimeError(error)
                else:
                    if row.get("build_dirty"):
                        raise RuntimeError("dirty benchmark build")
                emit(record="sample", arm=index, candidate=on, row=row)
    finally:
        timer.cancel()
        timer.join()
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            child.wait()


if __name__ == "__main__":
    main()
