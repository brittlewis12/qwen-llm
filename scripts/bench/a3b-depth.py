#!/usr/bin/env python3
"""Bound one depth witness to 570 seconds; hold build.lock + Metal lease outside.

Usage: a3b-depth.py LOGFILE COMMAND [ARGS...]
Keeps every stdout/stderr sample in LOGFILE; prints compact progress only.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


def thermal(where):
    raw = subprocess.check_output(
        ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], text=True
    ).strip()
    print(json.dumps(dict(record="thermal", arm="campaign", where=where, value=raw)), flush=True)
    if raw.split()[-1] != "0":
        raise RuntimeError("nonzero thermal pressure; stop without retry")


def main():
    log = Path(sys.argv[1])
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
        thermal("before")
        with log.open("x") as output:
            child = subprocess.Popen(sys.argv[2:], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, pass_fds=(fd,))
            for line in child.stdout:
                output.write(line)
                output.flush()
                try:
                    row = json.loads(line)
                    if row.get("record") in ("profile", "dispatches"):
                        continue
                except (ValueError, AttributeError):
                    pass
                print(line, end="", flush=True)
            code = child.wait()
            thermal("after")
            if code or expired:
                raise RuntimeError(f"campaign failed: code={code}, expired={expired}")
    finally:
        timer.cancel()
        timer.join()
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            child.wait()


if __name__ == "__main__":
    main()
