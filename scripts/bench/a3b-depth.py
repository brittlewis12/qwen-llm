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
