#!/usr/bin/env python3
"""Run one repricing model load inside an existing <=15-minute Metal lease.

Usage: solo_thermal.py OUT_DIR COMMAND [ARG ...]
Stores a continuous thermal trace; settle <=1 for 60 s, allow 2, abort at 3.
Only terminates the exact process it starts. COMMAND must exec its model process.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def pressure():
    result = subprocess.run(
        ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
        text=True, capture_output=True, check=True,
    )
    level = int(result.stdout.split()[-1])
    if level < 0 or level > 3:
        raise RuntimeError(f"unknown pressure: {level}")
    return level


def main():
    if not os.environ.get("QWEN_METAL_LEASE_FD", "").isdigit():
        raise RuntimeError("canonical Metal lease required")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    command = sys.argv[2:]
    if not command:
        raise RuntimeError("command required")
    started = time.monotonic()
    trace = []
    metadata = {"command": command, "started_unix": time.time(), "load": os.getloadavg()}
    with (out / "thermal.jsonl").open("x") as thermal:
        def sample(phase):
            level = pressure()
            point = {"time": time.time(), "phase": phase, "pressure": level}
            thermal.write(json.dumps(point) + "\n")
            thermal.flush()
            trace.append(level)
            return level
        low_since = None
        while True:
            level = sample("settle")
            now = time.monotonic()
            low_since = (low_since if low_since is not None else now) if level <= 1 else None
            if low_since is not None and now - low_since >= 60:
                break
            if now - started > 180:
                raise RuntimeError("settling timeout; release lease")
            time.sleep(1)
        processes = subprocess.check_output(["ps", "-axo", "pid,comm"], text=True)
        resident = [line for line in processes.splitlines() if line.split() and Path(line.split()[-1]).name == "qwen"]
        metadata["resident_qwen"] = resident
        if resident:
            raise RuntimeError(f"another qwen model process remains resident: {resident}")
        metadata["processes_before"] = processes
        metadata["settle_s"] = time.monotonic() - started
        (out / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        fd = int(os.environ["QWEN_METAL_LEASE_FD"])
        with (out / "stdout.log").open("x") as stdout, (out / "stderr.log").open("x") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, pass_fds=(fd,))
            metadata["pid"] = process.pid
            try:
                while process.poll() is None:
                    if sample("run") >= 3:
                        raise RuntimeError("thermal pressure reached 3")
                    if time.monotonic() - started >= 14 * 60:
                        raise RuntimeError("14 minute lease budget exhausted")
                    time.sleep(1)
                metadata["exit_code"] = process.wait()
                if metadata["exit_code"]:
                    raise RuntimeError(f"command exited {metadata['exit_code']}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                metadata.update(elapsed_s=time.monotonic()-started, max_pressure=max(trace), load_after=os.getloadavg())
                (out / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
