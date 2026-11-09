#!/usr/bin/env python3
"""Run the cache/prefetch physics packet; retain commands and compact metrics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["cache", "model", "all"], default="all")
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    a.out.mkdir(parents=True, exist_ok=True)
    thermal = lambda: subprocess.check_output(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], text=True).strip()
    stable = 0
    for _ in range(60):
        stable = stable + 1 if thermal().endswith(" 0") else 0
        if stable == 3:
            break
        time.sleep(5)
    else:
        raise RuntimeError("thermal pressure did not settle")
    done = threading.Event()

    def monitor():
        with (a.out / "thermal.jsonl").open("w") as f:
            while not done.is_set():
                f.write(json.dumps({"unix": time.time(), "thermal": thermal()}) + "\n")
                f.flush()
                done.wait(1)

    t = threading.Thread(target=monitor)
    t.start()
    try:
        with tempfile.TemporaryDirectory(prefix="fn-cache-metal-") as scratch:
            source = Path(scratch) / "probe.metal"
            source.write_text((here / "probe.metal").read_text() + "\n" + (here.parent / "layer/fused.metal").read_text())
            command = [str(a.binary.resolve()), str(a.model.resolve()), str(source), a.mode]
            manifest = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "command": command, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (a.out / "command.json").write_text(json.dumps(manifest, indent=2) + "\n")
            with (a.out / "metrics.jsonl").open("w") as stdout, (a.out / "run.log").open("w") as stderr:
                result = subprocess.run([str(root / "scripts/bench/with-metal-lease"), "--task", "FN-5M7T-cache-prefetch", "--for", "8m", "--", "nice", "-n", "10", *command], stdout=stdout, stderr=stderr, env={**os.environ, "QWEN_METAL_LEASE_WAIT": "1"}, timeout=7200)
            manifest.update(returncode=result.returncode, thermal_after=thermal())
            (a.out / "command.json").write_text(json.dumps(manifest, indent=2) + "\n")
            result.check_returncode()
    finally:
        done.set()
        t.join()


if __name__ == "__main__":
    main()
