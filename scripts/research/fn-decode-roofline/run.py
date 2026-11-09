#!/usr/bin/env python3
"""Run one short roofline predictor under the canonical Metal lease."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--depth", type=int, default=8192)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    a.out.mkdir(parents=True, exist_ok=True)
    thermal = lambda: subprocess.check_output(
        ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], text=True
    ).strip()
    # Cool outside the GPU lease; do not occupy the GPU while another lane runs.
    stable = 0
    for _ in range(60):
        stable = stable + 1 if thermal().endswith(" 0") else 0
        if stable == 3:
            break
        time.sleep(5)
    else:
        raise RuntimeError("thermal pressure did not settle")
    command = [str(a.binary.resolve()), str(a.model.resolve()), str(here / "controls.metal"), str(a.depth)]
    manifest = {
        "argv": command,
        "base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        "research_sources": {
            name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in ["probe.rs", "capture.patch", "controls.metal", "run.py", "summarize.py"]
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
        with (a.out / "probe.jsonl").open("w") as stdout, (a.out / "probe.log").open("w") as stderr:
            result = subprocess.run(
                [str(root / "scripts/bench/with-metal-lease"), "--task", "FN-5M7T-roofline", "--for", "3m", "--", "nice", "-n", "10", *command],
                stdout=stdout, stderr=stderr, env={**os.environ, "QWEN_METAL_LEASE_WAIT": "1"},
                timeout=900,
            )
        manifest["returncode"] = result.returncode
        manifest["thermal_after"] = thermal()
        (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        result.check_returncode()
    finally:
        done.set()
        t.join()


if __name__ == "__main__":
    main()
