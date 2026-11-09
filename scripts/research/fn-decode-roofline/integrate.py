#!/usr/bin/env python3
"""Measure committed whole-model decode; retain commands and compact metrics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--depths', type=int, nargs='+', default=[8192, 32768])
    p.add_argument('--steps', type=int, default=32)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    thermal = lambda: subprocess.check_output(['notifyutil', '-g', 'com.apple.system.thermalpressurelevel'], text=True).strip()
    a.out.mkdir(parents=True, exist_ok=True)
    thermal_file = 'thermal-'+'-'.join(map(str, a.depths))+'.jsonl'
    done = threading.Event()

    def monitor():
        with (a.out / thermal_file).open('w') as f:
            while not done.is_set():
                f.write(json.dumps({'unix': time.time(), 'thermal': thermal()})+'\n')
                f.flush()
                done.wait(1)

    t = threading.Thread(target=monitor)
    t.start()
    try:
        for depth in a.depths:
            stable = 0
            for _ in range(120):
                stable = stable+1 if thermal().endswith(' 0') else 0
                if stable == 3:
                    break
                time.sleep(5)
            else:
                raise RuntimeError('thermal pressure did not settle')
            command = [str(a.binary.resolve()), str(a.model.resolve()), str(depth), str(a.steps)]
            manifest = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), 'command': command, 'thermal_before': thermal(), 'thermal_file': thermal_file}
            with (a.out / f'{depth}.jsonl').open('w') as stdout, (a.out / f'{depth}.log').open('w') as stderr:
                # Time at the production process priority. Background builds
                # use nice 10; applying that to timing makes host-loop arms
                # vulnerable to unrelated build activity on this shared host.
                result = subprocess.run([str(root/'scripts/bench/with-metal-lease'), '--task', 'FN-5M7T-integration', '--for', '8m', '--', 'python3', str(Path(__file__).with_name('cool_exec.py')), *command], stdout=stdout, stderr=stderr, env={**os.environ, 'QWEN_METAL_LEASE_WAIT': '1'}, timeout=7200)
            manifest.update(returncode=result.returncode, thermal_after=thermal())
            (a.out / f'{depth}-command.json').write_text(json.dumps(manifest, indent=2)+'\n')
            result.check_returncode()
    finally:
        done.set()
        t.join()


if __name__ == '__main__':
    main()
