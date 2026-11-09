#!/usr/bin/env python3
"""Gate timing after acquiring the GPU lease, then exec the measured command."""
import json
import os
import subprocess
import sys
import time

assert 'QWEN_METAL_LEASE_FD' in os.environ, 'run inside the canonical Metal lease'
deadline = time.monotonic()+60
stable = 0
while True:
    pressure = subprocess.check_output(['notifyutil', '-g', 'com.apple.system.thermalpressurelevel'], text=True).strip()
    stable = stable+1 if pressure.endswith(' 0') else 0
    if stable == 3:
        break
    if time.monotonic() >= deadline:
        raise RuntimeError('thermal pressure did not settle inside the lease')
    time.sleep(5)
print(json.dumps({'kind': 'thermal_gate', 'unix': time.time(), 'thermal': pressure}), file=sys.stderr, flush=True)
os.execv(sys.argv[1], sys.argv[1:])
