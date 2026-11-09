#!/usr/bin/env python3
"""Run freshly built library/CLI test executables under one GPU lease.

Build first, under build.lock, with cargo test -p qwen-llm -p qwen-cli --no-run.
This runner never builds and retains only commands, counts and failure logs.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-log', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lease = int(os.environ['QWEN_METAL_LEASE_FD'])
    build = re.sub(r'\x1b\[[0-9;]*m', '', a.build_log.read_text())
    binaries = re.findall(r'^\s*Executable .*\(([^)]+)\)\s*$', build, re.M)
    assert binaries and 'could not compile' not in build, 'no successful test build'
    assert any('qwen_llm-' in p for p in binaries), 'library tests missing'
    assert any(re.search(r'/qwen-[0-9a-f]+$', p) for p in binaries), 'CLI tests missing'
    report = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'suites': []}
    for binary in dict.fromkeys(binaries):
        command = [str(Path(binary).resolve()), '--test-threads=1']
        start = time.monotonic()
        result = subprocess.run(command, text=True, capture_output=True, env={**os.environ, 'QWEN_REQUIRE_METAL_TESTS': '1'}, pass_fds=(lease,), timeout=1200)
        output = result.stdout+'\n'+result.stderr
        summary = re.findall(r'^test result:.*$', output, re.M)
        row = {'command': command, 'returncode': result.returncode, 'seconds': time.monotonic()-start, 'summary': summary}
        report['suites'].append(row)
        (a.out/'tests.json').write_text(json.dumps(report, indent=2)+'\n')
        if result.returncode:
            (a.out/(Path(binary).name+'-failure.log')).write_text(output)
        else:
            (a.out/(Path(binary).name+'.log')).write_text('\n'.join(summary)+'\n')
        print(json.dumps(row), flush=True)
        result.check_returncode()


if __name__ == '__main__':
    main()
