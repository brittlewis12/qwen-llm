"""Build the standalone Metal controls. Both arms use product Metal flags."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)

def run(*cmd):
    print(json.dumps({'command': cmd}), flush=True)
    subprocess.run(cmd, check=True, cwd=root)

run(sys.executable, str(Path(__file__).with_name('generate.py')), str(out / 'probe.metal'))
run('xcrun', '-sdk', 'macosx', 'metal', '-std=metal4.0', '-mmacosx-version-min=26.0', '-O3', '-ffast-math', '-c', str(out/'probe.metal'), '-o', str(out/'probe.air'))
run('xcrun', '-sdk', 'macosx', 'metallib', str(out/'probe.air'), '-o', str(out/'probe.metallib'))
run('xcrun', 'clang++', '-std=c++20', '-O3', '-fobjc-arc', '-framework', 'Foundation', '-framework', 'Metal', str(Path(__file__).with_name('probe.mm')), '-o', str(out/'probe'))
(out/'identity.json').write_text(json.dumps({'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(), 'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != 'identity.json'}}, indent=2)+'\n')
