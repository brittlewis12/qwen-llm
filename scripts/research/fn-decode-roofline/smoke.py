#!/usr/bin/env python3
"""Functional CLI/owner smoke. Invoke under the canonical Metal lease."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    binary, model = str(a.binary.resolve()), str(a.model.resolve())
    thermal = lambda: subprocess.check_output(['notifyutil', '-g', 'com.apple.system.thermalpressurelevel'], text=True).strip()
    manifest = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'thermal_before': thermal(), 'commands': []}
    prompt = 'Count from 1 to 200, separating each number with a comma.'
    for enabled in [False, True]:
        command = [binary, 'run', '--model', model, '--user', prompt, '--no-thinking', '--temp', '0', '--seed', '19', '--max-tokens', '32', '--max-context-tokens', '4096']
        env = {**os.environ, 'QWEN4EXP_GPU_TOKEN_LOOP': str(int(enabled))}
        manifest['commands'].append({'argv': command, 'env': {'QWEN4EXP_GPU_TOKEN_LOOP': str(int(enabled))}})
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
        (a.out/f'run-{int(enabled)}.log').write_text(result.stderr)
        (a.out/f'run-{int(enabled)}.txt').write_text(result.stdout)
        result.check_returncode()
        assert result.stdout.strip(), 'empty CLI output'
    assert (a.out/'run-0.txt').read_text() == (a.out/'run-1.txt').read_text(), 'greedy CLI output diverged'
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    command = [binary, 'serve', '--model', model, '--addr', f'127.0.0.1:{port}', '--max-context-tokens', '4096', '--max-tokens', '64', '--snapshot-cache-mib', '0', '--durable-snapshot-dir', 'off']
    manifest['commands'].append({'argv': command})
    with (a.out/'serve.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic()+180
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited {proc.returncode}')
                try:
                    with urllib.request.urlopen(base+'/v1/models', timeout=1) as r:
                        model_id = json.load(r)['data'][0]['id']
                    break
                except OSError:
                    if time.monotonic()>deadline:
                        raise
                    time.sleep(0.2)

            def request(label, disconnect=False):
                body = {'model': model_id, 'input': prompt+f' Request {label}.', 'stream': True, 'max_output_tokens': 64 if not disconnect else 512, 'temperature': 0, 'seed': 19}
                req = urllib.request.Request(base+'/v1/responses', json.dumps(body).encode(), {'Content-Type': 'application/json'})
                events = []
                with urllib.request.urlopen(req, timeout=120) as r:
                    for line in r:
                        if not line.startswith(b'data: '):
                            continue
                        payload = line[6:].strip()
                        if payload == b'[DONE]':
                            continue
                        event = json.loads(payload)
                        events.append(event)
                        if disconnect and event.get('type', '').endswith('.delta'):
                            break
                if not disconnect:
                    assert events[-1]['type'] in ('response.completed', 'response.incomplete'), events[-1]
                    assert any(e.get('type', '').endswith('.delta') for e in events)
                return {'label': label, 'disconnected': disconnect, 'events': len(events), 'terminal': events[-1].get('type')}

            results = [request('lone')]
            with ThreadPoolExecutor(max_workers=2) as pool:
                results.extend(pool.map(request, ['peer-a', 'peer-b']))
            results.append(request('cancel', True))
            results.append(request('after-cancel'))
            (a.out/'http.json').write_text(json.dumps(results, indent=2)+'\n')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    log = (a.out/'serve.log').read_text()
    feedback = [int(x) for x in re.findall(r'"gpu_feedback_transitions":\s*(\d+)', log)]
    native = [int(x) for x in re.findall(r'"batched_transitions":\s*(\d+)', log)]
    assert any(feedback), 'default owner never used GPU feedback'
    assert any(native), 'concurrent requests never used native B>1 decode'
    manifest.update(thermal_after=thermal(), gpu_feedback_transitions=feedback, batched_transitions=native)
    (a.out/'command.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'passed': True, 'gpu_feedback_transitions': feedback, 'batched_transitions': native}))


if __name__ == '__main__':
    main()
