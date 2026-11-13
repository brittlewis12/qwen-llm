#!/usr/bin/env python3
"""Real HTTP lifecycle witness. OUT BINARY MODEL CORPUS; run under solo_thermal."""
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OUT, BINARY, MODEL, CORPUS = map(Path, sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=False)
PORT = 18789
FD = int(os.environ['QWEN_METAL_LEASE_FD'])
SERVER = None

def stop(sig, frame):
    if SERVER is not None and SERVER.poll() is None:
        SERVER.send_signal(signal.SIGINT)
    raise KeyboardInterrupt
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, stop)

def request(body=None):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=180)
    try:
        c.request('GET' if body is None else 'POST', '/v1/models' if body is None else '/v1/responses',
                  body=None if body is None else json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
        r = c.getresponse()
        value = json.loads(r.read())
        if r.status != 200:
            raise RuntimeError(f'HTTP {r.status}: {value}')
        return value
    finally:
        c.close()

def body(model, maximum, greedy=False):
    value = dict(model=model, input=[dict(role='user', content=CORPUS.read_text()[:16000])],
                 instructions='Review every Rust function in this source, explain its invariants and propose tests. Keep writing a substantive detailed review until interrupted.',
                 max_output_tokens=maximum, reasoning=dict(effort='low'), x_qwen=dict(stats=True, seed=42))
    if greedy:
        value['temperature'] = 0
    return value

def pieces(value):
    return [dict(type=item.get('type'), role=item.get('role'), content=item.get('content'))
            for item in value.get('output', [])]

def stream(payload, until=5):
    payload['stream'] = True
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=180)
    c.request('POST', '/v1/responses', body=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f'stream HTTP {r.status}: {r.read()}')
    events = 0
    while events < until:
        line = r.readline()
        if not line:
            raise RuntimeError('stream ended before cancellation witness')
        if line.startswith(b'data:') and line[5:].strip() != b'[DONE]':
            event = json.loads(line[5:])
            if event.get('type') in ('response.output_text.delta', 'response.reasoning_text.delta'):
                events += 1
    return c, r, events

def settle():
    since = None
    deadline = time.monotonic() + 180
    while True:
        p = int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'], text=True).split()[-1])
        now = time.monotonic()
        if p >= 3:
            raise RuntimeError('thermal pressure 3')
        since = (now if since is None else since) if p <= 1 else None
        if since is not None and now-since >= 60:
            return
        if now >= deadline:
            raise RuntimeError('settling deadline')
        time.sleep(1)

reports = []
for arm in (0, 2):
    if arm:
        settle()
    ps = subprocess.check_output(['ps','-axo','pid,comm'], text=True)
    if any(line.split() and Path(line.split()[-1]).name == 'qwen' for line in ps.splitlines()):
        raise RuntimeError('resident qwen process before model load')
    env = dict(os.environ, QWEN_GLM53_SOLO_NEXTN=str(arm), RUST_LOG='info,qwen_diag=info')
    with (OUT/f'arm-{arm}.log').open('x') as log:
        SERVER = subprocess.Popen([str(BINARY.resolve()), 'serve', '-m', str(MODEL.absolute()),
            '--addr', f'127.0.0.1:{PORT}', '--max-context-tokens','16384','--max-tokens','4096'],
            env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(FD,))
        try:
            deadline = time.monotonic()+180
            while True:
                if SERVER.poll() is not None:
                    raise RuntimeError(f'server exited {SERVER.returncode}')
                try:
                    model = request()['data'][0]['id']
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.25)
            greedy = request(body(model,64,True))
            stopped_body = body(model,64,True)
            stopped_body['stop'] = ['Review', '\n']
            ignored_stop = request(stopped_body)
            if pieces(greedy) != pieces(ignored_stop):
                raise RuntimeError('ignored custom stop field changed greedy output')
            limits = []
            for maximum in (1,2,3,4,17):
                value = request(body(model,maximum))
                limits.append(dict(maximum=maximum, actual=value['usage']['output_tokens']))
                if limits[-1]['actual'] != maximum:
                    raise RuntimeError(f'token limit mismatch: {limits[-1]}')
            c,r,events = stream(body(model,4096))
            r.close();c.close()
            recovered = request(body(model,2))
            if recovered['usage']['output_tokens'] != 2:
                raise RuntimeError('disconnect recovery failed')
            c,r,events = stream(body(model,4096),until=12)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                peer = pool.submit(request,body(model,64))
                peer_result = peer.result(timeout=180)
            r.close();c.close()
            phase = peer_result.get('x_qwen',{}).get('phases',{})
            width = phase.get('batch',{}).get('width_max')
            if width is None or width < 2:
                raise RuntimeError(f'peer did not join a native batch: {phase}')
            three = None
            if arm == 2:
                streams=[]
                try:
                    for agent in range(3):
                        value=body(model,4096)
                        value['input'][0]['content']=f'Independent agent {agent}.\n'+value['input'][0]['content']
                        streams.append(stream(value,until=5))
                    first_line=len(Path(log.name).read_text().splitlines())
                    begin=time.time();end=begin+20
                    def drain(item):
                        c,r,_=item
                        count=0
                        while time.time()<end:
                            line=r.readline()
                            if not line or line.startswith(b'data: [DONE]'):
                                raise RuntimeError('three-agent stream ended inside smoke window')
                            if line.startswith(b'data:'):
                                event=json.loads(line[5:])
                                count+=int(event.get('type') in ('response.output_text.delta','response.reasoning_text.delta'))
                        return count
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                        counts=list(pool.map(drain,streams))
                    window=Path(log.name).read_text().splitlines()[first_line:]
                    speculative=[line for line in window if 'serve nextn:' in line]
                    if speculative:
                        raise RuntimeError(f'NextN engaged during three-agent smoke: {speculative}')
                    three=dict(seconds=time.time()-begin,events=counts,nextn_reports=len(speculative))
                finally:
                    for c,r,_ in streams:
                        r.close();c.close()
                request(body(model,2))
            report = dict(arm=arm,greedy=pieces(greedy),greedy_usage=greedy['usage'],limits=limits,three_agent=three,
                          ignored_custom_stop=True,disconnect_recovered=True,peer_native_width_max=width)
            reports.append(report)
            (OUT/f'arm-{arm}.json').write_text(json.dumps(report,indent=2)+'\n')
        finally:
            if SERVER.poll() is None:
                SERVER.send_signal(signal.SIGINT)
                try:
                    SERVER.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    SERVER.terminate();SERVER.wait(timeout=10)
            SERVER = None
if reports[0]['greedy'] != reports[1]['greedy']:
    raise RuntimeError('T=0 serve output differs with NextN enabled')
summary = dict(complete=True,greedy_bitwise_text=True,limits=True,ignored_custom_stop=True,
               disconnect_recovery=True,midturn_peer_handoff=True,reports=reports)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='reports'}),flush=True)
