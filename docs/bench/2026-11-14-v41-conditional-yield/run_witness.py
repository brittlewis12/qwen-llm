import json,os,signal,subprocess,time
from pathlib import Path
top=Path('/Users/tito/qwen-runs/2026-11-12-v41-prefill/conditional-yield');repo=Path('/Users/tito/code/qwen-llm-v41-conditional');os.chdir(repo)
import sys;sys.path.insert(0,str(repo/'scripts/bench'))
import v41_prefill_serve_pair as pair
import v41_prefill_packet as packet
fd=int(os.environ['QWEN_METAL_LEASE_FD']);out=top/'witness';out.mkdir();deadline=time.monotonic()+780
assert os.getpriority(os.PRIO_PROCESS,0)==0
(out/'processes-before.txt').write_text(pair.resident_check());(out/'load-before.txt').write_text(subprocess.check_output(['uptime'],text=True,timeout=5))
cool=None;thermal=[]
while True:
    level=packet.thermal();thermal.append(dict(time=time.time(),level=level))
    if level>=3:raise RuntimeError('thermal>=3')
    if level<=1:
        cool=cool or time.monotonic()
        if time.monotonic()-cool>=60:break
    else:cool=None
    if time.monotonic()+500>deadline:raise RuntimeError('settling budget')
    time.sleep(2)
model=Path('/Users/tito/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf')
engram=model.parent/'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf';corpus=Path('/Users/tito/qwen-runs/2026-11-07-serve-agents/code.txt')
subprocess.run(['python3','scripts/bench/invalidate_file_cache.py',str(engram),'--out',str(out/'purge.jsonl')],check=True,timeout=30,pass_fds=(fd,));assert json.loads((out/'purge.jsonl').read_text().splitlines()[-1])['resident_pages_after']==0
pair.resident_check();command=[str(top/'pins/v41_prefill_yield_witness'),str(model),str(corpus)]
with (out/'native.log').open('w') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,))
    nice=int(subprocess.check_output(['ps','-o','nice=','-p',str(child.pid)],text=True,timeout=5).strip());assert nice==0
    (out/'pid.json').write_text(json.dumps(dict(pid=child.pid,nice=nice,command=command)))
    try:
        while child.poll() is None:
            level=packet.thermal();thermal.append(dict(time=time.time(),level=level));(out/'thermal.json').write_text(json.dumps(thermal))
            if level>=3 or time.monotonic()>deadline:raise RuntimeError('witness thermal/deadline')
            time.sleep(1)
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
    if child.returncode:raise RuntimeError('native witness failed')
events=[]
for l in (out/'native.log').read_text().splitlines():
    if l.startswith('{'):
        try:events.append(json.loads(l))
        except ValueError:pass
assert {v.get('event') for v in events if v.get('bitwise')}=={'root','append','root_empty','append_empty'}
result=dict(events=events,pid=child.pid,nice=nice,thermal_max=max(v['level'] for v in thermal),load_after=subprocess.check_output(['uptime'],text=True,timeout=5))
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'processes-after.txt').write_text(pair.resident_check());print(json.dumps(result),flush=True)
