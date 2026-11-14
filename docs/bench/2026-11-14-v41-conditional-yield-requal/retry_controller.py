import datetime,json,os,signal,subprocess,time
from pathlib import Path
TOP=Path('/Users/tito/qwen-runs/2026-11-12-v41-prefill/conditional-yield')
REPO=Path('/Users/tito/code/qwen-llm-v41-conditional')
os.chdir(REPO)
(TOP/'dispatch/retry-controller.pid').write_text(str(os.getpid())+'\n')
records=json.loads((TOP/'retry-attempts.json').read_text()) if (TOP/'retry-attempts.json').exists() else []

def emit(**r):
    r['time']=time.time();print(json.dumps(r),flush=True)

def tree(root):
    rows={}
    for line in subprocess.check_output(['ps','-ww','-axo','pid=,ppid=,args='],text=True,timeout=5).splitlines():
        f=line.strip().split(None,2)
        if len(f)==3:rows[int(f[0])]=(int(f[1]),f[2])
    selected={root};changed=True
    while changed:
        before=len(selected);selected.update(pid for pid,(parent,_) in rows.items() if parent in selected);changed=before!=len(selected)
    return {pid:rows[pid][1] for pid in selected if pid in rows}

def stop_exact(child):
    started={pid:subprocess.run(['ps','-p',str(pid),'-o','lstart='],capture_output=True,text=True,timeout=5).stdout.strip() for pid in tree(child.pid)}
    def same(pid,stamp):
        return bool(stamp) and subprocess.run(['ps','-p',str(pid),'-o','lstart='],capture_output=True,text=True,timeout=5).stdout.strip()==stamp
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid,stamp in sorted(started.items(),reverse=True):
            if same(pid,stamp):
                try:os.kill(pid,sig)
                except ProcessLookupError:pass
        until=time.monotonic()+5
        while time.monotonic()<until:
            if not any(same(pid,stamp) for pid,stamp in started.items()):
                child.wait(timeout=5);return
            time.sleep(.2)
    child.wait(timeout=5)
    assert not any(same(pid,stamp) for pid,stamp in started.items())

def interrupted(signum,frame):raise KeyboardInterrupt(signum)
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
assert 'QWEN_METAL_LEASE_FD' not in os.environ
for tag,mode,latest in [('B2r','candidate',1794687900),('A2r','baseline',1794689040)]:
    if (TOP/tag/'result.json').exists():
        emit(event='arm_already_complete',tag=tag);continue
    attempt=2 if tag=='B2r' else 1
    while time.time()<latest:
        name=f'{tag}-lease{attempt}'
        path=TOP/(name+'-driver.log')
        if path.exists():attempt+=1;continue
        command=['scripts/bench/with-metal-lease','--task',f'v41-conditional-retry-{tag}-{attempt}','--for','15m','--','timeout','870','python3',str(TOP/'retry_arm.py'),tag,'--mode',mode]
        env=dict(os.environ,QWEN_METAL_LEASE_WAIT='1')
        with path.open('w') as log:
            # The controller has no lease FD; its child acquires one in FIFO.
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=())
            (TOP/'dispatch'/(name+'.pid')).write_text(str(child.pid)+'\n')
            record=dict(tag=tag,attempt=attempt,pid=child.pid,start=time.time(),command=command);records.append(record)
            (TOP/'retry-attempts.json').write_text(json.dumps(records,indent=2)+'\n')
            emit(event='queued',tag=tag,attempt=attempt,pid=child.pid)
            last=0
            try:
                while child.poll() is None:
                    now=time.time()
                    if now>latest and not (TOP/tag/'command.json').exists():
                        emit(event='start_cutoff',tag=tag,attempt=attempt);stop_exact(child);break
                    if now>=1794690000:
                        emit(event='stage_cleanup_cutoff',tag=tag);stop_exact(child);break
                    if now-last>=50:
                        emit(event='waiting_or_running',tag=tag,attempt=attempt);last=now
                    time.sleep(1)
            finally:
                if child.poll() is None:stop_exact(child)
                record.update(end=time.time(),exit_code=child.returncode)
                (TOP/'retry-attempts.json').write_text(json.dumps(records,indent=2)+'\n')
        if child.returncode==0 and (TOP/tag/'result.json').exists():
            emit(event='arm_complete',tag=tag,attempt=attempt);break
        if child.returncode==75:
            emit(event='released_and_requeuing',tag=tag,attempt=attempt);attempt+=1;continue
        emit(event='arm_incomplete',tag=tag,attempt=attempt,exit_code=child.returncode);raise SystemExit(1)
    else:
        emit(event='start_cutoff',tag=tag);raise SystemExit(75)
emit(event='pair_complete')
