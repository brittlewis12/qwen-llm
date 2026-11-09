#!/usr/bin/env python3
"""Short, cancellable native rounds. Builds are separate; timing never takes the build lock."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path


def pressure():
    return int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'],text=True).split()[-1])


def leased(a):
    fd=int(os.environ['QWEN_METAL_LEASE_FD']);assert pressure()<2
    cmd=[str(a.binary.resolve()),str(a.model.resolve()),str(a.depth),str(a.batch),str(a.steps)]
    if a.bounds:cmd.append('bounds')
    if a.waves:cmd.append('waves')
    if a.causal:cmd.append('causal')
    start=time.monotonic();reason=None;samples=[];child=None
    def stop(sig,frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT)
        raise KeyboardInterrupt
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    a.out.mkdir(parents=True,exist_ok=True)
    stem=a.out/a.leased
    try:
        with stem.with_suffix('.stdout').open('w') as out,stem.with_suffix('.stderr').open('w') as err:
            child=subprocess.Popen(cmd,stdout=out,stderr=err,pass_fds=(fd,),start_new_session=True)
            while child.poll() is None:
                time.sleep(1);p=pressure()
                proc=subprocess.run(['pgrep','-x','rustc|clang|clang[+][+]|metal'],capture_output=True,text=True)
                assert proc.returncode in (0,1),proc.stderr
                samples.append(dict(unix=time.time(),pressure=p,compiler_pids=proc.stdout.split()))
                if p>=2 or time.monotonic()-start>210:
                    reason='thermal' if p>=2 else 'deadline';child.send_signal(signal.SIGINT);break
            try:code=child.wait(timeout=35)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait();reason='drain timeout'
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=35)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        os.close(fd);os.environ.pop('QWEN_METAL_LEASE_FD',None)
    record=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                command=cmd,env={k:v for k,v in os.environ.items() if k.startswith('QWEN_')},
                exit=code,reason=reason,elapsed_s=time.monotonic()-start,thermal_samples=samples)
    stem.with_suffix('.conditions.json').write_text(json.dumps(record,indent=2)+'\n')
    if code or reason:raise RuntimeError(record)
    rows=[json.loads(s) for s in stem.with_suffix('.stdout').read_text().splitlines() if s.startswith('{')]
    result=rows if (a.bounds or a.waves or a.causal) else next(x for x in rows if x['kind']=='kill')
    for row in rows:
        if row.get('kind')=='causal_screen':
            observed=[s for s in samples if row['started_unix']<=s['unix']<=row['finished_unix']]
            row['thermal_max']=max([row['thermal_before'],row['thermal_after']]+[s['pressure'] for s in observed])
            row['thermal_level_1']=row['thermal_max']==1
            row['compile_overlapped']=any(s['compiler_pids'] for s in observed)
            row['monitor_samples']=observed
        if row.get('kind')=='wave_capture':
            meta=row['metadata'];meta['dispatches']=len(meta.pop('steps'))
    print(json.dumps(dict(event='complete',label=a.leased,commit=record['commit'],result=result)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['binary','model','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--depth',type=int,default=32768);p.add_argument('--batch',type=int,default=3)
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--cool',type=float,default=25);p.add_argument('--leased')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bounds',action='store_true')
    mode.add_argument('--waves',action='store_true')
    mode.add_argument('--causal',action='store_true')
    a=p.parse_args()
    if a.leased:leased(a);return
    assert 'QWEN_METAL_LEASE_FD' not in os.environ
    assert subprocess.run(['git','diff','--quiet','HEAD']).returncode==0,'commit code before timing'
    root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    active_child=None
    def cancel(sig,frame):
        if active_child is not None and active_child.poll() is None:
            os.killpg(active_child.pid,signal.SIGINT)
        raise KeyboardInterrupt
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,cancel)
    for i,arm in enumerate(['bounds'] if a.bounds else ['waves'] if a.waves else ['causal']):
        assert arm in ['bounds','waves','causal'];time.sleep(a.cool)
        for _ in range(3):
            assert pressure()<2;time.sleep(3)
        label=f'{arm}-{i}-d{a.depth}-b{a.batch}'
        cmd=[str(root/'scripts/bench/with-metal-lease'),'--task','v41-round-'+label,'--for','2m','--',
             'python3',str(Path(__file__).resolve()),'--leased',label,'--binary',str(a.binary.resolve()),
             '--model',str(a.model.resolve()),'--out',str(a.out.resolve()),'--depth',str(a.depth),
             '--batch',str(a.batch),'--steps',str(a.steps)]
        if a.bounds:cmd.append('--bounds')
        if a.waves:cmd.append('--waves')
        if a.causal:cmd.append('--causal')
        env=dict(os.environ,QWEN_METAL_LEASE_WAIT='1');env.pop('MTL_DEBUG_LAYER',None);env.pop('QWEN_REQUIRE_METAL_TESTS',None)
        print(json.dumps(dict(event='queued',label=label,command=cmd,requested_unix=time.time())),flush=True)
        active_child=subprocess.Popen(cmd,env=env,start_new_session=True)
        try:
            code=active_child.wait()
            if code:raise subprocess.CalledProcessError(code,cmd)
        finally:
            if active_child.poll() is None:
                os.killpg(active_child.pid,signal.SIGINT)
                try:active_child.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(active_child.pid,signal.SIGKILL);active_child.wait()
            active_child=None

if __name__=='__main__':main()
