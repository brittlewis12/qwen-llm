#!/usr/bin/env python3
"""One short leased append decomposition, no build lock during timing."""
import argparse,json,os,signal,subprocess,threading,time
from pathlib import Path

def pressure():
    return int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'],text=True).split()[-1])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['fn','glm'],required=True)
    for s in ['binary','model','corpus','out']:p.add_argument('--'+s,type=Path,required=True)
    p.add_argument('--inside',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    if not a.inside:
        cmd=[str(root/'scripts/bench/with-metal-lease'),'--task','append-roofline-'+a.family,'--for','5m','--',
             'nice','-n','10','python3',str(Path(__file__).resolve()),*os.sys.argv[1:],'--inside']
        return subprocess.call(cmd,env=dict(os.environ,QWEN_METAL_LEASE_WAIT='1'))
    a.out.mkdir(parents=True,exist_ok=False)
    time.sleep(25)
    for _ in range(3):
        assert pressure()==0,'nonzero thermal before arm';time.sleep(3)
    cmd=[str(a.binary.resolve()),a.family,str(a.model.resolve()),str(a.corpus.resolve()),str(Path(__file__).with_name('controls.metal'))]
    receipt=dict(command=cmd,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 instrument='python3 scripts/research/append-roofline/prepare.py',start_utc=time.strftime('%FT%TZ',time.gmtime()),
                 qwen_env={k:v for k,v in os.environ.items() if k.startswith('QWEN_')})
    samples=[];done=threading.Event();abort=[];start=time.time()
    with (a.out/'probe.jsonl').open('w') as out,(a.out/'stderr.txt').open('w') as err:
        child=subprocess.Popen(cmd,stdout=out,stderr=err,cwd=root,pass_fds=(int(os.environ['QWEN_METAL_LEASE_FD']),))
        def watch():
            while not done.is_set():
                value=pressure();samples.append(dict(time=time.time(),thermal=value))
                if value or time.time()-start>600:
                    abort.append('thermal' if value else 'deadline')
                    if child.poll() is None:child.send_signal(signal.SIGINT)
                    return
                done.wait(1)
        def stop(sig,frame):
            abort.append('signal')
            if child.poll() is None:child.send_signal(signal.SIGINT)
        for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
        watcher=threading.Thread(target=watch);watcher.start()
        try:code=child.wait(timeout=630)
        finally:
            if child.poll() is None:child.kill();child.wait()
            done.set();watcher.join()
    samples.append(dict(time=time.time(),thermal=pressure()))
    receipt.update(returncode=code,wall_s=time.time()-start,thermal_max=max(x['thermal'] for x in samples),abort=abort)
    (a.out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (a.out/'thermal.json').write_text(json.dumps(samples,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return 1 if code or abort or receipt['thermal_max'] else 0

if __name__=='__main__':raise SystemExit(main())
