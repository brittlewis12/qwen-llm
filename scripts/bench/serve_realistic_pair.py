#!/usr/bin/env python3
"""Retain/resume ABBA steady arms for one rollback knob; run under the Metal lease.

Pass ordinary serve_realistic.py run arguments after --, except --out.
Level-1 and compile-overlapped arms remain in the paired report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import serve_realistic as bench


def schedule():
    return [('reference-0',False),('candidate-0',True),
            ('candidate-1',True),('reference-1',False)]


def paired_report(root):
    pairs=[]
    for suffix in ('0','1'):
        a,b=root/f'reference-{suffix}',root/f'candidate-{suffix}'
        if (a/'summary.json').exists() and (b/'summary.json').exists():
            x=bench.compare(a,b);x['order']='AB' if suffix=='0' else 'BA';pairs.append(x)
    ratios=[x['throughput_ratio'] for x in pairs]
    return dict(pairs=pairs,complete=len(pairs)==2,
                order_effect_spread_percent=100*(max(ratios)-min(ratios)) if len(ratios)==2 else None,
                opposite_throughput_directions=len(ratios)==2 and (ratios[0]>1)!=(ratios[1]>1),
                policy='retain thermal1/compile flags; review both order effects, no automatic discard')


def fingerprint(args):
    files={}
    for key in ('--binary','--tokenizer','--plan','--corpus'):
        if key in args:
            p=Path(args[args.index(key)+1]).resolve()
            files[key]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    files['harness']=hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest()
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--flag',required=True)
    p.add_argument('--reference',default='0');p.add_argument('--candidate',default='unset')
    p.add_argument('--resume',action='store_true');p.add_argument('run_args',nargs=argparse.REMAINDER)
    a=p.parse_args();args=a.run_args[1:] if a.run_args[:1]==['--'] else a.run_args
    if not a.flag.startswith('QWEN_') or '--out' in args:p.error('use a QWEN_ rollback and let this driver supply --out')
    fd=int(os.environ['QWEN_METAL_LEASE_FD']);os.fstat(fd)
    protocol=dict(flag=a.flag,reference=a.reference,candidate=a.candidate,run_args=args,files=fingerprint(args))
    if a.out.exists():
        if not a.resume or json.loads((a.out/'protocol.json').read_text())!=protocol:
            p.error('existing pair requires --resume with identical inputs/binaries/harness')
    else:a.out.mkdir(parents=True)
    bench.dump(a.out/'protocol.json',protocol)
    child=None
    def stop(sig,frame):
        if child is not None and child.poll() is None:child.send_signal(sig)
        raise KeyboardInterrupt
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    for label,candidate in schedule():
        out=a.out/label
        if (out/'summary.json').exists():continue
        if out.exists():
            n=0
            while (a.out/f'{label}-failed-{n}').exists():n+=1
            out.rename(a.out/f'{label}-failed-{n}')
        if fingerprint(args)!=protocol['files']:raise RuntimeError('inputs or binaries changed during pair')
        env=dict(os.environ);env.pop(a.flag,None);value=a.candidate if candidate else a.reference
        if value!='unset':env[a.flag]=value
        cmd=[sys.executable,str(Path(bench.__file__)),'run',*args,'--out',str(out)]
        child=subprocess.Popen(cmd,env=env,pass_fds=(fd,));code=child.wait()
        with (a.out/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(label=label,code=code,command=cmd))+'\n')
        bench.dump(a.out/'pair.json',paired_report(a.out))
    report=paired_report(a.out);print(json.dumps(report),flush=True)
    return 0 if report['complete'] else 1


if __name__=='__main__':sys.exit(main())
