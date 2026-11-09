#!/usr/bin/env python3
"""Steady-window on/off ABBA, with builds and cooling outside the Metal lease."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['build-log','model','plan','corpus','harness','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--harness-commit',required=True);p.add_argument('--rollback',action='append')
    p.add_argument('--arms',nargs='+',choices=['off1','on1','on2','off2'],default=['off1','on1','on2','off2']);p.add_argument('--cooldown',type=int,default=40);a=p.parse_args()
    assert a.cooldown>=15 and len(set(a.arms))==len(a.arms)
    knobs=a.rollback or ['QWEN_GLM53_BATCH_PROJECTIONS','QWEN_GLM53_BATCH_SELECTION','QWEN_GLM53_BATCH_KDA']
    assert 'QWEN_METAL_LEASE_FD' not in os.environ
    a.out.mkdir(parents=True,exist_ok=True)
    while True:
        s=a.build_log.read_text()if a.build_log.exists()else''
        if 'could not compile'in s:raise RuntimeError(s[-2000:])
        if 'Finished `release` profile'in s:break
        time.sleep(2)
    binary=Path('target/release/qwen').resolve();tokenizer=Path('target/release/qwen-tok').resolve()
    for arm in a.arms:
        # Cooling never owns the Metal lease. Require sustained level zero;
        # other lanes may have heated the machine since the preceding arm.
        stable=0
        while stable<a.cooldown:
            pressure=int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'],text=True).split()[-1])
            stable=stable+5 if pressure==0 else 0
            time.sleep(5)
        env=dict(os.environ,QWEN_METAL_LEASE_WAIT='1',MTL_DEBUG_LAYER='0')
        for knob in knobs:
            env.pop(knob,None)
            if arm.startswith('off'):env[knob]='0'
        command=['scripts/bench/with-metal-lease','--task','glm-q6-steady-'+arm,'--for','7m','--','python3',str(a.harness),'run','--binary',str(binary),'--tokenizer',str(tokenizer),'--model',str(a.model),'--plan',str(a.plan),'--corpus',str(a.corpus),'--out',str(a.out/arm),'--arm','default','--warm-tokens','48000','--context','196608','--ramp','15','--cool','0','--timeout','900','--port','18769','--harness-commit',a.harness_commit]
        print(json.dumps(dict(start=arm,command=command)),flush=True)
        with (a.out/(arm+'.log')).open('w')as f:r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        print(json.dumps(dict(end=arm,code=r.returncode)),flush=True)
        assert r.returncode==0
        s=json.loads((a.out/arm/'summary.json').read_text());assert s['thermal_max']==0
        print(json.dumps(dict(arm=arm,aggregate_tps=s['aggregate_tps'],window_tokens=s['window_tokens'])),flush=True)
    print('complete',flush=True)
if __name__=='__main__':main()
