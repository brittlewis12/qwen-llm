#!/usr/bin/env python3
"""Run the self-leased model kill packet, preserving compact conditions only."""
import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-log',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--depth',type=int,default=8192);p.add_argument('--validation',choices=['0','1'],default='0');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert 'QWEN_METAL_LEASE_FD'not in os.environ
    while True:
        s=a.build_log.read_text()if a.build_log.exists()else''
        if 'error:'in s:raise RuntimeError(s[-2000:])
        m=re.search(r'Executable .*\((target/release/deps/qwen_llm-[^)]+)\)',s)
        if m:break
        time.sleep(2)
    binary=Path(m[1]);target='glm5_next_metal::packed::batch::tests::projection_model::m5_projection_abba_and_distribution'
    discovered=subprocess.check_output([str(binary),target,'--list'],text=True);assert target+': test'in discovered,discovered
    env=dict(os.environ,QWEN_METAL_LEASE_WAIT='1',GLM53_GGUF=str(a.model),GLM53_BATCH_DEPTH=str(a.depth),MTL_DEBUG_LAYER=a.validation)
    argv=[str(binary),target,'--exact','--ignored','--nocapture','--test-threads=1']
    temps=[];active=None;code=None
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'packet.log').open('w')as f:
        child=subprocess.Popen(argv,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll()is None:
                record=Path(f'/tmp/qwen-llm-{os.getuid()}/metal.lock').read_text()
                if re.search(rf'\bpid={child.pid}\b',record):
                    if active is None:active=time.monotonic()
                    thermal=int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'],text=True).split()[-1]);temps.append(thermal)
                    if thermal or time.monotonic()-active>600:os.killpg(child.pid,signal.SIGINT);break
                time.sleep(1)
            code=child.wait(timeout=25)
        finally:
            if child.poll()is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
    summary=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),argv=argv,env={k:env[k]for k in ['GLM53_GGUF','GLM53_BATCH_DEPTH','MTL_DEBUG_LAYER','QWEN_METAL_LEASE_WAIT']},code=code,samples=len(temps),max_thermal=max(temps,default=-1),active_seconds=None if active is None else time.monotonic()-active)
    (a.out/'conditions.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));assert code==0 and temps and max(temps)==0,summary
if __name__=='__main__':main()
