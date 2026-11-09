#!/usr/bin/env python3
"""Regenerate the source corpus and seeded steady-window plan used by this lane."""
import argparse,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='9b193b38');p.add_argument('--seconds',type=int,default=90);p.add_argument('--seed',type=int,default=20261109);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    paths=sorted(subprocess.check_output(['git','ls-tree','-r','--name-only',a.source,'--','crates/qwen-llm/src','crates/qwen-cli/src'],text=True).splitlines())
    parts=[];size=0
    for path in paths:
        if not path.endswith('.rs'):continue
        data=b'// '+path.encode()+b'\n'+subprocess.check_output(['git','show',a.source+':'+path]);parts.append(data);size+=len(data)
        if size>=3_000_000:break
    assert size>=3_000_000
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'corpus.rs').write_bytes(b''.join(parts))
    subprocess.run(['python3','scripts/bench/serve_realistic.py','plan','--family','glm','--seed',str(a.seed),'--seconds',str(a.seconds),'--out',str(a.out/'plan.json')],check=True)
    print('corpus bytes',size,flush=True)
if __name__=='__main__':main()
