#!/usr/bin/env python3
"""Run short Q6/router screens inside an inherited Metal lease; no build lock."""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def pressure():
    return int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'],text=True).split()[-1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();fd=int(os.environ['QWEN_METAL_LEASE_FD']);os.fstat(fd)
    a.out.mkdir(parents=True,exist_ok=True);temps=[pressure()];assert temps[0]==0
    done=threading.Event()
    def watch():
        while not done.wait(.5):temps.append(pressure())
    t=threading.Thread(target=watch);t.start();begin=time.monotonic();codes={}
    try:
        for mode in ['sweep','routers']:
            with (a.out/(mode+'.jsonl')).open('w')as f:
                r=subprocess.run([str(a.binary.resolve()),str(a.model),mode],stdout=f,stderr=subprocess.STDOUT,pass_fds=(fd,))
            codes[mode]=r.returncode
            if r.returncode:break
    finally:
        done.set();t.join();temps.append(pressure())
        summary=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seconds=time.monotonic()-begin,samples=len(temps),max_thermal=max(temps),before=temps[0],after=temps[-1],exit_codes=codes)
        (a.out/'conditions.json').write_text(json.dumps(summary)+'\n');print(json.dumps(summary))
    assert max(temps)==0 and codes==dict(sweep=0,routers=0),summary
if __name__=='__main__':main()
