#!/usr/bin/env python3
"""Monitor one inherited-lease command; keep compact conditions, discard bulk later."""
import argparse,json,os,subprocess,threading,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--binary-source',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
fd=int(os.environ['QWEN_METAL_LEASE_FD']);os.fstat(fd)
def pressure():return int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'],text=True).split()[-1])
a.out.mkdir(parents=True,exist_ok=True);temps=[pressure()];assert temps[0]==0
stop=threading.Event()
def watch():
 while not stop.wait(.5):temps.append(pressure())
t=threading.Thread(target=watch);t.start();start=time.monotonic()
try:
 with (a.out/'output.jsonl').open('w')as f:r=subprocess.run(a.command,stdout=f,stderr=subprocess.STDOUT,pass_fds=(fd,),timeout=600)
finally:stop.set();t.join();temps.append(pressure())
v=dict(binary_source=a.binary_source,argv=a.command,seconds=time.monotonic()-start,samples=len(temps),max_thermal=max(temps),code=r.returncode)
(a.out/'conditions.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True);assert r.returncode==0 and max(temps)==0
