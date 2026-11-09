#!/usr/bin/env python3
"""Short GPU command under lease, nice 10, cooled thermal zero. No build lock."""
import argparse,json,os,signal,subprocess,threading,time
from pathlib import Path

def pressure():return int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'],text=True).split()[-1])

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--seconds',type=int,default=600);p.add_argument('--inside',action='store_true');p.add_argument('cmd',nargs=argparse.REMAINDER);a=p.parse_args()
 cmd=a.cmd[1:] if a.cmd[:1]==['--'] else a.cmd
 assert cmd,'command required';root=Path(__file__).resolve().parents[3]
 if not a.inside:
  launch=[str(root/'scripts/bench/with-metal-lease'),'--task','fn-append-'+a.out.name,'--for','5m','--','nice','-n','10','python3',str(Path(__file__).resolve()),'--inside','--out',str(a.out),'--seconds',str(a.seconds),'--',*cmd]
  return subprocess.call(launch,env=dict(os.environ,QWEN_METAL_LEASE_WAIT='1'))
 a.out.mkdir(parents=True,exist_ok=False);time.sleep(25)
 for _ in range(3):assert pressure()==0,'nonzero thermal';time.sleep(3)
 start=time.time();samples=[];done=threading.Event();aborts=[]
 with (a.out/'stdout.txt').open('w') as out,(a.out/'stderr.txt').open('w') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err,cwd=root,start_new_session=True,pass_fds=(int(os.environ['QWEN_METAL_LEASE_FD']),))
  def stop(sig,frame):
   aborts.append('signal')
   if child.poll() is None:os.killpg(child.pid,signal.SIGINT)
  for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
  def watch():
   while not done.is_set():
    t=pressure();samples.append(dict(time=time.time(),thermal=t))
    if t or time.time()-start>a.seconds:
     aborts.append('thermal' if t else 'deadline')
     if child.poll() is None:os.killpg(child.pid,signal.SIGINT)
     return
    done.wait(1)
  watcher=threading.Thread(target=watch);watcher.start()
  try:code=child.wait(timeout=a.seconds+45)
  finally:
   if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
   done.set();watcher.join()
 samples.append(dict(time=time.time(),thermal=pressure()))
 keys=['QWEN_FN_APPEND_SHAPES','QWEN_FN_APPEND_EXPERT_PANELS','QWEN_FN_APPEND_PANEL_FUSED','QWEN_FN_APPEND_TENSOR_DOWN','QWEN_FN_APPEND_HC_SPLIT','FLASH_PREFILL_MODEL','FLASH_PREFILL_OUT','FLASH_APPEND_PLAN','MTL_DEBUG_LAYER','QWEN_REQUIRE_METAL_TESTS']
 receipt=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),command=cmd,env={k:os.environ[k] for k in keys if k in os.environ},returncode=code,wall_s=time.time()-start,thermal_max=max(s['thermal'] for s in samples),aborts=aborts)
 (a.out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');(a.out/'thermal.json').write_text(json.dumps(samples,indent=2)+'\n')
 print(json.dumps(receipt),flush=True);return int(bool(code or aborts or receipt['thermal_max']))

if __name__=='__main__':raise SystemExit(main())
