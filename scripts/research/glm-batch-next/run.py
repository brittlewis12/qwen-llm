#!/usr/bin/env python3
"""Background build outside the lease, cooled counterfactual probe, explicit waits."""
import argparse,json,os,signal,subprocess,threading,time
from pathlib import Path

def thermal():return int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'],text=True).split()[-1])
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--depth',type=int,default=8192);p.add_argument('--frames',type=int,default=4);p.add_argument('--example',choices=['glm_batch_bounds','glm_batch_waves','glm_batch_iq'],default='glm_batch_bounds');p.add_argument('--quality',type=int,default=128);p.add_argument('--cohort',choices=['distinct','forked'],default='distinct');p.add_argument('--mode',choices=['all','timing','quality'],default='all');p.add_argument('--inside',action='store_true');p.add_argument('--variants');p.add_argument('--width',type=int,default=4);p.add_argument('--queued-at',type=float);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if not a.inside:
  start=time.monotonic();cmd=['nice','-n','10','cargo','build','--offline','--locked','--release','-p','qwen-llm','--features','perf-ablations','--example',a.example]
  with(a.out/'build.log').open('w')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  report=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),build_command=cmd,build_seconds=time.monotonic()-start,build_code=r.returncode);(a.out/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);r.check_returncode()
  waits=[]
  for attempt in range(3):
   stable=0;cool_started=time.monotonic()
   while stable<12:
    stable=stable+1 if thermal()==0 else 0;time.sleep(5)
   queued=time.time();cmd=['scripts/bench/with-metal-lease','--task','GLM-batch','--for','5m','--','python3',__file__,'--inside','--cohort',a.cohort,'--mode',a.mode,'--example',a.example,'--quality',str(a.quality),'--model',str(a.model),'--out',str(a.out),'--depth',str(a.depth),'--frames',str(a.frames),'--queued-at',str(queued)]+(['--variants',a.variants,'--width',str(a.width)]if a.variants else[])
   print(json.dumps(dict(event='queued',cooling_seconds=time.monotonic()-cool_started,attempt=attempt,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),command=cmd)),flush=True)
   r=subprocess.run(cmd,env=dict(os.environ,QWEN_METAL_LEASE_WAIT='1',MTL_DEBUG_LAYER='0'))
   c=json.loads((a.out/'conditions.json').read_text());waits.append(c)
   (a.out/'attempts.json').write_text(json.dumps(waits,indent=2)+'\n')
   if r.returncode!=75:raise SystemExit(r.returncode)
  raise RuntimeError('three hot acquisitions; no probe ran')
 fd=int(os.environ['QWEN_METAL_LEASE_FD']);os.fstat(fd);wait=time.time()-a.queued_at;print(json.dumps(dict(event='acquired',wait_seconds=wait)),flush=True)
 temps=[thermal()]
 if temps[0]!=0:
  report=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),gpu_wait_seconds=wait,thermal_before=temps[0],active_seconds=0,code=75,reason='hot acquisition; no probe launched')
  (a.out/'conditions.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);raise SystemExit(75)
 stop=threading.Event()
 def watch():
  while not stop.wait(1):temps.append(thermal())
 t=threading.Thread(target=watch);t.start();start=time.monotonic();cmd=['target/release/examples/'+a.example,str(a.model),str(a.depth),str(a.frames if a.example=='glm_batch_bounds'else a.quality)]+([]if a.example=='glm_batch_bounds'else[a.mode,a.cohort])+([a.variants]if a.variants else[])+([str(a.width)]if a.example=='glm_batch_iq'and a.variants else[])
 with(a.out/'probe.jsonl').open('w')as f:
  child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(fd,))
  try:
   while child.poll()is None:
    if max(temps)>1 or time.monotonic()-start>600:os.killpg(child.pid,signal.SIGINT);break
    time.sleep(1)
   code=child.wait(timeout=20)
  finally:
   if child.poll()is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
   stop.set();t.join()
 temps.append(thermal());report=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),command=cmd,gpu_wait_seconds=wait,active_seconds=time.monotonic()-start,thermal_max=max(temps),thermal_samples=len(temps),code=code);(a.out/'conditions.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);assert code==0 and max(temps)<2
if __name__=='__main__':main()
