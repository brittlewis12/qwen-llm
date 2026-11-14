#!/usr/bin/env python3
import argparse,hashlib,json,os,signal,subprocess,sys,time,threading
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('--mode',choices=('baseline','candidate'),required=True);a=p.parse_args()
TOP=Path('/Users/tito/qwen-runs/2026-11-12-v41-prefill/intake-qualify');REPO=Path('/Users/tito/code/qwen-llm-intake-qualify');out=TOP/a.tag
if out.exists():raise RuntimeError('refusing arm overwrite')
# Stage started12:36:17 UTC. Leave cleanup/report time after the last hold.
if time.time()>1794665700:raise RuntimeError('14:15 UTC arm start cutoff reached')
out.mkdir();os.chdir(REPO);sys.path.insert(0,str(REPO/'scripts/bench'))
import v41_prefill_serve_pair as pair
import v41_prefill_packet as packet
for key in ('QWEN_SERVE_ASYNC_PREPARE','QWEN_SERVE_TOKEN_PREFIX_CACHE'):os.environ.pop(key,None)
fd=int(os.environ['QWEN_METAL_LEASE_FD']);assert os.getpriority(os.PRIO_PROCESS,0)==0
deadline=time.monotonic()+780;start=time.time()
(out/'processes-before.txt').write_text(pair.resident_check());(out/'load-before.txt').write_text(subprocess.check_output(['uptime'],text=True,timeout=5))
(out/'tmp').mkdir();env=dict(os.environ,TMPDIR=str(out/'tmp'),QWEN_SERVE_ASYNC_PREPARE='0',QWEN_SERVE_TOKEN_PREFIX_CACHE='0',QWEN_SERVE_TTFT_TIMING='0')
model=Path('/Users/tito/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf');engram=model.parent/'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf';corpus=Path('/Users/tito/qwen-runs/2026-11-07-serve-agents/code.txt')
warm=[]
def cpu_check(command,name,limit):
    if packet.thermal()>=3:raise RuntimeError('prewarm thermal>=3')
    with (out/name).open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=(fd,))
        try:
            nice=int(subprocess.check_output(['ps','-o','nice=','-p',str(child.pid)],text=True,timeout=5).strip());assert nice==0
            until=time.monotonic()+limit
            while child.poll() is None:
                if packet.thermal()>=3 or time.monotonic()>=until:raise RuntimeError('prewarm thermal/deadline')
                time.sleep(.25)
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
    if child.returncode:raise RuntimeError('prewarm failed: '+name)
    warm.append(dict(pid=child.pid,nice=nice,code=child.returncode,log=name))
cpu_check([str(TOP/'pins/qwen-cpu-tests'),'serve::token_prefix_cache::tests::','serve::batch_lifecycle_tests::accepted_request_renders_and_tokenizes_while_peer_pump_is_blocked','serve::transport::tests::preparation_cap_preserves_queued_socket_and_releases_on_shutdown','--nocapture','--test-threads=1'],'cpu-warmup.log',30)
assert '6 passed' in (out/'cpu-warmup.log').read_text()
cpu_check([str(TOP/'pins/token_prefix_exact'),str(model),str(corpus)],'released-tokenizer.log',100)
assert 'full_ids=true prefix_ids=true' in (out/'released-tokenizer.log').read_text()
(out/'warmup-pids.json').write_text(json.dumps(warm,indent=2))
cool=None;levels=[]
while True:
    level=packet.thermal();levels.append(dict(time=time.time(),level=level))
    if level>=3:raise RuntimeError('thermal>=3')
    if level<=1:
        cool=cool or time.monotonic()
        if time.monotonic()-cool>=60:break
    else:cool=None
    (out/'preload-thermal.json').write_text(json.dumps(levels))
    if time.monotonic()+550>=deadline:raise RuntimeError('preload settling budget exhausted')
    time.sleep(2)
(out/'preload-thermal.json').write_text(json.dumps(levels));pair.resident_check()
purge=out/'purge.jsonl';subprocess.run(['python3','scripts/bench/invalidate_file_cache.py',str(engram),'--out',str(purge)],check=True,timeout=60,pass_fds=(fd,));reset=json.loads(purge.read_text().splitlines()[-1]);assert reset['msync_rc']==0 and reset['resident_pages_after']==0
plan=TOP/'plan.json'
if not plan.exists():subprocess.run(['python3','scripts/bench/serve_realistic.py','plan','--family','v41','--seconds','120','--seed','20261112','--out',str(plan)],check=True,timeout=10)
binary=TOP/'pins/qwen';tok=TOP/'pins/qwen-tok'
command=['python3','scripts/bench/serve_realistic.py','run','--binary',str(binary),'--tokenizer',str(tok),'--model',str(model),'--corpus',str(corpus),'--plan',str(plan),'--out',str(out/'serve'),'--agents','3','--arm','default','--warm-tokens','49152','--context','196608','--warm-reply-tokens','32','--sampling','greedy','--cool','0','--ramp','15','--timeout','650','--port','19987','--max-thermal-pressure','2','--warmup-max-thermal-pressure','2','--settle-thermal-seconds','60','--settle-max-thermal-pressure','1','--keep-bulk','--harness-commit','3482f60cb','--env','QWEN_SERVE_TTFT_TIMING=1','--env','QWEN_SERVE_V41_PREFILL_TIMING=1','--env','QWEN_SERVE_V41_MAX_ACTIVE=4','--env','QWEN_SERVE_V41_SAMPLE_WORKERS=3','--env','QWEN_SERVE_V41_PREFILL_FAIR=1']
if a.mode=='baseline':command+=['--env','QWEN_SERVE_ASYNC_PREPARE=0','--env','QWEN_SERVE_TOKEN_PREFIX_CACHE=0']
(out/'command.json').write_text(json.dumps(command,indent=2))
monitor_stop=threading.Event()
def monitor():
    with (out/'host-load.jsonl').open('w') as f:
        while not monitor_stop.is_set():
            f.write(json.dumps(dict(time=time.time(),load=os.getloadavg()))+'\n');f.flush();monitor_stop.wait(1)
watch=threading.Thread(target=monitor);watch.start()
try:pids,nice=pair.run_harness(command,out/'arm.log',binary,deadline,fd)
finally:monitor_stop.set();watch.join(timeout=5)

io=[json.loads(l) for l in (out/'arm.process-io.jsonl').read_text().splitlines()]
result=dict(tag=a.tag,mode=a.mode,start=start,end=time.time(),model_pids=sorted(pids),model_niceness=nice,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),purge=reset,process_disk_read_bytes=max((r['disk_read'] for r in io),default=None),symmetric_warmup=warm,summary=json.loads((out/'serve/summary.json').read_text()))
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'load-after.txt').write_text(subprocess.check_output(['uptime'],text=True,timeout=5));(out/'processes-after.txt').write_text(pair.resident_check());print(json.dumps(result),flush=True)
