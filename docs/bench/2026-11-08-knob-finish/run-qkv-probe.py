import hashlib,json,os,re,signal,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path.cwd();BASE=ROOT/'bench/results/knob-finish';label=sys.argv[1]
log=BASE/('qkv-build-'+label+'.log')
match=re.findall(r'Executable unittests src/lib.rs \(([^)]*qwen_llm-[^)]+)\)',log.read_text());assert match
binary=ROOT/match[-1]
OUT=BASE/('qkv-'+label+'-'+time.strftime('%Y%m%d-%H%M%S'));OUT.mkdir()
fd=int(os.environ['QWEN_METAL_LEASE_FD']);os.fstat(fd)
def conditions():
 pressure=int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'],text=True).split()[-1])
 r=subprocess.run(['pgrep','-x','rustc'],capture_output=True,text=True);assert r.returncode in (0,1)
 return {'thermal':pressure,'rustc':[int(p) for p in r.stdout.split()]}
quiet=None;waiting=time.monotonic()
while True:
 now=time.monotonic();row=conditions()
 if now-waiting>120:sys.exit(75)
 if row['thermal']==0 and not row['rustc']:
  if quiet is None:quiet=now
  if now-quiet>=25:break
 else:quiet=None
 print(json.dumps({'record':'cooling',**row}),flush=True);time.sleep(5)
env={k:v for k,v in os.environ.items() if not k.startswith('QWEN_')}
assert all(k not in env for k in ['MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'])
env.update(QWEN_METAL_LEASE_FD=str(fd),QWEN_METAL_LEASE_WAIT='1',QWEN_MODEL_SEARCH_PATH='/Users/tito/models',QWEN_A3B_MOE_MODEL='/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',QWEN_PREFILL_ATTN_FUSED_QKV_G8='1',RUST_TEST_THREADS='1')
manifest={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'binary':str(binary.relative_to(ROOT)),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'machine':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'sdk':subprocess.check_output(['xcrun','--show-sdk-version'],text=True).strip(),'samples':[]}
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
child=None;stop=threading.Event()
def observe():
 while not stop.is_set():
  manifest['samples'].append({'unix':time.time(),**conditions()});stop.wait(.5)
def interrupt(sig,frame):
 if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
 raise KeyboardInterrupt
signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
watch=threading.Thread(target=observe);watch.start()
try:
 with (OUT/'stdout.log').open('w') as out,(OUT/'stderr.log').open('w') as err:
  child=subprocess.Popen([str(binary),'--exact','metal_forward::fused_qkv_probe::real_a3b_q8_bank_scratch_and_projection_cost','--ignored','--nocapture'],env=env,stdout=out,stderr=err,pass_fds=(fd,))
  try:code=child.wait(timeout=570)
  except subprocess.TimeoutExpired:
   child.send_signal(signal.SIGINT);child.wait();raise
 manifest['exit_code']=code
 assert code==0,(code,str(OUT/'stderr.log'))
 assert 'test result: ok. 1 passed; 0 failed;' in (OUT/'stdout.log').read_text()
 records=[]
 for line in (OUT/'stdout.log').read_text().splitlines():
  start=line.find('{')
  if start>=0:
   try:r=json.loads(line[start:])
   except ValueError:continue
   if isinstance(r,dict) and 'record' in r:records.append(r)
 assert len([r for r in records if r['record']=='bank'])==1
 assert len([r for r in records if r['record']=='scratch'])==2
 (OUT/'records.json').write_text(json.dumps(records,indent=2)+'\n')
 assert all(r['thermal']==0 and not r['rustc'] for r in manifest['samples']),'conditions changed'
 manifest['success']=True
 print(json.dumps({'out':str(OUT),'records':len(records),'success':True}),flush=True)
finally:
 stop.set();watch.join()
 if child is not None and child.poll() is None:child.send_signal(signal.SIGINT);child.wait()
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
