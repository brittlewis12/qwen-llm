import hashlib,json,os,re,signal,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path.cwd();BASE=ROOT/'bench/results/knob-finish';OUT=BASE/('lightning-'+time.strftime('%Y%m%d-%H%M%S'));OUT.mkdir()
log=(BASE/'lightning-build.log').read_text();match=re.findall(r'Executable unittests src/lib.rs \(([^)]*qwen_llm-[^)]+)\)',log);assert match
binary=ROOT/match[-1];fd=int(os.environ['QWEN_METAL_LEASE_FD']);os.fstat(fd)
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
manifest={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'machine':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'samples':[]}
def conditions():
 p=int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'],text=True).split()[-1]);r=subprocess.run(['pgrep','-x','rustc'],capture_output=True,text=True);assert r.returncode in (0,1);return {'thermal':p,'rustc':r.stdout.split()}
quiet=None;deadline=time.monotonic()+120
while True:
 now=time.monotonic();c=conditions()
 if c['thermal']==0 and not c['rustc']:
  if quiet is None:quiet=now
  if now-quiet>=25:break
 else:quiet=None
 assert now<deadline,'machine not quiet';time.sleep(5)
env={k:v for k,v in os.environ.items() if not k.startswith('QWEN_')}
env.update(QWEN_METAL_LEASE_FD=str(fd),QWEN_METAL_LEASE_WAIT='1',QWEN_REQUIRE_METAL_TESTS='1',RUST_TEST_THREADS='1',DSV4_LEGACY_MODEL='/Volumes/wdblack/weights-archive/deepseek-v4-flash-0731-ud-iq3_xxs/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf')
child=None;stop=threading.Event()
def observe():
 while not stop.is_set():manifest['samples'].append({'unix':time.time(),**conditions()});stop.wait(2)
def interrupt(sig,frame):
 if child and child.poll() is None:child.send_signal(signal.SIGINT)
 raise KeyboardInterrupt
signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
thread=threading.Thread(target=observe);thread.start()
try:
 with (OUT/'stdout.log').open('w') as out,(OUT/'stderr.log').open('w') as err:
  child=subprocess.Popen([str(binary),'--exact','deepseek_v4_metal::tests::lightning_real::nonzero_long_context_matrix_score_replay','--ignored','--nocapture'],stdout=out,stderr=err,env=env,pass_fds=(fd,))
  try:code=child.wait(timeout=4500)
  except subprocess.TimeoutExpired:child.send_signal(signal.SIGINT);child.wait();raise
 manifest['test_exit_code']=code
 lines=(OUT/'stderr.log').read_text(errors='replace').splitlines();summary=[x for x in lines if 'lightning_real summary' in x];manifest['summary']=summary
 manifest['thermal_levels']=sorted(set(x['thermal'] for x in manifest['samples']));manifest['rustc_observed']=any(x['rustc'] for x in manifest['samples'])
 manifest['qualification_passed']=code==0
 assert summary,'probe did not reach causal comparison'
 assert not manifest['rustc_observed'],'build disturbed run'
 print(json.dumps({'out':str(OUT),'qualification_passed':code==0,'summary':summary,'thermal_levels':manifest['thermal_levels']}),flush=True)
finally:
 if child and child.poll() is None:child.send_signal(signal.SIGINT);child.wait()
 stop.set();thread.join();(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
