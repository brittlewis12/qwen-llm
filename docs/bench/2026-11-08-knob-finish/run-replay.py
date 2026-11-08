import hashlib,json,os,re,signal,socket,subprocess,sys,threading,time,urllib.request
from pathlib import Path
ROOT=Path.cwd(); BASE=ROOT/'bench/results/knob-finish'; OUT=BASE/('replay-'+time.strftime('%Y%m%d-%H%M%S'));OUT.mkdir()
fd=int(os.environ['QWEN_METAL_LEASE_FD']);os.fstat(fd)
BIN=ROOT/'target/release/qwen'; server=None;stop=threading.Event();samples=[]
def conditions():
 thermal=int(subprocess.check_output(['notifyutil','-g','com.apple.system.thermalpressurelevel'],text=True).split()[-1])
 p=subprocess.run(['pgrep','-x','rustc'],capture_output=True,text=True);assert p.returncode in (0,1)
 return {'thermal':thermal,'rustc':p.stdout.split()}
def interrupt(sig,frame):
 if server and server.poll() is None:server.send_signal(signal.SIGINT)
 raise KeyboardInterrupt
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
manifest={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'binary_sha256':hashlib.sha256(BIN.read_bytes()).hexdigest(),'machine':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'sdk':subprocess.check_output(['xcrun','--show-sdk-version'],text=True).strip(),'packets':[]}
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
def observe():
 while not stop.is_set():samples.append({'unix':time.time(),**conditions()});stop.wait(.5)
thread=threading.Thread(target=observe);thread.start()
A='Write a Python function merge_sorted(a, b) that merges two sorted lists in linear time. Return only code.'
B='Write a Python function binary_search(xs, target) that returns the index or -1. Return only code.'
def post(port,prompt,temperature=0.0):
 body={'model':model_id,'input':prompt,'temperature':temperature,'top_p':1.0,'max_output_tokens':64,'stream':False,'x_qwen':{'no_thinking':True,'stats':True,'seed':42}}
 t=time.perf_counter();req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/responses',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:response=json.load(r)
 elapsed=time.perf_counter()-t
 texts=[{'type':x['type'],'content':x.get('content')} for x in response.get('output',[])]
 return response,elapsed,hashlib.sha256(json.dumps(texts,sort_keys=True).encode()).hexdigest()
try:
 for packet,arm in enumerate(['off','on','on','off']):
  quiet=None;deadline=time.monotonic()+120
  while True:
   now=time.monotonic();c=conditions()
   if c['thermal']==0 and not c['rustc']:
    if quiet is None:quiet=now
    if now-quiet>=25:break
   else:quiet=None
   assert now<deadline,'machine not quiet';time.sleep(5)
  sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
  log=OUT/f'{packet}-{arm}.log';env={k:v for k,v in os.environ.items() if not k.startswith('QWEN_')}
  env.update(QWEN_METAL_LEASE_FD=str(fd),QWEN_METAL_LEASE_WAIT='1',RUST_LOG='info')
  if arm=='off':env['QWEN_DFLASH_PREFIX_REPLAY']='0'
  command=[str(BIN),'serve','-m','/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf','--drafter','/Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q8_0.gguf','--addr',f'127.0.0.1:{port}','--max-context-tokens','4096','--snapshot-cache-mib','1024','--max-tokens','64']
  started=time.perf_counter()
  with log.open('w') as stream:
   server=subprocess.Popen(command,env=env,stdout=stream,stderr=stream,pass_fds=(fd,))
   deadline=time.monotonic()+240
   while 'serve: listening on' not in log.read_text(errors='replace'):
    assert server.poll() is None,log.read_text(errors='replace')[-2500:]
    assert time.monotonic()<deadline,'server startup timed out';time.sleep(.5)
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=5) as r:model_id=json.load(r)['data'][0]['id']
   entry={'packet':packet,'arm':arm,'startup_s':time.perf_counter()-started,'requests':[]}
   for i,prompt in enumerate([A,A,A,B,A,A]):
    offset=log.stat().st_size
    response,elapsed,sha=post(port,prompt)
    (OUT/f'{packet}-{arm}-{i}.json').write_text(json.dumps(response,indent=2)+'\n')
    with log.open('rb') as tail:
     tail.seek(offset);segment=tail.read().decode(errors='replace')
    replay=re.findall(r'prefix_replay=(\d+)/(\d+)/(\d+)/(\d+)',segment)
    phases=[line for line in segment.splitlines() if 'serve phases:' in line]
    row={'index':i,'prompt':'A' if prompt==A else 'B','wall_s':elapsed,'output_sha256':sha,'usage':response.get('usage'),'status':response.get('status'),'prefix_replay':replay,'phases':phases}
    entry['requests'].append(row);print(json.dumps({'packet':packet,'arm':arm,**row}),flush=True)
   offset=log.stat().st_size
   sampled=post(port,A,.7)[0]
   with log.open('rb') as tail:
    tail.seek(offset);segment=tail.read().decode(errors='replace')
   entry['sampled_replay']=re.findall(r'prefix_replay=(\d+)/(\d+)/(\d+)/(\d+)',segment)
   assert all(x==('0','0','0','0') for x in entry['sampled_replay']),entry['sampled_replay']
   assert all(c['thermal']==0 and not c['rustc'] for c in samples),'conditions changed'
   manifest['packets'].append(entry)
   server.send_signal(signal.SIGINT);server.wait(timeout=30);server=None
  (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 hashes=[{(x['index'],x['output_sha256']) for x in p['requests']} for p in manifest['packets']]
 assert all(h==hashes[0] for h in hashes),'greedy output differs'
 for packet in manifest['packets']:
  counts=[tuple(map(int,x)) for row in packet['requests'] for x in row['prefix_replay']]
  assert counts,'no DFlash counters'
  assert any(x[0]>0 for x in counts) if packet['arm']=='on' else all(x[0]==0 for x in counts)
 manifest['success']=True;print(json.dumps({'out':str(OUT),'success':True}),flush=True)
finally:
 if server and server.poll() is None:
  server.send_signal(signal.SIGINT)
  try:server.wait(timeout=30)
  except subprocess.TimeoutExpired:server.kill();server.wait()
 stop.set();thread.join();manifest['samples']=samples
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
