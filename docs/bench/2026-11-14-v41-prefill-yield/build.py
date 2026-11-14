import hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
top=Path('/Users/tito/qwen-runs/2026-11-12-v41-prefill/row-yield')
repo=Path('/Users/tito/code/qwen-llm-v41-yield')
fd=int(os.environ['QWEN_METAL_LEASE_FD']);started=time.time();deadline=time.monotonic()+800
pins=top/'pins';pins.mkdir(exist_ok=True);(top/'build-tmp').mkdir(exist_ok=True)
env=dict(os.environ,CARGO_TARGET_DIR=str(top/'target'),TMPDIR=str(top/'build-tmp'))
records=[]
def build(args,name):
    with (top/name).open('w') as log:
        child=subprocess.Popen(['nice','-n','10','cargo',*args,'--message-format=json'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,))
        records.append(dict(pid=child.pid,command=args,nice=10,log=name));(top/'build-pids.json').write_text(json.dumps(records,indent=2))
        try:
            while child.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('build deadline')
                time.sleep(.5)
        finally:
            if child.poll() is None:
                rows=[l.split(None,2) for l in subprocess.check_output(['ps','-axo','pid=,ppid=,comm='],text=True,timeout=5).splitlines()]
                children={child.pid};changed=True
                while changed:
                    before=len(children);children.update(int(p) for p,parent,*_ in rows if int(parent) in children);changed=before!=len(children)
                for pid in sorted(children,reverse=True):
                    try:os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                child.wait(timeout=10)
        if child.returncode:raise RuntimeError('build failed '+name)
    messages=[]
    for line in (top/name).read_text().splitlines():
        try:messages.append(json.loads(line))
        except ValueError:pass
    return messages
normal=build(['build','--offline','--locked','--release','-p','qwen-cli','--bin','qwen','--bin','qwen-tok'],'release-build.log')
for name in ('qwen','qwen-tok'):shutil.copy2(top/'target/release'/name,pins/name)
tests=build(['test','--offline','--locked','-p','qwen-cli','--bin','qwen','--no-run'],'test-build.log')
exe={m['executable'] for m in tests if m.get('reason')=='compiler-artifact' and m.get('target',{}).get('name')=='qwen' and m.get('profile',{}).get('test') and m.get('executable')}
assert len(exe)==1;shutil.copy2(next(iter(exe)),pins/'qwen-cpu-tests')
witness=build(['build','--offline','--locked','--release','-p','qwen-llm','--features','perf-ablations','--example','v41_prefill_yield_witness'],'witness-build.log')
shutil.copy2(top/'target/release/examples/v41_prefill_yield_witness',pins/'v41_prefill_yield_witness')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def libs(messages):
    dirs=[Path(m['out_dir']) for m in messages if m.get('reason')=='build-script-executed' and m.get('package_id','').split('#')[-1].startswith('qwen-llm@')]
    if not dirs:dirs=[Path(m['out_dir']) for m in messages if m.get('reason')=='build-script-executed' and 'qwen-llm' in m.get('package_id','')]
    return {p.name:sha(p) for d in dirs for p in d.glob('*.metallib')}
a,b=libs(normal),libs(witness);assert a and a==b,(a,b)
record=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),profile='release',witness_features=['perf-ablations'],cpu_tests_profile='test',start=started,end=time.time(),binaries={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in pins.iterdir()},serve_metallibs=a,witness_metallibs=b,metallibs_identical=True)
(top/'pin-manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
