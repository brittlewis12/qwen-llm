#!/usr/bin/env python3
"""Run every prebuilt release CLI target under one lease, without a build lock."""
import argparse,json,os,re,subprocess
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-log',type=Path)
parser.add_argument('--lib-build-log',type=Path)
args=parser.parse_args()
root=Path.cwd();fd=int(os.environ['QWEN_METAL_LEASE_FD'])
meta=json.loads(subprocess.check_output(['cargo','metadata','--offline','--no-deps','--format-version','1'],text=True));pkg=next(p for p in meta['packages']if p['name']=='qwen-cli')
expected={t['name'].replace('-','_')for t in pkg['targets']if t['test']and t['kind'][0]in ['lib','bin','test']}
found={}
if args.build_log:
 for path in re.findall(r'Executable .*\((target/release/deps/[^)]+)\)',args.build_log.read_text()):
  exe=root/path;name=exe.name.rsplit('-',1)[0];assert name not in found and exe.exists(),(name,exe);found[name]=exe
else:
 for j in (root/'target/release/.fingerprint').glob('qwen-cli-*/test-*.json'):
  for prefix in ['test-lib-','test-bin-','test-integration-test-']:
   if j.stem.startswith(prefix):
    name=j.stem.removeprefix(prefix).replace('-','_');exe=root/'target/release/deps'/(name+'-'+j.parent.name.removeprefix('qwen-cli-'))
    if exe.exists()and name in expected:
     assert name not in found,(name,found.get(name),exe);found[name]=exe
assert set(found)==expected,(expected,set(found))
results=[]
if args.lib_build_log:
 paths=re.findall(r'Executable .*\((target/release/deps/[^)]+)\)',args.lib_build_log.read_text())
 assert len(paths)==1,paths
 exe=root/paths[0];lib=next(p for p in meta['packages']if p['name']=='qwen-llm')
 print(json.dumps({'target':'qwen_llm','argv':[str(exe),'--test-threads=1']}),flush=True)
 r=subprocess.run([str(exe),'--test-threads=1'],cwd=Path(lib['manifest_path']).parent,pass_fds=(fd,));results.append(('qwen_llm',r.returncode))
 assert r.returncode==0

for name,exe in sorted(found.items()):
 print(json.dumps({'target':name,'argv':[str(exe),'--test-threads=1']}),flush=True)
 r=subprocess.run([str(exe),'--test-threads=1'],cwd=Path(pkg['manifest_path']).parent,pass_fds=(fd,));results.append((name,r.returncode))
print(json.dumps({'results':results}),flush=True);assert all(code==0 for _,code in results)
