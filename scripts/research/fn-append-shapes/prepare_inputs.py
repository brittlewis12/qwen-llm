#!/usr/bin/env python3
"""Regenerate frozen code/prose inputs and plans; keep recipe, delete text bulk."""
import argparse,hashlib,json,subprocess
from pathlib import Path

PREFIX_REV='dabb97021e65f192a15057b3f226281cbe1ad81f'
BASE='2e2f14e6dd95ba355d198bdf86cb3d0f26b1e01d'
PREFIX_FILES=['crates/qwen-llm/src/deepseek_v41/forward/gpu_step.rs',
 'crates/qwen-llm/src/deepseek_v41/forward/attention_gpu.rs',
 'crates/qwen-llm/src/deepseek_v41/forward/ffn_gpu.rs',
 'crates/qwen-llm/src/glm5_next_metal.rs','crates/qwen-llm/src/qwen4exp_runtime.rs']
DOCS=[('runtime','crates/qwen-llm/src/qwen4exp_runtime.rs'),
 ('text-session','crates/qwen-llm/src/qwen4exp_text_session.rs'),
 ('experts','crates/qwen-llm/src/qwen4exp_moe.rs'),
 ('serve','crates/qwen-cli/src/serve/backend_qwen4exp.rs')]+[(x,'docs/bench/2026-10-09-flash-frontier-quality/'+x+'.txt') for x in ('bash','curl','find','tar')]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False)
 def blob(rev,path):return subprocess.check_output(['git','show',rev+':'+path])
 prefix=b''.join(('\n\n// SOURCE: '+f+'\n\n').encode()+blob(PREFIX_REV,f) for f in PREFIX_FILES)
 assert hashlib.sha256(prefix).hexdigest()=='cf2059d77a3b3e70d8116b3e445c94e00aa515e5d792e94e6966def38e142969'
 (a.out/'prefix.txt').write_bytes(prefix)
 documents=[]
 for name,path in DOCS:
  (a.out/(name+'.txt')).write_bytes(blob(BASE,path))
  documents.append(dict(id=name,path=str((a.out/(name+'.txt')).resolve()),offset=128))
 for mode,depth,rows in [('timing',65536,[400,512,2000,2048]),('quality',65536,[400,2000]),('quality',8192,[400,2000]),('quality',8693,[400,2000])]:
  plan=dict(mode=mode,depth=depth,rows=rows,prefix=str((a.out/'prefix.txt').resolve()),targets=128,documents=documents)
  (a.out/f'{mode}-{depth}.json').write_text(json.dumps(plan,indent=2)+'\n')
 (a.out/'recipe.json').write_text(json.dumps(dict(prefix_revision=PREFIX_REV,prefix_files=PREFIX_FILES,document_revision=BASE,documents=DOCS),indent=2)+'\n')

if __name__=='__main__':main()
