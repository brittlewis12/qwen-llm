#!/usr/bin/env python3
"""One interleaved pair under an existing <=15-minute Metal lease.

ROOT BINARY MODEL CORPUS PLAN PAIR (0..2). Builds must precede this command.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root,binary,model,corpus,plan=map(Path,sys.argv[1:6])
pair=int(sys.argv[6])
if pair not in (0,1,2):
    raise ValueError('pair must be zero, one or two')
fd=int(os.environ['QWEN_METAL_LEASE_FD'])
os.fstat(fd)
order=(0,2) if pair!=1 else (2,0)
started=time.monotonic()
meta=dict(pair=pair,order=order,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),profile='iter',
          load_before=os.getloadavg(),started_unix=time.time(),metric='delivered aggregate_tps',
          gate='geometric mean of three paired candidate/reference rates >= 1.15; report all individual pairs')
(root/f'pair-{pair}-launch.json').write_text(json.dumps(meta,indent=2)+'\n')
for index,k in enumerate(order):
    ps=subprocess.check_output(['ps','-axo','pid,comm'],text=True)
    resident=[line for line in ps.splitlines() if line.split() and Path(line.split()[-1]).name=='qwen']
    if resident:
        raise RuntimeError(f'resident qwen process before arm: {resident}')
    out=root/f'pair-{pair}-{index}-k{k}'
    command=[sys.executable,'scripts/bench/serve_realistic.py','run','--binary',str(binary),
             '--tokenizer',str(binary.with_name('qwen-tok')),'--model',str(model),'--corpus',str(corpus),
             '--plan',str(plan),'--out',str(out),'--arm','default','--agents','1','--sampling','agent',
             '--warm-tokens','49152','--context','196608','--port','18787','--ramp','10','--cool','0',
             '--timeout','400','--settle-thermal-seconds','60','--settle-max-thermal-pressure','1',
             '--max-thermal-pressure','2','--warmup-max-thermal-pressure','2','--keep-bulk',
             '--harness-commit',meta['source_commit'],'--env',f'QWEN_GLM53_SOLO_NEXTN={k}',
             '--env',f'QWEN_GLM53_NEXTN_CACHE_DIR={root / "head-cache"}']
    print(json.dumps(dict(event='arm_start',pair=pair,k=k,command=command,elapsed_s=time.monotonic()-started)),flush=True)
    # The harness handles SIGINT/TERM by stopping its exact server PID.
    subprocess.run(command,pass_fds=(fd,),check=True)
meta.update(elapsed_s=time.monotonic()-started,load_after=os.getloadavg(),complete=True)
(root/f'pair-{pair}-launch.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(dict(event='pair_complete',pair=pair,elapsed_s=meta['elapsed_s'])),flush=True)
