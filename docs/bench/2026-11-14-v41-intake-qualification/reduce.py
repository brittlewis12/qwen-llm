#!/usr/bin/env python3
"""Apply the existing frozen serve reduction and expose per-arm root/host costs."""
import argparse,importlib.util,json,math,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts/bench'))
from v41_prefill_serve_compare import reduce
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
ANALYZER=ROOT/'docs/bench/2026-11-14-serve-ttft-instrumentation/analyze.py'
def percentile(xs,f):
    xs=sorted(xs);return xs[min(len(xs)-1,math.ceil(len(xs)*f)-1)] if xs else None
def events(path):
    out=[]
    for l in path.read_text().splitlines():
        if l.startswith('{'):
            try:out.append(json.loads(l))
            except ValueError:pass
    return out
arms={}
for tag in ('A0','A1','B1','B2','A2','A3','B3'):
    src=a.runs/tag
    if not (src/'result.json').exists():continue
    r=json.loads((src/'result.json').read_text());subprocess.run([sys.executable,str(ANALYZER),str(src/'serve'),'--out',str(a.out/tag)],check=True,stdout=subprocess.DEVNULL)
    timing=json.loads((a.out/tag/'summary.json').read_text());warm=[json.loads(l) for l in (src/'serve/warm-calls.jsonl').read_text().splitlines()];load=[json.loads(l) for l in (src/'host-load.jsonl').read_text().splitlines()]
    roots=[]
    for call in warm:
        ls=[l['load'][0] for l in load if call['sent']<=l['time']<=(call['first'] or call['end'])]
        roots.append(dict(label=call['label'],client_ttft_ms=(call['first']-call['sent'])*1000 if call['first'] else None,native_prefill_ms=call['x_qwen']['phases']['prefill_ms'],load1_min=min(ls) if ls else None,load1_p50=percentile(ls,.5),load1_max=max(ls) if ls else None))
    io=[json.loads(l) for l in (src/'arm.process-io.jsonl').read_text().splitlines()];window=[v for v in io if r['summary']['window_begin']<=v['time']<=r['summary']['window_end']]
    disk_window=dict(bytes=window[-1]['disk_read']-window[0]['disk_read'],begin=window[0]['time'],end=window[-1]['time'],samples=len(window)) if len(window)>1 else None
    arms[tag]=dict(sampled_window_disk_read=disk_window,mode=r['mode'],binary_sha256=r['binary_sha256'],tps=r['summary']['aggregate_tps'],roots=roots,process_disk_read_bytes=r['process_disk_read_bytes'],thermal_max=r['summary']['thermal_max'],compile_overlapped=r['summary']['compile_overlapped'],classes={k:dict(n=v['n'],outside_native=v['metrics'].get('outside_native'),client_ttft=v['metrics'].get('client_ttft'),native_prefill=v['metrics'].get('native_prefill')) for k,v in timing['distributions'].items()},censored=timing['censored'],missing=timing['missing'],spans_dropped=timing['trace_spans_dropped'])
pairs=[]
for number,base,candidate,order in [(1,'A1','B1','ab'),(2,'A2','B2','ba'),(3,'A3','B3','ab')]:
    if base not in arms or candidate not in arms:continue
    tag=f'pair{number}';folder=a.out/'frozen';folder.mkdir(exist_ok=True)
    for role,src in [('a',base),('b',candidate)]:
        dest=folder/f'{tag}-{role}'
        if not dest.exists():dest.symlink_to((a.runs/src/'serve').resolve(),target_is_directory=True)
    meta=dict(order=order,binary_sha256=arms[base]['binary_sha256'],arms=[])
    assert arms[base]['binary_sha256']==arms[candidate]['binary_sha256']
    for role,src in [('a',base),('b',candidate)]:
        r=json.loads((a.runs/src/'result.json').read_text());meta['arms'].append(dict(arm=role,summary=r['summary'],root_turns=[e for e in events(a.runs/src/'arm.log') if e.get('event')=='turn' and e.get('label','').endswith('-warm')]))
    (folder/f'{tag}-summary.json').write_text(json.dumps(meta))
    reduced=reduce(folder,tag);(a.out/f'{tag}.json').write_text(json.dumps(reduced,indent=2)+'\n')
    ratios=dict(tps=arms[candidate]['tps']/arms[base]['tps'],root_prefill=reduced['root_prefill_ratio'],short_prefill=reduced['bundle_short']['prefill_ratio'],short_ttft=reduced['bundle_short']['ttft_ratio'],one_k_prefill=reduced['bundle_1k']['prefill_ratio'],one_k_ttft=reduced['bundle_1k']['ttft_ratio'])
    ratios['root_client_ttft']=sum(r['client_ttft_ms'] for r in arms[base]['roots'])/sum(r['client_ttft_ms'] for r in arms[candidate]['roots'])
    pairs.append(dict(pair=number,order=order,baseline=base,candidate=candidate,ratios=ratios,matches=dict(short=reduced['bundle_short'],one_k=reduced['bundle_1k'])))
thresholds=dict(tps=.98,root_prefill=.98,short_prefill=1.,short_ttft=1.,one_k_prefill=1.,one_k_ttft=1.)
summary=dict(arms=arms,pairs=pairs,complete=len(pairs)==3,gate={})
if len(pairs)==3:
    for k,t in thresholds.items():
        xs=[r['ratios'][k] for r in pairs];gm=math.exp(sum(math.log(x) for x in xs)/3) if all(x is not None and x>0 for x in xs) else None
        summary['gate'][k]=dict(geomean=gm,threshold=t,pass_=gm is not None and gm>=t,pairs=xs)
    summary['performance_pass']=all(v['pass_'] for v in summary['gate'].values())
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
