#!/usr/bin/env python3
"""Compact whole-round counterfactual bounds; no additive attribution claim."""
import argparse,collections,json,statistics as st
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=[]
for line in a.input.read_text().splitlines():
 try:r=json.loads(line)
 except ValueError:continue
 if isinstance(r,dict)and'kind'in r:rows.append(r)
assert rows[-1]['kind']=='complete','incomplete probe'
header=rows[0];valid_bytes=not header.get('commit','').startswith(('c96796e','63fa881'));groups=header['groups'];native=[x for r in rows if r['kind']=='native'for x in r['wall_ms']];native_mean=st.mean(native)
means=lambda rs,key='gpu_ms':dict(mean=st.mean(r[key]for r in rs),median=st.median(r[key]for r in rs),sd=st.stdev(r[key]for r in rs)if len(rs)>1 else 0,n=len(rs))
refs={k:{(r['phase'],r['rep']):r['gpu_ms']for r in rows if r['kind']==k and r['group']=='reference'}for k in ['hold','private_skip']}
cells=[]
for i,g in enumerate(groups):
 cell=dict(group=g)
 for k in ['hold','private_skip']:
  rs=[r for r in rows if r['kind']==k and r['group']==g];delta=[refs[k][r['phase'],r['rep']]-r['gpu_ms']for r in rs]
  cell[k]=dict(**means(rs),saved_ms=st.mean(delta),paired_min=min(delta),paired_max=max(delta))
 ws=[r['bounds']for r in rows if r['kind']=='weight_bytes'];cell['readonly_logical_bytes']=st.mean(r['logical'][i]for r in ws);cell['readonly_unique_bytes']=st.mean(r['unique'][i]for r in ws)
 cell['unique_weight_floor_ms_at_1175GBps']=cell['readonly_unique_bytes']/1175.47/1e6 if valid_bytes else None
 if not valid_bytes:cell['readonly_logical_bytes']=cell['readonly_unique_bytes']=None
 cell['copy_output_bytes']=st.mean(r['group_bytes'][i]for r in rows if r['kind']=='hold_storage')
 cells.append(cell)
print('native wall ms',native_mean)
print('group\thold saved ms\tprivate skip saved ms\tunique readonly weight floor ms')
for c in sorted(cells,key=lambda c:c['hold']['saved_ms'],reverse=True):print(c['group'],round(c['hold']['saved_ms'],3),round(c['private_skip']['saved_ms'],3),round(c['unique_weight_floor_ms_at_1175GBps'],3)if valid_bytes else 'invalid-stale-bindings',sep='\t')
concurrency={}
for name,rs in [('causal_serial',[r for r in rows if r['kind']=='causal'and not r['dag']]),('causal_ordered',[r for r in rows if r['kind']=='causal'and r['dag']]),('causal_wavefront',[r for r in rows if r['kind']=='wavefront']),('causal_members',[r for r in rows if r['kind']=='member_waves'])]+[('private_'+m,[r for r in rows if r['kind']=='private_concurrency'and r['mode']==m])for m in ['serial','members','global']]:
 if rs:concurrency[name]=means(rs)
print('concurrency',json.dumps(concurrency))
summary=dict(header=header,readonly_bytes_valid=valid_bytes,native_wall_ms=native_mean,native_sd_ms=st.stdev(native),groups=cells,concurrency=concurrency,after=next(r for r in rows if r['kind']=='after'),note='Stage removals and concurrency bounds overlap. Held writes preserve real inputs; copying full cache/state can make holds pessimistic. Private controls change buffer/cache layout; causal wavefront retains all captured hazards. Weights are useful-byte bounds, not DRAM counters.')
a.out.write_text(json.dumps(summary,indent=2)+'\n')
