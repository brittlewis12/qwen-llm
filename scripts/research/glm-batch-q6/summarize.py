#!/usr/bin/env python3
"""Summarize the committed glm_batch_q6 operator sweep without retaining traces."""
import argparse
import collections
import json
import statistics as st
from pathlib import Path


def summarize(path):
    rows=[]
    for line in path.read_text().splitlines():
        try: row=json.loads(line)
        except ValueError: continue
        if isinstance(row,dict) and 'event' in row: rows.append(row)
    if not rows or rows[-1]['event']!='complete': raise ValueError('incomplete sweep')
    timings=collections.defaultdict(list)
    for row in rows:
        if row['event']=='timing' and row['rep']>=0:
            timings[row['k'],row['m'],row['b'],row['kernel']].append(row)
    cells=[]
    for case in (r for r in rows if r['event']=='case'):
        k,m,b=case['k'],case['m'],case['b']
        sg=16 if m<=6144 else 8
        while sg>8 and sg>k//64: sg//=2
        nt=4
        while nt>1 and nt>min(16384//(sg*256),m//1024): nt//=2
        baseline=f'kernel_mat_mat_q6_K_fewrow_gen_f32w_nt{nt}_rt1_nsg{sg}_f32'
        configs=[]
        for key,values in timings.items():
            if key[:3]!=(k,m,b): continue
            gpu=[v['gpu_us'] for v in values]
            configs.append(dict(kernel=key[3],n=len(gpu),mean_us=st.mean(gpu),median_us=st.median(gpu),sd_us=st.stdev(gpu),min_us=min(gpu),max_us=max(gpu),GBps=case['bytes_per_matrix']/(st.mean(gpu)*1e3),f64_rms_mean=st.mean(v['f64_relative_rms']for v in values),f64_rms_max=max(v['f64_relative_rms']for v in values)))
        configs.sort(key=lambda v:v['mean_us']);base=next(c for c in configs if c['kernel']==baseline)
        best=configs[0]
        cells.append(dict(**case,baseline=baseline,baseline_us=base['mean_us'],best=best['kernel'],best_us=best['mean_us'],weighted_saved_ms=case['matrices']*(base['mean_us']-best['mean_us'])/1000,configs=configs))
    totals=[]
    for b in [2,3,4]:
        subset=[c for c in cells if c['b']==b]
        base=sum(c['matrices']*c['baseline_us']/1000 for c in subset)
        best=sum(c['matrices']*c['best_us']/1000 for c in subset)
        totals.append(dict(b=b,baseline_estimate_ms=base,best_estimate_ms=best,saved_estimate_ms=base-best))
    return dict(source=rows[0],totals=totals,cells=cells,note='Isolated operators, not an additive full-round attribution; full-model kill gate is >=2 ms at B=4. All repeats retained; two warm iterations excluded.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--out',type=Path);p.add_argument('--discard-input',action='store_true');a=p.parse_args()
    v=summarize(a.input)
    if a.out: a.out.write_text(json.dumps(v,indent=2)+'\n')
    for t in v['totals']:print(json.dumps(t))
    for c in v['cells']:
        if c['b']==4:print(json.dumps({k:c[k]for k in ['k','m','matrices','baseline_us','best','best_us','weighted_saved_ms']}))
    if a.discard_input:
        if not a.out: p.error('--discard-input needs --out')
        a.input.unlink()
if __name__=='__main__': main()
