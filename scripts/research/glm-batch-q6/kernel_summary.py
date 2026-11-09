#!/usr/bin/env python3
"""Compact whole-round ABBA, actual routed-byte union and identity receipts."""
import argparse
import json
import statistics as st
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--discard-input',action='store_true');a=p.parse_args()
    rows=[]
    for line in a.input.read_text().splitlines():
        try:r=json.loads(line[line.index('{'):])
        except (ValueError,TypeError):continue
        if isinstance(r,dict)and'event'in r:rows.append(r)
    if rows[-1]['event']!='complete':raise ValueError('incomplete packet')
    q=[r for r in rows if r['event']=='quality'];assert len(q)==128 and all(r['bitwise']for r in q)
    membership=next(r for r in rows if r['event']=='membership');assert membership['state_and_logits_bitwise']
    result=dict(header=rows[0],quality_vectors=sum(r['vectors']for r in q),dnll=0,kl=0,membership=membership,census=[r for r in rows if r['event']=='census'],cells=[])
    control=rows[0].get('control',0);candidate=rows[0].get('candidate',7)
    for width in [2,3,4]:
        ts=[r for r in rows if r['event']=='timing'and r['width']==width and r['arm']<8]
        arms={v:[r for r in ts if r['variant']==v]for v in [control,candidate]}
        assert all(len(v)==4 for v in arms.values())
        means={v:{key:st.mean(x for r in group for x in r[key])for key in ['wall_ms','encode_ms','gpu_ms']}for v,group in arms.items()}
        pairs=[]
        for lo,hi in [(0,3),(4,7)]:
            sub=[r for r in ts if lo<=r['arm']<=hi]
            off=st.mean(st.mean(r['wall_ms'])for r in sub if r['variant']==control)
            on=st.mean(st.mean(r['wall_ms'])for r in sub if r['variant']==candidate)
            pairs.append(dict(saved_ms=off-on,speedup=off/on))
        cell=dict(width=width,rollback=means[control],candidate=means[candidate],saved_ms=means[control]['wall_ms']-means[candidate]['wall_ms'],speedup=means[control]['wall_ms']/means[candidate]['wall_ms'],pair_results=pairs)
        result['cells'].append(cell);print(json.dumps(cell))
    result['ablations']=[dict(variant=r['variant'],wall_ms=st.mean(r['wall_ms']),gpu_ms=st.mean(r['gpu_ms']),encode_ms=st.mean(r['encode_ms']))for r in rows if r['event']=='timing'and r['width']==4 and r['arm']>=8]
    b4=result['cells'][-1];result['kill_gate_pass']=all(x['saved_ms']>=2 for x in b4['pair_results'])
    print('kill_gate_pass',result['kill_gate_pass'])
    print('ablations',json.dumps(result['ablations']))
    for r in result['census']:print('census',json.dumps({k:v for k,v in r.items()if k!='counts'}))
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    if a.discard_input:a.input.unlink()
if __name__=='__main__':main()
