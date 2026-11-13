"""Recompute hot-set coverage from actual mixed GLM level-code block sizes."""
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
records=list(map(json.loads,open(root/'docs/bench/2026-11-13-glm-prefill/stage-attribution/census.jsonl')))
cells={(d['depth'],d['width']):d for d in records if d.get('kind')=='occupancy'}
plan=json.load(open(root/'../plans/glm-serve-plan.json'))
limits=(7,13,8)
appends=[c['append_tokens'] for a,n in zip(plan['agents'],limits) for c in a['calls'][:n]]
cell_rows={'root2048':{},'append256':{},'append1024':{}}
work={}; native={}; cached={}; speed_units={}

def code_bytes(layer):
    g=int(layer['gate_up_bytes_per_expert']); d=int(layer['down_bytes_per_expert'])
    if g==5_373_952: cg=round(g*106/82)
    elif g==7_208_960: cg=round(g*134/110)
    else: raise ValueError(f"unexpected gate/up bytes {g}")
    if d==3_604_480: cd=round(d*134/110); down_speeded=True
    elif d==4_456_448: cd=d; down_speeded=False
    else: raise ValueError(f"unexpected down bytes {d}")
    return g+d, cg+cd, 2+int(down_speeded)

def add_cell(name, data, row_scale):
    for layer in data['per_layer']:
        nb,cb,su=code_bytes(layer)
        for e,count in enumerate(layer['counts']):
            key=(layer['block'],e)
            amount=float(count)*row_scale
            cell_rows[name][key]=amount
            work[key]=work.get(key,0)+amount*3
            native[key]=nb;cached[key]=cb;speed_units[key]=su

# One captured root chunk replicated across the three 48K sessions (24 chunks each).
root_cell=cells[(49152,2048)]
add_cell('root2048',root_cell,72)
# Attribute each completed serve-plan append to its measured width bucket.
append_by_width={256:[],1024:[]}
for nrows in appends: append_by_width[256 if nrows<=384 else 1024].append(nrows)
for width,label in ((256,'append256'),(1024,'append1024')):
    data=cells[(81920,width)]
    for layer in data['per_layer']:
        nb,cb,su=code_bytes(layer)
        for e,count in enumerate(layer['counts']):
            key=(layer['block'],e)
            amount=sum(count*n/width for n in append_by_width[width])
            cell_rows[label][key]=amount
            work[key]=work.get(key,0)+amount*3
            native[key]=nb;cached[key]=cb;speed_units[key]=su

total_combined=sum(work.values())
result={"served_plan":{"agents":3,"calls":len(appends),"append_tokens":sum(appends),"root_chunk_multiplier":72},
        "full_bank_bytes":{"native":sum(native.values()),"level_cache":sum(cached.values())},
        "block_bits_per_weight":{"IQ2_S":{"native":82*8/256,"level":106*8/256},
                                  "IQ3_S":{"native":110*8/256,"level":134*8/256},
                                  "IQ4_XS_down_kept_native":136*8/256},
        "cell_stage_share":{},"coverage_pct":{}}
for name,(depth,width) in {'root2048':(49152,2048),'append256':(81920,256),'append1024':(81920,1024)}.items():
    cell=cells[(depth,width)]
    result['cell_stage_share'][name]=cell['stage_metrics']['routed_experts']['stage_ms']/cell['ordinary_wall_ms']
for budget in (57e9,60e9):
    entries=sorted(work,key=lambda k:work[k]/cached[k],reverse=True)
    used=0.0; hot=set()
    for key in entries:
        if used+cached[key]<=budget:
            used+=cached[key];hot.add(key)
    cov={}
    for name,rows in cell_rows.items():
        total=sum(x*3 for x in rows.values())
        all_cache=sum(x*3 for k,x in rows.items() if k in hot)
        speedable=sum(x*speed_units[k] for k,x in rows.items() if k in hot)
        cov[name]={"whole_expert_total_work_pct":100*all_cache/total,
                   "level_speeded_work_pct":100*speedable/total}
    result['coverage_pct'][str(int(budget/1e9))+'GB']={"used_bytes":used,"combined_plan_work_pct":100*sum(work[k] for k in hot)/total_combined,"cells":cov}
print(json.dumps(result,sort_keys=True,indent=2))
