"""Summarize the ordinary dispatch census and standalone counterfactuals."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import random
import math
import sys

root = Path(sys.argv[1])
def read(name):
    path = root / name
    return [json.loads(s) for s in path.read_text().splitlines() if s.startswith('{')] if path.exists() else []

out = {'census': [], 'holds': []}
for r in read('census-current.jsonl'):
    if r.get('kind') != 'occupancy':
        continue
    sums = defaultdict(float)
    calls = defaultdict(int)
    grids = defaultdict(set)
    for d in r['expert_dispatches']:
        name = d['kernel']
        if 'moe_' in name and ('_up_' in name or '_down_iq2_s_' in name or ('f16act' in name and 'grouped_slots_tensor' in name)):
            role = 'gate_up'
        elif 'wide_probe' in name:
            role = 'gate_up'
        elif 'moe_down_' in name or 'glm_iq3_down' in name:
            role = 'down'
        else:
            role = 'other'
        sums[role] += d['gpu_ns'] / 1e6
        calls[role] += 1
        grids[name].add(tuple(d['grid']))
    rec = {'rows': r['width'], 'stage_ms': r['routed_experts_ms'], 'kernel_ms': dict(sums),
           'dispatches': dict(calls), 'distribution': r['rows_per_expert_distribution'],
           'grids': {k: list(v) for k, v in grids.items()}}
    rec['GB_s'] = {role: r['union_'+role+'_bytes'] / sums[role] / 1e6 for role in ['gate_up', 'down']}
    rec['stage_GB_s'] = r['union_total_bytes'] / r['routed_experts_ms'] / 1e6
    out['census'].append(rec)

values = defaultdict(list)
for r in read('holds.jsonl') + read('empty.jsonl'):
    if r.get('event') == 'timing':
        values[r['rows'], r['dtype'], r['arm'], r['block']].append(r['ms'])
for rows in [32, 161, 256, 512]:
    for dtype in ['iq2_s', 'iq3_s']:
        arms = {arm for n, dt, arm, _ in values if n == rows and dt == dtype}
        sums = {arm: sum(statistics.median(v) for (n, dt, a, _), v in values.items() if n == rows and dt == dtype and a == arm) for arm in arms}
        if sums:
            out['holds'].append({'rows': rows, 'dtype': dtype, 'sum_layer_median_ms': sums,
                                 'relative_to_production': {k: v/sums['production'] for k,v in sums.items()}})
stage_values = defaultdict(list)
for r in read('model-ab.jsonl'):
    if r.get('kind') == 'tensor_ab':
        stage_values[r['rows'], r['candidate'], r['rep']].append(r['stages']['routed_experts'])
out['model_stage_ab'] = []
for rows in sorted({key[0] for key in stage_values}):
    arms = {arm: [v for (n,a,_), vs in stage_values.items() if (n,a)==(rows,arm) for v in vs] for arm in [False,True]}
    if any(len(v)!=6 for v in arms.values()):
        continue
    ratios = [statistics.mean(stage_values[rows,False,rep])/statistics.mean(stage_values[rows,True,rep]) for rep in range(3)]
    rng = random.Random(71)
    boot = sorted(math.exp(statistics.mean(math.log(rng.choice(ratios)) for _ in ratios)) for _ in range(20000))
    base, candidate = statistics.median(arms[False]), statistics.median(arms[True])
    out['model_stage_ab'].append({'rows': rows, 'baseline_ms': base, 'candidate_ms': candidate,
        'speedup': base/candidate, 'paired_block_ratios': ratios,
        'paired_block_geomean_90CI': [boot[1000],boot[18999]], 'samples_per_arm': 6})
print(json.dumps(out, indent=2))
