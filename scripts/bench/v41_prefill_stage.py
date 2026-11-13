#!/usr/bin/env python3
"""Reduce three interleaved isolated GPU stage pairs with complete bitwise witnesses."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def selected(row, stage):
    name=row['row']['kernel']
    if stage=='experts':return name.startswith(('v41_sorted_gu','v41_sorted_down'))
    if stage=='index':return 'index_heads_rows_mma' in name or 'index_scores_tensor' in name
    if stage=='publication':
        constants=dict(row.get('constants',[]));bindings=dict(row['bindings'])
        return (name.endswith(('_compress_publish','_compress_rows_exact','_attention_pool')) or
            name=='kernel_rms_norm_mul_f32' and constants.get(0,[0])[0]==512 or
            name=='kernel_mat_vec_bf16_f32_tg' and bindings.get(1,{}).get('shape')==[512,128] or
            name=='kernel_copy_offset_f32' and constants.get(0,[0])[0]==512)
    constants=dict(row.get('constants',[]))
    return ('_pv_' in name or name.endswith('_normalize_rows') or
        name.endswith('_normalize_rope_rows') or
        name.endswith('_rope_rows') and constants.get(3,[0,0])[1]==1)


def reduce(path,stage,warmup_pairs):
    cells={};gpu={};exact={};overflow=0;components={};wide={};masked={};cooperative={}
    for line in path.open():
        r=json.loads(line)
        if r['kind']=='cell':cells[r['rep']]=r
        elif r['kind']=='exact':exact[r['rep']]=r['exact']
        elif r['kind']=='dispatch_summary':overflow+=r['overflow']
        elif r['kind']=='dispatch' and selected(r,stage):
            ns=r['row']['gpu_ns']
            if ns is None:raise ValueError('unavailable stage timestamp')
            gpu[r['rep']]=gpu.get(r['rep'],0)+ns/1e6
            if stage=='index' and r['row']['kernel'].endswith('index_heads_rows_mma_masked'):
                masked[r['rep']]=masked.get(r['rep'],0)+1
            if stage=='experts':
                name=r['row']['kernel'];rep=r['rep']
                values=components.setdefault(rep,dict(gate_up_ms=0.0,down_ms=0.0,wide_ms=0.0))
                values['gate_up_ms' if name.startswith('v41_sorted_gu') else 'down_ms']+=ns/1e6
                if name.startswith(('v41_sorted_gu_cooperative_','v41_sorted_down_cooperative_')):
                    cooperative[rep]=cooperative.get(rep,0)+1
                if name.startswith(('v41_sorted_gu128','v41_sorted_down128')):
                    values['wide_ms']+=ns/1e6;wide[rep]=wide.get(rep,0)+1
            elif stage=='attention-output':
                name=r['row']['kernel'];rep=r['rep']
                values=components.setdefault(rep,dict(pv_dispatch_ms=0.0,normalize_ms=0.0,inverse_rope_ms=0.0))
                key='pv_dispatch_ms' if '_pv_' in name else 'normalize_ms' if 'normalize' in name else 'inverse_rope_ms'
                values[key]+=ns/1e6
                if 'pv_tensor64_wide_rope' in name:wide[rep]=wide.get(rep,0)+1
    if overflow or len(exact)!=len(cells) or not all(exact.values()):
        raise ValueError('incomplete or non-exact stage packet')
    reps=sorted(r for r in cells if r>=2*warmup_pairs)
    if len(reps)<6 or len(reps)%2:raise ValueError('need at least three retained pairs')
    pairs=[]
    for a,b in zip(reps[::2],reps[1::2]):
        if cells[a]['policy']!='baseline':a,b=b,a
        pairs.append(dict(pair=min(a,b)//2,baseline_ms=gpu[a],candidate_ms=gpu[b],ratio=gpu[a]/gpu[b]))
    a=statistics.mean(p['baseline_ms'] for p in pairs);b=statistics.mean(p['candidate_ms'] for p in pairs)
    result=dict(source=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        stage=stage,warmup_pairs=warmup_pairs,retained_pairs=len(pairs),pairs=pairs,
        baseline_ms=a,candidate_ms=b,ratio=a/b,positive_pairs=sum(p['ratio']>1 for p in pairs),
        all_exact=True,thermal_max=max(c['thermal'] for c in cells.values()),
        note='Isolated dispatch GPU sums; instrumented wall is not a native speedup. All declared warmup arms stay in raw evidence.')
    if stage=='index' and any(c['policy']=='masked-reindex-short' for c in cells.values()):
        for rep,cell in cells.items():
            if (cell['policy']=='masked-reindex-short') != (masked.get(rep,0)>0):
                raise ValueError('short masked index eligibility witness failed')
        result['masked_dispatches_by_rep']={r:masked.get(r,0) for r in cells}
    elif stage=='experts':
        for rep,cell in cells.items():
            if (cell['policy']=='expert-panels-128') != (wide.get(rep,0)>0):
                raise ValueError('expert panel eligibility witness failed')
            if cell['policy'] in ('baseline','cooperative-experts') and any(c['policy']=='cooperative-experts' for c in cells.values()):
                if (cell['policy']=='cooperative-experts') != (cooperative.get(rep,0)>0):
                    raise ValueError('cooperative expert eligibility witness failed')
        arms={arm:[r for r in reps if (cells[r]['policy']=='baseline')==(arm=='baseline')] for arm in ('baseline','candidate')}
        result['components']={arm:{key:statistics.mean(components[r][key] for r in selected) for key in ('gate_up_ms','down_ms','wide_ms')} for arm,selected in arms.items()}
        result['component_ratios']={key:result['components']['baseline'][key]/result['components']['candidate'][key] for key in ('gate_up_ms','down_ms')}
        result['wide_dispatches_by_rep']={r:wide.get(r,0) for r in cells}
        if any(c['policy']=='cooperative-experts' for c in cells.values()):
            result['cooperative_dispatches_by_rep']={r:cooperative.get(r,0) for r in cells}
    elif stage=='attention-output':
        if any(c['policy']=='pv-wide' for c in cells.values()):
            for rep,cell in cells.items():
                if (cell['policy']=='pv-wide') != (wide.get(rep,0)>0):
                    raise ValueError('wide PV eligibility witness failed')
        arms={arm:[r for r in reps if (cells[r]['policy']=='baseline')==(arm=='baseline')] for arm in ('baseline','candidate')}
        result['components']={arm:{key:statistics.mean(components[r][key] for r in selected) for key in ('pv_dispatch_ms','normalize_ms','inverse_rope_ms')} for arm,selected in arms.items()}
        result['wide_dispatches_by_rep']={r:wide.get(r,0) for r in cells}
        result['component_note']='The candidate PV dispatch includes normalized inverse-RoPE writeback; its matrix-only cost is not isolated.'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path)
    p.add_argument('--stage',choices=('index','attention-output','publication','experts'),required=True)
    p.add_argument('--warmup-pairs',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=reduce(a.source,a.stage,a.warmup_pairs)
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
