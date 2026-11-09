#!/usr/bin/env python3
"""Hardware floors and explicitly modelled additive append ledger."""
import argparse,collections,json,math,re,statistics as st
from pathlib import Path

PEAK=131.676e12

def category(kernel):
    k=kernel.lower()
    if any(x in k for x in ('selected','sparse_attn','sparse_attention','index_score','indexer_score','radix','select_top','qsa_pool','indexer_pool','attention_partial','attention_qk','attention_pv')):
        return 'context'
    if any(x in k for x in ('matmul','mat_mat','mat_vec','gemm','gemv','moe_swiglu','moe_down','moe_gate_up','moe_expert','iq3_s_down','iq4_nl_down','q6_k_dense')):
        return 'learned'
    return 'non_gemm'

def weight_floor(inventory,routes,rows,family):
    route_by_layer={int(r['layer']):r for r in routes}
    top=10 if family=='fn' else 8
    weight_bytes=0.;matrix_parameters=0.;head_parameters=0.;bank_bytes=0.;dense_bytes=0.;unknown=[]
    details=[]
    for layer,r in route_by_layer.items():
        counts=r['counts'];assert sum(counts)==rows*top
        active=sum(c>0 for c in counts);full=sum(r['bank_bytes'])
        useful=full*active/len(counts);weight_bytes+=useful;bank_bytes+=useful
        details.append(dict(layer=layer,active_experts=active,experts=len(counts),touched_bank_bytes=useful,
                            full_bank_bytes=full,slots=sum(counts),panels32=sum(math.ceil(c/32) for c in counts),
                            panels64=sum(math.ceil(c/64) for c in counts)))
    for t in inventory:
        name=t['name'];shape=t['shape'];role=t.get('role','');n=math.prod(shape)
        bank='_exps.' in name or role in ('ExpertGateUp','ExpertDown')
        if bank:
            e=shape[-1];matrix_parameters+=n/e*top
            continue
        if role=='Embedding' or 'token_embd' in name:continue
        if name=='output.weight':
            weight_bytes+=t['bytes'];dense_bytes+=t['bytes'];head_parameters+=n;continue
        weight_bytes+=t['bytes'];dense_bytes+=t['bytes']
        matrix=role in ('HyperMix','Projection','F32Projection','LatentAbsorb') if family=='glm' else (len(shape)>=2 and 'conv' not in name)
        if matrix:matrix_parameters+=n
        elif len(shape)>=2 and role not in ('Conv',):unknown.append(name)
    matrix_flops=2*matrix_parameters*rows+2*head_parameters
    # Existing fixed recurrent state R/W, pooled key union, minimum selected
    # cache union and new cache rows. Extra/repeated scratch traffic is excess.
    if family=='fn':
        context_flops=rows*(4*24*256*2048*12)+3072*rows*(65536+(rows-1)/2)
        state_bytes=226492416;index_bytes=12*(65536//4)*128*2
        selected_bytes=12*2048*2*256*2*2;write_bytes=rows*12*2*256*2*2
        gather_bytes=0 # Tiny embedding/PLE row gathers omitted from this lower bound.
    else:
        context_flops=rows*(4*64*512*2049.5*11)+22528*rows*(65536+(rows-1)/2)
        state_bytes=305267200;index_bytes=11*(65536//4)*128*2
        selected_bytes=11*2048*512*2;write_bytes=rows*11*512*2
        gather_bytes=0
    other_bytes=state_bytes+index_bytes+selected_bytes+write_bytes+gather_bytes
    return dict(matrix_flops=matrix_flops,context_flops=context_flops,useful_flops=matrix_flops+context_flops,
                unique_weight_bytes=weight_bytes,touched_expert_bytes=bank_bytes,dense_weight_bytes=dense_bytes,
                other_payload_bytes=other_bytes,useful_bytes=weight_bytes+other_bytes,
                matrix_parameters_per_row=matrix_parameters,head_parameters=head_parameters,
                expert_layers=details,nonmatrix_multidim=unknown)

def analyze(path):
    records=[json.loads(x) for x in path.read_text().splitlines()]
    family=next(x['family'] for x in records if x['kind']=='identity')
    inventory=next(x['tensors'] for x in records if x['kind']=='inventory')
    bw=st.median(x['gb_s'] for x in records if x['kind']=='bandwidth' and x['rep']>=2)*1e9
    counts=sorted(set(x['calls'] for x in records if x['kind']=='launch_floor'))
    ts=[st.median(x['gpu_ms'] for x in records if x['kind']=='launch_floor' and x['calls']==c) for c in counts]
    slope=sum((x-st.mean(counts))*(y-st.mean(ts)) for x,y in zip(counts,ts))/sum((x-st.mean(counts))**2 for x in counts)
    intercept=st.mean(ts)-slope*st.mean(counts)
    result=dict(family=family,bandwidth_gb_s=bw/1e9,tensor_peak_tflops=PEAK/1e12,
                launch_fit=dict(ms_per_dispatch=slope,intercept_ms=intercept,counts=counts,medians_ms=ts),cells={})
    for n in (400,512,2000,2048):
        native=[x for x in records if x['kind']=='native' and x['rows']==n and not x['warm']]
        w=st.mean(x['wall_ms'] for x in native);g=st.mean(x['gpu_ms'] for x in native)
        cell=dict(wall_ms=w,gpu_ms=g,host_nonoverlap_ms=w-g,native_samples=native)
        if n in (400,2000):
            routes=next(x['layers'] for x in records if x['kind']=='routes' and x['rows']==n)
            floor=weight_floor(inventory,routes,n,family)
            floor.update(compute_ms=floor['useful_flops']/PEAK*1e3,byte_ms=floor['useful_bytes']/bw*1e3,
                         learned_roof_ms=max(floor['matrix_flops']/PEAK,floor['unique_weight_bytes']/bw)*1e3)
            floor['roof_ms']=max(floor['compute_ms'],floor['byte_ms']);floor['roof_fraction']=floor['roof_ms']/w
            p=next(x for x in records if x['kind']=='profile' and x['rows']==n)
            rows=p['profile']['rows'];assert not p['profile']['overflow']
            assert all(x['gpu_start'] and x['gpu_end'] for x in rows)
            bykernel=collections.defaultdict(lambda:dict(calls=0,increment_ms=0.,isolated_ms=0.))
            previous=min(x['gpu_start'] for x in rows)
            for x in rows:
                k=bykernel[x['kernel']];k['calls']+=1;k['isolated_ms']+=x['gpu_ns']/1e6
                k['increment_ms']+=max(0,x['gpu_end']-previous)/1e6;previous=max(previous,x['gpu_end'])
            total=sum(x['increment_ms'] for x in bykernel.values());shares=collections.defaultdict(float);nc=collections.Counter()
            for kernel,x in bykernel.items():
                cat=category(kernel);shares[cat]+=x['increment_ms']/total;nc[cat]+=x['calls'];x['category']=cat
            launch=slope*len(rows)+intercept
            attention=g*shares['context']-slope*nc['context'];nongemm=g*shares['non_gemm']-slope*nc['non_gemm']
            ledger=dict(learned_compute_or_byte_floor=floor['learned_roof_ms'],context_attention=attention,
                        non_gemm=nongemm,launch_control_model=launch,host_nonoverlap=w-g)
            ledger['kernel_excess']=w-sum(ledger.values())
            assert abs(sum(ledger.values())-w)<1e-6
            cell.update(floor=floor,decomposition_ms=ledger,profile_gpu_ms=p['gpu_ms'],
                        profile_perturbation=p['gpu_ms']/g,dispatches=len(rows),kernels=dict(bykernel),
                        after=next(x for x in records if x['kind']=='after' and x['rows']==n))
        result['cells'][str(n)]=cell
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=analyze(a.input);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'}))
    for n,c in result['cells'].items():print(n,json.dumps({k:v for k,v in c.items() if k not in ('kernels','native_samples','floor')}))

if __name__=='__main__':main()
