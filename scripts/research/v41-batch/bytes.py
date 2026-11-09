#!/usr/bin/env python3
"""Header-only V4.1 B=3 logical byte floor; no tensor payload or GPU access."""
import argparse, collections, json, math, re, struct
from pathlib import Path

# Native GGML serialized layouts, including the real mixed IQ bank types.
LAYOUT = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),8:(32,34),9:(32,40),
          10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),
          16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),
          21:(256,110),22:(256,82),23:(256,136),24:(1,1),25:(1,2),
          26:(1,4),27:(1,8),28:(1,8),29:(256,56),30:(1,2)}


def header(path):
    with path.open('rb') as f:
        def scalar(fmt):
            size=struct.calcsize('<'+fmt); raw=f.read(size)
            if len(raw)!=size:raise ValueError('truncated GGUF header')
            return struct.unpack('<'+fmt,raw)[0]
        def string():
            n=scalar('Q');raw=f.read(n)
            if len(raw)!=n:raise ValueError('truncated GGUF string')
            return raw.decode()
        def skip(kind):
            sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
            if kind in sizes:f.seek(sizes[kind],1)
            elif kind==8:f.seek(scalar('Q'),1)
            elif kind==9:
                typ=scalar('I');n=scalar('Q')
                if typ in sizes:f.seek(n*sizes[typ],1)
                else:
                    for _ in range(n):skip(typ)
            else:raise ValueError(f'unknown metadata type {kind}')
        assert f.read(4)==b'GGUF' and scalar('I')==3
        nt,nkv=scalar('Q'),scalar('Q')
        for _ in range(nkv):string();skip(scalar('I'))
        tensors=[]
        for _ in range(nt):
            name=string();shape=[scalar('Q') for _ in range(scalar('I'))]
            kind=scalar('I');scalar('Q')
            block,size=LAYOUT[kind];count=math.prod(shape)
            assert count%block==0
            tensors.append(dict(name=name,shape=shape,dtype=kind,bytes=count//block*size))
        return tensors


def ledger(tensors,depth,batch=3,bw=1175.47e9):
    owners={2,8,14,20};index={2,8,14,20,24,28,32,36}
    shared=collections.Counter();routed=collections.Counter();types=collections.Counter();counts=collections.Counter();embedding=0
    for t in tensors:
        name=t['name'];n=t['bytes'];types[t['dtype']]+=n
        if name=='token_embd.weight':embedding=n//t['shape'][1];continue
        if name=='output.weight':shared['head']+=n;continue
        m=re.fullmatch(r'blk\.(\d+)\.(.*)',name)
        if not m:
            if name=='output_norm.weight':shared['hc_norm_controls']+=n
            continue
        layer,tail=int(m[1]),m[2]
        if layer>=40:continue
        if '_exps.' in tail:
            dims=[v for v in t['shape'] if v!=1];assert dims[-1]==384
            routed[str(t['dtype'])]+=n//384*6;counts['expert_banks']+=1;continue
        if 'engram' in tail and ('embd' in tail or 'embed' in tail):continue
        if tail.startswith('attn_compressor') and layer not in owners:continue
        if tail=='attn_compressor_gate.weight' and layer>=20:continue
        if tail.startswith('indexer.'):
            if layer not in index:continue
            if tail in ('indexer.attn_k.weight','indexer.k_norm.weight') and layer not in owners:continue
            cat='index_projections'
        elif tail.startswith('attn_compressor'):cat='compressor_projections'
        elif tail in ('attn_q_a.weight','attn_q_b.weight','attn_kv.weight','attn_output_a.weight','attn_output_b.weight'):cat='attention_projections'
        elif '_shexp.' in tail:cat='shared_expert'
        elif 'engram' in tail:cat='engram_projection_controls'
        elif tail=='ffn_gate_inp.weight':cat='router'
        else:cat='hc_norm_controls'
        shared[cat]+=n;counts[cat]+=1
    assert counts['expert_banks']==120 and counts['attention_projections']==200
    # One F32 latent key union per attention site; QK and PV reuse this union.
    cache_attention=(40*128+38*512)*512*4
    cache_index=sum(depth//(2 if i<20 else 1) for i in index)*128*4
    # Raw-ring writes, owner cache/index publications, compressor accumulator R/W.
    cache_writes=40*512*4+sum((512+128)*4/(2 if i<20 else 1) for i in owners)
    compressor_state=3*2*512*4*2
    # Native IQ4_NL disk Engram: two layers, 24 256-element rows, 18 B/32.
    gathers=embedding+2*24*256//32*18
    per_sequence=cache_attention+cache_index+cache_writes+compressor_state+gathers
    weight=sum(shared.values())+batch*sum(routed.values());total=weight+batch*per_sequence
    return dict(depth=depth,batch=batch,bandwidth_gb_s=bw/1e9,shared_once=dict(shared),
                routed_one_sequence=dict(routed),shared_weight_bytes=sum(shared.values()),
                routed_weight_bytes=batch*sum(routed.values()),per_sequence_cache_attention=cache_attention,
                per_sequence_index_keys=cache_index,per_sequence_other_state_gathers=per_sequence-cache_attention-cache_index,
                state_cache_bytes=batch*per_sequence,total_bytes=total,byte_floor_ms=total/bw*1e3,
                counts=dict(counts),assumptions='Logical requested bytes; no cross-sequence routed overlap credit; ideal reuse of shared weights and QK/PV key union. Excludes repeated activation/scratch/cacheline traffic. F32 cache/index geometry. No physical DRAM claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('--batch',type=int,default=3);a=p.parse_args()
    name=str(a.model);pattern=re.sub(r'-\d{5}-of-\d{5}\.gguf$','-*-of-*.gguf',a.model.name)
    files=sorted(a.model.parent.glob(pattern));assert files
    tensors=[t for f in files for t in header(f)]
    print(json.dumps(dict(model=name,shards=len(files),cells=[ledger(tensors,d,a.batch) for d in [32768,65536]])))

if __name__=='__main__':main()
