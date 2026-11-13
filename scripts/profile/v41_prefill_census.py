#!/usr/bin/env python3
"""Reduce bounded V4.1 dispatch captures; rates describe logical payloads.

Tensor peak is a reference, not the scalar/SFU roof. Packed expert traffic is
one complete bank pass, not measured traffic or proof every expert was touched.
Cache hits, padded tiles, repeated dequantization and hardware DRAM traffic are
not inferred from tensor bindings. Non-matrix work is not credited as FLOPs.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

PEAK=131.7


def classify(r):
    d=r["row"]; k=d["kernel"]; b=dict(r["bindings"]); c=dict(r.get("constants",[]))
    flops=payload=0; group="other"
    if k.startswith("v41_sorted_gu"):
        group="routed_gate_up"; shape=b[1]["shape"]; slots=math.prod(b[9]["shape"])
        flops=4*slots*shape[0]*shape[1]
        payload=b[1]["bytes"]+b[2]["bytes"]+b[3]["bytes"]+b[10]["bytes"]
    elif k.startswith("v41_sorted_down"):
        group="routed_down"; shape=b[1]["shape"]; slots=b[3]["shape"][0]//shape[0]
        flops=2*slots*shape[0]*shape[1]
        payload=b[1]["bytes"]+b[3]["bytes"]+b[6]["bytes"]
    elif k.startswith(("kernel_mat_mat_","kernel_mat_vec_")):
        group="dense"; shape=b[1]["shape"]; flops=2*shape[0]*math.prod(b[3]["shape"])
        payload=sum(b[s]["bytes"] for s in (1,2,3))
        if shape[:2]==[6144,25600]:group="engram_gpu"
    elif any(s in k for s in ("score_simdgroup_qk","score_tensor","score_mma","pv_tensor","pv_mma")):
        is_pv="pv_" in k; group="attention_pv" if is_pv else "attention_qk"
        if c:
            dim,window,topk,_=c[0]; start,_,_,ratio=c[1]; _,heads,n,_=c[11]
            keys=sum(min(start+i+1,window)+min((start+i+1)//ratio if ratio else 0,topk) for i in range(n))
            flops=2*keys*heads*dim
            payload=keys*dim*4 + keys*heads*4*((2 if 'pv_tensor64_wide' in k else 4) if is_pv else 1) + n*heads*dim*(4 if is_pv else 2)
    elif "index_heads_rows_mma" in k or "index_scores_tensor" in k:
        group="index_scores"
        if c:
            start,ratio,topk,_=c[0]; dim,heads,_,_=c[1]
            _,n,_,_=c[2];first=c[7][0]
            keys=sum((start+first+i+1)//ratio for i in range(n) if (start+first+i+1)//ratio>topk)
            flops=2*keys*heads*dim
            payload=keys*(dim+1)*4+n*heads*dim*4
    elif "select_" in k or "selection_" in k:
        group="index_topk"
        if "select_radix_rows" in k and c:
            width,_,n,_=c[1];topk=c[2][0]
            payload=n*(width+topk)*4 # one score scan + result, not all radix passes
    elif "hc_" in k:
        group="mhc"
        if "hc_projection_rows" in k:
            payload=sum(b[s]["bytes"] for s in (1,2,3))
            if c:
                n,width,_=c[0];flops=2*n*width*24
        elif "hc_collapse_batch" in k and c:
            width,n=c[0];payload=4*(5*width*n+4*n)
        elif "hc_post_batch" in k and c:
            width,n=c[0];payload=4*(9*width*n+20*n)
    elif "engram" in k:
        group="engram_gpu"
        if c:payload=13*c[0][0]*(d["grid"][0]//4)*4
    elif "normalize_rows" in k or "softmax_rows" in k:
        group="attention_sinks"
        if c:
            dim,_,_,stride=c[0];_,heads,n,_=c[11]
            payload=8*n*heads*(dim if "normalize" in k else stride)
    elif "rope_rows" in k:
        if c:
            dim,heads,_,_=c[0];n,inverse=c[3]
            group="attention_inverse_rope" if inverse else "query_index_rope"
            payload=8*n*heads*dim
        else:group="query_index_rope"
    # Adaptive M64/M128 dispatches partition expert runs. Credit the useful
    # matrix and one-bank payload once at M64; both dispatch times accumulate.
    if k.startswith(("v41_sorted_gu128", "v41_sorted_down128")):flops=payload=0
    return group,flops,payload


def reduce(path, rep=0):
    groups=defaultdict(lambda:dict(dispatches=0,gpu_ms=0,matrix_flops=0,logical_payload_bytes=0))
    kernels=defaultdict(lambda:dict(dispatches=0,gpu_ms=0))
    reports=[]; chunks=[]; overflow=0; cells=[]
    with path.open() as source:
        for line in source:
            r=json.loads(line)
            if r["kind"]=="dispatch_summary":overflow+=r["overflow"]
            elif r["kind"]=="chunk_dispatch":chunks.append(r)
            elif r["kind"]=="cell" and r["rep"]==rep:cells.append({k:v for k,v in r.items() if k!="profile"})
            elif r["kind"]=="dispatch" and r["rep"]==rep:
                group,f,payload=classify(r)
                ms=r["row"]["gpu_ns"]/1e6
                row=groups[group];row["dispatches"]+=1;row["gpu_ms"]+=ms
                row["matrix_flops"]+=f;row["logical_payload_bytes"]+=payload
                kernel=kernels[r["row"]["kernel"]];kernel["dispatches"]+=1;kernel["gpu_ms"]+=ms
    if overflow:raise ValueError(f"incomplete capture: {overflow} overflow dispatches")
    for name,r in groups.items():
        ms=r["gpu_ms"];f=r["matrix_flops"]
        r["matrix_tflops"]=f/(ms*1e9) if f else None
        r["tensor_peak_percent"]=100*r["matrix_tflops"]/PEAK if f else None
        r["logical_payload_gbps"]=r["logical_payload_bytes"]/(ms*1e6) if r["logical_payload_bytes"] else None
        r["tensor_ideal_ms"]=f/(PEAK*1e9) if f else None
        r["gpu_ms_above_tensor_ideal"]=ms-r["tensor_ideal_ms"] if f else None
    return dict(source=path.name,cells=cells,chunks=chunks,groups=dict(groups),kernels=dict(kernels),
                isolated_kernel_sum_ms=sum(r["gpu_ms"] for r in groups.values()),
                traffic_note=__doc__)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--rep",type=int,default=0)
    a=p.parse_args();result=reduce(a.source,a.rep)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    for name,r in sorted(result["groups"].items(),key=lambda x:-x[1]["gpu_ms"]):
        print(name,round(r["gpu_ms"],3),r["matrix_tflops"],r["tensor_peak_percent"],r["logical_payload_gbps"])
