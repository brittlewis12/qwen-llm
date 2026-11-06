// Exact IDs-only radix selection for Flash-Next: parallel stable scans and
// early termination when the threshold bucket is fully selected.
// No score rounding or additional device buffers.
#include <metal_stdlib>
using namespace metal;

struct qsa_select_args { uint row_capacity; uint top_k; uint query_count; uint emit_ranked; };

inline uint qsa_select_key(float x) {
    uint bits=as_type<uint>(x);
    // Match the incumbent's fast-math ordering: signed zeros/subnormals tie.
    if ((bits&0x7f800000u)==0u) bits=0u;
    return (bits&0x80000000u)!=0u ? ~bits : bits^0x80000000u;
}

inline uint qsa_exclusive_scan(uint value,threadgroup uint *scratch,
        threadgroup uint *shared,uint groups,ushort sg,ushort sl) {
    const uint before=simd_prefix_exclusive_sum(value);
    const uint sum=simd_sum(value);
    if (sl==0u) scratch[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg==0u) {
        const uint v=uint(sl)<groups ? scratch[sl] : 0u;
        const uint prefix=simd_prefix_exclusive_sum(v);
        const uint total=simd_sum(v);
        if (uint(sl)<groups) scratch[sl]=prefix;
        if (sl==0u) shared[0]=total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return before+scratch[sg];
}

inline void qsa_select_compact(constant qsa_select_args &a,
        device const float *scores,device const int *visible_counts,
        device int *ids,device int *counts,device int *status,
        threadgroup uint *scratch,threadgroup uint *shared,
        uint lane,uint query,uint width,ushort sg,ushort sl) {
    if (query>=a.query_count) return;
    const uint groups=width/32u;
    const int visible_i=visible_counts[query];
    const bool valid=visible_i>0 && uint(visible_i)<=a.row_capacity
        && a.top_k>0u && a.top_k<=a.row_capacity && a.emit_ranked==0u;
    const uint visible=valid ? uint(visible_i) : 0u;
    const uint selected=min(visible,a.top_k);
    device const float *row=scores+ulong(query)*a.row_capacity;
    device int *out=ids+ulong(query)*a.top_k;
    for (uint i=lane;i<a.top_k;i+=width) out[i]=-1;
    uint error=valid ? 0u : 1u;
    for (uint i=lane;i<visible;i+=width) if (!isfinite(row[i])) error=2u;
    const uint se=simd_max(error);
    if (sl==0u) scratch[sg]=se;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg==0u) {
        const uint e=uint(sl)<groups ? scratch[sl] : 0u;
        const uint total=simd_max(e);
        if (sl==0u) shared[0]=total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    error=shared[0];
    uint prefix=0u,mask=0u,rank=selected;
    if (error==0u && selected<visible) {
        for (int shift=28;shift>=0;shift-=4) {
            uint bins[16]={};
            for (uint i=lane;i<visible;i+=width) {
                const uint key=qsa_select_key(row[i]);
                if ((key&mask)==prefix) ++bins[(key>>uint(shift))&15u];
            }
            for (uint bin=0u;bin<16u;++bin) {
                const uint n=simd_sum(bins[bin]);
                if (sl==0u) scratch[uint(sg)*16u+bin]=n;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg==0u && sl<16u) {
                uint n=0u;
                for (uint g=0u;g<groups;++g) n+=scratch[g*16u+uint(sl)];
                shared[sl]=n;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane==0u) {
                uint remaining=rank,chosen=0u,n=0u;
                bool found=false;
                for (int b=15;b>=0;--b) {
                    n=shared[uint(b)];
                    if (remaining<=n) {chosen=uint(b);found=true;break;}
                    remaining-=n;
                }
                shared[16]=chosen;shared[17]=remaining;
                shared[18]=found ? 0u : 3u;shared[19]=n;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            error=shared[18];
            if (error!=0u) break;
            prefix|=shared[16]<<uint(shift);mask|=15u<<uint(shift);rank=shared[17];
            const uint bucket=shared[19];
            // Every key in this bucket belongs to the top-k. Its lower bound
            // is sufficient; compaction below includes the entire bucket.
            if (bucket==rank || shift==0) break;

        }
    }
    if (error==0u) {
        const uint chunk=(visible+width-1u)/width;
        const uint begin=min(lane*chunk,visible),end=min(begin+chunk,visible);
        uint equal=0u;
        for (uint i=begin;i<end;++i) equal+=uint(qsa_select_key(row[i])==prefix);
        const uint equal_before=qsa_exclusive_scan(equal,scratch,shared,groups,sg,sl);
        uint seen=0u,n=0u;
        for (uint i=begin;i<end;++i) {
            const uint key=qsa_select_key(row[i]);
            const bool at=key==prefix;
            n+=uint(key>prefix || (at && equal_before+seen<rank));
            seen+=uint(at);
        }
        const uint before=qsa_exclusive_scan(n,scratch,shared,groups,sg,sl);
        error=shared[0]==selected ? 0u : 3u;
        if (error==0u) {
            uint dst=before;seen=0u;
            for (uint i=begin;i<end;++i) {
                const uint key=qsa_select_key(row[i]);
                const bool at=key==prefix;
                const bool take=key>prefix || (at && equal_before+seen<rank);
                seen+=uint(at);
                if (take) out[dst++]=int(i);
            }
        }
    }
    if (error!=0u) {
        for (uint i=lane;i<a.top_k;i+=width) out[i]=i<selected ? int(i) : -1;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane==0u) {counts[query]=int(selected);status[query]=int(error);}
}

kernel void kernel_qwen4exp_qsa_select_scan_f32(
        constant qsa_select_args &a [[buffer(0)]],
        device const float *scores [[buffer(1)]],device const int *visible [[buffer(2)]],
        device int *ids [[buffer(3)]],device int *counts [[buffer(4)]],device int *status [[buffer(5)]],
        threadgroup uint *scratch [[threadgroup(0)]],threadgroup uint *shared [[threadgroup(1)]],
        uint lane [[thread_index_in_threadgroup]],uint query [[threadgroup_position_in_grid]],
        uint width [[threads_per_threadgroup]],ushort sg [[simdgroup_index_in_threadgroup]],
        ushort sl [[thread_index_in_simdgroup]]) {
    qsa_select_compact(a,scores,visible,ids,counts,status,scratch,shared,lane,query,width,sg,sl);
}
