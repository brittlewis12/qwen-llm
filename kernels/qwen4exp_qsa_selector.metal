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

// Singleton decode form of qsa_select_compact: identical radix-4 threshold,
// stable index-order compaction, ties and statuses. Each thread keeps one
// contiguous chunk of at most KPT keys in registers for every pass; per-thread
// bins are packed counters rather than an indexed local array, each radix pass
// needs one barrier (double-buffered bins), and compaction uses one packed
// (greater, equal) scan. The host sizes KPT * width >= the visible count; a
// violated bound reports status 4 instead of selecting from a truncated row.
template <uint KPT>
inline void qsa_select_decode(constant qsa_select_args &a, device const float *row,
        device const int *visible_counts, device int *out, device int *counts,
        device int *status, threadgroup uint *bins_tg, threadgroup uint *scan_tg,
        threadgroup uint *error_tg, uint lane, uint width, ushort sg, ushort sl) {
    const uint groups = width / 32u;
    const int visible_i = visible_counts[0];
    const bool valid = visible_i > 0 && uint(visible_i) <= a.row_capacity
        && a.top_k > 0u && a.top_k <= a.row_capacity && a.emit_ranked == 0u;
    const uint visible = valid ? uint(visible_i) : 0u;
    const uint selected = min(visible, a.top_k);
    const uint begin = lane * KPT;
    const uint owned = begin < visible ? min(KPT, visible - begin) : 0u;
    uint error = valid ? (visible > width * KPT ? 4u : 0u) : 1u;
    uint keys[KPT];
    _Pragma("unroll") for (uint k = 0u; k < KPT; ++k) {
        const float x = k < owned ? row[begin + k] : 0.0f;
        if (k < owned && !isfinite(x)) error = max(error, 2u);
        keys[k] = qsa_select_key(x);
    }
    const uint simd_error = simd_max(error);
    if (sl == 0u) error_tg[sg] = simd_error;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    error = simd_max(uint(sl) < groups ? error_tg[sl] : 0u);
    uint prefix = 0u, mask = 0u, rank = selected;
    if (error == 0u && selected < visible) {
        uint buffer = 0u;
        for (int shift = 28; shift >= 0; shift -= 4) {
            // Four words of four 8-bit bins: at most KPT <= 255 keys per thread.
            uint w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u;
            _Pragma("unroll") for (uint k = 0u; k < KPT; ++k) {
                const uint key = keys[k];
                if (k < owned && (key & mask) == prefix) {
                    const uint digit = (key >> uint(shift)) & 15u;
                    const uint inc = 1u << ((digit & 3u) * 8u);
                    const uint word = digit >> 2u;
                    w0 += word == 0u ? inc : 0u;
                    w1 += word == 1u ? inc : 0u;
                    w2 += word == 2u ? inc : 0u;
                    w3 += word == 3u ? inc : 0u;
                }
            }
            // Widen to 16-bit fields: word 2q holds bins 4q, 4q+2 and word 2q+1
            // holds bins 4q+1, 4q+3. Totals stay below 65536 (visible <= 32768).
            threadgroup uint *bins = bins_tg + buffer * 256u;
            const uint words[8] = {w0 & 0x00ff00ffu, (w0 >> 8u) & 0x00ff00ffu,
                                   w1 & 0x00ff00ffu, (w1 >> 8u) & 0x00ff00ffu,
                                   w2 & 0x00ff00ffu, (w2 >> 8u) & 0x00ff00ffu,
                                   w3 & 0x00ff00ffu, (w3 >> 8u) & 0x00ff00ffu};
            _Pragma("unroll") for (uint j = 0u; j < 8u; ++j) {
                const uint total = simd_sum(words[j]);
                if (sl == 0u) bins[uint(sg) * 8u + j] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint totals[8];
            _Pragma("unroll") for (uint j = 0u; j < 8u; ++j) {
                totals[j] = simd_sum(uint(sl) < groups ? bins[uint(sl) * 8u + j] : 0u);
            }
            uint remaining = rank, chosen = 0u, n = 0u;
            bool found = false;
            _Pragma("unroll") for (int b = 15; b >= 0; --b) {
                const uint ub = uint(b);
                n = (totals[(ub >> 2u) * 2u + (ub & 1u)] >> (((ub >> 1u) & 1u) * 16u)) & 0xffffu;
                if (remaining <= n) {
                    chosen = ub;
                    found = true;
                    break;
                }
                remaining -= n;
            }
            if (!found) {
                error = 3u;
                break;
            }
            prefix |= chosen << uint(shift);
            mask |= 15u << uint(shift);
            rank = remaining;
            if (n == rank || shift == 0) break;
            buffer ^= 1u;
        }
    }
    if (error == 0u) {
        uint greater = 0u, equal = 0u;
        _Pragma("unroll") for (uint k = 0u; k < KPT; ++k) {
            if (k < owned) {
                greater += uint(keys[k] > prefix);
                equal += uint(keys[k] == prefix);
            }
        }
        // Low 16 bits count keys above the threshold, high 16 bits ties.
        const uint packed = greater | (equal << 16u);
        const uint before_simd = simd_prefix_exclusive_sum(packed);
        const uint simd_total = simd_sum(packed);
        if (sl == 0u) scan_tg[sg] = simd_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint v = uint(sl) < groups ? scan_tg[sl] : 0u;
        const uint before = before_simd + simd_sum(uint(sl) < uint(sg) ? v : 0u);
        const uint total = simd_sum(v);
        const uint equal_before = before >> 16u;
        error = (total & 0xffffu) + min(total >> 16u, rank) == selected ? 0u : 3u;
        if (error == 0u) {
            uint dst = (before & 0xffffu) + min(equal_before, rank);
            uint seen = 0u;
            _Pragma("unroll") for (uint k = 0u; k < KPT; ++k) {
                if (k < owned) {
                    const uint key = keys[k];
                    const bool at = key == prefix;
                    const bool take = key > prefix || (at && equal_before + seen < rank);
                    seen += uint(at);
                    if (take) out[dst++] = int(begin + k);
                }
            }
            for (uint i = selected + lane; i < a.top_k; i += width) out[i] = -1;
        }
    }
    if (error != 0u) {
        for (uint i = lane; i < a.top_k; i += width) out[i] = i < selected ? int(i) : -1;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane == 0u) {
        counts[0] = int(selected);
        status[0] = int(error);
    }
}

#define QSA_SELECT_DECODE(N)                                                                    \
kernel void kernel_qwen4exp_qsa_select_decode##N##_f32(                                         \
        constant qsa_select_args &a [[buffer(0)]], device const float *scores [[buffer(1)]],    \
        device const int *visible [[buffer(2)]], device int *ids [[buffer(3)]],                 \
        device int *counts [[buffer(4)]], device int *status [[buffer(5)]],                     \
        uint lane [[thread_index_in_threadgroup]], uint width [[threads_per_threadgroup]],      \
        ushort sg [[simdgroup_index_in_threadgroup]], ushort sl [[thread_index_in_simdgroup]]) { \
    threadgroup uint bins[2u * 32u * 8u];                                                       \
    threadgroup uint scan[32];                                                                  \
    threadgroup uint errors[32];                                                                \
    qsa_select_decode<N>(a, scores, visible, ids, counts, status, bins, scan, errors, lane,     \
                         width, sg, sl);                                                        \
}
QSA_SELECT_DECODE(16)
QSA_SELECT_DECODE(32)
#undef QSA_SELECT_DECODE

kernel void kernel_fn_batch_qsa_select(
        constant qsa_select_args &a [[buffer(0)]], device const float *score0 [[buffer(1)]],
        device const float *score1 [[buffer(2)]], device const float *score2 [[buffer(3)]],
        device const float *score3 [[buffer(4)]], device const int *visible0 [[buffer(5)]],
        device const int *visible1 [[buffer(6)]], device const int *visible2 [[buffer(7)]],
        device const int *visible3 [[buffer(8)]], device int *ids0 [[buffer(9)]],
        device int *ids1 [[buffer(10)]], device int *ids2 [[buffer(11)]], device int *ids3 [[buffer(12)]],
        device int *count0 [[buffer(13)]], device int *count1 [[buffer(14)]],
        device int *count2 [[buffer(15)]], device int *count3 [[buffer(16)]],
        device int *status0 [[buffer(17)]], device int *status1 [[buffer(18)]],
        device int *status2 [[buffer(19)]], device int *status3 [[buffer(20)]],
        threadgroup uint *scratch [[threadgroup(0)]], threadgroup uint *shared [[threadgroup(1)]],
        uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
        uint threads [[threads_per_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    qsa_select_compact(a,
                       r == 0   ? score0
                       : r == 1 ? score1
                       : r == 2 ? score2
                                : score3,
                       r == 0   ? visible0
                       : r == 1 ? visible1
                       : r == 2 ? visible2
                                : visible3,
                       r == 0   ? ids0
                       : r == 1 ? ids1
                       : r == 2 ? ids2
                                : ids3,
                       r == 0   ? count0
                       : r == 1 ? count1
                       : r == 2 ? count2
                                : count3,
                       r == 0   ? status0
                       : r == 1 ? status1
                       : r == 2 ? status2
                                : status3,
                       scratch, shared, tid, 0, threads, sg, lane);
}
