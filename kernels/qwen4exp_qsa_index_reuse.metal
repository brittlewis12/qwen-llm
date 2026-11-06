#include <metal_stdlib>
using namespace metal;

struct qsa_index_reuse_args {
    uint start_position;
    uint query_count;
    uint ratio;
    uint block_capacity;
};

// Each SIMD owns one key, reused across Q query rows. Each dot retains
// four chronological lane-local FMAs, the same SIMD reduction, and the
// original head order. No query narrowing or additional device storage.
template <uint Q>
inline void qsa_index_reuse(constant qsa_index_reuse_args &a,
        device const float *queries, device const half *keys,
        device float *scores, device int *visible_counts,
        uint2 group, ushort simdgroup, ushort lane) {
    const uint q0=group.y*Q;
    if (q0+Q>a.query_count || a.ratio==0u) return;
    if (group.x==0u && simdgroup==0u && lane==0u) {
        for (uint q=0u;q<Q;++q) {
            const uint visible=(a.start_position+q0+q+1u)/a.ratio;
            visible_counts[q0+q]=visible<=a.block_capacity ? int(visible) : -1;
        }
    }
    const uint block=group.x*8u+uint(simdgroup);
    const uint maximum_visible=(a.start_position+q0+Q)/a.ratio;
    if (block>=maximum_visible || block>=a.block_capacity) return;
    device const half *key=keys+ulong(block)*128u;
    float4 score0=0.0f,score1=0.0f;
    for (uint head=0u;head<4u;++head) {
        float4 dot0=0.0f,dot1=0.0f;
        for (uint dimension=uint(lane);dimension<128u;dimension+=32u) {
            const float k=float(key[dimension]);
            const ulong base=ulong(q0)*512u+head*128u+dimension;
            dot0+=float4(queries[base],queries[base+512u],queries[base+1024u],queries[base+1536u])*k;
            if (Q==8u) dot1+=float4(queries[base+2048u],queries[base+2560u],queries[base+3072u],queries[base+3584u])*k;
        }
        const float4 sum0(simd_sum(dot0.x),simd_sum(dot0.y),simd_sum(dot0.z),simd_sum(dot0.w));
        if (lane==0u) score0+=max(sum0,float4(0.0f));
        if (Q==8u) {
            const float4 sum1(simd_sum(dot1.x),simd_sum(dot1.y),simd_sum(dot1.z),simd_sum(dot1.w));
            if (lane==0u) score1+=max(sum1,float4(0.0f));
        }
    }
    if (lane==0u) {
        for (uint q=0u;q<4u;++q) {
            const uint visible=(a.start_position+q0+q+1u)/a.ratio;
            if (block<visible) scores[ulong(q0+q)*a.block_capacity+block]=score0[q]*0.08838834764831845f;
            if (Q==8u) {
                const uint visible1=(a.start_position+q0+q+5u)/a.ratio;
                if (block<visible1) scores[ulong(q0+q+4u)*a.block_capacity+block]=score1[q]*0.08838834764831845f;
            }
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_qwen4exp_qsa_index_reuse4_f32(
        constant qsa_index_reuse_args &a [[buffer(0)]],
        device const float *q [[buffer(1)]],device const half *k [[buffer(2)]],
        device float *s [[buffer(3)]],device int *v [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]],ushort sg [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]) {
    qsa_index_reuse<4>(a,q,k,s,v,group,sg,lane);
}
[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_qwen4exp_qsa_index_reuse8_f32(
        constant qsa_index_reuse_args &a [[buffer(0)]],
        device const float *q [[buffer(1)]],device const half *k [[buffer(2)]],
        device float *s [[buffer(3)]],device int *v [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]],ushort sg [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]) {
    qsa_index_reuse<8>(a,q,k,s,v,group,sg,lane);
}
