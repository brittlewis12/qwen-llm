// Original F32 query/F16 keys; same dimension/FMA/reduction order.
#include <metal_stdlib>
using namespace metal;
struct qwen4exp_qsa_packed_attention_args {
    uint start_position;
    uint query_count;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    uint block_budget;
    uint ratio;
    uint row_stride;
    uint cache_capacity;
    float scale;
};
template <uint KeysPerSimd>
inline void qsa_logits_reuse(
        constant qwen4exp_qsa_packed_attention_args & args,
        device const float * query,
        device const half * key_cache,
        device const int * token_ids,
        device const int * selected_counts,
        device const int * status,
        device float * logits,
        uint3 group,
        ushort simdgroup,
        ushort simd_lane) {
    constexpr uint heads_per_group = 4u;
    constexpr uint slots_per_simdgroup = KeysPerSimd;
    constexpr uint simdgroups_per_threadgroup = 4u;
    constexpr uint slots_per_threadgroup = slots_per_simdgroup * simdgroups_per_threadgroup;
    if (args.head_dim != 256u
            || args.kv_heads == 0u
            || args.query_heads % args.kv_heads != 0u
            || args.ratio == 0u) return;
    const uint heads_per_kv = args.query_heads / args.kv_heads;
    if (heads_per_kv % heads_per_group != 0u
            || group.y >= args.kv_heads
            || group.z >= args.query_count) return;
    const uint slot_tiles = (args.row_stride + slots_per_threadgroup - 1u)
        / slots_per_threadgroup;
    const uint head_groups = heads_per_kv / heads_per_group;
    if (group.x >= head_groups * slot_tiles) return;
    const uint head_group = group.x / slot_tiles;
    const uint slot_tile = group.x % slot_tiles;
    const uint first_head = group.y * heads_per_kv + head_group * heads_per_group;
    const uint query_index = group.z;
    const uint sequence_length = args.start_position + query_index + 1u;
    const int selected_i = selected_counts[query_index];
    const bool metadata_valid = status[query_index] == 0
        && selected_i == int(args.block_budget);
    const uint active_count = metadata_valid
        ? uint(selected_i) * args.ratio + sequence_length % args.ratio
        : 0u;
    const ulong id_start = (ulong)query_index * args.row_stride;
    const ulong query_start = ((ulong)query_index * args.query_heads + first_head)
        * args.head_dim;
    const uint d0=uint(simd_lane)+0u;
    const float4 q0=float4(query[query_start+d0],query[query_start+256u+d0],query[query_start+512u+d0],query[query_start+768u+d0]);
    const uint d1=uint(simd_lane)+32u;
    const float4 q1=float4(query[query_start+d1],query[query_start+256u+d1],query[query_start+512u+d1],query[query_start+768u+d1]);
    const uint d2=uint(simd_lane)+64u;
    const float4 q2=float4(query[query_start+d2],query[query_start+256u+d2],query[query_start+512u+d2],query[query_start+768u+d2]);
    const uint d3=uint(simd_lane)+96u;
    const float4 q3=float4(query[query_start+d3],query[query_start+256u+d3],query[query_start+512u+d3],query[query_start+768u+d3]);
    const uint d4=uint(simd_lane)+128u;
    const float4 q4=float4(query[query_start+d4],query[query_start+256u+d4],query[query_start+512u+d4],query[query_start+768u+d4]);
    const uint d5=uint(simd_lane)+160u;
    const float4 q5=float4(query[query_start+d5],query[query_start+256u+d5],query[query_start+512u+d5],query[query_start+768u+d5]);
    const uint d6=uint(simd_lane)+192u;
    const float4 q6=float4(query[query_start+d6],query[query_start+256u+d6],query[query_start+512u+d6],query[query_start+768u+d6]);
    const uint d7=uint(simd_lane)+224u;
    const float4 q7=float4(query[query_start+d7],query[query_start+256u+d7],query[query_start+512u+d7],query[query_start+768u+d7]);
    for (uint iteration = 0u; iteration < slots_per_simdgroup; ++iteration) {
        const uint slot = slot_tile * slots_per_threadgroup
            + uint(simdgroup) * slots_per_simdgroup + iteration;
        if (slot >= args.row_stride) continue;
        const int position = slot < active_count ? token_ids[id_start + slot] : -1;
        const bool position_valid = metadata_valid
            && active_count <= args.row_stride
            && position >= 0
            && uint(position) < args.cache_capacity
            && uint(position) < sequence_length;
        if (!position_valid) {
            if (simd_lane == 0u) {
                for (uint head = 0u; head < heads_per_group; ++head) {
                    const ulong output = ((ulong)query_index * args.query_heads
                        + first_head + head) * args.row_stride + slot;
                    logits[output] = -INFINITY;
                }
            }
            continue;
        }
        const ulong key_start = ((ulong)uint(position) * args.kv_heads + group.y)
            * args.head_dim;
        float4 dot=0.0f;
        dot+=q0*float(key_cache[key_start+d0]);
        dot+=q1*float(key_cache[key_start+d1]);
        dot+=q2*float(key_cache[key_start+d2]);
        dot+=q3*float(key_cache[key_start+d3]);
        dot+=q4*float(key_cache[key_start+d4]);
        dot+=q5*float(key_cache[key_start+d5]);
        dot+=q6*float(key_cache[key_start+d6]);
        dot+=q7*float(key_cache[key_start+d7]);
        const float partial_0=simd_sum(dot.x),partial_1=simd_sum(dot.y),partial_2=simd_sum(dot.z),partial_3=simd_sum(dot.w);
        if (simd_lane == 0u) {
            const ulong output = ((ulong)query_index * args.query_heads + first_head)
                * args.row_stride + slot;
            logits[output] = partial_0 * args.scale;
            logits[output + args.row_stride] = partial_1 * args.scale;
            logits[output + 2u * args.row_stride] = partial_2 * args.scale;
            logits[output + 3u * args.row_stride] = partial_3 * args.scale;
        }
    }
}


#define QSA_LOGITS_ENTRY(NAME,N) \
kernel void NAME(constant qwen4exp_qsa_packed_attention_args &a [[buffer(0)]], \
        device const float *q [[buffer(1)]],device const half *k [[buffer(2)]], \
        device const int *ids [[buffer(3)]],device const int *counts [[buffer(4)]], \
        device const int *status [[buffer(5)]],device float *out [[buffer(6)]], \
        uint3 group [[threadgroup_position_in_grid]],ushort sg [[simdgroup_index_in_threadgroup]], \
        ushort lane [[thread_index_in_simdgroup]]) { \
    qsa_logits_reuse<N>(a,q,k,ids,counts,status,out,group,sg,lane); \
}
QSA_LOGITS_ENTRY(kernel_qwen4exp_qsa_logits_reuse4_f32,8)
QSA_LOGITS_ENTRY(kernel_qwen4exp_qsa_logits_reuse4_s16_f32,16)
QSA_LOGITS_ENTRY(kernel_qwen4exp_qsa_logits_reuse4_s32_f32,32)
