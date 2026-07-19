#include <metal_stdlib>
using namespace metal;

constant constexpr ushort ADFM_GROUP = 8;
constant constexpr ushort ADFM_HD = 256;
constant constexpr ushort ADFM_C = 32;

struct attn_direct_f16_matrix_args {
    uint n_pos;
    uint n_kv_heads;
    uint kv_stride;
    uint n_partitions;
    uint rows_per_partition;
    float scale;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_attn_direct_f16_matrix_g8_c32(
        constant attn_direct_f16_matrix_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const half *k_cache [[buffer(2)]],
        device const half *v_cache [[buffer(3)]],
        device float *o_partial [[buffer(4)]],
        device float *ml_partial [[buffer(5)]],
        threadgroup half *q_tile [[threadgroup(0)]],
        threadgroup float *scores [[threadgroup(1)]],
        uint3 tg [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = tg.x;
    const uint partition = tg.z;
    if (kvh >= args.n_kv_heads || partition >= args.n_partitions) return;

    const uint begin = partition * args.rows_per_partition;
    const uint end = min(begin + args.rows_per_partition, args.n_pos);
    if (begin >= end) return;

    for (ushort index = tid; index < ADFM_GROUP * ADFM_HD; index += 256) {
        q_tile[index] = half(q[(ulong)kvh * ADFM_GROUP * ADFM_HD + index]
                           * args.scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_state = -INFINITY;
    float l_state = 0.0f;
    float o_acc[ADFM_GROUP] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    threadgroup float *factors = scores + ADFM_GROUP * ADFM_C;

    for (uint tile_start = begin; tile_start < end; tile_start += ADFM_C) {
        const ushort count = ushort(min(uint(ADFM_C), end - tile_start));

        if (sg < 4) {
            simdgroup_float8x8 score_fragment =
                make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (ushort d0 = 0; d0 < ADFM_HD; d0 += 8) {
                simdgroup_half8x8 q_fragment;
                simdgroup_half8x8 k_fragment;
                simdgroup_load(q_fragment, q_tile + d0, ADFM_HD);
                const ulong k_base =
                    (ulong)(tile_start + sg * 8) * args.kv_stride
                    + (ulong)kvh * ADFM_HD + d0;
                simdgroup_load(
                    k_fragment,
                    k_cache + k_base,
                    args.kv_stride,
                    ulong2(0, 0),
                    true);
                simdgroup_multiply_accumulate(
                    score_fragment,
                    q_fragment,
                    k_fragment,
                    score_fragment);
            }
            simdgroup_store(
                score_fragment,
                scores + sg * 8,
                ADFM_C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = lane < count ? scores[sg * ADFM_C + lane] : -INFINITY;
        const float tile_max = simd_max(score);
        const float new_m = max(m_state, tile_max);
        const float factor = isinf(m_state) ? 0.0f : exp2(m_state - new_m);
        const float weight = lane < count ? exp2(score - new_m) : 0.0f;
        const float tile_l = simd_sum(weight);
        scores[sg * ADFM_C + lane] = weight;
        if (lane == 0) factors[sg] = factor;
        m_state = new_m;
        l_state = l_state * factor + tile_l;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float weights[ADFM_GROUP];
        for (ushort g = 0; g < ADFM_GROUP; ++g) {
            o_acc[g] *= factors[g];
            weights[g] = scores[g * ADFM_C + lane];
        }
        const ushort dim = sg * 32 + lane;
        for (ushort c = 0; c < count; ++c) {
            const ulong v_at = (ulong)(tile_start + c) * args.kv_stride
                             + (ulong)kvh * ADFM_HD + dim;
            const float value = float(v_cache[v_at]);
            for (ushort g = 0; g < ADFM_GROUP; ++g) {
                o_acc[g] += simd_shuffle(weights[g], c) * value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong partial =
        ((ulong)kvh * args.n_partitions + partition) * ADFM_GROUP;
    const ushort dim = sg * 32 + lane;
    for (ushort g = 0; g < ADFM_GROUP; ++g) {
        o_partial[(partial + g) * ADFM_HD + dim] = o_acc[g];
    }
    if (lane == 0) {
        ml_partial[(partial + sg) * 2] = m_state;
        ml_partial[(partial + sg) * 2 + 1] = l_state;
    }
}
