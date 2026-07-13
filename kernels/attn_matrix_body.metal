#include <metal_stdlib>
using namespace metal;

constant constexpr ushort AMB_GROUP = 8;
constant constexpr ushort AMB_HD = 256;
constant constexpr ushort AMB_C = 32;
constant constexpr ushort AMB_D_CHUNK = 128;
constant constexpr ushort AMB_ROW_BYTES = 288;

struct attn_matrix_body_args {
    uint n_pos;
    uint n_kv_heads;
    uint n_partitions;
    uint rows_per_partition;
    float scale;
};

inline void amb_stage_q(
        device const float *q,
        constant attn_matrix_body_args &args,
        threadgroup half *q_tile,
        uint kvh,
        ushort tid) {
    for (ushort index = tid; index < AMB_GROUP * AMB_HD; index += 256) {
        const ushort head = index / AMB_HD;
        const ushort dim = index % AMB_HD;
        const ushort block = dim / 8;
        const ushort inner = dim % 8;
        const ulong source = (ulong(kvh) * AMB_GROUP + head) * AMB_HD + dim;
        // Each D8 panel is one row-major Q[head, dim] matrix fragment.
        q_tile[block * 64 + head * 8 + inner] = half(q[source] * args.scale);
    }
}

inline void amb_stage_k_owner(
        device const uchar *cache,
        constant attn_matrix_body_args &args,
        threadgroup half *work,
        uint kvh,
        uint tile_start,
        ushort dim_base,
        ushort lane,
        ushort sg) {
    const ushort groups_per_row = AMB_D_CHUNK / 16;
    const ushort records = 8 * groups_per_row;
    threadgroup half *panel = work + sg * 8 * AMB_D_CHUNK;
    for (ushort record = lane; record < records; record += 32) {
        const ushort row = record / groups_per_row;
        const ushort group = record % groups_per_row;
        const ushort source_group = dim_base / 16 + group;
        device const uchar *source =
            cache + ((ulong(tile_start + sg * 8 + row) * args.n_kv_heads + kvh)
                     * AMB_ROW_BYTES);
        const float scale = float(((device const half *)source)[source_group]);
        device const char *payload =
            (device const char *)(source + 32 + source_group * 16);
        for (ushort value = 0; value < 16; ++value) {
            const ushort local_dim = group * 16 + value;
            const ushort d_block = local_dim / 8;
            const ushort d_inner = local_dim % 8;
            // Store K transposed so Q x K^T needs no fragment transpose.
            panel[d_block * 64 + d_inner * 8 + row] =
                half(scale * float(payload[value]));
        }
    }
}

inline void amb_stage_v_owner(
        device const uchar *cache,
        constant attn_matrix_body_args &args,
        threadgroup half *work,
        uint kvh,
        uint tile_start,
        ushort first_sg,
        ushort lane,
        ushort sg) {
    const ushort local_sg = sg - first_sg;
    threadgroup half *panel = work + local_sg * AMB_C * 32;
    for (ushort record = lane; record < AMB_C * 2; record += 32) {
        const ushort row = record / 2;
        const ushort group = record % 2;
        const ushort source_group = sg * 2 + group;
        device const uchar *source =
            cache + ((ulong(tile_start + row) * args.n_kv_heads + kvh) * AMB_ROW_BYTES);
        const float scale = float(((device const half *)source)[source_group]);
        device const char *payload =
            (device const char *)(source + 32 + source_group * 16);
        for (ushort value = 0; value < 16; ++value) {
            panel[row * 32 + group * 16 + value] =
                half(scale * float(payload[value]));
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_attn_matrix_body_g16(
        constant attn_matrix_body_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const uchar *k_cache [[buffer(2)]],
        device const uchar *v_cache [[buffer(3)]],
        device float *o_partial [[buffer(4)]],
        device float *ml_partial [[buffer(5)]],
        threadgroup half *q_tile [[threadgroup(0)]],
        threadgroup half *work [[threadgroup(1)]],
        uint3 tg [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = tg.x;
    const uint partition = tg.z;
    if (kvh >= args.n_kv_heads || partition >= args.n_partitions) return;

    amb_stage_q(q, args, q_tile, kvh, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_state = -INFINITY;
    float l_state = 0.0f;
    float o_acc[AMB_GROUP] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    const uint begin = partition * args.rows_per_partition;
    const uint end = begin + args.rows_per_partition;

    for (uint tile_start = begin; tile_start < end; tile_start += AMB_C) {
        simdgroup_float8x8 score_fragment =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
        if (sg < 4) {
            simdgroup_half8x8 q_fragment;
            simdgroup_half8x8 k_fragment;
            amb_stage_k_owner(k_cache, args, work, kvh, tile_start, 0, lane, sg);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (ushort block = 0; block < 16; ++block) {
                simdgroup_load(q_fragment, q_tile + block * 64, 8, 0, false);
                simdgroup_load(
                    k_fragment,
                    work + sg * 8 * AMB_D_CHUNK + block * 64,
                    8,
                    0,
                    false);
                simdgroup_multiply_accumulate(
                    score_fragment, q_fragment, k_fragment, score_fragment);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            amb_stage_k_owner(
                k_cache, args, work, kvh, tile_start, AMB_D_CHUNK, lane, sg);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (ushort block = 0; block < 16; ++block) {
                simdgroup_load(q_fragment, q_tile + (16 + block) * 64, 8, 0, false);
                simdgroup_load(
                    k_fragment,
                    work + sg * 8 * AMB_D_CHUNK + block * 64,
                    8,
                    0,
                    false);
                simdgroup_multiply_accumulate(
                    score_fragment, q_fragment, k_fragment, score_fragment);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float *scores = (threadgroup float *)work;
        if (sg < 4) {
            simdgroup_store(score_fragment, scores + sg * 8, AMB_C, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float score = scores[sg * AMB_C + lane];
        const float tile_max = simd_max(score);
        const float new_m = max(m_state, tile_max);
        const float factor = isinf(m_state) ? 0.0f : exp2(m_state - new_m);
        const float weight = exp2(score - new_m);
        const float tile_l = simd_sum(weight);
        scores[sg * AMB_C + lane] = weight;
        if (lane == 0) scores[AMB_GROUP * AMB_C + sg] = factor;
        m_state = new_m;
        l_state = l_state * factor + tile_l;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float weights[AMB_GROUP];
        for (ushort g = 0; g < AMB_GROUP; ++g) {
            weights[g] = scores[g * AMB_C + lane];
        }
        const float owned_factor = lane < AMB_GROUP
            ? scores[AMB_GROUP * AMB_C + lane]
            : 0.0f;
        for (ushort g = 0; g < AMB_GROUP; ++g) {
            o_acc[g] *= simd_shuffle(owned_factor, g);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg < 4) {
            amb_stage_v_owner(v_cache, args, work, kvh, tile_start, 0, lane, sg);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup half *panel = work + sg * AMB_C * 32;
            for (ushort row = 0; row < AMB_C; ++row) {
                const float value = float(panel[row * 32 + lane]);
                for (ushort g = 0; g < AMB_GROUP; ++g) {
                    o_acc[g] += simd_shuffle(weights[g], row) * value;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg >= 4) {
            amb_stage_v_owner(v_cache, args, work, kvh, tile_start, 4, lane, sg);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup half *panel = work + (sg - 4) * AMB_C * 32;
            for (ushort row = 0; row < AMB_C; ++row) {
                const float value = float(panel[row * 32 + lane]);
                for (ushort g = 0; g < AMB_GROUP; ++g) {
                    o_acc[g] += simd_shuffle(weights[g], row) * value;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong partial = (ulong(kvh) * args.n_partitions + partition) * AMB_GROUP;
    const ushort dim = sg * 32 + lane;
    for (ushort g = 0; g < AMB_GROUP; ++g) {
        o_partial[(partial + g) * AMB_HD + dim] = o_acc[g];
    }
    if (lane == 0) {
        ml_partial[(partial + sg) * 2] = m_state;
        ml_partial[(partial + sg) * 2 + 1] = l_state;
    }
}
