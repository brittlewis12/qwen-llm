#include <metal_stdlib>
using namespace metal;

constant constexpr ushort LF_GROUP = 8;
constant constexpr ushort LF_HD = 256;
constant constexpr ushort LF_C = 32;
constant constexpr ushort LF_ROW_Q8 = 288;

struct attn_long_fixed_args {
    uint n_q_heads;
    uint n_kv_heads;
    uint head_dim;
    uint n_pos;
    uint kv_stride;
    uint n_partitions;
    uint rows_per_partition;
    float scale;
};

template <bool Q8>
inline float lf_load(device const uchar *cache, constant attn_long_fixed_args &args,
                     uint pos, uint kvh, ushort dim) {
    if constexpr (!Q8) {
        device const half *p = (device const half *)cache;
        return float(p[(ulong)pos * args.kv_stride + (ulong)kvh * LF_HD + dim]);
    } else {
        device const uchar *row = cache + ((ulong)pos * args.n_kv_heads + kvh) * LF_ROW_Q8;
        const ushort group = dim >> 4;
        device const uchar *block = row + group * 18;
        const float scale = float(*(device const half *)block);
        return scale * float(((device const char *)(block + 2))[dim & 15]);
    }
}

template <bool Q8>
inline void attn_long_fixed_body(
        constant attn_long_fixed_args &args,
        device const float *q, device const uchar *k_cache, device const uchar *v_cache,
        device float *o_partial, device float *ml_partial,
        threadgroup float *qk_part, threadgroup float *weights,
        threadgroup float *state, threadgroup half *sq,
        uint3 tg, ushort tid, ushort lane, ushort sg) {
    const uint kvh = tg.x;
    const uint part = tg.z;
    if (kvh >= args.n_kv_heads || part >= args.n_partitions) return;
    const ushort dim = sg * 32 + lane;
    const uint begin = part * args.rows_per_partition;
    const uint end = min(begin + args.rows_per_partition, args.n_pos);
    float o[LF_GROUP] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (ushort i = tid; i < LF_GROUP * LF_HD; i += 256) {
        sq[i] = half(q[(ulong)kvh * LF_GROUP * LF_HD + i] * args.scale);
    }
    if (tid < LF_GROUP) {
        state[tid * 3] = -INFINITY;
        state[tid * 3 + 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile = begin; tile < end; tile += LF_C) {
        const ushort count = ushort(min(uint(LF_C), end - tile));
        for (ushort c = 0; c < count; ++c) {
            const float k = lf_load<Q8>(k_cache, args, tile + c, kvh, dim);
            for (ushort g = 0; g < LF_GROUP; ++g) {
                const float qv = float(sq[g * LF_HD + dim]);
                const float sum = simd_sum(qv * k);
                if (lane == 0) qk_part[(sg * LF_GROUP + g) * LF_C + c] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            for (ushort g = 0; g < LF_GROUP; ++g) {
                float score = -INFINITY;
                if (lane < count) {
                    score = 0.0f;
                    for (ushort s = 0; s < 8; ++s) {
                        score += qk_part[(s * LF_GROUP + g) * LF_C + lane];
                    }
                }
                const float tile_max = simd_max(score);
                const float old_m = state[g * 3];
                const float new_m = max(old_m, tile_max);
                const float factor = isinf(old_m) ? 0.0f : exp2(old_m - new_m);
                const float w = lane < count ? exp2(score - new_m) : 0.0f;
                const float tile_l = simd_sum(w);
                weights[g * LF_C + lane] = w;
                if (lane == 0) {
                    state[g * 3] = new_m;
                    state[g * 3 + 1] = state[g * 3 + 1] * factor + tile_l;
                    state[g * 3 + 2] = factor;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort g = 0; g < LF_GROUP; ++g) o[g] *= state[g * 3 + 2];
        for (ushort c = 0; c < count; ++c) {
            const float v = lf_load<Q8>(v_cache, args, tile + c, kvh, dim);
            for (ushort g = 0; g < LF_GROUP; ++g) o[g] += weights[g * LF_C + c] * v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong base = ((ulong)kvh * args.n_partitions + part) * LF_GROUP * LF_HD;
    for (ushort g = 0; g < LF_GROUP; ++g) o_partial[base + g * LF_HD + dim] = o[g];
    if (tid < LF_GROUP) {
        const ulong ml = ((ulong)kvh * args.n_partitions + part) * LF_GROUP * 2;
        ml_partial[ml + tid * 2] = state[tid * 3];
        ml_partial[ml + tid * 2 + 1] = state[tid * 3 + 1];
    }
}

#define LF_KERNEL(NAME, Q8) \
[[max_total_threads_per_threadgroup(256)]] kernel void NAME( \
    constant attn_long_fixed_args &args [[buffer(0)]], \
    device const float *q [[buffer(1)]], device const uchar *k [[buffer(2)]], \
    device const uchar *v [[buffer(3)]], device float *o [[buffer(4)]], \
    device float *ml [[buffer(5)]], threadgroup float *qp [[threadgroup(0)]], \
    threadgroup float *w [[threadgroup(1)]], threadgroup float *st [[threadgroup(2)]], \
    threadgroup half *sq [[threadgroup(3)]], \
    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) { \
    attn_long_fixed_body<Q8>(args, q, k, v, o, ml, qp, w, st, sq, tg, tid, lane, sg); \
}

LF_KERNEL(kernel_attn_long_fixed_g8_c32_f16_f32, false)
LF_KERNEL(kernel_attn_long_fixed_g8_c32_g16q8_f32, true)
