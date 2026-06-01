// Matrix-vector multiply, F32 weights × F32 vector → F32 result.
//
// y[o] = sum_i W[o, i] * x[i],     W has shape [n_in, n_out] in GGUF
// (i.e. ne[0]=n_in fastest, ne[1]=n_out slowest), so row o lives at
// offset `o * n_in` in W's data.
//
// Design:
//   * One simdgroup (32 threads) per output row.
//   * Each thread strides through the n_in axis with stride 32 (vec4-loaded).
//   * simd_sum reduces across the simdgroup.
//   * Multiple rows per threadgroup so we amortize x's reads through the
//     L1 cache (every simdgroup in the same threadgroup reads the same
//     x slice).
//
// Threads/threadgroup = ROWS_PER_TG * 32. We dispatch
//   threadgroups = (n_out + ROWS_PER_TG - 1) / ROWS_PER_TG.
//
// Tradeoffs vs llama.cpp's templated NR0 kernel: simpler, fewer registers
// per thread, no row-merging inside a simdgroup. We'll see how close we
// get to bandwidth peak; if it's within 70% of llama.cpp's, the
// simplicity is worth keeping until profile says otherwise.

#include <metal_stdlib>
using namespace metal;

struct mat_vec_args {
    uint n_in;
    uint n_out;
};

struct mat_mat_args {
    uint n_in;
    uint n_out;
    uint n_query;
};

constant float iq4nl_values[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f,
};

struct block_q4_0_local {
    half d;
    uchar qs[16];
};

struct block_q4_1_local {
    half d;
    half m;
    uchar qs[16];
};

struct block_q3_k_local {
    uchar hmask[32];
    uchar qs[64];
    uchar scales[12];
    half d;
};

struct block_q2_k_local {
    uchar scales[16];
    uchar qs[64];
    half d;
    half dmin;
};

struct block_iq4_nl_local {
    half d;
    uchar qs[16];
};

struct block_iq4_xs_local {
    half d;
    ushort scales_h;
    uchar scales_l[4];
    uchar qs[128];
};

static inline float bf16_to_float(ushort v) {
    return as_type<float>(((uint)v) << 16);
}

static inline float deq_q4_0(device const block_q4_0_local & b, uint i) {
    const uchar q = b.qs[i & 15u];
    const int v = (i < 16u) ? (int)(q & 0x0f) : (int)(q >> 4);
    return float(b.d) * float(v - 8);
}

static inline float deq_q4_1(device const block_q4_1_local & b, uint i) {
    const uchar q = b.qs[i & 15u];
    const int v = (i < 16u) ? (int)(q & 0x0f) : (int)(q >> 4);
    return float(b.d) * float(v) + float(b.m);
}

static inline int q3_k_scale_int(device const block_q3_k_local & b, uint sub) {
    const uint scale_2 = uint(b.scales[sub & 7u]);
    const uint scale_1 = uint(b.scales[8u + (sub & 3u)]);
    const uint quarter = sub >> 2;
    const uint kmask1 = quarter > 1u ? (quarter > 2u ? 192u : 48u) : (quarter > 0u ? 12u : 3u);
    const uint kmask2 = sub >= 8u ? 0xf0u : 0x0fu;
    uint raw;
    if ((quarter & 1u) != 0u) {
        raw = (scale_2 & kmask2) | ((scale_1 & kmask1) << 2);
    } else {
        raw = (scale_2 & kmask2) | ((scale_1 & kmask1) << 4);
    }
    if (sub >= 8u) {
        raw >>= 4;
    }
    return int(raw) - 32;
}

static inline float deq_q3_k(device const block_q3_k_local & b, uint i) {
    const uint sub = i >> 4;
    const uint lane = i & 15u;
    const uint q_index = 32u * (sub >> 3) + 16u * (sub & 1u) + lane;
    const uint h_index = 16u * (sub & 1u) + lane;
    const uint shift = ((sub >> 1) & 3u) * 2u;
    const int q_low = int((uint(b.qs[q_index]) >> shift) & 3u);
    const uint h_bit = 1u << (sub >> 1);
    const int q = q_low - ((uint(b.hmask[h_index]) & h_bit) != 0u ? 0 : 4);
    return float(b.d) * float(q3_k_scale_int(b, sub)) * float(q);
}

static inline float deq_q2_k(device const block_q2_k_local & b, uint i) {
    const uint sub = i >> 4;
    const uint lane = i & 15u;
    const uint q_index = 32u * (sub >> 3) + 16u * (sub & 1u) + lane;
    const uint shift = ((sub >> 1) & 3u) * 2u;
    const uint q = (uint(b.qs[q_index]) >> shift) & 3u;
    const uint sc = uint(b.scales[sub]);
    return float(b.d) * float(sc & 0x0fu) * float(q) - float(b.dmin) * float(sc >> 4);
}

static inline float deq_iq4_nl(device const block_iq4_nl_local & b, uint i) {
    const uchar q = b.qs[i & 15u];
    const uint idx = (i < 16u) ? uint(q & 0x0f) : uint(q >> 4);
    return float(b.d) * iq4nl_values[idx];
}

static inline float deq_iq4_xs(device const block_iq4_xs_local & b, uint i) {
    const uint ib32 = i >> 5;
    const uint lane = i & 31u;
    const uint scale_l = (uint(b.scales_l[ib32 >> 1]) >> (4u * (ib32 & 1u))) & 0x0fu;
    const uint scale_h = (uint(b.scales_h) >> (2u * ib32)) & 3u;
    const float d = float(b.d) * float(int(scale_l | (scale_h << 4)) - 32);
    const uchar q = b.qs[ib32 * 16u + (lane & 15u)];
    const uint idx = (lane < 16u) ? uint(q & 0x0f) : uint(q >> 4);
    return d * iq4nl_values[idx];
}

// Vec4 path. Requires n_in % 4 == 0 (true for our hidden / FFN dims).
// One simdgroup per output row. Multiple rows per threadgroup.
#ifndef MAT_VEC_ROWS_PER_TG
#define MAT_VEC_ROWS_PER_TG 4
#endif

kernel void kernel_mat_vec_f32_f32(
        constant mat_vec_args & args   [[buffer(0)]],
        device const float    * weight [[buffer(1)]], // [n_in, n_out]
        device const float    * x      [[buffer(2)]], // [n_in]
        device       float    * y      [[buffer(3)]], // [n_out]
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig * MAT_VEC_ROWS_PER_TG + sgitg;
    if (row >= args.n_out) return;

    const uint n_in_v4 = args.n_in / 4;
    device const float4 * w4 = (device const float4 *)(weight + row * args.n_in);
    device const float4 * x4 = (device const float4 *)x;

    float sum = 0.0f;
    for (uint i = tiisg; i < n_in_v4; i += 32) {
        float4 a = w4[i];
        float4 b = x4[i];
        sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    // Tail elements (n_in not divisible by 4 — won't trigger for our shapes
    // but we keep it correct).
    const uint tail_start = n_in_v4 * 4;
    for (uint i = tail_start + tiisg; i < args.n_in; i += 32) {
        sum += weight[row * args.n_in + i] * x[i];
    }

    sum = simd_sum(sum);
    if (tiisg == 0) {
        y[row] = sum;
    }
}

kernel void kernel_mat_vec_f16_f32(
        constant mat_vec_args & args   [[buffer(0)]],
        device const half     * weight [[buffer(1)]],
        device const float    * x      [[buffer(2)]],
        device       float    * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig * MAT_VEC_ROWS_PER_TG + sgitg;
    if (row >= args.n_out) return;

    const uint n_in_v4 = args.n_in / 4;
    device const half4 * w4 = (device const half4 *)(weight + row * args.n_in);
    device const float4 * x4 = (device const float4 *)x;

    float sum = 0.0f;
    for (uint i = tiisg; i < n_in_v4; i += 32) {
        float4 a = float4(w4[i]);
        float4 b = x4[i];
        sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    const uint tail_start = n_in_v4 * 4;
    for (uint i = tail_start + tiisg; i < args.n_in; i += 32) {
        sum += float(weight[row * args.n_in + i]) * x[i];
    }

    sum = simd_sum(sum);
    if (tiisg == 0) {
        y[row] = sum;
    }
}

kernel void kernel_mat_vec_bf16_f32(
        constant mat_vec_args & args   [[buffer(0)]],
        device const ushort   * weight [[buffer(1)]],
        device const float    * x      [[buffer(2)]],
        device       float    * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig * MAT_VEC_ROWS_PER_TG + sgitg;
    if (row >= args.n_out) return;

    const uint n_in_v4 = args.n_in / 4;
    device const ushort4 * w4 = (device const ushort4 *)(weight + row * args.n_in);
    device const float4 * x4 = (device const float4 *)x;

    float sum = 0.0f;
    for (uint i = tiisg; i < n_in_v4; i += 32) {
        ushort4 u = w4[i];
        float4 a = float4(bf16_to_float(u.x), bf16_to_float(u.y), bf16_to_float(u.z), bf16_to_float(u.w));
        float4 b = x4[i];
        sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    const uint tail_start = n_in_v4 * 4;
    for (uint i = tail_start + tiisg; i < args.n_in; i += 32) {
        sum += bf16_to_float(weight[row * args.n_in + i]) * x[i];
    }

    sum = simd_sum(sum);
    if (tiisg == 0) {
        y[row] = sum;
    }
}

kernel void kernel_mat_vec_q4_0_f32(
        constant mat_vec_args & args   [[buffer(0)]],
        device const block_q4_0_local * weight [[buffer(1)]],
        device const float    * x      [[buffer(2)]],
        device       float    * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig * MAT_VEC_ROWS_PER_TG + sgitg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 32u;
    device const block_q4_0_local * row_blocks = weight + row * nb;
    float sum = 0.0f;
    for (uint bidx = tiisg; bidx < nb; bidx += 32u) {
        device const block_q4_0_local & b = row_blocks[bidx];
        const uint base = bidx * 32u;
        for (uint j = 0; j < 32u; ++j) {
            sum += deq_q4_0(b, j) * x[base + j];
        }
    }

    sum = simd_sum(sum);
    if (tiisg == 0) {
        y[row] = sum;
    }
}

kernel void kernel_mat_vec_q4_1_f32(
        constant mat_vec_args & args   [[buffer(0)]],
        device const block_q4_1_local * weight [[buffer(1)]],
        device const float    * x      [[buffer(2)]],
        device       float    * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig * MAT_VEC_ROWS_PER_TG + sgitg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 32u;
    device const block_q4_1_local * row_blocks = weight + row * nb;
    float sum = 0.0f;
    for (uint bidx = tiisg; bidx < nb; bidx += 32u) {
        device const block_q4_1_local & b = row_blocks[bidx];
        const uint base = bidx * 32u;
        for (uint j = 0; j < 32u; ++j) {
            sum += deq_q4_1(b, j) * x[base + j];
        }
    }

    sum = simd_sum(sum);
    if (tiisg == 0) {
        y[row] = sum;
    }
}

kernel void kernel_mat_vec_q3_K_f32(
        constant mat_vec_args & args   [[buffer(0)]],
        device const block_q3_k_local * weight [[buffer(1)]],
        device const float    * x      [[buffer(2)]],
        device       float    * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig * MAT_VEC_ROWS_PER_TG + sgitg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 256u;
    device const block_q3_k_local * row_blocks = weight + row * nb;
    float sum = 0.0f;
    for (uint bidx = 0; bidx < nb; ++bidx) {
        device const block_q3_k_local & b = row_blocks[bidx];
        const uint base = bidx * 256u;
        for (uint j = tiisg; j < 256u; j += 32u) {
            sum += deq_q3_k(b, j) * x[base + j];
        }
    }

    sum = simd_sum(sum);
    if (tiisg == 0) {
        y[row] = sum;
    }
}

kernel void kernel_mat_vec_q2_K_f32(
        constant mat_vec_args & args   [[buffer(0)]],
        device const block_q2_k_local * weight [[buffer(1)]],
        device const float    * x      [[buffer(2)]],
        device       float    * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig * MAT_VEC_ROWS_PER_TG + sgitg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 256u;
    device const block_q2_k_local * row_blocks = weight + row * nb;
    float sum = 0.0f;
    for (uint bidx = 0; bidx < nb; ++bidx) {
        device const block_q2_k_local & b = row_blocks[bidx];
        const uint base = bidx * 256u;
        for (uint j = tiisg; j < 256u; j += 32u) {
            sum += deq_q2_k(b, j) * x[base + j];
        }
    }

    sum = simd_sum(sum);
    if (tiisg == 0) {
        y[row] = sum;
    }
}

kernel void kernel_mat_vec_q2_K_f32_fast(
        constant mat_vec_args & args [[buffer(0)]],
        device const block_q2_k_local * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NR0 = 4;
    const short NSG = 2;
    const uint first_row = (tgpig * NSG + uint(sgitg)) * NR0;
    if (first_row >= args.n_out) return;

    const uint nb = args.n_in / 256u;
    const short ix = tiisg / 8;  // 0..3
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    device const float * y4 = x + uint(ix) * 256u + 128u * uint(iq) + 8u * uint(ir);
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint ib = uint(ix); ib < nb; ib += 4u) {
        float yl[32];
        float4 sumy = {0.0f, 0.0f, 0.0f, 0.0f};
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
            yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
        }

        for (short row = 0; row < NR0; ++row) {
            const uint out_row = first_row + uint(row);
            if (out_row >= args.n_out) continue;
            device const block_q2_k_local & b = weight[out_row * nb + ib];
            device const uchar * sc = b.scales + 8 * iq + is;
            device const ushort * qs = (device const ushort *)b.qs + 16 * iq + 4 * ir;

            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
            }

            const float d = float(b.d);
            const float dmin = float(b.dmin) * (1.0f / 16.0f);
            sumf[row] += d * ((acc1[0] + (1.0f / 256.0f) * acc2[0]) * (sc[0] & 0x0f) * 1.0f
                            + (acc1[1] + (1.0f / 256.0f) * acc2[1]) * (sc[2] & 0x0f) * (1.0f / 4.0f)
                            + (acc1[2] + (1.0f / 256.0f) * acc2[2]) * (sc[4] & 0x0f) * (1.0f / 16.0f)
                            + (acc1[3] + (1.0f / 256.0f) * acc2[3]) * (sc[6] & 0x0f) * (1.0f / 64.0f))
                       - dmin * (sumy[0] * (sc[0] & 0xf0)
                               + sumy[1] * (sc[2] & 0xf0)
                               + sumy[2] * (sc[4] & 0xf0)
                               + sumy[3] * (sc[6] & 0xf0));
        }
        y4 += 4u * 256u;
    }

    for (short row = 0; row < NR0; ++row) {
        const uint out_row = first_row + uint(row);
        if (out_row >= args.n_out) continue;
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) y[out_row] = total;
    }
}

kernel void kernel_mat_vec_q3_K_f32_fast(
        constant mat_vec_args & args [[buffer(0)]],
        device const block_q3_k_local * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NR0 = 2;
    const short NSG = 2;
    const uint first_row = (tgpig * NSG + uint(sgitg)) * NR0;
    if (first_row >= args.n_out) return;

    const uint nb = args.n_in / 256u;
    const short tid = tiisg / 4;
    const short ix = tiisg % 4;
    const short ip = tid / 4;
    const short il = 2 * ((tid % 4) / 2);
    const short ir = tid % 2;
    const short l0 = 8 * ir;

    const ushort4 mm[4] = {
        ushort4(0x0001, 0x0100, 0x0002, 0x0200),
        ushort4(0x0004, 0x0400, 0x0008, 0x0800),
        ushort4(0x0010, 0x1000, 0x0020, 0x2000),
        ushort4(0x0040, 0x4000, 0x0080, 0x8000),
    };
    const uint4 qm[2] = {
        uint4(0x0003, 0x0300, 0x000c, 0x0c00),
        uint4(0x0030, 0x3000, 0x00c0, 0xc000),
    };

    const ushort4 hm = mm[2 * ip + il / 2];
    const short shift = 2 * il;
    const float v1 = il == 0 ? 4.0f : 64.0f;
    const float v2 = 4.0f * v1;
    const uint s_shift1 = 4u * uint(ip);
    const uint s_shift2 = s_shift1 + uint(il);
    const short q_offset = 32 * ip + l0;
    const short y_offset = 128 * ip + 32 * il + l0;

    device const float * y1 = x + uint(ix) * 256u + uint(y_offset);
    uint scales32;
    uint aux32;
    thread ushort * scales16 = (thread ushort *)&scales32;
    float sumf1[2] = {0.0f, 0.0f};
    float sumf2[2] = {0.0f, 0.0f};

    for (uint ib = uint(ix); ib < nb; ib += 4u) {
        float yl[32];
        for (short l = 0; l < 8; ++l) {
            yl[l +  0] = y1[l +  0];
            yl[l +  8] = y1[l + 16];
            yl[l + 16] = y1[l + 32];
            yl[l + 24] = y1[l + 48];
        }

        for (short row = 0; row < NR0; ++row) {
            const uint out_row = first_row + uint(row);
            if (out_row >= args.n_out) continue;

            device const block_q3_k_local & b = weight[out_row * nb + ib];
            device const ushort * q = (device const ushort *)(b.qs + q_offset);
            device const ushort * h = (device const ushort *)(b.hmask + l0);
            device const ushort * a = (device const ushort *)(b.scales);
            const float d_all = float(b.d);

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030u;
            scales16[0] = a[il + 0];
            scales16[1] = a[il + 1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0fu) | aux32;

            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            float s4 = 0.0f;
            float s5 = 0.0f;
            float s6 = 0.0f;
            for (short l = 0; l < 8; l += 2) {
                const uint qs = uint(q[l / 2]);
                const uint hs = uint(h[l / 2]);
                s1 += yl[l +  0] * float(qs & qm[il / 2][0]);
                s2 += yl[l +  1] * float(qs & qm[il / 2][1]);
                s3 += ((hs & uint(hm[0])) != 0u ? 0.0f : yl[l +  0])
                    + ((hs & uint(hm[1])) != 0u ? 0.0f : yl[l +  1]);
                s4 += yl[l + 16] * float(qs & qm[il / 2][2]);
                s5 += yl[l + 17] * float(qs & qm[il / 2][3]);
                s6 += ((hs & uint(hm[2])) != 0u ? 0.0f : yl[l + 16])
                    + ((hs & uint(hm[3])) != 0u ? 0.0f : yl[l + 17]);
            }
            float d1 = d_all * (s1 + (1.0f / 256.0f) * s2 - s3 * v1);
            float d2 = d_all * (s4 + (1.0f / 256.0f) * s5 - s6 * v2);
            sumf1[row] += d1 * (float((scales32 >>  0) & 0xffu) - 32.0f);
            sumf2[row] += d2 * (float((scales32 >> 16) & 0xffu) - 32.0f);

            s1 = s2 = s3 = s4 = s5 = s6 = 0.0f;
            for (short l = 0; l < 8; l += 2) {
                const uint qs = uint(q[l / 2 + 8]);
                const uint hs = uint(h[l / 2 + 8]);
                s1 += yl[l +  8] * float(qs & qm[il / 2][0]);
                s2 += yl[l +  9] * float(qs & qm[il / 2][1]);
                s3 += ((hs & uint(hm[0])) != 0u ? 0.0f : yl[l +  8])
                    + ((hs & uint(hm[1])) != 0u ? 0.0f : yl[l +  9]);
                s4 += yl[l + 24] * float(qs & qm[il / 2][2]);
                s5 += yl[l + 25] * float(qs & qm[il / 2][3]);
                s6 += ((hs & uint(hm[2])) != 0u ? 0.0f : yl[l + 24])
                    + ((hs & uint(hm[3])) != 0u ? 0.0f : yl[l + 25]);
            }
            d1 = d_all * (s1 + (1.0f / 256.0f) * s2 - s3 * v1);
            d2 = d_all * (s4 + (1.0f / 256.0f) * s5 - s6 * v2);
            sumf1[row] += d1 * (float((scales32 >>  8) & 0xffu) - 32.0f);
            sumf2[row] += d2 * (float((scales32 >> 24) & 0xffu) - 32.0f);
        }
        y1 += 4u * 256u;
    }

    for (short row = 0; row < NR0; ++row) {
        const uint out_row = first_row + uint(row);
        if (out_row >= args.n_out) continue;
        const float row_sum = (sumf1[row] + 0.25f * sumf2[row]) / float(1u << uint(shift));
        const float total = simd_sum(row_sum);
        if (tiisg == 0) y[out_row] = total;
    }
}

kernel void kernel_mat_vec_iq4_nl_f32(
        constant mat_vec_args & args   [[buffer(0)]],
        device const block_iq4_nl_local * weight [[buffer(1)]],
        device const float    * x      [[buffer(2)]],
        device       float    * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig * MAT_VEC_ROWS_PER_TG + sgitg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 32u;
    device const block_iq4_nl_local * row_blocks = weight + row * nb;
    float sum = 0.0f;
    for (uint bidx = tiisg; bidx < nb; bidx += 32u) {
        device const block_iq4_nl_local & b = row_blocks[bidx];
        const uint base = bidx * 32u;
        for (uint j = 0; j < 32u; ++j) {
            sum += deq_iq4_nl(b, j) * x[base + j];
        }
    }

    sum = simd_sum(sum);
    if (tiisg == 0) {
        y[row] = sum;
    }
}

kernel void kernel_mat_vec_iq4_xs_f32(
        constant mat_vec_args & args   [[buffer(0)]],
        device const block_iq4_xs_local * weight [[buffer(1)]],
        device const float    * x      [[buffer(2)]],
        device       float    * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig * MAT_VEC_ROWS_PER_TG + sgitg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 256u;
    device const block_iq4_xs_local * row_blocks = weight + row * nb;
    float sum = 0.0f;
    for (uint bidx = 0; bidx < nb; ++bidx) {
        device const block_iq4_xs_local & b = row_blocks[bidx];
        const uint base = bidx * 256u;
        for (uint j = tiisg; j < 256u; j += 32u) {
            sum += deq_iq4_xs(b, j) * x[base + j];
        }
    }

    sum = simd_sum(sum);
    if (tiisg == 0) {
        y[row] = sum;
    }
}

kernel void kernel_mat_vec_iq4_xs_f32_fast(
        constant mat_vec_args & args [[buffer(0)]],
        device const block_iq4_xs_local * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        threadgroup float * lut [[threadgroup(0)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NR0 = 2;
    const short NSG = 2;
    const uint first_row = (tgpig * NSG + uint(sgitg)) * NR0;
    if (first_row >= args.n_out) return;

    lut[tiisg] = iq4nl_values[tiisg & 15];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint nb = args.n_in / 256u;
    const short ix = tiisg / 16;
    const short it = tiisg % 16;
    const short ib = it / 2;
    const short il = it % 2;
    device const float * yb = x + uint(ix) * 256u + uint(ib) * 32u + uint(il) * 8u;

    float sumf[2] = {0.0f, 0.0f};
    uint aux32[2];
    thread const uchar * q8 = (thread const uchar *)aux32;

    for (uint ibl = uint(ix); ibl < nb; ibl += 2u) {
        device const float4 * y4 = (device const float4 *)yb;
        const float4 yl0 = y4[0];
        const float4 yl1 = y4[4];
        const float4 yl2 = y4[1];
        const float4 yl3 = y4[5];

        for (short row = 0; row < NR0; ++row) {
            const uint out_row = first_row + uint(row);
            if (out_row >= args.n_out) continue;
            device const block_iq4_xs_local & b = weight[out_row * nb + ibl];
            device const uint * q4 = (device const uint *)(b.qs + 16 * ib + 8 * il);

            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            const float4 qf10 = {lut[q8[0]], lut[q8[1]], lut[q8[2]], lut[q8[3]]};
            const float4 qf20 = {lut[q8[4]], lut[q8[5]], lut[q8[6]], lut[q8[7]]};
            acc1 += yl0 * qf10;
            acc2 += yl1 * qf20;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            const float4 qf11 = {lut[q8[0]], lut[q8[1]], lut[q8[2]], lut[q8[3]]};
            const float4 qf21 = {lut[q8[4]], lut[q8[5]], lut[q8[6]], lut[q8[7]]};
            acc1 += yl2 * qf11;
            acc2 += yl3 * qf21;
            acc1 += acc2;

            const int ls = int(((uint(b.scales_l[ib / 2]) >> (4 * (ib & 1))) & 0x0f)
                         | (((uint(b.scales_h) >> (2 * ib)) & 3u) << 4)) - 32;
            sumf[row] += float(b.d) * float(ls) * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }
        yb += 2u * 256u;
    }

    for (short row = 0; row < NR0; ++row) {
        const uint out_row = first_row + uint(row);
        if (out_row >= args.n_out) continue;
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) y[out_row] = total;
    }
}

// F32 mat-mat specialized for small n_out / prompt-time reuse of one weight row
// across many query rows. Output layout matches the quant mat-mat path:
// column-major [n_out, n_query], so element (q, o) writes to y[o + q*n_out].
kernel void kernel_mat_mat_f32_f32(
        constant mat_mat_args & args [[buffer(0)]],
        device const float   * weight [[buffer(1)]], // [n_in, n_out]
        device const float   * x      [[buffer(2)]], // [n_query, n_in] row-major
        device       float   * y      [[buffer(3)]], // [n_out, n_query] col-major
        threadgroup  float   * wtile  [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    const uint q = tgpig.y * 32u + tiisg;
    if (row >= args.n_out) return;

    float acc = 0.0f;
    device const float * w_row = weight + row * args.n_in;
    for (uint ib = 0; ib < args.n_in; ib += 32) {
        if (ib + tiisg < args.n_in) {
            wtile[tiisg] = w_row[ib + tiisg];
        } else {
            wtile[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q < args.n_query) {
            device const float * x_row = x + q * args.n_in + ib;
            const uint limit = min(32u, args.n_in - ib);
            for (uint j = 0; j < limit; ++j) {
                acc += x_row[j] * wtile[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (q < args.n_query) {
        y[row + q * args.n_out] = acc;
    }
}

kernel void kernel_mat_mat_f16_f32(
        constant mat_mat_args & args [[buffer(0)]],
        device const half    * weight [[buffer(1)]],
        device const float   * x      [[buffer(2)]],
        device       float   * y      [[buffer(3)]],
        threadgroup  float   * wtile  [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    const uint q = tgpig.y * 32u + tiisg;
    if (row >= args.n_out) return;

    float acc = 0.0f;
    device const half * w_row = weight + row * args.n_in;
    for (uint ib = 0; ib < args.n_in; ib += 32) {
        if (ib + tiisg < args.n_in) {
            wtile[tiisg] = float(w_row[ib + tiisg]);
        } else {
            wtile[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q < args.n_query) {
            device const float * x_row = x + q * args.n_in + ib;
            const uint limit = min(32u, args.n_in - ib);
            for (uint j = 0; j < limit; ++j) {
                acc += x_row[j] * wtile[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (q < args.n_query) {
        y[row + q * args.n_out] = acc;
    }
}

kernel void kernel_mat_mat_bf16_f32(
        constant mat_mat_args & args [[buffer(0)]],
        device const ushort  * weight [[buffer(1)]],
        device const float   * x      [[buffer(2)]],
        device       float   * y      [[buffer(3)]],
        threadgroup  float   * wtile  [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    const uint q = tgpig.y * 32u + tiisg;
    if (row >= args.n_out) return;

    float acc = 0.0f;
    device const ushort * w_row = weight + row * args.n_in;
    for (uint ib = 0; ib < args.n_in; ib += 32) {
        if (ib + tiisg < args.n_in) {
            wtile[tiisg] = bf16_to_float(w_row[ib + tiisg]);
        } else {
            wtile[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q < args.n_query) {
            device const float * x_row = x + q * args.n_in + ib;
            const uint limit = min(32u, args.n_in - ib);
            for (uint j = 0; j < limit; ++j) {
                acc += x_row[j] * wtile[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (q < args.n_query) {
        y[row + q * args.n_out] = acc;
    }
}

kernel void kernel_mat_mat_q4_0_f32(
        constant mat_mat_args & args [[buffer(0)]],
        device const block_q4_0_local * weight [[buffer(1)]],
        device const float   * x      [[buffer(2)]],
        device       float   * y      [[buffer(3)]],
        threadgroup  float   * wtile  [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    const uint q = tgpig.y * 32u + tiisg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 32u;
    device const block_q4_0_local * row_blocks = weight + row * nb;
    float acc = 0.0f;
    for (uint bidx = 0; bidx < nb; ++bidx) {
        device const block_q4_0_local & b = row_blocks[bidx];
        wtile[tiisg] = deq_q4_0(b, tiisg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q < args.n_query) {
            device const float * x_row = x + q * args.n_in + bidx * 32u;
            for (uint j = 0; j < 32u; ++j) {
                acc += x_row[j] * wtile[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (q < args.n_query) {
        y[row + q * args.n_out] = acc;
    }
}

kernel void kernel_mat_mat_q4_1_f32(
        constant mat_mat_args & args [[buffer(0)]],
        device const block_q4_1_local * weight [[buffer(1)]],
        device const float   * x      [[buffer(2)]],
        device       float   * y      [[buffer(3)]],
        threadgroup  float   * wtile  [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    const uint q = tgpig.y * 32u + tiisg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 32u;
    device const block_q4_1_local * row_blocks = weight + row * nb;
    float acc = 0.0f;
    for (uint bidx = 0; bidx < nb; ++bidx) {
        device const block_q4_1_local & b = row_blocks[bidx];
        wtile[tiisg] = deq_q4_1(b, tiisg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q < args.n_query) {
            device const float * x_row = x + q * args.n_in + bidx * 32u;
            for (uint j = 0; j < 32u; ++j) {
                acc += x_row[j] * wtile[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (q < args.n_query) {
        y[row + q * args.n_out] = acc;
    }
}

kernel void kernel_mat_mat_q3_K_f32(
        constant mat_mat_args & args [[buffer(0)]],
        device const block_q3_k_local * weight [[buffer(1)]],
        device const float   * x      [[buffer(2)]],
        device       float   * y      [[buffer(3)]],
        threadgroup  float   * wtile  [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    const uint q = tgpig.y * 32u + tiisg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 256u;
    device const block_q3_k_local * row_blocks = weight + row * nb;
    float acc = 0.0f;
    for (uint bidx = 0; bidx < nb; ++bidx) {
        device const block_q3_k_local & b = row_blocks[bidx];
        const uint base = bidx * 256u;
        for (uint k0 = 0; k0 < 256u; k0 += 32u) {
            wtile[tiisg] = deq_q3_k(b, k0 + tiisg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (q < args.n_query) {
                device const float * x_row = x + q * args.n_in + base + k0;
                for (uint j = 0; j < 32u; ++j) {
                    acc += x_row[j] * wtile[j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (q < args.n_query) {
        y[row + q * args.n_out] = acc;
    }
}

kernel void kernel_mat_mat_q2_K_f32(
        constant mat_mat_args & args [[buffer(0)]],
        device const block_q2_k_local * weight [[buffer(1)]],
        device const float   * x      [[buffer(2)]],
        device       float   * y      [[buffer(3)]],
        threadgroup  float   * wtile  [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    const uint q = tgpig.y * 32u + tiisg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 256u;
    device const block_q2_k_local * row_blocks = weight + row * nb;
    float acc = 0.0f;
    for (uint bidx = 0; bidx < nb; ++bidx) {
        device const block_q2_k_local & b = row_blocks[bidx];
        const uint base = bidx * 256u;
        for (uint k0 = 0; k0 < 256u; k0 += 32u) {
            wtile[tiisg] = deq_q2_k(b, k0 + tiisg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (q < args.n_query) {
                device const float * x_row = x + q * args.n_in + base + k0;
                for (uint j = 0; j < 32u; ++j) {
                    acc += x_row[j] * wtile[j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (q < args.n_query) {
        y[row + q * args.n_out] = acc;
    }
}

kernel void kernel_mat_mat_iq4_nl_f32(
        constant mat_mat_args & args [[buffer(0)]],
        device const block_iq4_nl_local * weight [[buffer(1)]],
        device const float   * x      [[buffer(2)]],
        device       float   * y      [[buffer(3)]],
        threadgroup  float   * wtile  [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    const uint q = tgpig.y * 32u + tiisg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 32u;
    device const block_iq4_nl_local * row_blocks = weight + row * nb;
    float acc = 0.0f;
    for (uint bidx = 0; bidx < nb; ++bidx) {
        device const block_iq4_nl_local & b = row_blocks[bidx];
        wtile[tiisg] = deq_iq4_nl(b, tiisg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q < args.n_query) {
            device const float * x_row = x + q * args.n_in + bidx * 32u;
            for (uint j = 0; j < 32u; ++j) {
                acc += x_row[j] * wtile[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (q < args.n_query) {
        y[row + q * args.n_out] = acc;
    }
}

kernel void kernel_mat_mat_iq4_xs_f32(
        constant mat_mat_args & args [[buffer(0)]],
        device const block_iq4_xs_local * weight [[buffer(1)]],
        device const float   * x      [[buffer(2)]],
        device       float   * y      [[buffer(3)]],
        threadgroup  float   * wtile  [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    const uint q = tgpig.y * 32u + tiisg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / 256u;
    device const block_iq4_xs_local * row_blocks = weight + row * nb;
    float acc = 0.0f;
    for (uint bidx = 0; bidx < nb; ++bidx) {
        device const block_iq4_xs_local & b = row_blocks[bidx];
        const uint base = bidx * 256u;
        for (uint k0 = 0; k0 < 256u; k0 += 32u) {
            wtile[tiisg] = deq_iq4_xs(b, k0 + tiisg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (q < args.n_query) {
                device const float * x_row = x + q * args.n_in + base + k0;
                for (uint j = 0; j < 32u; ++j) {
                    acc += x_row[j] * wtile[j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (q < args.n_query) {
        y[row + q * args.n_out] = acc;
    }
}

// Router logits F32 mat-mat specialized for small expert counts. One thread owns
// one query lane and computes 8 expert rows, reusing the same activation loads.
// Output layout matches kernel_mat_mat_f32_f32: [n_out, n_query] column-major.
kernel void kernel_mat_mat_f32_f32_router_e8p32(
        constant mat_mat_args & args [[buffer(0)]],
        device const float   * weight [[buffer(1)]], // [n_in, n_out]
        device const float   * x      [[buffer(2)]], // [n_query, n_in] row-major
        device       float   * y      [[buffer(3)]], // [n_out, n_query] col-major
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint row0 = tgpig.x * 8u;
    const uint q = tgpig.y * 32u + tid;
    if (q >= args.n_query) return;

    device const float * x_row = x + (ulong)q * args.n_in;
    const uint n_in_v4 = args.n_in / 4u;
    device const float4 * x4 = (device const float4 *)x_row;

    device const float4 * w0 = (device const float4 *)(weight + (ulong)(row0 + 0u) * args.n_in);
    device const float4 * w1 = (device const float4 *)(weight + (ulong)(row0 + 1u) * args.n_in);
    device const float4 * w2 = (device const float4 *)(weight + (ulong)(row0 + 2u) * args.n_in);
    device const float4 * w3 = (device const float4 *)(weight + (ulong)(row0 + 3u) * args.n_in);
    device const float4 * w4 = (device const float4 *)(weight + (ulong)(row0 + 4u) * args.n_in);
    device const float4 * w5 = (device const float4 *)(weight + (ulong)(row0 + 5u) * args.n_in);
    device const float4 * w6 = (device const float4 *)(weight + (ulong)(row0 + 6u) * args.n_in);
    device const float4 * w7 = (device const float4 *)(weight + (ulong)(row0 + 7u) * args.n_in);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    for (uint i = 0; i < n_in_v4; ++i) {
        const float4 xv = x4[i];
        acc0 += dot(w0[i], xv);
        acc1 += dot(w1[i], xv);
        acc2 += dot(w2[i], xv);
        acc3 += dot(w3[i], xv);
        acc4 += dot(w4[i], xv);
        acc5 += dot(w5[i], xv);
        acc6 += dot(w6[i], xv);
        acc7 += dot(w7[i], xv);
    }

    const uint tail_start = n_in_v4 * 4u;
    for (uint i = tail_start; i < args.n_in; ++i) {
        const float xv = x_row[i];
        acc0 += weight[(ulong)(row0 + 0u) * args.n_in + i] * xv;
        acc1 += weight[(ulong)(row0 + 1u) * args.n_in + i] * xv;
        acc2 += weight[(ulong)(row0 + 2u) * args.n_in + i] * xv;
        acc3 += weight[(ulong)(row0 + 3u) * args.n_in + i] * xv;
        acc4 += weight[(ulong)(row0 + 4u) * args.n_in + i] * xv;
        acc5 += weight[(ulong)(row0 + 5u) * args.n_in + i] * xv;
        acc6 += weight[(ulong)(row0 + 6u) * args.n_in + i] * xv;
        acc7 += weight[(ulong)(row0 + 7u) * args.n_in + i] * xv;
    }

    const ulong out_base = (ulong)q * args.n_out + row0;
    y[out_base + 0u] = acc0;
    y[out_base + 1u] = acc1;
    y[out_base + 2u] = acc2;
    y[out_base + 3u] = acc3;
    y[out_base + 4u] = acc4;
    y[out_base + 5u] = acc5;
    y[out_base + 6u] = acc6;
    y[out_base + 7u] = acc7;
}
