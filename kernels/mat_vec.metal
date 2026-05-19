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
