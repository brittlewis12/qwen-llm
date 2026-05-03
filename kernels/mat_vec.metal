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
