// Q5_K dense mat-mat on the Metal 4 tensor units.
// y[q, o] = sum_k x[q, k] * w[o, k]; w is Q5_K, x is F32.
// Tile: 64 queries x 64 outputs x 32 K, 128 threads (4 simdgroups).
// The activation and weight tiles are staged as half in 8 KiB of threadgroup
// memory; the 64x64 F32 accumulator lives in the cooperative tensor. Each
// thread dequantizes one 16-element Q5_K subtile, so two subtiles cover the
// 32-wide K step.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define Q5K_TENSOR_ROWS 64
#define Q5K_TENSOR_COLS 64
#define Q5K_TENSOR_K 32
#define Q5K_TENSOR_THREADS 128

struct q5k_tensor_args {
    uint rows;
    uint cols;
    uint k;
    uint nb01;
    uint stride_b;
};

// Same decode as kernels/mat_mat_q5_k.metal; `il` selects one 16-element
// subtile of the 256-element superblock.
inline void dequantize_q5_K_half_tensor(device const uchar *blk_bytes,
                                        short il,
                                        thread half4x4 &reg) {
    const half d_h = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar *scales = blk_bytes + 4;
    device const uchar *qh = blk_bytes + 4 + 12;
    device const uchar *qs = blk_bytes + 4 + 12 + 32;

    const short is = (il / 4) * 2;
    const short k01 = (il / 2) & 1;
    uchar sc_u, m_u;
    if (is < 4) {
        sc_u = scales[is + k01] & 63;
        m_u = scales[is + k01 + 4] & 63;
    } else {
        sc_u = (scales[is + k01 + 4] & 0x0F) | ((scales[is + k01 - 4] >> 6) << 4);
        m_u = (scales[is + k01 + 4] >> 4) | ((scales[is + k01] >> 6) << 4);
    }

    qs = qs + 32 * (il / 4) + 16 * (il & 1);
    qh = qh + 16 * (il & 1);
    const uchar ul = 1u << (il / 2);
    const short il_inner = il & 3;

    const float d = il_inner < 2 ? (float)d_h : (float)d_h / 16.0f;
    const float dmin = (float)dmin_h;
    const float dl = d * (float)sc_u;
    const float ml = dmin * (float)m_u;
    const ushort mask = il_inner < 2 ? 0x0F : 0xF0;
    const float qh_val = il_inner < 2 ? 16.0f : 256.0f;

    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const float q_low = (float)(qs[i] & mask);
        const float q_high = (qh[i] & ul) ? qh_val : 0.0f;
        reg[i / 4][i % 4] = (half)(dl * (q_low + q_high) - ml);
    }
}

kernel void kernel_mat_mat_q5_k_tensor(
    constant q5k_tensor_args &args [[buffer(0)]],
    device const uchar *srcA [[buffer(1)]],
    device const float *srcB [[buffer(2)]],
    device float *dst [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half xblk[Q5K_TENSOR_ROWS * Q5K_TENSOR_K];
    threadgroup half wblk[Q5K_TENSOR_COLS * Q5K_TENSOR_K];
    const uint q0 = tgpig.x * Q5K_TENSOR_ROWS;
    const uint o0 = tgpig.y * Q5K_TENSOR_COLS;

    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        Q5K_TENSOR_ROWS, Q5K_TENSOR_COLS, Q5K_TENSOR_K, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    using x_t = tensor<threadgroup half, extents<int, Q5K_TENSOR_K, Q5K_TENSOR_ROWS>, tensor_inline>;
    using w_t = tensor<threadgroup half, extents<int, Q5K_TENSOR_K, Q5K_TENSOR_COLS>, tensor_inline>;
    x_t xt(xblk, extents<int, Q5K_TENSOR_K, Q5K_TENSOR_ROWS>());
    w_t wt(wblk, extents<int, Q5K_TENSOR_K, Q5K_TENSOR_COLS>());
    auto acc = op.get_destination_cooperative_tensor<x_t, w_t, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        acc[i] = 0.0f;
    }

    for (uint k0 = 0; k0 < args.k; k0 += Q5K_TENSOR_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < Q5K_TENSOR_ROWS * Q5K_TENSOR_K / 4;
             e += Q5K_TENSOR_THREADS) {
            const uint q = e / (Q5K_TENSOR_K / 4), c4 = e % (Q5K_TENSOR_K / 4);
            float4 v = float4(0.0f);
            if (q0 + q < args.rows) {
                v = ((device const float4 *)(srcB + (ulong)(q0 + q) * args.stride_b + k0))[c4];
            }
            ((threadgroup half4 *)xblk)[e] = half4(v);
        }
        const uint sb = k0 / 256;
        const uint il0 = (k0 % 256) / 16;
        for (uint e = tid; e < Q5K_TENSOR_COLS * 2; e += Q5K_TENSOR_THREADS) {
            const uint o = e / 2, j = e % 2;
            if (o0 + o >= args.cols) continue;
            half4x4 reg;
            const device uchar *blk =
                srcA + (ulong)(o0 + o) * args.nb01 + (ulong)sb * 176;
            dequantize_q5_K_half_tensor(blk, short(il0 + j), reg);
            FOR_UNROLL (int i = 0; i < 16; ++i) {
                wblk[o * Q5K_TENSOR_K + j * 16 + i] = reg[i / 4][i % 4];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(xt, wt, acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *y = dst + (ulong)q0 * args.cols + o0;
    tensor<device float, extents<int, Q5K_TENSOR_COLS, Q5K_TENSOR_ROWS>, tensor_inline> ot(
        y, extents<int, Q5K_TENSOR_COLS, Q5K_TENSOR_ROWS>(),
        array<int, 2>{1, int(args.cols)});
    acc.store(ot);
}
