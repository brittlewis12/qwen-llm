// Q8_0 mat-vec.
//
// Block layout (block_q8_0, 34 bytes / 32 elements):
//   half  d                  (super-block scale)
//   int8  qs[32]             (signed 8-bit quants)
//
// Element value:
//   x[k] = d * (float)qs[k]
//
// Q8_0 is the simplest K-quant variant: one scale per 32-element block,
// no min/dmin, no scale packing, no high-bit path. Dramatically simpler
// than Q4_K/Q5_K/Q6_K.
//
// Layout pattern lifted from llama.cpp's `kernel_mul_mv_q8_0_f32_impl`
// at ggml/src/ggml-metal/ggml-metal.metal:3573, simplified to NR0=1
// NSG=2 (one row per simdgroup, two simdgroups per threadgroup) to
// match our existing Q5_K mat-vec pattern. Each lane processes NQ=8
// quants at a time across the K dim; one full K-pass per row.

#include <metal_stdlib>
using namespace metal;

constant constexpr int  QK8_0      = 32;
constant constexpr int  Q8_0_BYTES = 34;  // sizeof(half) + 32 int8

struct mat_vec_q8_0_args {
    uint n_in;
    uint n_out;
};

#define NR0_Q80 1
#define NSG_Q80 2
#define NQ_Q80 8       // quants per thread per super-block iter

kernel void kernel_mat_vec_q8_0_f32(
        constant mat_vec_q8_0_args & args   [[buffer(0)]],
        device const uchar         * weight [[buffer(1)]],
        device const float         * x      [[buffer(2)]],
        device       float         * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr ushort NW = 32;
    constexpr ushort NQ = NQ_Q80;
    // 32 lanes / 8 quants-per-lane = 4 lane-groups per K-step.
    // Each lane-group covers NQ=8 quants of the row's super-block.
    const ushort lane_grp_count = NW / NQ;          // 4
    const ushort ig             = tiisg / NQ;       // lane-group index in [0, 4)
    const ushort il             = tiisg % NQ;       // intra-group lane in [0, 8)

    // Number of super-blocks per row (Q8_0 has QK8_0=32 elements/block).
    const uint nb = args.n_in / QK8_0;
    const uint first_row = (tgpig * NSG_Q80 + sgitg) * NR0_Q80;
    if (first_row >= args.n_out) return;

    // Pointer to the start of this row's first super-block.
    const ulong row_stride_bytes = (ulong)nb * Q8_0_BYTES;
    device const uchar * row_blk = weight + (ulong)first_row * row_stride_bytes;

    float sumf = 0.0f;

    // Each lane-group iterates over super-blocks in stride lane_grp_count;
    // each iteration covers NQ=8 quants out of the super-block's 32.
    // Lane-group `ig` reads quants [ig*NQ .. ig*NQ + NQ).
    for (uint ib = 0; ib < nb; ++ib) {
        device const uchar * blk = row_blk + (ulong)ib * Q8_0_BYTES;
        device const half  * dh  = (device const half *)blk;
        device const int8_t* qs  = (device const int8_t *)(blk + 2);

        // Each lane reads one quant out of the lane-group's 8.
        const short qi = ig * NQ + il;
        const float xv = x[ib * QK8_0 + qi];
        const float qv = (float)qs[qi];
        sumf += xv * qv * (float)dh[0];
    }

    const float tot = simd_sum(sumf);
    if (tiisg == 0 && first_row < args.n_out) {
        y[first_row] = tot;
    }
}
