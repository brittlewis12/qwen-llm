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
// The default host path uses the `_lcpp` kernel below: NR0=2 / NSG=4,
// four simdgroups cooperate over K stripes, and each lane handles 8 quants
// per visited block. The original one-row-per-simdgroup kernel is kept as
// a rollback path via `QWEN_MATVEC_Q8_0_LCPP=0`.

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

#define NR0_Q80_LCPP 2
#define NSG_Q80_LCPP 4

kernel void kernel_mat_vec_q8_0_f32(
        constant mat_vec_q8_0_args & args   [[buffer(0)]],
        device const uchar         * weight [[buffer(1)]],
        device const float         * x      [[buffer(2)]],
        device       float         * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr ushort NQ = NQ_Q80;
    // 32 lanes / 8 quants-per-lane = 4 lane-groups per K-step.
    // Each lane-group covers NQ=8 quants of the row's super-block.
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

    // Each lane-group covers NQ=8 quants out of each super-block's 32.
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

kernel void kernel_mat_vec_q8_0_f32_lcpp(
        constant mat_vec_q8_0_args & args   [[buffer(0)]],
        device const uchar         * weight [[buffer(1)]],
        device const float         * x      [[buffer(2)]],
        device       float         * y      [[buffer(3)]],
        threadgroup  float         * shmem  [[threadgroup(0)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr ushort NW = 32;
    constexpr ushort NQ = NQ_Q80;
    constexpr ushort NR0 = NR0_Q80_LCPP;
    constexpr ushort NSG = NSG_Q80_LCPP;

    const uint nb = args.n_in / QK8_0;
    const uint first_row = tgpig * NR0;
    if (first_row >= args.n_out) return;

    const ushort ix = tiisg / (NW / NQ);
    const ushort il = tiisg % (NW / NQ);
    const uint ib0 = sgitg * NQ + ix;

    const ulong row_stride_bytes = (ulong)nb * Q8_0_BYTES;
    device const uchar * row0 = weight + (ulong)first_row * row_stride_bytes;
    device const float * xb = x + (ulong)ib0 * QK8_0 + (ulong)il * NQ;

    float sumf[NR0] = {0.0f, 0.0f};
    float xv[NQ];

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        for (ushort i = 0; i < NQ; ++i) {
            xv[i] = xb[i];
        }

        for (ushort row = 0; row < NR0; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uchar * blk = row0
                + (ulong)row * row_stride_bytes
                + (ulong)ib * Q8_0_BYTES;
            device const half   * dh = (device const half *)blk;
            device const int8_t * qs = (device const int8_t *)(blk + 2) + il * NQ;

            float sumq = 0.0f;
            for (ushort i = 0; i < NQ; ++i) {
                sumq += (float)qs[i] * xv[i];
            }
            sumf[row] += sumq * (float)dh[0];
        }

        xb += (ulong)NSG * NQ * QK8_0;
    }

    for (ushort row = 0; row < NR0; ++row) {
        threadgroup float * row_shmem = shmem + NW * row;
        if (sgitg == 0) {
            row_shmem[tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row = 0; row < NR0; ++row) {
        threadgroup float * row_shmem = shmem + NW * row;
        if (tiisg == 0) {
            row_shmem[sgitg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row = 0; row < NR0 && first_row + row < args.n_out; ++row) {
        threadgroup float * row_shmem = shmem + NW * row;
        const float total = simd_sum(row_shmem[tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            y[first_row + row] = total;
        }
    }
}
