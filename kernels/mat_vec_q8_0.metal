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

struct ds4_shared_swiglu_q8_0_args {
    uint n_in;
    uint n_out;
    float clamp;
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

// Group-axis variant of the `_lcpp` kernel above. Grid depth indexes
// `n_groups` consecutive weight blocks of `n_out` rows, consecutive
// `n_in`-element input slices, and consecutive `n_out`-element output
// slices. The per-row traversal, accumulation order, and reductions are
// byte-for-byte the singleton `_lcpp` body, so each group's output is
// bitwise identical to a separate singleton dispatch over its slice.
kernel void kernel_mat_vec_q8_0_f32_lcpp_grouped(
        constant mat_vec_q8_0_args & args   [[buffer(0)]],
        device const uchar         * weight [[buffer(1)]],
        device const float         * x      [[buffer(2)]],
        device       float         * y      [[buffer(3)]],
        threadgroup  float         * shmem  [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr ushort NW = 32;
    constexpr ushort NQ = NQ_Q80;
    constexpr ushort NR0 = NR0_Q80_LCPP;
    constexpr ushort NSG = NSG_Q80_LCPP;

    const uint nb = args.n_in / QK8_0;
    const uint first_row = tgpig.x * NR0;
    if (first_row >= args.n_out) return;

    const ushort ix = tiisg / (NW / NQ);
    const ushort il = tiisg % (NW / NQ);
    const uint ib0 = sgitg * NQ + ix;

    const ulong row_stride_bytes = (ulong)nb * Q8_0_BYTES;
    device const uchar * gweight = weight
        + (ulong)tgpig.z * (ulong)args.n_out * row_stride_bytes;
    device const float * gx = x + (ulong)tgpig.z * (ulong)args.n_in;
    device       float * gy = y + (ulong)tgpig.z * (ulong)args.n_out;

    device const uchar * row0 = gweight + (ulong)first_row * row_stride_bytes;
    device const float * xb = gx + (ulong)ib0 * QK8_0 + (ulong)il * NQ;

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
            gy[first_row + row] = total;
        }
    }
}

kernel void kernel_mat_vec_q8_0_f32_lcpp_batch(
        constant mat_vec_q8_0_args & args   [[buffer(0)]],
        device const uchar         * weight [[buffer(1)]],
        device const float         * x      [[buffer(2)]],
        device       float         * y      [[buffer(3)]],
        threadgroup  float         * shmem  [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr ushort NW = 32;
    constexpr ushort NQ = NQ_Q80;
    constexpr ushort NR0 = NR0_Q80_LCPP;
    constexpr ushort NSG = NSG_Q80_LCPP;

    const uint nb = args.n_in / QK8_0;
    const uint first_row = tgpig.x * NR0;
    if (first_row >= args.n_out) return;
    x += (ulong)tgpig.y * args.n_in;
    y += (ulong)tgpig.y * args.n_out;

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

kernel void kernel_shared_swiglu_q8_0_f32_lcpp(
        constant mat_vec_q8_0_args & args        [[buffer(0)]],
        device const uchar         * gate_weight [[buffer(1)]],
        device const uchar         * up_weight   [[buffer(2)]],
        device const float         * x           [[buffer(3)]],
        device       float         * y           [[buffer(4)]],
        threadgroup  float         * shmem       [[threadgroup(0)]],
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
    device const uchar * gate_row0 = gate_weight + (ulong)first_row * row_stride_bytes;
    device const uchar * up_row0   = up_weight   + (ulong)first_row * row_stride_bytes;
    device const float * xb = x + (ulong)ib0 * QK8_0 + (ulong)il * NQ;

    float sumg[NR0] = {0.0f, 0.0f};
    float sumu[NR0] = {0.0f, 0.0f};
    float xv[NQ];

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        for (ushort i = 0; i < NQ; ++i) {
            xv[i] = xb[i];
        }

        for (ushort row = 0; row < NR0; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uchar * gate_blk = gate_row0
                + (ulong)row * row_stride_bytes
                + (ulong)ib * Q8_0_BYTES;
            device const uchar * up_blk = up_row0
                + (ulong)row * row_stride_bytes
                + (ulong)ib * Q8_0_BYTES;
            device const half   * gate_dh = (device const half *)gate_blk;
            device const half   * up_dh   = (device const half *)up_blk;
            device const int8_t * gate_qs = (device const int8_t *)(gate_blk + 2) + il * NQ;
            device const int8_t * up_qs   = (device const int8_t *)(up_blk + 2) + il * NQ;

            float sum_gate = 0.0f;
            float sum_up = 0.0f;
            for (ushort i = 0; i < NQ; ++i) {
                sum_gate += (float)gate_qs[i] * xv[i];
                sum_up += (float)up_qs[i] * xv[i];
            }
            sumg[row] += sum_gate * (float)gate_dh[0];
            sumu[row] += sum_up * (float)up_dh[0];
        }

        xb += (ulong)NSG * NQ * QK8_0;
    }

    for (ushort row = 0; row < NR0; ++row) {
        threadgroup float * gate_shmem = shmem + NW * row;
        threadgroup float * up_shmem = shmem + NW * (NR0 + row);
        if (sgitg == 0) {
            gate_shmem[tiisg] = 0.0f;
            up_shmem[tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row = 0; row < NR0; ++row) {
        threadgroup float * gate_shmem = shmem + NW * row;
        threadgroup float * up_shmem = shmem + NW * (NR0 + row);
        if (tiisg == 0) {
            gate_shmem[sgitg] = sumg[row];
            up_shmem[sgitg] = sumu[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row = 0; row < NR0 && first_row + row < args.n_out; ++row) {
        threadgroup float * gate_shmem = shmem + NW * row;
        threadgroup float * up_shmem = shmem + NW * (NR0 + row);
        const float gate_total = simd_sum(gate_shmem[tiisg]);
        const float up_total = simd_sum(up_shmem[tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            y[first_row + row] = gate_total / (1.0f + exp(-gate_total)) * up_total;
        }
    }
}

kernel void kernel_ds4_shared_swiglu_q8_0_f32_lcpp(
        constant ds4_shared_swiglu_q8_0_args & args [[buffer(0)]],
        device const uchar         * gate_weight [[buffer(1)]],
        device const uchar         * up_weight   [[buffer(2)]],
        device const float         * x           [[buffer(3)]],
        device       float         * y           [[buffer(4)]],
        threadgroup  float         * shmem       [[threadgroup(0)]],
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
    device const uchar * gate_row0 = gate_weight + (ulong)first_row * row_stride_bytes;
    device const uchar * up_row0   = up_weight   + (ulong)first_row * row_stride_bytes;
    device const float * xb = x + (ulong)ib0 * QK8_0 + (ulong)il * NQ;

    float sumg[NR0] = {0.0f, 0.0f};
    float sumu[NR0] = {0.0f, 0.0f};
    float xv[NQ];

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        for (ushort i = 0; i < NQ; ++i) {
            xv[i] = xb[i];
        }

        for (ushort row = 0; row < NR0; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uchar * gate_blk = gate_row0
                + (ulong)row * row_stride_bytes
                + (ulong)ib * Q8_0_BYTES;
            device const uchar * up_blk = up_row0
                + (ulong)row * row_stride_bytes
                + (ulong)ib * Q8_0_BYTES;
            device const half   * gate_dh = (device const half *)gate_blk;
            device const half   * up_dh   = (device const half *)up_blk;
            device const int8_t * gate_qs = (device const int8_t *)(gate_blk + 2) + il * NQ;
            device const int8_t * up_qs   = (device const int8_t *)(up_blk + 2) + il * NQ;

            float sum_gate = 0.0f;
            float sum_up = 0.0f;
            for (ushort i = 0; i < NQ; ++i) {
                sum_gate += (float)gate_qs[i] * xv[i];
                sum_up += (float)up_qs[i] * xv[i];
            }
            sumg[row] += sum_gate * (float)gate_dh[0];
            sumu[row] += sum_up * (float)up_dh[0];
        }

        xb += (ulong)NSG * NQ * QK8_0;
    }

    for (ushort row = 0; row < NR0; ++row) {
        threadgroup float * gate_shmem = shmem + NW * row;
        threadgroup float * up_shmem = shmem + NW * (NR0 + row);
        if (sgitg == 0) {
            gate_shmem[tiisg] = 0.0f;
            up_shmem[tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row = 0; row < NR0; ++row) {
        threadgroup float * gate_shmem = shmem + NW * row;
        threadgroup float * up_shmem = shmem + NW * (NR0 + row);
        if (tiisg == 0) {
            gate_shmem[sgitg] = sumg[row];
            up_shmem[sgitg] = sumu[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row = 0; row < NR0 && first_row + row < args.n_out; ++row) {
        threadgroup float * gate_shmem = shmem + NW * row;
        threadgroup float * up_shmem = shmem + NW * (NR0 + row);
        const float gate_total = simd_sum(gate_shmem[tiisg]);
        const float up_total = simd_sum(up_shmem[tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            const float clamped_gate = min(gate_total, args.clamp);
            const float clamped_up = clamp(up_total, -args.clamp, args.clamp);
            y[first_row + row] = clamped_gate / (1.0f + exp(-clamped_gate)) * clamped_up;
        }
    }
}
