#include <metal_stdlib>
using namespace metal;

constant constexpr int DS4_PREP_QK8_0 = 32;
constant constexpr int DS4_PREP_Q8_0_BYTES = 34;
constant constexpr int DS4_PREP_QK_K = 256;
constant constexpr int DS4_PREP_Q6_K_BYTES = 210;

struct ds4_prepare_projection_pair_args {
    uint n_in;
    uint q_out;
    uint kv_out;
};

kernel void kernel_ds4_prepare_projection_pair_q8_q8_f32(
        constant ds4_prepare_projection_pair_args & args [[buffer(0)]],
        device const uchar * q_weight [[buffer(1)]],
        device const uchar * kv_weight [[buffer(2)]],
        device const float * x [[buffer(3)]],
        device float * q [[buffer(4)]],
        device float * kv [[buffer(5)]],
        threadgroup float * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr ushort NW = 32;
    constexpr ushort NQ = 8;
    constexpr ushort NR0 = 2;
    constexpr ushort NSG = 4;

    const uint n_out = tgpig.z == 0u ? args.q_out : args.kv_out;
    device const uchar * weight = tgpig.z == 0u ? q_weight : kv_weight;
    device float * y = tgpig.z == 0u ? q : kv;
    const uint nb = args.n_in / DS4_PREP_QK8_0;
    const uint first_row = tgpig.x * NR0;
    if (first_row >= n_out) return;

    const ushort ix = tiisg / (NW / NQ);
    const ushort il = tiisg % (NW / NQ);
    const uint ib0 = sgitg * NQ + ix;
    const ulong row_stride_bytes = (ulong)nb * DS4_PREP_Q8_0_BYTES;
    device const uchar * row0 = weight + (ulong)first_row * row_stride_bytes;
    device const float * xb = x + (ulong)ib0 * DS4_PREP_QK8_0 + (ulong)il * NQ;

    float sumf[NR0] = {0.0f, 0.0f};
    float xv[NQ];
    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        for (ushort i = 0; i < NQ; ++i) {
            xv[i] = xb[i];
        }
        for (ushort row = 0; row < NR0; ++row) {
            if (first_row + row >= n_out) break;
            device const uchar * blk = row0
                + (ulong)row * row_stride_bytes
                + (ulong)ib * DS4_PREP_Q8_0_BYTES;
            device const half * dh = (device const half *)blk;
            device const int8_t * qs =
                (device const int8_t *)(blk + 2) + il * NQ;
            float sumq = 0.0f;
            for (ushort i = 0; i < NQ; ++i) {
                sumq += (float)qs[i] * xv[i];
            }
            sumf[row] += sumq * (float)dh[0];
        }
        xb += (ulong)NSG * NQ * DS4_PREP_QK8_0;
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
    for (ushort row = 0; row < NR0 && first_row + row < n_out; ++row) {
        threadgroup float * row_shmem = shmem + NW * row;
        const float total = simd_sum(row_shmem[tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            y[first_row + row] = total;
        }
    }
}

kernel void kernel_ds4_prepare_projection_pair_q6_q8_f32(
        constant ds4_prepare_projection_pair_args & args [[buffer(0)]],
        device const uchar * q_weight [[buffer(1)]],
        device const uchar * kv_weight [[buffer(2)]],
        device const float * x [[buffer(3)]],
        device float * q [[buffer(4)]],
        device float * kv [[buffer(5)]],
        threadgroup float * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    if (tgpig.z == 0u) {
        if (sgitg >= 2u) return;
        constexpr uchar kmask1 = 0x03;
        constexpr uchar kmask2 = 0x0C;
        constexpr uchar kmask3 = 0x30;
        constexpr uchar kmask4 = 0xC0;
        constexpr ushort NR0 = 2;

        const uint nb = args.n_in / DS4_PREP_QK_K;
        const uint first_row = (tgpig.x * 2u + sgitg) * NR0;
        if (first_row >= args.q_out) return;
        const ulong row_stride_bytes = (ulong)nb * DS4_PREP_Q6_K_BYTES;
        device const uchar * row0 = q_weight + (ulong)first_row * row_stride_bytes;
        const ushort tid = tiisg / 2;
        const ushort ix = tiisg % 2;
        const ushort ip = tid / 8;
        const ushort il = tid % 8;
        const ushort l0 = 4u * il;
        const ushort is = 8u * ip + l0 / 16u;
        const ushort y_offset = 128u * ip + l0;
        const ushort q_offset_l = 64u * ip + l0;
        const ushort q_offset_h = 32u * ip + l0;
        float sumf[NR0] = {0.0f, 0.0f};
        float yl[16];

        for (uint i = ix; i < nb; i += 2) {
            device const float * y_blk = x + (ulong)i * DS4_PREP_QK_K + y_offset;
            for (short l = 0; l < 4; ++l) {
                yl[4*l + 0] = y_blk[l + 0];
                yl[4*l + 1] = y_blk[l + 32];
                yl[4*l + 2] = y_blk[l + 64];
                yl[4*l + 3] = y_blk[l + 96];
            }
            for (short row = 0; row < NR0; ++row) {
                if (first_row + row >= args.q_out) break;
                device const uchar * blk = row0 + row * row_stride_bytes
                    + (ulong)i * DS4_PREP_Q6_K_BYTES;
                device const uchar * q1 = blk + q_offset_l;
                device const uchar * q2 = q1 + 32;
                device const uchar * qh = blk + 128 + q_offset_h;
                device const int8_t * sc =
                    (device const int8_t *)(blk + 128 + 64) + is;
                device const half * dh =
                    (device const half *)(blk + 128 + 64 + 16);
                float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
                for (short l = 0; l < 4; ++l) {
                    sums[0] += yl[4*l + 0]
                        * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                    sums[1] += yl[4*l + 1]
                        * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                    sums[2] += yl[4*l + 2]
                        * ((int8_t)((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                    sums[3] += yl[4*l + 3]
                        * ((int8_t)((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
                }
                sumf[row] += (float)dh[0] * (
                      sums[0] * (float)sc[0]
                    + sums[1] * (float)sc[2]
                    + sums[2] * (float)sc[4]
                    + sums[3] * (float)sc[6]
                );
            }
        }
        for (short row = 0; row < NR0; ++row) {
            const float total = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.q_out) {
                q[first_row + row] = total;
            }
        }
        return;
    }

    constexpr ushort NW = 32;
    constexpr ushort NQ = 8;
    constexpr ushort NR0 = 2;
    constexpr ushort NSG = 4;
    const uint nb = args.n_in / DS4_PREP_QK8_0;
    const uint first_row = tgpig.x * NR0;
    if (first_row >= args.kv_out) return;
    const ushort ix = tiisg / (NW / NQ);
    const ushort il = tiisg % (NW / NQ);
    const uint ib0 = sgitg * NQ + ix;
    const ulong row_stride_bytes = (ulong)nb * DS4_PREP_Q8_0_BYTES;
    device const uchar * row0 = kv_weight + (ulong)first_row * row_stride_bytes;
    device const float * xb = x + (ulong)ib0 * DS4_PREP_QK8_0 + (ulong)il * NQ;
    float sumf[NR0] = {0.0f, 0.0f};
    float xv[NQ];
    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        for (ushort i = 0; i < NQ; ++i) {
            xv[i] = xb[i];
        }
        for (ushort row = 0; row < NR0; ++row) {
            if (first_row + row >= args.kv_out) break;
            device const uchar * blk = row0
                + (ulong)row * row_stride_bytes
                + (ulong)ib * DS4_PREP_Q8_0_BYTES;
            device const half * dh = (device const half *)blk;
            device const int8_t * qs =
                (device const int8_t *)(blk + 2) + il * NQ;
            float sumq = 0.0f;
            for (ushort i = 0; i < NQ; ++i) {
                sumq += (float)qs[i] * xv[i];
            }
            sumf[row] += sumq * (float)dh[0];
        }
        xb += (ulong)NSG * NQ * DS4_PREP_QK8_0;
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
    for (ushort row = 0; row < NR0 && first_row + row < args.kv_out; ++row) {
        threadgroup float * row_shmem = shmem + NW * row;
        const float total = simd_sum(row_shmem[tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            kv[first_row + row] = total;
        }
    }
}
