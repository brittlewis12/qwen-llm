// Q6_K multi-column mat-vec — companion to mat_vec_q4_k_nc.metal (see the
// design/motivation comment there; H5.6 M2-nc). Q6_K covers the two largest
// verify-path tensors on 27B-Q4_K_M: ffn_down [17408 -> 5120] and
// gdn_qkv [5120 -> 10240].
//
// Geometry: one threadgroup = 4 simdgroups (128 threads) covering
// 4 output rows x NC columns as 2 row-pairs x 2 column-halves. Weight
// coverage per TG matches mv1 (grid = n_out/4). Quant block reads sit
// inside the per-column loop (column-invariant; L1-hot on re-reads, and
// the two column-half simdgroups of a row-pair hit the same 210-byte
// blocks through L1) — see the Q4_K companion header for why explicit
// register staging measured slower.
//
// Body per (simdgroup, column) keeps the exact expression structure of
// `kernel_mat_vec_q6_K_f32` (inline mask/shift decode, NR0=2 rows sharing
// the yl registers) so per-column outputs are bit-exact vs mv1 (E0),
// ASSERTED by `multicol_gemv_micro_27b` in tests/dflash_correctness.rs.
//
// Layouts (same as the Q4_K nc kernel):
//   weight: raw block_q6_K bytes, [n_out, n_in]
//   x:      F32 [NC, n_in]  row-major (column c = x + c*n_in)
//   y:      F32 [NC, n_out] row-major (y[c*n_out + row])

#include <metal_stdlib>
using namespace metal;

constant constexpr int QK_K_Q6NC      = 256;
constant constexpr int Q6K_BYTES_Q6NC = 210;

struct mat_vec_q6k_nc_args {
    uint n_in;
    uint n_out;
};

#define NR0_Q6K_NC 2

template <short NC>
inline void mat_vec_q6_K_nc_impl(
        constant mat_vec_q6k_nc_args & args,
        device const uchar           * weight,
        device const float           * x,
        device       float           * y,
        uint tgpig, ushort sgitg, ushort tiisg) {
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    constexpr short NC_SG = NC / 2;

    const uint nb = args.n_in / QK_K_Q6NC;
    const uint first_row = (tgpig * 2 + (sgitg & 1)) * NR0_Q6K_NC;
    const short col_base = (short)(sgitg >> 1) * NC_SG;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q6K_BYTES_Q6NC;
    device const uchar * row0 = weight + first_row * row_stride_bytes;

    const ushort tid = tiisg / 2;
    const ushort ix  = tiisg % 2;
    const ushort ip  = tid / 8;
    const ushort il  = tid % 8;
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;

    float sumf[NR0_Q6K_NC][NC_SG];
    for (short r = 0; r < NR0_Q6K_NC; ++r) {
        for (short c = 0; c < NC_SG; ++c) {
            sumf[r][c] = 0.f;
        }
    }
    float yl[16];

    for (uint i = ix; i < nb; i += 2) {
        for (short col = 0; col < NC_SG; ++col) {
            device const float * y_blk = x + (ulong)(col_base + col) * args.n_in
                                           + (ulong)i * QK_K_Q6NC + y_offset;

            for (short l = 0; l < 4; ++l) {
                yl[4*l + 0] = y_blk[l +  0];
                yl[4*l + 1] = y_blk[l + 32];
                yl[4*l + 2] = y_blk[l + 64];
                yl[4*l + 3] = y_blk[l + 96];
            }

            for (short row = 0; row < NR0_Q6K_NC; ++row) {
                if (first_row + row >= args.n_out) break;

                device const uchar * blk = row0 + row * row_stride_bytes
                                          + (ulong)i * Q6K_BYTES_Q6NC;
                device const uchar * q1 = blk + q_offset_l;
                device const uchar * q2 = q1 + 32;
                device const uchar * qh = blk + 128 + q_offset_h;
                device const int8_t * sc = (device const int8_t *)(blk + 128 + 64) + is;
                device const half   * dh = (device const half *)(blk + 128 + 64 + 16);

                float4 sums = {0.f, 0.f, 0.f, 0.f};
                for (short l = 0; l < 4; ++l) {
                    sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                    sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                    sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                    sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
                }
                sumf[row][col] += (float)dh[0] * (
                      sums[0] * (float)sc[0]
                    + sums[1] * (float)sc[2]
                    + sums[2] * (float)sc[4]
                    + sums[3] * (float)sc[6]
                );
            }
        }
    }

    for (short row = 0; row < NR0_Q6K_NC; ++row) {
        for (short col = 0; col < NC_SG; ++col) {
            float total = simd_sum(sumf[row][col]);
            if (tiisg == 0 && first_row + row < args.n_out) {
                y[(ulong)(col_base + col) * args.n_out + first_row + row] = total;
            }
        }
    }
}

#define MAT_VEC_Q6K_NC_KERNEL(NCOLS, NAME)                                    \
kernel void NAME(                                                             \
        constant mat_vec_q6k_nc_args & args   [[buffer(0)]],                  \
        device const uchar           * weight [[buffer(1)]],                  \
        device const float           * x      [[buffer(2)]],                  \
        device       float           * y      [[buffer(3)]],                  \
        uint   tgpig [[threadgroup_position_in_grid]],                        \
        ushort sgitg [[simdgroup_index_in_threadgroup]],                      \
        ushort tiisg [[thread_index_in_simdgroup]]) {                         \
    mat_vec_q6_K_nc_impl<NCOLS>(args, weight, x, y, tgpig, sgitg, tiisg);     \
}

MAT_VEC_Q6K_NC_KERNEL(2, kernel_mat_vec_q6_K_nc2_f32)
MAT_VEC_Q6K_NC_KERNEL(4, kernel_mat_vec_q6_K_nc4_f32)
MAT_VEC_Q6K_NC_KERNEL(8, kernel_mat_vec_q6_K_nc8_f32)
