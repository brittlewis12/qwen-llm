// Q4_K multi-column mat-vec ("mv-ext"-shaped skinny GEMM experiment).
//
// Motivation (H5.6 M1a follow-up): the 64x32 simdgroup-matrix mat-mat tile
// runs the packed-verify projection shapes at 56-128 GB/s with N2..N32 all
// costing the SAME wall time (per-tile machinery floor + under-occupancy:
// n_out/64 threadgroups cannot fill 40 cores at skinny shapes), while the
// N=1 mat-vec streams the same weights at 283-436 GB/s from n_out/4
// threadgroups. This kernel keeps the mat-vec's dispatch geometry and lane
// mapping and runs the mv1 body per activation column: quant block reads
// are column-invariant (L1-hot on the re-reads; the compiler may hoist
// them) and each column's y-slice streams against the same blocks.
//
// Two register-staging variants were built and MEASURED SLOWER (v0.498):
// explicit uint16 quant staging (NSG=2) scaled the same but no better, and
// a convert-once/NR0=1 float staging variant register-spilled to 3.5-6x
// slower AND broke bit-exactness (fast-math reassociation under a changed
// expression graph). Keep the body expression-identical to mv1.
//
// Exactness: for a given column c, the accumulation order (super-block
// traversal, acc1/acc2 lane math, scale/min epilogue, simd_sum reduce) is
// IDENTICAL to `kernel_mat_vec_q4_K_f32` — outputs are bit-exact vs
// running mv1 per column (E0 tier), ASSERTED by
// `multicol_gemv_micro_27b` in tests/dflash_correctness.rs.
//
// Layouts (matches encode_mat_mat_dispatch conventions):
//   weight: raw block_q4_K bytes, [n_out, n_in]
//   x:      F32 [NC, n_in]  row-major (column c = x + c*n_in)
//   y:      F32 [NC, n_out] row-major (y[c*n_out + row])

#include <metal_stdlib>
using namespace metal;

constant constexpr int QK_K_NC      = 256;
constant constexpr int Q4K_BYTES_NC = 144;

struct mat_vec_q4k_nc_args {
    uint n_in;
    uint n_out;
};

// Geometry: one threadgroup covers 4 output rows x NC columns with FOUR
// simdgroups — 2 row-pairs x 2 column-halves. Same 4-rows-per-TG weight
// coverage as mv1 (n_out/4 threadgroups), but twice the ALU per weight
// byte: the attempt-1 measurement showed the per-column inner loop is
// ALU/issue-bound while mv1 runs ~70% under the weight-stream roofline,
// so the columns are split across the extra simdgroups. The two
// column-half simdgroups of a row-pair re-read the same 144-byte quant
// blocks through L1 (DRAM traffic unchanged).
//
// Body per (simdgroup, column): IDENTICAL expression structure to mv1
// (inline mask-multiply, NR0=2 rows sharing the y registers, interleaved
// load+sumy loop) — attempt-1 measured this form bit-exact vs mv1 under
// the production compile flags.
#define NR0_Q4K_NC 2

// RP = row-pairs per threadgroup (v0.500 sweep axis B1: TG shape).
// RP=2 reproduces the v0.499 kernels exactly ((sgitg % 2) == (sgitg & 1),
// (sgitg / 2) == (sgitg >> 1)); RP=4 packs 8 simdgroups (256 threads,
// 8 rows + both column halves per TG, grid n_out/8).
template <short NC, short RP>
inline void mat_vec_q4_K_nc_impl(
        constant mat_vec_q4k_nc_args & args,
        device const uchar           * weight,
        device const float           * x,
        device       float           * y,
        uint tgpig, ushort sgitg, ushort tiisg) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    // NC=2 -> one column per simdgroup; NC=4 -> two; NC=8 -> four.
    constexpr short NC_SG = NC / 2;

    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;

    const uint nb = args.n_in / QK_K_NC;
    // sgitg % RP -> row-pair within the TG; sgitg / RP -> column half.
    const uint first_row = (tgpig * RP + (sgitg % RP)) * NR0_Q4K_NC;
    const short col_base = (short)(sgitg / RP) * NC_SG;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q4K_BYTES_NC;
    device const uchar * row0 = weight + first_row * row_stride_bytes;

    device const float * y4_base = x + ix * QK_K_NC + 64u * iq + 8u * ir
                                     + (ulong)col_base * args.n_in;

    float yl[16];
    float yh[16];
    float sumf[NR0_Q4K_NC][NC_SG];
    for (short r = 0; r < NR0_Q4K_NC; ++r) {
        for (short c = 0; c < NC_SG; ++c) {
            sumf[r][c] = 0.f;
        }
    }

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        for (short col = 0; col < NC_SG; ++col) {
            device const float * y4 = y4_base + (ulong)col * args.n_in;

            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
                yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
                yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
                yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
            }

            for (short row = 0; row < NR0_Q4K_NC; row++) {
                if (first_row + row >= args.n_out) break;

                device const uchar * blk = row0 + row * row_stride_bytes
                                          + (ulong)ib * Q4K_BYTES_NC;
                device const half     * dh = (device const half *) blk;
                device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq;
                device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq + 4 * ir;
                device const uint16_t * q2 = q1 + 32;

                sc16[0] =  sc[0]                & kmask1;
                sc16[1] =  sc[2]                & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
                }

                sumf[row][col] += (float)dh[0] * (
                      (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0]
                    + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f
                    + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4]
                    + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
                ) - (float)dh[1] * (
                      sumy[0] * sc8[2] + sumy[1] * sc8[3]
                    + sumy[2] * sc8[6] + sumy[3] * sc8[7]
                );
            }
        }

        y4_base += 4 * QK_K_NC;
    }

    for (short row = 0; row < NR0_Q4K_NC; row++) {
        for (short col = 0; col < NC_SG; ++col) {
            float total = simd_sum(sumf[row][col]);
            if (tiisg == 0 && first_row + row < args.n_out) {
                y[(ulong)(col_base + col) * args.n_out + first_row + row] = total;
            }
        }
    }
}

#define MAT_VEC_Q4K_NC_KERNEL(NCOLS, RP, NAME)                                \
kernel void NAME(                                                             \
        constant mat_vec_q4k_nc_args & args   [[buffer(0)]],                  \
        device const uchar           * weight [[buffer(1)]],                  \
        device const float           * x      [[buffer(2)]],                  \
        device       float           * y      [[buffer(3)]],                  \
        uint   tgpig [[threadgroup_position_in_grid]],                        \
        ushort sgitg [[simdgroup_index_in_threadgroup]],                      \
        ushort tiisg [[thread_index_in_simdgroup]]) {                         \
    mat_vec_q4_K_nc_impl<NCOLS, RP>(args, weight, x, y, tgpig, sgitg, tiisg); \
}

MAT_VEC_Q4K_NC_KERNEL(2, 2, kernel_mat_vec_q4_K_nc2_f32)
MAT_VEC_Q4K_NC_KERNEL(4, 2, kernel_mat_vec_q4_K_nc4_f32)
MAT_VEC_Q4K_NC_KERNEL(8, 2, kernel_mat_vec_q4_K_nc8_f32)
// v0.500 sweep B1: TG-shape point (8 SGs / 256 threads / 8 rows per TG).
MAT_VEC_Q4K_NC_KERNEL(2, 4, kernel_mat_vec_q4_K_nc2_rp4_f32)
