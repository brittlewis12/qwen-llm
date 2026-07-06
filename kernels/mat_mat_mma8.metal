// Small-N MMA experiment (v0.498 follow-up): 8-row x 8-column
// simdgroup_matrix tile at mat-vec-grade occupancy.
//
// Question being falsified (see PERF-LOG v0.498 / PERF-ROADMAP): the
// incumbent 64x32 MM tile runs skinny-N verify shapes at 56-128 GB/s
// because 64-row tiles yield only n_out/64 threadgroups (272 at ffn
// shapes on 40 cores). This kernel asks whether simdgroup_matrix at
// n_out/8 threadgroups (2176 at ffn_gate) can beat the ~270 GB/s-equiv
// scalar-ALU cap measured for multi-column GEMV (v0.498). On M4
// simdgroup_matrix is NOT a separate pool (same FP32 pipes) — the
// candidate win is dequant-once-per-weight + dense FMA encoding +
// fixed occupancy. Kill line (recorded): c(2) <= 1.25 on ffn_gate AND
// ffn_down, else the shallow small-N lane closes too.
//
// Geometry: one simdgroup (32 threads) per threadgroup; each TG owns 8
// output rows and ALWAYS computes 8 activation columns (callers with
// N < 8 pad x to [8, n_in] and y to [8, n_out]; padded columns cost no
// extra weight bytes and FLOPs are below the roofline by construction).
// K-step 64: each thread dequants one 16-element chunk (row = lane/4,
// chunk = lane%4) into a static threadgroup A-tile [8][64]; B tiles are
// simdgroup_load'ed DIRECTLY from device F32 activations with
// transpose (no staging, no half-rounding of activations — tighter
// than the incumbent mat-mat's half-staged B). 8 MMAs per K-step into
// one simdgroup_float8x8 accumulator.
//
// Exactness tier: E1 (weights staged through the same half dequant as
// the incumbent mat-mat; MMA accumulation order differs from mv1).
// Gate: cos >= 0.999 per live column vs per-column mv1, asserted by
// `smalln_mma_micro_27b` in tests/dflash_correctness.rs.
//
// Layouts:
//   weight: raw quant block bytes, [n_out, n_in]
//   x:      F32 [8, n_in]  row-major (column c = x + c*n_in; caller pads)
//   y:      F32 [8, n_out] row-major (y[c*n_out + row]; caller pads)

#include <metal_stdlib>
using namespace metal;

#ifndef FOR_UNROLL
#define FOR_UNROLL _Pragma("clang loop unroll(full)") for
#endif

constant constexpr int QK_K_MMA8 = 256;

struct mat_mat_mma8_args {
    uint n_in;
    uint n_out;
};

// --- Q4_K dequant (identical math to mat_mat_q4_k.metal's helper) -------
inline void mma8_dequantize_q4_K_half(device const uchar * blk_bytes,
                                      short il,
                                      thread half4x4 & reg) {
    const half d_h    = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar * scales = blk_bytes + 4;
    device const uchar * qs     = blk_bytes + 4 + 12;

    const short is  = (il / 4) * 2;
    const short k01 = (il / 2) & 1;
    uchar sc_u, m_u;
    if (is < 4) {
        sc_u = scales[is + k01] & 63;
        m_u  = scales[is + k01 + 4] & 63;
    } else {
        sc_u = (scales[is + k01 + 4] & 0x0F) | ((scales[is + k01 - 4] >> 6) << 4);
        m_u  = (scales[is + k01 + 4] >>   4) | ((scales[is + k01    ] >> 6) << 4);
    }

    qs = qs + (il / 4) * 32 + 16 * (il & 1);
    short il_inner = il & 3;
    const float d    = il_inner < 2 ? (float)d_h : (float)d_h / 16.0f;
    const float dmin = (float)dmin_h;
    const float dl   = d * (float)sc_u;
    const float ml   = dmin * (float)m_u;
    const ushort mask = il_inner < 2 ? 0x0F : 0xF0;

    FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(dl * (float)(qs[i] & mask) - ml);
    }
}

// --- Q6_K dequant (identical math to mat_mat_q6_k.metal's helper) -------
inline void mma8_dequantize_q6_K_half(device const uchar * blk_bytes,
                                      short il,
                                      thread half4x4 & reg) {
    device const uint16_t * ql = (device const uint16_t *)(blk_bytes + 0);
    device const uint16_t * qh = (device const uint16_t *)(blk_bytes + 128);
    device const int8_t   * scales = (device const int8_t *)(blk_bytes + 128 + 64);
    const half d_all = ((device const half *)(blk_bytes + 128 + 64 + 16))[0];

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    float sc = scales[(il % 2) + 2 * ((il / 2))];
    short il_inner = (il / 2) & 3;

    const uint32_t kmask1 = il_inner > 1
        ? (il_inner > 2 ? 0xC0C0C0C0 : 0x30303030)
        : (il_inner > 0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il_inner > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float ml  = (float)d_all * sc * 32.0f;
    const float dl0 = (float)d_all * sc;
    const float dl1 = dl0 / 256.0f;
    const float dl2 = dl0 / (256.0f * 256.0f);
    const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
    const uint8_t shr_h = il_inner > 2 ? 2 : 0;
    const uint8_t shl_h = il_inner > 1 ? 0 : (il_inner > 0 ? 2 : 4);
    const uint8_t shr_l = il_inner > 1 ? 4 : 0;

    FOR_UNROLL (int i = 0; i < 4; ++i) {
        const uint32_t low  = (ql[2 * i] | (uint32_t)(ql[2 * i + 1] << 16)) & kmask2;
        const uint32_t high = (qh[2 * i] | (uint32_t)(qh[2 * i + 1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = (half)(dl0 * ((float)(q & 0xFF))         - ml);
        reg[i][1] = (half)(dl1 * ((float)(q & 0xFF00))       - ml);
        reg[i][2] = (half)(dl2 * ((float)(q & 0xFF0000))     - ml);
        reg[i][3] = (half)(dl3 * ((float)(q & 0xFF000000))   - ml);
    }
}

// --- Shared 8x8-tile body -------------------------------------------------
// DEQ: dequant fn; BLK_BYTES: quant block size in bytes.
//
// Variant record (v0.499 falsifier, both measured on the 27B shapes):
//   v1 (THIS body): K-step 64, A staged 8x64 float (2 KiB TG), B loaded
//      transposed DIRECTLY from device F32 -> c = 1.93-2.65, 4.4-5.2 TF.
//   v2 (reverted): K-step 128 + B staged through TG with coalesced
//      float4 reads (8 KiB TG total) -> c = 2.53-4.44, 3.0-3.4 TF.
//      The staging roundtrip + halved TG residency lost to v1's
//      L1-served scattered B loads; direct device B was NOT the binder.
#define MAT_MAT_MMA8_BODY(DEQ, BLK_BYTES)                                     \
    const uint r0 = tgpig * 8;                                                \
    if (r0 >= args.n_out) return;                                             \
    const uint nb = args.n_in / QK_K_MMA8;                                    \
    const ulong row_stride_bytes = (ulong)nb * (BLK_BYTES);                   \
                                                                              \
    /* Static A tile: 8 rows x 64 K, float (dequant half -> float). */        \
    threadgroup float sa[8 * 64];                                             \
                                                                              \
    /* Lane mapping: row = lane/4 in [0,8); chunk = lane%4 in [0,4). */       \
    const short lrow  = tiisg / 4;                                            \
    const short lchnk = tiisg % 4;                                            \
    device const uchar * wrow = weight + (r0 + lrow) * row_stride_bytes;      \
                                                                              \
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);    \
                                                                              \
    for (uint k0 = 0; k0 < args.n_in; k0 += 64) {                             \
        /* Dequant this thread's 16-element chunk into sa. */                 \
        {                                                                     \
            const uint kbase = k0 + (uint)lchnk * 16;                         \
            device const uchar * blk = wrow                                   \
                + (ulong)(kbase / QK_K_MMA8) * (BLK_BYTES);                   \
            const short il = (short)((kbase % QK_K_MMA8) / 16);               \
            half4x4 tmp;                                                      \
            DEQ(blk, il, tmp);                                                \
            FOR_UNROLL (int i = 0; i < 16; ++i) {                             \
                sa[lrow * 64 + lchnk * 16 + i] = (float)tmp[i / 4][i % 4];    \
            }                                                                 \
        }                                                                     \
        simdgroup_barrier(mem_flags::mem_threadgroup);                        \
                                                                              \
        FOR_UNROLL (short kt = 0; kt < 8; ++kt) {                             \
            simdgroup_float8x8 ma;                                            \
            simdgroup_float8x8 mb;                                            \
            simdgroup_load(ma, sa + kt * 8, 64);                              \
            /* B: 8 activation columns (device rows, stride n_in) x 8 K,  */  \
            /* loaded transposed -> [k][n]. F32 straight from device.     */  \
            /* Alignment facts: host enforces n_in % 256 == 0 and x is a  */  \
            /* dedicated F32 [8, n_in] buffer, so every row base and the  */  \
            /* k0 + kt*8 offset are >= 8-float aligned.                   */  \
            simdgroup_load(mb, x + k0 + (uint)kt * 8, args.n_in,              \
                           ulong2(0, 0), true);                               \
            simdgroup_multiply_accumulate(acc, ma, mb, acc);                  \
        }                                                                     \
        simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    }                                                                         \
                                                                              \
    /* Epilogue: stage C [8 rows][8 cols] and scatter to y[c*n_out+r]. */     \
    threadgroup float sc_out[8 * 8];                                          \
    simdgroup_store(acc, sc_out, 8);                                          \
    simdgroup_barrier(mem_flags::mem_threadgroup);                            \
    /* 32 lanes x 2 cells: cell = lane*2 + {0,1}, cell = row*8 + col. */      \
    FOR_UNROLL (short e = 0; e < 2; ++e) {                                    \
        const short cell = (short)tiisg * 2 + e;                              \
        const short row  = cell / 8;                                          \
        const short col  = cell % 8;                                          \
        if (r0 + (uint)row < args.n_out) {                                    \
            y[(ulong)col * args.n_out + r0 + row] = sc_out[row * 8 + col];    \
        }                                                                     \
    }

kernel void kernel_mat_mat_q4_K_mma8_f32(
        constant mat_mat_mma8_args & args   [[buffer(0)]],
        device const uchar         * weight [[buffer(1)]],
        device const float         * x      [[buffer(2)]],
        device       float         * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    MAT_MAT_MMA8_BODY(mma8_dequantize_q4_K_half, 144)
}

kernel void kernel_mat_mat_q6_K_mma8_f32(
        constant mat_mat_mma8_args & args   [[buffer(0)]],
        device const uchar         * weight [[buffer(1)]],
        device const float         * x      [[buffer(2)]],
        device       float         * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    MAT_MAT_MMA8_BODY(mma8_dequantize_q6_K_half, 210)
}
