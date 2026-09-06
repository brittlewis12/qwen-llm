// Trellis-quant (QTIP-class) decode GEMV — BENCH-ONLY FLOOR KERNELS.
//
// Preregistered falsifier: docs/bench/2026-07-19-trellis3-gemv-floor/.
// NEVER referenced by production dispatch paths. These kernels measure one
// thing: whether an L=16 bitshift-trellis decode at K=3 bits/weight fits
// under the memory-latency shadow of a bandwidth-bound GEMV on Apple
// silicon (gate: >= 85% of the incumbent Q4_K GEMV's achieved GB/s on the
// same logical shape).
//
// Layout (must match the Rust packer/CPU reference bit-for-bit):
//   * Row-major rows over n_in; groups of 256 weights per (row, group).
//   * Group = 8 spans x 32 weights. Span = 96 bits packed LSB-first into
//     three uint32 words (bit b of the span stream lives in word b>>5 at
//     bit position b&31). 12 bytes per span, 96 bytes per group.
//   * Trellis window is tail-biting within the span's 96-bit ring:
//       - per-weight variants: state_j = 16 bits starting at
//         o_j = (3*j + 83) mod 96  (window ENDS at bit 3*(j+1)).
//       - V=2 variant: state_t = 16 bits starting at
//         o_t = (6*t + 86) mod 96  (window ENDS at bit 6*(t+1)); one
//         state yields TWO weights.
//   * Per-group fp16 scale in a separate buffer, scales[row*nb + group].
//
// Decode codes:
//   * 3inst   : X = state*A + B; hb = (X & 0x8FFF8FFF) | FIXED;
//               w = as_type<half2>(hb).x + .y            (per weight)
//   * 3inst_v2: same hash; .x and .y ARE the two weights  (per 2 weights)
//   * lut8x2  : w = LUT_hi[state>>8] + LUT_lo[state&255], two 256-entry
//               half LUTs in threadgroup memory           (per weight)
//
// FIXED keeps fp16 exponent bits [14:12] = 011 so decoded magnitudes lie
// in [2^-3, 2) with random sign/mantissa/low-exponent — no Inf/NaN.
//
// Dispatch geometry mirrors kernel_mat_vec_q4_K_f32: 32-lane simdgroup,
// NSG=2 simdgroups per threadgroup, NR0=2 rows per simdgroup.
//   ix = tiisg/8 in [0,4)  — group stride
//   it = tiisg%8 in [0,8)  — span within group (one lane owns one span)

#include <metal_stdlib>
using namespace metal;

constant constexpr uint T3_GROUP_BYTES = 96;   // 8 spans * 12 B
constant constexpr uint T3_GROUP_W     = 256;
constant constexpr uint T3_LCG_A = 89226354u;
constant constexpr uint T3_LCG_B = 64248484u;
constant constexpr uint T3_MASK  = 0x8FFF8FFFu;
// 0.922h = 0x3B60; keep bits [14:12] of each half (0x7000 field masked to
// the pattern of 0x3B60 & 0x7000 = 0x3000 | 0x0800 -> 0x3800).
constant constexpr uint T3_FIXED = (0x3B603B60u & ~T3_MASK);

#define NR0_T3 2
#define NSG_T3 2

struct trellis3_args {
    uint n_in;
    uint n_out;
};

// 16-bit window starting at ring bit o (compile-time constant under full
// unroll; all selects fold to immediates).
inline uint t3_ring_extract16(uint w0, uint w1, uint w2, uint o) {
    const uint a = o >> 5;
    const uint s = o & 31u;
    const uint wa = (a == 0) ? w0 : ((a == 1) ? w1 : w2);
    const uint wb = (a == 0) ? w1 : ((a == 1) ? w2 : w0);
    const uint v = (s == 0) ? wa : ((wa >> s) | (wb << (32u - s)));
    return v & 0xFFFFu;
}

// ---------------------------------------------------------------------------
// Variant A: 3inst (per-weight computed code)

kernel void kernel_mat_vec_trellis3_3inst_f32(
        constant trellis3_args & args   [[buffer(0)]],
        device const uchar     * weight [[buffer(1)]],
        device const half      * scales [[buffer(2)]],
        device const float     * x      [[buffer(3)]],
        device       float     * y      [[buffer(4)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;

    const uint nb = args.n_in / T3_GROUP_W;
    const uint first_row = (tgpig * NSG_T3 + sgitg) * NR0_T3;
    if (first_row >= args.n_out) return;
    const ulong row_stride = (ulong)nb * T3_GROUP_BYTES;

    float sumf[NR0_T3] = {0.f, 0.f};

    for (uint ib = ix; ib < nb; ib += 4) {
        device const float * yp = x + ib * T3_GROUP_W + (uint)it * 32u;
        half yh[32];
#pragma clang loop unroll(full)
        for (short i = 0; i < 32; ++i) {
            yh[i] = (half)yp[i];
        }

        for (short row = 0; row < NR0_T3; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uint * wp = (device const uint *)(
                weight + (first_row + row) * row_stride
                       + (ulong)ib * T3_GROUP_BYTES + (uint)it * 12u);
            const uint w0 = wp[0];
            const uint w1 = wp[1];
            const uint w2 = wp[2];

            half acc = 0.0h;
#pragma clang loop unroll(full)
            for (uint j = 0; j < 32; ++j) {
                const uint o  = (3u * j + 83u) % 96u;
                const uint st = t3_ring_extract16(w0, w1, w2, o);
                const uint X  = st * T3_LCG_A + T3_LCG_B;
                const uint hb = (X & T3_MASK) | T3_FIXED;
                const half2 h = as_type<half2>(hb);
                acc = fma((half)(h.x + h.y), yh[j], acc);
            }
            const half sc = scales[(first_row + row) * nb + ib];
            sumf[row] = fma((float)sc, (float)acc, sumf[row]);
        }
    }

    for (short row = 0; row < NR0_T3; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}

// ---------------------------------------------------------------------------
// Variant C: 3inst_v2 (one hash yields two weights)

kernel void kernel_mat_vec_trellis3_3inst_v2_f32(
        constant trellis3_args & args   [[buffer(0)]],
        device const uchar     * weight [[buffer(1)]],
        device const half      * scales [[buffer(2)]],
        device const float     * x      [[buffer(3)]],
        device       float     * y      [[buffer(4)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;

    const uint nb = args.n_in / T3_GROUP_W;
    const uint first_row = (tgpig * NSG_T3 + sgitg) * NR0_T3;
    if (first_row >= args.n_out) return;
    const ulong row_stride = (ulong)nb * T3_GROUP_BYTES;

    float sumf[NR0_T3] = {0.f, 0.f};

    for (uint ib = ix; ib < nb; ib += 4) {
        device const float * yp = x + ib * T3_GROUP_W + (uint)it * 32u;
        half yh[32];
#pragma clang loop unroll(full)
        for (short i = 0; i < 32; ++i) {
            yh[i] = (half)yp[i];
        }

        for (short row = 0; row < NR0_T3; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uint * wp = (device const uint *)(
                weight + (first_row + row) * row_stride
                       + (ulong)ib * T3_GROUP_BYTES + (uint)it * 12u);
            const uint w0 = wp[0];
            const uint w1 = wp[1];
            const uint w2 = wp[2];

            half acc = 0.0h;
#pragma clang loop unroll(full)
            for (uint t = 0; t < 16; ++t) {
                const uint o  = (6u * t + 86u) % 96u;
                const uint st = t3_ring_extract16(w0, w1, w2, o);
                const uint X  = st * T3_LCG_A + T3_LCG_B;
                const uint hb = (X & T3_MASK) | T3_FIXED;
                const half2 h = as_type<half2>(hb);
                acc = fma(h.x, yh[2 * t + 0], acc);
                acc = fma(h.y, yh[2 * t + 1], acc);
            }
            const half sc = scales[(first_row + row) * nb + ib];
            sumf[row] = fma((float)sc, (float)acc, sumf[row]);
        }
    }

    for (short row = 0; row < NR0_T3; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}

// ---------------------------------------------------------------------------
// T=256 group-ring variants (T6, docs/bench/2026-07-19-trellis3g-t256-kernels/).
//
// Same byte layout as the span variants (24 uint32 + fp16 scale per
// 256-weight group); only window semantics change: states are 16-bit
// windows over the GROUP's 768-bit ring (the T5 quality design point).
// A lane covering weights [32*it, 32*it+32) loads a 4-word local frame
// {prev word | its own 3 words}; window start for weight l is local bit
// p = 3l + 19 (V=1) / pair t is p = 6t + 22 (V=2), compile-time under
// full unroll. Lane 0's prev word is the group's word 23 (ring wrap).

inline uint t3g_frame_extract16(uint wa, uint w0, uint w1, uint w2, uint p) {
    const uint a = p >> 5;   // 0..3 (a==3 only with s<16 for our p range)
    const uint s = p & 31u;
    const uint lo = (a == 0) ? wa : ((a == 1) ? w0 : ((a == 2) ? w1 : w2));
    const uint hi = (a == 0) ? w0 : ((a == 1) ? w1 : ((a == 2) ? w2 : 0u));
    const uint v = (s == 0) ? lo : ((lo >> s) | (hi << (32u - s)));
    return v & 0xFFFFu;
}

kernel void kernel_mat_vec_trellis3g_3inst_f32(
        constant trellis3_args & args   [[buffer(0)]],
        device const uchar     * weight [[buffer(1)]],
        device const half      * scales [[buffer(2)]],
        device const float     * x      [[buffer(3)]],
        device       float     * y      [[buffer(4)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;

    const uint nb = args.n_in / T3_GROUP_W;
    const uint first_row = (tgpig * NSG_T3 + sgitg) * NR0_T3;
    if (first_row >= args.n_out) return;
    const ulong row_stride = (ulong)nb * T3_GROUP_BYTES;

    float sumf[NR0_T3] = {0.f, 0.f};

    for (uint ib = ix; ib < nb; ib += 4) {
        device const float * yp = x + ib * T3_GROUP_W + (uint)it * 32u;
        half yh[32];
#pragma clang loop unroll(full)
        for (short i = 0; i < 32; ++i) {
            yh[i] = (half)yp[i];
        }

        const uint base = 3u * (uint)it;
        for (short row = 0; row < NR0_T3; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uint * gw = (device const uint *)(
                weight + (first_row + row) * row_stride
                       + (ulong)ib * T3_GROUP_BYTES);
            const uint wa = gw[it == 0 ? 23u : base - 1u];
            const uint w0 = gw[base + 0u];
            const uint w1 = gw[base + 1u];
            const uint w2 = gw[base + 2u];

            half acc = 0.0h;
#pragma clang loop unroll(full)
            for (uint l = 0; l < 32; ++l) {
                const uint p  = 3u * l + 19u;
                const uint st = t3g_frame_extract16(wa, w0, w1, w2, p);
                const uint X  = st * T3_LCG_A + T3_LCG_B;
                const uint hb = (X & T3_MASK) | T3_FIXED;
                const half2 h = as_type<half2>(hb);
                acc = fma((half)(h.x + h.y), yh[l], acc);
            }
            const half sc = scales[(first_row + row) * nb + ib];
            sumf[row] = fma((float)sc, (float)acc, sumf[row]);
        }
    }

    for (short row = 0; row < NR0_T3; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}

// Tuning iteration 1 (T6): NSG=4 occupancy variant of the V1 group-ring
// kernel — Apple9 overlaps int/FP16/FP32 pipes across resident
// simdgroups, so doubling simdgroups per threadgroup targets the
// decode-latency exposure of the per-weight code.
kernel void kernel_mat_vec_trellis3g_3inst_nsg4_f32(
        constant trellis3_args & args   [[buffer(0)]],
        device const uchar     * weight [[buffer(1)]],
        device const half      * scales [[buffer(2)]],
        device const float     * x      [[buffer(3)]],
        device       float     * y      [[buffer(4)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;

    const uint nb = args.n_in / T3_GROUP_W;
    const uint first_row = (tgpig * 4u + sgitg) * NR0_T3;
    if (first_row >= args.n_out) return;
    const ulong row_stride = (ulong)nb * T3_GROUP_BYTES;

    float sumf[NR0_T3] = {0.f, 0.f};

    for (uint ib = ix; ib < nb; ib += 4) {
        device const float * yp = x + ib * T3_GROUP_W + (uint)it * 32u;
        half yh[32];
#pragma clang loop unroll(full)
        for (short i = 0; i < 32; ++i) {
            yh[i] = (half)yp[i];
        }

        const uint base = 3u * (uint)it;
        for (short row = 0; row < NR0_T3; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uint * gw = (device const uint *)(
                weight + (first_row + row) * row_stride
                       + (ulong)ib * T3_GROUP_BYTES);
            const uint wa = gw[it == 0 ? 23u : base - 1u];
            const uint w0 = gw[base + 0u];
            const uint w1 = gw[base + 1u];
            const uint w2 = gw[base + 2u];

            half acc = 0.0h;
#pragma clang loop unroll(full)
            for (uint l = 0; l < 32; ++l) {
                const uint p  = 3u * l + 19u;
                const uint st = t3g_frame_extract16(wa, w0, w1, w2, p);
                const uint X  = st * T3_LCG_A + T3_LCG_B;
                const uint hb = (X & T3_MASK) | T3_FIXED;
                const half2 h = as_type<half2>(hb);
                acc = fma((half)(h.x + h.y), yh[l], acc);
            }
            const half sc = scales[(first_row + row) * nb + ib];
            sumf[row] = fma((float)sc, (float)acc, sumf[row]);
        }
    }

    for (short row = 0; row < NR0_T3; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}

// Tuning iteration 3 (T6): NR0=4 row-ILP variant of the V1 group-ring
// kernel — four independent per-row decode chains per lane hide integer
// latency; y conversion amortizes over 4 rows. NSG=2.
kernel void kernel_mat_vec_trellis3g_3inst_nr4_f32(
        constant trellis3_args & args   [[buffer(0)]],
        device const uchar     * weight [[buffer(1)]],
        device const half      * scales [[buffer(2)]],
        device const float     * x      [[buffer(3)]],
        device       float     * y      [[buffer(4)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;

    const uint nb = args.n_in / T3_GROUP_W;
    const uint first_row = (tgpig * NSG_T3 + sgitg) * 4u;
    if (first_row >= args.n_out) return;
    const ulong row_stride = (ulong)nb * T3_GROUP_BYTES;

    float sumf[4] = {0.f, 0.f, 0.f, 0.f};

    for (uint ib = ix; ib < nb; ib += 4) {
        device const float * yp = x + ib * T3_GROUP_W + (uint)it * 32u;
        half yh[32];
#pragma clang loop unroll(full)
        for (short i = 0; i < 32; ++i) {
            yh[i] = (half)yp[i];
        }

        const uint base = 3u * (uint)it;
        for (short row = 0; row < 4; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uint * gw = (device const uint *)(
                weight + (first_row + row) * row_stride
                       + (ulong)ib * T3_GROUP_BYTES);
            const uint wa = gw[it == 0 ? 23u : base - 1u];
            const uint w0 = gw[base + 0u];
            const uint w1 = gw[base + 1u];
            const uint w2 = gw[base + 2u];

            half acc = 0.0h;
#pragma clang loop unroll(full)
            for (uint l = 0; l < 32; ++l) {
                const uint p  = 3u * l + 19u;
                const uint st = t3g_frame_extract16(wa, w0, w1, w2, p);
                const uint X  = st * T3_LCG_A + T3_LCG_B;
                const uint hb = (X & T3_MASK) | T3_FIXED;
                const half2 h = as_type<half2>(hb);
                acc = fma((half)(h.x + h.y), yh[l], acc);
            }
            const half sc = scales[(first_row + row) * nb + ib];
            sumf[row] = fma((float)sc, (float)acc, sumf[row]);
        }
    }

    for (short row = 0; row < 4; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}

kernel void kernel_mat_vec_trellis3g_3inst_v2_f32(
        constant trellis3_args & args   [[buffer(0)]],
        device const uchar     * weight [[buffer(1)]],
        device const half      * scales [[buffer(2)]],
        device const float     * x      [[buffer(3)]],
        device       float     * y      [[buffer(4)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;

    const uint nb = args.n_in / T3_GROUP_W;
    const uint first_row = (tgpig * NSG_T3 + sgitg) * NR0_T3;
    if (first_row >= args.n_out) return;
    const ulong row_stride = (ulong)nb * T3_GROUP_BYTES;

    float sumf[NR0_T3] = {0.f, 0.f};

    for (uint ib = ix; ib < nb; ib += 4) {
        device const float * yp = x + ib * T3_GROUP_W + (uint)it * 32u;
        half yh[32];
#pragma clang loop unroll(full)
        for (short i = 0; i < 32; ++i) {
            yh[i] = (half)yp[i];
        }

        const uint base = 3u * (uint)it;
        for (short row = 0; row < NR0_T3; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uint * gw = (device const uint *)(
                weight + (first_row + row) * row_stride
                       + (ulong)ib * T3_GROUP_BYTES);
            const uint wa = gw[it == 0 ? 23u : base - 1u];
            const uint w0 = gw[base + 0u];
            const uint w1 = gw[base + 1u];
            const uint w2 = gw[base + 2u];

            half acc = 0.0h;
#pragma clang loop unroll(full)
            for (uint t = 0; t < 16; ++t) {
                const uint p  = 6u * t + 22u;
                const uint st = t3g_frame_extract16(wa, w0, w1, w2, p);
                const uint X  = st * T3_LCG_A + T3_LCG_B;
                const uint hb = (X & T3_MASK) | T3_FIXED;
                const half2 h = as_type<half2>(hb);
                acc = fma(h.x, yh[2 * t + 0], acc);
                acc = fma(h.y, yh[2 * t + 1], acc);
            }
            const half sc = scales[(first_row + row) * nb + ib];
            sumf[row] = fma((float)sc, (float)acc, sumf[row]);
        }
    }

    for (short row = 0; row < NR0_T3; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}

// ---------------------------------------------------------------------------
// T8b: dual-3INST group-ring variant (T=256). One extract + one imad per
// pair, then rotl+xor makes a second independent hash word; each word's
// half-sum is one weight. T8a: 0.977x V1 quality at ~6.5 ops-eq/weight.

kernel void kernel_mat_vec_trellis3g_3inst_d_f32(
        constant trellis3_args & args   [[buffer(0)]],
        device const uchar     * weight [[buffer(1)]],
        device const half      * scales [[buffer(2)]],
        device const float     * x      [[buffer(3)]],
        device       float     * y      [[buffer(4)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;

    const uint nb = args.n_in / T3_GROUP_W;
    const uint first_row = (tgpig * NSG_T3 + sgitg) * NR0_T3;
    if (first_row >= args.n_out) return;
    const ulong row_stride = (ulong)nb * T3_GROUP_BYTES;

    float sumf[NR0_T3] = {0.f, 0.f};

    for (uint ib = ix; ib < nb; ib += 4) {
        device const float * yp = x + ib * T3_GROUP_W + (uint)it * 32u;
        half yh[32];
#pragma clang loop unroll(full)
        for (short i = 0; i < 32; ++i) {
            yh[i] = (half)yp[i];
        }

        const uint base = 3u * (uint)it;
        for (short row = 0; row < NR0_T3; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uint * gw = (device const uint *)(
                weight + (first_row + row) * row_stride
                       + (ulong)ib * T3_GROUP_BYTES);
            const uint wa = gw[it == 0 ? 23u : base - 1u];
            const uint w0 = gw[base + 0u];
            const uint w1 = gw[base + 1u];
            const uint w2 = gw[base + 2u];

            // Tuning iteration 1: split accumulators (a-chain / b-chain)
            // double the serial-fma ILP; folded once per group.
            half acc = 0.0h;
#pragma clang loop unroll(full)
            for (uint t = 0; t < 16; ++t) {
                const uint p  = 6u * t + 22u;
                const uint st = t3g_frame_extract16(wa, w0, w1, w2, p);
                const uint h  = st * T3_LCG_A + T3_LCG_B;
                const uint g  = h ^ rotate(h, 13u);
                const half2 a = as_type<half2>((h & T3_MASK) | T3_FIXED);
                const half2 b = as_type<half2>((g & T3_MASK) | T3_FIXED);
                acc = fma((half)(a.x + a.y), yh[2 * t + 0], acc);
                acc = fma((half)(b.x + b.y), yh[2 * t + 1], acc);
            }
            const half sc = scales[(first_row + row) * nb + ib];
            sumf[row] = fma((float)sc, (float)acc, sumf[row]);
        }
    }

    for (short row = 0; row < NR0_T3; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}

// ---------------------------------------------------------------------------
// Variant D: hyb_v2 (QTIP HYB-style, V=2): one hash + one cached half2
// lookup per TWO weights. LUT (512 x half2 = 2 KiB) is read directly from
// device memory (Apple9 flexible cache; per cx research, prefer this over
// threadgroup staging). Sign of .y flipped by hash bit 15.

kernel void kernel_mat_vec_trellis3_hyb_v2_f32(
        constant trellis3_args & args    [[buffer(0)]],
        device const uchar     * weight  [[buffer(1)]],
        device const half      * scales  [[buffer(2)]],
        device const float     * x       [[buffer(3)]],
        device       float     * y       [[buffer(4)]],
        device const half      * lut_dev [[buffer(5)]], // 512 x half2 (1024 halfs)
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    device const half2 * lut2 = (device const half2 *)lut_dev;

    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;

    const uint nb = args.n_in / T3_GROUP_W;
    const uint first_row = (tgpig * NSG_T3 + sgitg) * NR0_T3;
    if (first_row >= args.n_out) return;
    const ulong row_stride = (ulong)nb * T3_GROUP_BYTES;

    float sumf[NR0_T3] = {0.f, 0.f};

    for (uint ib = ix; ib < nb; ib += 4) {
        device const float * yp = x + ib * T3_GROUP_W + (uint)it * 32u;
        half yh[32];
#pragma clang loop unroll(full)
        for (short i = 0; i < 32; ++i) {
            yh[i] = (half)yp[i];
        }

        for (short row = 0; row < NR0_T3; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uint * wp = (device const uint *)(
                weight + (first_row + row) * row_stride
                       + (ulong)ib * T3_GROUP_BYTES + (uint)it * 12u);
            const uint w0 = wp[0];
            const uint w1 = wp[1];
            const uint w2 = wp[2];

            half acc = 0.0h;
#pragma clang loop unroll(full)
            for (uint t = 0; t < 16; ++t) {
                const uint o  = (6u * t + 86u) % 96u;
                const uint st = t3_ring_extract16(w0, w1, w2, o);
                const uint h  = st * st + st;
                const uint idx = (h >> 6) & 511u;
                uint hb = as_type<uint>(lut2[idx]);
                hb ^= (h & 0x8000u) << 16;   // flip sign of .y by hash bit 15
                const half2 v = as_type<half2>(hb);
                acc = fma(v.x, yh[2 * t + 0], acc);
                acc = fma(v.y, yh[2 * t + 1], acc);
            }
            const half sc = scales[(first_row + row) * nb + ib];
            sumf[row] = fma((float)sc, (float)acc, sumf[row]);
        }
    }

    for (short row = 0; row < NR0_T3; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}

// ---------------------------------------------------------------------------
// Variant B: lut8x2 (two 256-entry threadgroup-memory LUTs)

kernel void kernel_mat_vec_trellis3_lut8x2_f32(
        constant trellis3_args & args    [[buffer(0)]],
        device const uchar     * weight  [[buffer(1)]],
        device const half      * scales  [[buffer(2)]],
        device const float     * x       [[buffer(3)]],
        device       float     * y       [[buffer(4)]],
        device const half      * lut_dev [[buffer(5)]], // [512]: 256 hi, 256 lo
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_position_in_threadgroup]],
        ushort ntg   [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup half lut_hi[256];
    threadgroup half lut_lo[256];
    for (ushort i = tpitg; i < 256; i += ntg) {
        lut_hi[i] = lut_dev[i];
        lut_lo[i] = lut_dev[256 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;

    const uint nb = args.n_in / T3_GROUP_W;
    const uint first_row = (tgpig * NSG_T3 + sgitg) * NR0_T3;
    if (first_row >= args.n_out) return;
    const ulong row_stride = (ulong)nb * T3_GROUP_BYTES;

    float sumf[NR0_T3] = {0.f, 0.f};

    for (uint ib = ix; ib < nb; ib += 4) {
        device const float * yp = x + ib * T3_GROUP_W + (uint)it * 32u;
        half yh[32];
#pragma clang loop unroll(full)
        for (short i = 0; i < 32; ++i) {
            yh[i] = (half)yp[i];
        }

        for (short row = 0; row < NR0_T3; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uint * wp = (device const uint *)(
                weight + (first_row + row) * row_stride
                       + (ulong)ib * T3_GROUP_BYTES + (uint)it * 12u);
            const uint w0 = wp[0];
            const uint w1 = wp[1];
            const uint w2 = wp[2];

            half acc = 0.0h;
#pragma clang loop unroll(full)
            for (uint j = 0; j < 32; ++j) {
                const uint o  = (3u * j + 83u) % 96u;
                const uint st = t3_ring_extract16(w0, w1, w2, o);
                const half wv = lut_hi[st >> 8] + lut_lo[st & 255u];
                acc = fma(wv, yh[j], acc);
            }
            const half sc = scales[(first_row + row) * nb + ib];
            sumf[row] = fma((float)sc, (float)acc, sumf[row]);
        }
    }

    for (short row = 0; row < NR0_T3; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}
