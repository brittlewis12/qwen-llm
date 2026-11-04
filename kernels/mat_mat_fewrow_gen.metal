// Few-row generic mat-mat (2..16 activation rows) on 8x8 simdgroup matrices.
//
// Adapted from llama.cpp's `kernel_mul_mv_mma_gen` and its `dequantize_*`
// helpers (ggml/src/ggml-metal/kernels/mul_mv_mma.metal and dequantize.h,
// PRs #29869 / #30065, MIT license; see docs/THIRD-PARTY-NOTICES.md).
// Differences from upstream: one 2D matrix (no broadcast batch dimensions),
// no fused residual add, and NSG is a template parameter instead of a
// function constant.
//
// A lane dequantizes its 16 consecutive weights of a 64-weight chunk once
// for all activation rows. The 16 weights fill a half4x4, and MMA step s
// reads weights 16*(j/2) + 8*(j%2) + s. Activations stay F32; the weight
// fragments are half, so accumulation order and weight rounding differ from
// the mat-vec decode chain (E1).
//
// Layout matches the mma8v kernels: weight is row-major block storage
// [n_out][n_in/block_weights], x is F32 [n_cols][n_in], y is F32
// [n_cols][n_out]. A threadgroup owns 8*NT output rows and 8*RT activation
// rows; its NSG simdgroups split K by 64-weight chunk and one threadgroup
// reduction sums their partials. Out-of-range rows and columns are clamped
// on load and masked on store, so n_out and n_cols need no tile multiple.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)") for

struct mat_mat_fewrow_gen_args {
    uint n_in;
    uint n_out;
    uint n_cols;
};

// The A-fragment row and first B-fragment column lane l holds in an 8x8
// simdgroup matrix.
inline short fewrow_gen_lane_fm(ushort l) { return ((l / 4) & 4) + ((l / 2) % 4); }
inline short fewrow_gen_lane_fn(ushort l) { return ((l / 4) & 2) * 2 + (l % 2) * 2; }

// Q4_K: d + dmin, 12 packed scale/min bytes, 128 quant bytes.
struct fewrow_gen_q4_K {
    enum { block_bytes = 144, block_weights = 256, nl = 16 };

    static inline uchar2 scale_min(int j, int k, device const uchar * q) {
        return j < 4
            ? uchar2(q[j + 0 + k] & 63, q[j + 4 + k] & 63)
            : uchar2((q[j + 4 + k] & 0x0F) | ((q[j - 4 + k] & 0xC0) >> 2),
                     (q[j + 4 + k] >> 4) | ((q[j + 0 + k] & 0xC0) >> 2));
    }

    static inline void dequant(device const uchar * xb, short il, thread float4x4 & reg) {
        device const uchar * scales = xb + 4;
        device const uchar * qs = xb + 16;

        const short is = (il / 4) * 2;
        qs += (il / 4) * 32 + 16 * (il & 1);
        const short il4 = il & 3;
        const uchar2 sc = scale_min(is, il4 / 2, scales);
        const float d = float(as_type<half>(*(device const ushort *) xb));
        const float dmin = float(as_type<half>(*(device const ushort *) (xb + 2)));
        const float dl = (il4 < 2 ? d : d / 16.0f) * float(sc[0]);
        const float ml = dmin * float(sc[1]);
        const ushort mask = il4 < 2 ? 0x0F : 0xF0;

        FOR_UNROLL (int i = 0; i < 16; ++i) {
            reg[i / 4][i % 4] = dl * float(qs[i] & mask) - ml;
        }
    }
};

// Q5_K: d + dmin, 12 packed scale/min bytes, 32 high-bit bytes, 128 quant
// bytes.
struct fewrow_gen_q5_K {
    enum { block_bytes = 176, block_weights = 256, nl = 16 };

    static inline uchar2 scale_min(int j, int k, device const uchar * q) {
        return j < 4
            ? uchar2(q[j + 0 + k] & 63, q[j + 4 + k] & 63)
            : uchar2((q[j + 4 + k] & 0x0F) | ((q[j - 4 + k] & 0xC0) >> 2),
                     (q[j + 4 + k] >> 4) | ((q[j + 0 + k] & 0xC0) >> 2));
    }

    static inline void dequant(device const uchar * xb, short il, thread float4x4 & reg) {
        device const uchar * scales = xb + 4;
        device const uchar * qh = xb + 16;
        device const uchar * qs = xb + 48;

        const short is = (il / 4) * 2;
        qs += 32 * (il / 4) + 16 * (il & 1);
        qh += 16 * (il & 1);
        const uchar ul = uchar(1u << (il / 2));
        const short il4 = il & 3;
        const uchar2 sc = scale_min(is, il4 / 2, scales);
        const float d = float(as_type<half>(*(device const ushort *) xb));
        const float dmin = float(as_type<half>(*(device const ushort *) (xb + 2)));
        const float dl = (il4 < 2 ? d : d / 16.0f) * float(sc[0]);
        const float ml = dmin * float(sc[1]);
        const ushort mask = il4 < 2 ? 0x0F : 0xF0;
        const float qh_val = il4 < 2 ? 16.0f : 256.0f;

        FOR_UNROLL (int i = 0; i < 16; ++i) {
            reg[i / 4][i % 4] = dl * (float(qs[i] & mask) + (qh[i] & ul ? qh_val : 0.0f)) - ml;
        }
    }
};

// Q6_K: 128 low-bit bytes, 64 high-bit bytes, 16 int8 scales, d.
struct fewrow_gen_q6_K {
    enum { block_bytes = 210, block_weights = 256, nl = 16 };

    static inline void dequant(device const uchar * xb, short il, thread float4x4 & reg) {
        device const uchar * ql = xb;
        device const uchar * qh = xb + 128;
        device const char  * scales = (device const char *) (xb + 192);
        const float d_all = float(as_type<half>(*(device const ushort *) (xb + 208)));

        ql += 64 * (il / 8) + 32 * ((il / 2) & 1) + 16 * (il & 1);
        qh += 32 * (il / 8) + 16 * (il & 1);
        float sc = float(scales[(il % 2) + 2 * (il / 2)]);
        const short il4 = (il / 2) & 3;

        const uint kmask1 = il4 > 1 ? (il4 > 2 ? 0xC0C0C0C0u : 0x30303030u)
                                    : (il4 > 0 ? 0x0C0C0C0Cu : 0x03030303u);
        const uint kmask2 = il4 > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
        const float ml = d_all * sc * 32.0f;
        const float dl0 = d_all * sc;
        const float dl1 = dl0 / 256.0f;
        const float dl2 = dl0 / (256.0f * 256.0f);
        const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
        const uint shr_h = il4 > 2 ? 2u : 0u;
        const uint shl_h = il4 > 1 ? 0u : (il4 > 0 ? 2u : 4u);
        const uint shr_l = il4 > 1 ? 4u : 0u;

        device const ushort * ql16 = (device const ushort *) ql;
        device const ushort * qh16 = (device const ushort *) qh;

        FOR_UNROLL (int i = 0; i < 4; ++i) {
            // Upstream reads two 16-bit words per 32-bit lane; the block
            // base is only 2-byte aligned, so 32-bit loads are not safe.
            const uint low = (uint(ql16[2 * i]) | (uint(ql16[2 * i + 1]) << 16)) & kmask2;
            const uint high = (uint(qh16[2 * i]) | (uint(qh16[2 * i + 1]) << 16)) & kmask1;
            const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
            reg[i][0] = dl0 * float(q & 0xFFu) - ml;
            reg[i][1] = dl1 * float(q & 0xFF00u) - ml;
            reg[i][2] = dl2 * float(q & 0xFF0000u) - ml;
            reg[i][3] = dl3 * float(q & 0xFF000000u) - ml;
        }
    }
};

// Q8_0: half d + 32 int8 quants; two 16-weight chunks.
struct fewrow_gen_q8_0 {
    enum { block_bytes = 34, block_weights = 32, nl = 2 };

    static inline void dequant(device const uchar * xb, short il, thread float4x4 & reg) {
        const float d = float(as_type<half>(*(device const ushort *) xb));
        const device const char * qs = (device const char *) (xb + 2) + 16 * il;
        FOR_UNROLL (int i = 0; i < 4; ++i) {
            reg[i] = float4(float(qs[4 * i + 0]), float(qs[4 * i + 1]),
                            float(qs[4 * i + 2]), float(qs[4 * i + 3])) * d;
        }
    }
};

// Few-row mat-mat for any type with a 16-weight dequantizer: a lane
// dequantizes 16 consecutive weights of a 64-weight chunk once for all
// activation rows.
template <typename Q, short NT, short RT, short NSG>
kernel void kernel_mat_mat_fewrow_gen_f32(
    constant mat_mat_fewrow_gen_args & args [[buffer(0)]],
    device const uchar * weight [[buffer(1)]],
    device const float * x [[buffer(2)]],
    device float * y [[buffer(3)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[NSG * RT * NT * 64];

    const short fm = fewrow_gen_lane_fm(tiisg);
    const short fn = fewrow_gen_lane_fn(tiisg);
    const int n_out = int(args.n_out);
    const int n_cols = int(args.n_cols);
    const int i01 = int(tgpig) * (8 * NT);
    const int nch = int(args.n_in / 64);
    const ulong row_bytes = ulong(args.n_in / Q::block_weights) * Q::block_bytes;

    device const uchar * xr[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        const int r = min(i01 + 8 * t + fm, n_out - 1);
        xr[t] = weight + ulong(r) * row_bytes;
    }

    // B lane j = fm reads chunk values 16*(fm/2) + 8*(fm%2) .. +7.
    device const float4 * yb[RT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            const int r = min(8 * rt + fn + e, n_cols - 1);
            yb[rt][e] = (device const float4 *) (x + ulong(r) * args.n_in) + 4 * (fm / 2) +
                        2 * (fm % 2);
        }
    }

    simdgroup_float8x8 mc[RT][NT];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            mc[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    for (int g = sgitg; g < nch; g += NSG) {
        simdgroup_float8x8 mb[RT][8];
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            const float4 a0 = yb[rt][0][16 * g + 0];
            const float4 a1 = yb[rt][0][16 * g + 1];
            const float4 b0 = yb[rt][1][16 * g + 0];
            const float4 b1 = yb[rt][1][16 * g + 1];
            FOR_UNROLL (short s = 0; s < 4; ++s) {
                mb[rt][s].thread_elements()[0] = a0[s];
                mb[rt][s].thread_elements()[1] = b0[s];
                mb[rt][s + 4].thread_elements()[0] = a1[s];
                mb[rt][s + 4].thread_elements()[1] = b1[s];
            }
        }

        const int ci = 4 * g + fn / 2;

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            float4x4 w;
            Q::dequant(xr[t] + (ci / Q::nl) * Q::block_bytes, ci % Q::nl, w);

            FOR_UNROLL (short s = 0; s < 8; ++s) {
                simdgroup_half8x8 ma;
                ma.thread_elements()[0] = half(w[s / 4][s % 4]);
                ma.thread_elements()[1] = half(w[s / 4 + 2][s % 4]);

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    simdgroup_multiply_accumulate(mc[rt][t], ma, mb[rt][s], mc[rt][t]);
                }
            }
        }
    }

    // Sum the K slices of the NSG simdgroups and store the 8*NT x 8*RT tile.
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            red[((sgitg * RT + rt) * NT + t) * 64 + 2 * tiisg + 0] =
                mc[rt][t].thread_elements()[0];
            red[((sgitg * RT + rt) * NT + t) * 64 + 2 * tiisg + 1] =
                mc[rt][t].thread_elements()[1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short idx = sgitg * 32 + tiisg; idx < RT * NT * 64; idx += NSG * 32) {
        float sum = 0.0f;
        for (short sg = 0; sg < NSG; ++sg) {
            sum += red[sg * (RT * NT * 64) + idx];
        }
        const short rt = idx / (NT * 64);
        const short t = (idx / 64) % NT;
        const short l = (idx % 64) / 2;
        const short e = idx % 2;
        const int r0 = i01 + 8 * t + fewrow_gen_lane_fm(l);
        const int r1 = 8 * rt + fewrow_gen_lane_fn(l) + e;
        if (r0 < n_out && r1 < n_cols) {
            y[ulong(r1) * args.n_out + r0] = sum;
        }
    }
}

typedef decltype(kernel_mat_mat_fewrow_gen_f32<fewrow_gen_q4_K, 1, 1, 8>)
    mat_mat_fewrow_gen_t;

#define FEWROW_GEN(NAME, Q, NT, RT, NSG)                                              \
    template [[host_name("kernel_mat_mat_" #NAME "_fewrow_gen_nt" #NT "_rt" #RT      \
                         "_nsg" #NSG "_f32")]] kernel mat_mat_fewrow_gen_t            \
        kernel_mat_mat_fewrow_gen_f32<Q, NT, RT, NSG>;

#define FEWROW_GEN_TYPE(NAME, Q)                                                      \
    FEWROW_GEN(NAME, Q, 1, 1, 8)                                                      \
    FEWROW_GEN(NAME, Q, 2, 1, 8)                                                      \
    FEWROW_GEN(NAME, Q, 4, 1, 8)                                                      \
    FEWROW_GEN(NAME, Q, 1, 1, 16)                                                     \
    FEWROW_GEN(NAME, Q, 2, 1, 16)                                                     \
    FEWROW_GEN(NAME, Q, 4, 1, 16)                                                     \
    FEWROW_GEN(NAME, Q, 1, 1, 32)                                                     \
    FEWROW_GEN(NAME, Q, 2, 1, 32)                                                     \
    FEWROW_GEN(NAME, Q, 4, 1, 32)                                                     \
    FEWROW_GEN(NAME, Q, 1, 2, 8)                                                      \
    FEWROW_GEN(NAME, Q, 2, 2, 8)                                                      \
    FEWROW_GEN(NAME, Q, 4, 2, 8)                                                      \
    FEWROW_GEN(NAME, Q, 1, 2, 16)                                                     \
    FEWROW_GEN(NAME, Q, 2, 2, 16)                                                     \
    FEWROW_GEN(NAME, Q, 1, 2, 32)

FEWROW_GEN_TYPE(q4_K, fewrow_gen_q4_K)
FEWROW_GEN_TYPE(q5_K, fewrow_gen_q5_K)
FEWROW_GEN_TYPE(q6_K, fewrow_gen_q6_K)
FEWROW_GEN_TYPE(q8_0, fewrow_gen_q8_0)
