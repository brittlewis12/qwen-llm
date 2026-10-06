// Few-row Q8_0 mat-mat (1..8 activation rows) on 8x8 simdgroup matrices.
//
// Adapted from llama.cpp's `kernel_mul_mv_mma_blk<NT, RT, mul_mv_mma_q8_0>`
// (ggml/src/ggml-metal/kernels/mul_mv_mma.metal, PR #29869, MIT license;
// see docs/THIRD-PARTY-NOTICES.md). Differences from upstream: one 2D
// matrix (no broadcast batch dimensions), RT = 1 (up to 8 activation rows),
// no fused residual add, and the simdgroup count NSG is a template
// parameter instead of a function constant.
//
// Layout matches the mma8v kernels: weight is row-major [n_out][n_in/32]
// Q8_0 blocks (half d + 32 int8), x is F32 [n_cols][n_in], y is F32
// [n_cols][n_out]. A threadgroup owns 8*NT output rows; its NSG simdgroups
// split K by block (ib = sgitg, sgitg + NSG, ...) and one threadgroup
// reduction sums their partials. Each weight is read once for all columns:
// the quants fill a half A fragment directly from registers (sign flip makes
// them unsigned; 1024 + q is an exact half built with an OR), the block scale
// is applied in FP32 after the MMA, and the next block is loaded into
// registers while the current one computes. Out-of-range rows and columns
// are clamped on load and masked on store, so n_out and n_cols need no
// multiple-of-tile shape.
//
// Exactness tier: E1 (MMA accumulation order differs from the mat-vec decode
// chain); the DFlash verify guard is calibrated on the resulting margin error.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)") for

struct mat_mat_q8_0_fewrow_args {
    uint n_in;
    uint n_out;
    uint n_cols;
};

// block_q8_0 = half d + int8 qs[32] = 34 bytes = 17 ushorts.
constant constexpr short Q8_BLOCK_US = 17;
constant constexpr ushort FEWROW_F16_1024_BITS = 0x6400;
constant constexpr half FEWROW_F16_1024 = 1024.0h;

// The A-fragment row and first B-fragment column lane l holds in an 8x8
// simdgroup matrix.
inline short fewrow_lane_fm(ushort l) { return ((l / 4) & 4) + ((l / 2) % 4); }
inline short fewrow_lane_fn(ushort l) { return ((l / 4) & 2) * 2 + (l % 2) * 2; }

// The halves 1024 + q for integers q < 1024: exact normal values.
inline half2 fewrow_1024_plus(ushort2 q) { return as_type<half2>(q | FEWROW_F16_1024_BITS); }

// A lane (m, j) holds qs bytes 4*j .. 4*j + 7 (j even); B lane k = fm holds
// x values b, b + 1 (b0) and b + 4, b + 5 (b1), b = 8*(k/2) + 2*(k%2). MMA
// step s uses b1 when s >= 2 and the .y value of a pair when s is odd.
inline half2 fewrow_q8_frag(ushort4 q, short s) {
    const ushort2 w = s < 2 ? q.xy : q.zw;
    const ushort2 qq = s % 2 == 0 ? w : w >> 8;
    return fewrow_1024_plus(qq & ushort2(0x00FF)) - (FEWROW_F16_1024 + 128.0h);
}

template <short NT>
inline void fewrow_load_a(device const ushort *const x[NT], int ib, thread ushort4 *q) {
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        device const ushort *qs = x[t] + ib * Q8_BLOCK_US;
        q[t] = ushort4(qs[0], qs[1], qs[2], qs[3]);
    }
}

template <short NT, short NSG>
kernel void kernel_mat_mat_q8_0_fewrow_f32(
    constant mat_mat_q8_0_fewrow_args &args [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[NSG * NT * 64];

    const short fm = fewrow_lane_fm(tiisg);
    const short fn = fewrow_lane_fn(tiisg);
    const int n_out = int(args.n_out);
    const int n_cols = int(args.n_cols);
    const int i01 = int(tgpig) * (8 * NT);
    const int nb = int(args.n_in / 32);
    const ulong row_bytes = ulong(nb) * 34;

    // Each lane's quants start after the half scale, 4*fn bytes into a block.
    device const ushort *xq[NT];
    device const ushort *xd[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        const int r = min(i01 + 8 * t + fm, n_out - 1);
        xd[t] = (device const ushort *)(weight + ulong(r) * row_bytes);
        xq[t] = xd[t] + 1 + 2 * fn;
    }
    device const float2 *yb[2];
    FOR_UNROLL (short e = 0; e < 2; ++e) {
        const int c = min(fn + e, n_cols - 1);
        yb[e] = (device const float2 *)(x + ulong(c) * args.n_in + 8 * (fm / 2) + 2 * (fm % 2));
    }

    float acc[NT][2];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        acc[t][0] = 0.0f;
        acc[t][1] = 0.0f;
    }

    ushort4 q[NT];
    float d[NT];
    float2 b0[2];
    float2 b1[2];

    const int ib0 = min(int(sgitg), nb - 1);
    fewrow_load_a<NT>(xq, ib0, q);
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        d[t] = float(as_type<half>(xd[t][ib0 * Q8_BLOCK_US]));
    }
    FOR_UNROLL (short e = 0; e < 2; ++e) {
        b0[e] = yb[e][ib0 * 16];
        b1[e] = yb[e][ib0 * 16 + 2];
    }

    for (int ib = sgitg; ib < nb; ib += NSG) {
        ushort4 qc[NT];
        float dc[NT];
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            qc[t] = q[t] ^ ushort4(0x8080);
            dc[t] = d[t];
        }
        const float2 b0c[2] = {b0[0], b0[1]};
        const float2 b1c[2] = {b1[0], b1[1]};

        // Prefetch the next block while this one computes.
        const int ibn = min(ib + NSG, nb - 1);
        fewrow_load_a<NT>(xq, ibn, q);
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            d[t] = float(as_type<half>(xd[t][ibn * Q8_BLOCK_US]));
        }
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            b0[e] = yb[e][ibn * 16];
            b1[e] = yb[e][ibn * 16 + 2];
        }

        simdgroup_float8x8 mp[NT];
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            mp[t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
        FOR_UNROLL (short s = 0; s < 4; ++s) {
            simdgroup_float8x8 mb;
            const float2 v0 = s >= 2 ? b1c[0] : b0c[0];
            const float2 v1 = s >= 2 ? b1c[1] : b0c[1];
            mb.thread_elements()[0] = s % 2 != 0 ? v0.y : v0.x;
            mb.thread_elements()[1] = s % 2 != 0 ? v1.y : v1.x;
            FOR_UNROLL (short t = 0; t < NT; ++t) {
                const half2 h = fewrow_q8_frag(qc[t], s);
                simdgroup_half8x8 ma;
                ma.thread_elements()[0] = h.x;
                ma.thread_elements()[1] = h.y;
                simdgroup_multiply_accumulate(mp[t], ma, mb, mp[t]);
            }
        }
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            acc[t][0] = fma(dc[t], mp[t].thread_elements()[0], acc[t][0]);
            acc[t][1] = fma(dc[t], mp[t].thread_elements()[1], acc[t][1]);
        }
    }

    // Sum the K slices of the NSG simdgroups and store the 8*NT x 8 tile.
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        red[(sgitg * NT + t) * 64 + 2 * tiisg + 0] = acc[t][0];
        red[(sgitg * NT + t) * 64 + 2 * tiisg + 1] = acc[t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short idx = sgitg * 32 + tiisg; idx < NT * 64; idx += NSG * 32) {
        float sum = 0.0f;
        for (short sg = 0; sg < NSG; ++sg) {
            sum += red[sg * (NT * 64) + idx];
        }
        const short t = idx / 64;
        const short l = (idx % 64) / 2;
        const short e = idx % 2;
        const int r0 = i01 + 8 * t + fewrow_lane_fm(l);
        const int r1 = fewrow_lane_fn(l) + e;
        if (r0 < n_out && r1 < n_cols) {
            y[ulong(r1) * args.n_out + r0] = sum;
        }
    }
}

typedef decltype(kernel_mat_mat_q8_0_fewrow_f32<4, 8>) mat_mat_q8_0_fewrow_t;

#define Q8_FEWROW(NT, NSG)                                                                  \
    template [[host_name("kernel_mat_mat_q8_0_fewrow_nt" #NT "_nsg" #NSG "_f32")]] kernel \
        mat_mat_q8_0_fewrow_t kernel_mat_mat_q8_0_fewrow_f32<NT, NSG>;

Q8_FEWROW(1, 8)
Q8_FEWROW(2, 8)
Q8_FEWROW(4, 8)
Q8_FEWROW(1, 16)
Q8_FEWROW(2, 16)
Q8_FEWROW(4, 16)
Q8_FEWROW(1, 32)
Q8_FEWROW(2, 32)
Q8_FEWROW(4, 32)
