// Bench-only Q6_K B=1 mappings for GLM's solo decode band. No runtime selector.
//
// The incumbent `kernel_mat_vec_q6_K_f32_r1` gives one simdgroup a whole row.
// Small or short-K rows then expose the reduction latency, and the row count
// bounds the threadgroup grid. These split-K forms keep one threadgroup per
// output row but let NSG simdgroups take every NSG-th 512-weight pass, meeting
// once in threadgroup memory. The pair forms fuse two projections that read
// the same activation, and the SwiGLU form writes only silu(gate)*up.
//
// Arithmetic is the 16-weight fragment dequantization of
// `kernels/research/glm_batch_q6.metal` (same F32 scale, no half weights) with
// a different summation order, so this is Fast-lineage only.
#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)") for

struct q6solo_gen {
    enum { block_bytes = 210, block_weights = 256, nl = 16 };

    static inline void dequant(device const uchar *xb, short il, thread float4x4 &reg) {
        device const uchar *ql = xb;
        device const uchar *qh = xb + 128;
        device const char *scales = (device const char *)(xb + 192);
        const float d_all = float(as_type<half>(*(device const ushort *)(xb + 208)));

        ql += 64 * (il / 8) + 32 * ((il / 2) & 1) + 16 * (il & 1);
        qh += 32 * (il / 8) + 16 * (il & 1);
        const float sc = float(scales[(il % 2) + 2 * (il / 2)]);
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

        device const ushort *ql16 = (device const ushort *)ql;
        device const ushort *qh16 = (device const ushort *)qh;

        FOR_UNROLL (int i = 0; i < 4; ++i) {
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

struct glm_q6_solo_args {
    uint n_in;
    uint n_out;
    float clamp;
};

template <ushort NSG>
kernel void glm_q6_splitk(
    constant glm_q6_solo_args &a [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[NSG];
    if (row >= a.n_out) return;
    const ulong row_bytes = ulong(a.n_in / 256u) * 210u;
    device const uchar *r = w + ulong(row) * row_bytes;
    float sum = 0.0f;
    for (uint k = (uint(simd) * 32u + uint(lane)) * 16u; k < a.n_in; k += NSG * 512u) {
        float4x4 v;
        q6solo_gen::dequant(r + (k / 256u) * 210u, short((k / 16u) % 16u), v);
        FOR_UNROLL (uint i = 0; i < 16; ++i) sum = fma(v[i / 4][i % 4], x[k + i], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        FOR_UNROLL (uint i = 0; i < NSG; ++i) total += partial[i];
        y[row] = total;
    }
}

typedef decltype(glm_q6_splitk<2>) glm_q6_splitk_t;
#define SPLITK(N) \
    template [[host_name("glm_q6_splitk_s" #N)]] kernel glm_q6_splitk_t glm_q6_splitk<N>;
SPLITK(2) SPLITK(4) SPLITK(8)

// Two same-input rows per threadgroup, K split across simdgroups. `mode` 1
// applies the clamped SwiGLU and writes only y0; otherwise both rows are
// stored.
template <ushort NSG, ushort SWIGLU>
kernel void glm_q6_pair_splitk(
    constant glm_q6_solo_args &a [[buffer(0)]],
    device const uchar *wg [[buffer(1)]],
    device const uchar *wu [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *yg [[buffer(4)]],
    device float *yu [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float pg[NSG];
    threadgroup float pu[NSG];
    if (row >= a.n_out) return;
    const ulong row_bytes = ulong(a.n_in / 256u) * 210u;
    device const uchar *rg = wg + ulong(row) * row_bytes;
    device const uchar *ru = wu + ulong(row) * row_bytes;
    float sg = 0.0f;
    float su = 0.0f;
    for (uint k = (uint(simd) * 32u + uint(lane)) * 16u; k < a.n_in; k += NSG * 512u) {
        float4x4 g;
        float4x4 u;
        const short il = short((k / 16u) % 16u);
        q6solo_gen::dequant(rg + (k / 256u) * 210u, il, g);
        q6solo_gen::dequant(ru + (k / 256u) * 210u, il, u);
        FOR_UNROLL (uint i = 0; i < 16; ++i) {
            const float xv = x[k + i];
            sg = fma(g[i / 4][i % 4], xv, sg);
            su = fma(u[i / 4][i % 4], xv, su);
        }
    }
    sg = simd_sum(sg);
    su = simd_sum(su);
    if (lane == 0) {
        pg[simd] = sg;
        pu[simd] = su;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tg = 0.0f;
        float tu = 0.0f;
        FOR_UNROLL (uint i = 0; i < NSG; ++i) {
            tg += pg[i];
            tu += pu[i];
        }
        if (SWIGLU) {
            const float cg = min(tg, a.clamp);
            const float cu = clamp(tu, -a.clamp, a.clamp);
            yg[row] = (cg / (1.0f + exp(-cg))) * cu;
        } else {
            yg[row] = tg;
            yu[row] = tu;
        }
    }
}

typedef decltype(glm_q6_pair_splitk<2, 0>) glm_q6_pair_splitk_t;
#define PAIR(N, S) \
    template [[host_name("glm_q6_pair_splitk_s" #N "_swiglu" #S)]] kernel \
        glm_q6_pair_splitk_t glm_q6_pair_splitk<N, S>;
PAIR(2, 0) PAIR(4, 0) PAIR(8, 0)
PAIR(2, 1) PAIR(4, 1) PAIR(8, 1)
