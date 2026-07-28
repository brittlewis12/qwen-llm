// Elementwise + small ops needed for the inference forward pass.
// These are the "boring" kernels: trivial math, trivial parallelism,
// single-row activation in / single-row activation out. Batch=1
// single-token decode is the v1 target.
//
// Production form: every kernel takes its `n` from a small Pod arg
// passed via setBytes, not via a buffer. Keeps dispatch overhead minimal.
//
// CPU oracle: the corresponding free fns in crate::forward (silu,
// sigmoid, softplus, etc.).

#include <metal_stdlib>
using namespace metal;

struct n_args {
    uint n;
};

// y[i] = x[i] / (1 + exp(-x[i]))
kernel void kernel_silu_f32(
        constant n_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const float v = x[tid];
    y[tid] = v / (1.0f + exp(-v));
}

// y[i] = 1 / (1 + exp(-x[i]))
kernel void kernel_sigmoid_f32(
        constant n_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    y[tid] = 1.0f / (1.0f + exp(-x[tid]));
}

// y[i] = log(1 + exp(x[i]))   (numerically-stable form)
kernel void kernel_softplus_f32(
        constant n_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const float v = x[tid];
    if (v > 20.0f) {
        y[tid] = v;
    } else if (v < -20.0f) {
        y[tid] = exp(v);
    } else {
        y[tid] = log(1.0f + exp(v));
    }
}

// out[i] = a[i] + b[i]
kernel void kernel_add_f32(
        constant n_args & args [[buffer(0)]],
        device const float * a [[buffer(1)]],
        device const float * b [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    out[tid] = a[tid] + b[tid];
}

// In-place residual: x[i] += y[i]. Used for the two residual adds per
// transformer block. Saves the read-modify-write traffic vs an out-of-
// place add.
kernel void kernel_add_inplace_f32(
        constant n_args & args [[buffer(0)]],
        device       float * x [[buffer(1)]],
        device const float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    x[tid] += y[tid];
}

struct fill_args {
    uint n;
    float value;
};

kernel void kernel_fill_f32(
        constant fill_args & args [[buffer(0)]],
        device       float * y    [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    y[tid] = args.value;
}

struct touch_bytes_args {
    uint n_steps;
    uint stride_bytes;
    ulong n_bytes;
};

kernel void kernel_touch_bytes_f32(
        constant touch_bytes_args & args [[buffer(0)]],
        device const uchar       * src   [[buffer(1)]],
        device       float       * sink  [[buffer(2)]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    float acc = 0.0f;
    for (uint i = tid; i < args.n_steps; i += ntg) {
        const ulong off = min((ulong)i * args.stride_bytes, args.n_bytes - 1);
        acc += (float)src[off];
    }
    sink[tid] = acc;
}

struct roofline_stream_args {
    uint n;
    float alpha;
};

kernel void kernel_roofline_stream_f32(
        constant roofline_stream_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    y[tid] = fma(y[tid], args.alpha, x[tid]);
}

struct roofline_fma_args {
    uint n;
    uint iters;
};

kernel void kernel_roofline_fma_f32(
        constant roofline_fma_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    float v = x[tid];
    for (uint i = 0; i < args.iters; ++i) {
        v = fma(v, 1.0000001f, 0.0000001f);
    }
    y[tid] = v;
}

// out[i] = a[i] * b[i]
kernel void kernel_mul_f32(
        constant n_args & args [[buffer(0)]],
        device const float * a [[buffer(1)]],
        device const float * b [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    out[tid] = a[tid] * b[tid];
}

struct axpy_args {
    uint n;
    float alpha;
};

// accum[i] += alpha * x[i]
kernel void kernel_axpy_f32(
        constant axpy_args & args [[buffer(0)]],
        device const float * x    [[buffer(1)]],
        device       float * accum [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    accum[tid] += args.alpha * x[tid];
}

struct topk_args {
    uint n;
    uint k;
};

// Naive single-thread top-k over one probability vector. n is small (<=256)
// and k is tiny (<=16), so this is adequate for the first MoE routing pass.
kernel void kernel_topk_select_f32(
        constant topk_args & args [[buffer(0)]],
        device const float * probs    [[buffer(1)]],
        device       int   * out_idx  [[buffer(2)]],
        device       float * out_w    [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const uint MAX_K = 16;
    if (args.k == 0 || args.k > MAX_K) return;

    int top_idx[MAX_K];
    float top_val[MAX_K];
    for (uint i = 0; i < args.k; ++i) {
        top_idx[i] = -1;
        top_val[i] = -INFINITY;
    }

    for (uint i = 0; i < args.n; ++i) {
        const float v = probs[i];
        for (uint j = 0; j < args.k; ++j) {
            const bool better = (v > top_val[j]) || (v == top_val[j] && (top_idx[j] < 0 || int(i) < top_idx[j]));
            if (better) {
                for (uint m = args.k - 1; m > j; --m) {
                    top_val[m] = top_val[m - 1];
                    top_idx[m] = top_idx[m - 1];
                }
                top_val[j] = v;
                top_idx[j] = int(i);
                break;
            }
        }
    }

    float sum = 0.0f;
    for (uint i = 0; i < args.k; ++i) {
        if (top_idx[i] >= 0) sum += max(top_val[i], 0.0f);
    }
    sum = max(sum, 6.103515625e-5f);
    for (uint i = 0; i < args.k; ++i) {
        out_idx[i] = max(top_idx[i], 0);
        out_w[i] = top_idx[i] >= 0 ? max(top_val[i], 0.0f) / sum : 0.0f;
    }
}

// SwiGLU FFN inner: ffn[i] = silu(gate[i]) * up[i].
// Two-input fused op, saves a separate silu kernel + intermediate buffer
// in the FFN path. (down(silu(gate(x)) * up(x)) is the full SwiGLU; this
// is the middle product, executed before the final down projection.)
kernel void kernel_silu_mul_f32(
        constant n_args & args [[buffer(0)]],
        device const float * gate [[buffer(1)]],
        device const float * up   [[buffer(2)]],
        device       float * out  [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const float g = gate[tid];
    const float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}

// Gated attention: out[i] = x[i] * sigmoid(gate[i]).
kernel void kernel_sigmoid_mul_f32(
        constant n_args & args [[buffer(0)]],
        device const float * gate [[buffer(1)]],
        device const float * x    [[buffer(2)]],
        device       float * out  [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const float g = gate[tid];
    out[tid] = x[tid] / (1.0f + exp(-g));
}

// Gated attention with a strided gate source (v0.432): reads the gate half
// of the interleaved q_proj output ([head_dim Q, head_dim gate] per head)
// in place, deleting the split_q_gate layout copy. Logical element i maps
// to gate[gate_offset + (i / head_dim) * gate_stride + (i % head_dim)];
// x/out stay compact. Same arithmetic as kernel_sigmoid_mul_f32.
struct sigmoid_mul_gate_strided_args {
    uint n;           // total elements = n_rows * head_dim
    uint head_dim;
    uint gate_stride; // elements between consecutive gate rows
    uint gate_offset; // element offset of gate row 0
};

// Batched hidden-capture copy (v0.432): copy `n_rows` contiguous source
// rows of `row_len` floats into dst rows at `dst_base + row * dst_stride`.
// Replaces a per-row encode_scatter_offset_f32 loop (chunk_p dispatches ->
// 1) in the DFlash prefill hidden-capture tap.
struct copy_rows_dst_strided_args {
    uint n_rows;
    uint row_len;
    uint dst_stride; // elements between consecutive dst rows
    uint dst_base;   // element offset of dst row 0
};

kernel void kernel_copy_rows_dst_strided_f32(
        constant copy_rows_dst_strided_args & args [[buffer(0)]],
        device const float * src [[buffer(1)]], // [n_rows, row_len] contiguous
        device       float * dst [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.n_rows * args.row_len;
    if (tid >= total) return;
    const uint row = tid / args.row_len;
    const uint d = tid - row * args.row_len;
    dst[(ulong)args.dst_base + (ulong)row * args.dst_stride + d] = src[tid];
}

kernel void kernel_sigmoid_mul_gate_strided_f32(
        constant sigmoid_mul_gate_strided_args & args [[buffer(0)]],
        device const float * gate [[buffer(1)]],
        device const float * x    [[buffer(2)]],
        device       float * out  [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const uint row = tid / args.head_dim;
    const uint d = tid - row * args.head_dim;
    const float g = gate[(ulong)args.gate_offset + (ulong)row * args.gate_stride + d];
    out[tid] = x[tid] / (1.0f + exp(-g));
}

// Softmax along the (only) dimension. One threadgroup; up to 1024 threads.
// Used for attention scores (n = context_len, typically ≤ 4-256K). For
// large n we'd want a 2-pass shared-memory reduce; this version does one
// simdgroup-then-shared-mem reduce.
kernel void kernel_softmax_f32(
        constant n_args & args [[buffer(0)]],
        device       float * x  [[buffer(1)]],
        threadgroup  float * shmem [[threadgroup(0)]],
        uint  tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint  ntg   [[threads_per_threadgroup]]) {
    // Pass 1: max.
    float m = -INFINITY;
    for (uint i = tpitg; i < args.n; i += ntg) {
        m = max(m, x[i]);
    }
    m = simd_max(m);
    if (tiisg == 0) shmem[sgitg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : -INFINITY;
    m = simd_max(m);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: sum of exp(x - m).
    float s = 0.0f;
    for (uint i = tpitg; i < args.n; i += ntg) {
        const float e = exp(x[i] - m);
        x[i] = e;
        s += e;
    }
    s = simd_sum(s);
    if (tiisg == 0) shmem[sgitg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    s = simd_sum(s);

    const float inv = 1.0f / s;

    // Pass 3: write normalized.
    for (uint i = tpitg; i < args.n; i += ntg) {
        x[i] *= inv;
    }
}

// L2 norm: y = x / max(||x||, eps). Per-vector; single threadgroup. Used
// for Q/K l2-norm inside the GDN per-head path. CPU oracle:
// crate::forward::l2_norm_in_place. ggml semantics: the "max with eps"
// form, NOT 1/sqrt(sum + eps).
struct l2_norm_args {
    uint n_dim;
    float eps;
};
kernel void kernel_l2_norm_f32(
        constant l2_norm_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        threadgroup  float * shmem [[threadgroup(0)]],
        uint   tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint   ntg   [[threads_per_threadgroup]]) {
    float sumsq = 0.0f;
    for (uint i = tpitg; i < args.n_dim; i += ntg) {
        const float v = x[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    if (tiisg == 0) shmem[sgitg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float scale = 1.0f / max(sqrt(sumsq), args.eps);
    for (uint i = tpitg; i < args.n_dim; i += ntg) {
        y[i] = x[i] * scale;
    }
}

// Batched per-head L2 norm: y[h, :] = x[h, :] / max(||x[h, :]||, eps),
// for h in 0..n_heads. One threadgroup per head. Used in the GDN
// front-end where Q and K are l2-normed per K-head before the
// recurrence kernel. Replaces n_heads separate dispatches with one.
struct l2_norm_batched_args {
    uint n_heads;
    uint head_dim;
    float eps;
};
kernel void kernel_l2_norm_batched_f32(
        constant l2_norm_batched_args & args [[buffer(0)]],
        device const float * x      [[buffer(1)]], // [n_heads, head_dim]
        device       float * y      [[buffer(2)]], // [n_heads, head_dim]
        threadgroup  float * shmem  [[threadgroup(0)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        uint   tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint   ntg   [[threads_per_threadgroup]]) {
    const uint hi = tgpig;
    if (hi >= args.n_heads) return;

    device const float * x_h = x + (ulong)hi * args.head_dim;
    device       float * y_h = y + (ulong)hi * args.head_dim;

    float sumsq = 0.0f;
    for (uint i = tpitg; i < args.head_dim; i += ntg) {
        const float v = x_h[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    if (tiisg == 0) shmem[sgitg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float scale = 1.0f / max(sqrt(sumsq), args.eps);
    for (uint i = tpitg; i < args.head_dim; i += ntg) {
        y_h[i] = x_h[i] * scale;
    }
}

kernel void kernel_l2_norm_pair_batched_f32(
        constant l2_norm_batched_args & args [[buffer(0)]],
        device const float * q_x    [[buffer(1)]],
        device       float * q_y    [[buffer(2)]],
        device const float * k_x    [[buffer(3)]],
        device       float * k_y    [[buffer(4)]],
        threadgroup  float * shmem  [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        uint2  tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint2  ntg   [[threads_per_threadgroup]]) {
    const uint hi = tgpig.x;
    if (hi >= args.n_heads || tgpig.y >= 2) return;

    device const float * x_h;
    device       float * y_h;
    if (tgpig.y == 0) {
        x_h = q_x + (ulong)hi * args.head_dim;
        y_h = q_y + (ulong)hi * args.head_dim;
    } else {
        x_h = k_x + (ulong)hi * args.head_dim;
        y_h = k_y + (ulong)hi * args.head_dim;
    }

    float sumsq = 0.0f;
    for (uint i = tpitg.x; i < args.head_dim; i += ntg.x) {
        const float v = x_h[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    if (tiisg == 0) shmem[sgitg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = (tiisg < (ntg.x + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float scale = 1.0f / max(sqrt(sumsq), args.eps);
    for (uint i = tpitg.x; i < args.head_dim; i += ntg.x) {
        y_h[i] = x_h[i] * scale;
    }
}

kernel void kernel_l2_norm_pair_hd128_r4_f32(
        constant l2_norm_batched_args & args [[buffer(0)]],
        device const float * q_x    [[buffer(1)]],
        device       float * q_y    [[buffer(2)]],
        device const float * k_x    [[buffer(3)]],
        device       float * k_y    [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        uint2  tpitg [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_TG = 4;
    const uint hi = tgpig.x * ROWS_PER_TG + tpitg.y;
    if (hi >= args.n_heads || args.head_dim != 128) return;

    device const float * x = (tgpig.y == 0 ? q_x : k_x) + (ulong)hi * 128;
    device       float * y = (tgpig.y == 0 ? q_y : k_y) + (ulong)hi * 128;

    float sumsq = 0.0f;
    for (uint i = tiisg; i < 128; i += 32) {
        const float v = x[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);

    const float scale = 1.0f / max(sqrt(sumsq), args.eps);
    for (uint i = tiisg; i < 128; i += 32) {
        y[i] = x[i] * scale;
    }
}

// Copy with offset: y[i] = x[src_off + i]  for i in 0..n.
// Used to slice the GDN post-conv qkv buffer into per-role Q/K/V views
// without doing a CPU round-trip. v2 will replace this with kernels
// that read the slice via offset directly.
struct copy_offset_args {
    uint n;
    uint src_off;
};
kernel void kernel_copy_offset_f32(
        constant copy_offset_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    y[tid] = x[args.src_off + tid];
}

// Scatter with destination offset: y[dst_off + i] = x[i]  for i in 0..n.
// Used for KV cache append (write current K/V at slot `position`).
struct scatter_offset_args {
    uint n;
    uint dst_off;
};

struct scatter_kv_vt_args {
    uint n;
    uint dst_off;
    uint base_pos;
    uint kv_dim;
    uint head_dim;
    uint vt_stride;
};

struct scatter_q8_args {
    uint n_blocks;
    uint dst_block_off;
};
kernel void kernel_scatter_offset_f32(
        constant scatter_offset_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    y[args.dst_off + tid] = x[tid];
}

// Same but writes F16 destination from F32 source. Used for F16 KV
// cache append. Precision: x is f32 in the model's residual stream,
// half() conversion is fine for K/V (llama.cpp also defaults to f16
// KV). The half value is what attn_decode_f16 will read back.
kernel void kernel_scatter_offset_f32_to_f16(
        constant scatter_offset_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       half  * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    y[args.dst_off + tid] = (half)x[tid];
}

// Fused K+V scatter: writes both K and V into their respective F16 caches
// in a single dispatch. K and V always share the same `n` (= kv_dim) and
// the same `dst_off` (= position * kv_dim) at decode time, so we amortize
// dispatch overhead across both writes.
//
// Per Jeff & Sanjay (Bulk APIs): "amortize boundary crossings". Saves
// 1 dispatch per attn layer × 16 attn layers = 16 dispatches/token.
kernel void kernel_scatter_offset_f32_to_f16_kv(
        constant scatter_offset_args & args [[buffer(0)]],
        device const float * k_src [[buffer(1)]],
        device const float * v_src [[buffer(2)]],
        device       half  * k_dst [[buffer(3)]],
        device       half  * v_dst [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const uint i = args.dst_off + tid;
    k_dst[i] = (half)k_src[tid];
    v_dst[i] = (half)v_src[tid];
}

// Fused K+V scatter plus transposed-V sidecar write. Used by the experimental
// non-flash matrix attention path so the KQV-ready V_T bank is populated at
// cache-fill time instead of re-transposing the whole prefix in the attention
// body.
kernel void kernel_scatter_offset_f32_to_f16_kv_vt(
        constant scatter_kv_vt_args & args [[buffer(0)]],
        device const float * k_src [[buffer(1)]],
        device const float * v_src [[buffer(2)]],
        device       half  * k_dst [[buffer(3)]],
        device       half  * v_dst [[buffer(4)]],
        device       half  * v_t   [[buffer(5)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const uint i = args.dst_off + tid;
    const half v = (half)v_src[tid];
    k_dst[i] = (half)k_src[tid];
    v_dst[i] = v;

    const uint pos_rel = tid / args.kv_dim;
    const uint chan = tid - pos_rel * args.kv_dim;
    const uint kvh = chan / args.head_dim;
    const uint d = chan - kvh * args.head_dim;
    const uint pos = args.base_pos + pos_rel;
    v_t[((ulong)kvh * args.head_dim + d) * args.vt_stride + pos] = v;
}

// Fused K+V scatter into Q8_0 caches. Exact ggml reference quantization per
// 32-element block:
//   d = amax / 127
//   qs[j] = round(src[j] / d)
// One simdgroup handles one Q8_0 block for K and V together.
kernel void kernel_scatter_offset_f32_to_q8_0_kv(
        constant scatter_q8_args & args [[buffer(0)]],
        device const float * k_src [[buffer(1)]],
        device const float * v_src [[buffer(2)]],
        device uchar * k_dst [[buffer(3)]],
        device uchar * v_dst [[buffer(4)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr ushort QK8_0 = 32;
    constexpr ushort Q8_0_BYTES = 34;
    const uint blk = tgpig;
    if (blk >= args.n_blocks) return;

    const uint src_base = blk * QK8_0;
    const uint dst_blk = args.dst_block_off + blk;
    const float k_val = k_src[src_base + tiisg];
    const float v_val = v_src[src_base + tiisg];

    const float k_abs = fabs(k_val);
    const float v_abs = fabs(v_val);
    const float k_amax = simd_max(k_abs);
    const float v_amax = simd_max(v_abs);

    const float k_d = k_amax / 127.0f;
    const float v_d = v_amax / 127.0f;
    const float k_id = (k_d != 0.0f) ? (1.0f / k_d) : 0.0f;
    const float v_id = (v_d != 0.0f) ? (1.0f / v_d) : 0.0f;

    device uchar * k_blk = k_dst + (ulong)dst_blk * Q8_0_BYTES;
    device uchar * v_blk = v_dst + (ulong)dst_blk * Q8_0_BYTES;
    if (tiisg == 0) {
        ((device half *)k_blk)[0] = (half)k_d;
        ((device half *)v_blk)[0] = (half)v_d;
    }
    threadgroup_barrier(mem_flags::mem_none);

    ((device int8_t *)(k_blk + 2))[tiisg] = (int8_t)round(k_val * k_id);
    ((device int8_t *)(v_blk + 2))[tiisg] = (int8_t)round(v_val * v_id);
}

// get_rows: y[r * n_cols + i] = embed[ids[r] * n_cols + i].
// For embedding lookup at decode (n_rows=1) and prefill (n_rows=batch).
struct get_rows_args {
    uint n_rows;
    uint n_cols;
    uint n_vocab;
};

static inline float bf16_to_float(ushort v) {
    return as_type<float>(((uint)v) << 16);
}

kernel void kernel_get_rows_f32(
        constant get_rows_args & args [[buffer(0)]],
        device const float * embed [[buffer(1)]],
        device const int   * ids   [[buffer(2)]],
        device       float * y     [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint r = gid.y;
    const uint i = gid.x;
    if (r >= args.n_rows || i >= args.n_cols) return;
    const int row = ids[r];
    const ulong out_index = (ulong)r * args.n_cols + i;
    if ((uint)row >= args.n_vocab) {
        y[out_index] = 0.0f;
        return;
    }
    y[out_index] = embed[(ulong)(uint)row * args.n_cols + i];
}

kernel void kernel_get_rows_f16(
        constant get_rows_args & args [[buffer(0)]],
        device const half  * embed [[buffer(1)]],
        device const int   * ids   [[buffer(2)]],
        device       float * y     [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint r = gid.y;
    const uint i = gid.x;
    if (r >= args.n_rows || i >= args.n_cols) return;
    const int row = ids[r];
    const ulong out_index = (ulong)r * args.n_cols + i;
    if ((uint)row >= args.n_vocab) {
        y[out_index] = 0.0f;
        return;
    }
    y[out_index] = float(embed[(ulong)(uint)row * args.n_cols + i]);
}

kernel void kernel_get_rows_bf16(
        constant get_rows_args & args [[buffer(0)]],
        device const ushort * embed [[buffer(1)]],
        device const int    * ids   [[buffer(2)]],
        device       float  * y     [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint r = gid.y;
    const uint i = gid.x;
    if (r >= args.n_rows || i >= args.n_cols) return;
    const int row = ids[r];
    const ulong out_index = (ulong)r * args.n_cols + i;
    if ((uint)row >= args.n_vocab) {
        y[out_index] = 0.0f;
        return;
    }
    y[out_index] = bf16_to_float(embed[(ulong)(uint)row * args.n_cols + i]);
}

static inline void get_rows_q4_K_scale_min(
        uint subblock,
        device const uchar * scales,
        thread uchar & scale,
        thread uchar & min_value) {
    if (subblock < 4) {
        scale = scales[subblock] & 63;
        min_value = scales[subblock + 4] & 63;
    } else {
        scale = (scales[subblock + 4] & 0x0F) |
                ((scales[subblock - 4] >> 6) << 4);
        min_value = (scales[subblock + 4] >> 4) |
                    ((scales[subblock] >> 6) << 4);
    }
}

kernel void kernel_get_rows_q4_K_f32(
        constant get_rows_args & args [[buffer(0)]],
        device const uchar * embed [[buffer(1)]],
        device const int   * ids   [[buffer(2)]],
        device       float * y     [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint r = gid.y;
    const uint i = gid.x;
    if (r >= args.n_rows || i >= args.n_cols) return;

    constexpr ulong Q4_K_BYTES = 144;
    constexpr uint QK_K = 256;
    const int row_i = ids[r];
    const ulong out_index = (ulong)r * args.n_cols + i;
    if ((uint)row_i >= args.n_vocab) {
        y[out_index] = 0.0f;
        return;
    }
    const uint row = (uint)row_i;
    const ulong blocks_per_row = (ulong)args.n_cols / QK_K;
    const ulong block_index = (ulong)row * blocks_per_row + i / QK_K;
    device const uchar * block = embed + block_index * Q4_K_BYTES;
    const uint in_block = i % QK_K;
    const uint subblock = in_block / 32;
    const uint quant_index = in_block % 32;
    device const uchar * scales = block + 4;
    device const uchar * qs = block + 16;

    uchar scale_u;
    uchar min_u;
    get_rows_q4_K_scale_min(subblock, scales, scale_u, min_u);
    const uchar packed = qs[(subblock / 2) * 32 + quant_index];
    const uchar quant = (subblock & 1) ? (packed >> 4) : (packed & 0x0F);
    const float dl = float(((device const half *)block)[0]) * float(scale_u);
    const float ml = float(((device const half *)block)[1]) * float(min_u);
    y[out_index] = dl * float(quant) - ml;
}

kernel void kernel_get_rows_q8_0_f32(
        constant get_rows_args & args [[buffer(0)]],
        device const uchar * embed [[buffer(1)]],
        device const int   * ids   [[buffer(2)]],
        device       float * y     [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint r = gid.y;
    const uint i = gid.x;
    if (r >= args.n_rows || i >= args.n_cols) return;

    constexpr ulong Q8_0_BYTES = 34;
    constexpr uint QK8_0 = 32;
    const int row_i = ids[r];
    const ulong out_index = (ulong)r * args.n_cols + i;
    if ((uint)row_i >= args.n_vocab) {
        y[out_index] = 0.0f;
        return;
    }
    const uint row = (uint)row_i;
    const ulong blocks_per_row = (ulong)args.n_cols / QK8_0;
    const ulong block_index = (ulong)row * blocks_per_row + i / QK8_0;
    device const uchar * block = embed + block_index * Q8_0_BYTES;
    const uint quant_index = i % QK8_0;
    const float d = float(((device const half *)block)[0]);
    const int8_t quant = ((device const int8_t *)(block + 2))[quant_index];
    y[out_index] = d * float(quant);
}

// GDN α-chain fusion (replaces 3 dispatches: add_inplace + softplus + mul).
//
//   out[i] = softplus(a[i] + dt_bias[i]) * a_log[i]
//
// Per the GDN forward pass:
//   gdn_alpha = softplus(gdn_a + dt_bias) * a_log
//
// Saves 2 dispatches per layer × 32 GDN layers = 64 dispatches/token.
// Per the v0.25 intra-profiler the alpha chain measured 0.043 ms/layer
// (4 dispatches), so this should shave ~1.5–2.5 ms/token. Keeps the same
// numerical-stability branch (v > 20: identity; v < -20: exp(v); else log(1+exp)).
//
// Inputs: a (mutable, will be added with dt_bias and softplus'd in-place),
// dt_bias (per-channel), a_log (per-channel scale).
// Output: written to `out` (out and a may alias).
kernel void kernel_gdn_alpha_chain_f32(
        constant n_args & args   [[buffer(0)]],
        device const float * a       [[buffer(1)]],
        device const float * dt_bias [[buffer(2)]],
        device const float * a_log   [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const float v = a[tid] + dt_bias[tid];
    float sp;
    if (v > 20.0f) {
        sp = v;
    } else if (v < -20.0f) {
        sp = exp(v);
    } else {
        sp = log(1.0f + exp(v));
    }
    out[tid] = sp * a_log[tid];
}

// Same as kernel_gdn_alpha_chain_f32, but stores exp(g) directly:
//   out[i] = exp(softplus(a[i] + dt_bias[i]) * a_log[i])
// GDN step uses this per-head decay for every state row, so precomputing it
// avoids repeating the same exp() head_dim times per head.
kernel void kernel_gdn_decay_chain_f32(
        constant n_args & args   [[buffer(0)]],
        device const float * a       [[buffer(1)]],
        device const float * dt_bias [[buffer(2)]],
        device const float * a_log   [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const float v = a[tid] + dt_bias[tid];
    float sp;
    if (v > 20.0f) {
        sp = v;
    } else if (v < -20.0f) {
        sp = exp(v);
    } else {
        sp = log(1.0f + exp(v));
    }
    out[tid] = exp(sp * a_log[tid]);
}

// Batched GDN α-chain (v0.73a layer-major batching).
//
//   out[r, c] = softplus(a[r, c] + dt_bias[c]) * a_log[c]
//
// Same numerics as kernel_gdn_alpha_chain_f32 (bit-identical when N=1)
// but broadcasts dt_bias and a_log across N rows so it can run once per
// GDN layer instead of N times. dt_bias and a_log are each `[n_v]`;
// a and out are each `[N, n_v]` row-major.
//
// Dispatch: 1D grid of size N * n_v; each thread handles one (r, c)
// element. r = tid / n_v ; c = tid % n_v.
struct gdn_alpha_chain_batched_args {
    uint n;       // total elements = N * n_v
    uint n_cols;  // n_v
};

kernel void kernel_gdn_alpha_chain_batched_f32(
        constant gdn_alpha_chain_batched_args & args [[buffer(0)]],
        device const float * a       [[buffer(1)]],
        device const float * dt_bias [[buffer(2)]],
        device const float * a_log   [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const uint c = tid % args.n_cols;
    const float v = a[tid] + dt_bias[c];
    float sp;
    if (v > 20.0f) {
        sp = v;
    } else if (v < -20.0f) {
        sp = exp(v);
    } else {
        sp = log(1.0f + exp(v));
    }
    out[tid] = sp * a_log[c];
}

kernel void kernel_gdn_decay_chain_batched_f32(
        constant gdn_alpha_chain_batched_args & args [[buffer(0)]],
        device const float * a       [[buffer(1)]],
        device const float * dt_bias [[buffer(2)]],
        device const float * a_log   [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const uint c = tid % args.n_cols;
    const float v = a[tid] + dt_bias[c];
    float sp;
    if (v > 20.0f) {
        sp = v;
    } else if (v < -20.0f) {
        sp = exp(v);
    } else {
        sp = log(1.0f + exp(v));
    }
    out[tid] = exp(sp * a_log[c]);
}

// =============================================================================
// kernel_argmax_f32 — GPU-side argmax for one row of length `n`.
//
// One threadgroup processes one row. Up to 1024 threads per threadgroup,
// reduced via simd_max + a small shmem step. Tie policy: LOWEST INDEX wins
// (matches numpy / torch semantics; tested explicitly in
// `metal::tests::argmax_matches_cpu_with_tie_to_lowest_index`).
//
// Used by H5.3a `packed_verify` to produce `verify_argmax: [N] i32`
// without a `[N, V]` CPU readback (saves 15.9 MB per outer step at
// V=248320, N=16). For packed-N argmax over `[N, V] -> [N] i32`, dispatch
// with `grid.width = N`; each TG handles row `tgpig.x`.
//
// Layout:
//   x         [n_rows, n] (row-major)        — input logits
//   out_idx   [n_rows]                       — output argmax index per row (i32)
//   stride_x  = n (row stride in elements)
//
// Reduction strategy:
//   Pass 1 — each lane scans strided slice; tracks (max_val, min_idx_at_max).
//   Pass 2 — simdgroup-wide reduce: pick max over lanes, breaking ties to
//            lowest index. Use simd_max for the value, then simd_min over
//            lanes whose value matches lane_max.
//   Pass 3 — across simdgroups via shmem; one thread writes out_idx.
//
// Edge-case contract (per codex H5.3a review + tests 9 / 10 in
// `argmax_matches_cpu_with_tie_to_lowest_index`):
//   * All-zero row     → returns 0 (every position ties; lowest idx wins).
//   * All −INFINITY row → returns 0 (every position ties at −inf;
//                          lowest idx wins, NOT -1).
//   * All-NaN row       → returns -1 (UINT_MAX cast). IEEE comparisons
//                          fail for any NaN operand, so no lane ever
//                          claims a candidate idx; simd_min(UINT_MAX) =
//                          UINT_MAX. Production lm_head should never
//                          produce a NaN row; if it does, the -1
//                          surfaces it.
//
// Bound: this kernel assumes `tg_threads ≤ 1024` (Metal hardware spec
// → n_simdgroups ≤ 32). The cross-simdgroup reduce in Pass 3 only runs
// in simdgroup 0; lanes 0..n_simdgroups-1 read shmem entries. If
// n_simdgroups > 32 we'd silently lose data — but Metal will reject
// the dispatch first.
// =============================================================================
struct argmax_args {
    uint n;          // length of each row
    uint stride_x;   // row stride in elements (= n unless caller asks for padding)
};

kernel void kernel_argmax_f32(
        constant argmax_args & args [[buffer(0)]],
        device const float   * x        [[buffer(1)]],
        device       int     * out_idx  [[buffer(2)]],
        threadgroup  float   * sh_val   [[threadgroup(0)]],
        threadgroup  uint    * sh_idx   [[threadgroup(1)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        uint   tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint   ntg   [[threads_per_threadgroup]]) {
    const uint row = tgpig;
    device const float * x_row = x + (ulong)row * args.stride_x;

    // Pass 1: per-lane scan, tracking (best_val, best_idx). Tie-break to
    // lower idx within a single lane's traversal.
    float best_val = -INFINITY;
    uint  best_idx = UINT_MAX;
    for (uint i = tpitg; i < args.n; i += ntg) {
        const float v = x_row[i];
        if (v > best_val || (v == best_val && i < best_idx)) {
            best_val = v;
            best_idx = i;
        }
    }

    // Pass 2: simdgroup reduce. simd_max gives us the value; then we
    // broadcast each lane's idx and pick the smallest idx among lanes
    // whose val equals the max.
    const float lane_max = simd_max(best_val);
    // Lanes whose val < lane_max get idx = UINT_MAX (will lose the
    // simd_min tiebreak); lanes that match get their actual idx.
    uint lane_idx = (best_val == lane_max) ? best_idx : UINT_MAX;
    const uint sg_idx = simd_min(lane_idx);

    if (tiisg == 0) {
        sh_val[sgitg] = lane_max;
        sh_idx[sgitg] = sg_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: cross-simdgroup reduce (simdgroup 0 only).
    const uint n_simdgroups = (ntg + 31) / 32;
    if (sgitg == 0) {
        const float sgv = (tiisg < n_simdgroups) ? sh_val[tiisg] : -INFINITY;
        const uint  sgi = (tiisg < n_simdgroups) ? sh_idx[tiisg] : UINT_MAX;
        const float global_max = simd_max(sgv);
        const uint  match_idx = (sgv == global_max) ? sgi : UINT_MAX;
        const uint  global_idx = simd_min(match_idx);
        if (tiisg == 0) {
            out_idx[row] = (int)global_idx;
        }
    }
}

// Production-greedy argmax matching Rust f32::total_cmp for every non-NaN
// bit pattern. Any NaN wins over a token result and is encoded as ~token_id,
// allowing the host to report the lowest offending index without reading the
// logits row. The input is intentionally loaded as uint so -ffast-math cannot
// weaken NaN detection or signed-zero ordering.
struct greedy_argmax_args {
    uint n;
    uint stride_x;
    uint n_simdgroups;
};

kernel void kernel_argmax_f32_greedy(
        constant greedy_argmax_args & args [[buffer(0)]],
        device const uint          * x_bits   [[buffer(1)]],
        device       int           * out_idx  [[buffer(2)]],
        threadgroup  uint          * sh_key   [[threadgroup(0)]],
        threadgroup  uint          * sh_idx   [[threadgroup(1)]],
        threadgroup  uint          * sh_nan   [[threadgroup(2)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        uint   tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint   ntg   [[threads_per_threadgroup]]) {
    const uint row = tgpig;
    device const uint * row_bits = x_bits + (ulong)row * args.stride_x;

    uint best_key = 0;
    uint best_idx = 0;
    uint first_nan = UINT_MAX;
    for (uint i = tpitg; i < args.n; i += ntg) {
        const uint bits = row_bits[i];
        const uint abs_bits = bits & 0x7fffffffu;
        if (abs_bits > 0x7f800000u) {
            first_nan = min(first_nan, i);
            continue;
        }

        const uint key = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
        if (key > best_key || (key == best_key && i > best_idx)) {
            best_key = key;
            best_idx = i;
        }
    }

    const uint lane_key = simd_max(best_key);
    const uint lane_idx = simd_max(best_key == lane_key ? best_idx : 0u);
    const uint lane_nan = simd_min(first_nan);
    if (tiisg == 0) {
        sh_key[sgitg] = lane_key;
        sh_idx[sgitg] = lane_idx;
        sh_nan[sgitg] = lane_nan;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        const bool active = tiisg < args.n_simdgroups;
        const uint group_key = active ? sh_key[tiisg] : 0u;
        const uint group_idx = active ? sh_idx[tiisg] : 0u;
        const uint group_nan = active ? sh_nan[tiisg] : UINT_MAX;
        const uint global_key = simd_max(group_key);
        const uint global_idx = simd_max(group_key == global_key ? group_idx : 0u);
        const uint global_nan = simd_min(group_nan);
        if (tiisg == 0) {
            out_idx[row] = global_nan == UINT_MAX ? int(global_idx) : ~int(global_nan);
        }
    }
}
