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

    // Pass 2: sum of exp(x - m).
    float s = 0.0f;
    for (uint i = tpitg; i < args.n; i += ntg) {
        s += exp(x[i] - m);
    }
    s = simd_sum(s);
    if (tiisg == 0) shmem[sgitg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    s = simd_sum(s);

    const float inv = 1.0f / s;

    // Pass 3: write normalized.
    for (uint i = tpitg; i < args.n; i += ntg) {
        x[i] = exp(x[i] - m) * inv;
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
kernel void kernel_scatter_offset_f32(
        constant scatter_offset_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    y[args.dst_off + tid] = x[tid];
}

// get_rows: y[r * n_cols + i] = embed[ids[r] * n_cols + i].
// For embedding lookup at decode (n_rows=1) and prefill (n_rows=batch).
struct get_rows_args {
    uint n_rows;
    uint n_cols;
};
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
    y[r * args.n_cols + i] = embed[(uint)row * args.n_cols + i];
}
