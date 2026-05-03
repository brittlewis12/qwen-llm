// SSM conv1d step (single-token decode).
//
// Per-channel depthwise convolution of width K (=4 for Qwen3.5/3.6),
// followed by SiLU. The conv buffer holds the last K-1 timesteps of
// qkv-mixed values per channel; we append the current qkv as the K-th
// time-row, dot with the per-channel kernel, slide the buffer, and
// emit the SiLU'd convolution output.
//
// Layouts:
//   qkv_now      [conv_dim]                     — this step's q+k+v projections
//   conv_buf     [K-1, conv_dim] row-major      — past K-1 timesteps
//   conv_w       [conv_dim, K]                  — channel-blocked weights
//   out          [conv_dim]                     — silu(conv result)
//
// Math per channel c:
//   t0..t_{K-2} = conv_buf[t, c]
//   t_{K-1}     = qkv_now[c]
//   out[c] = silu(sum_{k=0..K-1} conv_w[c, k] * t_k)
//
// Side effect: conv_buf is shifted in time — its K-2-th row becomes its
// (K-3)-th, ..., its 0-th row is dropped, and qkv_now is appended as the
// new K-2-th row. So after this step, conv_buf again holds the last K-1
// timesteps including the one we just consumed.
//
// One thread per channel; threadgroups of 1024 threads each.
//
// Hardcoded for K=4 to keep the unrolled inner loop fast. When the model
// family changes this we'll need a templated K.

#include <metal_stdlib>
using namespace metal;

constant constexpr int CONV_K = 4;

struct ssm_conv_args {
    uint conv_dim;
};

kernel void kernel_ssm_conv_silu_f32(
        constant ssm_conv_args & args      [[buffer(0)]],
        device const float     * qkv_now   [[buffer(1)]], // [conv_dim]
        device       float     * conv_buf  [[buffer(2)]], // [K-1, conv_dim] mutated
        device const float     * conv_w    [[buffer(3)]], // [conv_dim, K]
        device       float     * out       [[buffer(4)]], // [conv_dim]
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.conv_dim) return;

    const uint c = tid;
    const uint cd = args.conv_dim;

    // Load past K-1 timesteps for this channel.
    float t[CONV_K];
    for (int i = 0; i < CONV_K - 1; ++i) {
        t[i] = conv_buf[i * cd + c];
    }
    t[CONV_K - 1] = qkv_now[c];

    // Per-channel weight row: conv_w[c, 0..K).
    device const float * w = conv_w + (ulong)c * CONV_K;

    // Conv + SiLU.
    float s = 0.0f;
    for (int k = 0; k < CONV_K; ++k) {
        s += w[k] * t[k];
    }
    out[c] = s / (1.0f + exp(-s));

    // Slide the buffer: drop t[0], shift t[1..K-1] into rows 0..K-2.
    // I.e. new conv_buf[i, c] = t[i+1] for i in 0..K-1.
    for (int i = 0; i < CONV_K - 1; ++i) {
        conv_buf[i * cd + c] = t[i + 1];
    }
}

// RMSNormGated: post-GDN-recurrence norm with a gating output.
//
//   y[hi, dv] = norm(o[hi, dv]) * silu(z[hi, dv])
//
// where `norm` is RMSNorm along `dv` per head with per-channel weight,
// and `z` is the (already-projected) gate input. Used immediately after
// the GDN recurrence, before the out_proj mat-vec.
//
// Layout:
//   o     [n_heads, head_dim]
//   weight [head_dim]            — per-channel norm weight (broadcast across heads)
//   z     [n_heads, head_dim]
//   y     [n_heads, head_dim]
//
// One threadgroup per head; reduces sum-of-squares across head_dim.

struct rmsnorm_gated_args {
    uint n_heads;
    uint head_dim;
    float eps;
};

kernel void kernel_rmsnorm_gated_f32(
        constant rmsnorm_gated_args & args [[buffer(0)]],
        device const float * o      [[buffer(1)]],  // [n_heads, head_dim]
        device const float * weight [[buffer(2)]],  // [head_dim]
        device const float * z      [[buffer(3)]],  // [n_heads, head_dim]
        device       float * y      [[buffer(4)]],  // [n_heads, head_dim]
        threadgroup  float * shmem  [[threadgroup(0)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        uint  tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint  ntg   [[threads_per_threadgroup]]) {
    const uint hi = tgpig;
    if (hi >= args.n_heads) return;

    device const float * o_h = o + (ulong)hi * args.head_dim;
    device const float * z_h = z + (ulong)hi * args.head_dim;
    device       float * y_h = y + (ulong)hi * args.head_dim;

    // Pass 1: sum of squares.
    float sumsq = 0.0f;
    for (uint i = tpitg; i < args.head_dim; i += ntg) {
        const float v = o_h[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    if (tiisg == 0) shmem[sgitg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float mean = sumsq / float(args.head_dim);
    const float scale = 1.0f / sqrt(mean + args.eps);

    // Pass 2: y[i] = (o[i] * scale * weight[i]) * silu(z[i])
    for (uint i = tpitg; i < args.head_dim; i += ntg) {
        const float normed = o_h[i] * scale * weight[i];
        const float zi = z_h[i];
        const float silu_z = zi / (1.0f + exp(-zi));
        y_h[i] = normed * silu_z;
    }
}
