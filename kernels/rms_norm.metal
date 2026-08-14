// RMSNorm fused with element-wise weight multiply.
//
// Lifted in spirit from llama.cpp's `kernel_rms_norm_fuse_impl` template
// at ggml/src/ggml-metal/ggml-metal.metal:2990 (F=2 specialization).
// Same simdgroup-reduce design, simplified to the rank-1 single-row case
// we need for an inference forward pass:
//
//   y[i] = (x[i] / sqrt(mean(x²) + eps)) * weight[i]
//
// One threadgroup per row. Up to 1024 threads/tg for the parallel sum;
// each loads one or more elements based on n_dim.
//
// Args:
//   x:      [n_dim] f32, input (one row)
//   weight: [n_dim] f32, per-channel scale
//   y:      [n_dim] f32, output
//   n_dim:  total elements
//   eps:    additive eps inside the rsqrt

#include <metal_stdlib>
using namespace metal;

struct rms_norm_args {
    uint  n_dim;
    float eps;
};

struct rms_norm_rows_args {
    uint  n_dim;
    uint  row_count;
    float eps;
};

struct ds4_prepare_norm_pair_args {
    uint q_dim;
    uint kv_dim;
    float eps;
};

kernel void kernel_rms_norm_mul_f32(
        constant rms_norm_args & args     [[buffer(0)]],
        device const float     * x        [[buffer(1)]],
        device const float     * weight   [[buffer(2)]],
        device       float     * y        [[buffer(3)]],
        threadgroup float      * shmem    [[threadgroup(0)]],
        uint  tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint  ntg   [[threads_per_threadgroup]]) {
    // ---- pass 1: parallel sum of squares ----
    float sumsq = 0.0f;
    for (uint i = tpitg; i < args.n_dim; i += ntg) {
        const float v = x[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);

    if (tiisg == 0) {
        shmem[sgitg] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces the per-simdgroup partial sums.
    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float mean  = sumsq / float(args.n_dim);
    const float scale = rsqrt(mean + args.eps);

    // ---- pass 2: scale + weight ----
    for (uint i = tpitg; i < args.n_dim; i += ntg) {
        y[i] = (x[i] * scale) * weight[i];
    }
}

kernel void kernel_rms_norm_mul_rows_f32(
        constant rms_norm_rows_args & args [[buffer(0)]],
        device const float * x             [[buffer(1)]],
        device const float * weight        [[buffer(2)]],
        device       float * y             [[buffer(3)]],
        threadgroup float * shmem           [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.row_count) return;
    const ulong base = ulong(row) * args.n_dim;
    float sumsq = 0.0f;
    for (uint i = tpitg; i < args.n_dim; i += ntg) {
        const float v = x[base + i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);

    if (tiisg == 0) {
        shmem[sgitg] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float mean = sumsq / float(args.n_dim);
    const float scale = rsqrt(mean + args.eps);
    for (uint i = tpitg; i < args.n_dim; i += ntg) {
        y[base + i] = (x[base + i] * scale) * weight[i];
    }
}

kernel void kernel_ds4_prepare_norm_pair_f32(
        constant ds4_prepare_norm_pair_args & args [[buffer(0)]],
        device const float * q_x [[buffer(1)]],
        device const float * q_weight [[buffer(2)]],
        device float * q_y [[buffer(3)]],
        device const float * kv_x [[buffer(4)]],
        device const float * kv_weight [[buffer(5)]],
        device float * kv_y [[buffer(6)]],
        threadgroup float * shmem [[threadgroup(0)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    const uint n_dim = tgpig == 0u ? args.q_dim : args.kv_dim;
    device const float * x = tgpig == 0u ? q_x : kv_x;
    device const float * weight = tgpig == 0u ? q_weight : kv_weight;
    device float * y = tgpig == 0u ? q_y : kv_y;

    float sumsq = 0.0f;
    for (uint i = tpitg; i < n_dim; i += ntg) {
        const float v = x[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    if (tiisg == 0) {
        shmem[sgitg] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);
    const float mean = sumsq / float(n_dim);
    const float scale = rsqrt(mean + args.eps);
    for (uint i = tpitg; i < n_dim; i += ntg) {
        y[i] = (x[i] * scale) * weight[i];
    }
}

// Fused residual add + RMSNorm:
//
//   x[i] += residual[i]
//   y[i]  = (x[i] / sqrt(mean(x²) + eps)) * weight[i]
//
// This replaces the decode post-mixer pair `add_inplace(x, mixer_out)` followed
// by `rms_norm_mul(x, post_attn_norm, h)`, eliminating one dispatch and one
// extra residual-stream read. The reduction order intentionally mirrors
// kernel_rms_norm_mul_f32: each lane visits the same strided indices and uses
// the same simdgroup + threadgroup reduction tree, but it accumulates the
// post-add value before writing it back to x.
kernel void kernel_residual_rms_norm_mul_f32(
        constant rms_norm_args & args     [[buffer(0)]],
        device       float     * x        [[buffer(1)]],
        device const float     * residual [[buffer(2)]],
        device const float     * weight   [[buffer(3)]],
        device       float     * y        [[buffer(4)]],
        threadgroup float      * shmem    [[threadgroup(0)]],
        uint  tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint  ntg   [[threads_per_threadgroup]]) {
    float sumsq = 0.0f;
    for (uint i = tpitg; i < args.n_dim; i += ntg) {
        const float v = x[i] + residual[i];
        x[i] = v;
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);

    if (tiisg == 0) {
        shmem[sgitg] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float mean  = sumsq / float(args.n_dim);
    const float scale = rsqrt(mean + args.eps);

    for (uint i = tpitg; i < args.n_dim; i += ntg) {
        y[i] = (x[i] * scale) * weight[i];
    }
}
