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
