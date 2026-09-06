// Bench-only roofline probes (touch-bytes, streaming FMA). Compiled into the
// research metallib, not the product one.

#include <metal_stdlib>
using namespace metal;

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
