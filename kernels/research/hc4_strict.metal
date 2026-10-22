#include <metal_stdlib>
using namespace metal;

struct hc4_strict_args {
    uint n_in;
    uint n_out;
    uint n_query;
};

// Test-only ordinary packed HC projection: one query lane, four scalar outputs.
// Preserve generic F32's ascending K and xv * weight expression, not a dot/FMA
// reduction or split accumulators. No shared memory or threadgroup barriers.
kernel void kernel_mat_mat_f32_hc4p32_strict_test(
        constant hc4_strict_args & args [[buffer(0)]],
        device const float * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint q = tgpig.y * 32u + tid;
    if (q >= args.n_query) return;

    device const float * x_row = x + (ulong)q * args.n_in;
    device const float * w0 = weight;
    device const float * w1 = weight + (ulong)args.n_in;
    device const float * w2 = weight + 2ul * args.n_in;
    device const float * w3 = weight + 3ul * args.n_in;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint k = 0; k < args.n_in; ++k) {
        const float xv = x_row[k];
        acc0 += xv * w0[k];
        acc1 += xv * w1[k];
        acc2 += xv * w2[k];
        acc3 += xv * w3[k];
    }
    const ulong base = (ulong)q * args.n_out;
    y[base + 0u] = acc0;
    y[base + 1u] = acc1;
    y[base + 2u] = acc2;
    y[base + 3u] = acc3;
}
