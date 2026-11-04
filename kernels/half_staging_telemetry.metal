// Debug-only activation magnitude telemetry for half-staging sites.
//
// The production kernels are built with -ffast-math, so isfinite/isnan are not
// reliable. This kernel compares exponent bits and keeps the largest absolute
// value per site with a relaxed device atomic. It is only dispatched when
// QWEN_HALF_STAGING_TELEMETRY is truthy; the production entry points never
// bind it.
//
// The grid is capped by the host and each thread walks a strided slice, then
// the simdgroup/threadgroup reduces to one atomic per 256-thread group. A
// one-thread-per-element version serialized on a single atomic at every
// down-projection site and was unusably slow on a long prefill.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_half_staging_amax(
        constant uint & n [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device atomic_uint * out [[buffer(2)]],
        constant uint & site [[buffer(3)]],
        constant uint & stride [[buffer(4)]],
        uint tid [[thread_position_in_grid]],
        uint lid [[thread_position_in_threadgroup]],
        uint lsize [[threads_per_threadgroup]]) {
    uint bits = 0;
    for (uint i = tid; i < n; i += stride) {
        const uint b = as_type<uint>(x[i]) & 0x7fffffffu;
        // Exponent all ones: inf or NaN. Do not let it dominate the magnitude
        // record; the guard is about finite half-range values.
        if ((b & 0x7f800000u) != 0x7f800000u) {
            bits = max(bits, b);
        }
    }
    bits = simd_max(bits);
    threadgroup uint partial[32];
    if (lid % 32u == 0u) {
        partial[lid / 32u] = bits;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        uint m = 0;
        const uint groups = lsize / 32u;
        for (uint i = 0; i < groups; ++i) {
            m = max(m, partial[i]);
        }
        atomic_fetch_max_explicit(&out[site], m, memory_order_relaxed);
    }
}
