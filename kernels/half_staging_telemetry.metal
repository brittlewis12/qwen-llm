// Debug-only activation magnitude telemetry for half-staging sites.
//
// The production kernels are built with -ffast-math, so isfinite/isnan are not
// reliable. This kernel compares exponent bits and keeps the largest absolute
// value per site with a relaxed device atomic. It is only dispatched when
// QWEN_HALF_STAGING_TELEMETRY is truthy; the production entry points never
// bind it.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_half_staging_amax(
        constant uint & n [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device atomic_uint * out [[buffer(2)]],
        constant uint & site [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= n) return;
    const uint bits = as_type<uint>(x[tid]) & 0x7fffffffu;
    // Exponent all ones: inf or NaN. Do not let it dominate the magnitude
    // record; the guard is about finite half-range values.
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        atomic_fetch_max_explicit(&out[site], bits, memory_order_relaxed);
    }
}
