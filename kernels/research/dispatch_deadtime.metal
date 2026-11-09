#include <metal_stdlib>
using namespace metal;

// Bench-only kernels for the dependent-dispatch dead-time probe. Both kernels
// do a bounded amount of ordinary device-memory traffic on a caller-selected
// 1D grid, so the probe can separate scheduler/encoder cost from kernel work.
struct dispatch_deadtime_args {
    uint total_threads;
    uint words;
    uint dispatch_index;
};

// Every dispatch writes the same region, increments a shared counter word and
// reads back the first word of its payload, so consecutive dispatches in a
// serial encoder have real read-after-write dependencies. `words == 0` leaves
// only the counter increment.
kernel void kernel_dispatch_deadtime_dependent(
        device uint *touch [[buffer(0)]],
        constant dispatch_deadtime_args &args [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid == 0) {
        touch[0] = touch[0] + 1u;
    }
    if (gid >= args.total_threads) {
        return;
    }
    const uint base = 1u + gid * args.words;
    const uint previous = args.words > 0u ? touch[base] : 0u;
    for (uint i = 0; i < args.words; ++i) {
        touch[base + i] = (previous + gid) ^ (args.dispatch_index << 8) ^ i;
    }
}

// Disjoint output regions make consecutive dispatches independent, so they can
// share one concurrent compute encoder. The traffic per dispatch matches the
// dependent kernel for the same shape.
kernel void kernel_dispatch_deadtime_independent(
        device uint *touch [[buffer(0)]],
        constant dispatch_deadtime_args &args [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.total_threads) {
        return;
    }
    const uint base =
        1u + (args.dispatch_index * args.total_threads + gid) * args.words;
    for (uint i = 0; i < args.words; ++i) {
        touch[base + i] = gid ^ (args.dispatch_index << 8) ^ i;
    }
}
