#include <metal_stdlib>
using namespace metal;

// Bench-only device-scope phase handoff. Every threadgroup dequeues one ticket;
// ticket order is the dependency order, so a waiter always has a predecessor
// that has already been dequeued. This is not a grid-wide spin barrier.
kernel void kernel_hc_phase_sync(
        coherent(device) device atomic_uint *next_task [[buffer(0)]],
        coherent(device) device atomic_uint *ready [[buffer(1)]],
        coherent(device) device uint *payload [[buffer(2)]],
        coherent(device) device uint *owners [[buffer(3)]],
        constant uint &task_count [[buffer(4)]],
        constant uint &seed [[buffer(5)]],
        uint lane [[thread_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    if (lane != 0u) return;

    const uint task = atomic_fetch_add_explicit(
        next_task, 1u, memory_order_relaxed);
    if (task >= task_count) return;

    uint input = seed;
    if (task != 0u) {
        while (atomic_load_explicit(
                   &ready[task - 1u], memory_order_relaxed) == 0u) {
        }
        atomic_thread_fence(
            mem_flags::mem_device,
            memory_order_seq_cst,
            thread_scope::thread_scope_device);
        input = payload[task - 1u];
    }

    // Two dependent integer rounds keep a real payload live across the
    // device-scope handoff while staying below the measured dispatch workload.
    input ^= input >> 13;
    input = input * 1664525u + 1013904223u;
    input ^= input >> 13;
    input = input * 1664525u + 1013904223u;
    payload[task] = input;
    owners[task] = group;
    atomic_thread_fence(
        mem_flags::mem_device,
        memory_order_seq_cst,
        thread_scope::thread_scope_device);
    atomic_store_explicit(
        &ready[task], 1u, memory_order_relaxed);
}

// Contention kernel: many independent groups keep both M5 Ultra GPU slices
// occupied while the phase chain transfers data. The output is retained so
// the arithmetic cannot be discarded.
kernel void kernel_hc_phase_contention(
        device uint *output [[buffer(0)]],
        constant uint &iterations [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    uint value = gid ^ 0xa511e9b3u;
    for (uint i = 0u; i < iterations; ++i) {
        value ^= value >> 13;
        value = value * 1664525u + 1013904223u;
    }
    output[gid] = value;
}
