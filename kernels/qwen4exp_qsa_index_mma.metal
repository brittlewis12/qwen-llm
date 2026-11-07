#include <metal_stdlib>
using namespace metal;

struct fn_qsa_index_mma_args { uint k; uint n; uint m; };

// Full-F32 operands/accumulation for persistent QSA selector projections.
// Exactly widen BF16 weights instead of narrowing the input activations.
// Host qualifies K2560, N128/512 and M512..2048; partial M tiles are safe.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_qwen4exp_qsa_index_bf16_f32_mma(
    constant fn_qsa_index_mma_args &a [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    threadgroup float *panel [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint m0 = group.x * 32, n0 = group.y * 8;
    // Shift the final partial SIMD tile into the valid query range. Only
    // its original rows are scattered below, so tiles never overlap writes.
    const uint natural_m = m0 + sg * 8;
    const uint load_m = min(natural_m, a.m - 8);
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint k0 = 0; k0 < a.k; k0 += 128) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8 * 128; i += 128) {
            const uint row = i / 128, col = i % 128;
            panel[i] = n0 + row < a.n ? float(w[ulong(n0 + row) * a.k + k0 + col]) : 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ki = 0; ki < 128; ki += 8) {
            simdgroup_float8x8 q, b;
            simdgroup_load(q, x + ulong(load_m) * a.k + k0 + ki, a.k);
            simdgroup_load(b, panel + ki, 128, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, q, b, acc);
        }
    }
    // The last weight panel is dead; reuse its first 256 floats for output.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc, panel + sg * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 4 * 64; i += 128) {
        const uint row = i / 8, col = i % 8;
        const uint tile_m = m0 + (row / 8) * 8;
        const uint query = min(tile_m, a.m - 8) + row % 8;
        if (query >= tile_m && query < a.m && n0 + col < a.n)
            y[ulong(query) * a.n + n0 + col] = panel[i];
    }
}
