// Half-range / subnormal probe (research-only).
//
// 1. One 8x8x8 simdgroup MMA with half operands and an F32 accumulator. The
//    caller fills A with a test value and B with 1.0, so a preserved value
//    yields 8*v and a flushed half-subnormal operand yields 0.
// 2. F32 -> half conversion, so the host can see whether 65520/1e5 become inf
//    or saturate.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_half_range_mma_probe(
        device const half * a [[buffer(0)]],
        device const half * b [[buffer(1)]],
        device float * out [[buffer(2)]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half sa[64];
    threadgroup half sb[64];
    threadgroup float sc[64];
    for (uint i = tid; i < 64; i += 32) {
        sa[i] = a[i];
        sb[i] = b[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 ma;
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_load(ma, sa, 8);
    simdgroup_load(mb, sb, 8);
    simdgroup_multiply_accumulate(mc, ma, mb, mc);
    simdgroup_store(mc, sc, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) out[0] = sc[0];
}

kernel void kernel_f32_to_half_probe(
        constant uint & n [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device half * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid < n) y[tid] = (half)x[tid];
}
