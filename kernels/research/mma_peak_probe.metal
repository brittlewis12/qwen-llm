// Raw simdgroup MMA ceiling probe (research-only). Each simdgroup runs
// ITERS iterations of ACC independent 8x8x8 half-operand / F32-accumulate
// products on register-resident tiles loaded once from threadgroup memory.
// No device traffic inside the loop; the result is written once so the
// compiler keeps the work.
#include <metal_stdlib>
using namespace metal;

template <short ACC>
kernel void kernel_mma_peak(
        constant uint & iters [[buffer(0)]],
        device float * out [[buffer(1)]],
        uint tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[8 * 64];
    for (uint i = tid; i < 8 * 64; i += 128) sa[i] = half(float(i % 13) * 0.01f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 a[4], b[2];
    for (short i = 0; i < 4; ++i) simdgroup_load(a[i], sa + 64 * i, 8);
    for (short i = 0; i < 2; ++i) simdgroup_load(b[i], sa + 64 * (4 + i), 8);
    simdgroup_float8x8 c[ACC];
    for (short i = 0; i < ACC; ++i) c[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint it = 0; it < iters; ++it) {
        for (short i = 0; i < ACC; ++i) {
            simdgroup_multiply_accumulate(c[i], a[i % 4], b[(i / 4) % 2], c[i]);
        }
    }
    threadgroup float red[4];
    float s = 0.0f;
    for (short i = 0; i < ACC; ++i) s += c[i].thread_elements()[0] + c[i].thread_elements()[1];
    s = simd_sum(s);
    if (tid % 32 == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) out[tg] = red[0] + red[1] + red[2] + red[3];
}

template [[host_name("kernel_mma_peak_acc8")]] kernel void kernel_mma_peak<8>(constant uint &, device float *, uint, uint, ushort);
template [[host_name("kernel_mma_peak_acc16")]] kernel void kernel_mma_peak<16>(constant uint &, device float *, uint, uint, ushort);
template [[host_name("kernel_mma_peak_acc4")]] kernel void kernel_mma_peak<4>(constant uint &, device float *, uint, uint, ushort);
