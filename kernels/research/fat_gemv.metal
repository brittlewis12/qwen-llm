// Fat Q8_0 GEMV for decode: R output rows per simdgroup, SG simdgroups per
// threadgroup. The activation chunk is loaded once per block and reused across
// R rows; each lane keeps R independent F32 accumulators. No threadgroup
// memory and no cross-simdgroup reduction.
//
// Q8_0 block (34 bytes / 32 elements): half d, int8 qs[32].
//
// Lane mapping (one simdgroup): ix = lane/4 picks one of 8 blocks per step,
// il = lane%4 picks an 8-value chunk inside the block. Four lanes cover one
// block, eight lane-groups cover eight blocks per iteration.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint QK8_0 = 32;
constant constexpr uint Q8_0_BYTES = 34;

template <int R, int SG>
[[max_total_threads_per_threadgroup(SG * 32)]]
kernel void fat_gemv_q8_0_f32(
        constant uint2 & shape [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint n_in = shape.x;
    const uint n_out = shape.y;
    const uint nb = n_in / QK8_0;
    const uint first_row = (tgpig * SG + sgitg) * R;
    if (first_row >= n_out) return;

    const uint ix = lane / 4; // 0..7: block group
    const uint il = lane % 4; // 0..3: 8-value chunk inside the block
    const ulong row_stride = ulong(nb) * Q8_0_BYTES;
    device const uchar * row0 = weight + ulong(first_row) * row_stride;

    float acc[R];
#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint ib = ix; ib < nb; ib += 8) {
        const uint k0 = ib * QK8_0 + il * 8;
        float xv[8];
#pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) xv[i] = x[k0 + i];

#pragma clang loop unroll(full)
        for (int r = 0; r < R; ++r) {
            if (first_row + uint(r) >= n_out) break;
            device const uchar * blk = row0 + ulong(r) * row_stride + ulong(ib) * Q8_0_BYTES;
            const float d = float(*reinterpret_cast<device const half *>(blk));
            device const char * qs = reinterpret_cast<device const char *>(blk + 2) + il * 8;
            float sumq = 0.0f;
#pragma clang loop unroll(full)
            for (int i = 0; i < 8; ++i) sumq += float(qs[i]) * xv[i];
            acc[r] += sumq * d;
        }
    }

#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
        const float v = simd_sum(acc[r]);
        if (lane == 0 && first_row + uint(r) < n_out) y[first_row + uint(r)] = v;
    }
}


typedef decltype(fat_gemv_q8_0_f32<2, 1>) fat_gemv_r2_sg1_t;
template [[host_name("kernel_fat_gemv_q8_0_f32_r2_sg1")]] kernel fat_gemv_r2_sg1_t fat_gemv_q8_0_f32<2, 1>;
typedef decltype(fat_gemv_q8_0_f32<2, 2>) fat_gemv_r2_sg2_t;
template [[host_name("kernel_fat_gemv_q8_0_f32_r2_sg2")]] kernel fat_gemv_r2_sg2_t fat_gemv_q8_0_f32<2, 2>;
typedef decltype(fat_gemv_q8_0_f32<2, 4>) fat_gemv_r2_sg4_t;
template [[host_name("kernel_fat_gemv_q8_0_f32_r2_sg4")]] kernel fat_gemv_r2_sg4_t fat_gemv_q8_0_f32<2, 4>;
typedef decltype(fat_gemv_q8_0_f32<4, 1>) fat_gemv_r4_sg1_t;
template [[host_name("kernel_fat_gemv_q8_0_f32_r4_sg1")]] kernel fat_gemv_r4_sg1_t fat_gemv_q8_0_f32<4, 1>;
typedef decltype(fat_gemv_q8_0_f32<4, 2>) fat_gemv_r4_sg2_t;
template [[host_name("kernel_fat_gemv_q8_0_f32_r4_sg2")]] kernel fat_gemv_r4_sg2_t fat_gemv_q8_0_f32<4, 2>;
typedef decltype(fat_gemv_q8_0_f32<4, 4>) fat_gemv_r4_sg4_t;
template [[host_name("kernel_fat_gemv_q8_0_f32_r4_sg4")]] kernel fat_gemv_r4_sg4_t fat_gemv_q8_0_f32<4, 4>;
typedef decltype(fat_gemv_q8_0_f32<8, 1>) fat_gemv_r8_sg1_t;
template [[host_name("kernel_fat_gemv_q8_0_f32_r8_sg1")]] kernel fat_gemv_r8_sg1_t fat_gemv_q8_0_f32<8, 1>;
typedef decltype(fat_gemv_q8_0_f32<8, 2>) fat_gemv_r8_sg2_t;
template [[host_name("kernel_fat_gemv_q8_0_f32_r8_sg2")]] kernel fat_gemv_r8_sg2_t fat_gemv_q8_0_f32<8, 2>;
typedef decltype(fat_gemv_q8_0_f32<8, 4>) fat_gemv_r8_sg4_t;
template [[host_name("kernel_fat_gemv_q8_0_f32_r8_sg4")]] kernel fat_gemv_r8_sg4_t fat_gemv_q8_0_f32<8, 4>;
