#include <metal_stdlib>
using namespace metal;

struct muse_matrix_pv_args {
    uint row_count, base_position, end_position, kv_stride;
    uint query_head_count, kv_head_count, head_dim, sliding_window;
    float scale;
};

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_muse_prefill_matrix_pv_f32_h128(
    constant muse_matrix_pv_args & args [[buffer(0)]],
    device const float * query [[buffer(1)]],
    device const half * key [[buffer(2)]],
    device const half * value [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint first_row = group.y * 2;
    const uint row = first_row + sg / 2;
    const uint head_base = group.x * 16 + (sg % 2) * 8;
    const uint end = args.base_position + min(first_row + 1, args.row_count - 1) + 1;
    const uint first_end = args.base_position + first_row + 1;
    uint tile = args.sliding_window && first_end > args.sliding_window ? first_end - args.sliding_window : 0;
    threadgroup float panel[32 * 128];
    threadgroup float scores[32 * 40];
    float maximum = -INFINITY, denominator = 0.0f;
    simdgroup_float8x8 accumulator[16];
#pragma unroll
    for (ushort d = 0; d < 16; ++d) accumulator[d] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    while (tile < end) {
        const uint count = min(32u, end - tile);
        for (uint index = (uint)tid * 4; index < 32 * 128; index += 128 * 4) {
            const uint p = index / 128, dim = index % 128;
            const float4 k = p < count ? float4(*((device const half4 *)(key + ((ulong)tile + p) * args.kv_stride + group.x * 128 + dim))) : float4(0.0f);
            *((threadgroup float4 *)(panel + index)) = k;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            simdgroup_float8x8 dot_product[4];
#pragma unroll
            for (ushort column = 0; column < 4; ++column) dot_product[column] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            device const float * q = query + (ulong)min(row, args.row_count - 1) * 4096 + head_base * 128;
            for (ushort k0 = 0; k0 < 128; k0 += 8) {
                simdgroup_float8x8 qm;
                simdgroup_load(qm, q + k0, 128);
#pragma unroll
                for (ushort column = 0; column < 4; ++column) {
                    simdgroup_float8x8 km;
                    simdgroup_load(km, panel + column * 8 * 128 + k0, 128, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(dot_product[column], qm, km, dot_product[column]);
                }
            }
#pragma unroll
            for (ushort column = 0; column < 4; ++column) simdgroup_store(dot_product[column], scores + sg * 8 * 40 + column * 8, 40);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = (uint)tid * 4; index < 32 * 128; index += 128 * 4) {
            const uint p = index / 128, dim = index % 128;
            const float4 v = p < count ? float4(*((device const half4 *)(value + ((ulong)tile + p) * args.kv_stride + group.x * 128 + dim))) : float4(0.0f);
            *((threadgroup float4 *)(panel + index)) = v;
        }
        const uint visible_end = args.base_position + min(row, args.row_count - 1) + 1;
        const uint visible_start = args.sliding_window && visible_end > args.sliding_window ? visible_end - args.sliding_window : 0;
        bool needs_rescale = false;
#pragma unroll
        for (ushort h = 0; h < 8; ++h) {
            const ulong position = (ulong)tile + lane;
            float score = scores[(sg * 8 + h) * 40 + lane] * args.scale;
            if (row >= args.row_count || lane >= count || position < visible_start || position >= visible_end) score = -INFINITY;
            const float previous_maximum = simd_shuffle(maximum, h);
            const float next_maximum = max(previous_maximum, simd_max(score));
            const float alpha = isfinite(next_maximum) ? exp(previous_maximum - next_maximum) : 0.0f;
            needs_rescale = needs_rescale || alpha != 1.0f;
            const float probability = isfinite(score) ? exp(score - next_maximum) : 0.0f;
            const float next_denominator = simd_shuffle(denominator, h) * alpha + simd_sum(probability);
            if (lane == h) {
                denominator = next_denominator;
                maximum = next_maximum;
            }
            scores[(sg * 8 + h) * 40 + lane] = probability;
            if (lane < 8) scores[(sg * 8 + h) * 40 + 32 + lane] = lane == h ? alpha : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Diagonal scaling avoids relying on an opaque matrix-to-lane element mapping.
        if (needs_rescale) {
            simdgroup_float8x8 diagonal;
            simdgroup_load(diagonal, scores + sg * 8 * 40 + 32, 40);
#pragma unroll
            for (ushort d = 0; d < 16; ++d) {
                simdgroup_float8x8 scaled = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_multiply_accumulate(scaled, diagonal, accumulator[d], scaled);
                accumulator[d] = scaled;
            }
        }
        for (ushort p = 0; p < 32; p += 8) {
            simdgroup_float8x8 probability;
            simdgroup_load(probability, scores + sg * 8 * 40 + p, 40);
#pragma unroll
            for (ushort d = 0; d < 16; ++d) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, panel + p * 128 + d * 8, 128);
                simdgroup_multiply_accumulate(accumulator[d], probability, vm, accumulator[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile += count;
    }
#pragma unroll
    for (ushort d = 0; d < 16; ++d) simdgroup_store(accumulator[d], panel + sg * 1024 + d * 8, 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row < args.row_count) {
#pragma unroll
        for (ushort h = 0; h < 8; ++h) {
            const float4 result = *((threadgroup const float4 *)(panel + sg * 1024 + h * 128 + lane * 4));
            *((device float4 *)(output + (ulong)row * 4096 + (head_base + h) * 128 + lane * 4)) = result / simd_shuffle(denominator, h);
        }
    }
}
