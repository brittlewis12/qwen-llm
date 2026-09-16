// Split/GQA decode adapted from antirez/ds4 9139e2a, metal/qwen4.metal.
// Retired compatibility experiment; not a production route.
// Incumbent QK/global-softmax order; split only the value accumulation.
// MIT License
// Copyright (c) 2026 The ds4.c authors
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2023 DeepSeek
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <metal_stdlib>
using namespace metal;

struct qwen4exp_split_args {
    uint id_count;
    uint cache_capacity;
    uint splits;
    uint keys_per_split;
    uint row_stride;
};

kernel void kernel_qwen4exp_qsa_split_softmax_f32(
        constant qwen4exp_split_args & args [[buffer(0)]],
        device float * logits [[buffer(1)]],
        device float * partials [[buffer(2)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        uint lane [[thread_position_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    device float * row = logits + (ulong)head * args.row_stride;
    float local_maximum = -INFINITY;
    for (uint slot = lane; slot < args.id_count; slot += 256u) {
        local_maximum = max(local_maximum, row[slot]);
    }
    local_maximum = simd_max(local_maximum);
    if (simd_lane == 0u) scratch[uint(sg)] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const float candidate = simd_lane < 8u ? scratch[uint(simd_lane)] : -INFINITY;
        const float maximum = simd_max(candidate);
        if (simd_lane == 0u) scratch[8] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maximum = scratch[8];
    float local_denominator = 0.0f;
    for (uint slot = lane; slot < args.id_count; slot += 256u) {
        const float mass = maximum == -INFINITY ? 0.0f : exp(row[slot] - maximum);
        row[slot] = mass;
        local_denominator += mass;
    }
    local_denominator = simd_sum(local_denominator);
    if (simd_lane == 0u) scratch[uint(sg)] = local_denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (sg == 0u) {
        const float candidate = simd_lane < 8u ? scratch[uint(simd_lane)] : 0.0f;
        const float denominator = simd_sum(candidate);
        if (simd_lane == 0u) scratch[8] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < args.splits) {
        const ulong base = ((ulong)head * args.splits + lane) * 258u;
        partials[base] = scratch[8];
        partials[base + 1u] = 0.0f;
    }
}

// Released geometry only: 24 query heads / 2 KV heads / 256 dimensions.
// One SIMDgroup owns three heads; four SIMDgroups cover one KV head.
kernel void kernel_qwen4exp_qsa_split_f16(
        constant qwen4exp_split_args & args [[buffer(0)]],
        device const float * masses [[buffer(1)]],
        device const half * values [[buffer(3)]],
        device const int * ids [[buffer(4)]],
        device float * partials [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    if (group.x >= args.splits || group.y >= 2u) return;
    const uint first_head = group.y * 12u + uint(sg) * 3u;
    float numerator[3][8] = {};
    const uint end = min(args.id_count, (group.x + 1u) * args.keys_per_split);
    for (uint slot = group.x * args.keys_per_split; slot < end; ++slot) {
        const int position = ids[slot];
        if (position < 0 || uint(position) >= args.cache_capacity) continue;
        const ulong base = ((ulong)uint(position) * 2u + group.y) * 256u + uint(lane);
        float value[8];
        for (uint d = 0u; d < 8u; ++d) {
            value[d] = float(values[base + d * 32u]);
        }
        for (uint h = 0u; h < 3u; ++h) {
            const float mass = masses[(ulong)(first_head + h) * args.row_stride + slot];
            for (uint d = 0u; d < 8u; ++d) {
                numerator[h][d] += value[d] * mass;
            }
        }
    }
    for (uint h = 0u; h < 3u; ++h) {
        const ulong base = ((ulong)(first_head + h) * args.splits + group.x) * 258u;
        for (uint d = 0u; d < 8u; ++d) {
            partials[base + 2u + uint(lane) + d * 32u] = numerator[h][d];
        }
    }
}

kernel void kernel_qwen4exp_qsa_split_merge_f32(
        constant qwen4exp_split_args & args [[buffer(0)]],
        device const float * partials [[buffer(1)]],
        device const float * gate [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint head [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    if (head >= 24u) return;
    device const float * row = partials + (ulong)head * args.splits * 258u;
    const float denominator = row[0];
    float numerator[8] = {};
    for (uint split = 0u; split < args.splits; ++split) {
        device const float * part = row + split * 258u;
        for (uint d = 0u; d < 8u; ++d) {
            numerator[d] += part[2u + uint(lane) + d * 32u];
        }
    }
    for (uint d = 0u; d < 8u; ++d) {
        const uint index = head * 256u + uint(lane) + d * 32u;
        const float activation = 1.0f / (1.0f + exp(-gate[index]));
        output[index] = denominator > 0.0f ? numerator[d] / denominator * activation : 0.0f;
    }
}
