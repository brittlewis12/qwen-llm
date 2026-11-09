// Split/GQA decode adapted from antirez/ds4 9139e2a, metal/qwen4.metal.
// Production decode attention for the released 24/2/256 QSA geometry:
// incumbent QK logits and global softmax order; the value accumulation is
// split across key ranges (one threadgroup per split and KV head, each
// SIMD-group owning three query heads) and merged with the gate.
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

inline void fn_qsa_softmax_body(constant qwen4exp_split_args &args, device float *logits,
                                device float *partials, threadgroup float *scratch, uint head, uint lane,
                                ushort sg, ushort simd_lane) {
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
kernel void kernel_qwen4exp_qsa_split_softmax_f32(
        constant qwen4exp_split_args &args [[buffer(0)]], device float *logits [[buffer(1)]],
        device float *partials [[buffer(2)]], threadgroup float *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]], ushort simd_lane [[thread_index_in_simdgroup]]) {
    fn_qsa_softmax_body(args, logits, partials, scratch, head, lane, sg, simd_lane);
}

// Released geometry only: 24 query heads / 2 KV heads / 256 dimensions.
// One SIMDgroup owns three heads; four SIMDgroups cover one KV head.
inline void fn_qsa_value_body(constant qwen4exp_split_args &args, device const float *masses,
                              device const half *values, device const int *ids, device float *partials,
                              uint2 group, ushort sg, ushort lane) {
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
kernel void kernel_qwen4exp_qsa_split_f16(
        constant qwen4exp_split_args &args [[buffer(0)]], device const float *masses [[buffer(1)]],
        device const half *values [[buffer(3)]], device const int *ids [[buffer(4)]],
        device float *partials [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    fn_qsa_value_body(args, masses, values, ids, partials, group, sg, lane);
}

inline void fn_qsa_merge_body(constant qwen4exp_split_args &args, device const float *partials,
                              device const float *gate, device float *output, uint head, ushort lane) {
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
kernel void kernel_qwen4exp_qsa_split_merge_f32(constant qwen4exp_split_args &args [[buffer(0)]],
                                                device const float *partials [[buffer(1)]],
                                                device const float *gate [[buffer(2)]],
                                                device float *output [[buffer(3)]],
                                                uint head [[threadgroup_position_in_grid]],
                                                ushort lane [[thread_index_in_simdgroup]]) {
    fn_qsa_merge_body(args, partials, gate, output, head, lane);
}

kernel void kernel_fn_batch_qsa_softmax(
        constant qwen4exp_split_args *args [[buffer(0)]], device float *logits0 [[buffer(1)]],
        device float *logits1 [[buffer(2)]], device float *logits2 [[buffer(3)]],
        device float *logits3 [[buffer(4)]], device float *partial0 [[buffer(5)]],
        device float *partial1 [[buffer(6)]], device float *partial2 [[buffer(7)]],
        device float *partial3 [[buffer(8)]], threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]], uint2 local [[thread_position_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = group.y;
    fn_qsa_softmax_body(args[r],
                        r == 0   ? logits0
                        : r == 1 ? logits1
                        : r == 2 ? logits2
                                 : logits3,
                        r == 0   ? partial0
                        : r == 1 ? partial1
                        : r == 2 ? partial2
                                 : partial3,
                        scratch, group.x, local.x, sg, lane);
}
kernel void kernel_fn_batch_qsa_value(
        constant qwen4exp_split_args *args [[buffer(0)]], device const float *mass0 [[buffer(1)]],
        device const float *mass1 [[buffer(2)]], device const float *mass2 [[buffer(3)]],
        device const float *mass3 [[buffer(4)]], device const half *values0 [[buffer(5)]],
        device const half *values1 [[buffer(6)]], device const half *values2 [[buffer(7)]],
        device const half *values3 [[buffer(8)]], device const int *ids0 [[buffer(9)]],
        device const int *ids1 [[buffer(10)]], device const int *ids2 [[buffer(11)]],
        device const int *ids3 [[buffer(12)]], device float *partial0 [[buffer(13)]],
        device float *partial1 [[buffer(14)]], device float *partial2 [[buffer(15)]],
        device float *partial3 [[buffer(16)]], uint3 group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = group.z;
    fn_qsa_value_body(args[r],
                      r == 0   ? mass0
                      : r == 1 ? mass1
                      : r == 2 ? mass2
                               : mass3,
                      r == 0   ? values0
                      : r == 1 ? values1
                      : r == 2 ? values2
                               : values3,
                      r == 0   ? ids0
                      : r == 1 ? ids1
                      : r == 2 ? ids2
                               : ids3,
                      r == 0   ? partial0
                      : r == 1 ? partial1
                      : r == 2 ? partial2
                               : partial3,
                      group.xy, sg, lane);
}
kernel void kernel_fn_batch_qsa_merge(
        constant qwen4exp_split_args *args [[buffer(0)]], device const float *partial0 [[buffer(1)]],
        device const float *partial1 [[buffer(2)]], device const float *partial2 [[buffer(3)]],
        device const float *partial3 [[buffer(4)]], device const float *gate0 [[buffer(5)]],
        device const float *gate1 [[buffer(6)]], device const float *gate2 [[buffer(7)]],
        device const float *gate3 [[buffer(8)]], device float *out0 [[buffer(9)]],
        device float *out1 [[buffer(10)]], device float *out2 [[buffer(11)]],
        device float *out3 [[buffer(12)]], uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = group.y;
    fn_qsa_merge_body(args[r],
                      r == 0   ? partial0
                      : r == 1 ? partial1
                      : r == 2 ? partial2
                               : partial3,
                      r == 0   ? gate0
                      : r == 1 ? gate1
                      : r == 2 ? gate2
                               : gate3,
                      r == 0   ? out0
                      : r == 1 ? out1
                      : r == 2 ? out2
                               : out3,
                      group.x, lane);
}
