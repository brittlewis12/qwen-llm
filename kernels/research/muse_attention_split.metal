#include <metal_stdlib>
using namespace metal;

struct muse_split_args {
    uint positions;
    uint kv_stride;
    uint partitions;
    float scale;
};

// Partition the position scan; each SIMDgroup keeps the H128 online state.
[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_muse_split_attention_h128(
    constant muse_split_args & args [[buffer(0)]],
    device const float * query [[buffer(1)]],
    device const half * key [[buffer(2)]],
    device const half * value [[buffer(3)]],
    device float * partial [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = group.x;
    const uint partition = group.y;
    if (head >= 32 || partition >= args.partitions) return;
    const uint width = (args.positions + args.partitions - 1) / args.partitions;
    const uint start = partition * width;
    const uint end = min(start + width, args.positions);
    device float * destination = partial + ((ulong)head * args.partitions + partition) * 132;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float4 accumulator = 0.0f;
    if (start < end) {
        const float4 q = ((device const float4 *)(query + head * 128))[lane];
        ulong offset = (ulong)start * args.kv_stride + (head / 16) * 128 + lane * 4;
        maximum = simd_sum(dot(q, float4(*((device const half4 *)(key + offset))))) * args.scale;
        denominator = 1.0f;
        accumulator = float4(*((device const half4 *)(value + offset)));
        for (uint position = start + 1; position < end; ++position) {
            offset = (ulong)position * args.kv_stride + (head / 16) * 128 + lane * 4;
            const float score = simd_sum(dot(q, float4(*((device const half4 *)(key + offset))))) * args.scale;
            const float next_maximum = max(maximum, score);
            const float previous_weight = exp(maximum - next_maximum);
            const float current_weight = exp(score - next_maximum);
            accumulator = accumulator * previous_weight
                + float4(*((device const half4 *)(value + offset))) * current_weight;
            denominator = denominator * previous_weight + current_weight;
            maximum = next_maximum;
        }
    }
    ((device float4 *)destination)[lane] = accumulator;
    if (lane == 0) {
        ((device float4 *)(destination + 128))[0] = float4(maximum, denominator, 0.0f, 0.0f);
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_muse_split_attention_reduce_h128(
    constant muse_split_args & args [[buffer(0)]],
    device const float * partial [[buffer(1)]],
    device float * output [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (head >= 32) return;
    const ulong base = (ulong)head * args.partitions * 132;
    const float maximum = lane < args.partitions ? partial[base + lane * 132 + 128] : -INFINITY;
    const float global_maximum = simd_max(maximum);
    const float weight = lane < args.partitions ? exp(maximum - global_maximum) : 0.0f;
    const float denominator = lane < args.partitions ? partial[base + lane * 132 + 129] : 0.0f;
    const float global_denominator = simd_sum(denominator * weight);
    float4 accumulator = 0.0f;
    for (uint partition = 0; partition < args.partitions; ++partition) {
        const float factor = simd_shuffle(weight, partition);
        const float4 values = ((device const float4 *)(partial + base + partition * 132))[lane];
        accumulator += values * factor;
    }
    ((device float4 *)(output + head * 128))[lane] = accumulator / global_denominator;
}
