#include <metal_stdlib>
using namespace metal;

struct muse_prefill_online_args {
    uint row_count;
    uint base_position;
    uint end_position;
    uint kv_stride;
    uint query_head_count;
    uint kv_head_count;
    uint head_dim;
    uint sliding_window;
    float scale;
};

// Query rows supply parallelism; each SIMDgroup owns one causal H128 query.
[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_muse_prefill_online_h128(
    constant muse_prefill_online_args & args [[buffer(0)]],
    device const float * query [[buffer(1)]],
    device const half * key [[buffer(2)]],
    device const half * value [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = group.x;
    const uint row = group.y;
    if (head >= 32 || row >= args.row_count) return;
    const uint end = args.base_position + row + 1;
    const uint start = args.sliding_window && end > args.sliding_window
        ? end - args.sliding_window : 0;
    const ulong query_offset = (ulong)row * 4096 + head * 128;
    const float4 q = ((device const float4 *)(query + query_offset))[lane];
    ulong offset = (ulong)start * args.kv_stride + (head / 16) * 128 + lane * 4;
    float maximum = simd_sum(dot(q, float4(*((device const half4 *)(key + offset))))) * args.scale;
    float denominator = 1.0f;
    float4 accumulator = float4(*((device const half4 *)(value + offset)));
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
    ((device float4 *)(output + query_offset))[lane] = accumulator / denominator;
}
