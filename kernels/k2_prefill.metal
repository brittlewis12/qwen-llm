#include <metal_stdlib>
using namespace metal;

// K2 general batched prefill: row-parallel forms of the serial K2 grouped
// RMSNorm and H128/GQA4 online attention. Each row keeps the serial kernel's
// per-row arithmetic order; only the grid gains a token-row axis.

struct k2_grouped_norm_rows_args {
    uint  width;   // elements per norm group (hidden / groups)
    uint  groups;  // norm groups per hidden row
    uint  rows;    // token rows
    uint  threads; // threads per threadgroup (reduction width)
    float eps;
};

// One threadgroup per (group, row). Same two-level simd reduction and
// `(x * scale) * gamma` epilogue as kernel_rms_norm_mul_f32 on one group.
kernel void kernel_k2_grouped_rms_norm_rows_f32(
        constant k2_grouped_norm_rows_args & args [[buffer(0)]],
        device const float * x     [[buffer(1)]],
        device const float * gamma [[buffer(2)]],
        device       float * y     [[buffer(3)]],
        threadgroup  float * shmem [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint group = tgpig.x;
    const uint row = tgpig.y;
    if (group >= args.groups || row >= args.rows) return;
    const uint ntg = args.threads;
    const ulong base = (ulong)row * args.width * args.groups + (ulong)group * args.width;
    device const float * xg = x + base;
    device const float * wg = gamma + (ulong)group * args.width;
    device       float * yg = y + base;

    float sumsq = 0.0f;
    for (uint i = tid; i < args.width; i += ntg) {
        const float v = xg[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    if (tiisg == 0) {
        shmem[sgitg] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float mean  = sumsq / float(args.width);
    const float scale = rsqrt(mean + args.eps);
    for (uint i = tid; i < args.width; i += ntg) {
        yg[i] = (xg[i] * scale) * wg[i];
    }
}

struct k2_attn_rows_args {
    uint  rows;            // query rows in this chunk
    uint  first_positions; // visible positions of row 0 (cache index + 1)
    float scale;
};

constant constexpr uint K2_ROWS_PER_GROUP = 4u;
constant constexpr uint K2_TILE_POSITIONS = 32u;
constant constexpr uint K2_GROUP_THREADS = 512u; // 4 GQA heads x 4 rows x 32 lanes

// Chunked causal H128/GQA4 attention over an F16 cache that already holds
// every chunk row. A threadgroup owns one KV head and four consecutive query
// rows; its 16 SIMDgroups (4 query heads x 4 rows) share each 32-position K/V
// tile staged once in threadgroup memory. Every SIMDgroup then runs exactly
// kernel_k2_attn_online_blocked_f16kv_h128's recurrence for its own row:
// 256-position online blocks merged in order (one block == the unblocked
// kernel), with row r seeing positions [0, first_positions + r).
[[max_total_threads_per_threadgroup(512)]]
kernel void kernel_k2_attn_online_rows_f16kv_h128(
        constant k2_attn_rows_args & args [[buffer(0)]],
        device const float * query  [[buffer(1)]], // [rows, 32 heads, 128]
        device const half  * keys   [[buffer(2)]], // [positions, 8 heads, 128]
        device const half  * values [[buffer(3)]], // [positions, 8 heads, 128]
        device       float * output [[buffer(4)]], // [rows, 32 heads, 128]
        threadgroup half4 * tile [[threadgroup(0)]],
        uint2  group        [[threadgroup_position_in_grid]],
        ushort simd         [[simdgroup_index_in_threadgroup]],
        ushort lane         [[thread_index_in_simdgroup]],
        ushort thread_index [[thread_index_in_threadgroup]]) {
    const uint kv_head = group.x;
    if (kv_head >= 8u) return;
    const uint head = kv_head * 4u + (uint)(simd % 4u);
    const uint row = group.y * K2_ROWS_PER_GROUP + (uint)(simd / 4u);
    const bool active = row < args.rows;
    const uint last_row = min(args.rows, (group.y + 1u) * K2_ROWS_PER_GROUP) - 1u;
    const uint group_positions = args.first_positions + last_row;
    const uint positions = active ? args.first_positions + row : 0u;
    const ulong q_index = ((ulong)(active ? row : 0u) * 32u + head) * 32u + lane;
    const float4 q = ((device const float4 *)query)[q_index];
    threadgroup half4 * key_tile = tile;
    threadgroup half4 * value_tile = tile + K2_TILE_POSITIONS * 32u;
    const ulong channel = (ulong)kv_head * 128u;

    float total_maximum = 0.0f;
    float total_denominator = 0.0f;
    float4 total_accumulator = 0.0f;
    bool has_total = false;
    float maximum = 0.0f;
    float denominator = 0.0f;
    float4 accumulator = 0.0f;

    for (uint start = 0u; start < group_positions; start += K2_TILE_POSITIONS) {
        const uint count = min(K2_TILE_POSITIONS, group_positions - start);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = thread_index; i < count * 32u; i += K2_GROUP_THREADS) {
            const ulong offset = (ulong)(start + i / 32u) * 1024u + channel + (ulong)(i % 32u) * 4u;
            key_tile[i] = *((device const half4 *)(keys + offset));
            value_tile[i] = *((device const half4 *)(values + offset));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint end = min(start + count, positions);
        for (uint position = start; position < end; ++position) {
            const uint slot = (position - start) * 32u + lane;
            const float score = simd_sum(dot(q, float4(key_tile[slot]))) * args.scale;
            const float4 value = float4(value_tile[slot]);
            if (position % 256u == 0u) {
                if (position != 0u) {
                    if (!has_total) {
                        total_maximum = maximum;
                        total_denominator = denominator;
                        total_accumulator = accumulator;
                        has_total = true;
                    } else {
                        const float merged_maximum = max(total_maximum, maximum);
                        const float previous_weight = exp(total_maximum - merged_maximum);
                        const float current_weight = exp(maximum - merged_maximum);
                        total_accumulator = total_accumulator * previous_weight + accumulator * current_weight;
                        total_denominator = total_denominator * previous_weight + denominator * current_weight;
                        total_maximum = merged_maximum;
                    }
                }
                maximum = score;
                denominator = 1.0f;
                accumulator = value;
            } else {
                const float next_maximum = max(maximum, score);
                const float previous_weight = exp(maximum - next_maximum);
                const float current_weight = exp(score - next_maximum);
                accumulator = accumulator * previous_weight + value * current_weight;
                denominator = denominator * previous_weight + current_weight;
                maximum = next_maximum;
            }
        }
    }
    if (!active) return;
    if (!has_total) {
        total_maximum = maximum;
        total_denominator = denominator;
        total_accumulator = accumulator;
    } else {
        const float merged_maximum = max(total_maximum, maximum);
        const float previous_weight = exp(total_maximum - merged_maximum);
        const float current_weight = exp(maximum - merged_maximum);
        total_accumulator = total_accumulator * previous_weight + accumulator * current_weight;
        total_denominator = total_denominator * previous_weight + denominator * current_weight;
    }
    ((device float4 *)output)[q_index] = total_accumulator / total_denominator;
}
