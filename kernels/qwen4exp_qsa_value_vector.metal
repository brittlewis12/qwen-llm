// Same split8 slot traversal, explicit F32 vector accumulators.
#include <metal_stdlib>
using namespace metal;
struct qwen4exp_qsa_value_args {
    uint start_position;
    uint query_count;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    uint block_budget;
    uint ratio;
    uint row_stride;
    uint cache_capacity;
    float scale;
};
[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_qwen4exp_qsa_selected_value_vec8_f32(
        constant qwen4exp_qsa_value_args & args [[buffer(0)]],
        device const float * query_gate_projection [[buffer(1)]],
        device const half * value_cache [[buffer(2)]],
        device const int * token_ids [[buffer(3)]],
        device const int * selected_counts [[buffer(4)]],
        device const int * status [[buffer(5)]],
        device const float * masses [[buffer(6)]],
        device float * output [[buffer(7)]],
        device const float * denominators [[buffer(8)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    if (args.query_heads != 24u || args.kv_heads != 2u
            || args.head_dim != 256u || args.block_budget != 512u
            || args.ratio != 4u || args.row_stride != 2051u
            || args.query_count == 0u || args.query_count > 32u
            || group.x >= 8u || group.y >= 2u || group.z >= args.query_count) return;
    const uint query = group.z;
    const uint first_head = group.y * 12u;
    const uint d0 = group.x * 32u;
    const ulong sequence = ulong(args.start_position) + query + 1u;
    const bool valid = status[query] == 0 && selected_counts[query] == 512;
    const uint count = valid ? 2048u + uint(sequence % 4u) : 0u;
    if (!valid) {
        for (uint cell = tid; cell < 12u * 32u; cell += 256u) {
            output[(ulong(query) * 24u + first_head + cell / 32u) * 256u
                + d0 + cell % 32u] = 0.0f;
        }
        return;
    }
    float4 a0=0.0f,a1=0.0f,a2=0.0f;
    for (uint slot = simdgroup; slot < count; slot += 8u) {
        const int position = token_ids[ulong(query) * 2051u + slot];
        if (position < 0 || uint(position) >= args.cache_capacity
                || ulong(position) >= sequence) continue;
        const float value = float(value_cache[(ulong(position) * 2u + group.y) * 256u
            + d0 + lane]);
        device const float *m=masses+(ulong(query)*24u+first_head)*2051u+slot;
        a0+=value*float4(m[0],m[2051],m[4102],m[6153]);
        a1+=value*float4(m[8204],m[10255],m[12306],m[14357]);
        a2+=value*float4(m[16408],m[18459],m[20510],m[22561]);
    }
    threadgroup float partials[8u * 12u * 32u];
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 12u; ++h) {
        partials[(uint(simdgroup) * 12u + h) * 32u + lane] = (h<4u ? a0[h] : (h<8u ? a1[h-4u] : a2[h-8u]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint cell = tid; cell < 12u * 32u; cell += 256u) {
        float total = 0.0f;
        #pragma clang loop unroll(full)
        for (uint split = 0u; split < 8u; ++split) total += partials[split * 384u + cell];
        const ulong head = ulong(query) * 24u + first_head + cell / 32u;
        const uint dimension = d0 + cell % 32u;
        const float denominator = denominators[head];
        const float gate = 1.0f / (1.0f + exp(-query_gate_projection[head * 512u + 256u + dimension]));
        output[head * 256u + dimension] = denominator > 0.0f ? total / denominator * gate : 0.0f;
    }
}

