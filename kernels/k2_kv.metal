#include <metal_stdlib>
using namespace metal;

// K2 cache v1: LE half scale followed by 32 signed bytes, no padding.
// Integer exponent checks preserve invalid-input detection under fast math.
inline void k2_quantize_q8_block(device const float * source,
                                device uchar * block, ushort lane) {
    const float x = source[lane];
    const bool finite = (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u;
    if (!simd_all(finite)) {
        if (lane == 0) { block[0] = 0; block[1] = 0x7e; }
        block[2 + lane] = 0;
        return;
    }
    const float maximum = simd_max(fabs(x));
    const float d = maximum / 127.0f;
    const ushort bits = as_type<ushort>(half(d));
    if ((bits & 0x7c00u) == 0x7c00u) {
        if (lane == 0) { block[0] = 0; block[1] = 0x7e; }
        block[2 + lane] = 0;
        return;
    }
    if (lane == 0) {
        block[0] = uchar(bits & 255u);
        block[1] = uchar(bits >> 8);
    }
    if (bits == 0) {
        block[2 + lane] = 0;
    } else {
        const float q = clamp(round(x * (1.0f / d)), -127.0f, 127.0f);
        ((device char *)(block + 2))[lane] = char(q);
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_k2_store_q8_kv(
        device const float * key [[buffer(0)]],
        device const float * value [[buffer(1)]],
        device uchar * key_bytes [[buffer(2)]],
        device uchar * value_bytes [[buffer(3)]],
        uint block [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    if (block >= 32u) return;
    k2_quantize_q8_block(key + block * 32u, key_bytes + block * 34u, lane);
    k2_quantize_q8_block(value + block * 32u, value_bytes + block * 34u, lane);
}

inline float4 k2_q8_quad(device const uchar * rows, uint position, uint head, ushort lane) {
    const ulong block_index = (ulong)position * 32u + head * 4u + lane / 8u;
    device const uchar * block = rows + block_index * 34u;
    const ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
    const float scale = float(as_type<half>(bits));
    device const char * q = (device const char *)(block + 2 + (lane % 8u) * 4u);
    // Scalar byte loads: 34-byte blocks do not provide aligned vector payloads.
    return float4(float(q[0]), float(q[1]), float(q[2]), float(q[3])) * scale;
}

struct k2_q8_attention_args { uint positions; float scale; };

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_k2_attn_online_q8kv_h128(
        constant k2_q8_attention_args & args [[buffer(0)]],
        device const float * query [[buffer(1)]],
        device const uchar * keys [[buffer(2)]],
        device const uchar * values [[buffer(3)]],
        device float * output [[buffer(4)]],
        uint head [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    if (head >= 32u) return;
    const uint kv_head = head / 4u;
    const float4 q = ((device const float4 *)(query + (ulong)head * 128u))[lane];
    float maximum = simd_sum(dot(q, k2_q8_quad(keys, 0, kv_head, lane))) * args.scale;
    float denominator = 1.0f;
    float4 accumulator = k2_q8_quad(values, 0, kv_head, lane);
    for (uint position = 1; position < args.positions; ++position) {
        const float score = simd_sum(dot(q, k2_q8_quad(keys, position, kv_head, lane))) * args.scale;
        const float next_maximum = max(maximum, score);
        const float previous_weight = exp(maximum - next_maximum);
        const float current_weight = exp(score - next_maximum);
        accumulator = accumulator * previous_weight
            + k2_q8_quad(values, position, kv_head, lane) * current_weight;
        denominator = denominator * previous_weight + current_weight;
        maximum = next_maximum;
    }
    ((device float4 *)(output + (ulong)head * 128u))[lane] = accumulator / denominator;
}
