#include <metal_stdlib>
using namespace metal;

struct muse_glimmer_rope_args {
    uint q_pair_count;
    uint pair_count;
    uint head_dim;
    uint position;
    float theta;
};

kernel void kernel_muse_glimmer_rope_adjacent_pair_in_place_f32(
        constant muse_glimmer_rope_args & args [[buffer(0)]],
        device float * q [[buffer(1)]],
        device float * k [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.pair_count) return;
    const bool is_q = index < args.q_pair_count;
    const uint local_index = is_q ? index : index - args.q_pair_count;
    const uint pairs_per_head = args.head_dim / 2u;
    const uint head = local_index / pairs_per_head;
    const uint pair = local_index - head * pairs_per_head;
    const uint relative = pair * 2u;
    const uint first_index = head * args.head_dim + relative;
    const uint second_index = first_index + 1u;
    device float * values = is_q ? q : k;

    const float angle = float(args.position)
        * pow(args.theta, -float(relative) / float(args.head_dim));
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const float first = values[first_index];
    const float second = values[second_index];
    values[first_index] = first * cosine - second * sine;
    values[second_index] = first * sine + second * cosine;
}

struct muse_glimmer_rope_periodic_args {
    uint q_pair_count;
    uint pair_count;
    uint q_pairs_per_row;
    uint k_pairs_per_row;
    uint head_dim;
    uint position_period;
    uint inverse;
    float theta;
};

kernel void kernel_muse_glimmer_rope_adjacent_pair_periodic_in_place_f32(
        constant muse_glimmer_rope_periodic_args & args [[buffer(0)]],
        device float * q [[buffer(1)]],
        device float * k [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.pair_count) return;
    const bool is_q = index < args.q_pair_count;
    const uint local_index = is_q ? index : index - args.q_pair_count;
    const uint pairs_per_row = is_q ? args.q_pairs_per_row : args.k_pairs_per_row;
    const uint row = local_index / pairs_per_row;
    const uint pair_in_row = local_index - row * pairs_per_row;
    const uint pairs_per_head = args.head_dim / 2u;
    const uint relative = (pair_in_row % pairs_per_head) * 2u;
    const ulong first_index = (ulong)row * pairs_per_row * 2u
        + (ulong)pair_in_row * 2u;
    const ulong second_index = first_index + 1u;
    device float * values = is_q ? q : k;

    const uint position = row % args.position_period;
    const float angle = float(position)
        * pow(args.theta, -float(relative) / float(args.head_dim));
    const float cosine = cos(angle);
    float sine = sin(angle);
    if (args.inverse != 0u) sine = -sine;
    const float first = values[first_index];
    const float second = values[second_index];
    values[first_index] = first * cosine - second * sine;
    values[second_index] = first * sine + second * cosine;
}

struct muse_glimmer_softcap_args {
    uint n;
    float scale;
    float cap;
};

kernel void kernel_muse_glimmer_logit_softcap_f32(
        constant muse_glimmer_softcap_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device float * output [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.n) return;
    output[index] = args.cap * tanh(input[index] * args.scale / args.cap);
}

struct muse_glimmer_attn_decode_args {
    uint n_pos;
    uint kv_stride;
    float scale;
};

// Long-context scalar-decode attention for the released 32Q/2KV/H128 shape.
// One SIMDgroup owns one query head. Each lane owns one contiguous float4 of
// the output and all lanes cooperatively reduce the QK dot product. Online
// softmax keeps the working set in registers regardless of context length.
[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_muse_glimmer_attn_decode_online_f16kv_h128_f32(
        constant muse_glimmer_attn_decode_args & args [[buffer(0)]],
        device const float * query [[buffer(1)]],
        device const half * key_cache [[buffer(2)]],
        device const half * value_cache [[buffer(3)]],
        device float * output [[buffer(4)]],
        uint query_head [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint QUERY_HEAD_COUNT = 32;
    constexpr uint KV_HEAD_COUNT = 2;
    constexpr uint HEAD_DIM = 128;
    constexpr uint QUERY_GROUP = QUERY_HEAD_COUNT / KV_HEAD_COUNT;
    if (query_head >= QUERY_HEAD_COUNT) return;

    const uint kv_head = query_head / QUERY_GROUP;
    const device float4 * query4 =
        (device const float4 *)(query + (ulong)query_head * HEAD_DIM);
    device float4 * output4 =
        (device float4 *)(output + (ulong)query_head * HEAD_DIM);
    const float4 query_values = query4[lane];

    ulong cache_offset = (ulong)kv_head * HEAD_DIM + (ulong)lane * 4u;
    const float first_score = simd_sum(dot(
        query_values,
        float4(*((device const half4 *)(key_cache + cache_offset)))
    )) * args.scale;
    float maximum = first_score;
    float denominator = 1.0f;
    float4 accumulator =
        float4(*((device const half4 *)(value_cache + cache_offset)));

    for (uint position = 1; position < args.n_pos; ++position) {
        cache_offset = (ulong)position * args.kv_stride
            + (ulong)kv_head * HEAD_DIM
            + (ulong)lane * 4u;
        const float score = simd_sum(dot(
            query_values,
            float4(*((device const half4 *)(key_cache + cache_offset)))
        )) * args.scale;
        const float next_maximum = max(maximum, score);
        const float previous_weight = exp(maximum - next_maximum);
        const float current_weight = exp(score - next_maximum);
        accumulator = accumulator * previous_weight
            + float4(*((device const half4 *)(value_cache + cache_offset)))
                * current_weight;
        denominator = denominator * previous_weight + current_weight;
        maximum = next_maximum;
    }

    output4[lane] = accumulator / denominator;
}

constant constexpr uint MUSE_GLIMMER_VJP_MAX_TOKENS = 16;
constant constexpr uint MUSE_GLIMMER_SIMD_WIDTH = 32;

struct muse_glimmer_attention_vjp_args {
    uint basis_count;
    uint n_tokens;
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    float scale;
};

// One SIMDgroup owns one basis row and query head. Shared probabilities and
// primals are reused across the basis bank; K/V partials remain head-private.
[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_muse_glimmer_causal_gqa_vjp_bank_f32(
        constant muse_glimmer_attention_vjp_args & args [[buffer(0)]],
        device const float * q                 [[buffer(1)]],
        device const float * k                 [[buffer(2)]],
        device const float * v                 [[buffer(3)]],
        device const float * gate              [[buffer(4)]],
        device const float * attention_output  [[buffer(5)]],
        device const float * probabilities     [[buffer(6)]],
        device const float * grad_gated        [[buffer(7)]],
        device       float * grad_q            [[buffer(8)]],
        device       float * partial_grad_k    [[buffer(9)]],
        device       float * partial_grad_v    [[buffer(10)]],
        device       float * grad_gate         [[buffer(11)]],
        threadgroup  float * partials          [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint q_head = tgpig.x;
    const uint basis = tgpig.y;
    if (q_head >= args.q_heads || basis >= args.basis_count) return;

    const uint group = args.q_heads / args.kv_heads;
    const uint kv_head = q_head / group;
    const uint dims_per_lane = args.head_dim / MUSE_GLIMMER_SIMD_WIDTH;
    const ulong query_width = (ulong)args.q_heads * args.head_dim;
    const ulong kv_width = (ulong)args.kv_heads * args.head_dim;
    const ulong bank_query_base = (ulong)basis * args.n_tokens * query_width;
    threadgroup float * partial_k = partials;
    threadgroup float * partial_v = partials + args.n_tokens * args.head_dim;

    for (uint key_token = 0; key_token < args.n_tokens; ++key_token) {
        for (uint slot = 0; slot < dims_per_lane; ++slot) {
            const uint dim = (uint)tiisg + slot * MUSE_GLIMMER_SIMD_WIDTH;
            const uint partial_index = key_token * args.head_dim + dim;
            partial_k[partial_index] = 0.0f;
            partial_v[partial_index] = 0.0f;
        }
    }

    for (uint token = 0; token < args.n_tokens; ++token) {
        const ulong shared_q_base = (ulong)token * query_width
            + (ulong)q_head * args.head_dim;
        const ulong bank_q_base = bank_query_base + (ulong)token * query_width
            + (ulong)q_head * args.head_dim;
        float q_values[8];
        float grad_attention[8];
        float grad_q_values[8];

        for (uint slot = 0; slot < dims_per_lane; ++slot) {
            const uint dim = (uint)tiisg + slot * MUSE_GLIMMER_SIMD_WIDTH;
            const float gate_value = gate[shared_q_base + dim];
            const float sigmoid_gate = 1.0f / (1.0f + exp(-gate_value));
            const float incoming = grad_gated[bank_q_base + dim];
            q_values[slot] = q[shared_q_base + dim];
            grad_attention[slot] = incoming * sigmoid_gate;
            grad_q_values[slot] = 0.0f;
            grad_gate[bank_q_base + dim] = incoming
                * attention_output[shared_q_base + dim]
                * sigmoid_gate * (1.0f - sigmoid_gate);
        }

        float grad_probabilities[MUSE_GLIMMER_VJP_MAX_TOKENS];
        float probability_dot = 0.0f;
        const ulong probability_base =
            ((ulong)token * args.q_heads + q_head) * args.n_tokens;
        for (uint key_token = 0; key_token <= token; ++key_token) {
            const ulong v_base = (ulong)key_token * kv_width
                + (ulong)kv_head * args.head_dim;
            float partial = 0.0f;
            for (uint slot = 0; slot < dims_per_lane; ++slot) {
                const uint dim = (uint)tiisg + slot * MUSE_GLIMMER_SIMD_WIDTH;
                partial += grad_attention[slot] * v[v_base + dim];
            }
            const float grad_probability = simd_sum(partial);
            grad_probabilities[key_token] = grad_probability;
            probability_dot += probabilities[probability_base + key_token]
                * grad_probability;
        }

        for (uint key_token = 0; key_token <= token; ++key_token) {
            const float probability = probabilities[probability_base + key_token];
            const float grad_score = probability
                * (grad_probabilities[key_token] - probability_dot);
            const ulong k_base = (ulong)key_token * kv_width
                + (ulong)kv_head * args.head_dim;
            for (uint slot = 0; slot < dims_per_lane; ++slot) {
                const uint dim = (uint)tiisg + slot * MUSE_GLIMMER_SIMD_WIDTH;
                const uint partial_index = key_token * args.head_dim + dim;
                grad_q_values[slot] += args.scale * grad_score * k[k_base + dim];
                partial_k[partial_index] +=
                    args.scale * grad_score * q_values[slot];
                partial_v[partial_index] += probability * grad_attention[slot];
            }
        }

        for (uint slot = 0; slot < dims_per_lane; ++slot) {
            const uint dim = (uint)tiisg + slot * MUSE_GLIMMER_SIMD_WIDTH;
            grad_q[bank_q_base + dim] = grad_q_values[slot];
        }
    }

    const ulong partial_base =
        ((ulong)basis * args.q_heads + q_head) * args.n_tokens * args.head_dim;
    for (uint key_token = 0; key_token < args.n_tokens; ++key_token) {
        for (uint slot = 0; slot < dims_per_lane; ++slot) {
            const uint dim = (uint)tiisg + slot * MUSE_GLIMMER_SIMD_WIDTH;
            const uint local_index = key_token * args.head_dim + dim;
            partial_grad_k[partial_base + local_index] = partial_k[local_index];
            partial_grad_v[partial_base + local_index] = partial_v[local_index];
        }
    }
}

// Reduce the query-head-private K/V partials into each basis row's two KV heads.
[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_muse_glimmer_causal_gqa_vjp_reduce_kv_f32(
        constant muse_glimmer_attention_vjp_args & args [[buffer(0)]],
        device const float * partial_grad_k [[buffer(1)]],
        device const float * partial_grad_v [[buffer(2)]],
        device       float * grad_k         [[buffer(3)]],
        device       float * grad_v         [[buffer(4)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint kv_head = tgpig.x;
    const uint key_token = tgpig.y;
    const uint basis = tgpig.z;
    if (kv_head >= args.kv_heads
            || key_token >= args.n_tokens
            || basis >= args.basis_count) return;

    const uint group = args.q_heads / args.kv_heads;
    const uint first_q_head = kv_head * group;
    const uint dims_per_lane = args.head_dim / MUSE_GLIMMER_SIMD_WIDTH;
    const ulong output_base =
        (((ulong)basis * args.n_tokens + key_token) * args.kv_heads + kv_head)
        * args.head_dim;

    for (uint slot = 0; slot < dims_per_lane; ++slot) {
        const uint dim = (uint)tiisg + slot * MUSE_GLIMMER_SIMD_WIDTH;
        float sum_k = 0.0f;
        float sum_v = 0.0f;
        for (uint local_head = 0; local_head < group; ++local_head) {
            const uint q_head = first_q_head + local_head;
            const ulong partial_index =
                (((ulong)basis * args.q_heads + q_head) * args.n_tokens
                    + key_token) * args.head_dim + dim;
            sum_k += partial_grad_k[partial_index];
            sum_v += partial_grad_v[partial_index];
        }
        grad_k[output_base + dim] = sum_k;
        grad_v[output_base + dim] = sum_v;
    }
}
