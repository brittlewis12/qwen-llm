#include <metal_stdlib>
using namespace metal;

constant uint DS4_CONNECTIONS = 4;
constant uint DS4_PARAMETERS = 24;
constant uint DS4_SINKHORN_ITERATIONS = 20;

struct ds4_clamped_swiglu_args {
    uint n;
    float clamp;
};

struct ds4_route_args {
    uint expert_count;
    uint top_k;
    uint token_id;
    uint vocab_size;
    float routed_scale;
};

constant int DS4_ROUTE_PENDING = 0;
constant int DS4_ROUTE_READY = 1;
constant int DS4_ROUTE_NONFINITE_LOGIT = -1;
constant int DS4_ROUTE_NONFINITE_BIAS = -2;
constant int DS4_ROUTE_INVALID_TOKEN = -3;
constant int DS4_ROUTE_INVALID_EXPERT = -4;
constant int DS4_ROUTE_DUPLICATE_EXPERT = -5;
constant int DS4_ROUTE_NONFINITE_WEIGHT = -6;

inline float ds4_router_score_exact(float value) {
    float softplus;
    if (value > 20.0f) {
        softplus = value;
    } else if (value < -20.0f) {
        softplus = exp(value);
    } else {
        softplus = log(1.0f + exp(value));
    }
    return sqrt(softplus);
}

inline void ds4_route_initialize(
        constant ds4_route_args & args,
        device int * expert_ids,
        device float * weights,
        device int * status,
        uint tid) {
    if (tid == 0) {
        status[0] = DS4_ROUTE_PENDING;
        for (uint slot = 0; slot < args.top_k; ++slot) {
            expert_ids[slot] = -1;
            weights[slot] = 0.0f;
        }
    }
}

kernel void kernel_deepseek_v4_route_learned(
        constant ds4_route_args & args [[buffer(0)]],
        device const float * logits [[buffer(1)]],
        device const float * correction_bias [[buffer(2)]],
        device int * expert_ids [[buffer(3)]],
        device float * weights [[buffer(4)]],
        device int * status [[buffer(5)]],
        uint tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float group_scores[8];
    threadgroup uint group_ids[8];
    threadgroup uint selected_ids[6];
    threadgroup uint route_error;
    ds4_route_initialize(args, expert_ids, weights, status, tid);
    const bool active = tid < args.expert_count;
    const float logit = active ? logits[tid] : 0.0f;
    const float bias = active ? correction_bias[tid] : 0.0f;
    const float unbiased_score = active && isfinite(logit)
        ? ds4_router_score_exact(logit)
        : 0.0f;
    float selection_score = unbiased_score + bias;
    const uint local_error = !active ? 0u
        : (!isfinite(logit) || !isfinite(unbiased_score) ? 1u
        : (!isfinite(bias) || !isfinite(selection_score) ? 2u : 0u));
    const uint simd_error = simd_max(local_error);
    if (tiisg == 0) group_ids[sgitg] = simd_error;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const uint group_error = tiisg < 8 ? group_ids[tiisg] : 0u;
        const uint reduced_error = simd_max(group_error);
        if (tiisg == 0) route_error = reduced_error;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (route_error != 0u) {
        if (tid == 0) {
            status[0] = route_error == 1u
                ? DS4_ROUTE_NONFINITE_LOGIT
                : DS4_ROUTE_NONFINITE_BIAS;
        }
        return;
    }

    for (uint slot = 0; slot < args.top_k; ++slot) {
        const float simd_score = simd_max(active ? selection_score : -INFINITY);
        const uint simd_id = simd_min(
            active && selection_score == simd_score ? tid : UINT_MAX
        );
        if (tiisg == 0) {
            group_scores[sgitg] = simd_score;
            group_ids[sgitg] = simd_id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            const float candidate_score = tiisg < 8 ? group_scores[tiisg] : -INFINITY;
            const uint candidate_id = tiisg < 8 ? group_ids[tiisg] : UINT_MAX;
            const float best_score = simd_max(candidate_score);
            const uint best_id = simd_min(
                candidate_score == best_score ? candidate_id : UINT_MAX
            );
            if (tiisg == 0) {
                selected_ids[slot] = best_id;
                expert_ids[slot] = int(best_id);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active && tid == selected_ids[slot]) selection_score = -INFINITY;
    }

    if (tid == 0) {
        float selected_weights[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float sum = 0.0f;
        for (uint slot = 0; slot < args.top_k; ++slot) {
            selected_weights[slot] = ds4_router_score_exact(logits[selected_ids[slot]]);
            sum += selected_weights[slot];
        }
        const float denominator = max(sum, 6.1035156e-5f);
        for (uint slot = 0; slot < args.top_k; ++slot) {
            const float weight = selected_weights[slot] / denominator * args.routed_scale;
            if (!isfinite(weight)) {
                status[0] = DS4_ROUTE_NONFINITE_WEIGHT;
                return;
            }
            weights[slot] = weight;
        }
        status[0] = DS4_ROUTE_READY;
    }
}

kernel void kernel_deepseek_v4_route_hash(
        constant ds4_route_args & args [[buffer(0)]],
        device const float * logits [[buffer(1)]],
        device const int * token_to_expert [[buffer(2)]],
        device int * expert_ids [[buffer(3)]],
        device float * weights [[buffer(4)]],
        device int * status [[buffer(5)]],
        uint tid [[thread_index_in_threadgroup]]) {
    ds4_route_initialize(args, expert_ids, weights, status, tid);
    if (tid != 0) return;

    if (args.token_id >= args.vocab_size) {
        status[0] = DS4_ROUTE_INVALID_TOKEN;
        return;
    }
    for (uint expert = 0; expert < args.expert_count; ++expert) {
        if (!isfinite(logits[expert])) {
            status[0] = DS4_ROUTE_NONFINITE_LOGIT;
            return;
        }
    }

    int selected[6] = {-1, -1, -1, -1, -1, -1};
    float selected_weights[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const ulong row = (ulong)args.token_id * args.top_k;
    for (uint slot = 0; slot < args.top_k; ++slot) {
        const int expert = token_to_expert[row + slot];
        if (expert < 0 || uint(expert) >= args.expert_count) {
            status[0] = DS4_ROUTE_INVALID_EXPERT;
            return;
        }
        for (uint prior = 0; prior < slot; ++prior) {
            if (selected[prior] == expert) {
                status[0] = DS4_ROUTE_DUPLICATE_EXPERT;
                return;
            }
        }
        selected[slot] = expert;
        selected_weights[slot] = ds4_router_score_exact(logits[expert]);
    }

    float sum = 0.0f;
    for (uint slot = 0; slot < args.top_k; ++slot) {
        sum += selected_weights[slot];
    }
    const float denominator = max(sum, 6.1035156e-5f);
    for (uint slot = 0; slot < args.top_k; ++slot) {
        selected_weights[slot] = selected_weights[slot] / denominator * args.routed_scale;
        if (!isfinite(selected_weights[slot])) {
            status[0] = DS4_ROUTE_NONFINITE_WEIGHT;
            return;
        }
    }
    for (uint slot = 0; slot < args.top_k; ++slot) {
        expert_ids[slot] = selected[slot];
        weights[slot] = selected_weights[slot];
    }
    status[0] = DS4_ROUTE_READY;
}

kernel void kernel_deepseek_v4_clamped_swiglu(
        constant ds4_clamped_swiglu_args & args [[buffer(0)]],
        device const float * gate [[buffer(1)]],
        device const float * up [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.n) return;
    const float clamped_gate = min(gate[index], args.clamp);
    const float clamped_up = clamp(up[index], -args.clamp, args.clamp);
    output[index] = clamped_gate / (1.0f + exp(-clamped_gate)) * clamped_up;
}

struct ds4_position_zero_attention_args {
    uint head_count;
    uint head_dim;
    float scale;
};

struct ds4_attention_cache_roundtrip_args {
    uint width;
    uint rotary_dim;
};

struct ds4_rope_tail_args {
    uint head_count;
    uint head_dim;
    uint rotary_dim;
    uint position;
    uint inverse;
    uint yarn;
    float theta;
    float frequency_scale;
    float correction_low;
    float correction_high;
};

struct ds4_local_attention_args {
    uint head_count;
    uint head_dim;
    uint window;
    uint raw_count;
    uint raw_start;
    uint compressed_count;
    float scale;
};

struct ds4_packed_attention_args {
    uint head_count;
    uint head_dim;
    uint n_tokens;
    uint compression_ratio;
    uint start_position;
    uint window;
    float scale;
};

struct ds4_tiled_dense_attention_args {
    uint head_count;
    uint head_dim;
    uint query_count;
    uint query_token_offset;
    uint chunk_start_position;
    uint window;
    uint compression_ratio;
    float scale;
};

struct ds4_packed_selected_attention_args {
    uint head_count;
    uint head_dim;
    uint query_count;
    uint query_token_offset;
    uint chunk_start_position;
    uint window;
    uint selected_slots;
    uint compressed_capacity;
    float scale;
};

struct ds4_copy_u16_args {
    uint n;
};

struct ds4_compressor_frontier_args {
    uint width;
    uint row_offset;
};

struct ds4_compressor_pool_args {
    uint ratio;
    uint head_dim;
    uint width;
    uint rows;
};

struct ds4_compressor_roll_args {
    uint width;
};

struct ds4_hadamard_rows_args {
    uint row_count;
};

struct ds4_scale_args {
    uint count;
    float scale;
};

struct ds4_indexer_score_args {
    uint head_count;
    uint head_dim;
    uint row_capacity;
    uint query_count;
};

struct ds4_indexer_select_args {
    uint row_capacity;
    uint top_k;
    uint query_count;
    uint emit_ranked;
};

struct ds4_selected_attention_args {
    uint head_count;
    uint head_dim;
    uint window;
    uint raw_count;
    uint raw_start;
    uint compressed_count;
    uint selected_slots;
    float scale;
};

struct ds4_hc_batch_args {
    uint hidden_size;
    uint n_tokens;
};

struct ds4_hc_controls_batch_args {
    uint n_tokens;
    float eps;
};

struct ds4_group_pack_args {
    uint n_tokens;
    uint row_width;
    uint group_width;
    uint group;
};

struct ds4_hash_gather_args {
    uint n_tokens;
    uint top_k;
    uint vocab_size;
};

static inline float ds4_bf16_roundtrip(float value) {
    uint bits = as_type<uint>(value);
    bits += 0x00007fffu + ((bits >> 16) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float ds4_power_of_two(int exponent) {
    return as_type<float>(uint(exponent + 127) << 23);
}

static inline float ds4_e4m3fn_value(uint index) {
    const uint exponent = (index >> 3) & 0x0fu;
    const uint mantissa = index & 0x07u;
    if (exponent == 0u) {
        return float(mantissa) * 0.001953125f;
    }
    return float(8u + mantissa) * ds4_power_of_two(int(exponent) - 10);
}

static inline float ds4_e4m3fn_roundtrip(float value) {
    const float sign = value < 0.0f ? -1.0f : 1.0f;
    const float absolute = min(abs(value), 448.0f);
    uint low = 0u;
    uint high = 126u;
    while (low < high) {
        const uint middle = (low + high + 1u) / 2u;
        if (ds4_e4m3fn_value(middle) <= absolute) {
            low = middle;
        } else {
            high = middle - 1u;
        }
    }
    uint best = low;
    if (best < 126u) {
        const float midpoint =
            (ds4_e4m3fn_value(best) + ds4_e4m3fn_value(best + 1u)) * 0.5f;
        if (absolute > midpoint || (absolute == midpoint && ((best + 1u) & 1u) == 0u)) {
            best += 1u;
        }
    }
    return sign * ds4_e4m3fn_value(best);
}

static inline float ds4_attention_cache_scale(float maximum) {
    const uint bits = as_type<uint>(maximum);
    const int maximum_exponent = int((bits >> 23) & 0xffu) - 127;
    int scale_exponent = maximum_exponent - 8;
    float scale = ds4_power_of_two(scale_exponent);
    if (maximum > 448.0f * scale) {
        scale = ds4_power_of_two(++scale_exponent);
    }
    return scale;
}

kernel void kernel_deepseek_v4_attention_cache_roundtrip(
        constant ds4_attention_cache_roundtrip_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device float * output [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    const uint nope_dim = args.width - args.rotary_dim;
    for (uint block_start = 0u; block_start < nope_dim; block_start += 64u) {
        float maximum = 1.0e-4f;
        for (uint offset = 0u; offset < 64u; ++offset) {
            const uint dimension = block_start + offset;
            const float value = ds4_bf16_roundtrip(input[dimension]);
            output[dimension] = value;
            maximum = max(maximum, abs(value));
        }
        const float scale = ds4_attention_cache_scale(maximum);
        for (uint offset = 0u; offset < 64u; ++offset) {
            const uint dimension = block_start + offset;
            const float normalized = clamp(output[dimension] / scale, -448.0f, 448.0f);
            output[dimension] = ds4_e4m3fn_roundtrip(normalized) * scale;
        }
    }
    for (uint dimension = nope_dim; dimension < args.width; ++dimension) {
        output[dimension] = ds4_bf16_roundtrip(input[dimension]);
    }
}

kernel void kernel_deepseek_v4_rope_tail_adjacent_in_place(
        constant ds4_rope_tail_args & args [[buffer(0)]],
        device float * values [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
    const uint pairs_per_head = args.rotary_dim / 2u;
    const uint pair_count = args.head_count * pairs_per_head;
    if (index >= pair_count) return;
    const uint head = index / pairs_per_head;
    const uint pair = index % pairs_per_head;
    const uint relative = pair * 2u;
    const uint tail = head * args.head_dim + args.head_dim - args.rotary_dim;
    const uint first_index = tail + relative;
    const uint second_index = first_index + 1u;

    const float extrapolated = float(args.position)
        * pow(args.theta, -float(relative) / float(args.rotary_dim));
    float angle = extrapolated;
    if (args.yarn != 0u) {
        const float interpolated = args.frequency_scale * extrapolated;
        const float ramp = 1.0f - clamp(
            (float(pair) - args.correction_low)
                / max(0.001f, args.correction_high - args.correction_low),
            0.0f,
            1.0f);
        angle = interpolated * (1.0f - ramp) + extrapolated * ramp;
    }
    const float cosine = cos(angle);
    float sine = sin(angle);
    if (args.inverse != 0u) sine = -sine;
    const float first = values[first_index];
    const float second = values[second_index];
    values[first_index] = first * cosine - second * sine;
    values[second_index] = first * sine + second * cosine;
}

kernel void kernel_deepseek_v4_dense_sink_attention_f16(
        constant ds4_local_attention_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const half * raw_cache [[buffer(2)]],
        device const half * compressed_cache [[buffer(3)]],
        device const float * sinks [[buffer(4)]],
        device float * output [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    const uint width = args.head_count * args.head_dim;
    if (index >= width) return;
    const uint head = index / args.head_dim;
    const uint dimension = index % args.head_dim;
    const uint query_start = head * args.head_dim;
    float maximum = sinks[head];

    for (uint row = 0u; row < args.raw_count; ++row) {
        const uint logical_position = args.raw_start + row;
        const uint cache_start = (logical_position % args.window) * args.head_dim;
        float score = 0.0f;
        for (uint inner = 0u; inner < args.head_dim; ++inner) {
            score += queries[query_start + inner] * float(raw_cache[cache_start + inner]);
        }
        maximum = max(maximum, score * args.scale);
    }
    for (uint row = 0u; row < args.compressed_count; ++row) {
        const uint cache_start = row * args.head_dim;
        float score = 0.0f;
        for (uint inner = 0u; inner < args.head_dim; ++inner) {
            score += queries[query_start + inner] * float(compressed_cache[cache_start + inner]);
        }
        maximum = max(maximum, score * args.scale);
    }

    float denominator = exp(sinks[head] - maximum);
    float value = 0.0f;
    for (uint row = 0u; row < args.raw_count; ++row) {
        const uint logical_position = args.raw_start + row;
        const uint cache_start = (logical_position % args.window) * args.head_dim;
        float score = 0.0f;
        for (uint inner = 0u; inner < args.head_dim; ++inner) {
            score += queries[query_start + inner] * float(raw_cache[cache_start + inner]);
        }
        const float mass = exp(score * args.scale - maximum);
        denominator += mass;
        value += float(raw_cache[cache_start + dimension]) * mass;
    }
    for (uint row = 0u; row < args.compressed_count; ++row) {
        const uint cache_start = row * args.head_dim;
        float score = 0.0f;
        for (uint inner = 0u; inner < args.head_dim; ++inner) {
            score += queries[query_start + inner] * float(compressed_cache[cache_start + inner]);
        }
        const float mass = exp(score * args.scale - maximum);
        denominator += mass;
        value += float(compressed_cache[cache_start + dimension]) * mass;
    }
    output[index] = value / denominator;
}

kernel void kernel_deepseek_v4_copy_u16(
        constant ds4_copy_u16_args & args [[buffer(0)]],
        device const ushort * source [[buffer(1)]],
        device ushort * destination [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.n) return;
    destination[index] = source[index];
}

kernel void kernel_deepseek_v4_packed_dense_sink_attention_f16(
        constant ds4_packed_attention_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const half * raw_cache [[buffer(2)]],
        device const half * preserved_raw_cache [[buffer(3)]],
        device const half * compressed_cache [[buffer(4)]],
        device const float * sinks [[buffer(5)]],
        device float * output [[buffer(6)]],
        threadgroup float * masses [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint token = group.x;
    const uint head = group.y;
    if (token >= args.n_tokens || head >= args.head_count) return;
    const uint absolute_position = args.start_position + token;
    const uint visible_end = absolute_position + 1u;
    const uint raw_count = min(visible_end, args.window);
    const uint raw_start = visible_end - raw_count;
    const uint compressed_count = args.compression_ratio == 0u
        ? 0u
        : visible_end / args.compression_ratio;
    const uint row_count = raw_count + compressed_count;
    const uint query_start = (token * args.head_count + head) * args.head_dim;

    if (tid < row_count) {
        const bool compressed = tid >= raw_count;
        const uint row = compressed ? tid - raw_count : tid;
        const uint logical_position = raw_start + row;
        device const half * cache = compressed
            ? compressed_cache
            : (logical_position < args.start_position ? preserved_raw_cache : raw_cache);
        const uint cache_start = compressed
            ? row * args.head_dim
            : (logical_position % args.window) * args.head_dim;
        float score = 0.0f;
        for (uint dimension = 0u; dimension < args.head_dim; ++dimension) {
            score += queries[query_start + dimension] * float(cache[cache_start + dimension]);
        }
        masses[tid] = score * args.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float maximum = sinks[head];
        for (uint row = 0u; row < row_count; ++row) {
            maximum = max(maximum, masses[row]);
        }
        float denominator = exp(sinks[head] - maximum);
        for (uint row = 0u; row < row_count; ++row) {
            const float mass = exp(masses[row] - maximum);
            masses[row] = mass;
            denominator += mass;
        }
        masses[row_count] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < args.head_dim) {
        float value = 0.0f;
        for (uint row = 0u; row < raw_count; ++row) {
            const uint logical_position = raw_start + row;
            device const half * cache = logical_position < args.start_position
                ? preserved_raw_cache
                : raw_cache;
            const uint cache_start = (logical_position % args.window) * args.head_dim;
            value += float(cache[cache_start + tid]) * masses[row];
        }
        for (uint row = 0u; row < compressed_count; ++row) {
            value += float(compressed_cache[row * args.head_dim + tid])
                * masses[raw_count + row];
        }
        output[query_start + tid] = value / masses[row_count];
    }
}

kernel void kernel_deepseek_v4_tiled_dense_sink_attention_f16(
        constant ds4_tiled_dense_attention_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const half * raw_cache [[buffer(2)]],
        device const half * preserved_raw_cache [[buffer(3)]],
        device const half * compressed_cache [[buffer(4)]],
        device const float * sinks [[buffer(5)]],
        device float * output [[buffer(6)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tile_rows = 512u;
    threadgroup float * scores = scratch;
    threadgroup float * masses = scratch + tile_rows;
    const uint local_query = group.x;
    const uint head = group.y;
    if (local_query >= args.query_count || head >= args.head_count) return;

    const uint token = args.query_token_offset + local_query;
    const uint absolute_position = args.chunk_start_position + token;
    const uint visible_end = absolute_position + 1u;
    const uint raw_count = min(visible_end, args.window);
    const uint raw_start = visible_end - raw_count;
    const uint compressed_count = visible_end / args.compression_ratio;
    const uint row_count = raw_count + compressed_count;
    const uint query_start = (token * args.head_count + head) * args.head_dim;
    float maximum = sinks[head];

    for (uint tile_start = 0u; tile_start < row_count; tile_start += tile_rows) {
        const uint tile_count = min(tile_rows, row_count - tile_start);
        if (tid < tile_count) {
            const uint attention_row = tile_start + tid;
            const bool compressed = attention_row >= raw_count;
            const uint row = compressed ? attention_row - raw_count : attention_row;
            const uint logical_position = raw_start + row;
            device const half * cache = compressed
                ? compressed_cache
                : (logical_position < args.chunk_start_position
                    ? preserved_raw_cache
                    : raw_cache);
            const uint cache_start = compressed
                ? row * args.head_dim
                : (logical_position % args.window) * args.head_dim;
            float score = 0.0f;
            for (uint dimension = 0u; dimension < args.head_dim; ++dimension) {
                score += queries[query_start + dimension]
                    * float(cache[cache_start + dimension]);
            }
            scores[tid] = score * args.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            for (uint row = 0u; row < tile_count; ++row) {
                maximum = max(maximum, scores[row]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float denominator = tid == 0u ? exp(sinks[head] - maximum) : 0.0f;
    float value = 0.0f;
    for (uint tile_start = 0u; tile_start < row_count; tile_start += tile_rows) {
        const uint tile_count = min(tile_rows, row_count - tile_start);
        if (tid < tile_count) {
            const uint attention_row = tile_start + tid;
            const bool compressed = attention_row >= raw_count;
            const uint row = compressed ? attention_row - raw_count : attention_row;
            const uint logical_position = raw_start + row;
            device const half * cache = compressed
                ? compressed_cache
                : (logical_position < args.chunk_start_position
                    ? preserved_raw_cache
                    : raw_cache);
            const uint cache_start = compressed
                ? row * args.head_dim
                : (logical_position % args.window) * args.head_dim;
            float score = 0.0f;
            for (uint dimension = 0u; dimension < args.head_dim; ++dimension) {
                score += queries[query_start + dimension]
                    * float(cache[cache_start + dimension]);
            }
            scores[tid] = score * args.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            for (uint row = 0u; row < tile_count; ++row) {
                const float mass = exp(scores[row] - maximum);
                masses[row] = mass;
                denominator += mass;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < args.head_dim) {
            for (uint tile_row = 0u; tile_row < tile_count; ++tile_row) {
                const uint attention_row = tile_start + tile_row;
                const bool compressed = attention_row >= raw_count;
                const uint row = compressed ? attention_row - raw_count : attention_row;
                const uint logical_position = raw_start + row;
                device const half * cache = compressed
                    ? compressed_cache
                    : (logical_position < args.chunk_start_position
                        ? preserved_raw_cache
                        : raw_cache);
                const uint cache_start = compressed
                    ? row * args.head_dim
                    : (logical_position % args.window) * args.head_dim;
                value += float(cache[cache_start + tid]) * masses[tile_row];
            }
        }
    }
    if (tid == 0u) scores[0] = denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < args.head_dim) {
        output[query_start + tid] = value / scores[0];
    }
}

kernel void kernel_deepseek_v4_packed_selected_sink_attention_f16(
        constant ds4_packed_selected_attention_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const half * raw_cache [[buffer(2)]],
        device const half * preserved_raw_cache [[buffer(3)]],
        device const half * compressed_cache [[buffer(4)]],
        device const int * selected_ids [[buffer(5)]],
        device const int * selected_counts [[buffer(6)]],
        device const int * visible_counts [[buffer(7)]],
        device const float * sinks [[buffer(8)]],
        device float * output [[buffer(9)]],
        threadgroup float * masses [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint local_query = group.x;
    const uint head = group.y;
    if (local_query >= args.query_count || head >= args.head_count) return;
    const uint token = args.query_token_offset + local_query;
    const uint absolute_position = args.chunk_start_position + token;
    const uint visible_end = absolute_position + 1u;
    const uint raw_count = min(visible_end, args.window);
    const uint raw_start = visible_end - raw_count;
    const int selected_i = selected_counts[local_query];
    const int visible_i = visible_counts[local_query];
    const uint selected_count = selected_i > 0
        ? min(uint(selected_i), args.selected_slots)
        : 0u;
    const uint visible_count = visible_i > 0 ? uint(visible_i) : 0u;
    const uint row_count = raw_count + selected_count;
    const uint query_start = (token * args.head_count + head) * args.head_dim;
    const uint ids_base = local_query * args.selected_slots;

    if (tid < row_count) {
        const bool compressed = tid >= raw_count;
        const uint row = compressed ? tid - raw_count : tid;
        const int selected_id = compressed ? selected_ids[ids_base + row] : -1;
        const bool valid_selected = compressed && selected_id >= 0
            && uint(selected_id) < visible_count
            && uint(selected_id) < args.compressed_capacity;
        const uint logical_position = raw_start + row;
        device const half * cache = compressed
            ? compressed_cache
            : (logical_position < args.chunk_start_position
                ? preserved_raw_cache
                : raw_cache);
        const uint cache_start = compressed
            ? (valid_selected ? uint(selected_id) * args.head_dim : 0u)
            : (logical_position % args.window) * args.head_dim;
        float score = -INFINITY;
        if (!compressed || valid_selected) {
            score = 0.0f;
            for (uint dimension = 0u; dimension < args.head_dim; ++dimension) {
                score += queries[query_start + dimension] * float(cache[cache_start + dimension]);
            }
            score *= args.scale;
        }
        masses[tid] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float maximum = sinks[head];
        for (uint row = 0u; row < row_count; ++row) {
            maximum = max(maximum, masses[row]);
        }
        float denominator = exp(sinks[head] - maximum);
        for (uint row = 0u; row < row_count; ++row) {
            const float mass = exp(masses[row] - maximum);
            masses[row] = mass;
            denominator += mass;
        }
        masses[row_count] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < args.head_dim) {
        float value = 0.0f;
        for (uint row = 0u; row < raw_count; ++row) {
            const uint logical_position = raw_start + row;
            device const half * cache = logical_position < args.chunk_start_position
                ? preserved_raw_cache
                : raw_cache;
            const uint cache_start = (logical_position % args.window) * args.head_dim;
            value += float(cache[cache_start + tid]) * masses[row];
        }
        for (uint slot = 0u; slot < selected_count; ++slot) {
            const int selected_id = selected_ids[ids_base + slot];
            if (selected_id < 0 || uint(selected_id) >= visible_count
                    || uint(selected_id) >= args.compressed_capacity) continue;
            const uint cache_start = uint(selected_id) * args.head_dim;
            value += float(compressed_cache[cache_start + tid]) * masses[raw_count + slot];
        }
        output[query_start + tid] = value / masses[row_count];
    }
}

kernel void kernel_deepseek_v4_compressor_frontier_write(
        constant ds4_compressor_frontier_args & args [[buffer(0)]],
        device const float * projected_kv [[buffer(1)]],
        device const float * projected_score [[buffer(2)]],
        device const float * ape [[buffer(3)]],
        device float * kv_state [[buffer(4)]],
        device float * score_state [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.width) return;
    const uint destination = args.row_offset + index;
    kv_state[destination] = projected_kv[index];
    score_state[destination] = projected_score[index] + ape[index];
}

kernel void kernel_deepseek_v4_compressor_pool(
        constant ds4_compressor_pool_args & args [[buffer(0)]],
        device const float * kv_state [[buffer(1)]],
        device const float * score_state [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint dimension [[thread_position_in_grid]]) {
    if (dimension >= args.head_dim) return;
    float maximum = -INFINITY;
    if (args.ratio == 4u) {
        for (uint row = 0u; row < 4u; ++row) {
            maximum = max(maximum, score_state[row * args.width + dimension]);
            maximum = max(
                maximum,
                score_state[(4u + row) * args.width + args.head_dim + dimension]);
        }
    } else {
        for (uint row = 0u; row < args.rows; ++row) {
            maximum = max(maximum, score_state[row * args.width + dimension]);
        }
    }

    float weighted = 0.0f;
    float denominator = 0.0f;
    if (args.ratio == 4u) {
        for (uint row = 0u; row < 4u; ++row) {
            const uint previous = row * args.width + dimension;
            const uint current = (4u + row) * args.width + args.head_dim + dimension;
            const float previous_mass = isfinite(score_state[previous])
                ? exp(score_state[previous] - maximum)
                : 0.0f;
            const float current_mass = isfinite(score_state[current])
                ? exp(score_state[current] - maximum)
                : 0.0f;
            denominator += previous_mass + current_mass;
            weighted += kv_state[previous] * previous_mass + kv_state[current] * current_mass;
        }
    } else {
        for (uint row = 0u; row < args.rows; ++row) {
            const uint source = row * args.width + dimension;
            const float mass = isfinite(score_state[source])
                ? exp(score_state[source] - maximum)
                : 0.0f;
            denominator += mass;
            weighted += kv_state[source] * mass;
        }
    }
    output[dimension] = weighted / denominator;
}

kernel void kernel_deepseek_v4_compressor_roll_ratio4(
        constant ds4_compressor_roll_args & args [[buffer(0)]],
        device float * kv_state [[buffer(1)]],
        device float * score_state [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    const uint group_elements = 4u * args.width;
    if (index >= group_elements) return;
    kv_state[index] = kv_state[group_elements + index];
    score_state[index] = score_state[group_elements + index];
}

kernel void kernel_deepseek_v4_hadamard_128_in_place(
        device float * values [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    for (uint span = 1u; span < 128u; span <<= 1u) {
        for (uint start = 0u; start < 128u; start += span << 1u) {
            for (uint offset = 0u; offset < span; ++offset) {
                const uint first = start + offset;
                const uint second = first + span;
                const float a = values[first];
                const float b = values[second];
                values[first] = a + b;
                values[second] = a - b;
            }
        }
    }
    const float normalization = rsqrt(128.0f);
    for (uint element = 0u; element < 128u; ++element) {
        values[element] *= normalization;
    }
}

kernel void kernel_deepseek_v4_hadamard_128_rows_in_place(
        constant ds4_hadamard_rows_args & args [[buffer(0)]],
        device float * values [[buffer(1)]],
        uint row [[thread_position_in_grid]]) {
    if (row >= args.row_count) return;
    const uint base = row * 128u;
    for (uint span = 1u; span < 128u; span <<= 1u) {
        for (uint start = 0u; start < 128u; start += span << 1u) {
            for (uint offset = 0u; offset < span; ++offset) {
                const uint first = base + start + offset;
                const uint second = first + span;
                const float a = values[first];
                const float b = values[second];
                values[first] = a + b;
                values[second] = a - b;
            }
        }
    }
    const float normalization = rsqrt(128.0f);
    for (uint element = 0u; element < 128u; ++element) {
        values[base + element] *= normalization;
    }
}

kernel void kernel_deepseek_v4_scale_f32_in_place(
        constant ds4_scale_args & args [[buffer(0)]],
        device float * values [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.count) return;
    values[index] *= args.scale;
}

kernel void kernel_deepseek_v4_lightning_indexer_scores_f16(
        constant ds4_indexer_score_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const float * head_weights [[buffer(2)]],
        device const half * keys [[buffer(3)]],
        device const int * visible_counts [[buffer(4)]],
        device float * scores [[buffer(5)]],
        uint2 index [[thread_position_in_grid]]) {
    const uint row = index.x;
    const uint query = index.y;
    if (row >= args.row_capacity || query >= args.query_count) return;
    const uint score_index = query * args.row_capacity + row;
    const int visible = visible_counts[query];
    if (visible < 0 || row >= uint(visible)) {
        scores[score_index] = -INFINITY;
        return;
    }

    const uint query_base = query * args.head_count * args.head_dim;
    const uint weight_base = query * args.head_count;
    const uint key_base = row * args.head_dim;
    float score = 0.0f;
    for (uint head = 0u; head < args.head_count; ++head) {
        float dot = 0.0f;
        const uint head_base = query_base + head * args.head_dim;
        for (uint dimension = 0u; dimension < args.head_dim; ++dimension) {
            dot += queries[head_base + dimension] * float(keys[key_base + dimension]);
        }
        score += max(dot, 0.0f) * head_weights[weight_base + head];
    }
    scores[score_index] = score;
}

kernel void kernel_deepseek_v4_lightning_indexer_scores_f16_cooperative(
        constant ds4_indexer_score_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const float * head_weights [[buffer(2)]],
        device const half * keys [[buffer(3)]],
        device const int * visible_counts [[buffer(4)]],
        device float * scores [[buffer(5)]],
        threadgroup half * staged_keys [[threadgroup(0)]],
        threadgroup float * head_dots [[threadgroup(1)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_group = 8u;
    const uint row = group.x * rows_per_group + uint(simdgroup);
    const uint query = group.y;
    if (row >= args.row_capacity || query >= args.query_count) return;
    const uint score_index = query * args.row_capacity + row;
    const int visible = visible_counts[query];
    if (visible < 0 || row >= uint(visible)) {
        if (lane == 0) scores[score_index] = -INFINITY;
        return;
    }

    threadgroup half * key_row = staged_keys + uint(simdgroup) * 128u;
    for (uint dimension = uint(lane); dimension < 128u; dimension += 32u) {
        key_row[dimension] = keys[row * 128u + dimension];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint query_base = query * 64u * 128u;
    threadgroup float * dots = head_dots + uint(simdgroup) * 64u;
    for (uint head_group = 0u; head_group < 2u; ++head_group) {
        const uint head = uint(lane) + head_group * 32u;
        const uint head_base = query_base + head * 128u;
        float dot = 0.0f;
        for (uint dimension = 0u; dimension < 128u; ++dimension) {
            dot += queries[head_base + dimension] * float(key_row[dimension]);
        }
        dots[head] = dot;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        const uint weight_base = query * 64u;
        float score = 0.0f;
        for (uint head = 0u; head < 64u; ++head) {
            score += max(dots[head], 0.0f) * head_weights[weight_base + head];
        }
        scores[score_index] = score;
    }
}

kernel void kernel_deepseek_v4_select_top_k_f32(
        constant ds4_indexer_select_args & args [[buffer(0)]],
        device const float * scores [[buffer(1)]],
        device const int * visible_counts [[buffer(2)]],
        device int * selected_mask [[buffer(3)]],
        device int * ranked_ids [[buffer(4)]],
        device int * cache_order_ids [[buffer(5)]],
        device int * selected_counts [[buffer(6)]],
        device int * status [[buffer(7)]],
        uint query [[thread_position_in_grid]]) {
    if (query >= args.query_count) return;
    const uint mask_base = query * args.row_capacity;
    const uint ids_base = query * args.top_k;
    const int visible_i = visible_counts[query];
    const bool geometry_valid = visible_i > 0 && uint(visible_i) <= args.row_capacity
        && args.top_k > 0u && args.top_k <= args.row_capacity;
    const uint visible = geometry_valid ? uint(visible_i) : 0u;
    const uint selected_count = min(visible, args.top_k);

    for (uint row = 0u; row < args.row_capacity; ++row) {
        selected_mask[mask_base + row] = row < visible ? 1 : 0;
    }
    for (uint slot = 0u; slot < args.top_k; ++slot) {
        if (args.emit_ranked != 0u) ranked_ids[ids_base + slot] = -1;
        cache_order_ids[ids_base + slot] = -1;
    }

    int error = geometry_valid ? 0 : 1;
    if (error == 0) {
        for (uint row = 0u; row < visible; ++row) {
            if (!isfinite(scores[query * args.row_capacity + row])) {
                error = 2;
                break;
            }
        }
    }

    if (error == 0) {
        const uint removals = visible - selected_count;
        if (removals <= selected_count) {
            for (uint removal = 0u; removal < removals; ++removal) {
                int worst = -1;
                float worst_score = INFINITY;
                for (uint row = 0u; row < visible; ++row) {
                    if (selected_mask[mask_base + row] == 0) continue;
                    const float candidate = scores[query * args.row_capacity + row];
                    const bool worse = worst < 0 || candidate < worst_score
                        || (candidate == worst_score && int(row) > worst);
                    if (worse) {
                        worst = int(row);
                        worst_score = candidate;
                    }
                }
                if (worst < 0) {
                    error = 3;
                    break;
                }
                selected_mask[mask_base + uint(worst)] = 0;
            }
        } else {
            for (uint row = 0u; row < visible; ++row) {
                selected_mask[mask_base + row] = 0;
            }
            for (uint selection = 0u; selection < selected_count; ++selection) {
                int best = -1;
                float best_score = -INFINITY;
                for (uint row = 0u; row < visible; ++row) {
                    if (selected_mask[mask_base + row] != 0) continue;
                    const float candidate = scores[query * args.row_capacity + row];
                    const bool better = best < 0 || candidate > best_score
                        || (candidate == best_score && int(row) < best);
                    if (better) {
                        best = int(row);
                        best_score = candidate;
                    }
                }
                if (best < 0) {
                    error = 3;
                    break;
                }
                selected_mask[mask_base + uint(best)] = 1;
            }
        }
    }

    if (error != 0) {
        for (uint row = 0u; row < args.row_capacity; ++row) {
            selected_mask[mask_base + row] = row < selected_count ? 1 : 0;
        }
    }

    uint filled = 0u;
    for (uint row = 0u; row < visible && filled < selected_count; ++row) {
        if (selected_mask[mask_base + row] == 0) continue;
        cache_order_ids[ids_base + filled] = int(row);
        if (args.emit_ranked != 0u) {
            uint insertion = filled;
            if (error == 0) {
                const float candidate = scores[query * args.row_capacity + row];
                for (uint slot = 0u; slot < filled; ++slot) {
                    const int current_id = ranked_ids[ids_base + slot];
                    const float current = scores[query * args.row_capacity + uint(current_id)];
                    if (candidate > current || (candidate == current && int(row) < current_id)) {
                        insertion = slot;
                        break;
                    }
                }
                for (uint slot = filled; slot > insertion; --slot) {
                    ranked_ids[ids_base + slot] = ranked_ids[ids_base + slot - 1u];
                }
            }
            ranked_ids[ids_base + insertion] = int(row);
        }
        ++filled;
    }
    selected_counts[query] = int(filled);
    status[query] = error;
}

static inline uint ds4_selector_order_key(float value) {
    uint bits = as_type<uint>(value);
    // Match -ffast-math comparisons: subnormals and both signed zeros tie.
    if ((bits & 0x7f800000u) == 0u) bits = 0u;
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

template <bool Radix4>
__attribute__((always_inline)) inline void ds4_select_top_k_parallel_impl(
        constant ds4_indexer_select_args & args,
        device const float * scores,
        device const int * visible_counts,
        device int * selected_mask,
        device int * ranked_ids,
        device int * cache_order_ids,
        device int * selected_counts,
        device int * status,
        threadgroup uint * lane_scratch,
        threadgroup uint * shared,
        uint lane,
        uint query,
        uint width,
        ushort simdgroup,
        ushort simd_lane) {
    if (query >= args.query_count) return;
    const uint simdgroup_count = (width + 31u) / 32u;
    const uint mask_base = query * args.row_capacity;
    const uint ids_base = query * args.top_k;
    const int visible_i = visible_counts[query];
    const bool geometry_valid = visible_i > 0 && uint(visible_i) <= args.row_capacity
        && args.top_k > 0u && args.top_k <= args.row_capacity;
    const uint visible = geometry_valid ? uint(visible_i) : 0u;
    const uint selected_count = min(visible, args.top_k);

    for (uint row = lane; row < args.row_capacity; row += width) {
        selected_mask[mask_base + row] = 0;
    }
    for (uint slot = lane; slot < args.top_k; slot += width) {
        if (args.emit_ranked != 0u) ranked_ids[ids_base + slot] = -1;
        cache_order_ids[ids_base + slot] = -1;
    }
    int local_error = geometry_valid ? 0 : 1;
    if (local_error == 0) {
        for (uint row = lane; row < visible; row += width) {
            if (!isfinite(scores[query * args.row_capacity + row])) {
                local_error = 2;
                break;
            }
        }
    }
    const uint simd_error = simd_max(uint(local_error));
    if (simd_lane == 0u) {
        lane_scratch[simdgroup] = simd_error;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (simdgroup == 0u) {
        const uint group_error = simd_lane < simdgroup_count
            ? lane_scratch[simd_lane]
            : 0u;
        const uint reduced_error = simd_max(group_error);
        if (simd_lane == 0u) shared[0] = reduced_error;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int error = int(shared[0]);

    if (error != 0) {
        for (uint row = lane; row < args.row_capacity; row += width) {
            selected_mask[mask_base + row] = row < selected_count ? 1 : 0;
        }
        for (uint slot = lane; slot < args.top_k; slot += width) {
            const int id = slot < selected_count ? int(slot) : -1;
            if (args.emit_ranked != 0u) ranked_ids[ids_base + slot] = id;
            cache_order_ids[ids_base + slot] = id;
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (lane == 0u) {
            selected_counts[query] = int(selected_count);
            status[query] = error;
        }
        return;
    }

    uint prefix = 0u;
    uint prefix_mask = 0u;
    uint rank = selected_count;
    if (Radix4) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            uint local_bins[16];
            for (uint bin = 0u; bin < 16u; ++bin) local_bins[bin] = 0u;
            for (uint row = lane; row < visible; row += width) {
                const uint key = ds4_selector_order_key(scores[query * args.row_capacity + row]);
                if ((key & prefix_mask) == prefix) {
                    ++local_bins[(key >> uint(shift)) & 0xfu];
                }
            }
            for (uint bin = 0u; bin < 16u; ++bin) {
                const uint simd_count = simd_sum(local_bins[bin]);
                if (simd_lane == 0u) {
                    lane_scratch[uint(simdgroup) * 16u + bin] = simd_count;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdgroup == 0u && simd_lane < 16u) {
                uint total = 0u;
                for (uint group = 0u; group < simdgroup_count; ++group) {
                    total += lane_scratch[group * 16u + uint(simd_lane)];
                }
                shared[simd_lane] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                uint remaining = rank;
                uint chosen = 0u;
                bool found = false;
                for (int bin = 15; bin >= 0; --bin) {
                    const uint count = shared[uint(bin)];
                    if (remaining <= count) {
                        chosen = uint(bin);
                        found = true;
                        break;
                    }
                    remaining -= count;
                }
                shared[16] = chosen;
                shared[17] = remaining;
                shared[18] = found ? 0u : 3u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            error = int(shared[18]);
            if (error != 0) break;
            prefix |= shared[16] << uint(shift);
            rank = shared[17];
            prefix_mask |= 0xfu << uint(shift);
        }
    } else {
        // Resolve the 1-based kth score by counting one radix bit per pass.
        for (int bit = 31; bit >= 0; --bit) {
            const uint bit_mask = 1u << uint(bit);
            uint local_ones = 0u;
            for (uint row = lane; row < visible; row += width) {
                const uint key = ds4_selector_order_key(scores[query * args.row_capacity + row]);
                if ((key & prefix_mask) == prefix && (key & bit_mask) != 0u) {
                    ++local_ones;
                }
            }
            const uint simd_ones = simd_sum(local_ones);
            if (simd_lane == 0u) lane_scratch[simdgroup] = simd_ones;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdgroup == 0u) {
                const uint group_ones = simd_lane < simdgroup_count
                    ? lane_scratch[simd_lane]
                    : 0u;
                const uint total_ones = simd_sum(group_ones);
                if (simd_lane == 0u) shared[0] = total_ones;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint ones = shared[0];
            if (rank <= ones) {
                prefix |= bit_mask;
            } else {
                rank -= ones;
            }
            prefix_mask |= bit_mask;
        }
    }
    if (error != 0) {
        for (uint row = lane; row < args.row_capacity; row += width) {
            selected_mask[mask_base + row] = row < selected_count ? 1 : 0;
        }
        for (uint slot = lane; slot < args.top_k; slot += width) {
            const int id = slot < selected_count ? int(slot) : -1;
            if (args.emit_ranked != 0u) ranked_ids[ids_base + slot] = id;
            cache_order_ids[ids_base + slot] = id;
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (lane == 0u) {
            selected_counts[query] = int(selected_count);
            status[query] = error;
        }
        return;
    }

    const uint threshold_key = prefix;
    const uint threshold_take = rank;
    const uint chunk = (visible + width - 1u) / width;
    const uint chunk_start = min(lane * chunk, visible);
    const uint chunk_end = min(chunk_start + chunk, visible);
    // Ordered chunks make threshold ties and final compaction stable by row ID.
    uint local_equal = 0u;
    for (uint row = chunk_start; row < chunk_end; ++row) {
        const uint key = ds4_selector_order_key(scores[query * args.row_capacity + row]);
        if (key == threshold_key) ++local_equal;
    }
    lane_scratch[lane] = local_equal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        uint equal_total = 0u;
        for (uint index = 0u; index < width; ++index) {
            const uint count = lane_scratch[index];
            lane_scratch[index] = equal_total;
            equal_total += count;
        }
        shared[0] = equal_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint equal_before = lane_scratch[lane];

    uint local_equal_seen = 0u;
    uint local_selected = 0u;
    for (uint row = chunk_start; row < chunk_end; ++row) {
        const uint key = ds4_selector_order_key(scores[query * args.row_capacity + row]);
        const bool at_threshold = key == threshold_key;
        const bool selected = key > threshold_key
            || (at_threshold && equal_before + local_equal_seen < threshold_take);
        if (at_threshold) ++local_equal_seen;
        if (selected) ++local_selected;
    }
    lane_scratch[lane] = local_selected;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        uint selected_total = 0u;
        for (uint index = 0u; index < width; ++index) {
            const uint count = lane_scratch[index];
            lane_scratch[index] = selected_total;
            selected_total += count;
        }
        shared[0] = selected_total;
        shared[1] = selected_total == selected_count ? 0u : 3u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    error = int(shared[1]);
    if (error != 0) {
        for (uint row = lane; row < args.row_capacity; row += width) {
            selected_mask[mask_base + row] = row < selected_count ? 1 : 0;
        }
        for (uint slot = lane; slot < args.top_k; slot += width) {
            const int id = slot < selected_count ? int(slot) : -1;
            if (args.emit_ranked != 0u) ranked_ids[ids_base + slot] = id;
            cache_order_ids[ids_base + slot] = id;
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (lane == 0u) {
            selected_counts[query] = int(selected_count);
            status[query] = error;
        }
        return;
    }

    uint output_slot = lane_scratch[lane];
    local_equal_seen = 0u;
    for (uint row = chunk_start; row < chunk_end; ++row) {
        const uint key = ds4_selector_order_key(scores[query * args.row_capacity + row]);
        const bool at_threshold = key == threshold_key;
        const bool selected = key > threshold_key
            || (at_threshold && equal_before + local_equal_seen < threshold_take);
        if (at_threshold) ++local_equal_seen;
        if (!selected) continue;
        selected_mask[mask_base + row] = 1;
        cache_order_ids[ids_base + output_slot] = int(row);
        ++output_slot;
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (args.emit_ranked != 0u) {
        for (uint slot = lane; slot < selected_count; slot += width) {
            ranked_ids[ids_base + slot] = cache_order_ids[ids_base + slot];
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (lane == 0u) {
            for (uint slot = 1u; slot < selected_count; ++slot) {
                const int candidate_id = ranked_ids[ids_base + slot];
                const uint candidate_key = ds4_selector_order_key(
                    scores[query * args.row_capacity + uint(candidate_id)]
                );
                uint insertion = slot;
                while (insertion > 0u) {
                    const int prior_id = ranked_ids[ids_base + insertion - 1u];
                    const uint prior_key = ds4_selector_order_key(
                        scores[query * args.row_capacity + uint(prior_id)]
                    );
                    if (candidate_key < prior_key
                            || (candidate_key == prior_key && candidate_id > prior_id)) break;
                    ranked_ids[ids_base + insertion] = prior_id;
                    --insertion;
                }
                ranked_ids[ids_base + insertion] = candidate_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    if (lane == 0u) {
        selected_counts[query] = int(selected_count);
        status[query] = 0;
    }
}

kernel void kernel_deepseek_v4_select_top_k_parallel_f32(
        constant ds4_indexer_select_args & args [[buffer(0)]],
        device const float * scores [[buffer(1)]],
        device const int * visible_counts [[buffer(2)]],
        device int * selected_mask [[buffer(3)]],
        device int * ranked_ids [[buffer(4)]],
        device int * cache_order_ids [[buffer(5)]],
        device int * selected_counts [[buffer(6)]],
        device int * status [[buffer(7)]],
        threadgroup uint * lane_scratch [[threadgroup(0)]],
        threadgroup uint * shared [[threadgroup(1)]],
        uint lane [[thread_index_in_threadgroup]],
        uint query [[threadgroup_position_in_grid]],
        uint width [[threads_per_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    ds4_select_top_k_parallel_impl<false>(
        args, scores, visible_counts, selected_mask, ranked_ids,
        cache_order_ids, selected_counts, status, lane_scratch, shared,
        lane, query, width, simdgroup, simd_lane);
}

kernel void kernel_deepseek_v4_select_top_k_radix4_f32(
        constant ds4_indexer_select_args & args [[buffer(0)]],
        device const float * scores [[buffer(1)]],
        device const int * visible_counts [[buffer(2)]],
        device int * selected_mask [[buffer(3)]],
        device int * ranked_ids [[buffer(4)]],
        device int * cache_order_ids [[buffer(5)]],
        device int * selected_counts [[buffer(6)]],
        device int * status [[buffer(7)]],
        threadgroup uint * lane_scratch [[threadgroup(0)]],
        threadgroup uint * shared [[threadgroup(1)]],
        uint lane [[thread_index_in_threadgroup]],
        uint query [[threadgroup_position_in_grid]],
        uint width [[threads_per_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    ds4_select_top_k_parallel_impl<true>(
        args, scores, visible_counts, selected_mask, ranked_ids,
        cache_order_ids, selected_counts, status, lane_scratch, shared,
        lane, query, width, simdgroup, simd_lane);
}

kernel void kernel_deepseek_v4_selected_sink_attention_f16(
        constant ds4_selected_attention_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const half * raw_cache [[buffer(2)]],
        device const half * compressed_cache [[buffer(3)]],
        device const int * selected_ids [[buffer(4)]],
        device const float * sinks [[buffer(5)]],
        device float * output [[buffer(6)]],
        uint index [[thread_position_in_grid]]) {
    const uint width = args.head_count * args.head_dim;
    if (index >= width) return;
    const uint head = index / args.head_dim;
    const uint dimension = index % args.head_dim;
    const uint query_start = head * args.head_dim;
    float maximum = sinks[head];

    for (uint row = 0u; row < args.raw_count; ++row) {
        const uint logical_position = args.raw_start + row;
        const uint cache_start = (logical_position % args.window) * args.head_dim;
        float score = 0.0f;
        for (uint inner = 0u; inner < args.head_dim; ++inner) {
            score += queries[query_start + inner] * float(raw_cache[cache_start + inner]);
        }
        maximum = max(maximum, score * args.scale);
    }
    for (uint slot = 0u; slot < args.selected_slots; ++slot) {
        const int row = selected_ids[slot];
        if (row < 0 || uint(row) >= args.compressed_count) continue;
        const uint cache_start = uint(row) * args.head_dim;
        float score = 0.0f;
        for (uint inner = 0u; inner < args.head_dim; ++inner) {
            score += queries[query_start + inner] * float(compressed_cache[cache_start + inner]);
        }
        maximum = max(maximum, score * args.scale);
    }

    float denominator = exp(sinks[head] - maximum);
    float value = 0.0f;
    for (uint row = 0u; row < args.raw_count; ++row) {
        const uint logical_position = args.raw_start + row;
        const uint cache_start = (logical_position % args.window) * args.head_dim;
        float score = 0.0f;
        for (uint inner = 0u; inner < args.head_dim; ++inner) {
            score += queries[query_start + inner] * float(raw_cache[cache_start + inner]);
        }
        const float mass = exp(score * args.scale - maximum);
        denominator += mass;
        value += float(raw_cache[cache_start + dimension]) * mass;
    }
    for (uint slot = 0u; slot < args.selected_slots; ++slot) {
        const int row = selected_ids[slot];
        if (row < 0 || uint(row) >= args.compressed_count) continue;
        const uint cache_start = uint(row) * args.head_dim;
        float score = 0.0f;
        for (uint inner = 0u; inner < args.head_dim; ++inner) {
            score += queries[query_start + inner] * float(compressed_cache[cache_start + inner]);
        }
        const float mass = exp(score * args.scale - maximum);
        denominator += mass;
        value += float(compressed_cache[cache_start + dimension]) * mass;
    }
    output[index] = value / denominator;
}

kernel void kernel_deepseek_v4_position_zero_sink_attention(
        constant ds4_position_zero_attention_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const float * kv [[buffer(2)]],
        device const float * sinks [[buffer(3)]],
        device float * output [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    const uint width = args.head_count * args.head_dim;
    if (index >= width) return;
    const uint head = index / args.head_dim;
    const uint query_start = head * args.head_dim;
    float score = 0.0f;
    for (uint dimension = 0; dimension < args.head_dim; ++dimension) {
        score += queries[query_start + dimension] * kv[dimension];
    }
    score *= args.scale;
    const float maximum = max(score, sinks[head]);
    const float kv_mass = exp(score - maximum);
    const float denominator = kv_mass + exp(sinks[head] - maximum);
    output[index] = kv[index % args.head_dim] * (kv_mass / denominator);
}

kernel void kernel_deepseek_v4_hc_repeat(
        constant uint & hidden_size [[buffer(0)]],
        device const float * embedding [[buffer(1)]],
        device float * residual [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= hidden_size * DS4_CONNECTIONS) return;
    residual[index] = embedding[index % hidden_size];
}

kernel void kernel_deepseek_v4_hc_repeat_batch(
        constant ds4_hc_batch_args & args [[buffer(0)]],
        device const float * embeddings [[buffer(1)]],
        device float * residual [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    const uint residual_width = args.hidden_size * DS4_CONNECTIONS;
    const uint total = args.n_tokens * residual_width;
    if (index >= total) return;
    const uint token = index / residual_width;
    const uint dimension = index % args.hidden_size;
    residual[index] = embeddings[token * args.hidden_size + dimension];
}

kernel void kernel_deepseek_v4_hc_controls(
        constant float & eps [[buffer(0)]],
        device const float * mix [[buffer(1)]],
        device const float * scale [[buffer(2)]],
        device const float * base [[buffer(3)]],
        device float * pre [[buffer(4)]],
        device float * post [[buffer(5)]],
        device float * combination [[buffer(6)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    for (uint stream = 0; stream < DS4_CONNECTIONS; ++stream) {
        pre[stream] = 1.0f / (1.0f + exp(-(mix[stream] * scale[0] + base[stream]))) + eps;
        post[stream] = 2.0f / (1.0f + exp(-(mix[4 + stream] * scale[1] + base[4 + stream])));
    }

    // mix[8 + destination + 4*source] and combination[source*4 + destination]
    // are deliberately the same source-major, destination-fast layout.
    for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
        float row_max = -INFINITY;
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            const uint index = source * DS4_CONNECTIONS + destination;
            const float value = mix[8 + index] * scale[2] + base[8 + index];
            combination[index] = value;
            row_max = max(row_max, value);
        }
        float sum = 0.0f;
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            const uint index = source * DS4_CONNECTIONS + destination;
            combination[index] = exp(combination[index] - row_max);
            sum += combination[index];
        }
        const float inverse = 1.0f / sum;
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            const uint index = source * DS4_CONNECTIONS + destination;
            combination[index] = combination[index] * inverse + eps;
        }
    }

    for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
        float sum = 0.0f;
        for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
            sum += combination[source * DS4_CONNECTIONS + destination];
        }
        const float inverse = 1.0f / (sum + eps);
        for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
            combination[source * DS4_CONNECTIONS + destination] *= inverse;
        }
    }
    for (uint iteration = 1; iteration < DS4_SINKHORN_ITERATIONS; ++iteration) {
        for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
            float sum = 0.0f;
            for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
                sum += combination[source * DS4_CONNECTIONS + destination];
            }
            const float inverse = 1.0f / (sum + eps);
            for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
                combination[source * DS4_CONNECTIONS + destination] *= inverse;
            }
        }
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            float sum = 0.0f;
            for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
                sum += combination[source * DS4_CONNECTIONS + destination];
            }
            const float inverse = 1.0f / (sum + eps);
            for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
                combination[source * DS4_CONNECTIONS + destination] *= inverse;
            }
        }
    }
}

kernel void kernel_deepseek_v4_hc_controls_batch(
        constant ds4_hc_controls_batch_args & args [[buffer(0)]],
        device const float * mix [[buffer(1)]],
        device const float * scale [[buffer(2)]],
        device const float * base [[buffer(3)]],
        device float * pre [[buffer(4)]],
        device float * post [[buffer(5)]],
        device float * combination [[buffer(6)]],
        uint token [[thread_position_in_grid]]) {
    if (token >= args.n_tokens) return;
    const uint mix_start = token * DS4_PARAMETERS;
    const uint gate_start = token * DS4_CONNECTIONS;
    const uint combination_start = token * DS4_CONNECTIONS * DS4_CONNECTIONS;
    for (uint stream = 0; stream < DS4_CONNECTIONS; ++stream) {
        pre[gate_start + stream] = 1.0f / (1.0f + exp(-(
            mix[mix_start + stream] * scale[0] + base[stream]))) + args.eps;
        post[gate_start + stream] = 2.0f / (1.0f + exp(-(
            mix[mix_start + 4u + stream] * scale[1] + base[4u + stream])));
    }

    for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
        float row_max = -INFINITY;
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            const uint index = source * DS4_CONNECTIONS + destination;
            const float value = mix[mix_start + 8u + index] * scale[2] + base[8u + index];
            combination[combination_start + index] = value;
            row_max = max(row_max, value);
        }
        float sum = 0.0f;
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            const uint index = source * DS4_CONNECTIONS + destination;
            const uint offset = combination_start + index;
            combination[offset] = exp(combination[offset] - row_max);
            sum += combination[offset];
        }
        const float inverse = 1.0f / sum;
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            const uint index = source * DS4_CONNECTIONS + destination;
            const uint offset = combination_start + index;
            combination[offset] = combination[offset] * inverse + args.eps;
        }
    }

    for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
        float sum = 0.0f;
        for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
            sum += combination[combination_start + source * DS4_CONNECTIONS + destination];
        }
        const float inverse = 1.0f / (sum + args.eps);
        for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
            combination[combination_start + source * DS4_CONNECTIONS + destination] *= inverse;
        }
    }
    for (uint iteration = 1; iteration < DS4_SINKHORN_ITERATIONS; ++iteration) {
        for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
            float sum = 0.0f;
            for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
                sum += combination[combination_start + source * DS4_CONNECTIONS + destination];
            }
            const float inverse = 1.0f / (sum + args.eps);
            for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
                combination[combination_start + source * DS4_CONNECTIONS + destination] *= inverse;
            }
        }
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            float sum = 0.0f;
            for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
                sum += combination[combination_start + source * DS4_CONNECTIONS + destination];
            }
            const float inverse = 1.0f / (sum + args.eps);
            for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
                combination[combination_start + source * DS4_CONNECTIONS + destination] *= inverse;
            }
        }
    }
}

kernel void kernel_deepseek_v4_hc_collapse(
        constant uint & hidden_size [[buffer(0)]],
        device const float * residual [[buffer(1)]],
        device const float * pre [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint dimension [[thread_position_in_grid]]) {
    if (dimension >= hidden_size) return;
    float value = 0.0f;
    for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
        value += residual[source * hidden_size + dimension] * pre[source];
    }
    output[dimension] = value;
}

kernel void kernel_deepseek_v4_hc_collapse_batch(
        constant ds4_hc_batch_args & args [[buffer(0)]],
        device const float * residual [[buffer(1)]],
        device const float * pre [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.hidden_size;
    if (index >= total) return;
    const uint token = index / args.hidden_size;
    const uint dimension = index % args.hidden_size;
    const uint residual_start = token * DS4_CONNECTIONS * args.hidden_size;
    const uint gate_start = token * DS4_CONNECTIONS;
    float value = 0.0f;
    for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
        value += residual[residual_start + source * args.hidden_size + dimension]
            * pre[gate_start + source];
    }
    output[index] = value;
}

kernel void kernel_deepseek_v4_hc_post(
        constant uint & hidden_size [[buffer(0)]],
        device const float * block [[buffer(1)]],
        device const float * residual [[buffer(2)]],
        device const float * post [[buffer(3)]],
        device const float * combination [[buffer(4)]],
        device float * output [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    const uint residual_size = hidden_size * DS4_CONNECTIONS;
    if (index >= residual_size) return;
    const uint destination = index / hidden_size;
    const uint dimension = index % hidden_size;
    float value = block[dimension] * post[destination];
    for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
        value += combination[source * DS4_CONNECTIONS + destination]
            * residual[source * hidden_size + dimension];
    }
    output[index] = value;
}

kernel void kernel_deepseek_v4_hc_post_batch(
        constant ds4_hc_batch_args & args [[buffer(0)]],
        device const float * block [[buffer(1)]],
        device const float * residual [[buffer(2)]],
        device const float * post [[buffer(3)]],
        device const float * combination [[buffer(4)]],
        device float * output [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    const uint residual_width = args.hidden_size * DS4_CONNECTIONS;
    const uint total = args.n_tokens * residual_width;
    if (index >= total) return;
    const uint token = index / residual_width;
    const uint within = index % residual_width;
    const uint destination = within / args.hidden_size;
    const uint dimension = within % args.hidden_size;
    const uint combination_start = token * DS4_CONNECTIONS * DS4_CONNECTIONS;
    float value = block[token * args.hidden_size + dimension]
        * post[token * DS4_CONNECTIONS + destination];
    for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
        value += combination[combination_start + source * DS4_CONNECTIONS + destination]
            * residual[token * residual_width + source * args.hidden_size + dimension];
    }
    output[index] = value;
}

kernel void kernel_deepseek_v4_pack_attention_group(
        constant ds4_group_pack_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device float * output [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.group_width;
    if (index >= total) return;
    const uint token = index / args.group_width;
    const uint dimension = index % args.group_width;
    output[index] = input[token * args.row_width + args.group * args.group_width + dimension];
}

kernel void kernel_deepseek_v4_scatter_low_rank_group(
        constant ds4_group_pack_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device float * output [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.group_width;
    if (index >= total) return;
    const uint token = index / args.group_width;
    const uint dimension = index % args.group_width;
    output[token * args.row_width + args.group * args.group_width + dimension] = input[index];
}

kernel void kernel_deepseek_v4_hash_gather(
        constant ds4_hash_gather_args & args [[buffer(0)]],
        device const int * token_ids [[buffer(1)]],
        device const int * token_to_expert [[buffer(2)]],
        device int * output [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.top_k;
    if (index >= total) return;
    const uint token = index / args.top_k;
    const uint slot = index % args.top_k;
    const int token_id = token_ids[token];
    if (token_id < 0 || uint(token_id) >= args.vocab_size) {
        output[index] = -1;
        return;
    }
    output[index] = token_to_expert[uint(token_id) * args.top_k + slot];
}

struct ds4_hc_head_args {
    uint hidden_size;
    float eps;
};

kernel void kernel_deepseek_v4_hc_head(
        constant ds4_hc_head_args & args [[buffer(0)]],
        device const float * residual [[buffer(1)]],
        device const float * mix [[buffer(2)]],
        device const float * scale [[buffer(3)]],
        device const float * base [[buffer(4)]],
        device float * gates [[buffer(5)]],
        device float * output [[buffer(6)]],
        uint index [[thread_position_in_grid]]) {
    if (index < DS4_CONNECTIONS) {
        gates[index] = 1.0f / (1.0f + exp(-(mix[index] * scale[0] + base[index]))) + args.eps;
    }
    // The dispatch is ordered within a threadgroup, but no barrier can order
    // multiple threadgroups. Hidden sizes are larger than one group, so derive
    // the gate locally rather than consuming the just-written gate buffer.
    if (index < args.hidden_size) {
        float value = 0.0f;
        for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
            const float gate = 1.0f / (1.0f + exp(-(mix[source] * scale[0] + base[source]))) + args.eps;
            value += residual[source * args.hidden_size + index] * gate;
        }
        output[index] = value;
    }
}
