#include <metal_stdlib>
using namespace metal;

constant uint DS4_CONNECTIONS = 4;
constant uint DS4_PARAMETERS = 24;
constant uint DS4_SINKHORN_ITERATIONS = 20;

struct ds4_clamped_swiglu_args {
    uint n;
    float clamp;
};

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
    float scale;
};

struct ds4_compressor_frontier_args {
    uint width;
    uint row_offset;
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

kernel void kernel_deepseek_v4_local_sink_attention_f16(
        constant ds4_local_attention_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const half * raw_cache [[buffer(2)]],
        device const float * sinks [[buffer(3)]],
        device float * output [[buffer(4)]],
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
    output[index] = value / denominator;
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
