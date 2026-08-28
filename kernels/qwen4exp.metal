// Qwen3.8-Flash-Next architecture kernels.

#include <metal_stdlib>
using namespace metal;

struct hc_norm_args {
    uint branch_count;
    uint hidden_size;
    float eps;
};

struct hc_low_args {
    uint count;
    float inverse_branches;
};

struct hc_branch_args {
    uint branch_count;
    uint hidden_size;
};

kernel void kernel_qwen4exp_hc_rms_norm_f32(
        constant hc_norm_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device float * normalized [[buffer(3)]],
        threadgroup float * partial [[threadgroup(0)]],
        uint branch [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        uint thread_count [[threads_per_threadgroup]]) {
    if (branch >= args.branch_count) return;
    const ulong base = (ulong)branch * args.hidden_size;
    float sum_square = 0.0f;
    for (uint hidden = tid; hidden < args.hidden_size; hidden += thread_count) {
        const float value = input[base + hidden];
        sum_square += value * value;
    }
    sum_square = simd_sum(sum_square);
    if (lane == 0) partial[simdgroup] = sum_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum_square = lane < (thread_count + 31u) / 32u ? partial[lane] : 0.0f;
    sum_square = simd_sum(sum_square);
    const float scale = rsqrt(sum_square / float(args.hidden_size) + args.eps);
    for (uint hidden = tid; hidden < args.hidden_size; hidden += thread_count) {
        const ulong index = base + hidden;
        normalized[index] = input[index] * scale * weight[index];
    }
}

kernel void kernel_qwen4exp_hc_low_silu_f32(
        constant hc_low_args & args [[buffer(0)]],
        device float * values [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.count) return;
    const float value = values[index] * args.inverse_branches;
    values[index] = value / (1.0f + exp(-value));
}

kernel void kernel_qwen4exp_hc_gated_mean_f32(
        constant hc_branch_args & args [[buffer(0)]],
        device const float * normalized [[buffer(1)]],
        device const float * raw_gate [[buffer(2)]],
        device float * mixed [[buffer(3)]],
        uint hidden [[thread_position_in_grid]]) {
    if (hidden >= args.hidden_size) return;
    float sum = 0.0f;
    for (uint branch = 0; branch < args.branch_count; ++branch) {
        const ulong index = (ulong)branch * args.hidden_size + hidden;
        const float gate = 1.0f / (1.0f + exp(-raw_gate[index]));
        sum += gate * normalized[index];
    }
    mixed[hidden] = sum / float(args.branch_count);
}

struct hc_packed_branch_args {
    uint n_tokens;
    uint branch_count;
    uint hidden_size;
};

kernel void kernel_qwen4exp_hc_repeat_packed_f32(
        constant hc_packed_branch_args & args [[buffer(0)]],
        device const float * embedding [[buffer(1)]],
        device float * hyper_residual [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    const uint count = args.n_tokens * args.branch_count * args.hidden_size;
    if (index >= count) return;
    const uint hidden = index % args.hidden_size;
    const uint row = index / args.hidden_size;
    const uint token = row / args.branch_count;
    hyper_residual[index] = embedding[(ulong)token * args.hidden_size + hidden];
}

kernel void kernel_qwen4exp_hc_gated_mean_packed_f32(
        constant hc_packed_branch_args & args [[buffer(0)]],
        device const float * normalized [[buffer(1)]],
        device const float * raw_gate [[buffer(2)]],
        device float * mixed [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint count = args.n_tokens * args.hidden_size;
    if (index >= count) return;
    const uint token = index / args.hidden_size;
    const uint hidden = index % args.hidden_size;
    float sum = 0.0f;
    for (uint branch = 0; branch < args.branch_count; ++branch) {
        const ulong source = ((ulong)token * args.branch_count + branch)
            * args.hidden_size + hidden;
        const float gate = 1.0f / (1.0f + exp(-raw_gate[source]));
        sum += gate * normalized[source];
    }
    mixed[(ulong)token * args.hidden_size + hidden] =
        sum / float(args.branch_count);
}

kernel void kernel_qwen4exp_hc_inject_f32(
        constant hc_branch_args & args [[buffer(0)]],
        device const float * block_output [[buffer(1)]],
        device const float * raw_injection [[buffer(2)]],
        device float * residual [[buffer(3)]],
        uint2 index [[thread_position_in_grid]]) {
    const uint hidden = index.x;
    const uint branch = index.y;
    if (hidden >= args.hidden_size || branch >= args.branch_count) return;
    const float value = raw_injection[branch] / float(args.branch_count);
    const float injection = 2.0f / (1.0f + exp(-value));
    residual[(ulong)branch * args.hidden_size + hidden] += block_output[hidden] * injection;
}

kernel void kernel_qwen4exp_hc_inject_packed_f32(
        constant hc_packed_branch_args & args [[buffer(0)]],
        device const float * block_output [[buffer(1)]],
        device const float * raw_injection [[buffer(2)]],
        device float * residual [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint count = args.n_tokens * args.branch_count * args.hidden_size;
    if (index >= count) return;
    const uint hidden = index % args.hidden_size;
    const uint row = index / args.hidden_size;
    const uint branch = row % args.branch_count;
    const uint token = row / args.branch_count;
    const float value = raw_injection[(ulong)token * args.branch_count + branch]
        / float(args.branch_count);
    const float injection = 2.0f / (1.0f + exp(-value));
    residual[((ulong)token * args.branch_count + branch) * args.hidden_size + hidden]
        += block_output[(ulong)token * args.hidden_size + hidden] * injection;
}

struct qwen4exp_ple_gate_args {
    uint branch_count;
    uint hidden_size;
};

kernel void kernel_qwen4exp_ple_gate_f32(
        constant qwen4exp_ple_gate_args & args [[buffer(0)]],
        device const float * key [[buffer(1)]],
        device const float * query [[buffer(2)]],
        device const float * value [[buffer(3)]],
        device float * gated [[buffer(4)]],
        threadgroup float * partial [[threadgroup(0)]],
        uint branch [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        uint thread_count [[threads_per_threadgroup]]) {
    if (branch >= args.branch_count) return;
    const ulong base = (ulong)branch * args.hidden_size;
    float dot = 0.0f;
    for (uint hidden = tid; hidden < args.hidden_size; hidden += thread_count) {
        dot += key[base + hidden] * query[base + hidden];
    }
    dot = simd_sum(dot);
    if (lane == 0) partial[simdgroup] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        const uint simdgroups = (thread_count + 31u) / 32u;
        const float candidate = uint(lane) < simdgroups ? partial[lane] : 0.0f;
        const float total = simd_sum(candidate);
        if (lane == 0) partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float score = partial[0] * rsqrt(float(args.hidden_size));
    float transformed = 0.0f;
    if (score > 0.0f) {
        transformed = sqrt(max(score, 1.0e-6f));
    } else if (score < 0.0f) {
        transformed = -sqrt(max(-score, 1.0e-6f));
    }
    const float gate = 1.0f / (1.0f + exp(-transformed));
    for (uint hidden = tid; hidden < args.hidden_size; hidden += thread_count) {
        gated[base + hidden] = value[hidden] * gate;
    }
}

struct qwen4exp_ple_packed_norm_args {
    uint n_tokens;
    uint branch_count;
    uint hidden_size;
    float eps;
};

kernel void kernel_qwen4exp_ple_grouped_rms_norm_packed_f32(
        constant qwen4exp_ple_packed_norm_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device float * output [[buffer(3)]],
        threadgroup float * partial [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        uint thread_count [[threads_per_threadgroup]]) {
    const uint branch = group % args.branch_count;
    const uint token = group / args.branch_count;
    if (branch >= args.branch_count || token >= args.n_tokens) return;
    const ulong base = ((ulong)token * args.branch_count + branch) * args.hidden_size;
    const ulong weight_base = (ulong)branch * args.hidden_size;
    float sum_square = 0.0f;
    for (uint hidden = tid; hidden < args.hidden_size; hidden += thread_count) {
        const float value = input[base + hidden];
        sum_square += value * value;
    }
    sum_square = simd_sum(sum_square);
    if (lane == 0) partial[simdgroup] = sum_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum_square = lane < (thread_count + 31u) / 32u ? partial[lane] : 0.0f;
    sum_square = simd_sum(sum_square);
    const float scale = rsqrt(sum_square / float(args.hidden_size) + args.eps);
    for (uint hidden = tid; hidden < args.hidden_size; hidden += thread_count) {
        const ulong index = base + hidden;
        output[index] = input[index] * scale * weight[weight_base + hidden];
    }
}

struct qwen4exp_ple_packed_gate_args {
    uint n_tokens;
    uint branch_count;
    uint hidden_size;
};

kernel void kernel_qwen4exp_ple_gate_packed_f32(
        constant qwen4exp_ple_packed_gate_args & args [[buffer(0)]],
        device const float * key [[buffer(1)]],
        device const float * query [[buffer(2)]],
        device const float * value [[buffer(3)]],
        device float * gated [[buffer(4)]],
        threadgroup float * partial [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        uint thread_count [[threads_per_threadgroup]]) {
    const uint branch = group % args.branch_count;
    const uint token = group / args.branch_count;
    if (branch >= args.branch_count || token >= args.n_tokens) return;
    const ulong base = ((ulong)token * args.branch_count + branch) * args.hidden_size;
    float dot = 0.0f;
    for (uint hidden = tid; hidden < args.hidden_size; hidden += thread_count) {
        dot += key[base + hidden] * query[base + hidden];
    }
    dot = simd_sum(dot);
    if (lane == 0) partial[simdgroup] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        const uint simdgroups = (thread_count + 31u) / 32u;
        const float candidate = uint(lane) < simdgroups ? partial[lane] : 0.0f;
        const float total = simd_sum(candidate);
        if (lane == 0) partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float score = partial[0] * rsqrt(float(args.hidden_size));
    float transformed = 0.0f;
    if (score > 0.0f) {
        transformed = sqrt(max(score, 1.0e-6f));
    } else if (score < 0.0f) {
        transformed = -sqrt(max(-score, 1.0e-6f));
    }
    const float gate = 1.0f / (1.0f + exp(-transformed));
    const ulong value_base = (ulong)token * args.hidden_size;
    for (uint hidden = tid; hidden < args.hidden_size; hidden += thread_count) {
        gated[base + hidden] = value[value_base + hidden] * gate;
    }
}

struct qwen4exp_ple_conv_args {
    uint channels;
    uint history_len;
    uint kernel_size;
    uint dilation;
};

kernel void kernel_qwen4exp_ple_conv_epilogue_f32(
        constant qwen4exp_ple_conv_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device float * state [[buffer(3)]],
        device const float * residual [[buffer(4)]],
        device const float * gated [[buffer(5)]],
        device float * raw_output [[buffer(6)]],
        device float * output [[buffer(7)]],
        uint channel [[thread_position_in_grid]]) {
    if (channel >= args.channels) return;
    const ulong state_base = (ulong)channel * args.history_len;
    const ulong weight_base = (ulong)channel * args.kernel_size;
    float convolution = 0.0f;
    for (uint tap = 0; tap < args.kernel_size; ++tap) {
        const uint lag = (args.kernel_size - 1u - tap) * args.dilation;
        const float value = lag == 0u
            ? input[channel]
            : state[state_base + args.history_len - lag];
        convolution += weight[weight_base + tap] * value;
    }
    raw_output[channel] = convolution;
    const float activated = convolution / (1.0f + exp(-convolution));
    output[channel] = residual[channel] + gated[channel] + activated;

    if (args.history_len > 0u) {
        for (uint slot = 0; slot + 1u < args.history_len; ++slot) {
            state[state_base + slot] = state[state_base + slot + 1u];
        }
        state[state_base + args.history_len - 1u] = input[channel];
    }
}

struct qwen4exp_ple_packed_conv_args {
    uint n_tokens;
    uint channels;
    uint history_len;
    uint kernel_size;
    uint dilation;
};

kernel void kernel_qwen4exp_ple_conv_epilogue_packed_f32(
        constant qwen4exp_ple_packed_conv_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device float * state [[buffer(3)]],
        device const float * residual [[buffer(4)]],
        device const float * gated [[buffer(5)]],
        device float * raw_output [[buffer(6)]],
        device float * output [[buffer(7)]],
        uint channel [[thread_position_in_grid]]) {
    if (channel >= args.channels) return;
    const ulong state_base = (ulong)channel * args.history_len;
    const ulong weight_base = (ulong)channel * args.kernel_size;
    for (uint token = 0; token < args.n_tokens; ++token) {
        const ulong token_base = (ulong)token * args.channels;
        float convolution = 0.0f;
        for (uint tap = 0; tap < args.kernel_size; ++tap) {
            const uint lag = (args.kernel_size - 1u - tap) * args.dilation;
            const float value = lag == 0u
                ? input[token_base + channel]
                : state[state_base + args.history_len - lag];
            convolution += weight[weight_base + tap] * value;
        }
        raw_output[token_base + channel] = convolution;
        const float activated = convolution / (1.0f + exp(-convolution));
        output[token_base + channel] = residual[token_base + channel]
            + gated[token_base + channel] + activated;

        if (args.history_len > 0u) {
            for (uint slot = 0; slot + 1u < args.history_len; ++slot) {
                state[state_base + slot] = state[state_base + slot + 1u];
            }
            state[state_base + args.history_len - 1u] = input[token_base + channel];
        }
    }
}

struct qwen4exp_gdn_norm_args {
    uint n_heads;
    float eps;
};

kernel void kernel_qwen4exp_gdn_rmsnorm_sigmoid_hd128_r4_f32(
        constant qwen4exp_gdn_norm_args & args [[buffer(0)]],
        device const float * input           [[buffer(1)]],
        device const float * weight          [[buffer(2)]],
        device const float * gate            [[buffer(3)]],
        device       float * output          [[buffer(4)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint2 tpitg [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint HEAD_DIM = 128;
    constexpr uint ROWS_PER_TG = 4;
    const uint head = tgpig.x * ROWS_PER_TG + tpitg.y;
    if (head >= args.n_heads) return;

    device const float * input_head = input + (ulong)head * HEAD_DIM;
    device const float * gate_head = gate + (ulong)head * HEAD_DIM;
    device float * output_head = output + (ulong)head * HEAD_DIM;

    float sumsq = 0.0f;
    for (uint lane = tiisg; lane < HEAD_DIM; lane += 32) {
        const float value = input_head[lane];
        sumsq += value * value;
    }
    sumsq = simd_sum(sumsq);
    const float scale = 1.0f / sqrt(sumsq / float(HEAD_DIM) + args.eps);

    for (uint lane = tiisg; lane < HEAD_DIM; lane += 32) {
        const float normalized = input_head[lane] * scale * weight[lane];
        const float sigmoid_gate = 1.0f / (1.0f + exp(-gate_head[lane]));
        output_head[lane] = normalized * sigmoid_gate;
    }
}

struct qwen4exp_qsa_rope_args {
    uint head_count;
    uint head_dim;
    uint rotary_dim;
    uint position;
    float theta;
    float eps;
};

struct qwen4exp_qsa_pending_args {
    uint head_dim;
    uint slot;
};

struct qwen4exp_qsa_pool_args {
    uint ratio;
    uint head_dim;
    uint rotary_dim;
    uint block;
    float theta;
    float eps;
};

struct qwen4exp_qsa_packed_index_args {
    uint start_position;
    uint n_tokens;
    uint ratio;
    uint head_dim;
    uint rotary_dim;
    uint first_block;
    uint block_count;
    uint block_capacity;
    float theta;
    float eps;
};

struct qwen4exp_qsa_index_score_args {
    uint visible_blocks;
};

struct qwen4exp_qsa_packed_query_args {
    uint start_position;
    uint query_count;
    uint head_count;
    uint head_dim;
    uint rotary_dim;
    float theta;
    float eps;
};

struct qwen4exp_qsa_packed_score_args {
    uint start_position;
    uint query_count;
    uint ratio;
    uint block_capacity;
};

struct qwen4exp_qsa_ids_args {
    uint block_budget;
    uint visible_blocks;
    uint ratio;
    uint sequence_length;
    uint output_width;
};

struct qwen4exp_qsa_packed_ids_args {
    uint start_position;
    uint query_count;
    uint block_budget;
    uint ratio;
    uint output_width;
};

struct qwen4exp_qsa_qgate_args {
    uint head_count;
    uint head_dim;
    uint rotary_dim;
    uint position;
    float theta;
    float eps;
};

struct qwen4exp_qsa_publish_args {
    uint width;
    uint position;
};

struct qwen4exp_qsa_attention_args {
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    uint id_count;
    uint row_stride;
    uint cache_capacity;
    float scale;
};

struct qwen4exp_qsa_packed_attention_args {
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

struct qwen4exp_qsa_selected_audit_args {
    uint query_count;
    int expected_selected_count;
    int count_mismatch_status;
};

static inline float qwen4exp_qsa_rope_value(
        device const float * row,
        device const float * weight,
        uint lane,
        uint rotary_dim,
        uint position,
        float theta) {
    if (lane >= rotary_dim) return row[lane] * weight[lane];
    const uint half_dim = rotary_dim / 2u;
    const uint pair = lane % half_dim;
    const float frequency = pow(theta, -2.0f * float(pair) / float(rotary_dim));
    const float angle = float(position) * frequency;
    const float cosine = cos(angle);
    const float sine = sin(angle);
    if (lane < half_dim) {
        return row[pair] * weight[pair] * cosine
            - row[pair + half_dim] * weight[pair + half_dim] * sine;
    }
    return row[pair + half_dim] * weight[pair + half_dim] * cosine
        + row[pair] * weight[pair] * sine;
}

kernel void kernel_qwen4exp_qsa_norm_rope_f32(
        constant qwen4exp_qsa_rope_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint width = args.head_count * args.head_dim;
    if (index >= width) return;
    const uint head = index / args.head_dim;
    const uint lane = index % args.head_dim;
    device const float * row = input + (ulong)head * args.head_dim;
    float sum_square = 0.0f;
    for (uint inner = 0u; inner < args.head_dim; ++inner) {
        const float value = row[inner];
        sum_square += value * value;
    }
    const float norm = rsqrt(sum_square / float(args.head_dim) + args.eps);
    output[index] = qwen4exp_qsa_rope_value(
        row, weight, lane, args.rotary_dim, args.position, args.theta) * norm;
}

kernel void kernel_qwen4exp_qsa_norm_rope_packed_f32(
        constant qwen4exp_qsa_packed_query_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint width = args.head_count * args.head_dim;
    const ulong count = (ulong)width * args.query_count;
    if ((ulong)index >= count) return;
    const uint query = index / width;
    const uint local = index % width;
    const uint head = local / args.head_dim;
    const uint lane = local % args.head_dim;
    device const float * row = input
        + (ulong)query * width + (ulong)head * args.head_dim;
    float sum_square = 0.0f;
    for (uint inner = 0u; inner < args.head_dim; ++inner) {
        const float value = row[inner];
        sum_square += value * value;
    }
    const float norm = rsqrt(sum_square / float(args.head_dim) + args.eps);
    output[index] = qwen4exp_qsa_rope_value(
        row,
        weight,
        lane,
        args.rotary_dim,
        args.start_position + query,
        args.theta) * norm;
}

kernel void kernel_qwen4exp_qsa_write_pending_f32(
        constant qwen4exp_qsa_pending_args & args [[buffer(0)]],
        device const float * raw_key [[buffer(1)]],
        device float * pending [[buffer(2)]],
        uint lane [[thread_position_in_grid]]) {
    if (lane >= args.head_dim) return;
    pending[(ulong)args.slot * args.head_dim + lane] = raw_key[lane];
}

kernel void kernel_qwen4exp_qsa_pool_publish_f16(
        constant qwen4exp_qsa_pool_args & args [[buffer(0)]],
        device const float * pending [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device half * compressed [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    float sum_square = 0.0f;
    for (uint lane = 0u; lane < args.head_dim; ++lane) {
        float sum = 0.0f;
        for (uint slot = 0u; slot < args.ratio; ++slot) {
            sum += pending[(ulong)slot * args.head_dim + lane];
        }
        const float rounded = float(half(sum / float(args.ratio)));
        sum_square += rounded * rounded;
    }
    const float norm = rsqrt(sum_square / float(args.head_dim) + args.eps);
    const uint position = args.block * args.ratio;
    device half * destination = compressed + (ulong)args.block * args.head_dim;
    for (uint lane = 0u; lane < args.head_dim; ++lane) {
        float sum = 0.0f;
        for (uint slot = 0u; slot < args.ratio; ++slot) {
            sum += pending[(ulong)slot * args.head_dim + lane];
        }
        const float rounded = float(half(sum / float(args.ratio)));
        float rotated = rounded * weight[lane];
        if (lane < args.rotary_dim) {
            const uint half_dim = args.rotary_dim / 2u;
            const uint pair = lane % half_dim;
            float paired_sum = 0.0f;
            for (uint slot = 0u; slot < args.ratio; ++slot) {
                paired_sum += pending[(ulong)slot * args.head_dim
                    + pair + (lane < half_dim ? half_dim : 0u)];
            }
            const float paired = float(half(paired_sum / float(args.ratio)));
            const float frequency = pow(args.theta,
                -2.0f * float(pair) / float(args.rotary_dim));
            const float angle = float(position) * frequency;
            rotated = lane < half_dim
                ? rounded * weight[lane] * cos(angle)
                    - paired * weight[pair + half_dim] * sin(angle)
                : rounded * weight[lane] * cos(angle)
                    + paired * weight[pair] * sin(angle);
        }
        destination[lane] = half(rotated * norm);
    }
}

// Publish from the pre-chunk pending slots before the following dispatch
// advances those slots to their final per-residue values.
kernel void kernel_qwen4exp_qsa_pool_publish_packed_f16(
        constant qwen4exp_qsa_packed_index_args & args [[buffer(0)]],
        device const float * pending [[buffer(1)]],
        device const float * raw_keys [[buffer(2)]],
        device const float * weight [[buffer(3)]],
        device half * compressed [[buffer(4)]],
        uint block_offset [[thread_position_in_grid]]) {
    if (block_offset >= args.block_count
            || args.ratio == 0u
            || args.head_dim == 0u
            || args.rotary_dim == 0u
            || args.rotary_dim > args.head_dim
            || (args.rotary_dim & 1u) != 0u) return;

    const ulong start = args.start_position;
    const ulong end = start + args.n_tokens;
    const ulong block = (ulong)args.first_block + block_offset;
    if (block >= args.block_capacity) return;
    const ulong block_start = block * args.ratio;
    const ulong block_end = block_start + args.ratio;
    if (block_end <= start || block_end > end) return;

    float sum_square = 0.0f;
    for (uint lane = 0u; lane < args.head_dim; ++lane) {
        float sum = 0.0f;
        for (uint slot = 0u; slot < args.ratio; ++slot) {
            const ulong source_position = block_start + slot;
            sum += source_position < start
                ? pending[(ulong)slot * args.head_dim + lane]
                : raw_keys[(source_position - start) * args.head_dim + lane];
        }
        const float rounded = float(half(sum / float(args.ratio)));
        sum_square += rounded * rounded;
    }

    const float norm = rsqrt(sum_square / float(args.head_dim) + args.eps);
    const uint position = uint(block * args.ratio);
    device half * destination = compressed + block * args.head_dim;
    for (uint lane = 0u; lane < args.head_dim; ++lane) {
        float sum = 0.0f;
        for (uint slot = 0u; slot < args.ratio; ++slot) {
            const ulong source_position = block_start + slot;
            sum += source_position < start
                ? pending[(ulong)slot * args.head_dim + lane]
                : raw_keys[(source_position - start) * args.head_dim + lane];
        }
        const float rounded = float(half(sum / float(args.ratio)));
        float rotated = rounded * weight[lane];
        if (lane < args.rotary_dim) {
            const uint half_dim = args.rotary_dim / 2u;
            const uint pair = lane % half_dim;
            const uint paired_lane = pair + (lane < half_dim ? half_dim : 0u);
            float paired_sum = 0.0f;
            for (uint slot = 0u; slot < args.ratio; ++slot) {
                const ulong source_position = block_start + slot;
                paired_sum += source_position < start
                    ? pending[(ulong)slot * args.head_dim + paired_lane]
                    : raw_keys[(source_position - start) * args.head_dim + paired_lane];
            }
            const float paired = float(half(paired_sum / float(args.ratio)));
            const float frequency = pow(args.theta,
                -2.0f * float(pair) / float(args.rotary_dim));
            const float angle = float(position) * frequency;
            rotated = lane < half_dim
                ? rounded * weight[lane] * cos(angle)
                    - paired * weight[pair + half_dim] * sin(angle)
                : rounded * weight[lane] * cos(angle)
                    + paired * weight[pair] * sin(angle);
        }
        destination[lane] = half(rotated * norm);
    }
}

kernel void kernel_qwen4exp_qsa_commit_pending_packed_f32(
        constant qwen4exp_qsa_packed_index_args & args [[buffer(0)]],
        device const float * raw_keys [[buffer(1)]],
        device float * pending [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (args.ratio == 0u || args.head_dim == 0u) return;
    const ulong count = (ulong)args.ratio * args.head_dim;
    if (index >= count) return;
    const uint slot = index / args.head_dim;
    const uint lane = index % args.head_dim;
    const uint start_slot = args.start_position % args.ratio;
    const uint first = (slot + args.ratio - start_slot) % args.ratio;
    if (first >= args.n_tokens) return;
    const uint last = first
        + ((args.n_tokens - 1u - first) / args.ratio) * args.ratio;
    pending[(ulong)slot * args.head_dim + lane]
        = raw_keys[(ulong)last * args.head_dim + lane];
}

kernel void kernel_qwen4exp_qsa_index_scores_4x128_f16(
        constant qwen4exp_qsa_index_score_args & args [[buffer(0)]],
        device const float * query [[buffer(1)]],
        device const half * compressed_keys [[buffer(2)]],
        device float * scores [[buffer(3)]],
        uint group [[threadgroup_position_in_grid]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint block = group * 8u + uint(simdgroup);
    if (block >= args.visible_blocks) return;
    device const half * key = compressed_keys + (ulong)block * 128u;
    float score = 0.0f;
    for (uint head = 0u; head < 4u; ++head) {
        device const float * query_head = query + head * 128u;
        float dot = 0.0f;
        for (uint dimension = uint(lane); dimension < 128u; dimension += 32u) {
            dot += query_head[dimension] * float(key[dimension]);
        }
        dot = simd_sum(dot);
        if (lane == 0u) score += max(dot, 0.0f);
    }
    if (lane == 0u) scores[block] = score * 0.08838834764831845f;
}

kernel void kernel_qwen4exp_qsa_index_scores_packed_4x128_f16(
        constant qwen4exp_qsa_packed_score_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const half * compressed_keys [[buffer(2)]],
        device float * scores [[buffer(3)]],
        device int * visible_counts [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint query = group.y;
    if (query >= args.query_count || args.ratio == 0u) return;
    const uint visible = (args.start_position + query + 1u) / args.ratio;
    if (group.x == 0u && simdgroup == 0u && lane == 0u) {
        visible_counts[query] = visible <= args.block_capacity ? int(visible) : -1;
    }
    const uint block = group.x * 8u + uint(simdgroup);
    if (block >= visible || block >= args.block_capacity) return;
    device const float * query_row = queries + (ulong)query * 4u * 128u;
    device const half * key = compressed_keys + (ulong)block * 128u;
    float score = 0.0f;
    for (uint head = 0u; head < 4u; ++head) {
        device const float * query_head = query_row + head * 128u;
        float dot = 0.0f;
        for (uint dimension = uint(lane); dimension < 128u; dimension += 32u) {
            dot += query_head[dimension] * float(key[dimension]);
        }
        dot = simd_sum(dot);
        if (lane == 0u) score += max(dot, 0.0f);
    }
    if (lane == 0u) {
        scores[(ulong)query * args.block_capacity + block]
            = score * 0.08838834764831845f;
    }
}

kernel void kernel_qwen4exp_qsa_fill_block_ids_i32(
        constant qwen4exp_qsa_ids_args & args [[buffer(0)]],
        device int * selected_blocks [[buffer(1)]],
        device int * selected_count [[buffer(2)]],
        device int * status [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    const uint count = min(args.block_budget, args.visible_blocks);
    for (uint slot = 0u; slot < args.block_budget; ++slot) {
        selected_blocks[slot] = slot < count ? int(slot) : -1;
    }
    selected_count[0] = int(count);
    status[0] = 0;
}

kernel void kernel_qwen4exp_qsa_expand_ids_i32(
        constant qwen4exp_qsa_ids_args & args [[buffer(0)]],
        device const int * selected_blocks [[buffer(1)]],
        device int * token_ids [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    for (uint slot = 0u; slot < args.output_width; ++slot) token_ids[slot] = -1;
    uint output = 0u;
    for (uint slot = 0u; slot < args.block_budget; ++slot) {
        const int block = selected_blocks[slot];
        if (block < 0 || uint(block) >= args.visible_blocks) continue;
        const uint start = uint(block) * args.ratio;
        for (uint offset = 0u; offset < args.ratio && output < args.output_width; ++offset) {
            token_ids[output++] = int(start + offset);
        }
    }
    const uint tail_start = args.visible_blocks * args.ratio;
    for (uint position = tail_start;
            position < args.sequence_length && output < args.output_width;
            ++position) {
        token_ids[output++] = int(position);
    }
}

kernel void kernel_qwen4exp_qsa_expand_ids_packed_i32(
        constant qwen4exp_qsa_packed_ids_args & args [[buffer(0)]],
        device const int * visible_counts [[buffer(1)]],
        device const int * selected_blocks [[buffer(2)]],
        device const int * selected_counts [[buffer(3)]],
        device const int * status [[buffer(4)]],
        device int * token_ids [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    const ulong count = (ulong)args.output_width * args.query_count;
    if ((ulong)index >= count || args.ratio == 0u) return;
    const uint query = index / args.output_width;
    const uint slot = index % args.output_width;
    const int visible_i = visible_counts[query];
    const int selected_i = selected_counts[query];
    const bool valid = status[query] == 0
        && visible_i > 0
        && selected_i >= 0
        && uint(selected_i) <= args.block_budget;
    if (!valid) {
        token_ids[index] = -1;
        return;
    }
    const uint visible = uint(visible_i);
    const uint selected = uint(selected_i);
    const uint selected_tokens = selected * args.ratio;
    if (slot < selected_tokens) {
        const uint block_slot = slot / args.ratio;
        const int block = selected_blocks[(ulong)query * args.block_budget + block_slot];
        token_ids[index] = block >= 0 && uint(block) < visible
            ? block * int(args.ratio) + int(slot % args.ratio)
            : -1;
        return;
    }
    const uint sequence_length = args.start_position + query + 1u;
    const uint tail_start = visible * args.ratio;
    const uint tail_slot = slot - selected_tokens;
    const uint tail_count = sequence_length - tail_start;
    token_ids[index] = tail_slot < tail_count
        ? int(tail_start + tail_slot)
        : -1;
}

kernel void kernel_qwen4exp_qsa_qgate_norm_rope_f32(
        constant qwen4exp_qsa_qgate_args & args [[buffer(0)]],
        device const float * projected [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device float * query [[buffer(3)]],
        device float * gate [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    const uint width = args.head_count * args.head_dim;
    if (index >= width) return;
    const uint head = index / args.head_dim;
    const uint lane = index % args.head_dim;
    device const float * row = projected + (ulong)head * args.head_dim * 2u;
    float sum_square = 0.0f;
    for (uint inner = 0u; inner < args.head_dim; ++inner) {
        const float value = row[inner];
        sum_square += value * value;
    }
    const float norm = rsqrt(sum_square / float(args.head_dim) + args.eps);
    query[index] = qwen4exp_qsa_rope_value(
        row, weight, lane, args.rotary_dim, args.position, args.theta) * norm;
    gate[index] = row[args.head_dim + lane];
}

kernel void kernel_qwen4exp_qsa_publish_kv_f16(
        constant qwen4exp_qsa_publish_args & args [[buffer(0)]],
        device const float * key [[buffer(1)]],
        device const float * value [[buffer(2)]],
        device half * key_cache [[buffer(3)]],
        device half * value_cache [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.width) return;
    const ulong destination = (ulong)args.position * args.width + index;
    key_cache[destination] = half(key[index]);
    value_cache[destination] = half(value[index]);
}

kernel void kernel_qwen4exp_qsa_attention_logits_f16(
        constant qwen4exp_qsa_attention_args & args [[buffer(0)]],
        device const float * query [[buffer(1)]],
        device const half * key_cache [[buffer(2)]],
        device const int * token_ids [[buffer(3)]],
        device float * logits [[buffer(4)]],
        uint group [[threadgroup_position_in_grid]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    if (args.head_dim != 256u
            || args.kv_heads == 0u
            || args.query_heads % args.kv_heads != 0u) return;
    const uint pair = group * 8u + uint(simdgroup);
    const uint pair_count = args.query_heads * args.id_count;
    if (pair >= pair_count) return;
    const uint query_head = pair / args.id_count;
    const uint slot = pair % args.id_count;
    const int position = token_ids[slot];
    const uint output_index = query_head * args.row_stride + slot;
    if (position < 0 || uint(position) >= args.cache_capacity) {
        if (simd_lane == 0u) logits[output_index] = -INFINITY;
        return;
    }
    const uint kv_head = query_head / (args.query_heads / args.kv_heads);
    device const float * query_row = query + (ulong)query_head * args.head_dim;
    const ulong key_start = ((ulong)uint(position) * args.kv_heads + kv_head) * args.head_dim;
    float partial = 0.0f;
    for (uint dimension = uint(simd_lane); dimension < 256u; dimension += 32u) {
        partial += query_row[dimension] * float(key_cache[key_start + dimension]);
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) logits[output_index] = partial * args.scale;
}

kernel void kernel_qwen4exp_qsa_attention_softmax_value_f16(
        constant qwen4exp_qsa_attention_args & args [[buffer(0)]],
        device const float * raw_gate [[buffer(1)]],
        device const half * value_cache [[buffer(2)]],
        device const int * token_ids [[buffer(3)]],
        device float * logits [[buffer(4)]],
        device float * output [[buffer(5)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint query_head [[threadgroup_position_in_grid]],
        uint lane [[thread_position_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    if (query_head >= args.query_heads || args.head_dim != 256u
            || args.kv_heads == 0u
            || args.query_heads % args.kv_heads != 0u) return;
    device float * head_logits = logits + (ulong)query_head * args.row_stride;
    float local_maximum = -INFINITY;
    for (uint slot = lane; slot < args.id_count; slot += 256u) {
        local_maximum = max(local_maximum, head_logits[slot]);
    }
    local_maximum = simd_max(local_maximum);
    if (simd_lane == 0u) scratch[uint(simdgroup)] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        const float candidate = simd_lane < 8u ? scratch[uint(simd_lane)] : -INFINITY;
        const float maximum = simd_max(candidate);
        if (simd_lane == 0u) scratch[8] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maximum = scratch[8];

    float local_denominator = 0.0f;
    for (uint slot = lane; slot < args.id_count; slot += 256u) {
        const float mass = exp(head_logits[slot] - maximum);
        head_logits[slot] = mass;
        local_denominator += mass;
    }
    local_denominator = simd_sum(local_denominator);
    if (simd_lane == 0u) scratch[uint(simdgroup)] = local_denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (simdgroup == 0u) {
        const float candidate = simd_lane < 8u ? scratch[uint(simd_lane)] : 0.0f;
        const float denominator = simd_sum(candidate);
        if (simd_lane == 0u) scratch[8] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kv_head = query_head / (args.query_heads / args.kv_heads);
    float accumulator = 0.0f;
    for (uint slot = 0u; slot < args.id_count; ++slot) {
        const int position = token_ids[slot];
        if (position < 0 || uint(position) >= args.cache_capacity) continue;
        const ulong value_start = ((ulong)uint(position) * args.kv_heads + kv_head) * args.head_dim;
        accumulator += float(value_cache[value_start + lane])
            * head_logits[slot];
    }
    const uint output_index = query_head * args.head_dim + lane;
    const float gate = 1.0f / (1.0f + exp(-raw_gate[output_index]));
    output[output_index] = scratch[8] > 0.0f
        ? accumulator / scratch[8] * gate
        : 0.0f;
}

kernel void kernel_qwen4exp_qsa_attention_logits_packed_f16(
        constant qwen4exp_qsa_packed_attention_args & args [[buffer(0)]],
        device const float * query [[buffer(1)]],
        device const half * key_cache [[buffer(2)]],
        device const int * token_ids [[buffer(3)]],
        device const int * selected_counts [[buffer(4)]],
        device const int * status [[buffer(5)]],
        device float * logits [[buffer(6)]],
        threadgroup ushort * staged_key [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    if (args.head_dim != 256u
            || args.kv_heads == 0u
            || args.query_heads % args.kv_heads != 0u
            || args.ratio == 0u) return;
    const uint heads_per_kv = args.query_heads / args.kv_heads;
    if (heads_per_kv % 4u != 0u
            || group.x >= heads_per_kv / 4u
            || group.y >= args.kv_heads
            || group.z >= args.query_count) return;
    const uint query_index = group.z;
    const uint query_head = group.y * heads_per_kv + group.x * 4u + uint(simdgroup);
    const uint sequence_length = args.start_position + query_index + 1u;
    const int selected_i = selected_counts[query_index];
    const bool metadata_valid = status[query_index] == 0
        && selected_i == int(args.block_budget);
    const uint active_count = metadata_valid
        ? uint(selected_i) * args.ratio + sequence_length % args.ratio
        : 0u;
    const ulong query_start = ((ulong)query_index * args.query_heads + query_head)
        * args.head_dim;
    const ulong id_start = (ulong)query_index * args.row_stride;
    const ulong logit_start = ((ulong)query_index * args.query_heads + query_head)
        * args.row_stride;
    for (uint slot = 0u; slot < args.row_stride; ++slot) {
        const int position = slot < active_count ? token_ids[id_start + slot] : -1;
        const bool position_valid = metadata_valid
            && active_count <= args.row_stride
            && position >= 0
            && uint(position) < args.cache_capacity
            && uint(position) < sequence_length;
        if (!position_valid) {
            if (simd_lane == 0u) logits[logit_start + slot] = -INFINITY;
            continue;
        }
        const ulong key_start = ((ulong)uint(position) * args.kv_heads + group.y)
            * args.head_dim;
        staged_key[lane] = as_type<ushort>(key_cache[key_start + lane]);
        staged_key[lane + 128u] = as_type<ushort>(key_cache[key_start + lane + 128u]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float partial = 0.0f;
        for (uint dimension = uint(simd_lane); dimension < 256u; dimension += 32u) {
            partial += query[query_start + dimension]
                * float(as_type<half>(staged_key[dimension]));
        }
        partial = simd_sum(partial);
        if (simd_lane == 0u) logits[logit_start + slot] = partial * args.scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_qwen4exp_qsa_attention_softmax_value_packed_f16(
        constant qwen4exp_qsa_packed_attention_args & args [[buffer(0)]],
        device const float * query_gate_projection [[buffer(1)]],
        device const half * value_cache [[buffer(2)]],
        device const int * token_ids [[buffer(3)]],
        device const int * selected_counts [[buffer(4)]],
        device const int * status [[buffer(5)]],
        device float * logits [[buffer(6)]],
        device float * output [[buffer(7)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    if (args.head_dim != 256u
            || args.kv_heads == 0u
            || args.query_heads % args.kv_heads != 0u
            || args.ratio == 0u
            || group.x >= args.query_heads
            || group.y >= args.query_count) return;
    const uint query_head = group.x;
    const uint query_index = group.y;
    const uint sequence_length = args.start_position + query_index + 1u;
    const int selected_i = selected_counts[query_index];
    const bool metadata_valid = status[query_index] == 0
        && selected_i == int(args.block_budget);
    const uint id_count = metadata_valid
        ? uint(selected_i) * args.ratio + sequence_length % args.ratio
        : 0u;
    const ulong output_index = ((ulong)query_index * args.query_heads + query_head)
        * args.head_dim + lane;
    if (!metadata_valid || id_count == 0u || id_count > args.row_stride) {
        output[output_index] = 0.0f;
        return;
    }
    device float * head_logits = logits
        + ((ulong)query_index * args.query_heads + query_head) * args.row_stride;
    float local_maximum = -INFINITY;
    for (uint slot = lane; slot < id_count; slot += 256u) {
        local_maximum = max(local_maximum, head_logits[slot]);
    }
    local_maximum = simd_max(local_maximum);
    if (simd_lane == 0u) scratch[uint(simdgroup)] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        const float candidate = simd_lane < 8u ? scratch[uint(simd_lane)] : -INFINITY;
        const float maximum = simd_max(candidate);
        if (simd_lane == 0u) scratch[8] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maximum = scratch[8];

    float local_denominator = 0.0f;
    for (uint slot = lane; slot < id_count; slot += 256u) {
        const float mass = exp(head_logits[slot] - maximum);
        head_logits[slot] = mass;
        local_denominator += mass;
    }
    local_denominator = simd_sum(local_denominator);
    if (simd_lane == 0u) scratch[uint(simdgroup)] = local_denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (simdgroup == 0u) {
        const float candidate = simd_lane < 8u ? scratch[uint(simd_lane)] : 0.0f;
        const float denominator = simd_sum(candidate);
        if (simd_lane == 0u) scratch[8] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kv_head = query_head / (args.query_heads / args.kv_heads);
    const ulong id_start = (ulong)query_index * args.row_stride;
    float accumulator = 0.0f;
    for (uint slot = 0u; slot < id_count; ++slot) {
        const int position = token_ids[id_start + slot];
        if (position < 0
                || uint(position) >= args.cache_capacity
                || uint(position) >= sequence_length) continue;
        const ulong value_start = ((ulong)uint(position) * args.kv_heads + kv_head)
            * args.head_dim;
        accumulator += float(value_cache[value_start + lane]) * head_logits[slot];
    }
    const ulong gate_index = ((ulong)query_index * args.query_heads + query_head)
        * args.head_dim * 2u + args.head_dim + lane;
    const float gate = 1.0f / (1.0f + exp(-query_gate_projection[gate_index]));
    output[output_index] = scratch[8] > 0.0f
        ? accumulator / scratch[8] * gate
        : 0.0f;
}

kernel void kernel_qwen4exp_qsa_audit_selected_i32(
        constant qwen4exp_qsa_selected_audit_args & args [[buffer(0)]],
        device const int * selected_counts [[buffer(1)]],
        device const int * row_status [[buffer(2)]],
        device int * workspace_status [[buffer(3)]],
        device int * workspace_selected_count [[buffer(4)]],
        device int * audited_bands [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    if (index != 0u || args.query_count == 0u) return;
    int status = workspace_status[0];
    if (status == 0) {
        for (uint query = 0u; query < args.query_count; ++query) {
            if (row_status[query] != 0) {
                status = row_status[query];
                break;
            }
        }
    }
    if (status == 0) {
        for (uint query = 0u; query < args.query_count; ++query) {
            if (selected_counts[query] != args.expected_selected_count) {
                status = args.count_mismatch_status;
                break;
            }
        }
    }
    workspace_status[0] = status;
    workspace_selected_count[0] = selected_counts[args.query_count - 1u];
    audited_bands[0] += 1;
}
