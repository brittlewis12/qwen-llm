// Qwen3.8-Flash-Next gated-residual elementwise kernels.

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

struct qwen4exp_qsa_index_score_args {
    uint visible_blocks;
};

struct qwen4exp_qsa_ids_args {
    uint block_budget;
    uint visible_blocks;
    uint ratio;
    uint sequence_length;
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
