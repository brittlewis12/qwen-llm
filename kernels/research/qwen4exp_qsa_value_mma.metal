// TEST ONLY: ordinary selected Flash prefill, not decode or a product route.
// ABI matches PackedAttentionArgs. Stage 1 retains the production GQA4 F32
// max/denominator reduction sequence and overwrites logits with masses in place.
// The only extra device scratch is [query_count, 24] F32 denominators.
// Stage 2: [16 padded heads, D64, K32], 128 threads, 10240 bytes normal TGM.
// Both MMA operands are F32; cache half values are upcast, never half-staged.
// K32 partials change PV summation order: this is NOT a bitwise claim.
#include <metal_stdlib>
using namespace metal;
#define QSA_MMA_UNROLL _Pragma("clang loop unroll(full)") for

struct qsa_value_mma_args {
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
static_assert(sizeof(qsa_value_mma_args) == 40);

inline bool qsa_value_mma_shape(constant qsa_value_mma_args & args) {
    return args.query_heads == 24u && args.kv_heads == 2u
        && args.head_dim == 256u && args.block_budget == 512u
        && args.ratio == 4u && args.row_stride == 2051u
        && args.query_count > 0u && args.query_count <= 32u;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_qwen4exp_qsa_selected_softmax_gqa4_f32_test(
        constant qsa_value_mma_args & args [[buffer(0)]],
        device const int * selected_counts [[buffer(1)]],
        device const int * status [[buffer(2)]],
        device float * logits [[buffer(3)]],
        device float * denominators [[buffer(4)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    if (!qsa_value_mma_shape(args)
            || group.x >= 3u || group.y >= 2u || group.z >= args.query_count) return;
    constexpr uint scratch_stride = 9u;
    const uint query_index = group.z;
    const uint first_head = group.y * 12u + group.x * 4u;
    const ulong sequence_length = ulong(args.start_position) + query_index + 1u;
    const int selected_i = selected_counts[query_index];
    const bool metadata_valid = status[query_index] == 0
        && selected_i == int(args.block_budget);
    const uint id_count = metadata_valid
        ? uint(selected_i) * args.ratio + uint(sequence_length % args.ratio) : 0u;
    const ulong denominator_start = ulong(query_index) * 24u + first_head;
    if (!metadata_valid || id_count == 0u || id_count > args.row_stride) {
        if (lane < 4u) denominators[denominator_start + lane] = 0.0f;
        return;
    }

    device float * logits_0 = logits
        + (ulong(query_index) * args.query_heads + first_head) * args.row_stride;
    device float * logits_1 = logits_0 + args.row_stride;
    device float * logits_2 = logits_1 + args.row_stride;
    device float * logits_3 = logits_2 + args.row_stride;
    float maximum_0 = -INFINITY;
    float maximum_1 = -INFINITY;
    float maximum_2 = -INFINITY;
    float maximum_3 = -INFINITY;
    for (uint slot = uint(lane); slot < id_count; slot += 256u) {
        maximum_0 = max(maximum_0, logits_0[slot]);
        maximum_1 = max(maximum_1, logits_1[slot]);
        maximum_2 = max(maximum_2, logits_2[slot]);
        maximum_3 = max(maximum_3, logits_3[slot]);
    }
    maximum_0 = simd_max(maximum_0);
    maximum_1 = simd_max(maximum_1);
    maximum_2 = simd_max(maximum_2);
    maximum_3 = simd_max(maximum_3);
    if (simd_lane == 0u) {
        scratch[0u * scratch_stride + uint(simdgroup)] = maximum_0;
        scratch[1u * scratch_stride + uint(simdgroup)] = maximum_1;
        scratch[2u * scratch_stride + uint(simdgroup)] = maximum_2;
        scratch[3u * scratch_stride + uint(simdgroup)] = maximum_3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        maximum_0 = simd_max(simd_lane < 8u ? scratch[0u * scratch_stride + uint(simd_lane)] : -INFINITY);
        maximum_1 = simd_max(simd_lane < 8u ? scratch[1u * scratch_stride + uint(simd_lane)] : -INFINITY);
        maximum_2 = simd_max(simd_lane < 8u ? scratch[2u * scratch_stride + uint(simd_lane)] : -INFINITY);
        maximum_3 = simd_max(simd_lane < 8u ? scratch[3u * scratch_stride + uint(simd_lane)] : -INFINITY);
        if (simd_lane == 0u) {
            scratch[0u * scratch_stride + 8u] = maximum_0;
            scratch[1u * scratch_stride + 8u] = maximum_1;
            scratch[2u * scratch_stride + 8u] = maximum_2;
            scratch[3u * scratch_stride + 8u] = maximum_3;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum_0 = scratch[0u * scratch_stride + 8u];
    maximum_1 = scratch[1u * scratch_stride + 8u];
    maximum_2 = scratch[2u * scratch_stride + 8u];
    maximum_3 = scratch[3u * scratch_stride + 8u];

    float denominator_0 = 0.0f;
    float denominator_1 = 0.0f;
    float denominator_2 = 0.0f;
    float denominator_3 = 0.0f;
    for (uint slot = uint(lane); slot < id_count; slot += 256u) {
        const float mass_0 = exp(logits_0[slot] - maximum_0);
        const float mass_1 = exp(logits_1[slot] - maximum_1);
        const float mass_2 = exp(logits_2[slot] - maximum_2);
        const float mass_3 = exp(logits_3[slot] - maximum_3);
        logits_0[slot] = mass_0;
        logits_1[slot] = mass_1;
        logits_2[slot] = mass_2;
        logits_3[slot] = mass_3;
        denominator_0 += mass_0;
        denominator_1 += mass_1;
        denominator_2 += mass_2;
        denominator_3 += mass_3;
    }
    denominator_0 = simd_sum(denominator_0);
    denominator_1 = simd_sum(denominator_1);
    denominator_2 = simd_sum(denominator_2);
    denominator_3 = simd_sum(denominator_3);
    if (simd_lane == 0u) {
        scratch[0u * scratch_stride + uint(simdgroup)] = denominator_0;
        scratch[1u * scratch_stride + uint(simdgroup)] = denominator_1;
        scratch[2u * scratch_stride + uint(simdgroup)] = denominator_2;
        scratch[3u * scratch_stride + uint(simdgroup)] = denominator_3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (simdgroup == 0u) {
        denominator_0 = simd_sum(simd_lane < 8u ? scratch[0u * scratch_stride + uint(simd_lane)] : 0.0f);
        denominator_1 = simd_sum(simd_lane < 8u ? scratch[1u * scratch_stride + uint(simd_lane)] : 0.0f);
        denominator_2 = simd_sum(simd_lane < 8u ? scratch[2u * scratch_stride + uint(simd_lane)] : 0.0f);
        denominator_3 = simd_sum(simd_lane < 8u ? scratch[3u * scratch_stride + uint(simd_lane)] : 0.0f);
        if (simd_lane == 0u) {
            denominators[denominator_start] = denominator_0;
            denominators[denominator_start + 1u] = denominator_1;
            denominators[denominator_start + 2u] = denominator_2;
            denominators[denominator_start + 3u] = denominator_3;
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_qwen4exp_qsa_selected_value_m16d64k32_f32_test(
        constant qsa_value_mma_args & args [[buffer(0)]],
        device const float * query_gate_projection [[buffer(1)]],
        device const half * value_cache [[buffer(2)]],
        device const int * token_ids [[buffer(3)]],
        device const int * selected_counts [[buffer(4)]],
        device const int * status [[buffer(5)]],
        device const float * masses [[buffer(6)]],
        device float * output [[buffer(7)]],
        device const float * denominators [[buffer(8)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (!qsa_value_mma_shape(args)
            || group.x >= 4u || group.y >= 2u || group.z >= args.query_count) return;
    const uint query_index = group.z;
    const uint first_head = group.y * 12u;
    const uint d0 = group.x * 64u;
    const ulong sequence_length = ulong(args.start_position) + query_index + 1u;
    const int selected_i = selected_counts[query_index];
    const bool metadata_valid = status[query_index] == 0
        && selected_i == int(args.block_budget);
    const uint id_count = metadata_valid
        ? uint(selected_i) * args.ratio + uint(sequence_length % args.ratio) : 0u;
    if (!metadata_valid || id_count == 0u || id_count > args.row_stride) {
        for (uint cell = uint(tid); cell < 12u * 64u; cell += 128u) {
            const ulong out = (ulong(query_index) * 24u + first_head + cell / 64u) * 256u
                + d0 + cell % 64u;
            output[out] = 0.0f;
        }
        return; // Uniform TG exit; no thread can bypass a later barrier.
    }

    threadgroup float * probabilities = scratch; // [16, 32]
    threadgroup float * values = scratch + 16u * 32u; // [32, 64]
    const ulong id_start = ulong(query_index) * args.row_stride;
    simdgroup_float8x8 acc[2][2];
    QSA_MMA_UNROLL (ushort h = 0; h < 2; ++h) {
        QSA_MMA_UNROLL (ushort d = 0; d < 2; ++d) {
            acc[h][d] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }
    for (uint k0 = 0u; k0 < id_count; k0 += 32u) {
        for (uint cell = uint(tid); cell < 16u * 32u; cell += 128u) {
            const uint head = cell / 32u;
            const uint slot = k0 + cell % 32u;
            float mass = 0.0f;
            if (head < 12u && slot < id_count) {
                const int position = token_ids[id_start + slot];
                const ulong head_index = ulong(query_index) * 24u + first_head + head;
                if (position >= 0 && uint(position) < args.cache_capacity
                        && ulong(position) < sequence_length && denominators[head_index] > 0.0f) {
                    mass = masses[head_index * args.row_stride + slot];
                }
            }
            probabilities[cell] = mass;
        }
        for (uint cell = uint(tid); cell < 32u * 64u; cell += 128u) {
            const uint slot = k0 + cell / 64u;
            const int position = slot < id_count ? token_ids[id_start + slot] : -1;
            float value = 0.0f;
            if (position >= 0 && uint(position) < args.cache_capacity
                    && ulong(position) < sequence_length) {
                const ulong offset = (ulong(position) * 2u + group.y) * 256u
                    + d0 + cell % 64u;
                value = float(value_cache[offset]);
            }
            values[cell] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 partial[2][2];
        QSA_MMA_UNROLL (ushort h = 0; h < 2; ++h) {
            QSA_MMA_UNROLL (ushort d = 0; d < 2; ++d) {
                partial[h][d] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
        }
        QSA_MMA_UNROLL (ushort k = 0; k < 4; ++k) {
            simdgroup_float8x8 v[2];
            QSA_MMA_UNROLL (ushort d = 0; d < 2; ++d) {
                simdgroup_load(v[d], values + uint(k) * 8u * 64u
                    + uint(simdgroup) * 16u + uint(d) * 8u, 64);
            }
            QSA_MMA_UNROLL (ushort h = 0; h < 2; ++h) {
                simdgroup_float8x8 p;
                simdgroup_load(p, probabilities + uint(h) * 8u * 32u + uint(k) * 8u, 32);
                QSA_MMA_UNROLL (ushort d = 0; d < 2; ++d) {
                    simdgroup_multiply_accumulate(partial[h][d], p, v[d], partial[h][d]);
                }
            }
        }
        QSA_MMA_UNROLL (ushort h = 0; h < 2; ++h) {
            QSA_MMA_UNROLL (ushort d = 0; d < 2; ++d) {
                QSA_MMA_UNROLL (ushort e = 0; e < 2; ++e) {
                    acc[h][d].thread_elements()[e] += partial[h][d].thread_elements()[e];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // All staging reads have finished; reuse its first 4096 bytes for [16,64].
    QSA_MMA_UNROLL (ushort h = 0; h < 2; ++h) {
        QSA_MMA_UNROLL (ushort d = 0; d < 2; ++d) {
            simdgroup_store(acc[h][d], scratch + uint(h) * 8u * 64u
                + uint(simdgroup) * 16u + uint(d) * 8u, 64);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint cell = uint(tid); cell < 12u * 64u; cell += 128u) {
        const ulong head = ulong(query_index) * 24u + first_head + cell / 64u;
        const uint dimension = d0 + cell % 64u;
        const float denominator = denominators[head];
        const float gate = 1.0f / (1.0f + exp(-query_gate_projection[head * 512u + 256u + dimension]));
        output[head * 256u + dimension] = denominator > 0.0f
            ? scratch[cell] / denominator * gate : 0.0f;
    }
}

// One bounded arithmetic-preserving reuse falsifier. Each lane owns one D
// coordinate and twelve independent scalar chains, in production slot order.
// No explicit reassociation, MMA, normalized probabilities, or precision change;
// compiled exact bits remain a leaf gate under the ordinary math defaults.
[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_qwen4exp_qsa_selected_value_gqa12_f32_test(
        constant qsa_value_mma_args & args [[buffer(0)]],
        device const float * query_gate_projection [[buffer(1)]],
        device const half * value_cache [[buffer(2)]],
        device const int * token_ids [[buffer(3)]],
        device const int * selected_counts [[buffer(4)]],
        device const int * status [[buffer(5)]],
        device const float * masses [[buffer(6)]],
        device float * output [[buffer(7)]],
        device const float * denominators [[buffer(8)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_threadgroup]]) {
    if (!qsa_value_mma_shape(args)
            || group.x != 0u || group.y >= 2u || group.z >= args.query_count) return;
    const uint query_index = group.z;
    const uint first_head = group.y * 12u;
    const uint sequence_length = args.start_position + query_index + 1u;
    const int selected_i = selected_counts[query_index];
    const bool metadata_valid = status[query_index] == 0
        && selected_i == int(args.block_budget);
    const uint id_count = metadata_valid
        ? uint(selected_i) * args.ratio + sequence_length % args.ratio : 0u;
    const ulong output_start = ((ulong)query_index * args.query_heads + first_head)
        * args.head_dim;
    if (!metadata_valid || id_count == 0u || id_count > args.row_stride) {
        for (uint head = 0u; head < 12u; ++head) {
            output[output_start + (ulong)head * args.head_dim + lane] = 0.0f;
        }
        return;
    }

    device const float * logits_0 = masses
        + ((ulong)query_index * args.query_heads + first_head) * args.row_stride;
    device const float * logits_1 = logits_0 + args.row_stride;
    device const float * logits_2 = logits_1 + args.row_stride;
    device const float * logits_3 = logits_2 + args.row_stride;
    device const float * logits_4 = logits_3 + args.row_stride;
    device const float * logits_5 = logits_4 + args.row_stride;
    device const float * logits_6 = logits_5 + args.row_stride;
    device const float * logits_7 = logits_6 + args.row_stride;
    device const float * logits_8 = logits_7 + args.row_stride;
    device const float * logits_9 = logits_8 + args.row_stride;
    device const float * logits_10 = logits_9 + args.row_stride;
    device const float * logits_11 = logits_10 + args.row_stride;
    const ulong id_start = (ulong)query_index * args.row_stride;
    float accumulator_0 = 0.0f;
    float accumulator_1 = 0.0f;
    float accumulator_2 = 0.0f;
    float accumulator_3 = 0.0f;
    float accumulator_4 = 0.0f;
    float accumulator_5 = 0.0f;
    float accumulator_6 = 0.0f;
    float accumulator_7 = 0.0f;
    float accumulator_8 = 0.0f;
    float accumulator_9 = 0.0f;
    float accumulator_10 = 0.0f;
    float accumulator_11 = 0.0f;
    for (uint slot = 0u; slot < id_count; ++slot) {
        const int position = token_ids[id_start + slot];
        if (position < 0
                || uint(position) >= args.cache_capacity
                || uint(position) >= sequence_length) continue;
        const ulong value_start = ((ulong)uint(position) * args.kv_heads + group.y)
            * args.head_dim;
        const float value = float(value_cache[value_start + lane]);
        accumulator_0 += value * logits_0[slot];
        accumulator_1 += value * logits_1[slot];
        accumulator_2 += value * logits_2[slot];
        accumulator_3 += value * logits_3[slot];
        accumulator_4 += value * logits_4[slot];
        accumulator_5 += value * logits_5[slot];
        accumulator_6 += value * logits_6[slot];
        accumulator_7 += value * logits_7[slot];
        accumulator_8 += value * logits_8[slot];
        accumulator_9 += value * logits_9[slot];
        accumulator_10 += value * logits_10[slot];
        accumulator_11 += value * logits_11[slot];
    }
    const ulong gate_start = ((ulong)query_index * args.query_heads + first_head)
        * args.head_dim * 2u + args.head_dim + lane;
    const ulong gate_stride = args.head_dim * 2u;
    const float gate_0 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start]));
    const float gate_1 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start + gate_stride]));
    const float gate_2 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start + 2u * gate_stride]));
    const float gate_3 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start + 3u * gate_stride]));
    const float gate_4 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start + 4u * gate_stride]));
    const float gate_5 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start + 5u * gate_stride]));
    const float gate_6 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start + 6u * gate_stride]));
    const float gate_7 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start + 7u * gate_stride]));
    const float gate_8 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start + 8u * gate_stride]));
    const float gate_9 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start + 9u * gate_stride]));
    const float gate_10 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start + 10u * gate_stride]));
    const float gate_11 = 1.0f / (1.0f + exp(-query_gate_projection[gate_start + 11u * gate_stride]));
    const ulong denominator_start = (ulong)query_index * args.query_heads + first_head;
    const float denominator_0 = denominators[denominator_start];
    const float denominator_1 = denominators[denominator_start + 1u];
    const float denominator_2 = denominators[denominator_start + 2u];
    const float denominator_3 = denominators[denominator_start + 3u];
    const float denominator_4 = denominators[denominator_start + 4u];
    const float denominator_5 = denominators[denominator_start + 5u];
    const float denominator_6 = denominators[denominator_start + 6u];
    const float denominator_7 = denominators[denominator_start + 7u];
    const float denominator_8 = denominators[denominator_start + 8u];
    const float denominator_9 = denominators[denominator_start + 9u];
    const float denominator_10 = denominators[denominator_start + 10u];
    const float denominator_11 = denominators[denominator_start + 11u];
    output[output_start + lane] = denominator_0 > 0.0f
        ? accumulator_0 / denominator_0 * gate_0 : 0.0f;
    output[output_start + args.head_dim + lane] = denominator_1 > 0.0f
        ? accumulator_1 / denominator_1 * gate_1 : 0.0f;
    output[output_start + 2u * args.head_dim + lane] = denominator_2 > 0.0f
        ? accumulator_2 / denominator_2 * gate_2 : 0.0f;
    output[output_start + 3u * args.head_dim + lane] = denominator_3 > 0.0f
        ? accumulator_3 / denominator_3 * gate_3 : 0.0f;
    output[output_start + 4u * args.head_dim + lane] = denominator_4 > 0.0f
        ? accumulator_4 / denominator_4 * gate_4 : 0.0f;
    output[output_start + 5u * args.head_dim + lane] = denominator_5 > 0.0f
        ? accumulator_5 / denominator_5 * gate_5 : 0.0f;
    output[output_start + 6u * args.head_dim + lane] = denominator_6 > 0.0f
        ? accumulator_6 / denominator_6 * gate_6 : 0.0f;
    output[output_start + 7u * args.head_dim + lane] = denominator_7 > 0.0f
        ? accumulator_7 / denominator_7 * gate_7 : 0.0f;
    output[output_start + 8u * args.head_dim + lane] = denominator_8 > 0.0f
        ? accumulator_8 / denominator_8 * gate_8 : 0.0f;
    output[output_start + 9u * args.head_dim + lane] = denominator_9 > 0.0f
        ? accumulator_9 / denominator_9 * gate_9 : 0.0f;
    output[output_start + 10u * args.head_dim + lane] = denominator_10 > 0.0f
        ? accumulator_10 / denominator_10 * gate_10 : 0.0f;
    output[output_start + 11u * args.head_dim + lane] = denominator_11 > 0.0f
        ? accumulator_11 / denominator_11 * gate_11 : 0.0f;
}
