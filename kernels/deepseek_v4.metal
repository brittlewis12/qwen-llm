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

struct ds4_packed_route_args {
    uint expert_count;
    uint top_k;
    uint n_tokens;
    uint produced_tokens;
    uint vocab_size;
    uint generation;
    float routed_scale;
};

struct ds4_packed_route_schedule_args {
    uint expert_count;
    uint top_k;
    uint n_tokens;
    uint generation;
    uint max_tiles32;
    uint max_tiles16;
};

struct ds4_packed_route_tile {
    uint expert;
    uint start;
    uint count;
};

static_assert(sizeof(ds4_packed_route_tile) == 12);

constant int DS4_ROUTE_PENDING = 0;
constant int DS4_ROUTE_READY = 1;
constant int DS4_ROUTE_NONFINITE_LOGIT = -1;
constant int DS4_ROUTE_NONFINITE_BIAS = -2;
constant int DS4_ROUTE_INVALID_TOKEN = -3;
constant int DS4_ROUTE_INVALID_EXPERT = -4;
constant int DS4_ROUTE_DUPLICATE_EXPERT = -5;
constant int DS4_ROUTE_NONFINITE_WEIGHT = -6;

constant int DS4_PACKED_ROUTE_STALE_ROUTE = -101;
constant int DS4_PACKED_ROUTE_FAILED_ROUTE = -102;
constant int DS4_PACKED_ROUTE_INVALID_ID = -103;
constant int DS4_PACKED_ROUTE_DUPLICATE_ID = -104;
constant int DS4_PACKED_ROUTE_INVALID_WEIGHT = -105;
constant int DS4_PACKED_ROUTE_STALE_SCHEDULE = -106;
constant int DS4_PACKED_ROUTE_INVALID_COUNT = -107;
constant int DS4_PACKED_ROUTE_INVALID_SCHEDULE = -108;
constant int DS4_PACKED_ROUTE_INVALID_PADDING = -109;
constant int DS4_PACKED_ROUTE_INVALID_TOTAL = -110;
constant int DS4_PACKED_ROUTE_INVALID_AGGREGATE = -200;
constant int DS4_PACKED_ROUTE_COMPACT_INVALID_ROUTE = -301;
constant int DS4_PACKED_ROUTE_COMPACT_OVERFLOW = -302;

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

inline uint ds4_packed_route_completion(uint generation, uint n_tokens) {
    return 0xd5510000u ^ generation ^ (n_tokens << 8u);
}

inline uint ds4_packed_route_signature_completion(uint generation, uint n_tokens) {
    return 0xd5520000u ^ generation ^ (n_tokens << 8u);
}

inline uint ds4_packed_route_compact_completion(uint generation, uint n_tokens) {
    return 0xd5530000u ^ generation ^ (n_tokens << 8u);
}

inline void ds4_packed_route_initialize(
        constant ds4_packed_route_args & args,
        device int * expert_ids,
        device float * weights,
        device int * status,
        uint token,
        uint tid) {
    if (tid != 0) return;
    const ulong base = (ulong)token * args.top_k;
    status[token] = DS4_ROUTE_PENDING;
    for (uint slot = 0; slot < args.top_k; ++slot) {
        expert_ids[base + slot] = -1;
        weights[base + slot] = 0.0f;
    }
}

inline void ds4_packed_route_finish(
        constant ds4_packed_route_args & args,
        device uint * generations,
        device int * status,
        uint token,
        int result) {
    generations[token] = args.generation;
    status[token] = result;
}

kernel void kernel_deepseek_v4_packed_route_learned(
        constant ds4_packed_route_args & args [[buffer(0)]],
        device const float * logits [[buffer(1)]],
        device const float * correction_bias [[buffer(2)]],
        device int * expert_ids [[buffer(3)]],
        device float * weights [[buffer(4)]],
        device uint * generations [[buffer(5)]],
        device int * status [[buffer(6)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint token = tgpig.y;
    if (token >= args.produced_tokens) return;
    threadgroup float group_scores[8];
    threadgroup uint group_ids[8];
    threadgroup uint selected_ids[6];
    threadgroup uint route_error;
    ds4_packed_route_initialize(args, expert_ids, weights, status, token, tid);
    const ulong logits_base = (ulong)token * args.expert_count;
    const ulong output_base = (ulong)token * args.top_k;
    const bool active = tid < args.expert_count;
    const float logit = active ? logits[logits_base + tid] : 0.0f;
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
            ds4_packed_route_finish(
                args,
                generations,
                status,
                token,
                route_error == 1u
                    ? DS4_ROUTE_NONFINITE_LOGIT
                    : DS4_ROUTE_NONFINITE_BIAS
            );
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
                expert_ids[output_base + slot] = int(best_id);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active && tid == selected_ids[slot]) selection_score = -INFINITY;
    }

    if (tid == 0) {
        float selected_weights[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float sum = 0.0f;
        for (uint slot = 0; slot < args.top_k; ++slot) {
            selected_weights[slot]
                = ds4_router_score_exact(logits[logits_base + selected_ids[slot]]);
            sum += selected_weights[slot];
        }
        const float denominator = max(sum, 6.1035156e-5f);
        for (uint slot = 0; slot < args.top_k; ++slot) {
            const float weight = selected_weights[slot] / denominator * args.routed_scale;
            if (!isfinite(weight)) {
                ds4_packed_route_finish(
                    args,
                    generations,
                    status,
                    token,
                    DS4_ROUTE_NONFINITE_WEIGHT
                );
                return;
            }
            weights[output_base + slot] = weight;
        }
        ds4_packed_route_finish(
            args,
            generations,
            status,
            token,
            DS4_ROUTE_READY
        );
    }
}

kernel void kernel_deepseek_v4_packed_route_hash(
        constant ds4_packed_route_args & args [[buffer(0)]],
        device const float * logits [[buffer(1)]],
        device const int * token_ids [[buffer(2)]],
        device const int * token_to_expert [[buffer(3)]],
        device int * expert_ids [[buffer(4)]],
        device float * weights [[buffer(5)]],
        device uint * generations [[buffer(6)]],
        device int * status [[buffer(7)]],
        uint token [[thread_position_in_grid]]) {
    if (token >= args.produced_tokens) return;
    ds4_packed_route_initialize(args, expert_ids, weights, status, token, 0u);
    const ulong logits_base = (ulong)token * args.expert_count;
    const ulong output_base = (ulong)token * args.top_k;
    const int token_id = token_ids[token];
    if (token_id < 0 || uint(token_id) >= args.vocab_size) {
        ds4_packed_route_finish(
            args,
            generations,
            status,
            token,
            DS4_ROUTE_INVALID_TOKEN
        );
        return;
    }
    for (uint expert = 0; expert < args.expert_count; ++expert) {
        if (!isfinite(logits[logits_base + expert])) {
            ds4_packed_route_finish(
                args,
                generations,
                status,
                token,
                DS4_ROUTE_NONFINITE_LOGIT
            );
            return;
        }
    }

    int selected[6] = {-1, -1, -1, -1, -1, -1};
    float selected_weights[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const ulong row = (ulong)uint(token_id) * args.top_k;
    for (uint slot = 0; slot < args.top_k; ++slot) {
        const int expert = token_to_expert[row + slot];
        if (expert < 0 || uint(expert) >= args.expert_count) {
            ds4_packed_route_finish(
                args,
                generations,
                status,
                token,
                DS4_ROUTE_INVALID_EXPERT
            );
            return;
        }
        for (uint prior = 0; prior < slot; ++prior) {
            if (selected[prior] == expert) {
                ds4_packed_route_finish(
                    args,
                    generations,
                    status,
                    token,
                    DS4_ROUTE_DUPLICATE_EXPERT
                );
                return;
            }
        }
        selected[slot] = expert;
        selected_weights[slot]
            = ds4_router_score_exact(logits[logits_base + uint(expert)]);
    }

    float sum = 0.0f;
    for (uint slot = 0; slot < args.top_k; ++slot) sum += selected_weights[slot];
    const float denominator = max(sum, 6.1035156e-5f);
    for (uint slot = 0; slot < args.top_k; ++slot) {
        const float weight = selected_weights[slot] / denominator * args.routed_scale;
        if (!isfinite(weight)) {
            ds4_packed_route_finish(
                args,
                generations,
                status,
                token,
                DS4_ROUTE_NONFINITE_WEIGHT
            );
            return;
        }
        expert_ids[output_base + slot] = selected[slot];
        weights[output_base + slot] = weight;
    }
    ds4_packed_route_finish(
        args,
        generations,
        status,
        token,
        DS4_ROUTE_READY
    );
}

kernel void kernel_deepseek_v4_packed_route_compact(
        constant ds4_packed_route_schedule_args & args [[buffer(0)]],
        device const uint * route_generations [[buffer(1)]],
        device const int * route_status [[buffer(2)]],
        device const int * expert_ids [[buffer(3)]],
        device const float * weights [[buffer(4)]],
        device int * expert_counts [[buffer(5)]],
        device int * compact_rows [[buffer(6)]],
        device int * compact_slots [[buffer(7)]],
        device ds4_packed_route_tile * tiles32 [[buffer(8)]],
        device ds4_packed_route_tile * tiles16 [[buffer(9)]],
        device int * header [[buffer(10)]],
        uint tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup uint counts[256];
    threadgroup uint row_offsets[256];
    threadgroup uint tile32_offsets[256];
    threadgroup uint tile16_offsets[256];
    threadgroup uint group_errors[8];
    threadgroup uint route_error;
    threadgroup uint total_routes;
    threadgroup uint active_experts;
    threadgroup uint total_tiles32;
    threadgroup uint total_tiles16;

    if (tid == 0u) {
        header[0] = int(args.generation);
        header[1] = DS4_ROUTE_PENDING;
        for (uint index = 2u; index < 8u; ++index) header[index] = 0;
    }
    for (uint index = tid; index < args.max_tiles32; index += 256u) {
        tiles32[index] = ds4_packed_route_tile{0u, 0u, 0u};
    }
    for (uint index = tid; index < args.max_tiles16; index += 256u) {
        tiles16[index] = ds4_packed_route_tile{0u, 0u, 0u};
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    uint local_error = 0u;
    for (uint token = tid; token < args.n_tokens; token += 256u) {
        if (route_generations[token] != args.generation
                || route_status[token] != DS4_ROUTE_READY) {
            local_error = 1u;
            continue;
        }
        const uint base = token * args.top_k;
        for (uint slot = 0u; slot < args.top_k; ++slot) {
            const int expert = expert_ids[base + slot];
            if (expert < 0 || uint(expert) >= args.expert_count) {
                local_error = 1u;
            }
            for (uint prior = 0u; prior < slot; ++prior) {
                if (expert_ids[base + prior] == expert) local_error = 1u;
            }
            const float weight = weights[base + slot];
            if (!isfinite(weight) || weight < 0.0f) local_error = 1u;
        }
    }
    const uint simd_error = simd_max(local_error);
    if (tiisg == 0) group_errors[sgitg] = simd_error;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const uint value = tiisg < 8 ? group_errors[tiisg] : 0u;
        const uint reduced = simd_max(value);
        if (tiisg == 0) route_error = reduced;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (route_error != 0u) {
        if (tid == 0u) {
            header[1] = DS4_PACKED_ROUTE_COMPACT_INVALID_ROUTE;
            header[6] = as_type<int>(
                ds4_packed_route_compact_completion(args.generation, args.n_tokens));
            header[7] = int(args.n_tokens);
        }
        return;
    }

    uint count = 0u;
    for (uint token = 0u; token < args.n_tokens; ++token) {
        const uint base = token * args.top_k;
        for (uint slot = 0u; slot < args.top_k; ++slot) {
            count += expert_ids[base + slot] == int(tid);
        }
    }
    counts[tid] = count;
    expert_counts[tid] = int(count);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_offset = 0u;
    uint tile32_offset = 0u;
    uint tile16_offset = 0u;
    for (uint expert = 0u; expert < tid; ++expert) {
        row_offset += counts[expert];
        tile32_offset += (counts[expert] + 31u) / 32u;
        tile16_offset += (counts[expert] + 15u) / 16u;
    }
    row_offsets[tid] = row_offset;
    tile32_offsets[tid] = tile32_offset;
    tile16_offsets[tid] = tile16_offset;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint routes = 0u;
        uint active = 0u;
        uint n32 = 0u;
        uint n16 = 0u;
        for (uint expert = 0u; expert < args.expert_count; ++expert) {
            routes += counts[expert];
            active += counts[expert] != 0u;
            n32 += (counts[expert] + 31u) / 32u;
            n16 += (counts[expert] + 15u) / 16u;
        }
        total_routes = routes;
        active_experts = active;
        total_tiles32 = n32;
        total_tiles16 = n16;
        route_error = routes != args.n_tokens * args.top_k
                || n32 > args.max_tiles32 || n16 > args.max_tiles16;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (route_error != 0u) {
        if (tid == 0u) {
            header[1] = DS4_PACKED_ROUTE_COMPACT_OVERFLOW;
            header[6] = as_type<int>(
                ds4_packed_route_compact_completion(args.generation, args.n_tokens));
            header[7] = int(args.n_tokens);
        }
        return;
    }

    uint cursor = row_offsets[tid];
    for (uint token = 0u; token < args.n_tokens; ++token) {
        const uint base = token * args.top_k;
        for (uint slot = 0u; slot < args.top_k; ++slot) {
            const uint global_slot = base + slot;
            if (expert_ids[global_slot] == int(tid)) {
                compact_rows[cursor] = int(token);
                compact_slots[cursor] = int(global_slot);
                cursor += 1u;
            }
        }
    }
    for (uint offset = 0u; offset < count; offset += 32u) {
        tiles32[tile32_offsets[tid] + offset / 32u] = ds4_packed_route_tile{
            tid,
            row_offsets[tid] + offset,
            min(32u, count - offset),
        };
    }
    for (uint offset = 0u; offset < count; offset += 16u) {
        tiles16[tile16_offsets[tid] + offset / 16u] = ds4_packed_route_tile{
            tid,
            row_offsets[tid] + offset,
            min(16u, count - offset),
        };
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0u) {
        header[1] = DS4_ROUTE_READY;
        header[2] = int(total_routes);
        header[3] = int(active_experts);
        header[4] = int(total_tiles16);
        header[5] = int(total_tiles32);
        header[6] = as_type<int>(
            ds4_packed_route_compact_completion(args.generation, args.n_tokens));
        header[7] = int(args.n_tokens);
    }
}

kernel void kernel_deepseek_v4_packed_route_schedule(
        constant ds4_packed_route_schedule_args & args [[buffer(0)]],
        device const int * expert_ids [[buffer(1)]],
        device int * counts [[buffer(2)]],
        device int * slot_ids [[buffer(3)]],
        device uint * generations [[buffer(4)]],
        uint expert [[thread_position_in_grid]]) {
    if (expert >= args.expert_count) return;
    const ulong schedule_base = (ulong)expert * args.n_tokens;
    for (uint index = 0; index < args.n_tokens; ++index) {
        slot_ids[schedule_base + index] = -1;
    }
    uint count = 0;
    for (uint token = 0; token < args.n_tokens; ++token) {
        const uint route_base = token * args.top_k;
        for (uint slot = 0; slot < args.top_k; ++slot) {
            if (expert_ids[route_base + slot] == int(expert)) {
                if (count < args.n_tokens) {
                    slot_ids[schedule_base + count] = int(route_base + slot);
                }
                count += 1;
            }
        }
    }
    counts[expert] = int(count);
    generations[expert] = args.generation;
}

kernel void kernel_deepseek_v4_packed_route_validate(
        constant ds4_packed_route_schedule_args & args [[buffer(0)]],
        device const uint * route_generations [[buffer(1)]],
        device const int * route_status [[buffer(2)]],
        device const int * expert_ids [[buffer(3)]],
        device const float * weights [[buffer(4)]],
        device const int * counts [[buffer(5)]],
        device const int * slot_ids [[buffer(6)]],
        device const uint * schedule_generations [[buffer(7)]],
        device uint * aggregate [[buffer(8)]],
        uint tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup uint group_errors[8];
    threadgroup uint group_totals[8];
    uint local_error = 0u;
    uint local_total = 0u;

    for (uint token = tid; token < args.n_tokens; token += 256u) {
        if (route_generations[token] != args.generation) {
            local_error = max(local_error, 1u);
        } else if (route_status[token] != DS4_ROUTE_READY) {
            local_error = max(local_error, 2u);
        } else {
            const uint route_base = token * args.top_k;
            for (uint slot = 0; slot < args.top_k; ++slot) {
                const int expert = expert_ids[route_base + slot];
                if (expert < 0 || uint(expert) >= args.expert_count) {
                    local_error = max(local_error, 3u);
                }
                for (uint prior = 0; prior < slot; ++prior) {
                    if (expert_ids[route_base + prior] == expert) {
                        local_error = max(local_error, 4u);
                    }
                }
                const float weight = weights[route_base + slot];
                if (!isfinite(weight) || weight < 0.0f) {
                    local_error = max(local_error, 5u);
                }
            }
        }
    }

    if (tid < args.expert_count) {
        if (schedule_generations[tid] != args.generation) {
            local_error = max(local_error, 6u);
        } else {
            const int count_i = counts[tid];
            if (count_i < 0 || uint(count_i) > args.n_tokens) {
                local_error = max(local_error, 7u);
            } else {
                const uint count = uint(count_i);
                const ulong schedule_base = (ulong)tid * args.n_tokens;
                uint expected_count = 0u;
                for (uint token = 0; token < args.n_tokens; ++token) {
                    const uint route_base = token * args.top_k;
                    for (uint slot = 0; slot < args.top_k; ++slot) {
                        if (expert_ids[route_base + slot] == int(tid)) {
                            const int expected_slot = int(route_base + slot);
                            if (expected_count >= count
                                    || slot_ids[schedule_base + expected_count]
                                        != expected_slot) {
                                local_error = max(local_error, 8u);
                            }
                            expected_count += 1u;
                        }
                    }
                }
                if (expected_count != count) local_error = max(local_error, 8u);
                for (uint index = count; index < args.n_tokens; ++index) {
                    if (slot_ids[schedule_base + index] != -1) {
                        local_error = max(local_error, 9u);
                    }
                }
                local_total = count;
            }
        }
    }

    const uint simd_error = simd_max(local_error);
    const uint simd_total = simd_sum(local_total);
    if (tiisg == 0) {
        group_errors[sgitg] = simd_error;
        group_totals[sgitg] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg != 0) return;
    const uint group_error = tiisg < 8 ? group_errors[tiisg] : 0u;
    const uint group_total = tiisg < 8 ? group_totals[tiisg] : 0u;
    uint final_error = simd_max(group_error);
    const uint final_total = simd_sum(group_total);
    if (tiisg != 0) return;
    if (final_error == 0u && final_total != args.n_tokens * args.top_k) {
        final_error = 10u;
    }
    const int result = final_error == 0u
        ? DS4_ROUTE_READY
        : (final_error == 1u ? DS4_PACKED_ROUTE_STALE_ROUTE
        : (final_error == 2u ? DS4_PACKED_ROUTE_FAILED_ROUTE
        : (final_error == 3u ? DS4_PACKED_ROUTE_INVALID_ID
        : (final_error == 4u ? DS4_PACKED_ROUTE_DUPLICATE_ID
        : (final_error == 5u ? DS4_PACKED_ROUTE_INVALID_WEIGHT
        : (final_error == 6u ? DS4_PACKED_ROUTE_STALE_SCHEDULE
        : (final_error == 7u ? DS4_PACKED_ROUTE_INVALID_COUNT
        : (final_error == 8u ? DS4_PACKED_ROUTE_INVALID_SCHEDULE
        : (final_error == 9u ? DS4_PACKED_ROUTE_INVALID_PADDING
        : DS4_PACKED_ROUTE_INVALID_TOTAL)))))))));
    aggregate[0] = args.generation;
    aggregate[1] = as_type<uint>(result);
    aggregate[2] = final_total;
    aggregate[3] = ds4_packed_route_completion(args.generation, args.n_tokens);
}

inline uint ds4_packed_route_signature_mix(uint hash, uint value) {
    return (hash ^ value) * 16777619u;
}

kernel void kernel_deepseek_v4_packed_route_signature(
        constant ds4_packed_route_schedule_args & args [[buffer(0)]],
        device const uint * aggregate [[buffer(1)]],
        device const int * expert_ids [[buffer(2)]],
        device const float * weights [[buffer(3)]],
        device const int * counts [[buffer(4)]],
        device const int * slot_ids [[buffer(5)]],
        device uint * signature [[buffer(6)]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint valid_aggregate;
    threadgroup uint partial_hashes[256];
    const uint expected_total = args.n_tokens * args.top_k;
    if (tid == 0) {
        signature[0] = args.generation;
        signature[1] = as_type<uint>(DS4_ROUTE_PENDING);
        signature[2] = 0u;
        signature[3] = 0u;
        valid_aggregate = aggregate[0] == args.generation
            && as_type<int>(aggregate[1]) == DS4_ROUTE_READY
            && aggregate[2] == expected_total
            && aggregate[3]
                == ds4_packed_route_completion(args.generation, args.n_tokens);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid_aggregate == 0u) {
        if (tid == 0) {
            signature[1] = as_type<uint>(DS4_PACKED_ROUTE_INVALID_AGGREGATE);
            signature[3]
                = ds4_packed_route_signature_completion(args.generation, args.n_tokens);
        }
        return;
    }

    uint hash = 2166136261u ^ tid;
    for (uint index = tid; index < expected_total; index += 256u) {
        hash = ds4_packed_route_signature_mix(hash, as_type<uint>(expert_ids[index]));
        hash = ds4_packed_route_signature_mix(hash, as_type<uint>(weights[index]));
    }
    if (tid < args.expert_count) {
        hash = ds4_packed_route_signature_mix(hash, as_type<uint>(counts[tid]));
        const ulong base = (ulong)tid * args.n_tokens;
        for (uint index = 0; index < args.n_tokens; ++index) {
            hash = ds4_packed_route_signature_mix(
                hash,
                as_type<uint>(slot_ids[base + index])
            );
        }
    }
    partial_hashes[tid] = hash;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;
    hash = 2166136261u;
    for (uint index = 0; index < 256u; ++index) {
        hash = ds4_packed_route_signature_mix(hash, partial_hashes[index]);
    }
    signature[1] = as_type<uint>(DS4_ROUTE_READY);
    signature[2] = hash;
    signature[3]
        = ds4_packed_route_signature_completion(args.generation, args.n_tokens);
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

struct ds4_rope_tail_batch_args {
    uint head_count;
    uint head_dim;
    uint rotary_dim;
    uint start_position;
    uint row_count;
    uint position_stride;
    uint inverse;
    uint yarn;
    float theta;
    float frequency_scale;
    float correction_low;
    float correction_high;
};

struct ds4_raw_chunk_publish_args {
    uint head_dim;
    uint row_count;
    uint start_position;
    uint window;
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
    uint raw_cache_is_chunk;
    float scale;
};

struct ds4_splitk_hca_args {
    uint head_count;
    uint head_dim;
    uint compression_ratio;
    uint start_position;
    uint window;
    uint raw_cache_is_chunk;
    uint partitions;
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
    uint raw_cache_is_chunk;
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
    uint raw_cache_is_chunk;
    float scale;
};

struct ds4_copy_u16_args {
    uint n;
};

struct ds4_compressor_frontier_args {
    uint width;
    uint row_offset;
};

struct ds4_compressor_chunk_args {
    uint ratio;
    uint head_dim;
    uint width;
    uint row_count;
    uint start_position;
    uint output_rows;
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

struct ds4_indexer_fp4_rows_args {
    uint row_count;
};

struct ds4_indexer_fp4_preflight_args {
    uint query_rows;
    uint row_capacity;
    uint expected_visible;
};

struct ds4_indexer_fp4_contract_args {
    uint e2m1_count;
    uint scale_count;
};

struct ds4_indexer_select_args {
    uint row_capacity;
    uint top_k;
    uint query_count;
    uint emit_ranked;
};

struct ds4_indexer_multigroup_select_args {
    uint row_capacity;
    uint top_k;
    uint group_count;
    uint generation;
    uint digit;
    uint shift;
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

static inline uchar ds4_indexer_e2m1_code(float value) {
    const float absolute = min(abs(value), 6.0f);
    const uchar magnitude = absolute > 5.0f ? 7u
        : absolute >= 3.5f ? 6u
        : absolute > 2.5f ? 5u
        : absolute >= 1.75f ? 4u
        : absolute > 1.25f ? 3u
        : absolute >= 0.75f ? 2u
        : absolute > 0.25f ? 1u
        : 0u;
    return magnitude | uchar((as_type<uint>(value) >> 28u) & 0x08u);
}

static inline half ds4_indexer_e2m1_unit(uchar code) {
    half magnitude = 0.0h;
    switch (code & 0x07u) {
        case 1u: magnitude = 0.5h; break;
        case 2u: magnitude = 1.0h; break;
        case 3u: magnitude = 1.5h; break;
        case 4u: magnitude = 2.0h; break;
        case 5u: magnitude = 3.0h; break;
        case 6u: magnitude = 4.0h; break;
        case 7u: magnitude = 6.0h; break;
        default: break;
    }
    return (code & 0x08u) == 0u ? magnitude : -magnitude;
}

static inline uchar ds4_indexer_ue8m0_scale_code(
        float maximum,
        thread uint & status) {
    const float amax_floor = as_type<float>(0x01c00000u);
    const float one_sixth = as_type<float>(0x3e2aaaabu);
    const float ratio = max(maximum, amax_floor) * one_sixth;
    const uint bits = as_type<uint>(ratio);
    const uint exponent = (bits >> 23u) & 0xffu;
    const uint mantissa = bits & 0x007fffffu;
    const uint code = exponent + uint(mantissa != 0u);
    if (exponent == 0u || exponent == 0xffu || code < 1u || code > 253u) {
        status = 2u;
        return 0u;
    }
    return uchar(code);
}

static inline float ds4_indexer_ue8m0_scale(uchar code) {
    return as_type<float>(uint(code) << 23u);
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

kernel void kernel_deepseek_v4_rope_tail_adjacent_batch_in_place(
        constant ds4_rope_tail_batch_args & args [[buffer(0)]],
        device float * values [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
    const uint pairs_per_head = args.rotary_dim / 2u;
    const uint pairs_per_row = args.head_count * pairs_per_head;
    const uint pair_count = args.row_count * pairs_per_row;
    if (index >= pair_count) return;
    const uint row = index / pairs_per_row;
    const uint row_pair = index - row * pairs_per_row;
    const uint head = row_pair / pairs_per_head;
    const uint pair = row_pair - head * pairs_per_head;
    const uint relative = pair * 2u;
    const uint row_start = row * args.head_count * args.head_dim;
    const uint tail = row_start + head * args.head_dim
        + args.head_dim - args.rotary_dim;
    const uint first_index = tail + relative;
    const uint second_index = first_index + 1u;

    const float extrapolated = float(args.start_position + row * args.position_stride)
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

kernel void kernel_deepseek_v4_publish_raw_chunk_f16(
        constant ds4_raw_chunk_publish_args & args [[buffer(0)]],
        device const float * source [[buffer(1)]],
        device half * chunk [[buffer(2)]],
        device half * ring [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint count = args.row_count * args.head_dim;
    if (index >= count) return;
    const uint row = index / args.head_dim;
    const uint dimension = index - row * args.head_dim;
    const half value = half(source[index]);
    chunk[index] = value;

    const uint retained_start = args.row_count > args.window
        ? args.row_count - args.window
        : 0u;
    if (row >= retained_start) {
        const uint position = args.start_position + row;
        ring[(position % args.window) * args.head_dim + dimension] = value;
    }
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
        const bool preserved = !compressed
            && args.raw_cache_is_chunk != 0u
            && logical_position < args.start_position;
        device const half * cache = compressed
            ? compressed_cache
            : (preserved ? preserved_raw_cache : raw_cache);
        const uint cache_start = compressed
            ? row * args.head_dim
            : (args.raw_cache_is_chunk == 0u || preserved
                ? (logical_position % args.window) * args.head_dim
                : (logical_position - args.start_position) * args.head_dim);
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
            const bool preserved = args.raw_cache_is_chunk != 0u
                && logical_position < args.start_position;
            device const half * cache = preserved ? preserved_raw_cache : raw_cache;
            const uint cache_start = args.raw_cache_is_chunk == 0u || preserved
                ? (logical_position % args.window) * args.head_dim
                : (logical_position - args.start_position) * args.head_dim;
            value += float(cache[cache_start + tid]) * masses[row];
        }
        for (uint row = 0u; row < compressed_count; ++row) {
            value += float(compressed_cache[row * args.head_dim + tid])
                * masses[raw_count + row];
        }
        output[query_start + tid] = value / masses[row_count];
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_deepseek_v4_grouped_online_dense_sink_attention_f16(
        constant ds4_packed_attention_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const half * raw_cache [[buffer(2)]],
        device const half * preserved_raw_cache [[buffer(3)]],
        device const half * compressed_cache [[buffer(4)]],
        device const float * sinks [[buffer(5)]],
        device float * output [[buffer(6)]],
        threadgroup half4 * staged [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort simdgroup_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint grouped_heads = 8u;
    constexpr uint staged_rows = 16u;
    constexpr uint row_vectors = 128u;
    const uint token = group.x;
    if (token >= args.n_tokens || args.head_count != 64u || args.head_dim != 512u) return;

    const uint tid = uint(tid_u);
    const uint lane = uint(lane_u);
    const uint head = group.y * grouped_heads + uint(simdgroup_u);
    const uint absolute_position = args.start_position + token;
    const uint visible_end = absolute_position + 1u;
    const uint raw_count = min(visible_end, args.window);
    const uint raw_start = visible_end - raw_count;
    const uint compressed_count = args.compression_ratio == 0u
        ? 0u
        : visible_end / args.compression_ratio;
    const uint row_count = raw_count + compressed_count;
    const uint query_start = (token * args.head_count + head) * args.head_dim;
    device const float4 * query4 = (device const float4 *)(queries + query_start);
    const float4 q0 = query4[lane];
    const float4 q1 = query4[lane + 32u];
    const float4 q2 = query4[lane + 64u];
    const float4 q3 = query4[lane + 96u];

    float maximum = sinks[head];
    float denominator = 1.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = 0u; base < row_count; base += staged_rows) {
        const uint rows = min(staged_rows, row_count - base);
        for (uint offset = tid; offset < rows * row_vectors; offset += 256u) {
            const uint staged_row = offset / row_vectors;
            const uint vector = offset - staged_row * row_vectors;
            const uint attention_row = base + staged_row;
            const bool compressed = attention_row >= raw_count;
            const uint row = compressed ? attention_row - raw_count : attention_row;
            const uint logical_position = raw_start + row;
            const bool preserved = !compressed
                && args.raw_cache_is_chunk != 0u
                && logical_position < args.start_position;
            device const half * cache = compressed
                ? compressed_cache
                : (preserved ? preserved_raw_cache : raw_cache);
            const uint cache_start = compressed
                ? row * args.head_dim
                : (args.raw_cache_is_chunk == 0u || preserved
                    ? (logical_position % args.window) * args.head_dim
                    : (logical_position - args.start_position) * args.head_dim);
            staged[offset] = ((device const half4 *)(cache + cache_start))[vector];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint staged_row = 0u; staged_row < rows; ++staged_row) {
            threadgroup const half4 * row = staged + staged_row * row_vectors;
            const half4 h0 = row[lane];
            const half4 h1 = row[lane + 32u];
            const half4 h2 = row[lane + 64u];
            const half4 h3 = row[lane + 96u];
            const float score = simd_sum(
                dot(q0, float4(h0)) +
                dot(q1, float4(h1)) +
                dot(q2, float4(h2)) +
                dot(q3, float4(h3))) * args.scale;

            if (score > maximum) {
                const float previous_scale = exp(maximum - score);
                denominator = denominator * previous_scale + 1.0f;
                o0 = o0 * previous_scale + float4(h0);
                o1 = o1 * previous_scale + float4(h1);
                o2 = o2 * previous_scale + float4(h2);
                o3 = o3 * previous_scale + float4(h3);
                maximum = score;
            } else {
                const float row_scale = exp(score - maximum);
                denominator += row_scale;
                o0 += float4(h0) * row_scale;
                o1 += float4(h1) * row_scale;
                o2 += float4(h2) * row_scale;
                o3 += float4(h3) * row_scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inverse = 1.0f / denominator;
    device float4 * output4 = (device float4 *)(output + query_start);
    output4[lane] = o0 * inverse;
    output4[lane + 32u] = o1 * inverse;
    output4[lane + 64u] = o2 * inverse;
    output4[lane + 96u] = o3 * inverse;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_deepseek_v4_grouped_splitk_hca_main_f16(
        constant ds4_splitk_hca_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const half * raw_cache [[buffer(2)]],
        device const half * preserved_raw_cache [[buffer(3)]],
        device const half * compressed_cache [[buffer(4)]],
        device float * partial_output [[buffer(5)]],
        device float * partial_ml [[buffer(6)]],
        threadgroup half4 * staged [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort simdgroup_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint grouped_heads = 8u;
    constexpr uint staged_rows = 16u;
    constexpr uint row_vectors = 128u;
    const uint partition = group.z;
    if (group.x != 0u || partition >= args.partitions
            || args.head_count != 64u || args.head_dim != 512u) return;

    const uint tid = uint(tid_u);
    const uint lane = uint(lane_u);
    const uint head = group.y * grouped_heads + uint(simdgroup_u);
    const uint visible_end = args.start_position + 1u;
    const uint raw_count = min(visible_end, args.window);
    const uint raw_start = visible_end - raw_count;
    const uint compressed_count = visible_end / args.compression_ratio;
    const uint row_count = raw_count + compressed_count;
    const uint partition_begin = uint((ulong(row_count) * partition) / args.partitions);
    const uint partition_end = uint((ulong(row_count) * (partition + 1u)) / args.partitions);
    const uint query_start = head * args.head_dim;
    device const float4 * query4 = (device const float4 *)(queries + query_start);
    const float4 q0 = query4[lane];
    const float4 q1 = query4[lane + 32u];
    const float4 q2 = query4[lane + 64u];
    const float4 q3 = query4[lane + 96u];

    float maximum = -INFINITY;
    float denominator = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = partition_begin; base < partition_end; base += staged_rows) {
        const uint rows = min(staged_rows, partition_end - base);
        for (uint offset = tid; offset < rows * row_vectors; offset += 256u) {
            const uint staged_row = offset / row_vectors;
            const uint vector = offset - staged_row * row_vectors;
            const uint attention_row = base + staged_row;
            const bool compressed = attention_row >= raw_count;
            const uint row = compressed ? attention_row - raw_count : attention_row;
            const uint logical_position = raw_start + row;
            const bool preserved = !compressed
                && args.raw_cache_is_chunk != 0u
                && logical_position < args.start_position;
            device const half * cache = compressed
                ? compressed_cache
                : (preserved ? preserved_raw_cache : raw_cache);
            const uint cache_start = compressed
                ? row * args.head_dim
                : (args.raw_cache_is_chunk == 0u || preserved
                    ? (logical_position % args.window) * args.head_dim
                    : (logical_position - args.start_position) * args.head_dim);
            staged[offset] = ((device const half4 *)(cache + cache_start))[vector];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint staged_row = 0u; staged_row < rows; ++staged_row) {
            threadgroup const half4 * row = staged + staged_row * row_vectors;
            const half4 h0 = row[lane];
            const half4 h1 = row[lane + 32u];
            const half4 h2 = row[lane + 64u];
            const half4 h3 = row[lane + 96u];
            const float score = simd_sum(
                dot(q0, float4(h0)) +
                dot(q1, float4(h1)) +
                dot(q2, float4(h2)) +
                dot(q3, float4(h3))) * args.scale;

            if (score > maximum) {
                const float previous_scale = exp(maximum - score);
                denominator = denominator * previous_scale + 1.0f;
                o0 = o0 * previous_scale + float4(h0);
                o1 = o1 * previous_scale + float4(h1);
                o2 = o2 * previous_scale + float4(h2);
                o3 = o3 * previous_scale + float4(h3);
                maximum = score;
            } else {
                const float row_scale = exp(score - maximum);
                denominator += row_scale;
                o0 += float4(h0) * row_scale;
                o1 += float4(h1) * row_scale;
                o2 += float4(h2) * row_scale;
                o3 += float4(h3) * row_scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong partial_head = ulong(partition) * args.head_count + head;
    device float4 * output4 = (device float4 *)(
        partial_output + partial_head * args.head_dim);
    output4[lane] = o0;
    output4[lane + 32u] = o1;
    output4[lane + 64u] = o2;
    output4[lane + 96u] = o3;
    if (lane == 0u) {
        device float * ml = partial_ml + partial_head * 2u;
        ml[0] = maximum;
        ml[1] = denominator;
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_deepseek_v4_grouped_splitk_hca_reduce_f32(
        constant ds4_splitk_hca_args & args [[buffer(0)]],
        device const float * partial_output [[buffer(1)]],
        device const float * partial_ml [[buffer(2)]],
        device const float * sinks [[buffer(3)]],
        device float * output [[buffer(4)]],
        threadgroup float * factors [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane_u [[thread_index_in_simdgroup]]) {
    const uint head = group.x;
    const uint lane = uint(lane_u);
    if (head >= args.head_count || args.head_dim != 512u) return;

    float partition_max = -INFINITY;
    float partition_mass = 0.0f;
    if (lane < args.partitions) {
        const ulong ml_offset = (ulong(lane) * args.head_count + head) * 2u;
        partition_max = partial_ml[ml_offset];
        partition_mass = partial_ml[ml_offset + 1u];
    }
    const float global_max = max(sinks[head], simd_max(partition_max));
    float factor = 0.0f;
    float mass = 0.0f;
    if (lane < args.partitions) {
        if (partition_mass > 0.0f) {
            factor = exp(partition_max - global_max);
            mass = partition_mass * factor;
        }
        factors[lane] = factor;
    }
    const float denominator = exp(sinks[head] - global_max) + simd_sum(mass);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse = 1.0f / denominator;
    device float4 * output4 = (device float4 *)(output + ulong(head) * args.head_dim);
    for (uint vector = lane; vector < 128u; vector += 32u) {
        float4 value = 0.0f;
        for (uint partition = 0u; partition < args.partitions; ++partition) {
            const ulong partial_head = ulong(partition) * args.head_count + head;
            device const float4 * partial4 = (device const float4 *)(
                partial_output + partial_head * args.head_dim);
            value += partial4[vector] * factors[partition];
        }
        output4[vector] = value * inverse;
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
            const bool preserved = !compressed
                && args.raw_cache_is_chunk != 0u
                && logical_position < args.chunk_start_position;
            device const half * cache = compressed
                ? compressed_cache
                : (preserved ? preserved_raw_cache : raw_cache);
            const uint cache_start = compressed
                ? row * args.head_dim
                : (args.raw_cache_is_chunk == 0u || preserved
                    ? (logical_position % args.window) * args.head_dim
                    : (logical_position - args.chunk_start_position) * args.head_dim);
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
            const bool preserved = !compressed
                && args.raw_cache_is_chunk != 0u
                && logical_position < args.chunk_start_position;
            device const half * cache = compressed
                ? compressed_cache
                : (preserved ? preserved_raw_cache : raw_cache);
            const uint cache_start = compressed
                ? row * args.head_dim
                : (args.raw_cache_is_chunk == 0u || preserved
                    ? (logical_position % args.window) * args.head_dim
                    : (logical_position - args.chunk_start_position) * args.head_dim);
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
                const bool preserved = !compressed
                    && args.raw_cache_is_chunk != 0u
                    && logical_position < args.chunk_start_position;
                device const half * cache = compressed
                    ? compressed_cache
                    : (preserved ? preserved_raw_cache : raw_cache);
                const uint cache_start = compressed
                    ? row * args.head_dim
                    : (args.raw_cache_is_chunk == 0u || preserved
                        ? (logical_position % args.window) * args.head_dim
                        : (logical_position - args.chunk_start_position) * args.head_dim);
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

static inline void deepseek_v4_online_attend_f16_row(
        device const half4 * source,
        threadgroup half4 * staged,
        float4 q0,
        float4 q1,
        float4 q2,
        float4 q3,
        float scale,
        ushort lane,
        thread float & maximum,
        thread float & denominator,
        thread float4 & o0,
        thread float4 & o1,
        thread float4 & o2,
        thread float4 & o3) {
    staged[lane] = source[lane];
    staged[lane + 32] = source[lane + 32];
    staged[lane + 64] = source[lane + 64];
    staged[lane + 96] = source[lane + 96];
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const half4 h0 = staged[lane];
    const half4 h1 = staged[lane + 32];
    const half4 h2 = staged[lane + 64];
    const half4 h3 = staged[lane + 96];
    const float score = simd_sum(
        dot(q0, float4(h0)) +
        dot(q1, float4(h1)) +
        dot(q2, float4(h2)) +
        dot(q3, float4(h3))) * scale;

    if (score > maximum) {
        const float previous_scale = exp(maximum - score);
        denominator = denominator * previous_scale + 1.0f;
        o0 = o0 * previous_scale + float4(h0);
        o1 = o1 * previous_scale + float4(h1);
        o2 = o2 * previous_scale + float4(h2);
        o3 = o3 * previous_scale + float4(h3);
        maximum = score;
    } else {
        const float row_scale = exp(score - maximum);
        denominator += row_scale;
        o0 += float4(h0) * row_scale;
        o1 += float4(h1) * row_scale;
        o2 += float4(h2) * row_scale;
        o3 += float4(h3) * row_scale;
    }
}

static inline void deepseek_v4_online_attend_f16_row_direct(
        device const half4 * source,
        float4 q0,
        float4 q1,
        float4 q2,
        float4 q3,
        float scale,
        ushort lane,
        thread float & maximum,
        thread float & denominator,
        thread float4 & o0,
        thread float4 & o1,
        thread float4 & o2,
        thread float4 & o3) {
    const half4 h0 = source[lane];
    const half4 h1 = source[lane + 32];
    const half4 h2 = source[lane + 64];
    const half4 h3 = source[lane + 96];
    const float score = simd_sum(
        dot(q0, float4(h0)) +
        dot(q1, float4(h1)) +
        dot(q2, float4(h2)) +
        dot(q3, float4(h3))) * scale;

    if (score > maximum) {
        const float previous_scale = exp(maximum - score);
        denominator = denominator * previous_scale + 1.0f;
        o0 = o0 * previous_scale + float4(h0);
        o1 = o1 * previous_scale + float4(h1);
        o2 = o2 * previous_scale + float4(h2);
        o3 = o3 * previous_scale + float4(h3);
        maximum = score;
    } else {
        const float row_scale = exp(score - maximum);
        denominator += row_scale;
        o0 += float4(h0) * row_scale;
        o1 += float4(h1) * row_scale;
        o2 += float4(h2) * row_scale;
        o3 += float4(h3) * row_scale;
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_deepseek_v4_online_dense_sink_attention_f16(
        constant ds4_tiled_dense_attention_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const half * raw_cache [[buffer(2)]],
        device const half * preserved_raw_cache [[buffer(3)]],
        device const half * compressed_cache [[buffer(4)]],
        device const float * sinks [[buffer(5)]],
        device float * output [[buffer(6)]],
        threadgroup half4 * staged [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint local_query = group.x;
    const uint head = group.y;
    if (local_query >= args.query_count || head >= args.head_count) return;

    const uint token = args.query_token_offset + local_query;
    const uint absolute_position = args.chunk_start_position + token;
    const uint visible_end = absolute_position + 1u;
    const uint raw_count = min(visible_end, args.window);
    const uint raw_start = visible_end - raw_count;
    const uint compressed_count = visible_end / args.compression_ratio;
    const uint query_start = (token * args.head_count + head) * args.head_dim;
    device const float4 * query4 = (device const float4 *)(queries + query_start);
    const float4 q0 = query4[lane];
    const float4 q1 = query4[lane + 32];
    const float4 q2 = query4[lane + 64];
    const float4 q3 = query4[lane + 96];

    float maximum = sinks[head];
    float denominator = 1.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint row = 0u; row < raw_count; ++row) {
        const uint logical_position = raw_start + row;
        const bool preserved = args.raw_cache_is_chunk != 0u
            && logical_position < args.chunk_start_position;
        device const half * cache = preserved ? preserved_raw_cache : raw_cache;
        const uint cache_start = args.raw_cache_is_chunk == 0u || preserved
            ? (logical_position % args.window) * args.head_dim
            : (logical_position - args.chunk_start_position) * args.head_dim;
        deepseek_v4_online_attend_f16_row(
            (device const half4 *)(cache + cache_start), staged,
            q0, q1, q2, q3, args.scale, lane,
            maximum, denominator, o0, o1, o2, o3);
    }
    for (uint row = 0u; row < compressed_count; ++row) {
        deepseek_v4_online_attend_f16_row(
            (device const half4 *)(compressed_cache + row * args.head_dim), staged,
            q0, q1, q2, q3, args.scale, lane,
            maximum, denominator, o0, o1, o2, o3);
    }

    const float inverse = 1.0f / denominator;
    device float4 * output4 = (device float4 *)(output + query_start);
    output4[lane] = o0 * inverse;
    output4[lane + 32] = o1 * inverse;
    output4[lane + 64] = o2 * inverse;
    output4[lane + 96] = o3 * inverse;
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_deepseek_v4_online_dense_sink_attention_f16_direct(
        constant ds4_tiled_dense_attention_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const half * raw_cache [[buffer(2)]],
        device const half * preserved_raw_cache [[buffer(3)]],
        device const half * compressed_cache [[buffer(4)]],
        device const float * sinks [[buffer(5)]],
        device float * output [[buffer(6)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint local_query = group.x;
    const uint head = group.y;
    if (local_query >= args.query_count || head >= args.head_count) return;

    const uint token = args.query_token_offset + local_query;
    const uint absolute_position = args.chunk_start_position + token;
    const uint visible_end = absolute_position + 1u;
    const uint raw_count = min(visible_end, args.window);
    const uint raw_start = visible_end - raw_count;
    const uint compressed_count = visible_end / args.compression_ratio;
    const uint query_start = (token * args.head_count + head) * args.head_dim;
    device const float4 * query4 = (device const float4 *)(queries + query_start);
    const float4 q0 = query4[lane];
    const float4 q1 = query4[lane + 32];
    const float4 q2 = query4[lane + 64];
    const float4 q3 = query4[lane + 96];

    float maximum = sinks[head];
    float denominator = 1.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint row = 0u; row < raw_count; ++row) {
        const uint logical_position = raw_start + row;
        const bool preserved = args.raw_cache_is_chunk != 0u
            && logical_position < args.chunk_start_position;
        device const half * cache = preserved ? preserved_raw_cache : raw_cache;
        const uint cache_start = args.raw_cache_is_chunk == 0u || preserved
            ? (logical_position % args.window) * args.head_dim
            : (logical_position - args.chunk_start_position) * args.head_dim;
        deepseek_v4_online_attend_f16_row_direct(
            (device const half4 *)(cache + cache_start),
            q0, q1, q2, q3, args.scale, lane,
            maximum, denominator, o0, o1, o2, o3);
    }
    for (uint row = 0u; row < compressed_count; ++row) {
        deepseek_v4_online_attend_f16_row_direct(
            (device const half4 *)(compressed_cache + row * args.head_dim),
            q0, q1, q2, q3, args.scale, lane,
            maximum, denominator, o0, o1, o2, o3);
    }

    const float inverse = 1.0f / denominator;
    device float4 * output4 = (device float4 *)(output + query_start);
    output4[lane] = o0 * inverse;
    output4[lane + 32] = o1 * inverse;
    output4[lane + 64] = o2 * inverse;
    output4[lane + 96] = o3 * inverse;
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
        const bool preserved = !compressed
            && args.raw_cache_is_chunk != 0u
            && logical_position < args.chunk_start_position;
        device const half * cache = compressed
            ? compressed_cache
            : (preserved ? preserved_raw_cache : raw_cache);
        const uint cache_start = compressed
            ? (valid_selected ? uint(selected_id) * args.head_dim : 0u)
            : (args.raw_cache_is_chunk == 0u || preserved
                ? (logical_position % args.window) * args.head_dim
                : (logical_position - args.chunk_start_position) * args.head_dim);
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
            const bool preserved = args.raw_cache_is_chunk != 0u
                && logical_position < args.chunk_start_position;
            device const half * cache = preserved ? preserved_raw_cache : raw_cache;
            const uint cache_start = args.raw_cache_is_chunk == 0u || preserved
                ? (logical_position % args.window) * args.head_dim
                : (logical_position - args.chunk_start_position) * args.head_dim;
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

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_deepseek_v4_online_packed_selected_sink_attention_f16(
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
        threadgroup half4 * staged [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
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
    const uint query_start = (token * args.head_count + head) * args.head_dim;
    const uint ids_base = local_query * args.selected_slots;
    device const float4 * query4 = (device const float4 *)(queries + query_start);
    const float4 q0 = query4[lane];
    const float4 q1 = query4[lane + 32];
    const float4 q2 = query4[lane + 64];
    const float4 q3 = query4[lane + 96];

    float maximum = sinks[head];
    float denominator = 1.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint row = 0u; row < raw_count; ++row) {
        const uint logical_position = raw_start + row;
        const bool preserved = args.raw_cache_is_chunk != 0u
            && logical_position < args.chunk_start_position;
        device const half * cache = preserved ? preserved_raw_cache : raw_cache;
        const uint cache_start = args.raw_cache_is_chunk == 0u || preserved
            ? (logical_position % args.window) * args.head_dim
            : (logical_position - args.chunk_start_position) * args.head_dim;
        deepseek_v4_online_attend_f16_row(
            (device const half4 *)(cache + cache_start), staged,
            q0, q1, q2, q3, args.scale, lane,
            maximum, denominator, o0, o1, o2, o3);
    }
    for (uint slot = 0u; slot < selected_count; ++slot) {
        const int selected_id = selected_ids[ids_base + slot];
        if (selected_id < 0 || uint(selected_id) >= visible_count
                || uint(selected_id) >= args.compressed_capacity) continue;
        deepseek_v4_online_attend_f16_row(
            (device const half4 *)(compressed_cache + uint(selected_id) * args.head_dim),
            staged, q0, q1, q2, q3, args.scale, lane,
            maximum, denominator, o0, o1, o2, o3);
    }

    const float inverse = 1.0f / denominator;
    device float4 * output4 = (device float4 *)(output + query_start);
    output4[lane] = o0 * inverse;
    output4[lane + 32] = o1 * inverse;
    output4[lane + 64] = o2 * inverse;
    output4[lane + 96] = o3 * inverse;
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_deepseek_v4_online_packed_selected_sink_attention_f16_direct(
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
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
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
    const uint query_start = (token * args.head_count + head) * args.head_dim;
    const uint ids_base = local_query * args.selected_slots;
    device const float4 * query4 = (device const float4 *)(queries + query_start);
    const float4 q0 = query4[lane];
    const float4 q1 = query4[lane + 32];
    const float4 q2 = query4[lane + 64];
    const float4 q3 = query4[lane + 96];

    float maximum = sinks[head];
    float denominator = 1.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint row = 0u; row < raw_count; ++row) {
        const uint logical_position = raw_start + row;
        const bool preserved = args.raw_cache_is_chunk != 0u
            && logical_position < args.chunk_start_position;
        device const half * cache = preserved ? preserved_raw_cache : raw_cache;
        const uint cache_start = args.raw_cache_is_chunk == 0u || preserved
            ? (logical_position % args.window) * args.head_dim
            : (logical_position - args.chunk_start_position) * args.head_dim;
        deepseek_v4_online_attend_f16_row_direct(
            (device const half4 *)(cache + cache_start),
            q0, q1, q2, q3, args.scale, lane,
            maximum, denominator, o0, o1, o2, o3);
    }
    for (uint slot = 0u; slot < selected_count; ++slot) {
        const int selected_id = selected_ids[ids_base + slot];
        if (selected_id < 0 || uint(selected_id) >= visible_count
                || uint(selected_id) >= args.compressed_capacity) continue;
        deepseek_v4_online_attend_f16_row_direct(
            (device const half4 *)(compressed_cache + uint(selected_id) * args.head_dim),
            q0, q1, q2, q3, args.scale, lane,
            maximum, denominator, o0, o1, o2, o3);
    }

    const float inverse = 1.0f / denominator;
    device float4 * output4 = (device float4 *)(output + query_start);
    output4[lane] = o0 * inverse;
    output4[lane + 32] = o1 * inverse;
    output4[lane + 64] = o2 * inverse;
    output4[lane + 96] = o3 * inverse;
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

kernel void kernel_deepseek_v4_compressor_frontier_chunk(
        constant ds4_compressor_chunk_args & args [[buffer(0)]],
        device const float * projected_kv [[buffer(1)]],
        device const float * projected_score [[buffer(2)]],
        device const float * ape [[buffer(3)]],
        device float * kv_state [[buffer(4)]],
        device float * score_state [[buffer(5)]],
        device float * pooled_rows [[buffer(6)]],
        uint dimension [[thread_position_in_grid]]) {
    if (dimension >= args.head_dim) return;
    uint emitted = 0u;
    for (uint token = 0u; token < args.row_count; ++token) {
        const uint position = args.start_position + token;
        const uint phase = position % args.ratio;
        const uint source_base = token * args.width;
        const uint ape_base = phase * args.width;
        if (args.ratio == 4u) {
            const uint state_base = (4u + phase) * args.width;
            for (uint lane = 0u; lane < 2u; ++lane) {
                const uint offset = lane * args.head_dim + dimension;
                kv_state[state_base + offset] = projected_kv[source_base + offset];
                score_state[state_base + offset] =
                    projected_score[source_base + offset] + ape[ape_base + offset];
            }
        } else {
            const uint state = phase * args.width + dimension;
            kv_state[state] = projected_kv[source_base + dimension];
            score_state[state] =
                projected_score[source_base + dimension] + ape[ape_base + dimension];
        }

        if ((position + 1u) % args.ratio != 0u) continue;
        float maximum = -INFINITY;
        if (args.ratio == 4u) {
            for (uint row = 0u; row < 4u; ++row) {
                maximum = max(maximum, score_state[row * args.width + dimension]);
                maximum = max(
                    maximum,
                    score_state[(4u + row) * args.width + args.head_dim + dimension]);
            }
        } else {
            for (uint row = 0u; row < args.ratio; ++row) {
                maximum = max(maximum, score_state[row * args.width + dimension]);
            }
        }

        float weighted = 0.0f;
        float denominator = 0.0f;
        if (args.ratio == 4u) {
            for (uint row = 0u; row < 4u; ++row) {
                const uint previous = row * args.width + dimension;
                const uint current =
                    (4u + row) * args.width + args.head_dim + dimension;
                const float previous_mass = isfinite(score_state[previous])
                    ? exp(score_state[previous] - maximum)
                    : 0.0f;
                const float current_mass = isfinite(score_state[current])
                    ? exp(score_state[current] - maximum)
                    : 0.0f;
                denominator += previous_mass + current_mass;
                weighted +=
                    kv_state[previous] * previous_mass + kv_state[current] * current_mass;
            }
        } else {
            for (uint row = 0u; row < args.ratio; ++row) {
                const uint source = row * args.width + dimension;
                const float mass = isfinite(score_state[source])
                    ? exp(score_state[source] - maximum)
                    : 0.0f;
                denominator += mass;
                weighted += kv_state[source] * mass;
            }
        }
        if (emitted < args.output_rows) {
            pooled_rows[emitted * args.head_dim + dimension] = weighted / denominator;
        }
        ++emitted;

        if (args.ratio == 4u) {
            const uint group_elements = 4u * args.width;
            for (uint row = 0u; row < 4u; ++row) {
                const uint previous = row * args.width;
                const uint current = group_elements + previous;
                for (uint lane = 0u; lane < 2u; ++lane) {
                    const uint offset = lane * args.head_dim + dimension;
                    kv_state[previous + offset] = kv_state[current + offset];
                    score_state[previous + offset] = score_state[current + offset];
                }
            }
        }
    }
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

kernel void kernel_deepseek_v4_indexer_fp4_contract_primitives(
        constant ds4_indexer_fp4_contract_args & args [[buffer(0)]],
        device const float * e2m1_values [[buffer(1)]],
        device const float * scale_maxima [[buffer(2)]],
        device uchar * e2m1_codes [[buffer(3)]],
        device uchar * scale_codes [[buffer(4)]],
        device int * scale_status [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    if (index < args.e2m1_count) {
        e2m1_codes[index] = ds4_indexer_e2m1_code(e2m1_values[index]);
    }
    if (index < args.scale_count) {
        uint status = 0u;
        scale_codes[index] = ds4_indexer_ue8m0_scale_code(scale_maxima[index], status);
        scale_status[index] = int(status);
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_deepseek_v4_pack_indexer_fp4_rows_shadow(
        constant ds4_indexer_fp4_rows_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device uchar * packed_values [[buffer(2)]],
        device uchar * packed_scales [[buffer(3)]],
        device int * status [[buffer(4)]],
        uint group [[threadgroup_position_in_grid]],
        ushort thread_index [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const uint row = group;
    if (row >= args.row_count) return;

    if (thread_index == 0) status[row] = INT_MIN + 1;

    threadgroup float rounded_values[128];
    threadgroup uchar row_values[64];
    threadgroup uchar row_scales[4];
    threadgroup uint block_status[4];
    const uint input_base = row * 128u;

    const uint block = uint(simdgroup);
    const uint dimension = block * 32u + uint(lane);
    const uint input_bits = as_type<uint>(input[input_base + dimension]);
    uint local_status = (input_bits & 0x7f800000u) == 0x7f800000u ? 1u : 0u;
    uint rounded_bits = input_bits;
    rounded_bits += 0x00007fffu + ((rounded_bits >> 16u) & 1u);
    rounded_bits &= 0xffff0000u;
    if ((rounded_bits & 0x7f800000u) == 0x7f800000u) local_status = 1u;
    const float rounded = local_status == 0u ? as_type<float>(rounded_bits) : 0.0f;
    rounded_values[dimension] = rounded;
    const uint reduced_status = simd_max(local_status);
    const float maximum = simd_max(as_type<float>(as_type<uint>(rounded) & 0x7fffffffu));
    if (lane == 0) {
        uint scale_status = reduced_status;
        const uchar scale_code = scale_status == 0u
            ? ds4_indexer_ue8m0_scale_code(maximum, scale_status)
            : 0u;
        block_status[block] = scale_status;
        row_scales[block] = scale_code;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local_overflow = 0u;
    if (lane < 16 && block_status[block] == 0u) {
        const uchar scale_code = row_scales[block];
        const float scale = ds4_indexer_ue8m0_scale(scale_code);
        const float inverse_scale = as_type<float>((254u - uint(scale_code)) << 23u);
        const uint pair_dimension = block * 32u + uint(lane) * 2u;
        const uchar low = ds4_indexer_e2m1_code(
            rounded_values[pair_dimension] * inverse_scale);
        const uchar high = ds4_indexer_e2m1_code(
            rounded_values[pair_dimension + 1u] * inverse_scale);
        if (!isfinite(float(ds4_indexer_e2m1_unit(low)) * scale)
                || !isfinite(float(ds4_indexer_e2m1_unit(high)) * scale)) {
            local_overflow = 1u;
        }
        row_values[block * 16u + uint(lane)] = low | uchar(high << 4u);
    }
    const uint block_overflow = simd_max(local_overflow);
    if (lane == 0 && block_status[block] == 0u && block_overflow != 0u) {
        block_status[block] = 3u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_status = 0u;
    for (uint candidate = 1u; candidate <= 3u; ++candidate) {
        for (uint index = 0u; index < 4u; ++index) {
            if (row_status == 0u && block_status[index] == candidate) row_status = candidate;
        }
    }
    if (row_status == 0u) {
        if (thread_index < 64) {
            packed_values[row * 64u + uint(thread_index)] = row_values[thread_index];
        }
        if (thread_index < 4) {
            packed_scales[row * 4u + uint(thread_index)] = row_scales[thread_index];
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (thread_index == 0) status[row] = int(row_status);
}

kernel void kernel_deepseek_v4_unpack_indexer_fp4_units_shadow(
        constant ds4_indexer_fp4_rows_args & args [[buffer(0)]],
        device const uchar * packed_values [[buffer(1)]],
        device const int * status [[buffer(2)]],
        device half * units [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const uint row = index / 128u;
    const uint dimension = index - row * 128u;
    if (row >= args.row_count) return;
    if (status[row] != 0) {
        units[index] = 0.0h;
        return;
    }
    const uchar packed = packed_values[row * 64u + dimension / 2u];
    const uchar code = (dimension & 1u) == 0u ? packed & 0x0fu : packed >> 4u;
    units[index] = ds4_indexer_e2m1_unit(code);
}

kernel void kernel_deepseek_v4_validate_indexer_fp4_rows_shadow(
        constant ds4_indexer_fp4_rows_args & args [[buffer(0)]],
        device const uchar * packed_values [[buffer(1)]],
        device const uchar * packed_scales [[buffer(2)]],
        device int * status [[buffer(3)]],
        uint row [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    if (row >= args.row_count) return;
    uint row_status = 0u;
    for (uint block = 0u; block < 4u; ++block) {
        const uchar scale_code = packed_scales[row * 4u + block];
        if (scale_code < 1u || scale_code > 253u) {
            row_status = 2u;
            break;
        }
        const float scale = ds4_indexer_ue8m0_scale(scale_code);
        for (uint pair = 0u; pair < 16u; ++pair) {
            const uchar packed = packed_values[row * 64u + block * 16u + pair];
            for (uint lane = 0u; lane < 2u; ++lane) {
                const uchar code = lane == 0u ? packed & 0x0fu : packed >> 4u;
                if (!isfinite(float(ds4_indexer_e2m1_unit(code)) * scale)) {
                    row_status = 3u;
                    break;
                }
            }
            if (row_status != 0u) break;
        }
        if (row_status != 0u) break;
    }
    status[row] = int(row_status);
}

kernel void kernel_deepseek_v4_indexer_fp4_shadow_preflight(
        constant ds4_indexer_fp4_preflight_args & args [[buffer(0)]],
        device const int * query_status [[buffer(1)]],
        device const int * key_status [[buffer(2)]],
        device const int * requested_visible [[buffer(3)]],
        device int * eligible_visible [[buffer(4)]],
        device int * eligibility_record [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    const int visible = requested_visible[0];
    if (visible <= 0 || uint(visible) > args.row_capacity
            || uint(visible) != args.expected_visible) {
        eligible_visible[0] = -1;
        eligibility_record[0] = 1;
        eligibility_record[1] = visible;
        eligibility_record[2] = int(args.expected_visible);
        return;
    }
    for (uint head = 0u; head < args.query_rows; ++head) {
        const int candidate = query_status[head];
        if (candidate != 0) {
            eligible_visible[0] = -1;
            eligibility_record[0] = 2;
            eligibility_record[1] = int(head);
            eligibility_record[2] = candidate;
            return;
        }
    }
    for (uint row = 0u; row < uint(visible); ++row) {
        const int candidate = key_status[row];
        if (candidate != 0) {
            eligible_visible[0] = -1;
            eligibility_record[0] = 3;
            eligibility_record[1] = int(row);
            eligibility_record[2] = candidate;
            return;
        }
    }
    eligible_visible[0] = visible;
    eligibility_record[0] = 0;
    eligibility_record[1] = -1;
    eligibility_record[2] = 0;
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

// The 8-query by 32-row tile topology is adapted from DwarfStar's
// kernel_dsv4_indexer_scores_tiled_f32 at revision b0309611 under MIT.
// See docs/THIRD-PARTY-NOTICES.md. This variant retains qwen-llm's physical
// score stride and explicit per-query visibility contract.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_deepseek_v4_lightning_indexer_scores_f16_tiled_f32(
        constant ds4_indexer_score_args & args [[buffer(0)]],
        device const float * queries [[buffer(1)]],
        device const float * head_weights [[buffer(2)]],
        device const half * keys [[buffer(3)]],
        device const int * visible_counts [[buffer(4)]],
        device float * scores [[buffer(5)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort simdgroup_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint query_tile_rows = 8u;
    constexpr uint key_tile_rows = 32u;
    constexpr uint matrix_rows = 8u;
    constexpr uint head_dim = 128u;
    if (args.head_count != 64u || args.head_dim != head_dim) return;

    const uint tid = uint(tid_u);
    const uint lane = uint(lane_u);
    const uint simdgroup = uint(simdgroup_u);
    const uint query_base = group.y * query_tile_rows;
    const uint key_base = group.x * key_tile_rows;
    threadgroup float * query_tile = scratch;
    threadgroup float * key_tile = query_tile + query_tile_rows * head_dim;
    threadgroup float * dots = key_tile + key_tile_rows * head_dim;

    for (uint element = tid; element < key_tile_rows * head_dim; element += 128u) {
        const uint local_row = element / head_dim;
        const uint dimension = element - local_row * head_dim;
        const uint row = key_base + local_row;
        key_tile[element] = row < args.row_capacity
            ? float(keys[row * head_dim + dimension])
            : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint query0 = query_base + cell0 / matrix_rows;
    const uint query1 = query_base + cell1 / matrix_rows;
    const uint row0 = key_base + simdgroup * matrix_rows + cell0 % matrix_rows;
    const uint row1 = key_base + simdgroup * matrix_rows + cell1 % matrix_rows;
    float score0 = 0.0f;
    float score1 = 0.0f;

    for (uint head = 0u; head < 64u; ++head) {
        for (uint element = tid; element < query_tile_rows * head_dim; element += 128u) {
            const uint local_query = element / head_dim;
            const uint dimension = element - local_query * head_dim;
            const uint query = query_base + local_query;
            query_tile[element] = query < args.query_count
                ? queries[(query * 64u + head) * head_dim + dimension]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 head_dots = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint dimension = 0u; dimension < head_dim; dimension += matrix_rows) {
            simdgroup_float8x8 query_matrix;
            simdgroup_float8x8 key_matrix;
            simdgroup_load(
                query_matrix,
                query_tile + dimension,
                head_dim,
                0u,
                false);
            simdgroup_load(
                key_matrix,
                key_tile + simdgroup * matrix_rows * head_dim + dimension,
                head_dim,
                0u,
                true);
            simdgroup_multiply_accumulate(
                head_dots,
                query_matrix,
                key_matrix,
                head_dots);
        }
        simdgroup_store(
            head_dots,
            dots + simdgroup * matrix_rows,
            key_tile_rows,
            0u,
            false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (query0 < args.query_count && row0 < args.row_capacity) {
            score0 += max(dots[(query0 - query_base) * key_tile_rows
                               + row0 - key_base], 0.0f)
                * head_weights[query0 * 64u + head];
        }
        if (query1 < args.query_count && row1 < args.row_capacity) {
            score1 += max(dots[(query1 - query_base) * key_tile_rows
                               + row1 - key_base], 0.0f)
                * head_weights[query1 * 64u + head];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query0 < args.query_count && row0 < args.row_capacity) {
        const int visible = visible_counts[query0];
        scores[query0 * args.row_capacity + row0] =
            visible >= 0 && row0 < uint(visible) ? score0 : -INFINITY;
    }
    if (query1 < args.query_count && row1 < args.row_capacity) {
        const int visible = visible_counts[query1];
        scores[query1 * args.row_capacity + row1] =
            visible >= 0 && row1 < uint(visible) ? score1 : -INFINITY;
    }
}

// Queries are rounded to F16 once outside this kernel; eight simdgroups compute
// the 64x8 head/row dot tile with matrix instructions and retain F32 head-weight
// reduction. This is distinct from the official FP4 numerical contract.
[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_deepseek_v4_lightning_indexer_scores_f16_matrix_ceiling(
        constant ds4_indexer_score_args & args [[buffer(0)]],
        device const half * queries [[buffer(1)]],
        device const float * head_weights [[buffer(2)]],
        device const half * keys [[buffer(3)]],
        device const int * visible_counts [[buffer(4)]],
        device float * scores [[buffer(5)]],
        threadgroup half * staged_keys [[threadgroup(0)]],
        threadgroup float * head_dots [[threadgroup(1)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort thread_index [[thread_index_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_group = 8u;
    const uint row_base = group.x * rows_per_group;
    const uint query = group.y;
    const int visible = visible_counts[query];

    // Store K transposed as [dimension, local row], which is the right-hand
    // 128x8 operand consumed by simdgroup matrix multiplication.
    for (uint element = uint(thread_index); element < 8u * 128u; element += 256u) {
        const uint local_row = element / 128u;
        const uint dimension = element - local_row * 128u;
        const uint row = row_base + local_row;
        const bool row_visible = visible >= 0 && row < args.row_capacity && row < uint(visible);
        staged_keys[dimension * 8u + local_row] = row_visible
            ? keys[row * 128u + dimension]
            : (half)0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint head_base = uint(simdgroup) * 8u;
    const uint query_base = query * 64u * 128u + head_base * 128u;
    simdgroup_float8x8 dots = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint dimension = 0u; dimension < 128u; dimension += 8u) {
        simdgroup_half8x8 query_tile;
        simdgroup_half8x8 key_tile;
        simdgroup_load(query_tile, queries + query_base + dimension, 128u, 0u, false);
        simdgroup_load(key_tile, staged_keys + dimension * 8u, 8u, 0u, false);
        simdgroup_multiply_accumulate(dots, query_tile, key_tile, dots);
    }
    simdgroup_store(dots, head_dots + head_base * 8u, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_index < 8u) {
        const uint local_row = uint(thread_index);
        const uint row = row_base + local_row;
        if (row < args.row_capacity) {
            const uint score_index = query * args.row_capacity + row;
            if (visible < 0 || row >= uint(visible)) {
                scores[score_index] = -INFINITY;
            } else {
                const uint weight_base = query * 64u;
                float score = 0.0f;
                for (uint head = 0u; head < 64u; ++head) {
                    score += max(head_dots[head * 8u + local_row], 0.0f)
                        * head_weights[weight_base + head];
                }
                scores[score_index] = score;
            }
        }
    }
}

// Test-only packed-semantic shadow. Values and scales are separate raw-byte
// planes; this deliberately does not define the eventual paged cache ABI.
[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_deepseek_v4_lightning_indexer_scores_fp4_matrix_shadow(
        constant ds4_indexer_score_args & args [[buffer(0)]],
        device const half * query_units [[buffer(1)]],
        device const uchar * query_scales [[buffer(2)]],
        device const float * head_weights [[buffer(3)]],
        device const uchar * key_values [[buffer(4)]],
        device const uchar * key_scales [[buffer(5)]],
        device const int * visible_counts [[buffer(6)]],
        device float * scores [[buffer(7)]],
        threadgroup half * staged_keys [[threadgroup(0)]],
        threadgroup float * block_dots [[threadgroup(1)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort thread_index [[thread_index_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_group = 8u;
    const uint row_base = group.x * rows_per_group;
    const uint query = group.y;
    if (query >= args.query_count) return;
    const int visible = visible_counts[query];
    float first_accumulator = 0.0f;
    float second_accumulator = 0.0f;

    for (uint block = 0u; block < 4u; ++block) {
        for (uint element = uint(thread_index); element < 8u * 32u; element += 256u) {
            const uint local_row = element / 32u;
            const uint dimension = element - local_row * 32u;
            const uint row = row_base + local_row;
            const bool row_visible = visible >= 0 && row < args.row_capacity
                && row < uint(visible);
            uchar code = 0u;
            if (row_visible) {
                const uint byte_index = row * 64u + block * 16u + dimension / 2u;
                const uchar packed = key_values[byte_index];
                code = (dimension & 1u) == 0u ? packed & 0x0fu : packed >> 4u;
            }
            staged_keys[dimension * 8u + local_row] = ds4_indexer_e2m1_unit(code);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint head_base = uint(simdgroup) * 8u;
        const uint query_base = (query * 64u + head_base) * 128u + block * 32u;
        simdgroup_float8x8 dots = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint dimension = 0u; dimension < 32u; dimension += 8u) {
            simdgroup_half8x8 query_tile;
            simdgroup_half8x8 key_tile;
            simdgroup_load(
                query_tile,
                query_units + query_base + dimension,
                128u,
                0u,
                false);
            simdgroup_load(key_tile, staged_keys + dimension * 8u, 8u, 0u, false);
            simdgroup_multiply_accumulate(dots, query_tile, key_tile, dots);
        }
        simdgroup_store(dots, block_dots + head_base * 8u, 8u, 0u, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
            for (uint cell = uint(thread_index); cell < 64u * 8u; cell += 256u) {
                const uint head = cell / 8u;
                const uint local_row = cell - head * 8u;
                const uint row = row_base + local_row;
                const bool row_visible = visible >= 0 && row < args.row_capacity
                    && row < uint(visible);
                float scaled_dot = 0.0f;
                if (row_visible) {
                    const int exponent = int(query_scales[(query * 64u + head) * 4u + block])
                        + int(key_scales[row * 4u + block]) - 254;
                    scaled_dot = ldexp(block_dots[cell], exponent);
                }
                if (cell < 256u) {
                    first_accumulator = first_accumulator + scaled_dot;
                } else {
                    second_accumulator = second_accumulator + scaled_dot;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    block_dots[uint(thread_index)] = first_accumulator;
    block_dots[uint(thread_index) + 256u] = second_accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_index < 8u) {
        const uint local_row = uint(thread_index);
        const uint row = row_base + local_row;
        if (row < args.row_capacity) {
            const uint score_index = query * args.row_capacity + row;
            if (visible < 0 || row >= uint(visible)) {
                scores[score_index] = -INFINITY;
            } else {
                float score = 0.0f;
                {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
                    for (uint head = 0u; head < 64u; ++head) {
                        const float contribution = max(block_dots[head * 8u + local_row], 0.0f)
                            * head_weights[query * 64u + head];
                        score = score + contribution;
                    }
                }
                scores[score_index] = score;
            }
        }
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

constant uint DS4_SELECTOR_MG_RECORD_WORDS = 20u;
constant uint DS4_SELECTOR_MG_RECORD_GENERATION = 0u;
constant uint DS4_SELECTOR_MG_RECORD_DIGIT = 1u;
constant uint DS4_SELECTOR_MG_RECORD_ERROR = 2u;
constant uint DS4_SELECTOR_MG_RECORD_COMPLETION = 3u;
constant uint DS4_SELECTOR_MG_RECORD_BINS = 4u;
constant uint DS4_SELECTOR_MG_STATE_WORDS = 10u;
constant uint DS4_SELECTOR_MG_STATE_GENERATION = 0u;
constant uint DS4_SELECTOR_MG_STATE_DIGIT = 1u;
constant uint DS4_SELECTOR_MG_STATE_PREFIX = 2u;
constant uint DS4_SELECTOR_MG_STATE_PREFIX_MASK = 3u;
constant uint DS4_SELECTOR_MG_STATE_RANK = 4u;
constant uint DS4_SELECTOR_MG_STATE_STATUS = 5u;
constant uint DS4_SELECTOR_MG_STATE_THRESHOLD_KEY = 6u;
constant uint DS4_SELECTOR_MG_STATE_THRESHOLD_TAKE = 7u;
constant uint DS4_SELECTOR_MG_STATE_SELECTED_COUNT = 8u;
constant uint DS4_SELECTOR_MG_STATE_COMPLETION = 9u;
constant uint DS4_SELECTOR_MG_PLAN_WORDS = 5u;
constant uint DS4_SELECTOR_MG_PLAN_GREATER = 0u;
constant uint DS4_SELECTOR_MG_PLAN_EQUAL = 1u;
constant uint DS4_SELECTOR_MG_PLAN_TIE_QUOTA = 2u;
constant uint DS4_SELECTOR_MG_PLAN_SELECTED = 3u;
constant uint DS4_SELECTOR_MG_PLAN_ID_OFFSET = 4u;
constant uint DS4_SELECTOR_MG_COMPACT_PHASE = 8u;

static inline uint ds4_selector_mg_record_completion(
        uint generation,
        uint digit,
        uint group) {
    return 0xd5410000u ^ generation ^ (digit << 12u) ^ group;
}

static inline uint ds4_selector_mg_state_completion(uint generation, uint digit) {
    return 0xd5420000u ^ generation ^ (digit << 12u);
}

static inline uint ds4_selector_mg_compact_completion(uint generation, uint group) {
    return 0xd5430000u ^ generation ^ group;
}

kernel void kernel_deepseek_v4_select_top_k_multigroup_histogram_f32(
        constant ds4_indexer_multigroup_select_args & args [[buffer(0)]],
        device const float * scores [[buffer(1)]],
        device const int * visible_counts [[buffer(2)]],
        device const uint * state [[buffer(3)]],
        device uint * records [[buffer(4)]],
        threadgroup uint * scratch [[threadgroup(0)]],
        uint lane [[thread_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]],
        uint width [[threads_per_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    if (group >= args.group_count) return;
    const uint simdgroup_count = (width + 31u) / 32u;
    const uint record_base = group * DS4_SELECTOR_MG_RECORD_WORDS;
    const int visible_i = visible_counts[0];
    const bool geometry_valid = visible_i > 0 && uint(visible_i) <= args.row_capacity
        && args.top_k > 0u && args.top_k <= args.row_capacity;
    const uint visible = geometry_valid ? uint(visible_i) : 0u;

    uint prefix = 0u;
    uint prefix_mask = 0u;
    uint prior_status = 0u;
    if (args.digit > 0u) {
        const bool state_valid = state[DS4_SELECTOR_MG_STATE_GENERATION] == args.generation
            && state[DS4_SELECTOR_MG_STATE_DIGIT] + 1u == args.digit
            && state[DS4_SELECTOR_MG_STATE_COMPLETION]
                == ds4_selector_mg_state_completion(args.generation, args.digit - 1u);
        prior_status = state_valid ? state[DS4_SELECTOR_MG_STATE_STATUS] : 3u;
        if (state_valid) {
            prefix = state[DS4_SELECTOR_MG_STATE_PREFIX];
            prefix_mask = state[DS4_SELECTOR_MG_STATE_PREFIX_MASK];
        }
    }

    uint local_error = geometry_valid ? prior_status : 1u;
    uint local_bins[16];
    for (uint bin = 0u; bin < 16u; ++bin) local_bins[bin] = 0u;
    if (local_error == 0u) {
        const uint chunk = args.row_capacity / args.group_count
            + uint(args.row_capacity % args.group_count != 0u);
        const uint start = uint(min(ulong(group) * ulong(chunk), ulong(args.row_capacity)));
        const uint end = uint(min(ulong(start) + ulong(chunk), ulong(args.row_capacity)));
        for (uint row = start + lane; row < end && row < visible; row += width) {
            const float score = scores[row];
            if (!isfinite(score)) {
                local_error = 2u;
                continue;
            }
            const uint key = ds4_selector_order_key(score);
            if ((key & prefix_mask) == prefix) {
                ++local_bins[(key >> args.shift) & 0xfu];
            }
        }
    }

    for (uint bin = 0u; bin < 16u; ++bin) {
        const uint simd_count = simd_sum(local_bins[bin]);
        if (simd_lane == 0u) {
            scratch[uint(simdgroup) * 16u + bin] = simd_count;
        }
    }
    const uint simd_error = simd_max(local_error);
    if (simd_lane == 0u) scratch[128u + uint(simdgroup)] = simd_error;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u && simd_lane < 16u) {
        uint total = 0u;
        for (uint index = 0u; index < simdgroup_count; ++index) {
            total += scratch[index * 16u + uint(simd_lane)];
        }
        records[record_base + DS4_SELECTOR_MG_RECORD_BINS + uint(simd_lane)] = total;
    }
    if (lane == 0u) {
        uint error = 0u;
        for (uint index = 0u; index < simdgroup_count; ++index) {
            error = max(error, scratch[128u + index]);
        }
        records[record_base + DS4_SELECTOR_MG_RECORD_GENERATION] = args.generation;
        records[record_base + DS4_SELECTOR_MG_RECORD_DIGIT] = args.digit;
        records[record_base + DS4_SELECTOR_MG_RECORD_ERROR] = error;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (lane == 0u) {
        records[record_base + DS4_SELECTOR_MG_RECORD_COMPLETION]
            = ds4_selector_mg_record_completion(args.generation, args.digit, group);
    }
}

kernel void kernel_deepseek_v4_select_top_k_multigroup_reduce_f32(
        constant ds4_indexer_multigroup_select_args & args [[buffer(0)]],
        device const int * visible_counts [[buffer(1)]],
        device const uint * records [[buffer(2)]],
        device uint * partition_plan [[buffer(3)]],
        device uint * state [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    const int visible_i = visible_counts[0];
    const bool geometry_valid = visible_i > 0 && uint(visible_i) <= args.row_capacity
        && args.top_k > 0u && args.top_k <= args.row_capacity;
    const uint visible = geometry_valid ? uint(visible_i) : 0u;
    const uint selected_count = min(visible, args.top_k);

    bool stale = false;
    bool nonfinite = false;
    for (uint group = 0u; group < args.group_count; ++group) {
        const uint base = group * DS4_SELECTOR_MG_RECORD_WORDS;
        stale = stale
            || records[base + DS4_SELECTOR_MG_RECORD_GENERATION] != args.generation
            || records[base + DS4_SELECTOR_MG_RECORD_DIGIT] != args.digit
            || records[base + DS4_SELECTOR_MG_RECORD_COMPLETION]
                != ds4_selector_mg_record_completion(args.generation, args.digit, group);
        const uint record_error = records[base + DS4_SELECTOR_MG_RECORD_ERROR];
        nonfinite = nonfinite || record_error == 2u;
        stale = stale || record_error > 2u || (geometry_valid && record_error == 1u);
    }

    uint status = geometry_valid ? 0u : 1u;
    uint prefix = 0u;
    uint prefix_mask = 0u;
    uint rank = selected_count;
    if (args.digit > 0u) {
        const bool state_valid = state[DS4_SELECTOR_MG_STATE_GENERATION] == args.generation
            && state[DS4_SELECTOR_MG_STATE_DIGIT] + 1u == args.digit
            && state[DS4_SELECTOR_MG_STATE_COMPLETION]
                == ds4_selector_mg_state_completion(args.generation, args.digit - 1u);
        if (!state_valid) stale = true;
        if (state_valid) {
            status = state[DS4_SELECTOR_MG_STATE_STATUS];
            prefix = state[DS4_SELECTOR_MG_STATE_PREFIX];
            prefix_mask = state[DS4_SELECTOR_MG_STATE_PREFIX_MASK];
            rank = state[DS4_SELECTOR_MG_STATE_RANK];
        }
    }
    if (geometry_valid && stale) status = 3u;
    else if (geometry_valid && status == 0u && nonfinite) status = 2u;

    uint chosen = 0u;
    uint remaining = rank;
    if (status == 0u) {
        bool found = false;
        for (int bin = 15; bin >= 0; --bin) {
            uint count = 0u;
            for (uint group = 0u; group < args.group_count; ++group) {
                const uint base = group * DS4_SELECTOR_MG_RECORD_WORDS;
                count += records[base + DS4_SELECTOR_MG_RECORD_BINS + uint(bin)];
            }
            if (remaining <= count) {
                chosen = uint(bin);
                found = true;
                break;
            }
            remaining -= count;
        }
        if (!found) status = 3u;
    }

    if (status == 0u) {
        for (uint group = 0u; group < args.group_count; ++group) {
            const uint base = group * DS4_SELECTOR_MG_RECORD_WORDS;
            const uint plan_base = group * DS4_SELECTOR_MG_PLAN_WORDS;
            uint greater = args.digit == 0u
                ? 0u
                : partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_GREATER];
            for (uint bin = chosen + 1u; bin < 16u; ++bin) {
                greater += records[base + DS4_SELECTOR_MG_RECORD_BINS + bin];
            }
            partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_GREATER] = greater;
            partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_EQUAL] = args.digit == 7u
                ? records[base + DS4_SELECTOR_MG_RECORD_BINS + chosen]
                : 0u;
            if (args.digit == 0u) {
                partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_TIE_QUOTA] = 0u;
                partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_SELECTED] = 0u;
                partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_ID_OFFSET] = 0u;
            }
        }
        prefix |= chosen << args.shift;
        prefix_mask |= 0xfu << args.shift;
        rank = remaining;
    } else {
        for (uint group = 0u; group < args.group_count; ++group) {
            const uint plan_base = group * DS4_SELECTOR_MG_PLAN_WORDS;
            for (uint word = 0u; word < DS4_SELECTOR_MG_PLAN_WORDS; ++word) {
                partition_plan[plan_base + word] = 0u;
            }
        }
    }

    uint threshold_key = 0u;
    uint threshold_take = 0u;
    if (status == 0u && args.digit == 7u) {
        uint greater_total = 0u;
        uint equal_total = 0u;
        for (uint group = 0u; group < args.group_count; ++group) {
            const uint plan_base = group * DS4_SELECTOR_MG_PLAN_WORDS;
            greater_total += partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_GREATER];
            equal_total += partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_EQUAL];
        }
        if (rank == 0u || rank > equal_total || greater_total + rank != selected_count) {
            status = 3u;
        } else {
            uint remaining_ties = rank;
            uint output_offset = 0u;
            for (uint group = 0u; group < args.group_count; ++group) {
                const uint plan_base = group * DS4_SELECTOR_MG_PLAN_WORDS;
                const uint greater = partition_plan[
                    plan_base + DS4_SELECTOR_MG_PLAN_GREATER];
                const uint equal = partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_EQUAL];
                const uint quota = min(equal, remaining_ties);
                const uint selected = greater + quota;
                partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_TIE_QUOTA] = quota;
                partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_SELECTED] = selected;
                partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_ID_OFFSET] = output_offset;
                remaining_ties -= quota;
                output_offset += selected;
            }
            if (remaining_ties != 0u || output_offset != selected_count) {
                status = 3u;
            } else {
                threshold_key = prefix;
                threshold_take = rank;
            }
        }
    }
    if (args.digit == 7u && status != 0u) {
        for (uint group = 0u; group < args.group_count; ++group) {
            const uint plan_base = group * DS4_SELECTOR_MG_PLAN_WORDS;
            for (uint word = 0u; word < DS4_SELECTOR_MG_PLAN_WORDS; ++word) {
                partition_plan[plan_base + word] = 0u;
            }
        }
    }

    state[DS4_SELECTOR_MG_STATE_COMPLETION] = 0u;
    state[DS4_SELECTOR_MG_STATE_GENERATION] = args.generation;
    state[DS4_SELECTOR_MG_STATE_DIGIT] = args.digit;
    state[DS4_SELECTOR_MG_STATE_PREFIX] = prefix;
    state[DS4_SELECTOR_MG_STATE_PREFIX_MASK] = prefix_mask;
    state[DS4_SELECTOR_MG_STATE_RANK] = rank;
    state[DS4_SELECTOR_MG_STATE_STATUS] = status;
    state[DS4_SELECTOR_MG_STATE_THRESHOLD_KEY] = threshold_key;
    state[DS4_SELECTOR_MG_STATE_THRESHOLD_TAKE] = threshold_take;
    state[DS4_SELECTOR_MG_STATE_SELECTED_COUNT] = selected_count;
    threadgroup_barrier(mem_flags::mem_device);
    state[DS4_SELECTOR_MG_STATE_COMPLETION]
        = ds4_selector_mg_state_completion(args.generation, args.digit);
}

kernel void kernel_deepseek_v4_select_top_k_multigroup_compact_f32(
        constant ds4_indexer_multigroup_select_args & args [[buffer(0)]],
        device const float * scores [[buffer(1)]],
        device const int * visible_counts [[buffer(2)]],
        device const uint * state [[buffer(3)]],
        device const uint * partition_plan [[buffer(4)]],
        device uchar * private_mask [[buffer(5)]],
        device int * private_ids [[buffer(6)]],
        device uint * records [[buffer(7)]],
        threadgroup uint * scratch [[threadgroup(0)]],
        uint lane [[thread_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]],
        uint width [[threads_per_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    if (group >= args.group_count) return;
    const uint record_base = group * DS4_SELECTOR_MG_RECORD_WORDS;
    const uint plan_base = group * DS4_SELECTOR_MG_PLAN_WORDS;
    const uint simdgroup_count = (width + 31u) / 32u;
    const uint shared_base = 3u * width + 16u;
    if (lane == 0u) {
        records[record_base + DS4_SELECTOR_MG_RECORD_COMPLETION] = 0u;
    }

    const int visible_i = visible_counts[0];
    const bool geometry_valid = visible_i > 0 && uint(visible_i) <= args.row_capacity
        && args.top_k > 0u && args.top_k <= args.row_capacity;
    const uint visible = geometry_valid ? uint(visible_i) : 0u;
    uint local_error = geometry_valid ? 0u : 1u;
    if (geometry_valid) {
        const bool state_valid = state[DS4_SELECTOR_MG_STATE_GENERATION] == args.generation
            && state[DS4_SELECTOR_MG_STATE_DIGIT] == 7u
            && state[DS4_SELECTOR_MG_STATE_COMPLETION]
                == ds4_selector_mg_state_completion(args.generation, 7u);
        if (!state_valid) {
            local_error = 3u;
        } else {
            const uint state_status = state[DS4_SELECTOR_MG_STATE_STATUS];
            local_error = state_status == 0u || state_status == 2u || state_status == 3u
                ? state_status
                : 3u;
            if (local_error == 0u) {
                const bool final_state_valid = state[DS4_SELECTOR_MG_STATE_PREFIX_MASK]
                        == 0xffffffffu
                    && state[DS4_SELECTOR_MG_STATE_PREFIX]
                        == state[DS4_SELECTOR_MG_STATE_THRESHOLD_KEY]
                    && state[DS4_SELECTOR_MG_STATE_RANK]
                        == state[DS4_SELECTOR_MG_STATE_THRESHOLD_TAKE]
                    && state[DS4_SELECTOR_MG_STATE_THRESHOLD_TAKE] > 0u
                    && state[DS4_SELECTOR_MG_STATE_SELECTED_COUNT] == min(visible, args.top_k);
                if (!final_state_valid) local_error = 3u;
            }
        }
    }

    const uint capacity_chunk = args.row_capacity / args.group_count
        + uint(args.row_capacity % args.group_count != 0u);
    const uint partition_start = uint(min(
        ulong(group) * ulong(capacity_chunk), ulong(args.row_capacity)));
    const uint partition_end = uint(min(
        ulong(partition_start) + ulong(capacity_chunk), ulong(args.row_capacity)));
    const uint partition_rows = partition_end - partition_start;
    const uint lane_chunk = (partition_rows + width - 1u) / width;
    const uint lane_start = min(partition_start + lane * lane_chunk, partition_end);
    const uint lane_end = min(lane_start + lane_chunk, partition_end);
    const uint threshold_key = local_error == 0u
        ? state[DS4_SELECTOR_MG_STATE_THRESHOLD_KEY]
        : 0u;
    uint local_greater = 0u;
    uint local_equal = 0u;
    if (local_error == 0u) {
        for (uint row = lane_start; row < lane_end && row < visible; ++row) {
            const float score = scores[row];
            if (!isfinite(score)) {
                local_error = 2u;
                continue;
            }
            const uint key = ds4_selector_order_key(score);
            local_greater += uint(key > threshold_key);
            local_equal += uint(key == threshold_key);
        }
    }
    scratch[lane] = local_greater;
    scratch[width + lane] = local_equal;
    const uint simd_error = simd_max(local_error);
    if (simd_lane == 0u) scratch[3u * width + uint(simdgroup)] = simd_error;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        uint error = 0u;
        for (uint index = 0u; index < simdgroup_count; ++index) {
            error = max(error, scratch[3u * width + index]);
        }
        uint greater_total = 0u;
        uint equal_total = 0u;
        uint selected_total = 0u;
        uint remaining_lane_ties = error == 0u
            ? partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_TIE_QUOTA]
            : 0u;
        for (uint index = 0u; index < width; ++index) {
            const uint greater = scratch[index];
            const uint equal = scratch[width + index];
            const uint lane_ties = min(equal, remaining_lane_ties);
            scratch[index] = lane_ties;
            scratch[2u * width + index] = selected_total;
            greater_total += greater;
            equal_total += equal;
            selected_total += greater + lane_ties;
            remaining_lane_ties -= lane_ties;
        }
        if (error == 0u) {
            const uint plan_greater = partition_plan[
                plan_base + DS4_SELECTOR_MG_PLAN_GREATER];
            const uint plan_equal = partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_EQUAL];
            const uint plan_quota = partition_plan[
                plan_base + DS4_SELECTOR_MG_PLAN_TIE_QUOTA];
            const uint plan_selected = partition_plan[
                plan_base + DS4_SELECTOR_MG_PLAN_SELECTED];
            const uint plan_offset = partition_plan[
                plan_base + DS4_SELECTOR_MG_PLAN_ID_OFFSET];
            const bool plan_valid = plan_quota <= plan_equal
                && plan_selected == plan_greater + plan_quota
                && plan_offset <= args.top_k
                && plan_selected <= args.top_k - plan_offset
                && remaining_lane_ties == 0u
                && greater_total == plan_greater
                && equal_total == plan_equal
                && selected_total == plan_selected;
            if (!plan_valid) error = 3u;
        }
        scratch[shared_base] = error;
        scratch[shared_base + 1u] = greater_total;
        scratch[shared_base + 2u] = equal_total;
        scratch[shared_base + 3u] = selected_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint group_error = scratch[shared_base];
    uint written = 0u;
    if (group_error == 0u) {
        const uint lane_ties = scratch[lane];
        uint output_slot = partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_ID_OFFSET]
            + scratch[2u * width + lane];
        uint equal_seen = 0u;
        for (uint row = lane_start; row < lane_end; ++row) {
            bool selected = false;
            if (row < visible) {
                const uint key = ds4_selector_order_key(scores[row]);
                const bool at_threshold = key == threshold_key;
                selected = key > threshold_key || (at_threshold && equal_seen < lane_ties);
                if (at_threshold) ++equal_seen;
            }
            private_mask[row] = uchar(selected);
            if (selected) {
                private_ids[output_slot] = int(row);
                ++output_slot;
                ++written;
            }
        }
    }
    scratch[2u * width + lane] = written;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (lane == 0u && group_error == 0u) {
        uint written_total = 0u;
        for (uint index = 0u; index < width; ++index) {
            written_total += scratch[2u * width + index];
        }
        if (written_total != scratch[shared_base + 3u]) {
            group_error = 3u;
            scratch[shared_base] = group_error;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        for (uint word = 0u; word < DS4_SELECTOR_MG_RECORD_WORDS; ++word) {
            records[record_base + word] = 0u;
        }
        records[record_base + DS4_SELECTOR_MG_RECORD_GENERATION] = args.generation;
        records[record_base + DS4_SELECTOR_MG_RECORD_DIGIT] = DS4_SELECTOR_MG_COMPACT_PHASE;
        records[record_base + DS4_SELECTOR_MG_RECORD_ERROR] = scratch[shared_base];
        records[record_base + 4u] = scratch[shared_base + 1u];
        records[record_base + 5u] = scratch[shared_base + 2u];
        records[record_base + 6u] = scratch[shared_base + 3u];
        records[record_base + 7u] = scratch[shared_base] == 0u
            ? partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_ID_OFFSET]
            : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (lane == 0u) {
        records[record_base + DS4_SELECTOR_MG_RECORD_COMPLETION]
            = ds4_selector_mg_compact_completion(args.generation, group);
    }
}

kernel void kernel_deepseek_v4_select_top_k_multigroup_publish_f32(
        constant ds4_indexer_multigroup_select_args & args [[buffer(0)]],
        device const int * visible_counts [[buffer(1)]],
        device const uint * state [[buffer(2)]],
        device const uint * partition_plan [[buffer(3)]],
        device const uint * records [[buffer(4)]],
        device const uchar * private_mask [[buffer(5)]],
        device const int * private_ids [[buffer(6)]],
        device int * selected_mask [[buffer(7)]],
        device int * cache_order_ids [[buffer(8)]],
        device int * selected_counts [[buffer(9)]],
        device int * status_output [[buffer(10)]],
        threadgroup uint * scratch [[threadgroup(0)]],
        uint lane [[thread_index_in_threadgroup]],
        uint width [[threads_per_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    const uint simdgroup_count = (width + 31u) / 32u;
    const int visible_i = visible_counts[0];
    const bool geometry_valid = visible_i > 0 && uint(visible_i) <= args.row_capacity
        && args.top_k > 0u && args.top_k <= args.row_capacity;
    const uint visible = geometry_valid ? uint(visible_i) : 0u;
    const uint selected_count = min(visible, args.top_k);
    if (lane == 0u) {
        selected_counts[0] = -1;
        status_output[0] = -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (lane == 0u) {
        uint status = geometry_valid ? 0u : 1u;
        bool state_valid = false;
        if (geometry_valid) {
            state_valid = state[DS4_SELECTOR_MG_STATE_GENERATION] == args.generation
                && state[DS4_SELECTOR_MG_STATE_DIGIT] == 7u
                && state[DS4_SELECTOR_MG_STATE_COMPLETION]
                    == ds4_selector_mg_state_completion(args.generation, 7u);
            status = state_valid ? state[DS4_SELECTOR_MG_STATE_STATUS] : 3u;
            if (status > 3u || status == 1u) status = 3u;
        }
        if (geometry_valid && state_valid && (status == 0u || status == 2u)) {
            bool stale = false;
            ulong output_offset = 0ul;
            ulong quota_total = 0ul;
            for (uint group = 0u; group < args.group_count; ++group) {
                const uint record_base = group * DS4_SELECTOR_MG_RECORD_WORDS;
                const uint plan_base = group * DS4_SELECTOR_MG_PLAN_WORDS;
                stale = stale
                    || records[record_base + DS4_SELECTOR_MG_RECORD_GENERATION]
                        != args.generation
                    || records[record_base + DS4_SELECTOR_MG_RECORD_DIGIT]
                        != DS4_SELECTOR_MG_COMPACT_PHASE
                    || records[record_base + DS4_SELECTOR_MG_RECORD_ERROR] != status
                    || records[record_base + DS4_SELECTOR_MG_RECORD_COMPLETION]
                        != ds4_selector_mg_compact_completion(args.generation, group);
                if (status == 0u) {
                    const uint greater = partition_plan[
                        plan_base + DS4_SELECTOR_MG_PLAN_GREATER];
                    const uint equal = partition_plan[plan_base + DS4_SELECTOR_MG_PLAN_EQUAL];
                    const uint quota = partition_plan[
                        plan_base + DS4_SELECTOR_MG_PLAN_TIE_QUOTA];
                    const uint selected = partition_plan[
                        plan_base + DS4_SELECTOR_MG_PLAN_SELECTED];
                    const uint offset = partition_plan[
                        plan_base + DS4_SELECTOR_MG_PLAN_ID_OFFSET];
                    const bool range_valid = offset <= args.top_k
                        && selected <= args.top_k - min(offset, args.top_k);
                    stale = stale || quota > equal || selected != greater + quota
                        || ulong(offset) != output_offset
                        || !range_valid
                        || records[record_base + 4u] != greater
                        || records[record_base + 5u] != equal
                        || records[record_base + 6u] != selected
                        || records[record_base + 7u] != offset;
                    for (uint word = 8u; word < DS4_SELECTOR_MG_RECORD_WORDS; ++word) {
                        stale = stale || records[record_base + word] != 0u;
                    }
                    if (range_valid) output_offset += ulong(selected);
                    quota_total += ulong(quota);
                }
            }
            if (status == 0u) {
                stale = stale
                    || state[DS4_SELECTOR_MG_STATE_PREFIX_MASK] != 0xffffffffu
                    || state[DS4_SELECTOR_MG_STATE_PREFIX]
                        != state[DS4_SELECTOR_MG_STATE_THRESHOLD_KEY]
                    || state[DS4_SELECTOR_MG_STATE_RANK]
                        != state[DS4_SELECTOR_MG_STATE_THRESHOLD_TAKE]
                    || state[DS4_SELECTOR_MG_STATE_THRESHOLD_TAKE] == 0u
                    || state[DS4_SELECTOR_MG_STATE_SELECTED_COUNT] != selected_count
                    || quota_total != ulong(state[DS4_SELECTOR_MG_STATE_THRESHOLD_TAKE])
                    || output_offset != ulong(selected_count);
            }
            if (stale) status = 3u;
        }
        scratch[0] = status;
        scratch[1] = selected_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local_population = 0u;
    uint local_error = 0u;
    if (scratch[0] == 0u) {
        for (uint row = lane; row < args.row_capacity; row += width) {
            const uint value = uint(private_mask[row]);
            local_error = max(local_error, uint(value > 1u) * 3u);
            local_population += uint(value == 1u);
        }
        for (uint slot = lane; slot < selected_count; slot += width) {
            const int id = private_ids[slot];
            const bool in_range = id >= 0 && uint(id) < visible;
            const bool ordered = slot == 0u || id > private_ids[slot - 1u];
            const bool masked = in_range && private_mask[uint(id)] == uchar(1);
            if (!in_range || !ordered || !masked) local_error = 3u;
        }
    }
    const uint simd_population = simd_sum(local_population);
    const uint simd_error = simd_max(local_error);
    if (simd_lane == 0u) {
        scratch[4u + uint(simdgroup)] = simd_population;
        scratch[12u + uint(simdgroup)] = simd_error;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        const uint population = simd_lane < simdgroup_count
            ? scratch[4u + uint(simd_lane)]
            : 0u;
        const uint error = simd_lane < simdgroup_count
            ? scratch[12u + uint(simd_lane)]
            : 0u;
        const uint total_population = simd_sum(population);
        const uint reduced_error = simd_max(error);
        if (simd_lane == 0u && scratch[0] == 0u
                && (reduced_error != 0u || total_population != selected_count)) {
            scratch[0] = 3u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint final_status = scratch[0];
    for (uint row = lane; row < args.row_capacity; row += width) {
        selected_mask[row] = final_status == 0u
            ? int(private_mask[row])
            : int(row < selected_count);
    }
    for (uint slot = lane; slot < args.top_k; slot += width) {
        cache_order_ids[slot] = final_status == 0u && slot < selected_count
            ? private_ids[slot]
            : (final_status != 0u && slot < selected_count ? int(slot) : -1);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane == 0u) {
        selected_counts[0] = int(selected_count);
        status_output[0] = int(final_status);
    }
}

template <bool Radix4, bool EmitMask>
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

    if (EmitMask) {
        for (uint row = lane; row < args.row_capacity; row += width) {
            selected_mask[mask_base + row] = 0;
        }
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
        if (EmitMask) {
            for (uint row = lane; row < args.row_capacity; row += width) {
                selected_mask[mask_base + row] = row < selected_count ? 1 : 0;
            }
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
        if (EmitMask) {
            for (uint row = lane; row < args.row_capacity; row += width) {
                selected_mask[mask_base + row] = row < selected_count ? 1 : 0;
            }
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
        if (EmitMask) {
            for (uint row = lane; row < args.row_capacity; row += width) {
                selected_mask[mask_base + row] = row < selected_count ? 1 : 0;
            }
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
        if (EmitMask) selected_mask[mask_base + row] = 1;
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
    ds4_select_top_k_parallel_impl<false, true>(
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
    ds4_select_top_k_parallel_impl<true, true>(
        args, scores, visible_counts, selected_mask, ranked_ids,
        cache_order_ids, selected_counts, status, lane_scratch, shared,
        lane, query, width, simdgroup, simd_lane);
}

kernel void kernel_deepseek_v4_select_top_k_radix4_ids_f32(
        constant ds4_indexer_select_args & args [[buffer(0)]],
        device const float * scores [[buffer(1)]],
        device const int * visible_counts [[buffer(2)]],
        device int * cache_order_ids [[buffer(3)]],
        device int * selected_counts [[buffer(4)]],
        device int * status [[buffer(5)]],
        threadgroup uint * lane_scratch [[threadgroup(0)]],
        threadgroup uint * shared [[threadgroup(1)]],
        uint lane [[thread_index_in_threadgroup]],
        uint query [[threadgroup_position_in_grid]],
        uint width [[threads_per_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_lane [[thread_index_in_simdgroup]]) {
    ds4_select_top_k_parallel_impl<true, false>(
        args, scores, visible_counts, cache_order_ids, cache_order_ids,
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
