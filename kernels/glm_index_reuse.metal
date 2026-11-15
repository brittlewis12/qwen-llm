#include <metal_stdlib>
using namespace metal;

// Query tiling/key reuse follows the mechanism investigated in oMLX's
// glm_moe_dsa scorer (Apache-2.0). This implementation is written around our
// incumbent classic SIMD-matrix K8 sequence, not its tensor/MMA dot reduction.
// Each dot performs the same 16 ordered F16 x F16 -> F32 K8 operations as
// lightning_indexer_scores_f16_matrix_impl<32>; heads are summed in order.
struct glm_index_reuse_args {
    uint row_capacity;
    uint query_count;
    uint max_rows;
    uint pad;
};

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_glm_index_scores_reuse(
        constant glm_index_reuse_args & args [[buffer(0)]],
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
    const uint row_base = group.x * 8u;
    const uint query_base = group.y * 8u;
    int visible_max = -1;
    for (uint q = 0u; q < 8u; ++q) {
        visible_max = max(visible_max, visible_counts[query_base + q]);
    }
    for (uint element = uint(thread_index); element < 8u * 128u; element += 128u) {
        const uint local_row = element / 128u;
        const uint dimension = element - local_row * 128u;
        const uint row = row_base + local_row;
        const bool visible = visible_max >= 0 && row < args.row_capacity && row < uint(visible_max);
        staged_keys[dimension * 8u + local_row] = visible ? keys[row * 128u + dimension] : (half)0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Rows are queries, columns are keys. Each SIMD group handles eight heads.
    // A padded query stride avoids mapping all eight query rows to one bank.
    for (uint head = uint(simdgroup) * 8u; head < uint(simdgroup) * 8u + 8u; ++head) {
        simdgroup_float8x8 dots = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint dimension = 0u; dimension < 128u; dimension += 8u) {
            simdgroup_half8x8 query_tile;
            simdgroup_half8x8 key_tile;
            simdgroup_load(query_tile, queries + query_base * 4096u + head * 128u + dimension, 4096u, 0u, false);
            simdgroup_load(key_tile, staged_keys + dimension * 8u, 8u, 0u, false);
            simdgroup_multiply_accumulate(dots, query_tile, key_tile, dots);
        }
        simdgroup_store(dots, head_dots + head * 8u, 264u, 0u, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < 64u) {
        const uint local_query = uint(thread_index) / 8u;
        const uint local_row = uint(thread_index) % 8u;
        const uint query = query_base + local_query;
        const uint row = row_base + local_row;
        if (row < args.row_capacity) {
            const int visible = visible_counts[query];
            const uint score_index = query * args.row_capacity + row;
            if (visible < 0 || row >= uint(visible)) {
                scores[score_index] = -INFINITY;
            } else {
                const uint weight_base = query * 32u;
                float score = 0.0f;
                for (uint head = 0u; head < 32u; ++head) {
                    score += max(head_dots[local_query * 264u + head * 8u + local_row], 0.0f)
                        * head_weights[weight_base + head];
                }
                scores[score_index] = score;
            }
        }
    }
}
