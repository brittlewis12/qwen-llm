// Flash-attention-style single-token decode.
//
// Replaces the naive `kernel_attn_decode_f32` for long contexts. The
// naive version holds all `n_pos` scores in threadgroup memory before
// softmaxing, which caps context at ~7000 (28 KB tg memory budget /
// 4 B/score) AND scales poorly because:
//   * scoring loop is serial over the full n_pos,
//   * softmax requires global max + sum reductions across n_pos,
//   * V-aggregate is again serial over n_pos.
//
// This kernel uses online (Welford-style) softmax over K/V tiles. Per
// Q head, we maintain (m, l, o):
//   m = running max of scores so far
//   l = running sum of exp(score - m)
//   o = running V-weighted sum, in scaled form
// For each tile of K/V positions:
//   - compute the tile's scores and tile_max
//   - rescale (m, l, o) to absorb the new max
//   - update l += sum(exp(tile_score - new_max))
//   - update o += sum(exp(tile_score - new_max) * v[pos])
// At the end: out[d] = o[d] / l.
//
// Properties:
//   * Each KV row is read exactly once (bandwidth-optimal).
//   * No threadgroup-memory scaling with n_pos (constant per dispatch).
//   * Numerically stable (max subtracted before exp).
//
// One threadgroup per Q head (same as naive). 32 threads cooperate per
// tile via simdgroup operations.
//
// Reference: FlashAttention paper (Dao et al. 2022), but specialized for
// the single-token-Q case. llama.cpp's `kernel_flash_attn_ext_*` does
// the same thing in a heavily-templated way; we strip down to the
// single-Q-row case + 32-thread simdgroup tiling.

#include <metal_stdlib>
using namespace metal;

constant constexpr ushort SIMD_LANES = 32;

// Tile size: how many KV positions we process per inner loop iteration.
// Bigger tile = better instruction-level parallelism but more registers.
// Each lane processes 1 position per tile (32 positions per simdgroup
// iteration). Tile of 128 = 4 iterations.
constant constexpr ushort TILE = 128;

struct attn_decode_flash_args {
    uint  n_q_heads;
    uint  n_kv_heads;
    uint  head_dim;
    uint  n_pos;
    uint  kv_stride; // n_kv_heads * head_dim
    float scale;
};

kernel void kernel_attn_decode_flash_f32(
        constant attn_decode_flash_args & args [[buffer(0)]],
        device const float * q       [[buffer(1)]], // [n_q_heads, head_dim]
        device const float * k_cache [[buffer(2)]], // [capacity, n_kv_heads, head_dim]
        device const float * v_cache [[buffer(3)]], // [capacity, n_kv_heads, head_dim]
        device       float * out     [[buffer(4)]], // [n_q_heads, head_dim]
        threadgroup  float * tg_scratch [[threadgroup(0)]], // [head_dim] for o accumulator
        uint  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    if (sgitg != 0) return; // single-simdgroup design; ignore others
    const uint qh = tgpig;
    if (qh >= args.n_q_heads) return;

    const uint group = args.n_q_heads / args.n_kv_heads;
    const uint kvh = qh / group;

    device const float * q_h = q + (ulong)qh * args.head_dim;
    device       float * out_h = out + (ulong)qh * args.head_dim;

    // Online softmax state, register-resident:
    //   m: running max
    //   l: running sum of exp(score - m)
    //   o[d]: V-weighted accumulator (in shmem since head_dim > register count)
    float m = -INFINITY;
    float l = 0.0f;

    // Initialize o-accumulator in shared memory.
    for (uint d = tiisg; d < args.head_dim; d += SIMD_LANES) {
        tg_scratch[d] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate through K/V positions in tiles of TILE size.
    for (uint p_tile = 0; p_tile < args.n_pos; p_tile += TILE) {
        const uint p_end = min(p_tile + TILE, args.n_pos);
        // Each lane handles positions p_tile + tiisg, p_tile + tiisg + 32, ...
        // up to TILE/32 = 4 positions per lane.

        // (1) Compute scores for this tile, in registers.
        // Lane `tiisg` owns position p_tile + tiisg + 0, +32, +64, +96.
        const ushort POSES_PER_LANE = TILE / SIMD_LANES;
        float scores[POSES_PER_LANE];
        bool valid[POSES_PER_LANE];
        for (ushort i = 0; i < POSES_PER_LANE; ++i) {
            const uint p = p_tile + i * SIMD_LANES + tiisg;
            valid[i] = p < p_end;
            if (valid[i]) {
                device const float * k_p = k_cache
                    + (ulong)p * args.kv_stride
                    + (ulong)kvh * args.head_dim;
                float s = 0.0f;
                for (uint d = 0; d < args.head_dim; ++d) {
                    s += q_h[d] * k_p[d];
                }
                scores[i] = s * args.scale;
            } else {
                scores[i] = -INFINITY;
            }
        }

        // (2) tile_max = max over all lanes' scores.
        float tile_max = -INFINITY;
        for (ushort i = 0; i < POSES_PER_LANE; ++i) {
            tile_max = max(tile_max, scores[i]);
        }
        tile_max = simd_max(tile_max);

        // (3) Rescale running state to the new global max.
        const float m_new = max(m, tile_max);
        const float rescale = (m == -INFINITY) ? 0.0f : exp(m - m_new);

        // l_new = l * rescale + sum(exp(scores - m_new))
        float tile_sum = 0.0f;
        for (ushort i = 0; i < POSES_PER_LANE; ++i) {
            if (valid[i]) {
                scores[i] = exp(scores[i] - m_new);
                tile_sum += scores[i];
            } else {
                scores[i] = 0.0f;
            }
        }
        tile_sum = simd_sum(tile_sum);

        const float l_new = l * rescale + tile_sum;

        // (4) Rescale o accumulator and add tile contribution.
        // o_new[d] = o[d] * rescale + sum_p (scores[p] * v_cache[p, kvh, d])
        // Each lane accumulates v contributions from its 4 positions.
        // We do this dim-by-dim, distributed across lanes via simd_sum.
        //
        // For correctness: rescale step needs barrier before V-aggregation
        // because tg_scratch is shared between all lanes.
        if (rescale != 1.0f) {
            for (uint d = tiisg; d < args.head_dim; d += SIMD_LANES) {
                tg_scratch[d] *= rescale;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // V-aggregation: each (lane, position-in-tile) contributes its
        // scores[i] * v_cache[p, kvh, :] to the o-accumulator across head_dim.
        // We stride over head_dim by SIMD_LANES, with each lane handling
        // one (d) at a time, summing across all 4 positions and across
        // all lanes via simd_sum.
        for (uint d = tiisg; d < args.head_dim; d += SIMD_LANES) {
            // Each lane computes its own contribution: sum of scores[i] *
            // v_cache[p_tile + i*SIMD + tiisg, kvh, d] for i in 0..POSES_PER_LANE.
            // But that's the WRONG index — we want the lane that owns dim
            // `d` to sum over all positions in the tile. So we need to
            // gather scores from the lane that owns each position.
            //
            // Simplest pattern: each lane writes its scores[i] into shmem,
            // then any lane can iterate.
            // But we don't have shmem space for 128 scores. Instead:
            // re-broadcast the scores via simd_shuffle in the inner loop.
            //
            // Cleaner approach: each lane processes a SET of dims and a
            // SET of positions. With POSES_PER_LANE=4 and head_dim=256,
            // dims_per_lane = 8, so 32 lanes × 4 positions × 8 dims is
            // a 4×8 inner block per lane. We do this via simd_shuffle
            // to get scores[i] for any position from the owning lane.
            float acc = 0.0f;
            for (ushort i = 0; i < POSES_PER_LANE; ++i) {
                // Position p = p_tile + i * SIMD_LANES + lane_id
                // The score for that position lives in lane lane_id, register i.
                // So for each (i, lane_id), we want score = scores[i] from lane_id,
                // and v_cache[p, kvh, d].
                for (ushort lane = 0; lane < SIMD_LANES; ++lane) {
                    const float s = simd_shuffle(scores[i], lane);
                    const uint p = p_tile + i * SIMD_LANES + lane;
                    if (p < p_end) {
                        acc += s * v_cache[
                            (ulong)p * args.kv_stride
                            + (ulong)kvh * args.head_dim
                            + d
                        ];
                    }
                }
            }
            tg_scratch[d] += acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        m = m_new;
        l = l_new;
    }

    // Final: out[d] = o[d] / l.
    const float inv_l = 1.0f / l;
    for (uint d = tiisg; d < args.head_dim; d += SIMD_LANES) {
        out_h[d] = tg_scratch[d] * inv_l;
    }
}
