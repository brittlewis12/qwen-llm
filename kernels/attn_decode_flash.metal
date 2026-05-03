// Flash-attention-style single-token decode (v2).
//
// Replaces the naive `kernel_attn_decode_f32` for long contexts. Online
// (running-max + running-sum) softmax over K/V tiles. Each KV row is
// read EXACTLY ONCE (bandwidth-optimal).
//
// One threadgroup per Q head, single simdgroup (32 lanes). Per tile:
//   1. Each lane loads its 1-of-TILE K row, computes its score.
//   2. Tile max reduced across lanes via simd_max.
//   3. Rescale running (m, l, o); add tile contribution.
//
// Critical fix vs v1: V-aggregate uses sequential per-position
// accumulation, NOT simd_shuffle broadcasting. Each lane processes its
// own positions and adds (score * v_row) to the output accumulator
// (held in shared memory across head_dim). The cross-lane reduce
// happens via threadgroup-memory atomic accumulate (since each lane
// touches different positions but all dims).

#include <metal_stdlib>
using namespace metal;

constant constexpr ushort SIMD_LANES = 32;
constant constexpr ushort TILE = 32; // one position per lane per iter

struct attn_decode_flash_args {
    uint  n_q_heads;
    uint  n_kv_heads;
    uint  head_dim;
    uint  n_pos;
    uint  kv_stride;
    float scale;
};

kernel void kernel_attn_decode_flash_f32(
        constant attn_decode_flash_args & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const float * k_cache [[buffer(2)]],
        device const float * v_cache [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        threadgroup  float * tg_o    [[threadgroup(0)]], // [head_dim]
        uint  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    if (sgitg != 0) return;
    const uint qh = tgpig;
    if (qh >= args.n_q_heads) return;

    const uint group = args.n_q_heads / args.n_kv_heads;
    const uint kvh = qh / group;

    device const float * q_h = q + (ulong)qh * args.head_dim;
    device       float * out_h = out + (ulong)qh * args.head_dim;

    // Initialize the o-accumulator (head_dim floats in shared memory).
    for (uint d = tiisg; d < args.head_dim; d += SIMD_LANES) {
        tg_o[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state.
    float m = -INFINITY;
    float l = 0.0f;

    // Stride through K/V positions in tiles of TILE size = SIMD_LANES.
    // Each lane handles one position per tile (lane `tiisg` -> position
    // p_tile + tiisg).
    for (uint p_tile = 0; p_tile < args.n_pos; p_tile += TILE) {
        const uint p = p_tile + tiisg;
        const bool valid = p < args.n_pos;

        // (1) Score: q · k[p]. Each lane computes its own score.
        float score = 0.0f;
        if (valid) {
            device const float * k_p = k_cache
                + (ulong)p * args.kv_stride
                + (ulong)kvh * args.head_dim;
            for (uint d = 0; d < args.head_dim; ++d) {
                score += q_h[d] * k_p[d];
            }
            score *= args.scale;
        } else {
            score = -INFINITY;
        }

        // (2) Tile max + global rescale.
        const float tile_max = simd_max(score);
        const float m_new = max(m, tile_max);
        const float rescale = (m == -INFINITY) ? 0.0f : exp(m - m_new);

        // (3) Convert score → exp(score - m_new) (per-lane, then sum).
        const float weight = valid ? exp(score - m_new) : 0.0f;
        const float tile_sum = simd_sum(weight);
        const float l_new = l * rescale + tile_sum;

        // (4) Rescale o-accumulator. Then accumulate this tile's V
        // contribution: tg_o[d] += sum_p (weight[p] * v_cache[p, kvh, d]).
        //
        // Memory access strategy: each lane owns one position; it walks
        // its V row sequentially across head_dim, atomically (well, via
        // serial+barrier) accumulating into the shared o buffer.
        if (rescale != 1.0f) {
            for (uint d = tiisg; d < args.head_dim; d += SIMD_LANES) {
                tg_o[d] *= rescale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Per-lane V contribution. We need to add `weight * v_row` to
        // tg_o, summed across all 32 lanes. Do this via simd_sum per
        // dim — for each dim d we want sum_lane(weight_lane * v_lane[d]).
        //
        // Each lane reads its own v_row[d] (one read per d per lane,
        // total = head_dim reads per lane = head_dim*32 reads per tile,
        // = 32*32 = 1024 reads per tile for head_dim=32 ... but
        // head_dim=256 so 256*32 = 8192 reads per tile.
        //
        // BUT the natural pattern is: each lane reads its v_row once
        // (head_dim reads), and contributes to all dims of tg_o. That's
        // 256 reads per lane per tile = 8192 reads per tile total =
        // 32 KB / tile. With 4096 positions / 32 = 128 tiles, total =
        // 4 MB read of V per Q head. Per token: 24 Q heads × 4 MB = 96 MB
        // of V reads at 4K context. That's the LOWER bound — same as
        // naive.
        //
        // The serialization happens across lanes: each lane writes its
        // (weight × v_row[d]) into the shared accumulator for ONE
        // d-slice at a time, with the cross-lane sum via simd_sum.
        for (uint d = 0; d < args.head_dim; ++d) {
            float my_contrib = 0.0f;
            if (valid) {
                const float v_d = v_cache[
                    (ulong)p * args.kv_stride
                    + (ulong)kvh * args.head_dim
                    + d
                ];
                my_contrib = weight * v_d;
            }
            const float sum = simd_sum(my_contrib);
            if (tiisg == 0) {
                tg_o[d] += sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m = m_new;
        l = l_new;
    }

    // Final: out[d] = o[d] / l.
    const float inv_l = 1.0f / l;
    for (uint d = tiisg; d < args.head_dim; d += SIMD_LANES) {
        out_h[d] = tg_o[d] * inv_l;
    }
}
