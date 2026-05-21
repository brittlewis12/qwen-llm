// Flash-attention v4 — single-token decode with GQA dedup + online softmax + split-K.
//
// Key design property NOT present in llama.cpp / vllm-metal / candle / MLX:
//   ONE THREADGROUP PER (KV-HEAD, KV-PARTITION), processing all `GROUP`
//   sibling Q-heads cooperatively. K and V are read ONCE per tile and
//   shared across the 6 grouped Q heads, vs. 6× redundant reads in every
//   reference engine.
//
// Hardcoded for Qwen3.6-27B-Q4_K_M:
//   head_dim = 256  (DK = DV = 256)
//   n_q_heads = 24, n_kv_heads = 4  →  GROUP = 6
//   F16 KV cache (matches production path)
//
// Tile-size variants (compile-time, instantiated below):
//   C = 16, 32, 64, 128 — KV positions per inner tile.
//   C=32 is the default (matches llama.cpp vec).
//   C=16 may help short ctx (fewer dot products per tile, more loop overhead).
//   C=64 may help long ctx (fewer tiles, less softmax/barrier overhead, more
//        register pressure during the cc inner loop).
//
// Algorithm (per TG owning kv_head=kvh, partition=iwg):
//   1. Load Q for all 6 group siblings into shmem (prescaled by scale*log2(e)).
//   2. Online softmax over [p_start, p_end) tiles of C KV rows:
//      Phase A — compute QK[g][cc] for all (g, cc), stash in ss[].
//                K row read ONCE, dotted with all 6 Q vectors.
//      Phase B — per-group online softmax update (m, l) using exp2.
//      Phase C — V-aggregate; V row read ONCE, accumulated into 6 O regs.
//   3. Write per-partition (m, l, O_unnormalized) partials.
// A separate reduce kernel combines NWG partials per Q head and divides.
//
// For C != NW (32), Phase B uses an indexed-load pattern: lane t reads
// score `ss[g*C + (t mod C)]` (or `-inf` if t >= C for C<32). simd_max /
// simd_sum still produce the right reductions because the out-of-range
// lanes contribute identity values.
//
// CPU oracle: same as forward.rs attn — math is bit-equivalent up to fp32
// reorder noise (< 1e-4 typical for our shapes).

#include <metal_stdlib>
using namespace metal;

constant constexpr ushort NW = 32;        // simdgroup width
constant constexpr ushort DK = 256;       // K head_dim
constant constexpr ushort DV = 256;       // V head_dim
constant constexpr ushort DK4 = DK / 4;
constant constexpr ushort DV4 = DV / 4;
constant constexpr ushort DK4_PER_LANE = DK4 / NW;   // = 2 for DK=256
constant constexpr ushort DV4_PER_LANE = DV4 / NW;   // = 2 for DV=256
constant constexpr ushort QK8_0 = 32;
constant constexpr ushort Q8_0_BYTES = 34;
constant constexpr ushort DK_Q8_BLOCKS = DK / QK8_0;
constant constexpr ushort DV_Q8_BLOCKS = DV / QK8_0;

struct attn_v4_args {
    uint  n_q_heads;
    uint  n_kv_heads;
    uint  head_dim;            // expected = DK = DV
    uint  n_pos;               // total KV positions (current decode pos + 1)
    uint  kv_stride;           // n_kv_heads * head_dim, in elements
    uint  n_partitions;        // NWG
    uint  rows_per_partition;  // ceil(n_pos / NWG)
    float scale;               // (1/sqrt(head_dim)) * log2(e), prescaled for exp2
};

// =============================================================================
// Templated main kernel body — instantiated below for C ∈ {16, 32, 64}.
// =============================================================================

template <ushort GROUP, ushort C>
inline void attn_v4_main_body(
        constant attn_v4_args & args,
        device const float    * q,
        device const half     * k_cache,
        device const half     * v_cache,
        device       float    * o_partial,
        device       float    * ml_partial,
        threadgroup  half     * sq,
        threadgroup  float    * ss,
        uint3  tgpig,
        ushort tiisg) {
    const uint kvh = tgpig.x;
    const uint iwg = tgpig.z;
    if (kvh >= args.n_kv_heads || iwg >= args.n_partitions) return;

    const uint p_start = iwg * args.rows_per_partition;
    const uint p_end_raw = p_start + args.rows_per_partition;
    const uint p_end = p_end_raw < args.n_pos ? p_end_raw : args.n_pos;

    // ---- Load Q for the GROUP siblings, prescale by scale*log2(e) ----
    {
        device const float4 * q4_base =
            (device const float4 *)(q + (ulong)kvh * GROUP * DK);
        threadgroup half4 * sq4 = (threadgroup half4 *)sq;
        for (ushort g = 0; g < GROUP; ++g) {
            for (ushort ii = 0; ii < DK4_PER_LANE; ++ii) {
                const ushort idx = g * DK4 + ii * NW + tiisg;
                float4 qv = q4_base[idx];
                sq4[idx] = half4(qv * args.scale);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Online softmax state per group sibling, in registers ----
    float  m_state[GROUP];
    float  l_state[GROUP];
    float4 o_acc  [GROUP][DV4_PER_LANE];
    for (ushort g = 0; g < GROUP; ++g) {
        m_state[g] = -INFINITY;
        l_state[g] = 0.0f;
        for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
            o_acc[g][ii] = float4(0.0f);
        }
    }

    // Empty partition (NWG > n_pos for tail partitions): write sentinel and exit.
    if (p_start >= p_end) {
        device float4 * o_out_base = (device float4 *)(
            o_partial + ((ulong)kvh * args.n_partitions + iwg) * GROUP * DV
        );
        for (ushort g = 0; g < GROUP; ++g) {
            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                o_out_base[g * DV4 + ii * NW + tiisg] = float4(0.0f);
            }
        }
        if (tiisg == 0) {
            device float * ml_base = ml_partial
                + ((ulong)kvh * args.n_partitions + iwg) * GROUP * 2;
            for (ushort g = 0; g < GROUP; ++g) {
                ml_base[g * 2 + 0] = -INFINITY;
                ml_base[g * 2 + 1] = 0.0f;
            }
        }
        return;
    }

    // ---- Outer loop over K/V tiles within this partition ----
    for (uint tile_start = p_start; tile_start < p_end; tile_start += C) {
        const uint tile_end_raw = tile_start + C;
        const uint tile_end = tile_end_raw < p_end ? tile_end_raw : p_end;
        const ushort tile_count = (ushort)(tile_end - tile_start);

        // ===== Phase A: compute QK[g][cc] for all (g, cc) ; stash in ss =====
        for (ushort cc = 0; cc < C; ++cc) {
            float partial[GROUP];
            for (ushort g = 0; g < GROUP; ++g) partial[g] = 0.0f;

            if (cc < tile_count) {
                device const half4 * pk4 = (device const half4 *)(
                    k_cache + (ulong)(tile_start + cc) * args.kv_stride
                            + (ulong)kvh * DK
                );
                threadgroup const half4 * sq4 = (threadgroup const half4 *)sq;
                for (ushort ii = 0; ii < DK4_PER_LANE; ++ii) {
                    const half4 k_chunk = pk4[ii * NW + tiisg];
                    const float4 k_f32 = float4(k_chunk);
                    for (ushort g = 0; g < GROUP; ++g) {
                        const float4 q_f32 = float4(sq4[g * DK4 + ii * NW + tiisg]);
                        partial[g] += dot(k_f32, q_f32);
                    }
                }
            }
            for (ushort g = 0; g < GROUP; ++g) {
                const float qk = simd_sum(partial[g]);
                if (tiisg == 0) {
                    ss[g * C + cc] = (cc < tile_count) ? qk : -INFINITY;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ===== Phase B: online softmax update per group sibling =====
        //
        // For C == 32 (default): lane t owns column t.
        // For C == 16: lanes 0..15 own one column each; lanes 16..31 hold -inf.
        // For C == 64: lane t owns columns {t, t+32}; we accumulate two scores
        //              per lane and take the per-lane max separately, then
        //              do simd_max once.
        for (ushort g = 0; g < GROUP; ++g) {
            float scores[(C + NW - 1) / NW];
            float weights[(C + NW - 1) / NW];
            float per_lane_max = -INFINITY;
            float per_lane_sum = 0.0f;

            // Stage 1: load scores (one or two slots per lane)
            for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                const ushort col = k * NW + tiisg;
                if (col < C) {
                    scores[k] = ss[g * C + col];
                } else {
                    scores[k] = -INFINITY;
                }
                per_lane_max = max(per_lane_max, scores[k]);
            }

            const float tile_max = simd_max(per_lane_max);
            const float new_m = max(m_state[g], tile_max);
            const float factor = (m_state[g] == -INFINITY) ? 0.0f
                                                            : exp2(m_state[g] - new_m);

            // Stage 2: compute weights = exp2(score - new_m); accumulate per-lane sum
            for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                weights[k] = (scores[k] == -INFINITY) ? 0.0f : exp2(scores[k] - new_m);
                per_lane_sum += weights[k];
            }
            const float tile_l = simd_sum(per_lane_sum);

            // Rescale O accumulator.
            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                o_acc[g][ii] *= factor;
            }
            // Write weights back to ss for V-aggregate.
            for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                const ushort col = k * NW + tiisg;
                if (col < C) {
                    ss[g * C + col] = weights[k];
                }
            }

            l_state[g] = l_state[g] * factor + tile_l;
            m_state[g] = new_m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ===== Phase C: V-aggregate =====
        for (ushort cc = 0; cc < tile_count; ++cc) {
            device const half4 * pv4 = (device const half4 *)(
                v_cache + (ulong)(tile_start + cc) * args.kv_stride
                        + (ulong)kvh * DV
            );
            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                const half4 v_chunk = pv4[ii * NW + tiisg];
                const float4 v_f32 = float4(v_chunk);
                for (ushort g = 0; g < GROUP; ++g) {
                    const float w = ss[g * C + cc];
                    o_acc[g][ii] += w * v_f32;
                }
            }
        }
    }

    // ---- Write per-partition partials ----
    {
        device float4 * o_out_base = (device float4 *)(
            o_partial + ((ulong)kvh * args.n_partitions + iwg) * GROUP * DV
        );
        for (ushort g = 0; g < GROUP; ++g) {
            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                o_out_base[g * DV4 + ii * NW + tiisg] = o_acc[g][ii];
            }
        }
    }
    if (tiisg == 0) {
        device float * ml_base = ml_partial
            + ((ulong)kvh * args.n_partitions + iwg) * GROUP * 2;
        for (ushort g = 0; g < GROUP; ++g) {
            ml_base[g * 2 + 0] = m_state[g];
            ml_base[g * 2 + 1] = l_state[g];
        }
    }
}

template <ushort GROUP, ushort C>
inline void attn_v4_main_body_q8(
        constant attn_v4_args & args,
        device const float    * q,
        device const uchar    * k_cache,
        device const uchar    * v_cache,
        device       float    * o_partial,
        device       float    * ml_partial,
        threadgroup  half     * sq,
        threadgroup  float    * ss,
        uint3  tgpig,
        ushort tiisg) {
    const uint kvh = tgpig.x;
    const uint iwg = tgpig.z;
    if (kvh >= args.n_kv_heads || iwg >= args.n_partitions) return;

    const uint p_start = iwg * args.rows_per_partition;
    const uint p_end_raw = p_start + args.rows_per_partition;
    const uint p_end = p_end_raw < args.n_pos ? p_end_raw : args.n_pos;

    {
        device const float4 * q4_base =
            (device const float4 *)(q + (ulong)kvh * GROUP * DK);
        threadgroup half4 * sq4 = (threadgroup half4 *)sq;
        for (ushort g = 0; g < GROUP; ++g) {
            for (ushort ii = 0; ii < DK4_PER_LANE; ++ii) {
                const ushort idx = g * DK4 + ii * NW + tiisg;
                float4 qv = q4_base[idx];
                sq4[idx] = half4(qv * args.scale);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_state[GROUP];
    float l_state[GROUP];
    float o_acc[GROUP][DV_Q8_BLOCKS];
    for (ushort g = 0; g < GROUP; ++g) {
        m_state[g] = -INFINITY;
        l_state[g] = 0.0f;
        for (ushort ii = 0; ii < DV_Q8_BLOCKS; ++ii) {
            o_acc[g][ii] = 0.0f;
        }
    }

    if (p_start >= p_end) {
        device float * o_out_base =
            o_partial + ((ulong)kvh * args.n_partitions + iwg) * GROUP * DV;
        for (ushort g = 0; g < GROUP; ++g) {
            for (ushort ii = 0; ii < DV_Q8_BLOCKS; ++ii) {
                o_out_base[g * DV + ii * QK8_0 + tiisg] = 0.0f;
            }
        }
        if (tiisg == 0) {
            device float * ml_base = ml_partial
                + ((ulong)kvh * args.n_partitions + iwg) * GROUP * 2;
            for (ushort g = 0; g < GROUP; ++g) {
                ml_base[g * 2 + 0] = -INFINITY;
                ml_base[g * 2 + 1] = 0.0f;
            }
        }
        return;
    }

    for (uint tile_start = p_start; tile_start < p_end; tile_start += C) {
        const uint tile_end_raw = tile_start + C;
        const uint tile_end = tile_end_raw < p_end ? tile_end_raw : p_end;
        const ushort tile_count = (ushort)(tile_end - tile_start);

        for (ushort cc = 0; cc < C; ++cc) {
            float partial[GROUP];
            for (ushort g = 0; g < GROUP; ++g) partial[g] = 0.0f;

            if (cc < tile_count) {
                const ulong elem_base = (ulong)(tile_start + cc) * args.kv_stride
                                      + (ulong)kvh * DK;
                const ulong blk_base = elem_base / QK8_0;
                for (ushort ii = 0; ii < DK_Q8_BLOCKS; ++ii) {
                    device const uchar * blk = k_cache + (blk_base + ii) * Q8_0_BYTES;
                    const float d = (float)((device const half *)blk)[0];
                    const float kf = d * (float)((device const int8_t *)(blk + 2))[tiisg];
                    const ushort q_off = ii * QK8_0 + tiisg;
                    for (ushort g = 0; g < GROUP; ++g) {
                        partial[g] += kf * (float)sq[g * DK + q_off];
                    }
                }
            }
            for (ushort g = 0; g < GROUP; ++g) {
                const float qk = simd_sum(partial[g]);
                if (tiisg == 0) {
                    ss[g * C + cc] = (cc < tile_count) ? qk : -INFINITY;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort g = 0; g < GROUP; ++g) {
            float scores[(C + NW - 1) / NW];
            float weights[(C + NW - 1) / NW];
            float per_lane_max = -INFINITY;
            float per_lane_sum = 0.0f;

            for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                const ushort col = k * NW + tiisg;
                scores[k] = (col < C) ? ss[g * C + col] : -INFINITY;
                per_lane_max = max(per_lane_max, scores[k]);
            }

            const float tile_max = simd_max(per_lane_max);
            const float new_m = max(m_state[g], tile_max);
            const float factor = (m_state[g] == -INFINITY) ? 0.0f
                                                            : exp2(m_state[g] - new_m);

            for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                weights[k] = (scores[k] == -INFINITY) ? 0.0f : exp2(scores[k] - new_m);
                per_lane_sum += weights[k];
            }
            const float tile_l = simd_sum(per_lane_sum);

            for (ushort ii = 0; ii < DV_Q8_BLOCKS; ++ii) {
                o_acc[g][ii] *= factor;
            }
            for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                const ushort col = k * NW + tiisg;
                if (col < C) {
                    ss[g * C + col] = weights[k];
                }
            }

            l_state[g] = l_state[g] * factor + tile_l;
            m_state[g] = new_m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort cc = 0; cc < tile_count; ++cc) {
            const ulong elem_base = (ulong)(tile_start + cc) * args.kv_stride
                                  + (ulong)kvh * DV;
            const ulong blk_base = elem_base / QK8_0;
            for (ushort ii = 0; ii < DV_Q8_BLOCKS; ++ii) {
                device const uchar * blk = v_cache + (blk_base + ii) * Q8_0_BYTES;
                const float d = (float)((device const half *)blk)[0];
                const float vf = d * (float)((device const int8_t *)(blk + 2))[tiisg];
                for (ushort g = 0; g < GROUP; ++g) {
                    o_acc[g][ii] += ss[g * C + cc] * vf;
                }
            }
        }
    }

    {
        device float * o_out_base =
            o_partial + ((ulong)kvh * args.n_partitions + iwg) * GROUP * DV;
        for (ushort g = 0; g < GROUP; ++g) {
            for (ushort ii = 0; ii < DV_Q8_BLOCKS; ++ii) {
                o_out_base[g * DV + ii * QK8_0 + tiisg] = o_acc[g][ii];
            }
        }
    }
    if (tiisg == 0) {
        device float * ml_base = ml_partial
            + ((ulong)kvh * args.n_partitions + iwg) * GROUP * 2;
        for (ushort g = 0; g < GROUP; ++g) {
            ml_base[g * 2 + 0] = m_state[g];
            ml_base[g * 2 + 1] = l_state[g];
        }
    }
}

template <ushort GROUP_TOTAL, ushort GROUP_TILE, ushort C>
inline void attn_v4_main_subgroup_body(
        constant attn_v4_args & args,
        device const float    * q,
        device const half     * k_cache,
        device const half     * v_cache,
        device       float    * o_partial,
        device       float    * ml_partial,
        threadgroup  half     * sq,
        threadgroup  float    * ss,
        uint3  tgpig,
        ushort tiisg) {
    const uint kvh = tgpig.x;
    const uint gtile = tgpig.y;
    const uint iwg = tgpig.z;
    if (kvh >= args.n_kv_heads || iwg >= args.n_partitions) return;

    const ushort g_base = (ushort)(gtile * GROUP_TILE);
    if (g_base >= GROUP_TOTAL) return;

    const uint p_start = iwg * args.rows_per_partition;
    const uint p_end_raw = p_start + args.rows_per_partition;
    const uint p_end = p_end_raw < args.n_pos ? p_end_raw : args.n_pos;

    {
        device const float4 * q4_base =
            (device const float4 *)(q + ((ulong)kvh * GROUP_TOTAL + g_base) * DK);
        threadgroup half4 * sq4 = (threadgroup half4 *)sq;
        for (ushort g = 0; g < GROUP_TILE; ++g) {
            for (ushort ii = 0; ii < DK4_PER_LANE; ++ii) {
                const ushort idx = g * DK4 + ii * NW + tiisg;
                float4 qv = q4_base[idx];
                sq4[idx] = half4(qv * args.scale);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float  m_state[GROUP_TILE];
    float  l_state[GROUP_TILE];
    float4 o_acc  [GROUP_TILE][DV4_PER_LANE];
    for (ushort g = 0; g < GROUP_TILE; ++g) {
        m_state[g] = -INFINITY;
        l_state[g] = 0.0f;
        for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
            o_acc[g][ii] = float4(0.0f);
        }
    }

    if (p_start >= p_end) {
        device float4 * o_out_base = (device float4 *)(
            o_partial + ((ulong)kvh * args.n_partitions + iwg) * GROUP_TOTAL * DV
                      + (ulong)g_base * DV
        );
        for (ushort g = 0; g < GROUP_TILE; ++g) {
            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                o_out_base[g * DV4 + ii * NW + tiisg] = float4(0.0f);
            }
        }
        if (tiisg == 0) {
            device float * ml_base = ml_partial
                + ((ulong)kvh * args.n_partitions + iwg) * GROUP_TOTAL * 2
                + (ulong)g_base * 2;
            for (ushort g = 0; g < GROUP_TILE; ++g) {
                ml_base[g * 2 + 0] = -INFINITY;
                ml_base[g * 2 + 1] = 0.0f;
            }
        }
        return;
    }

    for (uint tile_start = p_start; tile_start < p_end; tile_start += C) {
        const uint tile_end_raw = tile_start + C;
        const uint tile_end = tile_end_raw < p_end ? tile_end_raw : p_end;
        const ushort tile_count = (ushort)(tile_end - tile_start);

        for (ushort cc = 0; cc < C; ++cc) {
            float partial[GROUP_TILE];
            for (ushort g = 0; g < GROUP_TILE; ++g) partial[g] = 0.0f;

            if (cc < tile_count) {
                device const half4 * pk4 = (device const half4 *)(
                    k_cache + (ulong)(tile_start + cc) * args.kv_stride
                            + (ulong)kvh * DK
                );
                threadgroup const half4 * sq4 = (threadgroup const half4 *)sq;
                for (ushort ii = 0; ii < DK4_PER_LANE; ++ii) {
                    const half4 k_chunk = pk4[ii * NW + tiisg];
                    const float4 k_f32 = float4(k_chunk);
                    for (ushort g = 0; g < GROUP_TILE; ++g) {
                        const float4 q_f32 = float4(sq4[g * DK4 + ii * NW + tiisg]);
                        partial[g] += dot(k_f32, q_f32);
                    }
                }
            }
            for (ushort g = 0; g < GROUP_TILE; ++g) {
                const float qk = simd_sum(partial[g]);
                if (tiisg == 0) {
                    ss[g * C + cc] = (cc < tile_count) ? qk : -INFINITY;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort g = 0; g < GROUP_TILE; ++g) {
            float scores[(C + NW - 1) / NW];
            float weights[(C + NW - 1) / NW];
            float per_lane_max = -INFINITY;
            float per_lane_sum = 0.0f;

            for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                const ushort col = k * NW + tiisg;
                scores[k] = (col < C) ? ss[g * C + col] : -INFINITY;
                per_lane_max = max(per_lane_max, scores[k]);
            }

            const float tile_max = simd_max(per_lane_max);
            const float new_m = max(m_state[g], tile_max);
            const float factor = (m_state[g] == -INFINITY) ? 0.0f
                                                            : exp2(m_state[g] - new_m);

            for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                weights[k] = (scores[k] == -INFINITY) ? 0.0f : exp2(scores[k] - new_m);
                per_lane_sum += weights[k];
            }
            const float tile_l = simd_sum(per_lane_sum);

            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                o_acc[g][ii] *= factor;
            }
            for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                const ushort col = k * NW + tiisg;
                if (col < C) {
                    ss[g * C + col] = weights[k];
                }
            }

            l_state[g] = l_state[g] * factor + tile_l;
            m_state[g] = new_m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort cc = 0; cc < tile_count; ++cc) {
            device const half4 * pv4 = (device const half4 *)(
                v_cache + (ulong)(tile_start + cc) * args.kv_stride
                        + (ulong)kvh * DV
            );
            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                const half4 v_chunk = pv4[ii * NW + tiisg];
                const float4 v_f32 = float4(v_chunk);
                for (ushort g = 0; g < GROUP_TILE; ++g) {
                    const float w = ss[g * C + cc];
                    o_acc[g][ii] += w * v_f32;
                }
            }
        }
    }

    {
        device float4 * o_out_base = (device float4 *)(
            o_partial + ((ulong)kvh * args.n_partitions + iwg) * GROUP_TOTAL * DV
                      + (ulong)g_base * DV
        );
        for (ushort g = 0; g < GROUP_TILE; ++g) {
            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                o_out_base[g * DV4 + ii * NW + tiisg] = o_acc[g][ii];
            }
        }
    }
    if (tiisg == 0) {
        device float * ml_base = ml_partial
            + ((ulong)kvh * args.n_partitions + iwg) * GROUP_TOTAL * 2
            + (ulong)g_base * 2;
        for (ushort g = 0; g < GROUP_TILE; ++g) {
            ml_base[g * 2 + 0] = m_state[g];
            ml_base[g * 2 + 1] = l_state[g];
        }
    }
}

// =============================================================================
// Concrete kernel entry points (one per C variant)
// =============================================================================

#define ATTN_V4_KERNEL_G(NAME, GROUP_VAL, C_VAL) \
kernel void NAME( \
        constant attn_v4_args & args      [[buffer(0)]], \
        device const float    * q          [[buffer(1)]], \
        device const half     * k_cache    [[buffer(2)]], \
        device const half     * v_cache    [[buffer(3)]], \
        device       float    * o_partial  [[buffer(4)]], \
        device       float    * ml_partial [[buffer(5)]], \
        threadgroup  half     * sq         [[threadgroup(0)]], \
        threadgroup  float    * ss         [[threadgroup(1)]], \
        uint3  tgpig [[threadgroup_position_in_grid]], \
        ushort tiisg [[thread_index_in_simdgroup]]) { \
    attn_v4_main_body<GROUP_VAL, C_VAL>(args, q, k_cache, v_cache, o_partial, ml_partial, \
                               sq, ss, tgpig, tiisg); \
}

ATTN_V4_KERNEL_G(kernel_attn_decode_v4_g4_c16_f32, 4, 16)
ATTN_V4_KERNEL_G(kernel_attn_decode_v4_g4_f32,     4, 32)
ATTN_V4_KERNEL_G(kernel_attn_decode_v4_g4_c64_f32, 4, 64)
ATTN_V4_KERNEL_G(kernel_attn_decode_v4_g4_c128_f32, 4, 128)

ATTN_V4_KERNEL_G(kernel_attn_decode_v4_c16_f32, 6, 16)
ATTN_V4_KERNEL_G(kernel_attn_decode_v4_f32,     6, 32)  // default name (GROUP=6, C=32 backward-compat)
ATTN_V4_KERNEL_G(kernel_attn_decode_v4_c64_f32, 6, 64)
ATTN_V4_KERNEL_G(kernel_attn_decode_v4_c128_f32, 6, 128)

#define ATTN_V4_Q8_KERNEL(NAME, C_VAL) \
kernel void NAME( \
        constant attn_v4_args & args      [[buffer(0)]], \
        device const float    * q          [[buffer(1)]], \
        device const uchar    * k_cache    [[buffer(2)]], \
        device const uchar    * v_cache    [[buffer(3)]], \
        device       float    * o_partial  [[buffer(4)]], \
        device       float    * ml_partial [[buffer(5)]], \
        threadgroup  half     * sq         [[threadgroup(0)]], \
        threadgroup  float    * ss         [[threadgroup(1)]], \
        uint3  tgpig [[threadgroup_position_in_grid]], \
        ushort tiisg [[thread_index_in_simdgroup]]) { \
    attn_v4_main_body_q8<6, C_VAL>(args, q, k_cache, v_cache, o_partial, ml_partial, \
                                   sq, ss, tgpig, tiisg); \
}

ATTN_V4_Q8_KERNEL(kernel_attn_decode_v4_q8_c16_f32, 16)
ATTN_V4_Q8_KERNEL(kernel_attn_decode_v4_q8_f32,     32)
ATTN_V4_Q8_KERNEL(kernel_attn_decode_v4_q8_c64_f32, 64)
ATTN_V4_Q8_KERNEL(kernel_attn_decode_v4_q8_c128_f32, 128)

kernel void kernel_attn_decode_v4_g8_c16_f32(
        constant attn_v4_args & args      [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_main_body<8, 16>(args, q, k_cache, v_cache, o_partial, ml_partial, sq, ss, tgpig, tiisg);
}

kernel void kernel_attn_decode_v4_g8_f32(
        constant attn_v4_args & args      [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_main_body<8, 32>(args, q, k_cache, v_cache, o_partial, ml_partial, sq, ss, tgpig, tiisg);
}

kernel void kernel_attn_decode_v4_g8_c64_f32(
        constant attn_v4_args & args      [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_main_body<8, 64>(args, q, k_cache, v_cache, o_partial, ml_partial, sq, ss, tgpig, tiisg);
}

kernel void kernel_attn_decode_v4_g8_c128_f32(
        constant attn_v4_args & args      [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_main_body<8, 128>(args, q, k_cache, v_cache, o_partial, ml_partial, sq, ss, tgpig, tiisg);
}

kernel void kernel_attn_decode_v4_g16_c16_f32(
        constant attn_v4_args & args      [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_main_body<16, 16>(args, q, k_cache, v_cache, o_partial, ml_partial, sq, ss, tgpig, tiisg);
}

kernel void kernel_attn_decode_v4_g16_f32(
        constant attn_v4_args & args      [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_main_body<16, 32>(args, q, k_cache, v_cache, o_partial, ml_partial, sq, ss, tgpig, tiisg);
}

kernel void kernel_attn_decode_v4_g16_c64_f32(
        constant attn_v4_args & args      [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_main_body<16, 64>(args, q, k_cache, v_cache, o_partial, ml_partial, sq, ss, tgpig, tiisg);
}

kernel void kernel_attn_decode_v4_g16_c128_f32(
        constant attn_v4_args & args      [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_main_body<16, 128>(args, q, k_cache, v_cache, o_partial, ml_partial, sq, ss, tgpig, tiisg);
}

#define ATTN_V4_G16_SUBGROUP_KERNEL(NAME, GROUP_TILE_VAL, C_VAL) \
kernel void NAME( \
        constant attn_v4_args & args      [[buffer(0)]], \
        device const float    * q          [[buffer(1)]], \
        device const half     * k_cache    [[buffer(2)]], \
        device const half     * v_cache    [[buffer(3)]], \
        device       float    * o_partial  [[buffer(4)]], \
        device       float    * ml_partial [[buffer(5)]], \
        threadgroup  half     * sq         [[threadgroup(0)]], \
        threadgroup  float    * ss         [[threadgroup(1)]], \
        uint3  tgpig [[threadgroup_position_in_grid]], \
        ushort tiisg [[thread_index_in_simdgroup]]) { \
    attn_v4_main_subgroup_body<16, GROUP_TILE_VAL, C_VAL>(args, q, k_cache, v_cache, o_partial, ml_partial, \
                                                          sq, ss, tgpig, tiisg); \
}

ATTN_V4_G16_SUBGROUP_KERNEL(kernel_attn_decode_v4_g16_t8_c16_f32,   8, 16)
ATTN_V4_G16_SUBGROUP_KERNEL(kernel_attn_decode_v4_g16_t8_f32,       8, 32)
ATTN_V4_G16_SUBGROUP_KERNEL(kernel_attn_decode_v4_g16_t8_c64_f32,   8, 64)
ATTN_V4_G16_SUBGROUP_KERNEL(kernel_attn_decode_v4_g16_t8_c128_f32,  8, 128)
ATTN_V4_G16_SUBGROUP_KERNEL(kernel_attn_decode_v4_g16_t4_c16_f32,   4, 16)
ATTN_V4_G16_SUBGROUP_KERNEL(kernel_attn_decode_v4_g16_t4_f32,       4, 32)
ATTN_V4_G16_SUBGROUP_KERNEL(kernel_attn_decode_v4_g16_t4_c64_f32,   4, 64)
ATTN_V4_G16_SUBGROUP_KERNEL(kernel_attn_decode_v4_g16_t4_c128_f32,  4, 128)

#define ATTN_V4_G8_SUBGROUP_KERNEL(NAME, GROUP_TILE_VAL, C_VAL) \
kernel void NAME( \
        constant attn_v4_args & args      [[buffer(0)]], \
        device const float    * q          [[buffer(1)]], \
        device const half     * k_cache    [[buffer(2)]], \
        device const half     * v_cache    [[buffer(3)]], \
        device       float    * o_partial  [[buffer(4)]], \
        device       float    * ml_partial [[buffer(5)]], \
        threadgroup  half     * sq         [[threadgroup(0)]], \
        threadgroup  float    * ss         [[threadgroup(1)]], \
        uint3  tgpig [[threadgroup_position_in_grid]], \
        ushort tiisg [[thread_index_in_simdgroup]]) { \
    attn_v4_main_subgroup_body<8, GROUP_TILE_VAL, C_VAL>(args, q, k_cache, v_cache, o_partial, ml_partial, \
                                                         sq, ss, tgpig, tiisg); \
}

ATTN_V4_G8_SUBGROUP_KERNEL(kernel_attn_decode_v4_g8_t4_c16_f32,   4, 16)
ATTN_V4_G8_SUBGROUP_KERNEL(kernel_attn_decode_v4_g8_t4_f32,       4, 32)
ATTN_V4_G8_SUBGROUP_KERNEL(kernel_attn_decode_v4_g8_t4_c64_f32,   4, 64)
ATTN_V4_G8_SUBGROUP_KERNEL(kernel_attn_decode_v4_g8_t4_c128_f32,  4, 128)
ATTN_V4_G8_SUBGROUP_KERNEL(kernel_attn_decode_v4_g8_t2_c16_f32,   2, 16)
ATTN_V4_G8_SUBGROUP_KERNEL(kernel_attn_decode_v4_g8_t2_f32,       2, 32)
ATTN_V4_G8_SUBGROUP_KERNEL(kernel_attn_decode_v4_g8_t2_c64_f32,   2, 64)
ATTN_V4_G8_SUBGROUP_KERNEL(kernel_attn_decode_v4_g8_t2_c128_f32,  2, 128)

// ============================================================================
// Reduce kernel: combines NWG partials per Q head, normalizes by global l.
//
// Grid: (n_q_heads, 1, 1).  Threadgroup: 32 lanes (one simdgroup).
//
// Constraint v1: NWG <= 64. The reduce path uses threadgroup scratch to hold
// up to 64 per-partition factors, so the main-pass split-K count can now go
// beyond one simdgroup if the synthetic sweeps justify it.

struct attn_v4_reduce_args {
    uint n_q_heads;
    uint n_kv_heads;
    uint head_dim;          // expected = DV
    uint n_partitions;      // NWG
};

template <ushort GROUP>
inline void attn_v4_reduce_body(
        constant attn_v4_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]], // [n_kv_heads, NWG, GROUP, head_dim]
        device const float * ml_partial  [[buffer(2)]], // [n_kv_heads, NWG, GROUP, 2]
        device       float * out         [[buffer(3)]], // [n_q_heads, head_dim]
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint qh = tgpig.x;
    if (qh >= args.n_q_heads) return;
    const uint kvh = qh / GROUP;
    const uint g = qh % GROUP;
    const uint nwg = args.n_partitions;

    // Load up to two partitions per lane into threadgroup scratch.
    for (ushort pass = 0; pass < 2; ++pass) {
        const uint part = tiisg + pass * 32;
        float m_part = -INFINITY;
        float l_part = 0.0f;
        if (part < nwg) {
            device const float * ml_base = ml_partial
                + ((ulong)kvh * nwg + part) * GROUP * 2;
            m_part = ml_base[g * 2 + 0];
            l_part = ml_base[g * 2 + 1];
        }
        sh_m[part] = m_part;
        sh_l[part] = l_part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_local = -INFINITY;
    for (uint part = tiisg; part < nwg; part += 32) {
        m_local = max(m_local, sh_m[part]);
    }
    const float m_global = simd_max(m_local);

    float l_local = 0.0f;
    for (uint part = tiisg; part < nwg; part += 32) {
        const float ef = (sh_m[part] == -INFINITY || m_global == -INFINITY)
            ? 0.0f
            : exp2(sh_m[part] - m_global);
        sh_ef[part] = ef;
        l_local += sh_l[part] * ef;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float l_global = simd_sum(l_local);
    const float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;

    for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
        float4 acc = float4(0.0f);
        for (uint i = 0; i < nwg; ++i) {
            const float ef = sh_ef[i];
            const ulong off = ((ulong)kvh * nwg + i) * GROUP * DV
                            + (ulong)g * DV
                            + (ulong)(ii * NW + tiisg) * 4;
            const device float4 * src4 = (device const float4 *)(o_partial + off);
            acc += (*src4) * ef;
        }
        device float4 * out4 = (device float4 *)(
            out + (ulong)qh * DV + (ulong)(ii * NW + tiisg) * 4
        );
        *out4 = acc * inv_l;
    }
}

kernel void kernel_attn_decode_v4_reduce_f32(
        constant attn_v4_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_reduce_body<6>(args, o_partial, ml_partial, out, sh_m, sh_l, sh_ef, tgpig, tiisg);
}

kernel void kernel_attn_decode_v4_reduce_g4_f32(
        constant attn_v4_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_reduce_body<4>(args, o_partial, ml_partial, out, sh_m, sh_l, sh_ef, tgpig, tiisg);
}

kernel void kernel_attn_decode_v4_reduce_g8_f32(
        constant attn_v4_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_reduce_body<8>(args, o_partial, ml_partial, out, sh_m, sh_l, sh_ef, tgpig, tiisg);
}

kernel void kernel_attn_decode_v4_reduce_g16_f32(
        constant attn_v4_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_reduce_body<16>(args, o_partial, ml_partial, out, sh_m, sh_l, sh_ef, tgpig, tiisg);
}
