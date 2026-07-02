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

template <ushort GROUP_TOTAL, ushort GROUP_TILE, ushort C, bool HEAD_MAJOR>
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
                const ulong k_off = HEAD_MAJOR
                    ? (((ulong)kvh * args.n_pos + (ulong)(tile_start + cc)) * DK)
                    : ((ulong)(tile_start + cc) * args.kv_stride + (ulong)kvh * DK);
                device const half4 * pk4 = (device const half4 *)(
                    k_cache + k_off
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
            const ulong v_off = HEAD_MAJOR
                ? (((ulong)kvh * args.n_pos + (ulong)(tile_start + cc)) * DV)
                : ((ulong)(tile_start + cc) * args.kv_stride + (ulong)kvh * DV);
            device const half4 * pv4 = (device const half4 *)(
                v_cache + v_off
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
[[max_total_threads_per_threadgroup(32)]] \
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
[[max_total_threads_per_threadgroup(32)]] \
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

[[max_total_threads_per_threadgroup(32)]]
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

[[max_total_threads_per_threadgroup(32)]]
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

[[max_total_threads_per_threadgroup(32)]]
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

[[max_total_threads_per_threadgroup(32)]]
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

[[max_total_threads_per_threadgroup(32)]]
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

[[max_total_threads_per_threadgroup(32)]]
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

[[max_total_threads_per_threadgroup(32)]]
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

[[max_total_threads_per_threadgroup(32)]]
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
[[max_total_threads_per_threadgroup(32)]] \
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
    attn_v4_main_subgroup_body<16, GROUP_TILE_VAL, C_VAL, false>(args, q, k_cache, v_cache, o_partial, ml_partial, \
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
[[max_total_threads_per_threadgroup(32)]] \
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
    attn_v4_main_subgroup_body<8, GROUP_TILE_VAL, C_VAL, false>(args, q, k_cache, v_cache, o_partial, ml_partial, \
                                                         sq, ss, tgpig, tiisg); \
}

#define ATTN_V4_G16_SUBGROUP_HM_KERNEL(NAME, GROUP_TILE_VAL, C_VAL) \
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
    attn_v4_main_subgroup_body<16, GROUP_TILE_VAL, C_VAL, true>(args, q, k_cache, v_cache, o_partial, ml_partial, \
                                                          sq, ss, tgpig, tiisg); \
}

#define ATTN_V4_G8_SUBGROUP_HM_KERNEL(NAME, GROUP_TILE_VAL, C_VAL) \
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
    attn_v4_main_subgroup_body<8, GROUP_TILE_VAL, C_VAL, true>(args, q, k_cache, v_cache, o_partial, ml_partial, \
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

ATTN_V4_G16_SUBGROUP_HM_KERNEL(kernel_attn_decode_v4_g16_t4_c64_hm_f32, 4, 64)
ATTN_V4_G16_SUBGROUP_HM_KERNEL(kernel_attn_decode_v4_g16_t4_c128_hm_f32, 4, 128)
ATTN_V4_G8_SUBGROUP_HM_KERNEL(kernel_attn_decode_v4_g8_t2_c64_hm_f32, 2, 64)

// ============================================================================
// Packed prefill microproof: A3B/group8 prompt-native attention over QT rows.
//
// This is intentionally narrow and specialized:
// - GROUP_TOTAL = 8, GROUP_TILE = 2
// - C = 64
// - F16 KV cache
// - Q rows are already packed, normed, and RoPE'd
// - keys 0..base_pos+q_row are causally visible for each packed query row
//
// Grid: (n_kv_heads, n_q_tiles * (GROUP_TOTAL/GROUP_TILE), n_partitions)
// where n_q_tiles = ceil(n_rows / QT).
//
// Partial output shape:
//   o_partial  [n_rows, n_kv_heads, NWG, GROUP_TOTAL, DV]
//   ml_partial [n_rows, n_kv_heads, NWG, GROUP_TOTAL, 2]
//
// Reduce kernel grid: (n_q_heads, n_rows, 1)
// ============================================================================

struct attn_v4_prefill_args {
    uint  n_rows;
    uint  n_q_heads;
    uint  n_kv_heads;
    uint  head_dim;
    uint  n_pos;
    uint  kv_stride;
    uint  n_partitions;
    uint  rows_per_partition;
    uint  base_pos;
    float scale;
};

template <ushort GROUP_TOTAL, ushort GROUP_TILE, ushort QT>
inline void attn_v4_prefill_main_subgroup_c64_body(
        constant attn_v4_prefill_args & args,
        device const float    * q,
        device const half     * k_cache,
        device const half     * v_cache,
        device       float    * o_partial,
        device       float    * ml_partial,
        threadgroup  half     * sq,
        threadgroup  float    * ss,
        uint3  tgpig,
        ushort tiisg) {
    constexpr ushort C = 64;

    const uint kvh = tgpig.x;
    const uint subgroup_idx = tgpig.y % (GROUP_TOTAL / GROUP_TILE);
    const uint q_tile = tgpig.y / (GROUP_TOTAL / GROUP_TILE);
    const uint iwg = tgpig.z;
    if (kvh >= args.n_kv_heads || iwg >= args.n_partitions) return;

    const ushort g_base = (ushort)(subgroup_idx * GROUP_TILE);
    const uint row_base = q_tile * QT;
    if (row_base >= args.n_rows) return;

    const uint p_start = iwg * args.rows_per_partition;
    const uint p_end_raw = p_start + args.rows_per_partition;
    const uint p_end = p_end_raw < args.n_pos ? p_end_raw : args.n_pos;

    threadgroup half4 * sq4 = (threadgroup half4 *)sq;
    for (ushort qr = 0; qr < QT; ++qr) {
        const uint row = row_base + qr;
        const bool row_active = row < args.n_rows;
        device const float4 * q4_base = row_active
            ? (device const float4 *)(q + ((ulong)row * args.n_q_heads + (ulong)kvh * GROUP_TOTAL + (ulong)g_base) * DK)
            : nullptr;
        for (ushort g = 0; g < GROUP_TILE; ++g) {
            for (ushort ii = 0; ii < DK4_PER_LANE; ++ii) {
                const ushort idx = (qr * GROUP_TILE + g) * DK4 + ii * NW + tiisg;
                float4 qv = row_active ? q4_base[g * DK4 + ii * NW + tiisg] : float4(0.0f);
                sq4[idx] = half4(qv * args.scale);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float  m_state[QT][GROUP_TILE];
    float  l_state[QT][GROUP_TILE];
    float4 o_acc  [QT][GROUP_TILE][DV4_PER_LANE];
    for (ushort qr = 0; qr < QT; ++qr) {
        for (ushort g = 0; g < GROUP_TILE; ++g) {
            m_state[qr][g] = -INFINITY;
            l_state[qr][g] = 0.0f;
            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                o_acc[qr][g][ii] = float4(0.0f);
            }
        }
    }

    if (p_start >= p_end) {
        for (ushort qr = 0; qr < QT; ++qr) {
            const uint row = row_base + qr;
            if (row >= args.n_rows) continue;
            device float4 * o_out_base = (device float4 *)(
                o_partial + ((((ulong)row * args.n_kv_heads + kvh) * args.n_partitions + iwg) * GROUP_TOTAL + g_base) * DV
            );
            for (ushort g = 0; g < GROUP_TILE; ++g) {
                for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                    o_out_base[g * DV4 + ii * NW + tiisg] = float4(0.0f);
                }
            }
            if (tiisg == 0) {
                device float * ml_base = ml_partial
                    + ((((ulong)row * args.n_kv_heads + kvh) * args.n_partitions + iwg) * GROUP_TOTAL + g_base) * 2;
                for (ushort g = 0; g < GROUP_TILE; ++g) {
                    ml_base[g * 2 + 0] = -INFINITY;
                    ml_base[g * 2 + 1] = 0.0f;
                }
            }
        }
        return;
    }

    for (uint tile_start = p_start; tile_start < p_end; tile_start += C) {
        const uint tile_end_raw = tile_start + C;
        const uint tile_end = tile_end_raw < p_end ? tile_end_raw : p_end;
        const ushort tile_count = (ushort)(tile_end - tile_start);

        for (ushort cc = 0; cc < C; ++cc) {
            float partial[QT][GROUP_TILE];
            float qk_sum[QT][GROUP_TILE];
            for (ushort qr = 0; qr < QT; ++qr) {
                for (ushort g = 0; g < GROUP_TILE; ++g) {
                    partial[qr][g] = 0.0f;
                }
            }

            const uint k_pos = tile_start + cc;
            if (cc < tile_count) {
                device const half4 * pk4 = (device const half4 *)(
                    k_cache + (ulong)k_pos * args.kv_stride + (ulong)kvh * DK
                );
                for (ushort ii = 0; ii < DK4_PER_LANE; ++ii) {
                    const half4 k_chunk = pk4[ii * NW + tiisg];
                    const float4 k_f32 = float4(k_chunk);
                    for (ushort qr = 0; qr < QT; ++qr) {
                        for (ushort g = 0; g < GROUP_TILE; ++g) {
                            const ushort idx = (qr * GROUP_TILE + g) * DK4 + ii * NW + tiisg;
                            const float4 q_f32 = float4(sq4[idx]);
                            partial[qr][g] += dot(k_f32, q_f32);
                        }
                    }
                }
            }

            for (ushort qr = 0; qr < QT; ++qr) {
                for (ushort g = 0; g < GROUP_TILE; ++g) {
                    qk_sum[qr][g] = simd_sum(partial[qr][g]);
                }
            }

            if (tiisg == 0) {
                for (ushort qr = 0; qr < QT; ++qr) {
                    const uint row = row_base + qr;
                    const bool row_active = row < args.n_rows;
                    const uint q_pos = args.base_pos + row;
                    for (ushort g = 0; g < GROUP_TILE; ++g) {
                        const bool allowed = row_active && cc < tile_count && k_pos <= q_pos;
                        ss[(qr * GROUP_TILE + g) * C + cc] = allowed ? qk_sum[qr][g] : -INFINITY;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort qr = 0; qr < QT; ++qr) {
            for (ushort g = 0; g < GROUP_TILE; ++g) {
                float scores[(C + NW - 1) / NW];
                float weights[(C + NW - 1) / NW];
                float per_lane_max = -INFINITY;
                float per_lane_sum = 0.0f;

                for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                    const ushort col = k * NW + tiisg;
                    scores[k] = (col < C) ? ss[(qr * GROUP_TILE + g) * C + col] : -INFINITY;
                    per_lane_max = max(per_lane_max, scores[k]);
                }

                const float tile_max = simd_max(per_lane_max);
                const float new_m = max(m_state[qr][g], tile_max);
                const float factor = (m_state[qr][g] == -INFINITY) ? 0.0f
                                                                     : exp2(m_state[qr][g] - new_m);

                for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                    weights[k] = (scores[k] == -INFINITY) ? 0.0f : exp2(scores[k] - new_m);
                    per_lane_sum += weights[k];
                }
                const float tile_l = simd_sum(per_lane_sum);

                for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                    o_acc[qr][g][ii] *= factor;
                }
                for (ushort k = 0; k < (C + NW - 1) / NW; ++k) {
                    const ushort col = k * NW + tiisg;
                    if (col < C) {
                        ss[(qr * GROUP_TILE + g) * C + col] = weights[k];
                    }
                }

                l_state[qr][g] = l_state[qr][g] * factor + tile_l;
                m_state[qr][g] = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort cc = 0; cc < tile_count; ++cc) {
            device const half4 * pv4 = (device const half4 *)(
                v_cache + (ulong)(tile_start + cc) * args.kv_stride + (ulong)kvh * DV
            );
            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                const half4 v_chunk = pv4[ii * NW + tiisg];
                const float4 v_f32 = float4(v_chunk);
                for (ushort qr = 0; qr < QT; ++qr) {
                    for (ushort g = 0; g < GROUP_TILE; ++g) {
                        const float w = ss[(qr * GROUP_TILE + g) * C + cc];
                        o_acc[qr][g][ii] += w * v_f32;
                    }
                }
            }
        }
    }

    for (ushort qr = 0; qr < QT; ++qr) {
        const uint row = row_base + qr;
        if (row >= args.n_rows) continue;
        device float4 * o_out_base = (device float4 *)(
            o_partial + ((((ulong)row * args.n_kv_heads + kvh) * args.n_partitions + iwg) * GROUP_TOTAL + g_base) * DV
        );
        for (ushort g = 0; g < GROUP_TILE; ++g) {
            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                o_out_base[g * DV4 + ii * NW + tiisg] = o_acc[qr][g][ii];
            }
        }
        if (tiisg == 0) {
            device float * ml_base = ml_partial
                + ((((ulong)row * args.n_kv_heads + kvh) * args.n_partitions + iwg) * GROUP_TOTAL + g_base) * 2;
            for (ushort g = 0; g < GROUP_TILE; ++g) {
                ml_base[g * 2 + 0] = m_state[qr][g];
                ml_base[g * 2 + 1] = l_state[qr][g];
            }
        }
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_attn_prefill_v4_g8_t2_q2_c64_f32(
        constant attn_v4_prefill_args & args [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_prefill_main_subgroup_c64_body<8, 2, 2>(args, q, k_cache, v_cache, o_partial, ml_partial,
                                                    sq, ss, tgpig, tiisg);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_attn_prefill_v4_g8_t2_q4_c64_f32(
        constant attn_v4_prefill_args & args [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_prefill_main_subgroup_c64_body<8, 2, 4>(args, q, k_cache, v_cache, o_partial, ml_partial,
                                                    sq, ss, tgpig, tiisg);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_attn_prefill_v4_g16_t4_q2_c64_f32(
        constant attn_v4_prefill_args & args [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_prefill_main_subgroup_c64_body<16, 4, 2>(args, q, k_cache, v_cache, o_partial, ml_partial,
                                                     sq, ss, tgpig, tiisg);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_attn_prefill_v4_g16_t4_q4_c64_f32(
        constant attn_v4_prefill_args & args [[buffer(0)]],
        device const float    * q          [[buffer(1)]],
        device const half     * k_cache    [[buffer(2)]],
        device const half     * v_cache    [[buffer(3)]],
        device       float    * o_partial  [[buffer(4)]],
        device       float    * ml_partial [[buffer(5)]],
        threadgroup  half     * sq         [[threadgroup(0)]],
        threadgroup  float    * ss         [[threadgroup(1)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_prefill_main_subgroup_c64_body<16, 4, 4>(args, q, k_cache, v_cache, o_partial, ml_partial,
                                                     sq, ss, tgpig, tiisg);
}

// ============================================================================
// Experimental non-flash matrix-attention sidecar.
//
// This mirrors llama.cpp's default non-flash graph shape: KQ matmul, rowwise
// softmax, then KQV matmul. It is intentionally narrow: head_dim=256,
// F16 KV cache, and validated group shapes only.
// ============================================================================

struct attn_matrix_args {
    uint  n_rows;
    uint  n_pos;
    uint  base_pos;
    uint  kv_stride;
    uint  vt_stride;
    uint  n_q_heads;
    uint  n_kv_heads;
    uint  group;
    uint  head_dim;
    float scale;
    uint  causal_skip;
};

constant constexpr int AM_NR0            = 64;
constant constexpr int AM_NR1            = 32;
constant constexpr int AM_NK             = 32;
constant constexpr int AM_NL0            = AM_NK / 16;
constant constexpr int AM_NL1            = AM_NK / 8;

kernel void kernel_attn_matrix_transpose_v_f16(
        constant attn_matrix_args & args [[buffer(0)]],
        device const half * v_cache [[buffer(1)]],
        device       half * v_t     [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.n_kv_heads * args.head_dim * args.n_rows;
    if (tid >= total) return;
    const uint pos_rel = tid % args.n_rows;
    const uint pos = args.base_pos + pos_rel;
    const uint tmp = tid / args.n_rows;
    const uint d = tmp % args.head_dim;
    const uint kvh = tmp / args.head_dim;
    v_t[(ulong)tmp * args.vt_stride + pos] =
        v_cache[(ulong)pos * args.kv_stride + (ulong)kvh * args.head_dim + d];
}

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_attn_matrix_kq_f32(
        constant attn_matrix_args & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const half  * k_cache [[buffer(2)]],
        device       float * scores  [[buffer(3)]],
        threadgroup  uchar * shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const uint kvh = tgpig.z;
    const int M = (int)args.n_pos;
    const int N = (int)(args.n_rows * args.group);
    const int K = (int)args.head_dim;
    const int r0 = (int)tgpig.y * AM_NR0;
    const int r1 = (int)tgpig.x * AM_NR1;
    const short nr0 = (M - r0 < AM_NR0) ? (short)(M - r0) : AM_NR0;
    const short nr1 = (N - r1 < AM_NR1) ? (short)(N - r1) : AM_NR1;
    const uint local_q_last = (uint)(r1 + nr1 - 1);
    const uint row_last = local_q_last / args.group;
    const uint max_visible = min(args.n_pos, args.base_pos + row_last + 1);
    if (args.causal_skip != 0u && (uint)r0 >= max_visible) return;
    const short lr0 = ((short)tiitg / AM_NL0) < nr0 ? ((short)tiitg / AM_NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg / AM_NL1) < nr1 ? ((short)tiitg / AM_NL1) : nr1 - 1;
    const short il0 = tiitg % AM_NL0;
    const short iy = 8 * (tiitg % AM_NL1);
    const uint pos = (uint)(r0 + lr0);
    const uint local_q_thread = (uint)(r1 + lr1);
    const uint safe_local_q = min(local_q_thread, (uint)(N - 1));
    const uint row_thread = safe_local_q / args.group;
    const uint g_thread = safe_local_q - row_thread * args.group;
    device const half * k_ptr =
        k_cache + (ulong)pos * args.kv_stride + (ulong)kvh * args.head_dim;
    device const float * q_base =
        q + ((ulong)row_thread * args.n_q_heads + (ulong)kvh * args.group + g_thread) * args.head_dim;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += AM_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / AM_NL0) / 8;
            const short lx = (tiitg / AM_NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            const uint kk = loop_k + 16 * il0 + i;
            sa[64 * ib + 8 * ly + lx] = (pos < args.n_pos && kk < K)
                ? k_ptr[kk]
                : (half)0.0f;
        }

        {
            const short sx = tiitg % AM_NL1;
            const short sy = (tiitg / AM_NL1) / 8;
            const short ly = (tiitg / AM_NL1) % 8;
            const short ib = 4 * sx + sy;
            const uint kk = loop_k + iy;
            threadgroup half * dst = sb + 64 * ib + 8 * ly;
            if (local_q_thread < (uint)N && kk + 7 < (uint)K) {
                device const float * q_ptr = q_base + kk;
                *(threadgroup half2x4 *)dst = (half2x4)(*((device const float2x4 *)q_ptr));
            } else {
                for (short i = 0; i < 8; ++i) {
                    const uint kki = kk + i;
                    dst[i] = (local_q_thread < (uint)N && kki < (uint)K)
                        ? half(q_base[kki])
                        : (half)0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < AM_NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    const ulong batch = (ulong)kvh * (ulong)N * (ulong)M;
    if (r0 + AM_NR0 <= M && r1 + AM_NR1 <= N) {
        device float * C = scores + batch + (r0 + 32 * (sgitg & 1))
                         + (r1 + 16 * (sgitg >> 1)) * M;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * M * (i / 4), M, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp = ((threadgroup float *)shmem)
            + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * AM_NR0;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * AM_NR0 * (i / 4), AM_NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += AM_NR1) {
                device float * Dst = scores + batch + r0 + (r1 + j) * M;
                threadgroup float * Src = temp + j * AM_NR0;
                for (int i = 0; i < nr0; ++i) {
                    Dst[i] = Src[i];
                }
            }
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_attn_matrix_kq_f32_full_tiles(
        constant attn_matrix_args & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const half  * k_cache [[buffer(2)]],
        device       float * scores  [[buffer(3)]],
        threadgroup  uchar * shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const uint kvh = tgpig.z;
    const int M = (int)args.n_pos;
    const int N = (int)(args.n_rows * args.group);
    const int K = (int)args.head_dim;
    const int r0 = (int)tgpig.y * AM_NR0;
    const int r1 = (int)tgpig.x * AM_NR1;
    const short lr0 = ((short)tiitg / AM_NL0);
    const short lr1 = ((short)tiitg / AM_NL1);
    const uint local_q_last = (uint)(r1 + AM_NR1 - 1);
    const uint row_last = local_q_last / args.group;
    const uint max_visible = min(args.n_pos, args.base_pos + row_last + 1);
    if (args.causal_skip != 0u && (uint)r0 >= max_visible) return;
    const short il0 = tiitg % AM_NL0;
    const short iy = 8 * (tiitg % AM_NL1);
    const uint pos = (uint)(r0 + lr0);
    const uint local_q_thread = (uint)(r1 + lr1);
    const uint row_thread = local_q_thread / args.group;
    const uint g_thread = local_q_thread - row_thread * args.group;
    device const half * k_ptr =
        k_cache + (ulong)pos * args.kv_stride + (ulong)kvh * args.head_dim;
    device const float * q_base =
        q + ((ulong)row_thread * args.n_q_heads + (ulong)kvh * args.group + g_thread) * args.head_dim;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += AM_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / AM_NL0) / 8;
            const short lx = (tiitg / AM_NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            const uint kk = loop_k + 16 * il0 + i;
            sa[64 * ib + 8 * ly + lx] = k_ptr[kk];
        }

        {
            const short sx = tiitg % AM_NL1;
            const short sy = (tiitg / AM_NL1) / 8;
            const short ly = (tiitg / AM_NL1) % 8;
            const short ib = 4 * sx + sy;
            const uint kk = loop_k + iy;
            threadgroup half * dst = sb + 64 * ib + 8 * ly;
            device const float * q_ptr = q_base + kk;
            *(threadgroup half2x4 *)dst = (half2x4)(*((device const float2x4 *)q_ptr));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < AM_NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    const ulong batch = (ulong)kvh * (ulong)N * (ulong)M;
    device float * C = scores + batch + (r0 + 32 * (sgitg & 1))
                     + (r1 + 16 * (sgitg >> 1)) * M;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * M * (i / 4), M, 0, false);
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_attn_matrix_softmax_f32(
        constant attn_matrix_args & args [[buffer(0)]],
        device float * scores [[buffer(1)]],
        threadgroup float * sh [[threadgroup(0)]],
        uint qid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    const uint n_local_q = args.n_rows * args.group;
    const uint n_query = args.n_kv_heads * n_local_q;
    if (qid >= n_query) return;
    const uint local_q = qid % n_local_q;
    const uint row = local_q / args.group;
    const uint visible = min(args.n_pos, args.base_pos + row + 1);
    device float * s = scores + (ulong)qid * args.n_pos;

    float local_max = -INFINITY;
    for (uint p = tid; p < visible; p += ntg) {
        local_max = max(local_max, s[p] * args.scale);
    }
    local_max = simd_max(local_max);
    if (tiisg == 0) sh[sgitg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local_max = (tiisg < (ntg + 31) / 32) ? sh[tiisg] : -INFINITY;
    const float max_all = simd_max(local_max);

    float local_sum = 0.0f;
    for (uint p = tid; p < visible; p += ntg) {
        local_sum += exp2(s[p] * args.scale - max_all);
    }
    local_sum = simd_sum(local_sum);
    if (tiisg == 0) sh[sgitg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local_sum = (tiisg < (ntg + 31) / 32) ? sh[tiisg] : 0.0f;
    const float sum_all = simd_sum(local_sum);
    const float inv_sum = sum_all > 0.0f ? 1.0f / sum_all : 0.0f;

    for (uint p = tid; p < args.n_pos; p += ntg) {
        s[p] = (p < visible) ? exp2(s[p] * args.scale - max_all) * inv_sum : 0.0f;
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_attn_matrix_kqv_f32(
        constant attn_matrix_args & args [[buffer(0)]],
        device const float * probs [[buffer(1)]],
        device const half  * v_t   [[buffer(2)]],
        device       float * out   [[buffer(3)]],
        threadgroup  uchar * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const uint kvh = tgpig.z;
    const int M = (int)args.head_dim;
    const int N = (int)(args.n_rows * args.group);
    const int K = (int)args.n_pos;
    const int r0 = (int)tgpig.y * AM_NR0;
    const int r1 = (int)tgpig.x * AM_NR1;
    const short nr0 = (M - r0 < AM_NR0) ? (short)(M - r0) : AM_NR0;
    const short nr1 = (N - r1 < AM_NR1) ? (short)(N - r1) : AM_NR1;
    const uint local_q_last = (uint)(r1 + nr1 - 1);
    const uint row_last = local_q_last / args.group;
    const uint max_visible = min(args.n_pos, args.base_pos + row_last + 1);
    const short lr0 = ((short)tiitg / AM_NL0) < nr0 ? ((short)tiitg / AM_NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg / AM_NL1) < nr1 ? ((short)tiitg / AM_NL1) : nr1 - 1;
    const short il0 = tiitg % AM_NL0;
    const short iy = 8 * (tiitg % AM_NL1);
    const uint d_thread = (uint)(r0 + lr0);
    const uint safe_d = min(d_thread, args.head_dim - 1u);
    const uint local_q_thread = (uint)(r1 + lr1);
    const uint safe_local_q = min(local_q_thread, (uint)(N - 1));
    device const half * vt_base =
        v_t + ((ulong)kvh * args.head_dim + safe_d) * args.vt_stride;
    device const float * probs_base =
        probs + ((ulong)kvh * (ulong)N + (ulong)safe_local_q) * args.n_pos;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.n_pos; loop_k += AM_NK) {
        if (args.causal_skip != 0u && loop_k >= max_visible) continue;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / AM_NL0) / 8;
            const short lx = (tiitg / AM_NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            const uint kk = loop_k + 16 * il0 + i;
            sa[64 * ib + 8 * ly + lx] = (d_thread < args.head_dim && kk < args.n_pos)
                ? vt_base[kk]
                : (half)0.0f;
        }

        {
            const short sx = tiitg % AM_NL1;
            const short sy = (tiitg / AM_NL1) / 8;
            const short ly = (tiitg / AM_NL1) % 8;
            const short ib = 4 * sx + sy;
            const uint kk = loop_k + iy;
            threadgroup half * dst = sb + 64 * ib + 8 * ly;
            const bool aligned_scores = (args.n_pos & 7u) == 0u;
            if (aligned_scores && local_q_thread < (uint)N && kk + 7 < args.n_pos) {
                device const float * p_ptr = probs_base + kk;
                *(threadgroup half2x4 *)dst = (half2x4)(*((device const float2x4 *)p_ptr));
            } else {
                for (short i = 0; i < 8; ++i) {
                    const uint kki = kk + i;
                    dst[i] = (local_q_thread < (uint)N && kki < args.n_pos)
                        ? half(probs_base[kki])
                        : (half)0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < AM_NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp = ((threadgroup float *)shmem)
        + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * AM_NR0;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * AM_NR0 * (i / 4), AM_NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += AM_NR1) {
            const uint local_q = (uint)(r1 + j);
            const uint row = local_q / args.group;
            const uint g = local_q % args.group;
            device float * Dst = out + ((ulong)row * args.n_q_heads + (ulong)kvh * args.group + g) * args.head_dim + r0;
            threadgroup float * Src = temp + j * AM_NR0;
            for (int i = 0; i < nr0; ++i) {
                Dst[i] = Src[i];
            }
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_attn_matrix_kqv_f32_full_tiles(
        constant attn_matrix_args & args [[buffer(0)]],
        device const float * probs [[buffer(1)]],
        device const half  * v_t   [[buffer(2)]],
        device       float * out   [[buffer(3)]],
        threadgroup  uchar * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const uint kvh = tgpig.z;
    const int N = (int)(args.n_rows * args.group);
    const int r0 = (int)tgpig.y * AM_NR0;
    const int r1 = (int)tgpig.x * AM_NR1;
    const uint local_q_last = (uint)(r1 + AM_NR1 - 1);
    const uint row_last = local_q_last / args.group;
    const uint max_visible = min(args.n_pos, args.base_pos + row_last + 1);
    const short lr0 = ((short)tiitg / AM_NL0);
    const short lr1 = ((short)tiitg / AM_NL1);
    const short il0 = tiitg % AM_NL0;
    const short iy = 8 * (tiitg % AM_NL1);
    const uint d_thread = (uint)(r0 + lr0);
    const uint local_q_thread = (uint)(r1 + lr1);
    device const half * vt_base =
        v_t + ((ulong)kvh * args.head_dim + d_thread) * args.vt_stride;
    device const float * probs_base =
        probs + ((ulong)kvh * (ulong)N + (ulong)local_q_thread) * args.n_pos;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.n_pos; loop_k += AM_NK) {
        if (args.causal_skip != 0u && loop_k >= max_visible) continue;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / AM_NL0) / 8;
            const short lx = (tiitg / AM_NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            const uint kk = loop_k + 16 * il0 + i;
            sa[64 * ib + 8 * ly + lx] = vt_base[kk];
        }

        {
            const short sx = tiitg % AM_NL1;
            const short sy = (tiitg / AM_NL1) / 8;
            const short ly = (tiitg / AM_NL1) % 8;
            const short ib = 4 * sx + sy;
            const uint kk = loop_k + iy;
            threadgroup half * dst = sb + 64 * ib + 8 * ly;
            device const float * p_ptr = probs_base + kk;
            *(threadgroup half2x4 *)dst = (half2x4)(*((device const float2x4 *)p_ptr));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < AM_NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp = ((threadgroup float *)shmem)
        + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * AM_NR0;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * AM_NR0 * (i / 4), AM_NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        for (int j = tiitg; j < AM_NR1; j += AM_NR1) {
            const uint local_q = (uint)(r1 + j);
            const uint row = local_q / args.group;
            const uint g = local_q % args.group;
            device float * Dst = out + ((ulong)row * args.n_q_heads + (ulong)kvh * args.group + g) * args.head_dim + r0;
            threadgroup float * Src = temp + j * AM_NR0;
            for (int i = 0; i < AM_NR0; ++i) {
                Dst[i] = Src[i];
            }
        }
    }
}

struct attn_v4_prefill_reduce_args {
    uint n_rows;
    uint n_q_heads;
    uint n_kv_heads;
    uint head_dim;
    uint n_partitions;
};

template <ushort GROUP>
inline void attn_v4_prefill_reduce_rows_body(
        constant attn_v4_prefill_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint qh = tgpig.x;
    const uint row = tgpig.y;
    if (qh >= args.n_q_heads || row >= args.n_rows) return;
    const uint kvh = qh / GROUP;
    const uint g = qh % GROUP;
    const uint nwg = args.n_partitions;

    for (ushort pass = 0; pass < 2; ++pass) {
        const uint part = tiisg + pass * 32;
        if (part < nwg) {
            device const float * ml_base = ml_partial
                + ((((ulong)row * args.n_kv_heads + kvh) * nwg + part) * GROUP + g) * 2;
            sh_m[part] = ml_base[0];
            sh_l[part] = ml_base[1];
        }
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
            const ulong off = ((((ulong)row * args.n_kv_heads + kvh) * nwg + i) * GROUP + g) * DV
                            + (ulong)(ii * NW + tiisg) * 4;
            const device float4 * src4 = (device const float4 *)(o_partial + off);
            acc += (*src4) * ef;
        }
        device float4 * out4 = (device float4 *)
            (out + ((ulong)row * args.n_q_heads + qh) * DV + (ulong)(ii * NW + tiisg) * 4);
        *out4 = acc * inv_l;
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_attn_prefill_v4_reduce_rows_g8_f32(
        constant attn_v4_prefill_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_prefill_reduce_rows_body<8>(args, o_partial, ml_partial, out, sh_m, sh_l, sh_ef, tgpig, tiisg);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_attn_prefill_v4_reduce_rows_g16_f32(
        constant attn_v4_prefill_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_prefill_reduce_rows_body<16>(args, o_partial, ml_partial, out, sh_m, sh_l, sh_ef, tgpig, tiisg);
}

// ============================================================================
// Reduce kernel: combines NWG partials per Q head, normalizes by global l.
//
// Grid: (n_q_heads, 1, 1).  Threadgroup: 32 lanes (one simdgroup).
//
// The reduce path uses threadgroup scratch sized by the host for NWG
// per-partition factors, so the main-pass split-K count can go beyond one
// simdgroup when long-context occupancy needs it.

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

    for (uint part = tiisg; part < nwg; part += 32) {
        float m_part = -INFINITY;
        float l_part = 0.0f;
        device const float * ml_base = ml_partial
            + ((ulong)kvh * nwg + part) * GROUP * 2;
        m_part = ml_base[g * 2 + 0];
        l_part = ml_base[g * 2 + 1];
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

template <ushort GROUP>
inline void attn_v4_reduce_h2_body(
        constant attn_v4_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
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
    const uint half_idx = tgpig.y;

    for (uint part = tiisg; part < nwg; part += 32) {
        device const float * ml_base = ml_partial
            + ((ulong)kvh * nwg + part) * GROUP * 2;
        sh_m[part] = ml_base[g * 2 + 0];
        sh_l[part] = ml_base[g * 2 + 1];
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
    const uint d = half_idx * 128 + uint(tiisg) * 4;

    float4 acc = float4(0.0f);
    for (uint i = 0; i < nwg; ++i) {
        const float ef = sh_ef[i];
        const ulong off = ((ulong)kvh * nwg + i) * GROUP * DV
                        + (ulong)g * DV
                        + (ulong)d;
        const device float4 * src4 = (device const float4 *)(o_partial + off);
        acc += (*src4) * ef;
    }
    device float4 * out4 = (device float4 *)(out + (ulong)qh * DV + (ulong)d);
    *out4 = acc * inv_l;
}

[[max_total_threads_per_threadgroup(32)]]
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

[[max_total_threads_per_threadgroup(32)]]
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

[[max_total_threads_per_threadgroup(32)]]
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

[[max_total_threads_per_threadgroup(32)]]
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

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_attn_decode_v4_reduce_h2_g4_f32(
        constant attn_v4_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_reduce_h2_body<4>(args, o_partial, ml_partial, out, sh_m, sh_l, sh_ef, tgpig, tiisg);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_attn_decode_v4_reduce_h2_g6_f32(
        constant attn_v4_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_reduce_h2_body<6>(args, o_partial, ml_partial, out, sh_m, sh_l, sh_ef, tgpig, tiisg);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_attn_decode_v4_reduce_h2_g8_f32(
        constant attn_v4_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_reduce_h2_body<8>(args, o_partial, ml_partial, out, sh_m, sh_l, sh_ef, tgpig, tiisg);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_attn_decode_v4_reduce_h2_g16_f32(
        constant attn_v4_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]],
        device const float * ml_partial  [[buffer(2)]],
        device       float * out         [[buffer(3)]],
        threadgroup  float * sh_m        [[threadgroup(0)]],
        threadgroup  float * sh_l        [[threadgroup(1)]],
        threadgroup  float * sh_ef       [[threadgroup(2)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    attn_v4_reduce_h2_body<16>(args, o_partial, ml_partial, out, sh_m, sh_l, sh_ef, tgpig, tiisg);
}
