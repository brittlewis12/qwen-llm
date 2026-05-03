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
// Algorithm (per TG owning kv_head=kvh, partition=iwg):
//   1. Load Q for all 6 group siblings into shmem (prescaled by scale*log2(e)).
//   2. Online softmax over [p_start, p_end) tiles of C=32 KV rows:
//      Phase A — compute QK[g][cc] for all (g, cc), stash in ss[].
//                K row read ONCE, dotted with all 6 Q vectors.
//      Phase B — per-group online softmax update (m, l) using exp2.
//      Phase C — V-aggregate; V row read ONCE, accumulated into 6 O regs.
//   3. Write per-partition (m, l, O_unnormalized) partials.
// A separate reduce kernel combines NWG partials per Q head and divides.
//
// Threadgroup: 32 lanes (single simdgroup). Grid: (n_kv_heads, 1, NWG).
//
// CPU oracle: same as forward.rs attn — math is bit-equivalent up to fp32
// reorder noise (< 1e-4 typical for our shapes).

#include <metal_stdlib>
using namespace metal;

constant constexpr ushort NW = 32;        // simdgroup width
constant constexpr ushort C  = 32;        // KV positions per inner tile (== NW)
constant constexpr ushort GROUP = 6;      // Q heads per KV head (n_q / n_kv)
constant constexpr ushort DK = 256;       // K head_dim
constant constexpr ushort DV = 256;       // V head_dim
constant constexpr ushort DK4 = DK / 4;
constant constexpr ushort DV4 = DV / 4;
constant constexpr ushort DK4_PER_LANE = DK4 / NW;   // = 2 for DK=256
constant constexpr ushort DV4_PER_LANE = DV4 / NW;   // = 2 for DV=256

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

kernel void kernel_attn_decode_v4_f32(
        constant attn_v4_args & args      [[buffer(0)]],
        device const float    * q          [[buffer(1)]], // [n_q_heads, head_dim] F32
        device const half     * k_cache    [[buffer(2)]], // [capacity, n_kv_heads, head_dim] F16
        device const half     * v_cache    [[buffer(3)]], // [capacity, n_kv_heads, head_dim] F16
        device       float    * o_partial  [[buffer(4)]], // [n_kv_heads, NWG, GROUP, head_dim] F32
        device       float    * ml_partial [[buffer(5)]], // [n_kv_heads, NWG, GROUP, 2] F32
        threadgroup  half     * sq         [[threadgroup(0)]], // [GROUP * head_dim]
        threadgroup  float    * ss         [[threadgroup(1)]], // [GROUP * C]
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint kvh = tgpig.x;
    const uint iwg = tgpig.z;
    if (kvh >= args.n_kv_heads || iwg >= args.n_partitions) return;

    // Partition row range.
    const uint p_start = iwg * args.rows_per_partition;
    const uint p_end_raw = p_start + args.rows_per_partition;
    const uint p_end = p_end_raw < args.n_pos ? p_end_raw : args.n_pos;

    // ---- Load Q for the GROUP siblings (q_head = kvh*GROUP .. kvh*GROUP+GROUP-1) ----
    // Apply scale*log2(e) prescale so softmax can use exp2.
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
    float4 o_acc  [GROUP][DV4_PER_LANE];  // 6 × 2 = 12 float4 per lane = 48 floats
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
        //
        // Per cc (one K row), all 32 lanes cooperate on the dot product:
        //   lane t reads K[cc, dim_subset = {t, t+32}] (DK4_PER_LANE = 2 chunks)
        // For each Q head g, accumulate per-lane partial dot, then simd_sum.
        //
        // Register usage during this loop: GROUP scalars (one per g) for partials.
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
            // Reduce across the 32 lanes; broadcast result to all lanes.
            // Lane 0 writes ss for this cc.
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
        // Each lane reads its column tiisg's score from ss; simd_max/simd_sum
        // give tile_max and tile_l. Use exp2 (Apple GPU has hw fast path).
        // After update, ss[g*C + tiisg] holds the WEIGHT for V-aggregate.
        for (ushort g = 0; g < GROUP; ++g) {
            const float s = ss[g * C + tiisg];   // each lane reads one cc
            const float tile_max = simd_max(s);
            const float new_m = max(m_state[g], tile_max);
            // Numerical guard for first iter (m_state == -inf):
            const float factor = (m_state[g] == -INFINITY) ? 0.0f
                                                            : exp2(m_state[g] - new_m);
            const float w = (s == -INFINITY) ? 0.0f : exp2(s - new_m);
            const float tile_l = simd_sum(w);

            // Rescale O accumulator.
            for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
                o_acc[g][ii] *= factor;
            }
            // Write weight back to ss for V-aggregate (overwrites the score).
            ss[g * C + tiisg] = w;

            l_state[g] = l_state[g] * factor + tile_l;
            m_state[g] = new_m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ===== Phase C: V-aggregate =====
        //
        // For each cc in tile, all lanes cooperate on V row reads.
        // Lane t handles dim subset {t, t+32}; multiplies by per-group weight.
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
        // No barrier needed at end of Phase C; Phase A overwrites ss next iter.
    }

    // ---- Write per-partition partials ----
    // o_partial layout: [n_kv_heads, NWG, GROUP, head_dim] row-major
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

// ============================================================================
// Reduce kernel: combines NWG partials per Q head, normalizes by global l.
//
// Grid: (n_q_heads, 1, 1).  Threadgroup: 32 lanes (one simdgroup).
//
// Constraint v0: NWG <= 32 (uses simd_shuffle to broadcast per-partition
// exp_factor). NWG > 32 will require an iteration; deferred until needed.

struct attn_v4_reduce_args {
    uint n_q_heads;
    uint n_kv_heads;
    uint head_dim;          // expected = DV
    uint n_partitions;      // NWG
};

kernel void kernel_attn_decode_v4_reduce_f32(
        constant attn_v4_reduce_args & args [[buffer(0)]],
        device const float * o_partial   [[buffer(1)]], // [n_kv_heads, NWG, GROUP, head_dim]
        device const float * ml_partial  [[buffer(2)]], // [n_kv_heads, NWG, GROUP, 2]
        device       float * out         [[buffer(3)]], // [n_q_heads, head_dim]
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint qh = tgpig.x;
    if (qh >= args.n_q_heads) return;
    const uint kvh = qh / GROUP;
    const uint g = qh % GROUP;
    const uint nwg = args.n_partitions;

    // Load (m, l) for partition tiisg (or sentinel if out of range).
    float m_part = -INFINITY;
    float l_part = 0.0f;
    if (tiisg < nwg) {
        device const float * ml_base = ml_partial
            + ((ulong)kvh * nwg + tiisg) * GROUP * 2;
        m_part = ml_base[g * 2 + 0];
        l_part = ml_base[g * 2 + 1];
    }
    const float m_global = simd_max(m_part);
    // exp_factor[i] = exp2(m_part_i - m_global), with -inf safeguarded.
    const float exp_factor = (m_part == -INFINITY || m_global == -INFINITY)
                             ? 0.0f
                             : exp2(m_part - m_global);
    const float l_global = simd_sum(l_part * exp_factor);
    const float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;

    // For each lane's owned dim subset, reduce o_partial across partitions.
    // Lane t needs exp_factor[i] for all i — broadcast via simd_shuffle.
    for (ushort ii = 0; ii < DV4_PER_LANE; ++ii) {
        float4 acc = float4(0.0f);
        for (uint i = 0; i < nwg; ++i) {
            const float ef = simd_shuffle(exp_factor, (ushort)i);
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
