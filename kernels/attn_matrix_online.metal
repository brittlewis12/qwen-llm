// Two-pass online-softmax matrix prefill attention.
//
// Replaces the three-kernel non-flash matrix sidecar (KQ matmul -> rowwise
// softmax -> KQV matmul in kernels/attn_v4.metal) with two kernels that keep
// the sidecar's simdgroup-GEMM shapes but fold the softmax into their
// epilogue/staging:
//
//   1. kernel_attn_matrix_kq_online_f32[_full_tiles]
//      Same K^T.Q GEMM as kernel_attn_matrix_kq_f32, but the epilogue applies
//      the causal mask and a per-64-pos-tile online softmax: it stores
//      P~ = exp2(s*scale - m_tile) as F16 (half the bytes of the raw F32
//      scores) plus a tiny per-(query, tile) (m, l) sidecar.
//
//   2. kernel_attn_matrix_kqv_norm_f32[_full_tiles]
//      Same probs.V^T GEMM as kernel_attn_matrix_kqv_f32, but staging reads
//      F16 P~ and multiplies by c_t = exp2(m_t - m_glob) on the fly ((m, l)
//      pre-reduced per query column at kernel start), and the epilogue
//      divides by the global l. The separate softmax dispatch disappears.
//
// Net score-tensor traffic drops from 16 B/elem (KQ write 4 + softmax
// read/write 8 + KQV read 4) to 4 B/elem (KQ write 2 + KQV read 2), and the
// device score scratch halves (F16 vs F32).
//
// Numerics: the GEMMs are bit-identical to the sidecar's (same staging, same
// simdgroup MMA order). P~ is demoted to half exactly where the sidecar
// demotes normalized probs to half; the extra c_t scale runs in F32 during
// staging. m/l bookkeeping is F32.
//
// Falsified alternative (v0.439 microbench): a true single-kernel
// flash-attention body (llama.cpp kernel_flash_attn_ext work shape: Q=8/C=64/
// NSG=4, O in threadgroup memory, direct-device K/V simdgroup loads) is
// CORRECT but 0.80x the sidecar at production shapes, degrading to 0.39x at
// Q=16 (threadgroup-memory occupancy cliff) and 0.24x with register-resident
// O (spills). head_dim=256 makes the 1-pass design L2-bound on K/V
// re-streaming: 8..16-row query tiles re-read K/V 4..8x more than the
// sidecar's 32-column GEMM tiles. Do not reopen 1-pass without a >=32-row
// tile design that fits registers/threadgroup memory.

#include <metal_stdlib>
using namespace metal;

// Must match `AttnMatrixArgs` in crates/qwen-llm/src/metal.rs (and the
// attn_matrix_args struct in kernels/attn_v4.metal; duplicated because each
// .metal file is a separate translation unit).
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

// Tile constants, identical to the sidecar GEMMs in attn_v4.metal.
constant constexpr int AMO_NR0 = 64;
constant constexpr int AMO_NR1 = 32;
constant constexpr int AMO_NK  = 32;
constant constexpr int AMO_NL0 = AMO_NK / 16;
constant constexpr int AMO_NL1 = AMO_NK / 8;

// Score-tile spill row stride (floats). Padded from AMO_NR0=64 to 68 so the
// per-column epilogue (32 lanes reading 32 different columns) spreads across
// threadgroup-memory banks instead of serializing 32-wide on one bank
// (64-float rows put every column's element p on the same bank).
constant constexpr int AMO_TS = 68;

// Threadgroup memory for the KQ online kernels (bytes):
//   [0, 8704)    : GEMM staging (sa: K tile half, sb: Q tile half; 8 KiB),
//                  then reused as the F32 [AMO_NR1 q][AMO_TS] score tile for
//                  the online-softmax epilogue (32*68*4 = 8704).
//   [8704, 9728) : per-column reduction scratch (32 cols x 4 partials x
//                  2 floats).
#define AMO_KQ_TG_BYTES 9728

// Threadgroup memory for the KQV norm kernels: same 8 KiB staging as the
// sidecar KQV plus 32 floats of per-column inv_l scratch.
//   [0, 8192)    : GEMM staging / output spill (as sidecar).
//   [8192, 8320) : per-column inv_l (32 floats).
#define AMO_KQV_TG_BYTES 8320

// ---------------------------------------------------------------------------
// Shared epilogue: online softmax over the staged [nr0 pos x nr1 q] F32 score
// tile. Applies scale + causal mask, computes per-column (m_tile, l_tile),
// stores P~ = exp2(s*scale - m_tile) as F16 to `scores_h` ([kvh][local_q][pos]
// layout, pos minor) and (m, l) to `ml` ([kvh][local_q][tile][2]).
// ---------------------------------------------------------------------------
inline void attn_matrix_kq_online_epilogue(
        constant attn_matrix_args & args,
        threadgroup float * tile,   // [AMO_NR1 q][AMO_TS], pos minor idx: q*AMO_TS + pos
        threadgroup float * red,    // 32 cols x 4 partials x 2 floats
        device       half  * scores_h,
        device       float * ml,
        int r0,
        int r1,
        short nr0,
        short nr1,
        uint kvh,
        ushort tiitg) {
    const int N = (int)(args.n_rows * args.group);
    const int M = (int)args.n_pos;
    const uint n_tiles = ((uint)M + (uint)AMO_NR0 - 1) / (uint)AMO_NR0;
    const uint tile_idx = (uint)r0 / (uint)AMO_NR0;

    // 128 threads: 4 threads per column, 16 positions each (four float4s).
    const short col = tiitg % AMO_NR1;
    const short quarter = tiitg / AMO_NR1;
    const uint local_q = (uint)(r1 + col);
    const bool col_ok = col < nr1;
    const uint row = col_ok ? min(local_q, (uint)(N - 1)) / args.group : 0u;
    const uint visible = min(args.n_pos, args.base_pos + row + 1);
    const short p_base = quarter * 16;
    // Visible count within this thread's 16-position strip, clamped by the
    // tile edge (nr0) and the causal bound.
    const int vis_in_tile = (int)min((uint)nr0, visible > (uint)r0 ? visible - (uint)r0 : 0u);
    const short n_here = (short)clamp(vis_in_tile - (int)p_base, 0, 16);

    // Pass 1: masked, scaled max over this thread's strip.
    float m_part = -INFINITY;
    if (col_ok && n_here == 16) {
#pragma unroll
        for (short i4 = 0; i4 < 4; ++i4) {
            const float4 v = *((threadgroup const float4 *)(tile + col * AMO_TS + p_base) + i4);
            m_part = max(m_part, max(max(v.x, v.y), max(v.z, v.w)));
        }
        m_part *= args.scale;
    } else if (col_ok) {
        for (short i = 0; i < n_here; ++i) {
            m_part = max(m_part, tile[col * AMO_TS + p_base + i] * args.scale);
        }
    }
    red[(col * 4 + quarter) * 2 + 0] = m_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m_tile = max(
        max(red[(col * 4 + 0) * 2 + 0], red[(col * 4 + 1) * 2 + 0]),
        max(red[(col * 4 + 2) * 2 + 0], red[(col * 4 + 3) * 2 + 0]));

    // Pass 2: exp2, F16 store, partial sums. m_tile == -inf means the whole
    // column is masked in this tile; store zeros and l = 0.
    float l_part = 0.0f;
    device half * out_base = scores_h
        + (ulong)kvh * (ulong)N * (ulong)M + (ulong)min(local_q, (uint)(N - 1)) * (ulong)M
        + (ulong)(r0 + p_base);
    // The half4 store needs the score row stride (n_pos) 4-aligned; the
    // full-tiles variant always is, edge shapes fall back to scalar.
    if (col_ok && n_here == 16 && m_tile != -INFINITY && ((uint)M & 3u) == 0u) {
#pragma unroll
        for (short i4 = 0; i4 < 4; ++i4) {
            const float4 v = *((threadgroup const float4 *)(tile + col * AMO_TS + p_base) + i4);
            const float4 e = exp2(v * args.scale - m_tile);
            l_part += e.x + e.y + e.z + e.w;
            *((device half4 *)out_base + i4) = half4(e);
        }
    } else if (col_ok) {
        for (short i = 0; i < 16; ++i) {
            const short p = p_base + i;
            if (p >= nr0) {
                break;
            }
            float e = 0.0f;
            if (i < n_here && m_tile != -INFINITY) {
                e = exp2(tile[col * AMO_TS + p] * args.scale - m_tile);
            }
            l_part += e;
            out_base[i] = (half)e;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[(col * 4 + quarter) * 2 + 1] = l_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (quarter == 0 && col_ok) {
        const float l_tile = red[(col * 4 + 0) * 2 + 1] + red[(col * 4 + 1) * 2 + 1]
                           + red[(col * 4 + 2) * 2 + 1] + red[(col * 4 + 3) * 2 + 1];
        device float * ml_base = ml
            + (((ulong)kvh * (ulong)N + local_q) * (ulong)n_tiles + tile_idx) * 2;
        ml_base[0] = m_tile;
        ml_base[1] = l_tile;
    }
}

// ---------------------------------------------------------------------------
// KQ online: edge-safe variant. GEMM body identical to
// kernel_attn_matrix_kq_f32 (attn_v4.metal), epilogue replaced.
// ---------------------------------------------------------------------------
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_attn_matrix_kq_online_f32(
        constant attn_matrix_args & args [[buffer(0)]],
        device const float * q        [[buffer(1)]],
        device const half  * k_cache  [[buffer(2)]],
        device       half  * scores_h [[buffer(3)]],
        device       float * ml       [[buffer(4)]],
        threadgroup  uchar * shmem    [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const uint kvh = tgpig.z;
    const int M = (int)args.n_pos;
    const int N = (int)(args.n_rows * args.group);
    const int K = (int)args.head_dim;
    const int r0 = (int)tgpig.y * AMO_NR0;
    const int r1 = (int)tgpig.x * AMO_NR1;
    const short nr0 = (M - r0 < AMO_NR0) ? (short)(M - r0) : AMO_NR0;
    const short nr1 = (N - r1 < AMO_NR1) ? (short)(N - r1) : AMO_NR1;
    const uint local_q_last = (uint)(r1 + nr1 - 1);
    const uint row_last = local_q_last / args.group;
    const uint max_visible = min(args.n_pos, args.base_pos + row_last + 1);
    if (args.causal_skip != 0u && (uint)r0 >= max_visible) return;
    const short lr0 = ((short)tiitg / AMO_NL0) < nr0 ? ((short)tiitg / AMO_NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg / AMO_NL1) < nr1 ? ((short)tiitg / AMO_NL1) : nr1 - 1;
    const short il0 = tiitg % AMO_NL0;
    const short iy = 8 * (tiitg % AMO_NL1);
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

    for (uint loop_k = 0; loop_k < (uint)K; loop_k += AMO_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / AMO_NL0) / 8;
            const short lx = (tiitg / AMO_NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            const uint kk = loop_k + 16 * il0 + i;
            sa[64 * ib + 8 * ly + lx] = (pos < args.n_pos && kk < (uint)K)
                ? k_ptr[kk]
                : (half)0.0f;
        }

        {
            const short sx = tiitg % AMO_NL1;
            const short sy = (tiitg / AMO_NL1) / 8;
            const short ly = (tiitg / AMO_NL1) % 8;
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
        for (short ik = 0; ik < AMO_NK / 8; ++ik) {
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

    // Online-softmax epilogue: spill the score tile to threadgroup memory
    // ([col q][pos] with padded row stride AMO_TS), then transform + store F16.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * tile = ((threadgroup float *)shmem);
    {
        threadgroup float * temp = tile + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * AMO_TS;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * AMO_TS * (i / 4), AMO_TS, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_matrix_kq_online_epilogue(
        args, tile, (threadgroup float *)(shmem + 8704), scores_h, ml,
        r0, r1, nr0, nr1, kvh, tiitg);
}

// ---------------------------------------------------------------------------
// KQ online: full-tile variant (n_pos % 64 == 0, N % 32 == 0). GEMM body
// identical to kernel_attn_matrix_kq_f32_full_tiles.
// ---------------------------------------------------------------------------
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_attn_matrix_kq_online_f32_full_tiles(
        constant attn_matrix_args & args [[buffer(0)]],
        device const float * q        [[buffer(1)]],
        device const half  * k_cache  [[buffer(2)]],
        device       half  * scores_h [[buffer(3)]],
        device       float * ml       [[buffer(4)]],
        threadgroup  uchar * shmem    [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const uint kvh = tgpig.z;
    const int M = (int)args.n_pos;
    const int N = (int)(args.n_rows * args.group);
    const int K = (int)args.head_dim;
    const int r0 = (int)tgpig.y * AMO_NR0;
    const int r1 = (int)tgpig.x * AMO_NR1;
    const short lr0 = ((short)tiitg / AMO_NL0);
    const short lr1 = ((short)tiitg / AMO_NL1);
    const uint local_q_last = (uint)(r1 + AMO_NR1 - 1);
    const uint row_last = local_q_last / args.group;
    const uint max_visible = min(args.n_pos, args.base_pos + row_last + 1);
    if (args.causal_skip != 0u && (uint)r0 >= max_visible) return;
    const short il0 = tiitg % AMO_NL0;
    const short iy = 8 * (tiitg % AMO_NL1);
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

    for (uint loop_k = 0; loop_k < (uint)K; loop_k += AMO_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / AMO_NL0) / 8;
            const short lx = (tiitg / AMO_NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            const uint kk = loop_k + 16 * il0 + i;
            sa[64 * ib + 8 * ly + lx] = k_ptr[kk];
        }

        {
            const short sx = tiitg % AMO_NL1;
            const short sy = (tiitg / AMO_NL1) / 8;
            const short ly = (tiitg / AMO_NL1) % 8;
            const short ib = 4 * sx + sy;
            const uint kk = loop_k + iy;
            threadgroup half * dst = sb + 64 * ib + 8 * ly;
            device const float * q_ptr = q_base + kk;
            *(threadgroup half2x4 *)dst = (half2x4)(*((device const float2x4 *)q_ptr));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < AMO_NK / 8; ++ik) {
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
    threadgroup float * tile = ((threadgroup float *)shmem);
    {
        threadgroup float * temp = tile + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * AMO_TS;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * AMO_TS * (i / 4), AMO_TS, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_matrix_kq_online_epilogue(
        args, tile, (threadgroup float *)(shmem + 8704), scores_h, ml,
        r0, r1, AMO_NR0, AMO_NR1, kvh, tiitg);
}

// ---------------------------------------------------------------------------
// Shared prologue for the KQV norm kernels: per-column global (m, inv_l).
// Each of the tile's 32 query columns reduces its (m_t, l_t) list:
//   m_glob = max_t m_t;  L = sum_t l_t * exp2(m_t - m_glob)
// and stores 1/L (or 0 for fully-masked columns, which cannot occur for
// valid causal shapes) into `inv_l[col]`.
// ---------------------------------------------------------------------------
inline void attn_matrix_kqv_norm_prologue(
        constant attn_matrix_args & args,
        device const float * ml,
        threadgroup float * red,     // 32 cols x 4 partials x 2 floats
        threadgroup float * inv_l,   // 32 floats
        int r1,
        short nr1,
        uint kvh,
        ushort tiitg) {
    const int N = (int)(args.n_rows * args.group);
    const uint n_tiles = (args.n_pos + (uint)AMO_NR0 - 1) / (uint)AMO_NR0;

    const short col = tiitg % AMO_NR1;
    const short quarter = tiitg / AMO_NR1;
    const uint local_q = (uint)min((int)(r1 + col), N - 1);
    const uint row = local_q / args.group;
    const uint visible = min(args.n_pos, args.base_pos + row + 1);
    const uint tiles_used = (visible + (uint)AMO_NR0 - 1) / (uint)AMO_NR0;

    device const float * ml_base = ml + ((ulong)kvh * (ulong)N + local_q) * (ulong)n_tiles * 2;

    float m_part = -INFINITY;
    for (uint t = quarter; t < tiles_used; t += 4) {
        m_part = max(m_part, ml_base[t * 2 + 0]);
    }
    red[(col * 4 + quarter) * 2 + 0] = m_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m_glob = max(
        max(red[(col * 4 + 0) * 2 + 0], red[(col * 4 + 1) * 2 + 0]),
        max(red[(col * 4 + 2) * 2 + 0], red[(col * 4 + 3) * 2 + 0]));

    float l_part = 0.0f;
    for (uint t = quarter; t < tiles_used; t += 4) {
        const float m_t = ml_base[t * 2 + 0];
        const float l_t = ml_base[t * 2 + 1];
        if (m_t != -INFINITY) {
            l_part += l_t * exp2(m_t - m_glob);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[(col * 4 + quarter) * 2 + 1] = l_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (quarter == 0) {
        const float l_glob = red[(col * 4 + 0) * 2 + 1] + red[(col * 4 + 1) * 2 + 1]
                           + red[(col * 4 + 2) * 2 + 1] + red[(col * 4 + 3) * 2 + 1];
        // Also fold m_glob into the scratch so staging threads can read it.
        red[(col * 4 + 0) * 2 + 0] = m_glob;
        inv_l[col] = l_glob > 0.0f ? 1.0f / l_glob : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------------------
// KQV norm: edge-safe variant. GEMM body identical to
// kernel_attn_matrix_kqv_f32, with F16 P~ staging scaled by c_t and the
// epilogue scaled by inv_l.
// ---------------------------------------------------------------------------
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_attn_matrix_kqv_norm_f32(
        constant attn_matrix_args & args [[buffer(0)]],
        device const half  * probs_h [[buffer(1)]],
        device const float * ml      [[buffer(2)]],
        device const half  * v_t     [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        threadgroup  uchar * shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const uint kvh = tgpig.z;
    const int M = (int)args.head_dim;
    const int N = (int)(args.n_rows * args.group);
    const int r0 = (int)tgpig.y * AMO_NR0;
    const int r1 = (int)tgpig.x * AMO_NR1;
    const short nr0 = (M - r0 < AMO_NR0) ? (short)(M - r0) : AMO_NR0;
    const short nr1 = (N - r1 < AMO_NR1) ? (short)(N - r1) : AMO_NR1;
    const uint local_q_last = (uint)(r1 + nr1 - 1);
    const uint row_last = local_q_last / args.group;
    const uint max_visible = min(args.n_pos, args.base_pos + row_last + 1);
    const uint n_tiles = (args.n_pos + (uint)AMO_NR0 - 1) / (uint)AMO_NR0;
    const short lr0 = ((short)tiitg / AMO_NL0) < nr0 ? ((short)tiitg / AMO_NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg / AMO_NL1) < nr1 ? ((short)tiitg / AMO_NL1) : nr1 - 1;
    const short il0 = tiitg % AMO_NL0;
    const short iy = 8 * (tiitg % AMO_NL1);
    const uint d_thread = (uint)(r0 + lr0);
    const uint safe_d = min(d_thread, args.head_dim - 1u);
    const uint local_q_thread = (uint)(r1 + lr1);
    const uint safe_local_q = min(local_q_thread, (uint)(N - 1));
    device const half * vt_base =
        v_t + ((ulong)kvh * args.head_dim + safe_d) * args.vt_stride;
    device const half * probs_base =
        probs_h + ((ulong)kvh * (ulong)N + (ulong)safe_local_q) * args.n_pos;
    device const float * ml_base =
        ml + ((ulong)kvh * (ulong)N + (ulong)safe_local_q) * (ulong)n_tiles * 2;

    // Per-column global (m, 1/l).
    threadgroup float * red   = (threadgroup float *)(shmem);
    threadgroup float * inv_l = (threadgroup float *)(shmem + 8192);
    attn_matrix_kqv_norm_prologue(args, ml, red, inv_l, r1, nr1, kvh, tiitg);
    const float m_glob_thread = red[(lr1 * 4 + 0) * 2 + 0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.n_pos; loop_k += AMO_NK) {
        if (args.causal_skip != 0u && loop_k >= max_visible) continue;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / AMO_NL0) / 8;
            const short lx = (tiitg / AMO_NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            const uint kk = loop_k + 16 * il0 + i;
            sa[64 * ib + 8 * ly + lx] = (d_thread < args.head_dim && kk < args.n_pos)
                ? vt_base[kk]
                : (half)0.0f;
        }

        {
            const short sx = tiitg % AMO_NL1;
            const short sy = (tiitg / AMO_NL1) / 8;
            const short ly = (tiitg / AMO_NL1) % 8;
            const short ib = 4 * sx + sy;
            const uint kk = loop_k + iy;
            threadgroup half * dst = sb + 64 * ib + 8 * ly;
            // c_t is constant across this 8-pos chunk (chunks never cross a
            // 64-pos tile boundary: kk % 32 == 0 and 8 <= 32).
            const uint t = kk / (uint)AMO_NR0;
            float c_t = 0.0f;
            if (local_q_thread < (uint)N && kk < args.n_pos) {
                const float m_t = ml_base[t * 2 + 0];
                c_t = (m_t != -INFINITY) ? exp2(m_t - m_glob_thread) : 0.0f;
            }
            const bool aligned_probs = (args.n_pos & 7u) == 0u;
            if (aligned_probs && local_q_thread < (uint)N && kk + 7 < args.n_pos) {
                const half4 p0 = *((device const half4 *)(probs_base + kk) + 0);
                const half4 p1 = *((device const half4 *)(probs_base + kk) + 1);
                *((threadgroup half4 *)dst + 0) = half4(float4(p0) * c_t);
                *((threadgroup half4 *)dst + 1) = half4(float4(p1) * c_t);
            } else {
                for (short i = 0; i < 8; ++i) {
                    const uint kki = kk + i;
                    dst[i] = (local_q_thread < (uint)N && kki < args.n_pos)
                        ? half((float)probs_base[kki] * c_t)
                        : (half)0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < AMO_NK / 8; ++ik) {
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
        + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * AMO_NR0;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * AMO_NR0 * (i / 4), AMO_NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += AMO_NR1) {
            const uint local_q = (uint)(r1 + j);
            const uint row = local_q / args.group;
            const uint g = local_q % args.group;
            const float inv = inv_l[j];
            device float * Dst = out + ((ulong)row * args.n_q_heads + (ulong)kvh * args.group + g) * args.head_dim + r0;
            threadgroup float * Src = ((threadgroup float *)shmem) + j * AMO_NR0;
            for (int i = 0; i < nr0; ++i) {
                Dst[i] = Src[i] * inv;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// KQV norm: full-tile variant (n_pos % 32 == 0, N % 32 == 0). GEMM body
// identical to kernel_attn_matrix_kqv_f32_full_tiles, staging scaled by c_t;
// the output pass runs through the threadgroup spill so inv_l can be applied.
// ---------------------------------------------------------------------------
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_attn_matrix_kqv_norm_f32_full_tiles(
        constant attn_matrix_args & args [[buffer(0)]],
        device const half  * probs_h [[buffer(1)]],
        device const float * ml      [[buffer(2)]],
        device const half  * v_t     [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        threadgroup  uchar * shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const uint kvh = tgpig.z;
    const int N = (int)(args.n_rows * args.group);
    const int r0 = (int)tgpig.y * AMO_NR0;
    const int r1 = (int)tgpig.x * AMO_NR1;
    const uint local_q_last = (uint)(r1 + AMO_NR1 - 1);
    const uint row_last = local_q_last / args.group;
    const uint max_visible = min(args.n_pos, args.base_pos + row_last + 1);
    const uint n_tiles = (args.n_pos + (uint)AMO_NR0 - 1) / (uint)AMO_NR0;
    const short lr0 = ((short)tiitg / AMO_NL0);
    const short lr1 = ((short)tiitg / AMO_NL1);
    const short il0 = tiitg % AMO_NL0;
    const short iy = 8 * (tiitg % AMO_NL1);
    const uint d_thread = (uint)(r0 + lr0);
    const uint local_q_thread = (uint)(r1 + lr1);
    device const half * vt_base =
        v_t + ((ulong)kvh * args.head_dim + d_thread) * args.vt_stride;
    device const half * probs_base =
        probs_h + ((ulong)kvh * (ulong)N + (ulong)local_q_thread) * args.n_pos;
    device const float * ml_base =
        ml + ((ulong)kvh * (ulong)N + (ulong)local_q_thread) * (ulong)n_tiles * 2;

    threadgroup float * red   = (threadgroup float *)(shmem);
    threadgroup float * inv_l = (threadgroup float *)(shmem + 8192);
    attn_matrix_kqv_norm_prologue(args, ml, red, inv_l, r1, AMO_NR1, kvh, tiitg);
    const float m_glob_thread = red[(lr1 * 4 + 0) * 2 + 0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.n_pos; loop_k += AMO_NK) {
        if (args.causal_skip != 0u && loop_k >= max_visible) continue;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / AMO_NL0) / 8;
            const short lx = (tiitg / AMO_NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            const uint kk = loop_k + 16 * il0 + i;
            sa[64 * ib + 8 * ly + lx] = vt_base[kk];
        }

        {
            const short sx = tiitg % AMO_NL1;
            const short sy = (tiitg / AMO_NL1) / 8;
            const short ly = (tiitg / AMO_NL1) % 8;
            const short ib = 4 * sx + sy;
            const uint kk = loop_k + iy;
            threadgroup half * dst = sb + 64 * ib + 8 * ly;
            const uint t = kk / (uint)AMO_NR0;
            const float m_t = ml_base[t * 2 + 0];
            const float c_t = (m_t != -INFINITY) ? exp2(m_t - m_glob_thread) : 0.0f;
            // Full tiles: n_pos % 32 == 0, so 8-wide vector loads are in
            // bounds and aligned.
            const half4 p0 = *((device const half4 *)(probs_base + kk) + 0);
            const half4 p1 = *((device const half4 *)(probs_base + kk) + 1);
            *((threadgroup half4 *)dst + 0) = half4(float4(p0) * c_t);
            *((threadgroup half4 *)dst + 1) = half4(float4(p1) * c_t);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < AMO_NK / 8; ++ik) {
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
        + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * AMO_NR0;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * AMO_NR0 * (i / 4), AMO_NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        for (int j = tiitg; j < AMO_NR1; j += AMO_NR1) {
            const uint local_q = (uint)(r1 + j);
            const uint row = local_q / args.group;
            const uint g = local_q % args.group;
            const float inv = inv_l[j];
            device float * Dst = out + ((ulong)row * args.n_q_heads + (ulong)kvh * args.group + g) * args.head_dim + r0;
            threadgroup float * Src = ((threadgroup float *)shmem) + j * AMO_NR0;
            for (int i = 0; i < AMO_NR0; ++i) {
                Dst[i] = Src[i] * inv;
            }
        }
    }
}
