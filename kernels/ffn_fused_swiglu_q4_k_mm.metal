// =============================================================================
// EXPERIMENTAL — FAILED A-LITE GATE — NOT IN PRODUCTION (v0.73c.2)
// =============================================================================
//
// Layer-major fused SwiGLU FFN — Q4_K mat-mat × 2 + silu_mul, NR1=16.
//
// Replaces the 3-dispatch sequence at v0.73c.1 with one fused kernel that:
//   * reads h_pack (the activation tile) ONCE per super-block, used by
//     both gate and up matmul passes
//   * dequants W_gate and W_up into separate shmem regions sa_g / sa_u
//   * runs TWO simdgroup_multiply_accumulate passes per ik (gate + up)
//   * in Phase 4: silu(mc_gate) * mc_up fused at write, eliminating
//     materialization of ffn_gate_pack and ffn_up_pack intermediates
//
// Bit-exactness vs unfused (mat_mat_q4_K + mat_mat_q4_K + silu_mul):
//   cos = 1.000000, max|Δ| = 0.0 — the half-staging in mat-mat is the
//   SAME in both paths (same simdgroup_load + dequant_q4_K_half flow),
//   and silu_mul is deterministic on identical inputs, so the diff
//   collapses to zero. Validated by `ffn_fused_swiglu_q4_K_mm_n16_matches_unfused`.
//
// Why this kernel is NOT in production: A-lite go/no-go bench
// (`ffn_fused_swiglu_q4_K_amortization_vs_unfused`) at production 64-layer
// 27B FFN shape (n_in=5120, n_out=17408, N=16) measured:
//   fused:    7.0 ms / 64 layers GPU
//   unfused:  7.5 ms / 64 layers GPU
//   ratio: 0.936 — only 1.07× speedup; codex threshold was ≤ 0.7.
//
// The structural ceiling on this lever is set by the unchanged
// W_gate + W_up weight reads (~94 MB / layer × 64 = ~6 GB / call), which
// fusion cannot avoid. The unfused 3-dispatch path is already
// scheduler-friendly inside one command buffer; the dispatch-count
// reduction codex projected at "3-6 ms / call" turned out to be ~0.5 ms
// in practice (Metal pipelines pre-merged dispatches well).
//
// Codex's optimistic estimate of 5-15 ms/call savings was 10-30× too high.
// This kernel is preserved as institutional memory: future readers
// considering FFN fusion can see we measured the actual ceiling. Do NOT
// plumb into the layer-major path without first running the A-lite bench
// and confirming the regime has changed.
//
// Tile geometry (lifted from mat_mat_q4_k.metal NR1=16 fast path):
//   NR0 = 64, NR1 = 16, NK = 32
//   threadgroup memory: 4096 B (sa_g) + 4096 B (sa_u) + 1024 B (sb)
//                       Phase 4 reuses sa_g / sa_u as fp32 scratch.
//   simdgroups per TG: 4
//
// Output layout: dst is fused `silu(W_g·x) * (W_u·x)` written col-major
// `[n_out, n_query]` (= bit-equivalent to row-major `[n_query, n_out]`,
// same llama convention as the un-fused mat-mat kernels).

#include <metal_stdlib>
using namespace metal;

constant constexpr int   QK_K           = 256;
constant constexpr int   Q4K_BYTES      = 144;
constant constexpr int   Q4K_NL         = QK_K / 16;     // 16

constant constexpr int   NR0_MM         = 64;
constant constexpr int   NR1_FUSED      = 32;
constant constexpr int   NR1_FUSED_N16  = 16;
constant constexpr int   NK_MM          = 32;
constant constexpr int   NL0_MM         = NK_MM / 16;    // 2
constant constexpr int   NL1_FUSED      = NK_MM / 8;     // 4
constant constexpr int   NL1_FUSED_N16  = NK_MM / 8;     // 4
constant constexpr int   B_LOAD_THREADS = NR1_FUSED_N16 * NL1_FUSED_N16; // 64

struct ffn_fused_swiglu_q4k_mm_args {
    uint  M;          // n_out (= F = 17408 for 27B)
    uint  N;          // n_query (host enforces == 16)
    uint  K;          // n_in (= H = 5120 for 27B)
    uint  nb01;       // weight row stride in BYTES = (K / QK_K) * Q4K_BYTES
    uint  stride_b;   // activation row stride in F32 ELEMENTS (= K)
};

// ---------------------------------------------------------------------------
// Q4_K dequant — same as mat_mat_q4_k.metal::dequantize_q4_K_half. Inlined
// here so this kernel stays self-contained at the .metal file level.
inline void dequantize_q4_K_half_fused(device const uchar * blk_bytes,
                                        short il,
                                        thread half4x4 & reg) {
    const half d_h    = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar * scales = blk_bytes + 4;
    device const uchar * qs     = blk_bytes + 4 + 12;

    const short is  = (il / 4) * 2;
    const short k01 = (il / 2) & 1;
    uchar sc_u, m_u;
    if (is < 4) {
        sc_u = scales[is + k01] & 63;
        m_u  = scales[is + k01 + 4] & 63;
    } else {
        sc_u = (scales[is + k01 + 4] & 0x0F) | ((scales[is + k01 - 4] >> 6) << 4);
        m_u  = (scales[is + k01 + 4] >>   4) | ((scales[is + k01    ] >> 6) << 4);
    }

    qs = qs + (il / 4) * 32 + 16 * (il & 1);
    short il_inner = il & 3;
    const float d   = il_inner < 2 ? (float)d_h : (float)d_h / 16.0f;
    const float dmin = (float)dmin_h;
    const float dl  = d   * (float)sc_u;
    const float ml  = dmin * (float)m_u;
    const ushort mask = il_inner < 2 ? 0x0F : 0xF0;

    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(dl * (float)(qs[i] & mask) - ml);
    }
}

// ---------------------------------------------------------------------------
// Stable SiLU: x / (1 + exp(-x)).
inline float silu_f(float x) {
    return x / (1.0f + exp(-x));
}

kernel void kernel_ffn_fused_swiglu_q4_K_mm_f32(
        constant ffn_fused_swiglu_q4k_mm_args & args  [[buffer(0)]],
        device const uchar          * srcA_gate [[buffer(1)]],
        device const uchar          * srcA_up   [[buffer(2)]],
        device const float          * srcB      [[buffer(3)]],
        device       float          * dst       [[buffer(4)]],
        threadgroup  uchar          * shmem     [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_FUSED;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;
    const short nr1 = ((int)args.N - r1 < NR1_FUSED) ? (short)((int)args.N - r1) : NR1_FUSED;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_FUSED) < nr1
                        ? ((short)tiitg / NL1_FUSED)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_FUSED);

    const short offset1 = il0 / Q4K_NL;
    device const uchar * x_ptr_g = srcA_gate
        + (ulong)args.nb01 * (r0 + lr0)
        + (ulong)offset1 * Q4K_BYTES;
    device const uchar * x_ptr_u = srcA_up
        + (ulong)args.nb01 * (r0 + lr0)
        + (ulong)offset1 * Q4K_BYTES;
    device const float * y_ptr = srcB + (ulong)args.stride_b * (r1 + lr1)
                                       + (ulong)iy;

    simdgroup_half8x8   ma_g[4];
    simdgroup_half8x8   ma_u[4];
    simdgroup_half8x8   mb[2];
    simdgroup_float8x8  mc_g[8];
    simdgroup_float8x8  mc_u[8];

    for (short i = 0; i < 8; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_fused(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa_g + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        {
            half4x4 temp_a;
            dequantize_q4_K_half_fused(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa_u + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_FUSED);
            const short sy = (tiitg / NL1_FUSED) / 8;
            const short ly = (tiitg / NL1_FUSED) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_u;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb[i / 4], ma_g[i % 4], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb[i / 4], ma_u[i % 4], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_base_g = (threadgroup float *)shmem;
    threadgroup float * temp_base_u = (threadgroup float *)(shmem + 8192);
    threadgroup float * temp_str_g = temp_base_g
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = temp_base_u
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 16 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.M);
    for (short c = 0; c < 16; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < (int)args.N) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            dst[global_m + (ulong)global_n * args.M] = silu_f(g_val) * u_val;
        }
    }
}

// ---------------------------------------------------------------------------
// kernel_ffn_fused_swiglu_q4_K_mm_n16_f32
//
// Computes inner = silu(W_gate · X^T) * (W_up · X^T) where:
//   W_gate, W_up: Q4_K [M, K]  (same shape; M=n_out, K=n_in)
//   X:           F32 [N, K]   row-major (host enforces N=16)
//   inner:       F32 [M, N]   col-major = bit-equivalent to row-major [N, M]
//
// Threadgroup grid: (cols / NR1_FUSED_N16, rows / NR0_MM, 1).
//                 = (1, n_out / 64, 1) at production shape.
kernel void kernel_ffn_fused_swiglu_q4_K_mm_n16_f32(
        constant ffn_fused_swiglu_q4k_mm_args & args  [[buffer(0)]],
        device const uchar          * srcA_gate [[buffer(1)]], // Q4_K W_gate [M, K]
        device const uchar          * srcA_up   [[buffer(2)]], // Q4_K W_up   [M, K]
        device const float          * srcB      [[buffer(3)]], // F32 X [N, K]
        device       float          * dst       [[buffer(4)]], // F32 [N, M] row-major
        threadgroup  uchar          * shmem     [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    // Threadgroup memory layout:
    //   sa_g: half[64M × 32K] = 4096 B, offset 0
    //   sa_u: half[64M × 32K] = 4096 B, offset 4096
    //   sb:   half[16N × 32K] = 1024 B, offset 8192 (live during matmul)
    // Post-matmul (Phase 4): we reuse sa_g / sa_u as float scratch
    // temp_str_g / temp_str_u for staging mc_gate[] / mc_up[] before the
    // fused silu_mul write.
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_FUSED_N16;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;

    // A-loading thread mapping (same across gate and up — both have shape [M, K]).
    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    // B-loading thread mapping: only first 64 threads load B.
    const short lr1 = (short)tiitg / NL1_FUSED_N16;
    const short iy = 8 * (tiitg % NL1_FUSED_N16);

    const short offset1 = il0 / Q4K_NL;
    device const uchar * x_ptr_g = srcA_gate
        + (ulong)args.nb01 * (r0 + lr0)
        + (ulong)offset1 * Q4K_BYTES;
    device const uchar * x_ptr_u = srcA_up
        + (ulong)args.nb01 * (r0 + lr0)
        + (ulong)offset1 * Q4K_BYTES;
    device const float * y_ptr = srcB + (ulong)args.stride_b * lr1
                                       + (ulong)iy;

    simdgroup_half8x8   ma_g[4];
    simdgroup_half8x8   ma_u[4];
    simdgroup_half8x8   mb;
    simdgroup_float8x8  mc_g[4];
    simdgroup_float8x8  mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        // PHASE 1a: A-tile load for W_gate.
        {
            half4x4 temp_a;
            dequantize_q4_K_half_fused(x_ptr_g, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        // PHASE 1b: A-tile load for W_up.
        {
            half4x4 temp_a;
            dequantize_q4_K_half_fused(x_ptr_u, il, temp_a);
            // Note: no barrier needed between 1a and 1b on the SAME data path
            // (different shmem regions; different lanes write different cells
            // of sa_g vs sa_u, and the matmul barriers below sync everything).
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        // PHASE 2: B-tile load (gated to first 64 threads). Same as
        // mat_mat_q4_k_f32_n16: ib = 2*sx + sy.
        if (tiitg < B_LOAD_THREADS) {
            const short sx = (tiitg % NL1_FUSED_N16);
            const short sy = (tiitg / NL1_FUSED_N16) / 8;
            const short ly = (tiitg / NL1_FUSED_N16) % 8;
            const short ib = 2 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        // Pointer advance (same Q4K_BYTES mitigation as mat_mat_q4_k.metal).
        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_u;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PHASE 3: simdgroup matmul (gate AND up), reading the SHARED sb tile.
        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    // PHASE 4: stage mc_g[] and mc_u[] into shmem, then per-lane silu*mul
    // and store. Use sa_g / sa_u space (matmul reads done) as fp32 scratch.
    // Per-sg covers 32 M × 8 N = 256 fp32 = 1024 B per scratch = WAY less
    // than the 4 KiB sa_g/sa_u regions, so reusing them is safe.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-lane fused silu*mul write. Each of the 4 sgs handles its own
    // 32-row × 8-col quadrant of the 64 × 16 fused output region.
    //   sgitg & 1 → M-quadrant (0=rows 0..31, 1=rows 32..63 of the tile)
    //   sgitg >> 1 → N-quadrant (0=cols 0..7, 1=cols 8..15)
    //
    // mc tile layout in staged scratch (mirrors mat_mat_q4_k.metal):
    //   simdgroup_store(mc[i], temp_str + 8*i, NR0_MM, 0, false)
    // lays mc tile i (an 8×8 fp32 block) into the 64-row × 8-col region
    // such that cell (mr ∈ [0,8), mc ∈ [0,8)) of tile i is at
    //   temp_str[(8*i + mr) + mc * NR0_MM]
    // and within an sg's quadrant the 4 mc tiles span M-rows 0..31
    // (mc[0]→rows 0..7, mc[1]→8..15, mc[2]→16..23, mc[3]→24..31)
    // all at the same 8-col span.
    {
        const short m_off = 32 * (sgitg & 1);
        const short n_off = 8  * (sgitg >> 1);
        const short m_local = (short)tiitg & 31; // 0..31 within sg's M-quadrant
        const short tile_i  = m_local >> 3;
        const short mr      = m_local & 7;
        const int  global_m = r0 + m_off + m_local;
        const bool m_in     = (global_m < (int)args.M);
        for (short c = 0; c < 8; ++c) {
            const int global_n = r1 + n_off + c;
            if (m_in && global_n < (int)args.N) {
                const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
                const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
                const float fused = silu_f(g_val) * u_val;
                dst[global_m + (ulong)global_n * args.M] = fused;
            }
        }
    }
}
