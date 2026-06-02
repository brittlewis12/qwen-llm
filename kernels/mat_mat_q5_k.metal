// Q5_K mat-mat (W · X^T → Y).
//
// kernel_mat_mat_q5_K_f32: Q5_K weight × F32 activations → F32 output.
// Same 64×32×32 simdgroup_matrix tile as `mat_mat_q4_k.metal` /
// `mat_mat_q6_k.metal`, lifted from llama.cpp `kernel_mul_mm` (classic
// non-MPS-tensor path, ggml-metal.metal:9440-9648 + dequantize_q5_K
// at line 700).
//
// Used by v0.73a.1 to lift the GDN out_proj weight-traffic out of the
// per-token mat-vec re-read loop. At Qwen3.6-27B Q4_K_M:
//   * out_proj (ssm_out.weight) is Q5_K [n_in=6144, n_out=5120].
//     Q5_K block bytes: (n_in/256)*176 = 24*176 = 4224 bytes per row;
//     × n_out=5120 rows = 21,626,880 bytes ≈ 20.6 MiB per layer.
//     × 48 GDN layers × 16 tokens (= per-token mat-vec re-reads) =
//     ~16 GiB of redundant weight traffic per outer step. Mat-mat
//     amortizes the per-K-step weight load across all N=16 cols, so
//     the actual reads collapse to ~990 MiB / outer step. (Earlier
//     "30 MiB per layer" framing was inflated; codex review caught
//     it. The 3.59× isolated-bench speedup is real and unchanged.)
//
// Output layout: same as Q4_K / Q6_K mat-mat — kernel writes
// `dst[r + c*M]` per llama, which is BIT-IDENTICAL to row-major
// `[N, n_out]`. Downstream consumers work without transpose.
//
// NOT bit-exact with N successive mat-vec calls (lifted kernel stages
// activations through half before float accumulation; cosine ≥ 0.999
// vs scalar-float mat-vec is the gate, same as Q4_K / Q6_K).
//
// Q5_K block layout (block_q5_K, 176 bytes / 256 elements):
//   half  d                  // super-block scale
//   half  dmin               // super-block min
//   u8    scales[12]         // 6-bit packed (sc, min) for 8 sub-blocks (same as Q4_K)
//   u8    qh[32]             // high bit of each 5-bit quant (256 bits)
//   u8    qs[128]            // low 4 bits of each quant (4-bit nibbles, paired)

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr int   QK_K           = 256;
constant constexpr int   Q5K_BYTES      = 176;
constant constexpr int   Q5K_NL         = QK_K / 16;     // 16 dequant calls per super-block

constant constexpr int   NR0_MM         = 64;
constant constexpr int   NR1_MM         = 32;
constant constexpr int   NK_MM          = 32;
constant constexpr int   NL0_MM         = NK_MM / 16;    // 2 dequant calls per K-step per row
constant constexpr int   NL1_MM         = NK_MM / 8;     // 4 activation chunks per K-step per col
constant constexpr int   NW_MM          = 32;
constant constexpr int   N_SIMD_GROUPS  = 4;
constant constexpr int   N_THREADS_MM   = NW_MM * N_SIMD_GROUPS; // 128

struct mat_mat_q5k_args {
    uint  M;          // output rows == n_out
    uint  N;          // output cols == n_query
    uint  K;          // inner dim   == n_in
    uint  nb01;       // weight row stride in BYTES = (K / QK_K) * Q5K_BYTES
    uint  stride_b;   // activation row stride in F32 ELEMENTS (= K)
};

// ---------------------------------------------------------------------------
// Q5_K dequant — produces 16 half values into a half4x4 register tile.
//
// Lifted directly from llama's `dequantize_q5_K` (ggml-metal.metal:700).
// `il ∈ [0, 16)` selects one 16-element sub-tile of the 256-element
// super-block. Same get_scale_min_k4_just2 packed-scale decode as Q4_K
// (the 12-byte `scales` table layout is identical), then adds the qh
// high-bit contribution: `q5[k] = (qs[k] & mask4) + (qh[k] & ul ? qh_val : 0)`.
//
// Note vs Q4_K: same `is`/`il_inner`/`mask`/`d`/`dmin`/`dl`/`ml` math;
// only the inner loop adds the `qh`-bit addition before scaling. The
// `qh_val` is 16.f when il<2 (low nibble path) and 256.f when il≥2
// (high nibble path) because the high-nibble quants carry an extra
// implicit ×16 from the nibble shift.
inline void dequantize_q5_K_half(device const uchar * blk_bytes,
                                 short il,
                                 thread half4x4 & reg) {
    // Field offsets matching block_q5_K:
    //   half d              @ 0
    //   half dmin           @ 2
    //   uchar scales[12]    @ 4
    //   uchar qh[32]        @ 16
    //   uchar qs[128]       @ 48
    const half d_h    = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar * scales = blk_bytes + 4;
    device const uchar * qh     = blk_bytes + 4 + 12;
    device const uchar * qs     = blk_bytes + 4 + 12 + 32;

    // get_scale_min_k4_just2 (unrolled, same as Q4_K kernel).
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

    // Match llama: `q  = q  + 32 * (il/4) + 16 * (il&1);`
    //              `qh = qh + 16 * (il&1);`
    //              `ul = 1 << (il/2);`
    //              `il = il & 3;`
    qs = qs + 32 * (il / 4) + 16 * (il & 1);
    qh = qh + 16 * (il & 1);
    const uchar ul = 1u << (il / 2);
    short il_inner = il & 3;

    const float d   = il_inner < 2 ? (float)d_h : (float)d_h / 16.0f;
    const float dmin = (float)dmin_h;
    const float dl  = d   * (float)sc_u;
    const float ml  = dmin * (float)m_u;
    const ushort mask = il_inner < 2 ? 0x0F : 0xF0;
    // qh contribution: 16.f for low-nibble path, 256.f for high-nibble path.
    // The high-nibble (il≥2) reads qs through mask 0xF0 which leaves the
    // value 16x the underlying nibble; the qh bit therefore needs to add
    // 16 * 16 = 256 to balance.
    const float qh_val = il_inner < 2 ? 16.0f : 256.0f;

    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const float q_low  = (float)(qs[i] & mask);
        const float q_high = (qh[i] & ul) ? qh_val : 0.0f;
        reg[i / 4][i % 4] = (half)(dl * (q_low + q_high) - ml);
    }
}

// ---------------------------------------------------------------------------
// kernel_mat_mat_q5_K_f32 — same tile as Q4_K / Q6_K mat-mat, just a
// different dequant path. Bit-by-bit comments mirror mat_mat_q4_k.metal.
kernel void kernel_mat_mat_q5_K_f32(
        constant mat_mat_q5k_args & args   [[buffer(0)]],
        device const uchar        * srcA   [[buffer(1)]], // Q5_K weight bytes [M, K]
        device const float        * srcB   [[buffer(2)]], // F32 activations [N, K] row-major
        device       float        * dst    [[buffer(3)]], // F32 output [N, M] row-major (= [M, N] col-major)
        threadgroup  uchar        * shmem  [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;
    const short nr1 = ((int)args.N - r1 < NR1_MM) ? (short)((int)args.N - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q5K_NL; // always 0 for our NL0 setup
    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q5K_BYTES;
    device const float * y_ptr = srcB + (ulong)args.stride_b * (r1 + lr1)
                                       + (ulong)iy;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb[2];
    simdgroup_float8x8  mc[8];

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        // Phase 1: dequant + write A tile.
        {
            half4x4 temp_a;
            dequantize_q5_K_half(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        // Phase 2: write B tile.
        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        // Pointer advance — same Q4_K-style scaling: byte-stride
        // multiplied by Q5K_BYTES because we hold uchar* (not block_q5_K*).
        // Codex Q3 fix from H5.3b.0; same bug class, same mitigation.
        il = (il + 2 < Q5K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q5K_BYTES * ((2 + Q5K_NL - 1) / Q5K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: simdgroup matmul.
        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        FOR_UNROLL (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // Phase 4: store mc[] to dst.
    if (r0 + NR0_MM <= (int)args.M && r1 + NR1_MM <= (int)args.N) {
        device float * C = dst + (r0 + 32 * (sgitg & 1))
                               + (r1 + 16 * (sgitg >> 1)) * args.M;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.M * (i / 4),
                            args.M, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shmem)
                                       + 32 * (sgitg & 1)
                                       + (16 * (sgitg >> 1)) * NR0_MM;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                            NR0_MM, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1_MM) {
                device float * D = dst + r0 + (r1 + j) * args.M;
                threadgroup float * C = temp_str + (j * NR0_MM);
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}

// =============================================================================
// kernel_mat_mat_q5_K_f32_n16 — H5.3b.5.5 NR1=16 specialization for Q5_K.
//
// Mirrors mat_mat_q4_k.metal::kernel_mat_mat_q4_K_f32_n16 and
// mat_mat_q6_k.metal::kernel_mat_mat_q6_K_f32_n16. Same tile shape
// (NR0=64 M × NR1=16 N × NK=32 K, 4 sg × 32M×8N each, mc[4]),
// same B-tile layout (`ib = 2*sx + sy`), same B-load gating to first
// 64 threads. Only the dequant function differs (Q5_K instead of
// Q4_K / Q6_K).
//
// Used for GDN out_proj (Q5_K [v_dim=6144, hidden=5120]) in the
// layer-major v0.73a.1 path, batched across N=16 verify tokens.

constant constexpr int NR1_SPECIAL_N16_Q5 = 16;
constant constexpr int NL1_SPECIAL_N16_Q5 = NK_MM / 8;
constant constexpr int B_LOAD_THREADS_N16_Q5 =
    NR1_SPECIAL_N16_Q5 * NL1_SPECIAL_N16_Q5;

kernel void kernel_mat_mat_q5_K_f32_n16(
        constant mat_mat_q5k_args & args   [[buffer(0)]],
        device const uchar        * srcA   [[buffer(1)]],
        device const float        * srcB   [[buffer(2)]],
        device       float        * dst    [[buffer(3)]],
        threadgroup  uchar        * shmem  [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_SPECIAL_N16_Q5;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = (short)tiitg / NL1_SPECIAL_N16_Q5;
    const short iy = 8 * (tiitg % NL1_SPECIAL_N16_Q5);

    const short offset1 = il0 / Q5K_NL;
    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q5K_BYTES;
    device const float * y_ptr = srcB + (ulong)args.stride_b * lr1
                                       + (ulong)iy;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb;
    simdgroup_float8x8  mc[4];

    FOR_UNROLL (short i = 0; i < 4; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        // PHASE 1: A-tile (Q5_K dequant).
        {
            half4x4 temp_a;
            dequantize_q5_K_half(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        // PHASE 2: B-tile (gated; ib = 2*sx + sy for NR1=16).
        if (tiitg < B_LOAD_THREADS_N16_Q5) {
            const short sx = (tiitg % NL1_SPECIAL_N16_Q5);
            const short sy = (tiitg / NL1_SPECIAL_N16_Q5) / 8;
            const short ly = (tiitg / NL1_SPECIAL_N16_Q5) % 8;
            const short ib = 2 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q5K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q5K_BYTES * ((2 + Q5K_NL - 1) / Q5K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PHASE 3: matmul (4 mc tiles per sg, 1 mb tile per sg).
        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 1 * 64 * (sgitg / 2));

        FOR_UNROLL (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 2 * 64;
        }
    }

    // PHASE 4: store.
    if (r0 + NR0_MM <= (int)args.M) {
        device float * C = dst + (r0 + 32 * (sgitg & 1))
                               + (r1 + 8 * (sgitg >> 1)) * args.M;
        FOR_UNROLL (short i = 0; i < 4; ++i) {
            simdgroup_store(mc[i], C + 8 * i, args.M, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shmem)
                                       + 32 * (sgitg & 1)
                                       + (8 * (sgitg >> 1)) * NR0_MM;
        FOR_UNROLL (short i = 0; i < 4; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * i, NR0_MM, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < NR1_SPECIAL_N16_Q5; j += NR1_SPECIAL_N16_Q5) {
                device float * D = dst + r0 + (r1 + j) * args.M;
                threadgroup float * C = temp_str + (j * NR0_MM);
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}

// =============================================================================
// kernel_mat_mat_q5_K_f32_n64 -- large-N prompt tile for Q5_K.
//
// Mirrors mat_mat_q4_k.metal::kernel_mat_mat_q4_K_f32_n64. Host dispatch only
// selects this for full N/M tiles (`N % 64 == 0`, `M % 64 == 0`), so there is no
// partial-tile store path here.

constant constexpr int NR1_SPECIAL_N64_Q5 = 64;
constant constexpr int N_SIMD_GROUPS_N64_Q5 = 8;
constant constexpr int N_THREADS_N64_Q5 = NW_MM * N_SIMD_GROUPS_N64_Q5;

kernel void kernel_mat_mat_q5_K_f32_n64(
        constant mat_mat_q5k_args & args   [[buffer(0)]],
        device const uchar        * srcA   [[buffer(1)]],
        device const float        * srcB   [[buffer(2)]],
        device       float        * dst    [[buffer(3)]],
        threadgroup  uchar        * shmem  [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_SPECIAL_N64_Q5;

    const bool load_a = tiitg < N_THREADS_MM;
    const short a_t = (short)(tiitg & (N_THREADS_MM - 1));
    const short lr0 = a_t / NL0_MM;
    const short il0 = a_t % NL0_MM;
    short il = il0;

    const short lr1 = (short)tiitg / NL1_MM;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q5K_NL;
    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q5K_BYTES;
    device const float * y_ptr = srcB + (ulong)args.stride_b * (r1 + lr1)
                                       + (ulong)iy;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb[2];
    simdgroup_float8x8  mc[8];

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        half4x4 temp_a;
        if (load_a) {
            dequantize_q5_K_half(x_ptr, il, temp_a);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (load_a) {
            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (a_t / NL0_MM) / 8;
                const short lx = (a_t / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 8 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q5K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q5K_BYTES * ((2 + Q5K_NL - 1) / Q5K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg & 1));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg >> 1));

        FOR_UNROLL (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 8 * 64;
        }
    }

    device float * C = dst + (r0 + 32 * (sgitg & 1))
                           + (r1 + 16 * (sgitg >> 1)) * args.M;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.M * (i / 4),
                        args.M, 0, false);
    }
}
