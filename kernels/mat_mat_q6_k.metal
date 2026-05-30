// Q6_K mat-mat (W · X^T → Y).
//
// kernel_mat_mat_q6_K_f32: Q6_K weight × F32 activations → F32 output.
// Same 64×32×32 simdgroup_matrix tile as `mat_mat_q4_k.metal`, lifted
// from llama.cpp `kernel_mul_mm` (classic non-MPS-tensor path,
// ggml-metal.metal:9440-9648 + dequantize_q6_K at line 723).
//
// Used by H5.3b.6 to lift the remaining weight-traffic-heavy paths
// (ffn_down, lm_head) out of the per-token mat-vec re-read loop. At
// Qwen3.6-27B Q4_K_M:
//   * ffn_down is Q6_K [17408, 5120] = 70 MiB per layer × 64 layers =
//     ~4.5 GiB of weight bytes per outer step at N=16 if each layer's
//     ffn_down stays per-token mat-vec.
//   * lm_head is Q6_K [5120, 248320] = ~1 GiB; per-token mat-vec at
//     N=16 = 16 GiB redundant traffic per outer step. lm_head is
//     LATENCY-CRITICAL (sits on the per-token tail, no other work
//     hides it).
// Lifting both to mat-mat eliminates that traffic.
//
// Output layout: same as mat_mat_q4_k.metal — kernel writes
// `dst[r + c*M]` per llama, which is BIT-IDENTICAL to row-major
// `[N, n_out]`. Downstream consumers work without transpose; the
// H5.3b.0 layout sanity test verified this for Q4_K and the same
// indexing arithmetic applies here.
//
// NOT bit-exact with N successive mat-vec calls (lifted kernel
// stages activations through half before float accumulation; cosine
// ≥ 0.999 vs scalar-float mat-vec is the gate).
//
// Q6_K block layout (block_q6_K, 210 bytes / 256 elements):
//   uint8  ql[128]    // low 4 bits of every quant
//   uint8  qh[64]     // high 2 bits of every quant (4 quants per byte)
//   int8   scales[16] // per-16-element scales
//   half   d          // super-block scale

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr int   QK_K           = 256;
constant constexpr int   Q6K_BYTES      = 210;
constant constexpr int   Q6K_NL         = QK_K / 16;     // 16 dequant calls per super-block

constant constexpr int   NR0_MM         = 64;
constant constexpr int   NR1_MM         = 32;
constant constexpr int   NK_MM          = 32;
constant constexpr int   NL0_MM         = NK_MM / 16;    // 2 dequant calls per K-step per row
constant constexpr int   NL1_MM         = NK_MM / 8;     // 4 activation chunks per K-step per col
constant constexpr int   NW_MM          = 32;
constant constexpr int   N_SIMD_GROUPS  = 4;
constant constexpr int   N_THREADS_MM   = NW_MM * N_SIMD_GROUPS; // 128

struct mat_mat_q6k_args {
    uint  M;          // output rows == n_out
    uint  N;          // output cols == n_query
    uint  K;          // inner dim   == n_in
    uint  nb01;       // weight row stride in BYTES
    uint  stride_b;   // activation row stride in F32 ELEMENTS (= K)
};

// ---------------------------------------------------------------------------
// Q6_K dequant — produces 16 half values into a half4x4 register tile.
//
// Lifted directly from llama's `dequantize_q6_K` (ggml-metal.metal:723).
// `il ∈ [0, 16)` selects one 16-element sub-tile of the 256-element
// super-block. The packed-quant decode unrolls a per-shift-mask formula
// chosen by which sub-tile we're decoding (il/2 % 4 selects shift /
// mask combination; il%2 picks the odd/even quant within the byte
// pairing).
inline void dequantize_q6_K_half(device const uchar * blk_bytes,
                                 short il,
                                 thread half4x4 & reg) {
    // Field layout matches block_q6_K:
    //   uchar ql[128];
    //   uchar qh[64];
    //   int8  scales[16];
    //   half  d;          // at offset 128 + 64 + 16 = 208
    device const uint16_t * ql = (device const uint16_t *)(blk_bytes + 0);
    device const uint16_t * qh = (device const uint16_t *)(blk_bytes + 128);
    device const int8_t   * scales = (device const int8_t *)(blk_bytes + 128 + 64);
    const half d_all = ((device const half *)(blk_bytes + 128 + 64 + 16))[0];

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    float sc = scales[(il % 2) + 2 * ((il / 2))];
    short il_inner = (il / 2) & 3;

    const uint32_t kmask1 = il_inner > 1
        ? (il_inner > 2 ? 0xC0C0C0C0 : 0x30303030)
        : (il_inner > 0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il_inner > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float ml  = (float)d_all * sc * 32.0f;
    const float dl0 = (float)d_all * sc;
    const float dl1 = dl0 / 256.0f;
    const float dl2 = dl0 / (256.0f * 256.0f);
    const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
    const uint8_t shr_h = il_inner > 2 ? 2 : 0;
    const uint8_t shl_h = il_inner > 1 ? 0 : (il_inner > 0 ? 2 : 4);
    const uint8_t shr_l = il_inner > 1 ? 4 : 0;

    FOR_UNROLL (int i = 0; i < 4; ++i) {
        const uint32_t low  = (ql[2 * i] | (uint32_t)(ql[2 * i + 1] << 16)) & kmask2;
        const uint32_t high = (qh[2 * i] | (uint32_t)(qh[2 * i + 1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = (half)(dl0 * ((float)(q & 0xFF))         - ml);
        reg[i][1] = (half)(dl1 * ((float)(q & 0xFF00))       - ml);
        reg[i][2] = (half)(dl2 * ((float)(q & 0xFF0000))     - ml);
        reg[i][3] = (half)(dl3 * ((float)(q & 0xFF000000))   - ml);
    }
}

// ---------------------------------------------------------------------------
// kernel_mat_mat_q6_K_f32 — same tile as Q4_K mat-mat, just a different
// dequant path. Bit-by-bit comments mirror mat_mat_q4_k.metal.
kernel void kernel_mat_mat_q6_K_f32(
        constant mat_mat_q6k_args & args   [[buffer(0)]],
        device const uchar        * srcA   [[buffer(1)]], // Q6_K weight bytes [M, K]
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

    const short offset1 = il0 / Q6K_NL; // always 0 for our NL0 setup
    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q6K_BYTES;
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
            dequantize_q6_K_half(x_ptr, il, temp_a);

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

        // Advance pointers. Same Q4_K-style scaling: byte-stride
        // multiplied by Q6K_BYTES because we hold uchar* (NOT block_q6_K*).
        // Codex Q3 fix from H5.3b.0; same bug class, same mitigation.
        il = (il + 2 < Q6K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q6K_BYTES * ((2 + Q6K_NL - 1) / Q6K_NL)
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
// kernel_mat_mat_q6_K_f32_n16 — H5.3b.5.5 NR1=16 specialization for Q6_K.
//
// Mirrors mat_mat_q4_k.metal::kernel_mat_mat_q4_K_f32_n16. Same tile
// shape (NR0=64 M × NR1=16 N × NK=32 K, 4 sg × 32M×8N each, mc[4]),
// same B-tile layout (`ib = 2*sx + sy`), same B-load gating to first
// 64 threads. Only the dequant function differs (Q6_K instead of Q4_K).
//
// Used for ffn_down (Q6_K [F=17408, H=5120]) and lm_head
// (Q6_K [H=5120, V=248320]) in the layer-major H5.3b.6 path. lm_head
// is LATENCY-CRITICAL (last op on the per-token tail before argmax),
// so the retune win compounds via Amdahl's law.

constant constexpr int NR1_SPECIAL_N16_Q6 = 16;
constant constexpr int NL1_SPECIAL_N16_Q6 = NK_MM / 8;
constant constexpr int B_LOAD_THREADS_N16_Q6 =
    NR1_SPECIAL_N16_Q6 * NL1_SPECIAL_N16_Q6;

kernel void kernel_mat_mat_q6_K_f32_n16(
        constant mat_mat_q6k_args & args   [[buffer(0)]],
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
    const int r1 = tgpig.x * NR1_SPECIAL_N16_Q6;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = (short)tiitg / NL1_SPECIAL_N16_Q6;
    const short iy = 8 * (tiitg % NL1_SPECIAL_N16_Q6);

    const short offset1 = il0 / Q6K_NL;
    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q6K_BYTES;
    device const float * y_ptr = srcB + (ulong)args.stride_b * lr1
                                       + (ulong)iy;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb;
    simdgroup_float8x8  mc[4];

    FOR_UNROLL (short i = 0; i < 4; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        // PHASE 1: A-tile (Q6_K dequant).
        {
            half4x4 temp_a;
            dequantize_q6_K_half(x_ptr, il, temp_a);

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
        if (tiitg < B_LOAD_THREADS_N16_Q6) {
            const short sx = (tiitg % NL1_SPECIAL_N16_Q6);
            const short sy = (tiitg / NL1_SPECIAL_N16_Q6) / 8;
            const short ly = (tiitg / NL1_SPECIAL_N16_Q6) % 8;
            const short ib = 2 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q6K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q6K_BYTES * ((2 + Q6K_NL - 1) / Q6K_NL)
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
            for (int j = tiitg; j < NR1_SPECIAL_N16_Q6; j += NR1_SPECIAL_N16_Q6) {
                device float * D = dst + r0 + (r1 + j) * args.M;
                threadgroup float * C = temp_str + (j * NR0_MM);
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}
