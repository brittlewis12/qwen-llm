// Q8_0 mat-mat (W · X^T → Y).
//
// kernel_mat_mat_q8_0_f32: Q8_0 weight × F32 activations → F32 output.
// Same 64×32×32 simdgroup_matrix tile geometry as `mat_mat_q4_k.metal`,
// `mat_mat_q5_k.metal`, `mat_mat_q6_k.metal`, lifted from llama.cpp's
// `kernel_mul_mm` template (classic non-MPS-tensor path,
// ggml-metal.metal:9440-9648 with `dequantize_q8_0` substituted, and
// the `nl=2` template parameter — see line 10108 host_name).
//
// Used by v0.73b.0 to switch the DFlash drafter from F32-dequant
// resident weights (~7.4 GB) to native Q8_0 (~1.85 GB), eliminating
// per-token re-read of dequant'd F32 at hot decode. Profile shows
// drafter is now 48.9% of decode wall (post-v0.73a.1); this is the
// single largest wall lever available.
//
// CRITICAL difference vs K-quant mat-mat: Q8_0 super-block is QK8_0=32
// elements, not QK_K=256. That means `Q8_0_NL = QK8_0/16 = 2` (vs
// `Q*K_NL = QK_K/16 = 16` for Q4_K/Q5_K/Q6_K). The pointer-advance
// inner loop:
//
//     il = (il + 2 < NL) ? il + 2 : il % 2;     // K-step index
//     x  = (il < 2) ? x + (2 + NL - 1) / NL : x; // weight ptr step
//
// reduces, with NL=2, to:
//
//     il = il % 2 (= il0 always; never advances within a super-block)
//     x advances by 1 super-block (= Q8_0_BYTES = 34) per K-step
//
// So each K-step (NK_MM=32 K elements) consumes exactly one Q8_0
// super-block per row. That's the natural alignment: NK_MM == QK8_0.
//
// Output layout: same as Q4_K/Q5_K/Q6_K mat-mat — kernel writes
// `dst[r + c*M]` per llama, which is BIT-IDENTICAL to row-major
// `[N, n_out]`. Downstream consumers work without transpose.
//
// NOT bit-exact with N successive Q8_0 mat-vec calls (lifted kernel
// stages activations through half before float accumulation; cosine
// ≥ 0.999 vs scalar-float mat-vec is the gate, same as Q4/Q5/Q6).
//
// Q8_0 block layout (block_q8_0, 34 bytes / 32 elements):
//   half  d                  // super-block scale
//   int8  qs[32]             // signed 8-bit quants

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr int   QK8_0          = 32;
constant constexpr int   Q8_0_BYTES     = 34;
constant constexpr int   Q8_0_NL        = QK8_0 / 16;    // = 2

constant constexpr int   NR0_MM         = 64;
constant constexpr int   NR1_MM         = 32;
constant constexpr int   NK_MM          = 32;            // matches QK8_0
constant constexpr int   NL0_MM         = NK_MM / 16;    // 2 dequant calls per K-step per row
constant constexpr int   NL1_MM         = NK_MM / 8;     // 4 activation chunks per K-step per col
constant constexpr int   NW_MM          = 32;
constant constexpr int   N_SIMD_GROUPS  = 4;
constant constexpr int   N_THREADS_MM   = NW_MM * N_SIMD_GROUPS; // 128

struct mat_mat_q8_0_args {
    uint  M;          // output rows == n_out
    uint  N;          // output cols == n_query
    uint  K;          // inner dim   == n_in
    uint  nb01;       // weight row stride in BYTES = (K / QK8_0) * Q8_0_BYTES
    uint  stride_b;   // activation row stride in F32 ELEMENTS (= K)
};

// ---------------------------------------------------------------------------
// Q8_0 dequant — produces 16 half values into a half4x4 register tile.
//
// Lifted directly from llama's `dequantize_q8_0` (ggml-metal.metal:574).
// `il ∈ [0, 2)` selects which 16-element half of the 32-element
// super-block: low (il=0) reads qs[0..16], high (il=1) reads qs[16..32].
// Each value is `d * (float)qs[i]`.
inline void dequantize_q8_0_half(device const uchar * blk_bytes,
                                 short il,
                                 thread half4x4 & reg) {
    const half d_h = ((device const half *)blk_bytes)[0];
    device const int8_t * qs = (device const int8_t *)(blk_bytes + 2);
    const float d = (float)d_h;
    const short base = 16 * il;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(d * (float)qs[base + i]);
    }
}

// ---------------------------------------------------------------------------
// kernel_mat_mat_q8_0_f32 — same 64x32x32 simdgroup_matrix tile as
// the K-quant mat-mats. Bit-by-bit comments mirror mat_mat_q4_k.metal,
// with the K-quant pointer-advance specialized to Q8_0_NL=2.
kernel void kernel_mat_mat_q8_0_f32(
        constant mat_mat_q8_0_args & args   [[buffer(0)]],
        device const uchar         * srcA   [[buffer(1)]], // Q8_0 weight bytes [M, K]
        device const float         * srcB   [[buffer(2)]], // F32 activations [N, K] row-major
        device       float         * dst    [[buffer(3)]], // F32 output [N, M] row-major (= [M, N] col-major)
        threadgroup  uchar         * shmem  [[threadgroup(0)]],
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

    // offset1 = il0 / NL = il0 / 2; with il0 ∈ {0, 1} this is 0.
    const short offset1 = il0 / Q8_0_NL;
    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q8_0_BYTES;
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
            dequantize_q8_0_half(x_ptr, il, temp_a);

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

        // Pointer advance — Q8_0 specialization of the K-quant formula.
        // With NL=2:
        //   il = (il + 2 < 2) ? il + 2 : il % 2  →  il = il % 2 = il0
        //     (il never advances; always reads the same half of the
        //      super-block. The OTHER lane in the lr0 group reads the
        //      OTHER half via il0 = 1 - this lane's il0.)
        //   x = (il < 2) ? x + (2 + Q8_0_NL - 1) / Q8_0_NL : x
        //     = (il < 2) ? x + 1 : x
        //   il < 2 always (since il ∈ {0, 1}), so x advances by 1 block
        //   = Q8_0_BYTES bytes per K-step = NK_MM elements per K-step.
        il = il % 2;  // unchanged for clarity; il is constant per call
        x_ptr = x_ptr + Q8_0_BYTES;
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

// Fixed precision-recovery falsifier: 16 output rows x 32 activation rows,
// K-step 64, one SIMD group. Q8 values and activations enter the matrix
// multiply as F32; only the stored Q8 scale remains F16.
kernel void kernel_mat_mat_q8_0_f32_r2c4k64(
        constant mat_mat_q8_0_args & args   [[buffer(0)]],
        device const uchar         * srcA   [[buffer(1)]],
        device const float         * srcB   [[buffer(2)]],
        device       float         * dst    [[buffer(3)]],
        threadgroup  float         * shmem  [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint r0 = tgpig.y * 16u;
    const uint c0 = tgpig.x * 32u;
    const uint nb = args.K / QK8_0;
    const ulong row_stride_bytes = (ulong)nb * Q8_0_BYTES;

    simdgroup_float8x8 acc[2][4];
    FOR_UNROLL (short rt = 0; rt < 2; ++rt) {
        FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
            acc[rt][ct] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    for (uint k0 = 0; k0 < args.K; k0 += 64u) {
        FOR_UNROLL (short chunk = 0; chunk < 2; ++chunk) {
            const uint cell = (uint)tiisg + 32u * (uint)chunk;
            const uint row = cell / 4u;
            const uint kchunk = cell % 4u;
            const uint kbase = k0 + 16u * kchunk;
            device const uchar * block = srcA
                + (ulong)(r0 + row) * row_stride_bytes
                + (ulong)(kbase / QK8_0) * Q8_0_BYTES;
            const float scale = (float)((device const half *)block)[0];
            device const int8_t * quants =
                (device const int8_t *)(block + 2) + (kbase % QK8_0);
            FOR_UNROLL (short i = 0; i < 16; ++i) {
                shmem[row * 64u + kchunk * 16u + (uint)i] =
                    scale * (float)quants[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short kt = 0; kt < 8; ++kt) {
            simdgroup_float8x8 activation[4];
            FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
                simdgroup_load(
                    activation[ct],
                    srcB + (ulong)(c0 + (uint)ct * 8u) * args.stride_b
                         + k0 + (uint)kt * 8u,
                    args.stride_b,
                    ulong2(0, 0),
                    true);
            }
            FOR_UNROLL (short rt = 0; rt < 2; ++rt) {
                simdgroup_float8x8 weight;
                simdgroup_load(
                    weight,
                    shmem + (uint)rt * 8u * 64u + (uint)kt * 8u,
                    64);
                FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
                    simdgroup_multiply_accumulate(
                        acc[rt][ct], weight, activation[ct], acc[rt][ct]);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (short rt = 0; rt < 2; ++rt) {
        FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
            const uint column = c0 + (uint)ct * 8u;
            simdgroup_store(acc[rt][ct], shmem, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            FOR_UNROLL (short part = 0; part < 2; ++part) {
                const uint cell = (uint)tiisg * 2u + (uint)part;
                const uint row = cell / 8u;
                const uint col = cell % 8u;
                if (column + col < args.N) {
                    dst[(ulong)(column + col) * args.M
                        + r0 + (uint)rt * 8u + row] = shmem[cell];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Four-SIMDgroup extension of the accepted R2C4K64 arithmetic. Each SIMDgroup
// owns an independent 16-output by 32-token result tile while all four reuse
// the same F32 16x64 weight tile. Dot-product order and matrix operands remain
// unchanged; only weight loading and dequantization are shared across 128
// adjacent tokens.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_mat_mat_q8_0_f32_r2c16k64(
        constant mat_mat_q8_0_args & args   [[buffer(0)]],
        device const uchar         * srcA   [[buffer(1)]],
        device const float         * srcB   [[buffer(2)]],
        device       float         * dst    [[buffer(3)]],
        threadgroup  float         * shmem  [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint r0 = tgpig.y * 16u;
    const uint c0 = tgpig.x * 128u + (uint)sgitg * 32u;
    const uint nb = args.K / QK8_0;
    const ulong row_stride_bytes = (ulong)nb * Q8_0_BYTES;

    simdgroup_float8x8 acc[2][4];
    FOR_UNROLL (short rt = 0; rt < 2; ++rt) {
        FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
            acc[rt][ct] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    for (uint k0 = 0; k0 < args.K; k0 += 64u) {
        if (sgitg == 0) {
            FOR_UNROLL (short chunk = 0; chunk < 2; ++chunk) {
                const uint cell = (uint)tiisg + 32u * (uint)chunk;
                const uint row = cell / 4u;
                const uint kchunk = cell % 4u;
                const uint kbase = k0 + 16u * kchunk;
                device const uchar * block = srcA
                    + (ulong)(r0 + row) * row_stride_bytes
                    + (ulong)(kbase / QK8_0) * Q8_0_BYTES;
                const float scale = (float)((device const half *)block)[0];
                device const int8_t * quants =
                    (device const int8_t *)(block + 2) + (kbase % QK8_0);
                FOR_UNROLL (short i = 0; i < 16; ++i) {
                    shmem[row * 64u + kchunk * 16u + (uint)i] =
                        scale * (float)quants[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short kt = 0; kt < 8; ++kt) {
            simdgroup_float8x8 activation[4];
            FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
                simdgroup_load(
                    activation[ct],
                    srcB + (ulong)(c0 + (uint)ct * 8u) * args.stride_b
                         + k0 + (uint)kt * 8u,
                    args.stride_b,
                    ulong2(0, 0),
                    true);
            }
            FOR_UNROLL (short rt = 0; rt < 2; ++rt) {
                simdgroup_float8x8 weight;
                simdgroup_load(
                    weight,
                    shmem + (uint)rt * 8u * 64u + (uint)kt * 8u,
                    64);
                FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
                    simdgroup_multiply_accumulate(
                        acc[rt][ct], weight, activation[ct], acc[rt][ct]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (short rt = 0; rt < 2; ++rt) {
        FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
            const uint column = c0 + (uint)ct * 8u;
            simdgroup_store(
                acc[rt][ct],
                dst + (ulong)column * args.M + r0 + (uint)rt * 8u,
                args.M,
                ulong2(0, 0),
                true);
        }
    }
}

struct mat_mat_q8_0_grouped_args {
    uint M;
    uint N;
    uint K;
    uint groups;
    uint nb01;
    uint stride_b;
    uint stride_c;
};

// Strided grouped-output form of R2C16K64. Each grid depth owns one output-A
// group and preserves the accepted per-group matrix arithmetic while reading
// and writing directly in the full attention and low-rank row layouts.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_mat_mat_q8_0_f32_r2c16k64_grouped(
        constant mat_mat_q8_0_grouped_args & args [[buffer(0)]],
        device const uchar                 * srcA [[buffer(1)]],
        device const float                 * srcB [[buffer(2)]],
        device       float                 * dst  [[buffer(3)]],
        threadgroup  float                 * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint group = tgpig.z;
    if (group >= args.groups) {
        return;
    }
    const uint r0 = tgpig.y * 16u;
    const uint c0 = tgpig.x * 128u + (uint)sgitg * 32u;
    const ulong row_stride_bytes = (ulong)args.nb01;
    const ulong group_weight_bytes = (ulong)args.M * row_stride_bytes;
    device const uchar * group_weights = srcA + (ulong)group * group_weight_bytes;

    simdgroup_float8x8 acc[2][4];
    FOR_UNROLL (short rt = 0; rt < 2; ++rt) {
        FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
            acc[rt][ct] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    for (uint k0 = 0; k0 < args.K; k0 += 64u) {
        if (sgitg == 0) {
            FOR_UNROLL (short chunk = 0; chunk < 2; ++chunk) {
                const uint cell = (uint)tiisg + 32u * (uint)chunk;
                const uint row = cell / 4u;
                const uint kchunk = cell % 4u;
                const uint kbase = k0 + 16u * kchunk;
                device const uchar * block = group_weights
                    + (ulong)(r0 + row) * row_stride_bytes
                    + (ulong)(kbase / QK8_0) * Q8_0_BYTES;
                const float scale = (float)((device const half *)block)[0];
                device const int8_t * quants =
                    (device const int8_t *)(block + 2) + (kbase % QK8_0);
                FOR_UNROLL (short i = 0; i < 16; ++i) {
                    shmem[row * 64u + kchunk * 16u + (uint)i] =
                        scale * (float)quants[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short kt = 0; kt < 8; ++kt) {
            simdgroup_float8x8 activation[4];
            FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
                simdgroup_load(
                    activation[ct],
                    srcB + (ulong)(c0 + (uint)ct * 8u) * args.stride_b
                         + group * args.K + k0 + (uint)kt * 8u,
                    args.stride_b,
                    ulong2(0, 0),
                    true);
            }
            FOR_UNROLL (short rt = 0; rt < 2; ++rt) {
                simdgroup_float8x8 weight;
                simdgroup_load(
                    weight,
                    shmem + (uint)rt * 8u * 64u + (uint)kt * 8u,
                    64);
                FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
                    simdgroup_multiply_accumulate(
                        acc[rt][ct], weight, activation[ct], acc[rt][ct]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (short rt = 0; rt < 2; ++rt) {
        FOR_UNROLL (short ct = 0; ct < 4; ++ct) {
            const uint column = c0 + (uint)ct * 8u;
            simdgroup_store(
                acc[rt][ct],
                dst + (ulong)column * args.stride_c + group * args.M
                    + r0 + (uint)rt * 8u,
                args.stride_c,
                ulong2(0, 0),
                true);
        }
    }
}

// =============================================================================
// kernel_mat_mat_q8_0_f32_n16 — H5.3b.5.5 NR1=16 specialization for Q8_0.
//
// Mirrors the Q4_K / Q5_K / Q6_K NR1=16 specializations: same tile shape
// (NR0=64 M × NR1=16 N × NK=32 K, 4 sg × 32M×8N each, mc[4]),
// same B-tile layout (`ib = 2*sx + sy`), same B-load gating to first
// 64 threads. Q8_0 dequant + Q8_0_NL=2 pointer-advance.
//
// The DFlash drafter at N=16 verify hits this path; the lift's main
// payoff (per the v0.72.0 batched-tail port and the v0.74-was-now-v0.73b
// pivot) is amortizing the per-token weight reads on Q8_0 lm_head and
// the projections.

constant constexpr int NR1_SPECIAL_N16_Q8 = 16;
constant constexpr int NL1_SPECIAL_N16_Q8 = NK_MM / 8;
constant constexpr int B_LOAD_THREADS_N16_Q8 =
    NR1_SPECIAL_N16_Q8 * NL1_SPECIAL_N16_Q8;

kernel void kernel_mat_mat_q8_0_f32_n16(
        constant mat_mat_q8_0_args & args   [[buffer(0)]],
        device const uchar         * srcA   [[buffer(1)]],
        device const float         * srcB   [[buffer(2)]],
        device       float         * dst    [[buffer(3)]],
        threadgroup  uchar         * shmem  [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_SPECIAL_N16_Q8;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = (short)tiitg / NL1_SPECIAL_N16_Q8;
    const short iy = 8 * (tiitg % NL1_SPECIAL_N16_Q8);

    const short offset1 = il0 / Q8_0_NL;
    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q8_0_BYTES;
    device const float * y_ptr = srcB + (ulong)args.stride_b * lr1
                                       + (ulong)iy;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb;
    simdgroup_float8x8  mc[4];

    FOR_UNROLL (short i = 0; i < 4; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        // PHASE 1: A-tile (Q8_0 dequant).
        {
            half4x4 temp_a;
            dequantize_q8_0_half(x_ptr, il, temp_a);

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
        if (tiitg < B_LOAD_THREADS_N16_Q8) {
            const short sx = (tiitg % NL1_SPECIAL_N16_Q8);
            const short sy = (tiitg / NL1_SPECIAL_N16_Q8) / 8;
            const short ly = (tiitg / NL1_SPECIAL_N16_Q8) % 8;
            const short ib = 2 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        // Q8_0_NL=2 pointer advance: x advances by 1 block per K-step.
        il = il % 2;
        x_ptr = x_ptr + Q8_0_BYTES;
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
            for (int j = tiitg; j < NR1_SPECIAL_N16_Q8; j += NR1_SPECIAL_N16_Q8) {
                device float * D = dst + r0 + (r1 + j) * args.M;
                threadgroup float * C = temp_str + (j * NR0_MM);
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}
