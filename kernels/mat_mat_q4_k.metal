// Q4_K mat-mat (W · Xᵀ → Yᵀ).
//
// kernel_mat_mat_q4_K_f32: Q4_K weight × F32 activations → F32 output, where
// the activation matrix has N_QUERY columns (N=16 in the H5.3b DFlash use
// case; arbitrary at the kernel level). Lifts the 64×32×32 simdgroup_matrix
// tile from llama.cpp `kernel_mul_mm_q4_K_f32` (ggml-metal.metal:9440-9648,
// non-MPS-tensor classic path).
//
// Why a separate kernel from `kernel_mat_vec_q4_K_f32`:
//   At N_QUERY > 1 the bandwidth ceiling for matvec-N-times collapses
//   because each weight super-block is re-read N times from device memory.
//   The mat-mat tile loads a 64-row × 32-K-element weight panel into
//   threadgroup memory ONCE per K-step then reuses it across all N output
//   columns in the tile. Total weight traffic: M*K bytes (vs N*M*K for
//   N-times-mat-vec). For Qwen3.6-27B FFN gate at N=16 that's 47 MiB
//   per layer per outer step instead of 752 MiB — a ~16× weight-BW win
//   that's the entire point of H5.3b.
//
// Tile geometry (lifted verbatim from llama):
//   NR0 = 64    M-tile (output rows per threadgroup)
//   NR1 = 32    N-tile (output cols per threadgroup; > our N=16 → half-fill)
//   NK  = 32    K-step (elements of K per outer-loop iteration)
//   threadgroup memory: 4096 B for sa (dequant'd weight tile, half),
//                       4096 B for sb (activation tile, half)
//   simdgroups per TG: 4 (= NR0/16 × NR1/16 in 8x8 simd matrices)
//   threads per TG: 4 × 32 = 128
//
// The mat-mat is NOT bit-exact with N successive mat-vec calls (codex Q3
// correction). Lifted kernel stages:
//   * weight: Q4_K → dequant_q4_K → half (in shmem)
//   * activations: float → half (in shmem) before simdgroup_multiply_accumulate
//   * accumulator: float (simdgroup_float8x8)
// The intermediate cast to half adds rounding noise on order of 2^-10 ≈ 1e-3
// per multiply-accumulate. Final per-element max|Δ| typically < 1e-3 vs
// scalar-float mat-vec; cosine ≥ 0.999 is the gate.
//
// Output layout:
//   The kernel writes `dst[r + c * M]` per llama. Notation in the
//   lifted kernel calls this "column-major [M, N]". CRITICALLY: this
//   is BIT-EQUIVALENT to row-major `[N, M]` — the flat byte offset
//   for cell `(r, c)` of [M, N] col-major equals the flat offset for
//   cell `(c, r)` of [N, M] row-major (both are `r + c*M`).
//
//   So downstream consumers should treat this buffer as row-major
//   `[N, M]` (= `[n_query, n_out]`). FFN silu_mul, residual_add,
//   and chained mat-mat all work without any transpose — the layouts
//   compose naturally because the bytes ARE in the row-major order
//   that the next-layer mat-mat call expects as its `srcB`.
//
//   The H5.3b.0 layout sanity test verifies this equivalence at three
//   corner cells; the chained layer-major path in H5.3b.4-5 relies on
//   it. Codex Q7 failure-mode pitfall (potential transpose mismatch)
//   does NOT apply at our specific tile shape because llama's
//   "col-major" naming corresponds bit-for-bit to our "row-major"
//   storage convention.
//
// Q4_K block layout (block_q4_K, 144 bytes / 256 elements):
//   half  d
//   half  dmin
//   uchar scales[12]    // 6-bit packed (sc, min) for 8 sub-blocks
//   uchar qs[128]       // 4-bit nibbles, paired sub-blocks per byte

#include <metal_stdlib>
using namespace metal;

constant constexpr int   QK_K           = 256;
constant constexpr int   Q4K_BYTES      = 144;
constant constexpr int   Q4K_NL         = QK_K / 16;     // 16 dequant calls per super-block

constant constexpr int   NR0_MM         = 64;
constant constexpr int   NR1_MM         = 32;
constant constexpr int   NK_MM          = 32;
constant constexpr int   NL0_MM         = NK_MM / 16;    // 2 dequant calls per K-step per row
constant constexpr int   NL1_MM         = NK_MM / 8;     // 4 activation chunks per K-step per col
constant constexpr int   NW_MM          = 32;            // simdgroup width
constant constexpr int   N_SIMD_GROUPS  = 4;             // simdgroups per TG
constant constexpr int   N_THREADS_MM   = NW_MM * N_SIMD_GROUPS; // 128

struct mat_mat_q4k_args {
    uint  M;          // output rows == n_out (= weight rows)
    uint  N;          // output cols == n_query (= activation cols)
    uint  K;          // inner dim   == n_in   (= weight cols == activation rows)
    // Weight strides in BYTES (block_q4_K rows are not float-strided):
    //   nb01 = bytes per weight row = (K / QK_K) * Q4K_BYTES
    uint  nb01;
    // Activation strides in ELEMENTS (F32):
    //   stride_b = K (when activations are row-major [N, K] passed as `[K * N]`)
    //   nb11_elems = K (activation row stride when src1 is [N, K] row-major;
    //                   matches llama's `nb11 / sizeof(T1)` term)
    uint  stride_b;
};

// ---------------------------------------------------------------------------
// Q4_K dequant — produces 16 half values into a half4x4 register tile.
//
// `il ∈ [0, 16)` selects which 16-element sub-tile of the 256-element
// super-block to dequantize. Layout matches llama:
//   * super-block has 8 sub-blocks of 32 elements each, stored as 4 packed
//     pairs of 32 nibbles (low + high) per pair.
//   * il / 4 picks which 64-element pair (0..3); il & 1 picks low or high
//     32 within the pair; the result is one 16-element slice (further
//     halved by the (il & 3) modulo selecting which of 4 such slices).
//
// Algorithm matches llama's `dequantize_q4_K`:
//     value[i] = (d * sc) * (q & mask) - (dmin * min)
inline void dequantize_q4_K_half(device const uchar * blk_bytes,
                                 short il,
                                 thread half4x4 & reg) {
    const half d_h    = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar * scales = blk_bytes + 4;
    device const uchar * qs     = blk_bytes + 4 + 12;

    // get_scale_min_k4_just2 unrolled inline for the (is, k01) we need.
    // Note: llama masks `il &= 3` before calling get_scale_min_k4_just2(is,
    // il/2, ...), so the second arg `k` is only ∈ {0, 1}, not 0..7. Our
    // `k01 = (il/2) & 1` matches that masking. Renamed from `j` per codex
    // mid-implementation review — `j` collided with llama's outer-loop
    // `j` symbol and was confusing on read.
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
// kernel_mat_mat_q4_K_f32
//
// Computes Y = W · Xᵀ where W is Q4_K [M, K], X is F32 [N, K] row-major,
// Y is F32 [M, N] COLUMN-major (i.e. dst[row + col * M]).
//
// Threadgroup grid: (cols / NR1_MM, rows / NR0_MM, 1).
//
// Notes on the lift from llama:
//   * llama supports batched matmul (im / args.r2 / args.r3 broadcast);
//     we drop that — single batch, im = 0 always. Saves args + a few
//     index ops in the hot path.
//   * llama supports variable input dtypes T0/T1 via templates. We
//     specialize to {Q4_K weight, F32 activations, F32 output}. Saves
//     template churn + makes the kernel compilable with -O3 -ffast-math
//     without metaprogramming.
//   * llama gates a `FC_mul_mm_bc_inp` function constant for unaligned-
//     K element-wise reads. We always have K aligned to 256 (Q4_K
//     super-block size); skip the gate.
//   * llama gates `FC_mul_mm_bc_out` for partial-output-tile write
//     paths. We ALWAYS take the partial-tile path when the tile is
//     not fully in-bounds — codex mid-impl review confirmed this is
//     correct: the partial path stages the 64×32 output tile via
//     threadgroup memory (8 KiB total, REUSING the sa+sb shmem after
//     all matmul reads are complete) then has lane 0 of simdgroup 0
//     copy only the in-bounds rows × cols out to device. The unused
//     output cells are not written; downstream sees only the live
//     `M × N` cells.
//
// Codex Q4 (i) + Q5: for N_QUERY < NR1_MM (e.g. our DFlash N=16),
// the host passes the actual `n_query` and the kernel takes the
// partial-tile path. NO host-level tile-padding required; the
// kernel handles N < 32 internally. The output buffer can be
// allocated at exactly `n_out × n_query` F32 elements.
kernel void kernel_mat_mat_q4_K_f32(
        constant mat_mat_q4k_args & args   [[buffer(0)]],
        device const uchar        * srcA   [[buffer(1)]], // Q4_K weight bytes [M, K]
        device const float        * srcB   [[buffer(2)]], // F32 activations [N, K] row-major
        device       float        * dst    [[buffer(3)]], // F32 output [M, N] col-major
        threadgroup  uchar        * shmem  [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    // Threadgroup memory partition (matches llama: 4 KiB sa, 4 KiB sb).
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    // Tile origin in output matrix.
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    // Bounds for partial tiles.
    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;
    const short nr1 = ((int)args.N - r1 < NR1_MM) ? (short)((int)args.N - r1) : NR1_MM;

    // A-loading thread mapping (matches llama):
    //   lr0 = tiitg / NL0  ∈ [0, NR0_MM=64), the row this thread loads
    //   il0 = tiitg % NL0  ∈ [0, NL0=2),     the K-chunk-of-16 within the K-step
    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    // B-loading thread mapping:
    //   lr1 = tiitg / NL1 ∈ [0, NR1_MM=32) row in N
    //   iy  = 8 * (tiitg % NL1) ∈ {0, 8, 16, 24} K-chunk-of-8 within K-step
    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    // Pointer to first super-block of A row (lr0).
    // offset within super-block = il0 / nl  (nl = QK_K/16 = 16 for Q4_K).
    const short offset1 = il0 / Q4K_NL; // always 0 for Q4_K, NL0=2, but keep for clarity
    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q4K_BYTES;

    // Pointer to activation row (r1 + lr1, K-offset iy).
    // srcB is row-major [N, K]; nb11_elems = stride_b = K.
    device const float * y_ptr = srcB + (ulong)args.stride_b * (r1 + lr1)
                                       + (ulong)iy;

    // Per-simdgroup register accumulator: 8 simdgroup_float8x8 tiles per SG.
    // 4 simdgroups × 8 tiles = 32 tiles total = 32 × (8×8) = 64×32 output tile.
    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb[2];
    simdgroup_float8x8  mc[8];

    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        // === PHASE 1: load + dequant A tile into sa[64 × 32 halves] ===
        {
            half4x4 temp_a;
            dequantize_q4_K_half(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Pack temp_a (16 halves) into sa with the swizzled layout
            // llama uses for simdgroup_load to consume directly.
            //
            // Layout: each 8×8 simdgroup tile occupies 64 contiguous halves
            // in sa. The 32-K × 64-M tile is stored as 8 (4-of-K × 2-of-M)
            // simdgroup tiles in row-major order over (sx, sy):
            //     sx ∈ [0, 4) = K-tile-of-8 within the 32-K step
            //     sy ∈ [0, 8) = M-tile-of-8 within the 64-M tile
            // Per llama: the (lx, ly) within each tile come from the lane
            // mapping into the 16 halves we just dequantized.
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        // === PHASE 2: load B tile into sb[32 × 32 halves] ===
        {
            // Read 8 contiguous K-elements as a half2x4 (8 halves), cast
            // from float2x4. llama assumes alignment; we have it because
            // K is multiple of 32 and stride_b=K.
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        // Advance pointers: A by ONE super-block (Q4K_BYTES) when il
        // wraps back to 0/1; B by NK_MM F32 elements every K-step.
        //
        // BUG FIX (codex mid-impl review): llama types `x` as
        // `device const block_q4_K *`, so its `x + ((2 + nl - 1) / nl)`
        // advances by N × sizeof(block_q4_K) = N × 144 bytes. Our
        // `x_ptr` is `uchar *`, so we MUST multiply by Q4K_BYTES
        // explicitly. Without this, the second K-step reads garbage
        // from device memory (offset by 1 byte, not 144 bytes).
        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 3: simdgroup matmul outer products ===
        // Each simdgroup processes its slice of the 64×32 output tile.
        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
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

    // === PHASE 4: store mc[] to dst[M, N] col-major ===
    if (r0 + NR0_MM <= (int)args.M && r1 + NR1_MM <= (int)args.N) {
        // Whole tile in-bounds: direct write.
        device float * C = dst + (r0 + 32 * (sgitg & 1))
                               + (r1 + 16 * (sgitg >> 1)) * args.M;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.M * (i / 4),
                            args.M, 0, false);
        }
    } else {
        // Partial tile: stage into shmem, per-row copy with bounds check.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shmem)
                                       + 32 * (sgitg & 1)
                                       + (16 * (sgitg >> 1)) * NR0_MM;
        for (short i = 0; i < 8; ++i) {
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
