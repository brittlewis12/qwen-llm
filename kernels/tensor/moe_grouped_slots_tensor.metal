// Metal 4 tensor-ops variant of the generic grouped-slots MoE tile
// (kernels/moe.metal kernel_moe_grouped_slots_mm_generic).
//
// Same host contract as the generic kernel so the dispatcher can swap the
// pipeline name and nothing else: 128 threads (4 simdgroups), 8192 B of
// threadgroup memory, grid (ceil(n_tokens / 32), ceil(M / 64), n_expert),
// buffers 0..5 as moe_group_generic_mm_args / weights / activations /
// counts / ids / dst, and buffer 6 = clamp limit for the UpClamped role.
//
// The tile stages the same half operands the generic kernel stages (the
// quant_tiles.h dequant of a 64-row x 32-k weight slab and the half cast
// of a 32-slot x 32-k activation slab) but row-major instead of 8x8
// blocked, and replaces the simdgroup_multiply_accumulate loop with one
// mpp::tensor_ops::matmul2d over threadgroup tensors, accumulating in a
// float cooperative tensor:
//
//   C[slot, row] += X[slot, k] * W[row, k]     (left X, right W transposed)
//
// so the result lands in the [slot][64] layout the per-slot epilogue reads.
// Sums are F32 like the generic tile; the reduction order differs, so
// results are close but not bitwise equal to it.
//
// EPI 0: dst[slot, :]  = W_e * srcB[slot / b_div, :]
// EPI 1: dst[slot, :]  = silu(dst[slot, :]) * (W_e * srcB[slot / b_div, :])
// EPI 2: dst[slot, :]  = silu(min(dst, limit)) * clamp(W_e * srcB, -limit, limit)
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp;

#include "../quant_tiles.h"

#define TT_NR0 64             // weight rows per tile
#define TT_NR1 32             // slots per tile
#define TT_NR1_WIDE 64        // slots per dense wide tile
#define TT_NK  32             // k per step
#define TT_NL0 (TT_NK / 16)   // threads per weight row (one 16-element dequant each)
#define TT_NL1 (TT_NK / 8)    // threads per activation row (one float2x4 each)

// Keep in sync with moe.metal's moe_group_generic_mm_args.
struct moe_group_generic_mm_args {
    uint M;          // output rows per slot
    uint N;          // ids row stride per expert (== n_tokens)
    uint K;          // inner dim
    uint nb01;       // weight row stride in bytes
    uint stride_b;   // activation row stride in f32 elements
    uint min_count;
    uint max_count;
    uint b_div;      // activation row = slot / b_div (1: per-slot input; topk: per-token input)
    uint slot_limit; // number of valid slots (output rows)
};

inline float tt_silu(float x) {
    return x / (1.0f + exp(-x));
}

// DENSE: one weight matrix (grid depth 1), slot == activation row ==
// output row, no counts/ids; args.N = rows, b_div = 1, slot_limit = rows.
template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &),
          short EPI, bool DENSE>
inline void moe_grouped_slots_mm_tensor_body(
        constant moe_group_generic_mm_args & args,
        device const uchar * srcA,
        device const float * srcB,
        device const int   * counts,
        device const int   * ids,
        device       float * dst,
        threadgroup  uchar * shmem,
        uint3  tgpig,
        uint tiitg_wide,
        float limit) {
    const ushort tiitg = ushort(tiitg_wide);
    threadgroup half * sa = (threadgroup half *)(shmem);          // [TT_NR0][TT_NK]
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);   // [TT_NR1][TT_NK]

    const int im = DENSE ? 0 : tgpig.z;
    const int r0 = tgpig.y * TT_NR0;
    const int r1 = tgpig.x * TT_NR1;

    const int count = DENSE ? int(args.N) : counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < TT_NR0) ? (short)((int)args.M - r0) : TT_NR0;
    const short nr1 = (count - r1 < TT_NR1) ? (short)(count - r1) : TT_NR1;

    // Weight staging: thread -> (row lr0, 16-element half il0 of the 32-k slab).
    // Rows past the tile edge clamp to the last valid row (duplicate writes of
    // identical values; those output rows are discarded).
    const short lr0 = ((short)tiitg / TT_NL0) < nr0 ? ((short)tiitg / TT_NL0) : nr0 - 1;
    const short il0 = (tiitg % TT_NL0);
    short il = il0;

    // Activation staging: thread -> (slot lr1, 8-element chunk iy).
    const short lr1 = ((short)tiitg / TT_NL1) < nr1 ? ((short)tiitg / TT_NL1) : nr1 - 1;
    const short iy = 8 * (tiitg % TT_NL1);

    const short offset1 = il0 / NL;
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * BYTES;
    const int slot_limit = int(args.slot_limit);
    const int slot_id = DENSE ? r1 + lr1 : ids[(ulong)im * args.N + (r1 + lr1)];
    const bool slot_valid = slot_id >= 0 && slot_id < slot_limit;
    const int b_row = (slot_valid ? slot_id : 0) / int(args.b_div);
    device const float * y_ptr = srcB + (ulong)args.stride_b * b_row + (ulong)iy;

    threadgroup half * stage_a = sa + lr0 * TT_NK + 16 * il0;
    threadgroup half * stage_b = sb + lr1 * TT_NK + iy;

    // left = activations [32 slots x 32 k], right = weights [64 rows x 32 k]
    // read transposed; destination [32 slots x 64 rows] in registers.
    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        TT_NR1, TT_NR0, TT_NK, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    using left_t = tensor<threadgroup half, extents<int, TT_NK, TT_NR1>, tensor_inline>;
    using right_t = tensor<threadgroup half, extents<int, TT_NK, TT_NR0>, tensor_inline>;
    left_t left(sb, extents<int, TT_NK, TT_NR1>());
    right_t right(sa, extents<int, TT_NK, TT_NR0>());
    auto acc = op.template get_destination_cooperative_tensor<left_t, right_t, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        acc[i] = 0.0f;
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += TT_NK) {
        half4x4 temp_a;
        DEQ(x_ptr, il, temp_a);
        const half2x4 yv = (half2x4)(*((device const float2x4 *)y_ptr));

        // The previous step's matmul must be done reading the slabs.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        *(threadgroup half4 *)(stage_a + 0)  = temp_a[0];
        *(threadgroup half4 *)(stage_a + 4)  = temp_a[1];
        *(threadgroup half4 *)(stage_a + 8)  = temp_a[2];
        *(threadgroup half4 *)(stage_a + 12) = temp_a[3];
        *(threadgroup half2x4 *)(stage_b) = yv;

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + BYTES * ((2 + NL - 1) / NL) : x_ptr;
        y_ptr += TT_NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left, right, acc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = (threadgroup float *)shmem;   // [TT_NR1][TT_NR0]
    tensor<threadgroup float, extents<int, TT_NR0, TT_NR1>, tensor_inline> out_tile(
        temp_str, extents<int, TT_NR0, TT_NR1>());
    acc.store(out_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: 4 threads per slot, 16 consecutive rows each.
    const short j = tiitg / 4;
    const short c0 = (tiitg % 4) * 16;
    if (j < nr1) {
        const int slot = DENSE ? r1 + j : ids[(ulong)im * args.N + r1 + j];
        if (slot >= 0 && slot < slot_limit) {
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup const float * C = temp_str + j * TT_NR0;
            const short c1 = (c0 + 16 < nr0) ? c0 + 16 : nr0;
            for (short i = c0; i < c1; ++i) {
                if (EPI == 0) {
                    D[i] = C[i];
                } else if (EPI == 2) {
                    D[i] = tt_silu(min(D[i], limit)) * clamp(C[i], -limit, limit);
                } else {
                    D[i] = tt_silu(D[i]) * C[i];
                }
            }
        }
    }
}

template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &),
          short EPI>
kernel void kernel_moe_grouped_slots_mm_tensor(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]]) {
    moe_grouped_slots_mm_tensor_body<BYTES, NL, DEQ, EPI, false>(
        args, srcA, srcB, counts, ids, dst, shmem, tgpig, tiitg_wide, 0.0f);
}

template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void kernel_moe_up_clamped_grouped_slots_tensor(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        constant float & limit            [[buffer(6)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]]) {
    moe_grouped_slots_mm_tensor_body<BYTES, NL, DEQ, 2, false>(
        args, srcA, srcB, counts, ids, dst, shmem, tgpig, tiitg_wide, limit);
}

// Dense mat-mat: dst[row, :] = W * src[row, :] for args.N rows (the
// mat_mat contract: x [rows x K] F32, y [rows x M] F32). Grid
// (ceil(rows / 32), ceil(M / 64), 1).
template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void kernel_mat_mat_tensor(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device       float * dst          [[buffer(3)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]]) {
    moe_grouped_slots_mm_tensor_body<BYTES, NL, DEQ, 0, true>(
        args, srcA, srcB, nullptr, nullptr, dst, shmem, tgpig, tiitg_wide, 0.0f);
}

// Dense 64-slot form: one 64-row weight slab is reused for two 32-slot
// activation groups, halving the weight dequant/staging work per output row.
// Dense only (EPI 0); the output tile grows to [64][64] F32 (16 KiB shmem).
template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &)>
inline void moe_mat_mat_tensor_wide_body(
        constant moe_group_generic_mm_args & args,
        device const uchar * srcA,
        device const float * srcB,
        device       float * dst,
        threadgroup  uchar * shmem,
        uint3  tgpig,
        uint tiitg_wide) {
    const ushort tiitg = ushort(tiitg_wide);
    threadgroup half * sa = (threadgroup half *)(shmem);          // [64][32]
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);   // [64][32]

    const int r0 = tgpig.y * TT_NR0;
    const int r1 = tgpig.x * TT_NR1_WIDE;
    const int count = int(args.N);
    if (r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < TT_NR0) ? (short)((int)args.M - r0) : TT_NR0;
    const short nr1 = (count - r1 < TT_NR1_WIDE) ? (short)(count - r1) : TT_NR1_WIDE;

    const short lr0 = ((short)tiitg / TT_NL0) < nr0 ? ((short)tiitg / TT_NL0) : nr0 - 1;
    const short il0 = (tiitg % TT_NL0);
    short il = il0;

    const short lr1 = (short)tiitg / TT_NL1; // 0..31, two slots per thread
    const short iy = 8 * (tiitg % TT_NL1);
    const short slot0 = (short)min((int)lr1, (int)nr1 - 1);
    const short slot1 = (short)min((int)lr1 + 32, (int)nr1 - 1);
    const short phys0 = lr1;
    const short phys1 = lr1 + 32;
    const bool valid0 = lr1 < nr1;
    const bool valid1 = (int)lr1 + 32 < (int)nr1;

    const short offset1 = il0 / NL;
    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * BYTES;
    device const float * y_ptr0 = srcB + (ulong)args.stride_b * (r1 + slot0) + (ulong)iy;
    device const float * y_ptr1 = srcB + (ulong)args.stride_b * (r1 + slot1) + (ulong)iy;

    threadgroup half * stage_a = sa + lr0 * TT_NK + 16 * il0;
    threadgroup half * stage_b0 = sb + phys0 * TT_NK + iy;
    threadgroup half * stage_b1 = sb + phys1 * TT_NK + iy;

    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        TT_NR1_WIDE, TT_NR0, TT_NK, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    using left_t = tensor<threadgroup half, extents<int, TT_NK, TT_NR1_WIDE>, tensor_inline>;
    using right_t = tensor<threadgroup half, extents<int, TT_NK, TT_NR0>, tensor_inline>;
    left_t left(sb, extents<int, TT_NK, TT_NR1_WIDE>());
    right_t right(sa, extents<int, TT_NK, TT_NR0>());
    auto acc = op.template get_destination_cooperative_tensor<left_t, right_t, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        acc[i] = 0.0f;
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += TT_NK) {
        half4x4 temp_a;
        DEQ(x_ptr, il, temp_a);
        const half2x4 yv0 = (half2x4)(*((device const float2x4 *)y_ptr0));
        const half2x4 yv1 = (half2x4)(*((device const float2x4 *)y_ptr1));

        threadgroup_barrier(mem_flags::mem_threadgroup);
        *(threadgroup half4 *)(stage_a + 0)  = temp_a[0];
        *(threadgroup half4 *)(stage_a + 4)  = temp_a[1];
        *(threadgroup half4 *)(stage_a + 8)  = temp_a[2];
        *(threadgroup half4 *)(stage_a + 12) = temp_a[3];
        if (valid0) *(threadgroup half2x4 *)(stage_b0) = yv0;
        else *(threadgroup half2x4 *)(stage_b0) = half2x4(0.0h);
        if (valid1) *(threadgroup half2x4 *)(stage_b1) = yv1;
        else *(threadgroup half2x4 *)(stage_b1) = half2x4(0.0h);

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + BYTES * ((2 + NL - 1) / NL) : x_ptr;
        y_ptr0 += TT_NK;
        y_ptr1 += TT_NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left, right, acc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = (threadgroup float *)shmem;   // [64][64]
    tensor<threadgroup float, extents<int, TT_NR0, TT_NR1_WIDE>, tensor_inline> out_tile(
        temp_str, extents<int, TT_NR0, TT_NR1_WIDE>());
    acc.store(out_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: 4 threads per slot, 16 rows each, two slot groups.
    const short j = tiitg / 4;
    const short c0 = (tiitg % 4) * 16;
#pragma clang loop unroll(full)
    for (short s = 0; s < TT_NR1_WIDE; s += 32) {
        const short js = j + s;
        if (js < nr1) {
            const int slot = r1 + js;
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup const float * C = temp_str + js * TT_NR0;
            const short c1 = (c0 + 16 < nr0) ? c0 + 16 : nr0;
            for (short i = c0; i < c1; ++i) {
                D[i] = C[i];
            }
        }
    }
}

template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void kernel_mat_mat_tensor_wide(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device       float * dst          [[buffer(3)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]]) {
    moe_mat_mat_tensor_wide_body<BYTES, NL, DEQ>(
        args, srcA, srcB, dst, shmem, tgpig, tiitg_wide);
}

// Dense 64-slot x 128-row form: a 128-row weight slab with two 32-slot
// activation groups. Against the 32-slot narrow tile it halves both weight
// and activation reads per output row; against the 64-slot wide tile it
// keeps the weight reuse and halves the activation reads. Dense only
// (EPI 0); the [64][128] F32 output staging is 32 KiB.
template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &)>
inline void moe_mat_mat_tensor_combined_body(
        constant moe_group_generic_mm_args & args,
        device const uchar * srcA,
        device const float * srcB,
        device       float * dst,
        threadgroup  uchar * shmem,
        uint3  tgpig,
        uint tiitg_wide) {
    const ushort tiitg = ushort(tiitg_wide);
    threadgroup half * sa = (threadgroup half *)(shmem);          // [128][32]
    threadgroup half * sb = (threadgroup half *)(shmem + 8192);   // [64][32]

    const int r0 = tgpig.y * 128;
    const int r1 = tgpig.x * 64;
    const int count = int(args.N);
    if (r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < 128) ? (short)((int)args.M - r0) : 128;
    const short nr1 = (count - r1 < 64) ? (short)(count - r1) : 64;

    const short lr0a = (short)tiitg / TT_NL0;
    const short lr0b = lr0a + 64;
    const short il0 = (tiitg % TT_NL0);
    short il = il0;
    const short rowa = lr0a < nr0 ? lr0a : nr0 - 1;
    const short rowb = lr0b < nr0 ? lr0b : nr0 - 1;

    const short lr1 = (short)tiitg / TT_NL1; // 0..31, two slots per thread
    const short iy = 8 * (tiitg % TT_NL1);
    const short slot0 = (short)min((int)lr1, (int)nr1 - 1);
    const short slot1 = (short)min((int)lr1 + 32, (int)nr1 - 1);
    const short phys0 = lr1;
    const short phys1 = lr1 + 32;
    const bool valid0 = lr1 < nr1;
    const bool valid1 = (int)lr1 + 32 < (int)nr1;

    const short offset1 = il0 / NL;
    device const uchar * x_ptr0 = srcA + (ulong)args.nb01 * (r0 + rowa)
                                        + (ulong)offset1 * BYTES;
    device const uchar * x_ptr1 = srcA + (ulong)args.nb01 * (r0 + rowb)
                                        + (ulong)offset1 * BYTES;
    device const float * y_ptr0 = srcB + (ulong)args.stride_b * (r1 + slot0) + (ulong)iy;
    device const float * y_ptr1 = srcB + (ulong)args.stride_b * (r1 + slot1) + (ulong)iy;

    threadgroup half * stage_a0 = sa + rowa * TT_NK + 16 * il0;
    threadgroup half * stage_a1 = sa + rowb * TT_NK + 16 * il0;
    threadgroup half * stage_b0 = sb + phys0 * TT_NK + iy;
    threadgroup half * stage_b1 = sb + phys1 * TT_NK + iy;

    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        64, 128, TT_NK, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    using left_t = tensor<threadgroup half, extents<int, TT_NK, 64>, tensor_inline>;
    using right_t = tensor<threadgroup half, extents<int, TT_NK, 128>, tensor_inline>;
    left_t left(sb, extents<int, TT_NK, 64>());
    right_t right(sa, extents<int, TT_NK, 128>());
    auto acc = op.template get_destination_cooperative_tensor<left_t, right_t, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        acc[i] = 0.0f;
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += TT_NK) {
        half4x4 ta0, ta1;
        DEQ(x_ptr0, il, ta0);
        DEQ(x_ptr1, il, ta1);
        const half2x4 yv0 = (half2x4)(*((device const float2x4 *)y_ptr0));
        const half2x4 yv1 = (half2x4)(*((device const float2x4 *)y_ptr1));

        threadgroup_barrier(mem_flags::mem_threadgroup);
        *(threadgroup half4 *)(stage_a0 + 0)  = ta0[0];
        *(threadgroup half4 *)(stage_a0 + 4)  = ta0[1];
        *(threadgroup half4 *)(stage_a0 + 8)  = ta0[2];
        *(threadgroup half4 *)(stage_a0 + 12) = ta0[3];
        *(threadgroup half4 *)(stage_a1 + 0)  = ta1[0];
        *(threadgroup half4 *)(stage_a1 + 4)  = ta1[1];
        *(threadgroup half4 *)(stage_a1 + 8)  = ta1[2];
        *(threadgroup half4 *)(stage_a1 + 12) = ta1[3];
        if (valid0) *(threadgroup half2x4 *)(stage_b0) = yv0;
        else *(threadgroup half2x4 *)(stage_b0) = half2x4(0.0h);
        if (valid1) *(threadgroup half2x4 *)(stage_b1) = yv1;
        else *(threadgroup half2x4 *)(stage_b1) = half2x4(0.0h);

        il = (il + 2 < NL) ? il + 2 : il % 2;
        if (il < 2) {
            x_ptr0 += BYTES * ((2 + NL - 1) / NL);
            x_ptr1 += BYTES * ((2 + NL - 1) / NL);
        }
        y_ptr0 += TT_NK;
        y_ptr1 += TT_NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left, right, acc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = (threadgroup float *)shmem;   // [64][128]
    tensor<threadgroup float, extents<int, 128, 64>, tensor_inline> out_tile(
        temp_str, extents<int, 128, 64>());
    acc.store(out_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short j = tiitg / 4;
    const short c0 = (tiitg % 4) * 32;
#pragma clang loop unroll(full)
    for (short s = 0; s < 64; s += 32) {
        const short js = j + s;
        if (js < nr1) {
            const int slot = r1 + js;
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup const float * C = temp_str + js * 128;
            const short c1 = (c0 + 32 < nr0) ? c0 + 32 : nr0;
            for (short i = c0; i < c1; ++i) {
                D[i] = C[i];
            }
        }
    }
}

template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void kernel_mat_mat_tensor_combined(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device       float * dst          [[buffer(3)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]]) {
    moe_mat_mat_tensor_combined_body<BYTES, NL, DEQ>(
        args, srcA, srcB, dst, shmem, tgpig, tiitg_wide);
}

typedef decltype(kernel_moe_grouped_slots_mm_tensor<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K, 0>) moe_grouped_slots_mm_tensor_t;
typedef decltype(kernel_moe_up_clamped_grouped_slots_tensor<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K>) moe_up_clamped_grouped_slots_tensor_t;
typedef decltype(kernel_mat_mat_tensor<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K>) mat_mat_tensor_t;

// One down / up_silu_mul / up_clamped / dense quadruple per dtype. Keep the
// list in sync with MOE_GROUPED_GENERIC_LAYOUTS (a CPU test parses it).
#define TT_INSTANTIATE(suffix, BYTES, NL, DEQ) \
    template [[host_name("kernel_moe_down_" #suffix "_f32_grouped_slots_tensor")]] kernel moe_grouped_slots_mm_tensor_t kernel_moe_grouped_slots_mm_tensor<BYTES, NL, DEQ, 0>; \
    template [[host_name("kernel_moe_up_silu_mul_" #suffix "_f32_grouped_slots_tensor")]] kernel moe_grouped_slots_mm_tensor_t kernel_moe_grouped_slots_mm_tensor<BYTES, NL, DEQ, 1>; \
    template [[host_name("kernel_moe_up_clamped_" #suffix "_f32_grouped_slots_tensor")]] kernel moe_up_clamped_grouped_slots_tensor_t kernel_moe_up_clamped_grouped_slots_tensor<BYTES, NL, DEQ>; \
    template [[host_name("kernel_mat_mat_" #suffix "_f32_tensor")]] kernel mat_mat_tensor_t kernel_mat_mat_tensor<BYTES, NL, DEQ>;

TT_INSTANTIATE(q2_0,    QT_Q2_0_BYTES,    QT_Q2_0_NL,    qt_dequantize_q2_0)
TT_INSTANTIATE(q2_K,    QT_Q2_K_BYTES,    QT_Q2_K_NL,    qt_dequantize_q2_K)
TT_INSTANTIATE(q3_K,    QT_Q3_K_BYTES,    QT_Q3_K_NL,    qt_dequantize_q3_K)
TT_INSTANTIATE(q4_K,    QT_Q4_K_BYTES,    QT_Q4_K_NL,    qt_dequantize_q4_K)
TT_INSTANTIATE(q5_K,    QT_Q5_K_BYTES,    QT_Q5_K_NL,    qt_dequantize_q5_K)
TT_INSTANTIATE(q6_K,    QT_Q6_K_BYTES,    QT_Q6_K_NL,    qt_dequantize_q6_K)
TT_INSTANTIATE(q8_0,    QT_Q8_0_BYTES,    QT_Q8_0_NL,    qt_dequantize_q8_0)
TT_INSTANTIATE(q4_0,    QT_Q4_0_BYTES,    QT_Q4_0_NL,    qt_dequantize_q4_0)
TT_INSTANTIATE(q4_1,    QT_Q4_1_BYTES,    QT_Q4_1_NL,    qt_dequantize_q4_1)
TT_INSTANTIATE(iq2_s,   QT_IQ2_S_BYTES,   QT_IQ2_S_NL,   qt_dequantize_iq2_s)
TT_INSTANTIATE(iq3_xxs, QT_IQ3_XXS_BYTES, QT_IQ3_XXS_NL, qt_dequantize_iq3_xxs)
TT_INSTANTIATE(iq3_s,   QT_IQ3_S_BYTES,   QT_IQ3_S_NL,   qt_dequantize_iq3_s)
TT_INSTANTIATE(iq4_nl,  QT_IQ4_NL_BYTES,  QT_IQ4_NL_NL,  qt_dequantize_iq4_nl)
TT_INSTANTIATE(iq4_xs,  QT_IQ4_XS_BYTES,  QT_IQ4_XS_NL,  qt_dequantize_iq4_xs)
TT_INSTANTIATE(f32,     QT_F32_BYTES,     QT_F32_NL,     qt_dequantize_f32)
TT_INSTANTIATE(f16,     QT_F16_BYTES,     QT_F16_NL,     qt_dequantize_f16)
TT_INSTANTIATE(bf16,    QT_BF16_BYTES,    QT_BF16_NL,    qt_dequantize_bf16)
TT_INSTANTIATE(mxfp4,   QT_MXFP4_BYTES,   QT_MXFP4_NL,   qt_dequantize_mxfp4)
TT_INSTANTIATE(iq2_xs,  QT_IQ2_XS_BYTES,  QT_IQ2_XS_NL,  qt_dequantize_iq2_xs)
TT_INSTANTIATE(iq2_xxs, QT_IQ2_XXS_BYTES, QT_IQ2_XXS_NL, qt_dequantize_iq2_xxs)

// The wide dense tile is instantiated for the Q8_0 projections V4.1 uses
// (q/kv/output and the shared expert); other dtypes keep the 32-slot form.
typedef decltype(kernel_mat_mat_tensor_wide<QT_Q8_0_BYTES, QT_Q8_0_NL, qt_dequantize_q8_0>) mat_mat_tensor_wide_t;
template [[host_name("kernel_mat_mat_q8_0_f32_tensor_wide")]] kernel mat_mat_tensor_wide_t kernel_mat_mat_tensor_wide<QT_Q8_0_BYTES, QT_Q8_0_NL, qt_dequantize_q8_0>;
typedef decltype(kernel_mat_mat_tensor_combined<QT_Q8_0_BYTES, QT_Q8_0_NL, qt_dequantize_q8_0>) mat_mat_tensor_combined_t;
template [[host_name("kernel_mat_mat_q8_0_f32_tensor_combined")]] kernel mat_mat_tensor_combined_t kernel_mat_mat_tensor_combined<QT_Q8_0_BYTES, QT_Q8_0_NL, qt_dequantize_q8_0>;
