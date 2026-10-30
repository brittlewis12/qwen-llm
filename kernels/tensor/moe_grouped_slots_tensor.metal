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
// HALF_ACT: srcB holds half activations (converted once by the caller)
// instead of F32; the staged operand is the same half value either way.
template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &),
          short EPI, bool DENSE, bool HALF_ACT>
inline void moe_grouped_slots_mm_tensor_body(
        constant moe_group_generic_mm_args & args,
        device const uchar * srcA,
        device const uchar * srcB_bytes,
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
    if (count < 0 || uint(count) > args.N
        || count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < TT_NR0) ? (short)((int)args.M - r0) : TT_NR0;
    const short nr1 = (count - r1 < TT_NR1) ? (short)(count - r1) : TT_NR1;

    // Clamp source rows only; every physical fragment has one writer, including
    // zero-filled tails consumed by the full tensor operation.
    const short pr0 = (short)tiitg / TT_NL0;
    const short lr0 = pr0 < nr0 ? pr0 : nr0 - 1;
    const short il0 = (tiitg % TT_NL0);
    short il = il0;

    // Activation staging: thread -> (physical slot pr1, 8-element chunk iy).
    const short pr1 = (short)tiitg / TT_NL1;
    const short lr1 = pr1 < nr1 ? pr1 : nr1 - 1;
    const short iy = 8 * (tiitg % TT_NL1);

    const short offset1 = il0 / NL;
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * BYTES;
    const int slot_limit = int(args.slot_limit);
    const int slot_id = DENSE ? r1 + lr1 : ids[(ulong)im * args.N + (r1 + lr1)];
    const bool slot_valid = pr1 < nr1 && slot_id >= 0 && slot_id < slot_limit;
    const int b_row = (slot_valid ? slot_id : 0) / int(args.b_div);
    const ulong y_off = (ulong)args.stride_b * b_row + (ulong)iy;
    device const float * y_ptr = (device const float *)srcB_bytes + y_off;
    device const half * yh_ptr = (device const half *)srcB_bytes + y_off;

    threadgroup half * stage_a = sa + pr0 * TT_NK + 16 * il0;
    threadgroup half * stage_b = sb + pr1 * TT_NK + iy;
    // Padded tails and invalid slots keep their pre-zeroed fragments; full
    // fragments are rewritten every K iteration.
    if (pr0 >= nr0) {
        *(threadgroup half4 *)(stage_a + 0)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 4)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 8)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 12) = half4(0.0h, 0.0h, 0.0h, 0.0h);
    }
    if (!slot_valid) {
        *(threadgroup half2x4 *)(stage_b) = half2x4(0.0h);
    }

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
        const half2x4 yv = HALF_ACT ? *((device const half2x4 *)yh_ptr)
                                    : (half2x4)(*((device const float2x4 *)y_ptr));

        // The previous step's matmul must be done reading the slabs.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (pr0 < nr0) {
            *(threadgroup half4 *)(stage_a + 0)  = temp_a[0];
            *(threadgroup half4 *)(stage_a + 4)  = temp_a[1];
            *(threadgroup half4 *)(stage_a + 8)  = temp_a[2];
            *(threadgroup half4 *)(stage_a + 12) = temp_a[3];
        }
        if (slot_valid) {
            *(threadgroup half2x4 *)(stage_b) = yv;
        }

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + BYTES * ((2 + NL - 1) / NL) : x_ptr;
        y_ptr += TT_NK;
        yh_ptr += TT_NK;

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

// 64-slot gate/up tile. One 64-slot panel per dispatch, same K order and
// epilogue as the narrow tile; `epi` selects the down/gate, up_silu_mul or
// clamped-up epilogue. The launch size is n_tokens/64 x M/64 x experts and
// the threadgroup is 16 KiB.
template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void kernel_moe_wide_probe_f16act(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const half  * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        constant float & limit            [[buffer(6)]],
        constant int   & epi              [[buffer(7)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);          // [64][32]
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);   // [64][32]
    const int im = tgpig.z;
    const int r0 = tgpig.y * TT_NR0;
    const int r1 = tgpig.x * TT_NR1_WIDE;
    const int count = counts[im];
    if (count < 0 || uint(count) > args.N
        || count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = min(int(args.M) - r0, TT_NR0);
    const short nr1 = min(count - r1, TT_NR1_WIDE);
    const short pr0 = tiitg / TT_NL0;
    const short lr0 = min(int(pr0), int(nr0) - 1);
    const short il0 = tiitg % TT_NL0;
    short il = il0;
    const short lr1 = tiitg / TT_NL1; // 0..31
    const short iy = 8 * (tiitg % TT_NL1);
    const short slot0 = min(int(lr1), int(nr1) - 1);
    const short slot1 = min(int(lr1) + 32, int(nr1) - 1);
    const bool valid0 = lr1 < nr1;
    const bool valid1 = lr1 + 32 < nr1;

    const int slot_limit = int(args.slot_limit);
    const int id0 = ids[(ulong)im * args.N + r1 + slot0];
    const int id1 = ids[(ulong)im * args.N + r1 + slot1];
    const bool ok0 = valid0 && id0 >= 0 && id0 < slot_limit;
    const bool ok1 = valid1 && id1 >= 0 && id1 < slot_limit;
    const int row0 = (ok0 ? id0 : 0) / int(args.b_div);
    const int row1 = (ok1 ? id1 : 0) / int(args.b_div);
    device const half * y0 = srcB + (ulong)args.stride_b * row0 + iy;
    device const half * y1 = srcB + (ulong)args.stride_b * row1 + iy;

    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im
        + (ulong)args.nb01 * (r0 + lr0)
        + (ulong)(il0 / NL) * BYTES;
    threadgroup half * stage_a = sa + pr0 * TT_NK + 16 * il0;
    threadgroup half * stage_b0 = sb + lr1 * TT_NK + iy;
    threadgroup half * stage_b1 = sb + (lr1 + 32) * TT_NK + iy;
    if (pr0 >= nr0) {
        *(threadgroup half4 *)(stage_a + 0)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 4)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 8)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 12) = half4(0.0h, 0.0h, 0.0h, 0.0h);
    }

    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        TT_NR1_WIDE, TT_NR0, TT_NK, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    using left_t = tensor<threadgroup half, extents<int, TT_NK, TT_NR1_WIDE>, tensor_inline>;
    using right_t = tensor<threadgroup half, extents<int, TT_NK, TT_NR0>, tensor_inline>;
    left_t left(sb, extents<int, TT_NK, TT_NR1_WIDE>());
    right_t right(sa, extents<int, TT_NK, TT_NR0>());
    auto acc = op.get_destination_cooperative_tensor<left_t, right_t, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    for (uint loop_k = 0; loop_k < args.K; loop_k += TT_NK) {
        half4x4 temp_a;
        DEQ(x_ptr, il, temp_a);
        const half2x4 yv0 = ok0 ? *((device const half2x4 *)y0) : half2x4(0.0h);
        const half2x4 yv1 = ok1 ? *((device const half2x4 *)y1) : half2x4(0.0h);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (pr0 < nr0) {
            *(threadgroup half4 *)(stage_a + 0)  = temp_a[0];
            *(threadgroup half4 *)(stage_a + 4)  = temp_a[1];
            *(threadgroup half4 *)(stage_a + 8)  = temp_a[2];
            *(threadgroup half4 *)(stage_a + 12) = temp_a[3];
        }
        if (valid0) *(threadgroup half2x4 *)(stage_b0) = yv0;
        else *(threadgroup half2x4 *)(stage_b0) = half2x4(0.0h);
        if (valid1) *(threadgroup half2x4 *)(stage_b1) = yv1;
        else *(threadgroup half2x4 *)(stage_b1) = half2x4(0.0h);

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + BYTES * ((2 + NL - 1) / NL)
                         : x_ptr;
        y0 += TT_NK;
        y1 += TT_NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left, right, acc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = (threadgroup float *)shmem;   // [64][64]
    tensor<threadgroup float, extents<int, TT_NR0, TT_NR1_WIDE>, tensor_inline> out_tile(
        temp_str, extents<int, TT_NR0, TT_NR1_WIDE>());
    acc.store(out_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short j = tiitg / 4;
    const short c0 = (tiitg % 4) * 16;
#pragma clang loop unroll(full)
    for (short s = 0; s < TT_NR1_WIDE; s += 32) {
        const short js = j + s;
        if (js < nr1) {
            const int slot = ids[(ulong)im * args.N + r1 + js];
            if (slot >= 0 && slot < slot_limit) {
                device float * D = dst + (ulong)slot * args.M + r0;
                threadgroup const float * C = temp_str + js * TT_NR0;
                const short c1 = (c0 + 16 < nr0) ? c0 + 16 : nr0;
                for (short i = c0; i < c1; ++i) {
                    if (epi == 0) {
                        D[i] = C[i];
                    } else if (epi == 2) {
                        D[i] = tt_silu(min(D[i], limit)) * clamp(C[i], -limit, limit);
                    } else {
                        D[i] = tt_silu(D[i]) * C[i];
                    }
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
    moe_grouped_slots_mm_tensor_body<BYTES, NL, DEQ, EPI, false, false>(
        args, srcA, (device const uchar *)srcB, counts, ids, dst, shmem, tgpig, tiitg_wide, 0.0f);
}

// Half-activation twin (srcB is half).
template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &),
          short EPI>
kernel void kernel_moe_grouped_slots_mm_tensor_f16act(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const half  * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]]) {
    moe_grouped_slots_mm_tensor_body<BYTES, NL, DEQ, EPI, false, true>(
        args, srcA, (device const uchar *)srcB, counts, ids, dst, shmem, tgpig, tiitg_wide, 0.0f);
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
    moe_grouped_slots_mm_tensor_body<BYTES, NL, DEQ, 2, false, false>(
        args, srcA, (device const uchar *)srcB, counts, ids, dst, shmem, tgpig, tiitg_wide, limit);
}

template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void kernel_moe_up_clamped_grouped_slots_tensor_f16act(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const half  * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        constant float & limit            [[buffer(6)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]]) {
    moe_grouped_slots_mm_tensor_body<BYTES, NL, DEQ, 2, false, true>(
        args, srcA, (device const uchar *)srcB, counts, ids, dst, shmem, tgpig, tiitg_wide, limit);
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
    moe_grouped_slots_mm_tensor_body<BYTES, NL, DEQ, 0, true, false>(
        args, srcA, (device const uchar *)srcB, nullptr, nullptr, dst, shmem, tgpig, tiitg_wide, 0.0f);
}

template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void kernel_mat_mat_tensor_f16act(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const half  * srcB         [[buffer(2)]],
        device       float * dst          [[buffer(3)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]]) {
    moe_grouped_slots_mm_tensor_body<BYTES, NL, DEQ, 0, true, true>(
        args, srcA, (device const uchar *)srcB, nullptr, nullptr, dst, shmem, tgpig, tiitg_wide, 0.0f);
}

// Dense 64-slot form: one 64-row weight slab is reused for two 32-slot
// activation groups, halving the weight dequant/staging work per output row.
// Dense only (EPI 0); the output tile grows to [64][64] F32 (16 KiB shmem).
template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &),
          bool HALF_ACT>
inline void moe_mat_mat_tensor_wide_body(
        constant moe_group_generic_mm_args & args,
        device const uchar * srcA,
        device const uchar * srcB_bytes,
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

    const short pr0 = (short)tiitg / TT_NL0;
    const short lr0 = pr0 < nr0 ? pr0 : nr0 - 1;
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
    const ulong y_off0 = (ulong)args.stride_b * (r1 + slot0) + (ulong)iy;
    const ulong y_off1 = (ulong)args.stride_b * (r1 + slot1) + (ulong)iy;
    device const float * y_ptr0 = (device const float *)srcB_bytes + y_off0;
    device const float * y_ptr1 = (device const float *)srcB_bytes + y_off1;
    device const half * yh_ptr0 = (device const half *)srcB_bytes + y_off0;
    device const half * yh_ptr1 = (device const half *)srcB_bytes + y_off1;

    threadgroup half * stage_a = sa + pr0 * TT_NK + 16 * il0;
    threadgroup half * stage_b0 = sb + phys0 * TT_NK + iy;
    threadgroup half * stage_b1 = sb + phys1 * TT_NK + iy;
    if (pr0 >= nr0) {
        *(threadgroup half4 *)(stage_a + 0)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 4)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 8)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 12) = half4(0.0h, 0.0h, 0.0h, 0.0h);
    }

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
        const half2x4 yv0 = HALF_ACT ? *((device const half2x4 *)yh_ptr0)
                                     : (half2x4)(*((device const float2x4 *)y_ptr0));
        const half2x4 yv1 = HALF_ACT ? *((device const half2x4 *)yh_ptr1)
                                     : (half2x4)(*((device const float2x4 *)y_ptr1));

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (pr0 < nr0) {
            *(threadgroup half4 *)(stage_a + 0)  = temp_a[0];
            *(threadgroup half4 *)(stage_a + 4)  = temp_a[1];
            *(threadgroup half4 *)(stage_a + 8)  = temp_a[2];
            *(threadgroup half4 *)(stage_a + 12) = temp_a[3];
        }
        if (valid0) *(threadgroup half2x4 *)(stage_b0) = yv0;
        else *(threadgroup half2x4 *)(stage_b0) = half2x4(0.0h);
        if (valid1) *(threadgroup half2x4 *)(stage_b1) = yv1;
        else *(threadgroup half2x4 *)(stage_b1) = half2x4(0.0h);

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + BYTES * ((2 + NL - 1) / NL) : x_ptr;
        y_ptr0 += TT_NK;
        y_ptr1 += TT_NK;
        yh_ptr0 += TT_NK;
        yh_ptr1 += TT_NK;

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
    moe_mat_mat_tensor_wide_body<BYTES, NL, DEQ, false>(
        args, srcA, (device const uchar *)srcB, dst, shmem, tgpig, tiitg_wide);
}

template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void kernel_mat_mat_tensor_wide_f16act(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const half  * srcB         [[buffer(2)]],
        device       float * dst          [[buffer(3)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]]) {
    moe_mat_mat_tensor_wide_body<BYTES, NL, DEQ, true>(
        args, srcA, (device const uchar *)srcB, dst, shmem, tgpig, tiitg_wide);
}

typedef decltype(kernel_moe_grouped_slots_mm_tensor<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K, 0>) moe_grouped_slots_mm_tensor_t;
typedef decltype(kernel_moe_up_clamped_grouped_slots_tensor<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K>) moe_up_clamped_grouped_slots_tensor_t;
typedef decltype(kernel_mat_mat_tensor<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K>) mat_mat_tensor_t;
typedef decltype(kernel_moe_grouped_slots_mm_tensor_f16act<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K, 0>) moe_grouped_slots_mm_tensor_f16act_t;
typedef decltype(kernel_moe_up_clamped_grouped_slots_tensor_f16act<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K>) moe_up_clamped_grouped_slots_tensor_f16act_t;
typedef decltype(kernel_mat_mat_tensor_f16act<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K>) mat_mat_tensor_f16act_t;

// One down / up_silu_mul / up_clamped / dense quadruple per dtype. Keep the
// list in sync with MOE_GROUPED_GENERIC_LAYOUTS (a CPU test parses it).
#define TT_INSTANTIATE(suffix, BYTES, NL, DEQ) \
    template [[host_name("kernel_moe_down_" #suffix "_f32_grouped_slots_tensor")]] kernel moe_grouped_slots_mm_tensor_t kernel_moe_grouped_slots_mm_tensor<BYTES, NL, DEQ, 0>; \
    template [[host_name("kernel_moe_up_silu_mul_" #suffix "_f32_grouped_slots_tensor")]] kernel moe_grouped_slots_mm_tensor_t kernel_moe_grouped_slots_mm_tensor<BYTES, NL, DEQ, 1>; \
    template [[host_name("kernel_moe_up_clamped_" #suffix "_f32_grouped_slots_tensor")]] kernel moe_up_clamped_grouped_slots_tensor_t kernel_moe_up_clamped_grouped_slots_tensor<BYTES, NL, DEQ>; \
    template [[host_name("kernel_mat_mat_" #suffix "_f32_tensor")]] kernel mat_mat_tensor_t kernel_mat_mat_tensor<BYTES, NL, DEQ>; \
    template [[host_name("kernel_moe_down_" #suffix "_f16act_grouped_slots_tensor")]] kernel moe_grouped_slots_mm_tensor_f16act_t kernel_moe_grouped_slots_mm_tensor_f16act<BYTES, NL, DEQ, 0>; \
    template [[host_name("kernel_moe_up_silu_mul_" #suffix "_f16act_grouped_slots_tensor")]] kernel moe_grouped_slots_mm_tensor_f16act_t kernel_moe_grouped_slots_mm_tensor_f16act<BYTES, NL, DEQ, 1>; \
    template [[host_name("kernel_moe_up_clamped_" #suffix "_f16act_grouped_slots_tensor")]] kernel moe_up_clamped_grouped_slots_tensor_f16act_t kernel_moe_up_clamped_grouped_slots_tensor_f16act<BYTES, NL, DEQ>; \
    template [[host_name("kernel_mat_mat_" #suffix "_f16act_tensor")]] kernel mat_mat_tensor_f16act_t kernel_mat_mat_tensor_f16act<BYTES, NL, DEQ>;

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
TT_INSTANTIATE(iq1_s,   QT_IQ1_S_BYTES,   QT_IQ1_S_NL,   qt_dequantize_iq1_s)
TT_INSTANTIATE(iq1_m,   QT_IQ1_M_BYTES,   QT_IQ1_M_NL,   qt_dequantize_iq1_m)

// The wide dense tile is instantiated for the Q8_0 projections V4.1 uses
// (q/kv/output and the shared expert); other dtypes keep the 32-slot form.
typedef decltype(kernel_mat_mat_tensor_wide<QT_Q8_0_BYTES, QT_Q8_0_NL, qt_dequantize_q8_0>) mat_mat_tensor_wide_t;
template [[host_name("kernel_mat_mat_q8_0_f32_tensor_wide")]] kernel mat_mat_tensor_wide_t kernel_mat_mat_tensor_wide<QT_Q8_0_BYTES, QT_Q8_0_NL, qt_dequantize_q8_0>;
typedef decltype(kernel_mat_mat_tensor_wide_f16act<QT_Q8_0_BYTES, QT_Q8_0_NL, qt_dequantize_q8_0>) mat_mat_tensor_wide_f16act_t;
template [[host_name("kernel_mat_mat_q8_0_f16act_tensor_wide")]] kernel mat_mat_tensor_wide_f16act_t kernel_mat_mat_tensor_wide_f16act<QT_Q8_0_BYTES, QT_Q8_0_NL, qt_dequantize_q8_0>;

// 64-slot gate/up probe tile, one instantiation per generic dtype.
typedef decltype(kernel_moe_wide_probe_f16act<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K>) moe_wide_probe_t;
#define TT_INSTANTIATE_WIDE(suffix, BYTES, NL, DEQ) \
    template [[host_name("kernel_moe_wide_probe_" #suffix "_f16act")]] kernel moe_wide_probe_t kernel_moe_wide_probe_f16act<BYTES, NL, DEQ>;

TT_INSTANTIATE_WIDE(q2_0,    QT_Q2_0_BYTES,    QT_Q2_0_NL,    qt_dequantize_q2_0)
TT_INSTANTIATE_WIDE(q2_K,    QT_Q2_K_BYTES,    QT_Q2_K_NL,    qt_dequantize_q2_K)
TT_INSTANTIATE_WIDE(q3_K,    QT_Q3_K_BYTES,    QT_Q3_K_NL,    qt_dequantize_q3_K)
TT_INSTANTIATE_WIDE(q4_K,    QT_Q4_K_BYTES,    QT_Q4_K_NL,    qt_dequantize_q4_K)
TT_INSTANTIATE_WIDE(q5_K,    QT_Q5_K_BYTES,    QT_Q5_K_NL,    qt_dequantize_q5_K)
TT_INSTANTIATE_WIDE(q6_K,    QT_Q6_K_BYTES,    QT_Q6_K_NL,    qt_dequantize_q6_K)
TT_INSTANTIATE_WIDE(q8_0,    QT_Q8_0_BYTES,    QT_Q8_0_NL,    qt_dequantize_q8_0)
TT_INSTANTIATE_WIDE(q4_0,    QT_Q4_0_BYTES,    QT_Q4_0_NL,    qt_dequantize_q4_0)
TT_INSTANTIATE_WIDE(q4_1,    QT_Q4_1_BYTES,    QT_Q4_1_NL,    qt_dequantize_q4_1)
TT_INSTANTIATE_WIDE(iq2_s,   QT_IQ2_S_BYTES,   QT_IQ2_S_NL,   qt_dequantize_iq2_s)
TT_INSTANTIATE_WIDE(iq3_xxs, QT_IQ3_XXS_BYTES, QT_IQ3_XXS_NL, qt_dequantize_iq3_xxs)
TT_INSTANTIATE_WIDE(iq3_s,   QT_IQ3_S_BYTES,   QT_IQ3_S_NL,   qt_dequantize_iq3_s)
TT_INSTANTIATE_WIDE(iq4_nl,  QT_IQ4_NL_BYTES,  QT_IQ4_NL_NL,  qt_dequantize_iq4_nl)
TT_INSTANTIATE_WIDE(iq4_xs,  QT_IQ4_XS_BYTES,  QT_IQ4_XS_NL,  qt_dequantize_iq4_xs)
TT_INSTANTIATE_WIDE(f32,     QT_F32_BYTES,     QT_F32_NL,     qt_dequantize_f32)
TT_INSTANTIATE_WIDE(f16,     QT_F16_BYTES,     QT_F16_NL,     qt_dequantize_f16)
TT_INSTANTIATE_WIDE(bf16,    QT_BF16_BYTES,    QT_BF16_NL,    qt_dequantize_bf16)
TT_INSTANTIATE_WIDE(mxfp4,   QT_MXFP4_BYTES,   QT_MXFP4_NL,   qt_dequantize_mxfp4)
TT_INSTANTIATE_WIDE(iq2_xs,  QT_IQ2_XS_BYTES,  QT_IQ2_XS_NL,  qt_dequantize_iq2_xs)
TT_INSTANTIATE_WIDE(iq2_xxs, QT_IQ2_XXS_BYTES, QT_IQ2_XXS_NL, qt_dequantize_iq2_xxs)
TT_INSTANTIATE_WIDE(iq1_s,   QT_IQ1_S_BYTES,   QT_IQ1_S_NL,   qt_dequantize_iq1_s)
TT_INSTANTIATE_WIDE(iq1_m,   QT_IQ1_M_BYTES,   QT_IQ1_M_NL,   qt_dequantize_iq1_m)

// Natural-row-major Q5_K -> F16 dequant for the dense preweight route.
// One thread dequantizes one 16-element sub-tile with `qt_dequantize_q5_K`
// (the same helper the quantized tensor tile uses), and writes it at the
// weight's natural block*256 + il*16 offset so the dense F16 tensor tile
// reads the same values the Q5_K tile would.
kernel void v41_dequant_q5_k_f16(
        device const uchar * src [[buffer(0)]],
        device       half  * dst [[buffer(1)]],
        constant     uint2 & dims [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint n_in = dims.x;
    const uint row = gid.y;
    const uint blocks = n_in / 256;
    const uint block = gid.x >> 4;
    const uint il = gid.x & 15;
    if (block >= blocks) return;
    const device uchar * blk = src + (row * blocks + block) * QT_Q5_K_BYTES;
    half4x4 reg;
    qt_dequantize_q5_K(blk, (short)il, reg);
    device half * out = dst + row * n_in + block * 256 + il * 16;
    for (int i = 0; i < 16; ++i) out[i] = reg[i / 4][i % 4];
}
