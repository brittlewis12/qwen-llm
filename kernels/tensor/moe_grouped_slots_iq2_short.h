// IQ2_XXS short routed gate/up, 16 slots x 64 outputs, two SIMD groups.
// Same half operands, ascending K32 accumulations and epilogues as the
// 32-slot tensor tile. Each thread stages two weight fragments. Only
// buckets with at most 16 slots select this tile; larger buckets retain
// the existing tile. Buffers 0..6 are unchanged; 64 threads and 5120 B
// dynamic shared memory. Included after the common tensor definitions.
#define V41_SHORT_NR0 64
#define V41_SHORT_NR1 16
#define V41_SHORT_NK 32
#define V41_SHORT_NL0 2
#define V41_SHORT_NL1 4

template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &),
          short EPI, bool DENSE, bool HALF_ACT>
inline void v41_short_grouped_slots_body(
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
    threadgroup half * sa = (threadgroup half *)(shmem);          // [V41_SHORT_NR0][V41_SHORT_NK]
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);   // [V41_SHORT_NR1][V41_SHORT_NK]

    const int im = DENSE ? 0 : tgpig.z;
    const int r0 = tgpig.y * V41_SHORT_NR0;
    const int r1 = tgpig.x * V41_SHORT_NR1;

    const int count = DENSE ? int(args.N) : counts[im];
    if (count < 0 || uint(count) > args.N
        || count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < V41_SHORT_NR0) ? (short)((int)args.M - r0) : V41_SHORT_NR0;
    const short nr1 = (count - r1 < V41_SHORT_NR1) ? (short)(count - r1) : V41_SHORT_NR1;

    // Clamp source rows only; every physical fragment has one writer, including
    // zero-filled tails consumed by the full tensor operation.
    const short pr0 = (short)tiitg / V41_SHORT_NL0;
    const short lr0 = pr0 < nr0 ? pr0 : nr0 - 1;
    const short lr0b = pr0 + 32 < nr0 ? pr0 + 32 : nr0 - 1;
    const short il0 = (tiitg % V41_SHORT_NL0);
    short il = il0;

    // Activation staging: thread -> (physical slot pr1, 8-element chunk iy).
    const short pr1 = (short)tiitg / V41_SHORT_NL1;
    const short lr1 = pr1 < nr1 ? pr1 : nr1 - 1;
    const short iy = 8 * (tiitg % V41_SHORT_NL1);

    const short offset1 = il0 / NL;
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * BYTES;
    device const uchar * x_ptr_b = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0b) + (ulong)offset1 * BYTES;
    const int slot_limit = int(args.slot_limit);
    const int slot_id = DENSE ? r1 + lr1 : ids[(ulong)im * args.N + (r1 + lr1)];
    const bool slot_valid = pr1 < nr1 && slot_id >= 0 && slot_id < slot_limit;
    const int b_row = (slot_valid ? slot_id : 0) / int(args.b_div);
    const ulong y_off = (ulong)args.stride_b * b_row + (ulong)iy;
    device const float * y_ptr = (device const float *)srcB_bytes + y_off;
    device const half * yh_ptr = (device const half *)srcB_bytes + y_off;

    threadgroup half * stage_a = sa + pr0 * V41_SHORT_NK + 16 * il0;
    threadgroup half * stage_ab = sa + (pr0 + 32) * V41_SHORT_NK + 16 * il0;
    threadgroup half * stage_b = sb + pr1 * V41_SHORT_NK + iy;
    // Padded tails and invalid slots keep their pre-zeroed fragments; full
    // fragments are rewritten every K iteration.
    if (pr0 >= nr0) {
        *(threadgroup half4 *)(stage_a + 0)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 4)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 8)  = half4(0.0h, 0.0h, 0.0h, 0.0h);
        *(threadgroup half4 *)(stage_a + 12) = half4(0.0h, 0.0h, 0.0h, 0.0h);
    }
    if (pr0 + 32 >= nr0) {
        *(threadgroup half4 *)(stage_ab + 0) = half4(0.0h);
        *(threadgroup half4 *)(stage_ab + 4) = half4(0.0h);
        *(threadgroup half4 *)(stage_ab + 8) = half4(0.0h);
        *(threadgroup half4 *)(stage_ab + 12) = half4(0.0h);
    }
    if (!slot_valid) {
        *(threadgroup half2x4 *)(stage_b) = half2x4(0.0h);
    }

    // left = activations [16 slots x 32 k], right = weights [64 rows x 32 k]
    // read transposed; destination [16 slots x 64 rows] in registers.
    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        V41_SHORT_NR1, V41_SHORT_NR0, V41_SHORT_NK, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroups<2>> op;
    using left_t = tensor<threadgroup half, extents<int, V41_SHORT_NK, V41_SHORT_NR1>, tensor_inline>;
    using right_t = tensor<threadgroup half, extents<int, V41_SHORT_NK, V41_SHORT_NR0>, tensor_inline>;
    left_t left(sb, extents<int, V41_SHORT_NK, V41_SHORT_NR1>());
    right_t right(sa, extents<int, V41_SHORT_NK, V41_SHORT_NR0>());
    auto acc = op.template get_destination_cooperative_tensor<left_t, right_t, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        acc[i] = 0.0f;
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += V41_SHORT_NK) {
        half4x4 temp_a;
        DEQ(x_ptr, il, temp_a);
        half4x4 temp_ab;
        DEQ(x_ptr_b, il, temp_ab);
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
        if (pr0 + 32 < nr0) {
            *(threadgroup half4 *)(stage_ab + 0) = temp_ab[0];
            *(threadgroup half4 *)(stage_ab + 4) = temp_ab[1];
            *(threadgroup half4 *)(stage_ab + 8) = temp_ab[2];
            *(threadgroup half4 *)(stage_ab + 12) = temp_ab[3];
        }
        if (slot_valid) {
            *(threadgroup half2x4 *)(stage_b) = yv;
        }

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + BYTES * ((2 + NL - 1) / NL) : x_ptr;
        x_ptr_b = (il < 2) ? x_ptr_b + BYTES * ((2 + NL - 1) / NL) : x_ptr_b;
        y_ptr += V41_SHORT_NK;
        yh_ptr += V41_SHORT_NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left, right, acc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = (threadgroup float *)shmem;   // [V41_SHORT_NR1][V41_SHORT_NR0]
    tensor<threadgroup float, extents<int, V41_SHORT_NR0, V41_SHORT_NR1>, tensor_inline> out_tile(
        temp_str, extents<int, V41_SHORT_NR0, V41_SHORT_NR1>());
    acc.store(out_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: 4 threads per slot, 16 consecutive rows each.
    const short j = tiitg / 4;
    const short c0 = (tiitg % 4) * 16;
    if (j < nr1) {
        const int slot = DENSE ? r1 + j : ids[(ulong)im * args.N + r1 + j];
        if (slot >= 0 && slot < slot_limit) {
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup const float * C = temp_str + j * V41_SHORT_NR0;
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



kernel void kernel_moe_down_iq2_xxs_f16act_grouped_slots_tensor_short(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const half * input [[buffer(2)]],
        device const int * counts [[buffer(3)]],
        device const int * ids [[buffer(4)]],
        device float * output [[buffer(5)]],
        threadgroup uchar * shmem [[threadgroup(0)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    v41_short_grouped_slots_body<QT_IQ2_XXS_BYTES, QT_IQ2_XXS_NL,
        qt_dequantize_iq2_xxs, 0, false, true>(
        args, weights, (device const uchar *)input, counts, ids, output,
        shmem, tg, tid, 0.0f);
}

kernel void kernel_moe_up_clamped_iq2_xxs_f16act_grouped_slots_tensor_short(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const half * input [[buffer(2)]],
        device const int * counts [[buffer(3)]],
        device const int * ids [[buffer(4)]],
        device float * output [[buffer(5)]],
        constant float & limit [[buffer(6)]],
        threadgroup uchar * shmem [[threadgroup(0)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    v41_short_grouped_slots_body<QT_IQ2_XXS_BYTES, QT_IQ2_XXS_NL,
        qt_dequantize_iq2_xxs, 2, false, true>(
        args, weights, (device const uchar *)input, counts, ids, output,
        shmem, tg, tid, limit);
}

#undef V41_SHORT_NR0
#undef V41_SHORT_NR1
#undef V41_SHORT_NK
#undef V41_SHORT_NL0
#undef V41_SHORT_NL1
