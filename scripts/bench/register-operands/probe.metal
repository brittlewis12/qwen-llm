// Standalone SDK-27 experiment, deliberately outside kernels/tensor and build.rs.
#include "../../../kernels/tensor/moe_grouped_slots_tensor.metal"

template <int BYTES, void (*DEQ)(device const uchar *, short, thread half4x4 &),
          bool WEIGHT_LEFT, bool RESIDENT, short EPI, bool AUDIT>
kernel void register_probe(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const float * activations [[buffer(2)]],
        device const int * counts [[buffer(3)]],
        device const int * ids [[buffer(4)]],
        device float * dst [[buffer(5)]],
        constant float & limit [[buffer(6)]],
        device half * audit [[buffer(7)]],
        threadgroup uchar * shmem [[threadgroup(0)]],
        uint3 grid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const int count = counts[grid.z];
    const int r0 = int(grid.y) * 64, s0 = int(grid.x) * 32;
    if (count < int(args.min_count) || count > int(args.max_count) || s0 >= count) return;
    const int nr0 = min(64, int(args.M) - r0), nr1 = min(32, count - s0);
    const int sg = int(tid / 32);
    // Same 64-weight-row x 32-slot macro tile, same K32 steps. The API requires
    // execution_simdgroup, so partition output rather than splitting K.
    const int row_base = WEIGHT_LEFT ? sg * 16 : (sg % 2) * 32;
    const int slot_base = WEIGHT_LEFT ? 0 : (sg / 2) * 16;
    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto left = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto right = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto acc = op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(left)>, remove_addrspace_t<decltype(right)>, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint k = 0; k < args.K; k += 32) {
        if (!RESIDENT || k == 0) {
            const uint kk = RESIDENT ? 0 : k;
            // Four adjacent cooperative elements share one dequant16. No
            // decoder arithmetic is changed; the compiler can discard unused
            // components of the existing half4x4 decoder.
#pragma clang loop unroll(full)
            for (ushort side = 0; side < 2; ++side) {
                const bool is_weight = WEIGHT_LEFT ? side == 0 : side == 1;
                const ushort capacity = side == 0 ? left.get_capacity() : right.get_capacity();
#pragma clang loop unroll(full)
                for (ushort i = 0; i < capacity; i += 4) {
                    const auto c = side == 0 ? left.get_multidimensional_index(i)
                                             : right.get_multidimensional_index(i);
                    half4x4 decoded;
                    const int row = min(row_base + c[1], nr0 - 1);
                    const int slot_index = min(slot_base + c[1], nr1 - 1);
                    const int slot = ids[ulong(grid.z) * args.N + s0 + slot_index];
                    const int b_row = (slot >= 0 && slot < int(args.slot_limit) ? slot : 0) / int(args.b_div);
                    if (is_weight) {
                        const uint wk = kk + uint(c[0]);
                        DEQ(weights + (ulong(grid.z) * args.M + r0 + row) * args.nb01
                            + ulong(wk / 256) * BYTES, short((wk % 256) / 16), decoded);
                    }
#pragma clang loop unroll(full)
                    for (ushort j = 0; j < 4; ++j) {
                        const auto ij = side == 0 ? left.get_multidimensional_index(i + j)
                                                  : right.get_multidimensional_index(i + j);
                        const uint d = uint(ij[0]) % 16;
                        const half value = is_weight ? decoded[d / 4][d % 4]
                            : half(activations[ulong(b_row) * args.stride_b + kk + ij[0]]);
                        if (side == 0) left[i + j] = value;
                        else right[i + j] = value;
                        if (AUDIT && grid.x == 0 && grid.y == 0 && grid.z == 0) {
                            const uint offset = side == 0 ? 0 : 512;
                            audit[(ulong(k / 32) * 4 + sg) * 1536 + offset
                                  + ij[1] * 32 + ij[0]] = value;
                        }
                    }
                }
            }
        }
        op.run(left, right, acc);
    }
    threadgroup float * output = (threadgroup float *)shmem;
    // Preserve the original [slot][64] output staging and epilogue. Only the
    // cooperative accumulator's ownership differs between the two mappings.
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        const auto c = acc.get_multidimensional_index(i);
        const int row = row_base + (WEIGHT_LEFT ? c[1] : c[0]);
        const int slot = slot_base + (WEIGHT_LEFT ? c[0] : c[1]);
        output[slot * 64 + row] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int j = int(tid / 4), c0 = int(tid % 4) * 16;
    if (j < nr1) {
        const int slot = ids[ulong(grid.z) * args.N + s0 + j];
        if (slot >= 0 && slot < int(args.slot_limit)) {
            device float * D = dst + ulong(slot) * args.M + r0;
            threadgroup const float * C = output + j * 64;
            for (int i = c0; i < min(c0 + 16, nr0); ++i) {
                if (EPI == 0) D[i] = C[i];
                else if (EPI == 2) D[i] = tt_silu(min(D[i], limit)) * clamp(C[i], -limit, limit);
                else D[i] = tt_silu(D[i]) * C[i];
            }
        }
    }
}

template <int BYTES, void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void decode_probe(constant moe_group_generic_mm_args & args [[buffer(0)]],
                         device const uchar * weights [[buffer(1)]],
                         device half * out [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t >= args.M * args.K / 16) return;
    const uint row = t / (args.K / 16), k = (t % (args.K / 16)) * 16;
    half4x4 v;
    DEQ(weights + ulong(row) * args.nb01 + ulong(k / 256) * BYTES, short((k % 256) / 16), v);
    for (uint i = 0; i < 16; ++i) out[ulong(row) * args.K + k + i] = v[i / 4][i % 4];
}

template <int BYTES, void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void staged_resident_probe(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * weights [[buffer(1)]], device const float * activations [[buffer(2)]],
        device const int * counts [[buffer(3)]], device const int * ids [[buffer(4)]],
        device float * dst [[buffer(5)]], threadgroup uchar * shmem [[threadgroup(0)]],
        uint3 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int count = counts[grid.z], r0 = int(grid.y) * 64, s0 = int(grid.x) * 32;
    if (count < int(args.min_count) || count > int(args.max_count) || s0 >= count) return;
    const int nr0 = min(64, int(args.M) - r0), nr1 = min(32, count - s0);
    const int row = min(int(tid / 2), nr0 - 1), il = tid % 2;
    const int slot = ids[ulong(grid.z) * args.N + s0 + min(int(tid / 4), nr1 - 1)];
    const int b_row = (slot >= 0 && slot < int(args.slot_limit) ? slot : 0) / int(args.b_div);
    half4x4 v;
    DEQ(weights + (ulong(grid.z) * args.M + r0 + row) * args.nb01, short(il), v);
    threadgroup half * sa = (threadgroup half *)shmem;
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);
    for (int j = 0; j < 16; ++j) sa[row * 32 + il * 16 + j] = v[j / 4][j % 4];
    for (int j = 0; j < 8; ++j) sb[min(int(tid / 4), nr1 - 1) * 32 + (tid % 4) * 8 + j]
        = half(activations[ulong(b_row) * args.stride_b + (tid % 4) * 8 + j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = tensor_ops::matmul2d_descriptor(32, 64, 32, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    using left_t = tensor<threadgroup half, extents<int, 32, 32>, tensor_inline>;
    using right_t = tensor<threadgroup half, extents<int, 32, 64>, tensor_inline>;
    left_t left(sb, extents<int, 32, 32>());
    right_t right(sa, extents<int, 32, 64>());
    auto acc = op.template get_destination_cooperative_tensor<left_t, right_t, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint k = 0; k < args.K; k += 32) op.run(left, right, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * output = (threadgroup float *)shmem;
    tensor<threadgroup float, extents<int, 64, 32>, tensor_inline> tile(output, extents<int, 64, 32>());
    acc.store(tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int j = int(tid / 4), c0 = int(tid % 4) * 16;
    if (j < nr1) {
        const int target = ids[ulong(grid.z) * args.N + s0 + j];
        if (target >= 0 && target < int(args.slot_limit))
            for (int i = c0; i < min(c0 + 16, nr0); ++i)
                dst[ulong(target) * args.M + r0 + i] = output[j * 64 + i];
    }
}

typedef decltype(register_probe<98, qt_dequantize_iq3_xxs, false, false, 0, false>) register_probe_t;
typedef decltype(decode_probe<98, qt_dequantize_iq3_xxs>) decode_probe_t;
typedef decltype(staged_resident_probe<98, qt_dequantize_iq3_xxs>) staged_resident_probe_t;
#define RP_ENTRY(NAME, BYTES, DEQ, WL, RES, EPI, AUDIT) \
template [[host_name(NAME)]] kernel register_probe_t register_probe<BYTES, DEQ, WL, RES, EPI, AUDIT>;
#define RP_TYPE(SUFFIX, BYTES, DEQ) \
RP_ENTRY("ax_" SUFFIX, BYTES, DEQ, false, false, 0, false) \
RP_ENTRY("wx_" SUFFIX, BYTES, DEQ, true, false, 0, false) \
RP_ENTRY("ax_audit_" SUFFIX, BYTES, DEQ, false, false, 0, true) \
RP_ENTRY("wx_audit_" SUFFIX, BYTES, DEQ, true, false, 0, true) \
RP_ENTRY("ax_resident_" SUFFIX, BYTES, DEQ, false, true, 0, false) \
RP_ENTRY("wx_resident_" SUFFIX, BYTES, DEQ, true, true, 0, false) \
RP_ENTRY("ax_silu_" SUFFIX, BYTES, DEQ, false, false, 1, false) \
RP_ENTRY("wx_silu_" SUFFIX, BYTES, DEQ, true, false, 1, false) \
RP_ENTRY("ax_clamp_" SUFFIX, BYTES, DEQ, false, false, 2, false) \
RP_ENTRY("wx_clamp_" SUFFIX, BYTES, DEQ, true, false, 2, false) \
template [[host_name("decode_" SUFFIX)]] kernel decode_probe_t decode_probe<BYTES, DEQ>; \
template [[host_name("staged_resident_" SUFFIX)]] kernel staged_resident_probe_t staged_resident_probe<BYTES, DEQ>;

RP_TYPE("iq3_xxs", QT_IQ3_XXS_BYTES, qt_dequantize_iq3_xxs)
RP_TYPE("iq2_xxs", QT_IQ2_XXS_BYTES, qt_dequantize_iq2_xxs)

template <int BYTES, void (*DEQ)(device const uchar *, short, thread half4x4 &), bool WEIGHT>
inline void mapped_fragment(constant moe_group_generic_mm_args & args,
        device const uchar * weights, device const float * x, device const int * ids,
        uint expert, int r0, int s0, int base, int valid_rows, uint k, int fn, int fm,
        thread half (&v)[8]) {
#pragma clang loop unroll(full)
    for (int i = 0; i < 2; ++i) {
        const int row = min(base + fm + i * 8, valid_rows - 1);
        half4x4 decoded;
        int b_row = 0;
        if (WEIGHT) DEQ(weights + (ulong(expert) * args.M + r0 + row) * args.nb01
            + ulong(k / 256) * BYTES, short((k % 256) / 16), decoded);
        else {
            const int slot = ids[ulong(expert) * args.N + s0 + row];
            b_row = (slot >= 0 && slot < int(args.slot_limit) ? slot : 0) / int(args.b_div);
        }
#pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j)
            v[i * 4 + j] = WEIGHT ? decoded[(fn + j) / 4][(fn + j) % 4]
                                 : half(x[ulong(b_row) * args.stride_b + k + fn + j]);
    }
}

// Alternate operand mapping follows the pinned MLX BaseNAXFrag K16 layout.
// Its explicit per-lane fragments avoid querying input coordinates in the hot
// loop. AUDIT uses the API's coordinates, independently checking that mapping.
inline half4 broadcast_half4(thread const half4x4 & producer, ushort owner, int column) {
    const uint2 a = simd_shuffle(as_type<uint2>(producer[0]), owner);
    const uint2 b = simd_shuffle(as_type<uint2>(producer[1]), owner);
    const uint2 c = simd_shuffle(as_type<uint2>(producer[2]), owner);
    const uint2 d = simd_shuffle(as_type<uint2>(producer[3]), owner);
    return as_type<half4>(column == 0 ? a : column == 1 ? b : column == 2 ? c : d);
}

template <int BYTES, void (*DEQ)(device const uchar *, short, thread half4x4 &), bool WEIGHT_LEFT, bool AUDIT,
          bool SHUFFLE = false>
kernel void mapped_probe(
        constant moe_group_generic_mm_args & args [[buffer(0)]],
        device const uchar * weights [[buffer(1)]], device const float * x [[buffer(2)]],
        device const int * counts [[buffer(3)]], device const int * ids [[buffer(4)]],
        device float * dst [[buffer(5)]], device half * audit [[buffer(7)]],
        threadgroup uchar * shmem [[threadgroup(0)]], uint3 grid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const int count = counts[grid.z], r0 = int(grid.y) * 64, s0 = int(grid.x) * 32;
    if (count < int(args.min_count) || count > int(args.max_count) || s0 >= count) return;
    const int nr0 = min(64, int(args.M) - r0), nr1 = min(32, count - s0);
    const int sg = tid / 32, lane = tid % 32, qid = lane >> 2;
    const int fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
    const int rb = WEIGHT_LEFT ? sg * 16 : (sg % 2) * 32;
    const int sb = WEIGHT_LEFT ? 0 : (sg / 2) * 16;
    constexpr auto desc = tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto left = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto right = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto acc = op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(left)>, remove_addrspace_t<decltype(right)>, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < 16; ++i) acc[i] = 0;
    for (uint k = 0; k < args.K; k += 16) {
        half a[8], b0[8], b1[8];
        if (SHUFFLE) {
            half4x4 producer(0.0h);
            if (!WEIGHT_LEFT || lane < 16) {
                const int row = min(rb + lane, nr0 - 1);
                DEQ(weights + (ulong(grid.z) * args.M + r0 + row) * args.nb01
                    + ulong(k / 256) * BYTES, short((k % 256) / 16), producer);
            }
            if (WEIGHT_LEFT) {
                mapped_fragment<BYTES, DEQ, false>(args, weights, x, ids, grid.z, r0, s0, sb, nr1, k, fn, fm, b0);
                mapped_fragment<BYTES, DEQ, false>(args, weights, x, ids, grid.z, r0, s0, sb + 16, nr1, k, fn, fm, b1);
#pragma clang loop unroll(full)
                for (int i = 0; i < 2; ++i) {
                    const half4 v = broadcast_half4(producer, ushort(fm + i * 8), fn / 4);
                    for (int j = 0; j < 4; ++j) a[i * 4 + j] = v[j];
                }
            } else {
                mapped_fragment<BYTES, DEQ, false>(args, weights, x, ids, grid.z, r0, s0, sb, nr1, k, fn, fm, a);
#pragma clang loop unroll(full)
                for (int i = 0; i < 2; ++i) {
                    const half4 v0 = broadcast_half4(producer, ushort(fm + i * 8), fn / 4);
                    const half4 v1 = broadcast_half4(producer, ushort(fm + i * 8 + 16), fn / 4);
                    for (int j = 0; j < 4; ++j) { b0[i * 4 + j] = v0[j]; b1[i * 4 + j] = v1[j]; }
                }
            }
        } else {
            mapped_fragment<BYTES, DEQ, WEIGHT_LEFT>(args, weights, x, ids, grid.z, r0, s0,
                WEIGHT_LEFT ? rb : sb, WEIGHT_LEFT ? nr0 : nr1, k, fn, fm, a);
            mapped_fragment<BYTES, DEQ, !WEIGHT_LEFT>(args, weights, x, ids, grid.z, r0, s0,
                WEIGHT_LEFT ? sb : rb, WEIGHT_LEFT ? nr1 : nr0, k, fn, fm, b0);
            mapped_fragment<BYTES, DEQ, !WEIGHT_LEFT>(args, weights, x, ids, grid.z, r0, s0,
                (WEIGHT_LEFT ? sb : rb) + 16, WEIGHT_LEFT ? nr1 : nr0, k, fn, fm, b1);
        }
#pragma clang loop unroll(full)
        for (ushort i = 0; i < 8; ++i) { left[i] = a[i]; right[i] = b0[i]; right[8 + i] = b1[i]; }
        if (AUDIT && grid.x == 0 && grid.y == 0 && grid.z == 0) {
#pragma clang loop unroll(full)
            for (ushort i = 0; i < 8; ++i) {
                const auto c = left.get_multidimensional_index(i);
                audit[(ulong(k / 32) * 4 + sg) * 1536 + c[1] * 32 + k % 32 + c[0]] = left[i];
            }
#pragma clang loop unroll(full)
            for (ushort i = 0; i < 16; ++i) {
                const auto c = right.get_multidimensional_index(i);
                audit[(ulong(k / 32) * 4 + sg) * 1536 + 512 + c[1] * 32 + k % 32 + c[0]] = right[i];
            }
        }
        op.run(left, right, acc);
    }
    threadgroup float * output = (threadgroup float *)shmem;
#pragma clang loop unroll(full)
    for (ushort i = 0; i < 16; ++i) {
        const auto c = acc.get_multidimensional_index(i);
        output[(sb + (WEIGHT_LEFT ? c[0] : c[1])) * 64 + rb + (WEIGHT_LEFT ? c[1] : c[0])] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int j = int(tid / 4), c0 = int(tid % 4) * 16;
    if (j < nr1) {
        const int target = ids[ulong(grid.z) * args.N + s0 + j];
        if (target >= 0 && target < int(args.slot_limit))
            for (int i = c0; i < min(c0 + 16, nr0); ++i)
                dst[ulong(target) * args.M + r0 + i] = output[j * 64 + i];
    }
}
typedef decltype(mapped_probe<98, qt_dequantize_iq3_xxs, false, false>) mapped_probe_t;
#define MAPPED_TYPE(SUFFIX, BYTES, DEQ) \
template [[host_name("ax16_" SUFFIX)]] kernel mapped_probe_t mapped_probe<BYTES, DEQ, false, false>; \
template [[host_name("wx16_" SUFFIX)]] kernel mapped_probe_t mapped_probe<BYTES, DEQ, true, false>; \
template [[host_name("ax16_audit_" SUFFIX)]] kernel mapped_probe_t mapped_probe<BYTES, DEQ, false, true>; \
template [[host_name("wx16_audit_" SUFFIX)]] kernel mapped_probe_t mapped_probe<BYTES, DEQ, true, true>; \
template [[host_name("axsh_" SUFFIX)]] kernel mapped_probe_t mapped_probe<BYTES, DEQ, false, false, true>; \
template [[host_name("wxsh_" SUFFIX)]] kernel mapped_probe_t mapped_probe<BYTES, DEQ, true, false, true>; \
template [[host_name("axsh_audit_" SUFFIX)]] kernel mapped_probe_t mapped_probe<BYTES, DEQ, false, true, true>; \
template [[host_name("wxsh_audit_" SUFFIX)]] kernel mapped_probe_t mapped_probe<BYTES, DEQ, true, true, true>;
MAPPED_TYPE("iq3_xxs", QT_IQ3_XXS_BYTES, qt_dequantize_iq3_xxs)
MAPPED_TYPE("iq2_xxs", QT_IQ2_XXS_BYTES, qt_dequantize_iq2_xxs)

#include "single_simd.metalh"
#include "tile32_f16.metalh"
