// Metal 4 tensor-ops form of the V4.1 attention sink tile
// (deepseek_v41_attention.metal kernel_deepseek_v41_attention_score_mma /
// pv_mma): same buffers, same score slab protocol, same softmax_rows and
// normalize_rows around them; QK^T and PV run as one matmul2d per 16-key
// block instead of a serial simdgroup-MMA accumulator chain.
//
// One threadgroup per (16-head block, query row), 128 threads = 4
// simdgroups. Keys are the row's raw ring prefix followed by its selected
// compressed rows (K = V = the same 512-wide latent), gathered 16 at a
// time into threadgroup memory as half; Q is staged once as half. Sums
// are F32; softmax and the final normalization stay in the shared
// kernels.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp;

#include "../deepseek_v41_attention.h"

#define V41_TT_HEADS 16
#define V41_TT_KEYS 16
#define V41_TT_DIM 512

// scores[h][k] = scale * sum_d Q[h][d] K[k][d] for the row's keys; invalid
// selected ids are zero keys here and are rejected by softmax as before.
kernel void kernel_deepseek_v41_attention_score_tensor(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    device float *scores [[buffer(7)]], constant uint4 &m [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half qblk[V41_TT_HEADS * V41_TT_DIM];
    threadgroup half kblk[V41_TT_KEYS * V41_TT_DIM];
    // The 16 x 16 score block is staged over the key slab once the matmul
    // has consumed it (Q + K already fill the 32 KiB threadgroup budget).
    threadgroup float *sblk = (threadgroup float *)kblk;
    const uint r = group.y;
    const uint4 rr = attention_row_raw(a, b, r);
    const uint length = attention_row_length(b, r);
    const uint selected = min(length, a.z);
    const uint count = rr.y + selected;
    device const uint *row_ids = ids + r * a.z;
    device const float *qrow = q + r * m.y * a.x + group.x * V41_TT_HEADS * a.x;
    device float *srow = scores + r * m.y * a.w + group.x * V41_TT_HEADS * a.w;
    const float scale = rsqrt(float(a.x));
    for (uint e = tid; e < V41_TT_HEADS * V41_TT_DIM / 4; e += 128) {
        ((threadgroup half4 *)qblk)[e] = half4(((device const float4 *)qrow)[e]);
    }

    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        V41_TT_HEADS, V41_TT_KEYS, V41_TT_DIM, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply);
    tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    using q_t = tensor<threadgroup half, extents<int, V41_TT_DIM, V41_TT_HEADS>, tensor_inline>;
    using k_t = tensor<threadgroup half, extents<int, V41_TT_DIM, V41_TT_KEYS>, tensor_inline>;
    using s_t = tensor<threadgroup float, extents<int, V41_TT_KEYS, V41_TT_HEADS>, tensor_inline>;
    q_t qt(qblk, extents<int, V41_TT_DIM, V41_TT_HEADS>());
    k_t kt(kblk, extents<int, V41_TT_DIM, V41_TT_KEYS>());
    s_t st(sblk, extents<int, V41_TT_KEYS, V41_TT_HEADS>());

    for (uint k0 = 0; k0 < count; k0 += V41_TT_KEYS) {
        // The previous block's matmul and scatter are done with the slabs.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < V41_TT_KEYS * V41_TT_DIM / 4; e += 128) {
            const uint key = e / (V41_TT_DIM / 4), c4 = e % (V41_TT_DIM / 4);
            const uint i = k0 + key;
            float4 v = float4(0.0f);
            if (i < count && (i < rr.y || row_ids[i - rr.y] < length)) {
                v = ((device const float4 *)attention_row_key(rr, raw, compressed, row_ids, i))[c4];
            }
            ((threadgroup half4 *)kblk)[e] = half4(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto acc = op.get_destination_cooperative_tensor<q_t, k_t, float>();
        op.run(qt, kt, acc);
        // Every simdgroup is done reading the key slab before it is reused.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc.store(st);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // sblk is [head][key]; two elements per thread.
        for (uint e = tid; e < V41_TT_HEADS * V41_TT_KEYS; e += 128) {
            const uint h = e / V41_TT_KEYS, kk = e % V41_TT_KEYS;
            if (k0 + kk < count) srow[h * a.w + k0 + kk] = sblk[e] * scale;
        }
    }
}

#define V41_TT_KEYS_WIDE 32
#define V41_TT_DIM_CHUNK 256

// Wider-N score tile: 32 keys per panel with the latent split into two
// 256-wide chunks. Q stays resident in 16 KiB and K uses the other 16 KiB,
// so the existing 32 KiB threadgroup budget is unchanged. The score block
// (32 x 16 F32) aliases the K slab after both matmuls, exactly as the
// 16-key form does.
kernel void kernel_deepseek_v41_attention_score_tensor32(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    device float *scores [[buffer(7)]], constant uint4 &m [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half qblk[V41_TT_HEADS * V41_TT_DIM];
    threadgroup half kblk[V41_TT_KEYS_WIDE * V41_TT_DIM_CHUNK];
    threadgroup float *sblk = (threadgroup float *)kblk;
    const uint r = group.y;
    const uint4 rr = attention_row_raw(a, b, r);
    const uint length = attention_row_length(b, r);
    const uint selected = min(length, a.z);
    const uint count = rr.y + selected;
    device const uint *row_ids = ids + r * a.z;
    device const float *qrow = q + r * m.y * a.x + group.x * V41_TT_HEADS * a.x;
    device float *srow = scores + r * m.y * a.w + group.x * V41_TT_HEADS * a.w;
    const float scale = rsqrt(float(a.x));
    for (uint e = tid; e < V41_TT_HEADS * V41_TT_DIM / 4; e += 128) {
        ((threadgroup half4 *)qblk)[e] = half4(((device const float4 *)qrow)[e]);
    }

    constexpr auto desc0 = tensor_ops::matmul2d_descriptor(
        V41_TT_HEADS, V41_TT_KEYS_WIDE, V41_TT_DIM_CHUNK, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply);
    constexpr auto desc1 = tensor_ops::matmul2d_descriptor(
        V41_TT_HEADS, V41_TT_KEYS_WIDE, V41_TT_DIM_CHUNK, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc0, execution_simdgroups<4>> op0;
    tensor_ops::matmul2d<desc1, execution_simdgroups<4>> op1;
    using q_t = tensor<threadgroup half, extents<int, V41_TT_DIM_CHUNK, V41_TT_HEADS>, tensor_inline>;
    using k_t =
        tensor<threadgroup half, extents<int, V41_TT_DIM_CHUNK, V41_TT_KEYS_WIDE>, tensor_inline>;
    using s_t =
        tensor<threadgroup float, extents<int, V41_TT_KEYS_WIDE, V41_TT_HEADS>, tensor_inline>;
    // Q rows are 512 wide; each chunk view reads every other 256 elements.
    q_t qt0(qblk, extents<int, V41_TT_DIM_CHUNK, V41_TT_HEADS>(),
            array<int, 2>{1, int(V41_TT_DIM)});
    q_t qt1(qblk + V41_TT_DIM_CHUNK, extents<int, V41_TT_DIM_CHUNK, V41_TT_HEADS>(),
            array<int, 2>{1, int(V41_TT_DIM)});
    k_t kt(kblk, extents<int, V41_TT_DIM_CHUNK, V41_TT_KEYS_WIDE>());
    s_t st(sblk, extents<int, V41_TT_KEYS_WIDE, V41_TT_HEADS>());

    auto acc = op0.get_destination_cooperative_tensor<q_t, k_t, float>();
    for (uint k0 = 0; k0 < count; k0 += V41_TT_KEYS_WIDE) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Latent chunk 0: dim 0..255.
        for (uint e = tid; e < V41_TT_KEYS_WIDE * V41_TT_DIM_CHUNK / 4; e += 128) {
            const uint key = e / (V41_TT_DIM_CHUNK / 4), c4 = e % (V41_TT_DIM_CHUNK / 4);
            const uint i = k0 + key;
            float4 v = float4(0.0f);
            if (i < count && (i < rr.y || row_ids[i - rr.y] < length)) {
                v = ((device const float4 *)attention_row_key(rr, raw, compressed, row_ids, i))[c4];
            }
            ((threadgroup half4 *)kblk)[e] = half4(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op0.run(qt0, kt, acc);
        // Latent chunk 1: dim 256..511.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < V41_TT_KEYS_WIDE * V41_TT_DIM_CHUNK / 4; e += 128) {
            const uint key = e / (V41_TT_DIM_CHUNK / 4), c4 = e % (V41_TT_DIM_CHUNK / 4);
            const uint i = k0 + key;
            float4 v = float4(0.0f);
            if (i < count && (i < rr.y || row_ids[i - rr.y] < length)) {
                v = ((device const float4 *)attention_row_key(rr, raw, compressed, row_ids, i))
                        [c4 + V41_TT_DIM_CHUNK / 4];
            }
            ((threadgroup half4 *)kblk)[e] = half4(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op1.run(qt1, kt, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc.store(st);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // st is [key][head] with the key extent contiguous.
        for (uint e = tid; e < V41_TT_HEADS * V41_TT_KEYS_WIDE; e += 128) {
            const uint h = e / V41_TT_KEYS_WIDE, kk = e % V41_TT_KEYS_WIDE;
            if (k0 + kk < count) {
                srow[h * a.w + k0 + kk] = sblk[kk + h * V41_TT_KEYS_WIDE] * scale;
            }
        }
    }
}

// out[h][r][:] = sum_k P[h][k] V[k][:] over the row's keys (unnormalized;
// normalize_rows divides by the softmax denominator).
kernel void kernel_deepseek_v41_attention_pv_tensor(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *raw [[buffer(3)]], device const float *compressed [[buffer(4)]],
    device const uint *ids [[buffer(5)]], device const float *scores [[buffer(7)]],
    device float *out [[buffer(8)]], constant uint4 &m [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half vblk[V41_TT_KEYS * V41_TT_DIM];
    threadgroup half pblk[V41_TT_HEADS * V41_TT_KEYS];
    const uint r = group.y;
    const uint4 rr = attention_row_raw(a, b, r);
    const uint length = attention_row_length(b, r);
    const uint selected = min(length, a.z);
    const uint count = rr.y + selected;
    device const uint *row_ids = ids + r * a.z;
    device const float *srow = scores + r * m.y * a.w + group.x * V41_TT_HEADS * a.w;
    device float *orow = out + r * a.x + (group.x * V41_TT_HEADS) * m.z * a.x;
    const bool invalid = srow[a.w - 1] == -1.0f;

    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        V41_TT_HEADS, V41_TT_DIM, V41_TT_KEYS, false, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    using p_t = tensor<threadgroup half, extents<int, V41_TT_KEYS, V41_TT_HEADS>, tensor_inline>;
    using v_t = tensor<threadgroup half, extents<int, V41_TT_DIM, V41_TT_KEYS>, tensor_inline>;
    p_t pt(pblk, extents<int, V41_TT_KEYS, V41_TT_HEADS>());
    v_t vt(vblk, extents<int, V41_TT_DIM, V41_TT_KEYS>());
    auto acc = op.get_destination_cooperative_tensor<p_t, v_t, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        acc[i] = 0.0f;
    }
    if (!invalid) {
        for (uint k0 = 0; k0 < count; k0 += V41_TT_KEYS) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid; e < V41_TT_KEYS * V41_TT_DIM / 4; e += 128) {
                const uint key = e / (V41_TT_DIM / 4), c4 = e % (V41_TT_DIM / 4);
                const uint i = k0 + key;
                float4 v = float4(0.0f);
                if (i < count) v = ((device const float4 *)attention_row_key(rr, raw, compressed, row_ids, i))[c4];
                ((threadgroup half4 *)vblk)[e] = half4(v);
            }
            // P block [16 heads x 16 keys], zero past count (the slab tail
            // is unspecified there).
            for (uint e = tid; e < V41_TT_HEADS * V41_TT_KEYS; e += 128) {
                const uint h = e / V41_TT_KEYS, kk = e % V41_TT_KEYS;
                pblk[e] = half(k0 + kk < count ? srow[h * a.w + k0 + kk] : 0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            op.run(pt, vt, acc);
        }
    }
    // out rows of one head are m.z * a.x floats apart.
    tensor<device float, extents<int, V41_TT_DIM, V41_TT_HEADS>, tensor_inline> ot(
        orow, extents<int, V41_TT_DIM, V41_TT_HEADS>(), array<int, 2>{1, int(m.z * a.x)});
    acc.store(ot);
}

// All-head form: 64 heads x 32 keys x 128 latent per matmul, one threadgroup
// per query row. Q (16 KiB) + K (8 KiB) + the 32x64 score block (8 KiB) use
// the full 32 KiB budget. The four 128-wide latent chunks accumulate into one
// cooperative tensor, so each key row is gathered once per panel instead of
// once per 16-head block.
#define V41_TT64_HEADS 64
#define V41_TT64_KEYS 32
#define V41_TT64_DIM 128

kernel void kernel_deepseek_v41_attention_score_tensor64(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    device float *scores [[buffer(7)]], constant uint4 &m [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half qblk[V41_TT64_HEADS * V41_TT64_DIM];
    threadgroup half kblk[V41_TT64_KEYS * V41_TT64_DIM];
    threadgroup float sblk[V41_TT64_KEYS * V41_TT64_HEADS];
    const uint r = group.y;
    const uint4 rr = attention_row_raw(a, b, r);
    const uint length = attention_row_length(b, r);
    const uint selected = min(length, a.z);
    const uint count = rr.y + selected;
    device const uint *row_ids = ids + r * a.z;
    device const float *qrow = q + r * m.y * a.x;
    device float *srow = scores + r * m.y * a.w;
    const float scale = rsqrt(float(a.x));

    constexpr auto desc0 = tensor_ops::matmul2d_descriptor(
        V41_TT64_HEADS, V41_TT64_KEYS, V41_TT64_DIM, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply);
    constexpr auto desc1 = tensor_ops::matmul2d_descriptor(
        V41_TT64_HEADS, V41_TT64_KEYS, V41_TT64_DIM, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc0, execution_simdgroups<4>> op0;
    tensor_ops::matmul2d<desc1, execution_simdgroups<4>> op1;
    using q_t = tensor<threadgroup half, extents<int, V41_TT64_DIM, V41_TT64_HEADS>, tensor_inline>;
    using k_t = tensor<threadgroup half, extents<int, V41_TT64_DIM, V41_TT64_KEYS>, tensor_inline>;
    using s_t = tensor<threadgroup float, extents<int, V41_TT64_KEYS, V41_TT64_HEADS>, tensor_inline>;
    q_t qt(qblk, extents<int, V41_TT64_DIM, V41_TT64_HEADS>(),
           array<int, 2>{1, int(V41_TT64_DIM)});
    k_t kt(kblk, extents<int, V41_TT64_DIM, V41_TT64_KEYS>());
    s_t st(sblk, extents<int, V41_TT64_KEYS, V41_TT64_HEADS>());
    auto acc = op0.get_destination_cooperative_tensor<q_t, k_t, float>();

    for (uint k0 = 0; k0 < count; k0 += V41_TT64_KEYS) {
        for (uint dc = 0; dc < 4; ++dc) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid; e < V41_TT64_HEADS * V41_TT64_DIM / 4; e += 128) {
                const uint h = e / (V41_TT64_DIM / 4), c4 = e % (V41_TT64_DIM / 4);
                const device float4 *src =
                    (device const float4 *)(qrow + h * a.x + dc * V41_TT64_DIM);
                ((threadgroup half4 *)qblk)[e] = half4(src[c4]);
            }
            for (uint e = tid; e < V41_TT64_KEYS * V41_TT64_DIM / 4; e += 128) {
                const uint key = e / (V41_TT64_DIM / 4), c4 = e % (V41_TT64_DIM / 4);
                const uint i = k0 + key;
                float4 v = float4(0.0f);
                if (i < count && (i < rr.y || row_ids[i - rr.y] < length)) {
                    const device float *krow =
                        attention_row_key(rr, raw, compressed, row_ids, i);
                    v = ((device const float4 *)(krow + dc * V41_TT64_DIM))[c4];
                }
                ((threadgroup half4 *)kblk)[e] = half4(v);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (dc == 0) {
                op0.run(qt, kt, acc);
            } else {
                op1.run(qt, kt, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc.store(st);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < V41_TT64_HEADS * V41_TT64_KEYS; e += 128) {
            const uint h = e / V41_TT64_KEYS, kk = e % V41_TT64_KEYS;
            if (k0 + kk < count) {
                srow[h * a.w + k0 + kk] = sblk[kk + h * V41_TT64_KEYS] * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_deepseek_v41_attention_pv_tensor64(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *raw [[buffer(3)]], device const float *compressed [[buffer(4)]],
    device const uint *ids [[buffer(5)]], device const float *scores [[buffer(7)]],
    device float *out [[buffer(8)]], constant uint4 &m [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half vblk[V41_TT64_KEYS * V41_TT64_DIM];
    threadgroup half pblk[V41_TT64_HEADS * V41_TT64_KEYS];
    const uint r = group.y;
    const uint4 rr = attention_row_raw(a, b, r);
    const uint length = attention_row_length(b, r);
    const uint selected = min(length, a.z);
    const uint count = rr.y + selected;
    device const uint *row_ids = ids + r * a.z;
    device const float *srow = scores + r * m.y * a.w;
    const bool invalid = srow[a.w - 1] == -1.0f;

    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        V41_TT64_HEADS, V41_TT64_DIM, V41_TT64_KEYS, false, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    using p_t = tensor<threadgroup half, extents<int, V41_TT64_KEYS, V41_TT64_HEADS>, tensor_inline>;
    using v_t = tensor<threadgroup half, extents<int, V41_TT64_DIM, V41_TT64_KEYS>, tensor_inline>;
    p_t pt(pblk, extents<int, V41_TT64_KEYS, V41_TT64_HEADS>());
    v_t vt(vblk, extents<int, V41_TT64_DIM, V41_TT64_KEYS>());

    for (uint dc = 0; dc < 4; ++dc) {
        auto acc = op.get_destination_cooperative_tensor<p_t, v_t, float>();
#pragma clang loop unroll(full)
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            acc[i] = 0.0f;
        }
        if (!invalid) {
            for (uint k0 = 0; k0 < count; k0 += V41_TT64_KEYS) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = tid; e < V41_TT64_KEYS * V41_TT64_DIM / 4; e += 128) {
                    const uint key = e / (V41_TT64_DIM / 4), c4 = e % (V41_TT64_DIM / 4);
                    const uint i = k0 + key;
                    float4 v = float4(0.0f);
                    if (i < count) {
                        const device float *krow =
                            attention_row_key(rr, raw, compressed, row_ids, i);
                        v = ((device const float4 *)(krow + dc * V41_TT64_DIM))[c4];
                    }
                    ((threadgroup half4 *)vblk)[e] = half4(v);
                }
                for (uint e = tid; e < V41_TT64_HEADS * V41_TT64_KEYS; e += 128) {
                    const uint h = e / V41_TT64_KEYS, kk = e % V41_TT64_KEYS;
                    pblk[e] = half(k0 + kk < count ? srow[h * a.w + k0 + kk] : 0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                op.run(pt, vt, acc);
            }
        }
        device float *orow = out + r * a.x + dc * V41_TT64_DIM;
        tensor<device float, extents<int, V41_TT64_DIM, V41_TT64_HEADS>, tensor_inline> ot(
            orow, extents<int, V41_TT64_DIM, V41_TT64_HEADS>(),
            array<int, 2>{1, int(m.z * a.x)});
        acc.store(ot);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
