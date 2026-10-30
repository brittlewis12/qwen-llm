// K64 score panel: four independent H16 simdgroups, device F16 Q,
// register-prefetched F32-to-F16 K, and 16 KiB of ping-pong shared storage.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp;

#include "../deepseek_v41_attention.h"

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_deepseek_v41_attention_score_simdgroup_qk(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const half *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    device float *scores [[buffer(7)]], constant uint4 &m [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint KT = 64, DK = 64, HS = 16;
    // Float backing also supplies aligned score staging after the last K read.
    threadgroup float storage[64 * KT];
    threadgroup half *keys = (threadgroup half *)storage;
    const uint r = group.y, k0 = group.x * KT;
    const uint4 rr = attention_row_raw(a, b, r);
    const uint length = attention_row_length(b, r);
    const uint count = rr.y + min(length, a.z);
    if (k0 >= count) return;
    device const uint *row_ids = ids + r * a.z;
    const uint kk = tid >> 1, part = tid & 1, i = k0 + kk;
    const bool live = i < count && (i < rr.y || row_ids[i - rr.y] < length);
    device const float *krow = raw;
    if (live) krow = attention_row_key(rr, raw, compressed, row_ids, i);
    half4 prefetched[8];
    auto fetch = [&](uint ds) {
        device const float4 *src = (device const float4 *)(krow + ds * DK + part * 32);
#pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) prefetched[j] = live ? half4(src[j]) : half4(0.0h);
    };
    auto stage = [&](uint buf) {
        threadgroup half4 *dst = (threadgroup half4 *)(keys + buf * KT * DK + kk * DK + part * 32);
#pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) dst[j] = prefetched[j];
    };

    constexpr auto desc0 = tensor_ops::matmul2d_descriptor(
        HS, KT, DK, false, true, false, tensor_ops::matmul2d_descriptor::mode::multiply);
    constexpr auto desc1 = tensor_ops::matmul2d_descriptor(
        HS, KT, DK, false, true, false, tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc0, execution_simdgroup> op0;
    tensor_ops::matmul2d<desc1, execution_simdgroup> op1;
    tensor<device half, dextents<int, 2>, tensor_inline> Q(
        (device half *)(q + r * m.y * a.x), dextents<int, 2>(int(a.x), int(m.y)));
    auto q0 = Q.slice<DK, HS>(0, int(sg * HS));
    using k_t = tensor<threadgroup half, extents<int, DK, KT>, tensor_inline>;
    k_t b0(keys, extents<int, DK, KT>());
    auto acc = op0.get_destination_cooperative_tensor<decltype(q0), k_t, float>();
    fetch(0);
    stage(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ds = 0; ds < a.x / DK; ++ds) {
        const uint buf = ds & 1;
        const bool more = ds + 1 < a.x / DK;
        if (more) fetch(ds + 1);
        auto qa = Q.slice<DK, HS>(int(ds * DK), int(sg * HS));
        k_t bt(keys + buf * KT * DK, extents<int, DK, KT>());
        if (ds == 0) op0.run(qa, bt, acc);
        else op1.run(qa, bt, acc);
        if (more) stage(buf ^ 1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<threadgroup float, extents<int, KT, HS>, tensor_inline> st(
        storage + sg * HS * KT, extents<int, KT, HS>());
    acc.store(st);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = rsqrt(float(a.x));
    device float *srow = scores + r * m.y * a.w;
    for (uint e = tid; e < 64 * KT; e += 128) {
        const uint h = e / KT, key = e % KT;
        if (k0 + key < count) srow[h * a.w + k0 + key] = storage[e] * scale;
    }
}
