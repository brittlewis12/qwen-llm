// Research-only Apple10 attention falsifier; no runtime dispatch selects it.
// Structure follows llama.cpp fa_tensor (MIT): two tensor matmuls, strided
// device K/V, float cooperative O resident across the complete key loop.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

struct flash_probe_args {
    uint n_rows, n_pos, base_pos, kv_stride, vt_stride;
    uint n_q_heads, n_kv_heads, group, head_dim;
    float scale;
    uint causal_skip;
};

// Causal block classes are shared by all KV heads. A query tile packs GQA
// siblings in exactly the incumbent matrix path's local-query order.
kernel void kernel_attn_flash_probe_mask(
        constant flash_probe_args &a [[buffer(0)]],
        device uchar *blocks [[buffer(1)]], uint2 id [[thread_position_in_grid]]) {
    uint nq = a.n_rows * a.group;
    if (id.x >= (nq + 31) / 32 || id.y >= a.n_pos / 64) return;
    uint first = id.x * 32 / a.group;
    uint last = min(id.x * 32 + 31, nq - 1) / a.group;
    uint lo = id.y * 64, hi = lo + 63;
    blocks[id.x * (a.n_pos / 64) + id.y] =
        lo > a.base_pos + last ? 0 : (hi <= a.base_pos + first ? 2 : 1);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_attn_flash_probe_f16_d256(
        constant flash_probe_args &a [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const half *k [[buffer(2)]], device const half *v [[buffer(3)]],
        device float *out [[buffer(4)]], device const uchar *blocks [[buffer(5)]],
        threadgroup uchar *scratch [[threadgroup(0)]],
        uint2 tile [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int Q = 32, C = 64, D = 256, NSG = 8;
    threadgroup half *sq = (threadgroup half *)scratch;
    threadgroup float *ss = (threadgroup float *)(sq + Q * D);
    threadgroup half *sp = (threadgroup half *)(ss + Q * C);
    threadgroup float *sr = (threadgroup float *)(sp + Q * C);
    threadgroup atomic_uint *changed = (threadgroup atomic_uint *)(sr + Q);
    const uint q0 = tile.x * Q, kvh = tile.y, nq = a.n_rows * a.group;
    const uint qdim = a.n_q_heads * D;
    for (uint i = tid; i < Q * D / 4; i += 256) {
        uint j = i / (D / 4), c = i % (D / 4), local = q0 + j;
        float4 value = 0;
        if (local < nq) {
            uint row = local / a.group, h = kvh * a.group + local % a.group;
            value = ((device const float4 *)(q + row * qdim + h * D))[c];
        }
        ((threadgroup half4 *)sq)[i] = half4(value * a.scale);
    }
    auto tq = tensor(sq, dextents<int, 2>(D, Q));
    auto ts = tensor(ss, dextents<int, 2>(C, Q));
    auto tp = tensor(sp, dextents<int, 2>(C, Q));
    tensor_ops::matmul2d<tensor_ops::matmul2d_descriptor(
        Q, C, D, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<NSG>> qk;
    tensor_ops::matmul2d<tensor_ops::matmul2d_descriptor(
        Q, D, C, false, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<NSG>> pv;
    // TensorOps requires non-const element types; these views only read K/V.
    auto tv0 = tensor((device half *)v, dextents<int, 2>(D, C), array<int, 2>{1, int(a.kv_stride)});
    auto co = pv.get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    for (int i = 0; i < co.get_capacity(); ++i) if (co.is_valid_element(i)) co[i] = 0;
    float maxima[4], sums[4];
    for (int j = 0; j < 4; ++j) { maxima[j] = -FLT_MAX / 2; sums[j] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pos = 0; pos < a.n_pos; pos += C) {
        uchar block = blocks[tile.x * (a.n_pos / C) + pos / C];
        if (block == 0) continue;
        if (tid == 0) atomic_store_explicit(changed, 0u, memory_order_relaxed);
        auto tk = tensor((device half *)k + ulong(pos) * a.kv_stride + kvh * D,
            dextents<int, 2>(D, C), array<int, 2>{1, int(a.kv_stride)});
        qk.run(tq, tk, ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int jj = 0; jj < 4; ++jj) {
            uint j = jj * NSG + sg, local = q0 + j;
            float s0 = ss[j * C + lane], s1 = ss[j * C + lane + 32];
            if (local >= nq || (block != 2 && pos + lane > a.base_pos + local / a.group)) s0 = -INFINITY;
            if (local >= nq || (block != 2 && pos + lane + 32 > a.base_pos + local / a.group)) s1 = -INFINITY;
            float m = simd_max(max(maxima[jj], max(s0, s1))), rescale = 1;
            if (m > maxima[jj] + 8) {
                rescale = exp(maxima[jj] - m); maxima[jj] = m;
                if (lane == 0) atomic_store_explicit(changed, 1u, memory_order_relaxed);
            }
            half p0 = half(exp(s0 - maxima[jj])), p1 = half(exp(s1 - maxima[jj]));
            sp[j * C + lane] = p0; sp[j * C + lane + 32] = p1;
            sums[jj] = sums[jj] * rescale + simd_sum(float(p0) + float(p1));
            if (lane == 0) sr[j] = rescale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(changed, memory_order_relaxed)) {
            for (int i = 0; i < co.get_capacity(); ++i)
                if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
        auto tv = tensor((device half *)v + ulong(pos) * a.kv_stride + kvh * D,
            dextents<int, 2>(D, C), array<int, 2>{1, int(a.kv_stride)});
        pv.run(tp, tv, co);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int jj = 0; jj < 4; ++jj)
        if (lane == 0) sr[jj * NSG + sg] = sums[jj] == 0 ? 0 : 1 / sums[jj];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < co.get_capacity(); ++i) if (co.is_valid_element(i)) {
        auto idx = co.get_multidimensional_index(i);
        uint local = q0 + idx[1];
        if (local < nq) {
            uint row = local / a.group, h = kvh * a.group + local % a.group;
            out[ulong(row) * qdim + h * D + idx[0]] = co[i] * sr[idx[1]];
        }
    }
}
