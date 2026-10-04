// GLM-5.3-Flash DSA indexer cache maintenance for one token of one MLA layer.
//
// Follows llama.cpp glm5-next build_kpool_select:
//   key  = LayerNorm(raw_key; weight, bias, eps)          (F32, then F16 cache)
//   gate = raw_gate                                       (F16 cache)
// When the token completes a pool of 4 consecutive positions (aligned to 0):
//   pooled[c] = sum_j softmax_j(gate_j[c] + ape[j][c]) * key_j[c]
// computed in F32 from the F16-rounded rows, then stored as F16.
// `pending` is F16 [4 slots][key | gate][128]; `pooled` is F16 [pools][128];
// `ape` is F32 [4][128] (GGUF [128, 4]).

#include <metal_stdlib>
using namespace metal;

struct glm53_indexer_append_args {
    uint position;
    float eps;
};

constant uint IDX_D = 128u;
constant uint IDX_POOL = 4u;

kernel void kernel_glm53_indexer_append(
        constant glm53_indexer_append_args & args [[buffer(0)]],
        device const float * raw_key [[buffer(1)]],
        device const float * raw_gate [[buffer(2)]],
        device const float * norm_weight [[buffer(3)]],
        device const float * norm_bias [[buffer(4)]],
        device const float * ape [[buffer(5)]],
        device half * pending [[buffer(6)]],
        device half * pooled [[buffer(7)]],
        threadgroup float * reduce [[threadgroup(0)]],
        ushort c [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const float x = raw_key[c];
    const float sum = simd_sum(x);
    if (lane == 0u) reduce[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = (reduce[0] + reduce[1] + reduce[2] + reduce[3]) / float(IDX_D);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float centered = x - mean;
    const float sq = simd_sum(centered * centered);
    if (lane == 0u) reduce[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float variance = (reduce[0] + reduce[1] + reduce[2] + reduce[3]) / float(IDX_D);
    const float key = centered * rsqrt(variance + args.eps) * norm_weight[c] + norm_bias[c];

    const uint slot = args.position % IDX_POOL;
    pending[slot * 2u * IDX_D + c] = half(key);
    pending[slot * 2u * IDX_D + IDX_D + c] = half(raw_gate[c]);
    if (slot != IDX_POOL - 1u) return;

    // Each thread reads only its own channel, written by itself in earlier
    // dispatches or above, so no barrier is needed.
    float logits[IDX_POOL];
    for (uint j = 0; j < IDX_POOL; ++j) {
        logits[j] = float(pending[j * 2u * IDX_D + IDX_D + c]) + ape[j * IDX_D + c];
    }
    float maximum = logits[0];
    for (uint j = 1; j < IDX_POOL; ++j) maximum = max(maximum, logits[j]);
    float total = 0.0f;
    float acc = 0.0f;
    for (uint j = 0; j < IDX_POOL; ++j) {
        const float weight = exp(logits[j] - maximum);
        total += weight;
        acc += weight * float(pending[j * 2u * IDX_D + c]);
    }
    pooled[(args.position / IDX_POOL) * IDX_D + c] = half(acc / total);
}
