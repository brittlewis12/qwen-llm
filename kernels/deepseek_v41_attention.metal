#include <metal_stdlib>
using namespace metal;
#include "deepseek_v41_attention.h"

kernel void kernel_deepseek_v41_attention_selection_prepare(
    constant uint4 &a [[buffer(0)]], device uint *ids [[buffer(1)]],
    device uint *mask [[buffer(2)]],
    uint t [[thread_position_in_grid]]) {
    if (t < a.w) ids[t] = t;
    if (a.z && t < a.y) mask[t] = 1;
}

// Batched selection preparation: one 2-D grid over (visible key, row).
// a: start, ratio, top_k, candidate_block. b: selected stride, candidates
// stride, rows, publish_mask. Unpruned rows write ids 0..length-1; the
// source layer writes an all-ones block mask for its unpruned rows.
kernel void kernel_deepseek_v41_attention_selection_prepare_rows(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device uint *selected [[buffer(2)]], device uint *candidates [[buffer(3)]],
    constant uint &row0 [[buffer(4)]], uint2 t [[thread_position_in_grid]]) {
    uint local = t.y;
    if (local >= b.z) return;
    uint row = row0 + local;
    uint length = (a.x + row + 1) / a.y;
    uint all = length <= a.z ? length : 0;
    uint blocks = (length + a.w - 1) / a.w;
    if (t.x < all) selected[(ulong)row * b.x + t.x] = t.x;
    if (b.w && t.x < blocks) candidates[(ulong)row * b.y + t.x] = 1;
}

kernel void kernel_deepseek_v41_attention_sort_tile(
    constant uint4 &a [[buffer(0)]], device float *scores [[buffer(1)]],
    device uint *ids [[buffer(2)]], threadgroup float *values [[threadgroup(0)]],
    threadgroup uint *order [[threadgroup(1)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint base = group * a.y;
    for (uint j = tid; j < a.y; j += 256) {
        values[j] = scores[base + j]; order[j] = ids[base + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint first = a.w ? 2 : a.z, last = a.w ? a.y : a.z;
    for (uint width = first; width <= last; width *= 2) {
        for (uint stride = min(width / 2, a.y / 2); stride; stride /= 2) {
            for (uint j = tid; j < a.y; j += 256) {
                uint other = j ^ stride;
                if (other <= j) continue;
                float l = values[j], r = values[other];
                uint li = order[j], ri = order[other];
                bool better = l > r || (l == r && (as_type<uint>(l) == as_type<uint>(r)
                    ? li < ri : as_type<uint>(l) < as_type<uint>(r)));
                if (better != (((base + j) & width) == 0)) {
                    values[j] = r; values[other] = l;
                    order[j] = ri; order[other] = li;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint j = tid; j < a.y; j += 256) {
        scores[base + j] = values[j]; ids[base + j] = order[j];
    }
}

kernel void kernel_deepseek_v41_attention_finite(
    constant uint &n [[buffer(0)]], device const float *x [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t < n && !isfinite(x[t])) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
}

// These deliberately simple F32 operators preserve the source cache format.
inline void attention_rotate(device const float *input, device float *output, float angle) {
    float c = cos(angle), s = sin(angle), u = input[0], v = input[1];
    output[0] = u * c - v * s;
    output[1] = u * s + v * c;
}

kernel void kernel_deepseek_v41_attention_rope(
    constant uint4 &a [[buffer(0)]], device float *x [[buffer(1)]],
    device const float *freq [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    // a: dimension, frequency count, position, inverse
    uint h = t / a.y, j = t % a.y;
    uint offset = h * a.x + a.x - 2 * a.y + 2 * j;
    float angle = float(a.z) * freq[j] * (a.w ? -1.0f : 1.0f);
    attention_rotate(x + offset, x + offset, angle);
}

kernel void kernel_deepseek_v41_attention_copy(
    constant uint &n [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t < n) y[t] = x[t];
}

kernel void kernel_deepseek_v41_attention_pool(
    constant uint &n [[buffer(0)]], device const float *v0 [[buffer(1)]],
    device const float *s0 [[buffer(2)]], device float *v1 [[buffer(3)]],
    device const float *s1 [[buffer(4)]], uint t [[thread_position_in_grid]]) {
    if (t >= n) return;
    float m = max(s0[t], s1[t]);
    float a = exp(s0[t] - m), b = exp(s1[t] - m), d = a + b;
    v1[t] = v0[t] * (a / d) + v1[t] * (b / d);
}

kernel void kernel_deepseek_v41_attention_index(
    constant uint4 &a [[buffer(0)]], device const float *q [[buffer(1)]],
    device const float *weights [[buffer(2)]], device const float *keys [[buffer(3)]],
    device float *scores [[buffer(4)]], uint t [[thread_position_in_grid]]) {
    if (t >= a.x) return;
    float score = 0, scale = rsqrt(float(a.y)) * rsqrt(float(a.z));
    for (uint h = 0; h < a.z; ++h) {
        float dot = 0;
        for (uint j = 0; j < a.y; ++j) dot += q[h * a.y + j] * keys[t * a.y + j];
        score += max(dot, 0.0f) * (weights[h] * scale);
    }
    scores[t] = score;
}

kernel void kernel_deepseek_v41_attention_index_heads(
    constant uint4 &a [[buffer(0)]], device const float *q [[buffer(1)]],
    device const float *weights [[buffer(2)]], device const float *keys [[buffer(3)]],
    device float *scores [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint t = group * 8 + sg;
    if (t >= a.x) return;
    float score = 0, scale = rsqrt(float(a.y)) * rsqrt(float(a.z));
    for (uint base = 0; base < a.z; base += 32) {
        uint h = base + lane;
        float dot = 0;
        if (h < a.z) {
            for (uint j = 0; j < a.y; ++j) dot += q[h * a.y + j] * keys[t * a.y + j];
        }
        // Preserve the scalar head-sum order: selection depends on these scores.
        for (uint head = 0; head < min(32u, a.z - base); ++head)
            score += max(simd_shuffle(dot, head), 0.0f) * (weights[base + head] * scale);
    }
    if (lane == 0) scores[t] = score;
}

// Batched index scores: one row per group.y, eight keys per threadgroup.
// a: start, ratio, top_k, max_length. b: dim, heads, q stride, weight stride.
// c: score stride, rows, 0, 0. Rows at or below top_k are unpruned and skip.
kernel void kernel_deepseek_v41_attention_index_heads_rows(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    constant uint4 &c [[buffer(2)]], device const float *q [[buffer(3)]],
    device const float *weights [[buffer(4)]], device const float *keys [[buffer(5)]],
    device float *scores [[buffer(6)]], constant uint &row0 [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint local = group.y;
    if (local >= c.y) return;
    uint row = row0 + local;
    uint length = (a.x + row + 1) / a.y;
    if (length <= a.z) return;
    uint t = group.x * 8 + sg;
    if (t >= length) return;
    device const float *qrow = q + (ulong)row * b.z;
    device const float *wrow = weights + (ulong)row * b.w;
    device float *srow = scores + (ulong)local * c.x;
    float score = 0, scale = rsqrt(float(b.x)) * rsqrt(float(b.y));
    for (uint base = 0; base < b.y; base += 32) {
        uint h = base + lane;
        float dot = 0;
        if (h < b.y) {
            for (uint j = 0; j < b.x; ++j) dot += qrow[h * b.x + j] * keys[t * b.x + j];
        }
        for (uint head = 0; head < min(32u, b.y - base); ++head)
            score += max(simd_shuffle(dot, head), 0.0f) * (wrow[base + head] * scale);
    }
    if (lane == 0) srow[t] = score;
}

// index_heads_rows for 32 heads x 128 dims with the same F32 arithmetic (each
// head's dot accumulates the dims in order; heads are summed in order after
// relu and weighting), restructured for long rows: the row query is staged
// once per threadgroup, transposed so lanes (heads) read consecutive words,
// and each simdgroup carries eight independent keys. Same arguments and score
// layout; 64 keys per threadgroup along group.x.
#define V41_ILP_KEYS 8u
kernel void kernel_deepseek_v41_attention_index_heads_rows_ilp(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    constant uint4 &c [[buffer(2)]], device const float *q [[buffer(3)]],
    device const float *weights [[buffer(4)]], device const float *keys [[buffer(5)]],
    device float *scores [[buffer(6)]], constant uint &row0 [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qt[128 * 32];
    const uint local = group.y;
    if (local >= c.y) return;
    const uint row = row0 + local;
    const uint length = (a.x + row + 1) / a.y;
    if (length <= a.z || group.x * 8u * V41_ILP_KEYS >= length) return;
    device const float *qrow = q + (ulong)row * b.z;
    for (uint e = tid; e < 128u * 32u; e += 256u) qt[(e % 128u) * 32u + e / 128u] = qrow[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint key0 = (group.x * 8u + sg) * V41_ILP_KEYS;
    if (key0 >= length) return;
    const uint n = min(V41_ILP_KEYS, length - key0);
    device const float *k[V41_ILP_KEYS];
    for (uint i = 0; i < V41_ILP_KEYS; ++i) k[i] = keys + (ulong)(key0 + min(i, n - 1u)) * 128u;
    float dot[V41_ILP_KEYS];
    for (uint i = 0; i < V41_ILP_KEYS; ++i) dot[i] = 0;
    for (uint j = 0; j < 128u; ++j) {
        const float qv = qt[j * 32u + lane];
        for (uint i = 0; i < V41_ILP_KEYS; ++i) dot[i] += qv * k[i][j];
    }
    device const float *wrow = weights + (ulong)row * b.w;
    device float *srow = scores + (ulong)local * c.x;
    const float scale = rsqrt(float(b.x)) * rsqrt(float(b.y));
    for (uint i = 0; i < n; ++i) {
        float score = 0;
        for (uint head = 0; head < 32u; ++head)
            score += max(simd_shuffle(dot[i], head), 0.0f) * (wrow[head] * scale);
        if (lane == 0) srow[key0 + i] = score;
    }
}

// index_heads_rows_ilp's scores (32 heads x 128 dims; F32 operands, F32
// accumulation) on simdgroup_float8x8 MMA, for long rows. A 256-thread
// threadgroup covers V41_IDX_MMA_KEYS keys along group.x: simdgroup (key
// group g = sg / 4, head group h = sg % 4) keeps Q^T of heads 8h..8h+7 in
// registers and, for its 8-key blocks, computes S^T = K x Q^T from 8x8 key
// tiles loaded straight from the key rows (dims accumulated in order, eight
// per step), applies relu, and reduces its eight heads with a second F32 MMA
// against a tile whose columns are the scaled head weights; the four
// head-group partials are summed in order through threadgroup memory. Equal
// to index_heads_rows up to F32 summation order; same arguments and score
// layout as the ILP kernel. One 8-key block per simdgroup step keeps the
// accumulators few, so many simdgroups stay resident to cover load latency.
// Rows are longer than top_k >= 8 keys: a short last block is computed as
// the row's last 8 keys and only its own keys are written.
#define V41_IDX_MMA_ROUNDS 8u
#define V41_IDX_MMA_KEYS (2u * 8u * V41_IDX_MMA_ROUNDS)
kernel void kernel_deepseek_v41_attention_index_heads_rows_mma(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    constant uint4 &c [[buffer(2)]], device const float *q [[buffer(3)]],
    device const float *weights [[buffer(4)]], device const float *keys [[buffer(5)]],
    device float *scores [[buffer(6)]], constant uint &row0 [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ws[256];
    threadgroup float partial[V41_IDX_MMA_KEYS * 4u];
    threadgroup float scratch[8u * 64u];
    const uint local = group.y;
    if (local >= c.y) return;
    const uint row = row0 + local;
    const uint length = (a.x + row + 1) / a.y;
    const uint first = group.x * V41_IDX_MMA_KEYS;
    if (length <= a.z || length < 8u || first >= length) return;
    device const float *qrow = q + (ulong)row * b.z;
    device const float *wrow = weights + (ulong)row * b.w;
    const float scale = rsqrt(float(b.x)) * rsqrt(float(b.y));
    // Head group h's weight tile [8 heads x 8]: every column is
    // w[8h .. 8h + 7] * scale.
    ws[tid] = wrow[(tid / 64u) * 8u + (tid % 64u) / 8u] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint hg = sg % 4u, kg = sg / 4u;
    simdgroup_float8x8 qt[16];
    for (uint i = 0; i < 16u; ++i)
        simdgroup_load(qt[i], qrow + hg * 8u * 128u + i * 8u, 128, ulong2(0, 0), true); // [8 dims x 8 heads]
    simdgroup_float8x8 wc;
    simdgroup_load(wc, ws + hg * 64u, 8);
    threadgroup float *o = scratch + sg * 64u;
    for (uint round = 0; round < V41_IDX_MMA_ROUNDS; ++round) {
        const uint key0 = first + (round * 2u + kg) * 8u;
        if (key0 >= length) break;
        const uint base = min(key0, length - 8u);
        device const float *kp = keys + (ulong)base * 128u;
        simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
        for (uint i = 0; i < 16u; ++i) {
            simdgroup_float8x8 km;
            simdgroup_load(km, kp + i * 8u, 128); // [8 keys x 8 dims]
            simdgroup_multiply_accumulate(acc, km, qt[i], acc);
        }
        acc.thread_elements()[0] = max(acc.thread_elements()[0], 0.0f);
        acc.thread_elements()[1] = max(acc.thread_elements()[1], 0.0f);
        simdgroup_float8x8 r;
        simdgroup_multiply(r, acc, wc);
        simdgroup_store(r, o, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint key = base + lane;
        if (lane < 8u && key >= key0 && key < length) partial[(key - first) * 4u + hg] = o[lane * 8u];
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *srow = scores + (ulong)local * c.x;
    for (uint j = tid; j < V41_IDX_MMA_KEYS; j += 256u) {
        const uint key = first + j;
        if (key < length)
            srow[key] = partial[j * 4u] + partial[j * 4u + 1u] + partial[j * 4u + 2u] + partial[j * 4u + 3u];
    }
}

kernel void kernel_deepseek_v41_attention_blocks(
    constant uint2 &a [[buffer(0)]], device const float *scores [[buffer(1)]],
    device float *blocks [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    uint count = (a.x + a.y - 1) / a.y;
    if (t >= count) return;
    float score = -INFINITY;
    for (uint j = t * a.y; j < min((t + 1) * a.y, a.x); ++j) score = max(score, scores[j]);
    blocks[t] = t + 1 == count ? INFINITY : score;
}

kernel void kernel_deepseek_v41_attention_sort_init(
    constant uint4 &a [[buffer(0)]], device const float *scores [[buffer(1)]],
    device const uint *mask [[buffer(2)]], device float *scratch [[buffer(3)]],
    device uint *ids [[buffer(4)]], device atomic_uint *status [[buffer(5)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= a.y) return;
    float score = t < a.x && (!a.z || mask[t / a.w]) ? scores[t] : -INFINITY;
    if (isnan(score)) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        score = -INFINITY;
    }
    scratch[t] = score;
    ids[t] = t;
}

kernel void kernel_deepseek_v41_attention_sort_step(
    constant uint4 &a [[buffer(0)]], device float *scores [[buffer(1)]],
    device uint *ids [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    uint other = t ^ a.y;
    if (t >= a.x || other <= t) return;
    float l = scores[t], r = scores[other];
    uint li = ids[t], ri = ids[other];
    bool better = l > r || (l == r && (as_type<uint>(l) == as_type<uint>(r)
        ? li < ri : as_type<uint>(l) < as_type<uint>(r)));
    if (better != ((t & a.z) == 0)) {
        scores[t] = r; scores[other] = l;
        ids[t] = ri; ids[other] = li;
    }
}

// One threadgroup per top-k row: load the row's score slab, run the same
// bitonic network and tie-break as sort_init/sort_tile in threadgroup memory,
// then emit chronological IDs directly. Padded visible lengths up to 2048.
kernel void kernel_deepseek_v41_attention_select_fused_rows(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *scores [[buffer(2)]], device uint *selected [[buffer(3)]],
    device atomic_uint *status [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    // a: start, ratio, index_top_k, score stride. b: row0, rows, 0, 0.
    uint local = group.y;
    if (local >= b.y) return;
    uint row = b.x + local;
    uint length = (a.x + row + 1) / a.y;
    if (length <= a.z) return; // all-selected rows are already prepared
    uint padded = 1;
    while (padded < length) padded <<= 1;
    if (padded > 2048) return; // host keeps the per-row fallback
    threadgroup float values[2048];
    threadgroup uint order[2048];
    device const float *srow = scores + (ulong)local * a.w;
    for (uint t = tid; t < padded; t += 256) {
        float v = t < length ? srow[t] : -INFINITY;
        if (isnan(v)) {
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
            v = -INFINITY;
        }
        values[t] = v;
        order[t] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2; width <= padded; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            for (uint j = tid; j < padded; j += 256) {
                uint other = j ^ stride;
                if (other <= j) continue;
                float l = values[j], r = values[other];
                uint li = order[j], ri = order[other];
                bool better = l > r || (l == r && (as_type<uint>(l) == as_type<uint>(r)
                    ? li < ri : as_type<uint>(l) < as_type<uint>(r)));
                if (better != ((j & width) == 0)) {
                    values[j] = r; values[other] = l;
                    order[j] = ri; order[other] = li;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint t = tid; t < a.z; t += 256) {
        uint id = order[t];
        uint rank = 0;
        for (uint j = 0; j < a.z; ++j) rank += order[j] < id;
        selected[(ulong)row * a.z + rank] = id;
    }
}

// One threadgroup per top-k row: load the row's score slab, run the same
// bitonic network and tie-break as sort_init/sort_tile in threadgroup memory,
// then emit chronological IDs directly. Padded visible lengths up to 4096.
kernel void kernel_deepseek_v41_attention_select_fused_rows_4k(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *scores [[buffer(2)]], device uint *selected [[buffer(3)]],
    device atomic_uint *status [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    // a: start, ratio, index_top_k, score stride. b: row0, rows, 0, 0.
    uint local = group.y;
    if (local >= b.y) return;
    uint row = b.x + local;
    uint length = (a.x + row + 1) / a.y;
    if (length <= a.z) return; // all-selected rows are already prepared
    uint padded = 1;
    while (padded < length) padded <<= 1;
    if (padded > 4096) return; // host keeps the per-row fallback
    threadgroup float values[4096];
    threadgroup uint order[4096];
    device const float *srow = scores + (ulong)local * a.w;
    for (uint t = tid; t < padded; t += 256) {
        float v = t < length ? srow[t] : -INFINITY;
        if (isnan(v)) {
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
            v = -INFINITY;
        }
        values[t] = v;
        order[t] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2; width <= padded; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            for (uint j = tid; j < padded; j += 256) {
                uint other = j ^ stride;
                if (other <= j) continue;
                float l = values[j], r = values[other];
                uint li = order[j], ri = order[other];
                bool better = l > r || (l == r && (as_type<uint>(l) == as_type<uint>(r)
                    ? li < ri : as_type<uint>(l) < as_type<uint>(r)));
                if (better != ((j & width) == 0)) {
                    values[j] = r; values[other] = l;
                    order[j] = ri; order[other] = li;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint t = tid; t < a.z; t += 256) {
        uint id = order[t];
        uint rank = 0;
        for (uint j = 0; j < a.z; ++j) rank += order[j] < id;
        selected[(ulong)row * a.z + rank] = id;
    }
}

// A total order after NaN normalization: numerical descending, equal-value bits
// ascending (+0 before -0), then original absolute key ID ascending.
inline bool attention_select_better(float l, uint li, float r, uint ri) {
    return l > r || (l == r && (as_type<uint>(l) == as_type<uint>(r)
        ? li < ri : as_type<uint>(l) < as_type<uint>(r)));
}

inline void attention_select_best_first(threadgroup float *values,
    threadgroup uint *order, uint count, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2; width <= count; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            for (uint j = tid; j < count; j += 256) {
                uint other = j ^ stride;
                if (other <= j) continue;
                float l = values[j], r = values[other];
                uint li = order[j], ri = order[other];
                // Each independent network ends best-first. In particular do
                // not use the legacy global-base alternating tile orientation.
                if (attention_select_better(l, li, r, ri) != ((j & width) == 0)) {
                    values[j] = r; values[other] = l;
                    order[j] = ri; order[other] = li;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// a: absolute chunk start, ratio, score stride, partitions (2 or 4).
// b: first output row, tile rows, 0, 0. Pairs: [tile row, partition, 512].
kernel void kernel_deepseek_v41_attention_select_hier_local(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *scores [[buffer(2)]], device uint2 *pairs [[buffer(3)]],
    device atomic_uint *status [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint local = group.y, partition = group.x;
    if (local >= b.y || partition >= a.w) return;
    uint length = (a.x + b.x + local + 1) / a.y;
    if (length <= 512) return; // prepare owns all-selected rows
    threadgroup float values[4096];
    threadgroup uint order[4096];
    for (uint j = tid; j < 4096; j += 256) {
        uint id = partition * 4096 + j;
        // Never even read future published scores, including poison NaNs.
        float v = id < length ? scores[(ulong)local * a.z + id] : -INFINITY;
        if (isnan(v)) {
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
            v = -INFINITY;
        }
        values[j] = v;
        // Padding IDs exceed every visible ID, so even real -Inf beats padding.
        order[j] = id;
    }
    attention_select_best_first(values, order, 4096, tid);
    for (uint j = tid; j < 512; j += 256)
        pairs[((ulong)local * a.w + partition) * 512 + j] =
            uint2(as_type<uint>(values[j]), order[j]);
}

kernel void kernel_deepseek_v41_attention_select_hier_merge(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const uint2 *pairs [[buffer(2)]], device uint *selected [[buffer(3)]],
    threadgroup float *values [[threadgroup(0)]],
    threadgroup uint *order [[threadgroup(1)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint local = group.y, row = b.x + local;
    if (local >= b.y || (a.x + row + 1) / a.y <= 512) return;
    uint count = a.w * 512;
    for (uint j = tid; j < count; j += 256) {
        uint2 pair = pairs[(ulong)local * count + j];
        values[j] = as_type<float>(pair.x);
        order[j] = pair.y;
    }
    attention_select_best_first(values, order, count, tid);
    for (uint j = tid; j < 512; j += 256) {
        uint id = order[j], rank = 0;
        for (uint k = 0; k < 512; ++k) rank += order[k] < id;
        selected[(ulong)row * 512 + rank] = id;
    }
}

// Exact top-k by radix selection instead of a bitonic sort network.
//
// The order is attention_select_better's: numerically descending, equal values
// with smaller bits first (+0 before -0), then smaller key ID. Mapping a score
// to `radix_key` (sign-folded bits) makes that the order of (key descending,
// ID ascending). Masked keys (outside the row's candidate blocks) and NaNs are
// -Inf, exactly as sort_init does; NaNs set status bit 1.
//
// One 1024-thread threadgroup per row: three digit passes (11/11/10 bits) find
// the k-th largest key T and how many keys equal to T are needed, then two
// passes over contiguous ID chunks emit every key > T and the lowest-ID keys ==
// T in ascending ID order, i.e. the chronological list the sort path writes.
#define V41_RADIX_THREADS 1024u

inline uint radix_key(float v) {
    const uint u = as_type<uint>(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Keys are re-derived from the scores on every pass (no length-sized on-chip
// copy); only the first pass reports NaNs.
inline uint radix_row_key(device const float *srow, device const uint *crow, uint t,
                          bool masked, uint block, bool report, device atomic_uint *status) {
    if (masked && crow[t / block] == 0) return radix_key(-INFINITY);
    float v = srow[t];
    if (isnan(v)) {
        if (report) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        v = -INFINITY;
    }
    return radix_key(v);
}

// Exclusive prefix sum over the 1024 threads in thread order, plus the total.
inline uint radix_exclusive_sum(uint v, threadgroup uint *partials, uint sg, uint lane,
                                thread uint &total) {
    const uint inclusive = simd_prefix_inclusive_sum(v);
    if (lane == 31) partials[sg] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint p = partials[lane];
        const uint p_inclusive = simd_prefix_inclusive_sum(p);
        partials[lane] = p_inclusive - p;
        if (lane == 31) partials[32] = p_inclusive;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint exclusive = partials[sg] + inclusive - v;
    total = partials[32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return exclusive;
}

// a: chunk start, ratio, top_k, candidate block. b: score stride, candidate
// stride, rows, mask (1: restrict pruned rows to their candidate blocks).
// c: selected stride, row0, candidate top-k (blocks), mode. A row is pruned
// when its visible length spans more than candidate top-k blocks; unpruned rows
// are never masked. Mode 0 selects top_k key IDs from the row's index scores;
// rows whose visible length is at most top_k are prepared elsewhere and skip.
// Mode 1 publishes the candidate mask of a pruned row: `scores` holds its block
// scores (`select_block_scores_rows`), the domain is its block count, k is the
// candidate top-k, and every block's mask entry is written (1 kept, 0 not), the
// same set the block sort plus `keep` chain keeps.
[[max_total_threads_per_threadgroup(1024)]]
kernel void kernel_deepseek_v41_attention_select_radix_rows(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    constant uint4 &c [[buffer(2)]], device const float *scores [[buffer(3)]],
    device uint *candidates [[buffer(4)]], device uint *selected [[buffer(5)]],
    device atomic_uint *status [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint hist[2048];
    threadgroup uint partials[33];
    threadgroup uint found[2];
    const uint local = group.y;
    if (local >= b.z) return;
    const uint row = c.y + local;
    const uint visible = (a.x + row + 1) / a.y;
    const uint visible_blocks = (visible + a.w - 1u) / a.w;
    const bool publish = c.w != 0;
    const uint length = publish ? visible_blocks : visible;
    const uint k = publish ? c.z : a.z;
    if (length <= k) return;
    device const float *srow = scores + (ulong)local * b.x;
    device uint *crow = candidates + (ulong)row * b.y;
    const bool masked = !publish && b.w != 0 && visible_blocks > c.z;

    uint prefix = 0, prefix_mask = 0, remaining = k;
    const uint shifts[3] = {21u, 10u, 0u};
    const uint widths[3] = {11u, 11u, 10u};
    for (uint pass = 0; pass < 3; ++pass) {
        const uint shift = shifts[pass], bins = 1u << widths[pass], digit = bins - 1u;
        for (uint j = tid; j < bins; j += V41_RADIX_THREADS)
            atomic_store_explicit(hist + j, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = tid; t < length; t += V41_RADIX_THREADS) {
            const uint key = radix_row_key(srow, crow, t, masked, a.w, pass == 0, status);
            if ((key & prefix_mask) == prefix)
                atomic_fetch_add_explicit(hist + ((key >> shift) & digit), 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Thread i owns bins (descending) [bins-1 - i*per, ..., bins - (i+1)*per].
        const uint per = bins / V41_RADIX_THREADS;
        uint mine = 0;
        for (uint j = 0; j < per; ++j)
            mine += atomic_load_explicit(hist + (bins - 1u - (tid * per + j)), memory_order_relaxed);
        uint total;
        uint above = radix_exclusive_sum(mine, partials, sg, lane, total);
        if (above < remaining && remaining <= above + mine) {
            for (uint j = 0; j < per; ++j) {
                const uint bin = bins - 1u - (tid * per + j);
                const uint n = atomic_load_explicit(hist + bin, memory_order_relaxed);
                if (above + n >= remaining) {
                    found[0] = bin;
                    found[1] = above;
                    break;
                }
                above += n;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= found[0] << shift;
        prefix_mask |= digit << shift;
        remaining -= found[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // prefix is now the k-th largest key T; `remaining` keys equal to T are needed.
    const uint threshold = prefix, need_equal = remaining;
    const uint chunk = (length + V41_RADIX_THREADS - 1u) / V41_RADIX_THREADS;
    const uint lo = min(length, tid * chunk), hi = min(length, lo + chunk);
    uint greater = 0, equal = 0;
    for (uint t = lo; t < hi; ++t) {
        const uint key = radix_row_key(srow, crow, t, masked, a.w, false, status);
        greater += key > threshold;
        equal += key == threshold;
    }
    uint total;
    const uint equal_before = radix_exclusive_sum(equal, partials, sg, lane, total);
    const uint equal_taken = equal_before >= need_equal ? 0u : min(equal, need_equal - equal_before);
    uint out = radix_exclusive_sum(greater + equal_taken, partials, sg, lane, total);
    device uint *dst = selected + (ulong)row * c.x;
    uint equal_seen = equal_before;
    for (uint t = lo; t < hi; ++t) {
        const uint key = radix_row_key(srow, crow, t, masked, a.w, false, status);
        bool take = key > threshold;
        if (key == threshold) {
            take = equal_seen < need_equal;
            ++equal_seen;
        }
        if (publish) crow[t] = take ? 1u : 0u;
        else if (take) dst[out++] = t;
    }
}

// select_radix_rows split across threadgroups for long rows (decode has one
// row, which one threadgroup leaves the GPU mostly idle on). Same arguments
// a/b/c, order, masks, NaN status, modes and outputs. d: digit pass, keys per
// threadgroup (`chunk`, at most 4 per thread), scratch row stride (words), 0.
// Grid (chunks, rows); 512 threads per threadgroup.
// Each dispatch-local row owns a scratch row: global digit histograms 0..2
// (2048, 2048, 1024 bins), the selection state, and per chunk a 1025-entry
// table whose entry j counts the chunk's keys >= (top 22 bits of T) | j
// (j = 1024: keys above that 22-bit prefix).
//
// grid_hist, one dispatch per digit pass p over (chunks, rows): every
// threadgroup histograms digit p of its chunk's keys that match the prefix
// so far in threadgroup memory and adds the nonzero bins to global
// histogram p. Passes 1 and 2 first find the previous digit in the previous
// histogram (redundantly per threadgroup; threadgroup 0 records the state);
// pass 2 also writes its chunk's table. grid_place finds the last digit,
// i.e. the k-th largest key T and how many keys equal to T are needed, takes
// the keys > T and == T in earlier chunks from their tables, and writes in
// ascending ID order every key > T and the lowest-ID keys == T (mode 0) or
// every block's mask entry (mode 1). Keys are loaded once per pass, before
// any dependent work.
//
// Clearing: pass 0 zeroes histograms 1 and 2, and pass 2 zeroes histogram 0
// for the next selection, so scratch must start zeroed (fresh allocations
// are) and each pass 0 is followed by pass 2 on the same scratch rows. Pass 1
// sets status bit 8 when histogram 0 does not count exactly the row's keys.
#define V41_RADIX_GRID_STATE 5120u
#define V41_RADIX_GRID_TABLES 5128u
#define V41_RADIX_GRID_TABLE 1028u
#define V41_RADIX_GRID_KEYS 4u
#define V41_RADIX_NEG_INF 0x007fffffu

struct RadixGridRow {
    uint row, length, k;
    bool publish, masked;
};

// The row of dispatch-local row `local`; false when it selects nothing here.
inline bool radix_grid_row(constant uint4 &a, constant uint4 &b, constant uint4 &c,
                           uint local, thread RadixGridRow &r) {
    if (local >= b.z) return false;
    r.row = c.y + local;
    const uint visible = (a.x + r.row + 1) / a.y;
    const uint visible_blocks = (visible + a.w - 1u) / a.w;
    r.publish = c.w != 0;
    r.length = r.publish ? visible_blocks : visible;
    r.k = r.publish ? c.z : a.z;
    r.masked = !r.publish && b.w != 0 && visible_blocks > c.z;
    return r.length > r.k;
}

// Exclusive prefix sum over a THREADS-thread group in thread order, plus the
// total (radix_exclusive_sum for any multiple of 32 up to 1024).
template <uint THREADS>
inline uint radix_grid_exclusive_sum(uint v, threadgroup uint *partials, uint sg, uint lane,
                                     thread uint &total) {
    constexpr uint groups = THREADS / 32u;
    const uint inclusive = simd_prefix_inclusive_sum(v);
    if (lane == 31) partials[sg] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint p = lane < groups ? partials[lane] : 0u;
        const uint p_inclusive = simd_prefix_inclusive_sum(p);
        if (lane < groups) partials[lane] = p_inclusive - p;
        if (lane == 31) partials[32] = p_inclusive;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint exclusive = partials[sg] + inclusive - v;
    total = partials[32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return exclusive;
}

// A chunk's keys, at most V41_RADIX_GRID_KEYS per thread (ID lo + i * THREADS
// + tid), loaded before any dependent work: radix_row_key's mapping with a
// bit-level NaN test (flagged, reported by the caller). Past `hi`: invalid.
struct RadixGridKeys {
    uint key[V41_RADIX_GRID_KEYS];
    bool valid[V41_RADIX_GRID_KEYS];
};

template <uint THREADS>
inline RadixGridKeys radix_grid_load(device const float *srow, device const uint *crow,
                                     uint lo, uint hi, uint tid, bool masked, uint block,
                                     thread bool &nan) {
    RadixGridKeys k;
    const uint slots = (hi - lo + THREADS - 1u) / THREADS;
    for (uint i = 0; i < V41_RADIX_GRID_KEYS; ++i) {
        k.valid[i] = false;
        k.key[i] = 0;
        if (i >= slots) continue; // uniform
        const uint t = lo + i * THREADS + tid;
        k.valid[i] = t < hi;
        const uint u = k.valid[i] ? as_type<uint>(srow[t]) : 0u;
        const bool kept = !masked || !k.valid[i] || crow[t / block] != 0;
        const bool is_nan = (u & 0x7fffffffu) > 0x7f800000u;
        nan = nan || (k.valid[i] && kept && is_nan);
        k.key[i] = !kept || is_nan ? V41_RADIX_NEG_INF
            : ((u & 0x80000000u) ? ~u : (u | 0x80000000u));
    }
    return k;
}

// The bin (of 2048 or 1024, scanned descending) holding the `remaining`-th
// largest key and the count above it; `total` is the histogram's sum.
template <uint THREADS>
inline uint2 radix_grid_find(device const uint *hist, uint bins, uint remaining,
                             threadgroup uint *partials, threadgroup uint *found,
                             uint tid, uint sg, uint lane, thread uint &total) {
    if (tid == 0) {
        found[0] = 0;
        found[1] = 0;
    }
    const uint per = bins / THREADS;
    uint mine = 0;
    for (uint j = 0; j < per; ++j) mine += hist[bins - 1u - (tid * per + j)];
    uint above = radix_grid_exclusive_sum<THREADS>(mine, partials, sg, lane, total);
    if (above < remaining && remaining <= above + mine) {
        for (uint j = 0; j < per; ++j) {
            const uint bin = bins - 1u - (tid * per + j);
            const uint n = hist[bin];
            if (above + n >= remaining) {
                found[0] = bin;
                found[1] = above;
                break;
            }
            above += n;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return uint2(found[0], found[1]);
}

#define V41_RADIX_GRID_ARGS                                                                        \
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],                             \
    constant uint4 &c [[buffer(2)]], device const float *scores [[buffer(3)]],                    \
    device uint *candidates [[buffer(4)]], device uint *selected [[buffer(5)]],                   \
    device atomic_uint *status [[buffer(6)]], device uint *scratch [[buffer(7)]],                 \
    constant uint4 &d [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],               \
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],         \
    uint lane [[thread_index_in_simdgroup]]
#define V41_RADIX_GRID_PARAMS                                                                      \
    constant uint4 &a, constant uint4 &b, constant uint4 &c, device const float *scores,          \
    device uint *candidates, device uint *selected, device atomic_uint *status,                   \
    device uint *scratch, constant uint4 &d, uint3 group, uint tid, uint sg, uint lane
#define V41_RADIX_GRID_FORWARD                                                                     \
    a, b, c, scores, candidates, selected, status, scratch, d, group, tid, sg, lane

template <uint THREADS>
inline void radix_grid_hist(V41_RADIX_GRID_PARAMS, threadgroup atomic_uint *hist,
                            threadgroup uint *partials, threadgroup uint *found,
                            threadgroup uint *sums) {
    RadixGridRow r;
    if (!radix_grid_row(a, b, c, group.y, r)) return;
    const uint pass = d.x, chunk = d.y, lo = group.x * chunk;
    if (lo >= r.length) return;
    const uint hi = min(r.length, lo + chunk);
    const uint groups = (r.length + chunk - 1u) / chunk;
    device const float *srow = scores + (ulong)group.y * b.x;
    device const uint *crow = candidates + (ulong)r.row * b.y;
    device uint *sc = scratch + (ulong)group.y * d.z;
    const bool leader = group.x == 0 && tid == 0;
    bool nan = false;
    const RadixGridKeys keys = radix_grid_load<THREADS>(srow, crow, lo, hi, tid, r.masked, a.w, nan);
    if (pass == 0 && simd_any(nan) && lane == 0)
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    const uint bins = pass == 2 ? 1024u : 2048u;
    for (uint j = tid; j < bins; j += THREADS) atomic_store_explicit(hist + j, 0u, memory_order_relaxed);

    uint prefix = 0, prefix_mask = 0, shift = 21u;
    if (pass == 1) {
        uint total;
        const uint2 f = radix_grid_find<THREADS>(sc, 2048u, r.k, partials, found, tid, sg, lane, total);
        prefix = f.x << 21;
        prefix_mask = 0xffe00000u;
        shift = 10u;
        if (leader) {
            if (total != r.length) atomic_fetch_or_explicit(status, 8u, memory_order_relaxed);
            sc[V41_RADIX_GRID_STATE] = prefix;
            sc[V41_RADIX_GRID_STATE + 1u] = r.k - f.y;
        }
    } else if (pass == 2) {
        prefix = sc[V41_RADIX_GRID_STATE];
        const uint remaining = sc[V41_RADIX_GRID_STATE + 1u];
        uint total;
        const uint2 f = radix_grid_find<THREADS>(sc + 2048u, 2048u, remaining, partials, found, tid,
                                                 sg, lane, total);
        prefix |= f.x << 10;
        prefix_mask = 0xfffffc00u;
        shift = 0u;
        if (leader) {
            sc[V41_RADIX_GRID_STATE + 2u] = prefix;
            sc[V41_RADIX_GRID_STATE + 3u] = remaining - f.y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint digit = bins - 1u;
    const uint slots = (hi - lo + THREADS - 1u) / THREADS;
    uint above = 0;
    for (uint i = 0; i < V41_RADIX_GRID_KEYS; ++i) {
        if (i >= slots) break; // uniform
        const uint key = keys.key[i];
        const bool match = keys.valid[i] && (key & prefix_mask) == prefix;
        above += keys.valid[i] && (key & prefix_mask) > prefix;
        // Runs of equal keys (masked blocks, -Inf, ties) would serialize on
        // one bin: keys equal to lane 0's matching key, and -Inf keys, are
        // added with one atomic each.
        const uint lead = simd_broadcast_first(match ? key : 0xffffffffu);
        const bool same = match && key == lead;
        const bool low = match && !same && key == V41_RADIX_NEG_INF;
        const uint n_same = simd_sum(uint(same)), n_low = simd_sum(uint(low));
        if (match && !same && !low)
            atomic_fetch_add_explicit(hist + ((key >> shift) & digit), 1u, memory_order_relaxed);
        if (lane == 0) {
            if (n_same)
                atomic_fetch_add_explicit(hist + ((lead >> shift) & digit), n_same, memory_order_relaxed);
            if (n_low)
                atomic_fetch_add_explicit(hist + ((V41_RADIX_NEG_INF >> shift) & digit), n_low,
                                          memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device atomic_uint *global = (device atomic_uint *)(sc + pass * 2048u);
    for (uint j = tid; j < bins; j += THREADS) {
        const uint n = atomic_load_explicit(hist + j, memory_order_relaxed);
        if (n) atomic_fetch_add_explicit(global + j, n, memory_order_relaxed);
    }
    if (pass == 2) {
        // Table entry j: keys above the 22-bit prefix plus matching keys with
        // digit >= j. Thread i owns bins 1023 - i * per, ... (descending).
        above = simd_sum(above);
        if (lane == 0) sums[sg] = above;
        constexpr uint per = 1024u / THREADS;
        uint n[per], mine = 0;
        for (uint j = 0; j < per; ++j) {
            n[j] = atomic_load_explicit(hist + (1023u - (tid * per + j)), memory_order_relaxed);
            mine += n[j];
        }
        uint total;
        uint running = radix_grid_exclusive_sum<THREADS>(mine, partials, sg, lane, total);
        uint greater = 0;
        for (uint j = 0; j < THREADS / 32u; ++j) greater += sums[j];
        device uint *table = sc + V41_RADIX_GRID_TABLES + group.x * V41_RADIX_GRID_TABLE;
        for (uint j = 0; j < per; ++j) {
            running += n[j];
            table[1023u - (tid * per + j)] = greater + running;
        }
        if (tid == 0) table[1024] = greater;
    }
    if (pass != 1) {
        // Pass 0: histograms 1 and 2 for this selection; pass 2: histogram 0
        // for the next one (pass 1 was its last reader).
        const uint first = pass == 0 ? 2048u : 0u, words = pass == 0 ? 3072u : 2048u;
        for (uint j = group.x * THREADS + tid; j < words; j += groups * THREADS) sc[first + j] = 0;
    }
}

template <uint THREADS>
inline void radix_grid_place(V41_RADIX_GRID_PARAMS, threadgroup uint *partials,
                             threadgroup uint *found, threadgroup uint *sums) {
    RadixGridRow r;
    if (!radix_grid_row(a, b, c, group.y, r)) return;
    const uint chunk = d.y, lo = group.x * chunk;
    if (lo >= r.length) return;
    const uint hi = min(r.length, lo + chunk);
    device const float *srow = scores + (ulong)group.y * b.x;
    device uint *crow = candidates + (ulong)r.row * b.y;
    device const uint *sc = scratch + (ulong)group.y * d.z;
    bool nan = false;
    const RadixGridKeys keys = radix_grid_load<THREADS>(srow, crow, lo, hi, tid, r.masked, a.w, nan);
    const uint prefix = sc[V41_RADIX_GRID_STATE + 2u];
    const uint remaining = sc[V41_RADIX_GRID_STATE + 3u];
    uint total;
    const uint2 f = radix_grid_find<THREADS>(sc + 4096u, 1024u, remaining, partials, found, tid, sg,
                                             lane, total);
    // T and how many keys equal to T are taken (lowest IDs first).
    const uint threshold = prefix | f.x, need_equal = remaining - f.y;
    // Keys > T and == T in earlier chunks, from their tables.
    uint greater = 0, equal = 0;
    for (uint j = tid; j < group.x; j += THREADS) {
        device const uint *table = sc + V41_RADIX_GRID_TABLES + j * V41_RADIX_GRID_TABLE;
        const uint at_least = table[f.x], above = table[f.x + 1u];
        greater += above;
        equal += at_least - above;
    }
    greater = simd_sum(greater);
    equal = simd_sum(equal);
    if (lane == 0) {
        sums[sg] = greater;
        sums[32u + sg] = equal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint equal_seen = 0, out = 0;
    for (uint j = 0; j < THREADS / 32u; ++j) {
        out += sums[j];
        equal_seen += sums[32u + j];
    }
    out += min(equal_seen, need_equal);
    device uint *dst = selected + (ulong)r.row * c.x;
    for (uint i = 0; i < V41_RADIX_GRID_KEYS; ++i) {
        if (lo + i * THREADS >= hi) break; // uniform
        const uint t = lo + i * THREADS + tid;
        const uint key = keys.key[i];
        const bool gt = keys.valid[i] && key > threshold;
        const bool eq = keys.valid[i] && key == threshold;
        uint equal_total, take_total;
        const uint equal_rank =
            equal_seen + radix_grid_exclusive_sum<THREADS>(uint(eq), partials, sg, lane, equal_total);
        const bool take = gt || (eq && equal_rank < need_equal);
        const uint position =
            out + radix_grid_exclusive_sum<THREADS>(uint(take), partials, sg, lane, take_total);
        if (r.publish) {
            if (keys.valid[i]) crow[t] = take ? 1u : 0u;
        } else if (take && position < r.k) {
            dst[position] = t;
        }
        equal_seen += equal_total;
        out += take_total;
    }
}

// Instantiated at 512 threads (the fastest of 256, 512 and 1024 on M4 Max).
#define V41_RADIX_GRID_KERNELS(suffix, THREADS)                                                    \
[[max_total_threads_per_threadgroup(THREADS)]]                                                    \
kernel void kernel_deepseek_v41_attention_select_radix_grid_hist##suffix(V41_RADIX_GRID_ARGS) {   \
    threadgroup atomic_uint hist[2048];                                                            \
    threadgroup uint partials[33];                                                                 \
    threadgroup uint found[2];                                                                     \
    threadgroup uint sums[32];                                                                     \
    radix_grid_hist<THREADS>(V41_RADIX_GRID_FORWARD, hist, partials, found, sums);                \
}                                                                                                  \
[[max_total_threads_per_threadgroup(THREADS)]]                                                    \
kernel void kernel_deepseek_v41_attention_select_radix_grid_place##suffix(V41_RADIX_GRID_ARGS) {  \
    threadgroup uint partials[33];                                                                 \
    threadgroup uint found[2];                                                                     \
    threadgroup uint sums[64];                                                                     \
    radix_grid_place<THREADS>(V41_RADIX_GRID_FORWARD, partials, found, sums);                     \
}
V41_RADIX_GRID_KERNELS(, 512u)

// Per-block maxima of pruned rows' index scores for candidate publication:
// the `blocks` kernel per row (NaN ignored, newest block forced to +Inf).
// a: chunk start, ratio, candidate block, candidate top-k. b: score stride,
// block-score stride, rows, row0. One thread per block, grid (blocks, rows).
kernel void kernel_deepseek_v41_attention_select_block_scores_rows(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *scores [[buffer(2)]], device float *blocks_out [[buffer(3)]],
    uint2 t [[thread_position_in_grid]]) {
    const uint local = t.y;
    if (local >= b.z) return;
    const uint row = b.w + local;
    const uint length = (a.x + row + 1) / a.y;
    const uint blocks = (length + a.z - 1u) / a.z;
    if (blocks <= a.w || t.x >= blocks) return;
    device const float *srow = scores + (ulong)local * b.x;
    float score = -INFINITY;
    for (uint j = t.x * a.z; j < min((t.x + 1u) * a.z, length); ++j) score = max(score, srow[j]);
    blocks_out[(ulong)local * b.y + t.x] = t.x + 1u == blocks ? INFINITY : score;
}

kernel void kernel_deepseek_v41_attention_keep(
    constant uint2 &a [[buffer(0)]], device const uint *ids [[buffer(1)]],
    device uint *mask [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t >= a.x) return;
    uint keep = 0;
    for (uint j = 0; j < a.y; ++j) keep |= ids[j] == t;
    mask[t] = keep;
}

// One group constructs a temporary on-chip set; the published mask stays dense.
kernel void kernel_deepseek_v41_attention_keep_fused(
    constant uint2 &a [[buffer(0)]], device const uint *ids [[buffer(1)]],
    device uint *mask [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup atomic_uint bits[4096];
    // Uniform guard bounds the rounding addition and every local access.
    if (a.x <= 131072u) {
        uint words = (a.x + 31u) >> 5;
        for (uint j = tid; j < words; j += threads.x)
            atomic_store_explicit(bits + j, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ulong j = tid; j < a.y; j += threads.x) {
            uint id = ids[j];
            if (id < a.x)
                atomic_fetch_or_explicit(bits + (id >> 5), 1u << (id & 31u), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = tid; j < a.x; j += threads.x)
            mask[j] = (atomic_load_explicit(bits + (j >> 5), memory_order_relaxed) >> (j & 31u)) & 1u;
    }
}

// Experimental only: clear, scatter, then consumers must be ordered dispatches
// on a serial encoder. This preserves the dense mask and has no status output.
kernel void kernel_deepseek_v41_attention_keep_clear(
    constant uint2 &a [[buffer(0)]], device uint *mask [[buffer(2)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= a.x) return;
    mask[t] = 0;
}

kernel void kernel_deepseek_v41_attention_keep_scatter(
    constant uint2 &a [[buffer(0)]], device const uint *ids [[buffer(1)]],
    device atomic_uint *mask [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t >= a.y) return;
    uint id = ids[t];
    if (id >= a.x) return;
    // Duplicate IDs are legal in keep; equal non-atomic writes would race.
    atomic_store_explicit(mask + id, 1u, memory_order_relaxed);
}

// Fixture-only next-dispatch observation, before any CPU completion/readback.
kernel void kernel_deepseek_v41_attention_keep_observe(
    constant uint2 &a [[buffer(0)]], device const uint *mask [[buffer(2)]],
    device uint *observed [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    if (t >= a.x) return;
    observed[t] = mask[t];
}

kernel void kernel_deepseek_v41_attention_chronological(
    constant uint &k [[buffer(0)]], device const uint *ids [[buffer(1)]],
    device uint *out [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t >= k) return;
    uint rank = 0, id = ids[t];
    for (uint j = 0; j < k; ++j) rank += ids[j] < id;
    out[rank] = id;
}

// Incumbent for same-run tests; production uses the three passes below.
kernel void kernel_deepseek_v41_attention_sink(
    constant uint4 &a [[buffer(0)]], constant uint2 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    device const float *sink [[buffer(6)]], device float *scores [[buffer(7)]],
    device float *out [[buffer(8)]], constant uint &length [[buffer(9)]],
    device atomic_uint *status [[buffer(10)]], uint h [[thread_position_in_grid]]) {
    // a: dim, raw count, ring start, ring capacity; b: selected count, score stride.
    float maximum = sink[h], scale = rsqrt(float(a.x));
    for (uint i = 0; i < b.x; ++i) {
        if (ids[i] >= length) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
            for (uint j = 0; j < a.x; ++j) out[h * a.x + j] = 0;
            return;
        }
    }
    for (uint i = 0; i < a.y + b.x; ++i) {
        device const float *row = i < a.y ? raw + ((a.z + i) % a.w) * a.x : compressed + ids[i - a.y] * a.x;
        float dot = 0;
        for (uint j = 0; j < a.x; ++j) dot += q[h * a.x + j] * row[j];
        scores[h * b.y + i] = dot * scale;
        maximum = max(maximum, dot * scale);
    }
    float denom = 0;
    for (uint i = 0; i < a.y + b.x; ++i) {
        float s = exp(scores[h * b.y + i] - maximum);
        scores[h * b.y + i] = s; denom += s;
    }
    denom += exp(sink[h] - maximum);
    for (uint j = 0; j < a.x; ++j) {
        float value = 0;
        for (uint i = 0; i < a.y + b.x; ++i) {
            device const float *row = i < a.y ? raw + ((a.z + i) % a.w) * a.x : compressed + ids[i - a.y] * a.x;
            value += row[j] * scores[h * b.y + i] / denom;
        }
        out[h * a.x + j] = value;
        if (!isfinite(value)) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
    }
}

inline void attention_score(uint4 a, uint2 b, device const float *q,
    device const float *raw, device const float *compressed, device const uint *ids,
    device float *scores, uint length, uint2 group, uint sg, uint lane) {
    uint count = a.y + b.x, h = group.y, i = group.x * 8 + sg;
    if (i >= count) return;
    if (i >= a.y && ids[i - a.y] >= length) return;
    device const float *row = i < a.y ? raw + ((a.z + i) % a.w) * a.x : compressed + ids[i - a.y] * a.x;
    float dot = 0;
    for (uint j = lane; j < a.x; j += 32) dot += q[h * a.x + j] * row[j];
    dot = simd_sum(dot);
    if (lane == 0) scores[h * b.y + i] = dot * rsqrt(float(a.x));
}

inline void attention_softmax(uint4 a, uint2 b, device const uint *ids,
    device const float *sink, device float *scores, uint length,
    device atomic_uint *status, uint h, uint lane) {
    // Last slot per head holds the denominator, or a local invalid-ID sentinel.
    bool invalid = false;
    for (uint i = lane; i < b.x; i += 32) invalid |= ids[i] >= length;
    if (simd_any(invalid)) {
        if (lane == 0) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
            scores[h * b.y + b.y - 1] = -1.0f;
        }
        return;
    }
    float maximum = sink[h];
    for (uint i = lane; i < a.y + b.x; i += 32) maximum = max(maximum, scores[h * b.y + i]);
    maximum = simd_max(maximum);
    float denom = 0;
    for (uint i = lane; i < a.y + b.x; i += 32) {
        float s = exp(scores[h * b.y + i] - maximum);
        scores[h * b.y + i] = s; denom += s;
    }
    denom = simd_sum(denom) + exp(sink[h] - maximum);
    if (lane == 0) scores[h * b.y + b.y - 1] = denom;
}

inline void attention_matrix_values(uint4 a, uint2 b,
    device const float *raw, device const float *compressed, device const uint *ids,
    device const float *scores, uint length, uint2 m, device float *values,
    device float *probabilities, uint probability_stride, threadgroup float tile[16][17],
    uint2 group, uint tid) {
    uint j = group.y * 16 + tid % 16, i = group.x * 16 + tid / 16, count = a.y + b.x;
    float v = 0;
    if (j < a.x) {
        if (scores[b.y - 1] != -1.0f) {
            if (i < a.y) v = raw[((a.z + i) % a.w) * a.x + j];
            else if (i < count && ids[i - a.y] < length) v = compressed[ids[i - a.y] * a.x + j];
        }
    }
    tile[tid / 16][tid % 16] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    j = group.y * 16 + tid / 16; i = group.x * 16 + tid % 16;
    if (j < a.x) values[j * m.x + i] = tile[tid % 16][tid / 16];
    if (j < m.y) {
        float denom = scores[j * b.y + b.y - 1];
        probabilities[j * probability_stride + i] = i < count && denom != -1.0f ? scores[j * b.y + i] / denom : 0;
    }
}

inline void attention_output(uint4 a, uint2 b, device const float *raw,
    device const float *compressed, device const uint *ids, device const float *scores,
    device float *out, uint output_stride, device atomic_uint *status, uint t) {
    uint h = t / a.x, j = t % a.x;
    float denom = scores[h * b.y + b.y - 1], value = 0;
    if (denom != -1.0f) {
        for (uint i = 0; i < a.y + b.x; ++i) {
            device const float *row = i < a.y ? raw + ((a.z + i) % a.w) * a.x : compressed + ids[i - a.y] * a.x;
            value += row[j] * scores[h * b.y + i] / denom;
        }
    }
    out[h * output_stride + j] = value;
    if (!isfinite(value)) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
}

kernel void kernel_deepseek_v41_attention_score(
    constant uint4 &a [[buffer(0)]], constant uint2 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    device float *scores [[buffer(7)]], constant uint &length [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    attention_score(a, b, q, raw, compressed, ids, scores, length, group, sg, lane);
}

// attention_score register-tiled: each simdgroup computes an 8-head x 4-key
// tile, so each lane loads 8 Q and 4 key values per step for 32 products
// instead of 1 + 1 per product. Every (head, key) dot product runs
// attention_score's operations in its order (lane-strided products
// accumulated in j order, then simd_sum, then the rsqrt(dim) scale) on the
// same F32 values, so scores are bitwise equal. group.x: block of 32 keys
// (4 per simdgroup); group.y: block of 8 heads.
#define V41_SCORE_TILE_HEADS 8
#define V41_SCORE_TILE_KEYS 4
#define V41_SCORE_KEY_BLOCK 32

kernel void kernel_deepseek_v41_attention_score_keys(
    constant uint4 &a [[buffer(0)]], constant uint2 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    device float *scores [[buffer(7)]], constant uint &length [[buffer(9)]],
    constant uint &heads [[buffer(14)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint dim = a.x, count = a.y + b.x, h0 = group.y * V41_SCORE_TILE_HEADS;
    const uint hn = h0 < heads ? min(uint(V41_SCORE_TILE_HEADS), heads - h0) : 0u;
    bool valid[V41_SCORE_TILE_KEYS];
    device const float *row[V41_SCORE_TILE_KEYS];
    uint first = group.x * V41_SCORE_KEY_BLOCK + sg * V41_SCORE_TILE_KEYS;
    bool any = false;
    for (uint k = 0; k < V41_SCORE_TILE_KEYS; k++) {
        const uint i = first + k;
        valid[k] = i < count && (i < a.y || ids[i - a.y] < length);
        row[k] = !valid[k] ? raw
            : i < a.y ? raw + ((a.z + i) % a.w) * dim : compressed + ids[i - a.y] * dim;
        any |= valid[k];
    }
    if (!any || hn == 0) return;
    float acc[V41_SCORE_TILE_HEADS][V41_SCORE_TILE_KEYS];
    for (uint h = 0; h < V41_SCORE_TILE_HEADS; h++)
        for (uint k = 0; k < V41_SCORE_TILE_KEYS; k++) acc[h][k] = 0;
    device const float *qh = q + h0 * dim;
    for (uint j = lane; j < dim; j += 32) {
        float kv[V41_SCORE_TILE_KEYS], qv[V41_SCORE_TILE_HEADS];
        for (uint k = 0; k < V41_SCORE_TILE_KEYS; k++) kv[k] = row[k][j];
        for (uint h = 0; h < V41_SCORE_TILE_HEADS; h++) qv[h] = h < hn ? qh[h * dim + j] : 0.0f;
        for (uint h = 0; h < V41_SCORE_TILE_HEADS; h++)
            for (uint k = 0; k < V41_SCORE_TILE_KEYS; k++) acc[h][k] += qv[h] * kv[k];
    }
    const float scale = rsqrt(float(dim));
    for (uint h = 0; h < V41_SCORE_TILE_HEADS; h++) {
        if (h >= hn) break;
        for (uint k = 0; k < V41_SCORE_TILE_KEYS; k++) {
            const float sum = simd_sum(acc[h][k]);
            if (valid[k] && lane == 0) scores[(h0 + h) * b.y + first + k] = sum * scale;
        }
    }
}

kernel void kernel_deepseek_v41_attention_softmax(
    constant uint4 &a [[buffer(0)]], constant uint2 &b [[buffer(1)]],
    device const uint *ids [[buffer(5)]], device const float *sink [[buffer(6)]],
    device float *scores [[buffer(7)]], constant uint &length [[buffer(9)]],
    device atomic_uint *status [[buffer(10)]], uint h [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    attention_softmax(a, b, ids, sink, scores, length, status, h, lane);
}

kernel void kernel_deepseek_v41_attention_matrix_values(
    constant uint4 &a [[buffer(0)]], constant uint2 &b [[buffer(1)]],
    device const float *raw [[buffer(3)]], device const float *compressed [[buffer(4)]],
    device const uint *ids [[buffer(5)]], device const float *scores [[buffer(7)]],
    constant uint &length [[buffer(9)]], constant uint2 &m [[buffer(11)]],
    device float *values [[buffer(12)]], device float *probabilities [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[16][17];
    attention_matrix_values(a, b, raw, compressed, ids, scores, length, m,
        values, probabilities, m.x, tile, group, tid);
}

kernel void kernel_deepseek_v41_attention_output(
    constant uint4 &a [[buffer(0)]], constant uint2 &b [[buffer(1)]],
    device const float *raw [[buffer(3)]], device const float *compressed [[buffer(4)]],
    device const uint *ids [[buffer(5)]], device const float *scores [[buffer(7)]],
    device float *out [[buffer(8)]], device atomic_uint *status [[buffer(10)]],
    uint t [[thread_position_in_grid]]) {
    attention_output(a, b, raw, compressed, ids, scores, out, a.x, status, t);
}

// Fused decode sink attention, flash-decoding style (decode rows only; the
// prefill sink tile is separate): one partial dispatch plus one combine
// replace score -> softmax -> output / value matrix and the attended inverse
// RoPE of a decode row. Bindings 0-10 are the chain's; 14 = heads, 15 =
// split scratch, 16 = (keys per split, scratch splits, 0, 0), 17 / 18 = RoPE
// frequencies and (pairs, position) for the _rope combine. Scratch: per split
// the unnormalized partial rows (heads x dim), then per (split, head) the
// split maximum and exp sum, then one invalid-ID flag per split.
//
// Partial kernels: grid (ceil(count / KB), ceil(heads / HG)), 256 threads.
// Threadgroup (s, g) owns keys [s KB, s KB + KB) of the row (raw ring prefix,
// then the selected compressed rows) and heads [g HG, g HG + HG):
//   1. scores: simdgroup sg computes an HG-head x KB/8-key tile;
//   2. per head over the split's visible keys (rows past the count and
//      selected IDs >= length excluded): m = max, e = exp(s - m), l = sum e;
//   3. unnormalized PV, sum_k e[h][k] V[k][d], in key order.
// A split holding a selected ID >= length flags itself and sets status bit 2,
// as the chain's softmax does.
//   kernel_deepseek_v41_attention_flash_h8_k32[_d512]: lane-strided F32
//     scores with attention_score_keys' per-(head, key) arithmetic, so scores
//     are bitwise the chain's; _d512 fixes dim = 512 (constant trip counts).
//   kernel_deepseek_v41_attention_flash2_h4_k16: for short rows: the group's
//     query is staged once with float4 loads and every float4 load of a
//     simdgroup's keys is issued before any product (float4 dot steps per
//     lane, then simd_sum: a different F32 order); dim % 4 == 0.
// Combine: grid (ceil(dim / 256), heads), 256 threads. M = max(sink, m_s),
// L = exp(sink - M) + sum_s l_s exp(m_s - M): out = sum_s partial_s
// exp(m_s - M) / L, or 0 for every head when any split saw an invalid ID;
// non-finite (pre-RoPE) values set status bit 4. _rope also applies
// kernel_deepseek_v41_attention_rope's inverse rotation.
// F32 throughout; the softmax and PV summation orders differ from the chain.
// Decode probe on M5 Ultra (64 heads x 512, 128 raw + 512 selected keys):
// chain 42.0 us, fused 19.0 us; 128 + 0 keys: 55.9 / 11.9 us.
#define V41_FLASH_THREADS 256u
#define V41_FLASH_MAX_SPLITS 64u

inline uint v41_flash_scratch_rows(uint splits, uint heads, uint dim) { return splits * heads * dim; }

inline device const float *v41_flash_key(uint4 a, device const float *raw,
    device const float *compressed, device const uint *ids, uint i) {
    return i < a.y ? raw + ((a.z + i) % a.w) * a.x : compressed + ids[i - a.y] * a.x;
}

// Phase 2 for a threadgroup's split: e[h * KB + k] holds scores on entry and
// exp(s - m) (0 for hidden keys) on exit; writes m, l and the split flag.
template <ushort HG, ushort KB>
inline void v41_flash_split_softmax(threadgroup float *e, threadgroup const uchar *visible,
    threadgroup const uchar *bad_keys, device float *scratch, uint splits, uint heads,
    uint dim, uint split, uint h0, uint hn, bool first_group, device atomic_uint *status,
    uint sg, uint lane, uint tid) {
    device float *partial_m = scratch + v41_flash_scratch_rows(splits, heads, dim);
    device float *partial_l = partial_m + splits * heads;
    device float *partial_bad = partial_l + splits * heads;
    for (uint h = sg; h < HG; h += 8) {
        float m = -FLT_MAX;
        for (uint k = lane; k < KB; k += 32)
            if (visible[k]) m = max(m, e[h * KB + k]);
        m = simd_max(m);
        float l = 0;
        for (uint k = lane; k < KB; k += 32) {
            const float x = visible[k] ? exp(e[h * KB + k] - m) : 0.0f;
            e[h * KB + k] = x;
            l += x;
        }
        l = simd_sum(l);
        if (lane == 0 && h < hn) {
            partial_m[split * heads + h0 + h] = m;
            partial_l[split * heads + h0 + h] = l;
        }
    }
    bool any_bad = false;
    for (ushort s = 0; s < 8; s++) any_bad |= bad_keys[s] != 0;
    if (tid == 0) {
        partial_bad[split] = any_bad ? 1.0f : 0.0f;
        if (any_bad && first_group) atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    }
}

template <ushort HG, ushort KB, ushort DIM>
[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_deepseek_v41_attention_flash(
    constant uint4 &a [[buffer(0)]], constant uint2 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    constant uint &length [[buffer(9)]], device atomic_uint *status [[buffer(10)]],
    constant uint &heads [[buffer(14)]], device float *scratch [[buffer(15)]],
    constant uint4 &f [[buffer(16)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr ushort KPS = KB / 8;
    threadgroup float e[HG * KB];
    threadgroup uchar visible[KB];
    threadgroup uchar bad_keys[8];
    const uint dim = DIM ? uint(DIM) : a.x, count = a.y + b.x, split = group.x, h0 = group.y * HG;
    const uint hn = h0 < heads ? min(uint(HG), heads - h0) : 0u;
    const uint k0 = split * KB + sg * KPS;
    bool valid[KPS];
    device const float *row[KPS];
    bool bad = false;
    for (ushort k = 0; k < KPS; k++) {
        const uint i = k0 + k;
        const bool hidden = i >= a.y && i < count && ids[i - a.y] >= length;
        bad |= hidden;
        valid[k] = i < count && !hidden;
        row[k] = valid[k] ? v41_flash_key(a, raw, compressed, ids, i) : raw;
    }
    float acc[HG][KPS];
    for (ushort h = 0; h < HG; h++)
        for (ushort k = 0; k < KPS; k++) acc[h][k] = 0;
    device const float *qh = q + h0 * dim;
    _Pragma("clang loop unroll_count(4)")
    for (uint j = lane; j < dim; j += 32) {
        float kv[KPS], qv[HG];
        for (ushort k = 0; k < KPS; k++) kv[k] = row[k][j];
        for (ushort h = 0; h < HG; h++) qv[h] = h < hn ? qh[h * dim + j] : 0.0f;
        for (ushort h = 0; h < HG; h++)
            for (ushort k = 0; k < KPS; k++) acc[h][k] += qv[h] * kv[k];
    }
    const float scale = rsqrt(float(dim));
    for (ushort h = 0; h < HG; h++) {
        for (ushort k = 0; k < KPS; k++) {
            const float sum = simd_sum(acc[h][k]);
            if (lane == 0) e[h * KB + sg * KPS + k] = sum * scale;
        }
    }
    if (lane == 0) {
        for (ushort k = 0; k < KPS; k++) visible[sg * KPS + k] = valid[k];
        bad_keys[sg] = bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v41_flash_split_softmax<HG, KB>(e, visible, bad_keys, scratch, f.y, heads, dim, split, h0,
        hn, group.y == 0, status, sg, lane, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Keys in groups of four, loads first; hidden keys load 0 (their e is 0).
    float out[HG][2];
    for (ushort h = 0; h < HG; h++) out[h][0] = out[h][1] = 0;
    const uint d0 = tid, d1 = tid + V41_FLASH_THREADS;
    for (uint k4 = 0; k4 < KB; k4 += 4) {
        float v0[4], v1[4];
        for (ushort u = 0; u < 4; u++) {
            const uint k = k4 + u;
            const bool shown = visible[k] != 0;
            device const float *v = shown ? v41_flash_key(a, raw, compressed, ids, split * KB + k) : raw;
            v0[u] = shown && d0 < dim ? v[d0] : 0.0f;
            v1[u] = shown && d1 < dim ? v[d1] : 0.0f;
        }
        for (ushort u = 0; u < 4; u++) {
            for (ushort h = 0; h < HG; h++) {
                const float w = e[h * KB + k4 + u];
                out[h][0] = fma(w, v0[u], out[h][0]);
                out[h][1] = fma(w, v1[u], out[h][1]);
            }
        }
    }
    for (ushort h = 0; h < HG; h++) {
        if (h >= hn) break;
        device float *o = scratch + (ulong(split) * heads + h0 + h) * dim;
        if (d0 < dim) o[d0] = out[h][0];
        if (d1 < dim) o[d1] = out[h][1];
    }
}

template <ushort HG, ushort KB>
[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_deepseek_v41_attention_flash2(
    constant uint4 &a [[buffer(0)]], constant uint2 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    constant uint &length [[buffer(9)]], device atomic_uint *status [[buffer(10)]],
    constant uint &heads [[buffer(14)]], device float *scratch [[buffer(15)]],
    constant uint4 &f [[buffer(16)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr ushort KPS = KB / 8;
    threadgroup float4 q_staged[HG * 128];
    threadgroup float e[HG * KB];
    threadgroup uchar visible[KB];
    threadgroup uchar bad_keys[8];
    const uint dim = a.x, d4 = dim / 4, count = a.y + b.x, split = group.x, h0 = group.y * HG;
    const uint hn = h0 < heads ? min(uint(HG), heads - h0) : 0u;
    device const float4 *qh = (device const float4 *)(q + h0 * dim);
    for (uint t = tid; t < hn * d4; t += V41_FLASH_THREADS) q_staged[(t / d4) * 128 + t % d4] = qh[t];
    const uint k0 = split * KB + sg * KPS;
    bool valid[KPS];
    bool bad = false;
    float4 kv[KPS][4];
    for (ushort k = 0; k < KPS; k++) {
        const uint i = k0 + k;
        const bool hidden = i >= a.y && i < count && ids[i - a.y] >= length;
        bad |= hidden;
        valid[k] = i < count && !hidden;
        device const float4 *row = (device const float4 *)(valid[k]
            ? v41_flash_key(a, raw, compressed, ids, i) : raw);
        for (ushort t = 0; t < 4; t++) {
            const uint c = lane + 32 * t;
            kv[k][t] = valid[k] && c < d4 ? row[c] : float4(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[HG][KPS];
    for (ushort h = 0; h < HG; h++)
        for (ushort k = 0; k < KPS; k++) acc[h][k] = 0;
    for (ushort t = 0; t < 4; t++) {
        const uint c = lane + 32 * t;
        if (c >= d4) break;
        for (ushort h = 0; h < HG; h++) {
            const float4 qv = h < hn ? q_staged[h * 128 + c] : float4(0.0f);
            for (ushort k = 0; k < KPS; k++) acc[h][k] += dot(qv, kv[k][t]);
        }
    }
    const float scale = rsqrt(float(dim));
    for (ushort h = 0; h < HG; h++) {
        for (ushort k = 0; k < KPS; k++) {
            const float sum = simd_sum(acc[h][k]);
            if (lane == 0) e[h * KB + sg * KPS + k] = sum * scale;
        }
    }
    if (lane == 0) {
        for (ushort k = 0; k < KPS; k++) visible[sg * KPS + k] = valid[k];
        bad_keys[sg] = bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v41_flash_split_softmax<HG, KB>(e, visible, bad_keys, scratch, f.y, heads, dim, split, h0,
        hn, group.y == 0, status, sg, lane, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // PV: float2 column pairs, eight keys of loads in flight.
    float2 out[HG];
    for (ushort h = 0; h < HG; h++) out[h] = float2(0.0f);
    const uint c2 = tid, d2 = dim / 2;
    for (uint k8 = 0; k8 < KB; k8 += 8) {
        float2 v[8];
        for (ushort u = 0; u < 8; u++) {
            const uint k = k8 + u;
            const bool shown = visible[k] != 0;
            device const float2 *row = (device const float2 *)(shown
                ? v41_flash_key(a, raw, compressed, ids, split * KB + k) : raw);
            v[u] = shown && c2 < d2 ? row[c2] : float2(0.0f);
        }
        for (ushort u = 0; u < 8; u++)
            for (ushort h = 0; h < HG; h++) out[h] = fma(float2(e[h * KB + k8 + u]), v[u], out[h]);
    }
    for (ushort h = 0; h < HG; h++) {
        if (h >= hn) break;
        if (c2 < d2) ((device float2 *)(scratch + (ulong(split) * heads + h0 + h) * dim))[c2] = out[h];
    }
}

#define V41_FLASH(NAME, HG, KB, DIM) \
    template [[host_name("kernel_deepseek_v41_attention_" NAME)]] kernel void \
    kernel_deepseek_v41_attention_flash<HG, KB, DIM>(constant uint4 &, constant uint2 &, \
        device const float *, device const float *, device const float *, device const uint *, \
        constant uint &, device atomic_uint *, constant uint &, device float *, constant uint4 &, \
        uint2, uint, uint, uint);
V41_FLASH("flash_h8_k32", 8, 32, 0)
V41_FLASH("flash_h8_k32_d512", 8, 32, 512)
template [[host_name("kernel_deepseek_v41_attention_flash2_h4_k16")]] kernel void
kernel_deepseek_v41_attention_flash2<4, 16>(constant uint4 &, constant uint2 &,
    device const float *, device const float *, device const float *, device const uint *,
    constant uint &, device atomic_uint *, constant uint &, device float *, constant uint4 &,
    uint2, uint, uint, uint);

// Partial-row sum of one output column, eight (then four) splits of loads in
// flight per step; summation in split order.
inline float v41_flash_combined(device const float *scratch, threadgroup const float *weight,
    uint used, uint heads, uint dim, uint h, uint column) {
    float value = 0;
    device const float *p = scratch + ulong(h) * dim + column;
    const ulong stride = ulong(heads) * dim;
    uint i = 0;
    for (; i + 8 <= used; i += 8) {
        float x[8];
        for (ushort u = 0; u < 8; u++) x[u] = p[(i + u) * stride];
        for (ushort u = 0; u < 8; u++) value = fma(x[u], weight[i + u], value);
    }
    for (; i + 4 <= used; i += 4) {
        float x[4];
        for (ushort u = 0; u < 4; u++) x[u] = p[(i + u) * stride];
        for (ushort u = 0; u < 4; u++) value = fma(x[u], weight[i + u], value);
    }
    for (; i < used; i++) value = fma(p[i * stride], weight[i], value);
    return value;
}

template <bool ROPE>
[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_deepseek_v41_attention_flash_combine(
    constant uint4 &a [[buffer(0)]], constant uint2 &b [[buffer(1)]],
    device const float *sink [[buffer(6)]], device float *out [[buffer(8)]],
    device atomic_uint *status [[buffer(10)]], constant uint &heads [[buffer(14)]],
    device const float *scratch [[buffer(15)]], constant uint4 &f [[buffer(16)]],
    device const float *freq [[buffer(17)]], constant uint2 &rope [[buffer(18)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float weight[V41_FLASH_MAX_SPLITS];
    threadgroup bool rejected;
    const uint dim = a.x, h = group.y, splits = f.y;
    const uint used = (a.y + b.x + f.x - 1) / f.x;
    device const float *partial_m = scratch + v41_flash_scratch_rows(splits, heads, dim);
    device const float *partial_l = partial_m + splits * heads;
    device const float *partial_bad = partial_l + splits * heads;
    if (sg == 0) {
        const float s = sink[h];
        float maximum = s;
        bool bad = false;
        for (uint i = lane; i < used; i += 32) {
            maximum = max(maximum, partial_m[i * heads + h]);
            bad |= partial_bad[i] != 0.0f;
        }
        maximum = simd_max(maximum);
        bad = simd_any(bad);
        float denom = 0;
        for (uint i = lane; i < used; i += 32)
            denom += partial_l[i * heads + h] * exp(partial_m[i * heads + h] - maximum);
        denom = simd_sum(denom) + exp(s - maximum);
        for (uint i = lane; i < used; i += 32)
            weight[i] = exp(partial_m[i * heads + h] - maximum) / denom;
        if (lane == 0) rejected = bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint j = group.x * V41_FLASH_THREADS + tid;
    if (j >= dim) return;
    const float value = rejected ? 0.0f : v41_flash_combined(scratch, weight, used, heads, dim, h, j);
    if (!isfinite(value)) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
    if (ROPE) {
        const uint rotary_start = dim - 2 * rope.x;
        if (j >= rotary_start) {
            const bool even = (j - rotary_start) % 2 == 0;
            const float partner = rejected ? 0.0f
                : v41_flash_combined(scratch, weight, used, heads, dim, h, even ? j + 1 : j - 1);
            const float u = even ? value : partner, v = even ? partner : value;
            const float angle = float(rope.y) * freq[(j - rotary_start) / 2] * -1.0f;
            const float c = cos(angle), s = sin(angle);
            out[h * dim + j] = even ? u * c - v * s : u * s + v * c;
            return;
        }
    }
    out[h * dim + j] = value;
}

#define V41_FLASH_COMBINE(ROPE, SUFFIX) \
    template [[host_name("kernel_deepseek_v41_attention_flash_combine" SUFFIX)]] kernel void \
    kernel_deepseek_v41_attention_flash_combine<ROPE>(constant uint4 &, constant uint2 &, \
        device const float *, device float *, device atomic_uint *, constant uint &, \
        device const float *, constant uint4 &, device const float *, constant uint2 &, \
        uint2, uint, uint, uint);
V41_FLASH_COMBINE(false, "")
V41_FLASH_COMBINE(true, "_rope")

kernel void kernel_deepseek_v41_attention_rope_rows(
    constant uint4 &a [[buffer(0)]], device const float *input [[buffer(1)]],
    device const float *freq [[buffer(2)]], constant uint2 &b [[buffer(3)]],
    device float *output [[buffer(4)]], uint t [[thread_position_in_grid]]) {
    // a: dim, heads, pairs, absolute start; b: queries, inverse/grouped input.
    uint j = t % a.x, h = (t / a.x) % a.y, q = t / (a.x * a.y);
    uint source = b.y ? (h * b.x + q) * a.x + j : t;
    uint rotary_start = a.x - 2 * a.z;
    if (j < rotary_start) { output[t] = input[source]; return; }
    if ((j - rotary_start) % 2) return;
    float angle = float(a.w + q) * freq[(j - rotary_start) / 2] * (b.y ? -1.0f : 1.0f);
    attention_rotate(input + source, output + t, angle);
}

kernel void kernel_deepseek_v41_attention_history(
    constant uint4 &a [[buffer(0)]], device const float *ring [[buffer(1)]],
    device const float *rows [[buffer(2)]], device float *history [[buffer(3)]],
    constant uint2 &prefix [[buffer(4)]],
    uint t [[thread_position_in_grid]]) {
    // a: dim, window, absolute chunk start, chunk rows; prefix: count, floor.
    uint old = prefix.x, row = t / a.x, j = t % a.x;
    history[t] = row < old ? ring[((prefix.y + row) % a.y) * a.x + j]
        : rows[(row - old) * a.x + j];
}

kernel void kernel_deepseek_v41_attention_publish_raw(
    constant uint4 &a [[buffer(0)]], device float *ring [[buffer(1)]],
    device const float *rows [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    uint row = a.w - min(a.w, a.y) + t / a.x, j = t % a.x;
    ring[((a.z + row) % a.y) * a.x + j] = rows[row * a.x + j];
}


// Simdgroup-MMA sink tile: QK^T and PV for one query row and a block of 16
// heads. Keys are the row's raw ring prefix followed by its selected
// compressed rows (K = V = the same 512-wide latent), gathered sixteen at a
// time into threadgroup memory as half; Q is staged once per threadgroup as
// half. Products accumulate in F32; softmax (shared kernel) and the final
// normalization stay F32. QK^T writes the row's score slab exactly where
// attention_score does (scaled by rsqrt(dim); invalid selected ids are
// zero keys here and are rejected by softmax as before); PV accumulates the
// unnormalized exp-weighted sum and attention_normalize_rows divides by the
// softmax denominator while checking finiteness.
#define V41_MMA_HEADS 16
#define V41_MMA_KEYS 16
#define V41_MMA_DIM 512


kernel void kernel_deepseek_v41_attention_score_mma(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    device float *scores [[buffer(7)]], constant uint4 &m [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    // group.x: head block, group.y: query row. 128 threads = 4 simdgroups, one
    // 8-head x 8-key output tile each per 16-key block.
    threadgroup half qblk[V41_MMA_HEADS * V41_MMA_DIM];
    threadgroup half kblk[V41_MMA_KEYS * V41_MMA_DIM];
    const uint r = group.y;
    const uint4 rr = attention_row_raw(a, b, r);
    const uint length = attention_row_length(b, r);
    const uint selected = min(length, a.z);
    const uint count = rr.y + selected;
    device const uint *row_ids = ids + r * a.z;
    device const float *qrow = q + r * m.y * a.x + group.x * V41_MMA_HEADS * a.x;
    device float *srow = scores + r * m.y * a.w + group.x * V41_MMA_HEADS * a.w;
    const float scale = rsqrt(float(a.x));
    for (uint e = tid; e < V41_MMA_HEADS * V41_MMA_DIM / 4; e += 128) {
        ((threadgroup half4 *)qblk)[e] = half4(((device const float4 *)qrow)[e]);
    }
    const uint hb = sg >> 1, kb = sg & 1;
    for (uint k0 = 0; k0 < count; k0 += V41_MMA_KEYS) {
        for (uint e = tid; e < V41_MMA_KEYS * V41_MMA_DIM / 4; e += 128) {
            const uint key = e / (V41_MMA_DIM / 4), c4 = e % (V41_MMA_DIM / 4);
            const uint i = k0 + key;
            float4 v = float4(0.0f);
            if (i < count && (i < rr.y || row_ids[i - rr.y] < length)) {
                v = ((device const float4 *)attention_row_key(rr, raw, compressed, row_ids, i))[c4];
            }
            ((threadgroup half4 *)kblk)[e] = half4(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
        threadgroup const half *qa = qblk + hb * 8 * V41_MMA_DIM;
        threadgroup const half *kt = kblk + kb * 8 * V41_MMA_DIM;
        for (uint d = 0; d < V41_MMA_DIM; d += 8) {
            simdgroup_half8x8 A, B;
            simdgroup_load(A, qa + d, V41_MMA_DIM);                              // [8 heads x 8 dims]
            simdgroup_load(B, kt + d, V41_MMA_DIM, ulong2(0, 0), true);         // K^T [8 dims x 8 keys]
            simdgroup_multiply_accumulate(acc, A, B, acc);
        }
        acc.thread_elements()[0] *= scale;
        acc.thread_elements()[1] *= scale;
        if (k0 + V41_MMA_KEYS <= count) {
            simdgroup_store(acc, srow + hb * 8 * a.w + k0 + kb * 8, a.w);
        } else {
            // Partial last block (uniform across the threadgroup): once every
            // simdgroup is done with the key block, reuse it as F32 scratch and
            // scatter only the valid keys.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float *tail = (threadgroup float *)kblk;
            simdgroup_store(acc, tail + sg * 64, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid & 31; e < 64; e += 32) {
                const uint h = hb * 8 + e / 8, kk = kb * 8 + e % 8;
                if (k0 + kk < count) srow[h * a.w + k0 + kk] = tail[sg * 64 + e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_deepseek_v41_attention_pv_mma(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *raw [[buffer(3)]], device const float *compressed [[buffer(4)]],
    device const uint *ids [[buffer(5)]], device const float *scores [[buffer(7)]],
    device float *out [[buffer(8)]], constant uint4 &m [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    // group.x: head block, group.y: query row. 256 threads = 8 simdgroups;
    // each owns 2 head tiles x 8 dim tiles of the [16 x 512] output.
    threadgroup half vblk[V41_MMA_KEYS * V41_MMA_DIM];
    threadgroup half pblk[V41_MMA_HEADS * V41_MMA_KEYS];
    const uint r = group.y;
    const uint4 rr = attention_row_raw(a, b, r);
    const uint length = attention_row_length(b, r);
    const uint selected = min(length, a.z);
    const uint count = rr.y + selected;
    device const uint *row_ids = ids + r * a.z;
    device const float *srow = scores + r * m.y * a.w + group.x * V41_MMA_HEADS * a.w;
    const bool invalid = srow[a.w - 1] == -1.0f;
    simdgroup_float8x8 acc[16];
    for (uint i = 0; i < 16; ++i) acc[i] = simdgroup_float8x8(0.0f);
    const uint dim0 = sg * 64;
    if (!invalid) {
        for (uint k0 = 0; k0 < count; k0 += V41_MMA_KEYS) {
            for (uint e = tid; e < V41_MMA_KEYS * V41_MMA_DIM / 4; e += 256) {
                const uint key = e / (V41_MMA_DIM / 4), c4 = e % (V41_MMA_DIM / 4);
                const uint i = k0 + key;
                float4 v = float4(0.0f);
                if (i < count) v = ((device const float4 *)attention_row_key(rr, raw, compressed, row_ids, i))[c4];
                ((threadgroup half4 *)vblk)[e] = half4(v);
            }
            // P block [16 heads x 16 keys], zero past count (the slab tail is
            // unspecified there).
            {
                const uint h = tid / V41_MMA_KEYS, kk = tid % V41_MMA_KEYS;
                pblk[tid] = half(k0 + kk < count ? srow[h * a.w + k0 + kk] : 0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0; ks < V41_MMA_KEYS; ks += 8) {
                simdgroup_half8x8 P0, P1;
                simdgroup_load(P0, pblk + ks, V41_MMA_KEYS);
                simdgroup_load(P1, pblk + 8 * V41_MMA_KEYS + ks, V41_MMA_KEYS);
                for (uint t = 0; t < 8; ++t) {
                    simdgroup_half8x8 V;
                    simdgroup_load(V, vblk + ks * V41_MMA_DIM + dim0 + t * 8, V41_MMA_DIM); // [8 keys x 8 dims]
                    simdgroup_multiply_accumulate(acc[t], P0, V, acc[t]);
                    simdgroup_multiply_accumulate(acc[8 + t], P1, V, acc[8 + t]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    device float *orow = out + r * a.x + (group.x * V41_MMA_HEADS) * m.z * a.x;
    for (uint t = 0; t < 8; ++t) {
        simdgroup_store(acc[t], orow + dim0 + t * 8, m.z * a.x);
        simdgroup_store(acc[8 + t], orow + 8 * m.z * a.x + dim0 + t * 8, m.z * a.x);
    }
}

// out[h][r][j] /= denominator(h, r); zero and flag invalid rows; finite check.
kernel void kernel_deepseek_v41_attention_normalize_rows(
    constant uint4 &a [[buffer(0)]], device const float *scores [[buffer(7)]],
    device float *out [[buffer(8)]], device atomic_uint *status [[buffer(10)]],
    constant uint4 &m [[buffer(11)]], uint t [[thread_position_in_grid]]) {
    // m: padded, heads, n; a.x = dim, a.w = score stride.
    const uint total = m.y * m.z * a.x;
    if (t >= total) return;
    const uint h = t / (m.z * a.x), rem = t % (m.z * a.x), r = rem / a.x;
    const float denom = scores[(r * m.y + h) * a.w + a.w - 1];
    float value = out[t];
    value = denom == -1.0f ? 0.0f : value / denom;
    out[t] = value;
    if (!isfinite(value)) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
}

kernel void kernel_deepseek_v41_attention_score_rows(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    device float *scores [[buffer(7)]], constant uint4 &m [[buffer(11)]],
    threadgroup float *q_shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint r = group.z, length = attention_row_length(b, r);
    uint4 ar = attention_row_raw(a, b, r);
    uint2 br = uint2(min(length, a.z), a.w);
    uint count = ar.y + br.x, h = group.y;
    device const float *q_row = q + r * m.y * a.x + h * a.x;
    device const uint *row_ids = ids + r * a.z;
    device float *score_row = scores + r * m.y * a.w;
    for (uint j = tid; j < a.x; j += 256) q_shared[j] = q_row[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Four keys per simdgroup: 4x fewer threadgroups, same per-key dot order.
    for (uint k = 0; k < 4; ++k) {
        uint i = group.x * 32 + k * 8 + sg;
        if (i >= count) continue;
        if (i >= ar.y && row_ids[i - ar.y] >= length) continue;
        device const float *row = i < ar.y
            ? raw + ((ar.z + i) % ar.w) * ar.x
            : compressed + row_ids[i - ar.y] * ar.x;
        float dot = 0;
        for (uint j = lane; j < ar.x; j += 32) dot += q_shared[j] * row[j];
        dot = simd_sum(dot);
        if (lane == 0) score_row[h * br.y + i] = dot * rsqrt(float(ar.x));
    }
}

// Four keys in parallel per simdgroup: each 8-lane group owns one key, so a
// 512-wide dot needs 64 FMAs and a 3-step group reduction instead of four
// 32-lane dots with 5-step reductions. Per-key K order changes only in the
// final lane tree.
kernel void kernel_deepseek_v41_attention_score_rows_lane8(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    device float *scores [[buffer(7)]], constant uint4 &m [[buffer(11)]],
    threadgroup float *q_shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint r = group.z, length = attention_row_length(b, r);
    uint4 ar = attention_row_raw(a, b, r);
    uint2 br = uint2(min(length, a.z), a.w);
    uint count = ar.y + br.x, h = group.y;
    device const float *q_row = q + r * m.y * a.x + h * a.x;
    device const uint *row_ids = ids + r * a.z;
    device float *score_row = scores + r * m.y * a.w;
    for (uint j = tid; j < a.x; j += 256) q_shared[j] = q_row[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint lane8 = lane & 7u;
    uint i = group.x * 32 + sg * 4 + (lane >> 3);
    if (i >= count) return;
    if (i >= ar.y && row_ids[i - ar.y] >= length) return;
    device const float *row = i < ar.y
        ? raw + ((ar.z + i) % ar.w) * ar.x
        : compressed + row_ids[i - ar.y] * ar.x;
    float dot = 0;
    for (uint j = lane8; j < ar.x; j += 8) dot += q_shared[j] * row[j];
    dot += simd_shuffle_down(dot, 4);
    dot += simd_shuffle_down(dot, 2);
    dot += simd_shuffle_down(dot, 1);
    if (lane8 == 0) score_row[h * br.y + i] = dot * rsqrt(float(ar.x));
}

// One dispatch replaces key RMS norm + key RoPE + latent RoPE + both cache
// copies. The key path uses 128 threads for the index-width reduction; the
// latent path copies its row and rotates the final 2*pairs values. The latent
// norm and key projection stay separate because the projection consumes the
// normalized latent.
kernel void kernel_deepseek_v41_attention_compress_publish(
    constant uint4 &a [[buffer(0)]], constant float &eps [[buffer(1)]],
    device const float *key [[buffer(2)]], device const float *key_norm [[buffer(3)]],
    device const float *latent [[buffer(4)]], device const float *freq [[buffer(5)]],
    device float *out_key [[buffer(6)]], device float *out_latent [[buffer(7)]],
    threadgroup float *shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint kd = a.x, ld = a.y, pairs = a.z, position = a.w;
    threadgroup float *normed = shmem;
    threadgroup float *scratch = shmem + 128;
    bool key_path = tid < 128;
    float v = 0.0f;
    if (key_path) {
        if (tid < kd) v = key[tid];
        float sumsq = simd_sum(v * v);
        if (lane == 0) scratch[sg] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = scratch[0] + scratch[1] + scratch[2] + scratch[3];
        scratch[4] = rsqrt(total / float(kd) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (key_path) {
        if (tid < kd) normed[tid] = (v * scratch[4]) * key_norm[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (key_path) {
        uint rope_start = kd - 2 * pairs;
        if (tid < kd && tid < rope_start) out_key[tid] = normed[tid];
        if (tid < pairs) {
            uint off = rope_start + 2 * tid;
            float angle = float(position) * freq[tid];
            float c = cos(angle), s = sin(angle);
            float u = normed[off], w = normed[off + 1];
            out_key[off] = u * c - w * s;
            out_key[off + 1] = u * s + w * c;
        }
    } else {
        uint lt = tid - 128;
        uint rope_start = ld - 2 * pairs;
        for (uint i = lt; i < rope_start; i += 128) out_latent[i] = latent[i];
        if (lt < pairs) {
            uint off = rope_start + 2 * lt;
            float angle = float(position) * freq[lt];
            float c = cos(angle), s = sin(angle);
            float u = latent[off], w = latent[off + 1];
            out_latent[off] = u * c - w * s;
            out_latent[off + 1] = u * s + w * c;
        }
    }
}

// Batched owner publication: one threadgroup per completed ratio-group.
// Pools the precomputed compressor value/gate rows, applies the compressor
// RMS norm, projects and norms the index key, then ropes and stores both
// cache rows. This replaces the per-row copy/pool/norm/GEMV/publish chain;
// the arithmetic order matches those kernels where it can, but the
// reductions are tree-shaped rather than the per-row kernels' own order.
struct compress_rows_args {
    uint ratio;
    uint group0;
    uint row0;
    uint pairs;
    float eps;
};

static inline float compress_rows_bf16(ushort v) {
    return as_type<float>((uint)v << 16);
}

kernel void kernel_deepseek_v41_attention_compress_rows(
    constant compress_rows_args &a [[buffer(0)]],
    device const float *value [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device const float *comp_norm [[buffer(3)]],
    device const ushort *wk [[buffer(4)]],
    device const float *key_norm [[buffer(5)]],
    device const float *freq [[buffer(6)]],
    device float *out_latent [[buffer(7)]],
    device float *out_key [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float lat[512];
    threadgroup float key[128];
    threadgroup float scratch[8];
    uint base = a.row0 + group * a.ratio;
    for (uint d = tid; d < 512; d += 256) {
        float v = value[base * 512 + d];
        if (a.ratio == 2) {
            float v1 = value[(base + 1) * 512 + d];
            float s0 = gate[base * 512 + d];
            float s1 = gate[(base + 1) * 512 + d];
            float m = max(s0, s1);
            float e0 = exp(s0 - m), e1 = exp(s1 - m), den = e0 + e1;
            v = v * (e0 / den) + v1 * (e1 / den);
        }
        lat[d] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f;
    for (uint d = tid; d < 512; d += 256) ss += lat[d] * lat[d];
    ss = simd_sum(ss);
    if (lane == 0) scratch[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) total += scratch[i];
        scratch[0] = rsqrt(total / 512.0f + a.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float latent_scale = scratch[0];
    for (uint d = tid; d < 512; d += 256) lat[d] = (lat[d] * latent_scale) * comp_norm[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // One simdgroup per output row; each lane reads its stride-32 float4
    // slice of the BF16 key projection, matching the mat-vec load shape.
    for (uint out = sg; out < 128; out += 8) {
        device const ushort *wrow = wk + (ulong)out * 512;
        float sum = 0.0f;
        for (uint v = lane; v < 128; v += 32) {
            uint d = v * 4;
            float4 x = float4(
                compress_rows_bf16(wrow[d]),
                compress_rows_bf16(wrow[d + 1]),
                compress_rows_bf16(wrow[d + 2]),
                compress_rows_bf16(wrow[d + 3]));
            sum += x.x * lat[d] + x.y * lat[d + 1] + x.z * lat[d + 2] + x.w * lat[d + 3];
        }
        sum = simd_sum(sum);
        if (lane == 0) key[out] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128) {
        float v = key[tid];
        float sumsq = simd_sum(v * v);
        if (lane == 0) scratch[sg] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = scratch[0] + scratch[1] + scratch[2] + scratch[3];
        scratch[0] = rsqrt(total / 128.0f + a.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128) key[tid] = (key[tid] * scratch[0]) * key_norm[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint position = (a.group0 + group) * a.ratio;
    device float *olat = out_latent + (ulong)(a.group0 + group) * 512;
    device float *okey = out_key + (ulong)(a.group0 + group) * 128;
    for (uint d = tid; d < 512; d += 256) olat[d] = lat[d];
    if (tid < 128) okey[tid] = key[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint lrope = 512 - 2 * a.pairs;
    if (tid < a.pairs) {
        uint off = lrope + 2 * tid;
        float angle = float(position) * freq[tid];
        float c = cos(angle), s = sin(angle);
        float u = lat[off], w = lat[off + 1];
        olat[off] = u * c - w * s;
        olat[off + 1] = u * s + w * c;
    }
    uint krope = 128 - 2 * a.pairs;
    if (tid < a.pairs) {
        uint off = krope + 2 * tid;
        float angle = float(position) * freq[tid];
        float c = cos(angle), s = sin(angle);
        float u = key[off], w = key[off + 1];
        okey[off] = u * c - w * s;
        okey[off + 1] = u * s + w * c;
    }
}

kernel void kernel_deepseek_v41_attention_softmax_rows(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const uint *ids [[buffer(5)]], device const float *sink [[buffer(6)]],
    device float *scores [[buffer(7)]], device atomic_uint *status [[buffer(10)]],
    constant uint4 &m [[buffer(11)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint r = group.y, length = attention_row_length(b, r);
    attention_softmax(attention_row_raw(a, b, r), uint2(min(length, a.z), a.w),
        ids + r * a.z, sink, scores + r * m.y * a.w, length, status, group.x, lane);
}

kernel void kernel_deepseek_v41_attention_matrix_values_rows(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *raw [[buffer(3)]], device const float *compressed [[buffer(4)]],
    device const uint *ids [[buffer(5)]], device const float *scores [[buffer(7)]],
    constant uint4 &m [[buffer(11)]], device float *values [[buffer(12)]],
    device float *probabilities [[buffer(13)]], uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[16][17];
    uint r = group.z, length = attention_row_length(b, r);
    attention_matrix_values(attention_row_raw(a, b, r), uint2(min(length, a.z), a.w),
        raw, compressed, ids + r * a.z, scores + r * m.y * a.w, length, m.xy,
        values + r * a.x * m.x, probabilities + r * m.x, m.x * m.z, tile, group.xy, tid);
}

kernel void kernel_deepseek_v41_attention_output_rows(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *raw [[buffer(3)]], device const float *compressed [[buffer(4)]],
    device const uint *ids [[buffer(5)]], device const float *scores [[buffer(7)]],
    device float *out [[buffer(8)]], device atomic_uint *status [[buffer(10)]],
    constant uint4 &m [[buffer(11)]], uint t [[thread_position_in_grid]]) {
    uint r = t / (m.y * a.x), local = t % (m.y * a.x), length = attention_row_length(b, r);
    attention_output(attention_row_raw(a, b, r), uint2(min(length, a.z), a.w),
        raw, compressed, ids + r * a.z, scores + r * m.y * a.w,
        out + r * a.x, m.z * a.x, status, local);
}

// Decode rank norms and KV epilogue in one dispatch: two 1024-thread
// threadgroups. Threadgroup 0 RMS-normalizes the query rank in place;
// threadgroup 1 RMS-normalizes the KV latent in place, applies RoPE to its
// rotary tail (one head) and stores the result into this position's raw ring
// row. The norm repeats kernel_rms_norm_mul_f32 at 1024 threads (same
// per-thread strided sums, simdgroup and first-simdgroup reductions, and
// (x * scale) * weight); the rotation repeats kernel_deepseek_v41_attention_rope.
// Bitwise equal to the four separate dispatches.
struct RankNormsArgs { uint q_n; uint kv_n; uint pairs; uint position; float eps; };
inline float v41_rms_scale(device const float *x, uint n, float eps, uint tid,
    uint sg, uint lane, threadgroup float *shmem) {
    float sumsq = 0.0f;
    for (uint i = tid; i < n; i += 1024) {
        const float v = x[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    if (lane == 0) shmem[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = lane < 32 ? shmem[lane] : 0.0f;
    sumsq = simd_sum(sumsq);
    return rsqrt(sumsq / float(n) + eps);
}
[[max_total_threads_per_threadgroup(1024)]]
kernel void kernel_deepseek_v41_attention_rank_norms(
    constant RankNormsArgs &a [[buffer(0)]],
    device float *q [[buffer(1)]], device const float *q_weight [[buffer(2)]],
    device float *kv [[buffer(3)]], device const float *kv_weight [[buffer(4)]],
    device const float *freq [[buffer(5)]], device float *ring_row [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float shmem[32];
    if (group == 0) {
        const float scale = v41_rms_scale(q, a.q_n, a.eps, tid, sg, lane, shmem);
        for (uint i = tid; i < a.q_n; i += 1024) q[i] = (q[i] * scale) * q_weight[i];
        return;
    }
    const float scale = v41_rms_scale(kv, a.kv_n, a.eps, tid, sg, lane, shmem);
    // Every value is read before any is written (the reduction's barrier
    // separates the reads above; the rotation reads its pair below).
    const uint rotary_start = a.kv_n - 2 * a.pairs;
    for (uint i = tid; i < a.kv_n; i += 1024) {
        if (i < rotary_start) {
            const float v = (kv[i] * scale) * kv_weight[i];
            kv[i] = v;
            ring_row[i] = v;
        } else if ((i - rotary_start) % 2 == 0) {
            const uint j = (i - rotary_start) / 2;
            const float u = (kv[i] * scale) * kv_weight[i];
            const float w = (kv[i + 1] * scale) * kv_weight[i + 1];
            const float angle = float(a.position) * freq[j];
            const float c = cos(angle), s = sin(angle);
            const float r0 = u * c - w * s;
            const float r1 = u * s + w * c;
            kv[i] = r0;
            kv[i + 1] = r1;
            ring_row[i] = r0;
            ring_row[i + 1] = r1;
        }
    }
}
