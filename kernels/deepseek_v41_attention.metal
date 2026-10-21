#include <metal_stdlib>
using namespace metal;

kernel void kernel_deepseek_v41_attention_selection_prepare(
    constant uint4 &a [[buffer(0)]], device uint *ids [[buffer(1)]],
    device uint *mask [[buffer(2)]],
    uint t [[thread_position_in_grid]]) {
    if (t < a.w) ids[t] = t;
    if (a.z && t < a.y) mask[t] = 1;
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

kernel void kernel_deepseek_v41_attention_keep(
    constant uint2 &a [[buffer(0)]], device const uint *ids [[buffer(1)]],
    device uint *mask [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t >= a.x) return;
    uint keep = 0;
    for (uint j = 0; j < a.y; ++j) keep |= ids[j] == t;
    mask[t] = keep;
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
    uint t [[thread_position_in_grid]]) {
    // a: dim, window, absolute chunk start, chunk rows.
    uint old = min(a.z, a.y), row = t / a.x, j = t % a.x;
    history[t] = row < old ? ring[((a.z - old + row) % a.y) * a.x + j]
        : rows[(row - old) * a.x + j];
}

kernel void kernel_deepseek_v41_attention_publish_raw(
    constant uint4 &a [[buffer(0)]], device float *ring [[buffer(1)]],
    device const float *rows [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    uint row = a.w - min(a.w, a.y) + t / a.x, j = t % a.x;
    ring[((a.z + row) % a.y) * a.x + j] = rows[row * a.x + j];
}

inline uint4 attention_row_raw(uint4 geometry, uint4 history, uint query) {
    uint end = history.x + query + 1, count = min(end, geometry.y);
    return uint4(geometry.x, count, end - count - history.y, history.z);
}
inline uint attention_row_length(uint4 history, uint query) {
    return history.w ? (history.x + query + 1) / history.w : 0;
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

inline device const float *attention_row_key(uint4 raw, device const float *history,
    device const float *compressed, device const uint *ids, uint i) {
    return i < raw.y ? history + ((raw.z + i) % raw.w) * raw.x
        : compressed + ids[i - raw.y] * raw.x;
}

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
    uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = group.z, length = attention_row_length(b, r);
    attention_score(attention_row_raw(a, b, r), uint2(min(length, a.z), a.w),
        q + r * m.y * a.x, raw, compressed, ids + r * a.z,
        scores + r * m.y * a.w, length, group.xy, sg, lane);
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
