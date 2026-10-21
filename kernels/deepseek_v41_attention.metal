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

inline void attention_score_shared(uint4 a, uint2 b, threadgroup const float *q,
    device const float *raw, device const float *compressed, device const uint *ids,
    device float *scores, uint length, uint2 group, uint sg, uint lane) {
    uint count = a.y + b.x, h = group.y, i = group.x * 8 + sg;
    if (i >= count) return;
    if (i >= a.y && ids[i - a.y] >= length) return;
    device const float *row = i < a.y ? raw + ((a.z + i) % a.w) * a.x : compressed + ids[i - a.y] * a.x;
    float dot = 0;
    for (uint j = lane; j < a.x; j += 32) dot += q[j] * row[j];
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

// Multiple query heads per threadgroup: one K row serves several heads,
// cutting compressed/history K read traffic. Dot order per key is unchanged.
kernel void kernel_deepseek_v41_attention_score_rows_heads(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const float *q [[buffer(2)]], device const float *raw [[buffer(3)]],
    device const float *compressed [[buffer(4)]], device const uint *ids [[buffer(5)]],
    device float *scores [[buffer(7)]], constant uint4 &m [[buffer(11)]],
    constant uint &heads_per_tg [[buffer(14)]],
    threadgroup float *q_shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint r = group.z, length = attention_row_length(b, r);
    uint4 ar = attention_row_raw(a, b, r);
    uint2 br = uint2(min(length, a.z), a.w);
    uint count = ar.y + br.x;
    uint h0 = group.y * heads_per_tg;
    device const float *q_rows = q + r * m.y * a.x + h0 * a.x;
    device const uint *row_ids = ids + r * a.z;
    device float *score_row = scores + r * m.y * a.w;
    for (uint j = tid; j < heads_per_tg * a.x; j += 256) q_shared[j] = q_rows[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 4; ++k) {
        uint i = group.x * 32 + k * 8 + sg;
        if (i >= count) continue;
        if (i >= ar.y && row_ids[i - ar.y] >= length) continue;
        device const float *row = i < ar.y
            ? raw + ((ar.z + i) % ar.w) * ar.x
            : compressed + row_ids[i - ar.y] * ar.x;
        for (uint hi = 0; hi < heads_per_tg; ++hi) {
            float dot = 0;
            for (uint j = lane; j < ar.x; j += 32) dot += q_shared[hi * ar.x + j] * row[j];
            dot = simd_sum(dot);
            if (lane == 0) score_row[(h0 + hi) * br.y + i] = dot * rsqrt(float(ar.x));
        }
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
