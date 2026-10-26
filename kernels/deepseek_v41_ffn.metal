#include <metal_stdlib>
using namespace metal;

kernel void kernel_deepseek_v41_group_rows(
    constant uint3 &shape [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    uint width = shape.x, groups = shape.y, rows = shape.z;
    if (i >= width * groups * rows) return;
    uint row = i / (width * groups), group = (i / width) % groups;
    output[(group * rows + row) * width + i % width] = input[i];
}

// Projection must precede RMS scaling; collapse uses a different, lagged pre.
kernel void kernel_deepseek_v41_projected_rms(
    constant uint &n [[buffer(0)]], constant float &eps [[buffer(1)]],
    device const float *x [[buffer(2)]], device float *mix [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    x += row * n;
    mix += row * 24;
    threadgroup float sums[256];
    float sum = 0.0f;
    for (uint i = tid; i < n; i += 256) sum += x[i] * x[i];
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 24) mix[tid] *= rsqrt(sums[0] / float(n) + eps);
}

// mHC pre-block for one decode row, as two dispatches instead of five.
//
// Projection: the 24 mixing logits are a 4*hidden -> 24 F32 GEMV. One
// simdgroup per output row (the generic F32 mat-vec) leaves 24 simdgroups
// on the GPU, each walking a 160-deep load chain; here one 1024-thread
// threadgroup owns each output and strides K, so the 2 MB of weights are in
// flight at once.
constant uint V41_HC_PROJECTION_THREADS = 1024;
kernel void kernel_deepseek_v41_hc_projection(
    constant uint &n_in [[buffer(0)]], constant uint &n_out [[buffer(1)]],
    device const float *weight [[buffer(2)]], device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[V41_HC_PROJECTION_THREADS / 32];
    if (row >= n_out) return;
    device const float4 *w4 = (device const float4 *)(weight + ulong(row) * n_in);
    device const float4 *x4 = (device const float4 *)x;
    const uint n4 = n_in / 4;
    float sum = 0.0f;
    // Two weight/activation pairs in flight per thread (the loop is short
    // and latency-bound: 4*hidden/4 float4s over 1024 threads).
    uint i = tid;
    for (; i + V41_HC_PROJECTION_THREADS < n4; i += 2 * V41_HC_PROJECTION_THREADS) {
        const float4 a0 = w4[i], b0 = x4[i];
        const float4 a1 = w4[i + V41_HC_PROJECTION_THREADS], b1 = x4[i + V41_HC_PROJECTION_THREADS];
        sum += a0.x * b0.x + a0.y * b0.y + a0.z * b0.z + a0.w * b0.w;
        sum += a1.x * b1.x + a1.y * b1.y + a1.z * b1.z + a1.w * b1.w;
    }
    for (; i < n4; i += V41_HC_PROJECTION_THREADS) {
        const float4 a = w4[i];
        const float4 b = x4[i];
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (uint i = n4 * 4 + tid; i < n_in; i += V41_HC_PROJECTION_THREADS) sum += weight[ulong(row) * n_in + i] * x[i];
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < V41_HC_PROJECTION_THREADS / 32; ++s) total += partial[s];
        y[row] = total;
    }
}

// Fused: projected RMS of the residual scales the 24 mixes; sigmoid pre/post
// and the Sinkhorn-normalized 4x4 combination (same arithmetic as
// kernel_deepseek_v4_hc_controls, in registers instead of device
// read-modify-write); collapse of the four streams with the *incoming*
// (lagged) pre; weighted RMS norm of the collapsed row. One 1024-thread
// threadgroup; reductions change F32 summation order relative to the
// separate kernels, nothing else.
struct HcPreArgs { uint hidden; float norm_eps; float hc_eps; };
constant uint V41_HC_PRE_THREADS = 1024;
constant uint V41_HC_CONNECTIONS = 4;
constant uint V41_HC_SINKHORN_ITERATIONS = 20;
inline float v41_hc_tg_sum(float v, threadgroup float *partial, uint lane, uint sg) {
    v = simd_sum(v);
    if (lane == 0) partial[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < V41_HC_PRE_THREADS / 32; ++s) total += partial[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}
kernel void kernel_deepseek_v41_hc_pre_scalar(
    constant HcPreArgs &a [[buffer(0)]],
    device const float *x [[buffer(1)]],            // residual [4*hidden]
    device float *mix [[buffer(2)]],                // [24] raw in, scaled out
    device const float *scale [[buffer(3)]],        // [3]
    device const float *base [[buffer(4)]],         // [24]
    device float *pre [[buffer(5)]],                // [4]
    device float *post [[buffer(6)]],               // [4]
    device float *combination [[buffer(7)]],        // [16]
    device const float *incoming_pre [[buffer(8)]], // [4] lagged
    device float *collapsed [[buffer(9)]],          // [hidden]
    device const float *norm_weight [[buffer(10)]], // [hidden]
    device float *normalized [[buffer(11)]],        // [hidden]
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[V41_HC_PRE_THREADS / 32];
    const uint n = V41_HC_CONNECTIONS * a.hidden;
    // A. projected RMS over the whole residual scales the mixes.
    float ss = 0.0f;
    for (uint i = tid; i < n; i += V41_HC_PRE_THREADS) ss += x[i] * x[i];
    const float inv = rsqrt(v41_hc_tg_sum(ss, partial, lane, sg) / float(n) + a.norm_eps);
    if (tid < 24) mix[tid] *= inv;
    threadgroup_barrier(mem_flags::mem_device);
    // B. controls (one thread; 24 inputs, 24 outputs).
    if (tid == 0) {
        const float eps = a.hc_eps;
        for (uint stream = 0; stream < V41_HC_CONNECTIONS; ++stream) {
            pre[stream] = 1.0f / (1.0f + exp(-(mix[stream] * scale[0] + base[stream]))) + eps;
            post[stream] = 2.0f / (1.0f + exp(-(mix[4 + stream] * scale[1] + base[4 + stream])));
        }
        float comb[16];
        for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) {
            float row_max = -INFINITY;
            for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) {
                const uint index = source * V41_HC_CONNECTIONS + destination;
                const float value = mix[8 + index] * scale[2] + base[8 + index];
                comb[index] = value;
                row_max = max(row_max, value);
            }
            float sum = 0.0f;
            for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) {
                const uint index = source * V41_HC_CONNECTIONS + destination;
                comb[index] = exp(comb[index] - row_max);
                sum += comb[index];
            }
            const float inverse = 1.0f / sum;
            for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) {
                const uint index = source * V41_HC_CONNECTIONS + destination;
                comb[index] = comb[index] * inverse + eps;
            }
        }
        for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) {
            float sum = 0.0f;
            for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) sum += comb[source * V41_HC_CONNECTIONS + destination];
            const float inverse = 1.0f / (sum + eps);
            for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) comb[source * V41_HC_CONNECTIONS + destination] *= inverse;
        }
        for (uint iteration = 1; iteration < V41_HC_SINKHORN_ITERATIONS; ++iteration) {
            for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) {
                float sum = 0.0f;
                for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) sum += comb[source * V41_HC_CONNECTIONS + destination];
                const float inverse = 1.0f / (sum + eps);
                for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) comb[source * V41_HC_CONNECTIONS + destination] *= inverse;
            }
            for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) {
                float sum = 0.0f;
                for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) sum += comb[source * V41_HC_CONNECTIONS + destination];
                const float inverse = 1.0f / (sum + eps);
                for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) comb[source * V41_HC_CONNECTIONS + destination] *= inverse;
            }
        }
        for (uint i = 0; i < 16; ++i) combination[i] = comb[i];
    }
    // C. collapse with the lagged incoming pre (independent of phase B).
    const float p0 = incoming_pre[0], p1 = incoming_pre[1], p2 = incoming_pre[2], p3 = incoming_pre[3];
    float ss2 = 0.0f;
    for (uint d = tid; d < a.hidden; d += V41_HC_PRE_THREADS) {
        float v = 0.0f;
        v += x[d] * p0;
        v += x[a.hidden + d] * p1;
        v += x[2 * a.hidden + d] * p2;
        v += x[3 * a.hidden + d] * p3;
        collapsed[d] = v;
        ss2 += v * v;
    }
    // D. weighted RMS norm of the collapsed row.
    const float inv2 = rsqrt(v41_hc_tg_sum(ss2, partial, lane, sg) / float(a.hidden) + a.norm_eps);
    for (uint d = tid; d < a.hidden; d += V41_HC_PRE_THREADS) {
        normalized[d] = (collapsed[d] * inv2) * norm_weight[d];
    }
}

// Vector form (the host's default when hidden % 4 == 0 and the residual,
// collapsed, normalized and norm-weight views are 16-byte aligned). One
// threadgroup runs on one GPU core, so the scalar loops above were bound by
// memory-level parallelism (one 4-byte load in flight per thread, ~20 us per
// call). Here every thread issues its float4 loads together, the controls run
// on simdgroup 0 with shuffle reductions (same per-entry arithmetic and
// operand order as the one-thread loop), and the collapsed values stay in
// registers for the norm. Reduction order differs from the scalar form.
#define V41_HC_PRE_MAX_D4 4
kernel void kernel_deepseek_v41_hc_pre(
    constant HcPreArgs &a [[buffer(0)]],
    device const float4 *x [[buffer(1)]],           // residual [4*hidden]
    device float *mix [[buffer(2)]],                // [24] raw in, scaled out
    device const float *scale [[buffer(3)]],        // [3]
    device const float *base [[buffer(4)]],         // [24]
    device float *pre [[buffer(5)]],                // [4]
    device float *post [[buffer(6)]],               // [4]
    device float *combination [[buffer(7)]],        // [16]
    device const float *incoming_pre [[buffer(8)]], // [4] lagged
    device float4 *collapsed [[buffer(9)]],         // [hidden]
    device const float4 *norm_weight [[buffer(10)]],// [hidden]
    device float4 *normalized [[buffer(11)]],       // [hidden]
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[V41_HC_PRE_THREADS / 32];
    const uint h4 = a.hidden / 4;
    const uint n4 = V41_HC_CONNECTIONS * h4;
    // A. projected RMS over the whole residual: four loads in flight.
    float4 acc = 0.0f;
    uint i = tid;
    for (; i + 3 * V41_HC_PRE_THREADS < n4; i += 4 * V41_HC_PRE_THREADS) {
        const float4 v0 = x[i];
        const float4 v1 = x[i + V41_HC_PRE_THREADS];
        const float4 v2 = x[i + 2 * V41_HC_PRE_THREADS];
        const float4 v3 = x[i + 3 * V41_HC_PRE_THREADS];
        acc += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    for (; i < n4; i += V41_HC_PRE_THREADS) {
        const float4 v = x[i];
        acc += v * v;
    }
    // C (loads). The collapse uses the lagged pre and is independent of the
    // controls; issue its loads before the first barrier.
    const float p0 = incoming_pre[0], p1 = incoming_pre[1], p2 = incoming_pre[2], p3 = incoming_pre[3];
    float4 col[V41_HC_PRE_MAX_D4];
    float4 ss2v = 0.0f;
    for (uint j = 0; j < V41_HC_PRE_MAX_D4; ++j) {
        const uint d4 = tid + j * V41_HC_PRE_THREADS;
        if (d4 < h4) {
            float4 v = 0.0f;
            v += x[d4] * p0;
            v += x[h4 + d4] * p1;
            v += x[2 * h4 + d4] * p2;
            v += x[3 * h4 + d4] * p3;
            col[j] = v;
            ss2v += v * v;
        }
    }
    const float inv = rsqrt(v41_hc_tg_sum((acc.x + acc.y) + (acc.z + acc.w), partial, lane, sg)
        / float(4 * n4) + a.norm_eps);
    if (tid < 24) mix[tid] *= inv;
    threadgroup_barrier(mem_flags::mem_device);
    // B. controls on simdgroup 0: lane e = source*4 + destination owns one
    // combination entry (lanes 16-31 mirror 0-15); row/column sums gather the
    // four operands by shuffle and add them in the one-thread loop's order.
    if (sg == 0) {
        const float eps = a.hc_eps;
        if (lane < V41_HC_CONNECTIONS) {
            pre[lane] = 1.0f / (1.0f + exp(-(mix[lane] * scale[0] + base[lane]))) + eps;
            post[lane] = 2.0f / (1.0f + exp(-(mix[4 + lane] * scale[1] + base[4 + lane])));
        }
        const uint e = lane & 15u;
        const uint row0 = lane & ~3u;
        const uint col0 = (lane & 16u) | (e & 3u);
        const float value = mix[8 + e] * scale[2] + base[8 + e];
        float row_max = -INFINITY;
        for (uint k = 0; k < V41_HC_CONNECTIONS; ++k) row_max = max(row_max, simd_shuffle(value, row0 + k));
        float comb = exp(value - row_max);
        float sum = 0.0f;
        for (uint k = 0; k < V41_HC_CONNECTIONS; ++k) sum += simd_shuffle(comb, row0 + k);
        const float inverse = 1.0f / sum;
        comb = comb * inverse + eps;
        sum = 0.0f;
        for (uint k = 0; k < V41_HC_CONNECTIONS; ++k) sum += simd_shuffle(comb, col0 + 4u * k);
        comb *= 1.0f / (sum + eps);
        for (uint iteration = 1; iteration < V41_HC_SINKHORN_ITERATIONS; ++iteration) {
            sum = 0.0f;
            for (uint k = 0; k < V41_HC_CONNECTIONS; ++k) sum += simd_shuffle(comb, row0 + k);
            comb *= 1.0f / (sum + eps);
            sum = 0.0f;
            for (uint k = 0; k < V41_HC_CONNECTIONS; ++k) sum += simd_shuffle(comb, col0 + 4u * k);
            comb *= 1.0f / (sum + eps);
        }
        if (lane < 16) combination[lane] = comb;
    }
    // C (stores) and D. weighted RMS norm of the collapsed row.
    const float inv2 = rsqrt(v41_hc_tg_sum((ss2v.x + ss2v.y) + (ss2v.z + ss2v.w), partial, lane, sg)
        / float(a.hidden) + a.norm_eps);
    for (uint j = 0; j < V41_HC_PRE_MAX_D4; ++j) {
        const uint d4 = tid + j * V41_HC_PRE_THREADS;
        if (d4 < h4) {
            collapsed[d4] = col[j];
            normalized[d4] = (col[j] * inv2) * norm_weight[d4];
        }
    }
}

// Decode mHC pre-block as ONE dispatch (25 threadgroups of 1024 threads).
// Threadgroups 0..23 each own one mixing logit; each already streams the whole
// residual for its dot product, so it also sums the squares and applies the
// projected RMS itself (projection before scaling, as before). Threadgroup 24
// collapses the four streams with the lagged incoming pre and applies the
// weighted RMS norm, in parallel with the projection. The sigmoid/Sinkhorn
// controls are first read after the block, so they move into
// kernel_deepseek_v41_hc_post_controls. Reduction order differs from
// projection + kernel_deepseek_v41_hc_pre; nothing else.
kernel void kernel_deepseek_v41_hc_pre_wide(
    constant HcPreArgs &a [[buffer(0)]],
    device const float4 *x [[buffer(1)]],           // residual [4*hidden]
    device const float4 *weight [[buffer(2)]],      // [24, 4*hidden]
    device float *mix [[buffer(3)]],                // [24] scaled out
    device const float *incoming_pre [[buffer(4)]], // [4] lagged
    device float4 *collapsed [[buffer(5)]],         // [hidden]
    device const float4 *norm_weight [[buffer(6)]], // [hidden]
    device float4 *normalized [[buffer(7)]],        // [hidden]
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[V41_HC_PRE_THREADS / 32];
    threadgroup float2 partial2[V41_HC_PRE_THREADS / 32];
    const uint h4 = a.hidden / 4;
    const uint n4 = V41_HC_CONNECTIONS * h4;
    if (row < 24) {
        device const float4 *w = weight + ulong(row) * n4;
        float4 dot = 0.0f, ss = 0.0f;
        uint i = tid;
        for (; i + 3 * V41_HC_PRE_THREADS < n4; i += 4 * V41_HC_PRE_THREADS) {
            const float4 v0 = x[i], v1 = x[i + V41_HC_PRE_THREADS];
            const float4 v2 = x[i + 2 * V41_HC_PRE_THREADS], v3 = x[i + 3 * V41_HC_PRE_THREADS];
            const float4 w0 = w[i], w1 = w[i + V41_HC_PRE_THREADS];
            const float4 w2 = w[i + 2 * V41_HC_PRE_THREADS], w3 = w[i + 3 * V41_HC_PRE_THREADS];
            dot += w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3;
            ss += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        }
        for (; i < n4; i += V41_HC_PRE_THREADS) {
            const float4 v = x[i];
            dot += w[i] * v;
            ss += v * v;
        }
        const float d = simd_sum((dot.x + dot.y) + (dot.z + dot.w));
        const float s = simd_sum((ss.x + ss.y) + (ss.z + ss.w));
        if (lane == 0) partial2[sg] = float2(d, s);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float2 total = 0.0f;
            for (uint k = 0; k < V41_HC_PRE_THREADS / 32; ++k) total += partial2[k];
            mix[row] = total.x * rsqrt(total.y / float(4 * n4) + a.norm_eps);
        }
        return;
    }
    // Collapse with the lagged incoming pre, then the weighted RMS norm.
    const float p0 = incoming_pre[0], p1 = incoming_pre[1], p2 = incoming_pre[2], p3 = incoming_pre[3];
    float4 col[V41_HC_PRE_MAX_D4];
    float4 ss2v = 0.0f;
    for (uint j = 0; j < V41_HC_PRE_MAX_D4; ++j) {
        const uint d4 = tid + j * V41_HC_PRE_THREADS;
        if (d4 < h4) {
            float4 v = 0.0f;
            v += x[d4] * p0;
            v += x[h4 + d4] * p1;
            v += x[2 * h4 + d4] * p2;
            v += x[3 * h4 + d4] * p3;
            col[j] = v;
            ss2v += v * v;
        }
    }
    const float inv2 = rsqrt(v41_hc_tg_sum((ss2v.x + ss2v.y) + (ss2v.z + ss2v.w), partial, lane, sg)
        / float(a.hidden) + a.norm_eps);
    for (uint j = 0; j < V41_HC_PRE_MAX_D4; ++j) {
        const uint d4 = tid + j * V41_HC_PRE_THREADS;
        if (d4 < h4) {
            collapsed[d4] = col[j];
            normalized[d4] = (col[j] * inv2) * norm_weight[d4];
        }
    }
}

// Controls for a site whose pre-block ran kernel_deepseek_v41_hc_pre_wide,
// computed in the prologue of its post. Simdgroup 0 of every threadgroup
// derives pre/post/combination from the 24 scaled mixes with phase B of
// kernel_deepseek_v41_hc_pre (same per-entry arithmetic and operand order);
// threadgroup 0 also stores them for the next site's lagged pre and for any
// reader after the layer. Call from every thread; ends with a barrier.
struct HcPostControlsArgs { uint hidden; float hc_eps; };
inline void v41_hc_post_controls(
    float eps, device const float *mix, device const float *scale, device const float *base,
    device float *pre, device float *post, device float *combination,
    threadgroup float *tg_post, threadgroup float *tg_comb, uint group, uint lane, uint sg) {
    if (sg == 0) {
        if (lane < V41_HC_CONNECTIONS) {
            const float pre_v = 1.0f / (1.0f + exp(-(mix[lane] * scale[0] + base[lane]))) + eps;
            const float post_v = 2.0f / (1.0f + exp(-(mix[4 + lane] * scale[1] + base[4 + lane])));
            tg_post[lane] = post_v;
            if (group == 0) {
                pre[lane] = pre_v;
                post[lane] = post_v;
            }
        }
        const uint e = lane & 15u;
        const uint row0 = lane & ~3u;
        const uint col0 = (lane & 16u) | (e & 3u);
        const float value = mix[8 + e] * scale[2] + base[8 + e];
        float row_max = -INFINITY;
        for (uint k = 0; k < V41_HC_CONNECTIONS; ++k) row_max = max(row_max, simd_shuffle(value, row0 + k));
        float comb = exp(value - row_max);
        float sum = 0.0f;
        for (uint k = 0; k < V41_HC_CONNECTIONS; ++k) sum += simd_shuffle(comb, row0 + k);
        const float inverse = 1.0f / sum;
        comb = comb * inverse + eps;
        sum = 0.0f;
        for (uint k = 0; k < V41_HC_CONNECTIONS; ++k) sum += simd_shuffle(comb, col0 + 4u * k);
        comb *= 1.0f / (sum + eps);
        for (uint iteration = 1; iteration < V41_HC_SINKHORN_ITERATIONS; ++iteration) {
            sum = 0.0f;
            for (uint k = 0; k < V41_HC_CONNECTIONS; ++k) sum += simd_shuffle(comb, row0 + k);
            comb *= 1.0f / (sum + eps);
            sum = 0.0f;
            for (uint k = 0; k < V41_HC_CONNECTIONS; ++k) sum += simd_shuffle(comb, col0 + 4u * k);
            comb *= 1.0f / (sum + eps);
        }
        if (lane < 16) {
            tg_comb[lane] = comb;
            if (group == 0) combination[lane] = comb;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// mHC post (kernel_deepseek_v4_hc_post body) with the site's controls.
kernel void kernel_deepseek_v41_hc_post_controls(
    constant HcPostControlsArgs &a [[buffer(0)]],
    device const float *block [[buffer(1)]],        // [hidden]
    device const float *residual [[buffer(2)]],     // [4*hidden]
    device const float *mix [[buffer(3)]],          // [24] scaled
    device const float *scale [[buffer(4)]],        // [3]
    device const float *base [[buffer(5)]],         // [24]
    device float *pre [[buffer(6)]],                // [4]
    device float *post [[buffer(7)]],               // [4]
    device float *combination [[buffer(8)]],        // [16]
    device float *output [[buffer(9)]],             // [4*hidden]
    uint index [[thread_position_in_grid]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tg_post[V41_HC_CONNECTIONS];
    threadgroup float tg_comb[V41_HC_CONNECTIONS * V41_HC_CONNECTIONS];
    v41_hc_post_controls(a.hc_eps, mix, scale, base, pre, post, combination, tg_post, tg_comb, group, lane, sg);
    if (index >= V41_HC_CONNECTIONS * a.hidden) return;
    const uint destination = index / a.hidden;
    const uint dimension = index % a.hidden;
    float value = block[dimension] * tg_post[destination];
    for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) {
        value += tg_comb[source * V41_HC_CONNECTIONS + destination]
            * residual[source * a.hidden + dimension];
    }
    output[index] = value;
}

// Decode FFN combine as one dispatch: the routing weights apply to the down
// outputs here (down(w*h) becomes w*down(h), so kernel_deepseek_v41_weight
// drops out), the routed sum and shared expert add (kernel_deepseek_v41_sum)
// happen per dimension, then the mHC post with the site's controls. One row.
struct FfnPostControlsArgs { uint hidden; float hc_eps; uint topk; };
kernel void kernel_deepseek_v41_ffn_post_controls(
    constant FfnPostControlsArgs &a [[buffer(0)]],
    device const float *slots [[buffer(1)]],        // [topk, hidden] down outputs
    device const float *weights [[buffer(2)]],      // [topk]
    device const float *shared [[buffer(3)]],       // [hidden]
    device const int *status [[buffer(4)]],         // [1]
    device const float *residual [[buffer(5)]],     // [4*hidden]
    device const float *mix [[buffer(6)]],          // [24] scaled
    device const float *scale [[buffer(7)]],        // [3]
    device const float *base [[buffer(8)]],         // [24]
    device float *pre [[buffer(9)]],                // [4]
    device float *post [[buffer(10)]],              // [4]
    device float *combination [[buffer(11)]],       // [16]
    device float *output [[buffer(12)]],            // [4*hidden]
    uint index [[thread_position_in_grid]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tg_post[V41_HC_CONNECTIONS];
    threadgroup float tg_comb[V41_HC_CONNECTIONS * V41_HC_CONNECTIONS];
    v41_hc_post_controls(a.hc_eps, mix, scale, base, pre, post, combination, tg_post, tg_comb, group, lane, sg);
    if (index >= V41_HC_CONNECTIONS * a.hidden) return;
    const uint destination = index / a.hidden;
    const uint dimension = index % a.hidden;
    float sum = 0.0f;
    if (status[0] == 1) {
        for (uint k = 0; k < a.topk; ++k) sum += weights[k] * slots[k * a.hidden + dimension];
    }
    const float block = sum + shared[dimension];
    float value = block * tg_post[destination];
    for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) {
        value += tg_comb[source * V41_HC_CONNECTIONS + destination]
            * residual[source * a.hidden + dimension];
    }
    output[index] = value;
}

struct RouteArgs { uint experts; uint topk; float temperature; float scale; };
inline bool finite_bits(float v) { return (as_type<uint>(v) & 0x7f800000u) != 0x7f800000u; }
// MSL has no log1p. Preserve tiny softplus tails instead of rounding 1+x to 1.
inline float log1p_positive(float x) {
    if (x < 0.01f) return x * (1.0f + x * (-0.5f + x * (1.0f / 3.0f - x * 0.25f)));
    return log(1.0f + x);
}
// One 256-thread threadgroup per row. Same per-expert arithmetic, status
// codes, lowest-ID tie-breaks, ascending ID order and sequential weight
// normalization as the serial reference below, so outputs are bitwise equal;
// only the work distribution changes (the serial form cost ~0.4 ms per layer
// on M5 Ultra, ~16 ms of a decode token).
#define ROUTE_THREADS 256u
#define ROUTE_MAX_TOPK 16u
#define ROUTE_NO_FAIL 0xffffffffu
kernel void kernel_deepseek_v41_route(
    constant RouteArgs &a [[buffer(0)]], device const float *logits [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *scores [[buffer(3)]],
    device int *ids [[buffer(4)]], device float *weights [[buffer(5)]],
    device int *status [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    logits += row * a.experts;
    scores += row * a.experts;
    ids += row * a.topk;
    weights += row * a.topk;
    status += row;
    threadgroup uint tg_fail[ROUTE_THREADS / 32];
    threadgroup float tg_value[ROUTE_THREADS / 32];
    threadgroup int tg_id[ROUTE_THREADS / 32];
    threadgroup int tg_ids[ROUTE_MAX_TOPK];
    if (tid == 0) {
        *status = -1;
        for (uint k = 0; k < a.topk; ++k) { ids[k] = -1; weights[k] = 0; }
    }
    // Phase 1: scores, with the serial kernel's first-failure precedence:
    // at index i, a non-finite logit (-1) precedes a non-finite bias (-2),
    // which precedes a non-finite score or score+bias (-1).
    uint fail = ROUTE_NO_FAIL;
    for (uint i = tid; i < a.experts; i += ROUTE_THREADS) {
        if (!finite_bits(logits[i])) { fail = min(fail, i * 4u); continue; }
        if (!finite_bits(bias[i])) { fail = min(fail, i * 4u + 1u); continue; }
        float z = logits[i] / a.temperature;
        scores[i] = sqrt(max(z, 0.0f) + log1p_positive(exp(-abs(z))));
        if (!finite_bits(scores[i]) || !finite_bits(scores[i] + bias[i])) fail = min(fail, i * 4u + 2u);
    }
    fail = simd_min(fail);
    if (lane == 0) tg_fail[sg] = fail;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    fail = ROUTE_NO_FAIL;
    for (uint s = 0; s < ROUTE_THREADS / 32; ++s) fail = min(fail, tg_fail[s]);
    if (fail != ROUTE_NO_FAIL) {
        if (tid == 0 && (fail & 3u) == 1u) *status = -2;
        return;
    }
    // Phase 2: top-k by candidate value, lowest ID on ties, excluding chosen.
    for (uint k = 0; k < a.topk; ++k) {
        int best = -1;
        float value = -INFINITY;
        for (uint i = tid; i < a.experts; i += ROUTE_THREADS) {
            bool used = false;
            for (uint j = 0; j < k; ++j) used |= tg_ids[j] == int(i);
            float candidate = scores[i] + bias[i];
            if (!used && (best < 0 || candidate > value)) { best = int(i); value = candidate; }
        }
        float m = simd_max(value);
        int id = simd_min(best >= 0 && value == m ? best : INT_MAX);
        if (lane == 0) { tg_value[sg] = m; tg_id[sg] = id; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float best_value = -INFINITY;
        int best_id = INT_MAX;
        for (uint s = 0; s < ROUTE_THREADS / 32; ++s) {
            if (tg_id[s] == INT_MAX) continue;
            if (tg_value[s] > best_value || (tg_value[s] == best_value && tg_id[s] < best_id)) {
                best_value = tg_value[s];
                best_id = tg_id[s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) tg_ids[k] = best_id == INT_MAX ? -1 : best_id;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Phase 3: ascending IDs, sequential normalization, serial order.
    if (tid == 0) {
        for (uint k = 0; k < a.topk; ++k) ids[k] = tg_ids[k];
        for (uint k = 1; k < a.topk; ++k) {
            int v = ids[k]; uint j = k;
            while (j && ids[j - 1] > v) { ids[j] = ids[j - 1]; --j; }
            ids[j] = v;
        }
        float denom = 1.0f;
        if (a.topk > 1) {
            denom = 0;
            for (uint k = 0; k < a.topk; ++k) denom += scores[ids[k]];
            denom += 1e-20f;
        }
        for (uint k = 0; k < a.topk; ++k) {
            weights[k] = scores[ids[k]] / denom * a.scale;
            if (!finite_bits(weights[k])) { *status = -6; return; }
        }
        *status = 1;
    }
}

// Serial reference (one thread per row): the parity oracle for the kernel above.
kernel void kernel_deepseek_v41_route_serial(
    constant RouteArgs &a [[buffer(0)]], device const float *logits [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *scores [[buffer(3)]],
    device int *ids [[buffer(4)]], device float *weights [[buffer(5)]],
    device int *status [[buffer(6)]], uint row [[threadgroup_position_in_grid]]) {
    logits += row * a.experts;
    scores += row * a.experts;
    ids += row * a.topk;
    weights += row * a.topk;
    status += row;
    *status = -1;
    for (uint k = 0; k < a.topk; ++k) { ids[k] = -1; weights[k] = 0; }
    for (uint i = 0; i < a.experts; ++i) {
        if (!finite_bits(logits[i])) return;
        if (!finite_bits(bias[i])) { *status = -2; return; }
        float z = logits[i] / a.temperature;
        scores[i] = sqrt(max(z, 0.0f) + log1p_positive(exp(-abs(z))));
        if (!finite_bits(scores[i]) || !finite_bits(scores[i] + bias[i])) return;
    }
    for (uint k = 0; k < a.topk; ++k) {
        int best = -1;
        float value = -INFINITY;
        for (uint i = 0; i < a.experts; ++i) {
            bool used = false;
            for (uint j = 0; j < k; ++j) used |= ids[j] == int(i);
            float candidate = scores[i] + bias[i];
            if (!used && (best < 0 || candidate > value)) { best = int(i); value = candidate; }
        }
        ids[k] = best;
    }
    for (uint k = 1; k < a.topk; ++k) {
        int v = ids[k]; uint j = k;
        while (j && ids[j - 1] > v) { ids[j] = ids[j - 1]; --j; }
        ids[j] = v;
    }
    float denom = 1.0f;
    if (a.topk > 1) {
        denom = 0;
        for (uint k = 0; k < a.topk; ++k) denom += scores[ids[k]];
        denom += 1e-20f;
    }
    for (uint k = 0; k < a.topk; ++k) {
        weights[k] = scores[ids[k]] / denom * a.scale;
        if (!finite_bits(weights[k])) { *status = -6; return; }
    }
    *status = 1;
}

kernel void kernel_deepseek_v41_weight(
    constant uint &n [[buffer(0)]], constant uint &width [[buffer(1)]],
    device const float *x [[buffer(2)]], device const float *weights [[buffer(3)]],
    device float *out [[buffer(4)]], device const int *status [[buffer(5)]],
    constant uint &topk [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = status[i / (width * topk)] == 1 ? x[i] * weights[i / width] : 0.0f;
}

kernel void kernel_deepseek_v41_sum(
    constant uint &hidden [[buffer(0)]], constant uint &topk [[buffer(1)]],
    device const float *slots [[buffer(2)]], device const float *shared [[buffer(3)]],
    device float *out [[buffer(4)]], device const int *status [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= hidden * rows) return;
    uint row = i / hidden, channel = i % hidden;
    float sum = 0;
    if (status[row] == 1) for (uint k = 0; k < topk; ++k) sum += slots[(row * topk + k) * hidden + channel];
    out[i] = sum + shared[i];
}
