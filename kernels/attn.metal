// Full-attention kernels for Qwen3.5/3.6 gated-attention layers.
//
// Three kernels:
//
//   * kernel_rms_norm_batched_f32 — per-head RMSNorm with shared
//     per-channel weight. Used for Q-norm and K-norm (both share the
//     same head_dim weight, applied across all heads).
//
//   * kernel_split_q_gate_f32 — un-interleave the Q-projection output
//     from `[h0_q(hd), h0_gate(hd), h1_q(hd), h1_gate(hd), ...]` into
//     separate `q[n_heads, head_dim]` and `gate[n_heads, head_dim]`
//     tensors. The input layout comes from the gated-attention trick
//     where q_proj outputs 2× the heads' worth, with Q and gate
//     interleaved per head.
//
//   * kernel_attn_decode_f32 — fused single-token attention: scoring,
//     softmax, and V-aggregate per Q head. GQA-aware (each Q head
//     reads its corresponding K/V head via `qh / group`). One
//     threadgroup per Q head; scores held in threadgroup memory.
//     For v1: requires n_pos * sizeof(float) ≤ threadgroup_memory_max
//     (~32 KB on Apple Silicon, so n_pos ≤ 8192). v2 will switch to
//     streaming softmax for long context.

#include <metal_stdlib>
using namespace metal;

// ---- per-head RMSNorm with shared weight ------------------------------------

struct rms_norm_batched_args {
    uint n_heads;
    uint head_dim;
    float eps;
};

kernel void kernel_rms_norm_batched_f32(
        constant rms_norm_batched_args & args [[buffer(0)]],
        device const float * x      [[buffer(1)]], // [n_heads, head_dim]
        device const float * weight [[buffer(2)]], // [head_dim]
        device       float * y      [[buffer(3)]], // [n_heads, head_dim]
        threadgroup  float * shmem  [[threadgroup(0)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        uint  tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint  ntg   [[threads_per_threadgroup]]) {
    const uint hi = tgpig;
    if (hi >= args.n_heads) return;

    device const float * x_h = x + (ulong)hi * args.head_dim;
    device       float * y_h = y + (ulong)hi * args.head_dim;

    float sumsq = 0.0f;
    for (uint i = tpitg; i < args.head_dim; i += ntg) {
        const float v = x_h[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    if (tiisg == 0) shmem[sgitg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float mean = sumsq / float(args.head_dim);
    const float scale = 1.0f / sqrt(mean + args.eps);

    for (uint i = tpitg; i < args.head_dim; i += ntg) {
        y_h[i] = (x_h[i] * scale) * weight[i];
    }
}

// ---- split Q + gate from interleaved layout --------------------------------
//
// q_full layout (per Q head): [head_dim Q values, head_dim gate values]
// Total size: n_heads * 2 * head_dim.
//
// Output layouts:
//   q[hi, d] = q_full[hi * 2 * head_dim + d]
//   gate[hi, d] = q_full[hi * 2 * head_dim + head_dim + d]
//
// Grid: (n_heads * head_dim) — one thread per output element of one tensor.
// Both writes happen per thread (same source row).

struct split_q_gate_args {
    uint n_heads;
    uint head_dim;
};

kernel void kernel_split_q_gate_f32(
        constant split_q_gate_args & args [[buffer(0)]],
        device const float * q_full [[buffer(1)]], // [n_heads, 2*head_dim]
        device       float * q      [[buffer(2)]], // [n_heads, head_dim]
        device       float * gate   [[buffer(3)]], // [n_heads, head_dim]
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.n_heads * args.head_dim;
    if (tid >= total) return;
    const uint hi = tid / args.head_dim;
    const uint d  = tid % args.head_dim;
    const uint src_off = hi * 2u * args.head_dim;
    q[tid]    = q_full[src_off + d];
    gate[tid] = q_full[src_off + args.head_dim + d];
}

// ---- fused attention decode (single Q step) --------------------------------
//
// One threadgroup per Q head. Each threadgroup:
//   (1) Computes scores[p] = q · k_cache[p, kvh, :] * scale, in shmem.
//   (2) Softmax over scores (numerically stable: max-subtract then sum-exp).
//   (3) Aggregates V: out[d] = sum_p scores[p] * v_cache[p, kvh, d].
//
// Constraint: scores buffer is in threadgroup memory of size
// n_pos * sizeof(float). Apple Silicon allows ~32 KB, so n_pos ≤ ~8000.
// For longer contexts we'd switch to a streaming-softmax pattern.

struct attn_decode_args {
    uint  n_q_heads;
    uint  n_kv_heads;
    uint  head_dim;
    uint  n_pos;
    uint  kv_stride; // n_kv_heads * head_dim
    float scale;
};

// F16 KV cache variant. Same algorithm, K/V read as half (cast to float
// in the math). Halves K/V bandwidth at long context — at 4K positions
// for 27B (16 layers × 4 KV heads × 256 head_dim × 4096 pos × 2B/elem
// vs ×4B for F32) this saves ~4 GB of reads per token. Matches what
// llama.cpp does by default (--cache-type-k f16).
//
// Precision: KV magnitudes are typically in [-8, +8] after q/k norm and
// before RoPE; F16 has ~10 bits of mantissa so quantization error is
// ~1e-3 per element. Far below the per-token rounding noise we already
// see in F32 K-quant kernels. Validation against the F32 KV path is
// expected to give cos > 0.999.
kernel void kernel_attn_decode_f16kv(
        constant attn_decode_args & args [[buffer(0)]],
        device const float * q       [[buffer(1)]], // [n_q_heads, head_dim] F32
        device const half  * k_cache [[buffer(2)]], // [capacity, n_kv_heads, head_dim] F16
        device const half  * v_cache [[buffer(3)]], // [capacity, n_kv_heads, head_dim] F16
        device       float * out     [[buffer(4)]],
        threadgroup  float * scores  [[threadgroup(0)]],
        threadgroup  float * shred   [[threadgroup(1)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        uint  tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint  ntg   [[threads_per_threadgroup]]) {
    const uint qh = tgpig;
    if (qh >= args.n_q_heads) return;
    const uint group = args.n_q_heads / args.n_kv_heads;
    const uint kvh = qh / group;
    device const float * q_h = q + (ulong)qh * args.head_dim;
    device       float * out_h = out + (ulong)qh * args.head_dim;

    // Pass 1: scores[p] = (q · half2float(k_cache[p, kvh, :])) * scale
    for (uint p = tpitg; p < args.n_pos; p += ntg) {
        device const half * k_p = k_cache
            + (ulong)p * args.kv_stride
            + (ulong)kvh * args.head_dim;
        float s = 0.0f;
        for (uint i = 0; i < args.head_dim; ++i) {
            s += q_h[i] * (float)k_p[i];
        }
        scores[p] = s * args.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: softmax (identical to F32 path; scores are already F32).
    float local_max = -INFINITY;
    for (uint p = tpitg; p < args.n_pos; p += ntg) local_max = max(local_max, scores[p]);
    local_max = simd_max(local_max);
    if (tiisg == 0) shred[sgitg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local_max = (tiisg < (ntg + 31) / 32) ? shred[tiisg] : -INFINITY;
    local_max = simd_max(local_max);

    float local_sum = 0.0f;
    for (uint p = tpitg; p < args.n_pos; p += ntg) {
        const float e = exp(scores[p] - local_max);
        scores[p] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (tiisg == 0) shred[sgitg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local_sum = (tiisg < (ntg + 31) / 32) ? shred[tiisg] : 0.0f;
    local_sum = simd_sum(local_sum);

    const float inv_sum = 1.0f / local_sum;
    for (uint p = tpitg; p < args.n_pos; p += ntg) scores[p] *= inv_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: out[d] = sum_p scores[p] * float(v_cache[p, kvh, d]).
    for (uint d = tpitg; d < args.head_dim; d += ntg) {
        float acc = 0.0f;
        for (uint p = 0; p < args.n_pos; ++p) {
            acc += scores[p] * (float)v_cache[
                (ulong)p * args.kv_stride
                + (ulong)kvh * args.head_dim
                + d
            ];
        }
        out_h[d] = acc;
    }
}

kernel void kernel_attn_decode_f32(
        constant attn_decode_args & args [[buffer(0)]],
        device const float * q       [[buffer(1)]], // [n_q_heads, head_dim]
        device const float * k_cache [[buffer(2)]], // [capacity, n_kv_heads, head_dim]
        device const float * v_cache [[buffer(3)]], // [capacity, n_kv_heads, head_dim]
        device       float * out     [[buffer(4)]], // [n_q_heads, head_dim]
        threadgroup  float * scores  [[threadgroup(0)]], // [n_pos]
        threadgroup  float * shred   [[threadgroup(1)]], // simdgroup reduce scratch
        uint  tgpig [[threadgroup_position_in_grid]],
        uint  tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint  ntg   [[threads_per_threadgroup]]) {
    const uint qh = tgpig;
    if (qh >= args.n_q_heads) return;

    const uint group = args.n_q_heads / args.n_kv_heads;
    const uint kvh = qh / group;

    device const float * q_h = q + (ulong)qh * args.head_dim;
    device       float * out_h = out + (ulong)qh * args.head_dim;

    // ---- Pass 1: scores[p] = (q_h · k_cache[p, kvh, :]) * scale ----
    for (uint p = tpitg; p < args.n_pos; p += ntg) {
        device const float * k_p = k_cache
            + (ulong)p * args.kv_stride
            + (ulong)kvh * args.head_dim;
        float s = 0.0f;
        for (uint i = 0; i < args.head_dim; ++i) {
            s += q_h[i] * k_p[i];
        }
        scores[p] = s * args.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Pass 2: softmax ----
    // (a) Max-reduction across threadgroup.
    float local_max = -INFINITY;
    for (uint p = tpitg; p < args.n_pos; p += ntg) {
        local_max = max(local_max, scores[p]);
    }
    local_max = simd_max(local_max);
    if (tiisg == 0) shred[sgitg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local_max = (tiisg < (ntg + 31) / 32) ? shred[tiisg] : -INFINITY;
    local_max = simd_max(local_max);

    // (b) Subtract max, exp, sum.
    float local_sum = 0.0f;
    for (uint p = tpitg; p < args.n_pos; p += ntg) {
        const float e = exp(scores[p] - local_max);
        scores[p] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (tiisg == 0) shred[sgitg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local_sum = (tiisg < (ntg + 31) / 32) ? shred[tiisg] : 0.0f;
    local_sum = simd_sum(local_sum);

    const float inv_sum = 1.0f / local_sum;
    for (uint p = tpitg; p < args.n_pos; p += ntg) {
        scores[p] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Pass 3: out[d] = sum_p scores[p] * v_cache[p, kvh, d] ----
    //
    // Each thread owns a stride-`ntg` subset of dims. For each owned dim,
    // it accumulates in a REGISTER across all positions. Critical: each
    // thread reads a different (d) column for a fixed p, and adjacent
    // threads (tpitg = 0, 1, 2, ...) read adjacent d's of the same v row,
    // which IS coalesced — adjacent threads, adjacent memory addresses.
    //
    // The original code had this structure too. The performance issue I
    // suspected was strided reads across the V cache, but it's actually
    // already cache-friendly: for each p, threads read v[p, kvh, tpitg],
    // v[p, kvh, tpitg+ntg], ... contiguous within the row, and the row is
    // contiguous in memory. Different p's use different rows but the page
    // pattern is sequential.
    //
    // Reverting to the original structure but keeping it explicit.
    for (uint d = tpitg; d < args.head_dim; d += ntg) {
        float acc = 0.0f;
        for (uint p = 0; p < args.n_pos; ++p) {
            acc += scores[p] * v_cache[
                (ulong)p * args.kv_stride
                + (ulong)kvh * args.head_dim
                + d
            ];
        }
        out_h[d] = acc;
    }
}
