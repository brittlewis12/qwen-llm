// Full-attention helper kernels: Q/K norm, gate split, and a naive F16-KV
// decode shared by the Qwen3.5/3.6, DFlash, MTP, and Muse paths.
//
// Kernels:
//
//   * kernel_rms_norm_batched_f32 / _src_strided_f32 — per-head RMSNorm
//     with shared per-channel weight. Used for Q-norm and K-norm (both
//     share the same head_dim weight, applied across all heads).
//
//   * kernel_qk_rms_norm_rope_f32_packed_consecutive — fused Q/K norm +
//     RoPE for packed consecutive positions.
//
//   * kernel_split_q_gate_f32 / kernel_split_qkv_fused_f32 — un-interleave
//     the Q-projection output from `[h0_q(hd), h0_gate(hd), ...]` into
//     separate `q[n_heads, head_dim]` and `gate[n_heads, head_dim]`
//     tensors (gated-attention layout where q_proj outputs 2× the heads).
//
//   * kernel_attn_decode_f16kv — naive fused single-token attention over
//     an F16 KV cache: scoring, softmax, V-aggregate per Q head, GQA-aware.
//     Scores live in threadgroup memory; the host caps n_pos at 7,168. The flash
//     kernels in attn_v4.metal are the production path; this remains the
//     small-context/reference decode used by several families.

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

// ---- per-head RMSNorm reading strided source rows ---------------------------
//
// v0.432: same math as kernel_rms_norm_batched_f32, but row `hi` of the
// source lives at `x[src_offset + hi * src_stride .. + head_dim]`. This
// lets the q-norm read the Q halves of the interleaved q_proj output
// ([head_dim Q, head_dim gate] per head) directly, deleting the
// split_q_gate layout-copy dispatch. Per-row arithmetic order is
// identical to the compact kernel, so results are bit-identical.

struct rms_norm_batched_strided_args {
    uint n_heads;
    uint head_dim;
    uint src_stride; // elements between consecutive source rows
    uint src_offset; // element offset of source row 0
    float eps;
};

kernel void kernel_rms_norm_batched_src_strided_f32(
        constant rms_norm_batched_strided_args & args [[buffer(0)]],
        device const float * x      [[buffer(1)]], // strided rows (see args)
        device const float * weight [[buffer(2)]], // [head_dim]
        device       float * y      [[buffer(3)]], // [n_heads, head_dim] compact
        threadgroup  float * shmem  [[threadgroup(0)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        uint  tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint  ntg   [[threads_per_threadgroup]]) {
    const uint hi = tgpig;
    if (hi >= args.n_heads) return;

    device const float * x_h = x + (ulong)args.src_offset + (ulong)hi * args.src_stride;
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

// ---- paired Q/K RMSNorm + consecutive-position RoPE ------------------------

struct qk_rms_norm_rope_args {
    uint n_tokens;
    uint n_q_heads;
    uint n_k_heads;
    uint head_dim;
    uint n_rot;
    uint start_position;
    float eps;
    float theta_base;
};

kernel void kernel_qk_rms_norm_rope_f32_packed_consecutive(
        constant qk_rms_norm_rope_args & args [[buffer(0)]],
        device const float * q_src    [[buffer(1)]], // [token, q_head, Q|gate]
        device const float * q_weight [[buffer(2)]], // [head_dim]
        device       float * q_out    [[buffer(3)]], // [token, q_head, head_dim]
        device const float * k_src    [[buffer(4)]], // [token, k_head, head_dim]
        device const float * k_weight [[buffer(5)]], // [head_dim]
        device       float * k_out    [[buffer(6)]], // [token, k_head, head_dim]
        threadgroup  float * shmem    [[threadgroup(0)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        uint  tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint  ntg   [[threads_per_threadgroup]]) {
    const uint q_rows = args.n_tokens * args.n_q_heads;
    const uint k_rows = args.n_tokens * args.n_k_heads;
    if (tgpig >= q_rows + k_rows) return;

    const bool is_q = tgpig < q_rows;
    const uint row = is_q ? tgpig : tgpig - q_rows;
    const uint heads = is_q ? args.n_q_heads : args.n_k_heads;
    const uint token = row / heads;
    device const float * x_h = is_q
        ? q_src + (ulong)row * 2u * args.head_dim
        : k_src + (ulong)row * args.head_dim;
    device const float * weight = is_q ? q_weight : k_weight;
    device float * y_h = is_q
        ? q_out + (ulong)row * args.head_dim
        : k_out + (ulong)row * args.head_dim;

    float sumsq = 0.0f;
    for (uint i = tpitg; i < args.head_dim; i += ntg) {
        const float value = x_h[i];
        sumsq += value * value;
    }
    sumsq = simd_sum(sumsq);
    if (tiisg == 0) shmem[sgitg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float mean = sumsq / float(args.head_dim);
    const float scale = 1.0f / sqrt(mean + args.eps);
    const uint half_rot = args.n_rot / 2u;
    const float position = float(args.start_position + token);
    for (uint i = tpitg; i < args.head_dim; i += ntg) {
        if (i < half_rot) {
            const float a = (x_h[i] * scale) * weight[i];
            const float b = (x_h[i + half_rot] * scale) * weight[i + half_rot];
            const float exponent = float(2u * i) / float(args.n_rot);
            const float angle = position / pow(args.theta_base, exponent);
            float cosine;
            const float sine = sincos(angle, cosine);
            y_h[i] = a * cosine - b * sine;
            y_h[i + half_rot] = a * sine + b * cosine;
        } else if (i >= args.n_rot) {
            y_h[i] = (x_h[i] * scale) * weight[i];
        }
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
//
// v0.432: production attention paths no longer dispatch this — the q-norm
// reads Q halves via kernel_rms_norm_batched_src_strided_f32 and the gate
// consumer reads gate halves via kernel_sigmoid_mul_gate_strided_f32
// (elementwise.metal). Kept for tests/tools.

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

struct split_qkv_fused_args {
    uint n_rows;
    uint q_full_dim;
    uint kv_dim;
    uint fused_stride;
};

kernel void kernel_split_qkv_fused_f32(
        constant split_qkv_fused_args & args [[buffer(0)]],
        device const float * src   [[buffer(1)]], // [n_rows, q_full_dim + 2*kv_dim]
        device       float * qfull [[buffer(2)]], // [n_rows, q_full_dim]
        device       float * k_out [[buffer(3)]], // [n_rows, kv_dim]
        device       float * v_out [[buffer(4)]], // [n_rows, kv_dim]
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.n_rows * args.fused_stride;
    if (tid >= total) return;
    const uint row = tid / args.fused_stride;
    const uint col = tid % args.fused_stride;
    const uint src_off = row * args.fused_stride;
    if (col < args.q_full_dim) {
        qfull[row * args.q_full_dim + col] = src[src_off + col];
    } else if (col < args.q_full_dim + args.kv_dim) {
        const uint kcol = col - args.q_full_dim;
        k_out[row * args.kv_dim + kcol] = src[src_off + col];
    } else {
        const uint vcol = col - args.q_full_dim - args.kv_dim;
        v_out[row * args.kv_dim + vcol] = src[src_off + col];
    }
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

    // Pass 2: exp-shifted softmax numerators (scores remain unnormalized;
    // the final inverse sum is folded into the V output below).
    float local_max = -INFINITY;
    for (uint p = tpitg; p < args.n_pos; p += ntg) local_max = max(local_max, scores[p]);
    local_max = simd_max(local_max);
    if (tiisg == 0) shred[sgitg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local_max = (tiisg < (ntg + 31) / 32) ? shred[tiisg] : -INFINITY;
    local_max = simd_max(local_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    // Pass 3: out[d] = inv_sum * sum_p scores[p] * float(v_cache[p, kvh, d]).
    for (uint d = tpitg; d < args.head_dim; d += ntg) {
        float acc = 0.0f;
        for (uint p = 0; p < args.n_pos; ++p) {
            acc += scores[p] * (float)v_cache[
                (ulong)p * args.kv_stride
                + (ulong)kvh * args.head_dim
                + d
            ];
        }
        out_h[d] = acc * inv_sum;
    }
}
