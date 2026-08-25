// Partial RoPE — IMROPE / NEOX pairing for Qwen3.5/3.6.
//
// For each head, rotates the first `n_rot` dims with NEOX pairing:
//   pair (buf[i], buf[i + n_rot/2])  for i in [0, n_rot/2)
// and leaves dims [n_rot, head_dim) unchanged. For text-only positions
// (which is what we target in v1), MRoPE/IMRoPE collapse to plain
// position-coordinate RoPE — sections only differ for vision/video.
//
// CPU oracle: `crate::forward::rope_in_place`, which uses the same NEOX
// pairing.
//
// One thread per (head, pair). Threadgroup geometry:
//   threadgroups = (n_heads * (n_rot/2)) / 64 — flat 1D
//   threads/group = 64
//
// `theta_base` is `arch.rope_theta` (10_000_000 for Qwen3.5/3.6).

#include <metal_stdlib>
using namespace metal;

struct rope_args {
    uint  n_heads;
    uint  head_dim;
    uint  n_rot;       // rotated dim count (head_dim * partial_rotary_factor)
    uint  position;    // sequence position
    float theta_base;
};

inline float2 rope_sincos_native(float angle) {
    float cosine;
    const float sine = sincos(angle, cosine);
    return float2(cosine, sine);
}

// Degree-9 sine and degree-10 cosine absolute-minimax polynomials from
// https://publik-void.github.io/sin-cos-approximations/. Cody-Waite-style
// reduction folds the f32 input into the fitted [0, pi/2] interval.
inline float2 rope_sincos_minimax_f32(float angle) {
    constexpr float inv_two_pi = 0.15915494309189535f;
    constexpr float two_pi_hi = 6.28125f;
    constexpr float two_pi_lo = 0.0019353071795864769f;
    constexpr float pi = 3.14159265358979323846f;
    constexpr float half_pi = 1.57079632679489661923f;
    constexpr float two_pi = 6.28318530717958647692f;

    const float quotient = rint(angle * inv_two_pi);
    float reduced = fma(-quotient, two_pi_hi, angle);
    reduced = fma(-quotient, two_pi_lo, reduced);
    if (reduced > pi) reduced -= two_pi;
    if (reduced < -pi) reduced += two_pi;

    const float magnitude = abs(reduced);
    const bool reflected = magnitude > half_pi;
    const float x = select(magnitude, pi - magnitude, reflected);
    const float x2 = x * x;

    const float sine_magnitude = x * fma(
        x2,
        fma(
            x2,
            fma(x2, fma(x2, 2.5904885005360523e-6f, -0.0001980089776279543f),
                0.008332899823351751f),
            -0.16666647634639713f),
        0.9999999765898821f);
    const float cosine_magnitude = fma(
        x2,
        fma(
            x2,
            fma(
                x2,
                fma(x2, fma(x2, -2.605149521548271e-7f, 0.000024760161352583124f),
                    -0.001388836140027525f),
                0.0416666362580703f),
            -0.4999999935847177f),
        0.9999999997806517f);

    const float sine = copysign(sine_magnitude, reduced);
    const float cosine = select(cosine_magnitude, -cosine_magnitude, reflected);
    return float2(cosine, sine);
}

inline void rope_neox_rotate_head(
        device float * values,
        uint head,
        uint head_dim,
        uint half_rot,
        uint pair,
        float2 cosine_sine) {
    const uint base = head * head_dim;
    const float a = values[base + pair];
    const float b = values[base + pair + half_rot];
    values[base + pair] = a * cosine_sine.x - b * cosine_sine.y;
    values[base + pair + half_rot] = a * cosine_sine.y + b * cosine_sine.x;
}

kernel void kernel_rope_neox_f32(
        constant rope_args & args  [[buffer(0)]],
        device       float * buf   [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
    const uint half_rot = args.n_rot / 2u;
    const uint total_pairs = args.n_heads * half_rot;
    if (tid >= total_pairs) return;

    const uint hi = tid / half_rot;
    const uint i  = tid % half_rot;

    // theta_i = position / theta_base^(2i / n_rot)
    const float exponent = float(2u * i) / float(args.n_rot);
    const float freq = float(args.position) / pow(args.theta_base, exponent);
    const float c = cos(freq);
    const float s = sin(freq);

    const uint base = hi * args.head_dim;
    const float a = buf[base + i];
    const float b = buf[base + i + half_rot];
    buf[base + i]            = a * c - b * s;
    buf[base + i + half_rot] = a * s + b * c;
}

// Q and K use the same position and frequency table. Process both in one
// dispatch; for the shared head prefix, calculate sin/cos once and apply it
// to both buffers. Heads beyond the shorter tensor still execute normally.
struct rope_pair_args {
    uint  n_q_heads;
    uint  n_k_heads;
    uint  head_dim;
    uint  n_rot;
    uint  position;
    float theta_base;
};

kernel void kernel_rope_neox_pair_f32(
        constant rope_pair_args & args [[buffer(0)]],
        device float * q              [[buffer(1)]],
        device float * k              [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    const uint half_rot = args.n_rot / 2u;
    const uint n_heads = max(args.n_q_heads, args.n_k_heads);
    const uint total_pairs = n_heads * half_rot;
    if (tid >= total_pairs) return;

    const uint hi = tid / half_rot;
    const uint i = tid % half_rot;
    const float exponent = float(2u * i) / float(args.n_rot);
    const float freq = float(args.position) / pow(args.theta_base, exponent);
    const float c = cos(freq);
    const float s = sin(freq);

    if (hi < args.n_q_heads) {
        const uint base = hi * args.head_dim;
        const float a = q[base + i];
        const float b = q[base + i + half_rot];
        q[base + i] = a * c - b * s;
        q[base + i + half_rot] = a * s + b * c;
    }
    if (hi < args.n_k_heads) {
        const uint base = hi * args.head_dim;
        const float a = k[base + i];
        const float b = k[base + i + half_rot];
        k[base + i] = a * c - b * s;
        k[base + i + half_rot] = a * s + b * c;
    }
}

// Same head-parallel geometry as kernel_rope_neox_pair_f32, but requests the
// combined Metal intrinsic so sine and cosine can share argument reduction.
kernel void kernel_rope_neox_pair_sincos_f32(
        constant rope_pair_args & args [[buffer(0)]],
        device float * q              [[buffer(1)]],
        device float * k              [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    const uint half_rot = args.n_rot / 2u;
    const uint n_heads = max(args.n_q_heads, args.n_k_heads);
    const uint total_pairs = n_heads * half_rot;
    if (tid >= total_pairs) return;

    const uint head = tid / half_rot;
    const uint pair = tid % half_rot;
    const float exponent = float(2u * pair) / float(args.n_rot);
    const float angle = float(args.position) / pow(args.theta_base, exponent);
    const float2 cosine_sine = rope_sincos_native(angle);

    if (head < args.n_q_heads) {
        rope_neox_rotate_head(q, head, args.head_dim, half_rot, pair, cosine_sine);
    }
    if (head < args.n_k_heads) {
        rope_neox_rotate_head(k, head, args.head_dim, half_rot, pair, cosine_sine);
    }
}

// One lane owns a rotary pair and walks every Q/K head. This preserves
// coalesced per-head accesses while computing pow/sincos once instead of once
// per head.
kernel void kernel_rope_neox_pair_shared_f32(
        constant rope_pair_args & args [[buffer(0)]],
        device float * q              [[buffer(1)]],
        device float * k              [[buffer(2)]],
        uint pair [[thread_position_in_grid]]) {
    const uint half_rot = args.n_rot / 2u;
    if (pair >= half_rot) return;

    const float exponent = float(2u * pair) / float(args.n_rot);
    const float angle = float(args.position) / pow(args.theta_base, exponent);
    const float2 cosine_sine = rope_sincos_native(angle);
    for (uint head = 0u; head < args.n_q_heads; ++head) {
        rope_neox_rotate_head(q, head, args.head_dim, half_rot, pair, cosine_sine);
    }
    for (uint head = 0u; head < args.n_k_heads; ++head) {
        rope_neox_rotate_head(k, head, args.head_dim, half_rot, pair, cosine_sine);
    }
}

kernel void kernel_rope_neox_pair_shared_minimax_f32(
        constant rope_pair_args & args [[buffer(0)]],
        device float * q              [[buffer(1)]],
        device float * k              [[buffer(2)]],
        uint pair [[thread_position_in_grid]]) {
    const uint half_rot = args.n_rot / 2u;
    if (pair >= half_rot) return;

    const float exponent = float(2u * pair) / float(args.n_rot);
    const float angle = float(args.position) / pow(args.theta_base, exponent);
    const float2 cosine_sine = rope_sincos_minimax_f32(angle);
    for (uint head = 0u; head < args.n_q_heads; ++head) {
        rope_neox_rotate_head(q, head, args.head_dim, half_rot, pair, cosine_sine);
    }
    for (uint head = 0u; head < args.n_k_heads; ++head) {
        rope_neox_rotate_head(k, head, args.head_dim, half_rot, pair, cosine_sine);
    }
}

struct rope_packed_args {
    uint  n_tokens;
    uint  n_heads;
    uint  head_dim;
    uint  n_rot;
    uint  start_position;
    float theta_base;
};

struct rope_packed_pair_args {
    uint  n_tokens;
    uint  n_q_heads;
    uint  n_k_heads;
    uint  head_dim;
    uint  n_rot;
    uint  start_position;
    float theta_base;
};

kernel void kernel_rope_neox_f32_packed_consecutive(
        constant rope_packed_args & args [[buffer(0)]],
        device       float * buf         [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
    const uint half_rot = args.n_rot / 2u;
    const uint pairs_per_token = args.n_heads * half_rot;
    const uint total_pairs = args.n_tokens * pairs_per_token;
    if (tid >= total_pairs) return;

    const uint tok = tid / pairs_per_token;
    const uint local = tid % pairs_per_token;
    const uint hi = local / half_rot;
    const uint i = local % half_rot;
    const uint position = args.start_position + tok;

    const float exponent = float(2u * i) / float(args.n_rot);
    const float freq = float(position) / pow(args.theta_base, exponent);
    const float c = cos(freq);
    const float s = sin(freq);

    const uint row_base = tok * args.n_heads * args.head_dim;
    const uint base = row_base + hi * args.head_dim;
    const float a = buf[base + i];
    const float b = buf[base + i + half_rot];
    buf[base + i]            = a * c - b * s;
    buf[base + i + half_rot] = a * s + b * c;
}

// Paired packed geometry keeps one thread per (token, head, pair), removing the
// second Q/K dispatch and sharing trig across the common head prefix.
kernel void kernel_rope_neox_pair_f32_packed_consecutive(
        constant rope_packed_pair_args & args [[buffer(0)]],
        device float * q                     [[buffer(1)]],
        device float * k                     [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    const uint half_rot = args.n_rot / 2u;
    const uint n_heads = max(args.n_q_heads, args.n_k_heads);
    const uint pairs_per_token = n_heads * half_rot;
    const uint total_pairs = args.n_tokens * pairs_per_token;
    if (tid >= total_pairs) return;

    const uint token = tid / pairs_per_token;
    const uint local = tid - token * pairs_per_token;
    const uint head = local / half_rot;
    const uint pair = local - head * half_rot;
    const uint position = args.start_position + token;
    const float exponent = float(2u * pair) / float(args.n_rot);
    const float angle = float(position) / pow(args.theta_base, exponent);
    const float2 cosine_sine = rope_sincos_native(angle);

    if (head < args.n_q_heads) {
        const uint q_offset = token * args.n_q_heads * args.head_dim;
        rope_neox_rotate_head(q + q_offset, head, args.head_dim, half_rot, pair, cosine_sine);
    }
    if (head < args.n_k_heads) {
        const uint k_offset = token * args.n_k_heads * args.head_dim;
        rope_neox_rotate_head(k + k_offset, head, args.head_dim, half_rot, pair, cosine_sine);
    }
}

// One lane per (token, pair) shares coefficients across every Q/K head.
kernel void kernel_rope_neox_pair_shared_f32_packed_consecutive(
        constant rope_packed_pair_args & args [[buffer(0)]],
        device float * q                     [[buffer(1)]],
        device float * k                     [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    const uint half_rot = args.n_rot / 2u;
    const uint total_pairs = args.n_tokens * half_rot;
    if (tid >= total_pairs) return;

    const uint token = tid / half_rot;
    const uint pair = tid - token * half_rot;
    const uint position = args.start_position + token;
    const float exponent = float(2u * pair) / float(args.n_rot);
    const float angle = float(position) / pow(args.theta_base, exponent);
    const float2 cosine_sine = rope_sincos_native(angle);
    device float * q_row = q + token * args.n_q_heads * args.head_dim;
    device float * k_row = k + token * args.n_k_heads * args.head_dim;
    for (uint head = 0u; head < args.n_q_heads; ++head) {
        rope_neox_rotate_head(q_row, head, args.head_dim, half_rot, pair, cosine_sine);
    }
    for (uint head = 0u; head < args.n_k_heads; ++head) {
        rope_neox_rotate_head(k_row, head, args.head_dim, half_rot, pair, cosine_sine);
    }
}

kernel void kernel_rope_neox_pair_shared_minimax_f32_packed_consecutive(
        constant rope_packed_pair_args & args [[buffer(0)]],
        device float * q                     [[buffer(1)]],
        device float * k                     [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    const uint half_rot = args.n_rot / 2u;
    const uint total_pairs = args.n_tokens * half_rot;
    if (tid >= total_pairs) return;

    const uint token = tid / half_rot;
    const uint pair = tid - token * half_rot;
    const uint position = args.start_position + token;
    const float exponent = float(2u * pair) / float(args.n_rot);
    const float angle = float(position) / pow(args.theta_base, exponent);
    const float2 cosine_sine = rope_sincos_minimax_f32(angle);
    device float * q_row = q + token * args.n_q_heads * args.head_dim;
    device float * k_row = k + token * args.n_k_heads * args.head_dim;
    for (uint head = 0u; head < args.n_q_heads; ++head) {
        rope_neox_rotate_head(q_row, head, args.head_dim, half_rot, pair, cosine_sine);
    }
    for (uint head = 0u; head < args.n_k_heads; ++head) {
        rope_neox_rotate_head(k_row, head, args.head_dim, half_rot, pair, cosine_sine);
    }
}
