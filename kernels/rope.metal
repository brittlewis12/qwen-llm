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
