#include <metal_stdlib>
using namespace metal;

struct muse_glimmer_rope_args {
    uint q_pair_count;
    uint pair_count;
    uint head_dim;
    uint position;
    float theta;
};

kernel void kernel_muse_glimmer_rope_adjacent_pair_in_place_f32(
        constant muse_glimmer_rope_args & args [[buffer(0)]],
        device float * q [[buffer(1)]],
        device float * k [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.pair_count) return;
    const bool is_q = index < args.q_pair_count;
    const uint local_index = is_q ? index : index - args.q_pair_count;
    const uint pairs_per_head = args.head_dim / 2u;
    const uint head = local_index / pairs_per_head;
    const uint pair = local_index - head * pairs_per_head;
    const uint relative = pair * 2u;
    const uint first_index = head * args.head_dim + relative;
    const uint second_index = first_index + 1u;
    device float * values = is_q ? q : k;

    const float angle = float(args.position)
        * pow(args.theta, -float(relative) / float(args.head_dim));
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const float first = values[first_index];
    const float second = values[second_index];
    values[first_index] = first * cosine - second * sine;
    values[second_index] = first * sine + second * cosine;
}

struct muse_glimmer_softcap_args {
    uint n;
    float scale;
    float cap;
};

kernel void kernel_muse_glimmer_logit_softcap_f32(
        constant muse_glimmer_softcap_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device float * output [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.n) return;
    output[index] = args.cap * tanh(input[index] * args.scale / args.cap);
}
