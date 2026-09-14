#include <metal_stdlib>
using namespace metal;

struct qsa_rms_args {
    uint head_count;
    uint head_dim;
    uint rotary_dim;
    uint position;
    float theta;
    float eps;
};

static inline float qsa_rope_exact(
        device const float * row, device const float * weight,
        uint lane, uint rotary_dim, uint position, float theta) {
    if (lane >= rotary_dim) return row[lane] * weight[lane];
    const uint half_dim = rotary_dim / 2u;
    const uint pair = lane % half_dim;
    const float frequency = pow(theta, -2.0f * float(pair) / float(rotary_dim));
    const float angle = float(position) * frequency;
    const float cosine = cos(angle);
    const float sine = sin(angle);
    if (lane < half_dim) {
        return row[pair] * weight[pair] * cosine
            - row[pair + half_dim] * weight[pair + half_dim] * sine;
    }
    return row[pair + half_dim] * weight[pair + half_dim] * cosine
        + row[pair] * weight[pair] * sine;
}

template <bool gated>
static inline void qsa_rms_broadcast(
        constant qsa_rms_args & args,
        device const float * input, device const float * weight,
        device float * output, device float * gate, uint head, ushort lane) {
    if (head >= args.head_count) return;
    device const float * row = input + (ulong)head * args.head_dim * (gated ? 2u : 1u);
    float norm = 0.0f;
    if (lane == 0u) {
        float sum_square = 0.0f;
        for (uint inner = 0u; inner < args.head_dim; ++inner) {
            const float value = row[inner];
            sum_square += value * value;
        }
        norm = rsqrt(sum_square / float(args.head_dim) + args.eps);
    }
    norm = simd_broadcast(norm, 0u);
    for (uint d = uint(lane); d < args.head_dim; d += 32u) {
        const ulong index = (ulong)head * args.head_dim + d;
        output[index] = qsa_rope_exact(row, weight, d, args.rotary_dim, args.position, args.theta) * norm;
        if (gated) gate[index] = row[args.head_dim + d];
    }
}

kernel void kernel_qwen4exp_qsa_rms_broadcast_f32(
        constant qsa_rms_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]], device const float * weight [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint head [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    qsa_rms_broadcast<false>(args, input, weight, output, output, head, lane);
}

kernel void kernel_qwen4exp_qsa_qgate_rms_broadcast_f32(
        constant qsa_rms_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]], device const float * weight [[buffer(2)]],
        device float * output [[buffer(3)]], device float * gate [[buffer(4)]],
        uint head [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    qsa_rms_broadcast<true>(args, input, weight, output, gate, head, lane);
}
