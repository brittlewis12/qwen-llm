// Decode-only mHC candidate. RMS is encoded before these kernels; input
// is normalized F32, never a raw residual dot scaled after reduction.
// No expanded weights; partial dots reuse the consumed unit-RMS scratch.
#include <metal_stdlib>
using namespace metal;
#include "native_dequant_f32.h"

constant uint DS4_CONNECTIONS = 4;
constant uint DS4_PARAMETERS = 24;
constant uint DS4_SINKHORN_ITERATIONS = 20;

struct mhc4_pre_native_args {
    uint hidden;
    uint chunks;
    uint row_bytes;
    float hc_eps;
    float norm_eps;
};

// Four simdgroups own four output channels for one 256-value K chunk.
// Six groups cover the 24 mixes. Each of lanes 0..15 decodes sixteen F32
// weights, with zero contributions from inactive lanes and tail chunks.
// Shared decoders consume native storage directly; no half staging.
template <ushort BYTES, ushort NL,
          void (*DEQ)(device const uchar *, short, thread float4x4 &)>
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_mhc4_pre_native_partial(
        constant mhc4_pre_native_args & args [[buffer(0)]],
        device const float * normalized [[buffer(1)]],
        device const uchar * weights [[buffer(2)]],
        device float * dots [[buffer(3)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint m = group.y * 4u + sg;
    const uint k = group.x * 256u + uint(lane) * 16u;
    float sum = 0.0f;
    if (lane < 16 && k < args.hidden * 4u) {
        device const uchar * row = weights + ulong(m) * args.row_bytes;
        float4x4 decoded;
        DEQ(row + (ulong(k) / (16u * NL)) * BYTES, short((k / 16u) % NL), decoded);
        QT_FOR_UNROLL (uint i = 0; i < 16; ++i) {
            sum = fma(decoded[i / 4][i % 4], normalized[k + i], sum);
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) dots[group.x * 24u + m] = sum;
}

typedef decltype(kernel_mhc4_pre_native_partial<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>) mhc4_pre_native_partial_t;
template [[host_name("kernel_mhc4_pre_native_partial_q2_0_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_Q2_0_BYTES, QT_Q2_0_NL, latent_dequantize_q2_0>;
template [[host_name("kernel_mhc4_pre_native_partial_q2_K_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_Q2_K_BYTES, QT_Q2_K_NL, latent_dequantize_q2_K>;
template [[host_name("kernel_mhc4_pre_native_partial_q3_K_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K>;
template [[host_name("kernel_mhc4_pre_native_partial_q4_K_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_Q4_K_BYTES, QT_Q4_K_NL, latent_dequantize_q4_K>;
template [[host_name("kernel_mhc4_pre_native_partial_q5_K_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_Q5_K_BYTES, QT_Q5_K_NL, latent_dequantize_q5_K>;
template [[host_name("kernel_mhc4_pre_native_partial_q6_K_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>;
template [[host_name("kernel_mhc4_pre_native_partial_q8_0_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0>;
template [[host_name("kernel_mhc4_pre_native_partial_q4_0_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_Q4_0_BYTES, QT_Q4_0_NL, latent_dequantize_q4_0>;
template [[host_name("kernel_mhc4_pre_native_partial_q4_1_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_Q4_1_BYTES, QT_Q4_1_NL, latent_dequantize_q4_1>;
template [[host_name("kernel_mhc4_pre_native_partial_iq4_nl_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_IQ4_NL_BYTES, QT_IQ4_NL_NL, latent_dequantize_iq4_nl>;
template [[host_name("kernel_mhc4_pre_native_partial_iq2_s_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_IQ2_S_BYTES, QT_IQ2_S_NL, latent_dequantize_iq2_s>;
template [[host_name("kernel_mhc4_pre_native_partial_iq3_xxs_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_IQ3_XXS_BYTES, QT_IQ3_XXS_NL, latent_dequantize_iq3_xxs>;
template [[host_name("kernel_mhc4_pre_native_partial_iq3_s_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_IQ3_S_BYTES, QT_IQ3_S_NL, latent_dequantize_iq3_s>;
template [[host_name("kernel_mhc4_pre_native_partial_iq4_xs_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_IQ4_XS_BYTES, QT_IQ4_XS_NL, latent_dequantize_iq4_xs>;
template [[host_name("kernel_mhc4_pre_native_partial_f32_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_F32_BYTES, QT_F32_NL, latent_dequantize_f32>;
template [[host_name("kernel_mhc4_pre_native_partial_f16_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_F16_BYTES, QT_F16_NL, latent_dequantize_f16>;
template [[host_name("kernel_mhc4_pre_native_partial_bf16_f32")]] kernel mhc4_pre_native_partial_t kernel_mhc4_pre_native_partial<QT_BF16_BYTES, QT_BF16_NL, latent_dequantize_bf16>;

// Controls preserve the existing fused-pre expressions, without editing Q8.
static inline void mhc4_native_controls_registers(
        float eps,
        device const float * mix,
        device const float * scale,
        device const float * base,
        device float * pre,
        device float * post,
        device float * combination) {
    for (uint stream = 0; stream < DS4_CONNECTIONS; ++stream) {
        pre[stream] = 1.0f / (1.0f + exp(-(mix[stream] * scale[0] + base[stream]))) + eps;
        post[stream] = 2.0f / (1.0f + exp(-(mix[4 + stream] * scale[1] + base[4 + stream])));
    }

    float c[DS4_CONNECTIONS * DS4_CONNECTIONS];
    for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
        float row_max = -INFINITY;
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            const uint index = source * DS4_CONNECTIONS + destination;
            c[index] = mix[8 + index] * scale[2] + base[8 + index];
            row_max = max(row_max, c[index]);
        }
        float sum = 0.0f;
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            const uint index = source * DS4_CONNECTIONS + destination;
            c[index] = exp(c[index] - row_max);
            sum += c[index];
        }
        const float inverse = 1.0f / sum;
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            const uint index = source * DS4_CONNECTIONS + destination;
            c[index] = c[index] * inverse + eps;
        }
    }

    for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
        float sum = 0.0f;
        for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
            sum += c[source * DS4_CONNECTIONS + destination];
        }
        const float inverse = 1.0f / (sum + eps);
        for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
            c[source * DS4_CONNECTIONS + destination] *= inverse;
        }
    }
    for (uint iteration = 1; iteration < DS4_SINKHORN_ITERATIONS; ++iteration) {
        for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
            float sum = 0.0f;
            for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
                sum += c[source * DS4_CONNECTIONS + destination];
            }
            const float inverse = 1.0f / (sum + eps);
            for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
                c[source * DS4_CONNECTIONS + destination] *= inverse;
            }
        }
        for (uint destination = 0; destination < DS4_CONNECTIONS; ++destination) {
            float sum = 0.0f;
            for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
                sum += c[source * DS4_CONNECTIONS + destination];
            }
            const float inverse = 1.0f / (sum + eps);
            for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
                c[source * DS4_CONNECTIONS + destination] *= inverse;
            }
        }
    }
    for (uint index = 0; index < DS4_CONNECTIONS * DS4_CONNECTIONS; ++index) {
        combination[index] = c[index];
    }
}

// Finish consumes normalized dots directly; no second RMS factor. The collapse
// and block RMS keep the existing 1024-thread fused-pre arithmetic lineage.
[[max_total_threads_per_threadgroup(1024)]]
kernel void kernel_mhc4_pre_native_finish(
        constant mhc4_pre_native_args & args [[buffer(0)]],
        device const float * residual [[buffer(1)]],
        device const float * partial_dots [[buffer(2)]],
        device const float * scale [[buffer(3)]],
        device const float * base [[buffer(4)]],
        device const float * norm_weight [[buffer(5)]],
        device float * mixes [[buffer(6)]],
        device float * pre [[buffer(7)]],
        device float * post [[buffer(8)]],
        device float * combination [[buffer(9)]],
        device float * collapsed [[buffer(10)]],
        device float * normed [[buffer(11)]],
        threadgroup float * shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    threadgroup float * reduced = shared;       // 24 normalized-activation dots
    threadgroup float * gates = shared + 32;    // pre[4]
    threadgroup float * partial = shared + 64;  // one sum per simdgroup
    const uint width = args.hidden * DS4_CONNECTIONS;
    if (sg < DS4_PARAMETERS) {
        float sum = 0.0f;
        for (uint c = lane; c < args.chunks; c += 32u) {
            const uint unit = row * args.chunks + c;
            sum += partial_dots[unit * DS4_PARAMETERS + sg];
        }
        sum = simd_sum(sum);
        if (lane == 0) reduced[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        device float * m = mixes + row * DS4_PARAMETERS;
        for (uint i = 0; i < DS4_PARAMETERS; ++i) {
            m[i] = reduced[i];
        }
        device float * row_pre = pre + row * DS4_CONNECTIONS;
        mhc4_native_controls_registers(
            args.hc_eps,
            m,
            scale,
            base,
            row_pre,
            post + row * DS4_CONNECTIONS,
            combination + row * DS4_CONNECTIONS * DS4_CONNECTIONS);
        for (uint s = 0; s < DS4_CONNECTIONS; ++s) {
            gates[s] = row_pre[s];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float * r = residual + row * width;
    device float * out = collapsed + row * args.hidden;
    float sumsq = 0.0f;
    for (uint i = tid; i < args.hidden; i += ntg) {
        float value = 0.0f;
        for (uint source = 0; source < DS4_CONNECTIONS; ++source) {
            value += r[source * args.hidden + i] * gates[source];
        }
        out[i] = value;
        sumsq += value * value;
    }
    sumsq = simd_sum(sumsq);
    if (lane == 0) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = (lane < (ntg + 31u) / 32u) ? partial[lane] : 0.0f;
    sumsq = simd_sum(sumsq);
    const float norm_scale = rsqrt(sumsq / float(args.hidden) + args.norm_eps);
    device float * y = normed + row * args.hidden;
    for (uint i = tid; i < args.hidden; i += ntg) {
        y[i] = (out[i] * norm_scale) * norm_weight[i];
    }
}
