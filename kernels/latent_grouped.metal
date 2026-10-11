// Native grouped latent projection: compressed [K,M,groups] weights and
// token-major F32 [K*groups,rows] -> [M*groups,rows]. No external scratch,
// no full weight expansion, and no half operand staging. Four simdgroups
// each own one output channel and reuse decoded weights across four tokens.
// Tail lanes/tokens/channels never read or write outside the logical view.
#include <metal_stdlib>
using namespace metal;
#include "native_dequant_f32.h"

struct latent_grouped_args {
    uint k;
    uint m;
    uint groups;
    uint rows;
    uint row_bytes;
    uint input_stride;
    uint output_stride;
};

// Each lane owns 16 consecutive K elements, so a 32-lane simdgroup
// advances 512 elements per iteration. Storage blocks are decoded in-place.
// ulong address arithmetic prevents byte offsets wrapping at 4 GiB.
template <ushort BYTES, ushort NL,
          void (*DEQ)(device const uchar *, short, thread float4x4 &)>
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_latent_grouped_native_f32(
        constant latent_grouped_args & args [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]]) {
    const uint channel = group.x * 4u + uint(simd);
    const uint token = group.y * 4u;
    const uint head = group.z;
    // Uniform across the simdgroup; no threadgroup barriers in this kernel.
    if (channel >= args.m || head >= args.groups || token >= args.rows) return;
    device const uchar * row = weights
        + (ulong(head) * args.m + channel) * args.row_bytes;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (ulong k = ulong(lane) * 16u; k < args.k; k += 512u) {
        float4x4 decoded;
        DEQ(row + (k / (16u * NL)) * BYTES, short((k / 16u) % NL), decoded);
        QT_FOR_UNROLL (uint t = 0; t < 4; ++t) {
            if (token + t < args.rows) {
                device const float * x = input
                    + ulong(token + t) * args.input_stride + ulong(head) * args.k + k;
                QT_FOR_UNROLL (uint i = 0; i < 16; ++i) {
                    sums[t] = fma(decoded[i / 4][i % 4], x[i], sums[t]);
                }
            }
        }
    }
    QT_FOR_UNROLL (uint t = 0; t < 4; ++t) {
        const float value = simd_sum(sums[t]);
        if (lane == 0 && token + t < args.rows) {
            output[ulong(token + t) * args.output_stride + ulong(head) * args.m + channel] = value;
        }
    }
}

typedef decltype(kernel_latent_grouped_native_f32<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>) latent_grouped_native_t;
template [[host_name("kernel_latent_grouped_q2_0_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_Q2_0_BYTES, QT_Q2_0_NL, latent_dequantize_q2_0>;
template [[host_name("kernel_latent_grouped_q2_K_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_Q2_K_BYTES, QT_Q2_K_NL, latent_dequantize_q2_K>;
template [[host_name("kernel_latent_grouped_q3_K_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K>;
template [[host_name("kernel_latent_grouped_q4_K_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_Q4_K_BYTES, QT_Q4_K_NL, latent_dequantize_q4_K>;
template [[host_name("kernel_latent_grouped_q5_K_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_Q5_K_BYTES, QT_Q5_K_NL, latent_dequantize_q5_K>;
template [[host_name("kernel_latent_grouped_q6_K_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>;
template [[host_name("kernel_latent_grouped_q8_0_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0>;
template [[host_name("kernel_latent_grouped_q4_0_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_Q4_0_BYTES, QT_Q4_0_NL, latent_dequantize_q4_0>;
template [[host_name("kernel_latent_grouped_q4_1_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_Q4_1_BYTES, QT_Q4_1_NL, latent_dequantize_q4_1>;
template [[host_name("kernel_latent_grouped_iq4_nl_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_IQ4_NL_BYTES, QT_IQ4_NL_NL, latent_dequantize_iq4_nl>;
template [[host_name("kernel_latent_grouped_iq2_s_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_IQ2_S_BYTES, QT_IQ2_S_NL, latent_dequantize_iq2_s>;
template [[host_name("kernel_latent_grouped_iq3_xxs_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_IQ3_XXS_BYTES, QT_IQ3_XXS_NL, latent_dequantize_iq3_xxs>;
template [[host_name("kernel_latent_grouped_iq3_s_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_IQ3_S_BYTES, QT_IQ3_S_NL, latent_dequantize_iq3_s>;
template [[host_name("kernel_latent_grouped_iq4_xs_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_IQ4_XS_BYTES, QT_IQ4_XS_NL, latent_dequantize_iq4_xs>;
template [[host_name("kernel_latent_grouped_f32_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_F32_BYTES, QT_F32_NL, latent_dequantize_f32>;
template [[host_name("kernel_latent_grouped_f16_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_F16_BYTES, QT_F16_NL, latent_dequantize_f16>;
template [[host_name("kernel_latent_grouped_bf16_f32")]] kernel latent_grouped_native_t kernel_latent_grouped_native_f32<QT_BF16_BYTES, QT_BF16_NL, latent_dequantize_bf16>;
