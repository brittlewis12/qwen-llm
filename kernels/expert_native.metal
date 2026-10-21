// GPU-indexed, F32-operand baseline. Four simdgroups own four output rows.
// Gate writes the existing inner buffer; up consumes it in-place. No scratch.
#include <metal_stdlib>
using namespace metal;
#include "native_dequant_f32.h"
#include "iq2_xs_grid.metalh"
#include "iq2_xxs_decode.metalh"
#include "mxfp4_decode.metalh"

inline void expert_dequantize_iq2_xxs(device const uchar * blk, short il,
                                     thread float4x4 & reg) {
    iq2_xxs_dequant16(blk, uint(il), reg);
}

// IQ2_XS uses the same packed grid/sign/scale interpretation as the existing
// all-slot decoder, widened directly to F32 (never through a half register).
inline void expert_dequantize_iq2_xs(device const uchar * blk, short il,
                                    thread float4x4 & reg) {
    const float d = float(*((device const half *)blk));
    device const ushort * qs = (device const ushort *)(blk + 2);
    device const uchar * scales = blk + 66;
    const uint ib32 = uint(il) >> 1;
    const uint ih = uint(il) & 1u;
    const float dl = d * (0.5f + float((uint(scales[ib32]) >> (4u * ih)) & 15u)) * 0.25f;
    QT_FOR_UNROLL (uint i = 0; i < 16; ++i) {
        const uint packed = uint(qs[4u * ib32 + 2u * ih + (i >> 3)]);
        constant uchar * grid = (constant uchar *)(mv_iq2xs_grid + (packed & 511u));
        const uint signs = uint(qt_ksigns_iq2xs[packed >> 9]);
        const float sign = (signs & uint(qt_kmask_iq2xs[i & 7u])) ? -1.0f : 1.0f;
        reg[i / 4][i % 4] = dl * float(grid[i & 7u]) * sign;
    }
}

struct expert_native_args {
    ulong row_bytes;
    uint n_in;
    uint n_out;
    uint experts;
    uint top_k;
    uint mode; // 0: gate; 1: up + in-place clamped SwiGLU; 2: slotwise down
    float clamp;
};

// DENSE: 0=quantized, 1=F32, 2=F16, 3=BF16. Dense tails permit any K;
// quantized K is an integral number of storage blocks, checked by the host.
template <ushort BYTES, ushort NL,
          void (*DEQ)(device const uchar *, short, thread float4x4 &),
          ushort DENSE = 0, bool MXFP4 = false>
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_expert_native_f32(
        constant expert_native_args & args [[buffer(0)]],
        device const uchar * bank [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device const int * ids [[buffer(3)]],
        device const int * status [[buffer(4)]],
        device float * output [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]]) {
    const uint channel = group.x * 4u + uint(simd);
    const uint slot = group.y;
    const uint token = group.z;
    if (channel >= args.n_out || slot >= args.top_k) return;
    const ulong selected = ulong(token) * args.top_k + slot;
    const ulong out_index = selected * args.n_out + channel;
    // Do not even read ids until READY. Both exits precede bank/input reads
    // and the up epilogue's read of the previous gate output.
    if (status[token] != 1) {
        if (lane == 0) output[out_index] = 0.0f;
        return;
    }
    const int expert = ids[selected];
    if (expert < 0 || uint(expert) >= args.experts) {
        if (lane == 0) output[out_index] = 0.0f;
        return;
    }
    device const uchar * row = bank
        + (ulong(expert) * args.n_out + channel) * args.row_bytes;
    device const float * x = input + (args.mode == 2 ? selected : ulong(token)) * args.n_in;
    float sum = 0.0f;
    for (ulong k = ulong(lane) * 16u; k < args.n_in; k += 512u) {
        float4x4 decoded;
        if (DENSE != 0 && k + 16u > args.n_in) {
            // Last dense fragment may end at the physical buffer boundary.
            for (uint i = 0; i < uint(args.n_in - k); ++i) {
                float w;
                if (DENSE == 1) w = ((device const float *)row)[k + i];
                else if (DENSE == 2) w = float(((device const half *)row)[k + i]);
                else w = as_type<float>(uint(((device const ushort *)row)[k + i]) << 16);
                sum = fma(w, x[k + i], sum);
            }
        } else {
            device const uchar * block = row + (k / (16u * NL)) * BYTES;
            const short il = short((k / 16u) % NL);
            if (MXFP4 && block[0] <= 1u) {
                // Metal flushes subnormal operands. Move an exact power of two
                // from x to the decoded weight so GGML's subnormal F32 weights
                // still produce normal products with large F32 activations.
                const float scale = as_type<float>((uint(block[0]) + 23u) << 23u);
                QT_FOR_UNROLL (uint i = 0; i < 16; ++i) {
                    const uint code = (uint(block[1 + i]) >> (4u * uint(il))) & 15u;
                    const float w = expert_mxfp4_values[code] * scale;
                    sum = fma(w, ldexp(x[k + i], -24), sum);
                }
                continue;
            }
            DEQ(block, il, decoded);
            QT_FOR_UNROLL (uint i = 0; i < 16; ++i) {
                sum = fma(decoded[i / 4][i % 4], x[k + i], sum);
            }
        }
    }
    const float value = simd_sum(sum);
    if (lane == 0) {
        if (args.mode == 1) {
            const float g = min(output[out_index], args.clamp);
            const float u = clamp(value, -args.clamp, args.clamp);
            output[out_index] = (g / (1.0f + exp(-g))) * u;
        } else {
            output[out_index] = value;
        }
    }
}

typedef decltype(kernel_expert_native_f32<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>) expert_native_t;
template [[host_name("kernel_expert_native_q2_0_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_Q2_0_BYTES, QT_Q2_0_NL, latent_dequantize_q2_0>;
template [[host_name("kernel_expert_native_q2_K_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_Q2_K_BYTES, QT_Q2_K_NL, latent_dequantize_q2_K>;
template [[host_name("kernel_expert_native_q3_K_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K>;
template [[host_name("kernel_expert_native_q4_K_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_Q4_K_BYTES, QT_Q4_K_NL, latent_dequantize_q4_K>;
template [[host_name("kernel_expert_native_q5_K_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_Q5_K_BYTES, QT_Q5_K_NL, latent_dequantize_q5_K>;
template [[host_name("kernel_expert_native_q6_K_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>;
template [[host_name("kernel_expert_native_q8_0_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0>;
template [[host_name("kernel_expert_native_q4_0_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_Q4_0_BYTES, QT_Q4_0_NL, latent_dequantize_q4_0>;
template [[host_name("kernel_expert_native_q4_1_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_Q4_1_BYTES, QT_Q4_1_NL, latent_dequantize_q4_1>;
template [[host_name("kernel_expert_native_iq4_nl_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_IQ4_NL_BYTES, QT_IQ4_NL_NL, latent_dequantize_iq4_nl>;
template [[host_name("kernel_expert_native_iq2_s_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_IQ2_S_BYTES, QT_IQ2_S_NL, latent_dequantize_iq2_s>;
template [[host_name("kernel_expert_native_iq3_xxs_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_IQ3_XXS_BYTES, QT_IQ3_XXS_NL, latent_dequantize_iq3_xxs>;
template [[host_name("kernel_expert_native_iq3_s_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_IQ3_S_BYTES, QT_IQ3_S_NL, latent_dequantize_iq3_s>;
template [[host_name("kernel_expert_native_iq4_xs_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_IQ4_XS_BYTES, QT_IQ4_XS_NL, latent_dequantize_iq4_xs>;
template [[host_name("kernel_expert_native_f32_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_F32_BYTES, QT_F32_NL, latent_dequantize_f32, 1>;
template [[host_name("kernel_expert_native_f16_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_F16_BYTES, QT_F16_NL, latent_dequantize_f16, 2>;
template [[host_name("kernel_expert_native_bf16_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_BF16_BYTES, QT_BF16_NL, latent_dequantize_bf16, 3>;
template [[host_name("kernel_expert_native_iq2_xs_f32")]] kernel expert_native_t kernel_expert_native_f32<74, 16, expert_dequantize_iq2_xs>;
template [[host_name("kernel_expert_native_iq2_xxs_f32")]] kernel expert_native_t kernel_expert_native_f32<66, 16, expert_dequantize_iq2_xxs>;
template [[host_name("kernel_expert_native_mxfp4_f32")]] kernel expert_native_t kernel_expert_native_f32<17, 2, expert_dequantize_mxfp4, 0, true>;

struct expert_native_gate_up_weight_args {
    ulong gate_row_bytes;
    ulong up_row_bytes;
    uint n_in;
    uint n_out;
    uint experts;
    uint top_k;
    float clamp;
};

struct expert_native_down_sum_args {
    ulong row_bytes;
    uint n_in;
    uint n_out;
    uint experts;
    uint top_k;
};

// Decode fusion: gate and up are decoded in one loop so each activation load
// feeds both banks, then one epilogue applies the same upper-only gate clamp,
// up clamp, SwiGLU and route weight as the separate up and weight passes.
// The lane partition, per-lane FMA order and SIMD reduction are unchanged.
// MXFP4 is intentionally not instantiated: its subnormal rescaling moves the
// exponent per bank, so it keeps the native two-pass path.
template <ushort BYTES, ushort NL,
          void (*DEQ)(device const uchar *, short, thread float4x4 &),
          ushort DENSE = 0>
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_expert_native_gate_up_weight_f32(
        constant expert_native_gate_up_weight_args & args [[buffer(0)]],
        device const uchar * gate_bank [[buffer(1)]],
        device const uchar * up_bank [[buffer(2)]],
        device const float * input [[buffer(3)]],
        device const int * ids [[buffer(4)]],
        device const int * status [[buffer(5)]],
        device const float * weights [[buffer(6)]],
        device float * output [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]]) {
    const uint channel = group.x * 4u + uint(simd);
    const uint slot = group.y;
    const uint token = group.z;
    if (channel >= args.n_out || slot >= args.top_k) return;
    const ulong selected = ulong(token) * args.top_k + slot;
    const ulong out_index = selected * args.n_out + channel;
    if (status[token] != 1) {
        if (lane == 0) output[out_index] = 0.0f;
        return;
    }
    const int expert = ids[selected];
    if (expert < 0 || uint(expert) >= args.experts) {
        // The weight pass multiplies the zeroed native result by the slot
        // weight. Rebuild that signed zero from the weight bit directly;
        // fast math would fold 0.0f * weight to positive zero.
        if (lane == 0) {
            output[out_index] =
                as_type<float>(as_type<uint>(weights[selected]) & 0x80000000u);
        }
        return;
    }
    device const uchar * grow = gate_bank
        + (ulong(expert) * args.n_out + channel) * args.gate_row_bytes;
    device const uchar * urow = up_bank
        + (ulong(expert) * args.n_out + channel) * args.up_row_bytes;
    device const float * x = input + ulong(token) * args.n_in;
    float gsum = 0.0f;
    float usum = 0.0f;
    for (ulong k = ulong(lane) * 16u; k < args.n_in; k += 512u) {
        float4x4 gdec;
        float4x4 udec;
        if (DENSE != 0 && k + 16u > args.n_in) {
            for (uint i = 0; i < uint(args.n_in - k); ++i) {
                float gw;
                float uw;
                if (DENSE == 1) {
                    gw = ((device const float *)grow)[k + i];
                    uw = ((device const float *)urow)[k + i];
                } else if (DENSE == 2) {
                    gw = float(((device const half *)grow)[k + i]);
                    uw = float(((device const half *)urow)[k + i]);
                } else {
                    gw = as_type<float>(uint(((device const ushort *)grow)[k + i]) << 16);
                    uw = as_type<float>(uint(((device const ushort *)urow)[k + i]) << 16);
                }
                const float xv = x[k + i];
                gsum = fma(gw, xv, gsum);
                usum = fma(uw, xv, usum);
            }
        } else {
            device const uchar * gblock = grow + (k / (16u * NL)) * BYTES;
            device const uchar * ublock = urow + (k / (16u * NL)) * BYTES;
            const short il = short((k / 16u) % NL);
            DEQ(gblock, il, gdec);
            DEQ(ublock, il, udec);
            QT_FOR_UNROLL (uint i = 0; i < 16; ++i) {
                const float xv = x[k + i];
                gsum = fma(gdec[i / 4][i % 4], xv, gsum);
                usum = fma(udec[i / 4][i % 4], xv, usum);
            }
        }
    }
    const float g = simd_sum(gsum);
    const float u = simd_sum(usum);
    if (lane == 0) {
        const float gl = min(g, args.clamp);
        const float ul = clamp(u, -args.clamp, args.clamp);
        // The native up pass stores this product and the weight pass reloads
        // it. Keep that intermediate rounding; fast math would otherwise
        // reassociate the two F32 multiplies.
        volatile float swiglu = (gl / (1.0f + exp(-gl))) * ul;
        output[out_index] = swiglu * weights[selected];
    }
}

// Decode fusion: each threadgroup sums the top-k down projections in slot
// order and adds the shared branch, matching the separate ordered-sum kernel.
template <ushort BYTES, ushort NL,
          void (*DEQ)(device const uchar *, short, thread float4x4 &),
          ushort DENSE = 0>
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_expert_native_down_sum_f32(
        constant expert_native_down_sum_args & args [[buffer(0)]],
        device const uchar * bank [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device const int * ids [[buffer(3)]],
        device const int * status [[buffer(4)]],
        device const float * shared [[buffer(5)]],
        device float * output [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]]) {
    const uint channel = group.x * 4u + uint(simd);
    const uint token = group.z;
    if (channel >= args.n_out) return;
    const ulong out_index = ulong(token) * args.n_out + channel;
    if (status[token] != 1) {
        if (lane == 0) output[out_index] = shared[out_index];
        return;
    }
    float total = 0.0f;
    for (uint slot = 0; slot < args.top_k; ++slot) {
        const ulong selected = ulong(token) * args.top_k + slot;
        const int expert = ids[selected];
        if (expert < 0 || uint(expert) >= args.experts) continue;
        device const uchar * row = bank
            + (ulong(expert) * args.n_out + channel) * args.row_bytes;
        device const float * x = input + selected * args.n_in;
        float sum = 0.0f;
        for (ulong k = ulong(lane) * 16u; k < args.n_in; k += 512u) {
            float4x4 decoded;
            if (DENSE != 0 && k + 16u > args.n_in) {
                for (uint i = 0; i < uint(args.n_in - k); ++i) {
                    float w;
                    if (DENSE == 1) w = ((device const float *)row)[k + i];
                    else if (DENSE == 2) w = float(((device const half *)row)[k + i]);
                    else w = as_type<float>(uint(((device const ushort *)row)[k + i]) << 16);
                    sum = fma(w, x[k + i], sum);
                }
            } else {
                device const uchar * block = row + (k / (16u * NL)) * BYTES;
                const short il = short((k / 16u) % NL);
                DEQ(block, il, decoded);
                QT_FOR_UNROLL (uint i = 0; i < 16; ++i) {
                    sum = fma(decoded[i / 4][i % 4], x[k + i], sum);
                }
            }
        }
        total += simd_sum(sum);
    }
    if (lane == 0) output[out_index] = total + shared[out_index];
}

typedef decltype(kernel_expert_native_gate_up_weight_f32<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>) expert_native_gate_up_weight_t;
template [[host_name("kernel_expert_native_gate_up_weight_q2_0_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_Q2_0_BYTES, QT_Q2_0_NL, latent_dequantize_q2_0>;
template [[host_name("kernel_expert_native_gate_up_weight_q2_K_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_Q2_K_BYTES, QT_Q2_K_NL, latent_dequantize_q2_K>;
template [[host_name("kernel_expert_native_gate_up_weight_q3_K_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K>;
template [[host_name("kernel_expert_native_gate_up_weight_q4_K_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_Q4_K_BYTES, QT_Q4_K_NL, latent_dequantize_q4_K>;
template [[host_name("kernel_expert_native_gate_up_weight_q5_K_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_Q5_K_BYTES, QT_Q5_K_NL, latent_dequantize_q5_K>;
template [[host_name("kernel_expert_native_gate_up_weight_q6_K_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>;
template [[host_name("kernel_expert_native_gate_up_weight_q8_0_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0>;
template [[host_name("kernel_expert_native_gate_up_weight_q4_0_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_Q4_0_BYTES, QT_Q4_0_NL, latent_dequantize_q4_0>;
template [[host_name("kernel_expert_native_gate_up_weight_q4_1_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_Q4_1_BYTES, QT_Q4_1_NL, latent_dequantize_q4_1>;
template [[host_name("kernel_expert_native_gate_up_weight_iq4_nl_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_IQ4_NL_BYTES, QT_IQ4_NL_NL, latent_dequantize_iq4_nl>;
template [[host_name("kernel_expert_native_gate_up_weight_iq2_s_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_IQ2_S_BYTES, QT_IQ2_S_NL, latent_dequantize_iq2_s>;
template [[host_name("kernel_expert_native_gate_up_weight_iq3_xxs_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_IQ3_XXS_BYTES, QT_IQ3_XXS_NL, latent_dequantize_iq3_xxs>;
template [[host_name("kernel_expert_native_gate_up_weight_iq3_s_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_IQ3_S_BYTES, QT_IQ3_S_NL, latent_dequantize_iq3_s>;
template [[host_name("kernel_expert_native_gate_up_weight_iq4_xs_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_IQ4_XS_BYTES, QT_IQ4_XS_NL, latent_dequantize_iq4_xs>;
template [[host_name("kernel_expert_native_gate_up_weight_f32_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_F32_BYTES, QT_F32_NL, latent_dequantize_f32, 1>;
template [[host_name("kernel_expert_native_gate_up_weight_f16_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_F16_BYTES, QT_F16_NL, latent_dequantize_f16, 2>;
template [[host_name("kernel_expert_native_gate_up_weight_bf16_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<QT_BF16_BYTES, QT_BF16_NL, latent_dequantize_bf16, 3>;
template [[host_name("kernel_expert_native_gate_up_weight_iq2_xs_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<74, 16, expert_dequantize_iq2_xs>;
template [[host_name("kernel_expert_native_gate_up_weight_iq2_xxs_f32")]] kernel expert_native_gate_up_weight_t kernel_expert_native_gate_up_weight_f32<66, 16, expert_dequantize_iq2_xxs>;

typedef decltype(kernel_expert_native_down_sum_f32<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>) expert_native_down_sum_t;
template [[host_name("kernel_expert_native_down_sum_q2_0_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_Q2_0_BYTES, QT_Q2_0_NL, latent_dequantize_q2_0>;
template [[host_name("kernel_expert_native_down_sum_q2_K_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_Q2_K_BYTES, QT_Q2_K_NL, latent_dequantize_q2_K>;
template [[host_name("kernel_expert_native_down_sum_q3_K_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K>;
template [[host_name("kernel_expert_native_down_sum_q4_K_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_Q4_K_BYTES, QT_Q4_K_NL, latent_dequantize_q4_K>;
template [[host_name("kernel_expert_native_down_sum_q5_K_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_Q5_K_BYTES, QT_Q5_K_NL, latent_dequantize_q5_K>;
template [[host_name("kernel_expert_native_down_sum_q6_K_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>;
template [[host_name("kernel_expert_native_down_sum_q8_0_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0>;
template [[host_name("kernel_expert_native_down_sum_q4_0_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_Q4_0_BYTES, QT_Q4_0_NL, latent_dequantize_q4_0>;
template [[host_name("kernel_expert_native_down_sum_q4_1_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_Q4_1_BYTES, QT_Q4_1_NL, latent_dequantize_q4_1>;
template [[host_name("kernel_expert_native_down_sum_iq4_nl_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_IQ4_NL_BYTES, QT_IQ4_NL_NL, latent_dequantize_iq4_nl>;
template [[host_name("kernel_expert_native_down_sum_iq2_s_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_IQ2_S_BYTES, QT_IQ2_S_NL, latent_dequantize_iq2_s>;
template [[host_name("kernel_expert_native_down_sum_iq3_xxs_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_IQ3_XXS_BYTES, QT_IQ3_XXS_NL, latent_dequantize_iq3_xxs>;
template [[host_name("kernel_expert_native_down_sum_iq3_s_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_IQ3_S_BYTES, QT_IQ3_S_NL, latent_dequantize_iq3_s>;
template [[host_name("kernel_expert_native_down_sum_iq4_xs_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_IQ4_XS_BYTES, QT_IQ4_XS_NL, latent_dequantize_iq4_xs>;
template [[host_name("kernel_expert_native_down_sum_f32_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_F32_BYTES, QT_F32_NL, latent_dequantize_f32, 1>;
template [[host_name("kernel_expert_native_down_sum_f16_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_F16_BYTES, QT_F16_NL, latent_dequantize_f16, 2>;
template [[host_name("kernel_expert_native_down_sum_bf16_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<QT_BF16_BYTES, QT_BF16_NL, latent_dequantize_bf16, 3>;
template [[host_name("kernel_expert_native_down_sum_iq2_xs_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<74, 16, expert_dequantize_iq2_xs>;
template [[host_name("kernel_expert_native_down_sum_iq2_xxs_f32")]] kernel expert_native_down_sum_t kernel_expert_native_down_sum_f32<66, 16, expert_dequantize_iq2_xxs>;
