// GPU-indexed projections. Four simdgroups own four output channels.
// Gate writes the existing inner buffer; up consumes it in-place. No scratch.
#include <metal_stdlib>
using namespace metal;
#include "native_dequant_f32.h"
#include "iq2_xs_grid.metalh"
#include "iq2_xxs_decode.metalh"
#include "mxfp4_decode.metalh"
#include "iq1_decode.metalh"

inline void expert_dequantize_iq1_s(device const uchar * blk, short il,
                                    thread float4x4 & reg) {
    iq1_dequant16(blk, uint(il), false, reg);
}

inline void expert_dequantize_iq1_m(device const uchar * blk, short il,
                                    thread float4x4 & reg) {
    iq1_dequant16(blk, uint(il), true, reg);
}

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

// Q5_K with 16-byte quant/high-bit loads (two vector loads instead of 32
// byte loads per 16 weights). Same arithmetic as latent_dequantize_q5_K, so
// decoded values are bitwise equal. Needs 16-byte aligned blocks: the 176-byte
// Q5_K block keeps every fragment aligned in a 16-byte aligned bank (checked
// by check_expert_bank).
inline void expert_dequantize_q5_K(device const uchar * blk_bytes, short il,
                                   thread float4x4 & reg) {
    const half d_h    = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar * scales = blk_bytes + 4;
    const short is  = (il / 4) * 2;
    const short k01 = (il / 2) & 1;
    uchar sc_u, m_u;
    if (is < 4) {
        sc_u = scales[is + k01] & 63;
        m_u  = scales[is + k01 + 4] & 63;
    } else {
        sc_u = (scales[is + k01 + 4] & 0x0F) | ((scales[is + k01 - 4] >> 6) << 4);
        m_u  = (scales[is + k01 + 4] >>   4) | ((scales[is + k01    ] >> 6) << 4);
    }
    const uint4 qs4 = *((device const uint4 *)(blk_bytes + 48 + 32 * (il / 4) + 16 * (il & 1)));
    const uint4 qh4 = *((device const uint4 *)(blk_bytes + 16 + 16 * (il & 1)));
    const uint ul = 1u << (il / 2);
    const short il_inner = il & 3;
    const float d   = il_inner < 2 ? (float)d_h : (float)d_h / 16.0f;
    const float dmin = (float)dmin_h;
    const float dl  = d   * (float)sc_u;
    const float ml  = dmin * (float)m_u;
    const uint mask = il_inner < 2 ? 0x0Fu : 0xF0u;
    const float qh_val = il_inner < 2 ? 16.0f : 256.0f;
    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uint shift = 8u * uint(i % 4);
        const float q_low  = (float)((qs4[i / 4] >> shift) & mask);
        const float q_high = ((qh4[i / 4] >> shift) & ul) ? qh_val : 0.0f;
        reg[i / 4][i % 4] = (float)(dl * (q_low + q_high) - ml);
    }
}

struct expert_native_args {
    ulong row_bytes;
    uint n_in;
    uint n_out;
    uint experts;
    uint top_k;
    uint mode; // low bits: 0 gate, 1 up + SwiGLU, 2 down; bit 8: half operands
    float clamp;
};

constant uint EXPERT_NATIVE_HALF_OPERANDS = 1u << 8;
constant uint EXPERT_NATIVE_PROJECTION_MASK = 3u;

// Same final coefficient narrowing as grouped half tiles, but a native SIMD
// reduction, not MMA. Keep the baseline F32 arithmetic body unchanged below.
template <ushort BYTES, ushort NL,
          void (*DEQ)(device const uchar *, short, thread float4x4 &)>
inline float expert_native_half_dot(device const uchar * row,
                                    device const float * x,
                                    uint n_in, ushort lane) {
    float partial = 0.0f;
    for (ulong k = ulong(lane) * 16u; k < n_in; k += 512u) {
        float4x4 decoded;
        DEQ(row + (k / (16u * NL)) * BYTES, short((k / 16u) % NL), decoded);
        QT_FOR_UNROLL (uint i = 0; i < 16; ++i) {
            const half w = half(decoded[i / 4][i % 4]);
            const half activation = half(x[k + i]);
            partial = fma(float(w), float(activation), partial);
        }
    }
    return simd_sum(partial);
}

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
    const uint projection = args.mode & EXPERT_NATIVE_PROJECTION_MASK;
    device const float * x = input + (projection == 2 ? selected : ulong(token)) * args.n_in;
    if ((args.mode & EXPERT_NATIVE_HALF_OPERANDS) != 0) {
        const float value = expert_native_half_dot<BYTES, NL, DEQ>(row, x, args.n_in, lane);
        if (lane == 0) {
            if (projection == 1) {
                const float g = min(output[out_index], args.clamp);
                const float u = clamp(value, -args.clamp, args.clamp);
                output[out_index] = (g / (1.0f + exp(-g))) * u;
            } else {
                output[out_index] = value;
            }
        }
        return;
    }
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
template [[host_name("kernel_expert_native_q5_K_f32")]] kernel expert_native_t kernel_expert_native_f32<QT_Q5_K_BYTES, QT_Q5_K_NL, expert_dequantize_q5_K>;
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
template [[host_name("kernel_expert_native_iq1_s_f32")]] kernel expert_native_t kernel_expert_native_f32<50, 16, expert_dequantize_iq1_s>;
template [[host_name("kernel_expert_native_iq1_m_f32")]] kernel expert_native_t kernel_expert_native_f32<56, 16, expert_dequantize_iq1_m>;

// Fused gate/up + SwiGLU for one storage type. A 4-simdgroup threadgroup owns
// two adjacent channels: simdgroups 0-1 compute gate, 2-3 compute up, then
// 0-1 apply the epilogue. Every dot product is the one-row kernel's (same
// fragment order, FMA chain and SIMD reduction) and the epilogue consumes the
// same F32 values the two-pass form stores and reloads, so outputs are bitwise
// identical; the projection runs as one dispatch with twice the simdgroups
// (the per-projection dispatches are latency-bound, ~300 GB/s for top-6).
template <ushort BYTES, ushort NL,
          void (*DEQ)(device const uchar *, short, thread float4x4 &)>
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_expert_native_gate_up_f32(
        constant expert_native_args & args [[buffer(0)]],
        device const uchar * gate_bank [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device const int * ids [[buffer(3)]],
        device const int * status [[buffer(4)]],
        device float * output [[buffer(5)]],
        device const uchar * up_bank [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float up_values[2];
    const uint channel = group.x * 2u + uint(simd & 1u);
    const bool up = simd >= 2;
    const uint slot = group.y;
    const uint token = group.z;
    const bool active = channel < args.n_out && slot < args.top_k;
    const ulong selected = ulong(token) * args.top_k + slot;
    const ulong out_index = selected * args.n_out + channel;
    // Route rejection precedes all bank/activation reads, as in the one-row
    // kernel; every thread still reaches the barrier.
    bool ready = active && status[token] == 1;
    const int expert = ready ? ids[selected] : -1;
    ready = ready && expert >= 0 && uint(expert) < args.experts;
    float value = 0.0f;
    if (ready) {
        device const uchar * row = (up ? up_bank : gate_bank)
            + (ulong(expert) * args.n_out + channel) * args.row_bytes;
        device const float * x = input + ulong(token) * args.n_in;
        float sum = 0.0f;
        for (ulong k = ulong(lane) * 16u; k < args.n_in; k += 512u) {
            float4x4 decoded;
            DEQ(row + (k / (16u * NL)) * BYTES, short((k / 16u) % NL), decoded);
            QT_FOR_UNROLL (uint i = 0; i < 16; ++i) {
                sum = fma(decoded[i / 4][i % 4], x[k + i], sum);
            }
        }
        value = simd_sum(sum);
    }
    if (up && lane == 0) up_values[simd - 2] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!up && lane == 0 && active) {
        if (!ready) {
            output[out_index] = 0.0f;
        } else {
            const float g = min(value, args.clamp);
            const float u = clamp(up_values[simd], -args.clamp, args.clamp);
            output[out_index] = (g / (1.0f + exp(-g))) * u;
        }
    }
}

typedef decltype(kernel_expert_native_gate_up_f32<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>) expert_native_gate_up_t;
#define EXPERT_NATIVE_GATE_UP(NAME, BYTES, NL, DEQ) \
    template [[host_name("kernel_expert_native_gate_up_" NAME "_f32")]] kernel expert_native_gate_up_t kernel_expert_native_gate_up_f32<BYTES, NL, DEQ>;
EXPERT_NATIVE_GATE_UP("q2_K", QT_Q2_K_BYTES, QT_Q2_K_NL, latent_dequantize_q2_K)
EXPERT_NATIVE_GATE_UP("q3_K", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K)
EXPERT_NATIVE_GATE_UP("q4_K", QT_Q4_K_BYTES, QT_Q4_K_NL, latent_dequantize_q4_K)
EXPERT_NATIVE_GATE_UP("q5_K", QT_Q5_K_BYTES, QT_Q5_K_NL, expert_dequantize_q5_K)
EXPERT_NATIVE_GATE_UP("q6_K", QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K)
EXPERT_NATIVE_GATE_UP("q8_0", QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0)
EXPERT_NATIVE_GATE_UP("iq4_nl", QT_IQ4_NL_BYTES, QT_IQ4_NL_NL, latent_dequantize_iq4_nl)
EXPERT_NATIVE_GATE_UP("iq4_xs", QT_IQ4_XS_BYTES, QT_IQ4_XS_NL, latent_dequantize_iq4_xs)
EXPERT_NATIVE_GATE_UP("iq2_s", QT_IQ2_S_BYTES, QT_IQ2_S_NL, latent_dequantize_iq2_s)
EXPERT_NATIVE_GATE_UP("iq3_xxs", QT_IQ3_XXS_BYTES, QT_IQ3_XXS_NL, latent_dequantize_iq3_xxs)
EXPERT_NATIVE_GATE_UP("iq3_s", QT_IQ3_S_BYTES, QT_IQ3_S_NL, latent_dequantize_iq3_s)
EXPERT_NATIVE_GATE_UP("iq2_xs", 74, 16, expert_dequantize_iq2_xs)
EXPERT_NATIVE_GATE_UP("iq2_xxs", 66, 16, expert_dequantize_iq2_xxs)
