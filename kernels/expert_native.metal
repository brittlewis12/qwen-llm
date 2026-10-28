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
EXPERT_NATIVE_GATE_UP("q6_K", QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K)
EXPERT_NATIVE_GATE_UP("q8_0", QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0)
EXPERT_NATIVE_GATE_UP("iq4_nl", QT_IQ4_NL_BYTES, QT_IQ4_NL_NL, latent_dequantize_iq4_nl)
EXPERT_NATIVE_GATE_UP("iq4_xs", QT_IQ4_XS_BYTES, QT_IQ4_XS_NL, latent_dequantize_iq4_xs)
EXPERT_NATIVE_GATE_UP("iq2_s", QT_IQ2_S_BYTES, QT_IQ2_S_NL, latent_dequantize_iq2_s)
EXPERT_NATIVE_GATE_UP("iq3_xxs", QT_IQ3_XXS_BYTES, QT_IQ3_XXS_NL, latent_dequantize_iq3_xxs)
EXPERT_NATIVE_GATE_UP("iq3_s", QT_IQ3_S_BYTES, QT_IQ3_S_NL, latent_dequantize_iq3_s)
EXPERT_NATIVE_GATE_UP("iq2_xs", 74, 16, expert_dequantize_iq2_xs)
EXPERT_NATIVE_GATE_UP("iq2_xxs", 66, 16, expert_dequantize_iq2_xxs)

// Q5_K F32-operand projection with the work split of llama.cpp's
// kernel_mul_mv_q5_K_f32 (ggml/src/ggml-metal/kernels/mul_mv.metal at
// 56e603cc64b95f024de8b17ebda489f88c8bf773, MIT License, Copyright (c)
// 2023-2026 The ggml authors): each lane takes eight byte positions of four
// sub-blocks of every fourth block, so one quant byte serves two weights and
// one high-bit byte four, scales apply once per eight-weight group, and the
// mins use the activation sums. Same bindings, grid (n_out / 4 threadgroups
// of four simdgroups, one channel each), guards and epilogue as
// kernel_expert_native_f32; F32 operands only (the host never sends the
// half-operand flag for Q5_K). Summation order differs from the one-row
// kernel (decode probe on M5: q_b -13%, attention output_b -17%, shared
// gate -19%).
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_expert_native_q5_K_f32_split(
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
    const uint nb = args.n_in / 256u;
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;
    const ushort tid = lane / 4;
    const ushort ix = lane % 4;
    const ushort iq = tid / 4;
    const ushort ir = tid % 4;
    const ushort l0 = 8 * ir;
    const ushort q_offset = 32 * iq + l0;
    const ushort y_offset = 64 * iq + l0;
    const uchar hm1 = 1u << (2 * iq);
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;
    float yl[16], yh[16];
    ushort sc16[4];
    thread const uchar * sc8 = (thread const uchar *)sc16;
    device const float * y1 = x + uint(ix) * 256u + y_offset;
    float sum = 0.0f;
    for (uint i = ix; i < nb; i += 4) {
        device const uchar * block = row + ulong(i) * QT_Q5_K_BYTES;
        device const uchar * q1 = block + 48 + q_offset;
        device const uchar * q2 = q1 + 64;
        device const uchar * qh = block + 16 + l0;
        device const half * dh = (device const half *)block;
        device const ushort * a = (device const ushort *)(block + 4) + iq;
        device const float * y2 = y1 + 128;
        float4 sumy = 0.0f;
        QT_FOR_UNROLL (ushort l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0]; sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32]; sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l + 0]; sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32]; sumy[3] += yh[l + 8];
        }
        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
        float4 acc1 = 0.0f;
        float4 acc2 = 0.0f;
        QT_FOR_UNROLL (ushort l = 0; l < 8; ++l) {
            const uchar h = qh[l];
            acc1[0] += yl[l + 0] * float(q1[l] & 0x0F);
            acc1[1] += yl[l + 8] * float(q1[l] & 0xF0);
            acc1[2] += yh[l + 0] * float(q2[l] & 0x0F);
            acc1[3] += yh[l + 8] * float(q2[l] & 0xF0);
            acc2[0] += h & hm1 ? yl[l + 0] : 0.0f;
            acc2[1] += h & hm2 ? yl[l + 8] : 0.0f;
            acc2[2] += h & hm3 ? yh[l + 0] : 0.0f;
            acc2[3] += h & hm4 ? yh[l + 8] : 0.0f;
        }
        sum += float(dh[0]) * (float(sc8[0]) * (acc1[0] + 16.0f * acc2[0]) +
                               float(sc8[1]) * (acc1[1] / 16.0f + 16.0f * acc2[1]) +
                               float(sc8[4]) * (acc1[2] + 16.0f * acc2[2]) +
                               float(sc8[5]) * (acc1[3] / 16.0f + 16.0f * acc2[3])) -
               float(dh[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                               sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
        y1 += 4 * 256;
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

// IQ2_XXS F32-operand projections with the work split of llama.cpp's
// kernel_mul_mv_iq2_xxs_f32 (same source and license as the Q5_K split
// above): each lane owns whole 32-weight groups, a simdgroup computes
// EXPERT_IQ2_ROWS adjacent channels sharing the activation registers, the
// block scale applies once per group and channel, and the grid/sign tables
// live in threadgroup memory. Summation order differs from the one-row
// kernel (decode probe on M5, top-6 routed: gate -16%, down -8%).
#define EXPERT_IQ2_ROWS 2u

inline void expert_iq2_xxs_tables(threadgroup ulong * grid, threadgroup uchar * signs, ushort tid) {
    for (uint i = tid; i < 256u; i += 128u) grid[i] = iq2_xxs_grid[i];
    if (tid < 128u) signs[tid] = iq2_xxs_signs[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Accumulates sum[r] += dot(row first_row + r, x) for r < rows.
inline void expert_iq2_xxs_rows_dot(
        device const uchar * row0, ulong row_bytes, uint rows,
        device const float * x, uint n_in, ushort lane,
        threadgroup const ulong * grid_table, threadgroup const uchar * sign_table,
        thread float * sum) {
    const uint nb32 = n_in / 32u;
    float yl[32];
    for (uint ib32 = lane; ib32 < nb32; ib32 += 32u) {
        device const float * y = x + 32u * ib32;
        for (ushort i = 0; i < 32; ++i) yl[i] = y[i];
        const ulong offset = ulong(ib32 / 8u) * 66u;
        const uint ib = ib32 % 8u;
        for (uint r = 0; r < EXPERT_IQ2_ROWS; ++r) {
            if (r >= rows) break;
            device const uchar * block = row0 + ulong(r) * row_bytes + offset;
            device const ushort * q2 = (device const ushort *)(block + 2) + 4u * ib;
            const float db = float(*((device const half *)block));
            device const uchar * aux8 = (device const uchar *)q2;
            const uint aux32 = uint(q2[2]) | (uint(q2[3]) << 16);
            const float d = db * (0.5f + float(aux32 >> 28));
            float s = 0.0f;
            for (ushort l = 0; l < 4; ++l) {
                threadgroup const uchar * g = (threadgroup const uchar *)(grid_table + aux8[l]);
                const uchar signs = sign_table[(aux32 >> (7u * l)) & 127u];
                for (ushort j = 0; j < 8; ++j) {
                    s += yl[8 * l + j] * float(g[j]) * ((signs & qt_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
            }
            sum[r] += d * s * 0.25f;
        }
    }
}

// One projection (gate 0 / up+SwiGLU 1 / down 2). Grid: (ceil(n_out / (4 *
// EXPERT_IQ2_ROWS)), top_k, rows), 128 threads.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_expert_native_iq2_xxs_f32_split(
        constant expert_native_args & args [[buffer(0)]],
        device const uchar * bank [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device const int * ids [[buffer(3)]],
        device const int * status [[buffer(4)]],
        device float * output [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup ulong grid_table[256];
    threadgroup uchar sign_table[128];
    expert_iq2_xxs_tables(grid_table, sign_table, tid);
    const uint first = (group.x * 4u + uint(simd)) * EXPERT_IQ2_ROWS;
    const uint slot = group.y;
    const uint token = group.z;
    if (first >= args.n_out || slot >= args.top_k) return;
    const uint rows = min(EXPERT_IQ2_ROWS, args.n_out - first);
    const ulong selected = ulong(token) * args.top_k + slot;
    const ulong out_first = selected * args.n_out + first;
    const bool ready = status[token] == 1;
    const int expert = ready ? ids[selected] : -1;
    if (!(ready && expert >= 0 && uint(expert) < args.experts)) {
        if (lane < rows) output[out_first + lane] = 0.0f;
        return;
    }
    device const uchar * row0 = bank + (ulong(expert) * args.n_out + first) * args.row_bytes;
    const uint projection = args.mode & EXPERT_NATIVE_PROJECTION_MASK;
    device const float * x = input + (projection == 2 ? selected : ulong(token)) * args.n_in;
    float sum[EXPERT_IQ2_ROWS] = {0.0f};
    expert_iq2_xxs_rows_dot(row0, args.row_bytes, rows, x, args.n_in, lane, grid_table, sign_table, sum);
    for (uint r = 0; r < EXPERT_IQ2_ROWS; ++r) {
        if (r >= rows) break;
        const float value = simd_sum(sum[r]);
        if (lane == 0) {
            const ulong out_index = out_first + r;
            if (args.mode == 1) {
                const float g = min(output[out_index], args.clamp);
                const float u = clamp(value, -args.clamp, args.clamp);
                output[out_index] = (g / (1.0f + exp(-g))) * u;
            } else {
                output[out_index] = value;
            }
        }
    }
}

// Fused gate/up + SwiGLU with the same split: simdgroups 0-1 compute gate
// for 2 x EXPERT_IQ2_ROWS channels, 2-3 up for the same channels.
// Grid: (ceil(n_out / (2 * EXPERT_IQ2_ROWS)), top_k, rows), 128 threads;
// buffer 6 is the up bank.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_expert_native_gate_up_iq2_xxs_f32_split(
        constant expert_native_args & args [[buffer(0)]],
        device const uchar * gate_bank [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device const int * ids [[buffer(3)]],
        device const int * status [[buffer(4)]],
        device float * output [[buffer(5)]],
        device const uchar * up_bank [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup ulong grid_table[256];
    threadgroup uchar sign_table[128];
    threadgroup float up_values[2 * EXPERT_IQ2_ROWS];
    expert_iq2_xxs_tables(grid_table, sign_table, tid);
    const bool up = simd >= 2;
    const uint base = group.x * 2u * EXPERT_IQ2_ROWS;
    const uint first = base + uint(simd & 1u) * EXPERT_IQ2_ROWS;
    const uint slot = group.y;
    const uint token = group.z;
    const bool active = first < args.n_out && slot < args.top_k;
    const uint rows = active ? min(EXPERT_IQ2_ROWS, args.n_out - first) : 0u;
    const ulong selected = ulong(token) * args.top_k + slot;
    bool ready = active && status[token] == 1;
    const int expert = ready ? ids[selected] : -1;
    ready = ready && expert >= 0 && uint(expert) < args.experts;
    float sum[EXPERT_IQ2_ROWS] = {0.0f};
    if (ready) {
        device const uchar * row0 = (up ? up_bank : gate_bank)
            + (ulong(expert) * args.n_out + first) * args.row_bytes;
        device const float * x = input + ulong(token) * args.n_in;
        expert_iq2_xxs_rows_dot(row0, args.row_bytes, rows, x, args.n_in, lane, grid_table, sign_table, sum);
    }
    for (uint r = 0; r < EXPERT_IQ2_ROWS; ++r) {
        const float value = simd_sum(sum[r]);
        if (up && lane == 0) up_values[uint(simd & 1u) * EXPERT_IQ2_ROWS + r] = value;
        sum[r] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!up && lane == 0 && active) {
        const ulong out_first = selected * args.n_out + first;
        for (uint r = 0; r < rows; ++r) {
            if (!ready) {
                output[out_first + r] = 0.0f;
            } else {
                const float g = min(sum[r], args.clamp);
                const float u = clamp(up_values[uint(simd & 1u) * EXPERT_IQ2_ROWS + r], -args.clamp, args.clamp);
                output[out_first + r] = (g / (1.0f + exp(-g))) * u;
            }
        }
    }
}

// Two dense Q8_0 projections of one F32 activation in one dispatch (the V4.1
// decode query-rank and KV-latent projections both read the layer input).
// One threadgroup per output row with K split across its four simdgroups
// (each takes every fourth 512-weight pass; partials meet in threadgroup
// memory), so narrow outputs still fill the GPU. Same 16-weight fragment
// dequantization and fma as kernel_expert_native_f32; the reduction order
// differs. Grid: n_a + n_b threadgroups of 128 threads.
struct dense_pair_args { ulong row_bytes; uint n_in; uint n_a; uint n_b; };
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_dense_pair_q8_0_f32_splitk(
        constant dense_pair_args & args [[buffer(0)]],
        device const uchar * bank_a [[buffer(1)]],
        device const uchar * bank_b [[buffer(2)]],
        device const float * x [[buffer(3)]],
        device float * y_a [[buffer(4)]],
        device float * y_b [[buffer(5)]],
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[4];
    if (group >= args.n_a + args.n_b) return;
    const bool second = group >= args.n_a;
    const uint row_index = second ? group - args.n_a : group;
    device const uchar * row = (second ? bank_b : bank_a) + ulong(row_index) * args.row_bytes;
    float sum = 0.0f;
    for (uint k = (uint(simd) * 32u + uint(lane)) * 16u; k < args.n_in; k += 2048u) {
        float4x4 decoded;
        latent_dequantize_q8_0(row + (k / (16u * QT_Q8_0_NL)) * QT_Q8_0_BYTES,
                               short((k / 16u) % QT_Q8_0_NL), decoded);
        QT_FOR_UNROLL (uint i = 0; i < 16; ++i) sum = fma(decoded[i / 4][i % 4], x[k + i], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) (second ? y_b : y_a)[row_index] = (partial[0] + partial[1]) + (partial[2] + partial[3]);
}

// Dense Q5_K projection of `cols` (1..=8) F32 activation columns as a skinny
// GEMM on the simdgroup matrix units: every weight is read and dequantized
// once (in registers) for all columns, so the cost is nearly flat in `cols`.
// Input is [n_in x cols] and output [n_out x cols], each column contiguous.
// In the 8x8 fragment layout (thread_elements() of lane l are row
// fm = (l / 4 & 4) + (l / 2) % 4, columns fn, fn + 1 with
// fn = (l / 4 & 2) * 2 + (l % 2) * 2) lane l holds A[fm][fn..fn+1] = weights
// of channel fm and B[fm][fn..fn+1] = activations of columns fn, fn + 1. K is
// permuted inside each 128-weight half block m so that every lane's A elements
// come from one 16-byte quant chunk (k-index 2p + e of tile t = nibble e of
// byte t of chunk 4m + p, p = fn / 2) and its B elements are 16 contiguous
// floats per column. Each simdgroup owns DENSE_SKINNY_T8 tiles of 8 channels
// that share the B fragments. The threadgroup is (32, ny, nz): ny simdgroups
// own disjoint channel sets, nz split the blocks of K (block b goes to slice
// b % nz); partials meet once in threadgroup memory after the K loop.
// Grid: ceil(n_out / (ny * 8 * DENSE_SKINNY_T8)) threadgroups; ny * nz <= 8.
// F32 operands and accumulation; the summation order differs from the
// one-row kernels (F32 rounding, not bitwise). Columns past `cols` re-read the
// last column and are never stored. Weight rows and the input are read with
// 16-byte vector loads: both must be 16-byte aligned, n_in % 256 == 0.
#define DENSE_SKINNY_T8 2
struct dense_skinny_args { ulong row_bytes; uint n_in; uint n_out; uint cols; uint pad; };
[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_dense_skinny_mma_q5_K_f32(
        constant dense_skinny_args & args [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 tpos [[thread_position_in_threadgroup]],
        ushort3 tdim [[threads_per_threadgroup]]) {
    constexpr ushort SG_CH = 8 * DENSE_SKINNY_T8;
    threadgroup float partial[8 * SG_CH * 8];
    const uint ncols = min(args.cols, 8u);
    if (ncols == 0u) return;
    const ushort lane = tpos.x, sy = tpos.y, sz = tpos.z;
    const ushort ny = tdim.y, nz = tdim.z;
    const ushort qid = lane / 4;
    const ushort fm = (qid & 4) + ((lane / 2) % 4);
    const ushort fn = (qid & 2) * 2 + (lane % 2) * 2;
    const uint base = (group.x * ny + sy) * SG_CH;
    device const uchar * rowp[DENSE_SKINNY_T8];
    QT_FOR_UNROLL (ushort t8 = 0; t8 < DENSE_SKINNY_T8; ++t8) {
        // Ragged channel tails re-read the last row; their sums are dropped.
        const uint channel = min(base + uint(t8) * 8u + fm, args.n_out - 1u);
        rowp[t8] = weight + ulong(channel) * args.row_bytes;
    }
    const uint p = fn / 2;
    const uint h = p & 1u;
    const uint sh = 16u * (p >> 1);
    const uint pb = fm >> 1, eb = fm & 1u;
    const uint xoff = 64u * (pb >> 1) + 32u * eb + 16u * (pb & 1u);
    device const float * x0 = input + ulong(min(uint(fn), ncols - 1u)) * args.n_in + xoff;
    device const float * x1 = input + ulong(min(uint(fn) + 1u, ncols - 1u)) * args.n_in + xoff;
    simdgroup_float8x8 acc[DENSE_SKINNY_T8];
    QT_FOR_UNROLL (ushort t8 = 0; t8 < DENSE_SKINNY_T8; ++t8) {
        acc[t8] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint nb = args.n_in / 256u;
    for (uint b = sz; b < nb; b += nz) {
        const ulong boff = ulong(b) * QT_Q5_K_BYTES;
        uint4 hdr[DENSE_SKINNY_T8], qh[DENSE_SKINNY_T8];
        QT_FOR_UNROLL (ushort t8 = 0; t8 < DENSE_SKINNY_T8; ++t8) {
            hdr[t8] = *((device const uint4 *)(rowp[t8] + boff));
            qh[t8] = *((device const uint4 *)(rowp[t8] + boff + 16u + 16u * h));
        }
        QT_FOR_UNROLL (ushort m = 0; m < 2; ++m) {
            const uint hs = 2u * ((p >> 1) + 2u * m);
            const uint kb = b * 256u + 128u * m;
            float4 xa[4], xb[4];
            QT_FOR_UNROLL (ushort i = 0; i < 4; ++i) {
                xa[i] = *((device const float4 *)(x0 + kb + 4u * i));
                xb[i] = *((device const float4 *)(x1 + kb + 4u * i));
            }
            QT_FOR_UNROLL (ushort t8 = 0; t8 < DENSE_SKINNY_T8; ++t8) {
                const uint4 qs = *((device const uint4 *)(rowp[t8] + boff + 48u + 16u * (4u * m + p)));
                const half2 dd = as_type<half2>(hdr[t8].x);
                uint scw, mnw;
                if (m == 0) {
                    scw = hdr[t8].y & 0x3F3F3F3Fu;
                    mnw = hdr[t8].z & 0x3F3F3F3Fu;
                } else {
                    scw = (hdr[t8].w & 0x0F0F0F0Fu) | ((hdr[t8].y >> 2) & 0x30303030u);
                    mnw = ((hdr[t8].w >> 4) & 0x0F0F0F0Fu) | ((hdr[t8].z >> 2) & 0x30303030u);
                }
                scw >>= sh;
                mnw >>= sh;
                const float dl0 = float(dd.x) * float(scw & 0xFFu);
                const float dl1 = float(dd.x) * float((scw >> 8) & 0xFFu);
                const float ml0 = float(dd.y) * float(mnw & 0xFFu);
                const float ml1 = float(dd.y) * float((mnw >> 8) & 0xFFu);
                QT_FOR_UNROLL (ushort w = 0; w < 4; ++w) {
                    const uint hbits = qh[t8][w] >> hs;
                    const uint lo = (qs[w] & 0x0F0F0F0Fu) | ((hbits << 4) & 0x10101010u);
                    const uint hi = ((qs[w] >> 4) & 0x0F0F0F0Fu) | ((hbits << 3) & 0x10101010u);
                    const float4 wl = fma(float4(as_type<uchar4>(lo)), dl0, -ml0);
                    const float4 wh = fma(float4(as_type<uchar4>(hi)), dl1, -ml1);
                    QT_FOR_UNROLL (ushort i = 0; i < 4; ++i) {
                        simdgroup_float8x8 a, x;
                        a.thread_elements()[0] = wl[i];
                        a.thread_elements()[1] = wh[i];
                        x.thread_elements()[0] = xa[w][i];
                        x.thread_elements()[1] = xb[w][i];
                        simdgroup_multiply_accumulate(acc[t8], a, x, acc[t8]);
                    }
                }
            }
        }
    }
    const uint slot = uint(sz) * ny + sy;
    QT_FOR_UNROLL (ushort t8 = 0; t8 < DENSE_SKINNY_T8; ++t8) {
        const uint at = (slot * SG_CH + uint(t8) * 8u + fm) * 8u + fn;
        partial[at] = acc[t8].thread_elements()[0];
        partial[at + 1u] = acc[t8].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sz != 0) return;
    for (uint e = lane; e < uint(SG_CH) * ncols; e += 32u) {
        const uint ch = e % SG_CH, c = e / SG_CH;
        float s = 0.0f;
        for (uint z = 0; z < nz; ++z) s += partial[((z * ny + sy) * SG_CH + ch) * 8u + c];
        const uint channel = base + ch;
        if (channel < args.n_out) output[ulong(c) * args.n_out + channel] = s;
    }
}
