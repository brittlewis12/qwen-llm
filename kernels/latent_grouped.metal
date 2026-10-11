// Native grouped latent projection: compressed [K,M,groups] weights and
// token-major F32 [K*groups,rows] -> [M*groups,rows]. No external scratch,
// no full weight expansion, and no half operand staging. Four simdgroups
// each own one output channel and reuse decoded weights across four tokens.
// Tail lanes/tokens/channels never read or write outside the logical view.
#include <metal_stdlib>
using namespace metal;
#include "quant_tiles.h"

// F32 forms of the canonical quant_tiles.h decoders: identical storage loads,
// tables, indices and arithmetic, changing only the output register type and
// half narrowing conversions to float. The latent source-parity test enforces
// this relationship for every decoder; keep these in sync with that header.
// The half storage scales are widened before arithmetic. F32 and BF16 weights
// retain their range and precision; inputs and accumulators are always F32.

inline void latent_dequantize_q2_0(device const uchar * blk,
                               short il,
                               thread float4x4 & reg) {
    const float d = float(*((device const half *)blk));
    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        const int j = 16 * il + i;
        const int q = (blk[2 + j / 4] >> (2 * (j % 4))) & 3;
        reg[i / 4][i % 4] = float(d * float(q - 1));
    }
}

inline void latent_dequantize_q2_K(device const uchar * blk_bytes,
                               short il,
                               thread float4x4 & reg) {
    device const uchar * scales = blk_bytes;
    device const uchar * qs = blk_bytes + 16;
    const half d_h = *((device const half *)(blk_bytes + 80));
    const half dmin_h = *((device const half *)(blk_bytes + 82));

    const uint sub = uint(il);
    const uint q_offset = 32u * (sub >> 3) + 16u * (sub & 1u);
    const uint shift = ((sub >> 1) & 3u) * 2u;
    const uint sc = uint(scales[sub]);
    const float dl = float(d_h) * float(sc & 0x0fu);
    const float ml = float(dmin_h) * float(sc >> 4);

    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uint q = (uint(qs[q_offset + uint(i)]) >> shift) & 3u;
        reg[i / 4][i % 4] = (float)(dl * float(q) - ml);
    }
}

inline void latent_dequantize_q3_K(device const uchar * blk_bytes,
                               short il,
                               thread float4x4 & reg) {
    device const uchar * hmask = blk_bytes;
    device const uchar * qs = blk_bytes + 32;
    device const uchar * scales = blk_bytes + 96;
    const half d_h = *((device const half *)(blk_bytes + 108));

    const uint sub = uint(il);
    const uint q_offset = 32u * (sub >> 3) + 16u * (sub & 1u);
    const uint h_offset = 16u * (sub & 1u);
    const uint shift = ((sub >> 1) & 3u) * 2u;
    const uint h_bit = 1u << (sub >> 1);
    const float dl = float(d_h) * float(qt_q3_K_scale_int(scales, sub));

    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uint lane = uint(i);
        const int q_low = int((uint(qs[q_offset + lane]) >> shift) & 3u);
        const int q = q_low - ((uint(hmask[h_offset + lane]) & h_bit) != 0u ? 0 : 4);
        reg[i / 4][i % 4] = (float)(dl * float(q));
    }
}

inline void latent_dequantize_q4_K(device const uchar * blk_bytes,
                               short il,
                               thread float4x4 & reg) {
    const half d_h    = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar * scales = blk_bytes + 4;
    device const uchar * qs     = blk_bytes + 4 + 12;

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

    qs = qs + (il / 4) * 32 + 16 * (il & 1);
    short il_inner = il & 3;
    const float d   = il_inner < 2 ? (float)d_h : (float)d_h / 16.0f;
    const float dmin = (float)dmin_h;
    const float dl  = d   * (float)sc_u;
    const float ml  = dmin * (float)m_u;
    const ushort mask = il_inner < 2 ? 0x0F : 0xF0;

    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (float)(dl * (float)(qs[i] & mask) - ml);
    }
}

inline void latent_dequantize_q5_K(device const uchar * blk_bytes,
                               short il,
                               thread float4x4 & reg) {
    const half d_h    = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar * scales = blk_bytes + 4;
    device const uchar * qh     = blk_bytes + 4 + 12;
    device const uchar * qs     = blk_bytes + 4 + 12 + 32;

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

    qs = qs + 32 * (il / 4) + 16 * (il & 1);
    qh = qh + 16 * (il & 1);
    const uchar ul = 1u << (il / 2);
    short il_inner = il & 3;

    const float d   = il_inner < 2 ? (float)d_h : (float)d_h / 16.0f;
    const float dmin = (float)dmin_h;
    const float dl  = d   * (float)sc_u;
    const float ml  = dmin * (float)m_u;
    const ushort mask = il_inner < 2 ? 0x0F : 0xF0;
    const float qh_val = il_inner < 2 ? 16.0f : 256.0f;

    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        const float q_low  = (float)(qs[i] & mask);
        const float q_high = (qh[i] & ul) ? qh_val : 0.0f;
        reg[i / 4][i % 4] = (float)(dl * (q_low + q_high) - ml);
    }
}

inline void latent_dequantize_q6_K(device const uchar * blk_bytes,
                               short il,
                               thread float4x4 & reg) {
    device const uint16_t * ql = (device const uint16_t *)(blk_bytes + 0);
    device const uint16_t * qh = (device const uint16_t *)(blk_bytes + 128);
    device const int8_t   * scales = (device const int8_t *)(blk_bytes + 128 + 64);
    const half d_all = ((device const half *)(blk_bytes + 128 + 64 + 16))[0];

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    float sc = scales[(il % 2) + 2 * ((il / 2))];
    short il_inner = (il / 2) & 3;

    const uint32_t kmask1 = il_inner > 1
        ? (il_inner > 2 ? 0xC0C0C0C0 : 0x30303030)
        : (il_inner > 0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il_inner > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float ml  = (float)d_all * sc * 32.0f;
    const float dl0 = (float)d_all * sc;
    const float dl1 = dl0 / 256.0f;
    const float dl2 = dl0 / (256.0f * 256.0f);
    const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
    const uint8_t shr_h = il_inner > 2 ? 2 : 0;
    const uint8_t shl_h = il_inner > 1 ? 0 : (il_inner > 0 ? 2 : 4);
    const uint8_t shr_l = il_inner > 1 ? 4 : 0;

    QT_FOR_UNROLL (int i = 0; i < 4; ++i) {
        const uint32_t low  = (ql[2 * i] | (uint32_t)(ql[2 * i + 1] << 16)) & kmask2;
        const uint32_t high = (qh[2 * i] | (uint32_t)(qh[2 * i + 1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = (float)(dl0 * ((float)(q & 0xFF))         - ml);
        reg[i][1] = (float)(dl1 * ((float)(q & 0xFF00))       - ml);
        reg[i][2] = (float)(dl2 * ((float)(q & 0xFF0000))     - ml);
        reg[i][3] = (float)(dl3 * ((float)(q & 0xFF000000))   - ml);
    }
}

inline void latent_dequantize_q8_0(device const uchar * blk_bytes,
                               short il,
                               thread float4x4 & reg) {
    const half d_h = ((device const half *)blk_bytes)[0];
    device const int8_t * qs = (device const int8_t *)(blk_bytes + 2);
    const float d = (float)d_h;
    const short base = 16 * il;
    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (float)(d * (float)qs[base + i]);
    }
}

inline void latent_dequantize_q4_0(device const uchar * blk,
                               short il,
                               thread float4x4 & reg) {
    const half d_h = *((device const half *)blk);
    const float d = float(d_h);
    device const uchar * qs = blk + 2;
    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uchar packed = qs[i];
        const int q = (il == 0) ? int(packed & 0x0f) : int(packed >> 4);
        reg[i / 4][i % 4] = float(d * float(q - 8));
    }
}

inline void latent_dequantize_q4_1(device const uchar * blk,
                               short il,
                               thread float4x4 & reg) {
    const half d_h = *((device const half *)blk);
    const half m_h = *((device const half *)(blk + 2));
    const float d = float(d_h);
    const float m = float(m_h);
    device const uchar * qs = blk + 4;
    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uchar packed = qs[i];
        const int q = (il == 0) ? int(packed & 0x0f) : int(packed >> 4);
        reg[i / 4][i % 4] = float(d * float(q) + m);
    }
}

inline void latent_dequantize_iq4_nl(device const uchar * blk_bytes,
                                 short il,
                                 thread float4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    device const uchar * qs = blk_bytes + 2;
    const bool hi = (il & 1) != 0;

    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uchar q = qs[i];
        const uint idx = hi ? uint(q >> 4) : uint(q & 0x0f);
        reg[i / 4][i % 4] = (float)(float(d_h) * qt_iq4nl_values[idx]);
    }
}

inline void latent_dequantize_iq2_s(device const uchar * blk_bytes,
                                short il,
                                thread float4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    device const uchar * qbase = blk_bytes + 2;
    device const uchar * qh = qbase + QT_QK_K / 4;
    device const uchar * scales = qh + QT_QK_K / 32;

    const uint ib32 = uint(il >> 1);
    const uint ih = uint(il & 1);
    device const uchar * qs = qbase + 4u * ib32 + 2u * ih;
    device const uchar * signs = qs + QT_QK_K / 8;

    const uint qh_lane = uint(qh[ib32]) >> (4u * ih);
    const uint scale = (uint(scales[ib32]) >> (4u * ih)) & 0x0fu;
    const float dl = float(d_h) * (0.5f + float(scale)) * 0.25f;

    constant uchar * grid1 = (constant uchar *)(qt_iq2s_grid +
        (uint(qs[0]) | ((qh_lane << 8u) & 0x300u)));
    constant uchar * grid2 = (constant uchar *)(qt_iq2s_grid +
        (uint(qs[1]) | ((qh_lane << 6u) & 0x300u)));

    QT_FOR_UNROLL (int i = 0; i < 8; ++i) {
        const float s1 = ((uint(signs[0]) & uint(qt_kmask_iq2xs[i])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((uint(signs[1]) & uint(qt_kmask_iq2xs[i])) != 0u) ? -1.0f : 1.0f;
        reg[i / 4 + 0][i % 4] = (float)(dl * float(grid1[i]) * s1);
        reg[i / 4 + 2][i % 4] = (float)(dl * float(grid2[i]) * s2);
    }
}

inline void latent_dequantize_iq3_xxs(device const uchar * blk_bytes,
                                  short il,
                                  thread float4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    device const uchar * qs = blk_bytes + 2;
    const uint ib32 = uint(il >> 1);
    const uint ih = uint(il & 1);
    device const uchar * q3 = qs + 8u * ib32;
    device const ushort * gas = (device const ushort *)(qs + QT_QK_K / 4) + 2u * ib32;
    const uint aux32 = uint(gas[0]) | (uint(gas[1]) << 16);
    const float dl = float(d_h) * (0.5f + float(aux32 >> 28)) * 0.5f;

    constant uchar * grid1 = (constant uchar *)(qt_iq3xxs_grid + q3[4u * ih + 0u]);
    constant uchar * grid2 = (constant uchar *)(qt_iq3xxs_grid + q3[4u * ih + 1u]);
    uint signs = uint(qt_ksigns_iq2xs[(aux32 >> (14u * ih)) & 127u]);
    QT_FOR_UNROLL (int i = 0; i < 4; ++i) {
        const float s1 = ((signs & uint(qt_kmask_iq2xs[i + 0])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((signs & uint(qt_kmask_iq2xs[i + 4])) != 0u) ? -1.0f : 1.0f;
        reg[0][i] = (float)(dl * float(grid1[i]) * s1);
        reg[1][i] = (float)(dl * float(grid2[i]) * s2);
    }

    grid1 = (constant uchar *)(qt_iq3xxs_grid + q3[4u * ih + 2u]);
    grid2 = (constant uchar *)(qt_iq3xxs_grid + q3[4u * ih + 3u]);
    signs = uint(qt_ksigns_iq2xs[(aux32 >> (14u * ih + 7u)) & 127u]);
    QT_FOR_UNROLL (int i = 0; i < 4; ++i) {
        const float s1 = ((signs & uint(qt_kmask_iq2xs[i + 0])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((signs & uint(qt_kmask_iq2xs[i + 4])) != 0u) ? -1.0f : 1.0f;
        reg[2][i] = (float)(dl * float(grid1[i]) * s1);
        reg[3][i] = (float)(dl * float(grid2[i]) * s2);
    }
}

inline void latent_dequantize_iq3_s(device const uchar * blk_bytes,
                                short il,
                                thread float4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    device const uchar * qs_base = blk_bytes + 2;
    device const uchar * qh = qs_base + QT_QK_K / 4;
    device const uchar * signs_base = qh + QT_QK_K / 32;
    device const uchar * scales = signs_base + QT_QK_K / 8;

    const uint ib32 = uint(il >> 1);
    const uint ih = uint(il & 1);
    device const uchar * qs = qs_base + 8u * ib32;
    device const uchar * signs = signs_base + 4u * ib32 + 2u * ih;
    const uint qh_lane = uint(qh[ib32]) >> (4u * ih);
    const uint scale = (uint(scales[ib32 >> 1]) >> (4u * (ib32 & 1u))) & 0x0fu;
    const float dl = float(d_h) * (1.0f + 2.0f * float(scale));

    constant uchar * grid1 = (constant uchar *)(qt_iq3s_grid +
        (uint(qs[4u * ih + 0u]) | ((qh_lane << 8u) & 256u)));
    constant uchar * grid2 = (constant uchar *)(qt_iq3s_grid +
        (uint(qs[4u * ih + 1u]) | ((qh_lane << 7u) & 256u)));
    QT_FOR_UNROLL (int i = 0; i < 4; ++i) {
        const float s1 = ((uint(signs[0]) & uint(qt_kmask_iq2xs[i + 0])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((uint(signs[0]) & uint(qt_kmask_iq2xs[i + 4])) != 0u) ? -1.0f : 1.0f;
        reg[0][i] = (float)(dl * float(grid1[i]) * s1);
        reg[1][i] = (float)(dl * float(grid2[i]) * s2);
    }

    grid1 = (constant uchar *)(qt_iq3s_grid +
        (uint(qs[4u * ih + 2u]) | ((qh_lane << 6u) & 256u)));
    grid2 = (constant uchar *)(qt_iq3s_grid +
        (uint(qs[4u * ih + 3u]) | ((qh_lane << 5u) & 256u)));
    QT_FOR_UNROLL (int i = 0; i < 4; ++i) {
        const float s1 = ((uint(signs[1]) & uint(qt_kmask_iq2xs[i + 0])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((uint(signs[1]) & uint(qt_kmask_iq2xs[i + 4])) != 0u) ? -1.0f : 1.0f;
        reg[2][i] = (float)(dl * float(grid1[i]) * s1);
        reg[3][i] = (float)(dl * float(grid2[i]) * s2);
    }
}

inline void latent_dequantize_iq4_xs(device const uchar * blk_bytes,
                                 short il,
                                 thread float4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    const ushort scales_h = *((device const ushort *)(blk_bytes + 2));
    device const uchar * scales_l = blk_bytes + 4;
    device const uchar * qs = blk_bytes + 8;

    const uint ib32 = uint(il >> 1);
    const uint scale_l = (uint(scales_l[ib32 >> 1]) >> (4u * (ib32 & 1u))) & 0x0fu;
    const uint scale_h = (uint(scales_h) >> (2u * ib32)) & 3u;
    const float d = float(d_h) * float(int(scale_l | (scale_h << 4)) - 32);
    const bool hi = (il & 1) != 0;
    device const uchar * q = qs + ib32 * 16u;

    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uint idx = hi ? uint(q[i] >> 4) : uint(q[i] & 0x0f);
        reg[i / 4][i % 4] = (float)(d * qt_iq4nl_values[idx]);
    }
}

inline void latent_dequantize_f32(device const uchar * blk_bytes,
                              short il,
                              thread float4x4 & reg) {
    device const float * src = (device const float *)blk_bytes + 16 * il;
    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = float(src[i]);
    }
}

inline void latent_dequantize_f16(device const uchar * blk_bytes,
                              short il,
                              thread float4x4 & reg) {
    device const half * src = (device const half *)blk_bytes + 16 * il;
    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = src[i];
    }
}

inline void latent_dequantize_bf16(device const uchar * blk_bytes,
                               short il,
                               thread float4x4 & reg) {
    device const ushort * src = (device const ushort *)blk_bytes + 16 * il;
    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = float(as_type<float>((uint)src[i] << 16));
    }
}

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
