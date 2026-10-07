// Flash-Next IQ2_S singleton SwiGLU, eight rows per 64-thread group.
// Decode/reduction order follows the DS4 all-slot IQ2_S fast kernel;
// Flash-Next has no activation clamp or DS4 route-status protocol.
#include <metal_stdlib>
using namespace metal;
#include "quant_tiles.h"

struct qwen4exp_expert_compat_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint top_k;
};

kernel void kernel_qwen4exp_expert_compat_iq2_s_swiglu_f32(
        constant qwen4exp_expert_compat_args & args [[buffer(0)]],
        device const uchar * gate_weight [[buffer(1)]],
        device const uchar * up_weight [[buffer(2)]],
        device const float * x [[buffer(3)]],
        device const int * expert_ids [[buffer(4)]],
        device float * inner [[buffer(5)]],
        threadgroup float * totals [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NR0 = 4;
    const short NSG = 2;
    const uint slot = tgpig.y;
    const uint first_row = (tgpig.x * NSG + uint(sgitg)) * NR0;
    const int expert = slot < args.top_k ? expert_ids[slot] : -1;
    const bool valid = slot < args.top_k
        && expert >= 0 && uint(expert) < args.n_expert;
    float gate_sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (valid && first_row < args.n_out) {
        const uint nb = args.n_in / 256u;
        const uint nb32 = nb * 8u;
        const ulong row_stride = (ulong)nb * 82u;
        const ulong expert_stride = (ulong)args.n_out * row_stride;
        device const uchar * gate_expert = gate_weight + (ulong)expert * expert_stride;
        device const uchar * up_expert = up_weight + (ulong)expert * expert_stride;
        const uint ix = uint(tiisg);
        device const float * y4 = x + 32u * ix;

        for (uint ib32 = ix; ib32 < nb32; ib32 += 32u) {
            float yl[32];
            for (short i = 0; i < 32; ++i) yl[i] = y4[i];
            const uint ibl = ib32 / 8u;
            const uint ib = ib32 & 7u;
            for (short row = 0; row < NR0; ++row) {
                const uint out_row = first_row + uint(row);
                if (out_row >= args.n_out) continue;
                device const uchar * gate_blk = gate_expert + (ulong)out_row * row_stride
                                                           + (ulong)ibl * 82u;
                device const uchar * up_blk = up_expert + (ulong)out_row * row_stride
                                                       + (ulong)ibl * 82u;
                const float gate_db = float(((device const half *)gate_blk)[0]);
                const float up_db = float(((device const half *)up_blk)[0]);
                device const uchar * gate_qbase = gate_blk + 2;
                device const uchar * up_qbase = up_blk + 2;
                device const uchar * gate_qs = gate_qbase + 4u * ib;
                device const uchar * up_qs = up_qbase + 4u * ib;
                device const uchar * gate_qh = gate_qbase + 256 / 4 + ib;
                device const uchar * up_qh = up_qbase + 256 / 4 + ib;
                device const uchar * gate_sc = gate_qbase + 256 / 4 + 256 / 32 + ib;
                device const uchar * up_sc = up_qbase + 256 / 4 + 256 / 32 + ib;
                device const uchar * gate_signs = gate_qs + 256 / 8;
                device const uchar * up_signs = up_qs + 256 / 8;
                const float gate_d1 = gate_db * (0.5f + float(gate_sc[0] & 0x0fu));
                const float gate_d2 = gate_db * (0.5f + float(gate_sc[0] >> 4));
                const float up_d1 = up_db * (0.5f + float(up_sc[0] & 0x0fu));
                const float up_d2 = up_db * (0.5f + float(up_sc[0] >> 4));
                float2 gate_sum = {0.0f, 0.0f};
                float2 up_sum = {0.0f, 0.0f};
                for (short l = 0; l < 2; ++l) {
                    constant uchar * gate_grid1 = (constant uchar *)(qt_iq2s_grid
                        + (uint(gate_qs[l + 0])
                            | ((uint(gate_qh[0]) << uint(8 - 2 * l)) & 0x300u)));
                    constant uchar * gate_grid2 = (constant uchar *)(qt_iq2s_grid
                        + (uint(gate_qs[l + 2])
                            | ((uint(gate_qh[0]) << uint(4 - 2 * l)) & 0x300u)));
                    constant uchar * up_grid1 = (constant uchar *)(qt_iq2s_grid
                        + (uint(up_qs[l + 0])
                            | ((uint(up_qh[0]) << uint(8 - 2 * l)) & 0x300u)));
                    constant uchar * up_grid2 = (constant uchar *)(qt_iq2s_grid
                        + (uint(up_qs[l + 2])
                            | ((uint(up_qh[0]) << uint(4 - 2 * l)) & 0x300u)));
                    for (short j = 0; j < 8; ++j) {
                        const float gate_s1 = ((uint(gate_signs[l + 0])
                            & uint(qt_kmask_iq2xs[j])) != 0u) ? -1.0f : 1.0f;
                        const float gate_s2 = ((uint(gate_signs[l + 2])
                            & uint(qt_kmask_iq2xs[j])) != 0u) ? -1.0f : 1.0f;
                        const float up_s1 = ((uint(up_signs[l + 0])
                            & uint(qt_kmask_iq2xs[j])) != 0u) ? -1.0f : 1.0f;
                        const float up_s2 = ((uint(up_signs[l + 2])
                            & uint(qt_kmask_iq2xs[j])) != 0u) ? -1.0f : 1.0f;
                        gate_sum[0] += yl[8 * l + j + 0] * float(gate_grid1[j]) * gate_s1;
                        gate_sum[1] += yl[8 * l + j + 16] * float(gate_grid2[j]) * gate_s2;
                        up_sum[0] += yl[8 * l + j + 0] * float(up_grid1[j]) * up_s1;
                        up_sum[1] += yl[8 * l + j + 16] * float(up_grid2[j]) * up_s2;
                    }
                }
                gate_sumf[row] += gate_d1 * gate_sum[0] + gate_d2 * gate_sum[1];
                up_sumf[row] += up_d1 * up_sum[0] + up_d2 * up_sum[1];
            }
            y4 += 32u * 32u;
        }
    }

    for (short row = 0; row < NR0; ++row) {
        const uint local_row = uint(sgitg) * uint(NR0) + uint(row);
        const float gate_total = simd_sum(gate_sumf[row]) * 0.25f;
        const float up_total = simd_sum(up_sumf[row]) * 0.25f;
        if (tiisg == 0) {
            totals[local_row] = gate_total;
            totals[8u + local_row] = up_total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        for (short row = 0; row < NR0; ++row) {
            const uint out_row = first_row + uint(row);
            if (out_row >= args.n_out) continue;
            const uint local_row = uint(sgitg) * uint(NR0) + uint(row);
            const float gate = totals[local_row];
            const float up = totals[8u + local_row];
            inner[(ulong)slot * args.n_out + out_row] = valid
                ? gate / (1.0f + exp(-gate)) * up
                : 0.0f;
        }
    }
}
