// Research-only IQ4_NL expert-down retile probe. The repacked bank stores each
// 32-weight block's 16 qs bytes contiguously (16-byte aligned) and keeps the
// per-block half scale in a separate array, so every weight load is aligned.
// Per-row arithmetic, reduction order and SIMD reduction are the incumbent
// kernel_expert_native_iq4_nl_f32_k640_r4's, so outputs should be bitwise.
#include <metal_stdlib>
using namespace metal;
#include "../native_dequant_f32.h"

struct fn_down_retile_args {
    ulong qs_row_bytes; // blocks * 16
    ulong d_row_bytes;  // blocks * 2
    uint n_in;
    uint n_out;
    uint experts;
    uint top_k;
};

inline void fn_down_load_x16(device const float * x, thread float * xv) {
    device const float4 * x4 = (device const float4 *)x;
    QT_FOR_UNROLL (ushort i = 0; i < 4; ++i) {
        const float4 v = x4[i];
        xv[4 * i + 0] = v[0];
        xv[4 * i + 1] = v[1];
        xv[4 * i + 2] = v[2];
        xv[4 * i + 3] = v[3];
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_fn_down_retile_iq4nl_k640_r4(
        constant fn_down_retile_args & args [[buffer(0)]],
        device const uchar * qs_bank [[buffer(1)]],
        device const uchar * d_bank [[buffer(2)]],
        device const float * input [[buffer(3)]],
        device const int * ids [[buffer(4)]],
        device const int * status [[buffer(5)]],
        device float * output [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]]) {
    const short ROWS = 4;
    const short SGS = 4;
    const uint first = (group.x * SGS + uint(simd)) * ROWS;
    const uint slot = group.y;
    const uint token = group.z;
    if (first >= args.n_out || slot >= args.top_k) return;
    const ulong selected = ulong(token) * args.top_k + slot;
    const ulong out_first = selected * args.n_out + first;
    if (status[token] != 1) {
        if (lane < ROWS) output[out_first + lane] = 0.0f;
        return;
    }
    const int expert = ids[selected];
    if (expert < 0 || uint(expert) >= args.experts) {
        if (lane < ROWS) output[out_first + lane] = 0.0f;
        return;
    }
    device const uchar * qs0 =
        qs_bank + (ulong(expert) * args.n_out + first) * args.qs_row_bytes;
    device const uchar * d0 =
        d_bank + (ulong(expert) * args.n_out + first) * args.d_row_bytes;
    device const float * x = input + selected * args.n_in;
    float sum[ROWS];
    QT_FOR_UNROLL (ushort r = 0; r < ROWS; ++r) sum[r] = 0.0f;
    for (uint k = uint(lane) * 16u; k < args.n_in; k += 512u) {
        float xv[16];
        fn_down_load_x16(x + k, xv);
        const uint block = k / 32u;
        const short il = short((k / 16u) % 2u);
        const bool hi = (il & 1) != 0;
        QT_FOR_UNROLL (ushort r = 0; r < ROWS; ++r) {
            const half d_h = *((device const half *)(d0 + ulong(r) * args.d_row_bytes
                                                      + ulong(block) * 2u));
            device const uchar * qs = qs0 + ulong(r) * args.qs_row_bytes
                                             + ulong(block) * 16u;
            float4x4 decoded;
            QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
                const uchar q = qs[i];
                const uint idx = hi ? uint(q >> 4) : uint(q & 0x0f);
                decoded[i / 4][i % 4] = float(d_h) * qt_iq4nl_values[idx];
            }
            QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
                sum[r] = fma(decoded[i / 4][i % 4], xv[i], sum[r]);
            }
        }
    }
    QT_FOR_UNROLL (ushort r = 0; r < ROWS; ++r) {
        const float value = simd_sum(sum[r]);
        if (lane == 0) output[out_first + r] = value;
    }
}
