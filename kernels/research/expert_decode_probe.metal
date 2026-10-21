// Routed-expert decode GEMV probe (research-only; never referenced by product
// PSO tables). Variants of kernel_expert_native_f32 that change only how the
// activation `x` is read and how many output rows a simdgroup owns, so the
// quant-independent ~0.5 ms/layer routed stage can be attributed.
//
//   ROWS  output rows per simdgroup (1 = production shape).
//   XMODE 0 scalar device loads (production replica)
//         1 float4 device loads of the lane's 16 contiguous elements
//         2 x staged once per threadgroup into threadgroup memory as half
//         3 no x loads at all (E-test: weight stream + decode only)
//         4 x staged once per threadgroup into threadgroup memory as float
//         5 packed_float4 device loads (4-byte alignment contract)
//
// Grid: (ceil(n_out / (4*ROWS)), top_k, 1), 128 threads. Arguments and buffer
// slots match expert_native_args so the harness can bind the same buffers.
#include <metal_stdlib>
using namespace metal;
#include "../native_dequant_f32.h"

#define XP_MAX_K 5120

struct xprobe_args {
    ulong row_bytes;
    uint n_in;
    uint n_out;
    uint experts;
    uint top_k;
    uint mode; // 0: gate; 1: up + in-place clamped SwiGLU; 2: slotwise down
    float clamp;
};

template <ushort BYTES, ushort NL,
          void (*DEQ)(device const uchar *, short, thread float4x4 &),
          ushort ROWS, ushort XMODE>
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_xprobe(
        constant xprobe_args & args [[buffer(0)]],
        device const uchar * bank [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device const int * ids [[buffer(3)]],
        device const int * status [[buffer(4)]],
        device float * output [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half xs_h[XMODE == 2 ? XP_MAX_K : 1];
    threadgroup float xs_f[XMODE == 4 ? XP_MAX_K : 1];
    const uint slot = group.y;
    const uint token = group.z;
    const uint first = (group.x * 4u + uint(simd)) * ROWS;
    const ulong selected = ulong(token) * args.top_k + slot;
    const bool ready = status[token] == 1;
    const int expert = ready ? ids[selected] : -1;
    const bool valid = ready && expert >= 0 && uint(expert) < args.experts;
    device const float * x = input + (args.mode == 2 ? selected : ulong(token)) * args.n_in;
    if (XMODE == 2 || XMODE == 4) {
        // Whole threadgroup stages x once (uniform branch; barrier is safe).
        if (valid) {
            for (uint k = tid; k < args.n_in; k += 128u) {
                if (XMODE == 2) xs_h[k] = half(x[k]);
                else xs_f[k] = x[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (first >= args.n_out) return;
    if (!valid) {
        if (lane == 0) {
            for (ushort r = 0; r < ROWS; ++r) {
                if (first + r < args.n_out) output[selected * args.n_out + first + r] = 0.0f;
            }
        }
        return;
    }
    device const uchar * rows[ROWS];
    for (ushort r = 0; r < ROWS; ++r) {
        const uint channel = min(first + r, args.n_out - 1u);
        rows[r] = bank + (ulong(expert) * args.n_out + channel) * args.row_bytes;
    }
    float sums[ROWS];
    for (ushort r = 0; r < ROWS; ++r) sums[r] = 0.0f;
    for (uint k = uint(lane) * 16u; k < args.n_in; k += 512u) {
        float4x4 xv;
        if (XMODE == 0) {
            QT_FOR_UNROLL (uint i = 0; i < 16; ++i) xv[i / 4][i % 4] = x[k + i];
        } else if (XMODE == 1) {
            device const float4 * x4 = (device const float4 *)(x + k);
            xv[0] = x4[0]; xv[1] = x4[1]; xv[2] = x4[2]; xv[3] = x4[3];
        } else if (XMODE == 2) {
            threadgroup const half4 * xh = (threadgroup const half4 *)(xs_h + k);
            xv[0] = float4(xh[0]); xv[1] = float4(xh[1]);
            xv[2] = float4(xh[2]); xv[3] = float4(xh[3]);
        } else if (XMODE == 4) {
            threadgroup const float4 * xf = (threadgroup const float4 *)(xs_f + k);
            xv[0] = xf[0]; xv[1] = xf[1]; xv[2] = xf[2]; xv[3] = xf[3];
        } else if (XMODE == 5) {
            device const packed_float4 * x4 = (device const packed_float4 *)(x + k);
            xv[0] = float4(x4[0]); xv[1] = float4(x4[1]);
            xv[2] = float4(x4[2]); xv[3] = float4(x4[3]);
        } else {
            QT_FOR_UNROLL (uint i = 0; i < 16; ++i) xv[i / 4][i % 4] = 1.0f;
        }
        const ulong block_off = (k / (16u * NL)) * BYTES;
        const short il = short((k / 16u) % NL);
        for (ushort r = 0; r < ROWS; ++r) {
            float4x4 decoded;
            DEQ(rows[r] + block_off, il, decoded);
            float s = sums[r];
            QT_FOR_UNROLL (uint i = 0; i < 16; ++i) {
                s = fma(decoded[i / 4][i % 4], xv[i / 4][i % 4], s);
            }
            sums[r] = s;
        }
    }
    for (ushort r = 0; r < ROWS; ++r) {
        const float value = simd_sum(sums[r]);
        const uint channel = first + r;
        if (lane == 0 && channel < args.n_out) {
            const ulong out_index = selected * args.n_out + channel;
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

#define XPROBE(NAME, BYTES, NL, DEQ, ROWS, XMODE) \
    template [[host_name(NAME)]] kernel void kernel_xprobe<BYTES, NL, DEQ, ROWS, XMODE>( \
        constant xprobe_args &, device const uchar *, device const float *, \
        device const int *, device const int *, device float *, uint3, ushort, ushort, ushort);

XPROBE("kernel_xprobe_q3_K_r1_x0", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 1, 0)
XPROBE("kernel_xprobe_q3_K_r1_x1", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 1, 1)
XPROBE("kernel_xprobe_q3_K_r1_x2", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 1, 2)
XPROBE("kernel_xprobe_q3_K_r1_x3", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 1, 3)
XPROBE("kernel_xprobe_q3_K_r1_x4", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 1, 4)
XPROBE("kernel_xprobe_q3_K_r1_x5", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 1, 5)
XPROBE("kernel_xprobe_q3_K_r2_x1", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 2, 1)
XPROBE("kernel_xprobe_q3_K_r4_x0", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 4, 0)
XPROBE("kernel_xprobe_q3_K_r4_x1", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 4, 1)
XPROBE("kernel_xprobe_q3_K_r4_x2", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 4, 2)
XPROBE("kernel_xprobe_q3_K_r4_x3", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 4, 3)
XPROBE("kernel_xprobe_q3_K_r8_x1", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 8, 1)
XPROBE("kernel_xprobe_q3_K_r8_x2", QT_Q3_K_BYTES, QT_Q3_K_NL, latent_dequantize_q3_K, 8, 2)
XPROBE("kernel_xprobe_q8_0_r1_x0", QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0, 1, 0)
XPROBE("kernel_xprobe_q8_0_r1_x3", QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0, 1, 3)
XPROBE("kernel_xprobe_q8_0_r4_x1", QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0, 4, 1)
XPROBE("kernel_xprobe_q8_0_r1_x1", QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0, 1, 1)
XPROBE("kernel_xprobe_q8_0_r1_x5", QT_Q8_0_BYTES, QT_Q8_0_NL, latent_dequantize_q8_0, 1, 5)
