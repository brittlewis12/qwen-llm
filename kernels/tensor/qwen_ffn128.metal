// Qualified dense 27B FFN tile: 128 token rows x64 outputs x32 K.
// Device half activations are prepared into dead FFN scratch; strict precision.
// Tensor topology adapted from llama.cpp v0.9.0 kernel_mul_mm, revision
// c4261a74a59c5a719e8f73038e28633b08f2bea4.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
#include "../quant_tiles.h"

struct qwen_ffn128_args {
    uint m;       // activation rows
    uint n;       // output channels
    uint k;       // multiple of the quant block (256)
    uint row_bytes;
};

template<int BYTES, void (*DEQ)(device const uchar *, short, thread half4x4 &)>
inline void qwen_ffn128_stage_weights(constant qwen_ffn128_args &a,
                                  device const uchar *w, threadgroup half *ws,
                                  uint n0, uint k0, uint tid) {
    // 128 work items = 64 channels x two 16-element subtiles. All padded
    // channels are explicitly zeroed on every K step; no invalid device reads.
    const uint row = tid / 2, j = tid % 2;
    half4x4 reg = half4x4(0.0h);
    if (n0 + row < a.n) {
        DEQ(w + ulong(n0 + row) * a.row_bytes + ulong(k0 / 256) * BYTES,
            short((k0 % 256) / 16 + j), reg);
    }
    QT_FOR_UNROLL (int i = 0; i < 16; ++i) {
        ws[row * 32 + j * 16 + i] = reg[i / 4][i % 4];
    }
}

template<int BYTES, void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void qwen_ffn128(constant qwen_ffn128_args &a [[buffer(0)]],
                             device const uchar *w [[buffer(1)]],
                             device const half *x [[buffer(2)]],
                             device float *y [[buffer(3)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half ws[64 * 32];
    const uint m0 = group.x * 128, n0 = group.y * 64;
    // Both operands are half; no relaxed conversion of a device F32 tensor.
    constexpr auto d = tensor_ops::matmul2d_descriptor(
        128, 64, static_cast<int>(dynamic_extent), false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<d, execution_simdgroups<4>> mm;
    // The SDK's cooperative-tensor type check requires an unqualified element
    // type. Like the upstream view, this mutable tensor is only read by mm.run.
    device half *input = (device half *)x;
    auto xt = tensor(input + ulong(m0) * a.k, dextents<int, 2>(int(a.k), int(a.m - m0)),
                     array<int, 2>({1, int(a.k)}));
    auto wt = tensor(ws, dextents<int, 2>(32, 64));
    auto acc = mm.template get_destination_cooperative_tensor<decltype(xt), decltype(wt), float>();
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (uint k0 = 0; k0 < a.k; k0 += 32) {
        qwen_ffn128_stage_weights<BYTES, DEQ>(a, w, ws, n0, k0, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto xv = tensor(input + ulong(m0) * a.k + k0, dextents<int, 2>(32, int(a.m - m0)),
                         array<int, 2>({1, int(a.k)}));
        mm.run(xv, wt, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto dst = tensor(y + ulong(m0) * a.n + n0,
                      dextents<int, 2>(min(64u, a.n - n0), min(128u, a.m - m0)),
                      array<int, 2>({1, int(a.n)}));
    acc.store(dst);
}


template [[host_name("kernel_qwen_ffn_q4_K_f16act_tensor128")]] kernel decltype(qwen_ffn128<144, qt_dequantize_q4_K>) qwen_ffn128<144, qt_dequantize_q4_K>;

template [[host_name("kernel_qwen_ffn_q6_K_f16act_tensor128")]] kernel decltype(qwen_ffn128<210, qt_dequantize_q6_K>) qwen_ffn128<210, qt_dequantize_q6_K>;
