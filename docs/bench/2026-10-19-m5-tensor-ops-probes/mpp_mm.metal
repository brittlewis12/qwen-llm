// MPP tensor-op matmul probes. C[m][n] = sum_k A[m][k] * W[n][k]
// A: M x K activations (K contiguous). W: N x K weights (K contiguous).
// C: M x N (N contiguous). Each threadgroup computes a TM x TN tile with
// four SIMD-groups; operands are read straight from device memory.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Dims {
  int M, N, K;
};

template <typename TA, typename TW, typename TC, int TM, int TN, bool RELAXED>
kernel void mm_nt(device TA *A [[buffer(0)]], device TW *W [[buffer(1)]],
                  device TC *C [[buffer(2)]], constant Dims &d [[buffer(3)]],
                  uint2 tg [[threadgroup_position_in_grid]]) {
  const int m0 = int(tg.y) * TM, n0 = int(tg.x) * TN;
  const int tm = min(TM, d.M - m0), tn = min(TN, d.N - n0);
  auto tA = tensor(A + size_t(m0) * d.K, dextents<int32_t, 2>(d.K, tm),
                   array<int, 2>({1, d.K}));
  auto tW = tensor(W + size_t(n0) * d.K, dextents<int32_t, 2>(d.K, tn),
                   array<int, 2>({1, d.K}));
  auto tC = tensor(C + size_t(m0) * d.N + n0, dextents<int32_t, 2>(tn, tm),
                   array<int, 2>({1, d.N}));
  constexpr auto desc = matmul2d_descriptor(
      TM, TN, static_cast<int>(dynamic_extent), false, true, RELAXED);
  matmul2d<desc, execution_simdgroups<4>> op;
  op.run(tA, tW, tC);
}

#define INST(NAME, TA, TW, TC, TM, TN, R)                                     \
  template [[host_name(NAME)]] kernel void mm_nt<TA, TW, TC, TM, TN, R>(       \
      device TA *, device TW *, device TC *, constant Dims &, uint2);

INST("f16_64x64_relaxed", half, half, float, 64, 64, true)
INST("f16_64x64_strict", half, half, float, 64, 64, false)
INST("f16_128x128_relaxed", half, half, float, 128, 128, true)
INST("f16_128x64_relaxed", half, half, float, 128, 64, true)
INST("bf16_64x64_relaxed", bfloat, bfloat, float, 64, 64, true)
INST("f32_64x64_relaxed", float, float, float, 64, 64, true)
INST("f32_64x64_strict", float, float, float, 64, 64, false)
INST("f32a_f16w_64x64_strict", float, half, float, 64, 64, false)
#if __METAL_VERSION__ >= 410
// int8 tensors need the macOS 26.4+ toolchain; keep them with the 4.1 set.
INST("f16a_i8w_64x64_relaxed", half, int8_t, float, 64, 64, true)
INST("i8_64x64", int8_t, int8_t, int32_t, 64, 64, false)
INST("i8_128x128", int8_t, int8_t, int32_t, 128, 128, false)
// Native MX: half activations x (E2M1 data plane + UE8M0 scale per 32 along K).
// Weights: data plane [N][K/2] bytes, scale plane [N][K/32] bytes.
using ScalePlane =
    tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;
using MxW =
    tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline,
           ScalePlane>;

template <int TM, int TN, bool RELAXED>
kernel void mm_mxfp4(device half *A [[buffer(0)]], device uchar *Wq [[buffer(1)]],
                     device float *C [[buffer(2)]], constant Dims &d [[buffer(3)]],
                     device uchar *S [[buffer(4)]],
                     uint2 tg [[threadgroup_position_in_grid]]) {
  const int m0 = int(tg.y) * TM, n0 = int(tg.x) * TN;
  const int tm = min(TM, d.M - m0), tn = min(TN, d.N - n0);
  auto tA = tensor(A + size_t(m0) * d.K, dextents<int32_t, 2>(d.K, tm),
                   array<int, 2>({1, d.K}));
  MxW tW(Wq + size_t(n0) * (d.K / 2), dextents<int32_t, 2>(d.K, tn),
         array<int, 2>({1, d.K}), ScalePlane(S + size_t(n0) * (d.K / 32)));
  auto tC = tensor(C + size_t(m0) * d.N + n0, dextents<int32_t, 2>(tn, tm),
                   array<int, 2>({1, d.N}));
  constexpr auto desc = matmul2d_descriptor(
      TM, TN, static_cast<int>(dynamic_extent), false, true, RELAXED);
  matmul2d<desc, execution_simdgroups<4>> op;
  op.run(tA, tW, tC);
}
template [[host_name("mxfp4_64x64_relaxed")]] kernel void
mm_mxfp4<64, 64, true>(device half *, device uchar *, device float *,
                       constant Dims &, device uchar *, uint2);
template [[host_name("mxfp4_64x64_strict")]] kernel void
mm_mxfp4<64, 64, false>(device half *, device uchar *, device float *,
                        constant Dims &, device uchar *, uint2);
#endif
