# M5 Ultra TensorOps Probes

Standalone measurements that size what Metal 4 `matmul2d` tensor ops offer
prefill, taken before any engine kernel used them. These are not qwen-llm
measurements: the probes read operands straight from device memory with no
quantized-weight decode, so they bound what an engine kernel can reach.

M5 Ultra (`metal4=1`, `apple10=1`), macOS 27.0.1, Xcode 27.1, MSL 4.1.

## Files

- `mpp_mm.metal`: `C = A * W^T` kernels (K contiguous for both operands), one
  output tile per threadgroup over four SIMD-groups.
- `mpp_probe.m`: host harness. Compiles the kernels at runtime for MSL 4.0 or
  4.1, times back-to-back dispatches with GPU timestamps, and checks sampled
  outputs against a float64 CPU reference built from the exact operand values.
- `mpp_run1.jsonl`: probe results.
- `mlx_anchor.py`, `mlx_run1.jsonl`: MLX 0.32.4 on the same machine, as an
  external reference for GEMM, quantized matmul, streaming bandwidth and SDPA.

```sh
clang -O2 -fobjc-arc -framework Metal -framework Foundation mpp_probe.m -o mpp_probe
./mpp_probe mpp_mm.metal 41 > mpp_run1.jsonl
uv run mlx_anchor.py > mlx_run1.jsonl
```

## Results

TFLOP/s by `M x N x K`. Activations x weights; F32 accumulation unless noted.

| Kernel | 256x2304x5120 | 1024x2304x5120 | 1024x5120x8192 | 2048x4096x4096 |
| --- | ---: | ---: | ---: | ---: |
| F16 x F16, 64x64 tile | 29.2 | 47.5 | 66.3 | 74.2 |
| F16 x F16, 128x64 tile | 22.7 | 53.1 | 88.0 | 79.7 |
| BF16 x BF16, 64x64 | 29.1 | 47.4 | 66.7 | 75.1 |
| F16 x I8, 64x64 | 28.4 | 49.7 | 87.5 | 77.6 |
| MXFP4 weights, 64x64 | 18.6 | 39.1 | 68.1 | 63.2 |
| I8 x I8 -> I32, 64x64 | 48.0 | 73.4 | 113.4 | 101.9 |
| F32 x F16, strict | 9.6 | 20.5 | 29.3 | 28.3 |
| F32 x F32, strict | 9.6 | 20.2 | 27.4 | 28.2 |
| F32 x F32, relaxed | 22.0 | 33.3 | 27.2 | 35.6 |

- F16/BF16 operands carry the throughput. Strict and relaxed F16 run at the
  same speed, and F16 results sit about 1e-6 relative RMS from the float64
  reference on the same operands. The accuracy cost of an F16 path is the cast
  of activations to F16, not the matmul itself.
- Relaxed F32 is reduced-precision internally (about 8e-4 relative RMS) and
  gains little over strict F32. Neither is a prefill speedup.
- Small M is weak across the board, and tile shape matters: 128x64 is best at
  large shapes and worst at M=256.
- The MXFP4 check also scores the opposite nibble order (`rel_rms_hi_first`,
  about 1.4), confirming the hardware decodes the low nibble first.
- MLX on the same GPU reaches up to 107 TFLOP/s for F16/BF16 GEMM at M=2048,
  up to 86-90 for 4-bit quantized matmul at M=1024, and about 1 TB/s
  streaming. F16 GEMV approaches that bandwidth only on the vocabulary-sized
  projection (920 GB/s); 8K-wide projections reach 166-248 GB/s.

## Capability Notes

From the Xcode 27.1 SDK headers (`MetalPerformancePrimitives`
`MPPTensorOpsMatMul2d.h` and its `__impl/` headers, `metal_tensor`):

- Float operands: any mix of F16/BF16/F32 with F16, BF16 or F32 destinations.
  Integer: I8 x I8 and U8 x U8 to I32; I8/U8 with a float operand.
- Right operand only: I4/U4/I2/U2 with F16/BF16, and FP4 E2M1 or FP8
  E4M3/E5M2 with F16. FP8 x FP8 and FP4 x FP4 of the same format are allowed.
  There is no BF16 x FP8/FP4.
- Block scales are UE8M0 only, 32 elements along K, one scale plane. The
  scaled right operand must be transposed.
- The macOS 26 SDK lacks the FP4/FP8/sub-byte entry points; they need MSL 4.1
  with a 27.0 deployment target.
- K-quants, IQ formats and Q8_0 have no native operand type. They decode to
  F16 or I8 in-kernel before the tensor op.
