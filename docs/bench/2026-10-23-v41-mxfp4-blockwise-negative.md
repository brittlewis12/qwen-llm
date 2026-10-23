# Native MXFP4 blockwise tensor op: negative on M5

Probe branch `perf/v41-mxfp4-probe` (local; `4eff3a27`), M5 Ultra,
macOS 27.0.1, Metal 4.1 toolchain. No production path was changed.

## What was tested

Metal 4.1 adds blockwise scale planes (`tensor_blockwise<tensor_plane_scales,
metal_fp8_ue8m0_format, 32, 1>`) and FP4 tensor operands. The probe stages
half activations and packed MXFP4 weights plus E8M0 block-32 scales in
threadgroup memory and runs one `matmul2d` per 256-element K block:

```
half [32 x 256]  x  packed FP4 [64 x 256] + E8M0 [64 x 8]  ->  F32 [32 x 64]
```

## Gates

- **Compile:** the blockwise form requires the packed K extent to be a
  multiple of **256 FP4 elements** (a 128-byte row stride). K=32 and K=128
  fail the compiler's implied-stride alignment check; K=256 compiles.
- **Numerics:** the half x FP4 result matches a CPU reference using the
  standard E2M1 code table and E8M0 `2^(e-127)` scales (GGML's doubled
  table plus `2^(e-128)` is numerically equivalent; the raw nibbles are
  standard). The output store needs the transposed `[N][M]` tensor extents.
- **Throughput:** 32 slots x 64 rows, K-loop over 2304/5120, 1024
  threadgroups:

| K | per-tile | aggregate |
|---:|---:|---:|
| 2304 | 1.0 us | **9.7 TFLOPS** |
| 5120 | 2.3 us | **9.2 TFLOPS** |

The existing half-staged MXFP4 expert tile reaches **47.3 TFLOPS** (down)
and **51.2 TFLOPS** (gate+up) with f16act operands at the same production
shape. The native blockwise FP4 path is about **5x slower**; a single
threadgroup takes ~650 us for K=2304, so this is not a launch/occupancy
artifact.

## Conclusion

Do not integrate native blockwise MXFP4 for the expert tile on M5. The
routed tiles are weight-byte-bound, but this API path is not the lever;
the software dequant half tile remains substantially faster. The probe
branch is retained locally as evidence. The remaining byte-reduction
routes are a lower-bit artifact (Maya-S is +4.4% speed at +0.08 held-text
NLL) or a different staging scheme, not the 27.0 blockwise FP4 tensor op.

## Limits

- One M5 Ultra, one tile shape, threadgroup-resident operands. Device
  tensors or much larger tiles were not measured; the production tile
  uses threadgroup operands, which is the relevant form.
- The 9.7/9.2 TFLOPS numbers include the K-loop staging and barriers, not
  only the tensor op.
