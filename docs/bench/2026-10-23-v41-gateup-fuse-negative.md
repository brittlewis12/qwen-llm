# Fused routed gate+up: negative

Probe branch `perf/v41-gateup-fuse` (local; `81d61dcd`), M5 Ultra,
production expert shape (384 experts, 2048 tokens, top-6, 32 slots),
f16act operands, `tensor_tile_production_shape_timing`. No production
path was changed.

## What was tested

The routed gate and up projections were two dispatches: each staged the
same half activation block, and the gate result round-tripped through
global memory before the up epilogue. The fused kernel stages the
activation once, runs both `matmul2d` ops against two weight slabs into
two F32 accumulators, and writes `silu(g)*u` (or the clamped form)
directly from the two threadgroup result tiles. The fused and
two-dispatch outputs are asserted bitwise equal for IQ3_XXS.

## Throughput

`gate+up tensor f16act`, two dispatches -> one fused kernel (us):

| dtype | two | fused | change |
|---|---:|---:|---:|
| Q3_K | 12,736.0 | 12,748.4 | +0.1% |
| IQ3_XXS | 9,867.6 | 9,852.5 | -0.2% |
| IQ2_XS | 9,069.5 | 9,079.1 | +0.1% |
| MXFP4 | 10,434.2 | 10,443.2 | +0.1% |
| Q8_0 | 14,350.3 | 14,571.4 | +1.5% |

Flat. The gate/up pair is bound by weight staging/dequant and the two
MMAs; the duplicated activation staging and the intermediate gate
round-trip are not material at this tile. Keep the two dispatches; the
fused branch stays local as negative evidence.

## Limits

- One M5 Ultra, one production shape, one 32-slot tile; no model-level
  ABBA because the per-dispatch result is flat.
- The fused kernel doubles the accumulator registers; Q8_0's small
  regression is consistent with that, but no further tuning is warranted
  after the flat IQ3/MXFP4 results.
