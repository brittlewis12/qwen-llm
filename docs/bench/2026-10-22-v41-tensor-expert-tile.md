# V4.1 Prefill: Metal 4 Tensor-Ops Expert Tile

Candidate `perf/v41-tensor-tile` against main `14377c46` (code `7ca458b6`).
Throughput was measured at `0616b595` and quality at `8a5a3df0`; the
landed commit squashes those with the env switch and docs and is
kernel-identical to both. M5 Ultra 80-core, macOS 27.0.1, Metal 4.1
compiler, Unsloth DeepSeek-V4.1-Flash-1017, default wired limit, production
lease, clean binaries (`build_identity.status = match` on every arm),
A/B/B/A, 2 runs per arm, seed 1. Raw:
`metagross:~/qwen-runs/2026-10-22-tensor-tile/` (`abba/`, `peak.log`,
`tile-ab-1.log`, `variants.log`, `quality/`).

## What changed

The generic half-staged expert tile (`kernel_moe_grouped_slots_mm_generic`,
32 slots x 64 rows x 32 k per step, 4 simdgroups) stages dequantized
weights and half-cast activations in threadgroup memory and runs 16
`simdgroup_multiply_accumulate` per 32-k step. The tensor tile
(`kernels/tensor/moe_grouped_slots_tensor.metal`) stages the same operands
row-major and runs one `mpp::tensor_ops::matmul2d<32, 64, 32>` per step on
threadgroup tensors into a float cooperative tensor, so the output lands
in the `[slot][64]` layout the per-slot epilogue reads. Same host contract
(128 threads, 8 KiB, grid, buffers 0..5, clamp at 6) and the same three
roles (down, up_silu_mul, up_clamped), instantiated for all 20 generic
dtypes; the V4.1 half route dispatches it where
`MetalContext::tensor_ops_status` is `Available` (M5-class GPU on macOS
26+), `QWEN_EXPERT_TENSOR_TILE=0` restores the generic tile.

Numerics: half operands and F32 sums exactly as the generic tile; only the
reduction order differs. Tensor-vs-generic max |delta| on the synthetic
equivalence cases is ~1e-6 relative for every dtype and role (down, gate/up
with and without clamp); both pass the same CPU reference bounds.

## Raw ceiling

`matmul2d_peak_timing` (threadgroup-resident half operands, float
cooperative accumulator, no device traffic in the loop) against the
simdgroup-MMA probe `simdgroup_mma_peak_timing`, 2560 threadgroups,
second pass (the first dispatch of a cold GPU reads ~50 TFLOPS):

| tile (m x n x k, simdgroups) | TFLOPS |
|---|---:|
| 32 x 64 x 32, 4 (the expert tile) | 129.6 |
| 64 x 32 x 32, 4 | 129.6 |
| 64 x 64 x 32, 4 | 131.2 |
| 64 x 64 x 64, 4 | 131.4 |
| 128 x 64 x 32, 4 | 126.2 |
| 32 x 64 x 64, 4 | 116.0 |
| 16 x 64 x 32, 4 | 108.6 |
| 32 x 32 x 32, 1 | 129.3 |
| simdgroup MMA 8x8x8, acc 4-16 | 29.0-30.7 |

4.3x the simdgroup-MMA ceiling at the production tile shape.

## Tile at the production shape

`tensor_tile_production_shape_timing` (384 experts, 2048 tokens x top-6,
32 slots per expert, gate/up K=5120 M=2304 as two dispatches, down K=2304
M=5120; heated, generic and tensor interleaved, two rounds; the rounds
agree within 0.3%):

| dtype | gate+up generic | gate+up tensor | down generic | down tensor |
|---|---:|---:|---:|---:|
| Q3_K | 28.8 ms (20.1 TFLOPS) | 15.0 ms (38.7) | 14.6 ms (19.8) | 7.9 ms (36.9) |
| IQ3_XXS | 27.5 (21.1) | 13.1 (44.2) | 14.0 (20.7) | 6.8 (42.9) |
| IQ2_XS | 27.5 (21.1) | 12.5 (46.3) | 14.0 (20.7) | 6.4 (45.1) |
| MXFP4 | 26.2 (22.1) | 13.1 (44.2) | 13.3 (21.8) | 6.7 (43.2) |

1.86-2.19x per dispatch. At 37-46 TFLOPS against a 130 TFLOPS matmul
ceiling the tile is now bound by dequant, operand staging and the
activation path, not by the MMA.

Tile-loop variants, same harness, interleaved with the plain tile
(`variants.log`; all negative, pruned before landing):

| variant | gate+up | down |
|---|---:|---:|
| K32, double-buffered slabs (one barrier per step) | -0.4 to -1.0% | -0.5 to -3.6% |
| K64 per step | -6 to -18% | -3 to -18% |
| K64, double-buffered | -12 to -28% | -12 to -32% |

Larger per-threadgroup footprints lose; the tile wants many small resident
threadgroups rather than fewer barriers.

## Throughput

`qwen-bench suite --pp 2048,4096 --tg 32`, depth 0:

| artifact | shape | main `14377c46` | candidate | change |
|---|---|---:|---:|---:|
| IQ3_XXS | pp2048 | 382.9 (382.4-383.4) | 475.2 (475.1-475.2) | +24.1% |
| IQ3_XXS | pp4096 | 358.5 (358.2-358.7) | 437.0 (436.9-437.1) | +21.9% |
| IQ3_XXS | tg32 | 40.5 (40.5-40.5) | 41.0 (40.7-41.3) | +1.3% |
| Q2_K_XL | pp2048 | 381.2 (381.0-381.3) | 476.2 (476.0-476.5) | +24.9% |
| Q2_K_XL | pp4096 | 356.2 (355.8-356.5) | 438.3 (438.3-438.3) | +23.1% |
| Q2_K_XL | tg32 | 42.9 (42.8-43.1) | 42.8 (42.7-42.9) | -0.2% |

Decode does not use the tile (single rows take the native GEMV route).
Cumulative from `4b1fca04`: pp2048 192 -> 476 tok/s (2.5x).

## Where the prefill time is now

Six sampled spans per layer (`QWEN_V41_PREFILL_STAGES=1`, one 2048-token
IQ3 prefill, same binary, `QWEN_EXPERT_TENSOR_TILE` on/off; sampling
inflates wall time, read the shares; raw `stages/`):

| span | generic tile | tensor tile | share after |
|---|---:|---:|---:|
| FFN (router, buckets, expert tiles, weight/sum, shared expert) | 59.6 ms/layer | 33.1 | 33% |
| attention tiles (QK^T, softmax, PV) | 26.5 | 26.5 | 25% |
| attention serial (owner/producer per-row loops, 8 layers) | 98.6 (x8) | 98.4 | 18% |
| attention finish/post (output_a/b projections) | 12.6 | 12.7 | 12.5% |
| attention prepare (q_a/q_b/kv projections, RoPE, history) | 8.2 | 8.2 | 8.6% |
| attention HC | 1.5 | 1.4 | 2.6% |

The dense Q8_0 projections (`q_b` 1280->32768, `output_a` 4096->8192,
`output_b` 8192->5120, `q_a`, `kv`, and the shared expert's gate/up/down
inside FFN) are ~324 MFLOP per token per layer against ~425 for the six
routed experts, and they run on the simdgroup `mat_mat` path at ~24
TFLOPS: ~28 ms per layer, about a quarter of the prefill, the largest
remaining bucket.

## Quality

Six frozen held-text documents, 64 teacher-forced targets each
(`v41_text_likelihood` witness, IQ3_XXS, `MTL_DEBUG_LAYER=1`, clean
builds), main `14377c46` vs the candidate:

| document | base NLL | candidate NLL | delta | base top-1 | candidate top-1 |
|---|---:|---:|---:|---:|---:|
| prose_1 | 2.6128 | 2.6094 | -0.0034 | 0.594 | 0.609 |
| prose_2 | 4.2011 | 4.1929 | -0.0082 | 0.203 | 0.188 |
| prose_3 | 2.2496 | 2.3092 | +0.0596 | 0.469 | 0.469 |
| code_4 | 1.2592 | 1.2562 | -0.0030 | 0.688 | 0.703 |
| code_5 | 1.5807 | 1.5835 | +0.0027 | 0.703 | 0.688 |
| code_6 | 0.9958 | 0.9793 | -0.0165 | 0.766 | 0.766 |
| mean (384 targets) | 2.1499 | 2.1551 | +0.0052 (+0.24%) | 219/384 | 219/384 |

The same band as the half-tile landings (reduction-order differences on
half operands); no top-1 count change. The 22-task screen
(`v41-grouped-task-screen.py`, same model) has an identical per-case
pass/fail/truncation pattern on both arms: code 8/10, reasoning 1/8 (one
truncation), JSON 0/4. Raw: `quality/{base,cand}-witness/`,
`quality/{base,cand}-screen/`.

## Limits

- M5-class only: on M4 the tier is `GpuFamilyAbsent` and the generic tile
  runs unchanged; no M4 measurement changes.
- The tensor library is built with `-std=metal4.0 -mmacosx-version-min=26.0`
  (the 26.0 entry points); the 27.0 `_v2` entry points with blockwise
  `ue8m0` scales (native MXFP4 right operands) are a separate bet.
- Synthetic equivalence banks, not model weights; the model-level witness
  and screen above are the quality evidence.
- GLM-5.3's grouped layers use the same generic tile and would take the
  same route through `moe_grouped_tensor_selected`; not wired or measured
  there.
