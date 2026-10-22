# V4.1 Prefill: Dense Projections On The Tensor Tile

Candidate `perf/v41-tensor-dense` (measured at `0184a857`) against the
tensor expert tile landing `perf/v41-tensor-tile` (`13fe7e7c`, code
`42df84ba`). M5 Ultra 80-core, macOS 27.0.1, Unsloth
DeepSeek-V4.1-Flash-1017, default wired limit, production lease, clean
binaries (`build_identity.status = match` on every arm), A/B/B/A, 2 runs
per arm, seed 1. Raw: `metagross:~/qwen-runs/2026-10-22-tensor-tile/`
(`abba-dense/`, `dense.log`, `quality-dense/`).

## What changed

The V4.1 multi-row dense projections (`q_a` 5120->1280, `q_b`
1280->32768, `kv` 5120->512, `output_a` 4096->8192, `output_b`
8192->5120 and the shared expert's gate/up/down, all Q8_0 in the Unsloth
artifacts) ran on the shared simdgroup-MMA `mat_mat` tiles at 24-29
TFLOPS: ~324 MFLOP per token per layer, three quarters of the routed
experts' FLOPs and about a quarter of a 2048-token prefill after the
tensor expert tile. The same tensor tile now has a dense form
(`kernel_mat_mat_<dtype>_f32_tensor`, `DENSE` template flag: slot ==
activation row == output row, no routing buffers, buffers 0..3) and
`Linear::encode_rows` routes every MMA-eligible projection to it where
the tier exists (`moe_grouped_tensor_selected`; the same
`QWEN_EXPERT_TENSOR_TILE` switch). F32 weights (router, mHC) keep their
F32-operand kernels as before. Half operands and F32 sums like the
`mat_mat` tile it replaces; reduction order differs.

## Per dispatch at the production shapes

`tensor_dense_production_shape_timing` (Q8_0, 2048 rows, heated,
interleaved, two rounds agreeing within 0.5%):

| projection | `mat_mat` MMA | tensor | gain |
|---|---:|---:|---:|
| q_b 1280->32768 | 5.91 ms (29.1 TFLOPS) | 2.52 ms (68.3) | 2.35x |
| output_a 4096->8192 | 4.78 (28.8) | 2.12 (64.8) | 2.25x |
| output_b 8192->5120 | 6.26 (27.5) | 2.97 (57.9) | 2.11x |
| shexp gate/up 5120->2304 (each) | 1.86 (26.0) | 0.83 (58.3) | 2.25x |
| shexp down 2304->5120 | 1.70 (28.4) | 0.72 (67.0) | 2.36x |
| q_a 5120->1280 | 1.01 (26.7) | 0.49 (54.1) | 2.05x |
| kv 5120->512 | 0.46 (23.6) | 0.19 (56.7) | 2.42x |

Per layer the dense set drops from ~23.9 ms to ~10.7 ms. The Q8_0 dequant
is a byte-to-half multiply, so the dense tile runs 55-68 TFLOPS against
the 37-46 of the routed tiles with their K/I-quant decoders: the expert
tile's remaining cost is its decoders.

## Throughput

`qwen-bench suite --pp 2048,4096 --tg 32`, depth 0:

| artifact | shape | tile landing `13fe7e7c` | candidate | change |
|---|---|---:|---:|---:|
| IQ3_XXS | pp2048 | 473.9 (473.0-474.8) | 541.7 (541.1-542.3) | +14.3% |
| IQ3_XXS | pp4096 | 435.6 (435.6-435.7) | 493.0 (493.0-493.1) | +13.2% |
| IQ3_XXS | tg32 | 40.3 (40.1-40.5) | 40.8 (40.6-40.9) | +1.0% |
| Q2_K_XL | pp2048 | 477.0 (476.8-477.2) | 545.2 (544.8-545.6) | +14.3% |
| Q2_K_XL | pp4096 | 438.1 (437.8-438.5) | 496.2 (496.1-496.3) | +13.2% |
| Q2_K_XL | tg32 | 43.1 (42.7-43.5) | 42.8 (42.8-42.9) | -0.5% |

Decode is single-row and untouched. From this morning's main
`14377c46`: pp2048 383 -> 542 tok/s (+41.5%), pp4096 358 -> 493
(+37.5%). Cumulative from `4b1fca04`: pp2048 192 -> 542 (2.8x).

## Quality

Six frozen held-text documents, 64 teacher-forced targets each
(`v41_text_likelihood` witness, IQ3_XXS, `MTL_DEBUG_LAYER=1`, clean
builds), tile landing vs the candidate:

| document | base NLL | candidate NLL | delta | base top-1 | candidate top-1 |
|---|---:|---:|---:|---:|---:|
| prose_1 | 2.6094 | 2.6043 | -0.0052 | 0.609 | 0.594 |
| prose_2 | 4.1929 | 4.2048 | +0.0119 | 0.188 | 0.188 |
| prose_3 | 2.3092 | 2.2601 | -0.0491 | 0.469 | 0.453 |
| code_4 | 1.2562 | 1.2525 | -0.0037 | 0.703 | 0.719 |
| code_5 | 1.5835 | 1.5700 | -0.0135 | 0.688 | 0.688 |
| code_6 | 0.9793 | 0.9865 | +0.0072 | 0.766 | 0.766 |
| mean (384 targets) | 2.1551 | 2.1464 | -0.0087 (-0.40%) | 219/384 | 218/384 |

Per-document deltas within +-0.05 nats in both directions (prose_3 gives
back the +0.06 the expert tile showed), the same reduction-order band as
the previous half-operand landings. The 22-task screen has an identical
per-case pass/fail/truncation pattern on both arms: code 8/10, reasoning
1/8 (one truncation), JSON 0/4. Raw: `quality-dense/`.

## Limits

- M5-class only; M4 keeps the `mat_mat` tiles.
- The dense tile keeps the expert tile's 32-row x 64-column geometry;
  dense GEMMs could use larger tiles (more operand reuse per threadgroup)
  and the activation block is still re-read per 64-column tile. Open.
- Routed through `Linear::encode_rows` only (the V4.1 prefill path);
  other families' dense prefill GEMMs still take `mat_mat`.
