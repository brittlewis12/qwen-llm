# V4.1 prefill final: half IQ2 layouts + MMA sink tile + fused compressor prep

Final branch `perf/v41-mma-final`, based on main `b69809fc` and rebasing the
`perf/v41-prefill-attn-series` landing (`20c50777`, `f16d2bc5`, `22560792`,
`09a939b7`, `7d8d323d`). Main landed the same series as `1fe2da7e`,
`11cb3f2f`, `65e8d0c1` and `bbe2e655`; the measurements here are on the
rebased branch. Code tip `09a939b7`; M5 Ultra 80-core, Unsloth
DeepSeek-V4.1-Flash-1017 UD-Q2_K_XL and UD-IQ3_XXS, production lease,
`qwen-bench suite --pp 1024,2048,4096 --runs 2 --seed 1`, A/B/B/A.

## Throughput against the other lane's series

Same code except the compressor norm/RoPE/copy fusion (`a7de76ab`) and the
dispatch census. Their binary is the landing series at `4a89301c`; the final
binary is `7d8d323d`. Raw:
`metagross:~/qwen-runs/2026-10-21-merge/final-mma-ab/`.

| shape | series | final | change |
|---|---:|---:|---:|
| Q2 pp1024 | 373.3 (373.0-373.5) | 375.3 (375.1-375.4) | **+0.54%** |
| Q2 pp2048 | 377.9 (377.8-377.9) | 380.5 (380.0-381.0) | **+0.70%** |
| Q2 pp4096 | 353.6 (353.4-353.7) | 356.4 (356.3-356.5) | **+0.81%** |
| IQ3 pp1024 | 373.7 (373.6-373.7) | 377.5 (377.2-377.7) | **+1.02%** |
| IQ3 pp2048 | 377.7 (377.5-377.8) | 381.0 (380.8-381.2) | **+0.89%** |
| IQ3 pp4096 | 354.0 (353.6-354.4) | 356.6 (356.3-356.8) | **+0.72%** |

tok/s, arm means with ranges. Both final arms sit above both series arms at
every shape and quant.

Against main `24ec157e` (the pre-prefill baseline), cumulative:

| shape | main | final | change |
|---|---:|---:|---:|
| Q2 pp1024 | 213.6 | 375.3 | +75.7% |
| Q2 pp2048 | 206.6 | 380.5 | +84.2% |
| Q2 pp4096 | 198.0 | 356.4 | +80.1% |
| IQ3 pp1024 | 273.5 | 377.5 | +38.0% |
| IQ3 pp2048 | 265.9 | 381.0 | +43.3% |
| IQ3 pp4096 | 251.7 | 356.6 | +41.7% |

## The duplicate IQ2 decision

Both implementations are present in the final tree, but only one is on the
production route:

- **Production: half-staged generic layouts** (`20c50777`) for MXFP4,
  IQ2_XS and IQ2_XXS expert tiles.
- **NativeF32 witness: F32-operand grouped tiles** (`f16d2bc5`, carried from
  `2c51322f`) for the forced-grouped tests and the every-layout-has-an-F32-
  down invariant.

One ABBA on the MMA path, PP2048, Q2 and IQ3, their half layout versus our
F32x production route:

| quant | half layout | F32x production | result |
|---|---:|---:|---|
| Q2 | 378.6 | 378.45 | tie (-0.04%) |
| IQ3 | 380.25 | 376.55 | half **+1.0%** |

Half layouts win the production slot. The F32x path is not deleted; it is
the witness route. Our earlier main landing of F32x as production
(`b3b71bb3`) and the 8-lane scalar score (`b992694a`) are superseded by this
branch. The MMA tile takes every released-geometry sink; non-released geometry keeps
a scalar fallback. Main's landed form keeps the 8-lane scalar fallback, while
this measurement branch used the original eight-key form; neither is
exercised by the released geometry.

## What the final tree contains

- Half-staged generic expert layouts for MXFP4/IQ2_XS/IQ2_XXS, with the
  F32-operand grouped tiles retained as the NativeF32 witness.
- Simdgroup-MMA sink tile: Q and 16-key blocks staged as half, F32
  accumulation, unchanged softmax/sink/invalid-id handling, MMA PV plus a
  normalize/finite pass. Scalar/gathered-GEMM fallback for other geometry.
- Six-span per-layer prefill profiler (`QWEN_V41_PREFILL_STAGES=1`).
- Compressor norm/RoPE/cache-publish fusion (`a7de76ab`): five per-group
  dispatches to one, +0.5-1.0% on top of the series.
- Prefill dispatch census (`QWEN_V41_CENSUS=1`, `c4125c6b`): 39,563
  dispatches at IQ3 PP1024 on the final tree.

## Quality

Final versus main `24ec157e`, same six-document held-text protocol (IQ3,
384 targets):

| document | NLL main | NLL final | delta | top-1 main | top-1 final |
|---|---:|---:|---:|---:|---:|
| prose_1 | 2.60648 | 2.61284 | +0.00637 | 0.578 | 0.594 |
| prose_2 | 4.18944 | 4.20111 | +0.01167 | 0.203 | 0.203 |
| prose_3 | 2.26673 | 2.24956 | -0.01717 | 0.453 | 0.469 |
| code_4 | 1.24928 | 1.25920 | +0.00992 | 0.703 | 0.688 |
| code_5 | 1.57311 | 1.58073 | +0.00762 | 0.656 | 0.703 |
| code_6 | 0.99666 | 0.99580 | -0.00086 | 0.766 | 0.766 |
| mean | 2.14695 | 2.14987 | +0.00292 (+0.136%) | 0.560 | 0.570 |

The final witness matches the MMA series packet exactly; the compressor
fusion is numerically identical on this protocol. The series packet's
22-task screen is 9/22 on both arms with the identical pattern and 19/22
character-identical answers. Two sources of numerical change remain: half
Q/K/V products in the MMA tile and F32 reduction order in the expert tiles.

## Remaining levers

- FFN tiles ~39%, decode-bound at ~32 rows per expert.
- Owner/producer selection loop ~12%: `selection_prepare` is still one
  dispatch per row; batch it across rows.
- Attention tiles ~20%; a fused score+softmax+PV kernel would remove the
  score-slab round trip and three dispatches per tile.
- Output projections ~8%, prepare ~6%, HC ~2%.

## Limits

- One artifact pair, one M5 Ultra, one protocol. The half layouts are
  artifact-dependent; the MMA tile is geometry-local; the fused prep is
  exact on the witness but uses a different key-norm reduction width.
- The MMA tile rounds Q/K/V to half before the products. This is the same
  half-staged policy as the dense and expert tiles, but it is not bitwise.
- The final branch is a rebase of the other lane's landing series plus the
  compressor/census commits from this lane; the other session should read
  the open map before its next bet.
