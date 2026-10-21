# V4.1 prefill on the merged main base: grouped IQ2 and score reduction

Candidate `perf/v41-prefill-land` (`c4125c6b`) rebased on local main
`f67fbfc2` (code base `24ec157e`). M5 Ultra 80-core, Unsloth
DeepSeek-V4.1-Flash-1017 UD-Q2_K_XL and UD-IQ3_XXS, production lease,
`qwen-bench suite --pp 1024,2048,4096 --runs 2 --seed 1`, clean binaries,
A/B/B/A. Raw: `metagross:~/qwen-runs/2026-10-21-merge/final/`.

## Throughput

| shape | main `24ec157e` | candidate `c4125c6b` | change |
|---|---:|---:|---:|
| Q2 pp1024 | 213.6 (213.5-213.6) | 341.4 (341.2-341.5) | **+59.8%** |
| Q2 pp2048 | 206.6 (206.4-206.7) | 339.2 (339.0-339.3) | **+64.2%** |
| Q2 pp4096 | 198.0 (197.9-198.0) | 318.9 (318.8-318.9) | **+61.1%** |
| IQ3 pp1024 | 273.5 (272.6-274.3) | 342.9 (342.5-343.2) | **+25.4%** |
| IQ3 pp2048 | 265.9 (265.6-266.1) | 337.0 (336.6-337.3) | **+26.7%** |
| IQ3 pp4096 | 251.7 (251.4-252.0) | 316.3 (315.9-316.6) | **+25.7%** |

tok/s, arm means with ranges; each arm is one process and two timed reps per
shape. The production allocator used 2048-row chunks at every shape. Both
candidate arms sit above both baseline arms at every shape and quant.
Q2 and IQ3 now converge at the same per-token prefill rate, which means the
old IQ2 expert bottleneck is gone and the remaining budget is common work.

## What changed

- `b3b71bb3` adds F32-operand grouped tiles for IQ2_XXS/IQ2_XS gate/up and
  down (`kernels/moe_grouped_f32.metal`, `moe_grouped_generic.rs`). Every
  routed layer in both artifacts now takes the grouped path. At 1024 rows
  the 384-expert/top-6 geometry averages 16 slots per expert and fills the
  16-slot tile exactly; at 2048 rows it needs two tiles. The decoders are
  the native IQ2 codecs; reductions stay F32.
- `b992694a` cuts score-reduction overhead. The four-key/simdgroup kernel
  stages the query row once; the landed default processes four keys in
  parallel per simdgroup, one per 8-lane group, and reduces each with three
  group shuffles instead of four 32-lane trees. Per-key dot order is F32 and
  differs only in the final tree. ABBA over the four-key dispatch:
  +1.2-1.4% at both PP1024 and PP2048 on both artifacts.
- `c4125c6b` keeps the opt-in dispatch census (`QWEN_V41_CENSUS=1`) for the
  next attribution pass.

The previous 1024-row chunk commit is not in this branch: main landed
2048-row chunks (`c9e56aeb`) with a larger measured gain.

## Where the remaining time goes

Merged candidate, Q2 PP1024, sampled stage spans and skip probes:

| span | time | share |
|---|---:|---:|
| FFN (router, shared, routed experts) | 1.18 s | 39% |
| attention score rows | 0.66 s | 22% |
| PV + softmax + output/rope | 0.55 s | 18% |
| owner/producer prepare + selection | 0.35 s | 12% |
| HC + Engram + head + glue | 0.10 s | 3% |

The score probe (`QWEN_V41_SKIP_SCORE=1`) saved 0.69 s before the 8-lane
change; FFN skip saved 1.18 s. Census: 50,720 dispatches, led by
`attention_copy` 8,192, `selection_prepare` 6,141, `attention_rope` 5,120,
`rms_norm_mul_f32` 5,120, `rope_rows` 4,038, and the sort/index family
2,560 each. Expert GEMMs are now at roughly 15 TFLOP/s effective; the scalar
score core is the largest attention item left.

## Negative results (same base, not landed)

- Eight heads per score threadgroup, sharing each K row: Q2 333.5 vs 336.6
  (-0.9%), IQ3 334.2 vs 338.4 (-1.2%). K is re-read from cache rather than
  re-fetched from L2, so the traffic saving did not pay for the lost
  threadgroup parallelism.
- Score keys per simdgroup 8/16/32: 337.0 / 334.5 / 331.7 vs 337.0 for four.
  Fatter threadgroups do not amortize a launch-bound kernel; it is not
  launch-bound.
- 32-slot fused gate/up tile (one weight dequantization for two 16-slot
  halves): Q2 pp2048 +0.25%, IQ3 pp2048 +0.9%, but -1.3%/-1.6% at PP1024
  where the second half is empty. Below the landing bar.

## Held-text witness and task screen

Same helper and protocol as the previous V4.1 packet (six frozen documents,
512-token prefix, 64 teacher-forced targets each, IQ3_XXS; helper
`49eb469f` on both arms, main `24ec157e` versus candidate `c4125c6b`):

| document | NLL main | NLL candidate | delta | top-1 main | top-1 candidate |
|---|---:|---:|---:|---:|---:|
| prose_1 | 2.60648 | 2.62046 | +0.01398 | 0.578 | 0.578 |
| prose_2 | 4.18944 | 4.19960 | +0.01016 | 0.203 | 0.203 |
| prose_3 | 2.26673 | 2.21548 | -0.05125 | 0.453 | 0.469 |
| code_4 | 1.24928 | 1.26191 | +0.01264 | 0.703 | 0.719 |
| code_5 | 1.57311 | 1.58963 | +0.01652 | 0.656 | 0.703 |
| code_6 | 0.99666 | 1.00185 | +0.00520 | 0.766 | 0.766 |
| mean | 2.14695 | 2.14816 | +0.00121 (+0.056%) | 0.560 | 0.573 |

Changes are F32 reduction order in the grouped expert tiles and the score
tree. Top-1 is +5/384 targets with no target losing its argmax. The broader
22-task screen was not rerun for this packet; the grouped F32x path is still
not bitwise and the ignored 257-token ring-dispatch witness still assumes
the pre-2048 chunk width.

## Limits

- One artifact pair, one M5 Ultra, one protocol. The chunk and grouped-tile
  gains scale with rows and are admission-priced; the score change is
  geometry-local.
- Sampled spans are inclusive and inflate wall time; use them for shares, not
  absolute throughput. The ABBA table is the throughput claim.
- The duplicate half-generic IQ2 layout on the other lane's branch was not
  isolated for a one-ABBA comparison. The F32x grouped path is the measured
  one here; if the half layout is materially faster in isolation, the two
  should be reconciled before both land.
