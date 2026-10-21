# V4.1 Prefill: IQ2/MXFP4 Half Tiles And The MMA Attention Sink Tile

Candidate `perf/v41-prefill-attn-rebased` (code `65e8d0c1`, measured at
`6de5cc42`) against main `b69809fc`, which already carries the other lane's
F32-operand IQ2 tiles, 8-lane score kernel and compressor fusion. M5 Ultra
80-core, Unsloth DeepSeek-V4.1-Flash-1017, default wired limit, production
lease, clean binaries, A/B/B/A, 2 runs per arm, seed 1. Raw:
`metagross:~/qwen-runs/2026-10-20-flashnext/abba-v41-attn2/`.

## Throughput

`qwen-bench suite --pp 2048,4096 --tg 32`, depth 0:

| artifact | shape | main `b69809fc` | candidate | change |
|---|---|---:|---:|---:|
| IQ3_XXS | pp2048 | 339.8 (339.6-340.0) | 382.7 (382.5-382.9) | +12.6% |
| IQ3_XXS | pp4096 | 318.8 (318.7-318.8) | 358.0 (357.8-358.1) | +12.3% |
| IQ3_XXS | tg32 | 41.3 | 41.5 (41.1-41.8) | +0.4% |
| Q2_K_XL | pp2048 | 341.4 (340.9-341.9) | 381.3 (381.2-381.5) | +11.7% |
| Q2_K_XL | pp4096 | 319.8 (319.6-320.1) | 356.0 (355.3-356.6) | +11.3% |
| Q2_K_XL | tg32 | 44.1 (43.5-44.8) | 44.0 (43.1-44.8) | -0.4% |

Earlier ABBA of the same code against the pre-landing main `24ec157e`
(prefills after a 2,048-token context, `-d 2048`, 3 runs per arm):
pp2048 240.1 -> 335.4 (+39.7%), pp4096 224.2 -> 325.6 (+45.2%), tg32@d2048
35.4 -> 36.7 (within A's 34.6-36.1 spread). Raw: `abba-v41-attn/`.

Depth-0 pp2048 from the single-binary runs along the way, on `24ec157e`:
266.0 (main) -> 324.0 (half layouts) -> 340.5 (F32-operand MMA tile) ->
379.6 (half-operand MMA tile). Cumulative today from `4b1fca04`: pp2048
192 -> 382 tok/s.

## Where the prefill time was (six sampled spans per layer, 2048-row chunk)

| span | after the layouts | after the MMA tile |
|---|---:|---:|
| attention HC | 107 ms | 106 |
| attention prepare (projections, RoPE, history) | 354 | 354 |
| attention serial (owner/producer per-row loops, 8 layers) | 775 | 775 |
| attention tiles (QK^T, softmax, PV, inverse RoPE) | 1,946 | 1,614 (F32 MMA); half MMA not re-sampled |
| attention finish/post (output_a/b MMA rows) | 506 | 507 |
| FFN | 2,379 | 2,386 |

Per-layer FFN before the layouts: layers 0-4 (IQ2_XS/IQ2_XXS gate/up, no
generic route, native rows) 312-329 ms each, layers 5-39 52-70 ms. After:
60-63 ms. Attention tiles cost 47-51 ms on every layer with the scalar
score kernel and the gathered-GEMM PV.

Per-dispatch timing of one 32-query tile at 640 keys (fixture,
`QWEN_V41_SINK_TIMING=1`), M5 Ultra, two runs under different contention:

| dispatch | run 1 | run 2 |
|---|---:|---:|
| old score_rows (scalar) | 1,070 us | 1,658 us |
| MMA score, F32 operands | 748 | - |
| MMA score, half operands | - | 682 |
| softmax_rows | 18 | 32 |
| old gather + F32 grouped GEMM + finite | 217 | 415 |
| MMA PV (half) | - | 354 |
| normalize_rows | 13 | 21 |
| full old sink | 983 | 1,691 |
| full MMA sink | 745 (F32) | 643 (half) |

The F32-operand MMA score kernel loaded Q fragments from device memory on
every reduction step and was latency-bound; staging Q once and the key
blocks as half in threadgroup memory is what made the tile pay.

## Quality

Held text (same helper and packet as the earlier landings, IQ3, 384
teacher-forced targets), main versus candidate:

| document | NLL main | NLL candidate | delta | top-1 main | top-1 candidate |
|---|---:|---:|---:|---:|---:|
| prose_1 | 2.60648 | 2.61284 | +0.0064 | 0.578 | 0.594 |
| prose_2 | 4.18944 | 4.20111 | +0.0117 | 0.203 | 0.203 |
| prose_3 | 2.26673 | 2.24956 | -0.0172 | 0.453 | 0.469 |
| code_4 | 1.24928 | 1.25920 | +0.0099 | 0.703 | 0.688 |
| code_5 | 1.57311 | 1.58073 | +0.0076 | 0.656 | 0.703 |
| code_6 | 0.99666 | 0.99580 | -0.0009 | 0.766 | 0.766 |
| mean | 2.14695 | 2.14987 | +0.0029 (+0.14%) | 0.560 | 0.570 |

Task screen (22 tasks): 9/22 on both arms with the identical per-task
pattern; 19/22 answers character-identical, c7/r8/j4 diverge in text with
the same outcomes. Raw: `metagross:~/qwen-runs/2026-10-20-flashnext/attn-quality/`.

## Tests

- Generic grouped equivalence against the CPU codec for MXFP4, IQ2_XS and
  IQ2_XXS in the down and gate/up roles.
- MMA sink tile against the scalar sink on the production fixture (k =
  256/512/640, 1/8/32 queries, ragged visibility, invalid ids, NaN poison):
  outputs and post-softmax scores within 2e-3 + 4e-3|x|, status bits
  identical; the batched-vs-serial attention layer at the real geometry
  within the same bound; census tests follow the new route. 169 V4.1 lib
  tests pass.

## Limits and what is left

- One artifact, one machine. The other V4.1 lane carries an F32-operand IQ2
  tile route (`perf/v41-chunk-1024`, `2c51322f`) and scalar score-kernel
  tuning; those are superseded or duplicated by this landing and need one
  ABBA to pick a route, see the open map in PERF-ROADMAP.
- Remaining prefill budget after this: FFN tiles (~39%, decode-bound at
  ~32 rows per expert), owner/producer per-row loops (13%, grows with
  prompt length), attention tiles (now ~20%), output projections (8%).
