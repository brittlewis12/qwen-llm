# V4.1 Prefill: IQ2/MXFP4 Half Tiles And The MMA Attention Sink Tile

Candidate `perf/v41-prefill-attn` (measured at wip tip `4a89301c`; the landing
series is the same code minus two dead-code removals) against main
`24ec157e`. M5 Ultra 80-core, Unsloth DeepSeek-V4.1-Flash-1017 UD-IQ3_XXS,
default wired limit, production lease, clean binaries, A/B/B/A, 3 runs per
arm, seed 1. Raw: `metagross:~/qwen-runs/2026-10-20-flashnext/abba-v41-attn/`.

## Throughput

`qwen-bench suite --pp 2048,4096 --tg 32 -d 2048`; the depth applies to every
shape, so the prefills run after a 2,048-token context.

| shape | main `24ec157e` | candidate | change |
|---|---:|---:|---:|
| pp2048@d2048 | 240.1 (240.0-240.1) | 335.4 (335.1-335.6) | +39.7% |
| pp4096@d2048 | 224.2 (222.8-225.7) | 325.6 (325.4-325.8) | +45.2% |
| tg32@d2048 | 35.4 (34.6-36.1) | 36.7 (36.6-36.9) | +3.8% (decode untouched; within A's spread) |

Depth-0 pp2048 from the single-binary runs along the way: 266.0 (main) ->
324.0 (half layouts) -> 340.5 (F32-operand MMA tile) -> 379.6 (half-operand
MMA tile); pp4096 355.5 at the end. Cumulative today from `4b1fca04`: pp2048
192 -> 380 tok/s.

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
