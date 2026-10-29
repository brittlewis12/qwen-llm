# V4.1 Depth Matrix: Cold Prefill, Append, And Decode

2026-10-29. Clean build `cf82e551` on the encoder-depth branch (runtime
identical to main `c1ba3e58` with the default-off skip hook). M5 Ultra
80-core, 256 GiB, GSQ-RCO IQ2_XS, production lease. Raw:
`~/qwen-runs/2026-10-29-v41-encoder-depth/`.

## Decision

The cold-prefill curve and the append curve point in opposite directions.
Cold CED prefill improves with depth (2K 1,222.8; 8K 1,895.6; 32K 2,086.1;
128K 2,128.1 tok/s), because a fixed chunk and boundary cost is amortized over
more rows. A fixed 1K append degrades with depth (941.95; 872.89; 759.07;
525.44 tok/s): the same late-context selection work is paid for only 1K rows.
At 128K the append costs **4.1x** the cold-prefill per-token time. Scalar
decode is nearly flat to 32K and about 12% lower at 128K in steady state.

The cold-prefill number alone is therefore not a roofline scoreboard. The
late-context append is the first measured saturation failure that the cold
curve hides.

## Protocol

`qwen-bench suite` with the synthetic uniform-vocabulary stream, CED prefill
(`QWEN_V41_PREFILL_EXECUTION=shrinking_ced`), one timed rep per cold point and
two for decode. The 8K cold point is consistent with the 8K CED pair
(1,895.6 here, 1,901.9 in the same-binary A/B). The 2K point uses a half-full
4,096-row chunk because the production allocator selected 4,096 rows; 8K and
deeper use full chunks. Single-run CED above 8K is outside the pair gate, so
these are mapping points, not promotion evidence.

## Cold prefill

| prompt | tok/s | wall ms |
| ---: | ---: | ---: |
| 2K | 1,222.8 | 1,674.8 |
| 8K | 1,895.6 | 4,321.5 |
| 32K | 2,086.1 | 15,707.9 |
| 128K | 2,128.1 | 61,592.3 |

## Fixed 1K append

| existing depth | tok/s | wall ms | ms per appended token |
| ---: | ---: | ---: | ---: |
| 0 | 941.95 | 1,087.1 | 1.087 |
| 8K | 872.89 | 1,173.1 | 1.173 |
| 32K | 759.07 | 1,349.0 | 1.349 |
| 128K | 525.44 | 1,948.9 | 1.949 |

The append wall per token is 1.79x higher at 128K than at depth 0, while
cold prefill at 128K is 4.1x cheaper per token than the 128K append.

## Decode

`tg32`, two reps per depth after the untimed fill:

| depth | tok/s | ms/token | notes |
| ---: | ---: | ---: | --- |
| 0 | 75.13 | 13.31 | two samples 75.22 / 75.04 |
| 8K | 73.23 | 13.66 | 73.28 / 73.18 |
| 32K | 72.46 | 13.80 | 72.37 / 72.56 |
| 128K | 66.45 | 15.05 | steady second sample; first 33.18 after the fill |

The 128K mean of the two samples is 49.82 tok/s, but the first sample is a
cold/thermal outlier after the long fill. Treat 66.45 as the steady point and
the 2x spread as a protocol warning: deep fills need their own warm-up or a
longer cool-down before decode timing.

The decode curve is not yet KV-dominated: 96K extra context tokens cost about
1.7 ms, consistent with a modest KV read on top of the fixed weight traffic.
The weight-byte floor is 7.17 ms/token at 1.09 TB/s; the observed 13.3 ms at
depth 0 is 1.86x that floor. The gap is not all recoverable bandwidth:
routed IQ2 accumulation has been measured as compute-bound in isolation, so a
streaming/cache-resident/loads-only control is required before another
decode-bandwidth claim.

## Stage hint for the 128K append

A stage-instrumented 1K append at 128K showed a large `attention_serial`
span (841 ms) beside `attention_tiles` (551 ms), `finish_post` (373 ms),
`prepare` (238 ms) and `ffn` (1,386 ms). The instrumented chunk wall was
1,884.9 ms versus 1,948.9 ms unprofiled, and stage spans overlap; read shares,
not absolutes. The serial/selection path is the leading candidate for the
late-context append cost, not the dense tile arithmetic that the cold prefill
curve emphasizes.

## Next

1. Split the 128K append serial span into index-score, block-score, top-k and
   per-row core, and separate encoder from decoder. The current span is too
   coarse to choose a kernel.
2. If selection dominates, test exact block pruning or shared candidate reuse
   before any approximate prefilter. A changed selection is a quality contract.
3. If the attention core dominates, continue the exact dataflow work (K/V
   half first, then SIMD-owned cooperative attention).
4. For decode, run the streaming versus cache-resident versus loads-only
   control, then price a decoder-oriented lossless IQ2 representation or a
   causal verifier against a measured perfect-acceptance bound.

## Limits

Synthetic tokens exercise uniform routing and Engram novelty; real code/prose
and tool traces can select different blocks and cache differently. Deep points
are one rep except decode, and the 128K decode spread shows thermal/cold
sensitivity. Cold 128K averages early and late positions; the 1K append is the
late-position probe. No quality claim is made for any scheduling or selection
change.
