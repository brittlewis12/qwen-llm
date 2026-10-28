# V4.1 CED Cross-Chunk Engram Prefetch: +2.4%, 106 ms

2026-10-28. Source: `94f0c907` on `perf/v41-engram-prefetch-main`, based on
main `1cef0778` (radix selection) plus the Engram producer and submit-ahead
stack. Artifact: GSQ-RCO IQ2_XS, 40 layers, 4,096-row chunks, M5 Ultra.
Raw: `~/qwen-runs/2026-10-28-v41-engram-prefetch/`.

## Decision

The prefetch is exact and passes the 8K pair, but saves **106 ms** (2.36%),
below the predeclared ≥150 ms bar. Keep as a candidate; move the next effort
to packed-RHS experts. The observed on rate is 1,816.2 tok/s, 6.7% above the
1,702 bar.

## What Changed

One-chunk lookahead: precompute chunk hashes from a private address-history
copy; after the current encoder command is committed, start a producer for the
next chunk's Engram rows; consume those values in the next chunk. The address
history commits only after the encoder loop succeeds. GPU commands,
arithmetic, command count and endpoint are unchanged.
`QWEN_V41_ENGRAM_PREFETCH=0` restores the synchronous per-chunk read.

## Evidence

8K pair, default on (`--runs 4`, P0 production timing, full-blocks-first):

| policy | samples (tok/s) | mean |
| --- | --- | ---: |
| Full | 1,126.70 / 1,126.54 / 1,127.18 / 1,126.96 | 1,126.84 |
| CED | 1,770.93 / 1,766.08 / 1,771.13 / 1,764.30 | **1,768.11** |

All 14 endpoints share `a2db19afd7c377c54160e90bdefc52dc04ab13a69b0f929abb34eab099da6edf`.

Same-binary CED A/B, `O S S O`, two runs per arm, 180 s between arms:

| arm | samples (tok/s) | mean | wall |
| --- | --- | ---: | ---: |
| off | 1,778.84 / 1,771.11 | 1,774.98 | 4.6153 |
| on | 1,819.48 / 1,811.88 | 1,815.68 | 4.5118 |
| on | 1,811.40 / 1,822.02 | 1,816.71 | 4.5092 |
| off | 1,769.35 / 1,778.24 | 1,773.80 | 4.6183 |

On/off: +2.36% throughput, 106 ms mean wall saving. Both controls agree
within 0.1%; both on arms within 0.06%.

## Limits

- The first chunk's Engram read and host preparation remain exposed; only the
  second chunk's read is hidden.
- The pair runs Full first and reports CED 1,768.1, below the CED-only A/B
  arms (1,816.2). Report both; use the A/B for the prefetch delta.
- One prompt, one artifact, one session. The prefetch is a candidate, not a
  promotion.

## Next

Packed-RHS IQ2_XXS expert discriminator: captured encoder and shrinking-decoder
route populations, complete pack + gate/up + down + scatter, all packing costs,
then native CED. Target 1.2-1.6x routed-stage speedup.
