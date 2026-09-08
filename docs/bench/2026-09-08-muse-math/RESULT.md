# Muse fresh math: numerical transfer PASS, attention dominates decode

Production Muse packed Q8 projections use exact batched GEMV, not matrix
multiplication. `packed_exact+scalar_tail` at 6229 tokens means 6224 packed rows
and only five scalar remainder rows. That label does not explain the whole
prefill/decode gap. The report's temperature1/top-k64/top-p0.95 is sampled, not
greedy; cold-load variance has not been causally attributed to mmap eviction.

## Existing matrix path

The existing test-only Q8 matrix path passes its N=128 screen: 3285.64 ->570.06 ms,
38.96 ->224.54 tokens/s. Endpoint cosine 0.999999528815, relative RMS 0.00155347,
max absolute delta 0.09200764. All 16 greedy continuation IDs agree; continuation
minimum cosine 0.999996276, maximum relative RMS 0.00529757, max delta 0.258074.
The inherited test prints means, not individual control times; this is diagnostic
evidence, not a new promotion-grade timing packet. Its isolated synthetic chains
are not additive stage attribution.

`91e72d8e` tests longer native Current ATEM input. The full 6229-token input has
token SHA-256 `a5471b1bbf33ad3537362eba7437f82755f3bd516d8854fdadd3a3236308a10c`.
The 1024-token cell is a prefix of that rendering, not a complete request.

| Prompt rows | Exact ms | Matrix ms | Endpoint cosine | Relative RMS | Max delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 prefix | 33215.884125 | 5948.889792 | 0.999999994373 | 0.000108617 | 0.010545731 |
| 6229 complete | 214890.504250 | 76095.767791 | 0.999999988569 | 0.000250469 | 0.028245926 |

All inherited endpoint gates pass unchanged: cosine >0.99999, RMS <0.002, absolute
delta <0.1, same argmax. Both cells also pass 16 greedy continuation comparisons
and every continuation's cosine >0.99999, RMS <0.006 and delta <0.3. Maximum
continuation delta at6229 is 0.040973067. There is no matrix-vs-GEMV bitwise K/V or
general sampled-distribution claim. Times are single sequential A/B diagnostics,
not ABBA qualification. No production matrix selector/default changes.

## Actual scalar decode attribution

The same scalar token graph is replayed with encoder-stage timestamps. Production
keeps its single encoder; only the test profile splits embedding, five groups per
layer and output tail into262 stages. Profiled logits and newly written K/V rows
are bitwise identical to an ordinary replay. Raw ticks are scaled to the sampled
command's GPU span; encoder gaps remain unassigned.

| Position | Ordinary wall ms | Profile wall ms | Profile GPU ms | Attention ms | FFN ms | Front projections ms | Attention output ms | Output tail ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 82.157 | 81.708 | 78.824 | 19.127 | 44.213 | 7.470 | 3.678 | 2.775 |
| 6229 | 116.533 | 117.087 | 115.877 | 56.175 | 44.173 | 7.435 | 3.690 | 2.770 |

Attention is approximately48.5% of the long-position profiled GPU token. FFN is
38.1%. This establishes a much stronger next decode target than blindly tuning
weight kernels. The profiled envelope excludes initial validation/token setup
included in ordinary wall; these are attribution diagnostics, not an overhead
speedup comparison. No hardware bandwidth attribution follows. Resident weight
bytes also include input embedding storage not fully scanned on each token.

The existing Qwen v4 G16 path hardcodes H256 and cannot be dispatched on Muse's
H128 cache. Muse already has a geometry-compatible online scalar kernel selected
above7168; testing it below that limit is the cheapest next attention falsifier.
An H128 split-K adaptation remains separate. The materialized packed attention
body can also limit gains after matrix projections remove the GEMV bottleneck.

One longer test passes in337.27s; all attempts and raw counters are retained under
`target/profiles/muse-live-prefix/`. These results do not close fresh/decode work:
they identify concrete compute paths with substantial remaining headroom.
