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
above7168; the subsequent overlap falsifier and partitioned replacement are
reported below. The materialized packed attention body can also limit gains after
matrix projections remove the GEMV bottleneck.

One longer test passes in337.27s; all attempts and raw counters are retained under
`target/profiles/muse-live-prefix/`. These results do not close fresh/decode work:
they identify concrete compute paths with substantial remaining headroom.

## Existing online overlap KILL; split primitive HOLD

`1741d3dd` forces the existing online kernel below7168 in tests only. Numerical
oracles at1/257/2048/6229/7168 pass with offset views and output guards. Separate
warm ABBA precedes eight-dispatch measured ABBA. GPU means at2048 are
0.337820 ->0.379888ms and at6229 1.012852 ->1.152206ms: 12.45/13.76% slower,
with control spreads0.236/0.00463%. **KILL**, no whole-model run or selector change.
Warm2048 A1/A2 were0.818578/0.337297ms; retained, not substituted for measured arms.

`ff0b87c2` changes the mechanism: partition the H128 online position scan into at
most32 parts, then merge softmax partials. Scratch is540672 driver bytes (528KiB),
held across both arms. Host checks reject concurrent encoders, overflow, invalid
views and aliases before dispatch; hazard notes cover both stages. Both kernels
remain research/test-only. Fourteen numerical cases cover1/2/31/32/127/128/129/
257/2048/6229/7168/7169/32769/257, NaN-poisoned partials, offset views and guards.
Worst cosine0.999999999505, maxdelta1.3411e-7 pass cosine>=0.999999/abs<=5e-4.
Split fixtures have stronger query/key/value scaling than the old online fixture.

| Visible positions | GPU ABBA ms | Mean saved | A spread | Disposition |
| --- | --- | ---: | ---: | --- |
| 2048 | .400874997 / .031281263 / .032718701 / .337307283 | 91.330% | 17.223% | HOLD |
| 6229 | 1.012109395 / .053953147 / .056130229 / 1.011140645 | 94.559% | 0.09576% | PASS |

Frozen primitive gates were >=50% mean AND both-pair GPU savings, <=5% A spread
in both cells. Overall **HOLD**, not rescued. Warm2048 A1/A2 were1.389870/0.522755ms.

## Independent whole-model split decode PASS

Before running `b0857a9d`, an explicit admission amendment allowed one separately
frozen whole-model packet despite the primitive HOLD. Independent review approved
this narrower question; it did not waive or repair the failed primitive control.

One resident actual Q8 model/session; exact default prefill primes the same native
Current6229 prompt and its1024 prefix. Each arm rewinds to the same causal frontier
and seed logits, then executes16 sequential greedy **consumed forwards**. This is
not the usual16-emitted-token generation interval (which needs15 transitions).
Wall includes rewind, greedy selection, allocations, encoding, completion and
full-logit readbacks. All16 CPU logits are retained for later checks. Scratch is
allocated once and held in both arms; no counters/profile encoders during timing.

All-logit/new-KV oracles and existing-prefix hashes precede a separate warm ABBA,
then measured ABBA. No oracle readback/hash is interposed between timed arms.
Post-packet checks compare all IDs/logits and verify final frontier/prefix hash.

| Prefix | Whole16-forward ABBA ms | Mean A -> B ms | Saved | A spread | Forwards/s A -> B |
| --- | --- | --- | ---: | ---: | --- |
| 1024 | 1264.424250 / 1006.313125 / 1010.307500 / 1275.376541 | 1269.900396 ->1008.310312 | 20.599% | 0.86245% | 12.599 ->15.868 |
| 6229 | 1849.522375 / 1004.228958 / 1003.605917 / 1844.526500 | 1847.024438 ->1003.917438 | 45.647% | 0.27048% | 8.663 ->15.938 |

The6229 primary clears >=20% mean/both-pair savings (pairs45.703/45.590%);
1024 clears <=3% regression (pairs20.413/20.784% saved); both controls clear5%.
All16 greedy IDs agree in each cell. Worst logits: cosine0.999999997517,
relative RMS0.000156957, absolute delta0.00797224, against >0.99999/<0.002/<0.1.
New-KV worst cosine0.999999988294, RMS0.000153007 pass >0.99999/<0.002;
maxdelta0.00390625 is reported, not separately gated. Existing active K/V prefixes
remain bitwise unchanged. Arithmetic is tolerance-qualified, not bitwise or
distribution-exact; sampled IDs need not agree seed-for-seed.

One leased packet passes in280.01s. Independent source/result review passes.
This establishes warm whole-forward latency only, not production selection,
HTTP/CLI generation throughput, matrix-prefill composition, cold loading or
other context/model/device coverage. Next: bounded real-runner opt-in qualification,
then existing matrix-prefill delivery before another packed-attention kernel.

Raw attempts, thermal/memory counters and mechanical score remain under
`target/profiles/muse-live-prefix/{online,split,decode}-01*` and
`attention-score.json`; build-online-01's missing-import failure was repaired
before execution. No timed packet was rerun or rescored to change its disposition.

## Bounded real-runner delivery PASS (opt-in)

`d8295fa3` moves the unchanged shader to the product metallib and replaces test TLS
selection with explicit session-owned scratch. `MuseGlimmerRuntimeOptions` defaults
off; `load_with_options` rejects split requests outside Q8_0/unified Apple M4 Max
and includes scratch in aggregate/session admission and allocation reconciliation.
Only ordinary generated-token positions1024..7168 (upper exclusive) select split;
prefill, capture and intervention graphs remain unchanged. Outside the range the
original path remains selected. GPU failure poisons the session, never retries.
The interval is bounded eligibility, not whole-model proof at every position.

`QWEN_MUSE_SPLIT_DECODE=1 qwen run ...` enables this path; `0` or unset retains
original math. No serving/lens environment opt-in. README documents scope and
numerical-versus-bitwise/sampled equivalence. There is no matrix-prefill change.

One leased admission/isolation regression passes in70.33s. Default and opt-in
prefill at31/1031 native Current-prefix tokens (packed plus scalar tails) agree
bitwise in all logits and active K/V. Eight generated forwards match the old pilot
bitwise at1031, and the default fallback bitwise at31. Logical/priced/observed
scratch deltas agree; invalid-token/frontier and poison/reset rejection pass.
The CPU selector boundary test and11 CLI generation/sampler/terminal-boundary
regressions pass. Independent implementation and result review passes.

Actual production `qwen run` executes one sequential separate-process A/B delivery
check at the same6229 native Current prompt, high reasoning, temp0, seed42,
17 outputs/16 transitions. Both finish at token limit; emitted bytes and token
fingerprint agree (`3d9ae02d476a7e3ed29c1f414ffeceb5dc6353c322f17d5a664fab12bf369578`).

| Phase / allocation | Original | Split opt-in |
| --- | ---: | ---: |
| Prefill ms | 217876.383750 | 218786.604625 |
| Generation ms | 1858.782083 | 1011.201583 |
| Emitted tokens/s (17-token numerator) | 9.14577 | 16.81168 |
| Transitions/s (16-forward numerator) | 8.60779 | 15.82276 |
| Observed session bytes | 376635392 | 377176064 |
| Aggregate required bytes | 30245060608 | 30245601280 |
| Process wall seconds | 220.269900 | 220.350083 |

This is a delivery smoke, **not another promotion-grade timing packet**. No cold
filesystem conditioning or new cold/fresh-request speedup claim; total request
time remains about220s because prefill remains28.5tokens/s. Generation includes
first-use pipeline setup, sampling and stdout delivery; the warm ABBA above is
still the whole-forward timing authority. Sampling policy is unchanged, but
floating-point differences can change sampled tokens even at the same seed.

Provenance caveat: execution records identify clean `d8295fa3`; the measured CLI
was built from the reviewed candidate before its commit, so embedded metadata
identifies `a9931487-dirty`. No source changed between that build, commit and run.
Both identities are retained. A subsequent committed-source CLI build succeeds
but does not retroactively certify the measured binary. No delivery rerun.
Raw `delivery-01*`, `cli-delivery-{A,B}-01*`, build and CPU logs remain under the
same profile directory. Primitive HOLD and old-online KILL remain unchanged.

The next delivery target is matrix prefill qualification and composition with
split decode; the fresh bottleneck is not closed by this decode result.

## Matrix prefill delivered; packed attention is the next bottleneck

`ce59bc4e` exposes the existing Q8 matrix path via default-off
`QWEN_MUSE_MATRIX_PREFILL=1`, restricted to unified M4 Max/Q8_0 and session
capacity<=7168. No new matrix buffers; scalar kernels remain unchanged but consume
numerically changed KV. CLI reports `packed_matrix+scalar_tail` honestly.

One composition regression passes in351.96s. Native Current prefixes16/31/128/1024/
6229 cover the N16 specialization, scalar tails and ordinary superchunks. All five
pass inherited endpoint cosine>0.99999/RMS<0.002/abs<0.1, all16 continuation logits
cosine>0.99999/RMS<0.006/abs<0.3 and all17 greedy emitted IDs versus exact reference.
Full6229 endpoint cosine0.999999988569, RMS0.000250469, maxdelta0.028245926.
Exact/matrix fresh diagnostics214685.718041/76366.274458ms; no paired timing claim.
The first N16 exact4311.959750ms includes first-use effects and is not a speedup.

Actual ordinary matrix packed chunk at6144+80 takes1367.552167ms. The shared-graph
stage profile takes1384.175750ms GPU, with261 stages and no output head, matching
the real chunk preceding five scalar tails. Profiled residual and all active KV
are bitwise identical to plain replay. Scaled stage costs: full attention460.043ms,
sliding attention495.194ms, FFN325.428ms, front projections75.357ms, attention output
24.124ms, attention preparation2.745ms, embedding0.140ms. Attention is69.01% of
this late chunk; that is not a measured whole-prefill fraction. Production remains
one serial encoder; these timestamps are attribution, not throughput authority.

One actual CLI composed-candidate run (same6229/high/temp0/seed42/17outputs,
16transitions) reports prefill76223.9ms /81.72tok/s, generation1019.9ms and total
77602.1ms (external process77.777731s). Stdout and token fingerprint match the retained
exact fixture; finish is token_limit. Session377176064B and aggregate30245601280B
match split-only: matrix adds no buffer. Embedded build identity is clean
`ce59bc4e`, built after commit. This is bounded opt-in delivery with diagnostic
timing, not a synthetic historical AB pair, default promotion or cold-cache claim.

Source review approved this smaller delivery qualification rather than spending
another25-minute ABBA packet merely reconfirming the existing matrix mechanism.
Next: batched H128 online attention using query-row parallelism, with causal/window
oracles followed by actual fresh-prefill composition and controlled timing if it
survives. Old singleton-online KILL and split-primitive HOLD remain unchanged.
Raw `prefill-01*`, `cli-prefill-M-01*` and committed-source builds remain under
`target/profiles/muse-live-prefix/`.
