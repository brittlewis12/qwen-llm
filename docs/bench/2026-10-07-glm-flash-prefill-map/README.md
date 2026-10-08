# GLM / Flash-Next prefill: measured opportunity map

Base: `09b0c9c44ff825614d995956a623c1c20932ad37`. M4 Max, 128 GiB.
Scope: single-request prefill throughput and TTFT, not decode tuning, general
refactoring, a new quant download, or a bit-identity campaign. No candidate is
promoted by this packet. Product builds retain the existing execution paths.

## Decision

The first implementation packet should expand **Flash-Next efficient router
coverage at actual planner widths**, starting at 2045. Then measure whether
merging the frontier commands still earns its extra arithmetic/state risk.
The largest shared research opportunity is **expert execution matched to real
route density**, particularly GLM's down projection. Finish the small GLM
router transfer alongside that work rather than mistaking it for the dominant
prefill bottleneck.

| Rank | Opportunity | Evidence now | Next discriminating experiment / stop rule |
| --- | --- | --- | --- |
| 1 | Flash actual-width router, then frontier scheduling | Mixed 2048 suffix saves 790 ms UD / 891 ms GSQ against 3+2045. Strict-router dispatches change from 0 to 48. Equivalent to about 8.9% / 9.9% of current warm full4096 GPU budget, **not a measured full-request candidate speedup**. | Substitute only the router at 2045 with schedule fixed, then measure mixed scheduling on that improved baseline. Keep one savings ledger. Confirm on full natural4K/8K requests and short deep suffixes; stop residual planner work if below ~1% full-prefill saving. |
| 2 | Density-aware expert down, then gate/up, both families | GLM active N32 down panels use 43.85% of lanes at N512, 17.60% at N128. All-layer counts retained. Flash/GSQ has reusable kernels, but no all-cohort census in this packet. | Actual-route IQ3_S down replay: M64N32 versus M128N16 or small-bucket execution, including bucketing/reduction. Time the leaf and whole command. Do not turn utilization into a claimed speedup; kill candidates with no repeatable material conversion. |
| 3 | Small GLM router transfer | Existing E8P32 gives 2.85% GPU saving at N512, 1.51% at N128, and 0.29% regression at N32 in the timestamped packet. | Narrow width qualification with repeated order-balanced blocks, natural prompts and continued positions. N32 stays incumbent. Cheap enough to execute before the larger #2 kernel project. |
| 4 | GLM ragged latent projections | Source uses grouped matrix work for complete 128-row groups, then one GEMV per leftover row in each absorption direction. Ordinary suite wall time: pp127 1061 ms, pp128 991 ms, pp129 1053 ms; pp255 1637 ms, pp256 1595 ms. | Isolate actual absorption/expansion bindings at 32/127/128/129/255/256. Whole-graph shape changes also affect experts and other tiles: the sweep is not causal attribution. Test an edge-safe grouped tile, not removal of the unguarded shader's row check. |
| 5 | Flash selected-QSA packed work | Accepted GSQ profile: layer7 mixer 33.87 ms for the dense2048 command, 134.61 ms for selected2045. A material depth-sensitive stage is expected to remain after router fixes. | Split index projection, scores/selection, QK and value on actual bands. Prefer query/head reuse or a measured reduction split. Do not transfer GLM singleton-attention speedups to packed QSA by assumption. |
| 6 | GDN/KDA preparation before recurrence redesign | Both have serial-over-token preparation; Flash has an existing parallel implementation. Recurrences already retain state in registers across rows. No valid current preparation/recurrence timing split. | Replay complete preparation including state update, then ordinary request conversion. Fund chunk/scan algebra only if recurrence's measured share and an all-in primitive support a substantial gain. |

Reserve: Flash's F32 HC injection is a concrete long-K/four-output short-prompt
shape worth a cheap isolated screen. It is not an established large gain;
current full selected-QSA attribution deserves priority over a substantial HC
rewrite. GLM larger prefill chunks are an expert-density arm, not a separately
additive source of savings. Few-row Q8 is not the first lever for 512/2048-row
prompts. Existing copy-only and generic tile-sweep failures remain failed.

For a measured disjoint stage fraction f and leaf speedup s, request latency
saving is f*(1-1/s). No overlapping parent/child spans, conditional cohort
extrapolations or placement savings get added to that budget.

## Live GLM router screen

Artifact: local Unsloth UD-IQ3_XXS, 109.48 GiB executed trunk. Natural token
prefixes at N32/128/512, one fresh admitted session per attempt. Each width
warms both arms, then executes A1/B1/B2/A2. A is unchanged Fast prefill; B
replaces only the 42 F32 router projections with existing E8P32. Exact lineage
is never substituted. Allocation and CPU diagnostics are outside prefill time.

| Rows | Packet1 wall saving | Packet2 wall saving | Packet3 wall / GPU saving |
| ---: | ---: | ---: | ---: |
| 32 | -0.79% | -0.28% | -0.33% / -0.29% |
| 128 | 0.60% | 1.49% | 1.47% / 1.51% |
| 512 | 3.95% | 1.82% | 2.86% / 2.85% |

Packet3 N512 ordinary GPU: A 2505.93 ms, B 2434.52 ms. Paired savings are
32.75 and 110.06 ms; A1/A2 GPU spread is 3.26%. Direction is consistent across
packets, precise effect size is not settled. Historical Flash's ~14% router
win does not transfer to GLM.

Final-block cache-hot router leaf at N512 is about 2.10 -> 0.519 ms. Multiplying
that leaf saving by 42 predicts ~66 ms, consistent in scale with the whole
graph but **not a measured all-router stage share**. Leaf bindings and hashes
are retained. Observed route IDs, weights and endpoint logits are unchanged
across router arms in this corpus; promotion still needs broader inputs and
positions, not bit equality as the goal.

Actual GLM route economics (42 layers, incumbent):

| Rows | Mean active experts / 288 | Active N16 lane use | Active N32 lane use | S16/(2*S32) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 99.14 | 15.95% | 8.07% | 0.506 |
| 128 | 177.33 | 32.88% | 17.60% | 0.535 |
| 512 | 242.83 | 64.45% | 43.85% | 0.680 |

S_B is sum(ceil(count_e/B)). The last column compares active compute panels
for hypothetical M128N16 versus M64N32 down tiles at these divisible output
widths. It omits occupancy, dequantization, additional weight reads and early
exit cost. The ~32% N512 panel reduction is not a 32% kernel/model prediction.
Gate/up already uses N16; the N32 issue specifically concerns down.

`glm-ragged-suite.json` adds a standard source-bound, production-build suite
(synthetic tokens, three measured repetitions after warmup, shape order
127/128/129/255/256). It confirms a useful boundary to investigate, not the
size of an implemented optimization. These short cells execute one packed
chunk; the common suite's `prefill_chunk` field is not a chunk override here.

## Live Flash command map and factorial

Artifacts: Unsloth UD-Q3_K_XL and GSQ-RCO IQ3_S. Both use native residency and
the same natural512 source, repeated as whole text and tokenized for4097.
The long prompt is repeated prose, not an independent agentic quality corpus.
One selected-capable session with capacity4097 is reused for diagnostic passes;
even the N512 run uses this allocation, unlike a request-shaped CLI N512 load.

Both artifacts admit selected packed execution and complete the full4096 graph.
The old short GSQ smoke's `selected_capable=false` was **a prompt-extent
decision, not a GSQ restriction**.

GSQ ordinary warm command GPU costs:

| Absolute range | Rows | GPU ms |
| --- | ---: | ---: |
| 0..2048 | 2048 | 3461.40 |
| 2048..2051 | 3 | 253.44 |
| 2051..4096 | 2045 | 5319.99 |
| Sum | 4096 | 9034.82 |

The shoulder traverses the whole model for three tokens; it is not simply
three attention queries. At the selected suffix, representative layer5 router
time is 15.88 ms versus 2.60 ms in dense2048; existing strict routing does not
admit width2045. The large selected suffix also has real QSA work: regrouping
does not remove its 2045 queries or 64 selected bands.

After an ordinary prefix to2048, restore the same checkpoint outside timing.
Run suffix A=3+2045 versus B=2048 mixed in warmed ABBA. Repeat with the strict
router disabled. Census only the warm witnesses; timed arms have no census.

| Artifact | Default A / B suffix GPU ms | Saved | Strict-disabled saved |
| --- | --- | ---: | ---: |
| UD | 5578.05 / 4787.95 | 790.10 ms | 199.34 ms |
| GSQ | 5560.30 / 4669.54 | 890.77 ms | 281.71 ms |

GSQ default pairs save 887.99 and 893.54 ms; incumbent spread is 0.031%.
UD pairs save 826.14 and 754.06 ms; spread is 1.79%. Router-policy blocks are
not interleaved. The unchanged A control shifts ~145.55 ms between UD policy
blocks, versus ~4.51 ms for GSQ. Therefore the 591/609 ms difference-in-
differences is interaction evidence, not pure causal router attribution.
There is not yet an efficient-router-at2045, unchanged-schedule arm.

Same-schedule repeats and instrumentation preserve the recorded endpoint and
state digests. Mixed scheduling changes arithmetic: GSQ endpoint relative L2
is 0.07834, max absolute logit difference 1.23585, top1 unchanged; the same
difference appears with the router disabled. One-token handoffs succeed but
their hashes differ. This does not make the speed result invalid, nor justify
promotion: test bounded distribution/loss and natural continuations before
changing scheduling. Router-only work is the narrower first move.

Profiles use the existing encoder-stage fallback. GSQ passes all four observer
checks (512 and the three4096 commands), raw timestamp coverage is essentially
one. Most UD observer checks fail and are retained as **unusable attribution**,
not silently discarded or used to infer stage shares. The unprofiled suffix
ABBA is separate from those observers.

Stage spans are inclusive. Fixed detailed layers5/7 are not all-cohort GSQ
measurements: layer5 uses IQ2_S gate/up and IQ4_NL down. Whole-layer timings
include HC, mixer and MoE; labeling every GDN layer's elapsed time "recurrence"
would be wrong. The diagnostic records parent/child identity explicitly.

## Placement and measurement boundaries

Normal GPU leases and memory admission were used; no gate changes, memory
limit raises, manual cache flushing or concurrent benchmark work. Timing runs
omit Metal API validation. Flash additionally admits its ~223 MB checkpoint
plus16 MiB CPU margin. No production behavior changes are present.

GLM's first N32 took 80.46 s in packet1 versus 1.36 s in packet2. GSQ's first
N512 took 39.76 s wall versus 1.00 s GPU; warmed N512 took 1.007 s wall. This
outside-GPU cost remains a real cold/lapsed-residency problem, not a router
or recurrent-kernel budget. This packet does not causally separate I/O,
wiring and compilation, or measure client TTFT.

GLM full-prefill wall includes encoding, wait, route checks and final logits
copy, excluding session construction. Flash reports sums of private executor
intervals, excluding between-command diagnostic work, checkpoint/restore,
hashing and the handoff. Neither is an HTTP TTFT measurement. No candidate
full4096 request speedup has been timed here.

## Reproduce / provenance

The ignored diagnostics live in:

- `glm5_next_metal::packed::router_prefill::router_prefill_abba`
- `qwen4exp_runtime::tests::prefill_map::native_prefill_map`

Their source headers contain invocations; output paths must be new files.
Compile `cargo test --release -p qwen-llm --lib --no-run`, then run only the
named test with `--exact --ignored --nocapture --test-threads=1`. They acquire
the production lease themselves. Use `QWEN_METAL_LEASE_WAIT=1` to wait.

GLM packets1/2 have the v1 source binding without whole-command GPU timestamps;
packet3 is v2. Keep them distinct. Flash packets bind the compiled test binary,
source inputs/metallib, artifact stamps and prompt IDs. All raw attempts remain
in JSONL, including first-use and invalid observer results. `complete` means
execution completed, not that a candidate passed a promotion gate.

Run `uv run docs/bench/2026-10-07-glm-flash-prefill-map/summarize.py FILE...`
to recompute timing, utilization and profile-validity summaries. The six data
files cover three GLM router packets, two Flash packets and the ragged suite.

Read-only research/design jams: `cx` sessions
`01a11893-3352-79d3-8d8d-8343c0d8f293` (GLM) and
`01a11893-3352-75b1-9ec7-2ed940154a10` (Flash).
Independent adversarial data/code review:
`01a118ca-c0f6-7c00-a0dd-23ae962883a5`. Its summary-context and causal-claim
findings are incorporated here. These sessions completed with 30-minute tool
allowances after the initial 5-minute interruptions.
