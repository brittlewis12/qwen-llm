# V4.1 Unsloth Bottleneck Baseline

Production source and installed binaries: `10a57719`, M5 Ultra 80-core,
macOS 27.0.1. Both artifacts use the same Q6_K disk Engram table:
`UD-Q2_K_XL` shards 1-6 plus `UD-IQ3_XXS` shard 7, and the complete
`UD-IQ3_XXS` set. The first is deliberately not the published Q5_K Q2 set.
No wired-memory setting or arithmetic precision changes; `iogpu.wired_limit_mb`
remains zero. Metal-resident weights are 199,616,856,064 and 233,447,129,088
bytes, respectively. Both pass validated cold/repeated/shared-prefix serving
smokes; this does not establish task quality or large-context admission.

## Canonical Throughput

Quant order is Q2, IQ3, IQ3, Q2. Each block runs separate fresh processes for
PP1024 and TG32 after a 2048-token prefix, three measured repetitions after
normal model/row warmups. That gives six samples per quant/workload. Capacities
are 1024 and 2080, respectively; prefill chunks are 256. Seed is 1; the normal
decode harness changes generated-token seeds per repetition. Metal validation
is unset. All rows have clean matching identity, no QWEN overrides, and no
recorded OS thermal/performance warning. Absence of warnings is not a clock
measurement. Values are tokens/s; SD is sample standard deviation.

| Quant | Workload | Mean +/- SD | Range |
| --- | --- | ---: | ---: |
| Q2 + Q6 | PP1024 | 55.288 +/- 0.021 | 55.264-55.321 |
| IQ3 | PP1024 | 53.259 +/- 0.067 | 53.190-53.326 |
| Q2 + Q6 | TG32 @ 2048 | 21.532 +/- 0.101 | 21.384-21.619 |
| IQ3 | TG32 @ 2048 | 21.048 +/- 0.055 | 20.988-21.141 |

These are observed production request-path rates, not isolated GPU timing.
Canonical rows have no per-window physical-I/O counters. Samples within one
process are not independent fresh-process trials. This is neither a long-context
result nor a quality comparison.

## Accepted Prefill Attribution

Private `634cdc97` changes only the benchmark adapter. Five identical PP1024
calls use fresh states: model warmup, one fixed instrumented warmup, control,
profile, control. Stage timestamps and dispatch census run on calls 2 and 4;
only call 4 is attribution. All raw calls are retained. Acceptance requires
zero observed process reads/page-ins for calls 3-5, finite full-vocabulary
logits, bitwise endpoint parity, complete valid timestamps/census, and no more
than 10% profile inflation against the mean controls.

**Q2 is accepted:** controls 18,478.022 and 18,501.446 ms; profile 18,514.932 ms,
0.136% above their mean. All three have zero scoped reads/page-ins and nominal
Foundation thermal state. Metal allocation stays at 199,641,268,224 bytes.

| Q2 measurement | Milliseconds | Fraction of profile wall |
| --- | ---: | ---: |
| Composite FFN GPU intervals | 15,971.862 | 86.26% |
| Composite attention GPU intervals | 2,306.595 | 12.46% |
| Enclosing command GPU intervals | 18,278.623 | 98.72% |
| CPU encoding | 103.640 | 0.56% |

FFN includes HC, router, shared/routed experts and the endpoint head. Attention
includes HC and Engram GPU projection/gating. CPU Engram preparation, uploads,
allocation and endpoint copying are not individually attributed. Enclosing
command time must not be added to stage time. The stage profiler changes encoder
segmentation; low whole-forward overhead does not prove unbiased individual
stage durations. The actual census has 239,428 dispatches, not per-kernel times.

### Rejected Captures Remain Rejected

The earlier `2cb50520` protocol had no instrumented warmup. Q2 call 4 takes
33,268.611 ms versus control 18,461.387 ms, with 3,270 page-ins and 110,510,080
physical-read bytes. IQ3 takes 52,645.457 versus 19,228.641 ms, with 2,666
page-ins and 52,895,744 read bytes. Both stop before control 5 and are rejected.
First-use instrumentation is a hypothesis, not an established cause.

Under the fixed warmed protocol, IQ3 still rejects call 4: 2,455 page-ins and
101,167,104 read bytes, despite 19,254.074 ms being near the preceding
19,213.290-ms control. There is no completed bracket, and its stage percentages
are not accepted. No gate is relaxed and no retry-until-pass selection is used.

## Accepted Populated Decode Attribution

Private `9e163c52` uses the same depth/capacity and a fixed seed-3 TG32 sequence
for the populated warmup and all three measured windows. Only the middle window
uses `Scope::begin()`: normal one-encoder/one-command-per-token execution remains
unchanged, with no stage splitting or census. All 32 vocabulary-wide outputs
match the warmed reference bitwise. Both brackets have zero reads/page-ins,
nominal thermal state, constant Metal allocation, and 32 valid GPU submissions.

| Quant | Controls, ms | Profile, ms | Profile overhead | GPU, ms | CPU encode, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2 + Q6 | 1323.174 / 1323.468 | 1324.725 | 0.106% | 1279.842 | 23.311 |
| IQ3 | 1358.827 / 1356.291 | 1358.877 | 0.097% | 1313.404 | 23.517 |

GPU command execution occupies approximately 96.6% of these warmed windows;
CPU encoding is approximately 1.8%. Host-side cleanup alone cannot supply a
large gain here. This does not establish bandwidth saturation or identify the
dominant GPU operator. Timers overlap and are not additive. The repeated-input
diagnostic is not interchangeable with canonical decode's varying seeds. Its
row warmups read 25,182,208 bytes each; their measurements remain in the logs.

## Optimization Priority

The accepted Q2 measurement prioritizes the FFN stage. Source/inventory analysis
finds only 10/40 routed layers grouped in each artifact; the other 30 loop the
routed path per token. All gate/up pairs match in dtype and satisfy geometry.
Missing grouped gate/up support, not mixed storage, blocks them. Shared/router/HC
work remains batched. Single-token decode uses native per-token kernels throughout.

Reusing existing F32 grouped decoders for IQ3_XXS and Q3_K gate/up would raise
eligibility to 25/40 layers for Q2 and 35/40 for IQ3. Eligibility is not a speedup
prediction: bucket occupancy, tile utilization, and newly reachable down paths
need qualification. Preserve weight-before-down, asymmetric clamp, routing/slot
order and full F32 operands. Independent codec/F64 tests and same-artifact
numerical/performance comparisons must precede promotion. This is separate from
the parallel Metal 4.1 tensor infrastructure and its F16 witness.

Evidence root: `~/qwen-runs/2026-10-20-v41-unsloth-profile/`.
Canonical rows/logs are `canonical-10a57719/block-{1..4}-*.{json,log}`;
original captures are `prefill-2cb50520/{q2,iq3}.log`; warmed captures are
`prefill-warm-634cdc97/{q2,iq3}.log`; decode captures are
`decode-9e163c52/{q2,iq3}.log`. Each diagnostic directory retains build identity,
tests, process snapshots and exit codes. Private diagnostics do not land in
production. No optimization gain or fresh task-quality result is claimed here.
