# V4.1 Complete Native Rows And Practical Grouped Acceptance

2026-10-20. Candidate `1388793dc6bce0de79aaf2df136085f47c02e9a3`, tree
`27a5974cf3cb8c67f330f42e273cf0e7e57c1c51`. Supplements the existing
`docs/bench/2026-10-20-v41-grouped-tradeoffs.md` checkpoint. The objective is
engineering throughput with bounded practical output evidence, not universal
baseline identity or an all-shapes no-worse gate.

Raw evidence lives on metagross under
`~/qwen-runs/2026-10-20-v41-unsloth-profile/`; the named evaluation directories
also have local `/tmp/` metadata copies. Q2 is shards 1-6 plus the hardlinked
IQ3 Q6_K Engram shard, not Unsloth's published Q5_K-ended artifact. Both use
NativeF32 with default `iogpu.wired_limit_mb=0`.

## Source And Checks

Relative to `a86fe231`, production edits only expand native-row capabilities
and select the FFN strategy; remaining changes are tests and checkpoint docs.
IQ2_XS/IQ2_XXS gate/up, including mixed eligible types, and IQ2_XXS down reuse
existing native kernels. IQ2_XS down remains excluded. No shader arithmetic
changes or grouped IQ2 MMA are introduced. Existing native coverage remains.
For otherwise-grouped same-type IQ3_XXS/Q3_K gate/up, chunks below 128 choose
native rows only when native down and alignment support also hold; eligible
chunks at/above 128 stay grouped. Other grouped cohorts retain their selection.
Both row-FFN entrypoints use this speed selector, not a precision fallback.
Single-token APIs/scalar decode, checked extents, weight-before-down and ordered
summation are unchanged. No production switch, F16 substitution or TensorOps
backend is added; force-grouped selection is test-only.

Full checks exited 0. The contracts footer identifies the exact tree above.
The summary record is `/tmp/v41-complete-rows-1388793d-workspace-summary.json`.

| Check | Passed | Failed | Ignored | Included result summaries |
| --- | ---: | ---: | ---: | ---: |
| Release workspace, including doctests | 3,151 | 0 | 651 | 26 |
| Contracts' Rust test invocations | 172 | 0 | 0 | 3 |
| M5 native/FFN/grouped targeted tests | 55 | 0 | 2 | 3 |

Four nested one-pass subprocess summaries at workspace log lines
2857/2864/2871/2878 are excluded; naive summation gives 3,155. The final
compile-fail doctest is included. Contracts' 10+161+1 passes overlap workspace
checks and are not additional unique tests. Versus `a86fe231`: +6 passes,
unchanged failures/ignored. Added IQ2 cohort, mixed-type, status/guard, CPU-codec/
F64, preflight and strategy witnesses pass; extended real-width witnesses pass.
These are fixture checks, not universal full-model equivalence. Both logs note
the unsupported Metal 4.1 compiler probe and skipped tensor-op tier.
Astra's `1388793d` source/testing review reports no actionable blocker
(`job_e1761d24-1ced-4154-bfe5-6918cbae9143`). The earlier retained grouped change
and private quality helpers also have completed Astra reviews; helper spawn and
grader-infrastructure findings were fixed and reviewed closed before execution.

Logs: `/tmp/v41-complete-rows-1388793d-contracts-full.log` and
`/tmp/v41-complete-rows-1388793d-workspace-full.log`.

## Completed Small-Shape Throughput

Fresh root: `/tmp/v41-complete-rows-evaluation-1388793d` (F).
Prior root: `/tmp/v41-grouped-current-evaluation-a86fe231` (P).
Inputs: F's `{q2,iq3}-short-shapes.json` and P's
`{native,grouped}-{q2,iq3}-short-shapes.json`. Native means `7561117e`,
grouped baseline means `a86fe231`, candidate means `1388793d`.
Capacity 257, prefill chunk 256, three measured repetitions per shape in one
process block per arm/quant: exploratory, not ABBA or independent trials.
All 34 rows have clean matching build/runtime commits and source-state stamps,
no identity problems/overrides, empty features, matching Metal compiler 32023.921
and SDK 27.0. Model paths/sizes agree within each quant. Recorded QWEN settings
contain only lease wait. These checks establish recorded identity consistency,
not model-content hashes or independent control of thermal/clock conditions.

Mean tokens/s +/- within-block sample SD; changes use unrounded means:

| Quant / PP | Native 756 | Grouped a86 | Candidate 138 | vs a86 | vs 756 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2 / 7 | 53.179 +/- 0.204 | 44.933 +/- 0.207 | 73.314 +/- 0.051 | +63.16% | +37.86% |
| Q2 / 31 | 70.009 +/- 0.116 | 65.770 +/- 0.023 | 116.792 +/- 0.303 | +77.58% | +66.82% |
| Q2 / 65 | 71.951 +/- 0.021 | 70.506 +/- 0.026 | 122.404 +/- 0.138 | +73.61% | +70.12% |
| Q2 / 129 | 74.224 +/- 0.063 | 76.570 +/- 0.092 | 136.076 +/- 0.130 | +77.72% | +83.33% |
| Q2 / 257 | 83.731 +/- 0.011 | 89.982 +/- 0.012 | 182.634 +/- 0.262 | +102.97% | +118.12% |
| IQ3 / 7 | 63.941 +/- 0.081 | 47.680 +/- 0.228 | 72.025 +/- 0.234 | +51.06% | +12.64% |
| IQ3 / 31 | 93.223 +/- 0.178 | 84.867 +/- 0.116 | 113.411 +/- 0.286 | +33.63% | +21.66% |
| IQ3 / 65 | 95.884 +/- 0.028 | 97.156 +/- 0.190 | 117.633 +/- 0.077 | +21.08% | +22.68% |
| IQ3 / 129 | 95.280 +/- 0.183 | 107.638 +/- 0.059 | 135.857 +/- 0.085 | +26.22% | +42.59% |
| IQ3 / 257 | 116.804 +/- 0.074 | 148.415 +/- 0.249 | 207.188 +/- 0.240 | +39.60% | +77.38% |

Fresh-only PP127/128: Q2 119.353 +/- 0.072 / 158.023 +/- 0.088;
IQ3 114.722 +/- 0.022 / 169.765 +/- 0.053 tokens/s. No prior 127/128 controls
exist here: this records a curve, not a causal estimate or an optimal threshold.
All common shapes improve in these measurements, but IQ2 batching also changed;
the gains cannot be assigned to the threshold alone. PP257 includes a 256-row
chunk and singleton tail. Capacity-257 rates do not substitute for PP1024/2048
or populated-decode results, and small within-block SD is not campaign uncertainty.

## Completed Task And Likelihood Evidence

F's `task-comparison.json` compares `a86fe231` with `1388793d`: 22 paired cases
per quant, both arms 9/22 (code 8/10, reasoning 1/8, JSON 0/4), all 22 visible
outputs equal, no gains/losses. All attempted; 21 complete and `r1` truncates
in each arm. Raw result files independently confirm the paired text/outcomes;
grader and adapter hashes match. One greedy generation/task/arm, effort none,
seed 42, context 4096, cache 0 MiB, Metal validation, unchanged task caps.
The earlier P comparison of `7561117e` to `a86fe231` has the same paired counts.
These are bounded screens, not strong absolute quality or independent new tasks.

Likelihood source: `/tmp/v41-grouped-text-likelihood-0379a30d-53868c55/likelihood-comparison.json`.
Identical additive helpers `0379a30d` / `53868c55` compare production bases
`7561117e` / `a86fe231`: six preselected documents (three prose, three code),
512-token prefix + 64 real continuation targets each, capacity 576, chunk 256,
NativeF32, teacher forcing, F64 logsumexp over full F32 vocabulary. No chat
rendering or sampling. Same caller-owned artifacts, without model hashes.

| Quant | Native mean NLL | a86 mean NLL | a86 minus native | Correct targets, both | Max abs logit difference | Max TV | Max KL(native || a86) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2 | 2.160671487645 | 2.160671733098 | +2.454537245e-7 | 214/384 | 1.404285431e-4 | 4.777855159e-6 | 6.957781014e-11 |
| IQ3 | 2.146362159822 | 2.146361600884 | -5.589380083e-7 | 222/384 | 8.702278137e-4 | 5.692393118e-6 | 8.304303764e-11 |

NLL is mean nats/target, equally weighted by document (64 targets each).
Document-mean NLL deltas span -5.765830e-8 to +7.170466e-7 (Q2) and
-2.206854e-6 to +1.447194e-7 (IQ3); both signs occur. Neither quant has top-1
disagreements or target-correctness gains/losses. Tokens are correlated; these
six documents are descriptive evidence, not a comprehensive quality sample.

F's `likelihood-bitwise-comparison.json` reports pass: all six documents and
384 full-vocabulary rows per quant are bitwise equal for fresh 138 versus a86.
Thus the a86 likelihood values also describe 138 on these checked cases, not a
new independent NLL campaign. Both full prefill chunks use grouped selection;
this supports IQ2 row arithmetic preservation here, NOT the changed small-chunk
policy. The comparison ran against complete raw logit files on metagross;
the local copy is its summary report.

## Retained Numerical Differences

The original `docs/bench/2026-10-20-v41-grouped-native-f32-rejection.md` remains:
retained grouped prototype versus native baseline failed 4/98 logit files for
Q2 + Q6 and 13/98 for IQ3, with selected-ID divergence in 4/22 and 3/22 cases.
Maximum abs-logit errors: 0.0072698593 / 0.0016160011; TV: 5.3055e-4 /
1.2316e-4; KL: 8.7708e-7 / 4.5434e-8. No top-1 failure was reported.
Repeated observer/restored/grown endpoints are not independent trials. Retained
boundary-tie diagnostics do not establish the earliest arithmetic cause.
Neither fresh 138-vs-a86 bitwise cases nor task matches erase those historical
baseline-equivalence failures; those failures alone do not prove worse task quality.
Q2 uses the retained Q6_K Engram artifact, not the published Q5_K-ended setup.

## Large-Shape Throughput

Completed M5 Ultra comparison: `a86fe231` versus `1388793d`, PP1024/2048 at
common capacity 2048, TG32 after depth 2048 at capacity 2080, chunk 256, seed 1.
Arm order is A/B/B/A with alternating quant order and three repetitions per
process, two process blocks per arm. A1 reuses the earlier matching a86
confirmation; other experiments intervened before B2. This is not an
uninterrupted ABBA campaign. Timing has validation unset and lease wait enabled;
build/runtime identities and compiler/SDK/features match. F's `summary.json`
retains all 72 samples, block means and adjacent comparisons.

Mean tokens/s +/- sample SD across the six repetitions per arm:

| Quant / workload | Grouped a86 | Complete rows 138 | Mean gain | Two order-pair gains |
| --- | ---: | ---: | ---: | ---: |
| Q2 / PP1024 | 87.252 +/- 0.025 | 162.796 +/- 9.073 | +86.58% | +96.12%, +77.05% |
| Q2 / PP2048 | 84.501 +/- 0.121 | 148.453 +/- 11.481 | +75.68% | +87.25%, +64.14% |
| Q2 / TG32 | 21.569 +/- 0.015 | 21.530 +/- 0.096 | -0.18% | +0.12%, -0.48% |
| IQ3 / PP1024 | 141.331 +/- 0.106 | 185.029 +/- 2.853 | +30.92% | +32.32%, +29.52% |
| IQ3 / PP2048 | 134.355 +/- 0.036 | 170.613 +/- 8.607 | +26.99% | +31.47%, +22.51% |
| IQ3 / TG32 | 21.067 +/- 0.037 | 20.990 +/- 0.056 | -0.37% | -0.39%, -0.34% |

The prefill gain persists in both orders, but candidate drift is substantial
and is not discarded. Q2's PP1024 block means fall from 171.079 to 154.514;
its PP2048 samples span 160.921 to 133.794. IQ3's PP2048 samples span 180.510
to 158.288. Baseline rates remain much steadier. No recorded OS thermal or
performance warning appears, which is not a clock measurement or an explanation.
The pre-launch process snapshots show no other model/compiler job; they do not
prove exclusive execution for the complete intervals. Canonical rows have no
scoped physical-I/O counters. Thus these are observed warmed request rates, not
sustained peak rates, isolated GPU costs or independent statistical trials.
Decode changes are small negative observations, not a claimed decode gain.

## Remaining Prefill Costs

Private `28ac34ed` adds the byte-identical earlier layer observer to `1388793d`.
Evidence: `complete-rows-layer-attribution-1388793d-28ac34ed/`. Both quant captures
pass: fixed PP1024/capacity1024, model warmup, instrumented warmup, then
control/profile/control. All measured calls have zero process reads/page-ins;
full-vocabulary observer endpoints match bitwise. All 326 operation records
are valid. Aggregate and layer views are alternatives, not additive.

| Quant | Profile wall ms | Attention GPU ms | Composite FFN GPU ms | IQ2-cohort FFN ms | Dispatches | Observer inflation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2 | 6005.537 | 2278.457 (37.94%) | 3558.937 (59.26%) | 1961.938 (32.67%) | 55,828 | +0.123% |
| IQ3 | 5306.387 | 2277.035 (42.91%) | 2861.607 (53.93%) | 691.990 (13.04%) | 55,828 | +0.250% |

The earlier identical observer on grouped `a86fe231` measured IQ2-cohort FFN
at 7672.481 ms for Q2 and 2632.112 ms for IQ3. The other FFN cohorts and attention
remain near their earlier costs. These are separately collected fixed-input
profiles, not a new interleaved throughput campaign or pure expert-kernel timing.
Attention includes Engram; final-chunk layer39 FFN includes head/logits.

Control walls are 5994.675/6001.660 ms for Q2 and 5292.791/5293.524 ms for IQ3.
This bounded, fixed-input 1K bracket is stable, but does not explain the
varying-input, capacity-2048 campaign's drift. Warmup reads remain in the raw
record and are excluded only according to the predeclared warm-attribution
protocol. The updated profile makes attention and grouped FFN substantial
targets; the former IQ2 residual is no longer a per-token dispatch fallback.

## Engineering Decision

Retain grouped NativeF32 and complete native IQ2 row batching: both order-pair
prefill comparisons show useful gains, practical task outcomes are unchanged,
held-text likelihood changes are tiny with both signs, and the IQ2 extension
preserves arithmetic on the checked fixtures and full-vocabulary cases. Baseline
route identity is not a universal acceptance requirement. Bitwise checks remain
appropriate for observer invariance and explicit arithmetic-preservation claims.
The performance selector below 128 rows avoids measured small-chunk overhead;
it is not a slow-path correctness guard.

The observed drift remains unresolved and must accompany the rates; no stable
peak, comprehensive quality, default-model, long-context or all-shapes
no-regression claim follows. Maintained-build/install identity and bounded
cache smokes are recorded separately in `delivery-<commit>` evidence directories;
they do not relabel the measured source revisions above.
