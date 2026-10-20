# V4.1 Grouped NativeF32: Speed And Output Tradeoffs

This experiment evaluates the retained grouped implementation against the
landed native-row batching, rather than treating baseline route identity as
the only useful numerical question. The [earlier equivalence rejection](2026-10-20-v41-grouped-native-f32-rejection.md)
remains valid evidence: the implementations are not baseline-equivalent under
that contract. It does not establish that the grouped implementation has worse
task accuracy.

## Source And Scope

Production candidate `a86fe231` carries the retained `f1708acd` grouped kernels
onto native-row baseline `7561117e`. The shared shader and grouped wrapper are
byte-identical to that retained prototype. Current native APIs, checked FFN
composition extents, mixed-type fallback and scalar decode remain intact.
New same-type IQ3_XXS/Q3_K gate/up paths take grouped precedence. No IQ2 batching,
F16 operand substitution, TensorOps backend or extra production runtime switch
is introduced.

The MXFP4 exponent-0/1 compensation remains included. Its range and cancellation
witnesses do not promise uniform relative accuracy for arbitrary ill-conditioned
dots, ordinary-path overflow/FTZ, or subnormal final outputs.

Exact `a86fe231`, tree `984883663b4c3cec757f3af164dd98845311fe5c`, passes full
contracts and the release workspace on the local M4: 3,145 passed, zero failed,
651 ignored, excluding four nested filtered subprocess summaries. Astra source
review finds no actionable source blocker. These checks are not a fresh task
quality result or a claim about every execution shape.

## Matched Throughput

The initial timing campaign compares `7561117e` against retained `f1708acd` on
metagross, M5 Ultra 80-core. The current-source port is evaluated separately;
do not silently relabel the retained prototype's measurements as `a86fe231`.

Arm order is native, grouped, grouped, native. Quant order alternates Q2/IQ3 and
IQ3/Q2 between blocks. Each quant/block runs separate fresh processes for PP and
populated decode, with normal warmups and three measured repetitions. Each
reported arm has six samples nested within two process blocks.

PP1024/2048 share capacity 2048. TG32 follows a 2048-token prefix at capacity
2080. Seed is 1, prefill chunk 256, NativeF32 and the default wired limit.
Metal validation is unset for timing; the existing production lease waits
normally. Clean build/runtime identities, compiler/SDK and empty diagnostic
feature sets match. The only QWEN environment setting is lease wait.

Values are mean tokens/s +/- sample SD:

| Quant | Shape | Native Rows | Grouped | Change |
| --- | --- | ---: | ---: | ---: |
| Q2 + Q6 | PP1024 | 81.056 +/- 0.065 | 87.289 +/- 0.071 | +7.69% |
| Q2 + Q6 | PP2048 | 78.689 +/- 0.059 | 84.587 +/- 0.052 | +7.49% |
| Q2 + Q6 | TG32 @ 2048 | 21.595 +/- 0.023 | 21.586 +/- 0.018 | -0.04% |
| IQ3 | PP1024 | 111.901 +/- 0.021 | 141.383 +/- 0.100 | +26.35% |
| IQ3 | PP2048 | 107.107 +/- 0.566 | 134.390 +/- 0.030 | +25.47% |
| IQ3 | TG32 @ 2048 | 21.059 +/- 0.033 | 21.069 +/- 0.018 | +0.05% |

Both adjacent order-balanced comparisons agree: Q2 PP1024 gains are
+7.55%/+7.83%; IQ3 gains are +26.40%/+26.29%. Decode is effectively unchanged,
not a meaningful gain or regression. No OS thermal/performance warning is
recorded; that is not a clock measurement. The native IQ3 PP2048 spread remains
reported rather than discarded.

These are warmed request-path rates, not isolated GPU timing or bandwidth.
Canonical rows have no scoped physical-I/O counters. Repetitions inside a
process are not independent trials. Q2 uses its retained Q6_K Engram hardlink,
not the published Q5_K-ended configuration. No long-context admission or
cross-artifact task-quality conclusion follows from this table.

## Where Time Goes

Private baseline `b5098e0e` and grouped `6f81b9c6` apply identical profiling
helpers to `7561117e` and `a86fe231`. They preserve the warmed control/profile/
control protocol and retain the already collected per-layer timestamps instead
of discarding layer identity during CPU reporting. No additional GPU samples,
encoders, synchronization or arithmetic are introduced by that reporting change.

All four PP1024 profiles are admitted: five expected calls, finite complete
outputs, bitwise observer agreement, zero scoped physical reads/pageins, valid
timestamps and bounded overhead. Capacity is **1024**, not the canonical
throughput suite's 2048. Layer and aggregate records are alternative views of
the same timestamps, not additive measurements.

Milliseconds per profiled PP1024:

| Quant / Arm | Forward Wall | Attention GPU | Composite FFN GPU | Profiling Inflation |
| --- | ---: | ---: | ---: | ---: |
| Q2 native | 12,642.156 | 2,287.277 | 10,153.946 | +0.127% |
| Q2 grouped | 11,755.794 | 2,286.520 | 9,268.954 | +0.160% |
| IQ3 native | 9,165.338 | 2,283.520 | 6,700.457 | +0.157% |
| IQ3 grouped | 7,260.513 | 2,279.801 | 4,801.163 | +0.211% |

Composite FFN milliseconds, classified by actual gate/up eligibility:

| Quant | Cohort | Layers | Native Rows | Grouped |
| --- | --- | ---: | ---: | ---: |
| Q2 | Existing grouped | 10 | 583.696 | 583.200 |
| Q2 | Newly grouped | 15 | 1,903.936 | 1,013.273 |
| Q2 | Residual IQ2_XS/IQ2_XXS | 15 | 7,666.314 | 7,672.481 |
| IQ3 | Existing grouped | 10 | 530.160 | 529.986 |
| IQ3 | Newly grouped | 25 | 3,541.197 | 1,639.066 |
| IQ3 | Residual IQ2_XS/IQ2_XXS | 5 | 2,629.100 | 2,632.112 |

The newly grouped cohorts account for the observed saving; attention and the
other cohorts remain essentially flat. Residual IQ2 cohorts occupy about 65%
of grouped Q2 wall time and 36% of grouped IQ3 wall time. Attention occupies
about 31% of grouped IQ3 wall time. These FFN intervals include router, shared
expert and hyper-connection work, not only routed projections. Attention includes
Engram GPU work, and final-chunk layer-39 FFN includes head/logits.

Dispatch counts remain 147,628 for Q2 and 86,428 for IQ3 across the two arms.
This improvement is not another reduction in launch count. The profiles do not
separate gate/up, down, bucket overhead or memory bandwidth.

An initial analyzer incorrectly treated Q3_K down storage as newly enabled even
when gate/up were already grouped. Its original `summary.json` is retained.
The table uses corrected `summary-gate-cohorts.json`, revision
`gate-up-cohorts-v2`: eligibility expansion concerns gate/up, not those already
supported down formats. Raw timestamps and total-stage results are unchanged.

## Evidence

Worker root: `~/qwen-runs/2026-10-20-v41-unsloth-profile/`.

- `grouped-tradeoff-pilot-f1708acd/`: all 16 timing JSON/log pairs, identities,
  process/thermal snapshots, protocol, copied binaries and derived summary.
- `grouped-layer-attribution-b5098e0e-6f81b9c6/`: all four profile attempts,
  controls, per-layer records, census, identities, original and corrected analyses.
- `v41-grouped-a86fe231-review-record.json`: source/helper review findings and
  closure of the task harness's grader-error and spawn-cancellation issues.
- `v41-grouped-a86fe231-*-full.log`: exact-source contracts and workspace checks.

Current-source short-shape timing, objective task outcomes and fixed-text
likelihood are separate experiments; the measurements above do not substitute
for their results.
