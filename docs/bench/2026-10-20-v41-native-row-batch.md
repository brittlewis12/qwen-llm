# V4.1 Native Expert Row Batching

Source candidate `fa40b081` batches eligible native expert GEMV over token/slot
while preserving the existing per-output arithmetic. The common baseline is
`abefb0af`, including the shared Metal 4/TensorOps infrastructure. V4.1 remains
NativeF32; this change does not select that infrastructure's F16 witness.

## Implementation

- Native projection dispatch gains a token depth axis. The shader's arithmetic
  body, decoder calls, lane ownership, FMA order, SIMD reduction, clamp and
  SwiGLU remain unchanged. Gate and up are still separate passes with an F32
  intermediate store. The single-token ABI and six buffer bindings are retained.
- Eligible fallback chunks issue six routed-stage dispatches instead of six
  per token: route, gate, up/SwiGLU, weight, down, ordered sum plus shared output.
  Existing row scratch is reused; no buckets or extra buffers are added for
  this path. Weight-before-down and shared-last addition remain unchanged.
- Existing ten grouped layers remain on their prior path. IQ3_XXS/Q3_K gate/up
  and their existing IQ3_XXS/IQ3_S/Q3_K/MXFP4 down cohort enable 15 native-batched
  layers in Q2 and 25 in IQ3. The remaining 15/five routed layers still use the
  per-token fallback. No IQ2_XS/IQ2_XXS batching is enabled.
- Shared wrappers preflight shapes, extents, alignment, writability, overlap,
  device ownership and both gate/up pipelines before either dispatch. FFN
  composition checks host and GPU-u32 products before encoding or allocation.
- Neither the rejected grouped-MMA gate/up change nor its compensated MXFP4
  arithmetic repair is included. The native MXFP4 arithmetic stays unchanged.

The prior [MMA rejection and router diagnostic](2026-10-20-v41-grouped-native-f32-rejection.md)
remain part of the evidence; their passing frozen comparisons did not establish
live full-model equivalence.

## Numerical Qualification

Private baseline `1cfb311e` and native candidate `3e9d84f1` use the same
caller-owned artifacts, assumed immutable: Q2_K_XL shards 1-6 with the shared
Q6_K Engram shard, and the complete IQ3_XXS set. The default wired limit stays
unchanged. Runs use
Metal validation, required Metal tests, and the existing production lease.

Clean expected build/runtime source identities, compiler, SDK, features, flags
and device descriptions are checked. Both use Metal compiler `32023.921` and
SDK 27.0. The capture hook moves mechanically after the new common row router;
capture bodies, cases, inputs, capacities, admission and completion logic remain
baseline-identical. This adaptation is reviewed explicitly; its patch IDs are
not claimed identical. Private comparison/frozen extensions do not land in
production. No runtime model hashes are computed.

Each quant covers 22 cases: seven-token-pattern and fixed-random inputs at
lengths 1/7/31/65/127/128/129/255/256/257/1024, with three teacher-forced
continuations. Capacities are prompt length plus 3/19; prefill chunks are 256.
The fixed-random 257 case also checks observational capture, snapshot restoration
and capacity growth. All **98 full-vocabulary logit files and all 22 complete
route ID/weight captures match the pre-change baseline bitwise**, including
signed zeros. The unchanged distribution contract is additionally required.
This is same-schedule agreement, not arbitrary whole-model partition invariance.

Real-bank replay uses saved baseline normalized inputs, selected routes, weights
and shared outputs at layers 15/25/30/35/38, positions 0-256, in 256+1 chunks:

| Quant | Scalar-native chunks | Native-batched chunks | Unchanged grouped controls |
| --- | ---: | ---: | ---: |
| Q2 + Q6 | 10 | 8 | 2 |
| IQ3 | 10 | 10 | 0 |

All activation, weighted activation, down and final-before-HC arrays match
the correct saved pre-change stages bitwise. Q2 layer 15 is explicitly an
unchanged grouped control, compared against saved grouped outputs rather than
native outputs. Native replay has five dispatches because its routes are frozen;
the production composition has six. Native census checks exact dtype pipelines,
grids and thread counts, without buckets/MMA. Independent sampled CPU-codec/F64
down checks and weight/sum invariants also pass. This is conditional operator
evidence, not a replacement for the live-routing model comparisons.

M5 operator/FFN validation reports 22 passed, zero failed, with two explicit
capture/replay fixture entrypoints ignored by default. Local saved pre-change
fixture replay passes separately. Full contracts and release workspace pass at
`fa40b081`, tree `746773d9c1e2545f107a87986dee93b61fc3cafa`: 3,129 passed,
zero failed, 651 ignored, excluding four nested filtered subprocess summaries.
This does not establish fresh GGUF task quality or large-context admission.

## Matched Throughput

Production builds without private instrumentation compare `abefb0af` against
`fa40b081` on metagross, M5 Ultra 80-core, macOS 27.0.1. Arm order is baseline,
candidate, candidate, baseline. Quant order alternates Q2/IQ3 and IQ3/Q2 between
blocks. Each quant/block runs separate fresh processes for the PP shape suite
and populated decode, with normal model/row warmups and three measured repeats.
There are six samples per arm/quant/shape, nested within two process blocks.

PP31/257/1024/2048 share capacity 2048; TG32 uses a 2048-token prefix and capacity
2080. Seed is 1, prefill chunk 256, NativeF32, default wired limit. Decode uses
the normal varying repetition seeds. Metal validation is unset for timing;
the production lease waits normally. Clean matching source/build identities,
empty diagnostic feature sets, and matching compiler/SDK are verified. The only
QWEN environment setting is the lease-wait control.

Values are mean tokens/s +/- sample SD. Percentage is candidate mean divided by
baseline mean minus one; these are observed warmed request-path measurements.

| Quant | Shape | Baseline | Native Rows | Change |
| --- | --- | ---: | ---: | ---: |
| Q2 + Q6 | PP31 | 50.008 +/- 0.039 | 70.140 +/- 0.055 | +40.26% |
| Q2 + Q6 | PP257 | 56.288 +/- 0.041 | 83.652 +/- 0.067 | +48.62% |
| Q2 + Q6 | PP1024 | 54.920 +/- 0.021 | 80.803 +/- 0.021 | +47.13% |
| Q2 + Q6 | PP2048 | 53.770 +/- 0.023 | 78.411 +/- 0.018 | +45.83% |
| Q2 + Q6 | TG32 @ 2048 | 21.477 +/- 0.088 | 21.598 +/- 0.030 | +0.56% |
| IQ3 | PP31 | 49.063 +/- 0.028 | 93.430 +/- 0.087 | +90.43% |
| IQ3 | PP257 | 54.263 +/- 0.038 | 116.924 +/- 0.088 | +115.48% |
| IQ3 | PP1024 | 52.998 +/- 0.015 | 111.677 +/- 0.052 | +110.72% |
| IQ3 | PP2048 | 51.950 +/- 0.016 | 106.818 +/- 0.681 | +105.62% |
| IQ3 | TG32 @ 2048 | 21.043 +/- 0.028 | 21.057 +/- 0.031 | +0.07% |

Both adjacent order-balanced comparisons reproduce the prefill improvement.
For PP1024 the paired gains are +47.08%/+47.18% for Q2 and +110.73%/+110.71%
for IQ3. PP2048 gains are +45.76%/+45.89% and +105.05%/+106.18%, respectively.
Decode shows no observed regression; its small differences are not treated as
a meaningful decode optimization. No OS thermal/performance warning is recorded,
which is not a clock measurement. The IQ3 PP2048 variability remains reported.

These canonical rows contain no scoped physical-I/O counters, so they are not
isolated GPU-time or bandwidth measurements. Repetitions within a process are
not independent fresh-process trials. Compare this matched baseline, not the
earlier PP1024 result with capacity 1024. The larger IQ3 gain covers more eligible
layers, but this is not an isolated quantization or model-quality comparison.

Evidence root: `~/qwen-runs/2026-10-20-v41-unsloth-profile/`, with
`numerical-baseline-1cfb311e/`, `numerical-native-3e9d84f1/`, and
`native-fa40b081-operator-validation/`. Timing evidence is in
`canonical-native-abba-fa40b081/`, including all block JSON/logs, process and
thermal snapshots, build identities, the protocol and derived summary. Raw
outputs, manifests, censuses, comparison reports and failures remain retained.
