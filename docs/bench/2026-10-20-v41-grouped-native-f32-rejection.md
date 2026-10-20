# V4.1 Grouped NativeF32 MMA: Full-Model Rejection

The IQ3_XXS/Q3_K grouped gate/up prototype is not qualified for promotion.
Real-bank conditional replay passes, but live full-model routing and logit
comparisons fail. No timing campaign, installation, tolerance relaxation, or
task-quality claim follows from this candidate.

## Compared Trees

Both arms include the shared Metal 4/TensorOps foundation `abefb0af`; V4.1
remains NativeF32. Baseline private capture is `1cfb311e`, candidate private
capture is `6d981291`, and the candidate without instrumentation is `f1708acd`.
The identical private helper patch has stable patch ID
`0395e67095b0772a403eca7ac821c703d2dcee29`. The production candidate patch has
stable patch ID `4d792c6dbba995f36438f010c6b74fc43479aeb6`, unchanged from
the reviewed pre-TensorOps-foundation `429535bd`.

Clean build/runtime source identities, compiler, SDK, feature settings, flags,
and device descriptions agree as required. The worker uses Metal compiler
`32023.921`, SDK 27.0, Metal validation, and the existing production lease.
Each arm uses the same caller-owned artifact: Q2_K_XL shards 1-6 plus the Q6_K
Engram shard, or the complete IQ3_XXS set. No model hashes are recomputed.

## Passing Conditional Evidence

- Both captures complete all 22 cases: seven-token-pattern and fixed-random
  inputs at lengths 1/7/31/65/127/128/129/255/256/257/1024, each followed by
  three teacher-forced tokens. Capacity is prompt length plus 3 or 19,
  respectively. Prefill chunks remain 256.
- The fixed-random 257 case has bitwise observer, snapshot-restoration, and
  capacity-growth parity within each arm. These checks do not compare the two
  implementations against one another.
- Frozen replay consumes baseline normalized FFN inputs, shared outputs,
  selected IDs, and route weights for layers 15/25/30/35/38 at positions
  0-256. Both quants pass all ten chunk cases, including forced grouped
  singleton replay, with the expected dispatch census.
- Activation, weighted activation, down, and ordered sum-plus-shared satisfy
  the existing `4e-5 + 4e-5 * abs(reference)` elementwise bound. Candidate
  native replay is bitwise equal to saved baseline native replay. Grouped MMA
  is not bitwise equal; its maximum per-chunk relative L2 error is about
  `2.28e-6` across these stages.
- Independent CPU-codec/F64 down checks pass at the first/last chunk row,
  every selected slot, and three output channels. These are sampled down
  checks using each arm's weighted activation, not an independent full gate/up
  or whole-model oracle.

Maximum grouped-versus-baseline-native absolute error over the frozen cases:

| Quant | Activation | Weighted | Down | Final, before HC post |
| --- | ---: | ---: | ---: | ---: |
| Q2 + Q6 | 1.984e-4 | 9.155e-5 | 2.289e-5 | 2.241e-5 |
| IQ3 | 1.106e-4 | 6.866e-5 | 1.359e-5 | 1.407e-5 |

The repaired MXFP4 grouped edge reduction and fused clamp witnesses also pass
30 targeted M5 operator/FFN tests at `6155a9a3`. A test-only comparison-helper
fix then passes full contracts and the release workspace at `429535bd`, tree
`5cc1efccd7fdd4922bf1fe554cee3255ef1485ff`: 3,133 passed, zero failed,
649 ignored, excluding four nested filtered subprocess summaries. Those
checks precede carrying the unchanged patch onto `abefb0af`; they do not
qualify the full model or the later combined tree for installation.

## Blocking Full-Model Results

The existing full-vocabulary contract remains maximum absolute error `1e-3`,
total variation `1e-4`, KL(expected || actual) `1e-8`, and matching top-1.
There are 98 logit-file comparisons per quant, including observer/restored/
grown continuations; these repeated endpoints are not independent trials.

| Quant | Logit files failing bounds | Largest absolute error | Cases with selected-ID divergence |
| --- | ---: | ---: | ---: |
| Q2 + Q6 | 4/98 | 0.0072698593 | 4/22 |
| IQ3 | 13/98 | 0.0016160011 | 3/22 |

Q2's fixed-random 128 endpoint and all three continuations fail. Its largest
TV is `5.3055e-4` and KL is `8.7708e-7`. The first captured selected-ID
disagreement, ordered by position then layer, is position 13/layer 34:
expert 355 versus 354. Twelve differing slots recur in the shared-prefix
128/256/257/1024 cases. The 127/129/255 cases do not reproduce this disagreement;
the original request shape must be retained during diagnosis.

IQ3 fails selected endpoints/continuations in the fixed-random 256/257 cases,
including repeated observer/restored/grown continuations. Largest TV is
`1.2316e-4` and KL is `4.5434e-8`. Its first captured disagreement is position
133/layer 33: expert 252 versus 199. Eleven differing slots recur in the
256/257/1024 cases. No top-1 failure is reported, which does not rescue the
distribution or routing contract failures.

These are first observed selection disagreements, not established first
arithmetic causes. The router executes before that layer's expert projections.
The candidate changes both gate/up and newly reachable down reductions; passing
conditional replay does not isolate the upstream cause. A routing-boundary
crossing caused by small F32 differences is plausible, not yet established.
Same-slot weight differences after IDs change are not same-expert error metrics.

## Next Experiment

Retain this rejection and capture only the two known router divergence points
with their original request shape. Compare actual transformed GPU scores plus
bias, selection margins, and selected IDs, without freezing full-model routes.
The next candidate starts from `abefb0af` and batches existing native GEMV
over token/slot while retaining each output's FMA and reduction order, both
gate/up passes, weight-before-down, and ordered summation. Existing grouped
layers stay unchanged. IQ2_XS/IQ2_XXS batching and the compensated MMA arithmetic
repair remain outside that equivalence experiment. Bitwise native-stage and
same-schedule full-model agreement must precede timing.

Evidence root: `~/qwen-runs/2026-10-20-v41-unsloth-profile/`, directories
`numerical-baseline-1cfb311e/`, `numerical-candidate-6d981291/`, and
`grouped-6155a9a3-operator-validation/`. Raw full-vocabulary logits, routes,
frozen stages, manifests, dispatch censuses, comparisons, and failed reports
remain retained. Private instrumentation and the rejected prototype stay off
the maintained production branch.
