# Normalized F32 mHC MMA Reuse

Bounded candidate: reuse the existing grouped F32-operand MMA template for F32
weights and complete eight-output fragments. M24 uses guarded weight loads and
stores, without padding or new GPU buffers. Q6 arithmetic and caller policy stay
unchanged; F32 HC permission is separate, test-only Fast Half. Normalize before
projection, retaining existing scratch. Exact and other precision modes do not
change.

First measure one actual final-block FFN HC projection at K16384/M24/N2048,
using surviving normalized scratch and weights. Scalar replay must match captured
mixes, confirming the binding. Both arms are checked independently against F64
dots for all 24 channels at ten tokens. The preset numerical tolerance is not a
derived forward-error bound or a whole-model quality guarantee.

Warm both arms, then one leaf ABBA without census/readback/oracle work in timing.
Only a convincing leaf win earns one ordinary whole-prefill ABBA. The entire
HC-pre parent budget is approximately 263 ms; projection owns less. M24/N2048
has only 32 MMA threadgroups, so insufficient parallelism is a concrete risk.
Stop on leaf loss or inconclusive whole-model conversion; no tile sweep.

Independent review finds no runtime blocker. Grouped regression under leased
Metal validation: 96 pass, 54 ignored, including all new F32 numerical/guard
tests and shipped Q6 tests. GLM runtime tests: 49 pass, 55 ignored. All-target
release type checks pass.

## Result: Park The Candidate

`maya-leaf.jsonl` uses final block 44 FFN HC. Scalar replay matches the captured
projection bitwise; both arms pass all sampled independent dots, and full output
is finite. GPU durations are finite and positive.

| Arm | Leaf wall ms | Leaf GPU ms | Whole prefill wall ms |
| --- | ---: | ---: | ---: |
| A1 scalar | 7.723875 | 7.145750 | 8097.769 |
| B1 MMA | 1.841584 | 1.145708 | 8409.403 |
| B2 MMA | 2.137250 | 1.452958 | 8436.907 |
| A2 scalar | 3.266958 | 2.668542 | 8430.192 |

The leaf favors MMA in both pairs but has large scalar drift; its 3.777x mean
GPU ratio is not a stable speedup estimate. It justified the planned ordinary
whole-prefill screen, not extrapolation across 90 HC calls.

`maya-whole.jsonl` substitutes only the 90 F32 HC projections. Non-HC dispatch
signatures, including shipped Q6 MMA geometry, are identical. Whole A/B is
0.981103x: both pairs favor the incumbent, the second effectively tied. This
does not establish a precise regression, but it does not demonstrate useful
whole-model conversion. Stop here without a tile sweep or favorable-repeat hunt.

Both arms reproduce their own logits bitwise through three common continuation
tokens. Between-arm KL ranges from 0.00000360 to 0.00758116. This is descriptive
reduction-order sensitivity, not a quality verdict or the reason to reject the
candidate. The decision is lack of demonstrated end-to-end benefit.

Retain source in the experiment checkpoint's Git history and raw evidence here;
remove the candidate, opt-in API, and harness from active builds. Production
continues using the existing generic normalized F32 HC projection. Next, exercise
the complete current Maya cached-conversation path rather than another kernel
search with diminishing measured opportunity.
