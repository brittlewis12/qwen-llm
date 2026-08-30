# Lens MVP

North star: make real R-lens, J-lens, and template-lens readouts plus ordered
post-block interventions usable from a simple local CLI soon enough to preserve
experiment time.

Deadline: first usable command today. Missed timeboxes trigger consultation,
not compensating abstraction.

## Done When

- [x] Native R-lens and J-lens fitting/readout commands exist.
- [x] Published full-J import and readout commands exist.
- [x] A real template-lens readout matches its source semantics.
- [x] Fixed and residual-L2-relative additions work.
- [x] Projection ablation and source-to-target displacement work.
- [x] Layer, prefill-position, and decode-step scopes work.
- [x] Matching operations execute in file order.
- [x] The minimal CLI runs with real lens directions.
- [x] Packed full-J traces cover layer x position top-k and occurrence counts.
- [x] Explicit layer:position masks can export transported J-space vectors.
- [x] Flash-Next exposes serial native hyper-state capture and fixed addition.
- [x] The CLI runs Flash-Next with explicit native hyper directions.
- [x] Muse Glimmer fits and runs composed full/sliding selected-token J/R lenses.
- [x] Muse Glimmer applies scoped, ordered selected-token interventions.

## Model Support

| Runtime | Readout/Fit | Intervention | Packed | Next Gate |
| --- | --- | --- | --- | --- |
| ordinary dense | native J/R + full-J trace | live CLI passed | vectors + top-k | complete |
| ordinary MoE | selected rows; fitting deferred | live CLI passed | runtime exists | lens fitting |
| Flash-Next/qwen4exp | native hyper capture; lenses deferred | CLI fixed add passed | serial only | rectangular readout |
| Muse Glimmer 30B | scalar selected-token J/R oracle | live CLI implemented | resumable B32 source-50 rows | bounded Q8 shard smoke |

## Current Status

The CLI handoff is verified on real dense, ordinary MoE, native J/R, and
workspace-template assets. Selected-mask vector export is complete for dense
published full-J traces. Muse Glimmer scalar text generation matches the local
Q8 llama.cpp logits oracle. Real Q8 `fit-tokens` and `run` commands now produce
and consume model-bound J/R artifacts across composed full/sliding blocks, with
multiple selected source layers and prefill/decode readouts at the post-block
residual coordinate. The same Muse runner now supports fixed, residual-relative,
projection-ablation, and source-to-target actions with ordered stacking and
exact layer/prefill/decode scopes.
The genuine compact full-R asset is hidden-to-hidden
`[51,6656,6656]` F16 (4.21 GiB), not a vocabulary-row tensor. The scalar
selected-token fitter remains its correctness oracle; extrapolating that path
to the default 25-prompt corpus takes roughly nine days and must not be launched.
The bank-specific native-Q8 transpose VJP is qualified at `25.36-26.69x` on
all dominant Muse FFN shapes without changing the scalar dispatcher.
The banked Metal causal-GQA VJP is CPU-oracle qualified and clears its B32/T16
5 ms gate without a backward CPU boundary.
The fixed-scratch full-attention one-block bank now composes the native Q8 and
attention mechanisms with shared-primal nonlinear VJPs. B32/T16 R-rule commands
measure 50.77 ms on a resident synthetic release-shape fixture, and all J/R
composition gates match the scalar oracle within `3.21e-7` scaled max error.
`fit-rows` now exposes that motor through R-only, B32, target-51/source-50
shards with at most 256 rows, prompt-order checkpoints, and an explicit local
model locator that hashes zero weight bytes. Model-free estimator and
filesystem resume gates pass; no corpus fit has run.
Flash-Next remains available as a raw hyper-state path, but lens work is frozen
until a genuine rectangular fitting or asset path exists.

## Next Gate

Run one bounded one-prompt, 32-row Q8 source-50 shard to verify real capture,
resume, and artifact publication. Then add the sliding-block bank seam needed
to expand the same engine below source 50. Do not run a corpus fit until that
seam projects the complete 25-prompt build into an agreed hours-scale envelope.

## Hard Exclusions

Artifact security/publication, scientific workflows or analysis, UI, generic
plugins, unrelated test repair, cross-device support, and production hardening.

## Blockers

- The source-50 row-slab CLI has not completed one bounded real-Q8 smoke.
- Sliding-block bank integration still needs inverse RoPE over Q/K cotangents.

## Fast Follows

1. Full-R F16 mmap consumption for full-vocabulary and vector readouts.
2. Rectangular Flash-Next transport/readout when a genuine fit or asset exists.
3. Thin local REST only after full-R CLI production is qualified.
4. Decide whether large operation schedules need an application-output cap.

This file is updated in place. It is not a work log or design document. Keep
one active gate, at most three blockers, and at most five fast follows.
