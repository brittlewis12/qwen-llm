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

## Model Support

| Runtime | Readout/Fit | Intervention | Packed | Next Gate |
| --- | --- | --- | --- | --- |
| ordinary dense | native J/R + full-J trace | live CLI passed | vectors + top-k | complete |
| ordinary MoE | selected rows; fitting deferred | live CLI passed | runtime exists | lens fitting |
| Flash-Next/qwen4exp | native hyper capture; lenses deferred | CLI fixed add passed | serial only | rectangular readout |

## Current Status

The CLI handoff is verified on real dense, ordinary MoE, native J/R, and
workspace-template assets. Selected-mask vector export is complete for dense
published full-J traces. Flash-Next now has a separately verified library seam
and restricted CLI path for its persistent 10,240-wide post-layer hyper state.

## Next Gate

Timebox the concrete path to a real rectangular Flash-Next transport/readout.
Do not invent a 5,120-to-10,240 lift, treat raw hyper control as a J/R lens, or
broaden into optimization, REST, UI, or experiments.

## Hard Exclusions

Artifact security/publication, scientific workflows or analysis, REST/UI,
generic plugins, unrelated test repair, cross-device support, and production
hardening.

## Blockers

- None.

## Fast Follows

1. Rectangular Flash-Next transport/readout support when a real asset exists.
2. Multiple Flash operations per token only when a real workflow requires it.
3. Sparse position filtering and traces beyond 128 positions.
4. Corpus batching only after measured throughput requires it.

This file is updated in place. It is not a work log or design document. Keep
one active gate, at most three blockers, and at most five fast follows.
