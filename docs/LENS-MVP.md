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

## Model Support

| Runtime | Readout/Fit | Intervention | Packed | Next Gate |
| --- | --- | --- | --- | --- |
| ordinary dense | native J/R + full-J trace | live CLI passed | full-J trace | vector export |
| ordinary MoE | selected rows; fitting deferred | live CLI passed | runtime exists | lens fitting |
| Flash-Next/qwen4exp | adapter deferred | adapter deferred | existing runtime | coordinate spike |

## Current Status

The first CLI handoff is complete and verified on real dense, ordinary MoE,
published full-J, native J/R, and workspace-template assets.

## Next Gate

Timebox the Flash-Next post-PLE coordinate/intervention spike to two hours.
Establish the narrow capture/mutation seam or stop with concrete integration
options; do not broaden into fitting, optimization, REST, UI, or experiments.

## Hard Exclusions

Artifact security/publication, scientific workflows or analysis, REST/UI,
generic plugins, unrelated test repair, cross-device support, and production
hardening.

## Blockers

- None.

## Fast Follows

1. Selected-mask transported-vector export.
2. Full-transport adapters for additional real J/R assets.
3. Flash-Next adapter implementation if the spike passes.
4. Sparse position filtering and traces beyond 128 positions.
5. Corpus batching only after measured throughput requires it.

This file is updated in place. It is not a work log or design document. Keep
one active gate, at most three blockers, and at most five fast follows.
