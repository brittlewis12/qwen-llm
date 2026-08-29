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
| Muse Glimmer 30B | fitting not yet wired | text runtime oracle passed | scalar only | adjacent-layer selected-token fit |

## Current Status

The CLI handoff is verified on real dense, ordinary MoE, native J/R, and
workspace-template assets. Selected-mask vector export is complete for dense
published full-J traces. Muse Glimmer scalar text generation matches the local
Q8 llama.cpp logits oracle and is ready for a bounded fitting vertical slice.
Flash-Next remains available as a raw hyper-state path, but lens work is frozen
until a genuine rectangular fitting or asset path exists.

## Next Gate

Fit and read one Muse selected-token transport across one adjacent full-attention
block. The slice must include production residual capture, replay agreement,
J-VJP finite-difference validation, an explicit Muse R rule, a model-bound
artifact, and fresh-run score agreement. Do not broaden to full rows, generic
backends, REST, UI, or experiments before this passes.

## Hard Exclusions

Artifact security/publication, scientific workflows or analysis, REST/UI,
generic plugins, unrelated test repair, cross-device support, and production
hardening.

## Blockers

- None.

## Fast Follows

1. Compose Muse transport across multiple blocks after the adjacent-block slice.
2. Cover Muse sliding-attention adjacent-pair RoPE after full attention passes.
3. Rectangular Flash-Next transport/readout when a genuine fit or asset exists.
4. Sparse position filtering and traces beyond 128 positions.
5. Corpus batching only after measured throughput requires it.

This file is updated in place. It is not a work log or design document. Keep
one active gate, at most three blockers, and at most five fast follows.
