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
- [x] Muse Glimmer fits resumable, query-batched full-transport J/R row shards.
- [x] Muse Glimmer assembles and applies self-contained F16 full transports.
- [x] `read-full` exposes Muse full-vocabulary logits and transported vectors.

## Model Support

| Runtime | Readout/Fit | Intervention | Packed | Next Gate |
| --- | --- | --- | --- | --- |
| ordinary dense | native J/R + full-J trace | live CLI passed | vectors + top-k | complete |
| ordinary MoE | selected rows; fitting deferred | live CLI passed | runtime exists | lens fitting |
| Flash-Next/qwen4exp | native hyper capture; lenses deferred | CLI fixed add passed | serial only | rectangular readout |
| Muse Glimmer 30B | selected + full J/R fit/read | selected-token CLI passed | hybrid B32 full/sliding transport | measure one hybrid shard |

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
all dominant Muse FFN shapes without changing the scalar dispatcher. The fixed-
scratch full-attention bank additionally moves causal-GQA and shared-primal
nonlinear reverse into one device-resident command. Its B32/T16 R command
measures 50.77 ms, with J/R composition error below `3.21e-7` scaled max.
The full-transport row fitter now dispatches full blocks through that bank and
retains the query-batched CPU inverse-RoPE path for sliding blocks. Outer shards
may hold 256 rows while execution remains B32; timing records separate full-bank
commands from replay, sliding reverse stages, and enclosing wall time.
Muse full-transport fitting now has bounded query batches, prompt-level resume,
model/corpus-bound row shards in `[source,row,hidden]` order, and a real Q8 R-lens
CLI proof. Self-contained F16 assembly, exact deployed output-tail replay, and
F16 hidden-transport application are implemented. `read-full` now dispatches
Muse assets with exact GGUF binding, selected-matrix verification, deployed
post-softcap full-vocabulary logits, and optional target vectors.
One real Q8 T=16 R row shard spanning all 51 sources completes in 15.61 seconds
for one prompt (1.53 seconds capture, 14.08 seconds VJP) on the pre-integration
CPU-attention path. Muse identity resolution now accepts only an existing cache
root or fresh Hugging Face declarations and fails rather than hashing weights.
Flash-Next remains available as a raw hyper-state path, but lens work is frozen
until a genuine rectangular fitting or asset path exists.

## Next Gate

Run one bounded one-prompt, 32-row, all-source Q8 shard to replace the
15.61-second baseline and project the complete fit. The model-free hybrid J/R
composition gate already passes. Do not launch a corpus fit as part of this
gate.

## Hard Exclusions

Scientific workflows or analysis, UI, generic plugins, unrelated test repair,
cross-device support, and production hardening beyond required fit/resume
integrity.

## Blockers

- The hybrid full-bank/sliding-fallback scheduler has no real-Q8 timing yet.
- The production fitting corpus and prompt count are not yet selected.

## Fast Follows

1. Bank the sliding inverse-RoPE path only if the hybrid shard prices it.
2. Rectangular Flash-Next transport/readout when a genuine fit or asset exists.
3. Thin local REST only after full-R CLI production is qualified.
4. Corpus batching only after measured throughput requires it.

This file is updated in place. It is not a work log or design document. Keep
one active gate, at most three blockers, and at most five fast follows.
