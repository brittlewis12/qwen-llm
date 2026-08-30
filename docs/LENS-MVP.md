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
One real Q8 T=16 R row shard spanning all 51 sources completed with 14.08
seconds of VJP wall on the pre-integration CPU-attention path. The integrated
hybrid completes the same B32 VJP in 5.016 seconds (`2.8068x`) and matches all
10,862,592 prior F32 outputs within `1.863e-7` max absolute error. At 208 B32
batches, the default 25-prompt job now projects to a 7.246-hour VJP-only floor.
The promoted block-major R256 engine reuses replay across eight B32 banks,
measures 32.817 seconds against a 45.052-second chunk-major control, and matches
all 86,900,736 outputs bitwise. Its 25-prompt/26-shard engine projection is
5.925 hours before capture, checkpoint I/O, and assembly. A RoPE-correct Metal
bank now replaces the CPU fallback for all 38 sliding blocks: block 50 improves
from 77.520 to 57.232 ms, while promoted R256 engine wall falls to 27.326
seconds. Full and sliding commands are at parity, and the engine projection is
now 4.934 hours.
Muse identity resolution accepts only an existing cache root or fresh Hugging
Face declarations and fails rather than hashing weights.
Flash-Next remains available as a raw hyper-state path, but lens work is frozen
until a genuine rectangular fitting or asset path exists.

## Next Gate

Qualify one private model-free Q8 kernel with 256-query tile reuse against the
incumbent 128-query tile at released `6656x19968` FFN gate/up and transposed down
shapes with `n_query=512`. Warm once, take five alternating GPU-time samples,
require bitwise equality and 15% on both directions, and hard-stop at 180
seconds. Do not load a model asset or launch a corpus fit.

## Hard Exclusions

Scientific workflows or analysis, UI, generic plugins, unrelated test repair,
cross-device support, and production hardening beyond required fit/resume
integrity.

## Blockers

- The common Metal bank command train owns 24.069 of 27.326 R256 seconds.
- The production fitting corpus and prompt count are not yet selected.

## Fast Follows

1. Integrate 256-query Q8 reuse only if both released FFN shapes clear the gate.
2. Rectangular Flash-Next transport/readout when a genuine fit or asset exists.
3. Thin local REST only after full-R CLI production is qualified.
4. Corpus batching only after measured throughput requires it.

This file is updated in place. It is not a work log or design document. Keep
one active gate, at most three blockers, and at most five fast follows.
