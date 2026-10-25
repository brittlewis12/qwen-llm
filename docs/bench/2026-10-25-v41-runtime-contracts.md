# V4.1 Runtime Contracts And Current Qualification

## Status

Main integrates the TensorOps staging-ownership repair (`e97bd5c0`, reviewed
source `7d80ac08`) and executable-contract/output-A fallback repair
(`de082865`, reviewed source `0d6bafaa`). The maintained source acceptance and
ring/suffix census repairs land separately in `16c70885`, `45a3ce68` and
`0a4c676f`.

This is not a CED, retained-window mapping, mask-publication or automatic-policy
promotion. Current-source independent attention qualification exposes a numerical
failure described below. Do not describe the whole acceptance surface as green.

## Staging Ownership

Physical threadgroup destinations no longer use clamped logical source rows.
Each physical activation/weight staging fragment has one writer, with invalid
physical rows initialized to zero. Valid source reads, quant decoding, K traversal
and epilogues remain unchanged. Both the narrow path and wide weight tails are
covered; the wide activation path already separated these coordinates.

The CPU structural proof covers 10,240 physical tiles and 39,845,888 Half cells;
10,111 historical-map tiles fail its ownership/initialization controls. Output
shape, guard and decoder-chunk checks are separate from numerical GPU comparisons.

Actual M5 validation-on qualification at `7d80ac08` passes eight focused CPU tests
and fourteen grouped/dense GPU tests: F32, Q8_0, Q4_K, IQ1_S, IQ1_M, IQ3_S and
IQ3_XXS. The GPU schedule contains 637 fixture admissions and 9,116 commands,
including counts around 1/2/15/16/17/31/32/33, output-row tails, invalid counts/IDs,
F32/F16 input storage, narrow/wide paths and all three epilogues.

The corrected fixtures independently price full decoded CPU vectors rather than
using compressed-bank size as a proxy. Their admission transaction spans nested
autorelease pools and mandatory weak-object release checks: 9,116 outputs plus
2,835 inputs, all 11,951 passing. These are buffer-object lifetime checks, not
claims about immediate RSS or driver-cache reclamation. An earlier fixture source
and its successful numerical runs remain historical evidence, not substitutes for
these repaired budget/lifetime assertions.

Frozen M5 test executable SHA256:
`92ed3c003a1a62f7f162c23d4193250701503679f1ef9fbd57a0b307e7038d97`.
The repair introduces no new performance claim.

## Executable Contracts And Output-A

Positive but non-20 Sinkhorn iteration counts and routing top-k above sixteen
remain descriptive plans where otherwise valid, but execution rejects them with
typed backend-contract reasons. Capability reporting, import preparation, binding,
run/serve and FFN execution share the CPU-only contract. Intrinsic unsupported
mathematics remains distinct from malformed metadata and local memory fit.
Both `hidden_act` and `hidden_activation` are validated, including conflicting or
unsupported declarations; absence retains the existing SiLU default.

Grouped decode output-A selects its native path only when the complete native
dtype, shape, group, stride, alignment and buffer contract holds. More than sixteen
output groups use the existing general per-group fallback, rather than rejecting
an otherwise executable shape or imposing an eight-group allowlist.

Actual M5 validation-on qualification at `0d6bafaa` passes eleven fixture CPU
tests and the ignored native/fallback GPU witness. Its fifteen cases cover
1/8/16/17/32 distinct groups across F32, BF16 and Q8_0 stored weights, rank seven,
nonzero binding offsets and two positions. Independent stored-precision decoding
plus F64 dots, exact dispatch counts, poisoned outputs, guards and wrong-bank/
permutation/no-op controls are checked. All thirty mandatory bank/low-buffer weak
release checks pass after the case pools drain.

Frozen M5 test executable SHA256:
`20f7caeee7596fcd6222275e4c3025222a835f9c2ab92095c97901606ed92a9b`.
This establishes bounded operator behavior, not full-artifact quality or speed.

## Current Independent Attention Failure

The repaired real-model ring/suffix witness runs against main `de082865` with the
existing REAP-272E MXFP4/Q8-Engram artifact and retained 10K token fixture. Requested
Metal validation is enabled. The model file's identity is unchanged across the
run; no fresh whole-file hash or new artifact-quality qualification is claimed.

The maintained schedule now really exercises calls `[127, 513]`, query tiles
`[127]` and `[512, 1]`, and fifteen checkpoints through position 639. Its selected
M5 MMA/tensor32 dispatch census passes, as does wrapped-prefix snapshot, restore,
growth and endpoint-logit self-parity through position 642.

The separate CPU GGUF attention replay completes all 120 checkpoint/layer rows:

- Selection and owner IDs match in all 120 rows.
- Candidate masks match in all 75 applicable rows.
- None of the 120 output rows passes the existing absolute `1e-4` tolerance.
- Worst max-absolute error is `0.0041255950927734375` at position 126, layer 14.
- Worst relative L2 is `0.0007212740093375828` at position 638, layer 32.

The test exits nonzero at the qualification gate, without a passing qualification
receipt. The underlying comparison, passing census and self-parity receipts are
retained. No tolerance is relaxed and no automatic retry is used. The broader odd
pruning run is deferred while stage-local localization distinguishes production
Half-operand arithmetic from any oracle/runtime defect. That distinction is not
established by the output metrics alone.

This replay is independent attention conditional on captured normalized inputs,
not independent full-model/logit acceptance. Likewise, native self-parity and
within-artifact Full/CED agreement cannot exclude an error common to both arms.

Frozen M5 oracle executable SHA256:
`3077af67b7907dadf937751fc82fee78913315ba208c8ef9fa475e09e31d1ded`.
Raw receipts are named `trust-v34-oracle-ring.*`; the retained comparison has
120 complete rows, not only the first reported failure. The explicit `.failure.json`
receipt confirms natural nonzero exit, no host-health abort and post-exit unchanged
source, binary, token-fixture and model-file identities; it is not a passing receipt.

## Exact Mask-Publication Experiment

Experimental source `3ab20576` compares the incumbent membership scan against
ordered clear plus duplicate-safe atomic scatter. Production dispatch is unchanged.
Dense mask allocation is unchanged. The next-dispatch observer, independent
sorted-set oracle, guards, invalid/duplicate inputs and mandatory post-pool weak
release assertions pass all 144 validation-on cases.

One native fixture process repeats those correctness cases, then retains all 864
timing samples across 36 cases, three rounds and both ABBA/BAAB orders. Each sample
contains four publications in one command buffer: four incumbent dispatches or
eight clear/scatter dispatches. CPU wall includes command creation, encoding,
commit and wait; allocation, compilation, readbacks and observer work are excluded.
CPU and GPU times overlap and must not be added.

Representative medians in microseconds per publication, normalized from each
four-publication sample:

| Domain / IDs | GPU Scan / Clear-Scatter | CPU Scan / Clear-Scatter |
| --- | --- | --- |
| 2,049 / 2,048 | 141.938 / 7.172 | 200.099 / 50.401 |
| 16,384 / 2,048 | 97.516 / 4.667 | 142.047 / 46.599 |
| 131,072 / 2,048 | 224.047 / 5.333 | 274.547 / 49.547 |

The first row's GPU round medians vary materially: incumbent
148.141/102.625/133.453 and candidate 8.969/4.859/6.854 microseconds. All samples
remain included, with no spread/win gate. Low-count controls show the added
dispatch overhead; these results do not justify replacing every shape.

Critically, these timed mixed-ID fixtures contain 256 invalid inputs and repeated
valid writes. At domain 2,049 they set only 1,367 distinct words, not the
production-shaped 2,048 of 2,049. The next bounded confirmation uses valid unique
IDs, includes the newest block and covers ascending plus score-ranked/permuted
order. Actual producer-to-consumer and model/context checks remain required before
promotion. Publication-only ratios are not engine speedups or memory savings.

Offline reduction preserves all 864 sample lines verbatim and validates chronology,
batch/dispatch counts, timestamps, source hashes, admission bounds, natural
completion and releases. One healthy and nineteen corrupt-copy parser controls
behave as expected; byte-identical re-reduction passes. These are parser controls,
not additional runtime evidence. Native launch rejects known validation flags;
external effective injection remains explicitly unattested.

Frozen M5 test executable SHA256:
`78eb7d8311c20ba89c489a39fef6d67a2d356a45f00a609584543b5c8628dca1`.
Native stderr SHA256:
`d5cf5d270455b5a8336d1864a12b2d698a8e306720b4ab58a639bc1e1882954f`.

## Remaining Integration Gates

The composed request-planning/mapping/CED source is not main. Review requires
repairing three boundaries before current-source GPU/model qualification:
request-admitted 2048-row CED with a 4096-row model preference; typed-pressure
Full fallback for optional CED at planning and allocation, distinct from strict
paired qualification; and null aggregate benchmark geometry when repetitions
actually differ. No partially mutated or poisoned state may be retried.

Cold explicit CED, CED-origin prefix reuse, automatic policy selection, independent
numerical acceptance and homogeneous/mixed native economics remain separate
proofs. Preserve frozen P4 as historical evidence rather than inheriting its
performance or arithmetic guarantees into a new composition.
