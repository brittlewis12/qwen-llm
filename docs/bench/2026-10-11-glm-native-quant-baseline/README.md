# Native GLM quant baseline

Implementation checkpoint on `fix/glm-quant-baseline`, based on `abe6a10d`.
This is execution support, not a claim that quantizations have equal quality
or that every supported baseline has the same performance.

## Changes

- Dense and embedding admission follows executable primitive capabilities.
- Hyperconnections retain fused Q8 execution and gain native dense composition.
  Non-Q8 HC needs two reusable decode activation buffers (128 KiB at release
  geometry) and a packed normalization buffer (32 MiB at 512 rows), included in
  session admission and capacity accounting. Packed projection batches native
  weights with F32 operands; formats without grouped support use row GEMVs.
- Grouped latent projections gain native compressed execution for 17 storage
  formats, without half operand staging or expanded weight banks.
- Packed execution uses existing row kernels when a batched optimization is
  unavailable. Missing optimization is no longer grounds for rejecting a dtype.
- Explicit F32 operand modes use F32 kernels or row fallbacks, not silent half
  staging. Existing eligible UD fast paths remain in place.

## Validation

Release library tests built offline with `-j2` and shared target directory
`/Users/tito/code/qwen-llm/target`. GPU correctness runs set
`QWEN_REQUIRE_METAL_TESTS=1 QWEN_METAL_LEASE_WAIT=1 MTL_DEBUG_LAYER=1` and run
serially. Timing runs omit Metal validation and do not overlap builds or GPU work.

The first native Maya integration run passed the existing 200-token
`packed_fast_matches_exact_over_a_grouped_block_and_tail` test, with matching
prefill top-1 and Fast-versus-Exact KL of 2.086e-5. Three continuation steps had
KL 2.105e-5, 1.808e-6 and 1.075e-6. Observed session allocations were 755,843,072
bytes against 759,021,568 priced. This is a native self-consistency check, not
an independent llama.cpp full-model oracle comparison. The final batched-HC run
also passed: prefill KL 5.962e-6, continuation KL at most 3.698e-6, and matching
prefill top-1. Observed allocations were 789,397,504 against 792,576,000 priced.
The UD integration control passed as well (prefill KL 7.775e-7).

Final targeted runs: 70 GLM tests and 11 HC tests passed, with 44 existing
opt-in model/timing tests ignored. Separate grouped-latent, expert, and shared
capability runs also passed (logs retained here). Actual model integration tests
were explicitly enabled separately; GPU tests required Metal, not silent skips.

## Initial Performance Measurements

Same release binary, teacher-forced tokens, no Metal validation, serialized GPU
ownership. These are short diagnostic measurements, not a stable comparative
benchmark or a quality comparison. Decode uses 4 warmup steps followed by 12
unprofiled steps at depth 512. Prefill uses a 512-token chunk, a warmup, two
unprofiled runs, then stage profiling; profiled logits must match unprofiled
logits bitwise.

| Case | UD | Maya |
| --- | ---: | ---: |
| Decode, initial baseline (ms/token) | 31.53 | 34.56 |
| Half-mode prefill, initial baseline (ms/512 tokens) | 2240.1 | 4272.5 |
| Half-mode prefill, batched HC (ms/512 tokens) | 2865.4 | 2777.3 |
| F32-mode prefill, batched HC (ms/512 tokens) | 3229.9 | 7570.8 |

The specific HC bottleneck improved decisively: Maya's profiled attention/FFN
pre stages dropped from 1804.6 ms combined to 71.8 ms after batching. Its default
prefill improved from about 120 to 184 tokens/s in these runs. The unchanged UD
control varied materially, so do not infer either artifact's superiority or
precise cross-artifact speedup from this sample. Decode is unchanged by HC
batching; its generic per-token HC remains an optimization opportunity. Maya's
F32-mode expert fallbacks and grouped latent projection also have optimization
room. F32 mode is slower intentionally rather than silently using half operands.

An initial depth-64-only timing request hit an existing benchmark-helper
assumption: it allocates 512 prefill rows against a 104-token session. That failed
before inference; `ud-decode.log` retains it. Measurements use depth 512 instead.

Artifacts used locally:

- Maya: `/Volumes/wdblack/weights-archive/glm-5.3-flash-maya-m/Maya-M/GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf`
- UD: `/Volumes/wdblack/weights-archive/glm-5.3-flash-ud-iq3_xxs/glm5-next/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf`

## Remaining Boundaries

The following describes the initial checkpoint; mixed expert support and
equivalent metadata normalization are now implemented in the follow-ups
documented in `MIXED-EXPERTS.md` and `METADATA.md`.

Mixed gate/up expert dtypes and formats without an all-slot clamped expert
implementation remain unsupported. This checkpoint does not normalize alternate
metadata conventions or implement missing codecs. Generic HC decode and expert
fallbacks have optimization room. F32 operand preservation does not imply bitwise
identity between generic packed and single-row reduction orders.
The latent F32 decoders currently mirror the shared half-register decoders;
source-parity and independent numerical tests check advertised formats, without
requiring future shared codecs to be implemented for latent projections.

## Retrospective

Small synthetic GPU tests plus an actual fitting artifact advanced compatibility
without making artifact-specific measurements an admission requirement. Review
caught two useful issues: an unnecessary packed admission restriction and a
silent precision fallback. The concrete adjustment is to check baseline
composition before declaring a role unsupported, and test selected dispatch as
well as numerical output. Unmeasured speed remains a measurement question, not
an execution veto.
