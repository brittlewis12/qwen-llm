# K2 Horizon adversarial review

Date: 2026-09-18. Base: `4d8716ab`. No engine changes, builds, model loads, GPU
execution, downloads of weights, or server operations occurred in this review.

## Reproduction

Initial command in the K2 worktree:

```sh
cx ask -m luna -e high --sandbox read-only --text \
  --prompt-file docs/K2-HORIZON-REVIEW-REQUEST.md
```

Session: `01a0b4e8-1183-7780-98fd-4a2038bf2504`. The initial call hit the CLI
tool's 120-second limit; a read-only resume with a longer timeout completed the
review. The latest response and complete review conversation remain available
through:

```sh
cx show-last --all 01a0b4e8-1183-7780-98fd-4a2038bf2504
cx transcript 01a0b4e8-1183-7780-98fd-4a2038bf2504 --assistant-only --include-commentary
```

Initial verdict: **revise before implementation**. No user decisions were
identified as blocking. The following is a disposition summary, not a verbatim
transcript or a claim of reviewer approval for implementation correctness.

## Findings and disposition

| Finding | Disposition |
| --- | --- |
| Ordinary loader/runtime encode Qwen hybrid assumptions | Accepted; explicitly use a family-local K2 binder/runtime, sharing primitives only |
| Optimized attention is not already qualified for K2 | Accepted; split M2a short-context correctness from M2b bounded-scratch long-context/packed attention |
| Existing full-lens imports bind Qwen geometry or fitting schemas | Accepted; specify forward-only, dynamically validated imported assets, no local fit dependency |
| Ordinary packed eligibility is Qwen-specific | Accepted; K2 packed implementation/qualification is explicit, not a flag enabling the Qwen path |
| RoPE bases and coordinates cannot inherit final/Qwen defaults | Already required; strengthen nonzero-base packed fixtures and stage coverage from M0 |
| Cache/admission/snapshot identity needs explicit layout and precision | Accepted; separate declared context, execution limit, request capacity, and physical allocation |
| Generic serving inherits Qwen ChatML | Accepted and extended: override both rendering and default output/tool protocol |
| Native BPE reuse is not tokenizer conformance | Already required; HF-independent fixtures and checkpoint-specific identities remain mandatory |
| Packed/serial bitwise equality is not generally appropriate | Accepted; declare numerical contracts and calibrated tolerances, not a global loose tolerance |
| MoVA and advanced KV can expand scope prematurely | Accepted; dense product utility is the MoVA gate; advanced KV remains separate measured research |

Local spot checks confirmed ordinary attention's head-dim256 restriction,
Muse's optimized packed Q32/KV2/H128 restriction, fit-token admission in
`lens_run/lenses.rs`, and Qwen defaults in `serve/http.rs`. Head-dim128 code
does exist in Muse: the gap is K2's complete geometry/layout/qualification, not
the absence of every reusable 128-wide algorithm.

The review's suggested first packet spanned profile, full CPU reference,
tokenizer, lens import, and family registration. That is too broad for the first
implementation checkpoint. Its suggested HTTP token-ID coverage also exceeds
the present serving requirement. Neither is adopted as an immediate obligation.

## First bounded implementation packet

Deliver only a stage-aware model-free K2 7B profile/binder foundation. Do not
make K2 selectable as runnable, touch Metal execution, update FFI dependencies,
change request schemas, add fitting, or begin MoVA.

Proposed touched files:

- `crates/qwen-llm/src/k2_horizon.rs`: validated 7B geometry and positional
  metadata, structural GGUF tensor-role binding, checked capacity estimates,
  and clear separation of model validity from execution capability.
- `crates/qwen-llm/src/lib.rs`: expose the foundation module only.
- Small synthetic fixtures/tests following the repository's existing GGUF test
  construction conventions. Pin small external config/reference artifacts if
  included; do not call hand-authored data independent checkpoint evidence.
- `docs/K2-HORIZON-PLAN.md`: record actual completed scope and outstanding gates.

Targeted CPU-only acceptance cases:

1. Pretrain/midtrain/final positional profiles preserve their own context/theta.
2. Raw-valid metadata does not require a chat template or final release name.
3. Incompatible geometry, unsupported optional features, missing/duplicate tensor
   roles, bad shapes/storage alignment, and arithmetic overflow fail explicitly.
4. Untied embedding/output roles and final grouped norm are structurally bound.
5. F16 KV is exactly 147456 bytes/token; proposed Q8_0 sizing is 78336 bytes/token
   with each K/V row independently block-aligned. Neither calculation claims an
   implemented Q8 backend or includes unstated allocator reserve.
6. Checkpoint-declared context does not turn into a runnable-capacity claim.
7. Existing family detection remains unchanged; no accidental Qwen fallback.

The next bounded packets are the native tokenizer conformance implementation
and independent scalar math fixtures, which can proceed independently once
their source pins are established. Lens manifest design starts early, but its
implementation does not need to share the first profile patch.

Live numerical and performance evidence remains a later coordinated step.
No model-free test can establish final/earlier checkpoint output quality or
full-context throughput.

## Closure review

The reviewer reread the revised plan and this disposition, returning:
**proceed with the first packet**. No blocking user decisions remain.

Three nonblocking clarifications are accepted:

- The first packet is a subset of M0, not completion of its CPU-math fixtures.
  RoPE numerical and grouped-normalization fixtures stay in the next packet.
- The binder is an explicit library API/test target only. Runtime family
  discovery and model selection remain unavailable until safe dispatch exists.
- Profile types must distinguish validated configuration, execution support,
  and missing required facts. Optional provenance can remain unknown; required
  execution metadata cannot default silently. Unknown is not interchangeable
  with known-but-unsupported.

This is approval to begin a bounded foundation patch, not correctness or
performance approval for an implemented K2 runtime.

## Packet 1 pre-commit review

The same session inspected `k2_horizon.rs`, its tests, and the module export.
Verdict: **proceed; commit-ready**, no blockers. Eight CPU-only tests passed.
After the download completed and its SHA256 was verified, the explicit ignored
CPU/header-only test also passed against the real Q8 GGUF. No model inference
or GPU execution occurred.

Nonblocking follow-ups: add a direct adversarial synthetic test of the public
backed-range path; split tensor error categories if callers need structured
classification; do not describe this packet as completing M0's scalar math.
The existing GGUF reader already rejects out-of-file and overlapping payloads,
and the new public binder rechecks bound ranges with `try_slice`.

## Packet 2 pre-commit review

Native tokenizer and independent HF fixture generation were reviewed in the
same session. Initial and closure verdicts: **commit-ready**, no blockers.
Follow-ups were addressed before commit: strict paired-separator metadata,
cumulative normalized-input byte accounting, and explicit metadata-only download
documentation. The HF single-sequence postprocessor inserts BOS but no trailing
EOS; the GGUF paired SEP flag must not change that behavior.

Validation: five regular K2 tests (including 256 Unicode fuzz cases), two
explicit ignored CPU-only conformance tests (both stage vocabularies and the
real Q8 tokenizer), and three existing Qwen/JoyAI splitter regressions pass.
Expected IDs come from pinned `tokenizers==0.22.2` with independently hashed HF
tokenizer artifacts, not from the new Rust encoder. No model forward or GPU
execution is involved. Lens span/identity integration is not claimed.

## Packet 3 design and pre-commit review

The same session recommended a composed tiny CPU reference over isolated math
helpers. The design review required an explicit prefill/continuation contract,
rounded cache reads, all-or-nothing commits, and pinned readout conventions.
The implementation review returned **proceed; no commit blocker** after reading
the Rust reference/tests, independent NumPy generator, and generated fixtures.
No GPU, model loads, or server operations occurred during review or validation.

The review confirmed output-major matrices, split-half RoPE, contiguous GQA,
causal masking/current-token inclusion, and full-append rollback. Two nonblocking
notes were addressed: documentation calls corrupted NumPy equations fixture
distinguishability controls rather than Rust mutation tests; tests explicitly
assert all six expected positional/storage cases. An additional end-to-end F16
cache-overflow test checks rollback after earlier staged rows/layers succeeded.

Seven targeted CPU tests pass. This packet supplies synthetic equation evidence,
not checkpoint parity, an executable GGUF path, GPU ABI qualification, actual
F16 cache memory savings, or runtime admission.
