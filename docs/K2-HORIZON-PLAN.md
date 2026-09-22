# K2 Horizon Current Scope And Next Steps

This is the current implementation map, not the superseded chronological plan.
Historical decisions, failed experiments, review findings and qualification
evidence remain in `K2-HORIZON-REVIEW.md`. User-facing contracts
live in `CLI-UX.md` and `SERVE.md`; reproduction is in
`scripts/reference/k2/README.md`.

## Implemented

- Native dense 7B architecture binding across compatible checkpoint stages;
  stage-specific context/RoPE metadata, exact tensor/storage validation, native
  NFC-aware tokenizer and pinned independent tokenization fixtures.
- Retained read-only GGUF execution, transactional session ownership, native F16
  KV and online attention without context-sized score scratch. Eligible Q8 block
  projections use 32-row packed prefill in run/bench/lens; serve stays singleton
  for cancellation. No context eviction, truncation, rolling window or recurrence.
- Run, request benchmark, raw HTTP JSON/SSE, forward-only captures/readouts and
  imported linear-lens transports. Library ordered causal interventions are
  available; local fitting/backward/VJP are not implemented.
- Checkpoint-context and actual device/memory admission replace the old research
  forward/position/response caps. Requested capacity still bounds every session;
  capacity is not a promise of numerical qualification or practical runtime.
- Verified final-artifact no-tools CLI and HTTP chat, native BOS ownership,
  high/medium/low reasoning effort, separate reasoning/final output, strict
  history admission, and raw-mode compatibility. Unknown compatible checkpoints
  remain raw-capable rather than inheriting an unverified chat contract.

## Artifact And Evidence

Local weights: `~/models/K2-Horizon-7B-Q8_0.gguf` (9,573,964,160 bytes), from
`abenzerps/K2-Horizon-7B-GGUF` revision
`a5094087a5a55c2de80264c11504d8ca95a022ff`.

- File SHA256: `5a98a289aba5c8c99ef05c9261287f19e8f586fd679bab45b632eb86b47809bf`.
- Verified content BLAKE3: `719ae3a7c9386c25db2c33b50be15d715f883aa5a762495d5a65776660179e99`.
- Standard Q4_K_M from the same publisher/revision is also downloaded and verified
  at `~/models/K2-Horizon-7B-Q4_K_M.gguf` (5,592,217,984 bytes). Raw generation,
  CPU admission and the 42-row same-Q4 independent integration screen pass. Q4
  embedding/mixed Q4_K-Q6_K projections use existing kernels, with serial prefill.
  This is not Q8-equivalent quality, packed-prefill or long-context evidence.
- Verified final Q4_K_M chat shares Q8's upstream semantics, but binds its own
  retained-content identity. CPU byte/token fixtures and actual CLI/HTTP checks
  cover high/medium/low effort, reasoning partitioning and raw-control parity.
- Native chat follows IFM revision `2c9659a84c4eea6f9f60462221fe762c8c84d75c`,
  with separate upstream and embedded-template hashes. Full retained-byte identity
  is required, not a filename or downloader assertion. Source association is the
  publisher's declaration, not independent BF16-to-GGUF fidelity proof.
- Strict 42-row short oracle and separately frozen v2 holdout pass on final Q8
  weights/F16 KV/M4 Max. V2 covers 4096 row checks, exact top-1 agreement, captures,
  generated tails and packed/serial partition checks under unchanged gates.
- Longer-position/context controls and 1024-position application boundaries pass;
  the full declared 524288 context is not quality-qualified. Historical strict-256
  and v1 near-tie failures remain recorded, not retroactively relabeled as passes.
- No-tools template fixtures cover upstream UTF-8 bytes and native token IDs.
  CLI/HTTP tests cover streaming partitions, incomplete/invalid termination, raw
  prefix parity and actual completed answers; they are not answer-quality claims.

## Remaining Work

The user's current priority is complete practical support for the released final
dense 7B, not a campaign across intermediate checkpoint geometries:

1. Run useful non-Q8 quantizations, starting with the publisher's standard Q4_K_M.
   Reuse the existing mixed-dtype binder and shared projection kernels. Compare
   native execution with an independent implementation of the *same artifact*;
   intentional Q4-versus-Q8 differences are not implementation failures. Lens
   qualification need not block generation support.
2. Complete the released interaction contract: native tool schemas, calls/results
   and history in CLI/HTTP, alongside existing high/medium/low reasoning controls.
   Follow pinned upstream presentation/call formats and fixtures rather than
   inventing a non-thinking mode or an in-process tool execution loop. Associate
   additional final quantizations with their own verified artifact identities.
   Implement rendering in native typed Rust: no dynamic template engines or
   runtime template dependencies. Pinned Python/Jinja remains a development-only
   independent fixture oracle, never an application dependency or execution path.
   Native call/history primitives cover XML, JSON and typed XML plus tool results.
   Native Markdown/XML/JSON schema presentation and tool system-turn instructions
   now match pinned fixtures too. Generated-call parsing and CLI/HTTP request/history
   wiring remain; current frontends still correctly report no-tools.
3. Preserve the full checkpoint-native context range subject to actual resource
   admission. Add focused independent history-boundary evidence without making
   the extent of existing evidence an artificial execution cap.
4. Apply established cross-architecture kernel optimizations where dtype, shape
   and numerical semantics make them applicable. Shared single-token projection
   dispatch is already used; audit mixed-quant packed prefill, reusable scratch
   and existing matrix/fusion paths before inventing new kernels. Measure actual
   product requests, not qualification-harness runtime.

Keep safety/contracts, kernel numerical correctness, semantic provenance and
model-quality/performance evidence distinct. Tests should detect a stated failure
mode, reuse existing shared-kernel coverage and allow justified floating-point
variation. Do not require bitwise equality across different quantizations or
execution topologies, tune frozen gates after observing results, or reinterpret
historical failed experiments as passes.

Packets 37-41 already address direct cache allocation, discarded prefill readouts,
cancellable identity verification, artifact-derived capabilities and setup timing.
The readout overflow design investigation is parked, not a canonical rejection
policy (local stash `151e20b6a648e11f118831cfcd792e2d275e66d1`). CLI interventions,
imported-lens provenance extensions, snapshots and intermediate-checkpoint campaigns
are not blockers for the priorities above.

F16 KV remains the default at 147456 logical bytes/token (72 GiB at 524288).
The private Q8 cache uses 78336 bytes/token (46.875% less), but its recorded
qualification failed; it is not promoted. No silent history loss is acceptable.

## Integration Discipline

Work proceeds on `feat/k2-horizon` in `/Users/tito/code/qwen-llm-k2-horizon`, with
reviewed checkpoints integrated locally into main. No remote pushes are implied.
Cross-family dispatch, shared serial-driver and family-profile reorganization
belong to the concurrent maintenance lane, not this implementation packet.

Use `cx` adversarial review before checkpoints. GPU checks use the production
exclusive lease, real wired-memory gate and `MTL_DEBUG_LAYER=1`; do not alter
foreign servers/jobs. Reuse retained evidence where appropriate rather than
duplicating costly research runs. Preserve unrelated main-worktree changes.
