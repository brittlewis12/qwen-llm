# K2 Horizon Current Scope And Next Steps

This is the current implementation map, not the superseded chronological plan.
Historical decisions, failed experiments, review findings and qualification
evidence remain in `K2-HORIZON-REVIEW.md` (packets 1-38). User-facing contracts
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

The September 20 out-of-band review is being addressed in leverage order. Chat
partitioning/HTTP and the current-state index were already delivered in packet 36.
Packet 37 removes unpriced tensor-sized host staging from K2 owned allocations;
the admitted 32K-capacity allocation/high-water check passes without claiming a
paired speedup. Immediate follow-ups, ahead of additional architectural work:

- Packet 38 separates cache advancement from readouts and preserves cancellation
  and poison semantics. Bitwise state/continuation tests and paired scheduling
  diagnostics pass; lazy admitted packed-scratch reuse remains a separate step.
- Derive artifact capability reporting from admission; correct request/setup
  timing; add cooperative cancellation to exhaustive identity checks and truthful
  inspection I/O/help/README descriptions.
- Reject overflowing externally supplied RMS inputs without silently changing
  ordinary arithmetic; tighten the fixed-token continuation wording.
- Freeze independent history-boundary checks beyond 256; pin named chat fixture
  inventory and keep diagnostic completion separate from promotion verdicts.
- Add structured producer/deployment execution compatibility to imported lens
  provenance, without banning useful cross-backend transfers.

Longer-term workflow/qualification work:

1. Extend useful long-context quality evidence without mistaking declared context
   or synthetic high-position agreement for full-context model qualification.
2. Improve KV efficiency only behind unchanged quality gates. F16 is the current
   default at 147456 logical bytes/token. The private Q8 cache uses 78336 bytes/token
   (46.875% less) but failed 808/1088 row and 8/16 capture comparisons, with seven
   teacher-forced top-1 differences. Do not promote it, retune gates, or silently
   drop history. Full 524288-position F16 KV alone is 72 GiB.
3. Expose existing library forward interventions through a carefully validated CLI
   interface; retain artifact/site/position ownership and ordered semantics.
4. Qualify additional quantizations and checkpoint stages before widening evidence
   claims. Chat remains bound to the verified final artifact, not all repacks.
5. Tool rendering/parsing, snapshots/prefix reuse and sustained-service qualification
   are separate future capabilities. Do not advertise them through token strings
   or template presence alone.

## Integration Discipline

Work proceeds on `feat/k2-horizon` in `/Users/tito/code/qwen-llm-k2-horizon`, with
reviewed checkpoints integrated locally into main. No remote pushes are implied.
Cross-family dispatch, shared serial-driver and family-profile reorganization
belong to the concurrent maintenance lane, not this implementation packet.

Use `cx` adversarial review before checkpoints. GPU checks use the production
exclusive lease, real wired-memory gate and `MTL_DEBUG_LAYER=1`; do not alter
foreign servers/jobs. Reuse retained evidence where appropriate rather than
duplicating costly research runs. Preserve unrelated main-worktree changes.
