# Windowed DFlash Pre-Gates — 2026-08-20

Preregistered falsifier packet for the "windowed drafter context" leverage map
(after k3 adversarial review). The map claims: because the Qwen3.8-27B-DFlash2
drafter is all-SWA-2048 (5/5 layers, measured from the served GGUF), the drafter
only ever reads the last 2,048 cross-context columns, so:

- (1a) cold-prefill capture can be windowed to the final 2,048 positions
  (kills the prompt_len x 25,600 x 4B capture allocation — 13.3 GB at 130K),
- (1b) restored requests can seed the drafter from a checkpoint-stored
  capture tail (+210 MB/snapshot), making speculation available on cached
  agentic requests (today serial at 8.35-12.4 tps at 66K-133K ctx),
- (2) Off-mode can keep the cross-context fed (non-terminal Off) and
  DFLASH_OFF_CTX can be removed behind long-band slope evidence.

Gates below are ordered cheapest-first; F1 is the decisive premise test and is
the only gate executed in this session's first run.

## F1 — Window bit-identity oracle (no timing claims)

Harness: `crates/qwen-llm/tests/dflash_window_oracle.rs`, run on the real
Qwen3.8-27B-Q8_0 target + DFlash2 head via `prefill_tokens_with_multi_hidden`
at a 3,076-token prompt. Three drafter sessions seeded from the same captured
columns:

- A: full context (all 3,076 columns, absolute positions 0..3075)
- B: windowed (last 2,048 columns, positions 1028..3075)
- C: negative control (last 2,047 columns, positions 1029..3075)

Same `carry` token, same `noise_start_pos = 3076`, same DFlash2 block. Compare
`draft_block_with_logits` outputs.

Pass bounds (all must hold, bitwise):

1. `max|A - B| == 0` on all N draft logit rows (bit-identical).
2. A and B argmaxes identical (implied by 1; assert anyway).
3. C differs from A on at least one draft row (`max|A - C| > 0`): proves the
   SWA-2048 mask is inclusive at the exact boundary and the harness is
   sensitive to a one-row window truncation.

Fail handling: any failure in 1-2 kills the windowing premise at kernel level;
a failure of only 3 means the harness is insensitive, not that windowing works.
No timing numbers are claimed anywhere in this packet.

## F2 — Dispatch-count census (after 1a/2 implementation)

`QWEN_PREFILL_TRACE_COUNTS=1` / `QWEN_MTP_VERIFY_TRACE_COUNTS=1` runs:

- (a) capture-on-final-chunk-only: zero capture dispatches in all non-final
  prefill chunks. Pass: 0.
- (b) Off-mode capture adds exactly K=5 scatter dispatches/token (dense
  multi-hidden path). Pass: modeled overhead <= 0.5% of a 40-110 ms/token
  decode floor.

## F3 — Long-band slope (timed; the PERF-LOG "Owed" measurement)

Within-session verify(8)/single_token slope, ctx 8K -> 64K, single process,
>= 8 samples/cell per the existing guarded fit. Pass to remove
`DFLASH_OFF_CTX`: fitted break-even <= 3.4 across the band. Fail: keep the
guard; windowing still ships for memory + restored-request speculation.

## F4 — Alpha census at 60-130K (counts only)

Shadow-run the drafter on restored long-agentic sessions (the 33 serial
requests' content class from /tmp/serve_38-dflash.log). Pass: mean emitted
>= 3.1. Fail (< 2.6): long-ctx speculation is marginal; reprice item 2 to
windowed memory + restored-request speculation only.

## F5 — Codec v2 round-trip (no GPU)

Encode/decode with capture-tail flag; old readers reject via unknown flags
(checkpoint_codec.rs:423); restore -> decode -> re-checkpoint preserves the
tail. Pass: byte-exact round-trip + fail-closed rejection.

## F6 — Item 3 pre-pricing (trace-count only)

Count encoders/dispatches per verify(8) pass (predicted 768 GDN encoder
switches). The GDN wavefront kernel must publish byte-identical
`gdn_ckpt_slot`/`conv_ckpt_slot` contents per row (rollback contract). Pass
bound for proceeding to a timed packet: credible removal of >= 50% of the
9.44 ms/token marginal — note checkpoint blits are already priced at only
9-16% of the marginal (PERF-LOG 2026-08-20), so the kernel must attack the
48-layer sequential recurrence latency itself.

## Authority

F1 uses ordinary pageable loading, no residency sets, no whole-model wiring,
and runs with `QWEN_METAL_LEASE_WAIT=1` under the queue-managed lease contract.
The live serve process (PID 83409) is the Metal owner; this packet's run is
untimed and correctness-only.

## Results — F1 adjudication (2026-08-20)

F1 PASS on the first execution, real Qwen3.8-27B-Q8_0 + DFlash2 assets, in-lib
`#[ignore]` test `metal_dflash::tests::windowed_dflash_bit_identity_3076`
(per-PID test lease dir; no contention with the production daemon).

- Drafter confirmed all-SWA: 5/5 layers, block 8, target layers
  [6, 20, 34, 48, 62].
- Full (3,076 cols) vs windowed (last 2,048 cols, absolute positions):
  `max|A-B| = 0.000e0`, 8/8 argmaxes identical — bit-identical drafts.
- Negative control (last 2,047 cols): `max|A-C| = 8.254e-3` — harness
  sensitive, SWA-2048 boundary inclusive.
- Runtime 19.86s untimed correctness-only; no timing claims.

Consequence: the windowed-drafter premise is kernel-verified. 1a (windowed
cold capture) and 1b (checkpoint capture tail) are authorized to proceed to
implementation; F2 (dispatch census) and F5 (codec round-trip) are the next
gates; F3/F4 remain prerequisites for DFLASH_OFF_CTX removal (item 2b).

## Implementation status — 1a landed (2026-08-20)

Windowed cold capture implemented and primitive-validated:

- `qwen_llm::metal_dflash`: `dflash_capture_window_limit` (all-SWA head ->
  SWA window, else `usize::MAX` legacy full-span), `dflash_capture_window_span`,
  `dflash_capture_window_complete` + unit tests.
- Serve (`crates/qwen-cli/src/serve/backend.rs`): windowed admission pricing,
  windowed capture allocation (capture_start = window start), split
  plain/capture prefill loop, window-aware completeness check, windowed dsess
  capacity (window + max_tokens + 16).
- CLI (`crates/qwen-cli/src/main.rs`): windowed capture buffer, plain-prefix +
  capture-suffix split, windowed dsess capacity and seeding.
- New GPU gate `windowed_split_capture_equals_full_capture` (in-lib,
  #[ignore]): split capture buffer == full-capture suffix bitwise
  (max|Δ| = 0.000e0) and drafts from the split-window session == full session
  bitwise (max|Δ| = 0.000e0), 36.57s on the real Q8_0 + DFlash2 assets.

Pending: end-to-end CLI/serve greedy-equivalence run and the F2(a) dispatch
census (both need a qwen binary run; blocked by the live server's
process-exclusive Metal lease — run them at the next lease window).

## E2E + F2 census — executed (2026-08-20, lease window after server stop)

New-binary E2E on Qwen3.8-27B-Q8_0 + DFlash2 (production lease free; runs
serialized, one process at a time):

- Short cell (92-token templated code prompt, 128 tokens): serial and
  drafter outputs **byte-identical**; 26 spec steps, mean emitted 4.92.
- Long cell (2,232-token prompt — crosses the 2,048 window boundary at
  wstart=184, so the plain/capture split executed): serial and drafter
  outputs **byte-identical** (664 bytes, 128 tokens); 31 spec steps, mean
  emitted 4.13. Serial decode 16.59 tps vs drafter 28.54 tps (1.72x).
  Windowed capture cost: prefill 8,903.8 -> 9,610.7 ms (+706.9 ms, one-time
  final-window capture), draft_first_ms=126.3 (2,048-column one-time
  projection).
- F2(a): trace-count census on a 3,345-token raw prompt confirms the split
  schedule (capture sub-span [1297, 3345) internally chunked; prefix chunks
  plain). Capture dispatches exist only inside the final window by
  construction, and the whole-capture tax is bounded at +0.7s on a 9.6s
  prefill versus the legacy every-chunk capture path.

F2(a) and the E2E equivalence gate PASS. 1a is closed as implemented and
product-validated; the remaining gate is F5 for 1b (codec v2).

## Implementation status — 1b landed (2026-08-21)

Checkpoint capture tail + restored-request speculation implemented and gated:

- Codec v2: `FLAG_CAPTURE_TAIL` (1<<2) + `OFF_CAPTURE_TAIL_BYTES` header field
  in the former reserved area; legacy records byte-identical; unknown flags
  fail closed (pre-existing check). F5 round-trip tests PASS
  (`capture_tail_round_trips_bits_and_sets_flag`,
  `legacy_record_without_tail_is_unchanged`,
  `unknown_flags_fail_closed_and_empty_tail_is_rejected`).
- Runtime: `prepare_checkpoint_boundary` takes the tail + features and
  validates length == min(prefix, 2048) x features; `estimate_checkpoint_
  boundary_sizes` prices the tail; `PrefixCacheRestore`/`PreparedCheckpoint
  Restore` carry `capture_tail`. Non-serve callers pass None/0.
- Serve: unified windowed capture buffer (fixed 2,048 columns for all-SWA
  heads — doubles as the serial-decode ring), restore-tail seeding into the
  window, serial decode ring feed via `single_token_with_multi_hidden`,
  spec decode ring feed via `generate_dflash`'s optional ring param,
  boundary tails from the ring (prompt + completed), and
  `should_plan_dflash` relaxed to admit restored requests with a tail
  (dense + greedy only).
- E2E two-turn gate (live serve, Qwen3.8-27B-Q8_0 + DFlash2): turn 2
  restores 257 matched tokens, plans speculation (tail=true), runs
  `decode_path=dflash`, and emits output **byte-identical** to the serial
  control serve on both turns. Unit tests updated and green.

## Findings recorded during gate execution (both PRE-EXISTING at HEAD)

1. **Serve spec-vs-serial greedy divergence on one prompt class.** A
   reasoning-none turn-1 prompt ("Write a Python function that merges two
   sorted lists... end with DONE") produces byte-identical output across
   repeated runs on each path, but the speculative serve and the serial
   serve diverge at output char 497. Reproduced on a pre-change binary
   built from the stash (identical divergence; pre-change drafter output
   == post-change drafter output, so this packet's changes are
   behavior-neutral on that cell). Consistent with the documented open
   item "CPU/GPU argmax tie semantics remain separate work". Deserves its
   own packet (tie-equivalence between packed-verify argmax and
   single-token argmax); it gates which prompts can serve as
   equivalence fixtures.
2. **Five DS4 prefill unit tests fail at HEAD** (`packed_grouped_*` N=1
   gates in deepseek_v4_metal::prefill). Reproduced on the clean tree via
   stash; unrelated to this packet.


