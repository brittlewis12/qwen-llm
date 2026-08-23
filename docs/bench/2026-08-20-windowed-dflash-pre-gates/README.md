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



## P2 census — verify attention bytes and the packed-N8 lever (2026-08-22)

Arch facts (target GGUF): block_count 65, full_attention_interval 4 -> 13
attn layers; head_count_kv 4, key/value length 256, KV F16.

- KV per token per layer: K 2KB + V 2KB = 4KB. Per verify(8) pass the
  per-row tail dispatches attn per row per layer (104 dispatches/pass),
  each reading the full KV: **416KB per ctx token** -> 54.1GB at 130K,
  3.3GB at 8K.
- Bandwidth floor at 130K: ~114ms at the 474GB/s stream — the current
  slope extrapolation (~103ms) under-predicts slightly (fit was
  L2-resident at 0.5-2K ctx), so the attention body is the whole slope.
- Packed-N8 single-read floor: 14.3ms at stream (8x byte reuse).
  Constraint: the 8-row x 24-Q-head score work (~50 GFLOP/layer) is
  compute-bound under scalar replay; the packed reader must use the
  matrix-score formulation (existing attn matrix kernel lineage).
- P2 gate: projected slope <= 0.40 ms/1K. The single-read floor is
  ~0.11 ms/1K; the compute floor is the real gate and needs the
  pre-pricing packet before kernel work.

## P4 coverage gates — executed (2026-08-22)

- P4a >2K-token generation: a 2,600-token essay (token_limit) through the
  speculative path is byte-identical to serial; the run exercised full ring
  wrap during speculation (2,600 generated >> 2,048 ring), 640 spec steps /
  728 off steps / 101 fallbacks in a mixed re-probe regime.
- P4b restored prefix > 2048: turn 2 restored 2,276 matched tokens
  (prompt 2,298, wstart 250, seed skip > 0) and ran decode_path=dflash;
  byte-identical to the serial control on both turns.
- P4c fallback-heavy completed boundary: the essay turn 1 (fallback=86)
  published its completed checkpoint, proven by turn 2 restoring from it.
  A strictly terminal-during-fallback-replay cell remains unpinned.
- P4d recovering-content re-entry transition: still unpinned (content
  dependent); the re-probe cadence is structurally exercised by P4a's
  mixed regime.

P2 census recorded above; P1 (Q8_0 long-band slope) and P3 (alpha census
+ ctx-aware probe tax) remain the timed gates for OFF_CTX removal.

## P5 preregistered — packed-N8 verify attention (the flagship lever, 2026-08-22)

### Problem

The verify(8) tail dispatches per-row attention (8 rows x 13 layers = 104
dispatches), each reading the full KV: at 130K post-retune that is 62.5ms
per row, ~500ms of a ~615ms verify pass. Serial is ~120ms/token, so
speculation at depth needs byte reuse, not more partitioning (nwg retune
already banked the 1.6-2.1x single-row win).

### Design

- One dispatch per attention layer: TG = (kv_head, partition); 4
  simdgroups per TG, each owning 2 query rows x 6 grouped heads = 12
  (row, head) streams. K tile C=32 staged once per TG in shmem (16KB),
  shared across all 8 query rows.
- Q for all 8 rows is precomputed: the verify tail restructure hoists
  the attention layer's Q projection + norm + RoPE into a batched
  8-row front (mat-mat lineage already exists), then one packed
  attention, then the existing batched gate/o_proj back.
- Causal mask: rows at pos..pos+7 have nested visibility; only the
  final <= 8 KV rows need per-row masking — passed as a per-row
  visible bound (pos + i). Non-final tiles are unmasked.
- Output: 8 per-row O buffers (existing attn_o slots per row) plus
  per-row m/l partials per partition, combined by the existing reduce
  lineage.

### Floors (from the P2 census + retuned audit)

- Bytes: 6.8GB/pass at 130K / 474GB/s = 14.3ms (vs ~500ms today).
- Compute: 8 rows x 24 heads x 130K scores x 2 (QK + PV) ~= 330 GFLOP;
  needs the matrix/MMA formulation; at 8-12 TFLOP/s effective that is
  28-40ms -> projected packed attention ~35-45ms per pass, ~14x.
- Projected verify(8) at 130K: 615 -> ~160ms; break-even at alpha=0.3
  drops from ~3.5 to ~1.3. N=16 becomes a width-parameterized follow-up
  of the same kernel (chained drafting + Q8 N=16 table arms).

### Gates

- G1 correctness: packed reader vs per-row kernel bitwise at small ctx
  (same partition count), cosine >= 0.9999 at 32K; per-row causal
  boundary probes at the last-8 mask edge.
- G2 equivalence: divergent + 10.8K + backoff cells byte-identical with
  the packed reader default-on; shadow-probe max delta re-measured and
  inside the 0.2 margin; acceptance parity (alpha within noise).
- G3 perf: verify attention phase at 130K <= 60ms (synthetic audit
  harness extended to the packed reader).
- Rollback: QWEN_ATTN_V4_PACKED_N8=0.

### Open design decisions

- Register pressure: 12 streams per simdgroup with C=32 may exceed the
  register budget; fallback Q4 rows per simdgroup (2 TGs per kv head).
- F16 staging of Q (scores in F32) matches the scorer lineage.

## P5 implementation update (2026-08-22) — no new kernel needed

Scouting closed the "new kernel" premise: the promoted prefill matrix
attention pipeline (kernel_attn_matrix_transpose_v_f16 ->
kernel_attn_matrix_kq_f32 -> softmax -> kernel_attn_matrix_kqv_norm_f32,
metal_dflash.rs ~9850-10400) is shape-general and already implements
per-row causal masking via `max_visible = min(n_pos, base_pos +
row_last + 1)` — exactly the nested visibility of 8 decode rows at
pos..pos+7. The packed-N8 reader reduces to a rewire:

1. Hoist the verify tail's per-row Q (projection + norm + RoPE) into a
   batched 8-row front, packed [8, 24, 256].
2. Call the existing matrix pipeline with n_rows=8, base_pos=pos,
   causal masks on; scores scratch [192, n_pos] F32 = 100MB at 130K
   (session-owned, admitted).
3. Scatter per-row O back into the existing attn_o slots; reuse the
   existing partial/reduce contract.

Floors unchanged (14.3ms bytes / 28-40ms compute vs ~500ms). Remaining
work: the Q-hoist restructure in encode_packed_verify_layer_major_inner,
scores scratch allocation + admission, and gates G1-G3. Rollback
QWEN_ATTN_V4_PACKED_N8=0.

## P5 perf verdict (2026-08-22) — q2 kernel quality-clean, perf-negative

The packed g6 q2 shared-KV attention, with both 64-partition couplings
fixed and F32 Q staging, is numerically clean (boundary oracle:
cos=1.000000, max|delta| ~1e-6 flat across 512-16K). But the synthetic
perf audit at 130K/nwg=512 measures **42.1ms per layer vs 38.5ms for
the per-row path** (8 rows x 4.81ms) — the 2x byte reduction is eaten
by the kernel's serial per-K-row score loop at ~51 GB/s. Default-on is
NOT warranted; the path stays behind QWEN_MTP_ATTN_QN_SHARED_KV with
its quality fixes banked.

The real P5 lever remains the MMA matrix-pipeline decode reader
(transpose_v -> matrix KQ with max_visible causal masks -> softmax ->
KQV-norm), which replaces the scalar score loop with simdgroup matrix
multiplies: floors 14.3ms bytes / 28-40ms compute at 130K. That is the
next implementation packet.

## P5 design refinement (2026-08-22) — the direct-V KQV requirement

Three-tier economics for the packed verify attention at 130K (13 layers,
KV 4KB/token/layer, 474 GB/s stream):

- Tier 1, q2 shared-KV kernel (landed, quality-clean): 42.1ms/layer,
  perf-negative vs the per-row 38.5ms — serial per-K-row score loop at
  ~51 GB/s.
- Tier 2, prefill matrix pipeline verbatim (transpose_v + KQ + softmax +
  KQV on v_t): the transpose doubles V traffic (write v_t + read v_t),
  ~1.8x at best — marginal.
- Tier 3, matrix pipeline with a NEW direct-V KQV kernel (read v_cache
  strided into shmem-staged MMA tiles, no v_t): KQ reads K once
  (268MB/layer), KQV reads V once (268MB), scores round-trip 200MB —
  ~736MB/layer ~ 1.6ms at stream, plus MMA compute (QK+PV ~ 2.2 GFLOP
  per layer over 8 rows x 24 heads) — projected 50-60ms per verify pass
  vs ~500ms today (8-10x).

Tier 3 is the implementation target: `kernel_attn_matrix_kqv_direct_v_f32`
(~150 lines, shmem-staged V tiles, F32 accumulation), scratch
[scores: 8 x 24 x n_pos F32 = 100MB at 130K, session-owned + admitted],
wired into the shared-KV branch behind QWEN_MTP_ATTN_QN_MATRIX. Gates
unchanged (oracle vs per-row at all ctx, byte-identity cells, synthetic
perf at 64-130K).
