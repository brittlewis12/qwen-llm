# S0 Results

Executed 2026-08-18 against `target/release/qwen` at 1ecf208 on Apple M4 Max
(unified memory), `RUST_LOG=warn,qwen_diag=info`, fresh private cache dir per
cell. Raw rows in `artifacts/<cell>/results.jsonl`; stderr per turn alongside.

## Prediction verdicts

- **P1a — CONFIRMED (both families).** Strip-scripted hits every turn from
  2 with `capture=prompt` and `matched_t = prompt_tokens_{t−1}` exactly
  (equality, not just ≥−2), monotone through turn 10 (A3B: 1128→2131;
  q38: identical trajectory). The generic ChatML renderer is
  byte-prefix-stable under strip on both models.
- **P1b — CONFIRMED (both families).** Preserve-scripted: `hit=false,
  matched=0, candidates=0, capture=completed` at every turn. Publication
  shadowing + whole-blob matching ⇒ zero reuse for any client that does
  not echo generated text verbatim.
- **P2 — CONFIRMED.** Strip-scripted tails 89–165 tokens
  (assistant-stripped + user + overhead), matching transcript arithmetic.
- **P3 — CONFIRMED.** Preserve-live matched through the full prior
  response every turn: turn 2 `matched=5224` = prompt 1128 + generated
  4096; tails 78–135 tokens while prompts grew to 17,953. Verbatim echo
  through JSON → render → re-tokenize preserved token identity exactly.
- **P4 — CONFIRMED, with a natural experiment.** Strip-live pays
  tail = response + user per turn. token_limit turns left `<think>`
  unclosed and therefore unstrippable (tails 4,195–4,213 tokens,
  ~14.2–14.7 s prefill); EOS turns stripped cleanly (tails 305–736,
  1.8–3.2 s). Preserve-vs-strip TTFT delta at heavy-thinking scale:
  **~1–1.9 s vs ~2–14.9 s per turn.**
- **P5 — PARTIAL.** Restore: 25 ms (miss probe) to 169 ms at ≤9.5k tokens
  (≤200 ms ✓); 263 ms at 17.9k (outside the ≤8k scope, noted). Capture:
  8–42 ms ✓. Publish: scripted 107–140 ms ✓; live blobs falsify the
  letter of the ≤300 ms bound (1.2 s turn-1 including identity
  computation; 277–540 ms thereafter for 259–517 MB blobs) — deferred
  post-response-flush, so none of it lands in TTFT.

## Gate outcomes

- **G1 — PASS.** No `matched` reset in any cell; renderer prefix
  stability is not a blocker. S1 proceeds without renderer redesign
  (golden tests still required for the modern/bespoke renderers, which
  this packet deliberately did not exercise).
- **G2 — QUANTIFIED.** Preserve rendering is economically free
  (~100-token tails regardless of context); strip carries a measured
  0.3k–4.2k-token / 2–15 s per-turn tax at heavy-thinking scale. Per
  Amendment 1, the preserve-by-default decision now rests on per-family
  behavioral evidence; a behavioral packet is the follow-up.
- **G3 — RENEGOTIATED BY F2 (below).** Checkpoint-machinery overheads
  fit the S1 TTFT budget, but the per-process weight-paging floor does
  not. The S1 `<150 ms warm turn-2 TTFT @8k` gate is unreachable in
  spawn-per-turn form on these assets and applies to the resident server
  — which S1 builds. S0 thereby strengthens the server rationale rather
  than weakening it.

## Unpredicted findings

- **F1 — Verbatim-echo requirement (design-critical).** Whole-blob-prefix
  matching (checkpoint_store.rs:393-405) + single-publication shadowing
  (completed replaces prompt-boundary, main.rs:6341→6616) means
  preserve-mode reuse exists only for byte-verbatim echo clients. S1 must
  either publish both boundaries per turn or guarantee echo server-side.
  Opaque reasoning items guarantee it structurally — an independent
  engine-level argument for the items facade.
- **F2 — Per-process paging floor dominates stateless TTFT.** Scripted
  hit turns still paid 5–8 s (A3B) / 1–5 s (q38) prefill on 100-token
  tails; preserve-scripted full prefills ran at ~22 tok/s (A3B,
  49.7 s→130.8 s linear in prompt) — weight first-touch is absorbed into
  each fresh process's prefill, and `load_ms` (~2.5 s) does not cover it.
  Interactive-grade TTFT requires model residency; checkpoints alone
  cannot provide it across process spawns.
- **F3 — Context admission fails closed** exactly as Open Responses
  `truncation:"disabled"` mandates: turn 4 of the first live attempt
  errored with `max context 16384 is smaller than prompt 13749 +
  generation 4096` instead of silently truncating. The facade gets
  spec-conformant behavior from the engine for free; the facade's
  budget accounting must surface it as the spec error rather than a
  process exit.
- **F4 — Truncated thinking is unstrippable.** token_limit turns leave
  `<think>` unclosed; strip-mode rendering retains the full dump. The
  facade must represent truncated reasoning explicitly (items
  `status: incomplete` is the natural encoding) rather than assume
  strip-on-render is always available.
- **F5 — Behavioral flag (single observation, not evidence).**
  Preserve-live hit token_limit all 5 turns; strip-live reached EOS from
  turn 3 on the same tasks/seed. Preserved heavy thinking may induce
  more thinking. Belongs to the behavioral packet's hypothesis list.
- **F6 — Storage economics.** A3B marginal snapshot cost ≈ 20.5 KB/token
  (259→517 MB over 9.4k→17.8k tokens); q38 ≈ 90–96 KB/token (2.8 GB for
  ten ≤2.9k-token blobs). Confirms dedupe/delta-chain priority for the
  store: consecutive blobs in one lane re-store nearly all of their
  predecessor's bytes.

## Decision

Proceed to S1 (resident serial server, Open Responses subset) with:
verbatim-echo or dual-boundary publication (F1), residency as the TTFT
mechanism with durable checkpoints as cross-restart substrate (F2),
spec-error surfacing of context admission (F3), incomplete-reasoning
representation (F4). Preregister the per-family behavioral packet
(preserve-vs-strip quality, F5 hypothesis included) before adopting
preserve-by-default beyond Qwen3.6.
