# S0: Render prefix stability and durable-checkpoint economics for agent loops

Status: preregistration, 2026-08-18. Predictions frozen before any cell
executed.

## Purpose

The serving-facade program (S1–S4: Open Responses subset server → tool items →
reasoning round-trip → public v0) rests on one load-bearing assumption:

> Chat-template re-rendering of a growing agent-loop transcript is
> byte-prefix-stable across turns, so durable checkpoints restore most of the
> prior context and each turn prefills only the new tail.

This packet falsifies or quantifies that assumption using only the existing
CLI — no new engine or server code — before S1 is designed. It follows the
adversarial jam conclusion that this measurement must precede server work
(k3 session `ses_fe8ee25c6ffe`, rounds 2–3; full investigation context is in
the OpenCode session titled by this program, retrievable via Recall).

## Mechanism under test

Per single-turn invocation with `--messages` + `--durable-prefix-cache DIR`:

- restore: longest-prefix durable lookup, stderr line
  `durable_prefix_cache: identity_cache=… checkpoint_hit=… matched=…
  restored=… exact=… candidates=… … restore_total_ms=…` (main.rs:6271).
- capture/publish: exactly one checkpoint publishes per invocation
  (main.rs:6341 prompt-boundary is *replaced* by main.rs:6616 completed-turn
  when eligible). Completed-turn eligibility requires preserve-thinking
  messages rendering plus an appended generation prompt (main.rs:2934-2938).
- stderr publish line: `durable_prefix_cache: publish=… capture=…
  matched_tokens=… … capture_ms=… publish_us=…` (main.rs:6675/6709).
- `stats:` line is tracing `qwen_diag` as of 1ecf208; runner sets
  `RUST_LOG=warn,qwen_diag=info`.

Scope notes (frozen):

1. Legacy `--messages` renders via the generic Qwen ChatML renderer with
   `QwenGenerationMode::Auto` for all Qwen-family checkpoints, including
   Qwen3.8-27B (main.rs:2924-2938 → messages.rs:195-217). The modern
   Qwen3.8 pre-closed-history renderer is NOT exercised here; strip mode is
   the proxy for reasoning-drop history contracts. Bespoke-renderer
   stability is S1 golden-test scope.
2. Tool calls are simulated as assistant text (`<tool_call>` JSON) and tool
   results as user-turn JSON, since no tool role exists on any current
   surface. This measures render/checkpoint arithmetic, not tool semantics
   (jam round 2, P2: user-embedded results measure reasoning-drop divergence
   faithfully; tool-role render stability is a pure string test that needs
   no checkpoint store).
3. DeepSeek V4 cells are deferred: the full model is ~90 GB and prior
   evidence already documents its durable-cache behavior and the
   completed-turn reasoning re-prefill residual (PERF-LOG 2026-08-13 GO
   entry). DSpark on disk is a speculative-decoding drafter, not a chat
   surface.
4. Amendment (pre-execution, 2026-08-18): the owner's standing position is
   that reasoning history should be preserved unconditionally — for all
   sessions, not only tool loops — in contrast to the conditional
   preserve/drop contracts of DS4 and Qwen3.8. S0 therefore treats
   preserve-vs-strip as the *default rendering policy* question, and
   measures only its economic term (the strip tax in tokens and prefill
   ms). The behavioral term — whether each family degrades, drifts, or
   improves with preserved reasoning history, given that Qwen3.8/DS4
   training contracts drop it — is out of S0 scope and requires a separate
   behavioral packet per family before preserve-by-default is adopted on
   surfaces where it deviates from the upstream contract. Qwen3.6
   preserve-by-default is already validated by sustained practice (the
   game harness defaults to it).

## Matrix

| Cell | Model | Render mode | Lane | Turns |
| --- | --- | --- | --- | ---: |
| smoke | Qwen3.5-0.8B-Q8_0 | preserve | scripted | 3 |
| smoke-strip | Qwen3.5-0.8B-Q8_0 | strip | scripted | 3 |
| a3b-preserve-scripted | Qwen3.6-35B-A3B-UD-Q4_K_S | preserve | scripted | 10 |
| a3b-strip-scripted | Qwen3.6-35B-A3B-UD-Q4_K_S | strip | scripted | 10 |
| q38-preserve-scripted | Qwen3.8-27B-Q4_K_M | preserve | scripted | 10 |
| q38-strip-scripted | Qwen3.8-27B-Q4_K_M | strip | scripted | 10 |
| a3b-preserve-live | Qwen3.6-35B-A3B-UD-Q4_K_S | preserve | live | 5 |
| a3b-strip-live | Qwen3.6-35B-A3B-UD-Q4_K_S | strip | live | 5 |

- **Scripted lane**: fixed 10-exchange agent-loop transcript (system prompt
  >1024 tokens so turn 1 publishes; assistant turns contain `<think>` blocks
  plus `<tool_call>` JSON; user turns alternate tasks and tool-result JSON).
  Turn *t* submits history through user turn *t*; generation is capped at 32
  tokens and discarded. Measures pure re-render prefix arithmetic.
- **Live lane**: like the game harness — each turn's actual stdout is
  appended as the next assistant turn (temp 0, seed 42, max 4096 tokens).
  Measures real checkpoint economics including completed-turn checkpoints.
- Fresh private cache dir per cell; one process per turn (stateless
  invocation model); binary `target/release/qwen` at 1ecf208 on Apple M4 Max
  (unified memory), models under `~/models`.

Per-turn record: `checkpoint_hit`, `matched`, `restored`, `exact`,
`restore_total_ms`, publish outcome/kind/`matched_tokens`/`capture_ms`,
`prompt_tokens`, `generated_tokens`, `stop_reason`, `load_ms`, `prefill_ms`,
`ttft_ms`. Derived: `tail_t = prompt_tokens_t − restored_t` (tokens the turn
must prefill), and its wall cost via `prefill_ms`.

## Amendment 2: smoke findings (pre-primary-execution, disclosed)

The smoke cell (pipeline validation) plus code reading forced two
mechanical corrections before any primary cell ran:

1. **Whole-blob-prefix matching.** The durable lookup admits a blob only
   if the request's token prefix at exactly the blob's stored length
   digest-matches (checkpoint_store.rs:393-405). There is no
   arbitrary-cut restore within a longer blob. Smoke turn 3:
   `candidates=0 matched=0` despite sharing its first 1,207 tokens with
   the turn-2 blob, because that blob's tail was generated text the
   scripted history does not reproduce.
2. **Publication shadowing.** One checkpoint publishes per invocation;
   completed-turn capture replaces prompt-boundary capture
   (main.rs:6341 → 6616). In preserve mode with an appended generation
   prompt, every published blob therefore embeds generated output, and
   any client that does not echo that output byte-verbatim gets zero
   reuse — permanently.

Consequences, disclosed before primary execution: P1 splits into P1a/P1b
below (original P1 predicted scripted hits in both modes; that was wrong
for preserve, for publication-policy reasons, not render instability).
The preserve-scripted cells become negative controls documenting
diverging-echo client fragility; preserve-mode render stability is
carried by the live-preserve cell (P3), whose hits require render
stability AND verbatim echo jointly. A strip-mode smoke cell was added to
validate the hit path. S1 design input recorded: the facade must either
guarantee byte-verbatim assistant echo (opaque items do this
structurally) or publish both boundaries per turn.

Harness repair (not a prediction change): system prompt lengthened ~90
tokens so turn-1 prompts exceed the 1,024-token durable minimum (smoke
turn 1 was 987).

## Predictions (frozen; P1 amended per Amendment 2)

- **P1a (strip-scripted, both families):** prefix-stable. From turn 2:
  `checkpoint_hit=true`, `capture=prompt`, and
  `matched_t ≥ prompt_tokens_{t−1} − 2`, growing monotonically. Rationale:
  strip renders deterministically and idempotently on the shared history,
  and strip-mode publishes prompt-boundary blobs, which the next scripted
  prompt extends verbatim.
- **P1b (preserve-scripted, both families):** zero reuse at every turn:
  `candidates=0, matched=0`, `capture=completed`. Not render instability —
  publication shadowing plus whole-blob-prefix matching (Amendment 2).
  This is the negative control for clients that post-process assistant
  text before resending.
- **P2 (scripted tail):** `tail_t` ≈ tokens(assistant_{t−1} rendered per
  mode) + tokens(user_t) + template overhead; strip-mode tails smaller than
  preserve by ≈ the think-block token count.
- **P3 (live preserve):** completed-turn checkpoints publish
  (`capture=completed`) and match through the prior *response*:
  `matched_t ≥ prompt_tokens_{t−1} + generated_{t−1} − 2`, so
  `tail_t` ≈ tokens(user_t) only.
- **P4 (live strip):** completed-turn ineligible (`capture=prompt` only), so
  `matched_t` tracks the prior *prompt* boundary and
  `tail_t` ≈ tokens(stripped response_{t−1}) + tokens(user_t). The
  preserve-vs-strip delta in prefill_ms quantifies the reasoning-drop tax
  that CC-style reasoning-dropping clients would pay per agent-loop turn.
- **P5 (overheads):** at these context lengths (≤ ~8k), `restore_total_ms`
  < 200 ms and capture+publish < 300 ms per turn on both primary models.

## Decision rules

- **G1:** If any scripted cell shows non-monotonic `matched` (a reset), the
  generic renderer is not prefix-stable → S1 renderer requires redesign and
  a repeat of this packet before server work proceeds.
- **G2:** If P3 holds and P4 shows a materially larger tail (as predicted),
  preserve-mode rendering is economically free (zero tail) while strip
  carries a quantified per-turn tax. The facade's default-policy decision
  (preserve unconditionally, per the owner's position in scope note 4) then
  rests solely on per-family behavioral evidence, to be preregistered
  separately; S3's zero-tail-re-prefill gate generalizes from
  tool-session optimization to default-contract property wherever
  preserve-by-default is adopted. CC-shim sessions that drop reasoning
  carry the measured tax, documented rather than unknown.
- **G3:** If P5 fails (overheads ≥ the S1 turn-2 TTFT budget), the S1 gate
  (<150 ms warm turn-2 TTFT @8k) must be renegotiated before S1 commits.

## Artifacts

- `run_s0.py` — uv runner (stdlib only), one process per turn, JSONL rows +
  summary tables.
- `artifacts/<cell>/results.jsonl`, `artifacts/<cell>/turn-*.stderr.log`,
  summary in `RESULTS.md` (written after execution; not part of this
  preregistration).
