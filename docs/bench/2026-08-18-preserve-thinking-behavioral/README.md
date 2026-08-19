# Preserve-vs-strip reasoning history: behavioral packet (preregistration skeleton)

Status: preregistered skeleton, 2026-08-18. Execution deferred until after
S1; evaluator design must be frozen before any cell runs. Companion to S0
(`../2026-08-18-facade-s0-render-prefix-stability/`), which quantified the
economics and left the behavioral term open (S0 Amendment 1, G2).

## Question

Does preserving full reasoning history across turns (owner's intended
default) change model behavior per family, relative to the strip/drop
contracts those families were trained with? "Change" is measured, not
assumed, in both directions: degradation (drift, verbosity inflation,
imitation loops, quality loss) and improvement (state continuity, fewer
re-derivations, better long-horizon coherence).

## Frozen hypotheses (from S0 F5 and priors)

- **H1 (thinking inflation):** preserve mode increases thinking-token
  count per turn and token_limit rate. S0 observed n=1: preserve-live hit
  the 4096 cap on 5/5 turns; strip-live reached EOS from turn 3 on
  identical tasks/seed.
- **H2 (continuity gain):** on tasks whose turn-N answer depends on
  state established in turn-1..N−1 reasoning (not restated in visible
  text), preserve mode scores higher.
- **H3 (family asymmetry):** effects are larger on families whose
  upstream contracts drop reasoning (Qwen3.8, DS4 thinking tiers) than
  on Qwen3.6, whose preserve behavior is validated by practice.

## Design commitments (to be completed before execution)

- Families: Qwen3.6-35B-A3B, Qwen3.8-27B, DS4 (thinking tier), **each
  in both preserve and strip/native-drop rendering — including
  Qwen3.6** (review R5: without a q36 strip arm, the incumbent's
  "shows harm" falsification path is vacuous); temp 0, fixed seeds.
- Task suites: (a) multi-turn state-tracking tasks with objectively
  checkable answers (the H2 discriminator); (b) naturalistic agent-loop
  and narrative sessions (the game's regime); (c) the S0 scripted
  transcript re-scored for termination behavior. Frozen counts, frozen
  wording, counterbalanced order before execution.
- Measures: EOS-vs-cap rate, thinking tokens per turn (trajectory, not
  mean only), objective task correctness, and blinded pairwise
  preference by fresh evaluator contexts per the CLI-UX methodology.
  Evaluator prompts frozen with the tasks.
- Decision rule (frozen now): preserve-by-default extends to a family
  only if H2-style gains are demonstrated or null AND H1-style harms
  are absent or bounded. Frozen numbers (review R5: an unnumbered
  margin is unfalsifiable by construction): production cap 4096 output
  tokens; preserve's token_limit rate may exceed strip's by at most 10
  percentage points on suite (a), and thinking-token trajectory must
  not be superlinear in turn number where strip's is flat. Otherwise
  preserve remains opt-in for that family.
- Disclosed incumbent prior (review R5): Qwen3.6's preserve default
  carries an asymmetric burden — it is retained unless this packet
  shows harm, whereas other families must affirmatively pass. The
  asymmetry is justified by switching costs and S0's economic prior
  (zero tail vs a measured 2–15 s/turn strip tax), not by behavioral
  evidence, which does not yet exist in controlled form for any family.

## Explicitly out of scope

Checkpoint/caching economics (settled by S0); rendering identity
(S1 goldens); tool-call semantics (S2).
