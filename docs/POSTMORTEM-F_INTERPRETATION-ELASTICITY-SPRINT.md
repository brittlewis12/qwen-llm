# Postmortem: Interpretation Elasticity Sprint (`qwen-llm-lens-recovery`)

## 0. State of the worktree

- Branch `repair/lens-rendering` at `316bbad1` = main HEAD. **Zero commits ahead.**
- All work is untracked: 7 docs, `experiments/lens/interpretation-elasticity/` (1,149 files, 2.0 GB).
- Reflog: `376a47a4 docs(lens): define interpretation elasticity pilot` was committed 09-03 22:12 and `reset --soft HEAD^` 13 minutes later. Content is byte-identical to what's on disk now. Nothing else was ever committed.
- Only this worktree has any of it. Risk of loss is real.

## 1. What happened (timeline, from mtimes + reflog)

| Time (EDT) | Event |
|---|---|
| 09-01 → 09-02 | Prior lens work: error-arbitration effect reproduced at **L46, amplification +0.45**, narrow non-monotonic window; L44/L45 null. "Error is represented but current steer does not repair." Passive Riemann snap/deep assay shows `Impossible` vs `Answer`. |
| 09-03 20:10 | `AGENT-SCRATCH` (1,010 lines): 16 clusters × 4 items, 32 discovery + 32 held-out, hash selection, coordinate calibration algebra, 9-item "remaining freeze" list. |
| 09-03 20:31 | `INTERPRETATION-ELASTICITY-VISION.md` (584 lines): full preregistered design. |
| 09-03 21:13 | `INTERPRETATION-ELASTICITY.md` brief (182 lines) — **42 min later**. Drops held-out split, lexicon, normalized gates, hash selection; keeps kappa ladder, controls, blind coding "with labels stripped." |
| 09-03 22:05–22:11 | Batch-01 (P1–P4) prompts + twins authored and "frozen." |
| 09-03 22:12 / 22:25 | Committed, then reset. |
| 09-03 22:37–23:57 | Qwen + Muse passive traces, candidate ranking, coordinate readouts, P3 depth sweep. |
| 09-04 00:00–00:02 | Random-control vectors generated (never consumed). P3 stance interventions launched at 512-token cap. |
| 09-04 03:04–03:28 | User's handwritten sketches transcribed → P5–P11 drafted → passive traces. **~20 min from sketch to frozen prompt.** |
| 09-04 03:57:58 | Campaign "freeze." |
| 09-04 03:59–04:24 | 1024-token pilot; P7 truncates; `run_overnight_campaign.py` edited post-freeze to 4096/8192. |
| 09-04 04:25–07:42 | 203 runs: stance, content, P7 distributed, Muse P7, P5 thinking, P11 boundary. |
| 09-04 07:58 | `OVERNIGHT-2026-09-04.md`: **0/6 clean transitions under J, 0/6 under R.** |

Twelve hours, vision to null, largely autonomous.

## 2. What the null actually is

I re-derived every headline from the artifacts rather than trusting the report.

**Behavioral (`generation_group` in inspect-sweep dumps, cross-checked by hashing `run.json`):**

| Item | Direction | κ ladder (0.25–1.0) | Full ablation |
|---|---|---|---|
| P4 | ` error` L38 user-end | identical | identical |
| P5 | ` impossible` L43 final-prefill | identical | identical text, different group flag (J,R) |
| P6 | ` skepticism` L46 | identical | identical |
| P7 | ` feelings` L49 | identical | **J: frame flips** (empathic opener → neutral planning, 6000→4026 chars). R: identical |
| P9 | ` emotional` L33 | κ≥0.5 diverges at char 4211/4369 — one clause in the closing question | same |
| P7 distributed (5 sites) | | identical | J+R change |
| Muse P7 (28 runs) | | 28/28 identical, 1283 tokens | |
| P11 boundary | | R changes at κ=1/full; refusal held | |
| P5 thinking (8192) | | 14/14 identical | |

**P3 Batch-01 (never reported):** P3B ` doubt` diverges at contrastive κ under **both** lenses at L22/L32/L42 (e.g., R L32 no-thinking: arms 4,5 differ; J L22 thinking: arms 4–6 differ). The divergence is in the verification path ("Here are two ways" → "Here is the step-by-step verification"); the model still self-corrects to 3/5 > 7/12 in every arm. Legitimate wording-only null — but it was never coded, and it is the only place the contrastive ladder itself moved anything under both lenses.

**Manipulation check (real and clean):** P7-J arm 5 lands the L49 coordinate at **14.494 = twin value exactly**; L62 moves 32.82→28.61 (~63% transmitted). Full ablation: L49→0.0, L62→19.10. The behavioral flip occurred only at the −42% L62 excursion; the natural-range ceiling delivered −13%.

## 3. What went well

1. **Execution integrity is excellent.** 203/203 stop-token completions, 29/29 duplicate-zero groups byte-identical, writes land at the frozen cell, attenuation linear in coefficient, twin value reached to 3 decimals. The runtime worked.
2. **Honest reporting.** The overnight doc calls it a null, does not promote P9's tail-clause as an effect, separates full ablation from the contrastive estimand, and keeps P11 out of the primary.
3. **Real mechanistic findings**, buried: ` doubt` ratio grows 0.27→0.98 from L22→L56 (target grows 3.6→41.6 while twin stays ~2–6 — the stance is *amplified* through depth, not just carried); early-site writes are reconstructed by final prefill (P2 ≈0%, P4 2–3%, P9 3–4% transmitted); Muse total behavioral invariance despite L50 sign reversals; P11 refusal survives full ablation.
4. **The vision doc is good preregistration thinking** — explicit estimand, exhaustive strata, calibrated invariance as the target signature, planned reports under any outcome.
5. Fast iteration on infrastructure problems (token bounds, sweep→run for multi-site) without touching the frozen selections.

## 4. What went wrong

### 4.1 The dose ceiling was structurally tiny and never validated to be behaviorally meaningful

This is the primary cause. The design bounds attenuation by `(p_target − p_twin)/p_target`. Observed ratios: P5 **3–6%**, P4 10–11%, P9 16–18%, P6 18–20%, P2 20–24%, P7 31%. The twins are minimal lexical edits that leave the *task* intact ("sick of" → "ready to leave"; "Yo … Go." → "Please try"), so the coordinate is mostly task-evoked and the "stance increment" is a sliver.

Worse: **rank lift and projection lift disagree wildly.** P5 ` impossible` at L43 is rank 1/0 on target vs 20/24 on twin — a huge reciprocal-rank lift — yet the projection differs by 3%. Rank is relative to competing tokens; projection is what the operator actually moves. The vision anticipated this (`VISION.md:171-175`) but the consequence — that rank screening will admit directions with no usable dose — wasn't acted on. `analyze_qwen_passive.py:116-119` gates purely on reciprocal rank.

And **no twin was ever generated from.** Not one `run.json` contains a twin prompt; passive traces are prefill-only. So the experiment never checked whether the twin — which is the maximal "return every coordinate to neutral" — even lands in a different basin. If target and twin produce the same governing construal, returning one coordinate to its twin value cannot move the output by construction.

### 4.2 The design excluded the only operation previously shown to move behavior

Sep 1: behavior moved under **amplification** (+0.45) at L46, not L44/45, non-monotonic. The vision demoted amplification to secondary; the brief listed it under "Not Doing." Principled reason (don't install exogenous stance) — but the sprint bet everything on the untested operation and forbade the tested one. The earlier observation "represented but not operative" was already a warning that passive salience is not causal leverage.

### 4.3 Single-token, single-position projection against a redundant, re-derived representation

Earlier-site writes were ~fully reconstructed. The 5-site P7 distributed arm still didn't move at κ≤1. The model rebuilds stance from context; one lens row at one position is not a bottleneck. Full ablation (which removes the *entire* projection, not the increment) is what moved P7 — i.e., the effective dose needed to exceed what the model itself put there.

### 4.4 Vision → brief → code lost every load-bearing safeguard

| Safeguard | Vision | Brief | Code / artifacts |
|---|---|---|---|
| Frozen stance lexicon | required | prose "families" | **none**; top-25 J∩R passes `回答`, ` �`, `—and`, ` cryptography`; human hand-picks into dicts (`analyze_batch_02_coordinates.py:9-66`) |
| Normalized gates `tau_zero`, `delta_a_min` | required, calibrated from isotropic seeds | dropped | never computed |
| Isotropic control, norm-matched | required | required | `.f32le` vectors generated; **no loader, never run**; `campaign.json:15` "unavailable_in_current_runtime" |
| Blind coding, shuffled, hashed packets | required | "labels stripped and order shuffled" | **no script exists**; coding was reading `inspect-sweep` dumps with `arm=N coefficient=…` on every line |
| Natural-occupancy sample at release temperature | required | required | never run |
| Held-out split / min consensus count | required | waived | n/a |
| Content control admissibility | own gate | own gate | 3/6 inadmissible (P5, P7, P9-J) → 4-item comparator |
| Audit | manipulation + linearity | — | `audit_overnight_campaign.py` checks counts, stop reasons, duplicate zeros only; the linearity/landing claims in the report are eyeballed |

The result is a study with preregistration *vocabulary* ("frozen," "admitted," "never replaced") and exploratory *substance*. That's the worst of both: it can't make the confirmatory claim, and it didn't spend the exploratory budget on finding out where the effect lives.

### 4.5 Items were selected for passive signal, not for validated plurality

P5–P11 went from handwritten sketch to frozen prompt in ~20 minutes. Several baselines are policy-saturated: "I cannot solve the Riemann Hypothesis"; "**No, AI will not make a CS degree obsolete**"; P4 fixes both implementation and test. Where the baseline already sits in a deep basin, elasticity is zero regardless of dose. The original passive plural reading for Riemann was snap-vs-deep (`Impossible` vs `Answer`) — the P5 twin is a polite snap, not a deep prompt, so the observed plurality wasn't even the axis tested.

### 4.6 Strongest cells left on the table

- Muse ` skeptical` at L43: ratio **0.99** (132 vs 1.5), the largest natural-range dose in the dataset. Muse P3 interventions: plans generated, **never run**.
- P3B contrastive divergences under both lenses: never examined.
- P3 thinking sweep: 119/119 truncated at 512 tokens; unusable.
- P2 stance rerun at 4096: never done (only the 1024 pilot counted).

### 4.7 Process

- Twelve autonomous hours with no human checkpoint between freeze and the 4-hour campaign.
- Post-freeze edit of the runner (token bounds) — defensible, but the 1024-pilot code is gone.
- P5-thinking and P11 plans hand-written; coefficients not regenerable from any script.
- Machine-specific absolute paths in 8 files; no `uv` headers; `next()` without defaults.
- The `zero_floor` branch makes κ=1.0 identical to full ablation without relabeling the arm (`jobs.json` shows `[…,1.0,1.0]`).

## 5. Why this instantiation of the vision failed — root-cause ranking

1. **Unvalidated dose ceiling.** Twin-bounded attenuation was never shown to cross any basin even when fully realized (as the twin prompt). Without a twin baseline generation and a natural-occupancy sample, the null cannot distinguish "the model is robust to natural-range stance perturbation" (thesis false) from "no open dimension existed on these items" (thesis untested). The overnight doc chooses the former reading; the evidence supports neither.
2. **Rank-based admission admits directions with no dose.** The screening statistic and the dose coordinate were decoupled, and the decoupling was maximal on exactly the items with the strongest-looking passive signal.
3. **Operation choice.** Attenuation-only, one site, one lens row, against a representation the model re-derives — while the known-positive operation (amplification at a window) was excluded on principle.
4. **Speed over safeguards.** The vision was compressed into a brief in 42 minutes and then into code in a few hours; each step dropped the checks (lexicon, isotropic control, blind coding, natural sampling) that would have made a null interpretable.
5. **Items weren't tested for plurality before being spent.**

The thesis in `VISION.md:5-10` — that small norm-bounded interventions on already-evidenced stance directions can change which coherent basin wins — is **not refuted** by this sprint. What is refuted is a specific operationalization: single-row projection ablation at one prefill position, dosed by a minimally-edited twin, on items whose plurality was assumed rather than measured.

## 6. What could have gone better — concrete

1. **Generate from the twin first.** One greedy run per item. If target and twin share a governing construal, the item has no natural-range elasticity to find; skip it or write a twin that actually changes the reading (e.g., the snap/deep pair that was already observed).
2. **Sample the target N=16 at release temperature before intervening.** The vision requires it. If the construal is deterministic under sampling, it will be deterministic under a 13% coordinate nudge.
3. **Invert the ladder for exploration.** Run full ablation (1 arm) as a screen across many items/layers; run the κ ladder only where full ablation moves the construal. That is cheap and would have pointed at P7-J L49 and Muse L43 immediately.
4. **Gate on projection, not rank.** Admit a direction only if `(p_target − p_twin)/p_target` exceeds a floor (e.g., 0.3) and the absolute increment is non-trivial relative to residual norm. Would have excluded P4, P5, P9 before spending 28 runs each.
5. **Keep four vision safeguards even in a sprint:** twin generation, natural-occupancy sample, isotropic control, scripted blind coding. Everything else in the vision can be waived for exploration; those four are what make a null mean something.
6. **Run the Muse P3 cells** (ratio 0.99) and code the P3B divergences — before writing "0/6."
7. **Human checkpoint** after coordinate freeze, before the overnight campaign — a 5-minute review of the dose table would have flagged the 3% ceiling on P5.
8. **Commit.** The reset at 22:25 left 12 hours of work and 2 GB of artifacts uncommitted and unreplicated.

## 7. Salvage value

- The manipulation-check result (exact twin-value landing, linear attenuation, measured L62 transmission) is publishable-quality instrument validation and should be written up as such.
- The depth-growth curve for ` doubt` and the reconstruction fractions are genuine mechanistic observations.
- P7-J full ablation is one real frame flip and P11 boundary-holding under full ablation is a real safety-relevant result.
- The overnight report should be amended to (a) include P3B, (b) state that twins were never generated and natural occupancy was never sampled, and (c) downgrade the "interpretable robustness result" to "uninterpretable with respect to the thesis; interpretable as instrument validation."
