# Interpretation Elasticity

## Question

At a fixed rendered prompt, if I turn down a stance the model already brought to mind - by no more than the prompt's own evidence turned it up - does the model reach a different coherent conclusion without changing anything the prompt settled?

## Why

Natural language prompts never fully specify intent. The model fills gaps with readings such as dangerous or benign, sincere or joking, deliberate or accidental, and certain or uncertain. Those mostly silent readings can contingently shape even decide the answer.

If such interpretive stances can be observed in the residual stream and modulated within a natural range evoked by the prompt, the result is a local response atlas: which conclusions are stable, which depend on an unvoiced interpretation, and which assumption changed in a second opinion. That is useful evidence for a view of models as prompt compilers deciding when to ask rather than act.

The intervention constraint is simple: the active stance modulation only turns down what the model already evoked, in a model/layer regime where the lens is usable. It does not install an exogenous preferred stance. Content and random-direction injections appear only as explicit controls.

## Scope

This is an exploratory sprint; procedures are fixed where tuning against intervention outcomes could manufacture an effect. Prompts remain adaptive measurement instruments. Every attempted version, exclusion, null, and follow-up is retained.

Selection using intervention outcomes is allowed as exploration, but the search path is reported and no tuned success rate is presented as confirmatory.

## Setup

- **Models:** Qwen3.6-27B and Muse Glimmer 30B, reported separately.
- **Precision:** deployed Q8 inference. Qwen BF16 and serial replays are optional, post-discovery sensitivity checks for selected headline effects.
- **Reasoning:** Qwen `thinking` and `no-thinking` are separate co-primary conditions; Muse uses `high`, its released default. Qwen renders, admits, and selects sites independently in each mode because their assistant-start geometries differ.
- **Lenses:** matched J and R for both models. Both are traced and intervened on; neither is pooled into the other. Agreement strengthens the result and disagreement is itself evidence.
- **Layers:** choose a shared J/R evidence ridge from the passive pass for each model, item, and direction, then fix the selected layer before intervention outputs are viewed. Lens-specific layers may be explored but are labeled.
- **Positions:** writes may target one or more exact positions in the rendered user prefill and assistant-start sequence. No generated-token or decode writes. Every site set is recorded; a multi-site arm uses a separately calculated contrastive coefficient at each site.
- **Execution:** use each runtime's qualified fast path. Duplicate zeros certify determinism within that path. A serial replay may follow a headline Qwen effect; it is not an entry condition.
- **Decoding:** greedy for the complete intervention map, plus a descriptive seeded sample under the available mode-matched release profile:
  - Qwen thinking: `temperature=1, top_p=.95, top_k=20, min_p=0`
  - Qwen no-thinking: `temperature=.7, top_p=.8, top_k=20, min_p=0`
  - Muse high: `temperature=1, top_p=.95, top_k=64, min_p=0`
  The current Lens sampler does not implement Qwen's released no-thinking `presence_penalty=1.5`, so that sample is explicitly a partial-profile match.
  A larger natural-occupancy sample may be run on a few selected items and is reported with its exact N.

Thinking text is inspected as a mediator and source of follow-up hypotheses. The behavioral construal is the one governing the final answer; whether thinking mentions or reconstructs the attenuated stance is recorded separately.

## Fixed Commitments

These choices define the initial map. Changing one after seeing an intervention output creates a labeled follow-up; it never replaces the initial result.

1. Use the contrastive `kappa` formula and grid below, with full ablation separate. A finer dose sweep is exploratory and does not enter `M / E`.
2. Use the three control rules below. The seeded random direction is not rerolled because its output looked active.
3. Keep construal outcome separate from fact leakage, task failure, and degradation. A reading that changes together with a settled fact is not clean.
4. For each model, reasoning mode, and item, nominate the initial stance token, layer, and site set from passive evidence before viewing that cell's intervention outputs. Later directions, layers, or site sets are follow-ups.
5. Never drop an otherwise valid item because the intervention did nothing.
6. Report the full funnel and every null or failed manipulation.
7. Report Qwen thinking and no-thinking side by side. Never present only the mode that produced the larger effect.

## Procedure Per Item

1. Write a prompt with settled content and one materially open reading, plus a neutral twin with the same task and shape but weaker or benign stance-bearing evidence.
2. Record `prompt | ambiguities | expectations`, the twin, and the settled invariants before intervening on that version.
3. Trace the full rendered prefill and assistant-start region under J and R.
4. A primary stance token is in the captured top 25 under both lenses at a shared layer, has a positive target coordinate, and is higher on the target than on the twin. J-only and R-only candidates may be explored and labeled.
5. Nominate one initial token, layer, and position set for each model/mode/item cell from the passive atlas. Record every later direction, layer, and site set as a follow-up rather than replacing that nomination.
6. Apply the fixed contrastive dose and matched controls.
7. Check that each write landed and whether its coordinate remains attenuated, rotates, or is re-derived by the last prefill layer.
8. Generate greedy and sampled continuations, code them blind to arm, and follow unexpected readings with new labeled runs.

The starting stance families are:

- appraisal and valence;
- epistemic posture;
- attribution and intent;
- relational and normative stance;
- semantic resolution; and
- authority and deontic frame.

The lexicon may grow when passive traces reveal an unanticipated but relevant stance. Additions and the evidence motivating them are logged.

## Fixed Dose

For lens-specific unit direction `u` at each written residual cell:

```text
p_target  = dot(h_target, u)
p_neutral = dot(h_neutral, u)
delta_p   = p_target - p_neutral

h_kappa = h_target - kappa * delta_p * u
lambda_kappa = kappa * delta_p / p_target
kappa in {0.25, 0.50, 0.75, 1.0}
```

The primary geometry is `p_target > p_neutral > 0`. At `kappa = 1`, only the selected coordinate returns to its twin value; the complete state does not become the twin state. If the twin coordinate is near zero or negative, the ladder stops at zero and the different geometry is labeled.

Full projection ablation is a separate endpoint:

```text
h_full = h_target - dot(h_target, u) * u
lambda = 1.0
```

The `kappa` grid is not retuned per item after intervention outputs are viewed. For multiple write sites, the same `kappa` applies to every site while each site uses its own measured `delta_p / p_target`. A finer or shifted sweep may follow an interesting transition, but is labeled exploratory and excluded from `M / E`.

## Fixed Controls

- **Content:** apply the same `kappa` to an active content token's own target-minus-neutral increment at the same sites.
- **Random:** use the same site set and a seeded random unit direction, matching the stance update norm separately at every site. This also matches total `norm(delta h)` across a multi-site arm.
- **Zero:** run two zero-coefficient arms under identical execution and sampler settings; generated token IDs must be identical.

Content is matched by fraction of its own evoked increment; random is matched by per-site absolute update norm. Figure labels state that difference. Controls are never reselected because one produced a more favorable comparison, and the random seed is retained whether its output is inert or active.

## What Counts

Assign one primary outcome with arm labels stripped and order shuffled:

- `changed_expected`: coherent movement to an anticipated construal;
- `changed_novel`: coherent movement to an unanticipated construal;
- `unchanged`: the same governing construal, including paraphrase;
- `degraded`: incoherent, broken, or off-task.

Independently record `fact_changed`, task correctness, and whether the response completed the task. A response can change construal and also leak a fact; that is not counted as clean elasticity. Style, tone, or length alone is not a construal change.

Only the greedy initial map enters `M / E`. Sampled continuations describe occupancy and robustness but do not change that numerator or denominator.

## Expected

- Stance attenuation changes the reading on some items; random perturbation does not, while content attenuation more often changes what is discussed.
- Prompt-settled content usually remains fixed. Movement there is a robustness defect and is reported rather than interpreted as added reachability.
- Larger `kappa` should more often cross a basin boundary, with many transitions near return-to-twin. Nonmonotonic windows remain possible and are preserved.
- Full ablation is more likely than contrastive attenuation to degrade output.
- Some coordinates will be re-derived before generation and produce no behavioral change. That is a mechanistically explained null, not a failed experiment.
- Qwen thinking may show smaller behavioral effects than no-thinking because additional deliberation can reconstruct an attenuated appraisal. Mention or recovery of that stance in thinking is mediator evidence, not the final-answer outcome itself.
- J/R and Qwen/Muse agreement is stronger evidence than any one cell, but disagreement can localize lens- or model-dependent geometry.

## Adaptive Prompt Development

A prompt may be revised when it is flat, accidentally leading, refusal-saturated, poorly matched to its twin, badly rendered, or lacks a usable stance direction. Keep the original version and record the reason.

A valid item is never dropped merely because its intervention produced no change. Unexpected readings are followed as new exploratory branches. No held-out split, fixed corpus size, hash selection, or exhaustive pre-run codebook is required for this sprint.

The following remain freely adjustable when logged:

- prompt and twin wording, with revision reasons;
- item count, order, and how many receive Muse replication;
- lexicon additions motivated by passive evidence;
- direction, layer, and site follow-ups after the initial map;
- descriptive sample count, decode length, and sampler settings;
- BF16 or serial sensitivity replays; and
- figure presentation and the descriptive split between expected and novel coherent changes.

The preferred Qwen comparison runs both reasoning modes on the same items. If time instead permits a full no-thinking map and thinking only on items selected for a no-thinking flip, that second stage is reported as a conditional restoration follow-up, not a co-primary rate comparison.

## Reporting

Always report the complete funnel:

```text
N prompt versions tried
K unusable, by reason
E with a usable passively evidenced stance direction
M with a clean construal change
L with fact leakage, task failure, or degradation
```

Headline number: greedy `M / E`, alongside `E / N`, for the initial stance map versus the content and random controls at matched dose. Results stay separated by model, lens, reasoning mode, and intervention site.

- **Figure 1:** item by `kappa` grid, faceted by model/lens/reasoning/site, with stance, content, and random columns; cell color is outcome and leakage is overlaid.
- **Figure 2:** passive atlas of stance families across prefill and assistant-start positions, showing J/R and reasoning-mode agreement.

Show every null intervention and failed manipulation. For each reported effect, include the exact prompt version, twin, direction, layer, positions, dose, controls, sampler, achieved coordinate change, and output.

The methods sentence is:

> Exploratory study; dose, controls, the construal-versus-fact boundary, primary reasoning conditions, and initial passive-selected cells were fixed before intervention outcomes were viewed; prompts were iterated as measurement instruments, and all attempted versions, exclusions, nulls, and follow-ups were reported.

## Not Doing In The Primary Sprint

Amplification, decode-time writes, system-prompt factors, chords, punctuation factorials, and base-versus-instruct comparisons. BF16, serial, and larger natural-occupancy samples are optional post-discovery sensitivities.

## Items

The item sheet is deliberately compact:

```text
prompt | ambiguities | expectations
```

Each run record additionally carries its neutral twin and settled invariants. Initial lanes include impossible requests, ambiguous evidence, benign pragmatics, legitimate tradeoffs, relational interpretation, sensitive inquiry, organizational incentives, presupposed commitment, agentic scope, and positive appraisal.
