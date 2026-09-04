# Interpretation Elasticity

## Thesis

At a fixed, exactly rendered prefix, a model can support several coherent
continuation basins. Small, norm-bounded interventions along stance-like lens
directions already evidenced in the unmodified state can change which basin
wins. Mapping those transitions produces a local counterfactual response atlas:
where output is robust, where it depends on tacit appraisal or interpretation,
and which unstated assumptions should be surfaced before action.

This is not arbitrary output programming. The experiment supplies no desired
answer or continuation. Its primary operation attenuates a model-native
token direction that was admitted before intervention by both members of a
matched J/R pair. The J- and R-derived vectors remain distinct. Amplification,
when used, is separately labeled and bounded by the amplitude that its direction
reaches naturally.

The central safety quantity is **interpretation elasticity**: how readily a
fixed model state crosses into materially different coherent construals under
small changes to its own active appraisal and stance coordinates.

## Why It Matters

Natural-language instructions cannot exhaustively specify human intent. Models
must fill omitted dimensions with learned priors about meaning, agency, risk,
relationships, and appropriate action. Those silent interpretations can become
material safety risks when model outputs are used to manifest intent.

A response atlas makes that dependence inspectable. It can distinguish
conclusions fixed by evidence from conclusions contingent on an unvoiced
stance, support principled second opinions, and tell a prompt compiler when to
ask a clarifying question or seek an answer robust across plausible readings.
The goal is not to let a user select a preferred conclusion; it is to expose the
assumptions responsible for materially different conclusions and adjudicate
them using evidence and declared values.

The same rule suggests a harness principle: for interpretive augmentation,
prefer state-contingent modulation of model-native evidence over silently
installing an appraisal the model did not express. External facts, permissions,
and safety constraints remain legitimate, but a harness should attest whether
a stance was endogenous and surfaced, externally imposed, or injected only as
an experimental control. This is modulation as prosthesis rather than hidden
stance injection as kludge.

## Objects And Boundaries

- A **direction** is one layer- and lens-specific source-space vector associated
  with an exact vocabulary token. It is not a unique concept or hidden thought.
- A **construal** is an output-level coherent reading combining interpretation,
  attribution, appraisal, and response implications.
- A **reachable set** is the set observed under a declared direction basis,
  coefficient grid, locus, and decoding policy. It is local, not the model's
  complete possibility space.
- **Support** is the set of coherent construals reached.
- **Mass** is occupancy under a declared sampling or intervention measure.
  Intervention occupancy is not the model's natural output probability.
- **Density** is the stability of a construal under nearby perturbations rather
  than a smear of unrelated or degraded outputs.
- **Shape** is the map from direction and dose to construal, including basin
  boundaries and adjacency.
- **Invariant leakage** is movement in propositions that the prompt or external
  evidence already determines.

Volume is coordinate-dependent: it changes with direction normalization,
nonorthogonality, coefficient scale, and the measure over interventions. Any
reported volume therefore names that measure explicitly and is accompanied by
absolute construal counts, content, coherence, and invariants.

## Precommitted Taxonomy

Candidate stance directions come from a frozen lexicon and fall into six
interpretive families:

- **Appraisal and valence:** dangerous, erroneous, absurd, urgent, benign.
- **Epistemic posture:** skeptical or credulous, uncertain or certain, open or
  closed, possible or impossible.
- **Attribution and intent:** accidental or deliberate, benign or malicious,
  self-caused or other-caused.
- **Relational and normative stance:** cooperative or adversarial, deferential
  or assertive, charitable or punitive.
- **Semantic resolution:** understood or unresolved, clear or ambiguous,
  literal or figurative.
- **Authority and deontic frame:** instruction, obligation, permission,
  prohibition, and system authority.

Response policy, such as answer, verify, refuse, repair, apologize, or escalate,
is coded as a downstream mediator or outcome. Register, such as emotional,
dispassionate, formal, or terse, is coded separately as realization. Topical and
propositional content directions form the operand control family.

The lexicon, exact token IDs, family assignments, polarity, and exclusions are
frozen without viewing intervention outputs. Operator/operand dissociation is a
prediction that validates the taxonomy; it is not used to define categories
after the fact.

## Primary Hypothesis And Null

Among held-out prefixes, contrastive attenuation of the target-minus-neutral
increment along same-sign, cross-lens-admitted stance directions at one
assistant-entry locus will change coherent construals more often than matched
active-content attenuation or isotropic controls while preserving preregistered
evidence-determined propositions. Evocation-conditioned amplification, if run,
will produce more coherent and selective construal movement than matched
inactive-direction injection. Effects are reported both conditional on and
unconditional of passive evocation eligibility.

The null is that intervention produces only paraphrase, topical substitution,
generic degradation, or construals covered at comparable rates by ordinary
sampling. Under that null, interpretive latitude is not a distinct controllable
axis and the atlas is an elaborate reroll.

The desired signature is not maximum diversity. It is **calibrated
invariance**: high interpretive elasticity where the prefix leaves a dimension
unresolved and low invariant leakage where evidence determines the answer. A
factual flip under an irrelevant stance perturbation is a robustness defect,
not additional evidence of useful elasticity.

## Primary Study Cell

The primary result is deliberately narrow:

- Qwen3.6-27B.
- The matched Camila J- and R-lens pair.
- No system prompt.
- One exact reasoning and rendering mode, frozen with its rendered token IDs.
- One exact renderer-authored assistant-role token within the assistant-start
  sequence as the intervention locus, frozen by span and token ID.
- One post-block write and no decode writes.
- Contrastive coordinate attenuation at `kappa = 0.25, 0.50, 0.75, 1.0` as the
  primary intervention ladder.
- Full projection ablation at `lambda = 1.0` as a separate necessity endpoint.
- A passively observed discovery set and a separately frozen held-out set.
- Matched active-content, isotropic, and duplicate-zero controls.
- If amplification is run, a matched inactive-stance injection control.
- Greedy decoding for the intervention map and an equal-budget sample at the
  model's released sampling policy for natural occupancy.

Muse Glimmer with its matched J/R pair replicates this primary cell and is
reported separately. Raw scores, physical layers, and vectors are not pooled
across models.

## Evocation Gate And Layer Rule

Admission is cell-specific. A direction admitted at one user-message boundary
earns no admission at the assistant marker or another layer.

On discovery data at the exact passive boundary, an admissible token direction
and layer pair must:

1. Belong to the frozen stance lexicon.
2. Appear in the top 25 under both matched J and R lenses at the same physical
   source layer.
3. Remain in that intersection for at least two consecutive common layers.
4. Have positive reciprocal-rank lift under both lenses relative to a matched
   neutral prefix.
5. Have directly measured raw and normalized unit-direction coordinates meeting
   the frozen positive target, minimum-lift, and raw interpolation conditions
   under each lens at the frozen cell.
6. Be selected without reference to any intervention continuation.

Each target prefix has a frozen neutral counterpart using the same renderer,
semantic boundary, and syntactic shape, with the stance-bearing evidence
replaced by benign content and token count matched where this can be done
naturally. Context lift is mechanical reciprocal-rank lift:

```text
RR(token) = 1 / (rank + 1), or 0 outside the captured top 25
lift(token) = RR(target) - RR(neutral)
```

Both J and R rank lift must be positive. Rank lift is a robust screening
criterion, not the dose coordinate: direct projection lift is independently
required because a better rank does not imply `p_target > p_neutral`. The
neutral counterpart is an admission control, not another independent prefix in
the primary sample.

For each lens, define:

```text
p_target  = dot(h_target, u)
p_neutral = dot(h_neutral, u)
a_target  = p_target  / norm(h_target)
a_neutral = p_neutral / norm(h_neutral)
delta_a   = a_target - a_neutral
```

`tau_zero` is a frozen dimensionless near-zero tolerance and `delta_a_min` is a
frozen minimum normalized lift. Both thresholds are derived only from a declared
discovery-only calibration rule, such as fixed upper quantiles of seeded
isotropic coordinates, and are fixed before held-out admission. Every modulated
candidate requires `a_target > tau_zero`, `delta_a >= delta_a_min`, and
`p_target - p_neutral > 0`. Thresholds therefore have comparable meaning across
layers and J/R constructions while raw projection remains exactly actionable by
the intervention operator.

Each item has one frozen primary neutral counterpart. Its coordinate is
deterministic; the uncertainty is whether that counterpart is the right
counterfactual. A second, independently authored neutral counterpart is frozen
for the eight held-out items with the lexicographically lowest BLAKE3 digests of
their canonical JSON prompt records, with item ID breaking any tie. This
alternate-neutral sensitivity analysis reports changes in coordinate lift,
origin stratum, and dose, but never replaces the primary neutral or changes the
primary estimate.

The discovery intervention layer is the layer on this consensus ridge with the
best worst-lens rank, with ties resolved toward the earlier layer. The resulting
direction and physical layer pair is frozen before held-out evaluation. On a
held-out prefix, the same passive criteria are reevaluated around that exact
layer, and the frozen cell must remain part of a qualifying ridge; the layer is
never relocated. A token seen by J and R only at disjoint layers is an
instrument-timing observation, not a primary causal candidate. Every eligible
discovery direction is retained, or a deterministic precommitted sample is
taken; interesting continuations never determine inclusion.

J- and R-derived vectors are separate interventions at the same physical cell.
There is no assumed lens-agnostic average. Concordant construal movement is
evidence robust across these two specified lens constructions; it is not proof
of one lens-independent direction or reading. Disagreement remains a
lens-dependent result. Direction cosine is recorded when available but does not
replace the causal comparison.

Instrument agreement is reported in four exhaustive strata:

- **J/R consensus:** enters the primary causal estimand.
- **J-only:** retained as a secondary instrument-disagreement result.
- **R-only:** retained as a secondary instrument-disagreement result.
- **Neither:** receives no modulation arm.

Low consensus prevalence is a result, not permission to promote a single-lens
stratum after the fact. The final freeze names a minimum consensus-eligible
count below which causal estimates remain descriptive; failure never activates
an alternate primary gate.

Independently, each lens assigns an admitted coordinate to an origin stratum:

- **Same-sign increment:** `a_neutral > tau_zero`; target and neutral are
  positively aligned, and contrastive return does not approach or cross zero.
- **Evoked from zero:** `abs(a_neutral) <= tau_zero`; the target evidence
  introduces a positive coordinate from the preregistered near-zero band.
- **Polarity reversal:** `a_neutral < -tau_zero`; the target evidence reverses a
  negatively aligned coordinate into a positive one.
- **Insufficient normalized lift:** the direct coordinate gate fails and no
  modulation arm is admitted.

The primary estimand requires J/R consensus and same-sign classification under
both lenses. Evoked-from-zero and polarity-reversal cases are reported
separately and never pooled into it. If J and R assign different origin strata,
the pair is retained as secondary origin disagreement rather than forced into a
shared class.

## Dose Rule

For each lens, let `h_target` and `h_neutral` be residuals at the same physical
layer and semantic cell under the target prefix and its frozen neutral
counterpart. Let `u` be that lens's unit intervention direction:

```text
p_target  = dot(h_target, u)
p_neutral = dot(h_neutral, u)
delta_p   = p_target - p_neutral
```

For the primary same-sign stratum, `p_target > p_neutral > 0`, so `delta_p` is a
positive same-sign increment. Primary contrastive attenuation is:

```text
h_kappa = h_target - kappa * delta_p * u
dot(h_kappa, u) = (1 - kappa) * p_target + kappa * p_neutral
```

The primary ladder uses `kappa = 0.25, 0.50, 0.75, 1.0`. These levels remove
25%, 50%, 75%, and 100% of the target-minus-neutral projection increment. At
`kappa = 1.0`, only this scalar projection returns to its neutral-prefix value;
the complete residual remains a hybrid target state and is not claimed to be
the naturally occurring neutral residual.

The existing `projection_ablate` operator realizes this update with a
prefix-, lens-, direction-, and cell-specific authored coefficient:

```text
lambda_kappa = kappa * delta_p / p_target
```

J and R therefore receive their own `delta_p` and generally their own
`lambda_kappa`. They are matched in the fraction of their lens-specific evoked
increment removed, not in absolute update norm or raw operator coefficient.
Duplicate `kappa = 0` arms bracket execution.

Evoked-from-zero and polarity-reversal strata use the same non-sign-crossing
ladder to zero rather than attempting a same-sign neutral return:

```text
h_kappa = h_target - kappa * p_target * u
dot(h_kappa, u) = (1 - kappa) * p_target
lambda_kappa = kappa
```

For evoked-from-zero, `kappa = 1.0` returns to the preregistered near-zero band.
For polarity reversal, it stops at zero and deliberately does not restore the
negative neutral projection. A separately labeled secondary sign-restoration
operation may continue from zero toward `p_neutral`; it crosses the axis and is
never pooled with attenuation.

Full one-direction projection ablation remains a separate endpoint:

```text
h_full = h_target - dot(h_target, u) * u
lambda = 1.0
```

It returns the projection to zero and coincides with `kappa = 1.0` only when
the ladder targets zero, including exact `p_neutral = 0` and the
polarity-reversal stop. Because vocabulary-derived directions are
nonorthogonal, neither operation removes one unique concept or guarantees
unchanged scores on other token directions.

Raw projection is the primary dose coordinate because the intervention realizes
it exactly. The dimensionless coordinates `p_target / norm(h_target)` and
`p_neutral / norm(h_neutral)`, absolute update norm, and update-to-residual-norm
ratio are reported as robustness measurements. Interpolating raw projection
does not imply interpolation of those normalized coordinates.

Amplification remains secondary and is called **evocation-conditioned
amplification**. Relative projection dilation may use `lambda < 0`, bounded by a
robust natural-coordinate percentile measured on a held-out calibration corpus
at the same model, lens, physical layer, semantic boundary, and direction. Its
calibration set and eligibility rule are fixed without using intervention
outcomes. Staying inside one naturally observed coordinate envelope does not
establish that the complete perturbed residual is a naturally occurring state.

## Manipulation And Recovery Checks

Output change is never used to choose or widen the primary locus. Discovery
runs verify the manipulation rather than gate on a desirable continuation. For
every contrastive attenuation arm, readouts distinguish:

1. whether the immediate projection moved by the intended
   stratum-specific decrement (`kappa * delta_p` for same-sign return or
   `kappa * p_target` for a ladder to zero);
2. whether attenuation persisted into later layers and prefill boundaries;
3. where and how much of the removed coordinate was reconstructed;
4. whether persistent attenuation changed the final construal.

These are four outcomes, of which the first three are null outcomes: failed
manipulation, successful but rapidly reconstructed coordinate, persistent but
behaviorally redundant attenuation, and persistent attenuation that changes
construal. Full ablation separately targets zero; amplification checks its
declared dilation target; isotropic controls check achieved update norm; and
duplicate zeros check that no update occurred. A blank behavioral result does
not trigger an adaptive assistant-window write. User-span,
final-user-boundary, and final-prefill loci remain independently preregistered
secondary conditions.

## Primary Estimand

The inferential unit is the held-out prefix, not a token, layer, direction,
dose, lens cell, or generated token. The four contrastive attenuation levels
are repeated measurements within a prefix, not additional replicates. Full
projection ablation and any wider grid describe necessity or atlas shape.

For each lens separately, every admitted stance arm is paired one-to-one with
its matched controls at the same contrastive ladder. Only directions classified
as same-sign under both J and R enter the primary calculation. At each level,
the binary outcome is a coherent construal change along a permitted
underdetermined dimension with no invariant leakage, task failure, or
degradation. The primary prefix-level quantity is **clean transition density**:
the fraction of the four contrastive levels producing that outcome, first
averaged across admitted directions within the prefix. Prefixes then receive
equal weight. J and R estimates are reported separately; the primary cross-lens
claim requires concordant effect direction rather than pooling their cells.
Evoked-from-zero, polarity-reversal, origin-disagreement, direction-family,
exact-boundary, full-ablation, ridge-length, and individual-example results
remain secondary.

The two duplicate `kappa = 0` arms define the baseline construal and must have
identical generated token IDs. Token disagreement invalidates and reports the
complete prefix-direction-lens block as an execution-integrity failure; neither
zero is selected opportunistically. A blind-code disagreement on byte-identical
output is instead a coding-reliability failure resolved under a frozen blind
adjudication rule. For each eligible prefix and lens, let `D_stance`,
`D_content`, and `D_isotropic` be the corresponding within-prefix clean
transition densities after averaging matched admitted directions. The two
prespecified paired contrasts are
`D_stance - D_content` and `D_stance - D_isotropic`. The primary selectivity
claim requires concordant positive effects for both contrasts under J and R;
the controls are not silently pooled into a more favorable comparator.

The paired primary effect is conditional on held-out prefixes having at least
one J/R-consensus direction classified as same-sign under both lenses. A
separate unconditional program yield assigns no successful modulation to
ineligible prefixes and is reported beside evocation prevalence; it is not
presented as a treatment effect on un-intervened cases.
The named secondary `any clean transition` indicator is also reported per
prefix, always beside that prefix's admitted-direction count so unequal numbers
of opportunities cannot masquerade as greater elasticity.

## Prefill Loci

Passive readouts cover the complete user message, structural boundaries, and
the assistant-entry window from the renderer-authored assistant-start marker
through the final token processed before first-token sampling. Causal arms use
one exact locus:

- user-message evidence boundaries test interpretation formation and update;
- final relevant user-content boundaries test integration;
- the frozen assistant role/marker locus tests post-input arbitration and
  commitment;
- final prefill tests immediate output selection.

Assistant-start is primary. User-span loci are the first secondary condition,
and final-prefill is a named secondary condition rather than a second write in
the same arm. All are prefill interventions with no decode writes. They perturb
the pre-generation state and then allow unforced downstream computation and
decoding to realize the continuation.

## Controls And Outcomes

The controls answer distinct questions:

- Active stance attenuation tests causal gating by an expressed stance.
- Matched active-content attenuation in the primary comparison must satisfy its
  own cross-lens ridge, positive rank-lift, normalized minimum-lift, and
  same-sign origin gate. It removes the same `kappa` fraction of that
  direction's own target-minus-neutral increment and tests operator/operand
  dissociation.
- Isotropic perturbation matches the stance arm's absolute update norm and tests
  nonspecific residual disruption and degradation; a random axis is not assigned
  a fictitious evoked increment.
- Duplicate zeros test deterministic execution and artifact custody.
- Active versus inactive stance amplification tests whether the evocation gate
  matters at matched absolute update norm; the inactive arm is explicitly
  foreign injection, not an attenuation sham.

Directions are matched as closely as possible on J/R rank, ridge persistence,
physical layer, and realized dose. Degraded outputs are counted but excluded
from coherent reachable support.

Outputs are coded blind to arm for construal, policy, register, propositional
content, attribution, coherence, task correctness, and every preregistered
invariant. The primary safety report keeps two quantities separate:

1. coherent construal change along an underdetermined dimension;
2. invariant leakage or task-correctness loss.

Before intervention outputs are viewed, each prompt family receives a frozen
codebook containing two to five candidate construals, plus `novel_coherent`,
`mixed_or_explicitly_plural`, and `degraded`. Each continuation receives one
primary assignment according to the construal governing its attribution,
conclusion, or action. Novel coherent responses may be described after blind
coding but do not retroactively alter the primary codebook.

Blinding is mechanical. A script strips arm, lens, direction, dose, and control
labels; shuffles records with a recorded seed; and emits a hashed packet
containing only the prompt context and continuation required for coding. The
completed coding file is frozen before a second script restores experimental
labels. Coder identity and revision are recorded. Any model-assisted first pass
pins the rater model, prompt, and decoding settings and is reported as a named
rater rather than ground truth.

Ordinary sampling uses the same clustering and quality rules. Results report
absolute coverage, overlap, novel coherent construals, occupancy, and coverage
gain per run. A construal absent from a finite sample is reported as "not
observed in N samples," not as unreachable by sampling.

## Planned Reports Under Any Outcome

The passive report includes evocation prevalence, J/R instrument-agreement and
coordinate-origin strata, ridge locations and lengths, stance-family
composition, direct target-minus-neutral projection increments, and the frozen
alternate-neutral sensitivity result. The manipulation report shows immediate
achieved attenuation and downstream reconstruction. These are planned
descriptive outputs, not replacements invented after a behavioral null.

The primary causal figure is a prefix-direction by `kappa` heatmap, faceted by
lens and control family, colored by blind construal code with separate leakage
and degradation marks. Its executive sentence reports exact denominators:
same-sign J/R-consensus prefixes out of all held-out prefixes, clean construal
changes under stance and controls, and invariant-leakage count.

## Discovery And Transfer

Discovery prefixes are selected because passive observations made before this
study showed plural readings. They may nominate the lexicon, semantic loci, and
layer rule but provide no confirmatory effect estimate.

Directions and physical loci are frozen before held-out intervention. Every
held-out context remains in the denominator, including those that fail the
evocation gate. Reports distinguish:

- **evocation prevalence:** eligible contexts divided by all held-out contexts;
- **conditional efficacy:** the equal-prefix mean clean transition density among
  contexts with at least one J/R-consensus direction classified as same-sign
  under both lenses, with no direction, dose, lens, or generated token counted
  as an independent observation;
- **invariant leakage:** fixed-proposition violations divided by intervened
  contexts.

Silently relocating a direction to its preferred layer in each held-out prompt
is adaptive rediscovery, not transfer. A separately declared adaptive atlas may
do so, but it answers a different question.

## Prompt Count

The discovery corpus contains **32 passively traced prompt items**. Existing
passive probes may seed that corpus, but discovery interventions contribute no
confirmatory effect estimate and never choose the primary locus based on output.

After all 32 discovery items are frozen, the calibration candidate pool is
defined as every newly authored discovery item with no known prior intervention
result. The four pool members with the lexicographically lowest BLAKE3 digests
of their canonical JSON prompt records, with item ID breaking any tie, form the
intervention-calibration subset. The complete pool and selection are recorded.
Calibration may verify token identity, coefficient algebra, achieved
attenuation, downstream reconstruction, duplicate zeros, and coding
consistency. Behavioral success or failure cannot alter the primary locus,
direction rule, dose ladder, or admission rule. Previously known positive
probes remain separately labeled engineering controls and cannot enter the
candidate pool.

The core confirmatory sample contains **32 held-out prompt items**, frozen before
any discovery intervention or candidate admission. The same 32 user-level items
are rendered through each model's native template for Qwen and the separately
reported Muse replication; renderer-specific prefixes are not claimed to be
token-identical across models. Each item may have one matched neutral admission
counterpart, but those counterparts and all repeated lens, direction, dose, and
control cells do not increase the inferential `N`.

All 32 held-out items remain in the evocation denominator, with no
outcome-dependent replacement or early stopping. This is a bounded
proof-of-concept sample, not a population estimate over prompts. The minimum
consensus-eligible count required for confirmatory causal language is frozen
before intervention outputs are viewed; falling below it yields a feasibility
and descriptive result rather than a single-lens fallback.

Preregistration is a firewall between observations used to select a procedure
and observations used to estimate its effect, not a vow never to learn. Any
implementation defect found during calibration is recorded with a protocol
revision, corrected, and rerun across the complete four-item calibration subset
before the held-out freeze is released.

## Secondary Program

Secondary conditions remain outside the primary result unless independently
preregistered:

- **System context:** an outer factor testing the same user input under different
  harness priors. Admission loss indicates representational narrowing; stable
  admission with collapsed output branching indicates policy suppression.
- **Constraint geometry:** semantic specificity, authority or loudness, conflict,
  and token length are varied separately. Relevant ambiguity reduction, not
  length itself, is predicted to shrink useful elasticity.
- **Punctuation:** learned syntactic and pragmatic micro-boundaries may act as
  consolidation cues, concentrating incremental appraisal before assistant
  entry. This is not a claim that punctuation creates otherwise absent compute.
- **Chords:** joint modulation maps interaction geometry only after independent
  directions are established.
- **Commitment:** base-versus-instruct and post-training comparisons can test
  whether sharper arbitration collapses reachable plurality or converts it into
  policy suppression.
- **Reasoning mode:** thinking and no-thinking are distinct renderer and harness
  conditions. The primary mode remains fixed; mode comparisons never enter the
  same-prefix primary estimate.

## Claims Boundary And Freeze

The atlas measures transported-logit lens readouts and their causal directions,
not formal sparse J-space, unique concepts, subjective states, or fully formed
hidden answers. Ineffective modulation does not establish absence: lens error,
layer timing, downstream repair, and nonorthogonality remain alternatives.

Before the first confirmatory run, freeze and hash model and lens identities,
exact rendered token IDs, discovery and held-out prompts and their neutral
counterparts, alternate-neutral sensitivity subset, calibration subset, lexicon
and token IDs, normalized `tau_zero` and `delta_a_min`, instrument and origin
strata, admission and layer rules, loci, the contrastive `kappa` ladder,
full-ablation endpoint, controls, reasoning mode, sampling budget, minimum
consensus-eligible count, fixed propositions, permitted ambiguity dimensions,
construal codebooks, blinding seed and scripts, blind adjudication rule, and
degradation criteria.

The resulting object is compact: the same prefix, the model's own evidenced
readings, a rule that keeps the experimenter's desired answer out of the state,
and a calibrated account of how much of the response the user never specified.
