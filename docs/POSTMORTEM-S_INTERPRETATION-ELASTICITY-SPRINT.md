**Executive Judgment**

The work produced a technically credible negative pilot, not a failed execution. The interventions ran, the intended coordinates moved, zero controls reproduced exactly, and the investigators reported the null rather than manufacturing a positive result.

What failed was the attempted scientific instantiation:

> Returning selected one-token target-minus-twin coordinates toward their twin values was not sufficient to change the governing greedy interpretation in the tested cells.

That narrow result is real. It does not falsify the broader interpretation-elasticity vision because the campaign did not instantiate several conditions that define the vision’s primary hypothesis:

- It did not establish that the selected prefixes actually occupied multiple nearby behavioral basins.
- It did not validate that the selected token directions were causal stance variables rather than correlates, topical features, or downstream response-plan readouts.
- It often did not preserve the intervention through the relevant decision point.
- It used greedy free-text generation as the sole primary behavioral endpoint.
- It omitted held-out transfer, natural sampling, isotropic controls, complete matched content controls, normalized admission thresholds, and blind outcome coding.

The cleanest postmortem is therefore:

> The campaign successfully manipulated measured coordinates, but it had not established the causal chain connecting those coordinates to competing interpretations and observable behavior.

The broad vision remains plausible, but this particular one-dimensional, one/few-position, greedy-output assay did not provide a fair or powerful test of it.

**What Happened**

1. **The team first built a serious intervention instrument.**  
   The preceding work established exact rendering, lens fidelity, position-specific writes, trace inspection, and reproducible intervention bundles. It also discovered important empirical facts: semantic shams can alter outputs, readouts can reconstruct after ablation, and effective layer/dose windows can be narrow and non-monotonic. The earlier `error` result, for example, only opened a selective window around layer 46 after failures at other layers and doses. See `../qwen-llm-lens-recovery/docs/LENS-OBSERVATIONS.md:463`.

2. **The vision then articulated a substantially more rigorous scientific program.**  
   It explicitly distinguishes token directions from unique concepts, construal from downstream response policy, local reachable sets from global control, and stance attenuation from output programming. It proposes frozen lexica, normalized admission gates, discovery-to-held-out transfer, content and isotropic controls, natural sampling, and prefix-level inference. See `../qwen-llm-lens-recovery/docs/INTERPRETATION-ELASTICITY-VISION.md:48`, `../qwen-llm-lens-recovery/docs/INTERPRETATION-ELASTICITY-VISION.md:143`, and `../qwen-llm-lens-recovery/docs/INTERPRETATION-ELASTICITY-VISION.md:481`.

3. **The executable brief deliberately relaxed that design into an exploratory sprint.**  
   It allowed adaptive prompt development, per-cell token/layer/site nomination, lexicon expansion from passive traces, and intervention-output-informed follow-ups. It explicitly disclaimed confirmatory status. That is legitimate exploration, but it sharply lowers the evidentiary ceiling. See `../qwen-llm-lens-recovery/docs/INTERPRETATION-ELASTICITY.md:15` and `../qwen-llm-lens-recovery/docs/INTERPRETATION-ELASTICITY.md:129`.

4. **Candidate directions were selected from passive traces.**  
   The analysis found same-token J/R candidates and then manually nominated directions and loci from the available output. The actual gate was largely “positive target coordinate and target greater than twin,” not the vision’s frozen lexicon, normalized thresholds, origin strata, or held-out transfer procedure. The score called `p_target` is actually a selected-row projection numerator, not a probability or globally comparable coordinate. See `../qwen-llm-lens-recovery/experiments/lens/interpretation-elasticity/analysis/analyze_primary_coordinates.py:103` and `../qwen-llm-lens-recovery/docs/QWEN-LENS-FIDELITY-AUDIT.md:322`.

5. **The final initial campaign comprised six Qwen prefixes:** P2, P4, P5, P6, P7, and P9.  
   P8 and P10 failed the measured coordinate-sign gate. P1 disappeared from the final campaign without a comparably explicit exclusion record. P3 became a separate, incomplete layer sweep. P11 was added later as a safety-boundary follow-up.

6. **The final producer was clean and the campaign was mechanically sound.**  
   The accepted overnight campaign used clean producer commit `ec1944ec`, including matched Muse J/R provenance fixes. It yielded 203 accepted normal runs. All intended writes landed, direct attenuation was approximately linear, and all 29 duplicate-zero groups were exact. See `../qwen-llm-lens-recovery/experiments/lens/interpretation-elasticity/OVERNIGHT-2026-09-04.md:139` and `../qwen-llm-lens-recovery/experiments/lens/interpretation-elasticity/artifacts/interventions/overnight-2026-09-04/audit.json:4`.

7. **The behavioral result was uniformly negative under the clean-transition criterion.**  
   The initial Qwen no-thinking result was 0/6 clean changes under J and 0/6 under R. Full ablations sometimes changed length or wording, but not the governing construal. Follow-ups on distributed P7, Muse P7, thinking-mode P5, and P11 did not produce a clean interpretation transition.

Importantly, the inferential sample is six prefixes, not 203 independent trials. Lens, dose, and control arms are repeated measurements on those prefixes. Counting all arms as independent evidence would be pseudoreplication.

**Diagnostic Breakdown**

| Evidence class | Observed result | Best interpretation |
|---|---|---|
| P2, P4, P9 | Direct attenuation landed, but only roughly 0–4% of the effect remained near assistant generation | Primarily reconstruction/manipulation failure, not strong behavioral robustness evidence |
| P5, P6, P7 | Roughly 45–65% of the effect survived to the final prefill state; natural-range doses did not change the governing answer | The chosen coordinate was insufficient, redundant, or the model remained deep within the same basin |
| Distributed P7 | Five selected sites were moved to their twin values; greedy output remained unchanged at natural doses | Strongest Qwen evidence against sufficiency of the exact `feelings` axis |
| Muse P7 | All 28 outputs were token-identical, including full target-direction ablation | Strong evidence that the exact Muse `empathy`/`burnout` coordinates were not governing this response |
| Thinking P5 | The intervention mostly reconstructed; after extending to 8192 tokens all 14 outputs were exact | The apparent earlier difference was a truncation artifact, correctly withdrawn |
| P11 | The safety boundary held; R at the strongest conditions switched to a different high-level refusal | Useful robustness result, but not an ordinary interpretation-elasticity test |
| P3 | Only 17 of 72 planned cohorts completed; content, Muse, random controls, and several depths were absent; thinking runs truncated | Engineering exploration only, unsuitable for campaign-level inference |

This decomposition matters. A null after the intervention disappears is categorically different from a null after a persistent, behaviorally relevant state change. The vision itself anticipated this distinction in its manipulation/recovery ladder. See `../qwen-llm-lens-recovery/docs/INTERPRETATION-ELASTICITY-VISION.md:332`.

**What Went Well**

- **The conceptual vision is unusually careful.** It explicitly warns that a lens token direction is not a unique hidden concept and that response policy can mask a changed construal. Those qualifications are exactly right.
- **The final runtime was auditable.** Plans, hashes, source revisions, position targeting, duplicate-zero arms, state traces, and inspectable sweep bundles make a silent no-op or accidental prompt mismatch unlikely.
- **Manipulation checks were taken seriously.** The report separates immediate attenuation, downstream survival, and behavior. That prevents “the write executed” from being mistaken for “the hypothesized mediator changed.”
- **The null was reported honestly.** Truncated P7 and P5 runs were identified, rerun with larger budgets, and not counted as positives. Wording and length differences under full ablation were not mislabeled as construal changes.
- **J and R were kept separate.** Their agreement was used as corroboration rather than silently pooling unlike instruments.
- **Safety invariants held.** The P11 result was correctly characterized as boundary robustness rather than a desired unsafe flip.
- **Objective failures were allowed to fail.** P8 and P10 were excluded when the target coordinate did not exceed the twin rather than forcing them through the intervention funnel.

These are meaningful successes. They make the negative result more trustworthy than a superficially positive but weakly controlled campaign would have been.

**What Went Wrong**

1. **The runnable study was much weaker than the vision it was meant to instantiate.**

   The vision’s primary study called for a frozen corpus, 32 discovery and 32 held-out prefixes, fixed transfer rules, a global stance taxonomy, normalized gates, alternate neutrals, matched controls, and sampled occupancy estimates. The scratch design still listed these as prerequisites immediately before the first real run. See `../qwen-llm-lens-recovery/docs/AGENT-SCRATCH-INTERPRETATION-ELASTICITY-PROBES.md:982`.

   The overnight campaign instead used six adaptively developed prefixes with per-cell manually nominated tokens, layers, and sites. That is useful discovery work, but it cannot support the vision’s population-level or transfer claims.

2. **The experiment gated on lexical readout evidence, not demonstrated behavioral elasticity.**

   The central premise is that a fixed prefix has several coherent nearby continuation basins. Yet the campaign did not first show that the target prefix actually produced distinct governing interpretations under paired natural sampling, nor that its twin reliably selected a different basin.

   The twins therefore functioned as coordinate references, not validated behavioral counterfactuals.

3. **Several target/twin pairs changed more than stance strength.**

   - P2’s twin explicitly inserts “good,” making it a disambiguated sentiment comparator rather than a neutral admission twin.
   - P4 changes a false test expectation into a correct one; correctness dominates any stance axis.
   - P5 changes register around a settled mathematical impossibility.
   - P7 changes affective framing, but both versions communicate readiness to leave a job.
   - P11 changes benign intent and requested scope as well as stance.

   These deltas can encode topic, facts, register, intent, policy, or lexical planning—not merely “more versus less of one stance.”

4. **The selected directions had weak construct validity.**

   Tokens such as `mediocre`, `error`, `impossible`, `skepticism`, `feelings`, `emotional`, `empathy`, and `burnout` are plausible probes, but they are not clean latent variables. A readout can become salient because the model has already decided what kind of response to write. Attenuating it then removes a consequence of the response plan rather than a cause of the interpretation.

   Same-token J/R agreement helps rule out a single-lens accident, but it does not prove causal semantics. Both lenses can recover the same output-head-associated feature.

5. **The tasks were often structurally hostile to a clean positive result.**

   P3 and P4 are fact/correctness cases. P5 concerns a known impossibility. P11 is a policy boundary. A strong “flip” in these cells would often violate the frozen invariants and therefore be rejected by the outcome code.

   P6, P7, and P9 are more genuinely open, but their baselines are long, comprehensive, “cover all sides” advice. Such answers absorb several interpretations into one generic response policy rather than forcing a basin choice. P7’s baseline was around 1,496 tokens and P6 around 1,254. A changed weighting among interpretations can remain invisible when the answer already mentions all of them.

6. **The intervention was extremely low-dimensional.**

   “Full ablation” means full removal of one selected token-direction component at one site—not full removal of the stance or construal. Natural arms merely removed the target-minus-twin increment, often with small ratios. P5’s admitted ratios were only about 3–6%.

   The ratio algebra is not obviously wrong: because numerator and direction norm cancel, it can return the selected projection toward the twin. What was missing was measurement of normalized coordinates and actual update norms. Consequently, dose magnitudes were not comparable across tokens, layers, models, or controls.

7. **Reconstruction was a first-order problem.**

   P2, P4, P9, P3, and thinking P5 largely regenerated the selected feature before the assistant decision point. Those cells do not strongly test output robustness; they show that a local write was transient.

   P7’s distributed intervention improved this diagnosis, but it still ablated one direction at one layer across positions. A construal can be encoded across dimensions, layers, attention-mediated state, and recurrent generation dynamics.

8. **The endpoint was underpowered for local changes.**

   Greedy decoding is discontinuous: meaningful probability changes remain invisible unless they cross an argmax boundary and then alter the response trajectory. No release-policy sampling campaign was executed, despite the vision’s emphasis on occupancy and transition density.

   Conversely, once the early greedy trajectory enters a familiar long-form answer template, later state differences can be washed out. P5’s full-ablation divergence only appeared late and affected wording rather than its conclusion.

9. **Scientific controls were incomplete.**

   Duplicate-zero controls were excellent implementation checks, but they are not causal specificity controls. The campaign lacked:

   - Runtime isotropic norm-matched controls.
   - Complete admissible content controls.
   - Demonstrably update-norm-matched content controls.
   - Wrong-layer, shuffled-position, and inactive-token controls.
   - A positive control showing that a richer state intervention could actually change the chosen endpoint.

   Several content candidates were inadmissible, and the admissible content-coordinate ratios were often much smaller. The all-null content result therefore does not establish stance-specific robustness.

10. **Outcome assessment was not independently auditable.**

    The artifact audit verifies expected counts, stop reasons, and duplicate-zero equality, but not semantic coding, freeze order, placement correctness, or blind agreement. No frozen codebook or independent coding artifact was present. Most outputs were plainly identical, so this does not undermine the central null, but it limits interpretation of subtle wording and policy changes.

11. **Research custody could be better.**

    The recovery worktree is on `repair/lens-rendering` at `316bbad1`, while the final clean producer came from the separate integration worktree at `ec1944ec`. The vision, brief, scratch notes, and experiment tree are currently untracked. The vision/brief/BATCH-01 material briefly existed in commit `376a47a4` and was then reset out of branch history.

    This is not evidence that the result is wrong—the final bundles bind producer and plan hashes—but it makes the study freeze, provenance, and reconstruction unnecessarily fragile. P3 also used a dirty producer and remains incomplete.

**Why This Instantiation Most Likely Failed**

The experiment depended on the following causal chain:

```text
plural nearby construals
    -> a token direction validly measures one construal
    -> changing that direction changes the latent construal
    -> the change survives to the decision point
    -> response policy preserves the difference
    -> greedy free text exposes it
```

Only one link was consistently demonstrated: the immediate coordinate could be changed as intended.

My ranked explanation is:

1. **High confidence: the selected prompts were not shown to be near competing behavioral basins.**  
   The dominant open-ended outputs were broad, policy-stable answers. Returning one scalar coordinate to its twin value likely stayed inside the same basin.

2. **High confidence: the selected token coordinates were not established as causal stance variables.**  
   Persistent and even full single-axis removals often left outputs exact. The simplest interpretation is that these coordinates were correlates, redundant readouts, or too narrow a slice of the representation.

3. **High confidence for several cells: the model reconstructed the feature.**  
   P2, P4, P9, P3, and thinking P5 directly demonstrate this. Their output nulls should not be treated as evidence that a persistent interpretation change would be behaviorally inert.

4. **Medium-high confidence: the intervention lacked spatial and temporal coverage.**  
   Prior work already showed narrow layer/dose windows and effects from all-prefill-plus-decode writes. The elasticity campaign used a much more restrained one/few-prefill-site attenuation. That is scientifically cleaner, but much harder to make causally sufficient.

5. **Medium-high confidence: greedy generation concealed sub-threshold effects.**  
   No output difference does not imply no probability redistribution or no latent construal movement. The study measured basin crossing, not sensitivity within a basin.

6. **Medium confidence: instruction tuning collapsed latent plurality into a stable response policy.**  
   Qwen’s long “balanced helpful answer” behavior and P11’s robust refusal suggest that several internal readings may feed the same downstream policy. The vision itself proposes base-versus-instruct and system-context comparisons to distinguish representational rigidity from policy suppression.

7. **Low-to-medium confidence: quantized transfer or execution topology reduced validity.**  
   Qwen’s prior passive metrology makes Q8 transfer an unlikely primary explanation, and direct manipulation checks rule out a no-op. Nevertheless, the lenses were fitted in BF16 and applied to GGUF Q8; Muse lacks equally direct transfer validation. `prefill-execution auto` may also have packed P4 differently from serial reference execution. These remain caveats rather than leading causes.

8. **Still live: the broader low-dimensional elasticity thesis may simply be false in these regions.**  
   The persistent P7 and Muse nulls genuinely count against the sufficiency of these exact axes. Full removal cannot be dismissed merely as “dose too small.” It is possible that stance is highly distributed, downstream rather than upstream, or that instruction-tuned behavior is genuinely inelastic around these prefixes.

The most defensible claim is therefore:

> The campaign falsified the sufficiency of these selected single-token coordinates, doses, loci, and greedy endpoints. It did not establish the absence of latent plurality or the impossibility of a richer local counterfactual atlas.

**Expectation Mismatch**

Some optimism appears to have transferred from earlier successful steering experiments, but those experiments asked a different question.

The earlier decision and `error` interventions used stronger additive writes, often across all prefill positions and decode, and effective settings were discovered by layer/dose search. The `error` case required approximately `+0.45` at a narrow layer-46 window; stronger settings reverted or became nonspecific.

The elasticity pilot instead asked whether a small, endogenous, target-to-twin attenuation at one or a few prefill cells could alter behavior without injecting a desired state. That is a much stronger and more interesting claim. Prior broad steering established that the runtime could alter behavior; it did not establish that restrained local attenuation would cross a natural interpretation boundary.

**What Could Have Gone Better**

A stronger follow-up should use a diagnostic ladder rather than immediately scaling the same assay:

1. **Demonstrate behavioral eligibility first.**  
   In discovery items, paired natural sampling should show that the fixed target prefix supports at least two coherent, invariant-preserving outcomes and that explicit A/B comparators select those outcomes. A prefix with one generic answer policy should not enter an elasticity trial.

2. **Separate three different comparators.**  
   Use an admission-neutral twin for estimating natural stance increment, explicit disambiguated A/B prefixes for construct validation, and an unchanged duplicate for implementation checks. They should not be treated as interchangeable.

3. **Add a positive causal-locus control.**  
   Patch the full residual state—or a richer residual difference—from an explicit comparator into the ambiguous prefix. If that does not change a bounded endpoint, the locus or task is unsuitable. If full-state patching works but the token direction does not, the failure is construct geometry rather than behavior.

4. **Map in discovery, then freeze transfer.**  
   Discover layer, site, dose, and possibly a small multivariate stance subspace on one sibling; transfer the exact rule to a held-out sibling. The earlier layer-46 result shows why unrestricted tuning and prospective validation must be separated.

5. **Use complementary endpoints.**  
   Retain open-ended text for leakage and ecological validity, but add a bounded action choice, a short recommendation, first-decision-token probabilities, and paired common-seed sampling. Greedy transition remains a stringent endpoint, not the only endpoint.

6. **Complete the causal controls before interpretation.**  
   Runtime support should include arbitrary isotropic vectors, applied-update norm reporting, norm-matched content controls, inactive stance directions, wrong layers/sites, and random controls.

7. **Treat the prefix or dependence cluster as the sample unit.**  
   Report eligible-prefix yield and conditional efficacy separately. J/R, doses, and repeated runs are measurements, not independent sample size.

8. **Test policy suppression directly.**  
   Repeat only qualified cells across base versus instruction-tuned models, no-thinking versus thinking, and minimal versus policy-heavy system contexts. This separates latent representational elasticity from downstream policy convergence.

9. **Freeze and preserve the study record.**  
   Commit the protocol, candidate registry, exclusions, codebook, producer revision, and manifests before behavior runs. Keep exploratory follow-ups in a separate analysis stratum.

A useful go/no-go sequence would be:

```text
Do target and comparators occupy distinct behavioral basins?
Does a full-state patch move the endpoint?
Does the proposed direction mediate that movement?
Does the write persist to the decision locus?
Does it beat matched controls?
Does the frozen rule transfer to held-out prefixes?
```

Stopping at the first failed rung would make each null diagnostically informative.

**Final Assessment**

The strongest success of the project is epistemic: it obtained a null under a functioning intervention stack and did not overclaim it. The strongest failure is construct-to-behavior validation: the study assumed that passively salient token directions represented causal, nearby interpretation coordinates before showing either causal mediation or nearby behavioral plurality.

The likely lesson is not “interpretation elasticity does not exist.” It is:

> A local lens coordinate can be measurable, reproducible, and directly manipulable while still being causally irrelevant to the model’s governing response policy.

That is itself an important result—and exactly why the broader vision’s stricter discovery, control, persistence, sampling, and held-out-transfer machinery is necessary. No files were modified during the review.
