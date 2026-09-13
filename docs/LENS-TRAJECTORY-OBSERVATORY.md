# Trajectory Observatory -- Living Tracker

## Focus

Observe how often different consequential responses arise from the exact same
prompt, and how layerwise readouts evolve throughout those trajectories. Let
empirical observations guide interpretation and later experiments. No steering,
manufactured contrast prompts, predefined stance axes, or basin claims here.

This is the working tracker, not another comprehensive research proposal. Keep
the current state, evidence, uncertainties and next observation visible.

## Current Collection

| Item | Decision / status |
| --- | --- |
| Source | Eight unchanged inputs from the existing 24-case private archive survey |
| First model | Qwen3.6-27B Q8, explicit native thinking mode |
| Follow-up model | Qwen3.8-27B Q8 xhigh; not yet collected in this series |
| First sampling block | Four distinct seeds per prompt, identical rendered prefix each time |
| Sampling | Temperature 1.0, top-k 20, top-p 0.95, min-p 0 |
| Penalties | None active; matches recommended presence 0 and repetition 1 for thinking |
| Output budget | 8,192 generated tokens; retain and mark censored outcomes, never resample to hide them |
| Observers | Matched R and J, all available source layers 0-62, every consumed position |
| Stored detail | Top eight plus full-vocabulary entropy, normalization, retained mass and logit moments on every replayed cell |
| Capture meaning | Exact-token replay after generation, not a claim of live original-state capture |
| Execution | Generation and replay settings, source identity, model/lens and executable identities recorded |
| Current status | 24/32 planned samples retained: three per prompt, all native EOS; eight seed-71 slots remain |
| Observer coverage | Eight sampled trajectories across four prompts, R/J complete: 16 logical traces, 1,866,564 cells |

The eight cases are selected for breadth, not predicted behavioral variability:

- `conceptual-01`: mathematical explanation, Euler's number.
- `conceptual-07`: linear-algebra learning, preserved opening U/A/U history.
- `conceptual-06`: history of computing conventions.
- `conceptual-02`: philosophical/playful evaluation.
- `practical-06`: film-reception discussion.
- `practical-05`: constrained vacation planning.
- `practical-07`: everyday cooking clarification, preserved opening U/A/U history.
- `conversation-07`: ordinary friendly greeting, standalone recontextualization.

The other six are standalone archive excerpts, not complete recovered histories.
Saved ancestry notes are provenance, not a fresh database verification. Retain
historical assistant text without treating it as factual authority. Private
source files and exact hashes belong in the collection manifest, not this page.

## Sampling Fidelity

Primary recommendations: [3.6 pinned card](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/README.md)
and [3.8 pinned card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md).
Both recommend the parameter values above for general thinking tasks. This is
not the old archive's temperature0.8/top-k40 configuration; do not pool them.

Native sampler-v1 order is top-k -> min-p -> temperature -> top-p -> draw,
using xoshiro256++/SplitMix64. Matching values is not bitwise provider replication:
filter ordering, RNG, precision and kernels can differ. Frequency penalty is not
specified in these cards. The 8,192-token budget is below the 3.6 card's ordinary
32,768 recommendation; it is a collection resource bound, not an equivalent
length configuration. Do not stop at the end-thinking marker.

## Coverage And Interpretation Rules

- Prompt identity excludes seed: require identical rendered input IDs within
  each prompt/model/mode group. Retry attempts are not extra samples.
- Count all outcomes, including duplicates, failures and capped responses.
  Report completion and observed response groups separately.
- Replay includes the prompt and generated tokens that were actually forwarded;
  the final sampled token was not forwarded. Any additional forward of that
  token must be labeled counterfactual, not original execution.
- R/J have no source-layer63 matrix in these fits. Their target62 identity is
  not the model's actual final block63. Do not silently call 63 fitted layers
  complete coverage of all native blocks.
- Verify every requested layer/position cell. If artifact limits require chunks,
  preserve complete coverage and mark their joins; never silently skip layers,
  tokens or difficult trajectories. The added topics use three disjoint layer
  shards per logical trace, each replaying the complete consumed-token sequence.
- Top-k ranks alone do not establish concentration. Distinguish raw logits,
  full-distribution summaries, transported geometry and actual sampling scores.
- Describe response differences from their text before naming latent states.
  Allow compatible, mixed and unresolved interpretations. LLM coding is an
  instrument with disagreements, not human ground truth.
- Layers, tokens and seed-paired conditions are not independent prompts.
  Four samples are an opening observation block, not a frequency estimate with
  useful precision or proof that unseen alternatives are impossible.

## Questions To Watch

1. Which prompts produce consequential differences, versus mostly wording changes?
2. Do different beginnings converge, or similar beginnings separate later?
3. Where do between-sample readout differences widen, narrow or change direction?
4. What recurs across topics after accounting for token identity, formatting,
   phase, length and common depth trends?
5. Which apparent patterns depend on the observer or dimensionality-reduction
   choice rather than appearing reliably in the underlying passages?

## Progress And Evidence

- Isolated worktree: `/Users/tito/code/qwen-llm-trajectory-observatory`.
- Source revision: `f9d16cf6bc2d5cf3da8d9c57430eadcd9a070f58`.
- Private artifact root: `target/trajectory-observatory/` in that worktree.
- Frozen clean generator SHA256:
  `1a195f19626a10fccca81862891432392fa07b2eefcb04552a36a08eef18103d`.
- Generation: 68,725 tokens, three draws (17/29/43) per prompt, all 24 naturally
  stopped. Exact prompt IDs match within each group; source inspection verifies
  fresh RNG per request. No active writes. Prefill uses the recorded native auto
  policy (qualified packed passive spans where eligible), followed by decode.
- The first cohort interrupted a fifth request after four completed children.
  Its unavailable partial stream and later explicit same-seed retry remain
  recorded; the slot is counted once. Individual request packaging avoids
  discarding in-flight samples at an aggregate cohort time bound.
- Replayed samples: greeting, cooking, aphorism and film, seeds17/29 each. All
  consumed positions and layers0-62 have both observers and full summaries.
  The other 16 retained samples await replay; no missing coverage is concealed.
- First replay block: 570,528 cells, about761 MB, 12m57s. Broader block:
  1,296,036 cells, 1.716 GB, about36m28s successful capture time. No new kernels.
- A real JSON roundtrip edge rejected a correct F64 maximum by one F64 ULP.
  Validation-v2 permits exactly adjacent F64 values, not neighboring F32 values.
  The retained shard passes without regeneration; original failure and mixed
  reader identities remain explicit. Reduction formulas and GPU arithmetic did
  not change. A separate interrupted replay attempt is also retained.
- Verification: 329 CLI and five reducer tests pass; nine opt-in CLI tests are
  ignored and one Metal-initializing test is explicitly skipped. Adversarial
  code review and live flag-on/off top-k parity pass. Independent analyses check
  cell coverage, joins, normalization consistency, phases and retained hashes;
  sparse output cannot independently reconstruct the omitted full-logit tail.

## Observations -- Provisional, Not An Interpretation Atlas

| Observation | What it does and does not support |
| --- | --- |
| Same broad answer approach can hide consequential differences | First two cooking samples differ in default action; hiking samples choose the same island but prescribe different workloads; aphorism samples give materially different verdicts. Classification grain matters. |
| All-layer and endpoint entropy move oppositely | Across all 16 replayed observer traces, answer mean entropy is higher over all layers, lower at62. This is observer-distribution breadth, not certainty or correctness. |
| Consuming the end-thinking marker produces a layer62 entropy pulse, then a drop after the following double newline | Present in all eight sampled trajectories. Ordinary formatting and quoted-word events can produce larger whole-profile changes; do not name a mental transition from this alone. |
| A compact distribution-profile map transfers to two additional topics | Ten PCs fitted only to seed17 greeting/cooking leave about12% reconstruction residual on aphorism/film, versus about11.5% on same-topic seed29. Three PCs leave35.6% R /37.8% J on new topics: the appealing 3D map loses substantial detail. |
| Local proximity is not yet a useful state classifier | Initial nearest-neighbor phase agreement barely beats the greeting majority baseline and falls below it for cooking. Token/phrase associations are substantial; no forecast or recurrent-state identification has been established. |

PCA uses all 63 layers' entropy and top-eight mass (126 standardized features),
separate observers, equal training weight per case trajectory, and frozen
seed17 scaling/loadings. New-topic residuals are normalized against the training
mean, not conventional test-centered R2. Topics were adaptively selected, not a
preregistered generalization test. The map is not native residual geometry;
phases and correlated features are not independently balanced.

Two seed-hidden text reviews agree on several action/verdict differences but
use different coarseness for cooking and cultural explanation. Keep that
disagreement rather than publish a premature construal-frequency statistic.
The third samples are retained, not yet independently coded. Factual assertions
and generated reasoning are observations, not authoritative explanations.

Evidence paths below are relative to the private artifact root:

- `block8-t1-k20-20260912/seed43-extension/index-after-8.json` and
  `packet-after-8.md`: current sample inventory and seed-hidden reading packet.
- `replay-private-20260912/report.json`: initial complete replays.
- `breadth-extension-validation-v2-20260912/final-index.json`, `final-report.json`,
  `final-seal.json`: complete layer-shard joins and both reader identities.
- `independent-cpu-20260912/REPORT.md` and `all-cells.svg`: first analyses and
  full layer/token heatmaps, not a subsampled plot.
- `independent-breadth-cpu-20260912/REPORT.md`: frozen-map transfer, counterexamples
  and independent coverage/normalization checks on all added cells.

## Sidequest: Dynamical Landscapes

The user's [2021 attractor-methods review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8085613/)
suggests useful exploratory tools, not evidence that our readouts contain chaotic
attractors. Candidate analyses on the planned data: short-history/delay features,
recurrence and observed transition maps, and held-out predictive comparisons.
Keep depth distinct from token time, context-conditioned maps distinct from
pooled mixtures, and density surfaces distinct from dynamical potential energy.
Intervention recovery can later test a specific pattern; this sidequest does not
replace the present observation-first collection or authorize a new control project.

## Next

Continue seed71 and the replay backlog, then inspect the four-draw response sets
together before fixing construal categories. Broaden readouts to the longer
learning, mathematical and planning histories rather than staying with the
cheapest traces. Test whether observed profile structure survives token/format
and phase controls before adding more elaborate dynamics models. 3.8 remains
a subsequent matched-input series, not silently pooled into these frequencies.
