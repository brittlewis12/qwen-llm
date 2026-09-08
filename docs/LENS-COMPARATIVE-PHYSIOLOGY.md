# Comparative Physiology: Context, Reader, And Response Boundary

Exploratory observations, 2026-09-08. The new data-only path is useful now:
follow the existing archive histories through both matched J/R pairs rather
than making another importer milestone or constructing a discriminative corpus.
This pass collects live observations and four natural continuations, not a
benchmark, attractor atlas, or measurement of conviction.

## What We Followed

Six existing inputs: bundles/sets (`conceptual-08`), spatial emoji interpretation
(`conversation-02`), shoobie/cartoon association (`conversation-03`), roasting
clarification (`practical-07`), outlet-method negotiation (`practical-08`), and
checkmate etymology (`practical-02`). Five are ancestry-checked three-turn windows;
bundles starts mid-conversation. Checkmate is a standalone recontextualization.
The saved recovery notes establish these boundaries; ancestry was not freshly
reconstructed from the database in this pass. None is a new synthetic stimulus.

First compare Qwen3.6 Q8 thinking and Qwen3.8 Q8 xhigh, using identical source
message files but explicit native renderers. Initial readouts suggest differences
between user-content boundaries and the final assistant/thinking prefix. Inspecting
the renderer exposes an important confound: xhigh inserts an instruction to check
assumptions and consider alternatives. Follow that lead with 3.8 medium, which
keeps thinking enabled without this instruction.

The resulting material:

- Eighteen R traces: six histories, three model/mode conditions, every input
  position and every source layer 0-62, top eight across the vocabulary.
- Boundary-vector supplements at layers 0/16/32/48/62; 180 vectors at the last
  user-content token and final-prefill token. Sparse vector capture does not
  limit the full-depth top-eight trace coverage.
- Eighteen scalar full-vocabulary bundles: bundles and shoobie, all three
  conditions, J/R/plain, every layer 0-62. Each holds 63 x 248,320 F32 scores;
  281,594,880 scores total. No top-eight renormalization in distribution metrics.
- Four greedy 3.8 continuations: both full-score histories in medium/xhigh,
  with identical explicit sampler parameters and zero interventions. All stop
  naturally below 4,096 tokens; generated lengths are 1,647/3,622 for bundles
  and 493/633 for shoobie, including thinking and stop tokens.

All use one frozen executable, SHA256
`12e69ef2c6e6856da2399fc7d47663578f4f18cb46531b6a46549ccbf15df9d1`.
It was built during the reviewed data-contract implementation, not from the
subsequent clean merge; its dirty source identity is recorded in outputs. No
numerical kernels, fitting, precision qualification or model dependencies change.

## 1. The Boundary Is More Informative Than A Headline Label

Removing xhigh changes exactly 237 leading rendered bytes / 42 tokens in all
six 3.8 inputs. It changes instruction content, system framing, length and
positions together, not just a reasoning-effort scalar.

Within 3.8, xhigh/medium R top1 agrees at 351/378 last-user-content layer-cells,
but only 197/378 final-prefill cells. Every history has the same direction.
Mean top-eight intersection changes from 7.44/8 to 4.15/8. These are correlated
layer-cells describing six selected histories, not independent trials.

The suffix localizes this further. Across layers 32-61 and all six histories:

| Position in the shared suffix | Xhigh/medium R top1 agreement |
| --- | ---: |
| User end marker | 152/180 |
| Following newline | 148/180 |
| Assistant start marker | 126/180 |
| Assistant role token | 146/180 |
| Following newline | 124/180 |
| Thinking marker | 60/180 |
| Final newline | 66/180 |

This is not explained by literal newline identity alone: repeated newline
tokens behave differently across positions. It also is not evidence of a
universal transition into a psychological state. The useful question is whether
these sites emphasize preparation for a response differently from processing
the user's material. Prefix length, attention, role framing and ordinary
next-token structure remain alternative explanations.

Within-checkpoint R geometry agrees with the localization: at layer48, mode
cosine is 0.686-0.776 at final prefill versus 0.990-0.999 at last content. At the
identity target62 the mode-sensitive final boundary remains, so fitted transport
alone cannot explain it. Between-case final-prefill similarity is additionally
confounded by a shared consumed newline; it is not an abstract-construal measure.

## 2. Reader Choice And Context Change Matter At Different Depths

For the two full-score histories, consider the 3.8 grid of two modes, two cases,
and J/R. Each table entry averages four one-factor edges per depth, then averages
depths within the stated band. Use temperature-one F64 softmax over complete F32
rows and Jensen-Shannon divergence in nats; its maximum is ln(2), about 0.6931.

| Layers | Mode change JS | J/R change JS | Case change JS |
| --- | ---: | ---: | ---: |
| 0-15 | 0.0227 | 0.3147 | 0.0308 |
| 16-31 | 0.0218 | 0.1706 | 0.0453 |
| 32-47 | 0.1253 | 0.0614 | 0.0957 |
| 48-55 | 0.1790 | 0.0242 | 0.1212 |
| 56-61 | 0.2965 | 0.0158 | 0.2944 |

Early readouts depend strongly on which fitted reader is used. This supports
retaining both readouts for exploration, not a verdict on which better tracks
computation. R's motivating early-layer calibration is a reason not to demand
agreement with J as a validity gate; this experiment does not establish that
calibration. Later, J/R agree more closely while the context perturbation has
a larger effect. Plain also changes substantially: at48 its mode JS is
0.2605/0.2453 for bundles/shoobie, versus R's 0.2375/0.2115. This is not merely
one fitted observer manufacturing a mode contrast.

These are descriptive trajectories, not causal variance components. The two
cases differ in length and history as well as subject. Early nearest-factor
labels are particularly fragile: nearest/runner-up gaps are below 0.01 nats in
66/128 layer0-15 anchors, versus 0/112 at48-61. Those thresholds are sensitivity
descriptions, not calibrated confidence bounds. No pre62 pair here exceeds 95%
of the JS ceiling, so the ordering is not just saturation among disjoint rows.

J/R convergence is not universal. Medium bundles61 has identical top1 and the
same eight top-ranked IDs, vector cosine 0.9889, but JS 0.0664 and entropies
7.810 versus 5.320 nats. Shared words, close vectors and distributional agreement
are three different observations.

## 3. Evocative Labels Can Represent Very Little Probability Mass

The first top-eight inspection described xhigh layer48 as emphasizing thinking
and topic. Full scores materially qualify that description: the R top-ranked
thinking token has only 0.600%/0.631% probability in bundles/shoobie, and the
entire top eight contains only 2.42%/2.69%. These are ranks in diffuse
distributions, not states dominated by those verbal concepts.

The 3.6 clarification/correction readouts are more concentrated but reader-
sensitive. At shoobie48, the clarification token has 18.48% under J, 9.30% under
R and 1.65% under plain. Shared labels do not establish agreement about strength,
let alone how the model will handle a correction.

Endpoint behavior is also non-monotone. Bundles mode JS reaches 0.5349 at62;
shoobie R mode JS instead falls from 0.4469 at61 to 0.0702 at62. All scalar rows
stop at post-block62, not the actual final block63. Identity agreement there is
an implementation consistency check, not three independent semantic witnesses.

## 4. The Extra Instruction Does Not Explain The Whole Model Contrast

For checkmate, 3.8 medium and 3.6 thinking have identical complete rendered
bytes, input IDs and spans; their readouts still differ. In the five histories
with an assistant turn, 3.8 medium retains an extra empty historical thinking
block: 19 bytes / four IDs. The model renderers therefore cannot be silently
equated merely because the source messages match.

The exact-input checkmate comparison rules out xhigh as the whole explanation,
not all instrument or checkpoint confounds. Fitted maps differ between models,
and equal hidden dimensions do not align their coordinate systems. Tokenizer
metadata identities differ; observed matching token pieces are not complete
vocabulary verification. No cross-model JS or raw-vector cosine is computed.

## 5. Actual Continuations Complicate A Simple Improvement Story

Readouts alone cannot establish preservation, repair or use of an interpretation.
The four completed continuations supply a small behavioral anchor, not an eval.
They use temperature0, seed17, top_k0, top_p1, min_p0 and serial prefill; prompt
IDs and renderer metadata match the corresponding traces exactly. The trace
capture is packed, while scalar bundles and generation are serial: identical
input is not a claim of bit-identical internal state across these execution modes.

Both bundle answers accept constructing a bundle while rejecting identity of a
bare set with the whole structured object. But the user's domain-specific set
is not a fixed technical category, and the supplied assistant history explicitly
allows it additional structure. Narrowing it to a bare set may miss a legitimate
question about representing structured objects, even when the literal-identity
distinction is defensible. Xhigh enumerates more possibilities:
whole fibers, sections, varying fibers, and a one-point base. Medium foregrounds
the concession; xhigh opens more categorically. More explicit alternatives are
therefore not the same as less categorical rhetoric or a reversal of the main
answer. This is not a mathematical correctness score. A bare set does not
specify bundle structure, but sets can encode structured objects; medium's
claim of no way to encode the distinction overstates this. A categorical
embedding needs specified categories and morphisms, not just an object-level
product construction. Xhigh's larger-total-space language is not generally
valid as a cardinality claim for infinite fibers. The mathematical object,
model interpretation, and user's understanding remain separate objects of inquiry.

In shoobie, both answers continue the cartoon-source framing and distinguish it
from footwear stereotypes rather than asking what the correction refers to.
Medium also retracts a prior unsupported offer to inspect earlier conversations;
xhigh does not. Xhigh replaces the supplied character list with a different list
in both its generated thinking and answer. This is a concrete change to available
information, not evidence establishing which source it used. A capability-claim
repair and categorical source claims coexist in medium; enumerating mathematical
alternatives and corrupting supplied details coexist across xhigh's two cases.
Neither mode uniformly dominates the other on these different dimensions.

A subsequent external spot-check reinforces why historical assistant text is
not ground truth: [A Way with Words / Double-Tongued Dictionary](https://waywordradio.org/shoobie/)
identifies regional New Jersey usage and explicitly calls the often-repeated
shoebox etymology unverified. The retrieved excerpt of Nick Ravo's
[1987 reporting](https://www.nytimes.com/1987/02/16/nyregion/talk-long-beach-island-for-early-tourists-tepid-welcome-jersey-resort.html)
uses the term for regional tourists, predating the cartoon. This is an accessible
article excerpt, not a fresh archival audit of the earliest attestation. Crucially,
the historical assistant proposes the cartoon as the user's personal source;
it does not explicitly assert that the cartoon coined the word. The new answers
strengthen this into the word itself being specifically from the cartoon and
the cartoon having given us the word. Pre-cartoon attestation checks that
universal-origin claim, not cartoon usage or the user's personal association.
The shoebox origin remains unverified rather than established false. Neither
external source was supplied to the model. This scope change could reflect
prior model knowledge/errors or reconstruction from the retained history; we
have not yet varied that evidence to identify its contribution.

## What Seems Worth Following

The interesting object is not another cluster of token labels. It is the relation
between **what the context makes available, boundary-conditioned readouts, and
what the eventual response actually preserves or repairs**. Response preparation
is one hypothesis about that relation, not an established latent mechanism.

Two nearby questions now have concrete footholds:

1. Does the localized thinking-boundary sensitivity follow instruction meaning,
   or system framing/length/position? A length-matched context contrast, native63
   readout and naturally generated thinking positions could separate these more
   directly than another arbitrary layer subset. The current result nominates
   these sites; it does not identify a response-strategy mechanism.
2. With renderer fixed, what happens when one consequential historical assertion
   is removed, attributed as uncertain, or contradicted by independently checked
   evidence? Keep those changes distinct: deletion removes information,
   attribution changes status, correction supplies evidence. Shoobie's selective
   capability repair and persistent source claims make it a particularly useful
   existing case. Test what changes in the answer, not just which labels move.

These are leads, not mandatory gates or a new umbrella program. No detector,
semantic clustering framework, steering direction, or new corpus is earned by
these observations yet. The broad sweep was useful precisely because it exposed
a boundary-conditioned difference that the original question did not specify.

## Evidence And Limits

Private artifacts remain under the observer-controls worktree's
`target/observer-controls/comparative-physiology/`:

- `INDEX.json`, `REPORT.json`, `SEAL.json`: first collection, exact inputs,
  identities, attempts and hashes.
- `medium-INDEX.json`, `medium-AUDIT-v2.json`, `medium-SEAL.json`: adaptive
  renderer follow-up, separate immutable extension.
- `independent-analysis/REPORT.md`, analysis scripts and JSON: independently
  verified complete payloads, full-score metrics, boundary geometry and caveats.
- `continuation-INDEX.json`, `continuation-AUDIT.json`, `continuation-SEAL.json`:
  exact generated streams, phase boundaries, greedy settings and answer text.

An independent analyst rehashes all 322 original/medium sealed files and six
source message files, reconstructs the metrics from all scalar scores, checks
trace coordinates/token binding and all shared vector cells. All twelve
fitted/plain layer62 full-vocabulary comparisons are byte-identical. Separate
scalar/packed R top1 agrees in 377/378 examined rows and ordered top8 in 362/378;
small rank differences remain, not a newly established numerical JS noise floor.

Adversarial review separately recomputes the boundary/suffix counts and all five
JS table rows from eight complete 3.8 J/R payloads, verifies rendered differences
and all four raw continuations, and reads their generated thinking. It identifies
the character-list substitution, distinguishes personal association from universal
word origin, and prevents reader disagreement from being presented as proof of
R faithfulness. It does not repeat the entire 322-file hash audit.

Fitted transfer remains explicitly unvalidated. Generic fitted traces have
deployment locators rather than authenticated content bindings; strong identities
in plain bundles do not retroactively authenticate earlier captures. Device and
kernel settings are not comprehensively bound by the current schema. Source
revision provenance for the older 3.6 fit is less complete. Bundles is far beyond
the T128 fitting horizon. These observations do not rerun or invalidate the
separate historical BF16/Q8 qualification.

Initial CLI setup failures, a lease failure followed by ordinary waiting, an
incompatible trace-comparison rejection, and a separate 16-token capped smoke
remain recorded. The capped smoke is not one of the four completed continuations.
No other session's process was interrupted, and no private transcript text or
raw activation payload is added to Git.
