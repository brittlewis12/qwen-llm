# Crossed Task: Strong Transfer, But Which Variable?

Follow-up to [depth and order](LENS-DEPTH-AND-ORDER.md), using authored fictional
records instead of another archive-wide generation pass. The principal finding
is a qualification, not an operator discovery: a task-associated final-prefill
contrast transfers strongly across content within one definition order, but
reverses across the two tested content/order holdouts. A subsequent exploratory
parity decomposition finds a smaller aligned order-even component as well.

## Design And Preserved Inputs

- [Exact conditions](experiments/lens-crossed-task/conditions.json): four
  fictional domains, two inferential jobs, and two counterbalanced A/B codebooks.
- [Frozen response rubric](experiments/lens-crossed-task/rubric.md): conditional
  consequence accounts and evidential discrimination rated separately, without
  requiring different conclusions or particular task-label words.

Explanation takes H1 as a stipulated premise and derives its consequences;
diagnosis compares H1/H2 predictions against observations. Both definitions are
present in every prompt. Evidence favors H1 throughout, so final hypothesis
identity is not the intended difference. This is an easy deterministic toy task,
not realistic uncertain diagnosis. Stipulation anchoring remains possible.

The base 16 conditions cross four subjects, two codebooks, and two selected
letters. Adversarial design review identified a remaining confound: explanation
was always the first definition. Before data collection, eight reversed-definition
controls were added on radiator-noise and software-cache subjects, retaining
both codebooks and selections. Total: 24 conditions; only two subjects have
order controls.

Tokenization verifies equal counts and checkpoint positions within content,
including the tested order reversal. Only three code-letter slots vary within
each content/order family. All final-prefill sites consume token271. Prefixes
are identical only within the appropriate content/order/mapping controls, not
across reordered definitions. Total supplied prompt tokens: 7,968.

The same frozen Qwen3.6-27B Q8 reader/generator executable was used throughout,
from build `14e4ce68`, SHA256
`dedadc401c23d665ea118ef67270e154f3b070b1c7d2a15a7f6e4d7b7d383cbc`.
Generation uses the explicit no-thinking prompt, greedy sampling, seed0, and a
384-token cap. No visible reasoning channel does not imply absent computation.

## Behavioral Check, Not Ground Truth

All 24 responses stop naturally, with 4,810 total generated tokens and no
resampling. All use the requested headings. All fail the 90-120-word constraint:
observed whitespace-delimited lengths, including headings, are 125-217 words.
Matched formatting is not matched length or syntax.

Two separate LLM-agent coding passes receive opaque IDs, the common vignette,
and response text, without requested job, order or code assignment. Each rates
both inferential relations under the frozen rubric before joining assignments:

- All 12 requested explanations: conditional relation present, diagnostic absent.
- Eleven requested diagnoses: diagnostic present, conditional absent.
- One reversed software-cache diagnosis: both ambiguous; the rationale includes
  an erroneous H2 prediction. It remains in every primary analysis.
- Coders agree on both relation labels for all 24 outputs. This is agreement
  between model-assisted instruments, not human evaluation, independent accuracy
  evidence, or proof that an activation implements either relation.

The original rubric's prohibition on an external judge model is preserved
verbatim. Actual coding used assistant subagents, not a separate judge endpoint;
do not read that wording as human-only or independently validated judgment.

On the two order-control subjects, the diagnosis-minus-explanation mean length
gap changes from -2.5 words forward to +30.25 reversed: a +32.75-word shift.
This does not contaminate the earlier prefill with future tokens, but prevents
interpreting later generated-anchor comparisons as length/format-matched without
additional controls. No generated-anchor readouts were collected here.

Generation used packed passive spans; the final primary states are scalar
recaptures. They are not demonstrated to be the exact states that produced the
responses. Behavioral coding is context, not exact-state causation.

## Execution Failure And Scalar-Only Amendment

The first generation attempt rejected an empty Lens plan before producing any
outputs. The established archive passive plan replaced it: one readonly token
readout at prefill position0/layer62, with no directions or interventions. The
first successful cohort then exposed omitted-versus-null optional-field handling
in validation; canonical-equivalence tests repaired validation without rerunning
its four outputs. Original attempts and producing/validation identities remain.

The initially planned R packed/scalar bridge failed its relative-L2 gate of
1e-4 at all seven inspected layers: observed errors 3.46e-4 to 7.81e-4. This
scale was already present in earlier numerical qualification. It is not a new
model-behavior failure or a reason to call the paths interchangeable. At layer7,
one cell's discrepancy was 31.5% of the packed factorial S norm; that ratio is
not a measured full-factorial error or statistical noise estimate.

After inspecting these partial magnitudes, execution was explicitly amended:
retain the failure, do not relax its threshold, and use scalar R/native for
every primary final-prefill comparison. This is a post-data execution amendment,
not an untouched preregistration. Stimuli, assignments, statistics and exclusions
did not change. The four collected packed prefix controls remain historical
diagnostics; none substitutes for uncollected scalar prefix states.

The final panel contains 48 scalar bundles, 3,048 vectors and full-vocabulary
rows: R layers0-62 and native/plain layers0-63 for all24 conditions. One native
bundle was reused and 47 new scalar calls completed. Raw score payload:
3,027,517,440 bytes. Total private artifacts approximately 4.03 GB under a
provisional 6 GiB storage budget; no hard in-flight quota is claimed.

All source tokens, vector shapes, finite values, payload hashes and full top-eight
rankings validate. The first seven scalar R rows match their repeated all63-layer
capture bit-for-bit, vectors and logits. All24 native62/R62 pairs also match
bit-for-bit. These establish scoped same-mode consistency and shared identity,
not independent replication, scalar/packed equivalence, or new BF16/Q8 authority.

## Contrasts And Main Results

For forward definition order, let x[m,a] be the native or R vector at a given
content and layer; plus/minus denotes codebook, A/B the selected letter:

```
d_plus  = x[plus,A] - x[plus,B]
d_minus = x[minus,B] - x[minus,A]
S = (d_plus + d_minus)/2       # explanation minus diagnosis
L = (d_plus - d_minus)/2       # literal A minus B
```

S uses the true semantic assignment under reversal: the signs in cell order
plus-A, plus-B, minus-A, minus-B are forward `+,-,-,+` and reversed `-,+,+,-`.
The mapping main effect is tracked separately. Orthogonal factorial coefficients
do not imply orthogonal activation vectors.

The signed statistic compares semantic-signed deltas across held-out content
and opposite codebook. Because d combines S with opposite L, large letter effects
can make it negative even with transferable S. Essential companion statistics
therefore compare factorial S and L across held-out contents and report norms.
Native and R geometries remain separate; no classifier or population p-value
is fitted from repeated layers.

Equal-depth means over layers0-62, using raw references:

| Measurement | Native | R |
| --- | ---: | ---: |
| Signed opposite-codebook transfer, forward | 0.22369 | 0.18294 |
| Cross-content S alignment, forward | 0.95030 | 0.95265 |
| Cross-content literal L alignment, forward | 0.90215 | 0.90770 |
| Cross-content AND cross-order S alignment | -0.62864 | -0.61219 |

Raw/unit references and recomputed content leaveouts preserve the main pattern.
Native63 is separate, not included in the 63-layer mean. Cross-order layer-mean
S is negative at every layer. Four directed cross-order folds reduce to two
unique cosine comparisons, duplicated symmetrically; only two contents support
this control, not four independent replications or 63 independent depth trials.

The signed statistic is negative early and positive late, while order transfer
remains negative throughout:

| Depth band | Native signed | R signed | Native order transfer | R order transfer |
| --- | ---: | ---: | ---: | ---: |
| 0-15 | -0.75650 | -0.83875 | -0.95803 | -0.95534 |
| 16-31 | 0.03918 | -0.05297 | -0.85893 | -0.86800 |
| 32-47 | 0.80076 | 0.81768 | -0.30543 | -0.26384 |
| 48-62 | 0.85051 | 0.84733 | -0.37643 | -0.34487 |

Within one order, abstract definition-slot selection can masquerade as a very
transferable semantic-job contrast. The reversal supports an order-sensitive or
context-dependent representation, not an invariant explanation/diagnosis axis.
It does not establish that semantic information is absent or inaccessible.
An ordinal contrast's positive cross-order score is algebraically the negative
of S here, not independent corroboration of an ordinal mechanism.

## Post-hoc Parity Diagnostic

After the negative cross-order result, the two order-control contents were
decomposed without fitting a rotation or selecting layers:

```
E = (S_forward + S_reverse)/2    # order-even contrast component
O = (S_forward - S_reverse)/2    # order-odd contrast component
```

This diagnostic is explicitly post-hoc. It cannot rescue confirmatory semantic
identification. E could reflect semantic-definition identity, order-invariant
binding or common form; O could reflect ordinal selection or context-dependent
semantic coordinates. Their activation vectors are not generally orthogonal.

Across the two contents, whole-depth mean E/O cosines are native 0.8112/0.9242
and R 0.8054/0.9273: a smaller aligned order-even component coexists with the
dominant reversal. This is one content pair per layer, not an expanded sample.

The mean layerwise E fraction, `||E||^2 / (||E||^2 + ||O||^2)`, is 16-18%.
Pooling squared norms over depth instead gives 32-34%; these are different
aggregates, neither model variance explained. Native mean layerwise E size is
1.98%/1.84% of mean state norm on radiator/cache, respectively.

Depth is consequential: in native0-15, E is about 0.12% of contrast parity
energy, rising to about 31-35% in32-47 and 31-32% in48-62. Early normalized
alignment must therefore not be confused with large early amplitude. Numerical
repeatability checks do not provide a statistical noise bound or semantic
signal-to-noise estimate.

Reconstruction is exact; the scaled parallelogram-identity error is at most
4.74e-16. Full per-layer norms, E/O inner products, denominator conventions and
fixed-band summaries are retained, including pooled-versus-layerwise aggregates.

## Review, Evidence And Next Question

Adversarial review occurred before execution, after the scalar-only amendment,
and after results. The result auditor independently re-extracted all3,048 vectors
from source bundles, verified all score hashes and top-eight rankings, rebuilt
the semantic truth table and all primary S/order calculations, and checked 467
local final-binding hashes. It separately reconstructed the parity diagnostic
from source vectors. No sign error or numerical blocker was found.

The auditor did not rerun inference, calibration, every secondary distribution
statistic, or historical coder timing. Local hashes bind current evidence, not
historical events or an untouched precommitment. The matched n25 R artifact's
transfer declaration is distinct from existing n1000 J precision qualification;
no precision qualification was repeated here.

Private root in the observer-controls worktree:
`target/observer-controls/crossed-task/`. Important evidence:

- `scalar-only/summary.json`, `results.md`, `final-bindings.json`: primary
  calculations, all folds and norms, producing/source hashes and amendment chain.
- `scalar-only/parity-analysis/REPORT.md` and `results.json`: separate post-hoc
  diagnostic and independently bound source/script evidence.
- `responses/coding-joined.json`: frozen coder hashes, all relation labels,
  assignment joins, word counts and ambiguous outputs, without exclusions.
- `generation-completion.json` and `readout-stop-diagnostics.json`: actual
  generation counts and retained mixed-mode failure.

Synthetic stimuli and rubric are versioned here; actual responses, exact token
arrays, binary scores and analysis scratch remain local. Preserve these artifacts
before cleaning target. No runtime source, dependency or inference defaults changed.

The useful next question is not whether the dominant S can be renamed semantic.
It is whether the smaller order-even component follows the requested inferential
relation beyond this wording/selection scheme, and when that relation becomes
operative. A new definition paraphrase or selection scheme can test transfer;
matched heading-state inspection can test recruitment during execution, while
retaining generated-history confounds explicitly. A later held-out-content,
equal-norm E-versus-O pulse test would address causal use. Neither low entropy
nor this geometric decomposition alone justifies calling either component an
operator, belief, or measure of conviction.
