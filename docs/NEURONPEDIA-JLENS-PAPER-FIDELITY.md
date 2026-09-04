# Neuronpedia JLENS Paper Fidelity

- Audit date: 2026-09-01
- Paper: *Verbalizable Representations Form a Global Workspace in Language Models*
- Upstream reference revision: `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- Neuronpedia source revision: `5580619db2509a34a06a384a519479938c24ee74`
- Scope: paper and appendices, current upstream reference, Neuronpedia serving
  and UI, and published lens artifacts

This assessment distinguishes paper fidelity, upstream-reference fidelity, and
product extensions. It does not assume that the audited current source revision
was the exact historical production deployment used for exploratory probes.

I completed the comparison against the full paper, its appendices, the current upstream repository, Neuronpedia’s serving/UI implementation, and the published lens artifacts.

**Bottom Line**

Neuronpedia is:

- **Highly faithful to the released Jacobian-lens estimator and basic readout.**
- **A strong, carefully engineered deployment of J-lens readouts across open-weight models.**
- **Not an implementation of the paper’s formal J-space decomposition or experimental suite.**
- **Only partially faithful in additive steering and single-direction ablation.**
- **Substantially unfaithful in what it calls a “swap”: Neuronpedia implements a different mathematical intervention.**
- **More confident in some user-facing “thought” and “workspace” claims than either its measurement or the paper warrants.**

The largest substantive problem is the swap. Neuronpedia’s intervention is a source-to-target projection transfer, not the paper’s pseudoinverse coordinate exchange. Even with orthogonal directions, they generally produce different activations.

The other major interpretive issue is terminology: the Neuronpedia “J-Space” sidebar is an occurrence histogram over filtered top-eight J-lens tokens. It is not the sparse, nonnegative, gradient-pursuit decomposition that formally defines J-space in the paper.

None of this invalidates Neuronpedia as an exploratory J-lens application. It does mean that a Neuronpedia result should not automatically be described as reproducing a paper intervention, measuring formal J-space contents, or verifying the global-workspace hypothesis.

**Sources And Branch Safety**

I reviewed:

- The complete paper and appendices: [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/), published July 6, 2026.
- `anthropics/jacobian-lens` at current GitHub `main`, commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- The full Neuronpedia fit, artifact loading, readout, intervention, UI, tour, and documentation paths.
- The published [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens) artifact repository.

I verified GitHub’s current `main` directly. The local reference checkout is clean on `main`; the separate local worktree remains on `feature/rlens`. I did not switch, update, or modify either checkout.

The upstream repository explicitly calls itself an unmaintained reference implementation at `/Users/tito/code/jacobian-lens/README.md:3`.

---

## Fidelity Overview

- **Average-Jacobian estimator:** High fidelity to upstream `main`; moderate-to-high fidelity to the paper’s general method.
- **Actual paper fitting recipe:** Partial. Target layer, early-position handling, corpus, model family, precision, and prompt count differ.
- **J-lens readout:** High fidelity.
- **Model-specific output decoding:** High fidelity and, in several respects, more robust than upstream.
- **Interactive visualization:** Conceptually faithful, but Neuronpedia’s aggregation and defaults are materially different.
- **Formal J-space:** Not implemented.
- **Additive J-lens steering:** Direction is faithful; scaling and saturation are not.
- **Single-direction ablation:** Faithful.
- **Paper-style top-k J-space ablation:** Not implemented.
- **Coordinate swap:** Low fidelity; different operation.
- **Layer and position protocols:** Flexible product extension, but unable to reproduce several paper protocols exactly.
- **Open-model and multi-stream support:** Principled, impressive extension.
- **Scientific/user-facing documentation:** Mixed—excellent internal engineering rationale, but important mathematical and epistemic caveats are absent from the UI.

---

# 1. The Paper’s Precise Mathematical Object

For layer \(l\), source position \(p\), and a future target position \(q \ge p\), the paper defines an average Jacobian mapping an intermediate residual into the target-layer residual basis:

\[
J_l
=
\mathbb{E}_{\text{prompt},p,q\ge p}
\left[
\frac{\partial h_{T,q}}{\partial h_{l,p}}
\right].
\]

The released estimator operationalizes this slightly more specifically:

\[
J_l
=
\mathbb{E}_{\text{prompt}}
\mathbb{E}_{p}
\left[
\sum_{q\ge p}
\frac{\partial h_{T,q}}{\partial h_{l,p}}
\right].
\]

It sums all causally reachable future-target gradients for a source position, averages source positions within each prompt, then gives each prompt equal weight.

An activation is read as:

\[
z_{l,p}=J_l h_{l,p},
\]

\[
\operatorname{logits}_{l,p}
=
W_U\,N(z_{l,p}),
\]

\[
P_{l,p}
=
\operatorname{softmax}(\operatorname{logits}_{l,p}).
\]

Here \(N\) is the model’s actual final normalization and \(W_U\) its unembedding.

For token \(t\), with unembedding row \(w_t\), its layer-\(l\) J-lens direction is:

\[
v_{t,l}=J_l^\top w_t.
\]

This is exactly the direction Neuronpedia constructs for a fitted J-lens layer at `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:661`.

---

# 2. Fitting The Average Jacobian

## What matches

Neuronpedia’s vendored fitter uses the same fundamental estimator as upstream:

- Replicate the prompt across a dimension batch.
- Place one-hot cotangents in several output dimensions at once.
- Place each cotangent at every valid target position.
- Backpropagate to every requested source layer.
- Average source-position gradients within the prompt.
- Average the resulting matrices equally across prompts.

The upstream explanation is especially clear at `/Users/tito/code/jacobian-lens/jlens/fitting.py:100`; the Neuronpedia copy performs the same operation at `utils/neuronpedia-utils/neuronpedia_utils/jlens/jlens/fitting.py:112`.

This is the strongest part of the fidelity story. The served matrices really are average causal Jacobian transports, not fitted classifiers or tuned-lens maps.

## Position mismatch

Current upstream and Neuronpedia use only positions:

\[
p\in\{16,\ldots,T-2\}.
\]

They exclude:

- Positions 0–15, described as attention-sink or burn-in positions.
- The final sequence position.

See `/Users/tito/code/jacobian-lens/jlens/fitting.py:40` and `utils/neuronpedia-utils/neuronpedia_utils/jlens/jlens/fitting.py:84`.

This is faithful to upstream `main`, but not to the paper’s reported default experiment. The paper appendix says excluding early positions was evaluated as a methodological variant and did not meaningfully improve the default. Its actual default included them.

Assessment:

- **Principled:** Yes. Attention-sink statistics are a plausible reason.
- **Faithful to released code:** Yes.
- **Faithful to the reported paper default:** No.
- **Well surfaced to users:** No. Neuronpedia warns about early *layers*, not early token positions, even though the fit omitted those token positions.

Neuronpedia nevertheless displays readouts at all prompt positions. Those first 16 positions are therefore applications of a corpus-average map fitted without corresponding source-position examples.

## Target-layer mismatch and paper ambiguity

The paper has three relevant statements:

1. The main method is written using the final residual layer.
2. The appendix says the actual default Sonnet 4.5 lens used throughout the paper targeted the **penultimate** layer.
3. The paper’s pseudocode again says the target is final by default.

The released repository defaults to the final transformer block output, while allowing another target:

`/Users/tito/code/jacobian-lens/jlens/fitting.py:75`

Neuronpedia’s native fits also default to final:

`utils/neuronpedia-utils/neuronpedia_utils/jlens/fit_lens.py:241`

Therefore Neuronpedia is faithful to the public reference default but not to the actual default lens used for the central Sonnet results.

This matters because the paper reports that including the final block can add noisy artifacts; it describes that block as heavily specialized toward next-token calibration.

## Corpus and prompt-count differences

The paper’s default is:

- 1,000 sequences.
- 128 tokens each.
- Sampled from a pretraining-like distribution.

Neuronpedia’s normal batch fit is:

- WikiText-103 raw training split.
- Streamed from the beginning without an explicit shuffle.
- Concatenated and cut into deterministic 2,000-character chunks.
- Tokenized and truncated to at most 128 tokens.
- Nominal cap of 1,000 prompts.
- Usually stopped early when a convergence statistic falls below a threshold.

The corpus construction is at `utils/neuronpedia-utils/neuronpedia_utils/jlens/fit_lens.py:45`; defaults are at `utils/neuronpedia-utils/neuronpedia_utils/jlens/run-all-fit-lens.py:38`.

The early-stop criterion is:

- At least 100 accumulated prompts.
- Mean of the last ten relative running-Jacobian changes.
- Stop when that mean is below \(2\times10^{-3}\).

See `utils/neuronpedia-utils/neuronpedia_utils/jlens/fit_lens.py:121`.

A representative Qwen 3.5 4B artifact requested 1,000 prompts but stopped after 417.

Assessment:

- WikiText is reasonably “pretraining-like,” but narrower than the paper’s generic corpus.
- Deterministically taking the first streamed chunks may introduce topic/order bias.
- Early stopping is computationally sensible and consistent with upstream’s statement that roughly 100 prompts produce a usable lens.
- Convergence of the Frobenius running mean is not the same as convergence of semantic rankings or intervention behavior. The chosen threshold is an engineering heuristic, not a paper-validated universal criterion.

## Published-artifact audit

As of September 1, 2026, the Hugging Face repository contained approximately:

- 39 model directories.
- 40 `.pt` lens artifacts.
- 38 sidecar configurations.
- 37 artifacts produced with Neuronpedia’s adapted fitter.
- 3 externally fitted/converted artifacts.

Native fits ranged from roughly 125 to 1,000 completed prompts; most stopped before 1,000.

The DeepSeek-V4-Flash artifact is an important exception:

- 25 Pile documents.
- Maximum sequence length 128.
- First four positions skipped.
- Target layer 41.
- Block-output activation reduced by the mean over four residual streams.

Its published configuration is much more explicit than native historical artifacts.

---

# 3. Applying The Lens

Neuronpedia’s core readout is highly faithful:

\[
h_{l,p}
\longmapsto
J_lh_{l,p}
\longmapsto
W_UN(J_lh_{l,p}).
\]

The transport is implemented at `apps/inference/neuronpedia_inference/endpoints/lens/lens_loader.py:323`.

It also correctly:

- Uses the model’s actual final normalization.
- Uses the model’s actual unembedding.
- Reproduces model-specific logit multipliers.
- Reproduces Gemma 2 final-logit softcapping.
- Computes softmax normalization over the full vocabulary.
- Appends the true model final-layer output as a direct, untransported row.

The model-specific behavior is documented at `apps/inference/neuronpedia_inference/endpoints/lens/model_specific.py:1`.

These are principled extensions and arguably more production-correct than a minimal reading of the public reference implementation.

## Precision difference

The path is not full precision:

1. Native fitting usually runs the model in BF16.
2. Per-prompt Jacobian values are accumulated in FP32.
3. The lens file stores matrices as FP16.
4. Serving normally loads them as the served model’s dtype, commonly BF16.
5. The transport matmul is performed in that lower precision.
6. Results are converted to FP32 before softmax/top-k processing.

See `utils/neuronpedia-utils/neuronpedia_utils/jlens/jlens/lens.py:61` and `apps/inference/neuronpedia_inference/endpoints/lens/lens_loader.py:152`.

This is a reasonable memory/bandwidth tradeoff. It can still alter near-tied rankings and intervention directions. The existing 3% transport test at `apps/inference/tests/unit/test_lens_loader_memory.py:110` checks BF16 multiplication against FP32 multiplication of the already-quantized matrix; it is not an end-to-end comparison against an FP32 paper lens.

---

# 4. Formal J-Space Versus Neuronpedia “J-Space”

This is the largest conceptual discrepancy after swap semantics.

## The paper’s formal definition

At layer \(l\), for sparsity \(k\), the paper defines J-space as:

\[
\mathcal{J}_{l,k}
=
\left\{
\sum_{i=1}^{m} a_i v_{t_i,l}
:
m\le k,\;a_i\ge0
\right\}.
\]

Because the J-lens dictionary is overcomplete and nonorthogonal, the top \(k\) tokens by logit or inner product are not a unique decomposition.

The paper estimates an activation’s J-space component by gradient pursuit:

\[
h_l^{J}
\approx
\arg\min_{x\in\mathcal{J}_{l,k}}\|h_l-x\|^2.
\]

This sparse nonnegative decomposition underlies its claims about:

- Workspace occupancy of approximately 25 concepts.
- Fraction of variance explained.
- J-space versus non-J-space components.
- Concept-vector and probe decompositions.
- Top-k J-space ablation.
- The claim that J-space is a sparse subframe rather than an ordinary subspace.

## What Neuronpedia computes

Neuronpedia does not perform that decomposition.

For every layer-position cell, it receives a ranked list, normally the top eight filtered tokens. The sidebar then:

1. Restricts cells to the selected layer and position range.
2. Counts every appearance of an exact decoded token string.
3. Gives rank 1 and rank 8 equal count.
4. Gives a high-probability and near-zero-probability appearance equal count.
5. Sorts by total occurrences.
6. Displays the first 100 entries.

See `apps/webapp/components/jlens/jlens-analysis.tsx:168`.

Because deduplication is currently disabled:

- `" spider"` and `"spider"` are different entries.
- Capitalization variants are different entries.
- Tokenizer-specific fragments remain distinct.

See `apps/webapp/components/jlens/jlens-token-popup.tsx:112`.

This is best described as a **top-k readout salience histogram**, not a J-space-coordinate inventory and not a frequency of thoughts.

The paper itself sometimes uses “J-space contents” informally when discussing high-ranked J-lens tokens, so the UI label is not wholly disconnected from the paper’s rhetoric. But the distinction becomes essential whenever making structural claims about occupancy, variance, selectivity, decomposition, or “the most active thought.”

The guided tour’s statement that the highest-count entry is the model’s “most frequently occurring thought” at `apps/webapp/app/[modelId]/jlens/jlens-tour.tsx:117` is not justified by the implemented statistic.

## Upstream code also omits formal J-space

This omission is not uniquely a Neuronpedia failure. Current `anthropics/jacobian-lens` contains fitting, application, visualization, and tests, but no sparse decomposition, gradient-pursuit, swap, steering, ablation, or experimental runner.

The experiment README documents intended protocols, but not executable implementations:

`/Users/tito/code/jacobian-lens/data/experiments/README.md:5`

Neuronpedia therefore implements much more product functionality than upstream released, but it should not imply that the unreleased formal J-space machinery is present.

---

# 5. Additive Steering

For token \(t\), Neuronpedia correctly derives the layer direction:

\[
v_{t,l}=J_l^\top w_t,
\qquad
\hat v_{t,l}=\frac{v_{t,l}}{\|v_{t,l}\|}.
\]

For multiple API tokens it sums the individually normalized directions:

\[
d_l=\sum_i \hat v_{t_i,l}.
\]

The normal UI steers one token.

For each selected layer and token position, Neuronpedia computes:

\[
u=s\,\|h\|\,d_l,
\]

then caps it:

\[
u_{\text{cap}}
=
u\cdot
\min\left(1,\frac{\|h\|}{\|u\|+\epsilon}\right),
\]

and applies:

\[
h'=h+u_{\text{cap}}.
\]

See `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:762`.

For one unit direction, this simplifies to:

\[
h'
=
h+\operatorname{clip}(s,-1,1)\,\|h\|\,\hat v.
\]

Consequently:

- `+1.0` and `+2.0` are mechanically identical for one-token steering.
- `-1.0` and `-2.0` are also identical.
- The public API permits strengths through ±50, but all magnitudes at or above 1 saturate for a single direction.
- Half of the UI’s ±2 slider range is redundant.

## Comparison with the paper

The paper’s generic operation is:

\[
h'=h+\alpha v_t.
\]

The companion experiment documentation specifies a more comparable normalization:

\[
h'
=
h+
\alpha
\cdot
\overline{\|h_l\|}
\cdot
\hat v_{t,l},
\]

where \(\overline{\|h_l\|}\) is a layer-level mean residual norm, not the current token position’s norm.

See `/Users/tito/code/jacobian-lens/data/experiments/README.md:28`.

Neuronpedia therefore preserves the intended idea of dimensionless, norm-relative steering but changes it in two ways:

- Per-position norm rather than a fixed layer mean.
- Hard one-residual-norm cap.

These are principled stability/portability choices, especially across many model families, but they change the intervention:

- Strength varies by token position.
- High-norm positions receive larger absolute changes.
- The response is nonlinear in requested strength.
- Published paper strengths cannot be reproduced directly.
- Multi-layer effects still compound even though each individual layer is capped.

The cap and saturation are well documented internally but not surfaced clearly to users.

---

# 6. Ablation

For one direction, Neuronpedia applies:

\[
h'
=
h-(h^\top\hat v)\hat v.
\]

This is a faithful one-direction projection ablation.

See `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:785`.

Important limitation: if multiple `steer_tokens` are supplied through the API, Neuronpedia sums their unit vectors and projects out that single summed direction. It does not project out their span or independently remove each coordinate.

Therefore:

- **Single-token ablation:** Faithful.
- **Paper-style top-10 or top-25 J-space ablation:** Not reproduced.
- **Formal J-space component ablation:** Not available.
- **Matched-norm and random-direction controls:** Not provided by the application.

A behavioral change after one-token ablation supports causal relevance of that readout direction. It does not reproduce the paper’s claims about suppressing the broader J-space.

---

# 7. Swap: The Major Mechanical Deviation

## Paper operation

For source and target vectors \(v_s,v_t\), define:

\[
V=[v_s\;\;v_t].
\]

Read their joint coordinates using the pseudoinverse:

\[
c=V^\dagger h.
\]

Exchange the two coordinates:

\[
c'=\sigma(c).
\]

Then patch:

\[
h'
=
h+\alpha V(c'-c).
\]

For \(\alpha=1\), the two coordinates are exchanged. The paper also uses \(\alpha=2\) “double-strength” swaps in flexible-generalization experiments.

This construction:

- Accounts for nonorthogonality.
- Reads source and target coordinates jointly.
- Swaps both existing coordinates.
- Leaves the component orthogonal to their joint span unchanged.

## Neuronpedia operation

Neuronpedia normalizes source and target separately and performs:

\[
c_s=h^\top\hat s,
\]

\[
h'
=
h-c_s\hat s+c_s\hat t.
\]

See `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:814`.

It:

- Reads only the source projection.
- Removes that source amount.
- Adds the same amount along the target.
- Ignores the target’s pre-existing coefficient.
- Does not solve joint coordinates.
- Has no \(\alpha\) parameter.
- Takes precedence over steering strength and ablation.

## Difference even for orthogonal directions

Suppose \(\hat s,\hat t\) are orthonormal and:

\[
h=c_s\hat s+c_t\hat t+r.
\]

The paper produces:

\[
h'_{\text{paper}}
=
c_t\hat s+c_s\hat t+r.
\]

Neuronpedia produces:

\[
h'_{\text{NP}}
=
0\hat s+(c_t+c_s)\hat t+r.
\]

They agree only in special cases, such as \(c_t=0\).

For nonorthogonal directions, Neuronpedia’s dot product is not even the correct source coordinate in the two-vector basis. After its update, the activation can retain a source projection proportional to \(\hat s^\top\hat t\).

## Assessment

- The implementation corresponds to the paper’s simplified prose—“subtract the source projection and add an equal target projection”—but not its formal method.
- It is computationally simpler and may be numerically more stable, but I found no explicit justification for choosing it over the pseudoinverse operation.
- Calling it “swap” without qualification is misleading.
- A successful Neuronpedia swap establishes that transferring source-aligned activation into the target direction changes behavior. It does not establish the result of exchanging the paper’s lens coordinates.
- Paper results involving \(\alpha=2\) cannot be reproduced at all.

This deserves either:

1. Replacement with the formal pseudoinverse swap, or
2. Renaming to “projection transfer,” while offering formal swap separately.

---

# 8. Layer Selection

## Paper

The paper generally evaluates 25 evenly spaced residual layers, normalized to 0–100 depth.

For Claude, it identifies approximately:

- Before L38: noisy/early regime.
- L38–L92: workspace regime.
- After L92: output or “motor” regime.

This is empirical and model-specific, not a universal transformer constant.

## Neuronpedia

Neuronpedia normally reads every fitted layer and the true final layer.

The default visible range begins at 29% depth and extends through the final layer:

`apps/webapp/components/jlens/use-jlens-analysis.tsx:307`

That is a reasonable approximation of the paper’s “first third is noisy,” and the UI explicitly warns users:

`apps/webapp/components/jlens/jlens-token-popup.tsx:30`

But the range extends through 100%, including the paper’s output/motor regime.

Fresh intervention defaults are:

- **Additive steering/ablation:** One layer—the layer where the token occurs most often in the current analysis scope. If it never occurs, use the last available layer.
- **Swap:** Every layer in the current sidebar range, normally approximately 29% through 100%.
- **Manual selection:** Any noncontiguous subset.

See `apps/webapp/components/jlens/use-jlens-analysis.tsx:67`.

Thus default swaps are substantially broader than the paper’s approximate L38–L92 workspace band and can include the model’s final output representation.

The blog itself observes that smaller models can be “oversteered” by broad layer selections at `apps/webapp/app/blog/posts/jacobian-lens.mdx:40`. That is useful empirical caution, but not a calibrated per-model workspace definition.

---

# 9. Token-Position Scope And Prefill/Decode

This differs significantly from several paper protocols.

## Neuronpedia’s exact behavior

At every selected layer, the intervention is applied to:

- Every prompt/prefill token position.
- Except positions whose token ID is exactly the BOS ID.
- Optionally every generated-token position as it is processed.

The BOS exemption is at `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:731`.

The UI’s token-position selection controls:

- Which cells are aggregated.
- Which layer looks like the occurrence peak.
- What the sidebar displays.

It does **not** constrain where the intervention is applied.

There is no arbitrary intervention position mask in the UI or request model.

## Prefill only

With generated-token intervention off:

1. Every non-BOS prompt position is modified during the prefill forward pass.
2. The modified prompt creates the KV cache.
3. The first generated token is sampled from the modified prompt’s logits.
4. Generated-token residuals are not directly modified.
5. They remain indirectly affected through the steered KV cache and changed generated history.

## Prefill and decode

With generated-token intervention on:

1. Prefill is modified as above.
2. Each generated token is also modified when fed into the model.
3. That modification affects its displayed readout and the distribution of the following token.
4. The clamp is therefore renewed through generation.

See `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:1290`.

Defaults differ by entry point:

- Inference API default: generated-token intervention **off**.
- Fresh web UI intervention: **on**, even though the toggle is under Advanced.
- Older restored shares that lack the field: off.
- Guided-tour fixture: not a live inference run.

## Comparison with paper protocols

The paper and companion data use several masks:

- Every prompt position.
- Every token in the user’s question turn.
- Every token in a stimulus span.
- Particular scored positions.
- Contiguous workspace layer bands.

Neuronpedia can reproduce “all prompt positions” approximately, but cannot reproduce “only the user turn” or “only the stimulus span” when additional system, template, history, or assistant-prefill tokens are present.

Persistent decode-time clamping is a useful novel experiment, but it is not specified by the released reference repository and generally is not equivalent to the paper’s prompt-only interventions.

---

# 10. Filtering And Display Defaults

Neuronpedia defaults to:

- Top eight tokens per layer-position cell.
- Non-word filtering enabled.
- Full-vocabulary normalization retained.
- Probabilities rounded to four decimals.
- Exact token strings kept distinct.

The paper commonly reports top 10; formal sparse decompositions often use \(k\le25\).

Filtering changes which tokens users see, though not their probability denominator. Special tokens and punctuation can disappear from intermediate rows even if they are the true argmax.

The implementation preserves only the final output row’s true unfiltered top-1:

`apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:1518`

The request-schema documentation says the true top-1 at “each layer” is preserved:

`apps/inference/neuronpedia_inference/schemas/lens.py:124`

That documentation is incorrect.

Filtering is a defensible UX feature—raw lens outputs can be dominated by tokenizer artifacts—but it means a displayed “top token” often means “top token among word-like vocabulary entries,” not the actual top J-lens token.

---

# 11. Multi-Stream Models And Other Extensions

Neuronpedia has thoughtfully extended J-lens beyond conventional single-stream transformers.

For hyper-connection models such as DeepSeek-V4, an artifact records:

- Architectural capture point.
- Stream reduction: mean, sum, or selected stream.
- Optional stream index.

The server refuses to guess when this provenance is missing because every candidate reduction has the same width and could produce plausible but incorrect tokens.

See `apps/inference/neuronpedia_inference/endpoints/lens/residual_spec.py:1`.

This is excellent scientific and engineering practice.

For interventions:

- A lens fitted on one selected stream writes only that stream.
- A mean/sum lens writes every stream.
- Linear ablation and Neuronpedia’s projection-transfer swap commute appropriately with mean/sum reduction.
- Additive steering is explicitly documented as only an analogue: each stream’s injection is scaled by that stream’s own norm rather than the norm of the reduced mixture.

See `apps/inference/neuronpedia_inference/endpoints/lens/residual_spec.py:105`.

These are principled extensions outside the original paper’s scope.

---

# 12. Artifact Provenance And Reproducibility

Neuronpedia records much useful information:

- Model ID.
- Dataset and split.
- Character chunk size.
- Prompt cap and actual completed count.
- Sequence length.
- Target layer argument.
- Model dtype and device map.
- Early-stop settings.
- Exact command.
- GPU details.

But the claim that the native `config.yaml` contains “everything needed to reproduce the lens” at `utils/neuronpedia-utils/neuronpedia_utils/jlens/README.md:81` is too strong.

Native configurations do not pin:

- Model revision.
- Tokenizer revision.
- Dataset revision.
- Exact prompt texts.
- Neuronpedia git commit.
- Vendored jlens revision.
- Hard-coded `skip_first=16`.

Thus they reproduce the intended protocol, not necessarily the exact matrix.

The `.pt` files historically contain only:

- `J`
- `n_prompts`
- `source_layers`
- `d_model`

Native serving does not read the sidecar config. Capture/reduction provenance has been added for external/multi-stream lenses, but target-layer and dataset provenance remain incomplete in many artifacts.

## Custom target layers

Decoding \(J_{l\to T}h_l\) directly with final norm/unembedding—without running omitted blocks after target \(T\)—matches the reference repository’s custom-target semantics. Neuronpedia’s DeepSeek target-layer handling is therefore numerically coherent.

The remaining problem is provenance and interpretation, not necessarily the readout calculation.

## Converter edge case

When an external artifact lacks target provenance, `convert-external-lens.py` defaults the target to the last stored source row:

`utils/neuronpedia-utils/neuronpedia_utils/jlens/convert-external-lens.py:204`

For a normal upstream artifact, stored source rows end at `target-1`, so that fallback is generally wrong. The identity-row validation should reject most such artifacts loudly rather than silently corrupting them, but the fallback should still be corrected.

---

# 13. Documentation And Motivation Assessment

## Clearly motivated and documented

The strongest documentation is internal:

- Why final normalization and model-specific logit transforms matter.
- Why transport uses the served model dtype.
- Why softmax is computed in FP32.
- Why the lens is placed near the residuals.
- Why BOS is skipped under local norm scaling.
- Why interventions must share the lens’s residual definition.
- Why multi-stream provenance must be explicit.
- Why write hooks run before capture hooks.
- How prefill-only differs from generated-token steering.

These implementation comments are unusually thorough.

## Principled but insufficiently exposed

Several choices are reasonable but not adequately visible to users:

- Per-position rather than layer-mean steering scale.
- Hard saturation at one residual norm.
- Fresh UI defaulting to generated-token intervention.
- Swap’s broad default layer range.
- Early fit-position exclusion.
- BF16/FP16 precision.
- Word-filtered versus true rankings.
- Exact-token distinctions such as `" ants"` versus `"ants"`.

## Not clearly justified

I found no documented scientific rationale for replacing the formal pseudoinverse coordinate swap with the projection-transfer operation.

It may have been chosen for simplicity, speed, or stability, but those are inferences rather than stated justification.

## Stale or misleading documentation

- The vendored package is described as “copied verbatim” at `utils/neuronpedia-utils/neuronpedia_utils/jlens/README.md:16`, but it contains Neuronpedia changes and now lags current upstream functionality.
- Current upstream has stronger layer validation, richer checkpoint metadata, `from_pretrained`, multi-position application, and configurable save dtype.
- The fit scripts contain old `anthropics/jlens` URLs even though the repository is `anthropics/jacobian-lens`.
- The Hugging Face card says all lenses were trained using Anthropic’s library, though most use an adapted vendored copy and several are externally converted.
- The inference schema says empty `steer_layers` means readout layers, but the endpoint treats an empty list as no intervention at `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:2370`.

---

# 14. Guided Tour And Scientific Claims

The tour is especially important because it teaches users what the results mean.

It says:

- The occurrence histogram is the model’s J-space.
- Its highest-count item is the model’s most frequent thought.
- Swapping spiders for ants verifies that the model uses J-space as a mental workspace.
- The resulting six-leg answer allows that conclusion.

See `apps/webapp/app/[modelId]/jlens/jlens-tour.tsx:114`.

There are three issues:

1. The histogram is not the formal J-space decomposition.
2. The swap is not the paper’s formal coordinate swap.
3. A single causal response change does not establish the paper’s five global-workspace properties.

Additionally, the guided spiders-to-ants result is not run live. It loads a committed fixture and waits two seconds to resemble a streaming intervention:

`apps/webapp/components/jlens/jlens-chat.tsx:671`

This is documented for maintainers at `utils/jlens-share-spans/README.md:79`, but not made clear in the tour itself.

A deterministic fixture is entirely reasonable for a tutorial. Calling it a live “verification” is not.

---

# 15. What Neuronpedia Results Do And Do Not Establish

A Neuronpedia J-lens readout supports:

- “This activation has a high average-Jacobian readout score for this token.”
- “This token repeatedly appears among the configured filtered top-k readouts.”
- “The J-lens and logit-lens differ at these layers.”
- “An intervention along this derived direction causally changed the continuation.”

It does not, without further controls, establish:

- That the token is literally a discrete thought.
- That it is the highest formal J-space coordinate.
- That the model’s activation is well reconstructed by that vector.
- That the token belongs to a \(k\)-sparse nonnegative decomposition.
- That the observed behavior depends on a global workspace.
- That the paper’s coordinate-swap result was reproduced.
- That the relevant concept is absent when it fails to enter the filtered top eight.
- That an intervention failure disproves causal relevance.
- That J-space monitoring is sufficient for alignment monitoring.

The paper itself is careful on the last point: it calls J-lens useful for auditing but explicitly rejects the stronger claim that monitoring it is sufficient.

It also emphasizes:

- Single-token vocabulary limitations.
- Lack of relational binding—a readout such as `spider`, `legs`, `eight` does not say how they are related.
- Nonuniqueness from overcompleteness.
- The J-space component often explaining less than 10% of activation variance.
- Uncertainty over whether early-layer failure is model structure or lens degeneracy.
- No claim that the result establishes subjective or phenomenal consciousness.

Those limitations should accompany Neuronpedia’s “hidden mental workspace” framing more prominently.

---

# Recommended Corrections

1. **Implement the paper’s pseudoinverse swap** and preserve the current operation under a name such as “projection transfer.”
2. **Separate “J-lens readouts” from “formal J-space” in UI copy**, especially in the tutorial and blog.
3. **Expose arbitrary token-position masks** and include presets for all-prompt, user-turn, assistant-prefill, stimulus span, prefill-only, and prefill-plus-decode.
4. **Fix strength semantics** so the slider remains linear, or visibly show saturation and limit the range to ±1.
5. **Add a paper-reproduction preset** using a workspace-only layer band, prompt-only clamping, unfiltered top-10, and formal swap.
6. **Record full artifact provenance**: model/tokenizer/dataset revisions, code commit, skip count, target layer, capture definition, and prompt hashes.
7. **Distinguish occurrence counts from probability/rank aggregation** and offer formal sparse decomposition if the interface continues to call the panel “J-Space.”
8. **Disclose that the guided-tour intervention is a fixture**, not live inference.
9. **Add controlled validation suites** for swap parity, layer-band sensitivity, position masking, random-direction controls, and cross-backend numerical agreement.

Overall, Neuronpedia is a faithful and sophisticated **J-lens reader**, a useful but modified **J-lens steering environment**, and not yet a faithful implementation of the paper’s full **J-space methodology or coordinate interventions**.
