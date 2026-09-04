I reviewed `~/code/qwen-llm-lens-integration` at `integration/lens`, commit `4b59800e`, against:

- The full J-lens/global-workspace paper.
- `anthropics/jacobian-lens` current `main`.
- The [R-lens post](https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers).
- The published [`camilablank/workspace-lenses`](https://huggingface.co/camilablank/workspace-lenses) artifacts.
- The local `feature/rlens` reference checkout.
- Neuronpedia’s deployed application-time variants from the preceding reviews.

I excluded native fitting as requested. I only examined fitting metadata where necessary to establish what an imported matrix means and whether J/R assets are comparable.

**Overall Verdict**

The Qwen lens capability is:

- **A highly faithful J/R transport and readout implementation.**
- **More mathematically faithful than Neuronpedia for paper-style coordinate swaps.**
- **Much more explicit and experimentally controllable about layers, token positions, prefill/decode phases, normalization, operation order, and sampling.**
- **Excellent as a low-level mechanistic research instrument.**
- **Not yet a complete behavioral-science or paper-replication framework.**
- **Limited by BF16-lens-to-quantized-GGUF transfer, incomplete Qwen run provenance, and lack of packaged evals, repeated-trial orchestration, controls, and statistics.**

Its intervention taxonomy is particularly good. It correctly separates:

- `coordinate_swap`: the paper’s two-coordinate exchange.
- `source_to_target`: Neuronpedia’s one-coordinate projection transfer.
- `projection_ablate`: projection removal.
- `fixed_add`: unscaled additive intervention.
- `residual_l2_fraction`: Neuronpedia-like local-norm steering without Neuronpedia’s cap.

That naming prevents the central semantic conflation found in Neuronpedia.

The scientifically strongest path is the imported **Qwen3.6-27B matched J/R pair**, run on one fixed GGUF under serial execution with exact token IDs and externally orchestrated paired trials. Muse J/R should not be compared as a method contrast because their fitting recipes, checkpoints, targets, and corpora differ.

---

# 1. Imported Asset Reality

Although the broader Hugging Face repository contains eight J/R pairs, this code supports five exact imported full-transport profiles.

## Qwen3.6-27B J/R

This is the sole controlled imported pair:

- J and R from the same HF repository revision.
- Same model family.
- Same target layer 62.
- Same source layers 0–62.
- Same 25 Pile documents.
- Same maximum length 128.
- Same first-four-position exclusion.
- Same FP16 storage.
- Different only in standard Jacobian versus RelP transport.

Profiles are pinned at `crates/qwen-cli/src/full_lens.rs:104`.

This pair is well suited for studying R-versus-J differences on a common deployed model.

Caveats:

- The fitted checkpoint revision was not recorded in the published artifact.
- The matrices were fitted against BF16 execution but applied here to a GGUF model.
- The deployed residuals, output norm, and LM head can differ because of quantization.
- The repository describes Qwen3.6 Q4 transfer as locally bounded, not broadly equivalent to the BF16 source.

## Qwen3.8-27B J

- Neuronpedia-produced J transport.
- Target layer 63.
- Sources 0–62.
- 1,000 WikiText prompts.
- First 16 positions excluded.
- No imported R counterpart.

This is a Neuronpedia/reference-code-style asset, not an R-lens paper pair.

Its profile is at `crates/qwen-cli/src/full_lens.rs:85`.

## Muse Glimmer J

- Neuronpedia-produced J transport.
- 900 WikiText prompts.
- First 16 positions excluded.
- Target layer 51.
- Sources 0–50.
- The declared convergence criterion was not reached.
- Model/checkpoint provenance comes primarily from the model card rather than embedded artifact metadata.

See `crates/qwen-cli/src/muse_published_full_lens_artifact.rs:235`.

## Muse Glimmer R

- R transport using the R-lens dense rules.
- 25 fixed Pile documents.
- First four positions excluded.
- Target layer 50.
- Sources 0–50, including exact identity at layer 50.
- Strong embedded corpus, model, tokenizer, arithmetic, and estimator provenance.

See `crates/qwen-cli/src/muse_published_full_lens_artifact.rs:301`.

## Muse J/R are not matched

They differ in:

- Fitted checkpoint revision.
- Corpus.
- Prompt count.
- Position exclusion.
- Target layer.
- Convergence.
- Provenance quality.

The repository correctly warns against treating Muse J/R differences as a method effect at `docs/LENS-MVP.md:143`.

## Unsupported published pairs

The importer does not currently accept the HF pairs for:

- Qwen3.5 4B, 9B, 27B, or 122B-A10B.
- Qwen3.6 35B-A3B.
- Gemma 3 27B.
- DeepSeek-V4-Flash.

Import dispatch is exact source-profile matching, not shape-based acceptance. That is safer, but means the application cannot reproduce the R-lens paper’s multi-model scale trend.

---

# 2. R-Lens Semantics

R-lens changes the transport matrix, not the application rule.

At application time, both lenses compute:

\[
y_{\ell,p}=T_\ell h_{\ell,p},
\]

then:

\[
\operatorname{logits}_{\ell,p}
=
W_U\,N(y_{\ell,p}).
\]

For J-lens, \(T_\ell=J_\ell\), an average ordinary Jacobian.

For R-lens, \(T_\ell=R_\ell\), an average relevance-propagation coefficient produced by the modified backward graph.

The R-lens post’s dense rules are:

- Detach the RMSNorm denominator.
- Use the identity rule for SiLU/GELU.
- Split SwiGLU relevance evenly across gate/up branches.
- Leave linear layers, attention, and Q/K norms as ordinary Jacobians.

The local reference implements those rules at:

- `/Users/tito/code/jacobian-lens-rlens/jlens/muse_relp.py:44`
- `/Users/tito/code/jacobian-lens-rlens/jlens/muse_relp.py:61`

The Qwen application correctly does not add special R-only readout or intervention logic. J and R matrices go through the same transport, normalization, output head, scopes, and intervention kernels. That is the right design.

---

# 3. Full J/R Readout

For a source residual \(x=h_{\ell,p}\), an imported matrix \(T_\ell\), deployed output RMSNorm gain \(\gamma\), and deployed head \(W\), Qwen computes:

\[
y=T_\ell x,
\]

\[
r(y)=\sqrt{\operatorname{mean}(y^2)+\epsilon},
\]

\[
z=W\left(\gamma\odot\frac{y}{r(y)}\right).
\]

Implementation:

- F16 transport matrix.
- F32 source residual.
- F32 transported residual.
- F32 RMSNorm reduction.
- Deployed, potentially quantized GGUF LM head.
- Full-vocabulary logits computed on Metal.
- No softmax.

See `crates/qwen-llm/src/workspace_lens.rs:1870`.

This is faithful to the J/R application equation. Omitting softmax does not change token ranking.

## Muse output tail

Muse additionally applies:

- Output multiplier.
- Final logit softcap.

Conceptually:

\[
z_t
=
C\tanh\left(
\frac{
a\,w_t^\top
\left(\gamma\odot y/r(y)\right)
}{C}
\right).
\]

See `docs/LENS-RUN.md:270`.

The positive multiplier and monotonic softcap preserve ranking while changing score magnitude.

## Target-layer handling

For Qwen3.6 J/R:

- Target is block 62.
- Layer 62 transport is exact identity.
- The readout applies final norm/head directly to block-62 coordinates.
- Block 63 is not executed by the lens readout.

This matches the published artifact semantics and the R-lens reference behavior.

It is important not to confuse:

- The lens’s target-layer logits.
- The model’s actual output after all remaining blocks.

`trace-full` does not automatically append the model’s actual final output row as Neuronpedia does.

## Captured coordinate

The source coordinate is precisely:

- Zero-based transformer block index.
- Post-block residual after attention and FFN/MoE residual additions.
- Token position \(p\), predicting position \(p+1\).

See `crates/qwen-llm/src/metal_forward.rs:10825`.

That is a clean and paper-compatible residual-stream convention.

## Precision

The transport is stored in FP16 and applied to F32 residuals. The model residual and output head derive from a quantized GGUF execution.

This means a Qwen run is not literally:

> Published lens applied to the checkpoint on which it was fitted.

It is:

> Published transport matrix applied to a geometry-compatible quantized deployment, using that deployment’s residuals, norm, and output head.

The explicit `allow_unvalidated_transfer` gate is good disclosure, but an acknowledgement is not validation.

---

# 4. Full Trace Semantics

`trace-full` is the strongest facility for reproducing R-lens pass@10-style readout analysis.

It:

- Captures all selected prompt positions.
- Captures arbitrary selected source layers.
- Applies the full transport separately at each layer/position.
- Ranks over the full vocabulary.
- Emits up to top 25 for Qwen and top 16 for Muse.
- Records exact token IDs and zero-based ranks.
- Can include selected transported target-coordinate vectors.
- Records occurrence counts as top-k list membership.

See `docs/LENS-RUN.md:237`.

This can directly represent the R-lens paper’s core readout outcome:

> At the annotated position and layer, does the expected intermediate appear in top 10?

## Good epistemic labeling

The trace output explicitly states:

- Scores are logits, not probabilities.
- Softmax was not applied.
- Candidate universe is the full model vocabulary.
- Missing means “outside captured top-k,” not zero or absent.
- One occurrence is one token ID in one returned top-k cell.

See `docs/LENS-RUN.md:312`.

This is much better than Neuronpedia’s casual “thought frequency” framing.

## Limitations

Although full-vocabulary logits are computed, only top-k results are retained. Therefore one cannot recover:

- Softmax probabilities.
- Log partition.
- Rank beyond the captured top-k.
- A target’s exact poor rank without rerunning at a larger \(k\).
- Full-distribution KL or calibration metrics.

The Qwen packed trace uses two MPS top-16 selections to form up to 25 returned candidates. The ordinary non-tie case is sound. A pathological tie involving more than 32 tokens at the cutoff does not have a fully demonstrated global token-ID tie-break guarantee.

## Passive packed prefill

Qwen `trace-full` uses packed prefill. Packed reductions are acknowledged as numerically different from serial single-token execution.

That is probably harmless for qualitative top-k work but should be treated as a distinct numerical execution contract. There is no serial `trace-full` control exposed in the same way as `run`.

---

# 5. Live Selected-Token Readout

Behavioral `run` does not repeatedly calculate the full vocabulary. It projects only selected token covectors through the imported transport.

For token \(t\), Qwen constructs:

\[
c_t=\gamma\odot w_t,
\]

then:

\[
d_{\ell,t}=T_\ell^\top c_t.
\]

The live score is:

\[
s_{\ell,p,t}
=
h_{\ell,p}^\top d_{\ell,t}
=
c_t^\top T_\ell h_{\ell,p}.
\]

See:

- `crates/qwen-llm/src/workspace_lens.rs:1174`
- `crates/qwen-llm/src/workspace_lens.rs:1285`

It omits:

- The activation-dependent RMS denominator.
- Softmax.
- Muse final softcap.

## What this preserves

For a fixed layer-position cell, the RMS denominator is shared and positive, so the numerator preserves ranking among the selected tokens.

## What it does not preserve

Live scores should not be interpreted as:

- Probabilities.
- Full-vocabulary ranks.
- Calibrated logits.
- Quantities directly comparable across layers or positions.

Residual norm and transported norm affect score magnitude across cells.

The output correctly records:

- `score_kind`.
- `candidate_universe`.
- Method and target layer.
- Only the selected artifact rows.

See `crates/qwen-cli/src/lens_run.rs:1354`.

## Gamma folding

Neuronpedia constructs \(T_\ell^\top w_t\) using a bare LM-head row. Qwen constructs:

\[
T_\ell^\top\operatorname{diag}(\gamma)w_t.
\]

This is not generally a scalar rescaling; coordinatewise gamma can rotate the source direction.

Qwen’s interpretation is defensible and arguably more faithful to the actual readout numerator because:

\[
W_U N(y)
=
W_U\operatorname{diag}(\gamma)y/r(y).
\]

The J-lens paper’s notation is ambiguous about whether learned norm scale is considered part of `norm` or folded into the effective unembedding. Therefore:

- Qwen is faithful to the deployed numerator.
- It is not exactly Neuronpedia-compatible.
- Intervention comparisons between the two systems should use direction cosine, not just the same token name.

---

# 6. Intervention Placement

Every intervention is applied:

1. After the selected transformer block completes.
2. Before capture/readout at that site.
3. Before the modified residual enters subsequent blocks.

See `crates/qwen-llm/src/metal_forward.rs:10869`.

Therefore live readouts observe the post-intervention residual.

Multiple operations at one site execute in plan-file order and are generally noncommutative.

This is scientifically good: order is explicit and recorded.

## Important causal boundary

Because intervention occurs after the block:

- It cannot alter that block’s already-computed attention routing.
- It cannot change that block’s current-token KV entry.
- It cannot change that block’s already-computed MoE route.
- It can change all later blocks and their state.
- It can affect future generation through those later representations.

This is the expected semantics of a post-block residual intervention, but should be stated when making claims about where a computation occurred.

---

# 7. Configurable Intervention Modes

## `fixed_add`

\[
h'=h+c\,v.
\]

This is the paper’s generic additive intervention.

With `as_stored`:

- Use the imported row’s native scale.
- Closest to \(h\leftarrow h+\alpha v_t\).

With `unit_l2`:

- Normalize direction first.
- Coefficient has residual-coordinate units.

This is highly faithful and cleanly named.

## `residual_l2_fraction`

\[
h'=h+c\,\|h\|_2\,\hat v.
\]

It requires a unit direction.

This is the closest option to Neuronpedia steering, but differs because it has no cap.

Neuronpedia uses:

\[
q=c\,\|h\|D,
\]

\[
h'=h+q\min\left(1,\frac{\|h\|}{\|q\|}\right).
\]

For one unit direction and \(|c|\le1\), the operations match, subject to direction and BOS differences.

For \(|c|>1\):

- Neuronpedia saturates.
- Qwen continues scaling linearly.

This is a useful distinction. Qwen’s version is better for dose-response studies because the coefficient remains linear, but it cannot reproduce the saturated Neuronpedia mode exactly.

There is no finite coefficient bound beyond “finite and nonzero,” and no update-norm cap. Extreme values can overflow downstream computation.

## `projection_ablate`

\[
h'=h-c(h^\top\hat v)\hat v.
\]

At \(c=1\), this is exact one-vector projection removal.

At other values:

- \(0<c<1\): partial ablation.
- \(c>1\): overshoot.
- \(c<0\): amplification.

The action name remains reasonable, but analyses should report the coefficient rather than saying simply “ablated.”

This faithfully implements the obvious interpretation of both J- and R-direction ablation. However, the R-lens post does not publish enough detail about its exact normalization/projection convention to establish bit-for-bit protocol identity.

## `source_to_target`

\[
h'
=
h+c(h^\top s)(t-s).
\]

At \(c=1\), with unit source and target, this is exactly Neuronpedia’s deployed “swap” operator:

\[
h'
=
h-(h^\top s)s+(h^\top s)t.
\]

Qwen’s name is much better. It accurately describes a directed projection transfer rather than a two-coordinate exchange.

Like Neuronpedia’s operation, it:

- Reads only the source coefficient.
- Leaves the pre-existing target coefficient in place.
- Does not jointly solve source/target coordinates.
- Does not generally zero the source readout when source and target are nonorthogonal.

## `coordinate_swap`

For unit source and target, define:

\[
u=\frac{s-t}{\|s-t\|}.
\]

Qwen applies:

\[
h'=h-2c(h^\top u)u.
\]

At \(c=1\), this is a Householder reflection that maps:

\[
s\mapsto t,\qquad t\mapsto s.
\]

For linearly independent unit directions it is algebraically equivalent to:

\[
h'
=
h+cV(P-I)V^\dagger h,
\]

where:

\[
V=[s\;t]
\]

and \(P\) exchanges the two coordinates.

The lowering is implemented at `crates/qwen-cli/src/lens_run.rs:4024`; the reflection direction is constructed at `crates/qwen-cli/src/lens_run.rs:3416`.

This is the paper-faithful swap operation Neuronpedia lacks.

Qualifications:

- Qwen swaps coordinates of individually unit-normalized directions.
- If the paper intended raw, differently scaled vectors in \(V\), the coordinate systems differ.
- Near-collinear pairs with \(1-\cos^2(s,t)\le10^{-10}\) are rejected.
- \(c=2\) implements the paper’s “double-strength” displacement, not a second ordinary swap.

This is one of the strongest parts of the implementation.

---

# 8. Multiple Tokens And Composite Directions

Each configured direction references one token or template row. There is no plan-level sum/mean/composite direction node.

Consequences:

## Exact through sequential operations

Multiple `fixed_add` operations exactly reproduce:

\[
h'=h+\alpha\sum_i v_i,
\]

because each update is state-independent and additions commute.

## Not exact through sequential operations

Multiple `residual_l2_fraction` operations are not equivalent to one summed Neuronpedia steer because every later operation measures the norm of an already-modified residual.

Likewise, sequential projection ablations are not:

- Projection out of the normalized vector sum.
- Joint projection out of the full nonorthogonal span.

They are ordered sequential projections.

The system cannot directly reproduce:

- Neuronpedia’s multi-token summed steering with a shared cap.
- Neuronpedia’s normalized-sum ablation.
- A swap whose source or target is a token-vector aggregate.
- The J-lens paper’s dynamic top-k J-space ablation.

A composite-direction plan primitive would substantially improve this.

---

# 9. Layers, Positions, Prefill, And Decode

The scope grammar is significantly better than Neuronpedia’s.

Every operation and readout selects:

- Layers.
- Prefill positions.
- Decode positions.

No phase is implicitly enabled.

## Numeric selectors

Supported selectors include:

- `all`
- Explicit sorted unique values.
- Inclusive ranges.

This can reproduce arbitrary sparse and contiguous intervention masks once token positions are known.

## Semantic selectors

Plan v2 can bind renderer-authored boundaries such as:

- Final user-content token.
- Assistant generation marker.
- Tool-result terminator.
- Thinking-channel boundary.

Bindings fail closed on ambiguity or BPE-inexact boundaries.

See `docs/LENS-RUN.md:467`.

This is excellent for model-behavior research because it avoids searching decoded delimiter-like text.

However, `rendered_spans` currently resolves a selector only to:

- The first token of a span, or
- The final token of a span.

It does not mean “all tokens in this semantic span.”

Therefore reproducing the paper’s:

- Every token in the user turn.
- Every token in the stimulus span.

still requires an externally resolved numeric range.

The name `rendered_spans` can misleadingly suggest full-span selection.

## BOS and special tokens

There is no automatic BOS exemption.

`prefill: all` includes:

- BOS.
- Chat markers.
- Tool markers.
- Thinking markers.
- Every other resolved prompt token.

This is better than an invisible special case for general research, but exact Neuronpedia reproduction requires explicitly excluding every BOS position.

A token-ID predicate would make that easier and more robust.

## Prefill-only

Set `prefill`; omit `decode`.

This is exact and explicit.

## Prefill plus decode

Set both.

Decode index 0 means:

- The first sampled token is fed back.
- The intervention is applied while processing it.
- It affects logits for the second generated token.

To affect the first sampled token, intervene at the final prefill position.

A sampled stop token or final permitted sample is reported but never fed back, so no intervention runs on it.

See `crates/qwen-cli/src/lens_run.rs:1934`.

This timing is very clearly documented and implemented.

---

# 10. Faithfulness To The J-Lens Paper

## Strongly faithful

- Average transport application.
- Model output normalization and unembedding.
- Post-block residual coordinates.
- Full-vocabulary token ranking.
- Fixed additive steering.
- One-vector projection ablation.
- Pseudoinverse-equivalent coordinate exchange.
- Arbitrary layer and position targeting.
- Double-strength swap coefficient.
- Continuation through the real downstream model after interventions.

## Conditionally faithful

### Paper experiment norm scaling

The companion experiment README uses:

\[
h'
=
h+\alpha\,m_\ell\,\hat v,
\]

where \(m_\ell\) is a layer-wide mean residual norm.

There is no built-in layer-mean-norm mode.

It can be reproduced manually by:

- Computing \(m_\ell\) externally.
- Creating one `fixed_add/unit_l2` operation per layer.
- Setting each coefficient to \(\alpha m_\ell\).

`residual_l2_fraction` is not the same because it uses each current position’s norm.

### Position masks

Paper user-turn and stimulus-span masks can be represented numerically, but are not supplied as reusable experiment definitions.

### Direction convention

Gamma folding is mathematically defensible but not identical to Neuronpedia’s bare-unembedding convention, and the paper’s notation is not explicit enough to eliminate that ambiguity.

## Not implemented

- Formal sparse nonnegative J-space decomposition.
- Gradient pursuit.
- J-space occupancy.
- Fraction of variance explained.
- J-space/non-J-space concept decomposition.
- Dynamic top-k J-space ablation.
- Random-direction and matched-norm controls.
- Paper experiment datasets and scoring.
- Workspace band calibration per model.
- Actual-output row appended beside lens rows.
- Paper-wide global-workspace evidence suite.

Therefore it is a strong J-lens instrument, not a complete J-space implementation.

---

# 11. Faithfulness To The R-Lens Post

## Readout fidelity: High

For the Qwen3.6 matched pair, application-time behavior is highly faithful:

- Exact pinned published matrices.
- Same model geometry.
- Same target and source layers.
- Same J/R pair.
- Same output operation for both.
- Full-vocabulary top-10-capable traces.
- Unfiltered early-layer readouts.

This directly supports qualitative and pass@10 analysis.

## Scale-trend fidelity: Unavailable

The R-lens post’s primary quantitative claim is that the R advantage increases with model size.

Only one supported imported pair is method-matched here. The application cannot test the reported multi-model trend.

## Early-layer examples: Supported

The application can reproduce claims such as:

- Intermediate appears earlier for R.
- R has fewer incoherent early tokens.
- R finds a token that J never puts in top 10.

`trace-full` is well suited to this.

## Ablation fidelity: Moderate

The required primitive exists:

- Unit R direction.
- Projection ablation.
- Penultimate prompt position.
- First half or all layers.
- Real downstream generation.

But the R-lens post does not disclose:

- Exact direction construction.
- Whether RMS gamma was folded.
- Exact vector normalization.
- Exact projection equation.
- Sampling settings.
- Layer-half boundaries.
- Autorater prompt/revision.
- Seed schedule.
- Error-bar computation.

Qwen implements the most natural orthogonal projection convention, but cannot claim exact protocol replication.

## R steering and swaps: Principled extensions, not paper results

The R-lens post establishes stronger readout and ablation effects. It does not establish:

- Positive R-direction steering.
- R-coordinate swapping.
- R source-to-target transfer.
- R persistent decode clamping.

Applying the same operators to R directions is mathematically natural and scientifically interesting, but these should be described as new experiments, not replications of the R-lens post.

## Other missing R analyses

No built-in support exists for:

- Probe-defined earliest-layer bounds.
- CKA.
- MLP gain.
- Trash-token metrics.
- Per-category pass@10.
- First-half/all-layer aggregate calculations.
- Paired confidence intervals.
- The frozen prompt/evaluation set.

---

# 12. Neuronpedia Compatibility

## Exact configurable equivalent

Conditioned on identical unit directions:

- Neuronpedia “swap” = Qwen `source_to_target` at coefficient 1.
- One-token Neuronpedia ablation = Qwen `projection_ablate` at coefficient 1.
- Neuronpedia prefill-only = Qwen prefill scope without decode.
- Neuronpedia prefill+decode = Qwen both phases.
- Neuronpedia all-layer or selected-layer behavior = explicit Qwen layer selectors.

## Conditional equivalent

Neuronpedia additive steering matches Qwen `residual_l2_fraction` only when:

- One unit direction is used.
- The Neuronpedia cap is inactive.
- Usually \(|c|\le1\).
- BOS selection is made equivalent.
- The actual direction bytes are equivalent.

## Not exactly reproducible

- Neuronpedia cap at one residual norm.
- Automatic exclusion of every BOS token.
- Bare-\(W_U\) direction rather than gamma-folded covector.
- Multi-token normalized sums.
- Multi-token shared cap.
- Multi-token summed ablation.
- Fresh UI defaults chosen automatically from occurrence counts.

## Superior to Neuronpedia

Qwen offers:

- Exact paper swap.
- Neuronpedia transfer as a distinct name.
- Arbitrary position masks.
- Explicit prefill/decode phases.
- Explicit direction normalization.
- Explicit coefficients for every operator.
- Ordered composition.
- Zero-coefficient paired controls.
- Exact renderer-bound token positions.
- No hidden word filtering.
- Full-vocabulary rank semantics.
- Better top-k censoring language.
- Literal token-ID control.

---

# 13. Quality As A Research Instrument

## Excellent qualities

### Explicit experiment plans

Plans preserve:

- Exact authored operation definitions.
- Resolved numeric layers and positions.
- Direction normalization.
- Coefficients.
- Operation order.
- Readout scopes.
- Input rendering.

This is substantially more auditable than an interactive UI.

### Exact prompt rendering

The lens lane reuses the actual model request renderers rather than maintaining a separate approximation.

It supports:

- Qwen thinking/no-thinking modes.
- Qwen3.8 effort tiers.
- Open Responses tools and reasoning history.
- Muse ATEM tools and channels.
- Raw prompts.
- Literal token IDs.

This is outstanding for behavioral experiments involving system/developer messages, tool calls, thinking channels, or forged structures.

### Fail-closed semantic boundaries

Semantic selectors reject:

- Ambiguous spans.
- Inexact BPE boundaries.
- Missing markers.
- Multiple matches without explicit occurrence selection.

This prevents a large class of silent position mistakes.

### Ordered operations

Noncommutative intervention sequences are preserved and recorded.

### Coefficient sweeps

Ordinary Qwen sweeps:

- Keep one model resident.
- Give every arm a fresh sequence.
- Give every arm the same sampler seed.
- Preserve coefficient order and duplicates.
- Permit zero and signed-zero controls.
- Disable the selected operation kernel at zero.
- Preserve the same scalar event topology.
- Hash immutable child artifacts.

See `docs/LENS-RUN.md:65`.

This is a very good paired-dose primitive.

### Artifact import security

Published `.pt` import:

- Pins exact source SHA.
- Pins exact ZIP inventory.
- Pins opaque pickle digest.
- Does not execute pickle.
- Validates FP16 finiteness.
- Validates identity anchors.
- Emits canonical local artifacts.

That is unusually careful.

### Honest transfer labeling

The documentation repeatedly distinguishes:

- Implementation qualification.
- Artifact integrity.
- Quantized-model transfer.
- Behavioral equivalence.

That epistemic discipline is strong.

---

# 14. Scientific Limitations

## Qwen `run` provenance is too weak

Ordinary Qwen run artifacts record:

- Model path.
- Plan path and digest.
- Exact prompt IDs.
- Rendering.
- Sampler.
- Operations and readouts.

But they do not bind:

- Model content digest.
- Tokenizer identity.
- Imported lens manifest digest.
- Selected matrix digests.
- Payload digest verified at execution.
- Build commit/source state.

`RunOutput` is at `crates/qwen-cli/src/lens_run.rs:1248`.

Muse published runs do record those stronger bindings. Qwen should reach parity with Muse.

This is the most important reproducibility defect for the stated imported-asset usage.

## Qwen `run` and `trace-full` do not verify matrix bytes strongly enough

- `read-full` hashes the entire payload while reading.
- `run` verifies requested matrices are finite but not that they match the artifact digest.
- `trace-full` checks file length and canonical manifest but does not rescan the payload.

A same-length post-import mutation could affect Qwen run/trace results without detection.

Muse avoids this through per-matrix digests.

## BF16-to-GGUF transfer

All imported lenses remain transfer experiments.

Potential differences include:

- Residual geometry.
- Norm gain.
- LM-head rows.
- Quantized matmul error.
- Accumulated state differences through the prompt.
- Changed J/R ranking near ties.
- Changed intervention direction after gamma/head projection.

Relative J/R comparisons on the same GGUF are still useful, but they should be called:

> Comparative application of a matched published J/R pair to a shared quantized deployment.

Not:

> Exact reproduction of the BF16 R-lens experiment.

## Default `auto` execution

Dense `run` defaults to packing passive prompt spans when available. Packed reductions are not bit-exact with the serial reference, and the choice can change with memory admission.

For scientific work, use:

```text
--prefill-execution serial
```

The documentation says this clearly at `docs/LENS-RUN.md:37`, but the scientific control should arguably be the default in `qwen-lens`.

## No exact replay command

Artifacts contain substantial replay information, but there is no command that:

- Loads a prior run.
- Verifies executable/model/tokenizer/lens identities.
- Re-executes exact token IDs and plan.
- Compares resulting states and outputs.

## No actual applied-dose measurements

Operation records contain site identity, but not:

- Pre-intervention residual norm.
- Direction norm before normalization.
- Applied update norm.
- Update/residual norm ratio.
- Source projection.
- Target projection.
- Pre/post readout score.
- Numerical clipping/overflow status.

These measurements are essential for comparing intervention strength across:

- Layers.
- Tokens.
- J versus R.
- Models.
- Normalization modes.
- Quantizations.

## No safety cap or post-intervention finite check

Any finite nonzero coefficient is accepted. Very large values can produce nonfinite residuals or logits.

A research tool should not silently cap by default, but it should offer:

- Optional named caps.
- Immediate finite checks.
- Applied-dose reporting.

## No end-to-end experiment harness

The tool is deliberately scoped as implementation infrastructure at `docs/LENS-MVP.md:7`.

It lacks:

- Prompt corpora.
- Trial manifests.
- Multiple-seed repetition.
- Arm randomization.
- Counterbalancing.
- Random-direction shams.
- Norm-matched controls.
- Off-target layer/position controls.
- Clean/J/R/logit-lens arm bundles.
- Confidence intervals.
- Bootstrap or permutation analysis.
- Grader prompts and revisions.
- Long-form/tabular result export.

That is acceptable for a primitive, but it limits claims that can be made directly from its output.

## Sweep order

Sweep arms execute sequentially in coefficient order.

Potential confounds:

- Thermal state.
- Memory pressure.
- GPU scheduling state.
- Caching and residency effects.

Fresh model sequence state prevents causal contamination, but arm order is not randomized.

## Top-k-only outputs

Readout artifacts are honest about censoring, but population analyses often need:

- Target score even when outside top-k.
- Full target rank.
- Log probability.
- KL divergence.
- Probability shifts across conditions.

A targeted-token query that always returns exact score/rank would help substantially.

---

# 15. Tests And Verification

I ran:

```text
cargo test -p qwen-cli --bin qwen-lens -- --nocapture
```

Result:

- 230 passed.
- 0 failed.
- 6 ignored.

I also ran:

```text
cargo test -p qwen-llm workspace_lens -- --nocapture
```

Result:

- 29 passed.
- 0 failed.
- 1 ignored.

And:

```text
cargo test -p qwen-llm post_block -- --nocapture
```

Result:

- 3 passed.
- 0 failed.

These tests strongly establish:

- Artifact schema validation.
- Import safety.
- Coordinate layouts.
- Layer/position scheduling.
- Exact action equations.
- Coordinate-swap algebra.
- Operation order.
- Top-k deterministic ordering.
- Semantic selector behavior.
- Sweep isolation and integrity.
- J/R matched profile metadata.
- F32 kernel agreement with CPU formulas.

They do not establish:

- BF16-to-GGUF equivalence.
- Real-model J/R paper outcomes.
- Full real-model readout path parity in normal CI.
- R-lens ablation effect sizes.
- Cross-machine reproducibility.
- Multi-model scale trends.

The real-model reusable full-readout parity test is ignored. Several other model-bound tests require local environment variables.

`docs/LENS-MVP.md:95` says 214 tests with four ignored; that is stale relative to the current 236-test CLI suite.

The worktree remained clean after testing.

---

# 16. Recommended Research Protocol

For a scientifically defensible Qwen3.6 J/R study:

1. Use the exact imported J/R pair from the same pinned HF revision.
2. Use one fixed GGUF content digest and quantization.
3. Force `--prefill-execution serial`.
4. Use literal token IDs or record exact rendered token IDs and spans.
5. Use identical target layers, positions, phases, normalization, and coefficients.
6. Prefer full traces for readout comparisons.
7. Record that logits use the deployed GGUF norm/head.
8. Treat target layer 62 as a lens target, not actual final model output.
9. Use `unit_l2` for causal J/R direction comparisons.
10. Record raw direction norm and J/R direction cosine.
11. Use zero controls with preserved topology.
12. Add random or orthogonal sham directions.
13. Randomize or counterbalance arm order externally.
14. Repeat multiple seeds and report item-level outcomes.
15. Keep clean, J, R, and logit-lens controls separate.
16. Do not compare Muse J versus Muse R as a method effect.
17. Describe positive R steering/swaps as novel experiments.
18. Report top-k censoring rather than treating missing targets as absent.

---

# 17. Prioritized Improvements

1. **Give ordinary Qwen runs Muse-grade execution binding:** model content, tokenizer, build, manifest, payload, and selected matrix digests.
2. **Verify selected Qwen matrices at use time** in both `run` and `trace-full`.
3. **Add an exact replay command** with dependency verification.
4. **Record intervention dose metrics** at every applied site.
5. **Add a target-token score/rank query** independent of top-k inclusion.
6. **Add composite direction definitions:** sum, weighted sum, mean, and joint span.
7. **Add a full semantic-span selector**, not only start/end edges.
8. **Add token-ID exclusion predicates** for BOS and special-token controls.
9. **Add named layer-mean steering and capped local-norm steering modes** if exact paper/Neuronpedia reproduction is a goal.
10. **Add paired experiment manifests** supporting J/R/logit/clean/sham arms, randomized order, repetitions, and statistics.
11. **Add the released R-lens evaluation annotations** and pass@10 analysis.
12. **Expose actual model-output logits beside lens rows** for controlled lens/output comparisons.
13. **Make serial execution the default for `qwen-lens` research commands.**
14. **Update stale docs and expose the lens CLI from the root README.**

In short: the implementation is an excellent application-time lens engine, especially in its exact coordinate conventions, explicit intervention taxonomy, and position/phase control. It is substantially more faithful and scientifically legible than Neuronpedia’s current intervention UI. Its remaining limitations are primarily experimental provenance and study orchestration rather than transport mathematics.
