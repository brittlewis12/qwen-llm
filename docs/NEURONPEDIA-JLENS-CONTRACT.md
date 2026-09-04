# Neuronpedia JLENS Contract

Companion paper comparison: `docs/NEURONPEDIA-JLENS-PAPER-FIDELITY.md`.

## Audit Provenance

- Audited checkout: `/Users/tito/code/neuronpedia`
- Source revision: `5580619db2509a34a06a384a519479938c24ee74`
- Source worktree state at verification: clean
- Pinned execution engine: `interp-engine==1.3.3`
- Audit date: 2026-09-01
- Scope: current checked-out UI, public API, inference service, tests,
  fitted-lens implementation, and pinned engine behavior

This is a source contract for the audited revision, not proof of every setting
or byte deployed during the historical exploratory probes. Historical results
remain bounded by their production commit, model and lens bytes, backend,
request fields, and whether they were live generations or exact-ID replays.

### Known Historical Boundaries

- The audited UI has top-N fixed at 8, while the historical exploratory
  workspace exposed top-25 slices. This establishes at least one current-versus-
  historical UI difference. It can change default intervention-layer selection
  because that selection counts occurrences inside the returned top-N.
- The audited source always appends the final model layer to current readouts.
  Earlier hosted/local comparison packets contained the 63 fitted Qwen3.6
  source layers. Final-layer inclusion should therefore be checked explicitly
  before comparing historical layer counts.
- Fresh current UI intervention configurations enable decode writes. Public API
  requests that omit the field and old shares that predate it use prefill only.
  Historical screenshots do not record the field.
- The guided-tour spiders-to-ants result is a committed fixture, not a live
  inference call. It is not independent evidence that the current hosted swap
  path produced the displayed arithmetic result.

## Local Parity Ledger

This table describes the Qwen3.6 application-period lane based on local commit
`4b59800e219d747ee74313e652acb49e50f7dfd9`, plus the uncommitted explicit
direction-covector work recorded in the experiment artifacts.

| Contract surface | Audited Neuronpedia | Local Lens lane | Status |
| --- | --- | --- | --- |
| Prompt rendering | Real model chat formatter; thinking disabled by public UI default | Exact Qwen3.6 messages renderer with explicit no-thinking mode | WRAP token IDs validated; each new historical prompt still records exact IDs |
| Readout transport | `final_norm(h @ J.T)` then real LM head and model logit transforms | Published full transport plus deployed output RMSNorm and full LM head | Strongly validated on n1000 J: 97.35% hosted/local top-1 and 100% hosted-top-8 recall in local top-25 |
| Capture convention | Artifact-declared point; Qwen common case is post-block residual | Post-block residual | Matched for current Qwen3.6 artifacts |
| Intervention target covector | Raw LM-head row `w`, transported as `w @ J` | Explicit `raw_lm_head`, deployed-logit-numerator, and raw-orthogonal direction bases | Raw product parity is implemented; omitted old plans retain gamma-folded behavior |
| Direction normalization | Each selected source token unit-L2 normalized | Unit-L2 normalization | Matched for one-token directions |
| Position scope | Every non-BOS position at selected layers | Explicit selectors; parity plans use all prefill/decode positions | Current chat probes contain no BOS and match; automatic BOS-ID exclusion still needs explicit local parity support |
| Layer scope | Explicit model layers; additive default derives from occurrence counts in the active position scope | Explicit model layers in every plan | Historical `error` popup implies layer 44 under current top-N tie rules, but production metadata remains absent |
| Phase scope | Prefill only or prefill plus decode | Independently selectable prefill and decode scopes | Matched when the historical field is known; otherwise an experimental factor |
| Additive operation | `h += clip(s,-1,1) * ||h|| * unit(d)` for one source token | `residual_l2_fraction`; current local kernel does not clamp above magnitude 1 | Operation family and tested `|s| <= 1` geometry match; exact cap and target-covector parity remain |
| Ablation | Orthogonal projection removal | Projection ablation | Formula and geometry match; target-covector mismatch remains |
| Swap | Directed source-to-target signed projection transfer | `source_to_target` | Formula matches; `coordinate_swap` is a different local operation and is not hosted parity |
| Source resolution | Exact decoded vocabulary string, then whitespace-trimmed fallback | Plans pin exact token IDs | More explicit locally; decoded string and token ID should both be recorded for reproduction |
| Generation | Greedy at temperature zero; no seed for stochastic runs | Deterministic greedy plus explicit seeded samplers | Greedy parity available; stochastic results are policy-specific |
| Readout candidates | Current UI top-8, word-filtered ranking, full-vocabulary probabilities | Canonical full-vocabulary top-25 logits, unfiltered | Intentional analysis difference; candidate policy must accompany every comparison |
| Precision/backend | Historical deployment not content-authenticated in screenshots | BF16 reference, Q8 primary exploration, Metal | Q8/BF16 readouts and tested behaviors transfer closely; backend residue remains bounded rather than assumed zero |
| Exact-ID replay | Reclassifies the entire fixed sequence as prefill | Literal token-ID input is explicit prefill | Token fidelity matches, but replay must not be used to infer original prompt/decode intervention phase |

## Experimental Consequences

1. Hosted-parity intervention plans use the explicit `raw_lm_head` target
   covector. The gamma-folded direction remains a named, self-consistent
   transported-logit condition rather than a superseded implementation.
2. Readout and intervention semantics must not be collapsed into one generic
   "lens row." Neuronpedia uses the real final norm for readout while omitting
   its derivative and gamma from the transported steering direction.
3. Historical replays should pin source token ID and exact decoded bytes,
   intervention layers, prefill/decode scope, prompt token IDs, candidate policy,
   model precision, and artifact identity in every packet.
4. Position-targeted local interventions are new experiments, not Neuronpedia
   reproductions. They may be more surgical, but must be contrasted with the
   hosted all-position operation.
5. Coefficients beyond magnitude 1 add no single-layer dose for one normalized
   source direction. Spend sweep budget inside `[-1,1]`; use layer count and
   phase as separate factors rather than treating `2.0` as a stronger dose.
6. Default-layer reconstruction depends on the historical top-N, filtering,
   and selected/hovered position scope. Prefer an explicitly photographed layer
   or report the reconstruction rule and ambiguity.
7. Saved-share or filter-change readouts can preserve token IDs while changing
   phase assignment. They are suitable token or passive-readout references, not
   automatic intervention-timeline oracles.
8. The arithmetic guided-tour fixture should be downgraded from live causal
   reproduction to a candidate effect requiring an independent local replay.
9. Exact general parity also requires BOS-ID exclusion and a one-residual-norm
   injection cap. Neither changes the current multi-token chat probes or their
   `|coefficient| <= 0.9` sweeps, but both matter for a faithful public surface.

## Paper Versus Product Boundary

The audited paper, current `anthropics/jacobian-lens` reference, and Neuronpedia
product share the average-Jacobian estimator and basic J-lens readout. They do
not expose the same full methodology.

### Transported-Logit Readout Versus Formal J-Space

Neuronpedia and the current local trace instrument rank vocabulary logits after
average-Jacobian transport and the model's deployed output decoding. The UI then
aggregates exact token-string occurrences from a filtered top-k list.

The paper's formal J-space is instead a sparse nonnegative approximation over
an overcomplete dictionary of transported token directions, estimated by
gradient pursuit. Top-k transported logits are not those decomposition
coefficients. Accordingly:

- Existing passive results characterize transported-logit readouts.
- Occurrence histograms are salience summaries, not workspace occupancy or
  variance-explained estimates.
- Formal J-space ablation and sparse decomposition remain a separate instrument
  lane that has not yet been implemented locally or in Neuronpedia.

### Three Distinct Intervention Contracts

1. Paper coordinate exchange jointly reads source and target coefficients with
   a pseudoinverse and exchanges them, optionally with strength `alpha`.
2. Neuronpedia projection transfer removes the signed source projection and
   adds it along the target direction. Local `source_to_target` matches this.
3. Local `coordinate_swap` is a Householder exchange. For independent equal-
   norm directions it equals the minimum-norm two-coordinate exchange; exact
   paper parity still depends on the paper experiment's direction-normalization
   convention.

Neuronpedia additive steering also differs from the paper protocol: it scales
by each current position's residual norm and caps one write at one residual
norm, while the companion paper protocol describes layer-mean norm scaling.
Single-direction ablation matches projection removal, but it is not the paper's
top-k formal-J-space ablation.

Position-targeted paper protocols are likewise outside Neuronpedia's product
contract. Neuronpedia writes every eligible position at selected layers;
local position selectors are a deliberate extension for mechanism experiments.

## Maintenance

- Update the source revision and parity ledger together when the audited
  Neuronpedia checkout changes.
- Preserve current-source facts separately from historical-production
  inferences.
- Record corrections as explicit qualifications rather than silently rewriting
  the evidentiary status of prior experiments.

## Audited Contract

I traced JLENS end-to-end through the UI, public API, inference service, tests, fitted-lens implementation, and the pinned `interp-engine==1.3.3`.

**Bottom Line**

JLENS interventions are:

- **Layer-selective:** the user chooses model layers where a write occurs.
- **Phase-selective:** writes affect prefill only, or prefill plus decode.
- **Operation-selective:** additive steering, ablation, or source-to-target projection swap.
- **Not token-position-selective:** selected or hovered transcript positions never restrict where an intervention is applied.

At each selected layer, the operation applies to every eligible sequence position in that forward pass. Only positions whose token ID exactly equals the tokenizer’s BOS ID are skipped. Token-position selection in the UI scopes analysis and influences the default layer choice, but is never sent to inference.

---

**Coordinate System**

Let:

- \(p\) be a zero-based position in the fully tokenized sequence.
- \(\ell\) be a zero-based decoder-block index.
- \(h_{\ell,p}\) be the activation at the lens artifact’s declared capture point.
- \(N\) be the model’s real final normalization.
- \(W\) be the model’s unembedding/LM-head matrix, with token row \(w_t\).
- \(\bar J_\ell\) be the fitted Jacobian transport for layer \(\ell\).

The common case is the output of decoder block \(\ell\), conventionally called `resid_post`.

A position’s readout is a prediction after processing the token at that position. Therefore:

- The final-layer readout at position \(p\) predicts token \(p+1\).
- It does not identify the token printed at position \(p\).
- A generated token’s readout comes from the forward pass in which that already-sampled token is fed back as input.

This convention is verified against the model’s true next-token prediction in `apps/inference/tests/integration/test_lens_prompt_parity.py:73`.

---

**Logit And Jacobian Readouts**

Using row-vector notation, the two readouts are:

```text
Logit Lens:
    z_l,p = unembed(final_norm(h_l,p))

Jacobian Lens:
    z^J_l,p = unembed(final_norm(h_l,p @ J_bar_l.T))
```

The service uses the model’s actual final norm, LM head, model-specific logit multiplier, and any final-logit softcap. It is not a bare multiplication by the embedding matrix.

At the final model layer:

- The activation is normally decoded directly, with no Jacobian transport.
- Consequently, Jacobian Lens and Logit Lens both show the actual next-token distribution of the current, possibly intervened forward pass.
- The final layer is always included in the readout, even if it was not requested or fitted.

Implementation: `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:1572`.

### What the fitted Jacobian means

The Jacobian is a corpus-average linear transport, not a Jacobian recomputed for the current prompt.

For each fitted layer:

\[
\bar J_\ell
=
\mathbb{E}_{\text{prompts},p}
\left[
\sum_{p' \ge p}
\frac{\partial h_{\text{target},p'}}
     {\partial h_{\ell,p}}
\right]
\]

The in-repository fitter:

- Usually targets the final model layer.
- Fits every earlier layer by default.
- Truncates fitting prompts to 128 tokens by default.
- Excludes the first 16 positions because of attention-sink behavior.
- Excludes the final position because it has no next-token target.
- Sums gradients over all causal target positions \(p' \ge p\), then averages over source positions and prompts.

Thus J-Space approximates how an earlier residual direction is expected, on average, to propagate into the late residual basis. It is not simply a same-position local Jacobian.

See `utils/neuronpedia-utils/neuronpedia_utils/jlens/jlens/fitting.py:3`.

---

**Readout Direction Used For Intervention**

A selected token string is first resolved to one vocabulary ID. Its unembedding row becomes the basis of the intervention.

For token \(t\):

```text
Logit-Lens source direction at every layer:
    d_l,t = w_t

Jacobian-Lens source direction at a fitted layer:
    d_l,t = w_t @ J_bar_l

Equivalent column-vector notation:
    d_l,t = J_bar_l.T @ w_t
```

At a layer without a fitted Jacobian, including the ordinary final layer, a Jacobian source falls back to \(w_t\).

This is the adjoint of the Jacobian readout transport:

```text
Readout transport:      h @ J_bar.T
Intervention direction: w @ J_bar
```

The direction does not include the derivative of final normalization. It is the transported unembedding direction, not the exact gradient of the final softmax probability.

Each source-token direction is normalized independently:

\[
u_{\ell,t} = d_{\ell,t}/\|d_{\ell,t}\|
\]

The normal UI sends one source token. The direct API permits multiple source tokens; in that case:

\[
d_\ell = \sum_t u_{\ell,t}
\]

The sum is not normalized again until an operation such as ablation or swap explicitly normalizes it.

Implementation: `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:661`.

---

**Token String Resolution**

A source or swap target must correspond to one independently decoded vocabulary entry.

Resolution proceeds as follows:

1. Look up the exact decoded string, preserving whitespace.
2. If absent, compare whitespace-trimmed strings.
3. The first trimmed-equivalent vocabulary form encountered wins.
4. If several IDs decode to that same selected string, use the lowest ID.
5. If nothing matches, return an error.

Consequences:

- `" cat"` and `"cat"` are distinct when an exact match exists.
- The fallback can silently map `"cat"` to `" cat"`.
- `"New York"` only works if one vocabulary token decodes to the entire string.
- It is never tokenized into multiple IDs and combined.
- The swap input’s visible `␣` glyph is converted back to a real space before sending.

Implementation: `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:586`.

---

## The Three Interventions

### 1. Additive Steering

For each affected activation \(h\), source direction \(d\), and signed strength \(s\):

```text
r = ||h||
v = s * r * d
v = v * min(1, r / max(||v||, eps))
h' = h + v
```

The direction is already unit length for the UI’s single-token case. Therefore, approximately:

\[
h' = h + \operatorname{clip}(s,-1,1)\|h\|d
\]

Important consequences:

- Positive strength adds the selected direction.
- Negative strength adds its negative, intended to suppress the readout.
- Strength is relative to the local activation norm, independently at every position and layer.
- Each layer’s injection is capped at one local residual norm.
- With a single source direction, `+1` and `+2` produce effectively the same per-layer injection; likewise `-1` and `-2`.
- The UI nevertheless offers `-2.0` through `+2.0`.
- Steering at several layers still compounds, even when each individual write is capped.
- A negative value does not guarantee that the token’s final probability falls; later nonlinear processing and contextual effects remain.

The UI default is `-0.1`, with step `0.1`.

Implementation: `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:754`.

### 2. Ablation

Ablation ignores the configured strength and projects the aggregate source direction out:

\[
\hat d = d/\|d\|
\]

\[
h' = h - (h \cdot \hat d)\hat d
\]

Properties:

- Applied independently to every affected position and layer.
- Removes both positive and negative projection along that axis.
- Does not remove a token from the prompt or force its readout probability to zero.
- If the direction has zero norm, it is a no-op.
- The UI retains the old strength setting while ablation is enabled, but the backend ignores it.
- Moving the strength slider disables ablation again.

Implementation: `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:762`.

### 3. Swap

Given aggregate source and target directions:

\[
\hat s=s/\|s\|,\quad \hat t=t/\|t\|
\]

\[
c=h\cdot\hat s
\]

\[
h' = h-c\hat s+c\hat t
\]

Equivalent delta form:

\[
\Delta h=c(\hat t-\hat s)
\]

Properties:

- Removes the activation’s signed coefficient along the source axis.
- Adds the same signed coefficient along the target axis.
- Negative source projection remains negative along the target.
- Components outside the removed source component remain intact.
- It is not a text replacement and does not replace token IDs.
- It does not force the target token to be generated.
- Source and target need not be orthogonal, so the added target component may reintroduce some source-axis projection.
- There is no swap-strength parameter.
- Swap takes precedence over both ablation and additive strength.

The UI sends source and target with the same lens type, although direct API validation does not enforce this.

Implementation: `apps/inference/neuronpedia_inference/endpoints/lens/prompt.py:814`.

---

## Position-Wise Behavior

### There is no position-targeted intervention

The UI allows selecting or hovering transcript positions, but this only changes:

- Which positions contribute to sidebar occurrence counts.
- Which positions contribute to the per-layer source-token histogram.
- Therefore, which layer additive steering chooses by default.
- Transcript highlighting and display.

It does not produce a position mask in the inference request.

The only backend position mask is the BOS skip mask.

Relevant UI logic: `apps/webapp/components/jlens/use-jlens-analysis.tsx:670`.

### Where the operation applies

At each selected layer, the write applies across the activation tensor’s complete position dimension:

- Every prompt token.
- Chat headers and scaffolding.
- User content.
- Assistant-prefill content.
- System messages.
- Turn delimiters and other special tokens.
- Generated positions when decode intervention is enabled.

The exception is any position whose token ID exactly equals `bos_token_id`. That position is returned unchanged because its attention-sink activation norm can make norm-scaled interventions disproportionately large.

This is an ID test, not a “first position” test:

- A BOS at position zero is skipped.
- A BOS ID appearing elsewhere is also skipped.
- Non-BOS chat special tokens are not skipped.

Tests: `apps/inference/tests/unit/test_lens_intervention_streams.py:250`.

### Causal effects within prefill

The token IDs of the prompt are fixed inputs. The intervention cannot change them.

A write at position \(p\):

- Changes that position’s downstream residual computation.
- Changes its displayed readout.
- Can affect future positions through causal attention in later blocks.
- Can change the distribution used to sample the first generated token.

It is not conditional on whether the selected source token was read out at position \(p\). The source token’s occurrence in the UI is only how the user discovered and selected the direction.

---

## Prefill Only Versus Prefill And Decode

The intended execution timeline is:

```text
intervened prompt forward
    -> sample generated token g1

forward with g1 as input
    -> sample g2

forward with g2 as input
    -> sample g3
...
```

### Prefill only: `steerGeneratedTokens=false`

- All non-BOS prompt positions are directly intervened.
- The first generated token can change because it is sampled from the intervened prompt.
- Generated-token forwards receive no additional direct write.
- Later generated tokens can still differ indirectly because:
  - Prompt KV/context was produced from the intervened prompt.
  - The first generated token may differ.
  - All later decoding conditions on that changed history.

“Prefill only” therefore does not mean “generation is unaffected.” It means generated positions are not themselves directly rewritten.

### Prefill and decode: `steerGeneratedTokens=true`

- Prompt behavior is unchanged: all eligible prompt positions are intervened.
- Each generated token is also intervened when it is fed back through the model.
- The write on generated token \(g_i\) cannot change \(g_i\), because \(g_i\) was already sampled.
- It can directly alter the logits that choose \(g_{i+1}\).
- Its readout reflects the modified residual and predicts \(g_{i+1}\).

### Defaults

There are two different defaults:

- Public API/backend default: `false`, prefill only.
- Fresh UI steer/swap configuration: `true`, prefill plus decode.
- An old shared run missing this field restores it as `false`.

UI initialization: `apps/webapp/components/jlens/use-jlens-analysis.tsx:710`.

### vLLM one-token edge case

The pinned vLLM engine classifies a segment as prefill using:

```text
is_prefill = number_of_tokens_in_segment > 1
```

Therefore, with `steerGeneratedTokens=false`:

- A one-token prompt is treated as decode and is not intervened.
- A one-token chunk of chunked prefill has the same issue.
- Eager inference does not have this discrepancy; it always intervenes on the prompt forward.

Normal JLENS raw/chat prompts generally contain BOS/template tokens and exceed one token, but exact one-ID replay exposes this edge.

---

## Layer Selection

JLENS has several unrelated concepts called “layer selection.”

### 1. Backend readout layers

The underlying inference endpoint supports a `layers` field:

- Empty means all available layers.
- Logit Lens availability is every model layer.
- Jacobian Lens availability is every fitted source layer.
- Explicit layers are intersected with availability.
- The final model layer is always appended.
- Results are sorted and deduplicated.

However, the public `/api/lens/prompt` route and normal JLENS UI do not expose this field. The UI always requests all available readout layers.

### 2. Sidebar layer range

The main layer-range slider is display and aggregation state only.

It changes:

- Which layers contribute to the sidebar’s common-token counts.
- Which layers contribute to transcript highlights.
- Which layer is previewed while hovering.
- Which layers become the initial swap selection.

It does not:

- Restrict captured layers.
- Reduce inference work.
- Change the existing model output.
- Directly specify intervention layers.

The initial displayed range is:

```text
bounds = [minimum layer in either lens, maximum layer in either lens]
start = min + floor(0.29 * (max - min + 1))
range = [start, max]
```

The first approximately 29% is hidden from the default aggregation because early J-Lens readouts are considered typically degenerate.

Implementation: `apps/webapp/components/jlens/use-jlens-analysis.tsx:289`.

### 3. Popup locked layer

Locking a layer in a token popup only changes the popup’s displayed readout. Internally it stores an index into that lens type’s layer array, not a model layer number.

It has no inference effect.

### 4. Intervention layers

These are the actual `steerLayers`.

Properties:

- Zero-based model layer numbers.
- May be non-contiguous.
- Clicking toggles individual layers.
- Dragging paints a selected or deselected run.
- Each selected layer receives an independent write.
- Effects compound through later layers.
- Changing intervention settings clears the old alternative output; the user must rerun.
- The UI refuses to run with zero selected layers.

### Additive/ablation default layer

For the selected exact token and lens type:

1. Choose a position scope:
   - Explicitly selected positions, if any.
   - Otherwise the hovered position, if any.
   - Otherwise all positions.
2. For every available layer of that lens type, count exact occurrences of the source string in all returned top-N cells.
3. Choose the first layer with the greatest positive count.
4. Since layers are sorted and ties only replace on strict `>`, ties choose the lowest layer.
5. If the token never appears, choose the highest available layer.
6. If there are no available layers, choose none.

This count is independent of the displayed layer range.

### Swap default layers

Swap initially selects every available source-lens layer inside the current sidebar range.

If no available layer intersects that range, it falls back to the additive peak-layer rule.

Switching between Steer and Swap recomputes and resets the selected layers according to the new mode.

### Empty and invalid layers

Actual behavior for `steerLayers: []` is no intervention.

This contradicts comments in the schema and public API documentation claiming empty means “readout layers.” The UI avoids the discrepancy by disabling execution.

Direct callers can submit arbitrary integers because range validation is absent:

- Too-large layers eventually fail.
- On eager Python indexing, negative layers may accidentally address layers from the end.
- vLLM failure behavior can differ.
- The normal UI only selects server-reported valid layers and does not expose this problem.

---

## Chat And Completion Semantics

### Completion mode

Raw text is encoded as follows:

1. If `prependBos=true`, a BOS token string exists, and the text does not literally begin with that BOS string, prepend it.
2. Tokenize with `add_special_tokens=false`.

Completion defaults:

- Temperature: `0`.
- Generated tokens: `32`.
- Maximum generated tokens: `128`.
- Top readouts: `8`.
- Non-word filtering: on.

### Chat mode

Chat uses the model’s real template or engine-provided formatter.

If the model has no chat format, the request is rejected; JLENS does not fabricate a ChatML prompt.

Ordinary chat behavior:

- If the last request message is not assistant, append the model’s assistant-generation scaffold.
- If the last request message is assistant, treat it as an open assistant prefill:
  - `add_generation_prompt=false`
  - `continue_final_message=true`
- Generation continues directly after the provided assistant text.

The assistant prefill UI is limited to 512 characters. User messages are limited to 1,024 characters.

Chat defaults:

- Temperature: `0`.
- Generated tokens: `128`.
- Maximum: `1,024`, except `deepseek-v4-flash` at `2,048`.
- Top readouts: `8`.
- Non-word filtering: on.

### Thinking controls

The normal UI does not expose thinking.

The public route supports `enableThinking`, default `false`:

- Passed as `enable_thinking` where the template accepts it.
- Mapped to high/low `reasoning_effort` for Harmony models.

Backend `preserveThinking` defaults to `true` but is not exposed by the public route. It preserves prior reasoning blocks where supported so chat-template token prefixes remain stable across turns.

### How a steered chat is produced

The original transcript remains untouched in a separate column.

For a steered regeneration:

1. Copy the conversation.
2. Remove its final assistant response.
3. If that response began with a real user-provided assistant prefill, reconstruct that non-generated prefix.
4. Reappend the recovered prefill as an open assistant turn.
5. Regenerate the assistant response under the intervention.

Thus steering regenerates the last assistant turn rather than applying an operation retroactively to its displayed text.

Completion mode simply reruns the same raw prompt under the intervention.

---

## Generation Controls

### Temperature

- `temperature <= 0`: greedy `argmax`.
- Positive temperature on eager:
  \[
  P(t)=\operatorname{softmax}(z/\text{temperature})
  \]
  followed by `torch.multinomial`.
- vLLM receives its own sampling parameters with temperature.
- JLENS supplies no seed.
- There is no exposed top-p, sampling top-k, repetition penalty, frequency penalty, or presence penalty.

Therefore positive-temperature runs are not exactly reproducible through the JLENS contract.

### Completion length

The configured count is an upper bound:

- EOS can end generation early.
- The server clamps generation so prompt plus completion fits its configured model sequence budget.
- The emitted metadata contains the clamped count.
- Actual token frames and `done.seq_len` are authoritative.

For vLLM, JLENS internally requests one extra sampled token:

- A generated token has a readout only after it is forwarded as input.
- Requesting exactly \(N\) samples would leave sample \(N\) without a residual row.
- vLLM samples \(N+1\), forwards the \(N\)th token, and discards the surplus sample.
- For an intervention with zero requested generated tokens, one throwaway token is requested to ensure the intervened prefill forward executes; it is not emitted.

The eager loop explicitly samples and then forwards every requested generated token, so it needs no visible surplus.

### Non-finite logits

If aggressive steering produces `NaN` or infinite generation logits, eager rejects the run before multinomial sampling and advises reducing strength or the number of layers. This prevents a CUDA device-side assertion.

---

## Readout Controls That Do Not Change Generation

### Lens mode

The Jacobian, Logit, and Diff tabs are display modes.

The UI always requests both types in this order:

```json
["JACOBIAN_LENS", "LOGIT_LENS"]
```

They share one model forward and the union of captured layers.

Diff mode does not compute a third lens. It compares occurrence counts:

- Keep a token in a lens’s column only when that lens has the greater count.
- Rank by:
  \[
  \frac{\text{this count}+2}{\text{other count}+2}
  \]

### Top-N readouts

Default and maximum in the UI are both 8.

This controls the number of vocabulary entries returned for every:

```text
(position, lens type, layer)
```

It does not alter model generation. It can indirectly alter a future intervention’s default layer because the default layer is chosen by counting source occurrences within the returned top-N.

### Non-word filtering

Filtering affects readout ranking only, never generation or probability normalization.

A word-like token:

- Is nonempty after trimming.
- Is not `<|...|>` or another `<...>` special form.
- Contains only Unicode letters or numbers.
- May contain `'`, `-`, or `’` only internally.

The service:

1. Computes `logsumexp` over the complete, unmasked vocabulary.
2. Masks non-word candidates for ranking.
3. Selects top-N.
4. Reports their true full-vocabulary probabilities.
5. Rounds probabilities to four decimal places.

Actual implementation preserves a non-word true top-1 only on the final/output-layer row, despite documentation claiming preservation at every layer.

The sidebar additionally removes `<`-prefixed entries while filtering is enabled, but popups and stored per-key statistics can still contain them.

Changing the filter after a run replays exact token IDs with generation disabled. It does not regenerate the text.

---

## Exact UI Request Shapes

A normal chat run is effectively:

```json
{
  "modelId": "qwen3.6-27b",
  "chat": [
    { "role": "user", "content": "..." }
  ],
  "type": ["JACOBIAN_LENS", "LOGIT_LENS"],
  "topN": 8,
  "temperature": 0,
  "numCompletionTokens": 128,
  "cachedTokenIds": [],
  "filterNonWordTokens": true
}
```

A fresh additive steer adds:

```json
{
  "steerTokens": [
    { "token": " spiders", "type": "JACOBIAN_LENS" }
  ],
  "steerLayers": [37],
  "steerStrength": -0.1,
  "steerAblate": false,
  "steerGeneratedTokens": true
}
```

Ablation changes:

```json
{
  "steerAblate": true
}
```

Swap adds a target and ignores strength/ablation:

```json
{
  "steerTokens": [
    { "token": " spiders", "type": "JACOBIAN_LENS" }
  ],
  "steerLayers": [18, 19, 20, 21],
  "swapToken": {
    "token": " ants",
    "type": "JACOBIAN_LENS"
  },
  "steerGeneratedTokens": true
}
```

The public route supplies defaults for omitted fields:

```text
prependBos              true
enableThinking          false
stream                  true
filterNonWordTokens     true
temperature             0
topN                    8
numCompletionTokens     128
steerGeneratedTokens    false when omitted
```

The direct inference endpoint has different defaults—temperature `1`, top-N `10`, generation `0`—so reproducing JLENS means using the public/UI defaults, not relying on direct backend defaults.

Public route: `apps/webapp/app/api/lens/prompt/route.ts:64`.

---

## Prefix Reuse, Exact Replay, And Shares

### `cachedTokenIds`

This is only a readout optimization.

The server:

1. Retokenizes the new prompt/chat.
2. Finds the longest exact common token-ID prefix.
3. Runs the full prompt forward anyway.
4. Skips readout computation and token-frame emission for reused positions.
5. Reuses the browser’s old readout data there.

The full forward remains necessary because later activations depend on the prefix.

Any real intervention disables prefix reuse because unsteered cached readouts are invalid.

### `inputTokenIds`

This is exact teacher-forced replay:

- Bypasses tokenization.
- Ignores prompt/chat.
- Forces generation to zero.
- Runs the complete fixed sequence as prompt/prefill.
- Inference marks every position as non-generated.
- The frontend may restore old generated styling separately.

This is used for:

- Non-word filter changes.
- Shared-run recomputation.
- Recomputing a saved steered output.

### Important replay discrepancy

Suppose an original run used:

```text
steerGeneratedTokens = false
```

Originally:

- Prompt positions were directly intervened.
- Generated positions were not.

During exact-ID share or filter replay:

- The whole prompt-plus-generated sequence becomes one teacher-forced prefill.
- Therefore every originally generated position is now directly intervened.
- This happens even though `steerGeneratedTokens` is still false.
- Token IDs remain pinned, so the replay does not regenerate a new completion; it changes hidden states and readouts over the fixed sequence.

A saved steered share is consequently a faithful reproduction of the token IDs, but not always of the original prompt-versus-decode intervention phase assignment.

Share path: `apps/webapp/app/api/lens/share/route.ts:512`.

---

## Where The Write Lands

A lens artifact may declare both a capture point and, for multi-stream models, a reduction.

Supported architectural capture points are:

```text
block_output
attn_out
mlp_out
attn_in
mlp_in
```

On conventional single-stream transformers, `block_output` maps to block `resid_post`.

On hyper-connection models, the block output may be a stack:

```text
[..., n_streams, d_model]
```

The artifact declares how that stack becomes the vector the lens was fitted on:

```text
none
mean
sum
select(stream_index)
```

The intervention is written at the same architectural point as the readout:

- `select(k)`: modify only stream \(k\).
- `mean` or `sum`: modify every stream.
- Ablation and swap commute exactly with mean/sum reduction because they are linear in \(h\).
- Additive steering scales against each written stream’s own norm, not the norm of the reduced mixture.

An undeclared lens is accepted on a single-stream model but rejected on a multi-stream model, because mean, sum, and individual streams all have the same final `d_model` shape and silently choosing one would produce plausible but wrong readouts.

Implementation: `apps/inference/neuronpedia_inference/endpoints/lens/residual_spec.py:76`.

The eager streaming path supports block-output artifacts. vLLM can resolve the other declared capture points through `interp-engine`.

---

## Streaming And Visible Results

Frames arrive as newline-delimited JSON:

```text
meta
prompt
token...
done
```

or an `error` frame after streaming has started.

A `token` frame contains:

- Sequence position.
- Exact token ID.
- Decoded display string.
- Prompt/generated flag.
- Chat role/channel/section metadata.
- One readout slice per lens type.
- `[layer][rank]` top-token and probability arrays.

Prompt tokens are sent before inference so the UI can render the conversation immediately. Readout token frames then replace those placeholders as computation finishes.

Readout hooks run after intervention hooks, so displayed readouts describe the modified residual.

Pressing Stop aborts the browser request and propagates cancellation to inference:

- Completion mode keeps the partial primary stream.
- Chat mode restores the prior conversation and discards the partial new primary turn.
- A partial steered alternative may remain visible.
- No host failover is possible after an HTTP 200 stream has begun.

---

## Reproduction Requirements

For faithful reproduction, record:

1. Exact model weights and tokenizer.
2. Exact chat template or engine formatter.
3. Neuronpedia model-to-host/model-ID mapping.
4. Exact Jacobian artifact, including capture-point provenance.
5. Exact input token IDs.
6. Requested lens types and their order.
7. Top-N and non-word filtering.
8. Intervention source decoded string and lens type.
9. Target decoded string for swap.
10. Exact intervention layer list.
11. Strength and ablation flag.
12. `steerGeneratedTokens`.
13. Prompt versus exact-ID replay mode.
14. Temperature and completion cap.
15. Backend; the current service pins `interp-engine==1.3.3` in `apps/inference/pyproject.toml:50`.

For deterministic generation, use temperature zero. JLENS exposes no random seed for positive-temperature sampling.

---

## Behavioral Traps

1. **Position selection is analysis-only.** There is no position-specific steer, ablate, or swap.
2. **Fresh UI runs use prefill plus decode.** API/backend default is prefill only.
3. **“Prefill only” still changes generation.** It directly changes the distribution for the first generated token and the context inherited by all later tokens.
4. **A decode write affects the next token.** It cannot change the generated token already being processed.
5. **Swap is not token replacement.** It transfers a residual projection coefficient between directions.
6. **Strength beyond approximately ±1 saturates per layer for one source token.**
7. **Empty intervention layers mean no intervention, despite documentation saying otherwise.**
8. **Only exact BOS IDs are protected.** Other template and special tokens are modified.
9. **A source or target must resolve to one vocabulary entry.** Multi-token phrases are unsupported.
10. **Exact-ID replay reclassifies originally generated positions as prefill.**
11. **Positive-temperature generation is not reproducible through the current contract.**
12. **The guided tour’s scripted spiders-to-ants swap is exceptional:** it loads a committed result fixture after a delay instead of invoking live inference, at `apps/webapp/components/jlens/jlens-chat.tsx:671`.
