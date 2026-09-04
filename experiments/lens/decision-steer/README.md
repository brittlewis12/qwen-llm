# Decision Direction Geometry Gate

Status: preregistered local gut check, not a behavioral result.

## Purpose

Validate the local additive intervention path before testing framework behavior.
The gate uses one cell where the Neuronpedia n1000 J-lens reads token 101725
(`decision`) at rank zero on the unchanged Q8 prompt.

## Frozen Inputs

- Model: pinned Qwen3.6-27B Q8_0 GGUF.
- Lens: pinned Neuronpedia Qwen3.6 n1000 J transport.
- Prompt: exact 57 token IDs from the validated WRAP no-thinking trace.
- Direction token ID: 101725.
- Operation site: post-block residual, layer 54, prefill position 52.
- Readout sites: position 52 at layers 54, 55, 58, and 62.
- Action: unit-L2 `residual_l2_fraction` addition.
- Ordered coefficients: `0,-0.10,-0.05,0.05,0.10,0`.
- Execution: serial prefill, greedy, one generated token.

## Predictions

1. The selected direction's layer-54 readout moves monotonically with authored
   coefficient and has the same sign as the coefficient.
2. Duplicate zero arms have identical generated token IDs, stop reason, and
   readouts.
3. The intervention remains detectable downstream. No monotonic decay shape is
   required because subsequent blocks may amplify, rotate, or rederive it.
4. All values remain finite and nonzero arms record exactly one application.

The readout score is not expected to increase numerically by exactly alpha.
`residual_l2_fraction` scales the unit direction by the current residual norm,
and the selected lens score has its own covector scale.

## Stop Conditions

Do not proceed to behavioral or swap experiments if:

- the same-layer effect has the wrong sign or is non-monotonic;
- duplicate zero arms differ;
- application records disagree with the authored site; or
- any arm produces non-finite values or execution failure.

## Exploratory Behavior Gate

The additive and ablation geometry gates passed on Q8 and BF16. The next plan,
`behavior-plan.json`, applies the same direction at layer 41 over all prefill and
decode positions. Layer 41 is the earliest layer tied for the highest local
top-25 occurrence count for token 101725 on this prompt.

The ordered greedy coefficients are `0,-0.4,-0.7,-1,0.4,0.7,1,0`. Outcomes are
descriptive and coded as:

- WRAP heading present or absent;
- count and order of canonical components: Widen, Reality-test, Attain
  distance, Prepare;
- acronym/component mutation;
- repetition or malformed-output failure; and
- stop reason and generated token count.

The historical screenshot supports graded framework mutation at `-0.7`, not a
binary framework omission claim. This gate does not preregister a confirmatory
behavioral endpoint.

## Phase Decomposition

`prefill-only-plan.json` and `decode-only-plan.json` hold layer, direction,
prompt, readouts, and coefficient semantics fixed while changing only whether
the operation applies during prompt processing or generated-token feedback.
Each is swept at `0,-0.7,+0.7,0` under greedy decoding. The existing
`behavior-plan.json` supplies the both-phases comparison.

## Semantic Sham

`semantic-sham-plan.json` replaces only the decision direction with unit-L2
token ID 31367 (`lightning`). Layer 41, all-phase scope, coefficients, prompt,
generation bound, and execution topology remain matched to the decision sweep.
It reads both token 101725 and token 31367 at the boundary. The sham is expected
to move its own readout; the selectivity question is whether it reproduces the
decision-specific WRAP mutations and phase behavior.

## Neuronpedia-Style Swap

Neuronpedia's UI `Swap` is a directed source-coordinate transfer, not a
symmetric coordinate exchange. Local `source_to_target` at coefficient `1.0`
matches its formula. `decision-to-lightning-plan.json` and
`lightning-to-decision-plan.json` test each direction separately at the
validated layer-54, position-52 cell using ordered coefficients `0,0.5,1,0`.
Local `coordinate_swap` is a different Householder operation and is not used for
this parity gate.

`decision-to-lightning-behavior-plan.json` applies the faithful directed
transfer at layer 41 over all prefill and decode positions with coefficients
`0,0.5,1,0`. This is a surgical behavioral gate. Neuronpedia's current Swap UI
defaults to every visible layer (normally 18 through 63), which is deliberately
not reproduced until the single-layer result is understood.
