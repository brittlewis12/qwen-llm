# Error Arbitration Replay

This panel starts from the historical hosted prompt whose mismatched closing tag
was passively associated with `error` but behaviorally ignored until positive
steering made typo repair operative.

The first gate is passive and deterministic. It varies whether the tag pair
matches and whether the misspelling `freind` is present, while preserving the
greeting payload:

| ID | Opening tag | Closing tag | Pair matches | Contains `freind` |
| --- | --- | --- | --- | --- |
| `closing-typo-mismatch` | `friend` | `freind` | no | yes |
| `matched-friend` | `friend` | `friend` | yes | no |
| `matched-freind` | `freind` | `freind` | yes | yes |
| `opening-typo-mismatch` | `freind` | `friend` | no | yes |

Initial questions:

1. Does an `error`-family readout appear near `freind` only when the pair
   mismatches, whenever the misspelling occurs, or in neither condition?
2. Is any signal localized to the mismatching tag side?
3. Does it survive the Neuronpedia n1000 J and Camila J/R artifacts?
4. Does the unsteered model greet rather than repair the typo?

No steering sweep is frozen until this passive precursor and baseline behavior
are inspected.

`invalid-composed-geometry-plan.json` is retained as a provenance record for a
discarded calibration attempt. Its two nonzero operations composed during each
sweep, so neither resulting target artifact is valid for effect estimation.

The first behavioral sweep is explicitly exploratory threshold localization.
It uses the historical closing-tag mismatch, layer 44, all prefill and decode
positions, greedy decoding, and coefficients
`0,-0.6,-0.3,0.15,0.3,0.45,0.6,0.75,0.9,0`.

Outputs are coded as:

- `greeting`: social reply with no typo or markup repair;
- `correction`: identifies or repairs the misspelling/tag mismatch;
- `lexical_leakage`: emits error-family labels without a useful repair;
- `degraded`: incoherent, repetitive, or structurally malformed output;
- `other`: coherent output outside the preceding classes.

No coefficient selected from this sweep is itself a confirmatory estimate.

The direction-basis discriminator keeps deployed readouts fixed and compares
three unit-L2 directions at matched injected-norm fractions:

- omitted `target_covector`: deployed-logit-numerator, with RMSNorm gamma folded
  into the LM-head row before transport;
- `raw_lm_head`: Neuronpedia-parity raw LM-head row before transport;
- `raw_lm_head_orthogonal_to_deployed_logit_numerator`: the per-layer component
  of the raw transported row orthogonal to the deployed readout covector.

The orthogonal arm is explicitly invisible to the same-layer linear deployed
readout before downstream model processing. Its behavioral effect is therefore
a discriminator between readout-aligned dose and a causal component outside the
transported-logit thermometer axis.

The first held-out discriminator freezes raw `error` at layer 46 and coefficient
`+0.45`, selected as the lowest clean correction in the exploratory historical
prompt sweep. It applies that fixed arm to the three unused panel conditions and
compares the historical prompt against raw `lightning` at the same injected-norm
fraction. Every held-out sweep uses ordered coefficients `0,0.45,0`.
