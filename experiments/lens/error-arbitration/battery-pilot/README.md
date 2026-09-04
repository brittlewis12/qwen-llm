# Error-Arbitration Battery Pilot

This is a construction and coding pilot, not the population battery. It freezes:

- Qwen3.6-27B Q8;
- Neuronpedia n1000 J transport;
- raw-LM-head `error` and `lightning` directions;
- layer 46;
- all prefill and decode positions;
- injected-norm fraction `+0.45`;
- greedy decoding;
- 64 generated-token maximum.

Six anomaly families each have one anomalous prompt and one matched-correct
control. Every prompt receives a baseline, error, and lightning condition.
The pilot decides which strata have baseline ceiling effects or ambiguous
outcome coding before any larger panel is frozen.

Outcome codes:

- `ordinary_response`
- `anomaly_mention`
- `correct_repair`
- `incorrect_repair`
- `spurious_repair`
- `response_and_repair`
- `lexical_leakage_or_degradation`
- `other`

Do not tune layer or coefficient from this pilot.

The primary control statistic is the `spurious_repair` rate among matched-
correct prompts under error steering. Incorrect repairs on anomalous prompts
are tallied separately.

For each pair, define `p*` as the first token position completing evidence of
the anomaly. The error sweep records the selected deployed error score at every
layer-46 prompt position. At `p*`, let baseline score be `b` and the local score
slope under raw error steering be `m = (score_0.45 - b) / 0.45`. With measured
raw/gamma direction cosine `c`, `b * c / m` is a normalized gamma-coordinate
proxy. This is exploratory in the six-pair pilot.

The prospective full battery will preregister an opposing-sign prediction:

- fixed crossing predicts larger baseline coordinates flip more often under a
  fixed dose;
- multiplicative amplification predicts larger baseline coordinates flip less
  often under a fixed dose.

No adaptive dose is fit or run in this pilot.
