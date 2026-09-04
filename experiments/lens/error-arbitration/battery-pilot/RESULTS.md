# Pilot Results

Status: construction pilot complete. Do not pool these rows into a population
estimate or expand the generator unchanged.

All 12 error sweeps had exact duplicate-zero token sequences and selected
readouts.

| Family | Anomalous baseline | Error `+0.45` | Matched-control error `+0.45` | Pilot disposition |
| --- | --- | --- | --- | --- |
| Tag closing transposition | Ordinary social reply | Exact diagnosis and repair plus social reply | Ordinary reply; no repair | Retain |
| Prose transposition | Ordinary empathetic reply | Generic "causing the error" framing; does not identify `freind` | Same generic error/troubleshooting framing | Redesign; "rough morning" is a semantic confound |
| JSON missing comma | Obeys `say hi back`; ignores syntax | Same task response; no repair | Ordinary response | Retain only as a possible nonresponse stratum |
| Unbalanced parenthesis | Ordinary empathetic reply | Ordinary empathetic reply; no repair | Ordinary empathetic reply | Retain only as a possible nonresponse stratum |
| Invalid date | Corrects September 31 without intervention | Corrects the same error | Adds error-avoidance framing and an unsupported weekday claim on the valid control | Baseline ceiling; redesign |
| Arithmetic inconsistency | Corrects the tip without intervention | Corrects the same error | Introduces a contradictory claim that the exact tip is below an "18% error margin" | Baseline ceiling and strict control false positive; redesign |

## Separate Tallies

- Clean error-induced repair on a non-ceiling anomalous prompt: `1/4`.
- Baseline-ceiling anomalous prompts: `2/6`.
- Strict spurious repair or incorrect anomaly claim on matched controls: `1/6`.
- Broader spurious error/troubleshooting framing on matched controls: `3/6`.
- Lightning semantic insertion on the matched-dose arm: `12/12`; several arms
  became repetitive. Lightning is a positive semantic-contamination control,
  not a benign null sham.

The strict false-positive and broad-framing rates have different numerators and
must remain separate.

## Interpretation

- The layer-46 result generalizes cleanly to a second tag name and payload.
- This pilot does not support a general cross-class anomaly-appraisal claim.
- Fixed error steering can inject error-oriented discourse on clean prompts,
  so "amplifier" is not yet sufficient without a leakage qualifier.
- JSON and parenthesis nonresponses may be genuine class boundaries or may have
  low natural error coordinates; the recorded layer-46 scores should be
  inspected before redesigning them.
- Date and arithmetic prompts are unusable for measuring a flip because the
  baseline already repairs them.

## Coordinate Proxy Check

Using each pair's first token divergence as the pilot `p*`, raw/gamma cosine
`0.997292`, baseline score `b`, and local active-arm score slope `m`, the proxy
`b * cosine / m` was:

| Family | Anomalous proxy | Control proxy |
| --- | ---: | ---: |
| Tag closing transposition | 0.05460 | 0.03358 |
| Prose transposition | 0.03773 | 0.01882 |
| JSON missing comma | 0.02593 | 0.03415 |
| Unbalanced parenthesis | 0.02576 | 0.02762 |
| Invalid date | 0.06201 | 0.01354 |
| Arithmetic inconsistency | 0.02914 | 0.04066 |

The clean tag repair and baseline-ceiling date correction occupy the two largest
anomalous proxies, directionally favoring a fixed-crossing account over the
multiplicative account. Six heterogeneous, partly invalid rows are not a test of
that prediction. The arithmetic and delimiter loci also need better evidence-
completion definitions before a larger battery.

## Next Decision

Inspect the recorded coordinate proxies and redesign only the confounded or
ceilinged pairs. Review the revised generator before running a larger panel. Do
not tune layer 46 or coefficient `+0.45` from these outcomes.
