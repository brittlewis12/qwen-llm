# GLM-5.3-Flash Fast prefill: quality on text it did not write (map #12, 2026-10-08)

Question: Fast prefill drifts from Exact by KL up to 0.18 on natural
trajectories (`../2026-10-07-glm53-natural-reuse/`). Does that drift cost
answer quality? And what produces it?

Answer: no measurable quality cost. On the preregistered cohort Fast at
512 rows (serve's default) is within every limit: the mean negative
log-likelihood of real continuations rises by 0.0004 nats/token (0.04%
perplexity), top-1 accuracy is unchanged, every stratum passes, and the
tool screen is 12/12 for both lineages. The overall upper bound, 0.0046,
sits close to its 0.005 limit, and this is a pass on this cohort and tool
screen, not general answer-quality equivalence. Separately, rounding only
the activations to half precision inside Exact produces drift comparable to
Fast's on the two cases probed: activation rounding alone is a sufficient
source of drift of this size. Its share within Fast, and what F32 staging
would leave, remain unmeasured.

## Preregistration

`scripts/reference/glm53/quality-v1.json` (cohort) and
`scripts/reference/glm53/quality_analysis.py` (limits and verdict logic)
were committed (`942f859b`) before the GPU run; the analysis was amended
(`02b08456`) after the cx 01a10cc preregistration review and before any
result was viewed (stricter tool scoring, separate text and tool
verdicts, per-task sampled weighting, completeness checks); the numeric
limits never changed.

- Text cohort: 22 repository documents (8 prose and 8 code at 512 and
  2,048-token prefixes, 6 long at 4,352 past the sparse frontier; 38
  items). Each prefix is read with Exact (512-row chunks), Fast 512 or
  Fast 97; the next 64 real tokens are fed identically, one at a time, in
  every arm. Measure: mean NLL per item; difference from Exact; document
  bootstrap (20,000 resamples).
- Limits: overall mean NLL increase, one-sided 95% upper bound <= 0.005
  nats/token; each stratum <= 0.010; top-1 accuracy loss >= -1 point.
- Tool screen (descriptive): 12 weather/currency tasks with known calls
  (four with 1.5K or 5K tokens of context), greedy and seeded release
  sampling, exactly one correct call required.

## Results (`report.json`, `verdict.json`; evaluated at `942f859b`, API validation on)

| Fast 512 vs Exact | Observed | Central 90% | Limit | Verdict |
|---|---:|---:|---:|---|
| Overall NLL (nats/token) | +0.0004 | [-0.0039, +0.0046] | <= 0.005 | PASS |
| Top-1 accuracy | +0.0004 | [-0.0041, +0.0049] | >= -0.01 | PASS |
| Prose NLL | -0.0017 | [-0.0104, +0.0068] | <= 0.010 | PASS |
| Code NLL | +0.0035 | [-0.0014, +0.0082] | <= 0.010 | PASS |
| Long (past the frontier) NLL | -0.0023 | [-0.0085, +0.0043] | <= 0.010 | PASS |

Text cohort: QUALIFIES. Tool screen: PASS (greedy 12/12 for both;
sampled 1.00 for both, short and long-context groups alike). Mean NLL
1.6422 (Exact), 1.6426 (Fast 512), 1.6423 (Fast 97). Fast at 97 rows
passes overall; its long stratum is inconclusive (upper bound 0.0136), as
is Fast 97 vs Fast 512 on long prompts: batching alone moves long-prompt
NLL by about as much as the stratum margin's uncertainty.

Scope: this measures continuation prediction on this repository's
documents and a narrow tool screen; it is not human-chat or general agent
quality, and it does not address reuse consistency (which is a separate
property: Exact reuse is bitwise; Fast warm and cold may differ, as
llama.cpp's do).

## What produces the drift (`rounding-probe.json`, `5e183317`)

Exact lineage with every quantized-weight matrix input rounded through
half precision first (weights still F32-dequantized, decode accumulation),
on the frozen natural cases:

| Case | Rounded Exact vs Exact: worst / mean KL | Fast vs Exact | Fast vs rounded Exact |
|---|---:|---:|---:|
| H2 code (1,755 tokens) | 0.164 / 0.028 | 0.113 / 0.026 | 0.098 / 0.027 |
| H5 long chat (4,962) | 0.154 / 0.019 | 0.095 / 0.013 | 0.332 / 0.018 |

On these two cases, half-rounded activations alone produce Fast-sized
drift (and flips), but not Fast's specific outputs: Fast is as far from
rounded Exact as each is from Exact. This shows activation rounding is a
sufficient source of drift of this size; it does not exclude kernel
defects, show that every half-precision path must drift this much, or
measure Fast's residual error after F32 staging. The F32-staging experiment
is deprioritized given the cohort result, not refuted.

## Reading for #12

The 2026-10-05 Fast policy (KL 2e-2, regret 0.2) was a regression bound
calibrated on one prompt, not a quality criterion. Measured against
quality, Fast passes; measured against agreement with Exact, so would no
half-precision batched path (including llama.cpp's own). Options:

1. Keep Fast as the default and replace the agreement bound with this
   quality criterion, with the properties stated separately (PERF-LOG
   2026-10-08 states the proposal and its review qualifications): reuse
   and restore bitwise within a lineage and schedule; drift of the measured
   class (worst KL to ~0.2, flips with regret up to 1.50) accepted and
   reported, as for DS4 and llama.cpp; quality non-inferior on a
   preregistered cohort for any non-bit-identical Fast change or new
   schedule. A policy change: needs an explicit decision.
2. Pursue F32 staging anyway: may remove one source; no quality gain is
   shown to win on this evidence.
3. Exact by default: 4.7-4.9x slower fresh prefill for no measured quality
   gain.

`x_qwen.prefill_lineage: "exact"` (`43d344a5`) already serves evaluations
and reproducible comparisons.

## Provenance after rebase

The raw artifacts record pre-rebase commits; the branch was rebased onto
main before integration with no change to the harness or analysis.
Mapping: `21917196` -> `6bf82db0` (cohort generation, recorded in
`quality-v1.json`), `8781fe0b` -> `942f859b` (evaluator, recorded in
`report.json` and `verdict.json`), `aaa79ca2` -> `02b08456`, `a2119edc` ->
`5e183317`. The rebase brought in the sparse IQ3_S down retile for Fast
prompts of 32-512 rows (`3a92067c`), which landed after this evaluation;
its packet observed bit-identical endpoint and continuation outputs against
the incumbent in all 120 comparisons at 32-512 and 4,096 rows
(`../2026-10-08-glm-iq3-down-retile/`), so these results are expected to
carry over. That is an observation, not a re-run.

In `report.json`, each tool row's `correct` is the harness's lenient
first-call diagnostic (renamed `lenient_first_call_correct` afterwards);
the tool verdicts come only from `quality_analysis.py`'s strict rescoring
of the recorded calls.
