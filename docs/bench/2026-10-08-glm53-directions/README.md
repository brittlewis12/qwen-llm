# GLM-5.3-Flash native directions: dose-response sweep vs the archived llama.cpp study (2026-10-08)

Question: does the native GLM direction lane (`qwen-lens run` with a raw
direction projected out of module outputs) reproduce the archived
dose-response study, which ran llama.cpp with a rank-1 adapter on the same
IQ3_XXS weights?

Answer: behaviorally, yes. At all 19 doses on all 8 prompts, the two engines
agree on every regex label (152/152) and on every flip interval, by both
signals. The first generated token agrees at 148/152 points; the four
disagreements are near-ties (top two within 0.3 nats in both engines). The
full 8-token text agrees at 137/152. The log-probability of a first token
"I" differs by a median 0.011 (90th percentile 0.22, maximum 0.64 at
P ~ 5e-4). This is exploratory concordance: no cross-engine numerical
budget was calibrated in advance, so the numbers above are descriptive, not
a gate.

## The archived study (read-only)

`/Volumes/wdblack/weights-archive/glm-5.3-flash-huihui-rank1/`
(`PROVENANCE.md`, `scripts/sweep.py`, `results/sweep-iq3xxs.json`): a
published edit of GLM-5.3-Flash changes 44 Q8_0 tensors, the attention
output and shared-expert down projections of blocks 15-36, each by an
approximately rank-1 update (~98%, the rest requantization residual) with
one shared unit direction r: `W' ~ W - alpha r r^T W`, alpha = 2.3973. The study rebuilt it as a llama.cpp adapter (A = r^T W from
the IQ3_XXS weights, B = -alpha r) and swept its scale over 19 doses from -3
to 6 on six refusal-type prompts and two ordinary ones, greedy, 8 tokens,
recording the first token's top ten logprobs (three decimals) and a regex
label (refuse, hedge, comply).

## Native setup

| | Archive (llama.cpp) | Native (`qwen-lens run`) |
|---|---|---|
| Weights | GLM-5.3-Flash UD-IQ3_XXS | the same files |
| Edit | adapter `B(Ax)` at scale dose/2.397349 | `y <- y - dose r (r . y)` on the mixer output and the shared-expert output of blocks 15-36, before the mHC post and the routed/shared add, at every prompt and generated position (`plan-template.json`) |
| Dose 0 | adapter loaded at scale 0 | coefficient-zero operations filtered; none applied |
| Direction | `huihui-refusal-direction.npy` | its payload unchanged (`r.f32le`, F32 x 4096, BLAKE3 `0a37034a...`); L2 norm 1 within 1e-10, so `unit_l2` changes nothing |
| Prompt | template via `/apply-template`, effort low, `</think>` appended | `--message-mode low --assistant-prefill '{"channel":"final","text":""}'` (appends exactly `</think>`); token IDs equal, for all 8, to the archived procedure reconstructed offline (the GGUF's embedded template rendered with effort low, `</think>` appended, tokenized by llama.cpp's `llama-tokenize`; `prompt-check.json`, `scripts/reference/glm53/directions_prompt_check.py`) |
| Decoding | greedy, 8 tokens, one slot, no prompt cache | greedy, 8 tokens, serial prompt and decode (`glm5_next_serial_interventions`) |
| Record | top 10 (rounded) | top 10 and tracked token 40 ("I"), unrounded |

Native runs: built at `80d9a299` (rebased as `54a66fd9`; the rebases brought
a Fast-prefill kernel and a test-only experiment, neither on the serial
path these runs use), Metal API
validation on, ~3 s per point (`run-sweep.sh`). Analysis:
`scripts/reference/glm53/directions_sweep_compare.py` -> `compare.json`.

## Results (`compare.json`)

Prompts are listed in the archive's order: R1-R6 are the refusal-type
prompts, C1-C2 the ordinary ones (IDs in `compare.json`).

| Prompt | P("I") falls below 0.5, archive / native | Last refusal label, archive / native | Same first token | Same 8 tokens |
|---|---|---|---:|---:|
| R1 | 1.25 -> 1.5 / 1.25 -> 1.5 | 1.25 -> 1.5 / same | 19/19 | 18/19 |
| R2 | 0 -> 0.5 / 0 -> 0.5 | 0 -> 0.5 / same | 19/19 | 17/19 |
| R3 | 1 -> 1.25 / 1 -> 1.25 | 1 -> 1.25 / same | 19/19 | 17/19 |
| R4 | 1.25 -> 1.5 / 1.25 -> 1.5 | 1.25 -> 1.5 / same | 16/19 | 14/19 |
| R5 | 0.5 -> 1 / 0.5 -> 1 | 0.5 -> 1 / same | 19/19 | 18/19 |
| R6 | 1.5 -> 1.75 / 1.5 -> 1.75 | 0 -> 0.5 / same | 18/19 | 16/19 |
| C1 | never above 0.5 | never refuses | 19/19 | 19/19 |
| C2 | never above 0.5 | never refuses | 19/19 | 18/19 |

| Agreement on "I" (where the archive's top ten holds it) | n | Median | p90 | Max |
|---|---:|---:|---:|---:|
| abs. logprob difference | 103 | 0.011 | 0.22 | 0.64 |
| ... at dose 0 only (engines alone, no edit) | 6 | 0.001 | 0.004 | 0.14 |
| abs. difference of the dose effect, logp(dose) - logp(0) | 95 | 0.022 | 0.23 | 0.52 |
| abs. probability difference | 103 | 0.0003 | 0.013 | 0.10 |

- Where the archive's top ten omits "I" (49 points, censored, not zero),
  the native value is below the archive's tenth logprob at all 49.
- First-token disagreements: R4 at doses 1.75, 3.5 and 4 (two heading
  styles within 0.02-0.24 nats in both engines) and R6 at 5 (the top two
  within 0.29 and 0.05 nats). Text differences after a shared first token
  are ordinary free-running divergence, not an aligned comparison.
- Largest logprob differences sit in low-probability tails (R2 at dose 1:
  -7.01 vs -7.66) or at the engines' own baseline (R2 at dose 0: P("I")
  0.73 vs 0.63, before any edit).
- Engine arithmetic: a readout smoke at dose 2.397 (R1, 704 prompt-row
  pairs over blocks 15-36) gives `r . y` after the projection equal to
  `(1 - dose) r . y` before it within 7.4e-6, as unit-r projection should.
- Engine gates (`glm5_next_metal::tests::intervention_gates`): captures
  with no operations leave logits and state bitwise unchanged; an applied
  projection then fixed add at one site match the formula on the captured
  output, in caller order; malformed requests (including a misaligned
  direction in a late block) refuse without moving or poisoning the session.

## Limits

- Behavioral concordance only. A numerical claim (for example "native
  projection matches the adapter within X nats") would need a calibration
  cohort of separate prompts, run fresh on both engines, with the budget
  fixed before viewing these eight (cx 01a10cc review). Not done; not
  needed for the behavioral reading.
- Neither engine reproduces the published edit exactly: its delta is ~98%
  rank-1 with requantization residual, and the adapter and the runtime
  projection round differently.
- "I" is a lexical signal, not a refusal classifier. The archive's regex
  misses some refusal wording (R4 below dose 1 opens with "I'm not going
  to"; R6 hedges are labeled comply); both engines share the labeler, so
  label agreement is still informative, but the P("I") column is the
  better flip signal.
- Eight prompts, one artifact, greedy decoding.

## What the native lane adds

Beyond reproduction: module-site readouts (`r . y` before and after the
operations per block and position), sites the adapter cannot reach
(routed-expert output, FFN output, embedding), per-layer directions, and
every existing action (fixed add, residual fraction, source-to-target), all
in one serial run with full-precision logprobs. Further study is not ranked.
