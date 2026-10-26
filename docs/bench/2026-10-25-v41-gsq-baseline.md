# V4.1 GSQ-RCO IQ2_XS Baseline Against Unsloth Q2_K_XL And IQ3_XXS

One build of main `de082865` (plus the additive private text-likelihood
witness helper as a local tmp commit, `b53be328`; `build_identity.status =
match` on every arm), M5 Ultra 80-core, 256 GiB, 240 GiB wired limit,
production lease. Artifacts:

| artifact | source | resident views |
|---|---|---:|
| GSQ-RCO IQ2_XS (2.25 bpw label) | ISTA-DASLab, pinned rev `cb02422c`, verified shard hashes; Engram IQ4_NL companion disk-only | ~144.6 GiB |
| UD-Q2_K_XL | Unsloth 1017 | ~185.9 GiB |
| UD-IQ3_XXS (incumbent) | Unsloth 1017 | ~217.4 GiB |

All three were admitted by the current admission policy. Raw:
`metagross:~/qwen-runs/2026-10-25-gsq-baseline/` (`suite/`, `warm/`,
`witness/`, `screen/`).

## Throughput

`qwen-bench suite --pp 2048,4096 --tg 32 --runs 2 --seed 1`, depth 0.
Prefill: model order GSQ, Q2, IQ3, IQ3, Q2, GSQ plus two later warm pairs
(GSQ, GSQ, Q2, Q2); prefill is insensitive to cache state (within 0.6%).

| artifact | pp2048 | pp4096 | tg32 cold load | tg32 warm repeat |
|---|---:|---:|---:|---:|
| GSQ IQ2_XS | 859.6 (857.4-862.2, n=4) | 873.8 (870.9-876.6, n=4) | 42.2-43.1 | 55.1 |
| Q2_K_XL | 843.8 (842.4-845.9, n=4) | 858.6 (856.9-860.0, n=4) | 41.7-42.0 | 54.2 |
| IQ3_XXS | 849.0 (848.6-849.5, n=2) | 879.5 (878.9-880.2, n=2) | 39.3 | 50.9 |

**Decode depends on page-cache state.** `tg32` is ~0.7 s of decoding; on
an arm whose model was not left resident by the previous arm, first-touch
page faults cut it by 22-26% (every model, prefill unchanged). The warm
column is the arm that immediately repeats the same model. Earlier decode
figures in the roadmap (IQ3 41.0, Q2 43.2) match the cold column, so the
current warm decode baseline is ~51-55 tok/s, not ~41-43. Decode
comparisons need warm repeats or an explicit prefault.

## Quality

Six frozen held-text documents, 64 teacher-forced targets each (the
`v41_text_likelihood` witness, `MTL_DEBUG_LAYER=1`). GSQ and Q2 packets are
prepared from the same corpus and carry token IDs identical to the IQ3
packet, so all three score the same 384 targets.

| document | GSQ NLL | Q2 NLL | IQ3 NLL | GSQ top-1 | Q2 top-1 | IQ3 top-1 |
|---|---:|---:|---:|---:|---:|---:|
| prose_1 | 2.5571 | 2.6459 | 2.6109 | 0.562 | 0.562 | 0.594 |
| prose_2 | 4.2594 | 4.2276 | 4.2000 | 0.188 | 0.203 | 0.188 |
| prose_3 | 2.2201 | 2.2233 | 2.2889 | 0.438 | 0.469 | 0.469 |
| code_4 | 1.2757 | 1.2320 | 1.2550 | 0.750 | 0.703 | 0.703 |
| code_5 | 1.5399 | 1.6129 | 1.5859 | 0.625 | 0.625 | 0.688 |
| code_6 | 1.0005 | 0.9820 | 0.9850 | 0.750 | 0.766 | 0.766 |
| mean | **2.1421** | 2.1539 | 2.1543 | 212/384 | 213/384 | 218/384 |

GSQ has the lowest mean NLL (-0.57% vs IQ3) and IQ3 the most top-1 hits
(+6/384); per-document differences run +-0.09 nats in both directions, so
on this witness the three are indistinguishable. The 22-task screen: code
9/10 (GSQ) vs 8/10 (Q2, IQ3), the single difference being case c3;
reasoning 1/8 with one truncation and JSON 0/4 for all three. The screen's
JSON and reasoning floors are properties of the screen, not of these
quants; it cannot rank models.

## Reading

GSQ-RCO IQ2_XS matches the incumbent IQ3_XXS on everything this setup can
measure while taking 2/3 of its resident memory (72.8 GiB less), with
equal prefill and ~8% faster warm decode. That makes it the strongest
V4.1 artifact candidate on metagross. A default change needs a quality
instrument stronger than six documents and a 22-case screen (the open
model-choice question), not more throughput data.

## Limits

- One build, one host session; the IQ3 warm figure is a single arm.
- The witness is six documents / 384 targets; differences below ~1% NLL or
  ~2% top-1 are not resolved.
- The vendor's benchmark claims for the GSQ artifact are not evaluated here.
