# GLM-5.3-Flash Fast lineage on natural trajectories (map #12, 2026-10-07)

Question: does serve's default Fast packed-prefill lineage stay within the Fast
policy (KL(Exact||Fast) <= 2e-2 per position; a top-1 flip only to a near tie,
both regrets <= 0.2) and the frozen warm/cold reuse gate (KL both ways <= 2e-2,
both regrets <= 0.2) on the model's own turns, rather than on the
unrelated-token continuations of the original reproducer? And what would the
Exact lineage cost as a default?

Answer: no. Exact reuse is bitwise everywhere. Fast diverges from Exact by KL
up to 0.18 with occasional top-1 flips of regret up to ~1, warm and cold Fast
differ by up to 0.165, and llama.cpp's own batched prefill diverges from its
serial run by the same order on the same tokens. Per-stage attribution finds
no single defective stage: every batched mat-mat stage that stages activations
in half produces ~0.1 KL on its own. Exact prefill costs 4.7-4.9x Fast.
No bound was changed.

## Design (jam and two reviews with cx 01a10cc)

- Two phases, so continuations are frozen before any Fast run:
  `glm5_next_metal::tests::natural::reuse_natural_generate` (Exact only)
  wrote `scripts/reference/glm53/reuse-natural-v1.json` at `eb787c5e`;
  serve's CPU round trip (`serve::render_glm5_next::natural_tests`) rendered
  every first and second turn exactly from serve's own output partition and
  response items, in the patched client's shapes too; the fixture was then
  committed (`9d426a6e`) and `reuse_natural_evaluate` ran at `9d426a6e`.
- Cohort (preregistered, ordered replacements; every case accepted on its
  first variant, journal beside the fixture): H1 short chat (157 tokens),
  H2 code context (1,755), H3 dense tool loop (351), H4 tool results
  crossing the sparse frontier (join 1,671, second turn 2,279), H5 sparse
  long chat (4,962), H6 max-effort seeded sampled tool turn (333). Low effort,
  greedy, except H6. Emitted and consumed tokens are kept apart (a stop is
  never forwarded). Continuations: 32 Exact-greedy tokens.
- Arms per case: Exact cold at 512 rows is the reference R; Exact warm at 512
  and 97 rows (and Exact cold at 97 on H4/H5) must equal R bitwise in logits
  and persistent state; Fast cold and warm at 512 and 97 rows against R under
  the Fast policy; Fast warm against Fast cold under the reuse gate.
  `MTL_DEBUG_LAYER=1`, production lease.

## Results (`report.json`)

Exact: warm equals cold bitwise in logits, and in persistent state at the
prompt end and at the end, in every case and schedule (and Exact cold at 97
on H4/H5).

| Case | Rows | Fast cold worst KL (flips) | Fast warm worst KL (flips) | Warm vs cold worst KL / regret |
|---|---:|---:|---:|---:|
| H1 short chat | 512 | 4.1e-2 (0) | 7.4e-2 (1) | 1.0e-1 / 0.08 |
| H1 | 97 | 8.0e-2 (0) | 7.4e-2 (1) | 1.1e-2 / 0.03 |
| H2 code | 512 | 1.1e-1 (1) | 1.5e-1 (1) | 7.5e-2 / 0.00 |
| H2 | 97 | 1.8e-1 (4) | 1.1e-1 (2) | 1.2e-1 / 1.07 |
| H3 dense tool | 512 | 3.6e-2 (0) | 4.5e-2 (0) | 8.3e-3 / 0.00 |
| H3 | 97 | 2.2e-2 (0) | 1.4e-2 (0) | 3.8e-3 / 0.00 |
| H4 across frontier | 512 | 2.0e-2 (0) | 9.6e-3 (0) | 3.5e-2 / 0.00 |
| H4 | 97 | 7.9e-3 (0) | 9.0e-3 (0) | 3.5e-3 / 0.00 |
| H5 sparse long chat | 512 | 9.5e-2 (0) | 1.5e-1 (1) | 1.7e-1 / 0.96 |
| H5 | 97 | 3.6e-2 (1) | 4.5e-2 (2) | 1.1e-1 / 0.16 |
| H6 max sampled tool | 512 | 2.7e-2 (0) | 8.7e-3 (0) | 4.4e-2 / 0.00 |
| H6 | 97 | 3.4e-2 (1) | 1.9e-2 (2) | 7.1e-3 / 0.18 |

The largest flips: H2 Fast cold 97 at continuation position 18 (Exact-side
regret 1.05), H5 Fast warm 512 at the prompt end (regrets 0.23 / 0.96).
Divergence starts in the prompt prefill and carries into the serially
decoded continuation.

## Attribution (`attribution-H2.json`, H2, 512 rows)

Each stage family alone in its Exact form inside a Fast session, and alone
Fast with every other family Exact (`packed::ExactStages`, test-only). The
all-Exact-in-Fast control equals Exact (KL 0). One-at-a-time substitutions
measure sensitivity, not additive shares.

| Stage family | Only it Fast: worst / mean KL | Fast except it: worst KL |
|---|---:|---:|
| MLA projections | 2.2e-1 / 3.3e-2 | 1.1e-1 |
| KDA low-rank expansions | 1.9e-1 / 2.5e-2 | 1.4e-1 |
| Shared expert | 1.6e-1 / 2.7e-2 | 1.5e-1 |
| KDA projections | 1.4e-1 / 2.5e-2 | 2.4e-1 |
| Dense FFN (blocks 0-2) | 9.9e-2 / 2.0e-2 | 2.6e-1 |
| Routed experts | 7.2e-2 / 1.6e-2 | 1.2e-1 |
| MLA absorption (F32 accumulation) | 1.8e-2 / 2.5e-3 | 1.8e-1 |
| Router (F32) | 1.3e-2 / 1.8e-3 | 2.0e-1 |
| Indexer projections (H2 is dense) | 0 / 0 | 1.1e-1 |
| All Fast (reference) | 1.1e-1 / 2.6e-2 | |

Every family that runs the batched mat-mat with half-staged activations
produces ~0.1 KL by itself; the two F32 families stay near 1e-2. Removing any
one family leaves 0.11-0.26. The model amplifies small early perturbations
(three dense blocks alone give 0.1).

## Reference engine envelope (`envelope.json`)

`glm53_oracle` (llama.cpp `e845373ff`, the qual-v1 producer) serial and
`--batch` over each second turn plus its continuation, compared at the same
positions. Native Exact matches llama.cpp serial (P2/P4 gates). The oracle's
batch covers the continuation too, so the prompt end is the closest analog.
H5 exceeds the oracle's 4,096-ID limit.

| Case | llama.cpp batched vs serial: prompt end / worst / flips | Native Fast cold 512 vs Exact: prompt end / worst / flips |
|---|---:|---:|
| H1 | 1.2e-3 / 4.2e-2 / 1 | 1.6e-3 / 4.1e-2 / 0 |
| H2 | 2.1e-2 / 1.7e-1 / 4 | 2.1e-2 / 1.1e-1 / 1 |
| H3 | 3.0e-5 / 2.3e-2 / 0 | 1.7e-4 / 3.6e-2 / 0 |
| H4 | 1.9e-4 / 5.8e-3 / 0 | 4.3e-4 / 2.0e-2 / 0 |
| H6 | 1.6e-2 / 1.6e-2 / 0 | 3.6e-3 / 2.7e-2 / 0 |

## Exact lineage cost (`lineage-cost.json`, no API validation)

Fresh-session prompt prefill (allocation excluded, final logits included;
both lineages' dense and sparse pipelines warmed first), F-E-E-F per length,
512-row chunks:

| Prompt | Fast | Exact | Ratio |
|---:|---:|---:|---:|
| 512 | 2.60 s (197 tok/s) | 12.68 s (40.3 tok/s) | 4.9x |
| 2,048 | 10.46 s (196 tok/s) | 51.41 s (39.8 tok/s) | 4.9x |
| 4,096 | 22.04 s (186 tok/s) | 103.81 s (39.5 tok/s) | 4.7x |
| 64-token suffix after a reused prefix | 0.80-0.86 s | 1.60-1.66 s | 1.9-2.0x |

## Status and options

Under the rule frozen with the design, a natural failure opens a Fast fix,
with attribution first; Exact by default is considered only if Fast cannot
be fixed, with its measured cost. Attribution and the reference envelope say
the divergence belongs to half-staged batched prefill on this model, not to a
defect in one stage, and the reference engine does not meet the 2e-2 policy
either. Options, none taken:

- **A. Keep Fast; replace the one-prompt calibration with a reference-anchored
  envelope** (for example, Fast vs Exact no worse than llama.cpp batched vs
  serial on a frozen holdout, plus a regret bound on flips). A bound change:
  needs an explicit decision.
- **B. F32-staged batched prefill:** keep batching, stage activations in F32
  for the mat-mat families (F32-tile variants exist for some dtypes, not for
  the Q6_K/Q8_0 projections or the grouped IQ2_S/IQ3_S experts). The two F32
  families' ~1e-2 suggests the size of the gain; kernel work and a pp cost to
  measure.
- **C. Exact by default:** 4.7-4.9x slower fresh prefill (a 13K-token agent
  prompt ~70 s -> ~5.5 min); not recommended.

The original unrelated-token reproducer stays as is (known failing).
