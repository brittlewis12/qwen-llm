# GLM-5.3-Flash P4 Sparse Selection (2026-10-04)

Native sparse DSA selection past visible length 2052 (dense attention below it),
packed prefill in fast lineage with 512-row chunks (sparse rows in 64-query
microbatches) and serial decode, commit `258055bd` (clean), against llama.cpp on
the same artifact (unsloth `UD-IQ3_XXS`, `glm5-next/` view, pages warm). M4 Max
128 GB, AC power, no foreign GPU jobs (Spotlight indexing was active), blocks in
A-B-B-A order, 3 timed reps each. pp4096 crosses the frontier inside the fifth
512-row chunk; tg128 at depth 4096 decodes entirely in the sparse range.

| Block | Engine | pp4096 (t/s) | tg128 (t/s) | tg128 @ d4096 (t/s) |
|---|---|---|---|---|
| A1 | qwen-llm | 178.60 +/- 5.63 | 27.87 +/- 0.02 | 20.79 +/- 0.00 |
| B1 | llama.cpp | 177.31 +/- 0.66 | 22.05 +/- 0.64 | 20.07 +/- 0.46 |
| B2 | llama.cpp | 177.44 +/- 0.34 | 22.46 +/- 0.89 | 19.91 +/- 0.66 |
| A2 | qwen-llm | 182.71 +/- 0.95 | 27.73 +/- 0.03 | 20.74 +/- 0.12 |

- Prefill: parity at pp4096 (+1% to +3%), up from 93-98% of llama.cpp at pp512
  in P3.
- Decode: +24% at depth 0; +3% to +4% at depth 4096. Native decode loses 25%
  from depth 0 to 4096 against llama.cpp's 10%: per token, sparse selection
  adds about 4.7 ms over dense decode just below the frontier
  (`sparse_decode_cost_across_the_frontier`), mostly the online selected
  attention (one simdgroup per head streaming 2051 rows), and dense attention
  over the first ~2k rows adds about 7 ms over short context. P6 targets both.

Numerics (`glm5_next_metal::tests`, bounds frozen before observation):
- The native 32-head scorer reproduces llama.cpp's lightning-indexer scores
  bitwise on its captured inputs (sparse-v1, positions 2050-2060, all 11 MLA
  blocks); native selection equals llama.cpp's set in all 110 sparse cases.
- Decode across the frontier: top-1 46/46 with zero choice regret, worst KL
  4.2e-5 (at most 1.8e-6 at sparse positions).
- Exact packed prefill equals serial decode bitwise across the frontier for
  every chunking tested; fast packed stays within KL 4.8e-3 of llama.cpp at
  position 2092 and 1.7e-6 of exact at 2400.

Commands:

```sh
QWEN_METAL_LEASE_WAIT=1 target/release/qwen-bench suite -m "$MODEL" --pp 4096 --tg 128 -d 0 --runs 3
QWEN_METAL_LEASE_WAIT=1 target/release/qwen-bench suite -m "$MODEL" --tg 128 -d 4096 --runs 3
~/code/llama.cpp/build-glm5/bin/llama-bench -m "$MODEL" -p 4096 -n 128 -d 0 -r 3 -ngl 99 -o json
~/code/llama.cpp/build-glm5/bin/llama-bench -m "$MODEL" -p 0 -n 128 -d 4096 -r 3 -ngl 99 -o json
```

llama.cpp binary as in `2026-10-04-glm53-p2-baseline` (build `e1425c0be`,
n_batch 2048, n_ubatch 512). Raw rows: `qwen-A*-d*.json`, `lcpp-B*-d*.json`.
