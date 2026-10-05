# GLM-5.3-Flash decode stage attribution (2026-10-05)

Leverage map 2026-10-05 #2. UD-IQ3_XXS, M4 Max 128 GB, commit `bee3b6ac`.

Reproduce:

```sh
QWEN_METAL_LEASE_WAIT=1 GLM53_GGUF=<shard 1> GLM53_STAGE_OUT=attribution-v1.json \
  cargo test -p qwen-llm --lib glm5_next_metal::tests::decode_stage_attribution \
  -- --ignored --nocapture --test-threads=1
```

Method: teacher-forced decode of the long qualification text at depths 64
and 4096; 4 warm-up steps, 12 unprofiled steps (wall), 12 profiled steps.
A profiled step is one command with one timestamp-sampled encoder per
stage; spans are scaled to the command's GPU time. The encoder boundaries
cost ~1.5 ms per step (command 37.07 ms vs span sum 35.61 ms at depth 64),
so shares, not the sum, transfer to the unprofiled step. The profiled step
equals the plain step bitwise (lens test, `MTL_DEBUG_LAYER=1`). Weight bytes
come from GGUF tensor names: routed experts count top-8 of 288, the
embedding one row.

| Stage (ms per token) | depth 64 | depth 4096 | weight MB | GB/s @64 |
|---|---:|---:|---:|---:|
| kda | 12.01 | 12.02 | 3,881 | 323 |
| routed_experts | 10.05 | 10.04 | 3,052 | 304 |
| sparse_attention | — | 11.60 | 0 | — |
| shared_expert | 2.62 | 2.60 | 873 | 334 |
| ffn_pre | 2.11 | 2.07 | 19.5 | 9 |
| attention_pre | 2.04 | 2.00 | 19.5 | 10 |
| mla_output | 1.67 | 1.67 | 720 | 432 |
| mla_projection | 1.49 | 1.49 | 475 | 320 |
| head | 1.03 | 1.04 | 520 | 505 |
| router | 0.82 | 0.82 | 198 | 242 |
| dense_ffn | 0.81 | 0.81 | 372 | 459 |
| sparse_select | — | 0.55 | — | — |
| dense_attention | 0.39 | — | — | — |
| sparse_query | — | 0.34 | 79 | — |
| attention_post + ffn_post | 0.40 | 0.38 | — | — |
| mla_indexer | 0.18 | 0.18 | 12 | — |
| sparse_scores | — | 0.08 | — | — |
| **unprofiled wall** | **38.98** | **51.21** | | |

Findings:

- The depth penalty is selected attention: 11.6 ms per token, 1.05 ms per
  MLA block for 2,048 selected latent rows and one query. That is a few MB of
  reads per block, so the kernel is latency- or occupancy-bound, not
  bandwidth-bound. Scoring (0.08 ms) and selection (0.55 ms) are small.
  The kernel is the family-neutral online selected attention that DS4 also
  uses.
- Weight streaming (kda, experts, shared, MLA, head, router, dense) is
  ~30.5 ms for ~9.8 GB at depth 64, or ~321 GB/s against 474 GB/s stream.
- mHC pre (RMSNorm, mix matvec, controls, collapse, norm) costs ~4.1 ms per
  token for 39 MB of weights: 90 small stages at ~46 µs each.

## v2: after the split selected attention (`a99792ba`)

`attribution-v2.json`: same method and command, with
`GLM53_STAGE_OUT=attribution-v2.json`. Packet
`../2026-10-05-glm53-split-selected-attention/` has the A/B.

| Stage (ms per token) | v1 depth 4096 | v2 depth 4096 | v2 depth 64 |
|---|---:|---:|---:|
| sparse_attention | 11.60 | **1.05** | — |
| sparse_select | 0.55 | 0.55 | — |
| sparse_query | 0.34 | 0.34 | — |
| sparse_scores | 0.08 | 0.08 | — |
| kda | 12.02 | 11.83 | 12.02 |
| routed_experts | 10.04 | 9.99 | 10.05 |
| ffn_pre + attention_pre | 4.07 | 3.92 | 4.13 |
| **unprofiled wall** | **51.21** | **40.60** | **38.58** |

- Selected attention now takes ~95 µs per MLA block. All sparse stages
  together cost ~2.0 ms per token, which is the whole remaining depth
  penalty (40.60 vs 38.58 ms).
- Other stages are unchanged. Decode is now led by weight streaming:
  kda 11.8, routed experts 10.0, shared expert 2.6, MLA 3.2, head 1.0,
  router 0.8 and dense FFN 0.8 ms. mHC pre (3.9 ms) is next.
- mHC pre per sub-block, from the `metal::mhc::tests::mhc4_pre_dispatch_costs`
  screen at GLM width with synthetic weights and 90 chained repetitions, in
  µs:
  - whole sequence 28.8;
  - controls 10.4 (one GPU thread, 20 Sinkhorn rounds through device memory);
  - Q8_0 mix 7.1;
  - RMS over 16,384 values 6.5;
  - block RMS 3.0;
  - collapse 1.7.
