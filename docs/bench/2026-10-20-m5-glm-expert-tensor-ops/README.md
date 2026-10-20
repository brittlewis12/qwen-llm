# M5 GLM expert TensorOps: phase A

## Status

The pp2048 route capture and test-only GLM-shaped TensorOps A/B are complete.
The microbenchmark wins at M10 and loses at M39 and M256; because production
groups many experts in one call, this per-expert result does not justify
production promotion.

## Measurement collected

`maya-2048-prefill-cost.jsonl` is from the existing
`glm5_next_metal::packed::prefill_cost::current_half_2048_cost` stage profiler
on Maya IQ2_S, zero prefix, Fast/Half, 2,048 rows. The profiler's two ordinary
walls were 3658.99 and 3653.79 ms; the profiled call was 3894.10 ms (1.065x
their mean). The sampled stage spans sum to 3631.61 ms. These are scaled
sampled spans, not observer-neutral ordinary wall fractions. The profile
changed the encoder boundaries, so small differences in fractions should not
be read as an exact decomposition.

| Existing profiler grouping | Scaled GPU ms | Share of span sum |
| --- | ---: | ---: |
| Routed experts | 1713.85 | 47.19% |
| KDA input and output projections plus MLA projection | 1214.40 | 33.44% |
| Everything else | 703.37 | 19.37% |

The profiler's `mla_projection` parent combines MLA dense, indexer, latent and
attention work. Routed experts combine bucketization, gate/up, down and
reduction. Dispatch census is coverage evidence, not per-kernel timing.

The warm census captured 1,702 dispatches across 30 kernel names. It includes
42 each of `kernel_moe_route_bucket_slots_f32`,
`kernel_moe_swiglu_clamped_iq2_s_f32_grouped_slots_generic`,
`kernel_moe_swiglu_clamped_iq2_s_f32_grouped_slots_n32`,
`kernel_moe_down_iq3_xxs_f32_grouped_slots_generic`, and
`kernel_moe_weighted_sum_packed_f32`; eight dispatches used
`kernel_moe_down_iq3_s_f32_grouped_slots_generic`. The full ordered census,
geometry and all weight bindings are in the JSONL.

The recorded prototype-independent Metal 4 witness passed on metagross and
selected `kernel_tensor_matmul2d_f16_f32`; it compared only the existing
16x16x32 F16 witness shape against the portable kernel. It does not represent
GLM expert shapes or quantized in-kernel decoding.

## Maya pp2048 route buckets

`maya-pp2048-bucket-distribution.json` records the rows per active expert for
all 168 layer/chunk pairs: 42 routed layers over four 512-row chunks. The
percentile population is 40,214 positive expert bucket counts. Active experts
per layer/chunk ranged from 199 to 287 (mean 239.37; median 232).

| Bucket rows | p10 | p50 | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| Active expert rows | 1 | 10 | 39 | 333 |

| Production path | Active buckets | Tiles | Full tiles | Tail tiles |
| --- | ---: | ---: | ---: | ---: |
| 16-row path (bucket <17) | 27,412 | 27,412 | 871 (3.18%) | 26,541 (96.82%) |
| 32-row path (bucket >=17) | 12,802 | 21,267 | 8,760 (41.19%) | 12,507 (58.81%) |

The tile counts sum across all captured layer/chunk buckets; the raw JSON
contains each per-expert count and per-layer active-expert count per chunk.
Counts conserve `512 * top_k` routed rows per full chunk and layer.

## GLM-shaped expert microbenchmark

`maya-iq2s-expert-tensor-ab.json` uses Maya's first routed layer and its GGUF
gate/up dimensions `[hidden=4096, ffn=2048, experts=288]`, IQ2_S for both
banks. Each arm receives the same deterministic F32 input rows, rounded to
F16 by the kernels; the TensorOps path decodes the IQ2_S weight tile in-kernel
to threadgroup F16 and reduces K tiles in F32. The production arm is the
existing grouped clamped gate/up encoder, with its 16/32-row split at M=17.
Each case has five interleaved A/B repetitions after warmup.

| M | Production ms / TFLOP/s | TensorOps ms / TFLOP/s | Max abs | Relative RMS |
| ---: | ---: | ---: | ---: | ---: |
| 10 (measured p50) | 0.7556 / 0.444 | 0.5609 / 0.598 | 1.67e-6 | 1.01e-6 |
| 39 (measured p90) | 1.2335 / 1.061 | 1.3098 / 0.999 | 2.86e-6 | 1.00e-6 |
| 256 | 1.0454 / 8.217 | 3.3561 / 2.560 | 3.16e-6 | 9.99e-7 |

The TensorOps path wins only at M=10 in this single-expert-call microbenchmark
(about 1.35x); it loses at M=39 (about 1.06x slower) and M=256 (about 3.21x
slower). This does not establish a layer
speedup: production groups all active expert buckets in one dispatch, while
the prototype measures one expert per call, and a layer/chunk averages about
239 active experts. Do not merge p50 buckets into M39-or-larger buckets for
this TensorOps kernel; those measured sizes lose. To exploit M10 buckets,
the next prototype should put many small expert buckets into one TensorOps
dispatch grid, rather than issue one call per expert or grow token chunks.
Before promotion, verify that expert output is independent of bucket size and
that Fast mode remains reproducible across snapshot restores. No production
behavior changed.

## Reproduction

The profile used the existing release test and the leased GPU:

```sh
QWEN_MODEL_SEARCH_PATH=$HOME/models \
QWEN_METAL_LEASE_WAIT=1 \
GLM53_GGUF=$HOME/models/glm-5.3-flash-maya-m/GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf \
GLM53_STAGE_OUT=docs/bench/2026-10-20-m5-glm-expert-tensor-ops/maya-2048-prefill-cost.jsonl \
cargo test -p qwen-llm --release glm5_next_metal::packed::prefill_cost::current_half_2048_cost -- --ignored --nocapture
```

Bucket capture and the expert A/B use the same fixture variables and release
lease:

```sh
QWEN_MODEL_SEARCH_PATH=$HOME/models \
QWEN_METAL_LEASE_WAIT=1 \
GLM53_GGUF=$HOME/models/glm-5.3-flash-maya-m/GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf \
GLM53_STAGE_OUT=docs/bench/2026-10-20-m5-glm-expert-tensor-ops/maya-pp2048-bucket-distribution.json \
cargo test -p qwen-llm --release glm5_next_metal::packed::route_bucket_distribution::maya_iq2_s_pp2048_bucket_distribution -- --ignored --nocapture

QWEN_MODEL_SEARCH_PATH=$HOME/models \
QWEN_METAL_LEASE_WAIT=1 \
GLM53_GGUF=$HOME/models/glm-5.3-flash-maya-m/GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf \
GLM53_STAGE_OUT=docs/bench/2026-10-20-m5-glm-expert-tensor-ops/maya-iq2s-expert-tensor-ab.json \
cargo test -p qwen-llm --release glm5_next_metal::packed::expert_tensor_ops::maya_iq2_s_glm_shaped_expert_tensor_ab -- --ignored --nocapture
```

The witness used `QWEN_METAL_LEASE_WAIT=1 QWEN_REQUIRE_METAL_TESTS=1` with
`metal::tests::tensor_ops_matmul2d_witness_selects_and_matches_portable`.
