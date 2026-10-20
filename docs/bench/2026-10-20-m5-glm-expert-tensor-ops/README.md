# M5 GLM expert TensorOps: phase A

## Status

The first Maya IQ2_S pp2048 profile shows routed experts consume a large
share of the profiled chunk, but the requested bucket-distribution capture
and GLM-shaped tensor-op A/B benchmark were not completed. There is no
measured crossover or performance recommendation in this packet. Do not
promote a tensor expert path from the witness result below.

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

## Not measured

This run does not provide per-layer/per-chunk rows per active expert, p10/p50/
p90/max buckets, or full/tail tile shares. It also does not provide pp512 or
pp4096 stage profiles, expert-shape tensor-op timings, five-repeat interleaved
A/B measurements, or tensor-vs-production numerical deltas. Therefore it
cannot answer whether TensorOps win for Maya's real bucket sizes. No test-only
expert kernel was added and no production behavior changed.

Before considering promotion, capture the per-layer routing counts for every
512-row chunk at pp512, pp2048 and pp4096, then compare the production grouped
expert path with the quantized-weight TensorOps prototype at observed p50,
p90 and full-chunk bucket sizes. Re-verify that expert outputs are independent
of bucket size and that Fast-mode results remain reproducible across snapshot
restores. A future grouping experiment should first establish whether merging
small expert buckets can move enough rows into a measured winning regime.

## Reproduction

The profile used the existing release test and the leased GPU:

```sh
QWEN_MODEL_SEARCH_PATH=$HOME/models \
QWEN_METAL_LEASE_WAIT=1 \
GLM53_GGUF=$HOME/models/glm-5.3-flash-maya-m/GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf \
GLM53_STAGE_OUT=docs/bench/2026-10-20-m5-glm-expert-tensor-ops/maya-2048-prefill-cost.jsonl \
cargo test -p qwen-llm --release glm5_next_metal::packed::prefill_cost::current_half_2048_cost -- --ignored --nocapture
```

The witness used `QWEN_METAL_LEASE_WAIT=1 QWEN_REQUIRE_METAL_TESTS=1` with
`metal::tests::tensor_ops_matmul2d_witness_selects_and_matches_portable`.
