# Current Half N2048 costs

One Maya Half chunk at actual workspace2048 after production latent MMA
promotion, using the existing stage profiler and bound-dispatch census.
Warm-only census, two ordinary runs, one profiled run on identical input;
finite, bit-identical endpoint logits required. Load and allocation excluded.

Existing spans combine expert internals, KDA internals and MLA mixer work.
Their costs are parent attribution, not proof of any leaf bottleneck or
fractions of ordinary wall time. Record observer overhead rather than assume
it away. No new recorder, numeric threshold or production dispatch change.

## Measured Budget

`maya-2048.jsonl` completes with identical finite endpoint logits in the warm,
two ordinary and profiled calls. Ordinary wall: 8984.64 / 9768.36 ms; profiled
wall: 9850.63 ms, 1.051x ordinary mean. Ordinary repeats themselves vary 8.7%,
so the profiler is a rough ranking aid, not an observer-neutral timing proof.

| Existing parent span | Scaled GPU ms |
| --- | ---: |
| Routed experts | 4580.35 |
| KDA | 2345.20 |
| MLA projection/mixer | 1287.52 |
| Router | 519.28 |
| Shared expert | 440.36 |

The warm census confirms batched Q6 N64 projection kernels for KDA, production
Q6 MMA for the latent projections, and 42 generic F32 router projections at
K4096/E288/N2048. The strict router path currently selects exactly N128/N512;
that is prior measurement coverage rather than primitive geometry. Test the
existing strict kernel at the actual larger width before writing a new KDA or
expert kernel. Router's parent includes sigmoid/top-k, so its full 519 ms is
not a projection-only opportunity.

The packet uses the existing profiler and records the full dispatched names,
geometry and weight descriptors. It adds no production observer or allocation.

## KDA Component Decision

After router promotion, schema v2 splits packed KDA into five components using
the existing recorder. Decode keeps its original parent stage. Ordinary packed
execution retains the same encoder, three KDA kernels, arguments and order;
shared validated helpers preflight all three pipelines before dispatch. Profiling
alone adds four encoder boundaries per KDA block, with no activation buffers.

`maya-2048-kda-components.jsonl` records 34 KDA blocks, 170 component spans and
984 timestamp samples within the existing 2048-sample capacity. Ordinary wall
7684.56 / 8031.44 ms; profiled 8631.17 ms, 1.098x ordinary mean. Ordinary repeats
vary 4.5%. Do not infer an optimization from timing differences versus v1: these
are separate runs and the router policy also changed.

| KDA component, across 34 blocks | Scaled GPU ms |
| --- | ---: |
| Input projections | 1244.76 |
| Preparation | 105.13 |
| Recurrence | 340.05 |
| Output normalization/gating | 12.71 |
| Output projection | 392.54 |

Projections dominate this diagnostic; recurrence is only about 340 ms of an
approximately 7.9-second ordinary chunk. The observer adds roughly 9.8% wall
overhead and can distort component shares, so these are not ordinary-runtime
fractions. A new chunkwise per-key-decay recurrence is not justified by the
2345 ms parent budget. Park that redesign rather than start a larger campaign.

All model endpoint logits remain bit-identical. Four KDA tests pass under leased
Metal validation, including the independent CPU chunk/decode oracle and an
ordinary-versus-sampled test checking all seven written buffers, incoming
nonzero state, and rows 1/37. No GPU kernel arithmetic changes are made.

The next bounded primitive-reuse candidate is normalized F32 mHC projection
through the existing grouped MMA template, with explicit Fast-only permission
and output-channel-tail guards. Its entire pre-stage budget is only about 263 ms,
so keep the experiment small and stop if reduced parallelism defeats reuse.
