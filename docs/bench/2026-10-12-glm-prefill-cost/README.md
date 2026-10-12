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
