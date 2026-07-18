# v0.603 Dense-27B Topology-Preserving Parallel-Copy Floor

Status: **GO** for one separately preregistered force-only dense-27B loader
pilot. The frozen parallel-copy implementation preserves the exact copied
resource topology while reducing host materialization wall by about 965 ms.

Frozen source, build, and runtime:
`e6e964ffad896a480ac280de8ff07634cdd344bd`.
Canonical packet:
`target/profiles/v0603-dense27b-parallel-copied-floor-p1/`.

## Scope

This packet measures warm-filesystem-cache, fresh-process host materialization
of the exact single-shard Qwen3.6-27B Q4_K_M asset. It is a floor, not a product
loader result. It does not measure model construction, inference correctness,
loaded performance, first byte, storage-cold behavior, serving, concurrent
loading, or energy efficiency.

The authenticated profile contains 851 all-direct requests over
`16,806,250,496` copied bytes. Both arms retain 851 independent exact-sized
Shared/DefaultCache/Tracked resources and 851 canonical offset-zero bindings.

## Integrity And Correctness

- All 12 children are unique sole attempts in the frozen
  `AB/BA/BA/AB/AB/BA` order and exit successfully without retry or overlap.
- Every child verifies all source bytes, resource identities, lengths, modes,
  bindings, request associations, profile fields, and schedule fields.
- Arm B uses the literal four-worker source-order schedule with cuts
  `136,377,618`; arm A uses the production copied primitive.
- Every cache and child interval records zero Pageouts, Compressions, Swapouts,
  swap-occupancy growth, compressor stored/occupied growth, and block input.
  Timer-local major faults are zero.
- The complete 29-entry artifact inventory, attempts ledger, decision, and
  completion seal verify.

## Result

| Endpoint | A copied | B parallel copied | Paired result |
| --- | ---: | ---: | ---: |
| Ready wall | `1519.5535 ms` | `557.7565 ms` | `965.0405 ms` saved |
| Ready throughput | `11.05999 GB/s` | `30.13187 GB/s` | `2.7244x` |
| Endpoint CPU | `1519.376 ms` | `2210.4965 ms` | `1.455x` B/A |
| CPU core-equivalents | `0.999888` | `3.965857` | descriptive |

The ready-wall values are medians of the six arm observations. The authoritative
paired statistics are:

- median saving `965.0405 ms`;
- median B/A `0.365706589x`, or about `2.7344x` A/B;
- 6/6 wins;
- AB median saving `966.018 ms`, with 3/3 wins;
- BA median saving `964.063 ms`, with 3/3 wins;
- maximum paired RSS B/A `1.000027241x`;
- maximum paired footprint B/A `1.000056335x`.

Do not derive the paired ratio or saving from the arm medians. Pairwise reduction
precedes median aggregation, so the two summaries are close but not identical.

Arm B spends a median `554.609 ms`, about 99.4% of its endpoint, in the copy
phase. Its median allocation, source/safety/schedule, and binding phases are only
`2.7705`, `0.240`, and `0.0295 ms`.

## CPU Accounting

The wall win is purchased with CPU parallelism, not less aggregate CPU work.
Median endpoint CPU rises about 45.5%, from `1519.376` to `2210.4965 ms`, while
the candidate uses about four CPU core-equivalents during its shorter endpoint.
Raw process counters show about 10.8% fewer retired instructions but about 27.7%
more cycles. Billed and serviced energy counters are zero, so no energy claim is
available.

Complete-process wall is not product-load evidence. It includes startup, an
exhaustive 16.8 GB post-timing payload audit, and teardown that production does
not perform. A product pilot must separately report endpoint and complete-process
CPU accounting rather than transferring these values as a load result.

## Interpretation

The exact frozen bundle transfers from A3B to dense 27B at the materialization
floor. It combines allocate-first behavior, parent-resolved source slices, manual
copying, the literal four-worker schedule, and exact copied resource topology.
The packet does not isolate worker count from the other coupled changes.

The supported causal boundary is narrower:

1. The topology-preserving parallel-copy bundle removes about 965 ms of dense
   host-materialization wall.
2. The candidate preserves the independent offset-zero resource topology needed
   to avoid the previously measured giant-resource/nonzero-offset loaded tax.
3. Memory and pressure remain unchanged under this serial floor.
4. Aggregate CPU cost rises materially and must remain visible in product
   admission and any future concurrent-loading decision.

## Authority

The result authorizes one separately preregistered force-only loader pilot for
the exact dense profile. That pilot must prove full-state bit exactness, loaded
1% noninferiority, and fresh output-128 first-prefill, first-byte, request, exit,
memory, pressure, and CPU endpoints before gaining product authority.

It does not authorize default selection, dense breadth beyond the frozen asset,
A3B inference, A10B, aliases, conversions, MTP variants, serving, concurrency,
or energy claims. Adversarial design and implementation review used `cx ask`
sessions `019f7363-fda4-72f2-b51f-882e2526c2c5`,
`019f738e-5c7f-7483-982b-34b61a440c01`, and
`019f73a4-d784-78a1-b0f2-d2eda3831805`; sealed-result review used
`019f73c8-b5c7-7ba0-bd80-aee69a2abdc2`.
