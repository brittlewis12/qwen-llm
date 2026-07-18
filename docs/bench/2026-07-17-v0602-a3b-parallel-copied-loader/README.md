# v0.602 A3B Topology-Preserving Parallel-Copied Loader

Status: **GO** with force-only exact-A3B authority. The production loader now
retains copied storage's 733 independent offset-zero resources while populating
them with the frozen four-worker schedule. It preserves loaded performance and
cuts process-cold first byte by more than half.

Frozen product source: `282b13a4d318d3162448b1a5f4e7211434ecddf2`.
Canonical packet:
`target/profiles/v0602-a3b-parallel-copied-loader-p2/`.

P1 at `74f5a1faa1000f3489a8fdfbc1e6f502968c8e25` launched no product
child. Its release correctness test passed, but the outer parser omitted the
first native-embedding line because `cargo test --nocapture` placed the exact
line after the test-name prefix. P2 authenticates all five sealed P1 artifacts,
repairs only that extraction rule, and completes the frozen packet.

## Correctness

- All `22,123,538,944` source bytes match 733 independent candidate resources.
- Resource identity, exact lengths, offset zero, Shared/DefaultCache/Tracked
  modes, frozen schedule, request order, provenance, marker, and copied ledger
  pass.
- Checked compute and blit writes both reject candidate weights.
- Packed-prefill logits, complete KV/GDN/conv state, greedy argmax, one forced
  transition, continuation logits, and continuation state are bit exact.
- P2 release correctness realizes the candidate in `866.871 ms`; the median
  candidate ready wall across all product children is `747.527 ms`, including
  `744.881 ms` of copy work.

## Loaded Noninferiority

All six fixed `AB/BA/BA/AB/AB/BA` pairs and all 12 children pass on their sole
attempt. Repetitions 3-5 are scored.

| Median of child late medians | A copied | B parallel copied |
| --- | ---: | ---: |
| Prefill | `309.05 ms` | `307.75 ms` |
| Decode, 127 calls | `1197.0 ms` | `1198.6 ms` |
| Complete request | `1506.5 ms` | `1507.8 ms` |

Every pair clears the frozen 1% gates:

- prefill A/B ranges `0.99418-1.00815x`;
- decode A/B ranges `0.99684-1.00234x`;
- request B/A ranges `0.99668-1.00266x`;
- maximum paired RSS/footprint B/A is `1.000006/1.000180x`;
- repetition-5/repetition-3 and late-range stability pass for both arms in all
  pairs; the largest late decode-TPS relative range is `0.5502%`.

## Fresh Output-128 Result

All 12 fresh children emit the same output, exactly 128 tokens and 127 target
transitions. B wins both first byte and process exit in 6/6 pairs and 3/3 in
each order stratum.

| Endpoint | A copied | B parallel copied | Paired result |
| --- | ---: | ---: | ---: |
| Process start to first byte | `2477.32 ms` | `1199.75 ms` | `2.06292x` median |
| Process start to exit | `3880.71 ms` | `2625.87 ms` | `1.47929x` median |
| Runtime plus model load | `2094.00 ms` | `811.78 ms` | `0.38742x` B/A |
| Model-ready TTFT | `379.84 ms` | `384.21 ms` | `1.01225x` B/A paired |
| Generation wall | `1200.68 ms` | `1201.76 ms` | `1.00068x` B/A paired |

Paired runtime/model-load saving is `1283.493 ms` median. First-byte AB/BA
medians are `2.06078/2.06478x`; exit AB/BA medians are
`1.48013/1.47847x`. Runtime savings are `1281.402/1285.583 ms` by stratum,
and B/A runtime ratios are `0.38687/0.38759x`.

The separate product-objective tradeoff is small but systematic. B's first fresh
prefill is slower in all six pairs by `2.212-8.852 ms`, paired median
`+4.768 ms`; paired TTFT regresses `4.659 ms`. Model-ready complete request is
slower in 5/6 pairs by paired median `+5.481 ms`, and final-output-to-exit wall is
slower 6/6 by `+22.159 ms`. The systematic first-request prefill/TTFT regression
in this frozen fresh-process cell is not observed in late post-warmup loaded
measurements. The packet does not localize the cause, and the large process-cold
endpoint win cannot erase the tradeoff from the separate model-ready objective.

Maximum fresh paired RSS/footprint B/A is `1.000035/1.000322x`. Every cache and
child interval records zero Pageouts, Compressions, Swapouts, and swap-occupancy
growth. Fresh children have zero major faults and all children have zero block
input. Loaded major faults remain the known advisory `92/99` harness floor.

## Interpretation

The floor transfers almost exactly. v0.599 measured `742.676 ms` candidate ready
wall and projected `2.06601x` first byte; production product medians are
`747.527 ms` and `2.06292x`. The remaining model construction and request work
behaves as projected.

Together v0.598-v0.602 support the following mechanism boundary:

1. The frozen topology-preserving parallel-copy implementation removes about
   1.28 seconds of process-cold work.
2. Keeping independent offset-zero resources preserves copied loaded speed.
3. The giant-resource/nonzero-offset class is the observed discriminator for the
   v0.598 warm tax; anonymous copying alone is not sufficient.
4. Loaded decode/prefill, generation, RSS, and footprint remain effectively
   unchanged; first fresh prefill pays the explicit `4.768 ms` median tax. The
   endpoint gain is actual work reduction in model loading.

The packet does not individually isolate resource count, offset topology,
allocate-first behavior, parent-resolved slices, manual copying, or worker count.

This is the largest exact fresh-process A3B responsiveness gain currently banked.
It closes resource-topology attribution as a prerequisite: the same-topology
candidate succeeded, so no MMU/resource-shape diagnostic is needed before product
use.

## Authority

The result authorizes only explicit `QWEN_GGUF_PARALLEL_COPY=1` on the frozen
single-shard A3B asset and authenticated geometry. It does not authorize default
selection, 27B/A10B breadth, split shards, aliases, conversions, MTP variants,
storage-cold claims, retained-memory savings, serving, or asynchronous promotion.

Default admission must explicitly accept the first-request tradeoff. If absent
policy would also affect serving or concurrent loading, it must either keep those
scopes force-only or separately accept their unmeasured CPU-contention and energy
surface. The next new cold optimization is authenticated exact-topology breadth,
beginning with dense 27B before the much larger split A10B asset. Adversarial
design and implementation review used
`cx ask` sessions `019f7131-3b49-7773-a867-341b2c66dd2f` and
`019f7196-d46b-71e1-9ae4-7804347f44a2`; sealed-result review used
`019f71e6-4700-7132-af49-fe7bc55cc84d`.
