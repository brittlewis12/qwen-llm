# Native GLM Row-Owned Pair: Below The Material Ceiling

## Decision

Keep this fixed row-owned native Q4 design off main and do not fund its model
adapter, quality campaign or production promotion. It is bit-correct and faster
than the maintained C operator on both synthetic distributions, but misses the
predeclared 5-6 ms actual-shape screening target and loses materially to N64 on
balanced routes. Do not continue a tile/barrier/geometry sweep from this result.

This closes the fixed design as the next ambitious application bet. It does not
establish that all native-codec or gate/up headroom is exhausted. The qualified
C production default remains unchanged.

## Design And Frozen Inputs

The [exact operand gate](2026-10-31-glm-row-owned-operand-gate.md) precedes this
native complete-pair experiment. Four SIMD32 cohorts each own 16 routed rows and
64 outputs. Each active cohort executes two ordered, non-relaxed
`matmul2d(16,64,32)` calls per K64, using cooperative Half activations, threadgroup
Half weights and an F32 accumulator. Only weights occupy the 16 KiB ping-pong
LDS. Empty cohorts still load weights and join every barrier.

The native Q4 loader shares packed payload/header work and caches superblock
metadata. Activation offsets are resolved outside K. Gate/up remain separate,
with raw F32 projection and incumbent clamped epilogues. There is no activation
or output LDS, expanded weight bank, new precision mode or production selector.

- Clean source: `4bb008c42b256a0906481e8eb2fd4f39520e6976`.
- Canonical release libtest SHA256:
  `14344c92c22237a6f78e7062a2e624d40cb0c2c771087fcc29e86f727bb2aaff`.
- Ordinary fat LTO, one codegen unit, opt-level 3, default features; SDK 26.2,
  Metal 32023.864, Metal 4.0. CPU build is unleased.
- Actual device: M5 Ultra. Native RowOwned PSO is SIMD32, maximum 128 threads,
  zero static plus 16,384 dynamic LDS bytes. C/N64 report SIMD32, maximum 1024
  threads and 8/16 KiB dynamic LDS respectively.

Pipeline thread limits are not compiled register counts or measured occupancy.
No ISA, spill, neural-core utilization or DRAM attribution is claimed.

## Correctness And Resources

Three exact ignored entries pass: canonical coefficient bits (0.26 s), raw/final
pair bits and guards/preflight (0.63 s), and separately authorized actual-shape
timing (9.14 s). The compiled small-timing entry is not executed.

The canonical suite includes adversarial metadata/nibbles and scales, multiple
superblocks, banks, offsets and row tails. Raw gate and raw up F32 outputs match
independent C/N64 references before clamping; complete outputs match afterward.
Bounded F64 samples, converted Half bits, guards, immutable inputs, preflight
non-mutation, invalid counts and invalid routes pass. Both actual-shape route
distributions repeat raw/final correctness checks before timing.

The full H4096/F2048/E288/top8/R2048 fixture reuses the existing three output
buffers. Live pricing/admission succeeds with this explicit ledger:

```text
Metal allocations:       3,173,744,640 bytes
CPU/upload/readback:     1,762,112,000 bytes
Additional reserve:        536,870,912 bytes
Combined:                5,472,727,552 bytes = 5.096875 GiB
```

The cap is 8 GiB; ordinary validation fixtures stay below 1 GiB. No owned buffer
or expanded oracle is added. This is not a measured peak-residency or driver bound.

## Direct Complete-Pair Timings

Each comparison records four warmups and eight uninterrupted ABBA/BAAB samples:
32 timed samples and 16 warmups total. Whole preflight, input conversion,
bucketing, gate, up and checked completion are included. Oracles, census, hashes
and JSON remain outside each continuous block. Timing disables Metal validation;
correctness entries enable it. Canonical acquisition is fail-fast under actual
external timeouts, with no wired bypass or model load.

GPU-time medians; negative latency change means RowOwned is faster:

| Synthetic Routes | Reference | Reference ms | RowOwned ms | Latency Change |
| --- | --- | ---: | ---: | ---: |
| Balanced | C | 10.128458 | 9.167896 | -9.4838% |
| Balanced | N64 | 7.849313 | 9.165875 | +16.7730% |
| Skewed | C | 11.128542 | 9.405188 | -15.4859% |
| Skewed | N64 | 9.819958 | 9.388396 | -4.3947% |

Both four-sample halves retain each comparison's direction. Maximum absolute
half-to-half drift is 0.3373% GPU and 0.4366% host. Host medians agree on all four
directions: C -9.1263% / N64 +16.2683% balanced, C -15.2777% / N64 -4.1271%
skewed. C and N64 each have their own direct RowOwned comparison, not one direct
C/N64 comparison.

Balanced issued work is identical across all arms: 618,475,290,624 FLOPs versus
549,755,813,888 useful FLOPs. On skewed routes, row cohorts reduce N64 issued
work from 858,993,459,200 to 588,410,519,552 FLOPs, or 31.5%, while GPU latency
falls only 4.3947%. N64/RowOwned logical weight-panel coverage remains equal.
Issued arithmetic and logical bytes are not latency or measured DRAM traffic.

## Interpretation And Next Gate

Removing activation/output LDS delivery recovers time versus the rejected K64
design, but this is not a causal comparison against that older run. The direct
decisions rely on current C/N64 controls. The simpler N64 reference remains
better balanced; row ownership has only a modest skewed advantage over it.

Routes and compact native weight recipes are synthetic, not captured model
traffic. Application-exclusive fraction, real-route conversion, model quality,
cache/DRAM behavior, register pressure and occupancy remain unknown. Numerical
evidence uses the frozen validators/reports; unlike the standalone operand gate,
these packets do not retain physical projection dumps for independent replay.

The old `current_half_2048_cost` packet is not a qualified unchanged CURRENT
profile: cold profiler/CPU/report bounds and product projection thresholds are
not established. Do not execute it or modify historical fixtures to manufacture
a denominator. Prepare a narrowly bounded post-C current-stack ranking packet
before another large implementation commitment; coarse parent spans are not an
exclusive affected fraction.

## Retained Evidence

Private build/validation/timing logs, all samples and independent recomputation
are retained. Public raw-report SHA256 identifiers:

| Report | SHA256 |
| --- | --- |
| Canonical Half bits | `f1a59d356630bd1e52803837ad4f342995e21bbea1a25b562b4399aa67d129a0` |
| Raw/final pair and preflight | `9b0bd00339c2360c7cc3cd62bd9b3f348b5793936214f8f4256d642b845f9d9d` |
| Actual-shape timing | `043fa6ffc3f06669fba0708c8a17212ffdec66c74fee480230a6c0df253f29a7` |
