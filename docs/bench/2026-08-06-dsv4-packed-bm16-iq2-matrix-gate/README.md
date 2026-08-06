# DeepSeek V4 Packed BM16 IQ2 Matrix Gate

Status: narrow default GO. All four model-free cells passed; the first
integration wiring was killed, and its corrected successor passed an ordinary
current-asset CLI pilot before promotion.

## Question

The 25 grouped-IQ2 layers spend about 527 ms of an N=128 packed chunk in
gate/up/SwiGLU. The deployed kernel assigns one scalar dot product to each
route lane and reaches only 34.325% useful width-32 occupancy on the original
route census.

Test one genuinely matrix-shaped work unit: one SIMD group computes 16 output
rows by as many as 16 expert-major routes, with IQ2_XS values and activations
entering F32 simdgroup matrix multiplication. Gate and up remain separate;
the existing exact clamped SwiGLU follows. No down projection or full-model
asset execution is part of this packet.

## Correctness

The candidate is diagnostics-only. Its contract includes:

- exact IQ2_XS dequantization into a 16x32 F32 weight tile;
- a 32x16 F32 route tile and four F32 8x8 accumulators;
- canonical expert/token/slot ordering and unique destination ownership;
- strict source/destination map validation and partial-tile zero padding;
- a 12-byte Rust/Metal tile ABI, with `341 * 12 <= 4096` const-asserted;
- one 32-thread SIMD group and exactly 4 KiB threadgroup memory; and
- unchanged production clamp and SwiGLU arithmetic.

Reduced-K tests cover N=1/12/15/16/17/31/32/33/64/128 and assert exact scalar
gate, up, and SwiGLU bits. A separate H=4096 production-K test covers
N=1/15/16/17/128, multiple experts and BM16 tiles, exact scalar gate/up/SwiGLU
bits, candidate repeat gate/up/final bits, guards, maps, and dispatch topology.

The K-tile basis probe found and repaired one bring-up defect: the first draft
used the absolute IQ2 dequant lane as the 32-wide scratch offset, shifting
output rows after K=32. Scratch placement now uses lane parity while the IQ2
dequant lane continues through all sixteen 16-value groups.

## Co-primary Schedules

The packet carries two exact current-asset route fixtures:

| Schedule | Active layer/expert buckets | BM16 tiles | Occupancy | Padding |
|---|---:|---:|---:|---:|
| Repeated boundary | 1,542 | 2,234 | 53.7153% | 16,544 |
| Representative strategy chat | 3,154 | 3,558 | 33.7268% | 37,728 |

Both canonical fixtures were validated before acquisition. The retained tests
recompute their schema, model, prompt, route, and schedule identities. Neither
schedule could rescue the other.

## Frozen Timing Gate

Four cells are co-primary: boundary/representative crossed with disjoint/warm
weight regimes. A sample executes all 25 layers serially with production
command boundaries:

- baseline: one deployed grouped gate/up/SwiGLU dispatch per layer;
- candidate: mapped BM16 gate, mapped BM16 up, and exact SwiGLU per layer.

Disjoint cells use 25 independently initialized bank pairs whose selected
expert sets cover both schedules. Warm cells reuse one fully initialized bank
pair. Within every pair, both arms consume byte-identical weights, inputs,
maps, and schedules.

Each cell runs one untimed `(AB,BA,BA,AB)` conditioning block, then retains
`(AB,BA,BA,AB) x 3`: twelve adjacent pairs and twelve samples per arm. Metal
command GPU intervals are authoritative; wall is diagnostic. Every sample is
retained.

For paired saving `d_i = GPU_A_i - GPU_B_i`:

- the third-smallest of twelve savings is the one-sided 98.0713%
  distribution-free lower confidence bound for the population median;
- at least 11/12 savings must be positive;
- candidate p95, the cell maximum, must not exceed baseline p95;
- the lower bound must reach both 158.3 ms and 15% of baseline median; and
- all four cells must pass independently.

Each arm's full range and first/second-half medians must remain within 5%.
AB/BA and first/second-half saving medians may differ by no more than 5% of
baseline median, and both orientation and temporal strata must favor the
candidate. No five-range charge is applied to active arms; 5R remains reserved
for zero-work screens.

## Typed Outcomes

- Candidate encode/command failure, non-finite or incomplete output, guard
  corruption, scalar mismatch, repeat mismatch, or a valid economic miss:
  `KILL_BM16_F32_IQ2_XS_MATRIX`.
- Fixture, baseline, immutable-state, timestamp, completeness, stationarity,
  or order failure: `INCONCLUSIVE_HOLD` with no automatic retry.
- Four-cell pass:
  `AUTHORIZE_SEPARATELY_FROZEN_CURRENT_ASSET_EXACT_FIRST_HIGHER_PRECISION_QUALITY_GATE_FOR_REVIEWED_BM16_ONLY`.

A pass did not authorize ordinary execution. The subsequent asset gate rejected
the first integration wiring before candidate execution because its SwiGLU
views had the wrong rank; that result is recorded in the adjacent asset-gate
packet.

## Result

| Schedule / regime | Baseline median | BM16 median | Paired median saving | Third-smallest lower bound | Wins |
|---|---:|---:|---:|---:|---:|
| Boundary / disjoint | 526.369 ms | 286.336 ms | 239.993 ms | 239.881 ms | 12/12 |
| Representative / disjoint | 889.606 ms | 357.441 ms | 532.223 ms | 532.076 ms | 12/12 |
| Representative / warm | 900.744 ms | 364.512 ms | 535.497 ms | 532.585 ms | 12/12 |
| Boundary / warm | 526.388 ms | 286.389 ms | 239.975 ms | 239.871 ms | 12/12 |

Every lower bound clears 158.3 ms and 15% of baseline by a wide margin.
Candidate p95 improves in every cell. Baseline/candidate full-range drift stays
below 1.98%/3.00%; orientation spread is at most 0.0256% and temporal spread at
most 0.5045% of baseline median. Every pre/post scalar output, candidate
gate/up/final repeat, guard, map, input, weight, and source seal passes.

The representative schedule strengthens rather than weakens the result. Its
larger active-expert population raises the deployed scalar phase from about
526 to 890-901 ms, while BM16 reaches 357-365 ms. BM16 removes roughly 60% of
that realistic-diversity phase despite only 33.73% useful tile occupancy.

The sole test ran for 202.17 seconds, reached 21,635,301,376 bytes maximum RSS,
and recorded zero swaps. It used 32,647,345,664 allocated Metal bytes and first
touched 20,155,203,584 weight bytes. These are packet provenance, not ordinary
prefill memory claims.

The complete retained machine-readable result is `result.json`. Acquisition
preceded the candidate's clean repository commit, so this is strong internal
mechanism evidence rather than a product-promotion packet.

## Corrected Product Path

The first integration gate found that its rank-two gate/up views were passed to
the flat SwiGLU contract and recorded a narrow KILL before candidate execution.
The successor centralizes both BM16 projections and the flattened SwiGLU call,
then enables it only for Apple M4 Max, full N=128 chunks, and
IQ2_XS/IQ2_XS/IQ3_XXS routed layers. Other devices, dtypes, and tail chunks keep
the existing path. `QWEN_DSV4_PACKED_BM16_IQ2=0` is the explicit rollback.

An ordinary 872-token raw-prompt CLI pilot exercised six full N=128 chunks and
one 104-token scalar tail on the current asset:

| Arm | Prefill run 1 | Prefill run 2 | Mean | Mean throughput |
|---|---:|---:|---:|---:|
| Deployed | 27,069.7 ms | 27,094.5 ms | 27,082.1 ms | 32.20 tok/s |
| BM16 | 24,004.2 ms | 23,835.2 ms | 23,919.7 ms | 36.46 tok/s |

The mean saving is 3,162.4 ms, or 11.68% of prefill wall and 527.1 ms per full
chunk. That per-chunk transfer closely reproduces the model-free representative
schedule result. Both arms produced token 87339 and the same complete
129,280-value F32 logit digest,
`c62a3f1fcf9a003ba968d3f232abfa6e5126ee9882e5b1553fdd29f64a27ffe3`.

The exact primitive contract is structural rather than learned-weight-specific,
so the current-asset product result authorizes the narrow qualified default. It
does not cover another device, routed dtype combination, or chunk size.
