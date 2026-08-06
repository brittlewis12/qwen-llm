# DeepSeek V4 Grouped-IQ2 Broadcast Gate

Status: `KILL_BROADCAST_SUBSTITUTION`. The exact dispatch-neutral SIMD
weight-broadcast candidate is removed from live source. It authorizes no asset
run, product change, or same-family rescue tuning.

## Question

The exact-route V2 phase ceiling assigns roughly 527 ms across the 25 grouped
IQ2 gate/up/SwiGLU layers. The first candidate reduced launched lane work by
packing FFN rows but regressed and remained formally inconclusive. Test one
materially different candidate that keeps every deployed threadgroup and
dispatch while removing only threadgroup publication and barriers.

## Candidate

Keep one FFN row, one assignment panel, and one 32-lane SIMD group per
threadgroup. For every 32-wide K chunk:

- each lane performs the unchanged scalar IQ2_XS gate and up dequantization;
- each dequantized F32 bit pattern remains in its producing lane;
- all lanes execute two scalar `simd_shuffle` operations for each of 32 K
  values; and
- active assignment lanes read the same input value and retain the literal
  `bidx -> k0 -> j` gate/up accumulation order.

This replaces the deployed 64-float threadgroup tile and both barriers per K
chunk. It retains the same 25 dispatches, grid, route schedule, slots, clamp,
SwiGLU expression, output ownership, logical input and weight reads, and scalar
dequantizer. It makes no physical cache-traffic claim and does not touch down.

The exact Rust encoder and Metal kernel source slices are sealed by SHA-256 in
the executed profiler:

- encoder: `080c35f7eb3ae08dc0c8a6c60cb8803d4d6c5bffbe67d82f3c832ffe7bc58c67`;
- kernel: `15649c66d7d6c71ab37c1f690e7257b9988034e83378ad1f194d078698a4b76e`.

## Correctness Gate

Active release tests require bit-exact candidate output against the deployed
kernel for mixed N=1/12/31/32/33/64/128 schedules and panel counts
1/2/3/4/5/8/9/16/17/31/32/33/63/64/65/96/97/127/128. They cover continuation
panels, row-unique weights, token-unique signed inputs, clamp, repeat identity,
offset guards, and immutable inputs, weights, and slots.

The frozen production-shape packet repeats exact H=4096, F=2048, clamp-10
validation before and after timing in both cache regimes. Candidate and baseline
digests are identical:

- disjoint: `bbf43456479d05e4097794dba99132df46588f25282f904917caf5ab68ad4cb2`;
- warm: `850ccf1105481a836ef0425e005c96257d8878a84d6719c5189a2740f98f5211`.

The packet also validates complete poison overwrite, finiteness, guards,
routes, repeatability, the full input digest, initialized weight-bank digests,
strict command success, and exactly 25 baseline and candidate dispatches.

## Frozen Performance Gate

Use the canonical 25x768 current-asset route fixture with 25 disjoint gate/up
bank pairs and one all-expert warm pair. Every sample submits and waits for 25
serial per-layer command buffers. After exactness and one untimed balanced
conditioning block, retain six blocks of:

```text
BD / CD / CW / BW / BW / CW / CD / BD
```

This yields 12 samples per baseline/candidate x disjoint/warm cell. Keep every
sample and permit no filtering or rerun. GPU timestamps must be finite, positive,
and strictly nonempty. Wall measurements are retained as diagnostics only.

All four GPU cells must have `(max - min) / max <= 5%`. For each regime:

```text
Q = max(5 * max(baseline_range, candidate_range), 0.492790 ms)
L_s = baseline_s - candidate_s - Q
threshold_s = max(158.3 ms, 0.25 * baseline_s)
```

Nearest-rank median is the sixth sorted sample and p95 is the maximum. Median
and p95 must both satisfy `L_s >= threshold_s`. Invalidity or instability is
`INCONCLUSIVE_HOLD`; a stable miss kills exactly this substitution; a full pass
only authorizes a separately reviewed current-asset A/B.

## Result

All four authoritative GPU cells are stable, and every candidate sample is
slower:

| Regime | Baseline median / p95 | Candidate median / p95 | Candidate / baseline median |
|---|---:|---:|---:|
| Disjoint GPU | 527.743 / 540.779 ms | 2,406.168 / 2,429.985 ms | 4.56x |
| Warm GPU | 527.151 / 546.902 ms | 2,406.267 / 2,456.642 ms | 4.57x |

GPU drift is 2.607086%/1.010918% for disjoint baseline/candidate and
3.699784%/2.092158% for warm baseline/candidate. Conservative median/p95 lower
bounds are deeply negative:

- disjoint: `-2001.251/-2012.031 ms`, with `Q=122.826 ms`;
- warm: `-2136.101/-2166.724 ms`, with `Q=256.984 ms`.

Disjoint wall diagnostics are nonstationary at 24.501257%/5.687660% drift.
Warm wall diagnostics are stable at 3.683199%/2.489513% and independently show
the same large regression. Wall data has no authority in the preregistered
GPU-only classifier.

## Decision

Record contractual `KILL_BROADCAST_SUBSTITUTION`. Remove the candidate encoder,
Metal kernel, active differential additions, and ignored profiler. Do not rerun,
tune shuffle width, spend an asset gate, or reinterpret this result as a generic
SIMD-shuffle rejection.

The bounded lesson is only that, at this production geometry on the qualified
M4 Max, replacing threadgroup-staged scalar IQ2 values and barriers with two
per-element scalar SIMD broadcasts is decisively worse despite unchanged
dispatch count. This does not reject shuffles in other geometries, reuse across
multiple rows, matrix/vector-native designs, or dispatch-reducing fusion. The
packet does not identify one unique microarchitectural cause.

The grouped-IQ2 scalar retuning lane is now exhausted under the current
asset/device contract. Before another packed kernel, recover the already-captured
`BeforeAttentionBody` and `AfterAttentionOutput` residual intervals from the
accepted four-pass pre-expert packet. Code only after a candidate-touched subset,
not merely its containing phase, clears the existing 158.3 ms floor.

## Provenance

- Base revision: `b2e770866fe4fc9de3aa428e42d2038b60dab469`.
- Device: Apple M4 Max.
- In-test/process elapsed: 165.74/165.85 seconds.
- Maximum RSS: 9,179,938,816 bytes; swaps: zero.
- Pre-run source diff SHA-256, preserved compressed:
  `46db08a4699ae0da59239706fbd2ae6b7c4de9f0586115e389cc188d28142a29`.
- Compressed pre-run source artifact SHA-256:
  `2680e4b3819b1793326f92a1f067a9b68b16b769c6d04dfb16f4f943669e9ec1`.
- Canonical zero-context reconstructing diff SHA-256:
  `67b666504a2b685b4a2c357992578b00d9a7c9968139f4d1b535c7c5da29a8c1`.
- Raw log SHA-256:
  `fa4cf4f44bdaa32a630d72e1932e22044e0f52e535bd788f90d028bef17cb668`.
- Executed release test binary SHA-256:
  `0e498ecb541bf393e51d84f23e80bc837818edc933322db024bbe8be3b75a963`.
- Final retained release test binary SHA-256:
  `2d3e2bc7478fefbb98890fa29bf6ce03a15404609006c2f8f40f1f8552c5a24b`.
- Retained source diff SHA-256, empty by construction:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Checksum transcript: `checksums.log`.
- CX review and adjudication: `019fd685-acb3-7890-83c9-192cbea48c6e`.

Executed command:

```bash
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::prefill::tests::\
profile_grouped_iq2_broadcast_production_shape \
  -- --ignored --exact --nocapture --test-threads=1
```
