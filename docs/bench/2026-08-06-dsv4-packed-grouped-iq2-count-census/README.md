# DeepSeek V4 Packed Grouped-IQ2 Count Census

Status: exact current-asset count census `GO` as operation-count evidence. Its
one bounded model-free phase-attribution authorization was consumed by the
subsequent timestamp-invalid campaign. It now authorizes no new execution,
replacement kernel, current-asset candidate, or performance claim.

The census closes dispatch aggregation structurally: each eligible layer already
uses one gate/up/SwiGLU and one down/scatter dispatch. Other schedule-derived
ratios remain operation-count ranking signals until direct phase attribution.

## Question

The accepted current-asset observer assigns a 597.107 ms mean routed-stage
envelope to the 25 production-grouped `IQ2_XS/IQ2_XS/IQ3_XXS` layers. Those
layers already execute only two expert dispatches each, but the grouped kernels
operate on 32-assignment expert tiles. The prior packet retained bucket counts,
not the exact per-expert populations needed to price padding, repeated panels,
or narrower down widths.

This checkpoint captures those populations once, binds them to the accepted
N=128 output/state identity, and then removes the one-shot asset harness.

## Capture Contract

The sole release-diagnostics execution uses:

- the pinned 97.05 GiB current asset and model content ID
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`;
- the exact accepted token pattern `[35,201,200,34]` repeated to 128 tokens,
  with little-endian token SHA-256
  `b57816bcb0d5fdf5a8e2ddc7a0afe9e57fb0ca6ffc2b849285e1635d04772843`;
- ordinary CPU routing and the qualified grouped-IQ2 policy;
- one unsampled packed execution, no continuation, no candidate, and no timed
  endpoint; and
- exact accepted logits, normalized hidden, causal, prefix, and compatibility
  digests before any census is emitted.

Committed tokens must equal the 128-token prefix exactly. The grouped layer IDs
are frozen as:

```text
0, 2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
21, 22, 23, 25, 35, 36, 37, 38, 39, 41
```

For each layer and expert, `c_e` is the exact assignment count. The capture
requires `0 <= c_e <= 128`, `sum(c_e) = 768`, and:

```text
B       = sum_e [c_e > 0]
T32     = sum_e ceil(c_e / 32)
padding = 32 * T32 - 768
```

The canonical payload hash covers the domain bytes
`qwen-llm:dsv4:packed-grouped-iq2-counts:v1\0`, then ascending little-endian
`u32` layer IDs and 256 little-endian `u16` counts per layer. JSON key order is
not part of the identity.

## Result

The one run passes every binding and emits payload SHA-256
`0ab9925350288116288794f3d7f5081595dfad4ffdb9671a9a146f27568358a6`.

| Quantity | Result |
|---|---:|
| Useful route assignments | 19,200 |
| Active layer-expert pairs `B` | 1,542 |
| Inactive layer-expert pairs | 4,858 |
| Width-32 tiles `T32` | 1,748 |
| Width-32 column capacity | 55,936 |
| Padded down columns | 36,736 |
| Useful column occupancy | 34.325% |
| Padded column share | 65.675% |

The tile-count histogram is `[1,395,100,35,12]` for experts requiring exactly
one through four width-32 panels. Therefore 1,395 of 1,542 active experts
(90.47%) fit in one panel, while continuation panels add only 206 of 1,748
tiles (11.78%). Uniformly scaling the entire 597.107 ms cohort by that ratio
would yield 70.368 ms, but uniform per-tile latency is unproven. Treat this only
as a weak operation-count ranking signal pending phase attribution, not as a
timing ceiling or KILL.

The current down kernel performs full width-32 SIMD matrix work for every tile
and masks only the final store. Exact schedule arithmetic gives these
column-work ceilings:

| Tail widths | Executed columns | Reduction from current |
|---|---:|---:|
| 32 | 55,936 | 0.00% |
| 16/32 | 35,744 | 36.10% |
| 8/16/32 | 27,456 | 50.92% |
| Exact useful width | 19,200 | 65.68% |

These are operation counts, not predicted milliseconds. Gate/up masks inactive
lane accumulation while retaining shared tile dequantization, so the 36,736
padded-column statement applies directly to the down matrix body, not to all
gate/up arithmetic.

## Decision

Retain the diagnostics-only exact count metadata, its fail-closed helper, and
active fixture validation. Remove the ignored one-shot asset harness after
archiving its reconstructing diff. The extracted JSON and committed test
fixture are byte-identical.

The subsequent exact-route census supersedes the synthetic-slot handoff and
binds all 19,200 production slot-major route IDs. That route packet consumed
this census's sole phase-attribution authorization. Its frozen campaign stops
`INCONCLUSIVE - timestamp-invalid` at the gate empty pre-bracket, before any
retained phase cell or down execution. No latency, drift, p95, `U_phase`, KILL,
candidate authorization, or product claim survives.

Do not retry that protocol or infer timing from these count ratios. Reopen only
after separately reviewed conservative handling for timestamp-resolution-
censored empty commands, relevant device/toolchain drift, or a new accepted
production observer. This census remains valid exact route geometry and
operation-count evidence; it authorizes no new profiler or kernel by itself.

## Provenance

- Base revision: `864fece95c6c44f8c7182046819eaa9775383de2`.
- Device: Apple M4 Max.
- Asset-run source diff SHA-256:
  `2de52f475519c38e57f7a1e38f5d6ab5698ab0758ef72749bfbc83fdec67ecc2`.
- Retained source diff SHA-256:
  `61b15af6c5f20607600eeb472521482061ddd0e98e1b967d5c5e538cdafe4629`.
- Raw log SHA-256:
  `28c3edca24e0a6255859aef965e000064ac46e940f5c0d47bc6b2c3ada1b2682`.
- Extracted JSON and committed fixture SHA-256:
  `c6a7519f047cc2c4fd14b7c97db60edcbcf7d3b268b388eaab33c50cd2327654`.
- Asset-run release test binary SHA-256, recorded before execution:
  `4e113a075073e0b20c9520f4a69cadcbc7e23fb156bdc7335c34bc4a14130dc9`.
- Final retained release test binary SHA-256, rechecked after validation:
  `c415cf2fd4a123cf1d0a75f4377f371372191cce52a70f695b58ac6487e4f3f9`.
- Sole asset test: 8.12 seconds in-test, 8.21 seconds elapsed, zero swaps.
- Checksum transcript: `checksums.log`.
- CX review: `019fd588-96b0-7033-afd1-66d7e354e523`.

Asset command:

```bash
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_grouped_iq2_count_census \
  -- --ignored --exact --nocapture --test-threads=1
```
