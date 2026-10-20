# Shared F32 Grouped Selector Candidate

Base: `1388793dc6bce0de79aaf2df136085f47c02e9a3`. Only the direct-F32 16x64x32
tile is carried from `7e94ba1942ea9a8919c5f9925131a19d31ec326c`; the unused 8x32
shader and explicit experimental MMA API are not carried. IQ2, native-row FFN,
decoder, cache, routing and precision policies are unchanged from the base.

## Retained M5 Evidence

These are completed parent measurements, not new timings of this candidate.
Clean release build `7e94ba19`, validation disabled, resident synthetic fixtures,
15 geometries, scalar/MMA/MMA/scalar x3, four launches per command. Evidence:
`/tmp/v41-f32-pv-mma-batch-7e94ba19/{timing,correctness}.log` and `build-info.json`.
The build identity matches source, with no overrides. Correctness: 8 passed,
1 ignored. All 1080 samples and 90 medians remain in the supplied evidence;
the experiment branch is preserved, and none are removed here. Timing log SHA256:
`ffe702ab719edeee2921bbab0bd261b926e9e7312b2e5c6f0228f53ffee5348a`.

Sink scope is score + softmax + pack + PV + finite, excluding inverse RoPE,
allocation, CPU encoding, readback and model execution. Median GPU speedups
(scalar/M16N64), M512, 64 heads:

| K / counts | 1 query | 8 queries | 32 queries |
| --- | ---: | ---: | ---: |
| 256 / full | 0.806 | 1.256 | 1.403 |
| 512 / full | 0.758 | 1.216 | 1.400 |
| 512 / ragged | 0.755 | 1.216 | 1.403 |
| 640 / full | 0.765 | 1.257 | 1.453 |
| 640 / ragged | 0.767 | 1.258 | 1.461 |

The first K256/M16 scalar sink control is anomalously 74.979 us versus 33.917 us
in the M8 control. Both are retained, not selectively discarded. M8 wins less at
batched shapes and also loses at singleton shapes. Pack/PV/finite alone improves
1.909-2.090x at 8 queries and 2.917-3.318x at 32 queries for M16; singletons lose.
The roughly 29-32% sink time reduction at 32 queries is NOT a whole-model claim.
The earlier rejection was singleton performance, not a quality veto. Independent
query groups expose more parallel tiles and change that measured decision.

## Selector And Caller Audit

The existing shared `encode_mat_mat_grouped_f32` selects M16N64 only for F32
weights, K >= 256 aligned to 32, M divisible by 16, groups >= 8, rows >= 64, and
checked `(M/16) * floor(rows/64) * groups >= 256` threadgroups. The floor excludes
tail work; overflow fails closed to scalar. This matches the smallest measured
winning grid without an M512/model/device whitelist. No K ceiling is imposed;
K1024 and non-M512 geometries have correctness coverage, NOT performance evidence.
More work does not guarantee speed on every architecture or geometry.

Both pipelines are preflighted before any dispatch. Unsupported MMA capacity or
pipeline falls back to scalar4. Complete 64-row prefixes use MMA; row remainders
reuse the existing bounded scalar-tail strategy. Existing K%32, binding extent,
alignment, provenance, overlap and device validation remains unchanged (invalid
bindings/non-32-aligned K are errors, not a new relaxation). Q6's eligibility,
128-row tile, decoder and 4096-byte scratch are unchanged. All F32 operands,
K32 partials and accumulators stay F32; reduction order need not be bitwise scalar.

Audited production callers:

- V41 query-batch PV uses the ordinary shared API, one group/value matrix per
  query, heads as rows. Eligible batches select MMA; groups 1-7 stay scalar.
  The existing count-256 split and inverse RoPE are not changed.
- V41 single-query value sink, prefill linear projections, and shared MHC4
  grouped projection all use one group: unchanged scalar selection for F32.
- GLM decode uses the single-row wrapper: unchanged. Packed F32 latent absorption
  and expansion in Fast F32/Half policy can newly use MMA (e.g. K256/M512 or
  K512/M256, 64 groups, >=64 rows). The policy label does not narrow these F32
  operands. Exact and DenseF32 retain scalar arithmetic through the existing
  caller permission. Quantized GLM paths remain unchanged. GLM performance and
  end-to-end model/task quality are not established by the V41 PV measurements.

## Controls And Verification

Test-only `Scalar` now calls the same private implementation with scalar policy
forced independently of thread-local overrides. `Production` calls the ordinary
public selector unforced. Census proves different kernels at eligible shapes,
scalar on both arms for singleton controls, and scalar row tails. No test helper
is a production API, no new environment switch is introduced, and the old M5
experiment remains intact for reproducing its original three-arm comparison.

Reused fixtures retain their existing F64, half-sensitivity, scalar comparison,
guard and finite/error bounds. Coverage includes groups 1/7/8/32, rows 64/65/129,
K1024, generic output/work-floor boundaries, overflow, preflight failures, and
unaligned/short/aliased buffers. Production query-batch tests cover visibility,
zero padding, invalid IDs, sticky finite flags, moving raw windows, pending pairs,
followers, inverse RoPE, decode continuation and threshold-split census. GLM
caller tests cover both real latent shapes and Exact/DenseF32/Fast policies.

Local validation command (no models, remote workers or timing):

```sh
MTL_DEBUG_LAYER=1 QWEN_REQUIRE_METAL_TESTS=1 QWEN_METAL_LEASE_WAIT=1 \
  cargo test --offline --locked -p qwen-llm --lib -- \
  metal::latent:: glm5_next_metal::packed::native_dispatch:: \
  query_batch:: f32_pv_batch:: --test-threads=1
```

Local Apple M4 Max result: 62 passed, 0 failed, 1 performance probe ignored,
with API validation and the normal Metal lease. The inherited optional Metal 4.1
tier is unavailable in this local compiler; it is reported, not overridden.
K256/512/640 positive PV fixtures differ from forced scalar by at most
`9.080395e-9`; K1024 by `6.1118044e-9`. Signed K288 fixtures reach `4.2915344e-6`
at scale 1, or `0.5625` absolute at activation scale 131072 (same scaled error).
Independent F64, half-sensitive, finite and original scalar bounds all pass.
Groups 1/7 retain scalar results. This is correctness evidence, not performance.

The ignored probe is now production-vs-forced-scalar (45 medians, not the old
90); it still requires validation variables unset. It has not been timed here.
Whole-model throughput, task/NLL checks, review and default installation remain
parent responsibilities; this commit is a production-eligible candidate only.
