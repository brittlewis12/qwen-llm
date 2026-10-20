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
winning grid without an M512/model/device whitelist. No K ceiling is imposed.
At candidate qualification, K1024 and non-M512 geometries had only correctness
coverage. The completed M4 measurements below add the two actual GLM shapes,
not performance evidence for arbitrary K/M geometries or K1024.
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

The ignored PV probe is production-vs-forced-scalar (45 medians, not the old
90); it requires validation variables unset. At the initial candidate commit it
had not yet been timed locally; completed M4 rollout evidence follows below.
Whole-model throughput, task/NLL checks, review and default installation remain
parent responsibilities; this commit is a production-eligible candidate only.

## Completed M4 Rollout

Local Apple M4 Max, 128 GiB, macOS 15.7.9, normal release Cargo builds and normal
test lease. These are completed local measurements, not M5 results. Production
code is byte-identical to `97a58f4ad432832bc410f9a498315aaa5712aec1` at both timed
revisions; changes after that candidate are tests and this evidence document.
No selector, device-name gate, precision policy, model or installation changed.

### Exact Timed Sources

- Run A: clean `152a7c54b0d7a34b77f9e57ecfb51d7183aa4761`. Original 15-geometry
  PV probe (540 samples, 45 medians) and eight GLM projection geometries
  (96 samples, 8 medians). Evidence: `/tmp/v41-f32-rollout-m4-gBZTVI/`, especially
  `pv-timing.log`, `glm-timing.log`, `release-correctness.log`, `summary.csv`,
  `summary.json`, `samples.json`, `commands.txt`, and `identity.json`.
- Run B: clean `6f96879f09f67bae687b2c07e81f9e7d2cd528ba`. Only the four new GLM
  rows-65/2048 geometries (48 samples, 4 medians), without rerunning PV or the
  original eight GLM timings. Evidence: `/tmp/v41-f32-rollout-m4-extra-KFtzxV/`,
  especially `glm-extra-timing.log`, `correctness.log`, `release-correctness.log`,
  `summary.csv`, `summary.json`, `samples.json`, `commands.txt`, `identity.json`,
  and `production-identity.json`.

This documentation is committed after the runs; its commit is NOT the timed
source revision. The two datasets are separate campaigns, not one combined run.
Run A binary SHA256 is `be552d4bbc4d66b03de766e2adefe7351183b662f47bca0721aa60ca64a18dc9`;
Run B is `3f3a1a34149f95d2106611e0c541e3fb89fc80e9f107efd333e768537a1c9c3c`.
Run A PV/GLM log SHA256 values are respectively
`ca2a2a9b9f3434bb4ae487553ab330bb124c3772f3090ab68ab324133693ef0d` and
`25d25fad46b24e35931895731e252d18de5baec790372d7c3d58bd352e5a16d4`.
Run B extra-timing log SHA256 is
`c558c808bcfc544a9932fc2c0f1f3b75621474eb150fae6fc6a8a654892f2768`.

### Method And Correctness

Each geometry uses scalar/production/production/scalar x3 (six samples per arm),
four serial launches per command, after both arms warm up. Both arms use the same
resident W/input/output bindings during timing. Census proves forced scalar vs
unforced production MMA, including the scalar remainder at row 65. GPU numbers
are command timestamps; wall includes encode + commit + wait. Both are per launch,
not per query. Allocation, compilation, census, oracle and readback are excluded.
All 684 measured samples across the two campaigns are retained without outlier
filtering; summaries retain per-arm ranges as well as medians. Warmups are not
measured samples. `MTL_DEBUG_LAYER` and `QWEN_REQUIRE_METAL_TESTS` were unset for
timing; `QWEN_METAL_LEASE_WAIT=1` was set without any admission bypass.

Correctness ran first with all three required flags set to 1. Run A passed seven
optimized-profile tests and two release fixture tests. Run B passed its bounded
GLM correctness test in both profiles, covering ten geometries through 512 rows,
including both new 65-row fixtures. The entire 65th row is bitwise equal to forced
scalar; prefix F64/scalar bounds and all offset/input/output guards are unchanged.
The inherited optional Metal 4.1 compiler-tier warning was reported, not overridden.

The 2048-row fixtures run only in the ignored performance probe. Before and after
timing they check 3,072 selected outputs across eight boundary/interior rows, all
64 groups and six channels against finite/scalar/F64 expectations, plus guards on
all four buffers. Readback and F64 work are bounded: no full-output finite scan,
full-matrix F64 oracle, or whole-buffer guard copy. These sampled checks passed;
they are not exhaustive 2048-row correctness or model-quality qualification.

### M4 PV Sink Results

Run A, M512 and 64 heads. Sink = score + softmax + pack + PV + finite; excludes
inverse RoPE and model execution. Medians in microseconds, scalar / production.
The full per-geometry PV-only and pack/PV/finite results remain in Run A's CSV/JSON.

| K | Queries | Ragged | GPU scalar / production | Wall scalar / production |
| --- | ---: | --- | ---: | ---: |
| 256 | 1 | no | 258.193 / 261.557 | 321.641 / 326.276 |
| 256 | 8 | no | 516.391 / 308.156 | 568.932 / 369.797 |
| 256 | 32 | no | 1330.365 / 778.635 | 1378.224 / 822.974 |
| 512 | 1 | no | 90.891 / 90.401 | 130.417 / 127.625 |
| 512 | 1 | yes | 86.818 / 85.224 | 118.115 / 119.224 |
| 512 | 8 | no | 560.563 / 392.661 | 608.547 / 438.672 |
| 512 | 8 | yes | 556.797 / 390.854 | 608.615 / 436.458 |
| 512 | 32 | no | 2174.531 / 1480.786 | 2214.630 / 1524.370 |
| 512 | 32 | yes | 2160.646 / 1463.823 | 2200.755 / 1504.932 |
| 640 | 1 | no | 121.385 / 121.219 | 168.432 / 165.636 |
| 640 | 1 | yes | 124.453 / 123.859 | 163.234 / 175.573 |
| 640 | 8 | no | 845.188 / 482.469 | 883.151 / 526.219 |
| 640 | 8 | yes | 842.906 / 475.453 | 883.198 / 515.286 |
| 640 | 32 | no | 3351.724 / 1853.417 | 3402.365 / 1902.464 |
| 640 | 32 | yes | 3330.250 / 1833.026 | 3377.875 / 1882.724 |

Eligible batched GPU speedups: PV alone 5.151-9.931x, pack/PV/finite
2.928-5.122x, Sink 1.425-1.817x. Singleton production is scalar/scalar control
(Sink 0.987-1.019x), not a newly enabled MMA path.

### M4 GLM Projection Results

G64, shared F32 absorption/expansion operator only, not surrounding GLM attention.
Rows 2048 represent the normal serve packed chunk. Medians in microseconds.

| Run | K | M | Rows | GPU scalar / production | Wall scalar / production | GPU speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 256 | 512 | 64 | 1645.047 / 179.333 | 1690.510 / 217.047 | 9.173 |
| B | 256 | 512 | 65 | 1303.708 / 187.786 | 1351.958 / 231.302 | 6.943 |
| A | 256 | 512 | 128 | 2552.000 / 263.781 | 2599.354 / 308.349 | 9.675 |
| A | 256 | 512 | 256 | 5040.594 / 523.984 | 5083.750 / 568.328 | 9.620 |
| A | 256 | 512 | 512 | 10508.010 / 1047.177 | 10557.354 / 1094.870 | 10.035 |
| B | 256 | 512 | 2048 | 46612.531 / 4204.167 | 46675.677 / 4256.354 | 11.087 |
| A | 512 | 256 | 64 | 827.641 / 135.661 | 861.427 / 171.141 | 6.101 |
| B | 512 | 256 | 65 | 857.156 / 187.422 | 899.641 / 285.500 | 4.573 |
| A | 512 | 256 | 128 | 1653.875 / 261.448 | 1695.271 / 297.422 | 6.326 |
| A | 512 | 256 | 256 | 3567.937 / 513.208 | 3607.203 / 552.646 | 6.952 |
| A | 512 | 256 | 512 | 7397.667 / 1019.682 | 7440.427 / 1065.474 | 7.255 |
| B | 512 | 256 | 2048 | 33497.641 / 4079.740 | 33564.755 / 4154.417 | 8.211 |

No measured eligible M4 geometry regressed in GPU or wall median. The new rows-65
and rows-2048 results support the current eligibility for these tested shapes,
without a device-name gate. They do not qualify arbitrary geometries, other
devices, cold buffers, whole-model throughput or task/NLL quality.

Drift/outliers remain visible. Run A's first GLM K256/M512/rows64 scalar samples
fell from 2757.583 to 1259.792 us, and production from 228.844 to 137.115 us;
early PV singleton absolute timings were also elevated. Run B's first rows-65
scalar sample was 1498.125 us versus later 1302.406-1305.313 us. K512/M256/rows65
production wall samples span 220.135-349.229 us despite GPU samples spanning
185.375-189.458 us. None were dropped, replaced or relabeled as warmed-only data.
Causes were not measured; do not treat the medians as stationary device peaks or
compare absolute timings across geometries/campaigns without that qualification.

### Parent Probe Commands

Run against test successor `6f96879f09f67bae687b2c07e81f9e7d2cd528ba`, or record
the actual build identity if using its documentation-only successor. The following
commands do not imply a worker/model launch and were not run remotely here.

```sh
MTL_DEBUG_LAYER=1 QWEN_REQUIRE_METAL_TESTS=1 QWEN_METAL_LEASE_WAIT=1 \
  cargo test --release --offline --locked -p qwen-llm --lib \
  grouped_f32_glm_shapes_correctness -- --nocapture --test-threads=1

env -u MTL_DEBUG_LAYER -u QWEN_REQUIRE_METAL_TESTS QWEN_METAL_LEASE_WAIT=1 \
  cargo test --release --offline --locked -p qwen-llm --lib \
  f32_pv_batch_performance -- --ignored --nocapture --test-threads=1

env -u MTL_DEBUG_LAYER -u QWEN_REQUIRE_METAL_TESTS QWEN_METAL_LEASE_WAIT=1 \
  cargo test --release --offline --locked -p qwen-llm --lib \
  grouped_f32_glm_shapes_performance -- --ignored --nocapture --test-threads=1
```

Expected: one correctness test covers ten bounded fixtures; PV prints 540 samples
and 45 medians; the full GLM probe prints 144 samples and 12 medians, including
both new row counts. The local Run B command instead used
`grouped_f32_glm_extra_shapes_performance` with the same performance flags to run
only rows 65/2048 (48 samples, 4 medians). The existing 15-geometry PV probe was
not rerun for this extension. No merge/rebase of maintained row-batching history,
production change, installation, remote GPU worker or model execution occurred.

## Completed M5 Operator And Quality Measurements

Clean source `25d67da6c69b37bc10d236a59cb1450eaa21768e`, Apple M5 Ultra,
normal release Cargo build and production Metal lease. These are new measurements
of the production selector, not the older explicit experimental MMA entry point.
Worker evidence is under
`~/qwen-runs/2026-10-20-v41-unsloth-profile/attention-current-evaluation-25d67da6/`;
local copies are in `/tmp/v41-attention-current-evaluation-25d67da6/`.

The four correctness filters in the campaign passed 63 tests, with three
performance tests ignored, under API validation and require-Metal flags. Timing
then ran with validation flags unset. All 684 samples and 57 medians from the
PV and full GLM probes are retained. The CPU analyzer
`/tmp/v41-attention-m5-operator-analysis.py` reconciles every geometry, ABBA arm
order, repetition and printed median; it produces `operator-summary.json` and
`operator-samples.json`. Each geometry has three ABBA repetitions and four
serial launches per command, not independent process-level trials.

### M5 Operator Results

Median GPU speedups, forced scalar / ordinary production selection:

| Scope | Eligible geometry | GPU speedup |
| --- | --- | ---: |
| V4.1 PV | 8 or 32 queries, K256/512/640 | 2.504-5.168x |
| V4.1 pack/PV/finite | same | 1.905-3.285x |
| V4.1 Sink | 8 queries | 1.215-1.258x |
| V4.1 Sink | 32 queries | 1.398-1.458x |
| GLM shared projection | G64, K256/M512, rows64/65/128/256/512/2048 | 4.602-5.659x |
| GLM shared projection | G64, K512/M256, same row counts | 3.820-5.334x |

Every eligible geometry also improves host-wall median and each ABBA repeat's
mean GPU comparison. Singleton V4.1 remains a scalar/scalar control, with Sink
GPU median ratios 0.997-1.004x. Sink excludes inverse RoPE and model execution;
the GLM scope is the shared projection, not GLM attention or end-to-end inference.
The 2048-row GLM checks retain the previously documented sampled coverage.

No samples are dropped. In particular, the first K256 singleton Sink crosses a
large timing change within its last ABBA repeat (scalar 75.031 then 51.156 us;
production 53.823 then 50.958 us). These controls do not establish stationary
device peaks or explain the cause of the change. Per-arm ranges and all three
repeat-level comparisons remain in the derived JSON.

### M5 Model Quality

Compared with complete-row baseline `1388793d`, using the same six frozen
documents, native-tokenized 512-token prefixes and 64 teacher-forced targets
per document. Identical additive helper sources are built on each production
revision: baseline `cd4534d8`, candidate `9235fd84`. Retained full-vocabulary
logits are scored in F64; this is 384 targets per quant, not broad quality or a
higher-precision execution oracle.

| Quant | Baseline mean NLL | Candidate mean NLL | NLL delta | Top-1 changes | Max TV | Max KL baseline to candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2_K_XL + Q6 Engram | 2.160671733098 | 2.160671312989 | -4.201095e-7 | 0 | 2.369355e-5 | 1.148125e-9 |
| IQ3_XXS | 2.146361600884 | 2.146361639566 | +3.868189e-8 | 0 | 5.419262e-5 | 1.012083e-8 |

Target correctness remains 214/384 and 222/384 respectively. Maximum absolute
logit changes are 0.004138947 and 0.001425743; arithmetic is not bitwise identical.
These numerical differences are observations, not baseline-exactness vetoes.

The unchanged 22-task greedy screen produces identical visible answers in both
quants: 8/10 code, 1/8 reasoning and 0/4 JSON, or 9/22 overall. Its existing r1
truncation remains counted; no task is removed or retried. The fixed none-effort,
greedy, cache-disabled screen is a bounded regression comparison, not a claim
about the model's best achievable task quality. Full evidence is retained in
`task-comparison.json` and `likelihood-comparison.json`.

These completed operator and quality measurements do not themselves establish
whole-model throughput or an installation result.
