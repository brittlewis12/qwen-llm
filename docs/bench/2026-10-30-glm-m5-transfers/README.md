# GLM-5.3-Flash M5 Prefill Transfers

## Status

**Both bounded routes are default-on for eligible M5 Fast/Half prefill.**
Implementation `bcf6fd6c` is integrated into main by `2ac3d907`, preserving the
independent flag-0 rollbacks. The frozen measurements below retain their
original source/binary identities; they are not retagged as final-main runs.
This promotion within the existing Fast/Half policy does not establish global
quality equality, formal noninferiority or a universal speedup.

At 2K/4K, median native wall falls 3.3526%/3.1844% for wide-only and
8.3784%/7.5065% for Q5-down-only. N512 does not improve. Late-run drift limits
the precision of the down estimate; these are not universal speedups or
full-model quality results.

These independent comparisons remain the performance headlines (about 3.2%
wide and 8% Q5 down), not additive percentages. The combined cohort below is
retained with its drift, not promoted to a pooled 18% headline.

## Frozen Identity And Artifact

- `qwen-bench` source: `0473b95c2e299bd39794d0350c5e18d60718d146`.
- Executable SHA256:
  `dbd6532a39444d7ef9a821636da8012fee49ccb501ad2015b6c3b54f4ab62ea6`.
- Build/runtime source state, `git-source-sha256-v2`:
  `7e7c8c0334c840c383ef31290cf12663ea598a59780b40f15501a0bb850e5456`.
  The original independent cohort's 24 report rows have matching clean
  build/runtime identities, with no overrides, identity problems or stamp
  errors. Report-file hashes also match the campaign events; this is not
  verification of model payload hashes.
- M5 Ultra, 80 GPU cores, 256 GiB, macOS 27.0.1. Metal SDK **26.2**, Apple
  Metal compiler **32023.864**. This is not the V4.1 SDK 27 release cohort.
- Actual six-shard GLM-5.3-Flash UD-Q4_K_XL; reported `model_size` is
  **199697800568 bytes**. Payload hashes are **not verified**. The corrected
  numerical screen below admits the actual model pins within its metadata-only
  identity scope; no artifact-equivalence claim is made to the older
  UD-IQ3_XXS landing packets.

The current artifact's main routed roles differ from those older packets:

| Role | Main layers 0-44 | MTP layer 45, unused here |
| --- | --- | --- |
| Down | 39 Q5_K; 3 Q6_K at layers 11, 12, 44 | Q5_K |
| Gate/up | 82 Q4_K; 2 Q5_K at layer 11 | Q4_K |
| MLA K/V | All 22 main K/V tensors Q8_0, at layers 3, 7, ..., 43 | Not exercised |

The corrected screen checks this candidate coverage, not weight-payload
integrity or general quality. Do not substitute the old IQ3 role inventory.

## Native Measurement Boundary

Each fresh process runs N512, N2048 and N4096 at depth zero, seed 1, two timed
runs per shape. Inputs are synthetic token sequences, **not an actual saved
request**. The admitted chunk is 2048 for every report row, including N512;
the CSV's `prefill_rows` means admitted chunk size, not prompt length.
Production Fast/Half policy is used, with reported mode
`packed_fast+last_logits`. There is no Metal debug layer, GPU capture,
profiler or dispatch census in this timed cohort.

Wall includes ordinary full-model encoding, wait and final-logit readback.
It excludes fresh-session allocation, model load, network, cache and restore.
It is not serving TTFT, whole-response latency, a cached turn, or a stage-only
GPU interval. No GPU-time decomposition or decode benefit is inferred.

The controls are never combined within the original independent blocks:

| Block | Process order | Baseline flags (wide, Q5 down) | Candidate flags |
| --- | --- | --- | --- |
| Wide | A0, W0, W1, A1 | A0/A1: (0, 0) | W0/W1: (1, 0) |
| Down | A2, D0, D1, A3 | A2/A3: (0, 0) | D0/D1: (0, 1) |

Flag names are `QWEN_GLM53_DENSE_Q8_WIDE` and
`QWEN_GLM53_TENSOR_Q5_DOWN`. Each arm is one fresh process, not a replicate
within one long-lived model session. There are four observations per
policy/shape within each block, 24 report rows and **48 timed samples** total.
Keep the two baseline blocks separate; do not add the candidate percentages
or pool them with the separate combined cohort below.

Arms are separated by a requested 15-second pause (recorded gaps about
15.010 seconds). The driver budget is 600 seconds, with a real 660-second
outer timeout. The cohort completes in **409.656 seconds**, with every arm
successful and no forced kill. No timed sample is dropped or replaced.

## All-Sample Medians

Medians below use all four observations per policy/shape. Wall change is
`100 * (candidate / baseline - 1)`; negative means less wall time.

| Block | Prompt tokens | Baseline median ms | Candidate median ms | Wall change |
| --- | ---: | ---: | ---: | ---: |
| Wide | 512 | 1238.142 | 1238.465 | +0.0261% |
| Wide | 2048 | 3053.629 | 2951.253 | -3.3526% |
| Wide | 4096 | 6482.452 | 6276.026 | -3.1844% |
| Down | 512 | 1257.496 | 1259.810 | +0.1840% |
| Down | 2048 | 3115.793 | 2854.739 | -8.3784% |
| Down | 4096 | 6606.264 | 6110.363 | -7.5065% |

Both candidate processes, and all their individual samples, remain faster
than every baseline sample in their respective block at 2K and 4K. That
separation is useful performance evidence, not a confidence interval or
evidence that 8.3784% is a precise universal Q5-down gain.

## Drift And Observer Sensitivity

Per-process means in ms retain the order-dependent shift; each cell averages
two samples. [native-samples.csv](native-samples.csv) retains every individual
wall/rate value, policy, shape, identity status and observer marker.

| Arm | N512 mean ms | N2048 mean ms | N4096 mean ms |
| --- | ---: | ---: | ---: |
| A0 | 1236.581 | 3051.366 | 6482.452 |
| W0 | 1238.752 | 2951.780 | 6274.226 |
| W1 | 1238.465 | 2951.169 | 6276.393 |
| A1 | 1240.002 | 3058.084 | 6483.757 |
| A2 | 1238.455 | 3055.659 | 6487.223 |
| D0 | 1239.232 | 2780.313 | 5941.534 |
| D1 | 1289.995 | 2937.866 | 6272.852 |
| A3 | 1286.020 | 3175.829 | 6739.205 |

D1 and A3 are slower even at N512, where neither candidate improves wall:
about 4.1% versus D0 and 3.8% versus A2, respectively. The late down-block
shift is therefore not hidden behind the pooled median. Recorded thermal
and performance warning fields are false in all reports; **that does not
establish constant frequency**, power, thermal state or absence of drift.
No frequency/power trace or GPU capture is available to attribute the shift.

During A0's N4096 shape, one one-second read-only CPU process sample checked
initial quiet progress. The CSV marks both samples in that shape with
`external_cpu_sample_in_shape=1`; it does not assert which timed sample
overlapped the observer. All samples remain in the primary result. Using
only A1 as the wide 4K control still gives a **3.2039%** median wall reduction,
so the observed wide benefit is not dependent on retaining A0.

## Completed Combined Native ABBA: Drift Retained

This separate ordinary native run uses the frozen `0473b95c2` qwen-bench
and `dbd6532a...` executable identified above, SDK 26.2, debug **off**, with
the same full-model timing boundary. Order is **A0, C0, C1, A1**: A0/A1 set
(wide, Q5 down) to (0, 0), C0/C1 to (1, 1). Each fresh process runs
N512/N2048/N4096, depth zero, seed 1, two runs, admitted chunk 2048.
Requested inter-arm pauses are 15 seconds (recorded about 15.010 s).
All four arms complete successfully in **188.063558583 seconds**. All 12
report rows have matching clean build/runtime identities, no overrides or
identity problems, and matching report hashes. All **24 samples** remain in
[combined-native-samples.csv](combined-native-samples.csv), without exclusions.

| Arm | N512 mean ms | N2048 mean ms | N4096 mean ms |
| --- | ---: | ---: | ---: |
| A0 | 1241.465 | 3055.915 | 6486.511 |
| C0 | 1241.231 | 2679.858 | 5728.755 |
| C1 | 1250.618 | 2680.101 | 6281.868 |
| A1 | 1419.403 | 3480.477 | 7462.718 |

Late C1/A1 drift matters: A1's unaffected N512 is **14.3329% slower** than
A0, and C1's 4K is slower than C0. The pooled medians are retained for audit,
but are **confounded, not optimization estimates**:

| Prompt tokens | A0/A1 median ms | C0/C1 median ms | Confounded wall change |
| --- | ---: | ---: | ---: |
| 512 | 1316.874 | 1243.437 | -5.5766% |
| 2048 | 3266.022 | 2680.101 | -17.9399% |
| 4096 | 6953.993 | 5945.537 | -14.5018% |

Every combined 2K/4K sample is still faster than every baseline sample at
that shape. In the early A0/C0 bracket, N512 means nearly match (0.0189%
less wall), while 2K/4K means fall **12.3059%/11.6820%**. This local bracket
is useful context, not a universal combined speedup or a reason to discard
the later samples. There is no frequency trace or causal attribution; an
early matched N512 does not establish constant clocks at wider/later shapes.

## Initial Three-Document Numerical Screen

The shared tiny M5 wide/Q5 leaf witnesses passed with Metal debugging enabled.
They are separate from these debug-off full-model timings and are **not a
full-model quality witness**. An earlier identity refusal and two actual-Q4
pin refusals occurred before model load, outside this completed cohort; their
records remain private. They are not timed failures or passing qualification.

The corrected screen and existing 22-document/tool extension build from clean
`5081fcc0`: the earlier 23 focused CPU tests pass, and independent source
review has no remaining P1/P2 findings. The separate test-executable SHA256 is
`801872a93e3e1e6c440555cb2a07b010e43822b00759c56b8c2fd1bfb1e7ead4`.
The actual-M5 screen **passes in 320.87 seconds**, with an execution-complete
record: **576 teacher-forced rows (192 per arm), 32 validation-on plain
samples, three profiles and three censuses**. Metal API validation is enabled
for this screen, separate from the debug-off native cohort. Strict actual metadata,
expert-role and Q8 MLA pins admit, including all 22 main MLA K/V banks; the
actual layout is not the older stock fixture. Shard names/sizes, primary
metadata SHA and tensor-table checks do not verify weight-payload hashes.

Each arm scores 64 targets in each of prose, code and long-document strata,
comparing all **154880 vocabulary components** per row. All values are finite.
Q8WIDE is full-vocabulary **bit-exact on all 192 scored rows** across the three
documents: zero changed components, max absolute error, KL and NLL delta.
This finite/bit check is bounded to the scored teacher-forced histories, not
a causal-state proof or arbitrary-conversation equivalence.

Q5DOWN changes all 192 rows: max absolute logit delta **1.8312613**, max KL
**0.04859741174044692**, mean KL **0.0035254466880138145**, mean NLL delta
**-0.002238788080381185**, and **five top-1 changes** relative to CURRENT.
NLL delta means candidate minus CURRENT; a negative mean is not a
noninferiority or equality verdict. Preserve the opposite-signed code result:

| Q5DOWN stratum | Scored rows | Mean NLL delta | Top-1 changes |
| --- | ---: | ---: | ---: |
| Prose | 64 | -0.0063893356979749325 | 2 |
| Code | 64 | +0.010071017545712975 | 0 |
| Long | 64 | -0.010398046088881598 | 3 |

[numerical-document-metrics.csv](numerical-document-metrics.csv) retains all
nine arm/document aggregates, including CURRENT and Q8WIDE zeros, row and
changed-component counts, finite status, maximum logit/KL errors and mean
KL/NLL deltas. This is a descriptive screen with no quality acceptance
threshold; the complete event explicitly supplies no quality verdict.

## Separate Validation-On Diagnostics

Each full 2K census records **1342 dispatches**. CURRENT has no wide/Q5-down
candidate calls; Q8WIDE has **507 wide calls** and no tensor Q5-down calls;
Q5DOWN has **39 tensor Q5-down calls** and no wide calls. These confirm
independent path selection, not a measured combined configuration.

Profile/plain ratios below divide the single profiled call by the mean of
its two same-arm plain calls. All are validation-on diagnostics, not the
earlier SDK 26.2 debug-off native performance samples; do not pool them.

| Arm | Plain mean ms | Profiled wall ms | Profile/plain mean | Observer overhead |
| --- | ---: | ---: | ---: | ---: |
| CURRENT | 3341.613 | 3419.216 | 1.023223 | +2.3223% |
| Q8WIDE | 3245.596 | 3334.310 | 1.027334 | +2.7334% |
| Q5DOWN | 3108.194 | 3409.479 | 1.096933 | +9.6933% |

Routed experts dominate the actual profiles: CURRENT is about **60.8% of
scaled diagnostic spans**. These spans are not recoverable saved wall.
Q5DOWN is 6.9852% below CURRENT in the paired plain means but only 0.2848%
below it in profiled wall: split profiling loses much of the plain benefit.
Observer overhead varies by arm; do not use a leaf span or this diagnostic
comparison to replace the ordinary performance result or claim combined gain.

## Completed Extended Continuation/Tool Run

The separate extended test **passes in 681.59 seconds** on `5081fcc0`, using
the same `801872a9...` test binary identified above, SDK 26.2 and Metal API
validation **on**. Its unique complete event accounts for **4224 quality
rows, 66 document arms, 24 tool generations and 450 sampled-token records**,
with **24 expected chat stops and zero caps**. Retained shard stamps are
reported unchanged and observers clear. There are no plain performance,
profile or census observations in this extension; do not pool it with native
debug-off timings or the earlier validation-on diagnostics.

All 13 declared fixture/source-file hashes match `5081fcc0`. The frozen
`quality-v1.json` SHA256 is
`7f9ac936b08ee0f4e47fbbb42dbfd46c3055f058b4f36fa5d832fbefcad43057`.
Selection, u32 little-endian token hashes, unique rows, positions, CURRENT
reference NLL/top-1 arithmetic and completion counts are independently
rechecked. Each arm has **1408 finite full-vocabulary rows**: 22 documents,
eight prose, eight code and six long, with 64 targets per document. Maximum
frozen prefixes are 2048 tokens for prose/code and 4352 for long documents.

CURRENT and Q8WIDE have mean NLL **1.4462970340516323** and **979/1408**
top-1 hits. Q8WIDE remains full-vocabulary **bit-exact on all 1408 rows**,
with zero logit/KL/NLL differences and no top-1 changes. This is still a
teacher-forced finite/bit check, not a causal-state proof.

Q5DOWN changes all 1408 rows: mean NLL **1.445751627193185**, mean delta
**-0.0005454068584474908**, **980/1408 hits**, **28 top-1 changes**, and mean
KL **0.0031548689816829206**. Its worst absolute logit delta is **9.899603**
(`long-04`, target index 58, position 4410); worst KL is
**0.2790210105159407** (`prose-07`, index 41, position 2089). Indices and
positions are zero-based. These outliers are not erased by the near-neutral
mean or the single additional top-1 hit.

Prose/code/long hit counts are 244/454/282 (CURRENT: 243/454/282), with
18/2/8 top-1 changes, respectively. All strata remain in the analysis.

[extended-document-metrics.csv](extended-document-metrics.csv) retains all
66 arm/document aggregates, including NLL means/deltas, hits, changed rows
and components, finite status, and max/mean errors. Public document labels
number documents within each stratum in frozen fixture order; no document
text or token IDs are included. All **576 original three-document quality
records repeat exactly** in the overlapping subset. This is repeatability,
not an independent replication, and those rows are not counted twice.

### Descriptive Document Bootstrap

Paired per-document NLL means and top-1 hit-rate differences are resampled
20,000 times, seed **20261008**, following the existing document-bootstrap
approach. Overall resampling keeps eight prose, eight code and six long
documents (weights 8/22, 8/22, 6/22); each stratum resamples its own documents.
Implementation uses stdlib `random.Random` (MT19937), reset per candidate and
scope, `randrange` draws in prose/code/long and frozen document order, shared
draws for both metrics, and linear percentiles at `(n - 1) * p` for p=0.05/0.95.
This avoids NumPy/environment changes; it is not NumPy-RNG-identical.

Q5DOWN deltas are relative to CURRENT. Accuracy changes/intervals are in
**percentage points**, not relative percentages. Intervals are descriptive
central 90%, not post-hoc formal pass bounds. Q8WIDE's observed deltas and
all corresponding intervals are exactly zero.

| Scope | Mean NLL delta | NLL central 90% | Accuracy delta, pp | Accuracy central 90%, pp |
| --- | ---: | --- | ---: | --- |
| Overall | -0.000545407 | [-0.003684075, +0.002519717] | +0.071023 | [-0.355114, +0.426136] |
| Prose | +0.002665668 | [-0.002326976, +0.007816731] | +0.195312 | [-0.585938, +0.976562] |
| Code | -0.000337876 | [-0.003412302, +0.002851714] | 0.000000 | [0.000000, 0.000000] |
| Long | -0.005103548 | [-0.013623127, +0.002380607] | 0.000000 | [-1.041667, +1.041667] |

Zero hit-rate intervals reflect observed document-level counts, not logit
equality or coverage beyond this cohort. The older prescribed Exact-versus-Fast
thresholds belong to a different cohort/reference. They are not retroactively
applied, and no preregistered pass or global quality noninferiority is claimed.

### Strict Tool Recheck

All arms are **8/8 strictly correct**: four frozen long-context tasks
(T09-T12), each once greedy (seed 0) and once sampled (seed 11). Scoring reuses
the unchanged `scripts/reference/glm53/quality_analysis.py::tool_correct`,
AST-extracting only `CURRENCY`, `norm_city` and `tool_correct` with
`math`/`unicodedata`, not
importing NumPy. The definitions also match source `5081fcc0`. This enforces
one expected call, normalized city/strict days or numeric amount/unambiguous
currency, expected observation stop and no parse error, not the lenient
first-call flag. All 24 stop correctly without errors or caps.

For **both candidates, all eight complete sampled-token sequences**, including
the stop, are identical to CURRENT; sampled/emitted hashes, token-event and
position alignment also check. This covers eight generations, not eight
independent tasks or arbitrary tools. Two tasks use 1455-row prefixes where
Q5 down is ineligible; two use 2048/2048/912 chunks with two eligible chunks.
The extension records no census and makes no new dispatch-count claim.

## Completed Combined Quality Check

The separate `e75cd6ab` test-only checkpoint **passes in 162.73 seconds**,
using test-executable SHA256
`5861235d09cfc0bec5ed80396fcce2481daf479fdf455ccf25028a6f99df6fdb`,
SDK 26.2 and Metal API validation **on**, not the native performance binary.
The source handoff reports **24 CPU tests passing and review clear**.
Its unique complete event accounts for **384 finite full-vocabulary rows**:
CURRENT (0, 0) and COMBINED (1, 1), three documents and 192 targets each;
**16 strict tool generations, 300 sampled-token records, 16 expected stops
and zero caps**. Retained shard stamps are unchanged and observers clear.
Frozen fixture/source hashes, CURRENT arithmetic, positions and counts verify.

All **192 COMBINED quality metric records** (difference, NLL, top-1 and related
fields) exactly match the earlier three-document Q5DOWN records after arm-name
substitution; CURRENT records repeat too. COMBINED mean NLL delta is
**-0.002238788080381185**, max absolute delta **1.8312613**, max KL
**0.04859741174044692**, mean KL **0.0035254466880138145**, with **five top-1
changes** and 115 hits versus CURRENT's 113. This is matching recorded
metrics, **not a full-vector bit-equivalence claim to Q5DOWN**: those vectors
are not retained for that comparison. The overlapping three-document check
is not independent replication or combined coverage of all 22 documents.
[combined-document-metrics.csv](combined-document-metrics.csv) retains all
six arm/document aggregates and every stratum.

The unchanged strict scorer gives **8/8 correct generations per arm** on the
same four tasks/two policies, with the eligibility limits described above.
COMBINED's eight complete sampled sequences equal CURRENT, including stops;
sampled/emitted hashes, token events and positions align. There are no parse
errors or caps. These validation-on quality observations are not pooled with
native timing, and no combined gain is inferred by adding independent gains.

## Decision And Integration

Both bounded M5 Fast/Half routes are **default-on** within the existing policy.
Each retains its separate, process-cached rollback:
`QWEN_GLM53_DENSE_Q8_WIDE=0` and `QWEN_GLM53_TENSOR_Q5_DOWN=0`.
Both at zero restore the previous baseline. Exact/F32, short-row and M4
behavior, and the existing global rollback remain unchanged. This is bounded
engineering judgment, not formal preregistered noninferiority/equality or a
universal speedup.

After qualification, `bcf6fd6c` changes only the two macro defaults, their
descriptions and CPU scope contracts; arithmetic, eligibility, buffers, shaders
and admission remain as measured. The source lands through `2ac3d907`.
The 19 focused CPU cases pass in five fresh-process flag configurations:
unset, 0/0, 1/1, 0/1 and 1/0 (95 passing executions). The composed main source
passes workspace-bin checks, release test typechecking, formatting and the
generated environment reference's self-test/check. Earlier 24-case release
protocol/selector checks and independent source reviews remain separately
recorded; no new GPU run is inferred from CPU checks.

Q8 wide keeps the existing Fast/Half, resolved-Fast, native-Q8, actual-N>=1024,
M5-tier and capacity guards. Q5 down keeps native Q5, K2048/M4096/E288/top-8,
and exactly 1024/2048 committed rows; the three Q6 down banks remain unchanged.
No persistent scratch or expanded weight bank is added. See the generated
[environment reference](../../ENV.md) for parsing and per-process caching.

Extended Q5DOWN outliers (**9.899603** absolute / **0.279021** KL),
the positive NLL interval upper bound and all earlier tool evidence remain
part of the decision, not erased by the combined check. General quality and
serving/decode conversion are not established; tiny tail passes are not
actual-model performance evidence.

Depth/ragged append performance is also unmeasured here. See the
[leverage map](../2026-10-30-glm-m5-leverage.md) for historical anchors and the
separate tiny M5 Q8 MLA tail witnesses, not actual-model tail performance.
