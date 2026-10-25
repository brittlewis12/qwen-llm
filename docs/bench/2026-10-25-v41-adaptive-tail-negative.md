# V4.1 Single-Panel Adaptive Tails: Correct, No Promotion

## Decision And Scope

Do not promote this uniform-branch, single-remainder-panel leaf. Actual M5
correctness passes, including 40 bitwise incumbent comparisons, but the measured
fixture cohort does not establish a useful latency gain. This does not close
adaptive tails generally or the broader expert-major/structural-occupancy lane.

Source `e65eb300a28b518f04ff9f2076af1a2dcde6edaf` on
`perf/v41-adaptive-tails` adds four explicit Q2_K/IQ3_XXS probe kernels and a
private test encoder. Production selectors and incumbent source bodies remain
unchanged. The product tensor floor remains Metal 4.0 / macOS 26.0; native M5
qualification uses SDK 27.0, not a new deployment requirement.

## Hypothesis And Implementation

Actual frozen-text CED counts have 34.13% routed-projection panel padding.
Keeping incumbent full panels and selecting one 16/32/64-slot descriptor for
each remainder models weighted row slots falling 5137056 -> 3958720 (-22.94%).
This is logical descriptor arithmetic, not measured hardware instructions,
DRAM traffic, shader-active FLOPs or predicted model latency.

The probe keeps the 64-slot encoder gate/up and 32-slot decoder/down grid
strides. A uniform live-count branch selects the static destination descriptor;
each incumbent panel still has one weight-decode K loop. Half operands, K32
traversal, F32 sums, epilogues, clamp and valid-ID rules remain. There are no
extra dispatches, command submissions, CPU count readbacks or split-tail weight
decodes. Staging has unique physical writers on underwidth/M tails.

Possible costs include descriptor branching, worst-branch register allocation
and changed activation-load scheduling relative to barriers. None is measured
as the causal explanation here. Unchanged weight decode/staging can dominate
even when the logical arithmetic model shrinks.

## Correctness And Fixture Repairs

The first frozen M5 run at `e399855a` stops on the retained clamp-coverage
assertion: its Q2_K/K256 data does not exceed the fixture limit. Three partial
bitwise-positive comparisons are preserved but are not called qualification.
The correction changes only fixed codec/K-specific fixture limits and CPU
coverage checks; banks, inputs, IDs, kernels and numerical tolerances remain.
CPU controls reproduce the original failure and reject unexercised, all-clipped,
empty and near-zero fixtures. The separate min/max checker also rejects missing
admitted writes rather than accepting finite canaries or earlier-grid outputs.

The corrected native M5 executable passes six CPU tests and four self-leased
validation-on GPU gates: boundary/epilogue cases, min/max count ranges, real
K/M shapes and F32 refusal. M5 absence is a hard failure. All 12 boundary plus
28 real-shape active-output comparisons are bitwise equal to the incumbent with
`QWEN_ADAPTIVE_REQUIRE_BITWISE=1`. CPU half-operand references, finite values,
input immutability, holes, offsets and outer guards pass. The min/max GPU
oracle's maximum error is about 4.76837e-7 for both grids. API validation and
guards alone are not a comprehensive threadgroup-race proof.

The 28 timed cases use seven observed five-expert count subsets, both codecs,
and K5120/M2304 gate/up or K2304/M5120 down. Experts are selected near counts
0/8/24/48/65. Banks, activations and ordered IDs are synthetic; some ordinal
cases construct identical workloads. These are not full 384-expert projections,
captured physical routes, Full/CED execution or a model-weighted sample. The
512 MiB fixture bound is checked before allocation; cases are dropped between
fixtures. Full-model and physical-route lineage remain unqualified.

## Complete-Projection Timing

After correctness, each case warms ABBA and retains three ABBA rounds: six
incumbent and six adaptive samples, 336 retained samples total. A gate/up sample
includes both dispatches; down includes its one dispatch. GPU command time and
inclusive host encode/commit/wait time are recorded. Input is already half;
router, conversion and other model work are excluded. No outliers are removed.

Known debug/shader/capture/injection variables are sanitized only at child
launch; the fixture explicitly requests validation off. Effective external
validation/injection remains independently unattested. This is a requested
timing regime, not an attested production model benchmark.

Adaptive/incumbent ratios from all 28 cases:

| Metric | Slower Cases | Geometric Mean Case Ratio | Median Case Ratio | Ratio Of Summed Fixture Medians |
| --- | ---: | ---: | ---: | ---: |
| GPU command time | 20/28 | 1.038824 | 1.028884 | 1.044078 |
| Complete-projection wall time | 19/28 | 1.022967 | 1.014349 | 1.025259 |

These are unweighted fixture summaries, not a population/model-weighted gain or
loss. Within-case drift is substantial: one Q2_K encoder gate/up incumbent spans
about 404-1094 microseconds. Preserve every sample rather than inferring a stable
intrinsic slowdown, a mechanism-specific cause or universal impossibility.
The bounded cohort supplies no basis for a production landing or a full-model
speedup claim. Do not continue a width/launch sweep of this losing leaf merely
because its arithmetic model is favorable.

## Frozen Evidence

Frozen executable SHA256:
`aa4e56481b6d8703e788c3e662d665345d3efefba54fe56adc95e9bc5b5415c6`.
Raw timing stderr SHA256:
`faa917324e49ac1767ab91d0223d927d8d14b11964545b854e19a782f1402ce5`.

Private labels include `adaptive-tails-v28-core.identity.json`,
`adaptive-m5-v28-{cpu,boundaries,count-range,real-shapes,f32-refusal,abba}.*`,
`adaptive-m5-v28-correctness.json` and `adaptive-m5-v28-summary.json`.
All executable/metallib copies freeze before releasing the build lease.
Independent read-only review recomputes all 336 samples, medians, ratios and
40 bitwise checks without finding a discrepancy. Historical failed runs and
original frozen sources remain intact; no prototype kernel ships on main.
