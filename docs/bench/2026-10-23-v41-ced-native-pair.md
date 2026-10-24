# Native V4.1 Full/CED Pair: Short-Prompt Loss, Long-Prompt Unqualified

## Scope

Initial experimental source `6896f87d` compares production Full forward and shrinking
CED inside one resident Maya-S model, with fresh KV state for every call. The
compiled/runtime identity matches and both are clean. This is synthetic cold-KV
prefill, not model loading, sampled generation, artifact quality or bounded128.
The explicit production-timing option is added by `5e6683a9`, with clean merged
source `a92e67d2`. Validation-on remains the default; there is no default-policy
promotion. Historical validation-on and subsequent requested production-timing
results are separated below.

Native timing includes prefill planning/admission, temporary allocation, CPU
Engram work, GPU encoding/completion and endpoint logits copy. State creation,
head fingerprinting, diagnostic serialization and teardown are outside that
timer. It is not GPU-only time or total request-lifecycle time.

## Protocol And Admission

- Warm A/B once, run an excluded ABBA pilot with two samples per policy, then
  retain a balanced BAAB block with two samples per policy. A is Full; B is CED.
- Check finite positive native samples and max/min <=1.5 separately per policy,
  including the pilot baseline, before scheduling further cases. Do not pool
  the intended difference between A and B into that stability ratio.
- Require the requested execution without fallback, completed CED commands,
  observed incremental Metal allocation no larger than its accepted estimate,
  and finite complete endpoint heads matching warm Full bitwise.
- Keep validation enabled by default; the explicit production-timing regime
  below changes only instrumentation controls, not arithmetic or admission.
  Keep the production lease and ordinary memory admission.
  Continuous secondary host supervision remains required; it does not replace
  core admission. Failure stops the sequence without automatic retries.

Public metadata-only cold planning now shares runtime geometry and prices
resident views/private shadows, binding allocations, one request state, the
sequential Full/CED workspace envelope and caller CPU storage. Binding controls
use actual validated tensor types and shared projection descriptors. The
256-layer all-owner ratio-two Q8_0 fixture needs 4,097 separate 8-byte controls:
at 16 KiB resource pricing, 64 MiB plus one page, not one aggregate allocation.

The caller envelope supports at most 16 retained samples per policy, 256 layers
and 1,048,576 logits. The maximum-size caller RSS fixture passes against the
64 MiB allowance, including live head/bit-vector capacities, both aggregates,
duplicated chronology and final JSON serialization. This is a regression
witness, not a formal heap proof.

## Historical Validation-On Measurements

M5 Ultra, 256 GiB; Maya-S plans 165.950 GiB resident views and zero private
shadow. Results concern this artifact, not the larger IQ3 artifact or its quality.

| Prompt | Full retained native samples | CED retained native samples | Verdict |
| --- | --- | --- | --- |
| 128 | 304.631, 305.713 ms | 462.065, 466.234 ms | Stable; CED latency is 52.1% higher |
| 2048 | Incomplete | Incomplete | Rejected; no comparative timing result |

The 128-token run completes all ten chronological calls and all endpoint
fingerprints agree. Minimum sampled conservative host availability is 73.66 GiB;
maximum sampled wired plus compressor usage is 174.18 GiB. The short-prompt
result does not predict long-prompt CED behavior.

At 2048, the pilot Full calls take 17.05 and 16.96 seconds. Its later retained
calls take 14.91 and 2.35 seconds, so Full's pilot-plus-main max/min reaches
7.272. The CLI rejects the sequence immediately after that ninth chronological
call, before scheduling the final CED case. Neither a partial mean nor the
single fast sample is a qualified comparison. No 8K timing follows this failure.

All recorded 2K endpoint heads still match. CED completes all 38 commands in
each executed case, with sampled incremental Metal allocation 2,981,986,304
bytes versus an accepted 3,004,235,776-byte estimate. Minimum sampled host
availability is 70.53 GiB; maximum wired plus compressor usage is 177.06 GiB.
There is no unhealthy host sample or watchdog incident in this sequence. These
samples are not continuous memory peaks or an OS reservation.

## What Remains Unknown

One resident object and one bound forward survive all cases; mappings, binding
and control-vector decoding are not recreated per case. Completed-work pools
and state teardown remain intrinsic. Engram still performs per-request reads
and dequantization, and pipelines are lazily cached, so retained ownership does
not imply constant physical residency, I/O cost or complete pipeline prewarming.
None of those facts attributes the observed phase shift.

CED also samples allocation through full memory signals inside its execution,
unlike Full. That instrumentation asymmetry is real but its cost is unmeasured;
it is not an explanation for the discontinuity or the 128-token difference.

Localize the 2K shift with explicitly instrumented case-level process I/O,
pipeline-cache and command encode/GPU/wait evidence, retaining the same stop
rules. Keep preparation/Engram and teardown outside command-timing attribution.
Instrumented results are diagnostic, not a substitute for a subsequent stable
unchanged native protocol. Do not infer thermal or external-job interference
from post-run observations alone.

## Verification And Next Kernel Bet

M4 verification passes 71 selected policy/planning/CED CPU tests, 1,513 CLI
library/binary tests and affected release all-target checks. The three bounded
metadata/runtime parity fixtures plus Full lifetime and half-allocation fixtures
pass on M4 and M5; corrected host-reserve boundary fixtures and explicitly
self-leased test entry points are recorded in `cc0a05c1`.

The earlier strict current-Full/CED equivalence still passes at 128/2048/8192:
all 257 full-vocabulary vectors bitwise and all 56 causal regions at every
position. See the [memory and qualification packet](2026-10-23-v41-memory-safety.md).

Exact hierarchical top512 selection above 4K remains an independent bounded
experiment (`perf/v41-select-hier`). Partition-local top512 plus exact merge
targets the source-derived 83,880 legacy sort/chronological dispatches in 8K CED.
It preserves signed-zero/tie/NaN behavior, causal visibility and absolute IDs,
with capacity-priced tile scratch and an incumbent rollback. At the initial
checkpoint, review alone does not establish GPU correctness or wall-time
improvement. Subsequent bounded and end-to-end proofs are linked in the follow-up.

## Follow-Up: Phase Planning And Attribution

The experimental planner separates full-future capacity feasibility from
availability at the actual phase boundary. It preserves the 80% ownership
ceiling, 20% host margin, positive process/device constraints and private/import
debt. Before import it admits resident views, binding and caller storage; after
binding and before each fresh state it reassesses the complete runtime envelope
with fresh signals. No active-cache credit or telemetry exception is introduced.
The original strict cold-peak wrapper remains available for qualification.

This avoids requiring every future allocation to fit a pre-import availability
snapshot, without assuming shared-cache reuse will occur. If runtime availability
does not suffice afterward, no new case state/prefill starts. Later production
CED fallback remains guarded Full execution and still invalidates a paired case.

Source `c1fdca25` passes a fitting supervised 2K diagnostic through those phase
checks, then rejects a 2.509x CED spread after its fifth chronological case.
Recorded GPU durations remain approximately 2.00 s Full / 1.75 s CED; CED's
completion-wall sum changes from 6.57 to 18.70 s while its GPU sum stays near
1.75 s. Warmed process reads/page-ins and recorded pipeline misses are zero.
These counters do not rule out all I/O or driver work, and the wait interval
does not identify scheduling, validation, residency or completion delivery.

The new aggregate block-row allocation/initialization span measures about
1.18 s across CED encoder/decoder constructors. It overlaps the encoder's host
preparation span; it is host wall, not additive CPU work. Request-local storage
reuse is a separate measured-code-path bet. Replacing allocation observations'
full host/task queries with the same raw device getter does not establish a
native speedup: a separate pp128 pair remains approximately 306 ms Full /
474 ms CED. Do not attribute the observed short-prompt loss to those queries.

The combined M4 checkpoint passes 156 forward, 17 FFN, 11 bounded CED, four
selector GPU and 1,525 CLI tests plus release all-target checks. Independent
reviews find no remaining staged-admission or lifecycle correctness issue.
All-current/frozen 8K head/state evidence is recorded in the
[exact-selector packet](2026-10-23-v41-select-hier.md).

## Explicit Production-Timing Regime And Verification

`--v41-ced-pair-production-timing` explicitly selects a production-timing child;
the default remains validation-on and correctness tests retain validation.
The controller unsets known `MTL_DEBUG_LAYER*`, `MTL_SHADER_VALIDATION*`, and
wrapper/injection/capture controls only in the child's launch environment.
The CLI refuses present controls rather than mutating its environment.
A raw requested-regime record is captured
before build identity checks, model opening or Metal initialization. Runtime
`QWEN` override values are redacted; launch-record write or flush failures stop
execution. This records requested controls, not verified effective instrumentation.
Historical rows establish requested API validation, not effective API-only
instrumentation or a complete shader-override record.

Arithmetic, leases, the 80% ownership ceiling, 20% conservative availability
margin, staged admission gates, inner checks, endpoint-head requirements and
per-policy stability rules are unchanged. Independent source review finds no
actionable P1/P2 issue. Frozen M4 verification passes all 1,535 CLI unit tests
(127 bench, 969 qwen, 302 lens, 119 library, eight census, nine grammar and one
tokenizer) plus 40 integration tests: 1,575 total. Affected release CLI
all-target checks pass; M5 passes all 127 bench CPU tests. Two actual process
refusal cases, environment and ablation, each emit exactly one request record
and refuse before opening a nonexistent model. Build refusal is covered only
by composed injected tests. The new maximum-caller RSS fixture also passes.

## Production-Timing Observations

One supervised Maya-S pp128 series accepts all ten chronological cases under
the unchanged warmup, excluded pilot and retained balanced protocol:

| Policy | Retained native samples (ns) | Mean (ns) |
| --- | --- | --- |
| Full | 296,901,792; 298,216,582 | 297,559,187 |
| CED | 449,668,500; 450,697,791 | 450,183,145.5 |

CED latency is approximately 51.3% higher. All ten uniquely recorded endpoint
heads have SHA-256
`6fc6e7b246b17af8bb0ad47d499b576e791cc243fb9340fe29a3e4cb91a0103e`,
matching validation-on reference `6896f87d`. Recorded artifact identity/location,
size, parameter count and family, plus prompt hash, capacity and seed, also
match. This is an endpoint and metadata bridge, not a full-weight content-hash
bridge. Neither this comparison nor the historical timings establishes validation
causality or a cross-regime speedup.

The subsequent single pp2048 series refuses before resident import: planned
load demand is 178,488,408,584 bytes versus fresh host remaining capacity of
168,515,023,667 bytes, from conservative availability of 223,490,605,056 bytes.
There is no timing result, retry, reserve change, active-cache credit or cleanup.
Recorded host-health samples remain healthy: pp128 minimum conservative
availability is 68.56 GiB and maximum wired plus compressor usage is 174.36 GiB;
the refused 2K launch stays below 7.78 GiB wired plus compressor. These sparse
samples do not override admission or establish continuous peaks. No 8K timing
follows. Long-prompt native performance remains unqualified.

Candidate scratch reuse continues through bounded proofs; it is not yet
runtime-verified and does not wait on resolving the timing ambiguity.

Whole-model explicit residency remains closed under the standing
[failure decision](../PERF-LOG.md#2026-08-13-whole-model-residency-optimization-lane-closed).
Do not reinterpret the wait gap as permission to pin, pre-wire, use `mlock`,
force cache cleanup or relax admission.
