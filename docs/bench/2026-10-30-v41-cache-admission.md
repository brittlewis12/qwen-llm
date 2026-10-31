# V4.1 Resident F16 Cache Admission

The default-on Q5_K resident F16 cache was missing from the cold/load memory
quote. The quoted dense-preweight workspace was only the largest single-matrix
per-chunk scratch, not the sum of retained caches. Binding also reached the
ordinary owned-buffer allocator without admitting the cache construction.
Later runtime gates observed the caches as live Metal usage, but that was too
late to cover binding or the full-demand-before-weights contract. This is an
accounting/admission repair, not evidence of historical OOM or latency causality.

## Inventory And Selection

`BindingMemoryPlan` uses the same active `LinearSpec` inventory as the binder,
with actual storage dtypes and normalized matrix dimensions. It freezes the
existing cache flag once; arithmetic preweight switches, tensor tier and row
thresholds do not silently change cache allocation eligibility. Each selected
Q5_K matrix prices checked `input * output * 2` bytes through ordinary physical
shared-buffer pricing, including device limits and independent buffer rounding.
The head participates when bound; embedding and inactive projections do not.
Grouped output_a has one whole-bank cache, then shared group views.

`ColdPrefillPlan.binding_metal_bytes` remains native controls only. The public
`binding_plan` reports cache selection, count, cache Metal bytes and control
Metal bytes separately. Paired preweight reports expose the cache terms too.
`Forward::from_resident_with_cold_plan` consumes and validates this frozen
inventory against the loaded tensors and current device pricing.

## Equations And Boundaries

Let R be resident Metal window lengths, S full additional private-COW shadow
exposure, B binding native controls, C resident F16 caches, U the full binding
CPU envelope, A caller CPU reserve, and T the request state plus componentwise
maximum complete sequential Full/CED workspace envelope. T_M and T_U are its
Metal and CPU components. T_s is the permitted future envelope selected at
binding: T for required Full/CED, or freshly admitted Full for optional CED
pressure fallback. Fixed cache/control and CPU reserves are never selectable.

| Boundary | Additional Metal | Additional CPU |
| --- | --- | --- |
| Full cold quote | R + B + C + T_M | S + U + T_U + A |
| Load quote | R + B + C | S + U + A |
| Fresh cold-plan binding, after source import | B + C + T_s_M | U + T_s_U + A |
| Fresh direct/serve binding, no request known | B + C | U |
| Matching request's postbinding/allocation gate | T_s_M | U + T_s_U + A |

At binding, source Metal R and private shadows S are already in live signals;
they are not charged again. At runtime, live signals also contain B and C.
The existing CPU-envelope recharges remain conservative. Direct/serve binding
admits construction only, not a fictitious zero-sized future request. Request,
state, Full and CED admission remain independently required.

The fresh aggregate binding gate precedes all binding cache/control allocation
and holds the ordinary allocation transaction through construction. Cold-plan
binding retains known future/caller demand; direct binding uses the loader's
retained full CPU quote. No per-cache pressure fallback is introduced. Existing
strict whole-cold admission and the separately documented staged load API retain
their respective availability/capacity semantics, now including C.

## Optional CED At Binding

Independent review found that the initial repair unconditionally admitted the
Full/CED maximum at binding. That refused staged optional-CED requests before
the ordinary postbinding pressure fallback could choose Full. Binding now uses
the same selector as request planning, adding B+C and U+A to every candidate.
Only optional CED pressure/unsupported cases may freshly admit Full; required
Full/CED, invalid telemetry and arithmetic errors do not acquire a bypass.

The bound Forward retains the selection for the matching cold geometry,
capacity, CPU reserve and original admission intent. `plan_request_from_cold`
still validates the original metadata/runtime quote, then reprices the selected
envelope and inherits its sticky fallback. Later allocation cannot silently
re-enable CED when availability recovers. Independent request planning is not
forced to inherit another request's binding decision.

The original cold quote remains conservative and unchanged: its standalone
`assess_runtime` still assesses that quote. Ordinary cold CLI/bench use the
matching bound request plan; paired Full/CED remains required and has no binding
fallback. Load-phase admission remains load-only after full peak capacity
assessment, not a whole-peak availability reservation.

CPU regressions extend existing classified names without inventory/runner
changes. They exercise the production binding factory boundary with staged
load admitted but whole-peak availability refused, optional CED refusing fresh
availability while Full fits, required CED refusing before the factory, sticky
postbinding/allocation handoff, repricing mismatch, invalid telemetry, changing
signals and one-byte-short cache/control/source-shadow/CPU/Full budgets.

Independent review of `41d3d862` clears this P2 correction, with no new P1/P2
finding in the reviewed repair. This source-review result is distinct from
real-artifact resource admission or execution qualification.

Quantized sources remain retained during and after conversion. GPU conversion
writes directly to each final F16 destination and waits; there is no additional
full-cache-sized conversion buffer. Source and accumulated destinations coexist.
Per-chunk dense-weight scratch still allocates and keeps its independent charge,
including eligible CED pipeline overlap. No lazy allocation, source-release
credit, arithmetic/kernel change or resource-policy relaxation is introduced.

## Verification Boundary

Host-only tests cover active mixed dtype/normalized-shape inventory, partial
schedules, head/embedding exclusions, grouped aliases, frozen selection,
per-buffer pricing and extent failures, exact-fit and one-byte-short admission
before factories, and the actual postbinding runtime gate. The exact acceptance
inventory classifies each new CPU name and one ignored GPU witness separately.

Local optimized non-LTO CPU acceptance passes exactly 557 tests: library 358,
oracle 33, benchmark 53 and CLI 113, including eight new cache regressions.
Eleven additional existing shared-buffer pricing/capacity/admission CPU tests
pass. Workspace/all-target checks, formatting, environment-reference checking
and acceptance-runner self-tests pass. This 557-test CPU source evidence remains
unchanged; the separate M5 toy result below does not extend it to model execution.

Acceptance discovery also exposed twelve pre-existing maintained-main serve
mapping/timing/native-failure tests missing from the exact registry. Each is
reviewed and classified individually as CPU-only. The runner captures diagnostic
stderr separately so native timing output cannot splice into libtest stdout;
the exact result/count validator and all discovery/ignore checks are unchanged.

The ignored bounded M5 allocation/lifetime witness was run separately with
`MTL_DEBUG_LAYER=1` and **PASS in 0.10 s**:

`deepseek_v41::forward::bind::tests::bounded_gpu_cache_backings_and_group_views_match_independent_quote`

Its combined toy quote is at most 1 GiB including a 128 MiB CPU envelope, with
normal admission before source allocation and again before binding. The pass
covers independent source/control/cache costs, sampled live allocation bounds,
distinct cache backings, shared group views and final buffer release. It is
**not** full-artifact execution, a continuous high-water measurement or timing
benchmark qualification. The frozen normal non-LTO dev libtest and retained log
are identified by SHA-256:

```text
libtest: 4de7d79569b42d50bb1f831ab784676c0ce6229b4cc7081b0f18516dd1ae9448
log:     a071197c2c1d3bf9c13d547a7439e2c71926c9c86c7a4cb506244627b3263cb4
```

A private metadata-only production-helper quote executable was also compiled
(SHA-256
`0435f8ca5fef486e0c68bd4d5df1f034837a3830b6293e28067e466bc891af16`). Its
first target attempt was **refused before context creation** because the
canonical lease was busy in another lane. That refusal remains a refusal, not
quotation or execution evidence.

## Actual GSQ Metadata Quote

A later authorized attempt **passes** on the actual GSQ artifact using source
`41d3d862`, live M5 production pricing, the normal canonical lease/wired guard
and no PATH changes. The retained quote JSON has SHA-256:

```text
38c666ffef1cebda74fdd39e2753ec8c8d9899fd6aad409d1cf59ad8b121762e
```

This is a **one-row, capacity-one, Full-only metadata quote**, with a finite
64 MiB caller CPU envelope. It is not 8K staged admission or full-artifact
construction/execution. The production active inventory selects 243 resident
caches, priced at **14,179,368,960 bytes = 13.20556640625 GiB**. Native controls
cost 5,292,032 bytes: 323 separately page-priced resources. Grouped views share
their bank backing; the cache sum remains distinct from reusable per-chunk
scratch.

| Metal term | Bytes |
| --- | ---: |
| Resident views R | 155220049920 |
| Resident caches C | 14179368960 |
| Native controls B | 5292032 |
| Load R+C+B | 169404710912 |
| One-row runtime T_M | 53629260 |
| Required R+C+B+T_M | 169458340172 |

| CPU term | Bytes |
| --- | ---: |
| Private shadows S | 0 |
| Binding envelope U | 225137600 |
| Caller envelope A | 67108864 |
| Load S+U+A | 292246464 |
| One-row runtime T_U | 20881402 |
| Required S+U+A+T_U | 313127866 |

The device page is 16,384 bytes. The 8-byte control sample reports heap size 8,
device alignment 256 and priced upper bound 16,384 bytes. These are live-device
prices, not an assumed CPU-only 16 KiB rounding model. The sampled Metal
allocation counter is unchanged at 458,752 bytes before/after planning; this is
not a continuous high-water measurement or proof of total process/framework
memory. The explicit caller envelope does not attest unmeasured parser, driver
or framework peaks.

No resident weights, binder, state or dispatch were constructed/executed by the
quote, and no whole-cold admission was performed. The 557-test CPU evidence and
bounded toy allocation/lifetime result above remain separate evidence classes.
The parent retains the private proof and verifies the quoted arithmetic.

The metadata quote alone left actual-artifact construction/execution open.
The subsequent bounded check below supplies that separate evidence; it does not
turn the one-row quote into 8K admission or a high-water measurement.

## Ordinary 8K Full/CED Artifact Smoke

The frozen canonical `qwen-bench` built from clean `bd9f4407` **passes** the
ordinary 8K Full/CED production-timing smoke on the actual GSQ artifact. The
parent-supervised canonical fail-fast invocation completes naturally in about
51 s under an actual 180 s timeout. This closes the bounded reviewed stack's
resident import, cache binding, fresh-state admission and Full/CED execution
check, not a stages/instrumentation or optimization campaign.

The artifact uses the default release fat LTO build. Its compiled/runtime
identity is MATCH at clean `bd9f4407`, SDK 26.2, Apple Metal 32023.864, diagnostic
features `[]`, no identity overrides, and source state:

```text
git-source-sha256-v2:c8cabb87dde77b8324a3d7ccead7ac5930716396eb899da36cb406efe61eff38
```

The frozen executable and retained suite JSON have SHA-256:

```text
qwen-bench: d6cf5b9623f33a92d73c2ddc79d3605bc83405b0c279fde3d068d6e0b073a3b5
suite:      db463fda9df47bc9f6b71ae5a68914bd293194fcb4da2050ef02fa440eb24026
```

Protocol `native_full_ced_abba_baab_production_timing_v2` completes one
10-case chronological packet: two warmup, four pilot and four retained cases,
five Full and five CED, with a fresh state per case. The two output arms repeat
that same packet; they are **not** 20 independent cases. All 10 cases have finite
full-vocabulary endpoint logits, match warm Full bitwise and share this digest:

```text
a2db19afd7c377c54160e90bdefc52dc04ab13a69b0f929abb34eab099da6edf
```

Every case reports its requested policy with required Full/CED admission and no
fallback. Preweight peak capacity, load admission and diagnostic whole-peak-now
assessment all admit. Ordinary loader, binding, state and prefill fresh gates
remain active and complete; these budgets are not OS reservations or continuous
allocation high-water measurements.

| 8K budget term | Bytes |
| --- | ---: |
| 243 resident caches C | 14179368960 |
| Load Metal R+C+B | 169404710912 |
| Runtime Metal T_M | 11527784728 |
| Required Metal R+C+B+T_M | 180932495640 |
| Load CPU S+U+A | 292246464 |
| Runtime CPU T_U | 1815144042 |
| Required CPU S+U+A+T_U | 2107390506 |

Full uses two default 4,096-row chunks. CED reports two input encoder chunks at
the same selected width, not an 8K encoder. This result gives no 8K-encoder or
per-stage profiling authorization and does not change another owner's seams.

Retained prefill host-wall observations are Full 6.115900625 / 5.821071 s and
CED 3.673439666 / 3.670493167 s. They are observations, **not a new performance
win or causal baseline**. The captured Qwen environment contains only the
canonical lease descriptor and `WAIT=0`; captured validation/injection entries
are null and the run is not instrumented. External injection is not independently
ruled out by that capture, and there is no external continuous frequency monitor
or formal performance claim.

Endpoint agreement within this one artifact/prompt-length Full/CED comparison
is not independent reference/model-quality qualification, intermediate-state
coverage or evidence for other contexts and workloads. The earlier 557-test CPU
source evidence, bounded toy witness and metadata quote retain their separate
scopes. The parent retains the private proof and owns final review, refreshed
main integration and the clean release handoff. **Main is not claimed merged**
by this checkpoint; no new source, stage or execution policy is authorized here.
