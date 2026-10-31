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
actual target attempt was **refused before context creation** because the
canonical lease was busy in another lane. That refusal is not a quote result:
there is still no actual GSQ active-cache count or live-device price, no 8K
staged admission, and no full-artifact execution evidence from this work.

V4 main integration remains pending. The parent retains the private proof and
owns the next fail-fast quote attempt after the concurrent lane releases its
lease; this checkpoint does not authorize a retry. Actual-artifact quotation,
integration, the applicable qualified build and authorized execution checks
remain prerequisites for the next cache-enabled profile.
