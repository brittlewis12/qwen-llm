# Bench Notes

For commands: `scripts/bench/family.py --help`.
For history: `docs/bench/<stamp>-family/`.
For priorities: `docs/PERF-ROADMAP.md`.

## What we copy from llama-bench

- `build_commit` on every row, so a comparison is decodable later.
- `pp<N>` / `tg<N>` as the shape vocabulary.
- JSON output as the machine-readable surface; stderr stays human-only.

## llama.cpp comparator provenance

- Scoreboard scripts default to `scripts/bench/llama-cpp.lock.json`, built by
  `scripts/bench/ensure_llama_cpp.py` into `~/.cache/qwen-llm/llama.cpp`.
- Ambient local llama.cpp binaries are allowed only as explicit one-offs with
  `--allow-unpinned-lcpp`; they are not canonical family baselines.
- In `scripts/bench/family.py`, `--shapes` is exact: `--shapes tg128` runs only
  `tg128`; omit `--shapes` for the default family grid.

## What we don't

- `MTL,BLAS` in lcpp's `backends` is a registration artifact — BLAS does
  no work on Qwen3.5/3.6 at any shape we measure (audit in
  `docs/PERF-LOG.md`). Treat it as informational.
- `--runs 3` not `-r 5`. The M4 Max stays stable enough that the extra
  reps mostly pay in elapsed time.

## Opinionated choices

- **No model-size bandwidth row for MoE models.** Naive `model_size x t/s`
  overstates by ~10x for A3B / A10B. A MoE packet may instead report
  census-derived `active_bytes_per_token`, effective active-byte bandwidth, and
  percent of the maintained stream anchor when it pins the dense/shared/routed
  role decomposition and quant byte widths. These fields belong in the JSON
  schema before becoming scoreboard authority.
- **One JSON file per (engine, model, shape).** Lets you re-run a single
  cell of the matrix without disturbing the rest.
- **Sanity flags inline in the digest.** `pp1024 < pp512` and friends
  appear at the bottom of the README, not in a separate file.

## Known gaps

- No cross-day / thermal variance tracking — we mitigate by running
  lcpp and qwen in the same sweep so drift cancels in the ratio.
- No auto-diff against the previous baseline.
- `qwen-bench prefix-cache` reports a TTFT proxy: cold prefill or warm restore
  plus suffix prefill, then first-token selection and one additional forward
  pass after that token is chosen.
- No census-to-active-bytes exporter in the bench tooling yet.

## Bounded memory experiments

- New mapping or residency behavior starts with a tiny representative fixture,
  then one fitting pilot, then repetitions. File size and virtual mapped bytes
  are not physical-memory bounds; private/COW exposure must be priced explicitly.
- Combined host admission remains mandatory when the raw process-limit signal
  is zero. Metal demand follows Metal's working-set limit; additional CPU/private
  exposure follows CPU availability. Combined demand must fit actual physical
  capacity. Shared Metal mappings are not charged against CPU cache availability.
  Do not add a physical-memory percentage cap or reserve on top of those limits.
  Unknown telemetry is not a zero-cost allowance.
- V4.1's `deepseek_v41_storage_plan` example reports the loader's exact windows
  and private exposure using supplied page/device geometry, without importing
  weights or reading tensor payloads. It is a storage audit, not runtime fit;
  runtime buffers and CPU demand still need admission. See the
  [memory-safety packet](bench/2026-10-23-v41-memory-safety.md).
- Compare execution policies inside one resident model when possible, resetting
  request state between repetitions. Distinguish cold KV from cold model loading.
- `scripts/bench/pilot.py` supplies a reusable pilot runner. It requires stable
  native timing samples and healthy host telemetry before further launches,
  monitors health every two seconds, and supports a deadline and stop file.
  Its default memory supervision observes VM counts without a percentage or
  free-page floor; an observed OS-critical pressure event stops the owned job.
  Missing optional pressure data is recorded as unknown, not assumed normal.
  Cancellation signals only the owned child, with bounded graceful waits; it
  does not retry, signal process groups, or force-kill an unresponsive child.
- The runner is secondary protection, not permission to exceed loader admission
  or rely on cancellation after the kernel stalls. Native binaries already take
  the production Metal lease; do not wrap them in another lease holder.

## Native V4.1 Full/CED Pair

`qwen-bench suite --v41-ced-pair` is an experimental, pp-only comparison using
one resident import and a fresh request state for each native call. It accepts
exactly one `--pp` in 2..=8192, `--depth 0`, no `--tg`, no chunk override and
even `--runs 2..=16` retained samples per policy for the default interleaved
schedule. Warmup is mandatory.

- A is Full forward; B is shrinking CED. By default, captured AB warmup calls are excluded
  from statistics; an excluded ABBA pilot collects two samples per policy, then
  retained BAAB/ABBA blocks alternate. Pilot/main samples must be finite and
  positive; spread alone never stops scheduling. Each policy's `native_dispersion`
  reports `min_ns`, `max_ns` and `max_over_min` separately for `pilot_plus_main`
  and `retained_main`, never pooling policies. Schedule/pilot status `completed`
  describes collection only, not stability or accepted performance. All four v2
  protocols replace the v1 spread gate; historical v1 records keep their meaning.
- Pre-import planning uses the public `plan_cold_prefill` ledger: exact resident
  windows and private shadows, binding allocations, one request state, the maximum
  complete sequential Full/CED workspace envelope and caller CPU storage. Suite
  owns the typed plan across its sole resident import and binding. The pair
  supports at most 256 layers and 1,048,576 logits; normal caller storage is
  bounded by 64 MiB. Planning is not a reservation.
- Immediately before the resident factory, `assess_load` requires the **full cold
  peak** to fit device capacity, actual host capacity and any positive
  process limit, with valid telemetry and checked arithmetic. Ordinary fresh
  availability admission applies to the **load phase** (resident windows, private
  shadows, binding and caller CPU), not to all future workspace simultaneously.
  No additional percentage headroom is applied; active file cache receives no
  credit. A whole-peak availability refusal alone can therefore coexist with an
  admitted load, but a peak-capacity or load refusal stops before the factory.
- After binding and before **every** fresh case state, `assess_runtime` uses
  actual current memory signals to admit one request plus the maximum complete
  Full/CED envelope. It recharges the full binding and caller CPU envelopes;
  resident Metal and private shadows are already in live signals, not added
  again. The prior state's logits/commands/state and case autorelease pool have
  drained first. All ordinary device, ownership, process, telemetry and fresh
  availability constraints remain active, as do the inner loader, `new_state`,
  Full and CED gates. A runtime refusal creates no new state or prefill callback,
  produces no execution report or valid result row, and releases the resident
  factory normally. There is no retry or bypass.
- JSON `preweight_plan` records the static request, workspace, binding, caller,
  full-cold-peak, load and runtime-gate CPU budgets. `peak_capacity_passed` and
  `peak_capacity` describe capacity only; `load_admission` admits only the load.
  `whole_peak_now` is explicitly diagnostic, may deny availability, and is **not**
  runtime admission. Fresh runtime signal reports are not retained per case.
  The strict `preflight_cold_prefill` wrapper still requires ordinary admission
  of the whole cold peak for its separate qualification callers.
- Every completed native report must name the requested policy without
  fallback. CED must finish all encoded commands and report observed incremental
  Metal allocation no larger than its accepted estimate. Every finite endpoint
  head must match warm Full bitwise; state and autoreleases drain between cases,
  including on validation errors and cancellation.
  Production CED admission may still fall back to Full. The pair rejects that
  report after guarded execution; a later CED denial does not promise zero GPU
  work. Neither the production fallback nor any inner admission is changed.
- Timings cover host-wall production prefill and the endpoint logits copy, not
  state allocation, GPU-only time or model loading. Schema-v4 output has one row
  per policy. Warmup/pilot are excluded from statistics but retained in both
  rows' chronological diagnostics, along with reports, fingerprints and budgets.
- By default use `MTL_DEBUG_LAYER=1`: this retains the validation-on protocol
  `native_full_ced_abba_baab_v2`, not canonical production throughput. The explicit
  production-timing regime below is separate. Both require the production lease
  and stock runtime/arithmetic environment. Profiling overrides and an active
  dispatch observer are refused.
  Run under external continuous host supervision, such as `pilot.Session`;
  the CLI does not monitor global VM health. A nonzero exit or unhealthy/missing
  telemetry invalidates the run; do not automatically retry.

This measures synthetic cold-KV prefill, not artifact quality, sampled
continuation or the separate bounded-128 policy. A matching endpoint fingerprint
is not a replacement for the strict full-vocabulary and causal-state CED
qualification. Native samples are observational, not automatic performance-gain
acceptance. Ordered cold Full-then-CED diagnostics are not speedup evidence.

### Explicit P4 Mapping

Add `--v41-ced-pair-p4` to select a 4 GiB (`4294967296` byte) soft resident
mapping coalescing target for both Full and CED, including 8K pairs. It requires
`--v41-ced-pair`; ordinary production and unselected pairs retain None/P0.
The selected cold plan carries the target into weight loading, which revalidates
the storage geometry before import. Larger singletons remain eligible up to the
device limit; this is not a 4 GiB tensor cap or an artifact/shape allowlist.
Early launch and paired result measurement metadata record
`resident_soft_coalescing_target_bytes` (null for P0), as does the preweight
report. Paired widths, arithmetic, schedules and memory gates are unchanged.
No environment override is involved.

### Explicit Blocked Schedules

`--v41-ced-pair-schedule interleaved` preserves the default v2 chronology and
protocol meaning. The native-only alternatives `full-blocks-first` and
`ced-blocks-first` each run one policy's warmup, two excluded pilots and all
retained main calls, then the other policy's corresponding stream. They require
even `--runs 4..=16`, refuse `--v41-ced-pair-diagnostic`, and retain the same
input, endpoint, bounds, admission, no-fallback and cleanup checks. They do not
retry or reject variance. CED-first heads must also match warm Full when its
stream starts; no successful aggregate is emitted on a mismatch.

Blocked launch/paired diagnostics carry `v41-ced-pair-blocked-v1`, the explicit
schedule, distinct native protocols and every chronological case, including
excluded phases. The outer bench row schema stays unchanged; only main calls
contribute per-policy summary statistics. See
[`2026-10-25-v41-current-runtime-composition.md`](bench/2026-10-25-v41-current-runtime-composition.md)
for exact protocol names, counts and pending qualification. Launch environment
capture remains before Metal, without in-process mutation. Production timing
still requires the explicit regime below.

### Explicit Production Timing

Add `--v41-ced-pair-production-timing` to request the production-timing regime;
it requires `--v41-ced-pair` and defaults to false. Launch the process with
`MTL_DEBUG_LAYER` and `MTL_SHADER_VALIDATION` **unset**, not `0`, `false` or empty.
The CLI never changes environment variables in process. GPU correctness tests
and numerical/causal-state qualification continue to require `MTL_DEBUG_LAYER=1`.

- Production timing refuses every present `MTL_DEBUG_LAYER*` or
  `MTL_SHADER_VALIDATION*` setting, plus `METAL_DEVICE_WRAPPER_TYPE`,
  `DYLD_INSERT_LIBRARIES`, `DYLD_IMAGE_SUFFIX` and `MTL_CAPTURE_ENABLED`, regardless
  of value. These are a small set of known validation/wrapper/instrumentation
  controls, not a claim to detect all external tooling. Validation-on mode still
  requires exactly `MTL_DEBUG_LAYER=1`; error/warning-mode and shader error-report
  overrides (`*_ERROR_MODE`, `*_WARNING_MODE`, `*_ENABLE_ERROR_REPORTING`,
  `*_REPORT_TO_STDERR`) are refused in both regimes. Other shader validation
  settings are recorded in validation-on mode, not described as API-only.
- Once parsed arguments identify a pair, the entry point captures one relevant
  environment `BTreeMap` and writes and flushes JSON `v41_pair_launch_requested`
  to stderr before build-identity checks, model opening or Metal context creation.
  The captured object is passed into Suite without recapture. The existing
  process-wide perf-ablation guard still runs before ordinary argument parsing:
  on refusal only, it attempts non-exiting argument parsing to identify a pair
  and emit its request before returning the original refusal. Help, invalid
  arguments and non-pair commands do not imply paired intent. No guard is waived,
  and launch-write or flush failure stops before model/device work.
  The event includes the requested protocol, regime and known Metal validation
  values even when ablation, build-identity or environment checks reject launch.
  Known primary keys are present as `null` when unset; `"0"`, `"1"` and `""`
  remain distinct strings. Present prefixed controls and stock-runtime override
  controls use the same map, but disallowed QWEN runtime override values are
  redacted to presence markers (`disallowed_runtime_values` is
  `"presence_only_not_recorded"`); rejection is unchanged. Recorded OS-string
  values use explicitly labeled lossy UTF-8 conversion. Unrelated environment
  variables are not captured. The launch event is not acceptance or performance
  evidence.
- Results retain this captured provenance under `paired_benchmark.measurement`,
  once per result row rather than per chronological case. No summary-time
  environment reread occurs; legacy `metal_debug_layer` comes from the same map.
  Regimes are labeled `validation_on_requested` or `production_timing_requested`,
  with `effective_validation: "not_verified_external_injection_possible"`.
  `canonical_production_timing_requested` is true only for uninstrumented
  production requests. It is a regime label, not proof of effective validation
  state, host validity, qualification or a gain.
- Native production timing uses
  `native_full_ced_abba_baab_production_timing_v2`. Neither native regime enables
  a profiling scope, sampled stages or pipeline-cache counters. Lease,
  device/process/host accounting, all staged/inner admissions, endpoint fingerprints,
  finite-positive samples, fallback rejection and the 64/80 MiB caller budgets
  remain unchanged. This adds no
  pinning, residency, retry or admission-bypass mode.
- Before reporting a production gain, the offline controller must match the
  artifact, prompt/capacity and seed and compare complete 129,280-logit endpoint
  head hashes against the validation-on qualified reference. The CLI does not
  enforce an external-reference flag or implement that controller check. When
  arithmetic sources are unchanged, reuse the existing all-257-head/56-state
  proofs at 128/2K/8K; this timing-only protocol does not require repeating those
  large qualifications. Within-run equality to warm Full is not the external
  validation/production bridge.

This separation follows the established validation-on correctness versus
validation-unset throughput practice in
[query batching](bench/2026-10-19-v41-query-batching.md),
[grouped tradeoffs](bench/2026-10-20-v41-grouped-tradeoffs.md) and
[IQ2 grouped prefill](bench/2026-10-21-v41-prefill-iq2-grouped.md). The
[toolchain baseline](bench/2026-10-17-m5-toolchain-baseline/README.md) likewise
distinguishes API-validation timings from production throughput. None of this
establishes validation as the cause of unexplained wait jitter. Compare gains
within the same requested regime, not validation-on against production timing.

### Instrumented Pair Diagnostic

Add `--v41-ced-pair-diagnostic` only for a separately authorized diagnostic
series; it requires `--v41-ced-pair`. It does not retry a rejected benchmark,
add warmups, change arithmetic, enable sampled stages, or relax any shape,
selected validation regime, lease, admission, fingerprint, sample-validity or
host-validity requirement. Start with a bounded 2K series (`--pp 2048 --runs 2`:
at most ten calls including warmup/pilot), not an automatic escalation to 8K.

- Every diagnostic result row, chronological case and diagnostic stderr event
  carries `instrumented: true` and `performance_evidence: false`. The row's
  paired protocol is `instrumented_full_ced_abba_baab_diagnostic_v2` by default.
  Combining with `--v41-ced-pair-production-timing` is supported for matched
  attribution under `instrumented_full_ced_abba_baab_production_timing_diagnostic_v2`.
  Both are distinct from native timing; diagnostic stderr events also name the
  protocol/regime without repeating the environment. Instrumented throughput
  columns must not be used as performance evidence, even if the schedule completes;
  `canonical_production_timing_requested` remains false.
- Case-tagged JSON observations stream and flush to stderr before the next
  cross-case validation, so completed observations survive a later rejection.
  Each profile is streamed once and dropped, never copied into both rows'
  chronology. Finish events explicitly mark cross-case fingerprint and sample
  validation pending; they are observations, not acceptance. Missing required
  process counters, profiling setup failures and observation-write errors stop
  scheduling rather than silently disabling data.
- The existing native prefill timer still excludes state allocation and includes
  endpoint logits copying, but is now instrumented. A separate unsampled
  `forward::profile::Scope::begin()` encloses the outer case: state allocation,
  prefill, report handling, endpoint fingerprinting, state destruction and case
  autorelease-pool drain. Its historical `token_wall_ms` field is **outer-case
  wall time**, not exact native prefill time. Model import/binding and the fresh
  pre-case runtime assessment are outside it. Runtime admission also precedes
  the diagnostic start snapshot/event, so a denied case starts no profile scope.
- Resources are captured before the case and after state drop/pool drain:
  process page-ins, physical disk-read bytes, current RSS/physical footprint and
  current Metal allocation, with signed memory deltas. These are process-wide
  boundary observations, not continuous peaks or attribution to Engram alone.
  Pipeline miss/compiler-time deltas use suite's fresh, private context/cache.
  Suite enables counters once before loading/binding and never resets or rolls
  them back across cases. No shared-context metric scope is introduced: context
  destruction is cleanup on return/error/unwind, without mutating any other
  context. State and profile scopes also unwind safely.
- Command GPU durations are valid completed-command timestamps only; invalid
  timestamp counts remain visible. Commit/wait wall includes GPU execution:
  do not add GPU duration to it. The explicitly unattributed gap is outer-case
  wall minus command encode and commit/wait wall, **not CPU time**. It includes
  state work, admission, preparation, reporting and teardown not enclosed by
  command timers. Aggregate Full/CED-encoder host-preparation spans expose
  allocation/upload work and nested Engram read/decode/upload work; these spans
  overlap and are not additive. No per-row tracing or new lookup cache is added.
- Diagnostic planning charges an additional fixed 16 MiB CPU reserve on top of
  the normal 64 MiB caller envelope (80 MiB total). At the existing bounds of
  8192 tokens, 256 layers and 16 runs/policy, one unsampled profile has at most
  4096 operation records, including fixed aggregate host spans. Preweight checks
  bound the producer by encoder chunks plus seed/output chunks and a scalar tail
  per decoder layer, independently of run count. A pure collector fixture counts
  the production command roles from real CED plans at default/256-layer geometry,
  including chunk boundaries, scalar tails and window extremes. A CPU-only
  maximum-payload streaming fixture charges live operation storage, serialized
  bytes and 1 MiB fixed slack against that reserve. No 38-profile retention or
  host-budget bypass is permitted. The existing maximum-caller-payload regression
  includes the staged budget/assessment and measurement provenance metadata
  without increasing the normal 64 MiB caller envelope or diagnostic 80 MiB total.
  Its provenance fixture uses more than 1,000 relevant keys and non-NUL control
  values requiring JSON escaping within a 240 KiB ARG_MAX-like input budget,
  including entry/pointer costs. This is bounded stress coverage, not a proof
  over every possible launch environment; the isolated RSS regression remains
  observational, not a formal heap bound.

This mode localizes unexplained wall time; it does not by itself establish
file-cache misses, a thermal cause, interference, a lifetime defect or a speedup.
