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
even `--runs 2..=16` retained samples per policy. Warmup is mandatory.

- A is Full forward; B is shrinking CED. Captured AB warmup calls are excluded
  from statistics; an excluded ABBA pilot collects two samples per policy, then
  retained BAAB/ABBA blocks alternate. The CLI checks each policy's finite,
  positive max/min <=1.5,
  including its pilot baseline, before scheduling further cases. It never
  treats the intended difference between policies as instability.
- Pre-import planning uses the public `preflight_cold_prefill` ledger: exact
  resident windows and private shadows, binding allocations, one request state,
  the sequential Full/CED workspace envelope and caller CPU storage. The pair
  supports at most 256 layers and 1,048,576 logits. This is not a reservation;
  normal loader, state and prefill admission remain active.
- Every completed native report must name the requested policy without
  fallback. CED must finish all encoded commands and report observed incremental
  Metal allocation no larger than its accepted estimate. Every finite endpoint
  head must match warm Full bitwise; state and autoreleases drain between cases,
  including on validation errors and cancellation.
- Timings cover host-wall production prefill and the endpoint logits copy, not
  state allocation, GPU-only time or model loading. Schema-v4 output has one row
  per policy. Warmup/pilot are excluded from statistics but retained in both
  rows' chronological diagnostics, along with reports, fingerprints and budgets.
- Use `MTL_DEBUG_LAYER=1`, the production lease and the stock runtime/arithmetic
  environment. Profiling overrides and an active dispatch observer are refused.
  Run under external continuous host supervision, such as `pilot.Session`;
  the CLI's timing gate does not monitor global VM health. A nonzero exit or
  unhealthy/missing telemetry invalidates the run; do not automatically retry.

This measures synthetic cold-KV prefill, not artifact quality, sampled
continuation or the separate bounded-128 policy. A matching endpoint fingerprint
is not a replacement for the strict full-vocabulary and causal-state CED
qualification. Ordered cold Full-then-CED diagnostics are not speedup evidence.

### Instrumented Pair Diagnostic

Add `--v41-ced-pair-diagnostic` only for a separately authorized diagnostic
series; it requires `--v41-ced-pair`. It does not retry a rejected benchmark,
add warmups, change arithmetic, enable sampled stages, or relax any shape,
debug-validation, lease, admission, fingerprint, stability or host-validity
requirement. Start with a bounded 2K series (`--pp 2048 --runs 2`: at most ten
calls including warmup/pilot), not an automatic escalation to 8K.

- Every diagnostic result row, chronological case and diagnostic stderr event
  carries `instrumented: true` and `performance_evidence: false`. The row's
  paired protocol is `instrumented_full_ced_abba_baab_diagnostic_v1`, distinct
  from the uninstrumented native protocol. Instrumented throughput columns must
  not be used as performance evidence, even if all stability checks pass.
- Case-tagged JSON observations stream and flush to stderr before the next
  stability decision, so completed observations survive a later rejection.
  Each profile is streamed once and dropped, never copied into both rows'
  chronology. Finish events explicitly mark cross-case fingerprint/stability
  checks pending; they are observations, not acceptance. Missing required process
  counters, profiling setup failures and observation-write errors stop scheduling
  rather than silently disabling data.
- The existing native prefill timer still excludes state allocation and includes
  endpoint logits copying, but is now instrumented. A separate unsampled
  `forward::profile::Scope::begin()` encloses the outer case: state allocation,
  prefill, report handling, endpoint fingerprinting, state destruction and case
  autorelease-pool drain. Its historical `token_wall_ms` field is **outer-case
  wall time**, not exact native prefill time. Model import/binding are outside it.
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
- Diagnostic preflight charges an additional fixed 16 MiB CPU reserve on top of
  the normal 64 MiB caller envelope (80 MiB total). At the existing bounds of
  8192 tokens, 256 layers and 16 runs/policy, one unsampled profile has at most
  4096 operation records, including fixed aggregate host spans. Preweight checks
  bound the producer by encoder chunks plus seed/output chunks and a scalar tail
  per decoder layer, independently of run count. A pure collector fixture counts
  the production command roles from real CED plans at default/256-layer geometry,
  including chunk boundaries, scalar tails and window extremes. A CPU-only
  maximum-payload streaming fixture charges live operation storage, serialized
  bytes and 1 MiB fixed slack against that reserve. No 38-profile retention or
  host-budget bypass is permitted.

This mode localizes unexplained wall time; it does not by itself establish
file-cache misses, a thermal cause, interference, a lifetime defect or a speedup.
