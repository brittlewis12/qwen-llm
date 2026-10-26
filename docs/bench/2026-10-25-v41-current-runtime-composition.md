# Current-Main CED Composition

This is a provisional current-main composition, not a transfer of frozen P0/P4
qualification. It merges frozen P4 `7398cb765178ba3ca0ccc022e12d9faef766497b`
onto `030e0b59ce4373d12269608bdbc9b516fa62b37e`. Fresh M5 and complete-model
qualification belong to the parent after the independent staging, execution
contract, request planning, acceptance and mapping repair lanes are integrated.
No automatic CED threshold, serving-origin reuse, or approximate bounded-128
policy is introduced. Full remains the default. Cold run/bench CED consumes the
whole logical prompt and uses ordinary scalar continuation.

## Composition Interfaces

- Keep GSQ companion lookup tables disk-only, bounded CPU control normalization,
  source bookkeeping, IQ1 native expert kernels, current Full chunk preference,
  and canonical grouped output-A binding geometry. Serialized 3D output-A is
  validated by the loader and canonicalized to the existing 2D binding view.
- `ColdPrefillPlan.loader_controls` and `loader_control_cpu_bytes` expose the
  loader subset already included in binding CPU. Charge
  `max(cold normalization, 6 * serialized control bytes) + retained controls +
  source bookkeeping`, without credit for shared weights or later phase overlap.
- `Plan::revalidate_sources` checks retained backbone and auxiliary FD stamps.
  `DiskEngram::source_groups` and `DiskEngram::revalidate_sources` remain the
  main APIs. Current-pair v2 additionally rejects loaded FD groups or normalized
  controls differing from the retained plan. This is not content authentication.
- Both active qualification inventories retain batching and soft-window sources
  and append loader-helper freshness checks. The frozen receipt is unchanged.
- The final private current-pair report is synced and made read-only while its
  resource lease is still held. Model import stays after combined admission.
- Product loading does not print retained-resource inventory. The native bench
  adapter and private current-pair explicitly emit the existing structured line.
- `SOFT_COALESCING_TARGET` is `None`. Parent constructs matched None/4 GiB builds
  after the mapping-only repair; this branch does not change that experiment.

## Explicit Native Schedules

`qwen-bench suite --v41-ced-pair --v41-ced-pair-schedule VALUE` accepts:

| Value | Chronology | Retained Runs Per Policy |
| --- | --- | --- |
| `interleaved` (also the default) | Existing warm AB, pilot ABBA, alternating main BAAB/ABBA | Even 2..=16 |
| `full-blocks-first` | Full warmup, two Full pilots, all Full main calls; then the corresponding CED stream | Even 4..=16 |
| `ced-blocks-first` | CED warmup, two CED pilots, all CED main calls; then the corresponding Full stream | Even 4..=16 |

Interleaved v2 protocol names, schedule and report meaning are unchanged. Blocked
modes use `v41-ced-pair-blocked-v1` inside paired diagnostics and launch
measurement, with distinct protocols:

- `native_full_ced_full_blocks_first_v1`
- `native_full_ced_ced_blocks_first_v1`
- `native_full_ced_full_blocks_first_production_timing_v1`
- `native_full_ced_ced_blocks_first_production_timing_v1`

The outer bench row schema stays unchanged. Every raw chronological case is
retained, including excluded warmup/pilot, with actual execution, phase, policy,
block, retention flag, elapsed time and endpoint hash. Summary samples contain
only main calls for their policy. There are exactly `6 + 2 * runs` cases (at most
38), two excluded pilots per policy and one live cold state at a time. In the
CED-first schedule, all heads first compare to warm CED and must also match warm
Full when it occurs; any mismatch stops without emitting a successful aggregate.

All schedules use identical deterministic input/endpoint and the same fresh
admission, no-fallback, finite-head, memory, completion and cleanup checks.
Blocked modes refuse diagnostic profiling. Canonical production timing still
requires the explicit production-timing flag and validation/injection settings
unset before child launch. Environment capture is before Metal creation; no
in-process environment mutation is used. Invalid schedule counts, shape/chunk
arguments and diagnostic combinations fail before model open/import. There is
no retry, variance rejection, free-floor gate, or automatic performance verdict.

## Pending Device Evidence

Focused optimized CPU verification of this provisional composition passes 160
top-level tests plus the isolated CPU RSS child (18 groups, zero failures).
Workspace all-target typechecking and formatting checks also pass. These checks
do not execute device/model witnesses or qualify the later repair composition.

Build/freeze fresh executables after all parent repairs. Bounded snapshot,
restore/growth and 256-step continuation tests already exist at
`deepseek_v41::forward::ced::tests::scalar_native_full_vs_ced_bitwise_state_logits_restore_growth_and_256_steps`
and
`deepseek_v41::forward::ced::tests::chunked::batched_f32_multichunk_4096_bitwise_restore_growth_and_256_scalar_steps`.
Their existence and CPU compilation are not device execution evidence. Serving
must continue rejecting CED-origin live state and snapshots until a separate
serving qualification. Preserve the bounded M5 matrix in
`2026-10-24-v41-ced-batches-m5-handoff.md`, plus current ignored inventory; do not
interpret zero tests or unsupported hardware returns as passes.

The private model witness is
`deepseek_v41::forward::ced::qualification::current_pair::current_full_vs_ced_paired_full_model`,
one exact invocation for each explicit N in `128, 2048, 2049, 4097, 8192`.
Each successful invocation must cover two arms, 257 state positions and 257
full-vocabulary vectors per arm, and 256 teacher-forced scalar steps per arm.
Five invocations mean 2570 state positions/vectors and 2560 scalar steps total.
External full-artifact verification and bounded M5 supervision remain required;
sparse CPU headers do not qualify absent model payloads or runtime arithmetic.
