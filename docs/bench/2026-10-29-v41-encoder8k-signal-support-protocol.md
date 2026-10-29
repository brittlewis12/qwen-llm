# V41 library-test signal support: source-only component

Baseline: `1bef2bb3`. This component extracts only
`crates/qwen-llm/src/metal/test_cancellation.rs` from reviewed source `2258eba6`,
blob `867e5c6f9d4e7cb46c3ca51c30ccbd6efda9a7cd`, byte-for-byte. Historical CPU
checks of that source are not fresh evidence for this composition. The base is
not fully CPU-qualified, and this component has not been Rust-built or executed.

The module is exposed only as `#[cfg(test)] pub(crate) mod test_cancellation`.
There is no automatic handler installation, production CLI handler change,
benchmark driver, or model/context call. Fixed-8K admission/execution policy,
kernels, math, scheduling, and the existing unfrozen performance gate are unchanged.

## Helper contract

The preserved helper installs process-wide POSIX INT/TERM dispositions only on
explicit `CancellationGuard::install`. Its handler only stores the atomic stop
flag. Process mutex ownership, active/stop state, signal masking, saved actions,
checked restoration, the restoration-error latch, non-Send ownership guard, and
nonpanicking Drop remain intact. A stop request is not command cancellation or
proof of completion. This does not establish portability beyond the reviewed
platform or compatibility with concurrent handler owners.

A future caller must install the guard before any context or resource creation,
pass `flag()` through model calls, and check `check()` before large allocations
and commits and after checked completion. Never abort between a commit and its
checked wait. Stop means no new work while already submitted work drains.
Workers, contexts, and autorelease pools must drain/tear down before explicit
checked `restore()` and subsequent lease-descriptor release. A restoration error
must remain a failure, not be hidden by Drop or treated as successful teardown.
CLI shutdown handlers do not automatically cover library-test processes.

## Two CPU controls

Only these exact nonignored CPU tests enter the narrow inventory scope:

- `metal::test_cancellation::tests::signals_wait_for_release_cpu`
- `metal::test_cancellation::tests::unwind_restores_handlers_cpu`

The first spawns fresh CPU-only children, targets each exact spawned PID with
INT and TERM, and requires the child to remain alive until the parent releases
simulated checked completion and restoration. The second confines handler
mutation to a fresh CPU subprocess and checks simulated resource teardown before
handler restoration on unwind, plus poisoned-owner refusal.

`V41_TEST_COOPERATIVE_SIGNAL_CHILD` is a test-only subprocess control, not a
runtime selector. It must be absent from the parent invocation; only the helper
sets it for its child. Tests contain bounded cleanup of the exact spawned
CPU-only child. That cleanup is never a GPU drain strategy or a process-group
recipe. A deadline, reaped process, or successful child status does not establish
GPU command completion. The controls simulate ownership/release, never GPU/model
work. Source inspection, formatting, and Python checks do not execute them.

## Remaining gate

Compose the separately reviewed base fixture correction before Rust qualification,
then review/freeze this source and run the two CPU subprocess controls under the
ordinary unleased runner. The inherited GPU-FD refusal, offline/locked Cargo,
exact classifications, and exclusion of GPU tests remain unchanged.

No signal has been sent by this component's source checks. No model, device
admission, GPU cancellation/drain, numerical, or performance qualification is
claimed. A future GPU harness still needs its own signal/resource ownership and
descriptor-safe checked-drain review and proof; none is added or queued here.
