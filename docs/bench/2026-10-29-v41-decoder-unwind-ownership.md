# Decoder submitted-command unwind ownership

## Source-only scope

Base: `444a32bccfe05fec951f9eb8cc53592724734092`, tree
`4481a0bf2aa1defc907ca315e7e892498702dad5`.

This repair is source-only and unbuilt. No Rust build, typecheck, test discovery,
CPU Rust test, signal test, Metal context, GPU test, model run, lease, install,
SSH, or publication is part of this change. Formatting, source review, inventory
validation and Git whitespace/staged-diff checks are the available evidence.
All actual fault execution and integration qualification remain owed.

The privately composed fixed-request qualification is not evidence for this
repair. No fixed-request semantics, defaults, math, admission, retention,
dispatch order, group depth, status precedence, scheduling, shaders, FFN,
Engram, prefetch, or pipeline selection are changed.

## Ownership repair

`PendingCed` owns one terminal blocking-wait obligation. All submission metadata,
including the timestamp, is constructed before commit. The owner is armed by a
plain boolean assignment immediately after commit, before any post-commit
bookkeeping or handoff. Returning, enqueueing and popping move the same owner;
there is no fallible reconstruction of ownership after submission.

Normal `ced_wait` blocks through that owner, discharges the lifetime obligation,
then performs the existing checked completion, profiling, command accounting and
completed-command accounting. A terminal command error discharges the wait but
does not accept its outputs or increment the successful-completion counter.
The check still returns the actual completion error. A normal owner drop does
not wait twice.

The fallback destructor only waits. It does not check/format errors, report
success, profile, count a command as complete, or accept status/output bytes.
`EncoderDrain` uses the same terminal obligation and retains its existing drain
observations. Decoder `PendingGroup` declares the command owner before its
status views, so field destruction waits before releasing them, including when
the group has already been popped into `finish_ced_group`. The queue is declared
after the ledger and seed; caller-owned pool/storage outlive the executor call.

This is a bounded Rust-unwind cleanup attempt, not a guarantee against foreign
exceptions, process termination, OOM abort, or a broken device/driver wait. The
fallback deliberately ignores completion acceptance and cannot prove every
submitted command succeeded. No global GPU tracker, fake empty drain command,
production panic catcher, or GPU hard kill is introduced. The fixture catches
panics only to inspect cleanup already performed by the actual owner. Existing
ordinary errors already drain pending commands, and the final normal loop has
at most one pending command with its own wait before error; neither is claimed
as a newly discovered ordinary-error defect.

## Fault source and exact inventory

New library selectors (prefix plus leaf shown here):

`deepseek_v41::forward::gpu_step::ced::batches::tests::unwind::`

- `default_pipeline_submitted_command_unwind_ownership_fault_matrix_gpu`:
  explicitly ignored, class `gpu`, bounded native toy fixture, debug layer and
  self-acquired lease required when eventually executed.
- `default_pipeline_successor_encode_cancel_ownership_gpu` and
  `default_pipeline_adopted_cancel_ownership_gpu`: explicitly ignored, class
  `gpu`, separate cancellation selectors; require fresh exact-process runs and
  specify no subsequent GPU work after their cancellation injection.
- `release_trace_rejects_missing_duplicate_and_late_terminal_waits_cpu`:
  nonignored, class `cpu`, host-only trace-validator negative controls. It does
  not instantiate Metal or claim to test the real command owner on CPU.

The three GPU selectors specify seventeen fault cases, all using the active
production `ced_decoder_groups_pipelined` entry and actual default two-layer
selection:

- Error, panic and cancellation after encoding a successor unit with the older
  submitted group pending (one submitted decoder command).
- Panic during first decoder-group encoding (no submitted decoder command and
  no decoder wait).
- Error and panic with the successor committed in a local `PendingGroup`,
  before queue adoption (two submitted decoder commands).
- Error, panic and cancellation after adopting the successor but before its
  speculative retirement (two submitted decoder commands).
- Error and panic after popping the older group, before its explicit wait,
  with the newer command pending (two submitted decoder commands).
- Panic after the older group's checked wait (two submissions, no duplicate
  older wait) and panic with the last single group popped before its wait
  (one submission).
- Older completion-check rejection and older FFN-status rejection with the
  newer command pending (two submitted decoder commands).
- Last single-group completion-check rejection and FFN-status rejection (one
  submitted decoder command).

The existing discarded-completion seam rejects AFTER the actual blocking wait;
it does not create a device fault or GPU recovery. The new FFN seam poisons one
host ledger word AFTER the blocking/checked wait, then uses the unchanged real
FFN validator. It is likewise synthetic, not a GPU-written fault. Both remain
GPU-class tests because real commands/resources are involved. All controls are
thread-local; no new environment knob is needed.

The small pipeline checkpoint adapter passes a distinct local synthetic flag
to the existing `checkpoint(Action::Cancel)`, which stores true into its flag.
It NEVER passes that action the real process-wide stop flag. Model calls retain
`Some(metal::test_cancellation::flag())`; the adapter checks that operator flag
before and after the checkpoint. The driver checks real cancellation after the
caught result and before acceptance/further work, and again through teardown.
A real operator stop fails the fixture; it is not accepted as a successful
synthetic-cancellation witness. There is no clearing/resetting of a live stop
flag, guard reinstallation to hide a stop, or extra tracking framework.

The preallocated, non-panicking destructor trace observes the real owner at
submission and terminal wait, plus lexical before-release witnesses for each
group's status views and the seed/ledger/backing-resource scope. Assertions
require exactly one terminal wait per submission, exact checkpoint/group
identity, the expected submission count, each group's own wait before its
status release, and all waits before
shared backing release. The fixture also specifies transient-allocation return
while the poisoned state remains alive, unchanged addresses/position/origin,
rejected state reuse and cleared public decoder progress. Non-cancel cases
specify a same-context healthy follow-up with validated counters. Cancellation
cases specify no healthy follow-up or later GPU work; that proof is deliberately
separate. These are source assertions, not executed proof.

Each native fixture installs `metal::test_cancellation::CancellationGuard`
before its lease/context/resources, queries INT/TERM masks as unblocked, rejects
the CPU signal-helper child environment, passes `Some(flag)` through GPU calls,
and tears down resources and autorelease pools before explicit checked restore.
No signals are sent and no GPU children are spawned. Cancellation selectors
must each run alone in a fresh exact test process, not together in a broad
ignored-test invocation. Their local injection is not proof of actual operator
signal handling. Fixtures retain the existing bounded toy F32 arithmetic helper
and memory admission, not fixed-request controls.

The older batch fault checkpoint names live in the serial
`ced_decoder_groups` implementation, not the active pipelined implementation.
They are not claimed as pipeline coverage. Existing encoder ignored fixtures
and their drain observations are retained; their rerun is still owed too.

## Integration still owed

After source reviews and composition with the independent support/harness
lanes: unleased builds/typechecks, exact inventory discovery, host-only
acceptance, then separately authorized bounded GPU fault execution with debug
validation, checked signal restoration and healthy follow-ups. No actual
execution result is asserted by this protocol.

Separate follow-up: production `gpu_step.rs::submit_split` has an analogous raw
committed-vector unwind gap. It is unchanged and not covered by this repair;
the scalar test default's single command is not split-path proof.
