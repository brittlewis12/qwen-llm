# Lens Workbench Reconstruction

## Status And Preservation

The Lens workbench was developed as uncommitted changes in `feat/lens-web`.
That branch contains no feature commits and is not an integration deliverable.
This document records an honest reconstruction, not recovered development history.

Reconstruction starts at main `8bc9e6b739e953023a1f2ffc5dc3379c22758e00` on
`reconstruct/lens-web`. The original worktree remains an untouched reference at
`2d158a71271c2e9522f745df062fdde4e153c935`, including its staged, unstaged and
untracked work. Do not reset it or apply its complete patch to current main.

A private preservation package is in the sibling directory
`qwen-llm-lens-preservation-20261002`. It contains a Git base bundle, binary-capable
diffs, original index and index inventory, complete tracked/untracked source,
selected ignored browser/numerical evidence, checksums and a verified restore.
It is a local backup, not a product commit or a published artifact.

- Source inventory: 2,827 tracked paths and 83 untracked paths.
- Archive: 3,938 entries, 62,792,100 compressed bytes.
- Archive SHA-256:
  `124686cbbf365b8c798fda478e89085c1cee9cf7f85b66455215284c179ac040`.
- Independently restored clone matches leaf contents, modes and symlink targets,
  staged index entries, staged/unstaged status, and the full worktree diff.
- Excluded rebuildable state: `target`, `web/dist`, `web/node_modules`; ephemeral
  `web/.browser-test/profile-*` browser profiles are also excluded. Retained
  `live-*` / `oracle-*` outputs, arrays, logs and local test configuration are kept.
- Earlier preservation stashes remain available but do not contain the latest UI:
  `c91bf5af7f70f459f60ab3e6d5d47fc4c29e15df` and
  `00984cf0a84b4c0dfff8c8d63f11138c8e6ccee8`.
  Their references are recorded; these older stash objects are not in the base
  bundle. The latest working state is independently preserved by this package.

The preservation script and `verified.json` live in the private package. No
private test configuration, model paths, browser data or generated bundle belongs
in this reconstruction's source commits.

## Recovery Rules

1. Preserve behavior, not merely files. A component is not recovered until its
   consumers and relevant tests work on this branch. No silent feature deletion.
2. Prefer current-main contracts over obsolete integration adapters. Mark something
   upstream-superseded only after demonstrating equivalent behavior.
3. Every commit describes one intention, builds with its ancestors and carries its
   required wiring and tests. No commits that need a later commit to work.
4. Internal refactors may precede product slices if current consumers exercise
   them. The first product milestone must be a usable diagnostic workflow.
5. Commit and hand off each completed slice before starting the next. Do not
   accumulate another feature-sized dirty worktree or fabricate historical dates.
6. Qualify against an exact main base. If main advances before integration, inspect
   that delta and rerun affected checks; never call old evidence current.
7. No merge into main, push, forced history change, or modification of another
   thread's worktree is implied by reconstruction commits.

## Behavioral Ledger

Paths below refer to the preserved source tree unless marked as current main.
`reuse` means a salvage candidate, not a claim of completion; `rework` identifies
integration that cannot be copied wholesale. Rows remain pending unless a partial
recovery is explicitly recorded below.

| ID | Behavior and source | Disposition / dependencies | Required gate |
| --- | --- | --- | --- |
| R01 | Bounded regular-file access: `bounded_file.rs`, Lens readers | Shared readers recovered with current CLI consumers; future HTTP reader wiring travels with its consuming slice | CPU symlink/type/length/mutation tests; all reader consumers build |
| R02 | Shared authored scopes and operation semantics: `lens_scope.rs`, `lens_intervention.rs`, `lens_run/{plan,execute,sweep}.rs` | Partially recovered: current-main scopes/validation, direction/action wire forms and normalization extracted with existing CLI consumers; lowering and behavior changes remain pending | Current CLI wire/binding/normalization/lowering regressions; no new service dependency |
| R03 | Typed prefills and annotated input: `lens_input.rs`, `model_request.rs`, `messages.rs`, `open_responses/render.rs`, CLI callers | Rework against current renderers; R02 uses existing span types only | Exact prompt bytes, token positions, reasoning-only continuation, tools and house/upstream rendering |
| R04 | Deployment binding and asset verification: `linear_transport{.rs,/deployment.rs,/cpu_fixture.rs}`, `full_lens/access.rs` | Reuse after R01; retain main's expected-profile checks | Binding mismatch refusal, retained payload hashes, CPU-before-Metal admission, no implicit transfer override |
| R05 | Shared ordinary execution: `ordinary_executor.rs`, `qwen/decode.rs`, `lens_run/execute.rs` | Rework around main's current decode paths; not an automatic replacement | CLI/serve sampling, cancellation, terminal nonconsumption and telemetry remain equivalent |
| R06 | Owner queue and CPU HTTP coordination: `serve/{control,queue,request_profile}.rs`, backend/HTTP wiring | Partially recovered: immutable request profiles and serial owner activity accounting; concurrent coordination still pending | Idle/request-finished/shutdown ownership, busy admission, cancellation/disconnect, JSON/SSE protocols |
| R07 | Durable job metadata: `serve/jobs/{state,store,preview}.rs` | Reuse with R06; distinct from main's durable model snapshots | Exact-key acceptance/retry, bounded publication, recovery and corruption handling, history without inference |
| R08 | Native request/routes/preconditions: `serve/lens_http/*`, `serve/native/preconditions.rs` | Reuse schema where compatible; R03/R06/R07 | Unknown-field rejection, local HTTP checks, binding coverage, stale rejection before acceptance, accepted-key recovery |
| R09 | Native baseline and observation lifecycle: `serve/native/{mod,execute,writer}.rs` | Rework admission around current resident/cache budgets; R05-R08 | Disconnected completion, cancellation, independent outcomes, bounded writer, isolated diagnostic state |
| R10 | Plain original-forward readouts: `serve/native/{readouts,observe}.rs` | Reuse after R09, not transformer replay | Scope/token coordinates, shared heads, unchanged samples, no terminal fabricated readout; bounded live check |
| R11 | Fitted readouts and direction staging: `serve/native/{registry,interventions}.rs`, `workspace_lens/*` | Reuse after R04/R09/R10 | Registered identity, matrix integrity, bounded ready-only staging/workspace, independent numerical oracle |
| R12 | Ordered scoped interventions: `serve/native/interventions/*`, `lens_intervention.rs` | Reuse after R02/R11 | Exact order/scopes, zero controls, deployed covector semantics, independent transformation checks |
| R13 | Retained full scores/source arrays: `serve/jobs/arrays.rs`, native observer/writer, `web/retention*` | Reuse after R07/R10 | Raw admission, dual-watermark durability, digest/finite/shape checks, offline rank/entropy, no implicit fetch |
| R14 | Whole-site pre/post capture: `metal_forward/token.rs`, `serve/native/measurements.rs` | Reuse after R12/R13; do not imply per-operation intermediates | Before/after placement, independent layer sets, alias refusal, actual-vector metrics and zero controls |
| R15 | Bun client and same-origin assets: `web/{build,dev,proxy}.*`, `serve/assets.rs` | Reuse with main's HTTP surface; R06/R08 | Local serving/proxy policy, manifest paths, production bundle and mobile/desktop loading |
| R16 | Durable browser submission/history: `web/{api,contract,durable,storage,jobs,history}.*` | Reuse with R07-R09/R15 | Persist-before-POST, identical recovery bytes, explicit cancellation, partial pages, other-client history |
| R17 | Draft and identity safety: `web/{draft,bindings,editor,diagnostics}.*` | Reuse with R03/R08/R16; required from first submission UI | No silent retarget, exact scope pinning, historical copy, async-edit preservation and failed persistence |
| R18 | Direct manipulation workbench: `web/{viewer,token-navigation,token-navigator,App,styles}.*` | Reuse after associated readout/intervention/retention slices | Saved-token/layer/candidate selection, missing-data distinctions, stable refresh, scoped staging, phone parity |
| R19 | Evidence and qualification: native CPU/live tests, HTTP fixtures, `web/*check.ts`, docs | Rework gates to current main; preserve old evidence separately | No old-base pass promoted to current-main qualification; bounded live checks where numerical, lifecycle, cache/durability or memory-admission changes require them |

CLI comparison/output/sweep changes must be accounted for under R02-R05, not
dropped because they are outside `web/`. Documentation and fixtures travel with
the behavior they describe. An unchanged/copied file is not proof of recovery.

## Integration Boundaries

Main's owner loop calls `idle()`, `request_finished()` and `shutdown()`. These
callbacks support expiry, pending snapshot spills, idle-publication debounce and
bounded durable flushing. The old Lens owner loop does not preserve them. Define
HTTP activity, preparation, queued work, execution and idle time explicitly;
history polling must not accidentally suppress idle publication forever.

Current request profiles must carry template-style defaults and family-owned
raw-JSON decoding, including current K2 tool/byte-budget rules. Reuse of an older
profile is not sufficient just because the Rust signatures compile.

Diagnostic job durability records requests, measurements and outcomes. Main's
durable tier preserves reusable model snapshots. Initially keep native diagnostic
execution cache-isolated: snapshot reuse could skip original-forward capture,
and intervention state must not leak into ordinary prefix caches. Recovering a
job record must not promise resuming interrupted inference.

Memory admission includes resident snapshots, durable workers, native staging,
readout workspaces and publication buffers. Preserve main's pressure-relief policy.
The old live harness's 15-second termination grace is shorter than main's default
30-second durable shutdown allowance; revise the gate before testing shutdown.

## Milestones And Evidence

The first technical checkpoint is ordinary serving retaining main's lifecycle
under the reconstructed coordination. Small shared-semantic refactors may land
first with their existing CLI consumers; they do not establish that checkpoint.

The first product checkpoint is a main-compatible plain-readout workbench:
submit a small job, inspect it, cancel or reload, and reopen saved results without
inference. Identity safeguards and exact retry semantics are part of that slice,
not optional later polish. Fitted interventions and richer inspection follow as
separately usable slices; narrower recovery requires explicit ledger deferral.

The preserved implementation last passed 238 serve tests (19 opt-in ignored),
53 Bun tests and browser/build checks on its old integration. It also holds prior
bounded numerical evidence, including a failed process exit later diagnosed as a
test-witness precision mismatch and independently checked from saved arrays.
Those are reference evidence only. No Lens HTTP executor or web UI is recovered
on this branch yet.

After recovery, interaction design and usable information density remain the
primary product focus. This repair is not authorization for new feature scope.

### R01: Shared Bounded File Readers

`bounded_file.rs` now owns current-main's four regular-file helpers; root imports
retain existing CLI consumer paths. Exact/bounded limits, fallible allocation,
current-offset reads, descriptor-derived lengths and trailing-byte checks remain
unchanged. Opening adds the preserved implementation's `O_NONBLOCK` flag alongside
`O_NOFOLLOW`, preventing blocking opens on certain substituted special files before
the opened-descriptor type check. This is not a filesystem latency deadline,
ancestor confinement, identity with the initial pathname inspection, or immutable
content capture. Same-length edits still require consumer-owned integrity checks.

Transport profile validation, retained handles, streaming scans and content hashes
are unchanged. No new size limits or HTTP module declarations accompany this slice.
Seven helper tests cover empty/boundary files, exact-length refusal, leaf symlinks,
directories/FIFOs, descriptor flags, deterministic shrink/growth, pathname
replacement, current offsets, same-length edits and allocation failure. Existing
nine transport tests and the artifact-size regression pass.

The broad default Lens run passed 351 tests but encountered one existing unmarked
Metal test: `muse_lens_run::tests::operation_only_execution_preserves_enabled_order_and_skips_zero_controls`.
Metal initialization was refused by the wired-memory safety gate; it was not
retried. With that unrelated test explicitly excluded, 351 pass, nine opt-in tests
remain ignored, and one is filtered out. All binaries and formatting check cleanly.
This is CPU reader qualification, not a full-suite or GPU pass. HTTP reader wiring
will land with its consumers, not as unused scaffolding.

### R02: Shared Authored Semantics

`lens_scope.rs` and `lens_intervention.rs` are extracted from the pinned main's
current plan implementation, not copied from the old worktree. Existing CLI plan,
sweep, comparison and Muse consumers retain their public paths through re-exports.
Schema tags/defaults, canonical serialization, plan digest, signed zero, direction
order, normalization math, selector validation and rendered-span binding remain
unchanged. No HTTP profile, backend, queue or renderer is changed by this slice.

Two golden regressions pass before and after extraction. Canonical bytes/digest,
the default covector and signed zero use the actual `parse_plan_bytes` CLI path.
The wire forms and direction order of all five actions use the same `Value`
conversion pattern directly. Lens run tests pass with 56 tests and two opt-in
cases ignored; Muse plan tests pass with seven and one ignored; comparison tests
pass with 31. All binaries check successfully. No GPU execution was performed.

The old tree's numeric selector cardinality helper, stricter coefficient decoder,
shared operation validation/lowering and native integration are deliberately not
included in this refactor. They require their own focused recovery and tests.

### R06: Immutable Family Request Semantics

The resident backends now provide an owned CPU `RequestProfile`; the existing
HTTP trait delegates JSON decoding, parsing, normalization, template defaults,
rendering and output protocol selection to it. Every production family selects
its own profile. The serial acceptor, model execution, memory admission, sessions
and lifecycle callbacks are unchanged. This is not yet CPU-worker coordination.

The profile preserves current main's distinctions: ordinary Qwen binds a cloned
request during rendering while Flash-Next binds before response echoes; DS4 keeps
house/upstream history provenance; Muse retains released defaults and ATEM IDs;
K2 retains its lossless outer decoder, verified chat capability, tool byte bounds
and output grammar, including raw fallback. K2 capability verification and its
cancellation boundary remain in preparation; an `Arc` shares only the verified
CPU metadata. Profiles are checked as `Send + Sync + 'static`.

New socket fixtures exercise the default delegates, not per-hook mock overrides,
for every family through JSON and SSE with byte-fragment output. Fixed prompt
expectations cover Qwen, DS4's pinned fixture, Muse and K2. Tests cover normalized
fields, style overrides changing exact history bytes, raw marker preservation,
K2 large-integer tool arguments, typed result decoding, duplicate-key rejection,
tool-byte overflow and unsupported style refusal. Current serving tests pass:
180 passed, 23 opt-in ignored; all binaries check without warnings. No GPU work.

### R06: Owner Activity And Lifecycle Ordering

The current serial owner now uses transferable, non-cloneable admission guards.
Active preparation/handling and pending completions both prevent idle maintenance;
guard destruction records one completion and only the owner delivers callbacks.
The accounting supports explicit read-only release, but no production endpoint is
exempted yet: ordinary GETs, errors and disconnects retain main's debounce behavior.

Admission begins after the post-receive shutdown checkpoint. A connection refused
at that checkpoint or rejected as busy produces no completion callback. Shutdown
closes admission, stops and joins the acceptor, drains completions, verifies settled
activity, then invokes the existing bounded backend shutdown. Closing admission
alone does not settle outstanding work; concurrent worker settlement is not yet
implemented. Generation, socket deadlines and memory admission remain unchanged.

Idle holds the admission mutex across the callback so new work cannot race idle
publication. Future worker admissions may wait for maintenance; the callback must
not reenter accounting or wait for workers needing that mutex. Completion callbacks
run outside it, and new completions block subsequent idle. Poisoned accounting
fails closed for admission/idle while guards can unwind without a poison panic.

Channel-coordinated CPU tests cover completion during draining, maintenance versus
admission, outstanding work after closure, read-only release and poison/unwinding.
Actual accept-loop traces cover success, ordinary GETs, malformed input, rendering
refusal, disconnect, busy rejection, post-receive shutdown and owner-only callbacks;
shutdown observes a closed listener. Serving tests pass: 190 passed, 23 opt-in
ignored; all binaries check. This qualifies bookkeeping and serial integration,
not concurrent coordination, real snapshot persistence or web recovery. No GPU work.
