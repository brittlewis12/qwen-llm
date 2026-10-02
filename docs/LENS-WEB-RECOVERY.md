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
| R01 | Bounded regular-file access: `bounded_file.rs`, Lens readers | Reuse; audit current reader contracts | CPU symlink/type/length/mutation tests; all reader consumers build |
| R02 | Shared authored scopes and operation semantics: `lens_scope.rs`, `lens_intervention.rs`, `lens_run/{plan,execute,sweep}.rs` | Partially recovered: current-main scopes/validation, direction/action wire forms and normalization extracted with existing CLI consumers; lowering and behavior changes remain pending | Current CLI wire/binding/normalization/lowering regressions; no new service dependency |
| R03 | Typed prefills and annotated input: `lens_input.rs`, `model_request.rs`, `messages.rs`, `open_responses/render.rs`, CLI callers | Rework against current renderers; R02 uses existing span types only | Exact prompt bytes, token positions, reasoning-only continuation, tools and house/upstream rendering |
| R04 | Deployment binding and asset verification: `linear_transport{.rs,/deployment.rs,/cpu_fixture.rs}`, `full_lens/access.rs` | Reuse after R01; retain main's expected-profile checks | Binding mismatch refusal, retained payload hashes, CPU-before-Metal admission, no implicit transfer override |
| R05 | Shared ordinary execution: `ordinary_executor.rs`, `qwen/decode.rs`, `lens_run/execute.rs` | Rework around main's current decode paths; not an automatic replacement | CLI/serve sampling, cancellation, terminal nonconsumption and telemetry remain equivalent |
| R06 | Owner queue and CPU HTTP coordination: `serve/{control,queue,request_profile}.rs`, backend/HTTP wiring | Rework on main lifecycle and family contracts; review before expanding execution | Idle/request-finished/shutdown ownership, busy admission, cancellation/disconnect, JSON/SSE protocols |
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
