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
| R02 | Shared authored scopes and operation semantics: `lens_scope.rs`, `lens_intervention.rs`, `lens_run/{plan,execute,sweep}.rs` | Partially recovered: shared scopes/wire forms/normalization and strict coefficient ingress with existing CLI consumers; shared lowering/action validation remain pending | Current CLI wire/binding/normalization/lowering regressions; no new service dependency |
| R03 | Typed prefills and annotated input: `lens_input.rs`, `model_request.rs`, `messages.rs`, `open_responses/render.rs`, CLI callers | Partially recovered: ordinary Qwen3.6/3.8 singleton CLI and native House prefills with exact retained context; tool/upstream and browser integration pending | Exact prompt bytes, token positions, reasoning-only continuation, tools and house/upstream rendering |
| R04 | Deployment binding and asset verification: `linear_transport{.rs,/deployment.rs,/cpu_fixture.rs}`, `full_lens/access.rs` | Partially recovered: shared CPU deployment binding with existing CLI consumers; native registry integration pending; main's expected-profile checks retained | Binding mismatch refusal, retained payload hashes, CPU-before-Metal admission, no implicit transfer override |
| R05 | Shared ordinary execution: `ordinary_executor.rs`, `qwen/decode.rs`, `lens_run/execute.rs` | Partially recovered: shared serial decode lifecycle and explicit request cancellation with existing CLI/serve/Lens consumers; prefill and forwarding adapters pending | CLI/serve sampling, cancellation, terminal nonconsumption and telemetry remain equivalent except documented added checkpoints |
| R06 | Owner queue and CPU HTTP coordination: `serve/{control,queue,request_profile}.rs`, backend/HTTP wiring | Partially recovered: shared one-execution admission plus two CPU control workers for ordinary Qwen history/native traffic; other-family control allowance and live qualification pending | Idle/request-finished/shutdown ownership, busy admission, cancellation/disconnect, JSON/SSE protocols |
| R07 | Durable job metadata: `serve/jobs/{state,store,preview}.rs` | Recovered with a real baseline producer and routes; distinct from main's durable model snapshots | CPU acceptance/retry, bounded publication, recovery/corruption, history without inference pass; live restart pending |
| R08 | Native request/routes/preconditions: `serve/lens_http/*`, `serve/native/preconditions.rs` | Baseline routes/preconditions recovered; diagnostic admission pending | CPU unknown-field/local HTTP/binding checks and accepted-key recovery pass |
| R09 | Native baseline and observation lifecycle: `serve/native/{mod,execute,writer}.rs` | Baseline recovered on current resident/cache admission, joined writer; observation production and live qualification pending | CPU disconnected completion, cancellation races, independent outcomes and shutdown pass; real-model qualification pending |
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
Those are reference evidence only. The baseline Lens HTTP executor/history is now
recovered with scoped CPU evidence; the browser and diagnostic execution remain
pending.

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

### R02: Preserve Authored Zero-Control Intent

All five action coefficient decoders now reject overflow and authored nonzero
numbers that narrow to f32 zero. Arbitrary-precision JSON spelling preserves the
evidence even when f64 underflows; significand inspection ignores exponent digits.
The same rule applies to sweep CLI input and sweep/cohort manifest coefficient
arrays and arms. Existing plan, CLI and artifact readers are the consumers; no
HTTP scaffolding or kernel changes are included.

Explicit signed/exponent-form zeros and nonzero subnormals remain accepted. Action
JSON keeps its buffered integer/f64 narrowing; manifest scalar fields keep direct
Number-to-f32 conversion, and CLI keeps Rust f32 parsing and finite non-JSON
spellings. Rounding-sensitive fixtures check these existing distinct results
rather than silently unifying them. The explicit Action decoder also recovers
acceptance of finite high-precision spellings that tagged serde previously
rejected; bit equivalence applies to successfully accepted legacy inputs.
Typed setters, plan finite checks,
coordinate-swap reflection-scale validation, operation order and enablement are
unchanged. Previously rounded-zero artifacts cannot recover lexical intent, and
accepted subnormals are not a numerical-effect or GPU preservation guarantee.

Regression tests first reproduced silent underflow through actual plan-file and
Clap entry points. Direct and Value-buffered serde paths cover every action;
manifest readers and the filesystem-backed sweep inspector reject malformed
coefficients in metadata, source plans and child authored/resolved plans. Signed
zero serialization/digest and existing sweep tests remain green. Final CPU Lens
gate passes 382 tests, with nine ignored and the known unmarked Muse Metal test
explicitly filtered. No model or GPU lease is needed for numeric ingress checks.
Serving remains at 212 passing tests with 24 ignored and the known unmarked Metal
context test explicitly filtered. All binaries/tests compile; format and whitespace
checks pass.

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
not concurrent coordination, real snapshot persistence or web recovery.

### R06: Trace Ownership And Request Correlation

One `TraceLog` owns the background writer; each HTTP request receives a separate
`TraceSubscriber` before decoding. Subscribers cannot keep the writer alive after
owner closure. Every trace record gains a `trace_request_id`, distinct from the
response protocol ID, including records for malformed JSON. Existing fields,
event payloads and HTTP wire bytes retain their meaning. This is an optional lossy
diagnostic log, not durable job history or a new persistence lane.

Queue overflow, disconnection and writer I/O failure disable all subscribers.
Payload construction, serialization and shutdown joining occur outside the sender
mutex; closing the owner during construction prevents a late enqueue. The existing
eight-record queue, private append-only file policy and 250 ms detach-on-stall
shutdown remain. Eight records are not a byte budget: concurrent transport still
needs to account for constructing, queued and in-flight trace payloads.

CPU tests force interleaved subscribers and verify correlation and per-request
ordering, shared failure, closure races, surviving subscriber lifetimes and writer
exit. Existing file-safety, bounded shutdown and wire-equivalence tests remain.
Serving qualification explicitly excludes the existing unmarked Metal-context
test `serve::backend::tests::early_capture_window_serves_boundary_tail_then_rebases`:
195 tests pass, 23 remain ignored, with that additional test filtered out. Binary,
format and whitespace checks pass.

Earlier unfiltered serving runs included that test, which silently returns when
Metal context creation fails. Their reported pass counts must not be interpreted
as GPU qualification or proof that every test was CPU-only. Future CPU gates must
exclude it explicitly, alongside the unrelated Muse test noted under R01.

### R06: Single-Admission Owner/HTTP Bridge

The accept loop now dispatches the admitted connection to a CPU HTTP worker and
executes only generation on the resident owner. This replaces the production
handler path rather than adding an optional interface lane. One connection remains
admitted through preparation, generation and response cleanup; concurrent clients
retain main's pre-header busy503 behavior. No multi-request reservation pool,
native routes or dispatcher scaffolding lands in this slice.

This narrows the earlier multiworker proposal deliberately: a many-request bridge
needs bounds for simultaneous bodies, parsed/rendered forms and response writers,
not merely the old token-channel allowance. The current bridge shares prepared
request/prompt storage through `Arc` rather than cloning them. Its two-slot channel
splits pieces at 4096 bytes; the owner waits for acknowledgement after downstream
processing of each original piece before advancing. This preserves the previous
synchronous sink boundary, including potentially large reasoning/tool event work.
Nonstream collection keeps one allocation per original piece. Streaming delta
boundaries may differ without changing text, UTF-8, output protocol or usage.

The added CPU allowance is a conservative 2 MiB worker stack plus 32 KiB bridge
buffer/metadata reserve, not a total memory bound or a pre-spawn gate. Qwen retains
durable plus transport bytes through optional-tail fallback and pressure-relief
retry. Muse/Flash retain resident-runner policy with an added CPU headroom check;
K2's fresh session admission includes the CPU reserve while its existing API
delegates with zero. DS4 checks before and after session construction and restores
residency on returned rejection. Its construction peak and panic recovery are not
qualified by that check. Existing response-collection/trace limits are inherited.

One activity guard survives through both worker and owner work. Cancellation is
separate from its last release. Stop/error paths close channel waits, shut down
the owned socket and join the HTTP worker before completion drain and backend
shutdown. No new worker detaches; the trace writer's existing timeout exception
remains. Generation itself is cooperative, not preemptible.

The existing all-family JSON/SSE profile fixtures now also traverse the real bridge
and compare results against the direct handler. CPU tests cover full-channel
backpressure, shared preparation, terminal ordering, waiting heartbeats, missing
terminals, cancelled queued work, failing/panicking subscribers, downstream
acknowledgement and cancellation, DS4 restoration control flow, and Qwen's three
pricing attempts retaining the same reserve. Actual accept-loop tests cover
busy503 and settlement during blocked reads, nonstream writes and streamed
generation backpressure with client reset. The latter is disconnect coverage,
not an injected SIGTERM test. A real streaming partition test crosses both UTF-8
and reasoning delimiters at bridge chunk boundaries.

Live Metal pressure/performance, cache continuity and signal/durable-shutdown
verification remain pending. No GPU or model loading is required for these CPU
gates, and the known unmarked Metal test is explicitly excluded.

Final CPU gates pass: 211 serving tests, 23 opt-in ignored (the unmarked Metal
context test is explicitly filtered); 351 Lens tests, nine opt-in ignored and the
unmarked Muse Metal test explicitly filtered. The library's CPU-only-versus-Metal
headroom arithmetic regression passes. All binaries, formatting and whitespace
checks pass without warnings. These are scoped CPU results, not a full-suite or
live qualification claim.

### R06: Process-Signal Shutdown

A CPU-only subprocess now runs the actual accept loop with installed process
signal handlers and a mock backend. The parent waits for both generation and a
test-only observation of a full-channel/acknowledgement wait, then sends SIGTERM
only to its owned, unreaped child. The response socket remains connected and
unread through child exit, so disconnect cancellation cannot mask a broken signal
path. Markers and assertions verify generation abort, exactly one completion,
owner-thread callbacks, listener closure before shutdown, and clean worker
settlement. Callback ordering/counts are checked through process exit.

The parent uses an overall five-second deadline and RAII cleanup of its exact
child and stdout reader. That deadline bounds successful protocol completion;
failure cleanup is best-effort and may exceed it. The test runs in about 60 ms
locally, with no model or Metal initialization. Observation hooks compile only in
test builds. The child asserts the termination error and passes through the Rust
test harness; this does not test the CLI's signal-derived exit status, Metal command
interruption, snapshot persistence or durable flushing.

The fixture is an ignored test invoked explicitly by its parent, not a new runtime
mode. Serving CPU qualification now passes 212 tests with 24 ignored entries
(23 existing opt-in tests plus that child fixture); the known unmarked Metal
context test remains explicitly filtered. Binaries, formatting and whitespace
checks pass. Live model-backed qualification remains pending.

### R05: Shared Serial Decode Lifecycle

`ordinary_executor.rs` now owns serial selection, stop handling, publication and
consumption order. Current CLI/serve adapters retain `GenerationResult` fields and
stop reasons; ordinary Lens run/sweep arms retain their token IDs and exact
`stop_token` / `max_new_tokens` strings. The mutable sampler/context remains with
the caller, including sampled-structural transactional telemetry. Qwen retains
upfront token-vector allocation; Lens retains incremental allocation and early-stop
memory behavior. Timer placement/field meaning is preserved, not exact elapsed
milliseconds.

Stop tokens are counted but not published or forwarded. A non-stop sample at the
token limit is published, then retained without a forward. The helper deliberately
adds a pre-transition checkpoint to Qwen and decode checkpoints to Lens: shutdown
after publication can prevent consuming that token; shutdown after a successful
transition does not roll back sequence, sampler or capture state. Callback errors
still win before the new checkpoint. Terminal decisions do not gain a late check.
Production adapters currently supply process-shutdown checkpoints, not an explicit
request-cancellation check at every decode boundary. Transport still checks local
cancellation through its sink.

The shared `ExecutionControl` owns explicit cancellation only. Transport composes
it with the existing process checkpoint, preserving HTTP error mapping, signal
diagnostics, socket shutdown and owner wakeup. Generic tests inject this control
into the decode loop; that proves the helper contract, not new per-request CLI
controls. No lifetime is released merely because cancellation was requested.

Packed prefill, schedule population, checked positions, forward routing, kernels,
readout/capture ownership and memory admission are untouched. The Lens adapter tests
exercise transition indices and stop strings; production phase/position mapping
and `forward_event` arguments are preserved by source comparison, not numerical
execution in those tests. Shared prefill and original-forward adapters still need
their own recovery. No job store or HTTP diagnostic route lands here.

CPU gates pass: 11 shared lifecycle tests; two Lens adapter tests; existing greedy,
seeded, sampled-structural, attributed and GPU-greedy simulation fixtures. The seven
greedy tests, seeded golden and sampled-structural transaction fixture pass both
before and after extraction. Serving passes 212 tests with 24 ignored entries;
Lens passes 364 with nine ignored. Known unmarked Metal tests remain explicitly
filtered. All binaries, formatting and whitespace checks pass without warnings.

The durable-job dependency check confirmed that the old store assumes this
execution-control contract and native admission assumes typed input/prefill
contracts. Recover those with existing consumers before adding a store without a
producer or routes that cannot submit meaningful work.

### R03: Typed CLI Prefill And Retained Input

Ordinary Qwen3.6/3.8 `run --user/--messages` now accepts a strict typed
`--assistant-prefill` with reasoning/final channel and exact text. Current-main's
resolved renderer/mode defines the starting state. A shared transition definition
supplies both closing bytes and structural spans; the complete prompt is tokenized
once and passed through the existing exact aligner. Prefill remains prompt input,
not completed history or generated output. Nonempty content has its own assistant
span, no message index, and nullable token bounds when a boundary crosses tokens.

Family admission precedes runtime dispatch; a compatible-looking protocol cannot
admit Flash, Muse, DS4 or K2. Unsupported protocols, modes and structural markers
fail closed. The complete retained rendered text has a 16 MiB limit. Cohorts,
sweeps, raw/token-ID input, Open Responses and HTTP prefills remain unrecovered;
no unused native renderer or route accompanies this CLI slice.

Prefilled v5 artifacts retain optional `generation_input`, including exact text,
bytes, token IDs, rendering, typed intent and output initial state. Writer and
strict reader share context validation: schema/runtime/source, template/mode,
top-level agreement, text/bytes, token-ID digest, byte coverage/attribution,
nullable/nonoverlapping token bounds and canonical generation suffix. Comparison
and sweep-context comparison include the retained record. Writer/reader retain
the existing 256 MiB artifact limit. This is offline consistency, not text
authentication, retokenization or model qualification.

Missing context is refused when prefill spans or a nonstandard generation closure
remain. Empty reasoning prefills and empty final prefills in no-thinking mode
leave no distinguishable bytes: deleting their records cannot be detected as
such, though comparison rejects retained-intent mismatch. Unprefilled output omits
the new field. New readers accept old artifacts; old strict readers reject
populated extensions. No schema-version bump claims bidirectional compatibility.

The adversarially requested reasoning-history matrix exposed an existing metadata
mismatch: the shared Qwen renderer emits historical reasoning as `message_content`
with assistant/thinking attribution, while the Lens validator refused it. The
validator now accepts that exact source-attributed case; nonassistant or unindexed
reasoning content still fails. This fixes reader acceptance without changing prompt
bytes or the absent-prefill preparation path.

CPU fixtures cover both protocols and all resolved modes, whitespace/empty content,
legacy reasoning history, writer-to-strict-reader roundtrips, malformed/deleted
records, channel transitions, identical-token intent mismatch, nullable boundaries
and exact selector binding. Unsupported-family tests traverse the actual CLI run
preflight with CPU GGUF descriptors. Byte-position token fixtures qualify plumbing,
not a real tokenizer or numerical continuation. Real-tokenizer and bounded live
qualification remain pending. Final gates: 376 Lens tests pass with nine ignored;
212 serving tests pass with 24 ignored. Both known unmarked Metal fixtures are
explicitly filtered. All binaries/tests compile, formatting and whitespace pass.
No model or GPU lease was used.

### R04: Shared CPU Deployment Binding

`linear_transport/deployment.rs` now owns current-main's deployment geometry,
output-head/mode checks, lightweight identity, optional retained-byte hashing,
binding record and loaded-identity check. Existing read/trace/run consumers still
enter through `FullAccess`; legacy published-import behavior is unchanged.
Validation order, binding-record insertion order, hexadecimal identity formatting,
claim labels and transfer opt-in remain intact. The constructor no longer accepts
an already ignored cache argument; the CLI facade retains its existing signature.
No expected-profile checks move or disappear, and payload opening/scanning is
unchanged. This is not native registry recovery or new deployment qualification.

A byte-level binding-record golden passes before and after extraction. Existing
CPU geometry/head/family, MoE scalar/projection-versus-packed, exact-binding refusal,
override and retained-byte-versus-poisoned-cache tests remain green. Lens CPU gates
pass 383 tests with nine ignored and the known unmarked Muse Metal test explicitly
filtered. All binaries/tests compile; formatting and whitespace checks pass.
No model is loaded onto Metal and no GPU lease is used.

### R06: Owner-Supervised Connection Lifetime

The existing owner loop now owns the ordinary `Connection` explicitly and advances
dispatch rather than entering a socket-scoped nested loop. Preparation stays on
the CPU worker and generation stays on the owner. Single-admission/pre-header busy
behavior, bounded piece acknowledgements, profiles and reserves are unchanged.
No scheduler, native route or dormant dispatcher accompanies this refactor.

Common teardown closes admission, stops acceptor delivery, releases queued work,
stops/joins the HTTP worker, joins the acceptor, drains completion callbacks and
invokes backend shutdown. Explicit settlement surfaces worker panic; RAII remains
the unwind backstop. On handling failure, completion callbacks now drain after
acceptor join rather than before teardown, still exactly once on the owner and
before backend shutdown. The test-only connection wrapper uses the new supervision;
it is not an independent old implementation.

Two CPU tests cover stopping a started read connection without dispatch and worker
panic reporting with settled activity. Existing all-family direct-versus-bridged
protocol, lifecycle and real SIGTERM subprocess fixtures pass. Serving gates pass
214 tests with 24 ignored and the known unmarked Metal-context test explicitly
filtered. All binaries/tests compile; format/whitespace checks pass. Durable jobs,
independent native cancellation and concurrent control traffic remain pending.

### R06 Integration Requirements

Concurrent diagnostic/history handling and native owner work must keep queue,
activity lifetime, cancellation, memory admission and worker settlement coherent.
Reuse the owner bridge; do not add another service or restore the old
readiness/dispatcher hop. The recovered baseline below is the first consumer.

The old piece-buffer allowance is not a total transport budget. Account for request
bodies, parsed/rendered forms, queued work, generation output, response assembly,
trace payloads and overlapping response writers. Qwen's initial admission and
snapshot-eviction retry must both retain durable reservations plus transport
reservations. Preserve the other families' distinct residency/session policies.

Reserve before expensive preparation; order closure atomically with delivery;
settle all acquired reservations. An ordinary HTTP request has one completion
lifetime shared with enqueued/executing work, but worker exit signals cancellation
independently. Preserve request-side half-close semantics and consume all pieces
before terminal success. Define overload behavior before SSE headers, and do not
copy the old arbitrary piece-size refusal without validating or splitting pieces.

Stop acceptance, cancel work, wake both socket and channel waits, and join workers
outside accounting locks before final completion drain and backend shutdown.
Include socket-registration/closure races and read watchdogs in that settlement.
CPU tests can qualify coordination and policy decisions, not real Metal pressure,
cache continuity or durable snapshot persistence. Bounded live checks remain
pending; never override memory safety to obtain them.

### R06-R09: Durable Baseline On The Resident Owner

`qwen serve --lens-data-dir` now has a real cache-isolated baseline producer,
durable acceptance/history and same-port native routes. The flag initially requires
ordinary Qwen; execution requires metadata-qualified House Qwen3.6/3.8. Unsupported
ordinary deployments expose history without claiming execution. Nonempty diagnostic
plans are refused. No fitted registry, readout head or browser is implied.

One execution gate covers ordinary preparation/response cleanup or a native job
through writer join. Two bounded CPU classifiers handle history, exact-key retry
and explicit cancellation while the owner runs. Ordinary traffic is refused before
body upload when execution is busy; history-disabled pre-header behavior remains.
Delivery and closure share one gate lock, and failed events are dropped outside it.
Local control closure reaches owner dispatch and ordinary/native checkpoints.
Mutating HTTP and independent native work have separate completion lifetimes;
valid read-only history releases activity before storage/socket I/O. Owner-only
maintenance callbacks and current durable snapshot shutdown remain in place.

Preparation shares CLI rendering, exact tokenization/alignment and typed prefill
semantics. The borrowed record is size-bounded before building its JSON Value tree.
Native decode shares the serial lifecycle, distinguishes request cancellation from
server interruption, and retains sampled versus consumed tokens without replay.
Piece bounds and borrowed serialization prevent sampled-byte Value amplification.
The joined CPU writer owns all execution-time job disk publication, with independent
terminal delivery and recording failure. Store start settles early cancellation
under its transition lock, including cancellation signalled before its publication.
Writer setup failure cannot become model success. Dropping undispatched work closes
publication as interrupted, rather than pretending it was an explicit user cancel.

Admission retains a conservative 512 MiB control allowance plus durable pending
bytes; native admission also reserves 82 MiB for writer/publication work. The same
allowance survives pressure-relief retry, snapshot capture/spill and both optional
DFlash checks. Capture allocation now follows admission rather than consuming
reserved headroom first; refusal keeps the serial fallback. These are conservative
allowances, not allocator-enforced bounds. Full fresh-headroom checks per connection
can refuse history/cancel even after some allowance is already in use. Filesystem
operations and joined writers are not deadline-bounded. See `docs/LENS-WEB.md` for
the complete current lifecycle and availability limitations.

Store recovery retains compatible saved array reads but does not advertise new
capture production. Schema fixtures preserve historical diagnostic extensions as
illustrations, explicitly separated from current capabilities. Model identity is
metadata-based, not content authentication. No arbitrary HTTP model/path loading,
alternate service, resumed inference or client-owned execution lifetime is added.

CPU gates: 285 serving tests pass with 25 opt-in/child entries ignored; the known
unmarked Metal-context test is explicitly excluded. Lens passes 383 with nine
ignored and its unmarked Muse Metal test explicitly excluded. Synthetic tokenizer
and mock-forward tests traverse real sockets/owner/store/writer for disconnect,
retry, read-only access without new forwards, cancellation, subsequent ordinary
serving, local stop during active native work, and publication failures. Coordinated
tests cover the pre-publication cancellation race and idle release while history
waits on store access. Existing real SIGTERM ordinary CPU fixture remains covered.
These are not released-model numerical, GPU pressure or KV snapshot qualifications.
No GPU lease was used. Origin/main was fetched again and remains the pinned base.
All binaries/tests compile without warnings; formatting and whitespace pass.
Iterative `cx` review found and then cleared the startup-cancellation race and
pre-admission DFlash allocation; final review approves this CPU-qualified slice.

Next: recover the preserved Bun baseline submission/history client with binding
and draft safety, before adding plain original-forward readouts. Then restore
fitted assets/interventions and retained/pair exploration in their consuming slices.

### R06: Disconnected Startup Probe

The first clean-release live baseline attempt exposed a macOS socket lifecycle
regression: configuring a reset connection already queued during model startup
returned `EINVAL`, and the control acceptor propagated it as service failure.
A second short launch with a backtrace located the exact configuration call. Both
processes exited and released their normal Metal lease; no memory gates were
overridden and no job execution is qualified by those failed launches.

Control now logs and discards only that connection, before acquiring activity or
execution ownership, matching the existing ordinary acceptor. Service-level errors
still close the shared gate and settle workers. A real reset-before-accept CPU test
fails against the original propagation and passes after the fix; it checks a later
successful request, settled activity and released listener. Serving now passes
286 CPU tests with 25 ignored and the known Metal test excluded. Compile/format
gates pass. `cx` approves this focused correction and identified live-harness
witness/timeout improvements before another model launch.

Separately, the opt-in CPU tokenizer matrix passes against the released Qwen3.6
35B A3B Q4_K_M GGUF: native and CLI prefills have identical token IDs, spans and
retained input across the tested mode/channel combinations. This qualifies that
tokenizer/template path, not numerical continuation or other deployments.
