# Lens Workbench Reconstruction

## Status And Preservation

The Lens workbench was developed as uncommitted changes in `feat/lens-web`.
That branch contains no feature commits and is not an integration deliverable.
This document records an honest reconstruction, not recovered development history.

Recovery is now complete as a committed reconstruction on `reconstruct/lens-web`,
with the explicit policy replacements and qualification limits recorded below.
It is not merged or pushed. The original worktree and preservation package remain
untouched; this branch, not `feat/lens-web`, is the integration deliverable.

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
The table is the current disposition. Dated/checkpoint prose later in this document
records what was pending at that checkpoint, not outstanding work today.

| ID | Behavior and source | Disposition / dependencies | Required gate |
| --- | --- | --- | --- |
| R01 | Bounded regular-file access: `bounded_file.rs`, Lens readers | Recovered with CLI, registry, history and static consumers | CPU symlink/type/length/mutation and reader-consumer gates pass |
| R02 | Shared authored scopes and operation semantics: `lens_scope.rs`, `lens_intervention.rs`, `lens_run/{plan,execute,sweep}.rs` | Recovered: scopes, normalization, strict coefficient ingress, shared validation/lowering | CLI wire/binding/normalization/lowering and native tests pass |
| R03 | Typed prefills and annotated input: `lens_input.rs`, `model_request.rs`, `messages.rs`, renderer/CLI callers | Recovered across ordinary singleton CLI, native House inputs and browser; strict artifact context comparison retained | Exact bytes/spans, CPU real-tokenizer native/CLI matrix, live native prefill and browser pass; no tool/cohort/raw prefill expansion |
| R04 | Deployment binding and asset verification: `linear_transport{.rs,/deployment.rs,/cpu_fixture.rs}`, `full_lens/access.rs` | Recovered with retained-source registry/staging; current-main expected-profile checks retained | CPU binding/integrity/cancellation and scoped fitted numerical gates pass |
| R05 | Shared ordinary execution: `ordinary_executor.rs`, `qwen/decode.rs`, `lens_run/execute.rs` | Recovered decode, bounded prefill and post-block adapters; obsolete generic production dispatch replaced by current routing | CLI/serve/native CPU gates and final production baseline pass; added checkpoints documented |
| R06 | Owner queue and CPU HTTP coordination: `serve/{control,queue,request_profile}.rs`, backend/HTTP wiring | Reworked for current lifecycle across all families; explicit single-execution/two-control-worker policy replaces old eight-reservation/sixteen-worker queue | All-family CPU protocols/lifecycle/reserves and scoped ordinary live gates pass; not cross-family numerical/pressure qualification |
| R07 | Durable job metadata: `serve/jobs/{state,store,preview}.rs` | Recovered with a real baseline producer and routes; distinct from main's durable model snapshots | CPU store gates and scoped live Qwen3.6 interruption/restart pass; not KV persistence qualification |
| R08 | Native request/routes/preconditions: `serve/lens_http/*`, `serve/native/preconditions.rs` | Recovered baseline and diagnostic admission, exact retry before current capability checks | CPU unknown-field/local HTTP/binding, unavailable-family retry and live accepted-key recovery pass |
| R09 | Native baseline and observation lifecycle: `serve/native/{mod,execute,writer}.rs` | Recovered on current owner/admission with joined staging/array writer and no inference replay | CPU lifecycle/publication failures and scoped original-forward/live restart gates pass |
| R10 | Plain original-forward readouts: `serve/native/{readouts,observe}.rs` | Producer recovered with browser/current admission; no replay | CPU scopes/lifecycle/publication/HTTP/browser and scoped Qwen3.6 live unchanged-sample/final-layer witness gate pass |
| R11 | Fitted readouts and direction staging: `serve/native/{registry,interventions}.rs`, `workspace_lens/*` | Recovered fitted producer and shared direction staging; R12 qualifies operations | Registered identity, matrix integrity, owner-admitted joined staging/workspace and independent numerical oracle pass |
| R12 | Ordered scoped interventions: `serve/native/interventions/*`, `lens_intervention.rs` | Three native operators recovered with CPU/browser and scoped first-site numerical evidence | Exact order/scopes, zero controls, deployed covector semantics, independent transformation checks |
| R13 | Retained full scores/source arrays: `serve/jobs/arrays.rs`, native observer/writer, `web/retention*` | Producer recovered with CPU/browser and production live integrity/restart evidence | Raw admission, dual-watermark durability, digest/finite/shape checks, offline rank/entropy, no implicit fetch |
| R14 | Whole-site pre/post capture: `metal_forward/token.rs`, `serve/native/measurements.rs` | Producer recovered with CPU/browser and bounded actual-before Metal numerical evidence | Before/after placement, independent layer sets, alias refusal, actual-vector metrics and zero controls |
| R15 | Bun client and same-origin assets: `web/{build,dev,proxy}.*`, `serve/assets.rs` | Recovered with independent optional history/assets across serving families | Bun types/build and real same-port CPU browser/static gates pass; no second production service |
| R16 | Durable browser submission/history: `web/{api,contract,durable,storage,jobs,history}.*` | Recovered with real baseline producer and historical fixture readers | Persist-before-POST, exact retry, history/reload/copy pass; baseline CPU browser traverses actual Rust store |
| R17 | Draft and identity safety: `web/{draft,bindings,editor,diagnostics}.*` | Recovered across baseline/diagnostics; creation remains capability-gated | No silent retarget, saved pinning, copy/edit races and persistence failure pass in fixtures; actual diagnostic HTTP/browser paths exercised |
| R18 | Direct manipulation workbench: `web/{viewer,token-navigation,token-navigator,App,styles}.*` | Recovered historical and newly produced diagnostic navigation, pinning/reordering, retained queries and pair inspection | Phone/desktop historical fixtures and actual Rust CPU producer browser gates pass; numerical gates separately scoped |
| R19 | Evidence and qualification: native CPU/live tests, HTTP fixtures, `web/*check.ts`, docs | Replaced obsolete fixture-specific process harnesses with owned bounded current-main gates; old evidence preserved | Exact-build evidence and limitations recorded; no old-base pass promoted to current qualification |

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
Those are reference evidence only. Current recovery has its own CPU/browser,
production lifecycle/retention and test-instrumented numerical gates recorded below.

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

### R09: Bounded Live Baseline Lifecycle

The clean release at `b850e3e7` now passes the reviewed owned-server Bun harness on
Qwen3.6 35B A3B UD-Q4_K_M, using the normal lease/wired-memory/process admission
gates. The first passing final-prefill fixture stopped on its first sample: valid
terminal evidence, but not a decode-transition witness. The strengthened committed
harness uses a reasoning prefill and requires positive generated-token consumption.
It passes with 16 prompt tokens, samples `[1683, 883, 411]`, two generated tokens
consumed and the final sample retained unconsumed. A disconnected submission recovers
the same accepted key and matches those sample records; history reads remain stable
and ordinary serving succeeds afterward.

A third job is observed active, then the owned server receives SIGTERM. It publishes
an interrupted status and terminal record before exit; the final counters are
16 prompt / three sampled / three generated consumed. Restart preserves that exact
status and the earlier completed records, with three jobs total and no automatic
retry. Both launches exit with code 143 and no OS-termination signal; no forced
cleanup was needed. Each model load took about 2.35 seconds. The model lease was
confirmed unheld afterward. RAM snapshot budget was zero and the durable KV tier
was off, so this is job durability, not a KV persistence or pressure-stress gate.

Local evidence: `target/lens-baseline-12cc5c4d-0812-4d90-90c1-95993e7cc377/`.
The tested `qwen` binary has Git blob fingerprint
`180acf03f915744a3691912d2937a7064d0b09a7`; the harness fingerprint is
`9b2a753e7fab09f27f77c07157aeb0f79a6369bf`. Generated artifacts/logs stay ignored.
The final-prefill first-sample-stop run remains in the separate local evidence
directory `target/lens-baseline-aa75266c-116a-46ce-9e33-d5f37bedf64d/`.
These checks establish scoped live execution/repeatability and settlement, not
independent CLI-versus-HTTP numerical equivalence or other-model qualification.

### R15-R18: Baseline Browser And Historical Exploration

The preserved React/TypeScript client is recovered with Bun-native HTML serving,
HMR and production bundling, retaining its established design rather than replacing
it with another interface. `--web-root` mounts immutable prebuilt assets on the
existing CPU control pool; it requires `--lens-data-dir` and the current ordinary
Qwen family boundary. One `Workbench { store, assets }` configuration replaces the
old optional history argument; no duplicate owner loop or service is introduced.
Assets load before model admission and serve borrowed bytes. Their 64 MiB catalog
limit counts payload lengths, not total allocation overhead; path checks do not
promise race-free ancestor confinement. `/v1` and its descendants cannot be assets.

Static GET/HEAD classification precedes execution reservation, refuses bodies and
releases activity before writing while retaining socket ownership through worker
join. CPU tests cover serving during native work, API precedence/missing assets,
early body refusal, idle release during blocked writes, socket shutdown/join and
CLI configuration requirements. Ordinary routes and history retain the same gate.

The client preserves exact-key persistence/recovery, independent job history,
identity-bound drafts, asynchronous-edit protection and explicit cancellation.
Discovery now sequences capabilities/assets; GET-only 503 handling has two bounded
retries to coexist with polling in the two-worker pool. POSTs are never retried as
reads. Explicit `execution.baseline_only` is decoded and contradictory diagnostic
claims refused, with fallback for historical responses lacking that field.

Historical token/layer/candidate navigation, retained-array readers and contextual
draft pinning return with the same client. Compatible stored diagnostics are useful
without new capture production. Unsupported copied plans stay intact and blocked;
no operation is silently removed. This does not qualify the server's unrecovered
plain/fitted/intervention/pair producers or restore the old numerical evidence.

All 55 preserved/updated Bun tests, typecheck and production build pass. The
GPU-disabled historical browser fixture passes mobile interaction, exact retry,
binding changes, late pages, pinning, retention, older history and desktop bounds.
A new CPU-owned Rust test child serves the actual production bundle and uses real
native preparation/store/writer with synthetic forwards. Browser submit/reload,
exact sampled-byte output, copied prefill/sampling, unchanged history and desktop
reopening pass under the real two-worker pool. No GPU lease is used for this slice;
layout/interaction assertions are not screenshot visual qualification.

Final gates pass: 292 serving CPU tests, 26 ignored (including the new browser
child fixture), with the known unmarked Metal test explicitly excluded; 13 CLI
tests; all binaries/tests compile; format and whitespace checks. Both browser
flows pass again after `cx` requested deterministic socket-buffer backpressure and
immediate, idempotent interruption cleanup/escalation for owned browser children.
Final adversarial review approves this slice. The actual-server browser evidence
is local at `web/.browser-test/baseline-4e5ab048-21b9-4f47-8954-368aa72a5863/`.

The next consuming recovery slice is plain original-forward readouts, followed by
fitted bindings/interventions and retained/pair capture production. The original
richer live/oracle scripts remain in the frozen source until their server consumers
return; the baseline live harness is independently committed and qualified.

### R05/R10: Shared Original-Forward Dispatch

The existing CLI `forward_event` and native baseline now immediately consume one
small `ordinary_executor::post_block_forward` adapter. Its four branches call the
same MetalForward methods with the same arguments for capture/no-capture and
logits/no-tail. Callers retain checkpoints, sequence validation, exactly one advance
after successful forward and subsequent consumption/observation accounting. CLI
Production routes remain direct `single_token`, including discard-logits behavior;
no old `LoadedModel::decode_token` substitution is introduced.

This is a source-equivalent extraction before recovering plain observation, not
another executor or a new capture capability. The existing route-topology matrix
remains green. Scoped CPU gates pass: Lens 383 with nine ignored and explicit Muse
Metal exclusion; serving 292 with 26 ignored and explicit Metal-context exclusion.
All binaries/tests compile and formatting passes. `cx` verifies method/argument,
checkpoint/advance and production-route fidelity and approves the small refactor.
No GPU is used; earlier live evidence remains tied to its tested source build.

### R10: Plain Original-Forward Producer

Qualified passive output heads expose `plain`/`full_vocabulary` on the same resident
server. Failed head qualification leaves baseline generation available. Discovery,
preconditions and prepared asset metadata share the same model identity; no direction
capability is advertised. Retention, fitted aliases, interventions and residual
pairs remain refused until their consuming recovery slices.

Numeric scope cardinality is checked before expansion, preserving inclusive ranges
and excluding unreachable final samples. Per-site capture occurs in the original
post-block forward through the committed shared adapter. Consumption and consumed
sample publication precede observation. Separate heads cannot alter the original
sampler logits. One head/ranking serves overlapping readout IDs, with requested
top-k counts, shared-cost attribution and conditional same-forward final-layer
witnesses. No extra prompt tails or transformer replay are added.

The preserved combined-memory estimate was replaced with separate aligned GPU
buffers and CPU peak accounting. Adversarial review caught the previous generation
logits remaining live during transition; all three previous/new/observer host arrays
are now explicit. Current native admission retains those allowances and durable,
control and writer reservations on eviction retry. A tested allocation callback
cannot run on denial. Existing passive-head inner admission remains intact.
The library's `into_topk` only changes visibility; current O(k) ranking and bit/tie
semantics are reused. Score records use bounded borrowed serialization; aggregate
raw label bytes are checked before lossy string allocation, without truncation.

CPU tests cover huge/unreachable selectors, cardinality/expansion, shared heads,
failed forward versus failed observation, known consumption, writer/control stop
between overlapping IDs, joined settlement, exact/overflow label bounds and invalid
UTF-8/escaped labels. Actual HTTP submission traverses reservation/acceptance/owner/
publication and reopens identical saved results without extra calls. A GPU-disabled
browser submits decode-scoped plain readouts through the actual Rust control/store
path with synthetic heads and verifies the displayed IDs/scores on phone and desktop.
Local browser evidence is `web/.browser-test/baseline-e7961fc0-5ce0-41d6-8073-402b74c52593/`.

Current gates: 303 serving CPU tests pass, 26 ignored plus explicit unmarked Metal
exclusion; Lens 384 pass, nine ignored plus explicit Muse exclusion; two scalar-head
CPU ranking tests; Bun 55 tests/typecheck/build; binaries/tests/fmt/whitespace clean.
No GPU was used for this slice. The next gate is a minimal same-seed baseline/plain
comparison across middle/final layers and original prefill/decode sites, requiring
all expected witnesses to exist and pass. R11-R14 remain explicitly pending.
Final `cx` review approves the CPU-qualified producer after the accounting and
publication-boundary fixes; it does not promote pending live numerical evidence.

### R10: Live Plain Readout Consistency

The clean release built at `3c74f3c9` passes the reviewed bounded readout extension
on Qwen3.6 35B A3B UD-Q4_K_M. Exactly one observed job is added to the existing
two-launch lifecycle gate: three sampled tokens, middle/final layers, first/last
prompt positions and consumed decode indices zero/one, with two overlapping IDs.
All 12 unique requested rows exist, with eight shared heads and correct phase,
position/prediction and top-k coordinates. The six required witness records cover
three distinct final-layer head comparisons; all report `max_abs_error=0`.
No-tail first-prompt and middle-layer sites correctly have null witnesses.

Sampling and consumed flags match the unobserved baseline after removing only
record sequence numbers. Shared rows preserve score prefixes and report head cost
only once. The observed job took about 207 ms (not a performance claim). Ordinary
serving, disconnected recovery, active interruption, pre-restart publication and
exact completed/observed-page preservation on restart all pass. Both loads take
about 2.36 seconds; both owned servers exit through handled SIGTERM, with no forced
cleanup. The normal GPU lease is confirmed released. KV snapshots remain disabled;
no safety gates are overridden.

Local evidence: `target/lens-baseline-473dcc46-16de-418a-8ac0-cd1e766517a2/`.
Tested binary Git blob fingerprint: `9947165448a4567bffb5f207fb2ef45abd711e07`;
harness fingerprint: `4eb6f567bcc52908a017bf29f3d1c675a578c7a5`.
This is scoped same-forward consistency and noninterference, not an independent
numerical oracle, a memory-pressure test or general model-family qualification.

### R04/R11: Cancellable Verified Transport Preparation

Shared data-only transport open/profile/read APIs now accept explicit cooperative
checkpoints, with existing unchecked entry points retaining no-op behavior. Current
ordinary, Muse and K2 CLI consumers supply shutdown checks before allocation, bounded
payload reads and verified-result publication. K2 expected geometry/target checks
still precede payload access. Retained descriptors, finite/identity validation,
matrix/whole hashes and successful binding records are unchanged.

Exact CPU binding uses the library's cancellable retained-byte verifier through a
shared adapter that preserves the original checkpoint diagnostic. No partial matrix,
content identity or deployment descriptor is returned after cancellation. Published
legacy formats and payload readers are unchanged; their shared deployment preflight
also gains entry/exit shutdown checks. Checkpoints cannot interrupt an already-blocked
filesystem syscall and do not establish a shutdown deadline.

Tests cancel at every observed open/read/binding checkpoint, including immediately
before publication; interrupt a multi-chunk scan and successfully re-read the same
descriptor; and compare successful binding JSON bytes. Existing profile-before-payload,
path replacement and binding regressions pass. Lens CPU gate: 388 pass, nine ignored,
with the known unmarked Muse Metal test explicitly excluded. All binaries/tests,
formatting and whitespace checks pass. No GPU used. `cx` gives GO for this independently
consumed prerequisite, not for pending registry staging or fitted execution. Fetched
main remains `8bc9e6b739e953023a1f2ffc5dc3379c22758e00`.

### R04/R11: Retain The Ordinary Serve Deployment Through Loading

Ordinary Qwen serving now moves the GGUF used for family/template/context preflight
into the existing opened-model Runtime loader instead of dropping it and reopening
the model pathname. This prevents a path replacement between preflight and loading
from silently switching the deployment, and gives fitted-asset CPU binding the same
source that will subsequently load. Retained descriptors do not make file contents
immutable. Reusable load intent, default configuration, memory admission, prefetch,
snapshot identity and diagnostic path are unchanged. The existing `load_ms` interval
no longer includes a second GGUF open/parse; this is not a performance improvement
claim.

Serving CPU gate: 303 pass, 26 ignored, with the known unmarked Metal test explicitly
excluded. Runtime CPU tests: 39 pass, 33 ignored. Binary/test compilation, formatting
and whitespace checks pass. `cx` approves the focused loader change. Clean-release
live baseline/plain-readout requalification is pending; registry and fitted execution
remain pending, not implied by this retained-source prerequisite.

The clean release at `892807d6` subsequently passes the unchanged bounded live
baseline/plain gate on Qwen3.6 35B A3B UD-Q4_K_M: all 12 original-forward rows, eight
shared heads and six required witness records pass; observed sampling remains
identical. Disconnected exact-key recovery, ordinary serving, active interruption,
pre-restart terminal publication and exact saved results after restart also pass.
Both owned servers exit through handled SIGTERM, without forced cleanup. The normal
GPU lease is confirmed released. RAM snapshots and durable KV remain off, with no
safety overrides. This requalifies startup/lifecycle consistency, not path-replacement
fault injection or fitted execution.

Local evidence: `target/lens-baseline-baa78f07-db9d-4030-8550-9def3c2af09f/`.
Tested binary Git blob fingerprint: `1ca3a930ec9e1dbed92dc2ac99a15b7b1c287420`;
unchanged harness fingerprint: `4eb6f567bcc52908a017bf29f3d1c675a578c7a5`.

### R11: Registered Fitted Readouts And Owner-Admitted Staging

`--lens-config` now consumes explicit data-only assets on the same ordinary House
Qwen3.6/3.8 server. Unsupported protocols fail before asset scanning/Metal. Config
keys and aliases are strict; geometry is checked before payload access. Exact bindings
verify retained model bytes and cannot be overridden. Unbound assets require explicit
transfer acknowledgement. The existing same-opened-GGUF loader and loaded-identity
check remain in use. Manifest identity, binding status and producer-only qualification
are advertised; direction rows and operators are not yet offered.

One original-forward capture serves all aliases at a site, with separate shared heads
by position/layer/alias. Fitted rows retain artifact identity, target and method, and
never carry a generation-distribution witness. The existing fitted readout workspace
is used, with all nine retained buffers individually aligned in admission alongside
any overlapping plain-head allocations and the three simultaneously live host logits.
Selected alias/layer matrices are bounded to 512 MiB, independent of other allowances.

The old ready dispatcher was not restored. Current owner admission includes sequence,
workspace, capture, matrices/scratch and dynamic pending durable/control/writer reserves
before authorizing one staging command on the existing joined artifact worker. The
owner performs no payload file reads. The worker rehashes retained matrices and transfers
them without cloning. Fresh process checks retain the entire originally authorized
future allowance, conservatively over-refusing if staged bytes are counted again.
Cancellation, shutdown, private owner abandonment and ordinary staging failure remain
distinct. Failure prevents execution; terminal publication uses the existing writer.
No extra queue, worker, model service or deadline-bounded filesystem guarantee is added.

Coordinated CPU tests cover once-only transfer, pressure loss, corrupt matrix, explicit
cancel/shutdown during staging, cancellation after handoff, publication failure with a
waiting command, worker panic/disconnection and restart settlement, and owner unwind
without synthetic cancellation. HTTP tests pin identities, reject stale bindings before
acceptance and reopen exact shared fitted/plain rows without execution. CPU browser
evidence `web/.browser-test/baseline-e94c2aa8-1a56-4699-aafb-8a12f6368052/` passes fitted
selection/provenance, prefills/sampling, submission/reload/copy/history and phone/desktop
parity through the actual Rust control/store/writer with synthetic heads.

CPU gates: 317 serving tests pass, 27 ignored and the known unmarked Metal test explicitly
excluded; Lens 388 pass, nine ignored plus explicit Muse exclusion; CLI 14 pass; two
workspace-plan tests pass; Bun 55 tests, typecheck/build and binaries/tests/fmt pass.
The independent original-forward oracle is opt-in and live-pending at this checkpoint.
It uses a release test executable with test-only F64 transport verification, synthetic
nonidentity middle-layer and identity final-layer matrices, baseline/observed sampling,
and required plain-generation witnesses. It is not fitted-quality evidence or an
uninstrumented production-binary qualification. Its owned-process harness rechecks
absolute deadlines/sticky interruption after evidence reads and cleanup, and owns build
interruption too. `cx` review requested and received these false-pass corrections before
commit. R12 interventions, R13 retained arrays and R14 pre/post pairs remain pending.

### R11: Original-Forward Fitted Numerical Qualification

The clean release test executable built at `e93691b0` passes the bounded opt-in
oracle on Qwen3.6 35B A3B UD-Q4_K_M. Two three-sample jobs use the actual backend
admission and joined writer staging. The observed job captures prompt position zero
and consumed decode index zero at layers 19 and 39. Four required F64 transport
witnesses pass with zero maximum absolute error, including the nonidentity rotated/
scaled-diagonal matrix at layer 19. Two identity final-layer controls agree with plain
head rankings/scores. The required final-decode generation witness also reports zero
error; the no-tail prompt witness remains null. Samples and consumed flags match the
unobserved baseline, with three samples and two consumed generated tokens in each.

GPU residency is about 6.9 seconds; the child exits normally after about 7.1 seconds.
The normal lease is confirmed released. No safety controls are overridden. Evidence:
`target/lens-fitted-8ddc3682-d156-48ca-ae9e-a579661a2d58/`. This is independent transport
arithmetic over original residuals plus same-tail/generation consistency, not an
independent quantized output-head oracle, learned-fit quality claim, memory-pressure
qualification or uninstrumented production-binary test. The test-only witness field
and F64 verification are absent from production builds.

### R02/R12: Shared Action Validation And Lowering

Existing ordinary and Muse CLI consumers now use shared action validation/lowering
in `lens_intervention`, instead of parallel five-case lowering matches. Scope checks
still precede action checks; finite and coordinate-swap validation, all reference
checks before normalization, family-specific missing-row diagnostics and lazy
reflection lookup retain their order. Native-hyper normalization may remain `None`
until binding as before. Coefficient decoding, normalization arithmetic and family
row preparation are unchanged. Zero coefficients still validate references.

Three new CPU regressions cover error precedence, scale checks, deferred normalization
and lazy lookup behavior. Lens CPU gate: 391 pass, nine ignored plus explicit unmarked
Muse Metal exclusion; all binaries/tests/fmt/whitespace pass. `cx` approves this
independently consumed refactor. No GPU used; native intervention production remains
the next consuming R12 slice, not implied by shared lowering alone.

### R12: Ordered Scoped Native Interventions

The existing server now admits fixed addition, residual-L2-relative addition and
projection attenuation from registered token-ID deployed-logit-numerator directions.
CPU selected-output qualification, not plain-head availability alone, controls discovery.
Source-to-target, coordinate-swap and alternate covector conventions remain native
capability refusals; shared CLI behavior is preserved. Scope cardinality is checked
before expansion. Zero controls validate fully but create no applications or prepared
rows. Order is transformer layer order, then authored operation order at each layer.

A combined staging plan deduplicates matrices across readouts/interventions and admits
host staging once. Operation-only jobs traverse the same admission/staging path without
allocating a fitted-readout workspace. Retained GPU directions and separately aligned
selection/projection buffers are priced alongside readout buffers, conservatively summing
partly separate lifetimes. Host copies, normalization/upload/record construction and
existing durable/control/writer allowances are retained. Projection uses unified
checkpoints before/between GPU calls and publications. Preparation cancellation remains
typed cancellation, not an execution failure. Applied records follow successful forward
and known consumption; readouts report the site's authored applied IDs.

CPU tests cover ordered events, shared projection with distinct normalization rows,
zero validation without allocations, operation-only preparation/publication, shared
matrix admission and failed-forward refusal. Serving CPU gate: 322 pass, 28 ignored
plus explicit known Metal exclusion; Lens 391 pass, nine ignored plus explicit Muse
exclusion; all binaries/tests/types compile. Bun 55 tests/build pass. Actual Rust CPU
browser evidence `web/.browser-test/baseline-44fdf064-292e-487f-835d-c59224f685e7/` covers
phone direction pinning, operation creation/reordering, decode scopes, submission,
same-site application/readout provenance and saved-copy/history with desktop parity.
Its forwards/projections are synthetic; it is not numerical qualification.

The opt-in release-test oracle adds five short jobs: baseline, zero, active, operation-only
and reordered. Test-only source values borrow existing original captures. At the first
intervened site (prefill zero, one middle layer, no earlier intervention), baseline is
a valid pre-operation reference. Sequential F64 transforms must match the active and
reordered observations; CPU references must have disjoint tolerance bands and each
observed order must reject the opposite reference. A CPU regression rejects overlapping
bands. Six projection witness records represent three projections with two normalization
variants each. Operation-only equivalence checks sampled records, not uncaptured residuals.
Live evidence is pending at this checkpoint; test-only vectors/witnesses are absent from
production. R13 retained arrays and R14 whole-site paired capture remain pending.

### R12: First-Site Numerical Qualification

The clean release test executable at `0ca27cd2` passes the reviewed five-job oracle
on Qwen3.6 35B A3B UD-Q4_K_M. Baseline and zero controls have identical samples,
source vectors and scores. Active and operation-only sampled records match. All six
projection/normalization witness records pass (three projections, two normalization
variants each). Two authored orders match independent sequential F64 transformations
of the valid first-site baseline input, with maximum absolute errors approximately
`2.41e-7` and `4.29e-7`. Their CPU reference tolerance bands are separated, and both
observed vectors reject the opposite order. Exact application counts/order/actions
and readout provenance pass.

GPU residency is about 7.8 seconds; the owned child exits normally after about 8.0
seconds. The lease is confirmed free, without safety overrides. Evidence:
`target/lens-fitted-d209965a-1f1d-4e9c-ab6a-a2e0683e0964/`. These are instrumented
release-test results at one qualified first site, not arbitrary later-site paired
evidence, fit-quality claims or general model-family qualification.

### R13: Retained Original Source And Full-Score Production

Readouts may now request `scores_and_residual` retention. Original post-block sources
deduplicate by position/layer; full logits deduplicate by position/layer/alias. Only
requesting readout IDs receive retained references, even when a head is shared. Exact
raw upper admission is four bytes per scalar, bounded to 4 MiB per array and 32 MiB
per job before durable acceptance. Prepared input saves widths, array counts and byte
upper bounds. Early stops may publish fewer arrays without inventing unconsumed sites.

Serialization borrows the existing captured source and full logits before top-k;
no float Value trees, new GPU buffers or replay are introduced. A typed Array event
uses the existing FIFO and metadata budget. A separate payload-capacity budget covers
up to 16 MiB queued/in active storage writes. RAII drops payload storage before its
permit. Optional memory admission adds the smaller of raw upper/queue budget plus a
4 MiB producer allowance; nonretaining jobs retain previous pricing. These are
conservative future allowances, not total allocator bounds.

The existing store owns array descriptors, SHA256 and binary/JSON committed watermarks.
Source, logits and dependent rows are FIFO-ordered separate transactions. A committed
source-only prefix is valid if later publication fails. Enqueue failure stops production,
latched disk failure discards dependent events, and generation/publication outcomes
remain independent. The original arrays reader/recovery path is reused unchanged.

CPU gates cover mixed plain/fitted shared arrays and per-ID references, exact boundary
pricing, actual HTTP rejection before acceptance, invalid/nonfinite payloads, disconnect,
partial-array failure/restart and a stalled real store write. The stalled test fills the
channel to prove the array is dequeued while its bytes remain charged. Serving: 330 pass,
28 ignored plus the explicit known Metal exclusion; binaries/tests compile. Bun 55
tests/types/build pass. CPU-owned Rust browser evidence
`web/.browser-test/baseline-50a63064-a99b-4ba5-934f-531c6c72ecff/` proves no implicit array
network requests, explicit outside-top-k rank/score lookup, retained-copy state and
history/desktop parity without new jobs; numerical heads remain synthetic.

`cx` approves the CPU-qualified producer. The existing bounded production-server gate
now has an opt-in retention mode checking 16 arrays, SHA256/finite/shape/coordinate joins,
full-vocabulary ranks against top-k, unchanged samples and exact bytes after restart.
Live verification is pending at this checkpoint. That gate establishes integrity and
top-k correspondence, not independent correctness of every retained numeric value.

### R13: Production Retained-Array Integrity And Restart

The clean production server at `23e7b8fa` passes the bounded retained-data extension
on Qwen3.6 35B A3B UD-Q4_K_M. The observed job saves eight source arrays and eight full
score arrays, totaling exactly 8,011,776 raw bytes. Every payload passes length, SHA256,
finite-value, width and coordinate checks. Saved top-k values round-trip to their F32
payloads and have the expected ranks over the complete vocabulary. Nonretaining IDs
remain unlinked. Samples/consumption match baseline and all six required generation
witness records pass. Every array and result page is identical after restart, without
new inference; disconnected retry, ordinary serving and active-job shutdown still pass.

Both owned servers exit via handled SIGTERM without forced cleanup; the lease is
confirmed free. Normal memory/lease policy remains in effect, with KV snapshots off.
Evidence: `target/lens-baseline-9819c805-abbe-46ec-93a8-d482887ca29a/`.
Binary Git blob fingerprint: `d4fa11bee2f73aa8d809befbac5f3112953c7ec1`;
harness fingerprint: `4e0e4a95db53b8c97edd39c5f45de8d83d06e10e`.

### R14: Original-Forward Whole-Site Pairs

The shared ordinary forward now captures independently selected before layers after
the block and before its ordered program, then after layers after the whole program.
Both scatters stay in the same command buffer; no extra forward or generation tail
is introduced. Existing CLI callers pass no before capture. Writable aligned F32,
shape/range/layer, session-storage and capture/direction overlap checks precede token
staging. Disjoint views of a shared allocation remain permitted.

Native pairs have bounded independent scopes, merged after captures and separate
before storage/readback. Pair-only work adds no readout heads or fitted workspace.
Archive admission counts unique before arrays plus after arrays shared with retained
readouts. The writer keeps its existing combined payload budget and FIFO dependency
ordering: before, after, pair, then any heads. Metrics promote actual F32 components
to F64 before subtraction; zero before norm has null relative delta. These are local
whole-program measurements, not per-operation intermediates; earlier interventions
can already have affected the before state.

CPU gates cover scope limits, independent layer sets, duplicate pair requests,
shared after arrays, exact archive boundaries, pair-only publication/restart and
HTTP exact retry after capability loss. A writer-boundary test injects cancellation
or partial-array failure after a committed before prefix; it exercises the real
after-array/FIFO dependency path, not the complete paired producer. The actual Rust
CPU browser gate passes phone authoring/reload, explicit array verification, request
copy/history and desktop reopening without new jobs or implicit array fetches:
`web/.browser-test/baseline-af5d9267-ebe1-4215-93c0-5a8dfe03f0e9/`.

The bounded `QWEN_LENS_ORACLE_MODE=pairs` test-binary harness is prepared but not yet
run. It checks actual retained before vectors at later prefill/decode sites against
an independent sequential F64 program, rejects the opposite order, requires exact
unique site/token coordinates, checks no-op/zero controls and unchanged sampling
and consumption with capture disabled. It also checks invalid capture destinations
and a valid disjoint shared allocation. Metal scatter placement/synchronization are
not established by the CPU/browser gates and remain pending this live qualification.

The first bounded live attempt at `907f18ab` stops during destination validation
after 2.89 seconds, before any native test job. The fixture incorrectly assumes a
materialized output-normalization tensor is read-only. It is writable, so that call
is not an invalid-destination test. The owned process exits and releases the lease;
this is not passing numerical evidence. The corrected fixture first requires a
retained read-only head view, then checks capture refusal with F32 shape metadata;
no deployment weight bytes are modified in that rejection case. Evidence of the
failed attempt: `target/lens-fitted-c7e9f2a2-ac7f-41b0-9788-89f8888bd59c/`.

The provenance precondition at `7ba3a46b` also refuses this deployment: its ordinary
head is writable. That second attempt exits in 2.73 seconds before test forwards or
native jobs, with the lease free (`target/lens-fitted-6e4ba8f0-ef44-4f36-8997-b2ed60d6f67d/`).
Read-only qualification is therefore separated from deployment-dependent weights:
the existing library test `hidden_capture_destinations_require_safe_independent_f32_ranges`
constructs explicit read-only provenance and exercises the same validator. It must
run with `QWEN_REQUIRE_METAL_TESTS=1`, so unavailable Metal cannot silently pass. The
loaded-model oracle retains all other full-forward destination and numerical checks.

### R14: Actual-Before Numerical Qualification

The clean release test executable at `ffe19faa` passes the bounded pairs gate on
Qwen3.6 35B A3B UD-Q4_K_M, with six three-sample jobs and 10,093ms model residency.
Four later intervention sites (prefill index 1 and decode index 0, in both authored
orders) match the independent F64 transform of their actual archived before vector.
Maximum absolute errors are `1.2449929513991265e-6`, `2.0419374635594068e-8`,
`2.845585221677993e-6` and `3.282920157943181e-8`; every actual result rejects the
opposite-order reference with disjoint tolerance bands. Twenty exact unique pair
sites pass coordinate/token joins and independently recomputed metrics. No-op and
zero-control arrays match; capture does not change same-program samples or consumed
counts; pair-only work matches baseline samples without evaluating readout heads.

The same owned process checks full-forward shape/dtype/alignment/range/layer,
capture/capture, capture/direction/source/target and session aliases, then accepts
disjoint capture/direction views of one allocation with no generation tail. It exits
normally, without deadline escalation. Evidence:
`target/lens-fitted-0fe0e5a8-bad2-4592-9297-fea988af312e/`.
This is a test-instrumented original-forward numerical gate, not an uninstrumented
production deployment or fitted-asset quality claim.

Separately, the exact existing library capture-validator test passes in 0.04 seconds
with `QWEN_REQUIRE_METAL_TESTS=1`, covering explicit read-only provenance rather than
assuming deployment weights are read-only. Both children use normal lease/memory
policy with inherited safety overrides removed; the lease is confirmed free after
each. Final CPU gates are 338 serving tests and 391 Lens CLI tests (named unmarked
Metal tests excluded); Bun has 55 passing tests, passing types/build, and the combined
operations/fitted/retention/pairs real CPU browser flow also passes:
`web/.browser-test/baseline-4486e7c7-0f4b-4f70-8ff2-c64cfe4684c0/`.

### R06/R15: Cross-Family History And Independent Assets

The final behavior audit identifies two formerly narrowed modes: saved history
alongside DS4/Muse/Flash/K2, and standalone `--web-root`. Both now use the existing
shared owner/control loop with independently optional store/assets. Unsupported
native families report unavailable execution but keep saved results and exact-key
recovery. Fitted configuration still fails early outside ordinary Qwen. No temporary
store, alternate executor or new service is introduced.

The owner sink already includes the standing control allowance in its request
reserve. Each family's existing request admission uses it exactly once; K2 keeps
that same amount for fresh/poisoned replacement sessions. Backend-owned allowances
cover Flash snapshot capture and DS4 promotion/capture/rebinding paths without a
request sink. DS4 eviction retry retains the allowance; the new rebinding check only
runs while control is enabled, preserving the no-workbench path. The allowance
remains active through worker joins and is cleared before backend shutdown.

CPU integration covers all four family request profiles with history and standalone
assets, ordinary responses, unavailable native discovery, history/results/exact
retry and owner callbacks. Exact sink-reserve and pressure-boundary tests prevent
double-counting control memory. These are control/protocol/admission-policy tests,
not numerical family inference or real-pressure qualification. DS4's post-session
reserve check still does not price its session-construction peak. No large model
load is justified for these CPU-owned behavior changes.

### R05: Bounded Shared Prefill Progress

The final consumed extraction recovers the small bounded prefill loop across
ordinary CLI/sweep arms, native jobs and ordinary serving. It owns monotonic progress
and final-step output, not scheduling, checkpoints, model routing or consumption.
Serve keeps packed spans, restored-prefix/no-work behavior, DFlash capture windows
and serial-tail decisions; CLI keeps current `forward_event` dispatch rather than
restoring the obsolete generic production dispatcher. Native consumption and
observation remain after successful forward and before sampling. CLI additionally
checks process shutdown before each prefill step, as the preserved consumer did.

Invalid initial/progress bounds fail rather than looping; a no-output final CLI or
native step cannot reuse stale logits. Obsolete step output drops before the next
closure, avoiding an unnecessary old-logit overlap through a serial tail. CPU tests
cover no-work restore, mixed spans, output lifetime, progress failures and failed
native prefill observation retaining known consumption without further forwards or
sampling. No new inference capability or scheduler is introduced. Final production
baseline requalification will cover this build, not claim DFlash or restored-prefix
numerical qualification from a cache-isolated native test.

## Final Recovery Audit And Handoff

The audited deliverable is the semantic commit series rooted at
`8bc9e6b739e953023a1f2ffc5dc3379c22758e00`, not the original dirty branch. A final
fetch still resolves `origin/main` to that base. No merge, push, amend or edits to
the preserved worktree/package occurred. The tree is clean before this final
documentation reconciliation; generated bundles and test evidence remain ignored.

The final inventory covers the frozen tracked changes and untracked source groups:

- CLI comparison/output/sweep changes are recovered, including optional retained
  `generation_input`, writer/strict-reader agreement and comparison-context checks.
  Sweeps/cohorts did not gain typed prefills in the frozen implementation; they are
  not missing recovered features. Nonordinary run dispatch retains current-main
  families and their existing artifact defaults.
- `assistant_prefill_tests.rs` is superseded by the consumed generation/input and
  comparison test modules. The production-unused `GenerationInput`/renderer wrapper in
  `open_responses/render.rs` is replaced by shared preparation using the current
  annotated renderer, not a second rendering authority. Frozen native tools,
  tool-prefill, raw input and upstream-style selection were also unavailable.
- Shared transport verification and the full-head readout workspace are consumed
  by CLI and native jobs. Current expected-profile and passive-head constraints
  remain; owned writer staging replaces the old readiness/dispatcher handoff.
- Frozen `serve/queue.rs` is intentionally replaced, not copied. The old sixteen
  HTTP workers/eight work reservations become two bounded control workers and one
  exclusive execution reservation. There is no multi-job backlog: new work is
  refused while busy, but accepted-key retries/history/cancel remain independent.
  This trades burst buffering for explicit owner lifetime and bounded preparation
  memory. It is a deliberate policy change, not an equivalent queue-depth claim.
- All-family ordinary profiles, history, standalone assets, traces and native
  original-forward producers have consuming paths and gates. The reconstructed
  owner retains current-main idle/request-finished/shutdown and snapshot semantics
  instead of reinstalling the obsolete frozen accept loop.
- The Bun client, fixtures, saved history, direct token/layer/score navigation,
  pinned directions, scope editing/reordering, variants, cancellation, retained
  queries and pair inspection are recovered. No desktop-only replacement is added.
- Frozen `web/live-check.ts`, `web/fitted-oracle-check.ts` and
  `web/residual-evidence-check.ts` remain preserved reference harnesses. Their
  numerical/integrity/lifecycle assertions are covered by the current owned
  `scripts/serve/lens_baseline_check.ts`, `scripts/serve/lens_fitted_check.ts`,
  Rust numerical oracles and actual-producer CPU browser checks. Old private
  fixtures/evidence are not copied into source or promoted to current proof.

### Final Build Gates

Production code `6ff11bc3` passes the bounded Qwen3.6 35B A3B UD-Q4_K_M retained
baseline gate: unchanged samples/consumption, six required same-forward witnesses,
16 arrays totaling 8,011,776 bytes, full-vocabulary rank/integrity checks, exact
results/array bytes after restart, disconnected exact-key recovery, ordinary serving
and active-job interruption. Both owned servers exit through handled SIGTERM with
no forced cleanup; the lease is confirmed free. Model load times are 2368.1ms and
2377.2ms. Evidence:
`target/lens-baseline-d65bf072-d14e-4246-8f85-3432fc817b67/`.
Production binary Git blob fingerprint: `3270b0f97797b82ce00933b5d9063f105b8afa07`;
harness fingerprint: `4e0e4a95db53b8c97edd39c5f45de8d83d06e10e`.

Final CPU checks pass 342 serving tests (28 opt-in ignored), 394 Lens tests (nine
opt-in ignored), 14 shared lifecycle tests and 14 CLI tests. The known unmarked
Metal cases are explicitly excluded, not silently treated as CPU evidence. All
binaries/tests compile and formatting/whitespace checks pass. Bun passes 55 tests
with 327 assertions, types and build. The final actual-Rust combined diagnostic
browser gate passes at phone/desktop widths:
`web/.browser-test/baseline-2c166911-8082-47b8-a6e2-86d4b09b8cba/`.
The full historical browser fixture suite also passes identity/persistence races,
token/layer/score manipulation, pagination, history and layout bounds without GPU.

Independent fitted/intervention/paired numerical evidence remains scoped to the
specific clean builds documented at R11/R12/R14; the final production baseline is
not a substitute for those or a claim that they were all rerun on the final build.
This recovery does not qualify every family numerically, DFlash/restored-prefix/KV
persistence, real memory pressure, fit quality, visual screenshots or CLI/HTTP
numerical equivalence across all modes. Those boundaries do not hide unrecovered
source or a feature-sized dirty worktree. Subsequent work returns to interaction
design and usable information density, not further capability expansion by default.

Final read-only `cx` review gives GO for complete semantic recovery, ready for
integration against the pinned main. It finds no unaccounted frozen capability loss
in the reviewed source/docs, while explicitly not rerunning the reported tests or
approving a merge/push. The final handoff contains only committed source and docs.

## Independent Pre-Merge Review Corrections

A subsequent independent review found operational defects missed by the earlier
readiness review. The original recovery evidence remains scoped to its recorded
builds; it did not qualify these failure/performance boundaries.

- `016b3c8f` includes `arrays.bin` in abandoned acceptance cleanup. Fault tests
  now cover archived/unarchived jobs on both sides of the acceptance rename.
- `3efee14b` invokes backend shutdown before reporting unsettled owner activity,
  with a regression proving exactly one flush callback despite accounting failure.
- `f43e616d` isolates connection clone/spawn failures and HTTP worker panics from
  owner lifetime, logs/retries transient control accept/readiness failures, and
  recovers valid gate/submission bookkeeping after poison. Model/maintenance state
  is not blindly recovered. Injected connection failures are followed by successful
  ordinary requests; process-shutdown/backend errors remain fatal to the owner.
- `13db80fc` uses a 16 MiB incremental control allowance for standalone assets,
  retaining 512 MiB only with history. Explicit watchdog stacks and selected-reserve
  propagation are tested. `e95ff909` refuses cancellation without a store before
  parsing its JSON body, so unavailable diagnostic routes do not invalidate that
  smaller allowance. Neither allowance is a total allocator bound.
- `26fb8f1d` classifies request heads before acquiring model activity. Trusted
  preparation inhibits idle but only accepted native work reports completion.
  Partial heads, history/static reads, rejected Host/Origin, malformed native input
  and exact retries do not reset idle publication. Tests include actual native
  preparation rejection and a header handoff before blocked history access.

### Bounded Batched Publication

The writer now batches available metadata rather than syncing every record and
progress snapshot independently. Already serialized producer objects are validated
without nested `Value` trees, then published with progress through the store's same
transaction implementation. Exact sequence/newline sizing, root-owned fields, full
JSON validation and JSONL framing are checked before writes. Progress-only batches
publish no fictitious records; every intermediate progress transition is validated.

Batches drain at most 128 events and respect the 4 MiB encoded store limit. Array
and staging events are FIFO barriers, including any deferred event. Original storage
and permits remain charged through append. The 8 MiB metadata budget, optional array
budget and 82 MiB writer admission allowance do not grow; the latter already covers
the bounded originals/batch buffer, validation and pending producer without batch
JSON tree amplification. Saturation cooperatively waits on the same event or claim,
checking stop/failure every 5ms. Interrupted blocked publication reports a separate
artifact error without rewriting a known generation outcome. Filesystem calls and
final joins remain non-preemptible.

CPU coverage includes a maximum-admitted actual-producer job (4096 heads, 16384 rows,
262144 scores) with a deliberately stalled writer; exact completion and bounded
peak permits are required. Overall batching ratio is only a measurement. A preloaded
126-record batch plus array/dependent barriers verifies deterministic batching and
durable prefix recovery. Channel/byte waiting tests inject cancel, shutdown and the
writer-failure latch; independent real store-fault tests cover actual append failures.
These do not claim every filesystem-failure/backpressure interleaving. Encoded/value
record equivalence, progress-only state, invalid intermediate progress, sequence
digit/overflow limits and snapshot-rename recovery also pass. The wider production
gate is prepared to exceed 128 readouts and recheck restart; its live evidence is
recorded separately after the code commit.
