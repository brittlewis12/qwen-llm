# Lens Workbench

## Workbench Capabilities

The workbench provides a durable baseline API and a Bun/React
client on `qwen serve`, including scoped plain and registered fitted original-forward
readouts, ordered scoped interventions, retained source/full-score arrays and
whole-site before/after pairs from the original forward.
Compatible historical records remain inspectable. The API and browser share the
same resident server; qualification boundaries are described below.

```sh
qwen serve -m MODEL --lens-data-dir JOB_DIRECTORY
```

For the browser, run `bun install --frozen-lockfile` and `bun run build` in `web/`,
then add `--web-root web/dist` to that same server command. Rust loads the prebuilt
catalog before the model; it never starts Bun or another production service.
`--web-root` and durable history are independently optional on every serving family.
Without history, the client loads with execution/history unavailable.
The catalog's 64 MiB limit counts payload lengths, not metadata/allocation overhead.
Symlink checks reject known unsafe paths but are not race-free ancestor confinement.
API paths cannot become static assets. See [`web/README.md`](../web/README.md) for
development, draft/retry safeguards and saved diagnostic exploration.

Native generation requires an ordinary Qwen backend with a metadata-qualified
Qwen3.6/3.8 House template. Other deployments/families still expose saved history
with `available:false`; explicit fitted configuration is ordinary-only.
The server remains loopback-only.
Lens routes require a loopback/localhost Host and, when supplied, a matching HTTP
Origin by default. For HTTPS through `tailscale serve`, explicitly allow the browser
origin while keeping the backend bound to loopback:

```sh
qwen serve -m MODEL --addr 127.0.0.1:8737 --lens-data-dir JOB_DIRECTORY \
  --web-root web/dist --lens-allowed-origin https://machine.tailnet.ts.net
tailscale serve --bg http://127.0.0.1:8737
```

Use the HTTPS origin reported by your Tailscale setup, with no trailing slash or
path. Repeat `--lens-allowed-origin` for additional exact origins; non-default ports
must be included. The proxy must preserve the original Host (as Tailscale Serve
does). A supplied Origin must match both the configured scheme and the request Host;
forwarded headers do not grant access. No-Origin clients are allowed at a permitted
Host. These checks are not authentication: Tailscale and its access policy control
who can reach the proxy. This does not change ordinary `/v1/responses` access.

The browser retries failed GETs at most twice for HTTP 502, 503 or 504, retaining the
same path/cursor. Submission and cancellation POSTs are never automatically retried.
The real Tailscale HTTPS CPU-fixture browser gate passes submission, reload, history
and phone/desktop navigation, including recovery from an observed transient 502.
This does not guarantee overload response delivery: the bounded acceptor currently
sends busy responses before reading requests, and early socket closure can lose that
response through a proxy. Bounded server-side response teardown remains a follow-up.

The same resident owner continues to serve `/v1/responses`. Native jobs use its
serial post-block forward path with fresh isolated sequence state: no prefix
restore, ordinary cache publication, speculative draft or redundant model pass.
Neither API requires another model server. Input rendering/tokenization and typed
prefill transitions are shared with the Lens CLI. Raw input, tool messages and
upstream-style native rendering are deliberately refused rather than approximated.

## Submit And Inspect

Read `GET /v1/lens/capabilities` and `GET /v1/lens/assets` first. Capabilities
advertise supported generation modes, context and byte limits. A supported passive
output head adds the `plain` alias and `full_vocabulary` readouts; otherwise baseline
generation remains available with empty assets. Model identity describes GGUF
metadata, not a weight content hash. New clients should supply it as a precondition:

```json
{
  "schema_version": 1,
  "idempotency_key": "one-stable-key-for-this-submission",
  "input": {
    "kind": "messages",
    "messages": [{"role": "user", "content": "Name an animal."}],
    "generation_mode": "thinking",
    "assistant_prefill": {"channel": "final", "text": "Answer: "}
  },
  "generation": {
    "max_new_tokens": 3,
    "sampling": {"temperature": 0, "top_k": 0, "top_p": 1, "min_p": 0, "seed": 7}
  },
  "preconditions": {"model_identity": "COPY_FROM_CAPABILITIES", "asset_identities": {}}
}
```

POST this JSON to `/v1/lens/jobs`. Unknown fields and unsupported diagnostics are
rejected, not ignored. Temperature zero explicitly requests greedy sampling;
positive values that narrow to zero are refused. Prefill text is exact prompt
context, never newly generated output. All template/mode combinations are checked.

Acceptance is persisted before dispatch and acknowledged with 202/Location. Save
the exact request/key before sending. If acknowledgement is lost, resend it:
identical normalized content returns the existing job, even if current discovery
or capacity changed. Different content under that key is a 409 conflict. Only a
matching-key `admission.state=not_accepted` response proves non-acceptance; generic
503/network/storage errors do not authorize creating a new key.

- `GET /v1/lens/jobs`: paginated cross-client history and last-user previews.
- `GET /v1/lens/jobs/{id}`: status, progress and independent publication outcome.
- `GET /v1/lens/jobs/{id}/request`: saved authored request, without retokenization.
- `GET /v1/lens/jobs/{id}/result`: bounded immutable record pages with opaque cursors.
- `POST /v1/lens/jobs/{id}/cancel`: empty body or `{}`; explicit cooperative cancel.
- `POST /v1/lens/jobs/{id}/delete`: empty body or `{}`; delete a settled terminal job's payloads, retaining its retry identity.

Disconnect and history navigation never cancel accepted native work. Result pages
retain exact prepared prompt text/bytes/token IDs/spans, sampling, sampled pieces,
known consumption and terminal outcome. Stop/token-limit samples remain unconsumed.
A failed forward reports unknown consumption, not a fabricated capture; a failed
piece decode retains the sampled token ID without inventing bytes. `result.complete`
means publication ended, not that every requested artifact was saved. Disk/queue
failure can make publication fail without rewriting already-completed generation.

The store syncs records before publishing their committed watermark. Reopening an
unfinished job marks it interrupted; it never resumes or reruns inference. Corrupt
committed content detected during recovery disables Lens acceptance while healthy
history stays readable. Retained-array descriptors can be downloaded and
verified without inference; baseline jobs without requested retention produce none.

## Plain Readouts

When discovery offers `plain`, add readouts in the browser or a request's
`diagnostics.readouts`, with empty `directions` and `operations`. Each readout has
`id`, `lens:"plain"`, `mode:"full_vocabulary"`, `top_k` and a numeric `scope`:
`layers` plus `prefill` and/or `decode` selectors. Selectors are `all`, sorted unique
`values`, or inclusive `range`. Omitted phases mean no observation in that phase.
Include `plain` with its advertised identity in `preconditions.asset_identities`.

Source layers are post-block residuals; prefill/decode indices name consumed input
tokens, not the position they predict. The final sampled token is never forwarded
merely to fill a requested readout. Residuals come from the original forward, then
the existing output head evaluates them without a transformer replay. The original
generation logits remain the sampler's input. One head and maximum-k ranking serves
overlapping IDs at a given position/layer; each ID retains its requested top-k row.
Only the first row reports head cost, with shared position/layer identity on all.

Final-layer observations get a same-forward generation-logit witness only when that
forward actually produced generation logits. Earlier no-tail prompt positions do
not acquire extra tails for comparison. A failing tolerance witness stays visible;
it is not passing qualification. Failed forward, failed observation and interrupted
publication remain distinct. Successful consumption is recorded before observation;
an observation failure never rewrites it as unknown forward consumption.

Admission bounds 1024 readouts, top-k up to min(1024, vocabulary), 4096 distinct
heads, 16384 rows and 262144 scores. Cardinality is checked before expanding `all`.
Per-head aggregate raw token-label bytes are capped at 64 KiB before lossy display
decoding; overflow fails publication explicitly rather than truncating labels or
scores. Records use bounded borrowed serialization. GPU admission prices each
aligned capture/head buffer. Host allowance includes capture readback, transported
vector, all three simultaneously live previous/new/observer logit arrays, bounded
ranking/labels/serialization, plus existing durable/control/writer reservations.
Eviction retry retains the same complete allowances; denial precedes capture setup.

## Fitted Readouts

Add `--lens-config lenses.json` to the same history-enabled server. This currently
requires an identified Qwen3.6/3.8 House protocol. Configuration is explicit; no
automatic lens discovery or second model service is introduced:

```json
{
  "schema_version": 1,
  "assets": [{"alias": "fitted", "path": "assets/my-transport"}]
}
```

Paths are relative to the config file. Assets use the data-only
`llm.lens.linear_transport` contract, not legacy published-import manifests. Exact
deployment bindings must match retained model bytes and tokenizer identity. For an
artifact without an exact binding, explicitly add `"allow_unvalidated_transfer": true`
to that asset; this cannot override a mismatched exact binding or geometry. Producer
qualification remains a claim, not independent proof of fit quality. Startup verifies
manifest geometry before payload scanning and loads the same retained GGUF afterward.

Select the advertised alias in the existing readout editor, or use it as `lens` with
the same scopes and top-k fields as plain readouts. Select only its advertised source
layers and pin its advertised identity. Heads are shared by position/layer/alias;
captures are shared across aliases. Saved fitted rows retain target, artifact identity,
method and binding status, and never claim to be the final generation distribution.

The owner admits all pending durable/control/writer memory, selected host matrices,
scan scratch, capture, sequence and aligned fitted workspace before any matrix staging.
The existing joined artifact worker then rehashes selected matrices and transfers them
once, with cancellation and shutdown checks. No additional worker or dispatch queue is
created. Fresh staging checks preserve the full original future process allowance;
already-staged bytes can be conservatively counted again, causing over-refusal under
pressure. Up to 512 MiB of unique alias/layer matrices may be staged per job. This cap
does not include the other allowances. Staging failure settles without model forwards;
filesystem calls remain cooperatively cancellable, not deadline-bounded.

## Scoped Interventions

Registered assets whose deployed output head supports selected-token covectors also
offer token-ID directions. Pin directions in the browser, then add/reorder `fixed_add`,
`residual_l2_fraction` or `projection_ablate` operations. The same authored scope fields
select layers and consumed prefill/decode positions. Operations execute in transformer
layer order, preserving authored array order within each layer. Operation-only jobs
need no readouts; adding observation does not add transformer passes.

Directions compute the transport transpose times the deployed, gamma-folded LM-head
row (`deployed_logit_numerator`). This is not the gradient of a normalized logit, and
is not the raw LM-head-row convention used by some other tools. Fixed addition allows
`as_stored` or `unit_l2`; residual-relative addition and projection attenuation require
`unit_l2`. Relative scaling uses the current residual norm, including preceding
operations. Projection subtracts the coefficient times the current projection onto
that unit direction. Source-to-target, coordinate-swap and other covector choices
remain unsupported natively; their existing CLI behavior is unchanged.

Zero coefficients validate references, scopes and normalization but produce no events,
matrix staging or direction uploads. Unused direction identities remain pinned. Matrix
staging is deduplicated across interventions and fitted readouts; GPU readout workspace
is allocated only for actual fitted readouts. Full owner admission includes retained
direction rows, aligned projection/selection scratch and host preparation/publication
buffers. The conservative estimate sums preparation and readout phases; it can
over-refuse compared with exact phase-peak accounting.

Saved `direction_prepared` records identify semantics, normalization, binding and vector
digest; `operation_application` records follow successful consumption, in applied order.
Readouts carry the operation IDs applied at that site. Failed forward never fabricates
successful applications; later recording failure does not imply model-state rollback.
Limits currently allow 1024 definitions/operations, 4096 prepared direction rows,
16384 applications, 256 MiB of direction vectors and 1,073,741,824 projection products.

## Retained Measurements

Enable "Retain full scores and source residuals" on a readout, or set
`"retain":"scores_and_residual"`. The original captured post-block residual is saved,
including interventions applied there; fitted transported vectors do not replace it.
Full pre-softmax logits are saved before top-k reduction. Source arrays deduplicate
by position/layer and score arrays by position/layer/alias. Other readout IDs sharing
the same head still have no retained reference unless they requested retention.

Raw archive admission is exact for the maximum reachable requested sites: four bytes
per retained scalar, at most 4 MiB per array and 32 MiB per job. Excess is rejected
before acceptance, never truncated. Early termination may consume less than this upper
reservation. Raw bytes are separate from JSON/result budgets and total process memory.
Optional memory admission includes the bounded 16 MiB payload queue (including its
active write) and a 4 MiB producer allowance. This is not a total allocator bound.

The joined writer syncs each F32LE payload and its descriptor before publishing their
watermarks. SHA256, length and finite values are validated on download/recovery. The
source, scores and dependent readout are ordered, but are separate transactions: a
durable source-only prefix is valid if later publication fails. Inspect the independent
result/observation error rather than treating generation completion as artifact success.

Saved readouts expose an explicit "Load retained scores / no inference" action. Only
then does the client fetch the array; token queries outside the original top-k, rank
and distribution summaries use those saved bytes. Reloading, history navigation and
reopening arrays never submit new work. Unretained or uncaptured sites cannot be
reconstructed from top-k rows alone.

## Whole-Site Before/After Pairs

Use "Add residual pair capture", or add a `diagnostics.residual_pairs` entry:

```json
{"id":"change","scope":{"layers":{"kind":"values","values":[19]},"decode":{"kind":"values","values":[0]}}}
```

The original forward captures just before and after the complete ordered post-block
program at each selected site. Scopes are independent of operations and readouts;
pair-only and no-operation/zero-control requests work without readout heads. Earlier
interventions may already have affected the before state. These are not independent
baseline comparisons or per-operation intermediate vectors.

Pairs always retain both residual arrays, sharing the after array with retaining
readouts at the same site. They use the same raw archive limits and add separate
before capture/readback memory. At most 1024 pair requests and 16384 pair rows are
admitted. Saved metrics include before/after norms, delta norm and relative delta
(null for zero before norm). "Verify measured change / no inference" explicitly
loads the saved arrays to recompute the metrics; history navigation never does so
implicitly. Arrays precede dependent pair records, but publication is not atomic
across the entire measurement.

## Bounds And Lifecycle

One execution reservation covers ordinary preparation/response cleanup or a native
job through writer completion. A new native request receives 429 while it is busy;
an ordinary request receives 503 before its body is read. With history or assets enabled,
two bounded CPU classifier workers read headers and handle history/retry/cancel
while the owner runs inference. With both disabled, existing pre-header busy
behavior is unchanged. This is not multi-model scheduling or concurrent inference.
Static GET/HEAD requests share that CPU pool and release read-only activity before
writing borrowed immutable bytes. They do not reserve execution. Bodies on static
reads are refused before allocation. Client discovery is sequenced; read-only 503s
receive at most two short retries, never automatic resubmission or cancellation.

Partial request heads, static/history reads and rejected Host/Origin traffic do not
claim model activity. Trusted Lens body/preparation work inhibits idle after head
classification, without resetting the completion timer on rejection or exact retry.
Durable acceptance promotes an independent native lifetime through joined publication.
Ordinary parsed requests retain their existing completion behavior, including errors
and model-list requests. Only the owner invokes maintenance and shutdown callbacks.
Shutdown closes admission/delivery, interrupts cooperative execution, settles
connections and joins control/artifact workers before the existing backend flush.
An individual GPU command is not preemptible. Filesystem operations and writer joins
have no deadline: a stalled filesystem can delay shutdown beyond the model-snapshot
flush allowance. No new detached writer hides that limitation.

Current bounds are explicit, not capability targets:

- Request body: 1 MiB. Prepared/sample record: under 1 MiB including framing.
- Token piece: 104755 bytes, derived from worst-case retained JSON expansion.
- Writer: 128 queued events and 8 MiB of retained record allocation capacity.
- Result page: 256 records / 2 MiB. History: 4096 retained jobs / 64 GiB charged
  storage and 65,536 permanent retry identities, including deleted jobs; no silent eviction.
- Native writer admission allowance: 82 MiB, including stack and publication work.
- History-enabled control allowance: 512 MiB across both worker slots.
- Standalone-assets control allowance: 16 MiB for classifier/watchdog stacks and buffers.

The writer drains available metadata into bounded batches, with a 4 MiB encoded
store limit including sequence fields/newlines. Records and their latest validated
progress share one durable snapshot publication. Array/staging events are FIFO
barriers; each array retains its independent durability transaction. Already encoded
metadata is validated without constructing nested JSON value trees. The existing
82 MiB allowance covers budgeted originals, batch serialization, validation and one
pending producer record; optional array queue/producer allowances remain separate.

Queue or byte-budget saturation waits cooperatively instead of failing a valid wide
scope. Waiting retains the same event and permits, checks cancellation/shutdown and
writer failure every 5ms, and never grows the budgets. An interrupted blocked
publication reports an explicit artifact error; known generation outcomes remain
separate. This does not make blocking filesystem calls or writer joins preemptible.

The control allowance covers overlapping bounded request/result JSON trees,
serialization, prepared-input token/span construction, watchdog/worker stacks,
ordinary handoff and native writer setup. It is a conservative future-allocation
allowance, not an allocator-enforced total bound. Qwen admission and pressure-relief
retry retain it alongside durable-snapshot reservations; capture/spill and optional
DFlash capture/state checks retain it too. Optional DFlash capture is admitted
before allocation and falls back to uncaptured serial execution on refusal.

Admission also asks for the full control allowance as fresh process headroom at
startup and each connection. This can conservatively return 503 for history or
cancellation when part of that allowance is already in use. Both workers can also
be occupied by slow clients or filesystem access. Cancellation remains cooperative,
not a guaranteed always-available priority lane. No memory safety override is used.

## History Cleanup

History shows charged storage and both retention/identity limits. Delete job payloads
explicitly removes a confirmed terminal job's prompt, records and arrays only after
its execution and writer have settled. The server first publishes and syncs a compact
tombstone containing its accepted key, request hash, observation flag and terminal
outcome. Exact submission retries return `deleted:true` without new inference;
conflicting bodies still return 409. Payload endpoints return 410 `job_deleted`.

Logical deletion precedes physical cleanup. A failed cleanup stays charged and can
be retried for the same job; unfinished cleanup remains visible in history. Restart
resumes cleanup before exposing the tombstone. Payload readers already in progress
finish before deletion; later readers see deletion rather than missing-file errors.
Retry identities never expire silently; exhausting their separate bound refuses new
acceptance even if payload storage is free. Existing jobs require no migration.

Deleting through History invalidates that browser's cached viewer. Completed viewers
do not poll forever: reopen a history entry or use Reconnect to discover deletion by
another client. A paginated history cache does not infer deletion from a missing row.
The separate Clean completed browser archives action verifies archived job IDs against
the server and removes only confirmed-terminal archives under the submission Web Lock.
It preserves the current intent, unresolved/active/unverifiable archives, and server
history. No automatic cleanup or eviction is enabled.

## Incomplete History Recovery

A job that fails startup recovery is isolated from the readable index and reported
in capabilities/history, with up to 64 unavailable IDs and an exact unavailable
count. This includes malformed snapshots/requests, missing or structurally invalid
committed JSONL records, invalid sequences, array integrity failures and failed
per-job recovery publication or deletion cleanup. Non-array records now receive the
same bounded JSONL/sequence scan as archived jobs; this adds startup I/O proportional
to committed history. It is structural validation, not a checksum of every record.

While any job is unavailable, all Lens submission attempts (including exact retries)
and server deletions fail with 500 `history_recovery_required`, never `not_accepted`.
Healthy status, prompts, results and arrays remain readable; ordinary inference is
unchanged. Reported storage usage excludes unavailable jobs. This conservative policy
does not guess which accepted identities are safe to forget. Repair stored files
from trustworthy evidence while stopped, then restart; no inference is replayed.

There is no automatic destructive repair or eviction. Normal recovery can still
truncate uncommitted tails, publish interruption or resume an already committed
deletion before a later recovery step fails. Root ownership/locking, root sync,
inventory-limit and abandoned-acceptance cleanup failures still prevent startup.
Without a separate acceptance journal, missing job directories or names moved outside
the recognized job namespace cannot be detected: do not manually remove a damaged
directory to bypass the refusal. Use the explicit deletion API for healthy history.

## Publication Failures

Publication failures have a process-local `runtime` overlay in status and history.
The normal state, revision and counters remain the last confirmed durable snapshot;
the overlay separately reports publication failure, whether execution and its writer
have settled, and any known in-memory generation outcome. It is never written into
the durable snapshot. The browser labels stale state, reads the committed prefix and
stops polling after settlement and prefix exhaustion (or an unreadable prefix).
Restart reconciles disk state normally and removes this process-local overlay;
neither failure reporting nor restart reruns inference. A healthy durable terminal
publication supersedes the overlay. Filesystem operations can still block settlement.

## Qualification

CPU tests exercise real sockets, the owner loop, the store and joined publication,
using the actual tokenizer implementation over a synthetic byte vocabulary and
mock forwards. They cover disconnect/retry/history without extra forwards,
cancellation during execution and startup, active native shutdown, read-only idle
accounting, publication failures and ordinary serving after native work. Existing
all-family ordinary protocol and real SIGTERM CPU fixtures remain covered.

The recovered client passes Bun types/tests/build and GPU-disabled historical
browser fixtures. A separate CPU-owned Rust child exercises the production assets,
two-worker control pool, durable store and native writer through mobile baseline
submission, reload, exact byte display, history/copy without new jobs and desktop
layout. Browser layout assertions do not establish screenshot-based visual review.
The optional CPU browser command `LENS_TEST_PLAIN_ONLY=1 bun run baseline-browser-check.ts`
in `web/` also submits actual scoped HTTP readouts and compares visible scores to
the saved records, using synthetic heads only. The separate bounded live gate at
`3c74f3c9` passes on Qwen3.6 35B A3B UD-Q4_K_M: first/last prompt and consumed decode
sites at middle/final layers, eight shared heads and 12 exact rows. All six expected
witness records (three distinct final-layer comparisons) have zero maximum absolute
error. Sampled IDs and consumption match the baseline exactly; the final unconsumed
sample has no capture. Observed pages remain identical after restart.

The released Qwen3.6 35B A3B Q4_K_M tokenizer also passes the opt-in native/CLI
prefill matrix on CPU. A clean release built at `b850e3e7` passes the bounded live
baseline lifecycle check on that model: exact greedy repeatability with consumed
decode tokens, disconnected exact-key recovery, ordinary serving afterward, and
active SIGTERM with durable terminal publication before restart. Both server
launches exit through handled SIGTERM (143, not OS signal termination); history
and the interrupted status remain unchanged after restart. The lease is released.

```sh
QWEN_LENS_TEST_MODEL=/path/to/qualified.gguf \
  bun run scripts/serve/lens_baseline_check.ts
# Add one observed job and require all same-forward witnesses:
QWEN_LENS_TEST_READOUTS=1 QWEN_LENS_TEST_MODEL=/path/to/qualified.gguf \
  bun run scripts/serve/lens_baseline_check.ts
# Also verify retained full arrays and byte-identical reopening after restart:
QWEN_LENS_TEST_RETENTION=1 QWEN_LENS_TEST_MODEL=/path/to/qualified.gguf \
  bun run scripts/serve/lens_baseline_check.ts
# Also check a wider scoped job producing more than 128 readouts:
QWEN_LENS_TEST_WIDE=1 QWEN_LENS_TEST_RETENTION=1 \
  QWEN_LENS_TEST_MODEL=/path/to/qualified.gguf \
  bun run scripts/serve/lens_baseline_check.ts
```

The reviewed production code at `30bc6779` passes the retained baseline gate on the
same Qwen3.6 fixture: all six witnesses, unchanged samples, 16 retained arrays,
exact restart bytes and handled active-job shutdown. An additional 160-row job across
40 layers passes exact-coordinate, final-layer witness and restart checks, publishing
14 metadata batches. A separately stalled maximum-admitted CPU producer completes
16,384 rows within the unchanged writer budgets. Separate fitted/intervention/
pair test-binary gates establish scoped independent numerical checks, not fit quality
or every-model coverage. The final CPU browser also exercises combined fitted heads,
ordered operations, retention and pairs through the actual Rust producer. These
checks do not establish all-model or all-mode numerical equivalence.

This opt-in script runs Metal, owns exactly its child servers, uses normal memory
admission and retains local evidence under `target/`. It has a 140-second protocol
deadline plus a separate 35-second cleanup allowance. A successful exit and final
PASS line are authoritative; a partial evidence file alone is not a passing gate.
It intentionally disables RAM/KV snapshot persistence. This is not qualification
of KV continuity, DFlash, memory pressure, every model family, an independent
readout oracle or CLI-versus-HTTP numerical parity. Preserved old-worktree evidence remains separate.
Wire fixtures include historical diagnostic extensions; returned capabilities,
not those examples, define support.
