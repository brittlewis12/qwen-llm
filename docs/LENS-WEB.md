# Lens Workbench

## Recovered Now

The reconstruction provides a durable baseline API and the preserved Bun/React
client on `qwen serve`, including scoped plain original-forward readouts. Fitted
assets, interventions and retained capture production are not yet recovered.
Compatible historical records remain inspectable. This is an intermediate workbench,
not completion of recovery. See [the recovery ledger](LENS-WEB-RECOVERY.md).

```sh
qwen serve -m MODEL --lens-data-dir JOB_DIRECTORY
```

For the browser, run `bun install --frozen-lockfile` and `bun run build` in `web/`,
then add `--web-root web/dist` to that same server command. Rust loads the prebuilt
catalog before the model; it never starts Bun or another production service.
`--web-root` requires durable history and inherits its current family restriction.
The catalog's 64 MiB limit counts payload lengths, not metadata/allocation overhead.
Symlink checks reject known unsafe paths but are not race-free ancestor confinement.
API paths cannot become static assets. See [`web/README.md`](../web/README.md) for
development, draft/retry safeguards and saved diagnostic exploration.

The flag currently requires an ordinary Qwen backend. Native generation requires
a metadata-qualified Qwen3.6/3.8 House template; unsupported ordinary deployments
still expose saved history with `available:false`. The server remains loopback-only.
Lens routes require a loopback/localhost Host and, when supplied, a matching HTTP
Origin. These checks are not authentication or a remote-access feature.

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

Disconnect and history navigation never cancel accepted native work. Result pages
retain exact prepared prompt text/bytes/token IDs/spans, sampling, sampled pieces,
known consumption and terminal outcome. Stop/token-limit samples remain unconsumed.
A failed forward reports unknown consumption, not a fabricated capture; a failed
piece decode retains the sampled token ID without inventing bytes. `result.complete`
means publication ended, not that every requested artifact was saved. Disk/queue
failure can make publication fail without rewriting already-completed generation.

The store syncs records before publishing their committed watermark. Reopening an
unfinished job marks it interrupted; it never resumes or reruns inference. Corrupt
committed content fails closed. Compatible historical retained-array descriptors
can be downloaded and verified, but new baseline jobs produce no such arrays.

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
`retain`, fitted aliases, directions, operations and residual pairs remain refused.

## Bounds And Lifecycle

One execution reservation covers ordinary preparation/response cleanup or a native
job through writer completion. A new native request receives 429 while it is busy;
an ordinary request receives 503 before its body is read. With history enabled,
two bounded CPU classifier workers read headers and handle history/retry/cancel
while the owner runs inference. With history disabled, existing pre-header busy
behavior is unchanged. This is not multi-model scheduling or concurrent inference.
Static GET/HEAD requests share that CPU pool and release read-only activity before
writing borrowed immutable bytes. They do not reserve execution. Bodies on static
reads are refused before allocation. Client discovery is sequenced; read-only 503s
receive at most two short retries, never automatic resubmission or cancellation.

Each mutating HTTP request has a completion lifetime; an accepted native job adds
an independent lifetime through joined publication. Valid read-only Lens routes
release activity before store/socket I/O, so polling does not reset idle snapshot
publication. Only the owner invokes idle, request-finished and shutdown callbacks.
Shutdown closes admission/delivery, interrupts cooperative execution, settles
connections and joins control/artifact workers before the existing backend flush.
An individual GPU command is not preemptible. Filesystem operations and writer joins
have no deadline: a stalled filesystem can delay shutdown beyond the model-snapshot
flush allowance. No new detached writer hides that limitation.

Current bounds are explicit, not capability targets:

- Request body: 1 MiB. Prepared/sample record: under 1 MiB including framing.
- Token piece: 104755 bytes, derived from worst-case retained JSON expansion.
- Writer: 128 queued events and 8 MiB of retained record allocation capacity.
- Result page: 256 records / 2 MiB. History: 4096 jobs / 64 GiB, no silent eviction.
- Native writer admission allowance: 82 MiB, including stack and publication work.
- History-enabled control allowance: 512 MiB across both worker slots.

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
the saved records, using synthetic heads only. Plain-head live numerical parity and
same-forward witnesses still require their separate bounded model-backed gate.

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
```

This opt-in script runs Metal, owns exactly its child servers, uses normal memory
admission and retains local evidence under `target/`. It has a 140-second protocol
deadline plus a separate 35-second cleanup allowance. A successful exit and final
PASS line are authoritative; a partial evidence file alone is not a passing gate.
It intentionally disables RAM/KV snapshot persistence. This is not qualification
of KV continuity, DFlash, memory pressure, every model family, or independent
CLI-versus-HTTP numerical parity. Preserved old-worktree evidence remains separate.
Wire fixtures include historical diagnostic extensions; returned capabilities,
not those examples, define support.
