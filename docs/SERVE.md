# qwen serve — contract and program

Status: S0–S3 functionality is present in the current working tree. Gate
evidence is mixed and is recorded separately below; implementation status must
not be read as a blanket gate pass.

Evidence base: S0 packet
(`docs/bench/2026-08-18-facade-s0-render-prefix-stability/`), the S1/S2/S3
gate records under `docs/bench/`, the successful real OpenCode session
`ses_fe307ea3effefOzYcDBgTbYiie` (2026-08-20, 5 h overnight / 58 requests /
133 k tokens), adversarial reviews
(`ses_fe8ee25c6ffe`), and the full investigation session (OpenCode Recall).

## Scope

Private, single-box engine serving local clients over loopback.

K2 Horizon dense 7B has separate raw-string and verified native chat/tools
`/v1/responses` subsets: see [K2 Horizon Raw Profile](#k2-horizon-raw-profile)
and [K2 Horizon Verified Chat](#k2-horizon-verified-chat). It does not inherit the
Qwen/DeepSeek chat, reasoning, tools, snapshots, or long-context capabilities.

Operating goals, in order:

1. **Foundations that hold** — durable continuity across restarts,
   honest behaviour when busy, and DeepSeek V4 that is actually usable.
2. **Concurrency / batch-serving responsiveness** (continuous batching).
3. **Long-context performance stability.**

## Program arc (decision record)

| Unit              | Contents                                                                                                                                                                                                                                                                     | Consumer                                    | Gate                                                                                                                                                                                                                                  |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0                | Prefix-stability falsifier                                                                                                                                                                                                                                                   | measurement                                 | DONE — see packet RESULTS                                                                                                                                                                                                             |
| S1                | Resident serial server, Open Responses subset, no tools                                                                                                                                                                                                                      | the game (thin HTTP client)                 | Functional evidence passed, but not the literal full gate: client was 187 lines versus `<150`, and the `<150 ms` TTFT gate failed at 186 ms. Conformance was 6/6 in scope and cancellation measured 24 ms — docs/bench/2026-08-18-serve-s1-gates/ |
| S2                | Tool items (XML-parameter form from the template oracle), exact declared/allowed-tool enforcement, continuation rendering                                                                                                                                                    | OpenCode via stock `@ai-sdk/open-responses` | Provider gate: 10/10 requests checkpoint-hit (94–100% restored), 5/5 turns tool-called. `allowed_tools` subsequently landed. Production evidence: real OpenCode session `ses_fe307ea3effefOzYcDBgTbYiie` ran successfully for 5 h overnight — docs/bench/2026-08-19-s2-agent-gate/ |
| S3                | Pre-opened (headless) reasoning support, DeepSeek V4 family backend, DFlash drafter integration. `encrypted_content` is **not** implemented.                                                                                                                                    | OpenCode, DS4 clients                       | Core cells 1–5 passed: warm snapshot hits, CLI byte identity including CJK/emoji, 91% reasoning replay restore, clean headless partition, fail-closed behavior. Live cells 6–9 still require rerun — docs/bench/2026-08-19-s3-ds4-gate/ |

## Remaining program (2026-08-20)

**F1 — Durable continuity.** Serve holds checkpoints in RAM, so a
restart drops the session; rebuilding the 133 k-token session from the
overnight run costs ~10 minutes of prefill. The engine ships
`DurableCheckpointStore`: private dir, flock'd single writer, blobs
named by compat-namespace + prefix length + digest, LRU to a byte
budget, corrupt-blob self-heal, atomic staged publish. Wire serve to it,
defaulting to a canonical location (`~/.cache/qwen-llm/…`, the path
`game/play.py` uses), enabled by default with the flag as override, and
log the resolved path and budget at startup.

**F2 — Per-family publication policy.** Publication cost is not uniform,
and this decides the policy:

| model | KV per token | snapshot at 133 k |
| --- | ---: | ---: |
| DeepSeek V4 | 6.9 KB | 0.94 GB |
| Qwen3.6 35B A3B | 20.5 KB | 2.7 GB |
| Qwen3.8 27B | 93 KB | 12.4 GB |

DeepSeek V4 can afford DwarfStar-style periodic writes during
generation plus a shutdown flush. A dense 27B cannot: 12 GB per turn is
unwritable, so Qwen families get **publish on graceful shutdown and idle
only** until content-addressed delta chains exist (S0 F6 — consecutive
snapshots share nearly all their bytes). F1 covers restart continuity; crash
resilience waits on those chains.

**F3 — Honest behaviour when busy: implemented, live rerun pending.** Serve
remains single-flight, but a separate acceptor now fails concurrent connections
fast with `503 Service Unavailable`, `Retry-After: 1`, and a `server_busy`
envelope instead of leaving them silent in the listen backlog.

**F4 — DeepSeek V4 tool support.** DS4 serve handles chat and reasoning;
tool definitions are rejected. Agent loops need them.

**F5 — Memory admission: implemented, live rerun pending.** Qwen requests are
priced before execution and fail with `503` when denied. Qwen and DS4 boundary
capture is memory-admitted and best-effort: denial or capture failure skips the
snapshot while the request continues. Both caches are byte-bounded; DS4 session
construction returns ownership on failure so residency remains recoverable.

**F6 — Concurrency / continuous batching.** The engine already provides
`admit_independent_queue2`, `create_dense_batch8_executor`,
`create_moe_batch16_executor`, and `concurrent_jsonl.rs` runs width-2
overlapped decode today. Route serve through them: wiring and
scheduling, with F3 as the on-ramp.

**F7 — Long-context performance stability.** Measured overnight:
decode fell 14.5 → 8.4 tok/s from 31 k → 133 k, and restore grew to
~1.5 s.

Also open: selector `emit_completion` telemetry (k3 R1.4), and fresh execution
of S3 live cells 6–9 (LRU eviction, DS4 cancellation/heartbeat, DS4 TTFT @8k,
memory envelope). Engine optimization candidates live in `docs/PERF-ROADMAP.md`.

## Parked, with reasons

- **WebSocket transport / connection-local continuation.** Optional in
  the spec. On loopback, posting 133 k of history costs ~16–23 ms of
  tokenization and ~1 ms of bandwidth against a ~1.5 s restore and tens
  of seconds of generation. Its value is skipping that restore by
  holding session state on the connection, which requires the server to
  stay up between turns.
- **`previous_response_id` stored mode** — same reasoning.
- **`encrypted_content` opaque reasoning** — the S2 capture showed plain
  reasoning content already replays verbatim through the stock provider,
  so this is hardening, not a requirement.
- **CC (chat-completions) shim, items npm provider, `/responses/compact`,
  installers** — no consumer on this box.
- **`tool_choice` `required` / `none` / forced-function** — unused.

## Current behaviour

One new subcommand:

```sh
qwen serve -m MODEL [--addr 127.0.0.1:8737] [--max-tokens N] \
  [--max-context-tokens N] [--snapshot-cache-mib auto|MIB] [--drafter GGUF] \
  [--snapshot-idle-ttl-secs 3600] [--snapshot-max-age-secs 86400] \
  [--snapshot-half-life-secs 600]
# Qwen: without --max-context-tokens the admission ceiling is the smaller of the
# 262,144 hard default and the GGUF's declared context length; --drafter is
# accepted for dense targets only (an MoE target fails startup rather than
# silently running serially)
# DeepSeek V4 additionally requires --max-context-tokens (startup-fixed forward budget)
# Muse Glimmer and Qwen3.8-Flash-Next require both --max-context-tokens and
# --max-tokens (resident session capacity is fixed at load); admitted capacity
# may extend through the model's declared context.
# K2 requires explicit --max-context-tokens (within checkpoint context) and
# --max-tokens; --snapshot-cache-mib must be auto (resolves to 0) or 0.
```

**Snapshot cache policy (Qwen and DS4).** Both RAM caches share one policy
(`qwen_llm::snapshot_policy`), logged on the `serve limits:` line:

- **Budget.** `auto` (default) = min(25% of physical RAM, 50% of the Metal
  recommended working set left after the model loads), at least 1 GiB; an
  integer is MiB, taken verbatim (0 disables capture).
- **Eviction** when over budget removes the entry with the lowest
  `score * reusable_tokens / bytes`, where a hit adds 1 to a score that decays
  with `--snapshot-half-life-secs` (ties: older access). `0` is exact LRU.
- **Expiry.** Entries unused for `--snapshot-idle-ttl-secs` or older than
  `--snapshot-max-age-secs` are dropped (0 disables each), lazily on lookup and
  insert and from the idle admission loop, so memory is returned while idle.
- **Pins.** The request's transcript entry is pinned while its completed
  boundary is inserted; pinned bytes count against the budget, so a completed
  boundary that would need to evict it is skipped.
- **Memory pressure.** A capture denied for process-footprint headroom evicts
  by the same rank for the deficit and re-checks once. Metal-headroom denials
  are not retried: snapshots are CPU arenas, invisible to the Metal signal.

The listener rejects every resolved non-loopback address and is bound before
the model loads, so an unresolvable or busy address fails startup immediately;
connections that arrive during the load wait in the backlog and are answered
once the accept loop starts. Request bodies are
limited to 16 MiB. The whole request read has a 30 s absolute deadline; the
socket read timeout is 35 s so it cannot preempt that mapping, and writes have a
30 s timeout. For an optional trace,
prepare a private directory rather than using a predictable shared `/tmp` name:

```sh
trace_dir="${XDG_CACHE_HOME:-$HOME/.cache}/qwen-llm/traces"
install -d -m 700 "$trace_dir"
qwen serve -m MODEL --trace-sse "$trace_dir/serve-$(date +%Y%m%d-%H%M%S).jsonl"
```

- **Residency:** model loads once; the process is the warm tier. S0's F2
  finding makes this the TTFT mechanism (per-process paging floor is
  5–8 s on A3B even with checkpoint hits; `load_ms` does not cover
  first-touch). **Current serve is RAM-cache-only**: durable checkpoint
  publication/restore is not wired into serve, so cross-restart warmth
  currently re-prefills (closing k3 review, D3).
  Durable checkpoints remain the intended cross-restart substrate. The finite
  default Qwen context ceiling is 262,144 tokens; an omitted limit sizes each
  request to need without making the ceiling unbounded. Muse keeps one fixed
  resident session and resets it between requests by default; its separate
  live-prefix opt-in reports reused tokens without snapshot restores.
  Its synthesized system prompt is stamped with the current
  UTC date for each request; when callers provide `instructions` or a system
  item, that explicit system text owns any date policy instead.
- **Speculative decode (`--drafter`, v0.77 DFlash):** a request
  speculates after a cold prefill or when a restored RAM-cache entry carries a
  compatible target-hidden capture tail. Missing or malformed tails fall back
  to serial decode while refreshing the tail for the next turn. Greedy requests
  use target-verified accept-prefix;
  sampled DFlash2 requests sample the selector's sparse top-16 distribution
  and use maximal coupling against the packed target distribution. The packed
  forward is numerically close to, but not bit-identical with, serial
  token-major arithmetic. The per-request
  `serve phases:` line reports `decode_path=dflash|serial`, with a
  `serve dflash:` line carrying acceptance and backoff counters.
  `QWEN_DFLASH_PREFIX_REPLAY=1` additionally enables a default-off,
  single-process experiment that verifies repeated exact-prompt completions
  before falling back to DFlash; sampled use requires two distinct-seed outputs
  with a 32-token consensus prefix. It is not a multi-tenant cache contract.
  `QWEN_DFLASH_OFF_CTX=<tokens>` overrides the default 16,384-token hard stop
  for explicit long-context canaries. Above the default boundary, admission
  backs off on low acceptance or dense exact fallback and prices recovery probes
  conservatively; the override is not a default-on long-context promotion.
  DFlash starts only after the required full-prompt or trailing SWA window has
  been captured and the target sequence is at prompt length. The old 25-token performance run is
  retracted: its short-prompt serial-tail path did not seed prompt hiddens, so it
  cannot support a DFlash performance or output-equivalence claim. The corrected
  path has since passed scoped GPU validation: release 27B prefill and
  packed-verify gates passed (24-token final-logit cosine 1.0, minimum hidden
  cosine 0.999998, and packed verify 16/16 argmax with minimum cosine 1.0), and a
  live cold 19-token Qwen3.8-27B Q8_0 + DFlash2 Q8_0 request logged
  `decode_path=dflash` and matched serial output (`orange`, EOS after two
  generated tokens). This is correctness evidence for that cell, not a revived
  performance claim or blanket output-equivalence claim. Optional DFlash
  admission prices prompt capture, drafter session, verify scratch, and
  layer-major scratch; allocation, capture, or seeding failure restarts or falls
  back to serial generation rather than rejecting an otherwise viable request.
- **Determinism scope (F7):** no serve surface promises temp-0 byte
  identity across differing checkpoint-restore topologies; transcripts
  are byte-deterministic conditional on restore partitioning.
- **Stats contracts (frozen):** the stderr `qwen_diag` line carries
  `version=serve_stats_v1`; the `x_qwen` response echo's field names
  `matched_tokens`, `restore_ms`, `prompt_tokens` are frozen;
  `usage.input_tokens_details.cached_tokens` reports tokens restored
  from checkpoints.
- **Serial generation, fail-fast admission.** `std::net`, one request in
  flight; the acceptor rejects other connections immediately with `503` and
  `Retry-After: 1`. HTTP/1.1 with
  `Connection: close`; hand-rolled request parse (loopback threat model;
  request bodies are `Content-Length` JSON).
- **Implemented families: Qwen3.5/3.6/3.8, Qwen3.8-Flash-Next, DeepSeek
  V4, Muse Glimmer, and K2 Horizon raw/verified native chat/tools.** DS4 runs its own session and snapshot stack
  (`serve/backend_ds4.rs`) with a startup-fixed forward budget and a
  serve-owned snapshot cache (`serve/snapshot_cache.rs`; DS4 has no
  engine-side RAM prefix cache). Flash-Next (`serve/backend_qwen4exp.rs`, since 2026-09-17) holds
  one text-session workspace sized at load and hands it back reset after
  every request: the Qwen3.8 contract (effort levels, thinking, tools), no
  prefix reuse, no snapshots. `--snapshot-cache-mib` configures the Qwen and
  DS4 caches (default `auto`; see the cache policy above). Muse does not claim
  snapshot reuse yet.
  K2 uses fresh per-request sessions, has a zero snapshot budget, and has
  no prefix reuse, drafter, or tool protocol. Its verified chat profile separates
  reasoning and final answers without inheriting another family's parser.
- **Stdout is never written.** All diagnostics via the existing stderr
  tracing surface; per-request `qwen_diag` stats line retained and
  extended with `matched_tokens` and `restore_ms` (the S2/S3 gates are
  defined on this log, never on "the session completed").
- **Opt-in asynchronous SSE trace:** `--trace-sse PATH` appends one JSON object per line
  for each `/v1/responses` request, plus each streamed response's heartbeat,
  event (including its exact JSON `data` payload), and terminal `[DONE]` marker.
  It is disabled by default and may contain prompts, tool definitions, and
  generated text. The file is opened append-only with `O_NOFOLLOW|O_CLOEXEC`,
  using nonblocking open so a FIFO cannot hang startup; it must be a regular file
  owned by the current user and is forced to mode 0600.
  A bounded background-writer queue keeps trace I/O off the response path; a
  full queue or writer failure disables tracing rather than blocking serving.
  Shutdown gives the writer 250 ms to drain, then detaches it so a stalled
  filesystem cannot hold process exit; queued trace events may be lost in that
  case.

## K2 Horizon Raw Profile

K2 dense 7B supports completion-style raw text on `POST /v1/responses`, not a
new `/v1/completions` endpoint. Choose an unused loopback address:

The verified final artifact also supports [HTTP chat](#k2-horizon-verified-chat)
and local `qwen run --user/--messages`. Raw strings never opt into templating:
their EOS-only stop set and literal output contract below are unchanged.

```sh
qwen serve -m "$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  --addr 127.0.0.1:8795 --max-context-tokens 256 --max-tokens 8 \
  --snapshot-cache-mib 0
```

Both limits must be explicit for resident memory planning and the response default.
Capacity must be positive and fit the checkpoint's declared context and actual
device/memory admission; the default output limit is 1..=capacity. There is no
K2-specific 256-forward/response or 7168-context cap. Nonzero snapshot
budgets and drafters fail startup. K2 config, runtime storage plan, native
tokenizer, and EOS metadata are checked before listener binding or Metal setup.
The normal production lease and memory gate apply; listener binding still
precedes weight loading so busy addresses fail cheaply.

```json
{
  "model": "K2-Horizon-7B-Q8_0",
  "input": "The capital of France is",
  "max_output_tokens": 8,
  "stream": false
}
```

The model ID is the filename stem, as returned by `GET /v1/models`. Input must be
a nonempty string. The family parser retains it explicitly and applies no chat
template; native tokenizer NFC normalization and special-token recognition still
apply. Native BOS insertion defaults on. For already serialized input, use
`"x_k2":{"add_special_tokens":false}`; there is no authored-token deduplication.
For example, that option with input
`"<|ifm|begin_of_text|>The capital of France is"` matches the plain example's IDs.

Accepted fields are `model`, `input`, `stream`, `max_output_tokens`, `temperature`,
`top_p`, `store`, `truncation`, `x_qwen`, and `x_k2` only:

- Omitted `max_output_tokens` uses the explicit startup default. The full
  `prompt_tokens + max_output_tokens - 1` budget must fit capacity; no truncation.
- Sampling defaults are temperature 0, top-p 1, top-k 0, min-p 0, seed 0.
  `x_qwen` accepts only `seed`, `top_k`, `min_p`, and `stats`.
- `x_k2` accepts only boolean `add_special_tokens`; other families reject that
  extension. If supplied, `store` must be false and `truncation` must be `"disabled"`.
- In raw mode, message/item arrays, token-ID arrays, instructions, tools, reasoning/history,
  previous-response controls, and unknown fields fail, including null-valued
  unsupported fields. Accepted fields cannot be null either. The whitelist uses
  the existing parsed JSON representation; it adds no duplicate-key guarantee.

EOS 1 is the only stopping token: sampled/countable, but never emitted or forwarded.
The final non-EOS budget token is emitted without an extra forward. JSON and SSE
both emit literal protocol text: think/tool/IFM marker strings are not stripped,
partitioned, or executed. Incremental UTF-8 assembly preserves split characters;
invalid bytes or a final incomplete sequence use replacement characters. This is
not a byte-preserving binary HTTP format. Budget exhaustion uses the ordinary
incomplete-response terminal semantics, not an EOS success.
An over-budget nonstream request returns HTTP 400. If SSE headers have already
been sent before backend tokenization, the same refusal is a `response.failed`
event after HTTP 200, with no generated text or request-session allocation.

The model remains resident, but each request owns fresh KV and a fresh sampler.
There are no snapshots, resets, prefix reuse, or hidden conversation history.
Synchronous per-token prefill appends provide real cancellation boundaries; abort
drops the entire session, so the next request cannot inherit partial KV. The
borrowing backend stays on the accept-loop thread without self-referential/leaked
model storage. `x_qwen.stats` is opt-in with cached/matched tokens always zero;
the response echo has no tools/reasoning and disables parallel tool calls.

Validation covers CPU wire/UTF-8/EOS controls and actual-Q8 borrowed-backend
JSON/SSE, BOS, abort isolation, and requested-capacity boundaries on ephemeral sockets.
Admission bounds are distinct from the lengths covered by numerical fixtures.
Numerical evidence covers the pinned final Q8_0-weight checkpoint with F16 KV on
M4 Max, including the frozen v2 holdout; it does not qualify F16 weights or every
compatible intermediate checkpoint. Serial online attention retains the full
history without a context-sized score buffer. Serving deliberately keeps singleton
appends for per-token cancellation, even when local run/bench/lens use eligible
32-row Q8 packed prefill. Beyond 256 visible positions, attention merges fixed-size
online summaries in registers to reduce accumulation error without dropping history.
Stored F16 KV is unchanged; compact KV is not enabled. At 524288 positions the
logical cache alone is 72 GiB; declared context does not promise that it fits a
particular device or is economical to run.
This is not sustained-service, full-context, tool, or cross-checkpoint
numerical qualification. Test reproduction is in
[`scripts/reference/k2/README.md`](../scripts/reference/k2/README.md); development
evidence remains in `K2-HORIZON-PLAN.md` and `K2-HORIZON-REVIEW.md`.

## K2 Horizon Verified Chat

For verified final Q8_0 and standard Q4_K_M artifacts, array `input` selects the pinned IFM
renderer; string `input` always remains raw. Startup verifies retained checkpoint
bytes once, before accepting requests, and owns an opaque chat capability. An
unverified artifact or verification failure emits a diagnostic and remains raw-only.
No request field can authorize chat or trigger rehashing. Profile details and
provenance limits are documented in `CLI-UX.md#k2-horizon-verified-chat`.

```json
{
  "model": "K2-Horizon-7B-Q8_0",
  "input": [{"role":"user","content":"What is 2+2? Answer briefly."}],
  "reasoning": {"effort":"low"},
  "max_output_tokens": 128,
  "stream": true
}
```

Use adequate server capacity and output defaults for reasoning workloads. The raw
profile's residency, sampling, cancellation, and no-truncation policies also apply.
Chat additionally accepts string `instructions` and `reasoning` containing only
`effort` (`high` default, `medium`, or `low`). Native BOS is mandatory; false
`x_k2.add_special_tokens` is rejected. Chat stops at EOS 1 or `im_end` 250019.

History permits one leading system (or `instructions`, not both), user and
assistant messages, and must end in a user turn. Message content is a string or
`input_text`/`output_text` parts matching the role. Every assistant needs a directly
preceding reasoning item with string content or `reasoning_text` parts; empty
reasoning is valid. Generic replayed reasoning uses IFM's canonical `reasoning`
alias (high/base history tag); the request effort controls only the new suffix.
Completed response items can be replayed verbatim, including string IDs, completed
status, empty reasoning `summary`, and empty output-text `annotations`. Nonempty
summaries/annotations and incomplete history are rejected rather than discarded.
Developer roles, multimodal data, unsupported fields and null controls are rejected
before transcript normalization. Nonempty tools select the extension below. K2
wire JSON rejects duplicate keys and nesting beyond 128; typed containers and
number lexemes are preserved, including Serde-internal-looking object keys. These
are parser safety rules, not inference-context limits; other families are unchanged.

JSON and SSE separate `reasoning` and final `message` items. Generation begins in
preopened reasoning; the first exact `</ifm|think>`, `</ifm|think_fast>` or
`</ifm|think_faster>` switches to final text. Effort is the requested prompt control,
not a promise that the emitted close matches it. An optional matching opener at
byte zero is removed. Tool-looking, Qwen and later IFM markers remain literal in
ordinary no-tools chat. Empty reasoning still emits a completed
reasoning item so history can be replayed. Budget exhaustion inside reasoning is
incomplete, never an answer; EOS before its close is a protocol failure. Abort
discards ambiguous buffered delimiter/UTF-8 bytes and never synthesizes completion.
Unlike vLLM's K2 parser, this no-tools subset does not use tool-call markers as
fallback boundaries or reinterpret unterminated reasoning as visible text.

Leased actual-Q8 and Q4_K_M checks cover all efforts, short incomplete reasoning, completed
low-effort answers, stop-aware raw-prefix parity, JSON/SSE equality, CLI answer
parity, and fresh sessions after cancellation. These are wiring/termination checks,
not reasoning-quality, sustained-service, tool, or full-context qualification.
Total output-token usage includes stop tokens; the existing shared
`reasoning_tokens: 0` detail is not a measured per-channel token count.

## K2 Horizon Tools

Verified final Q8_0/Q4_K_M chat accepts standard flat function definitions in
`tools`, `function_call` history with JSON-string `arguments`, and matching
`function_call_output` items. Tool outputs may be strings, objects or arrays;
objects/arrays render as native JSON data, not multimodal content. Each assistant
call group needs preceding reasoning (empty is valid), and every pending ID needs
exactly one result. Results may arrive out of order and render in original call
order. IDs remain wire metadata, not tokens inserted into the IFM prompt.

```json
{
  "model": "K2-Horizon-7B-Q4_K_M",
  "input": [{"role":"user","content":"Call lookup_code with key orbital."}],
  "tools": [{"type":"function","name":"lookup_code","parameters":{
    "type":"object","properties":{"key":{"type":"string"}},"required":["key"]
  }}],
  "reasoning": {"effort":"low"},
  "x_k2": {"tool_call_format":"json"},
  "max_output_tokens": 512
}
```

`x_k2.tool_presentation_format` accepts `markdown` (default), `xml`, or `json`;
`x_k2.tool_call_format` accepts `xml` (default), `json`, or `xml_typed`. The response
echo records these requested formats. Only omitted/`auto` tool choice and
omitted/`true` parallel calls are supported: the native template has no forced,
disabled, named or narrowed choice control. `strict: true` is rejected; schema
presentation is not constrained generation or complete argument validation.
Empty tools normalize to ordinary chat without tool-format echoes or marker parsing.

JSON and SSE publish completed `function_call` items with stable per-response call
IDs. Append the response's `output` items and caller-owned results to `input` for
continuation. qwen never runs the tool. Calls publish only after a whole terminal
block is valid; incomplete or malformed blocks never leak partial calls. A closed
valid call at the output budget may publish while the overall response stays
`incomplete`; EOS before closure fails. Cancellation drops pending calls.

XML is IFM's verbatim tagged text, not entity-escaped XML. Types must be unambiguous
under the definitions; use JSON for ambiguous values. Requested typed XML may
accept a fully valid untyped XML block when the model omits labels, but never
contradictory labels, mixed malformed structure, trailing text or ambiguous types.
The strict library dialect parser remains strict. No arbitrary token cap or
heuristic reasoning closure is introduced; buffering uses a checked byte bound
derived from output budget and native tokenizer piece size.

Actual Q4 checks exercise call -> caller-supplied result -> final answer in all
three requested formats through CLI, HTTP JSON and SSE. They demonstrate product
wiring, not universal tool reliability, schema compliance or answer quality.
Reproduction: `scripts/reference/k2/README.md`.

## Wire subset (Open Responses)

This section describes the Qwen/DeepSeek/Muse chat profiles; K2's narrower raw and
verified-chat contracts are specified separately above.

The parser, Qwen capability binding, and prompt renderer are one shared pure
module. `qwen-lens --open-responses FILE|-` uses that same path for offline
readout/intervention prompts and records renderer-authored role, reasoning, and
tool-channel spans. Lens does not emulate HTTP routing or response filtering:
its `--model` and sampler flags remain authoritative, so request sampling fields
and narrowed `allowed_tools` fail closed there.

`POST /v1/responses` accepting. For Qwen and DS4, omitted
`max_output_tokens` defaults to 65536 unless overridden with `--max-tokens` at
startup. Muse requires an explicit startup default:

- Supported non-null fields are type-checked strictly. Known standard controls
  outside this subset (including `max_tool_calls`, `text`, `metadata`, and
  `stream_options`) fail closed rather than being silently ignored. Other
  unknown fields are ignored; request, `reasoning`, and `x_qwen` scopes log each
  unknown name once, while input-item and tool-object extras are silently
  normalized away. Unknown item/content types fail closed.
- `model` — must equal the loaded model id; else `model_not_found`.
- `input` — string (one user message) or item array in the subset:
  `message` (roles `system`|`developer` (system-equivalent, documented
  mapping)|`user`|`assistant`, content string or `input_text` parts),
  `reasoning` (plain `content`; `encrypted_content` is unsupported).
  **Item-sequence validation,
  not turn grammar** (review defect 1: the spec's `input` is an item
  list; stock AI SDK traffic legally contains consecutive user
  messages and, in S2, interleaved tool items): system/developer/
  `instructions` at head only, `reasoning` items must immediately
  precede their assistant message or function call, and the final item must be
  a `user` message or `function_call_output`. Unknown item _types_ →
  `invalid_request`; unknown _fields_ are
  ignored (top-level request fields logged once per name; `id`/`status`
  on replayed input items accepted and ignored).
- `instructions` — optional system text (exclusive with a system item).
- `max_output_tokens`, `temperature`, `top_p` — standard.
- `reasoning.effort` — validated Qwen3.8 identities accept `none`, `low`,
  `medium`, and `xhigh`; absent defaults to `xhigh`. It is rejected on generic
  Qwen identities. DS4 applies its separate renderer rules. Muse accepts
  `low`, `medium`, `high`, and `xhigh`, defaulting to `high`.
- `x_qwen` extension object — `seed`, `top_k`, and `min_p` are generation
  controls for every served family. `no_thinking` renders the released
  preclosed suffix on any identified Qwen release (Qwen3.5/3.6/3.8), the same
  rule as `qwen run --no-thinking`; `thinking: true` requests the released
  `<think>\n` opener on templates whose default is no-thinking (Qwen3.5) and
  is a no-op where thinking is already the default. DS4 uses
  `reasoning.effort` instead, while Muse directs callers to effort `low`.
  This is a documented implementor extension.
- `stream` — SSE when true, single JSON response otherwise.
- `store` — `false`, `null`, or absent; `true` → `invalid_request`.
- `previous_response_id` — → error code `previous_response_not_found`.
- `truncation` — only `"disabled"` (default). The engine already fails
  closed on context overflow (S0 F3); serve maps that to the spec error
  instead of a process exit.
- `tools` — uniquely named function tools are supported on identified Qwen
  releases (Qwen3.5/3.6/3.8), DeepSeek V4 (DSML), and Muse Glimmer's ATEM
  protocol; known definition fields have strict types, and `strict:true` is
  rejected because schema enforcement is unsupported. Hosted tool types fail
  closed. Definitions render into the selected family tool block, byte-pinned
  to its renderer contract. An unidentified Qwen release refuses tools and
  replayed tool turns with code `tools_require_known_release` — the same
  family rule as `qwen run --messages`, advertised by `qwen info --json`
  under `capabilities.input.tools`.
  Function names must match `[A-Za-z0-9_.-]{1,64}` (the grammar `qwen run
  --messages` admits); replay `call_id` values are limited to 64 bytes.
- `tool_choice` — `"auto"` (default) or an `allowed_tools` object.
  `allowed_tools` accepts only mode `"auto"`, requires a non-empty unique list
  of declared function names, and defines the exact executable set. Narrowing
  is enforced as a hard constraint on emitted calls while
  leaving rendered bytes identical, so prompt prefixes and their
  checkpoints stay valid across tool-menu changes (the spec's
  cache-preserving intent, test-pinned). Enforcement is post-generation:
  a suppressed call has already consumed tokens and remains in the
  completed-turn checkpoint key, so that continuation will not hit.
  `required`, `none`, and forced-function are not implemented.
- Replayed `function_call_output` items must link by unique `call_id` to every
  call in the immediately preceding call batch; unknown, duplicate, or missing
  links fail closed. Argument delta/done events carry that same `call_id`.
- `/responses/compact` — not implemented (404); compaction is outside the
  S1–S4 arc and revisits with the WebSocket transport question.
- `x_qwen.stats: true` — echoes `{matched_tokens, restore_ms,
prompt_tokens}` into the response object, so thin clients (the game)
  get per-request checkpoint stats without correlating server stderr
  (review R4). `usage` is always populated (agent clients budget on
  it).

Provenance notes: `top_k`/`min_p` pass through normal sampling semantics;
`no_thinking` is identity-gated to the validated `qwen run` surface;
`stream: false` exists for the stock provider's `doGenerate` path in S2,
not for S1's consumer.

Output items: `reasoning` (when the model emits thinking) then `message` with
`output_text`, followed by typed function calls when present. Qwen/DS4 use the
`</think>` seam; Muse parses exact `to=self`, `to=user`, and ATEM recipient
segments from generated bytes. Family parsers own all model syntax, so ATEM or
Qwen-looking text in another family's visible response cannot become a call.
Muse rejects undeclared recipients, malformed controls, invalid UTF-8, and tool
values that cannot be rendered back into ATEM history without changing
structure; partial controls and calls are discarded on truncation or abort.
Truncated thinking (S0 F4) yields `reasoning` with `status:
"incomplete"` and `response.status = "incomplete"` with
`incomplete_details.reason = "max_output_tokens"`.

Response envelopes truthfully echo normalized, validated `instructions`, `tools`,
`tool_choice`, `reasoning`, `parallel_tool_calls`, sampling values, and output
limit rather than emitting fixed placeholders.

Streaming events: `response.created`, `response.in_progress`,
`response.output_item.added`, `response.content_part.added`,
`response.reasoning.delta|done` (the gate-5 contract), plus one
`response.reasoning_text.delta` compatibility alias per reasoning delta for the
stock AI SDK provider, `response.output_text.delta|done`,
`response.content_part.done`, `response.output_item.done`,
`response.completed|incomplete|failed`, terminal `[DONE]`. Tool turns
add `function_call` output items with
`response.function_call_arguments.delta|done` (S2). **Every
event carries a monotonic `sequence_number`** (review defect 4 — the
conformance suite asserts ordering; retrofitting into a hand-rolled SSE
writer later costs more). One SSE comment heartbeat (`: ping`) is written
immediately, then idle heartbeats are attempted between prefill chunks; this is
not a `>=1 Hz` guarantee because a chunk may run longer than one second.
Parse, validation, model-id, and render failures occur before SSE and return an
HTTP error. Backend failures after SSE begins emit `response.failed` with the
spec error envelope; disconnect/write failures cannot emit a terminal event.

`GET /v1/models` returns the single loaded model (trivial, ships in S1
because client model-pickers probe it).

## Render and checkpoint contract

- Items→tokens rendering reuses the existing family renderers
  (messages.rs); the S1 tree adds golden byte fixtures asserting
  prefix-stability across turn append, including the future
  tool-continuation shape (designed now, wired in S2), and documenting
  the intentional divergence points per family.
- Reasoning items render into the think block for the preserve-mode
  families; assistant `message` content containing inline `<think>` is
  rejected (`invalid_request`) — reasoning travels as items, never
  inline (F1: verbatim echo is what makes preserve reuse exact, and
  items make echo structural).
- Qwen serve snapshots the **transcript boundary**: the last `<|im_start|>`,
  where the generation header begins. Qwen templates re-render a prior
  assistant turn differently from the generation suffix the model consumed
  (`<think>\n` versus a preclosed `<think>\n\n</think>`, or reasoning dropped by
  the client), so prompt-end and completed snapshots stop prefixing the next
  request one token after `<think>`, and a hybrid's recurrent state cannot be
  rewound to recover. Prefill stops at the boundary, captures, then finishes
  the header. Without it, thinking-mode chat and tool loops whose client drops
  reasoning reused 0 tokens on every turn. A captured transcript boundary
  replaces the prompt-end capture (an exact resend re-prefills only the header),
  and the transcript entry is pinned while the completed boundary is inserted
  (skipped if it cannot fit beside it). Drafter requests keep single-pass prefill and
  the prompt/completed pair. DS4 captures prompt and completed boundaries and
  skips a completed boundary with no transition or a truncation inside open
  reasoning. Eligible boundaries enter the **RAM** prefix cache (8–42 ms each
  per S0). Evidence: PERF-LOG 2026-09-23 transcript-boundary entry.
  Capture is best-effort and admitted against cache bytes plus Metal/process
  headroom; denial or failure is logged and generation continues. Both family
  caches enforce the configured byte budget.
  Validated Qwen3.8/Q8 dense-27 requests without a drafter can prefill a restored
  7-32-token suffix in one packed block when the current request is greedy and
  the optional scratch price fits 128 MiB. Admission/allocation failure retains
  serial prefill. Other profiles and fresh/exact hits retain their existing
  policy. Packed-vs-serial state is numerical, not bitwise; sampled followups
  can inherit these checkpoints, so this carries no distributional-equivalence
  claim. Measured scope: `docs/bench/2026-09-06-single-chunk-vt/RESULT.md`.
  `QWEN_SERVE_FRESH_PACKED=1` additionally opts validated Qwen3.8/Q8 dense-27
  **fresh cache misses** of 19-48 prompt tokens into bounded packed prefill,
  greedy requests without a drafter only. Unset, `0` and invalid values disable
  this fresh path; Q4 is unsupported even with `1`. The same 128 MiB scratch cap
  and serial fallback apply. Q8 passes balanced endpoint gates, but automatic
  enablement remains held after an unpaired first-request outlier under substantial
  global compression. No first-request guarantee or sampled-distribution claim.
  Evidence: `docs/bench/2026-09-07-fresh-serving-http/RESULT.md`.
  Muse defaults to fresh packed prefill in superchunks of up to128 tokens
  with a16-token packing quantum, followed by a scalar remainder. Set
  `QWEN_MUSE_PREFIX_REUSE=1` to reuse the exact consumed-token prefix of the last
   completed backend generation in its resident session, without snapshot copies. It always
  recomputes at least the final prompt row, reports only actually reused tokens
  as cached/matched, and keeps `restore_ms=0`. Capacity rejection preserves the
   prior history; a detected generation abort clears reuse history. Publication
   precedes final HTTP framing, so a late transport/framing failure may retain
   the completed backend history. GPU poison remains
  fail-stop. This is one serial resident history, not durable or cross-process
  caching; it does not accelerate fresh prompts or per-token decode. Evidence:
   `docs/bench/2026-09-08-muse-live-prefix/RESULT.md`.
   Muse Q8_0 on unified Apple M4 Max defaults to optimized matrix prefill and
   split decode. `QWEN_SERVE_MUSE_MATRIX_PREFILL=0` and
   `QWEN_SERVE_MUSE_SPLIT_DECODE=0` independently roll back to original math.
   Each accepts only unset/`0`/`1`, is resolved once at startup, and keeps the
   existing model-context/capacity and device admission checks. Unset or `1`
   permits only qualified execution; BF16 and other devices retain original math.
   Matrix prefill uses tiled full128 chunks and online packed remainders; scalar
   tails remain. Split decode admits528 KiB scratch and requires1024 visible KV
   positions. CLI-only math variables remain independent. Native ATEM, sampling,
   reasoning and terminal-token accounting do not change. Token-prefix identity
   is exact, but changed chunk boundaries and generated-versus-prompt history
   make optimized warm/reset arithmetic numerical, not bitwise or sampled-exact.
   Diagnostics report resolved options, planned tiled/online token counts, and
   actual generation transitions; planned counts are not dispatch measurements.
  Durable publication keeps the existing completed-else-prompt shadowing
  policy. **Durable publication remains parked**: current serve is RAM-only,
  so cross-restart warmth still re-prefills. `--durable-dual-publish` is
  designed but unbuilt (review R2: at q38's ~90 KB/token, dual durable publish
  of a 32k context is ~6 GB/turn — LRU churn that evicts the prefixes it
  is meant to protect, and publish time serializes the next request on a
  serial server). Live cache-pressure behavior remains part of the S3 rerun.
- **Replay-fidelity coverage (review R6):** render→items→render byte
  identity is asserted by unit tests (`render.rs` split/render inverse,
  `tool_parse.rs` `qwen36_raw_echo_identity`, `render_ds4.rs` preserved
  history round-trip) rather than by a JSON fixture case.
- Tool-continuation golden fixtures are **written before the renderer**
  (review R3), so the renderer is fit to the fixture, never the reverse.
- Release identity: the renderer contract follows the Qwen release
  (3.5/3.6/3.8) identified from the GGUF's name metadata (`general.name`,
  `basename`, `base_model.0.name`, `base_model.0.repo_url`, `license.link`)
  plus the Qwen3.x tokenizer gate (`gpt2`/`qwen35`/248320 tokens);
  `qwen4exp` architecture implies the Qwen3.8 contract; `deepseek4` implies
  the V4-Flash-0731 encoder contract. The GGUF's `tokenizer.chat_template`
  is never consulted: per supported release there is one correct contract
  — the full-capability one (tools, thinking, effort where the release has
  them) — and embedded-template variance across repacks reflects
  simplification, not intent (local DS4 repacks embed three different
  templates: Unsloth-patched 0731, upstream 0731, and the pre-0731
  original). Until 2026-09-17 the digest selected the release and refused
  unknown digests, which rejected derivatives whose template differed by a
  no-op. For an identified
  release the renderer follows the released Jinja byte for byte (oracle fixture
  `tests/fixtures/qwen36_chat_template_oracle_v1.json`): content is trimmed,
  the generation suffix is always `<think>\n` or the preclosed block, and
  preserved reasoning replays as `<think>\n{reasoning}\n</think>\n\n{content}`.
  Qwen3.6 thinks unless `x_qwen.no_thinking`; Qwen3.5's released default is
  no-thinking (its template has no `preserve_thinking` at all, so preserve is
  a serve policy there). Documented divergences: history assistant turns keep
  the preclosed block in a no-thinking session (which is what the released
  template itself renders for preserved empty reasoning), a second system
  item is rejected rather than merged, and the template's
  `last_query_index` rule (reasoning kept only for assistant turns after the
  final user query) is not applied until tool continuation lands. Trimming
  follows Python `str.strip()`. An unidentified release (no version token in
  any name field, conflicting versions, or a foreign tokenizer) keeps the
  legacy generic ChatML contract (bare suffix, verbatim content) rather than
  guessing, logs a startup warning, and reports `capabilities.template`
  `{status: unknown, reason, fields_consulted}` in `qwen info --json`.
  `qwen run`, `qwen-lens`, `qwen-bench`, and `qwen-census` render through the
  same code.
- Tools on pinned templates render the way every released client feeds
  `tool | tojson`: the OpenAI-shaped `{"type": "function", "function": {...}}`
  object with Python's `", "`/`": "` separators (Transformers and llama.cpp
  agree), numbers in Python float repr (`10000000000.0`, `1e-07`), and
  replayed call arguments follow each template's own value rule: Qwen3.6
  passes non-container scalars through Jinja `string` (`True`/`None`),
  Qwen3.5 and Qwen3.8 pass every non-string through `tojson` (`true`/`null`).
  Every rule is pinned by the per-template jinja2 oracle fixtures. The released `last_query_index` rule applies:
  assistant turns after the final user query keep their think block (empty
  if no reasoning) even under strip. An unidentified release refuses tools
  (`tools_require_known_release`); the compact flat form
  `serve_tool_render_fixtures_v1.json` once froze for it was serve-invented
  and is superseded (2026-09-07). Function names on every lane match
  `[A-Za-z0-9_.-]{1,64}` — dotted names are released-protocol shapes
  (Qwen3.6 oracle `fs.list`, Muse ATEM namespaces).
- Preserve/strip rendering policy: preserve is the default for the validated
  Qwen3.6 identity (owner position, Amendment 1 of S0; economics measured in S0
  G2), and strip remains available there via `x_qwen`. Validated Qwen3.8 uses
  its preclosed-history renderer. DS4 rejects strip mode and preserves reasoning
  history whenever the current request selects a non-`none` thinking tier.
  Muse rejects strip mode and preserves structured ATEM reasoning/tool history.

## Cancellation

SIGINT/SIGTERM is checked around expensive startup phases, before admission,
while idle on a 100 ms admission poll, and at generation sink/chunk checkpoints.
The dedicated acceptor uses nonblocking accept with a 50 ms poll and is joined
during unwind, so a signal pending before the accept loop or arriving while idle
does not require a connection to wake the listener. Active-request shutdown is
bounded by the next checkpoint rather than immediate preemption.

For streaming requests, a disconnect is observed on an SSE write or a prefill
chunk heartbeat. Byte-oriented reasoning/tool partitioning can buffer token
boundaries that do not produce a write, so cancellation is write/chunk bounded,
not universally token- or time-bounded. The S1 cell measured 24 ms to the
following admission in its tested mid-decode case, but serve makes no universal
`<250 ms` claim. A non-stream response has no required write before completion;
graceful disconnect detection before that first write is inherently best-effort.

## S1 gate definitions (executed; see gate record)

1. **Thin client:** play.py rewritten against serve (direct HTTP + SSE,
   `x_qwen.seed`) at <150 lines. Artifact-named check: save-file bytes
   byte-identical to the CLI harness on a scripted 5-turn session plus
   one fork-by-truncation, temp 0, same seed; stderr explicitly outside
   the comparison.
2. **TTFT:** warm (resident, RAM-cache) turn-2 TTFT <150 ms at 8k
   context on Qwen3.6-35B-A3B; measured from request byte one to the
   first byte of the first `response.reasoning.delta` or
   `response.output_text.delta` event. Comments and lifecycle events
   excluded (review defect 5: heartbeats are SSE bytes; `response.created`
   fires at admission and measures nothing).
3. **Heartbeat:** first `: ping` within 1 s of request admission on a
   cold-prefill request (the client-timeout defense, gated separately).
4. **Prefix-stability goldens:** render fixtures pass — turn-append
   stability per family with documented divergence points,
   tool-continuation shape (fixture frozen before renderer), and the
   render→items→render replay-fidelity identity.
5. **Conformance:** Open Responses acceptance tests executed; every
   skipped test maps to a documented subset exclusion in this file; any
   failure or unlisted skip fails the gate.
6. **Cancellation:** mid-decode client abort → disconnect detected and
   next request's `response.created` emitted <250 ms after abort.
7. **No regression:** existing CLI surfaces untouched (parser tests,
   fixture tests, and a scripted legacy invocation matrix pass
   unchanged).

## Named risks

- Hand-rolled HTTP: scope strictly to loopback + Content-Length bodies +
  SSE writes; any request outside that shape gets a 4xx and a closed
  connection. If AI SDK fetch behavior (S2) demands more (keep-alive,
  chunked request bodies), prefer a minimal vendored parser over an
  async runtime — decision deferred to evidence.
- Dual-boundary capture doubles per-turn snapshot cost (S0: capture
  8–42 ms each, RAM); budget honestly in the stats line. Durable
  dual-publish doubles blob churn; F6 dedupe work may be pulled earlier
  if eviction pressure shows up in S2 agent loops.
- The serve stats surface is a new pinned contract; version it from day
  one (`serve_stats_v1`).

## Adversarial review record

k3 review (`ses_fe8ee25c6ffe`, 2026-08-18) returned REVISE with six
defects — item-list grammar vs turn grammar, `developer` role, unknown-
field policy, `sequence_number`, TTFT gate gameable by heartbeats, and
two interpretation-passable gates — all incorporated above, plus: dual
durable publish demoted to a default-off flag, replay-fidelity golden
pulled into S1, fixture-before-renderer ordering, and `x_qwen.stats`
response echo. Named riskiest assumption: stock-provider reasoning-item
replay fidelity (mitigated by the S1 golden; measured by the S2 gate's
`matched_tokens` definition).
