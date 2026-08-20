# qwen serve — facade contract (S1 preregistration)

Status: S1 + S2 implemented on `serve/s1`; S1 gates executed 2026-08-18, S2 gate 2026-08-19 — see docs/bench/2026-08-19-s2-agent-gate/ and
`docs/bench/2026-08-18-serve-s1-gates/` (1 pass, 2 conditional pass,
3–7 pass). Originally preregistered earlier the same day.
Evidence base: S0 packet
(`docs/bench/2026-08-18-facade-s0-render-prefix-stability/`), adversarial
jam (`ses_fe8ee25c6ffe`, three rounds), and the full investigation session
(OpenCode Recall). This document is the binding slice contract; superseded
jam positions are not.

## Program arc (decision record)

| Unit | Contents | Consumer | Gate |
|---|---|---|---|
| S0 | Prefix-stability falsifier | measurement | DONE — see packet RESULTS |
| **S1 (this doc)** | Resident serial server, Open Responses subset, no tools | the game (thin HTTP client) | below |
| S2 ✅ | Tool items (XML-parameter form from the template oracle), `allowed_tools`, continuation rendering | opencode via stock `@ai-sdk/open-responses` | **PASS: 10/10 requests checkpoint-hit** (94–100 % restored), 5/5 turns tool-called — docs/bench/2026-08-19-s2-agent-gate/ |
| S3 (in progress) | Pre-opened (headless) reasoning support, DeepSeek V4 family backend, DFlash drafter integration. `encrypted_content` opaque round-trip is **not** built — the S2 capture showed plain reasoning content already replays verbatim, so it demoted from necessity to hardening. | opencode, DS4 clients | DS4 live gate (pending): warm snapshot hit rate, byte identity vs CLI, headless partition conformance |
| S4 | Public v0: CC shim, install, memory admission UX, bench repro, compliance claim | the world | sub-100 ms turn-2 TTFT demo, resident @32k |

Parked: items npm provider (stock AI SDK provider exists), WS
connection-local continuation (post-S4, measurement-gated),
`previous_response_id` stored mode, concurrency.

## S1 scope

One new subcommand:

```sh
qwen serve -m MODEL [--addr 127.0.0.1:8737] [--max-tokens N] [--drafter GGUF]
# DeepSeek V4 additionally requires --max-context-tokens (startup-fixed forward budget)
```

- **Residency:** model loads once; the process is the warm tier. S0's F2
  finding makes this the TTFT mechanism (per-process paging floor is
  5–8 s on A3B even with checkpoint hits; `load_ms` does not cover
  first-touch). **S1-as-built is RAM-cache-only**: durable checkpoint
  publication/restore is not wired into serve and lands S2+;
  cross-restart warmth currently re-prefills (closing k3 review, D3).
  Durable checkpoints remain the intended cross-restart substrate.
- **Speculative decode (`--drafter`, v0.77 DFlash):** a request
  speculates only when it cold-prefills its whole prompt and decodes
  greedily. Restored checkpoint positions carry no captured target hidden
  states, so seeding the drafter's cross-context from them is impossible
  (same reason the CLI excludes `--durable-prefix-cache`); restored
  requests decode serially. Output is identical either way — greedy
  accept-prefix over an exact target verify — and the per-request
  `serve phases:` line reports `decode_path=dflash|serial`, with a
  `serve dflash:` line carrying acceptance and backoff counters.
  Measured on Qwen3.8-27B + DFlash2-Q8_0 (25-token prompt, 200 tokens
  out): cold/dflash 25.3 tok/s vs warm/serial 24.7 tok/s with
  acceptance 56/203 and alpha-backoff engaged — i.e. the policy
  correctly detected weak acceptance on this prompt and stopped
  speculating rather than losing time. Byte-identical outputs across
  both paths were verified live.
- **Determinism scope (F7):** no serve surface promises temp-0 byte
  identity across differing checkpoint-restore topologies; transcripts
  are byte-deterministic conditional on restore partitioning.
- **Stats contracts (frozen):** the stderr `qwen_diag` line carries
  `version=serve_stats_v1`; the `x_qwen` response echo's field names
  `matched_tokens`, `restore_ms`, `prompt_tokens` are frozen;
  `usage.input_tokens_details.cached_tokens` reports tokens restored
  from checkpoints.
- **Serial, blocking, no async runtime.** `std::net` listener, one
  request in flight, OS listen backlog queues the rest. HTTP/1.1 with
  `Connection: close`; hand-rolled request parse (loopback threat model;
  request bodies are `Content-Length` JSON).
- **Qwen35/Qwen35Moe and DeepSeek V4.** DS4 runs its own session and
  snapshot stack (`serve/backend_ds4.rs`) with a startup-fixed forward
  budget, a serve-owned snapshot LRU (DS4 has no engine-side RAM prefix
  cache), and no tool support yet — tool definitions fail closed there.
- **Stdout is never written.** All diagnostics via the existing stderr
  tracing surface; per-request `qwen_diag` stats line retained and
  extended with `matched_tokens` and `restore_ms` (the S2/S3 gates are
  defined on this log, never on "the session completed").

## Wire subset (Open Responses)

`POST /v1/responses` accepting:

- `model` — must equal the loaded model id; else `model_not_found`.
- `input` — string (one user message) or item array in the subset:
  `message` (roles `system`|`developer` (system-equivalent, documented
  mapping)|`user`|`assistant`, content string or `input_text` parts),
  `reasoning` (plain `content` only in S1). **Item-sequence validation,
  not turn grammar** (review defect 1: the spec's `input` is an item
  list; stock AI SDK traffic legally contains consecutive user
  messages and, in S2, interleaved tool items): system/developer/
  `instructions` at head only, `reasoning` items must immediately
  precede their assistant message, final item must be a `user` message
  (S1) or `function_call_output` (S2), everything else accepted in
  order. Unknown item *types* → `invalid_request`; unknown *fields* are
  ignored (top-level request fields logged once per name; `id`/`status`
  on replayed input items accepted and ignored).
- `instructions` — optional system text (exclusive with a system item).
- `max_output_tokens`, `temperature`, `top_p` — standard.
- `x_qwen` extension object — `seed`, `top_k`, `min_p`, `no_thinking`
  (identity-gated exactly as `qwen run`); spec-legal implementor
  extension, documented.
- `stream` — SSE when true, single JSON response otherwise.
- `store` — must be `false`/absent; `true` → `invalid_request`.
- `previous_response_id` — → error code `previous_response_not_found`.
- `truncation` — only `"disabled"` (default). The engine already fails
  closed on context overflow (S0 F3); serve maps that to the spec error
  instead of a process exit.
- `tools` — function tools are supported on Qwen families (S2); DeepSeek
  V4 fails closed on any tool definition. Hosted tool types fail closed. Definitions render into the family template's `# Tools` system
  block, byte-pinned to the template oracle.
- `tool_choice` — `"auto"` (default) or an `allowed_tools` object.
  Narrowing is enforced as a hard constraint on emitted calls while
  leaving rendered bytes identical, so prompt prefixes and their
  checkpoints stay valid across tool-menu changes (the spec's
  cache-preserving intent, test-pinned). Enforcement is post-generation:
  a suppressed call has already consumed tokens and remains in the
  completed-turn checkpoint key, so that continuation will not hit.
  `required`, `none`, and forced-function are not implemented.
- `/responses/compact` — not implemented (404); compaction is outside the
  S1–S4 arc and revisits with the WebSocket transport question.
- `x_qwen.stats: true` — echoes `{matched_tokens, restore_ms,
  prompt_tokens}` into the response object, so thin clients (the game)
  get per-request checkpoint stats without correlating server stderr
  (review R4). `usage` is always populated (agent clients budget on
  it).

Provenance notes: `no_thinking`/`top_k`/`min_p` in `x_qwen` are cheap
identity-gated pass-throughs of `qwen run` semantics, kept deliberately;
`stream: false` exists for the stock provider's `doGenerate` path in S2,
not for S1's consumer.

Output items: `reasoning` (when the model emits thinking; split at the
existing `</think>` partition seam) then `message` with `output_text`.
Truncated thinking (S0 F4) yields `reasoning` with `status:
"incomplete"` and `response.status = "incomplete"` with
`incomplete_details.reason = "max_output_tokens"`.

Streaming events: `response.created`, `response.in_progress`,
`response.output_item.added`, `response.content_part.added`,
`response.reasoning.delta|done` (spec event names, adjudicated by the gate-5 conformance suite), `response.output_text.delta|done`,
`response.content_part.done`, `response.output_item.done`,
`response.completed|incomplete|failed`, terminal `[DONE]`. Tool turns
add `function_call` output items with
`response.function_call_arguments.delta|done` (S2). **Every
event carries a monotonic `sequence_number`** (review defect 4 — the
conformance suite asserts ordering; retrofitting into a hand-rolled SSE
writer later costs more). SSE comment heartbeats (`: ping`) at ≥1 Hz
during prefill (client-timeout defense). Errors stream as
`response.failed` with the spec error envelope.

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
- Per turn the server captures **both** boundaries — prompt-boundary and
  completed — into the **RAM** prefix cache (8–42 ms each per S0; this
  makes the server's own next-turn path immune to client echo policy).
  Durable publication keeps the existing completed-else-prompt shadowing
  policy. **Durable publication remains parked**: S1-S3 shipped RAM-only,
  so cross-restart warmth still re-prefills. `--durable-dual-publish` is
  designed but unbuilt (review R2: at q38's ~90 KB/token, dual durable publish
  of a 32k context is ~6 GB/turn — LRU churn that evicts the prefixes it
  is meant to protect, and publish time serializes the next request on a
  serial server). Pre-implementation task: verify
  `prepare_checkpoint_boundary`/`cache_prepared_checkpoint` actually
  supports dual capture per turn (asserted, not yet demonstrated —
  review R6 runner-up).
- **Replay-fidelity coverage (review R6):** render→items→render byte
  identity is asserted by unit tests (`render.rs` split/render inverse,
  `tool_parse.rs` `qwen36_raw_echo_identity`, `render_ds4.rs` preserved
  history round-trip) rather than by a JSON fixture case.
- Tool-continuation golden fixtures are **written before the renderer**
  (review R3), so the renderer is fit to the fixture, never the reverse.
- Preserve/strip rendering policy: preserve is the default for the
  validated Qwen3.6 identities (owner position, Amendment 1 of S0;
  economics measured in S0 G2); strip remains available via `x_qwen`.
  Extension beyond Qwen3.6 waits on the behavioral packet.

## Cancellation

Client disconnect (write failure on SSE, read failure on socket) aborts
generation between token steps and prefill between chunks; the request
slot frees within 250 ms (measured: 24 ms). No signals, no threads beyond
the accept loop. Known limitation (D5): a TERM received while parked in
`accept()` unwinds on the next connection, not immediately.

## S1 gates

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
