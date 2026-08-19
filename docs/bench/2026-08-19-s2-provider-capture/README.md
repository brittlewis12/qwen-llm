# S2-0: stock provider traffic capture

Executed 2026-08-19 per the k3 S2 ordering (step 0: capture provider
traffic before finalizing item validation). Hermetic harness at
`scripts/serve/provider-capture/capture.ts`: a mock `fetch` logs every
request byte from `@ai-sdk/open-responses@2.0.29` (`ai@7.0.70`) and
returns scripted serve-shaped responses, exercising three phases —
multi-turn chat replay, a full tool loop (function_call → SDK executes →
continuation), and SSE streaming. Raw captures:
`provider_capture_v1.json` (auth header redacted). Zero harness failures.

## Findings

1. **F-S2.1 (live bug, fixed):** the provider sends `tool_choice:"auto"`
   on every request including plain chat. Merged S1 rejected any
   `tool_choice` → all stock traffic would 400. Fix: accept the no-op
   `"auto"`, reject other values until S2 tools land. Capture-derived
   regression test added
   (`stock_provider_chat_request_shape_is_accepted`).
2. **F-S2.2 (R6 resolved favorably):** reasoning items replay **byte
   verbatim** — `content` text exact, `id` and `summary` preserved, parts
   typed `reasoning_text`. The stock path is lossless at the item level;
   the S3 zero-tail gate is reachable with plain `content` and
   `encrypted_content` demotes from necessity to hardening.
3. **F-S2.3 (grammar extension pinned):** tool-loop replay order is
   `user → reasoning → function_call{id, call_id, name, arguments} →
   function_call_output{call_id, output}` — reasoning immediately
   precedes a *function_call*, not an assistant message, so the S1
   "reasoning binds to assistant message" rule must extend to
   function_call in S2. `function_call_output.output` is a JSON string;
   no `status` fields on replayed items; final item may be
   `function_call_output`.
4. **F-S2.4:** system text arrives as request-level `instructions` (not
   a system item); user content parts are `input_text`, assistant
   replay parts are `output_text` — all already accepted by S1
   validation (review defects 1-3 paid off unchanged).
5. **F-S2.5:** tool definitions arrive as
   `{type:"function", name, description, parameters}` with JSON-Schema
   draft-07 bodies including a `$schema` field — the S2 render layer
   must decide how schema JSON maps into the Qwen template's `## Tools`
   block (template oracle shows a tools_header + per-tool sections).

## Consequences for the S2 unit order

Step 1 (XML golden fixtures) proceeds as planned; step 3's grammar
extension is now specified by real bytes rather than spec reading; the
step 5 live gate's biggest residual risk shrinks to model-side XML
emission quality (parser fuzz in step 2 covers it) and the
checkpoint-poisoning decision (malformed-XML turn must fail-or-salvage
before capture wiring).
