# Empty reasoning round trip (2026-10-05)

Leverage map #4, bounded to 1-2 h. Question: when serve emits an empty
reasoning item (a thinking turn that closes its block immediately), what does
the next request carry, and does serve render that history as generated?

## Provider layer (`provider-capture.json`)

`scripts/serve/provider-capture/empty_reasoning.ts` drives the stock
`@ai-sdk/open-responses` provider at the version opencode ships (2.0.29,
loaded from opencode's own cache, so nothing was installed). It uses
`doGenerate`/`doStream` against a mock fetch with serve's exact item and SSE
shapes. For the empty item, serve emits `output_item.added`, an empty
`content_part.added`, `reasoning.done ""`, `content_part.done` and
`output_item.done`, with no delta events.

- **Output:** non-streamed and streamed, the provider yields a reasoning part
  with `text: ""` and metadata `{itemId, reasoningContent: [{reasoning_text,
  ""}]}` (streamed: `reasoning-start`, then `reasoning-end`).
- **Replay with that metadata:**
  `{"type":"reasoning","summary":[],"id":"rs_e1","content":[{"type":"reasoning_text","text":""}]}`.
  Serve accepts it as empty reasoning.
- **Replay without metadata:** `{"type":"reasoning","summary":[]}` with no
  `content`. **Serve refused it with a 400** ("reasoning items require plain
  content"). This is fixed: a reasoning item with neither `content` nor
  `encrypted_content` is empty reasoning in the shared parser and in both
  K2 parsers. Encrypted reasoning stays refused.

## Real client (`opencode-requests.json`, `mock_serve.py`, `opencode-isolated.json`)

Setup:
- opencode 0.1.0, `opencode run --local`, a second message with `-c`.
- Isolated config/data/state directories. The model entry mirrors the
  local `qwen-serve` provider entry (`@ai-sdk/open-responses`, `url`,
  `reasoning: true`, with and without `reasoning_options`).
- A stdlib mock answers every request with serve's shapes: a reasoning
  item that is empty or has text, then "Hi.".

Results:
- **Stored:** the session export shows opencode stored the reasoning part:
  `{"type":"reasoning","text":""}`, with **no provider metadata**.
- **Replayed:** the continued request carried **no reasoning item**, for
  empty and for non-empty reasoning alike:
  `user "Say hi."`, then `assistant "Hi."`, then `user "Again."`. The
  `reasoning` request parameter was null.

Interpretation:
- **Empty reasoning:** nothing empty-specific happens on this path. Serve
  renders the turn by its missing-reasoning rule (SERVE.md "Missing
  reasoning is empty reasoning"):
  - GLM and identified Qwen releases render it as the turn was generated;
  - DS4 house style renders a chat turn.
- **Non-empty reasoning:** history differs from what was generated
  wherever the family keeps past reasoning (Qwen3.8, DS4 house style).
- **Conflict with earlier evidence:** the PERF-LOG 2026-09-26 real-client
  run (opencode agent loop) matched every transcript boundary, so
  reasoning must have been replayed in that mode. The drop observed here
  may be specific to continuing a session across `run` invocations, to
  `--local`, or to this version.
- **Status:** not resolved within #4's bound. It is tracked as a separate
  map item.

## Decision

- Close #4. The reproduced defect (the 400 on a content-less empty
  reasoning item) is fixed.
- When the client keeps the provider metadata, empty reasoning round-trips
  exactly.
- The client-side omission of reasoning on continuation is a broader
  compatibility question, not an empty-reasoning one, and moves to its own
  item.
