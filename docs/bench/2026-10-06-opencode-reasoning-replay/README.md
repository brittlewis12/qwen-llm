# opencode reasoning replay (2026-10-06, map #13)

Question: in normal use, which past reasoning does opencode send back to a
Responses server? Is #4's observation (none, on `run --local -c`)
specific to `--local` or to cross-run continuation? Where is it lost?

## Method

- Driver: `scripts/reference/opencode_reasoning_replay.py`.
- Mock: `scripts/reference/opencode_tool_mock.py`. It answers in `qwen
  serve`'s item and SSE shapes, so no model or GPU is involved.
- Each assistant step returns a reasoning item with distinct text
  (`R-call-<n>` before a tool call, `R-answer-<n>` before the answer).
  Every request body is logged.
- Client: opencode 0.1.0 with isolated config, data and state directories.
  Every `OPENCODE*` variable of the calling session is removed from the
  client environment.
- Each mode runs the same two runs:
  1. a first run, in which the model calls one tool (`glob`) and then
     answers;
  2. a second run that continues the session (`-c`) and repeats the
     pattern.
- The four modes:

  | Mode | Client | Provider |
  |---|---|---|
  | `local` | `run --local` | `npm: @ai-sdk/open-responses` |
  | `attached` | `opencode serve` + `run --attach` | `npm: @ai-sdk/open-responses` |
  | `attached-file-provider` | as `attached` | the cached `@ai-sdk/open-responses` 2.0.29 build, through a `file://` npm path |
  | `attached-openai` | as `attached` | `npm: @ai-sdk/openai` |

## Results (`report.json`)

| Mode | Runs | Reasoning items replayed |
|---|---|---|
| `local` | both exit 0 | **none**, in all 5 requests |
| `attached` | both exit 0 | **none**, in all 5 requests |
| `attached-file-provider` | refused to load | not applicable: `AI_UnsupportedModelVersionError` (2.0.29 implements provider specification v4; this opencode runs AI SDK 5, which needs v2) |
| `attached-openai` | both exit 0 | **none**, in all 5 requests (`store: false`, `include: ["reasoning.encrypted_content"]`) |

**Within one multi-step run.** The request after the tool result is
`user, function_call, function_call_output`. The `R-call` reasoning that
preceded the call is missing.

**Across runs.** The continued request is `user, function_call,
function_call_output, assistant, user`. Neither earlier reasoning item is
present.

**Storage.** A session export shows that opencode stores every reasoning
part with its text (`R-call-2`, `R-answer-3`, …), without `partial` and
without provider metadata.

## Where it is lost

The version strings and code below were read from the installed opencode
binary.

1. `MessageV2.toModelMessages` keeps stored reasoning parts (it skips only
   `partial` ones). `ProviderTransform.message` changes reasoning only for
   Anthropic and Bedrock (empty parts), and for models declaring
   `interleaved.field` (moved into an OpenAI-compatible field). So
   reasoning reaches the provider.
2. opencode maps `npm: "@ai-sdk/open-responses"` to a **bundled** provider,
   `@ai-sdk/open-responses` 1.0.35 (`BUNDLED_PROVIDERS`). Its input
   converter handles assistant `text` and `tool-call` parts and has **no
   `reasoning` case**, so reasoning parts are dropped silently.
3. The bundled `@ai-sdk/openai` converter keeps a reasoning part only when
   it carries an OpenAI `itemId` (otherwise: "Non-OpenAI reasoning parts
   are not supported"). With `store: false` it replays a summary plus
   `encrypted_content`. Serve provides neither, and refuses summary-only
   reasoning.
4. The cached 2.0.29 build does convert reasoning parts, falling back to
   the part's text when there is no metadata. But opencode 0.1.0 cannot
   load it.

**Correction to #4** (`docs/bench/2026-10-05-empty-reasoning-round-trip/`).
That packet's provider-layer capture drove the cached 2.0.29 build and
called it "the version opencode ships". It is not. opencode 0.1.0 uses
its bundled 1.0.35, which never replays reasoning. #4's serve-side fix (a
content-less reasoning item is empty reasoning) stands: it is about what
serve accepts.

## Consequences for serve

- **Families that keep past reasoning in history** (GLM-5.3 by default,
  Qwen3.8, DS4 house style):
  - Every opencode step renders the previous assistant step with empty
    reasoning, although the model generated reasoning there.
  - This changes the model's conditioning inside an agent loop.
  - It also changes the live-session prefix. Reuse stops at the previous
    step's reasoning, and that step plus the tool results are prefilled
    again on every step.
  - Earlier turns are unaffected: they were already rendered without
    reasoning in the previous prompt.
- **Families that drop past reasoning by template:** unaffected.
- **Serve needs nothing** to accept these requests; they are valid.

## Status

#13's question is answered, and the cause is located in the client: the
omission is real in normal use, including within one agent loop.

Open: whether serve should restore the reasoning of the turn it just
generated, when the request's history matches that turn exactly except for
the missing reasoning item. That is a policy decision, still to be made.
A live GLM-serve opencode session would measure the reuse cost and the
answer-quality effect.
