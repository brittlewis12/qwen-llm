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
  Qwen3.8, DS4 house style): every opencode step renders the previous
  assistant step with empty reasoning, although the model generated
  reasoning there. This changes the model's conditioning inside an agent
  loop.
- **Reuse cost for GLM.** GLM serve has one live session and no snapshot
  cache, so reuse is all or nothing (`decode_loop::extending_prefix`): the
  prompt must extend the whole committed history. (Correction, 2026-10-06:
  an earlier version blamed recurrent state that "cannot rewind". The Qwen
  and Flash-Next hybrids reuse with snapshots at boundaries; GLM lacks the
  implementation, not the possibility. See the lane audit.)
  - The previous step's reasoning differs, so **every agent step is a full
    re-prefill**.
  - opencode's requests here carry about 50 KB before any conversation:
    11 KB of instructions and 39 KB of schemas for 12 tools, roughly 12-14k
    tokens. At GLM's roughly 180 tok/s prefill that is on the order of a
    minute per step, versus prefilling only the new tool results when reuse
    holds.
  - Families that can truncate (Qwen's prefix reuse) lose only the
    previous step.
- **Families that drop past reasoning by template:** unaffected.
- **Serve needs nothing** to accept these requests; they are valid.

## Options (design review, cx session 01a10cc)

- **C, preferred:** a client-side fix. A pinned, self-contained AI SDK 5
  (provider specification v2) Responses provider whose input converter
  replays reasoning parts as reasoning items, loaded by opencode through a
  `file://` npm path. It fixes both conditioning and reuse, and keeps
  request semantics stateless.
  - Building it needs either a compatible provider source (not
    downloaded) or a new provider.
  - Enabling it in a user's opencode configuration is the user's choice.
- **Engine-side reuse, independent of reasoning:** a prompt-boundary
  checkpoint for GLM. It keeps the recurrent state at the end of each
  prompt prefill, so a request that extends the previous *prompt* (but not
  the generated turn) rewinds to it and prefills only the re-rendered turn
  and the new items. That removes the full re-prefill for any client that
  drops or edits the last generated turn. Rendering, and therefore
  conditioning, is unchanged.
- **B, not recommended by default:** serve restores the reasoning of the
  turn it just generated when the history matches it except for the
  missing item.
  - It cannot tell omission from intent (explicit empty reasoning, edits,
    `history_thinking: "strip"`).
  - Content matching does not establish which client owns the turn.
  - Restoring only the newest turn is not stable across steps: the turn
    restored at step k is missing again at step k+1.
- **A:** document only.

## Fix (opencode fork, 2026-10-06)

opencode is a local fork, so the client is fixed at the source.
- **Commit:** `8084ffaf7b` on branch `fix/open-responses-reasoning-replay`, in
  worktree `~/code/opencode-reasoning-replay`.
- **Change:** a bun patch (`patches/@ai-sdk%2Fopen-responses@1.0.35.patch`)
  backports the 2.x converter's reasoning case. Pending assistant text is
  flushed, then the reasoning part becomes `{type: "reasoning", summary:
  [], content: [{type: "reasoning_text", text}]}` in part order, without
  `content` when the text is empty. Text and tool calls are unchanged.
- **Test:** a new provider test against a real local server fails on the
  unpatched converter and passes with the patch. The package typecheck and
  491 provider and session tests pass.

Rerun with the patched binary (`report-patched.json`):

| Mode | Reasoning items replayed |
|---|---|
| `local` | every past item, in order: 1, 2 and 3 at the three continuation requests |
| `attached` | the same |
| `attached-file-provider` | not applicable: still refused (AI SDK 6 provider) |
| `attached-openai` | none (a separate provider, unchanged) |

## Status

#13's question is answered and the client is fixed in the fork.
- **Landed:** `fork-dev` was fast-forwarded to `8084ffaf7b`, then `bun
  install` and `bun run install:local` ran from the fork root
  (2026-10-06 16:52).
- **Verified:** with the installed binary, local and attached runs carry 0,
  1, 2 and 3 reasoning items at their four model requests: every past item,
  in order.

Next:
- **Serve check (scripted, amended 2026-10-07):** a scripted client in the
  patched client's item shapes (reasoning as `reasoning_text` content,
  function calls and outputs), extending `serve_tool_loop_check.py` with
  the captured opencode instructions and tool schemas, against a
  separately launched GLM serve: three or more steps, plus branch,
  shared-prefix new-session and cold cases. Record per step: reasoning
  items sent, effective prompt tokens, reused and prefilled tokens, time to
  first token and task wall time. Continued steps should extend the live
  session. Screens do not drive opencode or an opencode server in use.
- **Prompt-boundary checkpoint:** still useful for clients that drop or
  edit the last turn, but no longer needed for opencode.
- One session is a mechanism screen. It is not a quality or default-policy
  decision.
