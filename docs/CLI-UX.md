# CLI UX

Evidence and decisions for the agent-facing `qwen` command line.

## Purpose

Make ordinary inference semantically safe and easy to discover without hiding
the exact raw and research interfaces. Treat observed behavior separately from
proposed solutions, preserve existing automation, and validate help by asking
fresh model contexts to use it.

## 2026-08-10 baseline

The initial study used Qwen3.6-35B-A3B UD-Q4_K_S at temperature zero. Each
cell was an independent, correctly ChatML-rendered request in one resident
JSONL process. This is direct interface evidence from one model/configuration,
not a population result.

Six frozen tasks covered ordinary user text, system plus user, exact raw input,
messages JSON from stdin, multiple requests under one model load, and response
text without operational noise.

Before seeing help, Qwen naturally expected:

- an ordinary positional or user-oriented prompt, not raw model input;
- `--system` for a system instruction;
- an explicit `--raw`-like escape hatch;
- `--messages` with stdin;
- a chat or batch mode that keeps the model loaded; and
- `--quiet`, with response data on stdout and diagnostics on stderr.

Given the shipped `qwen -h`, it used raw `-p` for an ordinary question, judged
system plus user impossible without an undocumented messages schema, naturally
tried the unsupported `--messages -`, found resident JSONL clearly, and found no
quiet mode. It described the help as a dense wall with no ordinary examples.

## Reasoning control evidence

With the ordinary Qwen assistant generation prefix, both baseline cells spent
1,024, 1,536, and then 2,048 generated tokens in `<think>` without completing
the requested final answer. The shipped default is 64 total generated tokens.

A user-text `/no_think` suffix did not change this behavior. The model-family
template transition did: preclosing the exact empty block
`<think>\n\n</think>\n\n` produced `OK.` and EOS in three generated tokens.

This supports a template-level generation control. It does not yet support a
separate thinking budget, forced mid-generation transition, changed token
default, or presentation-only answer filter.

## Candidate comparison

The study compared three concise surfaces:

1. `run` and `batch` subcommands with explicit `--user`;
2. the same subcommands with a positional user prompt; and
3. a repaired flat flag namespace.

Task cells completed successfully on all three, but the positional form caused
uncertainty about whether `USER` was a flag or positional value. The flat form
caused Qwen to emit an unsupported positional ordinary prompt despite help that
specified `--user`. Two counterbalanced ranking cells converged on subcommands
and explicit `--user`; one selected that form directly, while the other selected
the positional candidate only after recommending that its syntax be changed to
the explicit form.

## Selected first slice

Add a progressive, backward-compatible front door:

```sh
qwen run -m MODEL --user "Explain this"
qwen run -m MODEL --system "Be concise" --user "Explain this"
qwen run -m MODEL --user -
qwen run -m MODEL --messages -
qwen run -m MODEL --raw-prompt '<exact model input>'
qwen run -m MODEL --user "Explain this" --no-thinking
qwen run -m Qwen3.8-27B.gguf --user "Explain this" --reasoning-effort medium
```

Contract:

- `--user` and optional `--system` always use the detected model-family
  template.
- `--raw-prompt` is always untemplated.
- `--user -` reads one complete user message from stdin.
- `--messages -` accepts either a bare message array or a wrapped
  `{ "messages": [...] }` document.
- Ordinary chat renders through the serve renderer. On digest-pinned Qwen3.6
  templates the generation suffix is the released `<think>\n` (the model
  continues inside an open think block; output starts with reasoning text);
  pinned Qwen3.5 templates default to the released no-thinking suffix.
- `--no-thinking` is a prompt-rendering guarantee, not an output filter, and is
  rejected with raw input. On any model whose chat template digest is pinned
  (Qwen3.5, Qwen3.6, Qwen3.8, Flash-Next) it selects the released preclosed
  thinking suffix; unrecognized templates fail closed. (Until 2026-09-06 this
  was restricted to an exact Qwen3.6-35B-A3B metadata tuple, the only surface
  the transition had been tested on.) Qwen3.8 otherwise uses its upstream
  xhigh transition.
  DeepSeek ordinary chat is already non-thinking, so the flag is an idempotent
  guarantee there.
- `--reasoning-effort` is resolved by model family for structured
  `--user`/`--messages` input. Validated Qwen3.8 accepts
  `low|medium|xhigh` and defaults to upstream xhigh; Muse Glimmer accepts
  exact `low|medium|high|xhigh` reasoning strengths and defaults to `high`;
  DeepSeek V4 accepts its `low|high|max` thinking tiers (preserving replayed
  reasoning) and defaults to ordinary chat. Each family rejects the other
  families' levels by name rather than coercing; raw input and
  `--no-thinking` combinations fail closed.
- Muse Glimmer uses the released `temperature=1`, `top_p=.95`, `top_k=64`,
  `min_p=0` preset. Each explicitly supplied sampling flag overrides only its
  corresponding field; the request seed remains explicit and deterministic.
- Non-Muse modern messages accept the strict ordinary-chat subset (optional
  leading system or developer, alternating user/assistant turns, a final user
  turn) and, on pinned Qwen templates (`capabilities.input.tools` in `qwen
  info --json`; the same rule serve applies), OpenAI-shaped tool
  conversations: a
  wrapper `tools` list, assistant `tool_calls` (each followed by exactly one
  `tool` result), assistant `reasoning_content`, and a final user turn or a
  completed tool-result round awaiting the assistant. Inline `<think>` in
  assistant content, unknown fields, wrapper metadata, undeclared tools, and
  unmatched results are rejected rather than ignored. Rendering is
  byte-checked against the released Qwen3.6 template
  (`tests/fixtures/qwen36_chat_template_oracle_v1.json`).
- Muse modern messages use the shared ATEM request contract. They preserve
  validated reasoning and tool history, require declared calls and matching
  tool results, and accept only histories awaiting an assistant continuation.
- The Qwen3.8 surface is text-only. Image content arrays, developer/tool roles,
  structured calls/results, response formats, and projector execution remain
  outside this contract.
- Existing flat flags retain their parsing semantics. Omitted sampling values
  now resolve after family detection, so Muse flat and modern invocations both
  receive its released preset while explicit values remain unchanged.

Muse also has a request-native benchmark surface:

```sh
qwen-bench muse-request -m Muse-Glimmer.gguf \
  --prompt "Explain the result" --reasoning-strength high --tokens 64
qwen-bench muse-request -m Muse-Glimmer.gguf \
  --messages request.json --runs 3 -o json
```

`muse-request` uses the same strict `MuseGlimmerRequest` ATEM renderer and all
four reasoning strengths as run and Lens. Its sampler uses the released preset
shared by normal run and resident serve; Lens keeps its deterministic greedy
default. The benchmark loads one resident model and session, performs one
full-request warmup by default, then resets the resident position and
reconstructs the sampler for every timed run. Its versioned JSON object keeps
prompt forwards, sampled tokens, emitted tokens, and generation transitions as
separate denominators. It is intentionally not a `pp<N>`/`tg<N>` row: those
commands retain their llama-bench synthetic meaning and still use the ordinary
Qwen runtime. `workload_qualified` means only canonical source identity plus a
consistent warmup/timed token shape; raw rates do not claim an optimized build
profile or statistical confidence. The report validates annotated ATEM input
but marks output ATEM validation `not_performed`; a stop token alone is not
reported as proof of grammatical closure. Provenance records existing metadata
identities and tensor byte counts without hashing model weight payloads.

The short root help should lead with commands and copy-ready examples. The long
help may retain the expanded documented legacy/research flag surface.

## Adversarial calibration

A read-only CX review returned `REVISE` on the initial partial implementation.
It identified legacy stdin leakage, missing family dispatch, overbroad
no-thinking claims, permissive modern Qwen messages, duplicate defaults, and a
misleading raw-only `batch` neighbor. A resumed review endorsed the narrowed
direction subject to these gates:

- preserve a closed modern input kind rather than mutating legacy request
  fields during parsing;
- perform cheap validation and GGUF family detection before consuming stdin;
- keep modern generation values as optional overlays on canonical defaults;
- fail closed on unsupported modern message semantics;
- constrain Qwen no-thinking to a tested metadata tuple (since widened to any
  pinned template digest); and
- prepare prompt bytes before model residency without introducing a second
  tokenizer path.

The selected contract above incorporates those corrections. Legacy
`--messages -` remains unchanged and unsupported; stdin document support belongs
only to `qwen run --messages -` in this slice.

## Field evidence

Later on 2026-08-10, a separate OpenCode agent working on Recall interface
design (`ses_084e54a7bffevGEXaLl2Yhph12`) discovered this runner without being
prompted to use it. It located the project, read the README, inspected
`qwen run --help`, and selected the modern structured path:

```sh
qwen run -m MODEL --messages - --no-thinking --temp 0 --max-tokens 700
```

It used Qwen as a usability participant rather than falling back to legacy raw
`-p`. The resulting review changed the Recall candidate: the participant read
`applied=100` as 100 returned messages despite a separate `messages=37` field,
and exposed ambiguity about how to continue a capped result window. This is one
naturalistic episode, not evidence of prevalence, but it demonstrates
unprompted discovery, correct interface composition, and direct workflow value.

The response reached the 700-token cap with `stop_reason=token_limit`, so this
episode does not demonstrate reliable answer completion. It instead preserves
response-capacity and termination signaling as separate follow-up work; unlike
the baseline failure, the truncation was not hidden-thinking starvation.

## Post-change behavioral rerun

A fresh 10-cell rerun used the same pinned Qwen model, temperature zero, exact
preclosed no-thinking suffix, and one resident process executing requests
serially. Every cell reached EOS in 34 to 50 generated tokens.

The six common-path cells all selected `qwen run`. Ordinary user text, system
plus user, user stdin, messages stdin, and raw input were exact first attempts.
The no-thinking cell selected the intended `--user ... --no-thinking` shape;
its user text copied an ambiguity in the evaluator wording, so that cell is
command-shape evidence rather than exact-literal evidence.

The deferred capabilities exposed two useful failures:

- Given only short root help, the resident-batch cell chose repeated
  `qwen run` rather than escalating to long help. Given long root help, it
  correctly selected `qwen -m MODEL --requests-jsonl -`.
- The short-help quiet cell mistook `--no-thinking` for diagnostic suppression.
  Given long root help, it recognized that no suppression flag existed but
  still combined root-only `--prompt` with `qwen run`, producing an invalid
  hybrid command.

These failures do not justify adding the deferred features to this slice. They
do justify making the disclosure path task-oriented: short help now points to
long help specifically for resident JSONL batching, describes `--no-thinking`
as unrelated to CLI diagnostics, and states that legacy flags are flat rather
than composable with `qwen run`.

A focused five-cell rerun confirmed exact no-thinking composition and changed
the short-help resident-batch action to `qwen --help`. The model also correctly
stated that `--no-thinking` does not suppress diagnostics, but still offered a
non-quiet fallback instead of reporting the capability unavailable. With long
help it repeated the invalid `qwen run --prompt` hybrid; in a separate legacy
raw task it recognized root scope but omitted the model argument. The next
iteration therefore makes diagnostic suppression explicitly unavailable and
gives long help copy-ready flat examples containing both `-m MODEL` and the
legacy request flag.

In the final three-cell check, short help produced an explicit `UNAVAILABLE`
answer for diagnostic suppression, and the legacy raw cell copied
`qwen -m MODEL --prompt hello` exactly. The long-help quiet cell also preserved
the correct unavailable conclusion and invented no command, although it
malformed the requested response label as `COMMAND: UNAVAILABLE: ...`. That is
evidence for the interface semantics, not perfect participant-format
compliance.

## Deliberate deferrals

- quiet and diagnostic-level guarantees;
- answer-only or split reasoning/answer presentation;
- token-budget and default changes;
- per-request message *history* objects in batch (single-turn `user` rows
  landed on 2026-09-07, see below);
- a `batch` subcommand over the existing raw-only JSONL protocol;
- aliases, config resolution, and `QWEN_MODEL`;
- memory-policy changes;
- sessions, daemons, cancellation, and conversation state; and
- broader tool, developer, response-format, or multimodal message schemas.

These remain valuable, but combining them with the input-front-door change
would obscure behavioral attribution and enlarge the compatibility surface.

The raw-only batch wrapper was part of the initial candidate. Adversarial review
found that placing it beside templated `run` would invite agents to submit
`user`, `messages`, or `raw_prompt` rows even though the existing protocol
accepts only `prompt` or `prompt_file`. The existing `--requests-jsonl` surface
remains available and was already discoverable in the baseline study.

2026-09-07 update: `--requests-jsonl` rows now accept a templated single-turn
form — `user` with optional `system`, `no_thinking`, and `reasoning_effort` —
rendered by the same pinned-template path as `qwen run --user`. Exactly one of
`prompt`, `prompt_file`, or `user` is required; rendering controls on raw rows
are rejected; templated rows require an ordinary Qwen model; DeepSeek V4 batch
rejects them. Output rows echo `input: {kind, template}`. (Until 2026-09-07
templated rows also required a pinned template; they now follow the same
family rule as `run --user` — plain chat renders the legacy bare ChatML
contract on an unpinned template, visible as `template: "generic"`, while
`no_thinking`/`reasoning_effort` still refuse there. `qwen info --json`
advertises the rule under `capabilities.input`.) This resolves the ambiguity
the earlier review feared by making the input form structural, and it removed
the chat-template re-implementation from `scripts/bench/text_capability_eval.py`
(packet schema v2). Conversation-history rows remain deferred until a named
consumer migrates.

## Validation gate

- Pin modern and legacy parser behavior, conflicts, help, and explicit-default
  provenance.
- Fixture-test exact Qwen and DeepSeek user/system rendering and Qwen's
  no-thinking suffix.
- Process-test that modern stdin is acquired only after cheap validation and
  GGUF family detection, while legacy invocations never gain new stdin
  behavior.
- Prove legacy raw prompts and JSONL requests retain token identity, output
  shape, stop reason, and model-residency behavior.
- Rerun the applicable frozen help tasks. Ordinary chat, system plus user, user
  stdin, messages stdin, raw input, and no-thinking should produce valid
  first-attempt commands without selecting legacy raw `-p` for ordinary chat.
  Record resident batch and quiet output as explicit deferrals rather than
  implying that this slice added them.
