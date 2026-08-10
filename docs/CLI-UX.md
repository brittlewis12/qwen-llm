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
```

Contract:

- `--user` and optional `--system` always use the detected model-family
  template.
- `--raw-prompt` is always untemplated.
- `--user -` reads one complete user message from stdin.
- `--messages -` accepts either a bare message array or a wrapped
  `{ "messages": [...] }` document.
- `--no-thinking` is a prompt-rendering guarantee, not an output filter, and is
  rejected with raw input. On the exact validated Qwen3.6 model/tokenizer
  surface it selects the tested preclosed thinking suffix and fails closed
  elsewhere. DeepSeek ordinary chat is already non-thinking, so the flag is an
  idempotent guarantee there.
- Modern messages accept only the strict ordinary-chat subset: optional leading
  system, alternating user/assistant turns, and a final user turn. Unknown
  fields, wrapper metadata, unsupported roles, and structured assistant
  thinking are rejected rather than ignored.
- Existing flat flags retain their current parsing and runtime semantics.

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
- constrain Qwen no-thinking to a tested metadata tuple; and
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
- per-request message objects in batch;
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
