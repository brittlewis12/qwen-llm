# Real Rollout Prompt Lane

Sparse real-rollout prompt lane for qwen-llm perf work.

Purpose:

- Keep fast synthetic `pp<N>` loops for short-feedback kernel work.
- Preserve a small, durable set of real chat-template prompts so we do not have
  to remember by hand that real usage regimes exist.
- Use these fixtures when a hypothesis could change scaling, prompt-shape
  behavior, or template/rendering sensitivity.

TheCurrent source chain:

- live game messages / rollouts: `/Users/tito/code/llm/game/*.json`
- live system prompt: `/Users/tito/code/llm/game/the_current_ring0_v0.2.md`
- copied tokenizer-ready fixtures: `docs/bench/tokenizer-messages/`
- rendered chat-template prompts: `docs/bench/tokenizer-prompts/`

Thinking-mode rule:

- Qwen 3.6 replay: preserve assistant thinking.
- Qwen 3.5 replay: strip assistant thinking.

This is model-family compatibility behavior, not a benchmark preference.

Selected sparse corpus:

| lane | source | format | render note | approx size | use |
| --- | --- | --- | --- | ---: | --- |
| short | `docs/bench/tokenizer-messages/current-reva-short-qwen36.json` | wrapped messages | preserve-rendered Qwen 3.6 fixture | 7,986 tokens | quick real-rollout canary |
| medium | `docs/bench/tokenizer-messages/current-mei-medium-qwen36.json` | wrapped messages | preserve-rendered Qwen 3.6 fixture | 23,122 tokens | medium real-rollout regression check |
| control-long | `docs/bench/tokenizer-messages/current-marcus-long.json` | bare messages | strip-rendered control in `tokenizer-prompts/` is only 25,610 tokens | 25,610 tokens | alternate shape control; not a true long endpoint |
| real-long | `/Users/tito/code/llm/game/v02_reva.json` | wrapped messages | preserve-rendered full rollout | 34,502 tokens | real long TheCurrent rollout stress case |
| real-long-alt | `/Users/tito/code/llm/game/estonia.json` | wrapped messages | preserve-rendered full rollout | 27,289 tokens | alternate long rollout with preserved thinking |
| very-long | `docs/bench/tokenizer-messages/current-marcus-long.json` | bare messages | preserve-rendered full replay | 49,561 tokens | very long stress case |

Rules:

1. Do not replace synthetic `pp<N>` with this lane for everyday kernel work.
2. Do run this lane when the hypothesis is about:
   - scaling vs prompt length,
   - template/rendered prompt structure,
   - prompt-shape cliffs,
   - real history with `<think>` preservation/stripping,
   - regressions that might only show up outside synthetic prompts.
3. Use endpoints first, not full ladders:
   - short + medium for bounded prompt wins,
   - short + real-long for suspected scaling effects,
   - only add more points if endpoints imply a crossover story.
4. Treat `control-long` as an alternate prompt-shape control, not as proof that
   we characterized genuinely long real-rollout behavior.
5. Prefer `--messages` JSON inputs so the harness can render the prompt with the
   model-appropriate thinking policy.
6. Use rendered prompt text files only when you intentionally want a frozen,
   pre-rendered artifact for comparison.

Suggested command shapes:

Tokenizer / prompt-size characterization:

```sh
./target/release/qwen-bench tok \
  -m "$MODEL" \
  --messages docs/bench/tokenizer-messages/current-mei-medium-qwen36.json \
  --messages-preserve-thinking \
  --iters 1
```

Prompt-throughput characterization:

```sh
./target/release/qwen-bench pp \
  -m "$MODEL" \
  --messages docs/bench/tokenizer-messages/current-mei-medium-qwen36.json \
  --messages-preserve-thinking \
  --runs 1
```

Discipline:

- Treat this as a separate realism lane, not as a required sweep for every perf
  change.
- If a change only helps short synthetic prompts but regresses one of these real
  rollouts, document that explicitly.
- If a hypothesis is expected to reduce context-growing work, do not kill it only
  because the short fixture moved by less than 1%.
