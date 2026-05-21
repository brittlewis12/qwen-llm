# Tokenizer Messages Corpus

Generic chat-message JSON fixtures copied from long narrative rollouts for tokenizer and real-prompt perf work.

Important: the thinking-preserve/strip choice is **not** a user preference.

- Qwen 3.6-family replay should preserve prior assistant `<think>...</think>`
  history.
- Qwen 3.5-family replay should strip prior assistant thinking on full replay.

Use the message JSON as the canonical source when benchmarking real prompts so
the harness can render model-appropriate replay semantics at runtime.

| file | json shape | default thinking mode | notes |
| --- | --- | --- | --- |
| `current-reva-short-qwen36.json` | `{ meta, messages }` | preserve (Qwen3.6 auto) | Narrative rollout transcript fixture. |
| `current-mei-medium-qwen36.json` | `{ meta, messages }` | preserve (Qwen3.6 auto) | Narrative rollout transcript fixture. |
| `current-marcus-long.json` | `[messages]` | model-dependent (`strip` for Qwen 3.5 replay, `preserve` for Qwen 3.6 replay) | Narrative rollout transcript fixture. |
