# Real Rollout Characterization — v0.101

Checkpoint characterization of real TheCurrent-derived prompt prefills on clean
`v0.101` (`4903a84`) using the detached worktree at
`/Users/tito/code/qwen-llm-v0101`.

Host: `zekrom` (`Apple M4 Max`)

Method:

- `qwen-bench tok --messages ... --iters 1` for prompt token counts
- `qwen-bench pp --prompt <rendered text> --runs 1 -o json` for this first
  prompt-only checkpoint
- A10B rows use `QWEN_PP_WARM_MOE_BANKS=1` to remove expert-bank first-touch
  noise

Compatibility caveat:

- This first checkpoint used already-rendered prompt text files.
- That means the A10B rows are **not** yet model-canonical for Qwen 3.5 replay,
  because Qwen 3.5 should strip prior assistant thinking on full replay while
  the short/medium fixtures here were rendered in Qwen 3.6 preserve-thinking
  mode.
- Future canonical real-rollout prompt checks should use `qwen-bench pp --messages`
  so the harness can apply the correct model-family replay semantics.

Rendered prompt corpus:

| label | file | chars | tokens |
| --- | --- | ---: | ---: |
| short_preserve | `docs/bench/tokenizer-prompts/current-reva-short-qwen36-preserve.txt` | 36,938 | 7,986 |
| medium_preserve | `docs/bench/tokenizer-prompts/current-mei-medium-qwen36-preserve.txt` | 100,435 | 23,122 |
| long_strip | `docs/bench/tokenizer-prompts/current-marcus-long-strip.txt` | 106,360 | 25,610 |

Re-evaluation note:

- `long_strip` is **not** materially longer than `medium_preserve`; it is only a
  useful alternate prompt-shape control.
- True longer real-rollout candidates already available in source assets are:
  - `v02_reva.json` preserve: `34,502` tokens
  - `estonia.json` preserve: `27,289` tokens
  - `current-marcus-long.json` preserve: `49,561` tokens
- So this first `v0.101` note should be treated as an initial sparse realism
  check, not as sufficient characterization of the truly long rollout regime.

Prompt-only throughput on `v0.101`:

| model | prompt | tokens | t/s |
| --- | --- | ---: | ---: |
| 35B A3B | short_preserve | 7,986 | 512.85 |
| 35B A3B | medium_preserve | 23,122 | 267.15 |
| 35B A3B | long_strip | 25,610 | 166.47 |
| 122B A10B | short_preserve | 7,986 | 186.50 |
| 122B A10B | medium_preserve | 23,122 | 203.86 |
| 122B A10B | long_strip | 25,610 | 212.23 |

Current read:

- Real rollout prompt throughput is already far below the synthetic `pp512/1024`
  headline rows, so synthetic prompt work should never be read as the whole user
  story.
- A3B degrades materially across these longer prompts on `v0.101`; this deserves
  a proper follow-up characterization instead of assuming synthetic `pp1024`
  tells the whole prompt story.
- A10B does not show an obvious monotonic collapse across this small sparse set,
  which suggests prompt-shape sensitivity rather than a single smooth long-prompt
  cliff.
- This is exactly why the real-rollout lane exists: to catch prompt-shape and
  templating regimes that synthetic prompts can hide.

Open questions to answer in the next pass:

- whether the A3B behavior is a true length slope or a prompt-shape artifact
- whether preserve-vs-strip thinking materially changes long prompt throughput
- whether the same fixture, truncated across multiple prefix lengths, shows a
  real A3B cliff or just prompt-specific noise
- how these real prompt rows compare to llama.cpp on the same rendered prompts
