# v0.427 Mixed-Prompt Real Margin Packet

Goal: test whether the v0.426 real-prompt replay margin signal survives prompt
classes beyond narrative markdown.

Commands:

```sh
target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/qwen-llm/crates/qwen-cli/src/bench.rs \
  --file /Users/tito/code/qwen-llm/docs/PERF-ROADMAP.md \
  --file /Users/tito/code/llm/game/v02_3.6_deep.json \
  --tokens 4 --context 128,512 --blocks 4 --start-block 0,20,28,32,36 \
  > target/profiles/v0427-a3b-real-margin-mixed4-c128-512-s4.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/qwen-llm/crates/qwen-cli/src/bench.rs \
  --file /Users/tito/code/qwen-llm/docs/PERF-ROADMAP.md \
  --file /Users/tito/code/llm/game/v02_3.6_deep.json \
  --tokens 4 --context 2048 --blocks 4 --start-block 0,20,28,32,36 \
  > target/profiles/v0427-a3b-real-margin-mixed4-c2048-s4.out

uv run scripts/profile/block_slice_margin_summary.py \
  target/profiles/v0421-a3b-real-margin-the-current-c512-s1.out \
  target/profiles/v0421-a3b-real-margin-md4-c512-s4.out \
  target/profiles/v0421-a3b-real-margin-the-current-c2048-s1.out \
  target/profiles/v0421-a3b-real-margin-md4-c128-512-s4.out \
  target/profiles/v0421-a3b-real-margin-md3-c2048-s3.out \
  target/profiles/v0421-a3b-real-margin-the-current-c3072-s1.out \
  target/profiles/v0427-a3b-real-margin-mixed4-c128-512-s4.out \
  target/profiles/v0427-a3b-real-margin-mixed4-c2048-s4.out \
  > target/profiles/v0427-a3b-real-margin-summary.tsv
```

Validation:

- A3B mixed narrative/code/docs/JSON real-margin probes at c128/c512/c2048
- `uv run scripts/profile/block_slice_margin_summary.py` over v0.426 + v0.427
- cx adversarial review session `019f201d-9777-7810-9b23-dbdbeb6dbeae`

## Results

The mixed-prompt packet adds 15 rows to the v0.426 markdown-rollout sample.
Across the combined 47-row sample:

| Metric | Value |
| --- | ---: |
| Route-set mismatch rows | `0/47` |
| Route-order mismatch rows | `3/47` |
| Worst `min_replay_margin` | `0.000047` |
| Min `x` cosine | `0.999999889` |
| Max `x` abs | `0.002511` |
| Max router-logit abs delta | `0.002356` |

Window-level fallback rates if the policy falls back when
`min_replay_margin < threshold`:

| Threshold | Fallback windows | Fallback rate |
| ---: | ---: | ---: |
| `1e-4` | `1/47` | `2.13%` |
| `3e-4` | `4/47` | `8.51%` |
| `1e-3` | `11/47` | `23.40%` |
| `3e-3` | `21/47` | `44.68%` |
| `5e-3` | `29/47` | `61.70%` |

## Decision

This keeps replay alive but does not make it scheduler-grade. cx's adversarial
read: `0/47` still leaves a wide failure upper bound, synthetic failures remain
real, and the next gate must be net value after fallback rather than more raw
correctness optimism.

Use `3e-4` as the first plausible policy point: it is above the known synthetic
failure margins and models as roughly `~7.5%` net slice savings if post-replay
fallback costs one exact replay and raw replay saves `~16%`. `1e-4` may be too
close to known failure margins; `1e-3` likely burns the win unless fallback can
abort before most replay work.

Next: model validation overhead + exact fallback cost + ragged slot occupancy.
Do not build the production scheduler until that model clears a meaningful net
savings gate.
