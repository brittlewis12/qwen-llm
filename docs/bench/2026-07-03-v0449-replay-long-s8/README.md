# v0.449 Long-Context S8 Replay Gate

Goal: test the replay branch at longer contexts without touching the parallel
DFlash work or adding scheduler/product infrastructure.

## Command

```sh
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/chaos.json \
  --file /Users/tito/code/llm/game/marcus_full.json \
  --file /Users/tito/code/llm/game/v02_reva.json \
  --file /Users/tito/code/llm/game/estonia.json \
  --file /Users/tito/code/llm/game/v02_3.6_deep.json \
  --file /Users/tito/code/llm/game/gemma_oracle.json \
  --file /Users/tito/code/llm/game/cozy.json \
  --file /Users/tito/code/llm/game/gemma_claude.json \
  --tokens 8 --context 8192,16384 --blocks 2 --start-block 0,20 \
  --timing-iters 2 --timing-warmup 1 --margin-threshold 0.0003 \
  > target/profiles/v0449-a3b-real-economics-s8-c8192-16384-b0-b20.out

uv run scripts/profile/replay_economics.py \
  --real-margin target/profiles/v0449-a3b-real-economics-s8-c8192-16384-b0-b20.out \
  --occupancy '8=1.0' \
  > target/profiles/v0449-replay-economics-s8-long-fullocc.tsv
```

## Results

| Context | Window | Gross Save | Validated Net | Fallback Slots | Min Replay Margin |
| ---: | --- | ---: | ---: | ---: | ---: |
| `8192` | `0..2` | `21.50%` | `14.84%` | `0.00` | `0.002253` |
| `8192` | `20..22` | `21.74%` | `12.58%` | `0.00` | `0.001107` |
| `16384` | `0..2` | `21.22%` | `11.64%` | `0.00` | `0.002989` |
| `16384` | `20..22` | `21.50%` | `12.36%` | `0.00` | `0.001639` |

Full-S8 occupancy economics: blended save `12.85%` across the four rows.

## Caveat

The current harness requires one independent file per active slot. The local
corpus has eight files above `16k` tokens, but only three above `32k`; therefore
an honest S8 `ctx32768` packet is blocked unless we add single-file strided-slot
support or collect more independent true-long prompts.

## Decision

S8 replay remains alive at long context. Do not build a broad scheduler yet:
production promotion is still gated on real occupancy/request traces and p95
economics. The next replay implementation, if any, should be a minimal headless
S8 policy around this exact blocks=2 shape, not another S4/S6 or local-kernel
variant.
