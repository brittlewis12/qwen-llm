# v0.465 Single-File S8 Replay Gate

Goal: unblock true-long S8 replay economics when the local corpus has too few
independent files above 32k tokens.

## Commands

```sh
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/chaos.json \
  --tokens 8 --context 32768 --stride 1 --blocks 2 \
  --start-block 0,20 --timing-iters 1 --timing-warmup 1 \
  --margin-threshold 0.0003 \
  > target/profiles/v0465-a3b-real-economics-s8-c32768-single-file.tsv

uv run scripts/profile/replay_economics.py \
  --real-margin target/profiles/v0465-a3b-real-economics-s8-c32768-single-file.tsv \
  --occupancy '8=1.0' \
  > target/profiles/v0465-a3b-real-economics-s8-c32768-single-file-econ.tsv
```

Smoke before the true-long gate:

```sh
target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/chaos.json \
  --tokens 2 --context 16 --stride 1 --blocks 2 --start-block 0 \
  --timing-iters 0
```

## Results

| Context | Window | Gross Save | Validated Net | Fallback Slots | Min Replay Margin |
| ---: | --- | ---: | ---: | ---: | ---: |
| `32768` | `0..2` | `25.92%` | `14.41%` | `0.00` | `0.007845` |
| `32768` | `20..22` | `16.72%` | `17.78%` | `0.00` | `0.009202` |

Full-S8 occupancy economics: blended save `16.09%` across the two rows.

## Caveat

Single-file strided slots are a mechanism-positive gate, not a product gate. They
do not prove independent-prompt diversity, ragged occupancy, attention-containing
slices, or scheduler p95 behavior. They do remove the old local-corpus blocker for
true-long GDN-only replay windows.

## Decision

Replay remains the live high-EV decode branch for serving-style full S8 occupancy.
The next gate is a minimal S8-only shadow policy around the same `blocks=2`,
GDN-only shape with margin fallback, replayed-token share, fallback rate, and p95
latency accounting.
