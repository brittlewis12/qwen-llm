# v0.438 Replay Real-Window Economics Gate

Goal: move block-slice replay from gross synthetic savings toward a production
policy gate by measuring real-prompt windows with validation overhead, exact
fallback modeling, and ragged active-slot counts.

Implementation shape:

- `decode-block-slice-real-margin` now has optional timing knobs:
  - `--timing-iters N`
  - `--timing-warmup N`
  - `--margin-threshold T`
- When timing is enabled, each row reports:
  - exact baseline wall ms/token
  - replay wall ms/token
  - gross wall savings
  - validated replay wall and GPU ms/token
  - average fallback slots under the replay-margin threshold
  - net wall savings after charging exact fallback for fallback slots
- The validated path intentionally runs replay block-by-block and reads replay
  route margins after each block, so it includes the command-buffer split and CPU
  readback cost a conservative guard would pay.

Commands:

```sh
cargo check -p qwen-llm -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/llm/game/the_current_ring0.md \
  --file /Users/tito/code/llm/game/the_current_ring0_v0.2.md \
  --file /Users/tito/code/llm/game/the_current_ring0_obfuscated.md \
  --file /Users/tito/code/llm/game/the_current_ring0_cipher.md \
  --file /Users/tito/code/llm/game/v02_3.6_deep.json \
  --file /Users/tito/code/llm/game/v02_3.6_run.json \
  --file /Users/tito/code/llm/game/marcus_full.json \
  --tokens 8 --context 512,2048 --blocks 2 --start-block 0,20 \
  --timing-iters 2 --timing-warmup 1 --margin-threshold 0.0003 \
  > target/profiles/v0438-a3b-real-economics-s8-c512-2048-b0-b20.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/llm/game/the_current_ring0.md \
  --file /Users/tito/code/llm/game/the_current_ring0_v0.2.md \
  --file /Users/tito/code/llm/game/the_current_ring0_obfuscated.md \
  --file /Users/tito/code/llm/game/the_current_ring0_cipher.md \
  --file /Users/tito/code/llm/game/v02_3.6_deep.json \
  --tokens 6 --context 512,2048 --blocks 2 --start-block 0,20 \
  --timing-iters 2 --timing-warmup 1 --margin-threshold 0.0003 \
  > target/profiles/v0438-a3b-real-economics-s6-c512-2048-b0-b20.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/llm/game/the_current_ring0.md \
  --file /Users/tito/code/llm/game/the_current_ring0_v0.2.md \
  --file /Users/tito/code/llm/game/the_current_ring0_obfuscated.md \
  --tokens 4 --context 512,2048 --blocks 2 --start-block 0,20 \
  --timing-iters 2 --timing-warmup 1 --margin-threshold 0.0003 \
  > target/profiles/v0438-a3b-real-economics-s4-c512-2048-b0-b20.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/llm/game/the_current_ring0.md \
  --file /Users/tito/code/llm/game/the_current_ring0_v0.2.md \
  --file /Users/tito/code/llm/game/the_current_ring0_obfuscated.md \
  --file /Users/tito/code/llm/game/the_current_ring0_cipher.md \
  --file /Users/tito/code/llm/game/v02_3.6_deep.json \
  --file /Users/tito/code/llm/game/v02_3.6_run.json \
  --file /Users/tito/code/llm/game/marcus_full.json \
  --tokens 8 --context 128,512,2048 --blocks 2 \
  --start-block 0,20,28,32,36 \
  > target/profiles/v0438-a3b-real-margin-blocks2-s8-c128-512-2048.out

uv run scripts/profile/block_slice_margin_summary.py \
  target/profiles/v0438-a3b-real-margin-blocks2-s8-c128-512-2048.out \
  > target/profiles/v0438-a3b-real-margin-blocks2-s8-summary.tsv
```

Validation:

- `cargo check -p qwen-llm -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- Real-prompt S4/S6/S8 economics probes at contexts 512 and 2048
- Broader S8 blocks=2 margin packet at contexts 128, 512, and 2048
- cx ranking session `019f24af-a733-71c1-b218-f2878ff8e3dd`

## Results

S8, timing rows, no fallback slots at threshold `3e-4` in these four rows:

| Window | Context | Gross save | Validated net |
| --- | ---: | ---: | ---: |
| `block0..2` | `512` | `21.75%` | `13.10%` |
| `block20..22` | `512` | `27.60%` | `20.34%` |
| `block0..2` | `2048` | `21.40%` | `11.78%` |
| `block20..22` | `2048` | `21.18%` | `11.50%` |

S6 is marginal after validation overhead, again with no fallback slots in these
four rows:

| Window | Context | Gross save | Validated net |
| --- | ---: | ---: | ---: |
| `block0..2` | `512` | `15.80%` | `3.31%` |
| `block20..22` | `512` | `14.95%` | `2.65%` |
| `block0..2` | `2048` | `17.29%` | `5.47%` |
| `block20..22` | `2048` | `14.52%` | `4.04%` |

S4 is negative after validation overhead:

| Window | Context | Gross save | Validated net |
| --- | ---: | ---: | ---: |
| `block0..2` | `512` | `5.12%` | `-13.32%` |
| `block20..22` | `512` | `3.15%` | `-11.77%` |
| `block0..2` | `2048` | `4.35%` | `-13.67%` |
| `block20..22` | `2048` | `3.13%` | `-14.21%` |

The broader S8 blocks=2 margin packet has zero route-set mismatch rows, one
route-order-only row, and worst `min_replay_margin=4.7e-05`. Window fallback rate
at threshold `3e-4` is `1/15 = 6.67%`.

## Decision

Replay survives only as a high-occupancy policy. S8 blocks=2 still has meaningful
headroom after the conservative validated path (`~11.5-20.3%` in this packet),
but subtracting the broader `3e-4` fallback rate narrows that to roughly
`~4.8-13.7%`. S6 is too thin unless fallback is rare and validation overhead falls;
S4 is decisively negative.

Next replay gate should be a product-shaped S8-only scheduler sketch with real
ragged occupancy traces. Do not build a general S4/S6 scheduler path from these
numbers.
