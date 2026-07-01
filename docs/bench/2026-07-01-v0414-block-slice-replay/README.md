# v0.414 Block Slice Replay

Goal: move from GDN-only replay into a real MoE block slice. The new
`decode-block-slice-replay` command keeps normal attention and normal MoE
route/FFN execution in place, replacing only the GDN mixer subpath with packed
multi-slot qkv/z/out replay.

Commands:

```sh
target/release/qwen-bench decode-block-slice-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --blocks 2 --tokens 1 --warmup 0 --iters 1 \
  > target/profiles/v0414-a3b-block-slice-replay-smoke.out

target/release/qwen-bench decode-block-slice-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --blocks 4 --tokens 8,16 --warmup 1 --iters 3 \
  > target/profiles/v0414-a3b-block-slice-replay-b4-s8-s16.out
```

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- A3B two-block `S=1` smoke
- A3B four-block block-slice replay at `S=8/16`
- `cx ask` review, session `019f1edf-f3df-7e73-9b44-a9dbeceb6278`

Artifacts:

- `target/profiles/v0414-a3b-block-slice-replay-smoke.out`
- `target/profiles/v0414-a3b-block-slice-replay-b4-s8-s16.out`

## Results

A3B block slice: `start_block=0`, `blocks=4`, with `3` GDN blocks and `1`
attention block at synthetic `position=0`.

Correctness after all four real blocks across `S=16` slots:

| Check | Value |
| --- | ---: |
| `min_cos_x` | `0.999999329` |
| `max_abs_x` | `0.002357` |

Replay timing, in `ms/token` for the block slice:

| S | Baseline seq | Replay | Save | Save % |
| ---: | ---: | ---: | ---: | ---: |
| `8` | `0.8986` | `0.7577` | `0.1409` | `15.7` |
| `16` | `0.8733` | `0.6834` | `0.1899` | `21.7` |

## Decision

The GDN replay path survives the first integrated block-slice gate: normal
attention and normal MoE route/FFN execution can coexist with GDN replay, and the
net slice still wins at `S=8/16`.

This is still not scheduler-grade. It uses an early block window, full occupancy,
and position-0 attention. The next gate should test nonzero context positions,
mid/late windows, and ragged active slots before any production scheduler work.
