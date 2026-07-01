# v0.412 GDN Layer Sample

Goal: test whether the v0.411 one-layer GDN replay survives representative layer
diversity, without paying repeated model loads.

`decode-gdn-layer-replay` now supports:

- `--gdn-index N[,M,...]`: measure selected GDN-layer indexes;
- `--sample-gdn-layers`: measure first, middle, and last GDN layers.

The replay shape is unchanged from v0.411: per-slot pre/post norms and GDN tail
are real, while qkv/z/out projections use packed multi-slot matmat dispatches.

Commands:

```sh
target/release/qwen-bench decode-gdn-layer-replay \
  -m /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --sample-gdn-layers --tokens 1 --warmup 0 --iters 1 \
  > target/profiles/v0412-gdn-layer-replay-sample-0p8b-smoke.out

target/release/qwen-bench decode-gdn-layer-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --sample-gdn-layers --tokens 8,16 --warmup 1 --iters 3 \
  > target/profiles/v0412-a3b-gdn-layer-replay-sample-s8-s16.out
```

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- 0.8B sampled-layer smoke
- A3B sampled-layer replay at `S=8/16`
- `cx ask` review, session `019f1ecb-cc4d-7970-b04c-a65b69b962b4`

Artifacts:

- `target/profiles/v0412-gdn-layer-replay-sample-0p8b-smoke.out`
- `target/profiles/v0412-a3b-gdn-layer-replay-sample-s8-s16.out`

## Results

A3B sampled-layer correctness at `S=16`:

| Block | GDN index | `min_cos_h` | `max_abs_h` |
| ---: | ---: | ---: | ---: |
| `0` | `0` | `0.999999715` | `0.003791` |
| `20` | `15` | `0.999999840` | `0.002869` |
| `38` | `29` | `0.999999821` | `0.005905` |

A3B sampled-layer replay, in `ms/token`:

| Block | GDN index | S | Baseline seq | Replay | Save | Save % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0` | `8` | `0.1488` | `0.1160` | `0.0327` | `22.0` |
| `0` | `0` | `16` | `0.1540` | `0.0747` | `0.0793` | `51.5` |
| `20` | `15` | `8` | `0.1416` | `0.1064` | `0.0352` | `24.9` |
| `20` | `15` | `16` | `0.1428` | `0.0741` | `0.0687` | `48.1` |
| `38` | `29` | `8` | `0.1376` | `0.1028` | `0.0348` | `25.3` |
| `38` | `29` | `16` | `0.1380` | `0.0697` | `0.0682` | `49.4` |

## Decision

GDN replay economics are no longer a block-0-only artifact. The sampled layers
all pass all-slot correctness and show a similar positive crossover at `S=8/16`.
The more representative `S=8` save is about `0.033-0.035 ms/token/layer`; a naive
30-layer extrapolation is about `1.0 ms/token`, still before attention, MoE, and
ragged occupancy charges.

Next gate: an integrated multi-block decode slice with ragged masks. More isolated
GDN layer sampling is now lower value than proving that savings survive command
structure, layer interactions, and non-full active-slot occupancy.
