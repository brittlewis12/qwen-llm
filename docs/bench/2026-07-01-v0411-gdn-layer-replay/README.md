# v0.411 GDN Layer Replay

Goal: replace part of the decode-batching spreadsheet with a real Metal replay
of one GDN layer across multiple slots.

The new `decode-gdn-layer-replay` command compares:

- `baseline_seq`: per-slot pre RMSNorm, full `encode_gdn`, residual add, post
  RMSNorm;
- `replay_batched_qkv_z_out`: per-slot pre RMSNorm, GPU blit pack, batched
  qkv/z matmat, serial beta/alpha/GDN tail, batched out projection, residual and
  post RMSNorm.

This is a microproof, not a scheduler result. It uses real `MetalSession` GDN
state and conv mutation, but only one selected layer and full slot occupancy.

Commands:

```sh
target/release/qwen-bench decode-gdn-layer-replay \
  -m /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --tokens 1,2 --warmup 0 --iters 1 \
  > target/profiles/v0411-gdn-layer-replay-0p8b-smoke.out

target/release/qwen-bench decode-gdn-layer-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --tokens 4,8,16 --warmup 1 --iters 3 \
  > target/profiles/v0411-a3b-gdn-layer-replay-s4-s8-s16.out
```

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- 0.8B smoke with all-slot replay correctness check at `S=2`
- A3B block-0 replay with all-slot correctness check at `S=16`
- `cx ask` review, session `019f1ebe-14eb-7162-aaa5-a3d5b256eaf5`

Artifacts:

- `target/profiles/v0411-gdn-layer-replay-0p8b-smoke.out`
- `target/profiles/v0411-a3b-gdn-layer-replay-s4-s8-s16.out`

## Results

0.8B smoke correctness passes across both slots:

| Check | Value |
| --- | ---: |
| `tokens` | `2` |
| `min_cos_h` | `0.999999969` |
| `max_abs_h` | `0.001046` |

A3B block-0 correctness passes across all 16 slots:

| Check | Value |
| --- | ---: |
| `tokens` | `16` |
| `min_cos_h` | `0.999999715` |
| `max_abs_h` | `0.003791` |

A3B block-0 replay, in `ms/token` for this one-layer harness:

| S | Baseline seq | Replay | Save | Save % |
| ---: | ---: | ---: | ---: | ---: |
| `4` | `0.4264` | `0.4715` | `-0.0452` | `-10.6` |
| `8` | `0.3146` | `0.1783` | `0.1363` | `43.3` |
| `16` | `0.1862` | `0.0855` | `0.1007` | `54.1` |

## Decision

The first production-shaped GDN replay keeps the same crossover signature as the
projection-only model: `S=4` is still not enough, while `S=8/16` wins clearly even
after real state mutation, GDN tail work, residual/post-norm, and pack overhead.

Do not promote scheduler work from this alone. The next gate is representative
layer coverage and then a MoE/FFN replay: block-0 may be favorable, absolute
one-layer timings do not directly map to full-model phase rows, and perfect slot
occupancy is not a realistic serving assumption.
