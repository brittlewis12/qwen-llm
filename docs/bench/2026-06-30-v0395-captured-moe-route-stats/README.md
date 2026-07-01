# v0.395 Captured MoE Route-Stats Control

Added route occupancy stats to the captured MoE microbenches and a deterministic
`--route-capture-token-pattern ramp` control. The previous multi-token capture
used `token_id=0` for every captured token, which is useful as a locality stress
case but can overstate expert reuse.

The important update: batching still helps under varied token ids, and the t16
timings barely move even when route reuse drops sharply. The safer mechanism is
packed-slot shape and occupancy, not repeated-token expert reuse.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --route-capture-ctx 1024 \
  --tokens 16 \
  --warmup 1 \
  --iters 1 \
  > target/profiles/v0395-a3b-q4-moe-gateup-captured-t16-stats.out

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --route-capture-ctx 1024 \
  --tokens 16 \
  --warmup 1 \
  --iters 1 \
  > target/profiles/v0395-a10b-q4xl-moe-gateup-captured-t16-stats.out

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --route-capture-ctx 1024 \
  --route-capture-token-pattern ramp \
  --tokens 16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0395-a3b-q4-moe-gateup-captured-t16-ramp.out

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --route-capture-ctx 1024 \
  --route-capture-token-pattern ramp \
  --tokens 16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0395-a3b-q4-moe-down-captured-t16-ramp.out

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --route-capture-ctx 1024 \
  --route-capture-token-pattern ramp \
  --tokens 16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0395-a10b-q4xl-moe-gateup-captured-t16-ramp.out

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --route-capture-ctx 1024 \
  --route-capture-token-pattern ramp \
  --tokens 16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0395-a10b-q4xl-moe-down-captured-t16-ramp.out
```

## Results

Route stats are averaged over eligible layers. `slots_per_layer` is
`tokens * topk`; `avg_reuse` is `slots_per_layer / unique_experts` per layer,
then averaged.

| Model | Pattern | Slots | Avg unique | Avg max | Avg reuse | Gate/up | Down |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A3B | zero | `128` | `19.52` | `15.40` | `8.22` | `9.7817 ms` | `9.3886 ms` |
| A3B | ramp | `128` | `50.30` | `12.68` | `2.72` | `10.0725 ms` | `9.3001 ms` |
| A10B | zero | `128` | `8.34` | `16.00` | `15.42` | `33.2432 ms` | `37.9571 ms` |
| A10B | ramp | `128` | `54.40` | `13.53` | `2.45` | `34.5369 ms` | `37.3997 ms` |

Gate/up gets slightly worse under ramp (`~3-4%`), while down is flat to slightly
better. That is too small relative to the reuse collapse to make reuse the main
mechanism.

## cx Review

`cx ask` session `019f1b0a-9f3a-76e0-a09f-8ac837ad5556` concluded that zero
captures were contaminated by repeated-token locality. Use ramp as the default
synthetic timing control. Expert-major batching may still help, but the next
kernel question should be packed-slot shape and projection occupancy, not reuse
alone.

## Decision

Treat token batching as an occupancy/shape lever first. Expert reuse/locality is
second-order until real prompt traces and histograms show otherwise. Any expert-
major branch should beat the current packed-slot path under ramp or real-prompt
captures including grouping/scatter overhead; synthetic high-reuse wins alone are
not promotion evidence.
