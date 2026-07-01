# v0.397 Real-Prompt MoE Batch Sweep

Added `--file` support to `qwen-bench moe-batch-sweep`. When a file is supplied,
the sweep tokenizes the text, warms the first `--route-capture-ctx` tokens, and
captures the next max token window from the same real token stream. This closes
the first realism gap after the ramp-token synthetic control.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench moe-batch-sweep \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --route-capture-ctx 1024 \
  --tokens 1,2,4,8,16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0397-a3b-q4-moe-batch-sweep-the-current.out

target/release/qwen-bench moe-batch-sweep \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --route-capture-ctx 1024 \
  --tokens 1,2,4,8,16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0397-a10b-q4xl-moe-batch-sweep-the-current.out
```

## Results

| Model | Tokens | Gate/up | Down | Combined per token | Unique | Reuse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A3B Q4 | `1` | `1.0715 ms` | `0.8132 ms` | `1.8848 ms` | `8.00` | `1.00` |
| A3B Q4 | `2` | `1.7051` | `1.4433` | `1.5742` | `13.38` | `1.21` |
| A3B Q4 | `4` | `2.8452` | `2.5359` | `1.3453` | `22.02` | `1.48` |
| A3B Q4 | `8` | `5.3108` | `4.7319` | `1.2553` | `37.50` | `1.77` |
| A3B Q4 | `16` | `10.1660` | `9.2792` | `1.2153` | `57.85` | `2.30` |
| A10B Q4_XL | `1` | `3.1050` | `2.6044` | `5.7095` | `8.00` | `1.00` |
| A10B Q4_XL | `2` | `5.3008` | `4.7383` | `5.0195` | `13.72` | `1.18` |
| A10B Q4_XL | `4` | `9.4613` | `9.2400` | `4.6753` | `22.53` | `1.48` |
| A10B Q4_XL | `8` | `18.2384` | `18.5168` | `4.5944` | `39.28` | `1.75` |
| A10B Q4_XL | `16` | `34.9828` | `37.1416` | `4.5078` | `61.72` | `2.27` |

The real contiguous prompt sweep is very close to the ramp-token control. A3B
still gains about `1.55x` by t16, and A10B gains about `1.27x`. The knee remains
around t8.

## cx Review

`cx ask` session `019f1b61-4fee-7e33-b5ae-79a087a838fe` agreed this is enough to
keep batching hot, but not enough to start production multi-slot decode. The next
gate should use independent prompt files or disjoint context spans, because
production multi-slot decode lives or dies on cross-context packing efficiency.

## Decision

Do a multi-prompt or multi-context capture sweep next. Move to a thin end-to-end
multi-slot prototype only if t4/t8 real captures keep most of the MoE gain:
roughly `>=25-30%` combined MoE improvement on A3B and `>=15-20%` on A10B, with
reuse staying meaningfully above `~1.7` at t8 or `~2.2` at t16.
