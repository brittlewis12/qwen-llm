# v0.398 Strided Real-Context MoE Batch Gate

Added `--route-capture-stride` to `qwen-bench moe-batch-sweep`. With a file
prompt, the sweep now can capture tokens at `ctx + i * stride` while executing
the intervening prompt tokens normally. This samples disjoint positions from one
real context instead of only adjacent tokens.

This still is not true independent multi-user slots, but it is a stronger gate
than contiguous prompt capture because adjacent-token route locality can overstate
batching wins.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench moe-batch-sweep \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --route-capture-ctx 1024 \
  --route-capture-stride 128 \
  --tokens 1,2,4,8,16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0398-a3b-q4-moe-batch-sweep-the-current-stride128.out

target/release/qwen-bench moe-batch-sweep \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --route-capture-ctx 1024 \
  --route-capture-stride 128 \
  --tokens 1,2,4,8,16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0398-a10b-q4xl-moe-batch-sweep-the-current-stride128.out
```

`the_current.md` has `3441` tokens for these models, so stride `256` is too long
for `ctx1024` plus `16` captured tokens. Stride `128` reaches the stronger gate
without needing a different prompt source.

## Results

| Model | Tokens | Combined per token | Unique | Reuse | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| A3B Q4 | `1` | `1.8704 ms` | `8.00` | `1.00` | baseline |
| A3B Q4 | `2` | `1.6222` | `15.43` | `1.04` | small gain |
| A3B Q4 | `4` | `1.3788` | `25.57` | `1.26` | useful |
| A3B Q4 | `8` | `1.2722` | `45.12` | `1.43` | strong |
| A3B Q4 | `16` | `1.2194` | `72.25` | `1.80` | robust |
| A10B Q4_XL | `1` | `5.7513` | `8.00` | `1.00` | baseline |
| A10B Q4_XL | `2` | `5.3568` | `15.55` | `1.03` | modest |
| A10B Q4_XL | `4` | `5.1531` | `26.23` | `1.24` | peak useful |
| A10B Q4_XL | `8` | `5.2007` | `46.60` | `1.41` | no further gain |
| A10B Q4_XL | `16` | `5.5098` | `76.43` | `1.77` | regresses |

A3B remains robust under strided contexts. A10B no longer resembles the
contiguous prompt result: the t16 row worsens from contiguous `4.5078 ms/token`
to strided `5.5098 ms/token`.

## cx Review

`cx ask` session `019f1b6f-2750-7b10-ba2a-49432b48810b` recommended demoting
generic production batching. A3B likely benefits; A10B needs independent
multi-document validation plus route-aware packing or kernel-shape work before it
can justify production batching.

## Decision

Do not promote a generic production multi-slot decode branch from contiguous MoE
micros. The next gates are:

- independent multi-document slots, not one adjacent document span;
- A10B gate/up versus down sensitivity under original, sorted, high-reuse, and
  low-reuse route maps;
- a model-specific policy where A3B batching may advance but A10B is capped,
  disabled, or route-aware until it clears the strided/independent gate.
