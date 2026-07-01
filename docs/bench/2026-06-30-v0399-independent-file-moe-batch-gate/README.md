# v0.399 Independent-File MoE Batch Gate

Extended `qwen-bench moe-batch-sweep --file` so repeated `--file` entries capture
independent prompt slots. In this mode each file gets a fresh session, warms to
`--route-capture-ctx`, captures one token, and contributes one slot to the token
count sweep.

This is still a MoE projection microbench, not an end-to-end scheduler. It is,
however, the strongest current gate for whether production multi-slot batching is
likely to survive cross-document route heterogeneity.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

FILES="\
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/llm/game/the_current_ring0.md \
  --file /Users/tito/code/llm/game/the_current_ring0_v0.2.md \
  --file /Users/tito/code/llm/game/the_current_ring0_obfuscated.md \
  --file /Users/tito/code/llm/game/the_current_ring0_cipher.md \
  --file /Users/tito/code/llm/game/witness_v0_clinical.md \
  --file /Users/tito/code/llm/game/marcus_full.json \
  --file /Users/tito/code/llm/game/v02_3.6_deep.json"

target/release/qwen-bench moe-batch-sweep \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  $FILES \
  --route-capture-ctx 512 \
  --tokens 1,2,4,8 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0399-a3b-q4-moe-batch-sweep-independent8-ctx512.out

target/release/qwen-bench moe-batch-sweep \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  $FILES \
  --route-capture-ctx 512 \
  --tokens 1,2,4,8 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0399-a10b-q4xl-moe-batch-sweep-independent8-ctx512.out
```

The prompt set includes one file too short for `ctx1024`, so this gate uses
`ctx512` to keep all eight independent slots.

## Results

| Model | Tokens | Combined per token | Unique | Reuse | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| A3B Q4 | `1` | `1.8851 ms` | `8.00` | `1.00` | baseline |
| A3B Q4 | `2` | `1.4516` | `8.00` | `2.00` | strong |
| A3B Q4 | `4` | `1.3422` | `22.27` | `1.44` | strong |
| A3B Q4 | `8` | `1.2640` | `42.65` | `1.51` | robust |
| A10B Q4_XL | `1` | `5.7266` | `8.00` | `1.00` | baseline |
| A10B Q4_XL | `2` | `4.7561` | `8.00` | `2.00` | useful cap |
| A10B Q4_XL | `4` | `4.9123` | `22.83` | `1.41` | weaker than b2 |
| A10B Q4_XL | `8` | `5.2305` | `45.34` | `1.42` | too weak |

A3B passes the independent-file gate with about `1.49x` MoE projection speedup at
b8. A10B only has a clear b2 win; b4 and b8 are not production-worthy.

## cx Review

`cx ask` session `019f1b7d-9103-7872-ba93-b9ed1c4db7dd` recommended proceeding
with an A3B-targeted prototype, capping A10B at batch 2, and treating
expert-sorted A10B route replay as a separate microproof rather than a production
batching prerequisite.

## Decision

Move batching forward only as a model-specific path:

- A3B: eligible for a targeted prototype if the end-to-end scheduler preserves a
  clear b4/b8 throughput win without unacceptable latency regression.
- A10B: cap at b2 or disable above b2 by default; do not enable b4/b8 without an
  expert-sorted or route-aware microproof that beats b2 meaningfully.
