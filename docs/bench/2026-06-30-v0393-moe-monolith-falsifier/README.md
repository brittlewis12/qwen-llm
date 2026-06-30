# v0.393 Captured MoE Monolith Falsifier

Added a captured microbench lane for the existing one-token Q4/Q5 routed FFN
monolith. The goal was not to promote it by default, but to directly falsify the
recurring hypothesis that removing the routed intermediate buffer is enough to
beat the production split gate/up plus down path.

The result is decisive: the monolith is tens of times slower than the split path.
It collapses parallelism and streams active expert weights at only `7-10 GB/s`,
so this is not a tuning near-miss.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --route-capture-ctx 1024 \
  --fused-routed-q4q5 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0393-a3b-q4-moe-fused-q4q5-captured.out

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --route-capture-ctx 1024 \
  --fused-routed-q4q5 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0393-a10b-q4xl-moe-fused-q4q5-captured.out
```

## Results

| Model | Split captured gate/up+down | Fused monolith | Active weight GB/s |
| --- | ---: | ---: | ---: |
| A3B Q4 | `1.0794 + 0.8144 = 1.8938 ms` | `80.5129 ms` | `7.6` |
| A10B Q4_XL | `3.1068 + 2.5946 = 5.7014 ms` | `210.1424 ms` | `10.2` |

The split-path active weight bandwidth implied by the captured rows is roughly
`321 GB/s` on A3B and `376 GB/s` on A10B, so the monolith is about `40x` below
the production dataflow on the same captured routes.

## cx Review

`cx ask` session `019f1ad5-c571-7ab1-b141-6ce8fccb787d` agreed that this rules
out the current whole-routed-FFN monolith class, not MoE decode work broadly. The
highest-value next probes are projection-kernel throughput, active token/expert
batching, and only later tiled fusion that preserves the split path's parallel
decomposition.

## Decision

Do not pursue the existing monolith as a production path. Any future fusion must
preserve row/tile/expert parallelism and should first clear a captured MoE gate:
`>=10%` improvement for full captured gate/up plus down on A3B and A10B, or a
clear single-model objective. Toy wins that lose captured replay remain dead.
