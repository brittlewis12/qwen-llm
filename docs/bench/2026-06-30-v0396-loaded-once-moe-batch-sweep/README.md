# v0.396 Loaded-Once Captured MoE Batch Sweep

Added `qwen-bench moe-batch-sweep` so captured MoE token-batching studies no
longer require a separate model load and route capture per token count. The new
command captures the max token window once, then times gate/up and down for all
requested token counts in one process.

The sweep uses the ramp token pattern by default. That keeps zero-token replay
available as a locality stress case while making the default timing less biased
toward repeated-token expert reuse.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench moe-batch-sweep \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --route-capture-ctx 1024 \
  --route-capture-token-pattern ramp \
  --tokens 1,2,4,8,16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0396-a3b-q4-moe-batch-sweep-ramp.out

target/release/qwen-bench moe-batch-sweep \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --route-capture-ctx 1024 \
  --route-capture-token-pattern ramp \
  --tokens 1,2,4,8,16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0396-a10b-q4xl-moe-batch-sweep-ramp.out
```

## Results

| Model | Tokens | Gate/up | Down | Combined per token | Gate/up GB/s | Down GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A3B Q4 | `1` | `1.0771 ms` | `0.8120 ms` | `1.8891 ms` | `350.5` | `284.1` |
| A3B Q4 | `2` | `1.6822` | `1.4383` | `1.5602` | `448.8` | `320.8` |
| A3B Q4 | `4` | `2.8381` | `2.5300` | `1.3420` | `532.0` | `364.7` |
| A3B Q4 | `8` | `5.2512` | `4.7348` | `1.2483` | `575.1` | `389.8` |
| A3B Q4 | `16` | `10.0761` | `9.3069` | `1.2114` | `599.4` | `396.6` |
| A10B Q4_XL | `1` | `3.0804` | `2.5784` | `5.6588` | `432.0` | `315.4` |
| A10B Q4_XL | `2` | `5.3835` | `4.6155` | `4.9995` | `494.3` | `352.4` |
| A10B Q4_XL | `4` | `9.5590` | `9.0613` | `4.6551` | `556.8` | `359.0` |
| A10B Q4_XL | `8` | `17.9925` | `18.4918` | `4.5605` | `591.6` | `351.8` |
| A10B Q4_XL | `16` | `34.5075` | `37.1931` | `4.4813` | `617.0` | `349.8` |

A3B gains `1.56x` in captured MoE projection time by t16. A10B gains `1.26x`,
but its Q5 down path is almost flat after t2/t4, so further batching does not fix
that kernel's remaining work-unit limit.

## cx Review

`cx ask` session `019f1b57-0285-7be2-9967-71cdfe33fd22` agreed that batching
belongs on the roadmap, but not yet as the main production architecture branch.
The useful knee is around t8: t16 adds only `~3%` on A3B and `~2%` on A10B. The
next evidence should close the realism gap with real-prompt and multi-context
captures plus packed-slot shape controls.

## Decision

Use `moe-batch-sweep` as the cheap loaded-once decision tool. Start production
multi-slot decode only after representative real captures reproduce at least a
`25-30%` MoE combined gain on A3B and `15-20%` on A10B at t4/t8, and an
end-to-end decode prototype still clears `>=15%` at t4 or `>=20%` at t8 after
scheduler, KV, attention, sampling, and packing/scatter costs.
