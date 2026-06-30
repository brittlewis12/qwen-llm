# v0.371 A10B Q5 Down K1024 R2 Falsifier

Purpose: test the closest remaining Q5 routed-down analog to the banked A3B
`f_exp=512` R2 win. A dirty `f_exp=1024` sidecar packed two output rows per
simdgroup and looped over the two 512-wide K halves inside each routed slot.

This was not retained. It is exact, but it misses the single-token decode gate.

## Gate

- A10B `moe-down-micro --tokens 1`: improve `2.483 ms` to `<=2.20 ms`, preferably
  `<=2.10 ms`.
- A10B `moe-down-micro --tokens 16`: improve `36.04 ms` to `<=31.5 ms`, or reach
  about `>=410 GB/s` active-weight bandwidth.
- Only run full A10B phase/decode if the token-1 primitive predicts at least
  `0.25 ms` routed-down phase reduction.

## Measurements

Baseline current default:

```text
target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --iters 20 --warmup 5 --tokens 1 \
  > target/profiles/v0370-a10b-q4xl-moe-down-micro-t1.out

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --iters 10 --warmup 3 --tokens 16 \
  > target/profiles/v0370-a10b-q4xl-moe-down-micro-t16.out
```

Dirty sidecar:

```text
target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --iters 20 --warmup 5 --tokens 1 --k1024-r2 --check-k1024-r2 \
  > target/profiles/v0371-a10b-q4xl-moe-down-micro-k1024r2-t1.out

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --iters 10 --warmup 3 --tokens 16 --k1024-r2 \
  > target/profiles/v0371-a10b-q4xl-moe-down-micro-k1024r2-t16.out
```

| A10B Q5 down micro | Default | Dirty K1024 R2 | Read |
| --- | ---: | ---: | --- |
| `tokens=1` GPU | `2.4830 ms` | `2.4242 ms` | `1.024x` |
| `tokens=1` BW | `327.5 GB/s` | `335.4 GB/s` | too small |
| `tokens=16` GPU | `36.0421 ms` | `31.9546 ms` | `1.128x` |
| `tokens=16` BW | `361.0 GB/s` | `407.2 GB/s` | near gate |

Correctness: `--check-k1024-r2` reported `max_abs=0`, `cos=1.0`.

## Decision

Kill the branch for decode. The batched primitive improves materially, but the
production-relevant single-token primitive only saves `0.059 ms` across all Q5
routed-down expert banks. That cannot plausibly clear the `>=0.25 ms` A10B phase
gate, so a full long-ramp phase/decode A/B would be wasted GPU time.

Read: A10B `f_exp=1024` Q5 down is not the same problem as A3B `f_exp=512`. The
banked K512 R2 mechanism does not extend strongly enough to single-token K1024;
future Q5 down work needs a different dataflow/counter signal, not another row
packing variant.
