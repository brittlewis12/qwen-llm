# v0.394 Captured MoE Token-Batching Probe

Extended the captured MoE microbenches so they can replay multiple real decode
tokens from one context. This tests whether active-token batching improves the
production split gate/up and down projection kernels without relying on synthetic
expert-id patterns.

This is a microbench for the MoE FFN projection work only. It does not include
attention, GDN, scheduler overhead, KV slot management, sampling, queueing, or
multi-user route heterogeneity.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --route-capture-ctx 1024 \
  --tokens 4 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0394-a3b-q4-moe-gateup-captured-t4.out

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --route-capture-ctx 1024 \
  --tokens 4 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0394-a3b-q4-moe-down-captured-t4.out

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --route-capture-ctx 1024 \
  --tokens 16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0394-a3b-q4-moe-gateup-captured-t16.out

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --route-capture-ctx 1024 \
  --tokens 16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0394-a3b-q4-moe-down-captured-t16.out

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --route-capture-ctx 1024 \
  --tokens 4 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0394-a10b-q4xl-moe-gateup-captured-t4.out

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --route-capture-ctx 1024 \
  --tokens 4 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0394-a10b-q4xl-moe-down-captured-t4.out

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --route-capture-ctx 1024 \
  --tokens 16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0394-a10b-q4xl-moe-gateup-captured-t16.out

target/release/qwen-bench moe-down-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --route-capture-ctx 1024 \
  --tokens 16 \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0394-a10b-q4xl-moe-down-captured-t16.out
```

## Results

| Model | Tokens | Gate/up total | Down total | Combined per token | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| A3B Q4 | `1` | `1.0794 ms` | `0.8144 ms` | `1.8938 ms` | v0.391 baseline |
| A3B Q4 | `4` | `2.6265 ms` | `2.5415 ms` | `1.2920 ms` | `1.47x` per-token win |
| A3B Q4 | `16` | `9.7817 ms` | `9.3886 ms` | `1.1982 ms` | `1.58x` per-token win |
| A10B Q4_XL | `1` | `3.1068 ms` | `2.5946 ms` | `5.7014 ms` | v0.391 baseline |
| A10B Q4_XL | `4` | `8.4514 ms` | `9.5123 ms` | `4.4910 ms` | `1.27x` per-token win |
| A10B Q4_XL | `16` | `33.2432 ms` | `37.9571 ms` | `4.4500 ms` | `1.28x` per-token win |

A3B benefits strongly in both gate/up and down. A10B benefits mostly in gate/up;
Q5 down is almost flat between four and sixteen tokens, so its remaining issue is
not simple token-batch underfill.

## cx Review

`cx ask` session `019f1af1-2bf9-7e20-b717-942fd17e3c2a` agreed that captured
batching is a real MoE lever, but warned against claiming end-to-end continuous
batching speedup from MoE micros alone. Main confounds are same-context temporal
route correlation, missing scheduler/KV overhead, hidden layer variance, and
unknown expert occupancy histograms.

## Decision

Build a captured batching suite before productionizing the architecture: token
counts `1/2/4/8/16/32`, multiple prompts and contexts, per-layer timings, and
expert occupancy histograms. Keep projection-kernel work near-term because it
benefits both single-token latency and batched decode.
