# v0.385 Exact GPU Route Replay

Adds `QWEN_PHASE_MOE_ROUTE_REPLAY=1` for `qwen-bench phase`. In this diagnostic,
the real GPU route kernels still run after each layer's mixer to populate exact
top-k, weight, and shared-gate buffers, but their GPU time is excluded from the
phase sum. This gives an exact route-free lower bound while preserving downstream
MoE consumers, unlike the earlier CPU-route diagnostic.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

QWEN_PHASE_MOE_FFN_SPLIT=deep target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0382-a10b-q4xl-phase-ctx8192-ffn-deep.out

QWEN_PHASE_MOE_ROUTE_REPLAY=1 QWEN_PHASE_MOE_FFN_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0385-a10b-q4xl-phase-ctx8192-route-replay.out

QWEN_PHASE_MOE_FFN_SPLIT=deep target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  > target/profiles/v0384-a3b-q4-phase-ctx32768-ffn-deep.out

QWEN_PHASE_MOE_ROUTE_REPLAY=1 QWEN_PHASE_MOE_FFN_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  > target/profiles/v0385-a3b-q4-phase-ctx32768-route-replay.out
```

## Results

| Row | Default phase sum | Replay phase sum | Route row | Read |
| --- | ---: | ---: | ---: | --- |
| A10B ctx8192 | `23.98 ms` | `22.62 ms` | `1.33 -> 0.00 ms` | clean lower bound |
| A3B ctx32768 | `11.76 ms` | `10.75 ms` | `0.95 -> 0.00 ms` | clean lower bound |

Downstream consumers stayed stable:

| Row | Gate/up | Down | Shared gate/up | Shared down |
| --- | ---: | ---: | ---: | ---: |
| A10B default | `3.19 ms` | `2.57 ms` | `0.84 ms` | `0.66 ms` |
| A10B replay | `3.18 ms` | `2.58 ms` | `0.84 ms` | `0.66 ms` |
| A3B default | `1.07 ms` | `0.82 ms` | `0.35 ms` | `0.40 ms` |
| A3B replay | `1.06 ms` | `0.81 ms` | `0.34 ms` | `0.40 ms` |

## Decision

Route is now a real, exact, recoverable decode bucket. The earlier CPU-route path
was useful but perturbed downstream phases; this diagnostic does not. Production
route work is now justified, but it must target the post-logits top-k/shared route
preparation rather than router logits alone, and it must preserve the consumer
phase stability shown here.

Promotion gate for a production route branch:

- save at least `0.25 ms` on A10B ctx8192 or `0.15 ms` on A3B true-long;
- keep routed gate/up and routed down within noise;
- show at least `1%` end-to-end decode movement on the relevant row;
- pass normal correctness gates, not just phase replay.
