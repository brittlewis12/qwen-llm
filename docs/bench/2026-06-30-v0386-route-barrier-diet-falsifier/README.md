# v0.386 Route Barrier-Diet Falsifier

Tested and removed an env-gated exact fused route sidecar for
`kernel_topk_logits_softmax_dot_sigmoid_f32`. The sidecar kept the existing
inputs, outputs, shared-gate fusion, top-k tie semantics, and consumer layout,
but reduced obvious threadgroup barriers using simdgroup partials for shared gate
and a single owner-thread invalidation for each selected top-k slot.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

QWEN_MOE_ROUTE_TOPK_V2=1 QWEN_PHASE_MOE_ROUTE_SPLIT=1 \
  QWEN_PHASE_MOE_FFN_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0386-a10b-q4xl-phase-ctx8192-route-v2-split.out

QWEN_PHASE_MOE_ROUTE_SPLIT=1 QWEN_PHASE_MOE_FFN_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0386-a10b-q4xl-phase-ctx8192-route-default-split.out
```

The sidecar was then removed and the canonical release binary was rebuilt.

## Results

| A10B ctx8192 row | Default | Barrier-diet v2 | Delta |
| --- | ---: | ---: | ---: |
| `moe route logits` | `0.48 ms` | `0.47 ms` | noise |
| `moe route topk/shared` | `0.86 ms` | `0.82 ms` | `-0.04 ms` |
| phase sum | `23.98 ms` | `23.92 ms` | `-0.06 ms` |
| routed gate/up | `3.20 ms` | `3.21 ms` | noise |
| routed down | `2.56 ms` | `2.57 ms` | noise |

## Decision

Kill the sidecar. Exact route replay in v0.385 proves a large recoverable route
budget, but simply reducing barriers inside the fused top-k/shared kernel is far
below the production gate (`>=0.25 ms` A10B phase savings). Do not reintroduce a
barrier-diet-only route kernel without a new counter signal.

The next route attempt should change the top-k work shape more materially, such
as exact candidate compression, or move to a different proven route mechanism.
