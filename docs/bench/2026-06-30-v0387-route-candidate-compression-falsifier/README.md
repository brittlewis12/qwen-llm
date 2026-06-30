# v0.387 Route Candidate-Compression Falsifier

Tested and removed an exact fused route sidecar that changed the top-k work
shape more materially than v0.386. Each simdgroup computed an exact local top-k
over 32 experts, then one thread merged the 8 simdgroup candidate lists into the
global exact top-k. Shared-gate fusion and output layout were preserved.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

QWEN_MOE_ROUTE_TOPK_CAND=1 QWEN_PHASE_MOE_ROUTE_SPLIT=1 \
  QWEN_PHASE_MOE_FFN_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0387-a10b-q4xl-phase-ctx8192-route-cand-split.out
```

The sidecar was then removed and the canonical release binary was rebuilt.

## Results

| A10B ctx8192 row | Default | Candidate compression | Delta |
| --- | ---: | ---: | ---: |
| `moe route logits` | `0.48 ms` | `0.48 ms` | noise |
| `moe route topk/shared` | `0.86 ms` | `3.87 ms` | `+3.01 ms` |
| phase sum | `23.98 ms` | `27.05 ms` | `+3.07 ms` |
| routed gate/up | `3.20 ms` | `3.21 ms` | noise |
| routed down | `2.56 ms` | `2.61 ms` | noise |

## Decision

Kill the candidate-compression route sidecar. The single-thread local/merge work
shape is far worse than the current iterative threadgroup reductions. Combined
with v0.376 and v0.386, this closes the obvious exact top-k kernel variants:
single-simdgroup selection, barrier-only cleanup, and simdgroup candidate lists.

Route remains exactly recoverable per v0.385, but the next production attempt
needs a new mechanism rather than another local top-k rewrite.
