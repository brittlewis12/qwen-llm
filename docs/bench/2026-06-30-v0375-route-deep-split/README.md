# v0.375 MoE Route Deep Split

Purpose: split the post-logits route bucket one level deeper. This adds a
phase-only parallel top-k kernel so `QWEN_PHASE_MOE_ROUTE_SPLIT=deep` reports
router logits, top-k/softmax, and shared-gate dot separately.

New mode:

```text
QWEN_PHASE_MOE_ROUTE_SPLIT=deep
```

This affects only `qwen-bench phase`. It does not change the normal decode path.

## Measurements

```text
QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
QWEN_PHASE_MOE_ROUTE_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0375-a10b-q4xl-phase-ctx8192-route-deep.out

QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
QWEN_PHASE_MOE_ROUTE_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  > target/profiles/v0375-a3b-q4-phase-ctx32768-route-deep.out
```

| Model row | logits | topk-only | shared gate | Prior fused topk/shared |
| --- | ---: | ---: | ---: | ---: |
| A10B `ctx8192` | `0.48 ms` | `0.67 ms` | `1.50 ms` | `0.86 ms` |
| A3B `ctx32768` | `0.32 ms` | `0.60 ms` | `1.04 ms` | `0.68 ms` |

## Decision

Keep the deep split as attribution only. The separate shared-gate kernel is much
slower than the fused route kernel, so split wall time is not an optimization
candidate. The useful signal is that the fused kernel is doing substantial work
inside top-k reductions/barriers while making the shared-gate dot cheap in the
same threadgroup.

Next route implementation should not be `topk-only + shared-gate` split. The
credible exact kernels are either a simdgroup-local top-k inside the fused route
kernel, or an exact route-cache replay harness that proves the route bucket remains
large with realistic per-layer route buffers.
