# v0.377 CPU Route Phase Diagnostic

Purpose: add a phase-only diagnostic that removes GPU route kernels while feeding
downstream MoE with CPU-computed route buffers. This estimates the GPU route bucket
without zero/stale-route cache artifacts.

New knob:

```text
QWEN_PHASE_MOE_CPU_ROUTE=1
```

This affects only `qwen-bench phase`. It is not production and wall time is not
meaningful because each layer reads `h` on CPU, computes route, and writes route
buffers before the downstream FFN.

## Measurements

```text
QWEN_PHASE_MOE_CPU_ROUTE=1 \
QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0377-a10b-q4xl-phase-ctx8192-cpu-route.out

QWEN_PHASE_MOE_CPU_ROUTE=1 \
QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
  target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  > target/profiles/v0377-a3b-q4-phase-ctx32768-cpu-route.out
```

| Model row | Base phase | CPU-route phase | Route GPU | Routed gate/up delta |
| --- | ---: | ---: | ---: | ---: |
| A10B `ctx8192` | `23.85 ms` | `22.63 ms` | `1.34 -> 0.00 ms` | `3.18 -> 3.34 ms` |
| A3B `ctx32768` | `11.82 ms` | `10.90 ms` | `0.98 -> 0.00 ms` | `1.07 -> 1.21 ms` |

## Decision

Keep the diagnostic, but do not call it exact route replay. It is cleaner than the
zero/stale route no-op because it computes routes from the current hidden state,
but CPU/GPU rounding or ordering can still perturb selected experts. The routed
gate/up phase changes enough that it cannot justify another route kernel by itself.

Route work now needs exact GPU route-cache replay before more implementation.
Until then, pivot to larger downstream buckets such as routed gate/up, lm-head, or
structural GDN/attention work.
