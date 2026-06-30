# v0.380 LM Argmax Phase Diagnostic

Purpose: measure the exact greedy argmax pass after `lm_head` so lm-head fusion
ideas are bounded by evidence instead of the full lm-head bucket.

New knob:

```text
QWEN_PHASE_LM_ARGMAX=1
```

This affects only `qwen-bench phase`; it appends an `lm argmax` row after the
existing `lm head` mat-vec row.

## Measurements

```text
QWEN_PHASE_LM_ARGMAX=1 \
QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
QWEN_PHASE_MOE_ROUTE_SPLIT=1 \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0380-a10b-q4xl-phase-ctx8192-lm-argmax-v2.out

QWEN_PHASE_LM_ARGMAX=1 \
QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
QWEN_PHASE_MOE_ROUTE_SPLIT=1 \
  target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  > target/profiles/v0380-a3b-q4-phase-ctx32768-lm-argmax-v2.out
```

| Model row | lm head | lm argmax | Read |
| --- | ---: | ---: | --- |
| A10B `ctx8192` | `1.57 ms` | `0.06 ms` | argmax is too small |
| A3B `ctx32768` | `0.82 ms` | `0.05 ms` | argmax is too small |

## Decision

Keep the diagnostic, but do not prioritize exact greedy `lm_head+argmax` fusion.
The removable argmax pass is far below the implementation gate, and logits writes
are only about one megabyte. The live lm-head bucket is the Q6_K projection weight
read, not the sampler pass.
