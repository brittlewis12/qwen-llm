# v0.373 MoE Route Split Attribution

Purpose: split the aggregate MoE route phase into router-logits work and the
post-logits top-k/shared-weight preparation. This is phase-only instrumentation
for `qwen-bench phase`.

New knob:

```text
QWEN_PHASE_MOE_ROUTE_SPLIT=1
```

When enabled, `moe route` is reported as:

- `moe route logits`
- `moe route topk/shared`

## Measurements

```text
QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
QWEN_PHASE_MOE_ROUTE_SPLIT=1 \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0373-a10b-q4xl-phase-ctx8192-route-split.out

QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
QWEN_PHASE_MOE_ROUTE_SPLIT=1 \
  target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  > target/profiles/v0373-a3b-q4-phase-ctx32768-route-split.out
```

| Model row | logits | topk/shared | Read |
| --- | ---: | ---: | --- |
| A10B `ctx8192` | `0.48 ms` | `0.86 ms` | post-logits work is the larger route bucket |
| A3B `ctx32768` | `0.30 ms` | `0.68 ms` | topk/shared is comparable to routed down |

## Decision

Keep the diagnostic. Route logits remain bounded after the earlier router work,
but post-logits route preparation is no longer ignorable: it is larger than logits
and close to or above several named MoE/GDN tail buckets.

Do not treat route-logits mat-vecs as the next branch. If route work is reopened,
the credible target is the top-k/shared-weight/slot-prep path or its downstream
layout effect on routed gate/up and down. Require a split of that sub-bucket before
implementation, or a dirty no-op/cached-route lower bound that shows at least
`0.15 ms` recoverable on A10B or `0.10 ms` on A3B.
