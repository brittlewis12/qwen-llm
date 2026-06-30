# v0.374 MoE Route No-Op Lower Bound

Purpose: add and exercise an unsafe route-removal lower-bound diagnostic for
decode. `QWEN_DECODE_MOE_NOOP_ROUTE=1` skips `encode_moe_route_prepare` and reuses
zero or stale route buffers. It is not correctness-preserving and must only be
used for performance attribution.

New knob:

```text
QWEN_DECODE_MOE_NOOP_ROUTE=1
```

## Measurements

```text
uv run scripts/profile/decode_ctx_sweep.py \
  --model /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --checkpoints 8192 \
  --window 16 \
  --cooldown-seconds 15 \
  --repeat-blocks 1 \
  --shuffle-seed 101 \
  --variant base \
  --variant noop-route:QWEN_DECODE_MOE_NOOP_ROUTE=1 \
  --output target/profiles/v0374-a10b-q4xl-ctx8192-noop-route.json

uv run scripts/profile/decode_ctx_sweep.py \
  --model /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --checkpoints 32768 \
  --window 16 \
  --cooldown-seconds 15 \
  --repeat-blocks 1 \
  --shuffle-seed 103 \
  --variant base \
  --variant noop-route:QWEN_DECODE_MOE_NOOP_ROUTE=1 \
  --output target/profiles/v0374-a3b-q4-ctx32768-noop-route.json
```

| Model row | Base total/GPU | No-op total/GPU | Throughput read |
| --- | ---: | ---: | --- |
| A10B `ctx8192` | `23.22/22.71 ms` | `20.86/20.37 ms` | `43.1 -> 47.9 t/s` |
| A3B `ctx32768` | `11.31/10.82 ms` | `9.78/9.32 ms` | `88.5 -> 102.3 t/s` |

Phase deltas with GDN projection/tail and MoE FFN deep split:

| Model row | Route | Routed gate/up | Routed down | Phase sum |
| --- | ---: | ---: | ---: | ---: |
| A10B base | `1.34 ms` | `3.18 ms` | `2.57 ms` | `23.85 ms` |
| A10B no-op | `0.01 ms` | `2.41 ms` | `2.68 ms` | `21.87 ms` |
| A3B base | `0.98 ms` | `1.07 ms` | `0.82 ms` | `11.82 ms` |
| A3B no-op | `0.00 ms` | `0.84 ms` | `0.79 ms` | `10.54 ms` |

Negative sidecar: a dirty exact-ish `QWEN_DECODE_MOE_SORT_TOPK=1` probe sorted
top-k ids/weights by expert id after softmax. It did not help A10B `ctx8192`:
base/sort were `23.20/23.31 ms` and `43.1/42.9 t/s`; phase showed route
topk/shared worsening `0.86 -> 0.97 ms` with routed gate/up flat.

## Decision

Keep `QWEN_DECODE_MOE_NOOP_ROUTE=1` as an explicitly unsafe diagnostic because it
proves the route-plus-consumer lower bound clears the implementation gate. Do not
interpret the whole no-op delta as route-kernel headroom: it also changes hidden
states, expert entropy, and cache locality.

Do not keep the sorted-topk sidecar. Slot order alone is not the route-consumer
win. The next route branch must be an exact route-cache replay harness or a
post-logits split that separates top-k reductions from shared-gate dot cost.
