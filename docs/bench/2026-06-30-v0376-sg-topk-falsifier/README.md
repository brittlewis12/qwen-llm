# v0.376 SG Top-K Route Falsifier

Purpose: test a dirty fused route kernel that kept shared-gate dot in the fused
threadgroup but replaced the threadgroup-wide top-k reductions with a simdgroup
local selector.

Dirty knob, removed after the falsifier:

```text
QWEN_DECODE_MOE_SG_TOPK=1
```

## Measurements

Correctness smoke:

```text
QWEN_DECODE_MOE_SG_TOPK=1 \
  cargo test -p qwen-llm metal_35b_a3b_moe_matches_cpu_smoke \
  --release -- --nocapture
```

Result: A3B CPU smoke passed (`argmax=11`, `max|delta|=0.0015`, `cos=1.000000`).

Phase probes:

```text
QWEN_DECODE_MOE_SG_TOPK=1 \
QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
QWEN_PHASE_MOE_ROUTE_SPLIT=1 \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0376-a10b-q4xl-phase-ctx8192-sg-topk.out

QWEN_DECODE_MOE_SG_TOPK=1 \
QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
QWEN_PHASE_MOE_ROUTE_SPLIT=1 \
  target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  > target/profiles/v0376-a3b-q4-phase-ctx32768-sg-topk.out
```

| Model row | Default topk/shared | Dirty SG topk/shared | Read |
| --- | ---: | ---: | --- |
| A10B `ctx8192` | `0.86 ms` | `1.25 ms` | regressed |
| A3B `ctx32768` | `0.68 ms` | `1.00 ms` | regressed |

## Decision

Do not keep the sidecar. The simdgroup-local selector was correctness-safe in the
A3B smoke, but it made the fused route bucket materially slower on both MoE
targets. This falsifies the simple "remove top-k threadgroup barriers" route
kernel shape.

The live route question returns to exact route-cache replay. If route remains
large under exact replay, the next kernel needs a different top-k algorithm or a
larger route/consumer dataflow change, not this SG selector.
