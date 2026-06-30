# v0.390 Route Structural Audit

Rechecked exact route replay and decomposed the remaining route boundary after
the v0.386/v0.387 local route-kernel falsifiers and the v0.389 counter-tooling
probe. The question was whether the exact route budget still justifies another
implementation branch, or whether the remaining exact boundary is structurally
unattractive.

## Commands

```bash
QWEN_PHASE_MOE_ROUTE_SPLIT=1 \
  QWEN_PHASE_MOE_FFN_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 1024 \
  > target/profiles/v0390-a3b-q4-phase-ctx1024-route-split.out

QWEN_PHASE_MOE_ROUTE_REPLAY=1 \
  QWEN_PHASE_MOE_FFN_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 1024 \
  > target/profiles/v0390-a3b-q4-phase-ctx1024-route-replay.out

QWEN_PHASE_MOE_ROUTE_SPLIT=deep \
  QWEN_PHASE_MOE_FFN_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 1024 \
  > target/profiles/v0390-a3b-q4-phase-ctx1024-route-deep.out

QWEN_PHASE_MOE_ROUTE_SPLIT=1 \
  QWEN_PHASE_MOE_FFN_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 1024 \
  > target/profiles/v0390-a10b-q4xl-phase-ctx1024-route-split.out

QWEN_PHASE_MOE_ROUTE_REPLAY=1 \
  QWEN_PHASE_MOE_FFN_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 1024 \
  > target/profiles/v0390-a10b-q4xl-phase-ctx1024-route-replay.out
```

## Results

| Model | Phase | Logits | Topk/shared | Replay phase | Replay delta | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A3B Q4 | `9.90 ms` | `0.30 ms` | `0.68 ms` | `8.89 ms` | `1.01 ms` | route budget repeats |
| A10B Q4_XL | `22.80 ms` | `0.48 ms` | `0.86 ms` | `21.46 ms` | `1.34 ms` | route budget repeats |

A3B deep route split shows the production topk/shared fusion is already doing the
important structural overlap:

| A3B route mode | Logits | Topk | Shared gate | Total route rows |
| --- | ---: | ---: | ---: | ---: |
| production split | `0.30 ms` | n/a | n/a | `0.98 ms` |
| deep split | `0.31 ms` | `0.55 ms` | `0.88 ms` | `1.74 ms` |

## Decision

Demote exact route implementation work from the main branch. Route replay still
proves a real upper bound, but the production path already fuses the high-value
topk/shared half. The only obvious exact boundary left is `router_logits ->
topk/shared`: the logits tensor is tiny, while exact fusion would require either
a single underfilled route kernel, global atomics, or another reduction protocol
because router mat-vec currently uses many threadgroups and exact top-k needs a
global selection boundary.

Do not reopen local top-k route variants. Reopen exact route only if a concrete
prototype can save at least `0.4 ms` on A3B or `0.5 ms` on A10B route total
without moving time into consumers. The next implementation branch should pivot
to captured MoE gate/up/down microbenching.
