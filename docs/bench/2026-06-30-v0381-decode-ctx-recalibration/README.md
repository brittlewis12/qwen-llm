# v0.381 Decode Context Recalibration

Purpose: run a bounded current-HEAD decode context sweep after the route, gate/up,
and lm-head falsifiers. This re-ranks active bottlenecks before adding more
specialized harnesses.

## Measurements

```text
target/release/qwen-bench ctx-sweep \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --checkpoints 128,8192,32768 \
  --window 16 \
  > target/profiles/v0381-a3b-q4-ctx-sweep-128-8192-32768.out

target/release/qwen-bench ctx-sweep \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --checkpoints 128,8192,16384 \
  --window 16 \
  > target/profiles/v0381-a10b-q4xl-ctx-sweep-128-8192-16384.out

target/release/qwen-bench ctx-sweep \
  -m /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --checkpoints 128,8192 \
  --window 16 \
  > target/profiles/v0381-27b-q4-ctx-sweep-128-8192.out
```

| Model | Context | total ms | GPU ms | t/s |
| --- | ---: | ---: | ---: | ---: |
| A3B Q4_K_M | `128` | `9.28` | `8.80` | `107.8` |
| A3B Q4_K_M | `8192` | `10.32` | `9.84` | `96.9` |
| A3B Q4_K_M | `32768` | `11.27` | `10.80` | `88.7` |
| A10B Q4_XL | `128` | `22.13` | `21.65` | `45.2` |
| A10B Q4_XL | `8192` | `23.26` | `22.75` | `43.0` |
| A10B Q4_XL | `16384` | `24.56` | `24.04` | `40.7` |
| 27B Q4_K_M | `128` | `39.11` | `38.65` | `25.6` |
| 27B Q4_K_M | `8192` | `43.15` | `42.64` | `23.2` |

## Decision

The current slope is orderly, not cliff-shaped. MoE still has meaningful
long-context attention growth, but A10B remains dominated by non-attention work at
`ctx8192/16384`. Dense 27B loses about `9%` from `ctx128` to `ctx8192`, so dense
decode should stay in the guardrail set while MoE work continues.

Next active implementation should not be another route or row-shape microkernel.
Use the split rows plus this sweep to target either captured-route gate/up replay
or a structural byte-reduction branch with a clear phase budget.
