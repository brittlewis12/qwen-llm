# A3B True-Long Decode Spot - v0.356

Manual qwen-only spot after v0.355. Same machine, sequential runs, no concurrent
GPU workloads, AC power. The goal was to re-anchor true-long decode behavior
after the low-bit MoE decode fixes and identify whether the next branch should
be long-attention bytes, low-bit MoE, or coverage work.

## Context Sweep

Commands used `qwen-bench ctx-sweep --checkpoints 128,8192,32768 --window 3`
with the default single max-capacity session.

| A3B row | ctx128 | ctx8192 | ctx32768 | Read |
| --- | ---: | ---: | ---: | --- |
| Q4_K_M | `107.7 t/s` | `96.5 t/s` | `76.2 t/s` | primary long slope |
| Q3_K_M | `72.4 t/s` | `67.3 t/s` | `56.7 t/s` | low-bit FFN plus attention |
| IQ4_XS | `67.4 t/s` | `62.9 t/s` | `53.6 t/s` | low-bit FFN plus attention |

## Phase Attribution

Commands used `QWEN_PHASE_MOE_FFN_SPLIT=2 qwen-bench phase --ctx 32768`.

| A3B row | phase sum | attention | routed gate/up | routed down | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| Q4_K_M | `13.41 ms` | `5.09 ms` / `38.0%` | `1.07 ms` | `0.81 ms` | long slope is attention |
| IQ4_XS | `18.56 ms` | `5.15 ms` / `27.7%` | `4.60 ms` | `2.13 ms` | low-bit MoE remains real |

## Attention Intra-Layer

Command used `qwen-bench attn-intra --ctx 32768 --runs 3` on A3B Q4_K_M.

| Phase | One layer | Extrapolated | Read |
| --- | ---: | ---: | --- |
| full attention layer | `0.5268 ms` | `5.2678 ms` | matches phase scale |
| `attn_decode_v4_main` | `0.3893 ms` | `3.8932 ms` | estimated `692 GB/s` |
| `attn_decode_v4_reduce` | `0.0340 ms` | `0.3399 ms` | not the limiter |

## Negative: Q8 KV For Group8

A dirty sidecar added group8 Q8-KV decode kernels and allowed MoE group8 to use
`QWEN_KV_Q8=1`. It was tested with `QWEN_ATTN_V4_G8_TILE=8` because the Q8 body
only covered whole-group kernels, not the default subgroup tile2 path. The code
was removed after the measurement.

| A3B Q4_K_M variant | ctx128 | ctx8192 | ctx32768 | Read |
| --- | ---: | ---: | ---: | --- |
| default F16 KV, tile2 long | `107.7` | `96.5` | `76.2` | baseline |
| F16 KV, forced tile8 | n/a | `89.8` | `62.0` | full-group shape regressed |
| Q8 KV, forced tile8 | `106.6` | `85.5` | `55.8` | byte reduction lost to dequant/shape |

Read: A3B Q4 true-long is attention-body limited, but the current main body is
already a high-throughput same-byte kernel. The simple Q8-KV group8 route is not
the byte-reduction answer, because it gives up the proven subgroup execution
shape and regresses end-to-end. Future long-attention work needs either a Q8
subgroup reader that keeps tile2 occupancy, a more structural body rewrite, or a
different byte-reduction mechanism with an end-to-end `ctx32768` gate.
