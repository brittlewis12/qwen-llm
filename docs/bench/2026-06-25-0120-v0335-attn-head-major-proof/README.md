# Attention Head-Major KV Proof - v0.335

Synthetic v4 attention microproof for the proposed head-major F16 K/V layout.
This is an ignored test/bench harness only: production cache layout is unchanged.
The proof isolates K/V address layout by comparing the canonical token-major
layout `[pos, kv_head, dim]` against a head-major side buffer
`[kv_head, pos, dim]` for the current long MoE subgroup kernels.

## Command

```sh
cargo test -p qwen-llm attn_v4_head_major_main_reduce_breakdown_moe_shapes \
  --release -- --ignored --nocapture --test-threads=1
```

## Correctness

All compared shapes matched exactly after the shared reduce path:

| Shape | Context | cos | max abs |
| --- | ---: | ---: | ---: |
| A3B group8/tile2/C64 | `8192` | `1.00000000` | `0.000e0` |
| A3B group8/tile2/C64 | `16384` | `1.00000000` | `0.000e0` |
| A3B group8/tile2/C64 | `32768` | `1.00000000` | `0.000e0` |
| A10B group16/tile4/C64 | `8192` | `1.00000000` | `0.000e0` |
| A10B group16/tile4/C64 | `16384` | `1.00000000` | `0.000e0` |
| A10B group16/tile4/C128 | `32768` | `1.00000000` | `0.000e0` |

## Main-Body Timing

| Shape | Context | token-major per call | head-major per call | Read |
| --- | ---: | ---: | ---: | --- |
| A3B group8/tile2/C64 | `8192` | `0.145 ms` | `0.080 ms` | isolated win |
| A3B group8/tile2/C64 | `16384` | `0.191 ms` | `0.193 ms` | flat/slower |
| A3B group8/tile2/C64 | `32768` | `0.384 ms` | `0.389 ms` | flat/slower |
| A10B group16/tile4/C64 | `8192` | `0.092 ms` | `0.097 ms` | slower |
| A10B group16/tile4/C64 | `16384` | `0.215 ms` | `0.218 ms` | flat/slower |
| A10B group16/tile4/C128 | `32768` | `0.434 ms` | `0.436 ms` | flat/slower |

## Interpretation

- The layout transform is correctness-safe in isolation.
- It does not pass the productionization gate: the win is only A3B at `ctx8192`,
  while A3B true-long and all A10B rows are flat to slower.
- Do not build a production head-major KV sidecar from this evidence. Reopen only
  with a new counter signal or a body rewrite that changes more than address order.
