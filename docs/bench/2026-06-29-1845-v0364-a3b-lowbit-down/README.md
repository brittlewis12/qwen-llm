# A3B Low-Bit IQ4 Down — v0.364

Clean A3B low-bit decode checkpoint after defaulting the fast IQ4_XS routed-down
dataflow. Same physical box, sequential runs, no concurrent GPU benches.

## Identity

- `qwen-llm`: `6c6ce6941`
- `llama.cpp`: `c818263f2a` (build 9833, backends `MTL,BLAS`)
- GPU: `Apple M4 Max`
- Rollback: `QWEN_DECODE_MOE_IQ4_DOWN_FAST=0`

## Results

| A3B row | llama.cpp `tg128` | qwen `tg128` | Read |
| --- | ---: | ---: | --- |
| Q3_K_M | `81.54 t/s` | `103.27 t/s` (`1.27x`) | low-bit decode now strong |
| UD-IQ4_XS | `80.46 t/s` | `102.36 t/s` (`1.27x`) | low-bit decode now strong |
| Q4_K_M guard | n/a | `107.26 t/s` | guard unchanged |

## Phase Check

Clean `ctx128` deep-split phase profile:

| Row | Fast routed down | Rollback routed down | Phase read |
| --- | ---: | ---: | --- |
| Q3_K_M | `0.64 ms` | `2.13 ms` | `-70%` routed down |
| UD-IQ4_XS | `0.65 ms` | `2.12 ms` | `-69%` routed down |

After v0.362 and v0.364, the low-bit MoE FFN buckets are no longer dominant:
Q3 routed gate/up/down are `1.08/0.64 ms`, and IQ4 routed gate/up/down are
`1.13/0.65 ms`. Current low-bit decode is now gated more by shared GDN/attention
structure than by quant-specific MoE expert-bank coverage.

## Validation

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- `cargo test -p qwen-llm moe_grouped_down_iq4_xs_matches_f32_dequant_fixture --release -- --ignored --nocapture`
- Clean qwen/lcpp `tg128` paired spots, no parallel GPU workloads
