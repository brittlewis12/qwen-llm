# A3B Q6 MoE Decode Coverage - v0.358

Manual paired spot after `v0.358`. Same machine, sequential runs, no concurrent
GPU workloads, AC power. The goal was to verify that native Q6_K routed gate/up
coverage turns A3B Q6 MoE decode from unsupported into a usable fast path.

## Identity

- `qwen-llm`: `25edacb05`
- `llama.cpp`: `c818263f2a` (build 9833, backends `MTL,BLAS`)
- Qwen power state: AC power, high-power mode, no thermal or performance warning
- QWEN_* env for throughput rows: none

## Decode Throughput

| A3B row | llama.cpp `tg128` | qwen `tg128` | Delta | Read |
| --- | ---: | ---: | ---: | --- |
| Q6_K | `79.53 t/s` | `99.79 t/s` | `1.25x` | coverage row now green |
| Q4_K_M guard | n/a | `107.00 t/s` | n/a | existing path unchanged |

## Phase Attribution

Command used `QWEN_PHASE_MOE_FFN_SPLIT=2 qwen-bench phase --ctx 128` on Q6_K.

| Phase | Time | Share |
| --- | ---: | ---: |
| `gdn front proj` | `1.86 ms` | `18.5%` |
| `moe ffn routed gate/up` | `1.46 ms` | `14.5%` |
| `attn mixer` | `1.22 ms` | `12.1%` |
| `lm head` | `1.04 ms` | `10.4%` |
| `moe ffn routed down` | `0.86 ms` | `8.6%` |

Read: Q6 is no longer an unsupported quant hole. The new native routed SwiGLU
feeds the existing Q6 down weighted-sum path and clears the paired decode row.
Q8 remains the larger coverage gap because it still needs both gate/up and routed
down decode support.

## Validation

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- `cargo test -p qwen-llm moe_swiglu_q6_k_matches_f32_dequant_fixture --release -- --ignored --nocapture`
- clean qwen/lcpp `tg128` paired spot
