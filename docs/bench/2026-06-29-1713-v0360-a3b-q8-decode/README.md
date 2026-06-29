# A3B Q8 MoE Decode Coverage - v0.360

Manual paired spot after `v0.360`. Same machine, sequential runs, no concurrent
GPU workloads, AC power. The goal was to verify that native Q8_0 routed gate/up
and down coverage turns A3B Q8 MoE decode from unsupported into a usable fast
path.

## Identity

- `qwen-llm`: `5d4c5696c`
- `llama.cpp`: `c818263f2a` (build 9833, backends `MTL,BLAS`)
- Qwen power state: AC power, high-power mode, no thermal or performance warning
- QWEN_* env for throughput rows: none

## Decode Throughput

| A3B row | llama.cpp `tg128` | qwen `tg128` | Delta | Read |
| --- | ---: | ---: | ---: | --- |
| Q8_0 | `73.02 t/s` | `90.28 t/s` | `1.24x` | coverage row now green |
| Q4_K_M guard | n/a | `106.96 t/s` | n/a | existing path unchanged |

## Phase Attribution

Command used `QWEN_PHASE_MOE_FFN_SPLIT=2 qwen-bench phase --ctx 128` on Q8_0.

| Phase | Time | Share |
| --- | ---: | ---: |
| `gdn front proj` | `2.10 ms` | `19.2%` |
| `moe ffn routed gate/up` | `1.58 ms` | `14.5%` |
| `moe ffn routed down` | `1.32 ms` | `12.1%` |
| `attn mixer` | `1.28 ms` | `11.7%` |
| `lm head` | `1.03 ms` | `9.5%` |

Read: Q8 is no longer an unsupported MoE decode quant hole. The new native
routed SwiGLU and routed-down weighted-sum kernels clear the paired decode row.
The remaining quant branch returns to Q3/IQ4 low-bit performance rather than
basic Q6/Q8 coverage.

## Validation

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- `cargo test -p qwen-llm moe_q8_0_swiglu_down_weighted_matches_f32_dequant_fixture --release -- --ignored --nocapture`
- clean qwen/lcpp `tg128` paired spot
