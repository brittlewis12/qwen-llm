# A3B Low-Bit Decode — v0.362

Clean low-bit A3B decode checkpoint after defaulting the fast IQ3 routed
SwiGLU dataflow. Same physical box, sequential runs, no concurrent GPU benches.

## Identity

- `qwen-llm`: `ccdc452de`
- `llama.cpp`: `c818263f2a` (build 9833, backends `MTL,BLAS`)
- GPU: `Apple M4 Max`
- Rollback: `QWEN_DECODE_MOE_IQ3_FAST_SWIGLU=0`

## Results

| A3B row | llama.cpp `tg128` | qwen `tg128` | Read |
| --- | ---: | ---: | --- |
| Q3_K_M | `81.50 t/s` | `89.34 t/s` (`1.10x`) | low-bit row now green |
| UD-IQ4_XS | `80.71 t/s` | `89.00 t/s` (`1.10x`) | low-bit row now green |
| Q4_K_M guard | n/a | `107.49 t/s` | guard unchanged |

## Phase Check

Clean `ctx128` deep-split phase profile:

| Row | Fast gate/up | Rollback gate/up | Phase read |
| --- | ---: | ---: | --- |
| Q3_K_M | `1.07 ms` | `3.58 ms` | `-70%` routed gate/up |
| UD-IQ4_XS | `1.12 ms` | `4.59 ms` | `-76%` routed gate/up |

The remaining largest low-bit MoE FFN bucket is now routed down at `2.12 ms` on
both rows. The old gate/up gap is closed; do not reopen scalar IQ3 SwiGLU or
grouped-prefill reuse for single-token decode.

## Validation

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- `cargo test -p qwen-llm moe_swiglu_iq3_xxs_matches_f32_dequant_fixture --release -- --ignored --nocapture`
- `cargo test -p qwen-llm moe_swiglu_iq3_s_matches_f32_dequant_fixture --release -- --ignored --nocapture`
- Clean qwen/lcpp `tg128` paired spots, no parallel GPU workloads
