# v0.338 GDN Projection Microbench

Status: added `qwen-bench gdn-proj-micro` to time exact-shape GDN Q8 projection
primitives over all GDN layers. Also refreshed a compact qwen-side primary-family
suite with a clean `7e8808241` build stamp after the v0.337 checkpoint.

## Validation

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- `target/release/qwen-bench suite ...` compact family spot with clean stamp
- `target/release/qwen-bench gdn-proj-micro ...` on A3B and A10B

## Compact Suite

All rows: AC power, clean build stamp `7e8808241`, no thermal/performance warning.

| Model | `pp512` | `pp4096` | `tg128` |
| --- | ---: | ---: | ---: |
| 0.8B Q4_K_M | `8037.45` | `8008.30` | `357.98` |
| 27B Q4_K_M | `242.45` | `215.69` | `23.95` |
| A3B Q4_K_M | `1503.02` | `1624.27` | `103.77` |
| A10B Q4_K_XL | `449.37` | `497.55` | `44.01` |

Decode context refresh before the suite:

| Model | `ctx1024` | `ctx8192` |
| --- | ---: | ---: |
| A3B Q4_K_M | `99.7 t/s` | `93.4 t/s` |
| A10B Q4_K_XL | `42.4 t/s` | `42.3 t/s` |

## GDN Projection Primitive

Commands:

```sh
target/release/qwen-bench gdn-proj-micro \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --warmup 3 --iters 10 \
  > target/profiles/v0338-a3b-gdn-proj-micro.tsv

target/release/qwen-bench gdn-proj-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --warmup 3 --iters 10 \
  > target/profiles/v0338-a10b-gdn-proj-micro.tsv
```

Results use weight bytes only as the bandwidth denominator, so they are a lower
bound on actual traffic and a conservative saturation signal.

| Model | Phase | Weight GB | GPU ms | Weight GB/s |
| --- | --- | ---: | ---: | ---: |
| A3B | QKV | `0.5348` | `1.2297` | `434.9` |
| A3B | Z | `0.2674` | `0.6391` | `418.4` |
| A3B | QKV+Z | `0.8022` | `1.8033` | `444.8` |
| A3B | OUT | `0.2674` | `0.6766` | `395.2` |
| A10B | QKV | `1.4439` | `2.9763` | `485.1` |
| A10B | Z | `0.9626` | `2.0359` | `472.8` |
| A10B | QKV+Z | `2.4065` | `5.0516` | `476.4` |
| A10B | OUT | `0.9626` | `2.0892` | `460.7` |

Interpretation: GDN Q8 projection time is real, but the primitive is already at
or near the measured stream roofline (`474 GB/s`) on the live A10B cell. Local Q8
mat-vec retunes are now low EV unless they reduce bytes structurally; the next
serious decode work should move to MoE FFN dataflow or an attention body rewrite.
