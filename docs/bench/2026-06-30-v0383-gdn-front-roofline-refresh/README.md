# v0.383 GDN Front Roofline Refresh

Refreshes the current GDN projection primitive and phase attribution after the
v0.382 gate/up replay work. The goal was to decide whether A10B's large GDN
front bucket justifies an immediate projection-kernel branch.

## Commands

```bash
target/release/qwen-bench gdn-proj-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --warmup 5 --iters 20 \
  > target/profiles/v0383-a3b-q4-gdn-proj-micro.out

target/release/qwen-bench gdn-proj-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --warmup 5 --iters 20 \
  > target/profiles/v0383-a10b-q4xl-gdn-proj-micro.out

QWEN_PHASE_GDN_PROJ_SPLIT=1 QWEN_PHASE_MOE_FFN_SPLIT=deep \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0383-a10b-q4xl-phase-ctx8192-gdn-proj-split.out
```

## Results

| Model | Projection | Bytes | GPU | Nominal BW |
| --- | --- | ---: | ---: | ---: |
| A3B | `qkv` | `0.5348 GB` | `1.1787 ms` | `453.7 GB/s` |
| A3B | `z` | `0.2674 GB` | `0.6370 ms` | `419.7 GB/s` |
| A3B | `qkv+z` | `0.8022 GB` | `1.7949 ms` | `446.9 GB/s` |
| A3B | `out` | `0.2674 GB` | `0.7000 ms` | `382.0 GB/s` |
| A10B | `qkv` | `1.4439 GB` | `2.8763 ms` | `502.0 GB/s` |
| A10B | `z` | `0.9626 GB` | `1.9684 ms` | `489.0 GB/s` |
| A10B | `qkv+z` | `2.4065 GB` | `4.8675 ms` | `494.4 GB/s` |
| A10B | `out` | `0.9626 GB` | `2.0671 ms` | `465.7 GB/s` |

A10B ctx8192 split phase:

| Phase | GPU | Share |
| --- | ---: | ---: |
| `gdn qkv proj` | `2.91 ms` | `12.1%` |
| `gdn z proj` | `2.00 ms` | `8.3%` |
| `gdn beta proj` | `0.20 ms` | `0.8%` |
| `gdn alpha proj` | `0.19 ms` | `0.8%` |
| `gdn tail` | `1.18 ms` | `4.9%` |
| `gdn out_proj` | `2.33 ms` | `9.7%` |

## Decision

Do not start a local GDN projection row-shape or `qkv+z` fusion branch. The large
front bucket is almost entirely mandatory Q8_0 weight streaming, and the
`qkv+z` primitive is already at roughly stream-roofline bandwidth. Reusing the
activation vector cannot pay enough because activation bytes are tiny relative

Future GDN work needs one of these stronger signals:

- a model/format change that removes projection weight bytes;
- an algorithmic GDN recurrence/dataflow change;
- a counter trace showing stalls not explained by weight streaming;
- an end-to-end proof that saves at least `0.35 ms` on A10B ctx8192.
