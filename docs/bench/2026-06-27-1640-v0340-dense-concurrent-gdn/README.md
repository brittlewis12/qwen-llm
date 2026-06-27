# v0.340 Dense Concurrent-GDN Decode Default

Status: production-wired the existing dense GDN front-projection overlap path and
made it the dense decode default. Rollback is
`QWEN_DECODE_DENSE_CONCURRENT_GDN=0`. The path splits each dense GDN block into
pre-norm, concurrent front projections, and the dependent tail/FFN wave. MoE was
already wired separately and is unchanged.

Validation:

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- `cargo test -p qwen-llm metal_single_token_concurrent_gdn_matches_serial -- --nocapture`
- Sequential dense `tg128` A/B, no parallel GPU workloads

Sidecar A/B before default flip, `QWEN_DECODE_DENSE_CONCURRENT_GDN=1` versus the
old default:

| Model | Old default | Concurrent GDN | Ratio |
| --- | ---: | ---: | ---: |
| 0.8B | `360.36 t/s` | `372.23 t/s` | `1.033x` |
| 2B | `218.35 t/s` | `226.60 t/s` | `1.038x` |
| 4B | `112.57 t/s` | `116.81 t/s` | `1.038x` |
| 9B | `72.09 t/s` | `73.93 t/s` | `1.026x` |
| 27B | `24.28 t/s` | `24.86 t/s` | `1.024x` |

Default/rollback check after flipping the default:

| Model | Default | Rollback | Ratio |
| --- | ---: | ---: | ---: |
| 0.8B | `370.26 t/s` | `357.36 t/s` | `1.036x` |
| 27B | `25.05 t/s` | `24.23 t/s` | `1.034x` |

Interpretation: this banks the measured dense decode scheduling win across the
dense family. It is a real default-worthy cleanup, but it does not reopen local
GDN Q8 mat-vec retuning: v0.338 still shows the projection primitive near the
stream roofline. Future dense decode work should target structural byte reduction,
attention/KV traffic, or larger command-graph changes rather than row-shape tweaks.
