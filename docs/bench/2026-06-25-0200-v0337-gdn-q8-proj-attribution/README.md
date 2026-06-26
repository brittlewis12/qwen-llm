# v0.337 GDN Q8 Projection Attribution

Status: added `QWEN_PHASE_GDN_PROJ_SPLIT=1` for phase-only decode profiling and
used it to split the A10B GDN front projection bucket into QKV, Z, beta, and
alpha projections. Also tested and killed an env-gated Q8_0 R4 mat-vec sidecar;
the sidecar was removed after it regressed both MoE targets.

## Validation

- `cargo fmt`
- `QWEN_MATVEC_Q8_0_R4=1 cargo test -p qwen-llm mat_vec_q8_0_matches_cpu --release -- --nocapture`
- `cargo build --release --bin qwen-bench`
- Sequential `decode_ctx_sweep.py` A3B/A10B `ctx8192` R4 gates
- A10B `qwen-bench phase --ctx 8192` with `QWEN_PHASE_GDN_PROJ_SPLIT=1` and
  `QWEN_PHASE_MOE_FFN_SPLIT=deep`

## R4 Falsifier

| Model | Default | Q8 R4 | Read |
| --- | ---: | ---: | --- |
| A3B `ctx8192` block 0 | `92.6 t/s` | `92.5 t/s` | flat |
| A3B `ctx8192` block 1 | `94.0 t/s` | `91.8 t/s` | slower |
| A10B `ctx8192` | `42.6 t/s` | `42.0 t/s` | slower |

The Q8 R4 kernel was correctness-safe (`max|delta|=1.45e-6`) but failed the
end-to-end gate. Do not reopen row-widening for Q8_0 mat-vec without a specific
counter signal; the next Q8 branch must change data movement, fusion, or packing.

## A10B Phase Split

Command:

```sh
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_MOE_FFN_SPLIT=deep \
target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0337-a10b-ctx8192-gdn-proj-split-phase.out
```

Result:

| Phase | GPU ms | Share |
| --- | ---: | ---: |
| GDN QKV projection | `2.98` | `12.2%` |
| GDN Z projection | `2.04` | `8.3%` |
| GDN beta projection | `0.20` | `0.8%` |
| GDN alpha projection | `0.25` | `1.0%` |
| GDN out projection | `2.29` | `9.4%` |
| Attention mixer | `4.39` | `18.0%` |
| MoE routed gate/up | `3.16` | `12.9%` |
| MoE routed down | `2.57` | `10.5%` |
| MoE shared gate/up | `0.91` | `3.7%` |
| MoE shared down | `0.66` | `2.7%` |

Interpretation: the GDN projection budget is real and concentrated in QKV, Z, and
OUT. Beta/alpha remain explicitly killed. The next credible decode work is either
a QKV+Z/out projection dataflow change with an end-to-end A10B gate, or a MoE FFN
execution-shape branch; not another scalar skinny projection or Q8 row-count tweak.
