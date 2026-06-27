# v0.342 True-Long Decode Attention Rerank

Status: refreshed A3B/A10B true-long decode attribution after the v0.341 MoE
pipeline negative, then added `qwen-bench attn-intra` as a visible harness command
for per-attention-layer decode body accounting.

## Validation

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- sequential A3B/A10B `ctx-sweep` packets, no parallel GPU workloads
- sequential A3B/A10B `phase` and `attn-intra` probes at long context

## Context Sweep

Commands used `scripts/profile/decode_ctx_sweep.py --fresh-per-checkpoint --window 4`.

| Model | `ctx8192` | `ctx16384` | `ctx32768` |
| --- | ---: | ---: | ---: |
| A3B Q4_K_M | `93.1 t/s` | `85.2 t/s` | `72.8 t/s` |
| A10B Q4_XL | `42.2 t/s` | `39.9 t/s` | `36.0 t/s` |

## Phase Rerank

Deep split phase profiles used `QWEN_PHASE_MOE_FFN_SPLIT=deep` and
`QWEN_PHASE_GDN_PROJ_SPLIT=1`.

| Model/context | Phase sum | Attention | GDN qkv+z+out | Routed gate/up | Routed down |
| --- | ---: | ---: | ---: | ---: | ---: |
| A3B `ctx16384` | `13.95 ms` | `3.38 ms` / `24.2%` | `2.85 ms` / `20.4%` | `1.14 ms` | `1.39 ms` |
| A3B `ctx32768` | `15.84 ms` | `5.49 ms` / `34.7%` | `2.66 ms` / `16.8%` | `1.13 ms` | `1.65 ms` |
| A10B `ctx16384` | `27.77 ms` | `5.52 ms` / `19.9%` | `7.94 ms` / `28.6%` | `3.22 ms` | `2.75 ms` |
| A10B `ctx32768` | `30.60 ms` | `8.22 ms` / `26.9%` | `8.18 ms` / `26.7%` | `3.53 ms` | `2.72 ms` |

Interpretation: attention explains most of the `16K -> 32K` slope, but phase
share alone is not enough to justify a full same-byte attention rewrite.

## Attn-Intra Body Probe

`qwen-bench attn-intra` profiles one full-attention layer after ramping KV to the
requested context. It reports the v4 main/reduce split and a simple KV/partial byte
estimate for the body.

| Model/context | One layer | Extrapolated | Main body | Main est BW | Reduce |
| --- | ---: | ---: | ---: | ---: | ---: |
| A3B `ctx8192` | `0.252 ms` | `2.52 ms` | `0.109 ms` | `623 GB/s` | `0.034 ms` |
| A3B `ctx32768` | `0.573 ms` | `5.73 ms` | `0.416 ms` | `649 GB/s` | `0.052 ms` |
| A10B `ctx8192` | `0.465 ms` | `5.58 ms` | `0.130 ms` | `533 GB/s` | `0.035 ms` |
| A10B `ctx32768` | `0.761 ms` | `9.13 ms` | `0.465 ms` | `583 GB/s` | `0.034 ms` |

The main decode body is already streaming the estimated subgroup KV bytes very
quickly. The remaining attention branch should require byte reduction or stronger
hardware counters, not another same-byte fusion/layout pass.

## Artifacts

- `target/profiles/v0342-a3b-true-long-ctx-sweep.json`
- `target/profiles/v0342-a10b-true-long-ctx-sweep.json`
- `target/profiles/v0342-a3b-ctx16384-deep-phase.out`
- `target/profiles/v0342-a3b-ctx32768-deep-phase.out`
- `target/profiles/v0342-a10b-ctx16384-deep-phase.out`
- `target/profiles/v0342-a10b-ctx32768-deep-phase.out`
- `target/profiles/v0342-a3b-ctx8192-attn-intra.out`
- `target/profiles/v0342-a3b-ctx32768-attn-intra.out`
- `target/profiles/v0342-a10b-ctx8192-attn-intra.out`
- `target/profiles/v0342-a10b-ctx32768-attn-intra.out`
