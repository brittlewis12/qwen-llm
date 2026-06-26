# GDN Decode No-Op Ladder - v0.336

Diagnostic no-op ladder for GDN decode projections at `ctx8192`. These variants
are correctness-breaking lower-bound oracles: they replace selected projections
with F32 fills, so they still pay fill cost and should not be interpreted as an
attainable speedup.

## Aggregate Gate

| Model | Variant | t/s | GPU ms/token | Read |
| --- | --- | ---: | ---: | --- |
| A3B Q4_K_M | default | `94.0` | `10.19` | baseline |
| A3B Q4_K_M | GDN front no-op | `112.9` | `8.41` | `-1.78 ms` lower-bound budget |
| A3B Q4_K_M | GDN out no-op | `99.6` | `9.58` | `-0.61 ms` lower-bound budget |
| A10B Q4_K_XL | default | `42.4` | `23.06` | baseline |
| A10B Q4_K_XL | GDN front no-op | `53.0` | `18.38` | `-4.68 ms` lower-bound budget |
| A10B Q4_K_XL | GDN out no-op | `46.8` | `20.88` | `-2.18 ms` lower-bound budget |

## Subprojection Gate

| Model | Variant | t/s | GPU ms/token | Read |
| --- | --- | ---: | ---: | --- |
| A3B Q4_K_M | default | `93.7` | `10.22` | baseline |
| A3B Q4_K_M | QKV no-op | `104.6` | `9.11` | large budget |
| A3B Q4_K_M | Z no-op | `99.4` | `9.62` | real budget |
| A3B Q4_K_M | beta no-op | `93.3` | `10.28` | no budget |
| A3B Q4_K_M | alpha no-op | `93.9` | `10.19` | no budget |
| A3B Q4_K_M | out no-op | `99.7` | `9.59` | real budget |
| A10B Q4_K_XL | default | `42.5` | `23.02` | baseline |
| A10B Q4_K_XL | QKV no-op | `48.2` | `20.27` | large budget |
| A10B Q4_K_XL | Z no-op | `46.1` | `21.19` | large budget |
| A10B Q4_K_XL | beta no-op | `42.5` | `23.03` | no budget |
| A10B Q4_K_XL | alpha no-op | `41.9` | `23.34` | no budget/noise |
| A10B Q4_K_XL | out no-op | `46.5` | `21.02` | large budget |

## Interpretation

- The GDN branch is real, but the recoverable work is not beta/alpha skinny F32
  projection fusion.
- QKV, Z, and OUT dominate the no-op ladder and are Q8 projection-shaped work.
- The next credible implementation should be an exact-shape Q8 mat-vec/projection
  microbench for `h -> conv_dim`, `h -> v_dim`, and `v_dim -> h`, with strict
  end-to-end gates before touching production projection dispatch.
