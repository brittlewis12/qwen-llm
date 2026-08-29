# Flash-Next Natural N=512 Attribution

Decision: **GO** to an exact-N=512 strict E8P32 router falsifier before creating
another packed kernel.

## Source And Workload

- Source: `d3fbfe7d2fabc956cf9ce7fdc00e9f1924e94cc0`, rebuilt before acquisition.
- Device: Apple M4 Max with unified memory.
- Model: UD-Q3_K_XL from `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`, on the external PCIe SSD.
- Prompt: the first 512 no-special-token IDs of `docs/PERF-ROADMAP.md` at the
  source commit, decoded to the tracked `prompt.txt` fixture. Its 2,365 UTF-8
  bytes have SHA-256
  `9ea0ba6ce8218c9a2bc6129e8865dc1c0f562fb0ad6e01903b58912f8b8a8fc7`.
- Controls: `QWEN4EXP_PACKED_PREFILL_PROFILE=1`,
  `QWEN_MATVEC_F32_LCPP_R2=1`, selected packed QSA disabled, one generated token,
  and exact 512-forward capacity. The strict router remained outside its
  N=2,048-only qualification, so this is the generic-router baseline.

The existing CLI ran first unprofiled, reset, warm unprofiled, reset, then one
profiled command. It required bitwise-equal full-vocabulary endpoint logits
across all three passes. M4 Max rejected dispatch-boundary counters and used the
established 128 encoder-stage timestamps instead.

## Observer Results

| Pass | Wall (ms) | Command GPU (ms) | Outside GPU (ms) |
|:---|---:|---:|---:|
| first | 42,828.182 | 1,160.637 | 41,667.545 |
| warm | 1,151.697 | 1,149.033 | 2.664 |
| profiled | 1,154.128 | 1,150.073 | 4.055 |

- GPU observer ratio: `1.000905`.
- Wall observer ratio: `1.002111`.
- Raw timestamp coverage: `1.000000`.
- Profiled throughput: `443.52 tok/s`.
- Sampling and all observer gates passed; no profile warning was emitted.

The first pass is a cold-paging observation, not warm-kernel evidence. Its
roughly 41.7-second outside-GPU interval motivates the separately ranked
internal-SSD control.

## Representative Attribution

| Stage | GDN layer 5 (ms) | QSA layer 7 (ms) |
|:---|---:|---:|
| attention HC | 0.853 | 0.850 |
| mixer | 6.632 | 6.456 |
| mixer bridge + combine | 0.976 | 0.974 |
| FFN HC | 0.851 | 0.849 |
| MoE | 13.968 | 13.881 |
| MoE bridge + combine | 0.973 | 0.972 |
| complete block | 24.255 | 23.984 |

The established 34-GDN/12-QSA extrapolation assigns 55.78% to MoE, 19.61% to
the GDN mixer, 6.74% to QSA, 6.81% to HC, 7.79% to bridge/combine, 4.69% to
bootstrap, and 0.10% to the tail. It over-assigns the command by 1.51%, so the
representative values are leverage estimates rather than additive accounting.

Crediting only the established 43 standard MoE layers gives:

| MoE substage | ms/layer | Command share |
|:---|---:|---:|
| F32 router projection | 3.960750 | 14.81% |
| top-k/shared selection | 0.078250 | 0.29% |
| route bucket publication | 0.410125 | 1.53% |
| routed IQ3 gate/up | 5.397333 | 20.18% |
| routed IQ4_NL down | 3.515666 | 13.14% |
| ordered reduction | 0.091167 | 0.34% |
| shared tail | 0.511958 | 1.91% |

## Decision

The router is the first experiment because it combines a 14.81% measured
ceiling with an already-bit-exact implementation. Generic N=512 router time is
within 0.3% of one quarter of the N=2,048 baseline. Applying the measured
N=2,048 strict/generic residual ratio predicts `0.650065 ms/layer`, saving about
`142.36 ms` or 12.38% of command GPU.

Extend only the host qualification and exact differential gates to N=512.
Require component median `<=3.564675 ms/layer` and warm command GPU
`<=1137.543 ms`, alongside bitwise route metadata, endpoint logits, and state.
Failure closes N=512 without disturbing the qualified N=2,048 default.

Raw process logs remain in
`target/profiles/qwen4exp-natural-n512-20260829/`; stdout/stderr SHA-256 values
are `e530152ea80c3012dbfdb19a69e554de54aed4511184ae225fcec380233a46e9` and
`5fc3d4158e46fb5e6d813ced9f0718a16c2b9c8519c4c98f6e6467922aca15a5`.

Adversarial leverage review: `01a04de7-d66e-73f1-b3de-14ba60527c89`.
