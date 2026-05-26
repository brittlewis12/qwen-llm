# v0.137 A3B Matrix Attention Default

Status: default promotion gate after `v0.136`. The group-8 matrix-attention path
is now auto-on for the proven A3B attention shape and remains force-off with
`QWEN_PREFILL_ATTN_MATRIX_G8=0`. Dense group-6 matrix attention remains env-only
via `QWEN_PREFILL_ATTN_MATRIX_G6=1`.

## Policy

- Auto mode applies only when the existing packed group-8 attention path would
  already activate: F16 KV cache, `head_dim=256`, `n_q=16`, `n_kv=2`, group `8`,
  and the packed-G8 prompt threshold (`n_pos >= 128` by default).
- `QWEN_PREFILL_ATTN_MATRIX_G8=0` forces rollback to packed group-8 attention.
- `QWEN_PREFILL_ATTN_MATRIX_G8=1` forces matrix for eligible shape and keeps the
  existing hard error if matrix scratch is undersized.
- In auto mode, undersized matrix scratch falls back to packed attention instead
  of erroring. `qwen-bench pp`, `pp-wait`, and packed `decode` prefill already
  allocate matrix scratch from the prompt length, so they get the default win.
- The local packed-vs-old matrix oracle uses the documented matrix envelope:
  no nonfinite values, `cos >= 0.9999`, and `max_abs <= 2e-2`. The production
  correctness gate is the stronger prefill-vs-single model-state comparison.

## Fresh Current-HEAD Rows

All rows are A3B `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, sequential AC-power runs with
GDN skinny already defaulted.

| Shape | Packed/default rows | Matrix rows | Read |
| --- | ---: | ---: | --- |
| `pp128` | `671.33`, `676.27` | `726.81`, `719.09` | `~+7%` |
| `pp512` | `1095.23`, `1099.98` | `1249.47`, `1233.00` | `~+13%` |
| `pp1024` | `1239.69`, `1241.49` | `1421.72`, `1433.08` | `~+15%` |
| `pp4096` | `1192.95`, `1191.45` | `1411.38`, `1406.32` | `~+18%` |
| `pp16384` | `890.94`, `905.04` | `1185.98`, `1121.63` | `~+25-33%` |
| real `v02_reva` `34.5k` | `655.34` | `837.18` | `~+28%` |

Post-promotion auto/rollback canary at `pp512`:

| Mode | Rows | Read |
| --- | ---: | --- |
| auto/default | `1245.21`, `1235.93` | matrix active by default, clean `v0.137` build |
| `QWEN_PREFILL_ATTN_MATRIX_G8=0` | `1094.59`, `1093.84` | rollback to packed, clean `v0.137` build |

Power / residency support after the gate: AC power, no thermal or performance
warning, no CPU power warning, and `95%` free memory from `memory_pressure -Q`.

## Correctness And Coverage

- Full ignored A3B prefill-vs-single passed with matrix auto/default, including
  prefix `4096` / `8191` active attention shapes. Worst reported active row was
  prefix `4096`, `T=8`, `P=8`: final logits `0.999984`, GDN cos min `0.999611`.
- 0.8B prefill-vs-single passed as a non-A3B guardrail.
- Trace-label coverage at default/auto `pp512` showed: `attn-prefill-g8-matrix:10`,
  `moe-route-fused:40`, `moe-routed-grouped:40`, and `moe-shared-packed:40`.

## Residual Risk

- Matrix attention is a different numeric path; near-tie next-token decisions can
  still differ despite high cosine and clean model-state gates.
- Scratch memory scales with `chunk_p * n_q * n_pos` score scratch and per-attn
  V_T storage. Custom callers should use prompt-sized scratch or rely on auto
  fallback; force-on remains strict.
- The promotion is A3B/group-8 only. Do not generalize to dense G6, A10B/G16,
  non-F16 KV, or other attention shapes without a separate gate.

## Artifacts

- `v0137-a3b-pp128-matrix-default-gate.json`
- `v0137-a3b-pp512-matrix-default-gate.json`
- `v0137-a3b-pp1024-matrix-default-gate.json`
- `v0137-a3b-pp4096-matrix-default-gate.json`
- `v0137-a3b-pp16384-matrix-default-gate.json`
- `v0137-a3b-v02-reva-matrix-default-gate.json`
- `v0138-a3b-pp512-matrix-auto-vs-off-clean.json`
- `v0137-a3b-matrix-auto-pp512-coverage-summary.txt`
