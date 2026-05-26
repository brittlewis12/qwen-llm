# v0.135 GDN Skinny Default Promotion

Status: default promotion gate after `v0.134`. The GDN skinny E8xP32 path is now
default-on for eligible F32 prompt-prefill GDN `beta_proj` / `alpha_proj` mat-mat
work, with `QWEN_PREFILL_GDN_SKINNY_E8P32=0` as the rollback path.

## Shape Audit

Sampled GGUF tensor shapes:

| Model | `ssm_alpha/beta` dtype | Shape | Read |
| --- | --- | --- | --- |
| Qwen3.6 27B | F32 | `[5120,48]` | eligible |
| Qwen3.6 35B A3B | F32 | `[2048,32]` | eligible |
| Qwen3.5 0.8B/4B/9B/27B | Q8_0 | varies | falls back |
| Qwen3.5 122B A10B | Q8_0 | `[3072,64]` | falls back |

The production helper is only called at the GDN prompt-prefill `beta_proj` and
`alpha_proj` callsites. It is not a generic F32 mat-mat dispatcher replacement.

## Additional Canaries

All runs were sequential on AC power via `prefill_sweep.py --repeat-blocks 2`.

| Model / branch | Prompt | Baseline rows | Skinny rows | Read |
| --- | ---: | ---: | ---: | --- |
| A3B default | `pp128` | `642.32`, `663.98` | `678.86`, `675.24` | positive small-prompt canary |
| A3B default | `pp1024` | `1226.11`, `1226.55` | `1237.35`, `1240.68` | positive medium canary |
| A3B matrix-G8 | `pp4096` | `1386.19`, `1386.24` | `1407.17`, `1410.67` | composes with matrix branch |
| 27B matrix-G6/G8 | `pp128` | `193.80`, `193.79` | `199.08`, `198.87` | positive small dense canary |

These extend the `v0.134` evidence, where 27B matrix-G6/G8 was positive at
`pp512`, `pp1024`, `pp8192`, `pp16384`, and warmed `pp4096`.

Power / residency support after the canaries: AC power, no thermal or performance
warning, no CPU power warning, and `95%` free memory from `memory_pressure -Q`.

## Correctness

- A3B default-on equivalent correctness with `QWEN_PREFILL_GDN_SKINNY_E8P32=1`
  passed `prefill_tokens_matches_single_token_loop_35b_a3b_moe` at `T=12/P=8`:
  final logits `0.999985`, GDN state `0.999735`, GDN conv `0.999811`, KV K/V
  `>=0.999873`.
- The accepted residual risk is that A3B GDN/conv cosine is lower than the 27B
  gate while still above the established `0.999` threshold.
- `v0.134` already passed 0.8B fallback, 27B active prefill, and 27B matrix-prefix
  correctness with the skinny path enabled.

## Decision

Promote default-on. This is now a narrow callsite-scoped specialization with
positive evidence on both eligible F32 shape families and fallback behavior for
sampled Q8_0 families. Keep the rollback env documented and monitor future F32
GDN shapes.

## Artifacts

- `v0135-a3b-pp128-gdn-skinny-canary.json`
- `v0135-a3b-pp1024-gdn-skinny-canary.json`
- `v0135-a3b-pp4096-matrix-gdn-skinny-canary.json`
- `v0135-27b-pp128-gdn-skinny-canary.json`
