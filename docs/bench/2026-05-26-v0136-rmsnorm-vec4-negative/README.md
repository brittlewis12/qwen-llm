# v0.136 RMSNorm Vec4 Falsifier

Status: dirty-code negative spike after `v0.135`. A phase trace on A3B matrix
`pp4096` appeared to put a large share of attribution in `pre_norm`, so the spike
tested a llama-style `float4` variant of the batched RMSNorm kernel behind
`QWEN_PREFILL_RMSNORM_VEC4=1`.

## Result

A3B matrix `pp4096`, default GDN skinny active in both variants:

| Variant | Rows | Read |
| --- | ---: | --- |
| matrix | `1408.70`, `1409.10` | baseline |
| matrix + vec4 RMSNorm | `1408.64`, `1407.07` | flat/slightly negative |

The phase trace also stayed flat:

| Phase | Baseline ms | Vec4 ms | Read |
| --- | ---: | ---: | --- |
| GDN `pre_norm` | `957.96` | `952.99` | no meaningful movement |
| Attn `pre_norm` | `328.00` | `324.51` | no meaningful movement |

Correctness with the vec4 spike passed 0.8B prefill-vs-single and A3B
prefill-vs-single. The branch was stripped instead of carried env-only.

## Decision

Do not spend another branch on simple vector spelling of the existing batched
RMSNorm kernel. If `pre_norm` remains suspicious, the next proof needs a stronger
mechanism: fuse pre-norm with a following projection, reduce phase-trace
attribution artifacts, or compare against llama.cpp norm nodes directly.

## Artifacts

- `v0136-a3b-pp4096-matrix-rmsnorm-vec4-canary.json`
- `v0136-a3b-pp4096-matrix-phase-summary.tsv`
- `v0136-a3b-pp4096-matrix-rmsnorm-vec4-phase-summary.tsv`
