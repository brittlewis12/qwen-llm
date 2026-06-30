# v0.370 A3B HNorm Attention Partials Falsifier

Purpose: test the remaining concrete attention-byte hypothesis after v0.369:
store normalized half `o_partial` values for group-8 true-long decode attention,
keep `(m,l)` metadata in F32, and reduce with `o_norm * l_part * exp(m_part-m)`.

This was a dirty sidecar only. It was not retained because it missed the keep gate.

## Gate

- Keep only if A3B `attn-intra --ctx 32768` improves by about `>=1.06x`, or if a
  smaller primitive win plausibly gives `>=2%` full `ctx32768` decode.
- Kill on correctness drift beyond the existing v4 attention oracle.

## Validation

```text
QWEN_ATTN_V4_HNORM_PARTIALS=1 QWEN_ATTN_V4_G8_TILE=4 \
  cargo test -p qwen-llm attn_v4_group8_subgroup_matches_naive_f16kv \
  --release -- --ignored --nocapture
```

Result: passed. The `n_pos=4096, nwg=128, C=64` hnorm case had
`max|delta|=2.48e-7`, `cos=0.999994`.

## Measurement

```text
QWEN_ATTN_V4_HNORM_PARTIALS=1 \
  target/release/qwen-bench attn-intra \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 --runs 3 \
  > target/profiles/v0370-a3b-q4-attn-intra-ctx32768-hnorm.out
```

Compared with the v0.369 default artifact
`target/profiles/v0369-a3b-q4-attn-intra-ctx32768-h2-default.out`:

| A3B `ctx32768` attn-intra | v0.369 F32 partials | dirty hnorm partials | Read |
| --- | ---: | ---: | --- |
| one layer | `0.3457 ms` | `0.3408 ms` | `1.014x` |
| main | `0.1743 ms` | `0.1729 ms` | flat |
| reduce | `0.0677 ms` | `0.0641 ms` | small |

## Decision

Kill the branch. Normalized half partials are correctness-safe in this synthetic
oracle, but they only save about `1.4%` on one attention layer, far below the
attention keep gate and too small to justify another full true-long decode ramp.

Read: after v0.368/v0.369, A3B true-long attention is not primarily blocked by the
remaining F32 partial traffic. Move the active branch back to MoE/GDN dataflow or
a genuinely new read-once attention execution shape.
