# v0.134 GDN Skinny E8xP32 Spike

Status: env-only dense GDN projection spike on top of `v0.133`. The new
`QWEN_PREFILL_GDN_SKINNY_E8P32=1` path routes eligible F32 GDN `beta_proj` and
`alpha_proj` skinny mat-mat work through the existing E8xP32 router kernel
instead of the generic dense mat-mat dispatcher. The default path is unchanged.

## Attribution

27B dense `pp4096`, matrix-G6/G8 enabled, phase tracing:

| Bucket | Baseline ms | Skinny ms | Read |
| --- | ---: | ---: | --- |
| `gdn_qkv` | `1876.10` | `1844.11` | unchanged within trace noise |
| `gdn_z` | `1124.07` | `1102.01` | unchanged within trace noise |
| `gdn_beta_alpha` | `672.45` | `105.96` | targeted F32 skinny projection win |

The trace path commits/waits at phase boundaries, so this is attribution evidence
only. Promotion decisions use the repeated throughput rows below.

## End-to-End Rows

27B dense `Qwen3.6-27B-Q4_K_M.gguf`, matrix-G6/G8 enabled, sequential AC-power
runs via `prefill_sweep.py --repeat-blocks 2 --shuffle-seed <seed>`:

| Prompt | Baseline rows | Skinny rows | Read |
| --- | ---: | ---: | --- |
| `pp512` | `212.31`, `212.93` | `216.34`, `218.34` | `~+2-3%` |
| `pp1024` | `205.48`, `200.13` | `213.56`, `208.98` | `~+4%` |
| `pp4096` | `176.29`, `191.45` | `195.25`, `193.02` | first baseline cold; warmed `~+0.8%` |
| `pp8192` | `185.67`, `185.64` | `191.09`, `190.65` | `~+2.7%` |
| `pp16384` | `175.48`, `176.08` | `180.34`, `178.87` | `~+1.6-2.8%` |

Power / residency support after the validation batch: AC power, no thermal or
performance warning, no CPU power warning, and `95%` free memory from
`memory_pressure -Q`.

## Correctness

- `QWEN_PREFILL_GDN_SKINNY_E8P32=1 cargo test --release --lib -p qwen-llm prefill_tokens_matches_single_token_loop_0_8b -- --nocapture` passed.
- `QWEN_PREFILL_GDN_SKINNY_E8P32=1 QWEN_PREFILL_ATTN_MATRIX_G6=1 QWEN_PREFILL_ATTN_MATRIX_G8=1 QWEN_TEST_27B_PREFILL_T=32 QWEN_TEST_27B_PREFILL_P=32 cargo test --release -p qwen-llm prefill_tokens_matches_single_token_loop_27b -- --nocapture` passed: final logits `1.000000`, hidden min `0.999999`, GDN state/conv `0.999999`, KV K/V `>=0.999999`.
- `QWEN_PREFILL_GDN_SKINNY_E8P32=1 QWEN_PREFILL_ATTN_MATRIX_G6=1 QWEN_PREFILL_ATTN_MATRIX_G8=1 cargo test --release -p qwen-llm prefill_tokens_matches_single_token_loop_27b_matrix_g6_prefix_gate -- --ignored --nocapture` passed at prefix `4096`.

## Decision

Keep the path env-only for this checkpoint. The inner-bucket win is large and the
27B end-to-end rows are consistently positive, but the total win is still small
enough that default-on should require a narrow promotion gate rather than one-size
evidence.

Promotion gate before flipping default-on:

- Confirm the fast path only fires for intended GDN `beta_proj` / `alpha_proj`
  skinny F32 projections.
- Add family/shape coverage for smaller dense models and any GDN variant with
  different projection dimensions.
- Include awkward/small prompt sizes and one decode-after-prefill state gate.
- Require no tested row to regress by more than about `1%`, median improvement
  positive, and phase attribution still localized to `gdn_beta_alpha`.

## Artifacts

- `v0134-qwen-27b-pp4096-gdn-split-summary.tsv`
- `v0134-qwen-27b-pp4096-gdn-skinny-phase-summary.tsv`
- `v0134-27b-pp512-gdn-skinny-sweep.json`
- `v0134-27b-pp1024-gdn-skinny-sweep.json`
- `v0134-27b-pp4096-gdn-skinny-sweep.json`
- `v0134-27b-pp8192-gdn-skinny-sweep.json`
- `v0134-27b-pp16384-gdn-skinny-sweep.json`
