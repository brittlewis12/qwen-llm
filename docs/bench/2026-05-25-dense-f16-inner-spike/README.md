Dense F16-inner/Q6-F16-source falsifier, run from dirty `v0.126` code.

Hypothesis:
- Keep dense 27B matrix-G6 attention enabled.
- Compute fused Q4_K gate/up SwiGLU in F32, store the final FFN inner as F16.
- Feed that F16 inner directly to a Q6_K down mat-mat that reads F16 source rows.
- Expected win: halve inner write/read and down-source traffic without changing effective math, because the existing Q6_K F32-source mat-mat already stages B through half.

Correctness before timing:
- Q6_K F16-source matched the existing F32-source half-staged path at N=32 with `max|delta|=0`.
- Fused F16 inner bytes matched `scatter(F32 inner -> F16)` at N=32, and Q6 down output matched with `max|delta|=0`.
- Active 27B prefill correctness passed with `QWEN_PREFILL_ATTN_MATRIX_G6=1 QWEN_PREFILL_DENSE_FFN_F16_INNER=1`, `T=32`, `P=32`.

Commands:

```sh
uv run scripts/profile/prefill_sweep.py \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --n-prompt 4096 --runs 3 --cooldown-seconds 15 \
  --variant matrix-g6-a:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --variant fused-q4:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=1 \
  --variant f16-inner:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_PREFILL_DENSE_FFN_F16_INNER=1 \
  --variant matrix-g6-b:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --output docs/bench/2026-05-25-dense-f16-inner-spike/pp4096.json

uv run scripts/profile/prefill_sweep.py \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --n-prompt 16384 --runs 3 --cooldown-seconds 20 \
  --variant matrix-g6-a:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --variant fused-q4:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=1 \
  --variant f16-inner:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_PREFILL_DENSE_FFN_F16_INNER=1 \
  --variant matrix-g6-b:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --output docs/bench/2026-05-25-dense-f16-inner-spike/pp16384.json
```

Summary:

| Shape | matrix-G6 A | fused-Q4 | F16-inner | matrix-G6 B | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `pp4096` | `196.74` | `201.66` | `202.33` | `200.16` | F16 is only `+0.3%` over fused-Q4, within drift |
| `pp16384` | `180.45` | `182.00` | `180.59` | `185.85` | F16 loses to fused-Q4 and late baseline anchor |

Direct mat-mat microbench at `N=4096`, `dispatches=16` showed the core problem:

| Tensor | F32-source | F16-source | F16/F32 | Read |
| --- | ---: | ---: | ---: | --- |
| `blk.0.ffn_down.weight` Q6_K | `61.604 ms` | `64.000 ms` | `1.039x` | slower |
| `blk.0.attn_qkv.weight` Q6_K | `35.396 ms` | `35.953 ms` | `1.016x` | slower |

Conclusion:
- Correctness was clean, but performance falsified the hypothesis.
- The F16-source Q6 kernel itself is slower, so the reduced source bytes do not beat the existing F32-source half-staged path.
- The production/env code was not kept; preserve this directory as the negative-result record.
