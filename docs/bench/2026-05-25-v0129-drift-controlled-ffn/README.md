# v0.129 Drift-Controlled Dense FFN Candidates

Status: clean `v0.129` follow-up after adding `prefill_sweep.py`
`--repeat-blocks` / `--shuffle-seed`. Runs were sequential on AC power; post-run
`pmset -g therm` reported no thermal/performance warning and
`memory_pressure -Q` reported `95-96%` free.

Artifacts:

- `pp4096.json`
- `pp16384.json`

## pp4096

27B dense, `runs=3`, `repeat_blocks=2`, `shuffle_seed=11`, all variants include
`QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1`.

| Block | Order | Variant | Tokens/s | Read |
| ---: | ---: | --- | ---: | --- |
| 0 | 0 | `g6` | `193.36` | first-run/cold-process penalty |
| 0 | 1 | `g6-fused` | `209.37` | comparable to warmed baseline |
| 0 | 2 | `g6-smem` | `197.82` | not a stable win |
| 0 | 3 | `g6-fused-smem` | `197.20` | does not compose in first block |
| 1 | 0 | `g6` | `208.96` | warmed comparison anchor |
| 1 | 1 | `g6-smem` | `209.73` | `+0.4%`, below promote gate |
| 1 | 2 | `g6-fused` | `210.98` | `+1.0%`, below promote gate |
| 1 | 3 | `g6-fused-smem` | `211.09` | `+1.0%`, no meaningful composition |

## pp16384

27B dense, `runs=2`, `repeat_blocks=2`, `shuffle_seed=13`, all variants include
`QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1`.

| Block | Order | Variant | Tokens/s | Read |
| ---: | ---: | --- | ---: | --- |
| 0 | 0 | `g6` | `193.46` | strong anchor |
| 0 | 1 | `g6-fused` | `189.70` | loses in block 0 |
| 1 | 0 | `g6-fused` | `193.60` | strong row |
| 1 | 1 | `g6` | `181.14` | late anchor collapse |

## Read

- The interleaved harness worked: block/order metadata exposed first-run and late
  anchor drift directly instead of hiding it in a single averaged row.
- `QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=1` is still env-only. The stable pp4096
  signal is about `+1%`, and pp16384 is inconsistent.
- `QWEN_MATMAT_QK_LLAMA_SMEM=1` stays opt-in only. The pp4096 signal is below the
  noise floor and does not compose into a default-worthy row.
- Next dense work should use phase-local FFN/GDN timing to prove where any win
  lands before another total-throughput default decision.
