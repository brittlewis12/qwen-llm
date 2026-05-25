Mat-mat threadgroup-memory parity spike from dirty code after `v0.126`.

Hypothesis:
- llama.cpp's classic non-tensor `mul_mm` allocates less threadgroup memory when output tiles are full: `6144` bytes for generic NR1=32 and no edge-store scratch, `8192` only for partial-output tiles.
- Our Q4_K/Q5_K/Q6_K/Q8_0 mat-mat wrappers always requested `8192` bytes, even for dense 27B full-tile prefill chunks.
- New policy requests `5120` for NR1=16 full tiles, `6144` for NR1=32 full tiles, and keeps `8192` for partial tiles. `QWEN_MATMAT_QK_LEGACY_SMEM=1` restores the old behavior for A/B.

Correctness:
- `mat_mat_qk_threadgroup_memory_matches_full_tile_policy` passed.
- Q4_K, Q5_K, Q6_K, and Q8_0 mat-mat correctness tests passed across their existing N cases, including partial `n_query=1` and full `n_query=16/32` coverage.
- Active 27B matrix-G6 prefill correctness passed with `T=32/P=32`.

Commands:

```sh
uv run scripts/profile/prefill_sweep.py \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --n-prompt 4096 --runs 3 --cooldown-seconds 15 \
  --variant legacy-a:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_MATMAT_QK_LEGACY_SMEM=1 \
  --variant smem-new:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --variant legacy-b:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_MATMAT_QK_LEGACY_SMEM=1 \
  --output docs/bench/2026-05-25-matmat-smem-spike/pp4096.json

uv run scripts/profile/prefill_sweep.py \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --n-prompt 16384 --runs 3 --cooldown-seconds 20 \
  --variant legacy-a:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_MATMAT_QK_LEGACY_SMEM=1 \
  --variant smem-new:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --variant legacy-b:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_MATMAT_QK_LEGACY_SMEM=1 \
  --output docs/bench/2026-05-25-matmat-smem-spike/pp16384.json
```

Summary:

| Shape | legacy A | smem-new | legacy B | Read |
| --- | ---: | ---: | ---: | --- |
| `pp512` | `205.77` | `197.30` | `187.52` | too noisy; reversed repeat has smem `186.61/192.05` vs legacy `185.65` |
| `pp1024` | `188.60` | `181.58` | `156.68` | too noisy; reversed repeat has smem `167.51/181.71` vs legacy `175.48` |
| `pp4096` | `202.83` | `204.55` | `191.69` | new is `+0.8%` vs first legacy; second legacy anchor drifted badly |
| `pp16384` | `176.33` | `176.74` | `175.72` | small but consistent `+0.2-0.6%` vs legacy anchors |

Direct mat-mat microbench:

At `N=512`, `dispatches=64`, results are flat (`ffn_gate` slightly down,
`ffn_up/down/attn_qkv` essentially unchanged). At `N=1024`, `dispatches=64`, the
reduced smem policy is a clear microbench win:

| Tensor | legacy 8192 | smem-new | Read |
| --- | ---: | ---: | --- |
| `blk.0.ffn_gate.weight` Q4_K | `19.812 ms` | `14.567 ms` | `1.36x` faster |
| `blk.0.ffn_up.weight` Q4_K | `17.617 ms` | `14.853 ms` | `1.19x` faster |
| `blk.0.ffn_down.weight` Q6_K | `19.733 ms` | `16.119 ms` | `1.22x` faster |
| `blk.0.attn_qkv.weight` Q6_K | `11.194 ms` | `9.462 ms` | `1.18x` faster |

At `N=4096`, `dispatches=16`, the microbench win is smaller:

| Tensor | legacy 8192 | smem-new | Read |
| --- | ---: | ---: | --- |
| `blk.0.ffn_gate.weight` Q4_K | `58.791 ms` | `57.504 ms` | `1.02x` faster |
| `blk.0.ffn_up.weight` Q4_K | `58.477 ms` | `57.469 ms` | `1.02x` faster |
| `blk.0.ffn_down.weight` Q6_K | `62.615 ms` | `62.236 ms` | flat/slightly faster |
| `blk.0.attn_qkv.weight` Q6_K | `38.724 ms` | `38.016 ms` | `1.02x` faster |

Conclusion:
- Keep as a low-risk llama-parity cleanup with modest upside, not a dense breakthrough.
- The branch is correctness-preserving, reduces resource requests on full tiles, and has small microbench support.
- Short/medium end-to-end rows are noisy enough that they should not be used as a promotion claim.
- The end-to-end effect is below the threshold needed to close dense 27B by itself.
- Clean follow-up in `docs/bench/2026-05-25-matmat-smem-clean-v0127/` failed the
  default gate, so the reduced-smem policy is opt-in via `QWEN_MATMAT_QK_LLAMA_SMEM=1`.
