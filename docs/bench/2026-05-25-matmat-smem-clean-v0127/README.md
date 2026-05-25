Clean v0.127 mat-mat smem A/B follow-up.

Purpose:
- Rebuild `qwen-bench` from clean commit `fcab600` and retest the v0.127
  full-tile threadgroup-memory policy against the legacy always-8192-byte policy.
- The clean gate was run because dirty microbenches were positive but dirty
  end-to-end rows were too noisy to justify a default change.

Commands:

```sh
cargo build --release -p qwen-cli --bin qwen-bench

uv run scripts/profile/prefill_sweep.py \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --n-prompt 1024 --runs 3 --cooldown-seconds 10 \
  --variant smem-new-a:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --variant legacy:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_MATMAT_QK_LEGACY_SMEM=1 \
  --variant smem-new-b:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --output docs/bench/2026-05-25-matmat-smem-clean-v0127/pp1024.json

uv run scripts/profile/prefill_sweep.py \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --n-prompt 4096 --runs 3 --cooldown-seconds 15 \
  --variant smem-new-a:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --variant legacy:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_MATMAT_QK_LEGACY_SMEM=1 \
  --variant smem-new-b:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --output docs/bench/2026-05-25-matmat-smem-clean-v0127/pp4096.json

uv run scripts/profile/prefill_sweep.py \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --n-prompt 16384 --runs 3 --cooldown-seconds 20 \
  --variant smem-new-a:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --variant legacy:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_MATMAT_QK_LEGACY_SMEM=1 \
  --variant smem-new-b:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --output docs/bench/2026-05-25-matmat-smem-clean-v0127/pp16384.json
```

Summary:

| Shape | smem-new A | legacy | smem-new B | Read |
| --- | ---: | ---: | ---: | --- |
| `pp1024` | `221.09` | `210.26` | `202.98` | high drift; no robust default win |
| `pp4096` | `189.00` | `204.65` | `204.51` | first new run bad, second new equals legacy |
| `pp16384` | `190.83` | `191.68` | `188.44` | flat/slightly worse |

Conclusion:
- Clean end-to-end evidence does not justify the reduced-smem policy as default.
- Keep the dirty microbench result as a useful diagnostic, but flip the reduced-smem
  policy behind opt-in env instead of carrying it as the production path.
- Follow-up code uses legacy `8192` bytes by default and enables the llama-style
  full-tile policy only with `QWEN_MATMAT_QK_LLAMA_SMEM=1`.
