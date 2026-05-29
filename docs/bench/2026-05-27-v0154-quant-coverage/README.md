# v0.154 Quant Fast-Path Coverage Audit

Purpose: make quant/dtype/shape fast-path coverage explicit before trusting
prefill benchmarks. `scripts/profile/prefill_sweep.py` runs the static GGUF audit
by default and stores the row in sweep JSON; use `--require-fastpath-clean` to
fail before benchmarking when coverage gaps are present or the audit cannot run.
Without `--require-fastpath-clean`, audit tool failures warn and the sweep
continues.

The dense/GDN/attention/lm-tail primitive path now covers `F32`, `F16`, `BF16`,
`Q2_K`, `Q3_K`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ4_NL`, and
`IQ4_XS`. The new low-bit/IQ kernels are coverage-first scalar primitives, not a
claim of llama.cpp parity.

## Target Family

| Model | Dense FFN | GDN | Attention | MoE Grouped | LM Tail | Read |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Qwen3.6-35B-A3B-UD-Q4_K_M | n/a | n/a | n/a | 40/40 | yes | target MoE covered |
| Qwen3.5-122B-A10B-UD-Q4_K_XL | n/a | n/a | n/a | 48/48 | yes | target MoE covered |
| Qwen3.6-27B-Q4_K_M | 64/64 | 48/48 | 16/16 | n/a | yes | target dense covered |

## 0.8B Quant Family

| Quant | Dense FFN | GDN | Attention | LM Tail | Read |
| --- | ---: | ---: | ---: | --- | --- |
| F32 | 24/24 | 18/18 | 6/6 | yes | full precision covered |
| F16 | 24/24 | 18/18 | 6/6 | yes | native half mat-vec/mat-mat/get-rows |
| BF16 | 24/24 | 18/18 | 6/6 | yes | native bfloat mat-vec/mat-mat/get-rows |
| Q2_K | 24/24 | 18/18 | 6/6 | yes | native low-bit K mat-vec/mat-mat |
| Q3_K_M | 24/24 | 18/18 | 6/6 | yes | native low-bit K mat-vec/mat-mat |
| Q4_0 | 24/24 | 18/18 | 6/6 | yes | native legacy quant mat-vec/mat-mat |
| Q4_1 | 24/24 | 18/18 | 6/6 | yes | native legacy quant mat-vec/mat-mat |
| Q4_K_S | 24/24 | 18/18 | 6/6 | yes | existing K-quant path |
| Q4_K_M | 24/24 | 18/18 | 6/6 | yes | representative target quant |
| Q6_K | 24/24 | 18/18 | 6/6 | yes | existing K-quant path |
| Q8_0 | 24/24 | 18/18 | 6/6 | yes | existing Q8 path |
| IQ4_NL | 24/24 | 18/18 | 6/6 | yes | native IQ4 mat-vec/mat-mat |
| IQ4_XS | 24/24 | 18/18 | 6/6 | yes | native IQ4 mat-vec/mat-mat |
| UD-Q8_K_XL | 24/24 | 18/18 | 6/6 | yes | mixed F16/Q8 path covered |

## Wider Local Inventory

`target/profiles/v0154-local-qwen-fastpath-audit.tsv` shows dense 2B/4B/9B/27B
Q2/Q3/Q4/Q6/Q8/BF16/IQ4 variants covered by the primitive path. Remaining local
coverage gaps are not 0.8B dense primitives: they are alternate MoE quant banks
(`Q6_K/Q8_0` or `IQ3_*` gate/up with `IQ4_XS` down) and UD low-bit dense variants
that contain `IQ2_*`/`IQ3_*` tensors.

## llama.cpp Spot Anchors

Same-machine, one-row `pp128`, no-warmup spot anchors show the coverage-first
low-bit/IQ kernels are useful but not tuned parity yet:

| Quant | qwen-llm t/s | llama.cpp t/s | qwen/lcpp | Read |
| --- | ---: | ---: | ---: | --- |
| Q2_K | 932.36 | 1119.96 | 0.83x | native path clears F32 fallback, still below lcpp |
| Q3_K_M | 1201.97 | 1444.27 | 0.83x | same scalar-primitive gap class |
| IQ4_NL | 882.55 | 1642.50 | 0.54x | largest obvious parity gap |
| IQ4_XS | 895.88 | 1649.74 | 0.54x | largest obvious parity gap |

## Validation

- `cargo fmt --check`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- `cargo test --release -p qwen-llm prefill_tokens_matches_single_token_loop_0_8b -- --nocapture`
- `cargo test --release -p qwen-llm mat_vec_and_mat_mat_half_weights_match_cpu -- --nocapture`
- `cargo test --release -p qwen-llm mat_vec_and_mat_mat_q4_legacy_match_cpu -- --nocapture`
- `cargo test --release -p qwen-llm mat_vec_and_mat_mat_q3_k_match_cpu -- --nocapture`
- `cargo test --release -p qwen-llm mat_vec_and_mat_mat_q2_k_match_cpu -- --nocapture`
- `cargo test --release -p qwen-llm mat_vec_and_mat_mat_iq4_match_cpu -- --nocapture`
- `cargo test --release -p qwen-llm mat_mat_q5_k_matches_cpu_and_mat_vec -- --nocapture`
- `cargo test --release -p qwen-llm mat_mat_q8_0_matches_cpu_and_mat_vec -- --nocapture`
- `prefill_sweep.py --require-fastpath-clean` pp128 smokes passed for 0.8B `Q2_K`, `Q3_K_M`, `IQ4_NL`, and `IQ4_XS`.
- `qwen-bench tg8` smokes passed for 0.8B `Q2_K`, `Q3_K_M`, `IQ4_NL`, and `IQ4_XS`.
- `llama-bench -p 128 -n 0 -r 1 --no-warmup -o json` anchors captured for 0.8B `Q2_K`, `Q3_K_M`, `IQ4_NL`, and `IQ4_XS`.

Takeaway: the 0.8B local quant family is now explicit and clean for dense
prefill/decode primitive coverage. The remaining across-board coverage work is
MoE grouped support for alternate expert bank quants and optional IQ2/IQ3 dense
primitive coverage for UD low-bit variants. The remaining performance work is
optimization: especially IQ4, where current scalar primitives are about half of
llama.cpp at pp128.
