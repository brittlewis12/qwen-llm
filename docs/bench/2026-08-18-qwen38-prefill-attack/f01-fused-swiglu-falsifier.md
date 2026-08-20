# F01 — Fused gate+up+SwiGLU Q4_K on Qwen 27B: falsified

Date: 2026-08-18. HEAD `1ecf208` (or later; env-var toggle, no code change).
Machine: M4 Max 40c, 128 GB, AC power. Model:
`/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf` (h=5120, f=17408, Q4_K FFN weights).

## Hypothesis (from `baseline.md` phase attribution)

`encode_ffn_fused_swiglu_q4_K_mm_f32` exists at `metal.rs:13175`, wired into
production dflash packed prefill at `metal_dflash.rs:11703-11798`, but gated at
`hidden <= 2048` (`metal_dflash.rs:337-351`). For Qwen 27B (h=5120) it's
disabled by default. The kernel replaces 3 FFN launches (gate mm, up mm,
silu_mul) with 1 fused launch, avoiding a double-load of `h_pack` (5120 F32 =
20 KB per row × chunk_p rows). Kernel constraints (`metal.rs:13186-13200`)
require `n_in % 256 == 0` (5120 satisfies) and both weights Q4_K (satisfied).

Expected: 15-25% of FFN wall (FFN = 50.4% of prefill), i.e. **~7-12% of total
prefill** if it wins on 27B.

## Measurement

A/B: `qwen-bench pp --model Qwen3.8-27B-Q4_K_M.gguf --n-prompt N --runs 5
--output json` with `QWEN_METAL_LEASE_WAIT=1` on both sides,
`QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=1` on the fused arm only. Single
process per cell, warmup on.

| ctx | OFF (t/s) | ON (t/s) | Δ |
| ---: | ---: | ---: | ---: |
| 512   | 246.02 | 236.78 | **−3.8%** |
| 2048  | 222.92 | 227.96 | +2.3% |
| 4096  | 232.52 | 227.14 | **−2.2%** |
| 16384 | 222.43 | 217.58 | **−2.2%** |

Neutral-to-negative at every context except pp2048 (marginal +2.3%,
approximately noise-band).

## Conclusion

Fused SwiGLU kernel does not win on Qwen 27B (h=5120) at any tested prefill
context. The `hidden ≤ 2048` gate is not conservative — it reflects a real
performance regime. Most likely cause: register / threadgroup-memory pressure
at h=5120 outweighs the h_pack reload savings; the kernel's internal tile
structure was probably tuned for smaller h.

## Action

- **Do not lift the `hidden ≤ 2048` gate default.** The current gate is
  correct for 27B.
- **Fused SwiGLU falls out of Tier 1** in the prefill-attack ranking. Would
  need new kernel work (register-pressure-aware tile at h=5120) — recategorize
  to Tier 3.
- **Zero source changes required.** Env-var-only experiment.

## Artifacts

- Raw JSON: `/tmp/pp{512,2048,4096,16384}-{off,on}.json` (transient; regenerate
  from A/B command above).
- Command line: `QWEN_METAL_LEASE_WAIT=1 QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=1 target/release/qwen-bench pp --model Qwen3.8-27B-Q4_K_M.gguf --n-prompt N --runs 5 --output json`
