# Prefill-attack baseline — Qwen 3.8-27B-Q4_K_M

First step in closing the ~4× M4-adjusted prefill gap to MLX.fast leaderboard
record. Model: `Qwen3.8-27B-Q4_K_M.gguf` SHA
`7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`.
qwen-llm HEAD: `1ecf208`. lcpp: build 10480 `01818e495`.

Clean run: tree stabilized, rebuilt against `1ecf208`, sweep completed
uncontaminated. **The earlier contamination-affected numbers in the git history
of this file were misleading — the retracted conclusions ("we're at ±13% of
lcpp", "non-monotone scaling") were artifacts.**

## Numbers

`qwen-bench pp --runs 5` (warmup on, single fresh process per ctx). lcpp
`llama-bench -r 5 -n 0`.

| ctx | qwen t/s | σ | lcpp t/s | σ | qwen/lcpp | MLX/qwen | qwen ms/tok |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512   | 240.44 | 6.57 | 220.17 | 8.64 | **1.092** | 3.81× | 4.159 |
| 1024  | 216.88 | 4.59 | 213.90 | 0.29 | 1.014 | 4.23× | 4.611 |
| 2048  | 222.13 | 2.56 | 213.18 | 0.11 | 1.042 | 4.13× | 4.502 |
| 4096  | 227.52 | 1.23 | 211.19 | 0.38 | 1.077 | 4.03× | 4.395 |
| 8192  | 222.80 | 0.13 | 206.98 | 0.31 | 1.076 | 4.12× | 4.488 |
| 16384 | 215.57 | 0.04 | 187.60 | 5.28 | **1.149** | 4.25× | 4.639 |
| 32768 | 201.60 | 0.17 | 153.95 | 19.00 | **1.310** | 4.55× | 4.960 |

## Absolute wall (seconds) at each ctx

| ctx | qwen | lcpp | MLX (est) | qwen−MLX |
| ---: | ---: | ---: | ---: | ---: |
| 512   |   2.13 |   2.33 |  0.56 |   1.57 |
| 1024  |   4.72 |   4.79 |  1.12 |   3.60 |
| 2048  |   9.22 |   9.61 |  2.23 |   6.99 |
| 4096  |  18.00 |  19.39 |  4.47 |  13.53 |
| 8192  |  36.77 |  39.58 |  8.93 |  27.84 |
| 16384 |  76.00 |  87.33 | 17.87 |  58.13 |
| 32768 | 162.54 | 212.85 | 35.73 | 126.81 |

## Reading

1. **Qwen beats lcpp uniformly, and the lead grows with context.** +9% at pp512
   → +31% at pp32768. lcpp's per-token cost climbs faster (large-ctx
   attention body handled worse). Our packed-prefill code path is
   materially ahead of lcpp's approach at long contexts.

2. **The MLX gap is uniform ~4× across all contexts** (3.81× → 4.55×). Not
   context-specific, no chunk-boundary artifacts, no attention-quadratic
   inflection. This means the gap is **per-token-work at every layer**, not
   any specific ctx regime.

3. **ms/tok is near-flat 4.16-4.96** — small monotonic growth with context.
   Attention-body cost (which scales O(N²) per-token) is not dominant even at
   pp32768 on our path. Consistent with #2: the gap lives in the linear
   per-token work (FFN, projections, norms, dispatch), not attention.

4. **Both engines share the ~4× MLX gap.** Since lcpp has been extensively
   kernel-tuned and shares the gap, per-op kernel tuning is NOT the fix. The
   MLX advantage is graph-level: fused RMSNorm+matmul, different attention
   decomposition, activation layout / quantization, batching strategy.

5. **Absolute wall payoff is large.** p01 anchor (1926 tok, pp2048 tier)
   currently spends 9.22 s in prefill. MLX-parity would save ~7 s per anchor
   prefill. Even a 2× improvement (halving the gap) saves ~4.6 s.

## Implication for the MTP program

The 3.8 P0 baseline at HEAD `1ecf208` (rebuild pending) will show p01 D7
charged-total tot× LOWER than the `98aba93` baseline (`docs/bench/`
`2026-08-18-qwen38-mtp-program/baseline.md`) if this prefill change proved
free-relative-to-decode. Two possibilities to check when the next MTP
baseline runs:

- If ref and spec prefill BOTH improved proportionally → tot× unchanged, spec
  wall improved
- If only ref improved (or asymmetrically) → tot× dropped

Either way, published absolute-wall numbers should be preferred over paired
ratios as the reporting metric during the prefill-attack phase.

## Next step

**Prefill-phase attribution on qwen path** to identify which linear per-token
component (FFN? front projections? norms? dispatch overhead?) dominates our
~4.5 ms/tok. Available machinery:

- `metal_dflash.rs:22073 metal_27b_packed_prefill_phase_profile` —
  `#[cfg(test)]` invocation of `run_packed_dense_prefill_phase_profile` on
  dense 27B. Not CLI-exposed. Path of least resistance: run this test
  directly (`cargo test --release ... metal_27b_packed_prefill_phase_profile
  -- --nocapture --ignored`) and read its output.
- `deepseek_v4_metal/prefill.rs:6454 resolve_packed_prefill_layer_stage_samples`
  — full per-layer per-stage Metal-timestamp machinery. Would require
  adaptation to dense Qwen's packed-prefill call sites; overkill for a first
  screen.

Recommended: run the test-only phase profile first, use its output to rank
per-phase cost, then decide whether to invest in the deeper dsv4-style stage
recorder.

## Artifacts

- `run.sh` — sweep runner
- `qwen-pp{512,1024,2048,4096,8192,16384,32768}.json` — 7 qwen result JSONs
- `lcpp-pp.json` — first 4 lcpp cells (truncated by 45-min tool timeout during
  pp8192)
- `lcpp-pp-large.json` — remaining 3 lcpp cells (pp8192, pp16384, pp32768) run
  after the tool timeout
- Model: `/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf`
