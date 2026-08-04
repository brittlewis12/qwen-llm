# DeepSeek V4 Shallow Sparse-Selector Cliff

Date: 2026-08-04

Status: production-policy `GO` on Apple M4 Max.

## Scope

Sparse CSA begins when compressed row 513 becomes visible, at token position
2,051. Production previously retained the one-thread scalar top-512 selector
through row 1,024. That selector removes one worst row per pass, so its work in
this band grows as `O((visible - 512) * visible)`. Production returned to the
parallel selector only at row 1,025, creating a bounded decode cliff from token
positions 2,051 through 4,095.

The candidate treats the first pruned row as a complexity firewall: production
uses the existing four-bit parallel radix selector whenever
`max_visible_rows > top_k`. The scalar and one-bit parallel selectors remain
test differentials. No Metal kernel, score producer, selected-attention path,
cache representation, memory plan, snapshot ABI, or numerical contract changes.

## Identity

- Base revision: `ec40edb87cce431d5dce0374e88d3bc8f2da1d64`.
- Candidate source and this packet are captured by the same checkpoint commit.
- Preserved scalar-policy binary SHA-256:
  `29fc1ce6c3c4d196702e8ab9d348a3d555c6b21a995d8ef7a8cc0da1698edb75`.
- Candidate binary SHA-256:
  `70f26b7c1537770db229889517d922d2176a6c1340a7c4faa7e8c0c190cddbcb`.
- Model shard 1 SHA-256:
  `dec1cee704800267d9d836d5a61aefc33705be939bbb3058fa9006d98191576d`.
- Request SHA-256:
  `144d4e3753dd0c2c7d114a3f38ce0c26c47ad2a8b2c8f5aed63d57bda48e0270`.
- Hardware: MacBook Pro `Mac16,5`, Apple M4 Max, 128 GB unified memory.
- OS: macOS 15.6.1 (24G90), arm64.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)` and
  `cargo 1.97.1 (c980f4866 2026-06-30)`.
- Metal: Apple metal 32023.864 (`metalfe-32023.864`), target
  `air64-apple-darwin24.6.0`.

The logs do not embed executable hashes. This manifest binds the binaries above
to the run order and filenames below. The baseline executable was copied and
hashed before rebuilding the candidate from the same worktree.

## Product Protocol

The exact 2,385-token prompt runs scalar/candidate/scalar in separate processes
with `--reasoning none`, `-n 32`, and `--seed 42`. No other GPU workload runs
during the bracket. All three processes use:

```text
model=/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
messages=/Users/tito/models/battery-logs/qwen-llm-qual/ds4-old-vs-fresh-2026-08-04/perf_02k.json
QWEN_DSV4_PROMPT_IDS=1
QWEN_DSV4_TOP_LOGITS=8
```

Run order and retained logs:

1. `/tmp/qwen-dsv4-scalar-selector-baseline` ->
   `perf_02k_scalar_warm_a.log`
2. `target/release/qwen` -> `perf_02k_radix513_warm.log`
3. `/tmp/qwen-dsv4-scalar-selector-baseline` ->
   `perf_02k_scalar_warm_b.log`

The logs live under
`/Users/tito/models/battery-logs/qwen-llm-qual/ds4-old-vs-fresh-2026-08-04`.

## Product Result

| Arm | Prefill, ms | Generation, ms | Decode, token/s | Transition, token/s |
|---|---:|---:|---:|---:|
| scalar before | 90,694.4 | 4,780.3 | 6.69 | 6.50 |
| radix at row 513 | 90,239.4 | 1,414.7 | 22.62 | 22.14 |
| scalar after | 89,861.8 | 4,789.8 | 6.68 | 6.49 |

The scalar midpoint is 4,785.05 ms generation and 6.685 token/s. The candidate
reduces generation time by 70.4% and raises decode throughput by 3.38x. Its
prefill time lies between both scalar arms, so the policy change shows no packed
prefill regression in this bracket.

All three 32-token streams are byte-identical when encoded as signed i32
little-endian values, at SHA-256
`f87ca5a5d4e7787951cb02b979b54987a39d9968ca03ee577116e86f3e5e67ae`.
All eight first-token top-logit records are text-identical, at SHA-256
`b5499d064f8efb2b7115d1235b8398b4ccdec6a96903a4940819a0fb56050eb0`.

## Operation Bracket

An uncontended release profiler alternates scalar/radix/scalar on deterministic
finite scores. Each median has 12 retained samples after three warm calls per
arm. Outputs, selected counts, and statuses are exact after every row count.

| Visible rows | Scalar midpoint, ms | Radix, ms | Saving, ms/layer |
|---:|---:|---:|---:|
| 513 | 0.833 | 0.199 | 0.634 |
| 520 | 0.921 | 0.064 | 0.857 |
| 528 | 1.444 | 0.064 | 1.380 |
| 544 | 2.019 | 0.044 | 1.975 |
| 576 | 3.733 | 0.044 | 3.689 |
| 640 | 7.999 | 0.044 | 7.955 |
| 768 | 18.522 | 0.045 | 18.477 |
| 896 | 31.698 | 0.045 | 31.653 |
| 1,024 | 47.527 | 0.045 | 47.482 |

This demonstrates the old complexity shape; it is not a claim that radix is the
fastest possible selector at every individual row count.

Exact profiler command:

```bash
cargo test --release -p qwen-llm --lib \
  deepseek_v4_metal::tests::profile_shallow_sparse_selector_crossover \
  -- --ignored --exact --nocapture
```

## Correctness Gates

- `shallow_scalar_parallel_and_radix_selectors_match_cpu_contracts` compares
  scalar, one-bit parallel, four-bit radix, and the CPU contract from 513 through
  1,024 rows. Cases include ascending/descending/mixed scores, exact cutoff ties,
  signed zero, subnormals, and NaN/infinity fallback.
- `packed_sparse_visibility_tracks_publication_cadence` derives exact packed
  visibility `[513, 513, 513, 513, 514]`, checks sparse suffix offsets, and
  rejects a final published-row mismatch.
- `packed_publication_cadence_uses_the_same_scalar_and_production_selection`
  proves that the batch maximum does not alter any mask, cache-order ID, count,
  or status across that cadence.
- Existing 1,024-row and mixed packed selector gates remain exact. The focused
  release suite passes, as do formatting, `git diff --check`, and strict release
  Clippy.

CX review session `019fced5-fbcf-7a40-8b1d-62fdfc9fe360` returns `GO` with no
correctness blocker. Promotion is scoped to the M4 Max; the existing
32-lane/256-thread selector geometry continues to fail closed on incompatible
Metal devices.

## Decision

Promote the first-pruned-row production policy. The 2K/3K slowdown was a narrow
selector-policy cliff, not ordinary attention scaling or a cold-cache transient;
the 6.7K and 32K observations were already beyond the old row-1,024 switch and
therefore back on the parallel schedule.

The bounded short-context kernel-tuning KILL remains valid at contexts 128 and
512. This repair is a structural complexity guard outside those measurements,
not a reopening of sub-threshold quant-kernel shape sweeps. Resume the
force-ranked official FP4 indexer Metal shadow next.
