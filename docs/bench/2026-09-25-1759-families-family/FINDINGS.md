# Cross-family baseline: qwen-llm vs llama.cpp b11182 (2026-09-25)

Hand-written reading of this directory's results; `README.md` is the generated
digest and the JSON files are the data. First baseline beyond the Qwen size
ladder, and the first against llama.cpp since b9833 (2026-06-28).

## Setup

- M4 Max 128 GB, AC power. qwen-llm `70ec9a9b` (clean), llama.cpp b11182
  `e9f824d8c0` (MTL, BLAS; flash attention `auto`; residency sets on by default).
- One daily-driver artifact per family (`scripts/bench/models-families.toml`),
  the same GGUF for both engines.
- llama-bench semantics on both sides: synthetic random tokens, pp at depth 0,
  tg128 at depth 0 / 8192 (and 32768 for dense 27B and A3B), untimed depth fill.
- qwen-llm times the production calls (`LoadedModel::prefill`/`decode_token`
  for Qwen, the family runtimes otherwise), including the host logits copy.
- 2 ABBA blocks x 2 reps per engine (4 samples per cell); engine order
  alternates by block and by model; 10 s cooldown between commands.
- llama.cpp pp ran at `-ub 512` (its default) and `-ub 2048`; the ratio below is
  against the **better** of the two per cell.

## Results (tokens/s; ratio = qwen-llm / best llama.cpp)

| model | pp512 | pp4096 | tg128 | tg128 @8K | tg128 @32K |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B dense Q4_K_M | 246 / 244 **1.01x** | 233 / 218 **1.07x** | 25.6 / 24.4 **1.05x** | 23.2 / 21.7 **1.07x** | 21.3 / 19.4 **1.10x** |
| Qwen3.6-35B-A3B UD-Q4_K_M | 1535 / 1581 **0.97x** | 1793 / 1709 **1.05x** | 109.9 / 92.3 **1.19x** | 98.8 / 85.1 **1.16x** | 88.3 / 70.6 **1.25x** |
| Qwen3.5-122B-A10B UD-Q4_K_XL | 491 / 513 **0.96x** | 555 / 535 **1.04x** | 46.0 / 41.6 **1.11x** | 42.6 / 38.1 **1.12x** | — |
| Qwen3.8-Flash-Next UD-Q3_K_XL | 554 / 642 **0.86x** | 473 / 610 **0.77x** | 36.4 / 41.1 **0.88x** | 26.8 / 35.8 **0.75x** | — |
| DeepSeek-V4-Flash-0731 UD-IQ3_XXS | 141 / 295 **0.48x** | 254 / 305 **0.83x** | 28.7 / 28.6 **1.00x** | 22.4 / 26.3 **0.85x** | — |
| Muse-Glimmer-30B Q8_0 | 228 / 264 **0.86x** | 198 / 233 **0.85x** | 16.5 / 17.0 **0.97x** | 15.7 / 15.8 **0.99x** | — |
| K2-Horizon-7B Q4_K_M | 754 / — | 543 / — | 71.0 / — | **8.8** / — | — |

Block means agree within 3.2% for every cell except Flash-Next tg128 (qwen
blocks 34.7 / 38.0 t/s, 9.1%).

## What it says

1. **The Qwen3.5/3.6/3.8 families are still ahead, by much less than in June.**
   Decode leads narrowed from the June board (b9833, default flags):
   A3B tg128 1.42x -> 1.19x, dense 27B 1.26x -> 1.05x. Short-prompt prefill is
   now at or slightly below parity on MoE (A3B 0.97x, A10B 0.96x at pp512);
   pp4096 stays ahead (1.04-1.07x). The lead grows with depth (A3B 1.25x at
   32K, 27B 1.10x), so long-context decode is where the Qwen path is strongest.
2. **Every family added since June is behind llama.cpp somewhere.**
   - DS4 prefill: 0.48x at pp512, 0.83x at pp4096. Decode is at parity at depth
     0 but 0.85x at 8K. This is the largest gap on the board.
   - Flash-Next: behind on every cell (0.75-0.88x), with two internal
     anomalies: our pp4096 is *lower* than our pp512 (473 vs 554; llama.cpp is
     flat at 610), and decode loses 26% from depth 0 to 8K where llama.cpp
     loses 13%.
   - Muse: prefill 0.85-0.86x; decode at parity.
3. **K2 decode collapses with context: 71.0 t/s at depth 0, 8.8 t/s at 8K**
   (8x). No llama.cpp comparator (upstream has no K2 implementation), but no
   other family loses more than 26% at 8K, so this is a K2 decode path problem,
   not a hardware limit.

## Caveats

- **Residency.** llama.cpp creates Metal residency sets by default
  (`keep_alive = 180 s`); qwen-llm's DS4 residency set stays opt-in (owner
  decision, 2026-09-24), and the other families do not use one. August's DS4
  measurements put residency at roughly +20-25% prefill; it cannot account for
  a 2x pp512 gap on its own.
- **Synthetic tokens.** Random token ids spread MoE routing across experts
  more evenly than real text; both engines see the same kind of input, but
  real-prompt ratios can differ, especially for DS4 (256 experts) and
  Flash-Next.
- **Output boundaries.** Both engines copy logits to the host. At `-b 2048`,
  llama-bench's pp4096 computes last-token logits at both batch endpoints;
  qwen-llm's production prefill computes the final one.
- **Depth fills.** Untimed on both sides; llama-bench restores a cached depth
  state after the first fill while qwen-bench refills per rep.
- **Harness defects fixed after this run** (throughput and ratios are
  unaffected: they come from the merged samples): merged rows in this run keep
  block 0's `stddev_ns`, `avg_compute_ns`, allocation averages and
  `decode_gb_per_s`; block-to-block execution-mode invariance was not
  recorded. Block means agreeing within 3.2% (one 9.1% cell) make a mode
  change between blocks unlikely.
- **Not measured:** MLX or any other engine, output quality, serve-level
  multi-turn latency and reuse, cold start, speculative decoding.
