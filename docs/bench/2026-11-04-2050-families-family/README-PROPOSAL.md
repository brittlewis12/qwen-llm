# Proposed README performance-table update (not applied)

Replace the current `## Performance` snapshot paragraph and table with the
following. The lead decides whether to apply it. Values are qwen-llm `b08ca56d9`
(the engine crates match main `5c3da609`; only bench scripts differ) against
llama.cpp v0.9.0 `cd959bae99` on 2026-11-04, M4 Max 128 GB, same file for both.
The M4 host was shared with other builds during the sweep; `*` marks a cell whose
two ABBA blocks disagreed and which should be re-run before it is treated as
stable. M5 Ultra is deliberately absent: its 2026-11-04 host measured 7-12x
below the 2026-10-16 first-light reference and its data is not a baseline.

```markdown
A snapshot, not a current guarantee: qwen-llm `b08ca56d9` against llama.cpp
v0.9.0 (`cd959bae99`) on 2026-11-04, M4 Max 128 GB, same file for both. Tokens
per second for qwen-llm, with the ratio to the better of llama.cpp's `-ub 512`
and `-ub 2048` runs. The host was shared with other builds during the sweep;
the 27B 8K cell uses a clean stability rerun (1.07x); the remaining `*` cells had
materially disagreeing ABBA blocks and need a quiet-host repeat.

| Model, file size | pp512 | pp4096 | tg128 | tg128 at 8K |
|---|---:|---:|---:|---:|
| Qwen3.8-27B Q4_K_M, 15.9 GiB | 202 (0.95x) | 209 (1.02x) | 21.0 (0.95x) | 19.9 (1.07x) |
| Qwen3.6-35B-A3B UD-Q4_K_M, 20.6 GiB | 1536 (0.94x) | 1769 (1.06x) | 117 (1.21x) | 104 (1.18x) |
| Qwen3.5-122B-A10B UD-Q4_K_XL, 71.7 GiB | 491 (0.93x) | 436 (0.85x) | 45.2 (1.06x) | 41.7 (1.07x) |
| Qwen3.8-Flash-Next UD-Q3_K_XL, 83.8 GiB | 551 (0.87x) | 509 (0.87x) | 46.7 (1.08x) | 43.1 (1.08x) |
| DeepSeek-V4-Flash-0731 UD-IQ3_XXS, 97.1 GiB | 130 (0.43x) | 219 (0.74x)* | 28.7 (1.01x) | 19.1 (0.79x)* |
| Muse-Glimmer-30B Q8_0, 27.6 GiB | 224 (0.85x) | 189 (0.86x) | 16.2 (0.97x) | 15.5 (0.96x) |
| GLM-5.3-Flash UD-IQ3_XXS, 112.1 GiB | 214 (0.95x) | 226 (1.20x) | 29.9 (1.16x) | 27.8 @4K (1.20x) |
| K2-Horizon-7B Q4_K_M, 5.2 GiB | 747 (—) | 531 (—) | 71.0 (—) | 56.7 (—) |

These are synthetic-token tests with llama-bench's semantics, not request
latency, and several rows have changed since. K2-Horizon has no llama.cpp
implementation and is qwen-llm only. Setup, caveats and raw data:
[docs/bench/2026-11-04-2050-families-family/](docs/bench/2026-11-04-2050-families-family/FINDINGS.md).
```

Two follow-ups worth deciding separately:

1. Re-run the remaining `*` cells (DS4 qwen @8K, DS4 qwen pp4096) and A10B
   pp4096 on a quiet M4 before making them scoreboard authority.
2. Re-run the M5 Ultra grid once the host passes a 0.8B sanity check; until then
   the README should not claim M5 family ratios.
