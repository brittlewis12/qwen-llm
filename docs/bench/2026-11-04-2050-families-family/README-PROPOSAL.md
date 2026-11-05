# Proposed README performance-table update (not applied)

Replace the current `## Performance` snapshot paragraph and table with the
following. The lead decides whether to apply it. Values are qwen-llm `b08ca56d9`
(the engine crates match main `5c3da609`; only bench scripts differ) against
llama.cpp v0.9.0 `cd959bae99` on 2026-11-04, M4 Max 128 GB, same file for both.
The M4 host was shared with other builds during the sweep; `*` marks a cell whose
two ABBA blocks disagreed and which should be re-run before it is treated as
stable. The earlier 2026-11-04 M5 data was thermally throttled and is not used;
a valid M5 rerun from 2026-11-05 is proposed as a separate table because the
Flash-Next artifact differs (GSQ-RCO IQ3_S).

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

If the README also carries M5 numbers, add this separate table rather than
mixing hosts. It uses qwen-llm `cd700105b` and the same llama.cpp v0.9.0 pin.

```markdown
M5 Ultra, 256 GB, 2026-11-05. Thermal pressure level was 0 before and after
every arm. Flash-Next uses the M5-resident GSQ-RCO IQ3_S artifact, not the M4
UD-Q3_K_XL file. DS4-Flash, A10B, Muse and K2 artifacts were not present.

| Model, file size | pp512 | pp4096 | tg128 | tg128 at 8K |
|---|---:|---:|---:|---:|
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S, 77.9 GiB | 1064 (0.66x) | 1093 (0.53x) | 67.1 (1.02x) | 63.8 (1.00x) |
| Qwen3.6-35B-A3B UD-Q4_K_M, 20.6 GiB | 2916 (0.60x) | 3688 (0.60x) | 148.5 (1.06x) | 125.4 (0.96x) |
| Qwen3.8-27B Q4_K_M, 15.9 GiB | 509 (0.44x) | 526 (0.39x) | 49.2 (1.01x) | 45.2 (0.96x) |
| GLM-5.3-Flash UD-IQ3_XXS, 112.1 GiB | 473 (0.58x) | 695 (0.87x) | 59.4 (1.37x) | 55.1 @4K (1.31x) |

These are the same synthetic-token semantics. On M5 decode is at parity or
ahead, while prefill is the gap (0.39-0.87x). Setup and raw data:
[docs/bench/2026-11-04-2050-families-family/m5/](docs/bench/2026-11-04-2050-families-family/m5/README.md).
```

Follow-ups worth deciding separately:

1. Re-run the remaining `*` M4 cells (DS4 qwen @8K, DS4 qwen pp4096) and A10B
   pp4096 on a quiet M4 before making them scoreboard authority.
2. The M5 prefill gap is the first M5 family follow-up: llama.cpp's tensor-API
   path scales far more than ours at pp4096. A focused kernel census, not a
   rerun, is the next step.
