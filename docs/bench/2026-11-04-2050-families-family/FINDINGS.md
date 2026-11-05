# Family baseline re-pin: llama.cpp v0.9.0 (2026-11-04)

Hand-written reading of this directory's results; `README.md` is the generated
M4 digest, `SAMPLES.md` lists every retained sample, and the JSON files are the
data. This packet re-pins the *family-profiling baseline* role only. It is not
the GLM same-artifact oracle pin, the dequant codec, or a performance roofline.

## What changed

- Baseline lock: llama.cpp `b11182` (`e9f824d8c0`, 2026-09-25) ->
  `refs/tags/v0.9.0` (`cd959bae99ff4bed22bef2604f45241c09bbb8c8`, build 12110,
  backends `MTL,BLAS`).
- qwen-llm: main `5c3da609` plus this branch's bench-lock, family-harness and
  packet commits; the engine crates are unchanged from main. Bench binary
  identities: M4 `b08ca56d9`, M5 `cd700105b`, both clean.
- Both engines were built from the same checkouts on each host. llama.cpp was
  built by `scripts/bench/ensure_llama_cpp.py`; qwen-bench by
  `cargo build --release --locked -p qwen-cli --bin qwen-bench --bin qwen`.

## Setup

- M4 Max, 128 GB, AC power, macOS 15.7.9. Hardware used for the valid grid.
- M5 Ultra, 256 GB, macOS 27.0.0. Valid 2026-11-05 rerun with thermal pressure
  level 0 before and after every arm; see the M5 section.
- One representative artifact per family, the same GGUF for both engines.
  Depth 4096 is the GLM sparse range; the K2 cell has no llama.cpp
  implementation and is qwen-llm only.
- Protocol copied from 2026-09-25: two ABBA blocks per model, `-r 2` per block,
  10 s cooldown, pp at `-ub 512` and `-ub 2048`, tg128 at each model's
  `tg_depths` (0/8192, plus 32768 for 27B and A3B, 4096 for GLM). The tagged
  ratio uses the better of the two pp micro-batches; the generated README's main
  table uses `-ub 2048`.
- To keep campaigns bounded, pp and tg were separate campaigns, and 27B's
  32768-depth cell was a third campaign. Most campaigns finished in 2-9 minutes.
  Two did not: 27B deep tg took 25 minutes (depth refill dominates) and GLM pp
  took 15 minutes (112 GiB model). The lease was held for those durations.
- M4 was not a quiet host: other squad agents were compiling with LTO during the
  sweep, and the captured load average at the first campaign start was
  `14.87 13.60 16.06` (later 6-9, 8 logged-in users). Engine order and ABBA still
  pair the two engines in time, but absolute numbers are below the 09-25 quiet
  baseline for both engines. This is a condition of this packet, not a clean
  thermal comparison.

## llama.cpp family support

Actual `llama-bench` smoke on the artifact, not code reading:

- **GLM-5.3-Flash UD-IQ3_XXS**: loads and runs on both hosts. `lcpp = true`
  is now recorded for this family.
- **DeepSeek V4.1**: cannot load. Both the production
  `DeepSeek-V4.1-Flash-REAP-272E-MXFP4-Q8Engram.gguf` and the
  `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS` shards fail with
  `array key not found in model: deepseek41.attention.kv_source_layers`
  while reading hyperparameters. `lcpp = false` stays, and no substitute file
  is used.
- **K2 Horizon 7B**: still no upstream implementation; qwen-llm only.

## M4 ratios (tokens/s; qwen / best llama.cpp)

| model | pp512 | pp4096 | tg128 | tg128 @8K | tg128 @32K |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B Q4_K_M | 202 / 213 **0.95x** | 209 / 205 **1.02x** | 21.0 / 22.1 **0.95x** | 19.9 / 13.5 **1.48x*** | 18.8 / 15.3 **1.23x** |
| Qwen3.6-35B-A3B UD-Q4_K_M | 1536 / 1635 **0.94x** | 1769 / 1662 **1.06x** | 117 / 97 **1.21x** | 104 / 88 **1.18x** | 92 / 67 **1.38x** |
| Qwen3.5-122B-A10B UD-Q4_K_XL | 491 / 530 **0.93x** | 436 / 511 **0.85x** | 45.2 / 42.5 **1.06x** | 41.7 / 38.9 **1.07x** | — |
| Qwen3.8-Flash-Next UD-Q3_K_XL | 551 / 630 **0.87x** | 509 / 585 **0.87x** | 46.7 / 43.1 **1.08x** | 43.1 / 40.1 **1.08x** | — |
| DeepSeek-V4-Flash-0731 UD-IQ3_XXS | 130 / 299 **0.43x** | 219 / 295 **0.74x*** | 28.7 / 28.6 **1.01x** | 19.1 / 24.1 **0.79x*** | — |
| Muse-Glimmer-30B Q8_0 | 224 / 263 **0.85x** | 189 / 219 **0.86x** | 16.2 / 16.7 **0.97x** | 15.5 / 16.0 **0.96x** | — |
| K2-Horizon-7B Q4_K_M | 747 / — | 531 / — | 71.0 / — | 56.7 / — | — |
| GLM-5.3-Flash UD-IQ3_XXS | 214 / 226 **0.95x** | 226 / 188 **1.20x** | 29.9 / 25.8 **1.16x** | — | 27.8 / 23.2 @4K **1.20x** |

`*` unstable cell: the two blocks disagree materially. 27B lcpp @8K blocks
were 19.0 and 8.0 t/s (samples 19.9/18.0 then 7.4/8.6); DS4 qwen @8K blocks
were 23.0 and 15.1 t/s. A clean 27B rerun (same protocol) gave qwen 21.1 vs lcpp
19.8 t/s, **1.07x**, matching 09-25; use that, not the main-grid 1.48x. The DS4
cell remains provisional. Every sample is in `SAMPLES.md`; rerun data is under
`stability/`.

## Ratio movement against 2026-09-25

Ratios are qwen / better-of-micro-batches. `qwen abs` and `lcpp abs` are the
absolute throughput changes of each engine against 09-25, which is the only
clean way to separate the two sources of movement in this run.

| model | shape | ratio 09-25 | ratio 11-04 | ratio move | qwen abs | lcpp abs |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | pp512 | 1.01 | 0.95 | 0.94x | -17.9% | -12.8% |
| Qwen3.8-27B | pp4096 | 1.07 | 1.02 | 0.96x | -10.4% | -6.2% |
| Qwen3.8-27B | tg128 | 1.05 | 0.95 | 0.90x | -18.1% | -9.3% |
| Qwen3.8-27B | tg128@d8192 | 1.07 | 1.48* (rerun 1.07) | 1.38x* (rerun 1.00x) | -14.2% | -38.0% |
| Qwen3.8-27B | tg128@d32768 | 1.10 | 1.23 | 1.12x | -11.5% | -21.0% |
| Qwen3.6-35B-A3B | pp512 | 0.97 | 0.94 | 0.97x | +0.1% | +3.4% |
| Qwen3.6-35B-A3B | pp4096 | 1.05 | 1.06 | 1.01x | -1.4% | -2.7% |
| Qwen3.6-35B-A3B | tg128 | 1.19 | 1.21 | 1.01x | +6.0% | +4.7% |
| Qwen3.6-35B-A3B | tg128@d8192 | 1.16 | 1.18 | 1.02x | +5.0% | +2.9% |
| Qwen3.6-35B-A3B | tg128@d32768 | 1.25 | 1.38 | 1.10x | +4.5% | -5.2% |
| Qwen3.5-122B-A10B | pp512 | 0.96 | 0.93 | 0.97x | -0.0% | +3.4% |
| Qwen3.5-122B-A10B | pp4096 | 1.04 | 0.85 | 0.82x | -21.4% | -4.5% |
| Qwen3.5-122B-A10B | tg128 | 1.11 | 1.06 | 0.96x | -1.8% | +2.0% |
| Qwen3.5-122B-A10B | tg128@d8192 | 1.12 | 1.07 | 0.96x | -2.2% | +2.2% |
| Qwen3.8-Flash-Next | pp512 | 0.86 | 0.87 | 1.01x | -0.6% | -2.0% |
| Qwen3.8-Flash-Next | pp4096 | 0.77 | 0.87 | 1.12x | +7.6% | -4.1% |
| Qwen3.8-Flash-Next | tg128 | 0.88 | 1.08 | 1.23x | +28.3% | +4.7% |
| Qwen3.8-Flash-Next | tg128@d8192 | 0.75 | 1.08 | 1.43x | +60.6% | +12.0% |
| DeepSeek-V4-Flash | pp512 | 0.48 | 0.43 | 0.91x | -7.4% | +1.6% |
| DeepSeek-V4-Flash | pp4096 | 0.83 | 0.74* | 0.89x | -13.5% | -3.0% |
| DeepSeek-V4-Flash | tg128 | 1.00 | 1.01 | 1.00x | +0.1% | +0.0% |
| DeepSeek-V4-Flash | tg128@d8192 | 0.85 | 0.79* | 0.93x | -14.7% | -8.2% |
| Muse-Glimmer-30B | pp512 | 0.86 | 0.85 | 0.99x | -1.6% | -0.4% |
| Muse-Glimmer-30B | pp4096 | 0.85 | 0.86 | 1.01x | -4.6% | -5.9% |
| Muse-Glimmer-30B | tg128 | 0.97 | 0.97 | 1.00x | -1.4% | -1.6% |
| Muse-Glimmer-30B | tg128@d8192 | 0.99 | 0.96 | 0.97x | -1.4% | +1.6% |
| K2-Horizon-7B | tg128@d8192 | — | — | — | +545.1% | no lcpp |

## Attribution where the data allows

1. **Flash-Next is the clearest llama.cpp-side change.** llama.cpp's
   `qwen4exp` sparse-attention selection was wrong beyond ~2051 visible tokens
   at b11182. At v0.9.0 the lcpp @8K decode rose from 35.8 to 40.1 t/s (+12%)
   and 27B-scale pp4096 did not collapse. Our Flash-Next decode also rose
   sharply (36.4 -> 46.7, 26.8 -> 43.1), so this ratio move is *both* engines:
   the ratio went from 0.75-0.88 to 1.08. The lcpp share is consistent with the
   upstream selection fix; the larger qwen share is ours (qwen4exp decode path
   changes since 09-25). Without an old-vs-new qwen engine A/B this packet
   cannot split the two numerically.
2. **GLM is a new baseline cell.** v0.9.0 loads the stock UD-IQ3_XXS artifact.
   qwen is at 0.95x on pp512, 1.20x on pp4096 and 1.16-1.20x on decode. The
   earlier separate GLM measurement (prefill 93-98% of llama.cpp, decode 21-25%
   ahead) is directionally consistent, not directly comparable.
3. **A3B is stable or better.** Prefill is unchanged (0.94-1.06x); decode rose
   from 1.19/1.16/1.25 to 1.21/1.18/1.38. Absolute qwen decode rose 4.5-6.0%
   while lcpp rose only 2.9-4.7% at shallow depths and fell 5.2% at 32K. The
   32K improvement is ours, not a llama.cpp gain.
4. **DS4 and A10B prefill remain the weak cells.** DS4 pp512 is still ~0.43x
   and pp4096 ~0.74x with an outlier; block means at pp4096 were 193 and 245,
   so the honest value is nearer 0.8x. A10B pp4096 fell from 1.04x to 0.85x:
   qwen dropped 21% (555 -> 436) while lcpp fell only 4.5% (535 -> 511). The
   lcpp side is close to the 09-25 number, so this is mostly ours, but the busy
   host prevents a clean attribution. Re-measure A10B pp4096 on a quiet host.
5. **K2 long-context decode is fixed in qwen-llm.** At 8K it went from 8.8 to
   56.7 t/s (+545%); short decode and prefill are unchanged. This is entirely
   ours; there is no llama.cpp comparator.
6. **The 27B and A10B ratio drops at shallow shapes are not trustworthy as
   engine regressions.** Both engines were slower than 09-25 under load. The
   27B @8K lcpp cell has a 2.4x block disagreement, and the DS4 @8K qwen cell
   has a 1.5x block disagreement. Those cells need a quiet-host rerun before
   they become scoreboard authority.

## M5 Ultra (2026-11-05, valid)

The 2026-11-04 M5 attempt was thermally throttled (pressure level 4) and is
discarded: no number from it is used here. The host recovered on 2026-11-05.
Thermal pressure level was 0 before and after every arm (`m5/conditions.log`),
load average was 1.43 at the first campaign, and no other benchmark process was
running. The grid ran 17:47-18:16 UTC; the longest campaign was 9:08 (27B deep
tg) and every campaign stayed under the 10-minute target.

- M5 bench identity `cd700105b` (engine crates unchanged from main);
  llama.cpp v0.9.0 `cd959bae99`, build 12110. The qwen build ran under the
  per-machine build lock with `nice -n 10`; the pinned llama.cpp build was
  already in the cache.
- Flash-Next used the artifact present on M5, **GSQ-RCO IQ3_S** (77.9 GiB), not
  M4's UD-Q3_K_XL. A3B, 27B and GLM use the same artifacts as M4.
- DS4-Flash, A10B, Muse and K2 artifacts are not present on M5; they were not
  transferred in this slot.

| model | pp512 | pp4096 | tg128 | tg@8K | tg@32K/@4K |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S | 1064 / 1620 **0.66x** | 1093 / 2077 **0.53x** | 67.1 / 65.9 **1.02x** | 63.8 / 63.9 **1.00x** | — |
| Qwen3.6-35B-A3B UD-Q4_K_M | 2916 / 4898 **0.60x** | 3688 / 6151 **0.60x** | 148.5 / 139.7 **1.06x** | 125.4 / 130.6 **0.96x** | 115.7 / 118.3 **0.98x** |
| Qwen3.8-27B Q4_K_M | 509 / 1158 **0.44x** | 526 / 1338 **0.39x** | 49.2 / 48.6 **1.01x** | 45.2 / 46.9 **0.96x** | 38.3 / 42.3 **0.91x** |
| GLM-5.3-Flash UD-IQ3_XXS | 473 / 811 **0.58x** | 695 / 799 **0.87x** | 59.4 / 43.5 **1.37x** | — | 55.1 / 42.0 @4K **1.31x** |

The M5 splits cleanly by phase. Decode is at parity or ahead: A3B 1.06x, GLM
1.31-1.37x, 27B 1.01x, Flash-Next 1.00-1.02x, with A3B/27B slightly behind at
depth. Prefill is the gap: qwen is 0.39-0.87x of llama.cpp. llama.cpp's M5
tensor-API path scales much more than ours (A3B pp4096 6151 t/s on M5 vs 1662 on
M4; qwen 3688 vs 1769). The M5 baseline therefore points at qwen prefill
tensor-op coverage, not decode, as the first M5 family-packet follow-up. The
generated `m5/README.md` bandwidth table uses the M4 peak constant and should be
ignored; `SAMPLES.md` and the JSON are the data.

## Caveats

- Synthetic tokens on both sides; MoE routing is spread more evenly than real
  text. Ratios can differ on real prompts, especially DS4 and GLM.
- Both engines copy final logits to the host. At `-b 2048`, llama-bench computes
  last-token logits at both batch endpoints; qwen-llm computes the final one.
- Depth fills are untimed. llama-bench restores a cached depth state after the
  first fill; qwen-bench refills per rep.
- The M4 host was shared with other builds. The ABBA design cancels linear
  drift, not a 2.4x block-local stall.
- Not measured: MLX or other engines, output quality, serve-level latency,
  cold start, speculative decoding.
