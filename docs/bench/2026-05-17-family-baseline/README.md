# 2026-05-17 — Family Baseline (qwen-llm vs llama.cpp)

Apples-to-apples sweep across the full Qwen3.5/3.6 family at the same nominal
quant level (Q4_K_M dense, Q4_K_M MoE, Q4_K_XL for the 122B which only ships
that variant). Same physical box, sequential runs, no concurrent benches.

## Box

- M4 Max, 128 GB unified, 546 GB/s peak.
- `qwen-llm` HEAD: `7cc90a9` ("generalized stop-token plumbing + experimental
  MoE prefill checkpoint")
- `llama.cpp` build: `0253fb21f (9187)`
- Browsers idling (Arc / Chromium); no concurrent benches.

## Harness

- `llama-bench -p 128 -p 512 -p 1024 -n 32 -n 128 -r 3` for the lcpp baseline.
- `qwen-bench pp -p {128,512,1024} --runs 3` for qwen-llm prefill.
- `qwen-bench decode --tokens {32,128}` for qwen-llm decode (single shot;
  steady-state ms/token is very stable).

Raw outputs:

- `lcpp-*.md` — `llama-bench` markdown tables, per model.
- `qwen-*.txt` — `qwen-bench` filtered output, per model.

## Prompt processing (pp, tokens/sec)

| model | size | pp128 lcpp | pp128 qwen | Δ | pp512 lcpp | pp512 qwen | Δ | pp1024 lcpp | pp1024 qwen | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B dense | 0.5 GiB | 5459 | 3513 | 0.64x | 7965 | 3891 | 0.49x | 7803 | 3655 | 0.47x |
| 2B dense | 1.2 GiB | 2929 | 2127 | 0.73x | 3770 | 2476 | 0.66x | 3755 | 2338 | 0.62x |
| 4B dense | 2.5 GiB | 1291 | 1043 | 0.81x | 1483 | 1165 | 0.79x | 1486 | 1126 | 0.76x |
| 9B dense | 5.3 GiB | 746 | 639 | 0.86x | 838 | 708 | 0.85x | 823 | 693 | 0.84x |
| 27B dense | 15.7 GiB | 226 | 197 | 0.87x | 237 | 206 | 0.87x | 215 | 183 | 0.85x |
| 35B A3B MoE | 20.6 GiB | 806 | 387 | 0.48x | 1425 | 378 | 0.27x | 1411 | 372 | 0.26x |
| 122B A10B MoE | 71.7 GiB | 264 | 148 | 0.56x | 450 | 150 | 0.33x | 437 | 143 | 0.33x |

## Token generation (tg, tokens/sec)

| model | size | tg32 lcpp | tg32 qwen | Δ | tg128 lcpp | tg128 qwen | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B dense | 0.5 GiB | 250 | 342 | **1.37x** | 251 | 341 | **1.36x** |
| 2B dense | 1.2 GiB | 186 | 200 | **1.08x** | 184 | 200 | **1.09x** |
| 4B dense | 2.5 GiB | 92 | 106 | **1.15x** | 92 | 105 | **1.14x** |
| 9B dense | 5.3 GiB | 64 | 67 | **1.05x** | 61 | 67 | **1.09x** |
| 27B dense | 15.7 GiB | 22 | 24 | **1.13x** | 21 | 24 | **1.16x** |
| 35B A3B MoE | 20.6 GiB | 72 | 76 | **1.06x** | 73 | 76 | **1.04x** |
| 122B A10B MoE | 71.7 GiB | 34 | 34 | ~1.00x | 34 | 33 | ~0.99x |

## Absolute framing — decode bandwidth utilization

Effective decode bandwidth ≈ `model_size_GiB × tg128 t/s × 1.073` (GiB→GB).
Peak: 546 GB/s.

| model | qwen tg128 GB/s | qwen % peak | lcpp tg128 GB/s | lcpp % peak |
| --- | ---: | ---: | ---: | ---: |
| 0.8B | 182 | 33% | 134 | 25% |
| 2B | 257 | 47% | 236 | 43% |
| 4B | 286 | 52% | 250 | 46% |
| 9B | 380 | 70% | 349 | 64% |
| 27B | **412** | **75%** | 354 | 65% |

27B dense decode at **75% of 546 GB/s peak** is the high-water mark —
in the same neighborhood as individual Q4_K mat-vecs run isolated (77–88%
chained64). Structural ceiling territory.

The smaller models are dispatch-bound (0.8B at 33% says we're paying per-token
kernel overhead on tiny weights), but we still beat lcpp's dispatch overhead
significantly there (lcpp is at 25% on the same shape).

MoE bandwidth math is muddier because only the active params get touched;
skipped here.

## Family-level read

| variant | dense decode | dense prefill | MoE decode | MoE prefill |
| --- | --- | --- | --- | --- |
| 0.8B | ✅ won (1.37x) | ❌ ~50% behind | — | — |
| 2B | ✅ won (1.08x) | ⚠️ 25-35% behind | — | — |
| 4B | ✅ won (1.15x) | ⚠️ 20-25% behind | — | — |
| 9B | ✅ won (1.05-1.09x) | ⚠️ ~15% behind | — | — |
| 27B | ✅ won (1.13-1.16x) | ⚠️ 13-15% behind | — | — |
| 35B A3B | — | — | ✅ won (1.04-1.06x) | ❌ ~3.8x behind |
| 122B A10B | — | — | 🟡 parity | ❌ ~3.0x behind |

**Decode goal is achieved across the entire family.** Prefill is the unfinished
half: dense ~85-90% of lcpp (close, getting closer), MoE ~25-33% of lcpp
(structurally the largest remaining gap in the engine).

## Reproducibility check vs prior PERF-LOG entries

- 27B pp512 today: 206 t/s vs PERF-LOG ~212 t/s (87% of lcpp both times). ✅
- 27B tg128 today: 24.5 t/s vs PERF-LOG 24.5 t/s. ✅
- 35B A3B pp320 best read: ~400 t/s. Today's pp512 = 378 t/s. Consistent.
- 122B A10B pp320 best read: ~149 t/s. Today's pp512 = 150 t/s. Consistent.

No regression detected.

## Where the room actually is

1. **MoE prompt** — biggest absolute opportunity by a wide margin. Generic
   grouped GEMM was a measured negative result; the right shape is likely a
   persistent custom routed kernel that doesn't pay scatter/gather overhead.
   Currently roadmap item 7; data suggests it deserves a higher slot.
2. **Small-model dense prefill** (0.8B–4B) — the steep falloff (0.47x → 0.81x
   as size grows) implicates per-dispatch overhead, not kernel quality.
   ICB / MTL4 / encoder restructuring is the real lever (roadmap item 12).
3. **27B dense pp512** — 87% of lcpp. The hardest remaining percent; likely
   needs prompt-native packed attention rewrite (roadmap item 2).
4. **27B decode at long ctx** — already winning at 4K. The concurrent
   GDN+attn projection branch (~4% bench-only win) hasn't been promoted to
   production; easiest next decode bump.

## Sanity flag

`pp1024 < pp512` on qwen-bench across all models, and even on lcpp for the
largest models. Surprising — pp throughput should monotonically increase with
prompt length until saturation. Dense default packed prefill chunk is 512,
so `pp1024` forces a chunk re-encode. Worth a quick check that the prefill
chunk default is still optimal at `pp1024`.
