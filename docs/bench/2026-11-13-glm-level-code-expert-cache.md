# GLM real expert level-code cache: bitwise, stage gate fails

Status: **closed; no cache implementation**. The packed level-code path is a stored re-encoding of real IQ expert weights. It is distinct from the 2026-11-12 fast-decode candidate, which kept the IQ bytes and changed only the decoder. That same-byte decoder was bitwise but 1-16% slower; it still paid codebook access and added select/sign work ([prior record](2026-11-12-iq-fast-decode.md)). This candidate removes the large grid lookup from the measured path, but expands each value into scalar level codes and sign bits.

## Re-encoding and bitwise witness

The exporter read only selected expert tensor slices from the stock four-shard GGUF. It copied the original `d` and scale fields unchanged. The offline packer used the local IQ grid tables to write 2-bit IQ2_S level indices or 3-bit IQ3_S level indices, followed by a sign plane. There is no requantization or scale change.

| Format | Native block | Level-code block | Native bpw | Level-code bpw | Relative bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ2_S | 82 B | 106 B | 2.5625 | 3.3125 | +29.27% |
| IQ3_S | 110 B | 134 B | 3.4375 | 4.1875 | +21.82% |

The sampled five expert triplets (gate/up IQ2_S, down IQ3_S) grew from **44,892,160 B** to **56,688,640 B**, or **2.8542 -> 3.6042 bpw** across the three matrices. Across the full mixed GLM bank estimate, retaining IQ4_XS down matrices native, the duplicated level-code cache is **137.73 GB** versus **109.87 GB** native.

Selected real route samples:

| Rows per expert measured | Layer / expert | Observed rows in source trace |
| ---: | ---: | ---: |
| 7 | 3 / 33 | 7 at 80K, width 256 |
| 28 | 3 / 36 | 28 at 80K, width 1,024 |
| 57 | 3 / 1 | 57 in the 48K root chunk |
| 161 | 8 / 218 | 161 in the 48K root chunk |
| 512 | 39 / 179 | 509 in the 48K root chunk (512-row stress measurement) |

Across 15 real matrices, the device witness compared **125,829,120 decoded Half values** and **6,266,880 tile output floats**: zero bit mismatches in both witnesses.

## Isolated stage timing

M5 Ultra, same limiter tile body and geometry, native-IQ `h0` versus stored level codes. For every projection and row bucket: one warmup, then three AB/BA/AB pairs, four dispatch repetitions per sample. Measurements ran at nice 0; thermal pressure settled and stayed at 0. The first v1 attempt stopped on an uppercase IQ3_S label check and is excluded. The complete v2 packet used probe source `0c5fe8b99`, SHA256 `b79ec8865f83a8b1d05bd5abae240ef47c9fa8d479b5ae1b11c688b91f5ac00e`.

Each table entry sums the gate, up and down projection GPU times for the selected expert. Speedup is native / level-code; the pair column gives the three interleaved stage ratios.

| Rows | Native median-sum | Level-code median-sum | Median paired speedup | Three paired speedups |
| ---: | ---: | ---: | ---: | --- |
| 7 | 0.5560 ms | 0.6303 ms | 0.8851x | 0.8801x, 0.8851x, 1.0139x |
| 28 | 0.2905 ms | 0.3280 ms | 0.8854x | 0.8854x, 0.8885x, 0.8814x |
| 57 | 0.4302 ms | 0.4855 ms | 0.8844x | 0.8582x, 0.8872x, 0.8844x |
| 161 | 0.6176 ms | 0.7173 ms | 0.8611x | 0.8575x, 0.8611x, 0.9392x |
| 512 | 0.6893 ms | 0.8811 ms | 0.7897x | 0.7897x, 0.7820x, 0.8113x |

**Gate fails at every row count.** Level codes are 11-27% slower in the paired stage medians, missing the required 1.25x speedup. No whole-cell A/B or serve gate was run.

## Why the stored re-encoding loses

The re-encoding removes the large grid-codebook lookup, unlike the prior same-byte decoder, but it adds 22-29% bytes and replaces a compact grid lookup with per-weight bit extraction and sign-plane reads. IQ3_S needs 3-bit fields that straddle bytes; their shifts, masks and extra byte traffic cost more than the removed codebook gathers. At 512 rows the native stage is **0.689 ms** and level codes take **0.881 ms**.

The corrected hot-set model uses the actual per-format block sizes, the 48K root trace repeated 72 times for three sessions, and the 28 completed GLM plan calls (17,903 appended tokens); the calculator is `scripts/bench/glm-tensor-experts/level_code_coverage.py`. At a 57 GB budget the whole-expert set covers **78.04%** of total routed work; the work in projections actually accelerated by this candidate is **76.27%** for the root, **74.80%** for append-256, and **72.91%** for append-1,024. At 60 GB those speeded-work coverages are 78.05%, 76.55% and 74.73%.

Applying the measured median paired stage ratios to those coverages and the prior routed-stage wall shares gives a first-order regression projection: **-3.6%** at the root chunk, **-5.2%** at append-256 and **-3.6%** at append-1,024. These are modeled projections, not measured wall changes. The isolated stage gate failed, so no positive wall claim is made.

V4.1 remains outside this test. The one-line profiler cause is its architecture check: it accepts `deepseek4`, while the V4.1 artifact reports `deepseek41`. Adding a V4.1 route path is not a trivial change.

Raw M5 files: `metagross:~/qwen-runs/2026-11-12-glm-prefill/decoded-expert-cache/level-code/` (`raw/`, `encoded/`, `results/real-expert-level-codes-v2/`, pinned identity and hashes). The result closes the decoded-expert-cache line; no runtime default or production code changed.
