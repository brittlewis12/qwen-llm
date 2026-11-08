# Fused QKV disposition, 2026-11-08

Delete `QWEN_PREFILL_ATTN_FUSED_QKV_G8`, its duplicate banks, packed-output
scratch, selector and exclusive split kernel. Shared QKV/GDN, tensor and
attention primitives remain.

Measured source: `8d1d34011208d3c93ad2a948d35dfbd046f9811f`, main
`8883a612` plus [the characterization patch](qkv-probe.patch). The patch is
archival evidence, not an installed kernel or runtime option. Apply it to
`8883a612` with `git apply --unidiff-zero` when reproducing the probe.
[Driver](run-qkv-probe.py); raw samples and dispatch/numeric checks:
[M4](qkv-m4.json), [M5](qkv-m5.json). Binary/source/SDK identities and thermal
summaries: [M4](qkv-m4-manifest.json), [M5](qkv-m5-manifest.json).

Both hosts loaded the same real Qwen3.6-35B-A3B-UD-Q4_K_M artifact (733 tensors,
40 layers, no MTP): its ten attention blocks have **Q8_0 Q/K/V banks**. This is
an A3B Q8 projection comparison, not a full Q8-expert checkpoint claim. Input
activations are deterministic, bounded, nonzero values; every Q/K/V output is
compared. [Per-tensor SHA256](qkv-weights.json) agrees for all thirty projection
banks on both hosts; [fingerprint reader](qkv-fingerprint.py). Eight repetitions per timed command, two ABBA or ABCCBA packets,
four samples per arm. Table values are median GPU ms per layer, including split.

| Rows | M4 separate | M4 fused | M5 separate tensor | M5 original fused | M5 tensor-fused + split |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 5.693 | 6.055 | 0.924 | 2.814 | 1.006 |
| 4096 | 11.300 | 12.051 | 1.801 | 5.622 | 2.021 |

The first separate sample shows a clock ramp; all raw samples are retained.
Subsequent separate samples agree. M4 fusion loses 6.4–6.6%; M5's current
shape whitelist misses fused 9216 output channels, so the original fusion loses
~3x against three tensor dispatches. Explicitly composing the shared tensor
primitive with the fused bank still loses 8.8–12.2%. That diagnostic tensor
composition is bitwise equal to separate tensor outputs. M4 is also bitwise;
M5's older SIMDgroup fused arm has max absolute delta 6.92e-6, cosine >.999999999998.

The bank adds **200,540,160 resident bytes** (~191 MiB), retaining the separate
weights; construction costs 25.226 ms M4 / 23.244 ms M5. Packed output adds
**75,497,468 / 150,994,940 bytes** over the old one-float placeholder at the two
row widths. Metal's priced and observed buffer bytes agree. Allocation time is
charged separately in the scratch records. The bank is eagerly built even for
short requests; actual execution requires G8 and total position >=4096.

Builds and timed packets each held the shared build lock. Timings additionally
held the canonical Metal lease, ran nice 10, and monitored thermal pressure 0
and no rustc throughout. M5 acquired both nonblocking and released between
busy retries so other lanes could alternate. No end-to-end speedup is claimed:
all measured implementations lose before model-level overhead.
