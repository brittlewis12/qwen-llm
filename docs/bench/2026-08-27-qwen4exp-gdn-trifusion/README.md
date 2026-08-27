# Flash-Next GDN Tri-Fusion Falsifier

Decision: **KILL** the default-off experiment and remove its implementation.

## Candidate

Checkpoint `56bc66240f351934fa308e43d9686e65a6850da2` replaced the
per-layer beta-sigmoid projection, alpha R2 projection, and decay kernel with
one Metal kernel. It removed two dispatches from each of 36 GDN layers, or 72
dispatches per token, without removing material weight traffic.

The candidate retained beta's one-SIMD reduction, alpha's existing four-SIMD
R2 reduction, and the exact softplus/decay branch. Focused tests established:

- bit-exact beta, alpha, and decay at released shape `2560 x 48`, including
  values immediately below, at, and above both softplus thresholds;
- bit-exact two-token convolution state, delta state, and final output;
- the named three-to-one dispatch substitution with all other dispatch
  geometries unchanged; and
- real R2-off fallback to the complete baseline kernel sequence.

The benchmark binary was built immediately before acquisition from source
contents later checkpointed byte-for-byte as `56bc662`. The commit followed
acquisition so the measured tree was technically dirty; this packet is a
candidate kill, not promotion evidence.

## Protocol

- Device: Apple M4 Max with unified memory.
- Model: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q3_K_XL, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`; all three local shards were
  verified against their source hashes.
- Storage: external PCIe SSD at `/Volumes/wdblack` with warm page cache.
- Workload: no-thinking chat prompt `Write the numbers from 1 to 100, separated
  by commas.`, 28 prompt forwards, 32 generated tokens, and 31 measured decode
  transitions.
- Controls: `QWEN_MATVEC_F32_LCPP_R2=1`; only
  `QWEN4EXP_GDN_BETA_ALPHA_DECAY_FUSED` differed between arms; layer profiling
  was disabled.
- Order: `B-C-C-B` repeated three times, with five seconds between processes.
- Primary endpoint: complete command-buffer decode GPU time divided by 31
  transitions. Wall generation time was the corroborating endpoint.

Representative command, with `ARM` set to `0` or `1`:

```sh
MODEL=/Volumes/wdblack/weights-archive/qwen3.8-flash-next/UD-Q3_K_XL/\
Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf
QWEN_MATVEC_F32_LCPP_R2=1 \
QWEN4EXP_GDN_BETA_ALPHA_DECAY_FUSED=ARM \
target/release/qwen run -m "$MODEL" \
  --user "Write the numbers from 1 to 100, separated by commas." \
  --no-thinking --max-tokens 32
```

Raw stdout, stderr, and the acquisition summary remain under
`target/profiles/qwen4exp-gdn-trifusion-20260827/`.

## Results

| Run | Arm | Decode GPU total (ms) | GPU ms/transition | Wall generation (ms) |
|---:|:---|---:|---:|---:|
| 1 | B | 1498.491 | 48.3384 | 1567.9 |
| 2 | C | 1499.565 | 48.3731 | 1568.9 |
| 3 | C | 1496.590 | 48.2771 | 1566.1 |
| 4 | B | 1496.361 | 48.2697 | 1565.7 |
| 5 | B | 1499.413 | 48.3681 | 1568.8 |
| 6 | C | 1496.228 | 48.2654 | 1565.0 |
| 7 | C | 1489.920 | 48.0619 | 1559.0 |
| 8 | B | 1502.004 | 48.4518 | 1573.7 |
| 9 | B | 1496.927 | 48.2880 | 1566.2 |
| 10 | C | 1486.337 | 47.9464 | 1555.4 |
| 11 | C | 1498.180 | 48.3284 | 1567.3 |
| 12 | B | 1478.949 | 47.7080 | 1547.7 |

All processes completed with 31/31 GPU intervals. Every generated stdout had
SHA-256
`a43ab8b653ea3e75ed0933b22495869eccd123b3897b80b8a626cb7064d600d9`.

Baseline/candidate median decode GPU time was `48.3132/48.2713 ms` per
transition, for a `0.0419 ms` candidate saving. Means were
`48.2373/48.2087 ms`, a `0.0286 ms` saving. An independent-arm percentile
bootstrap of the median difference used 100,000 draws and seed `0x38fa57`; its
95% savings interval was `[-0.2882, +0.3090] ms`.

The three balanced-block mean savings were `-0.0210`, `+0.2463`, and
`-0.1394 ms/transition`. Wall median saving was `0.0484 ms/transition`, with a
95% bootstrap interval of `[-0.2968, +0.3323] ms`.

## Decision

The predeclared gate killed the dispatch-only thesis when the GPU interval's
upper bound fell below `0.4 ms/transition`. Its observed upper bound is
`0.3090 ms`, and the balanced blocks alternate loss, win, loss. The candidate
therefore receives no permanent flag, pipeline, kernel, or tests.

This result rejects the expected `0.8-1.2 ms/token` gain from this specific
tri-fusion. It does not reject larger epilogue fusions that also eliminate a
material activation pass, and it does not alter packed prefill priority.
