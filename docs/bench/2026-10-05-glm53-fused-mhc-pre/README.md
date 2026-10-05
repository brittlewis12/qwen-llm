# GLM-5.3-Flash fused mHC pre (2026-10-05)

Leverage map 2026-10-05 #2. UD-IQ3_XXS, M4 Max 128 GB, AC power. Parent
`a23a9202` (five-dispatch pre) vs `b1ff3b8e` (fused). Same session, back to
back, untraced, `QWEN_METAL_LEASE_WAIT=1`, no other GPU work.

Reproduce (each arm checked out and rebuilt):

```sh
qwen-bench suite -m <shard 1> --pp 32,128,512 --tg 128 -d 0,4096 --runs 3
qwen-bench suite -m <shard 1> --pp 4096 -d 0 --runs 2
```

## Change

Each of the 90 sub-blocks per token ran five serial dispatches:

1. RMS over the 16,384 flattened residual values;
2. the Q8_0 mix projection to 24 mixes;
3. the controls, with one GPU thread running 20 Sinkhorn rounds through
   device memory;
4. the collapse;
5. the block RMSNorm.

The fused pre is two dispatches (`metal::encode_mhc4_pre_q8_0`):

- split-K partials, one 256-thread threadgroup per (256-value chunk, row),
  writing the chunk's sum of squares and its 24 dots with the raw
  residual;
- one 1024-thread threadgroup per row, which reduces the partials and
  scales the dots by the RMS factor. Thread 0 then runs the controls in
  registers, and all threads do the collapse and the block norm.

Decode (rows = 1) and packed prefill (both lineages) share the kernels.
Scratch drops the [16384] ones row and the [16384, rows] normalized rows.
DS4 is unchanged.

Exactness: **Numerical (requalified)**.
- The RMS scale now follows the dot.
- The register-resident controls differ from the device-memory kernel by
  up to 35 ulps in the combination under fast math. That is why it is not
  a bitwise drop-in for DS4.
- Given the same gates, the collapse and the 1024-thread block norm match
  the standalone kernels bitwise.

Live GLM gates under `MTL_DEBUG_LAYER=1` (14/14):

- Exact packed == serial decode (max|dlogit| 0 in all 9 cases);
- near-4096 Exact prompt-end KL 9.3e-9, worst 9.0e-5, top-1 33/33;
- packed sparse vs llama.cpp 1.7e-5 at 2092 (gate 1e-2);
- frontier top-1 46/46; selection replays exact.

## Screen (`metal::mhc::tests::mhc4_pre_dispatch_costs`)

Warm GPU, GLM width, 90 chained repetitions, µs per sub-block:

| Path | µs |
|---|---:|
| fused (2 dispatches) | 8.3 |
| five dispatches | 27.6 |
| - controls (one thread) | 10.1 |
| - Q8_0 mix | 6.8 |
| - RMS over 16,384 | 6.4 |
| - block RMS | 2.7 |
| - collapse | 1.4 |

## Results (tok/s, mean ± stddev of 3; pp4096 2 runs)

| Test | Depth | a23a9202 | b1ff3b8e | Change |
|---|---:|---:|---:|---:|
| tg128 | 0 | 27.54 ± 0.05 | **29.70 ± 0.09** | **+7.8%** (36.31 → 33.67 ms) |
| tg128 | 4096 | 26.66 ± 0.07 | **28.45 ± 0.07** | **+6.7%** (37.51 → 35.15 ms) |
| pp32 | 0 | 59.86 ± 0.23 | 59.45 ± 1.04 | -0.7% |
| pp128 | 0 | 126.53 ± 0.31 | 126.73 ± 0.41 | +0.2% |
| pp512 | 0 | 201.11 ± 4.58 | 203.51 ± 2.13 | +1.2% |
| pp32 | 4096 | 55.43 ± 1.72 | 56.55 ± 1.52 | +2.0% |
| pp128 | 4096 | 103.72 ± 1.08 | 108.26 ± 0.30 | +4.4% |
| pp512 | 4096 | 173.35 ± 0.56 | 174.91 ± 1.15 | +0.9% |
| pp4096 | 0 | 179.11 ± 3.96 | 182.74 ± 0.02 | +2.0% |

Decode attribution v3 (`../2026-10-05-glm53-decode-attribution/attribution-v3.json`):
mHC pre (attention + ffn) goes from 4.13 to 1.33 ms per token at depth 64.
The unprofiled step goes from 38.58 to 35.97 ms at depth 64, and from 40.60
to 38.03 ms at depth 4096.

## Decision

Promote. Decode gains 2.4–2.6 ms per token. Packed prefill is neutral or
better: the 1024-thread finish threadgroups that idle through the
single-thread Sinkhorn do not cost packed rows anything measurable, which
was the review's concern. The next decode costs are weight streaming:
kda 12.0, routed experts 10.0, shared expert 2.6, MLA 3.2 and head 1.0 ms.
