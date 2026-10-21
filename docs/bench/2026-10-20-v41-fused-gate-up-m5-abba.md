# V4.1 fused gate/up decode: M5 ABBA rejection

Candidate `20507865` selects the shared equal-type all-slots gate/up selector for
eligible scalar V4.1 decode, replacing the native two-pass gate then up
dispatches. This is the production selector prepared by
`experiment/v41-fused-native-decode`; its doc explicitly left real model timing
to the parent. On M5 Ultra the change is a regression on both artifacts, so
`f7973281` reverts it. Native decode is unchanged.

## Setup

- M5 Ultra (80-core, 256 GB), Metal compiler `32023.921` (`metalfe-32023.921.6`),
  SDK 27.0, validation off, production lease via `QWEN_METAL_LEASE_WAIT=1`.
- Baseline `4b1fca04` (`~/code/qwen-llm-v41-base`), candidate `20507865`
  (`~/code/qwen-llm-fat-m5`, clean `build-info` status `match` on both).
- Q2: `unsloth-DeepSeek-V4.1-Flash-1017/UD-Q2_K_XL/...-00001-of-00007.gguf`.
- IQ3: `unsloth-DeepSeek-V4.1-Flash-1017/UD-IQ3_XXS/...-00001-of-00007.gguf`.
- `qwen-bench suite --tg 32 -d 2048 --runs 1 --seed 1 -o json`, arm order
  base, fused, fused, base, one fresh load per arm.

## Results (tok/s)

| Quant | base 1 | fused 1 | fused 2 | base 2 | base mean | fused mean | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2 | 22.083 | 21.709 | 21.851 | 22.042 | 22.062 | 21.780 | **-1.28%** |
| IQ3 | 21.400 | 20.409 | 20.381 | 21.530 | 21.465 | 20.395 | **-4.99%** |

The two fused samples agree within 0.6% and both base samples within 0.4%; the
sign is consistent across both adjacent pairs and both artifacts. Each arm
wrote the same per-arm JSON path, so only the final sample per arm retains a
raw JSON file; all eight timing lines were captured from the run log.

## Interpretation

Reusing the batched all-slots GEMV geometry is not a decode win. The fused
recipes assign eight output rows to a 64-thread group (eight threads per row),
while the native scalar GEMV spreads a whole 32-thread SIMD group over one row.
On single-row decode the latency per output row and the extra execution
serialization outweigh the saved dispatch and activation re-read. The result
rejects this selector, not gate/up fusion itself: a decode-shaped fused kernel
(rows per SIMD group and SIMD groups per threadgroup swept independently, as in
the MLX-Serve V4.1 decode GU kernel) remains open.
