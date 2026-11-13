# IQ fast decode: bitwise-identical decoders fail the tile gate

Status: **negative; nothing shipped.** The cheaper IQ2/IQ3 decoders are
bit-equal to the incumbents on device, but on the standalone tensor tile they
are 1.01-1.16x **slower**, so swing #1's required >=1.15x speedup fails. No
production decoder, dispatch, default or environment knob was added, and no
whole-model, append, or decode campaign was started.

Branch `perf/iq-fast-decode` at `d6948d880`, based on `5ee5c8c4f` (local
`main` was inspected before the worktree was created). M4 Max local; M5 Ultra
for tensor execution. Every M5 build and GPU pass queued on the canonical
`with-metal-lease` (`--task iq-decode`); the longest hold was 15 minutes and
the campaign waited behind other lanes rather than bypassing them.

## Claim under test

The 2026-11-12 MLX prefill review argued that the IQ codebook decoders can be
made ~2x cheaper without changing bits:

1. grid magnitudes are tiny integer sets, so they convert exactly to F32;
2. `half(dl * float(mag) * sign)` equals a single narrowed product with the
   sign bit XORed afterwards, because F32 negation is exact and
   round-to-nearest-even is sign-symmetric; and
3. the IQ2 grids have only three magnitudes, so three narrowed levels per 16
   weights can be precomputed and selected.

We verified the premises directly:

- `iq2_xxs_grid`, `mv_iq2xs_grid` and `qt_iq2s_grid` contain only bytes
  `{8, 25, 43}`, so `byte >> 4` is the 0/1/2 level index.
- `qt_iq3xxs_grid` contains `{4, 12, ..., 62}` (`byte >> 3` = level 0..7);
  `qt_iq3s_grid` contains `{1, 3, ..., 15}` (`byte >> 1` = level 0..7).
- The five routed formats therefore all satisfy the sign-symmetry argument.

The candidate implements both ends of that idea: a **level** variant
(precomputed IQ2 levels + sign-bit XOR) and a **vec** variant (one
`as_type<uchar4>` -> `float4` product + sign-bit XOR). IQ3 uses the vec form
because it has eight levels. The losing implementation is retained only in
`scripts/bench/glm-tensor-experts/iq_fast_decoders.metalh`; the production
headers `kernels/quant_tiles.h` and `kernels/tensor/v41_sorted_experts.metalh`
were restored to `main`.

## Method

The 2026-11-12 standalone counterfactual harness
(`scripts/bench/glm-tensor-experts/`) was extended with:

- `fast` and `vec` arms instantiated over the maintained K32 tensor body, so
  geometry, staging, barriers, MMA and epilogue are unchanged and only the
  decoder differs;
- a fast-vs-production tile witness that requires **zero** bit mismatches;
- all five routed IQ formats (IQ2_S/XS/XXS, IQ3_S/XXS) instead of the original
  IQ2_S/IQ3_S pair;
- a decode-only mode, plus a Metal 3.1 decode library so the M4 can run the
  shared dequantizers without the Metal 4 tensor runtime.

The route census is the same 81,920-token GLM snapshot used by the
2026-11-12 limiter packet (`routes.jsonl`, 42 layers per width). The full
`holds` run covers 32/161/256/512 rows at 42 layers. A shorter `screen` run
covers 32/161/512 rows at six layers for the vec arm. Both use four dispatch
repetitions per timing sample and rotate arms across layers and reps.

## Exactness

Zero mismatches everywhere.

| Check | Coverage | Result |
| --- | --- | ---: |
| Decode, M5 | 8,388,608 halves per format; fast arm on IQ2_S/XS/XXS and IQ3_S/XXS | 0 |
| Decode, M4 (Metal 3.1) | same five formats; fast+vec on IQ2, fast on IQ3, repeated after the harness-only split | 0 |
| Tile, M5 | 336 fast witnesses: IQ2_S/IQ3_S x 32/161/256/512 x 42 layers | 0 |

The tile witness compares the full `slot_limit x M` F32 output against the
unmodified production pipeline; the decode witness compares every narrowed
Half. This confirms the review's bitwise premise; it does not confirm its
instruction cost.

## Tile timing

Matched full control `h0` is the reference (the same probe body with the
incumbent decoder). `fast` is the IQ2 level variant / IQ3 vec variant. Times
are summed per-layer medians over 42 layers; ratio is `time / h0`.

| Rows | dtype | h0 ms | fast ms | fast / h0 |
| ---: | --- | ---: | ---: | ---: |
| 32 | iq2_s | 40.20 | 46.76 | **1.163** |
| 161 | iq2_s | 76.87 | 87.16 | **1.134** |
| 256 | iq2_s | 87.74 | 98.89 | **1.127** |
| 512 | iq2_s | 111.24 | 125.02 | **1.124** |
| 32 | iq3_s | 44.23 | 44.98 | **1.017** |
| 161 | iq3_s | 82.52 | 83.62 | **1.013** |
| 256 | iq3_s | 96.95 | 98.07 | **1.012** |
| 512 | iq3_s | 124.93 | 126.41 | **1.012** |

The level-select IQ2 form is 12-16% slower; the IQ3 vec form is ~1% slower.
The six-layer screen confirms the simpler IQ2 vec form is the better of the
two candidates but still loses:

| Rows | iq2_s h0 ms | fast / h0 | vec / h0 |
| ---: | ---: | ---: | ---: |
| 32 | 7.04 | 1.159 | **1.034** |
| 161 | 12.44 | 1.139 | **1.033** |
| 512 | 15.97 | 1.121 | **1.029** |

(The screen's own `h3`/`h4` labels in the raw `screen-fast.jsonl` are one hold
behind; the committed harness now uses explicit hold codes. The counterfactual
shares below use the full `holds` run.)

## Why it loses

The full run reproduces the limiter's decode share. Against `h0`:

| Rows | dtype | constant weights (`h3`) | resident quant (`h4`) | decode + codebook (`h4 - h3`) |
| ---: | --- | ---: | ---: | ---: |
| 32 | iq2_s | 0.639 | 0.942 | **0.303** |
| 512 | iq2_s | 0.617 | 0.903 | **0.286** |
| 32 | iq3_s | 0.569 | 0.889 | **0.320** |
| 512 | iq3_s | 0.507 | 0.867 | **0.360** |

So decoder work is 29-30% of the IQ2_S tile and 32-36% of IQ3_S. To reach a
1.15x tile, a decoder rewrite must be about **1.5-1.8x faster**, not merely
"fewer source operations". The measured candidates do not reduce the critical
path: the level tree adds compare/select work, and the sign-bit XOR adds a
divergent 16-entry constant load that the incumbent's arithmetic sign mask
does not pay. This is the same warning the limiter packet recorded: without
hardware counters, decoder ALU cannot be separated from codebook-load
instruction throughput, and the source-level op count was not a useful proxy.

## Decision and limits

- **Gate: fail.** Best case is 1.03x slower (IQ2 vec); the required speedup is
  1.15x. No format passes.
- No production path, no `QWEN_IQ_FAST_DECODE` knob, no whole-model bitwise
  pair, and no wall campaign (pp4096, 2,048-row/48K, 1,024-row/80K) were run,
  because the operator gate stops promotion first.
- The 4096-row arm and the V4.1 sorted-tile microbenchmark were also not run
  after the IQ2_S/IQ3_S operator gate failed; the decode witness already
  covers the shared V4.1 formats bitwise, and the tile share is measured on
  the same decoder functions.
- B=1/B=3 decode was not measured: no decoder reached a production path, so
  there is no default to qualify. The incumbent decode paths are untouched.
- The losing decoder stays in the harness header as the exact measured
  candidate. It is not included by any production metal file.

## Reproduction

```sh
# M4, no tensor runtime; build and run the decoder witnesses/timings
nice -n 10 python3 scripts/bench/glm-tensor-experts/build.py "$OUT/controls"
QWEN_METAL_LEASE_WAIT=1 scripts/bench/with-metal-lease --task iq-decode --for 5m -- \
  "$OUT/controls/probe" "$OUT/controls/decode.metallib" - decode

# M5, tensor tile; full counterfactual set and the short vec screen
nice -n 10 python3 scripts/bench/glm-tensor-experts/build.py "$OUT/controls"
QWEN_METAL_LEASE_WAIT=1 scripts/bench/with-metal-lease --task iq-decode --for 15m -- \
  "$OUT/controls/probe" "$OUT/controls/probe.metallib" "$OUT/routes.jsonl" holds
QWEN_METAL_LEASE_WAIT=1 scripts/bench/with-metal-lease --task iq-decode --for 10m -- \
  "$OUT/controls/probe" "$OUT/controls/probe.metallib" "$OUT/routes.jsonl" screen
```

Raw records: `metagross:~/qwen-runs/2026-11-12-iq-decode/`
(`decode-m5.jsonl`, `holds-fast.jsonl`, `screen-fast.jsonl`,
`build-controls.log`, `build2.log`, plus the local M4
`decode-m4*.jsonl`). The M5 bundle, worktree and build output are removed
after the packet was copied back. `scripts/check-contracts.sh` was run on the
final commit; `docs/ENV.md` was regenerated (revision line only).
