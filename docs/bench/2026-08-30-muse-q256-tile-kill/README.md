# Muse Q256 Weight-Tile Reuse KILL

Decision: **KILL** 256-query weight-tile reuse and retain the incumbent
128-query Q8 VJP kernel. Doubling SIMDgroups per threadgroup is slightly slower
at the first released FFN shape.

## Predeclared Gate

The private candidate kept the incumbent 16-input by 64-output shared weight
tile and per-query accumulation order, but used eight SIMDgroups to share it
across 256 queries. The control used four SIMDgroups and 128 queries. Both ran
at `n_query=512` with one warmup and five alternating command-GPU samples.

Promotion required bitwise equality and at least 15% median improvement at both
released FFN directions: `6656x19968` gate/up and `19968x6656` down. A small
Q256/Q128 release test passed bitwise before the shape gate.

## Result

Gate/up stopped the packet:

| Arm | Samples | Median |
| --- | --- | ---: |
| Q256 | `13.5920, 13.5992, 13.5995, 13.6186, 13.6357 ms` | `13.5995 ms` |
| Q128 | `13.4798, 13.4840, 13.4919, 13.4960, 13.5018 ms` | `13.4919 ms` |

Q256 is `0.8%` slower (`0.9921x`), decisively below the 15% floor. Because both
directions were required, the packet did not spend more work on down. The full
model-free gate finishes in `0.31 s`; no model asset, artifact, identity path,
or model hashing is involved.

## Interpretation

The incumbent already amortizes each dequantized 16x64 weight tile across 128
queries. Doubling query sharing halves threadgroup count but loses enough
occupancy to erase that reuse. Combined with the B64 KILL, this shows neither
outer-bank width nor query-axis threadgroup width is a live lever.

The candidate Metal entry point, cfg-test wrapper, bitwise test, and shape
harness were removed in full. Production remains on the qualified Q128 kernel.

## Next Gate

Keep Q128 and widen the input tile instead: private C32/Q128 versus incumbent
C16/Q128 at the same two released FFN shapes and `n_query=512`. A 32-input tile
may halve input-axis threadgroups and cotangent reloads while revisiting each Q8
block scale once. Require bitwise equality and 25% median GPU-time improvement
in both directions over five alternating samples; no model asset is needed.

Adversarial design and KILL review:
`01a0516b-475c-7291-92c4-7017a79fa3d8`.
