# Muse C32 Input-Tile KILL

Decision: **KILL** the C32 input tile and retain the incumbent C16/Q128 Q8
VJP kernel. The wider tile remains bitwise equal on the small correctness gate
but takes more than five times as long at the first released FFN shape.

## Predeclared Gate

The private candidate doubled the input tile from 16 to 32 values while
retaining 128 queries, 64 output rows, and the incumbent accumulation order.
The control remained C16/Q128/K64. Both ran at `n_query=512` with one warmup
and five alternating command-GPU samples.

Promotion required bitwise equality and at least 25% median improvement at both
released FFN directions: `6656x19968` gate/up and `19968x6656` down. A small
C32/C16 release test passed bitwise before the shape gate.

## Result

Gate/up stopped the packet:

| Arm | Samples | Median |
| --- | --- | ---: |
| C32 | `66.7192, 68.0857, 68.2898, 69.3075, 71.9168 ms` | `68.289750 ms` |
| C16 | `13.4982, 13.4986, 13.5068, 13.5097, 13.5155 ms` | `13.506750 ms` |

C32 achieves only `0.197786x` incumbent throughput and takes `5.06x` as long.
Because both directions were required, the packet did not spend more work on
down. The stopped model-free gate invocation finishes in `0.65 s`; no model
asset, artifact, identity path, or model hashing is involved.

## Interpretation

The result is consistent with an occupancy or register-pressure collapse from
doubling the input tile. Together with the B64 and Q256 KILLs, this rejects the
preregistered B64, Q256, and C32 widening candidates around B32/C16/Q128.

The candidate Metal entry point, cfg-test wrapper, bitwise test, and shape
harness were removed in full. Production remains on the qualified C16 kernel.

## Next Gate

Keep C16/Q128/K64 arithmetic and change only Q8 block layout. Use deterministic
nonzero, block-varying Q8 records and separate outputs. Warm pack, candidate,
and control once, then take five alternating paired trains. Each candidate
train packs the resident row-major tensor into a preallocated reusable
`[input_block,output_row,34-byte block]` sidecar and executes eight dependent
dispatches; each control train executes eight incumbent dispatches. Measure
both trains with the same enclosing command-GPU clock, compare complete outputs
bitwise, and require candidate median at or below `0.75x` control median for
both released FFN directions at `n_query=512`. No model asset is needed.

Adversarial design and KILL review:
`01a0516b-475c-7291-92c4-7017a79fa3d8`.
