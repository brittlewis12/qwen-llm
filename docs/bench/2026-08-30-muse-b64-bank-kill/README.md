# Muse B64 Bank KILL

Decision: **KILL** B64 widening and retain production B32. Doubling the bank
preserves exact results but does not reduce released-Q8 command wall enough to
matter.

## Predeclared Gate

The mechanism had to clear a 15% median wall saving on both sliding block 50
and full block 51 at released Q8/T16/R. Each block shared one prepared replay;
B64 was compared with two serial B32 calls over the same 64 cotangents. Timing
scopes included validation, allocation, command completion, readback, and the
finiteness scan, while output concatenation remained outside both arms.

The private seam first passed model-free full/sliding J/R equivalence. B64 and
concatenated B32 outputs were bitwise equal. Tiny-model B64 appeared 21-35%
faster, which proved non-predictive at released weight shapes.

## Released Result

Block 50 completed before the gate stopped:

| Arm | Samples | Median |
| --- | --- | ---: |
| B64 | `115.490, 115.501, 117.599 ms` | `115.501 ms` |
| two B32 | `114.572, 116.645, 117.387 ms` | `116.645 ms` |

B64 saves only `0.98%` (`1.0099x`), far below the 15% floor. All values are
bitwise equal. Because promotion required both blocks, block 50 alone falsifies
the global mechanism; the run intentionally did not spend another model pass
on block 51.

The complete failed-gate process finishes in `2.32 s`. It writes no artifact or
checkpoint, invokes no identity path, and hashes no model bytes.

## Interpretation

B64 doubles scratch and work while approximately doubling command time. Fixed
bank-call overhead is negligible; the incumbent Q8 kernel already shares each
16x64 weight tile across 128 query rows, and wider outer banks merely schedule
more of the same workgroups. Increasing outer row width is therefore not an
optimization mechanism.

The private B64 helper, model-free gate, and released harness were removed in
full after KILL. Production remains B32 with no experiment branch or runtime
switch.

## Next Gate

Test reuse inside the Q8 kernel instead: a private model-free 256-query tile
against the incumbent 128-query tile at released FFN gate/up and down shapes,
with `n_query=512`. Require bitwise output equality and 15% median GPU-time
savings in both directions over five alternating samples. No model asset is
needed.

Adversarial design and KILL review:
`01a0516b-475c-7291-92c4-7017a79fa3d8`.
