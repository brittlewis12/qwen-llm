# Rank-320 HC Up Scheduling Screen

Frozen before first GPU execution. Parent: `358a1609`. Research-only; no
production dispatch, option, persistent-state, or allocation-plan changes.

## Mechanism

Pinned donor `antirez/ds4` commit `9139e2ae58a41503968a500f36f75895c1ba63fc`,
`metal/qwen4.metal:217`, maps one SIMDgroup per hidden coordinate, eight lanes
per branch. Our generic Q8 LCPP up projection maps four SIMDgroups to two rows:
K320 has ten blocks, so SG0 covers eight, SG1 two, SG2/3 none. A fixed four-SG
threadgroup maps to four hidden coordinates instead: 640 groups rather than
5120 plus the separate gated-mean dispatch.

Retain our existing activated low vector (no repeated SiLU/division), native
Q8_0 bytes, local sigmoid, and sequential four-branch mean. Write raw gates for
trace compatibility. This still changes dot-product grouping and reduction;
there is no bitwise-arithmetic claim. Singleton four branches / hidden2560 /
rank320 only. Packed kernels are untouched.

## Numerical Gate

- Twelve independently generated full-shape down/up Q8 sets with offset views:
  83,558,400 weight bytes; up-only 41,779,200 bytes. No unverified cache-size claim.
- Real production lease and wired-memory check precede Metal initialization;
  retain the guard through resource destruction. Exact ignored test, serial.
- Independent F64 decoding of actual Q8 bytes using captured F32 normalized and
  activated low inputs. Compute reference mixed output from reference gates.
  Both incumbent and candidate must pass pointwise `3e-5*(1+abs(reference))`
  and relative RMS <=3e-5, for raw gates AND mixed output. Zero oracle energy
  requires zero error. All finite; no dilution through sigmoid saturation.
- Complete-HC upstream normalized/low outputs remain bitwise across arms.
  Additional zero, tiny, alternating-sign and saturated-low cases; zero scales,
  signed scales, and full signed-byte range occur in weights. Output poison,
  nonzero buffer offsets, canaries, immutable inputs/weights, exact timed replay.
  Hostile cases precede timing.

## Fixed Timing And Decision

For up+mean and complete HC read separately: warm ABBA then measured ABBA, each
command performs eight repeats over the twelve distinct fixtures in fixed order.
Report GPU and encode/submit/wait wall milliseconds normalized to one twelve-read
chain; poison/readbacks/assertions are outside clocks. Complete read includes
norm, down projection, low SiLU, up projection and gated mean, not injection.

GPU complete-read gate: control spread <=5%; >=10% saving AND
`(A-B)*97/12 >=0.5 ms`, for mean and both corresponding pairs. Ninety-seven is
two HC reads per each of 48 layers plus final read: an extrapolation, not native
cost attribution. Leaf timing is diagnostic and cannot substitute for this gate.
Instability is INCONCLUSIVE, budget miss HOLD. No width sweep, gate widening,
control-fishing repeats, full-model loser replay, or new model download.

A pass earns bounded native qualification and complete-MoE observation, not
default promotion or delivery. Rechart after the result; leave split/RMS tuning
parked. Read-only design review: `01a0a13a-f2f8-7673-ace3-4fbfd25a3aef`.
