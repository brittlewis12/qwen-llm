# Fused Weight-To-Half Producer: Exact, But Scalar Economics Gate Fails

The scalar research producer preserves the tested production arithmetic, but
does not meet its predeclared economics gate at both agent-sized and full
prefill shapes. Close this scalar variant without selecting only the winning
4K cases. Production FFN dispatch, allocation, decoding and defaults are unchanged.

## Exactness First

On M5 Ultra with native TensorOps, a release build using Rust 1.99.1 and SDK
27.0 compares the candidate against the unchanged
`kernel_deepseek_v41_weight` F32 materialization followed by
`HalfScratch::convert`. Both use the existing `-O3 -ffast-math` Metal policy.
The candidate computes `half(roundF32(activation * route_weight))` for READY
rows and positive half zero for every other status.

Frozen correctness source is `3d7fa84179a64ab53954ed5a060cadd5cd927aee`, tree
`c67e3e18ec5ee93bded50d40c4a5213fa4541b24`. Binary SHA-256 is
`0fbaa3fd8da2a18a5009ceb6caa5cfa6128b1c5130bdb44b055d8d7a97af158a`.

All 341 classified CPU checks pass. The single ignored GPU fixture passes
seven fixed cohorts and 141,559,600 raw half-word comparisons, including:

- Both double-rounding witnesses: `(1 + 2^-11)^2` and `(1 + 2^-12)^2`.
- Signed zero, midpoint neighbors, subnormals, overflow and negative weights.
- Failed, pending and poisoned statuses with NaN-poisoned operands.
- Nonzero input/output/status offsets, neighboring status layers, guards,
  unused capacity and live scalar tails at 1K/4K geometry.

Its 0.51 s runtime is correctness execution time, not a performance result.
The largest priced buffer/CPU ledger is 679,788,544 bytes. Shared buffers are
initialized and compared directly after checked completion; there is no
full-size CPU tensor clone. This finite compiler/device proof is not universal
floating-point equivalence or a production integration result.

## Separate Economics Packet

Economics source is `38c53b70b030712a7681408d5ccf1ff2d50da026`, tree
`03e246aff500909efd4156c226614f487550f1cd`. Frozen binary SHA-256 is
`7b78e283a36c6959a4f9f59831e12b82b9fe0702e82eb6f1002657cb5bd66996`.
All 342 classified CPU checks and 21 private controller/parser controls pass.

The fixed synthetic shapes are 1,024 and 4,096 rows, intermediate width 2,304
and top-6. Activations are finite and nonzero; positive route weights sum
exactly to 1.5 per token. All statuses are READY. This is not model routing.

A runs the old weight kernel and half converter; B runs one scalar fused
producer. Both use exactly the same physical half-output range, immutable
inputs and retained F32 scratch allocation. A GPU oracle freezes before the
first B. Every warm and primary call verifies all half bits, input/status
integrity, guards and capacity. A checks F32 products; B must leave that entire
F32 temporary poisoned. Reset, copies, comparisons and reporting are untimed.

Per geometry, one oracle and warm A/B precede primary `A B B A`, `A B B A`.
Warm census checks two incumbent dispatches versus one candidate, with actual
kernel names and grids. Primary calls have no census or diagnostic observer.
All 22 command envelopes are serial, positive and valid; the fixed 5 us
cross-API allowance does not relax the comparative gate.

Every block at both shapes must reduce mean producer GPU latency by at least
10% with no mean inclusive-host regression. Native success is not a win by
itself. An independent parser recomputes the four block decisions.

| Rows | Block | A GPU Mean, us | B GPU Mean, us | GPU Reduction | A Host Mean, us | B Host Mean, us | Pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1,024 | 1 | 440.083 | 465.250 | -5.7186% | 1491.230 | 1579.875 | No |
| 1,024 | 2 | 593.625 | 409.521 | 31.0136% | 1903.125 | 1738.229 | Yes |
| 4,096 | 1 | 2693.458 | 1298.667 | 51.7844% | 3839.312 | 2726.167 | Yes |
| 4,096 | 2 | 2672.271 | 1261.125 | 52.8070% | 4151.667 | 2647.646 | Yes |

The first 1K block fails both conditions. Keep all samples and record a valid
`completed_negative` / `close_leaf`; do not rerun or narrow eligibility to 4K
after seeing the result. The two 1K blocks disagree, but this packet does not
establish a clock-state or instruction-cost explanation for that disagreement.

The native selector reports 1.21 s; supervised child execution through drain
spans 1.279 s, under a 30 s cancellation trigger and 45 s outer acceptance
limit. Safe drain retains the lease until the owned
child exits and can exceed a cancellation limit; late work cannot qualify.
Compilation, freezing and CPU acceptance hold no GPU lease. Original exactness
and economics records remain separate and unchanged; adversarial review checks
their integrity and keeps correctness distinct from continuation.

## What This Does Not Establish

Eliminating the temporary's write/read would remove `8 * N` logical activation
bytes and one producer dispatch in an eligible future integration: 432 MiB at
4K. These are not measured DRAM savings. Both experiment arms retain the F32
allocation, and no production allocation reduction is implemented.

The 4K operator signal is not a whole-FFN or model-throughput result. Synthetic
producer timings cannot be added to production stage envelopes to project an
engine gain. A separately declared vector4 design may test shared indexing and
packed stores; it must earn new exactness and economics evidence rather than
reclassify or repeat the closed scalar result.
