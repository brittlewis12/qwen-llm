# Muse Full-R Banked Attention VJP KEEP

Decision: **KEEP** the two-dispatch Metal causal-GQA VJP for the qualified Muse
fitting envelope `T<=16`.

## Mechanism

CPU smooth-F32 replay retains one padded probability sidecar
`[T,q_heads,T]` (32 KiB at T=16). The first Metal dispatch assigns one
SIMDgroup to each `(basis,q_head)`, reuses shared primals and probabilities,
and writes direct Q/gate gradients plus head-private K/V partials. At release
T16/D128, those partials use 16 KiB of threadgroup memory. A second serial
dispatch reduces the 16 mapped query heads into each KV head.

The encoder owns no commands, allocations, waits, or readbacks. It rejects
concurrent custody, zero head width, malformed physical layouts, overlap, and
unsupported geometry.

## Correctness

Model-free release tests compare B1/T3 small geometry and B32/T16 release
geometry against the independent CPU VJP, which recomputes its own softmax.

| Gradient | Worst relative L2 | Worst scaled max |
|:--|--:|--:|
| Q | 1.70e-7 | 2.91e-10 |
| K | 2.88e-7 | 8.38e-9 |
| V | 3.92e-7 | 7.75e-7 |
| gate | 6.71e-8 | 1.40e-9 |

Every cosine is at least `0.9999999999997` apart from benign F64 reporting
roundoff. The scalar CPU path remains the finite-difference-backed oracle; the
K/V reduction only reassociates floating-point additions.

## Performance

The preregistered hard stop was 5 ms GPU at B32/T16. A first-use correctness
command measured `4.964 ms`. After one explicit warmup, the four gated commands
measured `4.016167`, `1.824375`, `1.763750`, and `1.586750 ms`: mean
`2.297760 ms`, max `4.016167 ms`. The last-three mean is `1.724958 ms` and is
descriptive, not a promoted steady-state estimate.

Focused correctness and timing tests complete in `0.17 s` and `0.05 s` after
compilation. No model asset or broad suite ran.

## Disposition

Integrate the primitive into a fixed-scratch full-attention one-block bank.
Add periodic shared-primal RMSNorm and SwiGLU VJPs first. Sliding-block inverse
RoPE remains outside this checkpoint.

Adversarial review: `01a05072-6951-7782-978a-2f274d90f474`.
