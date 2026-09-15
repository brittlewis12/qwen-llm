# Conditioning Diagnostic, Not Qualification

Subsequent review: the m20 incumbent estimate below assumes a five-level
`simd_sum` reduction tree that source alone does not establish. These are
conditional estimates, not certified compiled-kernel bounds. See `PROTOCOL-2.md`
for the explicitly revised common criterion; original observations stay intact.

Attempt01 fails the original frozen screen on the incumbent's scaled-low raw
gate, before timing and before the candidate's scaled-low comparison. Preserve
that failure; do not reinterpret it as a candidate failure or performance result.
All twelve ordinary fixtures and zero/tiny/alternating-sign cases pass both arms.

One diagnostic-only run retains the original first fixture and x256 low vector,
captures both complete raw/mixed outputs before any numerical assertions, and
reports all original gate failures unchanged. There is no timing bracket. F64
reference gates/mixed outputs and sum of absolute products are retained too.

## Independent Error Model, Frozen Before Diagnostic

F32 unit roundoff u=2^-24; gamma(m)=m*u/(1-m*u). For S=sum(abs(w*x)), use
gamma(m)*S, not the potentially cancelled abs(sum(w*x)). These are conservative
source-operation depth bounds, not multipliers selected to fit observed errors:

- Incumbent: at most one quant/input multiplication, eight sequential adds in
  a chunk, one scale multiply, five SIMD reduction levels, five final SIMD
  reduction levels: m=20. Each K320 lane visits at most one block.
- Candidate: two multiplies per term, forty sequential accumulation levels,
  three XOR reduction levels: m=45. FMA can reduce rounding, not worsen this
  conservative separate-operation bound.
- Include gamma(322) with F64 unit roundoff2^-53 for the independent reference
  and inflate computed S by 1/(1-gamma64). Inputs/terms are finite and normal
  (or exact zeros) in this deterministic case; overflow/underflow is not a
  claimed domain. These bounds presume ordinary round-to-nearest arithmetic.

Also test bitwise raw output x256 scaling from each arm's original run; powers
of two preserve rounding absent range effects. Report raw RMS, original
pointwise failures, conditioned-bound failures/fractions, condition number,
mixed pointwise/RMS, and scaling failures separately. Diagnostic test success
only means capture completed; it does NOT mean original qualification passed.

If either candidate-specific anomalies or original strict mixed/RMS gate
failures occur, park HC and move to MoE. Otherwise, consider an explicitly
versioned, predeclared conditioning-aware protocol, preserving attempt01 as
failed. A forward-error model is not a model-quality gate.

Read-only review: `01a0a13a-f2f8-7673-ace3-4fbfd25a3aef`.
