# Protocol 2: Conditioning-Aware Scaled Raw Dot Gate

Frozen after one diagnostic-only capture and before any timing. Protocol 1 and
attempt01 remain FAIL; this is a new screen, not retroactive promotion. Shader,
inputs, weight bytes, launch geometry, timing sequence, and performance floors
are unchanged. See `PROTOCOL.md` and `DIAGNOSTIC.md` for the original protocol
and preserved rationale/capture.

The diagnostic finds 12 incumbent /18 candidate original raw pointwise failures
on the x256 low vector. Raw RMS is8.77e-8/1.26e-7, with exact power-of-two output
scaling for both arms. Both pass the original mixed pointwise/RMS gates. This
supports a conditioning defect in the result-only raw pointwise criterion,
not a candidate-specific anomaly. Candidate raw errors are somewhat larger;
different bound fractions do not establish superior accuracy.

## Only Changed Criterion

Only the x256 hostile raw-dot pointwise criterion becomes the common threshold
`(gamma45_f32 + gamma322_f64) * S / (1-gamma322_f64)` for BOTH arms, where
`S=sum(abs(weight*activated_low))`, independently computed in F64. Original
pointwise violation counts are still reported. Original raw RMS <=3e-5, all
mixed-output gates, and all ordinary/other-hostile pointwise gates remain
unchanged. Require finite observations/reference/S and zero error at zero
reference energy. All numerical gates must pass before the one fixed timing
bracket; no stimulus reduction or timing-based threshold selection.

These are conditional forward-error estimates, not certified compiled-kernel
bounds: `simd_sum` does not specify its reduction tree, and the Metal build
uses fast math. Even a sequential 32-lane first incumbent reduction has at
most approximately42 rounding levels here: one product, eight chunk adds,
one scale product, 31 first-reduction adds, one final nonzero add (only two
subgroups have nonzero totals). Candidate has two products, forty accumulation
levels, three explicit XOR levels:45. The common45 threshold is conservative
under ordinary round-to-nearest arithmetic without range effects. F64 reference
and S allowances are retained. It is a screen, not a model-quality guarantee.

The diagnostic's original m20 incumbent estimate assumed a five-level SIMD
tree and must not be treated as source-proven. Its raw captures remain useful
without that assumption. Artifacts: `target/profiles/qwen4exp-hc-conditioning-94435/`.

Run exact ignored test `qwen4exp_metal::hc_up_screen::hc_up_k320_screen_v2`,
production lease held throughout, with Metal API validation. A useful result
earns bounded native qualification and complete-MoE observation only. Production
routing remains unchanged. Independent review endorses this limited correction:
`01a0a13a-f2f8-7673-ace3-4fbfd25a3aef`.
