# GLM Fast/Half Append Numerics: Code Sensitivity Starts at Sparse MLA

Status: numerical audit and probe only. **No production behavior changed.**
GLM-5.3-Flash UD-IQ3_XXS, M5 Ultra, 48K prefix, iter profile. The real code
prompt is `~/qwen-runs/2026-11-07-serve-agents/code.txt`; README and the V4.1
causal-price note are prose controls. Appends use the same token IDs and a
snapshot restored before each arm.

The `glm_append_audit --layers` mode adds three same-prefix arms: ordinary
1-token decode, one-row packed prefill repeated once per token, and the
8-row packed prefill. It captures every post-block mean-stream activation and
router IDs. `--layers-f32` keeps the prefix and sequential reference Fast/Half
and switches only the packed suffix to F32 operands. The probe and its
feature-gated route accessor are in
`crates/qwen-llm/examples/glm_append_audit.rs` and
`crates/qwen-llm/src/glm5_next_metal.rs`.

## Accepted floor and measured control

The PERF-LOG does **not** define a nonzero Fast/Half noise floor. The adopted
policy requires bitwise identity for repeats within one lineage and schedule;
a non-bit-identical schedule change must pass a non-inferiority cohort against
both the incumbent Fast schedule and the versioned Exact reference. The
2026-10-08 512-row Fast qualification passed its registered cohort against
Exact (mean NLL +0.0004; central 90% upper bound +0.0046), but that is a quality
qualification, not an arithmetic noise allowance for append schedules.

The one-row packed control gives an observed same-lineage schedule delta for
these exact tokens. It is prompt-dependent and is not an accepted universal
bound:

| Code append arm vs sequential decode | Mean KL | Max KL | Mean \|dNLL\| | Max \|dNLL\| | State KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-row packed control | 4.363e-3 | 2.255e-2 | 0.05330 | 0.12223 | 2.200e-2 |
| 8-row Fast/Half | 4.359e-3 | 1.827e-2 | 0.11297 | 0.40413 | 7.589e-2 |
| 8-row F32 suffix, same Half prefix | 4.970e-8 | 3.955e-7 | 2.381e-5 | 7.341e-5 | 5.556e-9 |

The 8-row Half arm matches the one-row control on mean KL, but its mean
absolute dNLL is 2.1x higher, its worst dNLL is 3.3x higher, and its state KL
is 3.4x higher. It is therefore outside this measured control on two
teacher-forced NLL metrics and state, even though KL alone does not distinguish
the two. This does not establish a project-wide numeric floor.

README stays bitwise at the output: 8-row Half mean KL 2.4e-8, max KL
1.5e-7, mean |dNLL| 5.1e-8, max |dNLL| 1.8e-7, state KL 1.2e-8. The
V4.1 causal-price prose prompt is similarly near zero. Both prose prompts do
show internal route and activation changes; those changes do not materially
move their final logits.

## First divergence and likely cause

The code prompt's first post-block relative RMS above 1e-3 is block 7
(`MLA` mixer, `MoE` FFN): 1.90e-3 for both one-row packed and 8-row Half.
Router IDs first differ from sequential decode at that same block (1 of 8
rows, 1 of 64 expert slots). Routes then diverge across more rows and blocks.
The Half path therefore reaches the shared sparse MLA/indexer-to-MoE-router
boundary with a code-sensitive activation; this is upstream of the expert
kernel, consistent with the prior finding that grouped and packet expert
kernels are not the offender.

F32 operands on the 8-row suffix from the *same 48K Fast/Half prefix* remove
that signature: all route IDs match decode, no post-block relative RMS exceeds
1e-3, and output/state deltas fall to the table's near-bitwise values. The
same-prompt F32 suffix takes 206.8 ms versus 140.4 ms for the Half suffix
(+47%); sequential decode takes 174.3 ms. An earlier broad F32 qualification
measured +13-16% fresh-prefill wall, while dense-only F32 measured +4.5%; the
8-row measurement shows the local price for the fix directly.

**Inference:** code's sparse-attention activations make the layer-7 router
decision consequential, while README's route flips are output-inert. The
probe locates the boundary but does not capture router score margins or split
layer 7 into MLA query, indexer selection, selected attention, and router
logits, so it does not name the exact subkernel that first nudges the route.

## Recommendation

Do not accept the 8-row Fast/Half code result as tier noise. For an append
that must follow the qualified sequential trajectory, use the existing
one-token decode path: 174.3 ms versus 140.4 ms for 8-row Half (+24%). If a
packed 8-row append is required, the measured specific fix is F32 packed
operands for that suffix: it removes the route changes and reduces the
distribution/state deltas to near-bitwise, at +47% wall for this width. No
production choice changed in this audit.

Raw M5 JSONL and lease/error records:
`metagross:~/qwen-runs/2026-11-11-glm-numerics/`
(`layer-audit.jsonl`, `route-audit-2.jsonl`, `f32-layer-audit.jsonl`,
`f32-suffix-audit.jsonl`; the first `route-audit.jsonl` attempt is retained
with its active-row capture assertion failure). Pressure was 0 on the code
measurements and 1 on prose completion; a pressure-2 interval gated launches
until pressure returned to 1.
