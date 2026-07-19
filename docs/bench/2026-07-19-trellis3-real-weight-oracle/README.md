# T7a: trellis3 real-weight fidelity oracle + V2G rested confirmation — PREREGISTERED

Phase 2b of the trellis swing. T5 priced the codes on a synthetic
Gaussian source; this packet prices them on REAL Qwen weights (the
incoherence-processing assumption meets reality), and grades T6's
drift-flagged V2G throughput claim on a rested box.

Identity: zekrom M4 Max, macOS 15.6.1, after t6 close. Source model:
`~/models/Qwen3.5-9B-BF16.gguf` (true-precision, production-scale dense
hybrid; same architecture family as the 27B anchor). Cross-check:
`Qwen3.5-0.8B.F32.gguf` if time permits.

## Part 1 — V2G rested-box A/B/A confirmation (GPU, run FIRST)

Three sequential `cargo bench` invocations of the pair
{q4_k chained64 ffn_gate, trellis3 3inst_v2_g256 chained64 ffn_gate}
on an otherwise idle box (>=30 min since last GPU bench). Metric:
per-invocation BW ratio; report median and spread. CONFIRM if median
>= 1.00 with spread contained (max-min <= 0.05); DEMOTE the T6 claim
to "parity band 0.95-1.00" if 0.95 <= median < 1.00; flag the whole
cell measurement-unstable if spread > 0.05 (feeding the control-plane
swing, not a kernel verdict).

## Part 2 — real-weight fidelity oracle (CPU-only)

Instrument: `crates/qwen-llm/src/trellis_offline.rs` (tooling-only
module, codec.rs-style banner; self-contained constants with a bridge
test asserting bit-identity against metal.rs's CPU reference on a
synthetic buffer) + example `trellis3_real_weight_oracle.rs`.

Method per tensor class (up to 6 largest 2D weight classes drawn from
one GDN block and one full-attn block, shape[0] % 256 == 0):

- Sample up to 64 seeded rows (~65K weights minimum per class).
- Trellis rows: input-side incoherence = seeded random per-row sign +
  H128 blockwise FWHT along n_in (1/sqrt(128)); per-256-group fp16
  scale (encode -> LS refit -> re-encode); exact Viterbi (V=1 and
  V=2 split codes, T=256 group-ring, two-pass tail-biting);
  reconstruction inverse-rotated back to original space.
  SIMPLIFICATION STATED UP FRONT: input-side rotation + row signs
  only (production would add output-side H128; two-sided RHT and
  LDLQ/Hessian weighting both IMPROVE trellis, so this measures a
  LOWER bound on trellis quality). No activation/imatrix information
  is used by ANY method in this packet.
- Scalar comparator: same rotation + Lloyd-Max-8 per-32 fp16 scale
  (TQ3_1S-class, 3.5 bpw).
- ggml comparators on the ORIGINAL (unrotated) rows via
  ggml_quantize_chunk, no imatrix: Q4_K (4.5 bpw anchor, no gate),
  Q3_K (3.4375), IQ3_XXS (3.0625).
- Metric: relative Frobenius error ||W - What||_F / ||W||_F per class,
  in ORIGINAL weight space.

## Gates (frozen)

- PASS (3.06-bpw tier lives on real weights): trellis-V2 rel-err <=
  Q3_K on >= 5 of 6 classes AND <= IQ3_XXS on >= 5 of 6.
- KILL the 3.06 tier: trellis-V2 > Q3_K on >= 3 classes.
- Record (ungated): real-weight V2/V1 error ratio (checks T5's
  0.16-bit synthetic tax); Q4_K anchor column; per-class bpw.
- Non-goals: end-to-end PPL/KL, imatrix-armed comparators, LDLQ,
  output-side rotation, MoE expert tensors — all named phase-2c
  upgrades, each expected to favor or refine trellis, none blocking
  this verdict.

## Results

(appended after the run)
