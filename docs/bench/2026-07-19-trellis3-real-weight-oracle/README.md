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

## Results — 2026-07-19

Part 1 (V2G rested A/B/A, ran pre-worktree-migration): ratios
1.063 / 1.039 / 0.985 across three invocations; median >= 1.00, spread
0.078 > 0.05 -> cell MEASUREMENT-UNSTABLE per the frozen rule. A live
co-tenant agent session in the main checkout was subsequently
identified as the probable confound; all same-invocation pairs remain
internally valid. No promotion-grade V2G claim; parity-band evidence
retained.

Part 2 (real-weight fidelity, 9B BF16 source, niced 8 threads,
~190 s/class): class selection note — the 6 largest qualifying classes
were ALL FFN tensors (gate/up/down x blk.0/blk.3); attention/GDN
classes are unexamined in this packet. Implementation note: prereg
said "per-row sign"; implemented as per-COLUMN signs (EXL3 su-style),
corrected before running.

| method | bpw | rel-Frobenius (range over 6 classes) |
| --- | ---: | --- |
| Q4_K (anchor, no gate) | 4.5 | 0.0717-0.0721 |
| trellis V1 mask/or | 3.0625 | 0.1402-0.1405 |
| Q3_K | 3.4375 | 0.1515-0.1520 |
| trellis V2 split | 3.0625 | 0.1557-0.1566 |
| LM8+RHT (TQ-class) | 3.5 | 0.1700-0.1711 |
| IQ3_XXS (no imatrix) | 3.0625 | 0.2132-0.2143 |

Gate outcome: V2 <= Q3_K on 0/6 -> preregistered KILL fires for the
V2 code at the 3.06-bpw tier. V2 <= IQ3_XXS on 6/6 (by ~27%).

### Findings

1. The V=2 split code's tax is EXACTLY as the synthetic oracle
   predicted: real-weight V2/V1 rel-err ratio 1.109-1.117 (mean 1.112)
   vs T5's Gaussian prediction 1.113. The T5 oracle is hereby
   validated as a quantitative predictor for code-design iteration —
   future decode-code searches can run CPU-only with confidence.
2. V1 (canonical-class code) WINS on real weights: beats Q3_K on 6/6
   by ~7.5% rel-err at 11% fewer bits, and byte-matched IQ3_XXS by
   ~34%. Paired with T6's measured V1G kernel band (0.82-0.84 of Q4_K
   BW -> 1.20-1.23x time-speedup on quantized streams), this is the
   surviving product point of the 3-bpw tier.
3. This pipeline is the documented LOWER BOUND for trellis quality:
   no Hessian/LDLQ, input-side-only rotation, one fp16 scale per 256.
   The Q4_K anchor (2x better rel-err at 1.47x bytes) shows how much
   the K-quant sub-block affine structure buys on real weights —
   LM8+RHT at 3.5 bpw losing to unrotated Q3_K at 3.44 bpw isolates
   the same lesson. LDLQ + two-sided RHT + finer scale structure are
   the named upgrades that carry published EXL3 to dPPL +0.015 at
   4.15 bpw on this exact model family.

### Verdict

- V2@3.06 as a "beat Q3_K" tier: KILLED (preregistered gate). Remains
  the fastest measured decode (parity-band with Q4_K BW) and better
  than byte-matched IQ3_XXS; usable only where that tradeoff is named.
- V1@3.06: quality-passes the same test 6/6 (recorded; V1 was gated
  out of prereg by its T6 kernel kill — that kill's reopen condition,
  >= 2 ops/weight removed, now carries the whole tier's upside).
- The decisive next experiments, in order of information value:
  (a) V=2 code-design search ON THE VALIDATED SYNTHETIC ORACLE for a
  code with <= 1.03x V1 error at V2 kernel cost (the tax is the only
  thing between the fast kernel and the quality pass);
  (b) LDLQ/Hessian phase (the EXL3 recipe) — lifts every trellis row;
  (c) attention/GDN class coverage;
  (d) K=4 tier probe vs Q4_K.
