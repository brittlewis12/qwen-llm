# T9: LDLQ + two-sided RHT pilot — the 3.06-bpw tier's repositioning lever

Status: PREREGISTERED (frozen before implementation/runs). Date: 2026-07-20.
Worktree: ~/code/qwen-llm-trellis @ trellis/t7-real-weight-oracle.
Design jam + recon: cx 019f8086-bd13-7550-b746-8ba31f5365ac (QTIP/QuIP#/
EXL3/GPTQ source-verified; all BLOCKER/HIGH findings adopted below).
Prior packets: T7a (V1 0.140 plain rel-F on 9B classes; Q3_K 0.152 @3.44
bpw; Q4_K 0.0717-0.0721 @4.5 bpw), T8b tier card (LDLQ/RHT named as the
repositioning lever).

## Question

How much of the V1 -> Q4_K quality gap does the calibrated stack
(two-sided RHT + BlockLDLQ) close at constant 3.06 bpw ON OUR ENCODER
(T=256 group-ring spans, T_x=1/T_y=256 geometry — NOT QTIP's 16x16)?

## Frozen algorithm specifications (from recon, deviations declared)

- BlockLDLQ, source-faithful to QTIP Alg. 4-5 mapped to our geometry:
  256-block LDL of the damped rotated Hessian (H_lambda = H_tilde +
  0.01*mean(diag)*I; block-Cholesky then diagonal-block normalization to
  unit-block-lower L; NO H^-1 path); spans visited in REVERSE k order;
  adjusted target Z_j = W_j + E_later A_later,j (A = L - I); the EXISTING
  Viterbi oracle runs on Z_j; per-span sub-scales fitted to Z_j (not raw
  W). Natural ordering (no act_order) — matches QTIP.
  DECLARED RISK: our g=256 feedback is 16x coarser than QTIP's g=16;
  gains budgeted accordingly (engineering prior, not published: A3 plain
  10-20% better, r_H 25-40% better).
- RHT arm spec (frozen): EXL3-style two-sided block-diagonal H_128
  (Sylvester) with independent random sign vectors per side, NO
  channel-RMS scaling (pure orthogonal — EXL3's regularize includes a
  non-orthogonal scaling we deliberately EXCLUDE). All our dims are
  divisible by 128. DECLARED DIVERGENCE from the cx recommendation of
  QTIP composite-Hadamard for the pure science question: we choose the
  transform a shipping kernel would actually use; CONTINGENCY (only if
  A1 underdelivers the 5-15% prior): one QTIP-composite arm
  (H_1024/H_128 (x) H_28 class) before concluding on RHT.
  Orthogonal-invariance scoring: plain Frobenius scored in the rotated
  domain (exactly invariant, both-sided); r_H scored with rotated
  held-out activations x_tilde = V x.
- Seeds: 3 fixed paired sign-seed sets at Stage A (median across seeds
  before class aggregation); primary + 1 confirmatory at Stage B.
- Hessian: H = sum x x^T accumulated in f64 (chunked); symmetrize;
  damping c=0.01 primary, c=0.025 as the SINGLE predeclared rescue
  (never selected on test data); record Cholesky failures/condition.
- Headline metric (frozen): root relative output error on HELD-OUT
  activations, streaming form r_H = ||E X_test||_F / ||W X_test||_F.
  Guardrail: plain rel-Frobenius P (continuity with T7a and the Q4_K
  anchor claim). Units: ROOT errors everywhere, never squared proxies.
- Calibration corpus: wikitext-2-raw (single small download, SHA
  recorded in artifacts), split by DOCUMENT into disjoint train/test
  manifests, frozen before runs. DECLARED LIMITATION: narrower domain
  than C4/RedPajama practice.
- Calibration sizes (cx BLOCKER adopted): Stage A (0.8B) 32K train /
  32K test tokens; Stage B (9B) 128K train / 64K test.
- Activation source: Stage A captures from the 0.8B F32 GGUF on the
  Metal engine (full-precision activations — no quantized-proxy issue).
  Stage B plan: convert 9B BF16 -> F32 GGUF and capture on-engine; if
  infeasible, Q8_0 capture REQUIRES the paired BF16 audit (16K tokens,
  blk.0+blk.3): A3 improvement delta under the two sources must agree
  within 2 percentage points or the proxy fails and Stage B blocks.

## Arms (Stage A, 0.8B F32, classes: blk.0 + blk.3 ffn gate/up/down = 6
tensors, 2 input spaces per block)

- A0: baseline V1, re-scored through THIS pipeline (T7a's 0.140 is the
  external 9B anchor, never the r_H denominator).
- A1: +RHT only.
- A2: +BlockLDLQ only (no rotation; H in original basis).
- A3: +RHT +BlockLDLQ.
All arms identical payload format (L=16 K=3, T=256 spans, sub-scales,
V1 canonical code), identical bpw accounting; encode wall time recorded
per arm. Instrumentation for R5 (feedback-vs-code mismatch): per
reverse-span-index traces of target RMS/kurtosis, sub-scale selection,
and per-span Viterbi distortion — mandatory in A2/A3 artifacts.

## Stage gates (frozen; cx-recommended numbers adopted verbatim)

Definitions: P = macro-mean_6 plain rel-F; Delta_H = 1 - macromean_6
r_H(arm)/macromean_6 r_H(A0); seed-median before class aggregation;
bootstrap over held-out DOCUMENTS (not tokens).

- STAGE A -> B: Delta_H(A3) >= 20% AND P(A3) no worse than P(A0) by
  > 2% AND r_H improves on >= 5/6 tensors. (Implementation/ranking
  screen — do NOT kill A3 at 0.8B for missing 9B-calibrated plain-error
  bars.)
- Stage B cells: A0, A3, and the better single-lever arm (by Stage A
  Delta_H), on the EXACT T7a 6 classes (9B blk.0/blk.3 ffn gate/up/down).
- REOPEN-TIER (Stage B): P(A3) <= 0.115 AND Delta_H(A3) >= 25% AND
  >= 5/6 classes improve in r_H AND no class regresses > 2% in r_H.
  (0.115 closes ~37% of the plain V1->Q4_K gap with real held-out
  output-error gain.)
- STRONG SUCCESS: P <= 0.110 AND Delta_H >= 30%.
- KILL: bootstrap upper-95% bound on Delta_H < 15% AND P > 0.126.
- INCONCLUSIVE: anything between — permits exactly ONE predeclared
  sensitivity run (damping 0.025 OR doubled calibration), then close.

## Execution plan

1. Capture plumbing: env-gated FFN-input capture (post-attn-norm h and
   SwiGLU intermediate) for a declared block set in the dense Metal
   decode path, drained per token to an f64 Gram accumulator (rayon);
   held-out activations streamed to disk chunks for r_H.
2. Offline math (trellis_offline or sibling module): H_128 sign/Hadamard,
   f64 blocked Cholesky + 256-block unit-lower normalization, BlockLDLQ
   driver wrapping the existing span encoder, streaming r_H scorer.
3. Smoke: tiny synthetic W/H where LDLQ has a known closed-form
   advantage over nearest-span quantization; Hadamard orthogonality
   test (H H^T = nI); LDL reconstruction test (L D L^T = H_lambda).
4. Stage A runs -> gate -> Stage B per gates above.

Wall-time budget honesty: Stage A capture ~64K tokens on 0.8B GPU
(minutes) + 24 encodes (6 tensors x 4 arms; 0.8B tensors are ~25x
smaller than 9B's ~190 s/class => minutes each) + LDL factorizations
(seconds at d<=3584). Stage B factorization at d=12288 in f64 (~1.2 GB
H) is the long pole — budget separately before starting it.

## Non-claims

No inference-speed claims (weight-error pilot; the two runtime
transforms' cost is a later kernel question). No PPL claims. No
end-to-end tier promotion — REOPEN-TIER reopens the roadmap
conversation with quality evidence, nothing ships from T9.

---

## EXECUTION STATUS (2026-07-20, mid-packet checkpoint; prereg above unmodified)

Built and committed (655ad16, e6ae2ca, 73f3ae5):
- Offline core (trellis_ldlq.rs, 4/4 tests green): two-sided block-H128
  RHT (roundtrip + Frobenius-invariance tested), f64 Gram accumulator
  (scoped-thread batched), dense f64 Cholesky + 256-block unit-lower A
  (reconstruction-tested; first cut solved the transposed system \u2014
  caught by test), generic BlockLDLQ driver with R5 moment traces,
  streaming r_H scorer. Mock-quantizer test isolates the feedback
  algebra (>=5% held-out r_H gain on lag-256-correlated H); trellis
  integration test >=2%. NEGATIVE quantified en route: adjacent-lag
  correlation is INVISIBLE to g=256 block feedback (the cx geometry
  concern made concrete \u2014 the original AR(1) test could not detect a
  working implementation).
- Capture: env-gated FFN-input hook (h + SwiGLU inner) + harness.
- Stage A capture COMPLETE: 32,768 train / 32,768 test tokens,
  64/66 disjoint documents, doc_max 512, 0.8B F32 on-engine (~120 t/s,
  517.8 s wall), spaces blk{0,3} x {h d=1024, inner d=3584}.
  Corpus: wikitext-2 WORD-LEVEL train.txt (raw-v1 zip link dead \u2014
  declared substitution), sha256
  9e9fa1ad55b1c2c95b08e37dd8e653f6$(unrecorded-tail) \u2014 full sha in
  target/t9 runner logs. Artifacts: target/t9/stage-a/ (Grams f64,
  test f32 chunks, manifest.json).

NEXT (fresh session): Stage A runner example \u2014 load the 6 tensor
classes from the 0.8B GGUF, run arms A0-A3 x 3 seeds per the frozen
prereg (RHT rotation of W + Hessian, damped LDL, BlockLDLQ encode,
plain + r_H scoring vs held-out chunks), apply the Stage A -> B gate.
All inputs exist; no open design questions.

---

# STAGE A RESULTS (2026-07-20; artifacts in ./stage-a/; runner
trellis3_t9_stage_a.rs; 3 seeds, seed-median then macro-mean over the 6
0.8B classes; declared amendments in the runner header: A0/A2 use the
T7a input-side incoherence recipe [the literal "A2 in original basis"
would have confounded LDLQ's increment with removing incoherence];
T7a-style row sampling restored after full-tensor encode measured
448 s/case; anchors additionally scored in r_H — additive measurement,
no arm or gate change)

| Arm | bpw | P (plain rel-F) | r_H (held-out) |
|---|---|---|---|
| Q4_K anchor | 4.5 | 0.0720 | 0.0665 |
| Q3_K anchor | 3.44 | 0.1522 | 0.1401 |
| A0 (T7a recipe) | 3.06 | 0.1404 | 0.1345 |
| A1 (+two-sided RHT) | 3.06 | 0.1402 | 0.1328 |
| A2 (+BlockLDLQ) | 3.06 | 0.1653 | 0.1130 |
| A3 (both) | 3.06 | 0.1654 | 0.1105 |

Seed spread on A3: negligible (per-class values differ in the third
decimal across seeds 11/22/33). Encode cost: LDLQ arms ~equal to plain
(22-41 s/case sampled; feedback overhead is not the bottleneck, the
span Viterbi is).

## Frozen-gate verdict: STAGE-A FAIL (as preregistered)

Delta_H(A3) = 17.9% (< 20%); P(A3) +17.8% vs A0 (gate allowed +2%);
r_H improved 6/6 (>= 5 required). The sensitivity run permitted by the
INCONCLUSIVE clause is NOT spent: damping 0.025 smooths feedback
(lowers Delta_H); doubled calibration is unlikely to bridge 2.1 points;
chasing the threshold with the one allowed knob would be gate-gaming.
Stage B is NOT authorized under this preregistration.

## Findings (the science, verdict-independent)

1. BlockLDLQ AT g=256 WORKS on real weights/Hessians: -16% to -19%
   held-out r_H (6/6 classes, 3 seeds), first-order consistent with the
   cx prior (15-30%) despite 16x-coarser feedback than QTIP.
2. The P-guardrail was miscalibrated at design time: LDLQ definitionally
   trades plain error for weighted error; measured trade at g=256 is
   +18% P for -18% r_H. A "no worse than +2% plain" screen requires the
   trade to be free — physics says it is not. Design lesson recorded.
3. Two-sided RHT increment ~ ZERO (A1: -1.3% r_H, -0.2% P) over the
   input-side-only incoherence already in the tier recipe; A3 ~= A2.
   The "RHT" half of the roadmap's repositioning lever is EXHAUSTED;
   LDLQ carries everything.
4. Anchor-relative repositioning (the extension's point): on r_H the
   tier's Q3_K edge grows from 4.0% (A0) to 21.1% (A3) at 11% fewer
   bits — but on plain P, A3 falls BELOW the Q3_K line (0.165 vs
   0.152). The tier's claim is now METRIC-DEPENDENT, and the two
   metrics disagree about sign vs the nearest anchor.
5. R5 instrumentation: adjusted targets remain near-Gaussian
   (kurtosis excess -0.06..0, rms flat across reverse-span index) —
   no code/source mismatch from feedback; the trellis code's Gaussian
   tuning stays valid under LDLQ.
6. A0's P=0.1404 on 0.8B replicates T7a's 9B 0.140 almost exactly —
   the trellis+incoherence error rate is dimension-stable, supporting
   0.8B->9B transfer of arm RANKINGS (not absolute bars).

## Disposition

CLOSED at STAGE-A FAIL / proxy-evidence-banked. The decisive next
question is not more proxy refinement — it is which metric predicts
END QUALITY at this operating point. r_H is the literature's validated
proxy (what GPTQ/QTIP/EXL3 optimize and what tracks PPL); plain
Frobenius is the uncalibrated fallback; our r_H is wikitext-domain.
A successor packet (T9b) should preregister the END-TO-END arbiter:
quantize a full small model both ways (A0 vs A3 recipe) and measure
PPL-class quality on held-out text vs Q3_K/Q4_K at matched bytes,
with gates written against the anchors rather than a plain-error
guardrail that LDLQ cannot satisfy by construction. No tier
reopening claim is made from Stage A proxies.
