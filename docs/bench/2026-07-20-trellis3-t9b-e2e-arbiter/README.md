# T9b: end-to-end arbiter — does the LDLQ trade buy real quality?

Status: PREREGISTERED (frozen before any PPL number exists; the Stage A
proxy table is known and is exactly why this packet exists).
Date: 2026-07-20. Worktree: ~/code/qwen-llm-trellis @
trellis/t7-real-weight-oracle. Predecessor: T9 Stage A
(../2026-07-20-trellis3-t9-ldlq-pilot/, closed FAIL-as-frozen with the
finding that plain-P and r_H DISAGREE about the tier's standing vs
Q3_K: A3 r_H -21% vs Q3_K, A3 plain +9% vs Q3_K).

## Question

At matched coverage and bytes, which proxy predicted end quality?
Quantize the SAME model the four ways and measure held-out
teacher-forced perplexity damage vs the F32 baseline.

## Method (frozen)

- Model: Qwen3.5-0.8B F32 (the engine's bit-tight oracle target).
- Coverage: all 72 FFN tensors (blk.0..23 x gate/up/down) — the class
  family Stage A studied and T7a validated; attention/GDN projections
  stay F32 in ALL variants (never trellis-validated; identical across
  variants so they cancel in damage comparisons).
- Variants (each = byte-identical F32 GGUF copy with the 72 tensor
  data ranges overwritten in place by quantize->dequantize output):
  V-A0: trellis V1, T7a input-side incoherence recipe, plain per-span
        encode (n_sub=1; 3.06 bpw effective for the quantized class).
  V-A3: trellis V1, same incoherence + BlockLDLQ with per-block
        per-space Hessians (damping 0.01) from 32K-token document-
        disjoint wikitext TRAIN calibration (24-block capture).
  V-Q3K / V-Q4K: ggml_quantize_chunk Q3_K / Q4_K (no imatrix),
        dequantized via ggml to_float (T7a anchor methodology).
  V-F32: untouched copy (baseline; also validates the surgery path:
        its PPL must equal the original file's bit-for-bit).
- Eval: wikitext-2 TEST split (pytorch/examples mirror, sha
  d790b833ef8cf03a90db7bf1271b7520b83c45ce, DISJOINT from the
  calibration train file), tokenized once; the FIRST 64 segments of
  512 tokens (32,768 predicted tokens) frozen as the eval slice;
  teacher-forced GPU single_token loop, fresh session per segment;
  nll = -ln softmax(logits)[next] accumulated in f64 over positions
  0..510 predicting 1..511 per segment. PPL = exp(mean nll).
  Identical segmentation/tokens for every variant; per-token nll
  saved per variant for PAIRED analysis.
- Damage D(X) = mean_nll(X) - mean_nll(V-F32). Paired per-segment
  bootstrap (10,000 resamples over the 64 segments) for CIs on damage
  differences.

## Frozen gates

- G1 (proxy arbitration, the packet's reason): if D(V-A3) <
  D(V-A0) with the paired-bootstrap 95% CI of the difference excluding
  zero => r_H VALIDATED as the tier's quality metric; if D(V-A3) >
  D(V-A0) with CI excluding zero => r_H proxy FALSIFIED end-to-end at
  this operating point (plain-P was right; LDLQ lever closed). CI
  straddling zero => proxies TIED (report; no metric claim).
- G2 (tier positioning): REOPEN-TIER requires D(V-A3) <= 0.75 x
  D(V-Q3K) with paired CI excluding the 0.75 boundary... precisely:
  the bootstrap upper-95% of D(A3)/D(Q3K) <= 0.85, point <= 0.75 —
  a real quality edge over the nearest-byte anchor at 11% fewer bits
  (mirrors Stage A's r_H prediction of ~0.79 for this ratio; plain-P
  predicts ~1.09 — the gates discriminate the proxies).
- G3 (tier kill): D(V-A3) >= D(V-Q3K) (point) AND D(V-A0) >=
  D(V-Q3K) => no quality story at 3.06 bpw on end metrics; tier
  stays closed regardless of proxy politics.
- Between G2 and G3: INCONCLUSIVE-BAND; report, no reopening, one
  permitted extension = second eval slice (segments 65..128) to
  tighten CIs, preregistered here, used only if boundaries straddle.
- Sanity preconditions (must hold or the run is INVALID, not
  interpreted): V-F32 PPL == original-file PPL bitwise; D(V-Q4K) <
  D(V-Q3K); all D > 0.
- DOMAIN LIMITATION (declared): calibration and eval share the
  wikitext domain; a REOPEN verdict is calibrated-domain evidence and
  says so on its face.

## Cost budget (honesty)

A3 encode = 72 tensors x 14336 spans ~ 1.03M span-Viterbis (~50 min at
12 threads) + 24x2 LDL factorizations (~4 min); A0 same encode cost
without LDL; anchors seconds; PPL 5 variants x 32,768 tokens at
~100 t/s ~ 28 min each (~2.3 h total, sequential, exclusive GPU).

## Non-claims

Wikitext-domain PPL only; no speed claims; no 27B/9B extrapolation
beyond the dimension-stability observation from Stage A; REOPEN-TIER
reopens the roadmap conversation — nothing ships from T9b.

---

## EXECUTION STATUS (2026-07-20 ~15:30, prereg above unmodified)

- 24-block Hessian capture complete (581.9s, 32K doc-disjoint train
  tokens, target/t9/hessians-24/).
- Evaluator smoke: original F32 -> ppl 19.4339 over 4 segments
  (sane; word-level wikitext).
- Pipeline launched (target/t9/run-t9b.sh, log
  target/t9/t9b-pipeline.log): f32 copy + hash check, q3k/q4k patches
  (~8s each) done; a0/a3 full-tensor encodes measured at ~283 s/tensor
  x 72 => ~5.6 h/arm (the prereg's ~50-min budget repeated the Stage A
  sampled-span arithmetic error against full tensors — cost note
  corrected here, before results). PPL x5 (~7 min each) runs
  automatically after. ETA complete ~03:00-04:00 local.
- Analyzer (analyze.py, frozen in this dir) self-tested on degenerate
  identical inputs: zero damages, TIED, sanity flags correctly raised.
- CLOSE PROTOCOL when the log shows "T9B PIPELINE COMPLETE":
  (1) verify the MATCH line (V-F32 bitwise == original);
  (2) python3 analyze.py target/t9/ppl;
  (3) check sanity preconditions (all D>0, D(q4k)<D(q3k));
  (4) record G1/G2/G3 verdicts verbatim + the five PPLs in RESULTS;
  (5) copy nll dumps + pipeline log into this dir; commit close.

---

# RESULTS (2026-07-21 03:10 pipeline completion; analyzer output verbatim
in ./analysis-output.txt; nll dumps + pipeline log in this dir; prereg
text above unmodified)

| Variant | mean_nll | PPL | Damage (nats) |
|---|---|---|---|
| V-F32 (baseline; hash-verified bitwise == original) | 2.984535 | 19.7773 | — |
| V-Q4K (4.5 bpw) | 3.029159 | 20.6798 | +0.044625 |
| V-Q3K (3.44 bpw) | 3.081613 | 21.7935 | +0.097078 |
| V-A0 (trellis 3.06 bpw) | 3.132760 | 22.9372 | +0.148225 |
| V-A3 (trellis+LDLQ 3.06 bpw) | 3.082884 | 21.8212 | +0.098349 |

Sanity preconditions: PASS (V-F32 bitwise MATCH; all D > 0;
D(q4k) < D(q3k)).

## Frozen-gate verdicts (analyzer verbatim)

- G1: D(a3)-D(a0) median -0.049893, 95% CI [-0.058690, -0.040480] —
  **r_H VALIDATED** (a3 strictly better end-to-end; LDLQ cut the
  trellis damage 34%).
- G2: D(a3)/D(q3k) median 1.0122, 95% CI [0.9033, 1.1429] — REOPEN
  (<= 0.75 / upper <= 0.85) decisively NOT met.
- G3: D(a3) >= D(q3k) on point AND D(a0) >= D(q3k) — **TIER STAYS
  CLOSED (no quality story at 3.06 bpw)**.
- Context: D(a0)/D(q3k) = 1.5271 [1.3828, 1.7033].

## Findings

1. LDLQ's held-out r_H gain is REAL end quality: -34% PPL damage
   (0.148 -> 0.098 nats), CI excluding zero by a wide margin. The
   calibrated phase works; within-format, r_H ranked the intervention
   correctly.
2. CROSS-FORMAT, both weight-space proxies misled: r_H predicted
   a3/q3k ~= 0.79, plain-P predicted ~= 1.09; reality 1.01. T7a's
   "V1 beats Q3_K by 9% plain at 11% fewer bits" did NOT survive
   end-to-end (a0 = 1.53x Q3_K's damage). METHOD RULE going forward:
   weight-space proxies (either kind) are valid for ranking
   interventions WITHIN a format; cross-format claims require the
   end-to-end arbiter.
3. The tier's final honest card: trellis+LDLQ at 3.06 bpw ==
   statistical quality PARITY with Q3_K (ratio CI [0.90, 1.14]) at 11%
   fewer bits, with kernel economics previously measured below the
   ship bar (T8b) and a calibration pipeline as an operational cost.
   Parity-at-minus-11%-bytes with no speed story does not clear any
   preregistered bar. The 3.06-bpw trellis tier is CLOSED on
   end-to-end evidence.

## Reopen conditions (program-final)

- A materially better code/bpw point (e.g. K=4 tier vs Q4_K — T7a's
  parked probe (d)) evaluated END-TO-END from the start, or
- an application where -11% weights bytes at quality parity carries
  product value that the roadmap prices above the format's
  operational cost, or
- LDLQ-class calibration applied to a format that already has a
  positive speed story (the machinery in trellis_ldlq.rs is
  format-agnostic on the feedback side).

Domain note (declared in prereg): all quality numbers are
wikitext-domain PPL on 0.8B; the closure is at the operating point
tested, extrapolated no further.
