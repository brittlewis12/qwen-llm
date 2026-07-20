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
