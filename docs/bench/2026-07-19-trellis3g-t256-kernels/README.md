# T6: trellis3 T=256 group-ring kernels — PREREGISTERED

Phase-2 continuation. T5 killed T=32 span windows on quality and set
the design point at T=256 group-ring (canonical-class quality at V=1,
0.0194 MSE; V=2 speed-fallback at +0.16 effective bits). Decode is
position-random-access, so the T1 kernel structure survives: a lane
covering weights [32t, 32t+32) of a 256-weight group reads FOUR words
(one overlap word from the neighboring span; ring wrap for lane 0)
instead of three. Buffer layout is byte-identical to T1 (24 uint32 +
one fp16 scale per group; 98 B / 256 weights; 3.0625 bpw).

Objects: `kernel_mat_vec_trellis3g_3inst_f32` (V=1 quality point) and
`kernel_mat_vec_trellis3g_3inst_v2_f32` (V=2 speed point) in
kernels/trellis_gemv_floor.metal; matching CPU-reference arms;
correctness rows in `mat_vec_trellis3_matches_cpu`; bench rows in the
`trellis3 mat_vec` group.

Window math (group ring R=768, lane it in [0,8), local frame =
words {(3it+23) mod 24, 3it, 3it+1, 3it+2} as 128 local bits):
V=1 weight l: window starts at local bit p = 3l + 19, l in [0,32).
V=2 pair t: p = 6t + 22, t in [0,16). All p compile-time under full
unroll; crossings stay inside the 4-word frame.

## Protocol

Same harness discipline as T1 (persistent buffers, single + chained64,
one invocation, AC, thermal capture, no parallel timed work). Cells:

1. PRIMARY (gated): chained64 ffn_gate_27b (5120x17408), same-session
   Q4_K comparator.
2. Big-pair (grades T1's cross-session 1.130x/1.660x claim): q4_k
   embed (715 MB) and trellis3g embed (476 MB) in the SAME invocation
   (sequential same-process pair; strict interleaving is not available
   in this harness — labeled accordingly).
3. Skinny/occupancy sweep (diagnostic, ungated): n_out in
   {1024, 2048} at n_in=5120 for the G variants, to bound the
   attn_v-class exposure the T4 review flagged.

Tuning budget: T1's unused budget applies — <= 3 tuning iterations per
variant after first measurement (NSG/occupancy, unroll, load
scheduling). Every iteration recorded.

## Gates (frozen)

- V1G HOLD: >= 0.85x same-session Q4_K GB/s on the primary cell
  (keeps the 1.25x+ time-speedup floor at canonical-class quality).
- V1G STRETCH: >= 0.95x (raises the quality point to ~1.4x+).
- V2G EXPECTATION: >= 1.00x (should track T1's 1.075 within the
  overlap-word cost; a large drop indicates the frame/overlap design
  is wrong, iterate within budget).
- Correctness gate green for both G variants (same thresholds as T1).
- KILL any variant that cannot hold its line after budget; the T5
  quality result stands regardless (encoder-side, kernel-independent).

## Results — V2G holds (drift caveat), V1G killed at this geometry — 2026-07-19

Correctness: both G variants + both tuning variants match the CPU
reference (group-ring windows, lane-0 wrap exercised), max|Δ| <= 1.9e-6,
cosine 1.0. One harness bug found en route: the NSG4 variant initially
dispatched at 64 threads (encode-side nsg not plumbed), producing
zero rows for sgitg 2-3 — caught by the correctness gate, fixed,
retested green.

Timed sessions (chained64, ffn_gate_27b primary cell; each row's ratio
uses its own same-invocation Q4_K comparator):

| session | Q4_K GiB/s | variant | GiB/s | ratio |
| --- | ---: | --- | ---: | ---: |
| S1 | 423.0 | 3inst_g256 (V1G, NSG2) | 347.6 | 0.822 |
| S1 | 423.0 | 3inst_v2_g256 (V2G) | 430.0 | **1.016** |
| S2 | 449.0 | 3inst_g256 (V1G rerun) | 382.2 | 0.851 |
| S3 | 438.9 | V1G nsg4 (+attr) [iter 1] | 333.8 | 0.761 |
| S3 | 438.9 | V1G nsg2 rerun | 367.3 | 0.837 |
| S4 | 458.1 | V1G nsg4 (no attr) [iter 2] | 374.4 | 0.817 |
| S5 | 461.0 | V1G nsg2 rerun | 374.1 | 0.811 |
| S5 | 461.0 | V2G rerun | 416.3 | 0.903 |
| S5 | 461.0 | V1G nr4 [iter 3] | 350.6 | 0.761 |

Diagnostics (S1): skinny cells confirm the occupancy cliff — 1024 rows
0.40-0.45x, 2048 rows 0.49-0.64x; attn_v-class tensors must stay
K-quant (they are <1% of 27B weight bytes, so exposure is small).
Embed big-row A/B/A grading: INCONCLUSIVE — q4_k embed itself moved
-8% between sessions and trellis embed rows inverted (V2G < V1G);
recorded, not claimed.

MEASUREMENT DRIFT NOTE: across back-to-back sessions the Q4_K
comparator climbed 423 -> 461 GiB/s (+9%, on a bandwidth-bound
kernel) while trellis absolutes moved the opposite direction
(V2G 430 -> 416). The two valid same-session V2G pairs therefore
disagree (1.016 vs 0.903). This is not a stable-box signature; no
thermal/perf warnings were recorded. Promotion-grade V2G numbers
require a rested-box rerun with interleaved pairs. (This session is
itself evidence for the measurement-control-plane swing.)

### Verdict

- V2G (T=256 speed point, quality 0.0245): meets its >= 1.00
  expectation on the S1 clean pair (1.016); S5 drift pair at 0.903
  flagged. HOLDS with a rested-box confirmation required before any
  promotion-grade claim. Combined with T5: 1.35-1.49x time-speedup on
  quantized streams at a priced ~0.16-bit quality tax vs canonical.
- V1G (T=256 quality point, 0.0194): 0.81-0.85 across five pairs;
  three tuning iterations spent (NSG4 -9%, attribute removal +12%
  local but ratio unchanged, NR4 -7%). KILLED at the 0.85 hold gate
  under the current geometry/code. Reopen condition: a mechanism
  removing >= 2 ops/weight from the V1 decode chain (cheaper
  imul-free near-Gaussian code, fused extract+hash, or a
  quality-neutral V=2-style pairing) — not another NSG/NR reshuffle.
- The T5 quality ladder is kernel-independent and stands: if T7
  real-weight evals price V2G's 0.16-bit tax as unacceptable, the
  fallback is V1-shaped decode at ~0.82-0.85 of Q4_K bandwidth
  (still 1.20-1.25x time-speedup at equal shape), not abandonment.

Next: T7 real-weight PTQ (Hadamard + Viterbi encoder on real Qwen
tensors; KL/PPL ladder vs Q4_K/Q3_K/IQ3_XXS/TQ3_1S with imatrix where
applicable; PonyExl3 4.15 bpw dPPL +0.015 anchor), with the V2G
rested-box confirmation folded into its first timed session.
