# T8a: V=2 / cheap-V1 code-design screen on the validated oracle — PREREGISTERED

Thread B1 from the post-T7a jam (cx session 019f7ce2). CPU-only; no
GPU, niced, co-tenant safe. The T5/T7a-validated synthetic oracle
(prediction error ~0.1% on the V2/V1 ratio) screens candidate decode
codes; only oracle winners proceed to real-weight confirmation and
Apple-instruction microbenches (T8b).

Question: does a state->pair map exist at ~V=2 kernel cost (one 16-bit
window extract + one imad + <= ~4 cheap ALU ops / register shuffles,
NO per-pair memory loads) with rel-err <= 1.03x the V=1 canonical
code? Secondary: does an imul-free V=1 code exist at <= 1.03x V1?
Tertiary (B8): do per-128/per-64 sub-scales (+0.0625/+0.1875 bpw)
close V2-split's 2.8% real-weight gap to Q3_K on the synthetic proxy?

## Rows

Controls (not kernel-realizable; deconfound topology vs mapping):
- RPTC-V1, RPTC-V2: independent seeded Gaussian value(s) per state —
  the random-codebook bound at this L/T. Per cx: if RPTC-V2 misses
  1.03x V1, the deficit is topological and the cheap-code search stops.
- HYB-512: 512 k-means 2D-Gaussian centroids, two hash sign flips —
  the tuned-small-alphabet bound (device-LUT kernels already lost;
  quality reference only).

Kernel-budget candidates (all: h = st*A+B unless noted):
- C1 cart32: w0=Q32[(h>>S0)&31], w1=Q32[(h>>S1)&31]; Q32 = signed
  Gaussian quantiles (register-shuffle realizable; 2 extracts + 2
  shuffles per pair). Sweep (S0,S1) in {(0,5),(3,9),(5,10),(2,11)}.
- C2 joint32: half2 C32[(h>>S)&31] (k-means positive-quadrant pair
  centroids), signs from two hash bits (1 packed shuffle + and/xor).
  S in {3,6,9}.
- C3 = C2 + hash-bit coordinate swap (alphabet 256).
- C4 sum/product: z = maskor(h); w0=z.x+z.y, w1=C*z.x*z.y;
  C in {0.5, 1.0, 2.0}.
- C5 dual-3INST (quality ceiling, over budget): g=h^rot(h,R); w0/w1 =
  half-sums of maskor(h)/maskor(g); R in {7,13}.
- V1-RLUT32: w=Q32[(st>>S)&31], NO imad; S in {3,7,11}; plus
  xor-folded index q=st^(st>>5)^(st>>10).
- V1-ARX: packed u16 add/rotate construction (imul-free), half-sum.

B8 sub-scale rows: V2-split and V1-maskor at per-128 and per-64 fp16
sub-scales (bpw 3.125 / 3.25).

## Protocol

Screen at 64 groups (16,384 weights, seed A) — SE ~1.1%. Every row
within 1.10x of V1 rescreens at 256 groups x 2 seeds (SE ~0.4%) before
any promotion claim (cx structural note: 3%-scale gates need more
sample than T5's default). T=256 group-ring, two-pass tail-biting,
per-256 fp16 scale with one LS refit (identical discipline to T5/T7a)
except the B8 rows which fit sub-scales.

## Gates (frozen)

- PROMOTE-V2: kernel-budget candidate <= 1.03x V1 rel-err at the
  confirm sample -> T8b (real-weight row + shuffle microbench).
- PROMOTE-V1-CHEAP: imul-free V1 row <= 1.03x V1 -> T8b (kernel
  variant targeting >= 0.95x Q4_K BW).
- B8-FLIP: V2-split @ per-128 scales <= 0.92x its per-256 rel-err on
  the synthetic proxy (the margin that would flip T7a's 2.8% gap,
  with headroom) -> rerun T7a real-weight row at per-128.
- STOP: if RPTC-V2 > 1.03x V1, close the V2 search as topological;
  record and fall back to the V1/B8 branches.

## Results — 2026-07-19

Screen (64 groups, seed A) + confirm (256 groups x seeds A/B) both ran;
confirm values quoted. V1 baseline rel-err 0.14093.

Controls:
| row | vs V1 | reading |
| --- | ---: | --- |
| RPTC-V2 (random Gaussian pairs) | 0.969x | topology bound: V=2 can BEAT V1; the tax was 100% the map |
| RPTC-V1 | 0.971x | even V1's computed code leaves ~3% vs random labels |
| HYB-512 2-sign | 0.993x | tuned 512-alphabet nearly closes; alphabet size is binding |
| C5 dual-3INST R7/R13 | 0.977x | a COMPUTED code fully closes the gap |

Kernel-budget candidates (screen): all FAIL the 1.03x gate — best
in-budget C2 joint32 S9 at 1.146x (worse than the V2-split baseline
1.122x); C1 cart32 1.33-1.96x (32-entry alphabets too small; LCG
low-bit slices worst); C4 sum/product 2.4-3.4x; imul-free V1 rows
1.30-2.14x (best: ARX R5 1.295x). B8 sub-scales: V2 /64 = 1.074x at
3.25 bpw (ratio vs per-256 V2 = 0.971 > 0.92 -> B8-FLIP does not
fire); V1 /64 = 0.977x at 3.25 bpw.

Frozen-gate outcomes: PROMOTE-V2 none; PROMOTE-V1-CHEAP none;
STOP did not fire (RPTC-V2 passed); B8-FLIP did not fire.

### The finding the frozen gates did not anticipate

C5 dual-3INST (h = st*A+B; g = h ^ rotl(h,13); each masked-or word's
half-sum is one weight) was preregistered as an over-budget quality
ceiling — but its cost is ~6.5 ops-eq/weight, CHEAPER than V1's ~8,
with 0.977x V1 quality. It dominates the V1 fallback on both axes.
Real-weight confirmation (T7 harness, same 6 classes): rel-Frobenius
0.13785-0.13810 vs V1 0.1402-0.1405 (0.983x; synthetic predicted
0.977x — third consecutive validated oracle transfer) and Q3_K
0.1515-0.1520 (beats by ~9% at 11% fewer bits). Best 3-bpw quality
measured in this program.

### T8b (queued for a coordinated GPU window)

One kernel packet: `kernel_mat_vec_trellis3g_3inst_d_f32` (T=256
group-ring, dual-3INST). Op-count model predicts 0.87-0.95x Q4_K BW.
Gates: >= 0.85x same-session Q4_K BW = HOLD (ships the quality tier at
>= 1.25x time-speedup on quantized streams with best-in-tier quality);
>= 0.92x = STRONG. Tuning budget 3 iterations. Plus assembly/microbench
verification of rotl/xor costs (cx structural note: source ops are not
Apple instructions). Real-weight fidelity for the code is already
banked above; a passing kernel completes the shipping triple
(quality + bytes + speed) for the 3-bpw tier, ahead of the LDLQ phase
which lifts all rows further.

