# T5: trellis3 Gaussian source-coding quality oracle — PREREGISTERED

Phase 2a of the trellis swing (throughput floor: SURVIVE-STRONG at
`../2026-07-19-trellis3-gemv-floor/`). CPU-only; no GPU time. Prices the
two unpublished deviations the throughput winner carries (V=2 split
computed code; T=32 span tail-biting) against canonical QTIP codes, the
optimal-scalar floor, and the product-relevant llama.cpp 3-bit formats,
on an i.i.d. Gaussian source (what weights look like after incoherence
processing).

Identity: zekrom M4 Max, commit at HEAD after t1 close, seeded sample
shared across every row.

## Instrument

`crates/qwen-llm/examples/trellis3_quality_oracle.rs`:

- Source: `N = 64 groups x 256 = 16,384` standard-normal weights
  (seeded splitmix64 Box-Muller), identical for all rows. MSE standard
  error ~ sqrt(2/N) ~ 1.1% — adequate for the gate margins below.
- Trellis encoder: exact Viterbi over the L=16 bitshift trellis
  (65,536 states; 2^k in-edges per state, k=3 V=1 / k=6 V=2), two-pass
  tail-biting heuristic (pass 1 free initial window, pass 2 pins the
  initial window to the winning path's tail — standard TB decoding,
  near-ML; same discipline for every trellis row so T comparisons are
  fair).
- Scale: one fp16 scale per 256-group for trellis rows (matches the
  shipped kernel layout), fitted by encode -> least-squares refit ->
  re-encode (one iteration, same discipline all rows).
- Decode-value tables precomputed per code over all 65,536 states.

## Rows (all bpw include scale overhead)

| row | config | bpw | role |
| --- | --- | ---: | --- |
| A | V=2 T=32 split mask/or (the shipped 3inst_v2 EXACTLY) | 3.0625 | candidate |
| A2 | V=2 T=32, pair = (hx+hy, hx-hy) | 3.0625 | rescue candidate (near-zero support restored; +1 half-op in kernel) |
| B | V=1 T=32 mask/or half-sum (shipped 3inst EXACTLY) | 3.0625 | candidate (slower kernel, 0.87-1.03 of Q4_K BW) |
| C | V=1 T=256 canonical 3INST (xor form) | 3.0625 | QTIP reference ceiling |
| C2 | V=1 T=256 mask/or half-sum | 3.0625 | isolates code-form (xor vs mask/or) |
| D | V=2 T=256 split mask/or | 3.0625 | isolates T effect vs A |
| E | Lloyd-Max 8-level scalar, fp16 scale per 32 | 3.5 | TQ3_1S-class incumbent floor (its real config) |
| E' | Lloyd-Max 8-level scalar, fp16 scale per 256 | 3.0625 | equal-rate scalar floor |
| F | Q3_K via ggml_quantize_chunk | 3.4375 | product comparator (indicative on Gaussian) |
| G | IQ3_XXS via ggml_quantize_chunk (no imatrix) | 3.0625 | byte-matched product comparator (indicative on Gaussian) |
| — | Shannon D_R = 2^-2R at R=3 | — | 0.015625 lower bound |

Known distribution hazard, stated before running: A's exposed halves
have |v| in [0.125, 2)·s — no near-zero support. The Viterbi encoder
will mitigate; the oracle measures what remains. A2 exists because
(sum, difference) restores near-zero support at identical bytes.

## Gates (frozen)

- KILL the V=2-split branch: MSE(A) > MSE(E') (equal-rate scalar floor
  beats the shipped code — shaping gain destroyed).
- PASS a candidate row X in {A, A2, B}: MSE(X) <= 0.85 x MSE(E') AND
  MSE(X) <= 1.20 x MSE(C). Preference order at pass: A (fastest
  kernel), A2 (+1 half-op), B (slower but proven >= 0.85 BW gate).
- If no candidate passes but C does (expected): the fast-code branch
  dies; phase 2b proceeds with a C-shaped (V=1 T=256) kernel whose
  throughput already cleared 0.85 in T1, and a new V=2 code search
  becomes optional future work, not a blocker.
- F/G rows are recorded, not gated (Gaussian-source numbers are
  indicative for grid quants; the binding IQ3_XXS comparison happens on
  real weights in phase 2b).

## Results — candidates FAIL, fallback clause fires — 2026-07-19

Instrument note: the first run failed its own self-check (every trellis
row at MSE ~ var) and exposed a trellis-direction bug in the encoder
(left-shift transitions vs the kernels' right-shift window convention;
the decode side was already GPU-validated). Fixed; the self-check then
passed. One preregistered-spirit extension was added before closing:
the A2 code at T=256 (same gates applied).

| row | bpw | MSE | vs E' | vs C |
| --- | ---: | ---: | ---: | ---: |
| A  v2_split T=32 (shipped 3inst_v2) | 3.0625 | 0.101501 | 2.617x | 5.160x FAIL |
| A2 v2_sumdiff T=32 | 3.0625 | 0.107562 | 2.773x | 5.469x FAIL |
| B  v1 T=32 (shipped 3inst) | 3.0625 | 0.152393 | 3.929x | 7.748x FAIL |
| C  v1 canonical 3INST T=256 | 3.0625 | 0.019669 | 0.507x | 1.000 |
| C2 v1 mask/or T=256 | 3.0625 | 0.019439 | 0.501x | 0.988x |
| D  v2_split T=256 | 3.0625 | 0.024473 | 0.631x | 1.244x fail (near) |
| A2' v2_sumdiff T=256 | 3.0625 | 0.024221 | 0.625x | 1.231x fail (near) |
| E  Lloyd-Max /32 (TQ3_1S-class) | 3.5 | 0.033230 | — | — |
| E' Lloyd-Max /256 (equal rate) | 3.0625 | 0.038784 | 1.000 | — |
| F  Q3_K (ggml, no imatrix) | 3.4375 | 0.022963 | — | — |
| G  IQ3_XXS (ggml, no imatrix) | 3.0625 | 0.045777 | — | — |
| D_R Shannon bound | — | 0.015737 | — | — |

Source var 1.0072, n=16,384. Self-check green: C at 1.25x the Shannon
bound is textbook TCQ-class, validating the encoder.

### Findings

1. T=32 span tail-biting is a quality catastrophe (4.1x the T=256 MSE
   at V=2; worse at V=1): the 96-bit ring's 13 seam bits double-duty
   for ~27% of window reads and the shaping gain collapses below even
   the scalar floor. Every preregistered candidate FAILS.
2. The sumdiff rescue is REFUTED at both T: the V=2 penalty is not
   value support but effective per-weight state resolution. At T=256
   it is a structural ~0.16-effective-bit tax
   (0.5*log2(0.0245/0.0197)); both V2@256 rows miss the 1.20x-of-C
   gate by 3-4% while comfortably beating scalar (0.63x E').
3. Code form is neutral (C2/C = 0.988): the mask/or constants shipped
   in T1 are quality-equivalent to canonical XOR 3INST — keep them.
4. T=256 does NOT break the T1 kernel structure: bitshift-trellis
   decode is position-random-access (only Viterbi encoding is
   sequential). A lane covering weights [32t, 32t+32) of a group ring
   reads one extra overlap word (+4 B per 32 weights, cache-served,
   ~+33% load instructions on the weight stream but ~zero DRAM bytes).
5. Comparators on this source: trellis-V1 beats Q3_K by 18% MSE at 11%
   fewer bits and byte-matched IQ3_XXS by 2.36x. Grid-quant rows are
   handicapped without imatrix on a synthetic Gaussian (stated in the
   prereg); the binding comparison is real-weight phase 2b.

### Verdict

Preregistered fallback clause fires: the fast-code branch as shipped
(T=32) is dead; the design point moves to T=256 group-ring with two
live kernel options:

- V1 mask/or (canonical-class quality 0.0194): throughput 0.867
  primary / 1.026 control from T1 with ZERO tuning iterations spent —
  the next packet tunes this (NSG=4 occupancy per the Apple9
  cross-simdgroup lever, load pipelining, overlap word) toward the
  0.95-1.00 band.
- V2 split (0.0245, ~0.16-bit tax): 1.075-1.130 throughput banked;
  speed-fallback if V1 tuning stalls, or the aggressive tier if
  real-weight evals price the tax as acceptable.

Next packets: T6 = T=256-layout kernel variants (V1 + V2, overlap
word) under T1's unused tuning budget, gate V1 >= 0.95x Q4_K BW
(stretch) / >= 0.85 (hold), plus the deferred skinny-shape cells and
same-session A/B/A big-buffer pair. T7 (phase 2b) = real-weight PTQ:
Hadamard/incoherence + Viterbi encoder against one real Qwen tensor
class, fidelity oracle vs Q4_K/Q3_K/IQ3_XXS/TQ3_1S with imatrix where
applicable; PonyExl3's Qwen3.6-27B 4.15 bpw dPPL +0.015 as external
anchor.
