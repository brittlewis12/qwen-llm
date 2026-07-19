# T1: trellis-quant (QTIP-class) decode-GEMV throughput floor — PREREGISTERED

Program: swing B3 ("trellis quantization engine for Metal"), phase 1 of 3
(1 = decode throughput floor, 2 = quality/PTQ pipeline, 3 = integration).
This packet is bench-only and makes NO quality, NO end-to-end, and NO
production claim. It answers exactly one question:

> Can a bitshift-trellis (L=16, K=3 bits/weight) decode-GEMV, with the
> trellis state walk and per-weight computed/LUT code executed honestly
> per weight, sustain >= 85% of the incumbent Q4_K GEMV's achieved
> compressed-bytes bandwidth on the same production shape on M4 Max?

Identity: zekrom M4 Max 40-core / 128 GB, macOS 15.6.1, base commit
`824dd65` (clean at preregistration), AC power, quiet box. Thermal/battery
state to be recorded at run time. Single benchmark process; no parallel
timed work (roadmap GPU-lease rule).

## Why this gate

Decode is bandwidth-bound; bytes are the currency. Q4_K_M effective rate
on quantizable matrices is ~4.5 bpw (144 B / 256 weights). Trellis K=3
with one fp16 scale per 256-weight group is 98 B / 256 weights (~3.06
bpw), a 0.681x byte ratio. Kernel-time speedup on quantized-tensor time
is (byte ratio)/(BW ratio):

| BW vs Q4_K | kernel-time speedup |
| ---: | ---: |
| 100% | 1.47x |
| 90% | 1.32x |
| 85% | 1.25x |
| 68% | 1.00x (breakeven) |

85% is the preregistered survive line: it preserves >= 1.25x on quantized
weight streams, which clears the lossy-lane admission band (>= 15%) with
margin left for quality-phase and integration losses. Below 85% the
representation cannot pay for its own risk; kill.

The ALU question this decides: QTIP-class decode needs an L=16 state
walk (shift/mask) plus a computed code (LCG imad32 + shaping) or small
LUT per weight. On Apple GPUs imul32 has historically been sub-full-rate;
at 5120x17408 a GEMV touches 89.1M weights, so a ~10 cycle-equivalent
per-weight decode could exceed the ~70 us memory floor. Whether decode
fits under the memory latency shadow is not derivable from specs with
confidence; it must be measured. That is this packet.

## Objects

- `kernels/trellis_gemv_floor.metal` — BENCH-ONLY kernels (never in
  production dispatch tables), >= 2 preregistered decode variants:
  - `kernel_mat_vec_trellis3_3inst_f32`: faithful QTIP-3INST-style
    computed code. Per weight: 3 fresh bits into a rolling window,
    `state = win & 0xFFFF`, LCG `X = state*A + B` (imad32), `X ^= magic`
    (packed fp16 pair mask), `w = (half)(as_type<half2>(X).x +
    as_type<half2>(X).y)`, fp32 FMA with the activation.
  - `kernel_mat_vec_trellis3_lut8x2_f32`: Apple-native lookup code.
    Same state walk; `w = LUT_hi[state>>8] + LUT_lo[state&255]`, two
    256-entry half LUTs in threadgroup memory (1 KB), loaded once per
    threadgroup.
  - `kernel_mat_vec_trellis3_3inst_v2_f32`: V=2 computed code
    (pre-run amendment, before any timed measurement): one state per
    TWO weights (step advances 6 bits, L=16 window), one LCG hash
    yields `as_type<half2>(X)` whose .x/.y are the two weight values
    directly (no half-sum). Halves state+code work per weight
    (~3.7 ops/weight est.); QTIP's shipped kernels use V=2 similarly
    (HYB). Quality implications deferred to phase 2 like T=32.
  - `kernel_mat_vec_trellis3_hyb_v2_f32` (pre-run amendment from cx
    research session `019f7c4b`): QTIP HYB-style, V=2 — one integer hash
    `h = st*st + st`, one half2 lookup from a 512-entry (2 KiB) DEVICE
    memory LUT (Apple9 flexible cache; research recommends against
    threadgroup staging), sign of .y flipped by hash bit 15. Lowest
    imul pressure per weight of the computed variants.
  - 1MAD and MUL1 are explicitly excluded (no packed byte-sum op on
    Apple); an FP-pipe hash is excluded (`fract` is 4-cycle complex-pipe).

Research notes binding this packet (cx `019f7c4b`, full transcript
retained): Apple7/8 measured imul32/imad32 at quarter-rate and no public
evidence M4 changed that; Apple9's documented gain is cross-simdgroup
overlap of int/FP16/FP32 pipes, so occupancy (NSG sweep) is a sanctioned
tuning axis. Gate arithmetic restated in wall time: at 0.667x bytes,
>= 85% compressed-GB/s means the trellis kernel must finish ~15-28%
FASTER than the same-session Q4_K wall time on the primary shape.
Nearest prior art: PonyExl3 (Metal/MLX EXL3 GEMV, exact inline decode;
reports Qwen3.6-27B @ 4.15 bpw dPPL +0.015, and end-to-end decode
SLOWER than this repo's Q4_K path on newer silicon) — strengthens the
case that a kernel-level floor, not an end-to-end port, is the right
discriminating instrument.
- CPU reference decoder (Rust, test-only) implementing bit-identical
  layout and code math; correctness gate for both kernels.
- `benches/kernels.rs` group `trellis3 mat_vec` with same-session Q4_K
  comparator rows, single + chained64 regimes, persistent buffers —
  identical harness discipline to the existing `q4_k mat_vec` group.

## Layout (frozen for this packet)

- Row-major rows over n_in; groups of 256 weights per (row, group).
- Group = 8 spans x 32 weights. Span = 32 weights x 3 bits = 96 bits =
  6 uint16 words, packed MSB-first (EXL3-style span self-alignment).
  Trellis window is tail-biting within the span (T=32): state for
  weight j is the 16-bit window ending at bit 3*(j+1) mod 96. One
  simdgroup lane decodes one span serially; 8 lanes cover a group.
  (T=32 tail-biting vs QTIP's T=256 is a quality question deferred to
  phase 2; per-weight decode COST is identical, which is all this
  packet measures.)
- Per-group fp16 scale, applied once per group to the group partial
  (mirrors Q4_K's per-super-block `d` fold cost).
- Bytes: 8*12 + 2 = 98 B per 256 weights. Primary shape ffn_gate_27b
  5120x17408: 34.13 MB trellis vs 50.13 MB Q4_K.
- Synthetic content: seeded uniform-random bitstream + unit-ish random
  scales. Throughput and mechanical correctness only.

## Dispatch geometry

Start at the Q4_K fast-kernel geometry (32-lane simdgroup, NSG=2,
NR0=2). Geometry, span-to-lane mapping, unroll depth, buffer word
interleave, and refill strategy are TUNING degrees of freedom inside
the packet budget. The decode-work definition is NOT tunable: every
weight must pass through a distinct L=16 state and the variant's code
function; no batch shortcut that skips per-weight state generation.

## Protocol

1. Build release, run correctness tests (CPU reference vs both GPU
   variants, cosine >= 0.999999 and max |rel| <= 1e-3 on y).
2. `cargo bench -p qwen-llm --bench kernels -- 'trellis3|q4_k mat_vec'`
   — one invocation, same session, AC power, quiet box; record
   `pmset -g therm` before/after.
3. Primary cell: chained64 on ffn_gate_27b (5120x17408). Secondary
   (diagnostic, non-gating): single regime; ffn_up shape; ffn_down-like
   transpose shape if time permits.
4. Metric: achieved GB/s = compressed bytes x 64 / median chained64
   time, vs same-session Q4_K chained64 achieved GB/s on the identical
   logical shape.
5. Tuning budget: at most 3 tuning iterations per variant after first
   measurement. Every iteration's number is recorded; no silent best-of.

## Gates (frozen)

- SURVIVE: any variant >= 85% of same-session Q4_K achieved GB/s on the
  primary cell, with correctness green. Authorizes phase-2 planning
  (quality/PTQ pipeline design, Hadamard runtime charge, real-weight
  fidelity oracle).
- STRONG: >= 95% — additionally authorizes a K=2 (2-bit) variant probe
  and an N=4..16 skinny mat-mat variant sketch in phase 2.
- KILL: all variants < 85% after budget. Record limiter diagnosis
  (occupancy, ALU class, TGM conflicts) via one diagnostic capture;
  close the lane with the diagnosis as the reopen condition.

## Non-collision statement

This does not reopen closed lanes: it is a WEIGHT representation with
fewer bytes (planar Q4/Q6 repack was closed because it removed no
bytes; routed-Q5 closures were same-layout retunes; KV-format closures
are KV-side). Nearest local prior art: `~/code/llama-cpp-turboquant`
TQ3_1S (WHT-rotated 3-bit scalar Lloyd-Max, ~3.5 bpw, Metal fused
GEMV) — a scalar comparator that strengthens phase 2 (quality ladder,
rotation machinery, GGUF type precedent) and does not overlap this
packet's trellis-decode question. Runtime Hadamard/incoherence cost is
explicitly OUT of this packet and charged in phase 2 (prior: one
128-point RHT per projection input vector, O(n log 128) fp16 ops on a
5120-vector, negligible vs 34 MB weight stream; must still be measured).

## Results — SURVIVE (STRONG gate cleared) — 2026-07-19

Correctness: all four variants match the bit-faithful CPU reference at
n_in=512/n_out=66 (partial-threadgroup guards exercised): max|Δ| <=
1.9e-6 at max|y| ~ 8-19, cosine 1.000000000. Gate green.

Timed session 1 (primary, same invocation for every row):
`cargo bench -p qwen-llm --bench kernels -- 'trellis3|q4_k mat_vec'`,
AC power, no thermal/performance warnings before or after, no parallel
work. Zero tuning iterations used (v0 geometry NSG=2/NR0=2 throughout).
Criterion medians; GiB/s on charged compressed bytes (98 B / 256 w
trellis, 144 B / 256 w Q4_K).

| chained64 row | wall ms | GiB/s | vs Q4_K ffn_gate 414.1 |
| --- | ---: | ---: | ---: |
| q4_k ffn_gate_27b (comparator) | 7.216 | 414.1 | 1.000 |
| trellis3 3inst_v2 ffn_gate_27b | 4.569 | 445.1 | **1.075 — STRONG pass** |
| trellis3 3inst ffn_gate_27b | 5.667 | 358.9 | 0.867 — passes 0.85 |
| trellis3 hyb_v2 ffn_gate_27b | 10.093 | 201.5 | 0.487 — fail |
| trellis3 lut8x2 ffn_gate_27b | 15.680 | 129.7 | 0.313 — fail |
| trellis3 3inst_v2 ffn_down_t_27b (diag) | 5.101 | 398.7 | 0.963 |
| trellis3 3inst ffn_down_t_27b (diag) | 5.965 | 340.9 | 0.823 |

Timed session 2 (post-hoc SLC-uncacheable control, ~15 min later, same
box/power state, labeled diagnostic): the 33.4 MB primary buffers could
in principle ride the ~48 MB SLC across chained dispatches while
Q4_K's 50.1 MB cannot; a 476 MB trellis buffer (embed shape
5120x248320) removes that concern.

| chained64 control row | wall ms | GiB/s | vs q4_k embed 403.9 |
| --- | ---: | ---: | ---: |
| trellis3 3inst_v2 embed_t3_27b (476 MB) | 63.571 | 456.3 | **1.130** |
| trellis3 3inst embed_t3_27b (476 MB) | 69.991 | 414.5 | 1.026 |
| q4_k embed_27b (715 MB, session 1) | 105.551 | 403.9 | 1.000 |

The uncacheable rows are FASTER than the small-buffer rows (more rows →
more threadgroups → better latency hiding), refuting cache inflation of
the primary cell.

### Verdict

SURVIVE at the STRONG level, scoped to the preregistered cell. The V=2
computed code (3inst_v2) streams compressed weights at ~456 GiB/s
(~490 GB/s decimal; credible at ~90% of the 546 spec — the 474 GB/s
v0.285 anchor is a three-stream F32 calibration, not a read ceiling,
and other repo GEMVs reach ~485-515 GB/s). Decode overhead is small
enough to preserve 107.5% of Q4_K's normalized bandwidth on the
primary cell and 113% on the uncacheable control. Faithful per-weight
3INST also clears the 0.85 gate (0.867 primary / 1.026 control).
LUT-family codes fail decisively (0.31-0.49): on Apple, computed codes
win and table lookups lose, consistent with llama.cpp's IQ-quant Metal
history.

Kernel-time speedup on quantized-tensor streams at equal logical
shape: (144/98) x (BW_T/BW_Q) = 1.469 x 1.075 = 1.580x on the paired
primary cell; the 1.660x figure from the control row is cross-session
and is not promotion-grade until a same-session interleaved A/B/A pair
reruns it. This is a representation floor, NOT an end-to-end decode
claim: lm_head/embed policy, non-quantizable tensors, attention/KV,
per-projection Hadamard runtime, skinny-shape coverage, and above all
QUALITY (real Viterbi-encoded weights vs this synthetic stream) are
phase-2/3 questions.

Per the preregistered STRONG clause, phase 2 is authorized to include a
K=2 (2-bit) probe and an N=4..16 skinny mat-mat sketch alongside the
quality/PTQ pipeline (encoder, incoherence processing, real-weight
fidelity oracle, TQ3_1S scalar comparator from the local
llama-cpp-turboquant fork, and the PonyExl3 4.15-bpw dPPL +0.015 prior
for this exact model family).

Open kernel questions carried to phase 2: why hyb_v2's single cached
half2 lookup per 2 weights loses ~2.2x to its computed twin (limiter
capture would say whether it is load-issue rate or cache behavior);
whether V=2's quality at K=3/L=16 matches QTIP's published V=1/V=2
bands for this window discipline (T=32 span tail-biting).

### Adversarial review addendum (cx session 019f7c67)

Verdict upheld: SURVIVE at STRONG, narrowly as the preregistered
throughput-floor claim for 5120x17408. Bound concerns:

1. Geometry sensitivity is real (gate 445 vs down_t 399 at equal
   weights): the result does NOT generalize to skinny shapes; attn_v
   class (~1024 rows / 256 TGs) needs its own chained64 cell with a
   256/512/1024/2048-row scaling sweep before any breadth claim.
2. The 476 MB control's 1.130x (and the 1.660x projection) are
   cross-session; rerun big-buffer trellis + Q4_K interleaved A/B/A in
   one invocation before quoting them at promotion grade.
3. QUALITY is the live risk, not throughput: published QTIP computed
   3INST is V=1 over T=256 tiles; validated V=2 is the HYB LUT code,
   and QTIP reports increasing V generally hurts quality. This
   packet's split-3INST V=2 (each mirrored-exponential half exposed as
   a separate weight) plus T=32 span tail-biting are two unpublished
   deviations that must be priced FIRST in phase 2.

Phase-2 first experiment (cheapest decisive): a pure Gaussian
source-coding oracle — adapt QTIP's bitshift Viterbi encoder, no model
integration — measuring normalized MSE for: (a) this packet's exact
K=3/L=16/V=2/T=32 mask-or code, (b) same code at T=256, (c) canonical
V=1 3INST, (d) HYB V=2, (e) local TQ3_1S Lloyd-Max scalar (the
llama-cpp-turboquant comparator), with one jointly optimized scale per
eight T=32 spans. If (a) cannot clearly beat (e)'s distortion at equal
bits, stop the fast-code branch and fall back to (b)/(c)-shaped
kernels before building any PTQ pipeline.
