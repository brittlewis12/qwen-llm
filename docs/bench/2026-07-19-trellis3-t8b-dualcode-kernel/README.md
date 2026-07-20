# T8b: dual-3INST group-ring kernel — PREREGISTERED

Executes the T8a-queued packet under exclusive GPU access (granted;
box quiet, co-tenant confound removed). Code: dual-3INST R13 (T8a
winner: 0.977x V1 synthetic / 0.983x V1 real-weight quality at ~6.5
ops-eq/weight). Kernel: `kernel_mat_vec_trellis3g_3inst_d_f32`, T=256
group-ring, byte-identical layout (98 B / 256 w, 3.0625 bpw).

Op-count model predicts 0.87-0.95x Q4_K bandwidth (between V1G's
0.82-0.85 at ~8 ops/w and V2G's ~1.0 at ~3.7 ops/w).

## Protocol

1. Correctness: bit-faithful CPU-reference arm; same thresholds as T1.
2. Timed: THREE sequential invocations of the four-row pair set
   {q4_k ffn_gate, 3inst_d_g256, 3inst_v2_g256, 3inst_g256} chained64
   on the primary cell (5120x17408). Per-invocation same-session
   ratios; median-of-3 decides. Thermal capture before/after.
   This simultaneously regrades the T6/T7a-deferred V2G claim on a
   genuinely quiet box (same median/spread rule: spread <= 0.05).
3. Diagnostics (one invocation): 3inst_d on ffn_down_t + embed
   (476 MB uncacheable) + skinny cells.
4. Tuning budget: 3 iterations (only if HOLD missed).

## Gates (frozen)

- HOLD: 3inst_d median ratio >= 0.85x same-session Q4_K GB/s ->
  ships the 3-bpw quality tier at >= 1.25x quantized-stream
  time-speedup with best-in-tier quality (clears the jam's
  "experimental Q3-successor" product bar in full).
- STRONG: >= 0.92x -> >= 1.35x; authorizes the integration spike
  (B5) next.
- V2G regrade: median >= 1.00 with spread <= 0.05 -> restore the T6
  parity claim at promotion grade; 0.95-1.00 -> demote to parity-band
  permanently.
- KILL: < 0.85 after budget (tier falls back to V1G economics).

## Results — 2026-07-19 (exclusive GPU)

Correctness green (bit-faithful CPU arm, cos 1.0). Thermal clean
before/after every session.

Primary protocol (3 invocations, same-invocation ratios vs Q4_K):

| kernel | inv ratios | median | spread |
| --- | --- | ---: | ---: |
| 3inst_d (original) | 0.841 / 0.824 / 0.794 | 0.824 | 0.047 |
| 3inst_v2_g256 | 1.052 / 0.967 / 0.933 | 0.967 | 0.119 |
| 3inst_g256 (V1) | 0.890 / 0.837 / 0.849 | 0.849 | 0.053 |

Tuning budget (3 iterations, all recorded):
1. Split accumulators: 0.803/0.847/0.807, median 0.807 — null/negative.
2. Cheap second word (shr3): ORACLE-KILLED before kernel work
   (1.346x V1 quality; shared bits destroy the joint — the rotl+xor
   mixing is load-bearing).
3. Bounded unroll_count(8): catastrophic 0.38x — partial unroll broke
   compile-time window-offset folding (diagnostic: full constant
   folding is worth ~2.2x in this kernel). Reverted.

Post-budget confirmation of the reverted original: 0.869 — its best
row. Four-invocation envelope 0.794-0.869 (spread 0.075) on an
EXCLUSIVE quiet box: near-wall ratio measurements carry ~±4-5%
session-scale nonstationarity even without co-tenancy. Third
independent evidence packet for the measurement-control-plane swing;
any future 3%-scale gate in this program needs >= 5 invocations and a
spread rule.

### Gate outcomes (frozen protocol)

- 3inst_d HOLD (>= 0.85): MISSED at the preregistered median-of-3
  (0.824) -> KILL fires for the ship-at->=1.25x claim. The kernel
  straddles the gate inside measurement noise; reopen condition: a
  >= 5-invocation protocol under improved measurement control, or any
  mechanism worth +0.03 ratio.
- V2G regrade: median 0.967, spread 0.119 -> demoted to PARITY-BAND
  permanently (0.93-1.05 envelope).

### What stands after T8b

The 3.06-bpw tier's honest card: dual-3INST decode at 0.79-0.87x Q4_K
bandwidth (~1.17-1.28x quantized-stream time-speedup), -32% quantized
bytes, quality 0.983x V1 = beats Q3_K by ~9% at 11% fewer bits —
best-in-tier quality with V1G speed. Whole-27B projection at 63%
coverage: ~1.10-1.14x decode, -20% weights memory. BELOW the
preregistered ship bar (>= 1.25x stream), so the tier does not ship on
speed grounds alone. The named lever that changes the economics is the
LDLQ/Hessian + two-sided RHT phase: it lifts quality at constant
bytes toward the Q4_K anchor, repositioning the tier as a
memory/quality product (the jam's second product bar) where 1.1-1.2x
decode is a bonus rather than the headline.

