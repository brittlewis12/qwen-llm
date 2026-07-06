# W1b: Partition-Packed Attention Decode Main

Status: RUN, FALSIFIED at the pre-registered control row. Design signed
(cx session `019f347b-c...`, conditional-go; five required gate edits
applied verbatim).

## RESULTS (A3B Qwen3.6, attn-intra ctx131072, runs 3, prefill-warm)

- G0 PASS: packed PSO reports maxTotalThreadsPerThreadgroup=128 (vs 32 on
  the flat kernel) - legal launch.
- G1 PASS (non-vacuously): `attn_v4_matches_naive_f16kv` green 3/3 with
  `QWEN_ATTN_V4_PACK=4 QWEN_ATTN_V4_G8_TILE=4` forcing the packed route
  (the default test sweep never reaches n_pos>=16384, so tile4 never
  fires without the override - recorded so future gates do not run
  vacuous G1s). NaN-prime partials test green with PACK on. One flaky
  matches_naive failure during gating reproduced the DOCUMENTED v0.433
  load-flake signature (n_pos=1024 nwg=64 C=16, cos 0.9665 vs recorded
  0.9662, rotating, on a tile2 path the pack code cannot reach) -
  ambient corruption class, not W1b.
- P-W1b-1 KILL FIRES: pack4@NWG256 (packaging-only control, identical
  partitions/streams/threads) regresses main `0.6842 -> 1.1050 ms/layer`
  (+61.5%). pack4@512 `0.7767` (+13.5%), pack4@1024 `0.8332` (+21.8%,
  reduce grows 0.0708 -> 0.3519 ms). Within packed rows more streams
  helps (512 beats 256-packed by 30%), consistent with the latency-hiding
  mechanism, but the 128-thread packaging cost swamps it.
- Verdict per the signed kill semantics: "pack4 register/TGM packaging
  failed" (leading suspect: register allocation at 4 resident simdgroups
  per TG - the 32-thread attribute's register budget is what makes each
  stream fast). This result does NOT license "attention main is
  stream-issue-bound"; the G2 occupancy-pair condition for that claim was
  never reached.
- Attention main is now falsifier-bracketed on SIX axes: split count
  (v0.495), per-TG byte shape (v0.463/488), score-broadcast (v0.490/495),
  Q8 KV bytes (v0.437), thread-cap hints (v0.435, neutral), and partition
  packing (this). The one-simdgroup 32-thread body is locally optimal in
  its entire tested neighborhood; future attention-main work must be a
  materially different body (the parked FA2-style matrix design branch),
  per the same conclusion v0.437 reached from the byte side.
- gdn_step packing check: the wave-collapse arithmetic (4096 TGs = ~3.4
  W32 waves -> 1024 packed TGs ~ 1 wave) SURVIVES this falsifier in
  principle (attention had no waves to collapse), but the +61% packaging
  cost haircuts the prediction below its ~0.3 ms ceiling on a 0.615 ms
  family - park unless a register-light packing shape appears. First engine experiment of the W-program
(width/concurrency, attribution-first mandate from session `019f347b-c...`).
Budget: ~0.5-1 day including runs.

## Evidence base (do not re-derive)

- v0.494: attention (`attn_mixer_route`) is `46.2%` of the A3B token at
  ctx131072 (~7.5 ms unperturbed); the 131k limiter regime is
  latency/occupancy-bound (occupancy ~26%, read BW ~66% of stream), NOT
  bandwidth-walled.
- v0.495 census: the decode main kernel
  (`kernel_attn_decode_v4_g8_t4_c64_f32`) runs a FIXED `1024 x 32` grid at
  both 16k and 131k = 2 kv-heads x 2 subgroups x NWG=256 one-simdgroup
  TGs. Measured occupancy ~26% ~= 32768 threads / stall-fill capacity
  (27.8%): the kernel is fully resident in one wave and occupancy IS its
  grid geometry.
- B0 arm R (v0.493): W32 TG-slot plateau ~30 TGs/core (max_alive noisy
  1191-2311; steady p10 ~720 = 18/core); W128 sustains 742-961 TGs
  (74-96 simdgroups/core); memory-stalled W32 fill reaches ~92 TGs/core.
  A memory-heavy 32-wide kernel at 25.6 TGs/core sits ~3.6x under the
  stall-fill ceiling because each simdgroup burns a whole TG slot.
- v0.495 NWG falsifier RE-READ: flat NWG=512 is 2048 TGs at W32 - at or
  ABOVE the measured W32 residency cap, so its +17.5% loss confounds
  stream-shortening with WAVE SERIALIZATION. Packing decouples the two:
  more resident simdgroups without more TG slots.
- The 32-thread cap on the main kernels is a SOURCE ATTRIBUTE
  (`[[max_total_threads_per_threadgroup(32)]]`), i.e. a chosen
  register-budget trade, not a hardware limit (v0.435 kept these as
  neutral hints; v0.458 showed no tool reports register counts - the
  packed PSO's own `maxTotalThreadsPerThreadgroup` is the register
  oracle).

Falsifier lineage (all DIFFERENT axes; none tested packing):
v0.437 killed Q8 KV bytes; v0.463/488 killed subgroup BYTE-shape variants
(V-staging, tile1); v0.490/495 killed score-broadcast promotion; v0.495
killed flat split-count changes at W32. Program A ("TG packaging") was
demoted at v0.491 on the boundary-drain hypothesis, which v0.493 itself
killed - packing returns with a counter-backed residency argument.

## Mechanism

Pack `PACK=4` partitions into one 128-thread TG. Each simdgroup executes
EXACTLY today's per-partition work: same K/V reads, same online-softmax
math, same partial writes. Only the packaging changes:

- partition index: `iwg = tgpig.z * PACK + simdgroup_index_in_threadgroup`;
- BARRIER/TAIL INVARIANT (cx-required, binding): NO simdgroup may exit
  before any TG-wide barrier. Empty/tail partitions (iwg >= n_partitions
  or empty position range) MUST participate in the shared Q-load
  `threadgroup_barrier`, then write their sentinels and idle WITHOUT
  returning across any remaining TG-wide barrier. After the single
  Q-load barrier all phase synchronization is `simdgroup_barrier` on
  per-simdgroup `ss` slices (no cross-simdgroup scratch dependence), so
  post-barrier early exit is safe and must be justified in a code
  comment against this invariant;
- `ss` (score/weight scratch) becomes per-simdgroup slices
  (`PACK x GROUP_TILE x C` floats = 4 KB at t4/C64) and the TG-wide
  barriers around simdgroup-private phases become `simdgroup_barrier`;
- `sq` (the Q tile) is IDENTICAL for all packed partitions of the same
  (kv-head, subgroup): loaded once per TG behind one `threadgroup_barrier`
  (side bonus: 4x fewer Q loads);
- `[[max_total_threads_per_threadgroup(128)]]` on the new entry point;
- host: grid z = `ceil(nwg / PACK)`, 128 threads/TG, TGM sized
  accordingly. Selection behind `QWEN_ATTN_V4_PACK=4` (default OFF; the
  production flat path is untouched).

Scope: ONE production-relevant instantiation first
(`kernel_attn_decode_v4_g8_t4_c64_pack4_f32`, F16 KV). Other shapes only
if this wins.

## Why this could move real time (and the register risk)

At pack4 the same 1024 simdgroups occupy 256 TG slots, freeing the
scheduler to hold MORE simdgroups: pack4@NWG512 doubles resident
simdgroups/threads (65536 = ~55% of stall-fill) in a single wave (512 TGs
vs the ~740-960 W128 cap), where flat@512 needed 2048 W32 slots (>= 2
waves). If attention main is occupancy/latency-hiding-bound (all limiter
evidence says the machine is), 2x outstanding KV streams at unchanged
byte shape should cut main time substantially; the A0-era arithmetic
bounds the win at roughly the 1.3x gap to its own streaming floor
(~7.5 -> ~5.8 ms attention-family time at 131k, i.e. ~10% e2e) with
honest uncertainty in both directions.

RISK, stated plainly: the 32-thread attribute currently buys maximum
per-thread registers (o_acc[GROUP_TILE][2] float4 + softmax state ~50-80
live values). At 128 threads/TG the compiler may spill or cap
`maxTotalThreadsPerThreadgroup` below 128. This is the FIRST gate and it
is decided in minutes by `qwen-bench metal-pipelines` on the new PSO.

## Pre-registered matrix and gates (ctx131072 primary; ctx16384 guard)

Rows (attn-intra micro first, then full decode for survivors):
1. flat@NWG256 (production default) - baseline;
2. pack4@NWG256 - packaging-only control: same partitions, same streams,
   same 32768 threads; isolates TG-slot/sq-sharing effects;
3. pack4@NWG512 - 2x simdgroups, single wave, 256-position streams;
4. pack4@NWG1024 - 4x simdgroups (borderline 1-1.4 waves at W128 cap);
   curve point.

Gates (as amended by the cx review):
- G0 (LEGAL-LAUNCH gate, per cx: not a register oracle): packed PSO
  reports `maxTotalThreadsPerThreadgroup >= 128`, else KILL the packing
  lane immediately and record the reported number. G0 proves the PSO can
  legally host 128 threads; register ECONOMICS are decided by timing and
  counters, not this gate.
- G1 (correctness): `attn_v4_matches_naive_f16kv` green with
  `QWEN_ATTN_V4_PACK=4` at every matrix NWG; bit-level parity NOT required
  vs flat (different partition count changes reduction order) but the
  existing cos/max_abs oracle gates apply.
- P-W1b-1 (cx wording): pack4@256 must not regress main time > 5%; if it
  IMPROVES, record that as packaging-only upside (no upper band).
- P-W1b-2: best pack row improves attn-intra MAIN >= 10% at ctx131072.
- G2 (PAIRED COUNTER gate, cx-required, before any full-decode claim):
  same-session limiter captures of flat@256 AND the best packed row;
  Kernel Occupancy / Compute SIMD Groups Inflight must MOVE in the
  predicted direction (packed > flat) before the mechanism is claimed. A
  single winning-row capture is insufficient; the pair is the gate.
- P-W1b-3: that row improves full-decode gpu_ms >= 3% at ctx131072 with
  ctx16384 regression <= 1%. Promotion decision (default flip vs opt-in)
  only via the standard repeated full-decode gate, separately.
- KILL SEMANTICS (split, per cx):
  - G0 fail, pack4@256 regression > 5%, or packed occupancy NOT rising in
    G2 => record "pack4 register/TGM packaging failed" - says nothing
    about what binds attention main.
  - Packed occupancy RISES in G2 and main time still does not improve
    (P-W1b-2 fail) => record "attention main is not TG-slot-bound;
    likely stream-issue-bound" - the packing axis becomes the fifth
    bracket on attention main.
  - Either way the W-program falls back to (a) the census narrow-glue
    list and (b) a gdn_step packing check ONLY if its 3.4-wave arithmetic
    still motivates it after this result.

Follow-up (only on P-W1b-2 success): same-mechanism confirmatory on
`gdn_step_decay` (4096 x 32 = ~3.4 W32 waves at 25.6/core; packing to
W128 could collapse wave count) - separate checkpoint, own gates.

## Validation plan

- `cargo test --release -p qwen-llm --lib attn_v4_matches_naive_f16kv`
  with and without `QWEN_ATTN_V4_PACK=4`, plus `QWEN_ATTN_V4_NWG`
  overrides for matrix rows.
- `qwen-bench metal-pipelines --kernel kernel_attn_decode_v4_g8_t4_c64_pack4_f32`
  for G0 (and record static TGM).
- `qwen-bench attn-intra -m <A3B 3.6> --ctx 131072` per row (window/runs
  per attn-intra defaults; quiet box).
- Full-decode rows via `ctx-sweep --prefill-warm` window 16, 2 reps
  interleaved, gpu_ms primary.
- One `gpu_limiter_capture.py` capture on the winning row (occupancy must
  MOVE if the mechanism is real - counter movement before e2e claims).
