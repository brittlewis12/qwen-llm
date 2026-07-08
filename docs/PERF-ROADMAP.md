# Performance Roadmap

Living cross-session performance plan for qwen-llm. Keep this file concise and
current: update the ranking when new measurements change expected value, risk,
or dependencies. Treat `docs/PLAN.md` as the architecture/history plan; this
file is the active optimization queue.

Working rule: optimize from causal performance hypotheses, not from measurement
novelty. Every measurement task in this file should exist only to kill or
confirm a concrete engine hypothesis.

For append-only checkpoint history and exact current handoff state, see
`docs/PERF-LOG.md`.

## Current North Star

Maximize useful throughput on Apple Silicon across dense and MoE Qwen 3.5/3.6
workloads. Beating llama.cpp is a required milestone and regression guard, not
the endpoint; when a path is far below measured or estimated hardware roofline,
keep hunting even if the current llama.cpp row is already green.

Primary guardrails:

- Dense: `Qwen3.6-27B-Q4_K_M.gguf`
- MoE A3B: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- MoE A10B: `Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf`
- Treat llama.cpp parity as the floor. Promotion-grade wins should also improve
  the hardware-utilization story: higher effective bandwidth for decode, higher
  effective FLOP/s for prefill, or removal of a measured serial/memory pass.
- Scoreboards should carry both comparison axes: qwen/lcpp for external parity
  and qwen/roofline for hardware headroom. v0.285 adds `qwen-bench roofline`;
  current M4 Max anchors are `474 GB/s` stream, `~12.5-12.8 nominal TFLOP/s`
  Q4_K mat-mat through our dispatcher, and `3.03 TFLOP/s` scalar FMA sanity.
- Never run performance benchmarks in parallel.
- Use the repo-pinned llama.cpp benchmark lock for scoreboard comparisons:
  `scripts/bench/llama-cpp.lock.json`, built by
  `scripts/bench/ensure_llama_cpp.py`. Ambient local llama.cpp binaries are
  one-off only and require `--allow-unpinned-lcpp`.
- Treat battery power, battery warnings, and thermal/performance warnings as
  benchmark confounds unless an AC-power rerun confirms the result.
- Treat `prefill_chunk=1024` as a safe default cap, not a long-context optimum;
  candidate long-prompt branches need larger chunk sweeps when feasible.
- Always keep dense 27B in perf analysis while optimizing MoE.
- `scripts/profile/prefill_sweep.py` runs the static GGUF fast-path audit by
  default; use `--require-fastpath-clean` for scoreboard runs where unexplained
  coverage misses should invalidate the comparison.
- Use `qwen-bench suite` for qwen-side synthetic family spot checks when one
  model should be loaded once across many pp/tg shapes. It still allocates fresh
  sequence state per measured row; keep env-variant A/B as process-per-variant
  until hot-path knobs move out of process-global env caches.
- v0.455 supersedes the old v0.389 counter caveat: a user-saved Instruments
  template (`metal-counters`) now makes Apple performance-limiter counters
  headlessly available via `scripts/profile/gpu_limiter_capture.py`. Use them for
  kernel-shape claims; keep throughput claims on untraced `qwen-bench` runs.

## Latest Baseline Snapshot

M4 Max, release `qwen-bench`, clean narrow family spot after `v0.344` against
fresh llama.cpp b9833 (`c818263f2`, `MTL,BLAS`). No recorded thermal or
performance warnings. Artifact:
`docs/bench/2026-06-28-1551-v0345-fresh-lcpp-c818-family/README.md`.

| Model | `pp512` qwen/lcpp | `pp4096` qwen/lcpp | `tg128` qwen/lcpp | Notes |
| --- | ---: | ---: | ---: | --- |
| 0.8B dense | `1.06x` | `1.07x` | `1.41x` | runs=1 spot |
| 2B dense | `1.01x` | `1.05x` | `1.20x` | runs=1 spot |
| 4B dense | `1.04x` | `1.05x` | `1.20x` | runs=1 spot |
| 9B dense | `1.01x` | `1.01x` | `1.07x` | runs=1 spot |
| 27B dense | `1.05x` | `1.15x` | `1.26x` | runs=1 spot |
| 35B A3B | `1.07x` | `1.19x` | `1.42x` | v0.344 Q5 K512 R2 included |
| 122B A10B | `1.02x` | `1.15x` | `1.25x` | runs=1 spot |

Read: the current narrow paired board remains green after refreshing llama.cpp.
Treat old v0.203/b9481 rows as history, not the active decision spine. Broader
long-context, MTP/speculative, or quant-specific sweeps may still expose red
cells, but near-term branches should be hardware-headroom driven unless a fresh
paired repeat contradicts this spot.

Dense decode update: v0.340 production-wires dense GDN front-projection overlap
and defaults it with `QWEN_DECODE_DENSE_CONCURRENT_GDN=0` as rollback. Sequential
`tg128` A/B improves the dense family by about `+2.4-3.8%`:

| Model | Old default | Concurrent GDN | Ratio |
| --- | ---: | ---: | ---: |
| 0.8B | `360.36` | `372.23` | `1.033x` |
| 2B | `218.35` | `226.60` | `1.038x` |
| 4B | `112.57` | `116.81` | `1.038x` |
| 9B | `72.09` | `73.93` | `1.026x` |
| 27B | `24.28` | `24.86` | `1.024x` |

Measured roofline anchors from v0.285 on M4 Max, AC power, high-power mode, no
recorded warnings:

| Anchor | Shape | Measured | Interpretation |
| --- | ---: | ---: | --- |
| Stream | `512 MiB` per buffer | `474.0 GB/s` | decode bandwidth denominator |
| Q4_K mat-mat | `4096x4096x1024` | `12.80 nominal TFLOP/s` | regular prompt projection ceiling |
| Q4_K mat-mat | `8192x8192x512` | `12.53 nominal TFLOP/s` | size-sensitivity check |
| Scalar FMA | `4M elems x4096` | `3.03 TFLOP/s` | scalar/latency sanity, not matmul peak |

Current caveats:

- The full v0.203 family `27B pp512` row showed `0.913x`, but immediate paired
  repeats showed `1.000x` and `1.008x`; treat that cell as parity/noise until a
  longer repeat packet says otherwise.
- Small dense short prefill is still the main primary-family residual. v0.262
  re-anchors 2B Q4 `pp512` as a real paired loss (`0.960x/0.953x`) while 0.8B
  remains too lcpp-variable to drive work alone (`0.936x/0.989x`). v0.263 widens
  dense Q4 fused-SwiGLU default coverage to `hidden <= 2048`, improving 2B
  `pp512` by `+0.5-1.3%` qwen-only and narrowing paired rows to
  `0.968x/0.978x`; rollback is `QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=0`.
  v0.264 also disables Q4 N64 mat-mat tiles for small `n_query <= 512` shapes
  when either projection side is `<= 2048`, giving another `~0-1%` at 0.8B/2B
  `pp512` and narrowing 2B paired rows to `0.971x/0.982x`. The live residual is
  no longer a clean single-kernel story. v0.265 wall reconciliation shows 0.8B
  `pp512` CPU setup/encode/commit is only `~0.17-0.28 ms`; wall is dominated by
  GPU wait, with GPU timestamps at `~65.9-66.0 ms`. Patched local llama.cpp op
  profiles put FFN and GDN-ish aggregates in the same broad range as qwen. v0.267
  counts `205` compute encoders and `433` dispatches for both 0.8B and 2B Q4
  `pp512`, so this is not an 0.8B-only dispatch-count explosion. v0.269 count
  clusters show GDN front+alpha/beta, dense FFN, attention front+rope/scatter,
  and GDN prep dominate dispatch count; v0.270 says a simple token-channel
  parallel GDN prep is correctness-safe but flat/noisy through `pp16384`, while
  v0.274 finds a real head_dim=128 paired-L2 shape win by replacing the oversized
  per-row threadgroup with one simdgroup per row and four rows per threadgroup.
  Keep attention, N64 tile policy, residual-add epilogues, shared-memory policy,
  GDN prep token-channel parallelization, and CPU orchestration off the primary
  branch unless fresh accounting contradicts this.
- v0.272 shows small dense has a benchmark-methodology trap: default llama.cpp
  `n_ubatch=512` creates wins for qwen just above `pp512`, but llama.cpp
  `-ub 1024` reopens 0.8B at `pp512/576/1024` (`0.948x/0.937x/0.960x`). 2B is
  much closer against the same tuned control (`0.978x/0.983x/1.000x`). Treat
  family-default wins as scoreboard wins, not as evidence the tuned small-dense
  gap is closed. v0.274 defaults the head_dim=128 paired-L2 R4 specialization
  with `QWEN_L2_PAIR_HD128_R4=0` as rollback; clean current-commit tuned rows are
  now 0.8B `pp512/1024` at `0.977x/0.996x` and 2B `pp512/1024` at
  `1.019x/1.022x`. The residual is now mostly 0.8B `pp512` fixed-shape overhead,
  not a broad small-dense failure. v0.279 then defaults the same R4 row shape for
  GDN rmsnorm-gated with `QWEN_RMSNORM_GATED_HD128_R4=0` as rollback. Clean tuned
  rows are now 0.8B `pp512/1024` at `1.023x/1.023x` and 2B `pp512/1024` at
  `0.999x/1.034x`; treat 2B `pp512` as parity/noise, not a new kernel red cell.
- A10B very-short prefill is not a kernel-roadmap item unless a warmed paired row
  regresses. The v0.261 default `pp128` paired repeat still loses from cold
  first-touch variance (`0.789x/0.628x`), but `QWEN_PP_WARM_MOE_BANKS=1` gives a
  same-session steady-state win (`284.20` qwen versus `256.47` llama.cpp,
  `1.108x`). Keep outer wall visible because the warm touch is not free.
- A10B decode is now decisively green after the v0.291 Q8 mat-vec default and
  v0.292 clean re-anchor: `tg64/tg128/tg256` are `1.20x/1.20x/1.21x` versus
  pinned llama.cpp. Keep MoE decode active because hardware utilization remains
  the objective, not parity alone.
- A3B long-context decode attention had a stale group8 execution-shape miss.
  v0.293 defaults `GROUP=8` decode to tile2 for `n_pos >= 4096` with
  `QWEN_ATTN_V4_G8_TILE=8` as rollback. The A3B `ctx128/1024/4096/8192/16384`
  sweep is now `100.0/95.0/94.0/89.1/82.2 t/s` versus rollback
  `99.5/94.8/89.4/83.7/72.8`; `ctx16384` attention drops
  `4.80 -> 3.60 ms`. Attention is still the long-context slope term, but the
  first split fix is now defaulted. v0.367 lowers the group-8/group-16
  subgroup/NWG/C64 threshold to `ctx256` with
  `QWEN_ATTN_V4_SUBGROUP_MIN_POS=4096` as rollback. This closes the medium-context
  MoE valley: A3B `ctx3072` moves `94.1 -> 102.8 t/s`, and A10B `ctx3072` moves
  `37.9 -> 43.7 t/s`; phase attribution says A10B attention drops
  `7.29 -> 3.47 ms`. Stop threshold fiddling here unless a fresh context-slope
  sweep finds a new valley; the next attention branch should be true-long KV or
  layout pressure. v0.368 then defaults A3B group-8 true-long decode to
  tile4/NWG256 at `ctx >= 16384`; A3B `ctx32768` moves `75.7 -> 85.4 t/s`, and
  attention drops `5.14 -> 3.68 ms`. This validates KV byte-reduction as the
  right true-long lens, but tile8/NWG256 remains slower than tile4/NWG256. The
  next A3B attention branch needs a better read-once/group-fused occupancy plan;
  otherwise switch to the broader MoE-down branch. v0.369 banks the cheap reduce
  follow-up for that selector: the two-threadgroup reduce moves A3B `ctx16384`
  `91.2 -> 95.0 t/s` and `ctx32768` `85.4 -> 88.2 t/s`. Further attention work
  now needs a structural read-once or partial-traffic reduction, not another local
  reduce split. v0.370 kills the concrete normalized-half partial-traffic proof:
  it is correctness-safe, but A3B `ctx32768` `attn-intra` only moves
  `0.3457 -> 0.3408 ms/layer` (`1.014x`). Do not keep drilling partial storage;
  switch back to MoE/GDN dataflow unless a genuinely new read-once attention
  execution shape appears. v0.403/v0.404 refreshes the A3B long-context board:
  attention is still phase-faithful and the largest slope term, but another local
  NWG/reduce selector is small. NWG192 saves only `0.13-0.15 ms` phase time at
  `ctx16384/32768`; the next attention branch must be a main-body KV/partial
  traffic oracle that clears `>=0.30-0.40 ms` full-decode-equivalent savings.
  v0.461/v0.462 timestamp splits sharpen the diagnosis: at A3B ctx16384/window16,
  `attn_body_out` is `1.917 ms` and attn-intra main+reduce growth explains the
  context slope, while split GDN-after totals are flat from ctx4096 to ctx16384
  (`~2.65 -> ~2.60 ms`). Promote attention work only through normal-path A/B
  (`>=3-5%` at ctx16384, no ctx4096 regression), but rank attention main/reduce
  ahead of fixed GDN-after cleanup for long-context decode. v0.463 then kills the
  most concrete read-once follow-up: a two-simdgroup group8/tile4 V-staged
  sidecar preserves the four-head register shape and shares only V, but A3B
  ctx16384 `attn-intra` regresses main from `0.1047 ms/layer` to `0.2056` at C16
  and `0.3344` at C32. Do not reopen TGM-staged K/V or V-only sharing without a
  counter signal proving lower traffic and preserved occupancy. v0.488 then kills
  the opposite occupancy-only split against the current tile4/NWG256 true-long
  path: group8 tile1 is exact, but A3B ctx16384 `attn-intra` main regresses
  `0.1035 -> 0.1993 ms/layer` and one-layer total regresses
  `0.3490 -> 0.4574 ms` because estimated main bytes jump
  `0.0714 -> 0.2727 GB`. Future attention work needs byte reduction with
  preserved reuse/occupancy, or a real online-attention rewrite, not smaller
  subgroup splits. v0.490 rejects promoting the existing G8 score-broadcast
  attention sidecar to an automatic true-long default: opt-in still has a main-body
  signal, but rollback-first A3B `ctx32768 --window 8` measured rollback
  `90.4 t/s` versus auto `90.1 t/s`. Keep `QWEN_ATTN_V4_G8_BCAST=1` opt-in until
  it clears a repeated full-decode gate. v0.464 splits the tail and kills
  standalone argmax work: A3B ctx16384 has `lm head 0.87 ms / 7.2%` but
  `lm argmax 0.05 ms / 0.4%`; treat lm-head as a batched/dataflow economics
  component, not a local argmax-fusion branch.
- Shared Q8_0 SwiGLU fusion is a small default MoE decode cleanup. v0.294 fuses
  shared gate/up Q8_0 mat-vec plus `silu_mul`; rollback is
  `QWEN_DECODE_SHARED_SWIGLU_Q8=0`. A3B `tg128` moves about `+0.7%`, A10B
  `tg128` about `+0.2-0.3%`, and the shared-FFN phase drops `0.99 -> 0.85 ms`
  on A3B / `2.08 -> 1.82 ms` on A10B. This confirms the shelf, but it is not a
  strategic discontinuity by itself.
- F32 row-pair mat-vec is a small default MoE route cleanup. v0.295 defaults a
  cooperative two-row / four-simdgroup F32 mat-vec with
  `QWEN_MATVEC_F32_LCPP_R2=0` as rollback. A3B `tg128` moves about `+0.9-1.0%`,
  A10B about `+0.2-0.4%`, and A3B `ctx16384` stayed neutral-positive. Treat this
  as a route-execution shelf win, not proof that F32 mat-vec retuning alone can
  close the remaining decode headroom.
- Fused Q5_K routed down+weighted-sum is a larger default MoE decode bucket win.
  v0.297 routes single-token Q5_K down through the existing packed-slot fused
  kernel, removing `[topk, hidden]` routed-output traffic plus the separate
  weighted-sum dispatch. Rollback is `QWEN_DECODE_MOE_Q5_DOWN_FUSED=0`. A3B
  `tg128` moves `101.86 -> 103.8-105.1 t/s`, A10B moves `43.96 -> 44.65 t/s`,
  and A3B `ctx16384` moves `85.4 -> 86.1 t/s` in the guard.
- Fused MoE decode finalization is another small default cleanup. v0.298 fuses
  shared accumulation, `mixer_out` update, and residual add after the shared FFN
  core. Rollback is `QWEN_DECODE_MOE_FUSED_FINALIZER=0`. A3B `tg128` moves
  `103.78 -> 104.2-104.3 t/s`; A10B moves `44.64 -> 44.7-44.9 t/s`. Treat this
  as opportunistic tail-pass cleanup, not a new strategic lane.
- MoE decode phase attribution now times the production FFN apply path instead of
  stale split routed/shared/residual phases. v0.301 `ctx128` rows put MoE FFN
  apply at `2.83 ms` / `28.6%` on A3B and `6.92 ms` / `30.7%` on A10B. This keeps
  MoE FFN execution shape as the live decode branch, while GDN-front retreads stay
  demoted unless fresh roofline evidence contradicts v0.293.
- v0.302 kills a row-group-4 F32 mat-vec widening probe for the route/GDN F32
  projection shelf. The correctness-safe sidecar regressed the mixed MoE route
  bucket (`0.96 -> 1.00 ms` A3B, `1.34 -> 1.43 ms` A10B at `ctx128`). Do not infer
  from the low route active-byte percentage that another F32 mat-vec retile is high
  EV; split route into named subphases first, and keep the production
  `moe ffn apply` split as the next decode attribution gate.
- v0.303 adds the production-wave FFN split gate. At `ctx128`, A3B FFN apply is
  `2.70 ms` aggregate / `1.31 ms` gate-up wave / `1.24 ms` down wave / `0.12 ms`
  finalizer; A10B is `6.89 ms` / `3.75 ms` / `2.62 ms` / `0.14 ms`. With
  concurrent shared disabled for diagnosis, routed FFN is much larger than shared
  core (`2.31` versus `0.76 ms` A3B, `5.86` versus `1.53 ms` A10B). Treat gate/up
  and down mechanics as live; finalizer, route widening, and naive monolith fusion
  are not next.
- v0.304 deep-splits FFN apply into routed/shared subphases. A10B `ctx128` is
  routed gate/up `3.14 ms`, routed down `2.57 ms`, shared gate/up `0.83 ms`, shared
  down `0.70 ms`; A3B is `1.10/1.22/0.46/0.40 ms`. Shared FFN is not the next
  target. The next cheap falsifier is a routed gate/up no-op versus routed down
  no-op inside the production wave schedule; only write a new routed Q4 kernel if
  the corresponding no-op recovers the production wave and end-to-end guard.
- v0.305 runs those no-op oracles. Same-build `tg128` goes A3B `103.71 -> 115.00`
  with routed gate/up no-op and `113.48` with routed down no-op; A10B goes
  `44.86 -> 51.79` and `49.05`. Production-wave phase confirms the mechanics:
  A10B gate/up wave `3.80 -> 0.83 ms`, down wave `2.61 -> 0.68 ms`; A3B gate/up
  `1.23 -> 0.35 ms`, down `1.24 -> 0.42 ms`. Attack routed Q4 gate/up first, down
  second. Do not optimize shared, route, finalizer, or the prior monolith.
- v0.306 kills a narrow routed Q4 gate/up row-shape sidecar (`NR0=1, NSG=4`). It
  was directionally positive (`44.86 -> 45.38 t/s` A10B, `103.71 -> 104.63 t/s`
  A3B) but far below the `+3%` keep gate and below the `15-20%` routed gate/up
  subphase gate. Do not spend the next pass on simple `NR0/NSG` reshaping alone.
- v0.307 adds subphase roofline estimates and re-ranks exact MoE decode work.
  A10B routed gate/up is already `~435 GB/s` (`~92%` of the measured stream
  anchor), while routed down is `~324 GB/s` (`~68%`) and A3B routed down is only
  `~192 GB/s` (`~40%`). Gate/up is still the largest no-op ceiling, but exact
  work should now target routed down unless gate/up has a credible byte-reduction
  or reuse mechanism.
- v0.308 kills Q5 routed-down `NSG=4` widening. A3B down wave regressed to
  `1.29 ms` and A10B to `2.75 ms`, so fewer threadgroups/more simdgroups is not
  the mechanism. The weight-only roofline likely undercounts repeated `moe_inner`
  activation traffic; the next routed-down gate is an inner-load no-op, not a
  full staging kernel.
- v0.309 runs the Q5 down inner-load no-op. It drops down wave to `1.07 ms` A3B /
  `2.21 ms` A10B and `tg128` to `+2.0%` on both targets, so activation replay is
  real but too small for broad staging after overhead. Prefer Q5 down weight/dequant
  slimming; only try inner staging as a tiny S4-stage microproof with a strict
  `>=5%` down-wave and `>=0.8%` A10B `tg128` keep gate.
- v0.310 kills naive Q5 down packed-byte `ulong` loads. Replacing scalar `qh/q1/q2`
  reads with `ulong` loads plus shifts regressed A3B down wave to `1.32 ms` and
  A10B to `3.28 ms`. Do not pursue dequant slimming through wide integer load plus
  shift extraction without a smaller instruction-level microproof.
- v0.311 runs the Q5 down no-weight oracle. It preserves inner loads, top-k,
  weighted accumulation, stores, and production scheduling while replacing
  weight/dequant math with constants. Down wave improves to `1.00 ms` A3B /
  `2.15 ms` A10B and `tg128` moves only `+2.2%/+2.4%`. Q5 down remains real, but
  the removable cost is smeared across several mechanisms; stop drilling local Q5
  down sidecars unless a new mechanism or counter trace isolates a larger term.
  v0.487 also kills the prompt-side all-SG scatter epilogue rewrite for Q5/Q6
  grouped down: correctness is green, but A3B `pp512/pp1024` moves
  `1487.45/1678.78 -> 1484.06/1675.05 t/s`. Do not reopen scalar scatter
  epilogue rewrites without a trace showing the epilogue itself is material.
- v0.312 adds `QWEN_DECODE_TRACE_COUNTS=1` tg JSON accounting and reruns the
  current-default versus `QWEN_DECODE_MOE_CONCURRENT_GDN=0` packet. The default
  remains a real win (`+8.8%` A3B, `+6.0%` A10B in the clean no-trace repeat), but
  warmed trace rows show `~95.8-97.8%` GPU/wall and unchanged dispatch counts
  between default and rollback. The GDN-concurrent win is banked GPU overlap, not
  a fresh CPU/encoder-bubble mandate. Demote short MoE decode scheduling work
  unless a new trace isolates a larger GPU-overlap mechanism.
- v0.313 kills the first structural long-attention sidecar. A four-simdgroup
  cooperative subgroup TG passed correctness and gave an A3B synthetic `ctx4096`
  main-only win, but it was flat at synthetic `ctx16384/32768`, flat/regressive on
  A10B, and regressed full-model A3B `ctx16384` attention (`3.24 -> 3.57 ms`). Do
  not pursue subgroup packing without a new cache/counter signal; the next
  long-attention proof should be KV layout/reader structure, not more subgroup
  shape work.
- Quant breadth is now an active MoE guardrail, not a documentation afterthought.
  v0.219-v0.221 add A3B Q3_K_M, Q6_K, and Q8_0 native grouped routed coverage;
  v0.233 adds `IQ3_S/IQ3_S/IQ4_XS` and moves the local `UD-IQ4_XS` A3B file to
  `40/40` grouped MoE coverage. The v0.237 paired `pp512` re-anchor is now won
  across measured non-sharded local A3B quants: Qwen3.6 `UD-Q4_K_M` `1.05x`,
  Qwen3.5 `Q3_K_M` `1.04x`, `Q6_K` `1.02x`, `Q8_0` `1.03x`, plus the v0.233
  `UD-IQ4_XS` repeat at `1.01x`. `UD-IQ4_XS` also wins at `pp1024` (`1.11x`),
  `pp4096` (`1.12x`), Marcus real rollout (`1.09x`), and one-run `pp16384`
  (`1.19x`).
  Remaining quant risk is no longer this known A3B MoE file; it is BF16 prompt
  performance after v0.249 restored `40/40` grouped MoE coverage, unmeasured
  expert-bank combinations outside the local target set, plus dense UD files with
  `IQ2/IQ3` tensors. The v0.254 BF16 probes falsified two tempting local exits:
  per-layer command-buffer splitting did not improve wall time, and a more
  llama-like separate `NR1=32` BF16 gate/up sidecar was only `~3%` at `pp512`.
  The v0.255 wall no-op budget then re-centered the BF16 cliff on routed MoE:
  bfloat-act `pp512` is `77.00 t/s`, no routed MoE is `825.63 t/s`, no grouped
  SwiGLU is `205.76 t/s`, and no grouped down is `142.44 t/s`. The v0.256
  reduce split says weighted sum is not the missing budget: no grouped reduce is
  flat/noise while no grouped down+reduce still moves to `~92-94 t/s`. The
  v0.257 bucket-bin trace says the expert-projection loss is not uniform:
  `<8` buckets are `6-17x` slower per slot than `>=64` buckets, while hot buckets
  still dominate aggregate SwiGLU at `pp1024`. The remaining BF16 work should be
  a deeper `mul_mm_id`/layout parity probe for routed expert projections, not
  route/reduce/finalizer or another local routed-SwiGLU tile tweak. v0.258
  kills a Q5-style BF16 tiny-down `MR16/NR8` clone; v0.259 kills a hot-only
  separate gate/up `NR1=32` SwiGLU graft. Any remaining BF16 branch must change
  the routed projection graph more deeply, or BF16 should stay demoted behind
  primary-family guardrails.

Prompt-only anchors, release `qwen-bench pp`, synthetic prompts:

- Dense group-4 matrix attention is now default for `head_dim=256` shapes with
  `QWEN_PREFILL_ATTN_MATRIX_G4=0` as rollback. Current 9B rows are `813.36/820.51/
  775.82/690.68 t/s` at `pp512/1024/4096/16384` versus llama.cpp
  `814.10/804.64/693.95/678.06`; 0.8B/2B/4B `pp1024` smokes land at
  `0.98x/1.00x/1.01x` versus llama.cpp.
- 27B dense prefill now uses paired cross-engine rows for scoreboard claims. The
  v0.187 paired comparator puts qwen/lcpp at `236.53/232.41` for `pp512`,
  `222.28/214.08` for `pp1024`, `220.81/212.50` for `pp4096`, and
  `206.27/193.89` for `pp16384`. Treat older isolated `pp1024` and `pp4096`
  rows as stale methodology artifacts unless reproduced by the paired harness.
- `qwen-llm` 35B A3B MoE prompt default now includes prompt-native packed
  attention for the proven `group=8`, `head_dim=256` shape with family-specific
  `NWG=64`, packed activation at `n_pos >= 128`, A3B-sized router `E8xP32`, fused
  route+bucket from `pp128`, and grouped routed down for both `Q5_K` and `Q6_K`
  expert-down layers. Latest post-Q6 rows: `783.1 t/s` at `pp128`,
  `1000.6 t/s` at `pp256`, `1061.5 t/s` at `pp320`, `1172.1 t/s` at `pp512`,
  `1287.4 t/s` at `pp1024`, `1265.5 t/s` at `pp2048`, noisy `1198.1 t/s` at
  `pp4096`, and `674.3 t/s` for the `34,502`-token `v02_reva` full rollout.
- `qwen-llm` 122B A10B MoE prompt default now includes prompt-native group-16
  matrix attention plus the grouped Q5 gate/up layer-46 coverage fix. Warmed Q5
  gate/up rows move rollback/default from `377.53 -> 448.31 t/s` at `pp512`,
  `400.11 -> 513.66 t/s` at `pp1024`, and `440.08 -> 484.54 t/s` at `pp4096`
  (single directional long row). The default now has `48/48` grouped routed MoE
  phase coverage at `pp512`; rollback is `QWEN_PREFILL_MOE_GROUPED_Q5_GATEUP=0`.
- A10B/G16 matrix attention is now the default for the proven group-16 prompt
  shape, with `QWEN_PREFILL_ATTN_MATRIX_G16=0` as rollback. Clean `v0.141` rows
  with `build_dirty=0`, AC power, no thermal/perf warnings, and `96%` free memory
  show `pp1024` `411.86 -> 430.67 t/s` (`+4.6%`) and `pp16384`
  `289.97 -> 338.07 t/s` (`+16.6%`). Warmed dirty rows are also positive at
  `pp512/1024/4096/16384`, and
  a dirty phase trace shows `12/12` G16 matrix attention layers with attention body
  reduced from the old `~73 ms` packed bucket to `~14 ms` total matrix phases at
  `pp512`. A follow-up test-scratch fix makes bare
  `QWEN_PREFILL_ATTN_MATRIX_G16=1` A10B smoke pass without manual
  `QWEN_PREFILL_ATTN_MATRIX_MAX_POS`. A clean current-commit `pp512` trace shows
  `12/12` G16 matrix attention layers, expected routed MoE/GDN counts, and matrix
  body phases totaling `14.53 ms`. The clean repeat packet is positive at
  `pp512/1024/4096/16384`; paired same-session llama.cpp default anchors put G16
  at `0.90x/1.02x/1.08x/1.03x`. Active G16 matrix correctness is green under the
  default auto policy when the packed threshold is lowered for the smoke. Routed
  MoE is next because `pp512` remains a lcpp gap.
- Post-default A10B `pp512` no-op budget confirms that pivot: base/default G16 is
  `383.26/382.87 t/s`, no-attention-body is only `389.39/390.08`, and no-routed-
  MoE is `764.19/764.46`. Stop opening attention branches for A10B until routed
  `SwiGLU/down` has had a fresh structural pass.
- The first routed-MoE structural pass found the same class of issue as A3B Q6
  down: A10B layer `46` escaped the grouped path because gate/up are `Q5_K`.
  Grouped Q5 gate/up SwiGLU fixes coverage and is now the A10B default candidate;
  the remaining exact A10B work should attack grouped down/dequant locality, not
  route/reduce/finalizer.
- Static fast-path audit is now part of the workflow. Current target coverage is
  `A3B 40/40` grouped MoE, `A10B 48/48` grouped MoE, and dense `27B 64/64` FFN +
  `48/48` GDN + `16/16` attention. The dense/GDN/attention/lm-tail predicates
  now use every dtype with primitive mat-mat support (`F32`, `F16`, `BF16`,
  `Q2_K`, `Q3_K`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`,
  `Q6_K`, `Q8_0`, `IQ4_NL`, `IQ4_XS`), so the local 0.8B quant family is clean
  across dense FFN, GDN, attention, and lm-tail coverage. MoE grouped coverage
  includes the target A3B
  Q3/Q4/Q6/Q8 and `UD-IQ4_XS` expert-bank combinations. Q2_K/Q3_K/IQ4_NL/IQ4_XS now have
  simdgroup_matrix prompt mat-mat tiles: clean local 0.8B low-bit rows move from
  `0.15-0.26x` llama.cpp at `pp1024` to `0.96-0.98x` across `pp1024/4096`.
  The v0176 dense validation generalizes this to 2B/9B Q2/Q3/IQ4_XS and 27B Q3:
  2B is `0.97-1.00x`, 9B is `0.99-1.06x`, and 27B Q3 `pp1024` is `1.08x`
  llama.cpp. Remaining explicit coverage gaps are MoE grouped expert-bank
  variants outside target quants and unmeasured UD low-bit dense tensors. The
  v0.240 dense `IQ2_S`/`IQ3_*` matrix-kernel pass closes the measured 4B UD
  low-bit prompt cliff: local `UD-Q2_K_XL` and `UD-IQ2_M` are now parity/wins at
  `pp512/1024/4096`. v0.241 closes the measured decode side too, with clean
  `tg128` rows at `1.13-1.15x` pinned llama.cpp. v0.242 adjacent 4B breadth is
  clean: `Q3_K_M`, `IQ4_XS`, and `Q4_K_M` are parity/win at `pp512/4096`, and
  all three win at `tg128`. Older
  A3B low-bit context: clean v0.189 A3B Q3 `pp1024` paired
  evidence is `90.28` qwen versus `1392.02` llama.cpp (`0.065x`), while no-FFN
  jumps to `2858.64 t/s`. The v0.192 env-gated grouped F32/IQ4_XS candidate
  (`QWEN_PREFILL_MOE_GROUPED_F32_GATEUP=1`) moves A3B Q3 to dirty paired
  `1191.03 t/s` at `pp1024` and `1225.51 t/s` at `pp4096`, but still trails
  llama.cpp by `0.856x/0.894x`. The v0.193 native `IQ3_XXS` candidate
  (`QWEN_PREFILL_MOE_GROUPED_IQ3_GATEUP=1`) adds direct matvec and grouped-SwiGLU
  oracles, then reaches clean paired `1486.94-1490.63/1383.51-1384.73`,
  `1532.84-1533.85/1367.76-1369.89`, and `1311.75/1163.09 t/s` at
  `pp1024/4096/16384`. It is still not defaulted. The v0.195 defaultability pass
  proved the T40 blocker is GDN/Q8 sensitivity rather than native IQ3 MoE. The
  v0.196 pass then found and fixed a separate G8 matrix-attention VT-coverage bug
  that only appeared when small chunks crossed the matrix threshold mid-call. With
  blk0 `qkv+alpha` GDN matvec as a repair oracle, Q3 T128/P32 now passes and dirty
  paired rows remain above llama.cpp at `pp1024/4096/16384`
  (`1.023x/1.034x/1.108x`). The v0.197 continuation-gate pass reclassified strict
  long internal GDN/KV cosine as diagnostic by default for multi-token probes:
  Q3 and Q4 awkward-chunk controls show no argmax divergence over 64 oracle-greedy
  continuation tokens despite sub-`0.999` internal or continuation cosine. The
  v0.199 real-prompt packet keeps Q3 native-IQ3 + blk0 `qkv+alpha` above
  same-length llama.cpp anchors on Reva short, Mei medium, and Marcus long
  (`1.018x/1.014x/1.022x`), but also proves exact long greedy parity is the wrong
  release gate: Q4 default mismatches over 64 continuation tokens at T1024 too.
  The v0.201 promotion packet keeps the Q3 native-IQ3 + blk0 `qkv+alpha` branch
  positive on repeated real-prompt script rows: Reva short retained pairs are
  `1.055x/1.025x`, and Marcus long is `1.047x` after discarding block 0.
  The next low-bit MoE branch is a top-k/rank-envelope defaultability policy, then
  either default native IQ3 or build a high-accuracy blk0 GDN projection kernel if
  rank escapes are materially worse than the incumbent envelope.
  Decode sentinels also matter: Q2_K and IQ4_XS `tg128` were `0.72x/0.70x`
  before v0.177 and are now `1.09x/1.22x`. Q3_K_M now has a native row-reuse
  fast mat-vec kernel and moves from `0.94x` to `1.31x` on 0.8B, with 2B/9B/27B
  one-run sentinels at `1.10x/1.12x/1.13x`. IQ4_NL also has a native row-reuse
  fast mat-vec kernel and moves from the prior `~0.97x` residual to `1.25x` on
  the only local IQ4_NL file. Clean post-commit 0.8B decode anchors now put
  Q2/Q3/IQ4_NL/IQ4_XS/Q4_K_M at `1.36x/1.30x/1.28x/1.26x/1.26x` llama.cpp.
  The measured dense low-bit local decode family is now a win; stop harvesting
  this lane until a primary re-anchor or larger quant file exposes a real miss.
- A10B routed MoE is no longer the active top bet after the warmed re-anchor. A
  default-warmup `pp512` phase trace has qwen timed-pass
  `routed_swiglu+routed_down = 600.52 ms` versus llama.cpp profile
  `ffn_moe_gate+up+GLU+down = 654.98 ms`, and current end-to-end anchors are
  qwen/lcpp `446/440` at `pp512`, `488/436` at `pp1024`, `481/401` at `pp4096`,
  and `394/354` at `pp16384`. Keep A10B MoE in monitoring unless a fresh matched
  trace shows a real warmed routed-tail deficit or coverage drops below `48/48`.
- Dense 27B prefill is now paired-won across the measured synthetic shapes after
  the Q4_K `N=64` prompt mat-mat tile and the v0.187 comparator pass. Rollback is
  `QWEN_MATMAT_Q4_K_N64=0`. The key lesson is methodological: stale cold llama
  anchors and unpaired qwen drift made `pp1024/4096` look worse than they are.
  Do not reopen dense 27B short/medium work without a paired residual.
- The first dense FFN/GDN pass is phase-positive but not scoreboard-complete by
  itself.
  `QWEN_PREFILL_TRACE_FFN_SUBPHASES=1` plus timed-only summaries showed the FFN
  residual was mat-mat throughput, not SwiGLU epilogue or chunk policy. Adding
  llama.cpp-style full-unroll pragmas to Q4/Q5/Q6/Q8 mat-mat kernels moves 27B
  timed `pp4096` gate/up/down from `4063/4068/4199 ms` to `3959/3964/4090 ms`,
  near the lcpp `3878/3932/4099 ms` buckets; GDN QKV/Z/back also drop `2-4%`.
  Post-commit clean rows on AC power are 27B `pp512=234.38`, `pp1024=226.63`,
  `pp4096=207.34`, `pp16384=193.41`, plus A10B `pp1024=499.83` and A3B
  `pp1024=1584.48` smokes.
  A same-principle GDN recurrence-loop unroll was falsified and removed
  (`gdn_step` `593.40 -> 597.17 ms` at `pp4096`).
  A timed-pass v0.157 differential makes attention body the next active branch:
  at `pp4096`, FFN and GDN projections are parity-or-faster while attention body
  is `~115 ms` slower than llama.cpp and GDN step is `~82 ms` slower; at
  `pp16384`, attention body is `~0.85 s` slower, GDN step `~0.36 s`, and FFN
  `~1.0 s`. No-op ceilings are large (`pp16384` attention body `194.35 ->
  217.23 t/s`, GDN body `194.35 -> 215.10`), but the first generic attention
  loop-unroll probe was negative and removed.
- A first exact attention-body cleanup is now default for the 27B G6 matrix path:
  causal-tail KQ/KQV tile skip with `QWEN_PREFILL_ATTN_MATRIX_CAUSAL_SKIP=0`
  rollback. It improves 27B `pp4096` by about `0.5%` and `pp16384` by about
  `0.55%` in randomized A/B, with clean post-commit anchors at `pp4096=208.06`
  and `pp16384=194.80`; it is neutral/noisy at `pp512/1024` and passes the default
  plus long-prefix 27B correctness gates. This does not close the llama.cpp gap;
  it just removes avoidable future-tile work from the current matrix body.
- Current same-shape A3B rows against recent `llama.cpp` anchors changed sharply
  after the Q6-down grouped fix and fresh same-session lcpp anchors: `pp320` is
  now `1061.45 / 1174.57 t/s` (`0.90x`), `pp512` is `1172.07 / 1347.79 t/s`
  (`0.87x`), `pp1024` is `1287.43 / 1345.07 t/s` (`0.96x`), `pp4096` is
  `1198.11 / 1259.21 t/s` (`0.95x`), and `pp34502` is `674.28 / 865.50 t/s`
  (`0.78x`). `pp16384` needs a post-Q6 rerun. `llama.cpp` also declines at true
  long context; medium A3B is now close, while true-long remains the largest A3B
  prefill gap.
- The existing env-only A3B/group-8 matrix-attention sidecar composes with the Q6
  fix and reaches/surpasses those lcpp anchors in spot rows: `1190.45 t/s` at
  `pp320`, `1340.82 t/s` at `pp512`, `1484.98 t/s` at `pp1024`, `1434.39 t/s` at
  `pp4096`, `878.72 t/s` at synthetic `pp34502`, and `870.94 t/s` on the real
  `v02_reva` `34,502`-token rollout. It remains env-only because max-pos scratch
  policy and matrix correctness tolerance still need productionization.
- First max-pos productionization step is done: `qwen-bench pp`, `pp-wait`, and
  packed `decode` prefill now size matrix scratch from the actual prompt length
  when `QWEN_PREFILL_ATTN_MATRIX_G8=1`, so user-facing bench paths no longer need
  `QWEN_PREFILL_ATTN_MATRIX_MAX_POS`. Auto-scratch spot rows are stronger:
  `1237.87/1390.93/1561.06/1492.39/928.62 t/s` at
  `pp320/512/1024/4096/34502`.
- The expanded clean family sweep at
  `docs/bench/2026-05-24-1331-matrix-pp4k16k-family/` ran
  `QWEN_PREFILL_ATTN_MATRIX_G8=1`, `runs=1`, and
  `pp128/512/1024/4096/16384` plus `tg32/tg128`. A3B now matches/beats lcpp at
  every swept prompt size in that candidate branch: `879/769` (`1.14x`) at
  `pp128`, `1387/1392` (`1.00x`) at `pp512`, `1558/1388` (`1.12x`) at `pp1024`,
  `1492/1352` (`1.10x`) at `pp4096`, and `1212/1107` (`1.09x`) at `pp16384`.
  Do not generalize this to the whole family: dense 27B is still only `0.67x` at
  `pp16384`, and A10B is `0.83x/0.92x/1.00x/0.86x` at
  `pp512/1024/4096/16384`.
- The same family sweep makes prompt chunk policy a first-class hypothesis: every
  qwen row drops from `pp4096` to `pp16384` under the current default
  `prefill_chunk=1024`. Some decline is not itself a bug because llama.cpp also
  falls at 16K in this run, but qwen's dense/A10B slope is now a likely whole-
  family blocker.
- A targeted clean chunk sweep falsifies chunk size as the dense 16K cure but
  keeps it live for MoE. Dense 27B `pp16384` is flat/slightly worse at chunk
  `2048` (`125.09 -> 124.52 t/s` repeated), while A3B `pp16384` improves at
  chunk `2048` (`1114.61 -> 1148.52 t/s`) and A10B improves at chunk `4096` both
  at `pp4096` (`357.87 -> 378.40 t/s`) and `pp16384` (`288.99 -> 306.46 t/s`).
  Treat `2048` as the conservative cross-MoE default-cap candidate; treat `4096`
  as an A10B-specific candidate until A3B/real-rollout gates say otherwise.
- The clean repeated A3B matrix promotion gate at
  `docs/bench/2026-05-24-1934-35B-A3B-matrix-promotion-repeat-family/` keeps the
  branch above lcpp at `pp128` (`1.10x`), `pp1024` (`1.11x`), `pp4096` (`1.10x`),
  and `pp16384` (`1.07x`), with `pp512` at parity (`0.99x`) and decode `tg32/tg128`
  both `1.13x`. A follow-up chunk-2048 interaction gate shows why the chunk cap
  must be prompt-length gated: `2048` helps A3B at `pp4096` (`1.06x` over default)
  and barely at `pp16384` (`1.01x`), but regresses `pp128` and `pp512`.
- The group-8 matrix sidecar has now been generalized to runtime group-shape args
  and an env-only 27B dense group-6 gate (`QWEN_PREFILL_ATTN_MATRIX_G6=1`). The
  clean `v0.120` repeat gate in
  `docs/bench/2026-05-24-dense-g6-clean-repeat-v0120/` keeps the win across every
  swept prompt size: `pp128` `188.33 -> 198.42 t/s` (`1.05x`), `pp512`
  `195.85 -> 210.99` (`1.08x`), `pp1024` `175.66 -> 188.60` (`1.07x`), `pp4096`
  `151.87 -> 185.85` (`1.22x`), and `pp16384` `125.21 -> 173.84` (`1.39x`). This
  is now a dense promotion candidate, not just a dirty spike, but long-prefix G6
  correctness coverage is still thin.
- The `v0.122` dense family gate in
  `docs/bench/2026-05-25-0246-27B-matrix-g6-v0122-family/` adds that long-prefix
  G6 correctness coverage and compares against fresh lcpp anchors. Dense 27B is
  now near parity but not beaten: `pp128` `198/213` (`0.93x`), `pp512` `211/222`
  (`0.95x`), `pp1024` `202/205` (`0.99x`), `pp4096` `186/198` (`0.94x`), and
  `pp16384` `176/188` (`0.93x`). Decode remains won at `tg32/tg128` (`1.13x` and
  `1.12x`). Matrix-G6 is therefore a real default candidate but not the final
  dense answer.
- With matrix-G6 enabled, fresh no-op rows move the dense residual priority to
  FFN: at `pp4096`, no-FFN is `180.50 -> 541.59 t/s` while no-GDN/no-attn are only
  `193.30/193.93`; at `pp16384`, no-FFN is `173.12 -> 421.55` while no-GDN/no-attn
  are `192.37/192.25`. A llama-like threadgroup pointer-store spelling for
  Q4/Q5/Q6/Q8 mat-mat is correctness-green and gives only small dirty-spike rows
  (`188.77 t/s` at `pp4096`, `176.13 t/s` at `pp16384`), so the high-EV dense work
  remains deeper FFN mat-mat/layout/fusion evidence, not more local attention.
- A high-N dense fused-Q4 SwiGLU spike (`QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=1`)
  validates the mechanism but not default readiness: with matrix-G6 on, dirty
  rows are flat/slightly down at `pp512` (`214.86 -> 214.45`) and noisy down at
  `pp1024` (`196.97 -> 193.45`), while `pp4096` improves in one noisy pair
  (`181.08 -> 191.67`) and `pp16384` improves slightly (`177.57 -> 179.25`). Keep
  it env-only; the remaining dense gap still needs a bigger FFN mat-mat/layout win.
- Clean `v0.125` repeats confirm the long-row fused-Q4 SwiGLU branch but also its
  limited ceiling: `pp4096` `182.05 -> 191.62 t/s` (`1.05x`) and `pp16384`
  `177.22 -> 179.24` (`1.01x`). It is a useful long-prompt candidate, not enough
  to crack lcpp, and the `pp512/pp1024` dirty rows block broad defaulting.
- A llama-parity smem cleanup for Q4/Q5/Q6/Q8 mat-mat is worth keeping but not
  over-reading. Dirty microbench rows are flat at `N=512`, clearly positive at
  `N=1024` (Q4 gate/up `19.812/17.617 -> 14.567/14.853 ms`, Q6 down
  `19.733 -> 16.119 ms`), and modest at `N=4096`, but a clean v0.127 matrix-G6
  end-to-end A/B does not prove a default win (`pp4096` new/legacy/new =
  `189.00/204.65/204.51`, `pp16384` `190.83/191.68/188.44`). Keep legacy 8192B
  requests as default; use `QWEN_MATMAT_QK_LLAMA_SMEM=1` only as an opt-in
  diagnostic while hunting larger FFN/GDN execution gaps.
- A follow-up F16-inner/Q6-F16-source dense FFN spike is falsified. Correctness was
  clean, but cooled rows show no stable end-to-end win (`pp4096` F16-inner
  `202.33` vs fused-Q4 `201.66`, `pp16384` `180.59` vs fused-Q4 `182.00`) and a
  direct Q6_K mat-mat microbench shows F16 source is slower (`ffn_down` `61.604 ms`
  F32-source vs `64.000 ms` F16-source at N=4096). Do not carry or retune this
  branch unless future profiling proves source-read bandwidth has become the wall.
- The first drift-controlled `v0.129` dense FFN rows keep both env candidates out
  of the default path. With matrix-G6/G8 on, warmed `pp4096` rows are only
  `g6=208.96`, `g6-smem=209.73`, `g6-fused=210.98`, and
  `g6-fused-smem=211.09`; `pp16384` is directionally inconsistent
  (`g6=193.46/181.14`, `g6-fused=189.70/193.60`). Treat the stable signal as
  `~1%` and below the default gate. The next dense move is phase-local FFN/GDN
  evidence, preferably paired in-process, not another total-throughput default
  decision on a sub-noise row.
- The same-process fused-SwiGLU A/B harness now exists, and it falsifies broad
  defaulting despite one exciting long-context outlier. Fused loses paired
  `pp4096` and `pp8192`; `pp16384-a` looks large-positive but the immediate
  `pp16384-b` repeat is flat. Keep the branch as an env-only long-context clue,
  not a default candidate, until a repeated same-process gate and phase-local
  FFN mechanism agree.
- A serialized qwen-vs-llama.cpp dense `pp4096` differential moves the next dense
  target away from standalone SwiGLU fusion. Qwen's split FFN buckets are only a
  few percent slower (`gate/up/SwiGLU` `8408` vs `8106 ms`, `down/resid` `4310`
  vs `4200 ms`), while GDN front projections are the larger local delta (`3613`
  vs `2998 ms`) and matrix attention body still has a smaller residual delta.
  Next dense work should inspect GDN front lowering and lcpp op shapes before
  another FFN fusion branch.
- That GDN-front inspection found one real dispatch-class mismatch: dense GDN
  `beta_proj` / `alpha_proj` are F32 skinny projections that benefit from the
  router-style E8xP32 kernel. Default-on `QWEN_PREFILL_GDN_SKINNY_E8P32` drops
  traced `gdn_beta_alpha` from `672.45 -> 105.96 ms` at 27B `pp4096` and improves
  repeated 27B matrix-G6/G8 rows by about `+2-4%` at `pp512/1024/8192/16384`
  (`pp4096` warmed is only `~+0.8%`). Correctness is green on 0.8B, active 27B
  prefill, 27B matrix-prefix, and A3B prefill gates. The promotion gate also found
  positive A3B `pp128/1024` and matrix `pp4096` canaries. Roll back with
  `QWEN_PREFILL_GDN_SKINNY_E8P32=0` if a future F32 GDN shape regresses.
- A 27B `pp16384` combined no-op budget confirms the dense gap is not only
  attention: baseline `127.57 t/s`, no-FFN `208.71`, no-attn `193.48`,
  no-FFN+no-attn `580.91`, and no-FFN+no-attn+no-GDN `845.88`. Keep dense FFN/GDN
  mat-mat work queued after the G6 matrix clean gate, not instead of it.
- `llama-bench -fa 0` is not auto for the bench tool: it disables flash attention.
  A3B `llama.cpp` `-fa 0` and `-fa 1` are flat at `pp1024` and `-fa 1` is slightly
  slower at `pp16384`, so the current A3B long target is the non-flash
  `KQ -> softmax -> KQV` Metal path and its cache/layout implementation details.
- An env-only A3B/group-8 matrix-attention sidecar
  (`QWEN_PREFILL_ATTN_MATRIX_G8=1`) became a real long-context branch after V_T
  writes moved to fused cache-fill time and the KQ/KQV B tile started using
  lcpp-like vector loads: spot rows are about `1042 t/s` at `pp4096`, `997 t/s`
  at `pp8192`, noisy `~864-904 t/s` at `pp16384`, and `725.7 t/s` at `pp34502`
  with chunk `1024`. A true-long chunk probe puts `pp34502` chunk `2048` at
  `736.9 t/s` and chunk `4096` at `718.2 t/s`. It is still not defaulted because
  allocation is manual via `QWEN_PREFILL_ATTN_MATRIX_MAX_POS` and KQV uses the
  looser half-probability correctness tolerance.
- Fresh `llama-bench` A3B anchors on the current local build are `1174.6 t/s` at
  `pp320`, `1347.8 t/s` at `pp512`, `1345.1 t/s` at `pp1024`, `1259.2 t/s` at
  `pp4096`, and `865.5 t/s` at `pp34502` (`-fa 0`, `has tensor = false`). A10B
  `pp320` still needs a fresh same-session lcpp rerun.
- prior repeated-prompt `qwen-llm` 27B dense packed prefill: `~205.4-205.9 t/s`
- current `llama.cpp` bounded `llama-cli -st` baseline: `~206.7 t/s` prompt,
  `~22.5 t/s` generation

Recent confirmed wins:

- Dense GDN front-projection overlap is now production-wired for logits and GPU
  argmax decode. The old bench-only path becomes the dense default in v0.340 and
  lifts `tg128` across measured dense Q4 models by `+2.4-3.8%`; rollback is
  `QWEN_DECODE_DENSE_CONCURRENT_GDN=0`. This banks scheduling overlap, not a new
  local Q8 mat-vec lane.
- A3B had a silent optimized-path escape: three late routed down-expert layers
  (`blk.34`, `blk.38`, `blk.39`) are `Q6_K`, while grouped routed prefill only
  accepted `Q5_K` down. Adding grouped Q6_K down and dtype dispatch moves trace
  coverage from `37/40` to `40/40` grouped routed layers and lifts A3B `pp320`
  `~830 -> 1061.45 t/s`, `pp512` `824.46 -> 1172.07 t/s`, `pp1024`
  `910.43 -> 1287.43 t/s`, and same-fixture `34,502`-token real rollout
  `587.60 -> 674.28 t/s`. This is now the canonical
  example for why fast-path coverage must be asserted, not inferred from logits.
- Prompt-native packed MoE attention is now a production default for the proven
  long/medium prompt shapes, not an experiment. The important details are now
  known and banked:
  - family-specific packed `NWG` (`g8=64`, `g16=32`)
  - family-specific packed activation thresholds (`g8 >= 128`, `g16 >= 320`)
  - per-layer packed-vs-old oracle green at the first newly activated prompt
    sizes (`pp128/pp320` for A3B, `pp320/pp512` for A10B) and at active
    long-context chunk shapes.
- A3B-sized route-logits `E8xP32` and fused route+bucket now activate from
  `pp128`; the old `512` floors left cheap medium/short-prompt wins on the table.
  A10B stays conservative at `512` for route work until cooled promotion sweeps
  resolve the mixed signal there.
- The medium-prompt board moved substantially once the packed-attention threshold
  dropped from `4096` into the medium-prompt regime and the A3B route threshold
  followed: A3B `pp128/pp320/pp512/pp1024` now lands around
  `~655 / ~830 / ~899 / ~952-966 t/s`, and A10B
  `pp320/pp512/pp1024` around `~276 / ~355 / ~418 t/s`.
- The true-long A3B board is now separated from medium-prompt work: qwen synthetic
  and real `34.5k` prompts agree (`596.88` vs `587.60 t/s`), and no-op attribution
  says attention-body cost is the long-context lever (`pp16384` `767.64 ->
  1101.72 t/s` with attention body skipped; routed-MoE skip only reaches
  `913.73 t/s`).
- The A3B matrix-attention sidecar found a real lcpp-like mechanism: V must be
  written in KQV-ready transposed layout at cache-fill time. Fused V_T scatter
  turns the prior long regression into a `~10-18%` spot win from `pp1024` through
  `pp34502`, but the remaining gap is now KQ/KQV/score traffic and mature
  `mul_mm_f16_f32` behavior.
- The next lcpp-like `mul_mm_f16_f32` detail, vector-loading the F32 B tile into
  `half2x4`, is now in the matrix sidecar. It improves the current true-long
  `pp34502` row to `725.7 t/s` at chunk `1024` and `736.9 t/s` at chunk `2048`,
  but `pp16384` is still noisy and chunk `4096` loses. Treat chunk `2048` as the
  current matrix-long candidate, not a default.
- After the Q6 grouped-down fix, that same matrix sidecar becomes the first A3B
  prefill branch to crack lcpp spot rows from medium through true-long. The next
  work is productionization and repeated promotion gates, not more proof that the
  mechanism exists.
- The matrix sidecar no longer needs manual max-pos env sizing in `qwen-bench`
  prompt paths; remaining promotion blockers are cooled repeatability, default
  activation policy, scratch budget accounting, and the correctness tolerance
  decision.
- Post-matrix `pp4096` no-op budgeting changes the next-priority read: with
  matrix attention held fixed, attention-body skip is only a `~3-7%` lever
  (`929/972 -> 997 t/s`), while routed-MoE skip reaches `1270 t/s` and broad
  FFN skip reaches `2498 t/s`. Direct KQV stores, vectorized KQV final copies,
  and F16 probability scratch were all falsified as local attention keepers.
  The next exact sprint should target routed FFN/MoE structure again.

- Grouped MoE routed prefill had a real correctness bug: grouped `Q4_K` SwiGLU
  used `u32::MAX` as an open-ended expert-count sentinel while the Metal kernel
  cast it to signed `int`, turning the bound into `-1` and early-returning active
  experts. After fixing the sentinel, A10B smoke / boundary and A3B
  prefill-vs-single gates recovered, the fused route+bucket oracle became clean,
  and the routed MoE prompt story had to be re-based.
- That corrected re-baseline changes the active MoE path ranking:
  - `route_bucket` itself is small,
  - routed `grouped_swiglu` is still the largest MoE prompt bucket,
  - router logits are the next meaningful routed cost.
- The first post-fix routed-compute attack that converts end-to-end is a GPU-owned
  hot-expert grouped-Q4 split over the existing per-expert `counts/ids` ledger.
  Combined with fused route+bucket and allowlisted at `chunk_p >= 512`, it moves
  A10B from about `297.0 -> 302.9 t/s` at `pp512` and `299.6 -> 309.3 t/s` at
  `pp1024`, and A3B from about `736.3 -> 744.9 t/s` at `pp512` and
  `754.7 -> 776.7 t/s` at `pp1024`.
- A router-only `F32` `E8xP32` kernel now removes most of the remaining router
  logits cost on the proven MoE prompt shapes. Corrected A10B `pp512`
  `route_logits` falls from `5.95 ms` to `0.47 ms`, and A10B `pp1024`
  `route_logits` is `1.03 ms`. Allowlisted with the existing composed MoE prompt
  path at `chunk_p >= 512`, it moves A10B from about `302.9 -> 315.5 t/s` at
  `pp512` and `309.3 -> 324.7 t/s` at `pp1024`, and A3B from about
  `744.9 -> 797.6 t/s` at `pp512` and `776.7 -> 815.1 t/s` at `pp1024`.
- The clean post-`v0.100` family sweep materially updates the MoE prompt picture:
  A10B is now about `0.72x` at `pp512` and `0.76x` at `pp1024`, while A3B is
  still much farther behind at about `0.55x` / `0.58x`. That keeps MoE prefill as
  the main scoreboard gap even though the current exact grouped path is much
  better than the old baseline.
- Post-`v0.100` exact local grouped-MoE search-space elimination is now large:
  `F16` grouped inner default, cold `n8`, hot `th32` rollout, hot grouped-down
  atomic accumulate, persistent/locality queues, hot `32x32`, active hot tile
  lists, and a resident paired gate/up mirror all failed to produce a default-
  worthy win on this hardware / repo shape. Treat the exact local
  `grouped_swiglu` variant family as plateaued until a new mechanism is found.
- Grouped routed `inner/out` zero-fill is now correctness-covered as a real but
  small cleanup lever (`~0.5-0.8%` on the current prompt guardrails), not a main
  roadmap item.
- A bounded exact MoE prompt concurrency branch is now real: overlapping the live
  grouped routed tail with the live shared FFN at `chunk_p >= 512` preserves the
  current correctness matrix and converts to about `1-3%` end-to-end on `pp512`,
  but fades by `pp1024`. This is the strongest near-term exact production branch,
  not the main structural MoE answer.
- Real-rollout and matched-token synthetic ladders now agree on a stronger story:
  A3B long-prompt collapse is primarily an attention/context-growth problem, not a
  prompt-template artifact and not mostly routed MoE compute. Attention no-op on
  long A3B prompts lifts throughput by roughly `4-5x`, while routed MoE no-op is
  much smaller and mostly constant per token.

- MoE decode now has a real GDN-side concurrency win. Reusing the dense
  concurrent-GDN front-projection split inside MoE decode and making it the repo
  default with `QWEN_DECODE_MOE_CONCURRENT_GDN=0` as a rollback path moves A3B
  from `~74.0/73.6 t/s` to `~77.8/78.0 t/s` at `tg32/tg128`, and A10B from
  `~32.4/32.5 t/s` to `~35.1/34.9 t/s`. At `ctx=4096`, decode-window also moves
  A3B `66.6 -> 71.5 t/s` and A10B `31.4 -> 34.0 t/s`, with GPU time improving on
  every measured row.
- A10B `pp128` cold variance is now explained as expert-bank first-touch, not as
  a steady-state runtime miss. Bench-only `QWEN_PP_WARM_MOE_BANKS=1` and
  `QWEN_PP_RESIDENCY_SET=1` collapse the cold outliers while leaving steady-state
  `pp320/pp512` flat, so treat them as benchmark methodology knobs rather than a
  top runtime roadmap item. v0.261 re-confirms this on the current stack:
  default paired `pp128` loses from cold outliers, while warm-bank paired `pp128`
  wins pinned llama.cpp at `1.108x`.

- `qwen-bench pp` now exposes a phase-matched prompt-only harness and lowering
  summary for dense/MoE prompt work. It confirmed the dense `pp320` miss is real
  (`~710/824 t/s` on 9B, `~211.9/240.9 t/s` on 27B) and uncovered the largest
  MoE issue: Q8 mixer projections were falling through to decode-shaped
  per-token paths.
- Enabling `Q8_0` packed mat-mat eligibility for MoE GDN/attention projections
  moves A3B pp320 from `~96 t/s` to `~194-198 t/s` and A10B pp320 from
  `~37.6 t/s` to `~85.1 t/s`, while dense 9B/27B guardrails stay flat.
- Activating the packed routed expert tail after fixing its production dtype gate
  and Q4_K byte layout moves A3B pp320 to `~261.5 t/s` and A10B pp320 to
  `~106.8 t/s`. The win holds across prompt lengths: A3B `pp64..pp1024` stays
  `~249-261 t/s` vs fallback `~188-197 t/s`; A10B `pp128..pp512` stays
  `~105-107 t/s` vs fallback `~84-85 t/s`.
- Batching the shared-expert branch plus rowwise shared gate/residual is another
  major MoE prompt win: A3B pp320 moves to `~399.8 t/s` and A10B pp320 to
  `~148.9 t/s`; default chunk-128 pp320 is `~375.9 t/s` on A3B and
  `~150.4 t/s` on A10B. Dense 9B/27B guardrails remain flat.
- A fully GPU-owned grouped expert-major routed backend now lands as the current
  production-resolution MoE prompt path for the proven `Q4_K/Q4_K/Q5_K`
  envelope. Default chunk-128 pp320 rises to `~556.4 t/s` on A3B and
  `~218.1 t/s` on A10B; chunk320 reaches `~710.4 t/s` / `~278.7 t/s`. Dense
  9B/27B guardrails stay flat.
- Packed attention-body cleanup now batches consecutive-position RoPE for Q/K and
  scatters the whole chunk's K/V rows into the cache in one dispatch before the
  per-token attention loop. On the repeated 320-token 27B prompt, packed prefill
  moves from ~201.5-202.3 t/s to `~205.4-205.9 t/s` and trims total wall from
  ~1582-1588 ms to ~1554-1558 ms.
- Fresh local `llama-cli -st` checks now show dense user-facing parity is real:
  `llama.cpp` is about `206.7 t/s` prompt / `22.5 t/s` generation on the same
  repeated prompt, while `qwen-llm` is already `~205.6 t/s` on prompt-only runs.
- Fresh local `llama-bench` still says the harder pure-prompt target is much
  higher: `pp320 ~240.9 t/s`. That keeps prompt-only parity, not CLI parity, as
  the main scoreboard target.
- Packed dense GDN prep is now over the whole prompt chunk. The old packed GDN
  prep loop used `P` launches of `ssm_conv_silu`, two L2 norms, and three
  scatters before the packed recurrence; the new path replaces that with one
  packed prep kernel plus two in-place batched L2 norms. On the repeated
  320-token 27B prompt, packed prefill moves from ~186.3 t/s to
  `~201.5-202.3 t/s` and trims total wall from ~1718 ms to ~1582-1588 ms.
- The new `QWEN_PREFILL_GDN_SPLIT` diagnostic confirmed the old packed GDN prep
  loop was the dominant GDN sub-bucket before the rewrite: about `~146 ms` wall
  / `~131 ms` GPU on the repeated prompt, versus only `~44 ms` for the packed
  recurrence itself.
- Production-shape packed-prefill profiling is now live in `qwen-bench decode`:
  prompt-only runs report total prefill GPU ms, and profiling no-op flags can
  remove dense FFN, GDN body, or attention body inside the real packed prefill
  graph.
- Packed dense GDN `rmsnorm_gated` is now batched over the whole prompt chunk
  instead of one dispatch per token. On the repeated 320-token 27B prompt, this
  moves packed prefill from ~183.1 t/s to ~186.2-186.5 t/s and trims GPU total
  from ~1717 ms to ~1690 ms.
- Latest same-prompt dense read is now `~205.6 t/s` on 27B, which cuts the fresh
  `llama-cli -st` prompt gap down to roughly three percent even though the harder
  `llama-bench` pure-prompt gap still remains.
- Production-shape dense prompt no-op profiling says the remaining packed-prefill
  cost is real GPU work. After the packed attention-body cleanup, the old
  attention-body no-op delta falls from about `~162 ms` wall / `~156 ms` GPU to
  roughly `~120 ms` wall / `~118 ms` GPU, leaving the remaining GDN tail /
  out-proj path as the clearest non-FFN prompt target. That keeps isolated FFN
  mat-mat kernels exonerated as the hidden prompt mystery.
- Prompt-prefill scratch now skips the unused `[P, V]` logits pack on no-spec
  prompt paths, removing a large dead allocation from timed prefill and nudging
  dense 27B prompt throughput to ~173.3 t/s.
- Packed dense `gdn_step_decay` over prompt tokens is now live and materially
  improves dense prompt processing: the same-prompt 27B plateau rises from
  ~165.0 t/s to ~172.9 t/s while decode stays unchanged.
- Dense packed GDN `alpha/beta` batching was a major prompt win: same-prompt 27B
  prefill rose from ~141.9 t/s plateau to ~165.0 t/s plateau, with decode
  unchanged.
- Dense packed-prefill GDN-tail attribution is now in place and points at the
  true `step_decay` recurrence as the sharpest next dense tail target.
- Dense packed prefill chunk tuning was a major win: on the same 321-token 27B
  prompt, moving from inherited `P=16` to dense default `P=256` improved prompt
  throughput from ~77.9 t/s to ~140.7 t/s with decode unchanged; one-chunk
  saturation is ~141.9 t/s once `P >= 321`.
- Dense packed prefill is now the default no-spec path in `qwen-bench decode`
  for dense models; on a 321-token 27B prompt it improved prefill from 24.3 t/s
  to 78.1 t/s (~3.22x) with decode unchanged.
- MoE packed prefill stage 1 is now live in `qwen-bench decode`; on a 321-token
  prompt it improves prefill from 74.8 -> 91.3 t/s on 35B A3B and 32.1 -> 36.6
  t/s on 122B A10B, with decode essentially unchanged.
- MoE packed prefill chunk tuning also matters: tuned default `P=128` lifts the
  same prompt to ~95.3 t/s on 35B A3B and ~37.6 t/s on 122B A10B.
- No-spec GPU argmax decode path landed. Dense decode is neutral within noise;
  MoE decode improves modestly by avoiding full logits readback (~1.1-1.5% on
  A3B / 122B in current 64-token runs).
- KV-Q8 remains a negative result on M4 for the existing v4 main-kernel
  structure. v0.437 extends the oracle to MoE/group8 and changes the reader to
  mirror the F16 vector shape, but attention is still slower.
- Attention v4 now supports `group=4`, unlocking the small dense family
  (0.8B / 2B / 4B / 9B) as real long-context canaries instead of failing back to
  the old threadgroup-memory-limited attention path. The local 9B sweep now runs
  cleanly through 32K: `64.0 t/s` at 4K, `59.5 t/s` at 16K, `53.8 t/s` at 32K.
- Attach-mode decode tracing is now practical via `qwen-bench decode-window`, and
  `scripts/profile/trace-metal.py` gives a compact Metal timeline summary
  without hand-written one-off parsers.
- Llama-style Q8_0 decode mat-vec is a large default MoE win with rollback
  `QWEN_MATVEC_Q8_0_LCPP=0`. A10B moves from `35.77 -> 42.90 t/s` at `tg64`,
  default `tg128` is `42.80/42.81` versus rollback `35.88`, and `tg256` moves
  `34.74 -> 42.45`; A3B `tg128` moves `84.22 -> 98.04`. Dense 27B `tg128` was
  slightly negative/noisy under forced env (`23.40 -> 23.13`), but the local 27B
  Q4 file has no Q8_0 tensors, so treat that as a repeat guardrail rather than a
  blocker.
- Group8 attention tile2 default for A3B long-context decode. At `ctx16384`, the
  default-vs-rollback phase packet moves total phase time `15.12 -> 13.66 ms` and
  attention `4.80 -> 3.60 ms`; context sweep throughput moves
  `72.8 -> 82.2 t/s` at 16K with neutral short-context behavior.
- Shared Q8_0 SwiGLU default for MoE decode, with `QWEN_DECODE_SHARED_SWIGLU_Q8=0`
  as rollback. This fuses shared expert gate/up plus `silu_mul`, moving A3B
  `tg128` roughly `98.84 -> 99.5-99.6 t/s` and A10B `42.84 -> 42.94-42.97 t/s`.
- F32 row-pair decode mat-vec default, with `QWEN_MATVEC_F32_LCPP_R2=0` as
  rollback. The cooperative two-row / four-simdgroup shape moves A3B `tg128`
  roughly `99.64 -> 100.5-100.7 t/s` and A10B `42.87 -> 42.97-43.05 t/s`.
- Fused Q5_K down+weighted-sum default for MoE decode, with
  `QWEN_DECODE_MOE_Q5_DOWN_FUSED=0` as rollback. It moves A3B `tg128` roughly
  `101.86 -> 103.8-105.1 t/s` and A10B `43.96 -> 44.65 t/s`.
- Group16 attention tile4 default for 122B long context.
- Group6 dense attention `NWG=64` at `n_pos >= 4096`.
- `QWEN_ATTN_V4_NWG`, `QWEN_ATTN_V4_TILE_C`, `QWEN_ATTN_V4_G8_TILE`, and
  `QWEN_ATTN_V4_G16_TILE` A/B knobs.
- Production-style `NWG=64` correctness coverage for attention v4.
- MoE intra-block profiler: 122B block ~0.594 ms, with mixer prep largest.

Recent measured negatives:

- v0.437 falsifies same-layout Q8_0 KV for MoE/group8 `attn_v4`. The Q8x4 reader
  is correctness-clean across group6/group8 main and group8 tile2/tile4, but A3B
  `attn-intra` regresses at both `ctx8192` (main `0.1113 -> 0.1221 ms`) and
  `ctx32768` (main `0.1767 -> 0.2213 ms`). Do not reopen scalar Q8, forced-tile
  Q8, group-tile/NWG sweeps, or same-layout Q8x4 variants without a new
  capture/counter signal and clean 8K+32K `attn-intra` wins.
- v0.341 kills the naive MoE FFN expert-pipeline proof. Splitting top-k routed
  experts into two groups and overlapping group-A down with group-B gate/up was
  exact on the A3B serial-vs-pipeline smoke, but regressed `tg128`: A3B
  `103.90 -> 100.64 t/s`, A10B `45.01 -> 44.09 t/s`. Do not revive this split
  without a counter signal proving real overlap and a design that avoids the
  extra group-B accumulation pass.
- A3B long-context attention-v4 exposed knobs are exhausted after the v0.293 tile2
  default. At `ctx16384`, `QWEN_ATTN_V4_NWG=32` regressed attention sharply
  (`3.39 -> 4.89 ms`), `QWEN_ATTN_V4_TILE_C=32` was worse/flat, and
  `QWEN_ATTN_V4_TILE_C=128` was phase-interesting but end-to-end flat/noise
  (`83.2/84.0/83.9 t/s` default/C128/default). Reopen A3B long attention only
  with a deeper body/KV-traffic mechanism, not another knob flip.
- MoE Q8-KV subgroup attention is falsified in the straightforward reader form.
  The sidecar was numerically close to F16 KV (`cos=0.999992`) but regressed
  long-context phase profiles: A3B `ctx16384` attention `3.32 -> 4.16 ms` and
  A10B `ctx16384` attention `5.78 -> 6.61 ms`. Do not reopen KV quantization for
  MoE decode unless the reader/dequant path changes materially; the next
  attention body probe should favor layout or a structural body rewrite.
- A3B group8 tile1 decode attention is falsified. The sidecar passed the group8
  correctness gate, but `ctx16384` phase was flat/worse versus tile2: phase sum
  `12.18 -> 12.53 ms`, attention `3.25 -> 3.27 ms`. A group16 tile2 probe was
  also stopped at correctness. Do not reopen smaller subgroup splits without a
  new dataflow mechanism.
- Fused Q8_0 GDN-front decode is falsified in the tested forms. Dispatch-only
  fusion was flat/noisy on battery (`35.75/35.78/36.56 t/s` base/fused/base), and
  the x-cached four-simdgroup version was correctness-safe but catastrophic
  (`36.73/25.04/35.90 t/s`, battery-confounded but too large to ignore). Do not
  reopen this as launch fusion or hidden-vector threadgroup staging; a future Q8
  front branch must change packing/tiling materially.
- Q8_0 mat-vec row-pairing is falsified for A10B decode. An `NR0=2` variant passed
  primitive and A10B decode correctness, but regressed `tg128` from
  `36.73/36.90 t/s` base to `33.37 t/s`. Do not copy Q4/Q6 row-pair geometry to
  Q8_0 without a new memory-access mechanism.
- Split-concurrent GDN front projections are falsified for MoE decode. The probe
  was correctness-safe on A10B, but regressed `tg128` from `36.72/36.90 t/s` base
  to `35.09 t/s`; future GDN front work needs a different work shape or input
  reuse, not four concurrent encoders around the existing mat-vecs.
- Generic grouped expert-major MoE routed FFN via CPU ledger + gather/scatter +
  generic per-expert mat-mat is strongly negative on both A3B and 122B.
- Re-based “split routed FFN sidecar” experiments now show that beating the old
  packed-slot path was the wrong denominator: against the live grouped backend,
  separate gate/up grouped matmats plus the existing `silu_mul` + grouped down
  only reach parity to slight loss on `pp512`, so broad split-sidecar work is
  demoted until a smaller `MUL_MAT_ID`-style proof beats the current grouped
  projection / routed tail directly.
- Shared-expert batched stage-2 rewrite is semantically correct but slower
  end-to-end on A3B packed prefill.
- F16 routed-inner traffic reduction on the live Q5-down MoE path is a wash to
  slight loser end-to-end.
- Dense paired `gate+up` prompt fusion is exact-correct but only `~1.04x` in the
  exact-shape 27B microbench at `N=321`, below the go gate.
- Forcing single Q4 prompt mat-mat to `NR1=16` is worse than the current `NR1=32`
  path at `N=321`, so easy tile narrowing is not the answer.
- A3B Q4 short-MoE threshold/dispatch retreads are demoted: hot-threshold sweeps
  are tiny/noisy, all-`n32` regresses, all-`n16` only helps the losing side of the
  knee, old packed-routed fallback is about half-speed at `pp512`, and a simple
  bounded range-width cap failed A/B.
- A `<8`-only grouped n8 tile is also demoted: it preserves too much grouped
  underfill/control overhead and regressed Q4 `pp512` GPU time (`~0.706/0.709`
  base to `~0.732/0.732`).
- A masked cold-packed SwiGLU proof is demoted too: reusing the old packed-slot
  direct kernel only for `<8` regressed Q4 `pp512` GPU time (`0.7086/0.7027`
  base versus `0.7274/0.7380`). Packed fallback variants need a new mechanism
  before reopening.
- A first Q5 down multi-expert tiny4 microtile showed positive GPU time but failed
  correctness (`logits cos=0.981292` on the A3B prefill-vs-single gate). A
  corrected microtile remains a possible branch, but correctness must run before
  any perf row is counted.
- The corrected Q5 tiny8 R16 down proof is exact and now defaults for
  `chunk_p <= 768`, with `QWEN_PREFILL_MOE_TINY8_DOWN_R16=0` as rollback. Q4
  `pp512` default/rollback rows are `0.6864/0.6871` versus `0.6889/0.6969`
  GPU ms/token; `pp1024` is left on the old path in auto mode.
- The naive Q4 SwiGLU port of that geometry is falsified. It passed correctness
  but regressed Q4 `pp512` GPU ms/token (`0.6881/0.6854` base versus
  `0.7078/0.7086`) and failed to move the `<8` target bin (`36.28 -> 36.91 ms`).
  Do not revive one-simdgroup R16 SwiGLU without a new mechanism.
- Split gate/up with existing grouped Q4 matmuls is also falsified. It passed the
  A3B prefill-vs-single correctness gate, but Q4 `pp512` GPU ms/token regressed
  from `0.6851/0.6820` to `0.7017/0.7009`; the `<8` SwiGLU phase moved only
  `37.98 -> 36.92 ms`, far below the go gate. Do not reopen split sidecars unless
  the second projection fuses the epilogue and materially changes dispatch shape.
- The MR32 Q4 `<8` microtile is falsified too. It packed two tiny experts per
  threadgroup with two simdgroups per expert and no separate F32 epilogue, passed
  correctness, but regressed Q4 `pp512` GPU ms/token (`0.6851/0.6858` default
  versus `0.7032/0.7006`) and left `<8` SwiGLU flat (`35.88 -> 36.02 ms`).

## Force-Ranked Next Bets

Current rank after v0.356, plus the post-v0.356 hardware-headroom audit digest:

External audit read: do not turn generic code-smell cleanup into the new top of
queue without measured primary-row gates. The useful additions are narrower:
RoPE precompute is a cheap correctness-bounded sidecar because the current
kernels still compute `pow(theta, exponent)` per pair; PSO-cache locks,
`view_subrange` allocation churn, and tensor clones need warmed CPU/alloc
attribution before implementation because current hot decode rows are mostly GPU
active; `[[max_total_threads_per_threadgroup]]` and simdgroup-barrier pruning are
microbench candidates on exact active kernels, not blanket edits. Keep the active
implementation spine on concrete quant/long-context rows, while adding these as
gated hardware-headroom probes.

v0.389 counter-pivot read: autonomous Metal hardware counters are unavailable on
this M4 Max beyond timestamps, so do not wait for counter tables that cannot be
captured. Use existing software gates instead. v0.455 SUPERSEDES this pivot:
a user-saved Instruments template (`metal-counters`: Performance Limiters
counter set, Performance State Maximum) unlocks the full 64-counter Apple
limiter stream headlessly via `xctrace record --template 'metal-counters'
--attach PID`. The measurement loop lives in
`scripts/profile/gpu_limiter_capture.py` (uv script; `hold`/`capture --reuse-pid`
amortizes the ramp; per-experiment CSV under `target/profiles/gpu-limiters/`,
~30 s warm capture, several minutes cold export, ~1 s cached re-analyze). A3B
ctx16384 verdict is
recorded: low effective residency (Kernel Occupancy ~28% vs manager ~72%),
bandwidth 57-68% of stream, ALU pipes <=31%; NWG192 discriminator moves
nothing so the cap is per-kernel residency shape. Newly promoted top item:
per-kernel residency audit + one occupancy-shape retune gated on the
counter loop (inflight must move before any e2e claim). Byte reduction stays
demoted. Interpretation guide: `docs/bench/2026-07-03-xcode-decode-capture/`.
v0.457 adds the phase-noop limiter packet. Full A3B ctx16384 is still `3` kicks
at `2.94/3.70/3.49 ms` with occupancy `~28%` and Read BW `265/270/310 GB/s`.
No-GDN remains `3` kicks but lower occupancy (`17.6/18.0/23.3`); no-MoE and
no-GDN+no-MoE collapse to `1` kick/token (`8.86 ms` and `6.36 ms`) without
approaching stream bandwidth or improving occupancy. Treat noops as budget
oracles, not family-counter truth, because topology changes. The rank is now:
(1) PSO/resource audit for hot decode kernels, (2) batch-2/two-stream concurrency
discriminator, (3) one surgical occupancy-shape retune only after the resource
table predicts the cap and the counter capture moves occupancy/SIMD inflight.
v0.458 adds the cheap PSO resource audit. `qwen-bench metal-pipelines` confirms
`thread_width=32`, no static threadgroup memory, and no ICB support across the
hot decode set. Attention main/reduce and GDN-step are one-simdgroup contracts
(`max_threads_per_tg=32`, packed NSG4 `128`), while mat-vec/MoE/elementwise rows
report `1024`; metal-objdump does not expose register/private-memory counts. This
kills static-TGM and ICB as visible caps but is too coarse to pick a concrete
retune. Current rank: (1) batch-2/two-stream counter capture, gated on occupancy
rising from `~28%` toward `>=38-45%` and aggregate throughput `>=1.15x`; (2)
per-dispatch labels that cover `>=95%` of GPU interval time; (3) one focused
attention/GDN occupancy retune only after labels/counters identify the live
dispatch.
v0.459 runs that batch-2/two-stream discriminator in-process with shared model
weights, separate sessions, and separate command queues. It overlaps well
(`gpu_sum/gpu_span=1.90x`) but fails the pivot gate: A3B ctx16384 aggregate t/s
is only `88.1 -> 109.6` (`1.24x`), per-stream falls to `54.8 t/s`, and device
occupancy rises only `23.7 -> 30.6` under the limiter capture. Do not promote
scheduler/multi-slot/replay as the primary branch from this evidence. Current
rank: (1) single-stream per-dispatch labels/counter attribution, gated on
covering `>=95%` of GPU interval time and matching phase/noop totals; (2) one
focused attention/GDN occupancy retune if the top 3-5 dispatch families dominate
GPU time and retain the low-residency signature; (3) revisit multi-slot only if a
future shared-submit experiment reaches aggregate `>=1.5x`, per-stream
`>=70-75%` of baseline, and occupancy `>=40%` without read BW saturation.
v0.460 replaces the unavailable per-dispatch xctrace labels with app-side Metal
timestamp sampling at existing compute-encoder stage boundaries. The probe passes
the coverage gate (`raw_coverage_assuming_ns=1.000`) and perturbs A3B ctx4096 GPU
time by about `+13%`, so use it for attribution, not throughput. A3B ctx16384
family map: `attn_mixer_route 24.46%`, `gdn_after_route 20.13%`, `gdn_front
17.55%`, GDN-block MoE gate/up+down `14.15%`, and `tail_lm_head_argmax 8.00%`.
Current rank: (1) split the dominant mixed families only enough to decide the
retune target, after a longer/repeated stage window confirms the top-three
concentration (`attn_mixer_route` into attention vs route/post-norm;
`gdn_after_route` into GDN step/output vs residual/post-route glue); (2) run a
focused limiter capture on the winning split family; (3) edit the kernel only if
one isolated subfamily is `>=25%` alone or the top three families stay `>=60%`,
and the counter signature still shows low occupancy/inflight rather than byte
saturation. Do not read the `window=4` percentages as fine-grained speedup
ceilings.
v0.461 splits the top attention bucket. Route prep is only `~2.1%`, front
projections are flat at `~0.49 ms`, and the ctx-scaling term is `attn_body_out`
(`1.2516 ms` at ctx4096 -> `1.9174 ms` at ctx16384). Existing `attn-intra`
proportions point at v4 main+reduce for that slope, but the absolute ctx16384
share is still `16.75%`, below the combined GDN front/after-route pool (`~38.7%`).
Current rank: (1) split `gdn_after_route` and `gdn_front` with the same stage
timestamp discipline; (2) run an attention-body limiter/counter probe tied to
main+reduce, not another blind NWG/tile sweep; (3) promote attention work only if
it shows `>=0.35-0.50 ms` recoverable ctx16384 upside without ctx4096 regression,
otherwise prefer the largest isolated GDN subfamily.
v0.491-v0.493 close the occupancy-attribution thread and KILL the persistence
branch (cx-gated topology program, session `019f347b-c...`). v0.491 (A0): the
v0.455 "28% occupancy" is a duty-cycle artifact - kicks start at 56-58%, burst
to 60-70%, and spend most interior 25 us bins in recurring valleys (H3-shaped,
no tail decay). v0.492: the only sub-kick trace structure is WindowServer
compositing (~3% of the shortfall; excluded), and within-encoder dispatch
timing is triply unobservable on M4 Max - do not re-attempt trace-based
alignment. v0.493 (B0 probe, `qwen-bench topology-probe`): 32-wide
co-residency PASSES (~18/core conservative, 30-58/core max_alive; fill
ceilings ~1764 threads/core compute-bound vs ~2950 memory-stalled; ~96
simdgroups/core hard cap at W128/256); bounded self-validating cross-TG
signaling is RELIABLE (100% delivery to 720 consumers, excess p99 <= 0.1 us)
but relaxed cross-object data-then-flag is UNSAFE under traffic (2e-3..6e-3
stale rate; payload-in-flag is mandatory); dispatch boundaries cost only
~2-3 us at decode-realistic shapes (~11 us extreme mixed), so persistence
recovers <= 18% of a 110-stage glue ladder vs a >= 30% kill line - Program B
is DEAD, B1a cancelled pre-build, and the v0.443 DFlash persistent-verify
reopen condition is CLOSED. Working attribution (cx: "roadmap working
hypothesis requiring an attribution-first discriminator"): the valley mass is
INTRA-dispatch under-parallelism - narrow one-simdgroup dispatches cannot
fill 40 cores regardless of boundaries. Next program (cx-signed direction):
WIDTH/CONCURRENCY restructuring, attribution-first - (1) per-family
dispatch-width census over the v0.462 stage splits; (2) a concurrent-encoder
discriminator on 2-3 provably independent narrow stage pairs (machinery
exists: `begin_concurrent` + hazard notes); (3) only then any production
widening. Program C (ctx-65536/131072 scoreboard vs pinned llama.cpp +
limiter capture) remains GO and may re-rank attention-side work first.
v0.494 executes Program C: ours vs pinned b9833 at ctx16384/32768/65536/
131072 = `1.34x/1.30x/1.1-1.2x/1.80x` (llama.cpp halves per doubling past
65k; the moat WIDENS at true-long). Attention @131k = `46.2%` of the token
(pre-registered 45-55% band CONFIRMED); the 131k limiter capture shows the
SAME latency/occupancy-bound signature as 16k (occupancy ~26%, read BW ~66%
of stream) - true-long is NOT bandwidth-walled, so the width/concurrency
attribution applies there too. `--prefill-warm` (validated: gpu_ms within
~0.5-2% of decode-ramp) makes deep-context measurement routine (131k warm
~4 min). Parked v0.490 condition is now MET if true-long becomes primary:
revisit `QWEN_ATTN_V4_G8_BCAST=1` promotion and re-rank attention byte/
occupancy work against the W-program (Britt's call).
v0.495-v0.497 then close the board (cx-signed program-level statement,
session `019f347b-c...`): the dispatch census revises narrow-glue to a
~1.2-1.5 ms ceiling; W1b partition packing is falsified at its control row
(+61.5%); the Q8-KV reopen is blocked at scope review (v0.437 covers it);
and Program T (tree speculation) is killed at T0 by DIRECT tree-decode
simulation - tree-sim lifts emitted/step +20-26% (code 4.27->5.12,
narrative-start 2.91->3.51) but even the v0.444 all-heroics cost ceiling
only touches 1.25x on code and fails narrative-start at 0.85x, so the
recorded DFlash reopen condition stays unmet, which in turn keeps the
v0.439 >=32-row matrix-decode precondition unmet. "Under current model
assets, current drafter, current N=16 verify budget, and current M4 Max
measurements, all recorded reopen conditions in this arc are closed by
measurement. Further large decode gains require new assets or a materially
new execution primitive, not another local kernel retune." Reopen recipes
are recorded per branch: DFlash/tree needs a larger-block or
stronger-shallow drafter (the `--tree-sim` harness prices any candidate in
minutes, no engine work); matrix decode needs a real multi-token source;
attention main needs a materially different body; persistence needs
nothing (its mechanism was measured and does not exist at useful
magnitude on this GPU).
v0.498 then runs the closure itself through a falsification audit (was the
v0.443/v0.444 skinny-verify cap a single-design artifact?) and REINFORCES
it: a clean-room multi-column GEMV family (mat-vec geometry, occupancy-
corrected, E0 bit-exact per column, Q4_K + Q6_K) lands at `~270
GB/s-equiv` under hot-repeat microbench conditions — the same wall as the
v0.444 F32-dot family, now two design families deep (the GEMV family was
measured in both ALU-heavy and occupancy-split bodies). `c(2) =
1.57-1.71`, `c(4) = 2.9-3.3`: deep-N scalar multi-column decode stays
closed, and the reopen arm (c) "MTP-side economics" resolves as MARGINAL:
composed MTP-1 ceiling `~1.15-1.2x` code-only at measured on-box alpha
(code `0.984`, prose `0.693`, equivalence PASS), gated on
single-command-buffer GPU-fed drafting (the measured `28-44 ms/call`
prototype drafting is the binding loss today). Two recorded follow-ups, both with kill lines: (1) MMA-unit
small-N kernels at mat-vec-grade occupancy (the scalar-ALU cap does not
bind the MMA pool; the MM tile's 56-128 GB/s was 64-row-tile
under-occupancy) — kill line `c(2) <= 1.25` on ffn shapes, else the
shallow lane closes too; (2) whole-step packed MTP-1 measurement only if
(1) or the drafting restructure moves first. See the v0.498 PERF-LOG entry
for the full gate-session numbers (fixed-layer anchors, DFlash drafter
health at HEAD).
v0.499 executes follow-up (1) and the kill line fires: an 8-row x
8-padded-column simdgroup_matrix kernel at `n_out/8` TGs (design jam
corrected the mechanism first — M4 has no separate MMA pool, so the
candidate win was dequant amortization + dense FMA encoding + occupancy)
measures `c = 1.92-2.41` on the kill shapes at `4.5-5.2 TF` (correct at
cos 1.000000; a staged-B/wider-K v2 regressed; two variants measured,
wider micro-variants explicitly not covered). Per pre-registration THE
SHALLOW SMALL-N LANE CLOSES: across both measured kernel classes the
best-known costs are `c(2) 1.57` (scalar), `c(4)/c(8) ~1.9-2.4` (mma8),
leaving composed MTP-1 at `~1.15-1.2x` code-only — below practical
value. The kernel-level speculative-verify arc on M4 is CLOSED at the
pre-registered scope. Reopens: an untried micro-variant beating
`c(2) <= 1.25` (the harness prices candidates in minutes); M5-class
tensor silicon; a step-change-alpha drafter/MTP asset (price in
`--tree-sim`, no engine work); or a real multi-token source arriving via
batching, which inherits mma8 as the best-known N in {4,8} primitive
(`~2.0-2.8x` the incumbent MM tile at N=8).
v0.500 then answers Britt's stale-assumption challenge with a staleness
audit + cx-vetted systematic sweep (see the PERF-LOG entry for the
table). The audit finds the verify path frozen at v0.44x selections:
GENERIC 32-wide tiles at N<16 (n16 needs n_query==16 exactly), per-token
GDN/rope/scatter/attention where prefill has packed bodies, one serial
encoder, and MTP drafting paying full-graph + full-V lm_head + sync per
call. The sweep (N {2,3,4,8,16} x 7 shapes x dtype/N-dependent configs,
up to 10 per cell, correctness asserted) fires R2 hard — the current
selection is dominated at EVERY N<16 cell (5.0-8.3x vs best 1.4-2.5x on
Q4_K/Q6_K/lm_head cells; Q5_K floors at zero-code pad16 3.9) and at
N=16 on 5/7 shapes — while
R1 stands closed (best c(2) 1.53, bar 1.25) and the interaction variants
+ n16_v2 are freshly falsified rather than assumed. Material revision:
best-kernel verify(4) projections land at ~1.6-2.3x per shape (was
generic ~5x), shifting composed MTP-3 to `~1.5-1.7x` code-only
(composition estimate; gates unchanged: drafting restructure +
whole-step measurement). Top-ranked follow-up is now the VERIFY-PATH
INTEGRATION PROJECT: wire the best-kernel table (pad16 as the zero-code
floor) + packed GDN/rope/scatter into packed verify behind
greedy-equivalence and whole-step re-measurement gates; then whole-step
MTP-3 re-pricing; then the Q5_K port and drafter phase-2/sync cleanup.
v0.501 lands stage one: the small-N table is production default
(`QWEN_MATMAT_SMALLN_TABLE`, rollback documented as re-exposing the
witness below) and moves whole-step DFlash static-16 verify
`0.862/0.871x -> 0.952/0.951x` (+10.4%, two samples each side,
equivalence PASS) and the MTP-N prototype to `0.836x` (3.32
emitted/step; drafting machinery still binding). The rollback A/B also
SURFACED A LATENT CORRECTNESS FINDING, confirmed on the pristine v0.500
binary: the pre-table path deterministically violates greedy
token-identity on a short-prompt static-16 witness (first divergence at
index 84, single near-tie flip; best-fit mechanism: E1 half-staged
verify tiles vs the mat-vec decode chain). The witness is now a
required gate row; the residual E1-accept-path risk class is recorded
with tie-guarded verify (E0 recompute under a logit-margin guard, MoE
3e-4 fallback as the house pattern) as the top correctness follow-up.
v0.502 reopens native MTP from the MTPLX M4 Max 64 GB signal (`~65-80
tok/s` on Qwen3.6 27B) but the new qwen-side replay/oracle probes sharply
change attribution: on 27B MTP, `--spec-tokens 3 --tokens 64 --no-warmup`,
`replay-current` records source `21.0 t/s` and reruns the same draft token
vectors with `mtp_calls=0` at only `21.8 t/s` (`0.967x` total), while the
perfect greedy oracle reaches only `27.0 t/s` (`1.201x` total). Therefore
single-CB drafting is necessary but not sufficient; the active MTP branch is
packed-verify structural debt first/alongside drafting, not drafting sync
alone. Force-rank inside MTP: (1) packed GDN or GDN tape/capture equivalent,
(2) packed q_len 2..4 verify attention with KV reuse, (3) packed rope/scatter
and encoder concurrency where phase evidence supports it, (4) draft-only
LM-head/top-k, then (5) single-CB GPU-resident D3 drafting once verify ceiling
is no longer the blocker. Tie-guarded exact verify remains the correctness gate
for any accept-path speed row.
v0.503 refines that again: the packed verify ceiling has a strong physical-N
ladder. 27B perfect-oracle D7/N8 reaches `52.0 t/s` (`2.306x`) and D15/N16
reaches `57.1 t/s` (`2.532x`), while off-ladder D4/N5 is a loss and D8/N9 is
only `32.1 t/s`. Bucketed verify (logical D over physical N8 with padded slots
rolled back) rescues the cliff: D4/N8 `36.1 t/s`, D5/N8 `41.6`, D6/N8 `47.6`.
The actual native D3 path over physical N8 is a small real 27B win: code prompt
128 tokens moves D3/N4 `0.912x` -> D3/N8 `1.031x` at the same alpha, equivalence
PASS. Therefore the active MTP branch is now: (1) acceptance tracing for D7/N8
and D15/N16, (2) bucketed physical N8/N16 verify as the only packet shapes to
optimize, (3) single-CB recursive drafting + draft-only LM head if emitted/step
clears the bucket cost model. Do not optimize arbitrary ragged N before padding
or bucket policy fails.
v0.504 extends the real native recursive MTP bench path to D15 and selects D7/N8
as the practical branch. On the 27B code prompt at 128 tokens, D7/N8 reaches
`27.7 t/s` (`1.153x`) with alpha `0.429` / `~4.0 emitted/step` despite current
per-draft sync and shared LM-head costs; D7/N8 replay-current reaches `32.6 t/s`
(`1.362x`) with `mtp_calls=0`. D15/N16 is rejected for the current one-head
recursive drafter: `17.1 t/s` (`0.716x`) at only `~4.1 emitted/step`, far below
the N16 cost model. Active branch: D7/N8-specific GPU-resident recursive drafting
and draft LM-head reduction; keep D15/N16 as an oracle/asset-watch row only.
v0.505 closes the pure orchestration part of that branch: chaining all D7 draft
slots into one command buffer is correct but moves the 27B code prompt only
`1.153x -> 1.162x` while replay-current remains `1.362x`. Therefore per-depth
wait/readback/command-buffer overhead is not the main remaining D7 draft tax;
force the next branch through a body-vs-LM-head ablation using recorded draft ids
before building a draft-only head.
v0.506 runs that ablation. On the 27B code prompt, D7/N8 body-no-lm-head reaches
`31.1 t/s` and bridge-only reaches `32.1 t/s` versus normal single-CB `27.7 t/s`
in the same probe family; all rows preserve greedy equivalence. This says draft
`lm_head+argmax` dominates the remaining draft-side tax, recursive body is much
smaller, and bridges/KV repair are not worth optimizing. But deleting draft head
entirely still only reaches the low-30s t/s, far below the `52.0 t/s` D7/N8
oracle and the external MTPLX `~65-80 t/s` signal. Therefore exact fused
`lm_head+argmax` is not the next strategic branch unless a microbench proves a
large whole-run gain; prioritize acceptance/rank tracing and N8/N16 verify
phase debt, with draft-only low-bit/top-k head gated on acceptance.
v0.507 adds that rank trace. On the 27B D7/N8 code prompt, terminal mismatches
still contain the target token in the draft top-2 for `14/28` rows, top-4 for
`19/28`, top-8 for `24/28`, and top-16 for `26/28`. This reopens reranking and
tree-rescue as plausible acceptance levers, but only behind an online-policy
gate: top-k containment is hindsight unless draft-side features can choose the
non-top1 candidate before target verification. Next branch is top-k logit/id
capture plus offline policy simulation; do not build tree verify or low-bit
draft head before that simulator clears.
v0.508 adds top-16 ids/logits and runs that simulator. Simple margin-swap
policies are effectively flat (`4.031` emitted/step best), and even an oracle
one-token terminal rescue only reaches `4.812` emitted/step at top-16 versus the
`>=5.2` investigation gate. Close simple rerank. The live acceptance branch is
now either real alternate continuation/tree economics, a stronger learned or
engineered corrector, a better MTP policy/asset, or an MTPLX measurement/reporting
difference; inspect MTPLX before building tree machinery.
v0.509 tests the cheapest draft-head clue from MTPLX and kills it: using resident
Q4_K `token_embd.weight` as the draft LM head preserves target equivalence but
collapses 27B D7/N8 acceptance to `0/127` (`5.3 t/s`). This only kills the
embedding-alias shortcut; a proper low-bit copy of `output.weight` remains a
separate, bounded cleanup idea.
v0.510 adds MTPLX-style hidden-semantics controls and decode-only accounting.
This materially changes the MTPLX-gap attribution: D7/N8 oracle reaches
`74.7 t/s` decode-only (`52.7 t/s` total), inside the external `65-80 t/s`
screenshot band once prompt/MTP prefill is excluded. Actual D7/N8 with legacy
pre/pre hidden feeds is only `30.2 t/s` decode-only and `3.879` emitted/step on
the code prompt; defaulting both base and recursive MTP hidden feeds to post-norm
raises that to `33.0 t/s` and `4.267` emitted/step. A narrative prompt moves
`26.8 -> 30.7 t/s` decode-only and `3.459 -> 4.000` emitted/step. Therefore
short-prompt MTP verify is not the current MTPLX-class limiter; acceptance and
MTP semantic/asset parity are.
v0.390 then demotes exact route from the main branch: A3B/A10B route replay still
repeats (`1.01/1.34 ms`), but
production already fuses the high-value topk/shared half and the only remaining
exact boundary is router logits into global exact top-k. The v0.390 local route
rank was: (1) captured MoE gate/up/down projection throughput
and active-token/expert batching; (2) a bounded attention read-once prototype only
if it changes the main-body memory shape; (3) exact route only if a prototype can
save `>=0.4 ms` A3B or `>=0.5 ms` A10B route total without consumer movement.
Local route barrier/candidate variants, GDN row-shape work, attention tile/NWG
knobs, and host-only cleanup stay closed unless fresh phase/no-op/microbench
evidence reopens them.
v0.391 adds the missing captured down gate and aligns it with production R2 for
`f_exp=512`: A3B/A10B captured gate/up and down micro rows now match phase within
noise (`1.079/0.814 ms` versus `1.08/0.82`, and `3.107/2.595 ms` versus
`3.21/2.56`). Use captured MoE micro first for compute-branch proposals, with a
required `5-10%` micro win before full phase promotion.
v0.393 then kills the existing one-token Q4/Q5 routed FFN monolith under captured
routes: A3B fused is `80.513 ms` versus split captured `1.894 ms`, and A10B fused
is `210.142 ms` versus split captured `5.701 ms`. This rules out fusion that
serializes the active weight stream into one threadgroup per token/layer. It does
not rule out projection-kernel changes, active token/expert batching, or tiled
fusion that preserves split-path parallelism; require `>=10%` full captured MoE
improvement before reopening another monolith-shaped decode branch.
v0.394 proves active-token batching is a real captured-route MoE lever: A3B split
gate/up+down improves from `1.894 ms/token` at one token to `1.198 ms/token` at
sixteen tokens, and A10B improves from `5.701` to `4.450 ms/token`. This justifies
a batching suite and multi-slot architecture branch, but not an end-to-end claim:
same-context route correlation, scheduler/KV overhead, layer variance, and expert
occupancy histograms remain unmeasured. Keep projection-kernel work active because
it benefits both single-token and batched decode.
v0.395 adds route occupancy stats and a ramp token-id control. The repeated-zero
captures overstated expert reuse: A10B t16 avg unique experts jumps `8.34 ->
54.40` under ramp, but gate/up only slows `33.24 -> 34.54 ms` and down improves
`37.96 -> 37.40 ms`. Therefore the batching win is primarily packed-slot
shape/occupancy, not repeated-token expert reuse. Use ramp or real prompt traces
for future timing; high-reuse synthetic wins alone are not promotion evidence.
v0.396 adds a loaded-once `moe-batch-sweep` to map that knee without repeated
loads. Ramp sweeps show A3B combined MoE projection time improves `1.889 ->
1.211 ms/token` from t1 to t16, while A10B improves `5.659 -> 4.481 ms/token`.
The useful knee is around t8, and t16 adds little. Batching is now a serious
architecture branch, but the next gate is real-prompt/multi-context capture plus
end-to-end decode overhead; do not promote production multi-slot decode from MoE
micro evidence alone.
v0.397 closes the first realism gap with a contiguous real prompt from
`the_current.md`: A3B is `1.885 -> 1.215 ms/token` from t1 to t16 and A10B is
`5.710 -> 4.508 ms/token`, essentially matching ramp. This justifies the next
gate, not production yet: run independent prompt/disjoint-context captures to
test cross-context packing before building a scheduler architecture branch.
v0.398 runs that stronger disjoint-position gate with stride128. A3B still passes
(`1.870 -> 1.219 ms/token` at t16), but A10B collapses after t4 and regresses at
t16 (`5.751 -> 5.510 ms/token`, versus `4.508` contiguous). Generic production
batching is demoted; keep A3B batching alive, and require A10B route-aware
packing/kernel evidence before any broad scheduler work.
v0.399 adds independent-file capture with one fresh session per prompt file. A3B
passes again at ctx512 (`1.885 -> 1.264 ms/token` at b8), while A10B only has a
useful b2 row (`5.727 -> 4.756`) and weakens at b4/b8. Production batching should
be model-specific: A3B can proceed to a targeted prototype gate; A10B should cap
at b2 or stay disabled above b2 unless an expert-sorted microproof beats that cap.
v0.400 runs that A10B expert-sorted kill-test as a perf-only locality upper
bound. It sorts captured slots by expert id without preserving exact token/slot
semantics, so an exact implementation would have to pay additional overhead.
A10B stays flat at b2 (`4.725 -> 4.723 ms/token`) and regresses at b4/b8
(`4.840 -> 4.945`, `5.135 -> 5.223`), which kills expert sorting as the A10B
rescue for this workload. A3B sees only sub-1% sorted changes; keep the next
batching branch focused on exact A3B end-to-end overhead, not route ordering.
v0.401 adds a routed-FFN batching upper-bound calculator and applies it to A3B
`ctx512/2048`. Even the optimistic b8 estimate, including Q6 down fallback, is
only `~0.77 ms/token` saved (`8.2-8.3%` of phase sum, ideal `~1.09x`) before any
real scheduler, pack/scatter, command-buffer, KV, attention, shared-FFN, or
sampling overhead. This demotes the immediate A3B multi-slot scheduler prototype:
batching remains a later system feature, but the top decode branch should return
to larger single-token byte-reduction/fusion unless a future upper-bound row
clears `>=12-15%` credible end-to-end savings.
v0.402 kills the cheap MoE `max_total_threads_per_threadgroup(64)` annotation
probe on the exact hot Q4 SwiGLU and Q5 down packed kernels. The patch builds,
but A3B independent-file captured rows are flat (`b1 1.824 -> 1.828 ms/token`,
`b8 1.221 -> 1.220`) and A3B phase is flat (`9.32 -> 9.34 ms`). A10B shows a
small batch-only `b8` improvement, but `b8` remains slower than `b2` and
single-token is flat/slower. Do not blanket-annotate kernels without manual
shader-profiler evidence or a named-kernel micro win.
v0.403 refreshes A3B long-context deep phase/roofline at `ctx8192/16384` after
the batching and annotation demotions. Attention is now the largest structural
slope term (`2.30 -> 2.53 ms`, `21-23%`), with KV subgroup+partial estimates only
`~282-296 GB/s`. GDN QKV/Z and LM head are near stream shelves (`424-515 GB/s`),
route topk/shared is real but locally falsified (`0.67-0.68 ms`), and routed
gate/down remains moderate but constrained by recent monolith/batching gates. The
next A3B long-context branch should be an attention main-body KV-read shape
change with an `attn-intra`/oracle gate; more tile/NWG/reduce retunes stay closed.
v0.404 then shows the local NWG shelf is small: NWG192 saves only
`0.13-0.15 ms` in phase at `ctx16384/32768`. v0.405 kills the concrete read-once
S2 oracle: correctness is exact, but staging K/V once for two simdgroups regresses
the main body by `3.7-5.6x`. Do not reopen attention read-once work by adding
large threadgroup-memory staging or reducing Q-head grid parallelism; the next
hardware-saturation branch should move to A3B multi-slot batching or a broad
decode byte/fusion audit unless a new attention idea avoids this failure mode.
v0.406 supplies the missing broader batching signal: A3B GDN `qkv+z+out` over all
30 GDN layers improves from about `2.44-2.48 ms/token` as repeated matvecs to
`1.26 ms/token` at batch 8 and `0.55 ms/token` at batch 16 using existing matmat
kernels. That is an always-on `~1.17-1.93 ms/token` primitive saving at
`ctx32768`, larger than the routed-MoE-only bound. Make the next architecture
gate a production-shaped decode-phase batch replay over `S={1,2,4,8,16}`; promote
multi-slot/layer-batched decode only if `S=8` clears `>=10%` or `>=1.0 ms/token`
phase-equivalent savings after attention, LM head, MoE routing, and layout costs.
v0.407 broadens that primitive gate with `decode-proj-batch`: A3B projection-only
aggregate remains negative through `S=4`, then crosses over hard at `S=8`
(`4.7255 -> 2.8866 ms/token`, `+1.8389 ms`) and `S=16`
(`4.6600 -> 1.2346`, `+3.4254 ms`). A10B and dense 27B confirmations are larger
(`+5.94/+9.55 ms` and `+11.23/+26.29 ms` at `S=8/16`). This passes the
projection-only kill gate and re-promotes a fuller decode-phase batch replay as
the highest-leverage architecture gate, but not a scheduler implementation yet.
The next replay must charge attention body/KV, routed-MoE capture/replay,
routing/topk, slot pack/scatter, layout copies, and ragged occupancy; require net
`S=8` savings to stay above `>=10%` or `>=1.0 ms/token` before scheduler work.
v0.408 digests the kernel-bypass audit against that result. The frame is useful,
but it does not outrank the measured batching gate: warm single-token decode has
historically been `~95.8-97.8%` GPU-active, so command-stack bypass is not yet the
dominant proven limiter. Add these profiling targets, in order: (1) full
`decode-phase-batch` replay with command/dispatch counts and net `ms/token`;
(2) ragged continuous-batching occupancy mixes (`50/75/90%` active, joins/exits,
mixed context lengths); (3) MoE route/pack/scatter/expert-utilization accounting;
(4) fused `lm_head+argmax/top-k` as an exact "do not materialize logits" probe;
(5) no-allocation/resource-binding audit for steady decode; (6) memory-format
experiments such as GDN/KV/residual BF16 only with correctness and quality gates;
(7) one-layer megakernel or persistent-work-queue proof only after the charged
replay shows command/encoder overhead is the remaining ceiling. Deprioritize
AMX/ANE coprocessor paths, GPU-side graph traversal, hierarchical `lm_head`,
layer skipping, KV clustering, sparse FFN, gate-based attention skipping, and
near-zero drafters until they have explicit quality gates and a measured phase
ceiling. `decode_phase_roofline.py` now reports stream lower-bound columns so the
"cycles per token" question can be asked from existing phase artifacts.
v0.409 adds a charged `decode_batch_upper_bound.py` estimator and applies it to
A3B `ctx32768` using fresh deep phase plus same-context routed-MoE replay. The
charged `S=8` row saves `2.4618 ms/token` (`19.55%`, ideal `1.243x`) after leaving
attention body/KV, route/topk/shared gate, GDN tail, residual/norm/layout, and all
other unmodeled work at baseline. `S=4` is still negative and `S=16` is strong
(`4.0858 ms`, `32.45%`). This clears the continuation gate for a real
`decode-phase-batch` replay, but still does not justify scheduler architecture.
The replay must report net `ms/token`, wall/GPU split, command/encoder/dispatch
counts, attention body/KV, route/topk, routed MoE, GDN tail, pack/scatter, layout,
logits/sampling, and ragged occupancy. Promotion gate remains `S=8 >=10%` and
`>=1.0 ms/token` under realistic active-slot availability.
v0.410 charges the obvious projection layout hole directly. `decode-proj-batch`
now times GPU blit pack/scatter plus matmat, and A3B `S=8/16` only loses
`0.0747/0.0638 ms/token` versus pure batched matmat. The layout-charged upper
bound still saves `2.4151 ms/token` at `S=8` (`19.18%`, ideal `1.237x`). Copy-only
layout is not the branch killer. Do not spend another isolated-layout cycle; the
next artifact must be actual `decode-phase-batch` replay with attention body/KV,
route/topk, GDN tail, routed MoE, logits/sampling, and ragged occupancy visible.
v0.411 lands the first real replay slice: one GDN layer with per-slot pre/post
norms, real GDN tail/state mutation, packed qkv/z/out projections, and all-slot
correctness checks. A3B block-0 remains negative at `S=4` but wins at `S=8/16`
(`0.1363/0.1007 ms/token` saved in the one-layer harness, `43-54%`). This is a
continuation signal, not a scheduler gate: absolute one-layer timings do not map
directly to full-model phase rows, block-0 may be favorable, and full occupancy is
idealized. Next: measure representative GDN layers, then MoE/FFN replay with route
divergence, before full integrated phase replay or scheduler architecture.
v0.412 removes the block-0 concern: first/middle/last A3B GDN layers all pass
all-slot correctness and save about `0.033-0.035 ms/token/layer` at `S=8`, and
`0.068-0.079 ms/token/layer` at `S=16`. This makes the GDN economics plausible,
but the next highest-leverage question is integration, not more isolated layers.
Build an integrated multi-block decode slice with ragged masks; promote only if
it preserves at least `0.5 ms/token` at `S=8` or `1.0 ms/token` at `S=16` on A3B,
and kill/deprioritize if integrated savings fall below roughly `25%` of the naive
sample extrapolation or correctness drifts across chained layers/tokens.
v0.413 adds that chained-state bridge for GDN-only replay: four consecutive GDN
layers preserve all-slot correctness (`min_cos_h=0.999999509`) and save
`0.1594 ms/token` at `S=8` and `0.2536 ms/token` at `S=16`. This raises confidence
that the GDN subpath is real, but it still skips attention/MoE blocks. Next exact
step: a full block-slice for 2-4 consecutive real blocks that executes normal
attention/MoE and swaps only GDN for replay. Ragged masks and scheduler work wait
until that block-slice shows net wall-clock savings.
v0.414 clears that first integration gate: a four-block A3B MoE slice with three
GDN blocks and one attention block, normal MoE route/FFN, and only GDN mixer replay
saves `0.1409 ms/token` at `S=8` and `0.1899 ms/token` at `S=16`, with
`min_cos_x=0.999999329`. Continue, but do not promote to scheduler yet. The next
decisive tests are nonzero/long attention positions, mid/late block windows, and
ragged active-slot masks; position-0 full occupancy is still a friendly case.
v0.415 keeps the early/mid path alive and quarantines late GDN-before-attention:
early `block0..4` still saves `0.1522 ms/token` at synthetic `pos4096` and
`0.1539` at `pos16384`, while mid `block20..24` saves `0.1375` at `pos4096`.
Late `block37..40` fails correctness at both `pos0` and `pos4096`, while pure
attention block 39 is exact. Treat late windows as unsafe until route/topk ids,
route margins, and per-block boundary deltas identify whether replay drift flips
discrete MoE routing or gets amplified by attention. Scheduler gates must be
window-specific; do not generalize early/mid wins to late layers.
v0.416 external-audit + cx digest: do not let the new hardware-saturation audit
displace the cheapest live uncertainty. Finish route/topk fingerprints and
per-block boundary deltas first; v0.415 localized a correctness cliff, not a
global kill. Keep multi-slot decode replay as the top active hardware-headroom
branch for A3B/serving, but treat it as a product-shaped multi-slot feature, not
a single-stream speed claim. Add these gates before major new rewrites: manual
Xcode GPU captures for large roofline claims; a narrow `attn-intra` win before
reopening KV-Q8; an exact one-layer row/block micro-oracle before chunked GDN
prefill; profile proof that verify attention dominates before packed-N verify
attention; and a same-shape long-prefill win before FA2-style fused prefill
attention. Current ordering: (1) route/topk and boundary diagnostics for replay,
(2) ragged/realistic replay gates if that passes, (3) KV-Q8 reader micro-oracle,
(4) chunked GDN prefill micro-oracle, (5) packed verify attention behind KV-Q8,
(6) FA2 prefill attention behind a fresh long-prefill phase gate.
v0.417 localizes the late-window failure mechanism: the visible cliff is
mediated by route-set flips at near-tie topk boundaries. Passing controls
(`block20..24` and `block36..39`) have zero route-set mismatches, while failing
windows first flip a single expert set at margins around `0.0001-0.0003`
(`block37 slot1` or `block39 slot3`) and then amplify through MoE/attention.
Do not treat route-order changes with the same expert set as semantic failures.
Next scheduler gate should be a conservative two-layer policy: static early/mid
block eligibility first, late blocks exact-only, and then a replay-side route
margin guard with rollback to exact pre-window state before ragged occupancy
work. Calibrate the margin threshold from exact-vs-replay router-logit deltas and
margin histograms, not from the observed failing margins alone.
v0.418 adds that first calibration hook. The observed failing route-set flips all
have replay margins below `1e-3`, but a passing `block20..24` control also has two
safe low-margin rows below `1e-3`. Therefore a dynamic guard is plausible but not
ready: the next gate is a broader margin/fallback-rate histogram over early/mid
blocks, prompts, positions, and slot counts. Do not promote a replay-side guard
from the five-window sample; static early/mid allowlist plus late exact-only stays
the current safe policy.
v0.419 adds the loaded-once margin sweep and complicates a naive static allowlist:
`pos4096` non-overlap windows ending at `block31`, `block35`, and `block39` can
flip route sets, while `pos0` non-overlap windows do not; targeted `block37..40`
at `pos0` still fails. Treat absolute block index as a coarse risk feature, not a
complete policy. The next replay work should measure fallback frequency under a
candidate replay-side margin threshold across real prompts/positions before any
ragged scheduler implementation.
v0.420 shows smaller `blocks=2` windows remove the `pos4096` flips from the
synthetic sweep, but still fail at `block38..40 pos0` (`min_x_cos=0.985815558`).
Therefore shorter windows are only a mitigation. The live replay branch now needs
product-shaped economics: real prompt slot sets, candidate margin thresholds,
fallback frequency, and net savings after exact fallback. Do not keep tuning
synthetic late windows without that fallback-rate accounting.
v0.426 adds the first real-prompt margin harness and broadens the initial sample:
32 non-smoke A3B rows across markdown rollouts, contexts `128/512/2048/3072`, and
windows `block0..4`, `20..24`, `28..32`, `32..36`, `36..40` show zero route-set
mismatches. Route order differs in 2 rows, worst `min_replay_margin` is
`0.000047`, and min `x` cosine is `0.999999965`. A window-level margin guard
would fallback on `3.12%/12.50%/25.00%/53.12%` of rows at
`1e-4/3e-4/1e-3/5e-3`. This keeps replay alive and suggests synthetic
constant-hidden fixtures are adversarial stressors, but cx session
`019f1ffe-1f3a-7120-a872-f068a90fd92d` correctly warns that this is still not a
scheduler gate. Next: widen prompt classes beyond markdown rollouts and pair the
margin table with net replay savings after exact fallback.
v0.427 broadens that packet beyond narrative markdown to mixed narrative/code/
docs/JSON prompts. The combined 47-row A3B sample still has zero route-set
mismatches, 3 route-order-only mismatches, worst replay margin `0.000047`, and
min `x` cosine `0.999999889`. Window fallback rates are
`2.13%/8.51%/23.40%/44.68%/61.70%` at `1e-4/3e-4/1e-3/3e-3/5e-3`. Per cx
session `019f201d-9777-7810-9b23-dbdbeb6dbeae`, treat this as a keep-alive
signal, not safety proof: `0/47` remains weak, synthetic flips remain real, and
the next live gate is net replay economics. Model validation overhead, exact
fallback cost, and ragged slot occupancy before building any production
scheduler. A first plausible threshold to model is `3e-4`; `1e-4` is too close to
known failure margins, while `1e-3` probably burns the win unless fallback can
abort before most replay work.
v0.428 adds that first occupancy-economics gate. Replay is a high-occupancy tool,
not a general slot filler: `blocks=4` loses at S=1/2/3, barely wins at S=4, and
only becomes material at S=6/8 (`~11-17%` gross). `blocks=2` GDN-only pairs show
the same low-occupancy loss but better S=6/8 gross savings (`~19-23%`) and were
already safer in the v0.420 synthetic margin sweep. Treat S>=6 plus a `3e-4`
margin guard as the first policy point to model; `1e-3` likely erases the win and
S<=4 is not scheduler-worthy without a fixed-overhead reduction. Next replay work
should quantify validation overhead and exact fallback cost, then model a realistic
ragged occupancy distribution. If that does not clear a durable `>5-8%` net gate,
park replay and move to the next hardware-headroom branch.
v0.429 codifies the conservative net model. At `3e-4`, S=8 `blocks=2` models at
`~14.6-14.8%` net, S=6 `blocks=2` at `~10.4-11.1%`, and S=8 `blocks=4` at
`~7.7-8.3%`; S=6 `blocks=4` is thin, S<=4 is negative, and `1e-3` erases the
win. Replay therefore has one remaining high-EV gate: measure the actual
validation/fallback path and a realistic ragged occupancy trace. If that mechanism
gate does not preserve S>=6 `blocks=2` above `>5-8%` end-to-end, stop the replay
scheduler branch and return to broader hardware-headroom items.
v0.438 runs that first real-window mechanism gate. The validated path splits
replay by block, reads replay route margins after each block, and charges exact
fallback slots at threshold `3e-4`. On real prompts, S8 blocks=2 still nets
`~11.5-20.3%` before broader fallback and roughly `~4.8-13.7%` after applying the
new `1/15 = 6.67%` S8 blocks=2 fallback packet. S6 is only `~2.7-5.5%` before
broader fallback, and S4 is negative. Update the live policy: replay is S8-only
unless future work lowers validation overhead or proves much lower fallback.
v0.430-v0.432 digest a do-less implementation audit (fixed roofline, fewer
bytes/dispatches) across decode, prefill, and kernels. Banked defaults:
matrix-attention causal tile skip generalized from G6 to all matrix groups
(A3B `pp4096` KQ/KQV `-14%` phase, A10B `pp1024` `-47%`; e2e `+0.9%` A3B),
packed-slot MoE fallback packs stubbed on production prefill scratch
(`~84-160 MB` resident saved; lazy-grown on fallback), split_q_gate layout
copies deleted via strided q-norm + strided fused gate epilogue (decode,
prefill, packed verify; rollback branch keeps the split), and DFlash hidden
capture batched to one strided-row copy per capture layer per chunk. Audit
verdicts recorded so they are not re-derived: production decode has NO
remaining zero-fills; mat-vec activation re-read is logical-only
(cache-resident, staging falsified v0.323); RoPE precompute fails its own
`>=0.5-1%` gate (~0.2-0.3 ms per 4096-chunk); remaining decode glue fusions
(residual+norm, K-chain, conv+L2, sigmoid/decay fold) total ~350 dispatches
+ ~8-10 MB per dense token with GPU already `96-99%` busy — bundle as one
gated experiment or skip; attn_v4 Phase A reduction shape is the largest
in-kernel ALU waste but stays fenced behind the byte-reduction rule. The two
promoted do-less bets are (1) a fused online-softmax matrix-attention body
holding score tiles in registers (same simdgroup tiles, no `[kvh,N,M]` f32
round-trip: `~16 B/elem`, A3B pp16384 `~365 GB`, 27B `~876 GB` per full
prefill; measured phase budget `~4-6%` A3B / `2-3%` 27B e2e at true-long)
and (2) an F16/BF16 repack of the F32 MoE router bank at load (`~84 MB/token
A3B decode, ~1.7%`), gated on an exact-top-k route-equivalence check across
real prompts. Also banked: v0.433 confirms the GPU test suite corrupts
itself under parallel execution (rotating victims; attn_v4 `cos=0.9662`,
mat_mat q8_0 `cos=0.48` under shader validation with no shader OOB logged
— uninitialized-partials hypothesis falsified by the NaN-prime gate). The
suite now runs serially via `.cargo/config.toml` `RUST_TEST_THREADS=1`
(green 2/2, no slower); the corruptor hunt (CPU-side test-helper memcpys /
blits suspected) is an open follow-up.
v0.430-v0.432 bank the near-free tier of the do-less implementation audit (same
roofline, fewer bytes/dispatches): (1) the KQ/KQV causal tile skip was G6-gated
at the host despite group-generic kernels — enabling it for G8/G4/G16 cuts the
A3B `pp4096` matrix body `~14%` (KQ `413->339-360 ms`, KQV `417->343-361 ms`)
and A10B `pp1024` KQ/KQV `~47%` each; (2) packed-slot MoE fallback packs are
now lazy stubs on production prefill scratch (`~84 MB` A3B / `~160 MB` A10B
resident saved; grouped default never reads them); (3) `split_q_gate` is
deleted from all production attention paths via strided-source q-norm and a
strided fused gate epilogue (one layout-copy dispatch + `2*q_dim` round-trip
per attn layer, decode and prefill), and the DFlash hidden-capture tap is one
strided-row copy per capture layer instead of `chunk_p` scatters. The audit's
remaining ranked items: (a) fused online-softmax matrix attention body — the
score tensor round-trips `~16 B/elem` (`365 GB` per A3B pp16384 full prefill,
`876 GB` dense 27B; est. `4-6%`/`2-3%` e2e at true-long) and is the top
prefill do-less item, roadmap-nominated but never built; (b) MoE router
`gate_inp` F32->F16 load-time repack (`~84 MB/token` A3B decode, `~1.7%`)
gated on an exact-top-k equivalence check across real prompts because routing
is discrete; (c) decode glue-dispatch fusions (residual+norm, K-chain,
conv+L2, sigmoid/decay fold) are bounded at `<=1-2%` total with decode
`96-99%` GPU-busy — one bundled A/B at most, not five branches. Measured
non-items: RoPE precompute fails its own `>=0.5-1%` gate (`~0.2-0.3 ms` per
4096-chunk); mat-vec activation re-read is logical-only (`3.5x` amplification,
`~0` DRAM — cache-resident); attn_v4 Phase A reduction shape is the largest
raw ALU waste (`~55%` of Phase A instructions) but the body is
bandwidth-bound and same-byte rewrites stay triple-falsified.
Test-infra triage carried with v0.432: `attn_v4_matches_naive_f16kv` is
load-flaky with a real numerical divergence at `group=4 n_pos=1024 nwg=64
C=16` (`cos=0.9662`; passes isolated) — suspected uninitialized-partials
sensitivity (`zeros_f32` is uninit); needs a scratch-init proof or zero-init
before the next attn-v4 branch trusts suite-load results.
v0.433 resolves that test-infra uncertainty by serializing the suite and
falsifying the uninitialized-partials hypothesis with a permanent NaN-prime gate;
rotating parallel GPU-test victims point at cross-test buffer corruption, not a
specific attn_v4 kernel bug. v0.434 then tests the router-repack bet. The F16
router is top-k safe in the measured packet (A3B 480 checks and A10B 384 checks,
zero order/set mismatches) and prefill-safe after adding F16 to the packed-route
eligibility, but it does not pay as a default: A3B/A10B decode is flat/noise,
A3B prefill is only `~+1%`, and A10B `pp1024` regresses `129.23 -> 114.53`
because F16 loses the F32 E8xP32 route-logits specialization. Keep
`QWEN_MOE_ROUTER_F16=1` plus `decode-moe-router-repack-check` as an opt-in
diagnostic harness, but demote router repack and do not widen to BF16. The
highest-EV do-less item is now the fused online-softmax matrix attention body;
decode glue remains a single bundled A/B only. v0.452 exercises the audit's
pre-authorized SKIP on that glue bundle instead: the ~350 glue dispatches move
~8-10 MB/token = ~20 us at stream against a ~9.3 ms A3B token budget (~0.2%),
and the dispatch-count savings land CPU-side where decode-window measures
med_cpu_enc ~0.64 ms of a GPU-bound ~9.9 ms token (med_wait ~= med_gpu).
Neither side reaches the audit's own 1% floor. Decode glue is CLOSED without
the A/B; do not re-derive residual+norm/K-chain/conv+L2/sigmoid-decay fusion
proposals unless decode stops being GPU-bound. v0.474 nevertheless executes the
residual+post-RMSNorm member of that glue bundle as a concrete exact sidecar; it
is correctness-safe but flat/noise (`~1.001x` A3B, `~1.011x` 0.8B, `~0.999x`
27B `tg128`) and remains default-off. Treat that as confirmation, not a branch to
repeat.
v0.435 executes the cheap `max_total_threads_per_threadgroup` audit on fixed hot
kernels (attn_v4 decode/packed/matrix entries plus GDN recurrence). It is
correctness-clean and keeps warmed A3B `tg128`/`pp512` and 27B G6-matrix `pp512`
inside the expected band, but it does not show a visible win. Keep the hints as
fixed-dispatch contracts; do not rank this as a broad hidden lever. cx also
pushes back on a softmax+KQV matrix bridge: with current KQV `head_dim/64`
y-tiling, a bridge would either recompute probabilities per y-tile or sacrifice
parallelism, so it is likely to teach the wrong lesson. Treat FA2-style matrix
attention as a serious design/capture branch, not a quick intermediate.
v0.436 dirty-tests cx's narrow GDN row/block oracle (`rb4_tgm16`: keep four
state rows resident like NSG4, stage `T=16` Q/K tiles in threadgroup memory) and
kills it. Correctness is green, but 0.8B `pp512` regresses `~11%`, 27B `pp512`
G6 matrix regresses `~2.6%`, and a 0.8B trace shows `gdn_step` itself worsens
`21.93 -> 32.59 ms`. Do not keep the flag/kernel. The active packed NSG4 kernel
already captures the easy row-residency win; Q/K load duplication is not worth
TGM barriers. If GDN recurrence is reopened, start with a floor/no-op ladder or
a genuinely chunked delta-rule formulation, not local Q/K staging. v0.479 tests
and kills a different sequential-recurrence rewrite: lazy state-decay scaling is
correctness-safe on the 27B prefill-vs-single gate, but regresses 27B `pp1024`
(`237.13/223.27 -> 233.89/218.11 t/s`). Do not reopen lazy-scale packed GDN; it
does not falsify chunked delta-rule, which must parallelize or reduce the
recurrence work rather than trade vector multiplies for scalar scale/divide
bookkeeping. v0.483 then tests the smallest exact chunked delta-rule work
reduction: a stable carry-product algebra, a tiny `K*K^T` / `K*Q^T` Gram
precompute, and an NSG4 `chunk16` recurrence kernel. Synthetic CPU/GPU and 27B
model correctness all clear, but the all-in 27B `pp1024` gate collapses
`238.75 -> 124.34 t/s` (`4.1500 -> 8.0000 gpu ms/token`). This kills host-loop
`chunk16` GDN. Future chunked GDN must be one-dispatch-per-layer or genuinely
matmul-shaped enough to beat the current packed kernel all-in before model
integration. v0.484 also kills the remaining cheap local row-loop cleanup: merging
the recurrence update loop with the post-update output-dot loop is exact and
correctness-clean, but 27B suite rows are flat/slightly negative (`pp1024`
`234.568 -> 234.579 t/s`, `tg128` `23.777 -> 23.629 t/s`). Do not reopen local
`gdn_step` loop reshuffles; only a materially different recurrence algorithm
belongs on the live queue.
v0.437 executes the reopened KV-Q8 attention-reader micro-oracle and kills it for
the current `attn_v4` execution model. The branch preserves the F16 grid, adds
MoE/group8 Q8 main and tile2/tile4 subgroup kernels, fixes `attn-intra` Q8
scatter, and changes the reader to Q8x4 float4 accumulation. Correctness is green
against F16 KV (`cos > 0.9999`, `max_abs < 0.01`), but clean A3B phase probes
are negative: `ctx8192` main `0.1113 -> 0.1221 ms`, `ctx32768` main
`0.1767 -> 0.2213 ms`. Keep `QWEN_KV_Q8=1` default-off as an oracle only. Future
compressed-KV work needs a materially different layout/body or a capture signal;
do not spend more blind time on Q8_0 reader variants.
v0.438 then returns to the top replay uncertainty and adds real-window economics
timing to `decode-block-slice-real-margin`. S8 blocks=2 survives the conservative
validated path, S6 is marginal, and S4 is dead. The next replay branch must be an
S8/ragged-occupancy scheduler sketch or nothing; do not spend implementation time
on low-occupancy replay.
v0.439 lands the score-round-trip bet as a TWO-pass design that resolves the
v0.435-parked softmax+KQV bridge objection (no per-y-tile probability
recompute: KQ owns the softmax where its pos-tile is resident, KQV applies
only a scalar `c_t` per (query, 64-pos tile) during staging, y-parallelism
untouched): KQ folds scale+mask+per-tile online softmax into its epilogue
(F16 `P~` + (m,l) sidecar, spill stride padded to 68 floats to kill a 32-wide
threadgroup-memory bank conflict), KQV folds `exp2(m_t - m_glob)` into its F16
staging and `1/l` into its epilogue. Score traffic 16 -> 4 B/elem, softmax
dispatch deleted, F32 score scratch halved to F16 (`~1.07 -> ~0.55 GB` A3B
pp16384). Microbench `1.22-1.37x` on the summed matrix body; phase A3B pp4096
`282 -> 222 ms`, 27B pp4096 `701 -> 541 ms`; e2e A3B pp16384 `+5.2%`, 27B
pp16384 `+1.7%`, pp512 guardrail neutral. Rollback
`QWEN_PREFILL_ATTN_MATRIX_ONLINE=0`. The matrix path also gained its first
isolated micro-oracle (CPU f64 reference, all groups/edges) and a kill-gate
microbench harness. FALSIFIED en route: a true 1-pass flash-attention body at
head_dim=256 (llama.cpp `kernel_flash_attn_ext` shape: Q=8/C=64/NSG=4, O in
threadgroup memory, direct-device K/V loads) is `0.80x` the sidecar at
production shapes; Q=16 occupancy-cliffs to `0.39x`, register-resident O
spills to `0.24x`. Cause: 8..16-row query tiles re-stream K/V 4..8x more than
32-column GEMM tiles and go L2-bound. Do not reopen 1-pass matrix attention
without a >=32-row-tile design that fits registers/threadgroup memory; the
remaining matrix-body headroom (P~ 4 B/elem, F16 Vᵀ sidecar) is bounded and
closed without a fresh phase budget. The v0.439 gate runs also produced the
strongest corruptor-hunt datum yet: the v0.433 corruption class reproduces in
a fully SERIAL process while a concurrent qwen-bench (separate process) loads
the GPU — rotating victims on untouched kernels, no shader-validation OOB,
3/3 green plus a full serial suite immediately after the bench exits. The
corruptor is NOT intra-process test parallelism; suspect a driver/multi-client
issue or a latent timing-sensitive race exposed by contention. Methodology
update: correctness gates require a quiet box (no concurrent GPU processes);
the hunt's next probes should include a cross-process load generator as the
reproducer instead of parallel tests.
v0.440 corrects the v0.438 replay timing confound: timed real-window repetitions
now reset `x`, KV, and GDN state from an immutable prepared seed before every
warmup/timed rep. Corrected S8 blocks=2 is still alive but much narrower:
validated net `~10.8-12.8%` before broader fallback and `~4.1-6.1%` after the
v0.438 `3e-4` fallback packet (`1/15 = 6.67%`). S6 averages only `~4.0%` before
fallback and becomes negative after it; S4 stays negative. `replay_economics.py`
now parses real-margin timing rows and can apply simple active-slot occupancy
mixes or FIFO request traces for p95 modeling. Update the live gate again: replay
is shadow-model only, S8-only, and must clear `>=5-8%` blended net on a real
occupancy/p95 trace before any production scheduler work.
v0.441 makes that gate executable: `replay_economics.py` now accepts active-slot
occupancy traces and FIFO request traces, reporting blended wall, throughput, p95,
and observed occupancy under an S8-only policy. The smoke trace is deliberately
best-case full occupancy and only reproduces the expected `4.85%` fallback-adjusted
win. The live replay branch is now blocked on a real request/occupancy trace, not
more replay microbench rows.
v0.449 rechecks that S8 policy at longer real-prompt contexts (`8192/16384`) with
eight independent prompt files and `blocks=2`, windows `0..2` and `20..22`. All
four rows have zero fallback slots at `3e-4` and validated net saves
`11.64-14.84%`; full-S8 occupancy economics blend to `12.85%`. This keeps S8
alive for long-context serving-style work, but does not remove the production
request-trace/p95 gate. True S8 `ctx32768` is currently corpus/harness-blocked
because only three local independent files exceed `32k` tokens. v0.465 removes
that harness blocker for GDN-only windows by letting one long prompt file supply
nearby slots at `context + slot * stride` with incremental prefix prep. A3B
`chaos.json` ctx32768/S8/stride1, `blocks=2`, windows `0..2` and `20..22`, has
zero fallback and validated net saves `14.41%`/`17.78%`; full-S8 occupancy blends
to `16.09%`. This promotes replay to a minimal S8-only shadow-policy branch, not
a scheduler/default branch: still require real or captured ragged occupancy,
fallback rate, replayed-token share, blended `>=5-8%` net wall save, and p95
non-regression before production scheduler work. Do not generalize this row to
independent-prompt diversity or attention-containing slices. v0.466 makes that
shadow gate executable by adding `--slot-counts` to the real-margin harness and
replayed step/token shares to request simulation. A3B `chaos.json` ctx8192 with
S1/S2/S4/S8 shows the expected policy shape: S1/S2 are strongly negative, S4 is
flat/negative, and S8 nets `7.51%`/`10.51%` on windows `0..2`/`20..22`; synthetic
request sims save `9.01%` saturated and `6.07%` on a ragged burst. The ctx16384
packet is weaker (`-3.13%`/`14.27%` S8, with the negative window repeating at
`7.25%` in an S8-only rerun), and the ragged synthetic sim saves only `3.73%`.
So replay remains live but not promoted: the next artifact must use real or
captured request traces plus robust repeats, not more full-occupancy-only rows.
Keep S8-only/GDN-only; do not build attention-slice support or runtime scheduler
plumbing until the ragged `>=5-8%` net and p95 gate clears. v0.467 adds p50 wall
columns to the real-margin timing rows so near-threshold packets can distinguish
average-wall noise from a real validation overhead loss. Keep the gate on average
wall/request p95 unless a later artifact explicitly justifies a p50 policy metric.
v0.468 adds `request_trace_from_game.py`, a provenance-labeled scenario builder
from game transcripts. It improves stress coverage with real transcript
completion lengths, but modeled `burst`/`fixed-gap` arrivals remain synthetic:
fixed-gap 500 ms scenarios save `8.13%` at ctx8192 and `5.02%` at ctx16384, while
burst scenarios save `8.76%`/`5.42%`. This does not satisfy the empirical
arrival/request gate. Replay is now explicitly parked as blocked on real/captured
arrival traces; do not add scheduler, attention-slice, or more replay kernel work
until that evidence exists. Scenario traces may be used only to keep the economics
tooling honest.
v0.450 adds a narrow current-HEAD qwen-only guardrail after the v0.444-v0.449
churn: A3B Q4 long decode remains `101.7/99.3/93.5 t/s` at
ctx `8192/16384/32768`, A10B Q4_XL `tg128` is `45.62 t/s`, and dense 27B Q4 is
`244.31` pp512, `224.93` pp4096, `23.51` tg128. No fresh guardrail regression;
do not burn a full paired family sweep until a branch changes a primary row.
v0.451 adds trace-only request plumbing: `qwen -p ... --trace-request PATH`
appends FIFO-compatible request rows, and `replay_economics.py --request-trace`
normalizes first-arrival time so real epoch timestamps simulate correctly. The
single-request smoke necessarily shows occupancy `1:2` and no replay save; this
does not promote replay, it only makes the real occupancy/request gate collectable.
v0.442 gives the cross-turn prefix/session cache its product-shaped probe. The
old H2 harness used a per-token prefill loop and overstated the benefit; the bench
now defaults to packed cold/prefix prefill and auto-selects per-token suffix
prefill for short cache-hit suffixes. A3B results: prefix 64 `1.27x` (fails 2x),
256 `1.92x` (fails 2x), 1024 `5.28x` (passes 5x), 4096 `18.47x` (passes 5x),
restore `3.0-7.3 ms`, exact greedy agreement. Product conclusion: cache repeated
1K+ prefixes aggressively, but do not sell prefix cache as a small-prefix win.
Next cache work is runtime/CLI integration plus bounded memory policy, not kernel
work.
v0.445 lands that runtime boundary: `LoadedModel` owns a bounded cache with
in-process model/tokenizer compatibility fingerprints, stats, resize/clear
controls, and exact restore helpers; `Sequence` can snapshot/restore through the
runtime wrapper with explicit identity/capacity checks; `qwen-bench prefix-cache`
now exercises this path. A3B runtime probes reproduce the v0.442 economics:
prefix 1024 `5.16x`, prefix 4096 `18.51x`, restore `4.1/7.4 ms`, exact greedy
agreement. Remaining cache work is request/CLI product wiring, observability, and
cross-process/persistent identity policy; the kernel gate is closed.
v0.448 adds the smallest CLI product seam: `qwen -m MODEL -p PROMPT -n TOKENS`
now runs a runtime-backed greedy single-turn path with packed prefill and
state-coherent decode. v0.469 adds the next narrow product seam:
`qwen --requests-jsonl FILE` keeps one `LoadedModel` resident, accepts explicit
per-request cache-prefix lengths, emits JSONL completions plus optional cache
stats, and demonstrates a 0.8B Q4 `1024`-token exact-prefix hit (`1629`-token
prompts) moving model-internal TTFT `299.8 -> 150.2 ms` with `1.75 ms` restore
and a `32.2 MiB` snapshot. Scope this narrowly: exact in-process token-prefix
reuse for repeated 1K+ prefixes. It is not streaming, concurrent serving,
cross-process persistence, automatic prompt-prefix discovery, or a user-visible
first-byte claim yet. Next cache work should collect real request traces with
hit-rate, prefix-length distribution, p50/p95, memory, and eviction stats before
building admission/discovery policy. v0.470 adds
`scripts/profile/prefix_cache_stats.py` as the reducer for those stats; use it on
real request captures before widening cache policy. v0.472 enriches the stats
schema with arrival/finish timestamps plus prompt/cache/matched-prefix hashes and
adds stdin JSONL (`--requests-jsonl -`), so the next cache artifact should be an
actual resident-process trace packet rather than another synthetic two-request
smoke. v0.473 adds paired `--compare BASELINE CANDIDATE` stats; use no-cache and
explicit-cache runs over the same request ids as the promotion unit, because a
single faster hit can still lose after cold insert and memory costs. v0.482
removes the manual-prefix tax for JSONL request packets by defaulting an automatic
repeated-prefix admission policy at `>=1024` tokens
(`--cache-prefix-auto-min-tokens 0` disables). The policy scans exact token
prefixes across the packet, picks one prefix per request by
`prefix_len * future_hit_count`, preserves explicit per-request/CLI overrides, and
records `cache_prefix_source`, `auto_cache_prefix_tokens`, and
`auto_cache_future_hits` in stats; stdin JSONL remains streaming and does not use
lookahead auto admission. A 0.8B release file-JSONL smoke with three requests and
`1260` shared tokens improves model-TTFT sum `511.4 -> 390.9 ms`, while the tiny
first-insert packet still worsens p95; this is the expected product shape, not a
full promotion. Next cache work is real request traces with no-cache versus
auto-cache paired stats, p50/p95, memory, and eviction behavior. If real traces
lack repeated `1K+` prefixes or p95 worsens after insert/eviction, park cache
policy and move back to S8 replay or chunked GDN.

0. Dense all-quant prompt guardrail: v0.347 found a blind spot in the old
   scoreboard. Static fast-path coverage was clean across 52 local Qwen GGUFs,
   but 0.8B Q4_0/Q4_1 and F16/BF16 `pp512` were still catastrophic because their
   dense prefill mat-mat lowering used scalar row/query kernels instead of the
   simdgroup-matrix shape used by the K-quants. The new Q4_0/Q4_1 legacy MM and
   F16/BF16 half/bfloat-activation defaults close the `5.8-14x` qwen-side cliffs
   with model-level drift gates (`logits_cos >= 0.999993` on BF16, exact-looking
   cosines on F16/Q4). The clean post-fix all-quant spot is green/parity across
   all 12 local 0.8B quants at `pp512` and `tg128`. Keep an all-quant 0.8B paired
   spot as a recurring guardrail before claiming broad quant wins; do not trust
   dtype coverage alone. A follow-up BF16 dense spot shows 4B/9B pp512 still
   `0.93-0.95x` llama.cpp, so BF16 is demoted from catastrophic to a small
   dense-prefill tile-shape gap. A dirty F16/BF16 direct-store epilogue proof
   regressed 4B/9B BF16 pp512, so do not chase that copyout without counter
   evidence. v0.351 broadens the dense guardrail to local 2B/4B/9B quants:
   non-BF16 prompt rows are all `0.98-1.04x`, decode is all green/parity, and
   BF16 4B/9B pp512 remains the only small dense quant softness. v0.352 runs the
   MoE quant guardrail. Supported quantized prefill is green/parity except BF16,
   but MoE decode now has three concrete quant gaps: A3B Q6_K/Q8_0 decode cannot
   enter the routed gate/up path, and A3B Q3_K_M/IQ4_XS decode is behind
   llama.cpp (`0.82x`/`0.75x`) despite prompt wins. Treat Q6/Q8 support and
   Q3/IQ4 decode attribution as the live quant-coverage branch; do not return to
   dense BF16 tile-shape work unless it repeats worse or gains a phase mechanism.
   v0.353 attributes Q3/IQ4 decode to routed gate/up and defaults a decode-only
   fused IQ3_XXS/IQ3_S SwiGLU kernel. Rollback is
   `QWEN_DECODE_MOE_IQ3_FUSED_SWIGLU=0`. Q3 `tg128` moves
   `66.54 -> 72.62 t/s` and IQ4 moves `60.50 -> 67.46 t/s`; routed gate/up drops
   `4.93 -> 3.59 ms` on Q3 and `6.39 -> 4.60 ms` on IQ4. Reusing grouped prefill
   IQ3 SwiGLU for one-token decode regressed, so do not reopen that shape. The
   remaining low-bit MoE decode work should be either Q6/Q8 gate/up coverage or
   a deeper decode-native IQ3/IQ4 dataflow proof, not another all-expert grouped
   prefill transplant. v0.354 clean low-bit spot confirms the external row moved:
   Q3_K_M `tg128` is now `0.90x` llama.cpp and IQ4_XS is `0.84x`, while their
   `pp512` rows remain green and Q4_K_M decode remains `1.41x`. These rows are
   still red but no longer first-order catastrophic; continuing low-bit work now
   requires a second concentrated bucket, not generic quant polishing. v0.355
   kills the simple IQ4_XS routed-down fused weighted-sum sidecar: Q3 `tg128`
   regressed `72.62 -> 70.78 t/s`, IQ4 regressed `67.46 -> 65.99 t/s`, and the
   routed-down phase worsened `2.13 -> 2.54 ms`. Do not reopen IQ4 down final-pass
   fusion without a deeper dequant/work-unit change or counter signal. v0.358
   closes the Q6_K MoE decode coverage hole with a decode-native fused routed
   SwiGLU kernel: clean A3B Q6_K `tg128` is `99.79 t/s` versus pinned llama.cpp
   `79.53 t/s` (`1.25x`), and the Q4 guard remains `107.00 t/s`. Q8_0 is now the
   remaining unsupported MoE decode quant because it still needs both routed
   gate/up and routed down coverage. v0.360 closes that Q8_0 coverage hole with
   native routed SwiGLU and routed-down weighted-sum kernels: clean A3B Q8_0
   `tg128` is `90.28 t/s` versus pinned llama.cpp `73.02 t/s` (`1.24x`), and the
   Q4 guard remains `106.96 t/s`. The quant branch now returns to Q3/IQ4 low-bit
   performance and recurring all-quant guardrails rather than unsupported Q6/Q8
   coverage. v0.362 cracks the Q3/IQ4 routed gate/up bucket by defaulting fast
   IQ3_XXS/IQ3_S decode SwiGLU kernels based on the dense IQ3 row-reuse dataflow.
   Clean A3B Q3_K_M `tg128` is now `89.34 t/s` versus pinned llama.cpp
   `81.50 t/s` (`1.10x`), and UD-IQ4_XS is `89.00 t/s` versus `80.71 t/s`
   (`1.10x`); Q4 guard remains `107.49 t/s`. The mechanism is concentrated:
   Q3 routed gate/up drops `3.58 -> 1.07 ms`, and IQ4 drops `4.59 -> 1.12 ms`.
   Low-bit MoE decode is now externally green; remaining low-bit work should
   target routed-down dataflow only if it changes the work unit or appears as a
   hardware-headroom row, not as a llama-parity panic. v0.364 then proves that
   routed-down dataflow branch by defaulting a dense-IQ4-style fast IQ4_XS down
   kernel. Clean A3B Q3_K_M `tg128` is now `103.27 t/s` versus pinned llama.cpp
   `81.54 t/s` (`1.27x`), and UD-IQ4_XS is `102.36 t/s` versus `80.46 t/s`
   (`1.27x`); Q4 guard remains `107.26 t/s`. Routed down drops `2.13 ->
   0.64 ms` on Q3 and `2.12 -> 0.65 ms` on IQ4. Low-bit MoE expert-bank decode
   is no longer the local bottleneck; next quant work should be a broad guard or
   hardware-headroom row, not more A3B low-bit coverage. v0.365's clean A3B
   all-quant spot guard keeps every measured local quant green at both `pp512`
   and `tg128`: Q3/Q4/Q6/Q8/IQ4 are `1.05-1.08x` for prompt and `1.25-1.39x`
   for decode. Retire A3B low-bit coverage from the active queue; keep it as a
   recurring guardrail only.

1. Decode long-context MoE FFN down/execution shape: v0.321 makes A3B Q4
   `ctx8192` a hardware-headroom row, not just a llama comparison row: MoE FFN
   apply is the largest named phase (`2.73 ms`, `24.5%`) and active-weight BW is
   only `243.8 GB/s` overall. v0.322 splits that FFN bucket: production-wave
   gate/up is `1.29 ms`, down wave is `1.40 ms`, finalizer is `0.17 ms`; the deep
   split puts routed Q5_K down at `1.27 ms` / `184 GB/s` and shared Q8_0 down at
   `0.48 ms` / `93 GB/s`. Simple exits are killed: Q5 fused-down rollback is
   slower (`93.0 -> 92.3 t/s`), dirty Q5_K `NSG={4,1}` fails the phase gate, and
   Q8_0 lcpp mat-vec rollback regresses badly (`80.2 t/s`). v0.323 also kills
   naive threadgroup staging of the Q5_K down inner vector (`93.0 -> 88.1 t/s`),
   so repeated inner reads are not removable by a simple copy/barrier wrapper.
   v0.324 confirms the pattern on A10B Q4_XL `ctx8192`: MoE FFN apply is
   `9.31 ms` / `34.8%`, down wave is `4.90 ms`, and routed Q5_K down is
   `4.58 ms` / `182 GB/s`. v0.327 re-anchors the clean same-build harness row:
   default is `42.4 t/s`, routed-down no-op is `46.0 t/s`, and Q5 fused-off is
   `41.8 t/s`. The next
   credible MoE FFN branch must change the Q5 down work unit more deeply or bring
   counters proving the down wave is compute/dequant-bound. v0.325 kills a Q5_K
   R2 two-output-row work-unit probe: same-build A10B `ctx8192` default/rollback
   rows were `41.9/42.0 t/s`, and rollback split down was slightly faster
   (`2.75 ms` versus `2.86 ms`). v0.333 rechecks both MoE targets after the prefill
   refresh: A3B `ctx8192` default/down-noop/fused-off is `93.1/100.4/91.8 t/s`,
   and A10B is `42.4/46.1/41.3 t/s`. Phase split still puts MoE FFN apply first
   (`25.3%` A3B, `29.3%` A10B), but the local Q5-down shelf remains exhausted.
   v0.341 kills the simplest execution-granularity/overlap proof: a two-stage
   expert pipeline was exact but regressed A3B/A10B `tg128`. v0.342 true-long
   rerank shows routed down is not the whole `ctx32768` story: A3B attention is
   the largest phase (`5.49 ms` / `34.7%`) while routed down is `1.65 ms`, and
   A10B splits between attention (`8.22 ms`), GDN qkv+z+out (`8.18 ms`), and MoE
   FFN. Keep Q5/MoE as a tooling/counter branch, not a local-kernel branch. The
   next branch should bring counters or change byte movement materially rather
   than retuning the same Q5 down kernel or adding split waves. v0.343 adds the
   fast primitive gate: `qwen-bench moe-down-micro` times all Q5_K routed-down
   expert banks without a long ramp. It matches the attribution scale and shows
   A3B Q5 down at only `183 GB/s` for one token, improving to `236 GB/s` at
   synthetic `tokens=16`, while A10B is already `325-354 GB/s`. This keeps A3B
   Q5 down real but points away from command overhead and toward the small-shape
   work unit/dequant dataflow. v0.344 cracks that specific small-K issue: the
   default Q5 down kernel used only two of four K-lane groups at `f_exp=512`, and
   the new R2 kernel computes two output rows per simdgroup. Rollback is
   `QWEN_DECODE_MOE_Q5_DOWN_K512_R2=0`. A3B default-vs-rollback improves
   `ctx128` (`106.6/104.7 -> 108.2/109.8 t/s`), `ctx1024`
   (`100.7/100.6 -> 103.7/104.0`), and `ctx8192`
   (`93.8/94.0 -> 98.0/97.7`, `+4.0-4.5%`). This branch is now banked for
   `f_exp=512`; future Q5 work should target new shapes or broader dataflow, not
   reopen NSG/R2 variants for the same kernel. v0.345's fresh llama.cpp b9833
   guard keeps MoE decode green externally (`A3B tg128 1.42x`, `A10B tg128
   1.25x`), so continue this lane only for hardware-headroom/dataflow wins, not
   parity panic. v0.346 kills the analogous Q8_0 K512 row-widening idea for MoE
   shared down: the sidecar improved the named phase but regressed A3B `tg128`,
   so do not promote shared-down-only micro/phase wins without an end-to-end gate.
   v0.371 kills the closest A10B analog, a dirty Q5 `f_exp=1024` row-pair down
   sidecar. It is exact and helps batched `tokens=16` micro (`36.04 -> 31.95 ms`),
   but single-token decode only moves `2.483 -> 2.424 ms`, far below the phase
   gate. Future Q5 down work needs a different dataflow/counter signal, not
   another row-packing variant. v0.373 splits the route bucket: A10B `ctx8192`
   route logits are `0.48 ms`, while top-k/shared preparation is `0.86 ms`; A3B
   `ctx32768` splits `0.30/0.68 ms`. Do not reopen router-logits mat-vec work,
   but keep post-logits route/slot-prep layout as a measured secondary branch if
   a deeper split or cached-route lower bound shows a recoverable `>=0.15 ms`
   on A10B or `>=0.10 ms` on A3B. v0.374 adds the unsafe route-noop lower-bound
   diagnostic and it clears that gate: A10B `ctx8192` GPU moves `22.71 ->
   20.37 ms`, while A3B `ctx32768` moves `10.82 -> 9.32 ms`. The phase delta is
   route removal plus consumer effects (`routed gate/up` improves `3.18 ->
   2.41 ms` on A10B), but a dirty sorted-topk sidecar regresses, so slot order
   alone is falsified. The next route branch must be exact route-cache replay or a
   top-k/shared-gate split; do not jump straight to a production route kernel.
   v0.375 adds that top-k/shared-gate split: separate shared-gate is far slower
   than the fused route kernel (`1.50 ms` A10B and `1.04 ms` A3B versus fused
   topk/shared `0.86/0.68 ms`), so production should keep shared gate fused. The
   remaining route kernel hypothesis is a fused simdgroup-local top-k or exact
   route-cache replay; do not build a split shared-gate route path. v0.376 kills
   the simple fused SG top-k version: it is correctness-safe in an A3B smoke but
   regresses fused route topk/shared to `1.25 ms` on A10B and `1.00 ms` on A3B.
   Route now needs exact route-cache replay before more route kernels.
   v0.377 adds a CPU-route phase diagnostic that removes GPU route while writing
   CPU-computed route buffers. It saves roughly the route bucket (`23.85 ->
   22.63 ms` A10B, `11.82 -> 10.90 ms` A3B), but routed gate/up worsens, so it is
   not exact replay. Keep it as infrastructure; pivot route implementation work
   to exact GPU replay later and move active optimization back to larger buckets.
   v0.385 adds exact GPU route replay for phase mode: the real route kernels still
   populate top-k/weight/shared buffers, but their time is excluded from the phase
   sum. This keeps consumers stable while showing exact recoverable budget: A10B
   ctx8192 phase sum moves `23.98 -> 22.62 ms` with routed gate/up/down unchanged,
   and A3B ctx32768 moves `11.76 -> 10.75 ms` with consumers unchanged. Production
   route work is now justified, but only for a post-logits/top-k/shared route
   design that saves `>=0.25 ms` A10B or `>=0.15 ms` A3B and moves end-to-end
   decode without perturbing MoE consumers. v0.386 kills the lowest-risk
   production attempt: a dirty exact fused top-k/shared barrier-diet sidecar only
   moved A10B ctx8192 `moe route topk/shared` `0.86 -> 0.82 ms` and phase sum
   `23.98 -> 23.92 ms`, far below gate. Do not reopen barrier-only route kernels;
   the next route implementation needs a materially different top-k work shape or
   a new counter signal. v0.387 kills the obvious candidate-compression variant:
   exact simdgroup-local candidate lists plus a single-thread merge regressed
   A10B `moe route topk/shared` `0.86 -> 3.87 ms`. Together with v0.376 and
   v0.386, this closes local top-k rewrites as the next route bet unless new
   counters identify a different mechanism. v0.390 repeats the exact route budget
   on current code but demotes exact route implementation: A3B route split is
   logits `0.30 ms` plus fused topk/shared `0.68 ms`, A10B is `0.48 + 0.86 ms`,
   and replay deltas are `1.01/1.34 ms`. A3B deep split shows the existing
   topk/shared fusion is already the important structural overlap (`1.43 ms`
   separate versus `0.68 ms` fused). Reopen exact route only for a prototype that
   saves `>=0.4 ms` A3B or `>=0.5 ms` A10B route total without moving time into
   consumers; otherwise pivot to captured MoE compute work.
   v0.378 adds `moe-gateup-micro` as the next active harness. A10B Q4 gate/up
   micro is close to phase (`3.0399 ms` versus `3.18 ms`), so bounded A10B Q4
   gate/up shape probes can use it. A3B synthetic top-k is not representative yet
   (`1.5179 ms` micro versus `1.07 ms` phase), so A3B promotion still requires
   captured route-pattern replay or full phase/ctx confirmation. v0.379 proves
   that guardrail: dirty `NR0_Q4K=4` improves A3B synthetic micro (`1.5179 ->
   1.3430 ms`) but fails full phase (`1.07 -> 1.09 ms`) and regresses/noises A10B.
   Do not continue row-widening without captured route patterns.
   v0.380 bounds the lm-head greedy-fusion shelf: `lm argmax` is only
   `0.06 ms` on A10B and `0.05 ms` on A3B, so exact `lm_head+argmax` fusion is
   below the implementation gate. Treat lm-head as projection-weight-read work,
   not sampler overhead.
   v0.381 recalibrates current decode slopes: A3B drops `107.8 -> 88.7 t/s` from
   `ctx128` to `ctx32768`, A10B drops `45.2 -> 40.7 t/s` from `ctx128` to
   `ctx16384`, and dense 27B drops `25.6 -> 23.2 t/s` from `ctx128` to `ctx8192`.
   No new cliff appears; use phase-budgeted structural branches, not more local
   route/gateup row tweaks. v0.382 adds captured hidden+route replay to
   `moe-gateup-micro`, which makes the harness phase-faithful on A3B ctx8192
   (`1.0836 ms` replay versus `1.08 ms` phase) and A10B ctx8192 (`3.1147 ms`
   replay versus `3.19 ms` phase). It also rejects the known dirty NR4 false
   positive (`1.1014 ms` captured versus `1.0836 ms` default). Use captured
   replay as a gate for future Q4 gate/up variants, but do not keep mining
   row-width tweaks without a structural byte/dataflow rationale. v0.475 kills
   the adjacent slot-packing work-unit proof: packing two routed Q4 gate/up slots
   into one 4-SG threadgroup is not better on captured A3B ctx8192 replay
   (`1.0207 -> 1.0317 ms` GPU) and was much worse on synthetic replay. Do not
   reopen Q4 gate/up slot-pair/row-width shapes unless the proposal changes expert
   dataflow or bytes, not just threadgroup packing.
   v0.391 extends the same captured-route discipline to routed down and makes the
   down microbench production-faithful: A3B Q5 down captured R2 is `0.814 ms`
   versus phase `0.82 ms`, and A10B captured Q5 down is `2.595 ms` versus phase
   `2.56 ms`. Future MoE compute variants should clear captured gate/up or down
   before phase promotion; synthetic-only wins remain non-promotional.
2. Decode long-context attention/KV second pass: v0.293 proves A3B group8 decode
   attention still had high-EV execution-shape headroom (`ctx16384` attention
   `4.80 -> 3.60 ms`, throughput `72.8 -> 82.2 t/s`). Attention remains large in
   v0.321 (`2.28 ms`, `20.5%` at A3B `ctx8192`), but prior KV-byte-only variants
   are falsified: v0.299 kills the straightforward MoE Q8-KV subgroup reader,
   v0.300 kills smaller subgroup splits, v0.313 kills cooperative four-simdgroup
   subgroup packing, and v0.315 kills simple group-tile/reread reduction. The
   v0.333 keeps attention/KV as a measured secondary lane at `21.3%` A3B and
   `18.3%` A10B `ctx8192`, behind MoE FFN on both. The next credible proof must
   change execution shape without losing occupancy, or
   show a counter signal beyond byte count/address order. Do not reopen subgroup
   shape, `NWG`, `TILE_C`, ggml-Q8 KV, or broad KV-head-major layout without new
   data. v0.334 banks a small epilogue cleanup by fusing gated-attention
   `sigmoid+mul`: A3B `ctx8192` moves `92.1/92.2 -> 93.6/93.3 t/s`, A10B is
   neutral/noise, and a dense 27B `ctx8192` guard moves `21.9 -> 22.6 t/s`. Treat
   this as a shelf win; the remaining attention branch is still the v4 body/KV
   path, not more scalar epilogue passes. v0.335 then falsifies the most concrete
   layout-only KV proof: synthetic head-major F16 K/V is exact but flat/slower at
   A3B `ctx16384/32768` and A10B `ctx8192/16384/32768`. Do not build production
   head-major KV sidecars without a new counter signal or a body rewrite that
   changes more than address order. v0.342 promotes attention as the true-long
   scaling limiter but demotes a same-byte rewrite: `qwen-bench attn-intra` puts
   the v4 main body at `623-649 GB/s` estimated KV bandwidth on A3B and
   `533-583 GB/s` on A10B, with reduce only `0.03-0.05 ms/layer`. Reopen the
   attention body only for byte reduction, hidden-traffic counters, or a
   full-model `ctx32768` prototype that moves throughput despite those body
   numbers.
    v0.356 re-anchors current A3B true-long decode after the low-bit fixes:
    Q4_K_M drops `107.7 -> 96.5 -> 76.2 t/s` at `ctx128/8192/32768`, and
    `ctx32768` phase split puts attention first (`5.09 ms`, `38.0%`). The
    `attn-intra` body matches that scale and estimates `~692 GB/s` on the main
    pass, while reduce is only `0.34 ms` extrapolated. A dirty group8 Q8-KV
    sidecar regressed badly (`ctx8192/32768` `96.5/76.2 -> 85.5/55.8 t/s`) and
    forced F16 full-group tile8 also regressed (`89.8/62.0`), so do not reopen
    Q8-KV by giving up the default group8 tile2 execution shape. The live
    long-attention branch must reduce bytes while preserving occupancy, bring
    counters for hidden traffic, or deliver an end-to-end `ctx32768` prototype;
    same-byte retunes and reduce work stay deprioritized. v0.367-v0.369 bank the
    concrete medium/true-long attention wins, but v0.370 kills the next partial
    byte-reduction proof: normalized-half partials are exact enough, yet only move
    A3B `ctx32768` `attn-intra` `1.014x`. v0.388 refreshes current `ctx32768`
    `attn-intra` knobs and keeps the default tile4/NWG256 path best: default is
    `0.3469 ms/layer` (`3.47 ms` extrapolated), while tile2/NWG64 is
    `0.5265 ms`, tile8/NWG256 is `0.3971 ms`, and tile4/NWG128 is `0.3810 ms`.
   `NWG=128` halves reduce but slows the main body; tile8 reads fewer logical
   bytes but loses occupancy. Existing group-tile/NWG retunes remain closed.
   v0.476 reopens the lane with a real body/dataflow change instead of another
   selector retune: the G8 C64 score-broadcast sidecar keeps score/weight state
   lane-local and uses `simd_shuffle` during PV, deleting the main-pass
   score/weight TGM round-trip while preserving F16 KV and the reduce contract.
   It is exact and moves A3B main `0.1076 -> 0.1019 ms` at ctx8192 and
   `0.1948 -> 0.1755 ms` at ctx32768; full ctx-sweep is neutral at 4096 and
   positive at ctx32768 (`86.5 -> 88.9 t/s`) but below the `>=5%` default gate.
   Keep `QWEN_ATTN_V4_G8_BCAST=1` as an opt-in true-long sidecar and recheck on
   real long rollouts before promotion. v0.477 kills a dirty attempt to broaden
   the same score-broadcast body to G16/tile4/C64 because correctness failed in
   non-target group6 rows. v0.478 fixes that implementation issue and proves the
   G16 body exact, but the full A10B decode gate still fails: attention-intra
   improves directionally while `ctx8192 --window 4 --fresh-per-checkpoint`
   regresses `42.5 -> 41.4 t/s`. Do not keep a G16 bcast env knob or reopen this
   exact shape. v0.480 then tests a more literal read-once-ish G16 shape: one
   4-simdgroup TG stages half of V at a time to cut tile traffic from roughly
   `K4 + V4` to `K4 + V1`. It is exact, but A10B `attn-intra ctx8192` regresses
   badly (`0.4340 -> 0.5614 ms`, main `0.1274 -> 0.2231`). Together with the
   v0.463 A3B V-stage kill, close TGM-staged V/KV variants unless counters prove
   read savings beat synchronization/TGM/occupancy loss. Keep attention below
   MoE/GDN unless the next proposal brings hidden-traffic counters, a non-TGM
   read-once execution shape, or an end-to-end `ctx32768` prototype that moves
   throughput rather than partial storage or reduce rows. v0.481 also kills a
   dirty dense group6 bcast broadening attempt at correctness: the added
   specialization disturbed non-target group4 rows (`cos=0.931786`). Do not keep
   broadening the G8 bcast template across groups without a narrow harness and a
   clean broad-suite pass.
3. GDN decode projection mechanics, with local Q8 retunes closed: v0.336 adds
   correctness-breaking no-op attribution for the GDN projection lane. The
   recoverable lower-bound budget is
   large at `ctx8192`: A3B front/out no-op moves GPU `10.19 -> 8.41/9.58 ms`, and
   A10B moves `23.06 -> 18.38/20.88 ms`. The subprojection ladder convicts QKV,
   Z, and OUT, while beta/alpha are flat/noise. Do not spend the next branch on
   beta/alpha fusion. The next credible implementation is an exact-shape Q8
   projection microbench for `h -> conv_dim`, `h -> v_dim`, and `v_dim -> h`, with
   a required primitive win before production decode changes. v0.337 adds an
   aggregate A10B `ctx8192` phase split: QKV is `2.98 ms` / `12.2%`, Z is
   `2.04 ms` / `8.3%`, beta+alpha are only `0.45 ms` combined, and OUT is
   `2.29 ms` / `9.4%`. A correctness-safe Q8_0 R4 row-widening sidecar failed the
   gate (`92.6/94.0 -> 92.5/91.8 t/s` on A3B and `42.6 -> 42.0 t/s` on A10B), so
   do not retread Q8 row-count tweaks without counter evidence. The live Q8 branch
   must change projection dataflow, fusion, or packing; otherwise return to MoE
   FFN execution shape or attention/KV body work. v0.338 adds the exact-shape
   `qwen-bench gdn-proj-micro` harness and shows the current primitive is already
   near the measured stream roofline on weight bytes alone: A10B QKV/Z/QKV+Z/OUT
   are `485/473/476/461 GB/s` versus the `474 GB/s` stream anchor, while A3B is
   `435/418/445/395 GB/s`. This demotes local GDN Q8 mat-vec retunes; reopen only
   for structural byte reduction or a primitive proof that beats this harness.
   v0.340 banks the known dense scheduling overlap instead: default dense
   concurrent-GDN moves `tg128` by `+2.4-3.8%` across 0.8B/2B/4B/9B/27B. v0.342
   keeps structural GDN byte reduction alive for A10B long decode because
   qkv+z+out totals `8.18 ms` / `26.7%` at `ctx32768`, but the branch must reduce
   bytes, fuse a larger dataflow, or prove a primitive win; do not spend another
   pass on row-count retunes. v0.372 splits the GDN tail and demotes local tail
   work: A10B `ctx8192` tail step is only `0.69 ms`, and A3B `ctx32768` tail step
   is only `0.34 ms`; conv, L2, and norm are smaller. GDN remains a projection or
   structural byte/dataflow branch, not a tail microkernel branch. v0.383 refreshes
   that projection gate on current HEAD: A10B `qkv+z` micro runs `2.4065 GB` in
   `4.8675 ms` (`494.4 GB/s`) and A3B runs `0.8022 GB` in `1.7949 ms`
   (`446.9 GB/s`), while A10B beta+alpha projections are only `0.39 ms` in the
   ctx8192 split. Do not build local GDN projection row-shape or `qkv+z` fusion
   branches without byte elimination, an algorithmic dataflow change, or counter
   evidence beyond weight streaming.
4. A10B memory-capacity/tooling hygiene: v0.315 shows a `ctx-sweep` that allocates
   for `32768` up front can poison even A10B `ctx570/2464` rows (`~0.6 t/s`), while
   capped sweeps are normal (`43.8/38.4/43.1 t/s` through `4096`, `41.6/39.9 t/s`
   at `8192/16384`). v0.316 adds `ctx-sweep --fresh-per-checkpoint`, which
   validates A10B `ctx570/2464` at `43.5/38.5 t/s` with right-sized sessions.
   v0.317 adds `decode --kv-capacity` and shows large unused capacity is mainly a
   cold first-touch/residency hazard: A10B `ctx570 cap32768 --no-warmup` has a
   `3089 ms` first decode token, but warmed `cap32768` returns to `43.3 t/s`.
   Use fresh/right-sized modes for measurement, keep capacity explicit in product
   benches, and handle cold-start residency separately from hot kernel tuning.
5. Utilization scoreboard plumbing, bundled with long-context decode: v0.285 adds
   the one-time roofline calibration packet (`474 GB/s` stream,
   `~12.5-12.8 nominal TFLOP/s` Q4_K mat-mat, and `3.03 TFLOP/s` scalar-FMA
   sanity), v0.293 adds `scripts/profile/decode_phase_roofline.py` for phase-level
   active-byte reads, v0.312 adds optional tg JSON command/encoder/dispatch
   accounting, v0.315 adds decode attention KV byte estimates, and v0.321 adds
   active decode weight bandwidth from bench JSON or manual context-sweep rows.
   v0.326 adds `scripts/profile/decode_ctx_sweep.py` for order-aware env-variant
   `ctx-sweep` packets, closing the measurement gap that let v0.325's false Q5
   R2 win survive too long. v0.342 adds visible `qwen-bench attn-intra` so
   attention main/reduce body evidence is not trapped in ignored tests. v0.343
   adds visible `qwen-bench moe-down-micro` for fast Q5 routed-down primitive
   gates before full long-ramp decode sweeps. v0.345 moves the pinned
   llama.cpp lock to b9833 (`c818263f2`) and fixes family digests to use the
   measured `474 GB/s` stream anchor, so future scoreboards should not silently
   drift back to the stale May benchmark or the old `546 GB/s` spec-sheet peak.
   The first A3B Q4 `ctx8192` sample is `93.0 t/s`, `2.6215 GB/token`, and
   `243.8 GB/s` (`51.4%` of measured stream roofline); attention KV subgroup
   traffic is `0.6711 GB/token` at `294.3 GB/s`. Next, keep using this output to
   explain the active branch with decode t/s, wall/GPU time, phase split,
   dispatch/encoder counts where available, and defensible bytes/token or
   bandwidth proxies. This is not dashboard work; it is the decision spine for
   capacity, execution-shape, and roofline calls.
6. Packed-verify/spec decode, structural rewrite only: v0.314 fixes production
   `qwen-bench dflash` scratch allocation and shows real narrative `static-16`
   decode is catastrophically slower than no-spec despite good acceptance
   (`0.349x` at `570` prompt tokens and `0.261x` at `2464`). Treat acceptance as
   interesting but not sufficient. Do not spend time on policy knobs; promote spec
   only if packed verify/KV restore/logits accounting identifies one removable
   structural villain and a proof can plausibly clear `>=1.25x` decode on real
   prompts after full verify cost. Fresh llama.cpp b9833 has a real
   `llama-cli --spec-type draft-mtp` path, but the local smoke does not reset
   priorities: no-spec 27B-MTP generation was `23.0 t/s`, while `draft-mtp`,
   `n_max=3`, `p_min=0.75` was `23.8 t/s` and lower prompt throughput. Keep MTP
   as a separate algorithmic target; do not mix it into no-spec parity boards.
   v0.443 EXECUTES this item's precondition and finds the villain: the skinny-N
   mat-mat kernel family runs `3-6x` off weight-stream on every verify/drafter
   projection shape (`44-136 GB/s` vs mat-vec's `288-382` on identical
   tensors); verify(N=16) is `5.2-5.5x` a decode step and `82%` of the DFlash
   step, drafter `13%`, glue/restore `<6%` (the v0.314-era restore/logits
   overhead is already engineered away). Durable alpha at 256 tokens:
   code `3.34`, narrative-start `2.11`, narrative-tail `0.386` — and the
   adaptive policy fails to protect low-alpha text (`0.524x`). M2 is
   sanctioned in two tracks: (a) skinny-N mat-mat retune (`>=250-300 GB/s`
   at N<=16 on the `packed_verify_skinny_gemm_micro_27b` harness; start from
   the existing N16 simdgroup kernels — scalar row-parallel is ALU-bound at
   `~38 ops/weight` and caps below target), (b) alpha-aware policy cutoff.
   Priced: `~1.4x` mid-alpha / `~2.0x` code decode-only if (a) lands; gate is
   the M1c packet re-run clearing `>=1.25x` on real prompts at 570/2464 ctx.
   Track (a) also serves replay S8 projections and MoE small-batch decode.
   v0.444 then EXECUTES track (a) and FALSIFIES the branch: the deficit is a
   per-dispatch machinery floor (~0.27 ms per skinny GEMM: sa-swizzle stores +
   fragment loads + loop), not bytes — a raw-block-staged v2 kernel wins +24%
   on the hot-L2 micro but ~0% in production verify (occupancy trade), and an
   A-path-FREE probe inside the verify still costs 168.6 ms = 4.2x a decode
   step. All-heroics ceiling (machinery -25%, A-path halved, v0.73b GDN tail
   kernel, packed-N attn) is ~1.0-1.1x at the best measured alpha — under the
   bar. F32 dot-product alternatives are FMA-roofline-capped (~250 GB/s-equiv)
   and half-acc designs die on activation-reuse register/L2 tension. DFlash
   verify-cost work and the M2b policy co-fix are DEMOTED; reopen only with
   (a) durable alpha_chain >= 4.5, (b) a persistent-kernel/fused-layer verify
   structure that amortizes per-dispatch machinery, or (c) MTP-side economics.
   Do not propose N-tile/staging/barrier variants against the N16 family.
7. Promotion-grade paired residual search for prompt prefill: v0.279 cracks the
   tuned small-dense control except for parity/noise 2B `pp512`; current sentinel
   rows keep 27B/A3B/A10B prefill won after discarding A10B cold noise. Reopen
   prefill only if a paired repeat exposes a real current-default red cell. v0.318
   clears the stale A3B true-long row: synthetic `pp34502` is `1.196x` and a real
   `56774`-token chaos rollout is `1.168x` versus pinned llama.cpp. Do not reopen
   fused online-softmax, chunk policy, or GDN-scan work from the old true-long row
   alone.
8. BF16 accounting/differential audit, explicitly scheduled only: BF16 remains a
   catastrophic demoted red cell, but v0.319 falsifies the most concrete
   grouped-MoE scalar A-load hypothesis. Dirty `bfloat4` loads regressed BF16
   grouped `routed_swiglu` (`130.80 -> 217.73 ms`) and `routed_down`
   (`64.78 -> 98.30 ms`). v0.320 then rechecks the existing
   `QWEN_MATMAT_BF16_BFLOAT_ACT=1` sidecar: traced GDN/attention phases collapse
   (`gdn_qkv 1141.65 -> 99.97 ms`, `attn 611.94 -> 69.44 ms`), but no-trace
   production moves only modestly/noisily and the paired row remains `0.059x`
   llama.cpp (`73.63` versus `1239.10 t/s`). Do not default bfloat-act or reopen
   BF16 kernel work until named production accounting explains `>=90%` of BF16
   `pp512` wall, or one identified category is `>=40%` of wall and has a plausible
   `>=1.5x` no-trace production fix. v0.489 independently rechecks grouped BF16
   `bfloat4x4` A loads and again loses on warmed A3B BF16 `pp512` samples
   (`657.3/1373.2` scalar versus `645.0/1371.3 t/s` vector). Do not reapply
   grouped-MoE vector A-loads.
9. Short MoE decode execution-shape, only with fresh evidence: v0.292-v0.311 make
   decode materially greener, but v0.312 says the current default is already
   GPU-active at `~95.8-97.8%` wall on warmed A3B/A10B `tg128`. The GDN-concurrent
   rollback win is real (`+8.8%/+6.0%`) but already banked, and disabling it mainly
   increases GPU active time with the same dispatch count. Reopen only for a new
   GPU-overlap or byte-reduction mechanism with a measured `>=2%` tg ceiling on
   both MoE targets, or `>=3%` A10B with A3B neutral. Do not optimize encoder count
   for its own sake: serial one-encoder rollback is slower.
10. Remaining dense prompt attribution, only if a paired red cell survives: the
   latest phase trace shows `gdn_gated` and `gdn_prep_l2` are now tiny. If short
   dense still regresses in a clean repeat, target projection/FFN, GDN step, or
   attention body with a shape-matched differential. Do not retread attention
   defaults, fast-path coverage, residual-add fusion, N64 policy, shared-memory
   policy, GDN token-channel parallelization, command-buffer streaming, GDN
   matvec fallback, or HD128 normalization row count.
11. Hardware-headroom audit watchlist, gated by primary-row evidence: ICB/MTL4
    encode-once decode, fused online-softmax prefill attention, chunked
    delta-rule GDN, production residency/warm expert banks, RoPE precompute,
    targeted kernel resource annotations, host encode allocation/PSO cleanup, and
    structural decode fusions are all plausible paths to dominate beyond
    llama.cpp. They need a trace-proven idle/bandwidth/phase gate before
    implementation, not merely a green or red llama row. GDN prep falsifiers do
    not falsify chunked `gdn_step`; they are different kernels and mechanisms.
    The specific post-audit gates are:

    - RoPE precompute: replace per-pair `pow` with inv-freq or sin/cos reuse only
      if an isolated kernel win converts into `>=0.5-1%` end-to-end on at least
      one primary decode or prefill row with no drift.
    - PSO cache, `view_subrange`, and clone cleanup: first capture warmed CPU
      encode/alloc evidence showing `>=0.3 ms/token` or `>=1%` wall in a primary
      decode path; current GPU/wall rows do not justify an implementation branch
      by themselves.
    - `max_total_threads_per_threadgroup` and simdgroup-barrier pruning: test only
      on exact hot kernels with primitive microbench wins, then run clean family
      guards. Do not blanket-annotate kernels; a wrong cap can reduce occupancy.
      v0.485 tests the concrete Q4/Q6 mat-mat SG-barrier deletion and kills it:
      correctness is green, but 27B `pp1024` regresses (`235.384 -> 231.967 t/s`)
      while `pp4096` is flat/noise. Do not repeat this without shader-counter
      evidence for a specific kernel.
    - Hot-weight K-quant repack: v0.486 executes the smallest exact compressed
      prepack proof, reordering one 27B Q4_K FFN gate tensor into N16 mat-mat
      consumption order without predequantizing. Correctness is bit-exact, but
      the primitive only moves `0.4078 -> 0.4030 ms/dispatch` (`1.012x`), far
      below the `>=1.12-1.15x` gate for model wiring. Demote broad K-quant repack
      until counters identify a larger layout stall or a different tensor/layout
      has a named primitive win.
    - Decode fusions: prefer byte-reuse/dataflow fusions such as GDN `qkv+z` over
      residual/elementwise shelves, but require a primitive proof that beats the
      current near-roofline projection harness or a phase bucket of at least `5%`
      with an end-to-end `>=1%` win.
    - Mmap no-copy, heaps/residency sets, binary archives, and ICB/MTL4 remain
      product/cold-start/encode-bubble work until traces show hot throughput is
      host- or residency-limited.
    - Kernel-bypass / persistent-megakernel framing is useful as a probe source,
      not a roadmap reset while hot decode remains mostly GPU-active. Measure
      abstraction ceilings directly: command/encoder/dispatch counts, warmed CPU
      encode time, allocation count, and theoretical cycles/token. Only escalate
      ICB, persistent work queues, argument buffers, or GPU-side graph traversal
      if those probes show command starvation, host bubbles, or cross-op data reuse
      that a normal kernel path cannot access. v0.389 proves autonomous hardware
      counters are not currently available here beyond timestamps, so hidden-stall
      or bandwidth-counter claims require manual Xcode GPU capture rather than
      routine `xctrace` or in-process counter sampling.
    - AMX/ANE coprocessor, hierarchical `lm_head`, layer skipping, sparse FFN,
      KV clustering, attention-head skipping, and residual/GDN-state precision
      changes are quality or fabric research lanes. They need explicit quality
      gates and a measured phase ceiling before speed claims.

12. Quant breadth guardrails: v0.240 proves prompt prefill for local 4B
   `UD-Q2_K_XL` and `UD-IQ2_M` at `pp512/1024/4096`, v0.241 proves their
   `tg128` decode sentinels, and v0.242 keeps adjacent 4B `Q3_K_M`, `IQ4_XS`,
   and `Q4_K_M` clean at `pp512/4096/tg128`. Reopen low-bit dense kernel work
   only when a paired file or static audit exposes a fresh miss.
13. `IQ4_XS` grouped-down precision/perf audit: strict internal-state cosine is
   below the usual `0.999` floor on `UD-IQ4_XS`, and F32 gate/up reproduces the
   same envelope. The v0.234 primitive grouped-down oracle passes (`cos=1.0`,
   `max_abs=1.386e-5`), so the next accuracy check needs real captured
   `moe_inner` activations rather than another synthetic row-stride oracle.
14. Bounded llama.cpp / counter attribution: verify whether llama.cpp is actually
   faster inside comparable routed gate/up/down arithmetic, or whether remaining
   differences are orchestration, fused GDN, graph fusion, warm/cold accounting,
   or profile scope. The pinned upstream b9481 build does not expose
   `GGML_METAL_PROFILE_OPS`; the local fork has a profiling patch, but any
   attribution claim needs either a canonical profiling patch or a clearly marked
   non-scoreboard capture before more kernel code.
15. True multi-expert work-unit reset: if attribution proves the gap is inside Q4
   `<8` SwiGLU arithmetic, design a kernel that changes the work unit more deeply
   than R16, MR32, or split gate/up. It must improve `<8` by at least `25-30%`
   before any end-to-end tuning.

The sections below preserve the rationale and reopen criteria from earlier
sprints; the current rank above overrides stale branch ordering when they
conflict.

Current MoE short-branch rule:

- Target active underfilled buckets, not route work or threshold policy. Q4
  `pp512` fine-bin traces show `<8` is only `6.3%` of routed slots but costs
  `3.485 ms/k-slot` in SwiGLU and `3.278 ms/k-slot` in down, versus `>=64` at
  `0.498` and `0.252` respectively. Q3/Q6/Q8 SwiGLU traces show the same shape.
- The next branch should be attribution-first, not another Q4 tiny-SwiGLU retread.
  If attribution proves the residual is inside comparable `<8` SwiGLU arithmetic,
  require a genuinely new multi-expert work unit and gate on bin-time movement
  first, then A3B `pp512` GPU time, then no regression at `pp768/1024+` and broad
  quant generalization.
- Down now has a correctness-safe force-only proof. It reduces the target bin but
  is not enough by itself to justify defaulting. The first analogous SwiGLU port
  is falsified; future SwiGLU work must explain the dual-dequant/epilogue cost
  before adding another tiny kernel. The split gate/up sidecar also failed, so
  the remaining SwiGLU path is a real execution-shape change, not merely
  unfusing the current grouped kernel. The MR32 attempt failed that bar too;
  pause Q4 tiny-SwiGLU code until attribution identifies a new mechanism.

### 1. Hypothesis: Dense 27B residuals have pivoted from attention to GDN/FFN

Optimizes: Qwen3.6 27B dense prompt prefill after the G6 matrix-attention body
received pointer-hoist and full-tile specializations.

Why it is at the top:

- The v0.165 full-tile KQ/KQV kernels change the dense read: at `pp4096`, KQ/KQV
  are now `136/135 ms`; at `pp16384`, they are `2170/2377 ms`, with qwen softmax
  still much faster than llama.cpp. Dense attention body is no longer the obvious
  lcpp-scale residual.
- The remaining large named buckets are GDN/FFN projections and GDN step/state
  work. Earlier projection unrolls helped but did not finish the scoreboard; the
  earlier GDN recurrence-loop unroll was negative, so the next GDN branch must be a
  state/read-write/layout audit rather than another loop pragma.
- A10B routed MoE no longer explains a scoreboard gap under warmed methodology;
  qwen's warmed routed tail is faster/equal to llama.cpp and end-to-end is
  parity-or-better from `pp512` through `pp16384`.
- A3B matrix default plus Q6 coverage already wins most synthetic anchors; true
  long/real-rollout work stays live but is less clean than dense 27B.

Current design rule:

- Keep `QWEN_PREFILL_ATTN_MATRIX_G6=0` as the rollback path and preserve the
  ignored G6 prefix correctness gate plus `16/16` matrix phase coverage.
- Keep the branch scoped to KQ/KQV/softmax body. Projections are not the target
  unless a fresh phase trace says they regressed.
- Preserve `QWEN_PREFILL_ATTN_MATRIX_CAUSAL_SKIP=0` as rollback for the current
  default cleanup.
- Runtime compact-Q staging and q-head-major score/body layout are falsified:
  compact Q reduces KQ/KQV locally but its copy cost overwhelms the win, while
  q-head-major regresses KQ and leaves KQV flat. Revive only if Q is produced in a
  compact layout for free or a true llama.cpp-kernel clone needs the layout.
- Producer-side compact-Q was also tested as the cheapest "free Q layout" upper
  bound by making Q RoPE write the compact KQ view directly. It is correctness-safe
  but only moves KQ/KQV by a few milliseconds at `pp4096` while adding a small
  RoPE/scatter copy cost, far below the phase gate. Demote Q-layout-only work.
- Loop-pragma attention tweaks are falsified. A selective inner-loop-only unroll
  probe improved dirty phase rows but failed the clean gate: tiny/noisy long wins
  and clear `pp512/1024` regressions. Do not carry a duplicate env-gated kernel for
  sub-1% long-context movement.
- The first positive llama.cpp-mechanics cleanup is pointer/base-address hoisting in
  KQ/KQV. It moves 27B phase rows from `179/180 -> 160/158 ms` at `pp4096` and
  `2823/2972 -> 2516/2660 ms` at `pp16384`, with clean rows positive at
  `pp512/1024/16384` and noisy-positive at `pp4096`. Keep this style of isolated
  mechanical diff alive.
- Full-tile KQ/KQV specialization is the second positive mechanics cleanup. It
  moves phase rows to `136/135 ms` at `pp4096` and `2170/2377 ms` at `pp16384`,
  and post-commit clean rows reach `213.80 t/s` at `pp4096` and `200.16 t/s` at
  `pp16384`. Attention is now monitoring/tail-work, not the default top branch.
- GDN step NSG4 row grouping plus token-pointer increments are the first
  post-attention dense cleanups. Together they move `gdn_step` from
  `595.56 -> 482.15 ms` at `pp4096` and `2379.00 -> 1939.31 ms` at `pp16384`;
  clean rows now sit at `237.20/235.23/222.28/203.37 t/s` for
  `pp512/1024/4096/16384`.
- The matched qwen-vs-llama dense differential has been rebased after the GDN
  pointer increment cleanup. GDN step and attention body are now faster than
  llama.cpp in the serialized comparison; the only notable dense residual is a
  long-context FFN projection delta. Follow-up falsifiers did not make that
  residual actionable: chunk `512` loses to default `1024`, reduced mat-mat smem
  is phase-positive but not total-robust, and dense fused-Q4 FFN loses again in
  same-process A/B.
- Fresh v0187 paired comparison changes the dense rule again: 27B dense prefill
  is paired-won at `pp512/1024/4096/16384`, and the apparent short/medium gap was
  a stale-anchor/unpaired-drift artifact. Larger chunks, reduced QK smem, dense
  fused-Q4 FFN, and FFN up-before-gate have all failed promotion gates. Keep Q4
  N64 default-on and move dense 27B to guardrail mode.

Next branch order:

- First, use `scripts/profile/prefill_compare.py` for any future cross-engine
  prompt claim. Stale isolated llama.cpp anchors are no longer enough, especially
  at `pp512/1024/4096` where thermal/session drift can change the conclusion.
  The harness now accepts real `--file`/`--messages` qwen prompts; because
  `llama-bench` cannot consume prompt text, those llama.cpp rows are explicitly
  recorded as same-length synthetic anchors via `lcpp_prompt_mode`.
- Second, return to breadth/generalization: primary family paired guardrails,
  real-rollout prompts, and quant coverage gaps should rank above another dense
  27B microkernel unless a paired residual appears. `scripts/bench/family.py`
  remains the synthetic breadth scoreboard and now records cooldown plus
  thermal/memory context per command; use `prefill_compare.py` repeat blocks for
  promotion-grade narrow cells.
- Third, keep low-bit MoE defaultability tied to continuation/rank behavior rather
  than strict internal-state cosine alone. v0.233 shows `UD-IQ4_XS` final logits
  and continuation pass while internal GDN/KV cosines sit below `0.999`, and F32
  gate/up points at grouped `IQ4_XS` down as the envelope source.
- Fourth, keep MoE quant breadth active only where audit finds a real uncovered
  target file. A3B Q3/Q4/Q6/Q8 and `UD-IQ4_XS` now have `40/40` grouped coverage;
  prefer paired family guardrails and routed-tail attribution over adding kernels
  for quants not present in the target/product set.
- Fifth, small dense remains a paired mismatch but is no longer above MoE quant
  breadth. Start from 0.8B `pp512`
  and require 2B plus 4B/9B/27B canaries before promotion. v0.215 landed the
  low-risk GDN prep dispatch cleanup; recent falsifiers say the next branch
  should target FFN/GDN projection mechanics or dataflow, not attention, encoder
  coalescing/streaming, GDN matvec fallback, NSG8 GDN-step grouping, fused Q4
  SwiGLU N64, or broad low-threshold N64 policy.
- Defer reduced-smem promotion, fused FFN, and fused online-softmax/PV until fresh
  same-process or phase evidence crosses a total-throughput gate.

Acceptance gates:

- 27B `pp512/4096/16384` must improve without decode regression.
- A3B and A10B prompt defaults must remain neutral, with MoE coverage still
  `40/40` and `48/48` respectively.
- Promote only from AC-power rows with no thermal/performance warnings and a
  paired llama.cpp comparison on the same GGUF.

### Monitoring: A10B grouped routed down/dequant locality is not currently active

Optimizes: Qwen3.5 122B A10B prompt prefill after the G16 matrix-attention default
and the layer-46 Q5 gate/up coverage fix.

Current status: superseded by the `v0.155` warmed re-anchor. Keep this section as
the reopen criteria and historical rationale, not as the active top queue.

Why it was at the top / reopen criteria:

- A10B/G16 matrix attention is already default-on for the proven group-16 shape;
  post-default no-op rows made attention body a low-single-digit `pp512` lever.
- The first routed-MoE structural pass found and fixed the layer-46 fast-path escape:
  `Q5_K/Q5_K/Q6_K` gate/up/down now uses grouped Q5 gate/up SwiGLU and grouped Q6
  down, restoring `48/48` grouped routed MoE coverage.
- Warmed rows for that fix are large enough to be real: `pp512` `377.53 -> 448.31
  t/s`, `pp1024` `400.11 -> 513.66 t/s`, and `pp4096` `440.08 -> 484.54 t/s`
  directionally.
- After coverage is fixed, traces still point at routed projection/dataflow:
  `routed_swiglu` and `routed_down` dominate, while route/reduce/finalizer remain
  much smaller.
- cx adversarial review agrees the next exact branch should target grouped down
  locality/dequant or a locality-preserving `SwiGLU+down` sidecar, not another
  attention or route-side branch.

Current design rule:

- Keep `QWEN_PREFILL_MOE_GROUPED_Q5_GATEUP=0` as the rollback path and keep the
  dedicated layer-46 Q5 SwiGLU oracle in the gate.
- Use warmed/interleaved methodology for A10B. Cold no-warmup A10B rows can be
  dominated by first-touch/model-residency effects and should not drive decisions.
- Require fast-path coverage assertions for MoE dtype variants. A3B `37/40` and
  A10B `47/48` are now canonical failure modes.
- Do not open new A10B attention branches until routed `SwiGLU/down` gets a fresh
  structural pass on the new 48/48 baseline.

Acceptance gates:

- Dedicated Q5 grouped-SwiGLU oracle on layer 46 must remain green, including
  poison/partial/zero-count bucket coverage.
- Default and rollback A10B prefill-vs-single smokes must stay green.
- A10B `pp512` phase coverage must show `48/48` route/grouped-routed/shared labels.
- Promote broader claims only from warmed AC-power rows with no thermal/performance
  warnings; prefer `pp512`, `pp1024`, `pp4096`, and at least one real rollout.
- For the next branch, require a combined routed-tail win, not just a standalone
  down-kernel or SwiGLU microbench win.

### 2. Hypothesis: A3B matrix attention plus grouped Q6 is the lcpp-cracking prefill candidate

Optimizes: A3B MoE prompt prefill from `pp320` through true-long after the Q6-down
grouped-path escape fix moved the board.

Why it is back at the top:

- Matrix attention is now default-on for the proven A3B group-8 shape, with
  `QWEN_PREFILL_ATTN_MATRIX_G8=0` as rollback. Fresh current-HEAD rows show the
  auto/default promotion materially beats the previous packed-attention default:
  `pp128` `~+7%`, `pp512` `~+13%`, `pp1024` `~+15%`, `pp4096` `~+18%`, `pp16384`
  `~+25-33%`, and real `v02_reva` `34.5k` `655.34 -> 837.18 t/s`.
- Matrix attention passes the full ignored A3B prefill-vs-single gate with auto
  default, including prefix `4096` / `8191` active shapes. The local matrix oracle
  keeps its documented numeric envelope (`cos >= 0.9999`, `max_abs <= 2e-2`), and
  the production gate is the stronger final logits / GDN / KV model-state check.
- The biggest A3B prompt win in this sprint came from a silent fast-path miss, not
  from another grouped-SwiGLU tile: A3B had `40` MoE layers but only `37` grouped
  routed trace labels because three late expert-down tensors are `Q6_K`. Closing
  that escape moved `pp320` `~830 -> 1061.45 t/s`, `pp512`
  `824.46 -> 1172.07 t/s`, and `pp1024` `910.43 -> 1287.43 t/s` while leaving
  warmed A10B neutral.
- Therefore, the next exact sprint should make the lcpp-cracking matrix path safe
  to promote while keeping the fast-path coverage matrix in the gate. Do not infer
  coverage from final-logit correctness alone.
- Post-Q6 no-op ceilings still leave routed MoE as a major lever, but not the only
  plausible lcpp delta: A3B `pp1024` baseline `1287.43 t/s` rises to
  `1578.34 t/s` with attention body skipped and `1979.46 t/s` with routed MoE
  skipped; A3B `pp4096` rises from `1198.11 t/s` to `1583.70 t/s` no-op
  attention and `1816.71 t/s` no-op routed. Shared MoE is small (`1332.52 t/s` at
  `pp1024`).
- Earlier no-op ceilings also showed routed FFN dominating shared FFN on the
  then-current default path:
  A3B `pp320` moves from about `765 -> 1136 t/s` with routed off, while shared
  off only reaches about `790 t/s`; A10B `pp320` moves from about `305 -> 648 t/s`
  with routed off, while shared off only reaches about `309 t/s`.
- The matrix-attention branch shifts the clean `pp4096` budget back toward MoE:
  attention-body skip is only `~3-7%`, while routed-MoE skip reaches `1270 t/s`
  and broad FFN skip reaches `2498 t/s` against `~929-972 t/s` matrix baselines.
- Post-route-threshold A3B `chunk_p=320` live grouped-tail profile still puts
  `grouped_swiglu` first (`4.06 ms`, `57.2%`) and `grouped_down` second
  (`1.60 ms`, `22.5%`); route logits are now `0.27 ms` (`3.8%`).
- Fresh routed-tail microprofiles at `chunk_p=512` keep the same shape: A3B tail
  is `3.84 ms` (`2.31 ms` grouped gate/up/SwiGLU, `1.55 ms` down+reduce), and
  A10B tail is `10.30 ms` (`6.81 ms` grouped gate/up/SwiGLU, `4.25 ms`
  down+reduce).
- A cheap hybrid falsifier, grouped `SwiGLU` into packed `down+weighted_sum`,
  correctness-passed but failed hard end-to-end on rebuilt sequential A3B `pp320`
  (`775.67 -> 478.84 t/s`). Removing grouped `out` / weighted-sum passes is not
  worth giving up grouped-down locality.
- Interleaved gate/up fused-bank grouped-`SwiGLU` is exact and can look strong in
  isolated SwiGLU microprofiles, but a full-tail diagnostic now kills it as a
  general expert-bank ABI: A3B `chunk320/512/1024` is `1.264x/1.074x/1.008x`,
  and A10B `chunk320/512/1024` is only `1.022x/1.006x/1.008x` once grouped down
  and weighted sum are included.
- A runtime duplicate fused-bank proof is already killed as a production path: A3B
  correctness passed, but it cost `11.25 GiB` extra resident memory and converted
  only `775.25 -> 790.38 t/s` at `pp320` (`~1.02x`).
- A down-only lcpp-shaped final-store probe that spread grouped Q5 down stores
  across all simdgroups was exact but slower on A3B `chunk512` (`1.38 -> 1.57 ms`
  for `split_down_reduce`), so final-store vectorization alone is not the crack.
- An all-`n16` grouped Q5 down probe was also exact but slower on A3B `chunk512`
  (`1.37 -> 1.70 ms` for `split_down_reduce`), so the current `n32` down tile is
  not obviously oversized despite diffuse bucket counts.
- Splitting `down+reduce` shows weighted sum is tiny at the key `chunk512` gate:
  A3B `split_down=1.36 ms`, `split_reduce=0.07 ms`; A10B `split_down=3.38 ms`,
  `split_reduce=0.11 ms`. The actionable bucket is grouped Q5 down, not reduce.
  Fresh bucket histograms are not sharply cold-biased enough to explain the gap by
  tile waste alone: both A3B and A10B have `p50_count=28`, `ge32=44` experts at
  `chunk512`, and the all-`n16` down probe still lost.
- A scan-ledger versus atomic-ledger falsifier now kills bucket order as the
  obvious Q5-down locality crack. The atomic ledger introduced many expert-ID
  back edges (A3B `1451/3987`, A10B `1568/4002`) while preserving exact output,
  but grouped routed-tail time was flat to slightly faster (`0.994x` A3B,
  `1.043x` A10B). Do not spend a branch on route-ledger ordering unless counters
  show a new locality mechanism.
- llama.cpp's remaining A3B MoE advantage on this M4 Max is not a hidden Metal
  tensor-API win: `llama-bench` reports `has tensor = false`, and forcing
  `GGML_METAL_TENSOR_ENABLE=1` does not satisfy the device-family gate. The same
  A3B/A10B GGUFs also have separate `ffn_gate_exps` / `ffn_up_exps`, not a fused
  `ffn_gate_up_exps` tensor, so the relevant external target is non-tensor
  simdgroup `mul_mm_id`, not a tensor-API or fused-bank path.
- A fresh all-`n32` rerun keeps the subtle point straight: all-`n32` is much
  faster than all-`n16` in the isolated grouped-tail proof (`1.792x` A3B,
  `1.384x` A10B at `chunk512`), but forcing all-`n32` does not beat the current
  default hot-`n32` path end-to-end (A3B `pp512` flat, A3B `pp1024` slightly
  negative, warmed A10B `pp512` negative). The default hot gate already captures
  the high-count tile win.

Current design rule:

- The A3B matrix-attention default gate has passed. Resume routed-MoE work only
  after preserving the new matrix default in coverage gates, because it changes
  the denominator for future A3B routed-tail claims.
- Do not expect prompt chunk policy to close dense long-context prefill. It is now
  a narrow MoE production knob: promote only if a prompt-length-gated `2048` or
  arch-specific cap keeps A3B wins and A10B gains without dense changes or
  memory-pressure warnings. Do not blanket-default A3B to chunk `2048`; the repeat
  gate regressed `pp128/512`.
- Matrix scratch policy is now explicit: prompt-aware callers should allocate with
  the actual last position; auto mode falls back to packed attention if scratch is
  undersized; force-on keeps the hard error. Keep the looser matrix oracle envelope
  documented and rely on prefill-vs-single as the production correctness gate.
- Before adding another local grouped kernel variant, prove the remaining lcpp gap
  with matched per-layer/per-op attribution on the same GGUF, prompt length,
  chunk/batch shape, warmup policy, power state, and flash-attention setting.
- Add/keep a coverage gate that asserts expected routed-layer counts and flags any
  fallback by dtype/role. The A3B `37/40` miss is the failure mode to prevent.
- Do not pursue dataflow branches that sacrifice grouped-down locality unless they
  first show parity on rebuilt sequential `pp320`/`pp512` gates.
- Do not repeat local `grouped_swiglu` knob sweeps unless a new phase ladder shows
  a new mechanism. The next exact proof must reduce the combined
  `grouped_swiglu + grouped_down` routed-tail bucket.
- Do not chase bucket-order rewrites as the next grouped-down lever; scan versus
  atomic ordering is exact and essentially flat despite very different ID order.
- Do not defer the next branch waiting for llama.cpp Metal tensor-path parity on
  this hardware. The live llama.cpp path is the regular simdgroup `mul_mm_id`
  path for these files.
- Do not retread all-`n32` as a default path; use it only as a diagnostic for
  tile-count sensitivity unless a new distribution changes the cold-bucket trade.
- Demote offline/interleaved gate+up ABI to a narrow A3B `chunk320` branch. The
  next serious exact FFN branch should attack grouped Q5 down locality/dequant or
  a true `SwiGLU+down` fusion that preserves grouped-down locality and beats the
  full-tail fused-bank diagnostic, not the isolated SwiGLU microprofile.
- Keep dense GDN skinny E8xP32 scoped to prompt-prefill `beta_proj` / `alpha_proj`.
  The safety case depends on that narrow callsite, the sampled Qwen3.5/Qwen3.6
  shape audit, and the rollback env `QWEN_PREFILL_GDN_SKINNY_E8P32=0`. The
  discovery raises the EV of searching for other F32 skinny projection dispatcher
  mismatches, but do not generalize this helper without a new gate.
- Keep grouped routed zero-fill default-off as small cleanup, not a main roadmap
  lever.

Acceptance gates:

- Matrix-attention promotion is complete for A3B/group-8. Any future change to this
  path requires preserving default/rollback rows, a full A3B prefill-vs-single
  gate, and trace-label coverage showing `10/10` matrix attention layers plus
  `40/40` MoE fast-path labels.
- A MoE chunk-cap default change requires clean-build rows showing `pp512/1024`
  unchanged by construction, repeated A3B and A10B long-prompt wins at
  `pp4096/16384`, one real-rollout A3B row, and no `pmset` or memory-pressure
  confounds. Dense rows are guardrails, not expected beneficiaries.
- Dense G6, A10B/G16, non-F16 KV, or different attention shapes are not covered by
  the A3B default promotion; they need separate gates before any auto-on policy.
- For the next commit/default promotion in this lane, require trace-label or
  equivalent in-process coverage showing all expected MoE layers use the intended
  route, grouped routed, and shared packed paths for A3B and A10B.
- Before claiming llama parity, require paired qwen/llama rows plus attribution on
  identical fixtures. Throughput alone is not enough after the Q6 escape lesson.
- Any expert-bank ABI / interleaved layout branch must be exact on A3B and A10B
  small gates, improve end-to-end A3B `pp320` by at least `~1.12x` and A10B
  `pp320` by at least `~1.15x`, and be no worse than `-2%` at `pp512` with no
  material `pp1024` regression. After the full-tail diagnostic, it also must beat
  the live grouped routed tail by `>=1.15x` at A3B/A10B `chunk512`, not merely the
  isolated grouped-SwiGLU kernel.
- Any true fused `SwiGLU+down` branch must preserve grouped-down locality and beat
  the offline-bank path, not just the old baseline.
- Any structural routed-tail branch must show at least `>=1.15x` routed-tail
  speedup on both A3B and A10B at `pp512`, stay positive/neutral at `pp1024`, and
  improve A3B `pp4096` with matrix attention fixed by `>=1.10x` before default
  consideration.
- Dense GDN skinny follow-up is monitoring, not a pre-merge blocker: add one
  long-context A3B/state sanity row and a real prompt/top-k check when convenient,
  and audit any future GGUF with new F32 GDN alpha/beta shapes before assuming the
  default is risk-free.
- A3B matrix `pp4096` phase traces make `pre_norm` look large, but a llama-style
  float4 spelling of batched RMSNorm is correctness-green and end-to-end flat
  (`1408.70/1409.10` baseline vs `1408.64/1407.07` vec4). Do not pursue local
  RMSNorm vector spelling; revisit norm only as fused norm+projection or after a
  direct llama norm-node differential proves a real gap.

### 2. Historical: true-long A3B matrix-attention branch is now folded into #1

Optimizes: A3B `16k/32k+` prefill where qwen now falls from the medium-prompt
plateau faster than `llama.cpp`.

Why this section is retained:

- Same-shape sparse rows show `llama.cpp` also declines after the medium-prompt
  peak, but remains faster: post-Q6 qwen/lcpp is `0.96x` at `pp1024`, `0.95x` at
  `pp4096`, and `0.78x` at `pp34502`; `pp16384` needs a post-Q6 rerun.
- Real rollout shape is not the first-order cause: the post-Q6 real `v02_reva`
  `34.5k` row is `674.28 t/s`, still below the latest lcpp synthetic long anchor
  but far above the old `587.60 t/s` same-fixture row.
- No-op attribution at true-long shapes points at attention body: A3B `pp16384`
  goes `767.64 -> 1101.72 t/s` with attention body skipped, nearly matching
  lcpp full prefill (`1112.03 t/s`), while routed-MoE skip reaches only
  `913.73 t/s`.
- `llama-bench -fa 0` disables flash attention rather than selecting auto, and
  `-fa 1` is flat/slightly slower for A3B at the checked prompt lengths. The
  relevant lcpp target is therefore the non-flash Metal path, not
  `GGML_OP_FLASH_ATTN_EXT`.
- A deliberately matrix-shaped A3B sidecar (`V^T`, `KQ`, softmax, `KQV`) first
  regressed long prompts because it re-transposed the full V prefix in the body.
  After moving V_T writes into fused cache fill and vector-loading the F32 B tile,
  it wins real long rows: about `1042 t/s` at `pp4096`, `997 t/s` at `pp8192`,
  noisy `~864-904 t/s` at `pp16384`, and `725.7-736.9 t/s` at `pp34502`
  depending on chunk size.
- After the grouped Q6-down fix, the same sidecar now reaches `1434.39 t/s` at
  `pp4096`, `878.72 t/s` at synthetic `pp34502`, and `870.94 t/s` on real
  `v02_reva`, enough to move matrix productionization to the #1 active bet.
- After those matrix wins, local attention follow-ons did not convert: direct KQV
  final stores regressed, vectorized KQV temp copies regressed, and F16
  probability scratch was flat/slower while failing the current matrix oracle
  max-abs limit.

Current design rule:

- Do not use `pp320/512/1024` routed-FFN wins as evidence that true-long behavior
  is fixed; measure same-shape `16k/32k+` rows.
- Treat packed-attention body/main-pass work as the long-context branch. The
  explicit reduce pass was already measured tiny; the remaining attention wall is
  main-pass execution shape, KV reads, partial writes, and online-softmax work.
- Compare against `llama.cpp` at the same prompt length before claiming long
  scaling progress.
- Treat fused V_T scatter as the first lcpp-derived mechanism worth
  productionizing, but do not promote the current sidecar as-is. It still needs a
  non-manual max-pos allocation policy and a tighter KQV correctness story.
- Treat chunk `2048` as the current matrix-long candidate for `34.5k` rows;
  chunk `4096` loses despite the larger query batch.
- Next inspect/copy deeper lcpp `mul_mm_f16_f32` KQ/KQV tiling and score layout
  only after matrix default gates fail or plateau; do not spend the next sprint on
  more local KQV store/probability variants before the productionization gates.

Acceptance gates:

- A long-attention branch must improve A3B same-shape `pp16384` and `pp34502`, not
  just medium prompt rows.
- Target at least `>=1.20x` at `pp16384` or a clear path to preserving the new
  `>=1.0x` matrix-sidecar spot rows after production gating.
- Any matrix/non-flash branch must be positive at `pp4096` and `pp8192` before it
  gets long-run time at `16k+`; `pp512/1024` wins alone are not a promotion signal.
- Defaulting the fused V_T matrix path requires repeated cooled wins at
  `pp1024/2048/4096/8192/16384`, a clear max-pos scratch policy, and no dense or
  decode regression from extra V_T memory/writes.
- Any matrix chunk-size change must include memory accounting for score scratch and
  at least one true-long row; `pp1024` is no longer a sufficient chunk oracle.
- Keep A3B packed-attention oracle/correctness green at the first activated
  long-context chunk shape.

### 3. Hypothesis: remaining packed-attention variants need end-to-end systems evidence, not local kernel knobs

Optimizes: the remaining A3B/A10B prompt gap after prompt-native packed
attention, family-specific `NWG`, and `min_pos=512` are already defaulted.

Why it moves to the top:

- Prompt-native packed attention is no longer hypothetical; it is the default for
  the proven MoE prompt shapes and it materially moves `pp512`, `pp1024`, and the
  long-rollout lanes.
- The next obvious packed-kernel knobs already produced hard negative lessons:
  `QT=4` is exact but slower, and body-only `NWG=32` microbench wins were a false
  promotion signal until cooled end-to-end sweeps re-ranked the family defaults.
- The A3B matrix-attention sidecar is now a systems-level lesson rather than a
  warning only: high-level graph copying was insufficient, but moving V_T writes
  to cache-fill time converted the long rows. Further attention work needs this
  kind of dataflow evidence, not row-kernel knob sweeps.
- Even a more faithful one-layer attention-stack microbench is still not a safe
  promotion oracle for A10B. That points at multi-layer interactions, scratch /
  residency behavior, or queue/scheduling effects rather than another easy kernel
  retune.

Current design rule:

- Promotion decisions for A10B packed variants must come from cooled end-to-end
  sweeps with repeated baseline anchors, not from body-only or one-layer micros.
- Use the attach-mode tracing helper and the packed per-layer oracle only to
  explain or falsify a candidate, not to outrank the end-to-end board.
- Treat further packed-kernel knobs as hypothesis generators until a systems-level
  capture says what the next real bottleneck is.

Acceptance gates:

- Show a systems-level explanation for any new packed-attention variant that beats
  the current default on repeated cooled sweeps for A3B and/or A10B.
- Keep per-layer packed-vs-old oracle green on the first newly activated prompt
  regime and at the active long-context chunk shape.

### 4. Historical: A3B long-prompt prefill was decode-shaped attention in disguise

Status: superseded by the packed-attention default and the calibrated true-long
same-shape rows above. Keep this section as historical context for why the
prompt-native packed-attention branch became the center of gravity.

Optimized: the clearest structural prompt gap after the A3B group-8 long-context
subgroup fix.

Why it moves to the top:

- Same-fixture real-rollout ladders and matched-token synthetic ladders now agree:
  A3B collapses with long prompts while A10B is much flatter.
- Attention no-op dominates the slope; GDN and routed MoE are much smaller by
  comparison.
- The new `g8_t2` subgroup path removes a major local A3B attention bottleneck,
  but the remaining slope is still strongly attention-shaped and weakly sensitive
  to larger `prefill_chunk`, which points at the decode-shaped prompt attention
  algorithm itself.
- `llama.cpp` was prompt-native on Metal for this stage while `qwen-llm` still ran
  per-token decode attention inside prefill chunks.

Current design rule:

- Keep the new A3B group-8 subgroup path as an experimental / guarded prefill
  win, not a universal decode selector.
- The prompt-native packed-attention microproof landed and is now defaulted for
  the proven MoE shapes.
- The remaining true-long attention work is no longer “write packed attention”;
  it is main-pass/context-growth optimization inside the packed path.

Acceptance gates:

- Show a prompt-native packed-attention microproof at A3B shape (`group=8`,
  `head_dim=256`, F16 KV) that beats repeated decode-shaped attention by at least
  `~1.25x` at `16K` and `~1.35x` at `32K`, or projects to a meaningful end-to-end
  long-prompt win.
- Keep correctness against the existing path / naive reference on active long-
  context shapes.

### 2. Hypothesis: part of the remaining `pp320` gap is harness/phase mismatch

Optimizes: decision quality and scoreboard fidelity.

Why it moves to the top:

- We are near parity on the user-facing CLI harness but still far behind on
  `llama-bench pp320`.
- Before another major rewrite, we need to know how much of that miss is real
  prompt work and how much is benchmark semantics.
- This is the one measurement-heavy item that remains justified because it tests
  a direct causal hypothesis about the gap.
- The `MTL,BLAS` backend string `llama-bench` prints is a registration artifact,
  not a hot-path signal: at `pp320` on 27B Q4_K_M every mat-mat / mat-vec node
  runs on Metal and the BLAS backend executes zero ops. See the BLAS hot-path
  audit in `docs/PERF-LOG.md` for the per-node sched-debug evidence. The gap is
  Metal vs Metal, not "we are missing a CPU sgemm lane".

Current design rule:

- `qwen-bench pp` now covers the core harness need: synthetic `pp<N>`, no decode
  loop, optional tail skip, wall/GPU reporting, and lowering summaries.
- Keep using it as the scoreboard harness for dense and MoE prompt work; do not
  overfit to the older repeated-prompt decode harness.

Acceptance gates:

- Maintain `qwen-bench pp` while adding any future prompt phase buckets.
- Recompute the remaining prompt gap against `llama-bench pp320` after each major
  prompt-path change.

### 2. Hypothesis: if the `pp320` gap is real, decode-shaped prompt attention is the dominant remaining engine problem

Optimizes: the largest likely remaining structural prompt-only gap once harness
semantics are aligned.

Why it moves up:

- The cheap packed attention-body cleanup already landed, but if the pure prompt
  gap survives harness matching, GDN-tail cleanup alone cannot close it.
- Our prompt attention still retains decode-shaped structure in the inner loop.
- That makes prompt-native packed attention the strongest causal explanation for
  a large remaining `pp320` miss.

Current design rule:

- Keep the new packed RoPE + chunk-scatter shape as the base path.
- Only escalate after item 1 confirms the remaining miss is real and not mostly
  harness semantics.
- Prefer prompt-native packed attention / verify primitives over more small glue
  cleanups once the hypothesis survives.

Acceptance gates:

- End-to-end pure prompt throughput must move materially against the new harness.
- Correctness must stay green on `prefill_tokens_matches_single_token_loop_27b`.

### 3. Hypothesis: if prompt attention is not enough, the next real dense prompt miss is GDN out-proj / recurrence tail

Optimizes: the residual dense prompt GDN work after the packed prep rewrite and
attention-body cleanup.

Why it moves back to the top:

- The split ladder still says the remaining GDN tail is mostly out-proj plus a
  smaller recurrence cost.
- This is now a bounded fallback hypothesis, not the assumed main story.

Current design rule:

- Keep using the real-graph `QWEN_PREFILL_GDN_SPLIT` ladder for bounded probes.
- Prefer narrow out-proj / recurrence cleanups over another broad rewrite.

Acceptance gates:

- End-to-end dense packed prefill must move on 27B against the repeated prompt.
- Correctness must stay green on `prefill_tokens_matches_single_token_loop_27b`.

### 4. Dense Prompt: Attention Body Cleanup

Optimizes: any additional dense prompt throughput still available in the
full-attention layers.

Why it stays near the top:

- The packed consecutive RoPE + chunk-scatter cleanup was a real win, but the
  attention body still costs on the order of `~120 ms` wall on the repeated
  prompt.
- If the remaining GDN-tail work stalls, this remains the best smaller-bore
  alternate lane before a true packed prompt attention rewrite.

Current design rule:

- Keep the new packed RoPE + chunk-scatter shape as the base path.
- Only escalate to more invasive packed causal prefill attention if smaller
  cleanups stop moving the prompt.

Acceptance gates:

- End-to-end dense packed prefill must move on 27B against the repeated prompt.
- Correctness must stay green on `prefill_tokens_matches_single_token_loop_27b`.

### 5. Decode Command-Model Overlap

Optimizes: apples-to-apples dense decode latency, especially 27B at 4K and up.

Why it moves up:

- Real 27B 4K attach-mode Metal trace now exists.
- It shows `128` command buffers for `128` decode tokens, `128` compute encoders,
  encoder duration median `~0.699 ms`, and previous completion -> next submit
  median `~0.538 ms`.
- The direct decode-window profiler at 4K is the decisive result:
  `med_total ~42.69 ms`, `med_gpu ~42.14 ms`, `med_cpu_enc ~0.20 ms`, so decode
  is about `98.7%` GPU-busy on the 27B dense guardrail at 4K.
- The token loop is fully serialized today. The trace does NOT show a giant
  hidden bubble, but it does show a real low-single-digit command-model gap.
- Process-scoped compute intervals split into a small short-gap population and a
  large token-cadence population; the short intra-CB gaps total only about
  `~1.2 ms/token` at 4K, which keeps this as a real but bounded lever.

Immediate focus:

- Double-buffered / pipelined decode submission first.
- Use the new attach-mode trace helper and parser to validate any overlap claim.
- Only escalate to heavier encoder restructuring if post-overlap traces still
  show meaningful serialized slack.

Acceptance gates:

- Reduce completion -> next-submit gap and total 27B decode ms/token at 4K.
- Keep exact-token behavior and current correctness gates intact.

Status:

- A dense-only bench path now exists at `qwen-bench decode-window --pipelined`.
- Measured at 27B dense:
  - `ctx=4096`: about `~0.3%` over alternating repeats
  - `ctx=32768`: about `~0.3%`
- Keep it as an experimental harness, not a production checkpoint, unless a
  future shape/context shows a materially larger win.
- A second bench-only dense decode branch now exists at
  `qwen-bench ctx-sweep --concurrent-gdn-proj`.
- At 27B dense `ctx=4096`, `window=64`, it improves decode from
  `43.87 -> 42.14 ms/token` (`22.8 -> 23.7 t/s`), with the gain showing up in
  GPU time rather than CPU encode.
- At 27B dense `ctx=16384`, `window=64`, it also improves decode from
  `47.67 -> 46.51 ms/token` (`21.0 -> 21.5 t/s`).
- This is the first command-model branch that has cleared the “real enough to
  checkpoint” bar; the gain survives 4K and 16K, though it narrows somewhat as
  attention grows.
- Attention-only overlap is smaller:
  - `ctx=4096`: `43.89 -> 43.33 ms/token`
  - `ctx=16384`: effectively flat (`47.26 -> 47.22 ms/token`)
- Running both projection-overlap branches together is still positive and
  checkpoint-worthy:
  - `ctx=4096`: `43.64 -> 42.31 ms/token`
  - `ctx=16384`: `47.23 -> 46.14 ms/token`
- The combined branch is not additive with GDN-only overlap, but it remains the
  strongest decode-focused command-model variant measured so far.
- Treat encoder-boundary removal in these concurrent paths as an anti-bet until
  a fresh A/B proves otherwise: the split is buying GPU-side overlap between
  independent front projections, not merely adding host encode overhead. Host
  encode is already only ~0.20 ms of ~42.14 ms GPU time at 27B dense 4K, so the
  upper bound on collapsing encoders is well under 0.5%; the load-bearing piece
  is the middle `begin_concurrent` encoder for the front projections, and any
  merge must preserve that concurrent-dispatch property.
- MoE no longer shares the same decode-overlap uncertainty: concurrent GDN front
  projections are now production-wired for MoE decode on this repo, and the A3B /
  A10B `tg32/tg128` sweeps are already materially positive.

### 6. Read-Only Weight Residency And Scratch Storage Cleanup

Optimizes: decode and prompt wall via cheaper Metal bookkeeping and cleaner GPU
memory behavior.

Why it belongs near the top now:

- The command-model trace says there is not a giant host bubble, so the cheap
  structural wins become more attractive than speculative scheduler work.
- The external review's `hazardTrackingMode: untracked` + residency-set idea is
  orthogonal to no-copy GGUF views and should help regardless of mmap strategy.
- Scratch is still `StorageModeShared` everywhere today, which is convenient but
  not obviously ideal for GPU-only hot tensors.
- The new A10B `pp128` cold-run study keeps this bounded: touching or pinning MoE
  expert-bank buffers fixes a benchmark-hotness artifact, but steady-state
  `pp320/pp512` stay flat, so this is still structural cleanup rather than the
  highest-EV runtime lever.

Immediate target order:

1. Mark read-only weights as untracked and managed by a residency set.
2. Audit CPU readback/debug use, then move only proven GPU-only scratch arenas
   toward `StorageModePrivate`; avoid a blanket allocator swap.
3. Measure decode/prefill again before bundling this with larger graph changes.

Acceptance gates:

- Any change must preserve correctness and avoid regressing steady-state decode.
- Keep these as cheap structural cleanup unless traces show a larger-than-expected
  wall effect.
- Specifically: `--full-logits-decode` (exact-token oracle) and any tap-based
  hidden-state captures must stay green across a scratch storage-mode change,
  since a blanket `StorageModePrivate` swap silently breaks CPU readback paths.
  `MetalSession::fresh` currently allocates ~39 scratch tensors via
  `MetalTensor::zeros_f32` (`StorageModeShared`); the audit must classify each
  as GPU-only vs CPU-readable before any allocator change lands.

### 7. MoE Next: Execution-Model Reset After The Exact Grouped Plateau

Optimizes: the remaining MoE prompt scoreboard gap after the current exact grouped
backend has locally plateaued on this hardware / repo shape.

Current read:

- The grouped expert-major MoE prompt backend is now the stable base path again.
- Fused route+bucket and hot-th48 grouped Q4 are both kept because they compose
  positively at `chunk_p >= 512`.
- `n32-all` is still not a ship candidate: it improves grouped compute locally but
  does not clear the end-to-end pp gate.
- Router logits no longer look like the next missing kernel. The `E8xP32` kernel
  is already good enough to ship on the proven regime.
- The obvious exact local `grouped_swiglu` variant family is now well sampled and
  mostly exhausted here: tile, threshold, queue/locality, atomic-down, and
  resident-mirror branches all went flat or negative.
- The obvious route-ledger ordering hypothesis is also falsified: an exact atomic
  bucket ledger creates many expert-ID back edges but leaves A3B/A10B `chunk512`
  grouped routed-tail time essentially flat.
- Two more exact reads now narrow the field further:
  - grouped `inner/out` zero-fill is safe to skip and slightly positive, but too
    small to change the scoreboard by itself;
  - a re-based split routed FFN proof only reaches parity to slight loss versus
    the live grouped backend, so “just make it llama-like by separating gate/up”
    is not enough.
- The strongest near-term exact branch is now the guarded concurrent-tail path:
  overlap the live grouped routed tail with the live shared FFN at `chunk_p >= 512`.
  It converts to roughly `1-3%` end-to-end on `pp512`, but collapses quickly by
  `pp1024`, so it should be treated as a bounded production win, not the final
  structural answer.
- That leaves the next serious structural MoE-first branch as a **narrow** kernel
  proof, not a broad sidecar rewrite: a true `MUL_MAT_ID`-style / id-aware
  projection microproof that beats the current grouped projection or routed tail
  directly on both A3B and A10B.

Acceptance gates:

- Keep `QWEN_PREFILL_MOE_GROUPED=0` as the kill switch while rollout evidence is
  still expanding.
- Keep the route-only and hot-only flags forceable, but do not default either one
  globally outside the allowlisted combo regime.
- Default-on only for the proven `Q4_K/Q4_K/Q5_K` MoE packed-prefill envelope;
  fallback stays live for unsupported dtypes/shapes and for prompt chunks below
  the allowlist.
- Keep the current correctness matrix green: A3B single-token-loop + hidden
  capture, A10B smoke, awkward chunk-boundary A10B (`T=129`, `P=128`), and dense
  9B/27B guardrails.
- Treat the concurrent-tail branch as shape-gated until a full active-shape
  prefill oracle exists for the covered `pp512+` regime; do not assume the large
  block-local overlap effect composes into a large end-to-end win.
- Do not spend another major engineering branch on a new exact local
  `grouped_swiglu` variant unless a new diagnostic shows a new mechanism beyond
  the already-falsified tile / threshold / queue / mirror family.
- Before writing a large new id-aware kernel, require a smaller microproof to beat
  the current grouped projection / routed tail directly on both A3B and A10B, not
  just the obsolete packed denominator.
- Any next MoE-first branch must beat the current allowlisted combo on both A3B
  and A10B at `pp512` / `pp1024`, not just improve a routed microprofile.

### 8. Use 9B As The Fast Dense Long-Context Canary

Optimizes: experiment throughput and long-context turnaround while preserving the
27B guardrail.

Why it is now active:

- `group=4` attention v4 is now enabled, which unlocks the whole small dense line
  (0.8B / 2B / 4B / 9B) for realistic long-context decode.
- Local 9B now reaches 32K cleanly and is dramatically faster to iterate on than
  27B: `64.0 t/s` at 4K, `59.5 t/s` at 16K, `53.8 t/s` at 32K.

Usage rule:

- Use 9B for fast falsification of long-context attention / dense prompt ideas.
- Keep 27B in the analysis loop before claiming a real win.

Acceptance gates:

- Long-context experiments should be reproducible first on 9B, then confirmed on
  27B before the roadmap moves.

### 9. No-Copy GGUF Views And Residency Warmup

Optimizes: TTFT, cold-start variance, load-time memory pressure, and possible VM
object overhead.

Why it enters the roadmap now:

- The `ds4` close read makes this the strongest non-kernel structural crib.
- Current `qwen-llm` still copies weights tensor-by-tensor into fresh shared
  buffers; `ds4` instead wraps a few large GGUF-backed no-copy Metal views and
  warms residency up front.

Why it is not above the current prompt work:

- This is more likely a load / first-token / memory-cleanliness lever than the
  next steady-state prompt-throughput unlock.
- It still looks high-EV enough to prototype once the current MoE and dense
  prompt branches have a stable checkpoint.

Acceptance gates:

- Prototype shows materially better load time, first measured token stability, or
  memory / VM-object behavior without regressing steady-state throughput.

### 10. Frontier Benchmark Harness With Snapshot / Restore

Optimizes: benchmark quality and long-context decision speed.

Why it matters:

- `ds4-bench`'s frontier measurement style is a better mental model for prompt vs
  decode frontiers than one blended tokens/sec number.
- This would sharpen long-context dense/MoE comparisons and future speculative
  work without changing model semantics.

Acceptance gates:

- Add exact frontier prompt/decode probes that can restore from snapshots and
  measure a fixed local window.
- Use it to compare qwen vs llama phase-for-phase, not on blended totals.

### 11. Speculative Path: Attack Repeated Long-Context Attention Cost

Optimizes: DFlash / MTP viability at realistic context lengths.

Priority rule:

- Keep this behind the current dense/MoE prompt push.
- When returning to speculative work, do not lead with policy/schedule tuning;
  lead with kernel work that removes repeated long-context attention cost.
- v0.416 audit digest reopened KV compression ahead of packed-N verify attention;
  v0.437 closes the same-layout Q8_0 reader variant as negative. Future
  compressed-KV work must start from a different layout/body and a fresh
  `attn-intra` win; do not keep packed verify blocked on Q8_0 specifically.

What the latest analysis says:

- Current MTP shape is structurally weak at long context because lazy verify does
  not amortize enough base work and MTP has its own growing KV attention cost.
- Current DFlash is made safe by adaptive verify `N`, but not fast, because the
  drafter and target still pay too much long-context attention work.
- v0.502 adds draft-free MTP probes: replaying current D3 draft tokens with zero
  MTP calls still loses on 27B (`0.967x` total), and a perfect greedy D3 oracle is
  only `1.201x`. This makes packed target-verify structure the current binding
  MTP loss; a single-CB drafter alone cannot explain or close the MTPLX gap.
- v0.503 finds the sharper constraint: physical verify N, not verify as a whole.
  Perfect-oracle D7/N8 and D15/N16 already reach `52.0` and `57.1 t/s`, while
  off-ladder N5/N9 are much weaker. Bucketed logical D over physical N8 turns the
  actual 27B D3 code prompt from `0.912x` to `1.031x` at the same alpha.
- v0.504 runs real recursive acceptance: D7/N8 wins on the 27B code prompt
  (`1.153x`, `~4.0 emitted/step`) and replay-current reaches `1.362x` with zero
  draft calls; D15/N16 loses (`0.716x`) because emitted/step is only `~4.1`.
- v0.505 shows single-command-buffer D7 drafting is only a thin win (`1.162x`),
  so command submission/readback is not the major remaining D7 draft tax.
- v0.506 ablates draft-side work: body-no-lm-head reaches `31.1 t/s` and
  bridge-only reaches `32.1 t/s` against normal single-CB `27.7 t/s`. Draft
  `lm_head+argmax` is the largest draft-side cost, recursive body is smaller,
  and bridges are closed; the global gap is now acceptance plus verify ceiling.
- v0.507 rank tracing shows D7/N8 terminal mismatches are often near misses:
  target is in draft top-2 for `50%`, top-4 for `67.9%`, top-8 for `85.7%`, and
  top-16 for `92.9%` of terminal mismatches on the 27B code prompt.
- v0.508 closes simple rerank: best margin-swap policy is only `4.031`
  emitted/step and oracle one-token top-16 terminal rescue is only `4.812`, short
  of the `>=5.2` gate.
- v0.509 kills `token_embd.weight` as a no-asset Q4 draft-head substitute on 27B:
  alpha drops to `0.000` and the row falls to `5.3 t/s`.
- v0.510 splits MTP prompt prefill from decode-loop timing and defaults the bench
  to MTPLX-style post-norm base/recursive hidden feeds. On the 27B code prompt,
  D7/N8 legacy pre/pre is `30.2 t/s` decode-only at `3.879` emitted/step, while
  post/post is `33.0 t/s` at `4.267` emitted/step; on a narrative prompt it moves
  `26.8 -> 30.7 t/s` and `3.459 -> 4.000` emitted/step. D7/N8 oracle is
  `74.7 t/s` decode-only, so target verify can already explain the MTPLX
  `65-80 t/s` screenshot band when acceptance is ideal and prefill is excluded.
- v0.511 tests the largest MTPLX history-policy semantic delta. `--mtp-history
  cycle` resets MTP KV every speculative step and keeps only within-chain draft KV,
  but it is a hard kill: code prompt emitted/step falls `4.267 -> 2.783` and
  narrative falls `4.000 -> 2.415`, both equivalence PASS. Committed MTP history
  remains the default.
- v0.511 also fixes the rank simulator for fixed-token benches. Post/post rank
  traces now show oracle one-terminal-rescue top16 would finish 128 tokens in
  26 steps on both prompts (`4.923` tokens/step), still below the `>=5.2` gate;
  margin-swap policies remain flat.
- v0.512 adds A3B MoE MTP asset coverage. Unsloth's
  `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` has a real MoE MTP head at `blk.40`, but qwen
  only recognizes it and reports unsupported execution. llama.cpp `b9833` runs
  the same file at `tg128`: `81.71 t/s` with `n_depth=0` and `82.12 t/s` with
  `n_depth=7`. This is now an explicit unsupported-model gap; the thin lcpp
  synthetic uplift argues for correctness/acceptance probing before Q2/Q3 MoE
  MTP kernel work.
- v0.513 replaces the MoE-MTP unsupported bail with a correctness-first executable
  path. The single MTP MoE expert bank is dequantized to F32, A3B UD-Q4_K_S lazy
  MTP-1 passes equivalence, and base MoE decode gains native `Q4_K` routed-down
  coverage. This is not a performance conclusion: D3/N4 still fails because the
  packed-N verifier does not implement qwen35moe base forward, and IQ2/Q2/Q3 base
  MoE still needs native expert-bank kernels rather than F32 residency expansion.
- v0.514 adds a correctness-first qwen35moe packed-N verifier branch. D3/N4 now
  executes and passes equivalence on A3B UD-Q4_K_S, but the verifier is
  row-sequential for MoE blocks and is therefore an acceptance/coverage tool, not
  the final throughput shape.
- v0.515 measures the intended A3B Q4_K_M MTP artifact. Same-file qwen no-spec
  `tg128` is green (`94.97 t/s` vs llama.cpp b9833 D0 `77.78` / D7 `79.02`).
  Native D7/N8 has the acceptance signal the Q4_K_S smoke lacked: `5.333`
  emitted/step, alpha `0.619`, equivalence PASS. But total remains a loss
  (`0.815x`) because cost dominates. Replay-current is now unblocked and reaches
  `0.962x` with `mtp_calls=0`, while a perfect oracle reaches `1.213x`.
  Therefore the A3B branch is live, but the current cost split is verifier
  row-sequential work plus full draft `lm_head+argmax`, not acceptance alone.
- v0.515 also kills the naive first batched-MoE-verifier idea. Reusing prefill
  grouped routed/shared FFN at physical N8 is correctness-safe but slower:
  `QWEN_MTP_MOE_VERIFY_GROUPED_FFN=1` replay-current regresses `337.2 ->
  377.3 ms`. The issue is execution shape, not merely missing a grouped FFN call;
  prefill grouped kernels were built for much larger prompt buckets.
- v0.516 kills the cheap low-bit draft-head copy. Setup-time GGML Q4_1 and Q4_0
  copies of `output.weight` preserve equivalence and A3B D7/N8 acceptance, but
  remain flat/slower than default: Q4_1 totals `399.7 ms`, Q4_0 totals
  `397.0 ms`, versus the v0.515 default `397.8 ms` on the same 16-token row.
  This does not kill MTPLX's actual 4-bit affine/group-size-64 draft-head layout
  or a top-k/head policy, but it closes legacy GGML output clones as the cheap
  missing lever.
- v0.517 adds explicit MTP decode phase buckets. A3B Q4_K_M D7/N8 default spends
  draft `51.7 ms`, verifier `242.8 ms`, restore `1.0 ms`, and bridge `4.9 ms`;
  replay-current spends verifier `239.2 ms`, restore `0.9 ms`, and bridge
  `0.5 ms`. Therefore the replay-current loss is verifier structure, not hidden
  MTP bridge/rollback overhead, and the normal-path draft tax is secondary until
  verifier cost moves.
- v0.518 lands the first N8-native verifier win and defaults it with rollback
  `QWEN_MTP_MOE_VERIFY_BATCHED_MIXER=0`: MoE packed verify now batches the mixer
  side across physical N8 and leaves only the MoE FFN tail row-sequential. A3B
  D7/N8 16-token total improves to `0.910x` from `0.805x`, replay-current becomes
  a win (`1.112x`), and the 64-token row improves from `0.864x` to `0.953x` while
  preserving equivalence. The remaining verifier work is now mostly the MoE FFN
  tail, not GDN/attention mixer projections.
- v0.519 defaults `QWEN_MTP_MOE_VERIFY_CONCURRENT_FFN=1` for Q4_K/Q4_K MoE
  verifier gate/up banks, with `=0` rollback. This reuses normal decode's
  routed/shared FFN wave split inside the N8 verifier. A3B D7/N8 64-token normal
  MTP now wins (`1.035x`) and replay-current reaches `1.262x`; the 16-token row
  is near parity (`0.980x`) but remains fixed-cost/draft-cost limited.

Highest-EV speculative kernel targets:

1. Replace row-sequential packed verify with an N8-native verifier shape. Do not
   reuse prompt-prefill grouped MoE kernels blindly: v0.515 kills that direct
   transplant at N8. v0.518 proves the branch by batching the mixer side and
   turning replay-current into a win; v0.519 then reuses decode's FFN wave split
   and turns 64-token normal MTP positive. Keep improving verifier only where it
   lowers `verify_ms` under replay-current; the next broad MTP blocker is now
   draft-side full-vocab cost and short-prompt fixed overhead.
2. Reduce D7/N8 MTP draft-head cost only through a new execution shape. v0.515
   A3B and v0.506 27B agree that `lm_head+argmax` is the largest draft-side tax,
   but v0.516 kills legacy GGML Q4_1/Q4_0 output copies as a cheap fix. Exact
   fused `lm_head+argmax` remains bounded; a larger win needs an
   MTPLX-isomorphic 4-bit affine/top-k kernel or a policy that lowers full-vocab
   work without unacceptable acceptance loss. v0.517 sizes the normal A3B D7/N8
   draft bucket at `~52 ms`, so this is meaningful but no longer first.
3. Keep A3B Q4_K_M D7/N8 as the MoE MTP acceptance/cost gate. It now clears the
   emitted/step investigation threshold, so use it alongside 27B code/narrative
   rows for MTP changes rather than relying on dense-only evidence.
4. Add native IQ2_S MoE expert-bank gate/up support, then evaluate Q2_K/Q3_K. Do
   not dequantize base MoE expert banks to F32. The one-block MTP F32 bridge is
   acceptable; base low-bit MoE needs native routed bank kernels for residency.
5. Continue dense-27B MTP semantic parity before tree work, but with cycle history closed.
   Remaining concrete variants are position-offset semantics, history-window
   variants rather than full reset, and MTPLX contract/draft-asset details. Gate
   continuation on emitted/step `>=5.2` or `>=25%` over the post-norm committed
   default, equivalence PASS.
6. Inspect MTPLX acceptance/reporting enough to separate decode-only vs total,
   D3 vs D7, and proper draft-head asset effects. The current qwen oracle already
   reaches the screenshot band decode-only; the open question is how MTPLX gets
   much closer to oracle in actual chain mode.
7. Prototype a proper low-bit draft head only if it is not another legacy GGML
   output clone. Do not use `token_embd.weight`; v0.509 killed that alias. Do not
   expect Q4_1/Q4_0 clones to help; v0.516 killed those. Gate a new layout/kernel
   on `>=5%` whole-run gain, `>=40%` recovery of the body-no-lm gap, and `<=5%`
   relative acceptance loss.
8. Build an alternate-continuation/tree simulator before any tree implementation,
   but only after post-norm rank traces still show unreachable chain acceptance.
   Gates: N8 `>=5.2` emitted/step to investigate, `>=5.8` to implement; N16 `>=9`
   interesting, `>=11` viable. Require stability across prompts.
9. Keep physical N8 as the first native MTP packet shape. Use padding/rollback for
   shallower adaptive depths; do not pursue D15/N16 until a better asset/policy
   proves much higher emitted/step.
10. Keep target-verify work tied to replay-current gates. A3B now clears the
    emitted/step threshold, but the direct prefill-grouped FFN transplant regressed;
    reopen packed GDN/tape, packed multi-query attention, or an N8-specific MoE
    kernel only with a measured replay-current win. The next useful attribution is
    inside the `verify_ms` bucket, not bridge/restore/draft bookkeeping.
11. Build bucket policy around supported packet shapes; prefer padding/rollback to
   arbitrary ragged N unless bucketed verification fails a correctness or waste
   gate.
12. DFlash two-range attention reading ctx-cache and noise directly, without
   `k_full` / `v_full` materialization.
13. Compressed-KV only if a non-Q8_0 layout/body first beats tuned F16 in
    `attn-intra` at both 8K and 32K.
14. Retile the `N=16` mat-mat specializations only if verify/draft phase profiles
    show N16 mat-mat-heavy surfaces remain material after the attention fixes.

### 12. Mid-Graph Flush / Overlap Before ICB / MTL4

Optimizes: decode and prompt wall only if later traces show more cadence slack at
other contexts or shapes.

Why it stays behind the others:

- The new 27B 4K trace shows a real but modest command-model gap, not a giant one.
- Cheaper overlap and residency work comes before heavier command-graph surgery.

Acceptance gates:

- Only pursue after double-buffered decode and structural cleanup are measured.
- Require trace evidence of additional idle gap before escalating further.

### 13. KV-Q8 / Quantized KV Cache For Long Context

Optimizes: long-context decode, DFlash usefulness at long context, memory.

Current read:

- The first dense KV-Q8 prototype is a negative result on M4: despite exact
  append quantization and good output similarity, the current Q8 v4 main path
  makes attention slower than the tuned F16 path.
- Codex-wrap review says the likely cause is structural: scalar Q8 dequant/load
  overhead is overpowering stored-byte savings against an already-strong F16
  vectorized path.
- v0.416 audit + cx review reopens KV-Q8 only as a narrow reader micro-oracle,
  not as a broad retry of the falsified implementation. The reason to keep it
  alive is cross-cutting: it can reduce no-spec long-context attention bytes,
  lower KV memory pressure, and raise the context length where DFlash/MTP verify
  remains viable.
- v0.437 closes that micro-oracle for same-layout Q8_0. A vector-shaped Q8x4
  reader is correctness-clean but regresses A3B `attn-intra` at both 8K and 32K.

Expected payoff: still potentially large for compressed KV in theory, but Q8_0
in the current layout/body is not the path. Do not spend more blind sweep time on
this implementation family.

Risks and constraints:

- Easy time sink.
- Needs a different compression format, layout, or attention body to be worth
  revisiting; same-layout Q8_0 reader retunes are closed.

Acceptance gates:

- Revisit only with a concrete non-Q8_0 or different-layout structure and a fast
  `attn-intra` feedback plan that preserves Q-head grid parallelism and avoids
  large staged-KV TGM.
- Cut quickly if the reader does not beat tuned F16 at both 8K and 32K before
  end-to-end wiring.
- Promote only after a no-spec long-context row improves and DFlash/verify phase
  attribution shows the new reader moves the adaptive context threshold.

### 14. Dense Decode Surgery And Small Decode Hygiene

Optimizes: dense decode throughput and measurement integrity.

Why it stays late:

- v0.340 already banks the obvious dense GDN front-projection scheduling overlap
  as a default, with `QWEN_DECODE_DENSE_CONCURRENT_GDN=0` rollback.
- Dense decode is already competitive enough that prompt work dominates the
  scoreboard.
- Prior FFN mega-fusion had weak payoff and GDN recurrence semantics are
  correctness-sensitive.
- GPU argmax is already landed; dense gain is neutral within noise and MoE gain
  is modest but real.

Acceptance gates:

- Any decode surgery must be driven by a fresh dense phase profile identifying a
  specific waste pocket.
- First candidate if the profile supports it: GDN decode recurrence tiling across
  multiple `dv` rows to reduce launch/TG overhead and duplicated Q/K loads, but
  only if `gdn_step_decay` is materially above its unavoidable state R/W floor.
  The reducible component is the redundant `q_h`/`k_h` device-pointer reload
  across the `head_dim × n_v_heads` TG grid (currently ~384× per head per token
  for `head_dim=128, n_v/n_k=3`); the state R/W itself (~6 MiB/layer/token,
  ~288 MiB/token across 48 GDN layers on 27B) is irreducible without restructuring
  the recurrence. Budget the win at ~2-4% decode if k/q-bound; 0% if state-R/W-bound.
- Exact-token A/B paths (`--full-logits-decode`) and argmax regression tests stay
  green while decode work proceeds.

## Deprioritized For Now

- FFN mega-fusion as a first move: prior layer-major fusion produced too little
  gain for the complexity.
- Giant GDN recurrence rewrite: correctness risk is too high without a sharper
  measured target.
- Synthetic-only attention tuning: attention is much improved; further tuning
  should be driven by full phase/ctx sweeps.
- Vocab pruning or approximate lm_head shortcuts without exact-token gates.

## How To Update This File

When a session changes performance direction, update only the smallest relevant
section:

1. Add new measured baseline rows or replace stale ones.
2. Move ranked bets only when measurements change expected value or risk.
3. Record accepted wins in "Recent confirmed wins".
4. Record failed experiments in "Deprioritized" or in the relevant bet's risks.
5. Keep benchmark notes sequential and reproducible; do not mix parallel runs.
6. At each improved checkpoint, make the diff tell one optimization story:
   short `v0.xx:` subject, detailed wrapped body with measurement + validation.
7. When a win changes the broader lowering or bottleneck picture, update
   `docs/INFERENCE-GRAPH.md` alongside this file so the semantic map stays in
   sync with the engine and the current performance story.

Useful pattern for future entries:

```text
Decision: <what changed>
Evidence: <bench command + key numbers>
Impact: <models / contexts affected>
Risk: <remaining validation gap>
Next: <one concrete follow-up>
```
