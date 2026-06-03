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

Beat llama.cpp across dense and MoE Qwen 3.5/3.6 workloads, ideally by more
than a little, without taking shortcuts that fail at long context or larger
model shapes.

Primary guardrails:

- Dense: `Qwen3.6-27B-Q4_K_M.gguf`
- MoE A3B: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- MoE A10B: `Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf`
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

## Latest Baseline Snapshot

M4 Max, release `qwen-bench`, clean family rows after `v0.203` against pinned
llama.cpp b9481 (`bfb4308b`, `MTL,BLAS`). AC power, no recorded thermal or
performance warnings.

| Model | Shape | qwen | llama.cpp | qwen/lcpp | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 27B dense | `pp4096` | `221.51` | `211.19` | `1.05x` | v0.203 pinned b9481 |
| 27B dense | `pp16384` | `204.44` | `193.49` | `1.06x` | v0.203 pinned b9481 |
| 35B A3B | `pp4096` | `1540.69` | `1319.85` | `1.17x` | v0.203 pinned b9481 |
| 35B A3B | `pp16384` | `1182.87` | `1114.11` | `1.06x` | v0.203 pinned b9481 |
| 122B A10B | `pp4096` | `457.25` | `393.69` | `1.16x` | v0.203 pinned b9481 |
| 122B A10B | `pp16384` | `391.84` | `357.46` | `1.10x` | v0.203 pinned b9481 |

Current short/decode guardrails:

| Model | Shape | qwen | llama.cpp | qwen/lcpp | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 27B dense | `pp512` | `236.30` | `236.25` | `1.00x` | v0.203 paired repeat |
| 27B dense | `pp1024` | `237.93` | `217.95` | `1.09x` | v0.203 pinned b9481 |
| 35B A3B | `pp512` | `1449.75` | `1415.12` | `1.02x` | v0.203 pinned b9481 |
| 35B A3B | `pp1024` | `1620.35` | `1411.80` | `1.15x` | v0.203 pinned b9481 |
| 122B A10B | `pp512` | `453.66` | `445.65` | `1.02x` | v0.203 pinned b9481 |
| 122B A10B | `pp1024` | `504.43` | `430.68` | `1.17x` | v0.203 pinned b9481 |
| 27B dense | `tg128` | `24.46` | `20.01` | `1.22x` | v0.203 pinned b9481 |
| 35B A3B | `tg128` | `82.31` | `77.42` | `1.06x` | v0.203 pinned b9481 |
| 122B A10B | `tg128` | `35.76` | `35.21` | `1.02x` | v0.203 repeat |

Current caveats:

- The full v0.203 family `27B pp512` row showed `0.913x`, but immediate paired
  repeats showed `1.000x` and `1.008x`; treat that cell as parity/noise until a
  longer repeat packet says otherwise.
- Small dense short/medium prefill is not won across the board, but v0.215's
  paired GDN Q/K L2 prep narrows the live gap again. Promotion rows move 0.8B
  `pp512/pp1024` by about `+2-3%` versus rollback and 2B by about `+1%`, while
  4B/9B/27B canaries are neutral-positive. v0.206-v0.215 say this is not
  attention, fast-path coverage, GDN matvec fallback, command-encoder
  coalescing/streaming, NSG8 GDN-step grouping, or another Q5/Q6 N64 threshold
  fiddle. The remaining high-EV branch is deeper short-prompt FFN/GDN projection
  mechanics, especially the 0.8B and 2B `pp512` cells.
- A10B very-short prefill remains a real uncovered corner: the b9481 repeat had
  `pp128` at `0.852x` even though `pp512+` and `tg128` were won/parity. The
  v0.204 G16 threshold cleanup moves qwen-only `pp128` from `~220-223 t/s` to
  `~242-246 t/s` averaged, with warmed samples above `280 t/s`; it is improved
  but not yet a cold-average pinned-lcpp win.

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
  `Q2_K`, `Q3_K`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ4_NL`,
  `IQ4_XS`), so the local 0.8B quant family is clean across dense FFN, GDN,
  attention, and lm-tail coverage. Q2_K/Q3_K/IQ4_NL/IQ4_XS now have
  simdgroup_matrix prompt mat-mat tiles: clean local 0.8B low-bit rows move from
  `0.15-0.26x` llama.cpp at `pp1024` to `0.96-0.98x` across `pp1024/4096`.
  The v0176 dense validation generalizes this to 2B/9B Q2/Q3/IQ4_XS and 27B Q3:
  2B is `0.97-1.00x`, 9B is `0.99-1.06x`, and 27B Q3 `pp1024` is `1.08x`
  llama.cpp. Remaining explicit coverage gaps are MoE grouped expert-bank
  variants outside target quants and UD low-bit `IQ2/IQ3` dense tensors. A3B
  Q3/IQ4_XS MoE now has a generic GPU fallback for F32-dequant gate/up plus
  native IQ4_XS down, so it runs instead of crashing, but audit still correctly
  reports `0/40` grouped coverage by default because those files use `IQ3_XXS`
  gate/up expert banks with `IQ4_XS` down. Clean v0.189 A3B Q3 `pp1024` paired
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
  top runtime roadmap item.

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
- Dense KV-Q8 prototype is currently a negative result on M4 for the existing
  v4 main-kernel structure; attention gets slower, not faster.
- Attention v4 now supports `group=4`, unlocking the small dense family
  (0.8B / 2B / 4B / 9B) as real long-context canaries instead of failing back to
  the old threadgroup-memory-limited attention path. The local 9B sweep now runs
  cleanly through 32K: `64.0 t/s` at 4K, `59.5 t/s` at 16K, `53.8 t/s` at 32K.
- Attach-mode decode tracing is now practical via `qwen-bench decode-window`, and
  `scripts/profile/trace-metal.py` gives a compact Metal timeline summary
  without hand-written one-off parsers.
- Group16 attention tile4 default for 122B long context.
- Group6 dense attention `NWG=64` at `n_pos >= 4096`.
- `QWEN_ATTN_V4_NWG`, `QWEN_ATTN_V4_TILE_C`, and `QWEN_ATTN_V4_G16_TILE` A/B knobs.
- Production-style `NWG=64` correctness coverage for attention v4.
- MoE intra-block profiler: 122B block ~0.594 ms, with mixer prep largest.

Recent measured negatives:

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

## Force-Ranked Next Bets

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
- Third, finish A3B Q3 native IQ3 defaultability before opening another low-bit MoE
  microbranch: direct oracles are green and env perf beats llama.cpp, and the G8
  matrix VT bug is fixed. Treat the blk0 `qkv+alpha` GDN matvec repair as a
  correctness oracle, not the final performance answer; decide whether long
  default gates should be continuation/generation based before spending more
  kernel time on strict internal-state cosine.
- Fourth, keep A3B/A10B MoE in guardrail mode unless coverage drops below
  `40/40` or `48/48` or a warmed short-prompt row regresses. Do not revive
  hot-threshold or concentration-only branches without new distribution evidence.
- Fifth, small dense is the active paired mismatch again. Start from 0.8B `pp512`
  and require 2B plus 4B/9B/27B canaries before promotion. v0.215 landed the
  low-risk GDN prep dispatch cleanup; recent falsifiers say the next branch
  should target FFN/GDN projection mechanics or dataflow, not attention, encoder
  coalescing/streaming, GDN matvec fallback, NSG8 GDN-step grouping, or broad
  low-threshold N64 policy.
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

What the latest analysis says:

- Current MTP shape is structurally weak at long context because lazy verify does
  not amortize enough base work and MTP has its own growing KV attention cost.
- Current DFlash is made safe by adaptive verify `N`, but not fast, because the
  drafter and target still pay too much long-context attention work.

Highest-EV speculative kernel targets:

1. Target packed-verify multi-query attention so consecutive verify queries share
   KV reads.
2. DFlash two-range attention reading ctx-cache and noise directly, without
   `k_full` / `v_full` materialization.
3. Retile the `N=16` mat-mat specializations only if verify/draft phase profiles
   show N16 mat-mat-heavy surfaces remain material after the attention fixes.
4. Adaptive draft compute width, not only adaptive verify width.

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

Expected payoff: still potentially large in theory, but only if a materially
different reader structure wins. Do not spend more blind sweep time on the
current implementation.

Risks and constraints:

- Easy time sink.
- Needs a fundamentally better Q8 read path or a different compression format to
  be worth revisiting.

Acceptance gates:

- Revisit only with a concrete new kernel structure and a fast feedback plan.
- Cut again quickly if attention does not beat F16 at 32K or 64K.

### 14. Dense Decode Surgery And Small Decode Hygiene

Optimizes: dense decode throughput and measurement integrity.

Why it stays late:

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
