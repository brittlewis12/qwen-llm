# Performance Log

Append-only checkpoint log for qwen-llm performance work. Use this to answer
"where are we right now?" without re-running audits or reconstructing context
from chat history. Keep entries short, factual, and tied to measurements.

See also: `docs/PERF-ROADMAP.md` for the active force-ranked queue.

## 2026-05-25 — Mat-Mat Threadgroup-Memory Parity Spike

Status: dirty-code spike after `v0.126` to match llama.cpp's classic `mul_mm`
threadgroup-memory policy: full output tiles request `5120` bytes for NR1=16 or
`6144` bytes for NR1=32 instead of always requesting `8192`; partial tiles still
use `8192`. `QWEN_MATMAT_QK_LEGACY_SMEM=1` restores the old request size for A/B.
Raw rows are in `docs/bench/2026-05-25-matmat-smem-spike/`.

### Measurements

27B dense with matrix-G6/G8 enabled, `runs=3`, cooled sequential variants:

| Shape | legacy A | smem-new | legacy B | Read |
| --- | ---: | ---: | ---: | --- |
| `pp512` | `205.77` | `197.30` | `187.52` | noisy; reversed repeat puts smem around legacy |
| `pp1024` | `188.60` | `181.58` | `156.68` | noisy; reversed repeat straddles legacy |
| `pp4096` | `202.83` | `204.55` | `191.69` | small positive vs first anchor; late anchor drifted |
| `pp16384` | `176.33` | `176.74` | `175.72` | small positive vs both anchors |

Direct mat-mat microbench: at `N=512`, results are flat; at `N=1024`, Q4 gate/up
improve from `19.812/17.617 ms` to `14.567/14.853 ms`, Q6 down improves
`19.733 -> 16.119 ms`, and Q6 attn_qkv improves `11.194 -> 9.462 ms`. At
`N=4096`, the win is smaller: Q4 gate/up `58.791/58.477 -> 57.504/57.469 ms`,
Q6 down `62.615 -> 62.236 ms`, Q6 attn_qkv `38.724 -> 38.016 ms`.

Correctness: smem policy unit test passed; Q4/Q5/Q6/Q8 mat-mat correctness tests
passed; active 27B matrix-G6 prefill `T=32/P=32` passed logits/hidden/GDN/KV.

Read: keep as low-risk llama-parity cleanup with clear microbench support and
modest/noisy end-to-end upside. It is not the dense breakthrough; the remaining
gap still points to larger FFN/GDN execution or layout differences.

## 2026-05-25 — Dense F16-Inner/Q6-F16-Source Falsifier

Status: dirty-code spike from clean `v0.126` to test whether dense 27B FFN can
benefit from F16 inner scratch: fused Q4_K gate/up SwiGLU computed in F32, final
inner stored as F16, then Q6_K down reading F16 source rows. Correctness passed,
but perf did not clear the keep gate, so the production/env code was stripped and
only the falsifier artifacts were kept. Raw rows and command notes are in
`docs/bench/2026-05-25-dense-f16-inner-spike/`.

### Measurements

27B dense with matrix-G6/G8 enabled, `runs=3`, cooled sequential variants:

| Shape | matrix-G6 A | fused-Q4 | F16-inner | matrix-G6 B | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `pp4096` | `196.74` | `201.66` | `202.33` | `200.16` | only `+0.3%` over fused-Q4 |
| `pp16384` | `180.45` | `182.00` | `180.59` | `185.85` | loses to fused-Q4 and late baseline |

Direct Q6_K mat-mat microbench at `N=4096`, `dispatches=16` showed F16 source is
slower: `ffn_down` `61.604 ms -> 64.000 ms` (`1.039x` slower) and `attn_qkv`
`35.396 ms -> 35.953 ms` (`1.016x` slower).

Correctness: Q6 F16-source matched the existing F32-source half-staged path at
N=32, fused F16 inner bytes matched `scatter(F32 inner -> F16)`, down output
matched exactly, and active 27B prefill `T=32/P=32` passed logits/hidden/GDN/KV.

Read: the useful fact is negative: halving dense FFN inner precision is not the
next dense breakthrough on this kernel shape. Keep the existing F32-inner path;
next dense work should isolate FFN phase deltas versus llama.cpp rather than
retuning this F16-source branch.

## 2026-05-25 — Dense High-N Fused SwiGLU Q4 Spike

Status: env-only high-N dense FFN fusion spike on top of `v0.124`, followed by a
clean `v0.125` long-row repeat. The new path adds
`QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=1`, fusing dense Q4_K gate/up mat-mat plus
SwiGLU for chunks with at least 32 rows. Dirty spike rows are in
`docs/bench/2026-05-25-dense-fused-ffn-q4-spike/`; clean repeat rows are in
`docs/bench/2026-05-25-dense-fused-ffn-q4-clean-v0125/`.

### Measurements

27B dense with `QWEN_PREFILL_ATTN_MATRIX_G6=1`, `runs=3`:

| Shape | matrix-G6 baseline | fused-Q4 SwiGLU | fused/baseline | Read |
| --- | ---: | ---: | ---: | --- |
| `pp512` | `214.86` | `214.45` | `1.00x` | flat/slightly down |
| `pp1024` | `196.97` | `193.45` | `0.98x` | noisy regression |
| `pp4096` | `181.08` | `191.67` | `1.06x` | possible long win, baseline noisy |
| `pp16384` | `177.57` | `179.25` | `1.01x` | small true-long win |

Correctness: the generic fused kernel matches the unfused Q4_K gate/up+silu path
at both `N=16` and `N=32` with `min_cos=1.000000`, and an active 27B prefill
correctness run with `T=32/P=32` passed final logits, hidden captures, GDN state,
and KV checks.

Read: the high-N fusion hypothesis is real enough to keep as an env candidate,
but it does not yet clear a default gate. It helps long rows a little and may help
`pp4096`, but it is flat/regressive at `pp512/pp1024` and remains far short of the
`~10-12%` FFN speedup needed to beat lcpp by itself.

Clean `v0.125` repeat, `runs=3`:

| Shape | matrix-G6 baseline | fused-Q4 SwiGLU | fused/baseline | Read |
| --- | ---: | ---: | ---: | --- |
| `pp4096` | `182.05` | `191.62` | `1.05x` | confirmed long win |
| `pp16384` | `177.22` | `179.24` | `1.01x` | confirmed small true-long win |

## 2026-05-25 — Dense Matrix-G6 FFN Attribution And Mat-Mat Pointer Spike

Status: follow-up after the clean `v0.123` 27B family gate. The no-op budget rows
used the clean `v0.123` build; pointer-store rows are a dirty spike after changing
Q4/Q5/Q6/Q8 mat-mat threadgroup writes from `sa[idx]` to `*(sa + idx)`, matching
the spelling llama.cpp explicitly comments as faster. Raw rows are in
`docs/bench/2026-05-25-dense-g6-noop-budget/` and
`docs/bench/2026-05-25-matmat-pointer-sa-spike/`.

### Measurements

Matrix-G6 no-op budget, `runs=1`:

| Shape | baseline | no-FFN | no-GDN | no-attn | no-FFN-attn | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pp4096` | `180.50` | `541.59` | `193.30` | `193.93` | `600.26` | FFN dominates |
| `pp16384` | `173.12` | `421.55` | `192.37` | `192.25` | `552.30` | FFN dominates |

Pointer-store spike with `QWEN_PREFILL_ATTN_MATRIX_G6=1`, `runs=3`:

| Shape | pointer-store qwen | clean v0.123 family qwen | Read |
| --- | ---: | ---: | --- |
| `pp4096` | `188.77` | `186` | small possible win |
| `pp16384` | `176.13` | `176` | flat/small possible win |

Correctness: Q4_K, Q5_K, Q6_K, and Q8_0 mat-mat correctness tests all passed.

Read: the no-op budget sharpens the next dense priority to FFN mat-mat/kernel
layout rather than more attention work. The pointer-store spelling is worth
keeping because it aligns with llama.cpp and is semantics-preserving, but it is
only a small spike, not the FFN breakthrough.

## 2026-05-25 — Dense Group-6 Matrix Family Gate

Status: clean `v0.122` 27B-vs-llama.cpp family gate after adding a long-prefix
G6 correctness test. `qwen-bench` was rebuilt from `62a9114a6`, workloads were
sequential on AC power, and post-run `pmset` / memory-pressure checks stayed
clean. Raw rows are in
`docs/bench/2026-05-25-0246-27B-matrix-g6-v0122-family/`.

### Measurements

27B dense, `QWEN_PREFILL_ATTN_MATRIX_G6=1 QWEN_PREFILL_ATTN_MATRIX_G8=1`,
`runs=3`, default chunk policy:

| Shape | llama.cpp | qwen | qwen/lcpp | Read |
| --- | ---: | ---: | ---: | --- |
| `pp128` | `213` | `198` | `0.93x` | still behind |
| `pp512` | `222` | `211` | `0.95x` | near parity |
| `pp1024` | `205` | `202` | `0.99x` | parity |
| `pp4096` | `198` | `186` | `0.94x` | still behind |
| `pp16384` | `188` | `176` | `0.93x` | still behind |
| `tg32` | `21` | `23` | `1.13x` | decode win |
| `tg128` | `21` | `23` | `1.12x` | decode win |

Correctness: the new ignored G6 prefix gate primes `4096` tokens through
single-token decode, then compares an 8-token matrix prefill extension; first
run passed with logits, GDN, and KV cosines at `1.000000`.

Read: matrix-G6 converts dense long prefill from a catastrophic `0.67x` 16K gap
in the earlier family sweep to a smaller `0.93x` gap, but it does not surpass
lcpp. The next dense work should target the remaining FFN/GDN mat-mat wall and
matrix-attention residuals rather than re-proving matrix-G6.

## 2026-05-24 — Dense Group-6 Matrix Clean Repeat

Status: clean `v0.120` repeat gate after rebuilding `qwen-bench` from commit
`3eb923e71`. Runs were sequential on AC power; post-run `pmset` reported no
thermal/performance/CPU-power warning and `memory_pressure -Q` reported `95%`
free. Raw rows are in `docs/bench/2026-05-24-dense-g6-clean-repeat-v0120/`.
The earlier accidental pre-commit clean rows are in
`docs/bench/2026-05-24-dense-g6-clean-repeat/` and are not the canonical gate.

### Measurements

27B dense, `QWEN_PREFILL_ATTN_MATRIX_G6=1`, `runs=3`, default chunk policy:

| Shape | baseline | matrix-g6 | matrix/baseline | Read |
| --- | ---: | ---: | ---: | --- |
| `pp128` | `188.33` | `198.42` | `1.05x` | short win |
| `pp512` | `195.85` | `210.99` | `1.08x` | medium win |
| `pp1024` | `175.66` | `188.60` | `1.07x` | medium win, noisy |
| `pp4096` | `151.87` | `185.85` | `1.22x` | long win |
| `pp16384` | `125.21` | `173.84` | `1.39x` | large true-long win |

Read: the dense group-6 matrix-attention branch survives the clean repeat and now
has enough evidence to be treated as a promotion candidate rather than a spike.
It still does not close the full dense gap to lcpp, and it still needs stronger
G6 long-prefix correctness/coverage before becoming a default.

## 2026-05-24 — Dense Group-6 Matrix Attention Spike

Status: dirty-code spike from checkpoint `1a211ceb5` after generalizing the
env-only matrix-attention sidecar from A3B/group-8 to runtime group shapes and
adding `QWEN_PREFILL_ATTN_MATRIX_G6=1` for the 27B dense group-6 shape. Raw rows
are in `docs/bench/2026-05-24-dense-g6-matrix-spike/`; the post-rename smoke row
is in `docs/bench/2026-05-24-dense-g6-post-rename-smoke/`. Treat these rows as
evidence for expected value, not as a clean promotion gate.

### Measurements

27B dense, synthetic prompt prefill, `runs=1`, default chunk policy:

| Shape | baseline | matrix-g6 | matrix/baseline | Read |
| --- | ---: | ---: | ---: | --- |
| `pp128` | `187.84` | `198.45` | `1.06x` | no short-prompt harm in spike |
| `pp512` | `197.95` | `213.06` | `1.08x` | medium win |
| `pp1024` | `194.01` | `210.70` | `1.09x` | medium win |
| `pp4096` | `153.42` | `187.54` | `1.22x` | long win |
| `pp16384` | `125.74` | `176.00` | `1.40x` | large true-long win |

Correctness/validation so far: `cargo fmt --check`, `git diff --check`, release
`qwen-bench` rebuild, small 27B prefill-vs-single with
`QWEN_PREFILL_ATTN_MATRIX_G6=1 QWEN_PREFILL_ATTN_MATRIX_MAX_POS=32`, and the
full ignored A3B matrix correctness gate with `QWEN_PREFILL_ATTN_MATRIX_G8=1`.
The post-rename G6 smoke row at `pp4096` was `188.50 t/s`, matching the original
spike. Remaining blockers are clean repeated 27B rows and better long-prefix G6
correctness coverage.

Related dense budget row: `docs/bench/2026-05-24-dense-pp16k-combined-noop/`
shows 27B `pp16384` baseline `127.57 t/s`, no-FFN `208.71`, no-attn `193.48`,
no-FFN+no-attn `580.91`, and no-FFN+no-attn+no-GDN `845.88`. Read: dense 16K is
jointly attention and FFN/GDN limited; matrix-g6 attacks a real wall but does not
make dense solved.

## 2026-05-24 — A3B Matrix Promotion Repeat Gate

Status: clean repeated A3B promotion evidence after checkpoint `ef32caec5`.
`qwen-bench` was rebuilt from that commit, workloads were sequential on AC power,
and post-run `pmset` / memory-pressure checks stayed clean. Raw rows are in
`docs/bench/2026-05-24-1934-35B-A3B-matrix-promotion-repeat-family/` and the
chunk-2048 interaction rows are in
`docs/bench/2026-05-24-a3b-matrix-chunk2048-repeat/`.

### Measurements

Repeated family gate, `QWEN_PREFILL_ATTN_MATRIX_G8=1`, `runs=3`, default chunk
policy:

| Shape | llama.cpp | qwen | qwen/lcpp | Read |
| --- | ---: | ---: | ---: | --- |
| `pp128` | `753.73` | `832.77` | `1.10x` | win |
| `pp512` | `1332.61` | `1319.10` | `0.99x` | parity, not a win |
| `pp1024` | `1325.52` | `1473.75` | `1.11x` | win |
| `pp4096` | `1238.34` | `1365.97` | `1.10x` | win |
| `pp16384` | `1025.47` | `1094.78` | `1.07x` | win |
| `tg32` | `69` | `78` | `1.13x` | decode win |
| `tg128` | `69` | `78` | `1.13x` | decode win |

Chunk-2048 interaction, qwen only, `runs=3`:

| Shape | default chunk qwen | chunk2048 qwen | chunk2048/default | Read |
| --- | ---: | ---: | ---: | --- |
| `pp128` | `832.77` | `755.36` | `0.91x` | do not blanket-default |
| `pp512` | `1319.10` | `1276.46` | `0.97x` | do not blanket-default |
| `pp1024` | `1473.75` | `1453.75` | `0.99x` | flat/slightly down |
| `pp4096` | `1365.97` | `1451.14` | `1.06x` | useful long-prompt win |
| `pp16384` | `1094.78` | `1101.85` | `1.01x` | small long-prompt win |

Read: the A3B/G8 matrix sidecar now has repeated clean evidence against lcpp
through 16K, with only `pp512` sitting at parity instead of a win. Chunk `2048`
is a long-prompt tuning candidate, not a universal MoE default: it helps at
`pp4096+` but regresses short/medium A3B rows in this gate. The remaining matrix
promotion blockers are correctness/tolerance policy, production default gating,
and coverage evidence, not proving the mechanism again.

## 2026-05-24 — Chunk Policy Is A MoE Lever, Not A Dense Cure

Status: follow-up to the expanded matrix family sweep. All rows used clean
`2d38ed5cb`, ran sequentially on AC power, and captured `pmset` / memory-pressure
snapshots. Raw rows are in `docs/bench/2026-05-24-chunk-policy-matrix/` and the
selected repeated gates are in
`docs/bench/2026-05-24-chunk-policy-matrix-repeat/`.

### Measurements

Single-run chunk sweep with `QWEN_PREFILL_ATTN_MATRIX_G8=1`:

| Model | Shape | Best chunk | Best vs chunk1024 | Read |
| --- | ---: | ---: | ---: | --- |
| 9B dense | `pp4096` | `512` | `1.01x` | larger chunks lose |
| 9B dense | `pp16384` | `2048` | `1.00x` | flat |
| 27B dense | `pp4096` | `1024` | `1.00x` | flat |
| 27B dense | `pp16384` | `2048` | `1.00x` | flat |
| 35B A3B MoE | `pp4096` | `2048` | `1.03x` | medium-long win |
| 35B A3B MoE | `pp16384` | `4096` | `1.08x` | long win, noisy |
| 122B A10B MoE | `pp4096` | `4096` | `1.09x` | clear long win |
| 122B A10B MoE | `pp16384` | `2048` | `1.06x` | clear long win |

Repeated selected gates (`runs=3`) sharpened the read:

| Model | Shape | chunk1024 | chunk2048 | chunk4096 | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| 27B dense | `pp16384` | `125.09` | `124.52` | n/a | no dense recovery |
| 35B A3B MoE | `pp16384` | `1114.61` | `1148.52` | `1123.51` | `2048` is safest |
| 122B A10B MoE | `pp4096` | `357.87` | `373.80` | `378.40` | larger is better |
| 122B A10B MoE | `pp16384` | `288.99` | `302.54` | `306.46` | larger is better |

Read: raising the MoE long-prompt default chunk cap is now a plausible small
production win, especially for A10B, but it is not the whole-family answer. Dense
prefill's 16K gap survives chunk sweeps almost unchanged, so the next dense work
needs phase attribution / attention-dataflow evidence rather than a chunk knob.
For MoE, `2048` is the conservative cross-MoE cap candidate; `4096` is A10B's
best measured long-prompt point but is less stable for A3B.

## 2026-05-24 — Matrix Family Sweep Adds 4K/16K

Status: expanded family scoreboard for clean `2d38ed5cb` with the env-only
group-8 matrix-attention sidecar. GPU workloads were sequential on AC power;
post-run `pmset` reported no thermal/performance/CPU-power warning and
`memory_pressure -Q` reported `95%` free. Raw rows are in
`docs/bench/2026-05-24-1331-matrix-pp4k16k-family/`.

### Scope

- `QWEN_PREFILL_ATTN_MATRIX_G8=1`, `runs=1`; shapes were
  `pp128/512/1024/4096/16384` plus `tg32/tg128`.
- The sweep intentionally kept token generation rows; no `no-tg` shortcut or
  harness behavior change was introduced.
- `qwen-bench` build stamp was clean `2d38ed5cb`; `llama.cpp` was `14aa3d375`
  build `9265` on the same M4 Max.

### Measurements

| Variant | `pp512` qwen/lcpp | `pp1024` | `pp4096` | `pp16384` | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| 27B dense | `211 / 243` (`0.87x`) | `206 / 230` (`0.89x`) | `179 / 216` (`0.83x`) | `136 / 203` (`0.67x`) | dense long prefill still not cracked |
| 35B A3B MoE | `1387 / 1392` (`1.00x`) | `1558 / 1388` (`1.12x`) | `1492 / 1352` (`1.10x`) | `1212 / 1107` (`1.09x`) | matrix candidate beats lcpp through 16K |
| 122B A10B MoE | `369 / 444` (`0.83x`) | `410 / 445` (`0.92x`) | `408 / 408` (`1.00x`) | `309 / 360` (`0.86x`) | group-16 MoE remains open |

Decode stayed won/parity at `tg128`: dense rows were `1.05-1.36x`, A3B was
`1.04x`, and A10B was `1.00x`.

Read: the A3B/G8 matrix candidate is now a real lcpp-cracking branch across the
family sweep's prompt sizes, not just isolated spot rows. This does not prove a
whole-family win or default readiness: dense prefill degrades badly by 16K, A10B
still lags outside the `pp4096` parity point, and every qwen row drops from
`pp4096` to `pp16384`. The immediate high-EV follow-up is a prompt chunk policy
sweep (`512/1024/2048/4096`, memory permitting) before writing more kernels.

## 2026-05-24 — Matrix Attention Scratch Uses Prompt Length In Bench Paths

Status: production-shape cleanup for the A3B/group-8 matrix-attention sidecar.
GPU workloads were run sequentially on AC power; raw rows are in
`docs/bench/2026-05-24-matrix-auto-scratch/`.

### What Changed

- Added `MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos`, so
  callers can size matrix-attention score/V_T scratch for the actual last prompt
  position instead of relying on `QWEN_PREFILL_ATTN_MATRIX_MAX_POS`.
- Updated `qwen-bench pp`, `pp-wait`, and packed `decode` prefill paths to pass
  the rendered/synthetic prompt length into prefill scratch allocation.
- Left the matrix path opt-in via `QWEN_PREFILL_ATTN_MATRIX_G8=1`; this removes
  the manual max-pos env wart without defaulting the branch yet.

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `QWEN_PREFILL_ATTN_MATRIX_G8=1 cargo test --release -p qwen-llm prefill_tokens_matches_single_token_loop_35b_a3b_moe -- --ignored --nocapture`
  - passed with no `QWEN_PREFILL_ATTN_MATRIX_MAX_POS` env; logits cosine stayed
    `0.999984-1.000000`, worst GDN cosine `0.999611`.

### Measurements

All rows below set `QWEN_PREFILL_ATTN_MATRIX_G8=1` and intentionally omit
`QWEN_PREFILL_ATTN_MATRIX_MAX_POS`.

| Shape | Tokens/s | Notes |
| --- | ---: | --- |
| A3B `pp320`, chunk `320` | `1237.87` | beats fresh lcpp `1174.57` spot |
| A3B `pp512`, chunk `512` | `1390.93` | beats fresh lcpp `1347.79` spot |
| A3B `pp1024`, chunk `1024` | `1561.06` | beats fresh lcpp `1345.07` spot |
| A3B `pp4096`, chunk `1024` | `1492.39` | beats fresh lcpp `1259.21` spot |
| A3B synthetic `pp34502`, chunk `2048` | `928.62` | beats fresh lcpp `865.50` spot |
| A3B real `v02_reva` `34502`, chunk `2048` | `926.76` | real rollout holds |
| A10B `pp512`, matrix flag + warm banks | `401.16` | group-16 ignores matrix-g8 path |

Read: the manual max-pos env was the first production wart, and it is now gone
for the bench/user-facing prompt paths. The branch still needs cooled repeated
promotion rows and a decision on the looser matrix correctness envelope before it
should become default, but the operational shape is now much closer to shippable.

## 2026-05-23 — A3B Q6 Down No Longer Falls Off Grouped MoE

Status: major A3B MoE prefill fix. GPU workloads were run sequentially on AC
power; raw compact rows are in `docs/bench/2026-05-23-q6-grouped-down/`.

### What Changed

- Added `kernel_moe_down_q6_K_f32_grouped_slots`, using the existing grouped Q5
  down execution shape with Q6_K dequant and 210-byte block stride.
- Added a Rust encoder for grouped Q6_K down and changed the grouped routed MoE
  gate to accept down experts in either `Q5_K` or `Q6_K`.
- Routed grouped prefill now dispatches the down stage by dtype, so A3B layers
  `blk.34`, `blk.38`, and `blk.39` no longer fall through to the per-token MoE
  fallback.

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `cargo test --release -p qwen-llm prefill_tokens_matches_single_token_loop_35b_a3b_moe -- --ignored --nocapture`
  - full ignored A3B gate passed; logits cosine `0.999985-1.000000`, worst GDN
    cosine `0.999611` across the packed-attention active cases.
- `cargo test --release -p qwen-llm prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke -- --nocapture`
  - A10B smoke passed; final-logits cosine `0.999934`, KV K/V minima
    `0.999567` / `0.999414`.
- Metal trace label count on A3B `pp512` now reports `moe-route-fused:40`,
  `moe-routed-grouped:40`, `moe-shared-packed:40`; the prior trace clue was
  `37` grouped routed layers.

### Measurements

| Model | Shape | Before Anchor | After | Read |
| --- | ---: | ---: | ---: | --- |
| A3B | `pp128`, chunk `128`, runs `3` | `~655 t/s` recent default | `783.11 t/s` | `1.20x` |
| A3B | `pp256`, chunk `256`, runs `3` | `~801 t/s` recent default | `1000.57 t/s` | `1.25x` |
| A3B | `pp320`, chunk `320`, runs `3` | `~830 t/s` recent default | `1061.45 t/s` | `1.28x` |
| A3B | `pp512`, chunk `512`, runs `3` | `824.46 t/s` recent default | `1172.07 t/s` | `1.42x` |
| A3B | `pp1024`, chunk `1024`, runs `3` | `910.43 t/s` recent default | `1287.43 t/s` | `1.41x` |
| A3B | `pp2048`, chunk `1024`, runs `3` | n/a | `1265.48 t/s` | medium win holds |
| A3B | `pp4096`, chunk `1024`, runs `2` | n/a | `1198.11 t/s` | noisy `1178-1218` |
| A3B | `v02_reva`, `34,502` tok, chunk `1024` | `587.60 t/s` same fixture | `674.28 t/s` | `1.15x` true-long |
| A10B | warmed `pp512`, chunk `512`, runs `2` | `~385.13 t/s` | `385.56 t/s` | neutral |
| A10B | warmed `pp1024`, chunk `1024`, runs `2` | `~418 t/s` | `424.99 t/s` | neutral/slightly up |

Read: the A3B medium-prompt gap was not just a bad grouped-SwiGLU tile; three
late Q6_K down-expert layers were silently escaping the optimized grouped routed
path. This validates the path-coverage lens and weakens conclusions drawn from
older A3B grouped-kernel microsearches that were running around a mixed fast/slow
layer set. It does not prove Q6 grouped down is now optimal; it proves the gross
fallback is gone. The next lcpp sprint should start with strict qwen-vs-llama
per-layer/per-op differential attribution and a dtype/layer fast-path coverage
gate, not another local Q4/Q5 knob sweep.

Fresh post-Q6 no-op ceilings keep routed MoE in the high-EV set, but attention is
also large enough to cover the residual lcpp gap at medium/4K prompts:

| Shape | Baseline | No-op Attention Body | No-op Routed MoE | No-op Shared MoE |
| --- | ---: | ---: | ---: | ---: |
| A3B `pp1024` | `1287.43 t/s` | `1578.34 t/s` | `1979.46 t/s` | `1332.52 t/s` |
| A3B `pp4096` | `1198.11 t/s` | `1583.70 t/s` | `1816.71 t/s` | n/a |

Read: shared MoE is not the next lever. At `pp1024`, both attention body and
routed MoE have enough budget to explain the remaining lcpp delta; at `pp4096`,
the same is true but routed remains the larger no-op ceiling. Attribution, not
another blind kernel branch, should pick the next attack.

Fresh same-session llama.cpp anchors with `-fa 0`, `n_batch=2048`,
`n_ubatch=512`, and `has tensor = false` shrink the calibrated gap a lot versus
older lcpp rows:

| Shape | qwen | llama.cpp | qwen/lcpp |
| --- | ---: | ---: | ---: |
| A3B `pp320` | `1061.45 t/s` | `1174.57 t/s` | `0.90x` |
| A3B `pp512` | `1172.07 t/s` | `1347.79 t/s` | `0.87x` |
| A3B `pp1024` | `1287.43 t/s` | `1345.07 t/s` | `0.96x` |
| A3B `pp4096` | `1198.11 t/s` | `1259.21 t/s` | `0.95x` |
| A3B `pp34502` / `v02_reva` | `674.28 t/s` | `865.50 t/s` | `0.78x` |

Read: after the Q6 escape fix, medium A3B is close enough that the next win must
be chosen by paired attribution, not scoreboard intuition. True-long is still the
largest remaining A3B prefill gap.

The existing env-only A3B matrix-attention sidecar composes with the Q6 fix and
changes the true-long picture again. Correctness was rerun with matrix attention
enabled (`QWEN_PREFILL_ATTN_MATRIX_G8=1`, `QWEN_PREFILL_ATTN_MATRIX_MAX_POS=8193`)
through the full ignored A3B prefill-vs-single gate; it passed with the same
looser matrix tolerance envelope (`0.999984-1.000000` logits cosine, worst GDN
cosine `0.999611`).

| Shape | qwen default | qwen matrix sidecar | llama.cpp | Read |
| --- | ---: | ---: | ---: | --- |
| A3B `pp320` | `1061.45 t/s` | `1190.45 t/s` | `1174.57 t/s` | beats lcpp spot |
| A3B `pp512` | `1172.07 t/s` | `1340.82 t/s` | `1347.79 t/s` | parity |
| A3B `pp1024` | `1287.43 t/s` | `1484.98 t/s` | `1345.07 t/s` | beats lcpp spot |
| A3B `pp4096` | `1198.11 t/s` | `1434.39 t/s` | `1259.21 t/s` | beats lcpp spot |
| A3B synthetic `pp34502` | `691.12 t/s` | `878.72 t/s` | `865.50 t/s` | beats lcpp spot |
| A3B real `v02_reva` `34502` | `674.28 t/s` | `870.94 t/s` | n/a | real rollout holds |

Read: the highest-EV next production branch is no longer speculative. It is to
turn the matrix-attention sidecar into a safe default candidate for A3B/group-8:
remove/manualize less of the `MAX_POS` scratch policy, tighten or explicitly own
the matrix correctness tolerance, repeat cooled rows, and check dense/A10B no-
regression. Routed MoE remains a large no-op ceiling, but matrix attention plus
Q6 already cracks the lcpp A3B prefill board in spot rows.

## 2026-05-23 — llama.cpp MoE Win Is Not Metal Tensor API On This Box

Status: differential recon after the atomic-bucket falsifier. GPU workloads were
run sequentially on AC power.

### What Changed

- Checked llama.cpp's `kernel_mul_mm_id` tensor path. On this M4 Max,
  `llama-bench` reports `has tensor = false`; even `GGML_METAL_TENSOR_ENABLE=1`
  cannot make the Metal4 tensor branch live because the device family gate is
  not satisfied.
- Ran the same A3B `pp512` shape through llama.cpp and qwen under Metal System
  Trace. The traces are saved under `target/profiles/`:
  - `llama-a3b-pp512-metal.trace`
  - `qwen-a3b-pp512-metal.trace`
  - `qwen-a3b-pp512-metal-counters.trace`
- Used `~/code/gguf` to verify the A3B/A10B files contain separate
  `ffn_gate_exps` and `ffn_up_exps` tensors, not `ffn_gate_up_exps`, so
  llama.cpp is taking the separate gate/up MoE graph path for these files.

### Measurements

- llama.cpp A3B `pp512`, `-fa 0`, `--no-warmup`: `1242.44 t/s`, with
  `has tensor = false`.
- qwen A3B `pp512`, chunk `512`, same AC session: `739.89-745.93 t/s` on the
  trace runs, `build_dirty=1` because local diagnostics/docs were present.
- Default `Metal System Trace` produced timeline tables, but useful hardware
  counters were not available: default capture only exposed `RT Unit Active`,
  and adding `--instrument "Metal GPU Counters"` warned that the selected
  counter profile is unsupported on this target device and produced empty
  counter tables.

Read: llama.cpp's remaining A3B MoE prefill advantage on this machine is not a
hidden Metal tensor-API advantage and not a fused gate/up tensor ABI. The exact
target is the non-tensor simdgroup `mul_mm_id` / graph execution shape, plus
whatever memory-system behavior falls out of that shape. For stable ALU/bandwidth
counters, `xctrace` CLI is not enough here; use Xcode GPU capture or add an
in-process `MTLCounterSampleBuffer` path before making counter-driven claims.

### All-`n32` Recheck

Rechecked the closest local proxy to llama.cpp's `NR1=32` `mul_mm_id` tile:
full-tail grouped-Q4 all-`n32` versus all-`n16` remains a strong isolated win,
but forcing all-`n32` still does not convert end-to-end over the current default
hot-`n32` path.

| Model | Gate | Default | All-`n32` | Read |
| --- | --- | ---: | ---: | --- |
| A3B | grouped-tail proof `chunk512` | `8.50 ms` all-`n16` | `4.74 ms` all-`n32` | `1.792x`, exact |
| A10B | grouped-tail proof `chunk512` | `13.95 ms` all-`n16` | `10.08 ms` all-`n32` | `1.384x`, exact |
| A3B | pp512 E2E | `824.46 t/s` | `826.40 t/s` | flat |
| A3B | pp1024 E2E | `910.43 t/s` | `903.06 t/s` | slight regression |
| A10B | warmed pp512 E2E | `385.13 t/s` | `369.31 t/s` | regression |

Read: the old all-`n32` kill is still valid after rerun. The default hot-`n32`
gate already captures the high-count win; applying `n32` to cold buckets adds
overhead and/or loses occupancy. Do not retread all-`n32` as the lcpp crack.

## 2026-05-23 — Atomic Bucket Order Is Not The MoE Tail Crack

Status: exact routed-tail diagnostic after the Q5-down parity audit. GPU workloads
were run sequentially on AC power.

### What Changed

- Extended `run_grouped_swiglu_down_backend_profile` to build a second exact route
  ledger with the fused atomic top-k bucketer, then run the same grouped
  `SwiGLU -> Q5 down -> weighted_sum` tail against both ledgers.
- Added bucket-order diagnostics for scan/atomic ledgers by counting expert-ID
  back edges in the packed token stream.

### Measurements

| Model | Chunk | Scan Back Edges | Atomic Back Edges | Live Tail | Atomic Tail | Atomic Speedup | Correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A3B | `512` | `0/3987` | `1451/3987` | `3.99 ms` | `4.01 ms` | `0.994x` | exact |
| A10B | `512` | `0/4002` | `1568/4002` | `10.90 ms` | `10.45 ms` | `1.043x` | exact |

Split attribution stayed familiar: A3B was `split_gate_up=2.60 ms`,
`split_down=1.35 ms`, `split_reduce=0.08 ms`; A10B was
`split_gate_up=6.89 ms`, `split_down=3.78 ms`, `split_reduce=0.14 ms`.

Read: the fused atomic ledger is much less expert-sorted, but the grouped routed
tail is flat to slightly faster. Bucket ordering/locality in the route ledger is
therefore not the missing MoE prefill lever. The remaining exact Q5-down gap is
more likely inside the grouped projection/dequant/dataflow itself, not in scan
versus atomic bucket construction.

## 2026-05-23 — Bench Rows Capture Power Context

Status: methodology cleanup after discovering routed-tail rows had been run while
the machine was on low battery. GPU reruns below were on AC power.

### What Changed

- `qwen-bench` now records a lightweight macOS `pmset` power snapshot in JSON
  rows and prints the same summary in text mode for `pp`, `tg`, `pp-wait`, and
  `decode` benches.
- Perf docs now treat battery power, battery warnings, and thermal/performance
  warnings as benchmark identity/confounds unless an AC rerun confirms the row.
- The old `prefill_chunk=1024` cap is documented as a safe default, not a
  principled long-context optimum; keeper long-prompt work should sweep larger
  chunks when scratch allows.

### AC Rerun Sanity

`pmset`: AC power, charging, no recorded thermal/performance/CPU-power warning.

| Model | Chunk | Live Tail | Fused Tail | Speedup | `split_gate_up` | `split_down` | `split_reduce` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3B | `512` | `3.87 ms` | `3.83 ms` | `1.010x` | `2.47 ms` | `1.39 ms` | `0.07 ms` |
| A3B | `1024` | `6.14 ms` | `6.31 ms` | `0.972x` | `4.62 ms` | `2.53 ms` | `0.14 ms` |
| A10B | `512` | `10.75 ms` | `10.15 ms` | `1.059x` | `6.47 ms` | `3.58 ms` | `0.14 ms` |
| A10B | `1024` | `17.97 ms` | `17.69 ms` | `1.016x` | `11.84 ms` | `6.56 ms` | `0.27 ms` |

Read: AC power confirms the main conclusion. Fused gate/up is not a general ABI
winner, weighted sum is tiny, and grouped Q5 down remains the secondary routed
tail bucket after grouped gate/up/SwiGLU.

## 2026-05-23 — Fused Gate/Up Does Not Survive Full Routed Tail

Status: diagnostic checkpoint after adding full-tail fused-bank attribution to the
ignored grouped `SwiGLU+down` profile. GPU workloads were run sequentially.

### What Changed

- Extended `run_grouped_swiglu_down_backend_profile` to compare the live grouped
  routed tail against an exact interleaved gate/up bank through grouped SwiGLU,
  grouped Q5 down, and weighted sum.
- Added `chunk_p=1024` A3B/A10B profile gates so the fused-bank question is no
  longer inferred from isolated SwiGLU microprofiles.
- Tried and reverted an lcpp-shaped grouped-down final-store probe that spread
  writes across all four simdgroups and used `float4` stores.

### Measurements

Full grouped routed tail with fused gate/up bank:

| Model | Chunk | Live Tail | Fused Tail | Speedup | Correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| A3B | `320` | `3.74 ms` | `2.96 ms` | `1.264x` | exact |
| A3B | `512` | `4.05 ms` | `3.77 ms` | `1.074x` | exact |
| A3B | `1024` | `6.73 ms` | `6.67 ms` | `1.008x` | exact |
| A10B | `320` | `7.11 ms` | `6.96 ms` | `1.022x` | exact |
| A10B | `512` | `9.61 ms` | `9.55 ms` | `1.006x` | exact |
| A10B | `1024` | `16.79 ms` | `16.66 ms` | `1.008x` | exact |

Down final-store vectorization falsifier:

- A3B `chunk_p=512` exact output, but `split_down_reduce` regressed from
  `1.38 ms` to `1.57 ms` (`0.875x`). The probe was reverted.

Grouped Q5 down `n16` falsifier:

- A3B `chunk_p=512` exact output, but all-`n16` grouped Q5 down regressed
  `split_down_reduce` from `1.37 ms` to `1.70 ms` (`0.807x`). The probe was
  reverted.

Down/reduce split after reverting the vecstore probe:

- A3B `chunk_p=512`: `active=109`, `p50=28`, `p90=70`, `max=165`,
  `ge16/ge32/ge48=74/44/25`, `split_down=1.36 ms`, `split_reduce=0.07 ms`.
- A10B `chunk_p=512`: `active=94`, `p50=28`, `p90=92`, `max=177`,
  `ge16/ge32/ge48=65/44/29`, `split_down=3.38 ms`, `split_reduce=0.11 ms`.
- Read: the reducer is not the meaningful bucket; the grouped Q5 down matmul is.

### Current Read

- Interleaved gate/up remains a narrow A3B `chunk320` win, but it is killed as a
  general/default expert-bank ABI. The old duplicate-resident end-to-end proof
  converting only `~1.02x` now has a causal explanation: the isolated SwiGLU win
  mostly evaporates once grouped down, weighted sum, and real bucket geometry are
  included.
- Do not spend loader/ABI complexity on fused gate/up replacement unless a future
  product target is explicitly A3B medium-short prompts.
- The next exact MoE prefill work should target grouped Q5 down itself or a truly
  different `SwiGLU+down` dataflow that preserves grouped-down locality. Local
  lcpp-like final-store vectorization and weighted-sum cleanup are not enough.

## 2026-05-23 — Matrix Attention Is No Longer The Main A3B `pp4096` Gap

Status: diagnostic checkpoint after the A3B/group-8 matrix sidecar. GPU workloads
were run sequentially; true-long rows showed order/sag noise and should not be
used as promotion evidence.

### What Changed

- Re-checked the remaining A3B prefill budget with matrix attention held fixed
  (`QWEN_PREFILL_ATTN_MATRIX_G8=1`, `QWEN_PREFILL_ATTN_MATRIX_MAX_POS=4096`).
- Falsified three local attention follow-ons and reverted their code:
  direct KQV stores into qwen's row-major output, vectorized temp-to-output KQV
  copies, and F16 probability scratch for matrix KQV.

### Measurements

Cooled `pp4096`, chunk `1024`, one-run variants:

| Variant | Tokens/s | Read |
| --- | ---: | --- |
| baseline-a | `929.47` | matrix sidecar anchor |
| noop-attn | `997.19` | attention body now only `+3-7%` |
| noop-routed | `1270.08` | routed MoE is a much larger lever |
| noop-ffn | `2497.99` | broad FFN upper bound remains enormous |
| baseline-b | `971.90` | run-order drift still visible |

Attention follow-on probes:

- Direct KQV final stores were correctness-sensitive and regressed `pp4096`
  materially (`~1042 -> ~970 t/s` spot read), so the row-major write cannot be
  fixed by a naive accumulator-store rewrite.
- Vectorized KQV temp copy also regressed (`~972 t/s` spot read).
- F16 probability scratch failed the current matrix oracle max-abs limit
  (`~4.6e-2` with `cos=1.0`) and was flat/slightly slower in a cooled `pp4096`
  sweep (`926.40` vs `933.30` / `929.91` baselines).

Grouped routed-tail microprofiles remain consistent with the old MoE read:

| Model | Chunk | Tail ms | `grouped_swiglu` | `grouped_down+reduce` |
| --- | ---: | ---: | ---: | ---: |
| A3B | `512` | `3.84` | `2.31` | `1.55` |
| A10B | `512` | `10.30` | `6.81` | `4.25` |

Additional negative after this checkpoint: distributing the hot `n32` grouped
SwiGLU/down final scatter across all 128 threads improved the down sub-bucket but
regressed A3B `chunk_p=512` live routed tail overall (`3.84 -> 3.97 ms`), so it
was reverted.

Fused grouped finalizer also failed as a quick cleanup lever at matrix `pp4096`:
`QWEN_PREFILL_MOE_FUSED_FINALIZER=1` measured `892.60 t/s` between `933.51` and
`941.87 t/s` baselines, so the extra-pass cleanup is not the current crack.

The offline expert-bank hypothesis remains alive: the existing duplicate-bank
microprofile still shows fused gate/up bank wins at `chunk_p=1024` (`1.271x` on
A3B, `1.106x` on A10B). The runtime duplicate version remains a production no-go
because of prior residency cost; any next version must replace the source banks
or prove a near-zero-residency ABI.

### Current Read

- The next exact sprint should pivot back to routed FFN/MoE structure. With the
  matrix sidecar active, more attention micro-knobs do not have enough measured
  headroom at `pp4096`.
- The strongest exact hypothesis is still a structural routed-tail proof that
  reduces the combined `grouped_swiglu + grouped_down` bucket, not another local
  tile/threshold variant inside `grouped_swiglu` alone.
- `pp34502` matrix/no-op sweeps need better methodology: a four-variant run sagged
  from `688.85` to `561.45 t/s` baseline while `pmset -g therm` and
  `memory_pressure -Q` reported no warning. Treat true-long spot rows as sparse
  anchors unless repeated/cooled with interleaved baselines.

## 2026-05-23 — Vector B-Tile Loads Lift A3B Matrix True-Long Row

Status: env-only A3B/group-8 matrix sidecar; not defaulted. GPU measurements were
run sequentially; rows are spot checks, not cooled promotion sweeps.

### What Changed

- Matched another `ggml-metal` `mul_mm_f16_f32` detail in the matrix sidecar:
  vector-load the F32 B tile (`q` for `KQ`, softmax probabilities for `KQV`) as
  `float2x4` and cast to `half2x4` in threadgroup memory instead of scalar-loading
  eight floats one by one.
- Kept the existing scalar fallback for ragged/non-aligned score rows.

### Correctness

- A3B two-chunk packed oracle remains green with the vector-load path:
  `pp256 --prefill-chunk 128`, `QWEN_PREFILL_ATTN_MATRIX_G8=1`,
  `QWEN_PREFILL_ATTN_PACKED_G8_ORACLE=1`.
- All oracle rows are finite, `cos=1.0`, and max_abs remains within the existing
  matrix tolerance (`<= 2e-2`).

### Measurements

Current vector-load matrix sidecar spot rows:

| Prompt | Chunk | Tokens/s | Read |
| ---: | ---: | ---: | --- |
| `4096` | `1024` | `1042.04` | flat/slightly up vs prior `1039.09` |
| `8192` | `1024` | `996.51` | up vs prior `956.61` |
| `16384` | `1024` | `863.79`, `903.99` | mixed/noisy around prior `892.45` |
| `34502` | `1024` | `725.67` | up vs prior `658.79` |

True-long chunk-size probe on the same current branch:

| Prompt | Chunk | Tokens/s | Read |
| ---: | ---: | ---: | --- |
| `34502` | `1024` | `725.67` | stale default cap is not obviously optimal |
| `34502` | `2048` | `736.90` | best spot row so far |
| `34502` | `4096` | `718.18` | loses despite larger query batch |

### Current Read

- The old `1024` prompt chunk cap is not principled for true-long matrix attention;
  it was a pragmatic cap from the `pp1024` feedback-loop era.
- Larger chunks trade better KQ/KQV shape and fewer chunk boundaries against score
  scratch (`chunk * n_q_heads * max_pos * 4` bytes) and heavier long-context matrix
  traffic. At `34.5k`, `2048` currently looks better than `1024`, while `4096`
  loses.
- Against the prior `llama.cpp -fa 0` `pp34502` anchor (`897.64 t/s`), the current
  best qwen matrix row is about `0.82x`, up from the earlier `0.66x` default and
  `0.73x` fused-V_T sidecar anchors.
- Next leverage is still KQ/KQV kernel parity and score/KQV traffic. Do not promote
  the matrix path until chunk sizing, max-pos allocation, and half-probability KQV
  correctness are production-shaped.

## 2026-05-23 — Fused V_T Scatter Converts A3B Matrix Attention At Long Context

Status: env-only A3B/group-8 matrix sidecar; not defaulted. GPU measurements were
run sequentially; rows are spot checks, not cooled promotion sweeps.

### What Changed

- Replaced the matrix sidecar's body-time V transpose with a persistent
  per-attention-layer V_T bank using fixed `vt_stride=max_pos`.
- Added fused K/V cache scatter plus V_T sidecar write, so V_T is populated at
  cache-fill time from `v_now_pack` instead of re-reading canonical V in the
  attention body.
- Kept canonical `[pos, kv]` V cache intact; the V_T bank is still sidecar scratch
  behind `QWEN_PREFILL_ATTN_MATRIX_G8=1` and
  `QWEN_PREFILL_ATTN_MATRIX_MAX_POS=<tokens>`.

### Correctness

- Two-chunk oracle is green: A3B `pp256 --prefill-chunk 128` with
  `QWEN_PREFILL_ATTN_PACKED_G8_ORACLE=1` passes both chunks for all attention
  layers.
- Oracle rows remain finite with `cos=1.0`; max_abs stays within the existing
  matrix tolerance (`<= 2e-2`) that accounts for the half-probability KQV path.

### Measurements

Fused V_T scatter sidecar versus same-session default spot rows:

| Prompt | Matrix sidecar | Default anchor | Delta |
| ---: | ---: | ---: | ---: |
| `1024` | `972.29 t/s` | `880.15 t/s` | `+10.5%` |
| `2048` | `953.29 t/s` | `868.46 t/s` | `+9.8%` |
| `4096` | `1039.09 t/s` | `905.81 t/s` | `+14.7%` |
| `8192` | `956.61 t/s` | `829.31 t/s` | `+15.3%` |
| `16384` | `892.45 t/s` | `753.03 t/s` | `+18.5%` |

True-long single row against prior anchors:

- A3B `pp34502` matrix sidecar: `658.79 t/s`.
- Prior same-shape qwen default anchor: `596.88 t/s`.
- Prior same-shape `llama.cpp -fa 0` anchor: `897.64 t/s`.
- Read: the branch recovers real true-long ground (`~1.10x` at `34.5k`) but still
  leaves qwen at about `0.73x` of lcpp on that row.

Phase trace at `pp4096` after fused V_T scatter:

- The body-time `body_matrix_vt` phase disappears for fresh prompts.
- `rope_scatter` remains tiny (`~0.05 ms/layer` in the traced run), so the V_T
  write is cheap when fused with cache fill.
- Remaining matrix body cost scales through `KQ`, softmax, and especially `KQV`:
  near `n_pos=4096`, per layer is roughly `KQ ~3.3 ms`, softmax `~1.0-1.5 ms`,
  `KQV ~4.0-4.7 ms`.

### Current Read

- The lcpp-copyable mechanism was not just “matrix attention”; it was writing V in
  KQV-ready transposed layout at cache-fill time. Body-time transpose, even
  incremental, was the wrong shape.
- This is the first A3B long-context attention branch that materially improves
  `4k/8k/16k+` instead of only `pp512/1024`.
- Do not default yet: the sidecar still needs an allocation policy that does not
  rely on manual `QWEN_PREFILL_ATTN_MATRIX_MAX_POS`, a stricter correctness story
  for the half-probability KQV path, and repeated cooled sweeps.
- Next leverage is lcpp `mul_mm_f16_f32` parity for KQ/KQV and a production V_T
  cache/scratch ABI, not more packed-attention row tiling.

## 2026-05-23 — A3B Matrix-Attention Sidecar Wins Medium, Fails Long

Status: env-only diagnostic branch; not a default candidate. GPU measurements were
run sequentially.

### What Changed

- Corrected the `llama-bench` attention target: for `tools/llama-bench`, `-fa 0`
  means flash attention disabled, not auto. The A3B long target rows we have been
  chasing are therefore `llama.cpp`'s non-flash Metal graph.
- Added a gated A3B/group-8 matrix-attention sidecar behind
  `QWEN_PREFILL_ATTN_MATRIX_G8=1`. Long runs must also set
  `QWEN_PREFILL_ATTN_MATRIX_MAX_POS=<tokens>` because the sidecar allocates
  `[N * n_q_heads, max_pos]` score scratch and V-transpose scratch.
- The sidecar is deliberately close to the high-level non-flash graph shape:
  transpose V, compute KQ, softmax, then KQV.

### `llama.cpp` Flash-Attention Sanity

| Prompt | `llama.cpp -fa 0` | `llama.cpp -fa 1` | Read |
| ---: | ---: | ---: | --- |
| `1024` | `1423.84 t/s` | `1429.92 t/s` | flat |
| `16384` | `1103.80 t/s` | `1097.20 t/s` | flash slightly slower |

Interpretation: copying the `llama.cpp` flash path is not the missing A3B long
lever. The stronger comparison is its non-flash `KQ -> softmax -> KQV` path plus
its cache/layout/kernel implementation details.

### Matrix-Sidecar Measurements

| Prompt | Current default | Matrix sidecar | Delta |
| ---: | ---: | ---: | ---: |
| `128` | `661.32 t/s` | `680.79 t/s` | `+2.9%` |
| `512` | `902.93 t/s` | `936.35 t/s` | `+3.7%` |
| `1024` | `963.65 t/s` | `1003.75 t/s` | `+4.2%` |
| `2048` | `946.60 t/s` | `961.88 t/s` | `+1.6%` |
| `4096` | `922.44 t/s` | `892.00 t/s` | `-3.3%` |
| `4096`, chunk `512` | `858.44 t/s` | `756.87 t/s` | `-11.8%` |
| `8192` | `864.97 t/s` | `771.60 t/s` | `-10.8%` |

Correctness:

- Active `pp128` packed-oracle run is finite and passes with `cos=1.0`; max_abs is
  looser than the default packed path (`~1.4e-2`) because this sidecar currently
  casts softmax probabilities through half for the KQV simdgroup path.
- Active small A3B prefill-vs-single test passes with final-logits cosine
  `0.999985` and all tracked state cosines above `0.9997`.

Negative side probe:

- A3B packed `GROUP_TILE=4` was exact at `pp128` but slower at `pp1024`, `pp4096`,
  and `pp16384`; the code was reverted.

### Current Read

- This sidecar falsifies the easy version of “just make qwen attention look like
  `llama.cpp` non-flash.” The high-level graph shape alone wins only medium
  prompts and crosses over negative by `4k/8k`.
- Keep `QWEN_PREFILL_ATTN_MATRIX_G8=1` as an env-only diagnostic. Do not promote it
  without repeated cooled `pp512/1024` wins, no `pp2048` fade, and explicit long
  disable logic.
- The highest-EV long branch is now a tighter `llama.cpp -fa 0` differential:
  kernel/layout trace, persistent V-transposed cache behavior, KQV layout, and
  score/partial traffic. Do not continue blind packed-kernel knob sweeps without
  that explanation.

## 2026-05-23 — Calibrated A3B True-Long Gap Against `llama.cpp`

Status: same-shape sparse rows, measured sequentially after `v0.109`.

### What Changed

- Replaced the session-memory framing around “monotonic long prefill” with actual
  same-shape rows.
- Verified that `llama.cpp` also declines at true long context after the medium
  prompt peak, but remains much faster across the sparse ladder.
- Re-ran qwen A3B no-op attribution at `16k` and `34.5k` to separate medium-prompt
  FFN gap from true-long attention slope.

### Same-Shape Synthetic Rows

| Prompt | qwen-llm | llama.cpp | qwen / llama |
| ---: | ---: | ---: | ---: |
| `1024` | `955.07 t/s` | `1417.24 t/s` | `0.67x` |
| `4096` | `919.51 t/s` | `1362.34 t/s` | `0.68x` |
| `16384` | `767.64 t/s` | `1112.03 t/s` | `0.69x` |
| `34502` | `596.88 t/s` | `897.64 t/s` | `0.66x` |

### No-Op Attribution

- A3B `pp16384` baseline: `767.64 t/s`.
- A3B `pp16384`, `QWEN_PREFILL_NOOP_ATTN_BODY=1`: `1101.72 t/s`.
- A3B `pp16384`, `QWEN_PREFILL_NOOP_MOE_ROUTED=1`: `913.73 t/s`.
- A3B `pp34502` baseline: `596.88 t/s`.
- A3B `pp34502`, `QWEN_PREFILL_NOOP_ATTN_BODY=1`: `1078.49 t/s`.

### Current Read

- The true-long drop is not primarily real-rollout prompt shape: qwen synthetic
  `34.5k` and real `v02_reva` `34.5k` are close (`596.88` vs `587.60 t/s`).
- `llama.cpp` does not stay monotonically faster forever; it falls from `pp1024`
  to `34.5k`, but from a much higher baseline.
- The same-shape qwen/lcpp ratio is broadly `~0.66-0.69x`, so the gap is not only
  a special long-rollout cliff.
- At true-long shapes, attention-body cost is the main slope lever: no-oping qwen
  attention at `16k` nearly reaches lcpp full prefill (`1101.72` vs
  `1112.03 t/s`).
- Medium prompt work should stay routed-FFN/layout focused; true-long work should
  target packed attention main-pass/context-growth behavior.

## 2026-05-23 — A3B Route+Bucket Fusion Drops To `pp128`

Status: local branch evidence. GPU runs were sequential.

### What Changed

- After dropping A3B route-logits `E8xP32` to `pp128`, route bucket itself became
  visible at A3B `chunk_p=320` (`0.58 ms`, `8.2%` in the live grouped-tail
  profile).
- Lowered fused route+bucket auto activation for A3B-sized hidden states
  (`hidden <= 2048`) from `512` to `128`; larger MoE shapes stay at `512`.

### Measurements

A3B default prompt anchors after lowering route+bucket fusion:

- `pp128`: `651.64 -> 655.48 t/s`.
- `pp256`: `783.48 -> 800.99 t/s`.
- `pp320`: `815.75 -> 830.37 t/s`.

Correctness:

- A3B `pp128` route+bucket fused oracle is exact:
  `cos(topk_w)=1.0`, `cos(shared_gate)=1.0`, `cos(reduced)=1.0`.

### Current Read

- This is a small but clean A3B medium/short-prompt follow-on to the route-logits
  threshold win.
- It does not materially change real long-rollout rows because those already use
  chunk sizes above the previous `512` fusion threshold.
- Do not generalize to A10B without cooled anchors; larger-router route-logits
  thresholding already showed shape-specific regressions.

## 2026-05-23 — A3B Route-Logits E8P32 Drops To `pp128`

Status: local branch evidence. GPU runs were sequential.

### What Changed

- The `pp320` grouped-tail profile exposed a stale threshold: route logits were
  still using the generic mat-mat below `512`, costing A3B `3.64 ms` in a single
  live MoE tail at `chunk_p=320`.
- Lowered the route-logits `E8xP32` auto threshold for the A3B-sized router
  (`hidden <= 2048`) from `512` to `128`.
- Kept larger MoE routers at the old `512` threshold for now: A10B `pp256` forced
  `E8xP32` regressed badly, and A10B `pp320` needs a cooled repeated-anchor pass
  before promotion.

### Measurements

A3B forced/default `E8xP32` route-logits wins:

- `pp128`: baseline `624.12 t/s`, new default `651.64 t/s`.
- `pp256`: baseline `748.85 t/s`, forced `E8xP32` `783.48 t/s`.
- `pp320`: baseline `775.25 t/s`, new default `815.75 t/s`.

Real-rollout sanity on the same branch:

- A3B `current-reva-short-qwen36` preserve replay: `7,986` tokens at
  `855.33 t/s`.
- A3B `v02_reva.json` preserve full rollout: `34,502` tokens at `587.60 t/s`.
- These real-rollout rows are mainly no-regression coverage for the branch: the
  route-threshold change is a short/medium synthetic win, while normal real
  rollout chunks were already above the old `512` router threshold.

Post-threshold A3B `chunk_p=320` grouped-tail profile:

- `route_logits`: `0.27 ms` (`3.8%`), down from the stale-threshold `3.64 ms` row.
- `grouped_swiglu`: `4.06 ms` (`57.2%`).
- `grouped_down`: `1.60 ms` (`22.5%`).
- Route-side is now back below the real routed-FFN wall at this shape.

Correctness:

- A3B `pp128` route-logits `E8xP32` oracle is exact:
  `probs_cos=1.0`, `topk_mismatches=0`, `w_cos=1.0`, `gate_cos=1.0`.
- A10B `pp320` oracle is also exact, but perf promotion is not yet clean.
- A10B default smoke remains green after the zero-fill default-off and route
  threshold cleanup.

Negative / not promoted:

- A10B `pp256`: forced `E8xP32` `219.21 t/s` vs baseline `248.99 t/s`; keep the
  larger-router auto threshold above this regime.
- A10B `pp320`: one forced run looked positive and one no-env run after heavy
  probes looked bad; do not default until a cooled repeated-anchor sweep resolves
  it.

### Current Read

- A3B medium/short MoE prompt still had a cheap route-side threshold win after
  packed attention moved the board.
- The remaining A3B `pp320` gap is now smaller, but routed FFN still dominates the
  no-op ceiling.
- A10B route thresholding must stay conservative; do not generalize the A3B
  threshold by prompt size alone.

## 2026-05-23 — Routed MoE FFN Residual: Hybrid Down-Sum Killed, Fused-Bank Evidence Strengthens

Status: local branch evidence. All GPU measurements below were run sequentially;
one stale pre-rebuild bench spot was discarded.

### What Changed

- Re-centered the residual MoE prompt gap after packed attention on routed FFN:
  routed-noop ceilings dwarf shared-noop ceilings on the current default path.
- Implemented and then removed a narrow hybrid falsifier:
  grouped `SwiGLU` feeding the existing packed `down+weighted_sum` kernel, to test
  whether skipping grouped `out` materialization was worth losing grouped-down
  locality.
- Added `pp320` exact fused-bank grouped-`SwiGLU` profiles to test whether the
  interleaved gate/up expert-bank signal survives on the medium-prompt board.
- Defaulted grouped routed `inner/out` zero-fill to off locally after prior
  coverage/poison oracles proved full slot coverage.

### Measurements

Current no-op ceilings on default packed-attention MoE prompt path:

- A3B `pp320`: base `~765 t/s`, routed-noop `~1136 t/s`, shared-noop `~790 t/s`.
- A10B `pp320`: base `~305 t/s`, routed-noop `~648 t/s`, shared-noop `~309 t/s`.

Hybrid grouped-`SwiGLU` -> packed `down+weighted_sum` falsifier:

- Correctness passed the small A3B prefill-vs-single gate.
- Correctness passed the small A10B smoke gate.
- Rebuilt sequential A3B `pp320` killed the idea:
  - baseline: `775.67 t/s`
  - hybrid: `478.84 t/s`
- Interpretation: removing grouped `out` / weighted-sum passes is not worth giving
  up grouped-down locality. Do not revive this path without a new dataflow premise.

Fused interleaved gate/up expert-bank grouped-`SwiGLU` profiles:

- A3B `chunk_p=320`: `5.61 -> 4.38 ms`, `1.280x`, exact.
- A10B `chunk_p=320`: `8.56 -> 6.22 ms`, `1.376x`, exact.
- Prior context:
  - A3B `512`: `1.078x`; A3B `1024`: `1.023x`
  - A10B `512`: `1.226x`; A10B `1024`: `1.015x`

Runtime duplicate fused-bank proof:

- A3B small correctness passed with fused gate/up banks in the real grouped path.
- Memory cost was severe: `40 * 288 MiB = 11.25 GiB` extra resident for A3B.
- Rebuilt sequential A3B `pp320` only moved `775.25 -> 790.38 t/s` (`~1.02x`).
- Interpretation: the duplicate-bank proof is not a production optimization path;
  if this direction returns, it must be a replacement/offline ABI or a deeper
  fusion that avoids duplicate residency.

### Current Read

- Routed FFN, not attention, is the dominant remaining MoE prompt residual.
- The hybrid falsifier says the next dataflow branch must preserve grouped-down
  locality; naive packed-token down-sum is dead.
- Fused/interleaved gate-up layout is now strong evidence at `pp320`, but the
  taper by `pp1024` means it should be treated as a medium-prompt expert-bank ABI
  candidate, not a universal production shape yet.
- The runtime duplicate fused-bank path is killed as an optimization branch: too
  much resident memory for too little end-to-end conversion.
- Zero-fill default-off is a small cleanup keeper; it is not the board-closing
  branch.

## 2026-05-22 — A3B Packed Threshold Drops Again: `min_pos=128`

Status: local branch only so far. The sub-`256` probe was only worth promoting
for A3B/group-8; A10B/group-16 remains capped at `320`.

### What Changed

- Re-ran the A3B sub-`256` threshold question with exactness-first oracles and a
  cooled repeated-anchor `pp128` sweep.
- Promoted only the A3B/group-8 packed threshold from `256` to `128`.

### Measurements

Correctness:

- A3B `min_pos=128` is green at `pp128` and `pp256` via the per-layer
  packed-vs-old oracle.

Disciplined `pp128` sweep:

- baseline-a: `559.40 t/s`
- `min128`: `592.08 t/s`
- baseline-b: `559.55 t/s`

Warm no-env spot checks after promotion:

- A3B `pp128`: `~584.14 t/s`
- A3B `pp256`: `~689.52 t/s`

### Current Read

- A3B/group-8 still had a real medium-short prompt win left below `256`.
- A10B/group-16 does not yet have the same evidence below `320`, so the family
  split is now:
  - A3B packed from `128`
  - A10B packed from `320`
- Threshold tuning below these points should now stop unless a new scoreboard or
  user-regime need specifically points back at the sub-`128` / sub-`320` zone.

## 2026-05-22 — Family-Specific Sub-512 Packed Thresholds Beat The Uniform 512 Floor

Status: local branch only so far. Packed MoE attention now appears promotable
below `512`, but not with one universal threshold.

### What Changed

- Re-characterized the previously-ambiguous sub-`512` zone with exactness-first
  per-layer packed-vs-old oracles plus cooled repeated-anchor sweeps.
- Proved exactness at the first newly activated prompt sizes:
  - A3B `min_pos=256` green at `pp256` and `pp320`
  - A10B `min_pos=320` green at `pp320`
- Defaulted the activation threshold locally to:
  - A3B / `group=8`: `n_pos >= 256`
  - A10B / `group=16`: `n_pos >= 320`

### Measurements

Disciplined `pp320` sweeps resolved the next threshold split:

- A3B `pp320` cooled anchors:
  - baseline-a: `698.70 t/s`
  - `min256`: `765.90 t/s`
  - baseline-b: `695.39 t/s`
- A10B `pp320` cooled anchors:
  - baseline-a: `109.91 t/s`
  - `min320`: `291.76 t/s`
  - baseline-b: `259.75 t/s`

Post-default warmed no-env prompt anchors:

- A3B:
  - `pp320`: `~774.68 t/s`
  - `pp512`: `~880.51 t/s`
  - `pp1024`: `~952.50 t/s`
  - `34.5k`: `~590.74 t/s`
- A10B:
  - `pp320`: `~275.51 t/s`
  - `pp512`: `~355.18 t/s`
  - `pp1024`: `~417.93 t/s`
  - `19.6k`: one warmed spot `~202.00 t/s` (keep cooled long sweeps as the real
    promotion oracle for this row)

### Current Read

- The packed-attention threshold should be family-specific just like packed
  `NWG`.
- A3B benefits cleanly from activating packed attention at `256`.
- A10B benefits cleanly from activating packed attention at `320`, while `256`
  remains too ambiguous/noisy to bank as a default.
- Threshold tuning should stop here until a real user or scoreboard need forces
  the `pp256` A10B question back onto the board.

## 2026-05-22 — Default Packed Min-Pos 512 Unlocks The Medium-Prompt Board

Status: local branch only so far. Prompt-native packed attention now appears safe
and profitable from `n_pos >= 512` on the proven MoE attention shapes.

### What Changed

- Added `QWEN_PREFILL_ATTN_PACKED_G{8,16}_MIN_POS` overrides and probed the
  activation threshold directly instead of assuming `4096` was the right floor.
- Proved exactness with the existing per-layer packed-vs-old oracle at:
  - `pp512` for both A3B and A10B
  - `pp768` for both A3B and A10B
  - `pp4100` early chunks for both A3B and A10B
- Re-measured warmed `pp512` / `pp1024` plus cooled long-prompt sweeps to decide
  whether lowering the threshold actually converts on the real board.

### Measurements

Warm `pp512` gains with `min_pos=512`:

- A3B: `~788.35 -> ~890.87 t/s`
- A10B: `~309.83 -> ~348.98 t/s`

Warm `pp1024` gains with `min_pos=1024` (which `min_pos=512` also implies):

- A3B: `~806.11 -> ~952.50 t/s`
- A10B: `~317.38 -> ~417.93 t/s`

Cooled long-prompt sweeps also stay positive once the early chunks are packed:

- A3B `34,502 tok`: `~579.66 -> ~592.72 t/s`
- A10B `19,591 tok`: `~235-246 -> ~265.17 t/s`

Using the old family-baseline llama.cpp anchors, the new warmed medium-prompt
position is now roughly:

- A3B `pp512`: `~0.62x`
- A3B `pp1024`: `~0.67x`
- A10B `pp512`: `~0.76x`
- A10B `pp1024`: `~0.96x`

That A10B `pp1024` row is the big regime shift: the packed-attention path is no
longer just a long-context niche; it materially changes the medium-prompt board.

### Current Read

- `min_pos=4096` was leaving a lot of real value on the table.
- `min_pos=512` looks like the right default for the proven MoE packed-attention
  families because it captures the large `pp512`/`pp1024` gains while staying in
  an exactness envelope we actually validated.
- The ambiguous zone is now below `512` (`pp256` / possibly `pp320` for A10B),
  which should stay experimental until separately re-characterized.

## 2026-05-22 — Family-Specific Packed NWG Defaults Beat The Universal Setting

Status: packed long-prefill attention now has a family-specific `NWG` default:

- A3B / group-8 packed prefill keeps `NWG=64`
- A10B / group-16 packed prefill now defaults to `NWG=32`

### What Changed

- Added a more faithful hidden one-layer stack microbench,
  `qwen-bench attn-layer-micro`, that runs the real attention-layer front + body
  + tail (`qkv -> split -> norms -> rope -> KV scatter -> attention -> gate/o`).
- Added a small attach-mode trace helper in `scripts/profile/trace_attach.py` so
  `xctrace` can start after model load instead of wasting the whole window on
  launch/load.
- Re-ran the packed `NWG=32 vs 64` question with cooled end-to-end sweeps and
  repeated baseline anchors instead of one-off spot checks.
- Defaulted packed `NWG` by family in `metal_dflash`:
  - `prefill_attn_packed_g8_nwg() -> 64`
  - `prefill_attn_packed_g16_nwg() -> 32`

### Measurements

The one-layer stack microbench helped, but it still was not a safe promotion
oracle for A10B.

- A3B rows=`4`: the one-layer stack now agreed with end-to-end that `NWG=64`
  beats `32`.
- A10B rows=`4`: the one-layer stack could still make `NWG=32` look attractive,
  even when the prior one-off end-to-end checks were contradictory.

The cooled end-to-end sweeps resolved the conflict.

A10B `19,591`-token cooled sweep (`--no-warmup`, repeated baselines):

- baseline-a (`NWG=64`): `229.68 t/s`
- `NWG=32`: `253.67 t/s`
- baseline-b (`NWG=64`): `230.73 t/s`

A3B `34,502`-token cooled sweep:

- baseline-a (`NWG=64`): `583.35 t/s`
- `NWG=32`: `482.45 t/s`
- baseline-b (`NWG=64`): `583.47 t/s`

So the correct packed default is explicitly family-specific, not universal.

Post-default spot checks:

- A3B synthetic `34,502 tok`: `~580.53 t/s`
- A10B synthetic `19,591 tok`: `~240.94 t/s`
- A10B real same-fixture rollout (`v02_reva`, `25` msgs, strip replay):
  `~224.08 t/s`

Correctness remained green after the A10B default switch:

- `prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke`
- packed per-layer oracle at `pp4100` with `QWEN_PREFILL_ATTN_PACKED_G16_ORACLE=1`

### Current Read

- The earlier “NWG32 regresses A10B” read was a measurement artifact. The cooled
  sweeps overruled the one-off spots.
- Packed `NWG` should be treated as a family/shape parameter, not a global MoE
  knob.
- A10B promotion decisions should keep using cooled end-to-end sweeps with
  repeated anchors; even the improved one-layer stack microbench is still only a
  debugging aid.
- The next high-EV unknown is no longer the coarse packed `NWG` default. It is
  the systems-level reason A10B can disagree with increasingly faithful
  microbenches.

## 2026-05-21 — Packed Main-Pass Knob Attack: QT=4 And NWG=32 Both Fail To Promote

Status: investigation-only follow-up after `v0.103`. No new default path change.

### What Changed

- Added experimental packed-prefill `QT=4` kernel variants for both A3B/group-8
  and A10B/group-16, plus hidden microbench support in `qwen-bench
  attn-prefill-micro --qt {2,4}`.
- Added packed-prefill `NWG` override envs:
  - `QWEN_PREFILL_ATTN_PACKED_G8_NWG`
  - `QWEN_PREFILL_ATTN_PACKED_G16_NWG`
- Warmed the hidden packed-prefill microbench so first-use pipeline compilation
  does not masquerade as kernel time.
- Added a first-pass Metal System Trace comparison for A3B packed prefill so the
  next branch is grounded in end-to-end timeline evidence rather than more body
  microbench optimism.

### Measurements

The obvious “align host row pack with kernel query tile” hypothesis is now
falsified.

Warmed packed-body microbench at `base_pos=32768`:

- A3B rows=`4`:
  - `qt=2`: packed `~0.97 ms`
  - `qt=4`: packed `~1.80 ms`
- A3B rows=`8`:
  - `qt=2`: packed `~1.88 ms`
  - `qt=4`: packed `~1.95 ms`
- A10B rows=`4`:
  - `qt=2`: packed `~2.61 ms`
  - `qt=4`: packed `~6.28 ms`
- A10B rows=`8`:
  - `qt=2`: packed `~3.48 ms`
  - `qt=4`: packed `~9.58 ms`

So `QT=4` is exact but not promising; it loses across both proven MoE shapes.

The first packed `NWG` sweep produced a more subtle trap.

Body-only warmed microbench, rows=`4`, `qt=2`:

- A3B packed body at `base_pos={8192,16384,32768}` favored `NWG=32` over `64`
  every time.
- A10B packed body at the same positions also favored `NWG=32` over `64`.

But that body-only win does **not** convert end-to-end on long prompts.

Long synthetic prompt spot checks:

- A3B `34,502 tok`:
  - packed `NWG=32`: `~432.2 t/s`
  - packed `NWG=64`: `~515.3 t/s`
- A10B `19,591 tok`:
  - packed `NWG=32`: `~194.7 t/s`
  - packed `NWG=64`: `~220.9 t/s`

That is the clearest negative result in this phase: packed-body timing alone can
positively mislead on `NWG`.

A3B `pp4100` Metal System Trace (`NWG=64` vs `32`) did **not** reveal a clean
queue-gap or command-buffer smoking gun:

- both landed around `~625 t/s`
- both used `5` command buffers and `15` encoders in the summarized trace
- `NWG=32` showed somewhat larger long compute gaps in the coarse parser, but the
  short active-shape prompt itself stayed basically flat

### Current Read

- The next packed-attention branch should **not** be another obvious knob retune.
- `QT=4` is already falsified enough to stop touching for now.
- `NWG=32` is a real example of a body-only micro win that fails the real board.
- The remaining high-EV lens is now either:
  - richer end-to-end Metal counters/capture on the packed path, or
  - a more faithful one-layer full-attention-stack microbench that mirrors the
    live dispatch sequence (`qkv/norm/rope/scatter/body/reduce/gate/o-proj`)
- Until one of those exists, packed main-pass tuning should be treated as a
  hypothesis generator, not a promotion gate.

## 2026-05-21 — Default Long-Prefill Packed Attention For A3B/A10B, And Re-Rank The Residual Gap

Status: the prompt-native packed prefill path is now default-on for the proven
long-prefill MoE attention shapes (`group=8` / `group=16`, `head_dim=256`,
`n_pos >= 4096`) with default packed row groups set to `4`.

### What Changed

- Defaulted the packed prefill attention selector in `metal_dflash` for the
  proven long-context MoE shapes instead of keeping both A3B/group-8 and
  A10B/group-16 behind env-only gates.
- Fixed the scratch-allocation bug that appeared once the auto path became live:
  packed-attention partial scratch can no longer stay `[1]` when the auto path
  is eligible.
- Added A10B packed-attention active-shape correctness coverage in
  `crates/qwen-llm/tests/dflash_correctness.rs` and generalized the per-layer
  packed-vs-old oracle plumbing to group-16.
- Added `scripts/profile/prefill_sweep.py`, a cooled sequential sweep harness
  that records per-variant thermal / memory snapshots and repeated baseline
  anchors so long-prompt row-group comparisons are less vulnerable to run-order
  drift.
- Split the packed prefill kernels into main-only and reduce-only entry points
  for attribution, then added `attn_prefill_v4_main_reduce_breakdown_moe_shapes`.

### Measurements

Coarse system signals stayed flat even when benchmark rankings drifted:

- `pmset -g therm`: still reported no thermal/performance warning state
- `memory_pressure -Q`: stayed around `94-95%` free

That is now an explicit negative result: on this box, those coarse OS probes are
too weak to catch the long-prompt run-order drift that can still move A10B by
double-digit percent. Repeated baseline anchors matter more.

Cooled A10B synthetic sweep (`19,591` tok, `--no-warmup`, fresh process per
variant, `15s` cooldown, sequential):

- baseline-a: `189.21 t/s`
- packed rows=`2`: `206.36 t/s`
- packed rows=`4`: `224.54 t/s`
- packed rows=`8`: `198.46 t/s`
- baseline-b: `196.05 t/s`

So the first clean A10B row-group ranking is:

- `rows=4` best
- `rows=2` positive but smaller
- `rows=8` roughly noise / mildly positive

Real same-fixture A10B long replay (`v02_reva`, `25` messages, strip replay)
also converts with the same row-group choice:

- baseline: `195.18 t/s`
- packed rows=`4`: `204.86 t/s`

Residual long synthetic gap versus `llama.cpp` after the packed-attention wins:

- A3B `34,502 tok`: qwen default packed `~507.8 t/s` vs llama `~820.3 t/s`
  (`~0.62x`)
- A10B `19,591 tok`: qwen default packed `~221.7 t/s` vs llama `~324.7 t/s`
  (`~0.68x`)

Packed-attention main/reduce attribution now says the remaining residual is not
primarily the standalone reduce pass:

- A3B rows=`4`, base_pos=`32768`: main `~0.469 ms/call`, reduce `~0.022 ms/call`
- A3B rows=`8`, base_pos=`32768`: main `~1.362 ms/call`, reduce `~0.028 ms/call`
- A10B rows=`4`, base_pos=`32768`: main `~1.985 ms/call`, reduce `~0.026 ms/call`
- A10B rows=`8`, base_pos=`32768`: main `~2.971 ms/call`, reduce `~0.034 ms/call`

This is the key negative result for the next kernel branch: the explicit
reduce-only reread is tiny. The remaining packed-attention wall is dominated by
the main pass, which still includes the partial writes, KV reads, online
softmax, and execution-shape costs.

### Validation

- `cargo test -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke --release -- --nocapture`
- `QWEN_PREFILL_ATTN_PACKED_G16=1 QWEN_PREFILL_ATTN_PACKED_G16_ORACLE=1 cargo test -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_122b_a10b_moe_packed_attn_active_shapes --release -- --ignored --nocapture`
- `cargo test -p qwen-llm attn_prefill_v4_main_reduce_breakdown_moe_shapes --release -- --ignored --nocapture`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- `uv run scripts/profile/prefill_sweep.py ...`
- sequential `qwen-bench pp` long synthetic / real-lane spot checks
- sequential `llama-bench -p <N> -n 0 -r 1 --no-warmup -o json` long synthetic spot checks

### Current Read

- The prompt-native packed path is now a banked production win for the proven
  long-prefill MoE attention envelopes, not just an experiment.
- A10B is no longer blocked on correctness or row-group uncertainty; `rows=4`
  is the keeper default until a new main-pass kernel proves otherwise.
- The next attention-side kernel branch should target the packed **main pass**,
  not the standalone reduce kernel.
- The strongest remaining systems lesson is methodological: long-prompt run-order
  drift is real even when coarse thermal/memory probes look flat, so repeated
  baselines and cooled sequential sweeps need to stay in the standard harness.

## 2026-05-21 — Real Rollout Prompt Lane And A3B Group-8 Long-Prefill Attention Breakthrough

Status: new prompt-benchmarking and attention-diagnostic work is landed as
experimental infrastructure, not as a default path change yet.

### What Changed

- Added real prompt sources to `qwen-bench pp`: inline text, `--file`, and
  canonical `--messages` rendering with model-family replay semantics.
- Added a durable “real rollout prompt lane” rooted in TheCurrent-derived
  message fixtures so prompt characterization no longer depends on memory or
  one-off local scripts.
- Added group-8 subgroup attention variants (`g8_t4`, `g8_t2`) for the v4 A3B
  long-context decode-shaped attention main pass, plus env-controlled selector
  support.
- Added focused correctness and microbench coverage for the new group-8 subgroup
  path, plus per-chunk prompt tracing for long synthetic prefills.

### Measurements

Real-rollout characterization at clean `v0.101` showed the first serious prompt
regime mismatch with synthetic headline rows:

- `current-reva-short-qwen36`: about `7,986` tokens
- `current-mei-medium-qwen36`: about `23,122` tokens
- full `v02_reva.json` preserve replay: about `34,502` tokens

Same-fixture and matched-token ladders now agree that the long-prompt beast is
real and not mostly prompt/template noise.

A3B synthetic long ladder (baseline):

- `7,841 tok`: `~519.7 t/s`
- `15,983 tok`: `~372.8 t/s`
- `26,059 tok`: `~264.1 t/s`
- `34,502 tok`: `~213.9 t/s`

A10B synthetic long ladder (warmed baseline):

- `6,598 tok`: `~232.6 t/s`
- `10,578 tok`: `~245.1 t/s`
- `15,549 tok`: `~242.3 t/s`
- `19,591 tok`: `~231.7 t/s`

Routed MoE no-op deltas across the same ladder are mostly a constant per-token
offset, not the growing slope term. Dense 27B also degrades materially on the
same synthetic token-count ladder, which exonerates “mostly MoE-specific” as the
lead story.

The decisive prompt-phase result is attention:

- A3B synthetic `noop_attn` lifts `~501 -> 1057 t/s` at `7,841 tok` and
  `~209 -> 973 t/s` at `34,502 tok`.
- A3B `noop_gdn` is tiny by comparison (`~501 -> 534`, `~209 -> 215`).
- A10B `noop_attn` is also large, but its baseline slope is much flatter.

Decode-phase snapshots already hinted at this direction: from `4K -> 32K`, A3B
`attn mixer` grows `~2.78 -> ~8.61 ms` while `gdn mixer`, `moe route`, and
`moe ffn` stay nearly flat.

Group-8 subgroup attention was the first real structural crack in the A3B path.

At A3B shape (`group=8`, `n_pos=32768`, `nwg=64`, `C=64`), attention v4 main
pass microbench:

- old group-8 main pass: `~0.660 ms/call`
- `g8_t4`: `~0.430 ms/call`
- `g8_t2`: `~0.416 ms/call`

End-to-end A3B long synthetic prompt throughput improves strongly with
`QWEN_ATTN_V4_G8_TILE=2`:

- `7,841 tok`: `~519.7 -> ~576.7 t/s`
- `15,983 tok`: `~372.8 -> ~471.7 t/s`
- `34,502 tok`: `~213.9 -> ~308.4 t/s`

Real same-fixture A3B endpoint (`v02_reva.json`, `25` msgs, preserve replay):

- baseline: `~208.3 t/s`
- `g8_t4`: `~286.0 t/s`
- `g8_t2`: `~307.6 t/s`

Per-chunk A3B long synthetic (`N=34502`, `P=1024`) shows the slope is reduced,
not erased:

- baseline chunk near `start=32768`: `~8.24 ms/token`
- `g8_t2` same chunk: `~5.17 ms/token`

The remaining long-prompt term is still strongly attention-shaped, and weak
`prefill_chunk` sensitivity after `g8_t2` argues it is not mostly chunk-count
overhead.

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `QWEN_ATTN_V4_G8_TILE=2 cargo test -p qwen-llm attn_v4_group8_subgroup_matches_naive_f16kv --release -- --ignored --nocapture`
- `cargo test -p qwen-llm attn_v4_main_reduce_breakdown_moe_shapes --release -- --ignored --nocapture`
- `QWEN_ATTN_V4_G8_TILE=4 cargo test -p qwen-llm attn_v4_main_reduce_breakdown_moe_shapes --release -- --ignored --nocapture`
- `QWEN_ATTN_V4_G8_TILE=2 cargo test -p qwen-llm attn_v4_nwg_sweep_moe_shapes --release -- --ignored --nocapture`
- `QWEN_ATTN_V4_G8_TILE=2 cargo test -p qwen-llm attn_v4_tile_c_sweep_moe_shapes --release -- --ignored --nocapture`
- Sequential `qwen-bench pp` synthetic and real-rollout ladders on A3B / A10B
- Sequential `qwen-bench pp` MoE-noop and attention-noop ladders

### Current Read

- The exploding long-prompt slope is now strongly *not* primarily routed MoE
  compute. Routed MoE is a meaningful constant tax; attention is the growing
  term.
- The old A3B group-8 attention main-pass shape was a major local long-context
  inefficiency. `g8_t2` removes a large part of it.
- The remaining structural miss is still likely the decode-shaped attention body
  inside prompt prefill chunks, not chunk-count overhead and not primarily MoE.
- The next serious branch should be prompt-native packed prefill attention.
- `g8_t2` should be treated as a prefill-focused guarded win until decode has a
  separate selector / regression matrix.

## 2026-05-20 — Exact MoE Tail Concurrency Converts; Generic Split Sidecar Does Not

Status: two new guarded experimental branches now exist, both off by default:

- `QWEN_PREFILL_MOE_GROUPED_ZERO_FILL=0` skips grouped routed `inner/out`
  zero-fills after new coverage proof.
- `QWEN_PREFILL_MOE_GROUPED_CONCURRENT_TAIL=1` overlaps the live grouped routed
  tail with the live shared FFN tail for `chunk_p >= 512`.

### What Changed

- Added grouped routed slot-coverage + poison-fill oracles and proved the current
  grouped `inner/out` zero-fills are not semantically required on the proven
  `Q4_K/Q4_K/Q5_K` MoE prompt path.
- Added Metal trace labels / signposts plus llama.cpp Metal graph-debug capture
  so the exact MoE prompt differential is anchored in real graph structure
  rather than intuition alone.
- Re-based the “llama-like split routed FFN” experiments against the **live
  grouped production backend**, not the old packed-slot denominator.
- Added a narrower id-aware grouped-Q4 proof that reuses the existing grouped
  `counts/ids` buckets directly for separate gate/up matmuls, then composes with
  the existing `silu_mul` + grouped Q5 down + weighted reduce.
- Added a bounded production-style overlap branch that keeps the live grouped
  routed kernels and live shared FFN kernels unchanged, but places them in a
  concurrent compute encoder before the final combine.

### Measurements

All runs are M4 Max, release `qwen-bench pp`, synthetic prompts, tail skipped.

Grouped zero-fill is real but small:

- 35B A3B `pp512`: `~797.3 -> ~803.7 t/s` (`~+0.8%`)
- 122B A10B `pp512` GPU time: about `1557.8 -> 1549.4 ms` (`~+0.5%` GPU)
- New slot-coverage + poison-fill oracles on A3B / A10B `pp512` are exact:
  `cos=1.0`, `max_abs=0`

The first “split sidecar” intuition was misleading until re-based against the
live grouped backend.

Fair `pp512` routed-tail comparator on the same route buckets:

- 35B A3B:
  - packed tail: `27.17 ms`
  - live grouped tail: `3.60 ms`
  - separate id-aware gate/up + `silu_mul` + grouped down/reduce:
    - gate/up GPU sum: `2.34 + 1.38 = 3.72 ms`
    - full prototype wall (with shell tax): `35.16 ms`
- 122B A10B:
  - packed tail: `48.82 ms`
  - live grouped tail: `9.55 ms`
  - separate id-aware gate/up + `silu_mul` + grouped down/reduce:
    - gate/up GPU sum: `6.17 + 3.48 = 9.65 ms`
    - full prototype wall (with shell tax): `11.45 ms`

This is the crucial corrected read: separate gate/up is only at parity to slight
loss versus the live grouped backend, not a meaningful routed-tail win.

Bounded overlap on the **live** grouped backend is exact and much more promising:

- Block-local MoE tail only, `pp512`:
  - 35B A3B: `serial_gpu 7.77 ms -> concurrent_gpu 4.26 ms`
  - 122B A10B: `serial_gpu 14.82 ms -> concurrent_gpu 9.88 ms`

That large local effect converts in the real prefill path, but only to a
bounded end-to-end win:

- 35B A3B `pp512`: `796.35 -> 816.28 t/s` (`1.025x`)
- 122B A10B `pp512` (warmed): `326.52 -> 335.11 t/s` (`1.026x`)
- 35B A3B `pp1024`: `816.64 -> 821.70 t/s` (`1.006x`)
- 122B A10B `pp1024` (warmed): `329.37 -> 332.90 t/s` (`1.011x`)

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `QWEN_PREFILL_MOE_GROUPED_ZERO_FILL=0 cargo test -p qwen-llm prefill_tokens_matches_single_token_loop_35b_a3b_moe --release -- --ignored --nocapture`
- `cargo test -p qwen-llm metal_35b_a3b_grouped_zero_fill_coverage_oracle_512 --release -- --ignored --nocapture`
- `cargo test -p qwen-llm metal_122b_a10b_grouped_zero_fill_coverage_oracle_512 --release -- --ignored --nocapture`
- `cargo test -p qwen-llm metal_35b_a3b_grouped_swiglu_down_backend_profile_512 --release -- --ignored --nocapture`
- `cargo test -p qwen-llm metal_122b_a10b_grouped_swiglu_down_backend_profile_512 --release -- --ignored --nocapture`
- `cargo test -p qwen-llm metal_35b_a3b_grouped_overlap_falsifier_512 --release -- --ignored --nocapture`
- `cargo test -p qwen-llm metal_122b_a10b_grouped_overlap_falsifier_512 --release -- --ignored --nocapture`
- `QWEN_PREFILL_MOE_GROUPED_CONCURRENT_TAIL=1 cargo test -p qwen-llm prefill_tokens_matches_single_token_loop_35b_a3b_moe --release -- --ignored --nocapture`
- `QWEN_PREFILL_MOE_GROUPED_CONCURRENT_TAIL=1 cargo test -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke --release -- --nocapture`
- `QWEN_PREFILL_MOE_GROUPED_CONCURRENT_TAIL=1 cargo test -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_122b_a10b_moe_chunk128_boundary --release -- --ignored --nocapture`
- Sequential `qwen-bench pp` checks on A3B / A10B at `pp512` and `pp1024`

### Current Read

- Grouped routed zero-fill is now a correctness-covered cleanup lever, not a big
  scoreboard lever.
- The large “generic split sidecar” hope was wrong once compared against the live
  grouped backend. Beating the old packed-slot denominator was not evidence that
  a broad split routed FFN sidecar was the right next branch.
- The narrower id-aware gate/up proof is the decisive read: fusion/register
  pressure is probably **not** the main remaining exact MoE prompt miss on this
  repo shape, because separate grouped gate/up matmuls only reach parity with the
  live grouped path.
- The best near-term exact branch is now the guarded concurrent-tail rollout:
  it is exact on the covered matrix and converts to a real, if bounded,
  low-single-digit prompt win at `pp512`.
- Do not spend another major branch on a large new exact routed sidecar unless a
  smaller `MUL_MAT_ID`-style microproof beats the current grouped projection /
  routed tail directly, not just the obsolete packed denominator.

## 2026-05-19 — Post-v0.100 MoE Preload Plateau And Search-Space Elimination

Status: the current exact grouped MoE prompt path is much stronger than the old
baseline, but the local `grouped_swiglu` micro-optimization family looks
plateaued on this M4 Max / Qwen3.5-3.6 shape. Preserve the evidence so we do
not retread the same dead branches.

### What Changed

- Recorded the first clean full-family scoreboard on committed `v0.100` in
  `docs/bench/2026-05-19-2313-family`.
- Re-ran grouped MoE tail profiles on the current validated default and added a
  real route split (`route_logits`, `route_select`, `route_bucket`) plus route
  weight concentration stats.
- Exhaustively tested the next obvious grouped-MoE prompt branches after
  `v0.100` and kept or killed them from end-to-end evidence rather than local
  kernel intuition.

### Measurements

Clean family scoreboard on committed `v0.100`:

- 35B A3B `pp512`: `788 t/s` vs `1420 t/s` (`~0.55x`)
- 35B A3B `pp1024`: `814 t/s` vs `1410 t/s` (`~0.58x`)
- 122B A10B `pp512`: `322 t/s` vs `450 t/s` (`~0.72x`)
- 122B A10B `pp1024`: `328 t/s` vs `433 t/s` (`~0.76x`)

Rebased current default routed-tail split:

- 122B A10B `pp512`: `grouped_swiglu ~13.47 ms`, `grouped_down ~4.64 ms`,
  route side `~0.65 ms` total
- 122B A10B `pp1024`: `grouped_swiglu ~16.86 ms`, `grouped_down ~5.79 ms`
- 35B A3B `pp512`: `grouped_swiglu ~5.99 ms`
- 35B A3B `pp1024`: `grouped_swiglu ~9.37 ms`

Route-weight concentration on the same default path is diffuse, not hot-top-k:

- A10B `pp512`: `avg_top1 ~0.2157`, `avg_top2 ~0.1628`, `avg_tail ~0.6215`
- A10B `pp1024`: `avg_top1 ~0.2150`, `avg_top2 ~0.1625`, `avg_tail ~0.6224`
- A3B `pp512`: `avg_top1 ~0.2238`, `avg_top2 ~0.1555`, `avg_tail ~0.6207`
- A3B `pp1024`: `avg_top1 ~0.2238`, `avg_top2 ~0.1555`, `avg_tail ~0.6207`

Dead or demoted post-`v0.100` branches:

| branch | result | evidence |
| --- | --- | --- |
| grouped inner `F16` default path | force-only | correctness-safe; A10B `pp512 ~315.5 -> 322.6`, `pp1024 ~324.7 -> 326.6`; A3B `pp512 ~797.6 -> 800.5`, `pp1024 ~815.1 -> 819.6` |
| cold `n8` grouped Q4 | kill | exact/correct after geometry fix, but regressed both A10B and A3B end-to-end |
| hot threshold `32` rollout | kill | correctness-safe, but 5-run medians were flat on A10B and slightly worse on A3B `pp512` |
| hot grouped-down atomic accumulate | kill | correctness-safe, but regressed both guardrails |
| hot persistent/locality grouped-swiglu queues | kill | correctness-safe, severe regressions |
| paired gate/up resident mirror | kill | correctness-safe, catastrophic residency / wall-time blow-up (`A10B pp512 ~1.45 t/s`) |
| hot `32x32` grouped-swiglu | kill | correctness-safe, flat / slightly worse on A10B, only noise-level positive on A3B |
| active hot tile-list with true indirect dispatch | kill | correctness-safe, effectively flat: A10B `~323.1 / 328.1 t/s` vs default `~324.2 / 328.5`; A3B `~798.3 / 818.0` vs `~795.9 / 815.6` |

### Current Read

- The current exact grouped MoE prompt path is **not** the old broken baseline;
  it is the corrected grouped backend plus route/logit and hot-expert wins. The
  clean family sweep proves the MoE prompt gap is materially smaller than the old
  `~0.5x` mental model on A10B, but still the largest scoreboard miss.
- The obvious exact local `grouped_swiglu` variant family is now well sampled and
  mostly exhausted on this hardware / repo shape. More tile / threshold / queue /
  mirror tweaks should be considered guilty until they show a new mechanism, not
  just a new geometry.
- The surviving MoE-first path is no longer “another small grouped kernel tweak.”
  It is either:
  - a diagnostic reset / exact execution-model differential against llama.cpp, or
  - a broader routed-work / model-shape change that does less MoE work rather than
    doing the same work differently.

## 2026-05-19 — Specialize MoE Router Logits Over Expert Rows

Status: default-on for the proven MoE prompt regime when `chunk_p >= 512`, with
`QWEN_PREFILL_MOE_ROUTE_LOGITS_E8P32=0` as the rollback path.

### What Changed

- Added a router-only `F32` `E8xP32` mat-mat kernel that computes 8 expert rows
  per threadgroup tile while reusing each token row's activation loads.
- Kept the kernel narrow:
  - router logits only,
  - `F32` router weights only,
  - requires `n_in % 4 == 0` and `n_out % 8 == 0`,
  - leaves top-k, shared gate, bucketing, and routed FFN math unchanged.
- Added a route oracle that compares generic vs `E8xP32` route logits on A3B and
  A10B at `pp512` and `pp1024`, checking router-prob cosine, exact top-k ids,
  top-k weight cosine, and shared-gate cosine.
- Allowlisted the kernel into the default MoE prompt path only when `chunk_p >= 512`;
  shorter prompts keep the existing route path.

### Measurements

All runs are M4 Max, release `qwen-bench pp`, synthetic prompts, tail skipped,
two reps.

Corrected A10B routed-tail split, `pp512`, before this kernel:

- `route_logits`: `5.95 ms`
- `route_select`: `0.11 ms`
- `route_bucket`: `0.84 ms`
- `grouped_swiglu`: `10.92 ms`
- `grouped_down`: `3.38 ms`

With `QWEN_PREFILL_MOE_ROUTE_LOGITS_E8P32=1`, the same A10B `pp512` routed-tail
profile moves `route_logits` from `5.95 ms` to `0.47 ms`. At A10B `pp1024`,
`route_logits` is `1.03 ms`.

End-to-end prompt impact in the default composed MoE path (`route-fused + hot-th48`
already active at `chunk_p >= 512`):

| Model | Shape | Previous default | With `E8xP32` | Speedup |
| --- | --- | ---: | ---: | ---: |
| 122B A10B Q4_K_XL | `pp512` | `~302.9 t/s` | `~315.5 t/s` | `1.04x` |
| 122B A10B Q4_K_XL | `pp1024` | `~309.3 t/s` | `~324.7 t/s` | `1.05x` |
| 35B A3B Q4_K_M | `pp512` | `~744.9 t/s` | `~797.6 t/s` | `1.07x` |
| 35B A3B Q4_K_M | `pp1024` | `~776.7 t/s` | `~815.1 t/s` | `1.05x` |

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `metal_35b_a3b_moe_route_logits_e8p32_oracle_512`
- `metal_35b_a3b_moe_route_logits_e8p32_oracle_1024`
- `metal_122b_a10b_moe_route_logits_e8p32_oracle_512`
- `metal_122b_a10b_moe_route_logits_e8p32_oracle_1024`
- `prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke`
- `prefill_tokens_matches_single_token_loop_122b_a10b_moe_chunk128_boundary`
- `prefill_tokens_matches_single_token_loop_35b_a3b_moe`
- `qwen-bench pp` A10B/A3B checks at `pp512` and `pp1024`

### Current Read

- The corrected route split was right: bucket construction was never the main
  routed bottleneck. Router logits were a real remaining cost and this kernel
  removes most of it on the proven MoE shapes.
- The current default MoE prompt path is now the corrected grouped backend plus:
  - fused route+bucket,
  - GPU-owned hot-expert grouped Q4 at threshold `48`,
  - `E8xP32` router logits when `chunk_p >= 512`.
- The next frontier is no longer another router kernel. It is whether the routed
  path can hide or reduce more of the remaining `grouped_swiglu` / scheduling
  cost without losing the clean rollout shape we have now.

## 2026-05-19 — Fix Grouped Q4 Sentinel, Validate Fused Route, And Allowlist Hot Routed SwiGLU

Status: default-on for the proven grouped MoE prompt path when `chunk_p >= 512`,
with `QWEN_PREFILL_MOE_ROUTE_BUCKET_FUSED=0` and
`QWEN_PREFILL_MOE_GROUPED_HOT_Q4_N32=0` as rollback flags.

### What Changed

- Fixed a real grouped-Q4 correctness bug in the routed MoE prompt path:
  grouped `Q4_K` SwiGLU used `u32::MAX` as the open-ended `max_count` sentinel,
  while the Metal kernel cast it to signed `int`. That turned the bound into `-1`
  and caused active experts to early-return.
- Re-based all grouped-Q4 conclusions after the fix:
  - A10B smoke and chunk128-boundary prefill-vs-single gates are green again,
  - A3B prefill-vs-single is green again,
  - the fused route+bucket oracle now compares real routed outputs and passes on
    A3B and A10B.
- Split the old `route+bucket` profile bucket into `route_logits`,
  `route_select`, and `route_bucket`. The corrected A10B `pp512` read showed the
  real routed costs are `grouped_swiglu` first and router logits second; bucket
  construction itself is small.
- Kept `n32-all` as a force-only/debug lever after the corrected end-to-end table
  showed local grouped-compute wins that did not reliably convert to prompt t/s.
- Added a GPU-owned hot-expert grouped-Q4 path over the existing per-expert
  `counts/ids` ledger:
  - hot experts use `n32` grouped SwiGLU,
  - cold experts stay on `n16`,
  - no CPU planning, no extra split buffers.
- Added auto allowlist policy for the two MoE prompt wins that do convert:
  - fused route+bucket,
  - hot expert `n32` with default threshold `48`,
  enabled only when `chunk_p >= 512` unless forced by env.

### Measurements

All runs are M4 Max, release `qwen-bench pp`, synthetic prompts, tail skipped,
two reps.

Corrected routed-tail split, A10B `pp512`, fused-route off:

- `route_logits`: `5.95 ms`
- `route_select`: `0.11 ms`
- `route_bucket`: `0.84 ms`
- `grouped_swiglu`: `10.92 ms`
- `grouped_down`: `3.38 ms`

This is the key corrected read: bucket construction was a distraction; routed
expert compute is still the largest MoE prompt bucket, and router logits are the
next meaningful routed cost.

2x2 routed ablation table:

| Model | Shape | Baseline | Route | Hot(th48) | Combo |
| --- | --- | ---: | ---: | ---: | ---: |
| 122B A10B Q4_K_XL | `pp256` | `265.71 t/s` | `260.43 t/s` | `259.11 t/s` | `266.93 t/s` |
| 122B A10B Q4_K_XL | `pp512` | `297.04 t/s` | `293.77 t/s` | `299.66 t/s` | `302.87 t/s` |
| 122B A10B Q4_K_XL | `pp1024` | `299.60 t/s` | `302.90 t/s` | `306.41 t/s` | `309.28 t/s` |
| 35B A3B Q4_K_M | `pp256` | `676.26 t/s` | `688.62 t/s` | `677.07 t/s` | `683.45 t/s` |
| 35B A3B Q4_K_M | `pp512` | `736.27 t/s` | `741.32 t/s` | `737.33 t/s` | `744.85 t/s` |
| 35B A3B Q4_K_M | `pp1024` | `754.68 t/s` | `768.15 t/s` | `760.35 t/s` | `776.74 t/s` |

Default policy after rebuild (`chunk_p >= 512` gets combo automatically; shorter
prompts stay on the corrected grouped baseline):

- 122B A10B: `pp256 ~264 t/s`, `pp512 ~303 t/s`, `pp1024 ~308 t/s`
- 35B A3B: `pp256 ~681 t/s`, `pp512 ~743 t/s`, `pp1024 ~770 t/s`

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `metal_35b_a3b_moe_route_bucket_fused_oracle_512`
- `metal_122b_a10b_moe_route_bucket_fused_oracle_512`
- `prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke`
- `prefill_tokens_matches_single_token_loop_122b_a10b_moe_chunk128_boundary`
- `prefill_tokens_matches_single_token_loop_35b_a3b_moe`
- corrected grouped-Q4 `n32` proofs on A3B/A10B for `pp512` and `pp1024`
- corrected hot-expert threshold scans on A3B and A10B at `pp512`
- 2x2 `qwen-bench pp` ablation table on A3B/A10B for `pp256` / `pp512` /
  `pp1024`

### Current Read

- The grouped-Q4 sentinel bug invalidated a large chunk of the earlier MoE prompt
  story. After fixing it, the routed-compute path is stronger than it looked and
  the fused route+bucket branch is genuinely correct.
- `n32-all` is not a ship candidate: it improves grouped compute locally but does
  not convert cleanly to end-to-end prompt throughput.
- The first post-fix routed-compute attack that does convert is GPU-owned
  hot-expert specialization at threshold `48`, especially when composed with the
  fused route+bucket cleanup at `pp512+`.
- The next MoE prompt frontier is no longer generic grouped-Q4 tuning. It is the
  upstream routed path around router logits / scheduling, with hot-expert compute
  retained as the best current routed FFN shape.

## 2026-05-18 — Concurrent GDN MoE Decode Front Projections

Status: default-on for MoE decode on this repo, with
`QWEN_DECODE_MOE_CONCURRENT_GDN=0` as the rollback path while broader device
rollout evidence is still expanding.

### What Changed

- Ported the existing dense concurrent-GDN front-projection split into MoE
  decode:
  - serial token embedding + pre-GDN RMSNorm,
  - `begin_concurrent` for GDN front projections,
  - serial GDN tail, residual, postnorm, MoE route, and MoE FFN.
- Added bounded decode A/B harnesses for the new path:
  - `qwen-bench tg --concurrent-gdn-proj` for apples-to-apples decode,
  - `qwen-bench decode-window --concurrent-gdn-proj` for fixed-context traces.
- Added serial-vs-concurrent decode correctness gates on A3B and A10B.
- Promoted the path into production decode behind a rollback flag instead of a
  bench-only switch.

### Measurements

All runs are M4 Max, release `qwen-bench`, random-token `tg` or fixed-context
`decode-window`, three reps unless noted.

Apples-to-apples decode (`tg`):

| Model | Shape | Serial | Concurrent GDN | Speedup |
| --- | --- | ---: | ---: | ---: |
| 35B A3B Q4_K_M | `tg32` | `73.97 t/s` | `77.76 t/s` | `1.05x` |
| 35B A3B Q4_K_M | `tg128` | `73.60 t/s` | `77.95 t/s` | `1.06x` |
| 122B A10B Q4_K_XL | `tg32` | `32.39 t/s` | `35.06 t/s` | `1.08x` |
| 122B A10B Q4_K_XL | `tg128` | `32.52 t/s` | `34.90 t/s` | `1.07x` |

Fixed-context decode-window (`window=32`):

- 35B A3B `ctx=128`: `73.2 -> 75.6 t/s`
- 35B A3B `ctx=4096`: `66.6 -> 71.5 t/s`
- 122B A10B `ctx=128`: `32.0 -> 34.3 t/s`
- 122B A10B `ctx=4096`: `31.4 -> 34.0 t/s`

The win is not just wall-time accounting: GPU time moves in the right direction
on every measured row, and A10B now clears the prior local `llama-bench tg128`
anchor (`~33.8 t/s`) with room.

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `metal_single_token_concurrent_gdn_moe_matches_serial_a3b`
- `metal_single_token_concurrent_gdn_moe_matches_serial_a10b_smoke`
- `qwen-bench tg` A/B sweep on A3B and A10B for `tg32` / `tg128`
- `qwen-bench decode-window` A/B on A3B and A10B at `ctx=128` and `ctx=4096`

### Current Read

- The earlier pipelined token-submission work was real but small. The meaningful
  MoE decode frontier was GDN mixer structure, not host overlap.
- This is now the material MoE decode win surface: A3B and A10B both move by
  roughly five to eight percent, and A10B crosses the old external decode
  parity anchor.
- The rollback path should stay live until more devices and a fuller A10B decode
  chain matrix are logged, but the path is strong enough to be the repo default.

## 2026-05-18 — A10B `pp128` Cold Variance Is Expert-Bank First-Touch

Status: benchmark methodology note, not a new steady-state runtime frontier.

### What Changed

- Added a bench-only GPU touch pass over MoE expert-bank weights:
  `QWEN_PP_WARM_MOE_BANKS=1`.
- Added a bench-only `MTLResidencySet` experiment over the existing copied MoE
  expert-bank buffers: `QWEN_PP_RESIDENCY_SET=1`.

### Measurements

Fresh-process A10B grouped `pp128`:

- baseline cold seeds: about `138-150 t/s`, GPU only `~84-85%` of wall.
- touch-warm expert banks: about `234.1 t/s`, GPU `~99.2%` of wall.
- residency-set expert banks: about `232.8-236.0 t/s`, GPU `~99.2%` of wall.
- steady-state `pp320/pp512` stay effectively flat with or without the
  residency-set path.

### Current Read

- The ugly A10B `pp128` cold variance is now explained as first-touch expert-bank
  residency/page behavior, not as a steady-state MoE math/kernel miss.
- Keep the touch/residency-set knobs as benchmark methodology tools for cold
  `pp128` fidelity. Do not treat them as the next production performance queue
  unless load latency or peak-memory duplication becomes a measured product pain.

## 2026-05-17 — GPU-Owned Grouped MoE Prompt Backend

Status: endorsed for production inside the proven `Q4_K/Q4_K/Q5_K` MoE packed
prefill envelope, with `QWEN_PREFILL_MOE_GROUPED=0` as the kill switch.

### What Changed

- Added a fully GPU-owned grouped routed backend for MoE packed prefill:
  - GPU route compaction into per-expert slot counts and slot id lists,
  - grouped `Q4_K` gate/up + fused `silu(gate) * up` kernel writing slot-major
    routed inner activations,
  - grouped `Q5_K` down kernel writing slot-major routed outputs,
  - packed weighted sum reducing slot-major routed outputs back to token-major
    mixer rows.
- Kept the previous packed token-major path as the fallback outside the proven
  envelope and as the opt-out path when `QWEN_PREFILL_MOE_GROUPED=0`.
- Added exact oracles for the grouped routed subkernels:
  - grouped `Q5_K` down vs known-good `mat_mat_q5_k`,
  - grouped `Q4_K` routed SwiGLU vs the existing packed `moe_inner_pack`.
- Added one awkward multi-chunk boundary correctness test for A10B (`T=129`,
  `P=128`) because the new default path is most at risk where chunk boundaries
  and grouped ids interact.

### Measurements

All runs are M4 Max, release `qwen-bench pp`, synthetic prompts, sequential.

Default chunking (`chunk=128`) at pp320:

| Model | Previous default | Grouped default | Speedup |
| --- | ---: | ---: | ---: |
| 35B A3B Q4_K_M | `~399.6 t/s` | `~556.4 t/s` | `1.39x` |
| 122B A10B Q4_K_XL | `~149.4 t/s` | `~218.1 t/s` | `1.46x` |

One-chunk (`chunk=320`) at pp320:

| Model | Previous chunk320 | Grouped chunk320 | Speedup |
| --- | ---: | ---: | ---: |
| 35B A3B Q4_K_M | `~399.6 t/s` | `~710.4 t/s` | `1.78x` |
| 122B A10B Q4_K_XL | `~149.4 t/s` | `~278.7 t/s` | `1.87x` |

Dense guardrails stayed flat on the same build:

- 9B dense pp320: `712.18 +/- 0.43 t/s`
- 27B dense pp320: `211.91 +/- 0.18 t/s`

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke`
- `prefill_tokens_matches_single_token_loop_35b_a3b_moe`
- `prefill_tokens_moe_hidden_capture_matches_p1_oracle_35b_a3b`
- `prefill_tokens_matches_single_token_loop_122b_a10b_moe_chunk128_boundary`
  (`T=129`, `P=128`) — passed with final-logit cos `0.999957`, GDN/KV minima
  `>= 0.999035`.

### Current Read

- The old generic grouped-expert objection no longer applies to this path. The
  winning version is GPU-owned, exact, and end-to-end positive on both MoE
  guardrails.
- Routed MoE prompt work is no longer the dominant family gap it was at the start
  of this investigation. The MoE prompt path now has a production-resolution
  backend for the proven quant envelope.
- The next performance queue should stop treating MoE prompt prefill as the main
  unresolved frontier and shift back toward the remaining dense prompt gap and the
  last decode parity edge cases.
- One rollout caveat remains: A10B `pp128` on the grouped backend is materially
  noisier than `pp320/pp512`. Seed sweeps still settle into the expected
  `~235 t/s` band, but some first runs are much slower (`~24-158 t/s`) even after
  broad synthetic warmup. Treat that as a cold-route/cold-residency risk until a
  better causal read is logged.

## 2026-05-16 — MoE Packed Shared Expert + Rowwise Residual

Status: second major MoE prompt win after packed routed experts. This batches the
shared-expert branch that became the largest remaining MoE tail bucket.

### What Changed

- Added packed shared-expert scratch in `MetalDFlashLayerMajorScratch`:
  `[P, F_shared]` gate/up/inner and `[P, H]` shared output.
- Added `kernel_axpy_rowwise_f32` plus Rust wrapper so one kernel can apply
  `mixer_out[token, :] += shared_gate[token] * shared_out[token, :]` across the
  prompt chunk.
- In packed MoE prefill, shared gate/up/down now run via `encode_mat_mat_dispatch`
  over `h_pack`, `silu_mul`, rowwise AXPY into routed `mixer_out_pack`, and one
  packed residual add into `x_pack`.
- Added `QWEN_PREFILL_MOE_PACKED_SHARED=0` as a kill switch.

### Measurements

All runs are M4 Max, release `qwen-bench pp`, synthetic prompts, one prompt chunk
unless noted, tail skipped, sequential.

35B A3B Q4_K_M:

| Prompt tokens | Packed routed + shared | Shared disabled | Routed disabled baseline | Notes |
| ---: | ---: | ---: | ---: | --- |
| 64 | `360.39 t/s` | — | `188.52 t/s` | prompt-length sweep |
| 128 | `389.68 t/s` | — | `194.46 t/s` | prompt-length sweep |
| 320 | `399.77 t/s` | `262.09 t/s` | `196.71 t/s` | `~2.03x` over routed-disabled baseline |
| 512 | `396.10 t/s` | — | `196.20 t/s` | prompt-length sweep |
| 1024 | `390.14 t/s` | — | `194.33 t/s` | prompt-length sweep |

122B A10B Q4_K_XL:

| Prompt tokens | Packed routed + shared | Shared disabled | Routed disabled baseline | Notes |
| ---: | ---: | ---: | ---: | --- |
| 128 | `143.96 t/s` | — | `83.95 t/s` | prompt-length sweep |
| 320 | `148.92 t/s` | `107.63 t/s` | `84.81 t/s` | `~1.76x` over routed-disabled baseline |
| 512 | `149.40 t/s` | — | `84.68 t/s` | prompt-length sweep |

Default MoE chunking (`chunk=128`) remains strong at pp320:

- 35B A3B: `375.91 +/- 0.43 t/s`
- 122B A10B: `150.37 +/- 0.02 t/s`

Dense guardrails after adding shared scratch and rowwise AXPY stayed flat:

- 9B dense pp320: `711.60 +/- 0.81 t/s`
- 27B dense pp320: `212.08 +/- 0.15 t/s`

### Validation

- `cargo fmt --all`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- A3B default packed path:
  - `prefill_tokens_matches_single_token_loop_35b_a3b_moe` passed.
  - `prefill_tokens_moe_hidden_capture_matches_p1_oracle_35b_a3b` passed.
- A10B default packed path:
  - `prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke` passed.
- A3B separate-process kill-switch matrix passed:
  - `QWEN_PREFILL_MOE_PACKED_SHARED=0`
  - `QWEN_PREFILL_MOE_PACKED_ROUTE=0`
  - `QWEN_PREFILL_MOE_PACKED_DOWN_SUM=0`

### Current Read

- MoE prompt prefill has now moved in two steps after the pp harness exposed the
  gap: Q8 mixer packing, then packed routed + shared expert tails.
- The A3B pp320 stack moved roughly `~96 -> ~197 -> ~261 -> ~400 t/s`.
- The A10B pp320 stack moved roughly `~37.6 -> ~85 -> ~107 -> ~149 t/s`.
- Remaining MoE gap to llama-bench is still large, but the live bottleneck is no
  longer obvious token-loop expert dispatch. The next attack should start with a
  fresh phase profile and prompt-length sweep rather than assuming another MoE
  FFN rewrite is the highest-EV move.

## 2026-05-16 — MoE Packed-Routed Branch Activated + Q4_K Layout Fix

Status: major MoE prompt win reached after correcting the checkpointed branch.
The previous "did not clear gate" result was a false negative: production was
gated on shared-expert dtypes, so the packed routed path was not actually active
in pp runs.

### What Changed

- Fixed the production gate for `QWEN_PREFILL_MOE_PACKED_ROUTED`: it now checks
  routed expert dtypes (`moe.gate_exps`, `moe.up_exps`, `moe.down_exps`) instead
  of shared-expert `ffn_gate/up/down` dtypes.
- Fixed `kernel_moe_swiglu_q4_K_f32_packed_slots`: the packed Q4_K routed
  gate/up kernel now mirrors the single-token Q4_K byte layout exactly, with only
  the token offset added. The earlier packed kernel used the wrong Q4_K layout and
  failed correctness once the branch was truly active.
- Added subpath kill switches for diagnosis:
  - `QWEN_PREFILL_MOE_PACKED_ROUTE=0`
  - `QWEN_PREFILL_MOE_PACKED_DOWN_SUM=0`
- Added an ignored MoE tail A/B profiler that compares old token-loop routed FFN
  against packed route + packed routed gate/up/down at multiple `P` values and
  reports both split-stage and one-command-buffer timings.
- Added an A10B single-token-loop vs packed-prefill smoke correctness test.

### Prompt-Length Sweep

All runs are M4 Max, release `qwen-bench pp`, synthetic prompts, one prompt chunk
(`--prefill-chunk == -p`), tail skipped, sequential. `fallback` means
`QWEN_PREFILL_MOE_PACKED_ROUTED=0`.

35B A3B Q4_K_M:

| Prompt tokens | Packed routed | Fallback | Speedup |
| ---: | ---: | ---: | ---: |
| 64 | `249.01 t/s` | `188.52 t/s` | `1.32x` |
| 128 | `259.18 t/s` | `194.46 t/s` | `1.33x` |
| 320 | `261.20 t/s` | `196.71 t/s` | `1.33x` |
| 512 | `259.88 t/s` | `196.20 t/s` | `1.32x` |
| 1024 | `256.19 t/s` | `194.33 t/s` | `1.32x` |

122B A10B Q4_K_XL:

| Prompt tokens | Packed routed | Fallback | Speedup |
| ---: | ---: | ---: | ---: |
| 128 | `104.71 t/s` | `83.95 t/s` | `1.25x` |
| 320 | `106.73 t/s` | `84.81 t/s` | `1.26x` |
| 512 | `107.05 t/s` | `84.68 t/s` | `1.26x` |

Default MoE chunking (`chunk=128`) remains above the gate at pp320:

- 35B A3B: `254.43 +/- 0.64 t/s`
- 122B A10B: `107.18 +/- 0.05 t/s`

Dense guardrails stayed flat when rerun sequentially:

- 9B dense pp320: `711.75 +/- 0.15 t/s`
- 27B dense pp320: `212.00 +/- 0.02 t/s`

### Validation

- `cargo fmt --all`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_35b_a3b_moe -- --nocapture`
  - final logits cos `0.999985`; GDN/KV minima `>=0.999736`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_moe_hidden_capture_matches_p1_oracle_35b_a3b -- --nocapture`
  - final logits cos `1.000000`; hidden cos_min `1.000000`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke -- --nocapture`
  - final logits/GDN/KV cos all `1.000000`

### Attribution

The fixed A3B one-command-buffer profiler shows the packed path wins across the
range and scales with `P`:

| P | Old tail | New tail | Speedup |
| ---: | ---: | ---: | ---: |
| 8 | `2.43 ms` | `2.00 ms` | `1.21x` |
| 16 | `3.59 ms` | `2.54 ms` | `1.41x` |
| 64 | `7.98 ms` | `5.15 ms` | `1.55x` |
| 128 | `15.86 ms` | `10.15 ms` | `1.56x` |
| 320 | `40.90 ms` | `25.71 ms` | `1.59x` |

At `P=320`, the new split-stage profile is:

- packed route/top-k/shared gate: `0.81 ms`
- packed Q4_K routed SwiGLU: `4.84 ms`
- packed Q5_K down+sum: `8.24 ms`
- shared expert + residual + copies: `15.56 ms`

### Current Read

- The packed routed branch is now a real keeper, not a neutral probe.
- `pp320` remains a useful llama-bench anchor, but the prompt-length sweep shows
  the win is not a 320-token artifact.
- The next MoE bottleneck is shared expert / residual / copy, not route/top-k or
  routed gate/up/down. Prior shared batching was negative, so the next attack
  needs a narrower stage A/B rather than reintroducing generic batching.
- Remaining risks to keep in view: packed route top-k near-tie stability, more
  A10B/odd-length correctness coverage, and avoiding future divergence between
  the single-token and packed Q4_K dequant layouts.

## 2026-05-16 — Checkpoint: MoE Packed-Routed Probe Did Not Clear Gate

Status: checkpointing a mixed worktree. The Q8 mixer fix remains a real keeper;
the newer packed-routed MoE branch is correctness-positive on A3B but
performance-neutral and should be treated as experimental until it is either
default-off or removed.

### What Changed Since The Q8 Mixer Entry

- Added an experimental token-major packed routed MoE prefill path behind
  `QWEN_PREFILL_MOE_PACKED_ROUTED`:
  - packed router logits via mat-mat over `[P, H] -> [P, E]` where eligible,
  - packed top-k/shared-gate selection into `moe_topk_idx_pack`,
    `moe_topk_weight_pack`, and `moe_shared_gate_pack`,
  - existing packed Q4_K routed gate/up SwiGLU over `[P, topk, F]`,
  - new packed Q5_K down + weighted-sum kernel writing `[P, H]`,
  - shared expert/residual still falls back to the existing per-token path.
- Added `moe_router_probs_pack` scratch and Metal/Rust wrappers for the packed
  top-k/shared-gate and packed Q5 down+sum kernels.
- Kept the kill switch because this path has not met the perf gate.

### Clean-Room Process Check

Before rerunning the latest pp sweep, checked active processes with `ps` + `rg`;
only the probe itself matched. No other repo benchmark/build process was alive.

### Latest Measurements

All runs are M4 Max, release `qwen-bench pp`, synthetic `pp320`, chunk `320`,
tail skipped, sequential.

| Model | Latest packed-routed worktree | Q8 mixer baseline | Gate | Result |
| --- | ---: | ---: | ---: | --- |
| 35B A3B Q4_K_M | `197.41 +/- 0.12 t/s` | `~197 t/s` | `>=208 t/s` | no material movement |
| 122B A10B Q4_K_XL | `85.16 +/- 0.75 t/s` | `~85.1 t/s` | `>=89 t/s` | no material movement |

Earlier A/Bs inside the same attack showed the same shape:

- Packed Q5 down+sum without packed route: A3B `196.91 +/- 0.24 t/s`, A10B
  `84.97 +/- 0.91 t/s`.
- A3B fallback with `QWEN_PREFILL_MOE_PACKED_ROUTED=0`: `197.04 +/- 0.40 t/s`.

### Validation State

Passed after the packed route/down changes:

- `cargo fmt --all`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_35b_a3b_moe -- --nocapture`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_moe_hidden_capture_matches_p1_oracle_35b_a3b -- --nocapture`

Not yet re-run after the final packed-route variant:

- A10B MoE correctness gate.
- Dense 9B/27B pp guardrails.
- Full stop-token / EOS test suite from the commingled worktree changes.

### Interpretation

- The Q8 mixer fix remains the validated MoE prompt win: A3B `~96 -> ~197 t/s`,
  A10B `~37.6 -> ~85.1 t/s`.
- The packed-routed branch does not earn production status. It likely removes too
  little of the live surface and may trade slot-level parallelism for longer
  per-row threadgroup lifetime in the Q5 down+sum kernel.
- The next MoE attack should not stack more generic packing onto this branch. If
  this code is separated later, either default it off or delete it unless a stage
  profile shows a clear local win and end-to-end pp320 clears the gate.
- Future MoE work should start from measured phase A/B at `P={8,16,64,128,320}`
  and only escalate to a custom persistent routed kernel if expert reuse/locality
  evidence supports it.

## 2026-05-16 — Phase-Matched PP Harness + MoE Q8 Packed Mixer

Status: major MoE prompt win reached, not yet committed in git.

### What Changed

- Added `qwen-bench pp` as a prompt-only frontier harness aligned with
  `llama-bench pp<N>` semantics: synthetic token ids, explicit repetitions,
  optional real prompt text, no decode loop, and optional tail skip so final
  norm / `lm_head` / logits readback are not charged to pure prompt throughput.
- Added packed-prefill lowering summaries to the pp harness so dense/MoE runs
  report whether GDN, attention, dense FFN, and MoE token-loop paths are active.
- Enabled `Q8_0` packed mat-mat eligibility for GDN and attention projections in
  prompt prefill and DFlash packed verify. `encode_mat_mat_dispatch` already had
  a `Q8_0` backend; MoE Q8 mixer weights were simply falling through to the
  decode-shaped per-token path.
- Extended `QWEN_PREFILL_NOOP_FFN=1` to MoE prompt prefill so mixer work can be
  isolated from the remaining routed/shared expert token loop.
- Expanded the Q8 mat-mat correctness gate to cover `N={1,16,32,64,128}`.

### Fresh Prompt-Only Baselines

All runs are M4 Max, release `qwen-bench pp`, synthetic `pp320`, sequential.

| Model | qwen pp320 | llama-bench pp320 | qwen / llama | Notes |
| --- | ---: | ---: | ---: | --- |
| 9B dense Q4_K_M | `~710.0 t/s` | `~824.0 t/s` | `~86%` | dense lowering already fully packed |
| 27B dense Q4_K_M | `~211.9 t/s` | `~240.9 t/s` | `~88%` | dense unchanged by Q8 eligibility |
| 35B A3B Q4_K_M | `~193.7-197.7 t/s` | `~1222.4 t/s` | `~16%` | `Q8_0` mixer packing landed |
| 122B A10B Q4_K_XL | `~85.1 t/s` | `~393.3 t/s` | `~22%` | `Q8_0` mixer packing landed |

### Measured Impact

- 35B A3B pp320 moved from `~96 t/s` to `~194-198 t/s` after lowering flipped
  from `gdn_batched=0/30 attn_batched=0/10` to `30/30` and `10/10`.
- 122B A10B pp320 moved from `~37.6 t/s` to `~85.1 t/s` after lowering flipped
  from `gdn_batched=0/36 attn_batched=0/12` to `36/36` and `12/12`.
- Dense guardrails stayed flat within noise:
  - 9B dense pp320: `~710.0 t/s`
  - 27B dense pp320: `~211.9 t/s`
- MoE no-FFN probes now show the remaining gap is dominated by the token-loop
  routed/shared expert path, not the mixer front end:
  - A3B `QWEN_PREFILL_NOOP_FFN=1`: `~1722 t/s`
  - A10B `QWEN_PREFILL_NOOP_FFN=1`: `~662 t/s`

### Validation

- `cargo fmt --all`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- `cargo test --release -p qwen-llm mat_mat_q8_0_matches_cpu_and_mat_vec -- --nocapture`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_35b_a3b_moe -- --nocapture`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_moe_hidden_capture_matches_p1_oracle_35b_a3b -- --nocapture`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_27b -- --nocapture`

### Current Read

- The largest MoE prefill hypothesis was real: Q8 mixer projections were not
  using the packed mat-mat path.
- The next MoE gap is now clearly the per-token MoE FFN path. Existing packed MoE
  tail profiling at `P=8` splits one block roughly into route/copy `~14-20%`,
  routed FFN `~47-58%`, and shared/residual/copy `~28-32%`.
- Dense remains a separate prompt-quality problem: the new pp harness confirms a
  stable `~12-14%` prompt-only gap on 9B/27B even with fully packed dense lowering.

## 2026-05-16 — llama.cpp BLAS Hot-Path Audit

Status: investigation-only, no code changes.

### Hypothesis

`llama-bench` reports `backend = MTL,BLAS` for our `pp320` baseline. Before
chasing more prompt structure, we wanted to know exactly what BLAS is doing
on the hot path on this Apple build, so the `pp320` scoreboard target isn't
quietly biased by a fast CPU sgemm path that we don't have, and so that we
don't over-attribute the gap to GPU work.

### What "BLAS" actually means in this build

- `~/code/llama.cpp/build/bin/libggml-blas.0.12.0.dylib` is built with
  `GGML_BLAS_USE_ACCELERATE`. `otool -L` confirms it links
  `/System/Library/Frameworks/Accelerate.framework`. The backend's
  `get_description` returns `"Accelerate"` but its `get_name` returns
  `"BLAS"`, which is what llama-bench prints in the `backend` column.
  See `ggml/src/ggml-blas/ggml-blas.cpp:328-340` and the device-name
  function at `ggml/src/ggml-blas/ggml-blas.cpp:213-217`.
- The BLAS backend is registered as a `GGML_BACKEND_DEVICE_TYPE_ACCEL`
  device (`ggml-blas.cpp:353`). `llama_context::init` adds every ACCEL
  device to the backend list right after the GPU devices and before the
  CPU backend (`src/llama-context.cpp:250-260`). `llama-bench`'s
  `test::get_backend()` then joins every non-CPU registered backend into
  the printed string (`tools/llama-bench/llama-bench.cpp:1481-1500`), so
  `MTL,BLAS` means "MTL plus BLAS were both registered", not "the
  scheduler is splitting work between them".

### What the BLAS backend will compute

`ggml_backend_blas_graph_compute` only handles two ops
(`ggml-blas.cpp:235-253`):

- `GGML_OP_MUL_MAT`
- `GGML_OP_OUT_PROD`

Plus the no-op view family (`NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE`). The
backend's `supports_op` has hard guards
(`ggml-blas.cpp:404-432`):

- both srcs contiguous
- `src1->type == F32`
- `ne0 >= 32 && ne1 >= 32 && ne10 >= 32` (so vector-like shapes never
  reach BLAS)
- `src0` must be F32 or have a `to_float` converter

If `src0` is quantized, the backend dequantizes it to F32 in
`work_data` (parallelized via OpenMP, `ggml-blas.cpp:67-116`) and then
calls `cblas_sgemm` with `m=ne1, n=ne01, k=ne10` for every `(i12,i13)`
slice (`ggml-blas.cpp:128-147`). So in principle BLAS can serve any
non-batched quant mat-mat with a long enough N-dimension, after a full
F32 dequant.

### Where the scheduler actually sends ops

In ggml-backend's sched the priority order is the backend list order
(`ggml-backend.cpp:836-842`). With both Metal and BLAS present, an op
is assigned to whichever backend currently holds its weight buffer
(`ggml-backend.cpp:908-929`). The only chance for BLAS to steal an op
from Metal is the `offload_op` upgrade path, but Metal's own
`offload_op` returns true for `MUL_MAT/MUL_MAT_ID` with batch >= 32
(`ggml-metal.cpp:746-763`, default `op_offload_min_batch_size = 32`,
`ggml-metal-device.m:798`), so as long as the weights are on `MTL0`,
Metal wins ties.

The crucial constraint: Metal's buffer types report `is_host() = false`
for the shared, private, and mapped variants
(`ggml-metal.cpp:275-279`, `351-355`, `427-431`). The sched-side
upgrade only fires when the source buffer is on the CPU and is host
memory (`ggml-backend.cpp:919`: `ggml_backend_buffer_is_host(src->buffer)`).
Weights mapped through Metal's mapped buffer type therefore can't be
hijacked by BLAS even though sgemm could in principle run on them.

### Empirical confirmation on the 27B `pp320` baseline

Ran `GGML_SCHED_DEBUG=2 llama-bench -m Qwen3.6-27B-Q4_K_M.gguf -n 0 -p 320`
and tallied the per-node backend assignments printed by the sched
debug dump (stderr, 15420 lines covering all reservation graphs plus
the live run):

```text
[ MTL0 ]   37980
[ NULL ]    1656   (views / placeholders)
[ BLAS ]       6
```

Every single `BLAS`-tagged node is `token_embd.weight` showing up as
the source of the very first `GET_ROWS` node, six times (once per
reservation graph + the live run). The actual `GET_ROWS` runs on
`CPU`; the embedding table is just labeled with the BLAS buffer type
because of how llama.cpp's CPU buft list orders ACCEL ahead of CPU
(`src/llama-model.cpp:816-830`). There are zero `MUL_MAT` or
`OUT_PROD` nodes routed to BLAS in any reservation or live graph at
`pp320`. The `pp320 = 241.x t/s` baseline is therefore an entirely
GPU+CPU result, with no sgemm calls in the hot path.

This also matches the CPU buffer accounting at the end of the run:
`CPU compute buffer size = 1.53 MiB`, which is dominated by tokenizer
/ embedding-input bookkeeping, not by any FFN/attention intermediate.

### Where Accelerate _is_ still hot

The BLAS backend isn't doing the work, but Accelerate is still linked
into the CPU backend through `GGML_USE_ACCELERATE`. Grepping
`ggml/src/ggml-cpu/` for vDSP / Accelerate references shows:

- `ggml-cpu/binary-ops.cpp` dispatches `vDSP_vadd / vsub / vmul / vdiv`
  for F32 element-wise ops.
- `ggml-cpu/vec.h` uses `vDSP_vsmsa`, `vDSP_vsmul`, `vDSP_sve`,
  `vDSP_maxv`, etc., for small vector ops.
- `ggml-cpu/ops.cpp` uses `vDSP_vadd`, `vDSP_vsadd`, `vDSP_measqv` in
  reductions / add1.
- A separate llamafile sgemm tile path (`ggml-cpu/llamafile/sgemm.cpp`,
  used from `ggml-cpu.c:1296` and `:1364`) handles CPU-side mat-mat
  for prompt processing when the CPU backend is the one actually
  running mat-mat. This is not Accelerate's sgemm; it's the bundled
  llamafile micro-kernels.

None of this is reachable from the live 27B Metal graph at `pp320`
because every mat-mat-class node is sitting on `[ MTL0 ]`. Accelerate
matters only for the slivers of CPU-side work — input embedding
gather, tokenizer prep, sampler — i.e. exactly the boundary work that
our own engine already does on Apple-CPU paths without sgemm.

### Implications for our scoreboard

1. `pp320 ~240.9 t/s` on `llama-bench` is a pure Metal number. Our gap
   to it is fully a GPU-engine gap, not a "missing fast CPU sgemm"
   gap.
2. The `MTL,BLAS` string in the backend column is a registration
   artifact, not a hot-path participation signal. We should mentally
   strip it when comparing to our `qwen-llm` numbers.
3. There is no upside in adding an Accelerate sgemm lane to qwen-llm
   for the dense Q4_K prompt path: even llama.cpp leaves BLAS idle
   here, because Metal buffer types report `is_host = false` and the
   sched's only BLAS-upgrade trigger requires host-mapped weights.
4. The one place a BLAS-equivalent path can still legitimately matter
   in llama.cpp builds is offloading mat-mats when weights are kept on
   CPU host memory (partial offload, MoE expert pinning, CPU-tier
   models). None of our guardrail Qwen3.5/3.6 configs do that, so it
   stays out of our scoreboard.

### Decision

- Do not pursue an Accelerate / BLAS lane for qwen-llm under current
  guardrails.
- Continue treating the `pp320` gap as a pure Metal-engine target;
  this matches the active roadmap item 1.
- Note the registration-string trap in the roadmap so we do not chase
  a phantom CPU lane in future llama-bench comparisons.

### Validation Method (for reproducing)

```text
# show registration vs. hot-path assignment
GGML_SCHED_DEBUG=2 ~/code/llama.cpp/build/bin/llama-bench \
    -m ~/models/Qwen3.6-27B-Q4_K_M.gguf -n 0 -p 320 -r 1 --verbose \
    2> /tmp/llamabench-sched.stderr

# tally per-op backend tags
rg -o '\[ ?(MTL0|CPU|BLAS|NULL) +\]' /tmp/llamabench-sched.stderr \
    | sort | uniq -c | sort -rn

# the BLAS rows
rg '\bBLAS\b' /tmp/llamabench-sched.stderr
```

Expected on M4 Max + dense Qwen3.6 Q4_K_M: tens of thousands of
`[ MTL0 ]`, ~6 `[ BLAS ]` rows, all of them on `token_embd.weight`'s
buffer label rather than a real compute node.

## 2026-05-16 — Reorient Around llama-bench Prompt Parity

Status: docs-only roadmap reset after fresh llama.cpp baselines.

### Fresh Baselines

- `qwen-llm` repeated 320-token dense prompt: `~205.4-205.9 t/s`
- current `llama-cli -st` on the same prompt: `~206.7 t/s` prompt,
  `~22.5 t/s` generation
- current `llama-bench pp320`: `~240.9 t/s`
- local merged-PR MTP single-turn check on
  `Qwen3.6-27B-MTP-Q4_K_M.gguf` with `draft-mtp`, `n_max=3`, `p_min=0.75`:
  `~182.3 t/s` prompt, `~22.4 t/s` generation

### Interpretation

- User-facing CLI parity is real enough now that it is no longer the hard target.
- The harder prompt-only scoreboard is `llama-bench`, and on that metric the
  remaining gap is still substantial.
- That means the roadmap should not be centered purely on the remaining GDN tail.
  The current GDN-tail headroom is real but not large enough by itself to close
  the full `llama-bench` pure-prompt miss.
- The merged llama.cpp MTP path is useful prior art, but not yet a scary speed
  baseline on this local single-turn Apple harness. Its own PR notes prompt-side
  penalties from D2H embedding transfers, which reinforces keeping prompt-path
  efficiency first-class in our spec thinking too.

### Method Reset

- Measurement work should only exist to test a concrete causal performance
  hypothesis.
- The immediate hypothesis worth testing is whether the remaining `pp320` miss is
  partly harness semantics rather than engine work.
- If that hypothesis fails, the next attack shifts back to engine structure,
  with prompt-native packed attention ahead of more small cleanup loops.

### On-Disk Priority Reset

1. Add a phase-matched pure prompt frontier harness in `qwen-bench` so we can
   compare against `llama-bench` on the right semantics.
2. Treat true packed prompt attention as the likely next major dense prompt lane
   if that harness confirms the remaining pure-prompt miss is real GPU work.
3. Keep the remaining GDN tail cleanup as a bounded follow-on, not the sole
   top-level plan.

## 2026-05-16 — Packed Attention Body Cleanup

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Used `cx` to scrutinize the packed attention-body plan before touching code.
- Added `kernel_rope_neox_f32_packed_consecutive` in `kernels/rope.metal` and
  `encode_rope_neox_f32_packed_consecutive` in `crates/qwen-llm/src/metal.rs`.
- Updated packed dense attention prefill in `crates/qwen-llm/src/metal_dflash.rs`
  so it now:
  - applies consecutive-position RoPE to packed Q once per chunk
  - applies consecutive-position RoPE to packed K once per chunk
  - scatters the whole chunk's K/V rows into the cache in one dispatch
  - keeps the per-token loop only for the actual attention decode step
- Kept the profiler-aligned packed attention path in sync with the new shape.

### Why This Was The Right Move

`cx` argued the best bounded next move was not a true packed causal-attention
 rewrite, but a cleanup of the existing decode-shaped attention body: remove the
 chunk's per-token RoPE(Q), RoPE(K), and K/V scatter dispatches first, then see
 what attention body still costs.

### Measured Impact

Repeated 320-token quick-brown-fox prompt, 27B dense, packed prefill chunk 512,
release build, sequential runs:

- Before this change (after v0.94): `~1582-1588 ms`, `201.5-202.3 t/s`,
  `~1570-1576 ms` GPU
- After this change: `~1554-1558 ms`, `205.4-205.9 t/s`,
  `~1545-1549 ms` GPU

Net:

- about `1.9-2.1%` faster prompt prefill on the repeated 27B prompt
- dense same-prompt gap versus current `llama.cpp` (`212.44 t/s`) is now down to
  roughly three percent

### Updated Dense Prompt Read

Representative post-change no-op on the same prompt:

- baseline: `~1556-1558 ms` wall, `~1545-1549 ms` GPU
- `QWEN_PREFILL_NOOP_ATTN_BODY=1`: `~1425-1449 ms` wall,
  `~1419-1441 ms` GPU

So the packed attention-body cleanup cut that bucket from roughly
`~162 ms` wall / `~156 ms` GPU down to about `~120 ms` wall / `~118 ms` GPU.

### Validation

- `cargo fmt --all`
- `cargo test --release -p qwen-llm rope_neox_packed_consecutive_matches_cpu -- --nocapture`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_27b -- --nocapture`
- `cargo build --release -p qwen-cli --bin qwen-bench`

Correctness stayed green:

- packed RoPE matches CPU in a dedicated unit test
- dense packed prefill oracle still passes with `cos(final logits)=1.000000`

### Current Next Step

1. Return to the remaining packed GDN tail / out-proj path with the split ladder,
   since attention is no longer the largest non-FFN dense prompt bucket.
2. Only revisit a more invasive packed attention rewrite if the narrower GDN-tail
   work stalls.

## 2026-05-16 — Packed GDN Prep Over Prompt Tokens

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Used `cx` to adversarially review the post-v0.93 dense prompt plan before the
  next implementation step.
- Added `QWEN_PREFILL_GDN_SPLIT={skip_all,out_only,prep_out,prep_step_out}` on
  the real packed-prefill graph to split packed GDN body cost without falling
  back to a cloned profiler path.
- Replaced the old packed GDN prep train in `crates/qwen-llm/src/metal_dflash.rs`
  with a new packed kernel in `kernels/ssm_conv.metal` plus two in-place batched
  L2 norms:
  - old shape: `P` launches of `ssm_conv_silu`, two L2 norms, and three scatters
    before the packed recurrence
  - new shape: one `kernel_gdn_prep_packed_f32` over the chunk, then two batched
    in-place L2 norms

### Why This Was The Right Move

The split ladder on the repeated 320-token 27B prompt, before the rewrite:

- baseline: `1718.3 ms` wall, `1691.5 ms` GPU
- `skip_all`: `1425.6 ms` wall, `1415.5 ms` GPU
- `out_only`: `1518.3 ms` wall, `1506.6 ms` GPU
- `prep_out`: `1664.4 ms` wall, `1637.2 ms` GPU
- `prep_step_out`: `1708.4 ms` wall, `1679.7 ms` GPU

That implies the old packed GDN prep loop was the largest GDN sub-bucket:

- out-proj tail: `~92.7 ms` wall / `~91.1 ms` GPU
- prep loop: `~146.1 ms` wall / `~130.6 ms` GPU
- packed recurrence: `~44.0 ms` wall / `~42.5 ms` GPU

So the prep loop, not the recurrence kernel itself, was the sharpest next dense
prompt target.

### Measured Impact

Repeated 320-token quick-brown-fox prompt, 27B dense, packed prefill chunk 512,
release build, sequential runs:

- Before packed prep rewrite: `~1715-1720 ms`, `186.0-186.5 t/s`,
  `~1687-1693 ms` GPU
- After packed prep rewrite: `~1582-1588 ms`, `201.5-202.3 t/s`,
  `~1570-1576 ms` GPU

Net:

- about `7.8-8.2%` faster prompt prefill on the repeated 27B prompt
- dense same-prompt gap versus current `llama.cpp` (`212.44 t/s`) is now down to
  roughly five percent

Fresh current dense prompt read after the rewrite:

- baseline: `1581.8 ms` wall, `1570.6 ms` GPU, `202.3 t/s`
- `QWEN_PREFILL_NOOP_ATTN_BODY=1`: `1420.0 ms` wall, `1414.3 ms` GPU,
  `225.3 t/s`

That makes packed attention body the next largest non-FFN dense prompt bucket at
about `~162 ms` wall / `~156 ms` GPU.

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_27b -- --nocapture`

Correctness stayed green:

- `cos(final logits)=1.000000`
- hidden / GDN state / conv / KV cache gates all remained effectively exact.

### Current Next Step

1. Packed attention body cleanup: batched consecutive-position RoPE and chunk-wise
   KV scatter / glue removal before considering a more invasive packed attention
   rewrite.
2. Then return to the remaining GDN out-proj / recurrence tail only if attention
   cleanup does not move the prompt enough.

## 2026-05-16 — Packed Prompt Differential Profiling + Batched GDN Gating

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- `qwen-bench decode` now reports total packed-prefill GPU time via
  `prefill_tokens_with_multi_hidden_profiled`.
- Added dense prompt differential profiling flags on the real packed prefill
  graph:
  - `QWEN_PREFILL_NOOP_FFN=1`
  - `QWEN_PREFILL_NOOP_GDN_BODY=1`
  - `QWEN_PREFILL_NOOP_ATTN_BODY=1`
- Batched packed GDN `rmsnorm_gated` over the whole prompt chunk instead of one
  dispatch per token.

### Fresh Dense Prompt Read

Repeated 320-token quick-brown-fox prompt, 27B dense, packed prefill chunk 512,
release build, sequential runs:

- Latest packed prefill plateau after batched GDN gating: `~1718 ms` wall,
  `~1691 ms` GPU, `~186.2 t/s`.
- Fresh current `llama.cpp` prompt baseline on the same machine/model family:
  `212.44 t/s`.
- Prompt remains behind, but the gap is now about `~14%`, not the earlier
  `~16%` pre-win read.

### Production-Shape Differential Prompt Deltas

Representative no-op runs after the new batching change:

- `QWEN_PREFILL_NOOP_FFN=1`: `793.0 ms` wall, `766.2 ms` GPU.
- `QWEN_PREFILL_NOOP_GDN_BODY=1`: `1432.1 ms` wall, `1422.2 ms` GPU.
- `QWEN_PREFILL_NOOP_ATTN_BODY=1`: `1557.8 ms` wall, `1535.7 ms` GPU.

Against the new `~1718 ms` / `~1691 ms` baseline, that says the live packed
prompt graph is approximately:

- FFN: `~925 ms`
- GDN body: `~286 ms` wall, `~269 ms` GPU
- attention body: `~160 ms` wall, `~155 ms` GPU

Interpretation:

- Prompt is still overwhelmingly real GPU work (`~98.3%` GPU / wall), not outer
  orchestration.
- The old "FFN is the hidden prompt mystery" theory is now dead: direct
  production-shape FFN delta matches the broad bucket story, and isolated exact
  shape FFN mat-mat refs already match or beat exported llama.cpp prompt ops.
- The highest-EV remaining dense prompt work is now the non-FFN prompt path:
  first packed GDN body staging, then attention body cleanup.

### Measured Win

- Before batched packed `rmsnorm_gated`: `~1747.5 ms`, `183.1 t/s`,
  `~1717.1 ms` GPU.
- After batching it over the full chunk: `~1715-1720 ms`, `186.0-186.5 t/s`,
  `~1687-1693 ms` GPU.
- Net: about `1.7-1.9%` faster prompt prefill on the repeated 27B prompt.

### Validation

- `cargo build --release -p qwen-cli --bin qwen-bench`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_27b -- --nocapture`

Correctness stayed green:

- `cos(final logits)=1.000000`
- hidden / GDN state / conv / KV cache gates all remained effectively exact.

### Current Next Step

1. Packed GDN body cleanup: attack SSM-conv prep and Q/K norm + V-pack staging
   around the existing packed recurrence.
2. Packed attention body cleanup: batched consecutive-position RoPE and chunk-wise
   KV scatter / glue removal before considering a more invasive packed attention
   rewrite.

## 2026-05-14 — Attention Parity Push + Roadmap Reset

Status: improved checkpoint reached, not yet committed in git.

### Current Performance State

Sequential release `qwen-bench` runs on M4 Max:

| Model | Context | Total ms/token | Tokens/s | Notes |
| --- | ---: | ---: | ---: | --- |
| 27B dense | 4K | 42.80 | 23.4 | dense group6 `NWG=64` |
| 27B dense | 16K | 46.11 | 21.7 | attention ~14.3 ms |
| 27B dense | 32K | 51.25 | 19.5 | attention ~19.3 ms |
| 35B A3B | 4K | 14.65 | 68.2 | MoE guardrail |
| 35B A3B | 16K | 17.12 | 58.4 | MoE guardrail |
| 35B A3B | 32K | 20.45 | 48.9 | MoE guardrail |
| 122B A10B | 4K | 31.42 | 31.8 | group16 tile4 + `NWG=64` |
| 122B A10B | 16K | 33.47 | 29.9 | group16 tile4 + `NWG=64` |
| 122B A10B | 32K | 35.14 | 28.5 | group16 tile4 + `NWG=64` |

### Confirmed Changes / Wins

- Promoted group16 tile4 default for long-context 122B A10B attention.
- Promoted dense group6 attention to `NWG=64` for `n_pos >= 4096`.
- Added attention A/B knobs: `QWEN_ATTN_V4_NWG`, `QWEN_ATTN_V4_TILE_C`,
  `QWEN_ATTN_V4_G16_TILE`.
- Expanded attention correctness to cover production-style `NWG=64` cases.
- Fixed stale attention/GDN intra-profilers so they time production paths.
- Added MoE intra-block profiler to split mixer, route, routed FFN, shared FFN,
  and residual pieces.
- Added `docs/PERF-ROADMAP.md` as the active force-ranked optimization queue.

### Key Measurement Deltas

Dense 27B long-context attention improved materially:

- 16K phase: old total ~50.42 ms / attention ~18.79 ms; `NWG=64` total
  ~46.19 ms / attention ~14.28 ms.
- 32K phase: old total ~60.32 ms / attention ~28.62 ms; `NWG=64` total
  ~51.12 ms / attention ~19.28 ms.

MoE `NWG=64` guardrails beat `NWG=32`:

- 35B A3B `NWG=32`: 4K 15.30 ms, 16K 20.20 ms, 32K 26.81 ms.
- 35B A3B default `NWG=64`: 4K 14.65 ms, 16K 17.12 ms, 32K 20.45 ms.
- 122B A10B `NWG=32`: 4K 31.91 ms, 16K 35.66 ms, 32K 40.54 ms.
- 122B A10B default `NWG=64`: 4K 31.42 ms, 16K 33.47 ms, 32K 35.14 ms.

Fresh subphase read:

- Dense 27B one GDN layer: ~0.646 ms; largest pieces are FFN gate/up/silu
  (~0.209 ms), FFN down (~0.162 ms), then GDN projections.
- 35B A3B one MoE block: ~0.299 ms; mixer prep dominates (~0.166 ms).
- 122B A10B one MoE block: ~0.594 ms; mixer prep dominates (~0.365 ms),
  shared FFN totals only ~0.074 ms.

### Validation Run

- `cargo fmt --all`
- `cargo check -p qwen-llm`
- `cargo test -p qwen-llm --no-run`
- `cargo test --release -p qwen-llm attn_v4_matches_naive_f16kv -- --nocapture`
- `cargo build --release -p qwen-cli --bin qwen-bench`

Notes:

- Existing warnings remain from upstream `llama-cpp-sys-2` and two ignored-test
  unused variables in `metal.rs`; no new functional failures observed.
- All performance runs above were sequential, not parallel.

### Current Force-Ranked Next Work

1. Wire dense packed prefill into normal no-spec `qwen-bench decode`.
2. Add no-spec GPU argmax / avoid full logits readback for greedy decode.
3. Prototype/design KV-Q8 for long-context attention.
4. Build MoE packed routed-expert prefill.
5. Measure command overhead before deciding on ICB / MTL4.
6. Tune prefill mat-mat quality after main packed prefill is wired.
7. Revisit dense GDN/FFN decode only with sharper subphase evidence.

### Workspace / Commit State

Current repo state is intentionally dirty and includes broad uncommitted work
from this performance arc. Do not assume all modified files belong to one small
change set.

Known currently uncommitted new docs from this checkpoint:

- `docs/PERF-ROADMAP.md`
- `docs/PERF-LOG.md`

Checkpoint commit style going forward:

```text
v0.xx: concise optimization headline

Explain the story in the body: what changed, why it matters, measured
impact, validation, risks, and next step. Keep the subject short enough
to scan cleanly; put numbers in the body unless they are essential to
the headline.
```

Before committing, inspect staged/unstaged diff carefully and include only the
intended checkpoint files/changes. Do not commit secrets or unrelated scratch.
Each commit should represent one measured optimization or one deliberate
workflow/documentation checkpoint.

### Next Handoff Instruction

Start by reading:

1. `docs/PERF-LOG.md`
2. `docs/PERF-ROADMAP.md`
3. `docs/INFERENCE-GRAPH.md`
4. `git status --short`

Then continue with the ranked item #1 unless fresh measurements or user
direction change priority.

## 2026-05-14 — Dense Packed Prefill Defaulted In No-Spec Decode

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- `qwen-bench decode` now defaults dense models to the existing packed prefill
  path (`prefill_tokens_with_multi_hidden` with no hidden capture).
- Added `--sequential-prefill` for explicit A/B against the legacy prompt replay
  loop.
- Kept MoE models on sequential prefill until routed-expert packed prefill lands.
- Warmup now uses the selected prefill mode, so packed vs sequential A/B is not
  confounded by one-token-only sequential warmup.
- Added `--oracle-phase {prefill,final}` so oracle comparisons can target either
  the last prompt-token logits or the final decode-step logits.
- Tightened CLI behavior for empty prompts and zero-decode reporting.

### Measured Impact

27B dense, 321-token prompt, 4 decode tokens, release build, sequential runs:

- Packed prefill default: 4109.2 ms prefill = 12.80 ms/token = 78.1 t/s.
- Legacy sequential prefill: 13235.5 ms prefill = 41.23 ms/token = 24.3 t/s.
- Prefill speedup: ~3.22x.
- Decode unchanged within noise: ~41 ms/token on both paths.
- Generated text matched on the measured A/B run.

### Validation

- `cargo fmt --all`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo test --release -p qwen-llm --test dflash_correctness prefill_tokens_matches_single_token_loop_27b -- --nocapture`
- `cargo build --release -p qwen-cli --bin qwen-bench`

Key correctness signal:

- `prefill_tokens_matches_single_token_loop_27b` passed again after wiring the
  CLI path; correctness test still shows ~2.78x standalone oracle-vs-packed
  speedup with cosine agreement on logits, hidden capture, GDN state, conv, and
  KV cache.

### Current Next Step

Roadmap item #1 is now no-spec GPU argmax / avoid full logits readback.

### Suggested Checkpoint Commit

```text
v0.77: packed dense prefill in no-spec decode
```

## 2026-05-14 — GPU Argmax Decode Path + A/B Harness

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Added `single_token_argmax` / `single_token_argmax_profiled` to the Metal
  forward path for dense and MoE decode.
- `qwen-bench decode` now defaults to GPU-argmax decode and avoids full logits
  readback on greedy steps unless `--full-logits-decode` is set.
- Added `--full-logits-decode` for direct A/B measurement.
- Warmup now exercises the selected decode mode, including the argmax path.
- Added `single_token_argmax*` regression tests for dense and MoE short chains.

### Measured Impact

Current sequential A/B runs, 321-token prompt, 64 decode tokens:

- 27B dense:
  - full logits: 41.01 ms/token
  - gpu argmax: 41.09 ms/token
  - result: neutral within noise on dense; no convincing dense decode win yet.
- 35B A3B:
  - full logits: 13.90 ms/token
  - gpu argmax: 13.74 ms/token
  - result: ~1.1% decode win.
- 122B A10B:
  - full logits: 31.62 ms/token
  - gpu argmax: 31.30 ms/token
  - result: ~1.0% decode win.

Interpretation:

- GPU argmax is not a major dense ceiling breaker; dense benefit is neutral in
  current measurements.
- It is still a modest positive for MoE decode and reduces readback volume for
  greedy generation.

### Validation

- `cargo fmt --all`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo check -p qwen-llm`
- `cargo test -p qwen-llm --no-run`
- `cargo test --release -p qwen-llm metal_argmax_chain_matches_full_logits_dense -- --nocapture`
- `cargo test --release -p qwen-llm metal_argmax_chain_matches_full_logits_moe -- --nocapture`

Regression tests passed:

- dense 0.8B chain: cos = 1.000000, argmax path matches full-logits path.
- MoE A3B chain: cos = 1.000000, argmax path matches full-logits path.

### Current Next Step

Shift main pressure to KV-Q8 for long-context decode, while keeping the new
GPU-argmax path honest via `--full-logits-decode` A/B and the new regression
tests.

### Suggested Checkpoint Commit

```text
v0.78: gpu argmax decode path
```

## 2026-05-14 — KV-Q8 Dense Prototype: Negative Result

Status: negative result reached; do not spend more blind sweep time on the
current dense KV-Q8 main-kernel shape.

### What Changed

- Added an experimental dense-only `QWEN_KV_Q8=1` path:
  - KV cache allocates as Q8_0 instead of F16 for dense group6 / head_dim=256.
  - Fused K+V append quantizes with exact ggml Q8_0 rules.
  - Dense v4 attention main pass can read Q8 KV; reduce path unchanged.
  - Snapshot identity now records KV byte width, so prefix/snapshot state
    cannot silently alias F16 and Q8 layouts.
- Added two gates:
  - `scatter_kv_q8_matches_ref_quant` exact-byte test
  - `attn_v4_q8_kv_close_to_f16_kv` similarity gate (`cos=0.999994`)

### Measured Impact

Dense 27B, sequential runs:

Baseline (current best F16 KV path):

- 4K: 42.80 ms/token
- 16K: 46.11 ms/token
- 32K: 51.25 ms/token
- 64K: 61.26 ms/token

KV-Q8 prototype:

- 4K: 43.11 ms/token
- 16K: 47.58 ms/token
- 32K: 53.49 ms/token

Phase evidence says the loss is in the attention read path itself, not append:

- F16 KV attention bucket:
  - 16K: ~14.28 ms
  - 32K: ~19.28 ms
- Q8 KV attention bucket:
  - 16K: ~15.36 ms
  - 32K: ~21.55 ms

Codex-wrap hypothesis: our current Q8 reader loses to a very good F16 path on
Apple/M4 because the scalar Q8 dequant/load structure outweighs the stored-byte
savings, especially since v4 already amortizes KV reads across the dense GQA
group.

### Kill Test

- Tried a fast scale-broadcast style rescue on the Q8 v4 path.
- Result: worse, not better.
- Decision: cut bait on this Q8 main-body shape for now.

### Conclusion

- Keep the experimental Q8 path as evidence / future reference only if useful.
- Do NOT invest more time in broad KV-Q8 sweeps on this implementation.
- Pivot the main optimization pressure to MoE packed routed-expert prefill.

### Suggested Checkpoint Commit

```text
v0.79: KV-Q8 negative result on dense v4
```

## 2026-05-14 — MoE Packed Prefill, Stage 1

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- `prefill_tokens_with_multi_hidden` now supports MoE target models.
- The new MoE packed path batches mixer prep and post-norm across the chunk,
  then runs routed FFN per token with the existing exact MoE route/apply path,
  scattering the updated `session.x` back into the packed chunk state.
- `qwen-bench decode` now defaults to packed prefill for MoE too; the legacy
  path remains available via `--sequential-prefill`.

### Validation

- New correctness gate:
  `prefill_tokens_matches_single_token_loop_35b_a3b_moe`
- Result:
  - `cos(final logits)=1.000000`
  - `GDN state cos_min=1.000000`
  - `KV K/V cos_min=1.000000`
  - small-gate speedup in correctness harness: `~2.69x`

Codex-wrap review:

- No obvious correctness blocker in the packed MoE shape.
- Main remaining test gap: MoE hidden-capture path still lacks a dedicated gate.
- Highest-EV next increment: grouped routed-expert execution, but add packed-MoE
  phase profiling first so the next step is aimed at the actual residual waste.

### Measured Impact

321-token prompt, 64 decode tokens, sequential runs:

- 35B A3B:
  - sequential prefill: 4290.1 ms = 74.8 t/s
  - packed prefill: 3514.4 ms = 91.3 t/s
  - improvement: `~22%` faster prefill
  - decode: essentially unchanged within noise
- 122B A10B:
  - sequential prefill: 10001.4 ms = 32.1 t/s
  - packed prefill: 8760.1 ms = 36.6 t/s
  - improvement: `~14%` faster prefill
  - decode: essentially unchanged within noise

### Current Next Step

Add packed-MoE phase timing and a MoE hidden-capture/chunk-boundary gate, then
attack grouped routed-expert execution for the routed branch.

### Suggested Checkpoint Commit

```text
v0.80: packed prefill for MoE no-spec decode
```

## 2026-05-14 — Dense Packed Prefill Chunk Tuning

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Added `--prefill-chunk` to `qwen-bench decode` for packed prefill A/B work.
- `qwen-bench decode` now chooses a model-aware packed prefill chunk by default:
  - dense: `256`
  - MoE: `16`

This was driven by `cx` review: our inherited `P=16` came from DFlash, not from
any dense prompt-time evidence.

### Dense 27B Prompt Sweep (same 321-token prompt, `--tokens 0`)

- `P=8`: `35.0 t/s`
- `P=16`: `77.9 t/s`
- `P=32`: `103.6 t/s`
- `P=64`: `127.2 t/s`
- `P=128`: `136.1 t/s`
- `P=256`: `139.9 t/s`
- `P=321`: `141.8 t/s`
- `P=384`: `141.9 t/s`
- `P=512`: `141.7 t/s`

Dense prefill effectively saturates once the whole 321-token prompt fits in a
single packed chunk. We keep the dense default at `256` as a conservative
near-optimal point with materially smaller scratch than `512`.

### Product-Shaped Dense 27B Result (same prompt, 64 decode tokens)

- Packed prefill with dense default `P=256`:
  - prefill: `2281.3 ms` = `140.7 t/s`
  - decode: `40.96 ms/token` = `24.4 t/s`

Comparison to earlier dense packed prefill default (`P=16`):

- old prefill: `~79.9 t/s`
- new prefill: `~140.7 t/s`
- improvement: `~1.76x` over the prior packed default

Comparison to same-prompt llama.cpp data the user supplied:

- llama.cpp prompt: `186.8 t/s`
- qwen-llm prompt after tuning: `140.7 t/s`

This does not close the full prompt gap, but it narrows it substantially.

### Conclusion

- Dense prompt processing was being artificially capped by a bad inherited chunk
  size, not just by deep kernel limitations.
- Dense prefill remains the biggest remaining dense gap vs llama.cpp, but the
  gap is now materially smaller.
- MoE chunk-size sweep is next; do not assume the dense result transfers.

### Suggested Checkpoint Commit

```text
v0.81: tune dense packed prefill chunk size
```

## 2026-05-15 — Batched Dense GDN Alpha/Beta Prompt Path

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Added F32 packed mat-mat support for prompt-style `[N, H] x [H, n_out]` in the
  narrow form we need for small output widths.
- Dense packed prefill now batches GDN `beta_proj` and `alpha_proj` across the
  whole packed chunk instead of re-running them token-by-token.
- Added a batched GDN decay-chain kernel so `[N, n_v]` alpha activations can be
  turned into per-token decay values in one pass.

### Dense Prompt Result

Same repeated 321-token 27B prompt:

- before this change: prompt plateau ~`141.9 t/s`
- after this change: prompt plateau ~`165.0 t/s`

Product-shaped run (`64` decode tokens):

- packed prefill with dense default `P=512`: `1947.4 ms` = `164.8 t/s`
- decode unchanged: `41.04 ms/token` = `24.4 t/s`

This moves the same-prompt dense prompt gap versus llama.cpp from roughly
`186.8 / 141.9 = 1.32x` behind to about `186.8 / 165.0 = 1.13x` behind.

### Updated Dense Packed-Prefill Attribution (`P=321`)

- total: `2016.30 ms`
- `ffn`: `1008.40 ms` (`50.0%`)
- `gdn_front`: `291.92 ms` (`14.5%`)
- `gdn_alpha_beta`: `0.26 ms` (effectively gone)
- `gdn_tail`: `359.69 ms` (`17.8%`)
- `gdn_back`: `100.05 ms` (`5.0%`)
- `attn_front`: `72.89 ms` (`3.6%`)
- `attn_decode`: `148.82 ms` (`7.4%`)
- `attn_back`: `31.73 ms` (`1.6%`)

Interpretation:

- The old dense prompt bottleneck from `alpha/beta` projections is no longer
  relevant.
- Dense prompt time is now dominated by FFN mat-mat and the true GDN tail.

### Supporting Fast-Feedback Probe

Exact production-shape prompt probes at `N=321` still show mat-mat strongly
beating repeated mat-vec on the real 27B weights:

- `ffn_gate` Q4_K: `34.45 ms` vec321 vs `5.32 ms` mat-mat (`6.47x`)
- `ffn_up` Q4_K: `32.24 ms` vec321 vs `5.27 ms` mat-mat (`6.12x`)
- `ffn_down` Q6_K: `51.59 ms` vec321 vs `5.68 ms` mat-mat (`9.08x`)
- `attn_qkv` Q6_K: `27.26 ms` vec321 vs `3.21 ms` mat-mat (`8.50x`)

No-code falsification check from `cx` recommendation:

- llama.cpp same-prompt run with Metal tensor path forced on/off on M4 Max
  showed no meaningful delta (`~207.0` vs `~207.2 t/s` prompt in the
  one-token probe), so the Metal tensor path is not obviously the missing trick
  on this hardware.

### Current Next Step

Per `cx`, the sharpest next dense attack is now the GDN tail bucket, not a
broad mat-mat backend port. The best fast-feedback next step is to split the
`gdn_tail` bucket further and/or prototype a packed `gdn_step_decay` time-loop
kernel before attempting a larger rewrite.

### Suggested Checkpoint Commit

```text
v0.83: batch dense GDN alpha and beta prompt path
```

### Follow-on State (same checkpoint arc)

- Added a denser packed-prefill profiler for the real 27B prompt and a
  representative one-layer GDN tail subprofile.

Updated dense packed-prefill attribution (`P=321`):

- total: `2016.30 ms`
- `ffn`: `1008.40 ms` (`50.0%`)
- `gdn_front`: `291.92 ms` (`14.5%`)
- `gdn_tail`: `359.69 ms` (`17.8%`)
- `gdn_back`: `100.05 ms` (`5.0%`)
- `attn_decode`: `148.82 ms` (`7.4%`)

Representative one-layer GDN tail split over the same 321-token prompt:

- total: `42.63 ms`
- `conv`: `1.56 ms`
- `l2`: `2.10 ms`
- `step_decay`: `6.16 ms`
- `rmsnorm_gated`: `1.67 ms`
- `out_proj`: `31.13 ms`

Interpretation:

- In the full dense prompt profile, `out_proj` already belongs to `gdn_back`, so
  the true `gdn_tail` bucket is the sum of `conv + l2 + step_decay + rmsnorm`.
- Within that true tail, `step_decay` is the largest sub-bucket.
- `cx` review says the sharpest next checkpoint is a bounded packed
  `gdn_step_decay` time-loop falsification kernel over prompt tokens; if it does
  not buy roughly `80-100 ms` end-to-end, pivot back toward broader FFN/backend
  work.

### Suggested Checkpoint Commit

```text
v0.84: profile dense GDN tail prompt bucket
```

## 2026-05-15 — Packed Dense GDN Step Time-Loop

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Added an experimental packed `gdn_step_decay` kernel that loops over prompt
  tokens inside the kernel while keeping each GDN state row resident across the
  whole packed chunk.
- Dense packed prefill now uses this packed step path by default; the old path is
  still available as a kill switch via `QWEN_DENSE_GDN_STEP_PACKED=0`.
- Updated the dense packed-prefill profiler so its phase numbers reflect the new
  active path.

### Validation

- `prefill_tokens_matches_single_token_loop_27b` passes on the default path:
  - `cos(final logits)=1.000000`
  - `hidden cos_min=0.999999`
  - `GDN state cos_min=0.999999`
  - `KV K/V cos_min=1.000000`

### Dense Prompt Result

Same repeated 321-token 27B prompt:

- before this change: prompt plateau ~`165.0 t/s`
- after this change: prompt plateau ~`172.9 t/s`

Product-shaped run (`64` decode tokens):

- packed prefill with dense default `P=512`: `1861.1 ms` = `172.5 t/s`
- decode unchanged: `41.12 ms/token` = `24.3 t/s`

This closes the same-prompt dense prompt gap vs the earlier llama.cpp reading
(`186.8 t/s`) to roughly `1.08x`.

### Updated Dense Packed-Prefill Attribution (`P=321`)

- total: `1896.62 ms`
- `ffn`: `1008.61 ms` (`53.2%`)
- `gdn_front`: `295.71 ms` (`15.6%`)
- `gdn_tail`: `236.87 ms` (`12.5%`)
- `gdn_back`: `100.12 ms` (`5.3%`)
- `attn_front`: `72.86 ms` (`3.8%`)
- `attn_decode`: `148.16 ms` (`7.8%`)
- `attn_back`: `31.73 ms` (`1.7%`)

Interpretation:

- Packed `gdn_step_decay` reduced the true dense `gdn_tail` bucket from roughly
  `359.69 ms` to `236.87 ms` in the same profiler.
- Dense prompt time is now even more dominated by the FFN mat-mat surface.

### Current Next Step

- The next dense lever is likely the broad FFN / projection mat-mat surface,
  unless a sharper bandwidth indictment says otherwise.
- MoE still wants grouped routed-expert execution as the next structural win.

### Suggested Checkpoint Commit

```text
v0.85: pack dense gdn step over prompt tokens
```

## 2026-05-15 — Drop Unused Prompt Logits Scratch

Status: improved checkpoint reached, not yet committed in git.

### What Changed

- Added a lighter `MetalDFlashLayerMajorScratch::fresh_prefill` constructor that
  skips the huge `[P, V]` `final_logits_pack` allocation when the caller only
  needs `prefill_tokens_with_multi_hidden`.
- Switched packed prompt-prefill call sites in `qwen-bench decode` and related
  no-spec prompt paths to use the lighter scratch.

Rationale:

- With dense `P=512`, the old scratch shape allocated a very large unused
  `[P, V]` buffer during timed prompt prefill. That was both unnecessary memory
  pressure and unnecessary timed wall.

### Measured Impact

Same repeated 321-token prompt, product-shaped runs:

- Dense 27B (`P=512`, packed step path already on):
  - before: `172.5 t/s` prefill
  - after: `173.3 t/s` prefill
  - decode unchanged / slightly better within noise (`~24.5 t/s`)
- 35B A3B (`P=128` default): `95.1 t/s` prefill, no regression.
- 122B A10B (`P=128` default): `37.6 t/s` prefill, no regression.

Interpretation:

- This is a small but real cleanup checkpoint, not a giant algorithmic leap.
- It removes a bad allocation pattern from the hot prompt path and shaves a bit
  more prompt wall on the dense target while keeping the MoE path clean.

### Current Dense Prompt State

- Same-prompt dense 27B prompt throughput is now about `173.3 t/s`.
- That is very close to the earlier same-prompt llama.cpp reading of `186.8 t/s`.

### Suggested Checkpoint Commit

```text
v0.86: drop unused prompt logits scratch
```

## 2026-05-15 — Dense Prompt Mat-Mat Audit + Direction Check

Status: measurement checkpoint reached, no production fast path changed.

### What Changed

- Added an exact-shape chained prompt mat-mat audit for real 27B production
  surfaces at `N=321`.
- Tried a Q4 large-`N` (`NR1=64`) prompt mat-mat specialization and measured it.
- It got worse, so it was reverted immediately.

### Exact-Shape Prompt Mat-Mat Audit

Chained prompt-shape numbers (`N=321`, `64` chained dispatches) on real 27B
weights:

- `blk.0.ffn_gate.weight` Q4_K:
  - `~5.09 ms / dispatch`
  - `~9.2 GiB/s` weight throughput
- `blk.0.ffn_up.weight` Q4_K:
  - `~5.09 ms / dispatch`
  - `~9.2 GiB/s` weight throughput
- `blk.0.ffn_down.weight` Q6_K:
  - `~5.45 ms / dispatch`
  - `~12.5 GiB/s` weight throughput
- `blk.0.attn_qkv.weight` Q6_K:
  - `~3.03 ms / dispatch`
  - `~13.2 GiB/s` weight throughput

Interpretation:

- Prompt mat-mat is already the right algorithmic shape, but backend quality is
  still very low versus the hardware envelope and versus our decode mat-vecs.
- The broad dense FFN / projection mat-mat surface is still a legitimate next
  dense lever, but the first easy Q4 large-`N` specialization was not the win.

### Direction Check

- `cx` review says the highest-EV branch overall is still grouped routed-expert
  execution for MoE packed prefill.
- For dense, the next checkpoint should be chosen carefully: either a sharper
  backend-quality experiment or a more principled mat-mat rewrite, not another
  casual tile tweak.

### Suggested Checkpoint Commit

```text
v0.87: audit dense prompt mat-mat backend
```

### Follow-on State (same checkpoint arc)

- MoE packed prefill chunk sweep on the same 321-token prompt (`--tokens 0`):
  - 35B A3B: `P=8 85.9`, `16 87.0`, `32 89.8`, `64 92.1`, `128 94.9`,
    `256 95.0`, `321 95.2` t/s
  - 122B A10B: `P=8 35.8`, `16 36.2`, `32 37.2`, `64 37.7`, `128 37.7`,
    `256 37.8`, `321 37.7` t/s
- Decision: move the default MoE packed prefill chunk from `16` to `128`.
- Product-shaped check with new default (`64` decode tokens):
  - 35B A3B prefill: `95.3 t/s`
  - 122B A10B prefill: `37.6 t/s`

- Added a MoE hidden-capture/chunk-boundary gate using a `P=1` packed oracle
  against `P=8` packed prefill; it passes with `cos(final logits)=1.0` and
  `hidden cos_min=1.0` on 35B A3B.
- Added packed-MoE tail attribution helpers.

Packed-MoE tail attribution (chunk_p=8):

- 35B A3B:
  - postnorm: `0.01 ms` (`0.4%`)
  - route+copy: `0.81 ms` (`20.5%`)
  - routed_ffn: `1.84 ms` (`47.0%`)
  - shared+resid+copy: `1.26 ms` (`32.1%`)
- 122B A10B:
  - postnorm: `0.02 ms` (`0.2%`)
  - route+copy: `0.91 ms` (`13.7%`)
  - routed_ffn: `3.78 ms` (`56.9%`)
  - shared+resid+copy: `1.94 ms` (`29.2%`)

Interpretation:

- Routed expert execution is clearly the largest remaining packed-MoE tail
  bucket on both A3B and 122B.
- Shared branch is still meaningful, but secondary.

Attempted next step:

- Tried a first packed-slot routed-expert execution path.
- Hard correctness gate failed immediately (`cos(final logits) ~ 0.97465`), so
  the active execution path was reverted to the last known-correct stage-1 MoE
  implementation.
- Result: keep the profiler/test scaffolding, but do not keep a broken fast path
  live in the tree.

## 2026-05-15 — Group-4 Attention v4 And 9B Long-Context Canary

Status: enablement checkpoint reached; small dense family can now use the v4
long-context attention path.

### What Changed

- Added `attn_v4` F16 kernels for `group=4` in `kernels/attn_v4.metal`, including
  the reduce path.
- Wired host dispatch selection through `crates/qwen-llm/src/metal.rs`,
  `crates/qwen-llm/src/metal_forward.rs`, `crates/qwen-llm/src/metal_dflash.rs`,
  and the packed correctness plumbing so `group=4` shapes stop falling back to
  the old threadgroup-memory-limited `attn_decode_f16kv` path.
- Extended the v4-vs-naive correctness test to cover the small dense shape
  (`n_q=8`, `n_kv=2`, `group=4`).

### Validation

- `cargo check -p qwen-llm`
- `cargo test --release -p qwen-llm attn_v4_matches_naive_f16kv -- --nocapture`
  passes with exact-style agreement for the new `group=4` shape across
  `n_pos ∈ {1, 32, 64, 256, 1024, 4096}`, `NWG ∈ {1, 2, 4, 8, 16, 64}` where
  applicable, and `C ∈ {16, 32, 64, 128}`.

### 9B Long-Context Canary Result

Model: `/Users/tito/models/Qwen3.5-9B-Q4_K_M.gguf`

`qwen-bench ctx-sweep --checkpoints 1,4096,8192,16384,32768 --window 2`:

- `1`: `14.75 ms` / `67.8 t/s`
- `4096`: `15.63 ms` / `64.0 t/s`
- `8192`: `15.95 ms` / `62.7 t/s`
- `16384`: `16.82 ms` / `59.5 t/s`
- `32768`: `18.57 ms` / `53.8 t/s`

Interpretation:

- The 9B no longer hits the old `~7K` long-context cliff.
- Group `4` is the one immediate unlock for the whole small dense family
  (`0.8B / 2B / 4B / 9B`), so we now have a much faster dense long-context
  canary without giving up the 27B guardrail.

### Direction Check

- The new `cx` review on the `ds4` close read reinforces the current ordering:
  grouped expert-major MoE prefill remains first, fast-path validation moves up
  beside it, and the dense branch should try paired same-input projection fusion
  before broader mat-mat gardening.
- `ds4` also surfaces two later but promising structural ideas to keep on deck:
  no-copy GGUF-backed Metal views with residency warmup, and a frontier
  snapshot/restore benchmark harness.

## 2026-05-15 — MoE Follow-On Falsifications + 27B 4K Trace Harness

Status: no new performance checkpoint; several important branches were cleanly
falsified and the command-model picture is now sharper.

### MoE Follow-On Results

Grouped expert-major routed FFN, implemented as CPU ledger + gather/scatter +
generic per-expert mat-mat, was semantically correct but strongly negative:

- 35B A3B:
  - `chunk=8`: `0.50 ms -> 8.64 ms` (`0.06x`)
  - `chunk=128`: `7.42 ms -> 20.13 ms` (`0.37x`)
- 122B A10B:
  - `chunk=8`: `0.99 ms -> 8.93 ms` (`0.11x`)
  - `chunk=128`: `15.19 ms -> 26.59 ms` (`0.57x`)

Interpretation:

- Generic grouped GEMM is the wrong organization here.
- Average expert groups are too small, and gather/scatter overhead dominates.

Two more MoE follow-ons also failed to earn a checkpoint:

- Batched shared-expert stage-2 rewrite: correct, but slower end-to-end on A3B.
- F16 routed-inner traffic reduction on the live Q5-down path: correct, but a
  wash-to-slight loser end-to-end.

Current MoE read after the falsifications:

- Stage-1 packed MoE prefill remains the live baseline.
- Further MoE upside likely needs either smaller token-major cleanup or a truly
  custom persistent grouped kernel, not another generic grouped experiment.

### Dense Prompt Follow-On Results

Dense paired prompt fusion was also pushed to a real go/no-go point and failed
to clear the bar:

- Shared-X paired `gate+up` Q4 prompt kernel at exact-shape 27B `N=321`:
  - `1.04x` microbench speedup over two separate mat-mats (`64` FFN layers)
  - exact-correct numerically
- Narrower `NR1=16` paired kernel: worse (`0.86x`)
- Forcing single Q4 prompt mat-mat itself to `NR1=16` at `N=321` also regressed:
  - `5.09 ms -> 6.37 ms` per dispatch

Interpretation:

- The easy paired-fusion / narrower-tile branch is mostly tapped out.
- Dense prompt should pivot toward less-staged Q4 mat-mat traversal/locality,
  not another fusion-first attempt.

### New Trace Tooling

- Added `qwen-bench decode-window`, an attach-friendly helper that warms to a
  target context, writes a ready file, waits for a go file, then runs a fixed
  decode window.
- Added `scripts/profile/trace-metal.py`, a repo-local Metal System Trace
  summarizer that reports command-buffer cadence and related stats without raw
  XML spelunking.

These exist specifically to keep Metal timeline work aligned with
`docs/PERF-TOOLS.md` instead of ad hoc one-off commands.

### 27B 4K Decode Trace

Using the new helper and parser, a real 27B decode window at `ctx=4096` now has
command-model evidence instead of guesswork:

- direct decode-window `TokenProfile` run (`128` tokens at `ctx=4096`):
  - `avg_total=42.61 ms`, `avg_gpu=42.08 ms`, `avg_cpu_enc=0.24 ms`
  - `med_total=42.69 ms`, `med_gpu=42.14 ms`, `med_cpu_enc=0.20 ms`
  - `p95_total=43.13 ms`, `p95_gpu=42.64 ms`, `p95_cpu_enc=0.27 ms`
  - GPU / total ratio: `~98.7-98.8%`
- Metal trace summary:
  - `128` decode tokens -> `128` command buffers -> `128` encoders
  - encoder duration median: `0.699 ms`, p95 `1.553 ms`
  - submission cadence median: `43.883 ms`, p95 `45.501 ms`
  - previous completion -> next submit median: `0.538 ms`, p95 `0.754 ms`
  - process-scoped compute intervals: `201`
  - process compute total: `2377.224 ms`, process gap total: `4696.175 ms`
  - process gap split:
    - `<= 10 ms`: `76` gaps, `158.166 ms` total
    - `> 10 ms`: `124` gaps, `4538.009 ms` total
  - compute-intervals-per-CB histogram: `1:75, 2:36, 3:15, 4:1, 5:1`

Interpretation:

- Decode is fully serialized token-by-token today.
- The model's own per-token profiler is the decisive source here: dense 27B 4K
  decode is overwhelmingly GPU-busy, not a giant hidden CPU/driver bubble.
- There is still a real but modest host/command-model gap at 4K, not a giant
  hidden bubble.
- The alarming raw process-gap median was a mixed population. Most of the large
  gaps are simply token-to-token cadence; the short intra-CB gaps total only
  about `158 ms / 128 tokens ≈ 1.2 ms/token` at 4K.
- Double-buffered decode submission remains a legitimate low-single-digit decode
  candidate, but not a miracle lever.
- Heavier encoder/fence restructuring should wait for more context-shape traces
  or a stronger kernel-side reason.

### Bench-Only Pipelined Decode Follow-Up

After settling the 4K decode question, I added a dense-only bench harness path:

- `qwen-bench decode-window --pipelined`

This ping-pongs only `ids_buf` and `argmax_tok`, pre-encodes the next token's
command buffer while the current token is running, and keeps it bench-only.

Measured result so far:

- 27B dense at `ctx=4096`, `window=128`
  - serial: `avg_total=43.08 ms`, `med_gpu=42.73 ms`
  - pipelined: `avg_total=42.54 ms`, `med_gpu=42.27 ms`
  - effect: about `0.54 ms/token` on the first A/B, but only `~0.14-0.16 ms`
    (`~0.3%`) across alternating repeats
- 27B dense at `ctx=32768`, `window=64`
  - serial: `avg_total=51.42 ms`, `med_gpu=50.97 ms`
  - pipelined: `avg_total=51.26 ms`, `med_gpu=50.96 ms`
  - effect: again about `~0.16 ms` (`~0.3%`)

Interpretation:

- The branch is real but tiny, exactly in line with the small completion -> next
  submit gap we saw in the trace.
- It is worth keeping behind the bench-only flag for future context checks, but
  it is not a production checkpoint on its own.

## 2026-05-15 — Concurrent GDN Front Projections At 4K

Status: improved checkpoint reached, bench-only / opt-in branch.

### What Changed

- Added a dense-only bench path that splits each GDN block across multiple
  encoders and runs the four independent front projections (`qkv`, `z`, `beta`,
  `alpha`) in a concurrent compute encoder.
- Left attention and the rest of the dense block logic unchanged.
- Exposed the branch through `qwen-bench ctx-sweep --concurrent-gdn-proj`.

### Validation

- Added a dense correctness gate on `Qwen3.5-0.8B.F32.gguf` comparing the new
  path against the serial path:
  - argmax identical
  - `cos = 1.000000`

### 27B 4K Result

Same harness, same context, same window (`ctx-sweep --checkpoints 4096 --window 64`):

- serial:
  - `43.87 ms/token` total
  - `43.23 ms/token` GPU
  - `0.31 ms/token` CPU encode
  - `22.8 t/s`
- concurrent GDN projections:
  - `42.14 ms/token` total
  - `41.53 ms/token` GPU
  - `0.38 ms/token` CPU encode
  - `23.7 t/s`

Interpretation:

- This is a real GPU-side decode win, not a CPU noise artifact.
- The branch improves total decode by about `1.73 ms/token` at 4K, about `4%`
  throughput.
- CPU encode rises slightly, which is fine because the gain is in GPU time.

### Current Next Step

- Keep this as a checkpoint-worthy experimental branch.
- 27B dense at `16K` now confirms the gain survives as attention cost grows:
  - serial: `47.67 ms/token`, `47.04 ms` GPU, `21.0 t/s`
  - concurrent GDN projections: `46.51 ms/token`, `45.94 ms` GPU, `21.5 t/s`
  - effect: about `1.16 ms/token`, roughly `2.5%`

Interpretation:

- The concurrent-GDN branch is not just a 4K-local artifact.
- The gain compresses somewhat as attention grows, but still holds at realistic
  longer context.

## 2026-05-15 — Concurrent GDN + Attention Front Projections

Status: improved checkpoint reached, still bench-only / opt-in.

### What Changed

- Added a second dense-only branch that applies the same concurrent compute
  encoder pattern to attention front projections (`q`, `k`, `v`).
- Exposed it through `qwen-bench ctx-sweep --concurrent-attn-proj`.
- Added support for running both projection-overlap branches together via
  `--concurrent-gdn-proj --concurrent-attn-proj`.

### Validation

- Added dense correctness gates on `Qwen3.5-0.8B.F32.gguf`:
  - concurrent attention vs serial: argmax matches, `cos = 1.000000`
  - concurrent GDN + attention vs serial: argmax matches, `cos = 1.000000`

### Bounded A/B Results

27B dense, `ctx=4096`, `window=64`, same `ctx-sweep` harness:

- serial: `43.64 ms/token`, `43.08 ms` GPU, `22.9 t/s`
- both branches on: `42.31 ms/token`, `41.77 ms` GPU, `23.6 t/s`
- effect: about `1.33 ms/token`, roughly `3.1%`

27B dense, `ctx=16384`, `window=64`, same harness:

- serial: `47.23 ms/token`, `46.65 ms` GPU, `21.2 t/s`
- both branches on: `46.14 ms/token`, `45.61 ms` GPU, `21.7 t/s`
- effect: about `1.09 ms/token`, roughly `2.3%`

Attention-only by itself was smaller:

- `ctx=4096`: `43.89 -> 43.33 ms/token` (`~1.3%`)
- `ctx=16384`: `47.26 -> 47.22 ms/token` (effectively flat)

Interpretation:

- The combined branch is real and positive at both 4K and 16K.
- It is not additive with GDN-only overlap; attention overlap helps at 4K, but
  contributes little by 16K.
- The combined branch is still a stronger overall decode checkpoint than either
  attention-only or pipelined submission.
