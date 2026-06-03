# Performance Log

Append-only checkpoint log for qwen-llm performance work. Use this to answer
"where are we right now?" without re-running audits or reconstructing context
from chat history. Keep entries short, factual, and tied to measurements.

See also: `docs/PERF-ROADMAP.md` for the active force-ranked queue.

## 2026-06-03 — v0.226 Rejected Tiny4 Down Microtile

Status: tried and removed a force-only Q5 down `<8` multi-expert microtile. The
prototype packed four tiny experts into one threadgroup and left `>=8` on the
current grouped down path.

Performance signal before correctness:

| Variant | Q4 `pp512` GPU ms/token rows |
| --- | ---: |
| base | `0.7016 / 0.7068` |
| tiny4 down | `0.6921 / 0.6897` |

Correctness gate failed, so the performance signal is invalid:

- `QWEN_PREFILL_MOE_TINY4_DOWN=1 cargo test -p qwen-llm prefill_tokens_matches_single_token_loop_35b_a3b_moe --release -- --ignored --nocapture`
- First scenario failed with `logits cos=0.981292 < 0.999`.

Read: the broad idea of multi-expert tiny tiling may still be live, but this
implementation is not. The likely bug is tile-layout geometry: one simdgroup per
expert did not reproduce the grouped A-tile fill/store structure correctly. Do
not treat the GPU-time row as a win unless a corrected microtile passes the A3B
prefill-vs-single gate.

## 2026-06-03 — v0.225 Rejected Cold-Packed SwiGLU Proof

Status: tried and removed a force-only cold-packed SwiGLU proof for A3B Q4
`<8` buckets. The branch masked topk slots by expert count on GPU, ran the
existing packed Q4 SwiGLU only for cold slots, and made grouped Q4 skip `0-7`.
Down stayed grouped, so this isolated whether the existing packed direct SwiGLU
kernel could be a cheap cold-slot answer.

Q4 `pp512` A/B:

| Variant | GPU ms/token rows | Read |
| --- | ---: | --- |
| base | `0.7086 / 0.7027` | current grouped path |
| cold-packed SwiGLU | `0.7274 / 0.7380` | regression |

Artifact: `target/profiles/v0224-a3b-cold-direct/q4-pp512-cold-packed-swiglu-sweep.json`.

Read: a masked reuse of the old packed-slot direct kernel is not the flat cold
path. It preserves too much slot-major control/weight traversal overhead even
when restricted to `<8`. The next branch needs either a genuinely new cold-slot
list/direct kernel or a higher-level algorithmic change; do not reopen packed
fallback variants without a new mechanism.

## 2026-06-03 — v0.224 A3B `<8` Tiny-Bucket Split

Status: refined `QWEN_PREFILL_TRACE_MOE_BUCKET_BINS=1` from `<16` to `<8` and
`8-15` bins. This keeps default execution unchanged and makes the tiny-bucket
target specific enough to avoid replaying broad n8 mistakes.

Q4 `pp512` fine-bin evidence:

| Phase | `<8` | `8-15` | `16-31` | `>=64` |
| --- | ---: | ---: | ---: | ---: |
| SwiGLU ms/k-slot | `3.485` | `1.269` | `1.093` | `0.498` |
| Down ms/k-slot | `3.278` | `1.341` | `0.798` | `0.252` |
| Slot fraction | `0.063` | `0.068` | `0.105` | `0.633` |

Negative prototype:

- Tried and removed an env-gated `<8` n8 grouped tile for Q4 SwiGLU plus Q5
  down. Q4 `pp512` GPU ms/token regressed from base `~0.706/0.709` to tiny n8
  `~0.732/0.732`; `pp128` smoke was also slow. This confirms the issue is not
  merely n16 row padding.

Read: `<8` is the true short-prompt monster: it is only `6.3%` of routed slots
but costs `36.1 ms` in SwiGLU and `34.0 ms` in down. A smaller grouped tile still
preserves the bad bucket-granularity execution model. The next exact branch
should be a flat cold-slot/direct-GEMV path for `<8` buckets, ideally with direct
weighted down accumulation once correctness is controlled. Gate on GPU time and
bin ms/k-slot, not noisy t/s alone.

## 2026-06-03 — v0.223 A3B Short-MoE Tiny-Bucket Diagnosis

Status: added disabled MoE bucket-bin trace support under
`QWEN_PREFILL_TRACE_MOE_BUCKET_BINS=1`. The trace splits Q4 grouped SwiGLU and
Q5 grouped down into `<16`, `16-31`, `32-47`, `48-63`, and `>=64` bucket bins
and extends bucket stats with per-bin expert and slot counts. Default execution
is unchanged; Q5/IQ4_XS grouped down now carries full-range min/max args for the
trace split.

Validation:

- `cargo build --release`
- `cargo test -p qwen-llm prefill_tokens_matches_single_token_loop_35b_a3b_moe --release -- --ignored --nocapture`

Q4 `pp512` bin-normalized trace:

| Phase | `<16` | `16-31` | `32-47` | `48-63` | `>=64` |
| --- | ---: | ---: | ---: | ---: | ---: |
| SwiGLU ms/k-slot | `2.276` | `1.307` | `1.133` | `1.310` | `0.549` |
| Down ms/k-slot | `2.104` | `0.870` | `1.096` | `1.326` | `0.282` |
| Slot fraction | `0.131` | `0.105` | `0.075` | `0.056` | `0.633` |

Q4 `pp768` confirms the same shape but with fewer tiny slots: `<16` is `0.092`
of slots, while `>=64` rises to `0.701`. `<16` remains expensive per slot:
SwiGLU `1.951 ms/k-slot`, down `1.982 ms/k-slot`.

Falsifiers:

- Hot-threshold sweeps are tiny/noisy: `th40` is only slightly best at `pp512`,
  tied with default at `pp768`; all-`n32` regresses and all-`n16` loses at the
  winning side of the knee.
- Old packed-routed fallback is not the cold answer: Q4 `pp512` packed is
  `~574 t/s` versus grouped `~1224-1232 t/s`.
- Range-width cap for bounded count ranges looked plausible from dispatch shape
  but failed A/B: Q4 `pp512` cap rows `1236.6/1217.6` versus rollback
  `1234.4/1246.5`; Q4 `pp768` was mixed/noisy. Removed from default code.

Read: the short-MoE miss is an active underfilled-bucket tax, not route work,
not scalar hot threshold policy, not old packed fallback, and not impossible
x-tile early returns. The next exact branch should target `<16` buckets for both
SwiGLU and down while leaving `>=16` on the current grouped path. A candidate
must prove bin-time movement first, then clear Q4 `pp512` end-to-end without
hurting `pp768/1024`, and only then generalize to Q6/Q8/Q3.

## 2026-06-03 — v0.222 Clean A3B Quant Breakpoint Sweep

Status: reran the A3B quant breakpoint sweep after rebuilding from v0.221.
Every qwen row reports `build_commit=a8baa0f6c` and `build_dirty=0`, so the
older stale-build v0.222 packet is superseded.

Clean paired breakpoint evidence:

| Quant | `pp256` | `pp384` | `pp512` | `pp768` | `pp1024` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q3_K_M | `0.883x` | `0.915x` | `0.910x` | `1.048x` | `1.065x` |
| Q4_K_M | `0.805x` | `0.910x` | `0.915x` | `1.111x` | `1.073x` |
| Q6_K | `0.627x` | `0.875x` | `0.883x` | `1.078x` | `1.041x` |
| Q8_0 | `0.646x` | `0.896x` | `0.903x` | `1.102x` | `1.044x` |

Artifact: `target/profiles/v0222-clean-a3b-quant-breakpoint/summary.tsv`.

Read: native grouped coverage across Q3/Q4/Q6/Q8 does generalize, but the
remaining miss is shared across covered quants: A3B loses `pp256-512` and wins
from `pp768` upward. That makes short-prompt grouped routed SwiGLU/down kernel
economics higher EV than another quant-coverage branch. UD-IQ4_XS still needs
IQ3_S gate/up support, but that should not displace the shared `pp512` miss
unless the product priority is specifically that quant.

## 2026-06-03 — v0.221 A3B Q3 Native IQ3 MoE Default

Status: defaulted native IQ3_XXS expert-bank residency and grouped IQ3_XXS
routed gate/up SwiGLU for the observed A3B Q3 shape. This moves A3B Q3_K_M
MoE from slow F32-dequant expert residency to native `40/40` grouped routed
coverage. Rollbacks are `QWEN_MOE_IQ3_EXPERT_NATIVE=0` and
`QWEN_PREFILL_MOE_GROUPED_IQ3_GATEUP=0`.

Correctness and build:

- `cargo test -p qwen-llm moe_grouped_swiglu_iq3_xxs_matches_f32_dequant_fixture -- --ignored --nocapture`
- `cargo test -p qwen-llm prefill_tokens_matches_single_token_loop_0_8b -- --nocapture`
- `cargo build --release`

Dirty A3B Q3 default evidence:

| Shape | qwen | llama.cpp | Ratio | Artifact |
| --- | ---: | ---: | ---: | --- |
| `pp512` | `1241.44` | `1323.36` | `0.938x` | `target/profiles/v0221-dirty-a3b-q3-pp512-default-compare.json` |
| `pp1024` | `1497.39` | `1387.80` | `1.079x` | `target/profiles/v0221-dirty-a3b-q3-pp1024-default-compare.json` |
| `pp4096` | `1542.34` | `1379.67` | `1.118x` | `target/profiles/v0221-dirty-a3b-q3-pp4096-default-compare.json` |

Runtime `pp512` trace confirms grouped coverage and the same short-prompt
signature as Q6/Q8: `routed_swiglu=118.56 ms`, `routed_down=68.38 ms`, and
`route_fused=7.02 ms`.

Read: A3B Q3/Q6/Q8 now all have native `40/40` grouped coverage and win
medium/long paired rows, but all still lose short `pp512`. The remaining local
A3B quant coverage miss is UD-IQ4_XS (`IQ3_S/IQ3_S/IQ4_XS`), which needs a new
IQ3_S gate/up dequant path rather than an auto-policy flip.

## 2026-06-03 — v0.220 A3B Q8 Grouped MoE Coverage

Status: added grouped Q8_0 routed gate/up SwiGLU and grouped Q8_0 routed down
for the observed A3B expert shape. This moves A3B Q8_0 MoE from `0/40` to
`40/40` grouped routed coverage. Rollback is
`QWEN_PREFILL_MOE_GROUPED_Q8_GATEUP=0`, but rollback cannot execute the Q8
routed gate/up path and fails on the old F32-only v1 driver.

Correctness and build:

- `cargo test -p qwen-llm moe_grouped_q8_0_swiglu_down_matches_f32_dequant_fixture -- --ignored --nocapture`
- `cargo test -p qwen-llm prefill_tokens_matches_single_token_loop_0_8b -- --nocapture`
- `cargo build --release`

Dirty A3B Q8 evidence:

| Shape | qwen | llama.cpp | Ratio | Artifact |
| --- | ---: | ---: | ---: | --- |
| `pp512` | `1216.15` | `1339.02` | `0.908x` | `target/profiles/v0220-dirty-a3b-q8-pp512-paired-compare.json` |
| `pp1024` | `1496.77` | `1408.30` | `1.063x` | `target/profiles/v0220-dirty-a3b-q8-pp1024-paired-compare.json` |
| `pp4096` | `1545.07` | `1387.55` | `1.114x` | `target/profiles/v0220-dirty-a3b-q8-pp4096-paired-compare.json` |

Runtime `pp512` trace confirms dynamic grouped coverage and the same short-
prompt signature as Q6: `routed_swiglu=119.51 ms`, `routed_down=63.68 ms`,
`route_fused=7.05 ms`, and attention is not the limiting bucket.

Read: Q8 is now a capability/coverage win with medium/long paired wins, but
short A3B `pp512` remains behind llama.cpp. The remaining quant coverage miss in
the local A3B set is UD-IQ4_XS (`IQ3_S/IQ3_S/IQ4_XS`), while the Q6/Q8 short
miss points to grouped routed SwiGLU/down mechanics rather than routing.

## 2026-06-03 — v0.219 A3B Q6 Grouped MoE Coverage

Status: added grouped Q6_K routed gate/up SwiGLU for the observed A3B Q6
expert shape (`h=2048`, `f_exp=512`, `n_expert=256`). This changes A3B
Q6 from an uncovered MoE quant to `40/40` grouped routed coverage. Rollback
is `QWEN_PREFILL_MOE_GROUPED_Q6_GATEUP=0`, but rollback cannot execute the
Q6 routed gate/up path and fails with the old F32-only v1 driver.

Correctness and build:

- `cargo test -p qwen-llm moe_grouped_swiglu_q6_k_matches_f32_dequant_fixture -- --ignored --nocapture`
- `cargo build --release`

Dirty A3B Q6 evidence:

| Shape | qwen | llama.cpp | Ratio | Artifact |
| --- | ---: | ---: | ---: | --- |
| `pp128` | `553.88` | n/a | capability smoke | `target/profiles/v0219-dirty-a3b-q6-pp128-grouped-gateup-smoke.json` |
| `pp512` | `1182.07` | `1269.16` | `0.931x` | `target/profiles/v0219-dirty-a3b-q6-pp512-paired-compare.json` |
| `pp1024` | `1416.97` | `1331.88` | `1.064x` | `target/profiles/v0219-dirty-a3b-q6-pp1024-paired-compare.json` |
| `pp4096` | `1472.49` | `1323.35` | `1.113x` | `target/profiles/v0219-dirty-a3b-q6-pp4096-paired-compare.json` |

Runtime `pp512` trace confirms dynamic grouped coverage: `routed_swiglu=40`,
`routed_down=40`, `route_fused=40`, and `shared_packed=40`. The same audit still
flags A3B Q8_0 MoE as `0/40`, so Q8 grouped gate/up remains the next quant-
coverage gap if the goal is fastest across all shipped quants.

Read: this is a high-value coverage/capability fix, not a finished Q6 speed
story. The `pp512` row is still behind llama.cpp and the phase trace books
`routed_swiglu` as the largest bucket, so Q6 grouped SwiGLU mechanics or Q8
coverage are better next moves than another small dense policy retread.

## 2026-06-03 — v0.218 Rejected Q4 N64 Shape Policy

Status: tried and reverted a default policy that disabled Q4_K N64 only for
`n_query < 1024 && n_out <= 1536`. The idea came from a clean retread where
global `QWEN_MATMAT_Q4_K_N64=0` looked positive on 0.8B/2B `pp512`, but the
narrow mixed-kernel policy did not reproduce that win.

Dirty 0.8B `pp512` gates after the policy edit:

| Variant | Rows | Artifact |
| --- | ---: | --- |
| default policy | `6858.60 / 6776.82` | `target/profiles/v0218-dirty-08b-pp512-q4-n64-policy-sweep.json` |
| force N64 | `6852.08 / 6859.29` | `target/profiles/v0218-dirty-08b-pp512-q4-n64-policy-sweep.json` |
| default policy | `6576.90 / 6794.06` | `target/profiles/v0218-dirty-08b-pp512-q4-n64-policy-triage.json` |
| force N64 | `6854.26 / 6824.99` | `target/profiles/v0218-dirty-08b-pp512-q4-n64-policy-triage.json` |
| global off | `6820.93 / 6888.30` | `target/profiles/v0218-dirty-08b-pp512-q4-n64-policy-triage.json` |

Read: the shape-local rule created a worse mixed Q4 kernel packet than either
forcing N64 or disabling N64 globally in the short retest. Do not default this
policy. Reopen Q4 N64 policy only with per-op dispatch attribution or a broader
family packet that distinguishes global-off, force-on, and mixed routing.

## 2026-06-03 — v0.217 Rejected Fused Q4 SwiGLU N64

Status: tried and removed an env-only fused Q4 SwiGLU N64 sidecar for the
small-dense FFN bucket. The kernel was correctness-safe, but the larger fused
epilogue/staging path regressed 0.8B `pp512`, so it is not a keep candidate.

Correctness:

- `QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4_N64=1 cargo test -p qwen-llm ffn_fused_swiglu_q4_K_mm_n16_matches_unfused -- --nocapture`
- `QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4_N64=1 cargo test -p qwen-llm prefill_tokens_matches_single_token_loop_0_8b -- --nocapture`

Warmed 0.8B `pp512` A/B:

| Variant | Rows | Artifact |
| --- | ---: | --- |
| base | `7446.73 / 7480.24 / 7468.51` | `target/profiles/v0216-dirty-08b-pp512-fused-swiglu-n64-sweep.json` |
| fused N64 | `7271.48 / 7288.58 / 7281.37` | `target/profiles/v0216-dirty-08b-pp512-fused-swiglu-n64-sweep.json` |

Read: the theoretical reuse from a 64-column fused tile is outweighed by the
larger fused epilogue/staging cost. Do not reopen fused Q4 N64 unless the
epilogue becomes direct-store or otherwise avoids the 64x64 gate/up staging
wall.

## 2026-06-03 — v0.215 GDN Paired Q/K L2 Prep

Status: promoted paired GDN Q/K L2 normalization in prompt prefill. The new
`kernel_l2_norm_pair_batched_f32` replaces the two per-GDN-layer Q and K L2
dispatches with one two-plane dispatch. Rollback is
`QWEN_PREFILL_GDN_PAIR_L2=0`.

Correctness and build:

- `cargo test -p qwen-llm l2_norm_pair_batched_matches_cpu -- --nocapture`
- `cargo test -p qwen-llm prefill_tokens_matches_single_token_loop_0_8b -- --nocapture`
- `cargo build --release`

Promotion A/B rows before checkpoint, comparing default against rollback:

| Model | Shape | default | rollback | Read | Artifact |
| --- | ---: | ---: | ---: | --- | --- |
| 0.8B | `pp512` | `7418.45 / 7439.47 / 7454.13` | `7276.97 / 7266.45 / 7268.90` | `+2.2-2.5%` | `target/profiles/v0215-dirty-08b-pp512-gdn-pair-l2-rollback-sweep.json` |
| 0.8B | `pp1024` | `7780.46 / 7776.14` | `7571.37 / 7578.19` | `+2.6-2.8%` | `target/profiles/v0215-dirty-08b-pp1024-gdn-pair-l2-rollback-sweep.json` |
| 2B | `pp512` | `3541.88 / 3516.46` | `3511.26 / 3488.60` | `+0.8-0.9%` | `target/profiles/v0215-dirty-2b-pp512-gdn-pair-l2-rollback-sweep.json` |
| 2B | `pp1024` | `3719.72 / 3716.41` | `3678.05 / 3670.50` | `+1.1-1.3%` | `target/profiles/v0215-dirty-2b-pp1024-gdn-pair-l2-rollback-sweep.json` |

Larger dense canaries were neutral-to-positive: 4B `pp512` moved
`1462.26/1458.22 -> 1467.68/1466.48`, 9B `pp512` moved
`808.13/806.93 -> 809.93/808.68`, and 27B `pp512` moved
`237.03/237.25 -> 237.68/238.71`.

Clean paired pinned-b9481 rows after v0.215:

| Model | Shape | qwen | llama.cpp | Ratio | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.8B | `pp512` | `7444.06 / 7417.89` | `7711.82 / 7865.60` | `0.965x / 0.943x` | `target/profiles/v0215-clean-08b-pp512-pair-l2-compare.json` |
| 0.8B | `pp1024` | `7772.91 / 7772.88` | `7748.63 / 7776.73` | `1.003x / 1.000x` | `target/profiles/v0215-clean-08b-pp1024-pair-l2-compare.json` |
| 2B | `pp512` | `3532.36 / 3534.02` | `3629.93 / 3653.37` | `0.973x / 0.967x` | `target/profiles/v0215-clean-2b-pp512-pair-l2-compare.json` |
| 2B | `pp1024` | `3712.66 / 3705.86` | `3673.19 / 3668.64` | `1.011x / 1.010x` | `target/profiles/v0215-clean-2b-pp1024-pair-l2-compare.json` |

Falsifiers in the same sprint:

- GDN packed-step NSG8 row grouping was correctness-safe but flat/regressive:
  0.8B `pp512` `nsg8` was `6596.58/6644.91/6596.93` versus base
  `6652.14/6617.27/6653.12`, and the traced `gdn_step` bucket did not drop.
- Lowering the existing Q5/Q6 N64 threshold to `512` is not the missing Q5 GDN
  projection fix. Warmed 0.8B `pp512` showed Q5 flat, Q5+Q6 flat, and Q6-only
  slower in `target/profiles/v0215-clean-08b-pp512-q5q6-split-warmed.json`.

Read: this is a real small-dense cleanup with a direct GDN-prep mechanism, not
the whole lcpp gap. It closes several percent on 0.8B short/medium and gives a
smaller positive guardrail on 2B, while larger dense shapes stay safe.

## 2026-06-02 — v0.212 Small-Dense Fused Q4 SwiGLU Gate

Status: promoted dense Q4 fused SwiGLU only for small hidden states
(`hidden <= 1536`). This narrows the 0.8B short/medium dense gap without taking
the known regressions on 4B/9B/27B. Rollback/force env remains
`QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=0|1`.

Correctness and build:

- `cargo test -p qwen-llm ffn_fused_swiglu_q4_K_mm_n16_matches_unfused -- --nocapture`
- `cargo test -p qwen-llm prefill_tokens_matches_single_token_loop_0_8b -- --nocapture`
- `cargo build --release`

Clean qwen-only rollback rows after v0.212:

| Model | Shape | default | rollback | Ratio | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.8B | `pp512` | `7253.98` | `7150.23` | `1.015x` | `target/profiles/v0212-clean-08b-pp512-shape-fused-rollback-sweep.json` |
| 0.8B | `pp1024` | `7601.30` | `7451.03` | `1.020x` | `target/profiles/v0212-clean-08b-pp1024-shape-fused-rollback-sweep.json` |

Clean paired pinned-b9481 rows after v0.212:

| Model | Shape | qwen | llama.cpp | Ratio | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.8B | `pp512` | `7254.74 / 7182.30` | `7874.91 / 7878.64` | `0.921x / 0.912x` | `target/profiles/v0212-clean-08b-pp512-shape-fused-compare.json` |
| 0.8B | `pp1024` | `7599.28 / 7593.34` | `7783.26 / 7836.44` | `0.976x / 0.969x` | `target/profiles/v0212-clean-08b-pp1024-shape-fused-compare.json` |
| 2B | `pp512` | `3482.97 / 3492.48` | `3626.60 / 3633.43` | `0.960x / 0.961x` | `target/profiles/v0213-clean-2b-pp512-shape-fused-compare.json` |
| 2B | `pp1024` | `3670.52 / 3665.02` | `3676.17 / 3672.49` | `0.998x / 0.998x` | `target/profiles/v0213-clean-2b-pp1024-shape-fused-compare.json` |

Canary read:

- Forced fused Q4 SwiGLU is not broadly safe: dirty canaries were thin-positive on
  2B, but negative on 4B, 9B, and 27B at `pp512`.
- The shape gate is intentionally small-dense only. Dirty default-vs-rollback rows
  were neutral for 4B and 27B because the fused path does not select there.
- The GDN tail encoder coalescing sidecar passed the 0.8B prefill-vs-single gate
  but was flat/noise at `pp512` and `pp1024`; do not keep or reopen encoder-
  coalescing as the next small-dense lever without new evidence.
- Streaming layer command buffers (`QWEN_PREFILL_STREAM_LAYERS`) was also tested as
  a llama.cpp-style CPU/GPU overlap sidecar. `stream12` was at most sub-1% positive
  on 0.8B and flat on 2B `pp512`, so it was not kept.

Read: this is a useful incremental scoreboard cleanup, not the lcpp-cracking
branch. 0.8B `pp1024` is now close to parity, but `pp512` is still materially
behind. No-op/split isolates still point at real FFN and GDN body work rather
than attention or command-encoder overhead.

## 2026-06-02 — v0.208 Q5/Q6 Long-Prompt N64 Tiles

Status: added large-N mat-mat tiles for Q5_K and Q6_K projections, with the
default gate limited to `n_query >= 1024`. This is not a 0.8B `pp512` fix; the
unrestricted `N64` tile regressed that short-prompt cell.

Why this branch was worth trying: the small-dense dtype audit showed the largest
0.8B GDN projection bucket, `gdn_qkv`, is Q5_K, `gdn_back` is Q5_K, and half of
`ffn_down` is Q6_K. Q4_K already had an N64 prompt tile, but Q5_K/Q6_K were still
using the generic N32 tile at long prompt chunk sizes.

Correctness:

- `cargo test -p qwen-llm mat_mat_q5_k_matches_cpu_and_mat_vec`
- `cargo test -p qwen-llm mat_mat_q6_k_matches_cpu_and_mat_vec`
- `cargo build --release`

Clean qwen-only A/B rows compare default `min1024` against rollback
`QWEN_MATMAT_Q5_K_N64=0,QWEN_MATMAT_Q6_K_N64=0`:

| Model | Shape | default | rollback | Ratio | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.8B | `pp512` | `6536.60` | `6611.10` | `0.989x` | guard row/noisy; N64 should not select |
| 0.8B | `pp4096` | `7135.04` | `7098.44` | `1.005x` | thin/mixed |
| 2B | `pp1024` | `3482.96` | `3458.00` | `1.007x` | thin positive |
| 4B | `pp1024` | `1443.94` | `1418.41` | `1.018x` | clear positive |
| 9B | `pp1024` | `800.66` | `791.05` | `1.012x` | clear positive |
| 27B | `pp1024` | `229.55` | `222.18` | `1.033x` | positive but noisy |

Follow-up pp512 threshold isolate (`default`, `MIN_N=2048`, rollback) did not
show a stable dispatch-specific regression, so treat the first pp512 guard as
noise/order until a larger packet says otherwise. The important policy result is
that the branch is explicitly not allowed below `N=1024` by default.

Artifacts:

- `target/profiles/v0208-clean-08b-pp512-q5q6-n64-min1024.json`
- `target/profiles/v0208-clean-08b-pp512-q5q6-n64-min1024-repeat.json`
- `target/profiles/v0208-clean-08b-pp512-q5q6-n64-min-threshold-isolate.json`
- `target/profiles/v0208-clean-08b-pp4096-q5q6-n64-min1024.json`
- `target/profiles/v0208-clean-2b-pp1024-q5q6-n64-min1024.json`
- `target/profiles/v0208-clean-4b-pp1024-q5q6-n64-min1024.json`
- `target/profiles/v0208-clean-9b-pp1024-q5q6-n64-min1024.json`
- `target/profiles/v0208-clean-27b-pp1024-q5q6-n64-min1024.json`

Read: keep the long-prompt Q5/Q6 N64 policy. It is incremental, not the missing
small-dense `pp512` breakthrough. The next small-dense branch still needs a
structural fix for short-prompt projection/GDN execution, not another low-threshold
N64 expansion.

Narrow pinned-b9481 scoreboard follow-up at exact `pp1024`:

| Model | qwen | llama.cpp | Ratio | Artifact |
| --- | ---: | ---: | ---: | --- |
| 0.8B | `7464` | `7798` | `0.96x` | `docs/bench/2026-06-02-2151-0.8B-v0209-q5q6-n64-pp1024-family/` |
| 2B | `3674` | `3679` | `1.00x` | `docs/bench/2026-06-02-2151-2B-v0209-q5q6-n64-pp1024-family/` |
| 4B | `1483` | `1451` | `1.02x` | `docs/bench/2026-06-02-2151-4B-v0209-q5q6-n64-pp1024-family/` |
| 9B | `803` | `804` | `1.00x` | `docs/bench/2026-06-02-2151-9B-v0209-q5q6-n64-pp1024-family/` |

Read: v0.208 closes 2B/9B `pp1024` to parity and keeps 4B ahead, but 0.8B
remains the clean small-dense `pp1024` miss. The next small-dense sprint should
use 0.8B as the primary target rather than celebrating family-average parity.

## 2026-06-02 — v0.207 Rejected Small-Hidden N64 Auto Policy

Status: tried and rejected a narrow Q4_K N64 auto-policy after v0.206. The idea
was to disable the N64 mat-mat tile only for small-hidden `N=512` chunks while
leaving 4B/9B/27B untouched.

Validation did not reproduce strongly enough to default:

- 0.8B `pp512` auto-vs-force-N64 was mixed (`7224/7251`, then `7289/7246`).
- 2B `pp512` was only mildly positive (`3534/3516`, `3530/3521`).
- 4B/9B canaries were effectively noise and should have been unchanged by the
  policy.

Artifact packet:

- `target/profiles/v0207-08b-pp512-n64-auto-policy.json`
- `target/profiles/v0207-2b-pp512-n64-auto-policy.json`
- `target/profiles/v0207-4b-pp512-n64-auto-policy.json`
- `target/profiles/v0207-9b-pp512-n64-auto-policy.json`

Read: keep `QWEN_MATMAT_Q4_K_N64` default unchanged. The small dense branch needs
a stronger structural kernel/policy win than conditional rollback of an existing
tile.

## 2026-06-02 — v0.206 Small Dense Short-Prefill Triage

Status: investigated the pinned-b9481 0.8B/2B short-prefill gap with cheap
qwen-only attribution. No default change.

0.8B `pp512` budget sweep:

- Base: `~7220 t/s`; pinned lcpp family row is `7784.60 t/s`.
- `QWEN_PREFILL_ATTN_MATRIX_G4=0`: collapses to `~4000 t/s`; matrix attention is
  necessary, not the gap.
- `QWEN_PREFILL_NOOP_ATTN_BODY=1`: only `~7620 t/s`; removing attention still
  does not catch lcpp.
- `QWEN_PREFILL_NOOP_GDN_BODY=1`: `~10500 t/s`; GDN is a large budget bucket.
- `QWEN_PREFILL_NOOP_FFN=1`: `~11380 t/s`; dense FFN is also a large bucket.

Phase trace read: attention matrix body is already tiny (`~3.6 ms` total). Top
0.8B `pp512` buckets are GDN projections/state (`gdn_qkv`, `gdn_step`,
`gdn_prep`, `gdn_back`, `gdn_z`) plus FFN `gate/up/down` projections. GDN matvec
fallbacks were catastrophic (`z` `~4700`, `qkv` `~3000`, `front` `~1500 t/s`), so
the answer is not reverting to matvec.

Existing positive toggles are real but too thin or too narrow:

- In-process `pp-ffn-ab`: fused Q4 SwiGLU was `1.012x/1.014x` median on 0.8B
  `pp512/pp1024` and `1.0085x` median on 2B `pp512`.
- `QWEN_MATMAT_Q4_K_N64=0`: `+0.7-1.0%` on 0.8B/2B `pp512`, but not safe as a
  broad policy.
- Combined N64-off + fused FFN: positive on 0.8B `pp512/1024/4096` and 2B
  `pp512`, but negative on 2B `pp1024`, 4B `pp1024`, and 9B `pp1024`.

Artifacts:

- `target/profiles/v0205-08b-pp512-budget-sweep.json`
- `target/profiles/v0205-08b-pp512-phase-summary.tsv`
- `target/profiles/v0205-08b-pp512-gdn-matvec-falsifiers.json`
- `target/profiles/v0206-08b-pp512-ffn-ab.tsv`
- `target/profiles/v0206-08b-pp1024-ffn-ab.tsv`
- `target/profiles/v0206-2b-pp512-ffn-ab.tsv`
- `target/profiles/v0206-08b-pp512-q4-n64-falsifier.json`
- `target/profiles/v0206-2b-pp512-q4-n64-falsifier.json`
- `target/profiles/v0206-08b-pp512-combo-falsifier.json`
- `target/profiles/v0206-2b-pp512-combo-falsifier.json`
- `target/profiles/v0206-4b-pp1024-combo-canary.json`
- `target/profiles/v0206-9b-pp1024-combo-canary.json`

Read: small dense short-prefill is a small-N projection execution problem, not a
missing fast-path or attention problem. The next real branch should be structural
and targeted: small-hidden Q4_K mat-mat/tile policy and/or fused projection work
for GDN/FFN, with 0.8B/2B failing cells as the target and 4B/9B/27B as canaries.

## 2026-06-02 — v0.205 Exact Family Shape Selection

Status: fixed a harness footgun found during the pinned-b9481 re-anchor. Before
this change, `scripts/bench/family.py --shapes tg128` still ran default prompt
shapes because unspecified pp/tg sides were filled from defaults. That caused the
A10B tg repeat to run `pp128/512/1024` too.

New behavior: `--shapes` is exact. No `--shapes` still uses the default family
grid, but `--shapes tg128` runs only `tg128`, and `--shapes pp512,tg128` runs
exactly those two cells.

## 2026-06-02 — v0.204 A10B pp128 G16 Threshold Cleanup

Status: attacked the fresh pinned-b9481 A10B `pp128` miss without another broad
family sweep. The culprit is a threshold cliff: default `pp128` skipped the G16
matrix attention path because `QWEN_PREFILL_ATTN_PACKED_G16_MIN_POS` defaulted to
`320`.

Falsifier packet, qwen-only variants at A10B `pp128`:

| Variant | Block 0 | Block 1 | Read |
| --- | ---: | ---: | --- |
| base | `223.30` | `219.55` | current miss |
| `G16_MIN_POS=1` | `242.24` | `244.88` | `+9-12%` |
| route fused | `221.95` | `222.53` | flat/slower |
| both | `245.69` | `246.29` | attention threshold is the win |

Trace attribution: default `pp128` booked attention body at `109.96 ms` across 12
attention layers. Forced G16 matrix booked `body_matrix_kq + softmax + kqv` at
`2.17 ms` total. The new default lowers the G16 threshold to `128`; a dirty
post-change smoke produced `242.38 t/s` average with steady samples at
`282.42/283.41 t/s`.

Artifacts:

- `target/profiles/v0204-a10b-pp128-threshold-falsifiers.json`
- `target/profiles/v0204-a10b-pp128-default-trace-summary.tsv`
- `target/profiles/v0204-a10b-pp128-g16min1-trace-summary.tsv`
- `target/profiles/v0204-a10b-pp128-g16min128-default-smoke.json`

Read: this does not yet prove A10B `pp128` is won in the cold averaged family
semantics, because the pinned lcpp row is `260.27 t/s` and qwen's first measured
sample is still cold-ish. It does remove the obvious threshold cliff and should
be treated as a very-short prompt cleanup with rollback via
`QWEN_PREFILL_ATTN_PACKED_G16_MIN_POS=320`.

## 2026-06-02 — v0.203 Pinned b9481 Family Re-Anchor

Status: rebuilt `qwen-bench` at `fa941aad7` and ran the full synthetic family
sweep against the pinned llama.cpp b9481 comparator. AC power, no recorded
thermal/performance warnings, memory free stayed `87-96%`.

Artifact: `docs/bench/2026-06-02-1855-v0203-b9481-family-family/`.

| Model | pp512 | pp1024 | pp4096 | pp16384 | tg128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B dense | `0.925x` | `0.950x` | `0.957x` | `0.988x` | `1.228x` |
| 2B dense | `0.954x` | `0.985x` | `0.987x` | `1.029x` | `1.048x` |
| 4B dense | `1.005x` | `1.009x` | `1.111x` | `1.085x` | `1.435x` |
| 9B dense | `1.000x` | `1.008x` | `1.056x` | `0.986x` | `1.113x` |
| 27B dense | `0.913x` | `1.092x` | `1.049x` | `1.057x` | `1.222x` |
| 35B A3B MoE | `1.024x` | `1.148x` | `1.167x` | `1.062x` | `1.063x` |
| 122B A10B MoE | `1.018x` | `1.171x` | `1.161x` | `1.096x` | `0.993x` |

Follow-up repeats for fragile cells:

- 27B `pp512` paired repeat: `1.000x` and `1.008x`, so the full-family
  `0.913x` row is likely drift/order noise, not a confirmed regression.
  Artifact: `target/profiles/v0203-27b-pp512-b9481-repeat.json`.
- A10B `tg128` repeat: `35.76/35.21 t/s` (`1.016x`). The same narrow run also
  exposed A10B `pp128` at `0.852x`, which is outside the primary long-prompt
  guardrails but relevant to the across-board goal.
  Artifact: `docs/bench/2026-06-02-1953-122B-A10B-v0203-b9481-tg-repeat-family/`.

Read: latest llama.cpp does not erase the MoE prefill wins; A3B and A10B are
still ahead across `pp512+` in the fresh pinned sweep. Remaining broad gaps are
small dense short/medium prefill, 9B/0.8B long parity cleanup, and A10B very-short
`pp128`. Treat 27B `pp512` as parity pending more repeats, not as an active
structural miss.

## 2026-06-02 — v0.202 Pinned llama.cpp Benchmark Target

Status: moved qwen-vs-llama.cpp scripts off the ambient
`~/code/llama.cpp/build/bin` comparator and onto a repo-pinned benchmark target.

- Lock: `scripts/bench/llama-cpp.lock.json` pins upstream llama.cpp tag `b9481`
  / commit `bfb4308b058b334c6e68085c661ec9eb7e3d59f4`.
- Builder: `scripts/bench/ensure_llama_cpp.py` uses a shared cache under
  `~/.cache/qwen-llm/llama.cpp`, a bare mirror plus one worktree per pinned SHA,
  and minimal CMake args: only `-DCMAKE_BUILD_TYPE=Release`.
- Script defaults: `scripts/bench/family.py` and
  `scripts/profile/prefill_compare.py` now resolve the locked cache by default,
  with explicit path/env overrides still available. They reject build-commit or
  backend mismatches unless `--allow-unpinned-lcpp` is passed.
- Smoke: built the pinned target, `llama-cli --version` reports `9481
  (bfb4308b05)`, and a tiny locked comparator run completed at
  `target/profiles/v0202-lcpp-lock-prefill-compare-smoke.json`.

Read: the local fork can stay useful for source study or runtime-seam work, but
it is no longer the scoreboard comparator. The next scoreboard action is a full
family re-anchor against pinned b9481 before making fresh qwen/lcpp claims.

## 2026-06-02 — v0.201 A3B Q3 Real-Prompt Promotion Packet

Status: ran the first repeated real-prompt promotion packet for the A3B Q3
native-IQ3 candidate through `scripts/profile/prefill_compare.py`, not the Rust
test harness. No Q3 default yet, but the script-led evidence is stronger.

Rows use Qwen3.5 strip prompts, AC power, `build_dirty=0`, qwen build
`f3de886a6`, and no qwen thermal/performance warnings. llama.cpp is still a
same-length synthetic anchor because `llama-bench` cannot consume prompt text.

| Prompt | Tokens | Retained qwen/lcpp pairs | Artifact |
| --- | ---: | ---: | --- |
| Reva short strip | `7344` | `1.055x`, `1.025x` after discarding block 0 | `target/profiles/v0201-a3b-q3-real-reva-short-strip-promotion.json` |
| Marcus long strip | `25610` | `1.047x` after discarding block 0 | `target/profiles/v0201-a3b-q3-real-marcus-long-strip-promotion.json` |

Read: the native-IQ3 + blk0 `qkv+alpha` repair remains positive on real prompt
content when run through the cooled, repeated script path. This answers the
workflow concern: use tests for in-process packed-vs-single oracle state, but use
`prefill_compare.py` for promotion-grade throughput. The remaining defaultability
question is policy/correctness: top-k/rank envelope versus exact long greedy
parity, not whether the perf win only exists in tests.

## 2026-06-02 — v0.200 Family Re-Anchor And Sweep Telemetry

Status: returned to script-led performance characterization after the A3B
correctness-gate work. `scripts/bench/family.py` now supports
`--cooldown-seconds` and records per-command thermal/memory/outer-wall metadata
in `manifest.json` so future family sweeps do not rely on operator memory for
power/thermal context.

Clean synthetic family rows on commit `ea2ae8d9a`, AC power, qwen rows with no
thermal/performance warnings:

| Model | Shape | qwen | llama.cpp | Ratio | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| 27B dense | `pp512` | `239.48` | `235.54` | `1.017x` | `docs/bench/2026-06-02-1518-27B-v0200-long-family/` |
| 27B dense | `pp1024` | `237.40` | `220.64` | `1.076x` | same |
| 27B dense | `pp4096` | `220.44` | `199.49` | `1.105x` | same |
| 27B dense | `pp16384` | `207.36` | `198.45` | `1.045x` | same |
| 27B dense | `tg128` | `24.35` | `19.41` | `1.254x` | same |
| 35B A3B | `pp512` | `1438.41` | `1374.55` | `1.046x` | `docs/bench/2026-06-02-1502-35B-A3B-v0200-long-family/` |
| 35B A3B | `pp1024` | `1612.97` | `1385.43` | `1.164x` | same |
| 35B A3B | `pp4096` | `1516.85` | `1336.47` | `1.135x` | same |
| 35B A3B | `pp16384` | `1231.05` | `1073.76` | `1.146x` | same |
| 35B A3B | `tg128` | `80.28` | `74.80` | `1.073x` | same |
| 122B A10B | `pp512` | `438.16` | `438.03` | `1.000x` | `docs/bench/2026-06-02-1505-122B-A10B-v0200-long-family/` |
| 122B A10B | `pp1024` | `500.11` | `433.51` | `1.154x` | same |
| 122B A10B | `pp4096` | `485.06` | `383.00` | `1.266x` | same |
| 122B A10B | `pp16384` | `406.21` | `341.45` | `1.190x` | same |
| 122B A10B | `tg128` | `35.30` | `34.38` | `1.027x` | same |

Read: the current committed default is now won across the measured primary
synthetic guardrails. The fragile cells are the narrow wins at A10B `pp512` and
27B `pp512/pp16384`; use `prefill_compare.py` repeat blocks for promotion-grade
claims there. The tests were used for in-process oracle correctness because they
can inspect GDN/KV/logit state; throughput and cross-engine claims should remain
script-led, with `prefill_compare.py` as the promotion harness and `family.py` as
the breadth scoreboard.

## 2026-06-02 — v0.199 A3B Q3 Real-Prompt Gate Packet

Status: ran the first compact real-prompt packet for the A3B Q3 native-IQ3
candidate and replaced long exact-argmax gating with rank/margin diagnostics.
No Q3 default yet.

New A3B correctness knobs:

- `QWEN_A3B_MOE_TEST_PROMPT_FILE=path`: use a rendered real prompt instead of
  synthetic token ids.
- `QWEN_A3B_MOE_TEST_LOGITS_COS_MIN=X`: optionally force final-logit cosine on
  long probes.
- `QWEN_A3B_MOE_TEST_REQUIRE_ARGMAX_MATCH=0|1`: override the default exact
  continuation-argmax policy; default is strict only for prompts `<=128` tokens.
- `QWEN_A3B_MOE_TEST_RANK_ESCAPE_MAX=N`: optionally gate continuation mismatches
  by cross-rank instead of exact argmax equality.

Real-prompt prefill evidence, AC power, sequential GPU runs. llama.cpp rows are
same-length synthetic anchors from `llama-bench`, not same-token-content prompt
runs:

| Prompt | Tokens | qwen | llama.cpp | Ratio | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| Reva short, strip | `7344` | `1333.62` | `1310.33` | `1.018x` | `target/profiles/v0199-a3b-q3-real-reva-short-strip-compare.json` |
| Mei medium, strip | `11287` | `1262.46` | `1245.08` | `1.014x` | `target/profiles/v0199-a3b-q3-real-mei-medium-strip-compare.json` |
| Marcus long, strip | `25610` | `1042.53` | `1019.71` | `1.022x` | `target/profiles/v0199-a3b-q3-real-marcus-long-strip-compare.json` |

Continuation/rank diagnostics:

| Probe | Result | Artifact |
| --- | ---: | --- |
| Q3 native IQ3 + blk0 `qkv+alpha`, Reva strip T128/P32 | pass; final logits `0.999755`, no argmax mismatch, continuation `cos_min=0.989769` | `target/profiles/v0199-a3b-q3-real-reva-strip-T128-P32-cont64.out` |
| Q3 native IQ3 + blk0 `qkv+alpha`, Reva strip T1024/P128 | rank-diagnostic pass; final logits `0.997346`, first mismatch step `27`, cross-ranks `2/2`, margins `0.2505/0.0035` | `target/profiles/v0200-a3b-q3-real-reva-strip-T1024-P128-rankdiag.out` |
| Q4 default, Reva preserve T1024/P128 | rank-diagnostic pass; final logits `0.994864`, first mismatch step `12`, cross-ranks `3/2`, margins `0.0448/0.4410` | `target/profiles/v0200-a3b-q4-real-reva-preserve-T1024-P128-rankdiag.out` |
| Q4 default, synthetic T1024/P128 | rank-diagnostic pass; final logits `0.999931`, first mismatch step `30`, cross-ranks `2/2`, margins `0.1691/0.0177` | `target/profiles/v0200-a3b-q4-synth-T1024-P128-rankdiag.out` |

Negative: a Q3 native-off/default same-prompt T1024 oracle did not reach the
prefill comparison within a 30-minute cap after dequanting IQ3 expert banks to
F32. Treat native-off long oracle comparison as impractical unless scoped much
smaller. Artifact: `target/profiles/v0200-a3b-q3-default-real-reva-strip-T1024-P128-rankdiag.out`.

Read: exact zero argmax divergence is the wrong long-prompt gate because Q4
default misses it too, including on synthetic T1024 with excellent final-logit
cosine. The useful signal is whether mismatches are high-rank/high-margin escapes
or near-top alternatives under teacher-forced oracle history. The Q3 native-IQ3
candidate remains promising on real prompt speed and its first long mismatch is a
rank-2/rank-2 flip, but defaulting still needs a same-prompt policy decision and
possibly a top-k/rank-envelope gate rather than strict internal cosine or exact
greedy parity.

## 2026-06-02 — v0.198 Real-Prompt Compare Harness

Status: extended `scripts/profile/prefill_compare.py` beyond synthetic `pp<N>`
inputs. It now accepts `--prompt`, `--file`, and `--messages` with the same
thinking-mode controls as `prefill_sweep.py`/`qwen-bench pp`.

Important methodology note: `llama-bench` still cannot consume real prompt text;
for real-prompt sources, the harness renders/counts the prompt with
`llama-tokenize --no-bos` and runs llama.cpp as a synthetic `pp<N>` length anchor.
The output records `lcpp_prompt_mode=synthetic_length_anchor` so these rows are
not overclaimed as same-token-content llama.cpp prompt runs.

Smoke evidence:

| Gate | Result | Artifact |
| --- | ---: | --- |
| `--file current-reva-short-qwen36-preserve.txt`, 0.8B Q4 | token count `7986`; harness completes and checkpoints | `target/profiles/v0198-prefill-compare-file-smoke.json` |
| `--messages current-reva-short-qwen36.json --messages-max 2`, 0.8B Q4 | token count `6482`; harness completes and checkpoints | `target/profiles/v0198-prefill-compare-messages-smoke.json` |

Read: this unblocks compact real-rollout qwen measurements with a same-length
llama.cpp anchor and preserves the strict no-concurrent-GPU workflow. It is still
not a substitute for a true llama.cpp real-prompt benchmark if we later repair or
replace the broken `llama-cli`/`llama-perplexity` binaries.

## 2026-06-02 — v0.197 A3B Continuation Gate

Status: replaced the A3B long continuation cosine gate with a generation-oriented
argmax gate for multi-token probes. No Q3 default yet; this is a correctness-gate
cleanup after the v0.196 matrix-VT fix.

New/changed A3B correctness knobs:

- `QWEN_A3B_MOE_TEST_CONT_TOKENS=N`: run an oracle-greedy continuation after the
  prefill-vs-single comparison.
- `QWEN_A3B_MOE_TEST_CONT_COS_MIN=X`: optionally force a strict continuation-logit
  cosine threshold for long probes.
- `QWEN_A3B_MOE_TEST_INTERNAL_COS_MIN=X`: optionally force strict internal GDN/KV
  cosine for long probes.
- Default policy: one-step probes keep the old `0.999` continuation cosine gate;
  prompts up to 64 tokens keep the old `0.999` internal GDN/KV gate; longer
  multi-token probes always require no argmax divergence, while continuation and
  internal cosine are tracked as diagnostics unless explicit thresholds are
  supplied.

Continuation evidence, AC power, sequential GPU runs:

| Gate | Result | Artifact |
| --- | ---: | --- |
| Q3 native IQ3, blk0 `qkv+alpha`, T128/P32, 64-token continuation | pass; final logits `0.999986`, continuation `cos_min=0.996345`, `worst_step=13`, no argmax mismatch | `target/profiles/v0197-a3b-q3-native-iq3-T128-P32-qkv-alpha-layer0-cont64-softcos.out` |
| Q3 native IQ3, blk0 `qkv+alpha`, T112/P112, 16-token diagnostic | final logits `0.999964`, continuation `cos_min=0.999175`, no argmax mismatch; strict GDN state/conv still fail | `target/profiles/v0197-a3b-q3-native-iq3-T112-P112-qkv-alpha-layer0-cont16.out` |
| Q3 native IQ3, blk0 `qkv+alpha`, T112/P112, 64-token diagnostic | pass with soft long gates; final logits `0.999964`, continuation `cos_min=0.946914`, `worst_step=38`, no argmax mismatch | `target/profiles/v0197-a3b-q3-native-iq3-T112-P112-qkv-alpha-layer0-cont64-softcos.out` |
| Q4 default, T128/P32, 64-token diagnostic | pass with soft long gates; continuation `cos_min=0.997681`, no argmax mismatch; strict internal KV-V is `0.998817` | `target/profiles/v0197-a3b-q4-default-T128-P32-cont64-softcos.out` |

Read: strict long internal GDN/KV cosine is now classified as a diagnostic lens,
not the sole release gate. The decisive release question is whether the prefill
state keeps generation stable under real prompts and awkward chunk boundaries.
The Q3 native IQ3 path still needs real-rollout continuation/perf evidence and a
possible high-accuracy blk0 GDN projection kernel before defaulting, but the old
gate no longer blocks solely on internal-state cosine drift that also appears in
Q4 controls.

## 2026-06-02 — v0.196 A3B Matrix-VT Fix And Q3 Long Gate

Status: fixed a real G8 matrix-attention threshold-crossing bug and sharpened
the A3B long-context correctness read. No Q3 default yet, but the remaining
blocker is now much narrower.

Findings and gates, AC power, sequential GPU runs:

| Gate | Result | Artifact |
| --- | ---: | --- |
| G8 matrix oracle before fix | fail at `layer=3 chunk_start=96`: `cos=0.760188`, `max_abs=3.506` | `target/profiles/v0196-a3b-q3-native-iq3-T128-P32-layer0-out-g8-oracle.out` |
| G8 matrix oracle after fix | all 10 A3B attn layers print `cos=1.000000`; max_abs `<=1.36e-2` | `target/profiles/v0196-a3b-q3-native-iq3-T128-P32-layer0-out-g8-oracle-vtfix.out` |
| Q4 A3B T40/P40 regression | pass; final/next logits `0.999428/0.999950` | `target/profiles/v0196-a3b-q4-default-T40-P40-vtfix-next.out` |
| Q4 A3B T128/P32 long probe | final/next logits pass `0.999939/0.999741`, strict KV-V is `0.998817` | `target/profiles/v0196-a3b-q4-default-T128-P32-vtfix-next.out` |
| Q3 native IQ3, blk0 `qkv+alpha` repair, T128/P32 | pass; final/next logits `0.999986/0.999990` | `target/profiles/v0196-a3b-q3-native-iq3-T128-P32-gdn-qkv-alpha-layer0-vtfix-next.out` |
| Q3 native IQ3, blk0 `qkv+alpha` repair, T112/P112 | strict GDN state/conv fail `0.998308/0.998942`, but final/next logits pass `0.999964/0.999668` | `target/profiles/v0196-a3b-q3-native-iq3-T112-P112-qkv-alpha-layer0-next.out` |
| Q3 `pp1024` paired, blk0 `qkv+alpha` | `1466.94 / 1434.23 t/s` (`1.023x`) | `target/profiles/v0196-a3b-q3-pp1024-native-iq3-gdn-qkv-alpha-layer0-paired.json` |
| Q3 `pp4096` paired, blk0 `qkv+alpha` | `1420.57 / 1373.50 t/s` (`1.034x`) | `target/profiles/v0196-a3b-q3-pp4096-native-iq3-gdn-qkv-alpha-layer0-paired.json` |
| Q3 `pp16384` paired, blk0 `qkv+alpha` | `1153.65 / 1040.77 t/s` (`1.108x`) | `target/profiles/v0196-a3b-q3-pp16384-native-iq3-gdn-qkv-alpha-layer0-paired.json` |

Mechanism: G8 matrix attention became active only after the first chunks for
small `prefill_chunk` values, but the transposed-V scratch had only been updated
for chunks that already used matrix attention. The first matrix chunk could
therefore read stale/uninitialized prefix V rows. The fix tracks per-attention
VT coverage and rebuilds the prefix when matrix attention first becomes active
mid-call.

Read: this was a real correctness bug in the long/chunked harness path, not a
MoE issue. The Q3 native-IQ3 path with a blk0 `qkv+alpha` GDN matvec oracle now
passes T128/P32 and still beats llama.cpp at `pp1024/4096/16384`, though the
short/medium margin is thin. T112 and Q4 T128 show that the old strict internal
GDN/KV cosine gates can fail while final and one-step continuation logits still
pass; the next defaultability question is whether to replace the long internal
state gate with a continuation/generation gate, or to build a high-accuracy GDN
projection kernel that avoids the layer0 matvec tax.

## 2026-06-02 — v0.195 A3B Q3 GDN Drift Isolation

Status: added env-gated GDN drift diagnostics and surgical repair oracles for the
A3B Q3 native-IQ3 candidate. No new default yet.

New diagnostic switches:

- `QWEN_PREFILL_GDN_BATCHED=0`: full per-token GDN fallback oracle. It fixes the
  T40 GDN drift but is far too slow for prefill.
- `QWEN_PREFILL_GDN_PROJ_ORACLE_LAYER=<blk[,blk]>`: compares real-path Q8_0
  GDN mat-mat projections against repeated Q8_0 matvec on the same activation
  rows.
- `QWEN_PREFILL_GDN_MATVEC_PROJ=<qkv|z|beta|alpha|out|front|all>` plus optional
  `QWEN_PREFILL_GDN_MATVEC_LAYER=<blk[,blk]>`: replaces selected batched GDN
  projections with repeated matvec as a surgical correctness oracle.

Key dirty findings, AC power, sequential GPU runs:

| Gate | Result | Artifact |
| --- | ---: | --- |
| baseline Q3 T40/P40 native IQ3 | fail: logits `0.999308`, GDN state/conv `0.998105/0.996946` | `target/profiles/v0195-a3b-q3-native-iq3-T40-worst-gdn.out` |
| worst layers | state `gdn=18 blk=24`, conv `gdn=9 blk=12` | same |
| real-proj oracle blk0 | local Q8 mat-mat vs matvec max deltas: qkv `1.97e-3`, z `1.17e-3`, out `2.02e-4`; cos prints `1.000000` | `target/profiles/v0195-a3b-q3-native-iq3-T40-gdn-proj-oracle-layer0.out` |
| all GDN out projections as matvec | T40 passes but pp1024 collapses to `0.409x` llama.cpp | `target/profiles/v0195-a3b-q3-pp1024-native-iq3-gdn-out-matvec-paired.json` |
| only blk0 GDN out as matvec | T34/T40/T64 pass; T112/T128 fail | `target/profiles/v0195-a3b-q3-native-iq3-T{34,64,128}-P*-gdn-out-matvec-layer0*.out` |
| blk0 GDN out paired `pp1024` | `1529.29 / 1433.77 t/s` (`1.067x`) | `target/profiles/v0195-a3b-q3-pp1024-native-iq3-gdn-out-matvec-layer0-paired.json` |
| blk0 GDN out paired `pp4096` | `1476.97 / 1371.75 t/s` (`1.077x`) | `target/profiles/v0195-a3b-q3-pp4096-native-iq3-gdn-out-matvec-layer0-paired.json` |
| blk0 GDN out paired `pp16384` | `1146.08 / 1103.23 t/s` (`1.039x`) | `target/profiles/v0195-a3b-q3-pp16384-native-iq3-gdn-out-matvec-layer0-paired.json` |

Read: the T40 blocker is not native IQ3 MoE. Tiny Q8_0 projection differences in
the first GDN block can amplify through the recurrent/residual stream; replacing
only `blk.0.ssm_out` with repeated matvec makes the strict T34/T40/T64 gates pass
while preserving a small llama.cpp win. However, T112/T128 failures remain and
full per-token GDN fallback does not fix T128, so there is a separate longer
multi-chunk correctness issue. Do not default A3B Q3 native IQ3 until that long
issue is classified or scoped with evidence. Next high-EV branch is a high-
accuracy Q8_0 batched GDN-out kernel for blk0 plus a T112/T128 chunk-boundary
diagnostic.

## 2026-06-02 — v0.193 A3B Q3 Native IQ3 MoE Candidate

Status: added an env-gated native `IQ3_XXS` routed gate/up path for A3B Q3 MoE.
`QWEN_PREFILL_MOE_GROUPED_IQ3_GATEUP=1` keeps IQ3 expert gate/up banks native at
load time, enables native IQ3 single-token MoE matvec, and routes prefill chunks
`>=32` through grouped IQ3 SwiGLU plus the existing grouped IQ4_XS down. Default
still dequants IQ3 gate/up to F32; do not default until the long Q3 GDN-state gate
is resolved or explicitly waived as unrelated.

Dirty validation and measurements, AC power, no thermal/performance warnings:

| Gate | Result | Artifact |
| --- | ---: | --- |
| IQ3 matvec oracle | `max|delta|=1.024e-7` | `target/profiles/v0193-moe-iq3-matvec-oracle.out` |
| grouped IQ3 SwiGLU oracle | `cos=1.000000`, `max|delta|=7.958e-7` | `target/profiles/v0193-moe-iq3-grouped-swiglu-oracle.out` |
| short A3B Q3 prefill-vs-single | passed | `target/profiles/v0193-a3b-q3-native-iq3-correctness-short.out` |
| dirty paired `pp1024` | `1483.11 / 1375.26 t/s` (`1.078x`) | `target/profiles/v0193-a3b-q3-pp1024-native-iq3-paired.json` |
| dirty paired `pp4096` | `1531.35 / 1370.66 t/s` (`1.117x`) | `target/profiles/v0193-a3b-q3-pp4096-native-iq3-paired.json` |
| dirty paired `pp16384` | `1334.30 / 1163.80 t/s` (`1.146x`) | `target/profiles/v0193-a3b-q3-pp16384-native-iq3-paired.json` |

Post-commit clean repeat rows with `build_commit=0474d7d22`, `build_dirty=0`:

| Shape | qwen | llama.cpp | Ratio | Artifact |
| --- | ---: | ---: | ---: | --- |
| `pp1024` | `1486.94`, `1490.63` | `1383.51`, `1384.73` | `1.075x`, `1.076x` | `target/profiles/v0193-clean-a3b-q3-pp1024-native-iq3-paired.json` |
| `pp4096` | `1533.85`, `1532.84` | `1369.89`, `1367.76` | `1.120x`, `1.121x` | `target/profiles/v0193-clean-a3b-q3-pp4096-native-iq3-paired.json` |
| `pp16384` | `1311.75` | `1163.09` | `1.128x` | `target/profiles/v0193-clean-a3b-q3-pp16384-native-iq3-paired.json` |

Phase trace at `pp1024` with the native IQ3 env books routed
SwiGLU/down/reduce at `181.34/96.44/6.20 ms`; prior grouped F32/IQ4_XS was about
`344.96/96.51/6.12 ms`. Artifact:
`target/profiles/v0193-a3b-q3-pp1024-native-iq3-phase-summary.tsv`.

Read: this is the discontinuous low-bit MoE catch-up branch. Native IQ3 removes
the resident-F32 gate/up tax and moves A3B Q3 from the clean fallback `90.28 t/s`
and the grouped-F32 candidate `1191.03 t/s` to clean llama.cpp-beating rows at
`pp1024/4096/16384`. `pp16` improvement is not grouped prefill; it is native IQ3
single-token fallback. Remaining defaultability work is the known long Q3 GDN
state drift.

## 2026-06-02 — v0.192 A3B Q3 Grouped F32/IQ4_XS Candidate

Status: added an env-gated low-bit A3B grouped routed-MoE candidate, not a new
default. `QWEN_PREFILL_MOE_GROUPED_F32_GATEUP=1` routes the runtime
`F32/F32/IQ4_XS` combination through grouped F32 gate/up SwiGLU and grouped
IQ4_XS down. The gate still requires `chunk_p >= 32`; default remains the exact
token fallback while the Q3 state-correctness gate is unresolved.

Dirty validation and measurements:

| Gate | Result | Artifact |
| --- | ---: | --- |
| Q3 short support gate | `cos(logits)=1.000000`; GDN/KV all `1.000000` | `QWEN_A3B_MOE_MODEL=...Q3_K_M cargo test ...35b_a3b_moe` |
| `pp16` threshold | `32.21 t/s` | `target/profiles/v0192-a3b-q3-pp16-th32.json` |
| `pp32` threshold | `66.10 t/s` | `target/profiles/v0192-a3b-q3-pp32-th32.json` |
| `pp128` grouped/fallback | `260.59 / 86.41 t/s` | `target/profiles/v0192-a3b-q3-pp128-grouped-vs-fallback.json` |
| `pp1024` paired | `1191.03 / 1391.98 t/s` (`0.856x`) | `target/profiles/v0192-a3b-q3-pp1024-grouped-paired-compare.json` |
| `pp4096` paired | `1225.51 / 1371.09 t/s` (`0.894x`) | `target/profiles/v0192-a3b-q3-pp4096-grouped-paired-compare.json` |
| Q4 regression smoke | `1445.25 t/s` at `pp1024` | `target/profiles/v0192-a3b-q4-pp1024-regression-smoke.json` |

Grouped Q3 `pp1024` phase trace books routed SwiGLU/down/reduce at
`344.96/96.51/6.12 ms`; GDN QKV and attention are next at `88.14/66.60 ms`.
Artifact: `target/profiles/v0192-a3b-q3-pp1024-grouped-f32-iq4xs-phase-summary.tsv`.

Longer Q3 prefill-vs-single (`T=40`, `chunk=32`) still fails the strict GDN state
gate for both default and env-grouped paths: final-logit cosine is about
`0.99931`, but GDN state/conv are about `0.99811/0.99695`. Artifacts:
`target/profiles/v0192-a3b-q3-default-correctness.out` and
`target/profiles/v0192-a3b-q3-grouped-f32-correctness.out`.

Read: the env candidate proves the structural headroom: A3B Q3 `pp1024` can move
from clean fallback `90.28 t/s` to dirty grouped `1191.03 t/s`, but the path is
not defaultable while the Q3 state gate is unresolved and still trails llama.cpp
by `10-15%` at `pp1024/4096`. The next gap is not route or down-only; it is mostly
grouped F32 gate/up throughput and the cost of dequanting IQ3 expert banks to
resident F32. Native grouped `IQ3_XXS/IQ3_S` gate/up is now the principled next
low-bit MoE branch.

## 2026-06-02 — v0.191 A3B Q3 Fallback Phase Split

Status: added trace-only phase splitting for the non-grouped MoE token fallback.
When `QWEN_PREFILL_TRACE_LAYER_PHASES=1`, the fallback now reports aggregated
per-layer totals for copy, route, routed gate/up/silu/down/weighted-sum, shared,
and residual scatter instead of booking the whole path into an attribution sink.
Default non-trace scheduling stays on the original single encoder per token.

Dirty instrumentation rows on A3B Q3 `pp16`:

| Phase | Total ms | Read |
| --- | ---: | --- |
| routed up F32 | `49.93` | major wall |
| routed gate F32 | `49.65` | major wall |
| routed down IQ4_XS | `46.82` | also major |
| shared token loop | `34.31` | secondary |
| route token loop | `20.05` | secondary |

Artifact: `target/profiles/v0191-a3b-q3-pp16-fallback-phase-summary.tsv`.
Read: grouped/batched structure is the missing primitive. Gate+up are the largest
combined slice, but down is close enough that a complete low-bit MoE answer needs
both grouped gate/up and grouped IQ4_XS down; down-only is not a plausible lcpp
catch-up path.

## 2026-06-02 — v0.189 A3B Q3 MoE Fallback Support

Status: closed the immediate A3B low-bit MoE support hole without claiming a fast
path. `Qwen3.5-35B-A3B-Q3_K_M.gguf` uses `IQ3_XXS/IQ3_XXS/IQ4_XS` routed expert
banks in all 40 MoE layers. The loader dequants `IQ3_XXS` gate/up banks to F32,
while `IQ4_XS` down stays native; before this branch the prefill token fallback
rejected that F32/IQ4_XS combination and failed during warmup.

Change: added generic GPU routed-MoE fallback kernels for F32 gate/up expert-bank
matvec and IQ4_XS down expert-bank matvec, then routed them through
`encode_moe_routed_ffn_gpu` and the profiler split path. This is an oracle/support
path, not grouped MoE coverage: `gguf_fastpath_audit.py` still reports `0/40`
grouped coverage with `IQ3_XXS/IQ3_XXS/IQ4_XS:40`.

Dirty-branch support smokes, AC power, no thermal/performance warnings:

| Shape | qwen row | Artifact |
| --- | ---: | --- |
| `pp1` | `1.90 t/s` | `target/profiles/v0189-a3b-q3-pp1-f32-iq4xs-fallback.json` |
| `pp16` | `32.16 t/s` | `target/profiles/v0189-a3b-q3-pp16-f32-iq4xs-fallback.json` |
| `pp1024` | `90.30 t/s` | `target/profiles/v0189-a3b-q3-pp1024-f32-iq4xs-fallback.json` |

Post-commit clean gate after rebuilding `qwen-bench` with
`build_commit=62797d0ed`, `build_dirty=0`:

- Paired A3B Q3 `pp1024`, no warmup:
  `qwen=90.28`, `llama.cpp=1392.02`, `qwen/lcpp=0.065x`.
  Artifact: `target/profiles/v0189-a3b-q3-pp1024-paired-compare.json`.
- No-FFN budget at the same shape: base `90.29 t/s`, no-FFN `2858.64 t/s`;
  GPU time falls from `11.15 s` to `0.331 s`. Artifact:
  `target/profiles/v0189-a3b-q3-pp1024-noffn-budget.json`.

Read: this converts a crash into a measurable baseline and gives a GPU oracle for
native low-bit MoE work. Clean evidence says the fallback is overwhelmingly FFN
bound and nowhere near llama.cpp; the next step is phase-split instrumentation,
then a grouped F32 gate/up scaffold before native IQ3 gate/up. Grouped IQ4_XS down
is still needed, but down-only is unlikely to close the gap. Do not mark low-bit
A3B grouped MoE as covered until native grouped expert-bank kernels exist.

## 2026-06-01 — v0.187 Paired Dense 27B Recheck

Status: clean paired qwen-vs-llama dense 27B prefill recheck using
`scripts/profile/prefill_compare.py` after the phase-summary and stale-anchor
methodology fixes. Artifacts are
`target/profiles/v0187-27b-pp{512,1024,4096,16384}-paired-compare.json`.
Qwen rows carry `build_commit=64b7c4415`, `build_dirty=0`; llama.cpp is
`14aa3d375`, `flash_attn=false`, `has tensor=false`. Each shape alternates
engine order by block and discards block `0` as system warmup.

Paired dense 27B results after the discard block:

| Shape | Paired qwen/lcpp rows | Mean qwen/lcpp | Read |
| --- | --- | ---: | --- |
| `pp512` | `238.55/238.11`, `239.41/232.84`, `231.63/226.29` | `1.02x` | short won |
| `pp1024` | `216.92/209.80`, `221.09/215.72`, `228.84/216.73` | `1.04x` | old gap invalidated |
| `pp4096` | `220.75/212.56`, `220.95/212.29`, `220.74/212.66` | `1.04x` | stable paired win |
| `pp16384` | `204.84/187.90`, `207.69/199.87` | `1.06x` | true-long win held |

Read: dense 27B prefill should now be treated as paired-won across the measured
synthetic shapes. The earlier `pp1024` and `pp4096` concerns were measurement
discipline problems: stale cold llama.cpp anchors and unpaired qwen drift made
isolated rows misleading. Keep Q4 N64 default-on; do not open another dense
short/medium prefill branch unless a paired comparator run shows a real residual.

## 2026-06-01 — Paired Cross-Engine Prefill Comparator

Status: added `scripts/profile/prefill_compare.py` after the 27B `pp1024`
investigation showed stale/cool-vs-hot llama.cpp anchors can dominate short and
medium prompt conclusions. The comparator alternates qwen and llama.cpp within
paired blocks, captures thermal and memory probes around each run, records
per-engine JSON rows and stderr, and emits paired qwen/lcpp deltas. It writes
compact checkpointed JSON so interrupted runs still leave usable evidence.

Use it when making cross-engine claims at drift-prone shapes:

```sh
uv run scripts/profile/prefill_compare.py \
  --model "$MODEL" \
  --n-prompt 1024 \
  --runs 3 \
  --cooldown-seconds 15 \
  --repeat-blocks 4 \
  --discard-first-block \
  --output target/profiles/prefill-compare.json
```

Smoke: `target/profiles/v0186-prefill-compare-smoke.json` ran 0.8B `pp16`
with one qwen row and one llama.cpp row, verified JSON parsing, checkpointing,
and paired delta output. Do not use the smoke as a performance claim.

## 2026-06-01 — Prefill Phase Summary Single-Chunk Fix

Status: methodology fix found while attacking the 27B `pp1024` dense residual.
`prefill_phase_summary.py --last-pass` previously inferred pass boundaries only
when chunk ids decreased, so single-chunk warmup+timed traces such as `pp512` and
`pp1024` were summarized as warmup+timed together. The helper now increments the
pass index on `(chunk,start,layer)` rewind, giving the expected `48/16` GDN/attn
counts for the 27B `pp1024` trace instead of doubled `96/32` counts.

Artifacts: the corrected summaries are
`target/profiles/v0185-27b-pp1024-{default,rollback}-phase-summary.*`. The
default corrected summary selects pass `1`, has `pass_counts={0:960,1:960}`, and
totals `4224.8 ms`, matching the qwen trace GPU time. Treat older single-chunk
`--last-pass` summaries as suspect unless regenerated with this fix.

Methodology read: serialized llama.cpp Metal profiling at `pp1024` drops to
`195.05 t/s`, so it is shape attribution only, not a normal-throughput anchor.
A dirty same-session normal warmup check saw qwen `235.97/223.49 t/s` and
llama.cpp `202.90 t/s`; rerun clean after this methodology commit before using
it to change the scoreboard.

## 2026-06-01 — v0.184 Clean Q4 N64 Re-Anchor

Status: clean post-commit re-anchor for `v0.184` after rebuilding
`qwen-bench` so artifacts carry `build_commit=28de0d0a8` and
`build_dirty=0`. Use the `target/profiles/v0184-postbuild-*` artifacts for
clean claims; earlier `v0184-clean-*` files were run with the pre-commit binary
metadata and should only be treated as same-code scratch.

All postbuild rows below were AC-power rows with no recorded thermal or
performance warnings and full fast-path coverage for the target model.

Dense 27B default rows after rebuild:

| Shape | Default rows | Read |
| --- | ---: | --- |
| `pp512` | `236.44 / 236.64` | small short gap vs lcpp `240.06` |
| `pp1024` | `228.67 / 223.41` | noisy short/medium gap vs lcpp `236.91` |
| `pp4096` | `207.31 / 223.35` | cold/noisy; not a strong win claim |
| `pp16384` | `208.06 / 209.57` | stable true-long win vs lcpp `198.96` |

Postbuild rollback/default A/B tightens the attribution:

| Shape | Rollback | Default | Read |
| --- | ---: | ---: | --- |
| `pp512` | `236.10 / 233.76` | `236.70 / 236.76` | default not harmful |
| `pp1024` | `215.09 / 215.34` | `223.97 / 221.13` | default helps, gap remains |
| `pp4096` | `215.16 / 211.86` | `212.86 / 216.38` | flat/noise, near lcpp `213.07` |
| `pp16384` | `199.45 / 202.44` | `208.06 / 209.11` | clear N64 true-long win |

Read: keep Q4 N64 default-on because it is neutral-to-positive at short shapes
and clearly wins at true-long, but stop overclaiming `pp4096`. The live dense
prefill residual is now the `pp1024` short/medium gap plus `pp4096` variance;
`pp16384` is the cleanest evidence that the larger Q4 tile is the right
direction for long prompts.

Warmed MoE smokes remain clean with full coverage: A3B `pp1024` is `1624.82 t/s`
with `40/40` grouped MoE coverage, and A10B `pp1024` is `505.04 t/s` with
`48/48` coverage. The cold `--no-warmup` A10B smoke produced an invalid
`129.71 t/s` row; use warmed MoE smokes for this huge model.

Decode sentinels after rebuild: 27B `tg128` is `24.04 t/s`, A3B `tg128` is
`78.46 t/s`, and A10B `tg128` is `34.84 t/s`. That preserves dense/A3B decode
wins and leaves A10B decode as parity/monitoring against the old `35.23 t/s`
llama.cpp anchor.

## 2026-06-01 — Dense 27B Q4 N64 Prompt Mat-Mat

Status: precommit dense-prompt win after the v0.183 falsifier packet. Raw
artifacts are in `target/profiles/v0184-*q4-n64*` plus fresh llama.cpp anchors
`target/profiles/v0184-27b-lcpp-pp{4096,16384}.json`. The qwen rows are
`build_dirty=1` because they were taken before this checkpoint; all were AC-power
rows with clean fast-path audits and no thermal/performance warnings.

Change: Q4_K prompt mat-mat now has a default-on `N=64` full-tile path for
`n_query % 64 == 0` and `n_out % 64 == 0`, with
`QWEN_MATMAT_Q4_K_N64=0` as rollback. The tile keeps the classic per-simdgroup
`32M x 16N` work but uses 8 simdgroups per threadgroup, so each dequantized Q4_K
weight panel is reused across 64 prompt columns instead of 32.

Correctness: release `qwen-bench` builds, and the Q4_K mat-mat oracle test passes
with the new `n_query=64` row (`min_cos=1.000000`, `max|Delta|=6.249e-5`).

Dense 27B candidate/rollback rows:

| Shape | Rollback/base | Q4 N64/default | Read |
| --- | ---: | ---: | --- |
| `pp512` | `238.56 / 238.41` | `239.40 / 239.34` | neutral-positive |
| `pp1024` | `237.33 / 232.99` | `241.42 / 234.56` | positive, noisy |
| `pp4096` candidate | `200.73 / 213.95` | `214.28 / 221.37` | positive |
| `pp4096` default vs rollback | `201.99 / 205.96` | `221.75 / 222.70` | strong positive |
| `pp16384` candidate | `200.07` | `207.84` | positive |
| `pp16384` default | n/a | `208.01` | confirms no-env default |

Fresh llama.cpp anchors (`-fa 0`, `has tensor = false`) are `213.07 t/s` at
`pp4096` and `198.96 t/s` at `pp16384`, so the dirty precommit qwen default is
about `1.04-1.05x` llama.cpp on both long dense rows. `pp512` remains roughly
parity, not a claimed short win.

Primary MoE smokes with the candidate enabled are neutral/positive: A3B `pp1024`
`1622.43 -> 1624.92 t/s`; A10B `pp1024` `506.97 -> 514.63 t/s` with nearly flat
GPU ms/token. Keep full primary re-anchor as the next clean checkpoint.

Negative result captured: matching llama.cpp's dense FFN graph order by computing
`up` before `gate` regressed 27B `pp4096` (`~203 t/s` vs base `~212 t/s`), so the
winning lever is larger Q4 prompt tiling, not operation order.

## 2026-06-01 — Dense 27B Post-Reanchor Falsifiers

Status: immediate follow-up after the primary re-anchor to avoid chasing stale
or cold-row artifacts. Raw artifacts are in `target/profiles/v0183-27b-*`.

27B `pp4096` chunk-size probe, qwen-only:

| prefill_chunk | qwen |
| ---: | ---: |
| `1024` | `208.79` |
| `2048` | `210.85` |
| `4096` | `209.53` |

27B `pp16384` chunk-size probe, qwen-only:

| prefill_chunk | qwen |
| ---: | ---: |
| `1024` | `199.97` |
| `2048` | `199.42` |
| `4096` | `198.05` |

Read: larger chunks do not clear a default gate. `2048` is a small/noisy
`pp4096` uptick but loses at true-long; keep `1024` as the safe default cap.

Other repeated `pp4096` A/B checks:

| Variant | Rows | Read |
| --- | --- | --- |
| `QWEN_MATMAT_QK_LLAMA_SMEM=1` | `213.95/213.39` vs base `211.97/213.65` | flat/noise |
| `QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=1` | `214.57/214.92` vs base `214.51/213.95` | sub-1% |

One `pp16384` fused-FFN row is similarly small: `200.63` vs base `199.70`.
Do not default either branch without a stronger paired gate.

No-warmup phase trace at 27B `pp4096` keeps the dense target clear: FFN gate/up/
down dominate (`~7.66 s`, `~7.66 s`, `~7.89 s` combined across GDN+attention
layers), while GDN projection/back is secondary (`gdn_qkv+gdn_z+gdn_back ~3.76
s`) and attention body is small (`~0.32 s`). Next exact work remains a
llama-shaped FFN/GDN mat-mat/dataflow differential, not chunk policy, smem policy,
or attention-body tuning.

## 2026-06-01 — Primary Guardrail Re-Anchor After Low-Bit Decode

Status: clean primary-family re-anchor after closing the local dense low-bit
decode lane. Raw artifacts are in `target/profiles/v0182-primary-*` plus the
27B no-op packet `target/profiles/v0182-27b-pp4096-noop-budget.json`.

All qwen rows below are `build_dirty=0`, AC power, and no thermal/performance
warnings:

| Model | Shape | qwen | llama.cpp | qwen/lcpp |
| --- | ---: | ---: | ---: | ---: |
| 27B dense | `pp1024` | `236.04` | `236.91` | `1.00x` |
| 27B dense | `pp4096` warmed/r3 | `212.02` | `212.22` | `1.00x` |
| 27B dense | `pp16384` | `199.57` | `199.92` | `1.00x` |
| 35B A3B | `pp1024` | `1624.80` | `1394.35` | `1.17x` |
| 35B A3B | `pp4096` | `1569.79` | `1346.36` | `1.17x` |
| 35B A3B | `pp16384` | `1305.47` | `1090.81` | `1.20x` |
| 122B A10B | `pp1024` | `511.62` | `444.00` | `1.15x` |
| 122B A10B | `pp4096` | `491.22` | `390.47` | `1.26x` |
| 122B A10B | `pp16384` | `408.96` | `355.22` | `1.15x` |
| 27B dense | `tg128` | `24.30` | `22.08` | `1.10x` |
| 35B A3B | `tg128` | `82.82` | `75.90` | `1.09x` |
| 122B A10B | `tg128` | `35.21` | `35.23` | `1.00x` |

Methodology note: the first one-run 27B `pp4096` row after the broad anchor
sequence was `201.47 / 206.72` (`0.97x`), but a repeated warmed qwen run and
paired lcpp repeat landed at `212.02 / 212.22`. Treat close 27B dense rows as
requiring repeated/warmed sweeps; a single cold/residency row can mis-rank the
work queue.

27B `pp4096` no-op budget, same sweep: base `213.12`, no-attention-body
`209.52`, no-GDN-body `230.24`, and no-FFN `644.87`. Read: primary MoE is still
won, primary decode is held, and dense 27B prefill is now the only narrow primary
lane. The next exact dense work should stay phase-driven and focus on FFN/GDN
projection/body headroom, not attention-body micro-tuning.

## 2026-06-01 — Clean Low-Bit Decode Re-Anchor After Fast Mat-Vecs

Status: clean post-commit 0.8B decode sentinel after the Q2/Q3/IQ4_NL/IQ4_XS
fast mat-vec sequence. Raw artifacts are in
`target/profiles/v0181-{clean,lcpp}-0p8b-*tg128.json`.

All qwen rows are `build_dirty=0`, AC power, and no thermal/performance
warnings:

| Quant | qwen | llama.cpp | qwen/lcpp |
| --- | ---: | ---: | ---: |
| Q2_K | `378.90` | `279.50` | `1.36x` |
| Q3_K_M | `339.43` | `260.61` | `1.30x` |
| IQ4_NL | `341.90` | `267.80` | `1.28x` |
| IQ4_XS | `343.51` | `271.95` | `1.26x` |
| Q4_K_M | `341.89` | `271.43` | `1.26x` |

Read: the local 0.8B dense low-bit decode lane is now a clean win across the
measured family. Per cx's review, stop harvesting local low-bit decode tails and
return to primary guardrail re-anchoring plus phase-driven 27B dense prefill
work unless a larger quant file exposes a new miss.

## 2026-06-01 — IQ4_NL Decode Mat-Vec Fast Kernel

Status: cheap follow-up after the Q3_K decode win to close the last measured
dense low-bit decode residual. Raw artifacts are in
`target/profiles/v0180-iq4nl-decode-*tg128.json`.

IQ4_NL decode now uses a llama-shaped row-reuse mat-vec kernel with `NR0=2` and
`NSG=2`, default-on behind `QWEN_MATVEC_IQ4_NL_FAST=0`.

Precommit 0.8B `tg128` A/B, AC power and no thermal/performance warnings:

| Variant | t/s |
| --- | ---: |
| rollback | `267.64` |
| default | `342.58` |
| llama.cpp | `274.74` |

Read: the local dense low-bit decode family is now parity/win in the measured
0.8B files; IQ4_NL moves from a small `~0.97x` residual to `1.25x` llama.cpp.
Only 0.8B IQ4_NL is available under `~/models`, so this is a local closure, not
a broader IQ4_NL family claim.

Validation: release `qwen-bench` builds and the full mat-vec/mat-mat unit-test
filter passes for half, Q2, Q3, legacy Q4, and IQ4 weights. The helper refactor
keeps Q2/Q3/IQ4_XS on 256-element block checks and uses a 32-element check for
IQ4_NL.

## 2026-06-01 — Q3_K Decode Mat-Vec Fast Kernel

Status: targeted low-bit decode fix for the last measured 0.8B decode miss. Raw
artifacts are in `target/profiles/v0179-q3-decode-*tg128.json`.

Q3_K decode now uses a llama-shaped row-reuse mat-vec kernel with `NR0=2` and
`NSG=2`, replacing the scalar coverage-first path by default. Rollback is
`QWEN_MATVEC_Q3_K_FAST=0`.

Precommit 0.8B `tg128` A/B, AC power and no thermal/performance warnings:

| Variant | t/s |
| --- | ---: |
| rollback | `245.65` |
| default | `338.76` |
| llama.cpp | `259.31` |

Broader one-run dense Q3 decode sentinels:

| Model | qwen | llama.cpp | qwen/lcpp |
| --- | ---: | ---: | ---: |
| 2B | `196.13` | `178.58` | `1.10x` |
| 9B | `65.35` | `58.59` | `1.12x` |
| 27B | `22.56` | `20.01` | `1.13x` |

Validation: release `qwen-bench` builds and the full mat-vec/mat-mat unit-test
filter passes for half, Q2, Q3, legacy Q4, and IQ4 weights. Read: Q3_K decode
is no longer the low-bit miss; the remaining measured dense low-bit decode
residual is IQ4_NL at roughly `0.97x`, which should be treated as lower EV than
primary guardrails or a fresh quant-family sweep.

## 2026-06-01 — Clean Low-Bit Decode Re-Anchor

Status: clean post-commit decode sentinel after the Q2/IQ4_XS mat-vec win. Raw
artifacts are in `target/profiles/v0178-clean-qwen-0p8b-*tg128.json`.

Current 0.8B `tg128`, all `build_dirty=0`:

| Quant | qwen | llama.cpp | qwen/lcpp |
| --- | ---: | ---: | ---: |
| Q2_K | `309.94` | `283.37` | `1.09x` |
| Q3_K_M | `240.87` | `257.30` | `0.94x` |
| IQ4_NL | `261.20` | `268.79` | `0.97x` |
| IQ4_XS | `341.86` | `281.26` | `1.22x` |
| Q4_K_M | `335.73` | `268.05` | `1.25x` |

Read: Q2_K and IQ4_XS decode are now wins, while Q4_K_M remains a large win.
Q3_K_M is the only measured 0.8B low-bit decode miss above noise; IQ4_NL is a
small residual and should not outrank Q3 or primary-family guardrails.

## 2026-06-01 — Q2/IQ4_XS Decode Mat-Vec Fast Kernels

Status: low-bit decode follow-up after the prefill mat-mat wins. Raw artifacts
are in `target/profiles/v0177-*tg128*`.

Decode sentinels found that prefill parity did not imply decode parity. Clean
baseline `tg128` rows before this branch were Q2_K `204.97 / 283.37` (`0.72x`)
and IQ4_XS `196.92 / 281.26` (`0.70x`) versus llama.cpp. Q3_K_M was smaller
but still behind at `240.23 / 257.30` (`0.93x`), and IQ4_NL was near parity at
`260.87 / 268.79` (`0.97x`). Q4_K_M remained won at `335.68 / 268.05`
(`1.25x`).

Q2_K and IQ4_XS decode now use llama-shaped row-reuse mat-vec kernels instead
of the scalar coverage-first one-row path. Rollbacks are
`QWEN_MATVEC_Q2_K_FAST=0` and `QWEN_MATVEC_IQ4_XS_FAST=0`.

Precommit A/B on 0.8B `tg128`:

| Quant | rollback | default | llama.cpp | default/lcpp |
| --- | ---: | ---: | ---: | ---: |
| Q2_K | `203.85` | `309.21` | `283.37` | `1.09x` |
| IQ4_XS | `196.97` | `342.64` | `281.26` | `1.22x` |

Validation: release `qwen-bench` builds and the mat-vec/mat-mat unit-test filter
passes for half, Q2, Q3, legacy Q4, and IQ4 weights. Q3_K_M decode remains the
next measured low-bit decode gap unless a clean post-commit re-anchor changes
the ordering.

## 2026-06-01 — Broader Dense Low-Bit Validation

Status: bounded family validation after the clean 0.8B low-bit re-anchor. Raw
artifacts are in `target/profiles/v0176-clean-*` and
`target/profiles/v0176-lcpp-*`.

All rows are `build_dirty=0`, `--require-fastpath-clean`, AC power, and no
thermal/performance warnings:

| Model | Quant | Shape | qwen | llama.cpp | qwen/lcpp |
| --- | --- | ---: | ---: | ---: | ---: |
| 2B | Q2_K | `pp1024` | `3648.37` | `3691.12` | `0.99x` |
| 2B | Q2_K | `pp4096` | `3593.04` | `3628.58` | `0.99x` |
| 2B | Q3_K_M | `pp1024` | `3599.90` | `3622.57` | `0.99x` |
| 2B | Q3_K_M | `pp4096` | `3534.15` | `3549.99` | `1.00x` |
| 2B | IQ4_XS | `pp1024` | `3652.37` | `3762.49` | `0.97x` |
| 2B | IQ4_XS | `pp4096` | `3591.45` | `3682.70` | `0.98x` |
| 9B | Q2_K | `pp1024` | `820.59` | `819.20` | `1.00x` |
| 9B | Q2_K | `pp4096` | `752.59` | `712.08` | `1.06x` |
| 9B | Q3_K_M | `pp1024` | `794.15` | `766.43` | `1.04x` |
| 9B | Q3_K_M | `pp4096` | `721.75` | `708.09` | `1.02x` |
| 9B | IQ4_XS | `pp1024` | `817.12` | `824.23` | `0.99x` |
| 9B | IQ4_XS | `pp4096` | `750.14` | `742.57` | `1.01x` |
| 27B | Q3_K_M | `pp1024` | `232.84` | `216.42` | `1.08x` |

Read: the dense low-bit tile work generalizes beyond 0.8B. The only residual
dense low-bit gap in this bounded packet is 2B IQ4_XS at `0.97-0.98x`, which is
not enough by itself to justify tile-tuning before broader guardrails. The audit
did find a different low-bit issue: A3B Q3/IQ4_XS MoE has `0/40` grouped expert
coverage because those GGUFs use `IQ3_XXS` or `IQ3_S` gate/up expert banks with
`IQ4_XS` down. That is the next quant-family structural gap if we stay on
low-bit MoE.

## 2026-06-01 — Clean Low-Bit Quant Re-Anchor

Status: clean post-commit re-anchor after the Q2/Q3/IQ4 simdgroup mat-mat
sequence. Raw artifacts are in `target/profiles/v0175-clean-0p8b-*`.

All rows below are `build_dirty=0`, AC power, clean fast-path audit, and no
thermal/performance warnings:

| Quant | Shape | qwen | llama.cpp | qwen/lcpp |
| --- | ---: | ---: | ---: | ---: |
| Q2_K | `pp1024` | `7455.29` | `7760.67` | `0.96x` |
| Q2_K | `pp4096` | `7324.36` | `7535.20` | `0.97x` |
| Q3_K_M | `pp1024` | `7478.83` | `7646.85` | `0.98x` |
| Q3_K_M | `pp4096` | `7245.10` | `7381.47` | `0.98x` |
| IQ4_NL | `pp1024` | `7677.57` | `7928.74` | `0.97x` |
| IQ4_NL | `pp4096` | `7430.59` | `7656.81` | `0.97x` |
| IQ4_XS | `pp1024` | `7572.44` | `7840.10` | `0.97x` |
| IQ4_XS | `pp4096` | `7299.96` | `7558.00` | `0.97x` |

Read: the local 0.8B low-bit cliff is no longer a structural gap; all four
audited low-bit families now sit within about `2-4%` of llama.cpp at medium and
long synthetic prefill. The remaining quant work should be broader family/shape
validation and targeted tuning only where a clean differential survives repeats.

## 2026-06-01 — Q2/Q3 Prompt Mat-Mat Simdgroup Tiles

Status: follow-up low-bit quant performance fix for dense Q2_K/Q3_K prefill.
Raw artifacts are in `target/profiles/v0174-0p8b-q{2,3}-*mm-ab.json` plus
`target/profiles/v0174-lcpp-0p8b-q{2,3}-pp4096.json`.

Q2_K and Q3_K now use the same prompt-native 64x32x32 simdgroup_matrix mat-mat
execution shape as the IQ4 fix. Rollbacks are `QWEN_MATMAT_Q2_K_MM=0` and
`QWEN_MATMAT_Q3_K_MM=0`.

Current 0.8B Q2_K A/B:

| Shape | rollback | default | llama.cpp | default/lcpp |
| --- | ---: | ---: | ---: | ---: |
| `pp1024` | `1404.38` | `7438.40` | `7760.67` | `0.96x` |
| `pp4096` | `1359.96` | `7351.80` | `7535.20` | `0.98x` |

Current 0.8B Q3_K_M A/B:

| Shape | rollback | default | llama.cpp | default/lcpp |
| --- | ---: | ---: | ---: | ---: |
| `pp1024` | `1999.66` | `7486.78` | `7646.85` | `0.98x` |
| `pp4096` | `1986.95` | `7230.47` | `7381.47` | `0.98x` |

Validation: release `qwen-bench` builds, `git diff --check` is clean, and the
Q2/Q3/Q4 Metal unit-test filter passes mat-vec plus mat-mat at `n_query=1/16/32`
for Q2/Q3. Prefill sweeps report clean fast-path coverage on AC power with no
thermal/performance warnings. The local 0.8B Q2/Q3/IQ4 performance cliff is now
mostly closed; next low-bit work should be clean post-commit anchors and broader
non-0.8B quant coverage, not more coverage-audit bookkeeping.

## 2026-06-01 — IQ4 Prompt Mat-Mat Simdgroup Tiles

Status: low-bit quant performance fix for dense IQ4_NL/IQ4_XS prefill. Raw
artifacts are in `target/profiles/v0173-0p8b-iq4*` and the low-bit baseline
packet is `target/profiles/v0173-{qwen,lcpp}-0p8b-*pp1024.json`.

The static fast-path audit was clean for local 0.8B low-bit files, but that was
only coverage: direct `pp1024` anchors showed Q2/Q3/IQ4 were still far behind
llama.cpp. Before this fix, 0.8B `pp1024` rows were Q2_K `1143.35 / 7760.67`
(`0.15x`), Q3_K_M `1996.98 / 7646.85` (`0.26x`), IQ4_NL `1196.28 / 7928.74`
(`0.15x`), and IQ4_XS `1197.54 / 7840.10` (`0.15x`).

IQ4_NL and IQ4_XS now use prompt-native 64x32x32 simdgroup_matrix mat-mat
tiles instead of the scalar coverage-first mat-mat kernel; rollback is
`QWEN_MATMAT_IQ4_NL_MM=0` or `QWEN_MATMAT_IQ4_XS_MM=0`.

Current 0.8B IQ4_NL A/B:

| Shape | rollback | default | llama.cpp | default/lcpp |
| --- | ---: | ---: | ---: | ---: |
| `pp512` | `1214.28` | `7276.16` | `7635.98` | `0.95x` |
| `pp1024` | `1196.78` | `7555.27` | `7928.74` | `0.95x` |
| `pp4096` | `1190.26` | `7424.67` | `7656.81` | `0.97x` |

Current 0.8B IQ4_XS A/B:

| Shape | rollback | default | llama.cpp | default/lcpp |
| --- | ---: | ---: | ---: | ---: |
| `pp1024` | `1198.66` | `7397.97` | `7840.10` | `0.94x` |
| `pp4096` | `1191.62` | `7307.61` | `7558.00` | `0.97x` |

Validation: release `qwen-bench` builds, the IQ4 Metal unit test passes for
mat-vec and mat-mat at `n_query=1/16/32`, prefill sweeps report clean fast-path
coverage, and the recorded rows were on AC power with no thermal/performance
warnings. At this checkpoint Q2_K and Q3_K remained the next low-bit performance
holes; the following entry closes them with the same execution shape.

## 2026-06-01 — Dense Group-4 Matrix Attention Default

Status: exact prompt-attention coverage fix for dense group-4, `head_dim=256`
models. Raw artifacts are in `target/profiles/v0172-*g4*`.

The static audit said 9B attention was covered, but the runtime matrix gate only
accepted the known G6/G8/G16 shapes. Qwen3.5 9B, 4B, 2B, and 0.8B use group-4
attention, so they were still falling back to the older packed attention body.
The default gate now enables matrix attention whenever `n_q == 4 * n_kv` and
`head_dim == 256`; rollback is `QWEN_PREFILL_ATTN_MATRIX_G4=0`.

The 9B `pp4096` A/B is decisive: default G4 rows are `762.49` and `771.51 t/s`,
while force-off rows are `604.46` and `604.73 t/s` (`~+27%`). Phase attribution
shows the attention body collapsing from the old packed `1884.93 ms` bucket to
matrix KQ/softmax/KQV totals of `52.79/19.80/53.89 ms`.

Same-session 9B rows now beat or match llama.cpp except for a noisy first long
row: `pp512` `813.36 / 814.10` (`1.00x`), `pp1024` `820.51 / 804.64` (`1.02x`),
`pp4096` `775.82 / 693.95` (`1.12x`), and repeat `pp16384` `690.68 / 678.06`
(`1.02x`). Smaller group-4 smokes at `pp1024` also move sharply versus force-off:
0.8B `7434.65 / 3726.61`, 2B `3636.43 / 2448.17`, and 4B `1463.29 / 1169.85`.
Against llama.cpp those land at `0.98x`, `1.00x`, and `1.01x` respectively.

Validation: release build is green and the default 27B prefill-vs-single gate
still passes. The G4 path reuses the same generic matrix-attention kernels already
covered by the G6/G8/G16 work; add a dedicated 9B oracle only if future changes
touch group-specific matrix math rather than the runtime gate.

## 2026-06-01 — Current Short And Decode Family Gate

Status: follow-up to the long `v0.170` re-anchor. Raw artifacts are in
`target/profiles/v0171-*`.

Current same-session `pp512/pp1024` rows against llama.cpp build `14aa3d375`:

| Model | Shape | qwen | llama.cpp | qwen/lcpp |
| --- | ---: | ---: | ---: | ---: |
| 27B dense | `pp512` | `237.37` | `240.06` | `0.99x` |
| 27B dense | `pp1024` | `227.44` | `228.03` | `1.00x` |
| 35B A3B | `pp512` | `1443.05` | `1380.23` | `1.05x` |
| 35B A3B | `pp1024` | `1624.98` | `1388.41` | `1.17x` |
| 122B A10B | `pp512` | `442.59` | `442.27` | `1.00x` |
| 122B A10B | `pp1024` | `514.53` | `439.27` | `1.17x` |

Current `tg128` decode rows:

| Model | qwen | llama.cpp | qwen/lcpp |
| --- | ---: | ---: | ---: |
| 27B dense | `24.17` | `22.06` | `1.10x` |
| 35B A3B | `78.08` | `75.56` | `1.03x` |
| 122B A10B | `35.65` | `35.19` | `1.01x` |

Read: decode remains won across the primary guardrails. Short prompt prefill is
also won or parity for MoE; the only remaining primary scoreboard blemish is
dense 27B at `pp512/1024`, where the gap is sub-`1.2%` and close enough to demand
repeat evidence before coding. A10B `pp512` has moved from the old gap to parity.

## 2026-06-01 — Current Long Family Re-Anchor

Status: post-`v0.169` benchmark checkpoint after the dense GDN pointer-hoist win.
Raw artifacts are in `target/profiles/v0170-*` plus the dense follow-up
falsifiers in `target/profiles/v0169-*chunk512*`, `v0169-*llama-smem*`, and
`v0169-*ffn-fused-ab*`.

The dense follow-up falsifiers are now current again. Matching llama.cpp's
`n_ubatch=512` by forcing qwen `prefill_chunk=512` loses to the default `1024`
chunk at 27B `pp4096` (`219.64` vs `222.28 t/s`) and `pp16384` (`200.74` vs
`203.37`). Llama-style reduced mat-mat threadgroup memory is phase-positive in
projection buckets but not total-robust: `pp512` is slightly negative,
`pp1024` is mixed/slightly negative, `pp4096` has one large low outlier, and
`pp16384` is only weakly mixed-positive. Dense fused-Q4 gate/up/SwiGLU also
remains rejected by same-process A/B: it loses on average at `pp4096` and loses
clearly at `pp16384`.

Current same-session long prefill rows against llama.cpp build `14aa3d375`:

| Model | Shape | qwen | llama.cpp | qwen/lcpp |
| --- | ---: | ---: | ---: | ---: |
| 27B dense | `pp4096` | `222.28` | `214.22` | `1.04x` |
| 27B dense | `pp16384` | `203.37` | `200.44` | `1.01x` |
| 35B A3B | `pp4096` | `1565.47` | `1353.70` | `1.16x` |
| 35B A3B | `pp16384` | `1328.69` | `1104.07` | `1.20x` |
| 122B A10B | `pp4096` | `497.72` | `401.05` | `1.24x` |
| 122B A10B | `pp16384` | `403.25` | `338.80` | `1.19x` |

Read: qwen now beats current llama.cpp on the long dense and MoE guardrails, with
the dense 27B `pp16384` margin narrow enough to keep monitoring. The A10B row is
warmed; a no-warm A10B `pp4096` row at `310.93 t/s` is a first-pass confound and
not the scoreboard anchor. Next highest-EV work is no longer dense FFN grinding;
run a full current family gate including `pp512/1024` and decode, then focus any
remaining gap, likely short-prompt MoE overhead, with phase evidence.

## 2026-06-01 — Dense GDN Step Pointer Increments

Status: exact dense GDN step address-lowering cleanup on top of the NSG4
row-grouping baseline. Raw artifacts are in `target/profiles/v0169-*gdn-ptrinc-*`.

The NSG4 kernel still recomputed token-stride addresses inside the recurrence
loop for Q, K, V, decay, beta, and output. This branch seeds per-lane pointers
once, hoists the `dk_base` add out of the token loop, and advances by the fixed
Q/K, V, and per-head strides each token. The generic packed fallback receives
the same cleanup, but the measured dense 27B path uses NSG4.

Correctness is green on the default 27B prefill-vs-single gate. Phase impact
clears the next GDN gate: at 27B `pp4096`, `gdn_step` moves from the NSG4
baseline `551.82 ms` to `482.15 ms` (`-12.6%`). At `pp16384`, it moves from
`2173.17 ms` to `1939.31 ms` (`-10.8%`). Relative to the pre-NSG4 full-tile
baseline, the two GDN-step cleanups together move `595.56 -> 482.15 ms` at
`pp4096` and `2379.00 -> 1939.31 ms` at `pp16384`.

Clean 27B rows on AC power are: `pp512=237.20`, `pp1024=235.23`,
`pp4096=222.28`, and `pp16384=203.37`. The big scoreboard movement is at
`pp4096`; `pp1024` is below the previous clean anchor, so do not overclaim a
uniform short-context win. Updated read: GDN step is no longer the top dense
residual. Rebase the qwen-vs-llama phase differential before coding more GDN
tail work; projection rows are now the higher-leverage dense target unless a
single narrow GDN staging probe clears a phase gate.

## 2026-06-01 — Dense GDN Step NSG4 Row Grouping

Status: exact dense GDN step execution-shape cleanup, modeled after llama.cpp's
multi-row gated-delta-net kernel ownership. Raw artifacts are in
`target/profiles/v0167-*gdn-nsg4*` and `target/profiles/v0167-clean-*`.

The packed GDN recurrence already amortized state read/write across the prompt
chunk, but launched one single-simdgroup threadgroup per `(v_head, dv)` row. The
new NSG4 kernel maps four `dv` rows into one threadgroup via four simdgroups,
cutting threadgroup count by `4x` without changing state layout or recurrence
math. Correctness is green on the default 27B prefill-vs-single gate.

Phase impact clears the GDN gate. At 27B `pp4096`, `gdn_step` moves from the
full-tile baseline `595.56 ms` to `551.82 ms` (`-7.3%`). At `pp16384`, it moves
from `2379.00 ms` to `2173.17 ms` (`-8.7%`). Neighboring GDN/FFN projections are
directionally lower in the same traces rather than paying for the step win.

Clean end-to-end rows before commit are the new dense anchors: 27B `pp512=236.25`,
`pp1024=238.33`, `pp4096=211.23`, and `pp16384=201.30`; post-commit build-clean
spot rows landed at `pp4096=210.17` and `pp16384=201.75`. Updated read: the lcpp
mechanics-copy strategy keeps paying when it targets a specific ownership or
addressing mismatch. Next GDN work should try pointer/increment hoists inside the
NSG4 loop and only then consider q/k staging or beta/decay packing.

## 2026-06-01 — Dense G6 Full-Tile Matrix Attention Kernels

Status: exact dense 27B G6 matrix-attention full-tile specialization. Raw
artifacts are in `target/profiles/v0165-*fulltile*` and `target/profiles/v0165-clean-*`.

The pointer-hoist win made inner-loop predicates the next nearest llama.cpp-style
mechanic. The KQ/KQV encoders now select full-tile kernels when `n_pos`,
`chunk_p*group`, and `head_dim=256` make the standard prompt tiles exact; arbitrary
edge shapes still use the checked kernels. Correctness is green on the default 27B
prefill-vs-single gate and an explicit full-tile gate via
`QWEN_TEST_27B_PREFILL_T=64 QWEN_TEST_27B_PREFILL_P=64`.

Phase impact is large and finally changes the dense-attention read. At 27B
`pp4096`, KQ/KQV move from pointer-hoist `160.41/158.21 ms` to `136.48/135.29 ms`
(`179.26/180.01 ms` before the two address-lowering cleanups). At `pp16384`, they
move from `2516.18/2660.00 ms` to `2169.60/2376.82 ms` (`2822.60/2971.57 ms`
before). Softmax remains flat around `60.7 ms` at `pp4096` and `~1.03 s` at
`pp16384`.

Clean end-to-end rows before commit were: 27B `pp512=234.50`, `pp1024=232.00`,
`pp4096=208.24`, and `pp16384=197.57`; A3B `pp1024=1588.65`; A10B
`pp1024=499.75`. Post-commit build-clean anchors then landed at 27B
`pp4096=213.80` and `pp16384=200.16`, the first clean dense long-context rows above
the recent llama.cpp anchors in the roadmap. Updated read: dense attention body is
no longer the obvious lcpp-scale residual. The next exact dense work should pivot to
GDN step state/layout and remaining FFN/GDN projection deltas, while continuing only
tiny isolated matrix-body cleanups.

## 2026-05-31 — Dense G6 Matrix KQ/KQV Pointer Hoist

Status: exact dense 27B G6 matrix-attention indexing cleanup. Raw artifacts are
in `target/profiles/v0163-*pointer-hoist*` and `target/profiles/v0163-clean-*`.

The existing KQ/KQV matrix kernels recomputed row/head/division and base-address
math inside the hot K loop. Hoisting the invariant K/Q/V/prob base pointers per
thread is the first llama.cpp-mechanics probe that cleared a real phase gate.
Correctness is green on the default 27B prefill-vs-single gate and the ignored
long-prefix G6 gate.

Phase impact is concentrated in attention body. At 27B `pp4096`, KQ/KQV move from
the post-causal baseline `179.26/180.01 ms` to `160.41/158.21 ms`. At `pp16384`,
they move from `2822.60/2971.57 ms` to `2516.18/2660.00 ms`. Softmax remains flat
and still much faster than llama.cpp, so the remaining dense attention deficit is
now narrower and more clearly KQ/KQV kernel mechanics, not score softmax.

Clean end-to-end rows are positive but still small/noisy because dense FFN/GDN
dominate: dirty-code 27B anchors were `pp512=234.96`, `pp1024=233.88`,
`pp4096=209.91` on rerun after an initial `207.08` noisy row, and
`pp16384=196.36`. Post-commit build-clean reruns landed at `pp4096=211.61` and
`pp16384=195.96` after an initial cooler/variance packet of `206.18/194.08`.
MoE smoke rows are neutral/noisy: A3B `pp1024=1584.12` and A10B `pp1024=497.03`,
with fast-path coverage still `40/40` and `48/48`.

Updated read: the lcpp gap is sensitive to boring integer/address lowering inside
the matrix body, not just tile shape. Continue the llama.cpp-mechanics audit with
one isolated diff at a time; do not reopen broad loop-unroll or Q-layout-only work.

## 2026-05-31 — Producer-Side Compact-Q Upper Bound Falsified

Status: dense 27B G6 matrix-attention Q-layout upper-bound probe. Source change
was removed; raw artifacts are in `target/profiles/v0162-*compact-q-producer*`.

The cheapest "free compact Q" production shape was tested without adding a runtime
pack: replace in-place Q RoPE with an out-of-place RoPE producer that writes Q into
the compact `[kv_head, token*group+g, dim]` view, reusing the old pre-norm Q scratch,
then use a compact-Q KQ kernel. This preserves correctness and avoids the previous
`~309 ms` runtime pack bill.

It still fails the phase gate. At 27B `pp4096`, compact-Q producer phase rows were
`body_matrix_kq=174.66 ms`, `body_matrix_kqv=177.58 ms`, `softmax=60.96 ms`, and
`rope_scatter=8.03 ms`. Relative to the post-causal baseline, KQ/KQV only move by a
few milliseconds while RoPE/scatter gets a small extra copy cost. That is far below
the required `~60 ms` `pp4096` attention-body movement and does not justify carrying
another sidecar or moving Q projection/norm layout.

Updated read: Q layout contributes some KQ friction, but it is not the missing
llama.cpp-scale delta by itself. Demote compact-Q layout work unless a future branch
changes the whole attention dataflow. The next exact attention bet is a mechanical
llama.cpp `mul_mat` differential or a narrow fused online-softmax/PV body; GDN-step
state/layout remains the parallel non-attention lever.

## 2026-05-31 — Selective Attention Unroll Probe Falsified

Status: dense 27B G6 matrix-attention KQ/KQV microprobe. Source change was
removed; raw artifacts are in `target/profiles/v0161-*attn-selective-unroll*` and
`target/profiles/v0161-clean-27b-*`.

The probe copied llama.cpp-style full-unroll pragmas only onto the existing matrix
KQ/KQV inner loops, leaving the outer K loop alone after the earlier broad-unroll
regression. It built and passed the default 27B prefill-vs-single correctness gate.
Dirty phase rows looked locally promising: at `pp4096`, KQ/KQV were about
`172.89/172.45 ms`; at `pp16384`, KQ/KQV were about `2722.76/2772.77 ms`.

The clean scoreboard gate did not clear. Against the post-causal-skip anchors, the
long rows were only tiny wins/noise (`pp4096` `208.06 -> 210.21 t/s`, `pp16384`
`194.80 -> 195.46 t/s`), while short contexts regressed versus the current dense
anchors (`pp512` `234.38 -> 226.48 t/s`, `pp1024` `226.63 -> 222.71 t/s`). This is
not a default-worthy trade, and an env-gated duplicate kernel is not justified for
sub-1% long-context movement.

Updated read: stop treating loop pragmas as the likely llama.cpp delta. The next
attention work should test either free/producer-side compact-Q layout, one isolated
llama.cpp `mul_mat` mechanical difference at a time, or a narrow fused online-
softmax/PV prototype. Keep GDN-step state/layout auditing as the parallel exact
scoreboard lever; do not spend a branch on chunk policy alone.

## 2026-05-29 — G6 Matrix KQ/KQV Layout Probes Falsified

Status: post-`v0.159` attention-body structural probes. Raw artifacts are in
`target/profiles/v0160-*`.

The next suspected lcpp delta was KQ/KQV layout rather than softmax: llama.cpp has
flash attention disabled on the matched profile and wins KQ/KQV with generic
`mul_mat`, while qwen's softmax is already much faster. Two narrow exact probes did
not clear the gate and were removed.

First, compacting Q into `[kvh, token*group+g, d]` before KQ improved the KQ/KQV
sub-buckets (`pp4096` KQ/KQV about `179/180 -> 170/173 ms` after vectorizing the
copy), but the required pack itself cost `~309 ms` per timed pass. The initial
scalar pack was catastrophically worse (`~4795 ms` per timed pass). This rules out
runtime compact-Q staging as a production path unless Q is produced in that layout
directly.

Second, a q-head-major score/body layout to mimic llama.cpp's per-Q-head batch
shape was correctness-safe but regressed KQ (`pp4096` `~179 -> ~196 ms`) and left
KQV flat. The current grouped-by-KV-head score layout is therefore not the obvious
KQ/KQV loss by itself.

Updated read: the remaining attention-body gap is not solved by cheap staging or
axis reorder. The next exact branch should either copy the actual llama.cpp generic
`mul_mat` kernel mechanics more directly, or jump to a fused online-softmax/PV body
with a real phase gate. Do not keep runtime compact-Q or q-head-major sidecars.

## 2026-05-29 — Default G6 Matrix Causal-Tail Skip

Status: exact dense 27B attention-body cleanup after the v0.157 attribution pass.
Raw artifacts are in `target/profiles/v0158-*`.

The G6 matrix KQ/KQV body now skips wholly future causal tiles inside the current
prefill chunk, with `QWEN_PREFILL_ATTN_MATRIX_CAUSAL_SKIP=0` as rollback. This is
scoped to the default 27B group-6 matrix path; G8/G16 matrix paths and packed MoE
attention are unchanged. Correctness is green on the default 27B prefill-vs-single
gate and the long-prefix G6 gate with the candidate path active.

Phase impact is exactly where expected. At 27B `pp4096`, `body_matrix_kq` moves
`216.15 -> 179.26 ms` and `body_matrix_kqv` moves `215.88 -> 180.01 ms`, with
softmax flat. At `pp16384`, KQ/KQV move `2937.39 -> 2822.60 ms` and `3057.11 ->
2971.57 ms`. A follow-up attempt to skip softmax future-zero writes was removed:
it saved only about `5 ms` in softmax at `pp4096` but pushed KQV backward enough
to lose the net phase gain.

End-to-end is positive but small, so this is a cleanup/default, not the final lcpp
crack: randomized A/B rows show `pp4096` `~208.55 -> ~209.58 t/s` and `pp16384`
`~194.20 -> ~195.27 t/s`; clean post-commit anchors are `pp4096=208.06` and
`pp16384=194.80` on AC power. Short prompts are neutral-to-noisy (`pp512` slightly
positive, `pp1024` within noise/slightly negative), below the `>1%` regression
guardrail. The next attention work must be structural beyond causal-tile skipping,
most likely a fused online-softmax/PV body or another shape-specific body kernel.

## 2026-05-29 — Dense Residual Attribution Points To Attention Body

Status: post-`v0.156` residual-attribution checkpoint. Raw artifacts are in
`target/profiles/v0157-*`.

`lcpp_metal_profile_summary.py` now supports `--prompt-tokens`, `--last-pass`,
and `--stats`, so llama.cpp's serialized Metal op profiles can be compared against
qwen timed-pass phase logs instead of warmup+timed totals. With that normalization,
27B `pp4096` no longer points at FFN/GDN mat-mat: qwen FFN and GDN projection rows
are parity-or-faster, while the residual named deficits are attention body
(`~115 ms`) and GDN step (`~82 ms`). At `pp16384`, the residual is distributed but
ordered as FFN mat-mat (`~1.0 s`), attention body (`~0.85 s`), and GDN step
(`~0.36 s`); GDN QKV/Z/back remain parity.

No-op ceilings agree that the remaining body work is high leverage: at `pp4096`,
base `207.5 t/s` moves to `218.4` with `QWEN_PREFILL_NOOP_ATTN_BODY=1` and
`~232-233` with `QWEN_PREFILL_NOOP_GDN_BODY=1`; at `pp16384`, base `194.35` moves
to `217.23` with attention body skipped and `215.10` with GDN body skipped. This
does not mean skip bodies in production; it means the next exact branch should name
one body sub-bucket and clear a phase gate before scoreboard testing.

The first obvious attention-body microprobe was falsified and removed. Adding
full-unroll pragmas to the existing matrix KQ/KQV small loops built and passed the
default 27B prefill-vs-single gate, but the `pp4096` phase row regressed/netted out
wrong (`body_matrix_kq` `216.15 -> 230.78 ms`, `body_matrix_kqv` `215.88 ->
210.91 ms`, softmax flat), with neighboring phases also noisier/slower. Do not
revive generic loop-unroll as the attention v2 plan.

Current recommendation: open a 27B G6 matrix-attention body v2 branch only if it
targets KQ/KQV/softmax structure directly. The strongest exact bets are a G6-shaped
body kernel or a more llama-like fused online-softmax/PV body; simple unroll,
chunk-size policy, and projection scheduling are not the next branch.

## 2026-05-29 — G6 Matrix Defaults And A10B MoE Re-Anchors

Status: post-`v0.154` re-anchor plus dense default promotion and first dense
mat-mat parity pass. Raw artifacts are in `target/profiles/v0155-*` and
`target/profiles/v0156-*` until the next bench packet is curated.

The A10B routed-MoE premise changed under a warmed trace. A diagnostic
`QWEN_PREFILL_TRACE_MOE_BUCKETS=1` path now prints per-layer routed bucket geometry
when layer phase tracing is enabled. With default warmup, the timed A10B `pp512`
pass has `routed_swiglu=371.23 ms` and `routed_down=229.29 ms`; the paired
llama.cpp Metal profile reports `ffn_moe_gate+up+GLU=428.47 ms` and
`ffn_moe_down=226.51 ms`. The earlier no-warm trace was therefore a first-pass
confound, not proof that qwen's warmed routed kernels trail llama.cpp.

The split gate/up sidecar was re-tested as the smallest llama-shaped falsifier:
separate grouped Q4 matmuls plus `silu_mul` into the existing grouped down path.
Correctness passed, but A3B `pp512` regressed slightly (`1397.82/1402.25` base vs
`1389.25/1384.96` split), so the branch was removed and remains a negative result.

Current A10B anchors now put qwen at parity-or-better against same-session
llama.cpp: `pp512` qwen `446.22` vs lcpp `440.44`, `pp1024` `488.38` vs
`435.75`, `pp4096` `480.92` vs `401.27`, and `pp16384` `393.89` vs `354.22`.
Keep A10B routed MoE frozen unless a fresh matched warm trace shows a combined
`routed_swiglu+routed_down` deficit or fast-path coverage drops below `48/48`.

Dense 27B was the real default miss: `QWEN_PREFILL_ATTN_MATRIX_G6=1` was still
env-only. It is now auto/default for the proven group-6 shape, with
`QWEN_PREFILL_ATTN_MATRIX_G6=0` as rollback and force-on strict scratch behavior
preserved. Default/rollback rows: `pp512` `230.27/228.65` vs `213.32/213.44`,
`pp4096` `202.09/204.72` vs `175.86/175.97`, and `pp16384` `190.93` vs
`136.58`. Correctness: release build, default 27B prefill-vs-single, ignored G6
prefix gate, and phase coverage showing `16/16` matrix-G6 attention layers.
Against current llama.cpp, dense 27B is still short at `pp512/4096/16384`, so the
next exact lever is dense FFN/GDN after the G6 default, not more A10B MoE.

The dense differential is now timed-only enough to act on. `prefill_phase_summary.py`
grew `--last-pass`, `--by-layer`, and `--stats`; `QWEN_PREFILL_TRACE_FFN_SUBPHASES=1`
splits dense FFN tracing into `ffn_gate`, `ffn_up`, and `ffn_swiglu` without changing
the production path. The chunk-size hypothesis was mostly falsified for dense 27B:
randomized `pp4096` and `pp16384` rows kept `prefill_chunk=1024` ahead of `2048`.

The first dense mat-mat parity pass copies llama.cpp's explicit full-unroll pragma
onto the Q4_K/Q5_K/Q6_K/Q8_0 mat-mat kernels. This is not a giant scoreboard move,
but it closes the phase-local gap that the subphase trace found: at 27B `pp4096`,
timed FFN subphases move from qwen/lcpp `4063/3878 ms` gate, `4068/3932 ms` up,
and `4199/4099 ms` down to `3959/3878`, `3964/3932`, and `4090/4099`. GDN mat-mat
buckets move similarly (`gdn_qkv` `1817 -> 1769 ms`, `gdn_z` `1088 -> 1057 ms`,
`gdn_back` `1177 -> 1135 ms`). Post-commit clean rows on AC power:
27B `pp512=234.38`, `pp1024=226.63`, `pp4096=207.34`, `pp16384=193.41`; A10B
`pp1024=499.83`; A3B `pp1024=1584.48`. Correctness is green on the default 27B
prefill-vs-single gate and ignored 27B G6 prefix gate.
The analogous GDN recurrence-loop unroll was tested and removed: the `pp4096`
phase trace was neutral/slightly worse (`gdn_step` `593.40 -> 597.17 ms`) and also
nudged neighboring mat-mat buckets the wrong way.

## 2026-05-27 — Static Quant Audit Closes 0.8B Dense Coverage

Status: dtype/shape audit harness plus primitive quant coverage after the A3B
Q6-down and A10B Q5-gate/up misses proved that one uncovered tensor class can
dominate prefill. Artifact summary:
`docs/bench/2026-05-27-v0154-quant-coverage/README.md`.

`scripts/profile/gguf_fastpath_audit.py` inspects GGUF tensor dtypes and shapes
with the local `gguf --tensors` parser and reports current prefill fast-path
eligibility for dense FFN, GDN, standard attention, MoE grouped routed FFN, and
lm-head tail.
This is a static guardrail, not a replacement for phase traces: it answers "can
this model hit the native path everywhere we think it should?" before a kernel
experiment starts. The dense/GDN/attention/lm-tail prefill predicates now accept
every dtype supported by the primitive mat-mat dispatcher: `F32`, `F16`, `BF16`,
`Q2_K`, `Q3_K`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ4_NL`, and
`IQ4_XS`.
`scripts/profile/prefill_sweep.py` now runs this audit by default and stores the
row in sweep JSON; `--require-fastpath-clean` turns any reported gap into a
pre-benchmark failure. Audit tool failures warn by default but fail under
`--require-fastpath-clean`.

The 0.8B quant sweep now says dense/GDN/attention/lm-tail prefill is fully
covered for every local 0.8B file: `F32`, `F16`, `BF16`, `Q2_K`, `Q3_K_M`,
`Q4_0`, `Q4_1`, `Q4_K_S`, `Q4_K_M`, `Q6_K`, `Q8_0`, `IQ4_NL`, `IQ4_XS`, and
mixed `UD-Q8_K_XL`. Half coverage added native `F16`/`BF16` mat-vec, mat-mat,
and embedding get-rows kernels; low-bit/legacy/IQ coverage added native
`Q2_K`/`Q3_K`/`Q4_0`/`Q4_1`/`IQ4_NL`/`IQ4_XS` mat-vec and mat-mat primitives.
These new kernels are coverage-first scalar primitives, not tuned llama.cpp
parity kernels. Paired one-row `pp128` anchors make that explicit: Q2/Q3 are
about `0.83x` llama.cpp (`932/1120`, `1202/1444 t/s`), while IQ4_NL/IQ4_XS are
about `0.54x` (`883/1643`, `896/1650 t/s`). Treat IQ4 as a future optimization
gap, not a completed performance win.

Target-family audit after the A10B Q5 fix remains clean: `A3B 40/40` grouped MoE,
`A10B 48/48` grouped MoE, and dense `27B 64/64` FFN + `48/48` GDN + `16/16`
attention. Wider local inventory shows dense 2B/4B/9B/27B Q2/Q3/Q4/Q6/Q8/BF16/
IQ4 variants clean; remaining gaps are alternate MoE expert bank quants
(`Q6_K/Q8_0` or `IQ3_*` gate/up with `IQ4_XS` down) and UD low-bit dense files
with `IQ2_*`/`IQ3_*` tensors. Validation: release `qwen-bench` builds, 0.8B F32
prefill-vs-single passes, Q5_K/Q8_0 mat-mat oracle tests pass, half/Q4/Q3/Q2/IQ4
primitive oracle tests pass, pp128 `--require-fastpath-clean` smokes pass for
`Q2_K`/`Q3_K_M`/`IQ4_NL`/`IQ4_XS`, and tg8 smokes pass for the same low-bit/IQ
0.8B models on AC power.

## 2026-05-27 — A10B Q5 Gate/Up Coverage Fix Clears Layer 46

Status: dirty-code exact dtype-coverage fix for A10B routed MoE. Raw artifacts are
in `docs/bench/2026-05-27-v0149-a10b-q5-gateup/`.

The A10B post-G16 trace had only `47/48` grouped routed MoE layers because layer
`46` uses `Q5_K/Q5_K/Q6_K` gate/up/down and fell off the grouped path. The new
grouped `Q5_K` gate/up SwiGLU kernel restores `48/48` `route_fused`,
`routed_swiglu`, `routed_down`, `routed_reduce`, and `shared_packed` coverage at
`pp512`. Auto mode is intentionally scoped to the proven A10B shape
(`hidden=3072`, `f_exp=1024`, `n_expert=256`); rollback is
`QWEN_PREFILL_MOE_GROUPED_Q5_GATEUP=0`.

Validation: the dedicated layer-46 Q5 SwiGLU oracle passes against per-expert
mat-mat (`cos=1.000000`, `max_abs=0.000e0`, `poison_count=0`), and both default
and rollback A10B prefill-vs-single smokes pass with final logits `0.999934` and
unchanged GDN/KV cosine envelopes. Warmed perf rows show `pp512` `377.53 ->
448.31 t/s` (`1.19x`), `pp1024` `400.11 -> 513.66 t/s` (`1.28x`), and `pp4096`
`440.08 -> 484.54 t/s` (`1.10x`, runs=1 directional). Discard the no-warmup rows;
they measured cold first-run residency effects, not the steady scoreboard protocol.

Read: this is the A10B analogue of the A3B Q6-down coverage miss: exact, narrow,
and high leverage. After this lands, stop revisiting attention for A10B and move
to routed down/dequant locality or a SwiGLU+down locality-preserving sidecar.

## 2026-05-26 — Post-G16 A10B Budget Pivots Back To Routed MoE

Status: clean no-op attribution after A10B/G16 matrix attention became default.
Raw artifacts are in `docs/bench/2026-05-26-v0147-a10b-post-g16-routed-budget/`.

The tight repeated `pp512` gate shows base/default G16 at `383.26`, `382.87 t/s`;
no-attention-body at `389.39`, `390.08 t/s`; and no-routed-MoE at `764.19`,
`764.46 t/s`. A broader one-block sweep agreed directionally (`no-routed 763.40`,
`no-attn 390.26`, `no-shared 402.04`, `no-gdn 446.97`, `no-FFN 1051.66`) but had a
bad late baseline (`328.50`), so do not ratio against that broad baseline.

Read: after G16 default, A10B attention body is only low-single-digit headroom at
`pp512`; routed MoE is about a `2x` no-op ceiling and owns the remaining lcpp gap.
The next optimization sprint should start from `routed_swiglu` and `routed_down`
subphase evidence, with GDN tracked as secondary.

## 2026-05-26 — A10B G16 Default/Rollback Canary Is Green

Status: clean post-commit canary after rebuilding `qwen-bench` from `0ee9c45d4`
(`build_dirty=0`). Artifact:
`docs/bench/2026-05-26-v0141-a10b-g16-matrix/v0146-clean-a10b-pp512-g16-default-rollback.json`.

At A10B `pp512`, `QWEN_PREFILL_ATTN_MATRIX_G16=0` rolls back to `365.14 t/s`,
while auto/default with no env reaches `379.23 t/s`. Power source was AC, no
thermal/performance/CPU-power warnings were recorded, and memory stayed at `95%`
free before/after each row.

Read: the default and rollback wiring works on the most conservative swept prompt
shape. A10B attention should stop being the active branch; the next optimization
frontier is routed MoE, especially because `pp512` still trails llama.cpp despite
the G16 default win over qwen base.

## 2026-05-26 — A10B G16 Matrix Attention Promotes To Default

Status: code promotion after `v0.145`. `QWEN_PREFILL_ATTN_MATRIX_G16` now uses the
same env-mode policy as A3B/G8: auto/default may use matrix attention for the
proven A10B group-16 prompt shape, `QWEN_PREFILL_ATTN_MATRIX_G16=0` rolls back to
packed attention, and `=1` force-enables strict matrix behavior.

Validation: `cargo fmt --check` and release `qwen-bench` rebuild passed. Active
G16 matrix correctness passed without setting `QWEN_PREFILL_ATTN_MATRIX_G16` by
lowering only the packed activation threshold for the smoke:
`QWEN_PREFILL_ATTN_PACKED_G16_MIN_POS=1 QWEN_PREFILL_TRACE_ATTN_PHASES=1 cargo test --release -p qwen-llm prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke -- --nocapture`.
The trace showed all `12` A10B attention layers using `prefill-attn-matrix-g16-shape`
in both chunks. Final logits were `0.999934`, GDN state `0.999810`, conv
`0.999657`, KV K `0.999567`, and KV V `0.999413`; AC power, thermal, performance,
and memory probes stayed clean.

Read: this is a narrow default, not a new universal matrix policy. It should not
change dense/G6 or other attention groups. Next gate is a clean post-commit
default-vs-rollback canary, then attention work should yield to A10B routed MoE.

## 2026-05-26 — A10B G16 Promotion Packet Clears The Qwen Default Gate

Status: clean current-build repeat packet after `v0.144`, plus paired llama.cpp
anchors. Raw artifacts are in `docs/bench/2026-05-26-v0141-a10b-g16-matrix/`.

| Shape | Qwen base rows | Qwen G16 rows | G16/base | llama.cpp default | G16/lcpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| A10B `pp512` | `367.61`, `368.77` | `382.72`, `383.36` | `+4.0%` | `427.24` | `0.90x` |
| A10B `pp1024` | `413.23`, `414.49` | `435.08`, `428.02` | `+4.3%` | `422.34` | `1.02x` |
| A10B `pp4096` | `380.16`, `373.43` | `408.67`, `402.29` | `+7.6%` | `374.29` | `1.08x` |
| A10B `pp16384` | `289.36`, `293.28` | `340.41`, `340.14` | `+16.8%` | `331.58` | `1.03x` |

The paired llama.cpp default packet reports `flash_attn=false`; an explicit
`-fa 0` packet also reports `flash_attn=false` and lands at `427.54/413.45/352.21/320.18 t/s`.
Power/thermal probes stayed clean. Memory pressure stayed warning-free; the final
llama.cpp default packet ended with `57%` free memory, likely from file cache /
residency, so do not over-read tiny lcpp deltas versus older family sweeps.

Read: G16 matrix is positive for qwen at every repeated A10B prompt size and beats
same-session llama.cpp from `pp1024` through `pp16384`. `pp512` remains a lcpp gap
despite the qwen-side win, which points back to routed MoE after defaulting G16.
The next code move is a narrow A10B/G16 auto-on policy with a rollback env, not
more attention microsearch.

## 2026-05-26 — A10B G16 Clean Coverage Trace Confirms The Mechanism

Status: clean trace after rebuilding `qwen-bench` from `b78a5d167` (`build_dirty=0`).
Artifacts are in `docs/bench/2026-05-26-v0141-a10b-g16-matrix/`.

Coverage: `12/12` A10B attention layers emit `prefill-attn-matrix-g16-shape` at
layers `3,7,11,15,19,23,27,31,35,39,43,47`. The phase summary also shows expected
fast-path counts: `47` routed MoE layers, `12` attention layers, and `36` GDN
layers. Power/thermal/memory probes stayed clean on AC power.

Attribution: matrix body phases total `14.53 ms` (`KQ 5.22`, softmax `3.39`,
`KQV 5.92`), while routed MoE is again the dominant residual bucket
(`routed_swiglu 636.17 ms`, `routed_down 340.65 ms`, `routed_reduce 7.33 ms`).
Do not use the traced `tokens/s` row as a throughput gate because phase tracing is
heavily intrusive.

Read: the clean coverage gate is no longer speculative for `pp512`; the branch is
actually running the intended G16 matrix path on all A10B attention layers. The
remaining default blockers are repeated clean family rows, optional long-prefix
coverage, and paired llama.cpp anchors.

## 2026-05-26 — A10B G16 Smoke No Longer Needs Manual Matrix Scratch

Status: code/test follow-up after `v0.142`. Multi-chunk prefill correctness tests
now allocate matrix scratch for the total/prefix position they exercise instead of
implicitly capping matrix scratch at the chunk size.

Validation: bare `QWEN_PREFILL_ATTN_MATRIX_G16=1 cargo test --release -p qwen-llm
prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke -- --nocapture`
passes on AC power with no thermal/performance/CPU-power warnings and `96%` free
memory. The smoke reports final logits `0.999934`, GDN state `0.999809`, conv
`0.999657`, KV K `0.999567`, and KV V `0.999414`. `cargo fmt --check` passed.

Read: the G16 matrix branch still remains env-only, but the immediate scratch
correctness wart is gone for the A10B smoke. The next promotion blockers are
long-prefix coverage, repeated family rows, and paired llama.cpp anchors.

## 2026-05-26 — A10B G16 Matrix Attention Becomes The Next Env Candidate

Status: clean follow-up after `v0.141` (`e70eef815`). The branch adds env-only
`QWEN_PREFILL_ATTN_MATRIX_G16=1` for the A10B group-16 attention shape. It is not
defaulted; raw artifacts are in `docs/bench/2026-05-26-v0141-a10b-g16-matrix/`.

### Measurements

Clean rows used `build_dirty=0`, AC power, no thermal/performance/CPU-power
warnings, and `96%` free memory before/after:

| Shape | Base | G16 matrix | Read |
| --- | ---: | ---: | --- |
| A10B `pp1024` | `411.86` | `430.67` | `+4.6%` |
| A10B `pp16384` | `289.97` | `338.07` | `+16.6%` |

Warmed dirty spikes also pointed the same way: `pp512` `361.09 -> 379.78`,
`pp1024` `410.32 -> 428.08`, `pp4096` `376.47 -> 404.63`, and `pp16384`
`281.09 -> 329.29`. A chunk-4096 spike helped the packed base at `pp4096`
(`376.47 -> 403.52`) but left G16 matrix still ahead (`412.26`); at `pp16384`,
G16 matrix was essentially chunk-flat (`329.29 -> 329.68`) while base stayed far
behind (`293.72`).

Dirty phase trace at `pp512` showed `12/12` G16 matrix attention layers. Matrix
body phases totaled about `14.3 ms` (`KQ 5.46`, softmax `3.11`, `KQV 5.76`),
versus the prior packed-attention body bucket around `73.21 ms`; routed MoE still
dominated (`routed_swiglu 585.07 ms`, `routed_down 317.90 ms`).

Correctness: initial bare `QWEN_PREFILL_ATTN_MATRIX_G16=1` A10B smoke hit a
test-path scratch allocation error (`max_pos=4 < required last_pos=6`). A follow-
up test-scratch fix now makes the same bare-env smoke pass without
`QWEN_PREFILL_ATTN_MATRIX_MAX_POS`: final logits `0.999934`, GDN state `0.999809`,
conv `0.999657`, KV K `0.999567`, KV V `0.999414`.

Read: A10B attention is no longer just a packed-path tuning problem. The G16
matrix branch is now the highest-EV A10B promotion candidate, especially for long
contexts, but it remains env-only until repeated clean rows, clean trace coverage,
and paired llama.cpp anchors agree. After this branch, the remaining A10B gap
should be re-attributed before more attention work.

## 2026-05-26 — A3B Matrix Attention Promotes To Default

Status: dirty-code promotion after `v0.136`. The group-8 matrix-attention path is
now auto-on for the proven A3B attention shape, with
`QWEN_PREFILL_ATTN_MATRIX_G8=0` as rollback. Dense group-6 matrix attention stays
env-only. Raw rows are in `docs/bench/2026-05-26-v0137-a3b-matrix-default/`.

### Measurements

A3B current HEAD, sequential AC-power rows, base packed attention vs matrix:

| Shape | Packed/default rows | Matrix rows | Read |
| --- | ---: | ---: | --- |
| `pp128` | `671.33`, `676.27` | `726.81`, `719.09` | `~+7%` |
| `pp512` | `1095.23`, `1099.98` | `1249.47`, `1233.00` | `~+13%` |
| `pp1024` | `1239.69`, `1241.49` | `1421.72`, `1433.08` | `~+15%` |
| `pp4096` | `1192.95`, `1191.45` | `1411.38`, `1406.32` | `~+18%` |
| `pp16384` | `890.94`, `905.04` | `1185.98`, `1121.63` | `~+25-33%` |
| real `v02_reva` `34.5k` | `655.34` | `837.18` | `~+28%` |

Post-commit auto/rollback canary at `pp512`: clean default auto rows are
`1245.21`, `1235.93`, while `QWEN_PREFILL_ATTN_MATRIX_G8=0` rolls back to
`1094.59`, `1093.84`.

Correctness/coverage: the full ignored A3B prefill-vs-single gate passed with
matrix auto/default, including prefix `4096` / `8191` active shapes; worst active
row was prefix `4096`, `T=8`, `P=8` with logits `0.999984` and GDN min `0.999611`.
Trace-label coverage at default `pp512` showed `10/10` matrix attention layers and
`40/40` route, grouped routed, and shared MoE labels. Power/thermal/memory probes
stayed clean.

Read: this clears the matrix production gate for A3B. The local matrix oracle keeps
its documented numeric envelope (`cos >= 0.9999`, `max_abs <= 2e-2`), while the
production gate is prefill-vs-single model-state equivalence. Auto mode falls back
to packed attention when scratch is undersized; force-on remains strict.

## 2026-05-26 — RMSNorm Vec4 Falsifier Does Not Move A3B Matrix Prefill

Status: dirty-code negative spike after `v0.135`, stripped before commit. Raw
artifacts are in `docs/bench/2026-05-26-v0136-rmsnorm-vec4-negative/`.

### Measurements

A3B matrix `pp4096`, default GDN skinny active in both variants:

| Variant | Rows | Read |
| --- | ---: | --- |
| matrix | `1408.70`, `1409.10` | baseline |
| matrix + `QWEN_PREFILL_RMSNORM_VEC4=1` | `1408.64`, `1407.07` | flat/slightly negative |

Phase tracing also stayed flat: GDN `pre_norm` moved only `957.96 -> 952.99 ms`,
and attn `pre_norm` only `328.00 -> 324.51 ms`. Correctness passed 0.8B and A3B
prefill-vs-single with the spike enabled.

Read: the A3B phase trace makes `pre_norm` look large, but simple float4 spelling
is not the missing lcpp mechanism. Do not carry the env branch. If norm remains a
suspect, require either fused norm+projection, a direct llama norm-node
differential, or a trace-methodology fix before more local RMSNorm kernel work.

## 2026-05-26 — GDN Skinny E8xP32 Promotes To Default

Status: clean follow-up after `v0.134`. Promoted the dense GDN skinny E8xP32 path
to default-on for eligible F32 prompt-prefill `beta_proj` / `alpha_proj`
projections, with `QWEN_PREFILL_GDN_SKINNY_E8P32=0` as rollback. Raw canaries are
in `docs/bench/2026-05-26-v0135-gdn-skinny-default/`.

### Measurements

Shape audit: Qwen3.6 27B has F32 `[5120,48]` alpha/beta; Qwen3.6 35B A3B has F32
`[2048,32]`; sampled Qwen3.5 0.8B/4B/9B/27B and 122B A10B use Q8_0 alpha/beta
and fall back.

Additional repeated canaries:

| Model / branch | Shape | Baseline rows | Skinny rows | Read |
| --- | ---: | ---: | ---: | --- |
| A3B default | `pp128` | `642.32`, `663.98` | `678.86`, `675.24` | positive |
| A3B default | `pp1024` | `1226.11`, `1226.55` | `1237.35`, `1240.68` | positive |
| A3B matrix-G8 | `pp4096` | `1386.19`, `1386.24` | `1407.17`, `1410.67` | positive |
| 27B matrix-G6/G8 | `pp128` | `193.80`, `193.79` | `199.08`, `198.87` | positive |

Correctness: A3B prefill-vs-single with skinny enabled passed at `T=12/P=8`
with final logits `0.999985`, GDN state `0.999735`, conv `0.999811`, and KV K/V
`>=0.999873`. This is lower than the 27B cosines but above the established gate.
Power/thermal/memory probes stayed clean.

Read: defaulting is now justified because the fast path is callsite-scoped to GDN
alpha/beta prompt prefill, both eligible F32 shape families are positive, and
non-F32 sampled families fall back. Monitor future GGUFs with new F32 GDN shapes
and use `QWEN_PREFILL_GDN_SKINNY_E8P32=0` for rollback.

## 2026-05-25 — GDN Skinny E8xP32 Finds A Dense Projection Mismatch

Status: dirty-code env-only spike after `v0.133`. Added
`QWEN_PREFILL_GDN_SKINNY_E8P32=1`, which routes eligible F32 dense GDN
`beta_proj` and `alpha_proj` skinny projections through the existing E8xP32
router mat-mat kernel instead of the generic dense dispatcher. Raw rows are in
`docs/bench/2026-05-25-v0134-gdn-skinny-e8p32/`.

### Measurements

27B dense with matrix-G6/G8 enabled:

| Shape | Baseline rows | Skinny rows | Read |
| --- | ---: | ---: | --- |
| `pp512` | `212.31`, `212.93` | `216.34`, `218.34` | `~+2-3%` |
| `pp1024` | `205.48`, `200.13` | `213.56`, `208.98` | `~+4%` |
| `pp4096` | `176.29`, `191.45` | `195.25`, `193.02` | cold first baseline; warmed `~+0.8%` |
| `pp8192` | `185.67`, `185.64` | `191.09`, `190.65` | `~+2.7%` |
| `pp16384` | `175.48`, `176.08` | `180.34`, `178.87` | `~+1.6-2.8%` |

Phase trace attribution at `pp4096` localizes the mechanism:
`gdn_beta_alpha` drops from `672.45 ms` to `105.96 ms`, while `gdn_qkv` and
`gdn_z` are roughly unchanged. Correctness passed 0.8B prefill-vs-single, 27B
prefill-vs-single with matrix-G6/G8 at `T=32/P=32`, and the ignored 27B
matrix-G6 prefix gate at prefix `4096`.

Read: this is a real dispatch-class mismatch, not another fused-SwiGLU mirage.
Keep it env-only for this checkpoint because the total gain is single-digit and
the shape predicate must stay semantically tight before default-on. The next
dense inspection should search for other F32 skinny projections using the wrong
dispatcher, then run a narrow promotion gate rather than a broad benchmark matrix.

## 2026-05-25 — Dense llama.cpp Differential Moves Focus Off SwiGLU

Status: dirty-code attribution after `v0.132`. Added a llama.cpp Metal profile
summary parser and split qwen dense FFN tracing into norm, gate/up/SwiGLU, and
down/residual buckets. Raw summaries are in
`docs/bench/2026-05-25-v0133-dense-lcpp-diff/`.

### Measurements

Serialized attribution profiles at dense 27B `pp4096`:

| Bucket | qwen ms | llama.cpp ms | Read |
| --- | ---: | ---: | --- |
| FFN gate/up/SwiGLU | `8408.19` | `8106.38` | qwen `~3.7%` slower |
| FFN down/residual | `4309.56` | `4199.97` | qwen `~2.6%` slower |
| GDN front projections | `3612.91` | `2997.76` | qwen `~20%` slower |
| Attention KQ/KQV/softmax | `498.99` | `371.30` | qwen still slower in matrix body |

Read: this explains why dense fused-SwiGLU kept failing total gates. The FFN
mat-mat delta exists but is not the dominant local differential under serialized
profiling. The next dense inspection should focus on GDN front projection lowering
and the remaining attention-body delta, with end-to-end gates before any default.

## 2026-05-25 — Same-Process FFN A/B Keeps Fused SwiGLU Env-Only

Status: dirty-code harness after `v0.131`. Added a runtime override plus hidden
`qwen-bench pp-ffn-ab` command so dense fused-Q4 SwiGLU can be alternated inside
one loaded process instead of comparing separate process rows. Raw artifacts are
in `docs/bench/2026-05-25-v0132-ffn-ab/`.

### Measurements

27B dense with matrix-G6/G8 enabled, base/fused warmup, then alternating pairs:

| Shape | Base rows | Fused rows | Read |
| --- | ---: | ---: | --- |
| `pp4096` | `179.14`, `187.50` | `177.88`, `185.19` | fused loses both orderings |
| `pp8192` | `185.70`, `182.90` | `180.65`, `177.04` | fused loses both orderings |
| `pp16384-a` | `178.25`, `183.84` | `195.67`, `194.18` | apparent large win, suspicious |
| `pp16384-b` | `174.82`, `170.10` | `174.84`, `170.39` | repeat is flat |

Read: `QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=1` remains env-only. The new
same-process harness is a keeper, but the candidate still fails default gates:
`pp4096/8192` regress and the large `pp16384` win did not reproduce. The paired
phase trace also showed unrelated phases moving with the same drift as FFN, so a
future long-context fused branch needs a repeatable phase-local mechanism.

## 2026-05-25 — Interleaved Dense FFN Rows Keep Candidates Env-Only

Status: clean `v0.129` follow-up using the new `prefill_sweep.py`
`--repeat-blocks` / `--shuffle-seed` controls. Raw rows are in
`docs/bench/2026-05-25-v0129-drift-controlled-ffn/`.

### Measurements

27B dense with matrix-G6/G8 enabled, sequential on AC power:

| Shape | Variant rows | Read |
| --- | --- | --- |
| `pp4096` | block 1 warmed rows: `g6=208.96`, `g6-smem=209.73`, `g6-fused=210.98`, `g6-fused-smem=211.09` | fused/smem at most `~1%`; below promote gate |
| `pp16384` | `g6=193.46/181.14`, `g6-fused=189.70/193.60` | directionally inconsistent; drift dominates |

Read: the harness exposed exactly the confound it was built for. The first
`pp4096` baseline row was a cold/process outlier (`193.36 t/s`) while the repeat
baseline was `208.96 t/s`; the last `pp16384` baseline also collapsed. Keep
`QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=1` and
`QWEN_MATMAT_QK_LLAMA_SMEM=1` env-only. The next dense lcpp-cracking move should
be phase-local FFN/GDN timing and same-process paired evidence, not defaulting a
small total-throughput artifact.

## 2026-05-25 — Clean Smem A/B Flips Reduced-Smem To Opt-In

Status: clean `v0.127` follow-up after the dirty mat-mat smem parity spike.
`qwen-bench` was rebuilt from `fcab600`, then matrix-G6 27B dense prompt sweeps
compared the new reduced-smem policy against `QWEN_MATMAT_QK_LEGACY_SMEM=1`.
Clean end-to-end rows did not prove a default win, so the code now keeps legacy
`8192`-byte mat-mat threadgroup-memory requests by default and exposes the
llama-style full-tile policy only through `QWEN_MATMAT_QK_LLAMA_SMEM=1`. Raw rows
are in `docs/bench/2026-05-25-matmat-smem-clean-v0127/`.

### Measurements

27B dense with matrix-G6/G8 enabled, `runs=3`, clean rebuilt binary:

| Shape | smem-new A | legacy | smem-new B | Read |
| --- | ---: | ---: | ---: | --- |
| `pp1024` | `221.09` | `210.26` | `202.98` | high drift; no robust win |
| `pp4096` | `189.00` | `204.65` | `204.51` | first new run bad, second equals legacy |
| `pp16384` | `190.83` | `191.68` | `188.44` | flat/slightly worse |

Read: the reduced-smem policy is microbench-positive but system-level unproven.
Defaulting it would be optimizing from kernel intuition rather than end-to-end
evidence, so it is now opt-in only. The bigger lesson is measurement hygiene:
run-order drift is large enough that fresh qwen-vs-lcpp claims need interleaved
or repeated anchors.

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
