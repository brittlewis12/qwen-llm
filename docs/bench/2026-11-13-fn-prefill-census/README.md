# Flash-Next prefill census and the wide expert band, 2026-11-13

Status: **one shipped lever** (64-slot routed gate/up at 1024..=2047 rows),
bitwise and paired. The census is measurement only. A larger packed-command
candidate was built but not qualified; its code is not on the ship branch.

## Cohort and cells

- Artifact: `ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S`, M5 Ultra
  (`metagross`), `--profile iter` for probes, release for serve.
- Geometry read from the GGUF: 48 layers, hidden 2560, 512 experts, top-10,
  routed/shared intermediate 640, GDN 16 key / 48 value heads at 128,
  QSA 4 query / 1 KV head at 128, full attention 24/2 heads at 256,
  HC 4 branches at rank 320.
- Cells: 48,192-token cold root; appends of 256 / 1,000 / 1,024 / 1,383 /
  2,000 rows at 48K; a 2,048-row continuation chunk at 48K.
- Probe: `scripts/research/fn-prefill-census/` (scratch seams installed by
  `prepare.py` in an isolated worktree). Native walls are ordinary packed
  commands; dispatches come from the in-tree `dispatch_profile`.
- The staged packed profiler (`prefill_continuation_profiled`) fell back to
  `encoder_stage` sampling and then failed with "resolved timestamps are not
  globally monotonic" on this OS/tier. The per-dispatch census was used
  instead. Its encoder split perturbs wall by about +3%; kernel shares are
  interval cores, not exclusive hardware attribution.

## Native append scaling

One packed command at 48K depth, three retained samples per cell (mean shown).
The 1,000 and 1,383-row cells are from the pre-change binary; 2,000 native
wall is the paired control below.

| Rows | GPU ms | Wall ms | GPU ms / row |
| ---: | ---: | ---: | ---: |
| 1,000 | 602.1 | 608.6 | 0.602 |
| 1,383 | 790.0 | 796.5 | 0.571 |
| 2,048 | 1,043.1 | 1,051.7 | 0.509 |
| 2,000 | 1,102.4 | 1,110.6 | 0.551 |

The 1,000-row cell costs **18.2%** more per row than 2,048. A linear fit over
1,383-2,048 leaves a fixed component of about 260 ms: at this row count almost
every expert is touched, so the routed banks are read and decoded almost once
per layer regardless of rows. That fixed bank cost, not a launch gap, is the
small-append floor.

## 48K chunk census

Cold 48K root, baseline packed capacity 2,048: **26.774 s wall / 26.243 s GPU**
over **25 commands** (23 full 2,048-row commands, one 3-row dense-frontier
command, one more full command), logits hash
`e2b2f664d2162320...`. The 3-row frontier command is a planner artifact, not
a candidate by itself.

One 2,048-row continuation at 48K, per-dispatch interval cores (profiled wall
1,085 ms; native 1,052 ms):

| Group | ms | Share | Useful TFLOP/s | GB/s | Byte roof ms | Distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QSA total | 336.6 | 31.0% | ~0.025 (attention) | 1.4 (weights) | 0.4 | latency-bound |
| Routed gate/up | 136.3 | 12.6% | 0.99 | 218 | 25.3 | **5.4x** |
| Routed down | 108.7 | 10.0% | 0.62 | 189 | 17.5 | **6.2x** |
| Dense mixer | 215.9 | 19.9% | — | — | — | — |
| HC | 83.4 | 7.7% | — | — | — | — |
| GDN | 68.6 | 6.3% | — | — | — | — |
| Router | 36.6 | 3.4% | — | — | — | — |
| Other | 51.6 | 4.8% | — | — | — | — |

Top kernels: `qsa_selected_value_vec8_f32` 140.0 ms and
`qsa_logits_reuse4_f32` 121.8 ms; `mat_mat_q6_K_f32_tensor_wide` 101.3 ms;
`moe_down_iq4_nl_..._tensor` 87.5 ms; `moe_wide_probe_iq2_s/iq3_xxs/iq3_s`
136.3 ms; `mat_mat_bf16_f32` 56.5 ms; `hc_bf16_device` 34.5 ms.

The routed family is **5-6x above its byte roof** and only ~1 TFLOP/s: the
limiter is per-weight IQ decode/ALU, not HBM. QSA selected value/logits are the
largest single consumer and are far off any roof because of gather/selection
latency; that is the next numerical-change candidate, not a bitwise one.

## Shipped lever: wide gate/up in 1,024–2,047 rows

Before this change the 64-slot gate/up tile was qualified only at exactly
2,048 rows. A 1,383-2,047-row append has ~27-40 slots per expert, so most
buckets need a second 32-slot panel and a second weight-dequant pass.

Change: admit the same 64-slot tile at 1024..=2047 rows. Operands, F32 sums,
and dead-scratch reuse are unchanged. Rollback:
**`QWEN_FN_PREFILL_EXPERT_WIDE_SHORT=0`**; the original
`QWEN_FN_PREFILL_EXPERT_WIDE=0` still disables every wide route.

Paired interleaved A/B, three pairs, 48K depth, thermal 0 throughout, same
binary and iter profile:

| Rows | Off wall ms (3 pairs) | On wall ms | Paired change | Endpoint logits |
| ---: | ---: | ---: | ---: | --- |
| 1,000 | 604.5 / 604.0 / 604.6 | 601.3 / 601.6 / 601.9 | **-0.46%** | bitwise |
| 1,383 | 795.6 / 795.0 / 795.2 | 780.9 / 781.7 / 781.4 | **-1.75%** | bitwise |
| 2,000 | 1,111.0 / 1,110.3 / 1,110.5 | 1,076.3 / 1,076.7 / 1,077.4 | **-3.04%** | bitwise |

Dispatch census at 2,000 rows prices the isolated gate/up at
**175.2 -> 139.0 ms (1.26x)**; the whole routed family is 1.14x because the
down projection is unchanged. At 1,000 rows the isolated gate/up is 1.05x and
the wall win is 0.46%; the default band starts at 1,024 so tiny appends keep
the 32-slot route.

The 256-row and 1,024-row cells requested by the brief: 256 rows stays on the
narrow route (measured 232-240 ms at 48K before this change); 1,024 is inside
the shipped band and is bracketed by the 1,000-row bitwise/neutral result.

## Three-agent serve screen

Release `qwen`/`qwen-tok` built from the ship tree (`qwen` sha256
`cf3fd872494001ef5446fc5b28de66123d14967bcc307e1cd30b0dbdf9497329`), 48K
warm roots, 120 s windows, real FN call plan, seeded agent sampling, thermal
settle 60 s at <=1 before load, ceiling 2 in-window. Two order-controlled
pairs, not the three-pair gate:

| Pair | Off tok/s | On tok/s | Ratio | Off/On thermal |
| --- | ---: | ---: | ---: | --- |
| off -> on | 126.658 | 130.325 | 1.0289 | 0 / 0 |
| on -> off | 129.750 | 129.275 | 0.9963 | 0 / 2 (flagged) |

TTFT medians in the clean pair moved 1.629/0.894/0.932 -> 1.379/0.856/0.883 s;
in the reversed pair they were flat (1.384/0.872/0.894 -> 1.389/0.864/0.891 s).
Gap p99 was 136.7 -> 134.1 ms in the clean pair, 0.4 -> 0.32 s worst gap.

**Serve is neutral/inconclusive, not a win.** The clean pair's +2.9% is order
and clock, not the lever: the reversed pair is -0.4% with the candidate arm at
thermal 2. The plan has only **2 of 25** calls in the 1024-2047-row band, so
the real mix cannot resolve a 3% per-cell change. Larger appends (23,603 rows)
only touch the band in their final tail. The stage gate (1.26x) and the
three-pair targeted wall gate are the evidence for this lever; serve needs a
cohort enriched in 1024-2047-row appends before it can be called.

## Parked: larger packed commands

An opt-in `QWEN_FN_PACKED_ROWS` path (4,096-row commands, QSA packed pans
relaxed from the indexer budget to the session capacity) was implemented and
built. It was **not** qualified:

- The 48K planner splits at the 2,051-token dense frontier, so a 4,096-row
  command sequence starts later commands at a different offset and changes
  the GDN chunk boundaries. That is a reduction-order change, not bitwise.
- The 4,096-row append A/B was not completed; the run's dispatch allocation
  failed before the second arm. No wall, quality, or serve claim is made.
- The code was removed from the ship branch. Reopening it requires the paired
  non-inferiority protocol and a command-boundary plan that keeps the GDN
  chunk grid aligned, or a sorted expert-major kernel that reuses a decoded
  slab across row batches.

## Reproduction and limits

```sh
# isolated scratch checkout only
python3 scripts/research/fn-prefill-census/prepare.py
nice -n 10 cargo build --offline --profile iter -p qwen-llm --example fn_prefill_census
# under the canonical Metal lease, normal priority for measurement
FN_CENSUS_DEPTHS=49152 FN_CENSUS_SIZES=1000,1383,2048 FN_CENSUS_SKIP_ROOT=1 \
FN_CENSUS_REPS=3 FN_CENSUS_DISPATCH=1 FN_CENSUS_STAGED=0 \
  target/iter/examples/fn_prefill_census MODEL CORPUS > census.jsonl
python3 scripts/research/fn-prefill-census/analyze_dispatch.py census.jsonl \
  --cell append --rows 2000
```

Raw dispatch/native JSONL and logs are under
`metagross:~/qwen-runs/2026-11-12-fn-prefill/`. The 48K root and chunk
baselines are from the pre-ship binary; the wide A/B and its dispatch pair
are from the ship source. Thermal pressure was 0 in every retained arm. The
routed byte accounting uses the actual GGUF bank bytes from the probe
inventory; activation traffic is omitted from the roof, so the distance is
slightly optimistic. QSA attention FLOPs use 2,048 selected keys per query and
4 heads; they are a diagnostic, not a lower bound on the gather cost.

Aven: no connector was available from this session, so no note was written on
`SRV-2K0Y`.
