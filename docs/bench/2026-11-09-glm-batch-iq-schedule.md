# GLM batch routed-expert schedule: two-simdgroup geometry survives, decoder rewrites do not

Whole-round screens of the B=2..4 GLM routed IQ2_S gate/up (K=4096) and IQ3_S
down (K=2048) kernels on the M5 Ultra. The shipped default runs four
simdgroups per threadgroup. This note records the candidates and the decision;
it is the evidence behind the `QWEN_GLM53_IQ_TG64` default.

## Protocol

- Stock `UD-IQ3_XXS` artifact, one immutable binary per screen, canonical Metal
  lease, builds at `nice -n 10` outside the lease.
- One process captures a 64-round warm prefix, snapshots it, then measures each
  variant in the order forward, reverse, reverse, forward with 16 timed rounds
  per arm. Every variant therefore has the same mean arm position, and the two
  halves give the two orders. `scripts/research/glm-batch-next/summarize_iq.py`
  reports the drift-balanced mean and the per-order split.
- Thermal pressure is sampled per arm. Any level-2 acquisition is rejected
  before a probe launches and retried after cooling; a level-2 timed arm would
  be redone. All retained arms below were level 0 at every timer sample.
- Arithmetic-preserving variants (threadgroup width) are accepted only when
  their 16-step, full-vocabulary teacher-forced witness is bitwise equal to the
  shipped arm. Reordered-arithmetic variants are scored by signed dNLL and KL
  against the shipped distributions; the same-run repeat witness has zero
  differing bits, so any signed dNLL is a real arithmetic change.

## Kills

The preserved WIP `sb`/`sx` sign decoders and two fresh decoder rewrites were
screened and removed.

| 8K B=4 variant | round ms | saved vs shipped | fwd / rev | witness |
| --- | ---: | ---: | ---: | --- |
| shipped | 31.555 | 0 | 0 / 0 | reference |
| `tg64` (2 simdgroups) | 31.337 | **+0.218** | +0.097 / +0.339 | bitwise |
| `sb` (byte-mask sign) | 32.084 | -0.529 | -0.712 / -0.346 | bitwise |
| `sb64` (`sb` + 2 simdgroups) | 31.992 | -0.437 | -0.508 / -0.366 | bitwise |
| `sx` (sign-bit flip) | 34.188 | -2.633 | -2.643 / -2.624 | bitwise |

`sb`/`sx` replace the shipped `select`-based sign handling with byte-mask and
sign-bit manipulation. They produce identical coefficient bits but cost 0.4-2.6
ms more, so the sign arithmetic is not the limiter.

The vectorized `v4` decoder gave each lane 64 consecutive weights and loaded the
four sub-blocks' control bytes with wide (8/2-byte) fetches. It changes K-order.

| 8K B=4 variant | round ms | saved vs shipped | fwd / rev | witness |
| --- | ---: | ---: | ---: | --- |
| shipped | 32.264 | 0 | 0 / 0 | reference |
| `tg64` | 32.191 | +0.073 | +0.196 / -0.050 | bitwise |
| `v4` (4-simdgroup) | 39.355 | **-7.091** | -7.014 / -7.167 | reordered |
| `v4tg64` | 39.040 | **-6.776** | -6.759 / -6.793 | reordered |

`v4` is 6.8-7.1 ms slower. With four lanes per superblock, each load
instruction now scatters 2-byte pieces across a 32-byte sector instead of one
contiguous half-warp run, so the DRAM request count rises. The shipped lane map
is already well coalesced; widening the per-lane fetch without keeping the
cross-lane contiguity loses far more than the removed instructions gain.

The `vf` decoder kept the shipped lane map and decoder loads (identical
coalescing and coefficient bits) but accumulated each 16-weight fragment into
one `float4` instead of a scalar FMA chain, changing only summation order.

| 8K B=4 variant | round ms | saved vs shipped | fwd / rev | dNLL max/mean | KL max/mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| shipped | 31.685 | 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `tg64` | 31.534 | **+0.150** | +0.125 / +0.176 | 0 / 0 | 0 / 0 |
| `vftg64` | 31.851 | -0.167 | -0.182 / -0.151 | 1.19e-2 / 3.02e-4 | 1.46e-4 / 2.92e-6 |
| `vf` | 31.959 | -0.275 | -0.347 / -0.203 | 1.19e-2 / 3.02e-4 | 1.46e-4 / 2.92e-6 |

`vf` is slower and its signed dNLL reaches 1.2e-2, so it is rejected on both
cost and accuracy. Together with `sb`/`sx` this says the per-channel FMA chain
is not the exposed limiter either: the round is dominated by weight streaming,
and the shipped scalar chain is hidden under it.

## The surviving schedule

The two-simdgroup (`tg64`) instances keep the shipped decoder, lane-to-fragment
map, F32 chain and SIMD reduction; only the threadgroup is narrower (64 threads,
four channels per group for gate/up and eight for down instead of 8/16). Outputs
are bitwise equal.

| Cell | shipped ms | tg64 ms | saved ms | fwd / rev | witness |
| --- | ---: | ---: | ---: | ---: | --- |
| 8K B=4 | 31.555 | 31.337 | +0.218 | +0.097 / +0.339 | bitwise |
| 8K B=4 (repeat) | 32.264 | 32.191 | +0.073 | +0.196 / -0.050 | bitwise |
| 32K B=4 | 32.470 | 32.311 | +0.160 | +0.130 / +0.190 | bitwise |
| 8K B=3 | 27.237 | 27.162 | +0.075 | -0.078 / +0.228 | bitwise |
| 32K B=3 | 27.327 | 27.286 | +0.041 | +0.022 / +0.060 | bitwise |
| 8K B=2 | 22.888 | 22.788 | +0.100 | +0.096 / +0.105 | bitwise |

Every cell is non-negative in the drift-balanced mean; 10 of 12 order-halves are
positive. The effect is small (about **0.2% of a B=3 round** and **0.5% of a
B=4 round**) but consistent across width and depth. At the measured GLM
pass-through of about 1.0 this is a correspondingly small delivered gain, not a
step change.

## Why the "excess over the useful-byte floor" is mostly not recoverable

Pricing earlier on this branch put the two IQ groups' marginal (wave-held) cost
at 7.478 + 5.326 ms against 5.710 + 3.862 ms of unique-weight floor at the
1.175 TB/s loads-only anchor, i.e. about 3.2 ms of "excess". The screens above
bound what is actually reachable with this artifact and bank layout:

- The `dalu` diagnostic (identical instruction stream, weights re-read from a
  cache-resident window) saved only 0.75-0.85 ms per B=4 round. Removing all
  full-range weight traffic therefore exposes no more than about 0.8 ms of
  wall time; the rest of the group's marginal cost is arithmetic and issue that
  overlaps it.
- Narrowing the arithmetic (`sb`/`sx`/`vf`) does not pay, and the one rewrite
  that also changed the load pattern (`v4`) lost 7 ms. The shipped access is
  already a contiguous per-row stream with full sector utilization.

The 1.175 TB/s anchor is a large-vector read, not a ceiling this 82/110-byte
superblock stride and dequant work can reach. Read as a schedule-practical
floor, the remaining IQ headroom is on the order of 1 ms, not 3 ms, and the
screens spent it without a larger win.

## Ship decision

`tg64` ships default-on for the two GLM batch shapes; rollback
**`QWEN_GLM53_IQ_TG64=0`** restores the shipped four-simdgroup kernels. The
feature-gated `glm_batch_iq` example retains a two-arm (`shipped`, `tg64`)
screen for future re-checks. `sb`, `sb64`, `sx`, `v4`, `v4tg64` and `vf` are
deleted; their measurements are above.

## Provenance

- Screen source for the kills: `2bef0a945` (geometry screen) plus the WIP
  decoder commit `86c112bd7`; the vectorized screens and their removal are on
  `perf/glm-batch-28`.
- Reproduce on M5: `nice -n 10 cargo build --offline --locked --release
  -p qwen-llm --features perf-ablations --example glm_batch_iq`, then
  `python3 scripts/research/glm-batch-next/run.py --model "$MODEL" --out "$RUN"
  --depth DEPTH --example glm_batch_iq --mode all --quality 16 --cohort
  distinct|forked --variants 0,1 --width N`, and summarize the `probe.jsonl`
  with `summarize_iq.py`.
- Raw packets: `~/qwen-runs/2026-11-09-glm-batch-28/{iq2-8k-b4,iqv4-8k-b4,
  iqvf-8k-b4,iq-32k-b4-confirm,iq-8k-b3-confirm,iq-32k-b3-confirm,
  iq-8k-b2-confirm}` on the M5. The 32K B=3 packet contains two rejected hot
  acquisitions (level 2 then level 1) before its retained level-0 arm.
