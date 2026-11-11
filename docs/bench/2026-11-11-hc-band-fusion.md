# HC/mHC small-band map and Flash-Next fusion price

This work starts from `2aa654ed7`, the 2026-11-11 solo census. The 2026-11-06
Flash-Next, GLM, and V4.1 census entries all describe the same structural
pattern: small control/projection chains are low-utilization and several
dependent launches sit on the critical schedule. The 11/11 ranking prices the
32K exposed gaps as Flash-Next HC **1.54 ms**, GLM mHC **1.08 ms**, and V4.1
HC plus posts **1.07 ms**. Source-level kernel attribution below is paired with
the M5 B=1 graph/profile captures in `~/qwen-runs/2026-11-11-hc-fuse/`.
The probe code described here is on `perf/hc-band-fusion` at `bead0f296`;
only this record and its PERF-LOG entry are landed. The product hooks, the
research kernel, the `fn_hc_band_probe` example, and the `QWEN_FN_HC_INJECT_NORM`
switch stay on that commit and are not on `main`.

## Flash-Next HC, B=1

Released geometry is four 2560-wide streams and rank 320. Each read site has
this dependency chain in `qwen4exp_metal.rs::encode_read`; the post block then
injects its output into the residual before the next site.

| Order | Dispatch | Inputs → outputs | Logical tensor bytes | Dependency / merge status |
| --- | --- | --- | ---: | --- |
| 1 | `kernel_qwen4exp_hc_rms_norm_f32` | residual `[4,2560]` + F32 norm `[4,2560]` → normalized `[4,2560]` | 40,960 + 40,960 → 40,960 | Per-branch RMS reduction. Its output feeds both projections. |
| 2 | `kernel_qwen4exp_hc_down_silu_inject_bf16` | normalized + BF16 down `[10240,320]` + F32 injection `[10240,4]` → low `[320]` + raw injection `[4]` | 40,960 + 6,553,600 + 163,840 → 1,280 + 16 | Already merges down GEMV, branch mean/SiLU, and injection preparation. Depends on #1. |
| 3 | `kernel_qwen4exp_hc_up_mix_bf16_k320` | low + normalized + BF16 up `[320,10240]` → raw gate `[10240]` + mixed `[2560]` | 1,280 + 40,960 + 6,553,600 → 40,960 + 10,240 | Already merges up GEMV, gate sigmoid, and gated branch mean. Depends on #1 and #2. |
| 4 | GDN or routed block | mixed → block output `[2560]` | 10,240 read/write bytes plus block-specific traffic | This long model block prevents merging its input preparation with post injection. |
| 5 | `kernel_qwen4exp_hc_inject_f32` | block output + raw injection + residual → updated residual `[4,2560]` | 10,240 + 16 + 40,960 → 40,960 | One TG per branch; its result is the next HC site's norm input. |

The causal dump (`fn_solo_bounds … 32768 2 graph`) recorded 898 waves, 49
scopes, and no truncated waves. The repeating read side is norm → fused
down/SiLU/injection-prep → fused up/gated-mean. Post injection starts the next
read chain. This rules out a gate-up or down-activation fusion as a new swing;
those elementwise/reduction runs are already fused.

The design candidate is the exact same-input boundary **post injection → next
RMSNorm**. One TG per branch computes the injection update into 10 KiB of
threadgroup scratch, reduces the updated values with the existing 1024-thread
RMS order, then publishes both the updated residual and normalized output.
The kernel uses F32 control/state operands; BF16 down/up remain in their
existing fast kernels. It does not recompute the large BF16 projections or
store an extra residual copy.

The first broader experiment tried to fold RMS into the 320-row BF16 down
projection, also preparing injection. It needed 40 large TGs to recompute the
four RMS values while producing eight output rows per TG. The measured full
HC chain slowed from **46.96 µs to 56.87 µs** after removing per-element
branch division (the initial mapping was **46.50 → 75.79 µs**); the mixed
output still differed by up to `4.77e-7`. That BF16 multi-output mapping is
rejected.

The narrower injection+RMSNorm probe uses the real block-0 GGUF weights and
the full neighboring HC chain, with an F32 conversion for the BF16-stored
injection auxiliary, matching the runtime binding. Its incumbent has five
dispatches (post injection, norm, down/SiLU/injection-prep, up-mix, next
injection); the candidate has four. At thermal 0, the medians were **42.83 µs
incumbent / 41.12 µs fused (1.042×)**, with bitwise-equal residual and mixed
outputs. This isolated chain result does not clear a 1.25× gate by itself;
the causal-schedule counterfactual below decides whether the launch tail makes
the change valuable in the real token loop.

The feature-gated B=1 schedule probe (`fn_solo_bounds … 32768 32 hc-band`)
alternated candidate off/on against the same 32K snapshot at thermal 1. It
engaged **47 fused transitions/token** (1457 dispatches across 31 transitions)
and produced the same full-logit/token hash in both reference arms. The
counterfactual did not reduce the exposed band: incumbent reference/HC-hold
was **12.309/9.934 ms**, candidate was **12.323/9.910 ms**. That prices the
HC wall at **2.374/2.413 ms** (candidate ratio **0.984×**) and the
floor-adjusted closable gap at **1.284/1.323 ms** (candidate ratio **0.971×**).
The candidate misses the real-schedule 1.25× gate, so there is no production
integration or default change. Its comparison route and
`QWEN_FN_HC_INJECT_NORM` remain on `perf/hc-band-fusion` at `bead0f296`, not on
`main`. The conditional 48K/B=3 A/B and family ports were not run after this
negative gate.

## GLM mHC, B=1

The M5 profile at depth 32768 has 1060 dispatches, no profile overflow, and
thermal 1. The HC calls are 90 pre-sites and 90 post-sites. Each pre uses two
ordered kernels; each post uses one.

| Order | Dispatch | Inputs → outputs | Logical tensor bytes | Dependency / merge status |
| --- | --- | --- | ---: | --- |
| 1 | `kernel_mhc4_pre_mix_partial_q8_0` | residual `[4096,4]` + Q8_0 mix `[16384,24]` → 24×64 partial dots + 64 sum-square partials | 65,536 + 417,792 → 6,144 + 256 | Computes all mix partials and the RMS partials from the same residual; one dispatch. |
| 2 | `kernel_mhc4_pre_finish` | residual, partial dots, scale `[3]`, base `[24]`, norm `[4096]` → mixes, controls, collapse, normalized block input | residual 65,536; dots 6,144; norm 16,384; outputs 32,768 plus 192 mix/control bytes | Global reduction, 20 Sinkhorn rounds, collapse, and block RMSNorm are already one 1024-thread TG. Depends on #1. |
| 3 | Attention/KDA/MLA or FFN block | normalized input → block output | block-specific | Cannot cross the control barrier before its output exists. |
| 4 | `kernel_deepseek_v4_hc_post` | block `[4096]` + residual `[4096,4]` + post `[4]` + comb `[4,4]` → next residual `[4096,4]` | 16,384 + 65,536 + 16 + 64 → 65,536 | Its updated residual is immediately read by the next pre's Q8 partial projection. |

The port candidate is a mixed-format multi-output dispatch that applies the
post transform while producing the next Q8 projection partials (and their
sum-square partials). It preserves a separate finish dispatch for the global
Sinkhorn/reduction. It removes one dependency boundary per mHC transition;
the 06-Nov proposal to combine same-input projections is consistent with this
boundary.

## V4.1 mHC, B=1

The width-1 `v41_batch_depth … 32768 1 1 graph` dump records the real
`Forward::step`: 701 waves, 41 scopes, no truncation, and 14.442 ms summed GPU
wave spans. Its harness thermal guard accepted pressure `<2` but does not
serialize the exact level. Inventory confirms hidden width **5120**.

| Order | Dispatch | Inputs → outputs | Logical tensor bytes | Dependency / merge status |
| --- | --- | --- | ---: | --- |
| 1 | `kernel_deepseek_v41_hc_pre_wide` | residual `[4,5120]`, F32 projection `[20480,24]`, incoming pre `[4]`, norm `[5120]` → mixes `[24]`, collapse `[5120]`, normalized block input `[5120]` | 81,920 + 1,966,080 + 16 + 20,480 → 96 + 20,480 + 20,480 | One 25-TG/1024-thread dispatch already performs all 24 projections beside collapse and RMSNorm. Controls are deferred to post. |
| 2 | Attention block | normalized input → attention output `[5120]` | block-specific | Output is required by the HC post. |
| 3 | `kernel_deepseek_v41_hc_post_controls` | block output, residual `[4,5120]`, mixes, scale `[3]`, base `[24]` → controls + next residual `[4,5120]` | 20,480 + 81,920 + 96 + 12 + 96 → 82,016 | One dispatch computes controls/Sinkhorn and updates residual. |
| 4 | `kernel_deepseek_v41_hc_pre_wide` | updated residual → next pre outputs | same pre geometry | Depends on #3. This is the same post → next-pre boundary as GLM. |
| 5 | Routed FFN and `kernel_deepseek_v41_ffn_post_controls` | slots `[6,5120]`, weights `[6]`, shared output `[5120]`, residual and controls → next residual | 122,880 + 24 + 20,480 + residual/control bytes | The FFN post already folds routed sum and mHC post controls into one dispatch. |

The port candidate is to compute the previous post transform on the fly in
each of the 24 pre-projection TGs, while the remaining TG writes the transformed
residual, collapse, and norm. That would replace post-controls → next-wide-pre
with one F32 multi-output dispatch; the existing pre and FFN-post fusions stay
intact.

## Reproduction

M5 evidence is in `~/qwen-runs/2026-11-11-hc-fuse/` (`fn-graph.jsonl`,
`glm-profile.jsonl`, `v41-graph.jsonl`, and the isolated probe outputs). GPU
commands use the `hc-fuse` lease. The probe code is at `bead0f296`; no GLM
scheduler or prefill-slicing files are in this change.
