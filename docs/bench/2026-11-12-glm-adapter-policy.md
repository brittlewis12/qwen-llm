# 2026-11-12 — GLM small-append explicit-adapter policy mismatch fixed

Status: fix qualified and committed on `fix/glm-tensor-adapter`; not merged.
Production behavior at 128..512 rows changes from the grouped F32-input route
to the intended explicit Half-storage composition, with bitwise logits and
state. `QWEN_GLM53_PREFILL_IQ_SMALL_ROWS=0` retains the pre-policy grouped
route.

Base: main `f691af6e1`. Fix commits `dd0641e5` (adapter + CPU tests),
`e0b6a0140` (down preflight before a capacity decline), `ee02d93f` and
`6530dd6f` (route-sweep state and stage probes). Probe-only baseline branch
`bench/glm-adapter-probe-base` at `579ea648` is main plus those two example
commits. Stock GLM-5.3-Flash UD-IQ3_XXS, M5 Ultra, `iter` profile
with `perf-ablations`, 98,304 capacity, 2,048-row prefix chunks, `code.txt`
corpus.

## Mismatch and census

`grouped_down_policy` returns `GroupedDownPolicy::Iq3STensorOrSmallCounts` for
IQ3_S down at 161/256/512 rows. The explicit adapter
(`expert_tensor_gate_up::prepare`) accepted only `Iq3STensor`, so those 39
blocks declined the explicit composition and took the grouped F32-input tensor
gate/up. The three IQ4_XS-down exception blocks carried `Incumbent` and were
already explicit. At 32 rows the policy is `Iq3STensor`, so all blocks were
explicit.

The production route census at 48K and 80K confirms it. Counts are gate/up
dispatches in one `occupancy` append at widths 128, 161, 256, 384 and 512; all
five widths are identical within a binary.

| arm | `f16act` up | `f32` up | selected route |
|---|---:|---:|---|
| main `f691af6e1` | 3 | 39 | IQ4_XS blocks explicit, IQ3_S blocks grouped |
| fix branch | 42 | 0 | all blocks explicit |
| fix + `IQ_SMALL_ROWS=0` | 0 | 42 | pre-policy grouped route |
| main + `IQ_SMALL_ROWS=0` | 0 | 42 | same as fix + `=0` |

The rollback census matches the baseline binary exactly, so the fix leaves the
documented grouped/expanded rollback unchanged. It is not a byte-for-byte
rollback of the mismatched main route: main's three IQ4_XS explicit blocks
return to the grouped route under `=0`, as `af49e2ace` documented.

## Fix

`down_route(down_dtype, policy)` now owns the policy-to-composition mapping.
`Iq3STensorOrSmallCounts` with IQ3_S down composes the native tensor down when
`expert_tensor_down::prepare` admits it, otherwise the
`iq3_s_down_retile` small-count retile, mirroring the grouped adapter. The
tensor tier, geometry, precision, binding, alias and capacity guards are
unchanged. A missing eligible pipeline remains an error; a capacity decline
falls back to the grouped route rather than silently using a different down.
The selected down is still prepared before a gate/up capacity decline is
returned, so invalid down bindings remain errors. No new knob or allocation
was added.

CPU tests cover every `GroupedDownPolicy`/down-dtype pairing and the production
GLM policy at 128..512 rows. No new `#[ignore]` tests.

## Gate

One probe-only baseline binary holds main's production arm; one fix binary
holds the fix. In each binary the route sweep measured the production arm
against the forced explicit arm (`tensor_iq`); the two arms are interleaved
three times per width, rotated by width and repetition. On the fix binary the
production arm is bitwise identical to the forced explicit arm at every width
(`max_abs_logit_diff=0`, `state_bitwise=true`), so the main-vs-fix pair below is
the actual route change. The routed-expert stage is a median of three
interleaved stage-profiled calls per arm.

Deltas are candidate minus incumbent; positive wall saved means the explicit
route is faster. The 90% interval resamples the three paired blocks 20,000
times, so it describes this packet rather than a population.

**48K**

| Rows | Incumbent wall ms | Explicit wall ms | Speedup | Wall saved ms [90% CI] | Stage saved ms [90% CI] |
| ---: | ---: | ---: | ---: | --- | --- |
| 128 | 314.4 | 308.8 | 1.018x | +7.82 [+5.56, +10.08] | +5.52 [+5.35, +5.68] |
| 161 | 377.4 | 369.7 | 1.021x | +8.14 [+7.36, +8.92] | +7.73 [+7.53, +7.92] |
| 256 | 459.5 | 449.7 | 1.022x | +8.35 [+5.98, +10.71] | +11.42 [+10.62, +12.22] |
| 384 | 576.5 | 561.1 | 1.027x | +16.31 [+15.41, +17.22] | +15.15 [+15.09, +15.21] |
| 512 | 688.9 | 667.6 | 1.032x | +23.32 [+21.08, +25.57] | +20.70 [+20.31, +21.09] |

**80K**

| Rows | Incumbent wall ms | Explicit wall ms | Speedup | Wall saved ms [90% CI] | Stage saved ms [90% CI] |
| ---: | ---: | ---: | ---: | --- | --- |
| 128 | 329.9 | 324.4 | 1.017x | +5.80 [+5.58, +6.01] | +5.98 [+5.86, +6.09] |
| 161 | 389.8 | 382.0 | 1.020x | +9.88 [+7.40, +12.35] | +7.28 [+7.12, +7.45] |
| 256 | 466.8 | 456.4 | 1.023x | +10.50 [+10.20, +10.81] | +10.69 [+10.47, +10.91] |
| 384 | 592.5 | 577.6 | 1.026x | +15.15 [+14.99, +15.32] | +16.51 [+15.40, +17.61] |
| 512 | 712.1 | 691.2 | 1.030x | +20.97 [+20.82, +21.12] | +20.61 [+20.46, +20.76] |

Every width and depth is bitwise: `max_abs_logit_diff=0` on both arms and
`state_bitwise=true` for the post-append persistent state. The wall interval
excludes zero at every cell. Thermal pressure was 0 or 1 during the sweeps; the
usual shared-host `ghost` work remained resident.

## Serve coherence smoke

A 20 s three-agent GLM serve window on the release build (release `qwen`
`a0ffcb8c…`, `qwen-tok` `7fdf292f…`) completed five non-empty turns across all
three agents: 767 delivered tokens, 38.35 tok/s, gap p50/p99/max
27.7/245.1/493.5 ms, thermal 1. Compiler overlap was flagged, so this is a
coherence smoke, not a throughput claim.

## Decision

Ship the adapter fix on the branch. The route is bitwise and the paired append
wall improves at every required width and depth, so no serve A/B was needed.
`QWEN_GLM53_PREFILL_IQ_SMALL_ROWS=0` remains the documented pre-policy
grouped/expanded rollback and its census is unchanged by this fix.

Raw files and logs:
`metagross:~/qwen-runs/2026-11-12-glm-adapter/`
(`occupancy-{base,fix,rollback,base-rollback}.jsonl`,
`routes-{probe,fix}-{48k,80k}.jsonl`, `gate-final-{48k,80k}.jsonl`,
`serve-smoke/`, build logs and binary digests).
