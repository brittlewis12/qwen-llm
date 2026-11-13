# GLM/V4.1 decoded expert-cache: first slice

Status: **measurement only; no cache implementation selected**. This slice evaluates whether a resident or losslessly repacked expert hot set can amortize IQ decode. It does not change product behavior. The GLM route census is available; V4.1 route capture failed in the profiler preflight, before loading model weights. Raw V4.1 attempts are at `metagross:~/qwen-runs/2026-11-12-glm-prefill/decoded-expert-cache/route-census/v41-80k-packed-v{2,3}/`.

## GLM route-work skew

The census trace is one held code-corpus 2,048-row chunk at 48K, plus 256- and 1,024-row append histograms at 80K (`stage-attribution/census.jsonl`). The root has 12,096 layer/expert bins (42 MoE layers x 288 experts), 11,742 active. It routes 688,128 rows across 72 repeated chunks for three 48K roots: mean 56.9 rows per bin, p50 41, p90 125, p99 311, maximum 1,144. This is the full-chunk regime; append averages are 7.1 rows/bin at 256 and 28.4 at 1,024.

The three-agent 180-second GLM plan contributes 28 completed calls (7/13/8 by agent), 17,903 appended tokens. Each call uses the 256-row route counts through 384 rows and the 1,024-row counts above that; calls longer than 1,024 rows use the 1,024-row per-token distribution. Whole layer/expert weights are greedily packed by routed work per byte. This is a plan-weighted projection from a single corpus route trace, not an observed multi-session cache hit rate.

| Representation | Expert bytes | Full GLM bank | Routed work covered at 57 GB | at 60 GB |
| --- | ---: | ---: | ---: | ---: |
| Native IQ | varies by layer | 109.9 GB | 85.70% | 87.43% |
| F16 / BF16 | 50.3 MB | 608.8 GB | 33.81% | 34.90% |
| int8, per-row scales | 25.2 MB | 304.6 GB | 51.08% | 52.61% |
| int8, group-128 scales | 25.6 MB | 309.2 GB | 50.65% | 52.17% |
| exact level-code estimate, 3.3 bpw | 10.4 MB | 125.6 GB | 81.01% | 82.83% |

## Crossover evidence and exactness

The existing tensor-expert limiter reports the all-expert F16 no-dequant control divided by its matched full control. It is bitwise equal, but slower:

| Total rows | IQ2_S | IQ3_S |
| ---: | ---: | ---: |
| 32 | 2.719x | 2.297x |
| 161 | 2.755x | 2.425x |
| 512 | 2.512x | 2.008x |

The F16 control streams 4.65-6.24x the native bytes. It loses for the measured append widths; it has not been measured at 2,048 rows. The 2,048-row GLM stage baseline is 721.3 ms routed experts / 1,999.7 ms ordinary chunk wall (36.1%); append-256 is 244.9/453.1 ms (54.1%), and append-1,024 is 446.7/1,154.8 ms (38.7%). These shares are from the stage-attribution packet and include a single profile trace.

No BF16 or int8 crossover was measured. Both change the decoded IQ values and are not bitwise-preserving; neither has paired non-inferiority evidence. BF16 has the same footprint as F16; int8 halves weight bytes but adds scale decode. The exact level-code format has not been implemented or benchmarked; its bitwise witness is still required.

## Conditional projection and next gate

At the 57 GB level-code budget, 81.01% of GLM routed work fits. If the cached subset reaches 1.25x isolated-stage speedup with no interference, the first-order wall projection is **+5.8%** at the 48K root chunk, **+8.8%** on append-256, and **+6.3%** on append-1,024. At 60 GB, the projections are +6.0%, +9.0%, and +6.4%. These are conditional at the minimum stage gate, not measured gains. No format has demonstrated a positive crossover, so level codes merit the next exactness/stage probe; no cache merits production implementation yet.

V4.1 is the 384-expert, top-6, H=5,120/F=2,304 IQ2_XS shape from the prior tensor-tile study. Its old generic route census rejected architecture `deepseek41` before loading; the DSV4 profiler probe also rejected it (`unsupported architecture Some("deepseek41"); expected deepseek4`). No V4.1 skew, crossover, or wall projection is claimed. Reuse a V4.1-native route capture path before attempting its cache projection.

References: `2026-11-12-glm-tensor-experts.md`; `2026-11-13-glm-prefill/stage-attribution/census.jsonl`; remote V4.1 preflight records above; run plan at `$HOME/qwen-runs/2026-11-12-glm-prefill/decoded-expert-cache/plans/glm-serve-plan.json`.
