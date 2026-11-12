# V4.1 sustained three-agent slice gate: inconclusive, default off

All six requested arms completed. **Do not enable by default:** paired
throughput is +0.21%, but its 95% CI is **-25.12% to +25.53%**, far wider
than +/-2 percentage points. A 2% non-inferiority claim is unsupported.
The grand-mean delivered rates are 86.502 off / 86.400 on tok/s; those means
alone conceal the opposite paired effects.

Observed worst gap across all windows falls 3436.72 -> 2873.68 ms (16.4%),
but one pair worsens and the reduction is not consistent. P99 increases from
30–55 ms off to 191–220 ms on. The native 1.39 -> 0.22 s warm probe does not
bound the live host/preparation stalls. No policy was changed mid-packet and
no slow turns or flagged arms were discarded. Flash-Next is priced next.

## Fixed method

Source/binary `abc3ba1a3`, release, nice10. Current GSQ-RCO IQ2_XS with IQ4_NL
Engram; three clients, production max-active4 / three sampler workers,
49152 warm tokens each, context196608, real agent sampling (no request sampling
overrides), seed20261109 call draws, 15-second ramp and180-second window.
MS0 versus MS200/minimum500ms. Same binary checksum verified before every arm.
Order **A1 B1 B2 A2 A3 B3** (ABBAAB), pairs(A1,B1),(A2,B2),(A3,B3).

The lead authorized pressure2 during warm-up and measurement, aborting at3.
The harness adds opt-in `--settle-max-thermal-pressure1` independent of
`--max-thermal-pressure2`; defaults stay unchanged. Each arm idles at<=1
continuously for60s before loading and again after warming. `settle.json`
retains the samples; packet validation checks both intervals. Each arm has
its own queued lease, below15min. No other model was resident at load.

## Per-arm results

Times in milliseconds except partial-prefill seconds. ITL is the nearest-rank
content-delta interval distribution, both endpoints in the measured window.
Append TTFT includes requests offered in-window that reached first output;
censored requests are retained in the raw call records.

| arm | MS | tok/s | ITL p50 / p99 / max | append TTFT p50 / p99 | partial_prefill_s |
| --- | ---: | ---: | --- | --- | ---: |
| A1 | 0 | 91.250 | 27.65 / 30.39 / 2488.50 | 964.47 / 2873.43 | 2.5974 |
| B1 | 200 | 88.856 | 27.89 / 192.51 / 2873.68 | 865.27 / 4605.83 | 1.5681 |
| B2 | 200 | 90.222 | 27.83 / 190.72 / 1761.28 | 888.09 / 2123.59 | 1.6054 |
| A2 | 0 | 80.906 | 27.85 / 55.42 / 3436.72 | 959.66 / 3563.37 | 2.5243 |
| A3 | 0 | 87.350 | 29.18 / 32.92 / 2357.54 | 982.32 / 2002.36 | 1.2871 |
| B3 | 200 | 80.122 | 29.37 / 220.25 / 2079.28 | 952.64 / 2196.34 | 3.4204 |

Sliced arms recorded99/99/141 ready-peer rounds, confirming interleaving was
active. Incumbent arms recorded zero. Partial-prefill accounting includes
prefill paid by turns that later cancelled, and excludes peer decode service.

## Thermal traces

Seconds are sample-and-hold durations clipped to the180-second window; all
unknown exposure is zero. Complete raw traces are retained.

| arm | level0 | level1 | level2 |
| --- | ---: | ---: | ---: |
| A1 | 180.00 | 0.00 | 0.00 |
| B1 | 113.14 | 66.86 | 0.00 |
| B2 | 180.00 | 0.00 | 0.00 |
| A2 | 180.00 | 0.00 | 0.00 |
| A3 | 109.42 | 1.04 | 69.54 |
| B3 | 58.86 | 27.43 | 93.71 |

Whole model-process observation, including warming/ramp and shutdown, at
levels0/1/2 respectively: A1 354.24/0/0s; B1 255.99/67.20/0s;
B2 279.73/52.47/0s; A2 355.69/0/0s; A3 281.08/1.04/69.91s;
B3 231.18/27.05/93.71s. Pre-load settling is separately in `settle.json`.
The two pressure2 windows are accepted under the new method.

## Paired uncertainty and remaining stalls

Relative differences use each pair's off rate as denominator. The95% paired
Student-t interval uses three arm pairs as replicates (df2), not individual
tokens. This small sample makes the uncertainty explicit.

| pair | order | on-minus-off tok/s | relative difference |
| --- | --- | ---: | ---: |
| A1/B1 | AB | -2.3944 | -2.624% |
| A2/B2 | BA | +9.3167 | +11.515% |
| A3/B3 | AB | -7.2278 | -8.275% |

Mean absolute difference -0.1019tok/s,95% CI[-21.2348,+21.0311]. Mean paired
relative difference +0.2056%,95% CI[-25.1175,+25.5288], half-width25.3231pp.
Thermal state, natural EOS/turn turnover, I/O and compiler exposure vary;
the packet cannot attribute the uncertainty solely to throttling. Every arm
has the compile-overlap flag retained. A1 had compiler presence in64/174
window samples, peaking near2.8 CPU cores; those processes had exited before
inspection, so their source was not identified.

B1's2.874s maximum is common to two peers. It ends at the first peer round of
a1798-row append at position51624; later rounds are roughly150–300ms apart.
That append records882769920 process disk-read bytes. The same-size turn:

| arm | active prefill ms | process disk-read bytes | peer rounds |
| --- | ---: | ---: | ---: |
| A1 | 1564.37 | 167591936 | 0 |
| B1 | 3825.43 | 882769920 | 9 |
| B2 | 1614.46 | 6012928 | 9 |

This supports preparation/I/O variability before GPU cuts as a remaining
bottleneck, not a precise attribution of all reads to Engram. The process
scope also includes peer work. No asynchronous-read workaround was added.

Sampled responses from every arm remain coherent source-specific Rust/Metal
reviews. Inspected samples have no response error or replacement characters;
output-limit truncation is expected. This is a qualitative sample check.
Native75/75 bitwise and cancellation results remain as recorded in the
[initial note](2026-11-12-v41-layer-slice.md).

## Records

Raw packet: `metagross:~/qwen-runs/2026-11-12-v41-layer-slice/protocol-v2/`.
`serve/<arm>/summary.json`, `slice-stats.json`, `settle.json`, `thermal.json`,
raw events/responses and native lines are retained. The compact
[results JSON](2026-11-12-v41-layer-slice-sustained.json) includes the CI and
both thermal-duration summaries. Twenty CPU harness/statistics tests pass;
contracts pass for the measured harness. No default, row arithmetic, artifact
or roadmap change was made for this gate.
