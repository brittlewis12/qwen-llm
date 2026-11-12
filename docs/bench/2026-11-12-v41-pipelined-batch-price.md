# V4.1 B=3 one-ahead loop: stage-one price fails

**Decision: stop at the price gate.** Across the four M5 B=3 cells, the
largest measured round-minus-GPU-service term is **1.40 ms**, below the
required **2 ms per round** even before accounting for token readback, Engram
row staging, or event wakeup. No pipeline implementation or serve campaign was
started; the temporary timeline code was removed.

## Method

Source was `f691af6e1` on M5 Ultra, DeepSeek V4.1 Flash 1017 GSQ-RCO IQ2_XS
with the disk-streamed IQ4_NL Engram sidecar. The temporary `iter`-profile
probe ran 32 sampled B=3 rounds at 32K and 48K, with QoS disabled and
user-initiated. Its sampler was the fast GPU path (`temperature=1`, `top_k=0`,
`top_p=1`, `min_p=0`, fixed seed 42); each sampled token fed the next round.

Each forward segment and sampler command recorded host encode start/end,
commit, GPU start/end, and wait start/end on the same monotonic clock. The
per-round wall includes the forward, GPU sample, and token readback. GPU service
is the sum of command-buffer GPU durations. GPU span runs from the first
command's GPU start through the sampler's GPU end and includes gaps between
commands. Both differences are reported because a span can include GPU idle
time.

## Results

| Context | QoS | Round wall ms | GPU service ms | GPU span ms | Wall − service ms | Wall − span ms | Forward encode ms | CPU Engram rows ms | Token readback ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | off | 27.93 | 26.53 | 27.05 | **1.40** | 0.88 | 6.00 | 0.82 | 0.024 |
| 32K | user-initiated | 25.72 | 24.72 | 25.09 | **1.00** | 0.63 | 4.24 | 0.29 | 0.014 |
| 48K | off | 27.11 | 25.82 | 26.25 | **1.28** | 0.85 | 5.24 | 0.34 | 0.024 |
| 48K | user-initiated | 25.60 | 24.59 | 24.97 | **1.01** | 0.63 | 4.07 | 0.27 | 0.013 |

QoS lowered round wall by 2.21 ms at 32K and 1.51 ms at 48K under this host
load. It reduced wall-minus-service by 0.40 and 0.27 ms, respectively. Median
commit-to-GPU-start was 0.21–0.40 ms; median GPU-end-to-wait-return was
1.10–2.13 ms. The host snapshot showed load averages of 36.21/38.20/29.51 at
probe start and 40.37/39.04/31.07 at end. Multiple `~/ghost` AOT workers were
active, reaching roughly 434% and 318% CPU. Thermal pressure was 0 at both
snapshots.

## Design estimates

| Design | Measured exposed work | Optimistic saving bound | Decision |
|---|---|---:|---|
| Encode the full next round and gate its first Engram dispatch on a shared event | Round-minus-service tops out at 1.40 ms. Under QoS, CPU Engram staging is 0.27–0.29 ms, while the Engram-free prefix consumes about 0.51–0.52 ms of GPU time. Token readback is 0.013–0.014 ms. A prior M5 probe measured about 0.222 ms per shared-event handoff, including encoder transitions and the return join. | Below 1.40 ms before costs; about 0.8 ms at best after the measured readback and event budget | Below 2 ms |
| Gather Engram rows on GPU from a staged table window | Removes some CPU row work, but adds GPU gather work and still has the same measured round ceiling. | At most 1.40 ms before gather cost | Below 2 ms |
| Encode only the Engram-free prefix ahead | Prefix command host encoding was 0.16–0.21 ms; its GPU work was about 0.51–0.52 ms. | At most 0.21 ms | Below 2 ms |

Even the first design cannot pass the gate using the measured wall-minus-GPU
service ceiling. **No V4.1 runtime code is retained.** The requested token
exactness, lifecycle, release, three-agent, and thermal gates were not entered.

## Run record

The M5 build completed in the `iter` profile. The 32K/48K, QoS-off/on probe
produced all four JSON rows under
`~/qwen-runs/2026-11-12-v41-pipe/hostgap-b3.jsonl`; command timelines and
host snapshots are retained alongside it. The wrapper command returned 2
after the final process-list `sort | head` hit a broken pipe; this happened
after all four timing rows and the final host snapshot had been written. The
lease wait was 664 seconds. The V4.1 model was loaded only after the pre-load
process check passed.

**V41-NQTZ Aven note:** Stage-one M5 price failed the 2 ms gate. Round-minus-GPU
service was 1.00–1.40 ms at B=3; the best event-gated estimate is about 0.8 ms.
No pipeline code or Stage 2 work will proceed from this price. An external Aven
annotation could not be posted because this session has no Aven connector.

The shared-event cost comes from the M5 results in
[`2026-11-11-two-queue-causal.md`](2026-11-11-two-queue-causal.md). No roadmap
change was made.
