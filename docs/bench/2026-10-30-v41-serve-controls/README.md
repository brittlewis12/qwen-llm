# V4.1 Serving Controls: Idle Driver Delay Reduced

## Result

On one fixed cached request, a ten-second resource-aware keep-alive window
reduces median first-delta latency after a five-second pause from **744.982 ms
to 220.555 ms**, a 70.4% reduction. The native GPU envelope stays approximately
145 ms while the Metal driver interval falls from 531.930 to 5.861 ms.

The separate **completed ordinary-logging/expiry follow-up** confirms a
525.869 ms five-second TTFT benefit (745.873 -> 220.004 ms), no obvious decode
regression, and expiry at the tested 15-second idle gap. It is not pooled with
the first cohort or presented as a general throughput result.

Four-GiB retained mappings do not materially improve this cohort. Neither
control is promoted to a family default. The historical 13-26-second cached
prefill tail does not recur and is not declared explained or fixed.

## Frozen Configuration

- Source `5bf7e4e5`, including controls `fb1002fa`; M5 Ultra, 80 GPU cores,
  256 GiB, macOS 27.0.1, SDK 27.0, canonical optimized release.
- Executable SHA256:
  `e666e75f875cc0ad6a536147558b81b3cbdb448ad9e38894207d42fda31ccb17`.
- GSQ-RCO IQ2_XS backbone and disk-only IQ4_NL Engram; unchanged arithmetic,
  numerical kernels, prefill chunk policy and output reservation.
- Fixed saved request: 15,564 input tokens, seed 3301, temperature 1, high
  reasoning and 32,768 output reservation. Every completed generation has
  534 output tokens and matching reasoning/answer byte fingerprints.
- Cached calls reuse 15,562 tokens from a RAM snapshot, process two new
  tokens and retain capacity 48,331. Requested context limit stays 131,072.
- Same local loopback client and temporary endpoint for every arm. First
  delta includes reasoning; this is not first visible-answer latency.

## Schedule And Scope

| Cohort | Process order | Logging | Completed generations | Cached observations |
| --- | --- | --- | ---: | ---: |
| `first6arm` | A0, M0, I0, I1, M1, A1 | `info,qwen_diag=debug` | 30 | 24 |
| `expiry` | A2, E0 | `info` | 10 | 8 |

Both cohorts use the frozen executable/request above and unsplit native timing.
Keep their rows and summaries separate: only the first cohort emits
per-quantum lifecycle debug records for enabled keep-alive arms.

### First Six-Arm Cohort

Three independent policies, never combined:

| Label | Mapping target | Idle window |
| --- | --- | --- |
| A | Ordinary / None | Off |
| M | Soft 4 GiB | Off |
| I | Ordinary / None | 10 seconds |

Process order is **A0, M0, I0, I1, M1, A1**. Each fresh process completes one
cold warm-up generation, then two pairs: wait five seconds and complete a
cached request, immediately repeat and complete another. There are four
retained observations per policy/gap combination, 24 cached calls total, plus
six cold warm-ups. No slow sample is replaced or discarded.

The campaign completes naturally in 392.9 seconds. All six servers stop
gracefully; no forced kill, pulse failure, malformed/truncated native record,
or leftover temporary listener is observed. The canonical lease is released.
The usage endpoint is not restored by this campaign.

Unsplit native timing is enabled throughout. `RUST_LOG=info,qwen_diag=debug`
is the same across processes, but only an enabled idle window emits the
per-quantum lifecycle debug records. Thus these are instrumented serving
controls, not uninstrumented throughput results. Full-generation/decode
differences are not attributed to keep-alive; the completed ordinary-logging
expiry follow-up is reported separately below.

### First-Cohort Cached Results

All values are medians of four observations. GPU and driver columns below
describe the single two-row native prefill submit after the five-second pause.

| Policy | Five-second TTFT ms | Immediate TTFT ms | Idle native wall ms | Idle GPU ms | Idle driver interval ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 744.982 | 174.784 | 679.581 | 144.586 | 531.930 |
| Soft 4 GiB | 734.716 | 175.693 | 675.607 | 144.471 | 528.340 |
| Keep-alive 10 s | 220.555 | 176.001 | 153.348 | 144.974 | 5.861 |

Every keep-alive idle observation is 218.360-223.698 ms; every ordinary idle
observation is 724.219-749.486 ms. The improvement appears in both forward
and reversed process order. It does not make the idle request identical to
an immediate repeat, and it is not a decode-throughput or 70% whole-turn gain.

The 4 GiB and ordinary idle TTFT ranges overlap. Full-response wall also does
not favor 4 GiB here: idle medians are 8.548 versus 8.095 seconds, and
immediate medians 8.022 versus 7.526 seconds. Those whole-response differences
are retained rather than hidden behind the tiny TTFT difference; no mapping
gain or precise decode-cause attribution is claimed.

[cached-samples.csv](cached-samples.csv) contains all 24 observations,
including restore, complete-response wall, native wall, encode, inclusive
wait, GPU and driver intervals. These intervals overlap and must not be added
as an exclusive decomposition; wait minus GPU is not pure queue time.

## Completed Ordinary-Logging And Expiry Follow-Up

Fresh processes **A2 (ordinary, idle window off), then E0 (ordinary, 10-second
idle window)** each complete one cold warm-up, a five-second idle/immediate
pair, then a 15-second idle/immediate pair. Both use `RUST_LOG=info`, without
per-quantum debug logging; native prefill timing remains enabled. No mapping
target is set. These are eight individual cached observations, not medians or
additional replicates of the first cohort.

| Arm | Gap | TTFT ms | Complete response ms | GPU ms | Driver interval ms |
| --- | --- | ---: | ---: | ---: | ---: |
| A2 | 5 s | 745.873 | 8074.832 | 144.551 | 531.559 |
| A2 | Immediate after 5 s pair | 172.878 | 7495.629 | 138.460 | 3.693 |
| A2 | 15 s | 757.076 | 8099.085 | 144.748 | 543.514 |
| A2 | Immediate after 15 s pair | 174.711 | 7516.922 | 138.568 | 4.301 |
| E0 | 5 s | 220.004 | 7557.553 | 145.081 | 5.907 |
| E0 | Immediate after 5 s pair | 174.098 | 7520.139 | 138.535 | 3.913 |
| E0 | 15 s | 738.144 | 8098.224 | 145.012 | 524.174 |
| E0 | Immediate after 15 s pair | 174.040 | 7538.083 | 138.463 | 3.838 |

Ordinary logging confirms the approximately 526 ms five-second TTFT benefit;
idle GPU envelopes remain approximately 145 ms. At 15 seconds, E0 returns to
ordinary-like TTFT and driver delay, while its immediate repeat is fast again.
This supports working ten-second expiry at this tested gap, not arbitrary
long-idle/arrival behavior or a debug lifecycle trace of the exact cutoff.

Complete-response wall improves by 517.278 ms at five seconds and is nearly
equal at 15 seconds. Cached decode-phase ranges are 7322.589-7342.338 ms for
A2 and 7337.748-7361.807 ms for E0: no obvious decode regression, but only one
process per policy and one observation per gap/pair, without reversed order.
This does not establish a precise decode effect or a broad throughput gain.

[expiry-cached-samples.csv](expiry-cached-samples.csv) retains all eight cached
rows. Both CSVs label cohort/logging explicitly and include decode-phase time.
All ten generations, including warm-ups, match the first cohort's output byte
hashes and 534-token count. The follow-up completes naturally in 146.5 seconds;
both servers stop without forced kill, request/native/pulse failure,
malformed/truncated native records or a leftover temporary listener.

## Correctness And Boundaries

The first cohort's 30 and follow-up's ten completed generations all match
reasoning/answer byte fingerprints and token counts. That is a bounded
same-request output check, not complete-logit, causal-state or general quality
equivalence. The separate actual-M5 tiny retained-resource witness passes with
Metal API validation: bank/dense buffer
coverage, identity deduplication, Engram exclusion and post-owner-drop guarded
GPU copy. Its locally compiled optimized test profile/SDK 26.2 is separate
from the SDK 27 release serving measurements.

These cohorts test RAM-snapshot reuse at one capacity, with five-second idle
and a narrow 15-second expiry check. They do not cover live-growing or arbitrary
conversations, memory pressure, arbitrary arrival phases, broader long-idle
behavior, energy/power, other artifacts or the original intermittent tail.
The earlier 1.14-second idle discriminator used a different client/run state;
do not pool its absolute latencies with this loopback cohort.

Keep the mapping target opt-in and ordinary loading as the default. The
keep-alive result supports the ten-second setting for this measured workload;
ordinary logging and narrow expiry are now checked, but power/pressure and
broader workload evidence remain absent. No global default promotion follows,
and this does not explain or fix the originally reported 15-40-second tails
(including the earlier 13-26-second cached-prefill stalls).
