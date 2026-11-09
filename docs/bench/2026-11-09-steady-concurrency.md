# Steady-state tier-1 agent concurrency, 2026-11-09

Engine **`9b193b3808932f985b82dfb62151826c71230a49`**, M5 Ultra 256 GiB;
GLM includes main's four-channel batched expert reuse. Same production tier-1
GGUF artifacts as the November 8 standing. This supersedes its finite-cohort
protocol. Natural EOS and unequal work are normal; comparisons use rates.

## Decision

| Family | MAX_ACTIVE=1 delivered tok/s | Default delivered tok/s | Default / serial | Window tokens, serial / default |
| --- | ---: | ---: | ---: | ---: |
| v41 | 69.617 | 91.044 | **1.308x** | 12531 / 16388 |
| glm | 41.628 | 72.033 | **1.730x** | 7493 / 12966 |
| Flash-Next, serial baseline | 67.200 | — | — | 12096 / — |

V4.1's default distributes about **29–31 tok/s per agent**. Serial is strongly
skewed toward the long call. GLM distributes **23.7–24.2 tok/s per agent**;
its MAX_ACTIVE=1 single-live-session control also rereads more state, so 1.730x is
the complete serving-policy benefit, not an isolated expert-kernel A/B.
Flash-Next still serializes generation. Current batching is materially useful
for continuously active coding clients; the finite cohort's solo tail obscured it.

## Per-agent latency and delivery

Quantiles are nearest-rank and use requests **sent and first-payload received
inside the shared window**. Pending TTFTs are right-censored, with lower bounds
retained in the calls CSV. Carry-in replies count toward delivery but not these
TTFT quantiles. A dash means no eligible observed TTFT, not zero wait.
`Exposure` is pre-first-payload time inside the window, including queueing and
prefill; these overlapping request-seconds are not additive cohort wall.

| Cell | Agent | Delivered tok/s | New turns | TTFT p50 / p95 s | Pending TTFTs at cutoff | Exposure s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v41-default | 0 | 31.03 | 0 | — / — | 0 | 0.00 |
| v41-default | 1 | 30.58 | 4 | 0.802 / 1.656 | 0 | 3.92 |
| v41-default | 2 | 29.43 | 19 | 0.959 / 2.044 | 0 | 21.35 |
| v41-serial | 0 | 51.82 | 3 | 4.439 / 23.147 | 1 | 50.51 |
| v41-serial | 1 | 10.13 | 2 | 22.563 / 129.229 | 0 | 153.90 |
| v41-serial | 2 | 7.67 | 2 | 6.698 / 6.698 | 1 | 160.60 |
| glm-default | 0 | 24.14 | 5 | 1.051 / 1.567 | 0 | 4.80 |
| glm-default | 1 | 23.68 | 10 | 0.570 / 4.920 | 0 | 15.68 |
| glm-default | 2 | 24.22 | 5 | 0.567 / 1.860 | 0 | 5.07 |
| glm-serial | 0 | 8.92 | 5 | 19.280 / 78.051 | 0 | 149.77 |
| glm-serial | 1 | 8.48 | 6 | 13.986 / 84.263 | 1 | 150.84 |
| glm-serial | 2 | 24.22 | 6 | 18.763 / 21.339 | 1 | 99.83 |
| fn-serial | 0 | 13.26 | 2 | 69.343 / 69.343 | 1 | 145.38 |
| fn-serial | 1 | 23.51 | 1 | — / — | 1 | 118.50 |
| fn-serial | 2 | 30.44 | 2 | 71.287 / 71.287 | 1 | 100.30 |

A V4.1 default agent has one 9K-budget carry-in reply spanning the whole window;
it still contributes 5586 delivered tokens. Counting only completed responses
would discard that work. Other agents continue issuing turns throughout it.
Serial V4.1 and GLM each have two TTFTs pending at cutoff; Flash-Next has three.
Observed latency quantiles alone therefore understate the serialized waiting.

| Cell | Content-gap p95 / max ms | Open post-first-token streams: 1 / 2 / 3 seconds |
| --- | ---: | ---: |
| v41-default | 28.54 / 2196.50 | 0.00 / 32.58 / 147.42 |
| v41-serial | 13.98 / 27.80 | 173.28 / 0.00 / 0.00 |
| glm-default | 35.98 / 2289.86 | 0.48 / 28.47 / 151.05 |
| glm-serial | 18.21 / 36.52 | 135.76 / 0.00 / 0.00 |
| fn-serial | 14.91 / 41.10 | 175.47 / 0.00 / 0.00 |

Stream overlap includes interleaved appends; it is not exclusive native GPU
batch service time. The default V4.1 window has **zero solo-stream time**.
The [20-second trace](2026-11-09-steady-concurrency-trace.csv) shows sustained
sharing instead of a finite drain. Multi-second gaps remain during appends.
All sampled thermal pressure is **0**, including warm-up and cancellation.

## Protocol and rate accounting

Three independent client loops start from distinct **48000-token code roots**.
Each keeps drawing paired new-content/reply-budget/tool-gap observations from
its own seeded RNG until cutoff. Seed **20261109**, population **54 V4.1 /
31 GLM / 25 Flash-Next** primary live-prefix calls from the real OpenCode CSV;
cold roots, titles and snapshot-replay input counts are excluded. The pairings
and natural long tails remain. A faster arm advances further in the same seeded
per-agent sequence. There is no per-agent turn quota and no exact-work gate.

Each fresh loopback server uses context capacity **196608**, allowing growth
from the 48K roots. V4.1 uses default effort, GLM/Flash-Next low; greedy, drafting
off. Priming is three sequential eight-token replies, excluded. After releasing
all three clients there is a **15 s ramp**, then one fixed **180 s window**.
"Active" means the client loop is alive, including model waits and sampled tool
gaps; it does not condition on three simultaneous GPU decoders. An early loop
exit fails the arm. At cutoff the server receives SIGINT; partial requests are
cancelled and **drain is excluded**. Queueing and tool pauses inside the window
remain part of the workload.

Untimed costs are reported separately; each arm uses a fresh server:

| Cell | Startup s | Three sequential warm roots s | Ramp s | Measured window s |
| --- | ---: | ---: | ---: | ---: |
| v41-default | 16.90 | 76.20 | 15 | 180 |
| v41-serial | 0.98 | 76.68 | 15 | 180 |
| glm-default | 8.73 | 140.65 | 15 | 180 |
| glm-serial | 8.78 | 140.78 | 15 | 180 |
| fn-serial | 8.04 | 80.84 | 15 | 180 |

The numerator counts **delivered text plus reasoning tokens** using the model's
own `qwen-tok` tokenizer on channel prefixes bracketing the window. This counts
ramp-crossing and interrupted long replies without attributing all their work
to completion time. EOS/format controls are excluded; a boundary BPE merge can
move about one token per channel. On wholly contained completed calls, delivered
counts agree with native output usage to **100% V4.1, 99.82% GLM, 99.96% FN**.
No tokenizer work runs inside the measured window. Deltas/text stay in memory
only; full prompts, responses and SSE transcripts are not retained by default.
The [per-turn CSV](2026-11-09-steady-concurrency-calls.csv) includes counts,
cache usage when available, carry-ins, TTFT censorship and sampled source IDs.
Partial calls have no final native usage; that is missing data, not zero work.

Timing takes **only the canonical Metal lease**, `QWEN_METAL_LEASE_WAIT=1`,
`nice -n 10`; **no build lock**. Builds alone take the build lock. A 25 s cooling
gap and three nominal readings three seconds apart precede each server.
One-second thermal monitoring aborts on nonzero pressure. One model is resident
at a time, bound **127.0.0.1:18737**. No OpenCode instance, config, SQLite DB or
3456 API is used. Other lanes/outside jobs own the lease between short arms.

## Reproduction and evidence

The committed harness is `scripts/bench/serve_realistic.py`. Measured driver
revisions are **e6d33da4** (V4.1/GLM) and **761a13c4** (FN); **4120be7e** hardens
cutoff/timeout handling without changing the rate metric or sampling. Production
engine source is identical across all arms. Evidence is **commit plus command**,
with compact summaries, calls and thermals in
`$HOME/qwen-runs/2026-11-08-realistic-steady/`; regenerable bulk is removed.
The first V4.1 launch's literal `--harness-commit recorded-laptop-commit` was
a bookkeeping placeholder; the preserved driver bytes were verified against
**e6d33da4**. Original launch argv is retained with this provenance note.

```sh
# Build once under the build lock; no GPU lease needed for compilation.
lockf -k ~/.cache/qwen-llm/locks/build.lock nice -n 10 \
  cargo build --offline --release -p qwen-cli --bin qwen --bin qwen-tok --bin qwen-bench
python3 scripts/bench/serve_realistic.py plan --family v41 \
  --seed 20261109 --seconds 180 --out plan.json
QWEN_METAL_LEASE_WAIT=1 nice -n 10 scripts/bench/with-metal-lease \
  --task steady-v41-default --for 6m -- \
  python3 scripts/bench/serve_realistic.py run --arm default \
  --binary target/release/qwen --tokenizer target/release/qwen-tok \
  --model MODEL.gguf --corpus code.txt --plan plan.json --out default
# Fresh server with --arm serial; then compare rates, allowing different counts:
python3 scripts/bench/serve_realistic.py compare serial default
```

Use `glm` for its pair; `fn` has serial baseline only on this engine commit.
The code corpus is exactly reproducible from **dabb97021e65f192a15057b3f226281cbe1ad81f**:
concatenate the following Git blobs in order, prefixing each with
`"\n\n// SOURCE: {path}\n\n"`: `crates/qwen-llm/src/deepseek_v41/forward/gpu_step.rs`,
`crates/qwen-llm/src/deepseek_v41/forward/attention_gpu.rs`,
`crates/qwen-llm/src/deepseek_v41/forward/ffn_gpu.rs`,
`crates/qwen-llm/src/glm5_next_metal.rs`, `crates/qwen-llm/src/qwen4exp_runtime.rs`.
This reproduces **660043 bytes** (verified against the old fixture). Prompt
controls are neutralized before tokenizing; roots and suffixes regenerate from
that corpus, the model vocab, commit and command.

Seven CPU checks cover independent infinite draws, boundary delivery, shared
window conservation, partial/censored requests, unequal-work rate comparisons,
partial reasoning replay and empty EOS. Contracts pass on **4120be7e**. This is
one finite-duration seeded observation per arm, intended for fast decisions;
small kernel changes may need another seed/window. There is no agent/task
completion-time claim across the different per-family workload distributions.

## Next levers under this workload

| Priority | Family / lever | Conditional value in this protocol | Rough work |
| --- | --- | --- | --- |
| 1 | Flash-Next continuous batching plus three retained live states | Target 1.3–1.6x its 67.2 tok/s baseline, roughly 87–108 tok/s; remove the 69–71 s observed queue waits. V4.1/GLM show the opportunity, not a measured FN result. | Existing batching lane; several days plus this qualification. |
| 2 | V4.1 routed-expert batch tiles | With no solo-stream time, qualify B=2/3 first. A 10% reduction in decode service time would target about 6–10% more delivered capacity, approximately 5–9 tok/s; tool gaps/appends dilute the ideal 11.1%. | Active tile lane, a few days including exactness and whole-loop checks. |
| 3 | GLM remaining few-row Q6/IQ decode work | Default already includes expert activation reuse. Another 10% reduction in decode service targets about 6–10% capacity, approximately 4–7 tok/s. This is a planning estimate, not another measured reuse gain. | Targeted B=3 census and kernel work, a few days. |

V4.1/GLM also have 2.2–2.3 s maximum content gaps while other agents append.
A smaller fair prefill quantum could reduce those stalls, with a possible
throughput tradeoff; there is no measured capacity gain to assign yet. The
[128K roofline screen](2026-11-09-128k-roofline.md) is separate from this warm
48K agent observation.
