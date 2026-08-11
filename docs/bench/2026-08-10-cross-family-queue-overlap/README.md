# Cross-family queue-overlap baseline

## Question

Can one resident model safely execute independent sequence-private singleton
graphs on separate Metal queues, and does that provide enough aggregate
throughput to serve as a useful baseline for true static batching?

This is deliberately **not** a batching claim. Each independent command still
walks the complete singleton graph and binds the same weights separately. The
probe measures correctness, command-timeline overlap, aggregate makespan, and
the contention paid by each command.

## Contract

- Families: Qwen dense, Qwen hybrid/MoE, and DeepSeek V4.
- Workload: two distinct deterministic teacher-forced requests at position 1,
  one timed transition per request.
- Arms: resident serialized execution and per-step lockstep independent queues.
- Ordering: one untimed warmup pair, then AB and BA with the same measured seed.
- Correctness: Qwen compares every argmax ID plus each final full-logit SHA-256;
  DeepSeek compares each final full-logit SHA-256. Any mismatch aborts.
- Safety: one process-wide Metal lease, queue-scoped residency sets rejected,
  multi-session admission before allocation, checked allocation reconciliation,
  all committed commands drained, and cooperative cancellation.
- Authority: diagnostic queue-overlap evidence only. It cannot authorize static
  layer batching, continuous batching, or production latency claims.

The binary was necessarily dirty because it contained the new probe. Every raw
row records the same full source identity:
`git-source-sha256-v2:2ace0ae62d2b470448751650abbe19e181744e3aeb0ca4153b42a8db9f39045b`.

## Results

| family / asset | serialized aggregate tok/s | independent aggregate tok/s | speedup | GPU concurrency factor | exact evidence |
|---|---:|---:|---:|---:|---|
| Qwen 0.8B dense Q4_K_M | 352.05 | 507.05 | **1.440x** | 1.919 | yes |
| Qwen 35B-A3B Q4_K_M | 98.18 | 144.40 | **1.471x** | 1.953 | yes |
| DeepSeek V4 K160 | 28.38 | 38.84 | **1.368x** | 1.993 | yes |

The interval factor near two proves that both command buffers overlap, but not
that every kernel executes simultaneously or efficiently. Contention lengthens
the average command by roughly 34% on 0.8B, 33% on A3B, and 47% on K160. The
aggregate win comes from overlapping those slower singleton graphs.

Observed two-session Metal allocation deltas were 57.5 MiB, 160.8 MiB, and
8.12 GiB respectively. K160 released its Metal state and returned wired memory
to baseline, but the 98.6 GB peak induced host compression. No larger DeepSeek
cell is authorized from this packet.

## Static-batch crossover follow-up

The existing true-reuse probes were then rerun against the same source state.
They issue one projection dispatch over multiple independent rows and therefore
test actual per-dispatch weight reuse.

| asset / bounded scope | B=2 | crossover | B=8 |
|---|---:|---:|---:|
| Qwen 0.8B, one GDN layer | 0.500x | B=4 | 1.355x |
| Qwen A3B, one GDN layer | 0.469x | B=6 | 1.317x |
| Qwen 27B, one GDN layer | 0.699x | B=6 | **2.063x** |
| Qwen A3B, integrated 3-GDN + 1-attention slice | not run | B=8 gate | **1.176x** |
| Qwen 27B, all isolated projection families plus layout | 1.016x | B=2 aggregate only | **3.155x** |

The B=2 static mechanism is not worth productizing: complete GDN-layer replay
loses badly on every Qwen shape despite a near-flat aggregate projection sum on
27B. The useful static cohort begins around B=6-8. Dense 27B is the lead backend
candidate because its mandatory weight stream makes B=8 reuse multiplicative;
A3B remains viable but has a smaller integrated gain and known route-near-tie
correctness cliffs.

## Decision

1. Keep independent queues as an opt-in resident-service throughput fallback,
   with explicit per-request latency and memory admission. Do not call it
   batching.
2. Close the static B=2 implementation path.
3. Open a fixed, equal-position B=8 static lane, dense 27B first. Its next gate
   is one complete dense block, then one complete attention block, both against
   serialized and independent-queue controls.
4. Keep continuous/ragged batching blocked until a whole-model fixed-cohort
   backend beats both baselines.

## Generated-feedback follow-up

The original packet teacher-forced every transition. A schema-2 follow-up feeds
each selected greedy token into the next transition for 32 steps per client,
then requires every generated ID and the final full-logit digest to match
serialized execution. The prompt and first transition token remain distinct and
deterministic per client.

| family / asset | serialized aggregate tok/s | independent aggregate tok/s | speedup | GPU concurrency factor | 32-step evidence |
|---|---:|---:|---:|---:|---|
| Qwen 35B-A3B Q4_K_M | 98.28 | 135.48 | **1.378x** | 1.988 | exact |
| DeepSeek V4 K160 | 26.87 | 37.66 | **1.402x** | 1.995 | exact |

A3B serialized pair walls were `650.343/652.011 ms`; independent walls were
`474.513/470.319 ms`. Its two-session Metal delta was `170,229,760` bytes.
K160 serialized pair walls were `2,451.020/2,316.198 ms`; independent walls
were `1,738.087/1,662.708 ms`. Its two-session delta remained
`8,714,649,600` bytes. K160 returned wired memory to baseline after normal exit;
host compression increased, while swap use remained unchanged.

Both rows used source identity
`git-source-sha256-v2:9a0a21ea274f8d80e5dccd9ad6ba71fd61ccfc92723842d3891ed90b26d39140`.
The dirty source-matched evidence authorizes a product-shaped spike, not a final
throughput claim. The next slice is explicit file-JSONL concurrency two: serial
prefill, two independent queues during decode, ordered output, odd-tail serial
fallback, and up-front multi-session admission. It is resident concurrency, not
static or continuous batching.

## Qwen product integration

The authorized slice now ships as `qwen --requests-jsonl FILE --concurrency 2`
for dense and MoE Qwen models. It accepts heterogeneous prompt lengths and token
limits, prefills each request serially, overlaps complete singleton decode graphs
while both lanes remain active, and moves the surviving lane back to ordinary
serial GPU-greedy decode. Consecutive pairing and pair-atomic output preserve
input order; an odd final request uses the existing serial request path.

The safety boundary is explicit:

- regular files and greedy requests only;
- no prefix-cache mutation, prompt lookup, request stats, or request traces;
- model-derived, Metal-priced admission for two complete sessions plus the
  largest candidate or fallback prefill scratch and a 2 GiB transient reserve;
- mutable-buffer alias checks, independent logical positions, precommit
  frontier rollback, and cohort poisoning after any committed failure;
- queue-scoped model residency attached to both additional queues and removed by
  guards before those queues are released.

Two release smoke comparisons used three heterogeneous requests: 5/10/10 prompt
tokens, 3/8/1 requested output tokens, two paired transitions, five serial-tail
transitions, and an odd serial request. Every output object, generated-token
SHA-256, decoded text, stop reason, and terminal-frontier flag matched ordinary
serial JSONL exactly on both `Qwen3.5-0.8B-Q4_K_M` and
`Qwen3.6-35B-A3B-UD-Q4_K_M`.

The 0.8B admission priced `1,908,720` bytes of prefill scratch and
`2,209,935,920` required bytes including reserve. A3B priced `139,372,039` bytes
of prefill scratch and `2,456,494,751` required bytes. The implied session plans
are about 30.3 MiB and 84.8 MiB per lane; the latter agrees with the earlier
measured 170.2 MB two-session delta. One product-shaped A3B pair reported 139.1
aggregate generated token/s, but this smoke was not counterbalanced and carries
no new throughput authority. Its role is lifecycle and serial-equivalence
validation.

The A10B extra-queue residency attachment is implemented but deliberately not
model-run here: loading that asset would spend far more machine pressure than
this lifecycle gate justifies. DeepSeek uses a separate family-appropriate
executor, recorded next.

## DeepSeek product integration

DeepSeek V4 now shares the same `--concurrency 2` surface with a
family-appropriate executor. Two worker-local sessions share immutable residency
and independent Metal queues. The coordinator serializes prefill, then releases
both prepared workers into concurrent generation. This ownership shape avoids an
unsafe `Send` claim for mutable Metal sessions and preserves pair-atomic,
input-ordered stdout. An odd tail runs at effective concurrency one.

The first complete-worker arm overlapped prefill as well as decode. It was exact
but immediately falsified as a product organization: K160 pair wall was
`2,742.826 ms`, while individual prefills expanded to `2,391/2,450 ms` from
roughly `1,201/626 ms` in the serial control. The promoted scheduler therefore
keeps packed prefill serial and overlaps generation only.

On the same heterogeneous 3/8/1-token greedy fixture used for Qwen, K160
concurrency reproduces every serial output object and token SHA-256. A warm
representative pair spends `1,789.692 ms` in serial prefill and `309.129 ms` in
concurrent generation, versus roughly `343.9 ms` of summed serial generation.
The whole-pair movement is only about 3-4% because these tiny prompts are
prefill-dominated; the earlier 32-step packet remains the decode-throughput
authority.

A second fixture gives the two requests distinct temperatures (`0.7/0.8`),
top-k/top-p/min-p settings, and seeds (`123/456`). Concurrent and serial outputs
again match exactly. Concurrent generation takes `180.358 ms` versus
`230.8 ms` summed serial (`1.280x`), and whole pair wall moves about `1.049x`.
DeepSeek can safely retain request-local seeded sampling because each worker owns
its sampler; Qwen concurrency remains GPU-greedy only.

The load plan now admits residency plus two complete sessions before realizing
the model. K160 requires `99,149,463,552` bytes including its 512 MiB dynamic
reserve. The observed two-session delta is `8,682,995,712` bytes versus
`8,691,613,696` priced session bytes. A truthy
`QWEN_DSV4_RESIDENCY_SET` fails before model load because that set remains scoped
to one command queue. After validation runs, host free memory returns to 92-93%
and swap remains at 2.44 MiB.

## Artifacts

- `qwen35-0p8b-b2.json`
- `qwen35moe-a3b-b2.json`
- `deepseek4-k160-b2.json`
- Generated-feedback reports were acquired as
  `target/profiles/qwen-a3b-b2-generated-feedback.json` and
  `target/profiles/dsv4-k160-b2-generated-feedback.json`; the durable values and
  source identity are recorded above.
- Product smoke JSONL and stderr were disposable `target/` artifacts; all
  contract, identity-independent outputs, admission values, and conclusions are
  recorded in this document.
