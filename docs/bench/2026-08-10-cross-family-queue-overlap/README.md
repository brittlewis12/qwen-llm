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

## Artifacts

- `qwen35-0p8b-b2.json`
- `qwen35moe-a3b-b2.json`
- `deepseek4-k160-b2.json`
