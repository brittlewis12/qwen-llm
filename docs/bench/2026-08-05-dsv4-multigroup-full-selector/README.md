# DeepSeek V4 Multi-Group Full Selector

Date: 2026-08-05

Status: test-only 32-group Phase B `GO`; production default `HOLD`.

## Question

Phase A proved that exact global threshold discovery could fit comfortably
inside the terminal radix4 budget. This checkpoint asks whether deterministic
cache-order compaction, fail-closed validation, and complete output publication
still clear the frozen all-in selector gate.

The gate requires both mixed and all-tied terminal scores to preserve every mask
bit, ID, count, and status while candidate GPU and wall median remain at most
1.35 ms, p95 at most 1.40 ms, savings against the faster controls reach at least
0.50 ms, and control drift remains at most 5%.

## Topology

The candidate is exactly 18 ordered dispatches:

1. Eight histogram producer/reducer pairs resolve the deployed descending F32
   finite key four bits at a time.
2. The final reducer assigns each contiguous partition its greater/equal counts,
   lower-row threshold-tie quota, selected count, and cache-order ID offset.
3. Thirty-two 256-thread compactors scan contiguous lane chunks twice. They
   write disjoint one-byte private mask rows and disjoint ascending ID ranges,
   then publish generation-tagged completion records last.
4. One 256-thread publisher validates geometry, final state, all plans and
   records, mask binary values/population, strict ascending in-range IDs, and
   ID-to-mask membership. It fully overwrites public output and publishes count
   and status only after a device fence.

Invalid geometry is status `1` and reads no scores/private payload. Visible
nonfinite input is status `2`. Missing/stale state, records, impossible plans,
or malformed private output become status `3`. Every failure publishes the
same deterministic first-K fallback as the current selector. The publisher
structurally validates the compacted payload; score membership remains proven by
the exact compactor differential rather than recomputed a third time.

## Identity

- Base revision: `6bdc73af40c39dabc89659b82da3661268c9153c`.
- Candidate source and raw log are captured by the checkpoint commit.
- Hardware: Apple M4 Max.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`, LLVM 22.1.6.
- Metal: Apple metal 32023.864, target `air64-apple-darwin24.6.0`.
- Shape: one query, 262,144 visible/capacity rows, top-K 512, 32 groups.
- Campaign: five warmups, 24 samples/arm, current/candidate/current.

No model weights or resident model memory are used. Every tensor allocation is
outside timed commands; wall timing intentionally includes command creation and
the host encoding of all 18 dispatches.

## Correctness

Three active release tests cover:

- exact mixed and all-tied output;
- cutoff ties and canonical positive/negative zero/subnormal ties crossing
  partition boundaries;
- visible counts below K, invisible nonfinite values, visible nonfinite fallback,
  and invalid geometry fallback;
- fresh-generation repeat identity, missing histogram and compactor groups;
- non-binary private masks, duplicate private IDs, corrupt plan offsets, and
  corrupt completion records; and
- exact partition plans, final state, compact records, private payload, public
  mask/IDs/count/status, and deterministic first-K fallback.

```bash
cargo test --release -p qwen-llm --lib multigroup_selector_
```

Result: three passed, two explicit profilers ignored.

## Performance

All values are milliseconds per complete selector invocation.

| Case | Arm | GPU median | GPU p95 | Wall median | Wall p95 |
|---|---|---:|---:|---:|---:|
| mixed | current before | 1.873917 | - | 2.040000 | - |
| mixed | candidate | 0.673583 | 0.676833 | 0.818167 | 0.874708 |
| mixed | current after | 1.876625 | - | 2.039834 | - |
| tied | current before | 2.024521 | - | 2.203333 | - |
| tied | candidate | 0.907417 | 0.916875 | 1.074791 | 1.136792 |
| tied | current after | 2.027896 | - | 2.213229 | - |

Mixed saves 1.200333 ms GPU and 1.221667 ms wall against the faster
controls. All-tied saves 1.117104/1.128541 ms. GPU/wall control drift is
0.144%/0.008% mixed and 0.167%/0.448% tied. Both cases clear every frozen
gate. Candidate and both control endpoints remain exact outside timing.

```bash
cargo test --release -p qwen-llm --lib \
  deepseek_v4_metal::tests::profile_multigroup_selector_full_ceiling \
  -- --ignored --exact --nocapture
```

`run.log` retains all six 24-sample timing arrays per case and the exact command.
Its 6,310 bytes have SHA-256
`272769c145f6a8f4db9877d10ee72bb644abce592e73e6b4644d8402db0366a8`.

## Decision

Qualify the full Phase-B topology and freeze 32 groups. Do not switch production
yet: this packet does not own session scratch, memory accounting, a wrap-safe
generation counter, or eligibility below terminal history, and it does not run
inside all 21 CSA layers of a real token.

Next measure a small model-free visible-row crossover. Then route eligible
singleton Q=1/K=512 selection under an experimental policy while packed,
shallow, multiquery, and ranked-output paths retain radix4. The production gate
requires exact per-layer output, logits, causal-state digest, token transcript,
and consumed-ID trace plus at least `0.50 ms * eligible CSA layers` GPU and wall
saving against the faster control. CX review:
`019fcf7d-e9d4-7150-b496-e70a31958e80`.
