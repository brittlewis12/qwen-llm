# DeepSeek V4 Multi-Group Threshold Ceiling

Date: 2026-08-05

Status: threshold-only diagnostics `GO`; full output and production `HOLD`.

## Question

The current exact radix4 selector scans all visible scores eight times in one
threadgroup. At 262,144 rows it costs about 1.88 ms/layer on mixed scores and
2.01 ms/layer when all scores tie. Before implementing parallel compaction, this
checkpoint asks whether exact global threshold discovery can leave enough
latency budget for a useful complete selector.

The frozen Phase-A gate requires, for both mixed and all-tied terminal scores:

- exact state, per-partition counts, completion records, and repeated output;
- candidate median at most 1.00 ms and p95 at most 1.05 ms;
- at least 0.70 ms saving against the faster current control;
- at most 5% current-control drift; and
- no allocation, reset, or readback inside a timed command.

## Schedule

The candidate uses 256 threads/group and eight producer/reducer pairs, one pair
for each four-bit digit of the deployed descending F32 finite key. Producers own
contiguous row partitions and publish 16-bin histograms plus error and completion
metadata. A reducer validates every generation/digit/completion record, advances
the exact one-based rank, and retains per-partition greater/equal counts at the
final threshold.

Status `1` has invalid-geometry precedence without reading scores. Status `2`
means a visible nonfinite score. Status `3` means missing/stale completion,
invalid prior state, impossible rank, or final count inconsistency. Signed zeros
and positive/negative subnormals share the deployed canonical key. No mask, IDs,
compaction, or first-K fallback is emitted in Phase A.

## Identity

- Base revision: `158b1cac4d305f3a4e52075af8507f4d21542169`.
- Candidate source and raw log are captured by the checkpoint commit.
- Hardware: Apple M4 Max.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`, LLVM 22.1.6.
- Metal: Apple metal 32023.864, target `air64-apple-darwin24.6.0`.
- Shape: one query, 262,144 visible/capacity rows, top-K 512.
- Campaign: five warmups, 24 samples/arm, current/candidate/current.

No model weights or 104 GB residency are used. Scores are deterministic
production-shape synthetic values.

## Correctness

The active release differential covers mixed values, all ties, a threshold take
of one at the last eligible positive subnormal, and 512 canonical ties cycling
positive zero, negative zero, positive minimum subnormal, and negative minimum
subnormal across a partition boundary. It also covers invisible nonfinite
values, visible NaN, invalid visibility, and a deliberately missing producer.

Every valid case matches the CPU oracle and the current selector's threshold,
take, count, and status. Repeated records, counts, and state are exact. The stale
case propagates status `3` through digit seven and publishes matching error-3
completion records for every final producer.

```bash
cargo test --release -p qwen-llm --lib \
  multigroup_selector_threshold_matches_current_and_fails_closed
```

## Performance

All values are Metal command-GPU milliseconds per selector invocation.

| Groups | Case | Current before | Candidate | p95 | Current after | Drift | Saving | Gate |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 32 | mixed | 1.858688 | 0.350542 | 0.350750 | 1.857833 | 0.046% | 1.507292 | PASS |
| 32 | tied | 2.010125 | 0.599625 | 0.600333 | 2.010083 | 0.002% | 1.410458 | PASS |
| 64 | mixed | 1.859375 | 0.546500 | 0.551333 | 1.857354 | 0.109% | 1.310854 | PASS |
| 64 | tied | 2.010875 | 1.043771 | 1.044625 | 2.008625 | 0.112% | 0.964854 | FAIL |
| 80 | mixed | 1.859271 | 0.649563 | 0.654625 | 1.860313 | 0.056% | 1.209708 | PASS |
| 80 | tied | 2.011438 | 1.273417 | 1.274250 | 2.011146 | 0.015% | 0.737729 | FAIL |

The ignored profiler checks both candidate snapshots and both current-control
endpoints against the CPU oracle outside the timed intervals. `run.log` retains
all 432 raw timing samples and the exact command. Its 11,257 bytes have SHA-256
`898d384306b33c882dd1342f059bd0eae38f00cd24a062041aa21408cc50edd1`.

```bash
cargo test --release -p qwen-llm --lib \
  deepseek_v4_metal::tests::profile_multigroup_selector_threshold_ceiling \
  -- --ignored --exact --nocapture
```

## Decision

Freeze 32 groups. The threshold stage saves 1.410 ms even in the worst tied
case, leaving about 0.76 ms under the complete-selector median gate.

Proceed only to the frozen 18-dispatch Phase B: the existing 16 threshold
dispatches, one deterministic per-partition mask/cache-order compaction dispatch,
and one completion validator that alone publishes success or first-K fallback.
Require exact full outputs, median at most 1.35 ms, p95 at most 1.40 ms, at least
0.50 ms GPU and wall saving, and at most 5% control drift before production.

The 64- and 80-group variants are tied-case failures, not tuning candidates.
Packed, multiquery, ranked output, shallow routing, and production remain out of
scope. CX review: `019fcf7d-e9d4-7150-b496-e70a31958e80`.
