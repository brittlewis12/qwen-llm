# Current-Source 8K CED: Homogeneous And Mixed Streams

Experimental source `2de8ce2f` composes current main `c8a3daaa` with the reviewed
runtime `dde9f8cc`, then adds explicit pair-only P4 mapping selection. This is
not a main/default promotion, an independent quality receipt, or a new kernel.

## Native Results

Metagross M5 Ultra, Maya-S IQ2_XXS six-shard artifact, seed 1, 8,192 prompt
tokens, capacity 8,192, actual selected width 4,096. Both arms use the same
4 GiB soft coalescing target and arithmetic. Larger singleton eligibility is
unchanged. Each process imports once, then uses fresh causal state per call;
"cold" here does not mean cold filesystem cache.

| Schedule | Full Median (s) | CED Median (s) | Lower CED Latency | Throughput Ratio |
| --- | ---: | ---: | ---: | ---: |
| Full block, then CED block | 9.777083 | 8.979566 | 8.16% | 1.089x |
| CED block, then Full block | 9.807589 | 8.871461 | 9.54% | 1.106x |
| Interleaved ABBA/BAAB | 11.415350 | 8.197147 | 28.19% | 1.393x |

Every row has four retained calls per policy. Every process also retains its
raw excluded warmup and two pilot calls per policy: fourteen chronological
calls per process, forty-two total. No samples are dropped or spread-vetoed.
This is one process per schedule, not a confidence interval or universal gain.

Retained samples in seconds, in per-policy chronological order:

| Schedule | Full | CED |
| --- | --- | --- |
| Full first | 9.745312, 9.777648, 9.780368, 9.776518 | 8.807563, 8.979756, 8.979376, 9.032877 |
| CED first | 9.808994, 9.806184, 9.803661, 9.809954 | 8.886101, 9.031342, 8.838132, 8.856820 |
| Interleaved | 11.435393, 9.780828, 11.618058, 11.395306 | 8.062100, 8.203180, 8.191115, 8.814199 |

The repeated homogeneous advantage survives reversing order. The larger mixed
separation also reappears, but must not replace the 8-10% homogeneous estimate.
These schedules demonstrate workload sensitivity; they do not isolate its cause
or prove an automatic crossover. No P0 comparison or new short-prompt win is
claimed by this run.

## Scope And Checks

- Host wall includes native prefill and endpoint logits copy; state allocation,
  fingerprinting and model loading are excluded. No stage instrumentation runs.
- All forty-two finite complete endpoint heads match bitwise, including across
  processes. No execution fallback occurs; all CED encoded commands complete,
  and observed incremental Metal allocations stay within the accepted bounds.
- Every case reports selected width 4,096. Early launch, preweight planning and
  result metadata agree on the 4,294,967,296-byte mapping target.
- All three runs finish naturally under the production lease and continuous
  host supervision, without health aborts, retries, cache purges or GPU settings
  changes. Source/binary identity matches and artifact stamps remain unchanged.
- The realized resource inventory matches the frozen P4 inventory: 1,002 tensor
  views, 178,188,648,448 unique bytes, maximum resource length 4,235,247,616;
  binding-layout SHA256
  `95b8179b243f532226d43d95d34118e1aa3f353fe01714e07b12139bc0919ce8`.
- Focused local CPU checks pass twelve library and thirty-eight CLI/startup
  tests, zero failed/ignored. Formatting, whitespace and inventory discovery
  checks pass. This is not a fresh full maintained/GPU qualification claim.

Endpoint agreement is not independent numerical acceptance or a full causal-state
and teacher-forced continuation comparison. Frozen full-state proofs do not
automatically extend to this composition. The existing independent F32 attention
discrepancy, narrowed-device fixture and CED-origin serving reuse remain separate
qualification questions; this experiment does not settle or weaken them.

Production mapping remains None and Full remains the default. The measured
experiment uses `suite --pp 8192 --depth 0 --runs 4 --seed 1 --v41-ced-pair
--v41-ced-pair-p4 --v41-ced-pair-production-timing`, with schedules
`full-blocks-first`, `ced-blocks-first`, then `interleaved`. Launch environment
requirements and timing scope are documented in `docs/BENCH.md`.

## Evidence Identity

- Source: `2de8ce2f1f3fc0eaba0b945af0db0c86d0f91dde`.
- Frozen benchmark SHA256:
  `66ec13ffbceecc4997e5ab6350896118ecb1e893ddfc3e7ef30720a7b44d0291`.
- Rust 1.99.1; macOS SDK 27.0; release default features. Known validation and
  injection environment settings are unset at child launch; external effective
  injection remains unattested, as recorded by the benchmark.
- Private archive: `v41-ced-current-8k-2026-10-25.dVpFQm`; remote run directory
  `2026-10-25-v41-ced-current-8k`. Cohort labels are
  `maya-8k-p4-full-first-r1`, `maya-8k-p4-ced-first-r1`, and
  `maya-8k-p4-mixed-r1`. Each retains stdout, stderr, continuous health events,
  controller identity and reduced observations; no new whole-artifact hash is
  claimed. The source-matched remote checkout remains frozen for replay.
