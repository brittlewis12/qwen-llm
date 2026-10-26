# Current-Source 8K CED: Homogeneous And Mixed Streams

Timing source `2de8ce2f` composes main `c8a3daaa` with the reviewed runtime
`dde9f8cc`, then adds explicit pair-only P4 mapping selection. Its timings are
not an independent quality receipt or evidence for changing defaults. The
subsequent cold-profile delivery checks are recorded separately here.

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

Endpoint agreement alone is not independent numerical acceptance or a full
causal-state and teacher-forced continuation comparison. The subsequent bridge
closes the latter bounded question; the independent F32 attention discrepancy
and CED-origin serving reuse remain separate and unwaived.

Production mapping remains None and Full remains the default. The measured
experiment uses `suite --pp 8192 --depth 0 --runs 4 --seed 1 --v41-ced-pair
--v41-ced-pair-p4 --v41-ced-pair-production-timing`, with schedules
`full-blocks-first`, `ced-blocks-first`, then `interleaved`. Launch environment
requirements and timing scope are documented in `docs/BENCH.md`.

## Cold-Profile Delivery

Delivery source `abda018e` passes the existing strict comparison through an
explicit P4 selector on M5, with unchanged comparators: all 257 complete
129,280-logit vectors match bitwise and all 56 causal-region hashes agree at
every checkpoint, including 256 true teacher-forced steps per arm. One
resident import and one live KV state are retained; twelve CED commands complete
without fallback. The entire 257-observation sequence also matches frozen P4
`7398cb76`. Stable shard stamps, metadata/layout hashes and normalized controls
match that retained reference after accounting for the report's separate-source
schema. No fresh whole-artifact content hash is claimed.

The corrected public-API witness also passes on this source: N=2050/capacity2051
selects a 2048-row admitted request on an unchanged 4096-row-preference model,
then checks Full/CED equality and release. This does not retrospectively qualify
the older N=2049 negative fixture.

Actual `qwen run` and ordinary `qwen-bench suite` pass validation-on 8K smoke
checks using `QWEN_V41_PREFILL_EXECUTION=shrinking_ced`: the selected cold storage
plan carries P4 into loading, actual execution is CED with width 4096 and no
fallback. Run uses the exact frozen 8192 prompt IDs after a checked text round
trip and generates two tokens, including scalar continuation. These are frontend
checks, not new speed measurements. Default/explicit Full and serving stay
Full/None; optional pre-mutation Full fallback retains the opted-in P4 mapping.

Independent source review finds no P1/P2 blockers. The maintained default gate
passes at `f011b6cd`: 474 exact CPU tests (307 library, 33 oracle, 53 bench,
81 qwen), ten comparison tripwires and fourteen source/Engram Python checks.
All selected Rust tests report zero ignored/skipped. The first gate stopped on
two unclassified soft-target constants; exact-token audit explanations fix that
without changing the scanner or runtime. There are no Rust/Metal changes from
`abda018e` to that gate source, only identity-audit metadata.

This qualifies the explicit cold profile in the stated scope, not automatic
policy selection, short-prompt speed, independent task quality or serving reuse.
Receipts are `delivery-p4-state-n8192.*`, `delivery-admitted-2k.*`,
`delivery-product-smoke.verified.json`, and `delivery-f011b6cd-contracts.log`
in the same private evidence archive. Frozen core witness SHA256:
`9050159731351f39876bde064473c1a7d32bbc0f214e8a6772702a5703718d19`.

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
