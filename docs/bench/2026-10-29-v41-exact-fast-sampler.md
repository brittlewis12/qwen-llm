# V4.1 Exact Fast Sampling: Key-Ordered Sampler and Pooled Batch Sampling

2026-10-29. Branch `perf/sampler-exact-fast`: sampler `dac14284`, batch
sampling pool `1a8c124c`, based on main `b4595136`. M5 Ultra, GSQ IQ2_XS,
`qwen serve` defaults (max active 4), temperature 1.0, 256 output tokens,
fixed request seeds; ABBA within one lease per comparison. CPU replay:
`qwen-bench sampler-replay` on 64 synthetic logits rows at vocab 129,280
(normal logits, sigma 1.5-3.5, eight boosted candidates per row), ten
passes, ABBA.

## Decision

Both changes keep sampler-v1 bit-for-bit (same tokens, candidate indices,
totals, errors, draw counts and RNG state) and both land:

- Positive-temperature sampling at the V4.1 vocabulary is **53% cheaper**
  (1.171 -> 0.550 ms per sample, release default config).
- Three concurrent requests: **+12.6% throughput** (94.4 -> 106.3 tok/s
  wall; owner time outside the forward 4.75 -> 1.78 ms per round of ~31 ms).
  A single request decodes 3.8% faster.
- Output digests of every seeded request are identical across old, new,
  owner-serial and pooled arms.

## Design

Sampler (`Sampler::sample`, `crates/qwen-llm/src/sampling.rs`):

- Keys `(descending f32 bits) << 32 | token` are built in token order into
  buffers each sampler reuses (clones start empty). A **stable** LSD radix
  sort of only the 32-bit logit half (3 x 11/11/10-bit passes, histograms
  taken while building, uniform digits skipped) yields the comparator order,
  since stability supplies the ascending-token tie-break.
- Top-k selects and sorts distinct whole `u64` keys.
- Logits decode exactly from the keys. min-p and the `+inf` filter are
  prefix searches. Weights keep the same f64 division (skipped only when
  t == 1, where it is the identity), subtraction and libm `exp`.
- One sequential fold from `-0.0` (which is what `Iterator::sum` does)
  stores the partial sums; top-p and the draw binary-search them. Partial
  sums of non-negative weights never decrease, so the search finds the same
  index as the linear scan.

Batch sampling (`serve/backend_v41/sample_pool.rs`, `scheduler.rs`):

- After a multi-member step, each member's `Sampler` and logits move by
  value to `QWEN_SERVE_V41_SAMPLE_WORKERS` persistent threads (default max
  active - 1; 0 = serial) and back before any member is consumed.
- Consumption, publication, cancellation, failure and reuse commits stay on
  the owner, in the original order.

## Evidence

Bitwise, lib tests:

- `sample` was checked against the kept candidate-vector reference over
  672 configs x 99 rows of length 0-2049, covering ties, signed zeros,
  subnormals, `+/-inf`, all `-inf`, huge ranges, random bits and NaN, with
  samplers reused across lengths and cloned mid-stream.
- The same check ran over 36 configs x 9 rows at 129,280.
- The binary searches were checked against the linear scans at and around
  every partial sum.
- Mutations (unstable sort, reciprocal temperature, draw `<` vs `<=`) fail
  these tests.

Serve tests: sampled requests over spread logits with 0/1/3/5 workers equal
each other and the serial path, and a NaN-logits member fails alone either
way. Mutations (ignoring the presample, cross-wired samplers) fail them.

CPU replay, ms per sample (mean of ABBA pairs):

| config | old | new | change |
| --- | ---: | ---: | ---: |
| t 1, top-p 1 (V4.1 default) | 1.171 | 0.550 | -53% |
| t 1, top-p 0.95 | 1.175 | 0.537 | -54% |
| t 1, top-k 40 | 0.195 | 0.138 | -29% |
| t 0.7, min-p 0.05 | 0.848 | 0.319 | -62% |

Digit width: 2 x 16-bit passes were 5% slower than 3 x 11-bit. Computing
weights inside the last scatter was 9% slower than a separate pass, because
a second scattered stream costs more than the overlap saves.

Remaining default-config cost on a development host: key build plus
histograms 0.11 ms, three scatter passes 0.22 ms, and exp plus partial sums
0.26 ms (bound by libm `exp`).

Serve, old `b4595136` vs new `1a8c124c` (ABBA, 2 rounds per arm):

| metric | old | new |
| --- | ---: | ---: |
| single: decode ms (median) | 3589 | 3451 |
| single: owner ms/token outside forward | 1.38 | 0.75 |
| 3 concurrent: wall ms (median) | 8135 | 7226 |
| 3 concurrent: tok/s | 94.4 | 106.3 |
| 3 concurrent: owner ms/round outside forward | 4.75 | 1.78 |

Pool attribution, new binary, workers 0 vs 3 (ABBA): owner time outside the
forward was 3.08 -> 1.87 ms per 3-wide round (about 4% of a round), and the
single request was unchanged at 0.78 ms. GPU step time drifted 15.4 -> 13.6
ms/token across that lease, so its wall-clock ratios (0.95 concurrent) are
not used.

## Limits

- Replay rows are synthetic; no captured V4.1 logits were used. The serve
  A/B uses real logits but only four seeded requests.
- Each sampler keeps about 24 B per vocabulary entry (3 MB at 129,280) for
  the life of its request, within the existing 64 B/vocab caller CPU reserve.
- The top-k path is still a `u64` select (0.14 ms at k=40).
- `exp` is now the floor of the full-vocabulary path. Going lower exactly
  would need parallel exp within a request or fewer sorted elements; both
  are untried.
