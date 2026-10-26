# V4.1 Exact Mask Publication: A Small Fused Win, A Rejected Two-Dispatch Path

## Decision

Promote the bounded single-dispatch publisher, not ordered clear/scatter.
An independent reversed-order real-model confirmation measures **0.359% lower
TG latency**, not the drift-inflated 7.811% pooled result from the first window.
This is a small exact-work-removal gain, not a large decode multiplier.

The fused kernel builds a temporary membership bitset in 16 KiB of threadgroup
memory, then emits the **same dense device mask**. There is no global mask
representation change, allocation reduction, new environment knob or quant gate.
Default eligibility requires a serial encoder, actual pruning, clamped retained
count >=2048, domain <=131072 words and supported threadgroup capacity/pipeline
width. Other shapes and concurrent encoders keep the incumbent. Invalid IDs and
duplicates retain their existing membership semantics.

The two-dispatch clear/scatter candidate is rejected: its real-model comparison
is 3.197% slower despite favorable publication-only and reduced-path results.
Its experimental implementation and negative evidence remain available; it is
not a production alternative selected by this delivery.

## Real-Model Protocol

- Metagross, M5 Ultra 256 GiB, SDK 27.0, Rust 1.99.1; default native execution,
  no validation/census/performance overrides in timing children.
- `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf`, four weight
  shards plus disk-backed `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`. Resident
  payload is 155220506880 bytes, approximately 144.56 GiB. Existing full LFS
  verification is retained; fresh before/after file stamps are not new content
  hashes. See [artifact baseline](2026-10-25-v41-gsq-baseline.md).
- Ordinary native familybench, Full/None in both arms:
  `qwen-bench suite -m MODEL --tg 128 --depth 16640 --runs 2 --seed 1 --no-warmup -o json`.
  Every repetition creates fresh causal state and performs its untimed depth
  fill before timed single-token steps. `--no-warmup` does not mean cold files.
- Inputs are the normal deterministic synthetic familybench IDs: depth seed
  `1 ^ DEPTH_SEED_SALT`, timed seed `1 + rep_index`. They are not the separate
  shared-text generation witness, and synthetic TG supports no quality claim.
- Each native binary self-acquires the production Metal lease. Another lane can
  run between processes, never concurrently. Queuing, raw chronology, host
  health and all samples are retained, with no retries or spread/win filters.
- All three tools are frozen before their build lease is released. Full source,
  binary/build identity and five model-file stamps agree before and after each
  process. Both fused windows use exactly the same frozen binaries.

## Fused Publisher: Two Windows, All Samples

Control `d66c0f0567b208c987917b85ec83f611e6e77c6a` composes main's
`3e13af20` expert staging with incumbent publication and the test fixtures.
Candidate `77d5c50e3cd65d021f7711528db28345d0ec1b6d` adds the fused default.
Both include the same untimed stderr depth-fill receipt. Every one of their
sixteen depth fills attests start 0, 16640 tokens, selected width 4096, Full
execution and no fallback. These reports do not contain capacity.

| Window / Process | Arm | Retained TG128 Latencies (s) |
| --- | --- | --- |
| Initial ABBA / `mask-fused-control-b1` | Incumbent | 3.194222166, 3.192993500 |
| Initial ABBA / `mask-fused-b1` | Fused | 2.734719124, 2.738582791 |
| Initial ABBA / `mask-fused-b2` | Fused | 2.740389958, 2.741352415 |
| Initial ABBA / `mask-fused-control-b2` | Incumbent | 2.748746750, 2.750212874 |
| Confirmation BAAB / `mask-fused-b3` | Fused | 2.737853417, 2.741291042 |
| Confirmation BAAB / `mask-fused-control-b3` | Incumbent | 2.751698541, 2.750221041 |
| Confirmation BAAB / `mask-fused-control-b4` | Incumbent | 2.749610667, 2.748993500 |
| Confirmation BAAB / `mask-fused-b4` | Fused | 2.741126083, 2.738954958 |

Initial pooled medians are 2.971603187/2.7394863745 s, arithmetically 7.811%
lower latency. **Do not attribute that ratio to the optimization:** the control
shifts from 3.193607833 to 2.749479812 s across the window. Relative to that late
control, the pooled candidate is only about 0.363% faster. Lease exclusion alone
does not establish identical cache/thermal/device state or diagnose this shift.

That observation motivated one fixed reversed-order confirmation, not a rerun
or removal of the first window. Its pooled incumbent/fused medians are
**2.749915854/2.7400405205 s**, 0.359114% lower latency, approximately 1.00360x
throughput. Every candidate sample is faster than every control sample in this
confirmation. The separation also agrees with the first window's late control.
This modest result, exact membership proofs and bounded fallback justify the
promotion. No further dimension or timing sweep is used to select the claim.

## Rejected Clear/Scatter Path

Frozen control `48e5062324b3c3660470c0f0277bc4a63aa6bcd5` and runtime candidate
`d88abd705c1886a6d61c4b859fa3de66108df1c1` used the same actual-model command,
Full/None and source ABBA, two fresh-state repetitions per process:

| Process | Arm | Retained TG128 Latencies (s) |
| --- | --- | --- |
| `mask-control-b1` | Incumbent | 3.223567542, 3.256487166 |
| `mask-candidate-b1` | Clear/scatter | 3.341088542, 3.363560959 |
| `mask-candidate-b2` | Clear/scatter | 3.356080709, 3.379192999 |
| `mask-control-b2` | Incumbent | 3.278196250, 3.254990750 |

Pooled medians are 3.255738958/3.359820834 s, **3.196874% higher latency**.
Every candidate sample loses to every control. No causal mechanism for this
regression is diagnosed here. Old ordinary Full TG rows lack per-repetition
geometry; do not borrow that attestation from the separate generation witness.

The prerequisite 48e leaf retained 144 valid-unique native samples across domains
2049/16384/131072, 2048 retained IDs, ascending/synthetic-permuted order and three
ABBA/BAAB rounds. All six case medians favor clear/scatter, but these are four
publications/sample, not model timings. The real four-layer producer -> mask ->
follower selection/attention fixture then retained two 24-sample processes:

| Process | Host Median, Incumbent -> Candidate (us) | GPU Median (us) |
| --- | ---: | ---: |
| R1 | 2872.771 -> 2629.313 | 2048.833 -> 1800.604 |
| R2 | 2004.1875 -> 1954.479 | 1351.688 -> 1240.792 |

R2's middle GPU round loses; R1 has substantial common drift. Neither reduced
fixture result overrides the model negative. Clear/scatter costs 229 dispatches
versus 228 incumbent in that fixture; fused correctness preserves the incumbent
228-dispatch span and verifies its single-workgroup publication.

## Correctness And Limits

- Eighteen focused CPU publication contracts pass. The new contracts cover 448
  default-policy combinations, forced-arm precedence, domain rounding,
  threadgroup limits and independent strided-bitset membership at five widths.
- M4 Max and M5 Ultra each pass the explicit 144-case incumbent/clear-scatter/
  fused grid, actual producer/follower correctness, and existing 16385 pruning
  boundary regression with validation enabled. Masks/IDs are exact; existing
  output, persistent-state, scalar-continuation and release checks are retained.
  The reduced attention comparison keeps `2e-5 + 4e-6*abs(reference)`.
- Separate shared-text generation, 16640 input +64 greedy output tokens, agrees
  in finish `token_limit` and fingerprint
  `0e3efafccdcaeb70110d0bcb4f21649f672e57f206148ff6b84ba67ad7df287d`.
  Generation reports selected4096/capacity16703/Full/no fallback in both current
  arms, as did the earlier clear/scatter pair. This is bounded engine agreement,
  not independent quality, all-logit or full-state equivalence.
- No M4 speed, prefill-rate, million-token, serving-reuse or universal gain is
  claimed. Independent prefill precision and artifact quality remain distinct
  qualifications, not repaired by an exact integer membership transformation.

## Evidence And Reconciliation

Raw archive: `v41-mask-pipeline-2026-10-25.UD1pU6`; remote run directory basename
`2026-10-25-v41-mask-pipeline-current`. Process stems in the tables index raw
JSON, stdout/stderr, events, frozen identity, before/after and verification
receipts. Integer nanoseconds are authoritative. Source bundles and frozen
executables remain with the experiment, not replaced by installed-tool builds.

`mask-fused-model-comparison.json` and `mask-fused-model-confirmation.json`
independently check chronology, raw hashes, all samples, native settings,
geometry and unchanged source/binary/model stamps. The confirmation intent links
the original packet's digest. Current frozen qwen-bench SHA256 identities:

- Incumbent: `0dab85664917ca767965b70bf3c79fb20cd07d8bace8de36650010b83acbdc32`.
- Fused: `2a7668083e30c3697dd4fa6c7d5bb1772c2c9bcb73eed4eb1f00476f70feec1c`.

Preserve two earlier private postparser failures: the valid-unique controller
confused six cases with 144 samples; the first old model-control parser expected
a full SHA/bool dirty instead of canonical short SHA/integer zero. Offline
checks reconcile the complete raw records without re-execution or discarded
observations. Preserve the initial canceled lease wait as a queue cancellation,
not a compiler/kernel failure. None changes a workload or binary.
