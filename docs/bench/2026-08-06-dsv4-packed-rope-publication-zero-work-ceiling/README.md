# DeepSeek V4 Packed RoPE/Publication Zero-Work Ceiling

Status: `HOLD - closed inconclusive`. A pre-coding model-free profiler screens
the exact joint chronological/attention candidate before any kernels are
written. Both permitted campaigns fail the preregistered 5% stationarity gate.
No economic decision, kernel implementation, asset run, retry, or sample
filtering is authorized.

The observed optimistic upper estimates, 46.299 and 65.659 ms, are far below
the 158.3 ms floor and are useful suggestive negative evidence only. They are
not decision-grade because each campaign is unstable under its frozen protocol.

## Question

Standalone N=128 attribution put chronological publication near 73 ms and the
attention body near 123 ms across 43 layers, but neither cleared its 150 ms/15%
authorization gate. Those measurements came from different sampled encoder
topologies and cannot be added.

The proposed exact joint candidate would:

- batch forward query RoPE;
- fuse KV RoPE with canonical F16 raw-ring publication; and
- fold inverse RoPE into the cooperative packed-attention epilogue.

It would retain exact compressor publication, F16 cache state, attention score
and softmax order, same-token visibility, and all output bits. Before paying for
three kernels and a complete exactness matrix, this packet asks whether the
entire current dispatch ladder is large enough to clear the existing economic
floor even if the candidate removed it for free.

## Optimistic Ceiling

One N=128 layer currently executes:

- 127 forward query-RoPE dispatches;
- 127 forward KV-RoPE dispatches;
- 128 F32-to-F16 raw publication dispatches; and
- 127 inverse attention-RoPE dispatches.

That is 509 dispatches per layer and 21,887 across the production 43-layer
sequence. The profiler executes all 43 ladders in one command buffer and one
serial encoder with real production-sized tensors:

- query and attention: F32 `[32768,128]` each;
- KV: F32 `[512,128]`; and
- raw ring: F16 `[512,128]`.

It uses the actual two SWA and 41 compressed/YARN per-layer RoPE parameter sets.
The disjoint regime owns 43 independent banks; the warm regime owns a separate
44th bank reused across layers. This consumes about 1.4 GB without loading model
weights. Current kernels materialize every rotated F32 intermediate and F16
destination. Row-one query and attention must change, KV must remain finite,
and raw row one must equal the exact F16 conversion of final KV row one.

The screen credits the candidate with removing every dispatch, arithmetic
operation, and memory access even though a real implementation retains at least
three dispatches and all arithmetic. It omits unchanged compressor and attention
work. This is deliberately favorable to the candidate.

After separate first touches and an empty command bracket, each campaign records
12 disjoint and 12 warm commands in balanced ABBA order. P95 is nearest-rank and
therefore the maximum of 12 samples. For each regime:

```text
drift = 2 * (max - min) / (max + min)
range_ms = max - min
```

The optimistic upper estimate is:

```text
U = max(disjoint_p95, warm_p95)
    + max(5 * max(disjoint_range_ms, warm_range_ms), empty_gpu_ms)
```

Both regime drifts must be at most 5%. Stable `U < 158.3 ms` would KILL kernel
coding; stable `U >= 158.3 ms` would permit only model-free implementation. An
unstable campaign is `INCONCLUSIVE`.

The 158.3 ms floor rounds up
`max(150 ms, 0.15 * 1054.962 ms) = 158.2443 ms` from the accepted pre-expert
anchor. It is not inferred by adding prior phase estimates.

## First Campaign

The first campaign first-touches each regime once, then records the timed ABBA
sequence.

| Metric | Disjoint | Warm |
|---|---:|---:|
| GPU range | 36.119-36.655 ms | 27.931-29.860 ms |
| P95 | 36.654625 ms | 29.859958 ms |
| Drift | 1.4723% | 6.6752% |

The warm failure is consistent with a startup-conditioning transient: 27.9311
ms is uniquely the second warm command in the first timed ABBA block. The
frozen formula gives
9.644168 ms uncertainty and `U = 46.298793 ms`, but warm drift exceeds 5%.
The test reports `INCONCLUSIVE` before its deliberate stability assertion.

Deleting the sample would be retrospective filtering. CX authorized one
measurement-condition repair: retain both first touches and the empty bracket,
then execute one complete untimed disjoint/warm/warm/disjoint block immediately
before a fresh full campaign. Every formula, sample count, gate, and outcome rule
remains unchanged.

## Conditioned Campaign

The repaired campaign keeps every timed sample. It exhibits broader late
nonstationarity rather than repeating only the first transient.

| Metric | Disjoint | Warm |
|---|---:|---:|
| GPU range | 36.008-40.950 ms | 28.879-33.634 ms |
| P95 | 40.949750 ms | 33.633875 ms |
| Drift | 12.8431% | 15.2126% |

The final three disjoint samples step to 40.571/40.877/40.950 ms, and one late
warm sample reaches 33.634 ms. The frozen formula gives 24.709374 ms uncertainty
and `U = 65.659124 ms`. Both stability gates fail, so the result remains
`INCONCLUSIVE`.

## Decision

Enforce the preregistered stop. Label the exact joint candidate
`HOLD - closed inconclusive`, remove the live profiler, and write no kernels.
Do not delete samples, add another conditioning block, run a third campaign,
reinterpret either upper estimate as a KILL, or pay a current-asset run.

The two unstable estimates are consistent with an uneconomic candidate but do
not authorize that claim. Reopen only after a material premise changes: stable
production-level GPU instrumentation, an independently identified and corrected
timing-environment cause, or relevant device/compiler drift. A new attempt needs
a separately reviewed and preregistered protocol.

Move the active packed-prefill queue to the direct production-shape grouped
MXFP4-down ceiling for the two current outlier layers. Further HCA tiling remains
deferred.

## Provenance

- Base revision: `1b912e6ac1915f630f398ef8ad845781eae5ae62`.
- Device: Apple M4 Max.
- Conditioned-campaign release test binary SHA-256:
  `7aebd59795bb62d8848ed01f48825e1f8176783bb8720a53113f5cae7b6ff59b`.
- First source diff SHA-256:
  `8c4f703cb00f8c6f14a50a83fa4052a7f564dbd6cbfcf9a257b809055aa27f61`.
- Conditioned source diff SHA-256:
  `c18952ac812261016d3b20922e2fb68ee8289cb59d920a51d6f53ed68133e5d1`.
- First raw log SHA-256:
  `369b22242fe920ef783b3801e3e4dc3d4815cc399872103e68570effacd164eb`.
- Conditioned raw log SHA-256:
  `84a09baa6db0add1ad82d564dd2e4cdb017f3af8172b26e32a80f4179898c868`.
- First/conditioned elapsed test times: 1.18/1.35 seconds.

Both source diffs reconstruct their exact uncommitted profiler from the base
revision. The conditioned diff supersedes the first only by one untimed ABBA
block. The live source is removed after archival.

Command for both campaigns:

```bash
cargo test --release -p qwen-llm --lib \
  deepseek_v4_metal::prefill::tests::\
profile_packed_rope_publication_ladder_zero_work_ceiling \
  -- --ignored --exact --nocapture --test-threads=1
```

CX design, static review, repair authorization, and disposition:
`019fd588-96b0-7033-afd1-66d7e354e523`.
