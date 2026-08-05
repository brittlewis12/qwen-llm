# DeepSeek V4 Packed Attention Attribution

Status: combined attention family `GO`; attention body `KILL` as a standalone
N=128 optimization; attention output pipeline `GO` as the next packed-prefill
target. Ordinary execution is unchanged.

## Question

The preceding chronological packet left about 1.0 second of current-asset
N=128 GPU work before Rust routing. It authorized one rotating boundary around
dense/sparse attention, inverse RoPE, and both attention output projections.
This packet asks two questions in sequence:

1. Is that combined family large and stable enough to decompose?
2. If so, does the attention body or output pipeline own the actionable cost?

The frozen component authorization floor is both 15% of ordinary pre-expert GPU
time and 150 ms after observer uncertainty. The signal must repeat in both CSA
and HCA cohorts. These are target-selection gates, not projected speedups.

## Instrument

The supported stage recorder uses serial compute encoders inside each unchanged
pre-expert command buffer. It records signed pass envelopes because adjacent
Metal pass timestamps may overlap. Physical starts and ends must be independently
monotonic, and every layer must close exactly:

```text
sum(stage duration) + sum(gap) - sum(overlap) = command span
```

The combined campaign uses three pre-expert passes per layer:

1. ingress through all chronological publication;
2. compressed-row lookup, dense/sparse attention, inverse RoPE, and both output
   projections; and
3. attention mHC post through router projection.

After that family clears, the split campaign uses four passes:

1. `BeforeAttentionBody`: ingress through chronological publication;
2. `AttentionBody`: compressed-row lookup, dense/sparse selection and attention,
   plus inverse RoPE;
3. `AttentionOutputProjections`: all packing, output-A projection, scattering,
   and output-B projection inside `encode_output`; and
4. `AfterAttentionOutput`: attention mHC post through router projection.

Ordinary arms retain one pre-expert and one post-route encoder per layer: 86
total. Combined sampled arms use 172 encoders; split sampled arms use 215. All
arms execute the same dispatch work. Recorder and routing code remain confined
to tests plus `dsv4-diagnostics`.

## Fixture And Protocol

- Base revision: `15ff783` (`perf(dsv4): attribute packed chronological work`)
- Device: Apple M4 Max
- OS: macOS 15.6.1 (24G90)
- Rust/Cargo: 1.97.1
- Asset: `deepseek-v4-flash-0731-ud-iq3_xxs-current-2026-08-04`
- Asset size: 104,207,848,032 bytes across four pinned shards
- Model content ID:
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`
- Prefix: token pattern `[35, 201, 200, 34]` repeated to N=128
- Continuation: canonical snapshot restore, then exact token ID 35
- Routing: Rust CPU routing and qualified 25-layer grouped IQ2 policy
- All-IQ3 widening: off
- MXFP4-down layers: unchanged fallback

Each campaign runs one untimed ordinary/sampled warm-up and timed `A/B/A/B/A`
arms. Every arm starts from a fresh session sharing one loaded residency.

Frozen gates:

- exact packed logits, normalized hidden output, restored continuation logits,
  state digests, committed tokens, dispatch count, and ordered full geometry;
- raw timestamp coverage within 2% per layer and 0.5% aggregate;
- each transition ambiguity at most 5%; combined ambiguity at most 7.5% per
  layer for three passes or 10% for four passes, and 2.5% aggregate;
- ordinary and sampled GPU drift at most 5%; sampled topology perturbation at
  most 10%;
- component repeat delta at most two percentage points and CSA/HCA component
  repeat delta at most three points; and
- split body-plus-output share within two points of each corresponding accepted
  combined share.

Representative split command:

```text
QWEN_DSV4_PACKED_GROUPED_EXPERTS=auto \
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_attention_split_attribution_packet \
  -- --ignored --exact --nocapture
```

## Exactness

All combined and split warm/timed arms preserve the same evidence:

- Dispatches per arm: 47,990
- Dispatch family/kernel/grid/thread geometry:
  `c87a6e9272f731016bd3f77a4d15f66c121ac557b60b0fbecddaa093801cc175`
- Packed logits:
  `45c414408cdc688eeeb4f40166013b2ac7261e5a2c2655cdf9f51e818941f2d5`
- Final normalized hidden:
  `8a9106b6c70c9069461ce1ef05d9dfb1349b6da7c4123b3e2e7b896c9d5278af`
- Causal digest:
  `3de61f2054ff7e540be5beb113c5c3884875b7018de800696aa58a0393535035`
- Prefix digest:
  `42428fa6efe61408f76b1ebc342f75cfa333af202b801c3031f34a7551cb870a`
- Compatibility digest:
  `e7fb64be2f9fd96c52c9d5aa9384e09652e899650558dd50c4ba998e834bf076`
- Continuation logits:
  `6c76cacc8242ec015421d7885efd1e9b6df5c1b075f2143cfee701181f3ade44`
- Continuation causal digest:
  `7bc950b04b735a172ccb3c5b467488f548407b8b38f497b304d5d4bef7ae0dd8`
- Committed tokens:
  `7c7d71da5df3b70e9d6f0555d949f7476acd5a68173331a228537c9e46dc728e`

The geometry digest binds ordered family, kernel, grid, and threadgroup shape.
It does not bind buffer addresses or scalar constants. Exact outputs and state
digests provide the complementary consumer evidence.

## Combined Result

Aggregate pre-expert GPU time:

| Arm | Run 0 | Run 1 | Run 2 | Drift |
|---|---:|---:|---:|---:|
| Ordinary | 1,065.052 ms | 1,092.055 ms | 1,058.807 ms | 3.092% |
| Sampled | 1,085.543 ms | 1,076.279 ms | - | 0.857% |

Sampled topology perturbation is +0.648%/+0.079%. Aggregate transition
ambiguity is 0.0069%.

| Metric | Sample 0 | Sample 1 |
|---|---:|---:|
| Raw combined target | 509.091 ms | 508.413 ms |
| Command share | 46.897% | 47.238% |
| Normalized ordinary estimate | 505.813 ms | 508.012 ms |

The normalized median is 506.913 ms. Applying the full 0.648% observer
uncertainty gives a 500.011 ms and 46.420% lower estimate, decisively above the
150 ms/15% split gate.

CSA target share is 46.349%/46.424%; HCA is 48.646%/48.548%. The two SWA layers
move more between repeats and carry no cross-cohort decision authority.

## Four-Pass Split

Aggregate pre-expert GPU time:

| Arm | Run 0 | Run 1 | Run 2 | Drift |
|---|---:|---:|---:|---:|
| Ordinary | 1,055.356 ms | 1,067.139 ms | 1,076.882 ms | 2.019% |
| Sampled | 1,051.888 ms | 1,077.942 ms | - | 2.447% |

Sampled topology perturbation is -0.882%/+0.553%. Aggregate transition
ambiguity is 0.0059%; the extra pass does not create a material timestamp seam.
The combined split share is 48.346%/47.257%, reproducing the accepted combined
packet within 1.449/0.019 percentage points.

| Component | Raw sample 0/1 | Share sample 0/1 | Normalized median | Lower estimate |
|---|---:|---:|---:|---:|
| Attention body | 122.332 / 123.056 ms | 11.630% / 11.416% | 122.900 ms | 113.488 ms / 10.641% |
| Output pipeline | 386.217 / 386.352 ms | 36.717% / 35.842% | 386.940 ms | 377.528 ms / 35.397% |

Component repeat deltas are 0.214 and 0.875 percentage points. Output share is
34.880%/34.689% in CSA and 39.289%/37.500% in HCA. Body share is
11.572%/11.583% in CSA and 11.873%/11.402% in HCA.

The output interval includes all pack/scatter work in `encode_output`; it is not
a matrix-arithmetic-only number. Both intervals are elapsed GPU pass envelopes,
not guaranteed removable cost or statistical confidence intervals. Charging the
entire 2.019% control drift would still leave roughly 365 ms of output work.

## Decision

Record the combined family as `GO` for decomposition.

Record attention body as `KILL` for a standalone optimization in this exact
synthetic initial-prefix N=128 cell. This does not kill deep-history attention,
where CSA selection and HCA history have separate measured scaling.

Record the output pipeline as `GO` and the next packed-prefill target. The first
candidate should attack the deployed Q8_0 output path without changing numerical
lineage:

- replace eight pack/projection/scatter triplets plus output B with one mapped,
  token-tiled output-A dispatch and one token-tiled output-B dispatch;
- preserve Q8_0 block traversal, product and reduction order independently for
  every token;
- write output-A groups directly into disjoint low-rank slices;
- retain the current scratch and 25-dispatch path as fallback initially; and
- admit no new session allocation or snapshot ABI.

The candidate remains a design handoff, not a speed claim. It needs model-free
intermediate/final bit differentials and a separate current-asset promotion
packet.

## Evidence

Combined authorization log:

- `target/profiles/dsv4-packed-pre-expert-stage/integration-n128-attention-output.log`
- SHA-256:
  `e735c536b742bb4f24c7dc89eaa7ceb40cbfa5ffdf65cf3e852130bf8be87cc0`
- Runtime: 22.53 seconds in-test
- Provenance limitation: exact source and executable hashes were not captured
  before the four-pass rewrite. This log authorizes only the subsequent split;
  it is not the final implementation checkpoint.

Accepted four-pass log:

- `target/profiles/dsv4-packed-pre-expert-stage/integration-n128-attention-split.log`
- SHA-256:
  `3fd96776530bd90903513aa790488cf314a81ce5a1ccfe96158b9d18a589278b`
- Runtime: 22.13 seconds in-test
- Reported swaps: zero
- Maximum process RSS: 3,128,098,816 bytes; unified Metal residency is not
  represented by this process-RSS counter

Final accepted source SHA-256:

- `deepseek_v4_metal.rs`:
  `bc39d52d7cf62782d1baec90f74d81e3b3dbacb85180aa213c7b408b85b5e92d`
- `prefill.rs`:
  `7987bb878af8963b93ba6826f7425914407654f419ea31127a1f3afc689083d7`
- release diagnostics test executable:
  `f623c458299c7a633d3af7d5b3496490eb5627cb90d51012a4fef113a0a1ce47`

Validation:

- two active signed-envelope resolver tests;
- archived prior-source combined log and ignored current-source split packet;
- formatting and patch whitespace;
- strict release workspace all-target/all-feature Clippy.

CX session `019fcf7d-e9d4-7150-b496-e70a31958e80` returned `GO` for the
family split, body `KILL`, output-pipeline `GO`, and a separate attribution
commit before implementation.
