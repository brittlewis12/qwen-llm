# DeepSeek V4 Packed Chronological Attribution

Status: profiler/evidence checkpoint `GO`; chronological row publication is
`KILL` as the standalone next packed-prefill optimization at N=128. It remains
a possible piggyback optimization. Attention body plus output is the next
rotating attribution target.

## Question

The qualified all-IQ3 widening left about 1.057 seconds of current-asset N=128
GPU work before Rust routing. That interval includes attention, compressor and
mHC work, chronological cache/frontier publication, and router projection. This
packet asks whether the position-ordered row loop alone is large enough to
justify a dispatch-batching or fusion project before measuring another family.

The frozen authorization floor is both:

- an uncertainty-adjusted lower bound of at least 15% of ordinary pre-expert
  GPU time; and
- approximately 150 ms across the 43-layer N=128 prefix.

The signal must also appear in both CSA and HCA cohorts rather than one outlier
layer.

## Supported Instrument

Apple M4 Max supports stage-boundary timestamp counters but not dispatch-boundary
sampling. A timestamped stage therefore requires its own serial compute encoder
inside the unchanged command buffer. The final diagnostic topology uses exactly
three pre-expert encoders per layer:

1. `BeforeChronological`: raw-ring preservation through compressor projections.
2. `ChronologicalRows`: all 128 ordered rows of Q/KV RoPE, raw-cache scatter,
   compressor frontier writes, pooling, normalization, publication, and roll.
3. `AfterChronological`: attention body/output through router projection.

The target starts immediately after `encode_layer_projections` and ends before
the compressed-row lookup and attention branch. Ordinary controls retain one
pre-expert encoder per layer. Both arms retain one post-route encoder per layer.
Thus controls execute 86 encoders and sampled arms execute 172; dispatches and
their order do not change.

The common Metal census now retains all grid and threadgroup dimensions in
addition to flattened launch volumes. The packet hashes stage family, kernel,
and all six dimensions for every dispatch in order.

`KernelEncoder::try_begin_sampled` makes sampled-pass creation fallible. Sample
buffer creation occurs before session mutation. A later encoder failure ends
the prior pass through RAII, prevents command commit and profile publication,
and leaves the already-mutating session poisoned.

All recorder, resolver, and live routing code is test plus diagnostics only.
Ordinary packed prefill retains one serial encoder and no timestamp allocation,
branch, environment switch, or profile publication.

## Timestamp Semantics

Metal does not guarantee disjoint adjacent compute-pass intervals. The resolver
therefore records pass envelopes and signed transitions:

```text
duration = end - start
delta = next_start - previous_end
gap = max(delta, 0)
overlap = max(-delta, 0)
```

It requires physical starts and ends to be independently nondecreasing and
checks the algebraic ledger exactly:

```text
sum(duration) + sum(gap) - sum(overlap) = final_end - first_start
```

This is a closure identity, not a claim that pass envelopes are additive or
recoverable speedup ceilings. Positive gaps and overlaps are reported separately.
Raw timestamp spans are checked independently against command-buffer GPU time;
scaled durations use the enclosing command only after that gate.

Two active resolver tests pin signed overlap accounting and explicit logical
empty stages. The final live chronological topology contains no empty physical
pass.

## Fixture And Protocol

- Base revision: `4315161136ab622f3b983d52bbaf4d7a7fca792c`
- Device: Apple M4 Max
- OS: macOS 15.6.1 (24G90)
- Rust/Cargo: 1.97.1
- Asset: `deepseek-v4-flash-0731-ud-iq3_xxs-current-2026-08-04`
- Asset size: 104,207,848,032 bytes across four pinned shards
- Model content ID:
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`
- Prefix: exact token pattern `[35, 201, 200, 34]` repeated to N=128
- Continuation: canonical snapshot restore, then exact token ID 35
- Routing: Rust CPU routing and the qualified 25-layer grouped IQ2 policy
- All-IQ3 widening: off
- MXFP4-down layers: unchanged fallback

The campaign runs one untimed ordinary/sampled warm-up, then timed
`A/B/A/B/A` from fresh sessions sharing one loaded residency. Every arm captures
the same prefix and restored continuation evidence.

Frozen evidence gates:

- every layer raw timestamp coverage within 2%; aggregate within 0.5%;
- each transition ambiguity at most 5% of its layer command;
- combined transition ambiguity at most 7.5% per layer and 2.5% aggregate;
- ordinary and sampled aggregate GPU drift each at most 5%;
- absolute sampled perturbation versus interpolated controls at most 10%;
- chronological share repeat delta at most two percentage points overall and
  three points in both CSA and HCA cohorts; and
- identical outputs, recorded causal identities, tokens, dispatch order, and
  complete launch geometry.

Representative command:

```text
QWEN_DSV4_PACKED_GROUPED_EXPERTS=auto \
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_pre_expert_stage_attribution_packet \
  -- --ignored --exact --nocapture
```

## Exactness And Topology

All seven warm/timed arms preserve bit-exact packed logits, final normalized
hidden output, and restored continuation logits. Causal, prefix, compatibility,
and continuation-causal digests plus committed tokens also match.

- Dispatches per arm: 47,990
- Dispatch family/kernel/grid/thread geometry SHA-256:
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

Encoder topology differs deliberately: 86 ordinary versus 172 sampled
encoders. The exact digest establishes unchanged dispatch work and complete
launch geometry, not identical encoder topology.

## Result

Aggregate pre-expert command GPU time, in milliseconds:

| Arm | Run 0 | Run 1 | Run 2 | Drift |
|---|---:|---:|---:|---:|
| Ordinary | 1,082.499 | 1,051.893 | 1,058.030 | 2.868% |
| Sampled | 1,058.170 | 1,074.016 | - | 1.486% |

Interpolated ordinary controls are 1,067.196 and 1,054.962 ms. Sampled topology
perturbation is -0.846% and +1.806%, inside the 10% gate. Audit-inclusive wall
times are retained in the raw log but are not topology or throughput evidence;
they include sample-buffer creation, timestamp resolution, dispatch census, and
host output validation.

Chronological target result:

| Metric | Sample 0 | Sample 1 |
|---|---:|---:|
| Raw sampled target | 73.989 ms | 73.224 ms |
| Sampled command share | 6.992% | 6.818% |
| Normalized ordinary estimate | 74.620 ms | 71.925 ms |

The normalized median is 73.273 ms. Aggregate transition ambiguity is only
0.0087% at worst; repeat share delta is 0.174 percentage points. The observer
uncertainty is therefore 1.806%, dominated by topology perturbation. The frozen
lower bound is 5.099% and 54.163 ms.

Attention-kind cohorts:

| Cohort | Target ms sample 0/1 | Share sample 0/1 |
|---|---:|---:|
| Sliding window (2) | 1.010 / 1.127 | 1.859% / 1.927% |
| CSA (21) | 55.832 / 55.371 | 10.266% / 10.192% |
| HCA (20) | 17.147 / 16.726 | 3.728% / 3.542% |

The full 43-layer per-command durations, stage ticks, scaled envelopes,
transitions, and cohort totals are retained in the accepted log. Maximum
per-layer raw-coverage error is below 0.0002%; maximum combined transition
ambiguity is below 0.020%.

## Decision

Kill chronological publication as the standalone next optimization target for
this current-asset N=128 workload. Its 54.163 ms conservative lower bound is far
below both the 15% and 150 ms authorization floors, and HCA contribution is only
about 3.6%. Do not build a dedicated row-loop fusion campaign now.

Retain the mechanism as a later piggyback opportunity: CSA rows consistently
spend about 10.2%, so a future attention/compressor rewrite that already touches
the chronological boundary may absorb it cheaply.

Next measure attention body plus output across all 43 layers with the same
three-pass instrument:

- begin after the chronological loop and before compressed-row lookup and the
  dense/sparse attention branch;
- include dense/sparse attention, inverse RoPE, and both attention output
  projections; and
- end immediately after `encode_output`, before attention mHC post.

Do not split attention body from output until their combined envelope clears an
authorization gate. If it misses, return to the remaining Q/KV, mHC/compressor,
and router families rather than speculating from launch counts.

## Rejected Broad Profiler

The original eight-stage all-layer topology is removed as a live lane. Three
bounded attempts are retained only as negative instrument evidence:

1. `integration-n128-audit.log`: an empty SWA compressor pass violates strict
   adjacent nonoverlap at layer 0; 8.14 seconds; SHA-256
   `81758d1d6061a1dab1d55c4c5b23795566309164165efbe12de7d7127af9560f`.
2. `integration-n128-audit-zero-stage.log`: after representing that logical
   stage as zero, non-empty layer-6 ingress/attention-mHC passes also overlap;
   6.37 seconds; SHA-256
   `843d86bf0877bd362c3bbc0fb92245ead8b18312c6c0336c97bf6a6bfc2a8338`.
3. `integration-n128-envelope.log`: signed envelopes reach the frozen layer-41
   transition-ambiguity rejection; 22.22 seconds; SHA-256
   `96857cf4ce4a6293696653508ddde3760c4af7f9e9f5fa25168c1e7211d336c0`.

No stage values or attribution conclusions are accepted from those runs. Their
historical source and executable bytes were not separately retained, so their
hashes bind only the raw logs. They justify the narrower supported instrument;
they are not promotion evidence.

## Evidence

Accepted log:

- `integration-n128-chronological.log`
- SHA-256:
  `358281740e4de2247cf45260f88cec26d52b97b5e5171c5df8d825559b56ce0f`
- Runtime: 21.75 seconds in-test, 21.86 seconds process wall
- Reported swaps: zero

Final accepted source SHA-256:

- `deepseek_v4_metal.rs`:
  `4c777ec2994035cae917e87f27820b5c92694dc0bb826d7797b46385015678a4`
- `prefill.rs`:
  `7301a962b0f00143dbf4572ae2678872fe8d6e941f0bb26b45395e10d329e3ab`
- `metal.rs`:
  `cf12614f4a7b79223d39cf522b20eff1fb373b2e4227c14e41d7ffbbffd55ec6`

Final release diagnostic test executable SHA-256:
`c956c1e457803c43e2b5f92509716ac9ddd8e8d5a7796b05e2cc34c1368e40dd`.

The logs live under `target/profiles/dsv4-packed-pre-expert-stage/`. Two active
resolver tests, the current-asset packet, diagnostics test compilation,
formatting, patch whitespace, and strict release workspace all-target/all-feature
Clippy pass. CX design, overlap adjudication, and final review session:
`019fcf7d-e9d4-7150-b496-e70a31958e80`.
