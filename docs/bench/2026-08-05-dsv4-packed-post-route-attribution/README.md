# DeepSeek V4 Packed Post-Route Attribution

Status: the diagnostics-only observer and its bounded current-asset attribution
are `GO`. Ordinary packed execution retains one post-route encoder and does not
query Metal command timestamps unless existing tracing or this test-only
observer requests them. No expert policy changes in this packet.

## Question

The all-IQ3 grouped experiment left roughly 688 ms of post-route GPU time in its
nominally unchanged 27-layer cohort. The packed Q8 output campaign then closed
the leading pre-expert target. This packet asks which physical post-route family
owns the current warm N=128 budget before another format or fusion is attempted.

The four named stages are:

1. routed experts, including gather, gate, up, clamped SwiGLU, down, and scatter;
2. the shared expert;
3. routed weighting plus routed/shared combination; and
4. post-FFN hyper-connection work plus the final head on layer 42.

The profile also records the routed gate/up/down dtypes, actual bucket count,
and whether the qualified grouped IQ2 path executed for every layer.

## Instrument

The M4 Max supports stage-boundary counter samples but not dispatch-boundary
samples. A direct in-encoder probe was attempted first and removed completely:
requesting a dispatch-boundary buffer returned an unsupported error, while
calling `sampleCountersInBuffer` with a stage-boundary buffer trapped in the
Metal driver. `metal-counters.log` freezes the supported surface.

The retained observer therefore uses four descriptor-backed serial compute
encoders inside each existing post-route command buffer. Start/end timestamps
are resolved per layer and scaled only against that layer's independent
`GPUEndTime-GPUStartTime`. Signed gaps and overlaps remain explicit and must
close each sampled span exactly. The ordinary arm uses the same wrapper but only
one ordinary encoder; it allocates no sample buffer and creates no boundary.

The observer is diagnostics-only under `test + dsv4-diagnostics`. Production
timestamp access remains conditional on the pre-existing layer trace. Command
error status is checked before timestamp access.

## Protocol

- Base revision: `21e824097cdc589d9e391bb60b409432d3a305fc`.
- Device: Apple M4 Max; macOS 15.6.1 (24G90).
- Toolchain: Rust/Cargo 1.97.1.
- Asset: `deepseek-v4-flash-0731-ud-iq3_xxs-current-2026-08-04`,
  104,207,848,032 bytes across the four shards pinned by
  `crates/qwen-llm/tests/fixtures/deepseek_v4_flash_0731_ud_iq3_xxs_current_2026_08_04_census_v1.json`.
- Model content ID:
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`.
- Prefix: 128 exact token IDs repeating `[35, 201, 200, 34]`.
- Continuation: restore the canonical in-memory causal snapshot, then inject
  exact token ID 35.
- Schedule: one untimed ordinary/sampled warm-up followed by timed
  `ordinary/sampled/ordinary/sampled/ordinary/sampled/ordinary`.

Every arm must preserve packed logits, normalized hidden output, snapshot
identity digests, restored continuation logits, continuation causal identity,
committed tokens, and the complete dispatch family/kernel/order/grid/thread
digest. Ordinary and sampled arms must execute 86 and 215 serial encoders,
respectively, while dispatch count remains 47,990.

At least two of three sampled arms must pass, and at most one may be rejected.
A sampled arm is valid only when:

- aggregate raw timestamp coverage is within 0.5% of command GPU time;
- every layer's raw coverage is within 2%;
- aggregate transition ambiguity is no more than 2.5%;
- every layer's transition ambiguity is no more than 10%; and
- GPU perturbation against its interpolated ordinary controls is no more than
  10% absolute.

Controls and accepted samples must each remain within 5% GPU drift. Accepted
whole-stage shares must repeat within two percentage points and cohort-stage
shares within three points. These are observer-validity gates, not promotion
thresholds for an expert candidate.

## Result

All three sampled arms pass. Ordinary post-route GPU totals are
1055.730/1057.236/1055.774/1056.495 ms, only 0.143% drift. Sampled totals are
1056.725/1055.336/1058.238 ms, 0.275% drift. Interpolated GPU perturbation is
+0.023%, -0.111%, and +0.199%.

Raw coverage rounds to 1.000000 at six-decimal precision on every aggregate.
Aggregate transition ambiguity is at most 0.0066%; the worst per-layer value is
0.0354%. All stage and cohort shares repeat far inside their gates.

| Post-route stage | Mean GPU ms | Mean share |
|---|---:|---:|
| routed experts | 1,015.042 | 96.052% |
| shared expert | 33.733 | 3.192% |
| expert combine | 1.998 | 0.189% |
| hyper post + final head | 5.923 | 0.561% |

The routed stage by cohort is:

| Cohort | Layers | Buckets | Routed GPU ms |
|---|---:|---:|---:|
| grouped IQ2_XS/IQ2_XS/IQ3_XXS | 25 | 1,542 | 597.107 |
| per-bucket IQ3_XXS/IQ3_XXS/IQ3_XXS | 16 | 1,036 | 355.062 |
| MXFP4-down outliers | 2 | 117 | 62.874 |

The prior 688 ms unchanged-cohort trace does not transfer as a warm budget.
Under this packet, the 16 all-IQ3 layers alone expose about 355 ms in the exact
stage changed by the already-bit-exact grouped candidate. A finer
gate/up/SwiGLU/down timestamp split would require reordering the 1,036
per-bucket chains or adding thousands of encoder boundaries, because direct
in-encoder samples are unavailable.

## Exactness And Scope

- Logit SHA-256:
  `45c414408cdc688eeeb4f40166013b2ac7261e5a2c2655cdf9f51e818941f2d5`.
- Normalized-hidden SHA-256:
  `8a9106b6c70c9069461ce1ef05d9dfb1349b6da7c4123b3e2e7b896c9d5278af`.
- Causal/prefix/compatibility digests:
  `3de61f2054ff7e540be5beb113c5c3884875b7018de800696aa58a0393535035`,
  `42428fa6efe61408f76b1ebc342f75cfa333af202b801c3031f34a7551cb870a`,
  and `e7fb64be2f9fd96c52c9d5aa9384e09652e899650558dd50c4ba998e834bf076`.
- Restored continuation logit SHA-256:
  `6c76cacc8242ec015421d7885efd1e9b6df5c1b075f2143cfee701181f3ade44`.
- Continuation causal digest:
  `7bc950b04b735a172ccb3c5b467488f548407b8b38f497b304d5d4bef7ae0dd8`.
- Dispatch geometry digest:
  `c87a6e9272f731016bd3f77a4d15f66c121ac557b60b0fbecddaa093801cc175`.

"Snapshot identity" here means matching causal, prefix, and compatibility
digests plus bit-exact restored continuation. The packet does not compare a
serialized snapshot byte stream. The dispatch digest covers family, kernel,
order, grid, and thread geometry, not resource bindings, constants, expert IDs,
or identical encoder topology.

The routed stage is not pure GEMM cost. Cohort comparisons are observational and
confound dtype, grouped/per-bucket topology, routes, bucket counts, and layer
mix. Layer 42's final output head is charged to its MXFP4 cohort. The packet is
limited to CPU routing, the fixed N=128 pattern, this current asset, and this M4
Max. Recorded wall samples are audit-inclusive and not stable enough for a wall
performance claim.

## Decision

Promote the observer, not an expert policy. Routed execution is the only
credible post-route target; do not spend a campaign splitting its interleaved
sub-operations first.

This stable stage-specific GPU observer is a materially changed measurement
condition for the held all-IQ3 grouped candidate. It authorizes exactly one new,
preregistered promotion packet using identical sampled encoder topology for both
policies. That packet must retain the existing balanced wall stationarity and
p95 rules, add affected-cohort routed GPU and total post-route GPU thresholds,
verify exact outputs/state, and prove 25/0 versus 25/16 grouped invocation
counts. It receives no same-condition retry if either GPU or wall stationarity
fails.

## Frozen Follow-Up Gate

This definition is identical to the authoritative gate in
`docs/PERF-ROADMAP.md`:

- A is the current grouped-IQ2 policy. B adds the existing all-IQ3 grouped
  candidate. Both execute the same four-stage sampled encoder topology.
- Run one untimed `ABBA BAAB` warm block, then time `(ABBA BAAB) x 2`. Each arm
  has eight timed samples, four in each half.
- For each arm, half, and endpoint, use the conventional even median: sort four
  values and average the middle two.
- The endpoints are complete packed wall, total post-route command GPU, and the
  affected 16-layer routed GPU subtotal.
- For each arm and endpoint, half-to-half stationarity is
  `2 * abs(median_1 - median_2) / (median_1 + median_2)` and must not exceed 5%.
- In each matched half, `1 - candidate_median / control_median` must reach 5%
  wall, 10% total post-route GPU, and 30% affected routed GPU.
- Overall p95 is nearest rank across each arm's eight samples. With eight
  samples it is the maximum; candidate p95 must not exceed control p95 for any
  endpoint.
- Every timed sample must pass the observer's timestamp coverage and ambiguity
  gates. None may be deleted or replaced.
- Exact packed/continuation outputs and state identities are adjudicated before
  timing. Control must record 25 IQ2 and zero IQ3 grouped invocations; candidate
  must record 25 and 16. Dispatch geometry must repeat within each arm, but the
  deliberately different policy topologies need not match across arms.
- Any failed gate ends the single authorized attempt. There is no
  same-condition retry.

## Evidence

Final command:

```bash
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_post_route_stage_attribution_packet \
  -- --ignored --exact --nocapture
```

- `integration.log` SHA-256:
  `7a8f37ea173b2dd6e1b87527ba59eb772ac29d18079c28aa1033194f35fc9dc7`.
- `integration-rejected-r2.log` SHA-256:
  `89ec523b738b4d67bc77e013ac54dcc5e7529322ee8e6a2c591508c22f02ee82`.
  This retained two-sample protocol rejected one unstable sample and therefore
  had fewer than two valid samples; it contains no accepted attribution.
- `metal-counters.log` SHA-256:
  `5a0f8f0bf4658248a156d151bf4eba72d362e2ca4a0eb6b0e15e1074816621e7`.
- Release test executable SHA-256:
  `7627774cfc7bf124bf719018dfb16208419b92f3f2340b12c22535f440caefa7`.
- `deepseek_v4_metal.rs` SHA-256:
  `44ae208184d9ac258fa33a9a18f1211f8845160f5183493ba898a694992e8e70`.
- `deepseek_v4_metal/prefill.rs` SHA-256:
  `b4f4eaeba9ed336fac668e80a943dc4e72c68727e0d9224b47b8bb52a44cc4c4`.
- CX review: `019fd4b0-a4d6-76f2-aee4-2172a72d8548`.

The final reviewer found one blocker before documentation: ordinary execution
queried command timestamps unconditionally and tensor lookups had moved outside
the existing trace envelope. Both are repaired in the source hashes above.
Non-diagnostic and diagnostic release checks pass, as does the model-free signed
overlap resolver test.
