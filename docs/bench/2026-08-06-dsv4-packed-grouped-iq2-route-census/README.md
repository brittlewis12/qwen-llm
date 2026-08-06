# DeepSeek V4 Packed Grouped-IQ2 Route Census

Status: exact current-asset route census `GO`. The retained fixture closes the
remaining source-row and destination-slot ambiguity in the grouped-IQ2 phase
profiler. It is provenance and schedule evidence only: it contains no phase
timing and authorizes no replacement kernel or product claim.

## Question

The preceding count census fixed all 25 per-expert populations and width-32
tile counts, but many legal slot-major route sequences produce those same
histograms. A phase-only result over synthetic slots could therefore close only
the tested control, not the current production schedule.

One additional current-asset execution is cheaper and more decisive than
carrying that ambiguity through a roughly 50 GiB model-free campaign. Capture
the exact six expert IDs for every token and grouped layer, prove that they
rebuild the production expert-major schedule, then remove the one-shot harness.

## Capture Contract

The sole release-diagnostics execution uses:

- base revision `c18ece549887ff5477af01a93f1fc548631c6f96`;
- the pinned 97.05 GiB current asset and model content ID
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`;
- the exact accepted token pattern `[35,201,200,34]` repeated to 128 tokens,
  with little-endian token SHA-256
  `b57816bcb0d5fdf5a8e2ddc7a0afe9e57fb0ca6ffc2b849285e1635d04772843`;
- ordinary CPU routing and the qualified grouped-IQ2 policy; and
- one unsampled packed execution, no continuation, candidate, or timed
  endpoint.

The harness binds the accepted logits, normalized hidden state, causal state,
prefix metadata, compatibility metadata, committed prefix, exact grouped layer
IDs, and count payload before emitting routes. Every layer must contain exactly
`128 * 6 = 768` in-range IDs, and every six-ID token group must be distinct.

The diagnostics helper independently reconstructs the expert-major rows,
slot-major destination indices, ascending buckets, and per-expert counts. Those
must equal the production schedule and the prior count fixture exactly.

## Identity

The canonical binary payload is:

```text
"qwen-llm:dsv4:packed-grouped-iq2-route-ids:v1\0"
|| model_content_id[32]
|| prompt_token_ids_sha256[32]
|| count_payload_sha256[32]
|| n_tokens_u32_le
|| top_k_u32_le
|| expert_count_u32_le
|| layer_count_u32_le
|| route_count_u32_le
|| for each grouped layer in ascending execution order:
     layer_u32_le
     || 768 * expert_id_u16_le
```

The 25x768 route payload hashes to
`505cb93ff9c3e1557bbad8f27a773e08b0e8c3475fbb4d7096069667c1fbafdd`.
Its bound count payload is
`0ab9925350288116288794f3d7f5081595dfad4ffdb9671a9a146f27568358a6`.
The extracted JSON and committed test fixture are byte-identical.

## Result

The sole capture passes in 6.76 seconds in-test and 6.85 seconds elapsed with
zero swaps. It records 19,200 exact slot-major route IDs. Every token has six
distinct experts, every layer reconstructs its production expert-major bucket
order exactly, and every reconstructed histogram equals the committed count
fixture.

Elapsed time, resident memory, and zero swaps are capture provenance only. They
are not routed-phase or prefill measurements.

## Decision

Retain the diagnostics-only route reconstruction helper, metadata, and active
canonical fixture test. Remove the ignored one-shot asset harness after
archiving its reconstructing source diff. Do not rerun the asset capture.

Use only these exact production slots in the primary grouped-IQ2 phase
campaign. For each gate/up/SwiGLU or down/scatter phase, compare 25 disjoint
banks with one all-expert warm bank reused across layers. Use production-shaped
IQ2_XS gate and up banks, IQ3_XXS down banks, and distinct production-sized F32
input, inner, output, and slot buffers for every layer. Full logical allocation
is required; first-touch every selected disjoint expert slice and the complete
warm bank with legal finite nonzero quant bytes. Every sample aggregates 25
serial per-layer command buffers, preserving production command boundaries.
Run gate and down campaigns sequentially so their large banks are never
resident together. Immediately before every timed down command, complete an
untimed gate command for the same layer to produce its real inner input.

After complete first touch and one untimed conditioning block, retain 12
samples per regime in balanced `D/W/W/D` order repeated six times. Bracket each
campaign with matched 25-command empty measurements. Delete or replace no
sample. Both regimes must satisfy the frozen 5% drift rule before computing:

```text
drift = 2 * (max(samples) - min(samples))
      / (max(samples) + min(samples))
```

Use nearest-rank p95; with 12 retained samples, p95 is the cell maximum. Then
compute:

```text
U_phase = max(disjoint_p95, warm_p95)
        + max(5 * max(disjoint_range, warm_range),
              empty_pre_gpu_ms,
              empty_post_gpu_ms)
```

Any allocation, Metal, validation, timestamp, or stability failure is
`INCONCLUSIVE`. Stable `U_phase < 158.3 ms` decisively closes only that exact
phase-only lane. Stable `U_phase >= 158.3 ms` authorizes candidate design only.
Neither outcome transfers to another phase, a joint rewrite, or product timing.

The sole subsequent campaign is `INCONCLUSIVE - timestamp-invalid`: the second
empty gate pre-bracket command has equal finite GPU start/end timestamps. No
retained gate cell or down campaign executes, and no phase timing claim
survives. The profiler is removed. Reopen only under a separately reviewed
conservative treatment of timestamp-resolution-censored empty commands or
relevant device/toolchain/observer drift; never treat equality as zero overhead.

V2 consumes that instrumentation reopen. Its gate/up/SwiGLU cells pass the 5%
stationarity gate with `C=U=537.050704 ms`, authorizing candidate design only
for that exact phase. Down warm drift reaches 5.483643% and remains
`HOLD - INCONCLUSIVE`; it cannot be bundled into the gate authorization.

## Validation

- The canonical route-fixture release test passes.
- The grouped schedule reconstruction and guard release test passes.
- Strict release `qwen-llm` diagnostics Clippy passes with warnings denied.
- `cargo fmt --all -- --check` and `git diff --check` pass.
- The evidence JSON and committed fixture are byte-identical.
- Every archived artifact hash and the final retained binary hash recheck.

## Provenance

- Device: Apple M4 Max.
- Asset-run source diff SHA-256:
  `a8b32c041f67a664a08f653bccd37ac5f7ae1e8e02cebae28fa897ff0e42275a`.
- Retained source diff SHA-256:
  `a38f7f78e218113174244e41642d35505c69c03e52b8b29835ebc6254e42d45e`.
- Raw log SHA-256:
  `d63715f4353b7a8e3c4a7406a5c774818aee60d938a20430ecc2e8b82610bb56`.
- Extracted JSON and committed fixture SHA-256:
  `8c907f02b026d05bacbaf861c114795d0a7b86086f7eb20566fc12c04370d67f`.
- Asset-run release test binary SHA-256, recorded before execution:
  `866619818f5c2672c3e96bd37fd460b6921d480a8ab068cd1018e6778cfdf577`.
- Final retained release test binary SHA-256:
  `ac084b0628d1044d100d971898dc232b1dc7af649e6661298a58ecdbe075f0de`.
- Checksum transcript: `checksums.log`.
- CX review: `019fd588-96b0-7033-afd1-66d7e354e523`.

Asset command:

```bash
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_grouped_iq2_route_id_census \
  -- --ignored --exact --nocapture --test-threads=1
```
